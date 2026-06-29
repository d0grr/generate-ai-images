// ─────────────────────────────────────────────────────────────
// Browser-engine shared helpers (ONNX Runtime Web on WebGPU).
//
// Common hub imported by the pipeline (sdxl-pipeline.js) and the
// post-processors (upscale-pipeline.js, face-restore-pipeline.js):
//
//   • loadOrt()             — bootstrap onnxruntime-web (globalThis.ort)
//   • loadTransformers()    — @huggingface/transformers (tokenizer only)
//   • fetchOnnxCached()     — stream ONNX weights into OPFS with Range resume,
//                             auto-retry on any non-abort error
//   • fetchJsonCachedOPFS() — small-JSON (tokenizer) fetch with OPFS cache
//   • preloadModelFiles()   — pre-cache a repo's ONNX files (shard-aware)
//   • f16ToF32()            — widen fp16 tensor data to fp32
//
// ONNX weights + tokenizer JSON stream from HF the first time, then live in
// OPFS — after the first run the engine is fully offline.
// ─────────────────────────────────────────────────────────────

const HF_BASE = "https://huggingface.co";

let _ort = null;
let _tokenizers = null;   // cached transformers module

export function loadOrt() {
  if (_ort) return _ort;
  if (!globalThis.ort) {
    throw new Error(
      "onnxruntime-web not loaded — check offscreen.html <script src=\"./vendor/ort.min.js\">",
    );
  }
  _ort = globalThis.ort;
  // MV3 blocks remote dynamic imports (ORT 1.20 uses dynamic ESM imports
  // for its WASM glue), so all ort-wasm-* files MUST be bundled locally.
  // See offscreen/vendor/ort-wasm-simd-threaded.{wasm,mjs} and the .jsep.*
  // pair for the WebGPU backend.
  _ort.env.wasm.wasmPaths = chrome.runtime.getURL("offscreen/vendor/");
  // "error" silences cosmetic warnings (e.g. shape ops staying on CPU
  // when both WebGPU and WASM EPs are listed) — real failures still surface.
  _ort.env.logLevel = "error";
  // No threading / proxy — both rely on `blob:` worker URLs which MV3 CSP
  // rejects. WebGPU runs on the GPU anyway, so single-thread CPU is only a
  // fallback price.
  _ort.env.wasm.numThreads = 1;
  _ort.env.wasm.proxy = false;
  return _ort;
}

/**
 * fetch a small JSON file (tokenizer.json / tokenizer_config.json) with a
 * persistent OPFS cache so a downloaded model never re-contacts HF.
 *   1st run  — fetch from HF, then write the JSON into the OPFS models-cache.
 *   later    — read straight from OPFS, zero network (works fully offline).
 */
export async function fetchJsonCachedOPFS(url) {
  const fileName = cacheKeyFor(url);

  const cached = await opfsReadJson(fileName);
  if (cached !== null) {
    console.info(`[sd] tokenizer cache hit ${fileName}`);
    return cached;
  }

  console.info("[sd] tokenizer fetch (will cache to OPFS):", url);
  const json = await smartFetchJson(url);
  await opfsWriteJson(fileName, json);
  return json;
}

/**
 * fetch JSON, trying the direct path first.  If it throws "Failed to fetch"
 * (likely a host_permissions edge case in the offscreen document), retry
 * via the service-worker proxy where extension privileges always apply.
 */
async function smartFetchJson(url) {
  try {
    const r = await fetch(url, { credentials: "omit" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch {
    // Direct fetch from chrome-extension:// origin sometimes trips CORS in
    // ways that the SW-side stream does not — silently fall through.
    const buf = await streamFetchViaSW(url, null, null);
    const text = new TextDecoder().decode(new Uint8Array(buf));
    return JSON.parse(text);
  }
}

export async function loadTransformers() {
  if (_tokenizers) return _tokenizers;
  // transformers.min.js is a webpack ESM bundle — dynamic import exposes
  // AutoTokenizer and friends as named exports. A literal (relative) specifier
  // keeps the lazy load but avoids AMO's "unsafe dynamic import" warning that a
  // computed URL triggers; it resolves to offscreen/vendor/ relative to here.
  const T = await import("./vendor/transformers.min.js");
  const module = T.default || T;
  if (!module.AutoTokenizer) {
    throw new Error(
      "transformers.js loaded but AutoTokenizer is missing — bundle may be the wrong format",
    );
  }
  if (module.env) {
    module.env.allowRemoteModels = true;
    // Cache API has quirks in extension origin — keep tokenizer fetches
    // direct so failures surface as plain HTTP errors we can debug.
    module.env.useBrowserCache = false;
    module.env.allowLocalModels = false;
  }
  _tokenizers = module;
  return module;
}

// ─────────────────────────────────────────────────────────────
// float16 helpers — ORT 1.20 wants `Float16Array` instances (not raw
// Uint16Array bit blobs) for fp16 tensors.  Float16Array landed in
// Chrome 122 / Edge 122 / Safari 18; we surface a clear error if the
// user's browser is older.
// ─────────────────────────────────────────────────────────────
const HAS_F16 = typeof Float16Array !== "undefined";

export function f16ToF32(arr) {
  // ORT may hand us a Float16Array (modern Chrome) or a Uint16Array
  // (older builds) — both can be widened by iterating into Float32Array.
  if (arr instanceof Float32Array) return arr;
  if (HAS_F16 && arr instanceof Float16Array) {
    return new Float32Array(arr);
  }
  // Uint16Array of raw fp16 bits — manual conversion.
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = f16BitsToF32(arr[i]);
  return out;
}

function f16BitsToF32(h) {
  const sign = (h >> 15) & 0x1;
  const exp  = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0)  return (sign ? -1 : 1) * (frac / 1024) * Math.pow(2, -14);
  if (exp === 31) return frac ? NaN : (sign ? -Infinity : Infinity);
  return (sign ? -1 : 1) * (1 + frac / 1024) * Math.pow(2, exp - 15);
}

// ─────────────────────────────────────────────────────────────
// OPFS-backed cache for ONNX weights with HTTP Range resume.
// Per-file: completed downloads stay forever, partial downloads survive
// cancel and are picked up on the next Generate via Range request.
// ─────────────────────────────────────────────────────────────
const ONNX_CACHE_DIR = "models-cache";
const SIZE_KEY_PREFIX = "onnx-size:";

// Parallel segmented download. HF's Xet CDN throttles then drops a single long
// connection over a multi-GB shard, so instead of one open-ended `bytes=start-`
// stream we slice each file into short, bounded byte ranges and fetch a few at
// once. Short ranges complete before the CDN throttles; parallelism lifts the
// aggregate rate; a dropped block costs only one cheap re-fetch, not the file.
const DL_BLOCK_BYTES = 8 * 1024 * 1024;   // size of each ranged block
const DL_CONCURRENCY = 4;                 // blocks fetched at once per file
const DL_SEG_MAX_ATTEMPTS = 5;            // per-block retries before the file fails
const SEG_KEY_PREFIX = "onnx-seg:";       // storage.local resume manifest per file

// One OPFS SyncAccessHandle per file is allowed at a time. The offscreen
// document is kept alive across generations while the popup is open (so the
// loaded pipeline is reused), which means a worker from a cancelled/failed
// download can still be holding a file's handle when the next attempt starts.
// Track the live writer per file so we can tear down a stale one before
// opening a fresh handle, instead of hitting "Access Handles cannot be
// created if there is another open Access Handle".
const _activeWriters = new Map();   // fileName → Worker

// `chrome.storage` isn't always reachable from offscreen documents on every
// Chrome build — fall back to a service-worker proxy when it's missing.
async function swStorageGet(keys) {
  if (chrome?.storage?.local?.get) {
    try { return await chrome.storage.local.get(keys); } catch {}
  }
  const resp = await chrome.runtime.sendMessage({ type: "action:storage-get", keys });
  return resp?.result || {};
}

// Firefox runs the engine inside the background page (no separate offscreen
// document), flagged by globalThis.__ENGINE_DIRECT__. There, runtime.sendMessage
// loops to the same context, so the SW proxies (url-info, the fetch: stream port)
// are unusable — but the background page can fetch directly. DIRECT switches those
// two paths to in-page fetch. Chrome (offscreen) keeps the SW-proxied behaviour.
const DIRECT = !!globalThis.__ENGINE_DIRECT__;

async function urlInfo(url) {
  if (DIRECT) {
    // HF now serves LFS weights via a 302 redirect to a *presigned* Xet CDN URL
    // whose signature is scoped to GET. A HEAD against it fails, so the old HEAD
    // probe wrongly reported sidecars (e.g. model_q4.onnx_data) as missing — ORT
    // then loaded the graph without its external weights and died with
    // "Failed to load external data file … Module.MountedFiles is not available".
    // Probe with a 1-byte ranged GET instead: it's identical to the real streamed
    // download (which works), and we abort before reading the body so nothing
    // large transfers even if the server ignores Range and answers 200.
    const ctrl = new AbortController();
    try {
      const resp = await fetch(url, {
        method: "GET",
        headers: { Range: "bytes=0-0" },
        credentials: "omit",
        signal: ctrl.signal,
      });
      // Prefer the total from "Content-Range: bytes 0-0/<total>" (a 206); fall
      // back to Content-Length (a 200 where Range was ignored).
      let size = 0;
      const m = /\/(\d+)\s*$/.exec(resp.headers.get("content-range") || "");
      if (m) size = Number(m[1]);
      if (!size) size = Number(resp.headers.get("content-length")) || 0;
      return { ok: resp.ok || resp.status === 206, status: resp.status, size };
    } catch {
      return { ok: false };
    } finally {
      try { ctrl.abort(); } catch {}   // drop the connection without reading the body
    }
  }
  try {
    return await chrome.runtime.sendMessage({ type: "action:url-info", url });
  } catch {
    return { ok: false };
  }
}

async function urlExists(url) {
  const info = await urlInfo(url);
  return !!info?.ok;
}

/**
 * Discover a component's external-data sidecar files for `${base}` (a
 * `.../resolve/main/<subdir>` URL). Returns the filenames to feed ORT's
 * `externalData`, handling all export conventions:
 *   • [] — weights embedded in model.onnx (small models)
 *   • ["weights.pb"] — legacy single sidecar
 *   • ["model.onnx_data", "model.onnx_data_1", …] — optimum, possibly sharded
 * (15-byte Git-LFS pointer files are skipped via the >1000-byte check.)
 */
export async function discoverShards(base, stem = "model") {
  const pb = await urlInfo(`${base}/weights.pb`);
  if (pb?.ok && pb.size > 1000) return ["weights.pb"];
  const out = [];
  for (let i = 0; ; i++) {
    const name = i === 0 ? `${stem}.onnx_data` : `${stem}.onnx_data_${i}`;
    const info = await urlInfo(`${base}/${name}`);
    if (!info?.ok || info.size <= 1000) break;
    out.push(name);
  }
  return out;
}

async function urlSize(url) {
  const info = await urlInfo(url);
  return info?.size || 0;
}

async function swStorageSet(patch) {
  if (chrome?.storage?.local?.set) {
    try { return await chrome.storage.local.set(patch); } catch {}
  }
  await chrome.runtime.sendMessage({ type: "action:storage-set", patch });
}

function cacheKeyFor(url) {
  // Strip protocol + collapse non-filename chars so it's a flat OPFS key.
  return url.replace(/^https?:\/\//, "").replace(/[^a-zA-Z0-9._-]/g, "_");
}

async function getCacheDir() {
  const root = await navigator.storage.getDirectory();
  return await root.getDirectoryHandle(ONNX_CACHE_DIR, { create: true });
}

// Small-JSON helpers (tokenizer files). These stay well under the multi-GB
// thresholds that force the SyncAccessHandle path for ONNX weights, so plain
// createWritable is fine and avoids spinning up the opfs-writer worker.
async function opfsReadJson(fileName) {
  try {
    const dir = await getCacheDir();
    const fh = await dir.getFileHandle(fileName);
    const file = await fh.getFile();
    if (!file.size) return null;
    return JSON.parse(await file.text());
  } catch {
    return null;   // not cached yet (or unreadable) — caller will fetch
  }
}

async function opfsWriteJson(fileName, obj) {
  try {
    const dir = await getCacheDir();
    const fh = await dir.getFileHandle(fileName, { create: true });
    const writable = await fh.createWritable();
    await writable.write(JSON.stringify(obj));
    await writable.close();
  } catch (err) {
    // Non-fatal: the tokenizer still loads this run, it just won't be cached
    // for offline use next time.
    console.warn(`[sd] could not cache ${fileName} to OPFS:`, err?.message || err);
  }
}

async function opfsFileSize(fileName) {
  try {
    const dir = await getCacheDir();
    const fh = await dir.getFileHandle(fileName);
    const file = await fh.getFile();
    return file.size;
  } catch {
    return 0;
  }
}

async function opfsReadFull(fileName) {
  const dir = await getCacheDir();
  const fh = await dir.getFileHandle(fileName);

  // Fast path — try the whole-file arrayBuffer first; works for typical
  // sizes (≤ ~2 GB on most Chrome builds). The slow path below handles the
  // multi-GB fp32 weight blobs that trip Chrome's catch-all NotReadableError
  // "permission problems that have occurred after a reference to a file
  // was acquired" message.
  try {
    const file = await fh.getFile();
    return await file.arrayBuffer();
  } catch (err) {
    const msg = String(err?.message || err);
    const oversize = msg.includes("could not be read") ||
                     msg.includes("permission problems") ||
                     err?.name === "NotReadableError" ||
                     err?.name === "InvalidStateError" ||
                     err?.name === "RangeError";
    if (!oversize) throw err;
    console.warn(`[sd] opfsReadFull: whole-file read failed (${msg}); falling back to chunked read for ${fileName}`);
  }

  // Slow path — pre-allocate one big Uint8Array, then fill it by reading
  // 64 MB slices. Each slice's own arrayBuffer() is small enough to not
  // trip the size ceiling, and the big destination buffer is a single
  // allocation that V8 can usually satisfy even when File.arrayBuffer()
  // refuses the equivalent transfer.
  const file = await fh.getFile();
  const size = file.size;
  let out;
  try {
    out = new Uint8Array(size);
  } catch (err) {
    // V8 refused the big allocation. This is the hard ceiling: nothing we
    // can do at the pipeline level — the user's Chrome cannot materialise
    // a buffer this large. Surface a clear, actionable message instead of
    // the bare RangeError.
    const gb = (size / 1e9).toFixed(1);
    throw new Error(
      `Cannot allocate ${gb} GB ArrayBuffer — this fp32 model is too large for your Chrome build.\n` +
      `Use an fp16 or 4-bit model instead, or update Chrome to 126+.`
    );
  }
  const CHUNK = 64 * 1024 * 1024;
  let written = 0;
  for (let offset = 0; offset < size; offset += CHUNK) {
    const end = Math.min(offset + CHUNK, size);
    const slice = file.slice(offset, end);
    let sliceBuf;
    try {
      sliceBuf = await slice.arrayBuffer();
    } catch (err) {
      throw new Error(`chunked OPFS read failed at ${offset}/${size}: ${err?.message || err}`);
    }
    out.set(new Uint8Array(sliceBuf), offset);
    written += sliceBuf.byteLength;
  }
  if (written !== size) {
    throw new Error(`chunked OPFS read short: ${written}/${size} bytes`);
  }
  console.info(`[sd] opfsReadFull: chunked read OK for ${fileName} (${(size / 1e9).toFixed(2)} GB)`);
  return out.buffer;
}

export async function fetchOnnxCached(url, onProgress, signal) {
  try {
    return await fetchOnnxCachedImpl(url, onProgress, signal);
  } catch (err) {
    if (err?.message === "aborted") throw err;
    // Decorate the error with the URL so the toast actually says what failed.
    const msg = err?.message || String(err);
    if (msg.includes(url)) throw err;     // already annotated
    throw new Error(`${msg}\n  → while fetching ${url}`);
  }
}

/** True for a model server running on the local machine (dev model testing). */
function isLocalUrl(url) {
  return /^https?:\/\/(localhost|127\.0\.0\.1)(:|\/)/.test(url);
}

/**
 * Resolve a component root: an HF repo id ("org/repo") → its resolve/main URL,
 * or a full http(s):// value used verbatim (local/custom model server). Shared
 * source of truth for both the pipeline loader and the OPFS preloader.
 */
export function repoRoot(repoId) {
  return /^https?:\/\//.test(repoId)
    ? repoId.replace(/\/+$/, "")
    : `${HF_BASE}/${repoId}/resolve/main`;
}

/**
 * Stream a URL straight into an ArrayBuffer (no OPFS cache). Used for local
 * dev model servers: caching a localhost URL by its key would serve stale
 * weights after you rebuild the model behind the same URL, so local fetches
 * are always fresh. Reports progress from Content-Length when available.
 */
async function fetchDirectToBuffer(url, onProgress, signal) {
  const res = await fetch(url, { cache: "no-store", signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) {                       // no streaming → one-shot
    const buf = await res.arrayBuffer();
    onProgress?.(buf.byteLength, buf.byteLength || total);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total || loaded);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out.buffer;
}

/** True for a user-initiated abort vs a network/server error. */
function isAbortErr(err, signal) {
  const msg = String(err?.message || err);
  return !!signal?.aborted || err?.name === "AbortError" || /aborted/i.test(msg);
}

/** setTimeout as an abortable promise (rejects "aborted" if the signal fires). */
function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const onAbort = () => { clearTimeout(t); cleanup(); reject(new Error("aborted")); };
    const cleanup = () => signal?.removeEventListener?.("abort", onAbort);
    const t = setTimeout(() => { cleanup(); resolve(); }, ms);
    if (signal) { if (signal.aborted) onAbort(); else signal.addEventListener("abort", onAbort, { once: true }); }
  });
}

/**
 * Probe a URL with a 1-byte ranged GET to learn its size AND whether the server
 * honours Range (a 206). Mirrors urlInfo's DIRECT trick (HF's presigned Xet URLs
 * reject HEAD), but also reports range support so the caller can pick the
 * parallel segmented path vs a single sequential stream. Aborts the connection
 * in `finally` so the body never transfers.
 */
async function probeRangeDirect(url, signal) {
  const ctrl = new AbortController();
  const onAbort = () => { try { ctrl.abort(); } catch {} };
  if (signal) { if (signal.aborted) onAbort(); else signal.addEventListener("abort", onAbort, { once: true }); }
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: { Range: "bytes=0-0" },
      credentials: "omit",
      signal: ctrl.signal,
    });
    if (res.status === 206) {
      const m = /\/(\d+)\s*$/.exec(res.headers.get("content-range") || "");
      return { ranges: true, size: m ? Number(m[1]) : 0 };
    }
    if (res.ok) return { ranges: false, size: Number(res.headers.get("content-length")) || 0 };
    throw new Error(`HTTP ${res.status} for ${url}`);
  } finally {
    onAbort();
    signal?.removeEventListener?.("abort", onAbort);
  }
}

/**
 * Fetch one inclusive byte range [start,end] straight from HF in the offscreen
 * document and hand each piece to onChunk(chunk, absoluteOffset). Requires a 206
 * — a 200 means the server ignored Range and would deliver the whole file, which
 * is useless mid-parallel-download, so we reject and let the caller fall back.
 */
async function fetchRangeDirect(url, start, end, onChunk, signal) {
  const res = await fetch(url, {
    method: "GET",
    headers: { Range: `bytes=${start}-${end}` },
    credentials: "omit",
    signal,
  });
  if (res.status !== 206) {
    if (res.ok) throw new Error(`range-ignored (HTTP 200) for ${url}`);
    throw new Error(`HTTP ${res.status} for ${url}`);
  }
  const reader = res.body.getReader();
  let at = start;
  for (;;) {
    if (signal?.aborted) { try { await reader.cancel("aborted"); } catch {} throw new Error("aborted"); }
    const { done, value } = await reader.read();
    if (done) break;
    await onChunk(value, at);
    at += value.byteLength;
  }
}

/**
 * Open the OPFS cache file for a model and return a writer that accepts
 * concurrent, out-of-order offset writes (one SyncAccessHandle worker per file —
 * holding two on one file is illegal, so the parallel segments funnel through
 * this single worker, which serialises them). Each write is id-matched to its
 * ack so producers get real back-pressure instead of flooding the worker.
 */
async function openOpfsWriter(fileName, truncate, size) {
  const stale = _activeWriters.get(fileName);
  if (stale) { try { stale.terminate(); } catch {} _activeWriters.delete(fileName); }
  const worker = new Worker(chrome.runtime.getURL("offscreen/workers/opfs-writer.js"));
  _activeWriters.set(fileName, worker);

  let seq = 0;
  const pending = new Map();           // id → {resolve, reject}
  let openResolve, openReject;
  const opened = new Promise((res, rej) => { openResolve = res; openReject = rej; });

  worker.addEventListener("message", (e) => {
    const d = e.data || {};
    if (d.type === "opened") { openResolve(d.size || 0); return; }
    if (d.type === "error") {
      if (d.id != null && pending.has(d.id)) { pending.get(d.id).reject(new Error(d.error)); pending.delete(d.id); }
      else { openReject(new Error(d.error)); pending.forEach((p) => p.reject(new Error(d.error))); pending.clear(); }
      return;
    }
    if (d.id != null && pending.has(d.id)) { pending.get(d.id).resolve(d); pending.delete(d.id); }
  });

  const call = (msg, transfer) => new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...msg, id }, transfer || []);
  });

  worker.postMessage({ type: "open", dirName: ONNX_CACHE_DIR, fileName, truncate, size });
  await opened;

  return {
    writeAt: (chunk, at) => call({ type: "write", chunk, at }, [chunk.buffer]),
    flush: () => call({ type: "flush" }),
    async close() {
      try { await call({ type: "close" }); } catch {}
      try { worker.terminate(); } catch {}
      if (_activeWriters.get(fileName) === worker) _activeWriters.delete(fileName);
    },
  };
}

/**
 * Download `url` into the OPFS cache file as fixed-size blocks fetched
 * DL_CONCURRENCY at a time, writing each at its absolute offset. A resume
 * manifest (which blocks are done) is persisted so a reload re-fetches only the
 * missing ones. `total` is the file size from a successful range probe.
 */
async function segmentedDownload(url, fileName, total, onProgress, signal) {
  const block = DL_BLOCK_BYTES;
  const segCount = Math.ceil(total / block);
  const segKey = `${SEG_KEY_PREFIX}${fileName}`;
  const segSize = (i) => Math.min((i + 1) * block, total) - i * block;

  // Resume only if the manifest matches this exact geometry AND the on-disk file
  // already spans `total` (so untouched blocks read back as real, not a hole).
  let done = new Array(segCount).fill(false);
  const man = (await swStorageGet(segKey))?.[segKey];
  const canResume = man && man.total === total && man.block === block &&
    Array.isArray(man.done) && man.done.length === segCount &&
    (await opfsFileSize(fileName)) >= total;
  if (canResume) done = man.done.map(Boolean);

  const writer = await openOpfsWriter(fileName, /* truncate */ !canResume, total);

  const segLoaded = new Array(segCount).fill(0);
  for (let i = 0; i < segCount; i++) if (done[i]) segLoaded[i] = segSize(i);
  const report = () => onProgress?.(segLoaded.reduce((a, b) => a + b, 0), total);
  report();

  let sinceSave = 0;
  const saveManifest = async (force) => {
    if (!force && ++sinceSave < DL_CONCURRENCY) return;
    sinceSave = 0;
    await swStorageSet({ [segKey]: { total, block, done } }).catch(() => {});
  };

  async function runSegment(i) {
    const start = i * block;
    const end = start + segSize(i) - 1;
    let lastErr;
    for (let attempt = 1; attempt <= DL_SEG_MAX_ATTEMPTS; attempt++) {
      if (signal?.aborted) throw new Error("aborted");
      segLoaded[i] = 0; report();          // a retry re-fetches the block from its start
      try {
        await fetchRangeDirect(url, start, end, async (chunk, at) => {
          const n = chunk.byteLength;
          await writer.writeAt(chunk, at);
          segLoaded[i] += n; report();
        }, signal);
        return;
      } catch (err) {
        lastErr = err;
        if (isAbortErr(err, signal) || attempt === DL_SEG_MAX_ATTEMPTS) throw err;
        const delayMs = Math.min(1000 * 2 ** (attempt - 1), 15000);
        console.warn(`[sd] block ${i}/${segCount} of ${fileName} failed (attempt ${attempt}): ` +
          `${err?.message || err} — retry in ${delayMs} ms`);
        await sleep(delayMs, signal);
      }
    }
    throw lastErr;
  }

  const queue = [];
  for (let i = 0; i < segCount; i++) if (!done[i]) queue.push(i);
  let qi = 0, fatal = null;
  async function pump() {
    for (;;) {
      const i = queue[qi++];
      if (i === undefined || fatal) break;
      try {
        await runSegment(i);
        done[i] = true;
        await saveManifest(false);
      } catch (err) { fatal = err; break; }
    }
  }

  await Promise.all(Array.from({ length: Math.min(DL_CONCURRENCY, queue.length || 1) }, pump));
  if (fatal) {
    await saveManifest(true);            // keep progress so the next call resumes
    await writer.close();
    throw fatal;
  }
  await writer.flush();
  await writer.close();
  await swStorageSet({ [segKey]: null }).catch(() => {});   // complete — drop the manifest
}

/**
 * Legacy single-stream download: one open-ended `bytes=start-` request via the
 * SW proxy (or in-page on Firefox), resuming from the on-disk offset with
 * exponential backoff. The fallback when range probing fails or the server
 * won't honour Range — kept for robustness, no longer the default path.
 */
async function legacyStream(url, fileName, onProgress, signal) {
  const MAX_ATTEMPTS = 6;          // 1 initial + 5 retries
  const MAX_BACKOFF_MS = 30000;
  let lastErr = null;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    if (signal?.aborted) throw new Error("aborted");
    const startBytes = await opfsFileSize(fileName);
    try {
      await streamOnnxToOpfs(url, fileName, startBytes, onProgress, signal);
      return;
    } catch (err) {
      lastErr = err;
      const msg = String(err?.message || err);
      if (isAbortErr(err, signal) || attempt === MAX_ATTEMPTS) throw err;
      const delayMs = Math.min(2000 * 2 ** (attempt - 1), MAX_BACKOFF_MS); // 2,4,8,16,30 s
      const resumeFrom = await opfsFileSize(fileName);
      console.warn(`[sd] download error for ${fileName} (attempt ${attempt}/${MAX_ATTEMPTS}): ${msg} — ` +
        `resuming from ${(resumeFrom / 1e6).toFixed(1)} MB in ${delayMs} ms`);
      await sleep(delayMs, signal);
    }
  }
  if (lastErr) throw lastErr;
}

/**
 * Download a URL into the OPFS cache file. Prefers parallel segmented direct
 * fetches (fast, throttle-resistant); falls back to the legacy single SW stream
 * when the offscreen doc can't probe HF directly or the server ignores Range.
 */
async function downloadToOpfs(url, fileName, onProgress, signal) {
  let probe;
  try {
    probe = await probeRangeDirect(url, signal);
  } catch (err) {
    if (isAbortErr(err, signal) || DIRECT) throw err;   // Firefox has no SW proxy fallback
    console.warn(`[sd] range probe failed for ${fileName} (${err?.message || err}); using SW stream`);
    return await legacyStream(url, fileName, onProgress, signal);
  }
  // Tiny files or a server that won't range → not worth segmenting.
  if (!probe.ranges || probe.size <= DL_BLOCK_BYTES) {
    return await legacyStream(url, fileName, onProgress, signal);
  }
  // One retry of the whole segmented pass: a terminal block failure persisted
  // the manifest, so this resumes and re-fetches only what's missing.
  try {
    return await segmentedDownload(url, fileName, probe.size, onProgress, signal);
  } catch (err) {
    if (isAbortErr(err, signal)) throw err;
    console.warn(`[sd] segmented download of ${fileName} failed (${err?.message || err}); resuming once more`);
    return await segmentedDownload(url, fileName, probe.size, onProgress, signal);
  }
}

async function fetchOnnxCachedImpl(url, onProgress, signal) {
  if (isLocalUrl(url)) return await fetchDirectToBuffer(url, onProgress, signal);
  const fileName = cacheKeyFor(url);
  const sizeKey = `${SIZE_KEY_PREFIX}${fileName}`;

  const stored = await swStorageGet(sizeKey);
  const expectedSize = stored?.[sizeKey] || 0;
  const onDiskSize = await opfsFileSize(fileName);

  // Complete cache hit — file size matches the size we recorded last time.
  if (expectedSize > 0 && onDiskSize === expectedSize) {
    console.info(`[sd] cache hit ${fileName} (${onDiskSize} bytes)`);
    onProgress?.(onDiskSize, onDiskSize);
    return await opfsReadFull(fileName);
  }

  console.info(onDiskSize > 0 ? `[sd] resuming ${fileName}` : `[sd] fresh download ${fileName}`);
  await downloadToOpfs(url, fileName, onProgress, signal);

  // Mark the file as complete by storing its size — only happens if the
  // download finished without aborting.
  const finalSize = await opfsFileSize(fileName);
  await swStorageSet({ [sizeKey]: finalSize });
  return await opfsReadFull(fileName);
}

/**
 * Stream a (possibly Range-resumed) HF response into the OPFS cache file
 * via a dedicated worker that owns a `FileSystemSyncAccessHandle`.
 *
 * Why a worker?  FileSystemWritableFileStream (the main-thread API) writes
 * to a swap file and atomically replaces the original on close — for a
 * multi-GB write that doubles peak disk usage and frequently trips
 * Chrome's quota check ("file could not be read … permission problems").
 * SyncAccessHandle is the lower-level API that writes straight to disk.
 * It's only available inside dedicated workers, hence the indirection.
 */
function streamOnnxToOpfs(url, fileName, rangeStart, onProgress, signal) {
  // Firefox background page: fetch in-page (no SW stream proxy) into the worker.
  if (DIRECT) return streamOnnxToOpfsDirect(url, fileName, rangeStart, onProgress, signal);
  return new Promise((resolve, reject) => {
    // Kill any stale writer still holding this file's handle from a prior
    // (cancelled/failed) attempt in this same long-lived offscreen document.
    const stale = _activeWriters.get(fileName);
    if (stale) {
      try { stale.terminate(); } catch {}
      _activeWriters.delete(fileName);
    }

    const port = chrome.runtime.connect({
      name: `fetch:${Date.now()}-${Math.random()}`,
    });
    const worker = new Worker(chrome.runtime.getURL("offscreen/workers/opfs-writer.js"));
    _activeWriters.set(fileName, worker);

    let received = 0;
    let total = 0;
    let settled = false;
    let chain = Promise.resolve();
    let writerOpen = false;

    const closeWriter = () =>
      new Promise((res) => {
        if (!writerOpen) return res();
        const onMsg = (e) => {
          if (e.data?.type === "closed" || e.data?.type === "error") {
            worker.removeEventListener("message", onMsg);
            res();
          }
        };
        worker.addEventListener("message", onMsg);
        worker.postMessage({ type: "close" });
      });

    const settle = (kind, payload) => {
      if (settled) return;
      settled = true;
      chain = chain.then(closeWriter, closeWriter).then(() => {
        try { worker.terminate(); } catch {}
        try { port.disconnect(); } catch {}
        if (_activeWriters.get(fileName) === worker) _activeWriters.delete(fileName);
        if (kind === "ok") resolve();
        else reject(payload instanceof Error ? payload : new Error(String(payload)));
      });
    };

    /** RPC the worker — returns a promise that resolves with its reply. */
    const callWorker = (msg) =>
      new Promise((res, rej) => {
        const onMsg = (e) => {
          worker.removeEventListener("message", onMsg);
          if (e.data?.type === "error") rej(new Error(e.data.error));
          else res(e.data);
        };
        worker.addEventListener("message", onMsg);
        worker.postMessage(msg);
      });

    const handleMessage = async (msg) => {
      if (settled) return;
      switch (msg?.type) {
        case "headers": {
          const actualStart = msg.rangeStartActual || 0;
          total = msg.total || 0;
          received = actualStart;
          const truncate = actualStart === 0;
          await callWorker({
            type: "open",
            dirName: ONNX_CACHE_DIR,
            fileName,
            truncate,
          });
          writerOpen = true;
          onProgress?.(received, total);
          break;
        }
        case "chunk": {
          if (!writerOpen) return;
          const bin = atob(msg.b64 || "");
          const arr = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          // Don't transfer arr.buffer — chunks are small (~64 KB) so the
          // structured-clone copy is cheap, and detaching it would zero
          // out arr.length when we read it for the progress update.
          await callWorker({ type: "write", chunk: arr });
          received += arr.length;
          onProgress?.(received, total);
          break;
        }
        case "done":
          settle("ok");
          break;
        case "error":
          settle("err", new Error(msg.message || "SW stream error"));
          break;
      }
    };

    port.onMessage.addListener((msg) => {
      if (signal?.aborted) { settle("err", new Error("aborted")); return; }
      chain = chain.then(() => handleMessage(msg)).catch((err) => {
        settle("err", err);
      });
    });
    port.onDisconnect.addListener(() => {
      settle("err", new Error(chrome.runtime.lastError?.message || "port disconnected"));
    });

    if (signal) {
      const onAbort = () => settle("err", new Error("aborted"));
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }

    port.postMessage({ type: "start", url, rangeStart });
  });
}

/**
 * Firefox path for streamOnnxToOpfs: fetch the (Range-resumed) response directly
 * in the background page and write it to the OPFS cache file via the same worker
 * (SyncAccessHandle), with no service-worker stream proxy. Mirrors the worker
 * RPC of the Chrome path; only the byte source differs (in-page fetch vs SW port).
 */
function streamOnnxToOpfsDirect(url, fileName, rangeStart, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const stale = _activeWriters.get(fileName);
    if (stale) {
      try { stale.terminate(); } catch {}
      _activeWriters.delete(fileName);
    }
    const worker = new Worker(chrome.runtime.getURL("offscreen/workers/opfs-writer.js"));
    _activeWriters.set(fileName, worker);

    const callWorker = (msg) =>
      new Promise((res, rej) => {
        const onMsg = (e) => {
          worker.removeEventListener("message", onMsg);
          if (e.data?.type === "error") rej(new Error(e.data.error));
          else res(e.data);
        };
        worker.addEventListener("message", onMsg);
        worker.postMessage(msg);
      });

    const cleanup = async () => {
      try { await callWorker({ type: "close" }); } catch {}
      try { worker.terminate(); } catch {}
      if (_activeWriters.get(fileName) === worker) _activeWriters.delete(fileName);
    };

    (async () => {
      try {
        const headers = {};
        if (rangeStart > 0) headers["Range"] = `bytes=${rangeStart}-`;
        const res = await fetch(url, { credentials: "omit", headers, signal });
        if (!(res.ok || res.status === 206)) throw new Error(`HTTP ${res.status} for ${url}`);
        // Range honoured → 206 resumes at rangeStart; a 200 means the server
        // ignored Range, so truncate and start from 0.
        const actualStart = res.status === 206 ? rangeStart : 0;
        const total = actualStart + (Number(res.headers.get("content-length")) || 0);
        await callWorker({ type: "open", dirName: ONNX_CACHE_DIR, fileName, truncate: actualStart === 0 });
        let received = actualStart;
        onProgress?.(received, total);
        const reader = res.body.getReader();
        for (;;) {
          if (signal?.aborted) { try { await reader.cancel("aborted"); } catch {} throw new Error("aborted"); }
          const { done, value } = await reader.read();
          if (done) break;
          await callWorker({ type: "write", chunk: value });
          received += value.byteLength;
          onProgress?.(received, total);
        }
        await cleanup();
        resolve();
      } catch (err) {
        await cleanup();
        reject(err?.name === "AbortError" ? new Error("aborted") : err);
      }
    })();
  });
}

async function fetchWithProgress(url, onProgress, signal) {
  console.info("[sd] GET", url);
  // First — direct fetch, fast path when the offscreen doc can reach HF.
  try {
    return await directFetchWithProgress(url, onProgress, signal);
  } catch (err) {
    if (signal?.aborted || err?.message === "aborted") throw err;
    console.warn(`[sd] direct fetch failed for ${url}: ${err.message}`);
    console.info("[sd] falling back to SW-stream proxy");
  }
  // Otherwise — SW-stream proxy, works even when offscreen has CORS issues.
  return await streamFetchViaSW(url, onProgress, signal);
}

async function directFetchWithProgress(url, onProgress, signal) {
  let res;
  try {
    res = await fetch(url, { credentials: "omit", signal });
  } catch (err) {
    if (signal?.aborted || err?.name === "AbortError") {
      throw new Error("aborted");
    }
    throw err;
  }
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);

  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    if (signal?.aborted) {
      try { await reader.cancel("aborted"); } catch {}
      throw new Error("aborted");
    }
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
  return buf.buffer;
}

/**
 * Fall-back path: ask the service worker to fetch the URL and stream the
 * response back to us as base64 chunks over a long-lived port.  Works
 * even when the offscreen document can't make the request directly.
 */
function streamFetchViaSW(url, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const port = chrome.runtime.connect({ name: `fetch:${Date.now()}-${Math.random()}` });
    const chunks = [];
    let total = 0;
    let received = 0;
    let settled = false;

    const fail = (err) => {
      if (settled) return; settled = true;
      try { port.disconnect(); } catch {}
      reject(err instanceof Error ? err : new Error(String(err)));
    };
    const finish = () => {
      if (settled) return; settled = true;
      const buf = new Uint8Array(received);
      let off = 0;
      for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
      try { port.disconnect(); } catch {}
      resolve(buf.buffer);
    };

    port.onMessage.addListener((msg) => {
      if (signal?.aborted) return fail(new Error("aborted"));
      switch (msg?.type) {
        case "headers":
          total = msg.total || 0;
          onProgress?.(0, total);
          break;
        case "chunk": {
          const bin = atob(msg.b64 || "");
          const arr = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          chunks.push(arr);
          received += arr.length;
          onProgress?.(received, total);
          break;
        }
        case "done": finish(); break;
        case "error": fail(new Error(msg.message || "SW stream error")); break;
      }
    });

    port.onDisconnect.addListener(() => {
      if (!settled) fail(new Error(chrome.runtime.lastError?.message || "port disconnected"));
    });

    if (signal) {
      const onAbort = () => fail(new Error("aborted"));
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }

    port.postMessage({ type: "start", url });
  });
}

// ─────────────────────────────────────────────────────────────
// Pre-cache all ONNX files for a repo without loading ORT sessions.
// Downloads text_encoder, unet, and vae_decoder to OPFS so the first
// Generate call only needs to compile shaders, not re-download weights.
// ─────────────────────────────────────────────────────────────
export async function preloadModelFiles(repoId, { progress, signal, encoderRepoId, unetFile = "model" } = {}) {
  // Multi-model split: the frozen encoders/VAE/tokenizers come from encoderRepoId
  // (shared across SDXL fine-tunes), only the UNet from repoId. Each subdir is
  // probed/cached against its own root. Local (localhost) roots are skipped —
  // their files aren't OPFS-cached (the loader fetches them fresh each run), and
  // a local dev server only exposes the UNet anyway. The UNet graph filename can
  // differ (quantized exports use a stem like "model_q4"); encoders/VAE are "model".
  const encRepo = encoderRepoId || repoId;
  const rootFor = (subdir) => repoRoot(subdir === "unet" ? repoId : encRepo);
  const stemFor = (subdir) => (subdir === "unet" ? unetFile : "model");
  const checkAbort = () => { if (signal?.aborted) throw new Error("aborted"); };

  async function ensureInOpfs(url, onProgress) {
    const fileName = cacheKeyFor(url);
    const sizeKey = `${SIZE_KEY_PREFIX}${fileName}`;
    const stored = await swStorageGet(sizeKey);
    const expectedSize = stored?.[sizeKey] || 0;
    const onDiskSize = await opfsFileSize(fileName);
    if (expectedSize > 0 && onDiskSize === expectedSize) {
      onProgress?.(onDiskSize);
      return;
    }
    await downloadToOpfs(url, fileName, onProgress, signal);
    const finalSize = await opfsFileSize(fileName);
    await swStorageSet({ [sizeKey]: finalSize });
  }

  async function probeSize(url) {
    const fileName = cacheKeyFor(url);
    const stored = await swStorageGet(`${SIZE_KEY_PREFIX}${fileName}`);
    const cached = stored?.[`${SIZE_KEY_PREFIX}${fileName}`] || 0;
    if (cached > 0) return cached;
    return urlSize(url).catch(() => 0);
  }

  // Discover a subdir's external-data sidecars. Handles both the single-file
  // convention (SD1.5: model.onnx_data or weights.pb) and SDXL's sharded UNet
  // (model.onnx_data, model.onnx_data_1, model.onnx_data_2, …) by probing the
  // numbered suffixes until one 404s. Returns [] for subdirs that don't exist
  // (e.g. text_encoder_2 on an SD1.5 repo).
  async function probeSubdir(subdir) {
    const base = `${rootFor(subdir)}/${subdir}`;
    const stem = stemFor(subdir);
    const graphFile = `${stem}.onnx`;
    // Local components aren't OPFS-cached — skip preload, they're fetched fresh.
    if (isLocalUrl(base)) return { subdir, base, graphFile, graphSize: 0, shards: [], dataSize: 0, exists: false };
    const graphSize = await probeSize(`${base}/${graphFile}`);
    const shards = [];
    let dataSize = 0;

    // weights.pb is a legacy single-file name; never sharded.
    const pb = await urlInfo(`${base}/weights.pb`);
    if (pb?.ok && pb.size > 1000) {
      shards.push({ name: "weights.pb", size: pb.size });
      dataSize += pb.size;
    } else {
      // ${stem}.onnx_data[, _1, _2, …]
      for (let i = 0; ; i++) {
        const name = i === 0 ? `${stem}.onnx_data` : `${stem}.onnx_data_${i}`;
        const info = await urlInfo(`${base}/${name}`);
        if (!info?.ok || info.size <= 1000) break;
        shards.push({ name, size: info.size });
        dataSize += info.size;
      }
    }
    const exists = graphSize > 0 || shards.length > 0;
    return { subdir, base, graphFile, graphSize, shards, dataSize, exists };
  }

  checkAbort();
  // text_encoder_2 is SDXL-only; probeSubdir reports exists:false on SD1.5.
  const all = await Promise.all(
    ["text_encoder", "text_encoder_2", "unet", "vae_decoder"].map(probeSubdir));
  const infos = all.filter((i) => i.exists);
  checkAbort();

  const overallTotal = infos.reduce((s, { graphSize, dataSize }) => s + graphSize + dataSize, 0);
  let cumulativeLoaded = 0;

  for (const { subdir, base, graphFile, graphSize, shards } of infos) {
    checkAbort();
    await ensureInOpfs(`${base}/${graphFile}`, (loaded) => {
      progress?.({ file: subdir, loaded: cumulativeLoaded + loaded, total: overallTotal, phase: "downloading" });
    });
    checkAbort();
    cumulativeLoaded += graphSize;

    for (const { name, size } of shards) {
      await ensureInOpfs(`${base}/${name}`, (loaded) => {
        progress?.({ file: `${subdir} weights`, loaded: cumulativeLoaded + loaded, total: overallTotal, phase: "downloading" });
      });
      checkAbort();
      cumulativeLoaded += size;
    }
  }
}
