// ─────────────────────────────────────────────────────────────
// Generate AI Images — WebGPU inference worker (runs in an offscreen document).
//
// Wires the StableDiffusionPipeline defined in ./sd-pipeline.js
// to the service worker's message protocol.
// ─────────────────────────────────────────────────────────────

// ORT-WASM emits its [W:...] warnings via emscripten's `printErr` which
// arrives at `console.error`.  Chrome's chrome://extensions page counts
// every console.error as a logged error and shows a red "Errors" badge.
// Demote ORT/Emscripten warnings to console.info so the badge stays clean
// without losing the messages from the actual DevTools console.
(function silenceOrtWarnings() {
  const origError = console.error.bind(console);
  console.error = (...args) => {
    const first = args[0];
    const text = typeof first === "string" ? first
              : first?.message ? String(first.message)
              : "";
    if (
      text.startsWith("[W:") ||
      text.includes("[W:onnxruntime") ||
      text.includes("VerifyEachNodeIsAssignedToAnEp") ||
      text.includes("non-minimal build will show node assignments") ||
      text.startsWith("[I:") ||
      text.startsWith("[V:")
    ) {
      console.info(...args);
      return;
    }
    origError(...args);
  };
})();

import { preloadModelFiles } from "./sd-pipeline.js";
import { SDXLPipeline } from "./sdxl-pipeline.js";
import { Upscaler } from "./upscale-pipeline.js";
import { FaceRestorer } from "./face-restore-pipeline.js";

/** The browser engine is SDXL-only (see catalog.js). `kind` is accepted for
 *  call-site stability but every catalog model is SDXL. */
function makePipeline(kind = "sdxl") {
  if (kind !== "sdxl") console.warn(`[offscreen] unknown model kind "${kind}" — using SDXL pipeline`);
  return new SDXLPipeline();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let pipeline = null;
let pipelineRepoId = null;
let upscaler = null;
let faceRestorer = null;
let aborter = null;
let lastDiagnostics = {};

// (C) Live engine-pipeline count, for leak diagnostics. Each pipeline holds ~5 GB
// of ORT/WebGPU sessions, so this must never exceed 1. ORT's native buffers are
// NOT freed by GC — only session.release() (via pipeline.dispose) frees them — so
// every pipeline MUST be routed through disposeActivePipeline()/swapPipeline()
// below rather than having its reference overwritten or dropped.
let livePipelines = 0;

// Release the current pipeline's ORT sessions and clear the refs. Safe to call
// when there is no pipeline. Use this everywhere instead of `pipeline = null`.
async function disposeActivePipeline() {
  if (!pipeline) return;
  try { await pipeline.dispose(); }
  catch (err) { console.warn("[offscreen] pipeline.dispose failed", err); }
  pipeline = null;
  pipelineRepoId = null;
  livePipelines = Math.max(0, livePipelines - 1);
  globalThis.__livePipelines = livePipelines;   // (C) readable in the bg-page console
}

// Build a fresh pipeline for `kind`, disposing the previous one first. This is the
// fix for the model-switch leak: warmup()/generate() used to reassign `pipeline`
// directly, orphaning the old model's sessions (unreachable, never released).
async function swapPipeline(kind) {
  await disposeActivePipeline();
  pipeline = makePipeline(kind);
  livePipelines++;
  globalThis.__livePipelines = livePipelines;   // (C) readable in the bg-page console
  if (livePipelines > 1) console.warn(`[offscreen] LEAK: ${livePipelines} live pipelines`);
  return pipeline;
}

// Build a fresh pipeline and load `repoId`, retrying ONCE on a non-abort failure.
// A cold WebGPU/wgpu init can panic on the first attempt ("index out of bounds",
// "Buffer unmapped") and then succeed on a clean re-init — this makes that
// recovery automatic instead of needing a manual reload. Each attempt disposes
// the prior sessions first (fresh GPU state). `warmupPass` runs the dummy
// shader-compile pass inside the retry. Sets pipelineRepoId on success.
// Returns { ok } | { ok:false, cancelled } | { ok:false, error }.
// Transient cold-start WebGPU/ORT failures that a clean re-init usually clears.
// Anything else (bad model, real OOM) won't improve on reload, so we DON'T retry
// it — reloading would just allocate another multi-GB copy for nothing.
function isColdInitError(err) {
  const m = String(err?.message || err).toLowerCase();
  return /index out of bounds|out of bounds|buffer unmapped|mapasync|device\s*lost|lost.*device|wgpu|panic/.test(m);
}

async function initPipelineWithRetry(kind, repoId, loadOpts, { signal, warmupPass = false } = {}) {
  const MAX = 2;
  let lastErr;
  for (let attempt = 1; attempt <= MAX; attempt++) {
    await swapPipeline(kind);
    try {
      await pipeline.load(repoId, loadOpts);
      if (warmupPass && pipeline.warmup) await pipeline.warmup();
      pipelineRepoId = repoId;
      return { ok: true };
    } catch (err) {
      await disposeActivePipeline();
      if (signal?.aborted || /aborted|cancel/i.test(String(err))) {
        return { ok: false, cancelled: true, error: "cancelled" };
      }
      lastErr = err;
      if (attempt < MAX && isColdInitError(err)) {
        console.warn(`[offscreen] cold engine init failed (attempt ${attempt}/${MAX}) — releasing GPU + retrying:`,
          String(err?.message || err));
        // Pause so Firefox's GPU process actually reclaims the just-released
        // buffers before we load a fresh copy — otherwise peak RAM doubles.
        await sleep(2000);
        continue;
      }
      break;   // not a cold-init error (or out of attempts) — fail fast, no reload
    }
  }
  return { ok: false, error: String(lastErr?.message || lastErr) };
}

// ─────────────────────────────────────────────────────────────
// Message routing
// ─────────────────────────────────────────────────────────────
// Chrome runs this engine in a separate offscreen document and talks to it over
// chrome.runtime.sendMessage, so we register an onMessage listener. Firefox runs
// the engine inside the (single) background page and calls handle() directly —
// runtime.sendMessage doesn't round-trip to the sender's own context — so it sets
// globalThis.__ENGINE_DIRECT__ (in background.html, before this module loads) to
// skip the listener. handle() and setProgressSink() are exported for that path.
if (!globalThis.__ENGINE_DIRECT__) {
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg?.target !== "offscreen") return;
    handle(msg).then(sendResponse).catch((err) => {
      const m = errorMessage(err);
      if (/aborted|cancel/i.test(m)) {
        sendResponse({ ok: false, error: m, cancelled: true });
      } else {
        console.error("[offscreen]", err);
        sendResponse({ ok: false, error: m, stack: err?.stack });
      }
    });
    return true;
  });
}

export async function handle(msg) {
  switch (msg.type) {
    case "probe":       return await probe();
    case "warmup":      return await warmup(msg.repoId, msg.schedulerType, msg.dtype, msg.kind, msg.encoderRepoId, msg.unetIO, msg.timestep, msg.unetFile);
    case "generate":    return await generate(msg.payload);
    case "cancel":      return cancel();
    case "unload":      return await unload();
    case "preload_model":        return await preloadModel(msg.repoId, msg.modelId, msg.encoderRepoId, msg.unetFile);
    case "delete_browser_model": return await deleteBrowserModel(msg.prefix);
    case "delete_opfs":          return await deleteOpfs(msg.opfsKey);
    case "clear_all_opfs": return await clearAllOpfs();
    default:               return { ok: false, error: `unknown type: ${msg.type}` };
  }
}

// ─────────────────────────────────────────────────────────────
// WebGPU probe — used by the SW on startup
// ─────────────────────────────────────────────────────────────
async function probe() {
  if (!("gpu" in navigator)) {
    lastDiagnostics = {
      engine: "browser",
      device: "cpu-fallback",
      webgpu: false,
      adapter: null,
      vramGb: 0,
      reason: "WebGPU unavailable — use Chrome 113+ or Edge 113+",
    };
    return { ok: true, info: lastDiagnostics };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      lastDiagnostics = {
        engine: "browser", device: "cpu-fallback", webgpu: false,
        adapter: null, vramGb: 0,
        reason: "No GPU adapter available",
      };
      return { ok: true, info: lastDiagnostics };
    }
    const info = await adapter.requestAdapterInfo?.().catch(() => ({}));
    lastDiagnostics = {
      engine: "browser",
      device: "webgpu",
      webgpu: true,
      adapter: info?.description || info?.device || "WebGPU GPU",
      vendor: info?.vendor || "",
      architecture: info?.architecture || "",
      vramGb: 4,
    };
    return { ok: true, info: lastDiagnostics };
  } catch (err) {
    lastDiagnostics = { engine: "browser", device: "cpu-fallback", webgpu: false, reason: String(err) };
    return { ok: false, error: String(err) };
  }
}


// ─────────────────────────────────────────────────────────────
// Generate — streams progress, returns thumb + full image
// ─────────────────────────────────────────────────────────────
async function generate(payload) {
  const {
    prompt,
    negative_prompt = "",
    model_id,
    browser_repo_id,
    browser_scheduler = "euler",
    browser_dtype = "float16",
    browser_kind = "sd15",
    browser_encoder_repo_id,
    browser_unet_io = "float32",
    browser_timestep = "int64",
    browser_unet_file = "model",
    // img2img / editing: VAE encoder repo (image → latent) + the source image
    // (base64 PNG/data-URL) and edit strength.
    browser_vae_encoder_repo_id = null,
    browser_vae_encoder_file = "model",
    browser_vae_encoder_io = "float32",
    init_image_b64 = null,
    mask_b64 = null,
    strength = 0.6,
    width = 512,
    height = 512,
    steps = 20,
    guidance_scale = 7.5,
    seed = -1,
    upscale: upscaleReq = 1,
    face_restore = false,
    face_restore_strength = 0.8,
  } = payload;

  const isEdit = !!init_image_b64;

  // SDXL renders at native 1024² — skip the Real-ESRGAN ×2 (would make 2048).
  const upscale = browser_kind === "sdxl" ? 1 : upscaleReq;

  console.info(`[offscreen] generate ${width}×${height} · upscale=${upscale} · model=${model_id}`);

  if (!browser_repo_id) {
    throw new Error("This model has no browser-compatible (ONNX/WebGPU) variant.");
  }

  if (!("gpu" in navigator)) {
    throw new Error("WebGPU not available in this browser. Use Chrome/Edge 122+ with WebGPU enabled.");
  }

  // Set up the abort controller BEFORE pipeline.load so cancel works during
  // the multi-minute first-run model download.
  aborter = new AbortController();
  const { signal } = aborter;
  const started = performance.now();

  // Load or reuse pipeline.
  if (!pipeline || pipelineRepoId !== browser_repo_id) {
    postProgress({ phase: "loading", step: 0, total: steps, file: "initialising" });
    // Retry-once on a cold WebGPU init failure (mirrors warmup); sets
    // pipelineRepoId on success and disposes partial sessions on failure.
    const r = await initPipelineWithRetry(browser_kind, browser_repo_id, {
      signal,
      schedulerType: browser_scheduler,
      dtype: browser_dtype,
      encoderRepoId: browser_encoder_repo_id,
      unetIO: browser_unet_io,
      timestepType: browser_timestep,
      unetFile: browser_unet_file,
      progress: (p) => {
        postProgress({
          phase: p.phase || "downloading",
          step: 0,
          total: steps,
          file: p.file,
          downloaded: p.loaded,
          total_bytes: p.total,
        });
      },
    }, { signal });
    if (!r.ok) {
      if (r.cancelled) return { ok: false, error: "cancelled", cancelled: true };
      throw new Error(r.error);
    }
  }

  const actualSeed = seed >= 0 ? seed : Math.floor(Math.random() * 2 ** 31);

  // img2img: configure the VAE encoder on the (possibly reused) pipeline and
  // decode the source image to a native-1024² ImageData before generating.
  let initImageData = null;
  let maskLatent = null;
  if (isEdit) {
    pipeline.vaeEncoderRepoId = browser_vae_encoder_repo_id;
    pipeline.vaeEncoderFile = browser_vae_encoder_file;
    pipeline.vaeEncoderIO = browser_vae_encoder_io;
    try {
      initImageData = await decodeInitImage(init_image_b64);
    } catch (err) {
      throw new Error(`Could not read the source image: ${err?.message || err}`);
    }
    if (mask_b64) {
      try {
        maskLatent = await decodeMaskToLatent(mask_b64);
      } catch (err) {
        console.warn("[offscreen] mask decode failed — falling back to global img2img:", err);
      }
    }
  }

  let imageData;
  try {
    imageData = await pipeline.generate({
      prompt,
      negative_prompt,
      steps,
      guidance_scale,
      width,
      height,
      seed: actualSeed,
      signal,
      initImage: initImageData,
      mask: maskLatent,
      strength,
      onEncodeProgress: (p) => {
        postProgress({
          phase: p.phase || "encoding",
          step: 0,
          total: steps,
          file: p.file,
          downloaded: p.loaded,
          total_bytes: p.total,
        });
      },
      onStep: (step, total) => {
        postProgress({ phase: "denoising", step, total });
      },
      // SDXL only: announce the ~2 s VAE decode and yield so the popup paints a
      // "Decoding image…" state before the GPU-bound pass stalls the compositor.
      onDecode: async () => {
        postProgress({ phase: "decoding", step: steps, total: steps });
        await sleep(200);
      },
      onPreview: async (latents, lh, lw) => {
        try {
          const preview = await encodeLatentPreview(latents, lh, lw);
          postProgress({ phase: "denoising", preview });
        } catch {}
      },
    });
  } catch (err) {
    if (signal.aborted || /aborted|cancel/i.test(String(err))) {
      return { ok: false, error: "cancelled", cancelled: true };
    }
    throw err;
  }

  // Inpaint: paste the original pixels back into the UNPAINTED region so it stays
  // bit-for-bit (the latent-blend keeps it coherent during denoise, but the global
  // VAE round-trip still drifts it slightly). Feathered by the upscaled mask so the
  // seam is soft. Only the painted region keeps the freshly generated pixels.
  if (maskLatent && initImageData && !signal.aborted) {
    try {
      imageData = compositeInpaint(imageData, initImageData, maskLatent);
    } catch (err) {
      console.warn("[offscreen] inpaint composite failed — using raw output:", err);
    }
  }

  // Optional GFPGAN face restoration. Runs BEFORE upscale so the model gets
  // the 512² face it expects; the restored frame is then upscaled normally.
  // Failures are non-fatal — fall through with the un-restored image.
  if (face_restore && !signal.aborted) {
    console.info(`[offscreen] face-restore starting (strength=${face_restore_strength})`);
    try {
      if (!faceRestorer) faceRestorer = new FaceRestorer();
      postProgress({ phase: "face-restore", step: steps, total: steps });
      await faceRestorer.load({
        signal,
        progress: (p) => {
          postProgress({
            phase: "face-restore",
            step: steps,
            total: steps,
            file: p.file,
            downloaded: p.loaded,
            total_bytes: p.total,
          });
        },
      });
      imageData = await faceRestorer.run(imageData, { strength: face_restore_strength });
      console.info("[offscreen] face-restore done");
    } catch (err) {
      if (signal.aborted || /aborted/i.test(String(err))) {
        return { ok: false, error: "cancelled", cancelled: true };
      }
      console.error("[offscreen] face-restore failed — using un-restored image:", err);
    }
  }

  // Optional Real-ESRGAN upscale. Runs as a second ORT session; the model
  // downloads once on first use and lives in OPFS thereafter.
  if (upscale >= 2 && !signal.aborted) {
    console.info(`[offscreen] upscale ×${upscale} starting (input ${imageData.width}×${imageData.height})`);
    try {
      if (!upscaler) upscaler = new Upscaler();
      postProgress({ phase: "upscaling", step: steps, total: steps });
      await upscaler.load({
        factor: upscale,
        signal,
        progress: (p) => {
          postProgress({
            phase: "upscaling",
            step: steps,
            total: steps,
            file: p.file,
            downloaded: p.loaded,
            total_bytes: p.total,
          });
        },
      });
      imageData = await upscaler.run(imageData);
      console.info(`[offscreen] upscale done (output ${imageData.width}×${imageData.height})`);
    } catch (err) {
      // Don't fail the whole generation if upscale dies — log loudly and
      // fall through with the original 512-sized image.
      if (signal.aborted || /aborted/i.test(String(err))) {
        return { ok: false, error: "cancelled", cancelled: true };
      }
      console.error("[offscreen] upscale failed — returning base resolution:", err);
    }
  } else {
    console.info(`[offscreen] upscale skipped (upscale=${upscale}, aborted=${signal.aborted})`);
  }

  // imageData may now be at upscaled dimensions — use its own size.
  const outW = imageData.width;
  const outH = imageData.height;

  // Thumb for storage.local (small), full PNG streams to OPFS.
  const thumbB64 = await encodeImage(imageData, 256 / Math.max(outW, outH));
  const opfsKey = await saveFullToOpfs(imageData);

  return {
    ok: true,
    thumb_b64: thumbB64,
    opfs_key: opfsKey,             // relative path inside origin's OPFS
    image_path: "",
    seed: actualSeed,
    elapsed_ms: Math.round(performance.now() - started),
    prompt,
    negative_prompt,
    model_id,
    width: outW,
    height: outH,
    // Generation params — persisted with the saved record so the workspace
    // gallery can show them in the image properties panel.
    steps,
    guidance_scale,
    scheduler: browser_scheduler,
  };
}

// ─────────────────────────────────────────────────────────────
// Warmup — loads the model into ORT sessions so the first Generate
// is instant.  Called when the popup opens.  No inference is run.
// ─────────────────────────────────────────────────────────────
async function warmup(repoId, schedulerType = "euler", dtype = "float16", kind = "sd15",
                      encoderRepoId, unetIO = "float32", timestepType = "int64", unetFile = "model") {
  if (!repoId) return { ok: false, error: "no repoId" };
  if (pipeline && pipelineRepoId === repoId) return { ok: true, cached: true };

  aborter = new AbortController();
  const { signal } = aborter;
  // Announce "preparing" and yield a beat so the popup paints the banner BEFORE
  // the synchronous WebGPU shader compile (esp. SDXL ~14 s) freezes the shared
  // GPU process — otherwise the freeze is a silent, unexplained hang. The last
  // painted frame (the banner) stays on screen through the freeze.
  postProgress({ warming: true });
  await sleep(250);
  // warmupPass runs the dummy shader-compile pass; retry-once recovers a cold
  // WebGPU init failure (e.g. wgpu "index out of bounds") transparently.
  const r = await initPipelineWithRetry(
    kind, repoId,
    { signal, schedulerType, dtype, encoderRepoId, unetIO, timestepType, unetFile, progress: () => {} },
    { signal, warmupPass: true },
  );
  aborter = null;
  postProgress({ warming: "done" });
  if (r.ok) return { ok: true };
  if (r.cancelled) return { ok: false, cancelled: true, error: "cancelled" };
  return { ok: false, error: r.error };
}

// ─────────────────────────────────────────────────────────────
// Pre-cache model files — downloads ONNX weights to OPFS without
// creating ORT sessions.  Triggered when the user clicks "Use" on
// a browser model so Generate can start immediately afterwards.
// ─────────────────────────────────────────────────────────────
async function preloadModel(repoId, modelId, encoderRepoId, unetFile = "model") {
  if (!repoId) return { ok: false, error: "No browserRepoId" };

  aborter = new AbortController();
  const { signal } = aborter;

  try {
    await preloadModelFiles(repoId, {
      signal,
      encoderRepoId,
      unetFile,
      progress: (p) => {
        postProgress({
          phase: p.phase || "downloading",
          file: p.file,
          downloaded: p.loaded,
          total_bytes: p.total,
          model_id: modelId,
          preloading: true,
        });
      },
    });
    aborter = null;
    return { ok: true };
  } catch (err) {
    aborter = null;
    if (signal.aborted || /aborted|cancel/i.test(String(err))) {
      return { ok: false, cancelled: true, error: "cancelled" };
    }
    throw err;
  }
}

// SDXL native side — the VAE encoder expects the source at the model's
// resolution, so we cover-fit (scale to fill, centre-crop) the user's image
// into a 1024² canvas. Cover (not contain) avoids letterbox bars that the model
// would otherwise try to "paint over".
const SDXL_SIDE = 1024;
async function decodeInitImage(b64) {
  // Decode the base64 ourselves — the extension-page CSP `connect-src` has no
  // `data:` source, so fetch("data:…") is blocked. atob → bytes → Blob instead.
  let mime = "image/png";
  let raw = b64;
  if (b64.startsWith("data:")) {
    const comma = b64.indexOf(",");
    const header = b64.slice(5, comma);          // e.g. "image/png;base64"
    mime = header.split(";")[0] || mime;
    raw = b64.slice(comma + 1);
  }
  const bin = atob(raw);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: mime });
  const bmp = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(SDXL_SIDE, SDXL_SIDE);
  const ctx = canvas.getContext("2d");
  const scale = Math.max(SDXL_SIDE / bmp.width, SDXL_SIDE / bmp.height);
  const dw = bmp.width * scale, dh = bmp.height * scale;
  ctx.drawImage(bmp, (SDXL_SIDE - dw) / 2, (SDXL_SIDE - dh) / 2, dw, dh);
  bmp.close?.();
  return ctx.getImageData(0, 0, SDXL_SIDE, SDXL_SIDE);
}

// Inpaint mask → single-channel Float32Array at latent resolution (128² = 1024/8),
// values 0..1 where 1 = regenerate. The mask PNG (white strokes on black) is
// cover-fit with the SAME centre-crop as decodeInitImage so it stays pixel-aligned
// with the source. Downscaling to 128² also feathers the stroke edges, which
// softens the inpaint seam.
const LATENT_SIDE = SDXL_SIDE / 8;   // 128
async function decodeMaskToLatent(b64) {
  let raw = b64, mime = "image/png";
  if (b64.startsWith("data:")) {
    const comma = b64.indexOf(",");
    mime = b64.slice(5, comma).split(";")[0] || mime;
    raw = b64.slice(comma + 1);
  }
  const bin = atob(raw);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const bmp = await createImageBitmap(new Blob([bytes], { type: mime }));
  const canvas = new OffscreenCanvas(LATENT_SIDE, LATENT_SIDE);
  const ctx = canvas.getContext("2d");
  const scale = Math.max(LATENT_SIDE / bmp.width, LATENT_SIDE / bmp.height);
  const dw = bmp.width * scale, dh = bmp.height * scale;
  ctx.drawImage(bmp, (LATENT_SIDE - dw) / 2, (LATENT_SIDE - dh) / 2, dw, dh);
  bmp.close?.();
  const { data } = ctx.getImageData(0, 0, LATENT_SIDE, LATENT_SIDE);
  const mask = new Float32Array(LATENT_SIDE * LATENT_SIDE);
  // Use luminance of the red channel (white stroke → 1). Anything painted at all
  // counts toward editing; the soft downscaled edge gives a gradient blend.
  for (let i = 0; i < mask.length; i++) mask[i] = data[i * 4] / 255;
  return mask;
}

// Blend generated over original by a feathered full-res mask (the 128² latent
// mask upscaled smoothly to the image size). out = m·generated + (1−m)·original.
function compositeInpaint(generated, original, maskLatent) {
  const W = generated.width, H = generated.height;
  // 128² mask → grayscale ImageData → upscale (smoothed) to W×H for soft edges.
  const small = new OffscreenCanvas(LATENT_SIDE, LATENT_SIDE);
  const mSmall = new ImageData(LATENT_SIDE, LATENT_SIDE);
  for (let i = 0; i < maskLatent.length; i++) {
    const v = Math.max(0, Math.min(255, Math.round(maskLatent[i] * 255)));
    mSmall.data[i * 4] = mSmall.data[i * 4 + 1] = mSmall.data[i * 4 + 2] = v;
    mSmall.data[i * 4 + 3] = 255;
  }
  small.getContext("2d").putImageData(mSmall, 0, 0);
  const big = new OffscreenCanvas(W, H);
  const bctx = big.getContext("2d");
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = "high";
  bctx.drawImage(small, 0, 0, W, H);
  const mFull = bctx.getImageData(0, 0, W, H).data;

  const g = generated.data, o = original.data;
  const out = new ImageData(W, H);
  for (let p = 0; p < W * H; p++) {
    const m = mFull[p * 4] / 255, k = 1 - m;
    out.data[p * 4]     = g[p * 4]     * m + o[p * 4]     * k;
    out.data[p * 4 + 1] = g[p * 4 + 1] * m + o[p * 4 + 1] * k;
    out.data[p * 4 + 2] = g[p * 4 + 2] * m + o[p * 4 + 2] * k;
    out.data[p * 4 + 3] = 255;
  }
  return out;
}

async function saveFullToOpfs(imageData) {
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle("generations", { create: true });
    const fname = `${crypto.randomUUID()}.png`;
    const handle = await dir.getFileHandle(fname, { create: true });
    const writable = await handle.createWritable();

    const canvas = new OffscreenCanvas(imageData.width, imageData.height);
    canvas.getContext("2d").putImageData(imageData, 0, 0);
    const blob = await canvas.convertToBlob({ type: "image/png" });

    await writable.write(blob);
    await writable.close();
    return `generations/${fname}`;
  } catch (err) {
    console.warn("[offscreen] OPFS save failed", err);
    return "";
  }
}

async function clearAllOpfs() {
  try {
    const root = await navigator.storage.getDirectory();
    const names = [];
    for await (const [name] of root.entries()) names.push(name);
    await Promise.all(names.map((n) => root.removeEntry(n, { recursive: true })));
    return { ok: true };
  } catch (err) {
    console.warn("[offscreen] OPFS clear failed", err);
    return { ok: false, error: String(err) };
  }
}

async function deleteBrowserModel(prefix) {
  if (!prefix) return { ok: false, error: "no prefix" };
  try {
    // `prefix` is the OPFS cache-key prefix for this model's files, computed by
    // the service worker (chrome.storage isn't reachable here on all builds).
    const root = await navigator.storage.getDirectory();
    const cacheDir = await root.getDirectoryHandle("models-cache", { create: false }).catch(() => null);
    const removed = [];
    if (cacheDir) {
      for await (const [name] of cacheDir.entries()) {
        if (name.startsWith(prefix)) {
          await cacheDir.removeEntry(name).catch(() => {});
          removed.push(name);
        }
      }
    }

    // The `onnx-size:` storage keys are cleaned by the service worker —
    // chrome.storage isn't reliably reachable from the offscreen document.
    console.info(`[offscreen] deleted ${removed.length} files for prefix ${prefix}`);
    return { ok: true, removed: removed.length };
  } catch (err) {
    console.warn("[offscreen] deleteBrowserModel failed", err);
    return { ok: false, error: String(err) };
  }
}

async function deleteOpfs(opfsKey) {
  if (!opfsKey) return { ok: true };
  try {
    const [dirName, file] = opfsKey.split("/");
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(dirName);
    await dir.removeEntry(file);
    return { ok: true };
  } catch (err) {
    console.warn("[offscreen] OPFS delete failed", err);
    return { ok: false, error: String(err) };
  }
}

function cancel() {
  if (aborter) aborter.abort();
  return { ok: true };
}

async function unload() {
  await disposeActivePipeline();
  try { faceRestorer?.dispose(); } catch {}
  faceRestorer = null;
  try { upscaler?.dispose(); } catch {}
  upscaler = null;
  return { ok: true };
}

// ─────────────────────────────────────────────────────────────
// Image encoding — ImageData → base64 PNG
// ─────────────────────────────────────────────────────────────
async function encodeImage(imageData, scale = 1.0) {
  const w = imageData.width, h = imageData.height;
  const tw = Math.max(1, Math.round(w * scale));
  const th = Math.max(1, Math.round(h * scale));

  // Draw the ImageData into a source canvas.
  const src = new OffscreenCanvas(w, h);
  src.getContext("2d").putImageData(imageData, 0, 0);

  let finalCanvas = src;
  if (tw !== w || th !== h) {
    finalCanvas = new OffscreenCanvas(tw, th);
    const ctx = finalCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(src, 0, 0, tw, th);
  }

  const blob = await finalCanvas.convertToBlob({ type: "image/png" });
  return await blobToBase64(blob);
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const comma = reader.result.indexOf(",");
      resolve(reader.result.slice(comma + 1));
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// ─────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────
// Progress is pushed through a sink. Chrome's default relays to the SW over
// runtime.sendMessage; Firefox's background page injects a direct callback via
// setProgressSink (same context — a message wouldn't reach the core).
let progressSink = (body) => {
  chrome.runtime.sendMessage({ type: "offscreen:progress", body }).catch(() => {});
};
export function setProgressSink(fn) { progressSink = fn; }

function postProgress(body) {
  progressSink(body);
}

async function encodeLatentPreview(latents, lh, lw) {
  const plane = lh * lw;
  const pixels = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    pixels[i * 4]     = Math.min(255, Math.max(0, latents[i]             * 127.5 + 127.5));
    pixels[i * 4 + 1] = Math.min(255, Math.max(0, latents[i + plane]     * 127.5 + 127.5));
    pixels[i * 4 + 2] = Math.min(255, Math.max(0, latents[i + 2 * plane] * 127.5 + 127.5));
    pixels[i * 4 + 3] = 255;
  }
  const canvas = new OffscreenCanvas(lw, lh);
  canvas.getContext("2d").putImageData(new ImageData(pixels, lw, lh), 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 });
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function errorMessage(err) {
  if (!err) return "Unknown error";
  if (typeof err === "string") return err;
  return err.message || String(err);
}
