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

// ─────────────────────────────────────────────────────────────
// Message routing
// ─────────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.target !== "offscreen") return;
  handle(msg).then(sendResponse).catch((err) => {
    const msg = errorMessage(err);
    if (/aborted|cancel/i.test(msg)) {
      sendResponse({ ok: false, error: msg, cancelled: true });
    } else {
      console.error("[offscreen]", err);
      sendResponse({ ok: false, error: msg, stack: err?.stack });
    }
  });
  return true;
});

async function handle(msg) {
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
    width = 512,
    height = 512,
    steps = 20,
    guidance_scale = 7.5,
    seed = -1,
    upscale: upscaleReq = 1,
    face_restore = false,
    face_restore_strength = 0.8,
  } = payload;

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
    pipeline = makePipeline(browser_kind);
    try {
      await pipeline.load(browser_repo_id, {
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
      });
    } catch (err) {
      if (signal.aborted || /aborted|cancel/i.test(String(err))) {
        // Drop the half-built pipeline so the next generate retries cleanly.
        pipeline = null;
        pipelineRepoId = null;
        return { ok: false, error: "cancelled", cancelled: true };
      }
      throw err;
    }
    pipelineRepoId = browser_repo_id;
  }

  const actualSeed = seed >= 0 ? seed : Math.floor(Math.random() * 2 ** 31);

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
  pipeline = makePipeline(kind);
  // Announce "preparing" and yield a beat so the popup paints the banner BEFORE
  // the synchronous WebGPU shader compile (esp. SDXL ~14 s) freezes the shared
  // GPU process — otherwise the freeze is a silent, unexplained hang. The last
  // painted frame (the banner) stays on screen through the freeze.
  postProgress({ warming: true });
  await sleep(250);
  try {
    await pipeline.load(repoId, { signal, schedulerType, dtype, encoderRepoId, unetIO, timestepType, unetFile, progress: () => {} });
    // Run one dummy pass to compile the remaining shaders (notably the VAE,
    // which otherwise compiles+freezes on the first real generation) here,
    // inside the framed "Preparing" warmup. Real generations then stay smooth.
    if (pipeline.warmup) await pipeline.warmup();
    pipelineRepoId = repoId;
    aborter = null;
    postProgress({ warming: "done" });
    return { ok: true };
  } catch (err) {
    pipeline = null;
    pipelineRepoId = null;
    aborter = null;
    postProgress({ warming: "done" });
    if (signal.aborted || /aborted|cancel/i.test(String(err))) {
      return { ok: false, cancelled: true, error: "cancelled" };
    }
    return { ok: false, error: String(err) };
  }
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
  if (pipeline) await pipeline.dispose();
  pipeline = null;
  pipelineRepoId = null;
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
function postProgress(body) {
  chrome.runtime.sendMessage({ type: "offscreen:progress", body }).catch(() => {});
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
