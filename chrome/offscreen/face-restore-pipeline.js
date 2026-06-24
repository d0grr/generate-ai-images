// ─────────────────────────────────────────────────────────────
// GFPGAN v1.4 face-restoration post-process.
//
//   model: d0gr/GFPGAN/GFPGANv1.4.onnx
//          ~324 MB, single file (no external-data sidecar), fp32 I/O,
//          standard ONNX ops (no com.microsoft.* contrib ops), so the
//          WebGPU EP runs it — anything it can't take falls back to WASM.
//
// Runs as a third ORT session inside the offscreen document, AFTER the SD
// pipeline emits its image and BEFORE the Real-ESRGAN upscale (GFPGAN wants
// a 512² face, so we restore at base resolution, then upscale the result).
//
// No face detector / alignment: we feed the whole frame resized to 512²,
// then resize back and blend over the original at `strength`. That works
// well for the close-up portraits SD smears the most; for tiny faces in a
// wide shot the gain is smaller. A RetinaFace crop+paste step is the next
// upgrade if/when needed.
// ─────────────────────────────────────────────────────────────

import { loadOrt, fetchOnnxCached } from "./sd-pipeline.js";

const HF_BASE = "https://huggingface.co";

const FACE_MODEL = {
  repoId: "d0gr/GFPGAN",
  path:   "GFPGANv1.4.onnx",
  size:   512,          // fixed square input the model was exported with
  sizeMb: 324,
};

export class FaceRestorer {
  constructor() {
    this.session = null;
  }

  /**
   * Download + compile the GFPGAN weights. Idempotent.
   * @param {object} opts
   * @param {(p:{file,loaded,total,phase})=>void} [opts.progress]
   * @param {AbortSignal} [opts.signal]
   */
  async load({ progress, signal } = {}) {
    if (this.session) return;

    const url = `${HF_BASE}/${FACE_MODEL.repoId}/resolve/main/${FACE_MODEL.path}`;
    console.info(`[face] load ${FACE_MODEL.repoId}`);

    const buffer = await fetchOnnxCached(url, (loaded, total) => {
      progress?.({ file: "face-restore weights", loaded, total, phase: "downloading" });
    }, signal);
    if (signal?.aborted) throw new Error("aborted");

    progress?.({ file: "face-restore", loaded: 1, total: 1, phase: "compiling" });
    const ort = loadOrt();
    this.session = await ort.InferenceSession.create(buffer, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all",
    });
  }

  /**
   * Restore faces in one ImageData. Returns a new ImageData at the SAME
   * dimensions as the input (so the upscale step downstream is unaffected).
   *
   * @param {ImageData} imageData
   * @param {object} [opts]
   * @param {number} [opts.strength=0.8]  blend of restored over original, 0..1
   */
  async run(imageData, { strength = 0.8 } = {}) {
    if (!this.session) throw new Error("FaceRestorer not loaded — call .load() first");
    const ort = loadOrt();
    const S = FACE_MODEL.size;
    const { width, height } = imageData;

    // 1. Resize the frame to the model's 512² input.
    const square = resizeImageData(imageData, S, S);
    const data = square.data;

    // 2. RGBA [0,255] → planar CHW float32 normalised to [-1, 1].
    const plane = S * S;
    const inF32 = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      const j = i * 4;
      inF32[i]             = (data[j]     / 255 - 0.5) / 0.5;   // R
      inF32[plane + i]     = (data[j + 1] / 255 - 0.5) / 0.5;   // G
      inF32[2 * plane + i] = (data[j + 2] / 255 - 0.5) / 0.5;   // B
    }

    const inputName  = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    const inputTensor = new ort.Tensor("float32", inF32, [1, 3, S, S]);

    const t0 = performance.now();
    const result = await this.session.run({ [inputName]: inputTensor });
    console.info(`[face] inference ${Math.round(performance.now() - t0)} ms`);

    // 3. Planar CHW [-1, 1] → RGBA [0, 255] at 512².
    const out = result[outputName];
    const outF32 = out.data instanceof Float32Array ? out.data : new Float32Array(out.data);
    const restoredRGBA = new Uint8ClampedArray(plane * 4);
    for (let i = 0; i < plane; i++) {
      const j = i * 4;
      restoredRGBA[j]     = denorm(outF32[i]);
      restoredRGBA[j + 1] = denorm(outF32[plane + i]);
      restoredRGBA[j + 2] = denorm(outF32[2 * plane + i]);
      restoredRGBA[j + 3] = 255;
    }
    const restoredSquare = new ImageData(restoredRGBA, S, S);

    // 4. Resize the restored frame back to the original dimensions and blend
    //    it over the source so we keep some original texture (GFPGAN alone
    //    can look plasticky) and don't fully repaint the background.
    const restored = (width === S && height === S)
      ? restoredSquare
      : resizeImageData(restoredSquare, width, height);

    const a = Math.max(0, Math.min(1, strength));
    if (a >= 1) return restored;
    const src = imageData.data;
    const dst = restored.data;
    for (let k = 0; k < dst.length; k += 4) {
      dst[k]     = src[k]     * (1 - a) + dst[k]     * a;
      dst[k + 1] = src[k + 1] * (1 - a) + dst[k + 1] * a;
      dst[k + 2] = src[k + 2] * (1 - a) + dst[k + 2] * a;
      dst[k + 3] = 255;
    }
    return restored;
  }

  dispose() {
    try { this.session?.release?.(); } catch {}
    this.session = null;
  }
}

// Denormalise one [-1, 1] sample → 0..255 (clip first, GFPGAN spills at edges).
function denorm(v) {
  const c = v < -1 ? -1 : v > 1 ? 1 : v;
  return Math.round(((c + 1) / 2) * 255);
}

// Resize an ImageData via OffscreenCanvas (available in the offscreen doc).
function resizeImageData(imageData, dstW, dstH) {
  const src = new OffscreenCanvas(imageData.width, imageData.height);
  src.getContext("2d").putImageData(imageData, 0, 0);
  const dst = new OffscreenCanvas(dstW, dstH);
  const ctx = dst.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, 0, 0, dstW, dstH);
  return ctx.getImageData(0, 0, dstW, dstH);
}
