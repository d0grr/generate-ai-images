// ─────────────────────────────────────────────────────────────
// Real-ESRGAN post-process upscaler.
//
//   x2 model: tamnvcc/RealESRGAN-onnx/onnx/RealESRGAN_x2plus.fp16.onnx
//             ~34 MB, fp16 weights, fp16 I/O. 512×512 → 1024×1024.
//
// Runs as a second ORT session inside the same offscreen document, after
// the SD pipeline emits its 512-sized ImageData. Shares ort/OPFS-cache
// helpers with sd-pipeline.js, so the model downloads once and lives in
// OPFS like any other browser model.
// ─────────────────────────────────────────────────────────────

import { loadOrt, fetchOnnxCached } from "./sd-pipeline.js";

const HF_BASE = "https://huggingface.co";

const UPSCALER_REPOS = {
  2: {
    repoId: "tamnvcc/RealESRGAN-onnx",
    path:   "onnx/RealESRGAN_x2plus.fp16.onnx",
    sizeMb: 34,
  },
};

export class Upscaler {
  constructor() {
    this.session = null;
    this.factor  = null;
  }

  /**
   * Download + compile the upscaler weights. Idempotent — re-calling with
   * the same factor is a no-op once the session is built.
   *
   * @param {object} opts
   * @param {2}      [opts.factor]   only 2× is supported today
   * @param {(p:{file,loaded,total,phase})=>void} [opts.progress]
   * @param {AbortSignal} [opts.signal]
   */
  async load({ factor = 2, progress, signal } = {}) {
    if (this.session && this.factor === factor) return;
    const spec = UPSCALER_REPOS[factor];
    if (!spec) throw new Error(`upscaler factor ${factor} not supported`);

    const url = `${HF_BASE}/${spec.repoId}/resolve/main/${spec.path}`;
    console.info(`[upscale] load ${spec.repoId} (×${factor})`);

    const buffer = await fetchOnnxCached(url, (loaded, total) => {
      progress?.({ file: "upscaler weights", loaded, total, phase: "downloading" });
    }, signal);
    if (signal?.aborted) throw new Error("aborted");

    progress?.({ file: "upscaler", loaded: 1, total: 1, phase: "compiling" });
    const ort = loadOrt();
    this.session = await ort.InferenceSession.create(buffer, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all",
    });
    this.factor = factor;
  }

  /**
   * Upscale one ImageData by the loaded factor.
   *
   * The model expects an [N, C, H, W] fp16 tensor in [0, 1]. We pack the
   * RGB channels of the source ImageData into planar CHW, run inference,
   * then unpack the output back into RGBA ImageData (alpha forced to 255
   * since Real-ESRGAN ignores alpha).
   */
  async run(imageData) {
    if (!this.session) throw new Error("Upscaler not loaded — call .load() first");
    const ort = loadOrt();
    const { data, width, height } = imageData;

    // RGBA Uint8ClampedArray → planar CHW Float32Array in [0, 1].
    const plane = width * height;
    const inF32 = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      const j = i * 4;
      inF32[i]             = data[j]     / 255;   // R plane
      inF32[plane + i]     = data[j + 1] / 255;   // G plane
      inF32[2 * plane + i] = data[j + 2] / 255;   // B plane
    }

    const inputName  = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    // tamnvcc's "fp16" build keeps fp16 weights internally but exposes fp32
    // I/O ports (Cast nodes at the boundaries). Passing float16 here trips
    // "Unexpected input data type. Actual: (tensor(float16)), expected:
    // (tensor(float))" at OrtRun. Feed and receive fp32.
    const inputTensor = new ort.Tensor("float32", inF32, [1, 3, height, width]);

    const t0 = performance.now();
    const result = await this.session.run({ [inputName]: inputTensor });
    console.info(`[upscale] ×${this.factor} inference ${Math.round(performance.now() - t0)} ms`);

    const out = result[outputName];
    const newW = width * this.factor;
    const newH = height * this.factor;
    const outPlane = newW * newH;
    const outF32 = out.data instanceof Float32Array ? out.data : new Float32Array(out.data);

    // Planar CHW [0, 1] → packed RGBA Uint8ClampedArray [0, 255].
    const rgba = new Uint8ClampedArray(outPlane * 4);
    for (let i = 0; i < outPlane; i++) {
      const j = i * 4;
      rgba[j]     = clamp255(outF32[i]);
      rgba[j + 1] = clamp255(outF32[outPlane + i]);
      rgba[j + 2] = clamp255(outF32[2 * outPlane + i]);
      rgba[j + 3] = 255;
    }
    return new ImageData(rgba, newW, newH);
  }

  dispose() {
    try { this.session?.release?.(); } catch {}
    this.session = null;
    this.factor  = null;
  }
}

function clamp255(v) {
  // Real-ESRGAN sometimes spills slightly out of [0, 1] at edges — clip.
  return Math.max(0, Math.min(255, Math.round(v * 255)));
}
