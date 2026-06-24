// ─────────────────────────────────────────────────────────────
// Generate AI Images — static model catalog (seed for the Models tab).
//
// Each entry carries:
//   • repoId         — the upstream checkpoint (reference / dedup key)
//   • browserRepoId  — the ONNX variant that runs in WebGPU (onnxruntime-web)
// ─────────────────────────────────────────────────────────────

const RAW_CATALOG = [
  {
    id: "sdxl-lightning-4step",
    repoId: "ByteDance/SDXL-Lightning",
    browserRepoId: "d0gr/sdxl-lightning-onnx-webgpu",
    browserKind: "sdxl",
    browserSizeGb: 6.9,
    browserScheduler: "euler",          // Euler with trailing spacing (set by kind)
    browserDtype: "float32",            // SDXL ONNX has fp32 I/O (fp16 internals)
    // img2img: the SDXL VAE is frozen, so a standalone SDXL VAE export pairs
    // with our (Lightning) decoder. Clean pytorch ONNX export (standard Conv —
    // ort-web has no kernel for Olive's NhwcConv) with fp32 I/O (fp16 VAE NaNs on
    // WebGPU). Its `latent_parameters` output is 8-ch moments; the pipeline takes
    // the mean (mode). ~137 MB, cached in OPFS after first use.
    browserVaeEncoderRepoId: "d0gr/sdxl-vae-onnx",
    browserVaeEncoderFile: "model",
    browserVaeEncoderIO: "float32",
    browserDefaultSteps: 6,
    browserDefaultGuidance: 1.5,        // mild CFG (2nd UNet pass per step)
    nameLead: "SDXL Lightning",
    nameTail: "Quality",
    description: "1024x1024 native SDXL · photorealistic, sharp faces, fine detail · 4-step Lightning, fast on most GPUs",
    kind: "sdxl",
    kindLabel: "SDXL",
    precision: "fp16",
    sizeGb: 6.9,
    minVramGb: 8,
    authorShort: "ByteDance",
    avgSecondsPerFrame: null,
    avgSecondsBrowser: 18,
    gated: false,
  },
  {
    // The SAME ByteDance SDXL-Lightning, but the UNet is 4-bit weight-only
    // quantized (MatMulNBits) → 1.86 GB single file, no sharding, hosted on HF
    // (no hosting on our side). The "light" variant for weak GPUs: smaller
    // download + RAM, lower fidelity than fp16. The only non-default field is
    // browserUnetFile (the q4 export names its graph model_q4.onnx); encoders/
    // VAE shared from the fp16 repo (cached). WebGPU runs MatMulNBits natively
    // (ORT 1.25+).
    id: "sdxl-lightning-q4",
    repoId: "ByteDance/SDXL-Lightning",
    browserRepoId:        "d0gr/sdxl-lightning-onnx-webgpu-int4",
    browserEncoderRepoId: "d0gr/sdxl-lightning-onnx-webgpu",
    browserUnetFile: "model_q4",        // q4 graph filename stem (vs default "model")
    browserKind: "sdxl",
    browserSizeGb: 3.6,                  // 1.86 GB UNet + ~1.75 GB shared encoders/VAE
    browserScheduler: "euler",
    browserDtype: "float32",            // q4 export keeps fp32 I/O (weights only quantized)
    // Same shared SDXL VAE encoder as the fp16 variant (frozen across SDXL).
    browserVaeEncoderRepoId: "d0gr/sdxl-vae-onnx",
    browserVaeEncoderFile: "model",
    browserVaeEncoderIO: "float32",
    browserDefaultSteps: 4,             // light/fast: single-pass, no CFG
    browserDefaultGuidance: 0,
    nameLead: "SDXL Lightning",
    nameTail: "Light",
    description: "1024x1024 · same Lightning model, 4-bit UNet · smaller download + RAM, softer detail · for weak GPUs",
    kind: "sdxl",
    kindLabel: "SDXL",
    precision: "int4",
    sizeGb: 3.6,
    minVramGb: 6,
    authorShort: "ByteDance",
    avgSecondsPerFrame: null,
    avgSecondsBrowser: 14,
    gated: false,
    // 4-bit MatMulNBits runs on Chrome's WebGPU but not Firefox's: Firefox can
    // download/compile it, yet inference dies in the int4 kernel's buffer
    // read-back ("Buffer unmapped" from OrtRun). Hidden on Firefox until Gecko's
    // WebGPU matures — Firefox users get the fp16 "Quality" variant instead.
    webgpuChromeOnly: true,
  },
];

// Gated models (those requiring an access token) are never surfaced — the
// catalog is token-free. `webgpuChromeOnly` models are also dropped on Firefox,
// whose WebGPU can't run them (see the per-entry note).
const IS_FIREFOX = typeof navigator !== "undefined" && /firefox/i.test(navigator.userAgent || "");
export const MODEL_CATALOG = RAW_CATALOG.filter(
  (m) => !m.gated && !(IS_FIREFOX && m.webgpuChromeOnly),
);

// ── Adding more SDXL models (multi-model) ─────────────────────────────────────
// The SDXL text encoders + VAE are frozen (identical across every SDXL fine-tune),
// so they're SHARED from one repo and only the fine-tuned UNet differs. To add a
// model: self-export it to fp16 ONNX (optimum), re-shard the UNet to <2 GB pieces
// (tools/reshard.py), host the UNet repo, and add an entry with:
//   browserRepoId:        "<you>/<model>-unet-onnx"   // the re-sharded UNet repo
//   browserEncoderRepoId: "d0gr/sdxl-lightning-onnx-webgpu"
//                                                      // shared encoders/VAE/tokenizers
//   browserUnetIO:        "float16"   // optimum self-exports have fp16 I/O…
//   browserTimestep:      "float16"   // …and a FLOAT16 scalar timestep
//   browserKind: "sdxl", browserDefaultSteps: 4, browserDefaultGuidance: 0, …
// (The default SDXL-Lightning build is a single-repo model: encoder repo == UNet
// repo, fp32 I/O, INT64 timestep — the defaults, so it needs none of these fields.)

// SD 1.5 (nmkd) and LCM Dreamshaper (aislamov) were removed when the engine
// moved to ORT 1.26 (its async pipeline compile keeps the UNet load from
// freezing the browser). Their fp16 UNets compute LayerNorm with fp64 (DOUBLE)
// Pow nodes, for which ORT 1.26's web build has no kernel, so they can't create
// a session. No drop-in fp16 SD1.5 replacement exists (the alternatives are
// either fp64-Pow too, or fp32 with a >2 GB UNet that V8 can't allocate). The
// catalog is therefore SDXL-family only — also the higher-quality models.

// Entries dropped after testing — kept here so future visitors don't
// re-add them blind:
//
//   tlwu/stable-diffusion-v1-5-onnxruntime
//     Microsoft Olive/ORT export. Uses com.microsoft.NhwcConv contrib op
//     which neither the WebGPU nor the WASM EP in ort-web ships a kernel
//     for, so InferenceSession.create() fails at the first Conv with
//     "Failed to find kernel for com.microsoft.NhwcConv(1)".
//
//   sharpbai/stable-diffusion-v1-5-onnx-directml-fp32
//   TheyCallMeHex/LCM-Dreamshaper-V7-ONNX
//   TheyCallMeHex/Ghibli-Diffusion-ONNX
//     fp32 variants whose UNet weights ship as a single ~3.4 GB external-
//     data sidecar. Chrome's V8 refuses that as a single Uint8Array
//     allocation on most machines today (RangeError "Array buffer
//     allocation failed"), so the pipeline can't hand the blob to ORT.
//     Re-enable when V8 reliably allocates >2 GB ArrayBuffers across user
//     agents, or when ORT-web supports incremental external-data delivery.

/** CSS disc class for the coloured chip in UI. */
export function discFor(model) {
  return model.kind || "";
}

/** Does this model have an ONNX/WebGPU variant? */
export function hasBrowserVariant(model) {
  return !!model?.browserRepoId;
}

export function findModel(id) {
  return MODEL_CATALOG.find((m) => m.id === id) || null;
}

export function findByRepo(repoId) {
  return MODEL_CATALOG.find((m) => m.repoId === repoId) || null;
}
