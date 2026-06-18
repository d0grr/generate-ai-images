# Generate AI Images — Local Diffusion for Chrome

Offline text-to-image. A Chrome (MV3) extension that runs **Stable Diffusion XL
directly in your browser via WebGPU** — no Python, no CUDA toolkit, no ComfyUI,
no server, no cloud. Install it, it downloads the model weights once, and after
that every image is generated **on your own GPU**, fully offline.

```
┌──────────────────────────────────────────────┐
│  Chrome extension (MV3)                        │
│                                                │
│  ┌─ popup ─┐      ┌─ service worker ─┐         │
│  │  UI     │ ←→   │  message router  │         │
│  └─────────┘      └────────┬─────────┘         │
│                            ▼                    │
│                   ┌──────────────────┐          │
│                   │  offscreen doc    │          │
│                   │  WebGPU + ONNX    │          │
│                   │  Runtime Web      │          │
│                   └──────────────────┘          │
└──────────────────────────────────────────────┘
```

Inference runs in an **offscreen document** (MV3 service workers can't access
WebGPU). It's torn down after each generation to release VRAM/RAM. Weights are
cached in **OPFS**, so generation works with no network after the first download.

## Models

- **SDXL-Lightning, fp16** — step-distilled, 4–6 steps/image, ~7 GB on disk,
  needs ~8 GB VRAM.
- **4-bit "light"** — weight-only 4-bit UNet (ONNX Runtime `MatMulNBits`),
  ~1.86 GB, runs on lower-VRAM GPUs.

The text encoders + VAE are frozen and shared across SDXL fine-tunes, so each
additional model only ships its UNet.

## Requirements

- **Chrome / Edge 122+** (needs WebGPU + `Float16Array`).
- A WebGPU-capable GPU, ~8 GB RAM, ~4–7 GB free disk for cached weights.

## Layout

```
manifest.json                   # MV3 + offscreen permission + CSP
popup/
  popup.html  popup.css  popup.js  catalog.js  …
background/
  service-worker.js             # message routing + download/state mgmt
offscreen/
  offscreen.html  offscreen.js   # WebGPU inference (ONNX Runtime Web)
  sd-pipeline.js  sdxl-pipeline.js  scheduler.js  …
  vendor/                        # onnxruntime-web, transformers.js (tokenizer)
workspace/
  workspace.html  workspace.css  workspace.js   # gallery + settings
icons/
  icon.svg  icon{16,32,48,128}.png
build.js                        # minify/package build script
index.html                      # design reference (not shipped)
```

## Install & load (developer mode)

1. Clone the repo.
2. `chrome://extensions` → enable **Developer mode** → **Load unpacked** → pick
   this folder.
3. Click the toolbar icon. The first generation downloads the model weights into
   browser storage (one-time). Subsequent runs start instantly.

## Build

```bash
npm install
npm run build       # minified, packaged into dist/ (+ release/ zip)
npm run build:dev   # source maps, for DevTools profiling
npm run build:raw   # copy as-is, no minification
```

The shipped JS is minified but **not obfuscated** — it stays readable so anyone
can inspect what the extension does.

## Notes

- **Cold load:** the first generation compiles WebGPU shaders, which briefly
  freezes the browser (~10–15 s, one-time). Warm runs are smooth.
- **Quality:** Lightning is a few-step distilled model — fast, but not a 40-step
  SDXL run.
- **Output:** square 1024×1024 for now.

## Design system

Tokens live in [`popup/popup.css`](popup/popup.css). The accent color is used
only on active verbs (Generate, Get); everything else is neutral. Design
reference page: [`index.html`](index.html).

## Keyboard

- `Alt+Shift+D` — open popup
- `⌘ / Ctrl + Enter` — generate
