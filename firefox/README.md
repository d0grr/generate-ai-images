# Generate AI Images — Firefox

Firefox (MV3) build of the extension. It reuses the engine, UI, locales, and assets
from [`../chrome`](../chrome) — only the manifest and the background page live here.

## How it differs from Chrome

Chrome can't run WebGPU in a service worker, so it hosts the ONNX engine in a
`chrome.offscreen` document. Firefox has no offscreen API, but its MV3 background is
a **DOM event page** that *can* run WebGPU / Workers / OPFS — so the engine runs
directly in the background page and the offscreen layer is dropped.

Shared orchestration lives in [`../chrome/background/core.js`](../chrome/background/core.js),
parameterised by a small platform adapter:

| Concern        | Chrome (`service-worker.js`)        | Firefox (`background/background.js`)         |
| -------------- | ----------------------------------- | -------------------------------------------- |
| Engine host    | `chrome.offscreen` document         | this background page (DOM)                   |
| `engineSend`   | `runtime.sendMessage` → offscreen   | `handle()` called directly                   |
| Engine progress| `offscreen:progress` message        | direct `setProgressSink` callback            |
| UI surface     | `side_panel`                        | `sidebar_action`                             |
| Big-file fetch | SW stream proxy + worker            | in-page fetch + worker (`__ENGINE_DIRECT__`) |

`globalThis.__ENGINE_DIRECT__` (set by [background/engine-direct.js](background/engine-direct.js)
before the engine module evaluates) makes the shared engine skip its `runtime.onMessage`
listener and fetch in-page instead of via the same-context (unreachable) SW proxy.

## Files here

- [manifest.json](manifest.json) — MV3: `sidebar_action`, background **page**, `gecko` settings.
- [background/background.html](background/background.html) — engine host page (mirrors Chrome's `offscreen.html`).
- [background/background.js](background/background.js) — Firefox platform adapter → `startCore(...)`.
- [background/engine-direct.js](background/engine-direct.js) — sets the in-page engine flag.

Everything else is pulled from `../chrome` at build time.

## Build

From `../chrome`:

```bash
node build.js --firefox
```

Outputs the unpacked extension to `firefox/dist/` and packages `release/firefox-v<version>.zip`.
The build overlays these `firefox/` files on the shared `chrome/` sources and excludes the
Chrome-only `background/service-worker.js` and `offscreen/offscreen.html`.

## Run / test

```bash
npx web-ext lint --source-dir dist --self-hosted    # 0 errors expected
npx web-ext run  --source-dir dist                  # launch a WebGPU-enabled Firefox
```

Or load manually: `about:debugging` → This Firefox → Load Temporary Add-on → pick
`firefox/dist/manifest.json`.

End-to-end: open the sidebar → Models tab shows the two `d0gr` SDXL models → install
(downloads to OPFS) → generate → image saves and the gallery window opens. Also exercise
upscale, face-restore, cancel-during-load, locale switch, NSFW filter.

## Status / remaining checks

- [x] Manifest ported (sidebar_action, background page, gecko id + min version).
- [x] Engine hosted in the background page; offscreen layer removed for Firefox.
- [x] Shared core + platform adapter; Chrome build unchanged.
- [x] In-page fetch path (no SW proxy) for url-info + big-file → OPFS streaming.
- [x] `--firefox` build target; `web-ext lint` passes with 0 errors.
- [x] **Runtime smoke test on Firefox 141** (WebGPU enabled): fp16 SDXL installs,
      generates, and model-switch / eject / cancel-during-load all work.
- [x] WebGPU EP is the primary (and working) path; ORT runs single-threaded WASM
      glue (`numThreads = 1`, no `proxy`), so cross-origin isolation /
      SharedArrayBuffer is not required.

## Firefox-specific gotchas (fixed)

- **int4 (q4) models are hidden on Firefox.** They download/compile but die in
  inference (`Buffer unmapped` from ORT) — Gecko's WebGPU can't run the int4
  `MatMulNBits` kernel yet. Gated via `webgpuChromeOnly` in `popup/catalog.js`.
- **HF Xet redirects + HEAD.** External-weight discovery uses a ranged GET (not
  HEAD) on the in-page path, because HF serves LFS weights via a GET-presigned
  Xet CDN redirect a HEAD can't follow.
- **CSS `zoom` quirks.** The sidebar scales the 400px design to its width with
  `zoom`; under it `position: sticky` and `backdrop-filter` corrupt Gecko
  hit-testing/compositing, so both are neutralised for Firefox in `popup.css`.
- **Sidebar width** can't be set from the manifest — it's user-controlled and
  persisted by Firefox; the UI fits itself to whatever width the sidebar opens at.

## Testing

Cross-browser UI tests (Playwright) run the popup in **real Firefox and Chromium**;
logic tests (Vitest) cover state transitions. See
[`../chrome/tests/README.md`](../chrome/tests/README.md).

## Known lint warnings (not blockers)

`UNSAFE_VAR_ASSIGNMENT` / `DANGEROUS_EVAL` come from the bundled `onnxruntime-web` and
`transformers.js` (WASM glue using the `Function` constructor + dynamic `import`). They are
inherent to those libraries and are warnings, not errors.
