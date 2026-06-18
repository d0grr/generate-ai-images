# Generate AI Images — Firefox (planned)

Firefox port of the extension. The Chrome (MV3) version lives in [`../chrome`](../chrome)
and is the source of truth for shared code (popup, offscreen pipeline, workspace, locales).

WebGPU SDXL generation runs fully client-side, so the engine itself is browser-agnostic —
the work here is adapting the Chrome MV3 extension shell to Firefox.

## TODO

### Manifest / packaging
- [ ] Port `manifest.json` to Firefox MV3 (add `browser_specific_settings.gecko.id` + `strict_min_version`).
- [ ] Replace `chrome.*` API calls with the `browser.*` (WebExtension) namespace or add a `webextension-polyfill`.
- [ ] Decide how to share code with `../chrome` (symlink, copy step in `build.js`, or a shared `src/`).

### Offscreen / background
- [ ] Firefox has no `chrome.offscreen` API — replace the offscreen document with a hidden
      extension page / tab or a background-page approach for running the ONNX pipeline.
- [ ] Verify the service worker / background script model maps to Firefox's event pages.

### WebGPU / engine
- [ ] Confirm WebGPU availability in the target Firefox version (flag vs. stable).
- [ ] Test `onnxruntime-web` WebGPU EP + WASM threads (COOP/COEP / cross-origin isolation headers).
- [ ] Verify OPFS model caching works (storage quota + persistence in Firefox).

### Models
- [ ] Confirm model downloads from Hugging Face (`d0gr/sdxl-lightning-onnx-web-fp16`,
      `d0gr/sdxl-lightning-onnx-web-small`) work under Firefox CSP / CORS.

### Build & release
- [ ] Add a Firefox build target to `build.js` (or a dedicated build script here).
- [ ] Validate with `web-ext lint` and test via `web-ext run`.
- [ ] Prepare AMO (addons.mozilla.org) submission.

### QA
- [ ] Generate on both models (fp16 + small/q4) and compare output with the Chrome build.
- [ ] Test upscale + face-restore pipelines.
- [ ] Verify all locales load (`_locales`).
