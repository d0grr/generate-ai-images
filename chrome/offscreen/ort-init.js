// Loads ORT 1.26's native WebGPU EP bundle and exposes it as globalThis.ort,
// the way the old UMD ort.min.js did — so sd-pipeline's loadOrt() (and the
// other pipelines) keep reading `globalThis.ort` unchanged.
//
// Must be a separate module file (not an inline <script>): MV3 CSP
// `script-src 'self'` blocks inline scripts. Loaded ahead of offscreen.js in
// offscreen.html; module scripts execute in document order, and loadOrt() runs
// later still (on warmup/generate), so globalThis.ort is always set in time.
import * as ortNS from "./vendor/ort.webgpu.bundle.min.mjs";

globalThis.ort = ortNS.default ?? ortNS;
