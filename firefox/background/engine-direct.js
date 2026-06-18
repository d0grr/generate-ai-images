// Firefox runs the engine inside this background page (not a separate Chrome
// offscreen document). This flag — read by offscreen.js and sd-pipeline.js when
// their modules first evaluate — makes the engine skip its runtime.onMessage
// listener and fetch in-page instead of via the (same-context, unreachable) SW
// proxy. It must run BEFORE background.js's imports evaluate, so it's a separate
// module script listed first in background.html (CSP forbids an inline script).
globalThis.__ENGINE_DIRECT__ = true;
