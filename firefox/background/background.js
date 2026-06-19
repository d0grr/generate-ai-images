// ─────────────────────────────────────────────────────────────
// Generate AI Images — Firefox background page (platform adapter over ./core.js).
//
// Firefox MV3's background is a DOM event page that can run WebGPU, so the engine
// (offscreen.js + pipelines) runs in THIS page — no chrome.offscreen document.
// engine-direct.js set globalThis.__ENGINE_DIRECT__ before these imports evaluated,
// so the engine skips its runtime.onMessage listener and fetches in-page.
// ─────────────────────────────────────────────────────────────

import { handle, setProgressSink } from "../offscreen/offscreen.js";
import { startCore } from "./core.js";

const api = startCore({
  // The engine is in this same page — call handle() directly. (runtime.sendMessage
  // doesn't deliver to the sender's own context, so messaging wouldn't work here.)
  // Mirror the Chrome offscreen wrapper: turn aborts into {cancelled:true} and any
  // other throw into {ok:false,error} so core's flows behave identically.
  engineSend: async (msg) => {
    try {
      return await handle(msg);
    } catch (err) {
      const m = String(err?.message || err);
      if (/aborted|cancel/i.test(m)) return { ok: false, error: m, cancelled: true };
      return { ok: false, error: m };
    }
  },
  // Free the compiled pipeline (~5 GB) by disposing it in-page.
  engineTeardown: () => handle({ type: "unload" }),
  // Firefox shows a sidebar toggle button for sidebar_action automatically; also
  // let a click on the main toolbar icon toggle the sidebar.
  setupUiSurface: () => {
    chrome.action?.onClicked?.addListener(() => {
      try { chrome.sidebarAction.toggle(); } catch {}
    });
  },
});

// Deliver the in-page engine's progress straight into core (no message hop).
setProgressSink(api.handleOffscreenProgress);
