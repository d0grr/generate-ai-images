// ─────────────────────────────────────────────────────────────
// Generate AI Images — Chrome background service worker (MV3).
//
// Thin platform adapter over the shared orchestration in ./core.js. Chrome can't
// run WebGPU in a service worker, so the engine lives in a chrome.offscreen
// document; this adapter creates/tears it down and relays commands to it. The
// Firefox build provides its own adapter (firefox/background/background.js) that
// runs the same engine inside the background page instead.
// ─────────────────────────────────────────────────────────────

import { startCore } from "./core.js";

const OFFSCREEN_URL = "offscreen/offscreen.html";

// Create the offscreen document on demand (idempotent).
async function ensureOffscreen() {
  const existing = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (existing.length > 0) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["WORKERS"],
    justification:
      "Runs WebGPU text-to-image inference off the service-worker thread",
  });
}

startCore({
  // Ensure the offscreen doc exists, then deliver the command and await its reply.
  engineSend: async (msg) => {
    await ensureOffscreen();
    return chrome.runtime.sendMessage({ ...msg, target: "offscreen" });
  },
  // Closing the document frees the compiled pipeline (~5 GB GPU/RAM).
  engineTeardown: async () => {
    try { await chrome.offscreen.closeDocument(); } catch {}
  },
  // Clicking the toolbar icon opens the side panel.
  setupUiSurface: () => {
    try { chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }); } catch {}
  },
});
