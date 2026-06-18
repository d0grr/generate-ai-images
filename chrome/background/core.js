// ─────────────────────────────────────────────────────────────
// Generate AI Images — background orchestration core (platform-agnostic).
//
// Shared by both builds: Chrome (background/service-worker.js) and Firefox
// (firefox/background/background.js). It probes the WebGPU engine, routes popup/
// sidebar actions to the engine, and persists state to chrome.storage.local.
//
// The engine itself runs in a different place per platform:
//   • Chrome  — a chrome.offscreen document (service workers can't touch WebGPU)
//   • Firefox — the background page (a DOM event page that CAN run WebGPU)
// so all engine access goes through an injected `platform` adapter:
//   platform.engineSend(msg)   → ensure the engine exists, send a command, await reply
//   platform.engineTeardown()  → free the engine (GPU/RAM)
//   platform.setupUiSurface()  → one-time UI wiring (Chrome side panel / Firefox sidebar)
// Engine→core progress is delivered to handleOffscreenProgress() (Chrome relays it
// over runtime.sendMessage; Firefox wires it as a direct progress sink).
// ─────────────────────────────────────────────────────────────

import { findModel } from "../popup/catalog.js";
const t = (key) => chrome.i18n.getMessage(key) || key;

// Injected by startCore() before any listener can fire.
let platform = null;

// ─────────────────────────────────────────────────────────────
// Default state
// ─────────────────────────────────────────────────────────────
const DEFAULT_STATE = {
  engine: "browser",
  engineInfo: {                 // reported back by the WebGPU engine
    device: "unknown",
    adapter: "",
    vramGb: 0,
  },
  browserReady: false,          // engine probed successfully

  activeModelId: null,
  browserModelPreparing: null,   // { modelId } while ONNX files download to OPFS
  pendingResultId: null,         // id of image generated while popup was closed
  models: [],
  downloads: [],
  recentImages: [],
  settings: {
    storageLimitGb: 20,
    hfToken: "",
    defaultSteps: 30,
    defaultGuidance: 7.0,
    previewDuringGeneration: true,
  },
};

// ─────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────
function registerLifecycle() {
  chrome.runtime.onInstalled.addListener(async () => {
    const existing = await chrome.storage.local.get(null);
    await chrome.storage.local.set({ ...DEFAULT_STATE, ...existing });
    await purgeStaleBrowserModels();
    probeBrowserEngine();
  });

  chrome.runtime.onStartup.addListener(async () => {
    await purgeStaleBrowserModels();
    probeBrowserEngine();
  });

  chrome.alarms.onAlarm.addListener((a) => {
    if (a.name === OFFSCREEN_TEARDOWN_ALARM) {
      // Grace window elapsed — release the engine GPU/RAM if the popup is still
      // closed and nothing is generating. A reopen would have cancelled this alarm.
      if (!popupIsOpen && !browserGenerating) platform.engineTeardown().catch(() => {});
    }
  });
}

// Remove "loading" model entries and clear browserModelPreparing — these
// are left behind when the background is killed mid-preload.
async function purgeStaleBrowserModels() {
  const { models = [], browserModelPreparing } = await chrome.storage.local.get(["models", "browserModelPreparing"]);
  const patch = {};
  const clean = models.filter((m) => m.status !== "loading");
  if (clean.length !== models.length) patch.models = clean;
  if (browserModelPreparing) patch.browserModelPreparing = null;
  if (Object.keys(patch).length > 0) await chrome.storage.local.set(patch);
}

// ─────────────────────────────────────────────────────────────
// Engine probing (WebGPU)
// ─────────────────────────────────────────────────────────────
async function probeBrowserEngine() {
  try {
    const resp = await platform.engineSend({ type: "probe" });
    if (resp?.ok && resp.info) {
      await persist({
        browserReady: !!resp.info.webgpu,
        engineInfo: {
          device: resp.info.device,
          adapter: resp.info.adapter || "",
          vendor: resp.info.vendor || "",
          vramGb: resp.info.vramGb || 0,
          reason: resp.info.reason || "",
        },
      });
      broadcastState();

      if (!resp.info.webgpu) {
        broadcast({
          type: "engine:error",
          title: t("err_no_engine"),
          message: resp.info.reason || t("err_no_engine_detail"),
        });
      }
    } else if (resp?.error) {
      broadcast({
        type: "engine:error",
        title: "WebGPU probe failed",
        message: resp.error,
      });
    }
  } catch (err) {
    console.warn("[core] browser engine probe failed", err);
    await persist({ browserReady: false });
    broadcast({
      type: "engine:error",
      title: t("err_engine_start"),
      message: String(err?.message || err),
    });
  }
}

let popupIsOpen = false;       // true while the side-panel / sidebar / popup is mounted
let browserGenerating = false; // true during runBrowserGenerate

// Engine keep-alive grace period. The engine holds the compiled WebGPU pipelines
// (~5 GB GPU/RAM); tearing it down the instant the popup closes means every reopen
// re-creates sessions and re-compiles shaders, freezing the shared GPU process for
// a few seconds. Instead we keep it warm for a grace window after close so a quick
// reopen reuses the warm pipeline. A chrome.alarm (survives background suspension,
// unlike setTimeout) fires the teardown if the popup stays closed; reopening cancels it.
const OFFSCREEN_TEARDOWN_ALARM = "offscreen-teardown";
const OFFSCREEN_GRACE_MIN = 3;
function scheduleOffscreenTeardown() {
  chrome.alarms.create(OFFSCREEN_TEARDOWN_ALARM, { delayInMinutes: OFFSCREEN_GRACE_MIN });
}
function cancelOffscreenTeardown() {
  chrome.alarms.clear(OFFSCREEN_TEARDOWN_ALARM).catch(() => {});
}

// MV3 background workers/event-pages suspend after ~30 s idle. A browser generation
// (or model preload/warmup) awaits the engine response for minutes — download + cold
// shader compile + denoise — during which the background is idle. If it suspends, the
// in-flight message channel closes. A ref-counted keepalive pings a trivial chrome API
// every 20 s (< the idle timeout) to keep the background alive across any long op.
let keepAliveCount = 0;
let keepAliveTimer = null;
function keepAlive(on) {
  keepAliveCount = Math.max(0, keepAliveCount + (on ? 1 : -1));
  if (keepAliveCount > 0 && !keepAliveTimer) {
    keepAliveTimer = setInterval(() => { chrome.runtime.getPlatformInfo().catch(() => {}); }, 20000);
  } else if (keepAliveCount === 0 && keepAliveTimer) {
    clearInterval(keepAliveTimer);
    keepAliveTimer = null;
  }
}

// ─────────────────────────────────────────────────────────────
// Connections — popup lifetime + streaming fetch proxy
// ─────────────────────────────────────────────────────────────
// Port-based streaming proxy — for fetches the Chrome offscreen document can't
// make directly (Chrome MV3 sometimes refuses CORS for offscreen even with
// host_permissions matching). Sends the response back as a stream of base64
// chunks so big ONNX models (~800 MB each) don't choke a single sendMessage.
// (Firefox fetches in-page, so it never opens these ports.)
function registerConnections() {
  chrome.runtime.onConnect.addListener((port) => {
    if (port.name === "popup-lifetime") {
      popupIsOpen = true;
      cancelOffscreenTeardown();   // reopened within the grace window — keep warm
      broadcastState();
      scheduleBrowserWarmup();
      port.onDisconnect.addListener(() => {
        popupIsOpen = false;
        // Keep the engine pipeline warm for a grace window instead of tearing
        // it down now (a quick reopen then skips the shader-compile freeze).
        if (!browserGenerating) scheduleOffscreenTeardown();
      });
      return;
    }
    if (!port.name.startsWith("fetch:")) return;
    handleStreamFetch(port);
  });
}

async function handleStreamFetch(port) {
  let aborted = false;
  port.onDisconnect.addListener(() => { aborted = true; });

  port.onMessage.addListener(async (msg) => {
    if (msg?.type !== "start") return;
    const url = msg.url;

    const rangeStart = Number(msg.rangeStart) || 0;

    const headers = {};
    if (rangeStart > 0) headers["Range"] = `bytes=${rangeStart}-`;

    console.info(`[core-stream] GET ${url}` + (rangeStart ? ` (range ${rangeStart}-)` : ""));

    try {
      const resp = await fetch(url, { credentials: "omit", headers });
      console.info(`[core-stream] response ${resp.status} for ${url}`);
      const ok = resp.ok || resp.status === 206;
      if (!ok) {
        try { port.postMessage({ type: "error", message: `HTTP ${resp.status} for ${url}` }); } catch {}
        return;
      }
      // If we asked for a Range and the server returned 200 instead of 206,
      // it ignored the request — caller should discard local bytes and start over.
      const rangeStartActual = (resp.status === 206) ? rangeStart : 0;
      const remaining = Number(resp.headers.get("content-length")) || 0;
      const total = rangeStartActual + remaining;
      const contentType = resp.headers.get("content-type") || "";
      try {
        port.postMessage({ type: "headers", total, remaining, rangeStartActual, contentType });
      } catch { return; }

      const reader = resp.body.getReader();
      while (!aborted) {
        const { done, value } = await reader.read();
        if (done) break;
        // Encode chunk as base64 — chunks from fetch readers are typically
        // 16–64 KB, well within String.fromCharCode.apply limits.
        const b64 = chunkToBase64(value);
        try {
          port.postMessage({ type: "chunk", b64, length: value.byteLength });
        } catch {
          aborted = true;
          break;
        }
      }
      if (!aborted) {
        try { port.postMessage({ type: "done" }); } catch {}
      }
    } catch (err) {
      const detail = `${err?.name || "Error"}: ${err?.message || err} (${url})`;
      console.error(`[core-stream] fetch failed — ${detail}`);
      try { port.postMessage({ type: "error", message: detail }); } catch {}
    }
  });
}

function chunkToBase64(uint8) {
  // Avoid stack overflow on rare jumbo chunks by walking in 32 KB steps.
  let bin = "";
  for (let i = 0; i < uint8.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, uint8.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

function registerMessages() {
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    // Anything targeted at the engine — ignore here (Chrome: handled by offscreen.js;
    // Firefox: engine is called directly, never via messages).
    if (msg?.target === "offscreen") return false;

    // Engine → core: progress or done (Chrome relay path).
    if (msg?.type === "offscreen:progress") {
      handleOffscreenProgress(msg.body);
      return false;
    }

    handlePopupMessage(msg).then(sendResponse).catch((err) => {
      console.error("[core] handler failed", err);
      sendResponse({ ok: false, error: String(err) });
    });
    return true;
  });
}

// Exported so the Firefox adapter can route the in-page engine's progress sink
// straight here (no runtime.sendMessage round-trip in a single-context page).
export function handleOffscreenProgress(body) {
  if (!body) return;
  // Warm-keep compile of an already-installed model (popup-open warmup). Surfaced
  // so the popup can show a "preparing" banner before the one-time GPU freeze.
  if (body.warming !== undefined) {
    broadcast({ type: "browser:model-warming", phase: body.warming === "done" ? "done" : "compiling" });
    return;
  }
  if (body.preloading) {
    broadcast({
      type: "browser:model-preloading",
      modelId: body.model_id,
      phase: body.phase || "downloading",
      file: body.file,
      loaded: body.downloaded || 0,
      total: body.total_bytes || 0,
    });
    return;
  }
  if (body.phase === "downloading" || body.phase === "compiling") {
    broadcast({
      type: "browser:model-downloading",
      phase: body.phase,
      file: body.file,
      downloaded: body.downloaded || 0,
      total: body.total_bytes || 0,
    });
    return;
  }
  const msg = {
    type: "progress:generate",
    source: "browser",
    phase: body.phase || "denoising",
  };
  if (body.step != null) {
    msg.step    = body.step;
    msg.total   = body.total || 1;
    msg.percent = body.total ? Math.round((body.step / body.total) * 100) : 0;
  }
  if (body.preview) msg.preview = body.preview;
  broadcast(msg);
}

// ─────────────────────────────────────────────────────────────
// Handle popup actions
// ─────────────────────────────────────────────────────────────
async function handlePopupMessage(msg) {
  switch (msg.type) {
    case "action:reprobe": {
      // User explicitly asked to re-check the engine (Diagnostics page).
      probeBrowserEngine();
      return { ok: true };
    }

    case "action:generate": {
      const state = await chrome.storage.local.get("models");
      const payload = msg.payload;

      // Find the catalog entry to get its browserRepoId.
      const modelEntry =
        findModel(payload.model_id) ||
        (state.models || []).find((m) => m.id === payload.model_id);
      const browserRepoId = modelEntry?.browserRepoId;
      if (!browserRepoId) {
        broadcast({
          type: "generate:error",
          message: t("err_no_browser_model"),
        });
        return { ok: false, error: "no-browser-variant" };
      }
      const browserScheduler = modelEntry?.browserScheduler || "euler";
      const browserDtype = modelEntry?.browserDtype || "float16";
      runBrowserGenerate({
        ...payload,
        browser_repo_id: browserRepoId,
        browser_scheduler: browserScheduler,
        browser_dtype: browserDtype,
        browser_kind: modelEntry?.browserKind || modelEntry?.kind || "sd15",
        // Multi-model SDXL: frozen encoders/VAE shared from one repo, fine-tuned
        // UNet from browserRepoId; per-export I/O convention.
        browser_encoder_repo_id: modelEntry?.browserEncoderRepoId || browserRepoId,
        browser_unet_io: modelEntry?.browserUnetIO || "float32",
        browser_timestep: modelEntry?.browserTimestep || "int64",
        // UNet ONNX filename stem (quantized exports use suffixed names, e.g.
        // "model_q4"); defaults to "model".
        browser_unet_file: modelEntry?.browserUnetFile || "model",
      });
      return { ok: true };
    }

    case "action:cancel-generate": {
      platform.engineSend({ type: "cancel" }).catch(() => {});
      return { ok: true };
    }

    case "action:download-model": {
      // Browser models don't need a separate download step — activating a model
      // streams its weights into OPFS on first use.
      const model = findModel(msg.modelId);
      if (model?.browserRepoId) return await useModel(msg.modelId);
      broadcast({ type: "generate:error", message: t("err_no_browser_model") });
      return { ok: false, error: "no-browser-variant" };
    }

    case "action:cancel-download":
      return await cancelBrowserPreload();

    case "action:hide-download":
      return { ok: true };

    case "action:unload-model": {
      // Free the active model from GPU/RAM: abort any in-flight warmup/generate
      // and tear down the engine (which holds the compiled pipeline, ~5 GB).
      // OPFS weights stay cached, so the next generate just recompiles.
      cancelOffscreenTeardown();
      try { await platform.engineSend({ type: "cancel" }); } catch {}
      try { await platform.engineTeardown(); } catch {}
      browserGenerating = false;
      keepAlive(false);
      broadcast({ type: "browser:model-warming", phase: "done" });   // clear loading UI
      broadcastState();
      return { ok: true };
    }

    case "action:use-model":
      return await useModel(msg.modelId);

    case "action:search-hf":
      return await searchHuggingFace(msg.query);

    case "action:request-token":
      chrome.tabs.create({ url: chrome.runtime.getURL("workspace/workspace.html#hf-token") });
      return { ok: true };

    case "action:open-image":
      chrome.tabs.create({ url: chrome.runtime.getURL(`workspace/workspace.html#image/${msg.id}`) });
      return { ok: true };

    case "action:url-info": {
      // Lightweight HEAD-equivalent — used by the Chrome offscreen pipeline to
      // discover whether `model.onnx_data` exists alongside model.onnx.
      try {
        const resp = await fetch(msg.url, { method: "HEAD", credentials: "omit" });
        return {
          ok: resp.ok || resp.status === 206,
          status: resp.status,
          size: Number(resp.headers.get("content-length")) || 0,
        };
      } catch (err) {
        return { ok: false, error: String(err?.message || err) };
      }
    }

    case "action:storage-get": {
      // Offscreen contexts sometimes can't reach chrome.storage directly
      // — proxy through the background which always has access.
      const result = await chrome.storage.local.get(msg.keys);
      return { ok: true, result };
    }

    case "action:storage-set": {
      await chrome.storage.local.set(msg.patch);
      return { ok: true };
    }

    case "action:proxy-fetch": {
      // CORS fallback for the Chrome offscreen document.
      try {
        const resp = await fetch(msg.url, { credentials: "omit" });
        if (!resp.ok) {
          return { ok: false, error: `HTTP ${resp.status} for ${msg.url}` };
        }
        if (msg.kind === "json") {
          return { ok: true, body: await resp.json() };
        }
        if (msg.kind === "text") {
          return { ok: true, body: await resp.text() };
        }
        // binary — return as base64
        const buf = await resp.arrayBuffer();
        const view = new Uint8Array(buf);
        let bin = "";
        for (let i = 0; i < view.length; i += 0x8000) {
          bin += String.fromCharCode.apply(null, view.subarray(i, i + 0x8000));
        }
        return { ok: true, body: btoa(bin), encoding: "base64" };
      } catch (err) {
        return { ok: false, error: `${err?.message || err} (${msg.url})` };
      }
    }

    case "action:clear-all-data": {
      try {
        await platform.engineSend({ type: "clear_all_opfs" });
      } catch {}
      await chrome.storage.local.clear();
      await chrome.storage.local.set(DEFAULT_STATE);
      broadcast({ type: "state:update", state: DEFAULT_STATE });
      return { ok: true };
    }

    case "action:delete-model": {
      const { modelId } = msg;
      const stored = await chrome.storage.local.get(["models", "activeModelId", "browserModelPreparing"]);
      // The model may be a stale install no longer in the catalog — fall back
      // to the stored record so it can always be removed and its files cleaned.
      const model = findModel(modelId) || (stored.models || []).find((m) => m.id === modelId);
      if (!model) return { ok: false, error: "unknown model" };

      if (model.browserRepoId) {
        // OPFS cache keys derive from the fetch URL: a custom base URL (legacy
        // local installs) or the HF repo layout.
        const prefix = model.browserBaseUrl
          ? `${model.browserBaseUrl.replace(/^https?:\/\//, "").replace(/[^a-zA-Z0-9._-]/g, "_")}_`
          : `huggingface.co_${model.browserRepoId.replace(/[^a-zA-Z0-9._-]/g, "_")}_resolve_main_`;
        try {
          await platform.engineSend({ type: "delete_browser_model", prefix });
          // Clean the onnx-size: keys here — chrome.storage isn't reachable
          // from the offscreen document on all Chrome builds.
          const all = await chrome.storage.local.get(null);
          const stale = Object.keys(all).filter(
            (k) => k.startsWith("onnx-size:") && k.slice("onnx-size:".length).startsWith(prefix)
          );
          if (stale.length) await chrome.storage.local.remove(stale);
        } catch (err) {
          console.warn("[core] OPFS delete failed:", err);
        } finally {
          platform.engineTeardown().catch(() => {});
        }
      }

      const patch = { models: (stored.models || []).filter((m) => m.id !== modelId) };
      if (stored.activeModelId === modelId) patch.activeModelId = null;
      if (stored.browserModelPreparing?.modelId === modelId) patch.browserModelPreparing = null;
      await persist(patch);
      broadcastState();
      return { ok: true };
    }

    case "action:delete-image": {
      const state = await chrome.storage.local.get("recentImages");
      const filtered = (state.recentImages || []).filter((i) => i.id !== msg.id);
      await persist({ recentImages: filtered });
      // OPFS removal is best-effort and runs in the engine
      platform.engineSend({ type: "delete_opfs", opfsKey: msg.opfsKey }).catch(() => {});
      broadcast({ type: "image:deleted", id: msg.id });
      broadcastState();
      return { ok: true };
    }
  }

  return { ok: false, error: "unknown-message-type" };
}

// ─────────────────────────────────────────────────────────────
// Browser-engine generate (runs in the engine; results flow back here)
// ─────────────────────────────────────────────────────────────
async function runBrowserGenerate(payload) {
  browserGenerating = true;
  keepAlive(true);   // keep the background alive across the multi-minute generation
  try {
    const resp = await platform.engineSend({ type: "generate", payload });
    if (!resp?.ok) {
      // User-initiated cancel is not an error — just clear the in-flight UI.
      if (resp?.cancelled || /cancel/i.test(resp?.error || "")) {
        broadcast({ type: "generate:cancelled" });
        return;
      }
      broadcast({ type: "generate:error", message: resp?.error || t("err_inference_failed") });
      return;
    }
    await completeGeneration(resp, "browser");
  } catch (err) {
    broadcast({ type: "generate:error", message: String(err?.message || err) });
  } finally {
    browserGenerating = false;
    keepAlive(false);
    // Keep the engine alive (model in RAM) while the popup is open so the next
    // generate reuses the loaded pipeline. If the popup already closed, keep it
    // warm for the grace window rather than tearing down immediately.
    if (!popupIsOpen) scheduleOffscreenTeardown();
  }
}

// Cancel an in-flight browser model preload (the "Get"/activate OPFS download).
async function cancelBrowserPreload() {
  const { browserModelPreparing } = await chrome.storage.local.get("browserModelPreparing");
  if (browserModelPreparing?.modelId) {
    platform.engineSend({ type: "cancel" }).catch(() => {});
    const { models = [] } = await chrome.storage.local.get("models");
    const clean = models.filter((m) => !(m.id === browserModelPreparing.modelId && m.status === "loading"));
    await persist({ models: clean, browserModelPreparing: null });
    broadcastState();
  }
  return { ok: true };
}

// Mark a browser-cached model as installed in state.models.  Idempotent —
// no-op if the model is already there.  Broadcasts state on change.
async function markBrowserModelInstalled(modelId) {
  const { models = [] } = await chrome.storage.local.get("models");
  if (models.some((m) => m.id === modelId && m.status === "installed")) return;
  const catalog = findModel(modelId);
  if (!catalog) return;
  const updated = [
    ...models.filter((m) => m.id !== modelId),
    { ...catalog, status: "installed" },
  ];
  await persist({ models: updated });
  broadcastState();
}

// ─────────────────────────────────────────────────────────────
// Use / switch models
// ─────────────────────────────────────────────────────────────
async function useModel(modelId) {
  await persist({ activeModelId: modelId });
  const model = findModel(modelId);
  if (model?.browserRepoId) {
    const { models = [] } = await chrome.storage.local.get("models");
    if (models.some((m) => m.id === modelId && m.status === "installed")) {
      // Already cached in OPFS — just activate.
      broadcastState();
      return { ok: true };
    }
    // Not yet downloaded — add as "loading" and start OPFS preload.
    const updated = [
      ...models.filter((m) => m.id !== modelId),
      { ...model, status: "loading" },
    ];
    await persist({ models: updated, browserModelPreparing: { modelId } });
    broadcastState();
    startBrowserModelPreload(model).catch(console.error);
    return { ok: true };
  }
  broadcastState();
  return { ok: true };
}

async function startBrowserModelPreload(model) {
  let preloadOk = false;
  keepAlive(true);   // OPFS download can take minutes — keep the background alive
  try {
    const resp = await platform.engineSend({
      type: "preload_model",
      repoId: model.browserRepoId,
      encoderRepoId: model.browserEncoderRepoId || model.browserRepoId,
      unetFile: model.browserUnetFile || "model",
      modelId: model.id,
    });
    if (resp?.ok) {
      await markBrowserModelInstalled(model.id);
      preloadOk = true;
    } else {
      const { models = [] } = await chrome.storage.local.get("models");
      const clean = models.filter((m) => !(m.id === model.id && m.status === "loading"));
      await persist({ models: clean });
    }
  } catch (err) {
    console.warn("[core] browser model preload failed", err);
    const { models = [] } = await chrome.storage.local.get("models");
    const clean = models.filter((m) => !(m.id === model.id && m.status === "loading"));
    await persist({ models: clean });
  } finally {
    keepAlive(false);
    await persist({ browserModelPreparing: null });
    // Tear the engine down now; warmup below will re-init it if popup is visible.
    platform.engineTeardown().catch(() => {});
    broadcastState();
  }
  // If the popup is open and files were downloaded, load them into ORT sessions now.
  if (preloadOk && popupIsOpen) scheduleBrowserWarmup();
}

// Load the active browser model into ORT sessions so Generate is instant.
// Skipped silently if no installed model or generate is running.
async function scheduleBrowserWarmup() {
  const { browserReady, activeModelId, models = [] } =
    await chrome.storage.local.get(["browserReady", "activeModelId", "models"]);
  if (!browserReady || !activeModelId) return;
  if (browserGenerating) return;
  const model = findModel(activeModelId);
  if (!model?.browserRepoId) return;
  if (!models.some((m) => m.id === activeModelId && m.status === "installed")) return;
  keepAlive(true);   // cold shader compile can outlast the idle timeout
  try {
    await platform.engineSend({
      type: "warmup",
      repoId: model.browserRepoId,
      schedulerType: model.browserScheduler || "euler",
      dtype: model.browserDtype || "float16",
      kind: model.browserKind || model.kind || "sd15",
      encoderRepoId: model.browserEncoderRepoId || model.browserRepoId,
      unetIO: model.browserUnetIO || "float32",
      timestep: model.browserTimestep || "int64",
      unetFile: model.browserUnetFile || "model",
    });
  } catch (err) {
    console.warn("[core] warmup failed", err);
  } finally {
    keepAlive(false);
  }
}

// ─────────────────────────────────────────────────────────────
// Generation result → storage + broadcast
// ─────────────────────────────────────────────────────────────
async function completeGeneration(msg, source) {
  const state = await chrome.storage.local.get("recentImages");
  const newImage = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    thumbB64: msg.thumb_b64 || msg.image_b64 || "",
    opfsKey:   msg.opfs_key   || "",   // browser-engine OPFS path
    seed: msg.seed,
    prompt: msg.prompt,
    negativePrompt: msg.negative_prompt || "",
    modelId: msg.model_id,
    timestamp: Date.now(),
    elapsedMs: msg.elapsed_ms,
    width: msg.width,
    height: msg.height,
    steps: msg.steps,
    guidanceScale: msg.guidance_scale,
    scheduler: msg.scheduler || msg.sampler || "",
    engine: source,
  };
  // Keep up to 200 thumbnails in chrome.storage.local (≈ 15–30 KB each as base64 at 256 px
  // → up to ≈ 6 MB). This is why the manifest declares unlimitedStorage — the default
  // chrome.storage.local quota (10 MB) can be exhausted by heavy users without it.
  const recentImages = [newImage, ...(state.recentImages || [])].slice(0, 200);
  const patch = { recentImages };
  if (!popupIsOpen) patch.pendingResultId = newImage.id;
  await persist(patch);
  broadcast({ type: "generate:done", recentImages });

  // Fallback: in case runBrowserGenerate's pre-mark missed (e.g. legacy
  // images generated before the fix landed), make sure the row exists.
  if (source === "browser" && msg.model_id) {
    await markBrowserModelInstalled(msg.model_id);
  }
}

// ─────────────────────────────────────────────────────────────
// HF search
// ─────────────────────────────────────────────────────────────
async function searchHuggingFace(query) {
  if (!query) return { ok: true, results: [] };
  try {
    const u = new URL("https://huggingface.co/api/models");
    u.searchParams.set("search", query);
    u.searchParams.set("filter", "text-to-image");
    u.searchParams.set("limit", "10");
    const res = await fetch(u, { headers: { Accept: "application/json" } });
    if (!res.ok) return { ok: false, error: `HF ${res.status}` };
    const data = await res.json();
    const results = data.map((m) => ({
      id: m.id.replace(/\//g, "--").toLowerCase(),
      repoId: m.id,
      nameLead: m.id.split("/")[1] || m.id,
      nameTail: "",
      kind: "sdxl",
      kindLabel: m.pipeline_tag || "sdxl",
      precision: "fp16",
      sizeGb: 0,
      minVramGb: 8,
      authorShort: m.id.split("/")[0] || "",
      avgSecondsPerFrame: 4.0,
      gated: m.gated || false,
    }));
    broadcast({ type: "hf:results", results });
    return { ok: true, results };
  } catch (err) {
    console.warn("[core] HF search failed", err);
    return { ok: false, error: String(err) };
  }
}

// ─────────────────────────────────────────────────────────────
// State fan-out to all open views
// ─────────────────────────────────────────────────────────────
async function persist(patch) {
  await chrome.storage.local.set(patch);
}

function broadcast(msg) {
  chrome.runtime.sendMessage(msg).catch(() => {});
}

async function broadcastState() {
  const state = await chrome.storage.local.get(null);
  broadcast({ type: "state:update", state });
}

// ─────────────────────────────────────────────────────────────
// Entry — wired by the platform-specific background once its adapter is ready.
// ─────────────────────────────────────────────────────────────
export function startCore(p) {
  platform = p;
  platform.setupUiSurface();
  registerLifecycle();
  registerConnections();
  registerMessages();
  return { handleOffscreenProgress };
}
