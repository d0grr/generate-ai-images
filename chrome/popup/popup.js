// ─────────────────────────────────────────────────────────────
// Imagine — popup controller
// ─────────────────────────────────────────────────────────────

import { MODEL_CATALOG, discFor, findModel, hasBrowserVariant } from "./catalog.js";
import { t, applyI18n, initI18n } from "./i18n.js";
import { PROMPTS } from "./prompts.js";
import { NSFW_NEGATIVE_PROMPT } from "./nsfw-filter.js";
import { QUALITY_NEGATIVE_PROMPT } from "./quality.js";

const DEFAULT_BROWSER_MODEL = "sd15-base";

// Prompt deck — Fisher–Yates shuffle; no repeats until exhausted.
let promptDeck = null;
let promptIdx = 0;
function shuffleDeck() {
  promptDeck = PROMPTS.slice();
  for (let i = promptDeck.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [promptDeck[i], promptDeck[j]] = [promptDeck[j], promptDeck[i]];
  }
  promptIdx = 0;
}
function nextExamplePrompt(avoid = "") {
  if (!promptDeck || promptIdx >= promptDeck.length) shuffleDeck();
  let p = promptDeck[promptIdx++];
  // skip if same as current (one retry)
  if (p === avoid && promptIdx < promptDeck.length) p = promptDeck[promptIdx++];
  return p;
}

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// Set to true on cancel click; prevents in-flight progress messages from
// restoring the running state before generate:cancelled arrives from SW.
let cancelling = false;

// Live progress for browser model preloading (not persisted — updated via
// browser:model-preloading broadcasts, cleared when preparing state clears).
let preloadProgress = null;
let latentPreviewB64 = null;

// Result of the most recent generation in this session only.
// Intentionally not loaded from storage — result pane starts empty on every open.
let currentResultImage = null;

// True after the user clicks "clear from memory" (eject), until the engine is
// warmed/used again. Drives immediate UI feedback (hide the "Ready" badge) so the
// action is visibly confirmed on every platform.
let engineCleared = false;

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
let state = {
  engine: "browser",
  engineInfo: { device: "unknown", adapter: "", vramGb: 0 },
  browserReady: false,
  activeModelId: null,
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
  activeTab: "generate",
  generating: null,
  orientation: "1:1",
  steps: 6,
  guidance: 1.5,
  seed: null,
  faceRestore: false,
  savedPrompt: "",
};

const ORIENTATIONS = {
  "1:1":  { label: "square",    sqClass: "",          sdxl: [1024, 1024], sd15: [512, 512],  flux: [1024, 1024] },
  "2:3":  { label: "portrait",  sqClass: "portrait",  sdxl: [832,  1216], sd15: [512, 768],  flux: [1024, 1024] },
  "3:2":  { label: "landscape", sqClass: "landscape", sdxl: [1216, 832],  sd15: [768, 512],  flux: [1024, 1024] },
  "9:16": { label: "tall",      sqClass: "portrait",  sdxl: [768,  1344], sd15: [512, 896],  flux: [1024, 1024] },
};

// ─────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────
// Firefox renders this UI in sidebar_action, which opens at its own (~320px)
// width that we can't set from the manifest and that isn't sized to content. The
// CSS scales the fixed 400px design via `html { zoom: var(--ff-zoom, .8) }`; here
// we make that scale adaptive — fit the design to the actual sidebar width (never
// upscaling past 1×) and re-fit when the user drags the sidebar wider/narrower.
// Chrome's side_panel keeps the native 400px (the @supports rule is Gecko-only).
const FF_DESIGN_W = 400;
function fitFirefoxSidebar() {
  // window.innerWidth is the real sidebar width in CSS px and is NOT affected by
  // our CSS `zoom`, so the scale is simply sidebarWidth / designWidth. No upper
  // cap: the whole UI grows proportionally as the user widens the sidebar — drag
  // it ~1.5× wider (~480px) and every block scales up 1.5× together. (Bounded at
  // 3× only to guard against an absurdly wide window.)
  document.documentElement.style.setProperty(
    "--ff-zoom",
    String(Math.min(3, window.innerWidth / FF_DESIGN_W)),
  );
}
if (navigator.userAgent.includes("Firefox")) {
  fitFirefoxSidebar();
  window.addEventListener("resize", fitFirefoxSidebar);
}

document.addEventListener("DOMContentLoaded", init);

async function init() {
  await initI18n();
  applyI18n();
  await loadStateFromStorage();

  if (state.pendingResultId) {
    const img = (state.recentImages || []).find(i => i.id === state.pendingResultId);
    if (img) currentResultImage = img;
    state.pendingResultId = null;
    chrome?.storage?.local?.remove?.("pendingResultId");
  }

  const promptEl = $("#prompt");
  if (promptEl && state.savedPrompt) {
    promptEl.value = state.savedPrompt;
    updateCharCount(state.savedPrompt);
    const clearBtn = $("#btn-clear-prompt");
    if (clearBtn) clearBtn.hidden = !state.savedPrompt;
  }

  renderAll();
  bindEvents();
  try {
    // Long-lived port — SW detects disconnect reliably even when the side
    // panel is destroyed before an async sendMessage could complete.
    chrome?.runtime?.connect?.({ name: "popup-lifetime" });
    chrome?.runtime?.onMessage?.addListener?.(onMessageFromSW);
  } catch {
    // preview mode
  }
}

async function loadStateFromStorage() {
  try {
    const stored = await chrome?.storage?.local?.get?.(null);
    if (stored) state = { ...state, ...stored };
  } catch (err) {
    console.warn("[popup] storage read failed", err);
  }

}

// ─────────────────────────────────────────────────────────────
// SW messaging
// ─────────────────────────────────────────────────────────────
function onMessageFromSW(msg) {
  if (!msg || typeof msg !== "object") return;
  switch (msg.type) {
    case "state:update": {
      const prevModelId = state.activeModelId;
      state = { ...state, ...msg.state };
      if (!state.browserModelPreparing) preloadProgress = null;
      if (state.activeModelId && state.activeModelId !== prevModelId) {
        applyModelParamDefaults(state.activeModelId);
      }
      renderAll();
      break;
    }
    case "progress:generate":
      if (cancelling) break;
      engineCleared = false;
      if (msg.preview) latentPreviewB64 = msg.preview;
      if (msg.step != null) {
        state.generating = {
          step:    msg.step,
          total:   msg.total,
          percent: msg.percent,
          phase:   msg.phase || "denoising",
        };
      }
      renderResultPane();
      renderGenerateButton();
      break;
    case "browser:model-downloading":
      if (cancelling) break;
      engineCleared = false;
      state.generating = {
        step: 0, total: 1,
        percent: msg.total ? Math.round((msg.downloaded / msg.total) * 100) : 0,
        phase: msg.phase || "downloading",
        file: msg.file,
        downloaded: msg.downloaded || 0,
        totalBytes: msg.total || 0,
      };
      renderResultPane();
      renderGenerateButton();
      renderModelBanner();
      break;
    case "progress:download":
      state.downloads = msg.downloads || state.downloads;
      renderModelBanner();
      renderInstalledList();
      renderAvailableList();
      renderDownloadCard();
      renderTabBadges();
      break;
    case "browser:model-preloading":
      engineCleared = false;
      preloadProgress = { modelId: msg.modelId, phase: msg.phase, loaded: msg.loaded, total: msg.total, file: msg.file };
      renderModelBanner();
      renderGenerateButton();
      break;
    case "browser:model-warming":
      // Warm-keep shader compile of an already-installed model (e.g. SDXL on
      // popup open). Show a banner so the one-time GPU freeze isn't a silent hang.
      state.warming = msg.phase !== "done";
      if (msg.phase === "done") cancelling = false;
      else engineCleared = false;   // warming again → engine is loading back in
      renderModelBanner();
      renderGenerateButton();
      renderResultPane();
      break;
    case "generate:done":
      cancelling = false;
      state.generating   = null;
      latentPreviewB64   = null;
      state.recentImages = msg.recentImages || state.recentImages;
      currentResultImage = (state.recentImages || [])[0] || null;
      renderGenerateButton();
      renderResultPane();
      renderRecentThumbs();
      renderFooter();
      break;
    case "generate:error":
      cancelling = false;
      state.generating = null;
      latentPreviewB64 = null;
      renderGenerateButton();
      renderResultPane();
      showError("Generation failed", msg.message || "Unknown error");
      break;
    case "generate:cancelled":
      cancelling = false;
      state.generating = null;
      latentPreviewB64 = null;
      renderGenerateButton();
      renderResultPane();
      hideError();
      break;
    case "download:error":
      showError("Download failed", msg.message || "Unknown error");
      break;
    case "engine:error":
      showError(msg.title || "Engine error", msg.message || "Unknown error");
      break;
  }
}

function sendSW(msg) {
  try {
    return chrome?.runtime?.sendMessage?.(msg)?.catch?.((err) => {
      console.warn("[popup] sendMessage failed", err);
    });
  } catch {
    return Promise.resolve();
  }
}

// ─────────────────────────────────────────────────────────────
// Event bindings
// ─────────────────────────────────────────────────────────────
function bindEvents() {
  // Tabs
  $$(".ph-tab").forEach((tab) => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));

  // Switch to Models tab from banner
  $$('[data-action="switch-models"]').forEach((el) => {
    el.addEventListener("click", (e) => { e.stopPropagation(); switchTab("models"); });
  });

  // Header icons
  $("#btn-settings").addEventListener("click", openWorkspace("#settings"));
  $("#btn-open-workspace").addEventListener("click", openWorkspace());

  // Error toast
  $("#toast-close")?.addEventListener("click", hideError);
  $("#toast-diagnostics")?.addEventListener("click", openWorkspace("#diagnostics"));

  // Freeze warning — dismiss permanently
  $("#freeze-close")?.addEventListener("click", () => {
    state.freezeWarningDismissed = true;
    const note = $("#freeze-note");
    if (note) note.hidden = true;
    try { chrome?.storage?.local?.set?.({ freezeWarningDismissed: true }); } catch {}
  });

  // Prompt
  const prompt = $("#prompt");
  prompt.addEventListener("input", onPromptChange);
  updateCharCount(prompt.value);

  $("#btn-clear-prompt")?.addEventListener("click", () => {
    prompt.value = "";
    updateCharCount("");
    $("#btn-clear-prompt").hidden = true;
    try { chrome?.storage?.local?.set?.({ savedPrompt: "" }); } catch {}
    prompt.focus();
  });

  // Surprise me — fill textarea with random prompt
  $("#btn-examples")?.addEventListener("click", () => {
    const current = prompt.value;
    const next = nextExamplePrompt(current);
    prompt.value = next;
    updateCharCount(next);
    const clearBtn = $("#btn-clear-prompt");
    if (clearBtn) clearBtn.hidden = false;
    try { chrome?.storage?.local?.set?.({ savedPrompt: next }); } catch {}
    prompt.focus();
    prompt.setSelectionRange(next.length, next.length);
  });

  // Keyboard shortcut
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      onGenerateClick();
    }
  });

  // Advanced toggle
  $("#btn-advanced")?.addEventListener("click", () => {
    const adv = $("#advanced");
    const btn = $("#btn-advanced");
    if (!adv) return;
    adv.hidden = !adv.hidden;
    btn.textContent = adv.hidden ? "Parameters ▾" : "Parameters ▴";
  });

  // Param inputs
  $("#param-steps")?.addEventListener("change", (e) => {
    const v = parseInt(e.target.value, 10);
    if (v >= 1 && v <= 150) {
      state.steps = v;
      try { chrome?.storage?.local?.set?.({ steps: v }); } catch {}
    }
  });
  $("#param-guidance")?.addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    if (v >= 1 && v <= 20) {
      state.guidance = v;
      try { chrome?.storage?.local?.set?.({ guidance: v }); } catch {}
    }
  });
  $("#param-seed")?.addEventListener("change", (e) => {
    const raw = e.target.value.trim();
    const v = raw === "" ? null : parseInt(raw, 10);
    state.seed = (v != null && !isNaN(v) && v >= 0) ? v : null;
    try { chrome?.storage?.local?.set?.({ seed: state.seed }); } catch {}
  });
  $("#param-face-restore")?.addEventListener("change", (e) => {
    state.faceRestore = !!e.target.checked;
    try { chrome?.storage?.local?.set?.({ faceRestore: state.faceRestore }); } catch {}
  });

  // Ratio chips — set output aspect (drives width/height via dimensionsFor)
  $$(".ratio-chip").forEach((chip) => {
    chip.addEventListener("click", () => setRatio(chip.dataset.ratio));
  });

  // Generate
  $("#btn-generate").addEventListener("click", onGenerateClick);

  // Cancel an in-flight generation OR a model load/compile (warmup into RAM).
  // Both share one offscreen aborter, so action:cancel-generate aborts either.
  const cancelInFlight = () => {
    cancelling = true;
    state.generating = null;
    state.warming = false;
    renderGenerateButton();
    renderResultPane();
    renderModelBanner();
    sendSW({ type: "action:cancel-generate" });
  };
  // ✕ in the result card (generation + load) and in the model bar (load into RAM).
  $("#btn-cancel")?.addEventListener("click", cancelInFlight);
  $("#mb-warm-cancel")?.addEventListener("click", cancelInFlight);

  // Cancel banner download (browser model preload)
  $("[data-action='cancel-download']")?.addEventListener("click", () => {
    sendSW({ type: "action:cancel-download" });
  });

  // Result image — lightbox on click
  $("#result-img")?.addEventListener("click", openLightboxWindow);

  // Result actions
  $("#btn-save")?.addEventListener("click", onSaveImage);
  $("#btn-copy")?.addEventListener("click", onCopyImage);
  $("#btn-retry")?.addEventListener("click", onRetry);

  // Recent strip
  $("#btn-view-all")?.addEventListener("click", openGallery);
  $("#btn-clear-history")?.addEventListener("click", () => {
    state.recentImages = [];
    try { chrome?.storage?.local?.set?.({ recentImages: [] }); } catch {}
    renderRecentThumbs();
    renderResultPane();
    renderFooter();
  });

  // Download card actions (delegation)
  $("#dl-card")?.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    const payload = { type: `action:${action}` };
    if (btn.dataset.repoId) payload.repoId = btn.dataset.repoId;
    sendSW(payload);
  });
}

// ─────────────────────────────────────────────────────────────
// Tabs
// ─────────────────────────────────────────────────────────────
function switchTab(name) {
  state.activeTab = name;
  document.body.classList.toggle("tab-models", name === "models");
  document.body.classList.toggle("tab-generate", name === "generate");
  $$(".ph-tab").forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("on", on);
    t.setAttribute("aria-selected", on);
  });
  $("#tab-generate").hidden = name !== "generate";
  $("#tab-models").hidden   = name !== "models";

  if (name === "models") {
    renderInstalledList();
    renderAvailableList();
  }
}

// ─────────────────────────────────────────────────────────────
// Prompt / chars
// ─────────────────────────────────────────────────────────────
function onPromptChange(e) {
  const val = e.target.value;
  updateCharCount(val);
  const clearBtn = $("#btn-clear-prompt");
  if (clearBtn) clearBtn.hidden = !val;
  try { chrome?.storage?.local?.set?.({ savedPrompt: val }); } catch {}
}

function updateCharCount(text) {
  const count = text.length;
  const el = $("[data-role='char-count']");
  if (!el) return;
  el.textContent = count;
  const wrap = el.closest(".char-count");
  if (wrap) wrap.classList.toggle("over", count > 350);
}

// ─────────────────────────────────────────────────────────────
// Ratio / orientation chips
// ─────────────────────────────────────────────────────────────
function setRatio(ratio) {
  if (!ORIENTATIONS[ratio]) return;
  state.orientation = ratio;
  try { chrome?.storage?.local?.set?.({ orientation: ratio }); } catch {}
  renderRatio();
}

function renderRatio() {
  $$(".ratio-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.ratio === state.orientation);
  });
}

function dimensionsFor(orientation) {
  const model = activeModelInfo();
  const kind = model?.kind || "sdxl";
  const def = ORIENTATIONS[orientation] || ORIENTATIONS["1:1"];
  return def[kind] || def.sdxl;
}

function activeModel() {
  if (!state.activeModelId) return null;
  return (state.models || []).find((m) => m.id === state.activeModelId) || null;
}

// Returns model metadata for the active model. Prefers the static catalog
// (newest info), but falls back to the spread copy persisted in state.models
// when the catalog no longer ships the entry (e.g. fp32 entries were
// dropped after the user already installed one). Without the fallback,
// a stale activeModelId would silently disable Generate and the banner.
function activeModelInfo() {
  if (!state.activeModelId) return null;
  return findModel(state.activeModelId) ||
         (state.models || []).find((m) => m.id === state.activeModelId) ||
         null;
}

// ─────────────────────────────────────────────────────────────
// Generate
// ─────────────────────────────────────────────────────────────
function onGenerateClick() {
  if (!canGenerate()) return;
  engineCleared = false;
  const promptText = $("#prompt").value;
  const userNeg = $("#negative-prompt")?.value?.trim() || "";
  // Order matters: user terms → quality terms → NSFW list. CLIP truncates the
  // negative at 77 tokens, so the quality terms must precede the long NSFW
  // list to survive the cutoff (see quality.js).
  const negativePrompt = [userNeg, QUALITY_NEGATIVE_PROMPT, NSFW_NEGATIVE_PROMPT]
    .filter(Boolean)
    .join(", ");
  const [w, h] = dimensionsFor(state.orientation);
  sendSW({
    type: "action:generate",
    payload: {
      prompt: promptText,
      negative_prompt: negativePrompt,
      model_id: state.activeModelId,
      width: w,
      height: h,
      steps: state.steps,
      guidance_scale: state.guidance,
      seed: (state.seed != null && state.seed >= 0) ? state.seed : -1,
      upscale: 2,
      face_restore: !!state.faceRestore,
    },
  });
  state.generating = { step: 0, total: state.steps, percent: 0 };
  renderGenerateButton();
  renderResultPane();
}

function onRetry() {
  const prompt = $("#prompt");
  if (!prompt?.value?.trim()) return;
  onGenerateClick();
}

function canGenerate() {
  const why = canGenerateReason();
  if (why !== "ok" && why !== _lastCanGenReason) {
    console.info(`[popup] Generate disabled: ${why}`, {
      activeModelId: state.activeModelId,
      browserReady: state.browserReady,
      generating: !!state.generating,
      modelCount: (state.models || []).length,
    });
  }
  _lastCanGenReason = why;
  return why === "ok";
}
let _lastCanGenReason = null;
function canGenerateReason() {
  if (state.generating) return "already generating";
  const model = activeModelInfo();
  if (!model) return "no active model (catalog + state.models both miss it)";
  if (!state.browserReady) return "browserReady=false — WebGPU probe failed or not run";
  if (!hasBrowserVariant(model)) return "model has no browserRepoId";
  const installed = (state.models || []).some((m) => m.id === state.activeModelId && m.status === "installed");
  if (!installed) return `model "${state.activeModelId}" not status=installed in state.models`;
  return "ok";
}

// ─────────────────────────────────────────────────────────────
// Save / Copy
// ─────────────────────────────────────────────────────────────
// Resolve the FULL-resolution PNG of a generated image as an image/png Blob.
// Images live in OPFS (read directly — same origin as offscreen). Falls back to
// the 256 px thumbnail only if no full-res source exists. (Previously Save/Copy
// used img.b64 — which is never set — so they silently fell back to the
// thumbnail, giving a 256×256 save and a useless copy.)
async function getFullImageBlob(img) {
  if (img?.opfsKey) {
    const [dirName, file] = img.opfsKey.split("/");
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(dirName);
    const handle = await dir.getFileHandle(file);
    const f = await handle.getFile();
    // OPFS files carry no MIME type; ClipboardItem needs an exact image/png.
    return new Blob([await f.arrayBuffer()], { type: "image/png" });
  }
  if (img?.thumbB64) return await (await fetch(`data:image/png;base64,${img.thumbB64}`)).blob();
  return null;
}

async function onSaveImage() {
  const img = (state.recentImages || [])[0];
  if (!img) return;
  try {
    const blob = await getFullImageBlob(img);
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `imagine-${img.id || Date.now()}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (err) {
    console.warn("[popup] save failed", err);
  }
}

async function onCopyImage() {
  const img = (state.recentImages || [])[0];
  if (!img) return;
  try {
    // Pass the Blob promise straight to ClipboardItem so the write() call stays
    // within the click's transient activation while the OPFS read resolves.
    await navigator.clipboard.write([
      new ClipboardItem({
        "image/png": getFullImageBlob(img).then((b) => b || Promise.reject(new Error("no image"))),
      }),
    ]);
  } catch (err) {
    console.warn("[popup] copy failed", err);
  }
}

// ─────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────
function renderAll() {
  renderModelBanner();
  renderGenerateButton();
  renderResultPane();
  renderRecentThumbs();
  renderFooter();
  renderStorageMeter();
  renderDownloadCard();
  renderInstalledList();
  renderAvailableList();
  renderTabBadges();
  renderRatio();
  renderParamsControls();
  renderFreezeNote();
}

// One-time freeze warning — visible until the user dismisses it (persisted).
function renderFreezeNote() {
  const note = $("#freeze-note");
  if (note) note.hidden = !!state.freezeWarningDismissed;
}

// Generation time row — shown under a finished result when timing is known.
// elapsedMs is persisted on each image record by the background, so it survives
// a popup reopen and is available for any recent image loaded back into view.
function renderGenTime() {
  const row = $("#gen-time");
  if (!row) return;
  const ms = currentResultImage?.elapsedMs;
  const show = currentResultImage && !state.generating && ms != null && ms > 0;
  if (show) {
    setText("[data-role='gen-time-val']", `${(ms / 1000).toFixed(1)}s`);
    row.hidden = false;
  } else {
    row.hidden = true;
  }
}

function applyModelParamDefaults(modelId) {
  const m = findModel(modelId);
  if (!m) return;
  let changed = false;
  if (m.browserDefaultSteps != null) { state.steps = m.browserDefaultSteps; changed = true; }
  if (m.browserDefaultGuidance != null) { state.guidance = m.browserDefaultGuidance; changed = true; }
  if (changed) {
    try { chrome?.storage?.local?.set?.({ steps: state.steps, guidance: state.guidance }); } catch {}
  }
}

function renderParamsControls() {
  const stepsEl    = $("#param-steps");
  const guidanceEl = $("#param-guidance");
  const seedEl     = $("#param-seed");
  if (stepsEl)    stepsEl.value    = state.steps ?? 20;
  if (guidanceEl) guidanceEl.value = state.guidance ?? 7.5;
  if (seedEl && document.activeElement !== seedEl) {
    seedEl.value = (state.seed != null && state.seed >= 0) ? state.seed : "";
  }
  const faceEl = $("#param-face-restore");
  if (faceEl) faceEl.checked = !!state.faceRestore;
}

function renderModelBanner() {
  const noneEl  = $("#mb-none");
  const dlEl    = $("#mb-downloading");
  const instEl  = $("#mb-installing");
  const readyEl = $("#mb-ready");
  const bannerEl = $("#model-banner");
  if (!noneEl) return;

  // The banner only carries transient status (no model / downloading / preparing).
  // Once a model is ready its name/size/badge live in the header, so the banner
  // would just be an empty strip + divider line — hide it then to reclaim space.
  // Visible by default; the ready branch below hides it.
  if (bannerEl) bannerEl.hidden = false;

  // Cancel ✕ in the loading row — shown only while loading into RAM / compiling
  // shaders (the phase that freezes), so the user can always abort the load.
  const warmCancelBtn = $("#mb-warm-cancel");
  if (warmCancelBtn) warmCancelBtn.hidden = true;

  // Use catalog info for display.
  const model = activeModelInfo();

  // Installed = model is cached in OPFS (status "installed") and WebGPU is ready.
  const isInstalled = (state.models || []).some((m) => m.id === state.activeModelId && m.status === "installed");
  const installed = model && state.browserReady && hasBrowserVariant(model) && isInstalled;

  // Warm-keep compile of an already-installed model (installed===true, so the
  // first-run "preparing" branch is skipped). Surface a distinct banner so the
  // one-time WebGPU shader-compile freeze (esp. SDXL ~14 s) isn't a silent hang.
  const warming = installed && state.warming === true;

  const phBadge = $("#ph-ready-badge");
  if (phBadge) phBadge.hidden = !installed || warming || engineCleared;

  // First-run preparation: shader/ONNX graph compilation after weights are local.
  // "downloading" phase is excluded — markBrowserModelInstalled runs before it starts,
  // so the model is already "installed" by then; showing "preparing" would be a flicker.
  const preparing = !installed && state.generating?.phase === "compiling";

  // Only track downloads for the active model — other downloads are shown in Models tab.
  const activeDownload = model && (state.downloads || []).find(
    (d) => d.repoId === model.repoId &&
      (d.status === "paused" || d.status === "active")
  );

  // Download hit 100 % but model record not yet persisted → verifying.
  const verifying = !installed && !preparing && activeDownload &&
    (activeDownload.progress || 0) >= 1;

  // ── warming (one-time shader compile of an installed model) ────────────────
  // The result-pane ring already shows the "Preparing / Compiling" state (with a
  // cancel button), so collapse the top banner to avoid duplicating it.
  if (warming) {
    noneEl.hidden = dlEl.hidden = readyEl.hidden = instEl.hidden = true;
    if (bannerEl) bannerEl.hidden = true;
    return;
  }

  // ── ready ──────────────────────────────────────────────────
  if (installed) {
    noneEl.hidden = dlEl.hidden = instEl.hidden = true;
    readyEl.hidden = false;
    if (bannerEl) bannerEl.hidden = true;   // nothing to show — collapse the strip

    const sizeStr = hasBrowserVariant(model)
      ? `${(model.browserSizeGb || 0).toFixed(1)} GB`
      : "";
    setText("[data-role='mb-model-name']", `${model.nameLead} ${model.nameTail || ""}`.trim());
    setText("[data-role='mb-model-size']", sizeStr);
    return;
  }

  // ── preparing (first-run setup) ────────────────────────────
  // Same as warming: the result-pane ring shows this phase, so don't duplicate
  // "Preparing…" in the top banner — collapse it.
  if (preparing) {
    noneEl.hidden = dlEl.hidden = readyEl.hidden = instEl.hidden = true;
    if (bannerEl) bannerEl.hidden = true;
    return;
  }

  // ── verifying (download done, installing) ──────────────────
  if (verifying) {
    noneEl.hidden = dlEl.hidden = readyEl.hidden = true;
    instEl.hidden = false;

    const name = activeDownload.repoId?.split("/")[1] || "Model";
    setText("[data-role='mb-inst-name']", name);
    setText("[data-role='mb-inst-sub']", t("mb_verifying"));
    return;
  }

  // ── downloading ────────────────────────────────────────────
  if (activeDownload) {
    noneEl.hidden = instEl.hidden = readyEl.hidden = true;
    dlEl.hidden = false;

    const pct = Math.floor((activeDownload.progress || 0) * 100);
    const name = activeDownload.repoId?.split("/")[1] || "Model";
    setText("[data-role='mb-dl-name']", name);
    setText("[data-role='mb-dl-got']", fmtBytes(activeDownload.downloadedBytes));
    setText("[data-role='mb-dl-total']", fmtBytes(activeDownload.totalBytes));
    setText("[data-role='mb-dl-pct']", `${pct}%`);
    setProp("[data-role='mb-dl-fill']", "style.width", `${pct}%`);
    return;
  }

  // ── browser model preloading (ONNX files downloading to OPFS) ─────────────
  if (state.browserModelPreparing?.modelId === state.activeModelId) {
    const p = preloadProgress;
    if (p?.phase === "compiling") {
      noneEl.hidden = dlEl.hidden = readyEl.hidden = true;
      instEl.hidden = false;
      setText("[data-role='mb-inst-name']", model ? `${model.nameLead} ${model.nameTail || ""}`.trim() : t("phase_preparing"));
      setText("[data-role='mb-inst-sub']", t("phase_compiling_first"));
    } else {
      noneEl.hidden = instEl.hidden = readyEl.hidden = true;
      dlEl.hidden = false;
      const pct = p?.total ? Math.round((p.loaded / p.total) * 100) : 0;
      setText("[data-role='mb-dl-name']", model ? `${model.nameLead} ${model.nameTail || ""}`.trim() : t("mb_downloading"));
      setText("[data-role='mb-dl-got']", fmtBytes(p?.loaded));
      setText("[data-role='mb-dl-total']", fmtBytes(p?.total));
      setText("[data-role='mb-dl-pct']", `${pct}%`);
      setProp("[data-role='mb-dl-fill']", "style.width", `${pct}%`);
    }
    return;
  }

  // ── none ───────────────────────────────────────────────────
  noneEl.hidden = false;
  dlEl.hidden = instEl.hidden = readyEl.hidden = true;
}

function renderGenerateButton() {
  const btn = $("#btn-generate");
  const lbl = $("[data-role='generate-label']");
  const hint = $("#gen-disabled-hint");
  if (!btn) return;

  const model = activeModelInfo();
  const ready = canGenerate();

  btn.disabled = !ready || !!state.generating || state.warming === true;

  // Warming = one-time shader compile of the active model; show it on the button
  // too (the model banner shows the full "preparing" state).
  if (state.warming === true && !state.generating) {
    lbl.textContent = t("compiling");
    if (hint) hint.hidden = true;
    return;
  }

  if (state.generating) {
    const g = state.generating;
    if (g.phase === "downloading") {
      lbl.textContent = t("loading_model");
    } else if (g.phase === "compiling") {
      lbl.textContent = t("compiling");
    } else if (g.phase === "upscaling") {
      lbl.textContent = t("phase_upscaling");
    } else if (g.phase === "face-restore") {
      lbl.textContent = t("phase_face_restore");
    } else {
      lbl.textContent = t("generating_pct", [String(g.percent ?? 0)]);
    }
    if (hint) hint.hidden = true;
    return;
  }

  lbl.textContent = t("btn_generate");

  if (!model) {
    if (hint) hint.hidden = false;
    return;
  }
  if (hint) hint.hidden = ready;
}

// Progress ring driver. kind: 'sparkle' | 'gear' | 'progress' | 'decoding'.
// 'progress' draws a determinate arc from pct (0–100); the others are
// indeterminate (CSS-animated) or full (decoding). r=40 → C = 2πr ≈ 251.33.
const RING_C = 2 * Math.PI * 40;
function setRing(kind, pct) {
  const run = $("#rp-running");
  const arc = $(".rp-ring-arc");
  if (!run) return;
  run.classList.remove("ring-sparkle", "ring-gear", "ring-progress", "ring-decoding");
  run.classList.add(`ring-${kind}`);
  if (!arc) return;
  // Set the gradient stroke as an SVG attribute (resolves against the document;
  // a CSS url(#id) from the external stylesheet would not).
  arc.setAttribute("stroke", kind === "decoding" ? "url(#rp-ring-grad-green)" : "url(#rp-ring-grad)");
  if (kind === "progress") {
    const frac = Math.max(0, Math.min(1, (pct ?? 0) / 100));
    arc.style.strokeDasharray = `${(RING_C * frac).toFixed(1)} ${RING_C.toFixed(1)}`;
  } else if (kind === "decoding") {
    arc.style.strokeDasharray = `${RING_C.toFixed(1)} 0`;
  } else {
    // indeterminate — clear inline override so the spinning preset dash applies
    arc.style.strokeDasharray = "";
  }
}

function renderResultPane() {
  const pane    = $("#result-pane");
  const emptyEl = $("#rp-empty");
  const runEl   = $("#rp-running");
  const resEl   = $("#rp-result");
  const acts    = $("#rp-actions");
  if (!pane) return;

  renderGenTime();

  // Model loading — first download's compile, warm-keep compile, OR the pre-step
  // (load/compile) phases of a generation, including a reload after "clear from
  // memory". Show the running card with phase text + cancel so the load is always
  // visible in the Create tab (like the first time) and the user can abort it.
  const gen = state.generating;
  const isLoading =
    (state.warming === true && !gen) ||
    (gen && (gen.step ?? 0) <= 0);
  if (isLoading) {
    pane.dataset.state = "running";
    emptyEl.hidden = resEl.hidden = true;
    runEl.hidden = false;
    if (acts) acts.hidden = true;
    const counterEl = $(".rp-counter");
    if (counterEl) counterEl.style.visibility = "hidden";
    setProp("[data-role='rp-fill']", "style.width", `${gen?.percent ?? 0}%`);
    setText(".rp-eyebrow", t("phase_preparing"));
    let loadPhase;
    if (!gen) loadPhase = t("phase_compiling_shaders");           // warm-keep compile
    else if (gen.phase === "downloading") loadPhase = t("phase_loading_model");
    else if (gen.phase === "compiling") loadPhase = t("phase_compiling_shaders");
    else loadPhase = t("phase_preparing");                        // initial / encoding
    setText("[data-role='rp-phase']", loadPhase);
    // Compiling shaders → gear; downloading/loading weights → sparkle.
    setRing((!gen || gen.phase === "compiling") ? "gear" : "sparkle");
    const previewImg = $("#rp-preview-img");
    if (previewImg) previewImg.classList.remove("visible");
    pane.classList.remove("has-preview");
    return;
  }

  if (state.generating) {
    const g = state.generating;

    pane.dataset.state = "running";
    setText(".rp-eyebrow", t("generating_eyebrow"));   // reset after a load phase
    emptyEl.hidden = resEl.hidden = true;
    runEl.hidden = false;
    if (acts) acts.hidden = true;
    const counterEl = $(".rp-counter");
    const showCounter = g.phase === "denoising" && g.step > 0;
    if (counterEl) counterEl.style.visibility = showCounter ? "visible" : "hidden";
    if (showCounter) {
      setText("[data-role='rp-step']", g.step);
      setText("[data-role='rp-total']", g.total ?? state.steps);
    }
    setProp("[data-role='rp-fill']", "style.width", `${g.percent ?? 0}%`);

    // Post-denoise finalization phases the engine emits (SDXL VAE decode, then
    // optional face-restore / upscale) — all happen after the last step, right
    // before the result appears.
    const finalizing = g.phase === "decoding" || g.phase === "upscaling" || g.phase === "face-restore";

    let phase;
    if (g.phase === "downloading") {
      phase = t("phase_loading_model");
    } else if (g.phase === "compiling") {
      phase = t("phase_compiling_shaders");
    } else if (g.phase === "decoding") {
      phase = t("phase_decoding");
    } else if (g.phase === "upscaling") {
      phase = t("phase_upscaling");
    } else if (g.phase === "face-restore") {
      phase = t("phase_face_restore");
    } else {
      phase = "";
    }
    setText("[data-role='rp-phase']", phase);

    // Ring visual per phase: weights→sparkle, shaders→gear, finalizing→decoding,
    // active denoising→determinate progress arc, otherwise→sparkle.
    if (g.phase === "downloading")            setRing("sparkle");
    else if (g.phase === "compiling")         setRing("gear");
    else if (finalizing)                      setRing("decoding");
    else if (showCounter)                     setRing("progress", g.percent);
    else                                      setRing("sparkle");

    const previewImg = $("#rp-preview-img");
    if (latentPreviewB64 && previewImg) {
      previewImg.src = `data:image/jpeg;base64,${latentPreviewB64}`;
      previewImg.classList.add("visible");
      pane.classList.add("has-preview");
    } else if (previewImg) {
      previewImg.classList.remove("visible");
      pane.classList.remove("has-preview");
    }
    return;
  }

  pane.classList.remove("has-preview");

  if (currentResultImage) {
    pane.dataset.state = "result";
    emptyEl.hidden = runEl.hidden = true;
    resEl.hidden = false;
    if (acts) acts.hidden = false;

    const img = $("#result-img");
    if (img) {
      const b64 = currentResultImage.thumbB64 || currentResultImage.b64;
      if (b64) img.src = `data:image/png;base64,${b64}`;
    }
    setText("[data-role='rp-dims']", `${currentResultImage.width || 512} × ${currentResultImage.height || 512}`);
    return;
  }

  pane.dataset.state = "empty";
  emptyEl.hidden = false;
  runEl.hidden = resEl.hidden = true;
  if (acts) acts.hidden = true;
}

function loadRecentIntoPane(img) {
  currentResultImage = img;
  renderResultPane();
  renderRecentThumbs();
  // scroll result pane into view
  $("#result-pane")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderRecentThumbs() {
  const wrap = $("#recent-thumbs");
  if (!wrap) return;
  wrap.innerHTML = "";

  const all  = state.recentImages || [];
  const has5plus = all.length > 5;
  const toShow = has5plus ? all.slice(0, 4) : all.slice(0, 5);

  toShow.forEach((img, idx) => {
    const btn = document.createElement("button");
    const isActive = currentResultImage ? img.id === currentResultImage.id : idx === 0;
    btn.className = "th" + (isActive ? " on" : "");
    btn.setAttribute("role", "listitem");
    btn.title = img.prompt || "generated image";

    const inner = document.createElement("span");
    inner.className = "th-inner";
    const b64 = img.thumbB64 || img.b64;
    if (b64) {
      inner.style.background = `center / cover url("data:image/png;base64,${b64}")`;
    }
    btn.appendChild(inner);
    btn.addEventListener("click", () => loadRecentIntoPane(img));
    wrap.appendChild(btn);
  });

  if (has5plus) {
    const tile = document.createElement("button");
    tile.className = "th th-overflow";
    tile.setAttribute("role", "listitem");
    tile.textContent = `+${all.length - 4}`;
    tile.addEventListener("click", openGallery);
    wrap.appendChild(tile);
  }

  // pad to 5 slots
  const needed = 5 - wrap.children.length;
  for (let i = 0; i < needed; i++) {
    const ph = document.createElement("div");
    ph.className = "th th-skeleton";
    ph.setAttribute("aria-hidden", "true");
    wrap.appendChild(ph);
  }
}

function renderTabBadges() {
  const active  = (state.downloads || []).filter((d) => d.status === "active");
  const dotEl   = $("[data-role='downloading-dot']");
  const badgeEl = $("[data-role='downloading-count']");
  if (!dotEl) return;
  dotEl.hidden   = active.length === 0;
  badgeEl.hidden = active.length === 0;
  if (active.length > 0) badgeEl.textContent = String(active.length);
}

// ─────────────────────────────────────────────────────────────
// Models tab rendering
// ─────────────────────────────────────────────────────────────
function renderDownloadCard() {
  const current =
    (state.downloads || []).find((d) => d.status === "active") ||
    (state.downloads || []).find((d) => d.status === "paused");
  const sec = $("#sec-downloading");
  if (!sec) return;

  if (!current) { sec.hidden = true; return; }
  sec.hidden = false;

  const paused = current.status === "paused";
  const pct = Math.floor((current.progress || 0) * 100);

  $("[data-role='dl-name']").innerHTML   = repoIdFmt(current.repoId);
  $("[data-role='dl-sub']").textContent  = paused
    ? t("dl_paused_sub")
    : `${current.totalShards || "—"} shards · SHA-256 verified`;
  $("[data-role='dl-pct']").textContent  = pct;
  $("[data-role='dl-fill']").style.width = `${pct}%`;
  $("[data-role='dl-got']").textContent  = fmtBytes(current.downloadedBytes);
  $("[data-role='dl-speed']").textContent = paused ? "paused" : fmtBps(current.speedBps);
  $("[data-role='dl-left']").textContent  = paused ? "—" : fmtSeconds(current.remainingSec);
  $("[data-role='download-rate']").textContent = paused ? "paused" : `1 · ${fmtBps(current.speedBps)}`;

  const pauseBtn = $("#dl-card [data-action='pause-download'], #dl-card [data-action='resume-download']");
  if (pauseBtn) {
    if (paused) {
      pauseBtn.textContent = t("dl_btn_resume");
      pauseBtn.dataset.action = "resume-download";
      pauseBtn.dataset.repoId = current.repoId;
    } else {
      pauseBtn.textContent = t("dl_btn_pause");
      pauseBtn.dataset.action = "pause-download";
      delete pauseBtn.dataset.repoId;
    }
  }
}

function renderInstalledList() {
  const root  = $("#installed-list");
  if (!root) return;
  const empty = root.querySelector("[data-role='installed-empty']");
  const installed = (state.models || []).filter((m) => m.status === "installed" || m.status === "loading");

  [...root.children].forEach((c) => { if (c !== empty) c.remove(); });
  empty.hidden = installed.length > 0;

  installed.forEach((m) => root.appendChild(renderModelRow(m, "installed")));
  setText(
    "[data-role='installed-count']",
    `${installed.length} · ${sumGb(installed).toFixed(1)} GB`,
  );
}

function renderAvailableList() {
  const root = $("#available-list");
  if (!root) return;
  root.innerHTML = "";

  const installedIds = new Set(
    (state.models || []).filter((m) => m.status === "installed" || m.status === "loading").map((m) => m.id)
  );
  MODEL_CATALOG
    .filter((m) => !installedIds.has(m.id))
    .forEach((m) => root.appendChild(renderModelRow(m, "available")));
}

function renderModelRow(m, kind) {
  const tpl  = $(kind === "installed" ? "#tpl-model-installed" : "#tpl-model-available");
  const node = tpl.content.firstElementChild.cloneNode(true);

  // Display fields (name/tail/description) come from the catalog so renames
  // propagate to already-installed records (which froze a copy at install time);
  // the stored record is the fallback for models no longer in the catalog.
  const cat = findModel(m.id) || {};

  node.querySelector("[data-role='disc']").className =
    `d ${discFor(m) || discFor(cat)}${m.id === state.activeModelId ? " active" : ""}`;
  node.querySelector("[data-role='name']").innerHTML =
    `${cat.nameLead || m.nameLead || m.id} <em>${cat.nameTail ?? m.nameTail ?? ""}</em>`;
  node.querySelector("[data-role='sub']").textContent = cat.description || m.description || "";

  const browserSize = m.browserSizeGb ?? cat.browserSizeGb;
  const displaySize = browserSize ?? m.sizeGb ?? cat.sizeGb;
  node.querySelector("[data-role='size']").textContent =
    displaySize != null ? `${displaySize.toFixed(1)} GB` : "—";

  const action       = node.querySelector("[data-role='action']");
  const repoId       = m.repoId || cat.repoId;
  const isDownloading = (state.downloads || []).some(
    (d) => d.repoId === repoId && d.status === "active"
  );

  if (kind === "installed") {
    if (m.status === "loading") {
      action.className = "mdl-btn downloading";
      action.textContent = "…";
      action.disabled = true;
      const subEl = node.querySelector("[data-role='sub']");
      if (subEl) subEl.textContent = t("mdl_loading_sub");
    } else if (m.id === state.activeModelId) {
      action.className = "mdl-btn in-use";
      action.textContent = t("mdl_btn_in_use");
      action.disabled = true;
      // Unload (clear from memory) — small button in the empty row-2/col-1 cell.
      const unload = document.createElement("button");
      unload.className = "mdl-btn unload";
      unload.textContent = "⏏";
      unload.title = t("mdl_unload_title");
      unload.setAttribute("aria-label", t("mdl_unload_title"));
      unload.addEventListener("click", (e) => {
        e.stopPropagation();
        sendSW({ type: "action:unload-model", modelId: m.id });
        // Immediate, platform-agnostic feedback: confirm the click and reflect
        // that the model is no longer held in memory (hide the Ready badge).
        engineCleared = true;
        unload.textContent = "✓";
        unload.disabled = true;
        const phBadge = $("#ph-ready-badge");
        if (phBadge) phBadge.hidden = true;
      });
      node.appendChild(unload);
    } else {
      action.className = "mdl-btn use";
      action.textContent = t("mdl_btn_use");
      action.addEventListener("click", (e) => {
        e.stopPropagation();
        sendSW({ type: "action:use-model", modelId: m.id });
      });
    }
  } else {
    // VRAM is advisory only — WebGPU adapter often under-reports VRAM (or
    // returns 0), and the user may still want to try. Show "Get" as normal
    // and surface the recommendation through a tooltip if the detected VRAM
    // is below the model's minimum.
    const minVram = m.minVramGb ?? cat.minVramGb ?? 0;
    const detectedVram = state.engineInfo?.vramGb || 0;
    const lowVramHint = (detectedVram > 0 && minVram > 0 && detectedVram < minVram)
      ? `Recommended ${minVram} GB VRAM · detected ${detectedVram} GB · may fail`
      : "";

    if (isDownloading) {
      action.className = "mdl-btn downloading";
      action.textContent = "…";
      action.disabled = true;
    } else if (!hasBrowserVariant(m) && !hasBrowserVariant(cat)) {
      action.className = "mdl-btn na";
      action.textContent = "—";
      action.disabled = true;
    } else {
      action.className = "mdl-btn get";
      action.textContent = t("mdl_btn_get");
      if (lowVramHint) action.title = lowVramHint;
      action.addEventListener("click", (e) => {
        e.stopPropagation();
        sendSW({ type: "action:use-model", modelId: m.id });
      });
    }
  }

  return node;
}

function renderStorageMeter() {
  const used  = sumGb((state.models || []).filter((m) => m.status === "installed"));
  const limit = state.settings.storageLimitGb || 20;
  const pct   = Math.min(100, (used / limit) * 100);
  setProp("[data-role='storage-fill']", "style.width", `${pct}%`);
  setText("[data-role='storage-used']", used.toFixed(1));
  setText("[data-role='storage-limit']", String(limit));
}

function renderFooter() {
  const dotEl  = $("[data-role='pf-dot']");
  const nameEl = $("[data-role='pf-model-name']");
  const histEl = $("[data-role='pf-history']");
  if (!dotEl) return;

  const model = activeModelInfo();
  if (model) {
    dotEl.className = `pf-dot ${discFor(model)}`;
    nameEl.textContent = `${model.nameLead} ${model.nameTail || ""}`.trim();
  } else {
    dotEl.className = "pf-dot";
    nameEl.textContent = "No model selected";
  }

  const count = (state.recentImages || []).length;
  histEl.textContent = count === 1 ? "1 in history" : `${count} in history`;
}

// ─────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────
function setText(sel, value) {
  const el = $(sel);
  if (el) el.textContent = value;
}

function setProp(sel, path, value) {
  const el = $(sel);
  if (!el) return;
  const parts = path.split(".");
  let target = el;
  for (let i = 0; i < parts.length - 1; i++) target = target[parts[i]];
  target[parts[parts.length - 1]] = value;
}

function repoIdFmt(repoId = "") {
  if (!repoId.includes("/")) return repoId;
  const [org, name] = repoId.split("/");
  return `${org} / <em>${name}</em>`;
}

function fmtBytes(n) {
  if (!n && n !== 0) return "—";
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function fmtBps(n) {
  if (!n && n !== 0) return "—";
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB/s`;
  return `${(n / 1024 ** 2).toFixed(1)} MB/s`;
}

function fmtSeconds(s) {
  if (!s && s !== 0) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m ? `${m}m ${sec}s` : `${sec}s`;
}

function fmtMB(bytes) {
  if (!bytes && bytes !== 0) return "—";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function sumGb(arr) {
  return arr.reduce((acc, m) => acc + (m.sizeGb || 0), 0);
}

function openGallery() {
  try {
    const url = chrome.runtime.getURL("workspace/workspace.html#gallery");
    chrome.tabs.create({ url });
  } catch {
    console.info("[popup] open gallery");
  }
}

function openWorkspace(hash = "") {
  return () => {
    try {
      const url = chrome.runtime.getURL("workspace/workspace.html") + hash;
      chrome.tabs.create({ url });
    } catch {
      console.info("[popup] open workspace", hash);
    }
  };
}

function showError(title, message) {
  const toast = $("#error-toast");
  if (!toast) { console.error("[popup]", title, message); return; }
  setText("[data-role='toast-title']", title);
  setText("[data-role='toast-msg']", message);
  toast.hidden = false;
  console.error("[popup]", title, "·", message);
}

function hideError() {
  const toast = $("#error-toast");
  if (toast) toast.hidden = true;
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ─────────────────────────────────────────────────────────────
// Lightbox — opens as a separate popup window
// ─────────────────────────────────────────────────────────────
async function openLightboxWindow() {
  if (!currentResultImage) return;

  await chrome.storage.session.set({
    lightbox: {
      opfsKey:  currentResultImage.opfsKey  || "",
      thumbB64: currentResultImage.thumbB64 || currentResultImage.b64 || "",
    },
  });

  const imgW  = currentResultImage.width  || 512;
  const imgH  = currentResultImage.height || 512;
  const maxW  = Math.min(1200, Math.round(screen.availWidth  * 0.8));
  const maxH  = Math.min(900,  Math.round(screen.availHeight * 0.8));
  const scale = Math.min(1, maxW / imgW, maxH / imgH);
  const winW  = Math.max(200, Math.round(imgW * scale));
  const winH  = Math.max(200, Math.round(imgH * scale));

  // Centre the lightbox over the focused browser window.
  const parent = await chrome.windows.getLastFocused({ populate: false });
  const left = Math.round((parent.left ?? 0) + ((parent.width  ?? screen.availWidth)  - winW) / 2);
  const top  = Math.round((parent.top  ?? 0) + ((parent.height ?? screen.availHeight) - winH) / 2);

  chrome.windows.create({
    url:    chrome.runtime.getURL("popup/lightbox.html"),
    type:   "popup",
    width:  winW,
    height: winH,
    left,
    top,
  });
}
