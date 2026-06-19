// ─────────────────────────────────────────────────────────────
// Generate AI Images — workspace controller (minimal Stage 1)
// ─────────────────────────────────────────────────────────────
// Responsibilities (Stage 1):
//   • Route between #gallery / #settings via hash
//   • Manage settings (installed models, clear data)
//
// Stage 5 will extend this with the real Gallery grid, image detail
// view, and bulk operations.
// ─────────────────────────────────────────────────────────────

import { findModel } from "../popup/catalog.js";
import {
  applyI18n, t, initI18n, IS_DEV, AVAILABLE_LOCALES, getStoredLocale, setLocale,
} from "../popup/i18n.js";

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const ROUTES = {
  gallery:      "#view-gallery",
  settings:     "#view-settings",
  diagnostics:  "#view-diagnostics",
};

document.addEventListener("DOMContentLoaded", async () => {
  await initI18n();
  applyI18n();
  bindRouter();
  await hydrateSettings();
  bindSettings();
  await setupDevLangSwitcher();
  bindDiagnostics();
  bindGallery();
  await renderDiagnostics();
  await renderGallery();

  chrome.runtime.onMessage?.addListener?.((msg) => {
    if (msg?.type === "state:update" && msg.state) {
      hydrateSettings();
      renderDiagnostics();
      renderGallery();
    }
    if (msg?.type === "generate:done" || msg?.type === "image:deleted") {
      renderGallery();
    }
  });
});

// ─────────────────────────────────────────────────────────────
// Routing
// ─────────────────────────────────────────────────────────────
function bindRouter() {
  showRouteFromHash();
  window.addEventListener("hashchange", showRouteFromHash);
  $$(".ws-tabs a").forEach((a) => {
    a.addEventListener("click", () => {
      setTimeout(showRouteFromHash, 0);  // after hash update
    });
  });
}

function showRouteFromHash() {
  const hash = (location.hash || "#gallery").slice(1);
  // support deep links like #image/abc
  const root = hash.split("/")[0];
  const targetId = ROUTES[root] || ROUTES.gallery;

  $$(".view").forEach((v) => { v.hidden = true; });
  const view = $(targetId);
  if (view) view.hidden = false;

  $$(".ws-tabs a").forEach((a) => {
    a.classList.toggle("on", a.hash.slice(1) === root);
  });

}

// ─────────────────────────────────────────────────────────────
// Settings
// ─────────────────────────────────────────────────────────────
async function hydrateSettings() {
  await renderSettingsModels();
}

// Installed-model list with per-model delete (replaces the old popup button).
async function renderSettingsModels() {
  const list = $("#settings-models-list");
  const empty = $("#settings-models-empty");
  if (!list) return;
  const { models = [], activeModelId } = await chrome.storage.local.get(["models", "activeModelId"]);
  const installed = (models || []).filter((m) => m.status === "installed" || m.status === "loading");

  list.innerHTML = "";
  if (empty) empty.hidden = installed.length > 0;

  for (const m of installed) {
    const cat  = findModel(m.id) || {};
    const name = `${m.nameLead || cat.nameLead || m.id} ${m.nameTail || cat.nameTail || ""}`.trim();
    const sizeGb = m.browserSizeGb ?? cat.browserSizeGb ?? m.sizeGb ?? cat.sizeGb;

    const row = document.createElement("div");
    row.className = "settings-model-row";

    const info = document.createElement("div");
    info.className = "settings-model-info";
    const nm = document.createElement("span");
    nm.className = "settings-model-name";
    nm.textContent = name + (m.id === activeModelId ? "  ·  " + t("mdl_btn_in_use") : "");
    const sz = document.createElement("span");
    sz.className = "settings-model-size";
    sz.textContent = sizeGb != null ? `${sizeGb.toFixed(1)} GB` : "";
    info.append(nm, sz);

    const btn = document.createElement("button");
    btn.className = "btn-danger";
    btn.textContent = m.status === "loading" ? "…" : t("btn_delete_model");
    btn.disabled = m.status === "loading";
    btn.addEventListener("click", async () => {
      if (!confirm(t("confirm_delete_model", [name]))) return;
      btn.disabled = true;
      btn.textContent = t("clearing");
      await chrome.runtime.sendMessage({ type: "action:delete-model", modelId: m.id }).catch(() => {});
      await renderSettingsModels();
    });

    row.append(info, btn);
    list.appendChild(row);
  }
}

function bindSettings() {
  const clearBtn = $("#btn-clear-all-data");
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      if (!confirm(t("confirm_clear_data"))) return;
      clearBtn.disabled = true;
      clearBtn.textContent = t("clearing");
      await chrome.runtime.sendMessage({ type: "action:clear-all-data" }).catch(() => {});
      clearBtn.textContent = t("status_done");
      setTimeout(() => window.location.reload(), 800);
    });
  }
}

// Dev-only language override: pin the UI to a packaged locale, independent of
// the browser's language. Hidden entirely in release builds.
async function setupDevLangSwitcher() {
  const field = $("#dev-lang-field");
  const sel = $("#dev-lang");
  if (!field || !sel) return;
  if (!IS_DEV) return; // stays hidden in release
  field.hidden = false;

  let names;
  try {
    names = new Intl.DisplayNames(["en"], { type: "language" });
  } catch {
    names = null;
  }
  const label = (code) => {
    const tag = code.replace(/_/g, "-");
    const n = names?.of(tag);
    return n && n !== tag ? `${n} · ${code}` : code;
  };

  const mkOpt = (value, text) => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = text;
    return o;
  };
  sel.replaceChildren(mkOpt("__native__", t("dev_lang_browser")));
  for (const code of AVAILABLE_LOCALES) sel.append(mkOpt(code, label(code)));
  sel.value = await getStoredLocale();

  sel.addEventListener("change", async () => {
    await setLocale(sel.value);
    window.location.reload();
  });
}

function flashSaved(btn) {
  const orig = btn.textContent;
  btn.textContent = t("status_saved");
  btn.disabled = true;
  setTimeout(() => {
    btn.textContent = orig;
    btn.disabled = false;
  }, 1200);
}

// ─────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────
function bindDiagnostics() {
  $("#btn-copy-diag")?.addEventListener("click", async () => {
    const snap = await buildDiagnosticsSnapshot();
    const json = JSON.stringify(snap, null, 2);
    await navigator.clipboard?.writeText?.(json);
    const raw = $("#diag-raw");
    if (raw) { raw.hidden = false; raw.textContent = json; }
    const btn = $("#btn-copy-diag");
    const orig = btn.textContent;
    btn.textContent = t("status_copied");
    setTimeout(() => { btn.textContent = orig; }, 1200);
  });

  $("#btn-reprobe")?.addEventListener("click", async () => {
    await chrome.runtime.sendMessage({ type: "action:reprobe" }).catch(() => {});
    setTimeout(renderDiagnostics, 800);
  });
}

async function buildDiagnosticsSnapshot() {
  const state = await chrome.storage.local.get(null);
  let opfsQuota = null;
  try {
    if (navigator.storage?.estimate) {
      const est = await navigator.storage.estimate();
      opfsQuota = { quotaGB: +((est.quota || 0) / 1e9).toFixed(1), usageGB: +((est.usage || 0) / 1e9).toFixed(2) };
    }
  } catch {}

  return {
    engine: state.engine,
    engineInfo: state.engineInfo,
    browserReady: state.browserReady,
    activeModelId: state.activeModelId,
    modelsCount: (state.models || []).length,
    recentCount: (state.recentImages || []).length,
    downloadsCount: (state.downloads || []).length,
    userAgent: navigator.userAgent,
    webgpu: "gpu" in navigator,
    opfsQuota,
  };
}

async function renderDiagnostics() {
  const state = await chrome.storage.local.get(null);
  const e = state.engineInfo || {};

  // Active engine card
  setDiag("diag-engine", t("engine_browser"));
  setDiagState("diag-engine", state.browserReady ? "accent" : "err");
  setDiag("diag-pref", "browser");
  setDiag("diag-device", e.device || "—");
  setDiag("diag-adapter", e.adapter || "—");
  setDiag("diag-vendor", e.vendor || "—");
  setDiag("diag-vram", e.vramGb ? `${e.vramGb} GB (est.)` : "—");

  // Browser path
  const webgpu = "gpu" in navigator;
  setDiag("diag-browser-status", webgpu ? (state.browserReady ? "Ready" : "Probing") : "No WebGPU");
  setDiagState("diag-browser-status",
    webgpu ? (state.browserReady ? "accent" : "warn") : "err");
  setDiag("diag-webgpu", webgpu ? "available" : "unavailable", webgpu ? "ok" : "err");
  setDiag("diag-ort", "bundled · /offscreen/vendor/ort.min.js");
  setDiag("diag-tj", "bundled · /offscreen/vendor/transformers.min.js");
  try {
    if (navigator.storage?.estimate) {
      const est = await navigator.storage.estimate();
      const quotaGB = ((est.quota || 0) / 1e9).toFixed(1);
      const usageGB = ((est.usage || 0) / 1e9).toFixed(2);
      setDiag("diag-opfs", `${usageGB} / ${quotaGB} GB`);
    } else {
      setDiag("diag-opfs", "unavailable");
    }
  } catch {
    setDiag("diag-opfs", "—");
  }

  // Storage
  setDiag("diag-models-count",
    `${(state.models || []).length} model${(state.models || []).length === 1 ? "" : "s"}`);
  setDiag("diag-models",
    (state.models || []).map((m) => m.id).join(", ") || "(none)");
  setDiag("diag-recent", `${(state.recentImages || []).length} thumbnails`);
  setDiag("diag-active", state.activeModelId || t("none_selected"));
}

function setDiag(role, value, cls = "") {
  const el = document.querySelector(`[data-role='${role}']`);
  if (!el) return;
  el.textContent = value;
  if (cls) el.classList.add(cls);
  else el.classList.remove("ok", "err");
}

function setDiagState(role, state) {
  const el = document.querySelector(`[data-role='${role}']`);
  if (!el) return;
  el.dataset.state = state;
}

// ─────────────────────────────────────────────────────────────
// Gallery
// ─────────────────────────────────────────────────────────────
let galleryFilters = { model: "", engine: "", time: "all" };
let galleryImages = [];
let selectionMode = false;
const selectedIds = new Set();

function bindGallery() {
  for (const key of ["model", "engine", "time"]) {
    const el = $(`#filter-${key}`);
    if (!el) continue;
    el.addEventListener("change", () => {
      galleryFilters[key] = el.value;
      renderGalleryGrid();
    });
  }

  // Selection / bulk actions
  $("#btn-toggle-select")?.addEventListener("click", toggleSelectionMode);
  $("#btn-cancel-select")?.addEventListener("click", () => toggleSelectionMode(false));
  $("#btn-compare")?.addEventListener("click", openCompareForSelected);
  $("#btn-export-zip")?.addEventListener("click", exportSelectedAsZip);
  $("#btn-bulk-delete")?.addEventListener("click", deleteSelected);

  // Modal close triggers (delegated)
  document.addEventListener("click", (e) => {
    const action = e.target.closest?.("[data-action]")?.dataset?.action;
    if (action === "close-modal") closeModal();
    if (action === "close-compare") closeCompare();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!$("#compare-modal")?.hidden) closeCompare();
      else if (!$("#detail-modal")?.hidden) closeModal();
      else if (selectionMode) toggleSelectionMode(false);
    }
  });
}

function toggleSelectionMode(force) {
  selectionMode = (force === undefined) ? !selectionMode : force;
  selectedIds.clear();
  document.body.classList.toggle("selection-mode", selectionMode);
  const bar = $("#action-bar");
  if (bar) bar.hidden = !selectionMode;
  renderGalleryGrid();
  refreshSelectionBar();
  $("#btn-toggle-select").textContent = selectionMode ? t("btn_exit_select") : t("btn_select");
}

function refreshSelectionBar() {
  const n = selectedIds.size;
  $("[data-role='sel-count']").textContent = t("n_selected", [String(n)]);
  $("#btn-compare").disabled = n !== 2;
  $("#btn-export-zip").disabled = n === 0;
  $("#btn-bulk-delete").disabled = n === 0;
}

async function renderGallery() {
  const { recentImages = [], models = [] } = await chrome.storage.local.get(["recentImages", "models"]);
  galleryImages = recentImages;

  // Populate model filter options from catalog + installed
  const modelSelect = $("#filter-model");
  if (modelSelect) {
    const seen = new Set();
    for (const opt of [...modelSelect.querySelectorAll("option")]) {
      if (opt.value) opt.remove();
    }
    for (const img of recentImages) {
      if (!img.modelId || seen.has(img.modelId)) continue;
      seen.add(img.modelId);
      const installed = models.find((m) => m.id === img.modelId);
      const opt = document.createElement("option");
      opt.value = img.modelId;
      opt.textContent = installed?.nameLead
        ? `${installed.nameLead} ${installed.nameTail || ""}`.trim()
        : img.modelId;
      modelSelect.appendChild(opt);
    }
  }

  renderGalleryGrid();
}

function renderGalleryGrid() {
  const grid = $("#gal-grid");
  const empty = $("#gal-empty");
  const emptyFiltered = $("#gal-empty-filtered");
  if (!grid) return;

  grid.innerHTML = "";

  const visible = applyGalleryFilters(galleryImages);
  const countEl = $("[data-role='gal-count']");
  if (countEl) countEl.textContent = visible.length;

  if (galleryImages.length === 0) {
    empty.hidden = false;
    emptyFiltered.hidden = true;
    return;
  }
  if (visible.length === 0) {
    empty.hidden = true;
    emptyFiltered.hidden = false;
    return;
  }
  empty.hidden = true;
  emptyFiltered.hidden = true;

  for (const img of visible) {
    grid.appendChild(renderCard(img));
  }
}

function applyGalleryFilters(images) {
  const now = Date.now();
  const windowMs = { "24h": 24*3600e3, "7d": 7*86400e3, "30d": 30*86400e3 }[galleryFilters.time];
  return images.filter((img) => {
    if (galleryFilters.model && img.modelId !== galleryFilters.model) return false;
    if (galleryFilters.engine && img.engine !== galleryFilters.engine) return false;
    if (windowMs && img.timestamp && now - img.timestamp > windowMs) return false;
    return true;
  });
}

function renderCard(img) {
  const card = document.createElement("div");
  card.className = `gal-card engine-${img.engine || "unknown"}`;
  if (selectedIds.has(img.id)) card.classList.add("selected");
  card.tabIndex = 0;

  if (img.thumbB64) {
    const imgEl = document.createElement("img");
    imgEl.src = `data:image/png;base64,${img.thumbB64}`;
    imgEl.alt = img.prompt || "generated image";
    imgEl.loading = "lazy";
    card.appendChild(imgEl);
  }

  const checkbox = document.createElement("div");
  checkbox.className = "gal-checkbox";
  card.appendChild(checkbox);

  // Built via DOM (textContent auto-escapes) instead of innerHTML — keeps AMO's
  // no-unsanitized happy and drops the manual escapeHtml.
  const overlay = document.createElement("div");
  overlay.className = "gal-overlay";
  const promptEl = document.createElement("div");
  promptEl.className = "gal-prompt";
  promptEl.textContent = img.prompt || t("no_prompt");
  const sub = document.createElement("div");
  sub.className = "gal-sub";
  const engEl = document.createElement("span");
  engEl.className = "engine";
  engEl.textContent = img.engine || "?";
  const dimEl = document.createElement("span");
  dimEl.textContent = `${img.width || "?"}×${img.height || "?"}`;
  const seedEl = document.createElement("span");
  seedEl.textContent = `seed ${img.seed ?? "?"}`;
  sub.append(engEl, dimEl, seedEl);
  overlay.append(promptEl, sub);
  card.appendChild(overlay);

  card.addEventListener("click", () => {
    if (selectionMode) toggleSelected(img, card);
    else openModal(img);
  });
  card.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (selectionMode) toggleSelected(img, card);
      else openModal(img);
    }
  });
  return card;
}

function toggleSelected(img, card) {
  if (selectedIds.has(img.id)) {
    selectedIds.delete(img.id);
    card.classList.remove("selected");
  } else {
    selectedIds.add(img.id);
    card.classList.add("selected");
  }
  refreshSelectionBar();
}

// ─────────────────────────────────────────────────────────────
// Detail modal
// ─────────────────────────────────────────────────────────────
let currentImage = null;
let currentFullBlobUrl = null;

async function openModal(img) {
  currentImage = img;
  const modal = $("#detail-modal");
  if (!modal) return;
  modal.hidden = false;

  // Meta panel
  const catalog = img.modelId ? findModel(img.modelId) : null;
  const modelText = catalog
    ? `${catalog.nameLead} ${catalog.nameTail || ""}`.trim()
    : (img.modelId || "—");
  setText("md-kicker", `Frame · ${new Date(img.timestamp).toLocaleDateString()}`);
  setText("md-model", modelText);
  setText("md-prompt", img.prompt || t("no_prompt"));
  if (img.negativePrompt) {
    $("[data-role='md-neg-section']").hidden = false;
    setText("md-neg", img.negativePrompt);
  } else {
    $("[data-role='md-neg-section']").hidden = true;
  }
  setText("md-seed", img.seed ?? "—");
  setText("md-steps", img.steps ?? "—");
  setText("md-guidance", img.guidanceScale != null ? Number(img.guidanceScale).toFixed(1) : "—");
  setText("md-scheduler", img.scheduler ? img.scheduler.toUpperCase() : "—");
  setText("md-engine", t("engine_browser"));
  setText("md-duration", img.elapsedMs ? `${(img.elapsedMs / 1000).toFixed(2)} s` : "—");
  setText("md-date", img.timestamp ? new Date(img.timestamp).toLocaleString() : "—");
  setText("md-dimensions", `${img.width || "?"} × ${img.height || "?"}`);
  setText("md-path", img.opfsKey || t("in_memory_only"));

  // Load full image.
  await loadFullImage(img);
  bindModalActions(img);
}

function closeModal() {
  const modal = $("#detail-modal");
  if (!modal) return;
  modal.hidden = true;
  if (currentFullBlobUrl) {
    URL.revokeObjectURL(currentFullBlobUrl);
    currentFullBlobUrl = null;
  }
  currentImage = null;
}

async function loadFullImage(img) {
  const imgEl = $("[data-role='md-img']");
  const loading = $("[data-role='md-loading']");
  imgEl.hidden = true;
  loading.hidden = false;
  loading.textContent = t("loading_fullres");

  try {
    let blobOrDataUrl;

    if (img.opfsKey) {
      // Read full-resolution PNG directly from OPFS.
      const [dirName, file] = img.opfsKey.split("/");
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle(dirName);
      const handle = await dir.getFileHandle(file);
      const blob = await handle.getFile();
      blobOrDataUrl = URL.createObjectURL(blob);
    } else if (img.thumbB64) {
      blobOrDataUrl = `data:image/png;base64,${img.thumbB64}`;
    } else {
      throw new Error("no image source");
    }

    if (currentFullBlobUrl) URL.revokeObjectURL(currentFullBlobUrl);
    currentFullBlobUrl = blobOrDataUrl.startsWith("blob:") ? blobOrDataUrl : null;
    imgEl.src = blobOrDataUrl;
    imgEl.hidden = false;
    loading.hidden = true;
  } catch (err) {
    loading.textContent = `failed: ${err.message || err}`;
  }
}

function bindModalActions(img) {
  const actionMap = {
    "close-modal": closeModal,
    "copy-prompt": () => copyToClipboard(img.prompt || ""),
    "copy-seed":   () => copyToClipboard(String(img.seed ?? "")),
    "download":    () => downloadFull(img),
    "reuse":       () => reuseSeed(img),
    "delete":      () => deleteImage(img),
  };
  const card = $("#detail-modal .modal-card");
  card?.querySelectorAll("[data-action]").forEach((btn) => {
    // Clone-and-replace to drop old listeners.
    const clone = btn.cloneNode(true);
    btn.replaceWith(clone);
    const action = clone.dataset.action;
    if (actionMap[action]) clone.addEventListener("click", actionMap[action]);
  });
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    console.warn("copy failed", err);
  }
}

async function downloadFull(img) {
  try {
    let url;
    if (img.opfsKey) {
      const [dirName, file] = img.opfsKey.split("/");
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle(dirName);
      const handle = await dir.getFileHandle(file);
      const blob = await handle.getFile();
      url = URL.createObjectURL(blob);
    } else if (img.thumbB64) {
      url = `data:image/png;base64,${img.thumbB64}`;
    } else {
      throw new Error("no source");
    }

    const filename = `generate-ai-images-${img.seed ?? "frame"}-${Date.now()}.png`;
    if (chrome.downloads?.download) {
      chrome.downloads.download({ url, filename, saveAs: true });
    } else {
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
    }
  } catch (err) {
    alert(`Download failed: ${err.message || err}`);
  }
}

async function reuseSeed(img) {
  // Ask the SW to re-run a generation with the same params.
  await chrome.runtime.sendMessage({
    type: "action:generate",
    payload: {
      prompt: img.prompt,
      negative_prompt: img.negativePrompt || "",
      model_id: img.modelId,
      width: img.width,
      height: img.height,
      steps: img.steps || 20,
      guidance_scale: img.guidanceScale != null ? img.guidanceScale : 7.0,
      seed: img.seed,
    },
  });
  closeModal();
  // Flash subtle hint that it's running
  showToast(t("reuse_seed_toast"));
}

async function deleteImage(img) {
  if (!confirm(t("confirm_delete_image"))) return;
  await chrome.runtime.sendMessage({
    type: "action:delete-image",
    id: img.id,
    path: img.opfsKey || "",
    opfsKey: img.opfsKey || "",
  });
  closeModal();
}

// ─────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────
function setText(role, text) {
  const el = document.querySelector(`[data-role='${role}']`);
  if (el) el.textContent = text;
}

// ─────────────────────────────────────────────────────────────
// Image bytes — full-resolution PNG from OPFS (thumb fallback)
// ─────────────────────────────────────────────────────────────
async function getFullImageBlob(img) {
  if (img.opfsKey) {
    const [dirName, file] = img.opfsKey.split("/");
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(dirName);
    const handle = await dir.getFileHandle(file);
    return await handle.getFile();
  }
  if (img.thumbB64) return base64ToBlob(img.thumbB64, "image/png");
  throw new Error("no source");
}

function base64ToBlob(b64, type) {
  const bin = atob(b64);
  const len = bin.length;
  const buf = new Uint8Array(len);
  for (let i = 0; i < len; i++) buf[i] = bin.charCodeAt(i);
  return new Blob([buf], { type });
}

function imgFilename(img) {
  const date = new Date(img.timestamp || Date.now())
    .toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const seed = img.seed ?? "noseed";
  return `generate-ai-images-${date}-${seed}.png`;
}

// ─────────────────────────────────────────────────────────────
// Compare modal — drag-to-reveal slider
// ─────────────────────────────────────────────────────────────
let compareCleanup = null;

async function openCompareForSelected() {
  if (selectedIds.size !== 2) return;
  const ids = [...selectedIds];
  const a = galleryImages.find((i) => i.id === ids[0]);
  const b = galleryImages.find((i) => i.id === ids[1]);
  if (!a || !b) return;
  await openCompare(a, b);
}

async function openCompare(a, b) {
  const modal = $("#compare-modal");
  if (!modal) return;
  modal.hidden = false;

  // Side panel info
  setText("cmp-prompt-a", a.prompt || "—");
  setText("cmp-prompt-b", b.prompt || "—");
  setText("cmp-meta-a", `seed ${a.seed ?? "?"} · ${a.engine || "?"} · ${a.width}×${a.height}`);
  setText("cmp-meta-b", `seed ${b.seed ?? "?"} · ${b.engine || "?"} · ${b.width}×${b.height}`);

  // Show thumbs immediately (instant feedback), then upgrade to full-res.
  const imgA = $("[data-role='cmp-img-a']");
  const imgB = $("[data-role='cmp-img-b']");
  imgA.src = a.thumbB64 ? `data:image/png;base64,${a.thumbB64}` : "";
  imgB.src = b.thumbB64 ? `data:image/png;base64,${b.thumbB64}` : "";

  bindCompareSlider();

  try {
    const [blobA, blobB] = await Promise.all([getFullImageBlob(a), getFullImageBlob(b)]);
    const urlA = URL.createObjectURL(blobA);
    const urlB = URL.createObjectURL(blobB);
    imgA.src = urlA; imgB.src = urlB;
    if (compareCleanup) compareCleanup();
    compareCleanup = () => {
      URL.revokeObjectURL(urlA);
      URL.revokeObjectURL(urlB);
    };
  } catch (err) {
    console.warn("[compare] full-res load failed", err);
  }
}

function bindCompareSlider() {
  const stage = $("#compare-stage");
  const clip = $("[data-role='cmp-clip']");
  const handle = $("[data-role='cmp-handle']");
  if (!stage || !clip || !handle) return;

  let pct = 50;
  const apply = () => {
    clip.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    handle.style.left = `${pct}%`;
  };
  apply();

  const setFromEvent = (e) => {
    const rect = stage.getBoundingClientRect();
    const x = ("touches" in e ? e.touches[0].clientX : e.clientX) - rect.left;
    pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    apply();
  };

  let dragging = false;
  const onDown = (e) => { dragging = true; setFromEvent(e); e.preventDefault(); };
  const onMove = (e) => { if (dragging) setFromEvent(e); };
  const onUp = () => { dragging = false; };

  stage.onmousedown = onDown;
  stage.onmousemove = onMove;
  document.addEventListener("mouseup", onUp);
  stage.ontouchstart = onDown;
  stage.ontouchmove = onMove;
  document.addEventListener("touchend", onUp);
}

function closeCompare() {
  const modal = $("#compare-modal");
  if (!modal) return;
  modal.hidden = true;
  if (compareCleanup) { compareCleanup(); compareCleanup = null; }
}

// ─────────────────────────────────────────────────────────────
// Bulk export — ZIP via JSZip (vendored)
// ─────────────────────────────────────────────────────────────
async function exportSelectedAsZip() {
  if (selectedIds.size === 0) return;
  if (typeof JSZip === "undefined") {
    showToast("JSZip not loaded — check workspace/vendor/");
    return;
  }
  showToast(t("packing_images", [String(selectedIds.size)]));

  const zip = new JSZip();
  const ids = [...selectedIds];

  for (const id of ids) {
    const img = galleryImages.find((i) => i.id === id);
    if (!img) continue;
    try {
      const blob = await getFullImageBlob(img);
      zip.file(imgFilename(img), blob);

      // Sidecar: write the metadata we have in storage as JSON next to the PNG.
      const meta = {
        prompt: img.prompt,
        negative_prompt: img.negativePrompt || "",
        seed: img.seed,
        steps: img.steps,
        guidance_scale: img.guidanceScale,
        scheduler: img.scheduler || "",
        model_id: img.modelId,
        engine: img.engine,
        width: img.width,
        height: img.height,
        elapsed_ms: img.elapsedMs,
        timestamp: img.timestamp,
      };
      zip.file(imgFilename(img).replace(/\.png$/, ".json"),
               JSON.stringify(meta, null, 2));
    } catch (err) {
      console.warn(`[export] skipping ${id}:`, err);
    }
  }

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const filename = `generate-ai-images-export-${stamp}.zip`;
  if (chrome.downloads?.download) {
    chrome.downloads.download({ url, filename, saveAs: true });
  } else {
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
  }
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  showToast(t("exported_images", [String(selectedIds.size)]));
}

async function deleteSelected() {
  const n = selectedIds.size;
  if (n === 0) return;
  if (!confirm(t("confirm_bulk_delete", [String(n), n === 1 ? t("image_noun") : t("images_noun")]))) return;
  for (const id of [...selectedIds]) {
    const img = galleryImages.find((i) => i.id === id);
    if (!img) continue;
    await chrome.runtime.sendMessage({
      type: "action:delete-image",
      id: img.id,
      path: img.opfsKey || "",
      opfsKey: img.opfsKey || "",
    });
  }
  toggleSelectionMode(false);
}

function showToast(text) {
  // Tiny transient — reuse modal close animation pattern.
  const t = document.createElement("div");
  t.textContent = text;
  t.style.cssText = `
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: var(--ink); color: var(--bg);
    padding: 10px 18px; border-radius: 10px;
    font-size: 13px; font-weight: 600;
    box-shadow: 0 8px 24px -8px rgba(0,0,0,0.3);
    z-index: 200; opacity: 0; transition: opacity 0.2s;
  `;
  document.body.appendChild(t);
  requestAnimationFrame(() => { t.style.opacity = "1"; });
  setTimeout(() => {
    t.style.opacity = "0";
    setTimeout(() => t.remove(), 250);
  }, 2200);
}
