// Injects a window.chrome mock BEFORE popup.js runs, so the popup renders as a
// normal page (real browser engine → real CSS: zoom, @supports, sticky, blur).
// Seeded storage drives the initial render; outgoing messages are recorded on
// window.__sent. Tests can push SW broadcasts via window.__emit(msg) (the popup
// registers its onMessage listener at init). mailto: links are captured to
// window.__lastMailto instead of navigating the page away.

export async function installChromeMock(page, seed = {}) {
  await page.addInitScript((seedJson) => {
    const store = JSON.parse(seedJson);
    const listeners = [];
    window.__sent = [];
    window.__lastMailto = null;
    // Drive the popup's runtime.onMessage handler like a SW broadcast.
    window.__emit = (msg) => listeners.forEach((fn) => fn(msg, {}, () => {}));

    const pick = (keys) => {
      if (keys == null) return { ...store };
      if (typeof keys === "string") return { [keys]: store[keys] };
      if (Array.isArray(keys)) return Object.fromEntries(keys.map((k) => [k, store[k]]));
      return Object.fromEntries(Object.keys(keys).map((k) => [k, store[k] ?? keys[k]]));
    };
    window.chrome = {
      runtime: {
        onMessage: { addListener: (fn) => listeners.push(fn) },
        sendMessage: (m) => { window.__sent.push(m); return Promise.resolve(); },
        connect: () => ({ onMessage: { addListener() {} }, postMessage() {}, disconnect() {} }),
        getManifest: () => ({ version: "test" }),
        getURL: (p) => "/" + p,
        lastError: null,
      },
      storage: {
        local: {
          get: (k) => Promise.resolve(pick(k)),
          set: (patch) => { Object.assign(store, patch); return Promise.resolve(); },
          remove: (k) => { (Array.isArray(k) ? k : [k]).forEach((x) => delete store[x]); return Promise.resolve(); },
        },
      },
      i18n: { getMessage: (key) => key },
      tabs: { create() {} },
      action: { onClicked: { addListener() {} } },
      sidebarAction: { toggle() {} },
    };

    // Capture mailto: clicks (the Report button) without navigating away.
    const origClick = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () {
      if (typeof this.href === "string" && this.href.startsWith("mailto:")) {
        window.__lastMailto = this.href;
        return;
      }
      return origClick.apply(this, arguments);
    };
  }, JSON.stringify(seed));
}

// WebGPU ready, one installed model, active.
export const SEED_ACTIVE_MODEL = {
  browserReady: true,
  activeModelId: "sdxl-lightning-4step",
  engineInfo: { device: "webgpu", webgpu: true, adapter: "Test GPU", vramGb: 8 },
  models: [{ id: "sdxl-lightning-4step", status: "installed" }],
  steps: 6,
  guidance: 1.5,
};

// WebGPU ready, nothing installed/selected → first-run "select a model" state.
export const SEED_NO_MODEL = {
  browserReady: true,
  activeModelId: null,
  engineInfo: { device: "webgpu", webgpu: true, adapter: "Test GPU", vramGb: 8 },
  models: [],
};

// A model mid-download (OPFS preload in progress).
export const SEED_DOWNLOADING = {
  browserReady: true,
  activeModelId: "sdxl-lightning-4step",
  engineInfo: { device: "webgpu", webgpu: true, adapter: "Test GPU", vramGb: 8 },
  models: [{ id: "sdxl-lightning-4step", status: "loading" }],
  browserModelPreparing: { modelId: "sdxl-lightning-4step" },
};
