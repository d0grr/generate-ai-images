// Minimal in-memory `chrome.*` mock for Tier-1 (Node/Vitest) tests.
//
// It captures the runtime.onMessage listener so a test can drive core.js exactly
// like the popup would (dispatch an action message), and records broadcasts and
// outgoing messages so tests can assert on them. storage.local is a plain object.
//
// Nothing here is app-specific — extend the surface as new code paths need it.

export function makeChromeMock(initialStorage = {}) {
  const store = { ...initialStorage };
  const listeners = [];      // runtime.onMessage listeners (core registers one)
  const broadcasts = [];     // messages core sends out via runtime.sendMessage
  const alarms = new Set();

  const chrome = {
    runtime: {
      onMessage: { addListener: (fn) => listeners.push(fn) },
      // core broadcasts state/progress through here; capture for assertions.
      sendMessage: (msg) => { broadcasts.push(msg); return Promise.resolve(); },
      connect: () => ({ onMessage: { addListener() {} }, postMessage() {}, disconnect() {} }),
      onConnect: { addListener() {} },
      onInstalled: { addListener() {} },
      onStartup: { addListener() {} },
      onSuspend: { addListener() {} },
      getManifest: () => ({ version: "test" }),
      getURL: (p) => `/${p}`,
      lastError: null,
    },
    storage: {
      local: {
        get: (keys) => {
          if (keys == null) return Promise.resolve({ ...store });
          if (typeof keys === "string") return Promise.resolve({ [keys]: store[keys] });
          if (Array.isArray(keys)) {
            return Promise.resolve(Object.fromEntries(keys.map((k) => [k, store[k]])));
          }
          // object of defaults
          return Promise.resolve(
            Object.fromEntries(Object.keys(keys).map((k) => [k, store[k] ?? keys[k]])),
          );
        },
        set: (patch) => { Object.assign(store, patch); return Promise.resolve(); },
        remove: (k) => {
          (Array.isArray(k) ? k : [k]).forEach((key) => delete store[key]);
          return Promise.resolve();
        },
      },
    },
    alarms: {
      create: (name) => alarms.add(name),
      clear: (name) => { alarms.delete(name); return Promise.resolve(true); },
      onAlarm: { addListener() {} },
    },
    action: { onClicked: { addListener() {} } },
    sidebarAction: { toggle() {} },
    tabs: { create() {} },
    i18n: { getMessage: (k) => k },
    // chrome.offscreen is absent in this mock (Firefox-like / not needed for logic).
  };

  // Drive core's popup-message handler the way the SW does, and resolve when the
  // handler calls sendResponse (core's listener returns `true` and responds async).
  function dispatch(msg) {
    return new Promise((resolve) => {
      let settled = false;
      const done = (r) => { if (!settled) { settled = true; resolve(r); } };
      let keptOpen = false;
      for (const fn of listeners) {
        const ret = fn(msg, { id: "test" }, done);
        if (ret === true) keptOpen = true;
      }
      // Sync handler (no async response) → resolve after microtasks drain.
      if (!keptOpen) queueMicrotask(() => done(undefined));
    });
  }

  return { chrome, store, listeners, broadcasts, alarms, dispatch };
}
