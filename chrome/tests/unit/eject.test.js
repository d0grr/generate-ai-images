// Tier-1 (logic): core.js handling of action:unload-model (the "eject" button).
//
// Regression: eject must fully DESELECT the model — clear activeModelId so the UI
// returns to the first-run "select a model" state — not just free GPU memory.

import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { makeChromeMock } from "../mocks/chrome.js";

let mock;

beforeEach(() => {
  // catalog.js reads navigator.userAgent at import; pretend Chrome here.
  // (navigator is a read-only global in Node — stub it.)
  vi.stubGlobal("navigator", { userAgent: "Mozilla/5.0 Chrome/120 Safari/537.36" });
  mock = makeChromeMock({
    activeModelId: "sdxl-lightning-4step",
    models: [{ id: "sdxl-lightning-4step", status: "installed" }],
  });
  vi.stubGlobal("chrome", mock.chrome);
});

afterEach(() => {
  vi.resetModules();
  vi.unstubAllGlobals();
});

describe("eject (action:unload-model)", () => {
  it("clears the active model and tears the engine down", async () => {
    const { startCore } = await import("../../background/core.js");
    const platform = {
      engineSend: vi.fn(async () => ({ ok: true })),
      engineTeardown: vi.fn(async () => {}),
      setupUiSurface: vi.fn(),
    };
    startCore(platform);

    await mock.dispatch({ type: "action:unload-model", modelId: "sdxl-lightning-4step" });

    expect(mock.store.activeModelId).toBe(null);          // deselected
    expect(platform.engineTeardown).toHaveBeenCalled();    // freed from memory
    // Model stays installed (eject ≠ delete).
    expect(mock.store.models.find((m) => m.id === "sdxl-lightning-4step").status)
      .toBe("installed");
  });
});
