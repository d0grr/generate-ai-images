// Tier-1 (logic): model state transitions in core.js — switch, cancel download,
// delete. Each starts a fresh core with a mocked chrome.* + platform.

import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { makeChromeMock } from "../mocks/chrome.js";

const A = "sdxl-lightning-4step";   // fp16 (always in catalog)
const B = "sdxl-lightning-q4";      // int4 (in catalog under a Chrome UA)

let mock, platform;

async function bootCore(storage) {
  vi.stubGlobal("navigator", { userAgent: "Mozilla/5.0 Chrome/120 Safari/537.36" });
  mock = makeChromeMock(storage);
  vi.stubGlobal("chrome", mock.chrome);
  platform = {
    engineSend: vi.fn(async () => ({ ok: true })),
    engineTeardown: vi.fn(async () => {}),
    setupUiSurface: vi.fn(),
  };
  const { startCore } = await import("../../background/core.js");
  startCore(platform);
}

afterEach(() => { vi.resetModules(); vi.unstubAllGlobals(); });

describe("model switching", () => {
  it("switching to a different installed model updates active + tears engine down", async () => {
    await bootCore({
      activeModelId: A,
      models: [{ id: A, status: "installed" }, { id: B, status: "installed" }],
    });
    await mock.dispatch({ type: "action:use-model", modelId: B });
    expect(mock.store.activeModelId).toBe(B);
    expect(platform.engineTeardown).toHaveBeenCalled();   // defense-in-depth on switch
  });

  it("re-selecting the SAME model does not tear the engine down", async () => {
    await bootCore({
      activeModelId: A,
      models: [{ id: A, status: "installed" }],
    });
    await mock.dispatch({ type: "action:use-model", modelId: A });
    expect(mock.store.activeModelId).toBe(A);
    expect(platform.engineTeardown).not.toHaveBeenCalled();
  });
});

describe("cancel download", () => {
  it("clears the preparing model and aborts the engine", async () => {
    await bootCore({
      activeModelId: A,
      browserModelPreparing: { modelId: A },
      models: [{ id: A, status: "loading" }],
    });
    await mock.dispatch({ type: "action:cancel-download" });
    expect(mock.store.browserModelPreparing).toBe(null);
    expect(mock.store.models.some((m) => m.id === A && m.status === "loading")).toBe(false);
    expect(platform.engineSend).toHaveBeenCalledWith(expect.objectContaining({ type: "cancel" }));
  });
});

describe("delete model", () => {
  it("removes it and deselects when it was active", async () => {
    await bootCore({
      activeModelId: A,
      models: [{ id: A, status: "installed" }],
    });
    await mock.dispatch({ type: "action:delete-model", modelId: A });
    expect(mock.store.models.some((m) => m.id === A)).toBe(false);
    expect(mock.store.activeModelId).toBe(null);
  });
});
