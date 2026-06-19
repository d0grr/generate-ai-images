// Tier-1 (logic): catalog visibility per browser.
//
// Regression: the int4 (q4) model must be hidden on Firefox (WebGPU can't run its
// MatMulNBits kernel) but visible on Chrome. Driven by navigator.userAgent at
// catalog import time, so each case re-imports with a fresh module registry.

import { afterEach, describe, expect, it, vi } from "vitest";

// Editable case list: add a { ua, label, q4Visible } row to cover a new browser.
const CASES = [
  { label: "Chrome", q4Visible: true,  ua: "Mozilla/5.0 (Windows) Chrome/120 Safari/537.36" },
  { label: "Firefox", q4Visible: false, ua: "Mozilla/5.0 (Windows; rv:141.0) Gecko/20100101 Firefox/141.0" },
  { label: "Edge",   q4Visible: true,  ua: "Mozilla/5.0 Chrome/120 Safari/537.36 Edg/120" },
];

afterEach(() => {
  vi.resetModules();
  vi.unstubAllGlobals();
});

describe("catalog: int4 (q4) model gating by browser", () => {
  for (const c of CASES) {
    it(`${c.label}: q4 ${c.q4Visible ? "visible" : "hidden"}`, async () => {
      vi.stubGlobal("navigator", { userAgent: c.ua });
      const { MODEL_CATALOG } = await import("../../popup/catalog.js");
      const q4 = MODEL_CATALOG.find((m) => m.id === "sdxl-lightning-q4");
      expect(Boolean(q4)).toBe(c.q4Visible);
      // The fp16 variant is always available.
      expect(MODEL_CATALOG.some((m) => m.id === "sdxl-lightning-4step")).toBe(true);
    });
  }
});
