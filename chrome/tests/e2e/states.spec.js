// Tier-2 (cross-browser UI): state transitions in the popup, on real Chromium
// AND Firefox. Each test seeds storage, opens popup.html, and (where needed)
// pushes SW broadcasts via window.__emit.

import { test, expect } from "@playwright/test";
import {
  installChromeMock, SEED_ACTIVE_MODEL, SEED_NO_MODEL, SEED_DOWNLOADING,
} from "./chrome-mock.js";

const POPUP = "/popup/popup.html";

test("tab switch: Generate ↔ Models toggles panes and body class", async ({ page }) => {
  await installChromeMock(page, SEED_ACTIVE_MODEL);
  await page.goto(POPUP);

  await page.locator(".ph-tab[data-tab='models']").click();
  await expect(page.locator("#tab-models")).toBeVisible();
  await expect(page.locator("body")).toHaveClass(/tab-models/);

  await page.locator(".ph-tab[data-tab='generate']").click();
  await expect(page.locator("#tab-generate")).toBeVisible();
  await expect(page.locator("body")).toHaveClass(/tab-generate/);
});

test("no model selected: header default + Generate disabled", async ({ page }) => {
  await installChromeMock(page, SEED_NO_MODEL);
  await page.goto(POPUP);

  await expect(page.locator("[data-role='mb-model-name']")).toHaveText("Generate AI Images");
  await expect(page.locator("#btn-generate")).toBeDisabled();
});

test("error broadcast shows toast; Report opens a prefilled mailto", async ({ page }) => {
  await installChromeMock(page, SEED_ACTIVE_MODEL);
  await page.goto(POPUP);

  await page.evaluate(() => window.__emit({ type: "generate:error", message: "Buffer unmapped" }));

  await expect(page.locator("#error-toast")).toBeVisible();
  await expect(page.locator("[data-role='toast-msg']")).toContainText("Buffer unmapped");

  await page.locator("#toast-report").click();
  const mailto = await page.evaluate(() => window.__lastMailto);
  expect(mailto).toContain("mailto:support@mindix.space");
  expect(decodeURIComponent(mailto)).toContain("Buffer unmapped");
  expect(decodeURIComponent(mailto)).toContain("WebGPU");   // system info block
});

test("downloading model shows a progress bar + cancel in Models", async ({ page }) => {
  await installChromeMock(page, SEED_DOWNLOADING);
  await page.goto(POPUP);
  await page.locator(".ph-tab[data-tab='models']").click();

  // Push preload progress like the SW would.
  await page.evaluate(() => window.__emit({
    type: "browser:model-preloading",
    modelId: "sdxl-lightning-4step", phase: "downloading", loaded: 50, total: 100,
  }));

  await expect(page.locator(".mdl-dl-bar")).toHaveCount(1);
  const cancel = page.locator(".mdl-btn.cancel-dl");
  await expect(cancel).toBeVisible();

  await cancel.click();
  const sent = await page.evaluate(() => window.__sent);
  expect(sent.some((m) => m.type === "action:cancel-download")).toBe(true);
});
