// Tier-2 (cross-browser UI): the popup runs as a page in real Chromium AND real
// Firefox (see playwright.config.js `projects`). These verify rendering + state
// on both engines — the class of bugs jsdom can't see (zoom / sticky / @supports
// / hit-testing) and cross-platform parity (e.g. eject behaving the same).

import { test, expect } from "@playwright/test";
import { installChromeMock, SEED_ACTIVE_MODEL } from "./chrome-mock.js";

const POPUP = "/popup/popup.html";

test.beforeEach(async ({ page }) => {
  await installChromeMock(page, SEED_ACTIVE_MODEL);
});

test("eject deselects the model (header + Models row reset) on this browser", async ({ page }) => {
  await page.goto(POPUP);

  // Sanity: a model is active → header brand shows its name.
  const brand = page.locator("[data-role='mb-model-name']");
  await expect(brand).not.toHaveText("Generate AI Images");

  // Go to Models, eject the active model.
  await page.locator(".ph-tab[data-tab='models']").click();
  await page.locator(".mdl-btn.unload").click();   // ⏏

  // Header returns to the first-run brand; the row reverts from "In use" to "Use".
  await expect(brand).toHaveText("Generate AI Images");
  await expect(page.locator(".mdl-btn.in-use")).toHaveCount(0);
  await expect(page.locator(".mdl-btn.use")).toHaveCount(1);

  // The action was dispatched to the SW.
  const sent = await page.evaluate(() => window.__sent);
  expect(sent.some((m) => m.type === "action:unload-model")).toBe(true);
});

// Regression guard for the Firefox-only freeze: under CSS `zoom`, `position:
// sticky` + `backdrop-filter` corrupted hit-testing so clicks died after a
// scroll. Runs on both engines; meaningful on the firefox project.
test("buttons stay clickable after scrolling the Models list", async ({ page }) => {
  await page.goto(POPUP);
  await page.locator(".ph-tab[data-tab='models']").click();

  const list = page.locator("#available-list");
  await list.evaluate((el) => el.scrollBy(0, 400)).catch(() => {});
  await page.mouse.wheel(0, 600);

  // A real click must still register after the scroll (hit-testing intact):
  // switching back to the Generate tab must work.
  await page.locator(".ph-tab[data-tab='generate']").click();
  await expect(page.locator("#tab-generate")).toBeVisible();
});
