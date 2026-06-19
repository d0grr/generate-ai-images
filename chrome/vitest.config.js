import { defineConfig } from "vitest/config";

// Tier-1 logic tests run in Node (core.js / catalog.js use only chrome.* globals,
// which the tests mock). No DOM needed here; Tier-2 covers real-browser rendering.
export default defineConfig({
  test: {
    include: ["tests/unit/**/*.test.js"],
    environment: "node",
  },
});
