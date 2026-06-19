import { defineConfig, devices } from "@playwright/test";

// Tier-2 runs the built popup as a page in real Chromium and Firefox.
// `webServer` serves chrome/dist (built first), so run `npm run build` before,
// or rely on CI to build. The popup's Firefox branches (@supports -moz-appearance,
// navigator.userAgent) light up automatically under the real Firefox engine.
const PORT = 5179;

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 30_000,
  expect: { timeout: 5_000 },
  use: { baseURL: `http://localhost:${PORT}` },
  webServer: {
    command: "node tests/e2e/static-server.mjs",
    url: `http://localhost:${PORT}/popup/popup.html`,
    reuseExistingServer: true,
    env: { E2E_PORT: String(PORT) },
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
  ],
});
