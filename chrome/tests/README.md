# Tests

Two tiers. Add cases by editing the files below — no app changes needed (tests
drive the real code through a mocked `chrome.*`).

## Tier 1 — logic (Vitest, Node, fast)

Drives `core.js` / `catalog.js` directly with an in-memory `chrome.*` mock. Covers
state-machine and data logic: eject deselect, model gating, message handling, etc.
No browser, no GPU.

```
npm run test:unit          # or: npm test
```

- Mock: `tests/mocks/chrome.js`
- Cases: `tests/unit/*.test.js`
- Add a case: new `it(...)` (or extend a data-driven list like
  `catalog-gating.test.js`'s `CASES`).

## Tier 2 — cross-browser UI (Playwright, real Chromium + Firefox)

Runs the built `popup.html` as a normal page in **both** engines with an injected
`chrome.*` mock. Covers real CSS / rendering / hit-testing (zoom, `@supports`,
`position: sticky`, `backdrop-filter`) and cross-platform parity — the bugs jsdom
can't see. No extension loading (so Playwright's Firefox works), no GPU, no model
downloads (the engine is not involved — state is seeded).

One-time setup (downloads browser binaries, ~hundreds of MB):

```
npm i -D @playwright/test
npx playwright install chromium firefox
```

Run (build first so chrome/dist is current):

```
npm run build && npm run test:e2e
npm run test:e2e -- --project=firefox      # one engine
```

- Page mock + seed: `tests/e2e/chrome-mock.js`
- Static server: `tests/e2e/static-server.mjs`
- Cases: `tests/e2e/*.spec.js`
- Add a case: new `test(...)`; seed state via `installChromeMock(page, {...})`.

## What is NOT automated

Real inference (WebGPU + multi-GB downloads + minutes/frame, flaky) — keep as a
manual smoke check. The engine is mocked/absent in both tiers.
