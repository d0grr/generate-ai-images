#!/usr/bin/env node
/**
 * Build script for the "Generate AI Images" Chrome extension.
 *
 * Build modes:
 *   npm run build      — release: Terser (minify + strip comments), no source maps
 *   npm run build:dev  — dev:     Terser + source maps, for Chrome DevTools profiling
 *   npm run build:raw  — raw:     copy as-is, no minification, with ZIP
 *
 * Pipeline per file type:
 *
 *  JS  (first-party)  — Terser compress×2 + mangle
 *  JS  (vendor)       — Terser compress×1, mangle:false (safe for libs)
 *  CSS (first-party)  — CleanCSS level 2
 *  HTML               — html-minifier-terser (collapse + inline CSS/JS)
 *  JSON (_locales, manifest) — JSON.parse → JSON.stringify (strip whitespace)
 *  WASM / PNG / SVG   — copy verbatim
 *
 * Outputs:
 *   dist/                       — unpacked extension (loadable in Chrome)
 *   release/extension-v<V>.zip  — packaged build (release/raw modes)
 */

import { minify as terserMinify }         from "terser";
import CleanCSS                           from "clean-css";
import { minify as htmlMinify }           from "html-minifier-terser";
import fs                                 from "node:fs";
import path                               from "node:path";
import { fileURLToPath }                  from "node:url";
import { execSync }                       from "node:child_process";

const DEV = process.argv.includes("--dev");
const RAW = process.argv.includes("--raw");
// --firefox: build the Firefox variant by overlaying ../firefox/ (manifest +
// background page) on top of the shared chrome/ sources. Everything else (engine,
// UI, locales, vendor, assets) is reused as-is from this directory.
const FIREFOX = process.argv.includes("--firefox");

const SRC         = path.dirname(fileURLToPath(import.meta.url));
const TARGET      = FIREFOX ? "firefox" : "chrome";
// Firefox-specific files live in ../firefox/ and shadow the shared chrome/ ones.
const OVERLAY     = FIREFOX ? path.resolve(SRC, "..", "firefox") : null;
// Shared chrome/ files that the Firefox build must NOT ship (replaced by the
// overlay's background page; Firefox has no offscreen document).
const TARGET_EXCLUDE = FIREFOX
  ? new Set(["background/service-worker.js", "offscreen/offscreen.html"])
  : new Set();

const DIST        = FIREFOX ? path.resolve(OVERLAY, "dist") : path.resolve(SRC, "dist");
// Packaged builds collect in the repo-root release/ (shared across platforms);
// the zip is platform-prefixed so each target drops alongside the other.
const RELEASE_DIR = path.resolve(SRC, "..", "release");

const MANIFEST_PATH = FIREFOX ? path.join(OVERLAY, "manifest.json") : path.join(SRC, "manifest.json");
const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
const VERSION  = manifest.version;
const ZIP_PATH = path.join(RELEASE_DIR, `generate-ai-images-${TARGET}_${VERSION}.zip`);

const RELEASE = !DEV && !RAW;

// Resolve a relative file to the overlay copy when present, else the shared one.
function srcFor(relPath) {
  if (OVERLAY) {
    const overlaid = path.join(OVERLAY, relPath);
    if (fs.existsSync(overlaid)) return overlaid;
  }
  return path.join(SRC, relPath);
}

// ─────────────────────────────────────────────────────────────────────────────
// FILE REGISTRY
// ─────────────────────────────────────────────────────────────────────────────

// First-party ES-module JS → Terser
const JS_ES_MODULE = new Set([
  "background/service-worker.js",     // chrome entry
  "background/background.js",         // firefox entry (overlay)
  "background/engine-direct.js",      // firefox flag module (overlay)
  "background/core.js",
  "offscreen/ort-init.js",
  "offscreen/offscreen.js",
  "offscreen/sd-pipeline.js",
  "offscreen/sdxl-pipeline.js",
  "offscreen/scheduler.js",
  "offscreen/upscale-pipeline.js",
  "offscreen/face-restore-pipeline.js",
  "popup/catalog.js",
  "popup/i18n.js",
  "popup/lightbox.js",
  "popup/nsfw-filter.js",
  "popup/popup.js",
  "popup/prompts.js",
  "popup/quality.js",
  "workspace/workspace.js",
]);

// First-party classic JS → Terser
const JS_CLASSIC = new Set([
  "offscreen/workers/opfs-writer.js",
]);

// Vendor JS (classic) → Terser compress-only
const JS_VENDOR_CLASSIC = new Set([
  "offscreen/vendor/transformers.min.js",
  "workspace/vendor/jszip.min.js",
]);

// Vendor MJS (ES module) → Terser compress-only
// ORT 1.26 native WebGPU bundle + its asyncify wasm glue.
const JS_VENDOR_MODULE = new Set([
  "offscreen/vendor/ort.webgpu.bundle.min.mjs",
  "offscreen/vendor/ort-wasm-simd-threaded.asyncify.mjs",
]);

// First-party CSS → CleanCSS
const CSS_FIRST_PARTY = new Set([
  "popup/lightbox.css",
  "popup/popup.css",
  "workspace/workspace.css",
]);

// HTML → html-minifier-terser
const HTML_FILES = new Set([
  "offscreen/offscreen.html",
  "background/background.html",       // firefox engine host page (overlay)
  "popup/lightbox.html",
  "popup/popup.html",
  "workspace/workspace.html",
]);

const INCLUDE_DIRS  = ["_locales", "background", "icons", "offscreen", "popup", "workspace"];
const INCLUDE_FILES = ["manifest.json"];
const SKIP_NAMES    = new Set([".DS_Store", "Thumbs.db", ".gitkeep"]);

// ─────────────────────────────────────────────────────────────────────────────
// PROCESSORS
// ─────────────────────────────────────────────────────────────────────────────

const cssCleaner = new CleanCSS({ level: 2, returnPromise: true });

async function processJS(relPath, srcPath, dstPath, { vendor = false } = {}) {
  const isModule = JS_ES_MODULE.has(relPath) || JS_VENDOR_MODULE.has(relPath);
  const src = fs.readFileSync(srcPath, "utf8");

  const terserOpts = vendor
    ? {
        module:   isModule,
        compress: { passes: 1, drop_console: false },
        mangle:   false,
        format:   { comments: false },
      }
    : {
        module:   isModule,
        compress: { passes: 2, drop_console: false },
        mangle:   { module: isModule },
        format:   { comments: false },
        ...(DEV && {
          sourceMap: {
            filename: path.basename(dstPath),
            url:      path.basename(dstPath) + ".map",
          },
        }),
      };

  const terserResult = await terserMinify(src, terserOpts);
  if (!terserResult.code) throw new Error(`Terser empty output: ${relPath}`);

  fs.writeFileSync(dstPath, terserResult.code, "utf8");
  if (terserResult.map) fs.writeFileSync(dstPath + ".map", terserResult.map, "utf8");
  log(vendor ? "minify JS↓" : "minify JS ", relPath, src.length, terserResult.code.length);
}

async function processCSS(relPath, srcPath, dstPath) {
  const code = fs.readFileSync(srcPath, "utf8");
  const result = await cssCleaner.minify(code);
  if (result.errors.length) throw new Error(`CleanCSS errors in ${relPath}: ${result.errors}`);
  fs.writeFileSync(dstPath, result.styles, "utf8");
  log("minify CSS", relPath, code.length, result.styles.length);
}

async function processHTML(relPath, srcPath, dstPath) {
  const code = fs.readFileSync(srcPath, "utf8");
  const result = await htmlMinify(code, {
    collapseWhitespace:            true,
    removeComments:                true,
    removeRedundantAttributes:     true,
    removeScriptTypeAttributes:    true,
    removeStyleLinkTypeAttributes: true,
    minifyCSS:                     true,
    minifyJS:                      true,
    useShortDoctype:               true,
  });
  fs.writeFileSync(dstPath, result, "utf8");
  log("minify HTML", relPath, code.length, result.length);
}

function processJSON(relPath, srcPath, dstPath) {
  const code = fs.readFileSync(srcPath, "utf8");
  const obj = JSON.parse(code);

  // src/manifest.json carries the literal "(Dev)" name so an unpacked copy
  // loaded straight from src/ is distinguishable from a packaged install. The
  // release build restores the localized __MSG_* placeholders + a clean,
  // HTTPS-only CSP (the dev manifest whitelists http://localhost for local
  // model testing).
  if (relPath === "manifest.json" && RELEASE) {
    obj.name       = "__MSG_appName__";
    obj.short_name = "__MSG_shortName__";
    if (obj.action) obj.action.default_title = "__MSG_action_title__";
    if (obj.sidebar_action) obj.sidebar_action.default_title = "__MSG_action_title__";
    if (obj.content_security_policy?.extension_pages) {
      obj.content_security_policy.extension_pages =
        "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'; connect-src 'self' https:;";
    }
  }

  const minified = JSON.stringify(obj);
  fs.writeFileSync(dstPath, minified, "utf8");
  log("minify JSON", relPath, code.length, minified.length);
}

function copyVerbatim(relPath, srcPath, dstPath) {
  fs.copyFileSync(srcPath, dstPath);
  console.log(`  copy      ${relPath}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// ROUTING
// ─────────────────────────────────────────────────────────────────────────────

async function processFile(relPath) {
  const srcPath = srcFor(relPath);
  const dstPath = path.join(DIST, relPath);
  fs.mkdirSync(path.dirname(dstPath), { recursive: true });

  if (RAW) return copyVerbatim(relPath, srcPath, dstPath);

  const ext = path.extname(relPath);

  if (ext === ".js" || ext === ".mjs") {
    if (JS_VENDOR_CLASSIC.has(relPath) || JS_VENDOR_MODULE.has(relPath))
      return processJS(relPath, srcPath, dstPath, { vendor: true });
    if (JS_ES_MODULE.has(relPath) || JS_CLASSIC.has(relPath))
      return processJS(relPath, srcPath, dstPath);
    console.warn(`  WARN: unregistered JS, processing as classic: ${relPath}`);
    return processJS(relPath, srcPath, dstPath);
  }

  if (ext === ".css") {
    if (CSS_FIRST_PARTY.has(relPath)) return processCSS(relPath, srcPath, dstPath);
    console.warn(`  WARN: unregistered CSS, copying: ${relPath}`);
    return copyVerbatim(relPath, srcPath, dstPath);
  }

  if (ext === ".html") {
    if (HTML_FILES.has(relPath)) return processHTML(relPath, srcPath, dstPath);
    console.warn(`  WARN: unregistered HTML, copying: ${relPath}`);
    return copyVerbatim(relPath, srcPath, dstPath);
  }

  if (ext === ".json") return processJSON(relPath, srcPath, dstPath);

  copyVerbatim(relPath, srcPath, dstPath);
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────

function log(label, relPath, origLen, finalLen) {
  const sign  = finalLen > origLen ? "+" : "-";
  const delta = Math.abs(((origLen - finalLen) / origLen) * 100).toFixed(0);
  console.log(
    `  ${label.padEnd(10)} ${relPath.padEnd(48)}` +
    ` ${(origLen / 1024).toFixed(1).padStart(7)} → ${(finalLen / 1024).toFixed(1).padStart(7)} KB  (${sign}${delta}%)`
  );
}

function walkDir(dir, base = dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP_NAMES.has(e.name)) continue;
    const full = path.join(dir, e.name);
    e.isDirectory() ? out.push(...walkDir(full, base)) : out.push(path.relative(base, full));
  }
  return out;
}

function buildZip() {
  fs.mkdirSync(RELEASE_DIR, { recursive: true });
  if (fs.existsSync(ZIP_PATH)) fs.rmSync(ZIP_PATH);
  execSync(`zip -r "${ZIP_PATH}" . --exclude "*.map"`, { cwd: DIST, stdio: "pipe" });
  const kb = (fs.statSync(ZIP_PATH).size / 1024).toFixed(0);
  console.log(`\n  → ${ZIP_PATH}  (${kb} KB)`);
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

async function main() {
  const modeLabel = RAW ? "raw — no minification"
    : DEV ? "dev — source maps"
    : "release — minified";
  console.log(`\nBuild: Generate AI Images v${VERSION}  [${TARGET} · ${modeLabel}]\n`);
  if (fs.existsSync(DIST)) fs.rmSync(DIST, { recursive: true });
  fs.mkdirSync(DIST, { recursive: true });

  // Collect shared chrome/ files (minus this target's exclusions), then merge in
  // the overlay's own files (e.g. the Firefox background page). A Set dedupes so
  // an overlaid file that also exists in chrome/ is processed once (srcFor picks
  // the overlay copy).
  const tasks = new Set(INCLUDE_FILES);
  for (const dir of INCLUDE_DIRS) {
    const abs = path.join(SRC, dir);
    if (fs.existsSync(abs)) {
      for (const r of walkDir(abs)) {
        const rel = path.join(dir, r);
        if (!TARGET_EXCLUDE.has(rel)) tasks.add(rel);
      }
    }
  }
  if (OVERLAY) {
    // Pull in overlay-only files under the shared dirs (e.g. background/*.html).
    // The overlay's build output (dist/) and docs (README.md) are not in
    // INCLUDE_DIRS, so they're never walked.
    for (const dir of INCLUDE_DIRS) {
      const abs = path.join(OVERLAY, dir);
      if (fs.existsSync(abs)) {
        for (const r of walkDir(abs)) tasks.add(path.join(dir, r));
      }
    }
  }

  for (const relPath of tasks) await processFile(relPath);

  if (!DEV || RAW) {
    console.log("\nPackaging…");
    buildZip();
  }
  console.log("Done.\n");
}

main().catch((e) => { console.error(e); process.exit(1); });
