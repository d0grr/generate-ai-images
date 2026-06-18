// i18n with an optional runtime locale override.
//
// By default messages come from chrome.i18n.getMessage, which is locked to the
// browser's UI language and cannot be changed at runtime. For the Dev build we
// layer a manual override on top: messages are loaded from the packaged
// _locales/<locale>/messages.json and resolved ourselves, so the dev UI can be
// pinned to a language regardless of the browser. Release builds are untouched
// and keep using native chrome.i18n.
//
// Call initI18n() once (and await it) before applyI18n()/t() in each context.

// Dev builds carry the literal "(Dev)" suffix in the manifest name; release
// builds restore the localized __MSG_appName__ placeholder (see build.js).
export const IS_DEV = /\(Dev\)/.test(chrome.runtime.getManifest().name);

// Packaged _locales (folder names — note the underscore variants es_419/pt_BR…).
export const AVAILABLE_LOCALES = [
  "am", "ar", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "es_419",
  "et", "fa", "fi", "fil", "fr", "gu", "he", "hi", "hr", "hu", "id", "it", "ja",
  "kn", "ko", "lt", "lv", "ml", "mr", "ms", "nl", "no", "pl", "pt_BR", "pt_PT",
  "ro", "ru", "sk", "sl", "sr", "sv", "sw", "ta", "te", "th", "tr", "uk", "vi",
  "zh_CN", "zh_TW",
];

const STORAGE_KEY = "uiLocale";
const NATIVE = "__native__"; // sentinel: follow the browser (use chrome.i18n)
const DEV_DEFAULT_LOCALE = "en"; // Dev defaults to English when nothing stored

// Loaded message maps for the override path (null ⇒ use native chrome.i18n).
let OVERRIDE = null;
let FALLBACK = null; // en, consulted when a key is missing in the chosen locale

async function fetchMessages(locale) {
  try {
    const url = chrome.runtime.getURL(`_locales/${locale}/messages.json`);
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function getStored() {
  return new Promise((resolve) => {
    try {
      chrome.storage.local.get(STORAGE_KEY, (r) => resolve(r?.[STORAGE_KEY] || null));
    } catch {
      resolve(null);
    }
  });
}

// The locale the dev switcher should show as selected. null override ⇒ NATIVE.
export async function getStoredLocale() {
  const stored = await getStored();
  if (stored) return stored;
  return IS_DEV ? DEV_DEFAULT_LOCALE : NATIVE;
}

export async function setLocale(locale) {
  await new Promise((resolve) => {
    try {
      chrome.storage.local.set({ [STORAGE_KEY]: locale }, resolve);
    } catch {
      resolve();
    }
  });
}

export async function initI18n() {
  // Release builds keep native chrome.i18n behavior untouched.
  if (!IS_DEV) return;
  const locale = (await getStored()) || DEV_DEFAULT_LOCALE;
  if (locale === NATIVE) return; // explicitly follow the browser
  OVERRIDE = await fetchMessages(locale);
  if (OVERRIDE && locale !== "en") FALLBACK = await fetchMessages("en");
}

// chrome.i18n-compatible substitution: named $PLACEHOLDER$ (resolved via the
// entry's placeholders → usually "$1"), then positional $1..$9, with $$ → $.
function substitute(entry, subs) {
  let msg = entry.message || "";
  const ph = entry.placeholders;
  if (ph) {
    msg = msg.replace(/\$([a-z0-9_@]+)\$/gi, (m, name) => {
      const def = ph[name.toLowerCase()];
      return def && def.content != null ? def.content : m;
    });
  }
  const arr = subs == null ? [] : Array.isArray(subs) ? subs : [subs];
  return msg.replace(/\$(\$|[1-9])/g, (m, d) => (d === "$" ? "$" : arr[+d - 1] ?? ""));
}

// Resolve a key: override map (+ en fallback) when active, else native.
function lookup(key, subs) {
  if (OVERRIDE) {
    const e = OVERRIDE[key] || (FALLBACK && FALLBACK[key]);
    if (e) return substitute(e, subs);
  }
  return chrome.i18n.getMessage(key, subs);
}

// Applies translations to DOM elements that carry data-i18n* attributes.
// Call once after DOMContentLoaded; safe to call multiple times (idempotent per element).
export function applyI18n(root = document) {
  const msg = (key) => lookup(key) || "";

  root.querySelectorAll("[data-i18n]").forEach((el) => {
    const m = msg(el.dataset.i18n);
    if (m) el.textContent = m;
  });
  root.querySelectorAll("[data-i18n-html]").forEach((el) => {
    const m = msg(el.dataset.i18nHtml);
    if (m) el.innerHTML = m;
  });
  root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const m = msg(el.dataset.i18nPlaceholder);
    if (m) el.placeholder = m;
  });
  root.querySelectorAll("[data-i18n-title]").forEach((el) => {
    const m = msg(el.dataset.i18nTitle);
    if (m) el.title = m;
  });
  root.querySelectorAll("[data-i18n-aria]").forEach((el) => {
    const m = msg(el.dataset.i18nAria);
    if (m) el.setAttribute("aria-label", m);
  });
  root.querySelectorAll("[data-i18n-alt]").forEach((el) => {
    const m = msg(el.dataset.i18nAlt);
    if (m) el.alt = m;
  });
}

// Shorthand for message lookup used in JS code.
export const t = (key, subs) => lookup(key, subs) || key;
