// ─────────────────────────────────────────────────────────────
// Dedicated worker that owns a FileSystemSyncAccessHandle for one
// model.onnx_data file.  SyncAccessHandle writes go straight to disk
// (no swap-file copy on close), which lets us write multi-GB files
// without doubling disk usage temporarily.
//
// The offscreen document opens this worker per ONNX file, sends chunks
// over postMessage, and receives a {ok, size} acknowledgement.  Errors
// surface back as {error}.
// ─────────────────────────────────────────────────────────────

let handle = null;       // FileSystemSyncAccessHandle
let written = 0;
let dirName = "";
let fileName = "";

// createSyncAccessHandle is exclusive — if a previous worker/page died
// mid-write, Chrome's OPFS can take a few hundred ms to release the lock.
// Retry with growing backoff so a stale lock unsticks itself instead of
// blowing up with "Access Handles cannot be created if there is another
// open Access Handle".
async function createHandleWithRetry(fh, attempts = 9) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fh.createSyncAccessHandle();
    } catch (err) {
      lastErr = err;
      const msg = String(err?.message || err);
      const phantom = msg.includes("Access Handle") || msg.includes("NoModificationAllowed");
      if (!phantom || i === attempts - 1) throw err;
      // Tearing down the previous writer (terminate + multi-GB flush/close)
      // can take well over a second to release the OS-level lock, so back off
      // further than the old 2.5 s ceiling: 300…1500 ms, ~9 s total.
      await new Promise((r) => setTimeout(r, Math.min(1500, 300 * (i + 1))));
    }
  }
  throw lastErr;
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    switch (msg?.type) {
      case "open": {
        dirName = msg.dirName;
        fileName = msg.fileName;
        const root = await navigator.storage.getDirectory();
        const dir = await root.getDirectoryHandle(dirName, { create: true });
        const fh = await dir.getFileHandle(fileName, { create: true });
        handle = await createHandleWithRetry(fh);
        const startSize = handle.getSize();
        // If user requested a fresh write, truncate. When the final size is
        // known (parallel segmented download), pre-extend to it so out-of-order
        // blocks land at absolute offsets and a resume can re-fetch only the
        // missing ones; the sequential path passes no size and truncates to 0.
        if (msg.truncate) {
          handle.truncate(typeof msg.size === "number" ? msg.size : 0);
          handle.flush();
          written = 0;
        } else {
          written = startSize;
          // Position the cursor at the end so writes append.
          // SyncAccessHandle writes accept an explicit offset via options.
        }
        self.postMessage({ type: "opened", size: written });
        break;
      }
      case "write": {
        if (!handle) throw new Error("handle not open");
        // msg.chunk is a transferred Uint8Array. `msg.at` lets the parallel
        // segmented downloader write blocks out of order at absolute offsets;
        // without it we append at the sequential cursor (legacy single stream).
        const chunk = msg.chunk;
        const at = typeof msg.at === "number" ? msg.at : written;
        const wrote = handle.write(chunk, { at });
        // Track the high-water mark, not a running sum — out-of-order writes
        // would otherwise inflate it past the real file size.
        written = Math.max(written, at + wrote);
        self.postMessage({ type: "wrote", id: msg.id, total: written, last: wrote });
        break;
      }
      case "flush": {
        if (handle) handle.flush();
        self.postMessage({ type: "flushed", id: msg.id, total: written });
        break;
      }
      case "close": {
        if (handle) {
          handle.flush();
          handle.close();
          handle = null;
        }
        self.postMessage({ type: "closed", id: msg.id, total: written });
        // self.close() not needed — offscreen will terminate the worker.
        break;
      }
      default:
        self.postMessage({ type: "error", id: msg?.id, error: `unknown message: ${msg?.type}` });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: msg?.id, error: String(err?.message || err) });
  }
};
