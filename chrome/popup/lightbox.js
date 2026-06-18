const img = document.getElementById("img");

async function load() {
  const { lightbox } = await chrome.storage.session.get("lightbox");
  if (!lightbox) { window.close(); return; }

  // Show thumb immediately as placeholder.
  if (lightbox.thumbB64) {
    img.src = `data:image/png;base64,${lightbox.thumbB64}`;
  }

  // Upgrade to full-res from OPFS if available.
  if (lightbox.opfsKey) {
    try {
      const [dir, file] = lightbox.opfsKey.split("/");
      const root = await navigator.storage.getDirectory();
      const blob = await (await (await root.getDirectoryHandle(dir)).getFileHandle(file)).getFile();
      const prev = img.src;
      img.src = URL.createObjectURL(blob);
      if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
    } catch {}
  }
}

load();

document.addEventListener("click", () => window.close());
document.addEventListener("keydown", (e) => { if (e.key === "Escape") window.close(); });
