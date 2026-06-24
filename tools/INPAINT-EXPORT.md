# SDXL inpaint UNet — export in Colab & host on d0gr

Goal: a dedicated **SDXL inpainting** UNet in ONNX so the Edit tab does real
inpainting (9-channel UNet trained for it) instead of the Lightning latent-blend
hack. Only the UNet is new — text encoders, tokenizers, VAE and VAE encoder are
the frozen standard SDXL parts the engine already loads from your d0gr repos.

The SDXL inpaint UNet differs from a normal SDXL UNet **only** in `sample` having
9 channels (`conv_in` is 9→320: 4 noisy latent + 1 mask + 4 masked-image latent).

---

## Colab walkthrough

Open a new Colab notebook → **Runtime ▸ Change runtime type ▸ T4 GPU** (free is
enough). Run these cells.

### Cell 1 — deps
```python
!pip -q install "torch" "diffusers>=0.27" "transformers" "onnx" "accelerate" "huggingface_hub"
```

### Cell 2 — get the export script
```python
# paste the contents of tools/export-sdxl-inpaint-unet.py into export.py, or:
!wget -q https://raw.githubusercontent.com/<your-fork>/.../tools/export-sdxl-inpaint-unet.py -O export.py
```
(If not pushed to GitHub yet, just paste the file's contents into a cell and run it.)

### Cell 3 — export + reshard (~5–10 min on T4)
```python
!python export.py
!ls -lh out/unet           # → model.onnx + model.onnx_data[, _1 …], each <2 GB
```

### Cell 4 — upload to your d0gr repo
```python
from huggingface_hub import login, create_repo, upload_folder
from getpass import getpass
login(getpass("HF write token (hf.co/settings/tokens): "))
REPO = "d0gr/sdxl-inpaint-unet-onnx"
create_repo(REPO, repo_type="model", exist_ok=True, private=False)
upload_folder(folder_path="out/unet", path_in_repo="unet", repo_id=REPO)
print("done →", REPO)
```

Then tell me the repo id and I wire the pipeline + catalog.

---

## I/O contract (the script already produces exactly this)
| name | role | dtype | shape |
|---|---|---|---|
| `sample` | latent(4) ⊕ mask(1) ⊕ masked_latent(4) | **float16** | [1, 9, 128, 128] |
| `timestep` | step | **float16** | [] (scalar) |
| `encoder_hidden_states` | CLIP-L ⊕ bigG | float16 | [1, 77, 2048] |
| `text_embeds` | pooled | float16 | [1, 1280] |
| `time_ids` | added cond | float16 | [1, 6] |
| `out_sample` (out) | predicted noise | float16 | [1, 4, 128, 128] |

fp16 I/O + fp16 weights, standard ops only (no `com.microsoft.NhwcConv`).

## Catalog entry I'll add (for reference)
```js
browserRepoId:           "d0gr/sdxl-inpaint-unet-onnx",      // the UNet you upload
browserEncoderRepoId:    "d0gr/sdxl-lightning-onnx-webgpu",  // shared encoders/VAE/tokenizers
browserVaeEncoderRepoId: "d0gr/sdxl-vae-onnx",
browserKind:    "sdxl-inpaint",
browserUnetIO:  "float16",
browserTimestep:"float16",
browserSchedulerSpacing: "linspace",   // full SDXL, NOT Lightning's trailing
browserDefaultSteps: 24,
browserDefaultGuidance: 7.5,
```

## Expectations
Full (non-distilled) SDXL → ~20–30 steps × CFG (2 UNet passes) at 1024². On
WebGPU expect roughly **1–3 min per edit** and a ~5 GB one-time UNet download
(cached in OPFS). That's the cost of real SDXL inpaint quality.
