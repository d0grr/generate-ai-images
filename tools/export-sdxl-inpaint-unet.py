#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Export the SDXL inpainting UNet to ONNX for the browser WebGPU engine, then
# reshard its weights into <2 GB pieces. Designed for a Colab GPU runtime
# (free T4 is enough) — fp16 throughout keeps it inside ~13 GB RAM.
#
# I/O convention (fp16 self-export profile; matches the catalog's RealVis note):
#   • fp16 I/O, fp16 weights        → browserUnetIO   = "float16"
#   • FLOAT16 scalar timestep []    → browserTimestep = "float16"
#   • inputs : sample, timestep, encoder_hidden_states, text_embeds, time_ids
#   • output : out_sample
#
# `sample` has 9 channels (4 noisy latent + 1 mask + 4 masked-image latent) —
# the only thing that differs from a normal SDXL UNet. Encoders / VAE / tokenizers
# are the frozen standard SDXL parts (reused from your d0gr repos), so ONLY this
# UNet needs hosting.
#
# Output:  out/unet/model.onnx  +  model.onnx_data[, _1, _2 …]   (<2 GB each)
# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path
import torch
from diffusers import UNet2DConditionModel

MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
OUT = Path("out/unet")
OUT.mkdir(parents=True, exist_ok=True)
OPSET = 17
SHARD_MAX = 1_900_000_000   # ~1.9 GB per weight file (browser/V8 ArrayBuffer limit)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32   # CPU can't trace fp16
print(f"[export] device={device} dtype={dtype}")

print(f"[export] loading UNet from {MODEL_ID} …")
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=dtype).to(device)
unet.eval()
assert unet.config.in_channels == 9, f"expected 9-channel inpaint UNet, got {unet.config.in_channels}"


# torch.onnx can't trace a dict (added_cond_kwargs); expose its members as flat
# positional ONNX inputs.
class Wrap(torch.nn.Module):
    def __init__(self, u):
        super().__init__()
        self.u = u

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        return self.u(
            sample, timestep, encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )[0]


wrap = Wrap(unet)

B, H, W = 1, 128, 128   # native SDXL latent (1024/8)
dummies = (
    torch.randn(B, 9, H, W, dtype=dtype, device=device),       # sample (9ch)
    torch.tensor(999.0, dtype=dtype, device=device),           # timestep scalar []
    torch.randn(B, 77, 2048, dtype=dtype, device=device),      # encoder_hidden_states
    torch.randn(B, 1280, dtype=dtype, device=device),          # text_embeds (pooled)
    torch.randn(B, 6, dtype=dtype, device=device),             # time_ids
)

raw = OUT / "model_raw.onnx"
print("[export] tracing → ONNX …")
with torch.no_grad():
    torch.onnx.export(
        wrap, dummies, str(raw),
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["out_sample"],
        dynamic_axes={
            "sample": {0: "B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "B"},
            "text_embeds": {0: "B"},
            "time_ids": {0: "B"},
            "out_sample": {0: "B", 2: "H", 3: "W"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,   # legacy TorchScript exporter — no onnxscript dep, proven for SD UNets
    )
print(f"[export] traced graph: {raw}")

# ── reshard: pack big weights into model.onnx_data[, _N], each <SHARD_MAX ──
# torch.onnx (legacy, >2 GB) writes one external file PER tensor; reload them all
# into the proto, then stream-repack into a few <2 GB shards the engine expects.
import onnx

print("[reshard] loading graph with weights …")
model = onnx.load(str(raw), load_external_data=True)

def shard_name(i):
    return "model.onnx_data" if i == 0 else f"model.onnx_data_{i}"

inits = [t for t in model.graph.initializer if t.HasField("raw_data") and len(t.raw_data) > 1024]
print(f"[reshard] {len(inits)} weight tensors to externalize")

# Stream each tensor straight to the current shard file and free its raw_data
# immediately, so peak RAM stays near the loaded-model size (no duplicate buffers).
shard_idx, written = 0, 0
fh = open(OUT / shard_name(0), "wb")
for t in inits:
    data = t.raw_data
    if written and written + len(data) > SHARD_MAX:
        fh.close(); shard_idx += 1; written = 0
        fh = open(OUT / shard_name(shard_idx), "wb")
    offset = written
    fh.write(data); written += len(data)
    # set external_data manually (the helper guards on raw_data still being present)
    del t.external_data[:]
    for k, v in (("location", shard_name(shard_idx)), ("offset", str(offset)), ("length", str(len(data)))):
        e = t.external_data.add(); e.key = k; e.value = v
    t.data_location = onnx.TensorProto.EXTERNAL
    t.ClearField("raw_data")
fh.close()
n_shards = shard_idx + 1

onnx.save(model, str(OUT / "model.onnx"), save_as_external_data=False)

# Clean up: drop the traced graph and torch's per-tensor external files, keeping
# only model.onnx + the new model.onnx_data[_N] shards.
keep = {"model.onnx"} | {shard_name(i) for i in range(n_shards)}
for p in OUT.iterdir():
    if p.name not in keep:
        p.unlink()
print(f"[done] {OUT}/model.onnx + {n_shards} shard(s). Upload out/unet/ → d0gr repo.")
