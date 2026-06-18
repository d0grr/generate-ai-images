// ─────────────────────────────────────────────────────────────
// Stable Diffusion XL pipeline — browser edition (Lightning 4-step).
//
//   2× CLIP tokenize → 2× text encode → concat 2048 + pooled(1280) + time_ids
//   → Euler-trailing denoise (UNet, guidance 0) → VAE(÷0.13025) → image
//
// Mirrors StableDiffusionPipeline's public shape (load / generate / dispose)
// so offscreen.js can dispatch on model kind with a one-line branch. Reuses
// sd-pipeline's OPFS weight cache, tokenizer cache, ORT bootstrap and fp16
// helpers — only the SDXL-specific math lives here.
//
// Validated end-to-end in standalone browser harnesses (since removed; findings
// recorded in docs/SDXL-BROWSER-DEVLOG.md). Key facts baked in from that work:
//   • text encoders MUST run on the WASM EP — CLIP-bigG fp16 on WebGPU NaNs.
//   • penultimate hidden states: text_encoder "hidden_states.11" (768),
//     text_encoder_2 "hidden_states.31" (1280); pooled = "text_embeds" (1280).
//   • all component I/O is fp32 (optimum keep_io_types) — no fp16 packing.
//   • VAE scaling factor 0.13025; native 1024² / latent 128.
// ─────────────────────────────────────────────────────────────

import {
  loadOrt,
  loadTransformers,
  fetchOnnxCached,
  fetchJsonCachedOPFS,
  discoverShards,
  repoRoot,
  f16ToF32,
} from "./sd-pipeline.js";
import { EulerScheduler, randn } from "./scheduler.js";

// repoRoot (from sd-pipeline) resolves an HF repo id or passes a full http(s)
// URL verbatim — lets a dev build point browserRepoId/browserEncoderRepoId at a
// local model server (tools/serve-local-model.py) to try a self-export before
// it's hosted. Reusable for any custom host, not just localhost.

// SDXL constants
const SEQ = 77;
const HID_L = 768;     // CLIP-L hidden
const HID_G = 1280;    // CLIP-bigG hidden
const HID = HID_L + HID_G; // 2048
const VAE_SCALE = 0.13025;
const LATENT = 128, LCH = 4;     // 1024² / 8
const SIDE = LATENT * 8;         // 1024
// Penultimate (-2) hidden-state output names for each encoder.
const TE_PENULT = "hidden_states.11";   // CLIP-L: 12 layers → 13 states, -2 = 11
const TE2_PENULT = "hidden_states.31";  // CLIP-bigG: 32 layers → 33 states, -2 = 31

export class SDXLPipeline {
  constructor() {
    this.repoId = null;
    this.tokenizer = null;
    this.tokenizer2 = null;
    this.textEncoder = null;
    this.textEncoder2 = null;
    this.unet = null;
    this.vaeDecoder = null;
    this.scheduler = null;
  }

  /**
   * Load tokenizers + 4 ONNX sessions for a SDXL repo.
   * Signature matches StableDiffusionPipeline.load; schedulerType/dtype are
   * accepted for interface parity but SDXL always uses Euler-trailing fp32 I/O.
   */
  async load(repoId, { progress, signal, encoderRepoId, unetIO = "float32", timestepType = "int64", unetFile = "model" } = {}) {
    // The frozen SDXL text encoders + VAE + tokenizers are identical across all
    // SDXL models, so they're loaded from a SHARED repo (encoderRepoId), while
    // only the fine-tuned UNet comes from `repoId`. For single-repo models
    // encoderRepoId defaults to repoId — same behaviour as before.
    // unetIO / timestepType describe the UNet export's I/O convention:
    //   • prebuilt re-shard: fp32 I/O, INT64 [1] timestep
    //   • our self-exports:  fp16 I/O, FLOAT16 scalar timestep
    const encRepo = encoderRepoId || repoId;
    this.repoId = repoId;
    this.unetIO = unetIO === "float16" ? "float16" : "float32";
    this.timestepType = timestepType;
    console.info(`[sdxl] load unet=${repoId} enc=${encRepo} io=${this.unetIO} ts=${timestepType}`);
    const ort = loadOrt();
    const T = await loadTransformers();
    const checkAbort = () => { if (signal?.aborted) throw new Error("aborted"); };

    checkAbort();
    progress?.({ file: "tokenizers", loaded: 0, total: 1, phase: "init" });
    this.tokenizer  = await loadCLIPTokenizer(T, encRepo, "tokenizer");
    this.tokenizer2 = await loadCLIPTokenizer(T, encRepo, "tokenizer_2");
    checkAbort();

    // Text encoders: WebGPU EP at "basic" (ORT 1.26's EXTENDED
    // SimplifiedLayerNormFusion crashes the CLIP graph; "basic" skips it, fp16
    // Pow has a WebGPU kernel). Shards auto-discovered, so any export layout
    // works. Component base URLs: encoders+VAE from the shared repo, UNet from
    // the per-model repo (OPFS caches by URL → shared parts download once).
    const encBase = repoRoot(encRepo);
    const unetBase = repoRoot(repoId);
    const loadOnnx = async (cbase, label, eps, optLevel = "all", fileStem = "model") => {
      checkAbort();
      progress?.({ file: label, loaded: 0, total: 1, phase: "downloading" });
      const graphBuf = await fetchOnnxCached(`${cbase}/${fileStem}.onnx`,
        (loaded, total) => progress?.({ file: label, loaded, total, phase: "downloading" }), signal);
      checkAbort();
      const externalData = [];
      for (const name of await discoverShards(cbase, fileStem)) {
        const buf = await fetchOnnxCached(`${cbase}/${name}`,
          (loaded, total) => progress?.({ file: `${label} weights`, loaded, total, phase: "downloading" }), signal);
        checkAbort();
        externalData.push({ path: name, data: new Uint8Array(buf) });
      }
      progress?.({ file: label, loaded: 1, total: 1, phase: "compiling" });
      const opts = { executionProviders: eps, graphOptimizationLevel: optLevel };
      if (externalData.length) opts.externalData = externalData;
      return ort.InferenceSession.create(graphBuf, opts);
    };

    this.textEncoder  = await loadOnnx(`${encBase}/text_encoder`,   "text_encoder",   ["webgpu"], "basic");
    this.textEncoder2 = await loadOnnx(`${encBase}/text_encoder_2`, "text_encoder_2", ["webgpu"], "basic");
    this.unet         = await loadOnnx(`${unetBase}/unet`,          "unet",           ["webgpu"], "all", unetFile);
    this.vaeDecoder   = await loadOnnx(`${encBase}/vae_decoder`,    "vae_decoder",    ["webgpu"], "all");

    this.scheduler = new EulerScheduler({
      num_train_timesteps: 1000,
      beta_start: 0.00085,
      beta_end: 0.012,
      beta_schedule: "scaled_linear",
      prediction_type: "epsilon",
      timestep_spacing: "trailing",
    });
  }

  // ── UNet input builders (honour the export's I/O convention) ───────────────
  _condTensors(ehs, pooled) {
    const ort = loadOrt(), dt = this.unetIO, A = dt === "float16" ? Float16Array : Float32Array;
    return {
      ehsT:     new ort.Tensor(dt, new A(ehs),    [1, SEQ, HID]),
      pooledT:  new ort.Tensor(dt, new A(pooled), [1, HID_G]),
      timeIdsT: new ort.Tensor(dt, A.from([SIDE, SIDE, 0, 0, SIDE, SIDE]), [1, 6]),
    };
  }
  _sampleTensor(scaled) {
    const ort = loadOrt(), dt = this.unetIO;
    return new ort.Tensor(dt, dt === "float16" ? new Float16Array(scaled) : scaled, [1, LCH, LATENT, LATENT]);
  }
  _timestepTensor(tInt) {
    const ort = loadOrt();
    if (this.timestepType === "float16") return new ort.Tensor("float16", Float16Array.from([tInt]), []);
    if (this.timestepType === "float32") return new ort.Tensor("float32", Float32Array.from([tInt]), []);
    return new ort.Tensor("int64", BigInt64Array.from([BigInt(tInt)]), [1]);
  }

  /**
   * Tokenize + run both CLIP encoders for one prompt → { ehs, pooled }.
   *   ehs:    Float32Array[77·2048] — per-token concat of CLIP-L (768) penult
   *           and CLIP-bigG (1280) penult hidden states.
   *   pooled: Float32Array[1280]    — CLIP-bigG pooled text_embeds.
   * Factored out so classifier-free guidance can encode the negative prompt too.
   */
  async _encode(prompt) {
    const ort = loadOrt();
    const ids1 = await tokenizeIds(this.tokenizer, prompt);
    const ids2 = await tokenizeIds(this.tokenizer2, prompt);
    const o1 = await this.textEncoder.run({
      input_ids: new ort.Tensor("int32", Int32Array.from(ids1), [1, SEQ]),
    });
    const h1 = dataF32(o1[TE_PENULT]);          // 77*768
    const o2 = await this.textEncoder2.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids2.map(BigInt)), [1, SEQ]),
    });
    const h2 = dataF32(o2[TE2_PENULT]);         // 77*1280
    const pooled = dataF32(o2.text_embeds);     // 1280
    const ehs = new Float32Array(SEQ * HID);
    for (let p = 0; p < SEQ; p++) {
      ehs.set(h1.subarray(p * HID_L, p * HID_L + HID_L), p * HID);
      ehs.set(h2.subarray(p * HID_G, p * HID_G + HID_G), p * HID + HID_L);
    }
    return { ehs, pooled };
  }

  /**
   * Force shader compilation for every session by running one dummy pass each.
   * Called during the framed "Preparing" warmup so the heavy compiles (the UNet
   * async-compiles at load, but the VAE's pipelines compile on first run and
   * freeze ~12 s) happen there instead of mid-generation. After this, real
   * generations reuse the compiled pipelines and only pay compute.
   */
  async warmup() {
    const ort = loadOrt();
    const z = (n) => new Float32Array(n);
    try {
      await this.textEncoder.run({ input_ids: new ort.Tensor("int32", new Int32Array(SEQ), [1, SEQ]) });
      await this.textEncoder2.run({ input_ids: new ort.Tensor("int64", new BigInt64Array(SEQ), [1, SEQ]) });
      const { ehsT, pooledT, timeIdsT } = this._condTensors(z(SEQ * HID), z(HID_G));
      await this.unet.run({
        sample: this._sampleTensor(z(LCH * LATENT * LATENT)),
        timestep: this._timestepTensor(999),
        encoder_hidden_states: ehsT, text_embeds: pooledT, time_ids: timeIdsT,
      });
      // VAE is the shared standard SDXL VAE — always fp32 I/O.
      await this.vaeDecoder.run({
        latent_sample: new ort.Tensor("float32", z(LCH * LATENT * LATENT), [1, LCH, LATENT, LATENT]),
      });
    } catch (e) {
      console.warn("[sdxl] warmup dummy pass failed (non-fatal):", e?.message || e);
    }
  }

  /**
   * Run one generation → RGBA ImageData at 1024×1024.
   * Lightning distillations run without CFG (guidance ≤ 1 → one UNet pass).
   * With guidance > 1 (e.g. RealVisXL ~1.5) classifier-free guidance kicks in:
   * a second UNet pass on the negative prompt per step, combined as
   * uncond + scale·(cond − uncond) — fixes duplication / bad anatomy at the
   * cost of ~2× UNet time. width/height are ignored — native 1024².
   */
  async generate(opts) {
    const ort = loadOrt();
    const {
      prompt,
      negative_prompt = "",
      guidance_scale = 0,
      steps = 4,
      seed = Math.floor(Math.random() * 2 ** 31),
      onStep,
      onDecode,
      signal,
    } = opts;
    const doCFG = guidance_scale > 1;

    // 1-3. Text encode the prompt (and the negative prompt when guiding).
    const cond = await this._encode(prompt);
    const uncond = doCFG ? await this._encode(negative_prompt) : null;

    // 4. scheduler + init latents
    this.scheduler.set_timesteps(steps);
    let latents = randn(LCH * LATENT * LATENT, seed >>> 0);
    const initSigma = this.scheduler.init_noise_sigma;
    for (let i = 0; i < latents.length; i++) latents[i] *= initSigma;

    const condT = this._condTensors(cond.ehs, cond.pooled);
    const uncondT = doCFG ? this._condTensors(uncond.ehs, uncond.pooled) : null;
    const timeIdsT = condT.timeIdsT;     // identical for both passes (native 1024²)
    const latShape = [1, LCH, LATENT, LATENT];

    // 5. denoise
    for (let i = 0; i < steps; i++) {
      if (signal?.aborted) throw new Error("cancelled");
      const scaled = this.scheduler.scale_model_input(latents, i);
      const tInt = this.scheduler.timesteps[i];
      const sampleT = this._sampleTensor(scaled);   // reused across both passes
      const timestepT = this._timestepTensor(tInt);
      let noisePred;
      if (doCFG) {
        const outU = await this.unet.run({
          sample: sampleT, timestep: timestepT,
          encoder_hidden_states: uncondT.ehsT, text_embeds: uncondT.pooledT, time_ids: timeIdsT,
        });
        const outC = await this.unet.run({
          sample: sampleT, timestep: timestepT,
          encoder_hidden_states: condT.ehsT, text_embeds: condT.pooledT, time_ids: timeIdsT,
        });
        const nU = dataF32(outU.out_sample), nC = dataF32(outC.out_sample);
        noisePred = new Float32Array(nU.length);
        for (let j = 0; j < nU.length; j++) noisePred[j] = nU[j] + guidance_scale * (nC[j] - nU[j]);
      } else {
        const out = await this.unet.run({
          sample: sampleT, timestep: timestepT,
          encoder_hidden_states: condT.ehsT, text_embeds: condT.pooledT, time_ids: timeIdsT,
        });
        noisePred = dataF32(out.out_sample);
      }
      latents = this.scheduler.step(noisePred, i, latents);
      onStep?.(i + 1, steps);
    }

    // 6. VAE decode (÷ scaling factor). The full SDXL VAE's 1024² decode is a
    // ~2 s GPU-bound pass that briefly stalls the compositor; signal "decoding"
    // first (with a yield, in offscreen) so the UI frames it instead of hanging.
    await onDecode?.();
    const latIn = new Float32Array(latents.length);
    for (let i = 0; i < latents.length; i++) latIn[i] = latents[i] / VAE_SCALE;
    const vaeOut = await this.vaeDecoder.run({
      latent_sample: new ort.Tensor("float32", latIn, latShape),
    });
    const pixels = dataF32(vaeOut.sample ?? Object.values(vaeOut)[0]);

    // 7. [1,3,1024,1024] in [-1,1] → ImageData
    return tensorToImageData(pixels, SIDE, SIDE);
  }

  async dispose() {
    for (const s of [this.textEncoder, this.textEncoder2, this.unet, this.vaeDecoder]) {
      try { await s?.release?.(); } catch {}
    }
    this.textEncoder = this.textEncoder2 = this.unet = this.vaeDecoder = null;
  }
}

// ── helpers ────────────────────────────────────────────────────────────────

/** Manual CLIPTokenizer from a repo subdir (subfolder URLs, OPFS-cached). */
async function loadCLIPTokenizer(T, repoId, subdir) {
  const base = `${repoRoot(repoId)}/${subdir}`;
  const [tj, tc] = await Promise.all([
    fetchJsonCachedOPFS(`${base}/tokenizer.json`),
    fetchJsonCachedOPFS(`${base}/tokenizer_config.json`),
  ]);
  const Klass = T.CLIPTokenizer || T.PreTrainedTokenizer;
  if (!Klass) throw new Error("transformers.js missing CLIPTokenizer/PreTrainedTokenizer");
  return new Klass(tj, tc);
}

/** Tokenize → plain Number[] of length 77 (padded/truncated). */
async function tokenizeIds(tok, text) {
  const enc = await tok(text || "", { padding: "max_length", max_length: SEQ, truncation: true });
  const raw = enc.input_ids.data ?? enc.input_ids;
  const ids = new Array(SEQ);
  for (let i = 0; i < SEQ; i++) ids[i] = Number(raw[i]);
  return ids;
}

/** ORT tensor data → Float32Array (widening fp16 if a build hands it back). */
function dataF32(tensor) {
  const d = tensor.data;
  if (d instanceof Float32Array) return d;
  return tensor.type === "float16" ? f16ToF32(d) : Float32Array.from(d);
}

/** [1,3,H,W] in [-1,1] → RGBA ImageData. */
function tensorToImageData(data, width, height) {
  const plane = width * height;
  const image = new ImageData(width, height);
  const out = image.data;
  for (let i = 0; i < plane; i++) {
    out[i * 4]     = clip8(data[i]);
    out[i * 4 + 1] = clip8(data[i + plane]);
    out[i * 4 + 2] = clip8(data[i + 2 * plane]);
    out[i * 4 + 3] = 255;
  }
  return image;
}

function clip8(v) {
  const x = (v + 1) * 0.5 * 255;
  return x < 0 ? 0 : x > 255 ? 255 : x | 0;
}
