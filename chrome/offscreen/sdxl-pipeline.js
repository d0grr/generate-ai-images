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
    // VAE encoder (image → latent) — loaded lazily on the first img2img edit so
    // text-to-image users never pay its download/compile. Config is supplied per
    // generate() call (the encoder repo is independent of the UNet repo).
    this.vaeEncoder = null;
    this.vaeEncoderRepoId = null;
    this.vaeEncoderFile = "model";
    this.vaeEncoderIO = "float32";   // standalone SDXL VAE export has fp32 I/O
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
      // img2img — when initImage (RGBA ImageData at native 1024²) is present, the
      // pipeline encodes it to a latent and starts the denoise partway through.
      initImage = null,
      strength = 0.6,
      onEncodeProgress,
      // inpainting — mask is a single-channel Float32Array at latent resolution
      // (LATENT²), 1 = regenerate, 0 = keep. When present, the kept region is
      // re-pinned to the (noised) source latent after every denoise step.
      mask = null,
    } = opts;
    const doCFG = guidance_scale > 1;
    const isImg2Img = !!initImage;
    const isInpaint = isImg2Img && !!mask;

    // 1-3. Text encode the prompt (and the negative prompt when guiding).
    const cond = await this._encode(prompt);
    const uncond = doCFG ? await this._encode(negative_prompt) : null;

    // 4. scheduler + init latents.
    this.scheduler.set_timesteps(steps);
    let latents;
    // startStep > 0 only for img2img: how much of the schedule we SKIP. Higher
    // strength → start earlier (more noise, freer reinterpretation); lower
    // strength → start later (closer to the source). 0 means "from scratch".
    let startStep = 0;
    // Kept across the loop for inpainting (re-pin the unmasked region each step).
    let initLatents = null;
    let noise = null;
    if (isImg2Img) {
      await this._ensureVaeEncoder({ progress: onEncodeProgress, signal });
      if (signal?.aborted) throw new Error("cancelled");
      initLatents = await this._encodeImageToLatents(initImage);
      const s = Math.min(1, Math.max(0.02, strength));
      // Run ≈ steps·strength denoise steps; keep at least one.
      startStep = Math.min(steps - 1, Math.max(0, Math.round(steps * (1 - s))));
      noise = randn(LCH * LATENT * LATENT, seed >>> 0);
      latents = this.scheduler.add_noise(initLatents, noise, startStep);
    } else {
      latents = randn(LCH * LATENT * LATENT, seed >>> 0);
      const initSigma = this.scheduler.init_noise_sigma;
      for (let i = 0; i < latents.length; i++) latents[i] *= initSigma;
    }

    const condT = this._condTensors(cond.ehs, cond.pooled);
    const uncondT = doCFG ? this._condTensors(uncond.ehs, uncond.pooled) : null;
    const timeIdsT = condT.timeIdsT;     // identical for both passes (native 1024²)
    const latShape = [1, LCH, LATENT, LATENT];

    // 5. denoise (from startStep for img2img; from 0 for text2img)
    const runSteps = steps - startStep;
    for (let i = startStep; i < steps; i++) {
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

      // Inpainting: lock the unmasked region back to the source latent, noised to
      // the NEXT step's level (sigmas[i+1]; the final step's sigma is 0, so the
      // kept region lands exactly on the clean source). The masked region keeps
      // the freshly denoised values → only it follows the prompt.
      if (isInpaint) {
        const plane = LATENT * LATENT;
        const origNoised = this.scheduler.add_noise(initLatents, noise, i + 1);
        for (let p = 0; p < plane; p++) {
          const m = mask[p];
          if (m >= 0.999) continue;            // fully editable — leave as denoised
          const keep = 1 - m;
          for (let c = 0; c < LCH; c++) {
            const idx = c * plane + p;
            latents[idx] = m * latents[idx] + keep * origNoised[idx];
          }
        }
      }

      onStep?.(i - startStep + 1, runSteps);
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
    for (const s of [this.textEncoder, this.textEncoder2, this.unet, this.vaeDecoder, this.vaeEncoder]) {
      try { await s?.release?.(); } catch {}
    }
    this.textEncoder = this.textEncoder2 = this.unet = this.vaeDecoder = this.vaeEncoder = null;
  }

  // ── img2img: VAE encoder (image → latent) ──────────────────────────────────
  /**
   * Lazily create the VAE-encoder ORT session. Loaded from a SEPARATE repo
   * (vaeEncoderRepoId — e.g. the ONNX Runtime team's tlwu/sdxl-turbo-onnxruntime)
   * because the bundled Lightning repo ships only a decoder. The SDXL VAE is
   * frozen, so any standard SDXL vae_encoder pairs with our decoder. The export
   * does the reparameterised sample (mean + std·ε) inside the graph, so its
   * `latent_sample` output is already a drawn latent — we only apply VAE_SCALE.
   */
  async _ensureVaeEncoder({ progress, signal } = {}) {
    if (this.vaeEncoder) return this.vaeEncoder;
    if (!this.vaeEncoderRepoId) {
      throw new Error("This model has no VAE encoder configured — image editing is unavailable.");
    }
    const ort = loadOrt();
    const base = `${repoRoot(this.vaeEncoderRepoId)}/vae_encoder`;
    const stem = this.vaeEncoderFile || "model";
    const checkAbort = () => { if (signal?.aborted) throw new Error("aborted"); };
    checkAbort();
    const graphBuf = await fetchOnnxCached(`${base}/${stem}.onnx`,
      (loaded, total) => progress?.({ file: "vae_encoder", loaded, total, phase: "downloading" }), signal);
    checkAbort();
    const externalData = [];
    for (const name of await discoverShards(base, stem)) {
      const buf = await fetchOnnxCached(`${base}/${name}`,
        (loaded, total) => progress?.({ file: "vae_encoder weights", loaded, total, phase: "downloading" }), signal);
      checkAbort();
      externalData.push({ path: name, data: new Uint8Array(buf) });
    }
    progress?.({ file: "vae_encoder", loaded: 1, total: 1, phase: "compiling" });
    const opts = { executionProviders: ["webgpu"], graphOptimizationLevel: "all" };
    if (externalData.length) opts.externalData = externalData;
    this.vaeEncoder = await ort.InferenceSession.create(graphBuf, opts);
    return this.vaeEncoder;
  }

  /**
   * RGBA ImageData at native 1024² → model-space latent Float32Array[4·128·128].
   * Pixels → CHW in [-1,1] (the encoder's `sample` input), run, widen the fp16
   * `latent_sample`, multiply by VAE_SCALE (the decoder divides by it).
   */
  async _encodeImageToLatents(imageData) {
    const ort = loadOrt();
    const plane = SIDE * SIDE;             // 1024²
    const io = this.vaeEncoderIO === "float32" ? "float32" : "float16";
    const A = io === "float16" ? Float16Array : Float32Array;
    const chw = new A(3 * plane);
    const px = imageData.data;
    for (let i = 0; i < plane; i++) {
      chw[i]           = px[i * 4]     / 127.5 - 1;
      chw[i + plane]   = px[i * 4 + 1] / 127.5 - 1;
      chw[i + 2 * plane] = px[i * 4 + 2] / 127.5 - 1;
    }
    const out = await this.vaeEncoder.run({
      sample: new ort.Tensor(io, chw, [1, 3, SIDE, SIDE]),
    });
    // Two export conventions:
    //   • `latent_sample` [1,4,h,w] — already a drawn latent (sampling baked in)
    //   • `latent_parameters` [1,8,h,w] — raw moments (mean | logvar concatenated)
    // We always take the FIRST LCH(=4) channels: for moments that's the mean (the
    // distribution mode — deterministic, the standard img2img init), and for a
    // 4-channel output it's the latent itself. Then scale (decoder divides back).
    const raw = dataF32(out.latent_sample ?? out.latent_parameters ?? Object.values(out)[0]);
    const n = LCH * LATENT * LATENT;
    const scaled = new Float32Array(n);
    for (let i = 0; i < n; i++) scaled[i] = raw[i] * VAE_SCALE;
    return scaled;
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
