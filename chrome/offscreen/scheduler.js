// ─────────────────────────────────────────────────────────────
// Euler ancestral scheduler for Stable Diffusion.
//
// Ported from the HuggingFace diffusers reference (Python) — this
// matches the sigmas, alphas, and step math of
// `EulerDiscreteScheduler` / `EulerAncestralDiscreteScheduler`.
//
// Intentionally dependency-free so it works inside an extension
// offscreen document without any bundling.
// ─────────────────────────────────────────────────────────────

export class EulerScheduler {
  constructor({
    num_train_timesteps = 1000,
    beta_start = 0.00085,
    beta_end = 0.012,
    beta_schedule = "scaled_linear",
    prediction_type = "epsilon",
    timestep_spacing = "linspace",
  } = {}) {
    this.num_train_timesteps = num_train_timesteps;
    this.prediction_type = prediction_type;
    // "linspace" — SD 1.x default (evenly spaced incl. the final t=0).
    // "trailing" — SDXL-Lightning / Turbo: t = round(N − i·N/steps) − 1,
    //   e.g. 4 steps → [999, 749, 499, 249]. Required by distilled SDXL.
    this.timestep_spacing = timestep_spacing;

    // Betas — "scaled_linear" is what SD 1.x ships with.
    const betas = new Float32Array(num_train_timesteps);
    if (beta_schedule === "scaled_linear") {
      const s = Math.sqrt(beta_start);
      const e = Math.sqrt(beta_end);
      for (let i = 0; i < num_train_timesteps; i++) {
        const t = (s + (e - s) * (i / (num_train_timesteps - 1)));
        betas[i] = t * t;
      }
    } else {
      for (let i = 0; i < num_train_timesteps; i++) {
        betas[i] = beta_start + (beta_end - beta_start) * (i / (num_train_timesteps - 1));
      }
    }

    // alphas, alphas_cumprod
    const alphas = new Float32Array(num_train_timesteps);
    const alphas_cumprod = new Float32Array(num_train_timesteps);
    let acc = 1;
    for (let i = 0; i < num_train_timesteps; i++) {
      alphas[i] = 1 - betas[i];
      acc *= alphas[i];
      alphas_cumprod[i] = acc;
    }

    // sigmas (training): sqrt((1 - a_bar) / a_bar)
    const sigmas_train = new Float32Array(num_train_timesteps);
    for (let i = 0; i < num_train_timesteps; i++) {
      sigmas_train[i] = Math.sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
    }
    this._sigmas_train = sigmas_train;
    this._alphas_cumprod = alphas_cumprod;

    // Placeholder sigmas — filled by set_timesteps.
    this.sigmas = new Float32Array(0);
    this.timesteps = new Float32Array(0);
    this.init_noise_sigma = sigmas_train[sigmas_train.length - 1];
  }

  set_timesteps(num_inference_steps) {
    this.num_inference_steps = num_inference_steps;
    const N = this.num_train_timesteps;

    // Descending timestep indices, high→low sigma.
    const indices = new Float32Array(num_inference_steps);
    if (this.timestep_spacing === "trailing") {
      // SDXL-Lightning: round(N − i·N/steps) − 1 → [999, 749, 499, 249] for 4.
      const step_ratio = N / num_inference_steps;
      for (let i = 0; i < num_inference_steps; i++) {
        indices[i] = Math.round(N - i * step_ratio) - 1;
      }
    } else {
      // "linspace" — SD 1.x: evenly spaced from N−1 down to 0 inclusive.
      for (let i = 0; i < num_inference_steps; i++) {
        indices[i] = (N - 1) - Math.round(i * (N - 1) / (num_inference_steps - 1));
      }
    }

    // Interpolate sigmas_train at those indices (they're already int here).
    const sigmas = new Float32Array(num_inference_steps + 1);
    for (let i = 0; i < num_inference_steps; i++) {
      sigmas[i] = this._sigmas_train[indices[i]];
    }
    sigmas[num_inference_steps] = 0;  // append final zero

    // Timesteps = indices (float for passing into ONNX)
    this.sigmas = sigmas;
    this.timesteps = indices;
    this.init_noise_sigma = sigmas[0];
  }

  /**
   * img2img: noise an existing (clean) latent up to the sigma of `step_index`.
   * Euler convention — the noised sample at sigma σ is x = x0 + σ·ε, so feeding
   * the result into the denoise loop starting at the SAME step_index is exact
   * (scale_model_input/step both read this.sigmas[step_index]). Caller picks
   * step_index from the requested strength.
   */
  add_noise(latents, noise, step_index) {
    const sigma = this.sigmas[step_index];
    const out = new Float32Array(latents.length);
    for (let i = 0; i < latents.length; i++) out[i] = latents[i] + sigma * noise[i];
    return out;
  }

  /** Scale model input according to current sigma (EulerDiscrete). */
  scale_model_input(sample, step_index) {
    const sigma = this.sigmas[step_index];
    const scale = 1.0 / Math.sqrt(sigma * sigma + 1);
    const out = new Float32Array(sample.length);
    for (let i = 0; i < sample.length; i++) out[i] = sample[i] * scale;
    return out;
  }

  /** Take one Euler step. Returns new latents. */
  step(noise_pred, step_index, sample) {
    const sigma = this.sigmas[step_index];
    const sigma_next = this.sigmas[step_index + 1];

    // pred_original_sample based on prediction_type
    const pred_original = new Float32Array(sample.length);
    if (this.prediction_type === "epsilon") {
      for (let i = 0; i < sample.length; i++) {
        pred_original[i] = sample[i] - sigma * noise_pred[i];
      }
    } else if (this.prediction_type === "v_prediction") {
      const denom = sigma * sigma + 1;
      for (let i = 0; i < sample.length; i++) {
        pred_original[i] = (sample[i] / denom) - noise_pred[i] * (sigma / Math.sqrt(denom));
      }
    } else {
      throw new Error(`Unsupported prediction_type: ${this.prediction_type}`);
    }

    // derivative = (sample - pred_original) / sigma
    const dt = sigma_next - sigma;
    const out = new Float32Array(sample.length);
    if (sigma <= 0) {
      // at the end, just return pred_original
      out.set(pred_original);
    } else {
      for (let i = 0; i < sample.length; i++) {
        const derivative = (sample[i] - pred_original[i]) / sigma;
        out[i] = sample[i] + derivative * dt;
      }
    }
    return out;
  }
}

// ─────────────────────────────────────────────────────────────
// LCM scheduler — Latent Consistency Models.
//
// Matches diffusers LCMScheduler with:
//   prediction_type = "epsilon"   (UNet predicts noise, same as SD 1.5)
//   timestep_spacing = "leading"
//   set_alpha_to_one = true
//   original_inference_steps = 50
//
// Requires only 4–8 inference steps; guidance_scale must be 1.0 (no CFG).
// ─────────────────────────────────────────────────────────────
export class LCMScheduler {
  constructor({
    num_train_timesteps = 1000,
    beta_start = 0.00085,
    beta_end = 0.012,
    beta_schedule = "scaled_linear",
    original_inference_steps = 50,
  } = {}) {
    this.num_train_timesteps = num_train_timesteps;
    this.original_inference_steps = original_inference_steps;

    const betas = new Float32Array(num_train_timesteps);
    if (beta_schedule === "scaled_linear") {
      const s = Math.sqrt(beta_start), e = Math.sqrt(beta_end);
      for (let i = 0; i < num_train_timesteps; i++) {
        const t = s + (e - s) * (i / (num_train_timesteps - 1));
        betas[i] = t * t;
      }
    } else {
      for (let i = 0; i < num_train_timesteps; i++) {
        betas[i] = beta_start + (beta_end - beta_start) * (i / (num_train_timesteps - 1));
      }
    }

    const alphas_cumprod = new Float32Array(num_train_timesteps);
    let acc = 1;
    for (let i = 0; i < num_train_timesteps; i++) {
      acc *= (1 - betas[i]);
      alphas_cumprod[i] = acc;
    }
    this._alphas_cumprod = alphas_cumprod;
    this.timesteps = null;
  }

  set_timesteps(num_inference_steps) {
    // "leading" spacing over original_inference_steps, then subsample.
    const c = Math.floor(this.num_train_timesteps / this.original_inference_steps);
    const origin = [];
    for (let i = 1; i <= this.original_inference_steps; i++) origin.push(i * c - 1);
    origin.reverse(); // descending: [999, 979, …, 19]
    const skip = Math.floor(this.original_inference_steps / num_inference_steps);
    const ts = [];
    for (let i = 0; i < num_inference_steps; i++) ts.push(origin[i * skip]);
    this.timesteps = ts;
  }

  get init_noise_sigma() { return 1.0; }

  scale_model_input(sample) { return sample; }

  step(eps, step_idx, sample) {
    // UNet predicted noise → DDIM denoising step.
    const t = this.timesteps[step_idx];
    const isLast = step_idx === this.timesteps.length - 1;

    const alphaProdT    = this._alphas_cumprod[t];
    // set_alpha_to_one = true: treat alpha_prev as 1.0 on the final step.
    const alphaProdPrev = isLast ? 1.0 : this._alphas_cumprod[this.timesteps[step_idx + 1]];

    const sqrtAt  = Math.sqrt(alphaProdT);
    const sqrtOAt = Math.sqrt(1 - alphaProdT);
    const sqrtAp  = Math.sqrt(alphaProdPrev);
    const sqrtOAp = Math.sqrt(1 - alphaProdPrev);

    const out = new Float32Array(sample.length);
    for (let i = 0; i < sample.length; i++) {
      const x0 = (sample[i] - sqrtOAt * eps[i]) / sqrtAt; // predict denoised
      out[i] = sqrtAp * x0 + sqrtOAp * eps[i];             // DDIM step
    }
    return out;
  }
}

// ─────────────────────────────────────────────────────────────
// Deterministic RNG (mulberry32) for reproducible seeds.
// ─────────────────────────────────────────────────────────────
export function mulberry32(seed) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal via Box–Muller. */
export function randn(size, seed) {
  const rng = mulberry32(seed >>> 0);
  const out = new Float32Array(size);
  for (let i = 0; i < size; i += 2) {
    let u1 = rng();
    let u2 = rng();
    if (u1 < 1e-9) u1 = 1e-9;
    const mag = Math.sqrt(-2 * Math.log(u1));
    out[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < size) out[i + 1] = mag * Math.sin(2 * Math.PI * u2);
  }
  return out;
}
