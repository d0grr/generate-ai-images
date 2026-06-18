// Quality-steering negative prompt, prepended to the NSFW safety list on
// every generation. SD 1.5 has no innate sense of "good anatomy" — these
// terms are what pushes it away from the smeared faces / mangled hands the
// base model produces by default.
//
// IMPORTANT: CLIP truncates the negative prompt at 77 tokens (see
// sd-pipeline.js `_encode`, truncation:true). These terms MUST sit ahead of
// the long NSFW list, otherwise they're silently dropped past the cutoff.
// Keep the list tight and front-load the highest-impact terms (anatomy /
// face / eyes) so they always land inside the window.
export const QUALITY_NEGATIVE_PROMPT = [
  // anatomy / faces — the terms that actually fix smeared portraits
  "deformed", "disfigured", "bad anatomy", "poorly drawn face",
  "distorted face", "asymmetric eyes", "bad eyes", "cross-eyed",
  "mutated", "extra limbs", "extra fingers", "fused fingers", "bad hands",
  // image quality
  "blurry", "lowres", "jpeg artifacts", "ugly",
].join(", ");
