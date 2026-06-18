// Appended to every negative prompt to steer the model away from
// explicit, violent and otherwise prohibited content.
export const NSFW_NEGATIVE_PROMPT = [
  // explicit / sexual
  "nsfw", "nude", "nudity", "naked", "topless", "bottomless",
  "explicit", "sexually explicit", "explicit content", "adult content",
  "pornographic", "pornography", "porn", "hentai", "erotic", "erotica",
  "xxx", "r18", "r-18", "18+",
  "lewd", "suggestive", "indecent", "obscene", "vulgar",
  "sexual", "sexual content", "sexual activity",
  "genitals", "penis", "vagina", "vulva", "anus",
  "nipples", "areola", "bare chest", "exposed body",
  "fetish", "bdsm", "bondage",
  "lingerie", "underwear provocative",
  // violence / gore
  "gore", "gory", "bloody", "graphic violence",
  "torture", "mutilation", "decapitation", "dismemberment",
  "snuff", "death scene",
  // child safety
  "lolicon", "shotacon", "underage", "minor sexualized",
].join(", ");
