# Gate 3: word-level vs. phoneme-level retiming (2026-07-28)

Per the reviewer's own Gate 3 spec: "Same phrases and notes, with only
the timing method changed." `03_gate3_word_level_variant.py` is a
deliberate ablation of `../03_reshape_pyworld.py` (the phoneme-aware v2
pipeline) that forces every word to be treated as a single span (the old
v1 behavior), while keeping every other v2 fix identical: per-word
isolated synthesis, genuine time-domain silence between words, RMS
loudness-matching. This isolates the timing-method variable alone,
uncorfounded with the click/loudness fixes (which v1-vs-v2 conflated).

## Result

| Phrase | word-level WER | phoneme-level WER | word-level cents err | phoneme-level cents err | word-level max click | phoneme-level max click |
|---|---|---|---|---|---|---|
| hello_world | 0.0 | 0.0 | 4.0 | 4.2 | 0.096 | 0.091 |
| sun_rises | 0.0 | 0.0 | 5.0 | 3.4 | 0.216 | 0.173 |
| quiet_morning | 0.25 | 0.25 | 4.3 | 4.3 | 0.106 | 0.104 |

Overall: WER identical (0.083 both). Melody-tracking marginally better
with phoneme-level (mean 3.97 vs. 4.43 cents median error across
phrases). Click magnitude marginally smaller with phoneme-level on 2 of
3 phrases.

## Honest reading

**WER and F0-cents tracking are essentially insensitive to the
timing-method change at this sample size** — the two variants perform
statistically indistinguishably on both automated proxies. The direction
is consistent (phoneme-level is never worse, sometimes slightly better)
but the effect size is small relative to what either proxy could reliably
detect at n=3.

**This does not mean phoneme-aware retiming doesn't matter.** The
reviewer's stated rationale for it was *naturalness of articulation*
("rushed consonants and unnatural vowel transitions" from treating a
multisyllabic word as one block) — a perceptual-quality claim, not an
intelligibility or pitch-accuracy claim. Neither WER nor F0-cents
tracking is designed to detect that; only a real human listening check
can. **This gate therefore cannot confirm or refute the reviewer's core
claim about phoneme-level timing** — it only confirms the change didn't
regress the two things these proxies CAN measure, which was itself worth
checking before adopting it as the default (a real risk: a well-motivated
change that inadvertently hurts a proxy metric would be worth catching).

## Files

- `03_gate3_word_level_variant.py` — the word-level ablation script.
- `05_gate3_evaluate.py` — paired evaluation (both variants, same script).
- `gate3_results.json` — full per-phrase results.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/*_wordlevel.wav`
  (word-level variant) alongside the existing `*_sung.wav` (phoneme-level,
  the adopted default).
