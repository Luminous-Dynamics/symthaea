# v3: syllable-level note mapping + basic vocal gestures (2026-07-28)

Per the reviewer's recommended engineering sequence (steps 2+3), on top
of the locked control (`03v0_LOCKED_control_backbone.py`, byte-identical
to `../03_reshape_pyworld.py` -- explicitly NOT modified, per "lock the
current implementation as the control" / "Kokoro-WORLD Exact Backbone
v0"). This is a bounded, single increment -- NOT the reviewer's full
8-layer expression roadmap, per their own "don't add all expressive
controls simultaneously" instruction.

## What changed

1. **Syllable, not word, note mapping** (step 2). Multisyllabic words
   ("rises", "valley", "morning") were previously one note per whole
   word. Now: maximal-onset syllabification (mirrors
   `symthaea-muse::singing_bridge::syllabify()` exactly, translated to
   Python against Kokoro's own misaki phoneme output) splits each word
   into its real syllables, one note per syllable. Word-level MMS_FA
   alignment remains the only ground-truth timing fact (no syllable-
   level forced alignment exists); natural per-syllable sub-durations
   are a proportional estimate, disclosed not hidden, same limitation
   class as v2's per-phoneme split, one level deeper.
2. **Inter-syllable pitch glide** (step 3, conservative start). A 40ms
   glide between consecutive syllables of the SAME word only --
   syllables in different words are separated by real silence, so
   there's nothing continuous to glide across.
3. **Gated vibrato** (step 3). `VIBRATO_RATE_HZ=5.5`,
   `VIBRATO_DEPTH_CENTS=30` -- reused verbatim from
   `singing_bridge.rs`'s own constants for continuity, not re-derived.
   Applied ONLY within a vowel segment exceeding 150ms, and only its
   inner 40%-90% (skipping onset/release), per "vibrato only after the
   vowel stabilizes."

Not attempted this pass, per the reviewer's own layering discipline:
phrase-level dynamics, breath modeling, any neural post-render.

## Result

| Phrase | v2 (word, no gestures) WER | v3 (syllable+gestures) WER | v2 median cents err | v3 median cents err | v2 frac-within-50c | v3 frac-within-50c |
|---|---|---|---|---|---|---|
| hello_world (3 syll) | 0.0 | 0.0 | 4.2 | 10.4 | 0.888 | 0.921 |
| sun_rises (9 syll) | 0.0 | 0.0 | 3.4 | 13.1 | 0.907 | 0.870 |
| quiet_morning (6 syll) | 0.25 | 0.25 | 4.3 | 7.7 | 0.900 | 0.919 |

Overall WER unchanged (0.083 both) — the transcripts are near-identical
(`v3_results.json` has full hypotheses).

## The cents-error increase is expected, not a regression

Median cents-error-to-nearest-target-note rose 2.5-4x with v3. **This is
the correct, deliberate signature of successfully adding vibrato/glide,
not a quality regression**: both mechanisms exist specifically to move
pitch away from an instantaneous, hard-quantized lock onto the target
frequency -- that deviation is the whole point (the reviewer's own
diagnosis was that v2's pitch was "almost unnaturally perfect" /
"mathematically flat semitone centers"). The magnitude matters more than
the direction: 7.7-13.1 cents is only ~8-13% of a semitone, well within
a musically credible "stable center with natural movement" range, not
"wandering off pitch" -- and `frac_within_50c` (0.87-0.92) confirms the
bulk of frames stay close to the target throughout. **This cannot be
confirmed as a genuine naturalness improvement by any automated proxy
here** -- only a real listening check (still the standing open item)
can tell whether it actually sounds more like singing or just
differently synthetic.

## Honest limitations, not yet addressed

- Same standing caveat as every prior gate: no human listening check.
- Consonants are still repitched to the syllable's flat target (only
  vowels get gestures) -- the reviewer's note that consonants "are not
  normally voiced pitch-bearing material" the same way vowels are isn't
  fully addressed; a voiced consonant like the /z/ in "rises" still gets
  locked to the target pitch, just now within a smaller, per-syllable
  frame budget.
- Onset/release at word boundaries (not mid-word syllable boundaries)
  still uses v2's simple ramp-based envelope, not a genuine note-attack
  gesture (approach/settle/sustain/release) — only inter-syllable
  transitions got the new glide treatment this pass.
- Phrase-level dynamics, breath placement, and emotional-intensity
  envelopes (reviewer's steps 4+) are entirely unstarted.
- Tested on the original 3-phrase set only, not yet re-run against the
  10-phrase Gate 2 hard-phonetic suite.

## Files

- `03v0_LOCKED_control_backbone.py` — the frozen control (v2, unchanged).
- `03v3_syllable_gestures.py` — the new syllable+gesture pipeline.
- `06_v3_evaluate.py` — paired v2-vs-v3 evaluation.
- `v3_results.json` — full per-phrase transcripts/metrics.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/*_sung_v3.wav`.
