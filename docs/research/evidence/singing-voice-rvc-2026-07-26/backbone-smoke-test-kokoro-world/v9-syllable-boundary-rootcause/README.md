# v9: phoneme-weighted syllable split -- root-causes the fricative_heavy misplacement, but does NOT fix WER (2026-07-28)

Follow-up to v8's exit-crossfade ablation, which found a real, previously
undetected regression: `fricative_heavy` ("she sells seashells by the
seashore") transcribes as "She sells T-shirts by the T-shirt" on
v7/v7b/v8 Arms A/B/C alike -- present identically regardless of exit-
crossfade policy, so the cause had to be upstream of the exit crossfade
mechanism. This pass investigates and finds a real, ground-truth-
confirmed root cause for ONE contributing defect -- but confirms fixing
it does not resolve (and mildly worsens) the WER regression, so the true
cause of the ASR mishearing remains open.

## Root cause found (confirmed via direct measurement, not inference)

The original `main()` (`03v8_exit_crossfade_ablation.py:565`, inherited
unchanged from v3 onward) splits a word's real (forced-aligned) natural
duration EQUALLY BY SYLLABLE COUNT:
`per_syll_natural_frames = max(1, remaining_frames // n_syll)`
-- ignoring each syllable's actual phoneme content. For "seashells"
(syllable 1 "sea" = /s,i/, 2 phonemes; syllable 2 "shells" = /S,E,l,z/,
4 phonemes) and "seashore" (similar imbalance), this places the
"sea"/"shells" (or "sea"/"shore") syllable boundary well into the true
/sh/ frication region instead of before it.

Confirmed by directly measuring the spoken source audio's high-band
(>=3kHz) energy fraction around the modeled boundary
(`fricative_heavy_spoken.wav`, fs=24000):

| Word | Modeled /sh/ raw-extraction span (v8/v7b) | True frication content there |
|---|---|---|
| seashells | samples 23400-24480 | hf_frac drops from ~0.7-0.8 (true frication, at samples ~22800-23280) down to ~0.07-0.1 (vowel-like) by the time this span even STARTS |
| seashore | samples 40320-41760 | true frication is genuinely present at the span's start but tapers to vowel-like by ~40900, roughly half the modeled span is already past the true consonant |

In both cases the modeled raw-consonant extraction window is mostly (or
entirely, for seashells) capturing VOWEL ONSET material mislabeled as
consonant, while the true frication noise gets absorbed into the
PRECEDING WORLD-synthesized "i" (the "sea" vowel) group's source window
-- contaminating that vowel's periodic (F0-driven) resynthesis with
noise it was never meant to carry.

## Fix implemented

`03v9_phoneme_weighted_syllable_split.py`: replaces the equal-per-
syllable-count split with a phoneme-count-weighted split (vowel weight
1.5, consonant weight 1.0 -- a round, disclosed choice, not fit to this
specific example) so syllables with more/heavier phonemes get a
correspondingly larger share of the word's natural frame budget. This
is the ONE changed variable versus v8 Arm A (current exit policy) --
exit-crossfade policy, F0 rules, voicing classification, gestures all
unchanged.

**Confirms the fix moves the raw-extraction boundary onto real
frication, precisely as intended:**

| | seashells /S/ span | mean hf_frac (true frication) | seashore /S/ span | mean hf_frac |
|---|---|---|---|---|
| v8 baseline | 23400-24480 | **0.104** (vowel-dominated) | 40320-41760 | **0.253** |
| v9 (fixed) | 21960-23280 | **0.809** (genuine frication) | 39480-40920 | **0.689** |

This is a large, unambiguous, ground-truth-confirmed correction of the
acoustic defect the fix targeted.

## But WER does not improve -- an honest negative result

Re-running WER (faster-whisper `base`, int8, fresh transcription, same
model instance) on all 10 Gate-2 phrases, v9 vs. a freshly re-verified
v8 Arm A baseline on the 3 obstruent-heavy phrases:

| Phrase | v8 Arm A WER (reverified) | v9 WER | Direction |
|---|---|---|---|
| consonant_clusters | 0.000 ("splashed" correct) | **0.250** ("flashed" -- wrong) | **worse** |
| fricative_heavy | 0.333 ("T-shirts"/"T-shirt") | **0.667** ("He sells t-shirts by this t-shirt" -- more words wrong, "she"->"He", "the"->"this") | **worse** |
| phrase_final_stops | 0.143 | 0.143 | unchanged |

**The targeted mishearing ("seashells"/"seashore" -> "t-shirts"/
"t-shirt") is NOT fixed** despite the raw-extraction span now containing
far more of the true consonant. This means the confirmed acoustic defect
above is real, but it is not (or not solely) the cause of the ASR
confusion -- fixing it did not help, and other phrases regressed.

**A plausible (NOT yet confirmed) secondary mechanism, found while
investigating the non-result**: `synthesize_word`'s `MIN_SYLLABLE_DUR_S`
(280ms) floor absorbs the natural-duration change for short syllables.
"sea"'s syllable in both v8 baseline and v9 renders to an IDENTICAL
280ms output (s=60ms + i=220ms in both) because its stretched target
(`natural_dur_s * STRETCH`) falls below the floor either way -- meaning
v9's smaller natural allocation for "sea" (since "shells"/"shore" now
correctly claims more of the word's budget) requires a LARGER
natural-to-target stretch ratio for the "i" vowel to still fill the same
280ms floor. This could plausibly introduce more WORLD frame-resampling
smear into "sea"'s own vowel -- a new, different distortion source the
original (wrong) boundary didn't have. Not confirmed; would need its own
isolated test (e.g. re-run with the floor removed or raised, or track
natural-vs-target stretch ratio directly).

**consonant_clusters' regression ("splashed"->"flashed") is NOT
explained by the syllable-weighting mechanism at all** -- "splashed" is
a single syllable, and single-syllable words are mathematically
unchanged by this fix (`weight_total` reduces to the one syllable's own
weight, so `per_syll_natural_frames` for `n_syll=1` equals
`remaining_frames` exactly as before). The regression must therefore
come from an indirect effect (e.g. a shift in `cumulative_output_sample`
from OTHER multi-syllable words in the phrase changing this word's
absolute position/context in the concatenated clip, which Whisper's
transcription could plausibly be sensitive to even with byte-identical
local audio) -- not investigated further this pass.

## What this does and doesn't establish

**Does establish**: the equal-per-syllable-count duration split is a
real, confirmed defect with a precise, ground-truth-measurable
consequence (misplaced consonant/vowel boundary, contaminated vowel
resynthesis) for syllables with unequal phoneme counts. This finding
stands independent of the WER outcome.

**Does NOT establish**: that this defect is the (or a) cause of the
fricative_heavy ASR mishearing, or that phoneme-count weighting is the
right fix. The WER result is a genuine negative for this specific fix,
not an inconclusive one -- it was measured, pre-committed to the full
declared test set (per the standing rule in
`feedback_verify_claim_across_full_test_set.md`), and it regressed on
2 of 3 phrases tested.

## Recommendation

**Do not promote v9.** The underlying acoustic insight (syllable
duration should be allocated by phonetic content, not raw syllable
count) is very likely still correct in principle, but this specific
heuristic fix doesn't clear WER and the `MIN_SYLLABLE_DUR_S` floor
interaction needs to be understood before trying a refined version.
Real forced-alignment-based source phone boundaries (still not
attempted anywhere in this arc) would sidestep this whole class of
heuristic-boundary problem and remains the more principled fix -- this
result is a data point in favor of prioritizing that over further
heuristic tuning of the syllable-split formula.

## Not yet done

- Confirming or ruling out the `MIN_SYLLABLE_DUR_S`-floor stretch-ratio
  hypothesis above.
- Explaining the consonant_clusters "splashed"->"flashed" regression.
- Real forced-alignment-based source phone boundaries.
- The human listening check -- still the standing, most important item.

## Files

- `03v9_phoneme_weighted_syllable_split.py` -- the one-variable fix
  (phoneme-count-weighted syllable split) on top of v8 Arm A.
- `fricative_heavy_sung_v9_lineage.json` -- exact lineage for the v9
  render of `fricative_heavy` (Gate-2 phrase set).
- `fricative_heavy_sung_v8_baseline_lineage.json` -- the v7b/v8-Arm-A
  baseline lineage for the same phrase (renamed from v7b's export;
  v8 Arm A reproduces v7b byte-for-byte, confirmed in v8's own README),
  for direct before/after comparison.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v9_syllable_split_fix/*_sung_v9.wav`
  (gitignored, not duplicated here, per this bundle's convention).
