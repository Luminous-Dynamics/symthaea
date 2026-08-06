# Melody-metric audit (2026-07-28)

Construct-validity check on `04_evaluate.py::melody_tracking_score`, the
metric behind this bundle's headline "median 3.4–4.3 cents" melody claim.

Motivated by this project's standing pattern of headline numbers turning
out to be measurement artifacts (the `dz=-33` noise-calibration mismatch,
the ECE bin-sparsity artifact, `consciousness_level`'s wall-clock
dependence). The suspicion under test: does a low cents-error demonstrate
that the render **follows the melody**, or only that its voiced frames sit
near **some** note of the scale?

## The structural concern

`melody_tracking_score` scores each voiced frame against its *nearest*
target note — `np.min(...)` over the whole melody — which is order-blind.
`03_reshape_pyworld.py:176` sets every voiced frame of a word to a constant
target note (`r_f0 = np.where(voiced, target_hz, 0.0)`, with
`assert len(words) == len(melody)`). So the metric may largely be measuring
WORLD round-tripping the F0 it was just handed.

## Part 1 — negative controls (`melody_metric_control.py`)

Four arms synthesized through one identical WORLD analyze→replace-F0→
resynthesize path from `hello_world_spoken.wav`; only the F0 policy differs.
Harness validated first by reproducing the recorded `results.json` numbers
for `hello_world_sung.wav` bit-for-bit (4.2c / 0.888 / 13.68 st).

| Arm | median cents err | frac within 50c | observed range |
|---|---|---|---|
| CORRECT (261.63 → 392.00) | 16.7c | 0.639 | 31.05 st |
| SCRAMBLED (392.00 → 261.63) — objectively wrong melody | **9.7c** | **0.717** | 29.07 st |
| MONOTONE (261.63 throughout) — no melody at all | **5.5c** | **0.896** | 3.36 st |
| SPOKEN (untouched natural F0) | 306.5c | 0.113 | 8.65 st |

**A monotone drone scores best on both headline numbers; a scrambled
melody beats the correct one.** Mechanism: singing a real melody requires
pitch *transitions*, and every transition frame falls between two targets
and incurs large error. A monotone has no transitions, so nearly every
frame sits exactly on a target. On this test the metric is *anti*-correlated
with melodic correctness.

The metric does measure one real thing — the SPOKEN row separates
"F0 quantized to the scale at all" (≈4c) from "natural speech" (≈306c).
That is not melody following.

Caveat: the CORRECT arm here (16.7c) is a cruder single-pass resynthesis
than the real pipeline (which does per-word isolated synthesis, true
inter-word silence, and phoneme-aware retiming), so its absolute numbers do
not represent pipeline quality. The finding is the *ordering* between arms
within one identical path, which is internally valid.

## Part 2 — order-sensitive re-scoring (`melody_metric_ordered.py`)

Run on the **real recorded sung renders**. Splits voiced frames into
`len(melody)` contiguous groups and scores group *i* against the note it was
supposed to sing.

| Phrase | order-blind (shipped) | order-sensitive | worst per-note |
|---|---|---|---|
| hello_world | 4.2c / 0.888 | 4.2c / 0.874 | 5.7c |
| sun_rises | 3.4c / 0.907 | 3.5c / 0.830 | 7.9c |
| quiet_morning | 4.3c / 0.900 | 4.5c / 0.841 | 22.1c (top note, 523.25 Hz) |

Every note's observed median lands within ~1 Hz of its target (261.5 vs
261.63; 392.1 vs 392.00; 439.6 vs 440.00; 523.0 vs 523.25).

## Conclusion — both things are true

1. **The metric is broken.** It cannot distinguish a correct melody from a
   monotone or a scrambled one, and would silently pass a
   collapse-to-drone regression. It must not be used as a regression gate
   in its current form.
2. **The pipeline's melody result is real, not an artifact.** An
   independent, stricter, order-sensitive metric confirms the right notes
   in the right order. The original suspicion motivating this audit was
   **refuted**.

## Recommended fix

Replace `min`-over-all-targets with per-group scoring against the
*expected* note, and keep MONOTONE and SCRAMBLED as permanent negative
controls that the metric **must fail**. A metric that cannot fail them is
not measuring melody.

Known limitation of the version here: it approximates word boundaries by
splitting voiced frames into equal contiguous groups rather than reading the
true forced-alignment boundaries. A production version should use the real
alignment.

## Part 3 — Gate 2 hard suite, true order-sensitive metric (`gate2_melody_ordered.py`)

Gate 2 recorded **WER only**; melody accuracy on the hard 10-phrase suite had
never been measured. This closes that gap and answers the question WER alone
cannot: when intelligibility collapses, does the pitch mechanism collapse too?

**Method improvement over Part 2.** `gate2_03_reshape.py` synthesizes each word
in complete isolation and joins them with `GAP_S = 0.06` of *genuine
time-domain silence* — literally `np.zeros` (line 193) — and the final RMS
scaling is a constant multiply, so those samples stay exactly zero. True
per-word output spans are therefore directly recoverable by splitting on
exact-zero runs. No equal-split approximation, no G2P reconstruction.
**Segment count matched note count for 10/10 phrases**, which validates the
segmentation independently.

| Phrase | WER | order-blind | **ordered** | frac<50c | worst note |
|---|---|---|---|---|---|
| positive_control | 0.000 | 4.1c | 3.9c | 0.880 | 15.6c |
| conversational | 0.167 | 3.3c | 2.9c | 0.887 | 18.5c |
| repeated_syllables | 0.200 | 2.4c | 2.4c | 0.926 | 7.5c |
| rapid_letter_names | **1.000** | 3.0c | **2.4c** | 0.831 | 49.9c |
| phrase_final_stops | 0.143 | 2.2c | 2.0c | 0.946 | 5.8c |
| fricative_heavy | **0.667** | 4.3c | 4.0c | 0.742 | 38.8c |
| consonant_clusters | 0.250 | 6.8c | **7.2c** | 0.798 | 13.7c |
| long_sustained_vowels | **0.800** | 4.1c | 3.7c | 0.873 | 7.7c |
| short_unstressed | 0.000 | 1.6c | 1.4c | 0.945 | 4.3c |
| semantically_unusual | 0.200 | 3.7c | 3.7c | 0.885 | 8.4c |

**Pitch and intelligibility are decoupled.**

- high-WER group (≥0.5, n=3): mean ordered median **3.4c**
- low-WER group (<0.5, n=7): mean ordered median **3.4c** — identical
- Pearson r(WER, ordered melody error) = **+0.060** (n=10) — no relationship

`rapid_letter_names` is the clearest case: WER 1.0 (Whisper hears "Oh, Lucy!"
— total intelligibility failure) with an ordered melody median of 2.4c, among
the *best* in the suite. Conversely `consonant_clusters` has the worst melody
median (7.2c) with a good WER (0.25) — the opposite of the expected pattern.

**Centre vs spread, all 57 notes across the 10 phrases:**

| | median | max |
|---|---|---|
| **centre** error (\|observed median − target\|) | **1.2c** | 19.5c |
| **spread** (median per-frame \|err\|) | 3.3c | 49.9c |

The high-"worst note" outliers are not mistargeting: e.g. `rapid_letter_names`'
final note has 49.9c median per-frame error but an observed median of 294.2 Hz
against a 293.66 Hz target (~3c). The pitch is centred correctly; the variance
is onset/offset transition frames, which dominate proportionally in short
segments (r(voiced_frames, spread) = −0.332, n=57).

### What this establishes

1. **F0 targeting is essentially exact** — median 1.2 cents off target across
   57 notes on the *hardest* phrase set. That is well below the ~5-cent human
   just-noticeable difference.
2. **It is content-independent** — unaffected by fricatives, clusters, letter
   names, or sustained vowels, and uncorrelated with whether the words survive.
3. **Therefore every remaining intelligibility failure lives in the
   timing / consonant-realization / event-extraction layer, not the pitch
   path.** This independently validates the v3–v9 and acoustic-event-audit
   line of work as correctly targeted, and says the F0 mechanism should be
   left alone.
4. **It sharpens the "too quantized" critique.** Exact-to-1.2-cents is *more*
   precise than a human singer ever is. The pitch is not merely quantized in
   principle — it is measurably inhuman in practice. Natural micro-variation
   is likely to matter more perceptually than any further accuracy work.

Limitations: n=10 phrases, single deterministic trial, one voice, one melodic
contour reused across most phrases. WER figures are the recorded Gate 2 values
(Whisper `base`/int8), not re-derived here.

## Reproduce

```
cd /var/lib/symthaea/training-runs/diffsinger && source ./env.sh
./venv/bin/python melody_metric_control.py
./venv/bin/python melody_metric_ordered.py
```
(`env.sh` supplies the NixOS `libz`/`libstdc++` `LD_LIBRARY_PATH` fix.)
Captured output: `control_output.txt`, `ordered_output.txt`.
