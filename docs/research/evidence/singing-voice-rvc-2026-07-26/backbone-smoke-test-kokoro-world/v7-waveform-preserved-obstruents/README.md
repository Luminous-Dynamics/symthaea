# v7: waveform-preserved voiceless obstruents -- the 5th arm (2026-07-28)

Per the reviewer's proposed 5th arm: "preserve the original Kokoro
consonant waveform... transform only the vowel and sonorant regions...
crossfade the two." Rationale: `F0=0` is necessary but may not be
sufficient -- WORLD's resynthesized spectral envelope/aperiodicity for
a forced-unvoiced obstruent can still sound smoothed/muffled relative
to preserving the source's own turbulence and transient structure.

On top of the locked v6 control (`03v6_LOCKED_control.py`, unmodified):
sonorant and voiced-obstruent phonemes render through WORLD parameters
exactly as in v6 (unchanged). Voiceless-obstruent phonemes instead use
the ORIGINAL Kokoro waveform for that phoneme's estimated natural time
span directly (resampled in time only if the target/natural duration
ratio is far from 1.0), with a 10ms linear crossfade at every boundary
between a raw-waveform run and a neighboring WORLD-synthesized run.

## Result

### consonant_clusters

| Variant | WER | voiced (harvest/dio) | centroid | max click |
|---|---|---|---|---|
| spoken reference | -- | 0.551 / 0.387 | 3559 Hz | 0.217 |
| B_mask_only (ablation) | 0.000 | 0.650 / 0.491 | 2755 Hz | 0.223 |
| v6_voiced_split | 0.000 | 0.786 / 0.516 | 2942 Hz | 0.218 |
| **v7_waveform** | 0.000 | 0.712 / 0.503 | **2654 Hz** | **0.203** |

### hello_world (negative control)

| Variant | WER | voiced (harvest/dio) | centroid | max click |
|---|---|---|---|---|
| spoken reference | -- | 0.346 / 0.301 | 3439 Hz | 0.294 |
| v6 | 0.000 | 0.873 / 0.804 | 2069 Hz | 0.094 |
| v7 | 0.000 | 0.842 / 0.817 | 2074 Hz | 0.097 |

## An honest, mixed result -- not a clean win

WER stays perfect (0.0) and no new severe artifact was introduced (max
click actually improves slightly, 0.203 vs v6's 0.218; crossfading is
functioning cleanly). **But the acoustic-brightness hypothesis is NOT
confirmed by this measurement**: v7's centroid (2654 Hz) is LOWER than
v6's (2942 Hz) on consonant_clusters -- moving further from the spoken
source (3559 Hz), the opposite of the intended direction. This is
reported precisely rather than reframed as a partial win it may not be.

**Most plausible explanations, not yet distinguished**:

1. **The extracted raw waveform may not precisely capture the true
   consonant burst/frication.** This project's duration model remains a
   PROPORTIONAL ESTIMATE from nominal phoneme durations (no true
   syllable/phoneme-level forced alignment exists) -- if the estimated
   boundary for e.g. "st" in "strong" doesn't line up with where the
   actual noise burst occurs in Kokoro's real audio, the extracted
   segment could contain more silence/vowel-bleed and less true
   turbulence than a correctly-placed one would.
2. **Whole-clip aggregate centroid is too coarse to detect a change
   confined to ~200-300ms of a ~2-second clip.** This is exactly the
   reviewer's own repeated point ("measure locally, not globally...
   this prevents vowel stretching from overwhelming the result") --
   still not implemented, and this result is a direct demonstration of
   why it matters: a real, localized change (or non-change) can be
   invisible or misleading in an aggregate statistic dominated by the
   unchanged majority of the clip.
3. **Crossfade dilution**: a 10ms crossfade on each side of an already-
   short (~40-60ms) raw consonant segment blends up to 20ms of it with
   the neighboring (already-muffled) WORLD-synthesized material,
   potentially diluting whatever brightness the raw content had.

## What this does and doesn't establish

**Does NOT establish**: that waveform preservation is a bad idea, or
that it fails to help. The measurement used here cannot distinguish
"the mechanism doesn't help" from "the mechanism helps locally but the
metric can't see it" -- and reason #2 above (aggregate metric coarseness)
is a real, demonstrated limitation independent of whether the mechanism
itself works.

**Does establish**: the engineering (crossfaded raw-waveform + WORLD-
parameter hybrid rendering) works mechanically -- runs cleanly, produces
no new severe artifacts, preserves perfect WER. Whether it actually
improves perceived naturalness is unresolved by any metric available so
far, more than ever pointing at the two still-missing pieces: per-
phoneme-span localized measurement, and the human listening check.

## Recommended immediate next step

**Build per-phoneme-span localized measurement before iterating
further on aggregate whole-clip metrics.** This has now been
independently flagged by the reviewer multiple times and directly
demonstrated as necessary by this result -- continuing to compare v6/v7
(or any future variant) on whole-clip statistics risks drawing the
wrong conclusion in either direction. Concretely: instrument the
renderer to report each phoneme entry's actual output sample range, then
measure voiced-fraction/centroid/ZCR/aperiodicity within just those
spans, comparing the SAME phoneme's rendering across variants directly.

## Not yet done

- Per-phoneme-span localized measurement (see above -- the clear next
  priority).
- Verification of whether the raw-waveform extraction is actually
  capturing the intended consonant content (e.g. by inspecting a
  spectrogram of just the extracted span) -- not done.
- Crossfade duration/shape tuning (10ms linear was a single untried
  starting value, not swept).
- The human listening check -- still the standing, most important item.

## Correction (2026-07-28, found during v8's exit-crossfade ablation)

This doc's "WER stays perfect (0.0)" claim was verified on
`consonant_clusters` only -- this README never tested `fricative_heavy`
or `phrase_final_stops` for WER at all (an oversight in this doc, not
just an unverified extrapolation). Testing all 3 obstruent-heavy
phrases during v8's ablation surfaced a real, previously-uncaught
problem: **`fricative_heavy` transcribes as "She sells T-shirts by the
T-shirt" on this v7 render** -- a genuine content error, not the "sea
shore" vs "seashore" tokenization-only artifact reported for v6 (which
does NOT have waveform preservation). Confirmed byte-identical audio
and deterministic across repeated Whisper calls -- this is a real
regression introduced by the waveform-preservation mechanism itself,
not a measurement fluke. `phrase_final_stops`'s WER (0.143, dropped
final "it") was first measured during v8's ablation too, not previously
reported here. Full detail: `../v8-exit-crossfade-ablation/README.md`.

## Files

- `03v6_LOCKED_control.py` -- the frozen v6 control for this comparison.
- `03v7_waveform_preserved_obstruents.py` -- the v7 renderer.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/*_sung_v7.wav`.
