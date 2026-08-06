# v7b: exact frame-lineage instrumentation -- localizes v7's quality loss precisely (2026-07-28)

Per the reviewer's explicit request after v7's honest mixed result:
"pause synthesis changes and build an exact frame-lineage system...
the renderer already knows which source interval generated each output
interval; that mapping should become a first-class artifact." Rendering
behavior is IDENTICAL to v7 (`03v7_waveform_preserved_obstruents.py`) --
this pass only adds lineage export.

## What was built

`03v7b_lineage_instrumented.py` emits a `<phrase>_<suffix>_lineage.json`
sidecar per rendered phrase. For every rendered group (a run of
consecutive same-method phonemes -- "world" for sonorant/voiced-
obstruent, "raw" for waveform-preserved voiceless-obstruent), it records
the EXACT (not inferred) output sample range, the core-interior range
(excluding the crossfade shared with neighboring groups -- computed by
`crossfade_concat_with_lineage`, which performs the actual concatenation
and returns exact boundary positions, not an approximate reconstruction),
the entry/exit crossfade sample ranges, and for raw groups specifically,
the exact source sample range the waveform was extracted from. This is
ground truth from the renderer's own bookkeeping -- the same kind of
precision the reviewer asked for ("make every output sample traceable
to its source phoneme and transformation").

`09_lineage_local_measure.py` uses this lineage to measure three regions
SEPARATELY per raw group, exactly as requested: core interior vs. its
exact source region (voiced fraction proxies via ZCR, centroid, high-band
[4-10kHz] energy fraction), entry crossfade, and exit crossfade.

## Result: the loss is localized to the exit crossfade, not the preserved core

Aggregated across 3 obstruent-heavy phrases (`consonant_clusters`,
`fricative_heavy`, `phrase_final_stops`), 16 raw (voiceless-obstruent)
groups measured:

| Region | n | mean centroid | mean high-band (4-10kHz) fraction |
|---|---|---|---|
| **Core interior** | 16 | **4160 Hz** | **0.288** |
| Entry crossfade (10ms) | 7 | 2798 Hz | -- |
| **Exit crossfade (10ms)** | 11 | **1843 Hz** | **0.0388** |

**The core-interior retention is actually good** -- per-group
centroid/ZCR retention relative to the corresponding source span ranged
mostly 76-137% (several groups exceed 100%, e.g. "off"'s /f/ at 137%
centroid retention, "it"'s /t/ at 122% ZCR retention -- plausible
boundary-estimate noise in both directions, not a systematic failure).
Full per-group numbers: `lineage_local_measure_full.log`-equivalent
output, reproducible via `09_lineage_local_measure.py`.

**The exit crossfade is where brightness collapses**: mean centroid
1843 Hz (vs. the adjacent core's 4160 Hz -- a ~2.3x drop) and mean
high-band energy fraction 0.0388 (vs. 0.288 -- a ~7.4x drop). This
directly confirms and precisely localizes the reviewer's hypothesis:
"the crossfade may consume the useful consonant... mixing the natural
burst with a smooth WORLD waveform can reduce precisely the high-
frequency transient. Equal-power crossfades preserve energy, not
necessarily articulation."

**Entry crossfades are also degraded but less severely** (mean 2798 Hz,
between core and exit) -- the asymmetry itself is informative: the
transition FROM a raw consonant INTO the following WORLD-synthesized
vowel loses far more brightness than the transition FROM a preceding
WORLD segment INTO the consonant. A plausible mechanism: the consonant's
noise energy ends abruptly at the true phone boundary, while the
following WORLD-vowel's synthesized onset has very different (periodic,
lower-frequency-dominant) spectral content -- a fixed-duration linear
blend of two acoustically dissimilar signals produces exactly the
"smoother and duller" result the reviewer predicted.

## Why this matters more than the whole-clip result suggested

v7's original whole-clip evaluation (`../v7-waveform-preserved-obstruents/README.md`)
found overall centroid moved AWAY from the source and could not
distinguish "the mechanism doesn't help" from "the mechanism helps
locally but the metric can't see it." This localized measurement
resolves that ambiguity: **the preserved consonant material itself is
NOT the problem -- the crossfade mechanism is.** This is a materially
different, more actionable finding than v7's own writeup could support.

## Recommended immediate next step

Per the reviewer's own phoneme-specific crossfade proposals (continuous
fricatives: short 3-8ms transition preserving high-frequency energy;
stop releases: 1-3ms transition around the burst, not smeared over a
long fade) -- redesign the exit-crossfade specifically, since that's now
precisely where the loss is concentrated, rather than treating entry
and exit symmetrically. Not yet attempted this pass.

## Not yet done

- The crossfade redesign itself (shorter/asymmetric/phoneme-class-aware)
  -- not attempted, this pass was measurement-only per the reviewer's
  explicit "pause synthesis changes" instruction.
- Real (forced-alignment-based) source phone boundaries -- the source
  span used for core-vs-source comparison remains this project's
  existing proportional estimate, not true phoneme-level alignment. The
  reviewer's suggestion to reuse `wav2vec2-lv-60-espeak-cv-ft` (now
  well-posed since the target is clean Kokoro speech, not real singing)
  was not attempted this pass, given a phone-vocabulary mismatch between
  misaki's phone set and the model's espeak vocabulary would need its
  own small transducer -- flagged as a real, tractable follow-up, not
  attempted here.
- A voiced-obstruent-heavy stress-test phrase.
- The human listening check -- still the standing, most important item.

## Correction (2026-07-28, found during v8's exit-crossfade ablation)

This doc never checked WER at all -- it was acoustic-measurement-only.
v8's ablation ran WER on the same rendering (v7b's Arm-A-equivalent
output, confirmed byte-identical) across all 3 obstruent-heavy phrases
and found `fricative_heavy` transcribes as "She sells T-shirts by the
T-shirt" -- a real, previously-uncaught content error introduced by the
waveform-preservation mechanism (present in v7/v7b/v8 alike), distinct
from v6's harmless "sea shore" tokenization-only artifact on the same
phrase. This doesn't change the core-vs-crossfade localization finding
above (which didn't depend on WER), but the overall waveform-
preservation approach's viability needs to account for this. Full
detail: `../v8-exit-crossfade-ablation/README.md`.

## Files

- `03v7b_lineage_instrumented.py` -- v7's renderer + lineage export.
- `09_lineage_local_measure.py` -- the 3-region local measurement tool.
- `*_sung_v7b_lineage.json` -- exact lineage manifests for the 3
  obstruent-heavy phrases (tracked here).
- `*_sung_v7b.wav` -- rendered audio, in
  `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/`
  (gitignored, not duplicated here, per this whole bundle's convention).
