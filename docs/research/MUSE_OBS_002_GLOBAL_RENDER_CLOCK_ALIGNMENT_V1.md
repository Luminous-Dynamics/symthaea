# MUSE-OBS-002 — Global render-clock alignment (V1)

**Parent programme:** [#10262 — MEL-TIME-001](https://github.com/Luminous-Dynamics/symthaea/issues/10262)  
**Symbolic oracle:** [#10263](https://github.com/Luminous-Dynamics/symthaea/pull/10263)  
**Audio observation substrate:** [#11564 — MUSE-OBS-001](https://github.com/Luminous-Dynamics/symthaea/pull/11564)  
**Date:** 11 November 2026  
**Status:** research implementation; tests authored, not yet compiled or executed

## The missing clock assumption

MUSE-OBS-001 maps MIDI ticks to WAV sample frames by assuming that MIDI tick zero is WAV frame zero. That assumption is often false in actual rendering pipelines: a WAV may have leading silence or pre-roll, and the renderer/detector may expose an attack after the symbolic note-on.

V1 introduces one explicit latent variable:

> **Global audio-minus-MIDI offset:** the common sample-frame displacement that best aligns the MIDI note-on sequence with the rendered audio's broadband onset evidence.

A positive offset means that the corresponding audio onset evidence occurs later than the MIDI event. The estimate may combine file pre-roll, leading silence, and average renderer/detector attack delay. These effects cannot be separated without an external timing reference or a renderer-specific calibration fixture.

The offset is **not** a local note-timing correction, a tempo map, or a performance-rubato model.

## Implementation

Adds \`crates/domains/symthaea-muse/src/audio_clock_alignment.rs\` and exposes it behind the existing \`theory\` feature.

The estimator:

1. Revalidates the exact MIDI bytes against the part-aware manifest using MUSE-OBS-001's MIDI parser and event-accounting rules.
2. Decodes the exact WAV bytes and binds its digest, format, sample rate, channels and frame count.
3. Converts each distinct MIDI note-on tick to a sample frame using checked integer arithmetic and the manifest's integer tempo/PPQ.
4. Builds a Hann-windowed, positive spectral-flux envelope with a 2,048-frame FFT and 256-frame hop by default. Stereo channels are transformed independently and their magnitude spectra are averaged; PCM downmixing is avoided so anti-phase content cannot cancel the timing evidence.
5. Searches a caller-declared audio-minus-MIDI offset range, defaulting to −250 ms through +2,000 ms.
6. Scores candidate offsets against the full distinct MIDI onset sequence, penalizing candidates that cannot account for expected events.
7. Measures the winning correlation peak, best competing peak, peak margin, peak ratio, matched-onset coverage, and the contiguous high-support peak width.
8. Emits an explicit \`Candidate\`, \`AmbiguousCandidate\`, \`InsufficientOnsetDiversity\`, \`InsufficientAudioCoverage\`, or \`NoPositiveEvidence\` state.
9. Seals the profile to the score digest, MIDI bytes, MIDI manifest, WAV bytes, algorithm, search configuration and alignment observations.

The high-support width is an empirical correlation-peak width, **not** a statistical confidence interval. The ambiguity-ratio threshold is a fixed diagnostic heuristic and is not a calibrated probability.

## Tests authored

Seven focused tests cover:

1. recovery of a known global 200 ms audio-minus-MIDI offset on irregularly spaced note onsets;
2. silence returning no estimate;
3. simultaneous events without enough unique onset times abstaining;
4. truncated WAV coverage of only part of the expected MIDI event sequence abstaining;
5. tampered MIDI bytes being rejected before alignment;
6. WAV replacement invalidating the profile's input binding;
7. anti-phase stereo preserving timing evidence instead of cancelling during channel downmix.

The tests have not been compiled or run in the authoring environment.

## Explicit V1 boundaries

V1 does **not** establish:

- local per-note onset or release correctness;
- the correct instrument/PartId for any detected transient;
- that spectral flux corresponds to a particular note rather than a simultaneous event;
- renderer-specific attack behavior or patch qualification;
- a piecewise tempo map, expressive rubato, swing, or time-varying clock drift;
- a statistically calibrated timing uncertainty interval;
- musician-perceived groove, timing naturalness, or musical quality.

The estimator operates on a constant-tempo MIDI manifest. It deliberately does not create a second beat-to-seconds engine. Piecewise tempo maps and score/performance timing semantics remain coordinated with [#9477](https://github.com/Luminous-Dynamics/symthaea/issues/9477) and [#9476](https://github.com/Luminous-Dynamics/symthaea/issues/9476). The existing \`theory_realize\` swing/rubato timeline remains authoritative for its own realization path.

## Benchmark and industry comparison

### TimingBench — temporal control, not transcription

The August 2026 TimingBench publication evaluates 250 prompts requesting 30-second instrumental clips, including 399 timed events across instrument entries, exits, silence starts, pauses and energy increases. Its published event-within-±500-ms results are 73.2% for Taan, 12.8% for ElevenLabs v2, 8.6% for Google Lyria, 8.0% for ElevenLabs v1, and 6.8% for Suno. This is a Taan-published benchmark snapshot, not an independent universal leaderboard. Its detector evaluation reports median localization floors from 8 ms for drums to 218 ms for bass. These differences are exactly why Melothaea must report detector family, calibration and uncertainty separately rather than using one universal timestamp tolerance.

Source: https://taan.ai/research/rushing-or-dragging/

**Comparison target:** reproduce the official TimingBench prompts, event schema, detector gates and tolerance protocol before comparing headline event-success rates. The first target is to match or exceed the published 73.2% event-within-±500-ms figure under the same protocol, with 95% confidence intervals and per-event-family results. Do not infer that the V1 global offset estimator itself achieves this: it is a clock-calibration primitive, not a complete event detector.

### Multitrack audio-to-MIDI — different task, complementary baselines

MT3 established a general multi-task, multitrack transcription baseline across several instrument datasets. YourMT3+ (2024) adds hierarchical time-frequency attention, mixture-of-experts and cross-dataset stem augmentation, with evaluation across ten public datasets. Slakh2100 provides 2,100 synthesized multitrack examples with aligned MIDI, 187 patches and 34 instrument classes. These are appropriate comparators for later pitch/onset/offset/event-assignment evaluation, not direct rankings for global clock-offset estimation.

- MT3: https://research.google/pubs/mt3-multi-task-multitrack-music-transcription/
- YourMT3+: https://ieeexplore.ieee.org/abstract/document/10734819
- Slakh2100: https://www.slakh.com/

### Standard note-event metrics

For downstream local event detection, use \`mir_eval\` note matching and report:
- onset/pitch note F1 with 50 ms onset tolerance and 50-cent pitch tolerance;
- onset/offset note F1 with offset tolerance \`max(20% of reference note duration, 50 ms)\`;
- onset-only and offset-only precision/recall/F1;
- mean, median, p95 and p99 absolute onset/offset timing error;
- detector coverage, ambiguous outcomes and abstention.

The 50 ms / 50-cent / offset criterion is the documented \`mir_eval\` default. We must also report raw localization errors; a single F1 score hides timing bias and detector coverage.

Reference: https://mir-eval.readthedocs.io/latest/api/transcription.html

### Rights and dataset use

PianoCoRe-A/A* and MAESTRO can support separate score-performance research subject to their non-commercial/share-alike terms; do not silently incorporate non-commercial source material into commercial training, assets or product outputs. For commercial development, create a consented or commissioned multi-instrument corpus and retain per-source rights metadata.

PianoCoRe: https://huggingface.co/datasets/SyMuPe/PianoCoRe  
MAESTRO: https://magenta.tensorflow.org/datasets/maestro

## B0–B5 benchmark protocol

**B0 — Known global offsets.** Render the same irregular MIDI onset sequence with known leading delays and pre-rolls at 0, 5, 10, 25, 50, 100, 200, 500 and 1,000 ms. Sweep sample rate, FFT/hop, density and note register. Report frame/ms error and peak ambiguity.

**B1 — Attack/release and renderer variation.** Test different soundfonts, patches, attack times, reverb, chorus, mastering and sample rates. Distinguish file origin from average patch attack only when an external reference makes that distinction identifiable.

**B2 — Rhythmic aliases.** Use periodic/repeating note patterns, repeated sections, ostinato, metronomic percussion and dense decoys. The estimator must mark close competing peaks as ambiguous rather than reporting an unjustified precise offset.

**B3 — Truncation and coverage.** Remove the start, middle or end of the expected onset sequence; include partial files and missing events. No estimate should be returned when too few expected unique onsets are present in the audio.

**B4 — Noise and spectral confounds.** Silence, pink/white noise, sustained pads, crescendo, reverb tails, percussion-only, octave/harmonic collisions, and unrelated rhythmic material. Record false alignment candidates separately from no-evidence cases.

**B5 — Independent synchronization reference.** For physical/DAW renderer tests, use a loopback impulse or externally timestamped calibration event. Compare the proposed global offset with that independent reference. Keep the calibration reference hidden from the estimator.

## Proposed targets (not results)

- Synthetic global-offset estimate: median absolute error ≤ one analysis hop and p95 ≤ two hops on supported, non-ambiguous fixtures.
- 100% abstention on silent WAVs and truncated fixtures that do not meet minimum distinct-onset coverage.
- 100% detection of MIDI/WAV/manifest digest substitution in the frozen adversarial suite.
- 0 profile promotion to calibrated status until external synchronization data and held-out renderers establish error distributions.
- Every result reports its algorithm/configuration, exact inputs, status, competing peak and support width. No denominator excludes ambiguous or insufficient cases.

## Next integration

1. Qualify this global offset estimator against known-answer synthetic renders.
2. Add renderer/patch-specific detector calibration with externally synchronized reference events.
3. Add local onset/offset detection and uncertainty by event family.
4. Add independent note transcription/temporal detector comparisons and \`mir_eval\` adapters.
5. Bind the resulting audio timing receipt into the existing study/render artifact bundle.
6. Connect generation-side event-window conformance from #10262/#10263 and perception-side grounding from #10265, while retaining separate benchmark scorecards.
