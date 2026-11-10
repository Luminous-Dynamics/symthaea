# Score-independent singing pitch tracker benchmark protocol

**Date:** 10 November 2026  
**Status:** standalone scorer contract and synthetic tests; no real-vocal result or SOTA claim  
**Owner:** [Issue #11425](https://github.com/Luminous-Dynamics/symthaea/issues/11425)

## Why a dedicated scorer is needed

A score-conditioned pitch estimator can hide a wrong rendition by searching only near the expected note. Muse's primary pitch audit has already been changed to remove that search prior. The next requirement is to qualify that estimator against independent singing F0 references and matched alternative trackers.

The evaluator at tools/vocal-pitch-benchmark-eval accepts aligned reference/estimate pairs and calculates FFE, GPE, VDE, voicing coverage, abstention, false-voiced, octave/subharmonic and cents-error metrics. It requires a primary score-independent track and records exact tracker, reference, preprocessing, split, frame-grid and voicing-policy identities. Each segment also binds the source WAV, independent reference-F0 track and tracker-output track digests. Language, register, vocal technique and SNR strata are retained. The expected corpus × condition group set is predeclared and complete; missing or unexpected groups reject the benchmark run rather than silently changing the macro denominator. The run manifest also binds separate calibration and evaluation split IDs/manifests, a voicing-calibration receipt and an optional numeric confidence threshold. A separate manifest audit must establish singer/song/session non-overlap; distinct split labels alone do not prove independence.

## Metric definitions

For each frame:
- reference voiced iff the independent reference F0 is present;
- tracker voiced iff the frozen tracker voicing policy emits an F0 estimate;
- cents error = 1200 × log2(estimate/reference), only when both are voiced;
- a gross pitch error is an absolute cents error above the predeclared threshold.

The report defines:
- FFE@threshold = (voicing mismatch frames + gross pitch error frames) / all frames;
- GPE@threshold = gross pitch error frames / jointly voiced frames;
- VDE = voicing mismatch frames / all frames;
- reference-voiced coverage = jointly voiced / reference voiced;
- abstention rate = false-negative voicing / reference voiced;
- false-voiced rate = false-positive voicing / reference unvoiced;
- octave error rate = octave/subharmonic error frames / jointly voiced frames.

These are explicit task definitions. A publication adapter must map them to the exact definitions of any external benchmark before quoting its metric name or comparing its score. In particular, FFE/GPE conventions and gross-error thresholds can differ across toolkits.

## External tracker benchmark context

The current Pitch Detection Benchmark v2.1 evaluates 19 monophonic trackers over 10 corpora and nine recording conditions (clean plus eight combinations of background, reverberation and microphone filtering). It uses a common 16 ms frame grid, selects confidence thresholds on a separate validation split, weights corpora and non-clean conditions equally, and reports paired confidence intervals with multiplicity-adjusted wins/losses.

Current headline results include:
- SwiftF0: F1@50c 0.781 (95% CI 0.768–0.795), 179.6× real time on one CPU core;
- RMVPE: 0.768 (0.752–0.783), 13.6×;
- FCPE: 0.728 (0.712–0.742), 27.8×;
- pYIN: 0.506 (0.482–0.525), 9.6×.

SwiftF0's lead over RMVPE is not statistically resolved; treat them as co-leading. The benchmark is a mixed monophonic pitch task, not singing-only qualification. Its prepared dataset is CC BY-NC-SA 4.0 and restricted to research use. Its code's MIT licence does not override the dataset licence.

SwiftF0 reports frame confidence and supports roughly 46.875–2,093.75 Hz; streaming requires 176 ms lookahead. It is a strong reference-tracker candidate, not a voice-specific singing analyser. It detects pitched sounds generally, including accompaniment when vocals pause.

## Matched tracker experiment

First wave:
- Muse's score-independent autocorrelation estimator;
- SwiftF0;
- RMVPE;
- FCPE;
- pYIN.

Optional second wave:
- TorchCREPE / CREPE;
- Praat;
- other algorithms only when their exact source, weights, runtime and licence can be pinned.

For every candidate:
1. Pin tracker source revision, model/checkpoint digest, library/runtime version and license.
2. Freeze input audio, downmix, resampling, frame grid, channel selection and preprocessing.
3. Select voiced confidence thresholds on a calibration split disjoint by singer/song/session from confirmation.
4. Never provide target score/F0 as an estimator prior in the primary independent track.
5. Use exactly the same sample-level reference F0 and unvoiced masks.
6. Retain no-estimate, crash, timeout and invalid-output dispositions; do not silently drop a tracker/clip.
7. Report per-segment, corpus × condition and per-register/technique/language/SNR results.

## Real-vocal reference set

- Use licensed or explicitly consented dry monophonic singing recordings with documented access, processing, derived-data, training, commercial and redistribution permissions.
- Obtain sample-level F0 references through manual expert annotation/adjudication or a separately qualified reference process. MIDI note labels are not sample-level ground truth.
- Bind each source audio and annotation file to a digest.
- Split by singer, song and recording session. Prevent near-duplicate phrases and prompt/reference leakage across calibration and confirmation.
- Record supported metadata only: register, technique, language, phonetic context, vibrato, portamento, SNR, source and session.

SoulX-Singer-Eval/GMO-SVS is a useful public reference candidate, but it combines source corpora and requires per-resource rights review. Its README reports 802 GMO-SVS samples from GTSinger, M4Singer and OpenCpop, plus 100 zero-shot segments from 50 singers (25 Mandarin, 25 English). The Hugging Face dataset card marks the released dataset CC-BY-NC-4.0. Do not assume the code, weights and every source corpus share one licence. Do not use non-commercial evaluation data for commercial training or redistribution.

## Statistical reporting

- Report micro frame metrics, per-segment distributions, and equal-weight corpus × recording-condition macro metrics separately.
- For tracker comparisons, use paired source segments and cluster uncertainty at singer/song/session level.
- Do not bootstrap frames or repeated seeds as independent recordings.
- Estimate sample size from a pilot and preregister primary comparisons and thresholds.
- Report confidence intervals for differences, not only separate point-score intervals.
- Preserve ties/undetermined comparisons and all missing/failed outputs.

## Proposed targets

Development targets, to be preregistered after a licensed pilot:
- 100% exact artifact/reference/preprocessing provenance on scored segments;
- 100% detection of injected synthetic known-F0 and voicing-error controls;
- zero score-prior use in the primary independent F0 track;
- 100% of missing/failed tracker outputs preserved in denominators;
- improve singing-specific FFE/GPE/VDE against the current Muse estimator without degrading per-register coverage;
- report latency, memory and CPU/GPU cost beside accuracy.

No absolute tracker superiority target should be fixed from the mixed benchmark's aggregate F1. First reproduce the benchmark methodology on its own terms (only under the research dataset terms), then separately qualify on licensed singing-only material.

## Sources

- Pitch Detection Benchmark v2.1: https://github.com/lars76/pitch-benchmark
- BENCHMARK.md: https://github.com/lars76/pitch-benchmark/blob/main/BENCHMARK.md
- SwiftF0: https://github.com/lars76/swift-f0
- SoulX-Singer-Eval: https://github.com/Soul-AILab/SoulX-Singer-Eval
- SoulX-Singer-Eval dataset card: https://huggingface.co/datasets/Soul-AILab/SoulX-Singer-Eval-Dataset
