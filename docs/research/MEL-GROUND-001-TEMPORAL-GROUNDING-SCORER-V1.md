# MEL-GROUND-001 — Temporal grounding scorer V1

Status: implementation proposal / draft PR. The Rust tests are authored but require exact-head execution before any pass claim.

## Purpose

This module is the first reusable measurement component for [#10265 — temporally grounded musical perception and event retrieval](https://github.com/Luminous-Dynamics/symthaea/issues/10265).

It evaluates a prediction against adjudicated intervals for one exact audio artifact. It does **not** run an audio detector, infer event labels, or prove that a supplied SHA-256 was computed from the correct bytes. A caller must first seal the exact audio bytes and supply their digest, sample rate, channel count, and frame count.

## Design choices

### 1. Frame coordinates, not floating-point seconds

Intervals are half-open sample-frame ranges, [start_frame, end_frame). The audio clock is explicit and binds sample rate, channel count, total frame count, and the exact audio SHA-256. This avoids ambiguities at adjacent boundaries and avoids scoring artifacts caused by floating-point timestamp rounding.

Adapters from TAG-Bench, MusTBench, MusicGroundingBench, or TEMPO must convert their native timestamp representation into this clock under a frozen, documented rounding rule. This evaluator is not a drop-in reproduction of any official benchmark's scorer.

### 2. One-to-many retrieval is a first-class task

A prediction may contain multiple intervals for the same query. Evaluation uses one-to-one bipartite matching: each predicted interval can match at most one adjudicated occurrence, and each truth occurrence can receive at most one match. An edge is admissible when interval IoU meets the frozen threshold.

The matcher uses deterministic augmenting paths to maximize the number of admissible matches. Candidate annotation edges are visited in descending IoU order, then annotation index. This guarantees maximum match cardinality; the reported mean matched IoU is for that deterministic maximum-cardinality assignment, not a claim that a globally maximum-IoU assignment was found.

The score reports occurrence precision, recall, F1, absolute count error, exact occurrence-count success, and the selected interval matches separately. Each match also retains signed onset/offset errors in frames. A dataset aggregator reports micro/macro event metrics, one-to-many count accuracy, explicit-absence accuracy and coverage, abstention and parse-failure counts, mean matched IoU, and p95 boundary errors in milliseconds. This directly exposes the under-counting and output-format failures highlighted by recent temporal-grounding benchmarks.

### 3. Absence is not detection failure

Prediction states are intentionally distinct:

- Occurrences: one or more accepted predicted intervals.
- QueryAbsent: an explicit negative claim that the queried event is absent.
- NotDetected: the system attempted detection but found no accepted event.
- Ambiguous: one or more candidate intervals remain unresolved.
- NotEstimable: the task cannot be estimated from available evidence.
- Unparseable: raw model output could not be converted into the versioned answer schema; its raw-output digest and failure reason are retained. This is a parsing failure, not an intentional abstention.

For a known-absent query, only QueryAbsent is scored as a correct absence claim. NotDetected is recorded as an abstention, not as proof of absence. For a present query, NotDetected incurs missed occurrences. Ambiguous and NotEstimable predictions retain their status and are accounted for as abstentions.

Ground truth also distinguishes Present, Absent, Ambiguous, and NotEstimable. Ambiguous or not-estimable truth records are explicitly excluded from interval-accuracy denominators rather than mislabeled as absent.

### 4. Exact artifact identity

The prediction and ground truth must match on audio digest, sample rate, channel count and frame count. A prediction over a different rendition cannot be scored against a previous artifact. The evaluator plan and both input records can be bound into a canonical JSON commitment using the repository's existing canonical evidence encoder. That encoder is repository-specific, not a cross-language canonical JSON standard.

## V1 contract and failure conditions

The evaluator rejects:
- wrong schema/evaluator versions;
- empty query identity/text or annotation policy identity;
- malformed lower-case SHA-256 values;
- invalid sample rate, channel count, or empty audio clock;
- zero-length, out-of-range, duplicate, or unsorted intervals;
- a Present ground truth without occurrences;
- a non-Present ground truth carrying positive intervals;
- missing ambiguity/not-estimable reasons;
- mismatched query IDs or exact audio artifact clocks;
- invalid IoU thresholds.

Overlapping and nested intervals are permitted because musical events can co-occur. Exact duplicate intervals are rejected.

## Dataset aggregation

The dataset-level aggregator rejects duplicate query IDs and mixed evaluator versions. It reports:
- positive-query micro precision/recall/F1 and macro positive-query F1;
- one-to-many exact-count accuracy;
- negative-query explicit-absence accuracy and coverage;
- ambiguity/not-estimable truth exclusions, explicit abstentions and parse failures;
- mean matched IoU and onset/offset boundary MAE and p95 in milliseconds, converting each match under that artifact's sample rate.

Boundary errors are not converted to beat fractions in V1 because the canonical beat-to-performance-time map is owned by #9477 and is not yet a universal audio clock. Do not infer beats from the output itself and score against that self-inferred grid.

## Test matrix

The module includes regression tests for:
1. half-open interval IoU;
2. one-to-many occurrence matching and extra false positives;
3. duplicate-like predictions not receiving duplicate credit;
4. explicit absence versus missed detection;
5. ambiguous and not-estimable truth exclusion;
6. exact artifact digest mismatch;
7. interval past artifact duration;
8. invalid clock and zero-length interval;
9. unsorted and duplicate intervals;
10. explicit ambiguous prediction abstention;
11. evaluator-input commitment changing when the frozen IoU threshold changes.

## How this compares with external work

- [TAG-Bench](https://arxiv.org/abs/2609.01542) evaluates every matching interval in natural-language temporal audio retrieval; it reports substantial gaps in long-duration localization and one-to-many count accuracy. This module adopts the multiple-occurrence requirement, but does not reproduce TAG-Bench's official annotation parsing or scoring.
- [MusTBench](https://arxiv.org/abs/2605.29300) provides music-specific temporal source grounding, local timestamp reasoning, transition ordering, and region grounding. Its task outputs and scoring remain separate.
- [MusicGroundingBench](https://arxiv.org/abs/2608.29480) is particularly relevant to synthetic fixtures with exact MIDI-to-audio alignment. Future adapters can retain symbolic source-event IDs alongside frame intervals without making the audio-only model invent IDs.
- [TEMPO](https://aclanthology.org/2026.emnlp-main.694/) demonstrates a unified timestamping model across audio, speech, and music. Its official evaluation remains an independent comparator.
- [TimingBench](https://taan.ai/research/rushing-or-dragging/) studies generation-side temporal control, not query-based event retrieval. Its detector calibration should inform audio timing verification but its results must not be pooled with grounding scores.

The current research frontier is fragmented by task and evaluation protocol. Report every official benchmark under its own versioned scorer, and publish the Melothaea frame-based extension separately.

## Qualification sequence

1. Run unit tests on the exact PR head using the repository's pinned Rust toolchain.
2. Add a manifest adapter for the official TAG-Bench schema and verify exact parity on a frozen local sample before reporting direct TAG-Bench numbers.
3. Add a separate MusTBench adapter with task-specific scoring; do not force multiple-choice and interval-retrieval tasks into one metric.
4. Generate synthetic Scores, render them with version-pinned renderers, and bind each source event ID to the exact audio digest, event frame interval, and renderer identity.
5. Validate detector boundaries and event-family uncertainty against multiple human annotations.
6. Evaluate held-out renderer, timbre, mix, density, tempo and performance-time perturbations.
7. Publish event P/R/F1, IoU, boundary error in frames/ms/beats, count accuracy, negative-query accuracy, abstention coverage, parser failures, and calibration separately.

## Claim ceiling

A passing scorer unit test would establish deterministic evaluation behavior only. It would not establish waveform perception, renderer alignment, annotation quality, benchmark rank, human preference, or musical intelligence. Those require independent datasets, baselines, and exact-head experimental evidence.
