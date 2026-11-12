# MUSE-MIR-015 — Dual-fidelity and cross-composer benchmark protocol (V1)

**Issue:** #11794  
**Parent:** MUSE-MIR-004 score-structure evaluation  
**Implementation:** `crates/domains/symthaea-muse/src/multi_axis_benchmark.rs`  
**Date:** 12 November 2026  
**Status:** protocol and Rust report validator authored; not compiled or executed in this environment

## The paradigm shift

Do not ask whether a transcription system produces a high score on one aggregate metric. Ask whether it reconstructs the written work, reproduces the performance, generalizes to unfamiliar music, knows when it is uncertain, and reduces the musician's actual correction burden.

These are related but distinct objectives. A score can be close to the reference notation yet render poorly; a performance can sound close while its notation is unreadable or structurally wrong. A system can excel on familiar composers yet fail on unseen composers. A system can achieve high apparent accuracy by abstaining on difficult cases unless coverage is reported.

MUSE therefore reports a **multi-axis scorecard**, not a weighted scalar.

## Current public and industry reference points

### 1. Commercial score transcription: Klangio (22 October 2026)

Klangio reports results across 130 one-minute recordings and 13 test sets, comparing five transcription systems. Its reported means include:

| Metric | Klangio | Next reference in that table |
|---|---:|---:|
| Raw note F1 at 100 ms | 0.82 | Songscription 0.77 |
| Final score F1 | 0.67 | Songscription 0.47 |
| Barline F1 | 0.72 | Songscription 0.47 |
| Time-signature accuracy | 92% | Songscription 78% |
| Score error rate (lower is better) | 0.41 | AnthemScore 0.72 |

The vendor reports a first-place score F1 on 12 of 13 sets. The benchmark mixes public and internal sets, including recordings assembled or recorded by the vendor. Its methodology and results are useful industry references, but it is not an independently administered leaderboard. The report also distinguishes raw note F1 on nine aligned sets from score F1 averaged across all thirteen.

Source: https://klang.io/blog/music-transcription-benchmark-2026/

### 2. MIREX 2026 Audio-to-Score: in-distribution versus held-out composers

MIREX separates Blind and Staves-Informed A2S tracks and reports WER, CER, LER and MV2H.

| Blind system / split | WER ↓ | CER ↓ | LER ↓ | MV2H ↑ |
|---|---:|---:|---:|---:|
| Hidalgo-Centeno, Quartets test | 0.0286 | 0.0276 | 0.0792 | 0.9492 |
| Cummins, Quartets test | 0.0346 | 0.0347 | 0.0980 | 0.9478 |
| Cummins, QuartetsMIREX | 0.3202 | 0.3150 | 0.7679 | 0.5981 |
| Hidalgo-Centeno, QuartetsMIREX | 0.3385 | 0.3258 | 0.7460 | 0.6053 |

QuartetsMIREX contains 373 excerpts drawn from eight string quartets by Schubert, Brahms and Mendelssohn, composers not represented in the original Quartets dataset. The drop from roughly 0.95 MV2H on the familiar test set to roughly 0.60 on QuartetsMIREX is the key finding: a leaderboard score on an in-distribution split can conceal weak composer generalization.

This is a direct reason to group splits by composer/work and publish worst-group as well as overall performance.

Source: https://futuremirex.com/mirex/wiki/2026:Audio-to-Score_Transcription_Results

The official task description makes clear that audio-to-score is not simply note-event detection: outputs must encode readable score structure such as voices/staves, meter, clef, key, bar structure and durations. It reports WER and MV2H as main ranking criteria, with CER/LER for detailed comparisons.

Source: https://music-ir.org/mirex/wiki/2026:Audio-to-Score_Transcription

### 3. Notation fidelity versus playback fidelity

*A Dual Evaluation for Music Transcription* (August 2026) studies complementary objectives: similarity to a reference score and similarity of rendered playback to the original performance. Its evaluation includes 24 pipelines built from eight audio-to-MIDI models paired with three MIDI-to-score converters, and a listening study with over 100 participants across 230 piano recordings, 23 works, 30 performers and six composers. It reports that the two objectives can favor different pipelines; in that study CLEWS was the lowest-cost playback metric and the best-correlated with listener judgments.

MUSE should therefore report score similarity and playback similarity separately. A renderer or MIDI-to-score converter must be part of the pinned evaluation pipeline, not an invisible implementation detail.

Sources:
- https://arxiv.org/abs/2608.04511
- https://github.com/pingw220/AMT-Dual-Eval

### 4. Beat perception and cultural/metrical plurality

GlobalTap (published 8 November 2026) contains 200 popular-music excerpts spanning 59 countries, with 746 participants recruited across ten countries, including South Africa. The paper reports systematic differences in metrical-level choices across listener groups. This is not notated barline ground truth: it is a multi-listener beat-perception resource.

MUSE must distinguish:
- the barlines explicitly written in a score;
- the beats/downbeats heard or tapped in a performance;
- alternative perceptual metrical interpretations supported by listener evidence.

Do not punish a system for disagreeing with one reference when multiple listener groups consistently perceive a different metrical level. Preserve the reference set and report agreement/disagreement structure.

Sources:
- https://arxiv.org/abs/2611.11137
- https://github.com/williambotticelli-wells/GlobalTap

### 5. Beat tracking is not barline reconstruction

MIREX 2026 BeatThis reports a JSIR baseline F1 of 94.70 on its experimental beat/downbeat test set. That result is useful for the beat-perception axis only. It is not a barline F1 or score-notation result and must not be used as one.

Source: https://music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking_Results

## The MUSE scorecard

Each report must include the following axes where applicable:

| Axis | Example measures | What it answers |
|---|---|---|
| Notation fidelity | WER, CER, LER, MV2H, note/rest/duration/voice/staff/key/meter accuracy | Can a musician read and edit the score? |
| Structure fidelity | Barline P/R/F1, exact measure-span accuracy, pickup/incomplete/irregular classification, polymeter correctness | Is the notation structurally organized correctly? |
| Playback fidelity | CLEWS where applicable, onset/offset/time-warp errors, blinded human preference | Does the score/render reproduce the performance? |
| Beat perception | Beat/downbeat metrics, multi-reference tapping agreement, metrical-level ambiguity | Does the system understand perceived pulse? |
| Generalization | Composer/artist/performer/session/instrument/genre/culture held-out slices, worst-group score | Does it work on unfamiliar music? |
| Human utility | Expert correction minutes, edit distance, successful task completion, protected-content violations | Does it save musicians time without damaging intent? |
| Uncertainty | Calibration, selective risk/coverage, abstention rate, unsupported-structure false success | Does it know when not to guess? |
| Efficiency | Real-time factor, latency, peak memory, model size, compute/energy per minute | Is the system practical to use? |

No aggregate score may hide a failure in a critical axis. Publish each axis and its confidence interval separately.

## Evaluation tracks

These tracks are not interchangeable:

- **Blind:** audio only; no staff or score metadata.
- **Staves-Informed:** score/staff metadata supplied according to the benchmark protocol.
- **Human-Assisted:** explicitly record the human-provided information, corrections, time and interaction budget.

A human-assisted run must not be ranked as a blind run. Any provided meter, key, tempo range, staff identity or instrument label is part of the protocol fingerprint.

## Split design and leakage control

Random excerpt splits are inadequate for the primary generalization claims. At least one grouped holdout must be preregistered for each of:

- composer/work;
- artist/performer;
- recording session/source recording;
- instrument family/ensemble;
- genre/cultural region.

Where sample counts support it, report more than one. The group key and opaque dataset-native group IDs belong in the protocol manifest. Training and evaluation group sets must be disjoint. The evaluator refuses to label random-excerpt splits as out-of-distribution tests.

For beat-perception tasks with multiple listener references, preserve participant group/source geography and report consensus as well as disagreement. For notation tasks, score against the authored score, not a beat consensus.

## Confidence intervals and worst-group performance

- Bootstrap at the work/composer/artist/performer level, not by treating excerpts from the same work as independent samples.
- Report the number of independent clusters.
- Report per-group and worst-group results, not just the mean.
- Report sample coverage and the exact evaluated sample-manifest digest per metric.
- Do not compare a metric when the two systems were scored on different sample subsets. Use a coverage-risk curve or evaluate explicit abstentions as outcomes instead.

A metric value must carry a protocol ID, unit, direction, tolerance/matching rule, sample count, exact evaluated-sample digest and (for OOD overall metrics) a cluster-aware confidence interval.

## Rust report contract

`multi_axis_benchmark.rs` introduces a versioned, serializable protocol and run report. It validates:
- dataset, training corpus, reference, split, source, model-input and build digests;
- an explicit Blind/Staves-Informed/Human-Assisted track;
- train/evaluation group disjointness and held-out group membership;
- unique metric definitions and per-slice metric IDs;
- renderer/version/soundfont-or-asset/render-configuration identity when playback metrics are used;
- clustered confidence intervals for OOD overall metrics, clustered by the declared group;
- exact evaluated-sample digests and counts;
- report and protocol commitments.

The comparison function requires an identical protocol fingerprint, identical overall sample counts and exact metric-sample manifest. It compares one named metric at a time and never produces a composite winner score. A protocol mismatch, track mismatch, renderer mismatch or different evaluated sample set prevents ranking. Out-of-distribution claims require a non-random grouping unit and an evaluation group set exactly equal to the declared held-out group set; in-distribution claims use random-excerpt splits.

These digests commit the report content; they do not authenticate the dataset or independently prove the split. Provenance and rights authorization remain separate evidence requirements.

## Stretch targets (targets, not achieved results)

1. Reproduce MIREX A2S public reference results on the exact public splits before claiming improvement.
2. On matched public score-transcription sets: score F1 >= 0.75 and barline F1 >= 0.80, with metric tolerances and confidence intervals published.
3. On a matched held-out-composer MIREX-style split: MV2H >= 0.70 and WER <= 0.22 as a stretch target. These targets are intentionally above the current 2026 QuartetsMIREX blind results, but must be compared within the same track/protocol.
4. Achieve 100% correctness on deterministic synthetic fixtures for exact measure maps, pickup/mixed meter, polymeter, repeats and unsupported-semantics refusal.
5. Zero false-success reports when score structure is absent or unsupported.
6. At least 30% lower median expert correction time than a matched commercial baseline on a preregistered task set, with no increase in protected-content violations.
7. Publish separate results for Blind, Staves-Informed and Human-Assisted tracks, as well as worst-group results.
8. Publish beat-perception and barline scores separately; do not present MIREX BeatThis F1 as barline accuracy.

## Qualification boundary

The Rust validator and tests are authored, not compiled or executed in this environment. The code includes 11 authored validator tests for valid OOD receipts, group leakage, invalid OOD split claims, in-distribution grouping, missing renderer identity, metric-protocol mismatch, exact sample-subset mismatch, report/protocol tampering, fraction-range errors, and one-metric comparison. It validates report provenance, split disjointness and comparison compatibility; it does not itself compute MV2H, CLEWS, score F1, human correction time or confidence intervals. Those metric engines and data collection protocols must be independently implemented and qualified. No MUSE SOTA claim is made here.
