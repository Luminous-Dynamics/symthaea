# MUSE-MIR-007 — Beat-grid and downbeat evaluation (V1)

**Parent score-structure programme:** [#11677 — MUSE-MIR-004](https://github.com/Luminous-Dynamics/symthaea/issues/11677)  
**Adjacent implementation:** [#11681 — MUSE-MIR-006 measure-phase diagnostic](https://github.com/Luminous-Dynamics/symthaea/pull/11681)  
**Date:** 12 November 2026  
**Status:** Rust implementation and tests authored; not compiled or executed in the authoring environment

## The measurement gap

A note transcription can have correct pitches and note onsets but still misinterpret the meter: half-time, double-time, offbeat phase, or an incorrect downbeat cycle. The score's notated clock is not the same thing as a performer's expressive clock.

MUSE-MIR-006 compares the within-measure phase of already matched notes. MUSE-MIR-007 adds a separate evaluation interface for beat-tracker and downbeat-tracker timestamp sequences. It does not infer beats from audio and does not pretend that note F1 is a beat metric.

## Implementation

Adds \`crates/domains/symthaea-muse/src/beat_grid_eval.rs\`, exported behind the \`theory\` feature.

The API accepts:
- a model prediction with ordered beat and downbeat timestamps, model ID and revision;
- one or more independent reference interpretations;
- a versioned evaluation configuration.

The evaluator reports:
1. **Strict beat F1**, using the documented \`mir_eval.beat.f_measure\` default ±70 ms tolerance;
2. **Cemgil score**, with the configured Gaussian sigma (40 ms default);
3. **Best Cemgil metric-level score**, over the reference, offbeat, double-time, and half-time odd/even variants;
4. **Downbeat F1**, separately scored and only reported when reference downbeat labels exist;
5. **Beat timestamp bias and absolute timing-error distribution**;
6. **Inter-beat-interval error**, calculated only for adjacent one-to-one matched pairs;
7. **Reference-interpretation aggregation**, showing median and minimum strict F1 across independent annotations;
8. **Pairwise reference agreement**, only when references explicitly share the same clip/group identifier.

The report is domain-separated and replay-verifiable against the exact prediction/reference structures supplied to the evaluator.

The optional \`minimum_evaluation_time_ms\` explicitly trims both reference and prediction. Its default is \`None\`; this avoids silently applying the frequently used five-second trim when a benchmark protocol specifies a different boundary.

## Why multiple reference interpretations matter

A single beat sequence is not always a complete ground truth for music. Listeners can legitimately disagree about metrical level or phase, especially in syncopated, slow, sparse, polyrhythmic, or culturally unfamiliar material.

The evaluation API therefore preserves each reference interpretation and reports its score separately. A median/worst-case summary is supplemental; it does not replace the per-reference results. Pairwise reference agreement is calculated only within an explicitly identified shared clip group. An unlabelled reference is never assumed to be from the same recording as another.

This is directly motivated by **GlobalTap** (arXiv submitted 8 November 2026): 200 popular-music excerpts across 59 countries, with 746 participants recruited from ten countries. Its separate validation study used 240 excerpts, 733 taps and 130 listener ratings. The paper reports that crowd-derived and existing reference annotations sometimes differ systematically in metrical phase and interpretation, and that these differences are not always resolved by listener preference for one annotation. GlobalTap is a resource for plural reference judgments, not an excuse to discard expert annotations.

## Tests authored

Twenty-three focused Rust tests cover:
- exact beat and downbeat sequences;
- inclusive 70 ms beat tolerance;
- duplicate predictions as false positives;
- missing beats lowering recall;
- half-time strict-F1 failure and metric-level-tolerant Cemgil behavior;
- exact and offbeat Cemgil metric-level semantics;
- doubled-pulse diagnostic separation;
- rejection immediately outside the strict 70 ms window;
- pinned expected outputs from the Python `mir_eval 0.8.2` reference for exact, offbeat, half-time, double-time, broken-continuity and 70 ms edge cases;
- CML/AML continuity scores for a broken sequence, half-time metric-level recovery and single-beat abstention;
- beat correctness separated from downbeat correctness;
- missing downbeat annotation remaining unknown;
- multiple reference interpretations and pairwise agreement;
- no pairwise comparison without a shared explicit group;
- opt-in five-second trimming applied symmetrically;
- empty beat references rejected;
- adjacent inter-beat-interval error;
- non-monotonic timestamps rejected;
- report replay rejecting changed inputs;
- empty predicted beat output scored as zero rather than treated as a parser failure.

The tests are authored, not compiled or executed. The reference fixture values were generated by calling the installed Python `mir_eval==0.8.2` functions directly (not `mir_eval.beat.evaluate`, whose default five-second trim removes these short fixtures). This fixture set is a targeted parity check, not full API equivalence.

## Industry and research comparators

### mir_eval beat protocol

\`mir_eval.beat.f_measure\` uses a 70 ms default tolerance. Its beat evaluation documentation also describes Cemgil, Goto, P-score, continuity metrics (CMLc/CMLt/AMLc/AMLt), and information gain. The current Rust V1 implements strict F1, direct/best-metric-level Cemgil, and the CMLc/CMLt/AMLc/AMLt continuity metrics using the documented 0.175 phase and period thresholds. It does **not** claim full \`mir_eval\` API parity. In particular, Goto, P-score and information gain remain follow-up work.

References:
- Pinned implementation: `mir_eval==0.8.2`; parity fixtures are included in `beat_grid_eval::tests::pinned_mir_eval_0_8_2_reference_fixtures_match_beat_metric_semantics`.
- https://mir-eval.readthedocs.io/latest/api/beat.html
- https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/beat.py

### MIREX 2026 Audio Beat Tracking

The MIREX 2026 evaluation includes:
- GTZAN: 999 songs;
- SMC: 217 songs;
- osu2beat2025: 703 full-length tracks;
- Yamaha_JPop: 250 tracks;
- Yamaha_Balanced: 239 tracks;
- JSIR: 260 jazz recordings.

The page lists BeatThis as a baseline, QM Tempo Tracker as a baseline, CASM (a context-aware semi-Markov post-processor), and a multi-scale expert re-verification system. Some Yamaha sets are private, and the MIREX page marks systems trained/tuned on a test set with an asterisk. We must preserve those distinctions; no numeric SOTA result is claimed here.

Reference: https://www.music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking_Results

### SMC failure analysis

A May 2026 analysis of the SMC benchmark identifies octave/metrical-level errors, continuity errors and complete beat-tracking failures. It reports that a common DBN minimum-tempo setting of 55 BPM prevented the evaluated system from inferring the correct tempo for 21% of SMC tracks, forcing double-tempo estimates. It also highlights confident-but-wrong beat activations.

This supports our central design choice: strict F1 and metric-level-tolerant Cemgil must be visible side by side. The tolerant score is diagnostic, not a way to conceal the wrong pulse level.

Reference: https://arxiv.org/abs/2605.12287

## Proposed benchmark targets (not results)

1. Reproduce the MIREX 2026 beat evaluation protocol for all accessible public sets before making a leaderboard comparison.
2. Report strict beat F1 at 70 ms, Cemgil direct and best metric level, downbeat F1, continuity metrics, and p50/p95 timing errors as separate dimensions.
3. Add Goto, P-score and information gain in a parity-tested follow-up.
4. Report results separately by dataset, genre, country/tradition where consent and metadata allow, tempo range, rhythmic density, and metrical ambiguity.
5. Use GlobalTap's multi-reference structure to report per-annotation and across-annotation distributions, not one flattened consensus.
6. Compare against BeatThis, DBN/madmom, and the relevant MIREX 2026 systems under identical input, trimming and post-processing rules.
7. Preregister tolerance sweeps and 95% song/recording-clustered confidence intervals. Do not tune on public test labels.

## Paradigm shifts for Melothaea

- **From one ground truth to a distribution of valid perceptions.** Preserve multiple annotator interpretations and identify clips where disagreement is systematic.
- **From one beat pulse to explicit metrical hypotheses.** Treat half/double/offbeat ambiguity as a measurable property, while keeping strict beat and downbeat correctness separate.
- **From note snapping to joint structure inference.** A future score builder should infer beat/downbeat, meter, measure origin and note placement jointly; it should not force notes onto a guessed grid after transcription.
- **From aggregate score to diagnosable error.** Separate phase error, interval/tempo error, continuity, missing/extra beats and downbeat errors.
- **From benchmark-only correctness to musician utility.** Evaluate whether a musician can read, edit and perform the resulting score without reconstructing its rhythm by hand.

## Boundaries

- V1 evaluates timestamp sequences; it does not produce beat/downbeat predictions.
- It does not evaluate audio onset detection, chord/rhythm understanding, tempo-map interpolation, barline placement, pickup-bar origins, or score engraving.
- Metrical-level-tolerant Cemgil is not strict beat accuracy.
- Multiple-reference median F1 is not a claim of human consensus or cultural correctness.
- No external model has been run and no benchmark win is claimed.
