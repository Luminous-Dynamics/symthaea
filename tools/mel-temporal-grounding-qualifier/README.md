# MEL temporal-grounding qualification harness

This nested Rust package imports the exact production temporal-grounding scorer and SHA-256 digest module by path. It remains isolated from the main workspace so focused qualification does not inherit unrelated legacy build/lint debt.

Run the scorer checks with:

    cargo +1.96.0 test --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --all-targets -- -D warnings

## Multi-annotator temporal grounding

The temporal_consensus module preserves every independent annotation record and runs the exact production temporal-grounding scorer once per annotator. It reports per-annotator TP/FP/FN, F1, occurrence count accuracy and boundary errors, plus panel summaries for positive grounding, negative claims, occurrence-count spread and boundary errors.

Ambiguous and not-estimable labels are retained in the report but excluded only from their own scored denominators. An absent label remains a scored negative reference. The evaluator does not create a majority-voted/merged consensus interval, rank annotators, compute pairwise inter-annotator agreement, or treat a small panel's score spread as a population confidence interval.

Run the synthetic six-annotator panel with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-multi-annotator-grounding-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-multi-annotator-temporal.synthetic.json

All annotator identities are represented by opaque digests in the fixture/report. Raw personal identifiers must not be placed in public benchmark artifacts.


The strict input envelope is also described by `schemas/mel-multi-annotator-temporal-grounding-v1.schema.json`; cross-record identity, state/interval semantics and digest uniqueness are enforced by the Rust adapter.

Count-gated one-to-many F1 is computed only over Present annotations whose reference contains more than one occurrence. Single-event references remain in ordinary F1 and exact-count accuracy but do not dilute the one-to-many denominator.

The report separates eligible annotation count from model abstention, parse failure and prediction coverage. A present/absent annotation stays eligible even if the model returns Ambiguous, NotEstimable, NotDetected or Unparseable. Coverage is reported overall and separately for Present and Absent references. The panel score spread is human-reference variability; it is not model confidence. The adapter does not compute formal pairwise inter-annotator agreement or population confidence intervals.

Research protocol: `docs/research/MEL-GROUND-006-MULTI-ANNOTATOR-TEMPORAL-GROUNDING-V1.md`.


For each predicted occurrence, the report also derives a **per-event reference-boundary distribution** from the production scorer's existing prediction↔annotation match records. It reports the number and fraction of eligible Present annotators who match that event, plus min/median/max start and end frames and boundary spread. Events that match no eligible Present annotation appear in `unmatched_prediction_indices`. This is not an interval consensus: original annotation records remain intact, and the support denominator is visible.


## Selective temporal grounding and reason-specific abstention

The temporal_selective module accepts a frozen adjudicated ground-truth case, raw prediction, typed decision, model-declared decision confidence and shared model/runtime/decoding identity. It scores each answer through the production temporal-grounding scorer. Decision dispositions are distinct: Answer, ClaimAbsent, AbstainAmbiguous and AbstainNotEstimable.

The report separates coverage, selective risk/accuracy, answer-only F1/recall/count-gated F1, false absence claims, ambiguity/not-estimable reason precision and recall, calibration bins/Brier/ECE, evidence-stratum summaries and a confidence-ranked risk-coverage curve/AURC. It includes always-answer, random-abstention and always-abstain baselines; always-abstain has zero coverage and is never treated as successful product behaviour.

Run the synthetic selective-control panel:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-selective-grounding-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-selective-temporal.synthetic.json

Decision confidence is self-reported. Runtime identity and evidence receipts are not authenticated by this evaluator, and it does not inspect hidden reasoning. Calibration thresholds must be selected on source-lineage-disjoint calibration data, not the held-out test. Do not compare a risk-coverage/AURC result to a multiple-choice abstention result without reproducing the same task, units and coverage protocol.


### Identifiability basis is separate from the abstention action

Every case carries one adjudicated basis: ClearPositive, ClearAbsence, AnnotatorDisagreement, TaskAmbiguity, SignalInsufficient or UnsupportedQuery. The evaluator checks the basis against the adjudicated truth state and reports reason correctness by basis. Annotator disagreement is not silently treated as sensor/signal insufficiency, and a not-estimable case cannot be relabelled as mere annotator disagreement.

The basis is an adjudication label backed by a frozen policy digest; it is not independently authenticated by this adapter. Use #13560 to retain the underlying per-annotator evidence.


### Counterfactual always-answer baseline

Each case retains both the selected prediction and the unconditional pre-abstention candidate. Covered Answer/ClaimAbsent decisions must use the same prediction as this candidate; abstained cases can retain the candidate that the selective policy suppressed. The evaluator scores the unconditional candidate against the same truth to calculate a paired always-answer baseline. Random-abstention risk is the expected error rate when retaining those candidates independently of confidence. The output includes both predictions and both scorer records per case.

The 10-case synthetic fixture has 2 correct unconditional candidates and 8 incorrect ones, so its always-answer and random-retention risk are 0.8. This is a deliberately adversarial evaluator control, not a performance result for Melothaea.


### Selective inference cost

Each case contains matched selective-path wall time, one-pass baseline wall time and model forward-pass count for the same query/audio/runtime. The report emits mean/p95 forward passes, selective latency and latency multiplier. Batch-level hardware identity is required, but runtime, hardware and inference-cost receipts remain unauthenticated unless joined to an independent execution attestation. Synthetic cost values prove report plumbing only; they are not device performance measurements.
