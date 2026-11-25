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


## Dataset-level inter-annotator reliability

The temporal_reliability module consumes multiple #13560 cases across query items and source works. It reports three different quantities without combining them:

1. Krippendorff's nominal alpha over Present/Absent labels across multiple query items. Ambiguous and NotEstimable labels remain in the source records but are excluded as missing for this nominal-state measure. If expected disagreement is zero, alpha is undefined rather than being reported as 1.
2. Symmetric interval-set agreement for Present/Present annotator pairs. Each pair is scored in both directions using the production scorer; interval F1, matched IoU, exact occurrence-count agreement and count difference remain separate.
3. Boundary agreement in milliseconds, with tolerance rates at the dataset's strictly ascending preregistered millisecond thresholds. These are event boundary comparisons, not model confidence.

Every dataset item binds a source_work_id_sha256. The bootstrap resamples all query items from a source work together using deterministic SplitMix64, and reports separate intervals for alpha, pairwise state agreement, interval F1/IoU, count agreement, start/end boundary MAE, and each preregistered start/end/joint boundary-tolerance rate. Point estimates for pairwise and boundary metrics are source-work macro-averages, matching the bootstrap estimand. A proposed 30-source-work floor is reported but does not by itself authorize population claims; the output always sets population_confidence_claim_permitted to false.

Run the synthetic six-query/three-source-work dataset with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-temporal-reliability-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-temporal-reliability-six-items.synthetic.json

The fixture is schema-backed by schemas/mel-multi-annotator-reliability-dataset-v1.schema.json. It is a deterministic implementation control, not a musician-reliability study or SOTA result. Dataset items must share one scoring plan and annotation-policy version; source-work cluster IDs and annotation record IDs must be unique/consistent under the dataset contract.


The reliability report also retains a **separate model-vs-annotator evaluation** for every query item. It includes the exact model prediction, source/audio identity, frozen scorer plan, original pseudonymous annotation records, per-annotator model scores, and the #13560 coverage/F1/count/absence summaries. Dataset model-performance summaries are macro-averaged by source work and keep Present F1, exact count, one-to-many count-gated F1, Absent claim accuracy, coverage and fully-correct rate separate. The inter-annotator alpha and pairwise interval reliability remain independent quantities. This prevents a high/low alpha from being mistaken for model performance. Model confidence calibration is not computed.


## Dual-stream annotation acquisition

The temporal_acquisition module consumes candidate-level outputs from the #13560 per-annotator model scorer and #13617 dataset reliability evaluator. Each candidate binds:
- exact source-work and recording digests;
- the purpose scope and pseudonymous cohort role;
- a rights/consent receipt digest;
- the #13560 model-grounding report digest;
- the #13617 inter-annotator reliability report digest;
- a frozen cohort-assignment manifest digest;
- annotation panel, cost, model-error, disagreement, count, coverage and boundary-dispersion signals.

The planner produces two separate streams:

1. Active refinement: deterministic priority bands for likely model failure on a stable/covered panel, human disagreement, coverage gaps and hidden repeated-event controls. It round-robins across source strata, enforces per-source-work caps, and respects active budget/quotas. An uncertain panel is routed to annotation refinement, not mislabelled as a clean model failure.
2. Probability audit: source works are sampled uniformly within each stratum, then one eligible item is drawn uniformly within each selected source work. The report records source-work, conditional-item, and product inclusion probabilities. Per-stratum sample sizes are fixed before any random draw using worst-case item-cost reservation, so a cheap random item in an earlier stratum cannot change the sample size/probability in later strata.

The active and audit source-work sets are required to be disjoint. Active selections are not a probability sample; probability-audit selections are not adapted to model error. Their results must not be pooled without the proper design weights. A non-selected item is unreviewed, not a negative event.

Rights/consent receipt digests and diagnostic report digests are structurally validated and included in the replayable receipt, but this planner does not authenticate those external records. Its selection receipt is also not an authentication token.

Run the synthetic dual-stream plan with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-annotation-acquisition-plan -- tools/mel-temporal-grounding-qualifier/fixtures/mel-annotation-acquisition.synthetic.json

The committed fixture is 11 candidates across two strata. It is a deterministic planner control only; no human study, unbiased population estimate, or annotation-quality improvement is established.
