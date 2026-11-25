# MEL temporal-grounding qualification harness

This nested Rust package imports the exact production temporal-grounding scorer and SHA-256 digest module by path. It remains isolated from the main workspace so focused qualification does not inherit unrelated legacy build/lint debt.

Run the scorer checks with:

    cargo +1.96.0 test --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --all-targets -- -D warnings

## Multi-annotator temporal grounding

The temporal_consensus module preserves every independent annotation record and runs the exact production temporal-grounding scorer once per annotator. It reports per-annotator TP/FP/FN, F1, occurrence count accuracy and boundary errors, plus panel summaries for positive grounding, negative claims, occurrence-count spread and boundary errors.

Ambiguous and not-estimable labels are retained in the report but excluded only from their own scored denominators. An absent label remains a scored negative reference. The evaluator does not create a majority-voted/merged consensus interval, rank annotators, or treat a small panel's score spread as a population confidence interval. It computes descriptive pairwise concordance, but not chance-corrected formal inter-annotator reliability.

Run the synthetic six-annotator panel with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-multi-annotator-grounding-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-multi-annotator-temporal.synthetic.json

All annotator identities are represented by opaque digests in the fixture/report. Raw personal identifiers must not be placed in public benchmark artifacts.


The strict input envelope is also described by `schemas/mel-multi-annotator-temporal-grounding-v1.schema.json`; cross-record identity, state/interval semantics and digest uniqueness are enforced by the Rust adapter.

Count-gated one-to-many F1 is computed only over Present annotations whose reference contains more than one occurrence. Single-event references remain in ordinary F1 and exact-count accuracy but do not dilute the one-to-many denominator.

The report separates eligible annotation count from model abstention, parse failure and prediction coverage. A present/absent annotation stays eligible even if the model returns Ambiguous, NotEstimable, NotDetected or Unparseable. Coverage is reported overall and separately for Present and Absent references. The panel score spread is human-reference variability; it is not model confidence. The adapter does not compute formal pairwise inter-annotator agreement or population confidence intervals.

Research protocol: `docs/research/MEL-GROUND-006-MULTI-ANNOTATOR-TEMPORAL-GROUNDING-V1.md`.


For each predicted occurrence, the report also derives a **per-event reference-boundary distribution** from the production scorer's existing prediction↔annotation match records. It reports the number and fraction of eligible Present annotators who match that event, plus min/median/max start and end frames and boundary spread. Events that match no eligible Present annotation appear in `unmatched_prediction_indices`. This is not an interval consensus: original annotation records remain intact, and the support denominator is visible.


## Pairwise annotation concordance (descriptive, not formal IAA)

Every annotator pair is compared. The report first measures exact disposition agreement across Present, Absent, Ambiguous and NotEstimable states. For Present/Present pairs only, it applies the production one-to-one interval matcher in both directions and reports symmetric event F1, mean matched interval IoU and exact occurrence-count agreement.

Pairs with different dispositions, including Present/Absent, do not receive fabricated interval scores. Ambiguous and NotEstimable states remain categorical. These pairwise metrics are descriptive for the current query panel and are not Cohen's kappa, Fleiss' kappa or Krippendorff's alpha. A defensible chance-corrected reliability estimate requires a multi-query corpus with preregistered item/rater structure, missingness policy and chance model. The report continues to set `pairwise_inter_annotator_agreement_computed = false`.


## Corpus-level temporal annotation reliability

The temporal_reliability module consumes multiple query-items, each with two or more independent annotation records. It computes **nominal Krippendorff's alpha** for the four disposition states (Present, Absent, Ambiguous, NotEstimable) using a coincidence matrix that supports a variable number of annotations per item. It returns the observed disagreement, expected disagreement, pooled category counts and alpha value. If expected disagreement is zero, alpha is null with an explicit status reason rather than an invented value.

Temporal event-set agreement is reported separately. The corpus pins a strictly increasing IoU threshold sweep; each Present/Present annotator pair is scored in both directions through the production one-to-one scorer. The output reports symmetric event F1, matched interval IoU, exact occurrence-count agreement and onset/offset error distributions at each threshold. No second interval matcher is introduced.

The evaluator requires at least two query-items and at least two annotators per item, supports variable annotator membership across items, caps each item at 256 annotations to bound pairwise scoring, and binds each item to a source-work lineage digest. Alpha coincidence totals and denominators use checked integer arithmetic and fail closed on overflow. The CLI adds a deterministic source-work-cluster bootstrap for nominal alpha. It resamples distinct source lineages with replacement and keeps all query-items from each sampled lineage together. The report contains requested/valid/degenerate replicate counts, seed, valid fraction, percentile 95% interval and an explicit proposed 30-source-work confirmatory floor. The same source-work resampling also reports threshold-specific uncertainty for event F1, matched IoU, count agreement and onset/offset MAE. A temporal metric with fewer than two eligible source works gets a point estimate and an explicit no-interval reason. Degenerate all-one-category resamples remain undefined rather than receiving a fabricated interval. The floor is only a warning, not a substitute for power analysis or representative work sampling. Three synthetic lineages test reproducibility and plumbing only; the report still sets population_confidence_interval_computed=false. The evaluator does not create consensus references or score model performance.

Run the synthetic corpus fixture with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-temporal-reliability-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-temporal-reliability-corpus.synthetic.json

The strict input envelope is defined in schemas/mel-temporal-annotation-reliability-corpus-v1.schema.json. Structural cross-record checks and the actual alpha/temporal scoring rules are enforced by Rust.
