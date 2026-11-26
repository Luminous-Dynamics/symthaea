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


## Corpus-level annotation reliability

The temporal_reliability module consumes a corpus of independently annotated items and reports two separate profiles:

1. Nominal Krippendorff alpha for item-level disposition states (Present, Absent, Ambiguous, NotEstimable), including the coincidence-derived observed and expected disagreement.
2. Symmetric pairwise event F1, matched IoU and exact-count agreement at frozen IoU thresholds 0.25, 0.50, 0.75 and 0.90, reusing the production temporal-grounding scorer in both directions.

This is not Krippendorff's alpha for unitizing (alpha_U): temporal interval agreement is deliberately reported separately. The report does not score a model, create consensus truth, calculate a source-work bootstrap or claim population-level reliability.

Run the synthetic corpus with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-temporal-reliability-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-temporal-reliability-corpus.synthetic.json

See docs/research/MEL-GROUND-007-CORPUS-ANNOTATION-RELIABILITY-V1.md for the mathematical definition and claim limits.


## Corrected multi-rater unitizing alpha-u

The temporal_unitizing module implements the corrected Krippendorff 2019 fourth-edition §12.6 u-alpha family for independent partitions of a continuous audio timeline. It reports overall nominal u-alpha, gap-vs-event binary u-alpha, coding-conditional c-u-alpha, and per-value alpha/overlap coverage. It retains gaps explicitly and never builds consensus labels.

The implementation target is not the withdrawn simplified first-printing c_u-alpha equation. Its golden test reproduces the 2019 fourth-edition Figure 12.11 reference vector (three observers, continuum length 60): nominal u-alpha ≈0.510, binary u-alpha ≈0.413, and coding-conditional c-u-alpha ≈0.72835.

For audio, each corpus item is one event query/category over the exact full-recording sample-frame timeline. Present intervals must be sorted, non-overlapping and half-open; Absent maps to a full-gap unitization; Ambiguous and NotEstimable are retained and excluded with explicit reasons. Distinct polyphonic categories require separate query-items.

Run the synthetic unitizing corpus with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-temporal-unitizing-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-temporal-unitizing-alpha-u.synthetic.json

The report preserves each annotator's original intervals and status reason, plus the exact audio clock, query text, annotation-policy version and source-work lineage. Ambiguous/NotEstimable exclusions remain visible. Corpus summaries are descriptive and unweighted by item; source-work summaries remain separate. It does not pool unrelated recordings into one coefficient or compute confidence intervals. See docs/research/MEL-GROUND-008-UNITIZING-ALPHA-U-V1.md for formulas, reference vectors and claim boundaries.
