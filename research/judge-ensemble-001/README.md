# Judge Ensemble Qualification (SCI-JUDGE-ENSEMBLE-001)

This crate is the first std-only implementation slice for issue #9431.

Run the formatter and tests with:

    cargo fmt --manifest-path research/judge-ensemble-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/judge-ensemble-001/Cargo.toml

## Current scope

- Requires exact judge slots for every planned trial, including explicit abstain, failed, and missing outcomes.
- Fits a strongest-single comparator using balanced accuracy, plus smoothed per-judge calibration weights derived from separate sensitivity and specificity estimates on both reference classes.
- Reports judge-specific accuracy and confidence Brier score.
- Reports pairwise double-fault rates and phi correlation of error indicators on jointly scored calibration items.
- Compares strongest single, unweighted majority, dependency-family-capped voting, and calibration-weighted voting on a disjoint evaluation split.
- Keeps reference-missing, abstention, failure and missing-call denominators visible.
- Reports planned query-call counts so a full-panel arm cannot be mistaken for a cost-matched comparison.

## Explicit limitations

Family-capped voting is a transparent baseline for clone amplification, not a full dependence-aware statistical estimator. The calibration weight is a one-dimensional heuristic, not an IRT or hierarchical model. This crate performs fixed-evaluation-set descriptive analysis only; it does not fit population models, compute uncertainty intervals, authenticate judge identities, prove human reference truth, or establish matched-budget superiority. JSON schemas, cryptographic run receipts, provider adapters, human adjudication, sample-size planning and a Pareto cost frontier remain separate work.
