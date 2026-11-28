# MACE paired benchmark effects and quality-resource frontier v0.1

**Status:** evaluator kernel authored with deterministic synthetic fixtures. It has not yet passed its dedicated Rust workflow and has not analyzed official benchmark outcomes.

## Why this is separate from the manifest

The matched-run contract proves that two arms declared the same benchmark, track, protocol, split, reader, judge, tool contract, evaluator, metrics, seeds and resource budget. It does not calculate task-level effect sizes or uncertainty. This crate consumes those matched manifests and receipts plus per-task evaluator outcomes.

## Pairing and dependence

- Every outcome belongs to an exact run, arm, independent cluster and task.
- All arms must have exactly the same `(cluster, task)` keys. Missing tasks are rejected; they are not silently dropped.
- The actual scored task count and independent cluster count must match each manifest.
- Within each cluster, task-level deltas are averaged first. Independent clusters—not tasks, turns, or repeated interventions—are the resampling unit.
- Bootstrap uses a deterministic SplitMix64 stream with 2,000 cluster-level percentile replicates and a 95% interval.

## Direction-aware paired effects

For each metric, the report preserves the native-unit candidate-minus-baseline delta. It also computes a positive-is-better oriented gain for objective metrics: candidate − baseline for higher-is-better, baseline − candidate for lower-is-better. Diagnostic-only metrics remain descriptive and do not receive directional improvement claims.

Each effect includes independent cluster count, paired task count, 95% interval and the worst observed cluster gain. This is an uncertainty summary under the declared cluster design, not a universal guarantee against arbitrary distribution shift.

## Pareto frontier, not a composite score

The evaluator computes equal-cluster-weighted per-arm means for each declared metric and returns the nondominated arms over all Higher/Lower objectives. Diagnostic-only metrics are excluded from dominance. Quality, latency, storage and cost remain separate axes; no weights are invented and no score can compensate for a hard governance failure.

## Validation invariants

- Matched-comparison receipt integrity is checked and recomputed from the full manifests.
- Metric IDs, task IDs, arm IDs and run IDs must resolve exactly.
- Duplicate outcomes, non-finite values, missing arms, incomplete paired task coverage, incorrect manifest counts and fewer than two independent clusters fail closed.
- Benchmark and track identity come from the matched receipt. Cross-track aggregation is impossible through this API.

## Run

```bash
cargo test --manifest-path research/mace-paired-benchmark-frontier-v0-1/Cargo.toml
cargo clippy --manifest-path research/mace-paired-benchmark-frontier-v0-1/Cargo.toml --all-targets -- -D warnings
```

Twelve deterministic Rust tests cover oriented accuracy/latency/storage effects, a non-scalar Pareto tradeoff, cluster-level inference, incomplete task pairing, duplicate/non-finite/metric-set failures, receipt tampering, baseline resolution, manifest counts and diagnostic-only metrics.

## Integration

Next integrate task outcomes from #14039's TimlMemBench host runtime and #14033's run manifests. The official raw-native task labels and grader outcomes stay evaluator-only; the candidate receives only the raw session and current query. Use separate evaluator custody to write outcome rows after candidate completion.