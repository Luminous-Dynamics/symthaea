# connectome-query-stability

A dependency-free Rust auditor for *query-specific observed stability* across provenance-linked connectome checkpoints.

## The shift

A connectome does not have one universal "percent complete" value. Global graph statistics can stabilize while a particular edge, synaptic partner, or circuit pathway remains uncertain. Conversely, one named query may be stable across every current structural alternative even while unrelated annotations remain unresolved.

This crate checks a deliberately limited claim:

> Did every declared query metric remain within its absolute tolerance across the complete supplied checkpoint window and every supplied admissible scenario?

It does not forecast future changes. The report carries the first/latest graph snapshot SHA-256, latest edit-history manifest, graph-builder configuration and structural-uncertainty manifest so downstream receipts can bind the result to the exact audited graph state. It hard-codes:
- `future_stability_certified = false`
- `universal_connectome_completeness_claimed = false`

A future-stability claim needs rolling-origin evaluation against later, held-out edit checkpoints.

## Input contract

- Query ID, frozen query-definition SHA-256 and analysis-code SHA-256.
- Uncertainty-set SHA-256 and one fixed source-volume ID.
- Metric IDs, absolute tolerances and unit IDs.
- Two or more cumulative graph checkpoints, each with:
  - monotonic cumulative edit count;
  - graph snapshot SHA-256 and parent snapshot SHA-256;
  - edit-history manifest and graph-builder configuration SHA-256;
  - structural-uncertainty manifest SHA-256;
  - one or more scenario-specific metric intervals.
- Every checkpoint must use the same source volume and graph-builder config.
- Parent snapshot digests must form a linear chain; edit counts must increase strictly.
- Every scenario must supply exactly the declared metric set, each with the exact declared unit.

Scenarios are an unweighted admissible set. The crate never averages them or accepts arbitrary scenario weights.

## Conservative stability rule

For each metric:
1. Preserve every latest-checkpoint scenario interval.
2. Compute a conservative hull across those intervals.
3. Compute the largest possible absolute difference between intervals at any two different checkpoints, over every pair of supplied scenarios.
4. Mark the metric stable over the observed window only when both the latest identified-set hull width and worst observed temporal drift are within its declared absolute tolerance.

The query status is:
- `InsufficientHistory`: fewer than two checkpoints;
- `StableOverObservedWindow`: every declared metric meets the rule;
- `UnstableOverObservedWindow`: at least one metric does not.

This is intentionally conservative. If structural alternatives disagree widely, the query can remain uncertified even when a weighted average would look stable. The individual intervals are returned so that the hull does not erase visible scenario gaps from the report.

## What it does not mean

- It does not predict the next edit batch.
- It does not establish that local labels or synapses are correct.
- It does not infer causal effects or anatomical truth.
- It does not claim all queries are stable because one query is.
- It does not reconstruct ancestry from current state; the caller must provide edit and snapshot lineage artifacts.
- It does not independently verify artifact bytes against the SHA fields; ingestion must do that.
- It requires graph-builder configuration to remain fixed across the sequence; changing it requires a new query-stability run.

## Reproducible input and report contracts

- Input schema: `research/connectomics/connectome-query-stability-input-v0.1.schema.json`.
- Report schema: `research/connectomics/connectome-query-stability-report-v0.1.schema.json`.
- Synthetic input counterexample: `research/connectomics/fixtures/connectome-query-stability-known-answer-v0.1.json`.
- Synthetic report receipt: `research/connectomics/fixtures/connectome-query-stability-report-known-answer-v0.1.json`.

The report includes first/latest graph snapshot SHA-256, latest edit-history manifest digest, graph-builder configuration digest and structural-uncertainty manifest digest. Downstream workflow receipts can therefore bind a query-stability decision to the exact graph state and evidence set that were audited.

## Reproducible input schema and known-answer fixture

- Input schema: `research/connectomics/connectome-query-stability-input-v0.1.schema.json`.
- Synthetic counterexample: `research/connectomics/fixtures/connectome-query-stability-known-answer-v0.1.json`.

The fixture has two linked checkpoints. `global_motif_count` remains in [10.00, 10.01] across both, within tolerance 0.05. `target_path_strength` moves from [0.20, 0.21] to [0.80, 0.81], exceeding tolerance 0.02; the overall query must be `UnstableOverObservedWindow`. All fixture hashes are synthetic sentinels, not real artifact digests.

The schema checks structure and digest formats. The Rust input validator must additionally check the parent chain, monotonic edit counts, same-volume lineage, stable graph-builder configuration, exact metric/unit sets and actual bytes behind the declared hashes.

## Tests

Run in an environment with Rust:

```sh
cargo test --locked --manifest-path tools/connectome-query-stability/Cargo.toml
```

Thirteen unit tests are authored for query-specific drift, global-metric/local-edge disagreement, insufficient history, structural-hypothesis width, edit/snapshot ancestry, unit/configuration mismatches, digest and interval validation, missing/duplicate scenario and metric controls, and deterministic input ordering. They have **not** been compiled or executed in the current authoring environment.

## Research connection

See [SCI-CONNECTOMICS-056](https://github.com/Luminous-Dynamics/symthaea/issues/9994). The first validation target is independent reproduction of the 2026 graph-invariant convergence preprint. This crate is a query-level observed-stability auditor that can serve as a comparator; it is not a replacement for that forecasting method.
