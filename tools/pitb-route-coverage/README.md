# PITB retrieved support-route coverage analyzer

Standalone Rust analyzer joining per-query ranked retrieval receipts to independently measured exact-target support antichains.

## What it measures

Traditional premise retrieval metrics (Recall@K, MRR, nDCG) treat retrieved declarations as independent hits. Many Lean proofs require a *joint bundle*. This analyzer measures whether ranked candidates cover at least one complete minimal sufficient support route from an exhaustive exact-target premise-ablation experiment.

For each matched retrieval query and support group, it reports:

- **Complete-route coverage:** whether the retrieved declaration set contains every member of at least one minimal sufficient route.
- **Minimal-route coverage:** fraction of the observed antichain's routes covered.
- **Route completion rank:** the highest rank needed to complete the earliest route.
- **Partial-route fraction:** mean fraction of each route present in the retrieved set.
- **Useful route-premise rate:** fraction of exact-mapped mutable candidates appearing in at least one minimal route.
- **Necessary-declaration recall:** recall of the intersection of all observed minimal routes (null when the intersection is empty).
- **Exact minimal-route recovery:** whether the entire exact-mapped mutable candidate set equals a minimal route.
- **Coverage at K:** K = 1, 5, 10, 32, 64 and 100.
- Retrieval time, tokens and cost, reported once per unique receipt in system summaries.

The retrieval query binds to support groups using exact target-statement SHA-256 and environment SHA-256. If the same target/environment has multiple support groups (e.g. distinct fixed contexts or prover budgets), the query is evaluated against each group. This is a diagnostic comparison; it does not imply that the retrieval budget equals the downstream ablation budget.

## Required inputs

1. **Support-antichain report JSON** from `pitb-support-antichain`, schema `pitb.support-antichain-report.v0`.
2. **Retrieval receipt JSONL** from `pitb-retrieval-evidence`, one record per query/retriever run. Query rows must include exact target-statement and target-environment digests.
3. **Candidate declaration mapping JSONL.** Every returned ranked graph node must have exactly one mapping row, keyed by graph-manifest SHA-256 and node ID.

Mapping row:

```json
{
  "schema_version": "pitb.retrieval-declaration-mapping.v0",
  "graph_manifest_sha256": "<64 lowercase hex>",
  "node_id": "graph-node-id",
  "mapping_status": "exact",
  "declaration_id": "Mathlib.Algebra.Group.Defs.mul_assoc",
  "mapping_method": "exact_formal_name",
  "mapping_evidence_sha256": "<64 lowercase hex>",
  "reviewer_attestation_sha256": null,
  "source_locator": null,
  "statement_sha256": null
}
```

`mapping_status` is `exact`, `unresolved`, or `ambiguous`. Non-exact rows must have a null `declaration_id` and cannot contribute to route coverage. Mapping methods are `exact_formal_name`, `exact_source_identity`, or `human_reviewed`. Exact formal-name mappings are checked against the actual graph node's `formal_name`. Exact source-identity mappings must provide `source_locator` and `statement_sha256`, both checked against the graph node. Human-reviewed mappings require a reviewer-attestation digest and must resolve to a formal declaration node. The analyzer checks the referenced graph bytes and the declared node fields, but does not open or authenticate the mapping-evidence or reviewer-attestation bytes; those remain a separate verifier responsibility. “Novel” is deliberately not claimed: novelty requires a matched baseline candidate-set comparison.

## Paired comparisons and uncertainty

When multiple retrievers emit receipts for the same benchmark query and support group, the report builds paired comparisons only where query ID, support-group ID, benchmark/split, target statement, graph snapshot and environment match exactly.

For each system pair it reports reference-only/challenger-only/both/neither complete-route outcomes, the paired complete-route coverage delta, the mean minimal-route coverage delta, and an exact two-sided McNemar p-value. P-values are Holm-adjusted across the generated pairwise family.

The mean minimal-route coverage difference also receives a deterministic percentile bootstrap interval that resamples `component_id` clusters (the proof-family connected components), not individual rows. When fewer than two clusters are present, the interval is null rather than an IID approximation. A small sample's p-value is still descriptive; the bootstrap is not a substitute for more independent clusters or a preregistered confirmatory study.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-route-coverage/Cargo.toml
cargo test --locked --manifest-path tools/pitb-route-coverage/Cargo.toml
cargo run --locked --manifest-path tools/pitb-route-coverage/Cargo.toml -- --self-test

cargo run --locked --manifest-path tools/pitb-route-coverage/Cargo.toml -- \
  --analyze graph-manifest.json support-antichain-report.json \
  retrieval-runs.jsonl declaration-mappings.jsonl retrieval-route-coverage.json
```

## Interpretation boundaries

A complete route is sufficient only in the frozen experiment that generated the antichain: exact target, fixed context, Lean environment/toolchain, prover policy and resource budget. It is not a theorem-level logical-necessity result.

This analyzer does not establish that a node-to-declaration crosswalk is semantically correct merely because it is labelled exact. It requires a provenance digest but does not authenticate the referenced bytes. It does not replay proofs, prove that a model used the candidates, or make external candidates scheduler-admissible. The output is retrieval-to-route coverage evidence, **not proof completion**.

For fair comparison, run BM25, LeanSearch v2, Lean Finder V2, LeanGraphSearch, DAPS and a PITB union/reranker on the same query, graph snapshot, split, candidate-pool policy and downstream budget. Compare complete-route coverage and cost, then run each arm through the same prover/checker to measure unique assurance-adjusted target completion. Do not compare raw retriever scores across systems or treat candidate-route coverage as a solved theorem.
