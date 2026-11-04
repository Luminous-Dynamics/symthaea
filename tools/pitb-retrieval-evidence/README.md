# PITB per-query retrieval evidence

Standalone Rust validator for per-query ranked candidate receipts. It gives DAPS, LeanGraphSearch, LeanSearch v2, TheoremGraph, BM25 and PITB unions one auditable retrieval-output contract.

## Why this layer is needed

A retrieval metric such as Recall@32 does not identify which premises were actually returned for each query, under which candidate pool and graph snapshot, by which model/checkpoint and ranking configuration, or at what resource cost. Aggregate CSVs are useful scorecards but are insufficient to reproduce the exact candidate context later passed to a prover.

The pinned DAPS Phase-2 evaluator currently:
- creates the filtered candidate pool as `selector_ids` plus premise embeddings;
- computes `sims = batch @ premise_embs.T` (raw dot products);
- calls `torch.topk` but discards the score values, retaining only row indices;
- maps row indices to DAPS graph IDs only indirectly through `selector_ids`;
- writes aggregate metric CSVs, not per-query ranked candidate IDs and scores.

PITB needs the per-query output. The receipt must preserve both the returned order and the source system's ID mapping. DAPS scores are raw dot products, not calibrated probabilities.

## Receipt contract

Every query receipt binds:
- benchmark ID/version, split ID/role, split-manifest digest and label-sequestering declaration;
- query ID, exact query-text digest, optional target-statement digest and target environment digest;
- exact PITB graph snapshot ID, repository commit, environment and raw graph-manifest SHA-256;
- retriever kind/name/version, source code repository commit, implementation digest, configuration digest, query-formulator digest and (for DAPS learned selector) checkpoint digest;
- filtered candidate-pool node IDs, candidate-pool policy digest, node-set digest, requested K;
- ordered candidates with contiguous 1-based rank, canonical PITB node ID, raw upstream candidate ID, preserved source relation label, raw score and a digest of its upstream evidence record;
- query wall time, retrieval CPU/GPU time, tokens and cost.

A matched experiment can include the same query once per retriever. Repeating the same query under the same benchmark/split/retriever identity is rejected. Candidate pools may be strict filtered subsets of the full graph, which is essential for DAPS's type/blacklist filtering.

All retrieval receipts remain `external_candidate_run` and explicitly declare `scheduler_admission=false`. The validator rejects unknown graph nodes, candidates outside the declared filtered pool, stale graph digest/environment, duplicate ranks, duplicate candidates, invalid score ordering, unpinned DAPS checkpoints, exposed labels, and candidate receipts claiming scheduler admission.

## Score semantics

- DAPS Phase-2: `dot_product`, `higher_better` (the pinned code multiplies query embeddings by premise embedding transpose; it currently discards values after `torch.topk`).
- Other learned retrievers must declare their actual raw metric, not relabel it as cosine or probability.
- Rank-only systems may declare `rank_only` and carry null scores.
- All raw external scores are marked `uncalibrated`. Calibration requires its own held-out calibration task set and report.

Scores from different retrievers are not directly comparable. Ranks are comparable only as ordinal positions within one candidate pool and query. Use a common downstream proof budget and measure verified target completion and cost, not just retrieval metrics.

## DAPS export requirement

A minimally sufficient change to the pinned DAPS evaluator is to retain values returned by `torch.topk` and emit one JSONL receipt per query. The export logic must map each returned row index through `selector_ids`, then map the DAPS graph node name to the canonical PITB node ID used by the imported graph. Do not serialize the row index as though it were the theorem identity.

Conceptually:
1. change `_, top_idx = torch.topk(sims, ...)` to `top_scores, top_idx = torch.topk(sims, ...)`;
2. for each query row, map each `top_idx` to `selector_ids[top_idx]`;
3. retrieve the full DAPS node name and map it to the imported PITB `node_id`;
4. preserve the integer DAPS candidate ID, original relation/source metadata and raw dot-product value;
5. bind the model checkpoint file SHA-256, evaluator script Git blob, graph blob/manifest, filter config digest, exact split file/cache digest, query-formulator digest and query text digest;
6. write one receipt per query. Keep the gold positives in a separate label file; never expose them to the retriever input.

The adapter crate does not patch or execute DAPS/PyTorch. It verifies the receipt file after a baseline's official evaluator/exporter produces it.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-retrieval-evidence/Cargo.toml
cargo test --locked --manifest-path tools/pitb-retrieval-evidence/Cargo.toml
cargo run --locked --manifest-path tools/pitb-retrieval-evidence/Cargo.toml -- --self-test

# Generate a schema-checkable synthetic receipt/report against a real PITB graph manifest:
cargo run --locked --manifest-path tools/pitb-retrieval-evidence/Cargo.toml -- \
  --self-test-report \
  tools/pitb-graph-evidence/fixtures/graph-evidence.synthetic.json \
  /tmp/retrieval-runs.synthetic.jsonl \
  /tmp/retrieval-report.synthetic.json

# Verify a measured retrieval run:
cargo run --locked --manifest-path tools/pitb-retrieval-evidence/Cargo.toml -- \
  --verify path/to/graph.json path/to/retrieval-runs.jsonl path/to/report.json
```

## Relationship to support antichains and SOTA comparison

The receipt makes it possible to run a fair paired comparison of DAPS, LeanGraphSearch, LeanSearch v2, TheoremGraph, BM25, and their union on identical target/environment/query identities. The next stage joins ranked candidates to counterfactual support ablations:
- retrieval recall@K and complete-route coverage remain diagnostic;
- useful-novel-premise rate measures whether a retrieved item contributes to any independently verified minimal route;
- bundle sufficiency measures whether a joint retrieved set closes the target;
- minimal-route recovery compares the retrieved bundle against the exact support antichain;
- verified exact-target completion and cost per verified completion are the primary outcomes.

A high Recall@K does not demonstrate that the returned set is sufficient, minimal, logically necessary or even type-compatible. A candidate can enter the scheduler only after it has been independently verified against the exact pinned Lean environment and typed evidence contract.
