# PITB per-query retrieval evidence

Standalone Rust receipt validator for per-query ranked candidates from BM25, DAPS, LeanSearch v2, Lean Finder V2, LeanGraphSearch, TheoremGraph and PITB retrieval unions.

## Evidence contract

Each query receipt binds the exact benchmark/split, query text digest, optional target statement digest, graph snapshot and manifest digest, repository commit, Lean environment, retriever implementation/configuration/query-formulator hashes, optional learned-model checkpoint, filtered candidate node set and policy, ordered ranked candidates, score semantics and resource budget.

Every returned candidate must be a node in the pinned graph and inside the declared filtered candidate pool. Ranks are contiguous and unique. Scores must be finite and agree with their declared ordering direction. External scores remain `uncalibrated`; rank-only systems use null values. DAPS selector receipts require an exact checkpoint digest.

The same benchmark query may appear once per retriever for paired comparisons. A duplicate query/retriever identity is rejected. All receipts are `external_candidate_run` and `scheduler_admission=false`.

## DAPS note

The pinned DAPS Phase-2 evaluator computes raw dot products using `batch @ premise_embs.T`, calls `torch.topk`, and keeps row indices rather than exporting per-query scores. A proper exporter must retain the top-k scores and map every row index through `selector_ids`, then to the upstream DAPS theorem ID and finally to the canonical PITB graph node ID. A row index is not a theorem identifier. The current crate validates an export produced by that adapter; it does not execute PyTorch.

## Comparison protocol

Use the same exact query IDs, graph snapshot, candidate-pool policy where feasible, resource budget and downstream prover for each system. Candidate pools that differ by documented filtering are allowed but must be explicit. Score fields cannot be compared across different retrievers as probabilities or common scales.

Report:
- nDCG@K, Recall@K, complete proof-route coverage;
- useful-novel-premise rate and support-antichain route recall;
- exact-target proof success under one shared downstream budget;
- wall time, CPU/GPU time, tokens and cost per unique verified task.

The retrieval receipt proves what was returned, not that a premise is useful, sufficient, logically necessary or safe to admit into a proof scheduler.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-retrieval-evidence/Cargo.toml
cargo test --locked --manifest-path tools/pitb-retrieval-evidence/Cargo.toml
cargo run --locked --manifest-path tools/pitb-retrieval-evidence/Cargo.toml -- --self-test
```
