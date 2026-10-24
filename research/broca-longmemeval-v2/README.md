# Broca × LongMemEval-V2: fail-closed evaluation lane

**State:** protocol and receipt gate implemented; benchmark not yet executed. No Broca accuracy, LAFS gain, cost advantage, or state-of-the-art claim is made here.

## Why V2 is the next external target

The official [LongMemEval-V2 repository](https://github.com/xiaowu0162/LongMemEval-V2) describes 451 manually curated questions across web and enterprise domains, with up to 500 trajectories and 115M tokens per haystack. It evaluates static-state recall, dynamic-state tracking, workflow knowledge, environment gotchas, and premise awareness. Its supported memory baselines include no retrieval, raw state-slice retrieval, state slices plus notes, AgentRunbook-R, Codex, and AgentRunbook-C. The official score frontier uses answer accuracy and mean memory-query latency.

These properties make V2 a complementary target to conversation-history QA: it asks whether an agent can use prior experience to act effectively in a changing environment. Keep LongMemEval V1 and V2 as different benchmark IDs; never compare or merge their scores.

## Required experiment design

The receipt gate requires a run to pin the upstream V2 commit, dataset digest, official harness digest, Symthaea commit, shared experiment config digest, reader and judge model revisions, reader/judge prompt digests, context-token cap, and the official reference-frontier digest. It requires both **web** and **enterprise** outputs for every arm and exactly 451 examples per arm across those domains. The three minimum matched arms are:

- **broca**: the candidate Symthaea-backed memory adapter;
- **no_retrieval**: the official no-memory control;
- **rag_query_to_slice**: the official raw-state retrieval baseline.

Additional official arms are accepted when run. Every arm must have exactly one completed result for each domain and a completed combined aggregate. Per-domain outputs must be canonical JSONL: one object per question with a unique **case_id**. The gate verifies each output has exactly its declared number of records and that the question-ID set is identical across matched arms within each domain. Each aggregate must carry all six official numeric metrics. The dataset manifest, harness manifest, reference frontier, shared config, domain outputs, and combined aggregates are byte-level hash checked relative to the receipt directory. Partial, skipped, duplicate, unpinned, mismatched, unpaired, missing, or path-traversing artifacts fail closed.

The adapter must use the official upstream evaluation harness and leaderboard aggregator for answer scoring and latency accounting. This repository's checker does **not** reimplement LAFS, judge answers, verify that the official harness was scientifically correct, or infer memory quality from retrieval hit rates. It checks receipt structure and byte-level integrity only.

## Run the local protocol controls

    scripts/test_broca_lme_v2_receipt.sh
    scripts/check_broca_lme_v2_receipt.sh path/to/receipt.json

The tests create temporary protocol-shaped fixtures only. Fixture pass output is not benchmark evidence. A real receipt's artifact paths must resolve to files under the directory containing that receipt.

## What to report

Report the official metrics separately: **overall_full_set**, **gotchas_accuracy**, **static_accuracy**, **dynamic_accuracy**, **procedure_accuracy**, and **memory_query_avg_seconds**, followed by the official LAFS gain at each operating point. Also report the Broca-vs-no-retrieval paired task-success lift under the same reader, judge, prompts, memory token cap, dataset, and question IDs. Retrieval Recall@K is diagnostic only and must not substitute for end-to-end task score.

For claims beyond the official aggregate, preserve per-question outcomes and report bootstrap intervals grouped by independent user/session/source when the benchmark metadata supports it. Show model costs, full memory storage cost, ingestion and query latency, p50/p95 where measurable, and all critical integrity failures. Any stale memory, unauthorized exposure, or deleted-memory resurrection is an explicit blocker, not something averaged away by accuracy.

## Practical execution gate

The upstream release documents Python 3.11 and endpoint configuration for a fixed reader, a judge, and embedding/controller models for baseline methods. Broca should remain Rust-first: keep the memory policy and durable state in Rust and use only a thin harness adapter where the upstream API requires it. Do not silently replace the official reader, judge prompt, context cap, screenshot path, or evaluator when those defaults are too expensive; create a separately named substrate-only pilot instead.

Before the first real run, use an exact upstream checkout; validate the selected tier and downloaded data with upstream scripts; run the three required arms on both domains; use the upstream aggregation tool; then generate a receipt with hashes for every per-domain output and combined aggregate. The current patch establishes the gate and its negative controls, **not** the adapter implementation or a completed evaluation.
