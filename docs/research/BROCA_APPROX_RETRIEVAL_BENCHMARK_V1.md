# Broca approximate-retrieval benchmark v1

Status: research harness; performance and quality are **unmeasured** until the command is run on a recorded machine. This benchmark is not a claim of competitive performance or an SOTA result.

## Question and falsifiable hypothesis

Can the existing LSH candidate path in `symthaea-hdc-store` cut the number of examined vectors and p95 search latency versus the exact full-store scan while preserving top-K recall?

This isolates the store retrieval path. It does not yet measure end-to-end `MemoryBridge` conversion, bundling, blending, task answer quality, or memory utility. It also does not compare against HNSW, FAISS, pgvector, BM25, or a hybrid retriever; those are required before making an industry comparison.

## Reproduce

Use a release build and pin the exact 40-hex source commit. Example:

```sh
cargo run -p symthaea-hdc-store --example ann_benchmark --release -- \
  --source-sha="$(git rev-parse HEAD)" \
  --size=10000 --queries=200 --rounds=5 --top-k=10 --seed=42
```

Run at multiple store sizes, using the same machine and compiler build:

```sh
for size in 1000 10000 50000; do
  cargo run -p symthaea-hdc-store --example ann_benchmark --release -- \
    --source-sha="$(git rev-parse HEAD)" \
    --size="$size" --queries=200 --rounds=5 --top-k=10 --seed=42 \
    > "broca-ann-$size.csv"
done
```

The LSH parameters are explicit and recorded in the output. The default store configuration is `--lsh-bands=32 --lsh-rows=8`; vary these in separate pinned runs when tuning the index, and never compare arms whose source, store, query corpus or cache protocol differs.

To test the approximate path with its candidate-count and empty-result fallbacks disabled, run a separate arm with `--candidate-multiplier=0 --fallback-on-empty=false`. Keep the default-fallback arm as well: disabling fallback may improve latency while degrading recall, so neither arm is sufficient by itself.

The example includes focused tests for deterministic workload replay, exact corruption count, Recall@K denominator behavior and latency percentile calculation. Run them with:

```sh
cargo test -p symthaea-hdc-store --example ann_benchmark
```

The CLI rejects an unpinned or malformed source SHA, unknown arguments, zero size/query/round/top-K values, and noise levels above the vector dimension.

## Workload contract

- The store contains deterministic, seeded 16,384-bit BinaryHV vectors, inserted through the actual HdcStore API.
- Eighty percent of query cases are probes based on an inserted vector with a deterministic 128-bit corruption by default; twenty percent are generated novel probes.
- Exact search defines the reference top-K IDs for that same query and store. Recall@K is intersection size divided by the number of exact top-K results (which is `min(K, live_count)`).
- The LSH path reports actual examined vectors, the live-vector count, whether the result was exhaustive, and whether it fell back to exact search.
- Exact and approximate retrieval calls alternate order per query/round. Query creation, store construction, ID comparison and CSV output are outside the timed retrieval intervals.
- The report records a unique-query quality summary from round zero and latency percentiles across all repeated query samples. Repeated rounds are latency repetitions, not independent statistical samples.

This is a controlled synthetic workload, not a realistic semantic-memory corpus. Seeded random vectors and bounded bit flips do not reproduce the geometry, duplicate rates, clustered topics, stale facts, user/session scopes or semantic neighborhoods found in long-running agents. A strong result here is a retrieval-substrate signal only.

## Initial acceptance targets (proposed, not achieved)

Keep the existing `AnnValidationThresholds` expectations aligned with this report. For a store size of 10,000 or more and at least 100 unique queries, the approximate path should meet **all** of the following in the evaluated workload:

| Measure | Initial target |
|---|---:|
| Mean Recall@K against exact results | ≥ 0.95 |
| Worst individual-query Recall@K | ≥ 0.80 |
| Mean fraction of live vectors examined | ≤ 0.50 |
| Fraction of exhaustive queries | ≤ 0.50 |
| Fraction of queries that fell back to exact | ≤ 0.05 |
| p95 retrieval latency relative to exact | ≥ 2× faster (exact p95 / approximate p95 ≥ 2) |

These are engineering gates to test and recalibrate, not published SOTA thresholds. Report every raw query, not only aggregate averages. A failure is useful evidence; do not loosen thresholds or omit failed query classes after seeing the result.

## Industry comparison: keep two evaluation layers separate

### Binary nearest-neighbor systems

The correct first external comparison is a metric-matched binary-vector benchmark, not a comparison against a dense-vector system using a different representation. Candidate baselines:

- **Exact anchor:** this HdcStore full scan.
- **FAISS binary indexes:** `IndexBinaryFlat` for optimized exhaustive ground truth and `IndexBinaryHNSW` or `IndexBinaryIVF` for approximate search. FAISS documents these indexes specifically for Hamming distance over byte-packed binary vectors: https://github.com/facebookresearch/faiss/wiki/Binary-Indexes
- **pgvector binary HNSW:** `bit` vectors with `bit_hamming_ops`, which supports a database-backed baseline with an operational/storage layer: https://github.com/pgvector/pgvector

The published FAISS binary-hashing methodology emphasizes recall alongside wall-clock latency, distance computations and random memory accesses: https://github.com/facebookresearch/faiss/wiki/Binary-hashing-index-benchmark. Match the exact vector bytes, queries, Hamming metric and top-K; capture build time, search p50/p95, recall, memory overhead, insert/delete/update cost and recovery/rebuild cost. Report adapter/database overhead separately from raw index time. Do not compare the current 16,384-bit vectors with FAISS's 256-bit example workloads and call the results matched.

### Agent-memory quality

A store-level Recall@K improvement is not sufficient evidence that an agent remembers or acts better. MemoryAgentBench describes four separate capabilities—accurate retrieval, test-time learning, long-range understanding and selective forgetting: https://arxiv.org/abs/2507.05257. Run that task-level lane, plus the Agent Memory Leaderboard's current public track when deployment is available: https://agentmemories.ai/docs. For the latter, the current Cycle 2 guide specifies a materials deadline of October 31, 2026 at 23:59 UTC+8 (17:59 in Johannesburg) and requires participant-hosted Add/Search APIs; repository-only submissions are not enough. Deployment feasibility is tracked separately in [PR #8476](https://github.com/Luminous-Dynamics/symthaea/pull/8476).

Do not pool the binary ANN result, MemoryAgentBench scores and the external leaderboard score into one number. They measure different layers and have different task and deployment contracts.

## Machine and run receipt

Archive the complete CSV (including its metadata and per-query exact/approximate IDs) alongside:
- exact commit SHA and clean/dirty source state;
- `rustc -Vv`, Cargo version, build flags and CPU model;
- OS/kernel, memory, storage medium and power mode;
- exact command line and exit status;
- whether each sample is warm-cache, cold-cache or mixed, and whether the store was reconstructed between repetitions.

The current program reports OS, target architecture, available parallelism, workload settings, source SHA, p50/p95 latency, recall, vector examinations and fallback/exhaustive fractions. A fully cold-cache test, memory/RSS accounting, energy measurements, production-scale corpus and confidence intervals are not implemented by this first slice. Record them separately rather than implying that the current CSV contains them.

## Next gates

1. Run the exact source-bound benchmark with default options and no-fallback options at 1k, 10k and 50k vectors.
2. Add clustered, duplicate, contradictory, stale-update, delete and scope-isolation workloads; report each class separately.
3. Compare against at least one established ANN implementation and a brute-force SIMD baseline on the identical vectors, queries and top-K. Include index build time, memory overhead, update/delete cost and cold-start behavior.
4. Wire the winning search policy into an opt-in MemoryBridge experiment only after recall and fallback criteria pass.
5. Measure end-to-end task success and error classes against no-memory, exact HDC, approximate HDC, and industry-standard retrieval baselines at matched budgets.

No result should be described as an industry win until these matched comparisons have actually run.
