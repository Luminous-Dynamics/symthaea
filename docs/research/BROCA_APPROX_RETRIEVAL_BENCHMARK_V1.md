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
  --size=10000 --clusters=64 --cluster-noise-bits=1024 --noise-bits=128 --hard-noise-bits=4096 --lsh-configs=32x8,16x16,64x4 --queries=200 --rounds=5 --top-k=10 --seed=42
```

Run the recommended multi-size/fallback matrix through the checked-in helper. It captures the exact source SHA, Rust/Cargo versions, machine information, command, exit code, CSV SHA-256, stderr log path and stderr SHA-256 in a sidecar receipt for each run:

```sh
bash scripts/run_broca_ann_sweep.sh
```

By default this performs 18 benchmark runs (1k / 10k / 50k vectors × seeds 42 / 43 / 44 × empty-result fallback on/off), at 200 queries × 5 timing rounds, and writes separate CSVs, stderr logs and receipts under `artifacts/broca-ann/`. Seeds are fixed before the experiment; do not drop an unfavorable seed after observing results. Override the workload without editing the script:

```sh
SIZES="10000 50000" SEEDS="11 12 13 14 15" QUERIES=500 ROUNDS=7 OUT_DIR=/tmp/broca-results \
  bash scripts/run_broca_ann_sweep.sh
```

After the runs finish, evaluate every CSV with the fail-closed proposed-gate checker:

```sh
bash scripts/check_broca_ann_gates.sh artifacts/broca-ann/*.csv
```

It first verifies the sidecar records exit status 0 and that the CSV SHA-256, stderr SHA-256, and source SHA match their receipts; then it checks CSV completeness, three-stratum coverage, at least ten near-neighbor source groups, and every proposed low-noise Recall@K/candidate/fallback/p95-speedup gate. It fails on missing metrics or incomplete raw rows. It **never certifies hard-neighbor routing**: that needs a preregistered unsafe event and an independent risk qualification step. The gate thresholds are research decision rules, not published industry standards.

For a single controlled arm or an explicit candidate-multiplier sweep, use the direct `cargo run` command above. Keep the same seed, query plan, vector dimension, top-K and cache protocol across matched arms.

The LSH parameters are explicit and recorded in the output. By default, each run sweeps three arms: `32x8`, `16x16`, and `64x4` (bands × rows). Override them with `--lsh-configs=32x8,16x16,64x4`; no arm may be duplicated and a run is limited to 16 arms. The same deterministic corpus and query vectors are reused across all arms, while each arm gets a separately built index. Do not compare arms whose source, vector corpus, query corpus or cache protocol differs.

To test the approximate path with its candidate-count and empty-result fallbacks disabled, run a separate matrix with `--candidate-multiplier=0 --fallback-on-empty=false`. Keep the default-fallback matrix as well: disabling fallback may improve latency while degrading recall, so neither matrix is sufficient by itself.

The example includes eleven authored source-level tests for deterministic workload replay and stratum composition, exact corruption count, clustered-neighborhood separation, Recall@K semantics including ties, LSH sweep parsing, and deterministic/group-aware bootstrap behavior. Rust tests have not yet run in this authoring environment. Run them with:

```sh
cargo test -p symthaea-hdc-store --example ann_benchmark
```

The CLI requires a full 40-hex source SHA that exactly matches `git rev-parse HEAD`, and it rejects a dirty tracked working tree. This binds results to the checked-out commit rather than merely echoing a caller-supplied identifier. It also rejects unknown arguments, zero size/query/round/top-K values, a cluster count outside `1..=size`, and either noise level above the vector dimension.

## Workload contract

- The store contains deterministic, seeded 16,384-bit BinaryHV vectors inserted through the real HdcStore API. Each vector is a deterministic prototype for one of `--clusters=64` clusters, corrupted with `--cluster-noise-bits=1024` flips by default. This creates controlled, non-identical neighbors within a cluster and separated prototypes across clusters.
- The default 200-query workload has three explicit strata: 60% low-noise known-memory probes (`--noise-bits=128`), 20% hard-neighbor probes (`--hard-noise-bits=4096`), and 20% random out-of-distribution (OOD) probes. Each known-memory query starts from a stored anchor, so the hard stratum preserves a real nearest-neighbor target while substantially increasing its Hamming perturbation; it probes whether LSH can miss the true neighbor even when that neighbor is present.
- The exact full scan defines reference top-K IDs. **Exact-ID Recall@K** is intersection size divided by the number of exact top-K results (which is `min(K, live_count)`). The harness also reports **tie-aware Recall@K**, which treats an approximate neighbor as equivalent when its similarity is at least the exact Kth-neighbor similarity. This controls for interchangeable Hamming-score ties at the top-K boundary. Exact-ID recall remains the conventional strict metric; tie-aware recall is a complement, not a replacement.
- For the synthetic OOD class, exact top-K is still a mathematical retrieval reference but does not imply there is a semantically relevant answer; do not use its Recall@K as the primary quality gate.
- The LSH path reports vectors examined, total live vectors, exhaustive status and exact-fallback status.
- Exact and approximate retrieval calls alternate order per query/round. Query creation, store construction, ID comparison and CSV output are outside the timed retrieval intervals.
- The report separately emits quality and latency summaries for `near_neighbor`, `hard_neighbor`, and `novel` queries for each LSH configuration. Quality uses unique queries from round zero; latency percentiles use all repeated samples. Each query's median exact/approximate speedup across rounds is computed first. The speedup interval then uses a deterministic 2,000-resample **group bootstrap**: known-memory probes are grouped by the source cluster that generated their anchor, while each random novel/OOD probe is a singleton group. Every query within a selected group travels together during resampling. The raw CSV includes `group_id`, and the report records the resampling unit and group count. Repeated timing rounds remain timing repetitions, not independent query samples. This corrects a dependence issue in the earlier query-level bootstrap; it is still synthetic-query uncertainty, not hardware/run-to-run uncertainty.

This controlled clustered workload is still not a realistic semantic-memory corpus. It does not reproduce organic topic structure, duplicate/contradictory facts, temporal updates, deletion semantics, user/session isolation or long-running agent behavior. Passing these gates is a retrieval-substrate signal only, not evidence that downstream memory tasks improve.

## Initial acceptance targets (proposed, not achieved)

For a store size of 10,000 or more and at least 100 unique queries, the approximate path should meet **all** of the following initial gates on the `near_neighbor` class. Report `hard_neighbor` as a separate adversarial stratum with the same per-class metrics, plus an explicit hard-query unsafe-acceptance risk analysis before a router is allowed to accept it; do not use the aggregate easy-query score to mask hard-query failures. The `novel` OOD class is reported separately because arbitrary random vectors do not define a meaningful answerability target. These are proposed engineering targets, not published SOTA thresholds:

| Measure | Initial target |
|---|---:|
| Mean exact-ID Recall@K against exact results | ≥ 0.95 |
| Worst individual-query exact-ID Recall@K | ≥ 0.80 |
| Mean tie-aware Recall@K | ≥ 0.98 |
| Mean fraction of live vectors examined | ≤ 0.50 |
| Fraction of exhaustive queries | ≤ 0.50 |
| Fraction of queries that fell back to exact | ≤ 0.05 |
| p95 retrieval latency relative to exact | ≥ 2× faster (exact p95 / approximate p95 ≥ 2) |
| Median per-query speedup, lower bootstrap 95% bound | > 1× (evidence against a no-speedup result) |

Report every raw query and all three classes, not only aggregate averages. A failure is useful evidence; do not loosen thresholds or omit failed query classes after seeing the result. Add a separate, labeled abstention/no-relevant-evidence evaluation before making claims about OOD behavior.

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

Archive the complete CSV (including its metadata, per-query exact/approximate IDs, and group ID used for uncertainty resampling) alongside:
- exact commit SHA and clean/dirty source state;
- `rustc -Vv`, Cargo version, build flags and CPU model;
- OS/kernel, memory, storage medium and power mode;
- exact command line and exit status;
- whether each sample is warm-cache, cold-cache or mixed, and whether the store was reconstructed between repetitions.

The current program reports OS, target architecture, available parallelism, workload settings, source SHA, per-arm p50/p95 latency, exact-ID and tie-aware recall, vector examinations and fallback/exhaustive fractions, plus per-query-class median speedup and deterministic group-bootstrap 95% intervals. Near-neighbor queries inherit cluster membership from their source memory, so resampling individual rows as though they were independent would overstate the independent evidence when within-cluster outcomes co-vary. Cluster-robust inference guidance emphasizes that the cluster—not an individual row—is the independent unit when the sample is clustered (see [MacKinnon, Nielsen & Webb, cluster-robust inference guide](https://doi.org/10.1016/j.jeconom.2022.04.001)). The synthetic OOD probes are treated as singleton groups because they have no stored source anchor in this generator. A fully cold-cache test, memory/RSS accounting, energy measurements and production-scale corpus are not implemented by this first slice. The interval does not cover hardware-to-hardware or run-to-run variation and does not establish reliability under real memory distributions.

## Next gates

1. Run the source-bound three-arm LSH sweep with default fallback settings and with no-fallback settings at 1k, 10k and 50k vectors; vary cluster counts/noise with the same seeded query plan.
2. Add duplicate, contradictory, stale-update, delete and scope-isolation workloads; report each class separately.
3. Compare each winning internal operating point against at least one established ANN implementation and a brute-force SIMD baseline on identical vector bytes, queries and top-K. Include index build time, memory overhead, update/delete cost and cold-start behavior.
4. Wire the winning search policy into an opt-in MemoryBridge experiment only after recall and fallback criteria pass.
5. Measure end-to-end task success and error classes against no-memory, exact HDC, approximate HDC, and industry-standard retrieval baselines at matched budgets.

No result should be described as an industry win until these matched comparisons have actually run.
