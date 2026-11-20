# CDL-Bench: query-affinity grouping and distribution drift (v1)

**Date:** 2026-11-20  
**Tracking:** [CDL-Bench-017 #12984](https://github.com/Luminous-Dynamics/symthaea/issues/12984), [CDL-Bench-014 #12964](https://github.com/Luminous-Dynamics/symthaea/issues/12964)  
**Status:** local C++20 synthetic benchmark; deterministic tests passed; no public dataset, disk I/O, measured latency or SOTA claim

## Research question

PZG2 showed a size/selectivity trade-off from compression-group size. This experiment holds group size fixed and changes key-to-group placement. It asks whether access affinity can reduce query bytes without materially harming complete-frame size, and how a layout trained on one query distribution behaves after workload shift.

The important distinction is between:
- **compression affinity:** placing streams with similar byte/change statistics together;
- **query affinity:** placing streams that are requested together in the same read group;
- **service/topology affinity:** placing services or containers that belong to one dependency or ownership group together.

These objectives can conflict.

## Exact experiment

- 4 synthetic workloads: sparse, bursty, mostly idle, noisy.
- 65,536 rows, 64 keys, 1,024 records per key; fixed seed.
- Zstandard 1.5.7, level 3.
- Group sizes: 1, 2, 4, 8, 16, 32, 64.
- Policies: first-seen; interleaved entity/series; deterministic random; greedy train-query-affinity; test-query-oracle affinity; compression-affinity.
- Train profile: 70% service-cohort queries, 20% same-series cross-service queries, 10% singleton queries.
- Shifted test profile: 20% service-cohort, 70% cross-service, 10% singleton.
- Query-affinity edges are derived only from the profile supplied to the policy; the test-oracle policy is deliberately non-deployable and serves only as a held-out upper-bound control.
- Compression-affinity uses a fixed per-series feature vector of field change rates, missingness and encoded bytes per row, with deterministic cosine-similarity grouping.

The query model sums actual compressed group bytes fetched and raw group bytes decompressed for every weighted query in the frozen profile. It also counts how many series are decoded. It does not simulate disk seek time, cache effects, network or CPU latency.

## Group size 8: storage versus shifted-query costs

| Workload | First-seen complete frame | Interleaved complete frame | Test fetched bytes, first-seen → interleaved | Test decompressed bytes, first-seen → interleaved |
|---|---:|---:|---:|---:|
| Sparse | 74,293 B | 74,283 B | 30,251 → 21,985 B | 57,458 → 41,787 B |
| Bursty | 228,055 B | 228,436 B | 93,696 → 68,248 B | 131,416 → 95,575 B |
| Mostly idle | 6,192 B | 6,229 B | 2,225 → 1,619 B | 2,772 → 2,016 B |
| Noisy | 2,443,911 B | 2,444,704 B | 1,007,702 → 733,095 B | 1,499,304 → 1,090,403 B |

Under the shifted profile, interleaved placement reduces weighted query bytes fetched/decompressed by approximately 27% for each fixture, while complete-frame storage changes by less than 0.6%. Average decoded series per query falls from 26.4 to 19.2; decompression amplification falls from 6.6× to 3.0×.

This is not a universal win. Under the train profile, first-seen grouping is better because that profile heavily favors same-service cohorts. At group size 8:
- Sparse train fetch cost: first-seen 21,084 B vs interleaved 54,046 B.
- Bursty: 65,303 B vs 167,776 B.
- Idle: 1,551 B vs 3,981 B.
- Noisy: 702,338 B vs 1,802,191 B.

The greedy train-query-affinity policy selects the first-seen layout for this profile. The test-oracle policy selects an interleaved layout. The test-oracle is not evidence of an online learner; it illustrates how much the best placement can change under query shift.

## Compression affinity is not query affinity

At group size 8, compression-affinity changes storage by only about 0.1–0.2% versus first-seen, but worsens shifted test-query bytes fetched by about 14–23% on these fixtures:
- Sparse: 37,112 B vs 30,251 B.
- Bursty: 107,027 B vs 93,696 B.
- Idle: 2,624 B vs 2,225 B.
- Noisy: 1,230,133 B vs 1,007,702 B.

The small storage differences do not justify the query penalty under this access model. Similar compression behaviour is not a proxy for co-access.

## Interpretation

1. For this synthetic workload, group placement has a much larger effect on query cost than on complete storage size.
2. One-time query-affinity grouping can become stale when service ownership, dependency topology or operator queries change.
3. A deployable system needs drift-aware grouping epochs, bounded compaction/re-sharding, or a secondary access overlay. Any such approach must count its metadata, rewrite bytes, CPU and temporary space.
4. The grouping policy must train on historical query windows and be evaluated on future windows, held-out runs and changed topology.
5. The objective should be a weighted storage/read/CPU/RSS Pareto surface, not a single compression score.

## Reproduction

Source and runner names:
- `research/cdl-bench/reference/grouping_policy_reference.cpp`
- `scripts/research/cdl_grouping_policy_reference.sh`

Run with `ZSTD_LEVEL=3` to reproduce this table. The program reuses the PZG2 C++20 frame reference, runs its core group tests, parses each candidate directory, and performs complete round-trip checks for representative sparse/group-8 policies. This is synthetic evidence only; the public dataset and real query replay remain pending under [CDL-Bench-015 #12970](https://github.com/Luminous-Dynamics/symthaea/issues/12970).

## Next experiments

- Drift detector + re-shard budget: trigger only when expected future-query gain exceeds rewrite cost.
- Dual/overlay access path: retain compression-friendly physical groups while adding a low-cost secondary index or hot-key indirection for cross-service reads.
- Multi-epoch layout: retain old and new epochs during bounded compaction; verify event identity and deduplication.
- Query cost oracle: compare greedy grouping with an offline min-cut/partition upper bound, but keep oracle data out of training.
- Real dataset replay: LO2v2 first, then Alibaba; split complete runs and query windows chronologically.
