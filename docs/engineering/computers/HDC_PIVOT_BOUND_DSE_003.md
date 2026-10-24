# HDC-PIVOT-BOUND-DSE-003 — exact metric lower bounds for BinaryHV retrieval

Status: experiment and benchmark authored; Rust compilation and wall-clock measurements have not been performed from this environment.

## Paradigm shift

The current exact scan evaluates the full 2,048-byte candidate vector for every candidate unless early-abandonment can reject it after reading part of the vector. This experiment asks whether a compact *metric signature* can avoid many full-vector accesses before reading those vectors.

For any query q, candidate x, and pivot p, Hamming distance obeys the triangle inequality:

    d(q, x) >= |d(q, p) - d(x, p)|

For a set of pivots P, define:

    LB(q, x) = max over p in P of |d(q, p) - d(x, p)|

LB is a lower bound on the exact Hamming distance. Once an exact incumbent with distance D is known, any candidate with LB > D cannot improve the answer. If LB = D, a later candidate can also be pruned because public tie policy chooses the lowest candidate ID; a lower-index candidate must still be checked. The index never returns an answer based on the bound alone: the winner and incumbent come from exact BinaryHV Hamming distance.

## New Rust benchmark

File: `crates/core/symthaea-core/benches/hdc_pivot_bound_search.rs`

Run:

    cargo bench -p symthaea-core --bench hdc_pivot_bound_search

Compared paths:
- exhaustive exact Hamming top-1;
- pivot-bound exact top-1 with 2, 4 or 8 pivots;
- separately timed pivot-index construction.

Axes:
- corpus sizes 256, 1,024, 4,096 and 16,384 (all divisible by 8);
- deterministic random corpus / query control;
- eight-cluster synthetic corpus and query near cluster 5;
- 2/4/8 pivot metadata entries per candidate.
- Two pivot selectors: a cheap evenly-spaced ID control and farthest-first/maximin selection.

The clustered corpus is interleaved by cluster member rank, so equal index strides do not reliably select representatives from different clusters. Farthest-first selection greedily chooses the candidate farthest from its nearest selected pivot. Each candidate stores one u16 Hamming distance per pivot (4, 8 or 16 bytes per candidate). This is much smaller than 2,048 bytes per vector, but index selection/construction, metadata scans and per-query pivot distances still cost work. The benchmark times complete selector-plus-index construction separately from query latency so amortization can be calculated from actual results.

The timed search includes query-signature and lower-bound buffer creation, because allocations are part of its current implementation. It does not yet model persistent scratch-buffer reuse; that is a follow-up only if this first version shows promise.

## Correctness and diagnostics

- Before timing, pivot results are compared with exhaustive exact Hamming top-1.
- The correctness fixture includes a 257-item corpus, despite the main benchmark corpus sizes being divisible by 8.
- Duplicate exact matches appear at non-adjacent candidate IDs; lowest-index tie-breaking must remain stable even if minimum-bound seeding evaluates a later candidate first.
- Untimed diagnostics report the number of candidate vectors receiving full exact evaluation and the logical signature metadata size. These are not physical DRAM-byte counters, and output is not a measured speedup.
- Index lower bounds are safe by triangle inequality. A benchmark candidate is invalid if the exact winner ID or distance differs from exhaustive search.

## Expected regimes and falsification conditions

Pivot bounds are not expected to help uniformly. Random high-dimensional Hamming vectors concentrate near half the dimensions; a small number of pivots can yield weak lower bounds, in which case metadata work is overhead. This negative control is mandatory.

The clustered fixture is deliberately structured but interleaved: eight anchors define groups, candidates within each group are deterministic perturbations of their anchor, and members are interwoven so corpus-order sampling cannot simply pick one pivot per cluster. The query is near a member of cluster 5. The comparison tests whether farthest-first pivot selection adds value over fixed index-stride selection. This synthetic case does not represent an arbitrary production embedding distribution.

Continue only if the exact path:
1. preserves exact IDs/distances on random, clustered and tie fixtures;
2. reduces full-vector evaluations meaningfully in a relevant workload;
3. improves measured latency after metadata reads, query-signature work and allocation are included;
4. amortizes index construction over a declared number of queries;
5. does not regress memory footprint or update cost beyond the target device's budget.

If random realistic traces show no latency benefit, keep the exhaustive SIMD baseline and do not treat a lower candidate-evaluation count as success.

## Industry comparison

FAISS already offers exact `IndexBinaryFlat` and binary index families including IVF, HNSW, hash and multi-index hash. Its binary guide describes exhaustive popcount-optimized search and the bucket/radius trade-offs of hash-based search:
https://github.com/facebookresearch/faiss/wiki/Binary-Indexes

Therefore, pivot bounds are a candidate for a specific exact high-dimensional clustered regime, not a replacement claim against FAISS. Compare against FAISS `IndexBinaryFlat` using identical candidate/query bytes and exact semantics first. Then compare indexed FAISS alternatives as separate speed/recall curves. Do not compare approximate results to an exact result without reporting recall.

For any phone/edge claim, follow an architecture-neutral methodology similar to MLPerf Tiny: identical workloads and quality targets, measured latency, and calibrated energy:
https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/

## Evidence boundary

No Rust compilation, Criterion timing, physical cache/DRAM traffic, energy, RTL, FPGA or SOTA evidence is claimed by this authored change. Expected speedup is deliberately unspecified until measurements exist.

## Independent mathematical reference pilot (not Rust execution)

An independent Python integer-bit model used 16,384-bit vectors, eight deterministic anchors, the same interleaved-cluster construction, the same two selectors, the same bound formula and exhaustive Hamming top-1 as oracle. Every exact winner ID and distance matched the oracle.

At 16,384 candidates:

| Corpus profile | Selector | Pivots | Full-vector evaluations | Signature payload |
|---|---|---:|---:|---:|
| Random control | Evenly spaced | 2 | 16,382 / 16,384 (99.99%) | 64 KiB |
| Random control | Farthest-first | 2 | 16,383 / 16,384 (99.99%) | 64 KiB |
| Random control | Evenly spaced | 4 | 16,381 / 16,384 (99.98%) | 128 KiB |
| Random control | Farthest-first | 4 | 16,380 / 16,384 (99.98%) | 128 KiB |
| Random control | Evenly spaced | 8 | 16,376 / 16,384 (99.95%) | 256 KiB |
| Random control | Farthest-first | 8 | 16,376 / 16,384 (99.95%) | 256 KiB |
| Interleaved clusters | Evenly spaced | 2 | 2,363 / 16,384 (14.42%) | 64 KiB |
| Interleaved clusters | Farthest-first | 2 | 2,173 / 16,384 (13.26%) | 64 KiB |
| Interleaved clusters | Evenly spaced | 4 | 1,556 / 16,384 (9.50%) | 128 KiB |
| Interleaved clusters | Farthest-first | 4 | 727 / 16,384 (4.44%) | 128 KiB |
| Interleaved clusters | Evenly spaced | 8 | 1,389 / 16,384 (8.48%) | 256 KiB |
| Interleaved clusters | Farthest-first | 8 | 24 / 16,384 (0.15%) | 256 KiB |

The diagnostic counts how many candidate vectors require complete exact Hamming evaluation. It does not measure latency, physical memory traffic or energy. The model supports two limited conclusions: the bound is almost useless on this random 16K-bit control, and pivot selection quality materially affects the deliberately clustered fixture. That matches known limitations of metric-pivot indexing under high-dimensional distance concentration; it is not a novelty or SOTA claim. Python's PRNG generated the reference vectors, so this does not test Rust `BinaryHV::random`, Rust source correctness or a compiled benchmark. The Rust target must reproduce exact results and show measured latency improvement after selector, index-build, metadata and allocation costs before any win can be claimed.
