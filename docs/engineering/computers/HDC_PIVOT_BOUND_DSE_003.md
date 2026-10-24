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

Pivots are sampled as evenly spaced cluster/corpus representatives. Each candidate stores one u16 Hamming distance per pivot (4, 8 or 16 bytes per candidate). This is much smaller than 2,048 bytes per vector, but it adds index-construction work, a metadata scan and per-query pivot distances. The benchmark times index construction separately from query latency so amortization can be calculated from actual results.

The timed search includes query-signature and lower-bound buffer creation, because allocations are part of its current implementation. It does not yet model persistent scratch-buffer reuse; that is a follow-up only if this first version shows promise.

## Correctness and diagnostics

- Before timing, pivot results are compared with exhaustive exact Hamming top-1.
- The correctness fixture includes a 257-item corpus, despite the main benchmark corpus sizes being divisible by 8.
- Duplicate exact matches appear at non-adjacent candidate IDs; lowest-index tie-breaking must remain stable even if minimum-bound seeding evaluates a later candidate first.
- Untimed diagnostics report the number of candidate vectors receiving full exact evaluation and the logical signature metadata size. These are not physical DRAM-byte counters, and output is not a measured speedup.
- Index lower bounds are safe by triangle inequality. A benchmark candidate is invalid if the exact winner ID or distance differs from exhaustive search.

## Expected regimes and falsification conditions

Pivot bounds are not expected to help uniformly. Random high-dimensional Hamming vectors concentrate near half the dimensions; a small number of pivots can yield weak lower bounds, in which case metadata work is overhead. This negative control is mandatory.

The clustered fixture is deliberately structured: eight anchors define groups, candidates within each group are deterministic perturbations of their anchor, and the query is near a member of cluster 5. Using all eight anchor pivots may separate clusters; 2 or 4 pivots test whether fewer signatures suffice. This synthetic case does not represent an arbitrary production embedding distribution.

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
