# HME canonical f32 interval kernel

**Date:** 2026-11-08  
**Status:** source prototype committed; Rust compilation, tests, cross-platform qualification and production certificate validation remain outstanding  
**Owner:** #11078 — HME-INTERVAL-001  
**Score contract:** `continuous_hv_cosine_full_f32_group8_v1`

## 1. Why this kernel exists

The global interval select-and-refine theorem in #11078 is exact only when every stored interval encloses the score produced by the actual canonical scorer. A real-arithmetic cosine bound is insufficient: the canonical score includes a particular sequence of binary32 multiplications, additions, a norm-product multiplication, square root, threshold branch, division and clamp.

A residual certificate and a query/candidate score interval therefore answer different questions:

- the certificate claims the original payload lies within an L2 radius of its reconstruction;
- the interval kernel propagates that uncertainty through the actual versioned f32 score program.

The source prototype is at:

- `crates/core/symthaea-core/src/hdc/continuous_hv_interval.rs`
- public module: `symthaea_core::hdc::continuous_hv_interval`
- method: `ContinuousHV::similarity_interval_full_resolution_for_l2_residual`

## 2. Construction

Let `x_hat` be the compact reconstruction and let an independently validated certificate establish:

`||x - x_hat||_2 <= epsilon`.

It follows coordinate-wise that `|x_j - x_hat_j| <= epsilon`. The reference implementation relaxes the L2 ball to the conservative box:

`x_j in [max(-1, x_hat_j - epsilon), min(1, x_hat_j + epsilon)]`.

The endpoint calculations are expanded outward by one adjacent binary32 value. The interval is then evaluated through the canonical reduction order:

1. eight-coordinate groups, each reduced left-to-right;
2. accumulated group totals in order;
3. remaining tail coordinates in order;
4. query norm squared, candidate norm-squared interval and dot interval;
5. norm-product interval, correctly rounded f32 square-root interval;
6. the canonical `denom < 1e-10` behavior;
7. positive-denominator quotient interval and final cosine clamp.

The query norm-squared is evaluated as a scalar using the same group-eight/tail expression as the canonical scorer. Candidate-side uncertainty remains interval-valued.

For multiplication, all four endpoint products are evaluated and expanded. For squaring, the interval's nearest absolute value and farthest absolute value are used, preserving non-negativity. Addition, multiplication, square root and division all expand the rounded endpoint by one adjacent f32 value.

If a non-finite intermediate, overflow, ambiguous denominator or unhandled condition appears, the method returns the uninformative interval `[-1, 1]`. Such a result cannot prune. When the entire denominator interval is below the canonical threshold, the score interval is exactly `[0, 0]`; when the interval straddles the threshold, the implementation returns `[-1, 1]`.

A zero residual radius returns the exact canonical scalar score as a point interval.

## 3. Numeric contract and preconditions

The enclosure argument assumes:

- IEEE-754 binary32 arithmetic with round-to-nearest, ties-to-even;
- the default Rust floating-point environment with gradual underflow / subnormals preserved;
- no fast-math, reassociation, or floating-point transformations that change the canonical reduction contract;
- finite query and reconstruction components in the closed range [-1, 1];
- an original candidate payload that also satisfies the same range;
- a finite, non-negative residual upper bound that is genuinely conservative.

Rust documents the relevant binary32 primitive rounding semantics, the correctly rounded square-root operation, and adjacent representable values via `next_up` / `next_down`. The source implements adjacent-value stepping directly to avoid imposing a new MSRV.

This source routine does **not** establish the residual inequality. A certificate digest proves byte identity only. Admission must independently construct or verify the residual bound against the original payload, codebook, transform and generation.

## 4. Why the first kernel is deliberately conservative

The coordinate-box relaxation discards the joint constraint `sum_j (x_j - x_hat_j)^2 <= epsilon^2`. It is therefore valid but can be wider than a blockwise or query-aware interval.

This is intentional. It creates a simple reference kernel with an inspectable numeric contract. Future tightening must be tested for interval containment against this reference and against the original-payload canonical score. A narrower empirical interval is not acceptable unless its enclosure guarantee is independently established.

Potential follow-on tightening, still under #10978 and #11078:

- blockwise L2 residual radii, using query block norms;
- joint block bounds that retain the global residual budget instead of relaxing every coordinate to the full radius;
- exact blockwise accumulation intervals that preserve the canonical group-eight reduction;
- a separate SIMD interval implementation that is differentially checked against the scalar reference and its interval envelope.

## 5. Tests authored and evidence status

Rust unit tests have been authored for:

- zero-radius point intervals;
- non-multiple-of-eight dimensions and dimensions spanning several groups;
- deterministic generated candidates whose one-coordinate reconstruction residual is within the declared L2 radius;
- zero query and denominator-threshold overlap;
- invalid radius values;
- dimension mismatch and non-finite profile rejection;
- signed zero, subnormal and infinity adjacent-value edge cases.

A separate NumPy binary32 simulation exercised 1,024 generated query/candidate cases over dimensions 1, 7, 8, 9, 17, 64, 257 and 1,024. It observed zero interval escapes. This is an independent arithmetic sanity check only; it is not execution of this Rust source, a proof, a cross-platform result, or a production residual-certificate test.

The current container does not have `rustc` or `cargo`. No claim is made that formatting, compilation, unit tests, SIMD differential checks, or CI have passed.

## 6. Memory-crate global selector and canonical reranker

A first runtime-layer algorithmic reference now exists in the existing `symthaea-memory` crate:

- `crates/domains/symthaea-memory/src/hme/interval_select.rs`
- module export: `symthaea_memory::hme`
- `select_refinement_set` computes the kth-largest lower-bound threshold and retains all upper bounds at or above it;
- `stable_exact_top_k` validates finite in-range canonical scores, rejects duplicate IDs, sorts score-descending and ID-ascending, and treats signed zero values as numerically equal.

The selector rejects count mismatch, any unbounded unopened partition, malformed interval, duplicate ID and invalid k. Its `IntervalCoverageCounts` are only consistency counters: a caller-supplied count is not proof of manifest or eligibility coverage. The selector returns an `IntervalSelectionPlan`, not an ExactComplete receipt. Original-payload refinement and receipt verification remain outside this pure algorithm.

Authored Rust tests cover the seven-record fixture, strict threshold equality, input-order independence, invalid/duplicate/incomplete controls, signed-zero ties and 4,096 deterministic generated comparisons between full exact top-k and exact reranking of the selected set. They have not been run because the current execution environment does not provide `cargo` or `rustc`.

## 7. Required qualification before ExactComplete


1. Run `cargo fmt --check` and the focused `symthaea-core` test target.
2. Confirm that the canonical scalar scorer and the interval kernel compile under the same default floating-point assumptions on x86_64 and aarch64.
3. Inspect generated LLVM IR / assembly for unexpected contraction or reassociation in the canonical scorer.
4. Differentially test score containment against original payloads at zero, subnormal, near-threshold, high-dimension, tail and near-tie cases.
5. Add property-based tests generating valid reconstructions and certificates, including adversarial residual directions.
6. Validate that corrupted, stale, underreported and wrong-generation certificates are rejected before interval creation.
7. Benchmark interval width, refinement fraction, metadata cost, and end-to-end bytes/latency. The coordinate-box reference may be too loose to beat full scan; that is an empirical question.
8. Keep `ExactComplete` unavailable until the entire eligible corpus has valid intervals (or admissible unopened-segment aggregate bounds) and all the above checks are qualified.

## 8. Competitive evaluation

The kernel is not an ANN index and has no throughput claim. Compare the *complete retrieval architecture* against:

- exact original-payload full scan;
- #10945 progressive exact scan;
- global interval select-and-refine (#11078);
- hierarchical segment-bound selection;
- Recast-style probabilistic refinement, explicitly as ApproximateComplete;
- PQR-Search under its own per-fixed-query guarantee;
- HNSW, DiskANN/Vamana, IVF-PQ/ScaNN, RaBitQ, and QBAT-adapted budgets.

Use the same query/corpus snapshots, metric, k, filters, hardware, concurrency and total memory. Report both per-query guarantee class and system performance. Record per-query recall distribution and OOD cohorts; a corpus-average Recall@k is not adequate.

Primary HME differentiators to measure:

- percentage of eligible original-payload bytes not read while returning exact canonical top-k;
- total index + interval + certificate + codebook bytes;
- false-prune count (must be zero in deterministic mode);
- eligible-record coverage (must be complete);
- p50/p95/p99 cold and warm latency;
- query-weighted and shifted-workload refinement rates;
- insert/delete/update/compaction/recovery overhead;
- rights-filtered and temporal-query payload reads (excluded rows must remain unread).

The initial project target remains at least 50% cold-payload bytes avoided on one named HDC workload after total fast-tier overhead, with 100% canonical top-k agreement and zero false pruning. It is a target, not a result or industry standard.
