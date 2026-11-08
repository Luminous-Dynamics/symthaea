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


## 9. Current source extension: residual verification and joint group bounds

The source has progressed beyond the coordinate-box reference described in the original sections above. The following APIs now exist in crates/core/symthaea-core/src/hdc/continuous_hv_interval.rs:

- verify_l2_residual_certificate(original, reconstruction, epsilon) checks the actual original/reconstruction payload pair with outward f64 accumulation and rejects under-reported bounds. Its returned structure is an admission check, not a signed or persistent proof receipt.
- similarity_interval_full_resolution_for_l2_ball_center(center, radius) retains the joint L2 residual constraint. It bounds the real dot product using Cauchy–Schwarz, the candidate norm using the reverse triangle inequality, and then adds a conservative f32 reduction-error envelope for the canonical scorer.
- similarity_interval_full_resolution_for_component_bounds(lower, upper) propagates exact group-eight/tail f32 score intervals through caller-supplied coordinate bounds. It validates numeric shape, not group membership or synopsis provenance.
- similarity_interval_full_resolution_for_l2_residual(reconstruction, radius) remains the simpler per-coordinate box relaxation for one reconstructed candidate.

The in-memory group builder is at crates/domains/symthaea-memory/src/hme/group_envelope.rs. It scans original members, derives extrema and a deterministic f32 centroid, verifies every member's centroid residual, binds member IDs/payload bits/generations into content digests, then intersects the component-box and joint-ball score intervals. Only a strict group upper bound below the exact kth threshold is eligible for pruning. Persistent segment headers, reopen validation and binding to authoritative eligibility remain with #10917; this in-memory type is not an ExactComplete receipt.

The joint-ball routine uses an operation-count error envelope. Its assumptions and target-specific qualification remain explicit: round-to-nearest binary32 operations, gradual underflow, and no transformations that change the canonical reduction sequence. The code does not establish that every target/compiler/runtime is configured this way. Cross-platform qualification must inspect the actual compiler flags and emitted instructions for the canonical scorer, test supported CPUs (including x86_64 and aarch64), and fail closed where assumptions cannot be established.

## 10. New high-dimensional differential tests authored on 2026-11-08

Two additional Rust tests were added:

- l2_ball_interval_contains_high_dimensional_adversarial_payloads tests dimensions 8, 9, 257 and 16,384 with aligned, anti-aligned, sparse and small-magnitude query profiles. Each generated original payload is first checked against its residual certificate, then its canonical full-resolution score must be enclosed by the joint-ball interval.
- high_dimensional_group_envelope_contains_every_original_score constructs a four-member 16,384D group, builds both synopsis types, intersects their intervals and checks every original member score remains enclosed.

These tests complement, but do not replace, property-based generation of arbitrary residual directions, near-threshold/small-norm adversaries, compiler-target qualification and persisted-segment corruption/reopen tests. They have been authored but NOT executed. No Rust formatting, compilation, test, benchmark or CI pass is claimed.


## 11. Spherical-cap candidate intervals (research-only, 2026-11-08)

The core source now has two deliberately separate types and methods:

- ContinuousHVRealCosineInterval from real_cosine_interval_for_l2_ball_center encloses real-arithmetic cosine using an outward f64 spherical-cap calculation. It does not enclose the canonical f32 score.
- ContinuousHVSphericalCapF32ScoreInterval from similarity_interval_full_resolution_for_l2_ball_center_spherical_cap starts with the real-cosine interval and adds a separate gamma/eta error envelope for the canonical dot and norm reductions, then accounts for the f32 norm-product multiplication, square root and final division.

The second type is intentionally not accepted by ScoreIntervalRecord or select_refinement_set. This avoids accidentally treating a promising experimental bound as a certified retrieval interval. It must remain outside production selection until its derivation and target/compiler behavior are independently reviewed.

### Numeric derivation implemented

For dimension D, the prototype uses unit roundoff u = 2^-24 and an operation count n = 2D for each canonical dot/norm reduction. It bounds relative accumulation error by gamma = n*u/(1-n*u), and absolute underflow contribution by eta. It then constructs interval factors for the two norm accumulations, the f32 norm-product multiplication and f32 square root. The normalized dot interval is expanded by gamma + eta/(||q||*||x||_lower); the final quotient includes f32 division rounding and an absolute minimum-subnormal allowance.

The interval is uninformative when the lower norm/denominator is unsafe, the canonical near-zero denominator branch may be crossed, any error term is unbounded, or the real-cone interval is uninformative. If the entire possible denominator is below the canonical 1e-10 threshold, the result is exactly [0,0]. A zero-radius input returns a point at the canonical scalar score.

Assumptions requiring compiler/target review remain: IEEE-754 round-to-nearest f32 operations, gradual underflow rather than FTZ/DAZ, no fast-math transformations, and the versioned canonical reduction semantics. The f32 type remains separate from the selector even though the current formula attempts to enclose the canonical f32 output.

### New Rust tests authored

- patterned candidate containment at dimensions 8, 9, 17, 257, 2,048 and 16,384;
- randomized residual directions at dimensions 1, 7, 8, 9, 17, 257, 2,048 and 16,384, with each residual independently checked before canonical-score containment;
- zero-radius point behavior and zero-query exact-zero behavior;
- canonical near-zero denominator branch.

These tests have not been executed because no Rust toolchain is available in the current environment.

### Independent simulation (not Rust execution)

A separate Python f32 arithmetic simulation compared the candidate spherical-cap score interval with the current joint-ball canonical interval on 900 generated member cases across dimensions 8, 9, 17, 257, 2,048 and 16,384. It observed zero canonical-score escapes for both interval models. In that sample:

- spherical-cap mean interval width: 0.0585;
- joint-ball mean interval width: 0.0732;
- spherical-cap interval narrower in all 900 cases.

This is exploratory numerical evidence only. It is not proof, Rust test execution, cross-platform qualification, an HME benchmark, or evidence of physical payload I/O savings. No selector integration is authorized by this simulation.
