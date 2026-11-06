# SPECTRA Ellipsoid Support Geometry v0.1

**Status:** deterministic geometry primitive and Rust test declarations; not compiled or executed.
**Research parent:** #10817
**Method ID:** spectra-positive-definite-ellipsoid-support-geometry-v1

## Scope

This module consumes an already-qualified ellipsoid and computes deterministic linear support intervals. It does not estimate the ellipsoid, establish its finite-sample or time-uniform coverage, validate statistical assumptions, or authenticate its source.

The ellipsoid is represented as:

E = { m + L y : ||y||₂ ≤ r }

where:
- m is the frozen center;
- L is a finite, nonsingular lower-triangular shape factor;
- B = L Lᵀ is the positive-definite shape matrix;
- r is a finite nonnegative radius.

For a linear direction c:

sup_(x∈E) cᵀx = cᵀm + r ||Lᵀc||₂

inf_(x∈E) cᵀx = cᵀm - r ||Lᵀc||₂

This is exact, not an asymptotic approximation. The support interface only depends on the already-qualified ellipsoid geometry.

## Why support functions are the useful adapter

A joint vector confidence region covers all coordinates simultaneously. Every downstream linear functional is deterministic post-processing of that same event. Therefore, once the ellipsoid itself has a valid simultaneous coverage guarantee, evaluating many directions does not require a separate Bonferroni correction per direction.

This is especially useful for:
- conjunctive linear guardrails;
- weighted score/utility contrasts;
- directional safety/performance trade-offs;
- post-hoc queries over a *fixed coordinate vector* already covered by the joint region.

This does not create new statistical coverage for arbitrary newly observed metrics, new action/scenario streams, unregistered outcome coordinates, or external provenance claims. The ellipsoid's own method must control its vector coverage and action/scenario multiplicity.

## Rust implementation

File: tools/spectra-uvsd-fit/src/ellipsoid_geometry.rs

- Frozen coordinate IDs and order.
- Finite center and finite lower-triangular shape factor.
- Strictly positive diagonal, which guarantees positive definiteness of L Lᵀ.
- Finite nonnegative radius.
- Exact directional support and interval evaluation.
- Canonical multi-direction ordering and duplicate direction rejection.
- Deterministic replay that recomputes every direction and compares the full output.
- Stable Euclidean norm accumulation via repeated hypot rather than a naive sum of squares.

No statistical method ID is inferred from the geometry; the method ID only identifies the deterministic support calculation.

## Tests declared

- 1D analytic support.
- Strict shape-matrix Cholesky recovery for a correlated 2D SPD matrix.
- Reject asymmetric, singular and indefinite shape matrices.
- 2D correlated ellipsoid support using full shape matrix rather than diagonal-only variances.
- Multiple direction ordering and replay mutation.
- Positive homogeneity and negative-direction endpoint swapping.
- Zero direction singleton.
- Invalid/non-SPD/malformed geometry.
- Invalid dimension/nonfinite/duplicate direction failures.
- 512 deterministic generated directions compared with an independent direct matrix formula.

Rust tests have not run: cargo, rustc and rustfmt are unavailable in the current environment. No compile, test, format, clippy, CI or statistical-coverage result is claimed.

A separate Python mirror of the geometry formula was exercised on the same two analytic cases and 512 generated directions, including direction scaling and endpoint reversal. This is deterministic algebra smoke, not Rust validation or theorem verification.

## Relationship to the 2026 covariance-adaptive paper

Ramdas & Martínez-Taboada (30 Oct 2026), arXiv:2610.42058:
https://arxiv.org/abs/2610.42058
https://arxiv.org/html/2610.42058v1

The paper's Section 4 gives:
1. A planned-time matrix-heat ellipsoid for direct vector means (Theorem 4.1 / Appendix D.3) with finite-sample coverage, pilot-based whitening, covariance localization, explicit eigenvalue/heat gates, and score-mixture fallback.
2. A time-uniform adaptive linear-bandit confidence sequence (Theorem 4.3 / Appendix D.4.3) for scalar outcomes with predictable bounded design vectors. Its simultaneous event covers matrix frames, directions, bets and time, with an ellipsoid output or R^d for singular design.

The second method is not a drop-in anytime confidence sequence for SPECTRA's directly observed multi-metric score vectors. A separate time-uniform vector-mean construction would require its own proof. The planned-time vector-mean ellipsoid must be used only at its frozen analysis horizon, after allocation and selection are frozen. These are distinct inferential constructions. This module implements neither; it is the exact geometry layer they can reuse after their theorem-specific builders produce a valid center, factor and radius.

For the paper's mean-ellipsoid notation {x : (x-m)ᵀ B⁻¹ (x-m) ≤ r²}, provide L satisfying B=LLᵀ. Its support in direction c is cᵀm ± r sqrt(cᵀBc). For the regression parameter ellipsoid with metric G B⁻¹ G, the corresponding shape matrix is G⁻¹ B G⁻¹ when G is symmetric positive definite.

## Locked integration / benchmark contract

Keep separate receipt identities for:
- planned-time matrix-heat ellipsoid;
- time-uniform vector confidence sequence;
- this deterministic geometry helper.

Compare:
- current scalar Hoeffding + Bonferroni;
- #10009 bounded-vector empirical-Bernstein confidence set;
- tuned normal-mixture vector ball;
- the matrix-heat planned-time ellipsoid;
- the time-uniform covariance-frame directional-bet CS.

The full inference benchmark must report joint coverage, directionally selected support widths, ellipsoid volume, early-look behavior, fallback and singular-design rates, matrix runtime/memory and cost-to-valid decision. Include positive, negative, block and near-singular covariance, plus common-mean and isotropic-proxy negative controls.

No method receives a production advantage by switching between planned-time and time-uniform guarantees without a separately valid method selection or joint alpha allocation.
