# SPECTRA Five-Metric Score Covariance Diagnostics v0.1

**Status:** deterministic descriptive Rust diagnostics; tests declared, not compiled or executed.
**Research parent:** #10793
**Method ID:** spectra-five-metric-joint-covariance-diagnostic-v1

## Scope

Every SPECTRA score replication contains five metrics generated on the same run:
1. target information;
2. model-discrimination information;
3. representativeness;
4. target utility;
5. participant burden seconds.

Those metrics may be positively or negatively correlated. Five scalar intervals with a familywise alpha split do not preserve that geometry. Before implementing a joint confidence ellipsoid, SPECTRA needs exact, replayable covariance diagnostics to determine which action/scenario streams and registered decision directions exhibit useful dependence.

This module computes descriptive covariance geometry only. It does not estimate a confidence set, validate a fixed conditional mean, prove a vector sub-Gaussian proxy, allocate alpha or change allocation/stopping.

## Theoretical comparator and claim boundary

Ramdas & Martínez-Taboada (30 October 2026), *Variance-adaptive Chernoff bounds: nonasymptotically valid, asymptotically sharp confidence sets for unbounded means*, arXiv:2610.42058:
https://arxiv.org/abs/2610.42058

The paper's Section 4:
- assumes a vector observation with a common conditional mean and isotropic conditional sub-Gaussian proxy;
- gives a fixed-horizon covariance-adaptive matrix-heat ellipsoid with finite-sample coverage and true-covariance chi-square efficiency;
- separately develops a continuous-monitoring confidence sequence for adaptive linear-regression scores under predictable bounded designs.

Do not describe the fixed-horizon matrix-heat ellipsoid as a generic vector confidence sequence. Do not transfer the adaptive-linear-score CS theorem to arbitrary SPECTRA simulator vectors by analogy.

A useful adapter derivation: if the five frozen metric coordinate ranges have widths w_j, then for any direction u the scalar projection uᵀX has range at most Σ_j |u_j|w_j ≤ ||u||₂||w||₂. Hoeffding's lemma therefore provides the isotropic conditional sub-Gaussian proxy s=||w||₂/2, even with dependence between metric dimensions, provided the coordinate bounds hold conditionally. This is a valid but potentially loose envelope, not a covariance estimate.

## Rust implementation

File: tools/spectra-uvsd-fit/src/score_covariance.rs

For each frozen action × scenario cell:
- validates every five-vector metric observation is finite and inside its declared range;
- requires contiguous replication indices beginning at zero and at least two observations;
- computes the mean vector and full unbiased sample covariance using online symmetric Welford updates;
- computes the correlation matrix;
- computes ascending covariance eigenvalues using a deterministic five-dimensional Jacobi solver;
- reports effective rank and condition number (or None for rank-deficient / numerically singular covariance);
- computes covariance-aware and diagonal-only variance for each pre-registered direction, plus their ratio;
- canonicalizes action/scenario/replication order and supports exact receipt replay.

The receipt retains full means, covariance and correlation matrices, covariance spectrum, effective rank, conditioning and directional projections.

## Directional variance diagnostic

For frozen weight vector w and sample covariance S:

- joint estimate: v_joint = wᵀ S w;
- diagonal-only approximation: v_diag = Σ_j w_j² S_jj;
- ratio = v_joint / v_diag when v_diag > 0.

A ratio below one means cross-metric covariance reduces variance for this direction. A ratio above one means it increases variance. Different directions can have opposite effects. Neither ratio proves a confidence interval or coverage.

This output is particularly useful for deciding whether the current marginal alpha-split bounds are wasteful for a registered SPECTRA utility or constraint direction. It is not a basis for selecting the direction after inspecting the covariance and claiming inferential validity.

## Tests declared

- perfectly correlated five-vector: common-mode direction variance is larger than the diagonal-only approximation while the opposite-contrast direction cancels;
- input-order invariance;
- exact replay and replay mutation rejection;
- missing middle replication, insufficient sample count, duplicate, unknown action and nonfinite score negative controls;
- invalid metric ranges, duplicate/zero directional probes;
- orthogonal five-coordinate vector fixture recovers diagonal covariance and variance ratio one.

A standalone Python mirror reproduced the correlated and orthogonal fixtures. It compared the Jacobi eigenvalue routine with NumPy over 2,000 random positive-semidefinite 5×5 matrices; maximum normalized eigenvalue discrepancy was approximately 8×10⁻¹⁵. This is a numerical diagnostic check, not Rust validation or inference coverage.

Rust test declarations remain unexecuted because cargo, rustc and rustfmt are unavailable in this environment. No compile/test/fmt/clippy/CI or performance pass is claimed.

## Follow-on proof-matched ellipsoid work

1. Reproduce the complete matrix-heat pilot/covariance calibration and fallback in Section 4.1 plus Appendix D.3 of arXiv:2610.42058.
2. Prove that the actual five-vector score stream satisfies the conditional vector sub-Gaussian proxy under the real simulator filtration.
3. Begin with a planned fixed look and compare the fixed-horizon joint ellipsoid with an optimally scaled isotropic vector ball and existing scalar familywise bounds.
4. For adaptive allocation, first use pre-registered fixed-horizon checkpoints with explicit alpha spending. Only use a continuous-monitoring ellipsoid after proving the exact vector filtration and importing an applicable sequential result.
5. Propagate the joint ellipsoid through frozen directional objectives and hard constraints. For nonlinear Maximin/CVaR, optimize over the ellipsoid or use conservative marginal fallback; never substitute the ellipsoid center.
6. Keep scenario-level Maximin/CVaR and the final evidence composer separate from this cell-level diagnostic.

## Locked benchmark

Sweep:
- d = 2, 3, 5, 8, 16, 32;
- K = 4, 8, 16, 32, 64 actions;
- S = 2, 4, 8, 16, 32 scenarios;
- correlation, eigenvalue decay, condition number and rank deficiency;
- true variance / isotropic proxy ratios 0.05–1;
- independent, Gaussian correlated, bounded non-Gaussian and valid martingale vector streams;
- fixed-look and adaptive stopping as separate regimes.

Compare marginal familywise Hoeffding, scalar variance-adaptive CS, optimally scaled isotropic normal-mixture ball, theorem-matched matrix-heat ellipsoid, and oracle true-covariance ellipsoid as analysis-only reference.

Report:
- joint familywise coverage and directional coverage;
- ellipsoid volume / log-volume and directional width;
- false-good / false-no-good and correct Pareto-set rate;
- cost-to-valid decision;
- covariance pilot/fallback frequency;
- rank/conditioning failures and numeric stability;
- p50/p95/p99 matrix update, eigensolver and replay latency;
- memory and allocation count.

Aspirational target after theorem-matched reproduction: ≥20% lower median ellipsoid log-volume than the optimally scaled isotropic ball in strongly correlated, well-conditioned regimes, with exact familywise coverage and zero negative-control false certification. This is a hypothesis, not an observed result.
