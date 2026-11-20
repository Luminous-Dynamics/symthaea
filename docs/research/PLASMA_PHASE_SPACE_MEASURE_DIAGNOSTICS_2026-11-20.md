# Particle-label-independent phase-space diagnostics

**Date:** 2026-11-20  
**Status:** executable reference implementation authored; CI / Rust execution pending  
**Tracking:** [#13020](https://github.com/Luminous-Dynamics/symthaea/issues/13020)

## Research question

Can we compare kinetic simulations on the object they actually evolve—the weighted phase-space empirical measure—without conflating numerical error with arbitrary macro-particle identities?

For a particle representation
\[
\mu_N = \sum_{i=1}^{N_p} w_i\,\delta_{(x_i,v_i)},
\]
per-particle position/velocity norms assume a correspondence between particle index i in the candidate and reference. That correspondence is often artificial after permutation, particle splitting/merging, resampling, or independent Monte Carlo sampling.

The same represented measure may therefore produce a large particle-index error. Conversely, particle-index agreement can obscure distributional defects if the comparison cohort is too narrow. The metric must describe the measure representation and must not become the only fidelity measure.

## Executable diagnostic

For frozen periodic position length L, velocity reference v*, velocity scale vs, and declared modes (k, eta), the crate computes

\[
\widehat{\mu}(k,\eta)=
\sum_i w_i\exp\!\left(i\left[
2\pi k\frac{x_i\bmod L}{L}
+\eta\frac{v_i-v_*}{v_s}
\right]\right).
\]

The comparison is a feature-importance-weighted RMS of complex coefficient differences. It is normalized by the larger total-variation weight of the two measures. The mandatory DC feature exposes net-weight mismatch. The implementation also reports per-mode error, maximum mode error, total-variation mismatch, and particle counts.

The summation uses Neumaier compensation to reduce sensitivity to particle ordering. This cannot eliminate all floating-point differences or provide bitwise representation invariance.

## What makes this a paradigm shift—and what it does not prove

The comparison object changes from a list of nominally matched particles to a finite signature of the weighted empirical measure. This is appropriate when the physical object is a distribution and individual macro-particle labels carry no physical meaning.

The finite feature signature is not injective for arbitrary measures. Distinct measures can have identical values on a finite mode set. It is not a Wasserstein distance or a complete total-variation distance. Its main value is an explicit and reproducible family of distribution-sensitive diagnostics with known representation invariances and known aliasing limits.

The mixed modes (k != 0, eta != 0) matter: position-only and velocity-only marginals do not capture correlations between position and velocity. The negative-control test therefore swaps velocities between fixed positions and requires the mixed-mode signature to change.

## Test surface

The standalone crate authors 13 tests:

1. Particle permutation invariance.
2. Co-located weight split/merge invariance with different particle counts.
3. Periodic representative invariance (x and x+nL).
4. Velocity-position correlation negative control.
5. DC/net-weight negative control.
6. Signed-weight normalization by total variation.
7. Zero-weight measure equivalence.
8. Empty-measure equivalence.
9. Moved-support negative control.
10. Missing DC mode rejection.
11. Invalid scales and feature importance rejection.
12. Non-finite particle rejection.
13. Empty feature-set rejection.

The Rust tests have not been executed in the current authoring runtime. Focused GitHub Actions is configured as the executable validation path; until an exact-head green run is observed, this remains source-authored and not test-passed.

## How to use it in the solver tournament

Keep physically distinct equations in separate cohorts. For a 1D1V periodic electrostatic cohort, report this phase-space signature beside:

- field error against a trusted, resolution-converged reference;
- in-loop rollout/distribution error over independent trajectories;
- Landau damping rate and damping level;
- two-stream growth rate, saturation and long-horizon stability;
- particle-noise spectrum and sampling uncertainty;
- mass, momentum and total-energy error, where applicable;
- Gauss-law and discrete charge-continuity residuals, where applicable;
- positivity / realizability, where applicable;
- time-step and spatial-resolution convergence.

For particle resampling, compare measures before and after resampling. For independently sampled references, report confidence intervals across independent seeds; do not treat many correlated time snapshots as independent experiments.

Benchmark identity must include the mode set and importance weights. Do not retune modes on the held-out evaluation set. Freeze coordinate scales, species conventions, solver builds and independent trajectory splits. Retain full state digests and source/build provenance in receipts.

## Targets and go/no-go gates

- Exact representation fixtures (permutation, periodic representatives, co-located split/merge): relative characteristic error <= 1e-12 in deterministic f64 known-answer tests. This is a proposed implementation acceptance target, not an experimental result.
- Physical distribution tolerances: pre-register per task from reference solver convergence and uncertainty; there is no universal numeric threshold across equations or normalizations.
- Selective solver acceptance: never let this finite signature authorize a learned field proposal by itself. Combine it with independent field residual certificates and task-specific invariant/constraint checks.
- Performance: report end-to-end time-to-solution only at matched physical fidelity and identical hardware/precision/runtime constraints. The earlier >=2x complete-workflow target remains prospective; this metric does not establish it.

## Industry comparison and route to a real SOTA claim

The serious baseline set is not one generic "PIC" implementation. It includes particle/grid/particle-Fourier methods and GPU/CPU implementations within matched equation cohorts. NEOPIC is a relevant learned empirical-measure field-operator comparator; WarpX and Smilei are important production PIC/HPC reference families; structure-preserving methods such as local Gauss-law PIC and discrete-gradient Vlasov-Poisson-Landau schemes belong to their exact equation cohorts.

A defensible superiority claim needs: pinned source/configuration, source-reported results reproduced first, independent initial-condition trajectories, reference convergence, matched precision/platform and complete-workflow timing, uncertainty intervals, and predeclared physical tolerances. No such tournament has yet been run.

## Source references

- NEOPIC (2026): https://arxiv.org/abs/2610.40528
- WarpX documentation: https://warpx.readthedocs.io/
- Smilei documentation: https://smileipic.github.io/Smilei/
- Finn et al. (2026), discrete-gradient Vlasov-Poisson-Landau: https://doi.org/10.1016/j.jcp.2026.114749
- Qiao et al. (2026), intrinsic local Gauss-law PIC: https://doi.org/10.1016/j.cpc.2026.110296
- Taitano, Burby & Alekseenko (2026), conditional Vlasov-Ampere formulation: https://doi.org/10.1017/S0022377826000159
