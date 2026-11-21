# Characteristic-kernel MMD oracle for phase-space measures

**Date:** 2026-11-21  
**Status:** standalone reference implementation authored; Rust compile/test/format/Clippy evidence pending  
**Tracking:** [Issue #13022](https://github.com/Luminous-Dynamics/symthaea/issues/13022)  
**Separate implementation:** [PR #13021](https://github.com/Luminous-Dynamics/symthaea/pull/13021) adds the fast finite-feature diagnostic; this work does not modify that PR.

## 1. Why this increment exists

The finite characteristic-feature diagnostic in #13021 is fast and deterministic, but a finite mode set cannot be injective on arbitrary measures. The reference oracle here uses a mathematically characteristic kernel and evaluates the exact finite empirical quadratic form. It is deliberately a small-N reference: correctness first, scalable approximation second.

The comparison object is the weighted phase-space measure, not particle IDs. A solver is not penalized merely for reordering particles or representing the same weighted measure using co-located split particles.

## 2. Kernel and characteristicness argument

The declared state space is the locally compact abelian group \(G=S^1_L\times\mathbb{R}\), with dual group \(\widehat G=\mathbb{Z}\times\mathbb{R}\). Choose

\[
k((x,v),(x',v'))=k_x(x-x')\,k_v(v-v'),
\]

where

\[
k_x(\Delta x)=\exp\!\left[-2\kappa\sin^2\!\left(\frac{\pi\Delta x}{L}\right)\right]
=e^{-\kappa}\exp\!\left[\kappa\cos\!\left(\frac{2\pi\Delta x}{L}\right)\right],
\quad \kappa>0,
\]

and

\[
k_v(\Delta v)=\exp\!\left[-\frac{\Delta v^2}{2\ell_v^2}\right],
\quad \ell_v>0.
\]

The position factor has the Fourier expansion

\[
k_x(\Delta x)=\sum_{n\in\mathbb{Z}}c_n
\exp\!\left(i\frac{2\pi n\Delta x}{L}\right),
\qquad
c_n=e^{-\kappa}I_{|n|}(\kappa)>0,
\]

where \(I_n\) is the modified Bessel function. Thus every integer spatial harmonic has strictly positive weight. The Gaussian velocity factor has a Fourier transform proportional to

\[
\ell_v\exp\!\left(-\frac{\ell_v^2\omega^2}{2}\right),
\]

which is strictly positive for every \(\omega\in\mathbb{R}\). The product spectral measure therefore has full support on the dual group: every spatial harmonic has positive spectral weight, and the velocity spectral density is positive everywhere on the real axis.

For a finite signed Borel measure \(\sigma=\mu-\nu\), the squared RKHS embedding norm has the spectral form

\[
\|\Phi(\mu)-\Phi(\nu)\|_{\mathcal H}^2
=\int_{\widehat G}|\widehat{\sigma}(\chi)|^2\,d\Lambda(\chi).
\]

If this norm is zero, continuity of the Fourier-Stieltjes transform and full spectral support imply \(\widehat\sigma\) is zero on the entire dual group. Uniqueness of the Fourier-Stieltjes transform on \(G\) then gives \(\sigma=0\), hence \(\mu=\nu\). This establishes injectivity for finite signed measures on the declared domain, not merely for probability measures. The argument depends on \(\kappa>0\) and \(\ell_v>0\); a zero or truncated spatial spectrum would destroy the characteristicness claim.

This is consistent with the characteristic-kernel / Fourier-support framework of Sriperumbudur et al. (2010). Gretton et al. (2012) establish the MMD RKHS distance and its quadratic-time computation, with linear-time approximations in the statistical two-sample setting. Here the direct finite-sum value is used as a deterministic reference metric, not as a p-value or hypothesis test.

Sources:
- Sriperumbudur et al., [Hilbert Space Embeddings and Metrics on Probability Measures](https://jmlr.org/papers/v11/sriperumbudur10a.html), JMLR 11 (2010).
- Gretton et al., [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html), JMLR 13 (2012).

## 3. Numerical contract

For finite empirical measures

\[
\mu=\sum_{i=1}^{N}a_i\delta_{z_i},\qquad
\nu=\sum_{j=1}^{M}b_j\delta_{z_j},
\]

the crate evaluates

\[
\operatorname{MMD}_k^2(\mu,\nu)=
\sum_{i,i'}a_i a_{i'}k(z_i,z_{i'})
+\sum_{j,j'}b_j b_{j'}k(z_j,z_{j'})
-2\sum_{i,j}a_i b_j k(z_i,z_j).
\]

Before evaluating the quadratic form, it reduces periodic positions modulo \(L\), canonicalizes signed zero, sorts the atoms deterministically, and aggregates the coefficients at each exactly co-located position/velocity. The measure total variation is the sum of absolute aggregated atom weights. This is important for signed macro-particles: co-located \(+0.5\) and \(-0.25\) represent one \(+0.25\) atom, with measure TV \(0.25\), not the raw particle-weight sum \(0.75\).

Weights are normalized before the pair sums by

\[
s=\max\left(\operatorname{TV}(\mu),\operatorname{TV}(\nu)\right).
\]

The returned quantity is \(\operatorname{MMD}_k(\mu,\nu)/s\). If both total variations are zero, the normalization scale is set to one and the distance is zero. This makes the reported value invariant to common positive rescaling of both measures and to equivalent co-located split/merge representations.

The unnormalized MMD is the RKHS metric induced by the characteristic kernel. The reported pair-normalized ratio is a scale-normalized discrepancy; because its denominator depends on both inputs, we do not claim that the ratio itself satisfies the triangle inequality.\n\nNeumaier compensated sums reduce accumulation error. A negative computed MMD-squared is clamped only when it lies within \(128\epsilon\max(1,S_{\mathrm{abs}})\), where \(S_{\mathrm{abs}}\) is the sum of the absolute pair contributions. A more negative value returns an error. This is a fail-closed numerical guard, not a formal floating-point proof.

The direct calculation costs \(U^2+V^2+UV\) kernel evaluations, where \(U,V\) are the unique nonzero support atom counts. Canonical aggregation costs \(O(N\log N+M\log M)\), so the overall worst-case complexity remains \(O(N^2+M^2+NM)\), with \(O(N+M)\) temporary memory. The implementation reports raw particle counts, unique atom counts and kernel evaluation count separately.

## 4. Falsification suite

The new crate authors tests for:

1. Particle permutation invariance.
2. Co-located weighted split/merge invariance with different particle counts.
3. Periodic coordinate representatives \(x\) and \(x+nL\).
4. A distinct-support pair that agrees on the DC, first and second spatial Fourier modes but is separated by the characteristic reference kernel.
5. Position/velocity joint-correlation changes when the one-dimensional marginals are held fixed.
6. Positive and signed weighted measures, including cancellation of opposite signed weights at a co-located support point.
7. Common mass rescaling under total-variation normalization.
8. Empty measures, zero-weight particles and one-empty/one-nonempty controls.
9. Invalid scales, non-finite particle data and total-variation overflow.
10. Nonnegative-MMD enforcement and pair-evaluation accounting.

The finite-mode alias control uses two four-point supports offset by \(L/8\). Both have zero first and second spatial harmonics and equal DC coefficient, while their fourth harmonic differs. At the frozen profile \(L=8\), \(\kappa=2\), \(\ell_v=1\), an independent Python formula check gives normalized MMD approximately 0.234356 for this fixture; this is a formula check, not a Rust execution result.

A second independent formula check swaps velocities between two fixed positions. Position and velocity marginals remain equal but the joint phase-space measures differ; the normalized MMD is approximately 0.621500. Again, these numbers are analytical implementation checks in Python, not Rust test results.

## 5. What the MMD number does and does not mean

Characteristicness says that zero population-kernel distance implies equality of the represented measures in exact mathematics. It does not say that a chosen bandwidth has good finite-precision sensitivity to every scientifically relevant perturbation. A very broad kernel can make sharp differences numerically small; a very narrow kernel can become insensitive to broader structure and ill-conditioned under finite precision.

Therefore:
- Freeze the bandwidth or multi-scale mixture in the benchmark profile before held-out evaluation.
- Report per-cohort MMD beside field error, rollout error, spectra, convergence and physical constraints.
- Use confidence intervals across independent initial conditions / random seeds for stochastic particle representations.
- Do not treat correlated snapshots from one trajectory as independent samples.
- Do not infer Wasserstein or optimal-transport equivalence from MMD.
- Calibrate the reference against refined histograms and small-problem optimal transport without replacing one metric by the other.

## 6. Next scalable approximation

The exact oracle is the trusted small-N target. A follow-on \(O(ND)\) approximation can sample spectral features from the known product spectrum: an integer harmonic \(n\) with probability proportional to \(c_n\), and a velocity frequency \(\omega\) from the Gaussian spectral density. For each feature, accumulate the weighted complex phase

\[
\sum_i a_i\exp\!\left(i\left[\frac{2\pi n x_i}{L}+\omega v_i\right]\right).
\]

With \(D\) frozen random features, evaluation is \(O(ND)\). Approximation error must be calibrated against the direct oracle over particle counts, bandwidths, signed-weight regimes, independent feature seeds and adversarial alias fixtures. Record runtime, memory, feature count, error quantiles and worst-case tail—not just mean error. Common random features must be reused for candidate/reference comparisons so the approximation does not add avoidable variance.

A positive mixture of characteristic kernels at pre-registered bandwidths remains characteristic. This is the preferred robustness direction: multi-scale diagnostics can preserve injectivity while reducing sensitivity to a single bandwidth. It must be frozen before evaluation rather than tuned on held-out trajectories.

## 7. Research frontier and competitive positioning

The industry baseline is not a single neural solver:

- **NEOPIC (arXiv:2610.40528, 29 October 2026)** learns a particle-measure field operator, reports \(O(N_p)\) complexity and representation equivalence, and evaluates electrostatic mini-apps including Landau damping, two-stream and bump-on-tail. Its paper-reported claims should be reproduced independently before being treated as measured comparative results.
- **Smilei 5.1** already performs adaptive local scalar/vectorized particle-operator selection by patch and species using particle occupancy; documentation reports architecture-specific SIMD speedups, not universal whole-simulation speedups. It also documents GPU offloading.
- **WarpX 26.11** exposes multiple current-deposition and field-gathering algorithms and GPU-aware particle sorting. We must compare total simulation time at matched field/trajectory fidelity, not neural inference time in isolation.
- **Local Gauss-law-preserving PIC (Computer Physics Communications, October 2026)** updates fields locally for a Vlasov–Ampère formulation without global Poisson solves. It is a relevant local-field baseline and a separate physical-method cohort.
- **Discrete-gradient Vlasov–Poisson–Landau (Journal of Computational Physics, June 2026)** targets collisional dynamics with mass, momentum and energy conservation plus entropy-production structure. Its collisional guarantees must not be collapsed into the collisionless score.

Sources:
- [NEOPIC preprint](https://arxiv.org/abs/2610.40528)
- [Smilei adaptive vectorization](https://smileipic.github.io/Smilei/Understand/vectorization.html)
- [Smilei GPU offloading](https://smileipic.github.io/Smilei/Understand/GPU_offloading.html)
- [WarpX 26.11 parameters](https://warpx.readthedocs.io/en/26.11/usage/parameters.html)
- [Local Gauss-law-preserving PIC, CPC 327 (October 2026)](https://doi.org/10.1016/j.cpc.2026.110296)
- [Discrete-gradient Vlasov–Poisson–Landau, JCP 554 (June 2026)](https://doi.org/10.1016/j.jcp.2026.114749)

## 8. Paradigm shift: risk-qualified hybrid evolution

The next solver architecture should be benchmarked as an evidence-qualified execution policy rather than a universal learned field replacement:

1. A learned global proposal handles smooth, low-frequency field structure.
2. A local/spectral correction operator handles unresolved or near-field residuals.
3. Independent residual and physical-constraint observers estimate risk.
4. A trusted classical solver remains available for unresolved or out-of-distribution states.
5. A structure-preserving integrator advances the coupled particle-field state.
6. The runtime records proposal, correction, residual, fallback, conservation, charge/Gauss and timing receipts.

Adaptation should depend on a declared risk budget: measured residuals, uncertainty and physical constraints—not only particle density or predicted runtime. Unknown verification outcome means no accepted state transition until reconciled.

Do not blend all physics into one score. Keep separate cohorts for collisionless electrostatic Vlasov–Poisson, Vlasov–Ampère/local-Gauss, collisional Vlasov–Poisson–Landau, quasi-neutral/asymptotic-preserving, relativistic and full electromagnetic problems.

## 9. Prospective benchmark gates (not achieved results)

| Dimension | Prospective target / required report |
|---|---|
| Exact representation invariance | \( \le 10^{-12} \) normalized discrepancy on deterministic f64 permutation/split fixtures |
| Kernel oracle | Nonnegative MMD-squared within the declared tolerance; otherwise fail closed |
| Finite-feature approximation | Error quantiles and worst-case error versus feature count, checked against exact MMD |
| Bandwidth robustness | Frozen single- and multi-scale profiles; no held-out tuning |
| Physical fidelity | Task-specific limits established from resolution-converged references |
| Constraint safety | Zero accepted trajectories outside mandatory declared residual/invariant tolerances |
| OOD behavior | Risk-versus-coverage, false-safe rate, fallback rate, and fallback cost |
| Generalization | Independent initial conditions, species parameters, particle counts, seeds and held-out regimes |
| Performance | Prospective \(\ge 2\times\) end-to-end speedup at matched physical fidelity on identical hardware, including preprocessing, correction, communication, diagnostics, fallback and training amortization |
| Reproducibility | Exact input/build/config digests and replayable evidence receipts |

The matched-budget tournament remains [Issue #12976](https://github.com/Luminous-Dynamics/symthaea/issues/12976). Required observables include Landau damping, two-stream linear growth and saturation, bump-on-tail evolution, field and particle-noise spectra, long-horizon drift, convergence, and all applicable invariants/residuals.

A speed claim is admissible only when particle update, field solve, deposition/gather, communication, correction, diagnostics and fallback are included at equal accuracy. A 2x target is a research gate, not a present result.

## 10. Execution state

- New standalone Rust crate and 17 authored unit tests: source created.
- Frozen JSON kernel/profile contract: source created.
- Focused exact-revision Rust 1.96 workflow: source created.
- Python formula checks for the two negative controls: completed.
- Local rustc/cargo unavailable in the authoring runtime; Rust format, compilation, tests and Clippy have not been executed locally.
- The focused workflow must be observed on the exact PR head. Queued, skipped or absent jobs are not passes.
