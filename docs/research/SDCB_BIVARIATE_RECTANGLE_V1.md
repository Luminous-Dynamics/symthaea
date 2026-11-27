# SDCB bivariate Gaussian rectangle reference v1

**Status:** bounded, two-channel numerical reference authored in Rust. Sixteen tests are authored but have not been compiled or executed here. Fixture values are checked using the analytic bivariate orthant identity and an independent conditional-normal integration. This is not a general multivariate solver or a SOTA claim.

## 1. Why this increment

PR #13944 maps a scalar Gaussian predictive distribution to categorical bins. It cannot preserve joint probabilities for correlated channels. Multiplying its marginal likelihood rows would create fictitious independent evidence.

This implements the first M0 part of issue #13945: a bivariate rectangle evaluator and complete two-channel finite-bin adapter. It reuses the scalar tail-integration routine in #13944 and does not create another experiment planner; #13762, #13793 and #13804 retain policy ownership.

## 2. Numerical method

For standardized channels with correlation rho,

\[
Z_1\sim N(0,1), \qquad
Z_2\mid Z_1=z\sim N(\rho z,1-\rho^2),
\]

so

\[
P(\ell_1<Y_1\le u_1,\ell_2<Y_2\le u_2)
=
\int_{a_1}^{b_1}
\phi(z)\left[
\Phi\!\left(\frac{b_2-\rho z}{\sqrt{1-\rho^2}}\right)
-
\Phi\!\left(\frac{a_2-\rho z}{\sqrt{1-\rho^2}}\right)
\right] dz .
\]

The implementation uses deterministic adaptive Simpson integration for the outer conditional integral. Inner standard-normal interval masses reuse the direct-tail 64-point Gauss–Legendre routine from #13944. It splits the integration range at conditional-transition locations (including several conditional-standard-deviation offsets) so a narrow high-correlation transition is less likely to hide between initial samples.

This is a low-dimensional reference method. The reported outer refinement difference is an **error indicator**, not a formally certified total error bound. Inner quadrature and total results must still be checked against independent high-precision calculations.

## 3. Covariance and tail policy

- Requires a finite, symmetric, strictly positive-definite 2×2 covariance matrix.
- No diagonal jitter, eigenvalue flooring, or silent covariance repair.
- Exact rank deficiency / duplicate channels return SingularCovarianceUnsupported.
- Correlation condition estimates above \(10^8\) return IllConditionedCovarianceUnsupported.
- Standardized Cholesky coordinates are truncated to \([-12,12]\). The result carries an explicit union-bound allowance of \(7.1\times10^{-33}\).
- A requested rectangle whose probability is unresolved above that truncation allowance is rejected as ProbabilityBelowResolution.
- Zero-width rectangles return zero; invalid or reversed bounds are rejected.

This is deliberate fail-closed behavior, not a claim that all near-singular Gaussian problems are solved. A rank-revealing support reduction is a separate research increment.

## 4. Joint categorical adapter

The joint-bin function accepts ordered thresholds per channel and emits the complete row-major table for all finite bins. It enforces:
- a bounded table size (at most 64 cells);
- total probability within \(10^{-9}\) of one;
- both row/column marginals agree with the #13944 scalar likelihood adapter within \(10^{-9}\).

The table is a likelihood for simultaneous observations. It must be conditioned on the correct dynamic state/model before being supplied to a finite policy tree. It does not authorize actions, update the evidence ledger, or claim physical common-cause identity.

## 5. Frozen known-answer fixtures

Independent calculations used:
- the bivariate Gaussian orthant identity \(P(Z_1>0,Z_2>0)=1/4+\arcsin(\rho)/(2\pi)\);
- a separate adaptive integration of the conditional-normal integral for asymmetric and rare-tail fixtures.

| Fixture | Expected probability |
|---|---:|
| Independent N(0,I), rectangle (-1,1] × (-0.5,0.8] | 0.32742270205361657 |
| Positive orthant, rho = 0.6 | 0.35241638234956674 |
| Positive orthant, rho = -0.75 | 0.11502672808130796 |
| Shifted/scaled asymmetric rectangle | 0.35895496926483000 |
| Correlated tail (8,9] × (8,9], rho = 0.5 | \(1.7805614713759896\times10^{-21}\) |

These are mathematical numerical fixtures, not measured sensor probabilities. Orthant fixtures have analytic cross-checks; asymmetric and rare-tail fixtures were independently integrated with SciPy adaptive quadrature.

## 6. Tests and limits

The 16 authored tests cover independence, positive/negative correlation, analytic orthants, finite asymmetric bounds, a rare joint tail, joint-bin normalization/marginalization, exact duplicate support, perfect anti-correlation, one deterministic coordinate, rank-zero deterministic support, non-PSD rejection, near-singular rejection, zero-width rectangles, invalid covariance/bounds, and bounded evaluation.

Not implemented:
- trivariate or general n-variate integration;
- randomized QMC / Genz variable-cube transformation;
- general (n)-variate rank-revealing singular PSD support integration;
- certified total numerical error bounds;
- SDCB dynamic-state/interval-observation conditional adapter;
- physical dataset or policy tournament.

## 7. Competitive protocol

Compare against Genz–Bretz, Miwa and TVPACK where their assumptions apply, plus Peterlin's 2026 doubly blocked Genz implementation. Match input matrices, rectangle bounds, dimension, tail probability, stopping tolerance, replicate count, hardware, thread count and preprocessing. Compare errors against independent high-precision reference values, not against one solver treated as truth.

Publish error–latency–memory Pareto curves and include:
- absolute and relative error;
- outer error indicator and independent error;
- covariance condition estimate;
- tail probability and tail-resolution status;
- function evaluations, runtime and peak memory;
- table normalization and marginalization error;
- all unsupported / non-converged cases.

Peterlin's reported near/above-100× speedups are for the paper's tested high-dimensional SPD workloads. They are not a comparable baseline for this 2D reference. The next performance step should be a separate Genz/QMC implementation and matched reproduction of the published workloads.

## 8. Evidence boundary

- Source and fixtures are authored; Rust has not been compiled or executed here.
- No CI pass is claimed.
- Independent mathematical fixtures are not sensor observations.
- No external package tournament, real sensor dataset, or physical validation is claimed.
- This PR remains a research draft until its tests and downstream integration have been executed and reviewed.
