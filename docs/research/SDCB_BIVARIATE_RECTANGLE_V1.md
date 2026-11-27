# SDCB bivariate Gaussian rectangle reference v1

**Status:** bounded, two-channel numerical reference authored in Rust. Twenty-five tests are authored but have not been compiled or executed here. Fixture values are checked using the analytic bivariate orthant identity and an independent conditional-normal integration. This is not a general multivariate solver or a SOTA claim.

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

The implementation uses deterministic adaptive Simpson integration for the outer conditional integral. Inner standard-normal interval masses reuse the direct-tail 64-point Gauss–Legendre routine from #13944. The scalar module now initializes that quadrature rule once through a thread-safe OnceLock and exposes a standard-normal interval API; the bivariate loop calls that cached API rather than rebuilding the 64-point rule for every evaluation. It splits the integration range at conditional-transition locations (including several conditional-standard-deviation offsets) so a narrow high-correlation transition is less likely to hide between initial samples.

This is a low-dimensional reference method. The reported outer refinement difference is an **error indicator**, not a formally certified total error bound. The one-million-call cap is a hard limit on outer integrand evaluations; it does not count the fixed 64-point inner Gaussian quadrature nodes and must not be presented as a total floating-point operation count. Inner quadrature and total results must still be checked against independent high-precision calculations.

## 3. Covariance and tail policy

- Requires a finite, symmetric, positive-semidefinite 2×2 covariance matrix.
- Full-rank SPD inputs use conditional integration and are rejected when the correlation condition estimate exceeds \(10^8\).
- Rank-one PSD inputs are integrated on their exact one-dimensional support by intersecting the two affine channel constraints. This includes exact duplicate channels, perfect negative correlation, and one deterministic channel.
- Rank-zero covariance is handled as a deterministic rectangle-membership test.
- A zero-variance coordinate with a nonzero cross-covariance is rejected as non-PSD. Covariance symmetry is checked exactly; no near-symmetric averaging, eigenvalue flooring, diagonal jitter, or silent covariance repair is used. Correlation magnitude above one is rejected as non-PSD.
- The result reports covariance support rank. Correlation itself is absent when a zero-variance coordinate makes it undefined; the correlation condition estimate is absent for all rank-deficient inputs because it is not a finite SPD condition measure.
- Standardized normal coordinates are truncated to ([-12,12]); the full-rank result carries an explicit union-bound allowance of (7.1\times10^{-33}). The rank-zero case is deterministic and has no truncation allowance.
- A rectangle with positive support only beyond the numerical truncation window is rejected as ProbabilityBelowResolution.
- Zero-width rectangles return zero; invalid or reversed bounds are rejected.

This is a bounded bivariate support-aware method, not a general singular MVN algorithm. Higher-dimensional rank-revealing transformations remain future work.

## 4. Joint categorical adapter

The joint-bin function accepts ordered thresholds per channel and emits the complete row-major table for all finite bins. It enforces:
- a bounded table size (at most 64 cells);
- total probability within \(10^{-9}\) of one;
- both row/column marginals agree with the #13944 scalar likelihood adapter within \(10^{-9}\).

The bound adapter also preserves a caller-supplied conditional context beside the table:
- episode and prediction identity/version;
- covariance/dynamics model identity/version;
- the full conditioning-state identity/version (the caller must identify the scientific, common-cause and device-state assignment used to produce the conditional Gaussian);
- common-cause identity/version;
- channel and device identities with each device evidence version;
- **separate acquisition-window ID and start/end timestamps for each channel**, including overlapping or unequal windows;
- per-channel calibration context and epoch;
- measurement action identity.

It also returns the conditional mean/covariance, per-channel threshold arrays and numerical-method settings (inner Gauss–Legendre order, truncation cutoff, tolerances and outer evaluation cap). This makes the output inspectable and replay-keyed rather than an unlabelled probability table. For each model/common/device conditioning assignment, invoke the adapter with that exact state-conditioned Gaussian; do not marginalize hidden states or model indices prematurely.

**Trust boundary:** the context is caller-supplied identity propagation. This reference validates required fields and internal consistency but does not authenticate signatures, verify digest preimages, or establish source custody. It is not a verifier or evidence ledger. It does not authorize actions, update #13805, or claim physical common-cause identity.

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
| Exact duplicate channels, common interval (0,1] | 0.3413447460685429 |
| Duplicate-channel joint bins at threshold 0 | [[0.5, 0], [0, 0.5]] |
| Perfect anti-correlation joint bins at threshold 0 | [[0, 0.5], [0.5, 0]] |
| Perfect anti-correlation, support intersection (0,1) | 0.3413447460685429 |
| One deterministic channel; other in (-0.5,0.5] | 0.3829249225480262 |
| Rank-zero deterministic point inside rectangle | 1.0 |

These are mathematical numerical fixtures, not measured sensor probabilities. Orthant fixtures have analytic cross-checks; asymmetric and rare-tail fixtures were independently integrated with SciPy adaptive quadrature.

## 6. Tests and limits

The 25 authored tests cover independence, positive/negative correlation, analytic orthants, finite asymmetric bounds, a rare joint tail, joint-bin normalization/marginalization, duplicate-channel tables with zero off-diagonal support, perfect-anti-correlation tables with zero diagonal support, exact duplicate support, perfect anti-correlation, one deterministic coordinate, rank-zero deterministic support, non-PSD rejection, even-small-asymmetry rejection, near-singular rejection, zero-width rectangles, invalid covariance/bounds, bound-context preservation/rejection (including per-channel window and calibration versions), and bounded evaluation, including a no-overrun check for the outer integrand-evaluation budget.

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
