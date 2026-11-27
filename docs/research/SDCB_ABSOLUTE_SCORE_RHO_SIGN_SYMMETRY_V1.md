# SDCB absolute-score dependence sign symmetry v1

**Status:** exact-source Rust metamorphic control authored; Rust 1.96 formatting and test run pending. Stacked on the Gaussian/Student-t(5) common-scale stress experiment. No production behavior changes.

## Result

For the stationary AR(1) process

`X_t = rho X_(t-1) + sqrt(1-rho^2) epsilon_t`

define `Y_t = (-1)^t X_t`. Then

`Y_t = -rho Y_(t-1) + sqrt(1-rho^2) eta_t`

where `eta_t = (-1)^t epsilon_t`. For symmetric Gaussian innovations, `eta_t` has the same joint innovation law. A path-level positive common scale (including the standardized Student-t(5) mixture) commutes with the sign transform. Therefore the finite-dimensional law of `(|X_t|)` is identical under `rho` and `-rho`.

Consequently, the distribution of the block-maximum absolute-residual score

`S = max_t |X_t|`

cannot distinguish positive from negative AR(1) correlation of the same magnitude. Adding negative rho values to the existing absolute-score AR(1) grid would duplicate the same score-distribution regimes, not broaden this particular test.

## Executable metamorphic control

The new test applies the exact alternating-sign transformation to the innovations, for both Gaussian and standardized Student-t(5) common-scale paths and `|rho| ∈ {0.05, 0.50, 0.95}`. It checks equality of every absolute residual and of the block maximum, not merely Monte Carlo agreement.

This is an algebraic metamorphic control for the implemented AR(1) generator and score. It does not imply that negative dependence is irrelevant to:
- signed residual scores or one-sided maxima;
- cross-channel covariance and lagged cross-correlation;
- non-symmetric innovations, skew, asymmetric contamination or nonlinear dependence;
- nonstationary or time-varying regimes;
- arbitrary real-world sensor processes.

## Research implication

Benchmark expansion should add genuinely new dependence structures rather than simply adding rho signs that the current absolute score cannot distinguish. The next useful stress dimensions are:
1. lag-dependent/non-AR dependence with equal marginal and equal lag-1 rho;
2. cross-channel common causes and covariance rotations;
3. time-varying dependence and abrupt/gradual transitions;
4. asymmetric/skewed innovations where the sign symmetry assumptions fail;
5. signed or directional score functions, evaluated as a separate score family.

The metamorphic test prevents wasting trials on a redundant negative-rho grid while making the invariance assumption explicit and falsifiable.
