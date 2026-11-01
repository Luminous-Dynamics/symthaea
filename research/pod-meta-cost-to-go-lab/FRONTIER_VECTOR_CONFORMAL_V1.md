# Proof-Frontier-Conditioned Simultaneous Resource Prediction V1

**Status:** deterministic linear structural baseline plus split-conformal research calibration. No learned-model or SOTA claim.

Issue: [#9686](https://github.com/Luminous-Dynamics/symthaea/issues/9686).

## 1. Proof-frontier representation

The planner exports a frozen `ProofFrontierFeatures` summary before each action:

- unresolved and mandatory obligation counts;
- maximum dependency depth and dependency-edge count;
- admissible alternatives;
- missing evidence requirements;
- distinct operation-family and evidence-family counts;
- mandatory verifier-operation count.

The feature digest is supplied by the caller and must bind the canonical authoritative frontier snapshot. The simulator stores the digest; it does not reconstruct the graph or independently hash the source snapshot.

## 2. Structural predictor baseline

A nonnegative linear model predicts a six-dimensional point cost and strictly positive uncertainty scale:

`point_j = intercept_j + sum_k coefficient[j,k] * feature[k]`

`scale_j = scale_intercept_j + sum_k scale_coefficient[j,k] * feature[k]`

All coefficients and feature counts are nonnegative integers. Arithmetic is checked. Coefficients are frozen before the calibration split and identified by `model_digest`. This is a transparent structural baseline, not an estimator trained by this crate. A future learned predictor can replace the point/scale functions while preserving the receipt and calibration contract.

## 3. One-sided simultaneous conformal score

For completed calibration episode `i`, let `y_i` be the observed remaining resource vector, `mu_i` the frozen predictor's point vector, and `s_i` its strictly positive scale vector. Use one score per episode:

`A_i = max_j max(0, (y[i,j] - mu[i,j]) / s[i,j])`

The maximum is over all six resource dimensions. It yields a single calibration rank for the event that every resource is below its corresponding upper bound; it does not calibrate six marginal intervals separately.

For `n` calibration episodes and target miscoverage `alpha`, select the order statistic with rank:

`k = ceil((n + 1) * (1 - alpha))`

using one-based indexing. If `k > n`, the threshold is unbounded. The implementation never substitutes the largest observed score for the missing order statistic.

The future upper vector is `mu_test + q * s_test`, rounded upward component-wise. Resource arithmetic overflow fails closed.

## 4. Statistical validity boundary

The split-conformal marginal simultaneous-coverage claim requires:

1. the predictor and feature / scale design are frozen independently of the calibration outcomes;
2. calibration examples and the future episode are exchangeable within the declared population cohort;
3. every calibration label is the actual remaining resource vector to independently verified sufficient proof;
4. the model digest and cohort identity match between calibration and prediction;
5. the task's frontier lies within the declared in-domain regime.

Calibration uses completed episodes only and requires a nonzero verifier digest. Right-censored and competing-terminal episodes do not have an observed full cost-to-proof and are excluded. Consequently, this V1 calibration does not claim coverage for arbitrary started tasks, failed tasks, or the unconditional rollout population. That limitation must be stated in every result. Do not drop censored episodes and then silently generalize coverage to all episodes.

Coverage is marginal over the declared exchangeable cohort. It does not imply conditional coverage for every task family, frontier size, operation family, or rare subgroup. Shifted predictions return an unbounded envelope rather than reusing a stale finite threshold.

## 5. Budget gate separation

`assess_joint_resource_fit` checks whether the simultaneous resource upper vector plus mandatory verification reserve fits the remaining capacity in the already-qualified ledger. It reports resource fit only. It does not estimate completion probability, enforce provider-side spend, or grant action authorization. Those remain independent controls.

## 6. Qualification and negative controls

Unit tests cover structural feature sensitivity, feature consistency, multiple mandatory verifier operations per obligation, max-score semantics, the `n+1` conformal rank, insufficient calibration size, simultaneous six-resource containment, shifted-context abstention, model/cohort digest mismatch, duplicate episodes, missing verifier evidence, actual ledger fit, and overflow.

The deterministic matrix emits a finite joint envelope, a shifted-context unbounded case, and an insufficient-sample unbounded case.

## 7. External methodology comparators

- [BAGEN](https://arxiv.org/abs/2606.00198): progressive remaining-budget estimation and infeasibility detection.
- [HARP](https://arxiv.org/abs/2610.07362): hard-budget allocation with reflow and predictive calibration for censored time-to-event evaluations.
- [Multi-Variable Conformal Prediction](https://arxiv.org/abs/2605.12341): vector-valued score sets and simultaneous calibration design.
- [Geometric Conformal Prediction with Spatial Ranks and Multivariate Quantiles](https://proceedings.mlr.press/v306/conrad26a.html): multivariate conformal prediction regions.

This implementation is a max-normalized split-conformal baseline applied to proof-frontier resource costs. It is not a reproduction of these methods and does not claim to outperform them. Future comparisons must report joint coverage, envelope volume / widths, subgroup coverage, and the computational overhead of each method.