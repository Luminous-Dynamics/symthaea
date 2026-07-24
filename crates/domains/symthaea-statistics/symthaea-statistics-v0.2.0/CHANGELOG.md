# Changelog

## 0.2.0

- Added structured `StatisticsError` and validated `try_*` APIs.
- Replaced naive means with scaled compensated summation and added mergeable Welford/parallel moment accumulators.
- Eliminated quantile panics on NaN input.
- Fixed binomial and Poisson degenerate endpoint PMFs.
- Added direct normal, Student-t, chi-square, binomial, and Poisson survival functions.
- Added convergence-aware incomplete beta/gamma implementations and a large-shape transition approximation.
- Expanded t-tests with estimates, standard errors, confidence intervals, effect sizes, alternatives, and paired testing.
- Expanded OLS regression with standard errors, slope significance, and response/prediction intervals.
- Split Bayesian updating, classification diagnostics, and probabilistic calibration into focused modules.
- Moved probability-space Bayesian updates into stable log-weight arithmetic to avoid joint underflow.
- Added robust summaries and multiple-testing corrections.
- Added committed differential reference vectors and numerical invariant tests.

## 0.1.0

- Initial descriptive statistics, core distributions, hypothesis tests, simple linear regression, Bayesian updating, and confusion-matrix metrics.
