# Migrating to symthaea-statistics 0.4

Version 0.4 is additive. Existing v0.3 APIs remain available; the new modules provide model fitting, Bayesian conjugate analysis, power planning, sequential evidence, and data auditing.

## Multiple regression

Use row-major predictor observations. With an intercept, coefficient zero is the intercept.

```rust
use symthaea_statistics::try_multiple_linear_regression;

let x = vec![vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 0.0], vec![3.0, 1.0]];
let y = [1.0, 3.0, 5.0, 7.0];
let fit = try_multiple_linear_regression(&x, &y, true, 0.95)?;
let predicted = fit.try_predict(&[1.5, 0.5])?;
# Ok::<(), symthaea_statistics::StatisticsError>(())
```

The implementation rejects rank-deficient and severely ill-conditioned designs rather than returning unstable coefficients.

## Logistic regression

```rust
use symthaea_statistics::{LogisticRegressionOptions, try_logistic_regression};

let options = LogisticRegressionOptions {
    ridge: 0.1,
    ..Default::default()
};
let fit = try_logistic_regression(&predictors, &outcomes, options)?;
# let _ = fit;
# Ok::<(), symthaea_statistics::StatisticsError>(())
```

A small positive ridge is recommended for sparse binary data or near-separation. The reported covariance uses the penalized observed-information matrix. Ridge fits therefore should not be interpreted as classical unpenalized maximum-likelihood inference.

## Robust regression

`try_huber_regression` is an M-estimator for contaminated outcomes. Its observation weights are diagnostics, not posterior probabilities or deletion flags.

## Conjugate Bayesian models

- `BetaBinomialModel`: Bernoulli/binomial probability
- `GammaPoissonModel`: Poisson event rate under shape-rate parameterization
- `NormalMeanKnownVarianceModel`: unknown normal mean with known observation variance

Updates consume sufficient statistics, so domains can aggregate batches without retaining raw observations.

## Power analysis

The v0.4 power functions use transparent large-sample normal approximations. They are appropriate for planning and screening, but exact non-central-t or simulation-based designs may be preferable near minimum sample sizes, under unequal allocation, or with complex dependence.

## Sequential evidence

Do not repeatedly inspect ordinary fixed-sample p-values. Use:

- `BernoulliSprt` for two predeclared Bernoulli hypotheses
- `BoundedMeanConfidenceSequence` for a mean known to lie inside declared bounds

Both APIs preserve their stopping contract explicitly. `BernoulliSprt` rejects observations after a boundary has been crossed.

## Stable seed hierarchy

`SeedSequence` derives child seeds by semantic label. This prevents adding a new randomized analysis from changing every later PRNG stream.

## Matrix access

`DenseMatrix` exposes checked `try_get` and `try_row` methods plus `as_slice`. Internal unchecked indexing is crate-private.
