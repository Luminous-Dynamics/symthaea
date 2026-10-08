# RH-006 Forecast-Accuracy Null Hierarchy — v1

## Status

Research diagnostic only. Not approved for formal inference.

RH-006 contains two distinct null concepts.

### Structural / coefficient null

H0_structural: beta_relational = 0

This means the incremental relational channels have zero population contribution to the conditional outcome mean.

### Forecast-accuracy null

H0_accuracy: E[D[o]] = 0

where D[o] is the mean target-level squared-loss differential:

d[o,j] = loss_non_relational_context[o,j] - loss_relational_augmented[o,j]

and D[o] = mean_j d[o,j].

This is the null relevant to equal average forecast accuracy.

For nested fitted forecasting methods, these two nulls need not coincide in finite samples. Clark and McCracken explicitly motivate finite-sample equal-accuracy inference through small nonzero coefficients, rather than treating coefficient zero as sufficient for finite-sample equal accuracy.

## Finite-sample risk equation

For a fixed feature path, fixed rolling geometry, fixed ridge, and fixed error covariance, the fitted linear forecast is a linear function of the training outcomes.

Let a predeclared relational direction v have amplitude kappa:

mu(kappa) = mu_non_relational + kappa Z v

Conditional expected forecast MSE is the sum of forecast-bias squared and forecast-variance terms. Because the forecast operator is linear in kappa, the difference between restricted and full expected MSE is quadratic:

Delta(kappa) = A*kappa^2 + B*kappa + C

where C is the finite-sample risk difference at the coefficient-zero point.

Finite-sample equal forecast accuracy occurs at a root of:

Delta(kappa) = 0

This gives an explicit local/least-favorable null construction in the simplified conditional-i.i.d. setting.

## Executed diagnostic

An independently executed SplitMix64/Rademacher reimplementation evaluated 50 deterministic feature paths using:

train = 48
test = 16
gap = 4
origins = 24
step = 16
ridge = 1e-8
i.i.d. error variance = 0.08^2

The relational direction was the equal-amplitude vector over [a_to_b, b_to_a, turn_taking].

Observed:

positive equal-risk root existed: 50 / 50
root mean: 0.1061495
root median: 0.1061063
root minimum: 0.1048493
root maximum: 0.1075285
Delta(0) mean: -0.00056771

This is stable across the 50 synthetic feature paths. It is not a claim about real RH-006 data.

## Consequence

A final predictive-accuracy bootstrap should not silently substitute the structural coefficient null for the forecast-risk null.

The strongest next candidate is:

restricted nuisance DGP
    ->
finite-sample risk difference Delta(kappa)
    ->
predeclared risk-null root
    ->
dependent bootstrap outcome generation
    ->
full RH-006 rolling refits
    ->
same origin-level statistic

The difficult remaining problem is nuisance covariance estimation. Under serial correlation or heteroskedasticity, the variance term must include the declared train/train, test/test, and train/test covariance structure.

## Anti-leakage rule

The relational direction, nuisance estimator, covariance estimator, root-selection policy, block/dependence rule, and bootstrap size must be fixed before inspecting the confirmatory held-out relational advantage.

No result-dependent tuning is permitted.

## Current boundary

applicability = not-approved-for-execution
selection = stop-assumption-failure
formal p-value = disabled
formal confidence interval = disabled

Reference: Clark and McCracken (2015), Journal of Econometrics 186(1), 160–177, DOI 10.1016/j.jeconom.2014.06.016.
Reference: Doko Tchatoka and Haque (2023), Journal of Forecasting 42(7), 1844–1864, DOI 10.1002/for.2987.