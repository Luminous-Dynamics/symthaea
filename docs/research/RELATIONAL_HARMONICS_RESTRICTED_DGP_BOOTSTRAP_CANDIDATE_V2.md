# RH-006 Restricted-DGP Nested Rolling Bootstrap — candidate v2

## Status

Candidate only. Not approved for formal inference.

## Main correction from v1

The candidate no longer treats beta_relational = 0 as the final inferential null.

Two distinct objects are retained:

- structural null: beta_relational = 0;
- forecast-accuracy null: E[D_o] = 0.

The second is the target for an equal-predictive-accuracy test.

The v1 bootstrap architecture remains useful because it identifies how to reproduce the finite-sample nested estimator distribution. The v2 candidate changes the null imposed inside that bootstrap world.

## Candidate null construction

For a fixed feature path, rolling geometry, ridge, and error covariance, the expected forecast-risk difference can be expressed as a function of a predeclared incremental-signal path:

Delta(kappa) = E[MSE_context(kappa) - MSE_relational(kappa)]

Under the fixed-design linear estimator, Delta(kappa) is quadratic in a scalar signal amplitude kappa.

The candidate therefore seeks a predeclared root:

Delta(kappa_0) = 0

and generates the bootstrap world at kappa_0 rather than automatically at kappa = 0.

This is analogous in spirit to the finite-sample equal-accuracy/local-parameter logic used in nested forecast work, but the exact root is estimator- and geometry-specific and must therefore be derived and validated for RH-006 rather than imported from a paper.

## Why this is better

At kappa = 0, the larger model can still have a different finite-sample risk because it estimates additional parameters.

Therefore a bootstrap calibrated at kappa = 0 can answer the wrong question:

"How does this estimator behave under zero incremental coefficients?"

while the scientific forecast test needs to answer:

"How does this estimator behave at the boundary where the two forecasting methods have equal expected predictive accuracy?"

Those are not identical questions in finite samples.

## Required nuisance inputs

The risk-null root depends on nuisance quantities:

- training/test feature geometry;
- ridge lambda;
- restricted-model coefficients;
- covariance of training errors;
- covariance of test errors;
- train/test cross-covariance when errors are serially dependent;
- the predeclared relational direction v.

For dependent or heteroskedastic errors, the risk calculation must use the appropriate covariance matrices rather than the i.i.d. simplification.

## Feature conditioning

The first empirical implementation may condition on the observed feature path only if the scientific design establishes that this conditioning is legitimate.

Absence of lagged outcomes from the current predictor vector is not, by itself, a proof of strict exogeneity.

If features are endogenous with the outcome process, the null DGP must generate the feature and outcome paths jointly.

## Full bootstrap refit

Every bootstrap replicate must:

1. generate a contiguous null series;
2. preserve the exact RH-006 schedule;
3. refit training-only standardization;
4. refit fixed ridge at every origin;
5. recompute both nested forecasts;
6. recompute target-level losses;
7. recompute origin-level D_o;
8. compute the same studentized statistic.

No retained observed fitted coefficients may be reused as bootstrap estimates.

## Null hierarchy

The research suite should retain all three diagnostics:

1. coefficient-null diagnostic: beta_relational = 0;
2. finite-sample risk-null diagnostic: Delta(kappa_0) = 0;
3. alternative/power diagnostics: Delta(kappa) > 0 and < 0 around the null.

The structural-null run is valuable because it diagnoses estimator-induced nesting effects. It is not sufficient for the equal-accuracy test.

## Acceptance gate

Before formal execution:

- derive Delta(kappa) under the exact estimator;
- show the root-selection rule is predeclared;
- validate the covariance estimator;
- validate conditional-feature assumptions or implement joint feature/outcome generation;
- reproduce the oracle risk-null in independent code;
- test size under dependence, heteroskedasticity, ridge variation, overlap variation, and near-singularity;
- compare oracle and estimated-nuisance versions;
- keep method selection independent of the observed confirmatory effect.

## Current decision

applicability = not-approved-for-execution
selection = stop-assumption-failure
formal p-value = disabled
formal confidence interval = disabled

References:
Clark and McCracken (2015), Journal of Econometrics 186(1), 160–177, DOI 10.1016/j.jeconom.2014.06.016.
Doko Tchatoka and Haque (2023), Journal of Forecasting 42(7), 1844–1864, DOI 10.1002/for.2987.