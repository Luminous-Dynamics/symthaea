# Migration guide — v0.7 to v0.8

v0.8 is additive. Existing v0.7 APIs remain available. The release adds multicategory probability, weighted classification, measurement agreement and reliability, circular statistics, serial-dependence-aware resampling, scalar state-space estimation, extreme-value analysis, and multivariate mean tests.

## Multicategory probability

Use `try_dirichlet_*` for probability vectors and `try_multinomial_*` for category-count likelihoods.

- Dirichlet density is intentionally defined on the **open** simplex. Boundary coordinates are rejected because concentrations below one can create path-dependent infinite density limits.
- `DirichletMultinomialModel` accumulates category counts transactionally and exposes posterior concentrations, predictive category probabilities, and predictive count-vector mass.
- Integer counts above `2^53` are rejected before conversion to `f64`, preserving the crate's explicit numerical contract.

For single-label classification with more than two classes, use `MulticlassConfusion`. Rows are actual classes and columns are predictions. Macro metrics average only classes for which the corresponding denominator is defined; balanced accuracy is macro recall over represented actual classes.

## Agreement and reliability

- `try_weighted_kappa` supports unweighted, linear, and quadratic ordinal disagreement weights.
- `try_bland_altman` reports the mean paired difference and normal-theory limits of agreement.
- `try_concordance_correlation` reports Lin's concordance coefficient, which combines correlation and location/scale agreement.
- `try_cronbach_alpha` and `try_standardized_cronbach_alpha` operate on complete participant-by-item tables.
- `try_intraclass_correlations` reports ICC(1,1), ICC(2,1), and ICC(3,1) from a complete target-by-rater table.

Choose the ICC design before looking at the result. ICC(2,1) treats raters as a random sample and measures absolute agreement; ICC(3,1) conditions on the observed raters and measures consistency.

## Circular observations

Angles are supplied in radians. `try_circular_summary` avoids the artificial discontinuity at zero, `try_rayleigh_test` tests uniformity against a unimodal concentration alternative, and `try_circular_correlation` measures paired angular association.

`wrap_angle` and `angular_difference` are lightweight helpers; validated inferential APIs reject non-finite angles.

## Dependent-data bootstrap

`try_block_bootstrap` and `try_block_bootstrap_mean` require an explicit seed and one of:

- non-wrapping moving blocks,
- circular fixed-length blocks, or
- stationary geometric blocks with a declared expected length.

Block length is a study-design decision. The crate does not estimate an optimal block length or claim that a chosen block scheme captures all dependence.

## Local-level state-space model

`try_local_level_filter` implements a scalar Gaussian local-level Kalman filter with caller-declared process, observation, and initial variances. `try_local_level_smoother` runs the Rauch-Tung-Striebel backward pass, and `try_select_local_level_model` chooses among explicit variance grids by Gaussian log likelihood.

The selector is a transparent grid search, not a continuous optimizer. Forecast variance includes future process uncertainty and observation noise but conditions on the selected parameters.

## Extreme values

The generalized Pareto APIs use a shape/scale parameterization for non-negative threshold exceedances. Direct log-survival and survival functions preserve small upper-tail probabilities.

- `try_fit_generalized_pareto_moments` is a method-of-moments fit and requires finite positive first and second moments.
- `try_hill_estimator` requires strictly positive data and a caller-declared number of upper order statistics.
- `try_generalized_pareto_return_level` extrapolates beyond a declared threshold and exceedance rate.

Threshold choice, tail stability, independence, and stationarity remain scientific assumptions outside the algebra checked by the crate.

## Multivariate means

`try_one_sample_hotelling` and `try_two_sample_hotelling` use unbiased sample covariance, checked Cholesky solves, and the classical exact F transformation. They require more observations than dimensions and reject singular covariance matrices.

Hotelling inference assumes independent multivariate-normal observations; high-dimensional or strongly non-normal data generally require a different protocol.
