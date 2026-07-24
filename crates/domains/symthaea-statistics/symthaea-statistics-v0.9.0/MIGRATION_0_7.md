# Migration guide — v0.6 to v0.7

v0.7 is additive. Existing v0.6 APIs remain available. The release extends the crate into overdispersed counts, sparse modeling, robust pairwise estimation, missing-data workflows, design-based survey analysis, competing risks, paired binary outcomes, empirical-Bayes shrinkage, and exponential smoothing.

## Overdispersed counts

Use the negative-binomial NB2 functions when count variance exceeds its mean:

- `try_negative_binomial_pmf`, `try_negative_binomial_cdf`, and `try_negative_binomial_sf` use `Var(X)=μ+αμ²`.
- `try_negative_binomial_regression` fits a log-linear NB2 mean model with caller-declared dispersion.

Dispersion is intentionally explicit. v0.7 does not silently estimate it from the same data used to fit coefficients. Estimate or predeclare dispersion in the study protocol, then record it with the fitted result.

## Sparse and robust regression

- `try_elastic_net_regression` supports lasso (`l1_ratio=1`), mixed elastic net, and ridge-like coordinate descent (`l1_ratio=0`).
- `try_elastic_net_path` preserves the caller's lambda order for reproducible selection records.
- `try_theil_sen_regression` reports a median slope, median intercept, median absolute residual, and a descriptive central pairwise-slope interval.
- `try_kendall_tau_b` reports tie-aware ordinal association and explicit pair counts.

The Theil-Sen slope interval is not labeled as a calibrated confidence interval. Elastic-net coefficient uncertainty is not reported because ordinary OLS standard errors are not valid after selection and shrinkage.

## Missing data

`try_audit_missing` records per-column counts and joint missingness patterns. `try_complete_cases` preserves original and omitted row indices. `try_impute_numeric` provides deterministic mean, median, or constant single imputation.

Single imputation does not account for missing-data uncertainty and does not establish MCAR, MAR, or MNAR assumptions. Keep the returned imputation counts in downstream evidence.

## Survey estimators

- `try_horvitz_thompson_total` and `try_hajek_mean` use explicit first-order inclusion probabilities and Poisson-sampling variance approximations.
- `try_stratified_mean` applies declared population weights and within-stratum finite-population corrections.

These are design-based estimators, not replacements for undocumented convenience weights. Joint inclusion probabilities are not yet modeled.

## Survival extensions

- `try_nelson_aalen` estimates cumulative all-cause hazard.
- `try_cumulative_incidence` implements the Aalen-Johansen recurrence for one positive integer cause code while treating `0` as censoring.

Competing events reduce event-free survival and remain distinct from censoring. The v0.7 cumulative-incidence result does not yet report a variance estimator.

## Paired categorical evidence

- `try_mcnemar_test` reports continuity-corrected asymptotic and exact two-sided binomial p-values.
- `try_cochran_q` handles matched binary outcomes across three or more treatments.

Both tests require paired or repeated observations. They are invalid for independent groups.

## Empirical Bayes and forecasting

- `try_beta_binomial_empirical_bayes` estimates a moment-matched beta prior and shrinks grouped rates.
- `try_normal_normal_empirical_bayes` performs deterministic normal-normal shrinkage from estimates and standard errors.
- `try_simple_exponential_smoothing`, `try_holt_linear_trend`, and `try_select_holt_linear` support explicit or deterministic-grid smoothing.

Empirical-Bayes intervals condition on estimated hyperparameters and therefore omit hyperparameter uncertainty. Exponential-smoothing forecasts are point forecasts; no innovation-distribution interval is claimed.

## Verification

Run `./scripts/verify.sh` under Rust 1.85+. Independent v0.7 references can be regenerated with `./scripts/verify_v0_7_references.sh` when NumPy, SciPy, statsmodels, and scikit-learn are available.
