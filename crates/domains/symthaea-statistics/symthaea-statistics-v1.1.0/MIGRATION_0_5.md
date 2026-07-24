# Migration guide — v0.4 to v0.5

v0.5 is additive. Existing v0.4 APIs remain available; the release introduces new modules rather than silently changing established formulas.

## Predictive validation

Use `try_regression_metrics` for held-out continuous predictions and `try_binary_discrimination` for ROC AUC, average precision, and tie-stable threshold curves. Partition observations with `try_k_fold` or `try_stratified_k_fold`; both require an explicit seed.

`try_stratified_k_fold` deliberately requires at least one observation of each class per fold. Reduce the fold count when the minority class is smaller than the requested number of folds.

## Generalized and robust models

- Use `try_poisson_regression` for non-negative integer counts with a log link.
- Use `try_robust_linear_inference` when OLS point estimates are appropriate but homoskedastic standard errors are not. HC3 is the conservative default for small or leverage-sensitive samples.
- Use `try_principal_components` for covariance-based PCA. Inputs are centered but not standardized; standardize variables before fitting when scale invariance is intended.

Poisson and Cox ridge penalties affect the inferential target. Report the penalty and treat Wald intervals as penalized-model approximations.

## Survival analysis

- `try_kaplan_meier` accepts right-censored times and event flags.
- `try_log_rank_test` compares two product-limit curves.
- `try_cox_regression` fits covariate hazard ratios with Breslow handling of tied event times.

Censoring is assumed non-informative conditional on the modeled information. Cox coefficients require proportional hazards; the crate does not infer that assumption from the data.

## Causal evidence

`try_difference_in_means`, `try_covariate_balance`, `try_inverse_probability_ate`, and `try_propensity_matching` expose estimands and overlap diagnostics without claiming exchangeability. Propensity values must lie strictly between zero and one. Matching is deterministic, greedy, and caliper-bounded; it is not a global optimal-matching solver.

## Evidence synthesis and practical decisions

- Use `try_meta_analysis` for fixed-effect or DerSimonian-Laird random-effects pooling.
- Use `try_one_sample_equivalence` or `try_two_sample_equivalence` for TOST rather than interpreting a non-significant difference as equivalence.
- Use `try_select_threshold` only with a predeclared objective or cost model. Optimizing and reporting performance on the same observations is optimistic.

## Distribution fitting and temporal diagnostics

Normal and exponential fits are maximum-likelihood estimates. Gamma and beta fits are explicitly named method-of-moments fits. `try_autoregression` uses conditional least squares; `EwmaChart` and `CusumChart` require a declared target and process scale.

## Verification

Run `./scripts/verify.sh` in a Rust 1.85+ environment. v0.5 adds independent NumPy, SciPy, scikit-learn, and statsmodels references under `validation/`.
