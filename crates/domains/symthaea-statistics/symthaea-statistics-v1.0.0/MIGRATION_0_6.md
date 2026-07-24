# Migration guide — v0.5 to v0.6

v0.6 is primarily additive, with one intentional numerical implementation change: multiple OLS now uses Householder QR rather than the normal equations. Public OLS result fields and prediction methods remain unchanged. Well-conditioned fits should agree within floating-point tolerance; nearly rank-deficient designs may now fail earlier and more reliably.

## Linear models and covariance

- Use `QrDecomposition` or `try_solve_least_squares` for checked tall least-squares systems.
- Use `try_ridge_regression` when coefficient shrinkage is part of the declared estimand. The intercept is not penalized. Predictor standardization is deterministic and coefficients are returned on the original scale.
- Use `try_regression_diagnostics` for leverage, Cook's distance, PRESS residuals, and predicted R².
- Use `try_variance_inflation_factors` and `try_breusch_pagan_test` as diagnostics, not automatic model-selection rules.
- Use `try_cluster_robust_inference` for one-way clustered observations and `try_newey_west_inference` for ordered residual dependence. These change covariance estimates, not OLS point estimates or mean-model assumptions.

## Calibration and predictive uncertainty

`try_isotonic_calibration` fits a monotone probability map. Fit and evaluate it on held-out predictions; calibrating and scoring on the same observations is optimistic.

Split-conformal APIs require predictions from a model fitted without the calibration observations:

- `try_split_conformal_regression` provides constant-width intervals.
- `try_normalized_conformal_regression` uses caller-supplied positive local scales.
- `try_binary_conformal_classification` returns set-valued binary predictions.

Coverage is marginal under exchangeability. It is not conditional coverage for every subgroup or covariate value.

## Causal and randomized designs

- `try_augmented_inverse_probability_ate` combines supplied propensity and potential-outcome predictions. Generate nuisance predictions out-of-fold or on a separate sample.
- `try_panel_difference_in_differences` compares unit-level changes; `try_repeated_cross_section_difference_in_differences` estimates the treatment×post interaction.
- `try_complete_random_assignment` and `try_blocked_random_assignment` produce deterministic seeded assignments for reproducible protocols.
- `try_randomization_test_ate` enumerates assignments when the state space is below the declared limit and otherwise uses add-one-corrected Monte Carlo inference.

These APIs expose algebra under exchangeability, positivity, SUTVA/consistency, parallel trends, or randomized-assignment assumptions. They do not establish those assumptions from outcomes.

## Nonparametric distributions and diagnostics

- `try_gaussian_kernel_density` supports fixed, Scott, and Silverman bandwidths.
- `try_dkw_band` returns a simultaneous distribution-free empirical-CDF band.
- `try_ljung_box_test`, `try_jarque_bera_test`, and `try_runs_test` provide residual diagnostics with explicit sample and degrees-of-freedom checks.
- `try_bca_bootstrap` and `try_bca_bootstrap_mean` add deterministic BCa intervals; the statistic must be defined for every delete-one sample and bootstrap replicate.

## Verification

Run `./scripts/verify.sh` under Rust 1.85+. Independent v0.6 references can be regenerated with `./scripts/verify_v0_6_references.sh` when NumPy, SciPy, statsmodels, and scikit-learn are available.
