# Changelog

## 1.0.0

- Stabilized a curated `prelude`, `API_LEVEL = 1`, and the `validated-v1` numerical contract.
- Added exact hypergeometric probabilities and finite-population moments.
- Added beta-binomial predictive probabilities, direct tails, and moments.
- Added Clopper-Pearson and Jeffreys intervals plus probability-ordered exact binomial tests.
- Added closure, CLR/ALR transforms, Aitchison distance, compositional centers, and log-ratio variation matrices.
- Added Deming measurement-error regression with explicit error-variance ratios.
- Added rank-normalized split R-hat, folded R-hat, effective sample size, and Monte Carlo standard errors.
- Added multivariate-normal densities, deterministic sampling, and conditional distributions.
- Added randomized Halton quasi-Monte Carlo integration with between-replication uncertainty.
- Added Cramer-von Mises, Anderson-Darling, and Kolmogorov-Smirnov goodness-of-fit summaries with deterministic Monte Carlo calibration.
- Added the high-breakdown Qn robust scale estimator.
- Strengthened complete production-surface auditing and added independent SciPy, NumPy, ArviZ, and statsmodels references.

## 0.9.0

- Added validated finite-state Markov chains with stationary distributions, entropy rates, distribution evolution, and deterministic trajectory sampling.
- Added categorical hidden Markov filtering, forward-backward smoothing, log likelihoods, and Viterbi decoding.
- Added reference-class multinomial logistic regression with ridge stabilization, backtracked Newton updates, coefficient uncertainty, deviance, AIC, and BIC.
- Added zero-inflated and hurdle Poisson probability functions and intercept-only fitted models for excess-zero counts.
- Added exact paired sign tests, tie-corrected Wilcoxon signed-rank inference, and Friedman repeated-measures testing.
- Added first-order delta-method and deterministic multivariate-normal Monte Carlo uncertainty propagation.
- Added scalar and multivariate Rubin pooling for multiple-imputation estimates and covariance matrices.
- Added deterministic Bayesian-bootstrap posterior sampling for arbitrary weighted scalar statistics.
- Added deterministic univariate Gaussian-mixture EM with posterior component responsibilities.
- Added independent NumPy/SciPy/scikit-learn/statsmodels references and adversarial v0.9 contracts.

## 0.8.0

- Added Dirichlet densities/moments, multinomial count-vector probabilities, and transactional Dirichlet-multinomial updating.
- Added weighted multiclass confusion metrics with macro/micro summaries and marginally corrected Cohen kappa.
- Added ordinal weighted kappa, Bland-Altman limits of agreement, and Lin concordance correlation.
- Added raw/standardized Cronbach alpha and ICC(1,1), ICC(2,1), and ICC(3,1).
- Added circular summaries, Rayleigh uniformity testing, and paired circular correlation.
- Added deterministic moving-block, circular-block, and stationary bootstrap procedures.
- Added local-level Kalman filtering, Rauch-Tung-Striebel smoothing, forecasting, and explicit variance-grid selection.
- Added generalized Pareto density/direct-tail/quantile functions, moment fitting, Hill estimation, and return levels.
- Added one- and two-sample Hotelling T² inference through checked covariance solves.
- Added independent NumPy/SciPy/scikit-learn references and adversarial v0.8 contracts.

## 0.7.0

- Added negative-binomial NB2 densities, direct tails, and fixed-dispersion log-linear regression for overdispersed counts.
- Added deterministic lasso and elastic-net coordinate descent with original-scale coefficients and ordered regularization paths.
- Added Theil-Sen median-slope regression and tie-aware Kendall tau-b association.
- Added explicit missingness-pattern audits, complete-case indexing, and deterministic single imputation.
- Added Horvitz-Thompson totals, Hajek means, Kish effective sample size, and finite-population-corrected stratified means.
- Added Nelson-Aalen cumulative hazards and Aalen-Johansen cumulative incidence for competing risks.
- Added exact/asymptotic McNemar tests and Cochran Q repeated-binary inference.
- Added beta-binomial and normal-normal empirical-Bayes shrinkage.
- Added simple and damped Holt linear-trend exponential smoothing with deterministic grid selection.
- Added a validated median API, independent SciPy/statsmodels/scikit-learn references, and adversarial v0.7 contracts.

## 0.6.0

- Added checked Householder QR factorization and least-squares solving, and migrated multiple OLS away from normal equations.
- Added standardized ridge regression with unpenalized intercepts, effective degrees of freedom, and generalized cross-validation.
- Added OLS leverage, internally studentized residuals, Cook's distance, PRESS, predicted R², variance-inflation factors, and Breusch-Pagan diagnostics.
- Added one-way cluster-robust CR0/CR1 covariance and Newey-West HAC covariance.
- Added weighted isotonic probability calibration with stable tie aggregation and pool-adjacent-violators fitting.
- Added split-conformal regression, locally normalized conformal intervals, and marginal binary conformal prediction sets.
- Added augmented inverse-probability ATE estimation from explicit out-of-fold nuisance predictions.
- Added two-period panel and repeated-cross-section difference-in-differences.
- Added deterministic complete/block randomization and exact-or-Monte-Carlo sharp-null randomization inference.
- Added Gaussian kernel-density estimation, Silverman/Scott bandwidths, leave-one-out log scoring, and DKW simultaneous empirical-CDF bands.
- Added Ljung-Box, Jarque-Bera, and Wald-Wolfowitz runs diagnostics.
- Added deterministic bias-corrected and accelerated bootstrap intervals.
- Added independent NumPy, SciPy, statsmodels, and scikit-learn references plus adversarial v0.6 contracts.

## 0.5.0

- Added held-out regression metrics, ROC AUC, average precision, and exact tie-stable threshold curves.
- Added deterministic ordinary and binary-stratified K-fold partitions and fold-score summaries.
- Added checked Poisson log-linear regression through IRLS with deviance, likelihood criteria, intervals, and optional ridge regularization.
- Added HC0-HC3 sandwich covariance inference for OLS.
- Added Jacobi symmetric eigendecomposition and covariance-based principal-component analysis.
- Added Kaplan-Meier product-limit estimation, Greenwood log-log intervals, median survival, and restricted mean survival.
- Added two-sample log-rank inference and Cox proportional-hazards regression with Breslow tie handling.
- Added covariate-balance diagnostics, randomized difference-in-means, inverse-probability ATE estimation, and deterministic caliper matching.
- Added fixed-effect and DerSimonian-Laird random-effects meta-analysis.
- Added one- and two-sample TOST equivalence tests.
- Added normal/exponential maximum-likelihood and gamma/beta method-of-moments fitting.
- Added deterministic delete-one jackknife bias, uncertainty, pseudo-value, and influence diagnostics.
- Added conditional autoregression and recursive forecasting.
- Added streaming EWMA and two-sided CUSUM change detection with transactional invalid-input behavior.
- Added explicit Youden, F1, balanced-accuracy, and cost-sensitive threshold selection.
- Added independent NumPy/SciPy/scikit-learn/statsmodels references and adversarial v0.5 contracts.

## 0.4.0

- Added checked row-major dense matrices and Cholesky factorization for symmetric positive-definite statistical systems.
- Added mergeable multivariate moments, covariance/correlation matrices, and Mahalanobis distance.
- Added multiple OLS with coefficient standard errors, t tests, confidence intervals, mean-response intervals, prediction intervals, R², adjusted R², and likelihood criteria.
- Added binary logistic regression through deterministic IRLS with optional L2 regularization, convergence checks, coefficient intervals, deviance, McFadden R², AIC, and BIC.
- Added deterministic Huber robust regression with robust scale and per-observation weights.
- Added beta and gamma densities, CDFs, direct tails, and monotone quantiles.
- Added beta-binomial, gamma-Poisson, and normal-known-variance conjugate Bayesian models.
- Added Cohen's d, Hedges' g, paired dz/gz, Fisher-z correlation intervals, and zero-cell-safe odds-ratio intervals.
- Added prospective approximate power and minimum sample-size calculations for one-sample means, equal-sized two-sample means, and correlations.
- Added Bernoulli SPRTs and time-uniform confidence sequences for bounded means.
- Added likelihood-ratio tests, AIC/AICc/BIC helpers, and Akaike weights.
- Added stable semantic seed derivation and non-destructive numeric/table data audits.
- Hardened matrix symmetry, rank, overflow, endpoint, and transactional multivariate-update contracts.
- Added cross-layer v0.4 model invariants and independent SciPy/NumPy/statsmodels reference comparisons.

## 0.3.0

- Added deterministic SplitMix64 sampling, unbiased bounded indices, shuffling, and resampling primitives.
- Added seeded non-parametric bootstrap intervals and Monte-Carlo permutation tests with add-one-corrected p-values.
- Added Wilson proportion intervals and one-/two-proportion score tests.
- Added rectangular chi-square independence tests, Cramer's V, 2×2 risks/odds, and Fisher exact inference.
- Added Fisher-Snedecor F density/CDF/direct survival functions and one-way ANOVA with eta-squared and omega-squared.
- Added entropy, cross entropy, KL divergence, Jensen-Shannon divergence/distance, total variation, Hellinger distance, and mutual information.
- Added mergeable fixed-width histograms with explicit underflow/overflow and approximate in-range quantiles.
- Added mergeable weighted moments, covariance, correlation, and Kish effective sample size.
- Added tied average ranks, Spearman correlation, and tie-corrected Mann-Whitney U inference.
- Added autocorrelation, Levinson-Durbin partial autocorrelation, and Durbin-Watson diagnostics.
- Added empirical distributions and one-/two-sample Kolmogorov-Smirnov diagnostics.
- Hardened F calculations for extreme degree-of-freedom ratios through log-domain logistic/softplus transforms and fail-closed incomplete-beta overflow checks.
- Added v0.3 independent reference vectors, deterministic-resampling checks, merge invariants, and adversarial numerical tests.

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
