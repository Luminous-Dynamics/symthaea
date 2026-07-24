# symthaea-statistics

A dependency-free statistical foundation for Symthaea and the wider Luminous Dynamics workspace.

The crate is designed around twelve rules:

1. **Validated APIs are explicit.** `try_*` functions return `StatsResult<T>` with a structured `StatisticsError` rather than hiding unrelated failures behind `None` or `NaN`.
2. **Compatibility APIs stay lightweight.** Familiar functions such as `mean`, `linear_regression`, and `students_t_cdf` remain available as `Option`/`f64` wrappers.
3. **Tail probabilities are computed as tails.** Student-t, chi-square, and F p-values use direct survival functions instead of subtracting a rounded CDF from one.
4. **Streaming and parallel aggregation are first-class.** Moment, calibration, weighted, histogram, and multivariate accumulators merge without retaining raw observations.
5. **Models report uncertainty, not only predictions.** OLS, multiple regression, logistic regression, and effect-size APIs expose intervals, standard errors, likelihoods, and diagnostics where mathematically justified.
6. **Randomized evidence is reproducible.** Bootstrap and permutation procedures require explicit seeds; semantic seed derivation is stable and independent of call order.
7. **Sequential evidence requires declared stopping rules.** Bernoulli SPRTs and time-uniform bounded-mean confidence sequences prevent ordinary fixed-sample p-values from being reused under optional stopping.
8. **Numerical claims are bounded and tested.** Reference vectors, algebraic invariants, adversarial boundaries, merge-partition tests, and model cross-checks live in the repository.
9. **Predictive evaluation is separated from fitting.** Deterministic folds, held-out metrics, discrimination curves, and threshold objectives make validation policy explicit.
10. **Scientific assumptions remain visible.** Survival, causal, meta-analytic, equivalence, and process-control APIs expose their estimands and diagnostics without claiming that data alone proves the assumptions.
11. **Ill-conditioned models fail before pretending precision.** QR, Cholesky, rank, overlap, and convergence gates refuse unsupported results.
12. **Predictive uncertainty is separated from model fitting.** Calibration and conformal APIs require explicit held-out predictions rather than silently reusing training data.

## Included layers

### Numerical and data foundations

- Scaled compensated means and Welford/parallel moments
- Mergeable multivariate moments, covariance, correlation, and Mahalanobis distance
- Checked dense matrices, Cholesky SPD solves/inversion, Householder QR least squares, Jacobi eigendecomposition, and covariance PCA
- Quantiles, robust summaries, weighted moments, and fixed-memory histograms
- Non-destructive numeric/table audits that preserve NaN and infinity counts

### Probability distributions

- Normal, beta, gamma, Dirichlet, binomial, multinomial, Poisson, zero-inflated/hurdle Poisson, negative-binomial NB2, generalized Pareto, Student-t, chi-square, and Fisher-Snedecor F
- Densities, CDFs, direct survival functions, and selected quantiles
- Convergence-aware incomplete beta/gamma implementations

### Inference and experimental design

- One-sample, Welch, paired t-tests, exact sign tests, Wilcoxon signed-rank, Friedman repeated-measures tests, and one-/two-sample Hotelling T² with intervals or exact F transforms
- Wilson proportion intervals and one-/two-proportion score tests
- Chi-square goodness-of-fit/independence, Fisher exact, and one-way ANOVA
- Mann-Whitney U, Spearman correlation, Kendall tau-b, Theil-Sen regression, and Kolmogorov-Smirnov diagnostics
- Deterministic percentile/basic/BCa bootstrap, moving/circular/stationary block bootstrap, permutation, jackknife, exact randomization inference, ordinary K-fold, and stratified K-fold validation
- Cohen/Hedges effects, Fisher-z correlation intervals, and odds-ratio intervals
- Approximate one-/two-sample and correlation power/sample-size planning
- Bonferroni, Holm, and Benjamini-Hochberg corrections

### Modeling, Bayesian analysis, and decisions

- Simple and QR-based multiple OLS with coefficient and prediction uncertainty
- Binary and multinomial logistic, Poisson, zero-inflated/hurdle Poisson, and fixed-dispersion negative-binomial models with checked optimization, optional ridge regularization, deviance, AIC, and BIC
- Ridge, lasso/elastic-net, and Huber regression; HC0-HC3, one-way cluster-robust, and Newey-West covariance inference
- Kaplan-Meier and Nelson-Aalen estimation, Aalen-Johansen cumulative incidence, log-rank comparison, and Cox proportional-hazards regression
- Covariate balance, randomized difference-in-means, inverse-probability weighting, cross-fit-ready AIPW, difference-in-differences, and deterministic propensity matching
- Fixed/random-effects meta-analysis and one-/two-sample TOST equivalence tests
- Beta-binomial, gamma-Poisson, and normal-known-variance conjugate models plus empirical-Bayes grouped-rate and normal-means shrinkage, deterministic Bayesian bootstrap posteriors, and Rubin multiple-imputation pooling
- Likelihood-ratio tests, AIC/AICc/BIC, and Akaike weights
- Bernoulli sequential probability-ratio tests
- Time-uniform confidence sequences for bounded means
- Binary/multiclass classification, finite Markov chains, discrete hidden Markov filtering/smoothing/Viterbi decoding, Gaussian mixtures, agreement/reliability, circular statistics, weighted isotonic calibration, split-conformal prediction, ROC/precision-recall discrimination, explicit threshold policy, entropy/divergence measures, KDE/DKW bands, ACF/PACF, local-level Kalman filtering/smoothing, Durbin-Watson, Ljung-Box, Jarque-Bera, runs tests, conditional autoregression, EWMA, and CUSUM
- Checked normal/exponential maximum-likelihood and beta/gamma moment fitting
- Missingness-pattern audits, deterministic single imputation, Rubin pooling, delta-method and deterministic Gaussian Monte Carlo propagation, design-based survey totals/means, paired binary tests, and simple/Holt exponential smoothing

## Example

```rust
use symthaea_statistics::{
    CoxRegressionOptions, PoissonRegressionOptions, ThresholdObjective,
    try_binary_discrimination, try_k_fold, try_poisson_regression,
    try_select_threshold,
};

let folds = try_k_fold(100, 5, 0x5eed)?;
assert_eq!(folds.len(), 5);

let scores = [0.9, 0.8, 0.8, 0.1];
let outcomes = [true, false, true, false];
let discrimination = try_binary_discrimination(&scores, &outcomes)?;
assert!((discrimination.roc_auc - 0.875).abs() < 1e-12);

let threshold = try_select_threshold(&scores, &outcomes, ThresholdObjective::YoudenJ)?;
assert!(threshold.sensitivity >= 0.5);

let predictors = (0..6).map(|index| vec![index as f64]).collect::<Vec<_>>();
let counts = [1, 1, 2, 3, 5, 8];
let model = try_poisson_regression(
    &predictors,
    &counts,
    PoissonRegressionOptions::default(),
)?;
assert!(model.try_mean(&[5.0])? > model.try_mean(&[0.0])?);

let _cox_defaults = CoxRegressionOptions::default();
# Ok::<(), symthaea_statistics::StatisticsError>(())
```

## Verification

Run:

```sh
./scripts/verify.sh
```

The verification lane checks formatting, Clippy with warnings denied, every target, doctests, documentation, metadata, and package construction. The repository contains **381 unit/integration tests**.

See [NUMERICAL_SCOPE.md](NUMERICAL_SCOPE.md) for exact claims and limitations, [MIGRATION_0_9.md](MIGRATION_0_9.md) for adoption guidance, and [VALIDATION_V0_9.md](VALIDATION_V0_9.md) for the v0.9 evidence record.
