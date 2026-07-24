# Numerical scope and guarantees

This crate uses `f64` throughout. It is intended for simulation, calibration, scientific diagnostics, and ordinary statistical inference—not arbitrary-precision proof work.

## Error and domain contracts

Validated `try_*` APIs reject empty or insufficient samples, non-finite observations, invalid probabilities, non-positive scales/degrees of freedom, negative counts or weights, impossible success/trial counts, length mismatches, degenerate designs, invalid lags, incompatible merge configurations, oversized exact computations, and iterative non-convergence.

Compatibility wrappers preserve the original ergonomic surface:

- `Option<T>` wrappers return `None` on invalid or degenerate inputs.
- Scalar distribution wrappers return `NaN` on invalid parameters.
- Mathematically meaningful endpoint values remain explicit (`±∞` quantiles, zero/infinite boundary densities, and degenerate PMFs).

## Algorithms

- Mean: max-magnitude scaling with Neumaier compensated summation.
- Variance/covariance: Welford and parallel-merge recurrences.
- Weighted moments: West-style online updates and parallel weighted merges.
- Quantiles: type-7 linear interpolation after total ordering of validated finite values.
- Histograms: fixed-width bounded-memory counts; quantiles interpolate uniformly inside occupied bins.
- Log-gamma: Lanczos approximation for positive arguments.
- `erf`/`erfc`: direct-tail approximation with roughly `1e-7` relative tail accuracy.
- Incomplete beta: modified Lentz continued fraction with explicit iteration budget.
- Incomplete gamma: power series or modified Lentz fraction; a Temme transition approximation is used for large shape near `x ≈ a`.
- Normal quantile: Acklam rational approximation.
- Student-t quantile: bracketed monotone solve over the validated CDF.
- F distribution: incomplete-beta CDF/tail with log-domain transforms for extreme scale ratios; unrepresentable beta-shape sums fail explicitly.
- Fisher exact test: fixed-margin hypergeometric probabilities accumulated with log-sum-exp.
- Bootstrap/permutation: deterministic SplitMix64 sampling; BCa acceleration uses delete-one jackknife statistics; Monte-Carlo permutation p-values use the `(extreme+1)/(iterations+1)` correction.
- ANOVA: centered classical sums of squares with direct F survival probability.
- Mann-Whitney: average ranks, tie-adjusted asymptotic variance, and continuity correction.
- Kolmogorov-Smirnov: exact empirical distance with finite-sample-corrected asymptotic probability.
- PACF: Yule-Walker equations via Levinson-Durbin recurrence.
- Simple regression: centered OLS sufficient statistics and classical homoskedastic standard errors.
- Multiple regression: Householder QR least squares with explicit rank/conditioning refusal and classical homoskedastic covariance.
- Logistic regression: deterministic IRLS with stable logistic/softplus arithmetic, optional L2 regularization, and observed-information covariance.
- Huber regression: IRLS with normal-consistent MAD residual scale and Huber weights.
- Beta/gamma quantiles: monotone bisection over validated CDFs.
- Power analysis: transparent normal approximations with monotone integer sample-size search.
- Sequential Bernoulli testing: Wald likelihood-ratio boundaries.
- Bounded-mean confidence sequence: Hoeffding bounds with an `alpha_t ∝ 1/t²` union-bound schedule.
- Conjugate models: exact beta-binomial, gamma-Poisson (shape-rate), and normal-known-variance updates.
- Predictive discrimination: exact empirical ROC trapezoids and grouped-tie average precision.
- Cross-validation: deterministic Fisher-Yates partitions; binary stratification distributes each class round-robin after independent shuffles.
- Poisson regression: log-link IRLS with observed-information covariance and bounded exponent evaluation.
- Robust OLS covariance: HC0-HC3 sandwich estimators using exact leverage from `(XᵀX)⁻¹`.
- Ridge regression: augmented-system QR with an unpenalized intercept, deterministic optional standardization, effective degrees of freedom, and generalized cross-validation.
- OLS diagnostics: exact hat leverage from the QR covariance basis, internally studentized residuals, Cook's distance, PRESS, VIF auxiliary regressions, and Breusch-Pagan score testing.
- Dependent OLS covariance: one-way cluster score aggregation with CR0/CR1 correction and Bartlett-kernel Newey-West HAC covariance.
- Isotonic calibration: stable tie aggregation and weighted pool-adjacent-violators fitting.
- Conformal prediction: finite-sample split-conformal order statistics for symmetric/normalized regression intervals and marginal binary label sets.
- Doubly robust ATE: augmented inverse-probability influence scores from caller-supplied propensity and potential-outcome predictions.
- Difference-in-differences: Welch inference on panel changes or the treatment-by-post OLS interaction for repeated cross sections.
- Randomization inference: exact fixed-arm-size assignment enumeration below a declared state limit, otherwise deterministic add-one-corrected Monte Carlo sampling.
- Kernel density: Gaussian kernels with fixed, Scott, or Silverman bandwidths; empirical-CDF bands use the Dvoretzky-Kiefer-Wolfowitz inequality.
- Residual diagnostics: Ljung-Box portmanteau, Jarque-Bera moment, and asymptotic Wald-Wolfowitz runs tests.
- Symmetric eigen/PCA: maximum-off-diagonal Jacobi rotations over the unbiased sample covariance.
- Kaplan-Meier: product-limit survival, Greenwood variance, and log-log confidence intervals.
- Log-rank: classical observed-minus-expected score with finite-population variance.
- Cox regression: Newton-Raphson maximization of the Breslow tied partial likelihood and Breslow baseline cumulative hazard.
- Causal estimators: Welch difference-in-means, normalized inverse-probability arm means, and deterministic greedy nearest-neighbor matching.
- Meta-analysis: inverse-variance fixed effects and DerSimonian-Laird moment tau-squared.
- Equivalence: two one-sided Student-t tests with the corresponding `(1-2α)` confidence interval.
- Distribution fitting: normal/exponential MLE; beta/gamma method of moments with fitted log likelihood.
- Jackknife: delete-one pseudo-values, first-order bias correction, and normal interval.
- Autoregression: conditional least squares through the checked multiple-OLS layer.
- Process control: exact-startup EWMA limits and standardized tabular CUSUM.

## Known limits

- Extremely small probabilities may underflow below the representable `f64` range. Log-tail APIs should be preferred where provided.
- Normal-tail and normal-derived p-value accuracy is bounded by the dependency-free `erfc` approximation rather than the platform C library.
- F, t, chi-square, and exact-table calculations validate arithmetic domains, not whether a study's modeling assumptions are scientifically justified.
- Fisher exact enumeration is intentionally capped at one million feasible states.
- ANOVA and OLS uncertainty assume independent, homoskedastic residuals and correctly specified mean models.
- Multiple regression uses Householder QR with a diagonal-ratio gate; SVD remains preferable for rank-revealing analysis of extremely ill-conditioned or underdetermined designs.
- Logistic Wald intervals can be unreliable with small samples, separation, rare events, or material ridge regularization.
- Huber regression protects against outcome contamination but does not by itself solve leverage-point, dependence, or heteroskedasticity problems.
- Power results are large-sample approximations, not non-central-t or simulation-exact guarantees.
- Confidence sequences require every observation to respect the declared finite bounds.
- SPRT operating characteristics assume the two Bernoulli hypotheses and error rates were fixed before observing data.
- Mann-Whitney and KS p-values are asymptotic; ties and very small samples can make exact or permutation inference preferable.
- Histogram quantiles are approximations and require zero underflow/overflow because out-of-range order is not retained.
- Reliability-weight unbiased variance is not interchangeable with frequency-weight or survey-design variance.
- PACF can fail when the finite-sample Yule-Walker system is singular.
- BCa intervals depend on smooth delete-one behavior; discontinuous, boundary, or undefined jackknife statistics can make BCa unsuitable.
- Cross-validation fold standard errors describe dispersion of fold scores and do not make overlapping training sets independent.
- AUC, average precision, and selected thresholds are optimistic when optimized and evaluated on the same data.
- Poisson regression assumes conditional equidispersion; use diagnostics or a different model when overdispersion is material.
- HC covariance changes coefficient uncertainty, not OLS sensitivity to nonlinear mean misspecification or influential outcomes.
- Covariance PCA is scale-dependent and does not standardize variables automatically.
- Kaplan-Meier/log-rank require non-informative censoring; Cox inference additionally requires proportional hazards.
- Inverse-probability and matching estimators require exchangeability, positivity, consistency, and a scientifically adequate propensity model.
- Greedy matching is deterministic but not globally optimal and does not account for propensity-model estimation uncertainty.
- DerSimonian-Laird random-effects inference can be anti-conservative with few studies or extreme heterogeneity.
- TOST margins must be scientifically justified before examining outcomes.
- Gamma/beta moment fits are not maximum-likelihood estimates.
- Conditional AR forecasts omit parameter and innovation uncertainty bands.
- EWMA/CUSUM operating characteristics assume the declared target and process standard deviation remain meaningful.
- ECE/MCE depend on the selected bin count and are descriptive, not uniquely defined population parameters.
- Ridge coefficients, effective degrees of freedom, and GCV describe the declared penalty and preprocessing; they are not ordinary unpenalized OLS inference.
- Cluster-robust covariance assumes independent clusters and requires enough clusters for the t reference; Newey-West results depend on observation order and the declared lag.
- Breusch-Pagan, VIF, Cook's distance, Jarque-Bera, Ljung-Box, and runs tests are diagnostics, not automatic proof of model validity or invalidity.
- Isotonic calibration and conformal coverage require held-out calibration predictions; reuse of training observations invalidates the intended evidence split.
- Split-conformal guarantees are marginal under exchangeability and do not provide subgroup-conditional coverage.
- AIPW requires valid nuisance predictions, positivity, consistency, and exchangeability; ordinary influence-function standard errors presume an appropriate cross-fitting protocol.
- Difference-in-differences requires parallel trends and stable composition beyond the algebra checked by the crate.
- KDE results are bandwidth-sensitive and boundary-biased without domain-specific correction.
- DKW bands cover the empirical CDF uniformly but can be conservative and do not smooth the distribution.

## Truth harness

The test suite includes committed reference vectors generated independently with SciPy/mpmath. Runtime and test execution remain dependency-free. Reference regimes include ordinary values, distribution boundaries, deep tails, large-shape incomplete-gamma transitions, F ratios, beta/gamma quantiles, proportions, Fisher exact tables, ANOVA, tied ranks, deterministic resampling, multivariate merges, linear/logistic/Poisson/robust/Cox models, PCA spectra, survival curves, causal overlap, meta-analysis, predictive discrimination, power designs, sequential boundaries, and adversarial configuration errors.

## v0.7 additions

- Negative-binomial functions and regression use the NB2 convention `Var(Y|X)=μ+αμ²`; regression dispersion is caller-declared and coefficient covariance conditions on it.
- Elastic-net optimization uses deterministic cyclic coordinate descent on centered/optionally standardized predictors. It reports penalized point estimates, not post-selection standard errors.
- Theil-Sen intervals are central empirical pairwise-slope quantiles and are not claimed to have calibrated confidence coverage.
- Missing-data imputation is deterministic single imputation and omits missingness-model uncertainty.
- Horvitz-Thompson/Hajek variance formulas assume independent Poisson inclusion; stratified means use within-stratum simple-random-sampling variance with finite-population correction.
- Nelson-Aalen uses event/risk increments; cumulative incidence uses the Aalen-Johansen recurrence but v0.7 does not expose its variance estimator.
- Empirical-Bayes intervals condition on moment-estimated hyperparameters.
- Exponential-smoothing outputs are point forecasts without innovation-distribution prediction intervals.

## v0.8 additions

- Dirichlet log density is evaluated only on the open simplex; boundary density limits are deliberately not collapsed into a single value. Dirichlet covariance uses normalized concentrations to avoid avoidable large-concentration overflow.
- Multinomial and Dirichlet-multinomial count calculations reject totals above `2^53` before integer-to-`f64` conversion.
- Multiclass macro metrics omit classes whose relevant denominator is undefined. Cohen kappa and ordinal weighted kappa are descriptive agreement corrections and do not establish exchangeability or rater validity.
- Bland-Altman limits use a normal multiplier and describe paired differences, not confidence limits for the population limits of agreement. Cronbach alpha assumes a common latent construct; ICC interpretation depends on the declared one-way/two-way and agreement/consistency design.
- Rayleigh p-values use the standard finite-sample approximation and target unimodal departure from circular uniformity.
- Block-bootstrap uncertainty depends on a scientifically defensible block scheme and length; deterministic seeding provides reproducibility, not automatic dependence-model adequacy.
- Local-level filtering and smoothing condition on caller-declared Gaussian variances. Grid selection maximizes the same conditional Gaussian likelihood and does not include parameter-selection uncertainty in forecasts.
- Generalized Pareto moment fitting requires finite variance (`shape < 1/2`). Hill estimates require a positive heavy tail and are highly sensitive to the selected number of upper order statistics. Return levels extrapolate under threshold stability and stationarity assumptions.
- Hotelling T² uses classical pooled/sample covariance and exact F transformations under independent multivariate normal sampling. It rejects singular/high-dimensional covariance rather than regularizing silently.

## v0.9 additions

- Finite Markov stationary distributions use power iteration and therefore require a convergent chain under the requested tolerance; reducible or periodic chains can legitimately fail to converge to a unique result.
- Hidden Markov inference uses scaled forward-backward recursions for categorical emissions. Zero-probability observation sequences fail closed rather than returning a fabricated posterior.
- Multinomial logistic regression uses the final class as the reference, a convex negative-log-likelihood, optional L2 regularization, Cholesky-checked Newton systems, and backtracking. Wald intervals condition on the fitted model and declared penalty; separation and weakly identified classes can still make them scientifically fragile.
- Zero-inflated Poisson EM separates a structural-zero mixture from a Poisson count process. Hurdle Poisson models all zeroes separately and condition positive counts on truncation. Both v0.9 fits are intercept-only and do not establish the substantive cause of excess zeroes.
- Wilcoxon and Friedman p-values use asymptotic normal/chi-square references with tie corrections; the paired sign test is exact after dropping zero differences.
- Delta-method intervals are first-order local approximations. Monte Carlo propagation assumes the supplied parameter approximation is multivariate normal and requires a positive-definite covariance matrix.
- Rubin pooling assumes imputations are proper and complete-data variance estimates are comparable. Coordinate intervals use the large-sample Rubin degrees-of-freedom expression and do not replace congeniality diagnostics.
- Bayesian bootstrap draws Dirichlet(1,…,1) observation weights. Its posterior interpretation conditions on the empirical support and does not extrapolate beyond observed values.
- Gaussian-mixture EM uses deterministic quantile initialization and can converge to a local optimum. The variance floor prevents numerical collapse but is an explicit modeling regularizer, not evidence for a true component variance.

## v1.0 additions

- Hypergeometric calculations use log-gamma combinatorics and direct lower/upper summation. Counts above `2^53` are rejected before integer-to-`f64` conversion, and very wide exact sums are bounded explicitly.
- Beta-binomial probabilities use the standard beta-function predictive form. The model represents exchangeable Bernoulli trials conditional on a beta-distributed probability; it does not identify the physical source of overdispersion.
- Two-sided exact binomial tests use probability ordering. Clopper-Pearson intervals are conservative; Jeffreys intervals are Bayesian equal-tailed intervals and should not be described as frequentist exact coverage statements.
- CLR, ALR, Aitchison distance, compositional centers, and variation matrices require positive parts. Zero replacement is domain-specific and is never performed implicitly.
- Deming regression conditions on a caller-supplied measurement-error variance ratio. Its coefficients describe a linear method-comparison model and do not correct nonlinear bias, heteroskedastic error, or correlated measurement errors.
- Rank-normalized split R-hat and folded R-hat diagnose scalar-chain mixing, not model correctness. Effective sample size assumes chains target the same stationary distribution and uses a Geyer initial-positive/initial-monotone autocorrelation truncation.
- Multivariate-normal density and conditioning require a symmetric positive-definite covariance. Singular Gaussian laws are deliberately rejected rather than assigned generalized densities.
- Randomized Halton integration operates over `[0,1]^d` for at most 64 dimensions. Between-shift variation estimates randomized-QMC uncertainty; deterministic low-discrepancy error is not claimed to be IID Monte Carlo error.
- Cramer-von Mises, Anderson-Darling, and KS statistics target a fully specified continuous CDF. Parameter estimation from the tested sample changes the null distribution; Monte Carlo calibration is valid only when the supplied sampler reproduces the complete fitted-null protocol.
- Qn is computed exactly from pairwise distances with the standard asymptotic normal-consistency factor. This implementation is `O(n^2)` and enforces a bounded pair count.
