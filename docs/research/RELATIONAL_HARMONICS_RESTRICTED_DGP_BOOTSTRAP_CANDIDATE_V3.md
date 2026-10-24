# RH-006 Restricted-DGP Forecast-Risk Bootstrap — candidate v3

## Status

Candidate only. Not approved for formal inference.

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

This pass moves the finite-sample risk calculation from the conditional-i.i.d. scalar quadratic to the exact fixed-design quadratic induced by a declared serially correlated and/or heteroskedastic outcome-error covariance.

The implementation is an independent Python/numpy estimator-mechanics reimplementation. It is not the Symthaea Rust implementation.

## 1. Exact conditional risk identity

For origin `o`, let `W_m` map the training outcomes into the held-out forecast for model `m`, with `m ∈ {context, relational}`.

Let:

- `mu_tr`, `mu_te` be the deterministic conditional means;
- `Sigma_tt = Cov(e_tr, e_tr)`;
- `Sigma_yy = Cov(e_te, e_te)`;
- `Sigma_ty = Cov(e_tr, e_te)`.

The expected held-out squared-error risk is:

`R_m = H^-1 [ ||W_m mu_tr - mu_te||^2
             + tr(W_m Sigma_tt W_m')
             + tr(Sigma_yy)
             - 2 tr(W_m Sigma_ty) ]`

where `H = test_samples`.

For the paired context-versus-relational loss difference, the `tr(Sigma_yy)` term is identical and cancels. It is nevertheless retained in the full risk identity because omitting it silently would obscure why the cancellation is valid.

The remaining stochastic contribution requires all of:

- train/train covariance;
- train/test cross-covariance;
- test/test covariance as part of the full risk bookkeeping, although it cancels from the paired MSE differential under a common outcome process.

## 2. Vector finite-sample risk null

Let the relational coefficient perturbation be a vector `gamma` over the declared incremental feature columns and write:

`mu(gamma) = mu0 + Z gamma`.

Because the rolling estimator is linear in the training outcomes once the feature path, training standardization, ridge value, and forecast operator are fixed, each origin's risk difference is quadratic in `gamma`.

After averaging the retained origins:

`Delta(gamma) = gamma' Q gamma + b' gamma + c`

For a predeclared direction `v` and scalar amplitude `kappa`:

`gamma = kappa v`

gives:

`Delta(kappa; v) = (v'Qv) kappa^2 + (b'v) kappa + c`.

The equal-predictive-accuracy null along that slice is any positive root of:

`Delta(kappa; v) = 0`.

## 3. New methodological finding: the null is a surface

The scalar root is not direction-free.

The general accuracy null is a quadratic surface in the full relational coefficient vector. Therefore:

`equal-amplitude root`

means:

> the equal-risk boundary along one predeclared scientific direction.

It does **not** mean:

> the unique finite-sample equal-risk point for the whole relational model.

This distinction matters for bootstrap validity. A general test of equal predictive accuracy has a nuisance-point problem under the null: different points on the accuracy-null surface can induce different finite-sample distributions for the statistic.

The research lane therefore needs one of two explicit choices before formal inference:

1. narrow the estimand to a predeclared relational direction; or
2. specify and validate a null-surface nuisance policy, such as a constrained/profiler bootstrap or a least-favorable construction that controls size over the admissible null surface.

No such general policy is approved yet.

## 4. Dependent / heteroskedastic covariance construction

The oracle uses:

`e_t = rho e_{t-1} + sigma_t u_t`

with unit-variance Rademacher innovations.

For the heteroskedastic scenario:

`sigma_t = 0.08 * [1 + 0.7(common_t - 0.5)]`.

The covariance is constructed analytically from the lower-triangular innovation-loading matrix, so there is no numerical HAC approximation inside the risk equation itself.

This is useful as a reference oracle:

- dependence enters the finite-sample risk exactly;
- train/test cross-covariance is explicit;
- heteroskedasticity changes the covariance weighting rather than being reduced to an IID variance label.

## 5. Executed independent oracle pilot

Canonical command:

`python scripts/research/rh006_oracle_risk_null_covariance_splitmix64.py --paths 50 --directions 128 --calibration-outcomes-per-path 4 --mc-verify 500 --bootstrap 199 --seed 20261009`

Geometry:

- train = 48
- gap = 4
- test = 16
- origins = 24
- step = 16
- ridge = `1e-8`
- Bartlett lag = 3
- nominal alpha = 0.05
- 50 deterministic feature paths per scenario
- 128 deterministic directions on the 3-dimensional unit sphere
- 4 independent risk-null outcomes per feature path
- 199 bootstrap replicates

### Root movement under dependence

| Error process | Positive roots | Mean root | Root range |
|---|---:|---:|---:|
| IID | 50/50 | 0.107196 | 0.101685–0.113663 |
| AR(0.5) | 50/50 | 0.131894 | 0.122682–0.139677 |
| AR(0.8) | 50/50 | 0.187311 | 0.159180–0.203860 |
| AR(0.5) + heteroskedastic | 50/50 | 0.133992 | 0.127385–0.147477 |

The important result is not the particular numbers but the shift: serial dependence materially moves the finite-sample equal-risk boundary for the same estimator geometry.

### Direction sensitivity

The deterministic 128-direction sweep produced approximate positive-root envelopes of:

| Error process | Minimum root observed | Maximum root observed |
|---|---:|---:|
| IID | 0.095899 | 0.331758 |
| AR(0.5) | 0.113537 | 0.402432 |
| AR(0.8) | 0.155616 | 0.600129 |
| AR(0.5) + heteroskedastic | 0.116122 | 0.401273 |

Thus the equal-amplitude slice is materially narrower than the full null geometry. A direction-independent forecast-accuracy null cannot be claimed merely because one fixed direction has a stable root.

### Oracle risk-null bootstrap calibration

The oracle bootstrap was run at the predeclared equal-amplitude risk-null root, with full rolling refits and regenerated dependent outcome series.

| Error process | Bootstrap outcomes | Rejection rate at 5% | Mean p-value |
|---|---:|---:|---:|
| IID | 200 | 0.055 | 0.524 |
| AR(0.5) | 200 | 0.060 | 0.504 |
| AR(0.8) | 200 | 0.050 | 0.514 |
| AR(0.5) + heteroskedastic | 200 | 0.070 | 0.475 |

These are pilot calibration results only. With 200 outcomes per scenario, the Monte Carlo uncertainty around a 5% size estimate is still roughly 1.5 percentage points, and the nuisance quantities are oracle-known.

The pilot therefore supports the **mechanical plausibility** of the risk-null construction under this synthetic DGP, not formal validity for RH-006.

### Independent risk-at-root check

A direct Monte Carlo risk check generated dependent errors from the same oracle covariance and evaluated the realized mean loss differential at the analytic root. Across the 50 feature paths, the mean Monte Carlo residual from zero was small relative to the path-to-path simulation noise.

This is the right kind of check because it asks whether the algebraic root actually corresponds to an empirical equal-risk point under the declared DGP.

## 6. Origin dependence remains a separate layer

The per-origin risk equation only needs covariance blocks for that origin because expectation is additive across origins.

The inferential distribution of the aggregate statistic is different.

`D_o` and `D_{o+1}` can be dependent because:

- outcome errors are serially correlated;
- rolling training windows overlap;
- each origin refits a model using partially shared observations.

Therefore the covariance structure used for the **risk mean** and the dependence structure used for the **sampling distribution of the loss-differential statistic** must not be conflated.

The contiguous bootstrap series plus exact rolling refit is useful precisely because it preserves this joint origin dependence instead of replacing it with independent origin resampling.

## 7. Endogeneity remains a hard gate

Everything above conditions on the feature path.

That is defensible only when the scientific design supports the required conditioning/exogeneity assumptions.

Not having lagged outcomes in the predictor vector is not sufficient evidence of strict exogeneity.

If relational features respond to the same latent shocks as the target, the oracle must instead generate feature and outcome paths jointly. The null then becomes a joint DGP problem, not merely a covariance-of-errors problem.

## 8. Literature mapping

Clark & McCracken (2015) is directly relevant because it distinguishes finite-sample equal forecast accuracy from coefficient-zero population restrictions and constructs bootstrap inference around small nonzero coefficients.

Doko Tchatoka & Haque (2023) is relevant to the dependence problem: their nested-forecast work develops a hybrid moving-block/residual bootstrap and reports important finite-sample distortions under serial correlation and longer horizons.

Zhu & Timmermann show why a generic Giacomini–White conditional-predictive-ability bridge should not be assumed valid merely because rolling estimation is used.

Harvey, Leybourne & Zu emphasize that equal-average forecast accuracy can be complicated by temporal instability in the mean loss differential; this supports retaining an explicit instability gate rather than treating the origin-level mean as automatically stationary.

## 9. Stronger candidate architecture

The current v3 candidate is:

`restricted nuisance DGP`

→ `exact dependent/heteroskedastic finite-sample risk equation`

→ `full quadratic null surface`

→ `predeclared null-surface nuisance policy`

→ `dependent bootstrap DGP`

→ `full rolling refit`

→ `same T`

→ `empirical size / power / instability qualification`.

The `predeclared null-surface nuisance policy` is now the critical missing scientific bridge.

## 10. Next attack

The most valuable next experiment is not another larger bootstrap count.

It is a **null-surface coverage experiment**:

- parameterize the admissible equal-risk surface for the three relational coefficients;
- select multiple non-equivalent points on that surface without inspecting the bootstrap outcome;
- compare bootstrap size across the surface;
- include serial correlation, heteroskedasticity, near-singularity, and ridge variation;
- determine whether one point is least favorable or whether size varies materially over the null;
- only then choose between direction-restricted inference and a uniform null-surface procedure.

A result showing material null-surface size variation would be a genuine reason to keep formal inference closed.

## 11. Evidence boundary

No part of this candidate establishes:

- formal p-values;
- formal confidence intervals;
- estimator/procedure asymptotic validity;
- validity under endogenous features;
- uniform validity over the full equal-accuracy null surface;
- empirical evidence from real RH-006 targets.

Current decision remains:

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

## References

Clark, T. E., & McCracken, M. W. (2015). *Nested forecast model comparisons: A new approach to testing equal accuracy*. Journal of Econometrics, 186(1), 160–177. DOI: 10.1016/j.jeconom.2014.06.016.

Doko Tchatoka, F., & Haque, Q. (2023). *On bootstrapping tests of equal forecast accuracy for nested models*. Journal of Forecasting, 42(7), 1844–1864. DOI: 10.1002/for.2987.

Zhu, Y., & Timmermann, A. (2020). *Can Two Forecasts Have the Same Conditional Expected Accuracy?* arXiv:2006.03238.

Harvey, D. I., Leybourne, S. J., & Zu, Y. (2025). *Testing for Equal Average Forecast Accuracy in Possibly Unstable Environments*. Journal of Business & Economic Statistics, 43(3), 643–656. DOI: 10.1080/07350015.2024.2418835.
