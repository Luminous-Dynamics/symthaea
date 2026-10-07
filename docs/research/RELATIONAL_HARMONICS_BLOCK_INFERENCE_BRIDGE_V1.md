# RH-006 Block Inference Bridge — v1

## Status

**Research diagnostic only. Not approved for formal inference.**

This note derives a candidate block-level inferential object for the existing RH-006 rolling geometry and records deterministic null-size simulations. It does not authorize a p-value, confidence interval, or change the existing stop-assumption-failure selector.

## Calibration correction

An earlier draft used relational coefficients that were too large to serve as clean equal-risk stress targets under several dependent/error-heteroskedastic DGPs. Before retaining the result as methodological evidence, the targets were rechecked with a separate DGP-only 120-replication calibration audit and replaced with:

~~~text
IID:              0.080
AR(0.5):          0.088
AR(0.8):          0.126
AR(0.5)+hetero:   0.1125
~~~

The updated simulation is the canonical result. The prior draft is superseded by this correction in Git history.

## 1. Exact statistical object

For origin o and target j, define the retained squared-loss differential:

~~~text
d[o,j] = loss_non_relational_context[o,j]
         - loss_relational_augmented[o,j]
~~~

so positive values favor the relationally augmented forecast.

Because RH-006 uses disjoint target windows, define the origin-level block mean:

~~~text
D[o] = (1 / m) * sum_j d[o,j]
~~~

where m is the fixed held-out target-window width.

The rolling estimand considered here is:

~~~text
theta = E[D[o]]
~~~

with one-sided scientific alternative H1: theta > 0 and candidate null H0: theta = 0.

The inferential unit is the origin block, not the individual target observation. Target observations remain ordered inside each block for dependence characterization, but origin means are the object resampled in this first candidate.

## 2. Why the null must be distinguished from the coefficient restriction

For a nested forecasting problem, the statement

~~~text
beta_relational = 0
~~~

is not identical in finite samples to

~~~text
E[D[o]] = 0
~~~

because the larger forecasting method estimates additional coefficients even when their population values are zero. That estimation can change finite-sample forecast risk.

Clark–McCracken explicitly formulate nested forecast testing around finite-sample equal accuracy rather than assuming zero additional coefficients automatically produces equal finite-sample MSFE. Doko Tchatoka–Haque likewise treat nested forecast inference as a special bootstrap problem.

Therefore the simulation suite separates:

1. **Coefficient-null:** incremental relational coefficients are exactly zero. This diagnoses the finite-sample nesting effect; it is not the final equal-risk null.
2. **DGP-only approximate equal-risk targets:** fixed coefficients are selected in simulation-only calibration work so the expected origin-level loss differential is approximately zero for the declared geometry. These values are not estimated from confirmatory data and are not scientific effect estimates.

## 3. Candidate block statistic

Let bar_D be the mean of the ordered origin means. Estimate the long-run variance with fixed Bartlett bandwidth K:

~~~text
gamma_k = (1/O) * sum_{o=k}^{O-1}
          (D[o]-bar_D) * (D[o-k]-bar_D)

Omega_hat = gamma_0
            + 2 * sum_{k=1}^K
              (1 - k/(K+1)) * gamma_k
~~~

The candidate studentized statistic is:

~~~text
T = sqrt(O) * bar_D / sqrt(Omega_hat)
~~~

This is a candidate diagnostic statistic, not an approved RH-006 test statistic.

## 4. Candidate origin-level moving-block bootstrap

The candidate imposes its null by centering the origin means:

~~~text
D0[o] = D[o] - bar_D
~~~

and then samples contiguous origin blocks of length L with replacement until O origin means are generated. The same statistic T* is computed on every bootstrap sequence using the same fixed Bartlett bandwidth.

The one-sided empirical tail is:

~~~text
p_candidate = (1 + count(T* >= T_obs)) / (B + 1)
~~~

This is explicitly non-formal.

### Design-derived block-length anchor

Consecutive rolling training windows overlap across:

~~~text
L_overlap = 1 + floor((train_samples - 1) / step_samples)
~~~

origins.

For the current geometry:

~~~text
train_samples = 48
step_samples  = 16
L_overlap     = 3
~~~

so L=3 is the minimum design-derived block anchor. L=6 is a prespecified sensitivity case. Neither is chosen from the confirmatory effect.

The overlap formula captures one source of cross-origin dependence. It does not prove that the complete forecast-loss process has dependence length three.

## 5. Why this candidate is not yet the final nested bootstrap

The candidate resamples retained origin-loss means. It therefore does not recreate:

- the raw outcome-generating process;
- finite-sample parameter-estimation uncertainty;
- training-window standardization estimation;
- ridge fitting at every bootstrap origin;
- the joint dependence created when overlapping rolling training windows are refit on one resampled series.

Doko Tchatoka–Haque's published procedure instead constructs a bootstrap DGP and refits the forecasting system, using dependence-aware residual/block generation. That distinction is central here.

## 6. Deterministic simulation design

The research harness mirrors the RH-006 estimator:

- fixed-width rolling training;
- fixed gap;
- disjoint held-out target windows;
- one-step horizon;
- training-only feature means/scales;
- standardized predictor space;
- fixed positive ridge penalty;
- unpenalized intercept;
- deterministic pivoted Gaussian elimination;
- fresh nested fits at every origin;
- target-level squared-loss differentials;
- origin-level studentization;
- moving-block bootstrap over origins.

The RNG is a pinned SplitMix64 stream with Rademacher innovations.

Pilot geometry:

~~~text
train_samples      = 48
test_samples       = 16
gap_samples        = 4
origin_count       = 24
step_samples       = 16
forecast_horizon   = 1
ridge_lambda       = 1e-8
Bartlett K         = 3
alpha              = 0.05
bootstrap_reps     = 199
Monte Carlo reps   = 120
~~~

Generated simulation artifact SHA-256:

~~~text
01ac0ec64b8d17292ba2162fb9f9d9f06b840a5494ebace3502b6d609a817043
~~~

Machine-readable record:

~~~text
docs/research/RELATIONAL_HARMONICS_BLOCK_INFERENCE_SIMULATION_V1.json
~~~

## 7. Calibration audit

The recalibrated stress targets were checked in a separate DGP-only run before the size simulation.

| Scenario | beta_relational | Mean D | SE(mean D) | z |
|---|---:|---:|---:|---:|
| IID | 0.080 | 0.0000895 | 0.0001066 | 0.84 |
| AR(0.5) | 0.088 | -0.0000937 | 0.0001081 | -0.87 |
| AR(0.8) | 0.126 | -0.0003102 | 0.0003008 | -1.03 |
| AR(0.5) + heteroskedastic | 0.1125 | -0.0000138 | 0.0001882 | -0.07 |

These are approximate finite-sample equal-risk targets, not exact analytical solutions.

## 8. Size-simulation findings

### Coefficient-null

With all incremental relational coefficients set to zero:

~~~text
mean origin loss differential = -0.00188764
rejection = 0 / 120
~~~

This is not evidence of a correctly calibrated 5% test. It demonstrates the finite-sample nesting effect.

### Approximate equal-risk targets

| Scenario | L | Rejections / 120 | Empirical size | MC SE |
|---|---:|---:|---:|---:|
| IID | 3 | 6 | 0.050 | 0.0199 |
| IID | 6 | 14 | 0.117 | 0.0293 |
| AR(0.5) | 3 | 3 | 0.025 | 0.0143 |
| AR(0.5) | 6 | 6 | 0.050 | 0.0199 |
| AR(0.8) | 3 | 8 | 0.067 | 0.0228 |
| AR(0.8) | 6 | 11 | 0.092 | 0.0263 |
| AR(0.5) + heteroskedastic | 3 | 6 | 0.050 | 0.0199 |
| AR(0.5) + heteroskedastic | 6 | 11 | 0.092 | 0.0263 |

The L=3 results are broadly near the 5% target in this pilot; the L=6 results are systematically more liberal. The AR(0.8) case is modestly above nominal at L=3, but the Monte Carlo experiment is still only 120 replications.

The strongest conclusion is therefore **not** that the candidate is valid. It is that the design-derived overlap block L=3 is the only block length tested here that remains plausibly calibrated across the declared stress cases, while a larger block cannot be assumed safer.

## 9. New methodological conclusion

The corrected evidence keeps the origin-level statistic as a **candidate worth deeper testing**, but does not justify promoting it.

The next candidate should be a **restricted-DGP nested rolling bootstrap**:

1. preserve the exact RH-006 feature and split geometry;
2. define the null DGP explicitly;
3. generate dependence-preserving bootstrap errors/innovations;
4. reconstruct the outcome series under that null;
5. refit both nested methods from scratch at every rolling origin;
6. recompute training-only standardization and fixed ridge inside every replicate;
7. retain the complete origin-level loss-differential vector;
8. compute the same origin-level statistic;
9. compare its empirical size across the complete declared failure-mode matrix.

This directly addresses the finite-sample nesting effect that the retained-loss bootstrap cannot reproduce.

## 10. Approval gate

Before the applicability artifact can leave not-approved-for-execution, the future candidate must demonstrate prospectively:

- nominal null size under IID, weak dependence, strong dependence, and heteroskedasticity;
- stability across declared ridge values;
- correct training-only preprocessing;
- correct horizon alignment;
- exact reproduction of rolling overlap;
- deterministic behavior near singular designs;
- prespecified dependence/block choices;
- byte-stable replay from source + manifest + seed;
- no result-dependent method or bandwidth switching.

A successful pilot is not sufficient for promotion.

## 11. Hard stop

Until the raw-series/refit bridge is derived and passes the simulation gate:

- ForecastInferenceSelectionPath remains stop-assumption-failure;
- the estimator applicability artifact remains not-approved-for-execution;
- no formal p-value or confidence interval is emitted by RH-006;
- dependence profiles remain descriptive;
- surrogate exceedance fractions remain empirical diagnostics;
- rolling MSE differences remain estimands/diagnostics only.

## References

- Clark, T. E. & McCracken, M. W. (2015), *Nested forecast model comparisons: A new approach to testing equal accuracy*, Journal of Econometrics 186(1), 160–177. DOI: 10.1016/j.jeconom.2014.06.016.
- Doko Tchatoka, F. & Haque, Q. (2023), *On bootstrapping tests of equal forecast accuracy for nested models*, Journal of Forecasting 42(7), 1844–1864. DOI: 10.1002/for.2987.
- Giacomini, R. & White, H. (2006), *Tests of Conditional Predictive Ability*, Econometrica 74(6), 1545–1578. DOI: 10.1111/j.1468-0262.2006.00718.x.
- Zhu, Y. & Timmermann, A. (2020), *Can Two Forecasts Have the Same Conditional Expected Accuracy?* https://arxiv.org/abs/2006.03238.

This document records the corrected simulation boundary and advances the raw-series/refit bootstrap as the next methodological target.