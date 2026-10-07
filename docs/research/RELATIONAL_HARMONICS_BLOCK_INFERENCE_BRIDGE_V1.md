# RH-006 Block Inference Bridge — v1

## Status

**Research diagnostic only. Not approved for formal inference.**

This note derives a candidate block-level inferential object for the existing RH-006 rolling geometry and records deterministic null-size simulations. It does not authorize a p-value, confidence interval, or change the existing stop-assumption-failure selector.

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
E[D[o]] = 0.
~~~

The larger forecasting method estimates additional coefficients even when their population values are zero. That estimation can change finite-sample forecast risk. Clark–McCracken explicitly formulate their nested forecast framework around finite-sample equal-accuracy rather than assuming that zero additional coefficients automatically yields equal finite-sample MSFE. Doko Tchatoka–Haque likewise treat nested forecast inference as a special bootstrap problem.

Therefore the simulation suite separates:

1. **Coefficient-null:** incremental relational coefficients are exactly zero. This diagnoses the finite-sample nesting effect; it is not the final equal-risk null.
2. **DGP-only equal-risk calibration targets:** small fixed relational coefficients are chosen in simulation-only calibration work so the expected origin-level loss differential is approximately zero for a declared geometry. These targets are not estimated from confirmatory data and are not scientific effect estimates.

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

The first executable candidate imposes its null by centering the origin means:

~~~text
D0[o] = D[o] - bar_D.
~~~

A moving-block bootstrap then samples contiguous origin blocks of length L with replacement until O origin means have been generated. The same statistic T* is computed on every bootstrap sequence, using the same fixed Bartlett bandwidth.

The one-sided empirical tail is:

~~~text
p_candidate = (1 + count(T* >= T_obs)) / (B + 1).
~~~

The +1 correction is only a deterministic finite-sample bootstrap convention. The result remains explicitly non-formal in this artifact.

### Design-derived block-length anchor

Consecutive rolling training windows overlap across:

~~~text
L_overlap = 1 + floor((train_samples - 1) / step_samples)
~~~

origins.

For the current default geometry:

~~~text
train_samples = 48
step_samples  = 16
L_overlap     = 3
~~~

so L=3 is the minimum design-derived block anchor. L=6 is retained as a prespecified sensitivity case. Neither is selected using the confirmatory loss realization.

This overlap formula captures one known source of cross-origin dependence. It does not prove that the complete forecast-loss process has dependence length three.

## 5. Why this candidate is not yet the final nested bootstrap

The candidate resamples the retained origin-loss means. It therefore does not recreate:

- the raw outcome-generating process;
- the finite-sample parameter-estimation distribution;
- training-window standardization estimation;
- ridge fitting at every bootstrap origin;
- the joint dependence created when overlapping rolling training windows are refit on the same resampled series.

That is a decisive limitation for RH-006.

Doko Tchatoka–Haque's published hybrid procedure instead constructs a bootstrap DGP and refits the nested forecasting system, with dependence handled through a moving-block component and residual resampling. The present loss-level bootstrap deliberately does less, so its simulation behavior is a diagnostic of the candidate, not a citation-based validity claim.

## 6. Deterministic simulation design

The simulator mirrors the RH-006 estimator:

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

The random generator is a pinned SplitMix64 stream with Rademacher innovations. This avoids dependence on platform-specific normal RNG behavior.

Default pilot geometry:

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

The deterministic result artifact is:

~~~text
docs/research/RELATIONAL_HARMONICS_BLOCK_INFERENCE_SIMULATION_V1.json
SHA-256: 25a85de9000a51953ff17edc96928d271d3e2f2496996bbd7016a89568a959d1
~~~

## 7. Simulation findings

### Coefficient-null

With all incremental relational coefficients set to zero:

~~~text
mean origin loss differential = -0.00188764
empirical rejection = 0 / 120
~~~

This is not evidence of a correctly calibrated 5% test. It demonstrates the finite-sample nesting point: the coefficient restriction can induce a non-zero risk difference for the two estimated forecasting methods.

### Approximate equal-risk DGP targets

The simulation-only calibrated scenarios produced:

| Scenario | L | Rejections / 120 | Empirical size | Mean D |
|---|---:|---:|---:|---:|
| IID equal-risk target | 3 | 12 | 0.100 | 0.000116 |
| IID equal-risk target | 6 | 16 | 0.133 | 0.000116 |
| AR(0.5) equal-risk target | 3 | 10 | 0.083 | 0.000514 |
| AR(0.5) equal-risk target | 6 | 13 | 0.108 | 0.000514 |
| AR(0.8) equal-risk target | 3 | 13 | 0.108 | 0.000981 |
| AR(0.8) equal-risk target | 6 | 18 | 0.150 | 0.000981 |
| AR(0.5) heteroskedastic target | 3 | 10 | 0.083 | 0.000357 |
| AR(0.5) heteroskedastic target | 6 | 14 | 0.117 | 0.000357 |

The Monte Carlo size standard errors for the non-zero rejection rates are roughly 0.025–0.033 at 120 repetitions, so these are still pilot-scale measurements. Nevertheless, the result is not close enough to nominal behavior to justify promotion to formal inference, particularly as serial dependence and heteroskedasticity strengthen.

The larger L=6 sensitivity case is worse in every equal-risk scenario in this pilot. This is another reason not to choose block length by intuition alone.

## 8. New methodological conclusion

**Do not promote the origin-loss MBB to the formal method.**

The more promising next candidate is a **restricted-DGP nested rolling bootstrap**:

1. preserve the exact RH-006 feature and split geometry;
2. define an explicit null forecasting DGP for the nested comparison;
3. resample dependent errors/innovations with a declared dependence mechanism;
4. reconstruct the outcome series under the null;
5. refit both nested methods from scratch at every rolling origin;
6. recompute training-only standardization and fixed ridge fitting inside every bootstrap replicate;
7. retain the full origin-level loss-differential vector;
8. compute the same origin-level statistic;
9. verify size across dependence, heteroskedasticity, ridge, horizon, overlap, and near-singularity scenarios.

This would directly address the finite-sample nesting problem that the current loss-level bootstrap cannot see.

The null DGP itself remains an open specification problem. A restricted context regression with dependence-preserving residual generation is a plausible starting point, but its assumptions must be written and tested rather than inherited by citation.

## 9. Required approval matrix

Before the applicability artifact can leave not-approved-for-execution, the future candidate should demonstrate, prospectively and independently of the confirmatory result:

- nominal null size under IID, weak dependence, strong dependence, and heteroskedasticity;
- stability across declared fixed ridge values;
- correct handling of training-only preprocessing;
- correct horizon alignment;
- correct overlap reproduction;
- deterministic failure near singular designs;
- sensitivity to prespecified block/dependence choices;
- reproducibility of bootstrap artifacts from identical source + manifest + seed;
- no result-dependent method or bandwidth switching.

A small number of successful simulations is not sufficient. The acceptance target is calibration across the declared failure modes.

## 10. Hard stop

Until the raw-series/refit bridge is derived and passes the above simulation gate:

- ForecastInferenceSelectionPath remains stop-assumption-failure;
- the estimator applicability artifact remains not-approved-for-execution;
- no p-value or confidence interval is emitted by RH-006;
- dependence profiles remain descriptive;
- surrogate exceedance fractions remain empirical diagnostics;
- the rolling MSE difference remains an estimand/diagnostic only.

## References

- Clark, T. E. & McCracken, M. W. (2015), *Nested forecast model comparisons: A new approach to testing equal accuracy*, Journal of Econometrics 186(1), 160–177. DOI: 10.1016/j.jeconom.2014.06.016.
- Doko Tchatoka, F. & Haque, Q. (2023), *On bootstrapping tests of equal forecast accuracy for nested models*, Journal of Forecasting 42(7), 1844–1864. DOI: 10.1002/for.2987.
- Giacomini, R. & White, H. (2006), *Tests of Conditional Predictive Ability*, Econometrica 74(6), 1545–1578. DOI: 10.1111/j.1468-0262.2006.00718.x.
- Zhu, Y. & Timmermann, A. (2020), *Can Two Forecasts Have the Same Conditional Expected Accuracy?* https://arxiv.org/abs/2006.03238.

This document freezes the current negative result and identifies the raw-series/refit bootstrap as the next methodological target.