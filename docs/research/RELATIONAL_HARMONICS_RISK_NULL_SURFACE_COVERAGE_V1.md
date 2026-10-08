# RH-006 Risk-Null Surface Coverage — v1

## Status

Exploratory diagnostic only. Not approved for formal inference.

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

This pass attacks the next unresolved issue after the dependent/heteroskedastic quadratic risk derivation: whether oracle bootstrap behavior depends materially on which nuisance point of the finite-sample equal-risk surface is used.

## 1. Why this experiment is necessary

The current risk equation is:

`Delta(gamma) = gamma' Q gamma + b' gamma + c`.

The scalar equal-amplitude construction chooses:

`gamma = kappa v`

for a predeclared direction `v`.

That gives a valid algebraic slice of the accuracy-null surface, but not a direction-free characterization of:

`E[D_o] = 0`.

A general equal-accuracy test therefore has a nuisance-point problem. Different points on the null surface can imply different forecast means and different finite-sample distributions after rolling refitting.

A bootstrap that calibrates at one arbitrary point could therefore be point-valid without being uniform over the scientific null.

## 2. Predeclared surface points

The pilot evaluates eight fixed directions in the three-dimensional relational coefficient space:

- a2b
- b2a
- turn_taking
- plus_plus_plus
- plus_plus_minus
- plus_minus_plus
- minus_plus_plus
- plus_minus_minus

For each feature path and direction, the positive root of the exact dependent/heteroskedastic risk quadratic defines a null point.

These directions were fixed before examining bootstrap outcomes.

## 3. Executed design

Canonical command:

`python scripts/research/rh006_oracle_null_surface_coverage_splitmix64.py --paths 20 --outcomes-per-point 2 --bootstrap 199 --seed 20261010`

Geometry:

- train = 48
- gap = 4
- test = 16
- origins = 24
- step = 16
- ridge = `1e-8`
- Bartlett lag = 3
- alpha = 0.05
- 20 feature paths per scenario
- 8 fixed null-surface directions
- 2 independent outcome calibrations per direction/path
- 199 bootstrap replicates

This yields 160 null points and 320 outcome calibrations per scenario.

Scenarios:

- IID
- AR(0.5)
- AR(0.8)
- AR(0.5) + heteroskedasticity

## 4. Observed pilot results

Overall rejection rates at nominal 5%:

| Error process | Outcomes | Overall rejection |
|---|---:|---:|
| IID | 320 | 0.040625 |
| AR(0.5) | 320 | 0.05625 |
| AR(0.8) | 320 | 0.040625 |
| AR(0.5) + heteroskedastic | 320 | 0.034375 |

No catastrophic over-rejection appeared in this pilot.

However, direction-specific rejection rates ranged from 0.000 to 0.125 in individual scenarios. Each direction has only 40 outcome calibrations, so those point estimates are too noisy to establish meaningful size differences.

The scientifically relevant finding is therefore:

> the pilot does not yet show a clear size failure across the sampled null surface, but it also does not justify a uniform-validity claim.

## 5. The geometry is substantially nonuniform in coefficient amplitude

The positive risk-null roots differed materially by direction.

For example, in the AR(0.8) scenario, the pilot's direction means were approximately:

- a2b: 0.202
- b2a: 0.203
- turn_taking: 0.293
- plus_plus_plus: 0.185
- mixed-sign directions: approximately 0.30

The exact values are retained in the executed JSON receipt.

Thus two parameter vectors can both satisfy the finite-sample equal-risk equation while having materially different relational coefficient magnitudes and orientations.

That is enough to keep the null-surface nuisance issue open even though the current bootstrap pilot is well behaved.

## 6. What the pilot does and does not establish

It supports:

- the implementation can place multiple predeclared coefficient vectors on the same finite-sample risk-null surface;
- the oracle full-refit bootstrap remains mechanically executable at those points;
- this synthetic DGP does not immediately produce explosive over-rejection across the sampled points.

It does not establish:

- uniform size control across the full null surface;
- validity with estimated nuisance parameters;
- validity under endogenous feature processes;
- validity for real RH-006 targets;
- asymptotic validity of the fixed-ridge, training-standardized estimator;
- adequacy of the Bartlett studentizer for the true rolling-origin dependence.

## 7. Next decisive experiment

The next experiment should increase surface coverage rather than merely increasing bootstrap repetitions at one point.

Required dimensions:

1. more null points per feature path;
2. both signs for asymmetric directions when `b != 0`;
3. near-eigenvector directions that approach the smallest and largest curvature of `Q`;
4. ridge sensitivity;
5. near-singular feature windows;
6. stronger and weaker serial correlation;
7. stronger heteroskedasticity;
8. estimated-nuisance rather than oracle-known covariance;
9. joint feature/outcome DGPs for endogenous-feature stress tests.

The output should summarize the maximum observed rejection rate over the predeclared null grid and distinguish genuine surface instability from Monte Carlo noise.

## 8. Decision boundary

A materially nonuniform size surface would force one of two outcomes:

- narrow the scientific estimand to a predeclared direction and state that restriction explicitly; or
- construct a uniform/least-favorable null procedure whose size is controlled over the admissible equal-accuracy surface.

Neither has been approved.

Current boundary remains:

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

## References

Clark, T. E., & McCracken, M. W. (2015). *Nested forecast model comparisons: A new approach to testing equal accuracy*. Journal of Econometrics, 186(1), 160–177. DOI: 10.1016/j.jeconom.2014.06.016.

Doko Tchatoka, F., & Haque, Q. (2023). *On bootstrapping tests of equal forecast accuracy for nested models*. Journal of Forecasting, 42(7), 1844–1864. DOI: 10.1002/for.2987.

Zhu, Y., & Timmermann, A. (2020). *Can Two Forecasts Have the Same Conditional Expected Accuracy?* arXiv:2006.03238.

Harvey, D. I., Leybourne, S. J., & Zu, Y. (2025). *Testing for Equal Average Forecast Accuracy in Possibly Unstable Environments*. Journal of Business & Economic Statistics, 43(3), 643–656. DOI: 10.1080/07350015.2024.2418835.
