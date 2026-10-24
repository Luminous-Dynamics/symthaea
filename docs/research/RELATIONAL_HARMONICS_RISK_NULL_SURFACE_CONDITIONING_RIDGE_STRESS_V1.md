# RH-006 Risk-Null Surface Conditioning and Ridge Stress — v1

## Status

Exploratory diagnostic only. Not approved for formal inference.

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

This experiment attacks numerical conditioning and identifiability of the finite-sample equal-risk surface.

## Why this matters

The finite-sample accuracy null is

`Delta(gamma) = gamma' Q gamma + b' gamma + c`.

A positive ridge can make the forecasting linear system solvable when a feature window is badly conditioned. That does **not** imply that the coefficient direction, or the equal-risk surface itself, is identified.

The experiment therefore separates:

1. linear-algebra solvability of the fitted forecasting operator;
2. conditioning of the standardized feature window and risk quadratic `Q`;
3. stability of equal-risk roots across coefficient directions.

## Executed design

Canonical command:

`python scripts/research/rh006_null_surface_conditioning_ridge_stress_splitmix64.py --paths 8 --directions 256 --seed 20261013 --ridges 0 1e-12 1e-10 1e-8 1e-6 1e-4 1e-2`

Geometry:

- train = 48
- gap = 4
- test = 16
- origins = 24
- step = 16
- outcome AR coefficient = 0.8
- heteroskedastic innovation modulation = 0.1
- 8 feature paths per condition
- 256 deterministic directions on the coefficient sphere
- ridge grid: `0`, `1e-12`, `1e-10`, `1e-8`, `1e-6`, `1e-4`, `1e-2`

Near-collinearity was injected into the three relational channels with:

- `epsilon = 1e-2`
- `epsilon = 1e-3`
- `epsilon = 1e-4`
- `epsilon = 0` for exact collinearity

## Observed screening results

| epsilon | ridge | median max design condition | median `Q` condition | median min eigenvalue of `Q` | mean directional root median | global max root | singular paths |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0e+00 | 7.65816e+16 | — | — | — | — | 8 |
| 0 | 1e-08 | 7.65816e+16 | 2.5825e+16 | -3.846e-18 | 0.160364 | 37.2976 | 0 |
| 0 | 1e-04 | 7.65816e+16 | 2.56844e+16 | -3.9295e-18 | 0.160363 | 37.2976 | 0 |
| 0.0001 | 0e+00 | 33774.4 | 5.43467e+08 | 1.76502e-10 | 0.269099 | 61.9193 | 0 |
| 0.0001 | 1e-08 | 33774.4 | 5.45023e+08 | 1.75947e-10 | 0.263648 | 60.5016 | 0 |
| 0.0001 | 1e-04 | 33774.4 | 1.433e+10 | -6.92721e-12 | 0.160348 | 37.3252 | 0 |
| 0.001 | 0e+00 | 3282.14 | 5.42376e+06 | 1.780e-08 | 0.277914 | 61.7861 | 0 |
| 0.001 | 1e-08 | 3282.14 | 5.42376e+06 | 1.780e-08 | 0.277857 | 61.7728 | 0 |
| 0.001 | 1e-04 | 3282.14 | 2.00356e+07 | 4.89701e-09 | 0.188503 | 43.4413 | 0 |
| 0.01 | 0e+00 | 358.178 | 52558.2 | 1.83434e-06 | 0.278055 | 26.4355 | 0 |
| 0.01 | 1e-08 | 358.178 | 52558.2 | 1.83434e-06 | 0.278054 | 26.4355 | 0 |
| 0.01 | 1e-04 | 358.178 | 52724 | 1.82909e-06 | 0.272031 | 25.9157 | 0 |

## Interpretation

The decisive boundary is exact collinearity:

- with `ridge = 0`, all 8 feature paths fail in the relational fit because the normal equations are singular;
- adding positive ridge makes those fits executable;
- despite that, the design condition number remains about `10^17` and the smallest eigenvalue of `Q` is numerically zero, so the surface retains an effectively flat direction;
- ridge therefore changes **solvability**, not the underlying identification problem.

Near-collinearity also produces very large positive equal-risk roots. At `epsilon = 1e-4`, the `Q` condition number is already above `5 × 10^8` in the low-ridge rows and the sampled maximum root exceeds 60.

These coefficients are not claims about plausible RH-006 effects. They are a stress-test result showing that the finite-sample null geometry can become numerically extreme as the feature design approaches rank deficiency.

## Required next gate

Run a predeclared identifiability-policy simulation over:

`condition number / minimum singular value`

×

`ridge`

×

`surface direction`

and compare:

1. full parameterization;
2. identifiable-subspace treatment;
3. hard fail-closed admissibility.

No threshold should be selected from the observed effect. The policy must be frozen before any real-data inferential execution.

## What this does not establish

This experiment does not establish that real RH-006 feature windows are near singular, that any particular threshold is scientifically correct, that ridge should be increased, or that any inferential size control has been achieved.

## Provenance

Script SHA-256:

`192b601ba776bcd6ef44646cfb7ce69a570106059a38a8511a88cba1ed2ab338`

Executed payload SHA-256:

`51f2d79aa75fe2b3cceddf43516b57475644d579dabf3e0ad7bdff3d69517fd5`
