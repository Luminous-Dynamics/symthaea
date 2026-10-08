# RH-006 Origin-Local Conditioning Adversary — v1

## Status

Exploratory diagnostic only. Not approved for formal inference.

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

## Finding

The conditioning policy cannot be evaluated only on a pooled relational design.

RH-006 refits the forecasting method separately at each rolling origin. Therefore the admissibility of the forecasting operator is an **origin-local** property.

A pooled SVD can conceal a singular window.

## Adversarial construction

The diagnostic uses the existing synthetic feature DGP and rolling geometry:

- train = 48
- gap = 4
- test = 16
- origins = 24
- step = 16

For the adversarial condition, only the final origin's 48-sample training window is forced to exact rank-one relational structure by setting both additional relational channels equal to the primary relational channel inside that window.

This leaves the rest of the time series unchanged.

The purpose is not to imitate a realistic data pathology. It is to test whether a proposed conditioning summary detects a failure at the actual estimator unit.

## Executed diagnostic

Canonical command:

`python scripts/research/rh006_origin_local_conditioning_adversary_splitmix64.py --paths 8 --seed 20261017`

The local execution was a deterministic mirror of the committed repository experiment. The receipt explicitly records that distinction.

## Results

| condition | median pooled min singular ratio | median minimum origin-local ratio | paths with any origin below 1e-3 |
|---|---:|---:|---:|
| baseline | 0.350 | 0.272 | 0 / 8 |
| singular final origin | 0.330 | 4.9 × 10^-33 | 8 / 8 |

The key observation is that the pooled statistic barely moves:

[
0.350 ightarrow 0.330
]

while the actual affected rolling origin becomes numerically singular:

[
0.272 ightarrow 4.9	imes10^{-33}.
]

A pooled policy could therefore accept a dataset whose final forecast operator cannot be validly fit without regularization or an explicit fail-closed rule.

## Scientific consequence

The admissibility test must be attached to the **same unit at which the forecasting method is estimated**.

For RH-006 this implies:

[
	ext{origin } o
ightarrow
	ext{training-window conditioning}
ightarrow
	ext{origin admissibility}.
]

The aggregate qualification should then fail closed if any required origin violates the predeclared admissibility condition.

A pooled diagnostic can remain useful as a descriptive global summary, but it cannot serve as the primary admissibility gate.

## Interaction with ridge

This result also sharpens the earlier ridge boundary.

Ridge can make a singular origin numerically executable, but it cannot retroactively make the underlying feature geometry well identified.

Therefore the policy should not be:

[
	ext{pooled design looks fine} Rightarrow 	ext{run}.
]

Nor should it be:

[
	ext{ridge solves} Rightarrow 	ext{identified}.
]

Instead:

[
	ext{every required origin passes the predeclared conditioning policy}
]

must be established before the inferential path becomes eligible.

## Next gate

The next simulation should combine:

1. origin-local conditioning;
2. ridge variation;
3. near-singular rather than only exact-singular windows;
4. null-surface direction coverage;
5. full-space versus identifiable-subspace estimands;
6. empirical null-size and local-power diagnostics.

The admissibility rule itself must be frozen before any confirmatory result is inspected.

## What this does not establish

This experiment does not establish that real RH-006 windows are singular or near singular. It does not establish a scientifically correct threshold, inferential validity, or a preferred ridge value.

It only closes the methodological loophole that pooled conditioning is sufficient to certify every rolling forecast origin.

## Boundary

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`
