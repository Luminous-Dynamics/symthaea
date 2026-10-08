# RH-006 Identifiable-Subspace Policy — v1

## Status

Exploratory diagnostic only. Not approved for formal inference.

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

## Purpose

The previous conditioning/ridge stress test showed that positive ridge can restore numerical solvability without restoring identification of the equal-risk surface.

This experiment asks whether an apparently natural repair — projecting the three relational channels onto an identifiable SVD subspace — preserves the original scientific hypothesis.

The answer is generally **no**.

## Policy simulation

The relational feature columns are standardized within each rolling training window and pooled across origins only for this diagnostic.

For singular values (s_1 ge s_2 ge s_3), a candidate identifiability threshold (	au) retains components satisfying:

[
s_j / s_1 ge 	au.
]

Candidate thresholds are fixed before evaluation:

- `1e-2`
- `1e-3`
- `1e-4`
- `1e-5`

No threshold is approved by this experiment.

For each retained subspace, 256 deterministic directions on the three-dimensional coefficient sphere are projected onto the subspace. The retained norm measures how much of the original coefficient direction survives.

## Executed design

Canonical command:

`python scripts/research/rh006_null_surface_identifiable_subspace_policy_splitmix64.py --paths 8 --directions 256 --seed 20261014`

Geometry:

- train = 48
- test = 16
- gap = 4
- origins = 24
- step = 16
- 8 feature paths
- 256 deterministic surface directions
- relational collinearity stress:
  - epsilon = `1e-2`
  - epsilon = `1e-3`
  - epsilon = `1e-4`
  - epsilon = `0`

The deterministic SplitMix64 generator and feature mechanics mirror the conditioning stress battery.

## Observed results

The stressed rank behavior was:

| epsilon | threshold | median retained rank | median retained direction norm | directions retaining >=90% |
|---:|---:|---:|---:|---:|
| 1e-2 | 1e-2 | 1 | 0.510 | 10.2% |
| 1e-2 | 1e-3 | 3 | 1.000 | 100% |
| 1e-3 | 1e-2 | 1 | 0.510 | 10.2% |
| 1e-3 | 1e-3 | 1 | 0.510 | 10.2% |
| 1e-4 | 1e-4 | 1 | 0.510 | 10.2% |
| 1e-4 | 1e-5 | 3 | 1.000 | 100% |
| 0 | 1e-5 | 1 | 0.510 | 10.2% |

The result is geometric rather than inferential.

Whenever the threshold removes two of the three relational directions, the projection preserves only about half the coefficient-direction norm for a typical surface direction. Only about one tenth of the deterministic directions retain at least 90% of their original norm.

## Scientific consequence

A subspace projection is not a neutral numerical repair.

Suppose the original scientific alternative is:

[
H_1:gamma 
e 0
]

in the full three-channel relational coefficient space.

After projection onto a rank-(r) subspace (P), the effective alternative becomes:

[
H_1^{(P)}:Pgamma 
e 0.
]

Any component in the discarded complement is deliberately ignored.

Therefore a projected test can have a very different answer even when the original full-space hypothesis is unchanged.

The projection is scientifically equivalent only under a separate, predeclared claim that the discarded directions are outside the admissible scientific parameter space.

That is a **change of estimand**, not merely a conditioning fix.

## Implication for RH-006

The preferred primary policy remains:

[
	ext{inadmissible conditioning}
ightarrow
	ext{hard stop}.
]

An identifiable-subspace lane can still be useful, but only as an explicitly separate research estimand with its own:

- parameter-space definition;
- feature construction contract;
- scientific interpretation;
- null surface;
- power analysis;
- inferential bridge.

It must not silently substitute for the full three-channel RH-006 hypothesis.

## What this does not establish

This experiment does not establish:

- that real RH-006 data are rank deficient;
- that any particular singular-value threshold is appropriate;
- that subspace projection has valid inferential size;
- that the discarded directions are scientifically unimportant;
- that ridge should be changed;
- that formal inference is available.

## Next gate

The next methodological experiment should compare three explicitly separated policies under the same synthetic DGP:

1. full-space + hard fail on inadmissible conditioning;
2. full-space + no projection but ridge-only numerical stabilization;
3. separately declared identifiable-subspace estimand.

The comparison should measure null size, local power, alternative-direction coverage, and rejection behavior under near-singularity.

No policy should be promoted based on whichever produces the most favorable confirmatory result.

## Provenance

The committed execution receipt is:

`docs/research/RELATIONAL_HARMONICS_RISK_NULL_SURFACE_IDENTIFIABLE_SUBSPACE_POLICY_EXECUTED_V1.json`

The local deterministic mirror used for the executed screening had SHA-256:

`12367b3942a0b2b04e5f939254019da5323447c4529ee9bf5efc31f37446fd60`

The repository script remains independently reviewable and executable; the current receipt explicitly distinguishes the local executed mirror from the committed source file.

## Boundary

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`
