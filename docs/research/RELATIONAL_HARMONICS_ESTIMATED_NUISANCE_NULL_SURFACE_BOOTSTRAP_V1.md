# RH-006 Estimated-Nuisance Null-Surface Bootstrap — v1

## Status

Research diagnostic only. Not approved for formal inference.

\`applicability = not-approved-for-execution\`

\`selection = stop-assumption-failure\`

\`formal p-value = disabled\`

\`formal confidence interval = disabled\`

## Why this experiment exists

RH-006's equal-accuracy null is not necessarily \`beta_relational = 0\`. For nested forecasting methods, finite-sample equal predictive risk can occur at nonzero extra coefficients because estimation variance can offset the forecasting gain from the larger model. Clark and McCracken explicitly develop bootstrap methods around this finite-sample equal-risk null. https://doi.org/10.1016/j.jeconom.2014.06.016

The remaining question was whether that null can still be calibrated when the nuisance quantities required to construct it are estimated rather than oracle-known.

The experiment therefore separates:

\`oracle finite-sample null\`

from

\`plug-in estimated-nuisance finite-sample null\`.

Clark and McCracken's overlapping-model work explicitly notes that the null distribution can depend on unknown nuisance quantities and develops bootstrap treatment for that situation.

## Exact diagnostic construction

Each Monte Carlo dataset:

1. generates the fixed RH-006 feature path;
2. computes the oracle finite-sample equal-risk root on a predeclared relational direction;
3. generates the observed outcome from that oracle null point with AR(0.5) and heteroskedastic innovations;
4. estimates nuisance parameters from the observed dataset;
5. reconstructs the finite-sample equal-risk root using those plug-in nuisance estimates;
6. generates restricted-null bootstrap worlds from that estimated null point;
7. evaluates every bootstrap world using the same rolling estimator and Bartlett studentizer.

The bootstrap is vectorized over draws, but the forecasting operators are still recomputed separately for every rolling origin.

The experiment intentionally does not use a double-bootstrap correction and does not claim uniform validity over the composite null.

## Frozen geometry

- train = 48
- gap = 4
- test = 16
- rolling origins = 24
- step = 16
- horizon = 1
- ridge = \`1e-8\` and \`1e-4\`
- conditioning stress = \`1e-2\`, \`1e-3\`, \`1e-4\`
- dependent error process = AR(0.5)
- heteroskedastic innovation scale depends on the common-driver channel
- Bartlett lag = 3

Predeclared directions:

- strong: \`[1,1,0.5]\`
- weak: \`[0,1,-2]\`

## Main finding: null-surface topology dominates the worst failure

The deepest root-failure diagnostic used:

- 1000 feature paths;
- \`epsilon = 1e-4\`;
- ridge \`1e-4\`;
- weak direction.

Results:

| Outcome | Count |
|---|---:|
| Oracle positive equal-risk root available | 419 / 1000 |
| Oracle positive root unavailable | 581 / 1000 |
| Estimated positive root available | 417 / 1000 |
| Estimated root available conditional on oracle root | 417 / 419 |
| Estimated-root failures from negative discriminant | 1 |
| Estimated-root failures with nonnegative discriminant but no positive root | 1 |

So the severe-cell failure is overwhelmingly **not** caused by nuisance estimation.

The deeper statement is:

> The finite-sample equal-risk surface itself often has no intersection with the chosen positive weak-direction ray.

Estimated nuisance shifts the existing root, but does not create the dominant failure.

Among the paths where both roots exist, the median estimated/oracle root ratio was approximately \`0.967\`, a roughly 3–4% downward shift.

## Bootstrap calibration result

At \`epsilon = 1e-3\`, ridge \`1e-4\`, weak direction, a 500-dataset × 99-bootstrap diagnostic produced:

- oracle root availability = 100%;
- estimated root availability = 100%;
- oracle bootstrap rejection = 5.6%;
- estimated-nuisance bootstrap rejection = 6.0%;
- median estimated/oracle root ratio = 0.969.

At the more pathological \`epsilon = 1e-4\`, ridge \`1e-4\`, weak direction, a separate 500×99 diagnostic produced:

- oracle root availability = 43.8%;
- estimated root availability = 43.8%;
- oracle bootstrap rejection = 4.11%;
- estimated-nuisance bootstrap rejection = 3.20%;
- median estimated/oracle root ratio = 0.965.

The latter cell is based on only the 219 datasets for which the composite null point exists in this direction. Those rejection frequencies are Monte Carlo diagnostics, not validated test-size statements.

## Interpretation

Three boundaries are now separated.

### 1. Estimability of nuisance

The AR/heteroskedastic nuisance estimates are imperfect, but on root-supported cells they move the finite-sample equal-risk root by only a few percent in the median.

### 2. Existence of the finite-sample null point

In the weak, nearly rank-one regime, this is much more severe. The oracle null point itself can fail to exist on a large fraction of feature paths.

A bootstrap procedure cannot repair that by better estimating nuisance parameters. There is no positive directional equal-risk point to bootstrap from.

### 3. Calibration of the resulting composite-null bootstrap

Where the null point exists, the pilot does not show catastrophic divergence between oracle and estimated-nuisance calibration. But the sample is still too small to establish uniform size, especially over the full risk surface.

## Why this matters for the projected lane

This result reinforces the previous projection-preservation certificate.

There are now two separate reasons not to treat projection as a generic rescue:

\`projection can change the hypothesis\`

and

\`the original finite-sample null may not intersect the chosen directional parameterization at all\`.

A projected model can therefore become numerically well behaved while solving a different statistical problem.

## Evidence boundary

This experiment does not establish:

- a valid RH-006 p-value;
- a confidence interval;
- uniform size control over the equal-risk surface;
- validity under feature/outcome endogeneity;
- validity under the actual Symthaea Rust implementation;
- validity after adaptive subspace selection;
- a scientifically preferred ridge value.

The formal gate therefore remains closed.

## Canonical artifacts

- \`scripts/research/rh006_estimated_nuisance_null_surface_bootstrap_v4.py\`
- \`docs/research/RELATIONAL_HARMONICS_ESTIMATED_NUISANCE_NULL_SURFACE_BOOTSTRAP_V1.md\`
- \`docs/research/RELATIONAL_HARMONICS_ESTIMATED_NUISANCE_NULL_SURFACE_BOOTSTRAP_EXECUTED_V1.json\`

The repository script git blob SHA-1 is:

\`20ca25b16a5a446afdaeecf435171ad01e78e6fd\`

The repository script SHA-256 is:

\`a7edf0122415d8e8307346ad9e1dbd5a66b28259bfa0df55af5786e974e67904\`

The local execution source had a different byte hash and was therefore not represented as byte-identical to the repository blob. This is intentional.

## Next gate

The strongest next attack is a least-favorable null-surface envelope:

\`direction / root-existence / conditioning\`

×

\`estimated nuisance\`

×

\`bootstrap rejection\`

with the envelope computed prospectively rather than selecting the worst direction after inspecting a confirmatory dataset.

Only after that should RH-006 choose between a direction-restricted estimand and a uniform composite-null procedure.
