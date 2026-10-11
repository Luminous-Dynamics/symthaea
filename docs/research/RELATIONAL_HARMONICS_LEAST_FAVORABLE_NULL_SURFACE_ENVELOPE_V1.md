# RH-006 Least-Favorable Null-Surface Envelope — v1

## Status

Research diagnostic only. Not approved for formal inference.

applicability = not-approved-for-execution

selection = stop-assumption-failure

formal p-value = disabled

formal confidence interval = disabled

## Purpose

The RH-006 equal-risk null is a composite surface:

Delta(gamma) = gamma' Q gamma + b' gamma + c = 0

A single directional root cannot establish uniform calibration over that surface.

This experiment therefore uses a predeclared finite directional grid and records the maximum and minimum bootstrap rejection rates across the grid as a diagnostic envelope.

The maximum is not used as a critical value. A finite-grid maximum is not itself a mathematically justified least-favorable distribution.

## Frozen grid

Eight normalized coefficient directions were predeclared:

- [1,0,0]
- [0,1,0]
- [0,0,1]
- [1,1,1]
- [1,1,-1]
- [1,-1,1]
- [1,-1,-1]
- [2,1,0]

Each direction's antipode was also included, yielding 16 total directions.

The grid was fixed before bootstrap outcomes were generated.

## Frozen evaluator

- train = 48
- gap = 4
- test = 16
- rolling origins = 24
- step = 16
- ridge = 1e-8
- Bartlett lag = 3
- nominal alpha = 0.05
- independent estimator-mechanics implementation
- dependent/heteroskedastic outcome DGP where applicable

For every feature path and direction, the exact finite-sample equal-risk quadratic is solved first. Directions without a positive root are recorded as unsupported rather than silently omitted from the scientific interpretation.

## Completed pilot

The first completed envelope execution used:

- 12 feature paths per scenario
- 1 null outcome per supported direction/path
- 49 bootstrap draws
- four outcome processes: IID, AR(0.5), AR(0.8), AR(0.5)+heteroskedastic

The resulting directional rejection envelopes were:

| Outcome DGP | Minimum directional rejection | Maximum directional rejection | Median directional rejection | Root support |
|---|---:|---:|---:|---:|
| IID | 0% | 25% | 8.3% | 100% |
| AR(0.5) | 0% | 16.7% | 0% | 100% |
| AR(0.8) | 0% | 8.3% | 0% | 100% |
| AR(0.5) + heteroskedastic | 0% | 16.7% | 8.3% | 100% |

These maxima are not evidence of size failure. With only 12 paths and one outcome per direction, the direction-level rejection fraction is extremely coarse.

The important methodological result is that an envelope statistic must be interpreted together with its Monte Carlo support; otherwise the worst observed direction is itself dominated by sampling noise.

## Deeper completed focus

The most relevant AR(0.5)+heteroskedastic case was then increased to:

- 50 feature paths
- 2 independent null outcomes per supported direction/path
- 99 bootstrap draws

This produced:

- 800 supported null points
- maximum directional rejection = 2%
- minimum directional rejection = 0%
- median directional rejection = 0%
- all 16 directions had 100% root support

The five highest direction-level rejection rates were only 2% or 0% in this deeper focus.

This materially reduces concern that the earlier 16.7–25% pilot maxima represented a systematic over-rejection mechanism. They were primarily a consequence of tiny per-direction cells.

## Important limitation

The deeper focus does not establish uniform size.

It uses:

- a single fixed ridge value
- a finite direction grid
- oracle nuisance quantities
- conditionally generated features
- a synthetic DGP
- only two null outcomes per direction/path

It therefore establishes only that this particular synthetic calibration did not expose catastrophic directional over-rejection at the deeper focus scale.

## Relation to the previous root-support result

The envelope pilot itself uses ridge 1e-8 and the ordinary synthetic feature construction, where all tested grid directions had positive roots.

That should not be confused with the previously established severe weak-direction result at epsilon = 1e-4, ridge = 1e-4, weak direction [0,1,-2], where only about 42% of feature paths had a positive finite-sample equal-risk root even with oracle nuisance.

Thus:

surface calibration

and

surface support / existence

remain distinct gates.

A direction cannot participate in a uniform accuracy-null procedure if the declared null parameterization has no admissible point there.

## Why the least-favorable concept remains unresolved

Elliott, Müller & Watson show that nuisance parameters present under the null create composite-hypothesis problems where a least-favorable distribution can matter for valid testing, and they construct approximate least-favorable distributions numerically in nonstandard settings. They frame the problem around valid tests under composite nulls rather than simply choosing the most adverse observed cell. 

Hill's weak-identification work likewise warns that bootstrap methods can become invalid under weak or non-identification and that naive supremum or average approaches do not automatically provide robustness.

Therefore the correct next step is not:

choose the largest observed rejection rate

but:

define the admissible null surface

→ define the nuisance/identification state

→ derive a uniform calibration rule

→ show its worst-case size prospectively.

## Policy consequence

The current architecture should not expose a least-favorable p-value API yet.

The only safe status is:

- envelope = diagnostic
- direction grid = fixed research design
- maximum rejection = descriptive worst-cell diagnostic
- no automatic promotion to critical value
- root-support failure = explicit inapplicability
- weak-identification regimes remain a separate gate

## Reproducibility

Canonical script:

scripts/research/rh006_least_favorable_null_surface_envelope_splitmix64.py

Local script SHA-256:

927c959e38b0067d43cf6a0152449cd916f364e21b3eeeecfc9290c69dcc9d24

Pilot command:

python scripts/research/rh006_least_favorable_null_surface_envelope_splitmix64.py --paths 12 --outcomes 1 --bootstrap 49 --seed 20261040

Focus command:

scenario(0.5, True, 20261050, paths=50, outcomes=2, bootstrap=99)

The first unvectorized implementation timed out and was discarded as non-evidence. A later 100-path × 2-outcome × 99-bootstrap focus run also exceeded the execution ceiling and was not counted.

## Next boundary

The next scientific step is a uniform null-surface parameterization rather than a larger arbitrary direction grid.

A stronger candidate is to parameterize the actual quadratic surface by local curvature/eigenstructure of Q, then predeclare coverage regions such as:

- high-curvature directions
- low-curvature / weakly identified directions
- sign-asymmetric branches
- root-existence boundary
- admissible origin-local conditioning region

The estimated-nuisance bootstrap should then be evaluated across those regions, with weak-identification robustness treated as an explicit requirement rather than an afterthought.
