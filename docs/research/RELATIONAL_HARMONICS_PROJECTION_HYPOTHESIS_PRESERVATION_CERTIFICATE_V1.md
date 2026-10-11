# RH-006 Origin-Local Projection Hypothesis-Preservation Certificate — v1

## Status

Research diagnostic only. Not approved for formal inference.

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

This experiment tests a stricter condition than numerical rank: whether an origin-local SVD projection preserves the declared RH-006 hypothesis itself.

## Core rule

A projection may define a valid **new projected estimand**, but it cannot be treated as inference for the original RH-006 coefficient-space hypothesis unless the declared effect space is preserved at every required rolling origin.

For the full three-channel hypothesis, preservation means all three singular directions are retained at every origin. For a predeclared directional hypothesis, the origin's training-standardized coefficient vector must lie in the retained subspace up to a numerical tolerance of `1e-10`.

This is deliberately an identity condition, not a scientific tuning threshold.

## Frozen design

- train = 48
- gap = 4
- test = 16
- origins = 24
- step = 16
- conditioning stresses = `1e-2`, `1e-3`, `1e-4`, exact `0`
- SVD retention thresholds = `1e-2`, `1e-3`, `1e-4`
- Monte Carlo paths = 1000 per cell
- deterministic RNG = SplitMix64 + Rademacher innovations

Predeclared coefficient directions:

- `strong = [1,1,0.5]`
- `weak = [0,1,-2]`

Because the SVD is computed on training-standardized features, each raw coefficient direction is transformed by the origin's training scales before projection retention is measured.

## Executed result

| epsilon | tau | full 3-channel preserved | strong path-preserved | strong worst-origin retention (median) | weak path-preserved | weak worst-origin retention (median) |
|---:|---:|---:|---:|---:|---:|---:|
| `1e-2` | `1e-2` | 0% | 0% | 0.904 | 0% | 6.93e-5 |
| `1e-2` | `1e-3` | 100% | 100% | 1.000 | 100% | 1.000 |
| `1e-2` | `1e-4` | 100% | 100% | 1.000 | 100% | 1.000 |
| `1e-3` | `1e-2` | 0% | 0% | 0.904 | 0% | 5.12e-6 |
| `1e-3` | `1e-3` | 0% | 0% | 0.904 | 0% | 7.09e-6 |
| `1e-3` | `1e-4` | 100% | 100% | 1.000 | 100% | 1.000 |
| `1e-4` | `1e-2` | 0% | 0% | 0.905 | 0% | 5.16e-7 |
| `1e-4` | `1e-3` | 0% | 0% | 0.905 | 0% | 5.16e-7 |
| `1e-4` | `1e-4` | 0% | 0% | 0.905 | 0% | 6.97e-7 |
| `0` | `1e-2` | 0% | 0% | 0.905 | 0% | 9.11e-18 |
| `0` | `1e-3` | 0% | 0% | 0.905 | 0% | 9.11e-18 |
| `0` | `1e-4` | 0% | 0% | 0.905 | 0% | 9.11e-18 |

## Main finding

The SVD gate can preserve a numerically useful strong direction while deleting the weak relational contrast entirely. At `epsilon = 1e-3` and `tau = 1e-3`, the full three-channel hypothesis is preserved on 0% of paths; the strong direction is also not path-preserved, even though its worst-origin standardized coefficient retention has a median of about 90.4%. The weak direction is effectively removed, with a median worst-origin retention of about `7.1e-6`.

This strengthens the previous global-direction result because it is now evaluated as an **all-origins property**, matching the rolling estimator's actual unit of admissibility.

Therefore:

`rank improvement` ≠ `hypothesis preservation`

`strong-direction retention` ≠ `full RH-006 hypothesis preservation`

and, critically:

`high pooled/path-average retention` ≠ `all-required-origins preservation`.

At `epsilon = 1e-4`, even lowering `tau` to `1e-4` leaves the full three-channel hypothesis non-preserved on this construction. The weak contrast remains essentially discarded. Only the less severe `epsilon = 1e-2` construction retains the full three-channel space at `tau <= 1e-3`.

## Policy consequence

The repository should distinguish two non-interchangeable modes:

1. **Original full-space RH-006 estimand:** projection is forbidden unless an origin-local hypothesis-preservation certificate passes for the entire declared effect space.
2. **Projected estimand:** projection is permitted only when the result is explicitly labeled as a different estimand, with its retained subspace and origin schedule recorded.

For a directional confirmatory claim, the certificate should be evaluated for every declared direction and every required origin. A single failed origin is sufficient to block representation of the original directional hypothesis under that projected lane.

This prevents an adaptive SVD rescue from silently converting a test of the original relational channels into a test of whichever directions happen to survive the conditioning filter.

## Literature bridge

Post-selection and post-regularization inference literature treats selection/regularization as part of the inferential problem rather than a harmless preprocessing step. Chernozhukov, Hansen & Spindler develop conditions for valid inference on a low-dimensional target after regularization, while the broader post-selection literature emphasizes that the inferential target and selection mechanism must be specified explicitly. The present certificate is narrower: it is a deterministic semantic check that the declared RH-006 effect space has not been altered by projection.

## Reproducibility receipt

- script SHA-256: `f7f51828b6d2c63924ce40ce65409781de4873194709f7396dba7d9d9fb8230a`
- command: `python scripts/research/rh006_projection_preservation_certificate_splitmix64.py --mc 1000 --seed 20261022`
- seed: `20261022`
- MC paths per epsilon/tau cell: `1000`
- preservation tolerance: `1e-10`
- payload SHA-256 (canonical pre-field payload): `22609aa38493dfb0956759ccf2c9c0076cbba22529a7df85c144730a817d9bfd`

## Next boundary

The next gate is estimated-nuisance null calibration. For the projected lane, the bootstrap must replay the origin-local projection rule and treat the selected subspace as part of the statistical procedure. For the original RH-006 hypothesis, any bootstrap replicate that fails the full-space preservation certificate must fail closed rather than being silently projected.
