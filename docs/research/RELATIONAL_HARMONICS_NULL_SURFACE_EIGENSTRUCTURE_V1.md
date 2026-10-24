# RH-006 Null-Surface Eigenstructure and Curvature — v1

## Status

Research diagnostic only. Not approved for formal inference.

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

This artifact is an independent Python mechanics mirror. It is not a validation of the Symthaea Rust implementation and does not constitute empirical validation.

## Why the directional grid was not enough

The RH-006 finite-sample equal-risk null is the quadratic surface

`Delta(gamma) = gamma' Q gamma + b' gamma + c = 0`.

A fixed coefficient-direction grid samples rays through this surface, but it does not expose the geometry that determines whether those rays intersect the surface.

Parameterize a ray as

`gamma = t u`, with `||u|| = 1` and `t > 0`.

Then:

`A(u) = u' Q u`

`B(u) = b' u`

`C = c`

and the ray discriminant is

`D(u) = B(u)^2 - 4 A(u) C = u' R u`

with

`R = b b' - 4 c Q`.

Thus RH-006 has two coupled spectral geometries:

- `Q` controls directional curvature of the equal-risk quadratic;
- `R` controls the root-existence/discriminant geometry of rays.

This makes root support itself a geometric property rather than a nuisance-calibration detail.

## Frozen direction design

The diagnostic samples:

- the three principal axes of `Q`, with both antipodes;
- fixed 5°, 15°, 30°, 45°, 60°, 75°, 85°, and 89° sweeps in every `Q) eigenplane, with both signs;
- the three principal axes of `R`, with both antipodes;
- the predeclared weak direction `[0,1,-2]/sqrt(5)`, with both antipodes.

The solver retains both positive roots when a ray has two. Near/far labels are ordering labels only; no branch is declared scientifically preferred.

## Surface differential geometry

At a regular surface point `x`:

`g(x) = 2 Q x + b`

and the tangent projector is

`P = I - g g' / ||g||^2`.

The diagnostic obtains local principal-curvature magnitudes from the projected Hessian `2Q`, normalized by `||g||`.

Therefore `||g|| -> 0` is treated as a critical/weak-identification condition. It is not averaged into ordinary surface behavior.

## Severe weak-direction execution

Configuration:

- epsilon = `1e-4`
- ridge = `1e-4`
- 1,000 feature paths
- deterministic SplitMix64 seed = `20261028`

Results:

| Diagnostic | Result |
|---|---:|
| Weak-direction root support | **42.1%** |
| Weak-direction failures from negative discriminant | **578 / 1000** |
| Linear/no-positive residual failures | **1 / 1000** |
| Eigenstructure-grid support-rate range across paths | **58.1% to 100%** |
| Q second-eigenvalue / max-absolute-eigenvalue median | ~`8.38e-11` |
| R second-eigenvalue / max-absolute-eigenvalue median | ~`8.38e-11` |
| max observed ||bb'|| / ||-4cQ|| | ~`4.14e-12` |
| R vs -4cQ Frobenius alignment | ~`1.0` |
| Weak-direction median positive root magnitude | **~8,895** |
| Weak-direction minimum positive root in this run | **~3,643** |
| Weak-direction maximum positive root in this run | **~1.62e9** |

The root support is therefore much more fragile than a coarse isotropic direction grid suggests.

Most importantly, in this severe cell `b b'` is numerically negligible relative to `-4 c Q`. Consequently the root-existence matrix `R` is almost entirely inherited from `Q`. The no-root cone is therefore fundamentally tied to the weak/indefinite curvature geometry.

## The epsilon ladder exposes coefficient-space blow-up

At the same ridge (`1e-4`), the weak-direction positive-root medians were approximately:

| epsilon | Median weak-direction root |
|---:|---:|
| `1e-2` | ~14.24 |
| `1e-3` | ~110.68 |
| `1e-4` | ~8,895 |

The transition is not a small perturbation. The represented null point moves rapidly outward as the weak direction loses effective curvature.

This creates a distinct scientific boundary:

> A finite-sample null point can exist while being far outside a scientifically local parameter neighborhood.

A formal procedure cannot silently treat such remote roots as interchangeable with local alternatives. A predeclared coefficient-radius or, preferably, an effect/risk-radius policy would be a separate estimand decision, not a numerical convenience.

## Stratified estimated-nuisance bootstrap

A second diagnostic then evaluated plug-in nuisance calibration over eigenstructure-selected roots using:

- `Q) principal axes ±;
- `R) principal axes ±;
- weak direction ±;
- curvature bands based on `|u'Q u| / lambda_max(|Q|)`;
- root-radius bands: local (`<=1`), moderate (`1..100`), remote (`>100`);
- surface criticality based on normalized gradient magnitude;
- bootstrap sizes up to 49.

The completed 60-path × 49-bootstrap severe-cell run produced two populated strata:

| Stratum | Points | Est. root support | Median root ratio | Oracle rejection | Plug-in rejection |
|---|---:|---:|---:|---:|---:|
| high curvature / local / regular / A>0 | 240 | 100% | 0.967 | 3.33% | 3.33% |
| near-flat / remote / critical / A>0 | 206 | 100% | 0.966 | 1.94% | 2.91% |

A separate 30-path × 39-bootstrap comparison at epsilon `1e-3` yielded:

- high-curvature/local/regular: oracle 3.33%, plug-in 3.33%;
- near-flat/remote/critical: oracle 4.42%, plug-in 3.40%.

These results do **not** establish valid size. They do establish that the mechanically difficult region can now be isolated and measured rather than hidden inside a single aggregate rejection fraction.

The severe 60×49 run does not show catastrophic nuisance-root loss: every supported null point retained an estimated positive root. The remaining issue is the geometry and locality of the null itself.

## New gate: null locality

RH-006 should now distinguish:

`surface support`

→ `curvature/identification state`

→ `surface criticality`

→ `root radius / effect radius`

→ `nuisance calibration`

→ `uniform calibration`.

A root-support pass is not enough.

For any future formal lane, a locality rule must be frozen before seeing confirmatory outcomes. The rule must be expressed in the scientific estimand's parameter or risk/effect coordinates rather than as an arbitrary numerical safeguard.

## What this does not solve

The following remain unresolved:

- a mathematically valid least-favorable distribution or uniform critical value;
- uniform size under weak identification;
- a validated estimator/procedure bridge for the exact Rust implementation;
- empirical validation on independently observed targets;
- whether the scientifically intended RH-006 estimand is global equal accuracy over the whole surface or a local equal-risk neighborhood.

The finite grid remains diagnostic.

## Reproducibility

Primary geometry script:

`scripts/research/rh006_null_surface_eigenstructure_splitmix64.py`

Committed repository Git blob:

`ea67f2ee93b53fea8615d583008436d021927fe2`

Local execution SHA-256:

`7a01bd2e866d767fe6a46527a7b1ad4656454bc7d22da99670225595bc24e5d5`

The execution record preserves the distinction between repository Git identity and local execution SHA-256; no byte-identity claim is required for the scientific conclusions.

Stratified nuisance script:

`scripts/research/rh006_null_surface_eigenstructure_nuisance_splitmix64.py`

Committed repository Git blob:

`7263e002a43578318e50a12a81197c8488fd1245`

Local execution SHA-256:

`577cbc630ba5c34d50b84f8468757c1cc4389c3d2911173963186a43724696eb`

Deep severe-cell execution:

`--paths 60 --bootstrap 49 --epsilons 0.0001`

Payload SHA-256:

`2af47f8d33ca01ab12aa191e6d8c322e4c4f8a77592bf5c5ba87073e24ecbf62`

## Next boundary

The next hardening step should be a **surface chart with explicit local/effect-radius coordinates** rather than another larger arbitrary direction grid.

That chart should separately map:

1. the `Q)-curvature cone;
2. the `R)-discriminant/root-existence cone;
3. the surface critical set `||2Qx+b|| ≈ 0`;
4. root/effect radius;
5. nuisance-estimation error conditional on each stratum.

Only after that should a candidate robust calibration rule be compared across strata.

