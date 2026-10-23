# LQCD-020G — tree-level lattice Coulomb kernel oracle

## Scope

Independent standard-library Python qualification subject for the tree-level three-dimensional Wilson-action lattice Coulomb term used in coarse-lattice static-potential fits.

Stable convention ID:

`tree_level_wilson_lattice_coulomb_midpoint_richardson_v1`

The kernel is

`[1/R]_lat = 4π ∫_BZ d^3k/(2π)^3 cos(k·R) / (4 Σ_j sin²(k_j/2))`.

This is a short-distance basis function for later potential fitting, not a measured string tension or physical scale.

## Numerical method

The Brillouin-zone integral is evaluated on an even midpoint grid, which avoids sampling the integrable `k=0` singularity directly. Because the midpoint error converges slowly, the oracle uses first-order Richardson extrapolation from `N` and `2N`.

Primary values use `N=64/128`. An independent convergence check uses `N=80/160`.

## Exact executed subject

- subject SHA-256: `15362b906f488eab26e10c4cc8a97402ff277f3431f68d9c73bd155e3341f9f9`
- stdout SHA-256: `a4fc06b04cb8f0dae1936f78e5eee8a44fae91b827e0eceb4b601166ad27be73`
- canonical result SHA-256: `429c13c0f308bd21659cc4c7bc5212a770b9f99e706e71557e339984994b3fef`
- max disagreement between the two independently extrapolated resolution pairs: `2.2303630382580764e-06`.

## Frozen primary values

- `R=(1,0,0)`: `1.081520691702028`
- `R=(2,0,0)`: `0.5389673179754996`
- `R=(3,0,0)`: `0.3461467982875027`
- `R=(1,1,0)`: `0.6935602595349944`
- `R=(1,1,1)`: `0.5476259824378233`
- `R=(2,1,0)`: `0.4515341044657458`

The nearest-neighbor value visibly differs from continuum `1/R=1`, demonstrating why a coarse-lattice fit should not silently substitute the continuum Coulomb basis at short separation.

## Scientific boundary

This oracle establishes the numerical definition and convergence behavior of one tree-level lattice Coulomb basis function only. It does not establish the correct static-potential fit range, the adequacy of a Cornell ansatz, rotational-symmetry control, a string tension, Sommer scale, finite-volume adequacy, or a continuum result.

Later potential fits must freeze the included separation vectors and fit model before final analysis, use the full covariance of the measured `V(R)`, and report sensitivity to alternative fit ranges rather than choosing the range automatically.
