# LQCD-020H — correlated declared-range lattice-Cornell fit oracle

## Scope

Independent standard-library Python qualification subject for a covariance-aware static-potential fit using the tree-level Wilson lattice Coulomb basis frozen by LQCD-020G.

Stable analysis ID:

`correlated_declared_range_lattice_cornell_gls_v1`

The declared model is

`V(R) = V0 + sigma |R| - e [1/R]_lat`.

The separation-vector set is frozen before fitting; there is no automatic `R_min`, `R_max`, orientation, or model selection.

## Exact executed subject

- subject SHA-256: `37e5049c62d36c6d7505541d49549620a2ed1649512ef2120afe41a96e59ff76`
- stdout SHA-256: `75502eb325728d7e36b54e832454e825c38358587ecb6c6f79fb9914b39a8e7c`
- canonical result SHA-256: `9e150c0a33e887c87e13ff7e263bfb76717242c48e9b93b9f1078f6352ffa4a5`

Injected fixture parameters:
- `V0 = 0.7`
- `sigma = 0.18`
- `e = 0.25`

Declared separation vectors:
`(1,0,0)`, `(2,0,0)`, `(3,0,0)`, `(1,1,0)`, `(1,1,1)`, `(2,1,0)`.

## Lattice-Coulomb fit

Fitted parameters:
- `V0 = 0.6909244500871523`
- `sigma = 0.1824571461084421`
- `e = 0.24273022628559993`

Fit quality:
- `chi2 = 0.5599351027496333`
- `dof = 3`
- `chi2/dof = 0.18664503424987777`

The fitted string-tension coefficient recovers the injected `0.18` within the qualification tolerance.

## Continuum-Coulomb negative control

Using the same correlated observations and covariance but replacing `[1/R]_lat` with continuum `1/|R|` gives:

- `V0 = 0.738233980114213`
- `sigma = 0.17040103440875498`
- `e = 0.2915974969982216`
- `chi2/dof = 6.242206162235313`

The parameters still look superficially plausible, but the correlated goodness-of-fit exposes the wrong short-distance basis.

## Scientific boundary

This oracle qualifies model algebra and covariance semantics only. It does not establish that the Cornell/lattice-Coulomb model is adequate for a real dataset, that the declared separation range is physically appropriate, that rotational artifacts are controlled, or that a fitted `sigma` is a physical string tension.

A production analysis must bind the exact measured `V(R)` covariance, exact separation vectors, lattice-Coulomb convention, fit model, and declared range before final fitting. Alternative ranges/models should be sensitivity evidence, not automatic replacements chosen after seeing chi-square.
