# LQCD-020E — effective-potential / declared-plateau oracle

## Scope

Independent standard-library Python qualification subject for nonlinear Wilson-loop effective-potential analysis. It does not import Symthaea/Rust code and is not a physical static-potential result.

Stable analysis ID:

`wilson_effective_potential_declared_plateau_gls_v1`

## Semantics

The primary data are complete per-configuration Wilson-loop trajectories `W_i(T)`.

The effective potential is defined from the ratio of ensemble means:

`V_eff(T) = log(<W(T)> / <W(T+1)>)`.

It is deliberately **not** the mean of per-configuration log-ratios.

Delete-one-block jackknife replicates remove complete configuration trajectories across every `T`, recompute the nonlinear effective-potential vector, and yield the full correlated covariance matrix. A constant plateau value is then obtained by a generalized least-squares fit over a caller-declared `T` window.

No automatic plateau search exists in the oracle. Neighboring windows are reported only as diagnostics.

## Exact executed subject

- subject SHA-256: `f7875ce6521494b10b114f7dafa7927245691fc8c76057978fbfa6ce498bac67`
- stdout SHA-256: `a1886fff568ac4a253d4be541cd1fac5c253d7e85bd1086692824b4840a13c0d`
- canonical result SHA-256: `1203f4edc0ae4bbf376a4e0dc20b15328a59e576c4ce51e6e624ab946b9fddb3`
- configurations: `96`
- block size: `8`
- jackknife blocks: `12`
- injected ground-state potential: `0.43`

## Excited-state fixture

The deterministic fixture contains one ground contribution and one faster-decaying excited contribution plus correlated positive configuration-level fluctuations.

The frozen effective-potential sequence is:

`[0.4893301242, 0.4537091345, 0.4389356119, 0.4333102521, 0.4304787313, 0.4301188916, 0.4298681776]`.

### Declared late window

`T = 5..7`

- GLS estimate: `0.43001421487448777`
- GLS SE: `0.0014347332153113823`
- chi2/dof: `0.020678206772295253`

This recovers the injected ground-state value within the qualification tolerance.

### Deliberately contaminated early window

`T = 1..3`

- estimate: `0.4405888064355111`
- chi2/dof: `620.736563144277`

The early window therefore fails strongly rather than being accepted because its central value appears plausible.

### Neighboring-window diagnostics

- `T=4..6`: estimate `0.43225026107840564`, chi2/dof `4.2221366072067035`
- `T=4..7`: estimate `0.4336794877365569`, chi2/dof `3.1888392738231652`

These are diagnostic comparisons only. They do not authorize changing the declared `5..7` window after seeing the result.

## Estimator negative control

The maximum difference between the declared ratio-of-means estimator and the mean of per-configuration log-ratios is:

`0.00031536663268022513`.

The distinction is therefore executable rather than merely documented.

## Scientific boundary

This oracle qualifies analysis algebra and covariance semantics only. It does not establish that a real ensemble is equilibrated, that any plateau window is physically adequate, that APE parameters are optimal, or that a measured `V(R)` can be promoted to string tension or a continuum result.

The production layer must receive a window frozen before final analysis and should report neighboring-window sensitivity without selecting a new window automatically.
