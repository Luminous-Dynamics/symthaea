# LQCD-020S — independent joint Sommer-scale jackknife oracle

Authority: executed standard-library Python numerical oracle. No Symthaea/Rust code is imported.

Exact executed subject SHA-256:

`ba2453a29ae1fb8e3e614ac51ef934b8fbb4973892f08214318efefd6c7df06f`

Canonical result SHA-256:

`f6c7470dd54ad15393791339605a0fa434496d335731337697a023320f2acd9f`

## Qualified algebra

For a fitted continuum potential `V(r) = V0 + sigma*r - e/r`, the generic Sommer-type scale is

`r_c = sqrt((c - e) / sigma)`

when `sigma > 0` and `c > e`.

The oracle evaluates the standard target set

- `r0`: `c = 1.65`
- `r4`: `c = 4`
- `r6`: `c = 6`

from one central `(sigma,e)` pair and from every member of the same delete-one jackknife parameter-replicate set. It then computes the full joint jackknife covariance across `(r0,r4,r6)` rather than propagating three independent errors.

Frozen central scales for the synthetic fixture `(sigma,e)=(0.18,0.25)`:

- `r0 = 2.788866755113585`
- `r4 = 4.564354645876384`
- `r6 = 5.65194165260439`

Frozen jackknife standard errors:

- `0.05308828240486231`
- `0.08142441695758301`
- `0.10017345414229001`

Frozen covariance matrix:

`[[0.002818365728698413,0.004203509237759222,0.005100095468994386],[0.004203509237759222,0.006629935676882331,0.008145573852299368],[0.005100095468994386,0.008145573852299368,0.01003472091479748]]`

The subject also fails closed for `sigma <= 0`, non-finite inputs, and `c <= e`.

## Scientific boundary

This qualifies the nonlinear scale transformation and joint resampling semantics only. It does not establish a physical `r0/a`, `r4/a`, or `r6/a`; the input fit parameters must come from a promoted equilibrium ensemble and a preregistered static-potential fit. A real campaign should preserve the same fit replicate across all scale targets so their covariance survives benchmark comparison.
