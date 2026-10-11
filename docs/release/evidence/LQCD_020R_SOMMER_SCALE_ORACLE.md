# LQCD-020R — independent Sommer-type scale oracle

Exact executed standard-library subject SHA-256:

`c8186622a5b1f81e737b484bc1ad5429a30c4d448f08857ab2590fdbaddbb452`

Canonical result SHA-256:

`91a4d6711c09b02d898ca0dd73c9e6d7e6ecb335ba516bd95c0e7afa9fc4466c`

The oracle qualifies

`r_c = sqrt((c - e) / sigma)`

for valid `sigma > 0` and `c > e`, and propagates uncertainty from the correlated static-potential fit.

For a free-`e` fit it includes

`Var(r_c) = g_sigma^2 Var(sigma) + g_e^2 Var(e) + 2 g_sigma g_e Cov(sigma,e)`

with `g_sigma = -r_c/(2 sigma)` and `g_e = -1/(2 sigma r_c)`.

For a fixed-`e` fit only the `sigma` variance contributes. The analytic gradients are checked against central finite differences. A negative control proves that removing the nonzero `sigma-e` covariance changes the reported `r0` uncertainty.

The oracle freezes conventional targets `c=1.65` (`r0`), `c=4` (`r4`) and `c=6` (`r6`) for later Wilson-action benchmark work.

## Scientific boundary

This qualifies scale algebra and covariance propagation only. It does not choose a physical fit interval, establish a valid potential fit, reproduce the EHK benchmark, or set the lattice spacing.
