# Metric geometry lab — independent reference results v0.1

Date: 2026-11-07
Status: independent Python differential-geometry reference; Rust build and tests pending.

## Reference scope

The new std-only Rust crate implements 4D symmetric metric tensors, matrix
inversion, central-difference metric derivatives, Levi-Civita Christoffel
symbols and RK4 affine geodesics. Its first analytic metrics are:

- Minkowski Cartesian coordinates, signature (-,+,+,+).
- Schwarzschild exterior in standard spherical coordinates (t,r,theta,phi),
  with G=c=1, restricted to r>2M and 0<theta<pi.

This report checks representative outputs with an independently written
Python implementation using NumPy matrix inversion and a separate
Christoffel / geodesic routine. It does not run the Rust code.

## Schwarzschild Christoffel reference

At M=1, r=10, theta=pi/2, f=1-2M/r=0.8, the following analytic connection
components are:

- Gamma^t_tr = M/(r^2 f) = 0.0125
- Gamma^r_tt = f M/r^2 = 0.008
- Gamma^r_rr = -M/(r^2 f) = -0.0125
- Gamma^r_theta-theta = -r f = -8
- Gamma^theta_r-theta = 1/r = 0.1

The independent centered-difference implementation returned a maximum
absolute error of 1.87e-11 across these five distinct components and their
lower-index symmetry counterparts.

## Geodesic reference

Initial conditions:
- M=1, r=10, theta=pi/2, phi=0, t=0
- dr/dlambda=-0.05
- dt/dlambda=sqrt((1 + (dr/dlambda)^2/f)/f), chosen so g(u,u)=-1
- 1,000 RK4 steps of delta-lambda=0.001

The independent implementation obtained:

| Quantity | Initial | Final |
|---|---:|---:|
| g_mu_nu u^mu u^nu | -1.000000000000 | -1.000000000000147 |
| Radius | 10.000000000000 | 9.944982427699 |
| Coordinate time | 0 | 1.120529585266 |

The normalization drift was approximately -1.47e-13 for this particular
smooth exterior trajectory and step size. This is a reference point, not a
general error guarantee; convergence must be checked by decreasing the
integration step and comparing trajectory observables.

## What this does and does not establish

It establishes that the independent finite-difference reference equations
produce the expected Schwarzschild connection values and maintain the
timelike normalization for this short radial trajectory.

It does not establish that:
- The Rust implementation compiles or passes its unit tests.
- The metric is sourced by a physically realizable stress-energy tensor.
- The Einstein field equations or ADM constraints are being solved.
- The current GravCraft metric ansatz is consistent or physically realizable.
- A warp drive or other engineered spacetime can be produced.

## Next gates

1. Run cargo fmt and cargo test for symthaea-metric-geometry-lab.
2. Cross-check Minkowski and Schwarzschild results against an independent
   symbolic/algebra system and a specialist numerical-relativity library.
3. Add coordinate-transformation covariance tests and a metric signature gate.
4. Add Riemann, Ricci and curvature-invariant checks; Schwarzschild exterior
   should be vacuum with Kretschmann scalar 48 M^2/r^6.
5. Verify geodesic energy and angular-momentum conservation, normalization
   convergence and gauge/coordinate sensitivity.
6. Only then integrate the kernel into a revised GravCraft simulator.

The current crate is a local analytic reference kernel, not a GR evolution
code. It is intentionally isolated from the speculative propulsion model.
