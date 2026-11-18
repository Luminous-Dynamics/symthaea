# symthaea-metric-geometry-lab

A std-only, auditable 4D differential-geometry reference kernel for analytic
spacetimes. The first version implements symmetric metric tensors, pivoted
matrix inversion, finite-difference metric derivatives, Levi-Civita
Christoffel symbols and RK4 geodesic integration.

## Implemented reference geometries

- Minkowski Cartesian coordinates, signature (-,+,+,+).
- Schwarzschild exterior in standard spherical coordinates, G=c=1,
  restricted to r>2M and 0<theta<pi.

## Verification targets included

- Metric/inverse product equals identity.
- Minkowski connection vanishes.
- Schwarzschild Christoffel components match their analytic expressions.
- Minkowski geodesics are straight lines.
- Schwarzschild radial timelike geodesics preserve g(u,u)=-1 to a stated
  numerical tolerance.
- Singular metrics and invalid coordinate-chart points fail closed.

Run:

    cargo test -p symthaea-metric-geometry-lab
    cargo run -p symthaea-metric-geometry-lab --example schwarzschild_radial_geodesic

## Boundaries

This is not a numerical-relativity code. It does not solve Einstein's field
equations, ADM Hamiltonian or momentum constraints, matter evolution,
gravitational waves, black-hole initial data, or a source problem for an
engineered metric. It is a verified reference baseline for local metric,
connection and geodesic calculations.

The next steps are curvature tensors, exact-vs-finite-difference convergence,
coordinate transformations, weak-field comparisons and independent solver
checks. Only after those should the kernel be considered as a component in a
revised GravCraft simulator.
