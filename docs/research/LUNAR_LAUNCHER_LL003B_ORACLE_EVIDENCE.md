# LL-003B Spherical Two-Body Ballistic Oracle — Evidence Boundary

Status: independent dynamics oracle for issue #1567.

## Purpose

`scripts/ll-spherical-ballistic-oracle.py` provides a zero-dependency, caller-parameterized spherical two-body reference for lunar-style surface ballistic arcs.

It is intentionally independent of `symthaea-lunar-launcher`, `symthaea-orbital`, and future production trajectory code.

## Model

The oracle:

- accepts central-body `GM` and reference radius explicitly;
- constructs a 3-D launch state from latitude, longitude, azimuth, elevation, and speed;
- integrates `r'' = -mu r / |r|^3` with fixed-step RK4;
- root-refines the next reference-sphere intersection;
- reports flight time, great-circle ground range, peak radial altitude, arrival speed, arrival flight-path angle, arrival coordinates, and numerical invariant drift;
- classifies outward non-negative-energy trajectories as escape rather than forcing a landing;
- returns `no-return-within-window` when a bounded propagation window expires.

No lunar GM or radius is embedded in the solver.

## Independent execution checks

A separate implementation/execution of the same published dynamics contract was used to verify the test tolerances before committing the oracle.

For a normalized body with `mu=1`, `R=1`:

- a short `v=0.05`, 45-degree arc differs from the flat constant-g range by about 0.125%, providing the intended short-range asymptotic check;
- relative specific-energy drift was approximately `1.2e-15`;
- relative angular-momentum drift was approximately `5.6e-17`;
- east/west equatorial arcs produced identical time/range with reflected arrival longitude;
- an outward launch above local escape speed was classified as escape;
- halving the step for a representative suborbital arc changed range/time by far less than the declared `1e-5` refinement tolerance.

These are synthetic numerical-verification fixtures, not proposed lunar transport parameters.

## What this does not model

- lunar topography or local slope;
- nonspherical gravity / mascons;
- Moon rotation or libration;
- Earth/Sun third-body forces;
- ephemeris-frame transformations;
- launch dispersion or covariance;
- receiver/catcher dynamics;
- terminal correction;
- corridor occupancy or protected sites;
- launcher electromagnetic physics;
- safe-release authority.

Therefore a trajectory produced here must never be labeled an operationally safe lunar corridor.

## Promotion gate

Before LL-003C inverse targeting:

1. the oracle must remain independently reproducible;
2. a production/second implementation must agree on shared fixtures;
3. two-body energy and angular momentum conservation must remain within declared tolerance;
4. short arcs must converge toward LL-003A;
5. terrain/protected-zone analysis remains a separate prerequisite for any safety claim.

Tracks #1542, #1563, and #1567.
