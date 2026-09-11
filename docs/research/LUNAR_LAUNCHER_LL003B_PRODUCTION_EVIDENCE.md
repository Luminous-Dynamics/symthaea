# LL-003B Production Forward Solver — Evidence Note

Status: research implementation; not launch qualification or release authority.

## Purpose

This tranche implements the first Rust spherical two-body forward propagator for `symthaea-lunar-launcher`, stacked on LL-001/002. It exists to provide an auditable production-side implementation that can be compared against the independent Python oracle in PR #1571.

## Model boundary

The solver models only:

- a spherical central body;
- caller-supplied gravitational parameter `mu` [m^3/s^2];
- caller-supplied reference radius [m];
- a non-rotating body-fixed surface proxy used to construct the launch local frame;
- 3-D point-mass central gravity;
- fixed-step RK4 propagation;
- the next intersection with the declared reference sphere.

It does **not** model topography, body rotation/libration, GRAIL harmonics, Earth/Sun perturbations, launcher rail dynamics, electromagnetic fields, payload attitude, drag/plumes, terminal correction, catcher behavior, trajectory dispersion, protected zones, safe-miss policy, or hardware release authority.

A `Reimpact` result is therefore only a mathematical surface intersection, not an operationally safe corridor.

## Provenance contract

No lunar constant is compiled into the model. Every run supplies:

- `mu_m3_s2`;
- `radius_m`;
- model identifier;
- frame identifier;
- constants/evidence reference.

This preserves synthetic fixtures, literature reproduction, and future DE440/GRAIL studies without silently invalidating previous evidence.

## Independent parity fixture

The independent Python oracle (`scripts/ll-spherical-ballistic-oracle.py`, PR #1571) was executed for the synthetic fixture:

- radius = `1_000_000 m`;
- surface gravity = `2 m/s^2`, hence `mu = 2e12 m^3/s^2`;
- speed = `100 m/s`;
- elevation = `45 deg`;
- azimuth = `90 deg`;
- launch latitude/longitude = `0/0 deg`;
- RK4 step = `0.05 s`.

Reference outputs:

- outcome: `reimpact`;
- flight time: `71.00636995976859 s`;
- ground range: `5012.520833156787 m`;
- maximum sampled altitude: `1254.7031614161097 m`;
- arrival speed: approximately `100 m/s`;
- arrival flight-path angle: approximately `-45 deg`;
- relative energy/angular-momentum drift: near machine precision in the oracle.

The Rust regression test uses these values as cross-implementation fixtures. They are synthetic numerical evidence, not launcher performance claims.

## Additional regression gates

The Rust module also encodes:

- short-arc convergence toward the LL-003A flat-gravity oracle;
- east/west equatorial symmetry;
- outward non-negative-energy escape classification;
- fixed-step refinement stability;
- fail-closed malformed input and metadata handling.

## Crate integration

LL-003B is isolated in `src/ballistics.rs`. A thin `src/root.rs` facade re-exports the existing LL-001/002 public API and exposes `pub mod ballistics`; LL-001/002 implementation remains unchanged.

## Promotion boundary

Before LL-003C inverse targeting:

1. GitHub CI must compile and run this crate;
2. Rust results must agree with the independent Python oracle on shared fixtures;
3. invariant drift and step-refinement gates must pass;
4. any discrepancy must be resolved without changing the oracle and production implementation in lockstep.

Before any safe corridor claim, the program must additionally satisfy #1572 terrain, uncertainty, protected-zone, and safe-miss gates.
