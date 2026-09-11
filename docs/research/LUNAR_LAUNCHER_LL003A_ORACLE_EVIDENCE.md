# LL-003A Surface Ballistic Oracle — Evidence Boundary

Status: independent analytic validation aid for issue #1567.

## Purpose

`scripts/ll-surface-ballistic-oracle.py` implements only the equal-elevation,
flat-surface, constant-gravity projectile equations used as LL-003A's short-range
asymptotic oracle.

It is deliberately independent of `symthaea-lunar-launcher`, orbital code, and
future spherical-Moon propagation.

## Equations

For caller-supplied launch speed `v`, elevation `theta`, and constant gravity `g`:

- range: `R = v^2 sin(2 theta) / g`;
- flight time: `t = 2 v sin(theta) / g`;
- maximum height: `h = v^2 sin^2(theta) / (2 g)`;
- equal-elevation arrival velocity: horizontal component preserved, vertical
  component sign-reversed.

No lunar constant is embedded in the script.

## Self-test fixtures

The script contains synthetic fixtures checking:

1. the 45-degree range identity;
2. complementary-angle equal range;
3. longer flight/higher apogee for the higher complementary angle;
4. quadratic range scaling with speed;
5. inverse range scaling with gravity;
6. fail-closed handling of non-finite/non-positive and invalid-angle inputs.

A separate arithmetic check reproduced the principal fixture values:

- `v=100 m/s`, `theta=45 deg`, `g=2 m/s^2` -> `R=5000 m`,
  `t=70.71067811865474 s`, `h≈1250 m`, arrival speed `100 m/s`;
- 30/60-degree complementary fixtures agree in range to floating-point roundoff;
- doubling speed produces 4x range;
- halving gravity produces 2x range.

These values are synthetic oracle arithmetic, not proposed lunar launcher
parameters.

## Explicit non-claims

This oracle does **not** model:

- lunar curvature;
- varying gravity;
- topography or local slope;
- Moon rotation/libration;
- Earth/Sun perturbations;
- launch/catcher geometry;
- uncertainty or launch dispersion;
- terminal correction;
- payload aerodynamics (irrelevant in lunar vacuum but still outside this model);
- electromagnetic launcher physics;
- safe-miss corridors;
- operational launch approval.

The model must not be used to label any lunar route safe or feasible.

## Promotion gate

LL-003B should not infer regional lunar performance from this oracle. Its role is
to provide a short-range limit check for a separately implemented spherical
two-body Moon propagator. A future spherical solver must conserve two-body
energy/angular momentum to declared tolerance and agree with an independent
implementation before inverse targeting is added.

Tracks #1542, #1563, and #1567.
