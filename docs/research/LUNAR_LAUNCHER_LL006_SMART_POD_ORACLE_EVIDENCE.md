# LL-006 Smart-Pod Correction Oracle Evidence

## Purpose

This artifact studies whether a reusable lunar freight pod with a **bounded** correction budget can reduce release-state dispersion in an ideal spherical two-body hop.

It is not a flight-guidance implementation, thruster design, catcher controller, or release authority.

## Model lineage

- Forward dynamics: independent LL-003B Python oracle.
- Correction event: fixed fraction of nominal flight time.
- Local control basis: radial, along-track, cross-track at correction epoch.
- Endpoint target: nominal unperturbed reimpact point.
- Endpoint error: target-centered tangent-plane miss.
- Controllability model: finite-difference 2x3 endpoint Jacobian.
- Correction solve: minimum-norm pseudoinverse `x = -J^T (J J^T)^-1 e`.
- Conditioning: singular or overly ill-conditioned `J J^T` fails closed.
- Budget: nominal guidance may consume only the non-reserved fraction of total declared delta-v.

## Executed self-test

The committed self-test was executed before commit and passed.

It verifies:

1. nominal release state requests approximately zero correction;
2. a small synthetic release error is materially reduced by an unsaturated correction;
3. applied delta-v never consumes the protected reserve;
4. an intentionally tiny budget saturates and leaves a measurable residual miss;
5. the tiny correction remains helpful rather than being reported as perfect;
6. protected reserve accounting remains explicit.

## Synthetic reference result

For the same synthetic LL-003B body used by the ballistic oracle:

- `R = 1,000,000 m`;
- `mu = 2e12 m^3/s^2`;
- nominal launch `100 m/s`, `45 deg`, eastward;
- release perturbation `+0.1 m/s` speed and `+0.01 deg` azimuth;
- correction at 50% of nominal flight time;
- total correction budget `1.0 m/s`;
- protected reserve `20%`;
- Jacobian probe impulse `0.001 m/s`;

the independent run produced approximately:

- pre-correction miss: `10.0933 m`;
- requested/applied correction: `0.20133 m/s`;
- protected reserve: `0.2 m/s`;
- remaining nominal correction budget: `0.59867 m/s`;
- Jacobian condition number: `2.003`;
- corrected residual miss: `0.01523 m`;
- improvement ratio: about `663x`;
- corrected outcome: reimpact.

This result is a local synthetic controllability fixture. It is **not** evidence that a real lunar pod needs only ~0.2 m/s terminal correction.

## Why the reserve is structural

A smart pod should not spend its entire correction capability pursuing nominal receiver accuracy. The model therefore partitions the declared total correction budget into:

- nominal targeting allocation;
- protected contingency reserve.

Nominal correction is capped before propagation. The protected fraction is not available to improve the nominal score.

## Remaining gaps

Before operational relevance:

- couple LL-006 to the full LL-005 seeded dispersion ensemble;
- add navigation/target-state uncertainty at correction time;
- add finite thruster impulse, deadband, minimum impulse bit, plume constraints, and actuator failures;
- add pod mass/propellant accounting through an explicit propulsion model;
- move correction timing into ephemeris/catcher-relative state rather than a nominal-flight fraction;
- evaluate receiver capture covariance, terrain, protected zones, and safe-miss consequences after correction;
- validate the local Jacobian method over the region where it is used and fail closed outside that region.

Successful synthetic correction is not launch or capture qualification.
