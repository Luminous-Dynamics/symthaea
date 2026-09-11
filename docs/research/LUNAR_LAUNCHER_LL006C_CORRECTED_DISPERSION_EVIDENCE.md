# LL-006C Corrected-Dispersion Ensemble Evidence

## Purpose

This artifact measures how a bounded smart-pod correction policy changes a seeded LL-005 release-state dispersion cloud. It is intentionally a **second correction implementation** rather than an import of the LL-006B single-case oracle, providing some cross-check value.

It is not flight guidance or a real lunar pod performance claim.

## Scope

The current v0 ensemble:

- uses the LL-005 SplitMix64 release-state sample stream;
- perturbs speed, elevation, and azimuth;
- preserves the LL-005 random-stream shape;
- independently estimates a local 2x3 endpoint Jacobian at the correction epoch;
- solves a minimum-norm radial/along-track/cross-track impulse;
- caps nominal correction to the non-reserved delta-v budget;
- repropagates every corrected sample;
- reports pre/post RMS and P95 miss, applied delta-v statistics, saturation fraction, and correction failures.

Launch-site north/east uncertainty is deliberately rejected in v0 because the LL-006 correction state does not yet carry a displaced launch site. It must not be silently ignored.

## Executed self-test

The committed self-test was executed before commit and passed.

For the synthetic fixture:

- 128 seeded samples, seed `42`;
- speed sigma `0.1 m/s`;
- elevation sigma `0.01 deg`;
- azimuth sigma `0.01 deg`;
- correction at 50% of nominal flight;
- total correction budget `0.5 m/s`;
- protected reserve `20%` (`0.1 m/s`);

independent execution produced approximately:

- pre-correction RMS miss: `10.0479 m`;
- post-correction RMS miss: `1.0484 m`;
- pre-correction P95 miss: `18.5909 m`;
- post-correction P95 miss: `0.0521 m`;
- mean applied correction: `0.1608 m/s`;
- P95 applied correction: `0.3725 m/s`;
- saturation fraction: `0.046875` (6/128);
- correction failures: `0`.

The RMS tail is dominated by the small saturated subset, illustrating why a single corrected trajectory can substantially overstate ensemble performance.

## Acceptance behavior

The self-test also verifies:

- exact seeded replay;
- protected reserve is not consumed by the P95 nominal correction;
- a deliberately tiny correction budget produces nonzero saturation;
- even the tiny budget remains directionally helpful in the synthetic fixture;
- zero release-state uncertainty collapses both pre/post dispersion to numerical precision.

## Remaining gaps

Before operational relevance:

- add displaced launch-site state to the correction model;
- couple navigation/target-state uncertainty at correction time;
- include minimum impulse bit, deadband, thrust-axis errors, actuator faults, plume constraints, and finite burns;
- attach a pod propulsion/propellant/energy model;
- propagate residual dispersion into terrain, protected zones, catcher covariance, and LL-008 admission;
- replace assumed Gaussian release-state errors with measured launcher/navigation distributions;
- evaluate ephemeris/rotating-target timing uncertainty.

The current numbers are synthetic research fixtures only.
