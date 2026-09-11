# LL-005 Seeded Ballistic Dispersion Oracle Evidence

## Purpose

This artifact provides an implementation-independent research oracle for propagating declared release-state uncertainty through the LL-003B spherical two-body surface-hop model.

It does **not** define real lunar launcher tolerances, acceptable impact probability, safe-release policy, or a qualified transport corridor.

## Model lineage

- Forward dynamics: `scripts/ll-spherical-ballistic-oracle.py` from LL-003B.
- Uncertainty oracle: `scripts/ll-ballistic-dispersion-oracle.py`.
- Random stream: explicitly specified SplitMix64 plus Box-Muller Gaussian transform.
- Endpoint errors: gnomonic projection into the nominal impact tangent plane.
- Site-position uncertainty: sampled as north/east tangent displacements and remapped onto the spherical body, avoiding a longitude singularity near the poles.

Every result records:

- seed;
- sample count;
- uncertainty magnitudes;
- dynamics model reference;
- constants reference;
- frame reference;
- counts of reimpact / escape / bounded no-return samples.

## Important physical boundary

Pod mass is intentionally **not** a ballistic uncertainty variable once the release state is fixed. In ideal central-gravity vacuum motion, inertial and gravitational mass cancel. Pod-mass uncertainty belongs upstream where it can influence achieved launcher exit state, structural loads, or terminal-correction performance.

Likewise, timing uncertainty is not yet modeled because the current spherical two-body proxy has no rotating/ephemeris target state. Timing becomes meaningful in LL-004/catcher-frame work.

## Executed self-test

The committed oracle self-test was executed before commit and passed.

It checks:

1. deterministic replay from identical seed/input;
2. zero-noise collapse to numerical integration precision;
3. larger speed uncertainty produces larger endpoint spread in a declared local-linear synthetic fixture;
4. azimuth uncertainty produces cross-track spread;
5. site north/east perturbations remain well-defined near the lunar-pole analogue;
6. identical release state is independent of external cargo-mass metadata;
7. one-source-at-a-time sensitivity output is finite and nonnegative.

## Synthetic reference result

For the synthetic LL-003B body:

- `R = 1,000,000 m`;
- `mu = 2e12 m^3/s^2`;
- nominal speed `100 m/s`;
- elevation `45 deg`;
- eastward azimuth;
- seed `42`;
- 512 samples;

with declared 1-sigma uncertainties:

- speed `0.1 m/s`;
- elevation `0.01 deg`;
- azimuth `0.01 deg`;
- site north `0.5 m`;
- site east `0.5 m`;

the independent execution produced approximately:

- 512 / 512 reimpacts;
- RMS radial endpoint miss `9.5959 m`;
- radial P50 `6.4762 m`;
- radial P95 `19.1889 m`;
- radial P99 `23.6319 m`;
- tangent-plane covariance `[[91.2300, -0.1939], [-0.1939, 1.0083]] m^2`.

For this **specific synthetic trajectory only**, one-source-at-a-time RMS endpoint misses were approximately:

- speed: `9.4613 m`;
- azimuth: `0.9215 m`;
- site east: `0.5499 m`;
- site north: `0.5220 m`;
- elevation: `0.00424 m`.

This ranking is not universal. In particular, the 45-degree short-hop fixture is close to a range extremum with respect to elevation, which is why its local elevation sensitivity is unusually small.

## Coupling target

The LL-005 distribution is intended to feed:

1. terrain-clearance distributions from #1572;
2. protected-zone intersection probability/bounds;
3. catcher intercept covariance;
4. safe-miss consequence distributions;
5. deterministic LL-008 admission inputs.

A Monte Carlo percentile by itself is never release authority.

## Remaining gaps

Before real lunar corridor analysis:

- cross-check the same seeded stream in the Rust production implementation;
- introduce measured launcher/navigation error models rather than assumed Gaussian sigmas;
- add provenance-bound LOLA topography and uncertainty;
- add GRAIL/high-fidelity gravity only where sensitivity requires it;
- add ephemeris/rotating-target timing uncertainty;
- model smart-pod terminal correction and catcher state uncertainty;
- define protected zones and acceptable-risk policy through separate governance/qualification evidence.
