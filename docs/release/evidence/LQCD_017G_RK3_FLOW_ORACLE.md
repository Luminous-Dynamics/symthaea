# LQCD-017G independent Lie-group RK3 Wilson-flow qualification

## Subject

`scripts/lqcd-topology-flow-rk3-oracle.py`

Exact executed-subject SHA-256:

`89b69d11f30e8a86a4fb5962a07430d7e2b15a65a2c0afb8beea8c6922d7777e`

The subject is Python-standard-library only and imports no Symthaea/Rust implementation code.

## Integrator

The subject uses the standard Lie-group third-order gradient-flow Runge-Kutta composition:

- `W0 = V(t)`;
- `W1 = exp((1/4) Z0) W0`;
- `W2 = exp((8/9) Z1 - (17/36) Z0) W1`;
- `V(t+eps) = exp((3/4) Z2 - (8/9) Z1 + (17/36) Z0) W2`;
- `Zi = -eps * grad S_W(Wi)` in the conventions frozen by the LQCD-017D Wilson-action gradient oracle.

The coefficients are the standard third-order lattice gradient-flow RK scheme used in the Wilson/gradient-flow literature.

## Independent fixture

The oracle uses a compact nontrivial periodic `2^4` pure-SU(3) field defined by eight explicit Gell-Mann link rotations. No RNG is involved. The analytic Wilson-action gradient is computed from the direct six-staple expression already independently parity-checked against finite differences in LQCD-017D.

The fixture is intentionally independent of the 40-operation topology fixture so the RK3 order test is not simply another evaluation of the same frozen field.

## Fixed-flow-time step halving

All runs terminate at the same total flow time `t=0.008`:

- `dt=0.004`, 2 steps;
- `dt=0.002`, 4 steps;
- `dt=0.001`, 8 steps;
- `dt=0.0005`, 16 steps.

Executed stdout is frozen in `LQCD_017G_RK3_FLOW_RESULT.txt`.

Key results:

| dt | steps | Wilson action | clover Q |
| ---: | ---: | ---: | ---: |
| 0.004 | 2 | 2.5114345910097962 | 1.0288904490680599e-05 |
| 0.002 | 4 | 2.5114347854282113 | 1.0288905158564748e-05 |
| 0.001 | 8 | 2.5114348093973327 | 1.0288905241074472e-05 |
| 0.0005 | 16 | 2.5114348123728787 | 1.0288905251328004e-05 |

Successive fixed-time differences give:

- action ratios: `8.1112032548170756`, `8.0553690308572676`;
- clover-Q ratios: `8.0946114233847659`, `8.0469562174510632`.

Halving the step therefore reduces the global error by approximately `2^3`, which is the expected third-order convergence signature.

Every frozen run is Wilson-action monotone on this fixture, and the worst observed determinant/unitarity drift remains below `5e-15`.

## Authority boundary

This establishes the integration order and Lie-group preservation of the RK3 composition on the frozen nontrivial fixture. It does **not** establish:

- a production `dt`;
- a physical flow scale `t0` or `w0`;
- topology sampling or tunnelling;
- continuum topological susceptibility;
- that action monotonicity is guaranteed for arbitrary finite step size;
- a performance advantage over Lie-Euler.

The production Rust RK3 implementation must independently reproduce the stage semantics and the frozen fixed-time values before it can replace Lie-Euler for production topology measurements. The first-order Lie-Euler oracle remains retained as a simpler semantic and regression reference.
