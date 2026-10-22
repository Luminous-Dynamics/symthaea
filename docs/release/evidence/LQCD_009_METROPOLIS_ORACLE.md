# LQCD-009 — Independent Metropolis-Hastings oracle evidence

Date: 2026-09-11

## Scope

This evidence note records execution of `scripts/lqcd-metropolis-oracle.py --self-test`.
The script imports only the Python standard library and no Symthaea code.

It is a compact periodic-angle / U(1)-like Metropolis-Hastings theorem fixture.
It is **not** an SU(3) lattice-QCD sampler and is not evidence of equilibrium QCD physics.

## Executed fixture

Parameters:

- `beta = 1.7`
- proposal half-width `0.8`
- deterministic LCG seed `0xC0FFEE`
- replay length `64` updates
- exact finite-state stationarity check on a 16-state periodic angular ring

Observed self-test output:

```text
status = ok
accepted_moves = 53
final_theta = -0.9955904309881722
max_pairwise_detailed_balance_residual = 4.440892098500626e-16
max_transition_row_sum_residual = 0.0
stationarity_residual = 1.3877787807814457e-17
```

The test also verifies:

- wrapped proposal symmetry across the `-pi/pi` branch cut;
- Metropolis acceptance is bounded in `[0,1]`;
- downhill acceptance equals `exp(delta log target)`;
- uphill acceptance saturates at `1`;
- exact deterministic trajectory replay for the same seed;
- the finite-state transition matrix preserves the declared target distribution within floating-point tolerance.

## Authority boundary

This establishes only reference semantics for a symmetric-proposal Metropolis-Hastings kernel.
It does **not** establish:

- ergodicity of any future SU(3) proposal;
- correctness of Cabibbo-Marinari / heat-bath / HMC updates;
- gauge-field ensemble generation;
- thermalization;
- autocorrelation control;
- continuum or finite-volume control;
- string tension, deconfinement, glueball masses, or any physical QCD prediction.

Any production SU(3) sampler must independently demonstrate proposal-group validity, detailed balance (or the appropriate HMC theorem), deterministic qualification fixtures, and parity with separate reference calculations.
