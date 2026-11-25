# Positive mass-constrained kinetic transition projection v0.1

This benchmark isolates one feasible correction problem that appears during spectral truncation, zero-padding transfer, basis changes, and low-rank reconstruction.

## Mathematical problem

Given discrete input values f_i, positive cell volumes v_i, and a target discrete mass M, compute

`min_g 0.5 * sum_i v_i * (g_i - f_i)^2`

subject to

`g_i >= 0` and `sum_i v_i * g_i = M`.

The exact solution has the water-filling / active-set form

`g_i = max(0, f_i - lambda)`

where lambda is selected so that the weighted discrete mass equals M. The Rust function `project_nonnegative_preserving_mass` computes this active-set solution and reports the multiplier, active-cell count, achieved mass, mass residual, minimum output value and weighted correction norm.

## Known-answer witness

The fixture starts with input values [-0.1, 1.1], equal cell volumes [1,1] and target mass 1. The correction is [0,1], with lambda=0.1, zero mass residual and weighted correction norm sqrt(0.02).

The executable audit is:

`cargo run --bin positive_mass_projection_audit -- benchmarks/plasma/positivity-projection/fixtures/negative-spectral-overshoot-known-answer-v0.1.json`

The CLI verifies every projected cell and the reported mass, multiplier, active-set count and correction norm against the fixture's expected values; it exits non-zero on mismatch. The focused Plasma Probe Contract workflow validates the Draft 2020-12 schema, runs the known-answer audit, and checks the same case in the Rust unit tests.

## Strict scope boundary

This primitive preserves **positivity and one scalar discrete mass only**. It does not preserve momentum, energy, L1, L2, entropy, all Vlasov Casimirs, Poisson residual, or future responses. The transition is not physically qualified merely because this projection succeeds.

The general IRCKP programme is [#13629](https://github.com/Luminous-Dynamics/symthaea/issues/13629). Its next constrained problems must:
- add declared moment constraints only where the selected discrete reference defines them;
- handle infeasible low-rank constraints by explicit refusal or rank increase;
- use a separate collisional entropy/energy-exchange contract for VPFP/VPL;
- check future-response loss after correction, not assume the smallest correction is response-preserving;
- account for QP/projection cost, active-set work, state storage and fallback.

This is a synthetic numerical primitive, not a Vlasov/VPFP simulation, a new positivity algorithm claim, or a physical result.
