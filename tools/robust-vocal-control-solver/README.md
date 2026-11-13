# Robust vocal control solver (VRCO-001)

A dependency-free Rust solver for bounded, regularized inverse control using interval-valued acoustic response matrices.

## What it does

Given a measured local response model and frozen metric guards, it:
1. solves a damped weighted least-squares problem using the Jacobian midpoint;
2. applies projected gradient descent under actuator and per-step bounds;
3. uses interval arithmetic on the full Jacobian coefficient ranges;
4. returns a control proposal only when every target and collateral metric interval is wholly inside its accepted band.

The target band must exclude zero so a no-op cannot count as target success. Collateral/no-op tolerance bands must include zero. Every metric must have an explicit unit, role and positive weight.

## Mathematical contract

For matrix midpoint J and desired metric delta y:

~~~text
min_delta_u  Σ_i w_i (J_mid[i] · delta_u - goal_i)^2
             + λ ||delta_u||²
subject to   max(control_min - current, -max_step)
             ≤ delta_u
             ≤ min(control_max - current, max_step)
~~~

For each metric, the full response interval is computed by interval multiplication and summation. A target or preservation guard passes only when the entire predicted interval lies inside the frozen acceptance band. If any metric is outside or overlaps a guard boundary, the solver returns NOT_ROBUSTLY_FEASIBLE and includes the diagnostic predictions; it returns no control delta proposal.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/robust-vocal-control-solver/Cargo.toml
cargo +1.96.0 test --manifest-path tools/robust-vocal-control-solver/Cargo.toml
~~~

## Inputs and provenance

- response model digest and metric/control IDs and units;
- interval-valued Jacobian coefficients;
- no-op calibration receipt digest;
- target and collateral accepted delta bands;
- current control state, hard actuator bounds and per-step trust region;
- regularization, convergence tolerance and maximum iteration budget.

Metric constraints and control states can be supplied in any order; the solver aligns by stable IDs and exact unit matches.

## Boundaries

This solver proposes a control step but does not execute it, render audio or grant runtime authorization. The response matrix and calibration receipts are supplied by upstream qualified components; this crate validates digest syntax and identity equality but does not recompute their canonical hashes or prove the response model is empirically correct.

A robust prediction is not a measurement. After an authorized render, independent analyzers must measure the actual sound and VCCR/VMEA must qualify the actual correction trajectory and edit result. The local linear model is only valid within its calibrated renderer/voice/register/context operating regime.
