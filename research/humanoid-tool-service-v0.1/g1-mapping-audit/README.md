# G1 decoupled_v1 mapping audit

A dependency-free Rust reference for the pinned HumanoidToolBench G1 policy mapping.

## What it does

- Projects 43 named G1 joint positions into the exact 32-value benchmark policy state.
- Resolves joints by explicit name rather than assuming that raw array positions are identity.
- Keeps the previous base-height command at state index 31 typed as command history, not physical measurement.
- Decodes and round-trips all 36 `decoupled_v1` action values.
- Applies the documented left-hand flat/natural-order permutation exactly once.
- Keeps torso RPY, base height, navigation velocity, turn flag and absolute target yaw semantically distinct.
- Rejects wrong dimensions, duplicate/unknown joints and non-finite values.

## Run

```bash
cargo fmt --manifest-path research/humanoid-tool-service-v0.1/g1-mapping-audit/Cargo.toml -- --check
cargo test --manifest-path research/humanoid-tool-service-v0.1/g1-mapping-audit/Cargo.toml --locked
```

The crate has no third-party dependencies and includes a committed minimal lockfile. It is intentionally isolated from the main workspace.

## Exact non-goals

This is not a G1 driver, UEC v2 platform implementation, safety projector, trajectory planner, transport, physical calibration, hardware binding, or actuator command path. It does not transform position targets into Symthaea torque/PWM values and must not be routed to the current 21-joint PCA9685 HAL.

## Unresolved upstream source boundary

The pinned HumanoidToolBench policy code consumes `observation["joint_qpos"]` with 43 values. G1Sonic exposes a 43-joint declared name profile and a named joint-position view, but the exact producer that materializes that observation field has not yet been traced end-to-end in the public source. The projection function therefore requires caller-supplied joint names paired with every qpos value. Until the raw materialization path is pinned and independently confirmed, the adapter status remains blocked and this crate demonstrates only the declared mapping arithmetic—not end-to-end benchmark conformance.

Upstream reference: `SNU-PI/HumanoidToolBench@3fec9bd36933a1648ff7f95468c983f3452243a0`.
