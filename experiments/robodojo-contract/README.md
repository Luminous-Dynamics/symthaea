# Standalone RoboDojo contract gate

This dependency-free Rust crate is the first runnable, simulator-independent test layer for the Symthaea × RoboDojo work. It checks episode boundaries, observation shape/finiteness/order, stale action rejection, action dimension and numeric bounds, prediction-error-triggered replanning, and latched stop behavior.

## Run without Isaac Sim or the root workspace

```sh
cargo test --locked --manifest-path experiments/robodojo-contract/Cargo.toml
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml
```

The nested workspace and lockfile are intentional: contract tests should not have to resolve or compile Symthaea's large workspace, download a policy checkpoint, start Isaac Sim, or wait for its runners.

## What approval means

An approved proposal is only valid against this crate's declared vector dimensions and per-coordinate numeric limits, for the current episode and latest accepted observation. It is **not** a kinematics check, collision check, workspace guarantee, policy-quality measure, or physical safety certificate. Robot-specific adapters must add those constraints where the appropriate state/telemetry is available.

Prediction error is a caller-defined scalar and must be calibrated for the selected policy. A high value suppresses action approval and requests replanning; repeated high values latch a stop. It is not used as proof of safety, consciousness, or task competence.

## Scope boundary

This crate currently validates the integration boundary only. It does not connect to RoboDojo, XPolicyLab, a pretrained policy, a simulator, or actuators. Passing these tests is not a RoboDojo task result. The next integration step is to write the thin XPolicyLab adapter and run the official simulator-side contract before comparing baseline versus sidecar task success.
