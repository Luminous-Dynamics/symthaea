# Standalone RoboDojo contract gate

This dependency-free Rust crate is the first runnable, simulator-independent test layer for the Symthaea × RoboDojo-RC work. It checks episode boundaries, observation shape/finiteness/order, stale action rejection, action dimension and numeric bounds, prediction-error-triggered replanning, and latched stop behavior. It also provides partial-progress trial records, task-by-variant summaries that retain invalid-run/timeout counts, deterministic bootstrap confidence intervals, and paired baseline-versus-candidate comparisons stratified by task.

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

The statistics layer treats a valid partial grade as scored progress and a missing grade as an invalid run. Invalid runs remain in the attempted-trial and reliability denominators; they are excluded only from the mean of graded progress. Paired progress deltas require the same task/trial IDs on both sides and macro-average tasks equally before bootstrapping so an overrepresented easy task cannot dominate the comparison.

This crate currently validates the integration boundary and scoring logic only. It does not connect to RoboDojo-RC, XPolicyLab, a pretrained policy, a simulator, or actuators. Passing these tests is not a RoboDojo-RC task result. The next integration step is to write the thin XPolicyLab adapter and run a simulator/physical evaluation before comparing baseline versus sidecar task progress.
