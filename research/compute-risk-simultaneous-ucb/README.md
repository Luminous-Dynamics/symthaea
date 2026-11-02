# Simultaneous Compute Risk Calibration

Standalone Rust reference for dual-threshold reasoning stop policies. Tracking: #10052.

## Run

    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run --manifest-path research/compute-risk-simultaneous-ucb/Cargo.toml --bin compute-risk-replay
    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo test --manifest-path research/compute-risk-simultaneous-ucb/Cargo.toml

Generate Cargo.lock on first dependency resolution, commit it, then rerun with --locked.

## Implemented

- Paper-faithful upper false-positive and lower farsighted false-negative losses.
- Frozen finite grid of joint upper-threshold/lower-schedule policies (fixture contract: `fixture-v0.1.schema.json`).
- Simultaneous one-sided Hoeffding UCBs across both risks and every candidate (2M bounds).
- Exact rational record of delta/(2M) and conservative PPM loss/mean/radius rounding.
- Feasibility against separate upper/lower risk budgets.
- Selection of minimum mean normalized token compute among simultaneously feasible candidates.
- JSON report digest and run receipt bound to the exact fixture, candidate grid, trajectory dataset and source commit.

Full derivation, fixture numbers, industry comparison and benchmark expansion: COMPOSITIONAL_COMPUTE_RISK_V0_1.md.

## Claim ceiling

Requires iid/exchangeable complete reasoning trajectories and a frozen candidate grid. Steps inside a trajectory are not independent units. No guarantee under distribution shift, changing model/prompt/verifier, online optional stopping or adaptive candidate expansion. This is not a full reproduction of Conformal Thinking or a production compute controller.
