# Joint Compute-Risk Calibration v0.1

Standalone Rust research reference for simultaneous calibration of reasoning-stop wrong-answer risk, premature-abandon risk, and emitted-answer error. Tracking: #10052.

Design and theorem boundary: COMPUTE_RISK_JOINT_GRID_V0_1.md.

## Run

    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run --manifest-path research/compute-risk-control/Cargo.toml --bin compute-risk-replay
    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo test --manifest-path research/compute-risk-control/Cargo.toml

Generate Cargo.lock after first dependency resolution, commit it, then use --locked.

## Selection rule

For G fixed candidates and N held-out tasks, the method evaluates three bounded losses for every candidate and uses one simultaneous one-sided Hoeffding correction:

    correction = sqrt( ln(3G / delta) / (2N) )

A candidate is feasible only when upper wrong-answer risk, lower premature-abandon risk, and emitted-answer error are all within their separate budgets. It enumerates the full grid and does not assume threshold-risk monotonicity. Among feasible candidates it minimizes mean token fraction. No candidate is returned when the budgets are not met.

## Claim boundary

The bundled panel is a deterministic stratified synthetic diagnostic, not iid. UCB values test the calculation and ranking only; the runner sets `confidence_claim_eligible_under_declared_design=false`. Any population statement is conditional on genuinely independent iid held-out tasks, bounded losses, a frozen grid, and target-distribution matching. This code does not verify those assumptions.

The lower sigmoid and Hoeffding calculation use f64 with outward PPM rounding; this is not directed-interval proof. The report always sets `formal_theorem_verified=false`. The author repository is preliminary and is provenance-only; the published paper is the algorithmic source.
