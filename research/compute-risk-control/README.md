# Joint Compute-Risk Calibration (v0.1)

Standalone Rust research reference for jointly calibrating reasoning-stop wrong-answer risk and premature-abandon risk over a frozen finite candidate grid. Tracking: #10052.

Design and theorem boundary: COMPUTE_RISK_JOINT_GRID_V0_1.md.

## Run

    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run --manifest-path research/compute-risk-control/Cargo.toml --bin compute-risk-replay
    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo test --manifest-path research/compute-risk-control/Cargo.toml

Generate Cargo.lock after first dependency resolution, commit it, then use --locked.

## Joint calibration

For G frozen candidates and N independent held-out tasks, each candidate has two bounded losses: upper wrong-answer exit and lower premature-abandon loss. One simultaneous one-sided Hoeffding correction is applied to all 2G claims:

    correction = sqrt( ln(2G / delta) / (2N) )

Candidate i is feasible only when both adjusted upper bounds are below their independently declared operational risk budgets. Among feasible candidates, v0.1 chooses the lowest mean token fraction. It enumerates the entire grid and assumes no monotonicity in threshold or risk.

This is a joint-grid variant, not the paper's sequential two-stage threshold-fitting procedure. The finite-grid union bound avoids uncorrected adaptive selection over candidate-risk pairs under the iid held-out task assumption.

## Claim boundary

The bundled panel is stratified and synthetic, explicitly not iid. The runner computes UCB values to test the algorithm and ranking, but sets `confidence_claim_eligible_under_declared_design=false`. The method's population-level simultaneous statement is conditional on genuine independent iid held-out tasks, bounded losses, a grid frozen before calibration outcomes, and correct target-distribution matching. This code does not verify those assumptions or theorem validity.

The logistic lower-threshold and Hoeffding calculations use f64 with outward PPM rounding. Directed-interval numerical verification remains future work. The report always sets `formal_theorem_verified=false`.

The author repository cited by the fixture is preliminary and not used as an implementation source; the paper is the algorithmic authority. Do not interpret this crate as a paper-matched reproduction.
