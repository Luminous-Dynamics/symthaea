# Adaptive Temporal Replay

Standalone Rust research crate for executing the v0.1 adaptive temporal calibration stress registry.

## Implemented

- SplitMix64 deterministic PRNG with independent scenario/stream seed derivation.
- Integer-only PPM mapping, ties-away-from-zero residual arithmetic, tail contamination and unbiased bounded feedback delays.
- Piecewise location/scale/covariate regimes.
- Explicit source availability and source-outage recovery schedules.
- Delayed outcome feedback and deterministic supersede/retract events.
- Append-only bitemporal label ledger with as-known queries and replay verification.
- Scalar ACI baseline applied only when feedback is delivered.
- Finite-sample split-conformal order statistic with an explicit unbounded result when the calibration sample is insufficient.

## Commands

From this directory:

    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run --bin atcb-synthetic -- abrupt_location_shift 20261102
    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run --bin atcb-synthetic -- superseded_outcome_feedback 83
    cargo test

The crate is standalone and uses serde/serde_json plus sha2 for fixture parsing and receipt digests. The CLI refuses to emit a receipt unless GIT_COMMIT_SHA is set to a 40-character lowercase commit SHA. Run from a clean worktree and commit Cargo.lock after the first Cargo resolution; then rerun with --locked. The receipt schema is run-receipt-v0.1.schema.json. The code SHA is caller-supplied metadata, not a cryptographic signature or proof that the worktree is clean.

## Limitations

- The v0.1 generator implements the registry's IID-uniform residual mechanism. It does not implement AR(1), GARCH(1,1), or Markov-switching dependence. Those are separate pending stress families and must exist before any delay-to-memory claim.
- The ACI update is a scalar baseline applied in feedback-arrival order. It is not an implementation of the fixed-horizon interleaved delayed-ACI theorem, SAOCP, UP-OCP, MACP, AcMCP, or BMAC.
- Supersede uses a sign-flipped residual as an adversarial replacement; retract removes the active label. This is synthetic stress behavior, not a model of any particular real data source.
- A source outage delays availability of affected target feedback until recovery; it does not alter the hidden synthetic outcome.
- No real dataset, model, CI run, Rust test, performance result, or formal coverage guarantee is claimed by the source artifact itself.
