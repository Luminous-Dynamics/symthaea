# POD-Meta Cost-to-Go Lab

A standalone Rust research crate for the first typed slice of [issue #9652](https://github.com/Luminous-Dynamics/symthaea/issues/9652).

The lab separates forecast estimation from resource enforcement. Its initial execution slice focuses on the invariant that forecast quality must never be confused with resource enforcement:

- six-dimensional integer resource vectors;
- atomic reserve-before-dispatch budget checks;
- reservations remain held while action outcomes are unknown;
- authoritative reconciliation distinguishes applied, not-applied, and still-unknown effects;
- settlement charges measured usage, not the forecast reservation;
- reservation overruns and hard-cap violations remain observable;
- a shift-aware feasibility gate includes the mandatory verification reserve;
- a non-parametric time-to-sufficient-proof baseline separates right-censoring from competing terminal events;
- an append-only forecast journal stores pre-execution predictions separately from evaluator, meter, and verifier outcomes;
- a proof-frontier structural baseline feeds one-sided split-conformal simultaneous resource-vector envelopes;
- a landmark IPCW calibration baseline adapts time-varying history-conditioned censoring weights to proof-completion resource scores.

## Run

```bash
cargo test --locked --manifest-path research/pod-meta-cost-to-go-lab/Cargo.toml
cargo clippy --locked --manifest-path research/pod-meta-cost-to-go-lab/Cargo.toml --all-targets -- -D warnings
cargo run --locked --manifest-path research/pod-meta-cost-to-go-lab/Cargo.toml --bin pod-meta-cost-to-go
```

The qualification workflow pins Rust 1.96.0 to match the repository toolchain.

## Scope and limitations

This slice is not yet a frontier-conditioned remaining-cost predictor, IPCW estimator, conformal predictor, HARP implementation, concurrent distributed ledger, or production adapter. The forecast journal is a serialized append-only research log, not a signed tamper-evident audit log or distributed consensus service. The survival baseline is unconditional from task start and assumes censoring is independent conditional on the declared cohort. It supplies typed resource / receipt boundaries and deterministic survival / ledger invariants for subsequent experiments. The in-process simulator cannot prove provider-side spend enforcement; a production hard-cap claim requires real adapters with enforceable maximums and authoritative metering.

No Rust test, formatting, Clippy, or matrix result is claimed until the exact PR head has qualified.
