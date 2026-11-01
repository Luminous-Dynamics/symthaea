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
- a landmark IPCW calibration baseline adapts time-varying history-conditioned censoring weights to proof-completion resource scores;
- selected-policy conformal calibration conditions joint proof-cost envelopes on the frozen scheduler selection mechanism;
- a canonical adapter derives frontier features from the actual target-reachable AND/OR graph and scheduler obligation states;
- an online replay ledger records selected prediction statements and delayed exact, coverage-bit, unknown, censored, and competing-terminal feedback.

## Run

```bash
cargo test --locked --manifest-path research/pod-meta-cost-to-go-lab/Cargo.toml
cargo clippy --locked --manifest-path research/pod-meta-cost-to-go-lab/Cargo.toml --all-targets -- -D warnings
cargo run --locked --manifest-path research/pod-meta-cost-to-go-lab/Cargo.toml --bin pod-meta-cost-to-go
```

The qualification workflow pins Rust 1.96.0 to match the repository toolchain.

## Scope and limitations

This is a research lab, not a production scheduler or physical resource enforcer. The cost-to-go predictor and conformal components are explicit structural baselines, not learned predictors or a claimed SOTA result. The graph adapter binds features to canonical bytes using a non-cryptographic identity checksum for deterministic tests only; production provenance needs cryptographic hashing and signed or attested pre-action state capture. The forecast journal and online-selection ledger are append-only in-process research ledgers, not durable distributed consensus or provider-side metering.

The survival/IPCW components preserve censoring and competing-terminal distinctions but do not inherit HAPS or any other theorem automatically. The frozen selected-policy calibrator is not CAP and does not control online FCR across repeated picks. The online feedback ledger added in #9720 reports descriptive observed FCR on feedback-resolved selected statements; it is not a selection-corrected estimator or a coverage guarantee. Scalar results do not imply simultaneous six-resource coverage.

Qualification status is always tied to the exact PR head and workflow run. A passing synthetic matrix establishes implementation invariants, not forecasting quality or superiority over external benchmarks.
