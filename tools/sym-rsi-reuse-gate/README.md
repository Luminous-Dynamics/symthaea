# Symthaea RSI REUSE gate harness

This package is a deliberately small, standalone Cargo workspace for the dependency-free statistical kernel used by the RSI adaptive-evaluation gate.

It imports the production source files directly with Rust `#[path]` attributes:

- `src/consciousness/recursive_improvement/paired_sign_test.rs`
- `src/consciousness/recursive_improvement/reuse_error_budget.rs`

There is no copied implementation and no dependency on the full Symthaea application graph.

Run the focused qualification locally with:

```sh
cargo test --manifest-path tools/sym-rsi-reuse-gate/Cargo.toml --locked
cargo clippy --manifest-path tools/sym-rsi-reuse-gate/Cargo.toml --all-targets --locked -- -D warnings
```

This harness verifies the allocation and decision-kernel source only. It does not prove integration with the complete Symthaea crate, qualify a holdout service, implement the cumulative REUSE confidence certificate, or establish the REUSE simultaneous guarantee.
