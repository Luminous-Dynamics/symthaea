# Landauer Erasure Verification 001

This is a dependency-free standalone Rust harness that compiles the exact
production source file
`crates/core/symthaea/src/physics/landauer_erasure.rs` by path.

Run:

```sh
cargo fmt --manifest-path research/landauer-erasure-001/Cargo.toml -- --check
cargo test --locked --manifest-path research/landauer-erasure-001/Cargo.toml
cargo clippy --locked --manifest-path research/landauer-erasure-001/Cargo.toml --all-targets -- -D warnings
```

The test corpus pins one-bit Landauer heat values at 300 K and 310 K, checks
linear scaling with the declared erased-bit count and bath temperature, and
rejects negative/non-finite bit counts, invalid temperatures, invalid
Boltzmann constants, and arithmetic overflow.

This validates the pure formula implementation only. It does not prove that
any upstream Shannon-entropy, FLOP, belief-change, or token-count telemetry is
a count of physically erased bits. That measurement/instrumentation gap remains
separate.
