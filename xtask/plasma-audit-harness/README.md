# Isolated plasma energy audit harness

This package compiles the exact source file at `../src/plasma_energy_audit.rs` without resolving the full Symthaea workspace. It exists because a missing glob-discovered workspace member on the research branch prevents normal Cargo workspace metadata resolution.

Run from the repository root:

```sh
cargo fmt --manifest-path xtask/plasma-audit-harness/Cargo.toml -- --check
cargo test --manifest-path xtask/plasma-audit-harness/Cargo.toml --locked
cargo clippy --manifest-path xtask/plasma-audit-harness/Cargo.toml --all-targets --locked -- -D warnings
cargo run --manifest-path xtask/plasma-audit-harness/Cargo.toml --locked
```

The harness does not replace or fork the production xtask module. It imports the exact file by relative path. It audits only frozen source/thermal manifests and does not run plasma physics, a plasma chemistry solver, or a physical test.
