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


## Pathway selection v0.2 diagnostics

The pathway report includes Wilson 95% intervals and Monte Carlo standard errors for feasibility, Pareto membership and conditional pairwise dominance. Pairwise dominance intervals use the number of joint-feasible draws as their denominator. Mean regret also includes an MCSE.

Four baseline rankings are reported with `diagnostic_only=true`: best qualified non-plasma incumbent, fixed equal-weight normalized sum, TOPSIS, and central-mean non-dominated sorting. The last is not NSGA-II. No baseline is allowed to override hard constraints or abstention.

These statistics quantify only simulation sampling error for the declared synthetic distributions. They do not demonstrate empirical probability calibration or physical route performance.
