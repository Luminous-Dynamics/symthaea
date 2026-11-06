# DOPRI Minimum-Step Termination Verification

This standalone harness compiles the actual `src/dynamics/ode_solvers.rs` source file without resolving the root Symthaea workspace.

## Commands

```bash
cargo fmt --manifest-path research/dopri-min-step-termination-v0/Cargo.toml -- --check
cargo test --manifest-path research/dopri-min-step-termination-v0/Cargo.toml --locked
cargo clippy --manifest-path research/dopri-min-step-termination-v0/Cargo.toml --locked --all-targets -- -D warnings
```

## What it qualifies

- DOPRI tolerance failure at the configured minimum step returns a typed failure and retained accepted prefix.
- Non-finite derivatives/stages do not get mistaken for zero error.
- Attempt-budget exhaustion terminates.
- An analytic harmonic oscillator still completes within the declared numerical tolerance.
- Existing in-module tests for the same solver source also run.

It does not validate the production HDC bridge, FEP model, biological equations, or a physical animal. The root workspace and its independent test suites must still be tested separately.
