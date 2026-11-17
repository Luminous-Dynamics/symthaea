# subterra-inspection-voi-adapter

A thin Rust adapter over the existing `subterra-survival-reference` planner and `subterra-robust-survival-referee`.

## What it does

- computes a no-test decision baseline and prior entropy;
- computes per-action expected information gain and one-step gross EVSI;
- invokes the existing finite-horizon planner for a contingent policy;
- independently evaluates that exact policy across supplied model scenarios;
- keeps resource cost vectors separate from decision-loss units;
- includes a synthetic case demonstrating high EIG / zero EVSI versus lower EIG / positive EVSI.

## Verify

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml --all-targets --offline
cargo clippy --manifest-path Cargo.toml --all-targets --offline -- -D warnings
```

The crate is a task adapter, not a sensor model, robot controller, authority system, or field-validation package. Its synthetic likelihoods and loss functions are illustrative known-answer fixtures only.
