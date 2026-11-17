# subterra-inspection-voi-adapter

A thin Rust adapter over the existing `subterra-survival-reference` planner and `subterra-robust-survival-referee`.

## What it does

- computes a no-test decision baseline and prior entropy;
- computes per-action expected information gain and one-step gross EVSI;
- invokes the existing finite-horizon planner for a contingent policy;
- independently evaluates that exact policy across supplied model scenarios;
- requires the exact primary model to be present in the referee's model set;
- exposes referee cache-mismatch counts when candidate branch probabilities/posteriors do not match a shifted model;
- keeps resource cost vectors separate from decision-loss units;
- includes a synthetic case demonstrating high EIG / zero EVSI versus lower EIG / positive EVSI.

## Run the synthetic comparison

```sh
cargo run --manifest-path Cargo.toml --example synthetic_evsi --offline
```

The example reports no-test risk, entropy-greedy selection, myopic EVSI selection, the finite-horizon planner's choice, robust multi-model evaluation and expected resources for three task labels: pipe/cable, drawpoint volume and support verification.

## Frozen synthetic known answer

With the current four-hypothesis prior and two candidate observation models:
- Prior/no-test Bayes risk: 9.0 loss units.
- High-EIG scan: 0.6167159329 nats expected information gain, but exactly 0.0 gross EVSI because every observation branch keeps the same optimal decision.
- Lower-EIG defect follow-up: 0.0099459834 nats expected information gain and 1.09 gross EVSI because it changes the preferred decision on the consequential-defect branch.
- The finite-horizon planner should choose the lower-EIG/high-EVSI action; the independent referee evaluates that exact policy under both nominal and shifted-prior synthetic models.

These values are arithmetic known answers for the fixture, not sensor-performance estimates or field results. The action resource vector remains separate (35 J, 10 s, 100 bytes, 0.01 exposure units for the decision-relevant fixture action); the example does not convert those unlike units into a monetized cost.

## Verify

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml --all-targets --offline
cargo clippy --manifest-path Cargo.toml --all-targets --offline -- -D warnings
```

The crate is a task adapter, not a sensor model, robot controller, authority system, or field-validation package. Its synthetic likelihoods and loss functions are illustrative known-answer fixtures only.
