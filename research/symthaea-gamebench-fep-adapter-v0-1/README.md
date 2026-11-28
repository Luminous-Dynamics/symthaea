# Symthaea GameBench FEP navigation adapter

This is an isolated research adapter for the actual symthaea-fep::ActiveInferenceAgent. It composes with GameBench #14137 and uses the observation/action-only interface in symthaea-gamebench-v0-1.

## Scope

- F1 deterministic navigation only.
- O1 structured full-map observation only: this baseline requires a visible goal.
- Two continuous position channels are normalized to [0,1].
- Four action indices map to North, East, South, West.
- The FEP generative model starts from an identity transition matrix plus action-conditioned directional biases.
- The adapter passes the next candidate-visible observation through ActiveInferenceAgent::perceive, allowing its existing temporal-difference learner to use the previous belief/action transition.
- The candidate returns normalized coordinate forecasts based on predicted-state means and precision.
- Action-selection softmax probability is not represented as success confidence. Confidence is absent until an independent calibration map is fitted and qualified.

## What this does not establish

This is a named FEP baseline, not the complete HDC+CfC Symthaea architecture. It does not yet use a Symthaea HDC encoder, CfC/LTC temporal network, episodic memory adapter, active sensing in F4, hidden-rule inference in F6, or visual input. It does not demonstrate learning improvement, transfer, game-winning ability, or SOTA performance.

The underlying FEP transition model uses a deliberately simple structured prior. Report it as fep-active-inference/navigation-v0.1, not as an undifferentiated Symthaea score.

## Isolation

This adapter has its own Cargo workspace and is not a production Symthaea workspace member. It depends on the actual path package crates/core/symthaea-fep and on the standalone GameBench contract crate. No production manifest or dependency has been changed.

## Validation

From this directory:

```sh
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

Tests are source-authored. No local test result is claimed until the exact branch head passes the dedicated adapter workflow.