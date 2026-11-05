# SAM policy-trajectory support v0.1

A small, dependency-free Rust reference for the finite path-support conformance slice in Symthaea #10514.

This crate distinguishes exact membership in a declared finite joint-trajectory support set, absence under a declared exhaustive finite support universe, absence when the evidence universe is not known to be exhaustive, incomplete trajectories, and policy/context scope mismatches.

It intentionally does **not** estimate propensities, learn a support classifier, infer real-world support, run a simulator, or certify a policy as safe. Exact path membership is not a practical sample-size guarantee.

## Run

    cargo fmt --manifest-path tools/sam-policy-trajectory-support/Cargo.toml -- --check
    cargo clippy --locked --manifest-path tools/sam-policy-trajectory-support/Cargo.toml --all-targets -- -D warnings
    cargo test --locked --manifest-path tools/sam-policy-trajectory-support/Cargo.toml

## Conformance cases

The unit tests include:
1. Marginal-action support can falsely accept a path absent from the joint trajectory corpus.
2. Per-agent marginal support does not imply joint ego/actor action support.
3. Full history identity prevents compressed-state aliasing.
4. An incomplete/censored path cannot be admitted.
5. Absence in a non-exhaustive corpus remains unknown.
6. Target-policy version and context are bound to the support manifest.
7. Canonical action ordering is independent of input order and duplicate actor commands reject.
8. Exact conditional chain-rule mass can become small without becoming structural zero.

The chain-rule helper accepts probabilities only when each input is already the conditional probability given the full preceding **joint** history under one declared estimand. Multiplying marginal probabilities or per-agent propensities is not justified by this helper.

## Claim boundary

Passing the finite fixtures establishes correct bookkeeping for these declared finite cases. It does not establish practical support in an empirical dataset, valid propensities, causal identification, human-driver behavior, simulator fidelity, collision-risk calibration, vehicle safety, or SAE Level 4/5 capability.
