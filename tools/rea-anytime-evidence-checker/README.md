# REA anytime evidence acquisition checker

Standalone Rust fixture checker and finite readiness reference for REA-RUST-053.

## Run

```bash
cargo test --manifest-path tools/rea-anytime-evidence-checker/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-anytime-evidence-checker/Cargo.toml --all-targets --locked -- -D warnings
```

The checker:
- meta-validates both Draft 2020-12 schemas and validates both instances;
- enforces unique claim/obligation/action identities and monotonically increasing claim burdens;
- verifies agent/grader episode correspondence and that the agent view contains no grader-only fields;
- checks event outcomes, action eligibility, action/cost budgets, evidence-class admissibility and invalidated Evidence references;
- recomputes Ready after every event from current obligation and defeater-duty states;
- validates stop checkpoints and the strongest ready claim;
- tests revocation, bounded claims, challenge duties and event-order invariance.

This is a synthetic development checker. It does not authenticate Evidence, execute experiments, calibrate likelihoods, adjudicate world state, or provide a production stop policy. The grader file is public development data, not a hidden benchmark.
