# REA evidence-force fixture checker

Read-only structural and semantic consistency checks for the original synthetic development corpus in #11061.

Run:

```bash
cargo test --manifest-path tools/rea-evidence-force-fixture-check/Cargo.toml --locked
cargo fmt --manifest-path tools/rea-evidence-force-fixture-check/Cargo.toml -- --check
cargo clippy --manifest-path tools/rea-evidence-force-fixture-check/Cargo.toml --all-targets --locked -- -D warnings
```

This checker validates the Draft 2020-12 schema, dataset conformance, exact substring anchors for each declared force mutation, axis coverage, identifiers, and positive evidence-growth controls.

It does not adjudicate the semantic correctness of synthetic labels, prove each pair changes only one psychological/linguistic dimension, evaluate a model, or replace the required human/domain review. The spans are mechanical anchors, not a substitute for adjudication.