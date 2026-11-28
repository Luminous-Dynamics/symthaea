# MEL-HEAR-016 scene grouping evaluator v0.1

Standalone Rust evaluator for permutation-invariant auditory component/source grouping.

```sh
cargo +1.96.0 fmt --all -- --check
cargo +1.96.0 test --locked --all-targets
cargo +1.96.0 clippy --locked --all-targets -- -D warnings
```

The evaluation bundle contains case, prediction and reference records for the evaluator. In production benchmark execution, the reference record is held by the evaluator and never sent to the model. Abstention, ambiguity and not-estimable cases retain separate counts and denominators.
