# MEL-HEAR-014 EVSI active auditory analysis v0.1

Standalone Rust policy for choosing the next auditory analysis by expected value of sample information and a declared resource budget.

```sh
cargo +1.96.0 fmt --all -- --check
cargo +1.96.0 test --locked --all-targets
cargo +1.96.0 clippy --locked --all-targets -- -D warnings
```

Likelihood models are supplied and externally calibrated. This crate validates probability contracts and computes one-step decision value; it does not estimate auditory probabilities or provide a globally optimal multi-step sensing policy.
