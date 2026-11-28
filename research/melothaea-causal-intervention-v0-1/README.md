# MEL-HEAR-012 causal acoustic intervention contract v0.1

Standalone Rust validator for byte-bound intervention manifests and execution receipts.

```sh
cargo +1.96.0 fmt --all -- --check
cargo +1.96.0 test --locked --all-targets
cargo +1.96.0 clippy --locked --all-targets -- -D warnings
```

This crate does not run audio transforms and does not certify their acoustic or perceptual correctness. It ensures that a benchmark case's source bytes, exact manifest bytes, transform parameters, implementation identity, deterministic seed, declared support and materialized output bytes are bound together.
