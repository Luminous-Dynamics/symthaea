# Applying Symthaea Music Theory Patch Series 22

This kit contains patch plans, not mail patches. Produce the real series only against the exact verified Series 21 tree.

Required final gates:

```text
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo check -p symthaea-music-theory --examples
```

Run every conformance fixture through the Rust reference verifier and at least one independent implementation. Any disagreement blocks release.
