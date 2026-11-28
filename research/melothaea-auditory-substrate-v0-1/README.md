# MEL-HEAR-011 auditory evidence substrate v0.1

This standalone Rust crate validates the first versioned evidence contract and a deterministic known-signal metadata fixture.

## Run

```sh
cargo +1.96.0 fmt --all -- --check
cargo +1.96.0 test --locked --all-targets
cargo +1.96.0 clippy --locked --all-targets -- -D warnings
cargo +1.96.0 run --locked --example validate_fixture -- .
```

The fixture contains 16 float32 little-endian sample values encoded as hexadecimal text. It is a small metadata/control signal, not a music recording and not a DSP qualification case.

All four feature views are marked `planned`. No transformed feature artifacts, event hypotheses, scene assignments or musical claims are present. This is intentional: a representation declaration must not be mistaken for a computed or qualified representation.
