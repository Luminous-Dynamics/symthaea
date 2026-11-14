# MEL-MUSE-MERGE-001 reference evaluator

This is a deliberately bounded, pure-standard-library Rust preflight evaluator for the MUSE-MERGE-001 research contract.

## Run

```sh
cargo fmt --manifest-path tools/mel-muse-merge/Cargo.toml --all -- --check
cargo test --manifest-path tools/mel-muse-merge/Cargo.toml --locked
cargo clippy --manifest-path tools/mel-muse-merge/Cargo.toml --all-targets --locked -- -D warnings
```

The crate declares its own `[workspace]` so it remains independent of the large parent workspace and can be evaluated without loading unrelated packages.

## Claim boundary

- No audio, DAW, score rendering, Studio persistence, identity provider or production commit path is used.
- A `SafeAutomaticMerge` outcome is only a preflight footprint result; it never authorizes commit.
- Actual patch application, reverse-order result equivalence, semantic invariant verification, compare-and-set, durable commit and independent replay remain later stages.
- The fixture manifest is checked for stable IDs/outcome declarations but is not yet deserialized by this no-dependency harness.