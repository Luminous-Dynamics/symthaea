# Connectome Identity Verification Harness

This standalone package compiles the exact Symthaea connectome identity source file outside the root workspace.

## Commands

```bash
cargo fmt --manifest-path research/connectome-identity-verify-v0/Cargo.toml -- --check
cargo test --manifest-path research/connectome-identity-verify-v0/Cargo.toml --locked
cargo clippy --manifest-path research/connectome-identity-verify-v0/Cargo.toml --locked --all-targets -- -D warnings
```

## Scope

It runs the crate's synthetic source-identity tests, including:
- exact row/column/annotation identity joins;
- independent source-axis permutations;
- semantic canonical digest versus source-binding digest;
- explicit canonical matrix and annotation/parameter remapping;
- stable-ID stimulus and keep/remove target resolution;
- malformed, missing, duplicated and unknown identity rejection.

These tests qualify the Rust identity contract only. They do not import external Drosophila/C. elegans matrix artifacts or validate their anatomical truth. Root workspace integration remains a separate gate.
