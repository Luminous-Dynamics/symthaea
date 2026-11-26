# Bootstrap Survivability Reference

This crate is a dependency-free Rust reference for the *decision semantics* of bootstrap/control-plane qualification. It is intentionally not an optical reader, calibration model, metadata ECC implementation, signature verifier, JSON Schema validator or payload decoder.

## Responsibilities

- Reject duplicate or empty required bootstrap object identifiers.
- Require a valid SHA-256-shaped expected digest for every required object.
- Require observed current/recovered record digest to match the expected digest.
- Require calibration records to be current and scoped to the expected reader/firmware.
- Keep missing/unknown bootstrap evidence as incomplete.
- Reject stale, corrupt, not-applicable-required and wrong-but-valid metadata.
- Pass a complete archive only when both bootstrap and payload dispositions are exact/recovered.

Operational dependency independence is deliberately not evaluated here. Mycelix #5762 remains the canonical common-cause/dependency authority. The Recovery Capsule owns package and representation-information semantics.

## Run locally

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

The current assistant execution environment has Python 3.13.5 but no Cargo, rustc or rustfmt, so these commands have **not** been executed here. CI or a local Rust toolchain must run them against the exact branch head.

## Important limit

The synthetic bootstrap metadata code in the parent fixture declares a GF(256) [6,4,3] shard code using polynomial 0x11d and two known-erasure recovery. This crate does not implement that code. It evaluates whether required bootstrap objects and final payload outcomes permit an archive-level pass after an external decoder has produced candidate records.
