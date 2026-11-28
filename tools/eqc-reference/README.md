# Luminous EQC Reference Validator

This is an isolated Rust workspace for validating the Evidence-Qualified Capability v0.1 seed corpus. Its own Cargo workspace and lockfile deliberately keep it independent of Symthaea's main workspace membership and root lockfile.

From the repository root:

```bash
cargo fmt --manifest-path tools/eqc-reference/Cargo.toml -- --check
cargo test --locked --manifest-path tools/eqc-reference/Cargo.toml
cargo run --locked --manifest-path tools/eqc-reference/Cargo.toml -- \
  --corpus docs/research/eqc-seed-corpus-v0.1.json \
  --out /tmp/eqc-reference-report.json
```

The tool validates the task and receipt JSON Schemas offline, checks each seed task against Draft 2020-12, verifies repository-contained fixture paths and byte-level SHA-256 digests, checks fixture contents against task authority and timestamps, and confirms that no-op and always-allow baselines fail their respective controls.

The output is labelled `reference_contract_only_not_system_under_test`. This is not a clinical/EHR integration, cryptographic protocol audit, external system evaluation, or production qualification.
