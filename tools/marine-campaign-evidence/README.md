# Marine campaign evidence verifier

This is a standalone Rust tool workspace. It consumes the marine campaign-binding JSON profile and checks cross-record evidence joins, including method-declared minimum sample rate versus the actual channel sample rate; it does not add a new observation, calibration, quantity/unit, clock or raw-data ontology.

## Run

```sh
cargo test --manifest-path tools/marine-campaign-evidence/Cargo.toml
cargo fmt --manifest-path tools/marine-campaign-evidence/Cargo.toml -- --check
```

The package pins `serde_json` to 1.0.150. Because this tool is intentionally outside the root Symthaea workspace, its dependency lock is separate from the root `Cargo.lock`; the first Cargo invocation will generate a local `Cargo.lock` in this directory. Commit that generated lock after a toolchain run before making a reproducibility claim.

## Disposition semantics

- `Invalid`: malformed identity/time/instrument references or contradictory record semantics.
- `NotAdmissible`: known hard prerequisite fails (configuration mismatch, stale calibration, missing sampling review, integrity issue, etc.).
- `Indeterminate`: timing alignment or uncertainty is insufficient to decide.
- `SyntheticOnly`: valid fixture/replay/simulation, not a new physical acquisition.
- `EligibleForIndependentReview`: references and semantic gates are present; this is not qualification or approval.

JSON Schema structural validation remains a separate step. This crate performs cross-record semantic checks. It does not validate instrument accuracy, water quality, electrical safety, power quality, seaworthiness or authorization.
