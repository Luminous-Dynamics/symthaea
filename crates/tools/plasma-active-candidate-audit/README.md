# plasma-active-candidate-audit

Fail-closed Rust auditor for one proposed active kinetic plasma simulation run.

## Run

    cargo fmt --manifest-path crates/tools/plasma-active-candidate-audit/Cargo.toml -- --check
    cargo test --locked --manifest-path crates/tools/plasma-active-candidate-audit/Cargo.toml
    cargo clippy --locked --manifest-path crates/tools/plasma-active-candidate-audit/Cargo.toml --all-targets -- -D warnings
    cargo run --locked --manifest-path crates/tools/plasma-active-candidate-audit/Cargo.toml -- benchmarks/plasma/active-kinetic-candidate-valid-v0.json

The positive fixture exits 0. The rights-negative fixture is intentionally rejected and exits 1:

    cargo run --locked --manifest-path crates/tools/plasma-active-candidate-audit/Cargo.toml -- benchmarks/plasma/active-kinetic-candidate-rights-negative-v0.json

## Contract

The auditor parses a strict typed candidate schema and then applies cross-field admission rules:
- the physics lane is explicit and must use its canonical wire name;
- geometry grid dimensions must match the declared spatial dimension;
- PIC requires a positive particles-per-cell value;
- Vlasov/VFP require a positive phase-space resolution map;
- solver and acquisition policy use immutable 40-character commit SHAs;
- evidence receipts use 64-character SHA-256 digests;
- acquisition signals and cost values must be finite;
- cost estimates use raw resource units, not hidden weighted or monetary scores;
- only training and development/validation roles can be acquired; a sealed-test role is rejected at deserialization;
- selected or executed candidates require rights_state=verified;
- executed candidates require an execution receipt.

The JSON Schema Draft 2020-12 lives at benchmarks/plasma/active-kinetic-candidate.schema.json. The Rust implementation duplicates the operationally important cross-field constraints and refuses unknown fields at every typed object boundary. Keep the schema and tests in sync.

## Receipts and limitations

The CLI emits a deterministic request digest and report digest. verify_candidate_report replays the original typed candidate and compares the full report, rejecting modified reports even if the unkeyed receipt digest is recomputed.

These hashes are integrity metadata, not signatures, not legal opinions, and not authentication of solver authors or data providers. A rights_state=verified value is a recorded assertion in the candidate manifest; an independent rights review remains required before actual use.

This tool validates proposal metadata only. It does not generate acquisition scores, select runs, launch PIC/Vlasov, download data, decide that physics is safe, or authorise experiments. No candidate pool is included and no kinetic simulation is selected by this crate.
