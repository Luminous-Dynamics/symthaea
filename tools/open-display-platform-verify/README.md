# open-display-platform-verify

Standalone Rust 2024 / Rust 1.96 reference verifier for the open display reference platform and run-record contract.

From the repository root:

    cargo fmt --manifest-path tools/open-display-platform-verify/Cargo.toml -- --check
    cargo clippy --manifest-path tools/open-display-platform-verify/Cargo.toml --all-targets -- -D warnings
    cargo test --manifest-path tools/open-display-platform-verify/Cargo.toml
    cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml

To validate a run-specific geometry receipt after the OpenSCAD checks:

    cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml -- \
      --cad-receipt target/odrh-validation/odrh-cad-runtime-receipt.json \
      --cad-receipt-schema hardware/open-display-reference/validation/odrh-cad-runtime-receipt.schema.json

The verifier validates three contracts:

- The platform manifest and its Draft 2020-12 schema: component openness, module/source closure, benchmark claim ceilings, and manifest-to-native-CAD default parity.
- The run-record template and its Draft 2020-12 schema: unique identities, cross-record closure, calibration identity/expiry, raw-artifact linkage, observation status semantics and uncertainty-versus-minimum-decision-effect consistency.
- An optional run-generated CAD receipt: schema validity, native-source SHA-256, repository/run URL linkage, expected default/alternate mesh extents, all required negative controls, and the physical-validation claim ceiling.

Negative tests cover missing/duplicate IDs, dangling references, unsupported openness claims, self-promoted physical validation, benchmark/gate self-promotion, CAD-source drift, measurement data smuggled into a plan-only record, inconsistent decision-capability claims, malformed RFC 3339 acquisition/review timestamps, impossible calibration expiry dates, missing calibration/capability artifacts, wrong artifact roles, digest-to-artifact mismatch, wrong measurement units, plus CAD-receipt source tampering, bad run linkage, altered expected bounds and omitted/relabelled negative controls.

Success means the schemas/records conform to their contracts. It does not mean the design is fit-tested or build-ready, any selected part is available or licensed for redistribution, any instrument is calibrated, or physical display performance has passed. The checked-in run record is deliberately PLAN_ONLY: it contains four NOT_EVALUATED observation slots, no instruments, and no raw artifacts.
