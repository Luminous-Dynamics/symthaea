# open-display-platform-verify

Standalone Rust 2024 / Rust 1.96 reference verifier for the open display reference platform and run-record contract.

From the repository root:

    cargo fmt --manifest-path tools/open-display-platform-verify/Cargo.toml -- --check
    cargo clippy --manifest-path tools/open-display-platform-verify/Cargo.toml --all-targets -- -D warnings
    cargo test --manifest-path tools/open-display-platform-verify/Cargo.toml
    cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml

The verifier validates two distinct contracts:

- The platform manifest and its Draft 2020-12 schema: component openness, module/source closure, benchmark claim ceilings, and manifest-to-native-CAD default parity.
- The run-record template and its Draft 2020-12 schema: unique identities, cross-record closure, calibration identity/expiry, raw-artifact linkage, observation status semantics and uncertainty-versus-minimum-decision-effect consistency.

Negative tests cover missing/duplicate IDs, dangling references, unsupported openness claims, self-promoted physical validation, benchmark/gate self-promotion, CAD-source drift, measurement data smuggled into a plan-only record, and an inconsistent decision-capability claim.

Success means the schemas/records conform to their contracts. It does not mean the design is fit-tested or build-ready, any selected part is available or licensed for redistribution, any instrument is calibrated, or physical display performance has passed. The checked-in run record is deliberately PLAN_ONLY: it contains four NOT_EVALUATED observation slots, no instruments, and no raw artifacts.
