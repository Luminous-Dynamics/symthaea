# open-display-platform-verify

Standalone Rust 2024 / Rust 1.96 reference verifier for the Open Display Reference Hardware (ODRH-001) manifest.

From the repository root:

    cargo fmt --manifest-path tools/open-display-platform-verify/Cargo.toml -- --check
    cargo clippy --manifest-path tools/open-display-platform-verify/Cargo.toml --all-targets -- -D warnings
    cargo test --manifest-path tools/open-display-platform-verify/Cargo.toml
    cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml

The verifier compiles and executes the checked-in Draft 2020-12 JSON Schema, then checks cross-record source closure and claim ceilings the schema alone does not establish. Negative tests cover missing identity, duplicate IDs, dangling module references, unsupported openness claims, self-promoted physical validation, and evidence/gate promotion.

Success means only CONTRACT_VALID_UNQUALIFIED: the specification manifest conforms to its contract. It does not mean the design is build-ready, any selected part is available or licensed for redistribution, the platform is safe, any instrument is calibrated, or physical display performance has passed.
