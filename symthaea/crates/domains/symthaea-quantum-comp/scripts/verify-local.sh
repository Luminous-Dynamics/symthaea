#!/usr/bin/env sh
set -eu
cargo fmt --check
cargo test --all-features
cargo run --example binding_probe
cargo run --example noise_sweep
cargo run --example comparative_report
cargo run --example negative_control
cargo run --example entanglement_proxy
cargo run --example robustness_summary
cargo run --example report_exports
cargo run --example audit_controls
cargo run --example experiment_matrix
cargo run --example significance_probe
cargo run --example research_receipt
cargo run --example preflight_presets
cargo run --example research_bundle
cargo run --example fixture_catalog
cargo run --example replay_plan
cargo run --example release_gate
cargo run --example interop_boundary
cargo run --example release_manifest
cargo run --example api_inventory
cargo run --example verification_matrix
cargo run --example migration_guide
cargo run --example beta_readiness
cargo run --example validation_snapshot
cargo run --bin symthaea-quantum-comp -- presets
cargo run --bin symthaea-quantum-comp -- schemas
cargo run --bin symthaea-quantum-comp -- fixtures
cargo run --bin symthaea-quantum-comp -- replay smoke
cargo run --bin symthaea-quantum-comp -- gate smoke-binding
cargo run --bin symthaea-quantum-comp -- inventory
cargo run --bin symthaea-quantum-comp -- manifest
cargo run --bin symthaea-quantum-comp -- verify-matrix
cargo run --bin symthaea-quantum-comp -- migration
cargo run --bin symthaea-quantum-comp -- beta
cargo run --bin symthaea-quantum-comp -- snapshot
cargo run --bin symthaea-quantum-comp -- binding smoke
cargo run --bin symthaea-quantum-comp -- noise smoke
cargo run --bin symthaea-quantum-comp -- matrix smoke
cargo test --features qasm-export
