# Standalone scientific decision/replay contract harness

This crate compiles the production support modules directly:

- scientific_decision_replay.rs: prospective decision, action payload, causal-design, evaluator-resolution and cost contracts.
- evolutionary_rescue_custody.rs: custodian-only replay archive, evaluator capsule, digest linkage and decision-prefix construction.

The harness is intentionally outside the root Cargo workspace. The root workspace currently fails during member enumeration because crates/core/* includes crates/core/symthaea, which has no checked-in Cargo.toml. Cargo fails before package selection or compilation. Do not fabricate a package manifest to mask that workspace defect.

The small common::SerializedWorld and symthaea_futures_symtropy::evolutionary_rescue shims reproduce the serde field shape used by the custody projection. They allow schema, digest, leakage and temporal-prefix tests to execute without pulling the full simulator/domain graph into this isolated target.

Qualification boundary: this harness tests the actual production support-module source and its unit tests, but its shim does not prove integration against the real symthaea-futures-symtropy crate, the real SerializedWorld helper, the example CLI or the full workspace. Those still require a successful package-level build after the workspace closure is repaired.

Run:

    cargo test --manifest-path tools/sym-rsi-decision-replay-contract/Cargo.toml
    cargo clippy --manifest-path tools/sym-rsi-decision-replay-contract/Cargo.toml --all-targets -- -D warnings
