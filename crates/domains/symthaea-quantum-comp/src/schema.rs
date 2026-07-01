//! Versioned schema labels for exported alpha reports.
//!
//! These constants give downstream scripts stable labels without pulling in a
//! serialization framework. They are not a replacement for formal schemas.

/// Crate package version at compile time.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Schema label for binding probe reports.
pub const BINDING_PROBE_SCHEMA: &str = "symthaea.quantum_comp.binding_probe.v0.alpha10";

/// Schema label for noise sweep reports.
pub const NOISE_SWEEP_SCHEMA: &str = "symthaea.quantum_comp.noise_sweep.v0.alpha10";

/// Schema label for experiment matrix reports.
pub const EXPERIMENT_MATRIX_SCHEMA: &str = "symthaea.quantum_comp.experiment_matrix.v0.alpha10";

/// Schema label for local research receipts.
pub const RESEARCH_RECEIPT_SCHEMA: &str = "symthaea.quantum_comp.research_receipt.v0.alpha10";

/// Schema label for local research bundles.
pub const RESEARCH_BUNDLE_SCHEMA: &str = "symthaea.quantum_comp.research_bundle.v0.alpha10";

/// Schema label for replay plans.
pub const REPLAY_PLAN_SCHEMA: &str = "symthaea.quantum_comp.replay_plan.v0.alpha10";

/// Schema label for local release-gate reports.
pub const RELEASE_GATE_SCHEMA: &str = "symthaea.quantum_comp.release_gate.v0.alpha10";

/// Schema label for fixture catalogs.
pub const FIXTURE_CATALOG_SCHEMA: &str = "symthaea.quantum_comp.fixture_catalog.v0.alpha10";

/// Schema label for integration-boundary declarations.
pub const INTEGRATION_DECLARATION_SCHEMA: &str =
    "symthaea.quantum_comp.integration_declaration.v0.alpha10";

/// Returns all known alpha.10 schema labels.
pub fn known_schema_labels() -> &'static [&'static str] {
    &[
        BINDING_PROBE_SCHEMA,
        NOISE_SWEEP_SCHEMA,
        EXPERIMENT_MATRIX_SCHEMA,
        RESEARCH_RECEIPT_SCHEMA,
        RESEARCH_BUNDLE_SCHEMA,
        REPLAY_PLAN_SCHEMA,
        RELEASE_GATE_SCHEMA,
        FIXTURE_CATALOG_SCHEMA,
        INTEGRATION_DECLARATION_SCHEMA,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_labels_are_namespaced() {
        for label in known_schema_labels() {
            assert!(label.starts_with("symthaea.quantum_comp."));
            assert!(label.ends_with("alpha10"));
        }
    }
}
