use symthaea_quantum_comp::{HDQF_PILOT_EXPERIMENT_SCHEMA, known_schema_labels};

const PREREGISTRATION: &str = include_str!("../docs/HDQF_PILOT_PREREGISTRATION.md");
const JSON_SCHEMA: &str = include_str!("../schemas/hdqf-pilot-experiment-v1.schema.json");
const SMOKE_MANIFEST: &str = include_str!("../fixtures/hdqf-pilot-smoke.json");

#[test]
fn hdqf_schema_label_is_registered_and_consistent() {
    assert!(known_schema_labels().contains(&HDQF_PILOT_EXPERIMENT_SCHEMA));
    assert!(JSON_SCHEMA.contains(HDQF_PILOT_EXPERIMENT_SCHEMA));
    assert!(SMOKE_MANIFEST.contains(HDQF_PILOT_EXPERIMENT_SCHEMA));
}

#[test]
fn hdqf_artifacts_share_the_frozen_protocol_identity() {
    for artifact in [PREREGISTRATION, JSON_SCHEMA, SMOKE_MANIFEST] {
        assert!(artifact.contains("symthaea-hdqf-pilot-2026-01"));
        assert!(artifact.contains("2026-07-13"));
    }
}

#[test]
fn hdqf_preregistration_preserves_claim_and_accounting_boundaries() {
    assert!(PREREGISTRATION.contains("structure-aware classical"));
    assert!(PREREGISTRATION.contains("Explicit reversible ROM"));
    assert!(PREREGISTRATION.contains("resource_censored"));
    assert!(PREREGISTRATION.contains("Simulator wall-clock performance"));
    assert!(SMOKE_MANIFEST.contains("controlled_hypervector_factorization_only"));
    assert!(SMOKE_MANIFEST.contains("simulator_time_is_advantage_evidence\": false"));
}

#[test]
fn hdqf_schema_requires_core_pilot_axes() {
    for field in [
        "dimension_d",
        "factor_count_f",
        "codebook_size_n",
        "mu",
        "epsilon",
        "instance_family",
        "reuse_count_r",
        "evidence_level",
        "pareto_coordinates",
    ] {
        assert!(JSON_SCHEMA.contains(&format!("\"{field}\"")));
    }
}
