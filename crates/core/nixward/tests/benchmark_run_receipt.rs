//! Validation checks for the NixWard benchmark run-receipt contract.
//!
//! The checked-in example is explicitly not-run; it is not a benchmark result.

use serde_json::Value;

const RECEIPT_SCHEMA: &str = include_str!(
    "../../../../research/nixward-benchmark-v0.1/schemas/benchmark-run-receipt-v0.1.schema.json"
);
const NOT_RUN_EXAMPLE: &str = include_str!(
    "../../../../research/nixward-benchmark-v0.1/examples/run-receipt-not-run.example.json"
);

#[test]
fn run_receipt_schema_is_versioned_and_requires_provenance() {
    let schema: Value = serde_json::from_str(RECEIPT_SCHEMA).expect("receipt schema JSON");
    assert_eq!(
        schema["$schema"].as_str(),
        Some("https://json-schema.org/draft/2020-12/schema")
    );
    assert_eq!(
        schema["$id"].as_str(),
        Some("urn:luminous-dynamics:nixward:benchmark-run-receipt:0.1")
    );

    let required = schema["required"].as_array().expect("root required fields");
    for field in [
        "corpus_digest",
        "protocol_revision",
        "agent",
        "environment",
        "resource_budget",
        "tasks",
        "scorecards",
        "quarantine_records",
    ] {
        assert!(
            required.iter().any(|item| item.as_str() == Some(field)),
            "receipt schema must require {field}"
        );
    }
}

#[test]
fn not_run_example_contains_no_execution_or_measurement_claim() {
    let receipt: Value =
        serde_json::from_str(NOT_RUN_EXAMPLE).expect("not-run receipt example JSON");

    assert_eq!(receipt["schema_version"].as_str(), Some("0.1"));
    assert_eq!(receipt["receipt_status"].as_str(), Some("not_run"));
    assert_eq!(receipt["tasks"].as_array().map(Vec::len), Some(0));
    assert_eq!(receipt["scorecards"].as_array().map(Vec::len), Some(0));
    assert_eq!(receipt["corpus_digest"], Value::Null);
    assert_eq!(receipt["protocol_revision"], Value::Null);
    assert_eq!(receipt["environment"]["nixpkgs_revision"], Value::Null);
    assert_eq!(receipt["environment"]["flake_lock_digest"], Value::Null);
    assert_eq!(receipt["environment"]["module_set_digest"], Value::Null);
    assert_eq!(receipt["run_started_at"], Value::Null);
    assert_eq!(receipt["run_completed_at"], Value::Null);

    assert!(
        receipt["notes"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|note| note.as_str().is_some_and(|value| value.contains("no execution")))
    );
}

#[test]
fn task_receipts_require_distinct_execution_stages_and_effect_flags() {
    let schema: Value = serde_json::from_str(RECEIPT_SCHEMA).expect("receipt schema JSON");
    let task_required = schema["$defs"]["task"]["required"]
        .as_array()
        .expect("task required fields");
    for field in [
        "case_id",
        "cluster_id",
        "trial_index",
        "status",
        "stages",
        "effect_flags",
        "cost",
        "evidence_refs",
    ] {
        assert!(
            task_required.iter().any(|item| item.as_str() == Some(field)),
            "task receipt must require {field}"
        );
    }

    let effects = schema["$defs"]["task"]["properties"]["effect_flags"]["required"]
        .as_array()
        .expect("effect flags");
    for field in [
        "unauthorized_protected_effect",
        "secret_exposure",
        "expression_injection_accepted",
        "target_mismatch",
        "duplicate_mutation",
        "recovery_required",
        "recovery_succeeded",
    ] {
        assert!(
            effects.iter().any(|item| item.as_str() == Some(field)),
            "task receipt must track {field}"
        );
    }
}
