// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::{Value, json};
use symthaea_culinary::process_trace::validate_culinary_process_trace_v1;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/process-trace-v1/synthetic-cook-batch.json"
    )).unwrap()
}

#[test]
fn fixture_schema_shape_and_summary_are_stable() {
    let result = validate_culinary_process_trace_v1(&fixture()).unwrap();
    assert_eq!(result.step_count, 2);
    assert_eq!(result.stream_count, 8);
    assert_eq!(result.boundary_input_count, 3);
    assert_eq!(result.boundary_output_count, 4);
    assert!(result.mass_balance_export_ready);
    assert!(!result.all_evidence_records_marked_qualified);
}

#[test]
fn unknown_quantity_never_becomes_zero_or_export_ready() {
    let mut trace = fixture();
    let stream = trace["streams"].as_array_mut().unwrap()
        .iter_mut().find(|s| s["stream_id"] == "evaporated-water").unwrap();
    stream["mass_g"] = json!({
        "status": "unknown", "value": null, "lower": null, "upper": null,
        "unit": "g", "unknown_reason": "not weighed in this synthetic example"
    });
    let report = validate_culinary_process_trace_v1(&trace).unwrap();
    assert!(!report.mass_balance_export_ready);
    assert!(report.blockers.iter().any(|b| b.contains("explicit unknown quantity")));
}

#[test]
fn missing_reason_for_unknown_is_invalid() {
    let mut trace = fixture();
    trace["streams"][0]["mass_g"] = json!({
        "status": "unknown", "value": null, "lower": null, "upper": null, "unit": "g"
    });
    assert!(validate_culinary_process_trace_v1(&trace).is_err());
}

#[test]
fn undeclared_stream_reference_is_invalid() {
    let mut trace = fixture();
    trace["steps"][1]["input_stream_ids"].as_array_mut().unwrap()
        .push(Value::String("no-such-stream".to_owned()));
    assert!(validate_culinary_process_trace_v1(&trace).is_err());
}

#[test]
fn stream_without_step_membership_is_invalid() {
    let mut trace = fixture();
    trace["steps"][0]["output_stream_ids"] = json!(["prepared-staple"]);
    assert!(validate_culinary_process_trace_v1(&trace).is_err());
}

#[test]
fn stream_id_must_be_unique() {
    let mut trace = fixture();
    let duplicate = trace["streams"][0].clone();
    trace["streams"].as_array_mut().unwrap().push(duplicate);
    assert!(validate_culinary_process_trace_v1(&trace).is_err());
}

#[test]
fn synthetic_evidence_cannot_claim_qualified_status() {
    let mut trace = fixture();
    trace["boundary_inventory_evidence"]["evidence_status"] = json!("qualified");
    assert!(validate_culinary_process_trace_v1(&trace).is_err());
}

#[test]
fn every_stream_requires_source_version_and_timestamp() {
    let mut trace = fixture();
    trace["streams"][0]["evidence"]["source_version"] = Value::Null;
    assert!(validate_culinary_process_trace_v1(&trace).is_err());
}

#[test]
fn invalid_mass_interval_is_rejected() {
    let mut trace = fixture();
    trace["streams"][0]["mass_g"]["lower"] = json!(110.0);
    assert!(validate_culinary_process_trace_v1(&trace).is_err());
}

#[test]
fn export_readiness_does_not_mean_mass_balance_passed() {
    let report = validate_culinary_process_trace_v1(&fixture()).unwrap();
    assert!(report.mass_balance_export_ready);
    // The adapter report deliberately exposes no exact/possible/unbalanced verdict.
}
