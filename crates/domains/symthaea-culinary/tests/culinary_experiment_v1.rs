// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use symthaea_culinary::culinary_experiment::validate_culinary_experiment_v1;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/culinary-experiment-v1/synthetic-paired-cooking-study.json"
    ))
    .unwrap()
}

#[test]
fn synthetic_paired_study_is_structurally_valid_but_not_empirically_ready() {
    let report = validate_culinary_experiment_v1(&fixture()).unwrap();
    assert_eq!(report.block_count, 2);
    assert_eq!(report.run_count, 4);
    assert_eq!(report.measurement_count, 12);
    assert_eq!(report.estimated_count, 12);
    assert_eq!(report.observed_count, 0);
    assert!(!report.all_evidence_records_marked_qualified);
    assert!(!report.paired_analysis_ready);
    assert!(report.blockers.iter().any(|item| item.contains("synthetic")));
}

#[test]
fn randomized_order_must_match_recorded_execution_order() {
    let mut value = fixture();
    value["blocks"][0]["runs"][0]["condition_id"] = Value::String("variant".into());
    assert!(validate_culinary_experiment_v1(&value).is_err());
}

#[test]
fn each_block_requires_one_run_per_condition() {
    let mut value = fixture();
    value["blocks"][0]["runs"][1]["condition_id"] = Value::String("baseline".into());
    value["blocks"][0]["randomized_order"] =
        serde_json::json!(["baseline", "baseline"]);
    assert!(validate_culinary_experiment_v1(&value).is_err());
}

#[test]
fn preregistration_must_precede_every_run() {
    let mut value = fixture();
    value["preregistered_at"] = Value::String("2026-10-19T00:15:00Z".into());
    assert!(validate_culinary_experiment_v1(&value).is_err());
}

#[test]
fn protocol_digest_is_required_and_sha256_shaped() {
    let mut value = fixture();
    value["protocol_sha256"] = Value::String("not-a-digest".into());
    assert!(validate_culinary_experiment_v1(&value).is_err());
}

#[test]
fn every_run_must_contain_every_declared_outcome() {
    let mut value = fixture();
    value["blocks"][0]["runs"][0]["outcomes"].as_array_mut().unwrap().pop();
    assert!(validate_culinary_experiment_v1(&value).is_err());
}

#[test]
fn outcome_units_must_match_the_preregistered_definition() {
    let mut value = fixture();
    value["blocks"][0]["runs"][0]["outcomes"][0]["unit"] = Value::String("kg".into());
    assert!(validate_culinary_experiment_v1(&value).is_err());
}

#[test]
fn unknown_values_are_preserved_as_unknown_and_block_analysis() {
    let mut value = fixture();
    value["blocks"][0]["runs"][0]["outcomes"][0]["status"] = Value::String("unknown".into());
    value["blocks"][0]["runs"][0]["outcomes"][0]["value"] = Value::Null;
    value["blocks"][0]["runs"][0]["outcomes"][0]["lower"] = Value::Null;
    value["blocks"][0]["runs"][0]["outcomes"][0]["upper"] = Value::Null;
    value["blocks"][0]["runs"][0]["outcomes"][0]["unknown_reason"] =
        Value::String("synthetic scale unavailable".into());
    let report = validate_culinary_experiment_v1(&value).unwrap();
    assert_eq!(report.unknown_count, 1);
    assert!(!report.paired_analysis_ready);
    assert!(report.blockers.iter().any(|item| item.contains("explicitly unknown")));
}

#[test]
fn inverted_uncertainty_interval_is_rejected() {
    let mut value = fixture();
    value["blocks"][0]["runs"][0]["outcomes"][0]["lower"] = serde_json::json!(300.0);
    assert!(validate_culinary_experiment_v1(&value).is_err());
}

#[test]
fn duplicate_run_ids_across_blocks_are_rejected() {
    let mut value = fixture();
    value["blocks"][1]["runs"][0]["run_id"] = value["blocks"][0]["runs"][0]["run_id"].clone();
    assert!(validate_culinary_experiment_v1(&value).is_err());
}

#[test]
fn synthetic_source_cannot_be_promoted_to_qualified() {
    let mut value = fixture();
    value["blocks"][0]["runs"][0]["outcomes"][0]["evidence"]["evidence_status"] =
        Value::String("qualified".into());
    assert!(validate_culinary_experiment_v1(&value).is_err());
}

#[test]
fn timestamp_ordering_preserves_nanosecond_resolution() {
    let mut value = fixture();
    value["preregistered_at"] = Value::String("2026-10-19T00:00:00.000000001Z".into());
    value["blocks"][0]["runs"][0]["started_at"] =
        Value::String("2026-10-19T00:00:00.000000002Z".into());
    value["blocks"][0]["runs"][0]["ended_at"] =
        Value::String("2026-10-19T00:00:00.000000003Z".into());
    assert!(validate_culinary_experiment_v1(&value).is_ok());
}

#[test]
fn non_string_entry_in_randomized_order_is_rejected() {
    let mut value = fixture();
    value["blocks"][0]["randomized_order"] = serde_json::json!(["baseline", 7]);
    assert!(validate_culinary_experiment_v1(&value).is_err());
}
