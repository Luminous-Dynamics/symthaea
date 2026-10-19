// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use symthaea_culinary::thermal_observation::validate_thermal_observation_log_v1;

fn fixture_trace() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-trace.json"
    )).unwrap()
}

fn fixture_log() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/thermal-observation-v1/synthetic-boil-temperature-log.json"
    )).unwrap()
}

#[test]
fn trace_and_observation_log_are_joined() {
    let report = validate_thermal_observation_log_v1(&fixture_trace(), &fixture_log()).unwrap();
    assert_eq!(report.observation_count, 3);
    assert_eq!(report.estimated_count, 3);
    assert_eq!(report.unknown_count, 0);
}

#[test]
fn unknown_reading_is_explicit_not_a_numeric_default() {
    let mut log = fixture_log();
    log["observations"][1]["temperature_c"] = serde_json::json!({
        "status":"unknown","value_c":null,"lower_c":null,"upper_c":null,
        "unit":"degC","unknown_reason":"sample missing"
    });
    let report = validate_thermal_observation_log_v1(&fixture_trace(), &log).unwrap();
    assert_eq!(report.unknown_count, 1);
    assert!(!report.blockers.is_empty());
}

#[test]
fn observation_reference_to_unmodeled_step_is_rejected() {
    let mut log = fixture_log();
    log["observations"][0]["step_id"] = Value::String("not-in-trace".into());
    assert!(validate_thermal_observation_log_v1(&fixture_trace(), &log).is_err());
}

#[test]
fn synthetic_calibration_cannot_support_calibrated_status() {
    let mut log = fixture_log();
    log["sensors"][0]["calibration_status"] = Value::String("calibrated".into());
    log["sensors"][0]["last_calibrated_at"] = Value::String("2026-10-18T00:00:00Z".into());
    assert!(validate_thermal_observation_log_v1(&fixture_trace(), &log).is_err());
}

#[test]
fn result_type_is_not_a_safety_assessment() {
    let log = fixture_log();
    let result = validate_thermal_observation_log_v1(&fixture_trace(), &log).unwrap();
    assert!(!format!("{result:?}").contains("safe"));
}
