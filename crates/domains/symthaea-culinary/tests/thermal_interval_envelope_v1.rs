// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use symthaea_culinary::thermal_interval::{
    evaluate_thermal_interval_envelope_v1, ThermalKineticsModelV1,
};

fn trace() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-trace.json"
    ))
    .unwrap()
}

fn log_fixture() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/thermal-observation-v1/synthetic-boil-temperature-log.json"
    ))
    .unwrap()
}

fn model() -> ThermalKineticsModelV1 {
    ThermalKineticsModelV1 {
        model_id: "synthetic-dz-oracle-v1".into(),
        target_organism: "synthetic-organism".into(),
        food_matrix: "synthetic-liquid-matrix".into(),
        d_ref_minutes: 1.0,
        t_ref_c: 90.0,
        z_c: 10.0,
        evidence: serde_json::json!({
            "source_ref": "urn:synthetic:dz-parameters",
            "source_version": "fixture-v1",
            "known_at": "2026-10-19T00:00:00Z",
            "evidence_status": "synthetic"
        }),
    }
}

fn evaluate(log: &Value, model: &ThermalKineticsModelV1) -> symthaea_culinary::thermal_interval::ThermalIntervalEnvelopeV1 {
    evaluate_thermal_interval_envelope_v1(
        &trace(),
        log,
        "synthetic-pot-probe",
        "process_medium_temperature",
        model,
    )
    .unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.15e}, expected={expected:.15e}, tolerance={tolerance:.3e}"
    );
}

// Independent closed form: the integral of an exponential segment is the
// logarithmic mean of endpoint rates times duration. This differs algebraically
// from the production expm1(x)/x form.
fn oracle_log10_reduction(
    points: &[(f64, f64)],
    d_ref_minutes: f64,
    t_ref_c: f64,
    z_c: f64,
) -> f64 {
    let rate = |temperature_c: f64| {
        10.0_f64.powf((temperature_c - t_ref_c) / z_c) / d_ref_minutes
    };
    let mut total = 0.0;
    for pair in points.windows(2) {
        let (t0, temp0) = pair[0];
        let (t1, temp1) = pair[1];
        let r0 = rate(temp0);
        let r1 = rate(temp1);
        let dt = t1 - t0;
        let contribution = if (r1 - r0).abs() <= f64::EPSILON * r0.abs().max(r1.abs()) {
            dt * r0
        } else {
            dt * (r1 - r0) / (r1.ln() - r0.ln())
        };
        total += contribution;
    }
    total
}

#[test]
fn interval_envelope_matches_independent_endpoint_rate_oracle() {
    let result = evaluate(&log_fixture(), &model());
    let lower = [(0.0, 20.0), (5.0, 72.0), (10.0, 92.0)];
    let nominal = [(0.0, 22.0), (5.0, 75.0), (10.0, 95.0)];
    let upper = [(0.0, 24.0), (5.0, 78.0), (10.0, 98.0)];
    let expected_lower = oracle_log10_reduction(&lower, 1.0, 90.0, 10.0);
    let expected_nominal = oracle_log10_reduction(&nominal, 1.0, 90.0, 10.0);
    let expected_upper = oracle_log10_reduction(&upper, 1.0, 90.0, 10.0);

    close(result.lower_log10_reduction, expected_lower, 1e-9);
    close(result.nominal_log10_reduction, expected_nominal, 1e-9);
    close(result.upper_log10_reduction, expected_upper, 1e-9);
    assert!(result.lower_log10_reduction <= result.nominal_log10_reduction);
    assert!(result.nominal_log10_reduction <= result.upper_log10_reduction);
    assert_eq!(result.sample_count, 3);
    assert_eq!(result.estimated_count, 3);
    assert_eq!(result.observed_count, 0);
    close(result.elapsed_minutes, 10.0, 1e-12);
}

#[test]
fn peak_temperature_bounds_remain_ordered_and_not_collapsed() {
    let result = evaluate(&log_fixture(), &model());
    assert_eq!(result.lower_peak_temperature_c, 92.0);
    assert_eq!(result.nominal_peak_temperature_c, 95.0);
    assert_eq!(result.upper_peak_temperature_c, 98.0);
    assert!(result.lower_peak_temperature_c < result.nominal_peak_temperature_c);
    assert!(result.nominal_peak_temperature_c < result.upper_peak_temperature_c);
}

#[test]
fn unqualified_sensor_and_synthetic_model_are_reported_as_blockers() {
    let result = evaluate(&log_fixture(), &model());
    assert!(result.qualification_blockers.iter().any(|blocker| blocker.contains("calibration status")));
    assert!(result.qualification_blockers.iter().any(|blocker| blocker.contains("synthetic")));
    assert!(result.qualification_blockers.iter().any(|blocker| blocker.contains("not independently validated")));
    assert!(result.assumptions.iter().any(|assumption| assumption.contains("not a food-safety")));
}

#[test]
fn observation_order_permutation_does_not_change_result() {
    let original = log_fixture();
    let expected = evaluate(&original, &model());
    let mut reversed = original;
    reversed["observations"].as_array_mut().unwrap().reverse();
    let actual = evaluate(&reversed, &model());
    assert_eq!(actual, expected);
}

#[test]
fn selected_unknown_temperature_fails_closed() {
    let mut log = log_fixture();
    log["observations"][1]["temperature_c"] = serde_json::json!({
        "status": "unknown",
        "value_c": null,
        "lower_c": null,
        "upper_c": null,
        "unit": "degC",
        "unknown_reason": "synthetic sensor dropout"
    });
    assert!(evaluate_thermal_interval_envelope_v1(
        &trace(),
        &log,
        "synthetic-pot-probe",
        "process_medium_temperature",
        &model(),
    ).is_err());
}

#[test]
fn moving_sensor_location_requires_a_separate_series() {
    let mut log = log_fixture();
    log["observations"][1]["location_description"] =
        Value::String("synthetic fixture; moved to vessel wall".into());
    let error = evaluate_thermal_interval_envelope_v1(
        &trace(),
        &log,
        "synthetic-pot-probe",
        "process_medium_temperature",
        &model(),
    )
    .unwrap_err();
    assert!(error.iter().any(|message| message.contains("changes location_description")));
}

#[test]
fn widening_declared_temperature_bounds_widens_the_integral_envelope() {
    let original = log_fixture();
    let before = evaluate(&original, &model());
    let mut wider = original;
    wider["observations"][1]["temperature_c"]["lower_c"] = serde_json::json!(60.0);
    wider["observations"][1]["temperature_c"]["upper_c"] = serde_json::json!(85.0);
    let after = evaluate(&wider, &model());
    assert!(after.lower_log10_reduction < before.lower_log10_reduction);
    assert!(after.upper_log10_reduction > before.upper_log10_reduction);
}

#[test]
fn shifting_every_temperature_by_one_z_scales_dose_by_ten() {
    let original = log_fixture();
    let before = evaluate(&original, &model());
    let mut warmer = original;
    for observation in warmer["observations"].as_array_mut().unwrap() {
        for key in ["value_c", "lower_c", "upper_c"] {
            let value = observation["temperature_c"][key].as_f64().unwrap();
            observation["temperature_c"][key] = serde_json::json!(value + 10.0);
        }
    }
    let after = evaluate(&warmer, &model());
    close(after.lower_log10_reduction, before.lower_log10_reduction * 10.0, 1e-8);
    close(after.nominal_log10_reduction, before.nominal_log10_reduction * 10.0, 1e-8);
    close(after.upper_log10_reduction, before.upper_log10_reduction * 10.0, 1e-8);
}

#[test]
fn synthetic_kinetics_evidence_cannot_be_labelled_qualified() {
    let mut synthetic_claim = model();
    synthetic_claim.evidence["evidence_status"] = Value::String("qualified".into());
    let error = evaluate_thermal_interval_envelope_v1(
        &trace(),
        &log_fixture(),
        "synthetic-pot-probe",
        "process_medium_temperature",
        &synthetic_claim,
    )
    .unwrap_err();
    assert!(error.iter().any(|message| message.contains("cannot be marked qualified")));
}

#[test]
fn invalid_dz_parameters_and_invalid_calendar_timestamp_fail_closed() {
    let mut invalid_model = model();
    invalid_model.z_c = 0.0;
    assert!(evaluate_thermal_interval_envelope_v1(
        &trace(),
        &log_fixture(),
        "synthetic-pot-probe",
        "process_medium_temperature",
        &invalid_model,
    )
    .is_err());

    let mut invalid_time = log_fixture();
    invalid_time["observations"][1]["measured_at"] =
        Value::String("2026-19-19T00:05:00Z".into());
    assert!(evaluate_thermal_interval_envelope_v1(
        &trace(),
        &invalid_time,
        "synthetic-pot-probe",
        "process_medium_temperature",
        &model(),
    )
    .is_err());
}

#[test]
fn selected_sensor_and_measurement_kind_are_explicit() {
    let log = log_fixture();
    assert!(evaluate_thermal_interval_envelope_v1(
        &trace(),
        &log,
        "not-present",
        "process_medium_temperature",
        &model(),
    )
    .is_err());
    assert!(evaluate_thermal_interval_envelope_v1(
        &trace(),
        &log,
        "synthetic-pot-probe",
        "unclassified",
        &model(),
    )
    .is_err());
}


#[test]
fn microsecond_timestamp_spacing_is_not_lost_to_epoch_float_precision() {
    let mut log = log_fixture();
    log["observations"][0]["measured_at"] =
        Value::String("2026-10-19T00:00:00.000001Z".into());
    log["observations"][1]["measured_at"] =
        Value::String("2026-10-19T00:00:00.000002Z".into());
    log["observations"][2]["measured_at"] =
        Value::String("2026-10-19T00:00:00.000003Z".into());

    let result = evaluate(&log, &model());
    close(result.elapsed_minutes, 2e-6 / 60.0, 1e-14);
    assert!(result.nominal_log10_reduction.is_finite());
}
