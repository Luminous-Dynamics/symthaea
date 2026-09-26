use std::collections::HashMap;

use serde_json::Value;
use sha2::{Digest, Sha256};
use symthaea_semiconductor::{
    ideal_diode_conductance, ideal_diode_current, thermal_voltage, AnalyticalDiodeProfile,
    EvaluationError, ProfileError,
};

const FIXTURE: &[u8] = include_bytes!("../fixtures/ref-001b-known-answers-v1.json");
const FIXTURE_SHA256: &str =
    "650a32d01a1f0af485dd97516d15bd446306e245e0b82aa760e1b91a806ebfa6";

fn parse_f64(value: &Value, key: &str) -> f64 {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} must be a string"))
        .parse::<f64>()
        .unwrap_or_else(|error| panic!("failed to parse {key}: {error}"))
}

fn profile_error_name(error: ProfileError) -> &'static str {
    match error {
        ProfileError::NonFiniteTemperature => "NonFiniteTemperature",
        ProfileError::NonPositiveTemperature => "NonPositiveTemperature",
        ProfileError::NonFiniteSaturationCurrent => "NonFiniteSaturationCurrent",
        ProfileError::NonPositiveSaturationCurrent => "NonPositiveSaturationCurrent",
        ProfileError::NonFiniteIdealityFactor => "NonFiniteIdealityFactor",
        ProfileError::NonPositiveIdealityFactor => "NonPositiveIdealityFactor",
        ProfileError::NonFiniteVoltageBound => "NonFiniteVoltageBound",
        ProfileError::ReversedVoltageDomain => "ReversedVoltageDomain",
    }
}

fn evaluation_error_name(error: EvaluationError) -> &'static str {
    match error {
        EvaluationError::InvalidProfile(inner) => profile_error_name(inner),
        EvaluationError::NonFiniteVoltage => "NonFiniteVoltage",
        EvaluationError::VoltageOutsideDomain { .. } => "VoltageOutsideDomain",
        EvaluationError::NumericalOverflow => "NumericalOverflow",
    }
}

fn assert_close(case_id: &str, actual: f64, case: &Value) {
    let expected = parse_f64(case, "expected");
    let abs_tolerance = parse_f64(case, "abs_tolerance");
    let rel_tolerance = parse_f64(case, "rel_tolerance");
    let delta = (actual - expected).abs();
    let allowed = abs_tolerance.max(rel_tolerance * expected.abs());
    assert!(
        delta <= allowed,
        "{case_id}: actual={actual:.17e} expected={expected:.17e} delta={delta:.3e} allowed={allowed:.3e}"
    );
}

#[test]
fn frozen_fixture_bytes_are_exact() {
    let digest = Sha256::digest(FIXTURE);
    let actual = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(actual, FIXTURE_SHA256);
}

#[test]
fn independent_known_answers_match_rust_oracle() {
    let root: Value = serde_json::from_slice(FIXTURE).expect("fixture must be valid JSON");
    assert_eq!(
        root["schema"].as_str(),
        Some("eng-semi-ref-001b-known-answers-v1")
    );

    let mut profiles = HashMap::new();
    for raw in root["profiles"].as_array().expect("profiles must be an array") {
        let id = raw["id"].as_str().expect("profile id").to_owned();
        let profile = AnalyticalDiodeProfile {
            temperature_kelvin: parse_f64(raw, "temperature_kelvin"),
            saturation_current_amps: parse_f64(raw, "saturation_current_amps"),
            ideality_factor: parse_f64(raw, "ideality_factor"),
            voltage_min_volts: parse_f64(raw, "voltage_min_volts"),
            voltage_max_volts: parse_f64(raw, "voltage_max_volts"),
        };
        profile.validate().expect("frozen profile must validate");
        assert!(profiles.insert(id, profile).is_none(), "duplicate profile id");
    }

    let cases = root["cases"].as_array().expect("cases must be an array");
    assert_eq!(cases.len(), 21);

    for case in cases {
        let case_id = case["id"].as_str().expect("case id");
        let operation = case["operation"].as_str().expect("operation");
        let expected_error = case.get("expected_error").and_then(Value::as_str);

        match operation {
            "thermal_voltage" => {
                let result = thermal_voltage(parse_f64(case, "temperature_kelvin"));
                match expected_error {
                    Some(expected) => {
                        let actual = result.expect_err("case expected an error");
                        assert_eq!(profile_error_name(actual), expected, "{case_id}");
                    }
                    None => assert_close(
                        case_id,
                        result.expect("numeric thermal-voltage case"),
                        case,
                    ),
                }
            }
            "current" | "conductance" => {
                let profile_id = case["profile_id"].as_str().expect("profile_id");
                let profile = *profiles.get(profile_id).expect("known profile id");
                let voltage = parse_f64(case, "voltage_volts");
                let result = if operation == "current" {
                    ideal_diode_current(profile, voltage)
                } else {
                    ideal_diode_conductance(profile, voltage)
                };

                match expected_error {
                    Some(expected) => {
                        let actual = result.expect_err("case expected an error");
                        assert_eq!(evaluation_error_name(actual), expected, "{case_id}");
                    }
                    None => assert_close(case_id, result.expect("numeric diode case"), case),
                }
            }
            other => panic!("{case_id}: unknown operation {other}"),
        }
    }
}
