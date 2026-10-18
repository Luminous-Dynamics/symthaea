//! Pointwise longitudinal wheel-demand model for two-wheeler screening.
//!
//! This small reference kernel is not a validated vehicle simulator. It
//! computes signed wheel force/power from declared mass, acceleration, grade,
//! rolling resistance, aerodynamic drag and wind. It does not model a tire
//! contact patch, powertrain efficiency maps, battery dynamics, brakes,
//! regenerative energy recovery, thermal behaviour, or physical qualification.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const G: f64 = 9.80665;
const ALLOWED_STATUS: &[&str] = &["COMPUTED", "REJECT_INPUT"];

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
struct WheelDemandInput {
    mass_kg: f64,
    rotating_equivalent_mass_kg: f64,
    speed_mps: f64,
    acceleration_mps2: f64,
    grade_rad: f64,
    rolling_resistance_coefficient: f64,
    air_density_kg_m3: f64,
    drag_area_m2: f64,
    wind_speed_mps: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
struct WheelDemandOutput {
    inertial_force_n: f64,
    rolling_resistance_force_n: f64,
    grade_force_n: f64,
    aerodynamic_force_n: f64,
    wheel_force_required_n: f64,
    signed_wheel_power_w: f64,
    positive_traction_power_w: f64,
    negative_wheel_power_w: f64,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    manifest_id: String,
    schema_version: String,
    status: String,
    model_version: String,
    evidence_class: String,
    gravity_mps2: f64,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    scenario_id: String,
    class: String,
    title: String,
    input: WheelDemandInput,
    expected_status: String,
    expected_error: Option<String>,
    expected_output: Option<WheelDemandOutput>,
    absolute_tolerance: Option<f64>,
    relative_tolerance: Option<f64>,
    must_not_claim: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CaseResult {
    scenario_id: String,
    class: String,
    title: String,
    expected_status: String,
    computed_status: String,
    decision_matches: bool,
    expected_error: Option<String>,
    computed_error: Option<String>,
    expected_output: Option<WheelDemandOutput>,
    computed_output: Option<WheelDemandOutput>,
    maximum_absolute_error: Option<f64>,
    must_not_claim: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    report_version: String,
    manifest_id: String,
    manifest_sha256: String,
    model_version: String,
    evidence_class: String,
    case_count: usize,
    matched_cases: usize,
    mismatched_cases: usize,
    fixture_conformance_pass: bool,
    claim_ceiling: String,
    cases: Vec<CaseResult>,
}

fn validate_input(input: &WheelDemandInput) -> Result<(), &'static str> {
    let values = [
        input.mass_kg,
        input.rotating_equivalent_mass_kg,
        input.speed_mps,
        input.acceleration_mps2,
        input.grade_rad,
        input.rolling_resistance_coefficient,
        input.air_density_kg_m3,
        input.drag_area_m2,
        input.wind_speed_mps,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err("non_finite_input");
    }
    if input.mass_kg <= 0.0 {
        return Err("invalid_mass");
    }
    if input.rotating_equivalent_mass_kg < 0.0 {
        return Err("invalid_rotating_mass");
    }
    if input.speed_mps < 0.0 {
        return Err("negative_speed");
    }
    if !(-std::f64::consts::FRAC_PI_2..std::f64::consts::FRAC_PI_2)
        .contains(&input.grade_rad)
    {
        return Err("grade_out_of_domain");
    }
    if input.rolling_resistance_coefficient < 0.0 {
        return Err("invalid_rolling_resistance");
    }
    if input.air_density_kg_m3 <= 0.0 {
        return Err("invalid_air_density");
    }
    if input.drag_area_m2 < 0.0 {
        return Err("invalid_drag_area");
    }
    Ok(())
}

/// Calculate forward-axis force required at the wheel for a single declared
/// instant. Positive grade is uphill; positive wind speed is a tailwind.
/// Negative wheel power is reported as an overrun/braking opportunity, never
/// as recovered battery energy.
fn calculate(input: &WheelDemandInput) -> Result<WheelDemandOutput, &'static str> {
    validate_input(input)?;

    let mass = input.mass_kg;
    let equivalent_inertial_mass = mass + input.rotating_equivalent_mass_kg;
    let inertial_force = equivalent_inertial_mass * input.acceleration_mps2;

    // Rolling resistance opposes forward movement. At an exact standstill,
    // use the initial forward acceleration direction; if neither speed nor
    // forward launch exists, no rolling-resistance direction is invented.
    let rolling_direction = if input.speed_mps > 0.0 {
        1.0
    } else if input.acceleration_mps2 > 0.0 {
        1.0
    } else {
        0.0
    };
    let rolling_force = input.rolling_resistance_coefficient
        * mass
        * G
        * input.grade_rad.cos()
        * rolling_direction;
    let grade_force = mass * G * input.grade_rad.sin();

    // Relative air speed is ground speed minus signed along-track wind.
    // The signed quadratic allows a stronger tailwind to assist the vehicle.
    let relative_air_speed = input.speed_mps - input.wind_speed_mps;
    let aerodynamic_force =
        0.5 * input.air_density_kg_m3 * input.drag_area_m2
            * relative_air_speed
            * relative_air_speed.abs();

    let wheel_force = inertial_force + rolling_force + grade_force + aerodynamic_force;
    let signed_power = wheel_force * input.speed_mps;
    let output = WheelDemandOutput {
        inertial_force_n: inertial_force,
        rolling_resistance_force_n: rolling_force,
        grade_force_n: grade_force,
        aerodynamic_force_n: aerodynamic_force,
        wheel_force_required_n: wheel_force,
        signed_wheel_power_w: signed_power,
        positive_traction_power_w: signed_power.max(0.0),
        negative_wheel_power_w: (-signed_power).max(0.0),
    };

    let output_values = [
        output.inertial_force_n,
        output.rolling_resistance_force_n,
        output.grade_force_n,
        output.aerodynamic_force_n,
        output.wheel_force_required_n,
        output.signed_wheel_power_w,
        output.positive_traction_power_w,
        output.negative_wheel_power_w,
    ];
    if output_values.iter().any(|value| !value.is_finite()) {
        return Err("numeric_overflow");
    }
    Ok(output)
}

fn close(expected: f64, actual: f64, abs_tol: f64, rel_tol: f64) -> bool {
    (expected - actual).abs() <= abs_tol + rel_tol * expected.abs()
}

fn compare_outputs(expected: &WheelDemandOutput, actual: &WheelDemandOutput, abs_tol: f64, rel_tol: f64) -> (bool, f64) {
    let pairs = [
        (expected.inertial_force_n, actual.inertial_force_n),
        (expected.rolling_resistance_force_n, actual.rolling_resistance_force_n),
        (expected.grade_force_n, actual.grade_force_n),
        (expected.aerodynamic_force_n, actual.aerodynamic_force_n),
        (expected.wheel_force_required_n, actual.wheel_force_required_n),
        (expected.signed_wheel_power_w, actual.signed_wheel_power_w),
        (expected.positive_traction_power_w, actual.positive_traction_power_w),
        (expected.negative_wheel_power_w, actual.negative_wheel_power_w),
    ];
    let mut all_close = true;
    let mut maximum_absolute_error: f64 = 0.0;
    for (reference, computed) in pairs {
        let error = (reference - computed).abs();
        maximum_absolute_error = maximum_absolute_error.max(error);
        if !close(reference, computed, abs_tol, rel_tol) {
            all_close = false;
        }
    }
    (all_close, maximum_absolute_error)
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.schema_version != "0.1.0" {
        bail!("unsupported manifest schema version: {}", manifest.schema_version);
    }
    if manifest.status != "frozen_expected_cases_not_yet_executed" {
        bail!("unexpected fixture status: {}", manifest.status);
    }
    if manifest.model_version != "wheel_demand_screen_v0_1" {
        bail!("unsupported wheel-demand model version: {}", manifest.model_version);
    }
    if manifest.evidence_class != "Synthetic" {
        bail!("wheel-demand oracle fixture must remain Synthetic evidence");
    }
    if (manifest.gravity_mps2 - G).abs() > f64::EPSILON {
        bail!("fixture gravity constant does not match the model contract");
    }
    if manifest.cases.is_empty() {
        bail!("fixture has no cases");
    }

    let mut ids = BTreeSet::new();
    let mut computed_count = 0;
    let mut rejected_count = 0;
    for case in &manifest.cases {
        if case.scenario_id.trim().is_empty() || !ids.insert(case.scenario_id.as_str()) {
            bail!("empty or duplicate scenario ID: {}", case.scenario_id);
        }
        if !case.input.mass_kg.is_finite() {
            // Non-finite inputs are rejected by calculate(); this check only
            // verifies the fixture itself does not rely on non-JSON numbers.
        }
        if !ALLOWED_STATUS.contains(&case.expected_status.as_str()) {
            bail!("unknown expected status in {}", case.scenario_id);
        }
        if case.must_not_claim.is_empty() {
            bail!("case {} has no claim ceiling", case.scenario_id);
        }
        match case.expected_status.as_str() {
            "COMPUTED" => {
                computed_count += 1;
                if case.expected_output.is_none() || case.expected_error.is_some() {
                    bail!("computed case {} needs output and no error", case.scenario_id);
                }
                if case.absolute_tolerance.is_none_or(|v| !v.is_finite() || v < 0.0)
                    || case.relative_tolerance.is_none_or(|v| !v.is_finite() || v < 0.0)
                {
                    bail!("computed case {} needs valid comparison tolerances", case.scenario_id);
                }
            }
            "REJECT_INPUT" => {
                rejected_count += 1;
                if case.expected_error.is_none() || case.expected_output.is_some() {
                    bail!("rejected case {} needs error code and no output", case.scenario_id);
                }
            }
            _ => unreachable!("expected status validated above"),
        }
    }
    if computed_count == 0 || rejected_count == 0 {
        bail!("fixture needs both computed and rejected-input controls");
    }
    Ok(())
}

pub fn run(input_path: &Path, output_path: Option<&Path>) -> Result<()> {
    let raw = fs::read(input_path)
        .with_context(|| format!("reading wheel-demand fixture {}", input_path.display()))?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&raw));
    let manifest: Manifest = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing wheel-demand fixture {}", input_path.display()))?;
    validate_manifest(&manifest)?;

    let mut results = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        let calculated = calculate(&case.input);
        let (computed_status, computed_error, computed_output, error, decision_matches) =
            match calculated {
                Ok(output) => {
                    let expected = case.expected_output.as_ref();
                    if case.expected_status != "COMPUTED" || expected.is_none() {
                        ("COMPUTED".to_string(), None, Some(output), None, false)
                    } else {
                        let (matches, error) = compare_outputs(
                            expected.expect("checked above"),
                            &output,
                            case.absolute_tolerance.expect("validated above"),
                            case.relative_tolerance.expect("validated above"),
                        );
                        ("COMPUTED".to_string(), None, Some(output), Some(error), matches)
                    }
                }
                Err(code) => {
                    let matches = case.expected_status == "REJECT_INPUT"
                        && case.expected_error.as_deref() == Some(code);
                    ("REJECT_INPUT".to_string(), Some(code.to_string()), None, None, matches)
                }
            };

        results.push(CaseResult {
            scenario_id: case.scenario_id.clone(),
            class: case.class.clone(),
            title: case.title.clone(),
            expected_status: case.expected_status.clone(),
            computed_status,
            decision_matches,
            expected_error: case.expected_error.clone(),
            computed_error,
            expected_output: case.expected_output,
            computed_output,
            maximum_absolute_error: error,
            must_not_claim: case.must_not_claim.clone(),
        });
    }

    let matched_cases = results.iter().filter(|case| case.decision_matches).count();
    let mismatched_cases = results.len() - matched_cases;
    let report = Report {
        report_version: "0.1.0".to_string(),
        manifest_id: manifest.manifest_id,
        manifest_sha256,
        model_version: manifest.model_version,
        evidence_class: manifest.evidence_class,
        case_count: results.len(),
        matched_cases,
        mismatched_cases,
        fixture_conformance_pass: mismatched_cases == 0,
        claim_ceiling: "Synthetic pointwise wheel-demand oracle only; no vehicle-model validation, powertrain efficiency, battery energy, physical qualification, or roadworthiness claim.".to_string(),
        cases: results,
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating report directory {}", parent.display()))?;
        }
        fs::write(path, format!("{encoded}\n"))
            .with_context(|| format!("writing wheel-demand report {}", path.display()))?;
    }
    println!("{encoded}");

    if mismatched_cases > 0 {
        bail!(
            "wheel-demand oracle mismatch: {mismatched_cases} of {} cases differ",
            report.case_count
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../docs/research/data/two_wheeler_wheel_demand_v0_1.json");

    fn manifest() -> Manifest {
        serde_json::from_str(FIXTURE).expect("fixture parses")
    }

    #[test]
    fn frozen_wheel_demand_cases_match_expected_outputs() {
        let manifest = manifest();
        validate_manifest(&manifest).expect("manifest is valid");
        for case in &manifest.cases {
            let computed = calculate(&case.input);
            match computed {
                Ok(output) => {
                    assert_eq!(case.expected_status, "COMPUTED", "{} unexpectedly computes", case.scenario_id);
                    let expected = case.expected_output.as_ref().expect("expected output");
                    let (matches, _) = compare_outputs(
                        expected,
                        &output,
                        case.absolute_tolerance.unwrap(),
                        case.relative_tolerance.unwrap(),
                    );
                    assert!(matches, "{} ({})", case.scenario_id, case.title);
                }
                Err(error) => {
                    assert_eq!(case.expected_status, "REJECT_INPUT", "{}", case.scenario_id);
                    assert_eq!(case.expected_error.as_deref(), Some(error), "{}", case.scenario_id);
                }
            }
        }
    }

    #[test]
    fn tailwind_decreases_drag_and_headwind_increases_it() {
        let manifest = manifest();
        let tail = calculate(&manifest.cases[3].input).unwrap();
        let head = calculate(&manifest.cases[4].input).unwrap();
        assert!(tail.aerodynamic_force_n < head.aerodynamic_force_n);
    }

    #[test]
    fn downhill_overrun_is_not_counted_as_regenerated_energy() {
        let output = calculate(&manifest().cases[5].input).unwrap();
        assert!(output.signed_wheel_power_w < 0.0);
        assert_eq!(output.positive_traction_power_w, 0.0);
        assert!(output.negative_wheel_power_w > 0.0);
        // Deliberately no regenerative-energy field exists in v0.1.
    }

    #[test]
    fn standing_vehicle_can_require_force_but_has_zero_wheel_power() {
        let output = calculate(&manifest().cases[6].input).unwrap();
        assert!(output.wheel_force_required_n > 0.0);
        assert_eq!(output.signed_wheel_power_w, 0.0);
    }

    #[test]
    fn invalid_numeric_input_is_rejected() {
        let mut input = manifest().cases[0].input;
        input.wind_speed_mps = f64::NAN;
        assert_eq!(calculate(&input).unwrap_err(), "non_finite_input");
    }

    #[test]
    fn moving_aerodynamics_use_signed_relative_air_speed() {
        let mut input = manifest().cases[0].input;
        input.wind_speed_mps = 15.0; // Tailwind exceeds forward ground speed.
        let output = calculate(&input).unwrap();
        assert!(output.aerodynamic_force_n < 0.0);
    }

    #[test]
    fn zero_demand_control_produces_zero_force_and_power() {
        let output = calculate(&manifest().cases[7].input).unwrap();
        assert_eq!(output.wheel_force_required_n, 0.0);
        assert_eq!(output.signed_wheel_power_w, 0.0);
    }

    #[test]
    fn malformed_empty_fixture_is_rejected() {
        let mut empty = manifest();
        empty.cases.clear();
        assert!(validate_manifest(&empty).is_err());
    }
}
