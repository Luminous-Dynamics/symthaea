//! Trace integration for the pointwise wheel-demand reference kernel.
//!
//! Integrates signed mechanical wheel power and distance over timestamped
//! samples. Negative wheel work is not interpreted as regenerated battery
//! energy. This is still a simplified screening model, not a vehicle digital
//! twin or a prediction of electrical/fuel energy consumption.

use super::two_wheeler_wheel_demand::{calculate, WheelDemandInput};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const ALLOWED_STATUS: &[&str] = &["COMPUTED", "REJECT_INPUT"];

#[derive(Debug, Clone, Copy, Deserialize)]
struct TraceSample {
    timestamp_s: f64,
    input: WheelDemandInput,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    manifest_id: String,
    schema_version: String,
    status: String,
    evidence_class: String,
    cases: Vec<TraceCase>,
}

#[derive(Debug, Deserialize)]
struct TraceCase {
    scenario_id: String,
    class: String,
    title: String,
    samples: Vec<TraceSample>,
    expected_status: String,
    expected_error: Option<String>,
    expected_output: Option<TraceOutput>,
    absolute_tolerance: Option<f64>,
    relative_tolerance: Option<f64>,
    must_not_claim: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
struct TraceOutput {
    duration_s: f64,
    distance_m: f64,
    positive_wheel_work_j: f64,
    negative_wheel_work_j: f64,
    net_signed_wheel_work_j: f64,
    positive_mechanical_wh_per_km: Option<f64>,
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
    expected_output: Option<TraceOutput>,
    computed_output: Option<TraceOutput>,
    maximum_absolute_error: Option<f64>,
    must_not_claim: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    report_version: String,
    manifest_id: String,
    manifest_sha256: String,
    evidence_class: String,
    case_count: usize,
    matched_cases: usize,
    mismatched_cases: usize,
    fixture_conformance_pass: bool,
    claim_ceiling: String,
    cases: Vec<CaseResult>,
}

/// Integrate linearly interpolated signed wheel-power samples. When adjacent
/// power samples straddle zero, solve for the crossing and integrate each
/// triangle separately; this avoids falsely treating the whole interval as
/// positive traction work or as negative overrun work.
fn integrate_trace(samples: &[TraceSample]) -> Result<TraceOutput, &'static str> {
    if samples.len() < 2 {
        return Err("trace_needs_two_samples");
    }
    if samples.iter().any(|sample| !sample.timestamp_s.is_finite()) {
        return Err("invalid_timestamp");
    }

    let mut points = Vec::with_capacity(samples.len());
    for sample in samples {
        let demand = calculate(&sample.input).map_err(|_| "invalid_wheel_demand_input")?;
        points.push((sample.timestamp_s, sample.input.speed_mps, demand.signed_wheel_power_w));
    }

    let mut positive_work = 0.0;
    let mut negative_work = 0.0;
    let mut net_work = 0.0;
    let mut distance = 0.0;

    for pair in points.windows(2) {
        let (t0, v0, p0) = pair[0];
        let (t1, v1, p1) = pair[1];
        let dt = t1 - t0;
        if dt <= 0.0 {
            return Err("non_monotonic_time");
        }

        match (p0 >= 0.0, p1 >= 0.0) {
            (true, true) => {
                positive_work += 0.5 * (p0 + p1) * dt;
            }
            (false, false) => {
                negative_work += 0.5 * (-p0 - p1) * dt;
            }
            (true, false) => {
                // p0 > 0 and p1 < 0: positive triangle ends at zero crossing.
                let positive_duration = dt * p0 / (p0 - p1);
                positive_work += 0.5 * p0 * positive_duration;
                negative_work += 0.5 * (-p1) * (dt - positive_duration);
            }
            (false, true) => {
                // p0 < 0 and p1 > 0: negative triangle ends at zero crossing.
                let negative_duration = dt * (-p0) / (p1 - p0);
                negative_work += 0.5 * (-p0) * negative_duration;
                positive_work += 0.5 * p1 * (dt - negative_duration);
            }
        }
        net_work += 0.5 * (p0 + p1) * dt;
        distance += 0.5 * (v0 + v1) * dt;
    }

    let duration = points.last().expect("two samples validated").0
        - points.first().expect("two samples validated").0;
    let positive_mechanical_wh_per_km = if distance > 0.0 {
        Some(positive_work / 3600.0 / (distance / 1000.0))
    } else {
        None
    };

    let values = [
        duration,
        distance,
        positive_work,
        negative_work,
        net_work,
    ];
    if values.iter().any(|value| !value.is_finite())
        || positive_mechanical_wh_per_km.is_some_and(|value| !value.is_finite())
    {
        return Err("numeric_overflow");
    }

    Ok(TraceOutput {
        duration_s: duration,
        distance_m: distance,
        positive_wheel_work_j: positive_work,
        negative_wheel_work_j: negative_work,
        net_signed_wheel_work_j: net_work,
        positive_mechanical_wh_per_km,
    })
}

fn compare_outputs(
    expected: &TraceOutput,
    computed: &TraceOutput,
    abs_tolerance: f64,
    rel_tolerance: f64,
) -> (bool, f64) {
    let mut errors = vec![
        (expected.duration_s - computed.duration_s).abs(),
        (expected.distance_m - computed.distance_m).abs(),
        (expected.positive_wheel_work_j - computed.positive_wheel_work_j).abs(),
        (expected.negative_wheel_work_j - computed.negative_wheel_work_j).abs(),
        (expected.net_signed_wheel_work_j - computed.net_signed_wheel_work_j).abs(),
    ];
    let mut close = errors
        .iter()
        .zip([
            expected.duration_s,
            expected.distance_m,
            expected.positive_wheel_work_j,
            expected.negative_wheel_work_j,
            expected.net_signed_wheel_work_j,
        ])
        .all(|(error, reference)| {
            *error <= abs_tolerance + rel_tolerance * reference.abs()
        });

    match (
        expected.positive_mechanical_wh_per_km,
        computed.positive_mechanical_wh_per_km,
    ) {
        (Some(reference), Some(actual)) => {
            let error = (reference - actual).abs();
            errors.push(error);
            close &= error <= abs_tolerance + rel_tolerance * reference.abs();
        }
        (None, None) => {}
        _ => {
            errors.push(f64::INFINITY);
            close = false;
        }
    }

    let maximum_error = errors.into_iter().fold(0.0_f64, f64::max);
    (close, maximum_error)
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.schema_version != "0.1.0" {
        bail!("unsupported trace manifest schema: {}", manifest.schema_version);
    }
    if manifest.status != "frozen_expected_cases_not_yet_executed" {
        bail!("unexpected trace fixture status: {}", manifest.status);
    }
    if manifest.evidence_class != "Synthetic" {
        bail!("trace-energy fixture must remain Synthetic evidence");
    }
    if manifest.cases.is_empty() {
        bail!("trace-energy fixture has no cases");
    }

    let mut ids = BTreeSet::new();
    let mut positive_controls = 0usize;
    let mut negative_controls = 0usize;
    for case in &manifest.cases {
        if case.scenario_id.trim().is_empty() || !ids.insert(case.scenario_id.as_str()) {
            bail!("empty or duplicate trace scenario ID: {}", case.scenario_id);
        }
        match case.class.as_str() {
            "positive_control" => positive_controls += 1,
            "negative_control" => negative_controls += 1,
            other => bail!("unknown control class {other} in {}", case.scenario_id),
        }
        if !ALLOWED_STATUS.contains(&case.expected_status.as_str()) {
            bail!("unknown expected status in {}", case.scenario_id);
        }
        if case.must_not_claim.is_empty() {
            bail!("trace case {} lacks claim ceiling", case.scenario_id);
        }
        match case.expected_status.as_str() {
            "COMPUTED" => {
                if case.expected_output.is_none() || case.expected_error.is_some() {
                    bail!("computed trace {} requires output and no error", case.scenario_id);
                }
                if case.absolute_tolerance.is_none_or(|v| !v.is_finite() || v < 0.0)
                    || case.relative_tolerance.is_none_or(|v| !v.is_finite() || v < 0.0)
                {
                    bail!("computed trace {} needs comparison tolerances", case.scenario_id);
                }
            }
            "REJECT_INPUT" => {
                if case.expected_error.is_none() || case.expected_output.is_some() {
                    bail!("rejected trace {} requires error and no output", case.scenario_id);
                }
            }
            _ => unreachable!("expected status validated above"),
        }
    }
    if positive_controls == 0 || negative_controls == 0 {
        bail!("trace fixture requires positive and negative controls");
    }
    Ok(())
}

pub fn run(input_path: &Path, output_path: Option<&Path>) -> Result<()> {
    let raw = fs::read(input_path)
        .with_context(|| format!("reading trace-energy fixture {}", input_path.display()))?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&raw));
    let manifest: Manifest = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing trace-energy fixture {}", input_path.display()))?;
    validate_manifest(&manifest)?;

    let mut results = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        match integrate_trace(&case.samples) {
            Ok(computed) => {
                let expected = case.expected_output.as_ref();
                let matches = case.expected_status == "COMPUTED"
                    && expected.is_some_and(|reference| {
                        compare_outputs(
                            reference,
                            &computed,
                            case.absolute_tolerance.expect("tolerance validated"),
                            case.relative_tolerance.expect("tolerance validated"),
                        )
                        .0
                    });
                let max_error = expected.map(|reference| {
                    compare_outputs(
                        reference,
                        &computed,
                        case.absolute_tolerance.unwrap_or(0.0),
                        case.relative_tolerance.unwrap_or(0.0),
                    )
                    .1
                });
                results.push(CaseResult {
                    scenario_id: case.scenario_id.clone(),
                    class: case.class.clone(),
                    title: case.title.clone(),
                    expected_status: case.expected_status.clone(),
                    computed_status: "COMPUTED".into(),
                    decision_matches: matches,
                    expected_error: case.expected_error.clone(),
                    computed_error: None,
                    expected_output: case.expected_output,
                    computed_output: Some(computed),
                    maximum_absolute_error: max_error,
                    must_not_claim: case.must_not_claim.clone(),
                });
            }
            Err(error) => {
                let matches = case.expected_status == "REJECT_INPUT"
                    && case.expected_error.as_deref() == Some(error);
                results.push(CaseResult {
                    scenario_id: case.scenario_id.clone(),
                    class: case.class.clone(),
                    title: case.title.clone(),
                    expected_status: case.expected_status.clone(),
                    computed_status: "REJECT_INPUT".into(),
                    decision_matches: matches,
                    expected_error: case.expected_error.clone(),
                    computed_error: Some(error.into()),
                    expected_output: case.expected_output,
                    computed_output: None,
                    maximum_absolute_error: None,
                    must_not_claim: case.must_not_claim.clone(),
                });
            }
        }
    }

    let matched_cases = results.iter().filter(|result| result.decision_matches).count();
    let mismatched_cases = results.len() - matched_cases;
    let report = Report {
        report_version: "0.1.0".into(),
        manifest_id: manifest.manifest_id,
        manifest_sha256,
        evidence_class: manifest.evidence_class,
        case_count: results.len(),
        matched_cases,
        mismatched_cases,
        fixture_conformance_pass: mismatched_cases == 0,
        claim_ceiling: "Synthetic time integration of simplified signed wheel demand only; no battery/engine energy, regenerative recovery, powertrain efficiency, physical vehicle validation or roadworthiness claim.".into(),
        cases: results,
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating report directory {}", parent.display()))?;
        }
        fs::write(path, format!("{encoded}\n"))
            .with_context(|| format!("writing trace-energy report {}", path.display()))?;
    }
    println!("{encoded}");

    if mismatched_cases != 0 {
        bail!("trace-energy oracle mismatch: {mismatched_cases} of {} cases differ", report.case_count);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../docs/research/data/two_wheeler_trace_energy_v0_1.json");

    fn manifest() -> Manifest {
        serde_json::from_str(FIXTURE).expect("trace fixture parses")
    }

    #[test]
    fn frozen_trace_cases_match_expected_outputs_and_errors() {
        let manifest = manifest();
        validate_manifest(&manifest).expect("trace manifest is valid");
        for case in &manifest.cases {
            match integrate_trace(&case.samples) {
                Ok(output) => {
                    assert_eq!(case.expected_status, "COMPUTED", "{}", case.scenario_id);
                    let (matches, _) = compare_outputs(
                        case.expected_output.as_ref().expect("expected output"),
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
    fn zero_crossing_preserves_positive_and_negative_work_separately() {
        let output = integrate_trace(&manifest().cases[1].samples).unwrap();
        assert!(output.positive_wheel_work_j > 0.0);
        assert!(output.negative_wheel_work_j > 0.0);
        assert!(
            (output.net_signed_wheel_work_j
                - (output.positive_wheel_work_j - output.negative_wheel_work_j))
                .abs()
                < 1e-8
        );
    }

    #[test]
    fn zero_distance_keeps_per_distance_metric_unknown() {
        let output = integrate_trace(&manifest().cases[3].samples).unwrap();
        assert_eq!(output.distance_m, 0.0);
        assert_eq!(output.positive_mechanical_wh_per_km, None);
    }

    #[test]
    fn invalid_timestamps_and_single_sample_are_rejected() {
        let m = manifest();
        assert_eq!(integrate_trace(&m.cases[4].samples).unwrap_err(), "non_monotonic_time");
        assert_eq!(integrate_trace(&m.cases[5].samples).unwrap_err(), "trace_needs_two_samples");
    }

    #[test]
    fn constant_trace_integrates_positive_wheel_work() {
        let output = integrate_trace(&manifest().cases[0].samples).unwrap();
        assert!((output.positive_wheel_work_j - 6322.7955).abs() < 1e-4);
        assert_eq!(output.negative_wheel_work_j, 0.0);
        assert!((output.distance_m - 100.0).abs() < 1e-9);
    }
}
