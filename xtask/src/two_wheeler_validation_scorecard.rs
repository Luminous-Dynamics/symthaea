//! Matched-pair energy validation scorecard for two-wheeler model evaluation.
//!
//! A pair is scoreable only when its cycle, configuration, mission, energy
//! boundary and model-input trace match the observed record. Synthetic cases
//! exercise arithmetic, never physical model validity.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const ALLOWED_STATUS: &[&str] = &["SCORE", "REJECT_INPUT"];
const PHYSICAL_EVIDENCE: &[&str] = &["InstrumentedPhysical", "FieldObserved"];

#[derive(Debug, Clone, Deserialize)]
struct Pair {
    observed_cycle_id: String,
    predicted_cycle_id: String,
    observed_configuration_digest: String,
    predicted_configuration_digest: String,
    observed_mission_digest: String,
    predicted_mission_digest: String,
    observed_energy_boundary: String,
    predicted_energy_boundary: String,
    observed_trace_sha256: String,
    model_input_trace_sha256: String,
    observed_distance_km: f64,
    predicted_distance_km: f64,
    observed_energy_wh: f64,
    predicted_energy_wh: f64,
    evidence_class: String,
    split: String,
    measurement_protocol_id: String,
    calibration_receipt_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
struct ScoreMetrics {
    observed_wh_per_km: f64,
    predicted_wh_per_km: f64,
    signed_energy_error_wh: f64,
    absolute_energy_error_wh: f64,
    signed_error_wh_per_km: f64,
    absolute_error_wh_per_km: f64,
    relative_energy_error_percent: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    manifest_id: String,
    schema_version: String,
    status: String,
    evidence_class: String,
    metric_contract: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    scenario_id: String,
    class: String,
    title: String,
    pair: Pair,
    expected_status: Option<String>,
    expected_error: Option<String>,
    expected_output: Option<ScoreMetrics>,
    absolute_tolerance: Option<f64>,
    relative_tolerance: Option<f64>,
    #[serde(default)]
    must_not_claim: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CaseResult {
    scenario_id: String,
    class: String,
    title: String,
    expected_status: Option<String>,
    computed_status: String,
    decision_matches: Option<bool>,
    expected_error: Option<String>,
    computed_error: Option<String>,
    expected_output: Option<ScoreMetrics>,
    computed_output: Option<ScoreMetrics>,
    validation_eligible: bool,
    eligibility_reasons: Vec<String>,
    must_not_claim: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EligibleSummary {
    pair_count: usize,
    mean_absolute_error_wh_per_km: Option<f64>,
    median_absolute_error_wh_per_km: Option<f64>,
    p90_absolute_error_wh_per_km: Option<f64>,
    maximum_absolute_error_wh_per_km: Option<f64>,
    maximum_error_scenario_id: Option<String>,
    mean_signed_bias_wh_per_km: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Report {
    report_version: String,
    manifest_id: String,
    manifest_sha256: String,
    evidence_class: String,
    mode: String,
    case_count: usize,
    expected_decisions_checked: usize,
    matched_cases: usize,
    mismatched_cases: usize,
    scoreable_pairs: usize,
    validation_eligible_pairs: usize,
    fixture_conformance_pass: Option<bool>,
    eligible_summary: EligibleSummary,
    claim_ceiling: String,
    cases: Vec<CaseResult>,
}

fn valid_sha256(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_pair(pair: &Pair) -> Result<(), &'static str> {
    let identities = [
        pair.observed_cycle_id.as_str(),
        pair.predicted_cycle_id.as_str(),
        pair.observed_energy_boundary.as_str(),
        pair.predicted_energy_boundary.as_str(),
        pair.measurement_protocol_id.as_str(),
    ];
    if identities.iter().any(|value| value.trim().is_empty()) {
        return Err("missing_identity");
    }
    if !valid_sha256(&pair.observed_configuration_digest)
        || !valid_sha256(&pair.predicted_configuration_digest)
        || !valid_sha256(&pair.observed_mission_digest)
        || !valid_sha256(&pair.predicted_mission_digest)
        || !valid_sha256(&pair.observed_trace_sha256)
        || !valid_sha256(&pair.model_input_trace_sha256)
    {
        return Err("invalid_digest");
    }
    if pair.observed_cycle_id != pair.predicted_cycle_id {
        return Err("cycle_mismatch");
    }
    if pair.observed_configuration_digest != pair.predicted_configuration_digest {
        return Err("configuration_mismatch");
    }
    if pair.observed_mission_digest != pair.predicted_mission_digest {
        return Err("mission_mismatch");
    }
    if pair.observed_energy_boundary != pair.predicted_energy_boundary {
        return Err("energy_boundary_mismatch");
    }
    if pair.observed_trace_sha256 != pair.model_input_trace_sha256 {
        return Err("trace_mismatch");
    }
    let numeric = [
        pair.observed_distance_km,
        pair.predicted_distance_km,
        pair.observed_energy_wh,
        pair.predicted_energy_wh,
    ];
    if numeric.iter().any(|value| !value.is_finite()) {
        return Err("non_finite_measurement");
    }
    if pair.observed_distance_km <= 0.0 || pair.predicted_distance_km <= 0.0 {
        return Err("invalid_distance");
    }
    if pair.observed_energy_wh < 0.0 || pair.predicted_energy_wh < 0.0 {
        return Err("invalid_energy");
    }
    let scale = pair.observed_distance_km.abs().max(pair.predicted_distance_km.abs());
    if (pair.observed_distance_km - pair.predicted_distance_km).abs() > 1e-6 + 1e-5 * scale {
        return Err("distance_mismatch");
    }
    Ok(())
}

fn score(pair: &Pair) -> ScoreMetrics {
    let observed_wh_per_km = pair.observed_energy_wh / pair.observed_distance_km;
    let predicted_wh_per_km = pair.predicted_energy_wh / pair.predicted_distance_km;
    let signed_energy_error_wh = pair.predicted_energy_wh - pair.observed_energy_wh;
    let signed_error_wh_per_km = predicted_wh_per_km - observed_wh_per_km;
    ScoreMetrics {
        observed_wh_per_km,
        predicted_wh_per_km,
        signed_energy_error_wh,
        absolute_energy_error_wh: signed_energy_error_wh.abs(),
        signed_error_wh_per_km,
        absolute_error_wh_per_km: signed_error_wh_per_km.abs(),
        relative_energy_error_percent: (pair.observed_energy_wh > 0.0)
            .then(|| 100.0 * signed_energy_error_wh.abs() / pair.observed_energy_wh),
    }
}

fn eligibility_reasons(pair: &Pair) -> Vec<String> {
    let mut reasons = Vec::new();
    if !PHYSICAL_EVIDENCE.contains(&pair.evidence_class.as_str()) {
        reasons.push("evidence_not_physical_measurement".into());
    }
    if pair.split != "HOLDOUT" {
        reasons.push("not_independent_holdout".into());
    }
    if pair.measurement_protocol_id.trim().is_empty() {
        reasons.push("missing_measurement_protocol".into());
    }
    if pair.evidence_class == "InstrumentedPhysical"
        && pair.calibration_receipt_id.as_deref().is_none_or(str::is_empty)
    {
        reasons.push("missing_calibration_receipt".into());
    }
    reasons
}

fn close(expected: f64, actual: f64, abs_tol: f64, rel_tol: f64) -> bool {
    (expected - actual).abs() <= abs_tol + rel_tol * expected.abs()
}

fn compare_metrics(expected: &ScoreMetrics, actual: &ScoreMetrics, abs_tol: f64, rel_tol: f64) -> bool {
    let numeric = [
        (expected.observed_wh_per_km, actual.observed_wh_per_km),
        (expected.predicted_wh_per_km, actual.predicted_wh_per_km),
        (expected.signed_energy_error_wh, actual.signed_energy_error_wh),
        (expected.absolute_energy_error_wh, actual.absolute_energy_error_wh),
        (expected.signed_error_wh_per_km, actual.signed_error_wh_per_km),
        (expected.absolute_error_wh_per_km, actual.absolute_error_wh_per_km),
    ];
    let numeric_match = numeric.iter().all(|(e, a)| close(*e, *a, abs_tol, rel_tol));
    let relative_match = match (expected.relative_energy_error_percent, actual.relative_energy_error_percent) {
        (Some(e), Some(a)) => close(e, a, abs_tol, rel_tol),
        (None, None) => true,
        _ => false,
    };
    numeric_match && relative_match
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    let fixture_mode = manifest.status == "frozen_expected_cases_not_yet_executed";
    let measured_mode = manifest.status == "measured_dataset";
    if !fixture_mode && !measured_mode {
        bail!("unsupported scorecard mode: {}", manifest.status);
    }
    if manifest.schema_version != "0.1.0" {
        bail!("unsupported validation-scorecard schema: {}", manifest.schema_version);
    }
    if fixture_mode && manifest.evidence_class != "Synthetic" {
        bail!("oracle fixture must remain Synthetic evidence");
    }
    if measured_mode && !PHYSICAL_EVIDENCE.contains(&manifest.evidence_class.as_str()) {
        bail!("measured_dataset mode requires InstrumentedPhysical or FieldObserved evidence");
    }
    if manifest.metric_contract != "matched_cycle_energy_per_distance_v0_1" {
        bail!("unsupported metric contract: {}", manifest.metric_contract);
    }
    if manifest.cases.is_empty() {
        bail!("validation scorecard contains no cases");
    }

    let mut ids = BTreeSet::new();
    let mut positive_controls = 0usize;
    let mut negative_controls = 0usize;
    for case in &manifest.cases {
        if case.scenario_id.trim().is_empty() || !ids.insert(case.scenario_id.as_str()) {
            bail!("empty or duplicate scenario ID: {}", case.scenario_id);
        }
        if case.class.trim().is_empty() {
            bail!("case {} has an empty class", case.scenario_id);
        }
        if fixture_mode {
            match case.class.as_str() {
                "positive_control" => positive_controls += 1,
                "negative_control" => negative_controls += 1,
                other => bail!("unknown fixture case class {other} in {}", case.scenario_id),
            }
            if case.must_not_claim.is_empty() {
                bail!("fixture case {} lacks a claim ceiling", case.scenario_id);
            }
            let status = case.expected_status.as_deref().unwrap_or("");
            if !ALLOWED_STATUS.contains(&status) {
                bail!("unknown or missing expected status in {}", case.scenario_id);
            }
            match status {
                "SCORE" => {
                    if case.expected_output.is_none() || case.expected_error.is_some() {
                        bail!("scored case {} needs expected metrics and no error", case.scenario_id);
                    }
                    if case.absolute_tolerance.is_none_or(|v| !v.is_finite() || v < 0.0)
                        || case.relative_tolerance.is_none_or(|v| !v.is_finite() || v < 0.0)
                    {
                        bail!("scored case {} needs finite comparison tolerances", case.scenario_id);
                    }
                }
                "REJECT_INPUT" => {
                    if case.expected_error.is_none() || case.expected_output.is_some() {
                        bail!("rejected case {} needs an error and no metrics", case.scenario_id);
                    }
                }
                _ => unreachable!("expected status validated above"),
            }
        } else {
            if case.expected_status.is_some()
                || case.expected_error.is_some()
                || case.expected_output.is_some()
                || case.absolute_tolerance.is_some()
                || case.relative_tolerance.is_some()
            {
                bail!("measured row {} must not carry frozen oracle expectations", case.scenario_id);
            }
            if case.pair.evidence_class != manifest.evidence_class {
                bail!("evidence-class mismatch in measured row {}", case.scenario_id);
            }
        }
    }
    if fixture_mode && (positive_controls == 0 || negative_controls == 0) {
        bail!("scorecard fixture requires positive and negative controls");
    }
    Ok(())
}
fn nearest_rank(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = ((p * sorted.len() as f64).ceil() as usize).max(1);
    sorted.get(rank - 1).copied()
}

fn summarize_eligible(rows: &[(String, ScoreMetrics)]) -> EligibleSummary {
    if rows.is_empty() {
        return EligibleSummary {
            pair_count: 0,
            mean_absolute_error_wh_per_km: None,
            median_absolute_error_wh_per_km: None,
            p90_absolute_error_wh_per_km: None,
            maximum_absolute_error_wh_per_km: None,
            maximum_error_scenario_id: None,
            mean_signed_bias_wh_per_km: None,
        };
    }
    let mut absolute: Vec<f64> = rows.iter().map(|(_, m)| m.absolute_error_wh_per_km).collect();
    absolute.sort_by(f64::total_cmp);
    let max_index = rows.iter().enumerate().max_by(|(_, a), (_, b)| {
        a.1.absolute_error_wh_per_km.total_cmp(&b.1.absolute_error_wh_per_km)
    }).map(|(index, _)| index);
    let maximum_error_scenario_id = max_index.map(|index| rows[index].0.clone());
    let mean_absolute_error = absolute.iter().sum::<f64>() / absolute.len() as f64;
    let mean_signed_bias = rows.iter().map(|(_, m)| m.signed_error_wh_per_km).sum::<f64>()
        / rows.len() as f64;
    EligibleSummary {
        pair_count: rows.len(),
        mean_absolute_error_wh_per_km: Some(mean_absolute_error),
        median_absolute_error_wh_per_km: nearest_rank(&absolute, 0.50),
        p90_absolute_error_wh_per_km: nearest_rank(&absolute, 0.90),
        maximum_absolute_error_wh_per_km: absolute.last().copied(),
        maximum_error_scenario_id,
        mean_signed_bias_wh_per_km: Some(mean_signed_bias),
    }
}

pub fn run(input_path: &Path, output_path: Option<&Path>) -> Result<()> {
    let raw = fs::read(input_path)
        .with_context(|| format!("reading validation scorecard {}", input_path.display()))?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&raw));
    let manifest: Manifest = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing validation scorecard {}", input_path.display()))?;
    validate_manifest(&manifest)?;
    let fixture_mode = manifest.status == "frozen_expected_cases_not_yet_executed";

    let mut results = Vec::with_capacity(manifest.cases.len());
    let mut eligible_rows = Vec::new();
    let mut scoreable_pairs = 0usize;
    for case in &manifest.cases {
        match validate_pair(&case.pair) {
            Ok(()) => {
                scoreable_pairs += 1;
                let computed = score(&case.pair);
                let numeric_match = case.expected_output.as_ref().is_some_and(|reference| {
                    compare_metrics(
                        reference,
                        &computed,
                        case.absolute_tolerance.unwrap_or(0.0),
                        case.relative_tolerance.unwrap_or(0.0),
                    )
                });
                let decision_matches = if fixture_mode {
                    Some(case.expected_status.as_deref() == Some("SCORE") && numeric_match)
                } else {
                    None
                };
                let reasons = eligibility_reasons(&case.pair);
                let eligible = reasons.is_empty();
                if eligible {
                    eligible_rows.push((case.scenario_id.clone(), computed));
                }
                results.push(CaseResult {
                    scenario_id: case.scenario_id.clone(),
                    class: case.class.clone(),
                    title: case.title.clone(),
                    expected_status: case.expected_status.clone(),
                    computed_status: "SCORE".into(),
                    decision_matches,
                    expected_error: case.expected_error.clone(),
                    computed_error: None,
                    expected_output: case.expected_output,
                    computed_output: Some(computed),
                    validation_eligible: eligible,
                    eligibility_reasons: reasons,
                    must_not_claim: case.must_not_claim.clone(),
                });
            }
            Err(error) => {
                let decision_matches = if fixture_mode {
                    Some(case.expected_status.as_deref() == Some("REJECT_INPUT")
                        && case.expected_error.as_deref() == Some(error))
                } else {
                    None
                };
                results.push(CaseResult {
                    scenario_id: case.scenario_id.clone(),
                    class: case.class.clone(),
                    title: case.title.clone(),
                    expected_status: case.expected_status.clone(),
                    computed_status: "REJECT_INPUT".into(),
                    decision_matches,
                    expected_error: case.expected_error.clone(),
                    computed_error: Some(error.into()),
                    expected_output: case.expected_output,
                    computed_output: None,
                    validation_eligible: false,
                    eligibility_reasons: vec!["pair_rejected".into()],
                    must_not_claim: case.must_not_claim.clone(),
                });
            }
        }
    }

    let expected_decisions_checked = results.iter().filter(|result| result.decision_matches.is_some()).count();
    let matched_cases = results.iter().filter(|result| result.decision_matches == Some(true)).count();
    let mismatched_cases = results.iter().filter(|result| result.decision_matches == Some(false)).count();
    let validation_eligible_pairs = eligible_rows.len();
    let report = Report {
        report_version: "0.1.0".into(),
        manifest_id: manifest.manifest_id,
        manifest_sha256,
        evidence_class: manifest.evidence_class,
        mode: manifest.status,
        case_count: results.len(),
        expected_decisions_checked,
        matched_cases,
        mismatched_cases,
        scoreable_pairs,
        validation_eligible_pairs,
        fixture_conformance_pass: fixture_mode.then_some(mismatched_cases == 0),
        eligible_summary: summarize_eligible(&eligible_rows),
        claim_ceiling: "Fixture mode proves synthetic contract conformance only. measured_dataset mode summarizes only physically measured, independently held-out, identity-matched pairs. Neither mode certifies vehicle safety, roadworthiness or regulatory compliance.".into(),
        cases: results,
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating report directory {}", parent.display()))?;
        }
        fs::write(path, format!("{encoded}\n"))
            .with_context(|| format!("writing scorecard report {}", path.display()))?;
    }
    println!("{encoded}");
    if fixture_mode && mismatched_cases != 0 {
        bail!("validation scorecard oracle mismatch: {mismatched_cases} of {expected_decisions_checked} cases differ");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../docs/research/data/two_wheeler_validation_scorecard_v0_1.json");

    fn manifest() -> Manifest {
        serde_json::from_str(FIXTURE).expect("scorecard fixture parses")
    }

    #[test]
    fn frozen_cases_match_metrics_and_rejections() {
        let manifest = manifest();
        validate_manifest(&manifest).expect("manifest is valid");
        for case in &manifest.cases {
            match validate_pair(&case.pair) {
                Ok(()) => {
                    assert_eq!(case.expected_status.as_deref(), Some("SCORE"), "{}", case.scenario_id);
                    assert!(compare_metrics(
                        case.expected_output.as_ref().expect("expected output"),
                        &score(&case.pair),
                        case.absolute_tolerance.unwrap(),
                        case.relative_tolerance.unwrap(),
                    ), "{}", case.scenario_id);
                }
                Err(error) => {
                    assert_eq!(case.expected_status.as_deref(), Some("REJECT_INPUT"), "{}", case.scenario_id);
                    assert_eq!(case.expected_error.as_deref(), Some(error), "{}", case.scenario_id);
                }
            }
        }
    }

    #[test]
    fn synthetic_and_training_rows_never_count_as_physical_holdout() {
        let mut pair = manifest().cases[0].pair.clone();
        pair.evidence_class = "Synthetic".into();
        pair.split = "TRAIN".into();
        assert!(!eligibility_reasons(&pair).is_empty());
        assert!(eligibility_reasons(&pair).contains(&"evidence_not_physical_measurement".into()));
        assert!(eligibility_reasons(&pair).contains(&"not_independent_holdout".into()));
    }

    #[test]
    fn physical_holdout_needs_calibration_for_instrumented_measurements() {
        let mut pair = manifest().cases[0].pair.clone();
        pair.evidence_class = "InstrumentedPhysical".into();
        pair.split = "HOLDOUT".into();
        pair.measurement_protocol_id = "protocol-v1".into();
        pair.calibration_receipt_id = None;
        assert!(eligibility_reasons(&pair).contains(&"missing_calibration_receipt".into()));
        pair.calibration_receipt_id = Some("calibration-receipt-123".into());
        assert!(eligibility_reasons(&pair).is_empty());
    }

    #[test]
    fn measured_dataset_mode_accepts_physical_manifest_without_oracle_expectations() {
        let mut manifest = manifest();
        manifest.status = "measured_dataset".into();
        manifest.evidence_class = "InstrumentedPhysical".into();
        for case in &mut manifest.cases {
            case.class = "measurement".into();
            case.pair.evidence_class = "InstrumentedPhysical".into();
            case.pair.calibration_receipt_id = Some("test-calibration-receipt".into());
            case.expected_status = None;
            case.expected_error = None;
            case.expected_output = None;
            case.absolute_tolerance = None;
            case.relative_tolerance = None;
            case.must_not_claim.clear();
        }
        validate_manifest(&manifest).expect("measured-dataset mode is structurally valid");
    }

    #[test]
    fn zero_measured_energy_has_no_relative_error_percentage() {
        let case = &manifest().cases[3];
        let computed = score(&case.pair);
        assert_eq!(computed.relative_energy_error_percent, None);
        assert_eq!(computed.absolute_energy_error_wh, 2.0);
    }

    #[test]
    fn nearest_rank_p90_does_not_hide_the_largest_residual() {
        let values = [0.1, 0.2, 0.3, 0.4, 4.5];
        assert_eq!(nearest_rank(&values, 0.90), Some(4.5));
    }

    #[test]
    fn empty_eligible_set_reports_no_physical_metric() {
        let summary = summarize_eligible(&[]);
        assert_eq!(summary.pair_count, 0);
        assert_eq!(summary.p90_absolute_error_wh_per_km, None);
        assert_eq!(summary.maximum_error_scenario_id, None);
    }
}
