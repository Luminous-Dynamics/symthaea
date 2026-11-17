// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Specimen-level evidence admission for Atelier Lightshell coupon testing.
//!
//! This module binds reported coupon observations to specimen, lot, process,
//! conditioning, fixture, instrument, calibration and raw-data identity.
//! It does not execute tests, infer physical truth, select materials, or
//! qualify a robot-mounted garment.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CouponCampaign {
    pub schema_version: String,
    pub campaign_id: String,
    pub state: String,
    pub doe_manifest_ref: Option<String>,
    pub specimens: Vec<Specimen>,
    pub test_runs: Vec<TestRun>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Specimen {
    pub specimen_id: String,
    pub candidate_id: String,
    pub state: String,
    pub material_lot_ref: Option<String>,
    pub construction_ref: Option<String>,
    pub process_revision_ref: Option<String>,
    pub manufactured_article_ref: Option<String>,
    pub orientation: String,
    pub width_mm: Option<f64>,
    pub length_mm: Option<f64>,
    pub thickness_mm: Option<f64>,
    pub conditioning: Conditioning,
    pub geometry_qc_state: String,
    pub geometry_qc_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conditioning {
    pub method_ref: Option<String>,
    pub environment_ref: Option<String>,
    pub temperature_c: Option<f64>,
    pub relative_humidity_pct: Option<f64>,
    pub duration_hours: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestRun {
    pub run_id: String,
    pub specimen_id: String,
    pub family: String,
    pub status: String,
    pub protocol_ref: Option<String>,
    pub fixture_ref: Option<String>,
    pub fixture_revision: Option<String>,
    pub environment_ref: Option<String>,
    pub instrument_refs: Vec<String>,
    pub calibration_refs: Vec<String>,
    pub raw_data_ref: Option<String>,
    pub raw_data_digest: Option<String>,
    pub operator_ref: Option<String>,
    pub reviewer_ref: Option<String>,
    pub acceptance_rule_ref: Option<String>,
    pub acquired_at: Option<String>,
    pub replicate_count: u32,
    pub sample_count: u32,
    pub measurements: Vec<Measurement>,
    pub response_curve: Option<ResponseCurve>,
    pub resonance_assessment: String,
    pub resonance_peaks: Vec<ResonancePeak>,
    pub failure_reason: Option<String>,
    pub invalidation_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub metric_id: String,
    pub unit: String,
    pub value: f64,
    pub uncertainty_kind: String,
    pub uncertainty_value: f64,
    pub coverage_factor: Option<f64>,
    pub replicate_count: u32,
    pub sample_count: u32,
    pub measurement_boundary: String,
    pub method_ref: Option<String>,
    pub uncertainty_model_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseCurve {
    pub kind: String,
    pub x_unit: String,
    pub y_unit: String,
    pub points: Vec<CurvePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurvePoint {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResonancePeak {
    pub frequency_hz: f64,
    pub transmissibility_ratio: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub raw_data_required_for_executed_runs: bool,
    pub calibration_required_for_executed_runs: bool,
    pub uncertainty_required_for_passed_runs: bool,
    pub failed_and_inconclusive_results_retained: bool,
    pub coupon_is_not_robot_qualification: bool,
    pub supplier_claims_are_not_independent_evidence: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub specimen_count: usize,
    pub run_count: usize,
    pub executed_run_count: usize,
    pub passed_run_count: usize,
    /// Candidates with at least one failed or inconclusive physical run.
    /// The candidate-level qualification owner must not rank these silently.
    pub adverse_candidate_ids: Vec<String>,
    /// Candidates with records invalidated due to stale or superseded evidence.
    pub invalidated_candidate_ids: Vec<String>,
    pub campaign_complete: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// Campaign completion means the recorded work is structurally complete.
    /// It is not a candidate ranking or robot qualification.
    pub fn is_complete(&self) -> bool {
        self.is_valid() && self.campaign_complete
    }

    pub fn candidate_has_adverse_run(&self, candidate_id: &str) -> bool {
        self.adverse_candidate_ids.iter().any(|id| id == candidate_id)
    }

    pub fn candidate_has_invalidated_run(&self, candidate_id: &str) -> bool {
        self.invalidated_candidate_ids.iter().any(|id| id == candidate_id)
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let campaign: CouponCampaign = serde_json::from_str(input)?;
    Ok(campaign.validate())
}

impl CouponCampaign {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut specimens: HashMap<&str, &Specimen> = HashMap::new();
        let mut run_ids = HashSet::new();
        let mut adverse_candidate_ids = HashSet::new();
        let mut invalidated_candidate_ids = HashSet::new();
        let mut executed_run_count = 0usize;
        let mut passed_run_count = 0usize;

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only coupon evidence schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.campaign_id) {
            push(&mut findings, "missing_campaign_id", "campaign_id",
                "Campaign identity must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "running" | "complete" | "invalidated") {
            push(&mut findings, "invalid_campaign_state", "state",
                "Campaign state must be proposed, running, complete or invalidated.");
        }

        if self.specimens.is_empty() {
            push(&mut findings, "no_specimens", "specimens",
                "A coupon campaign must declare at least one specimen, even while its physical identity is pending.");
        }

        for (index, specimen) in self.specimens.iter().enumerate() {
            let path = format!("specimens[{index}]");
            if is_blank(&specimen.specimen_id) || is_blank(&specimen.candidate_id) {
                push(&mut findings, "missing_specimen_identity", &path,
                    "Specimen and candidate identities must be non-empty.");
            }
            if specimens.insert(specimen.specimen_id.as_str(), specimen).is_some() {
                push(&mut findings, "duplicate_specimen_id", format!("{path}.specimen_id"),
                    "Specimen IDs must be unique within a campaign.");
            }
            if !matches!(specimen.state.as_str(), "proposed" | "available" | "rejected" | "invalidated") {
                push(&mut findings, "invalid_specimen_state", format!("{path}.state"),
                    "Unknown specimen lifecycle state.");
            }
            if !matches!(specimen.orientation.as_str(), "coursewise" | "walewise" | "face_a" | "face_b" | "unknown") {
                push(&mut findings, "invalid_specimen_orientation", format!("{path}.orientation"),
                    "Unknown specimen orientation.");
            }
            if !matches!(specimen.geometry_qc_state.as_str(), "unknown" | "passed" | "failed") {
                push(&mut findings, "invalid_geometry_qc_state", format!("{path}.geometry_qc_state"),
                    "Geometry QC state must be unknown, passed or failed.");
            }
            for (field, value) in [
                ("width_mm", specimen.width_mm),
                ("length_mm", specimen.length_mm),
                ("thickness_mm", specimen.thickness_mm),
            ] {
                if value.is_some_and(|v| !v.is_finite() || v <= 0.0) {
                    push(&mut findings, "invalid_specimen_dimension", format!("{path}.{field}"),
                        "Supplied specimen dimensions must be finite and positive.");
                }
            }
            if specimen.conditioning.temperature_c.is_some_and(|v| !v.is_finite()) {
                push(&mut findings, "invalid_conditioning_temperature", format!("{path}.conditioning.temperature_c"),
                    "Conditioning temperature must be finite.");
            }
            if specimen.conditioning.relative_humidity_pct.is_some_and(|v| !v.is_finite() || !(0.0..=100.0).contains(&v)) {
                push(&mut findings, "invalid_conditioning_humidity", format!("{path}.conditioning.relative_humidity_pct"),
                    "Relative humidity must be between 0 and 100 percent.");
            }
            if specimen.conditioning.duration_hours.is_some_and(|v| !v.is_finite() || v <= 0.0) {
                push(&mut findings, "invalid_conditioning_duration", format!("{path}.conditioning.duration_hours"),
                    "Conditioning duration must be finite and positive.");
            }

            if specimen.state == "available" {
                validate_available_specimen(specimen, &path, &mut findings);
            }
        }

        for (index, run) in self.test_runs.iter().enumerate() {
            let path = format!("test_runs[{index}]");
            if is_blank(&run.run_id) || is_blank(&run.specimen_id) {
                push(&mut findings, "missing_run_identity", &path,
                    "Run and specimen identities must be non-empty.");
            }
            if !run_ids.insert(run.run_id.as_str()) {
                push(&mut findings, "duplicate_run_id", format!("{path}.run_id"),
                    "Run IDs must be unique within a campaign.");
            }
            let specimen = specimens.get(run.specimen_id.as_str()).copied();
            if specimen.is_none() {
                push(&mut findings, "run_unknown_specimen", format!("{path}.specimen_id"),
                    "Test run references a specimen absent from this campaign.");
            }
            if !is_known_family(&run.family) {
                push(&mut findings, "unknown_test_family", format!("{path}.family"),
                    "Unknown Lightshell coupon test family.");
            }
            if !matches!(run.status.as_str(), "planned" | "passed" | "failed" | "inconclusive" | "invalidated") {
                push(&mut findings, "invalid_run_status", format!("{path}.status"),
                    "Unknown test-run status.");
            }

            if run.status == "planned" {
                if !run.measurements.is_empty() || run.response_curve.is_some() || run.raw_data_digest.is_some() {
                    push(&mut findings, "planned_run_has_observations", &path,
                        "A planned run cannot contain measured observations or raw-data claims.");
                }
                continue;
            }

            executed_run_count += 1;
            validate_executed_run(run, &path, &mut findings);
            if run.status == "passed" {
                passed_run_count += 1;
                if let Some(specimen) = specimen {
                    if specimen.state != "available" {
                        push(&mut findings, "passed_run_specimen_not_available", format!("{path}.specimen_id"),
                            "A passed physical run requires an available, identity-bound specimen.");
                    }
                }
                validate_passed_run(run, &path, &mut findings);
            } else if run.status == "failed" || run.status == "inconclusive" {
                if let Some(specimen) = specimen {
                    adverse_candidate_ids.insert(specimen.candidate_id.clone());
                }
                if is_blank_option(&run.failure_reason) {
                    push(&mut findings, "nonpass_missing_reason", format!("{path}.failure_reason"),
                        "Failed and inconclusive results must retain an explicit reason.");
                }
            } else if run.status == "invalidated" {
                if let Some(specimen) = specimen {
                    invalidated_candidate_ids.insert(specimen.candidate_id.clone());
                }
                if is_blank_option(&run.invalidation_reason) {
                    push(&mut findings, "invalidated_run_missing_reason", format!("{path}.invalidation_reason"),
                        "Invalidated results must retain an explicit invalidation reason.");
                }
            }

            if run.family == "vibration" {
                validate_vibration_run(run, &path, &mut findings);
            } else if run.response_curve.is_some() {
                validate_curve_for_family(run, &path, &mut findings);
            }
        }

        if self.state == "complete" {
            if self.specimens.iter().any(|s| s.state == "proposed") {
                push(&mut findings, "complete_campaign_has_proposed_specimen", "specimens",
                    "A complete campaign cannot retain a proposed specimen.");
            }
            if self.test_runs.is_empty() || self.test_runs.iter().any(|r| r.status == "planned") {
                push(&mut findings, "complete_campaign_has_planned_runs", "test_runs",
                    "A complete campaign must have at least one run and no planned runs.");
            }
            if executed_run_count == 0 {
                push(&mut findings, "complete_campaign_without_execution", "test_runs",
                    "A complete campaign requires at least one executed run.");
            }
        }

        if !self.evidence_policy.raw_data_required_for_executed_runs
            || !self.evidence_policy.calibration_required_for_executed_runs
            || !self.evidence_policy.uncertainty_required_for_passed_runs
            || !self.evidence_policy.failed_and_inconclusive_results_retained
            || !self.evidence_policy.coupon_is_not_robot_qualification
            || !self.evidence_policy.supplier_claims_are_not_independent_evidence
        {
            push(&mut findings, "unsafe_evidence_policy", "evidence_policy",
                "Raw data, calibration, uncertainty, adverse-result retention and claim boundaries must remain mandatory.");
        }

        let mut adverse_candidate_ids: Vec<_> = adverse_candidate_ids.into_iter().collect();
        let mut invalidated_candidate_ids: Vec<_> = invalidated_candidate_ids.into_iter().collect();
        adverse_candidate_ids.sort();
        invalidated_candidate_ids.sort();

        ValidationReport {
            findings,
            specimen_count: self.specimens.len(),
            run_count: self.test_runs.len(),
            executed_run_count,
            passed_run_count,
            adverse_candidate_ids,
            invalidated_candidate_ids,
            campaign_complete: self.state == "complete",
        }
    }
}

fn validate_available_specimen(specimen: &Specimen, path: &str, findings: &mut Vec<Finding>) {
    for (field, value) in [
        ("material_lot_ref", specimen.material_lot_ref.as_deref()),
        ("construction_ref", specimen.construction_ref.as_deref()),
        ("process_revision_ref", specimen.process_revision_ref.as_deref()),
        ("manufactured_article_ref", specimen.manufactured_article_ref.as_deref()),
        ("conditioning.method_ref", specimen.conditioning.method_ref.as_deref()),
        ("conditioning.environment_ref", specimen.conditioning.environment_ref.as_deref()),
        ("geometry_qc_ref", specimen.geometry_qc_ref.as_deref()),
    ] {
        if value.is_none_or(is_blank) {
            push(findings, "available_specimen_missing_provenance", format!("{path}.{field}"),
                "An available specimen must bind its material, process, conditioning and geometry evidence.");
        }
    }
    for (field, value) in [
        ("width_mm", specimen.width_mm),
        ("length_mm", specimen.length_mm),
        ("thickness_mm", specimen.thickness_mm),
        ("conditioning.temperature_c", specimen.conditioning.temperature_c),
        ("conditioning.relative_humidity_pct", specimen.conditioning.relative_humidity_pct),
        ("conditioning.duration_hours", specimen.conditioning.duration_hours),
    ] {
        if value.is_none() {
            push(findings, "available_specimen_missing_measurement", format!("{path}.{field}"),
                "An available specimen requires measured dimensions and conditioning values.");
        }
    }
    if specimen.orientation == "unknown" {
        push(findings, "available_specimen_orientation_unknown", format!("{path}.orientation"),
            "An available specimen must declare its cut/test orientation.");
    }
    if specimen.geometry_qc_state != "passed" {
        push(findings, "available_specimen_geometry_not_passed", format!("{path}.geometry_qc_state"),
            "An available specimen requires a passed geometry-preparation check.");
    }
}

fn validate_executed_run(run: &TestRun, path: &str, findings: &mut Vec<Finding>) {
    for (field, value) in [
        ("protocol_ref", run.protocol_ref.as_deref()),
        ("fixture_ref", run.fixture_ref.as_deref()),
        ("fixture_revision", run.fixture_revision.as_deref()),
        ("environment_ref", run.environment_ref.as_deref()),
        ("raw_data_ref", run.raw_data_ref.as_deref()),
        ("raw_data_digest", run.raw_data_digest.as_deref()),
        ("operator_ref", run.operator_ref.as_deref()),
        ("acquired_at", run.acquired_at.as_deref()),
    ] {
        if value.is_none_or(is_blank) {
            push(findings, "executed_run_missing_provenance", format!("{path}.{field}"),
                "An executed run must bind protocol, fixture, environment, raw data, operator and acquisition time.");
        }
    }
    if run.raw_data_digest.as_deref().is_some_and(|v| !is_sha256_digest(v)) {
        push(findings, "invalid_raw_data_digest", format!("{path}.raw_data_digest"),
            "Raw data digest must use sha256: followed by 64 hexadecimal characters.");
    }
    if run.instrument_refs.is_empty() || run.instrument_refs.iter().any(|value| is_blank(value)) {
        push(findings, "executed_run_missing_instrument", format!("{path}.instrument_refs"),
            "An executed run requires at least one non-empty instrument identity.");
    }
    if run.calibration_refs.is_empty() || run.calibration_refs.iter().any(|value| is_blank(value)) {
        push(findings, "executed_run_missing_calibration", format!("{path}.calibration_refs"),
            "An executed run requires non-empty calibration evidence for its instruments.");
    }
    if run.replicate_count == 0 || run.sample_count == 0 {
        push(findings, "executed_run_missing_replication", format!("{path}.replicate_count/sample_count"),
            "Executed runs require nonzero replicate and sample counts.");
    }
}

fn validate_passed_run(run: &TestRun, path: &str, findings: &mut Vec<Finding>) {
    for (field, value) in [
        ("reviewer_ref", run.reviewer_ref.as_deref()),
        ("acceptance_rule_ref", run.acceptance_rule_ref.as_deref()),
    ] {
        if value.is_none_or(is_blank) {
            push(findings, "passed_run_missing_acceptance_evidence", format!("{path}.{field}"),
                "A passed run requires reviewer and acceptance-rule references.");
        }
    }
    if run.status == "passed" {
        match run.family.as_str() {
            "compression" if run.response_curve.is_none() => push(findings, "passed_compression_without_stress_strain_curve", format!("{path}.response_curve"),
                "A passed compression run must preserve its stress-strain response curve."),
            "dynamic_impact" if run.response_curve.is_none() => push(findings, "passed_impact_without_force_time_curve", format!("{path}.response_curve"),
                "A passed dynamic-impact run must preserve its force-time response curve."),
            _ => {}
        }
    }
    if run.measurements.is_empty() {
        push(findings, "passed_run_without_measurements", format!("{path}.measurements"),
            "A passed physical run must contain at least one measured observation.");
    }
    let mut metric_ids = HashSet::new();
    for (index, measurement) in run.measurements.iter().enumerate() {
        let mpath = format!("{path}.measurements[{index}]");
        if !metric_ids.insert(measurement.metric_id.as_str()) {
            push(findings, "duplicate_measurement_metric", format!("{mpath}.metric_id"),
                "Metric IDs must be unique within one test run.");
        }
        if !metric_belongs_to_family(&run.family, &measurement.metric_id) {
            push(findings, "metric_family_mismatch", format!("{mpath}.metric_id"),
                "Measurement metric is not compatible with this test family.");
        }
        if !unit_matches_metric(&measurement.metric_id, &measurement.unit) {
            push(findings, "measurement_unit_mismatch", format!("{mpath}.unit"),
                "Measurement unit does not match the canonical unit for this metric.");
        }
        if !measurement.value.is_finite()
            || !measurement.uncertainty_value.is_finite()
            || measurement.uncertainty_value < 0.0
        {
            push(findings, "invalid_measurement_value_or_uncertainty", &mpath,
                "Observed value must be finite and uncertainty must be finite and nonnegative.");
        }
        if !matches!(measurement.uncertainty_kind.as_str(), "standard" | "expanded") {
            push(findings, "invalid_uncertainty_kind", format!("{mpath}.uncertainty_kind"),
                "Uncertainty kind must be standard or expanded.");
        }
        if measurement.uncertainty_kind == "expanded"
            && measurement.coverage_factor.is_none_or(|k| !k.is_finite() || k < 1.0)
        {
            push(findings, "expanded_uncertainty_missing_coverage_factor", format!("{mpath}.coverage_factor"),
                "Expanded uncertainty requires a finite coverage factor of at least one.");
        }
        if measurement.uncertainty_kind == "standard" && measurement.coverage_factor.is_some() {
            push(findings, "standard_uncertainty_has_coverage_factor", format!("{mpath}.coverage_factor"),
                "A standard uncertainty must not be labelled with an expanded-uncertainty coverage factor.");
        }
        if measurement.replicate_count == 0 || measurement.sample_count == 0 {
            push(findings, "measurement_missing_replication", &mpath,
                "Measurement uncertainty must bind nonzero replicate and sample counts.");
        }
        for (field, value) in [
            ("measurement_boundary", Some(measurement.measurement_boundary.as_str())),
            ("method_ref", measurement.method_ref.as_deref()),
            ("uncertainty_model_ref", measurement.uncertainty_model_ref.as_deref()),
        ] {
            if value.is_none_or(is_blank) {
                push(findings, "measurement_missing_method_or_boundary", format!("{mpath}.{field}"),
                    "A passed measurement requires an explicit boundary, method and uncertainty model.");
            }
        }
    }
}

fn validate_vibration_run(run: &TestRun, path: &str, findings: &mut Vec<Finding>) {
    if run.status != "planned" {
        match run.resonance_assessment.as_str() {
            "not_assessed" => push(findings, "vibration_resonance_not_assessed", format!("{path}.resonance_assessment"),
                "Vibration results must explicitly assess resonance peaks."),
            "none_observed" if !run.resonance_peaks.is_empty() => push(findings, "vibration_resonance_state_conflict", format!("{path}.resonance_peaks"),
                "No-peak assessment cannot include reported peaks."),
            "peaks_reported" if run.resonance_peaks.is_empty() => push(findings, "vibration_resonance_state_conflict", format!("{path}.resonance_peaks"),
                "A peaks-reported assessment must contain at least one peak."),
            "none_observed" | "peaks_reported" => {}
            _ => push(findings, "invalid_resonance_assessment", format!("{path}.resonance_assessment"),
                "Unknown vibration resonance assessment."),
        }
        if let Some(curve) = run.response_curve.as_ref() {
            validate_response_curve(curve, "frequency_response", "Hz", "ratio", path, findings);
        }
        if run.status == "passed" && run.response_curve.is_none() {
            push(findings, "passed_vibration_without_frequency_response", format!("{path}.response_curve"),
                "A passed vibration run must preserve its frequency-response curve.");
        }
        if run.resonance_assessment == "peaks_reported" {
            if let Some(curve) = run.response_curve.as_ref() {
                let min_frequency = curve.points.first().map(|point| point.x).unwrap_or(f64::NAN);
                let max_frequency = curve.points.last().map(|point| point.x).unwrap_or(f64::NAN);
                for (index, peak) in run.resonance_peaks.iter().enumerate() {
                    if peak.frequency_hz < min_frequency || peak.frequency_hz > max_frequency {
                        push(findings, "resonance_peak_outside_sweep", format!("{path}.resonance_peaks[{index}].frequency_hz"),
                            "A reported resonance peak must lie within the measured frequency sweep.");
                    }
                }
            }
        }
    }
    for (index, peak) in run.resonance_peaks.iter().enumerate() {
        if !peak.frequency_hz.is_finite() || peak.frequency_hz <= 0.0
            || !peak.transmissibility_ratio.is_finite() || peak.transmissibility_ratio < 0.0
        {
            push(findings, "invalid_resonance_peak", format!("{path}.resonance_peaks[{index}]"),
                "Resonance peak frequency must be positive and transmissibility finite and nonnegative.");
        }
    }
}

fn validate_curve_for_family(run: &TestRun, path: &str, findings: &mut Vec<Finding>) {
    let Some(curve) = run.response_curve.as_ref() else { return; };
    let expected = match run.family.as_str() {
        "compression" => Some(("stress_strain", "strain_fraction", "kPa")),
        "dynamic_impact" => Some(("impact_force_time", "s", "N")),
        "vibration" => Some(("frequency_response", "Hz", "ratio")),
        _ => None,
    };
    if let Some((kind, x_unit, y_unit)) = expected {
        validate_response_curve(curve, kind, x_unit, y_unit, path, findings);
    } else {
        push(findings, "unexpected_response_curve", format!("{path}.response_curve"),
            "This test family does not define a response-curve type in schema version 0.1.0.");
    }
}

fn validate_response_curve(
    curve: &ResponseCurve,
    expected_kind: &str,
    expected_x_unit: &str,
    expected_y_unit: &str,
    path: &str,
    findings: &mut Vec<Finding>,
) {
    if curve.kind != expected_kind || curve.x_unit != expected_x_unit || curve.y_unit != expected_y_unit {
        push(findings, "response_curve_axis_mismatch", format!("{path}.response_curve"),
            "Response-curve kind or axis units do not match the test family.");
    }
    if curve.points.len() < 3 {
        push(findings, "response_curve_too_short", format!("{path}.response_curve.points"),
            "Response curves require at least three ordered points.");
    }
    let mut previous_x: Option<f64> = None;
    for (index, point) in curve.points.iter().enumerate() {
        if !point.x.is_finite() || !point.y.is_finite() || point.y < 0.0 {
            push(findings, "invalid_response_curve_point", format!("{path}.response_curve.points[{index}]"),
                "Response-curve values must be finite and the response ordinate nonnegative.");
        }
        if previous_x.is_some_and(|previous| point.x <= previous) {
            push(findings, "response_curve_not_strictly_ordered", format!("{path}.response_curve.points[{index}].x"),
                "Response-curve abscissa values must be strictly increasing.");
        }
        if point.x < 0.0 {
            push(findings, "negative_response_curve_abscissa", format!("{path}.response_curve.points[{index}].x"),
                "Response-curve abscissa values must be nonnegative.");
        }
        if expected_x_unit == "Hz" && point.x <= 0.0 {
            push(findings, "invalid_frequency_point", format!("{path}.response_curve.points[{index}].x"),
                "Frequency-response points must have positive frequencies.");
        }
        previous_x = Some(point.x);
    }
}

fn is_known_family(value: &str) -> bool {
    matches!(value,
        "compression" | "dynamic_impact" | "vibration" | "airflow" |
        "thermal" | "flex_fatigue" | "abrasion" | "cleaning_retention")
}

fn metric_belongs_to_family(family: &str, metric: &str) -> bool {
    match family {
        "compression" => matches!(metric, "compression_energy_j_g" | "compression_stiffness_kpa" |
            "compression_set_pct" | "compression_hysteresis_fraction"),
        "dynamic_impact" => matches!(metric, "transmitted_force_n" | "impulse_ns" | "absorbed_energy_j_g"),
        "vibration" => metric == "vibration_transmissibility_ratio",
        "airflow" => metric == "air_permeability_mm_s",
        "thermal" => metric == "temperature_delta_c",
        "flex_fatigue" => metric == "flex_cycles_to_endpoint",
        "abrasion" => metric == "abrasion_cycles_to_endpoint",
        "cleaning_retention" => metric == "post_cleaning_property_retention_pct",
        _ => false,
    }
}

fn unit_matches_metric(metric: &str, unit: &str) -> bool {
    let expected = match metric {
        "compression_energy_j_g" | "absorbed_energy_j_g" => "J/g",
        "compression_stiffness_kpa" => "kPa",
        "compression_set_pct" | "post_cleaning_property_retention_pct" => "%",
        "compression_hysteresis_fraction" | "vibration_transmissibility_ratio" => "ratio",
        "transmitted_force_n" => "N",
        "impulse_ns" => "N*s",
        "air_permeability_mm_s" => "mm/s",
        "temperature_delta_c" => "degC",
        "flex_cycles_to_endpoint" | "abrasion_cycles_to_endpoint" => "cycles",
        _ => return false,
    };
    expected == unit
}

fn push(findings: &mut Vec<Finding>, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
    findings.push(Finding { code, path: path.into(), message: message.into() });
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_blank_option(value: &Option<String>) -> bool {
    value.as_deref().is_none_or(is_blank)
}

fn is_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else { return false; };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn campaign_fixture() -> Value {
        json!({
            "schema_version": "0.1.0",
            "campaign_id": "LIGHTSHELL-COUPON-SYNTHETIC-001",
            "state": "proposed",
            "doe_manifest_ref": "DOe-FIXTURE-12724",
            "specimens": [{
                "specimen_id": "SPECIMEN-001",
                "candidate_id": "A0-R",
                "state": "proposed",
                "material_lot_ref": null,
                "construction_ref": null,
                "process_revision_ref": null,
                "manufactured_article_ref": null,
                "orientation": "unknown",
                "width_mm": null,
                "length_mm": null,
                "thickness_mm": null,
                "conditioning": {
                    "method_ref": null,
                    "environment_ref": null,
                    "temperature_c": null,
                    "relative_humidity_pct": null,
                    "duration_hours": null
                },
                "geometry_qc_state": "unknown",
                "geometry_qc_ref": null
            }],
            "test_runs": [{
                "run_id": "RUN-COMPRESSION-001",
                "specimen_id": "SPECIMEN-001",
                "family": "compression",
                "status": "planned",
                "protocol_ref": null,
                "fixture_ref": null,
                "fixture_revision": null,
                "environment_ref": null,
                "instrument_refs": [],
                "calibration_refs": [],
                "raw_data_ref": null,
                "raw_data_digest": null,
                "operator_ref": null,
                "reviewer_ref": null,
                "acceptance_rule_ref": null,
                "acquired_at": null,
                "replicate_count": 0,
                "sample_count": 0,
                "measurements": [],
                "response_curve": null,
                "resonance_assessment": "not_assessed",
                "resonance_peaks": [],
                "failure_reason": null,
                "invalidation_reason": null
            },{
                "run_id": "RUN-VIBRATION-001",
                "specimen_id": "SPECIMEN-001",
                "family": "vibration",
                "status": "planned",
                "protocol_ref": null,
                "fixture_ref": null,
                "fixture_revision": null,
                "environment_ref": null,
                "instrument_refs": [],
                "calibration_refs": [],
                "raw_data_ref": null,
                "raw_data_digest": null,
                "operator_ref": null,
                "reviewer_ref": null,
                "acceptance_rule_ref": null,
                "acquired_at": null,
                "replicate_count": 0,
                "sample_count": 0,
                "measurements": [],
                "response_curve": null,
                "resonance_assessment": "not_assessed",
                "resonance_peaks": [],
                "failure_reason": null,
                "invalidation_reason": null
            }],
            "evidence_policy": {
                "raw_data_required_for_executed_runs": true,
                "calibration_required_for_executed_runs": true,
                "uncertainty_required_for_passed_runs": true,
                "failed_and_inconclusive_results_retained": true,
                "coupon_is_not_robot_qualification": true,
                "supplier_claims_are_not_independent_evidence": true
            }
        })
    }

    fn available_specimen(manifest: &mut Value) {
        let specimen = &mut manifest["specimens"][0];
        specimen["state"] = json!("available");
        specimen["material_lot_ref"] = json!("LOT-POLYESTER-001");
        specimen["construction_ref"] = json!("KNIT-SPACER-A0-REF");
        specimen["process_revision_ref"] = json!("MACHINE-PROGRAM-REV-01");
        specimen["manufactured_article_ref"] = json!("COUPON-BATCH-001");
        specimen["orientation"] = json!("coursewise");
        specimen["width_mm"] = json!(50.0);
        specimen["length_mm"] = json!(50.0);
        specimen["thickness_mm"] = json!(7.2);
        specimen["conditioning"] = json!({
            "method_ref": "COND-METHOD-01",
            "environment_ref": "ENV-23C-50RH",
            "temperature_c": 23.0,
            "relative_humidity_pct": 50.0,
            "duration_hours": 24.0
        });
        specimen["geometry_qc_state"] = json!("passed");
        specimen["geometry_qc_ref"] = json!("GEOMETRY-QC-001");
    }

    fn passed_vibration_run(manifest: &mut Value) {
        available_specimen(manifest);
        let run = &mut manifest["test_runs"][1];
        run["status"] = json!("passed");
        run["protocol_ref"] = json!("VIBRATION-PROTOCOL-REV-A");
        run["fixture_ref"] = json!("FIXTURE-SHAKER-001");
        run["fixture_revision"] = json!("REV-A");
        run["environment_ref"] = json!("ENV-23C-50RH");
        run["instrument_refs"] = json!(["ACCELEROMETER-001"]);
        run["calibration_refs"] = json!(["CAL-ACCEL-001"]);
        run["raw_data_ref"] = json!("RAW-DATA-VIB-001");
        run["raw_data_digest"] = json!(format!("sha256:{}", "a".repeat(64)));
        run["operator_ref"] = json!("OPERATOR-001");
        run["reviewer_ref"] = json!("REVIEWER-001");
        run["acceptance_rule_ref"] = json!("VIBRATION-ACCEPTANCE-REV-A");
        run["acquired_at"] = json!("2026-11-18T10:00:00+02:00");
        run["replicate_count"] = json!(3);
        run["sample_count"] = json!(3);
        run["measurements"] = json!([{
            "metric_id": "vibration_transmissibility_ratio",
            "unit": "ratio",
            "value": 0.82,
            "uncertainty_kind": "expanded",
            "uncertainty_value": 0.06,
            "coverage_factor": 2.0,
            "replicate_count": 3,
            "sample_count": 3,
            "measurement_boundary": "fixture-to-specimen transmissibility from 20 to 1000 Hz",
            "method_ref": "FRF-METHOD-REV-A",
            "uncertainty_model_ref": "UQ-MODEL-REV-A"
        }]);
        run["response_curve"] = json!({
            "kind": "frequency_response",
            "x_unit": "Hz",
            "y_unit": "ratio",
            "points": [{"x":20.0,"y":0.8},{"x":40.0,"y":0.9},{"x":80.0,"y":1.2},{"x":160.0,"y":0.7}]
        });
        run["resonance_assessment"] = json!("peaks_reported");
        run["resonance_peaks"] = json!([{"frequency_hz":80.0,"transmissibility_ratio":1.2}]);
    }

    #[test]
    fn proposed_campaign_and_planned_runs_are_admitted_not_qualified() {
        let report = parse_and_validate(&campaign_fixture().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.specimen_count, 1);
        assert_eq!(report.run_count, 2);
        assert_eq!(report.executed_run_count, 0);
        assert!(!report.is_complete());
    }

    #[test]
    fn duplicate_specimen_and_run_ids_are_rejected() {
        let mut manifest = campaign_fixture();
        let duplicate_specimen = manifest["specimens"][0].clone();
        manifest["specimens"].as_array_mut().unwrap().push(duplicate_specimen);
        let duplicate_run = manifest["test_runs"][0].clone();
        manifest["test_runs"].as_array_mut().unwrap().push(duplicate_run);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_specimen_id"));
        assert!(report.findings.iter().any(|f| f.code == "duplicate_run_id"));
    }

    #[test]
    fn available_specimen_requires_material_process_conditioning_and_geometry() {
        let mut manifest = campaign_fixture();
        available_specimen(&mut manifest);
        manifest["specimens"][0]["material_lot_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "available_specimen_missing_provenance"));
    }

    #[test]
    fn run_must_reference_known_specimen() {
        let mut manifest = campaign_fixture();
        manifest["test_runs"][0]["specimen_id"] = json!("UNKNOWN");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "run_unknown_specimen"));
    }

    #[test]
    fn passed_vibration_requires_raw_data_calibration_uncertainty_and_resonance() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.passed_run_count, 1);
    }

    #[test]
    fn passed_vibration_without_curve_is_rejected() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        manifest["test_runs"][1]["response_curve"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "passed_vibration_without_frequency_response"));
    }

    #[test]
    fn vibration_cannot_hide_an_unassessed_resonance() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        manifest["test_runs"][1]["resonance_assessment"] = json!("not_assessed");
        manifest["test_runs"][1]["resonance_peaks"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "vibration_resonance_not_assessed"));
    }

    #[test]
    fn duplicate_or_nonmonotonic_frequency_points_are_rejected() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        manifest["test_runs"][1]["response_curve"]["points"][2]["x"] = json!(40.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "response_curve_not_strictly_ordered"));
    }

    #[test]
    fn raw_data_digest_must_be_sha256() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        manifest["test_runs"][1]["raw_data_digest"] = json!("sha256:short");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "invalid_raw_data_digest"));
    }

    #[test]
    fn expanded_uncertainty_requires_coverage_factor() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        manifest["test_runs"][1]["measurements"][0]["coverage_factor"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "expanded_uncertainty_missing_coverage_factor"));
    }

    #[test]
    fn failed_and_inconclusive_runs_need_reason_and_keep_raw_evidence() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        manifest["test_runs"][1]["status"] = json!("inconclusive");
        manifest["test_runs"][1]["reviewer_ref"] = json!(null);
        manifest["test_runs"][1]["acceptance_rule_ref"] = json!(null);
        manifest["test_runs"][1]["failure_reason"] = json!("Signal-to-noise ratio below preregistered threshold.");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
    }

    #[test]
    fn adverse_run_remains_candidate_scoped_even_when_another_run_passes() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        let mut failed = manifest["test_runs"][1].clone();
        failed["run_id"] = json!("RUN-VIBRATION-FAILED-002");
        failed["status"] = json!("failed");
        failed["failure_reason"] = json!("Unexpected resonance exceeded the approved limit.");
        manifest["test_runs"].as_array_mut().unwrap().push(failed);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.is_empty(), "{:?}", report.findings);
        assert_eq!(report.passed_run_count, 1);
        assert_eq!(report.executed_run_count, 2);
        assert!(report.candidate_has_adverse_run("A0-R"));
    }

    #[test]
    fn executed_run_without_calibration_is_rejected() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        manifest["test_runs"][1]["calibration_refs"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "executed_run_missing_calibration"));
    }

    #[test]
    fn measurement_metric_and_unit_must_match_test_family() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        manifest["test_runs"][1]["measurements"][0]["metric_id"] = json!("air_permeability_mm_s");
        manifest["test_runs"][1]["measurements"][0]["unit"] = json!("mm/s");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "metric_family_mismatch"));
    }

    #[test]
    fn candidate_with_invalidated_run_is_explicitly_reported() {
        let mut manifest = campaign_fixture();
        passed_vibration_run(&mut manifest);
        let mut invalidated = manifest["test_runs"][1].clone();
        invalidated["run_id"] = json!("RUN-VIBRATION-INVALIDATED-003");
        invalidated["status"] = json!("invalidated");
        invalidated["invalidation_reason"] = json!("Material lot identity was superseded.");
        manifest["test_runs"].as_array_mut().unwrap().push(invalidated);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.candidate_has_invalidated_run("A0-R"));
    }

    #[test]
    fn campaign_without_specimens_is_rejected() {
        let mut manifest = campaign_fixture();
        manifest["specimens"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "no_specimens"));
    }

    #[test]
    fn completed_campaign_cannot_contain_planned_runs() {
        let mut manifest = campaign_fixture();
        manifest["state"] = json!("complete");
        available_specimen(&mut manifest);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "complete_campaign_has_planned_runs"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = campaign_fixture();
        manifest["unreviewed_claim"] = json!("qualified");
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}
