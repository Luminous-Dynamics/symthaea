// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-only spatial plasma exposure-map provenance and coverage audit.
//!
//! This validates coordinate, geometry, region, calibration, timing and
//! uncertainty semantics. It does not infer plasma dose or surface chemistry.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const EVIDENCE_CLASS: &str = "SYNTHETIC";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Disposition {
    Pass,
    Blocked,
    Rejected,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    benchmark_id: String,
    evidence_class: String,
    synthetic_only: bool,
    physical_measurements: bool,
    plasma_solver_outputs: bool,
    automatic_field_acceptance: bool,
    physical_actuation_authorized: bool,
    claim_ceiling: String,
    thresholds: Thresholds,
    required_sensor_channels: Vec<String>,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Thresholds {
    maximum_clock_offset_ms: u64,
    maximum_transform_residual_mm: f64,
    maximum_interpolation_uncertainty_mm: f64,
    required_calibration_status: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    sample_lineage_ref: String,
    material_interface_ref: String,
    geometry_id: String,
    geometry_revision: String,
    geometry_digest: String,
    expected_geometry_digest: String,
    tool_id: String,
    process_configuration_digest: String,
    expected_process_configuration_digest: String,
    trajectory_id: String,
    coordinate_frame: String,
    transform_receipts: Vec<TransformReceipt>,
    run_timestamp_unix_ms: u64,
    required_region_ids: Vec<String>,
    calibration_records: Vec<CalibrationRecord>,
    regions: Vec<Region>,
    expected: Expected,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransformReceipt {
    transform_id: String,
    source_frame: String,
    target_frame: String,
    calibration_record_id: String,
    residual_mm: f64,
    raw_trace_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationRecord {
    channel_id: String,
    sensor_id: String,
    calibration_record_id: String,
    calibration_status: String,
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
    clock_offset_ms: i64,
    raw_trace_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Region {
    region_id: String,
    status: String,
    article_frame_position_mm: Vec<f64>,
    local_standoff_mm: Option<f64>,
    local_ground_distance_mm: Option<f64>,
    incidence_angle_deg: Option<f64>,
    relative_speed_mm_s: Option<f64>,
    dwell_ms: Option<f64>,
    electrical_power_w: Option<f64>,
    optical_signal_au: Option<f64>,
    interpolation_uncertainty_mm: Option<f64>,
    geometry_transform_id: Option<String>,
    electrical_calibration_record_id: Option<String>,
    optical_calibration_record_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    disposition: Disposition,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub disposition: Disposition,
    pub expected_disposition: Disposition,
    pub reasons: Vec<String>,
    pub required_region_count: usize,
    pub observed_region_count: usize,
    pub coverage_fraction: f64,
    pub maximum_interpolation_uncertainty_mm: Option<f64>,
    pub maximum_clock_offset_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub schema_version: String,
    pub benchmark_id: String,
    pub evidence_class: String,
    pub synthetic_only: bool,
    pub physical_measurements: bool,
    pub plasma_solver_outputs: bool,
    pub automatic_field_acceptance: bool,
    pub physical_actuation_authorized: bool,
    pub scenario_count: usize,
    pub matched_expectations: usize,
    pub mismatched_expectations: usize,
    pub pass_count: usize,
    pub blocked_count: usize,
    pub rejected_count: usize,
    pub results: Vec<ScenarioResult>,
    pub claim_ceiling: String,
}

pub fn run_audit(path: &Path) -> Result<AuditReport> {
    let bytes = fs::read(path)
        .with_context(|| format!("read plasma exposure-map fixture {}", path.display()))?;
    let input = std::str::from_utf8(&bytes).context("exposure-map fixture is not UTF-8")?;
    audit_fixture_json(input).map_err(anyhow::Error::msg)
}

pub fn audit_fixture_json(input: &str) -> std::result::Result<AuditReport, String> {
    let fixture: Fixture =
        serde_json::from_str(input).map_err(|error| format!("invalid exposure-map fixture: {error}"))?;

    if fixture.schema_version != "0.1.0" || fixture.benchmark_id != "PLASMA-EXPOSURE-MAP-001" {
        return Err("unsupported exposure-map schema or benchmark ID".to_owned());
    }
    if fixture.evidence_class != EVIDENCE_CLASS
        || !fixture.synthetic_only
        || fixture.physical_measurements
        || fixture.plasma_solver_outputs
        || fixture.automatic_field_acceptance
        || fixture.physical_actuation_authorized
    {
        return Err("fixture must remain synthetic-only and non-authoritative".to_owned());
    }
    if fixture.claim_ceiling.trim().is_empty()
        || fixture.required_sensor_channels.is_empty()
        || fixture.scenarios.is_empty()
    {
        return Err("claim ceiling, required sensor channels and scenarios must be non-empty".to_owned());
    }
    if fixture.thresholds.maximum_clock_offset_ms == 0
        || !fixture.thresholds.maximum_transform_residual_mm.is_finite()
        || fixture.thresholds.maximum_transform_residual_mm <= 0.0
        || !fixture.thresholds.maximum_interpolation_uncertainty_mm.is_finite()
        || fixture.thresholds.maximum_interpolation_uncertainty_mm <= 0.0
        || fixture.thresholds.required_calibration_status.trim().is_empty()
    {
        return Err("exposure-map thresholds are invalid".to_owned());
    }

    let mut channels = BTreeSet::new();
    for channel in &fixture.required_sensor_channels {
        if channel.trim().is_empty() || !channels.insert(channel.as_str()) {
            return Err("required sensor channels are empty or duplicated".to_owned());
        }
    }
    let mut scenario_ids = BTreeSet::new();
    for scenario in &fixture.scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.as_str())
        {
            return Err("scenario IDs are empty or duplicated".to_owned());
        }
    }

    let results: Vec<_> = fixture
        .scenarios
        .iter()
        .map(|scenario| evaluate_scenario(scenario, &fixture))
        .collect();
    let matched = results
        .iter()
        .filter(|result| result.disposition == result.expected_disposition)
        .count();
    let pass_count = results.iter().filter(|result| result.disposition == Disposition::Pass).count();
    let blocked_count = results.iter().filter(|result| result.disposition == Disposition::Blocked).count();
    let rejected_count = results.iter().filter(|result| result.disposition == Disposition::Rejected).count();
    let scenario_count = results.len();

    Ok(AuditReport {
        schema_version: fixture.schema_version,
        benchmark_id: fixture.benchmark_id,
        evidence_class: fixture.evidence_class,
        synthetic_only: fixture.synthetic_only,
        physical_measurements: fixture.physical_measurements,
        plasma_solver_outputs: fixture.plasma_solver_outputs,
        automatic_field_acceptance: fixture.automatic_field_acceptance,
        physical_actuation_authorized: fixture.physical_actuation_authorized,
        scenario_count,
        matched_expectations: matched,
        mismatched_expectations: scenario_count - matched,
        pass_count,
        blocked_count,
        rejected_count,
        results,
        claim_ceiling: fixture.claim_ceiling,
    })
}

fn evaluate_scenario(scenario: &Scenario, fixture: &Fixture) -> ScenarioResult {
    let mut rejected = Vec::<String>::new();
    let mut blocked = Vec::<String>::new();

    for (name, value) in [
        ("sample lineage", scenario.sample_lineage_ref.as_str()),
        ("material interface", scenario.material_interface_ref.as_str()),
        ("geometry ID", scenario.geometry_id.as_str()),
        ("geometry revision", scenario.geometry_revision.as_str()),
        ("geometry digest", scenario.geometry_digest.as_str()),
        ("expected geometry digest", scenario.expected_geometry_digest.as_str()),
        ("tool ID", scenario.tool_id.as_str()),
        ("process configuration", scenario.process_configuration_digest.as_str()),
        ("expected process configuration", scenario.expected_process_configuration_digest.as_str()),
        ("trajectory ID", scenario.trajectory_id.as_str()),
        ("coordinate frame", scenario.coordinate_frame.as_str()),
    ] {
        if value.trim().is_empty() {
            blocked.push(format!("missing {name} identity"));
        }
    }
    if scenario.geometry_digest != scenario.expected_geometry_digest
        || scenario.process_configuration_digest != scenario.expected_process_configuration_digest
    {
        rejected.push("geometry or process-configuration digest differs from the frozen run identity".to_owned());
    }
    if scenario.run_timestamp_unix_ms == 0 {
        blocked.push("run timestamp is missing".to_owned());
    }

    let mut transform_map = BTreeMap::<String, &TransformReceipt>::new();
    for transform in &scenario.transform_receipts {
        if transform.transform_id.trim().is_empty()
            || transform_map.insert(transform.transform_id.clone(), transform).is_some()
        {
            rejected.push("transform identity is empty or duplicated".to_owned());
        }
        if transform.calibration_record_id.trim().is_empty() || transform.raw_trace_digest.trim().is_empty() {
            blocked.push(format!("transform {} lacks calibration/raw trace identity", transform.transform_id));
        }
        if !transform.residual_mm.is_finite() || transform.residual_mm < 0.0 {
            rejected.push(format!("transform {} has invalid registration residual", transform.transform_id));
        } else if transform.residual_mm > fixture.thresholds.maximum_transform_residual_mm {
            blocked.push(format!("transform {} residual exceeds the frozen limit", transform.transform_id));
        }
    }
    let machine_to_article = scenario.transform_receipts.iter().any(|item| {
        item.transform_id == "machine-to-article"
            && item.source_frame == "machine"
            && item.target_frame == "article"
    });
    let article_to_surface = scenario.transform_receipts.iter().any(|item| {
        item.transform_id == "article-to-surface"
            && item.source_frame == "article"
            && item.target_frame == "surface"
    });
    if !machine_to_article || !article_to_surface {
        rejected.push("required machine/article/surface frame-transform chain is incomplete or mismatched".to_owned());
    }

    let required_channels: BTreeSet<_> =
        fixture.required_sensor_channels.iter().map(String::as_str).collect();
    let mut calibrations = BTreeMap::<String, &CalibrationRecord>::new();
    let mut max_clock_offset_ms = None::<u64>;
    for calibration in &scenario.calibration_records {
        if calibration.channel_id.trim().is_empty()
            || calibration.sensor_id.trim().is_empty()
            || calibration.calibration_record_id.trim().is_empty()
            || calibrations.insert(calibration.channel_id.clone(), calibration).is_some()
        {
            rejected.push("calibration channel/sensor/record identity is empty or duplicated".to_owned());
        }
        if calibration.raw_trace_digest.trim().is_empty()
            || calibration.calibration_status != fixture.thresholds.required_calibration_status
        {
            blocked.push(format!("channel {} lacks qualified calibration/raw trace", calibration.channel_id));
        }
        if scenario.run_timestamp_unix_ms == 0
            || calibration.valid_from_unix_ms > scenario.run_timestamp_unix_ms
            || calibration.valid_until_unix_ms < scenario.run_timestamp_unix_ms
            || calibration.valid_from_unix_ms > calibration.valid_until_unix_ms
        {
            blocked.push(format!("channel {} calibration is not valid at run time", calibration.channel_id));
        }
        let offset = calibration.clock_offset_ms.unsigned_abs();
        max_clock_offset_ms = Some(max_clock_offset_ms.unwrap_or(0).max(offset));
        if offset > fixture.thresholds.maximum_clock_offset_ms {
            blocked.push(format!("channel {} clock offset exceeds the synchronization limit", calibration.channel_id));
        }
    }
    let observed_channels: BTreeSet<_> = calibrations.keys().map(String::as_str).collect();
    if !required_channels.is_subset(&observed_channels) {
        blocked.push("one or more required calibrated sensor channels are missing".to_owned());
    }

    let mut required_regions = BTreeSet::new();
    for id in &scenario.required_region_ids {
        if id.trim().is_empty() || !required_regions.insert(id.as_str()) {
            rejected.push("required region identity is empty or duplicated".to_owned());
        }
    }
    if required_regions.is_empty() {
        rejected.push("declared surface region set is empty".to_owned());
    }

    let mut region_map = BTreeMap::<String, &Region>::new();
    let mut observed_region_ids = BTreeSet::new();
    let mut max_interpolation_uncertainty_mm = None::<f64>;
    for region in &scenario.regions {
        if region.region_id.trim().is_empty()
            || region_map.insert(region.region_id.clone(), region).is_some()
        {
            rejected.push("region ID is empty or duplicated".to_owned());
            continue;
        }
        if !required_regions.contains(region.region_id.as_str()) {
            rejected.push(format!("region {} is not in the frozen required-region set", region.region_id));
        }
        if region.article_frame_position_mm.len() != 3
            || region.article_frame_position_mm.iter().any(|value| !value.is_finite())
        {
            rejected.push(format!("region {} has an invalid article-frame position", region.region_id));
        }
        match region.status.as_str() {
            "UNKNOWN" => {
                let any_value = [
                    region.local_standoff_mm,
                    region.local_ground_distance_mm,
                    region.incidence_angle_deg,
                    region.relative_speed_mm_s,
                    region.dwell_ms,
                    region.electrical_power_w,
                    region.optical_signal_au,
                    region.interpolation_uncertainty_mm,
                ]
                .iter()
                .any(Option::is_some);
                if any_value
                    || region.geometry_transform_id.is_some()
                    || region.electrical_calibration_record_id.is_some()
                    || region.optical_calibration_record_id.is_some()
                {
                    rejected.push(format!("unknown region {} is filled with numeric exposure values", region.region_id));
                }
            }
            "OBSERVED" => {
                observed_region_ids.insert(region.region_id.as_str());
                for (field, value) in [
                    ("local standoff", region.local_standoff_mm),
                    ("local ground distance", region.local_ground_distance_mm),
                    ("incidence angle", region.incidence_angle_deg),
                    ("relative speed", region.relative_speed_mm_s),
                    ("dwell", region.dwell_ms),
                    ("electrical power", region.electrical_power_w),
                    ("optical signal", region.optical_signal_au),
                    ("interpolation uncertainty", region.interpolation_uncertainty_mm),
                ] {
                    if value.is_none_or(|number| !number.is_finite()) {
                        blocked.push(format!("observed region {} lacks finite {field}", region.region_id));
                    }
                }
                if region.local_standoff_mm.is_some_and(|value| value <= 0.0)
                    || region.local_ground_distance_mm.is_some_and(|value| value < 0.0)
                    || region.incidence_angle_deg.is_some_and(|value| !(-90.0..=90.0).contains(&value))
                    || region.relative_speed_mm_s.is_some_and(|value| value < 0.0)
                    || region.dwell_ms.is_some_and(|value| value < 0.0)
                    || region.electrical_power_w.is_some_and(|value| value < 0.0)
                    || region.optical_signal_au.is_some_and(|value| value < 0.0)
                    || region.interpolation_uncertainty_mm.is_some_and(|value| value < 0.0)
                {
                    rejected.push(format!("observed region {} contains out-of-domain physical values", region.region_id));
                }
                if region.geometry_transform_id.as_deref() != Some("article-to-surface") {
                    rejected.push(format!("region {} does not use the declared article-to-surface transform", region.region_id));
                }
                if region.electrical_calibration_record_id.as_deref()
                    != calibrations.get("electrical_power").map(|record| record.calibration_record_id.as_str())
                    || region.optical_calibration_record_id.as_deref()
                    != calibrations.get("optical_signal").map(|record| record.calibration_record_id.as_str())
                {
                    rejected.push(format!("region {} channel calibration references do not match the run manifest", region.region_id));
                }
                if let Some(value) = region.interpolation_uncertainty_mm {
                    max_interpolation_uncertainty_mm =
                        Some(max_interpolation_uncertainty_mm.unwrap_or(0.0_f64).max(value));
                    if value > fixture.thresholds.maximum_interpolation_uncertainty_mm {
                        blocked.push(format!("region {} interpolation uncertainty exceeds the frozen limit", region.region_id));
                    }
                }
            }
            other => rejected.push(format!("region {} has unsupported status {other}", region.region_id)),
        }
    }

    if region_map.keys().any(|id| !required_regions.contains(id.as_str())) {
        rejected.push("map contains unregistered region identities".to_owned());
    }
    let missing: Vec<_> = required_regions
        .iter()
        .filter(|id| !region_map.contains_key(**id))
        .copied()
        .collect();
    if !missing.is_empty() {
        blocked.push(format!("required surface regions are unmapped: {}", missing.join(",")));
    }
    let coverage_fraction = if required_regions.is_empty() {
        0.0
    } else {
        observed_region_ids.len() as f64 / required_regions.len() as f64
    };

    let (disposition, reasons) = if !rejected.is_empty() {
        (Disposition::Rejected, rejected)
    } else if !blocked.is_empty() {
        (Disposition::Blocked, blocked)
    } else {
        (Disposition::Pass, Vec::new())
    };

    ScenarioResult {
        scenario_id: scenario.scenario_id.clone(),
        disposition,
        expected_disposition: scenario.expected.disposition,
        reasons,
        required_region_count: required_regions.len(),
        observed_region_count: observed_region_ids.len(),
        coverage_fraction,
        maximum_interpolation_uncertainty_mm: max_interpolation_uncertainty_mm,
        maximum_clock_offset_ms: max_clock_offset_ms,
    }
}

pub fn print_report(report: &AuditReport, as_json: bool) -> Result<()> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        println!("Benchmark: {}", report.benchmark_id);
        println!("Evidence: {}", report.evidence_class);
        println!("Claim ceiling: {}", report.claim_ceiling);
        for result in &report.results {
            println!(
                "  {:?} {} coverage={:.3} regions={}/{} max_clock_ms={:?} max_interp_mm={:?}",
                result.disposition,
                result.scenario_id,
                result.coverage_fraction,
                result.observed_region_count,
                result.required_region_count,
                result.maximum_clock_offset_ms,
                result.maximum_interpolation_uncertainty_mm
            );
            for reason in &result.reasons {
                println!("    - {reason}");
            }
        }
        println!(
            "Expected dispositions: {}/{} matched; PASS={} BLOCKED={} REJECTED={}",
            report.matched_expectations,
            report.scenario_count,
            report.pass_count,
            report.blocked_count,
            report.rejected_count
        );
    }
    if report.mismatched_expectations > 0 {
        anyhow::bail!(
            "plasma exposure-map fixture has {} disposition mismatches",
            report.mismatched_expectations
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_exposure_map_fixture_v0_1.json");

    fn report() -> AuditReport {
        audit_fixture_json(FIXTURE).expect("exposure-map fixture audit")
    }

    fn result_for<'a>(report: &'a AuditReport, id: &str) -> &'a ScenarioResult {
        report.results.iter().find(|item| item.scenario_id == id).expect("scenario exists")
    }

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let report = report();
        assert_eq!(report.scenario_count, 10);
        assert_eq!(report.mismatched_expectations, 0);
        assert_eq!(report.matched_expectations, 10);
        assert_eq!(report.pass_count, 2);
        assert_eq!(report.blocked_count, 4);
        assert_eq!(report.rejected_count, 4);
        assert!(!report.physical_measurements);
        assert!(!report.automatic_field_acceptance);
    }

    #[test]
    fn complete_spatial_map_passes_synthetically() {
        assert_eq!(result_for(&report(), "complete-planar-map").disposition, Disposition::Pass);
    }

    #[test]
    fn top_standoff_does_not_erase_ground_boundary_difference() {
        let report = report();
        let result = result_for(&report, "same-top-gap-different-ground-boundary");
        assert_eq!(result.disposition, Disposition::Pass);
        assert_eq!(result.coverage_fraction, 1.0);
    }

    #[test]
    fn missing_trailing_edge_region_blocks_map() {
        let result = result_for(&report(), "missing-trailing-edge-region");
        assert_eq!(result.disposition, Disposition::Blocked);
        assert!(result.coverage_fraction < 1.0);
    }

    #[test]
    fn unknown_region_cannot_be_zero_filled() {
        assert_eq!(result_for(&report(), "unknown-region-zero-filled").disposition, Disposition::Rejected);
    }

    #[test]
    fn clock_skew_blocks_map() {
        assert_eq!(result_for(&report(), "clock-skew").disposition, Disposition::Blocked);
    }

    #[test]
    fn frame_transform_mismatch_is_rejected() {
        assert_eq!(result_for(&report(), "transform-frame-mismatch").disposition, Disposition::Rejected);
    }

    #[test]
    fn duplicate_region_identity_is_rejected() {
        assert_eq!(result_for(&report(), "duplicate-region-identity").disposition, Disposition::Rejected);
    }

    #[test]
    fn excessive_interpolation_uncertainty_blocks_map() {
        assert_eq!(result_for(&report(), "interpolation-uncertainty-high").disposition, Disposition::Blocked);
    }

    #[test]
    fn expired_calibration_blocks_map() {
        assert_eq!(result_for(&report(), "expired-calibration").disposition, Disposition::Blocked);
    }

    #[test]
    fn geometry_digest_mismatch_is_rejected() {
        assert_eq!(result_for(&report(), "geometry-configuration-mismatch").disposition, Disposition::Rejected);
    }
}
