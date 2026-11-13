// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-only contract validator for plasma telemetry-to-product virtual
//! metrology. It checks run/specimen linkage, calibrated telemetry, energy
//! boundary completeness, domain status and false-accept controls. It does not
//! train a model, predict physical bond strength or authorize quality release.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const EXPECTED_EVIDENCE_CLASS: &str = "SYNTHETIC";
const SYNTHETIC_LABEL_CLASS: &str = "SYNTHETIC_LABEL";
const SYNTHETIC_PREDICTION_CLASS: &str = "SYNTHETIC_MODEL_PREDICTION";
const FLOAT_EPSILON: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Disposition {
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
    model_is_trained: bool,
    automatic_quality_release: bool,
    physical_actuation_authorized: bool,
    claim_ceiling: String,
    service_profile: ServiceProfile,
    allowed_material_profiles: Vec<String>,
    required_sensors: Vec<String>,
    maximum_sensor_time_offset_ms: u64,
    energy_boundary_contract_id: String,
    required_calibration_status: String,
    required_energy_subsystems: Vec<String>,
    required_holdout_dimensions: Vec<String>,
    minimum_in_spec_probability: f64,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceProfile {
    profile_id: String,
    substrate_grade: String,
    adhesive_system: String,
    bond_test_method: String,
    acceptance_threshold_mpa: f64,
    prediction_interval_coverage_target: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    service_profile_id: String,
    material_lot_id: String,
    device_id: String,
    production_day: String,
    process_run_id: String,
    specimen_id: String,
    configuration_digest: String,
    run_timestamp_unix_ms: u64,
    telemetry: Vec<TelemetrySample>,
    energy: EnergyRecord,
    downstream_outcome: Option<DownstreamOutcome>,
    model_prediction: ModelPrediction,
    material_in_model_domain: bool,
    expected_disposition: Disposition,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetrySample {
    sensor_id: String,
    quantity: String,
    unit: String,
    value: f64,
    sample_offset_ms: i64,
    calibration_record_id: String,
    calibration_status: String,
    calibration_valid_from_unix_ms: u64,
    calibration_valid_until_unix_ms: u64,
    uncertainty: f64,
    configuration_digest: String,
    raw_trace_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnergyRecord {
    energy_boundary_contract_id: String,
    complete_system_energy_kwh: f64,
    included_subsystems: Vec<String>,
    subsystem_measurements: Vec<EnergySubsystemMeasurement>,
    complete: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnergySubsystemMeasurement {
    subsystem_id: String,
    energy_kwh: f64,
    meter_id: String,
    calibration_record_id: String,
    calibration_status: String,
    calibration_valid_from_unix_ms: u64,
    calibration_valid_until_unix_ms: u64,
    uncertainty_kwh: f64,
    raw_trace_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DownstreamOutcome {
    specimen_id: String,
    test_method: String,
    bond_strength_mpa: f64,
    failure_mode: String,
    evidence_class: String,
    physically_measured: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelPrediction {
    model_id: String,
    evidence_class: String,
    predicted_bond_strength_interval_mpa: Vec<f64>,
    probability_of_meeting_acceptance: f64,
    predicted_disposition: String,
    training_scope: String,
    training_group_manifest: Option<TrainingGroupManifest>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingGroupManifest {
    manifest_id: String,
    partition_receipt_digest: String,
    grouping_dimensions: Vec<String>,
    training_material_lot_ids: Vec<String>,
    training_device_ids: Vec<String>,
    training_production_days: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub disposition: Disposition,
    pub expected_disposition: Disposition,
    pub reasons: Vec<String>,
    pub telemetry_samples: usize,
    pub downstream_label_present: bool,
    pub energy_boundary_complete: bool,
    pub virtual_metrology_false_accept: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub schema_version: String,
    pub benchmark_id: String,
    pub evidence_class: String,
    pub synthetic_only: bool,
    pub physical_measurements: bool,
    pub model_is_trained: bool,
    pub automatic_quality_release: bool,
    pub physical_actuation_authorized: bool,
    pub scenario_count: usize,
    pub matched_expectations: usize,
    pub mismatched_expectations: usize,
    pub results: Vec<ScenarioResult>,
    pub claim_ceiling: String,
}

pub fn run_audit(fixture_path: &Path) -> Result<AuditReport> {
    let bytes = fs::read(fixture_path)
        .with_context(|| format!("read plasma virtual-metrology fixture {}", fixture_path.display()))?;
    let input = std::str::from_utf8(&bytes).context("virtual-metrology fixture is not UTF-8")?;
    audit_fixture_json(input).map_err(anyhow::Error::msg)
}

pub fn audit_fixture_json(input: &str) -> std::result::Result<AuditReport, String> {
    let fixture: Fixture =
        serde_json::from_str(input).map_err(|error| format!("invalid virtual-metrology fixture: {error}"))?;

    if fixture.schema_version != "0.1.0" {
        return Err("unsupported virtual-metrology schema version".to_owned());
    }
    if fixture.benchmark_id != "PLASMA-VM-001" {
        return Err("unexpected virtual-metrology benchmark ID".to_owned());
    }
    if fixture.evidence_class != EXPECTED_EVIDENCE_CLASS
        || !fixture.synthetic_only
        || fixture.physical_measurements
        || fixture.model_is_trained
        || fixture.automatic_quality_release
        || fixture.physical_actuation_authorized
    {
        return Err(
            "fixture must remain synthetic-only, untrained, non-physical, advisory and non-actuating"
                .to_owned(),
        );
    }
    if fixture.claim_ceiling.trim().is_empty() {
        return Err("empty virtual-metrology claim ceiling".to_owned());
    }
    if fixture.service_profile.profile_id.trim().is_empty()
        || fixture.service_profile.substrate_grade.trim().is_empty()
        || fixture.service_profile.adhesive_system.trim().is_empty()
        || fixture.service_profile.bond_test_method.trim().is_empty()
        || !fixture.service_profile.acceptance_threshold_mpa.is_finite()
        || fixture.service_profile.acceptance_threshold_mpa <= 0.0
        || !fixture.service_profile.prediction_interval_coverage_target.is_finite()
        || !(0.0..=1.0).contains(&fixture.service_profile.prediction_interval_coverage_target)
    {
        return Err("service profile is incomplete or invalid".to_owned());
    }
    if fixture.energy_boundary_contract_id.trim().is_empty() {
        return Err("empty fixture energy-boundary contract ID".to_owned());
    }
    if fixture.required_sensors.is_empty() || fixture.required_energy_subsystems.is_empty() {
        return Err("required telemetry / energy boundary lists cannot be empty".to_owned());
    }
    if fixture.required_holdout_dimensions.is_empty() {
        return Err("required holdout dimensions cannot be empty".to_owned());
    }
    let mut holdout_dimensions = BTreeSet::new();
    for dimension in &fixture.required_holdout_dimensions {
        if !matches!(
            dimension.as_str(),
            "material_lot_id" | "device_id" | "production_day"
        ) || !holdout_dimensions.insert(dimension.as_str())
        {
            return Err("holdout dimensions are unsupported or duplicated".to_owned());
        }
    }
    if !fixture.minimum_in_spec_probability.is_finite()
        || !(0.0..=1.0).contains(&fixture.minimum_in_spec_probability)
    {
        return Err("minimum in-spec probability must be within [0, 1]".to_owned());
    }
    if fixture.required_calibration_status != "QUALIFIED" {
        return Err("required calibration status must be QUALIFIED".to_owned());
    }
    let allowed_profiles: BTreeSet<_> =
        fixture.allowed_material_profiles.iter().map(String::as_str).collect();
    if !allowed_profiles.contains(fixture.service_profile.profile_id.as_str()) {
        return Err("service profile is not in the declared material-domain list".to_owned());
    }
    if fixture.scenarios.is_empty() {
        return Err("virtual-metrology fixture contains no scenarios".to_owned());
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in &fixture.scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.as_str())
        {
            return Err("empty or duplicate virtual-metrology scenario ID".to_owned());
        }
    }

    let results: Vec<_> = fixture
        .scenarios
        .iter()
        .map(|scenario| evaluate_scenario(scenario, &fixture))
        .collect();
    let matched_expectations = results
        .iter()
        .filter(|result| result.disposition == result.expected_disposition)
        .count();
    let scenario_count = results.len();

    Ok(AuditReport {
        schema_version: fixture.schema_version,
        benchmark_id: fixture.benchmark_id,
        evidence_class: fixture.evidence_class,
        synthetic_only: fixture.synthetic_only,
        physical_measurements: fixture.physical_measurements,
        model_is_trained: fixture.model_is_trained,
        automatic_quality_release: fixture.automatic_quality_release,
        physical_actuation_authorized: fixture.physical_actuation_authorized,
        scenario_count,
        matched_expectations,
        mismatched_expectations: scenario_count - matched_expectations,
        results,
        claim_ceiling: fixture.claim_ceiling,
    })
}

fn evaluate_scenario(scenario: &Scenario, fixture: &Fixture) -> ScenarioResult {
    let mut rejected = Vec::<String>::new();
    let mut blocked = Vec::<String>::new();
    let mut false_accept = false;

    for (field, value) in [
        ("service profile", scenario.service_profile_id.as_str()),
        ("material lot", scenario.material_lot_id.as_str()),
        ("device", scenario.device_id.as_str()),
        ("production day", scenario.production_day.as_str()),
        ("process run", scenario.process_run_id.as_str()),
        ("specimen", scenario.specimen_id.as_str()),
        ("configuration digest", scenario.configuration_digest.as_str()),
    ] {
        if value.trim().is_empty() {
            blocked.push(format!("missing {field} identity"));
        }
    }

    if !fixture
        .allowed_material_profiles
        .iter()
        .any(|profile| profile == &scenario.service_profile_id)
    {
        blocked.push("material/adhesive profile is outside the declared model domain".to_owned());
    }
    if scenario.service_profile_id != fixture.service_profile.profile_id {
        blocked.push("scenario service profile differs from the frozen fixture profile".to_owned());
    }
    if !scenario.material_in_model_domain {
        blocked.push("scenario explicitly marked outside model domain".to_owned());
    }
    if scenario.run_timestamp_unix_ms == 0 {
        blocked.push("run timestamp is missing".to_owned());
    }

    match &scenario.model_prediction.training_group_manifest {
        None => blocked.push("prediction lacks a training-group partition manifest".to_owned()),
        Some(manifest) => {
            if manifest.manifest_id.trim().is_empty()
                || manifest.partition_receipt_digest.trim().is_empty()
            {
                blocked.push("training-group manifest lacks identity/partition receipt".to_owned());
            }
            let declared: BTreeSet<_> =
                manifest.grouping_dimensions.iter().map(String::as_str).collect();
            let required: BTreeSet<_> =
                fixture.required_holdout_dimensions.iter().map(String::as_str).collect();
            if declared != required {
                blocked.push("training-group manifest does not cover the required holdout dimensions".to_owned());
            }
            if manifest.training_material_lot_ids.is_empty()
                || manifest.training_device_ids.is_empty()
                || manifest.training_production_days.is_empty()
            {
                blocked.push("training-group manifest has an empty training-group axis".to_owned());
            }
            if manifest.training_material_lot_ids.iter().any(|x| x.trim().is_empty())
                || manifest.training_device_ids.iter().any(|x| x.trim().is_empty())
                || manifest.training_production_days.iter().any(|x| x.trim().is_empty())
            {
                blocked.push("training-group manifest contains an empty group identity".to_owned());
            }
            let mut lot_ids = BTreeSet::new();
            let mut device_ids = BTreeSet::new();
            let mut production_days = BTreeSet::new();
            if manifest.training_material_lot_ids.iter().any(|x| !lot_ids.insert(x.as_str()))
                || manifest.training_device_ids.iter().any(|x| !device_ids.insert(x.as_str()))
                || manifest.training_production_days.iter().any(|x| !production_days.insert(x.as_str()))
            {
                rejected.push("training-group manifest contains duplicate group identities".to_owned());
            }
            if lot_ids.contains(scenario.material_lot_id.as_str())
                || device_ids.contains(scenario.device_id.as_str())
                || production_days.contains(scenario.production_day.as_str())
            {
                rejected.push("held-out material lot, device or production day overlaps the model training groups".to_owned());
            }
        }
    }

    let required_sensor_ids: BTreeSet<_> =
        fixture.required_sensors.iter().map(String::as_str).collect();
    let mut sensor_map = BTreeMap::<String, &TelemetrySample>::new();
    for sample in &scenario.telemetry {
        if sample.sensor_id.trim().is_empty()
            || sensor_map.insert(sample.sensor_id.clone(), sample).is_some()
        {
            rejected.push("empty or duplicate sensor identity".to_owned());
        }
        if sample.quantity.trim().is_empty() || sample.unit.trim().is_empty() {
            blocked.push(format!("sensor {} lacks quantity or unit", sample.sensor_id));
        }
        if !sample.value.is_finite()
            || !sample.uncertainty.is_finite()
            || sample.uncertainty < 0.0
        {
            rejected.push(format!("sensor {} has invalid value/uncertainty", sample.sensor_id));
        }
        if sample.sample_offset_ms.unsigned_abs() > fixture.maximum_sensor_time_offset_ms {
            blocked.push(format!("sensor {} sample is stale/future-dated/outside run window", sample.sensor_id));
        }
        if sample.calibration_record_id.trim().is_empty()
            || sample.calibration_status != fixture.required_calibration_status
        {
            blocked.push(format!("sensor {} lacks qualified calibration", sample.sensor_id));
        }
        if sample.calibration_valid_from_unix_ms > scenario.run_timestamp_unix_ms
            || sample.calibration_valid_until_unix_ms < scenario.run_timestamp_unix_ms
            || sample.calibration_valid_from_unix_ms > sample.calibration_valid_until_unix_ms
        {
            blocked.push(format!("sensor {} calibration is not valid at run time", sample.sensor_id));
        }
        if sample.configuration_digest != scenario.configuration_digest {
            rejected.push(format!("sensor {} configuration digest differs from run", sample.sensor_id));
        }
        if sample.raw_trace_digest.trim().is_empty() {
            blocked.push(format!("sensor {} lacks raw-trace identity", sample.sensor_id));
        }
    }
    let observed_sensor_ids: BTreeSet<_> = sensor_map.keys().map(String::as_str).collect();
    if !required_sensor_ids.is_subset(&observed_sensor_ids) {
        blocked.push("one or more required telemetry channels are missing".to_owned());
    }

    let energy = &scenario.energy;
    if energy.energy_boundary_contract_id.trim().is_empty()
        || energy.energy_boundary_contract_id != fixture.energy_boundary_contract_id
        || !energy.complete_system_energy_kwh.is_finite()
        || energy.complete_system_energy_kwh < 0.0
    {
        rejected.push("energy boundary identity/value is invalid".to_owned());
    }
    let included: BTreeSet<_> = energy.included_subsystems.iter().map(String::as_str).collect();
    let required: BTreeSet<_> = fixture.required_energy_subsystems.iter().map(String::as_str).collect();
    if included.len() != energy.included_subsystems.len() {
        rejected.push("energy boundary contains duplicate included subsystem IDs".to_owned());
    }
    if !required.is_subset(&included) {
        blocked.push("complete grid-to-service energy boundary is missing one or more required loads".to_owned());
    }
    if !included.is_subset(&required) {
        rejected.push("energy boundary contains undeclared subsystem IDs".to_owned());
    }

    let mut energy_measurements = BTreeMap::<String, &EnergySubsystemMeasurement>::new();
    for measurement in &energy.subsystem_measurements {
        if measurement.subsystem_id.trim().is_empty()
            || energy_measurements
                .insert(measurement.subsystem_id.clone(), measurement)
                .is_some()
        {
            rejected.push("energy subsystem measurement ID is empty or duplicated".to_owned());
        }
        if !required.contains(measurement.subsystem_id.as_str()) {
            rejected.push(format!("energy measurement {} is not in the frozen boundary", measurement.subsystem_id));
        }
        if !measurement.energy_kwh.is_finite()
            || measurement.energy_kwh < 0.0
            || !measurement.uncertainty_kwh.is_finite()
            || measurement.uncertainty_kwh < 0.0
        {
            rejected.push(format!("energy measurement {} has invalid value/uncertainty", measurement.subsystem_id));
        }
        if measurement.meter_id.trim().is_empty()
            || measurement.calibration_record_id.trim().is_empty()
            || measurement.raw_trace_digest.trim().is_empty()
            || measurement.calibration_status != fixture.required_calibration_status
        {
            blocked.push(format!("energy measurement {} lacks a calibrated meter/raw-trace identity", measurement.subsystem_id));
        }
        if measurement.calibration_valid_from_unix_ms > scenario.run_timestamp_unix_ms
            || measurement.calibration_valid_until_unix_ms < scenario.run_timestamp_unix_ms
            || measurement.calibration_valid_from_unix_ms > measurement.calibration_valid_until_unix_ms
        {
            blocked.push(format!("energy meter {} calibration is not valid at run time", measurement.subsystem_id));
        }
    }
    let measured_ids: BTreeSet<_> = energy_measurements.keys().map(String::as_str).collect();
    if !required.is_subset(&measured_ids) {
        blocked.push("one or more required energy subsystems lack an individual meter record".to_owned());
    }
    if !measured_ids.is_subset(&required) {
        rejected.push("energy measurements contain undeclared subsystem IDs".to_owned());
    }

    let mut energy_sum = 0.0_f64;
    let mut energy_values_valid = true;
    for subsystem in &required {
        match energy_measurements.get(*subsystem) {
            Some(measurement)
                if measurement.energy_kwh.is_finite() && measurement.energy_kwh >= 0.0 =>
            {
                energy_sum += measurement.energy_kwh;
            }
            _ => energy_values_valid = false,
        }
    }
    if energy_values_valid {
        let energy_tolerance = FLOAT_EPSILON
            * energy.complete_system_energy_kwh.abs().max(1.0);
        if (energy_sum - energy.complete_system_energy_kwh).abs() > energy_tolerance {
            rejected.push("energy subsystem measurements do not close to the complete-system total".to_owned());
        }
    }
    if !energy.complete {
        blocked.push("energy boundary is not declared complete".to_owned());
    }
    let energy_boundary_complete = energy.complete
        && included == required
        && measured_ids == required
        && energy_values_valid
        && energy_sum.is_finite()
        && (energy_sum - energy.complete_system_energy_kwh).abs()
            <= FLOAT_EPSILON * energy.complete_system_energy_kwh.abs().max(1.0)
        && energy_measurements.values().all(|measurement| {
            !measurement.meter_id.trim().is_empty()
                && !measurement.calibration_record_id.trim().is_empty()
                && !measurement.raw_trace_digest.trim().is_empty()
                && measurement.calibration_status == fixture.required_calibration_status
                && measurement.calibration_valid_from_unix_ms <= scenario.run_timestamp_unix_ms
                && measurement.calibration_valid_until_unix_ms >= scenario.run_timestamp_unix_ms
        });

    let outcome = match &scenario.downstream_outcome {
        Some(outcome) => Some(outcome),
        None => {
            blocked.push("downstream physical-outcome label is missing".to_owned());
            None
        }
    };
    if let Some(outcome) = outcome {
        if outcome.specimen_id != scenario.specimen_id {
            rejected.push("downstream outcome specimen identity does not match run/specimen join".to_owned());
        }
        if outcome.test_method != fixture.service_profile.bond_test_method {
            rejected.push("downstream outcome test method differs from frozen service profile".to_owned());
        }
        if !outcome.bond_strength_mpa.is_finite() || outcome.bond_strength_mpa < 0.0 {
            rejected.push("downstream bond-strength label is invalid".to_owned());
        }
        if outcome.failure_mode.trim().is_empty() {
            blocked.push("downstream failure mode is missing".to_owned());
        }
        if outcome.evidence_class != SYNTHETIC_LABEL_CLASS || outcome.physically_measured {
            rejected.push("synthetic fixture outcome cannot claim physical measurement".to_owned());
        }
    }

    let prediction = &scenario.model_prediction;
    if prediction.model_id.trim().is_empty()
        || prediction.evidence_class != SYNTHETIC_PREDICTION_CLASS
        || prediction.training_scope != "SYNTHETIC_FIXTURE_ONLY"
    {
        rejected.push("prediction identity/evidence class violates synthetic claim boundary".to_owned());
    }
    if !prediction.probability_of_meeting_acceptance.is_finite()
        || !(0.0..=1.0).contains(&prediction.probability_of_meeting_acceptance)
    {
        rejected.push("predicted acceptance probability is outside [0,1]".to_owned());
    }
    if prediction.predicted_bond_strength_interval_mpa.len() != 2
        || prediction.predicted_bond_strength_interval_mpa.iter().any(|x| !x.is_finite() || *x < 0.0)
        || prediction.predicted_bond_strength_interval_mpa[0]
            > prediction.predicted_bond_strength_interval_mpa[1]
    {
        rejected.push("predicted bond-strength interval is invalid".to_owned());
    }
    if prediction.predicted_disposition == "IN_SPEC"
        && (prediction.predicted_bond_strength_interval_mpa.len() == 2
            && prediction.predicted_bond_strength_interval_mpa[0]
                + FLOAT_EPSILON < fixture.service_profile.acceptance_threshold_mpa
            || prediction.probability_of_meeting_acceptance
                + FLOAT_EPSILON < fixture.minimum_in_spec_probability)
    {
        rejected.push("IN_SPEC prediction violates the frozen lower-bound/probability decision rule".to_owned());
    }
    if prediction.predicted_disposition == "OUT_OF_SPEC"
        && (prediction.predicted_bond_strength_interval_mpa.len() == 2
            && prediction.predicted_bond_strength_interval_mpa[1]
                >= fixture.service_profile.acceptance_threshold_mpa
            || prediction.probability_of_meeting_acceptance
                > 1.0 - fixture.minimum_in_spec_probability + FLOAT_EPSILON)
    {
        rejected.push("OUT_OF_SPEC prediction violates the frozen upper-bound/probability decision rule".to_owned());
    }
    if !matches!(
        prediction.predicted_disposition.as_str(),
        "IN_SPEC" | "OUT_OF_SPEC" | "ABSTAIN"
    ) {
        rejected.push("unsupported virtual-metrology disposition".to_owned());
    }
    if let Some(outcome) = outcome {
        let threshold = fixture.service_profile.acceptance_threshold_mpa;
        if prediction.predicted_disposition == "IN_SPEC"
            && outcome.bond_strength_mpa + FLOAT_EPSILON < threshold
        {
            false_accept = true;
            rejected.push("virtual metrology falsely accepts an out-of-spec synthetic bond".to_owned());
        }
        if prediction.predicted_disposition == "OUT_OF_SPEC"
            && outcome.bond_strength_mpa >= threshold
        {
            rejected.push("virtual metrology falsely rejects an in-spec synthetic bond".to_owned());
        }
    }

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
        expected_disposition: scenario.expected_disposition,
        reasons,
        telemetry_samples: scenario.telemetry.len(),
        downstream_label_present: scenario.downstream_outcome.is_some(),
        energy_boundary_complete,
        virtual_metrology_false_accept: false_accept,
    }
}

pub fn print_report(report: &AuditReport, as_json: bool) -> Result<()> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    println!("Benchmark: {}", report.benchmark_id);
    println!("Evidence: {}", report.evidence_class);
    println!("Claim ceiling: {}", report.claim_ceiling);
    for result in &report.results {
        println!(
            "  {:?} {} sensors={} label={} energy_complete={} false_accept={}",
            result.disposition,
            result.scenario_id,
            result.telemetry_samples,
            result.downstream_label_present,
            result.energy_boundary_complete,
            result.virtual_metrology_false_accept
        );
        for reason in &result.reasons {
            println!("    - {reason}");
        }
    }
    println!(
        "Expected dispositions: {}/{} matched",
        report.matched_expectations, report.scenario_count
    );
    if report.mismatched_expectations > 0 {
        anyhow::bail!("virtual-metrology fixture has {} disposition mismatches", report.mismatched_expectations);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(
        "../../data/research/plasma_virtual_metrology_fixture_v0_1.json"
    );

    fn run(value: &serde_json::Value) -> AuditReport {
        audit_fixture_json(&value.to_string()).expect("mutated virtual-metrology fixture parses")
    }

    fn result_for<'a>(report: &'a AuditReport, id: &str) -> &'a ScenarioResult {
        report.results.iter().find(|x| x.scenario_id == id).expect("scenario exists")
    }

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(report.scenario_count, 16);
        assert_eq!(report.mismatched_expectations, 0);
        assert_eq!(report.matched_expectations, 16);
        assert!(report.synthetic_only);
        assert!(!report.model_is_trained);
        assert!(!report.physical_measurements);
        assert!(!report.automatic_quality_release);
    }

    #[test]
    fn aligned_telemetry_and_outcome_pass() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "aligned-heldout-valid").disposition, Disposition::Pass);
    }

    #[test]
    fn confident_false_accept_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        let result = result_for(&report, "confident-false-accept");
        assert_eq!(result.disposition, Disposition::Rejected);
        assert!(result.virtual_metrology_false_accept);
    }

    #[test]
    fn uncalibrated_sensor_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "uncalibrated-oes").disposition, Disposition::Blocked);
    }

    #[test]
    fn stale_sensor_clock_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "stale-sensor-clock").disposition, Disposition::Blocked);
    }

    #[test]
    fn configuration_mismatch_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "configuration-mismatch").disposition, Disposition::Rejected);
    }

    #[test]
    fn specimen_outcome_mismatch_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "specimen-outcome-mismatch").disposition, Disposition::Rejected);
    }

    #[test]
    fn unsupported_material_profile_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "unsupported-material-profile").disposition, Disposition::Blocked);
    }

    #[test]
    fn incomplete_energy_boundary_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "incomplete-energy-boundary").disposition, Disposition::Blocked);
    }

    #[test]
    fn missing_downstream_outcome_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "missing-downstream-outcome").disposition, Disposition::Blocked);
    }

    #[test]
    fn unverified_group_holdout_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "model-holdout-unverified").disposition, Disposition::Blocked);
    }

    #[test]
    fn mismatched_energy_boundary_contract_is_rejected() {
        let mut value: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        value["scenarios"][0]["energy"]["energy_boundary_contract_id"] =
            serde_json::json!("GENERATOR_ONLY_NOT_GRID_TO_BOND");
        let report = audit_fixture_json(&value.to_string()).unwrap();
        assert_eq!(
            result_for(&report, "aligned-heldout-valid").disposition,
            Disposition::Rejected
        );
    }

    #[test]
    fn training_group_overlap_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "training-group-overlap").disposition, Disposition::Rejected);
    }

    #[test]
    fn missing_training_group_manifest_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "model-holdout-unverified").disposition, Disposition::Blocked);
    }

    #[test]
    fn future_dated_sensor_sample_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "future-dated-sensor").disposition, Disposition::Blocked);
    }

    #[test]
    fn expired_sensor_calibration_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "expired-sensor-calibration").disposition, Disposition::Blocked);
    }

    #[test]
    fn missing_energy_meter_record_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "missing-energy-meter").disposition, Disposition::Blocked);
    }

    #[test]
    fn energy_subsystem_sum_mismatch_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "energy-subsystem-sum-mismatch").disposition, Disposition::Rejected);
    }

    #[test]
    fn training_group_overlap_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "training-group-overlap").disposition, Disposition::Rejected);
    }

    #[test]
    fn missing_training_group_manifest_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "model-holdout-unverified").disposition, Disposition::Blocked);
    }

    #[test]
    fn future_dated_sensor_sample_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "future-dated-sensor").disposition, Disposition::Blocked);
    }

    #[test]
    fn expired_sensor_calibration_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "expired-sensor-calibration").disposition, Disposition::Blocked);
    }

    #[test]
    fn missing_energy_meter_record_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "missing-energy-meter").disposition, Disposition::Blocked);
    }

    #[test]
    fn energy_subsystem_sum_mismatch_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(result_for(&report, "energy-subsystem-sum-mismatch").disposition, Disposition::Rejected);
    }

    #[test]
    fn in_spec_prediction_below_probability_gate_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "in-spec-probability-below-gate").disposition,
            Disposition::Rejected
        );
    }

    #[test]
    fn automatic_quality_release_is_rejected_at_manifest_boundary() {
        let mut value: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        value["automatic_quality_release"] = serde_json::json!(true);
        let error = audit_fixture_json(&value.to_string()).unwrap_err();
        assert!(error.contains("advisory"));
    }
}
