// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline, fail-closed audit for plasma-energy source registers and synthetic
//! high-temperature thermal-storage energy-flow fixtures.
//!
//! This module does not simulate plasma, heat transfer, a thermal store, or a
//! physical plant. It validates source metadata and declared accounting
//! identities. A balanced synthetic scenario is not a physical observation.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const EXPECTED_ENERGY_UNIT: &str = "J";
const SYNTHETIC_EVIDENCE_CLASS: &str = "SYNTHETIC_ACCOUNTING_FIXTURE";

#[derive(Debug, Clone, Serialize)]
pub struct SourceManifestSummary {
    pub path: String,
    pub programme: String,
    pub case_count: usize,
    pub negative_control_count: usize,
    pub derived_reference_check_count: usize,
    pub benchmark_target_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ScenarioDisposition {
    Pass,
    Blocked,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ComparisonDisposition {
    Comparable,
    NotComparable,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum MeteringMode {
    NonOverlappingBranchMeters,
    TotalMeterOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum InputRole {
    PrimaryProcess,
    Auxiliary,
    Total,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThermalFixtureManifest {
    schema_version: String,
    programme: String,
    generated_date: String,
    energy_unit: String,
    claim_ceiling: String,
    scenarios: Vec<ThermalScenario>,
    comparisons: Vec<ComparisonPair>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThermalScenario {
    scenario_id: String,
    technology: String,
    evidence_class: String,
    energy_unit: String,
    /// Exact declared wall-plug input boundary, including the included auxiliary systems.
    energy_boundary_contract_id: String,
    service: ServiceBoundary,
    metering_mode: MeteringMode,
    wall_plug_inputs: Vec<EnergyInput>,
    charge: ChargeAccounting,
    hold: HoldAccounting,
    discharge: DischargeAccounting,
    expected_disposition: ScenarioDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceBoundary {
    service_contract_id: String,
    service_type: String,
    #[serde(rename = "outlet_temperature_mK")]
    outlet_temperature_mk: Option<u64>,
    outlet_pressure_pa: Option<u64>,
    mass_flow_mg_s: Option<u64>,
    delivery_duration_s: u64,
    delivery_trace_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnergyInput {
    input_id: String,
    meter_boundary_id: String,
    role: InputRole,
    energy_j: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChargeAccounting {
    stored_energy_increase_j: u64,
    charge_conversion_loss_j: u64,
    auxiliary_dissipation_j: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HoldAccounting {
    stored_energy_after_hold_j: u64,
    storage_hold_loss_j: u64,
    hold_duration_s: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DischargeAccounting {
    stored_energy_withdrawn_j: u64,
    useful_heat_delivered_j: u64,
    discharge_loss_j: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComparisonPair {
    comparison_id: String,
    baseline_scenario_id: String,
    candidate_scenario_id: String,
    expected_disposition: ComparisonDisposition,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub technology: String,
    pub disposition: ScenarioDisposition,
    pub reason: String,
    pub wall_plug_input_j: Option<u64>,
    pub useful_heat_delivered_j: Option<u64>,
    /// Parts per million, derived only for complete and balanced scenarios.
    pub wall_plug_to_useful_heat_ppm: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonResult {
    pub comparison_id: String,
    pub disposition: ComparisonDisposition,
    pub reason: String,
    pub baseline_scenario_id: String,
    pub candidate_scenario_id: String,
    pub baseline_efficiency_ppm: Option<u64>,
    pub candidate_efficiency_ppm: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub schema_version: String,
    pub programme: String,
    pub generated_date: String,
    pub source_manifests: Vec<SourceManifestSummary>,
    pub scenario_results: Vec<ScenarioResult>,
    pub comparison_results: Vec<ComparisonResult>,
    pub authored_fixture_scenarios: usize,
    pub authored_comparisons: usize,
    pub all_expected_dispositions_matched: bool,
    pub claim_ceiling: String,
}

pub fn run_audit(
    plasma_source_path: &Path,
    hts_source_path: &Path,
    thermal_fixture_path: &Path,
) -> Result<AuditReport> {
    let plasma = audit_source_manifest(plasma_source_path)?;
    let hts = audit_source_manifest(hts_source_path)?;

    let fixture_bytes = fs::read(thermal_fixture_path)
        .with_context(|| format!("read thermal fixture {}", thermal_fixture_path.display()))?;
    let fixture: ThermalFixtureManifest = serde_json::from_slice(&fixture_bytes)
        .with_context(|| format!("parse thermal fixture {}", thermal_fixture_path.display()))?;

    validate_fixture_header(&fixture)?;

    let mut scenario_ids = BTreeSet::new();
    let mut scenario_results = BTreeMap::<String, ScenarioResult>::new();
    let mut scenario_sources = BTreeMap::<String, &ThermalScenario>::new();

    for scenario in &fixture.scenarios {
        ensure!(
            !scenario.scenario_id.trim().is_empty(),
            "thermal fixture contains an empty scenario ID"
        );
        ensure!(
            scenario_ids.insert(scenario.scenario_id.clone()),
            "duplicate thermal scenario ID: {}",
            scenario.scenario_id
        );
        let result = evaluate_scenario(scenario);
        ensure!(
            result.disposition == scenario.expected_disposition,
            "scenario {} expected {:?}, evaluator returned {:?}: {}",
            scenario.scenario_id,
            scenario.expected_disposition,
            result.disposition,
            result.reason
        );
        scenario_sources.insert(scenario.scenario_id.clone(), scenario);
        scenario_results.insert(scenario.scenario_id.clone(), result);
    }

    let mut comparison_ids = BTreeSet::new();
    let mut comparison_results = Vec::with_capacity(fixture.comparisons.len());
    for comparison in &fixture.comparisons {
        ensure!(
            !comparison.comparison_id.trim().is_empty(),
            "thermal fixture contains an empty comparison ID"
        );
        ensure!(
            comparison_ids.insert(comparison.comparison_id.clone()),
            "duplicate comparison ID: {}",
            comparison.comparison_id
        );
        let result = evaluate_comparison(comparison, &scenario_sources, &scenario_results);
        ensure!(
            result.disposition == comparison.expected_disposition,
            "comparison {} expected {:?}, evaluator returned {:?}: {}",
            comparison.comparison_id,
            comparison.expected_disposition,
            result.disposition,
            result.reason
        );
        comparison_results.push(result);
    }

    comparison_results.sort_by(|a, b| a.comparison_id.cmp(&b.comparison_id));
    let mut ordered_scenario_results: Vec<_> = scenario_results.into_values().collect();
    ordered_scenario_results.sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));

    Ok(AuditReport {
        schema_version: fixture.schema_version,
        programme: fixture.programme,
        generated_date: fixture.generated_date,
        source_manifests: vec![plasma, hts],
        authored_fixture_scenarios: ordered_scenario_results.len(),
        authored_comparisons: comparison_results.len(),
        scenario_results: ordered_scenario_results,
        comparison_results,
        all_expected_dispositions_matched: true,
        claim_ceiling: fixture.claim_ceiling,
    })
}

fn audit_source_manifest(path: &Path) -> Result<SourceManifestSummary> {
    let bytes =
        fs::read(path).with_context(|| format!("read source manifest {}", path.display()))?;
    let root: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse source manifest {}", path.display()))?;
    audit_source_value(&root, &path.display().to_string())
}

fn validate_fixture_header(fixture: &ThermalFixtureManifest) -> Result<()> {
    ensure!(
        fixture.schema_version == "0.1.0",
        "unsupported thermal fixture schema version: {}",
        fixture.schema_version
    );
    ensure!(
        fixture.programme == "THERMAL-STORE-INDUSTRIAL-001",
        "unexpected thermal fixture programme: {}",
        fixture.programme
    );
    ensure!(
        fixture.energy_unit == EXPECTED_ENERGY_UNIT,
        "thermal fixture energy unit must be J, got {}",
        fixture.energy_unit
    );
    ensure!(
        fixture.claim_ceiling == "SYNTHETIC_ACCOUNTING_FIXTURE_ONLY",
        "thermal fixture must preserve the synthetic-only claim ceiling"
    );
    ensure!(
        !fixture.generated_date.trim().is_empty(),
        "thermal fixture generated_date must not be empty"
    );
    ensure!(
        !fixture.scenarios.is_empty(),
        "thermal fixture has no scenarios"
    );
    ensure!(
        !fixture.comparisons.is_empty(),
        "thermal fixture has no comparison controls"
    );
    Ok(())
}

fn evaluate_scenario(scenario: &ThermalScenario) -> ScenarioResult {
    let reject = |reason: &str| {
        scenario_result(
            scenario,
            ScenarioDisposition::Rejected,
            reason,
            None,
            None,
            None,
        )
    };
    let blocked = |reason: &str| {
        scenario_result(
            scenario,
            ScenarioDisposition::Blocked,
            reason,
            None,
            None,
            None,
        )
    };

    if scenario.scenario_id.trim().is_empty() || scenario.technology.trim().is_empty() {
        return reject("scenario and technology identities must be non-empty");
    }
    if scenario.evidence_class != SYNTHETIC_EVIDENCE_CLASS {
        return reject("fixture scenario evidence class must remain synthetic");
    }
    if scenario.energy_unit != EXPECTED_ENERGY_UNIT {
        return reject("scenario energy unit must be joules (J)");
    }
    if scenario.energy_boundary_contract_id.trim().is_empty() {
        return blocked("declared wall-plug energy-boundary contract identity is missing");
    }

    if scenario.service.service_contract_id.trim().is_empty()
        || scenario.service.service_type.trim().is_empty()
        || scenario
            .service
            .outlet_temperature_mk
            .is_none_or(|value| value == 0)
        || scenario.service.delivery_duration_s == 0
        || scenario.service.delivery_trace_digest.trim().is_empty()
    {
        return blocked("service boundary is incomplete; comparison is not admissible");
    }

    if scenario.wall_plug_inputs.is_empty() {
        return blocked("no wall-plug input meters were supplied");
    }

    let mut input_ids = BTreeSet::new();
    let mut meter_ids = BTreeSet::new();
    let mut total_input_j = 0_u64;
    let mut primary_input_j = 0_u64;
    let mut auxiliary_input_j = 0_u64;
    let mut auxiliary_meter_count = 0_usize;
    let mut total_meter_count = 0_usize;

    for input in &scenario.wall_plug_inputs {
        if input.input_id.trim().is_empty() || input.meter_boundary_id.trim().is_empty() {
            return reject("input and meter-boundary identities must be non-empty");
        }
        if !input_ids.insert(input.input_id.clone()) {
            return reject("duplicate wall-plug input identity");
        }
        if !meter_ids.insert(input.meter_boundary_id.clone()) {
            return reject("duplicate meter boundary; overlapping readings cannot be summed");
        }
        let Some(energy_j) = input.energy_j else {
            return blocked("one or more wall-plug input energies are unknown");
        };
        let Some(next_total) = total_input_j.checked_add(energy_j) else {
            return reject("wall-plug input sum overflow");
        };
        total_input_j = next_total;

        match input.role {
            InputRole::PrimaryProcess => {
                let Some(next) = primary_input_j.checked_add(energy_j) else {
                    return reject("primary-process input sum overflow");
                };
                primary_input_j = next;
            }
            InputRole::Auxiliary => {
                auxiliary_meter_count += 1;
                let Some(next) = auxiliary_input_j.checked_add(energy_j) else {
                    return reject("auxiliary input sum overflow");
                };
                auxiliary_input_j = next;
            }
            InputRole::Total => total_meter_count += 1,
        }
    }

    match scenario.metering_mode {
        MeteringMode::TotalMeterOnly => {
            if scenario.wall_plug_inputs.len() != 1
                || total_meter_count != 1
                || scenario.wall_plug_inputs[0].role != InputRole::Total
            {
                return reject("total-meter mode requires exactly one exclusive TOTAL meter");
            }
        }
        MeteringMode::NonOverlappingBranchMeters => {
            if total_meter_count != 0 || primary_input_j == 0 {
                return reject("branch-meter mode requires primary input and forbids TOTAL meters");
            }
            if auxiliary_meter_count == 0 {
                return blocked(
                    "no auxiliary meter was supplied; absence is not evidence of zero auxiliary energy",
                );
            }
        }
    }

    if total_input_j == 0 {
        return reject("wall-plug input energy must be positive");
    }

    let charge = &scenario.charge;
    let hold = &scenario.hold;
    let discharge = &scenario.discharge;

    if scenario.metering_mode == MeteringMode::NonOverlappingBranchMeters {
        let Some(primary_expected) = charge
            .stored_energy_increase_j
            .checked_add(charge.charge_conversion_loss_j)
        else {
            return reject("charge-stage energy sum overflow");
        };
        if primary_input_j != primary_expected {
            return reject(
                "primary-process input does not close to stored-energy increase plus charge loss",
            );
        }
        if auxiliary_input_j != charge.auxiliary_dissipation_j {
            return reject("auxiliary meters do not close to declared auxiliary dissipation");
        }
    } else {
        let Some(charge_total) = charge
            .stored_energy_increase_j
            .checked_add(charge.charge_conversion_loss_j)
            .and_then(|value| value.checked_add(charge.auxiliary_dissipation_j))
        else {
            return reject("charge-stage energy sum overflow");
        };
        if total_input_j != charge_total {
            return reject(
                "exclusive total meter does not close to charge-stage stored energy and losses",
            );
        }
    }

    let Some(hold_reconstructed) = hold
        .stored_energy_after_hold_j
        .checked_add(hold.storage_hold_loss_j)
    else {
        return reject("hold-stage energy sum overflow");
    };
    if hold_reconstructed != charge.stored_energy_increase_j {
        return reject(
            "charge-end stored energy does not close to post-hold energy plus hold loss",
        );
    }
    if hold.hold_duration_s == 0 {
        return blocked("storage hold duration is absent");
    }

    let Some(discharge_reconstructed) = discharge
        .useful_heat_delivered_j
        .checked_add(discharge.discharge_loss_j)
    else {
        return reject("discharge-stage energy sum overflow");
    };
    if discharge.stored_energy_withdrawn_j != hold.stored_energy_after_hold_j
        || discharge_reconstructed != discharge.stored_energy_withdrawn_j
    {
        return reject(
            "post-hold energy does not close to useful delivered heat plus discharge loss",
        );
    }
    if discharge.useful_heat_delivered_j == 0 {
        return reject(
            "positive-duration heat service cannot PASS with zero useful heat delivered",
        );
    }

    let Some(all_losses_j) = charge
        .charge_conversion_loss_j
        .checked_add(charge.auxiliary_dissipation_j)
        .and_then(|value| value.checked_add(hold.storage_hold_loss_j))
        .and_then(|value| value.checked_add(discharge.discharge_loss_j))
    else {
        return reject("end-to-end loss sum overflow");
    };
    let Some(end_to_end_reconstructed) =
        discharge.useful_heat_delivered_j.checked_add(all_losses_j)
    else {
        return reject("end-to-end energy reconstruction overflow");
    };
    if end_to_end_reconstructed != total_input_j {
        return reject(
            "end-to-end wall-plug input does not close to useful heat plus all declared losses",
        );
    }

    let efficiency = ((discharge.useful_heat_delivered_j as u128) * 1_000_000_u128
        / (total_input_j as u128)) as u64;

    scenario_result(
        scenario,
        ScenarioDisposition::Pass,
        "all meter, service, charge, hold, discharge and end-to-end accounting gates passed",
        Some(total_input_j),
        Some(discharge.useful_heat_delivered_j),
        Some(efficiency),
    )
}

fn scenario_result(
    scenario: &ThermalScenario,
    disposition: ScenarioDisposition,
    reason: &str,
    wall_plug_input_j: Option<u64>,
    useful_heat_delivered_j: Option<u64>,
    wall_plug_to_useful_heat_ppm: Option<u64>,
) -> ScenarioResult {
    ScenarioResult {
        scenario_id: scenario.scenario_id.clone(),
        technology: scenario.technology.clone(),
        disposition,
        reason: reason.to_owned(),
        wall_plug_input_j,
        useful_heat_delivered_j,
        wall_plug_to_useful_heat_ppm,
    }
}

fn evaluate_comparison(
    comparison: &ComparisonPair,
    scenarios: &BTreeMap<String, &ThermalScenario>,
    results: &BTreeMap<String, ScenarioResult>,
) -> ComparisonResult {
    let baseline = scenarios.get(&comparison.baseline_scenario_id);
    let candidate = scenarios.get(&comparison.candidate_scenario_id);
    let baseline_result = results.get(&comparison.baseline_scenario_id);
    let candidate_result = results.get(&comparison.candidate_scenario_id);

    let (Some(baseline), Some(candidate), Some(baseline_result), Some(candidate_result)) =
        (baseline, candidate, baseline_result, candidate_result)
    else {
        return comparison_result(
            comparison,
            ComparisonDisposition::Blocked,
            "one or both referenced scenarios are missing",
            None,
            None,
        );
    };

    if baseline_result.disposition != ScenarioDisposition::Pass
        || candidate_result.disposition != ScenarioDisposition::Pass
    {
        return comparison_result(
            comparison,
            ComparisonDisposition::Blocked,
            "one or both scenarios did not pass their own accounting gates",
            baseline_result.wall_plug_to_useful_heat_ppm,
            candidate_result.wall_plug_to_useful_heat_ppm,
        );
    }

    if baseline.service.service_contract_id != candidate.service.service_contract_id
        || baseline.service != candidate.service
        || baseline.energy_boundary_contract_id != candidate.energy_boundary_contract_id
    {
        return comparison_result(
            comparison,
            ComparisonDisposition::NotComparable,
            "functional service contract or complete wall-plug energy-boundary contract differs",
            baseline_result.wall_plug_to_useful_heat_ppm,
            candidate_result.wall_plug_to_useful_heat_ppm,
        );
    }

    comparison_result(
        comparison,
        ComparisonDisposition::Comparable,
        "both complete scenarios deliver the exact same declared service boundary",
        baseline_result.wall_plug_to_useful_heat_ppm,
        candidate_result.wall_plug_to_useful_heat_ppm,
    )
}

fn comparison_result(
    comparison: &ComparisonPair,
    disposition: ComparisonDisposition,
    reason: &str,
    baseline_efficiency_ppm: Option<u64>,
    candidate_efficiency_ppm: Option<u64>,
) -> ComparisonResult {
    ComparisonResult {
        comparison_id: comparison.comparison_id.clone(),
        disposition,
        reason: reason.to_owned(),
        baseline_scenario_id: comparison.baseline_scenario_id.clone(),
        candidate_scenario_id: comparison.candidate_scenario_id.clone(),
        baseline_efficiency_ppm,
        candidate_efficiency_ppm,
    }
}

pub fn print_report(report: &AuditReport, as_json: bool) -> Result<()> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    println!("Programme: {}", report.programme);
    println!("Schema: {}", report.schema_version);
    println!("Generated: {}", report.generated_date);
    println!("Claim ceiling: {}", report.claim_ceiling);
    for source in &report.source_manifests {
        println!(
            "SOURCE PASS {} cases={} negative_controls={} derived_checks={} benchmark_targets={} path={}",
            source.programme,
            source.case_count,
            source.negative_control_count,
            source.derived_reference_check_count,
            source.benchmark_target_count,
            source.path
        );
    }
    println!("Thermal scenarios:");
    for scenario in &report.scenario_results {
        let efficiency = scenario
            .wall_plug_to_useful_heat_ppm
            .map(|value| format!("{:.3}%", value as f64 / 10_000.0))
            .unwrap_or_else(|| "n/a".to_owned());
        println!(
            "  {:?} {} technology={} efficiency={} reason={}",
            scenario.disposition,
            scenario.scenario_id,
            scenario.technology,
            efficiency,
            scenario.reason
        );
    }
    println!("Comparisons:");
    for comparison in &report.comparison_results {
        println!(
            "  {:?} {} baseline={} candidate={} reason={}",
            comparison.disposition,
            comparison.comparison_id,
            comparison.baseline_scenario_id,
            comparison.candidate_scenario_id,
            comparison.reason
        );
    }
    println!(
        "Expected dispositions matched: {}",
        report.all_expected_dispositions_matched
    );
    Ok(())
}

fn audit_source_value(root: &Value, display_name: &str) -> Result<SourceManifestSummary> {
    let programme = root
        .get("programme")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .with_context(|| format!("{display_name}: no non-empty programme"))?
        .to_owned();
    let claim_ceiling = root
        .get("claim_ceiling")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .with_context(|| format!("{display_name}: no explicit claim ceiling"))?;
    ensure!(
        !claim_ceiling.trim().is_empty(),
        "{display_name}: claim ceiling must not be empty"
    );

    let evidence_classes = root
        .get("evidence_classes")
        .and_then(Value::as_array)
        .with_context(|| format!("{display_name}: no evidence_classes array"))?;
    let mut admitted_classes = BTreeSet::new();
    for class in evidence_classes {
        let class = class
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{display_name}: empty/non-string evidence class"))?;
        ensure!(
            admitted_classes.insert(class.to_owned()),
            "{display_name}: duplicate declared evidence class {class}"
        );
    }
    ensure!(
        !admitted_classes.is_empty(),
        "{display_name}: no declared evidence classes"
    );

    let cases = root
        .get("cases")
        .and_then(Value::as_array)
        .with_context(|| format!("{display_name}: no cases array"))?;
    ensure!(!cases.is_empty(), "{display_name}: no source cases");

    let mut case_ids = BTreeSet::new();
    for case in cases {
        let case_id = case
            .get("case_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{display_name}: source case has no case_id"))?;
        ensure!(
            case_ids.insert(case_id.to_owned()),
            "{display_name}: duplicate source case ID {case_id}"
        );
        let evidence_class = case
            .get("evidence_class")
            .and_then(Value::as_str)
            .with_context(|| format!("{display_name}: case {case_id} has no evidence_class"))?;
        ensure!(
            admitted_classes.contains(evidence_class),
            "{display_name}: case {case_id} uses undeclared evidence class {evidence_class}"
        );
        ensure!(
            case.get("luminous_reproduction").and_then(Value::as_bool) == Some(false),
            "{display_name}: case {case_id} must explicitly preserve luminous_reproduction=false"
        );
        let source_url = case
            .get("source")
            .and_then(|source| source.get("url"))
            .and_then(Value::as_str)
            .filter(|url| has_http_authority(url))
            .with_context(|| format!("{display_name}: case {case_id} has no valid HTTP(S) source URL with a non-empty authority"))?;
        ensure!(
            !source_url.trim().is_empty(),
            "{display_name}: case {case_id} has empty source URL"
        );

        if case_id == "CO2-COUNTERFLOW-ARC-EXPERIMENT-2026" {
            validate_co2_counterflow_operating_points(case, display_name)?;
        }
        if case_id == "NH3-PLASMA-SYNTHESIS-ROUTES-TEA-2026" {
            validate_nh3_synthesis_routes_case(case, display_name)?;
        }
        if case_id == "CO2-DBD-ML-358-DATASET-2026" {
            validate_co2_dbd_ml_case(case, display_name)?;
        }
        if case_id == "CO2-RWGS-GLIDING-ARC-2026" {
            validate_co2_rwgs_operating_points(case, display_name)?;
        }
        if case_id == "CO2-CYCLIC-REFORMING-GASIFICATION-2026" {
            validate_co2_crg_operating_points(case, display_name)?;
        }

        if case_id == "NH3-CRACK-MW-ATM-MECHANISM-2026" {
            validate_nh3_operating_points(case, display_name)?;
        }
        if case_id == "CH4-PLASMA-PULSED-HEATING-REGEN-2026" {
            validate_ch4_pph_operating_points(case, display_name)?;
        }
        if case_id == "CH4-PLASMA-THERMAL-APPJ-2026" {
            validate_ch4_thermal_appj_case(case, display_name)?;
        }
        if case_id == "NH3-PLASMA-ELECTROCATALYSIS-JACS-2026" {
            validate_nh3_n2_vibrational_transfer_case(case, display_name)?;
        }
        let lane = case.get("lane").and_then(Value::as_str);
        if lane == Some("METHANE_PYROLYSIS")
            || lane == Some("METHANE_PLASMA_THERMAL_APPJ")
        {
            let coproduct = case
                .get("coproduct_accounting")
                .with_context(|| {
                    format!("{display_name}: methane-pyrolysis case {case_id} has no coproduct_accounting contract")
                })?;
            let policy = coproduct
                .get("policy")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .with_context(|| {
                    format!("{display_name}: methane-pyrolysis case {case_id} has no coproduct accounting policy")
                })?;
            ensure!(
                matches!(
                    policy,
                    "NO_SOURCE_COPRODUCT_CREDIT_REPORTED"
                        | "MODELED_CARBON_BLACK_SALES_REVENUE"
                        | "AVOIDED_PRODUCT_CARBON_BLACK_CREDIT"
                        | "REVIEW_LEVEL_NO_SPECIFIC_CREDIT_ASSUMPTION"
                ),
                "{display_name}: methane-pyrolysis case {case_id} has unsupported coproduct policy {policy}"
            );
            ensure!(
                coproduct.get("credit_claim_status").and_then(Value::as_str)
                    == Some("SOURCE_MODEL_ONLY"),
                "{display_name}: methane-pyrolysis case {case_id} must label carbon credit as SOURCE_MODEL_ONLY"
            );
            ensure!(
                coproduct
                    .get("carbon_product_qualification_status")
                    .and_then(Value::as_str)
                    == Some("NOT_VALIDATED"),
                "{display_name}: methane-pyrolysis case {case_id} must not promote carbon-product grade to validated"
            );
            ensure!(
                coproduct
                    .get("carbon_market_capacity_status")
                    .and_then(Value::as_str)
                    == Some("NOT_VALIDATED"),
                "{display_name}: methane-pyrolysis case {case_id} must not promote carbon-market capacity to validated"
            );
            ensure!(
                coproduct.get("offtake_status").and_then(Value::as_str) == Some("NOT_VALIDATED"),
                "{display_name}: methane-pyrolysis case {case_id} must not infer verified offtake"
            );
            ensure!(
                coproduct
                    .get("realized_revenue_evidence")
                    .and_then(Value::as_bool)
                    == Some(false),
                "{display_name}: methane-pyrolysis case {case_id} cannot claim realized carbon-black revenue"
            );
            ensure!(
                coproduct
                    .get("no_credit_baseline_required")
                    .and_then(Value::as_bool)
                    == Some(true),
                "{display_name}: methane-pyrolysis case {case_id} must preserve a no-credit baseline"
            );
        }

        if lane == Some("FUSION_DIRECT_ENERGY_CONVERSION")
            || lane == Some("FUSION_FUEL_ENERGY_PARTITION")
        {
            let particle_origin = case
                .get("particle_energy_origin")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .with_context(|| {
                    format!("{display_name}: fusion DEC/fuel-partition case {case_id} has no explicit particle_energy_origin")
                })?;
            ensure!(
                !particle_origin.trim().is_empty(),
                "{display_name}: fusion DEC/fuel-partition case {case_id} has an empty particle_energy_origin"
            );
            ensure!(
                case.get("fusion_born_output_demonstrated")
                    .and_then(Value::as_bool)
                    == Some(false),
                "{display_name}: fusion DEC/fuel-partition case {case_id} must explicitly preserve fusion_born_output_demonstrated=false until source-matched conversion output is admitted"
            );
            ensure!(
                case.get("comparison_status")
                    .and_then(Value::as_str)
                    .is_some(),
                "{display_name}: fusion DEC/fuel-partition case {case_id} has no comparison_status"
            );
        }
    }

    let derived_checks: &[Value] = match root.get("derived_reference_checks") {
        None => &[],
        Some(value) => value.as_array().with_context(|| {
            format!("{display_name}: derived_reference_checks must be an array")
        })?,
    };
    let mut derived_check_ids = BTreeSet::new();
    for check in derived_checks {
        let check_id = check
            .get("check_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{display_name}: derived reference check has no check_id"))?;
        ensure!(
            derived_check_ids.insert(check_id.to_owned()),
            "{display_name}: duplicate derived reference check ID {check_id}"
        );

        let evidence_class = check
            .get("evidence_class")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| {
                format!("{display_name}: derived check {check_id} has no evidence_class")
            })?;
        ensure!(
            evidence_class.starts_with("DERIVED_ARITHMETIC_"),
            "{display_name}: derived check {check_id} must use a DERIVED_ARITHMETIC evidence class"
        );
        ensure!(
            check.get("luminous_measurement").and_then(Value::as_bool) == Some(false),
            "{display_name}: derived check {check_id} must preserve luminous_measurement=false"
        );
        ensure!(
            check
                .get("inputs")
                .and_then(Value::as_object)
                .is_some_and(|value| !value.is_empty()),
            "{display_name}: derived check {check_id} has no input values"
        );
        ensure!(
            check
                .get("derivation")
                .and_then(Value::as_object)
                .is_some_and(|value| !value.is_empty()),
            "{display_name}: derived check {check_id} has no derivation values"
        );
        ensure!(
            check
                .get("interpretation")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty()),
            "{display_name}: derived check {check_id} has no interpretation/claim ceiling"
        );
        if let Some(source_case_id) = check.get("source_case_id").and_then(Value::as_str) {
            ensure!(
                case_ids.contains(source_case_id),
                "{display_name}: derived check {check_id} references unknown source case {source_case_id}"
            );
        }
        if let Some(reference_case_id) = check.get("reference_case_id").and_then(Value::as_str) {
            ensure!(
                case_ids.contains(reference_case_id),
                "{display_name}: derived check {check_id} references unknown reference case {reference_case_id}"
            );
        }

        if check_id == "NH3-MW-ENERGY-YIELD-TO-SPECIFIC-ENERGY-2026" {
            validate_nh3_energy_yield_check(check, display_name)?;
        }
        if check_id == "NH3-PLASMA-TO-SOEC-HB-ENERGY-GAP-2026" {
            validate_nh3_synthesis_energy_gap_check(check, root, display_name)?;
        }
        if check_id == "NH3-PLASMA-SYNTHESIS-RECIPROCAL-YIELD-2026" {
            validate_nh3_synthesis_energy_reciprocal_check(check, root, display_name)?;
        }
    }

    let benchmark_targets: &[Value] = match root.get("benchmark_targets") {
        None => &[],
        Some(value) => value
            .as_array()
            .with_context(|| format!("{display_name}: benchmark_targets must be an array"))?,
    };
    let mut benchmark_target_ids = BTreeSet::new();
    for target in benchmark_targets {
        let target_id = target
            .get("target_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{display_name}: benchmark target has no target_id"))?;
        ensure!(
            benchmark_target_ids.insert(target_id.to_owned()),
            "{display_name}: duplicate benchmark target ID {target_id}"
        );
        ensure!(
            target.get("status").and_then(Value::as_str) == Some("RESEARCH_TARGET_NOT_RESULT"),
            "{display_name}: benchmark target {target_id} must be marked RESEARCH_TARGET_NOT_RESULT"
        );
        let lane = target
            .get("lane")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{display_name}: benchmark target {target_id} has no lane"))?;
        let comparators = target
            .get("comparator_case_ids")
            .and_then(Value::as_array)
            .filter(|value| !value.is_empty())
            .with_context(|| {
                format!("{display_name}: benchmark target {target_id} has no comparator_case_ids")
            })?;
        let mut comparator_ids = BTreeSet::new();
        for comparator in comparators {
            let comparator_id = comparator
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .with_context(|| {
                    format!(
                        "{display_name}: benchmark target {target_id} has an empty comparator ID"
                    )
                })?;
            ensure!(
                comparator_ids.insert(comparator_id.to_owned()),
                "{display_name}: benchmark target {target_id} has duplicate comparator {comparator_id}"
            );
            ensure!(
                case_ids.contains(comparator_id),
                "{display_name}: benchmark target {target_id} references unknown comparator case {comparator_id}"
            );
        }
        ensure!(
            target
                .get("metric_thresholds")
                .and_then(Value::as_object)
                .is_some_and(|value| !value.is_empty()),
            "{display_name}: benchmark target {target_id} has no metric_thresholds"
        );
        let guards = target
            .get("claim_guards")
            .and_then(Value::as_object)
            .filter(|value| !value.is_empty())
            .with_context(|| {
                format!("{display_name}: benchmark target {target_id} has no claim_guards")
            })?;

        match lane {
            "AMMONIA_CRACKING" => {
                let metrics = target.get("metric_thresholds").expect("validated above");
                ensure!(
                    metrics
                        .get("all_thresholds_must_hold_at_same_operating_point")
                        .and_then(Value::as_bool)
                        == Some(true),
                    "{display_name}: ammonia benchmark target {target_id} must bind all metrics to one operating point"
                );
                ensure!(
                    guards
                        .get("full_wall_plug_to_accepted_h2_boundary_required")
                        .and_then(Value::as_bool)
                        == Some(true),
                    "{display_name}: ammonia benchmark target {target_id} must require full wall-plug-to-product accounting"
                );
                for key in [
                    "ammonia_conversion_min_fraction",
                    "hydrogen_energy_yield_min_g_H2_per_kWh",
                    "hydrogen_production_rate_min_g_H2_per_hour",
                ] {
                    ensure!(
                        metrics
                            .get(key)
                            .and_then(Value::as_f64)
                            .is_some_and(|value| value.is_finite() && value > 0.0),
                        "{display_name}: ammonia benchmark target {target_id} has invalid or missing threshold {key}"
                    );
                }
            }
            "METHANE_PYROLYSIS_PULSED_HEATING" => {
                let metrics = target.get("metric_thresholds").expect("validated above");
                ensure!(
                    metrics.get("all_thresholds_must_hold_at_same_operating_point_and_feed")
                        .and_then(Value::as_bool) == Some(true),
                    "{display_name}: PPH target {target_id} must bind conversion, selectivity and energy to one feed/operating point"
                );
                for (key, minimum) in [
                    ("methane_conversion_min_fraction", 0.90),
                    ("hydrogen_selectivity_min_fraction", 0.99),
                    ("minimum_reaction_regeneration_cycles", 100.0),
                    ("accepted_h2_productivity_retention_after_cycles_min_fraction", 0.90),
                ] {
                    ensure!(
                        metrics.get(key).and_then(Value::as_f64)
                            .is_some_and(|value| value.is_finite() && value >= minimum),
                        "{display_name}: PPH target {target_id} has an insufficient or invalid threshold {key}"
                    );
                }
                ensure!(
                    close_f64(required_f64(metrics, "accepted_h2_specific_energy_max_kWh_per_kg", &format!("{display_name}: {target_id}"))?, 25.0),
                    "{display_name}: PPH target {target_id} must retain the <=25 kWh/kg accepted-H2 stretch threshold"
                );
                ensure!(
                    target.get("credit_policies").and_then(Value::as_array)
                        .is_some_and(|items| items.iter().any(|item| item.as_str() == Some("NO_CREDIT"))),
                    "{display_name}: PPH target {target_id} must include a no-credit carbon baseline"
                );
                for key in [
                    "diluted_and_pure_methane_operating_points_must_remain_separate",
                    "full_grid_to_accepted_h2_energy_boundary_required",
                    "gas_separation_purification_and_compression_included",
                    "co2_plasma_regeneration_energy_and_carbon_streams_included",
                    "methane_derived_carbon_balance_and_unknowns_reported",
                    "no_credit_carbon_baseline_required",
                    "catalyst_support_electrode_and_replacement_lifecycle_included",
                    "long_run_cycles_and_failed_batches_in_denominator",
                    "matched_shock_wave_and_microwave_comparators_required",
                ] {
                    ensure!(
                        guards.get(key).and_then(Value::as_bool) == Some(true),
                        "{display_name}: PPH target {target_id} is missing required guard {key}"
                    );
                }
                for comparator in [
                    "CH4-PLASMA-PULSED-HEATING-REGEN-2026",
                    "METHANE-PYROLYSIS-HEATING-REVIEW-2026",
                ] {
                    ensure!(
                        comparators.iter().any(|value| value.as_str() == Some(comparator)),
                        "{display_name}: PPH target {target_id} must reference comparator {comparator}"
                    );
                }
            }
            "METHANE_PYROLYSIS" => {
                let metrics = target.get("metric_thresholds").expect("validated above");
                ensure!(
                    metrics
                        .get("all_thresholds_must_hold_at_same_operating_point")
                        .and_then(Value::as_bool)
                        == Some(true),
                    "{display_name}: methane benchmark target {target_id} must bind metrics to one operating point"
                );
                ensure!(
                    target
                        .get("credit_policies")
                        .and_then(Value::as_array)
                        .is_some_and(|value| value
                            .iter()
                            .any(|item| item.as_str() == Some("NO_CREDIT"))),
                    "{display_name}: methane benchmark target {target_id} must include a NO_CREDIT scenario"
                );
                ensure!(
                    guards
                        .get("no_credit_baseline_required")
                        .and_then(Value::as_bool)
                        == Some(true),
                    "{display_name}: methane benchmark target {target_id} must require a no-credit baseline"
                );
                ensure!(
                    guards
                        .get(
                            "avoided_product_credit_requires_independent_carbon_grade_qualification"
                        )
                        .and_then(Value::as_bool)
                        == Some(true),
                    "{display_name}: methane benchmark target {target_id} must require independent carbon product qualification"
                );
                for key in [
                    "methane_conversion_min_fraction",
                    "hydrogen_selectivity_min_fraction",
                    "levelized_cost_hydrogen_max_usd_per_kg",
                ] {
                    ensure!(
                        metrics
                            .get(key)
                            .and_then(Value::as_f64)
                            .is_some_and(|value| value.is_finite() && value > 0.0),
                        "{display_name}: methane benchmark target {target_id} has invalid or missing threshold {key}"
                    );
                }
            }
            "AMMONIA_SYNTHESIS" => {
                let metrics = target.get("metric_thresholds").expect("validated above");
                ensure!(
                    metrics
                        .get("all_thresholds_must_hold_at_same_operating_point")
                        .and_then(Value::as_bool)
                        == Some(true),
                    "{display_name}: ammonia synthesis target {target_id} must bind metrics to one operating point"
                );
                for key in [
                    "hydrogen_generation_energy_included",
                    "plasma_and_electrolysis_electricity_included",
                    "gas_feed_and_product_separation_energy_included",
                    "accepted_nh3_product_purity_and_pressure_contract_required",
                    "matched_haber_bosch_baseline_includes_hydrogen_energy",
                ] {
                    ensure!(
                        guards.get(key).and_then(Value::as_bool) == Some(true),
                        "{display_name}: ammonia synthesis target {target_id} is missing required full-chain guard {key}"
                    );
                }
                ensure!(
                    metrics
                        .get("ammonia_plant_gate_energy_yield_min_g_NH3_per_kWh")
                        .and_then(Value::as_f64)
                        .is_some_and(|value| value.is_finite() && value > 0.0),
                    "{display_name}: ammonia synthesis target {target_id} has invalid plant-gate energy-yield threshold"
                );
            }
            "AMMONIA_SYNTHESIS_COMPETITIVENESS" => {
                let metrics = target.get("metric_thresholds").expect("validated above");
                ensure!(
                    metrics
                        .get("all_thresholds_must_hold_at_same_operating_point")
                        .and_then(Value::as_bool)
                        == Some(true),
                    "{display_name}: ammonia competitiveness target {target_id} must bind thresholds to one operating point"
                );
                let baseline = required_f64(
                    metrics,
                    "integrated_soec_hb_reference_specific_energy_kWh_per_kg_NH3",
                    &format!("{display_name}: {target_id}"),
                )?;
                let candidate_max = required_f64(
                    metrics,
                    "plasma_candidate_max_specific_energy_kWh_per_kg_NH3",
                    &format!("{display_name}: {target_id}"),
                )?;
                let improvement = required_f64(
                    metrics,
                    "required_improvement_vs_integrated_reference_fraction",
                    &format!("{display_name}: {target_id}"),
                )?;
                ensure!(
                    close_f64(baseline, 8.14),
                    "{display_name}: ammonia competitiveness target {target_id} must retain the 8.14 kWh/kg modeled SOEC-HB reference"
                );
                ensure!(
                    (0.10..1.0).contains(&improvement),
                    "{display_name}: ammonia competitiveness target {target_id} must require at least a 10% improvement"
                );
                ensure!(
                    candidate_max <= baseline * (1.0 - improvement),
                    "{display_name}: ammonia competitiveness target {target_id} candidate limit does not beat the matched SOEC-HB reference by the required margin"
                );
                for key in [
                    "plant_gate_grid_to_accepted_nh3_boundary_required",
                    "hydrogen_generation_nitrogen_separation_and_all_balance_of_plant_included",
                    "same_nh3_purity_pressure_and_delivery_contract_required",
                    "baseline_exclusions_wind_farm_storage_ammonia_storage_and_transport_explicit",
                    "desalination_and_feedwater_conditioning_boundary_reconciled",
                    "availability_capacity_factor_and_duty_cycle_matched",
                    "no_faraday_efficiency_or_area_rate_substitutes_for_full_chain_energy",
                    "no_superiority_claim_before_source_data_reconstruction_and_uncertainty_analysis",
                ] {
                    ensure!(
                        guards.get(key).and_then(Value::as_bool) == Some(true),
                        "{display_name}: ammonia competitiveness target {target_id} is missing required full-chain gate {key}"
                    );
                }
                ensure!(
                    comparators.iter().any(|value| value.as_str() == Some("NH3-SOEC-HB-OFFSHORE-2026")),
                    "{display_name}: ammonia competitiveness target {target_id} must reference the integrated SOEC-HB baseline"
                );
            }
            "AMMONIA_SYNTHESIS_DYNAMIC_FLEXIBILITY" => {
                let metrics = target.get("metric_thresholds").expect("validated above");
                for (key, minimum) in [
                    ("accepted_nh3_min_load_fraction", 0.10),
                    ("accepted_nh3_max_load_fraction", 1.20),
                    ("accepted_nh3_ramp_rate_percent_nominal_per_minute", 3.0),
                ] {
                    ensure!(
                        metrics.get(key).and_then(Value::as_f64)
                            .is_some_and(|value| value.is_finite() && value >= minimum),
                        "{display_name}: ammonia dynamic-flex target {target_id} has an insufficient or invalid envelope threshold {key}"
                    );
                }
                ensure!(
                    metrics.get("year_long_renewable_profile_required").and_then(Value::as_bool) == Some(true)
                        && metrics.get("all_full_chain_quality_and_fatigue_gates_must_hold").and_then(Value::as_bool) == Some(true),
                    "{display_name}: ammonia dynamic-flex target {target_id} must require year-long profile and all quality/fatigue gates"
                );
                for key in [
                    "plant_level_accepted_nh3_ramp_measured_not_plasma_source_ramp",
                    "hydrogen_generation_and_storage_dynamics_included",
                    "nitrogen_and_plasma_or_nox_capture_transfer_dynamics_included",
                    "synthesis_loop_pressure_and_temperature_fatigue_included",
                    "accepted_product_purity_pressure_and_delivery_contract_matched",
                    "startup_shutdown_standby_and_failed_batches_counted",
                    "buffer_capacity_and_renewable_curtailment_reported",
                    "baseline_uses_10_to_120_percent_3_percent_per_minute_haber_bosch_model",
                    "no_flexibility_superiority_claim_without_same_trace_replay",
                ] {
                    ensure!(
                        guards.get(key).and_then(Value::as_bool) == Some(true),
                        "{display_name}: ammonia dynamic-flex target {target_id} is missing required guard {key}"
                    );
                }
                for comparator in [
                    "NH3-HB-ROBUST-LOAD-FLEX-2026",
                    "NH3-HB-FATIGUE-YEAR-SCHEDULING-2026",
                    "NH3-P2A-THREE-LEVEL-DISPATCH-2026",
                ] {
                    ensure!(
                        comparators.iter().any(|value| value.as_str() == Some(comparator)),
                        "{display_name}: ammonia dynamic-flex target {target_id} must reference comparator {comparator}"
                    );
                }
            }
            "AMMONIA_CRACKING_THERMAL_INTEGRATION" => {
                let metrics = target.get("metric_thresholds").expect("validated above");
                ensure!(
                    metrics
                        .get("all_thresholds_must_hold_at_same_operating_point")
                        .and_then(Value::as_bool)
                        == Some(true),
                    "{display_name}: ammonia thermal-integration target {target_id} must bind all improvements to one operating point"
                );
                for (key, minimum) in [
                    ("ammonia_conversion_min_fraction", 0.99),
                    ("accepted_h2_production_rate_min_g_per_hour", 150.0),
                    ("net_grid_specific_energy_reduction_min_fraction", 0.15),
                ] {
                    ensure!(
                        metrics
                            .get(key)
                            .and_then(Value::as_f64)
                            .is_some_and(|value| value.is_finite() && value >= minimum),
                        "{display_name}: ammonia thermal-integration target {target_id} has an insufficient or invalid threshold {key}"
                    );
                }
                let residual = metrics
                    .get("heat_balance_abs_residual_fraction_max")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite() && *value > 0.0 && *value <= 0.05);
                ensure!(
                    residual.is_some(),
                    "{display_name}: ammonia thermal-integration target {target_id} must declare a positive heat-balance residual cap <= 5%"
                );
                for key in [
                    "paired_no_recovery_baseline_same_feed_and_operating_point_required",
                    "full_grid_to_accepted_h2_boundary_required",
                    "recovered_heat_must_be_measured_and_useful_at_declared_process_interface",
                    "insulation_or_recuperator_pressure_drop_and_all_auxiliaries_included",
                    "thermal_storage_and_heat_exchanger_losses_included",
                    "no_double_counting_of_recirculated_heat_or_reaction_enthalpy",
                    "thermal_energy_balance_must_close_with_uncertainty",
                    "product_purity_pressure_and_delivery_contract_unchanged",
                    "duty_cycle_and_steady_state_duration_matched",
                    "comparison_must_report_measurement_uncertainty_and_grouped_repeats",
                ] {
                    ensure!(
                        guards.get(key).and_then(Value::as_bool) == Some(true),
                        "{display_name}: ammonia thermal-integration target {target_id} is missing required guard {key}"
                    );
                }
            }
            "AMMONIA_N2_VIBRATIONAL_TRANSFER" => {
                let metrics = target.get("metric_thresholds").expect("validated above");
                for key in [
                    "all_required_evidence_gates_must_hold",
                    "state_resolved_interface_population_or_qualified_operator_required",
                    "discharge_to_interface_transfer_uncertainty_required",
                    "isotope_attributed_product_control_required",
                    "matched_thermal_and_electrical_controls_required",
                    "full_grid_to_accepted_nh3_boundary_required",
                ] {
                    ensure!(
                        metrics.get(key).and_then(Value::as_bool) == Some(true),
                        "{display_name}: N2 vibrational-transfer target {target_id} is missing required metric gate {key}"
                    );
                }
                for key in [
                    "direct_catalyst_interface_state_measurement_or_independently_validated_measurement_operator_required",
                    "modelled_state_temperature_remains_model_evidence",
                    "measured_intermediate_and_quantitative_flux_remain_distinct_measurands",
                    "plasma_only_and_electrolysis_only_controls_required",
                    "isotope_product_attribution_required_for_nitrogen_source_claim",
                    "catalyst_lot_and_surface_identity_must_match",
                    "all_energy_inputs_and_accepted_product_separation_included",
                    "no_claim_of_unique_vibrational_causality_without_matched_thermal_controls",
                ] {
                    ensure!(
                        guards.get(key).and_then(Value::as_bool) == Some(true),
                        "{display_name}: N2 vibrational-transfer target {target_id} is missing claim guard {key}"
                    );
                }
            }
            "PLASMA_POWER_TO_X_BIOGAS_REFORMING" => {
                let metrics = target.get("metric_thresholds").expect("validated above");
                ensure!(
                    metrics
                        .get("all_thresholds_must_hold_at_same_operating_point")
                        .and_then(Value::as_bool)
                        == Some(true),
                    "{display_name}: Power-to-X target {target_id} must bind cost and efficiency to one operating point"
                );
                for key in [
                    "matched_syngas_composition_and_quality_required",
                    "full_plant_grid_side_energy_boundary_required",
                    "electricity_price_and_capacity_factor_scenario_required",
                    "downstream_product_service_boundary_required",
                    "unqualified_coproduct_credit_forbidden",
                ] {
                    ensure!(
                        guards.get(key).and_then(Value::as_bool) == Some(true),
                        "{display_name}: Power-to-X target {target_id} is missing required guard {key}"
                    );
                }
                for key in [
                    "syngas_cost_eur_per_tonne_max",
                    "plant_efficiency_min_fraction",
                ] {
                    ensure!(
                        metrics
                            .get(key)
                            .and_then(Value::as_f64)
                            .is_some_and(|value| value.is_finite() && value > 0.0),
                        "{display_name}: Power-to-X target {target_id} has invalid or missing threshold {key}"
                    );
                }
            }
            _ => {}
        }
    }

    let controls = root
        .get("negative_controls")
        .and_then(Value::as_array)
        .with_context(|| format!("{display_name}: no negative_controls array"))?;
    ensure!(!controls.is_empty(), "{display_name}: no negative controls");
    let mut control_ids = BTreeSet::new();
    for control in controls {
        let control_id = control
            .get("control_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{display_name}: negative control has no control_id"))?;
        ensure!(
            control_ids.insert(control_id.to_owned()),
            "{display_name}: duplicate negative-control ID {control_id}"
        );
        ensure!(
            control
                .get("expected_disposition")
                .and_then(Value::as_str)
                .is_some(),
            "{display_name}: negative control {control_id} has no expected disposition"
        );
    }

    Ok(SourceManifestSummary {
        path: display_name.to_owned(),
        programme,
        case_count: cases.len(),
        negative_control_count: controls.len(),
        derived_reference_check_count: derived_checks.len(),
        benchmark_target_count: benchmark_targets.len(),
    })
}

fn has_http_authority(url: &str) -> bool {
    if url.trim() != url {
        return false;
    }
    let Some(remainder) = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
    else {
        return false;
    };
    let authority = remainder
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    !authority.is_empty() && !authority.chars().any(char::is_whitespace)
}

fn close_f64(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= 1.0e-9 * actual.abs().max(expected.abs()).max(1.0)
}

fn required_f64(object: &Value, key: &str, context: &str) -> Result<f64> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .with_context(|| format!("{context}: missing/non-finite numeric field {key}"))
}

fn validate_ch4_pph_operating_points(case: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: pulsed-plasma methane operating-point and regeneration audit");
    let points = case
        .get("operating_points")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: operating_points array is missing"))?;
    ensure!(
        points.len() == 2,
        "{context}: expected two distinct source operating points"
    );
    let mut ids = BTreeSet::new();
    for point in points {
        let point_id = point
            .get("operating_point_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{context}: operating point has no identity"))?;
        ensure!(
            ids.insert(point_id.to_owned()),
            "{context}: duplicate operating-point identity {point_id}"
        );
    }

    let diluted = points
        .iter()
        .find(|point| {
            point.get("operating_point_id").and_then(Value::as_str)
                == Some("CH4-PPH-DILUTED-HIGH-CONVERSION")
        })
        .with_context(|| format!("{context}: diluted-feed operating point is missing"))?;
    let pure = points
        .iter()
        .find(|point| {
            point.get("operating_point_id").and_then(Value::as_str)
                == Some("CH4-PPH-PURE-METHANE-LOWEST-ENERGY")
        })
        .with_context(|| format!("{context}: pure-methane energy operating point is missing"))?;

    ensure!(
        close_f64(required_f64(diluted, "methane_mole_fraction", &context)?, 0.10)
            && close_f64(required_f64(diluted, "helium_mole_fraction", &context)?, 0.90),
        "{context}: the 95% conversion point must retain its 10/90 CH4/He feed"
    );
    ensure!(
        close_f64(required_f64(diluted, "pulsed_heating_voltage_V", &context)?, 46.0)
            && close_f64(required_f64(diluted, "discharge_voltage_kV", &context)?, 5.0),
        "{context}: diluted-feed conversion point must retain 46 V / 5 kV conditions"
    );
    ensure!(
        close_f64(required_f64(diluted, "methane_conversion_fraction", &context)?, 0.95)
            && close_f64(required_f64(diluted, "hydrogen_selectivity_fraction", &context)?, 1.0),
        "{context}: diluted-feed point must retain 95% conversion and 100% H2 selectivity"
    );
    ensure!(
        diluted
            .get("specific_energy_kWh_per_kg_H2")
            .is_some_and(Value::is_null),
        "{context}: pure-methane 27 kWh/kg energy value must not be borrowed by the diluted-feed point"
    );

    ensure!(
        close_f64(required_f64(pure, "methane_mole_fraction", &context)?, 1.0)
            && close_f64(required_f64(pure, "helium_mole_fraction", &context)?, 0.0),
        "{context}: 27 kWh/kg energy point must retain pure-methane feed"
    );
    ensure!(
        close_f64(required_f64(pure, "pulsed_heating_voltage_V", &context)?, 46.0)
            && close_f64(required_f64(pure, "discharge_voltage_kV", &context)?, 3.0),
        "{context}: pure-methane energy point must retain 46 V / 3 kV conditions"
    );
    ensure!(
        close_f64(required_f64(pure, "specific_energy_kWh_per_kg_H2", &context)?, 27.0),
        "{context}: pure-methane point must retain 27 kWh/kg H2"
    );
    ensure!(
        pure.get("methane_conversion_fraction").is_some_and(Value::is_null)
            && pure.get("hydrogen_selectivity_fraction").is_some_and(Value::is_null),
        "{context}: diluted-feed conversion/selectivity must not be borrowed by the pure-methane energy point"
    );

    let regeneration = case
        .get("regeneration_evidence")
        .with_context(|| format!("{context}: regeneration evidence is missing"))?;
    ensure!(
        regeneration.get("method").and_then(Value::as_str) == Some("CO2 plasma"),
        "{context}: catalyst regeneration must explicitly identify CO2 plasma"
    );
    ensure!(
        regeneration
            .get("reported_regeneration_cycles")
            .and_then(Value::as_u64)
            == Some(4),
        "{context}: source-reported four regeneration cycles must be preserved exactly"
    );
    ensure!(
        regeneration
            .get("energy_per_regeneration_cycle_kWh")
            .is_some_and(Value::is_null),
        "{context}: unreported regeneration energy must remain unknown, not zero"
    );
    ensure!(
        regeneration
            .get("carbon_capture_and_offtake_qualified")
            .and_then(Value::as_bool)
            == Some(false),
        "{context}: carbon product qualification and offtake must remain unvalidated"
    );
    Ok(())
}


fn validate_ch4_thermal_appj_case(case: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: hybrid APPJ methane source-claim audit");
    ensure!(
        case.get("evidence_class").and_then(Value::as_str) == Some("PUBLISHED_LAB_EXPERIMENT"),
        "{context}: APPJ source must remain a published laboratory experiment"
    );
    ensure!(
        case.get("luminous_reproduction").and_then(Value::as_bool) == Some(false),
        "{context}: APPJ source cannot be promoted to Luminous reproduction"
    );
    ensure!(
        case.get("source").and_then(|source| source.get("doi")).and_then(Value::as_str)
            == Some("10.1016/j.ijhydene.2026.158179"),
        "{context}: APPJ source DOI identity does not match the registered article"
    );

    let configuration = case.get("configuration")
        .with_context(|| format!("{context}: configuration is missing"))?;
    ensure!(
        configuration.get("reactor").and_then(Value::as_str)
            == Some("thermally assisted atmospheric-pressure plasma jet (APPJ)"),
        "{context}: reactor identity must remain a thermally assisted APPJ"
    );
    let configs = configuration.get("configurations_compared")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: compared configurations are missing"))?;
    for required in [
        "single-electrode APPJ with plasma outside the heated zone",
        "single-electrode APPJ with plasma directly inside the heated zone",
        "double-electrode APPJ configuration",
    ] {
        ensure!(
            configs.iter().any(|value| value.as_str() == Some(required)),
            "{context}: missing source configuration {required}"
        );
    }

    let claims = case.get("reported_claims")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: source reported_claims are missing"))?;
    let conversion = claims.iter().find(|claim| {
        claim.get("claim_id").and_then(Value::as_str)
            == Some("PLASMA-IN-THERMAL-ZONE-CONVERSION-RELATIVE-GAIN")
    }).with_context(|| format!("{context}: relative conversion claim is missing"))?;
    ensure!(
        close_f64(required_f64(conversion, "value_lower_bound", &context)?, 5.0)
            && conversion.get("unit").and_then(Value::as_str) == Some("fold"),
        "{context}: source conversion result must remain a >5-fold relative gain"
    );
    ensure!(
        conversion.get("absolute_conversion_fraction").is_some_and(Value::is_null),
        "{context}: relative conversion gain cannot be promoted to absolute methane conversion"
    );
    ensure!(
        conversion.get("comparator_pair_definition_status").and_then(Value::as_str)
            == Some("RECONSTRUCT_FROM_FULL_ARTICLE_AND_SUPPLEMENT"),
        "{context}: exact comparator pair must remain a source-reconstruction requirement"
    );

    let deposition = claims.iter().find(|claim| {
        claim.get("claim_id").and_then(Value::as_str) == Some("LOWEST-CARBON-DEPOSITION-APPJ")
    }).with_context(|| format!("{context}: carbon-deposition claim is missing"))?;
    ensure!(
        close_f64(required_f64(deposition, "value_approx", &context)?, 0.36),
        "{context}: approximately 36% source carbon-deposition result must be preserved"
    );
    ensure!(
        deposition.get("denominator_definition_status").and_then(Value::as_str)
            == Some("RECONSTRUCT_FROM_FULL_ARTICLE_AND_SUPPLEMENT"),
        "{context}: carbon-deposition denominator must remain unresolved pending source reconstruction"
    );
    ensure!(
        deposition.get("captured_product_mass").is_some_and(Value::is_null)
            && deposition.get("feed_carbon_recovery_fraction").is_some_and(Value::is_null),
        "{context}: carbon deposition cannot be promoted to captured or recovered carbon yield"
    );

    let morphology = claims.iter().find(|claim| {
        claim.get("claim_id").and_then(Value::as_str) == Some("TURBOSTRATIC-CARBON-STRUCTURE")
    }).with_context(|| format!("{context}: carbon morphology claim is missing"))?;
    ensure!(
        morphology.get("value").and_then(Value::as_str)
            == Some("turbostratic nanostructural features")
            && morphology.get("product_grade_qualification_status").and_then(Value::as_str)
                == Some("NOT_VALIDATED"),
        "{context}: carbon structure must not be promoted to qualified product grade"
    );

    let slate = case.get("product_slate")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: multi-product slate is missing"))?;
    let h2 = slate.iter().find(|product| {
        product.get("species_or_material").and_then(Value::as_str) == Some("H2")
    }).with_context(|| format!("{context}: H2 product stream is missing"))?;
    ensure!(
        h2.get("quantitative_rate").is_some_and(Value::is_null)
            && h2.get("purity_fraction").is_some_and(Value::is_null)
            && h2.get("pressure").is_some_and(Value::is_null)
            && h2.get("accepted_product_status").and_then(Value::as_str) == Some("NOT_QUANTIFIED"),
        "{context}: H2 rate, purity and pressure must remain unquantified"
    );
    let c2c3 = slate.iter().find(|product| {
        product.get("species_or_material").and_then(Value::as_str) == Some("C2-C3 hydrocarbons")
    }).with_context(|| format!("{context}: C2-C3 product stream is missing"))?;
    ensure!(
        c2c3.get("species_resolved_yields").is_some_and(Value::is_null)
            && c2c3.get("accepted_product_status").and_then(Value::as_str) == Some("NOT_QUANTIFIED"),
        "{context}: C2-C3 species-resolved yields must remain unquantified"
    );
    let carbon = slate.iter().find(|product| {
        product.get("species_or_material").and_then(Value::as_str) == Some("solid carbon")
    }).with_context(|| format!("{context}: solid carbon stream is missing"))?;
    ensure!(
        carbon.get("captured_mass").is_some_and(Value::is_null)
            && carbon.get("grade").is_some_and(Value::is_null)
            && carbon.get("accepted_product_status").and_then(Value::as_str) == Some("NOT_QUALIFIED"),
        "{context}: solid carbon deposition/morphology is not captured qualified product"
    );

    ensure!(
        case.get("comparison_status").and_then(Value::as_str)
            == Some("SOURCE_SPECIFIC_RELATIVE_CONFIGURATION_RESULT_NOT_FULL_CHAIN_COMPARABLE"),
        "{context}: APPJ result must remain non-comparable until full boundaries are reconstructed"
    );
    Ok(())
}

fn validate_nh3_synthesis_routes_case(case: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: modeled plasma ammonia synthesis route comparison");
    let configuration = case
        .get("configuration")
        .with_context(|| format!("{context}: configuration is missing"))?;
    ensure!(
        required_f64(configuration, "nominal_plant_capacity_tonne_nh3_per_day", &context)? == 1.0,
        "{context}: source comparison must preserve the 1 tonne/day modeled plant basis"
    );
    ensure!(
        close_f64(
            required_f64(configuration, "accepted_product_purity_weight_fraction", &context)?,
            0.98
        ),
        "{context}: source comparison must preserve the 98 wt% NH3 product basis"
    );
    let points = case
        .get("operating_points")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: operating_points are missing"))?;
    ensure!(
        points.len() == 3,
        "{context}: expected three distinct modeled synthesis routes"
    );
    let mut ids = BTreeSet::new();
    for point in points {
        let id = point
            .get("operating_point_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{context}: operating point has no identity"))?;
        ensure!(
            ids.insert(id.to_owned()),
            "{context}: duplicate synthesis route identity {id}"
        );
    }
    for (id, route, expected_energy) in [
        ("NH3-SYN-RGA-NORR-2026", "RGA_NORR", 84.0),
        ("NH3-SYN-RGA-LNT-2026", "RGA_LNT", 154.0),
        ("NH3-SYN-DIRECT-DBD-2026", "DIRECT_DBD", 435.0),
    ] {
        let point = points
            .iter()
            .find(|point| point.get("operating_point_id").and_then(Value::as_str) == Some(id))
            .with_context(|| format!("{context}: missing route {id}"))?;
        ensure!(
            point.get("route").and_then(Value::as_str) == Some(route),
            "{context}: route {id} has a mismatched route label"
        );
        ensure!(
            close_f64(
                required_f64(point, "specific_energy_kWh_per_kg_nh3", &context)?,
                expected_energy
            ),
            "{context}: route {id} specific energy does not match its source point"
        );
    }
    let norr = points
        .iter()
        .find(|point| point.get("operating_point_id").and_then(Value::as_str) == Some("NH3-SYN-RGA-NORR-2026"))
        .expect("validated above");
    ensure!(
        close_f64(
            required_f64(norr, "plasma_share_of_total_energy_fraction", &context)?,
            0.68
        ),
        "{context}: RGA-NORR plasma share must remain 68% of total modeled energy"
    );
    ensure!(
        close_f64(
            case_observation_value(case, "green_haber_bosch_small_scale_reference_specific_energy")
                .unwrap_or(f64::NAN),
            11.5
        ),
        "{context}: green Haber-Bosch comparator must remain 11.5 kWh/kg NH3"
    );
    ensure!(
        case.get("luminous_reproduction").and_then(Value::as_bool) == Some(false),
        "{context}: modeled process results cannot be promoted to Luminous reproduction"
    );
    Ok(())
}

fn validate_co2_dbd_ml_case(case: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: DBD CO2 grouped ML source benchmark");
    let dataset = case
        .get("dataset")
        .with_context(|| format!("{context}: dataset metadata is missing"))?;
    ensure!(
        required_f64(dataset, "data_point_count", &context)? == 358.0,
        "{context}: source dataset must retain its 358 data points"
    );
    ensure!(
        required_f64(dataset, "feature_count", &context)? == 6.0,
        "{context}: source model must retain its six reported features"
    );
    ensure!(
        dataset.get("split_protocol").and_then(Value::as_str)
            == Some("publication-wise grouped five-fold cross-validation"),
        "{context}: source validation must retain publication-wise grouped folds"
    );
    let features = dataset
        .get("features")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: feature list is missing"))?;
    ensure!(
        features.len() == 6,
        "{context}: feature list must contain exactly six declared features"
    );
    let best_fold = case_observation_value(case, "best_fold_grouped_cross_validation_r2")
        .unwrap_or(f64::NAN);
    let prospective = case_observation_value(case, "prospective_unseen_data_r2")
        .unwrap_or(f64::NAN);
    ensure!(
        close_f64(best_fold, 0.791),
        "{context}: best-fold grouped-CV R2 must remain source-pinned at 0.791"
    );
    ensure!(
        close_f64(prospective, 0.92),
        "{context}: prospective in-domain R2 must remain source-pinned at 0.92"
    );
    ensure!(
        case.get("comparison_status").and_then(Value::as_str)
            == Some("PUBLISHED_GROUPED_DATA_MODEL_WITH_DOMAIN_LIMITED_PROSPECTIVE_VALIDATION"),
        "{context}: prospective validation must retain its domain-limited claim ceiling"
    );
    ensure!(
        case.get("luminous_reproduction").and_then(Value::as_bool) == Some(false),
        "{context}: published DBD model cannot be promoted to Luminous reproduction"
    );
    Ok(())
}

fn validate_co2_rwgs_operating_points(case: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: gliding-arc RWGS operating-point pairing");
    let points = case
        .get("operating_points")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: operating_points are missing"))?;
    ensure!(
        points.len() == 2,
        "{context}: expected separate H2-rich and stoichiometric operating points"
    );
    let mut ids = BTreeSet::new();
    for point in points {
        let id = point
            .get("operating_point_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{context}: operating point has no identity"))?;
        ensure!(ids.insert(id.to_owned()), "{context}: duplicate operating point {id}");
    }
    let h2_rich = points
        .iter()
        .find(|point| point.get("operating_point_id").and_then(Value::as_str) == Some("RWGS-H2-RICH-2026"))
        .with_context(|| format!("{context}: H2-rich point is missing"))?;
    let stoich = points
        .iter()
        .find(|point| point.get("operating_point_id").and_then(Value::as_str) == Some("RWGS-STOICHIOMETRIC-2026"))
        .with_context(|| format!("{context}: stoichiometric point is missing"))?;
    ensure!(
        required_f64(h2_rich, "h2_to_co2_feed_ratio", &context)? == 4.0,
        "{context}: high-conversion point must preserve H2:CO2=4"
    );
    ensure!(
        close_f64(required_f64(h2_rich, "co2_conversion_fraction", &context)?, 0.93),
        "{context}: H2-rich point must preserve approximately 93% conversion"
    );
    ensure!(
        close_f64(required_f64(h2_rich, "co_selectivity_fraction", &context)?, 1.0),
        "{context}: H2-rich point must preserve 100% CO selectivity"
    );
    ensure!(
        h2_rich.get("specific_energy_cost_kj_per_mol_co").is_none(),
        "{context}: do not borrow stoichiometric specific energy into the H2-rich point"
    );
    ensure!(
        required_f64(stoich, "h2_to_co2_feed_ratio", &context)? == 1.0,
        "{context}: stoichiometric point must preserve H2:CO2=1"
    );
    ensure!(
        close_f64(required_f64(stoich, "co2_conversion_fraction", &context)?, 0.68),
        "{context}: stoichiometric point must preserve approximately 68% conversion"
    );
    ensure!(
        close_f64(required_f64(stoich, "co_selectivity_fraction", &context)?, 1.0),
        "{context}: stoichiometric point must preserve 100% CO selectivity"
    );
    ensure!(
        close_f64(required_f64(stoich, "specific_energy_cost_kj_per_mol_co", &context)?, 277.0),
        "{context}: stoichiometric energy cost must preserve the 277 kJ/mol abstract value"
    );
    let discrepancies = case
        .get("source_text_discrepancies")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: source text discrepancy record is required"))?;
    let discrepancy = discrepancies.iter().find(|item| {
        item.get("field").and_then(Value::as_str)
            == Some("minimum_specific_energy_cost_kj_per_mol_co")
    }).with_context(|| format!("{context}: 264/277 kJ/mol discrepancy is not recorded"))?;
    let values = discrepancy
        .get("reported_values")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: discrepancy values are missing"))?;
    ensure!(
        values.iter().any(|value| value.as_f64() == Some(264.0))
            && values.iter().any(|value| value.as_f64() == Some(277.0)),
        "{context}: both 264 and 277 kJ/mol source values must remain visible"
    );
    ensure!(
        discrepancy.get("resolution").and_then(Value::as_str)
            .is_some_and(|value| value.contains("UNRESOLVED")),
        "{context}: 264/277 discrepancy must remain unresolved until source artifact review"
    );
    Ok(())
}

fn validate_co2_crg_operating_points(case: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: cyclic reforming-gasification phase identity");
    let configuration = case
        .get("configuration")
        .with_context(|| format!("{context}: configuration is missing"))?;
    ensure!(
        configuration.get("energy_basis").and_then(Value::as_str) == Some("plasma-deposited power, not full wall-plug/grid input"),
        "{context}: deposited power must not be relabelled as grid-side power"
    );
    let modes = configuration
        .get("operation_modes")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: operation_modes are missing"))?;
    ensure!(
        modes.iter().any(|value| value.as_str() == Some("discontinuous"))
            && modes.iter().any(|value| value.as_str() == Some("continuous")),
        "{context}: continuous and discontinuous operation must remain separate"
    );
    let points = case
        .get("operating_points")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: operating_points are missing"))?;
    ensure!(points.len() == 2, "{context}: expected separate DRM and methane-pyrolysis modes");
    let mut ids = BTreeSet::new();
    for point in points {
        let id = point
            .get("operating_point_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{context}: operating point has no identity"))?;
        ensure!(ids.insert(id.to_owned()), "{context}: duplicate operating point {id}");
    }
    let drm = points
        .iter()
        .find(|point| point.get("operating_point_id").and_then(Value::as_str) == Some("CRG-DISCONTINUOUS-DRM-2026"))
        .with_context(|| format!("{context}: discontinuous DRM point is missing"))?;
    let continuous = points
        .iter()
        .find(|point| point.get("operating_point_id").and_then(Value::as_str) == Some("CRG-CONTINUOUS-CH4-PYROLYSIS-2026"))
        .with_context(|| format!("{context}: continuous methane-pyrolysis point is missing"))?;
    ensure!(
        close_f64(required_f64(drm, "h2_to_co_ratio", &context)?, 1.7),
        "{context}: discontinuous DRM H2/CO ratio must remain 1.7"
    );
    ensure!(
        close_f64(required_f64(continuous, "methane_conversion_fraction", &context)?, 0.60),
        "{context}: continuous methane conversion must remain about 60%"
    );
    ensure!(
        close_f64(required_f64(continuous, "maximum_oxygen_concentration_after_three_passes_vol_percent", &context)?, 0.5),
        "{context}: maximum oxygen concentration after three passes must remain 0.5 vol%"
    );
    Ok(())
}

fn validate_nh3_synthesis_energy_reciprocal_check(
    check: &Value,
    root: &Value,
    display_name: &str,
) -> Result<()> {
    let context = format!("{display_name}: ammonia synthesis reciprocal-energy check");
    ensure!(
        check.get("source_case_id").and_then(Value::as_str)
            == Some("NH3-PLASMA-SYNTHESIS-ROUTES-TEA-2026"),
        "{context}: source_case_id must identify the three-route synthesis TEA"
    );
    let cases = root
        .get("cases")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: cases array is missing"))?;
    let source = cases
        .iter()
        .find(|case| case.get("case_id").and_then(Value::as_str) == Some("NH3-PLASMA-SYNTHESIS-ROUTES-TEA-2026"))
        .with_context(|| format!("{context}: NH3 synthesis source case is missing"))?;
    validate_nh3_synthesis_routes_case(source, display_name)?;

    let inputs = check
        .get("inputs")
        .with_context(|| format!("{context}: inputs are missing"))?;
    let derivation = check
        .get("derivation")
        .with_context(|| format!("{context}: derivation is missing"))?;
    let route_values = [
        ("rga_norr_specific_energy_kWh_per_kg_NH3", "rga_norr_energy_yield_g_NH3_per_kWh", 84.0),
        ("rga_lnt_specific_energy_kWh_per_kg_NH3", "rga_lnt_energy_yield_g_NH3_per_kWh", 154.0),
        ("direct_dbd_specific_energy_kWh_per_kg_NH3", "direct_dbd_energy_yield_g_NH3_per_kWh", 435.0),
        ("small_scale_green_hb_reference_kWh_per_kg_NH3", "green_hb_energy_yield_g_NH3_per_kWh", 11.5),
    ];
    for (input_key, derived_key, expected_specific_energy) in route_values {
        let specific_energy = required_f64(inputs, input_key, &context)?;
        ensure!(
            close_f64(specific_energy, expected_specific_energy),
            "{context}: {input_key} does not match its source-pinned process point"
        );
        let derived = required_f64(derivation, derived_key, &context)?;
        ensure!(
            close_f64(derived, 1000.0 / specific_energy),
            "{context}: {derived_key} must equal 1000 / source kWh per kg NH3"
        );
    }
    Ok(())
}

fn validate_nh3_operating_points(case: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: NH3 microwave operating-point pairing");
    let points = case
        .get("operating_points")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: operating_points array is missing"))?;
    ensure!(
        points.len() == 3,
        "{context}: expected exactly three separately paired source operating points"
    );

    let mut point_ids = BTreeSet::new();
    for point in points {
        let point_id = point
            .get("operating_point_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| format!("{context}: operating point has no identity"))?;
        ensure!(
            point_ids.insert(point_id.to_owned()),
            "{context}: duplicate operating-point identity {point_id}"
        );
    }

    let high_conversion = points
        .iter()
        .find(|point| {
            point.get("operating_point_id").and_then(Value::as_str)
                == Some("NH3-MW-ATM-HIGH-CONVERSION")
        })
        .with_context(|| format!("{context}: missing high-conversion operating point"))?;
    let high_throughput = points
        .iter()
        .find(|point| {
            point.get("operating_point_id").and_then(Value::as_str)
                == Some("NH3-MW-ATM-HIGH-THROUGHPUT")
        })
        .with_context(|| format!("{context}: missing high-throughput operating point"))?;
    let high_yield = points
        .iter()
        .find(|point| {
            point.get("operating_point_id").and_then(Value::as_str)
                == Some("NH3-MW-ATM-HIGH-ENERGY-YIELD")
        })
        .with_context(|| format!("{context}: missing high-energy-yield operating point"))?;

    let conversion_lower_bound = required_f64(
        high_conversion,
        "ammonia_conversion_lower_bound_fraction",
        &context,
    )?;
    ensure!(
        close_f64(conversion_lower_bound, 0.99),
        "{context}: high-conversion point must preserve its >99% lower bound"
    );
    ensure!(
        close_f64(
            required_f64(
                high_conversion,
                "hydrogen_energy_yield_g_H2_per_kWh",
                &context
            )?,
            45.0
        ),
        "{context}: >99% conversion point must retain its 45 g/kWh yield"
    );
    ensure!(
        required_f64(high_conversion, "hydrogen_energy_cost_kJ_per_mol", &context)? == 242.0,
        "{context}: >99% conversion point must retain its 242 kJ/mol energy cost"
    );
    ensure!(
        high_conversion
            .get("hydrogen_production_rate_g_per_hour")
            .is_some_and(Value::is_null),
        "{context}: high-conversion point must not borrow the 150 g/h throughput value"
    );

    ensure!(
        close_f64(
            required_f64(high_throughput, "ammonia_conversion_fraction", &context)?,
            0.924
        ),
        "{context}: 150 g/h point must retain 92.4% conversion"
    );
    ensure!(
        close_f64(
            required_f64(
                high_throughput,
                "hydrogen_energy_yield_g_H2_per_kWh",
                &context
            )?,
            50.0
        ),
        "{context}: 150 g/h point must retain its 50 g/kWh energy yield"
    );
    ensure!(
        required_f64(high_throughput, "hydrogen_energy_cost_kJ_per_mol", &context)? == 217.0,
        "{context}: 150 g/h point must retain its 217 kJ/mol energy cost"
    );
    ensure!(
        high_throughput
            .get("hydrogen_production_rate_g_per_hour")
            .and_then(Value::as_f64)
            .is_some_and(|value| close_f64(value, 150.0)),
        "{context}: 150 g/h throughput must remain paired with the 92.4% conversion point"
    );

    ensure!(
        close_f64(
            required_f64(high_yield, "ammonia_conversion_fraction", &context)?,
            0.76
        ),
        "{context}: highest-yield point must retain 76% conversion"
    );
    ensure!(
        close_f64(
            required_f64(high_yield, "hydrogen_energy_yield_g_H2_per_kWh", &context)?,
            51.0
        ),
        "{context}: highest-yield point must retain 51 g/kWh energy yield"
    );
    ensure!(
        required_f64(high_yield, "hydrogen_energy_cost_kJ_per_mol", &context)? == 210.0,
        "{context}: highest-yield point must retain 210 kJ/mol energy cost"
    );
    ensure!(
        high_yield
            .get("hydrogen_production_rate_g_per_hour")
            .is_some_and(Value::is_null),
        "{context}: highest-yield point must not borrow the 150 g/h throughput value"
    );
    Ok(())
}

fn validate_nh3_energy_yield_check(check: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: NH3 microwave energy-yield derivation");
    ensure!(
        check.get("source_case_id").and_then(Value::as_str)
            == Some("NH3-CRACK-MW-ATM-MECHANISM-2026"),
        "{context}: source_case_id must point to NH3-CRACK-MW-ATM-MECHANISM-2026"
    );
    let inputs = check
        .get("inputs")
        .with_context(|| format!("{context}: inputs are missing"))?;
    let derivation = check
        .get("derivation")
        .with_context(|| format!("{context}: derivation is missing"))?;

    let near_complete_yield = required_f64(
        inputs,
        "near_complete_conversion_energy_yield_g_H2_per_kWh",
        &context,
    )?;
    let best_yield = required_f64(inputs, "best_energy_yield_g_H2_per_kWh", &context)?;
    ensure!(
        near_complete_yield > 0.0 && best_yield > 0.0,
        "{context}: energy yields must be positive"
    );

    let expected_near_complete = 1000.0 / near_complete_yield;
    let expected_best = 1000.0 / best_yield;
    let expected_reduction = 1.0 - near_complete_yield / best_yield;

    let actual_near_complete = required_f64(
        derivation,
        "near_complete_conversion_specific_energy_kWh_per_kg_H2",
        &context,
    )?;
    let actual_best = required_f64(
        derivation,
        "best_energy_yield_specific_energy_kWh_per_kg_H2",
        &context,
    )?;
    let actual_intermediate = required_f64(
        derivation,
        "intermediate_92_4_percent_specific_energy_kWh_per_kg_H2",
        &context,
    )?;
    let actual_reduction_92_4 = required_f64(
        derivation,
        "relative_specific_energy_reduction_at_92_4_percent_conversion",
        &context,
    )?;
    let actual_reduction = required_f64(
        derivation,
        "relative_specific_energy_reduction_at_76_percent_conversion",
        &context,
    )?;

    let intermediate_yield = required_f64(
        inputs,
        "intermediate_92_4_percent_energy_yield_g_H2_per_kWh",
        &context,
    )?;
    ensure!(
        intermediate_yield > 0.0,
        "{context}: intermediate energy yield must be positive"
    );
    let expected_intermediate = 1000.0 / intermediate_yield;
    let expected_reduction_92_4 = 1.0 - near_complete_yield / intermediate_yield;

    ensure!(
        close_f64(actual_near_complete, expected_near_complete),
        "{context}: near-complete specific-energy derivation does not equal 1000 / reported yield"
    );
    ensure!(
        close_f64(actual_intermediate, expected_intermediate),
        "{context}: 92.4-percent specific-energy derivation does not equal 1000 / reported yield"
    );
    ensure!(
        close_f64(actual_best, expected_best),
        "{context}: best-yield specific-energy derivation does not equal 1000 / reported yield"
    );
    ensure!(
        close_f64(actual_reduction_92_4, expected_reduction_92_4),
        "{context}: 92.4-percent relative specific-energy reduction is inconsistent with the source points"
    );
    ensure!(
        close_f64(actual_reduction, expected_reduction),
        "{context}: 76-percent relative specific-energy reduction is inconsistent with the source points"
    );
    Ok(())
}

fn case_observation_value(case: &Value, metric: &str) -> Option<f64> {
    case.get("observations")?
        .as_array()?
        .iter()
        .find(|observation| observation.get("metric").and_then(Value::as_str) == Some(metric))?
        .get("value")?
        .as_f64()
}

fn validate_nh3_synthesis_energy_gap_check(
    check: &Value,
    root: &Value,
    display_name: &str,
) -> Result<()> {
    let context = format!("{display_name}: NH3 plasma-to-SOEC-HB source metric audit");
    ensure!(
        check.get("source_case_id").and_then(Value::as_str)
            == Some("NH3-GAS-LIQUID-PLASMA-ELECTROCATALYTIC-TANDEM-2026"),
        "{context}: source_case_id must identify the gas-liquid plasma tandem source"
    );
    ensure!(
        check.get("reference_case_id").and_then(Value::as_str)
            == Some("NH3-SOEC-HB-OFFSHORE-2026"),
        "{context}: reference_case_id must identify the integrated SOEC-HB source"
    );
    ensure!(
        check.get("comparison_status").and_then(Value::as_str)
            == Some("NOT_COMPARABLE_BOUNDARIES_UNRESOLVED"),
        "{context}: source values must remain NOT_COMPARABLE until boundaries are reconciled"
    );

    let cases = root
        .get("cases")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: source cases array is missing"))?;
    let plasma_case = cases.iter().find(|case| {
        case.get("case_id").and_then(Value::as_str)
            == Some("NH3-GAS-LIQUID-PLASMA-ELECTROCATALYTIC-TANDEM-2026")
    }).with_context(|| format!("{context}: gas-liquid plasma source case is missing"))?;
    let soec_case = cases.iter().find(|case| {
        case.get("case_id").and_then(Value::as_str)
            == Some("NH3-SOEC-HB-OFFSHORE-2026")
    }).with_context(|| format!("{context}: SOEC-HB reference case is missing"))?;
    let jacs_case = cases.iter().find(|case| {
        case.get("case_id").and_then(Value::as_str)
            == Some("NH3-PLASMA-ELECTROCATALYSIS-JACS-2026")
    }).with_context(|| format!("{context}: JACS area-rate source case is missing"))?;

    let inputs = check.get("inputs")
        .with_context(|| format!("{context}: inputs are missing"))?;
    let derivation = check.get("derivation")
        .with_context(|| format!("{context}: derivation is missing"))?;
    let plasma_energy = required_f64(inputs, "plasma_tandem_specific_energy_kWh_per_kg_NH3", &context)?;
    let soec_energy = required_f64(inputs, "soec_hb_model_specific_energy_kWh_per_kg_NH3", &context)?;
    let area_rate = required_f64(inputs, "jacs_area_rate_nmol_NH3_per_s_per_cm2", &context)?;
    let molar_mass = required_f64(inputs, "ammonia_molar_mass_g_per_mol", &context)?;
    ensure!(
        plasma_energy > 0.0 && soec_energy > 0.0 && area_rate > 0.0 && molar_mass > 0.0,
        "{context}: source inputs must be positive"
    );
    ensure!(
        close_f64(
            case_observation_value(plasma_case, "specific_energy_consumption").unwrap_or(f64::NAN),
            plasma_energy
        ),
        "{context}: plasma energy input does not match the source case"
    );
    ensure!(
        close_f64(
            case_observation_value(soec_case, "specific_electrical_energy").unwrap_or(f64::NAN),
            soec_energy
        ),
        "{context}: SOEC-HB energy input does not match the source case"
    );
    ensure!(
        close_f64(
            case_observation_value(jacs_case, "ammonia_area_production_rate").unwrap_or(f64::NAN),
            area_rate
        ),
        "{context}: JACS area-rate input does not match the source case"
    );

    let expected_plasma_yield = 1000.0 / plasma_energy;
    let expected_soec_yield = 1000.0 / soec_energy;
    let expected_ratio = plasma_energy / soec_energy;
    let expected_area_productivity =
        area_rate * 1.0e-9 * molar_mass * 86_400.0 * 10_000.0 / 1000.0;
    let expected_area_for_tonne = 1000.0 / expected_area_productivity;

    let actual_plasma_yield = required_f64(derivation, "plasma_tandem_yield_g_NH3_per_kWh", &context)?;
    let actual_soec_yield = required_f64(derivation, "soec_hb_yield_g_NH3_per_kWh", &context)?;
    let actual_ratio = required_f64(derivation, "source_metric_energy_intensity_ratio", &context)?;
    let actual_area_productivity = required_f64(
        derivation,
        "jacs_area_productivity_kg_NH3_per_m2_per_day",
        &context,
    )?;
    let actual_area_for_tonne = required_f64(
        derivation,
        "active_area_m2_for_1_tonne_NH3_per_day_at_jacs_rate",
        &context,
    )?;

    ensure!(
        close_f64(actual_plasma_yield, expected_plasma_yield),
        "{context}: plasma specific-energy reciprocal is inconsistent"
    );
    ensure!(
        close_f64(actual_soec_yield, expected_soec_yield),
        "{context}: SOEC-HB specific-energy reciprocal is inconsistent"
    );
    ensure!(
        close_f64(actual_ratio, expected_ratio),
        "{context}: source metric energy-intensity ratio is inconsistent"
    );
    ensure!(
        close_f64(actual_area_productivity, expected_area_productivity),
        "{context}: JACS geometric area productivity conversion is inconsistent"
    );
    ensure!(
        close_f64(actual_area_for_tonne, expected_area_for_tonne),
        "{context}: JACS geometric active-area extrapolation is inconsistent"
    );
    Ok(())
}

fn co2_configuration<'a>(case: &'a Value, configuration_id: &str) -> Option<&'a Value> {
    case.get("configurations")?
        .as_array()?
        .iter()
        .find(|configuration| {
            configuration.get("id").and_then(Value::as_str) == Some(configuration_id)
        })
}

fn co2_metric<'a>(configuration: &'a Value, metric_name: &str) -> Option<&'a Value> {
    configuration
        .get("observations")?
        .as_array()?
        .iter()
        .find(|observation| observation.get("metric").and_then(Value::as_str) == Some(metric_name))
}

fn validate_co2_counterflow_operating_points(case: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: CO2 counterflow operating-point audit");
    let configurations = case
        .get("configurations")
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: configurations array is missing"))?;
    ensure!(
        configurations.len() == 3,
        "{context}: expected exactly three separately identified source configurations"
    );

    let mut configuration_ids = BTreeSet::new();
    for configuration in configurations {
        let id = configuration
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .with_context(|| format!("{context}: configuration has no non-empty id"))?;
        ensure!(
            configuration_ids.insert(id.to_owned()),
            "{context}: duplicate configuration ID {id}"
        );
    }

    let magd = co2_configuration(case, "counterflow-magd-792w")
        .with_context(|| format!("{context}: missing 792 W MAGD configuration"))?;
    ensure!(
        magd.get("electrical_power_w").and_then(Value::as_f64) == Some(792.0),
        "{context}: MAGD configuration must retain its 792 W operating point"
    );
    ensure!(
        magd.get("gas_flow_slm").and_then(Value::as_f64) == Some(4.0),
        "{context}: MAGD configuration must retain its 4 slm feed-flow point"
    );
    for (metric_name, expected) in [("co2_conversion", 0.236), ("energy_efficiency", 0.251)] {
        let observation = co2_metric(magd, metric_name)
            .with_context(|| format!("{context}: 792 W configuration lacks {metric_name}"))?;
        ensure!(
            observation
                .get("value")
                .and_then(Value::as_f64)
                .is_some_and(|value| close_f64(value, expected)),
            "{context}: 792 W {metric_name} does not match its source-pinned operating point"
        );
    }

    let low_power = co2_configuration(case, "counterflow-low-power-302w")
        .with_context(|| format!("{context}: missing separate 302 W configuration"))?;
    ensure!(
        low_power.get("electrical_power_w").and_then(Value::as_f64) == Some(302.0),
        "{context}: low-power configuration must retain its 302 W operating point"
    );
    let low_power_efficiency = co2_metric(low_power, "energy_efficiency")
        .with_context(|| format!("{context}: 302 W configuration lacks energy_efficiency"))?;
    ensure!(
        low_power_efficiency
            .get("value")
            .and_then(Value::as_f64)
            .is_some_and(|value| close_f64(value, 0.462)),
        "{context}: 302 W energy efficiency must remain the source-reported 46.2% point"
    );
    let low_power_conversion = co2_metric(low_power, "co2_conversion").with_context(|| {
        format!("{context}: 302 W configuration must explicitly preserve unknown CO2 conversion")
    })?;
    ensure!(
        low_power_conversion
            .get("value")
            .is_some_and(Value::is_null),
        "{context}: 302 W conversion must remain unknown; do not borrow the 792 W conversion"
    );

    let sweep =
        co2_configuration(case, "counterflow-ratio-distance-optimization").with_context(|| {
            format!("{context}: missing separate ratio/distance optimization record")
        })?;
    for (metric_name, expected, expected_unit) in [
        ("relative_conversion_increase", 2.073, "fractional_increase"),
        (
            "relative_energy_efficiency_increase",
            2.036,
            "fractional_increase",
        ),
    ] {
        let observation = co2_metric(sweep, metric_name)
            .with_context(|| format!("{context}: optimization record lacks {metric_name}"))?;
        ensure!(
            observation
                .get("value")
                .and_then(Value::as_f64)
                .is_some_and(|value| close_f64(value, expected)),
            "{context}: {metric_name} does not match the source-reported relative increase"
        );
        ensure!(
            observation.get("unit").and_then(Value::as_str) == Some(expected_unit),
            "{context}: {metric_name} must remain a relative increase, not an absolute fraction"
        );
    }
    ensure!(
        sweep
            .get("observations")
            .and_then(Value::as_array)
            .is_some_and(|items| {
                items.iter().any(|item| {
                    item.get("metric").and_then(Value::as_str) == Some("counterflow_gas_ratio")
                        && item.get("value").and_then(Value::as_f64) == Some(2.5)
                }) && items.iter().any(|item| {
                    item.get("metric").and_then(Value::as_str) == Some("counterflow_distance")
                        && item.get("value").and_then(Value::as_f64) == Some(30.0)
                })
            }),
        "{context}: relative-improvement record must retain its 2.5 ratio / 30 mm geometry context"
    );

    Ok(())
}

fn validate_nh3_n2_vibrational_transfer_case(case: &Value, display_name: &str) -> Result<()> {
    let context = format!("{display_name}: NH3 N2 vibrational transfer evidence");
    let state = case
        .get("state_transfer_evidence")
        .with_context(|| format!("{context}: state_transfer_evidence is missing"))?;
    let physical = state
        .get("physical_observations")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .with_context(|| format!("{context}: physical_observations must be a non-empty array"))?;
    let model = state
        .get("model_derived_estimates")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .with_context(|| format!("{context}: model_derived_estimates must be a non-empty array"))?;

    ensure!(
        physical.iter().any(|item| {
            item.get("evidence_class").and_then(Value::as_str)
                == Some("PUBLISHED_EXPERIMENTAL_CONTROL")
                && item.get("observation").and_then(Value::as_str)
                    .is_some_and(|value| value.contains("15N2"))
        }),
        "{context}: source must preserve 15N2 attribution as an experimental control"
    );
    ensure!(
        physical.iter().any(|item| {
            item.get("evidence_class").and_then(Value::as_str)
                == Some("PUBLISHED_IN_SITU_SPECTROSCOPY")
                && item.get("observation").and_then(Value::as_str)
                    .is_some_and(|value| value.to_ascii_lowercase().contains("raman"))
        }),
        "{context}: in-situ Raman intermediate observation must remain separate physical evidence"
    );
    ensure!(
        physical.iter().any(|item| {
            item.get("evidence_class").and_then(Value::as_str)
                == Some("PUBLISHED_EXPERIMENTAL_PRODUCT_OBSERVATION")
        }),
        "{context}: product rate/FE must remain source-classified experimental observations"
    );

    for (metric, expected) in [
        ("effective_N2_vibrational_state_temperature", 4300.0),
        ("N2_population_in_v4_to_v8", 0.118),
    ] {
        let entry = model.iter().find(|item| {
            item.get("metric").and_then(Value::as_str) == Some(metric)
        }).with_context(|| format!("{context}: missing model-derived metric {metric}"))?;
        ensure!(
            entry.get("evidence_class").and_then(Value::as_str) == Some("MODEL_DERIVED"),
            "{context}: {metric} must remain MODEL_DERIVED, not a physical measurement"
        );
        ensure!(
            entry.get("interface_direct_measurement").and_then(Value::as_bool) == Some(false),
            "{context}: {metric} cannot claim a direct catalyst-interface measurement"
        );
        ensure!(
            entry.get("value").and_then(Value::as_f64)
                .is_some_and(|value| close_f64(value, expected)),
            "{context}: {metric} source-pinned model estimate changed"
        );
    }

    let policy = state.get("claim_policy")
        .with_context(|| format!("{context}: claim_policy is missing"))?;
    for key in [
        "model_derived_population_must_not_be_relabelled_physical_measurement",
        "discharge_exit_population_must_not_be_substituted_for_interface_population",
        "raman_intermediate_signal_must_not_be_substituted_for_quantitative_state_flux",
        "isotope_attribution_is_not_equivalent_to_state_resolved_mechanism_proof",
        "faradaic_efficiency_is_not_equivalent_to_full_system_energy_efficiency",
    ] {
        ensure!(
            policy.get(key).and_then(Value::as_bool) == Some(true),
            "{context}: missing evidence-separation policy {key}"
        );
    }
    ensure!(
        state.get("not_established").and_then(Value::as_array).is_some_and(|items| {
            items.iter().any(|item| item.as_str()
                .is_some_and(|value| value.contains("catalyst interface")))
        }),
        "{context}: missing explicit statement that catalyst-interface state is not established"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frozen_source_manifest() -> Value {
        serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        ))
        .expect("frozen plasma source manifest must parse")
    }

    fn frozen_case(case_id: &str) -> Value {
        frozen_source_manifest()["cases"]
            .as_array()
            .expect("cases array")
            .iter()
            .find(|case| case.get("case_id").and_then(Value::as_str) == Some(case_id))
            .expect("registered case exists")
            .clone()
    }

    #[test]
    fn source_manifest_accepts_new_nh3_synthesis_and_co2_cases() {
        let source = frozen_source_manifest();
        let summary = audit_source_value(&source, "frozen-plasma-energy.json").unwrap();
        assert_eq!(summary.case_count, 53);
        assert_eq!(summary.benchmark_target_count, 12);
        assert!(source["cases"].as_array().unwrap().iter().any(|case|
            case.get("case_id").and_then(Value::as_str) == Some("CO2-CYCLIC-REFORMING-GASIFICATION-2026")
        ));
    }

    #[test]
    fn ammonia_synthesis_three_route_source_case_is_point_separated() {
        let source = frozen_source_manifest();
        let case = source["cases"].as_array().unwrap().iter().find(|case|
            case.get("case_id").and_then(Value::as_str) == Some("NH3-PLASMA-SYNTHESIS-ROUTES-TEA-2026")
        ).unwrap();
        validate_nh3_synthesis_routes_case(case, "nh3-routes.json").unwrap();
    }

    #[test]
    fn ammonia_synthesis_rejects_energy_borrowed_between_routes() {
        let mut case = frozen_case("NH3-PLASMA-SYNTHESIS-ROUTES-TEA-2026");
        case["operating_points"][0]["specific_energy_kWh_per_kg_nh3"] = serde_json::json!(11.5);
        let error = validate_nh3_synthesis_routes_case(&case, "nh3-routes.json").unwrap_err();
        assert!(error.to_string().contains("source point"));
    }

    #[test]
    fn co2_dbd_ml_source_retains_grouped_dataset_scope() {
        let case = frozen_case("CO2-DBD-ML-358-DATASET-2026");
        validate_co2_dbd_ml_case(&case, "co2-dbd.json").unwrap();
    }

    #[test]
    fn co2_dbd_ml_rejects_random_row_split() {
        let mut case = frozen_case("CO2-DBD-ML-358-DATASET-2026");
        case["dataset"]["split_protocol"] = serde_json::json!("random row split");
        let error = validate_co2_dbd_ml_case(&case, "co2-dbd.json").unwrap_err();
        assert!(error.to_string().contains("publication-wise grouped"));
    }

    #[test]
    fn co2_dbd_ml_rejects_prospective_score_promoted_to_cross_domain_transfer() {
        let mut case = frozen_case("CO2-DBD-ML-358-DATASET-2026");
        case["comparison_status"] = serde_json::json!("CROSS_REACTOR_VALIDATED");
        let error = validate_co2_dbd_ml_case(&case, "co2-dbd.json").unwrap_err();
        assert!(error.to_string().contains("domain-limited"));
    }

    #[test]
    fn rwgs_preserves_h2_rich_and_stoichiometric_points_separately() {
        let case = frozen_case("CO2-RWGS-GLIDING-ARC-2026");
        validate_co2_rwgs_operating_points(&case, "rwgs.json").unwrap();
    }

    #[test]
    fn rwgs_rejects_energy_cost_borrowed_into_h2_rich_point() {
        let mut case = frozen_case("CO2-RWGS-GLIDING-ARC-2026");
        case["operating_points"][0]["specific_energy_cost_kj_per_mol_co"] = serde_json::json!(277.0);
        let error = validate_co2_rwgs_operating_points(&case, "rwgs.json").unwrap_err();
        assert!(error.to_string().contains("do not borrow stoichiometric"));
    }

    #[test]
    fn rwgs_rejects_silently_resolved_264_277_source_discrepancy() {
        let mut case = frozen_case("CO2-RWGS-GLIDING-ARC-2026");
        case["source_text_discrepancies"][0]["resolution"] = serde_json::json!("RESOLVED");
        let error = validate_co2_rwgs_operating_points(&case, "rwgs.json").unwrap_err();
        assert!(error.to_string().contains("must remain unresolved"));
    }

    #[test]
    fn cyclic_reforming_requires_deposited_power_boundary() {
        let case = frozen_case("CO2-CYCLIC-REFORMING-GASIFICATION-2026");
        validate_co2_crg_operating_points(&case, "crg.json").unwrap();
    }

    #[test]
    fn cyclic_reforming_rejects_plasma_deposited_power_as_grid_energy() {
        let mut case = frozen_case("CO2-CYCLIC-REFORMING-GASIFICATION-2026");
        case["configuration"]["energy_basis"] = serde_json::json!("grid-side power");
        let error = validate_co2_crg_operating_points(&case, "crg.json").unwrap_err();
        assert!(error.to_string().contains("must not be relabelled"));
    }

    #[test]
    fn nh3_synthesis_reciprocal_yield_derivation_is_arithmetic_only() {
        let source = frozen_source_manifest();
        let check = source["derived_reference_checks"].as_array().unwrap().iter().find(|check|
            check.get("check_id").and_then(Value::as_str)
                == Some("NH3-PLASMA-SYNTHESIS-RECIPROCAL-YIELD-2026")
        ).unwrap();
        validate_nh3_synthesis_energy_reciprocal_check(check, &source, "nh3-energy.json").unwrap();
    }

    #[test]
    fn nh3_synthesis_reciprocal_yield_rejects_wrong_value() {
        let mut source = frozen_source_manifest();
        let check = source["derived_reference_checks"].as_array_mut().unwrap().iter_mut().find(|check|
            check.get("check_id").and_then(Value::as_str)
                == Some("NH3-PLASMA-SYNTHESIS-RECIPROCAL-YIELD-2026")
        ).unwrap();
        check["derivation"]["rga_norr_energy_yield_g_NH3_per_kWh"] = serde_json::json!(99.0);
        let error = validate_nh3_synthesis_energy_reciprocal_check(check, &source, "nh3-energy.json").unwrap_err();
        assert!(error.to_string().contains("must equal 1000 / source"));
    }

    fn base_scenario() -> ThermalScenario {
        ThermalScenario {
            scenario_id: "resistance".to_owned(),
            technology: "RESISTANCE_JOULE_HEATER".to_owned(),
            evidence_class: SYNTHETIC_EVIDENCE_CLASS.to_owned(),
            energy_unit: "J".to_owned(),
            energy_boundary_contract_id: "WALL-PLUG-COMPLETE-THERMAL-STORE-V1".to_owned(),
            service: ServiceBoundary {
                service_contract_id: "HEAT-500C-1H".to_owned(),
                service_type: "process_heat".to_owned(),
                outlet_temperature_mk: Some(773_150),
                outlet_pressure_pa: None,
                mass_flow_mg_s: None,
                delivery_duration_s: 3_600,
                delivery_trace_digest: "synthetic-trace-001".to_owned(),
            },
            metering_mode: MeteringMode::NonOverlappingBranchMeters,
            wall_plug_inputs: vec![
                EnergyInput {
                    input_id: "heater".to_owned(),
                    meter_boundary_id: "meter:heater".to_owned(),
                    role: InputRole::PrimaryProcess,
                    energy_j: Some(3_600_000_000),
                },
                EnergyInput {
                    input_id: "aux".to_owned(),
                    meter_boundary_id: "meter:aux".to_owned(),
                    role: InputRole::Auxiliary,
                    energy_j: Some(180_000_000),
                },
            ],
            charge: ChargeAccounting {
                stored_energy_increase_j: 3_240_000_000,
                charge_conversion_loss_j: 360_000_000,
                auxiliary_dissipation_j: 180_000_000,
            },
            hold: HoldAccounting {
                stored_energy_after_hold_j: 3_078_000_000,
                storage_hold_loss_j: 162_000_000,
                hold_duration_s: 86_400,
            },
            discharge: DischargeAccounting {
                stored_energy_withdrawn_j: 3_078_000_000,
                useful_heat_delivered_j: 2_770_000_000,
                discharge_loss_j: 308_000_000,
            },
            expected_disposition: ScenarioDisposition::Pass,
        }
    }

    #[test]
    fn balanced_resistance_case_passes_and_reports_efficiency() {
        let result = evaluate_scenario(&base_scenario());
        assert_eq!(result.disposition, ScenarioDisposition::Pass);
        assert_eq!(result.wall_plug_input_j, Some(3_780_000_000));
        assert_eq!(result.useful_heat_delivered_j, Some(2_770_000_000));
        assert_eq!(result.wall_plug_to_useful_heat_ppm, Some(732_804));
    }

    #[test]
    fn unknown_auxiliary_energy_is_blocked_not_zeroed() {
        let mut scenario = base_scenario();
        scenario.wall_plug_inputs[1].energy_j = None;
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Blocked
        );
    }

    #[test]
    fn duplicate_meter_boundary_is_rejected() {
        let mut scenario = base_scenario();
        scenario.wall_plug_inputs[1].meter_boundary_id =
            scenario.wall_plug_inputs[0].meter_boundary_id.clone();
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Rejected
        );
    }

    #[test]
    fn total_meter_cannot_be_mixed_with_branch_meters() {
        let mut scenario = base_scenario();
        scenario.metering_mode = MeteringMode::TotalMeterOnly;
        scenario.wall_plug_inputs.push(EnergyInput {
            input_id: "total".to_owned(),
            meter_boundary_id: "meter:total".to_owned(),
            role: InputRole::Total,
            energy_j: Some(3_780_000_000),
        });
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Rejected
        );
    }

    #[test]
    fn incorrect_charge_energy_is_rejected() {
        let mut scenario = base_scenario();
        scenario.charge.stored_energy_increase_j -= 1;
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Rejected
        );
    }

    #[test]
    fn incorrect_hold_energy_is_rejected() {
        let mut scenario = base_scenario();
        scenario.hold.storage_hold_loss_j -= 1;
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Rejected
        );
    }

    #[test]
    fn incorrect_discharge_energy_is_rejected() {
        let mut scenario = base_scenario();
        scenario.discharge.discharge_loss_j -= 1;
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Rejected
        );
    }

    #[test]
    fn incomplete_delivery_temperature_is_blocked() {
        let mut scenario = base_scenario();
        scenario.service.outlet_temperature_mk = None;
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Blocked
        );
    }

    #[test]
    fn service_contract_mismatch_is_not_comparable() {
        let baseline = base_scenario();
        let mut candidate = base_scenario();
        candidate.scenario_id = "candidate".to_owned();
        candidate.service.outlet_temperature_mk = Some(873_150);

        let baseline_result = evaluate_scenario(&baseline);
        let candidate_result = evaluate_scenario(&candidate);

        let scenarios = BTreeMap::from([
            (baseline.scenario_id.clone(), &baseline),
            (candidate.scenario_id.clone(), &candidate),
        ]);
        let results = BTreeMap::from([
            (baseline.scenario_id.clone(), baseline_result),
            (candidate.scenario_id.clone(), candidate_result),
        ]);
        let pair = ComparisonPair {
            comparison_id: "temperature-mismatch".to_owned(),
            baseline_scenario_id: baseline.scenario_id.clone(),
            candidate_scenario_id: candidate.scenario_id.clone(),
            expected_disposition: ComparisonDisposition::NotComparable,
        };
        assert_eq!(
            evaluate_comparison(&pair, &scenarios, &results).disposition,
            ComparisonDisposition::NotComparable
        );
    }

    #[test]
    fn energy_boundary_mismatch_is_not_comparable_even_when_service_matches() {
        let baseline = base_scenario();
        let mut candidate = base_scenario();
        candidate.scenario_id = "candidate".to_owned();
        candidate.energy_boundary_contract_id = "WALL-PLUG-HEATER-ONLY-V1".to_owned();

        let baseline_result = evaluate_scenario(&baseline);
        let candidate_result = evaluate_scenario(&candidate);
        assert_eq!(baseline_result.disposition, ScenarioDisposition::Pass);
        assert_eq!(candidate_result.disposition, ScenarioDisposition::Pass);

        let scenarios = BTreeMap::from([
            (baseline.scenario_id.clone(), &baseline),
            (candidate.scenario_id.clone(), &candidate),
        ]);
        let results = BTreeMap::from([
            (baseline.scenario_id.clone(), baseline_result),
            (candidate.scenario_id.clone(), candidate_result),
        ]);
        let pair = ComparisonPair {
            comparison_id: "energy-boundary-mismatch".to_owned(),
            baseline_scenario_id: baseline.scenario_id.clone(),
            candidate_scenario_id: candidate.scenario_id.clone(),
            expected_disposition: ComparisonDisposition::NotComparable,
        };
        let result = evaluate_comparison(&pair, &scenarios, &results);
        assert_eq!(result.disposition, ComparisonDisposition::NotComparable);
        assert!(result.reason.contains("energy-boundary contract"));
    }

    #[test]
    fn absent_auxiliary_meter_is_blocked_even_when_ledger_arithmetic_closes() {
        let mut scenario = base_scenario();
        scenario
            .wall_plug_inputs
            .retain(|input| input.role != InputRole::Auxiliary);
        scenario.charge.auxiliary_dissipation_j = 0;
        scenario.wall_plug_inputs[0].energy_j = Some(3_600_000_000);
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Blocked
        );
    }

    #[test]
    fn total_meter_only_case_is_valid_when_exclusive() {
        let mut scenario = base_scenario();
        scenario.metering_mode = MeteringMode::TotalMeterOnly;
        scenario.wall_plug_inputs = vec![EnergyInput {
            input_id: "total".to_owned(),
            meter_boundary_id: "meter:total".to_owned(),
            role: InputRole::Total,
            energy_j: Some(3_780_000_000),
        }];
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Pass
        );
    }

    #[test]
    fn frozen_plasma_and_hts_source_registers_pass_metadata_audit() {
        let plasma: Value = serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        ))
        .unwrap();
        let hts: Value = serde_json::from_str(include_str!(
            "../../data/research/hts_fusion_magnet_reference_cases_v0_1.json"
        ))
        .unwrap();

        let plasma_summary = audit_source_value(&plasma, "plasma-source-cases").unwrap();
        let hts_summary = audit_source_value(&hts, "hts-source-cases").unwrap();

        assert_eq!(plasma_summary.case_count, 49);
        assert_eq!(plasma_summary.negative_control_count, 88);
        assert_eq!(plasma_summary.benchmark_target_count, 9);
        let plasma_case_ids: BTreeSet<_> = plasma
            .get("cases")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|case| case.get("case_id").and_then(Value::as_str))
            .collect();
        for expected in [
            "PLASMA-MINI-APPJ-QUEENBELFAST-2026",
            "PLASMA-MICROFLUIDIC-MEMBRANE-CHIP-2025",
            "PLASMA-ARRAY-LTCC-43-CELL-10CM-2025",
            "PLASMA-ARRAY-INDIVIDUAL-RF-IGNITION-5-CELL-2023",
            "PLASMA-PORTABLE-MOBIPLAS-ARGON-2024",
        ] {
            assert!(plasma_case_ids.contains(expected), "missing {expected}");
        }
        assert!(
            plasma
                .get("cases")
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .any(|case| {
                    case.get("case_id").and_then(Value::as_str)
                        == Some("FUSION-FUEL-DHE3-ARC-NEUTRON-PARTITION-2026")
                })
        );
        assert_eq!(hts_summary.case_count, 6);
        assert_eq!(hts_summary.negative_control_count, 12);
    }

    #[test]
    fn frozen_thermal_fixture_matches_every_expected_disposition() {
        let fixture: ThermalFixtureManifest = serde_json::from_str(include_str!(
            "../../data/research/thermal_store_accounting_fixtures_v0_1.json"
        ))
        .unwrap();
        validate_fixture_header(&fixture).unwrap();

        let mut scenarios = BTreeMap::<String, &ThermalScenario>::new();
        let mut results = BTreeMap::<String, ScenarioResult>::new();
        for scenario in &fixture.scenarios {
            let result = evaluate_scenario(scenario);
            assert_eq!(
                result.disposition, scenario.expected_disposition,
                "unexpected result for {}: {}",
                scenario.scenario_id, result.reason
            );
            scenarios.insert(scenario.scenario_id.clone(), scenario);
            results.insert(scenario.scenario_id.clone(), result);
        }

        for comparison in &fixture.comparisons {
            let result = evaluate_comparison(comparison, &scenarios, &results);
            assert_eq!(
                result.disposition, comparison.expected_disposition,
                "unexpected comparison result for {}: {}",
                comparison.comparison_id, result.reason
            );
        }
    }

    #[test]
    fn source_manifest_rejects_luminous_reproduction_promotion() {
        let source = serde_json::json!({
            "programme": "TEST",
            "claim_ceiling": "SOURCE_ONLY",
            "evidence_classes": [
                "PUBLISHED_LAB_EXPERIMENT",
                "PUBLISHED_REVIEW_OR_DATA_SYNTHESIS",
                "PUBLISHED_NUMERICAL_STUDY",
                "PUBLISHED_TECHNO_ECONOMIC_ANALYSIS"
            ],
            "cases": [{
                "case_id": "case-1",
                "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                "luminous_reproduction": true,
                "source": {"url": "https://example.org/source"}
            }],
            "negative_controls": [{
                "control_id": "nc-1",
                "expected_disposition": "REJECTED"
            }]
        });
        let error = audit_source_value(&source, "test.json").unwrap_err();
        assert!(error.to_string().contains("luminous_reproduction=false"));
    }

    #[test]
    fn source_manifest_rejects_undeclared_evidence_class() {
        let source = serde_json::json!({
            "programme": "TEST",
            "claim_ceiling": "SOURCE_ONLY",
            "evidence_classes": [
                "PUBLISHED_LAB_EXPERIMENT",
                "PUBLISHED_TECHNO_ECONOMIC_ANALYSIS",
                "PUBLISHED_REVIEW_OR_DATA_SYNTHESIS",
                "PUBLISHED_NUMERICAL_STUDY",
                "MANUFACTURER_REPORTED_OPERATIONAL_CLAIM",
                "MANUFACTURER_PRODUCT_CLAIM",
                "GOVERNMENT_RESEARCH_FACILITY_DESCRIPTION",
                "GOVERNMENT_ROADMAP"
            ],
            "cases": [{
                "case_id": "case-1",
                "evidence_class": "MANUFACTURER_GUARANTEE",
                "luminous_reproduction": false,
                "source": {"url": "https://example.org/source"}
            }],
            "negative_controls": [{
                "control_id": "nc-1",
                "expected_disposition": "REJECTED"
            }]
        });
        let error = audit_source_value(&source, "test.json").unwrap_err();
        assert!(error.to_string().contains("undeclared evidence class"));
    }

    fn minimal_dec_source_case(
        particle_energy_origin: Option<&str>,
        fusion_born_output_demonstrated: Option<bool>,
    ) -> Value {
        let mut case = serde_json::json!({
            "case_id": "dec-case-1",
            "lane": "FUSION_DIRECT_ENERGY_CONVERSION",
            "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
            "source": {"url": "https://example.org/dec"},
            "comparison_status": "SOURCE_SPECIFIC_NOT_MATCHED",
            "luminous_reproduction": false
        });
        if let Some(origin) = particle_energy_origin {
            case["particle_energy_origin"] = Value::String(origin.to_owned());
        }
        if let Some(demonstrated) = fusion_born_output_demonstrated {
            case["fusion_born_output_demonstrated"] = Value::Bool(demonstrated);
        }
        case
    }

    fn minimal_source_manifest_with_case(case: Value) -> Value {
        serde_json::json!({
            "programme": "TEST",
            "claim_ceiling": "SOURCE_ONLY",
            "evidence_classes": ["PUBLISHED_LAB_EXPERIMENT"],
            "cases": [case],
            "negative_controls": [{
                "control_id": "nc-1",
                "expected_disposition": "REJECTED"
            }]
        })
    }

    #[test]
    fn fusion_fuel_partition_source_case_requires_explicit_particle_origin() {
        let mut case = minimal_dec_source_case(None, Some(false));
        case["lane"] = Value::String("FUSION_FUEL_ENERGY_PARTITION".to_owned());
        let source = minimal_source_manifest_with_case(case);
        let error = audit_source_value(&source, "fuel-origin.json").unwrap_err();
        assert!(error.to_string().contains("particle_energy_origin"));
    }

    #[test]
    fn dec_source_case_requires_explicit_particle_energy_origin() {
        let source = minimal_source_manifest_with_case(minimal_dec_source_case(None, Some(false)));
        let error = audit_source_value(&source, "dec-origin.json").unwrap_err();
        assert!(error.to_string().contains("particle_energy_origin"));
    }

    #[test]
    fn dec_source_case_cannot_silently_claim_fusion_born_output() {
        let source = minimal_source_manifest_with_case(minimal_dec_source_case(
            Some("EXTERNALLY_HEATED_DEUTERIUM_PLASMA"),
            Some(true),
        ));
        let error = audit_source_value(&source, "dec-origin.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("fusion_born_output_demonstrated=false")
        );
    }

    fn nh3_synthesis_energy_gap_manifest() -> Value {
        serde_json::json!({
            "programme": "PLASMA-ENERGY-001",
            "claim_ceiling": "SOURCE_REGISTER_ONLY",
            "evidence_classes": [
                "PUBLISHED_LAB_EXPERIMENT",
                "PUBLISHED_TECHNO_ECONOMIC_ANALYSIS"
            ],
            "cases": [
                {
                    "case_id": "NH3-GAS-LIQUID-PLASMA-ELECTROCATALYTIC-TANDEM-2026",
                    "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                    "luminous_reproduction": false,
                    "comparison_status": "NOT_COMPARABLE_BOUNDARIES_UNRESOLVED",
                    "source": {"url": "https://doi.org/10.1002/aic.70407"},
                    "observations": [{"metric":"specific_energy_consumption","value":1161.0}]
                },
                {
                    "case_id": "NH3-SOEC-HB-OFFSHORE-2026",
                    "evidence_class": "PUBLISHED_TECHNO_ECONOMIC_ANALYSIS",
                    "luminous_reproduction": false,
                    "source": {"url": "https://doi.org/10.1016/j.ijhydene.2026.155184"},
                    "observations": [{"metric":"specific_electrical_energy","value":8.14}]
                },
                {
                    "case_id": "NH3-PLASMA-ELECTROCATALYSIS-JACS-2026",
                    "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                    "luminous_reproduction": false,
                    "source": {"url": "https://doi.org/10.1021/jacs.6c04148"},
                    "observations": [{"metric":"ammonia_area_production_rate","value":7.7}],
                    "state_transfer_evidence": {
                        "physical_observations": [
                            {"observation":"15N2 isotope labeling and plasma/electrolysis on-off controls","evidence_class":"PUBLISHED_EXPERIMENTAL_CONTROL"},
                            {"observation":"in-situ Raman evidence of N-N-containing surface intermediates","evidence_class":"PUBLISHED_IN_SITU_SPECTROSCOPY"},
                            {"observation":"NH3 area-normalized production rate and electrochemical Faradaic efficiency","evidence_class":"PUBLISHED_EXPERIMENTAL_PRODUCT_OBSERVATION"}
                        ],
                        "model_derived_estimates": [
                            {"metric":"effective_N2_vibrational_state_temperature","value":4300.0,"evidence_class":"MODEL_DERIVED","interface_direct_measurement":false},
                            {"metric":"N2_population_in_v4_to_v8","value":0.118,"evidence_class":"MODEL_DERIVED","interface_direct_measurement":false}
                        ],
                        "claim_policy": {
                            "model_derived_population_must_not_be_relabelled_physical_measurement":true,
                            "discharge_exit_population_must_not_be_substituted_for_interface_population":true,
                            "raman_intermediate_signal_must_not_be_substituted_for_quantitative_state_flux":true,
                            "isotope_attribution_is_not_equivalent_to_state_resolved_mechanism_proof":true,
                            "faradaic_efficiency_is_not_equivalent_to_full_system_energy_efficiency":true
                        },
                        "not_established":["calibrated state-resolved N2(v) distribution measured at the catalyst interface"]
                    }
                }
            ],
            "derived_reference_checks": [{
                "check_id":"NH3-PLASMA-TO-SOEC-HB-ENERGY-GAP-2026",
                "source_case_id":"NH3-GAS-LIQUID-PLASMA-ELECTROCATALYTIC-TANDEM-2026",
                "reference_case_id":"NH3-SOEC-HB-OFFSHORE-2026",
                "comparison_status":"NOT_COMPARABLE_BOUNDARIES_UNRESOLVED",
                "evidence_class":"DERIVED_ARITHMETIC_FROM_PUBLISHED_AMMONIA_SYSTEM_METRICS",
                "luminous_measurement":false,
                "inputs":{
                    "plasma_tandem_specific_energy_kWh_per_kg_NH3":1161.0,
                    "soec_hb_model_specific_energy_kWh_per_kg_NH3":8.14,
                    "jacs_area_rate_nmol_NH3_per_s_per_cm2":7.7,
                    "ammonia_molar_mass_g_per_mol":17.03052
                },
                "derivation":{
                    "plasma_tandem_yield_g_NH3_per_kWh":1000.0/1161.0,
                    "soec_hb_yield_g_NH3_per_kWh":1000.0/8.14,
                    "source_metric_energy_intensity_ratio":1161.0/8.14,
                    "jacs_area_productivity_kg_NH3_per_m2_per_day":7.7e-9*17.03052*86400.0*10000.0/1000.0,
                    "active_area_m2_for_1_tonne_NH3_per_day_at_jacs_rate":1000.0/(7.7e-9*17.03052*86400.0*10000.0/1000.0)
                },
                "interpretation":"Derived arithmetic only; boundaries remain unresolved."
            }],
            "negative_controls":[{"control_id":"NC-TEST","expected_disposition":"NOT_COMPARABLE"}]
        })
    }

    fn nh3_energy_yield_source_manifest() -> Value {
        serde_json::json!({
            "programme": "PLASMA-ENERGY-001",
            "claim_ceiling": "SOURCE_REGISTER_ONLY",
            "evidence_classes": ["PUBLISHED_LAB_EXPERIMENT"],
            "cases": [{
                "case_id": "NH3-CRACK-MW-ATM-MECHANISM-2026",
                "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                "luminous_reproduction": false,
                "source": {"url": "https://doi.org/10.1016/j.ijhydene.2026.156912"},
                "operating_points": [
                    {
                        "operating_point_id": "NH3-MW-ATM-HIGH-CONVERSION",
                        "ammonia_conversion_lower_bound_fraction": 0.99,
                        "hydrogen_energy_yield_g_H2_per_kWh": 45.0,
                        "hydrogen_energy_cost_kJ_per_mol": 242.0,
                        "hydrogen_production_rate_g_per_hour": null
                    },
                    {
                        "operating_point_id": "NH3-MW-ATM-HIGH-THROUGHPUT",
                        "ammonia_conversion_fraction": 0.924,
                        "hydrogen_energy_yield_g_H2_per_kWh": 50.0,
                        "hydrogen_energy_cost_kJ_per_mol": 217.0,
                        "hydrogen_production_rate_g_per_hour": 150.0
                    },
                    {
                        "operating_point_id": "NH3-MW-ATM-HIGH-ENERGY-YIELD",
                        "ammonia_conversion_fraction": 0.76,
                        "hydrogen_energy_yield_g_H2_per_kWh": 51.0,
                        "hydrogen_energy_cost_kJ_per_mol": 210.0,
                        "hydrogen_production_rate_g_per_hour": null
                    }
                ]
            }],
            "derived_reference_checks": [{
                "check_id": "NH3-MW-ENERGY-YIELD-TO-SPECIFIC-ENERGY-2026",
                "source_case_id": "NH3-CRACK-MW-ATM-MECHANISM-2026",
                "evidence_class": "DERIVED_ARITHMETIC_FROM_PUBLISHED_HYDROGEN_ENERGY_YIELD",
                "luminous_measurement": false,
                "inputs": {
                    "near_complete_conversion_energy_yield_g_H2_per_kWh": 45.0,
                    "intermediate_92_4_percent_energy_yield_g_H2_per_kWh": 50.0,
                    "best_energy_yield_g_H2_per_kWh": 51.0
                },
                "derivation": {
                    "near_complete_conversion_specific_energy_kWh_per_kg_H2": 1000.0 / 45.0,
                    "intermediate_92_4_percent_specific_energy_kWh_per_kg_H2": 1000.0 / 50.0,
                    "best_energy_yield_specific_energy_kWh_per_kg_H2": 1000.0 / 51.0,
                    "relative_specific_energy_reduction_at_92_4_percent_conversion": 1.0 - 45.0 / 50.0,
                    "relative_specific_energy_reduction_at_76_percent_conversion": 1.0 - 45.0 / 51.0
                },
                "interpretation": "Two distinct source operating points; not a matched process result."
            }],
            "negative_controls": [{
                "control_id": "NC-024",
                "expected_disposition": "REJECT_OPERATING_POINT_MISMATCH"
            }]
        })
    }


    fn hybrid_appj_source_manifest() -> Value {
        let registered: Value = serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        )).unwrap();
        let case = registered["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["case_id"] == "CH4-PLASMA-THERMAL-APPJ-2026")
            .unwrap()
            .clone();
        serde_json::json!({
            "programme": "PLASMA-ENERGY-001",
            "claim_ceiling": "SOURCE_REGISTER_ONLY",
            "evidence_classes": [
                "PUBLISHED_LAB_EXPERIMENT",
                "PUBLISHED_TECHNO_ECONOMIC_ANALYSIS",
                "PUBLISHED_REVIEW_OR_DATA_SYNTHESIS",
                "PUBLISHED_NUMERICAL_STUDY",
                "MANUFACTURER_REPORTED_OPERATIONAL_CLAIM",
                "MANUFACTURER_PRODUCT_CLAIM",
                "GOVERNMENT_RESEARCH_FACILITY_DESCRIPTION",
                "GOVERNMENT_ROADMAP"
            ],
            "cases": [case],
            "negative_controls": [{
                "control_id": "NC-APPJ",
                "expected_disposition": "BLOCKED"
            }]
        })
    }

    #[test]
    fn source_manifest_accepts_hybrid_appj_relative_claims_without_absolute_performance_invention() {
        let source = hybrid_appj_source_manifest();
        let summary = audit_source_value(&source, "hybrid-appj.json").unwrap();
        assert_eq!(summary.case_count, 1);
    }

    #[test]
    fn source_manifest_rejects_relative_appj_gain_promoted_to_absolute_conversion() {
        let mut source = hybrid_appj_source_manifest();
        source["cases"][0]["reported_claims"][0]["absolute_conversion_fraction"] =
            serde_json::json!(0.95);
        let error = audit_source_value(&source, "hybrid-appj.json").unwrap_err();
        assert!(error.to_string().contains("cannot be promoted to absolute methane conversion"));
    }

    #[test]
    fn source_manifest_rejects_carbon_deposition_promoted_to_feed_carbon_recovery() {
        let mut source = hybrid_appj_source_manifest();
        source["cases"][0]["reported_claims"][1]["feed_carbon_recovery_fraction"] =
            serde_json::json!(0.36);
        let error = audit_source_value(&source, "hybrid-appj.json").unwrap_err();
        assert!(error.to_string().contains("cannot be promoted to captured or recovered carbon yield"));
    }

    #[test]
    fn source_manifest_rejects_appj_carbon_as_qualified_coproduct() {
        let mut source = hybrid_appj_source_manifest();
        source["cases"][0]["coproduct_accounting"]["carbon_product_qualification_status"] =
            serde_json::json!("VALIDATED");
        let error = audit_source_value(&source, "hybrid-appj.json").unwrap_err();
        assert!(error.to_string().contains("must not promote carbon-product grade"));
    }

    #[test]
    fn source_manifest_rejects_appj_h2_rate_inferred_without_product_measurement() {
        let mut source = hybrid_appj_source_manifest();
        source["cases"][0]["product_slate"][0]["quantitative_rate"] =
            serde_json::json!(100.0);
        let error = audit_source_value(&source, "hybrid-appj.json").unwrap_err();
        assert!(error.to_string().contains("H2 rate, purity and pressure must remain unquantified"));
    }

    #[test]
    fn source_manifest_accepts_pulsed_plasma_methane_points_and_regeneration_evidence() {
        let source: Value = serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        )).unwrap();
        let summary = audit_source_value(&source, "pulsed-methane-source.json").unwrap();
        assert!(summary.case_count >= 1);
    }

    #[test]
    fn source_manifest_rejects_pure_methane_energy_borrowed_by_diluted_conversion_point() {
        let mut source: Value = serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        )).unwrap();
        let case = source["cases"].as_array_mut().unwrap().iter_mut().find(|item| {
            item["case_id"] == "CH4-PLASMA-PULSED-HEATING-REGEN-2026"
        }).unwrap();
        case["operating_points"][0]["specific_energy_kWh_per_kg_H2"] = serde_json::json!(27.0);
        let error = audit_source_value(&source, "pulsed-methane-source.json").unwrap_err();
        assert!(error.to_string().contains("must not be borrowed by the diluted-feed point"));
    }

    #[test]
    fn source_manifest_rejects_diluted_conversion_borrowed_by_pure_methane_energy_point() {
        let mut source: Value = serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        )).unwrap();
        let case = source["cases"].as_array_mut().unwrap().iter_mut().find(|item| {
            item["case_id"] == "CH4-PLASMA-PULSED-HEATING-REGEN-2026"
        }).unwrap();
        case["operating_points"][1]["methane_conversion_fraction"] = serde_json::json!(0.95);
        let error = audit_source_value(&source, "pulsed-methane-source.json").unwrap_err();
        assert!(error.to_string().contains("must not be borrowed by the pure-methane energy point"));
    }

    #[test]
    fn source_manifest_rejects_missing_co2_plasma_regeneration_energy_status() {
        let mut source: Value = serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        )).unwrap();
        let case = source["cases"].as_array_mut().unwrap().iter_mut().find(|item| {
            item["case_id"] == "CH4-PLASMA-PULSED-HEATING-REGEN-2026"
        }).unwrap();
        case["regeneration_evidence"]["method"] = serde_json::json!("unspecified");
        let error = audit_source_value(&source, "pulsed-methane-source.json").unwrap_err();
        assert!(error.to_string().contains("must explicitly identify CO2 plasma"));
    }

    #[test]
    fn source_manifest_accepts_pulsed_plasma_methane_full_chain_target() {
        let mut source = minimal_benchmark_target_manifest("METHANE_PYROLYSIS_PULSED_HEATING");
        replace_placeholder_with_registered_source_case(
            &mut source,
            "CH4-PLASMA-PULSED-HEATING-REGEN-2026",
        );
        append_registered_source_case(&mut source, "METHANE-PYROLYSIS-HEATING-REVIEW-2026");
        source["benchmark_targets"][0]["comparator_case_ids"] = serde_json::json!([
            "CH4-PLASMA-PULSED-HEATING-REGEN-2026",
            "METHANE-PYROLYSIS-HEATING-REVIEW-2026"
        ]);
        let summary = audit_source_value(&source, "pph-target.json").unwrap();
        assert_eq!(summary.benchmark_target_count, 1);
    }

    #[test]
    fn source_manifest_accepts_ammonia_dynamic_flex_target_with_matched_haber_bosch_baselines() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_SYNTHESIS_DYNAMIC_FLEXIBILITY");
        replace_placeholder_with_registered_source_case(
            &mut source,
            "NH3-HB-ROBUST-LOAD-FLEX-2026",
        );
        append_registered_source_case(&mut source, "NH3-HB-FATIGUE-YEAR-SCHEDULING-2026");
        append_registered_source_case(&mut source, "NH3-P2A-THREE-LEVEL-DISPATCH-2026");
        source["benchmark_targets"][0]["comparator_case_ids"] = serde_json::json!([
            "NH3-HB-ROBUST-LOAD-FLEX-2026",
            "NH3-HB-FATIGUE-YEAR-SCHEDULING-2026",
            "NH3-P2A-THREE-LEVEL-DISPATCH-2026"
        ]);
        let summary = audit_source_value(&source, "ammonia-flex-target.json").unwrap();
        assert_eq!(summary.benchmark_target_count, 1);
    }

    #[test]
    fn source_manifest_rejects_plasma_source_ramp_substituted_for_accepted_nh3_ramp() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_SYNTHESIS_DYNAMIC_FLEXIBILITY");
        replace_placeholder_with_registered_source_case(
            &mut source,
            "NH3-HB-ROBUST-LOAD-FLEX-2026",
        );
        append_registered_source_case(&mut source, "NH3-HB-FATIGUE-YEAR-SCHEDULING-2026");
        append_registered_source_case(&mut source, "NH3-P2A-THREE-LEVEL-DISPATCH-2026");
        source["benchmark_targets"][0]["comparator_case_ids"] = serde_json::json!([
            "NH3-HB-ROBUST-LOAD-FLEX-2026",
            "NH3-HB-FATIGUE-YEAR-SCHEDULING-2026",
            "NH3-P2A-THREE-LEVEL-DISPATCH-2026"
        ]);
        source["benchmark_targets"][0]["claim_guards"]["plant_level_accepted_nh3_ramp_measured_not_plasma_source_ramp"] = serde_json::json!(false);
        let error = audit_source_value(&source, "ammonia-flex-target.json").unwrap_err();
        assert!(error.to_string().contains("plant_level_accepted_nh3_ramp_measured_not_plasma_source_ramp"));
    }

    #[test]
    fn source_manifest_accepts_nh3_plasma_to_soec_hb_gap_arithmetic_without_claiming_comparability() {
        let source = nh3_synthesis_energy_gap_manifest();
        let summary = audit_source_value(&source, "nh3-synthesis-gap.json").unwrap();
        assert_eq!(summary.case_count, 3);
        assert_eq!(summary.derived_reference_check_count, 1);
    }

    #[test]
    fn source_manifest_rejects_wrong_nh3_source_energy_intensity_ratio() {
        let mut source = nh3_synthesis_energy_gap_manifest();
        source["derived_reference_checks"][0]["derivation"]["source_metric_energy_intensity_ratio"] =
            serde_json::json!(10.0);
        let error = audit_source_value(&source, "nh3-synthesis-gap.json").unwrap_err();
        assert!(error.to_string().contains("source metric energy-intensity ratio"));
    }

    #[test]
    fn source_manifest_rejects_nh3_source_gap_promoted_to_comparable() {
        let mut source = nh3_synthesis_energy_gap_manifest();
        source["derived_reference_checks"][0]["comparison_status"] =
            serde_json::json!("COMPARABLE");
        let error = audit_source_value(&source, "nh3-synthesis-gap.json").unwrap_err();
        assert!(error.to_string().contains("must remain NOT_COMPARABLE"));
    }

    #[test]
    fn source_manifest_rejects_wrong_jacs_geometric_scale_extrapolation() {
        let mut source = nh3_synthesis_energy_gap_manifest();
        source["derived_reference_checks"][0]["derivation"]
            ["active_area_m2_for_1_tonne_NH3_per_day_at_jacs_rate"] = serde_json::json!(100.0);
        let error = audit_source_value(&source, "nh3-synthesis-gap.json").unwrap_err();
        assert!(error.to_string().contains("geometric active-area extrapolation"));
    }

    #[test]
    fn ammonia_synthesis_competitiveness_target_requires_ten_percent_full_chain_gain() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_SYNTHESIS_COMPETITIVENESS");
        source["cases"][0]["case_id"] = serde_json::json!("NH3-SOEC-HB-OFFSHORE-2026");
        source["benchmark_targets"][0]["comparator_case_ids"] =
            serde_json::json!(["NH3-SOEC-HB-OFFSHORE-2026"]);
        let summary = audit_source_value(&source, "nh3-competitiveness-target.json").unwrap();
        assert_eq!(summary.benchmark_target_count, 1);
    }

    #[test]
    fn ammonia_synthesis_competitiveness_target_rejects_intermediate_three_g_per_kwh_as_parity() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_SYNTHESIS_COMPETITIVENESS");
        source["cases"][0]["case_id"] = serde_json::json!("NH3-SOEC-HB-OFFSHORE-2026");
        source["benchmark_targets"][0]["metric_thresholds"]["plasma_candidate_max_specific_energy_kWh_per_kg_NH3"] =
            serde_json::json!(333.333);
        source["benchmark_targets"][0]["comparator_case_ids"] =
            serde_json::json!(["NH3-SOEC-HB-OFFSHORE-2026"]);
        let error = audit_source_value(&source, "nh3-competitiveness-target.json").unwrap_err();
        assert!(error.to_string().contains("does not beat the matched SOEC-HB reference"));
    }

    #[test]
    fn source_manifest_accepts_correct_nh3_energy_yield_derivation() {
        let source = nh3_energy_yield_source_manifest();
        let summary = audit_source_value(&source, "nh3-derived.json").unwrap();
        assert_eq!(summary.case_count, 1);
        assert_eq!(summary.derived_reference_check_count, 1);
    }

    #[test]
    fn source_manifest_rejects_150_grams_per_hour_borrowed_by_high_conversion_point() {
        let mut source = nh3_energy_yield_source_manifest();
        source["cases"][0]["operating_points"][0]["hydrogen_production_rate_g_per_hour"] =
            serde_json::json!(150.0);
        let error = audit_source_value(&source, "nh3-derived.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must not borrow the 150 g/h throughput value")
        );
    }

    #[test]
    fn source_manifest_rejects_cross_paired_ammonia_energy_yield() {
        let mut source = nh3_energy_yield_source_manifest();
        source["cases"][0]["operating_points"][1]["hydrogen_energy_yield_g_H2_per_kWh"] =
            serde_json::json!(45.0);
        let error = audit_source_value(&source, "nh3-derived.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("150 g/h point must retain its 50 g/kWh energy yield")
        );
    }

    #[test]
    fn source_manifest_rejects_incorrect_nh3_specific_energy_arithmetic() {
        let mut source = nh3_energy_yield_source_manifest();
        source["derived_reference_checks"][0]["derivation"]["best_energy_yield_specific_energy_kWh_per_kg_H2"] =
            serde_json::json!(18.0);
        let error = audit_source_value(&source, "nh3-derived.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("best-yield specific-energy derivation")
        );
    }

    #[test]
    fn source_manifest_rejects_cross_point_throughput_energy_yield_mismatch() {
        let mut source = nh3_energy_yield_source_manifest();
        source["derived_reference_checks"][0]["derivation"]["intermediate_92_4_percent_specific_energy_kWh_per_kg_H2"] =
            serde_json::json!(22.2222222222);
        let error = audit_source_value(&source, "nh3-derived.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("92.4-percent specific-energy derivation")
        );
    }

    #[test]
    fn source_manifest_rejects_derived_check_with_unknown_case_reference() {
        let mut source = nh3_energy_yield_source_manifest();
        source["derived_reference_checks"][0]["source_case_id"] = serde_json::json!("MISSING-CASE");
        let error = audit_source_value(&source, "nh3-derived.json").unwrap_err();
        assert!(error.to_string().contains("references unknown source case"));
    }

    #[test]
    fn source_manifest_rejects_derived_check_claimed_as_luminous_measurement() {
        let mut source = nh3_energy_yield_source_manifest();
        source["derived_reference_checks"][0]["luminous_measurement"] = serde_json::json!(true);
        let error = audit_source_value(&source, "nh3-derived.json").unwrap_err();
        assert!(error.to_string().contains("luminous_measurement=false"));
    }

    fn co2_counterflow_source_manifest() -> Value {
        serde_json::json!({
            "programme": "PLASMA-ENERGY-001",
            "claim_ceiling": "SOURCE_REGISTER_ONLY",
            "evidence_classes": ["PUBLISHED_LAB_EXPERIMENT"],
            "cases": [{
                "case_id": "CO2-COUNTERFLOW-ARC-EXPERIMENT-2026",
                "lane": "CO2_WARM_PLASMA_QUENCHING",
                "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                "luminous_reproduction": false,
                "source": {"url": "https://doi.org/10.1016/j.jece.2026.122322"},
                "configurations": [
                    {
                        "id": "counterflow-magd-792w",
                        "gas_flow_slm": 4.0,
                        "electrical_power_w": 792.0,
                        "observations": [
                            {"metric":"co2_conversion","value":0.236,"unit":"fraction"},
                            {"metric":"energy_efficiency","value":0.251,"unit":"fraction"}
                        ]
                    },
                    {
                        "id": "counterflow-low-power-302w",
                        "electrical_power_w": 302.0,
                        "observations": [
                            {"metric":"energy_efficiency","value":0.462,"unit":"fraction"},
                            {"metric":"co2_conversion","value":null,"unit":"fraction"}
                        ]
                    },
                    {
                        "id": "counterflow-ratio-distance-optimization",
                        "observations": [
                            {"metric":"counterflow_gas_ratio","value":2.5,"unit":"ratio"},
                            {"metric":"counterflow_distance","value":30.0,"unit":"mm"},
                            {"metric":"relative_conversion_increase","value":2.073,"unit":"fractional_increase"},
                            {"metric":"relative_energy_efficiency_increase","value":2.036,"unit":"fractional_increase"}
                        ]
                    }
                ]
            }],
            "negative_controls": [{
                "control_id":"NC-CO2-QUENCH-001",
                "expected_disposition":"REJECT_OPERATING_POINT_IDENTITY_MISMATCH"
            }]
        })
    }

    #[test]
    fn source_manifest_accepts_separate_co2_counterflow_operating_points() {
        let summary =
            audit_source_value(&co2_counterflow_source_manifest(), "co2-counterflow.json").unwrap();
        assert_eq!(summary.case_count, 1);
    }

    #[test]
    fn source_manifest_rejects_borrowing_792w_conversion_for_302w_case() {
        let mut source = co2_counterflow_source_manifest();
        source["cases"][0]["configurations"][1]["observations"][1]["value"] =
            serde_json::json!(0.236);
        let error = audit_source_value(&source, "co2-counterflow.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("302 W conversion must remain unknown")
        );
    }

    #[test]
    fn source_manifest_rejects_relative_improvement_as_absolute_efficiency() {
        let mut source = co2_counterflow_source_manifest();
        source["cases"][0]["configurations"][2]["observations"][3]["unit"] =
            serde_json::json!("fraction");
        let error = audit_source_value(&source, "co2-counterflow.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must remain a relative increase")
        );
    }

    #[test]
    fn source_manifest_rejects_duplicate_co2_operating_point_ids() {
        let mut source = co2_counterflow_source_manifest();
        source["cases"][0]["configurations"][1]["id"] = serde_json::json!("counterflow-magd-792w");
        let error = audit_source_value(&source, "co2-counterflow.json").unwrap_err();
        assert!(error.to_string().contains("duplicate configuration ID"));
    }

    fn minimal_methane_source_manifest() -> Value {
        serde_json::json!({
            "programme": "PLASMA-ENERGY-001",
            "claim_ceiling": "SOURCE_REGISTER_ONLY",
            "evidence_classes": ["PUBLISHED_TECHNO_ECONOMIC_ANALYSIS"],
            "cases": [{
                "case_id": "METHANE-CASE",
                "lane": "METHANE_PYROLYSIS",
                "evidence_class": "PUBLISHED_TECHNO_ECONOMIC_ANALYSIS",
                "luminous_reproduction": false,
                "source": {"url": "https://example.org/methane"},
                "coproduct_accounting": {
                    "policy": "MODELED_CARBON_BLACK_SALES_REVENUE",
                    "credit_claim_status": "SOURCE_MODEL_ONLY",
                    "carbon_product_qualification_status": "NOT_VALIDATED",
                    "carbon_market_capacity_status": "NOT_VALIDATED",
                    "offtake_status": "NOT_VALIDATED",
                    "realized_revenue_evidence": false,
                    "no_credit_baseline_required": true
                }
            }],
            "negative_controls": [{
                "control_id": "NC-METHANE",
                "expected_disposition": "BLOCKED"
            }]
        })
    }

    #[test]
    fn methane_source_case_requires_conservative_coproduct_accounting() {
        let source = minimal_methane_source_manifest();
        let summary = audit_source_value(&source, "methane.json").unwrap();
        assert_eq!(summary.case_count, 1);
    }

    #[test]
    fn methane_source_case_rejects_unvalidated_carbon_black_as_qualified() {
        let mut source = minimal_methane_source_manifest();
        source["cases"][0]["coproduct_accounting"]["carbon_product_qualification_status"] =
            serde_json::json!("VALIDATED");
        let error = audit_source_value(&source, "methane.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must not promote carbon-product grade")
        );
    }

    #[test]
    fn methane_source_case_rejects_realized_revenue_without_physical_evidence() {
        let mut source = minimal_methane_source_manifest();
        source["cases"][0]["coproduct_accounting"]["realized_revenue_evidence"] =
            serde_json::json!(true);
        let error = audit_source_value(&source, "methane.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot claim realized carbon-black revenue")
        );
    }

    #[test]
    fn methane_source_case_requires_no_credit_baseline() {
        let mut source = minimal_methane_source_manifest();
        source["cases"][0]["coproduct_accounting"]["no_credit_baseline_required"] =
            serde_json::json!(false);
        let error = audit_source_value(&source, "methane.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must preserve a no-credit baseline")
        );
    }

    fn registered_source_case(case_id: &str) -> Value {
        let registered: Value = serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        )).unwrap();
        registered["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["case_id"] == case_id)
            .unwrap_or_else(|| panic!("registered comparator case missing: {case_id}"))
            .clone()
    }

    fn replace_placeholder_with_registered_source_case(source: &mut Value, case_id: &str) {
        let case = registered_source_case(case_id);
        let cases = source["cases"].as_array_mut().unwrap();
        if let Some(index) = cases.iter().position(|item| item["case_id"] == "case-1") {
            cases[index] = case;
        } else if let Some(index) = cases.iter().position(|item| item["case_id"] == case_id) {
            cases[index] = case;
        } else {
            cases.push(case);
        }
    }

    fn append_registered_source_case(source: &mut Value, case_id: &str) {
        let case = registered_source_case(case_id);
        let cases = source["cases"].as_array_mut().unwrap();
        if !cases.iter().any(|item| item["case_id"] == case_id) {
            cases.push(case);
        }
    }

    fn minimal_benchmark_target_manifest(lane: &str) -> Value {
        let metrics = if lane == "AMMONIA_CRACKING" {
            serde_json::json!({
                "ammonia_conversion_min_fraction": 0.99,
                "hydrogen_energy_yield_min_g_H2_per_kWh": 51.0,
                "hydrogen_production_rate_min_g_H2_per_hour": 150.0,
                "all_thresholds_must_hold_at_same_operating_point": true
            })
        } else if lane == "METHANE_PYROLYSIS" {
            serde_json::json!({
                "methane_conversion_min_fraction": 0.92,
                "hydrogen_selectivity_min_fraction": 0.99,
                "levelized_cost_hydrogen_max_usd_per_kg": 0.70,
                "all_thresholds_must_hold_at_same_operating_point": true
            })
        } else if lane == "AMMONIA_SYNTHESIS" {
            serde_json::json!({
                "ammonia_plant_gate_energy_yield_min_g_NH3_per_kWh": 3.0,
                "all_thresholds_must_hold_at_same_operating_point": true
            })
        } else if lane == "AMMONIA_SYNTHESIS_COMPETITIVENESS" {
            serde_json::json!({
                "integrated_soec_hb_reference_specific_energy_kWh_per_kg_NH3": 8.14,
                "plasma_candidate_max_specific_energy_kWh_per_kg_NH3": 7.32,
                "required_improvement_vs_integrated_reference_fraction": 0.10,
                "all_thresholds_must_hold_at_same_operating_point": true
            })
        } else if lane == "METHANE_PYROLYSIS_PULSED_HEATING" {
            serde_json::json!({
                "methane_conversion_min_fraction": 0.90,
                "hydrogen_selectivity_min_fraction": 0.99,
                "accepted_h2_specific_energy_max_kWh_per_kg": 25.0,
                "minimum_reaction_regeneration_cycles": 100.0,
                "accepted_h2_productivity_retention_after_cycles_min_fraction": 0.90,
                "all_thresholds_must_hold_at_same_operating_point_and_feed": true
            })
        } else if lane == "AMMONIA_SYNTHESIS_DYNAMIC_FLEXIBILITY" {
            serde_json::json!({
                "accepted_nh3_min_load_fraction": 0.10,
                "accepted_nh3_max_load_fraction": 1.20,
                "accepted_nh3_ramp_rate_percent_nominal_per_minute": 3.0,
                "year_long_renewable_profile_required": true,
                "all_full_chain_quality_and_fatigue_gates_must_hold": true
            })
        } else if lane == "AMMONIA_N2_VIBRATIONAL_TRANSFER" {
            serde_json::json!({
                "all_required_evidence_gates_must_hold": true,
                "state_resolved_interface_population_or_qualified_operator_required": true,
                "discharge_to_interface_transfer_uncertainty_required": true,
                "isotope_attributed_product_control_required": true,
                "matched_thermal_and_electrical_controls_required": true,
                "full_grid_to_accepted_nh3_boundary_required": true
            })
        } else if lane == "AMMONIA_CRACKING_THERMAL_INTEGRATION" {
            serde_json::json!({
                "ammonia_conversion_min_fraction": 0.99,
                "accepted_h2_production_rate_min_g_per_hour": 150.0,
                "net_grid_specific_energy_reduction_min_fraction": 0.15,
                "heat_balance_abs_residual_fraction_max": 0.05,
                "all_thresholds_must_hold_at_same_operating_point": true
            })
        } else {
            serde_json::json!({
                "syngas_cost_eur_per_tonne_max": 600.0,
                "plant_efficiency_min_fraction": 0.44,
                "all_thresholds_must_hold_at_same_operating_point": true
            })
        };
        let guards = if lane == "AMMONIA_CRACKING" {
            serde_json::json!({
                "full_wall_plug_to_accepted_h2_boundary_required": true
            })
        } else if lane == "METHANE_PYROLYSIS" {
            serde_json::json!({
                "no_credit_baseline_required": true,
                "avoided_product_credit_requires_independent_carbon_grade_qualification": true
            })
        } else if lane == "AMMONIA_SYNTHESIS" {
            serde_json::json!({
                "hydrogen_generation_energy_included": true,
                "plasma_and_electrolysis_electricity_included": true,
                "gas_feed_and_product_separation_energy_included": true,
                "accepted_nh3_product_purity_and_pressure_contract_required": true,
                "matched_haber_bosch_baseline_includes_hydrogen_energy": true
            })
        } else if lane == "AMMONIA_SYNTHESIS_COMPETITIVENESS" {
            serde_json::json!({
                "plant_gate_grid_to_accepted_nh3_boundary_required": true,
                "hydrogen_generation_nitrogen_separation_and_all_balance_of_plant_included": true,
                "same_nh3_purity_pressure_and_delivery_contract_required": true,
                "baseline_exclusions_wind_farm_storage_ammonia_storage_and_transport_explicit": true,
                "desalination_and_feedwater_conditioning_boundary_reconciled": true,
                "availability_capacity_factor_and_duty_cycle_matched": true,
                "no_faraday_efficiency_or_area_rate_substitutes_for_full_chain_energy": true,
                "no_superiority_claim_before_source_data_reconstruction_and_uncertainty_analysis": true
            })
        } else if lane == "METHANE_PYROLYSIS_PULSED_HEATING" {
            serde_json::json!({
                "diluted_and_pure_methane_operating_points_must_remain_separate": true,
                "full_grid_to_accepted_h2_energy_boundary_required": true,
                "gas_separation_purification_and_compression_included": true,
                "co2_plasma_regeneration_energy_and_carbon_streams_included": true,
                "methane_derived_carbon_balance_and_unknowns_reported": true,
                "no_credit_carbon_baseline_required": true,
                "catalyst_support_electrode_and_replacement_lifecycle_included": true,
                "long_run_cycles_and_failed_batches_in_denominator": true,
                "matched_shock_wave_and_microwave_comparators_required": true
            })
        } else if lane == "AMMONIA_SYNTHESIS_DYNAMIC_FLEXIBILITY" {
            serde_json::json!({
                "plant_level_accepted_nh3_ramp_measured_not_plasma_source_ramp": true,
                "hydrogen_generation_and_storage_dynamics_included": true,
                "nitrogen_and_plasma_or_nox_capture_transfer_dynamics_included": true,
                "synthesis_loop_pressure_and_temperature_fatigue_included": true,
                "accepted_product_purity_pressure_and_delivery_contract_matched": true,
                "startup_shutdown_standby_and_failed_batches_counted": true,
                "buffer_capacity_and_renewable_curtailment_reported": true,
                "baseline_uses_10_to_120_percent_3_percent_per_minute_haber_bosch_model": true,
                "no_flexibility_superiority_claim_without_same_trace_replay": true
            })
        } else if lane == "AMMONIA_N2_VIBRATIONAL_TRANSFER" {
            serde_json::json!({
                "direct_catalyst_interface_state_measurement_or_independently_validated_measurement_operator_required": true,
                "modelled_state_temperature_remains_model_evidence": true,
                "measured_intermediate_and_quantitative_flux_remain_distinct_measurands": true,
                "plasma_only_and_electrolysis_only_controls_required": true,
                "isotope_product_attribution_required_for_nitrogen_source_claim": true,
                "catalyst_lot_and_surface_identity_must_match": true,
                "all_energy_inputs_and_accepted_product_separation_included": true,
                "no_claim_of_unique_vibrational_causality_without_matched_thermal_controls": true
            })
        } else if lane == "AMMONIA_CRACKING_THERMAL_INTEGRATION" {
            serde_json::json!({
                "paired_no_recovery_baseline_same_feed_and_operating_point_required": true,
                "full_grid_to_accepted_h2_boundary_required": true,
                "recovered_heat_must_be_measured_and_useful_at_declared_process_interface": true,
                "insulation_or_recuperator_pressure_drop_and_all_auxiliaries_included": true,
                "thermal_storage_and_heat_exchanger_losses_included": true,
                "no_double_counting_of_recirculated_heat_or_reaction_enthalpy": true,
                "thermal_energy_balance_must_close_with_uncertainty": true,
                "product_purity_pressure_and_delivery_contract_unchanged": true,
                "duty_cycle_and_steady_state_duration_matched": true,
                "comparison_must_report_measurement_uncertainty_and_grouped_repeats": true
            })
        } else {
            serde_json::json!({
                "matched_syngas_composition_and_quality_required": true,
                "full_plant_grid_side_energy_boundary_required": true,
                "electricity_price_and_capacity_factor_scenario_required": true,
                "downstream_product_service_boundary_required": true,
                "unqualified_coproduct_credit_forbidden": true
            })
        };
        serde_json::json!({
            "programme": "PLASMA-ENERGY-001",
            "claim_ceiling": "SOURCE_REGISTER_ONLY",
            "evidence_classes": [
                "PUBLISHED_LAB_EXPERIMENT",
                "PUBLISHED_TECHNO_ECONOMIC_ANALYSIS",
                "PUBLISHED_REVIEW_OR_DATA_SYNTHESIS",
                "PUBLISHED_NUMERICAL_STUDY",
                "MANUFACTURER_REPORTED_OPERATIONAL_CLAIM",
                "MANUFACTURER_PRODUCT_CLAIM",
                "GOVERNMENT_RESEARCH_FACILITY_DESCRIPTION",
                "GOVERNMENT_ROADMAP"
            ],
            "cases": [
                {
                    "case_id": "case-1",
                    "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                    "luminous_reproduction": false,
                    "source": {"url": "https://example.org/source"}
                }
            ],
            "benchmark_targets": [{
                "target_id": "target-1",
                "lane": lane,
                "status": "RESEARCH_TARGET_NOT_RESULT",
                "comparator_case_ids": ["case-1"],
                "metric_thresholds": metrics,
                "claim_guards": guards,
                "credit_policies": ["NO_CREDIT", "ALLOCATION", "AVOIDED_PRODUCT_CREDIT"]
            }],
            "negative_controls": [{
                "control_id": "nc-1",
                "expected_disposition": "BLOCKED"
            }]
        })
    }

    #[test]
    fn source_manifest_accepts_ammonia_pareto_target_with_matched_operating_point() {
        let source = minimal_benchmark_target_manifest("AMMONIA_CRACKING");
        let summary = audit_source_value(&source, "ammonia-target.json").unwrap();
        assert_eq!(summary.benchmark_target_count, 1);
    }

    #[test]
    fn source_manifest_accepts_methane_target_with_no_credit_baseline() {
        let source = minimal_benchmark_target_manifest("METHANE_PYROLYSIS");
        let summary = audit_source_value(&source, "methane-target.json").unwrap();
        assert_eq!(summary.benchmark_target_count, 1);
    }

    #[test]
    fn source_manifest_accepts_ammonia_synthesis_full_chain_target() {
        let source = minimal_benchmark_target_manifest("AMMONIA_SYNTHESIS");
        let summary = audit_source_value(&source, "ammonia-synthesis-target.json").unwrap();
        assert_eq!(summary.benchmark_target_count, 1);
    }

    #[test]
    fn source_manifest_rejects_ammonia_synthesis_target_without_h2_energy() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_SYNTHESIS");
        source["benchmark_targets"][0]["claim_guards"]["hydrogen_generation_energy_included"] =
            serde_json::json!(false);
        let error = audit_source_value(&source, "ammonia-synthesis-target.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("hydrogen_generation_energy_included")
        );
    }

    #[test]
    fn source_manifest_accepts_n2_vibrational_transfer_target() {
        let source = minimal_benchmark_target_manifest("AMMONIA_N2_VIBRATIONAL_TRANSFER");
        let summary = audit_source_value(&source, "n2-vibrational-target.json").unwrap();
        assert_eq!(summary.benchmark_target_count, 1);
    }

    #[test]
    fn source_manifest_rejects_n2_vibrational_target_without_interface_gate() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_N2_VIBRATIONAL_TRANSFER");
        source["benchmark_targets"][0]["claim_guards"]
            ["direct_catalyst_interface_state_measurement_or_independently_validated_measurement_operator_required"] =
            serde_json::json!(false);
        let error = audit_source_value(&source, "n2-vibrational-target.json").unwrap_err();
        assert!(error.to_string().contains("direct_catalyst_interface_state_measurement"));
    }

    #[test]
    fn source_manifest_rejects_modelled_n2_population_promoted_to_interface_measurement() {
        let mut source: Value = serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        )).unwrap();
        let case = source["cases"].as_array_mut().unwrap().iter_mut().find(|item| {
            item["case_id"] == "NH3-PLASMA-ELECTROCATALYSIS-JACS-2026"
        }).unwrap();
        case["state_transfer_evidence"]["model_derived_estimates"][0]
            ["interface_direct_measurement"] = serde_json::json!(true);
        let error = audit_source_value(&source, "n2-vibrational-source.json").unwrap_err();
        assert!(error.to_string().contains("cannot claim a direct catalyst-interface measurement"));
    }

    #[test]
    fn source_manifest_rejects_missing_n2_isotope_attribution_control() {
        let mut source: Value = serde_json::from_str(include_str!(
            "../../data/research/plasma_energy_reference_cases_v0_1.json"
        )).unwrap();
        let case = source["cases"].as_array_mut().unwrap().iter_mut().find(|item| {
            item["case_id"] == "NH3-PLASMA-ELECTROCATALYSIS-JACS-2026"
        }).unwrap();
        case["state_transfer_evidence"]["physical_observations"][0]["observation"] =
            serde_json::json!("on/off controls only");
        let error = audit_source_value(&source, "n2-vibrational-source.json").unwrap_err();
        assert!(error.to_string().contains("preserve 15N2 attribution"));
    }

    #[test]
    fn source_manifest_accepts_ammonia_thermal_integration_target() {
        let source = minimal_benchmark_target_manifest("AMMONIA_CRACKING_THERMAL_INTEGRATION");
        let summary = audit_source_value(&source, "ammonia-thermal-target.json").unwrap();
        assert_eq!(summary.benchmark_target_count, 1);
    }

    #[test]
    fn source_manifest_rejects_ammonia_thermal_target_without_heat_double_count_guard() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_CRACKING_THERMAL_INTEGRATION");
        source["benchmark_targets"][0]["claim_guards"]["no_double_counting_of_recirculated_heat_or_reaction_enthalpy"] =
            serde_json::json!(false);
        let error = audit_source_value(&source, "ammonia-thermal-target.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no_double_counting_of_recirculated_heat")
        );
    }

    #[test]
    fn source_manifest_rejects_ammonia_thermal_target_below_energy_reduction_floor() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_CRACKING_THERMAL_INTEGRATION");
        source["benchmark_targets"][0]["metric_thresholds"]["net_grid_specific_energy_reduction_min_fraction"] =
            serde_json::json!(0.10);
        let error = audit_source_value(&source, "ammonia-thermal-target.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("net_grid_specific_energy_reduction")
        );
    }

    #[test]
    fn source_manifest_rejects_ammonia_thermal_metrics_from_different_operating_points() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_CRACKING_THERMAL_INTEGRATION");
        source["benchmark_targets"][0]["metric_thresholds"]["all_thresholds_must_hold_at_same_operating_point"] =
            serde_json::json!(false);
        let error = audit_source_value(&source, "ammonia-thermal-target.json").unwrap_err();
        assert!(error.to_string().contains("one operating point"));
    }

    #[test]
    fn source_manifest_accepts_biogas_syngas_pareto_target() {
        let source = minimal_benchmark_target_manifest("PLASMA_POWER_TO_X_BIOGAS_REFORMING");
        let summary = audit_source_value(&source, "biogas-target.json").unwrap();
        assert_eq!(summary.benchmark_target_count, 1);
    }

    #[test]
    fn source_manifest_rejects_biogas_target_without_energy_price_scenario() {
        let mut source = minimal_benchmark_target_manifest("PLASMA_POWER_TO_X_BIOGAS_REFORMING");
        source["benchmark_targets"][0]["claim_guards"]["electricity_price_and_capacity_factor_scenario_required"] =
            serde_json::json!(false);
        let error = audit_source_value(&source, "biogas-target.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("electricity_price_and_capacity_factor_scenario_required")
        );
    }

    #[test]
    fn source_manifest_rejects_benchmark_target_promoted_to_result() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_CRACKING");
        source["benchmark_targets"][0]["status"] = serde_json::json!("ACHIEVED");
        let error = audit_source_value(&source, "target.json").unwrap_err();
        assert!(error.to_string().contains("RESEARCH_TARGET_NOT_RESULT"));
    }

    #[test]
    fn source_manifest_rejects_benchmark_target_with_unknown_comparator() {
        let mut source = minimal_benchmark_target_manifest("AMMONIA_CRACKING");
        source["benchmark_targets"][0]["comparator_case_ids"][0] =
            serde_json::json!("unknown-case");
        let error = audit_source_value(&source, "target.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("references unknown comparator case")
        );
    }

    #[test]
    fn source_manifest_rejects_methane_target_without_no_credit_policy() {
        let mut source = minimal_benchmark_target_manifest("METHANE_PYROLYSIS");
        source["benchmark_targets"][0]["credit_policies"] =
            serde_json::json!(["ALLOCATION", "AVOIDED_PRODUCT_CREDIT"]);
        let error = audit_source_value(&source, "target.json").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must include a NO_CREDIT scenario")
        );
    }

    #[test]
    fn source_manifest_rejects_duplicate_case_and_control_ids() {
        let duplicate_case = serde_json::json!({
            "programme": "TEST",
            "claim_ceiling": "SOURCE_ONLY",
            "evidence_classes": ["PUBLISHED_LAB_EXPERIMENT"],
            "cases": [
                {
                    "case_id": "same-case",
                    "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                    "luminous_reproduction": false,
                    "source": {"url": "https://example.org/source-a"}
                },
                {
                    "case_id": "same-case",
                    "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                    "luminous_reproduction": false,
                    "source": {"url": "https://example.org/source-b"}
                }
            ],
            "negative_controls": [{
                "control_id": "nc-1",
                "expected_disposition": "REJECTED"
            }]
        });
        let case_error = audit_source_value(&duplicate_case, "duplicate-case.json").unwrap_err();
        assert!(case_error.to_string().contains("duplicate source case ID"));

        let duplicate_control = serde_json::json!({
            "programme": "TEST",
            "claim_ceiling": "SOURCE_ONLY",
            "evidence_classes": ["PUBLISHED_LAB_EXPERIMENT"],
            "cases": [{
                "case_id": "case-1",
                "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                "luminous_reproduction": false,
                "source": {"url": "https://example.org/source"}
            }],
            "negative_controls": [
                {"control_id": "same-control", "expected_disposition": "REJECTED"},
                {"control_id": "same-control", "expected_disposition": "BLOCKED"}
            ]
        });
        let control_error =
            audit_source_value(&duplicate_control, "duplicate-control.json").unwrap_err();
        assert!(
            control_error
                .to_string()
                .contains("duplicate negative-control ID")
        );
    }

    #[test]
    fn zero_useful_heat_cannot_pass_a_positive_duration_service() {
        let mut scenario = base_scenario();
        scenario.discharge.useful_heat_delivered_j = 0;
        scenario.discharge.discharge_loss_j = scenario.discharge.stored_energy_withdrawn_j;
        assert_eq!(
            evaluate_scenario(&scenario).disposition,
            ScenarioDisposition::Rejected
        );
    }

    #[test]
    fn source_manifest_rejects_http_scheme_without_authority() {
        let source = serde_json::json!({
            "programme": "TEST",
            "claim_ceiling": "SOURCE_ONLY",
            "evidence_classes": ["PUBLISHED_LAB_EXPERIMENT"],
            "cases": [{
                "case_id": "case-1",
                "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                "luminous_reproduction": false,
                "source": {"url": "https://"}
            }],
            "negative_controls": [{
                "control_id": "nc-1",
                "expected_disposition": "REJECTED"
            }]
        });
        let error = audit_source_value(&source, "invalid-source-url.json").unwrap_err();
        assert!(error.to_string().contains("non-empty authority"));
    }

    #[test]
    fn source_manifest_rejects_missing_source_url() {
        let source = serde_json::json!({
            "programme": "TEST",
            "claim_ceiling": "SOURCE_ONLY",
            "evidence_classes": ["PUBLISHED_LAB_EXPERIMENT"],
            "cases": [{
                "case_id": "case-1",
                "evidence_class": "PUBLISHED_LAB_EXPERIMENT",
                "luminous_reproduction": false,
                "source": {}
            }],
            "negative_controls": [{
                "control_id": "nc-1",
                "expected_disposition": "REJECTED"
            }]
        });
        let error = audit_source_value(&source, "test.json").unwrap_err();
        assert!(error.to_string().contains("valid HTTP(S) source URL"));
    }
}