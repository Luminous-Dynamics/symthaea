// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline, fail-closed audit for plasma-energy source registers and synthetic
//! high-temperature thermal-storage energy-flow fixtures.
//!
//! This module does not simulate plasma, heat transfer, a thermal store, or a
//! physical plant. It validates source metadata and declared accounting
//! identities. A balanced synthetic scenario is not a physical observation.

use anyhow::{Context, Result, bail, ensure};
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ScenarioDisposition {
    Pass,
    Blocked,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ComparisonDisposition {
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
        let result = evaluate_comparison(
            comparison,
            &scenario_sources,
            &scenario_results,
        );
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
    let bytes = fs::read(path).with_context(|| format!("read source manifest {}", path.display()))?;
    let root: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse source manifest {}", path.display()))?;

    let programme = root
        .get("programme")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .context("source manifest has no non-empty programme")?
        .to_owned();
    let claim_ceiling = root
        .get("claim_ceiling")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .context("source manifest has no explicit claim ceiling")?;
    ensure!(
        !claim_ceiling.trim().is_empty(),
        "source manifest claim ceiling must not be empty"
    );

    let evidence_classes = root
        .get("evidence_classes")
        .and_then(Value::as_array)
        .context("source manifest has no evidence_classes array")?;
    let mut admitted_classes = BTreeSet::new();
    for class in evidence_classes {
        let class = class
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .context("source manifest contains an empty/non-string evidence class")?;
        ensure!(
            admitted_classes.insert(class.to_owned()),
            "duplicate declared evidence class: {class}"
        );
    }
    ensure!(
        !admitted_classes.is_empty(),
        "source manifest has no declared evidence classes"
    );

    let cases = root
        .get("cases")
        .and_then(Value::as_array)
        .context("source manifest has no cases array")?;
    ensure!(!cases.is_empty(), "source manifest has no source cases");

    let mut case_ids = BTreeSet::new();
    for case in cases {
        let case_id = case
            .get("case_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .context("source case has no non-empty case_id")?;
        ensure!(
            case_ids.insert(case_id.to_owned()),
            "duplicate source case ID: {case_id}"
        );

        let evidence_class = case
            .get("evidence_class")
            .and_then(Value::as_str)
            .context("source case has no evidence_class")?;
        ensure!(
            admitted_classes.contains(evidence_class),
            "source case {case_id} uses undeclared evidence class {evidence_class}"
        );

        ensure!(
            case.get("luminous_reproduction").and_then(Value::as_bool) == Some(false),
            "source case {case_id} does not explicitly preserve luminous_reproduction=false"
        );

        let source_url = case
            .get("source")
            .and_then(|source| source.get("url"))
            .and_then(Value::as_str)
            .filter(|value| {
                value.starts_with("https://") || value.starts_with("http://")
            })
            .with_context(|| format!("source case {case_id} has no valid HTTP(S) source URL"))?;
        ensure!(
            !source_url.trim().is_empty(),
            "source case {case_id} source URL is empty"
        );
    }

    let controls = root
        .get("negative_controls")
        .and_then(Value::as_array)
        .context("source manifest has no negative_controls array")?;
    ensure!(!controls.is_empty(), "source manifest has no negative controls");
    let mut control_ids = BTreeSet::new();
    for control in controls {
        let control_id = control
            .get("control_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .context("negative control has no non-empty control_id")?;
        ensure!(
            control_ids.insert(control_id.to_owned()),
            "duplicate negative-control ID: {control_id}"
        );
        ensure!(
            control.get("expected_disposition").and_then(Value::as_str).is_some(),
            "negative control {control_id} has no expected_disposition"
        );
    }

    Ok(SourceManifestSummary {
        path: path.display().to_string(),
        programme,
        case_count: cases.len(),
        negative_control_count: controls.len(),
    })
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
    ensure!(!fixture.scenarios.is_empty(), "thermal fixture has no scenarios");
    ensure!(
        !fixture.comparisons.is_empty(),
        "thermal fixture has no comparison controls"
    );
    Ok(())
}

fn evaluate_scenario(scenario: &ThermalScenario) -> ScenarioResult {
    let reject = |reason: &str| scenario_result(scenario, ScenarioDisposition::Rejected, reason, None, None, None);
    let blocked = |reason: &str| scenario_result(scenario, ScenarioDisposition::Blocked, reason, None, None, None);

    if scenario.scenario_id.trim().is_empty() || scenario.technology.trim().is_empty() {
        return reject("scenario and technology identities must be non-empty");
    }
    if scenario.evidence_class != SYNTHETIC_EVIDENCE_CLASS {
        return reject("fixture scenario evidence class must remain synthetic");
    }
    if scenario.energy_unit != EXPECTED_ENERGY_UNIT {
        return reject("scenario energy unit must be joules (J)");
    }

    if scenario.service.service_contract_id.trim().is_empty()
        || scenario.service.service_type.trim().is_empty()
        || scenario.service.outlet_temperature_mk.is_none_or(|value| value == 0)
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
            return reject("primary-process input does not close to stored-energy increase plus charge loss");
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
            return reject("exclusive total meter does not close to charge-stage stored energy and losses");
        }
    }

    let Some(hold_reconstructed) = hold
        .stored_energy_after_hold_j
        .checked_add(hold.storage_hold_loss_j)
    else {
        return reject("hold-stage energy sum overflow");
    };
    if hold_reconstructed != charge.stored_energy_increase_j {
        return reject("charge-end stored energy does not close to post-hold energy plus hold loss");
    }
    if hold.hold_duration_s > scenario.service.delivery_duration_s.saturating_mul(86_400) {
        return reject("hold duration exceeds representable scenario horizon");
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
        return reject("post-hold energy does not close to useful delivered heat plus discharge loss");
    }

    let Some(all_losses_j) = charge
        .charge_conversion_loss_j
        .checked_add(charge.auxiliary_dissipation_j)
        .and_then(|value| value.checked_add(hold.storage_hold_loss_j))
        .and_then(|value| value.checked_add(discharge.discharge_loss_j))
    else {
        return reject("end-to-end loss sum overflow");
    };
    let Some(end_to_end_reconstructed) = discharge
        .useful_heat_delivered_j
        .checked_add(all_losses_j)
    else {
        return reject("end-to-end energy reconstruction overflow");
    };
    if end_to_end_reconstructed != total_input_j {
        return reject("end-to-end wall-plug input does not close to useful heat plus all declared losses");
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
    {
        return comparison_result(
            comparison,
            ComparisonDisposition::NotComparable,
            "functional service contract or one or more service-boundary fields differ",
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
            "SOURCE PASS {} cases={} negative_controls={} path={}",
            source.programme, source.case_count, source.negative_control_count, source.path
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

#[cfg(test)]
mod tests {
    use super::*;

    fn base_scenario() -> ThermalScenario {
        ThermalScenario {
            scenario_id: "resistance".to_owned(),
            technology: "RESISTANCE_JOULE_HEATER".to_owned(),
            evidence_class: SYNTHETIC_EVIDENCE_CLASS.to_owned(),
            energy_unit: "J".to_owned(),
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
    fn source_manifest_rejects_luminous_reproduction_promotion() {
        let source = serde_json::json!({
            "programme": "TEST",
            "claim_ceiling": "SOURCE_ONLY",
            "evidence_classes": ["PUBLISHED_LAB_EXPERIMENT"],
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
            "evidence_classes": ["PUBLISHED_LAB_EXPERIMENT"],
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
            .filter(|url| url.starts_with("https://") || url.starts_with("http://"))
            .with_context(|| format!("{display_name}: case {case_id} has no valid HTTP(S) source URL"))?;
        ensure!(!source_url.trim().is_empty(), "{display_name}: case {case_id} has empty source URL");
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
            control.get("expected_disposition").and_then(Value::as_str).is_some(),
            "{display_name}: negative control {control_id} has no expected disposition"
        );
    }

    Ok(SourceManifestSummary {
        path: display_name.to_owned(),
        programme,
        case_count: cases.len(),
        negative_control_count: controls.len(),
    })
}

