use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CORPUS_VERSION: &str = "ev-mission-capability-corpus/0.1";
const EPS: f64 = 1e-9;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub corpus_version: String,
    pub cases: Vec<MissionCase>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MissionCase {
    pub case_id: String,
    pub expected_disposition: String,
    pub mission: Mission,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Mission {
    pub vehicle_ref: String,
    pub pack_ref: String,
    pub configuration_sha256: String,
    pub topology_sha256: String,
    pub profile_ref: String,
    pub profile_sha256: String,
    pub declared_cell_refs: Vec<String>,
    pub start_time_seconds: f64,
    pub end_time_seconds: f64,
    /// Lower-bound energy available above the required end-of-mission reserve.
    pub initial_energy_above_reserve_kwh_lower: f64,
    pub regenerative_efficiency_lower: f64,
    pub minimum_cell_soc_pct: f64,
    pub maximum_cell_soc_pct: f64,
    pub maximum_cell_temperature_c: f64,
    pub minimum_cell_voltage_v: f64,
    pub maximum_cell_voltage_v: f64,
    pub steps: Vec<MissionStep>,
    pub capability_slices: Vec<CapabilitySlice>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MissionStep {
    pub step_id: String,
    pub start_time_seconds: f64,
    pub end_time_seconds: f64,
    /// Battery-terminal power: positive discharges the pack; negative charges it.
    pub requested_battery_power_kw: f64,
    /// Symmetric power uncertainty. The evaluator uses adverse bounds.
    pub power_uncertainty_kw: f64,
    /// Separately metered external/trolley/charger energy. Never credited to battery energy.
    pub external_energy_kwh: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilitySlice {
    pub step_id: String,
    pub start_time_seconds: f64,
    pub end_time_seconds: f64,
    pub vehicle_ref: String,
    pub pack_ref: String,
    pub configuration_sha256: String,
    pub topology_sha256: String,
    pub model_version: String,
    pub model_evidence_ref: String,
    pub model_evidence_sha256: String,
    pub source_class: String,
    pub domain_soc_min_pct: f64,
    pub domain_soc_max_pct: f64,
    pub domain_temperature_min_c: f64,
    pub domain_temperature_max_c: f64,
    /// Conservative lower bounds supplied by the upstream qualified pack model.
    pub discharge_power_lower_kw: f64,
    pub charge_acceptance_lower_kw: f64,
    pub cells: Vec<CellEnvelope>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CellEnvelope {
    pub cell_ref: String,
    pub soc_lower_pct: f64,
    pub soc_upper_pct: f64,
    pub temperature_upper_c: f64,
    pub voltage_lower_v: f64,
    pub voltage_upper_v: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct StepAudit {
    pub step_id: String,
    pub disposition: String,
    pub duration_seconds: f64,
    pub worst_case_discharge_request_kw: f64,
    pub worst_case_charge_request_kw: f64,
    pub discharge_power_margin_kw: Option<f64>,
    pub charge_acceptance_margin_kw: Option<f64>,
    pub energy_remaining_above_reserve_kwh_lower: Option<f64>,
    pub external_energy_kwh_not_credited: f64,
    pub limiting_cell_ref: Option<String>,
    pub thermal_margin_c: Option<f64>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseAudit {
    pub case_id: String,
    pub disposition: String,
    pub input_errors: Vec<String>,
    pub step_audits: Vec<StepAudit>,
    pub minimum_energy_remaining_above_reserve_kwh_lower: Option<f64>,
    pub total_battery_discharge_energy_kwh_upper: f64,
    pub total_regenerative_energy_credited_kwh_lower: f64,
    pub total_external_energy_kwh_not_credited: f64,
    pub worst_discharge_power_margin_kw: Option<f64>,
    pub worst_charge_acceptance_margin_kw: Option<f64>,
    pub limiting_cell_ref: Option<String>,
    pub source_grade: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditReport {
    pub corpus_version: String,
    pub input_sha256: String,
    pub canonical_corpus_sha256: String,
    pub report_sha256: String,
    pub case_count: usize,
    pub feasible_count: usize,
    pub infeasible_count: usize,
    pub not_computable_count: usize,
    pub unsupported_domain_count: usize,
    pub conflicting_evidence_count: usize,
    pub rejected_input_count: usize,
    pub cases: Vec<CaseAudit>,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn compile_schema(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn validate_schema(validator: &Validator, value: &Value) -> Result<(), String> {
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|error| error.to_string())
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("mission capability schema errors: {}", errors.join("; ")))
    }
}

fn reject_input(case_id: &str, errors: Vec<String>) -> CaseAudit {
    CaseAudit {
        case_id: case_id.to_owned(),
        disposition: "REJECTED_INPUT".to_owned(),
        input_errors: errors,
        step_audits: Vec::new(),
        minimum_energy_remaining_above_reserve_kwh_lower: None,
        total_battery_discharge_energy_kwh_upper: 0.0,
        total_regenerative_energy_credited_kwh_lower: 0.0,
        total_external_energy_kwh_not_credited: 0.0,
        worst_discharge_power_margin_kw: None,
        worst_charge_acceptance_margin_kw: None,
        limiting_cell_ref: None,
        source_grade: "NOT_EVALUATED".to_owned(),
    }
}

fn structurally_validate(mission: &Mission) -> Vec<String> {
    let mut errors = Vec::new();
    if !nonempty(&mission.vehicle_ref)
        || !nonempty(&mission.pack_ref)
        || !nonempty(&mission.profile_ref)
        || !is_sha256(&mission.configuration_sha256)
        || !is_sha256(&mission.topology_sha256)
        || !is_sha256(&mission.profile_sha256)
    {
        errors.push("MISSION_IDENTITY_OR_DIGEST_INVALID".to_owned());
    }
    for value in [
        mission.start_time_seconds,
        mission.end_time_seconds,
        mission.initial_energy_above_reserve_kwh_lower,
        mission.regenerative_efficiency_lower,
        mission.minimum_cell_soc_pct,
        mission.maximum_cell_soc_pct,
        mission.maximum_cell_temperature_c,
        mission.minimum_cell_voltage_v,
        mission.maximum_cell_voltage_v,
    ] {
        if !value.is_finite() {
            errors.push("MISSION_NONFINITE_PARAMETER".to_owned());
            break;
        }
    }
    if mission.end_time_seconds <= mission.start_time_seconds
        || !finite_nonnegative(mission.initial_energy_above_reserve_kwh_lower)
        || !(0.0..=1.0).contains(&mission.regenerative_efficiency_lower)
        || !(0.0..=100.0).contains(&mission.minimum_cell_soc_pct)
        || !(mission.minimum_cell_soc_pct..=100.0).contains(&mission.maximum_cell_soc_pct)
        || !finite_nonnegative(mission.maximum_cell_temperature_c)
        || mission.minimum_cell_voltage_v <= 0.0
        || mission.maximum_cell_voltage_v <= mission.minimum_cell_voltage_v
    {
        errors.push("MISSION_LIMIT_OR_HORIZON_INVALID".to_owned());
    }
    if mission.declared_cell_refs.is_empty() {
        errors.push("MISSION_HAS_NO_DECLARED_CELL_POPULATION".to_owned());
    }
    let mut declared = BTreeSet::new();
    for cell in &mission.declared_cell_refs {
        if !nonempty(cell) || !declared.insert(cell.clone()) {
            errors.push("DECLARED_CELL_IDENTITY_EMPTY_OR_DUPLICATED".to_owned());
            break;
        }
    }
    if mission.steps.is_empty() {
        errors.push("MISSION_HAS_NO_STEPS".to_owned());
    }
    let mut step_ids = BTreeSet::new();
    let mut prior_end = mission.start_time_seconds;
    for step in &mission.steps {
        if !nonempty(&step.step_id) || !step_ids.insert(step.step_id.clone()) {
            errors.push(format!("STEP_ID_EMPTY_OR_DUPLICATED:{}", step.step_id));
        }
        if !step.start_time_seconds.is_finite()
            || !step.end_time_seconds.is_finite()
            || !step.requested_battery_power_kw.is_finite()
            || !finite_nonnegative(step.power_uncertainty_kw)
            || !finite_nonnegative(step.external_energy_kwh)
            || step.end_time_seconds <= step.start_time_seconds
            || (step.start_time_seconds - prior_end).abs() > EPS
        {
            errors.push(format!("STEP_TIME_POWER_OR_UNITS_INVALID:{}", step.step_id));
        }
        prior_end = step.end_time_seconds;
    }
    if (mission.steps.first().map(|step| step.start_time_seconds).unwrap_or(f64::NAN)
        - mission.start_time_seconds)
        .abs()
        > EPS
        || (mission.steps.last().map(|step| step.end_time_seconds).unwrap_or(f64::NAN)
            - mission.end_time_seconds)
            .abs()
            > EPS
    {
        errors.push("MISSION_STEP_COVERAGE_DOES_NOT_MATCH_HORIZON".to_owned());
    }
    let mut slice_ids = BTreeSet::new();
    for slice in &mission.capability_slices {
        if !slice_ids.insert(slice.step_id.clone()) {
            errors.push(format!("DUPLICATE_CAPABILITY_SLICE:{}", slice.step_id));
        }
        if !step_ids.contains(&slice.step_id) {
            errors.push(format!("CAPABILITY_SLICE_REFERENCES_UNKNOWN_STEP:{}", slice.step_id));
        }
    }
    errors
}

fn evaluate_case(case: &MissionCase) -> CaseAudit {
    let mission = &case.mission;
    let errors = structurally_validate(mission);
    if !errors.is_empty() {
        return reject_input(&case.case_id, errors);
    }

remove unused step map    let slices_by_id: BTreeMap<&str, &CapabilitySlice> = mission
        .capability_slices
        .iter()
        .map(|slice| (slice.step_id.as_str(), slice))
        .collect();

    let mut disposition = "FEASIBLE_WITHIN_DECLARED_ENVELOPE".to_owned();
    let mut step_audits = Vec::new();
    let mut energy_remaining = mission.initial_energy_above_reserve_kwh_lower;
    let mut minimum_energy_remaining = energy_remaining;
    let mut total_discharge = 0.0;
    let mut total_regen_credit = 0.0;
    let mut total_external = 0.0;
    let mut worst_discharge_margin: Option<f64> = None;
    let mut worst_charge_margin: Option<f64> = None;
    let mut global_limiting_cell: Option<String> = None;
    let mut source_grade = "SYNTHETIC_KNOWN_TRUTH".to_owned();

    for step in &mission.steps {
        total_external += step.external_energy_kwh;
        let Some(slice) = slices_by_id.get(step.step_id.as_str()).copied() else {
            disposition = "NOT_COMPUTABLE".to_owned();
            step_audits.push(StepAudit {
                step_id: step.step_id.clone(),
                disposition: disposition.clone(),
                duration_seconds: step.end_time_seconds - step.start_time_seconds,
                worst_case_discharge_request_kw: (step.requested_battery_power_kw
                    + step.power_uncertainty_kw)
                    .max(0.0),
                worst_case_charge_request_kw: (-step.requested_battery_power_kw
                    + step.power_uncertainty_kw)
                    .max(0.0),
                discharge_power_margin_kw: None,
                charge_acceptance_margin_kw: None,
                energy_remaining_above_reserve_kwh_lower: None,
                external_energy_kwh_not_credited: step.external_energy_kwh,
                limiting_cell_ref: None,
                thermal_margin_c: None,
                reason: Some("No capability slice covers this exact mission interval".to_owned()),
            });
            continue;
        };
        let duration = step.end_time_seconds - step.start_time_seconds;
        if (slice.start_time_seconds - step.start_time_seconds).abs() > EPS
            || (slice.end_time_seconds - step.end_time_seconds).abs() > EPS
        {
            disposition = "CONFLICTING_EVIDENCE".to_owned();
            step_audits.push(StepAudit {
                step_id: step.step_id.clone(),
                disposition: disposition.clone(),
                duration_seconds: duration,
                worst_case_discharge_request_kw: 0.0,
                worst_case_charge_request_kw: 0.0,
                discharge_power_margin_kw: None,
                charge_acceptance_margin_kw: None,
                energy_remaining_above_reserve_kwh_lower: None,
                external_energy_kwh_not_credited: step.external_energy_kwh,
                limiting_cell_ref: None,
                thermal_margin_c: None,
                reason: Some("Capability slice time bounds do not exactly match the mission step".to_owned()),
            });
            continue;
        }
        if slice.vehicle_ref != mission.vehicle_ref
            || slice.pack_ref != mission.pack_ref
            || slice.configuration_sha256 != mission.configuration_sha256
            || slice.topology_sha256 != mission.topology_sha256
            || !is_sha256(&slice.configuration_sha256)
            || !is_sha256(&slice.topology_sha256)
            || !is_sha256(&slice.model_evidence_sha256)
            || !nonempty(&slice.model_version)
            || !nonempty(&slice.model_evidence_ref)
        {
            disposition = "CONFLICTING_EVIDENCE".to_owned();
            step_audits.push(StepAudit {
                step_id: step.step_id.clone(),
                disposition: disposition.clone(),
                duration_seconds: duration,
                worst_case_discharge_request_kw: 0.0,
                worst_case_charge_request_kw: 0.0,
                discharge_power_margin_kw: None,
                charge_acceptance_margin_kw: None,
                energy_remaining_above_reserve_kwh_lower: None,
                external_energy_kwh_not_credited: step.external_energy_kwh,
                limiting_cell_ref: None,
                thermal_margin_c: None,
                reason: Some("Capability slice asset/configuration/topology/model evidence does not bind to this mission".to_owned()),
            });
            continue;
        }
        if slice.source_class != "SYNTHETIC_KNOWN_TRUTH"
            && slice.source_class != "QUALIFIED_PHYSICAL_MODEL"
        {
            disposition = "UNSUPPORTED_DOMAIN".to_owned();
            step_audits.push(StepAudit {
                step_id: step.step_id.clone(),
                disposition: disposition.clone(),
                duration_seconds: duration,
                worst_case_discharge_request_kw: 0.0,
                worst_case_charge_request_kw: 0.0,
                discharge_power_margin_kw: None,
                charge_acceptance_margin_kw: None,
                energy_remaining_above_reserve_kwh_lower: None,
                external_energy_kwh_not_credited: step.external_energy_kwh,
                limiting_cell_ref: None,
                thermal_margin_c: None,
                reason: Some("Capability source class is not admitted".to_owned()),
            });
            continue;
        }
        if slice.source_class == "QUALIFIED_PHYSICAL_MODEL" {
            source_grade = "QUALIFIED_MODEL_INPUTS_NOT_INDEPENDENTLY_REVALIDATED_HERE".to_owned();
        }
        if !slice.domain_soc_min_pct.is_finite()
            || !slice.domain_soc_max_pct.is_finite()
            || !slice.domain_temperature_min_c.is_finite()
            || !slice.domain_temperature_max_c.is_finite()
            || slice.domain_soc_min_pct < 0.0
            || slice.domain_soc_max_pct > 100.0
            || slice.domain_soc_min_pct > slice.domain_soc_max_pct
            || slice.domain_temperature_min_c > slice.domain_temperature_max_c
            || !finite_nonnegative(slice.discharge_power_lower_kw)
            || !finite_nonnegative(slice.charge_acceptance_lower_kw)
        {
            disposition = "REJECTED_INPUT".to_owned();
            step_audits.push(StepAudit {
                step_id: step.step_id.clone(),
                disposition: disposition.clone(),
                duration_seconds: duration,
                worst_case_discharge_request_kw: 0.0,
                worst_case_charge_request_kw: 0.0,
                discharge_power_margin_kw: None,
                charge_acceptance_margin_kw: None,
                energy_remaining_above_reserve_kwh_lower: None,
                external_energy_kwh_not_credited: step.external_energy_kwh,
                limiting_cell_ref: None,
                thermal_margin_c: None,
                reason: Some("Capability envelope has invalid domains or bounds".to_owned()),
            });
            continue;
        }

        let actual_cells: BTreeSet<String> =
            slice.cells.iter().map(|cell| cell.cell_ref.clone()).collect();
        let expected_cells: BTreeSet<String> =
            mission.declared_cell_refs.iter().cloned().collect();
        if actual_cells.len() != slice.cells.len()
            || slice.cells.iter().any(|cell| !expected_cells.contains(&cell.cell_ref))
        {
            disposition = "CONFLICTING_EVIDENCE".to_owned();
            step_audits.push(StepAudit {
                step_id: step.step_id.clone(),
                disposition: disposition.clone(),
                duration_seconds: duration,
                worst_case_discharge_request_kw: 0.0,
                worst_case_charge_request_kw: 0.0,
                discharge_power_margin_kw: None,
                charge_acceptance_margin_kw: None,
                energy_remaining_above_reserve_kwh_lower: None,
                external_energy_kwh_not_credited: step.external_energy_kwh,
                limiting_cell_ref: None,
                thermal_margin_c: None,
                reason: Some("Capability slice contains duplicate or undeclared cell identities".to_owned()),
            });
            continue;
        }
        if actual_cells != expected_cells {
            if disposition == "FEASIBLE_WITHIN_DECLARED_ENVELOPE" {
                disposition = "NOT_COMPUTABLE".to_owned();
            }
            step_audits.push(StepAudit {
                step_id: step.step_id.clone(),
                disposition: "NOT_COMPUTABLE".to_owned(),
                duration_seconds: duration,
                worst_case_discharge_request_kw: 0.0,
                worst_case_charge_request_kw: 0.0,
                discharge_power_margin_kw: None,
                charge_acceptance_margin_kw: None,
                energy_remaining_above_reserve_kwh_lower: None,
                external_energy_kwh_not_credited: step.external_energy_kwh,
                limiting_cell_ref: None,
                thermal_margin_c: None,
                reason: Some("The declared cell population is not completely covered".to_owned()),
            });
            continue;
        }

        let mut cell_problem: Option<(String, String, f64)> = None;
        for cell in &slice.cells {
            if ![
                cell.soc_lower_pct,
                cell.soc_upper_pct,
                cell.temperature_upper_c,
                cell.voltage_lower_v,
                cell.voltage_upper_v,
            ]
            .iter()
            .all(|value| value.is_finite())
            || cell.soc_lower_pct < 0.0
            || cell.soc_upper_pct > 100.0
            || cell.soc_lower_pct > cell.soc_upper_pct
            || cell.voltage_lower_v > cell.voltage_upper_v
        {
                cell_problem = Some((
                    cell.cell_ref.clone(),
                    "REJECTED_INPUT".to_owned(),
                    0.0,
                ));
                break;
            }
            if cell.soc_lower_pct < slice.domain_soc_min_pct - EPS
                || cell.soc_upper_pct > slice.domain_soc_max_pct + EPS
                || cell.temperature_upper_c < slice.domain_temperature_min_c - EPS
                || cell.temperature_upper_c > slice.domain_temperature_max_c + EPS
            {
                cell_problem = Some((
                    cell.cell_ref.clone(),
                    "UNSUPPORTED_DOMAIN".to_owned(),
                    0.0,
                ));
                break;
            }
            if cell.soc_lower_pct < mission.minimum_cell_soc_pct - EPS
                || cell.soc_upper_pct > mission.maximum_cell_soc_pct + EPS
                || cell.temperature_upper_c > mission.maximum_cell_temperature_c + EPS
                || cell.voltage_lower_v < mission.minimum_cell_voltage_v - EPS
                || cell.voltage_upper_v > mission.maximum_cell_voltage_v + EPS
            {
                let thermal_margin = mission.maximum_cell_temperature_c - cell.temperature_upper_c;
                cell_problem = Some((
                    cell.cell_ref.clone(),
                    "INFEASIBLE".to_owned(),
                    thermal_margin,
                ));
                break;
            }
        }
        if let Some((cell_ref, cell_disposition, thermal_margin)) = cell_problem {
            if cell_disposition == "INFEASIBLE" {
                disposition = "INFEASIBLE".to_owned();
            } else if cell_disposition == "UNSUPPORTED_DOMAIN"
                && disposition == "FEASIBLE_WITHIN_DECLARED_ENVELOPE"
            {
                disposition = "UNSUPPORTED_DOMAIN".to_owned();
            } else if cell_disposition == "REJECTED_INPUT" {
                disposition = "REJECTED_INPUT".to_owned();
            }
            if global_limiting_cell.is_none() || cell_disposition == "INFEASIBLE" {
                global_limiting_cell = Some(cell_ref.clone());
            }
            step_audits.push(StepAudit {
                step_id: step.step_id.clone(),
                disposition: cell_disposition.clone(),
                duration_seconds: duration,
                worst_case_discharge_request_kw: (step.requested_battery_power_kw
                    + step.power_uncertainty_kw)
                    .max(0.0),
                worst_case_charge_request_kw: (-step.requested_battery_power_kw
                    + step.power_uncertainty_kw)
                    .max(0.0),
                discharge_power_margin_kw: None,
                charge_acceptance_margin_kw: None,
                energy_remaining_above_reserve_kwh_lower: None,
                external_energy_kwh_not_credited: step.external_energy_kwh,
                limiting_cell_ref: Some(cell_ref),
                thermal_margin_c: Some(thermal_margin),
                reason: Some("A cell trajectory violates the mission limit or the model support domain".to_owned()),
            });
            continue;
        }

        let lower_power = step.requested_battery_power_kw - step.power_uncertainty_kw;
        let upper_power = step.requested_battery_power_kw + step.power_uncertainty_kw;
        let worst_discharge = upper_power.max(0.0);
        let worst_charge = (-lower_power).max(0.0);
        let guaranteed_regen = (-upper_power).max(0.0);
        let discharge_margin = slice.discharge_power_lower_kw - worst_discharge;
        let charge_margin = slice.charge_acceptance_lower_kw - worst_charge;
        worst_discharge_margin = Some(
            worst_discharge_margin.map_or(discharge_margin, |current| current.min(discharge_margin)),
        );
        worst_charge_margin = Some(
            worst_charge_margin.map_or(charge_margin, |current| current.min(charge_margin)),
        );

        let mut step_disposition = "FEASIBLE_WITHIN_DECLARED_ENVELOPE".to_owned();
        let mut reason = None;
        if discharge_margin < -EPS {
            step_disposition = "INFEASIBLE".to_owned();
            reason = Some("Worst-case discharge request exceeds conservative time-indexed pack power".to_owned());
        }
        if charge_margin < -EPS {
            step_disposition = "INFEASIBLE".to_owned();
            reason = Some("Worst-case regenerative/charge request exceeds conservative pack charge acceptance".to_owned());
        }
        let energy_out = worst_discharge * duration / 3600.0;
        total_discharge += energy_out;
        energy_remaining -= energy_out;
        let energy_in = guaranteed_regen
            * mission.regenerative_efficiency_lower
            * duration
            / 3600.0;
        total_regen_credit += energy_in;
        energy_remaining += energy_in;
        if energy_remaining < -EPS {
            step_disposition = "INFEASIBLE".to_owned();
            reason = Some("Cumulative mission energy crosses the required end-of-mission reserve".to_owned());
        }
        minimum_energy_remaining = minimum_energy_remaining.min(energy_remaining);
        if step_disposition == "INFEASIBLE" {
            disposition = "INFEASIBLE".to_owned();
            if global_limiting_cell.is_none() {
                global_limiting_cell = slice.cells.iter()
                    .max_by(|left, right| {
                        left.temperature_upper_c.partial_cmp(&right.temperature_upper_c)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|cell| cell.cell_ref.clone());
            }
        }
        let max_cell_temp = slice.cells.iter()
            .max_by(|left, right| left.temperature_upper_c.partial_cmp(&right.temperature_upper_c)
                .unwrap_or(std::cmp::Ordering::Equal));
        let thermal_margin = max_cell_temp.map(|cell| {
            mission.maximum_cell_temperature_c - cell.temperature_upper_c
        });
        let limiting_cell = max_cell_temp.map(|cell| cell.cell_ref.clone());
        if let Some(ref cell) = limiting_cell {
            if thermal_margin.is_some_and(|margin| margin < 0.0) {
                global_limiting_cell = Some(cell.clone());
            }
        }
        step_audits.push(StepAudit {
            step_id: step.step_id.clone(),
            disposition: step_disposition,
            duration_seconds: duration,
            worst_case_discharge_request_kw: worst_discharge,
            worst_case_charge_request_kw: worst_charge,
            discharge_power_margin_kw: Some(discharge_margin),
            charge_acceptance_margin_kw: Some(charge_margin),
            energy_remaining_above_reserve_kwh_lower: Some(energy_remaining),
            external_energy_kwh_not_credited: step.external_energy_kwh,
            limiting_cell_ref: limiting_cell,
            thermal_margin_c: thermal_margin,
            reason,
        });
    }

    if disposition == "FEASIBLE_WITHIN_DECLARED_ENVELOPE"
        && energy_remaining >= -EPS
        && step_audits.len() == mission.steps.len()
        && step_audits.iter().all(|step| {
            step.disposition == "FEASIBLE_WITHIN_DECLARED_ENVELOPE"
        })
    {
        disposition = "FEASIBLE_WITHIN_DECLARED_ENVELOPE".to_owned();
    } else if disposition == "FEASIBLE_WITHIN_DECLARED_ENVELOPE" {
        disposition = "NOT_COMPUTABLE".to_owned();
    }

    CaseAudit {
        case_id: case.case_id.clone(),
        disposition,
        input_errors: Vec::new(),
        step_audits,
        minimum_energy_remaining_above_reserve_kwh_lower: Some(minimum_energy_remaining),
        total_battery_discharge_energy_kwh_upper: total_discharge,
        total_regenerative_energy_credited_kwh_lower: total_regen_credit,
        total_external_energy_kwh_not_credited: total_external,
        worst_discharge_power_margin_kw: worst_discharge_margin,
        worst_charge_acceptance_margin_kw: worst_charge_margin,
        limiting_cell_ref: global_limiting_cell,
        source_grade,
    }
}

fn canonicalize(corpus: &mut Corpus) {
    corpus.cases.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    for case in &mut corpus.cases {
        case.mission.declared_cell_refs.sort();
        case.mission.capability_slices.sort_by(|left, right| left.step_id.cmp(&right.step_id));
        for slice in &mut case.mission.capability_slices {
            slice.cells.sort_by(|left, right| left.cell_ref.cmp(&right.cell_ref));
        }
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<AuditReport, String> {
    let validator = compile_schema(schema_bytes)?;
    let value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    validate_schema(&validator, &value)?;
    let mut corpus: Corpus =
        serde_json::from_value(value).map_err(|error| format!("corpus decode: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.cases.is_empty() {
        return Err("unsupported or empty mission capability corpus".to_owned());
    }
    canonicalize(&mut corpus);
    let canonical = serde_json::to_vec(&corpus)
        .map_err(|error| format!("canonical corpus: {error}"))?;
    let mut case_ids = BTreeSet::new();
    let mut audits = Vec::new();
    for case in &corpus.cases {
        if !case_ids.insert(case.case_id.clone()) {
            return Err(format!("duplicate mission case identity: {}", case.case_id));
        }
        let audit = evaluate_case(case);
        if case.expected_disposition != audit.disposition {
            return Err(format!(
                "mission fixture expectation mismatch for {}: got {}, expected {}",
                case.case_id, audit.disposition, case.expected_disposition
            ));
        }
        audits.push(audit);
    }
    let mut report = AuditReport {
        corpus_version: corpus.corpus_version,
        input_sha256: sha256(corpus_bytes),
        canonical_corpus_sha256: sha256(&canonical),
        report_sha256: String::new(),
        case_count: audits.len(),
        feasible_count: audits.iter().filter(|row| row.disposition == "FEASIBLE_WITHIN_DECLARED_ENVELOPE").count(),
        infeasible_count: audits.iter().filter(|row| row.disposition == "INFEASIBLE").count(),
        not_computable_count: audits.iter().filter(|row| row.disposition == "NOT_COMPUTABLE").count(),
        unsupported_domain_count: audits.iter().filter(|row| row.disposition == "UNSUPPORTED_DOMAIN").count(),
        conflicting_evidence_count: audits.iter().filter(|row| row.disposition == "CONFLICTING_EVIDENCE").count(),
        rejected_input_count: audits.iter().filter(|row| row.disposition == "REJECTED_INPUT").count(),
        cases: audits,
    };
    let bytes = serde_json::to_vec(&report)
        .map_err(|error| format!("report serialization: {error}"))?;
    report.report_sha256 = sha256(&bytes);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_mission_capability_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/mission_capability_v0_1.json");

    fn report() -> AuditReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen mission capability corpus")
    }

    fn case<'a>(report: &'a AuditReport, id: &str) -> &'a CaseAudit {
        report.cases.iter().find(|row| row.case_id == id).expect("case audit")
    }

    #[test]
    fn frozen_corpus_replays_expected_mission_dispositions() {
        let report = report();
        assert_eq!(report.case_count, 10);
        assert_eq!(report.feasible_count, 3);
        assert_eq!(report.infeasible_count, 3);
        assert_eq!(report.not_computable_count, 1);
        assert_eq!(report.unsupported_domain_count, 1);
        assert_eq!(report.conflicting_evidence_count, 1);
        assert_eq!(report.rejected_input_count, 1);
    }

    #[test]
    fn sustained_mission_is_not_approved_by_a_peak_only_power_number() {
        let report = report();
        let row = case(&report, "peak_passes_but_sustained_energy_fails");
        assert_eq!(row.disposition, "INFEASIBLE");
        assert!(row.minimum_energy_remaining_above_reserve_kwh_lower.unwrap() < 0.0);
    }

    #[test]
    fn regenerative_acceptance_is_checked_separately_from_discharge_power() {
        let report = report();
        let row = case(&report, "regen_exceeds_charge_acceptance");
        assert_eq!(row.disposition, "INFEASIBLE");
        assert!(row.worst_charge_acceptance_margin_kw.unwrap() < 0.0);
    }

    #[test]
    fn external_trolley_energy_is_reported_but_never_credited_to_pack_reserve() {
        let report = report();
        let row = case(&report, "external_energy_not_double_counted");
        assert_eq!(row.disposition, "FEASIBLE_WITHIN_DECLARED_ENVELOPE");
        assert_eq!(row.total_external_energy_kwh_not_credited, 10.0);
        assert_eq!(row.total_battery_discharge_energy_kwh_upper, 1.0);
    }

    #[test]
    fn incomplete_cell_population_is_not_computable() {
        let report = report();
        assert_eq!(case(&report, "missing_cell_coverage").disposition, "NOT_COMPUTABLE");
    }

    #[test]
    fn model_domain_mismatch_is_not_treated_as_feasible() {
        let report = report();
        assert_eq!(case(&report, "temperature_out_of_model_domain").disposition, "UNSUPPORTED_DOMAIN");
    }

    #[test]
    fn configuration_mismatch_is_conflicting_evidence() {
        let report = report();
        assert_eq!(case(&report, "wrong_configuration").disposition, "CONFLICTING_EVIDENCE");
    }

    #[test]
    fn thermal_accumulation_is_checked_over_the_mission_not_pulse_by_pulse() {
        let report = report();
        assert_eq!(case(&report, "thermal_limit_crossed_in_later_segment").disposition, "INFEASIBLE");
    }

    #[test]
    fn mission_and_capability_array_order_is_canonical_but_time_order_is_not() {
        let original = report();
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        corpus.cases.reverse();
        for item in &mut corpus.cases {
            item.mission.declared_cell_refs.reverse();
            item.mission.capability_slices.reverse();
            for slice in &mut item.mission.capability_slices {
                slice.cells.reverse();
            }
        }
        let bytes = serde_json::to_vec(&corpus).expect("permuted corpus");
        let replay = audit_corpus(SCHEMA, &bytes).expect("permuted audit");
        assert_eq!(original.canonical_corpus_sha256, replay.canonical_corpus_sha256);
        assert_eq!(
            serde_json::to_value(&original.cases).unwrap(),
            serde_json::to_value(&replay.cases).unwrap()
        );
    }

    #[test]
    fn noncontiguous_mission_intervals_are_rejected() {
        let mut corpus: Value = serde_json::from_slice(CORPUS).expect("corpus");
        corpus["cases"][0]["mission"]["steps"][1]["start_time_seconds"] = Value::from(999.0);
        corpus["cases"][0]["expected_disposition"] = Value::String("REJECTED_INPUT".into());
        let bytes = serde_json::to_vec(&corpus).expect("mutated corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("expected rejection");
        assert_eq!(report.rejected_input_count, 2);
    }

    #[test]
    fn external_energy_cannot_hide_a_pack_reserve_shortfall() {
        let mut corpus: Value = serde_json::from_slice(CORPUS).expect("corpus");
        let item = corpus["cases"].as_array_mut().unwrap()
            .iter_mut().find(|item| item["case_id"] == "peak_passes_but_sustained_energy_fails")
            .unwrap();
        item["mission"]["steps"][0]["external_energy_kwh"] = Value::from(500.0);
        let bytes = serde_json::to_vec(&corpus).expect("mutated corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("expected still infeasible");
        assert_eq!(case(&report, "peak_passes_but_sustained_energy_fails").disposition, "INFEASIBLE");
    }

    #[test]
    fn cell_voltage_margin_is_a_hard_constraint() {
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        let item = corpus.cases.iter_mut()
            .find(|item| item.case_id == "feasible_mixed_mission")
            .unwrap();
        item.mission.capability_slices[0].cells[0].voltage_lower_v = 2.0;
        item.expected_disposition = "INFEASIBLE".to_owned();
        let bytes = serde_json::to_vec(&corpus).expect("mutated corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("voltage limit violation");
        assert_eq!(case(&report, "feasible_mixed_mission").disposition, "INFEASIBLE");
    }

    #[test]
    fn reserve_exact_boundary_is_feasible() {
        let report = report();
        let row = case(&report, "exact_reserve_boundary");
        assert_eq!(row.disposition, "FEASIBLE_WITHIN_DECLARED_ENVELOPE");
        assert!((row.minimum_energy_remaining_above_reserve_kwh_lower.unwrap()).abs() < EPS);
    }
}
