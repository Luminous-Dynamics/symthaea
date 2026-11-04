#![forbid(unsafe_code)]

//! Strict, offline integrity checks for the **synthetic-only** household-service
//! benchmark fixtures.
//!
//! This crate validates that versioned scenario definitions retain their declared
//! scope, required hostile cases, dispositions and fail-closed invariants. It does
//! not execute a water system, repair procedure, appliance controller or physical
//! qualification. A fixture passing this validator means only that the synthetic
//! corpus is structurally and semantically intact.

use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const FIXTURE_VERSION: &str = "0.1";
const FIXTURE_STATUS: &str = "synthetic_test_fixture_only";
const WATER_CORPUS_ID: &str = "HOME-UTILITY-COORD-001-WATER-CONTINUITY-SYNTHETIC-001";
const REPAIR_CORPUS_ID: &str = "HOME-REPAIR-QUAL-001-SYNTHETIC-REPAIR-TASKS-001";

const WATER_SCENARIOS: [(&str, &str); 10] = [
    ("WCONT-001", "process_accounting_only"),
    ("WCONT-002", "reserve_estimate_with_uncertainty"),
    ("WCONT-003", "degraded_service_and_recovery_tracking"),
    ("WCONT-004", "sensor_disagreement_unresolved"),
    ("WCONT-005", "suspected_leak_human_review_or_qualified_isolation"),
    ("WCONT-006", "delivery_unavailable_pump_dependent_service"),
    ("WCONT-007", "quality_unresolved_isolate_or_human_assessment"),
    ("WCONT-008", "mass_balance_failure_unresolved"),
    ("WCONT-009", "command_observation_disagreement"),
    ("WCONT-010", "recovery_monitored_with_pressure_constraints"),
];

const REPAIR_TASKS: [(&str, &str); 10] = [
    ("REPAIR-001", "repairable_if_exact_part_and_post_service_leak_check_pass"),
    ("REPAIR-002", "repairable_only_after_hydraulic_electrical_and_functional_requalification"),
    ("REPAIR-003", "repairable_if_seal_and_integrity_checks_pass"),
    ("REPAIR-004", "unqualified_until_calibration_and_independent_check_pass"),
    ("REPAIR-005", "safe_local_recovery_without_cloud_dependency"),
    ("REPAIR-006", "repair_blocked_parts_unavailable"),
    ("REPAIR-007", "substitution_rejected_or_held_for_qualification"),
    ("REPAIR-008", "initial_function_restored_durable_repair_failed"),
    ("REPAIR-009", "service_access_blocked_and_reported"),
    ("REPAIR-010", "not_released_until_product_specific_requalification"),
];

/// A deterministic validation failure with a JSON-path-like location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl ValidationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for ValidationError {}

/// Summary of a validated synthetic water-continuity corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaterCorpusSummary {
    pub corpus_id: String,
    pub schema_version: String,
    pub scenario_ids: Vec<String>,
    pub scenario_count: usize,
}

/// Summary of a validated synthetic repairability corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairCorpusSummary {
    pub corpus_id: String,
    pub schema_version: String,
    pub task_ids: Vec<String>,
    pub task_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WaterCorpus {
    schema_version: String,
    corpus_id: String,
    status: String,
    created_date: String,
    units: BTreeMap<String, String>,
    assumptions: BTreeMap<String, Value>,
    claim_ceiling: Vec<String>,
    scenarios: Vec<WaterScenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WaterScenario {
    id: String,
    title: String,
    input: BTreeMap<String, Value>,
    expected_disposition: String,
    invariants: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RepairCorpus {
    schema_version: String,
    corpus_id: String,
    status: String,
    created_date: String,
    claim_ceiling: Vec<String>,
    task_profiles: Vec<RepairTask>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RepairTask {
    id: String,
    title: String,
    product_class: String,
    task_class: String,
    tool_class: String,
    synthetic_baseline: BTreeMap<String, Value>,
    expected_disposition: String,
    invariants: Vec<String>,
}

/// Validate the frozen water-continuity fixture from UTF-8 JSON.
///
/// Passing means that the corpus is still a correctly labelled, internally
/// consistent synthetic challenge set. It does not establish a tank size,
/// pump duty point, water quality, plumbing design or household service level.
pub fn validate_water_continuity_corpus(
    input: &str,
) -> Result<WaterCorpusSummary, ValidationError> {
    let corpus: WaterCorpus = serde_json::from_str(input)
        .map_err(|error| ValidationError::new("$", format!("invalid JSON: {error}")))?;

    validate_header(
        &corpus.schema_version,
        &corpus.corpus_id,
        &corpus.status,
        &corpus.created_date,
        WATER_CORPUS_ID,
    )?;

    let required_units = [
        ("volume", "L"),
        ("flow", "L/min"),
        ("pressure", "kPa"),
        ("duration", "h"),
        ("energy", "kWh"),
    ];
    for (key, expected) in required_units {
        let observed = corpus.units.get(key).ok_or_else(|| {
            ValidationError::new(format!("$.units.{key}"), "required unit is missing")
        })?;
        if observed != expected {
            return Err(ValidationError::new(
                format!("$.units.{key}"),
                format!("expected {expected:?}, found {observed:?}"),
            ));
        }
    }

    require_bool(
        &corpus.assumptions,
        "all_values_are_synthetic",
        true,
        "$.assumptions",
    )?;
    require_bool(
        &corpus.assumptions,
        "no_live_actuation_authority",
        true,
        "$.assumptions",
    )?;
    require_bool(
        &corpus.assumptions,
        "non_potable_reuse",
        false,
        "$.assumptions",
    )?;
    require_string(
        &corpus.assumptions,
        "potable_quality",
        "unknown_unqualified",
        "$.assumptions",
    )?;

    require_claim(
        &corpus.claim_ceiling,
        "does_not_establish_potable_water_safety",
        "$.claim_ceiling",
    )?;
    require_claim(
        &corpus.claim_ceiling,
        "does_not_authorize_automatic_shutoff_or_source_switching",
        "$.claim_ceiling",
    )?;
    require_claim(
        &corpus.claim_ceiling,
        "does_not_establish_plumbing_or_electrical_compliance",
        "$.claim_ceiling",
    )?;

    if corpus.scenarios.len() != WATER_SCENARIOS.len() {
        return Err(ValidationError::new(
            "$.scenarios",
            format!(
                "expected exactly {} frozen scenarios, found {}",
                WATER_SCENARIOS.len(),
                corpus.scenarios.len()
            ),
        ));
    }

    let mut seen = BTreeSet::new();
    let mut ids = Vec::with_capacity(corpus.scenarios.len());
    for (index, scenario) in corpus.scenarios.iter().enumerate() {
        let base = format!("$.scenarios[{index}]");
        validate_record(
            &scenario.id,
            &scenario.title,
            &scenario.invariants,
            &base,
            &mut seen,
        )?;
        let expected = expected_disposition(&scenario.id, &WATER_SCENARIOS, &base)?;
        if scenario.expected_disposition != expected {
            return Err(ValidationError::new(
                format!("{base}.expected_disposition"),
                format!(
                    "scenario {} must remain {:?}, found {:?}",
                    scenario.id, expected, scenario.expected_disposition
                ),
            ));
        }
        validate_water_semantics(scenario, &base)?;
        ids.push(scenario.id.clone());
    }

    Ok(WaterCorpusSummary {
        corpus_id: corpus.corpus_id,
        schema_version: corpus.schema_version,
        scenario_count: ids.len(),
        scenario_ids: ids,
    })
}

/// Validate the frozen repairability fixture from UTF-8 JSON.
///
/// Passing means that synthetic repair tasks retain their fail-closed
/// constraints. It does not establish that a real part is available, a repair
/// is safe, a device is repairable or post-repair qualification has passed.
pub fn validate_repairability_corpus(
    input: &str,
) -> Result<RepairCorpusSummary, ValidationError> {
    let corpus: RepairCorpus = serde_json::from_str(input)
        .map_err(|error| ValidationError::new("$", format!("invalid JSON: {error}")))?;

    validate_header(
        &corpus.schema_version,
        &corpus.corpus_id,
        &corpus.status,
        &corpus.created_date,
        REPAIR_CORPUS_ID,
    )?;

    require_claim(
        &corpus.claim_ceiling,
        "synthetic_task_results_do_not_establish_actual_repairability",
        "$.claim_ceiling",
    )?;
    require_claim(
        &corpus.claim_ceiling,
        "no_task_authorizes_bypassing_safety_protection",
        "$.claim_ceiling",
    )?;
    require_claim(
        &corpus.claim_ceiling,
        "no_repair_is_successful_without_post_repair_function_and_required_safety_checks",
        "$.claim_ceiling",
    )?;

    if corpus.task_profiles.len() != REPAIR_TASKS.len() {
        return Err(ValidationError::new(
            "$.task_profiles",
            format!(
                "expected exactly {} frozen repair tasks, found {}",
                REPAIR_TASKS.len(),
                corpus.task_profiles.len()
            ),
        ));
    }

    let mut seen = BTreeSet::new();
    let mut ids = Vec::with_capacity(corpus.task_profiles.len());
    for (index, task) in corpus.task_profiles.iter().enumerate() {
        let base = format!("$.task_profiles[{index}]");
        validate_record(
            &task.id,
            &task.title,
            &task.invariants,
            &base,
            &mut seen,
        )?;
        let expected = expected_disposition(&task.id, &REPAIR_TASKS, &base)?;
        if task.expected_disposition != expected {
            return Err(ValidationError::new(
                format!("{base}.expected_disposition"),
                format!(
                    "task {} must remain {:?}, found {:?}",
                    task.id, expected, task.expected_disposition
                ),
            ));
        }
        require_non_empty(&task.product_class, &format!("{base}.product_class"))?;
        require_non_empty(&task.task_class, &format!("{base}.task_class"))?;
        require_non_empty(&task.tool_class, &format!("{base}.tool_class"))?;
        validate_repair_semantics(task, &base)?;
        ids.push(task.id.clone());
    }

    Ok(RepairCorpusSummary {
        corpus_id: corpus.corpus_id,
        schema_version: corpus.schema_version,
        task_count: ids.len(),
        task_ids: ids,
    })
}

fn validate_header(
    version: &str,
    corpus_id: &str,
    status: &str,
    created_date: &str,
    expected_id: &str,
) -> Result<(), ValidationError> {
    if version != FIXTURE_VERSION {
        return Err(ValidationError::new(
            "$.schema_version",
            format!("expected {FIXTURE_VERSION:?}, found {version:?}"),
        ));
    }
    if corpus_id != expected_id {
        return Err(ValidationError::new(
            "$.corpus_id",
            format!("expected canonical corpus ID {expected_id:?}"),
        ));
    }
    if status != FIXTURE_STATUS {
        return Err(ValidationError::new(
            "$.status",
            "fixture status must remain synthetic_test_fixture_only",
        ));
    }
    require_non_empty(created_date, "$.created_date")
}

fn validate_record(
    id: &str,
    title: &str,
    invariants: &[String],
    path: &str,
    seen: &mut BTreeSet<String>,
) -> Result<(), ValidationError> {
    require_non_empty(id, &format!("{path}.id"))?;
    require_non_empty(title, &format!("{path}.title"))?;
    if !seen.insert(id.to_owned()) {
        return Err(ValidationError::new(
            format!("{path}.id"),
            format!("duplicate record ID {id:?}"),
        ));
    }
    if invariants.is_empty() {
        return Err(ValidationError::new(
            format!("{path}.invariants"),
            "at least one explicit invariant is required",
        ));
    }
    let mut unique = BTreeSet::new();
    for (index, invariant) in invariants.iter().enumerate() {
        require_non_empty(invariant, &format!("{path}.invariants[{index}]"))?;
        if !unique.insert(invariant) {
            return Err(ValidationError::new(
                format!("{path}.invariants[{index}]"),
                format!("duplicate invariant {invariant:?}"),
            ));
        }
    }
    Ok(())
}

fn expected_disposition(
    id: &str,
    table: &[(&str, &str)],
    path: &str,
) -> Result<String, ValidationError> {
    table
        .iter()
        .find(|(known_id, _)| *known_id == id)
        .map(|(_, disposition)| (*disposition).to_owned())
        .ok_or_else(|| {
            ValidationError::new(
                format!("{path}.id"),
                format!("unknown frozen scenario/task ID {id:?}"),
            )
        })
}

fn require_claim(claims: &[String], required: &str, path: &str) -> Result<(), ValidationError> {
    if claims.iter().any(|claim| claim == required) {
        Ok(())
    } else {
        Err(ValidationError::new(
            path,
            format!("required claim ceiling entry {required:?} is missing"),
        ))
    }
}

fn require_non_empty(value: &str, path: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError::new(path, "must not be empty"))
    } else {
        Ok(())
    }
}

fn required_value<'a>(
    values: &'a BTreeMap<String, Value>,
    key: &str,
    path: &str,
) -> Result<&'a Value, ValidationError> {
    values
        .get(key)
        .ok_or_else(|| ValidationError::new(format!("{path}.{key}"), "required field is missing"))
}

fn required_bool(
    values: &BTreeMap<String, Value>,
    key: &str,
    expected: bool,
    path: &str,
) -> Result<(), ValidationError> {
    let observed = required_value(values, key, path)?
        .as_bool()
        .ok_or_else(|| ValidationError::new(format!("{path}.{key}"), "expected a boolean"))?;
    if observed == expected {
        Ok(())
    } else {
        Err(ValidationError::new(
            format!("{path}.{key}"),
            format!("expected {expected}, found {observed}"),
        ))
    }
}

fn required_number(
    values: &BTreeMap<String, Value>,
    key: &str,
    path: &str,
) -> Result<f64, ValidationError> {
    required_value(values, key, path)?
        .as_f64()
        .ok_or_else(|| ValidationError::new(format!("{path}.{key}"), "expected a number"))
}

fn required_string<'a>(
    values: &'a BTreeMap<String, Value>,
    key: &str,
    expected: &str,
    path: &str,
) -> Result<(), ValidationError> {
    let observed = required_value(values, key, path)?
        .as_str()
        .ok_or_else(|| ValidationError::new(format!("{path}.{key}"), "expected a string"))?;
    if observed == expected {
        Ok(())
    } else {
        Err(ValidationError::new(
            format!("{path}.{key}"),
            format!("expected {expected:?}, found {observed:?}"),
        ))
    }
}

fn require_invariant(
    invariants: &[String],
    expected: &str,
    path: &str,
) -> Result<(), ValidationError> {
    if invariants.iter().any(|invariant| invariant == expected) {
        Ok(())
    } else {
        Err(ValidationError::new(
            path,
            format!("required fail-closed invariant {expected:?} is missing"),
        ))
    }
}

fn validate_water_semantics(
    scenario: &WaterScenario,
    path: &str,
) -> Result<(), ValidationError> {
    let input = &scenario.input;
    match scenario.id.as_str() {
        "WCONT-001" => {
            if required_number(input, "duration_h", path)? <= 0.0
                || required_number(input, "municipal_inlet_flow_l_min", path)? <= 0.0
            {
                return Err(ValidationError::new(
                    format!("{path}.input"),
                    "stable-supply fixture requires positive duration and inlet flow",
                ));
            }
            require_string(
                input,
                "stored_water_quality_evidence",
                "not_provided",
                &format!("{path}.input"),
            )?;
        }
        "WCONT-002" => {
            if required_number(input, "duration_h", path)? < 48.0
                || required_number(input, "municipal_inlet_flow_l_min", path)? != 0.0
            {
                return Err(ValidationError::new(
                    format!("{path}.input"),
                    "48-hour outage fixture requires at least 48 hours and zero inlet flow",
                ));
            }
            require_invariant(
                &scenario.invariants,
                "reserve_duration_is_reported_as_a_range_or_unresolved_if_demand_is_unknown",
                &format!("{path}.invariants"),
            )?;
        }
        "WCONT-003" => {
            if required_number(input, "municipal_inlet_flow_l_min", path)? <= 0.0 {
                return Err(ValidationError::new(
                    format!("{path}.input.municipal_inlet_flow_l_min"),
                    "staggered recovery requires measured positive but constrained flow",
                ));
            }
            require_string(
                input,
                "supply_pressure_profile",
                "intermittent_below_nominal",
                &format!("{path}.input"),
            )?;
        }
        "WCONT-004" => {
            let indicated = required_number(input, "tank_level_sensor_l", path)?;
            let reference = required_number(input, "independent_reference_tank_level_l", path)?;
            if indicated == reference {
                return Err(ValidationError::new(
                    format!("{path}.input"),
                    "stuck-sensor challenge must disagree with the independent reference",
                ));
            }
            require_string(input, "sensor_fault", "stuck_value", &format!("{path}.input"))?;
        }
        "WCONT-005" => {
            if required_number(input, "measured_unexplained_flow_l_min", path)? <= 0.0 {
                return Err(ValidationError::new(
                    format!("{path}.input.measured_unexplained_flow_l_min"),
                    "leak challenge requires positive unexplained flow",
                ));
            }
            require_string(input, "leak_detected", "candidate_alarm", &format!("{path}.input"))?;
        }
        "WCONT-006" => {
            require_bool(input, "grid_available", false, &format!("{path}.input"))?;
            require_bool(input, "pump_available", false, &format!("{path}.input"))?;
            require_bool(input, "backup_power_available", false, &format!("{path}.input"))?;
            require_invariant(
                &scenario.invariants,
                "stored_volume_is_not_misreported_as_deliverable_service",
                &format!("{path}.invariants"),
            )?;
        }
        "WCONT-007" => {
            if required_number(input, "stored_water_age_h", path)? <= 0.0 {
                return Err(ValidationError::new(
                    format!("{path}.input.stored_water_age_h"),
                    "stale-quality challenge requires positive stored-water age",
                ));
            }
            require_string(
                input,
                "water_quality_evidence",
                "stale_or_missing",
                &format!("{path}.input"),
            )?;
            require_invariant(
                &scenario.invariants,
                "volume_availability_does_not_establish_water_quality",
                &format!("{path}.invariants"),
            )?;
        }
        "WCONT-008" => {
            let inlet = required_number(input, "municipal_inlet_volume_l", path)?;
            let storage_delta = required_number(input, "tank_change_volume_l", path)?;
            let outlet = required_number(input, "outlet_meter_volume_l", path)?;
            let purge = required_number(input, "purge_and_overflow_meter_volume_l", path)?;
            let uncertainty = required_number(input, "meter_uncertainty_l", path)?;
            if uncertainty < 0.0 || (inlet - storage_delta - outlet - purge).abs() <= uncertainty {
                return Err(ValidationError::new(
                    format!("{path}.input"),
                    "mass-balance challenge must contain a residual greater than declared meter uncertainty",
                ));
            }
            require_invariant(
                &scenario.invariants,
                "unexplained_mass_balance_residual_is_reported",
                &format!("{path}.invariants"),
            )?;
        }
        "WCONT-009" => {
            require_bool(input, "commanded_refill", true, &format!("{path}.input"))?;
            if required_number(input, "municipal_inlet_flow_l_min", path)? != 0.0 {
                return Err(ValidationError::new(
                    format!("{path}.input.municipal_inlet_flow_l_min"),
                    "command/observation challenge requires zero measured inflow",
                ));
            }
        }
        "WCONT-010" => {
            require_string(
                input,
                "municipal_inlet_flow_profile",
                "unstable_recovery",
                &format!("{path}.input"),
            )?;
            require_string(
                input,
                "pressure_profile",
                "surge_then_decay",
                &format!("{path}.input"),
            )?;
        }
        _ => {
            return Err(ValidationError::new(
                format!("{path}.id"),
                "scenario is not part of the frozen v0.1 water corpus",
            ));
        }
    }
    Ok(())
}

fn validate_repair_semantics(task: &RepairTask, path: &str) -> Result<(), ValidationError> {
    let baseline = &task.synthetic_baseline;
    match task.id.as_str() {
        "REPAIR-001" => {
            require_bool(baseline, "repair_instruction_available", true, &format!("{path}.synthetic_baseline"))?;
            require_bool(baseline, "offline_access", true, &format!("{path}.synthetic_baseline"))?;
            require_invariant(
                &task.invariants,
                "part_identity_and_compatibility_are_verified",
                &format!("{path}.invariants"),
            )?;
        }
        "REPAIR-002" => {
            require_bool(baseline, "calibration_required", true, &format!("{path}.synthetic_baseline"))?;
            require_invariant(
                &task.invariants,
                "pressure_flow_leak_and_protection_checks_are_repeated",
                &format!("{path}.invariants"),
            )?;
        }
        "REPAIR-003" => {
            require_bool(
                baseline,
                "adhesive_or_destructive_disassembly",
                false,
                &format!("{path}.synthetic_baseline"),
            )?;
            require_invariant(
                &task.invariants,
                "closure_and_leak_checks_are_performed",
                &format!("{path}.invariants"),
            )?;
        }
        "REPAIR-004" => {
            require_bool(baseline, "calibration_required", true, &format!("{path}.synthetic_baseline"))?;
            require_invariant(
                &task.invariants,
                "old_calibration_is_not_inherited",
                &format!("{path}.invariants"),
            )?;
        }
        "REPAIR-005" => {
            require_bool(baseline, "internet_available", false, &format!("{path}.synthetic_baseline"))?;
            require_bool(baseline, "signed_image_available", true, &format!("{path}.synthetic_baseline"))?;
            require_bool(
                baseline,
                "previous_known_good_image_available",
                true,
                &format!("{path}.synthetic_baseline"),
            )?;
            require_invariant(
                &task.invariants,
                "basic_safe_operation_does_not_require_cloud",
                &format!("{path}.invariants"),
            )?;
        }
        "REPAIR-006" => {
            require_bool(baseline, "catalog_part_exists", true, &format!("{path}.synthetic_baseline"))?;
            require_bool(baseline, "local_stock", false, &format!("{path}.synthetic_baseline"))?;
            require_bool(baseline, "approved_substitute", false, &format!("{path}.synthetic_baseline"))?;
            if !required_value(baseline, "quoted_lead_time_days", &format!("{path}.synthetic_baseline"))?.is_null() {
                return Err(ValidationError::new(
                    format!("{path}.synthetic_baseline.quoted_lead_time_days"),
                    "unknown lead time must remain null, not an invented value",
                ));
            }
        }
        "REPAIR-007" => {
            require_bool(baseline, "physical_fit", true, &format!("{path}.synthetic_baseline"))?;
            require_bool(baseline, "material_grade_known", false, &format!("{path}.synthetic_baseline"))?;
            require_string(
                baseline,
                "qualification_state",
                "unqualified",
                &format!("{path}.synthetic_baseline"),
            )?;
            require_invariant(
                &task.invariants,
                "physical_fit_does_not_establish_functional_compatibility",
                &format!("{path}.invariants"),
            )?;
        }
        "REPAIR-008" => {
            require_bool(
                baseline,
                "initial_post_repair_function_pass",
                true,
                &format!("{path}.synthetic_baseline"),
            )?;
            require_bool(
                baseline,
                "recurrence_observed_after_30_days",
                true,
                &format!("{path}.synthetic_baseline"),
            )?;
            require_invariant(
                &task.invariants,
                "initial_restoration_and_durable_success_are_separate_outcomes",
                &format!("{path}.invariants"),
            )?;
        }
        "REPAIR-009" => {
            require_bool(baseline, "vendor_cloud_available", false, &format!("{path}.synthetic_baseline"))?;
            require_bool(baseline, "offline_diagnostics_available", false, &format!("{path}.synthetic_baseline"))?;
            require_invariant(
                &task.invariants,
                "cloud_dependency_is_reported_as_a_repairability_failure",
                &format!("{path}.invariants"),
            )?;
        }
        "REPAIR-010" => {
            require_bool(baseline, "critical_interface_changed", true, &format!("{path}.synthetic_baseline"))?;
            require_bool(baseline, "post_repair_test_available", false, &format!("{path}.synthetic_baseline"))?;
            require_invariant(
                &task.invariants,
                "repair_completion_does_not_equal_safe_release",
                &format!("{path}.invariants"),
            )?;
        }
        _ => {
            return Err(ValidationError::new(
                format!("{path}.id"),
                "task is not part of the frozen v0.1 repair corpus",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const WATER_FIXTURE: &str =
        include_str!("../../../../docs/research/household-services/fixtures/water-continuity-scenarios-v0.1.json");
    const REPAIR_FIXTURE: &str =
        include_str!("../../../../docs/research/household-services/fixtures/repairability-scenarios-v0.1.json");

    fn water_value() -> Value {
        serde_json::from_str(WATER_FIXTURE).expect("frozen water fixture must be valid JSON")
    }

    fn repair_value() -> Value {
        serde_json::from_str(REPAIR_FIXTURE).expect("frozen repairability fixture must be valid JSON")
    }

    #[test]
    fn frozen_water_corpus_is_accepted_as_synthetic_only() {
        let summary = validate_water_continuity_corpus(WATER_FIXTURE).unwrap();
        assert_eq!(summary.corpus_id, WATER_CORPUS_ID);
        assert_eq!(summary.scenario_count, 10);
        assert_eq!(summary.scenario_ids.first().map(String::as_str), Some("WCONT-001"));
        assert_eq!(summary.scenario_ids.last().map(String::as_str), Some("WCONT-010"));
    }

    #[test]
    fn frozen_repairability_corpus_is_accepted_as_synthetic_only() {
        let summary = validate_repairability_corpus(REPAIR_FIXTURE).unwrap();
        assert_eq!(summary.corpus_id, REPAIR_CORPUS_ID);
        assert_eq!(summary.task_count, 10);
        assert_eq!(summary.task_ids.first().map(String::as_str), Some("REPAIR-001"));
        assert_eq!(summary.task_ids.last().map(String::as_str), Some("REPAIR-010"));
    }

    #[test]
    fn water_fixture_cannot_be_relabelled_as_a_physical_result() {
        let mut value = water_value();
        value["status"] = json!("physical_test_passed");
        let encoded = serde_json::to_string(&value).unwrap();
        let error = validate_water_continuity_corpus(&encoded).unwrap_err();
        assert_eq!(error.path, "$.status");
    }

    #[test]
    fn water_outage_control_requires_zero_inlet_flow() {
        let mut value = water_value();
        value["scenarios"][1]["input"]["municipal_inlet_flow_l_min"] = json!(0.5);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(validate_water_continuity_corpus(&encoded).is_err());
    }

    #[test]
    fn water_quality_control_cannot_be_changed_to_a_pass() {
        let mut value = water_value();
        value["scenarios"][6]["input"]["water_quality_evidence"] = json!("independent_pass");
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(validate_water_continuity_corpus(&encoded).is_err());
    }

    #[test]
    fn water_meter_conflict_must_exceed_declared_uncertainty() {
        let mut value = water_value();
        value["scenarios"][7]["input"]["outlet_meter_volume_l"] = json!(320.0);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(validate_water_continuity_corpus(&encoded).is_err());
    }

    #[test]
    fn duplicate_water_scenario_is_rejected() {
        let mut value = water_value();
        let duplicate = value["scenarios"][0].clone();
        value["scenarios"].as_array_mut().unwrap().push(duplicate);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(validate_water_continuity_corpus(&encoded).is_err());
    }

    #[test]
    fn repair_fixture_cannot_promote_unqualified_substitute_to_release() {
        let mut value = repair_value();
        value["task_profiles"][6]["synthetic_baseline"]["qualification_state"] = json!("qualified");
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(validate_repairability_corpus(&encoded).is_err());
    }

    #[test]
    fn repair_recurrence_remains_a_durable_failure() {
        let mut value = repair_value();
        value["task_profiles"][7]["synthetic_baseline"]["recurrence_observed_after_30_days"] = json!(false);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(validate_repairability_corpus(&encoded).is_err());
    }

    #[test]
    fn repair_without_post_repair_qualification_cannot_be_released() {
        let mut value = repair_value();
        value["task_profiles"][9]["invariants"] = json!([
            "critical_interface_change_reopens_affected_qualification",
            "missing_test_capability_requires_hold_or_qualified_referral"
        ]);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(validate_repairability_corpus(&encoded).is_err());
    }

    #[test]
    fn duplicate_repair_task_is_rejected() {
        let mut value = repair_value();
        let duplicate = value["task_profiles"][0].clone();
        value["task_profiles"].as_array_mut().unwrap().push(duplicate);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(validate_repairability_corpus(&encoded).is_err());
    }
}
