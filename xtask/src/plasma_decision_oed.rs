// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-only, plasma-specific decision contract for multifidelity OED.
//!
//! This module validates application outcomes and selection receipts. It does
//! not rank experiments or implement an optimizer; UAP-OED owns that engine.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const EXPECTED_EVIDENCE_CLASS: &str = "SYNTHETIC";
const SYNTHETIC_LABEL_CLASS: &str = "SYNTHETIC_LABEL";
const FLOAT_EPSILON: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AuditDisposition {
    Pass,
    Blocked,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProductDecision {
    AcceptWindow,
    RejectWindow,
    Abstain,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    benchmark_id: String,
    evidence_class: String,
    synthetic_only: bool,
    physical_measurements: bool,
    optimizer_executed: bool,
    automatic_experiment_execution: bool,
    automatic_quality_release: bool,
    claim_ceiling: String,
    action_catalog: Vec<Action>,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Action {
    action_id: String,
    fidelity: String,
    cost_units: f64,
    latency_hours: f64,
    specimens: u64,
    provides: Vec<String>,
    admissible: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    profile: Profile,
    budget: Budget,
    selection_receipt: SelectionReceipt,
    measurements: Vec<Measurement>,
    covariance_models: Vec<CovarianceModel>,
    outcomes: Outcomes,
    expected: Expected,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    profile_id: String,
    material_family: String,
    requires_barrier_preservation: bool,
    min_bond_strength_mpa: f64,
    min_aged_bond_strength_mpa: f64,
    min_retention: f64,
    max_wvtr_ratio: Option<f64>,
    max_otr_ratio: Option<f64>,
    acceptable_failure_modes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Budget {
    max_cost_units: f64,
    max_latency_hours: f64,
    max_specimens: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionReceipt {
    receipt_id: String,
    owner: String,
    method: String,
    objective: String,
    selected_action_ids: Vec<String>,
    training_group_manifest: TrainingGroupManifest,
    model_adequacy_status: String,
    model_adequacy_receipt_id: String,
    other_model_registered: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingGroupManifest {
    manifest_id: String,
    training_lots: Vec<String>,
    training_devices: Vec<String>,
    training_days: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurement {
    measurement_id: String,
    action_id: String,
    specimen_id: String,
    lot_id: String,
    device_id: String,
    production_day: String,
    sample_preparation_group: String,
    independence_group: String,
    covariance_model_id: Option<String>,
    quantity: String,
    unit: String,
    calibration_record_id: String,
    calibration_status: String,
    uncertainty: f64,
    raw_trace_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CovarianceModel {
    model_id: String,
    evidence_class: String,
    receipt_id: String,
    measurement_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Outcomes {
    wca_deg: Option<f64>,
    bond_strength_mpa: Option<f64>,
    aged_bond_strength_mpa: Option<f64>,
    bond_retention: Option<f64>,
    failure_mode: Option<String>,
    wvtr_ratio: Option<f64>,
    otr_ratio: Option<f64>,
    evidence_class: String,
    physically_measured: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    audit_disposition: AuditDispositionInput,
    product_decision: ProductDecisionInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum AuditDispositionInput {
    Pass,
    Blocked,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ProductDecisionInput {
    AcceptWindow,
    RejectWindow,
    Abstain,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub audit_disposition: AuditDisposition,
    pub product_decision: ProductDecision,
    pub expected_audit_disposition: AuditDispositionInput,
    pub expected_product_decision: ProductDecisionInput,
    pub reasons: Vec<String>,
    pub selected_action_count: usize,
    pub measurement_count: usize,
    pub selected_cost_units: Option<f64>,
    pub selected_latency_hours: Option<f64>,
    pub selected_specimens: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub schema_version: String,
    pub benchmark_id: String,
    pub evidence_class: String,
    pub synthetic_only: bool,
    pub optimizer_executed: bool,
    pub automatic_experiment_execution: bool,
    pub automatic_quality_release: bool,
    pub scenario_count: usize,
    pub matched_expectations: usize,
    pub mismatched_expectations: usize,
    pub pass_count: usize,
    pub blocked_count: usize,
    pub rejected_count: usize,
    pub results: Vec<ScenarioResult>,
    pub claim_ceiling: String,
}

pub fn run_audit(fixture_path: &Path) -> Result<AuditReport> {
    let bytes = fs::read(fixture_path)
        .with_context(|| format!("read plasma decision-OED fixture {}", fixture_path.display()))?;
    let input = std::str::from_utf8(&bytes).context("plasma decision-OED fixture is not UTF-8")?;
    audit_fixture_json(input).map_err(anyhow::Error::msg)
}

pub fn audit_fixture_json(input: &str) -> std::result::Result<AuditReport, String> {
    let fixture: Fixture = serde_json::from_str(input)
        .map_err(|error| format!("invalid plasma decision-OED fixture: {error}"))?;

    if fixture.schema_version != "0.1.0" || fixture.benchmark_id != "PLASMA-OED-001" {
        return Err("unsupported plasma decision-OED schema or benchmark ID".to_owned());
    }
    if fixture.evidence_class != EXPECTED_EVIDENCE_CLASS
        || !fixture.synthetic_only
        || fixture.physical_measurements
        || fixture.optimizer_executed
        || fixture.automatic_experiment_execution
        || fixture.automatic_quality_release
    {
        return Err(
            "fixture must remain synthetic-only, non-physical, optimizer-unexecuted, advisory and non-actuating"
                .to_owned(),
        );
    }
    if fixture.claim_ceiling.trim().is_empty() {
        return Err("empty plasma decision-OED claim ceiling".to_owned());
    }
    if fixture.action_catalog.is_empty() || fixture.scenarios.is_empty() {
        return Err("action catalogue and scenario set must be non-empty".to_owned());
    }

    let mut action_map = BTreeMap::<String, &Action>::new();
    for action in &fixture.action_catalog {
        if action.action_id.trim().is_empty()
            || !matches!(action.fidelity.as_str(), "F0" | "F1" | "F2" | "F3" | "F4")
            || !action.cost_units.is_finite()
            || action.cost_units <= 0.0
            || !action.latency_hours.is_finite()
            || action.latency_hours < 0.0
            || action.specimens == 0
            || action.provides.is_empty()
            || action.provides.iter().any(|item| item.trim().is_empty())
            || action_map.insert(action.action_id.clone(), action).is_some()
        {
            return Err("action catalogue contains an invalid or duplicate action".to_owned());
        }
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in &fixture.scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.as_str())
        {
            return Err("scenario identity is empty or duplicated".to_owned());
        }
    }

    let results: Vec<_> = fixture
        .scenarios
        .iter()
        .map(|scenario| evaluate_scenario(scenario, &action_map))
        .collect();

    let matched_expectations = results
        .iter()
        .filter(|result| {
            result.audit_disposition == audit_input_to_output(result.expected_audit_disposition)
                && result.product_decision
                    == product_input_to_output(result.expected_product_decision)
        })
        .count();
    let pass_count = results
        .iter()
        .filter(|result| result.audit_disposition == AuditDisposition::Pass)
        .count();
    let blocked_count = results
        .iter()
        .filter(|result| result.audit_disposition == AuditDisposition::Blocked)
        .count();
    let rejected_count = results
        .iter()
        .filter(|result| result.audit_disposition == AuditDisposition::Rejected)
        .count();
    let scenario_count = results.len();

    Ok(AuditReport {
        schema_version: fixture.schema_version,
        benchmark_id: fixture.benchmark_id,
        evidence_class: fixture.evidence_class,
        synthetic_only: fixture.synthetic_only,
        optimizer_executed: fixture.optimizer_executed,
        automatic_experiment_execution: fixture.automatic_experiment_execution,
        automatic_quality_release: fixture.automatic_quality_release,
        scenario_count,
        matched_expectations,
        mismatched_expectations: scenario_count - matched_expectations,
        pass_count,
        blocked_count,
        rejected_count,
        results,
        claim_ceiling: fixture.claim_ceiling,
    })
}

fn audit_input_to_output(value: AuditDispositionInput) -> AuditDisposition {
    match value {
        AuditDispositionInput::Pass => AuditDisposition::Pass,
        AuditDispositionInput::Blocked => AuditDisposition::Blocked,
        AuditDispositionInput::Rejected => AuditDisposition::Rejected,
    }
}

fn product_input_to_output(value: ProductDecisionInput) -> ProductDecision {
    match value {
        ProductDecisionInput::AcceptWindow => ProductDecision::AcceptWindow,
        ProductDecisionInput::RejectWindow => ProductDecision::RejectWindow,
        ProductDecisionInput::Abstain => ProductDecision::Abstain,
    }
}

fn evaluate_scenario(
    scenario: &Scenario,
    actions: &BTreeMap<String, &Action>,
) -> ScenarioResult {
    let mut rejected = Vec::<String>::new();
    let mut blocked = Vec::<String>::new();

    let profile = &scenario.profile;
    if profile.profile_id.trim().is_empty()
        || profile.material_family.trim().is_empty()
        || !profile.min_bond_strength_mpa.is_finite()
        || profile.min_bond_strength_mpa <= 0.0
        || !profile.min_aged_bond_strength_mpa.is_finite()
        || profile.min_aged_bond_strength_mpa <= 0.0
        || !profile.min_retention.is_finite()
        || !(0.0..=1.0).contains(&profile.min_retention)
        || profile.acceptable_failure_modes.is_empty()
        || profile.acceptable_failure_modes.iter().any(|mode| mode.trim().is_empty())
    {
        rejected.push("frozen material/adhesive decision profile is invalid".to_owned());
    }
    if profile.requires_barrier_preservation
        && (profile.max_wvtr_ratio.is_none_or(|value| !value.is_finite() || value <= 0.0)
            || profile.max_otr_ratio.is_none_or(|value| !value.is_finite() || value <= 0.0))
    {
        rejected.push("barrier-preservation profile lacks finite positive WVTR/OTR limits".to_owned());
    }

    if scenario.budget.max_cost_units.is_finite() == false
        || scenario.budget.max_cost_units < 0.0
        || scenario.budget.max_latency_hours.is_finite() == false
        || scenario.budget.max_latency_hours < 0.0
        || scenario.budget.max_specimens == 0
    {
        rejected.push("experiment resource budget is invalid".to_owned());
    }

    let receipt = &scenario.selection_receipt;
    if receipt.receipt_id.trim().is_empty()
        || receipt.owner != "UAP-OED-001"
        || receipt.method.trim().is_empty()
        || receipt.objective.trim().is_empty()
        || receipt.selected_action_ids.is_empty()
    {
        rejected.push("selection receipt identity/ownership/action set is invalid".to_owned());
    }
    if !matches!(
        receipt.model_adequacy_status.as_str(),
        "ADEQUATE" | "INADEQUATE" | "UNRESOLVED"
    ) {
        rejected.push("selection receipt has an unsupported model-adequacy status".to_owned());
    } else if receipt.model_adequacy_status != "ADEQUATE" {
        blocked.push("model family is INADEQUATE/UNRESOLVED; retain OTHER and abstain".to_owned());
    }
    if receipt.model_adequacy_receipt_id.trim().is_empty() || !receipt.other_model_registered {
        blocked.push("model-adequacy receipt or explicit OTHER hypothesis registration is missing".to_owned());
    }

    let mut selected_ids = BTreeSet::new();
    let mut selected_cost = 0.0_f64;
    let mut selected_latency = 0.0_f64;
    let mut selected_specimens = 0_u64;
    let mut provided = BTreeSet::<String>::new();

    for action_id in &receipt.selected_action_ids {
        if !selected_ids.insert(action_id.as_str()) {
            rejected.push(format!("selected action {action_id} is duplicated"));
            continue;
        }
        match actions.get(action_id) {
            None => rejected.push(format!("selected action {action_id} is not in the frozen catalogue")),
            Some(action) => {
                if !action.admissible {
                    rejected.push(format!("selected action {action_id} is not admissible"));
                }
                selected_cost += action.cost_units;
                selected_latency += action.latency_hours;
                selected_specimens = selected_specimens.saturating_add(action.specimens);
                provided.extend(action.provides.iter().cloned());
            }
        }
    }

    if selected_cost > scenario.budget.max_cost_units + FLOAT_EPSILON
        || selected_latency > scenario.budget.max_latency_hours + FLOAT_EPSILON
        || selected_specimens > scenario.budget.max_specimens
    {
        rejected.push("selected experiment portfolio exceeds its frozen cost, latency or specimen budget".to_owned());
    }

    let mut measurement_ids = BTreeSet::new();
    let mut measurement_by_action = BTreeMap::<String, usize>::new();
    let mut dependency_groups = BTreeMap::<(String, String), Vec<&Measurement>>::new();
    let mut lot_ids = BTreeSet::<String>::new();
    let mut device_ids = BTreeSet::<String>::new();
    let mut production_days = BTreeSet::<String>::new();

    for measurement in &scenario.measurements {
        if measurement.measurement_id.trim().is_empty()
            || !measurement_ids.insert(measurement.measurement_id.as_str())
        {
            rejected.push("measurement identity is empty or duplicated".to_owned());
        }
        if !selected_ids.contains(measurement.action_id.as_str()) {
            rejected.push(format!("measurement {} references an unselected action", measurement.measurement_id));
        }
        if measurement.specimen_id.trim().is_empty()
            || measurement.lot_id.trim().is_empty()
            || measurement.device_id.trim().is_empty()
            || measurement.production_day.trim().is_empty()
            || measurement.sample_preparation_group.trim().is_empty()
            || measurement.independence_group.trim().is_empty()
            || measurement.quantity.trim().is_empty()
            || measurement.unit.trim().is_empty()
            || measurement.calibration_record_id.trim().is_empty()
            || measurement.raw_trace_digest.trim().is_empty()
        {
            blocked.push(format!("measurement {} lacks specimen/group/quantity/unit/calibration/raw-trace identity", measurement.measurement_id));
        }
        if measurement.calibration_status != "QUALIFIED" {
            blocked.push(format!("measurement {} lacks qualified calibration", measurement.measurement_id));
        }
        if !measurement.uncertainty.is_finite() || measurement.uncertainty < 0.0 {
            rejected.push(format!("measurement {} has invalid uncertainty", measurement.measurement_id));
        }
        *measurement_by_action.entry(measurement.action_id.clone()).or_default() += 1;
        dependency_groups
            .entry(("SAMPLE_PREPARATION".to_owned(), measurement.sample_preparation_group.clone()))
            .or_default()
            .push(measurement);
        dependency_groups
            .entry(("INDEPENDENCE".to_owned(), measurement.independence_group.clone()))
            .or_default()
            .push(measurement);
        lot_ids.insert(measurement.lot_id.clone());
        device_ids.insert(measurement.device_id.clone());
        production_days.insert(measurement.production_day.clone());
    }

    for action_id in &receipt.selected_action_ids {
        if !measurement_by_action.contains_key(action_id) {
            blocked.push(format!("selected action {action_id} has no admitted measurement record"));
        }
    }

    if lot_ids.len() > 1 || device_ids.len() > 1 || production_days.len() > 1 {
        rejected.push("one scenario joins measurements across different lot/device/day identities".to_owned());
    }

    let manifest = &receipt.training_group_manifest;
    if manifest.manifest_id.trim().is_empty()
        || manifest.training_lots.is_empty()
        || manifest.training_devices.is_empty()
        || manifest.training_days.is_empty()
        || manifest.training_lots.iter().any(|value| value.trim().is_empty())
        || manifest.training_devices.iter().any(|value| value.trim().is_empty())
        || manifest.training_days.iter().any(|value| value.trim().is_empty())
    {
        blocked.push("training-group holdout manifest is incomplete".to_owned());
    }
    let training_lots: BTreeSet<_> = manifest.training_lots.iter().map(String::as_str).collect();
    let training_devices: BTreeSet<_> = manifest.training_devices.iter().map(String::as_str).collect();
    let training_days: BTreeSet<_> = manifest.training_days.iter().map(String::as_str).collect();
    if training_lots.len() != manifest.training_lots.len()
        || training_devices.len() != manifest.training_devices.len()
        || training_days.len() != manifest.training_days.len()
    {
        rejected.push("training-group manifest contains duplicate identities".to_owned());
    }
    for measurement in &scenario.measurements {
        if training_lots.contains(measurement.lot_id.as_str())
            || training_devices.contains(measurement.device_id.as_str())
            || training_days.contains(measurement.production_day.as_str())
        {
            rejected.push("a held-out measurement lot/device/day overlaps model training groups".to_owned());
            break;
        }
    }

    let mut covariance_map = BTreeMap::<String, &CovarianceModel>::new();
    for model in &scenario.covariance_models {
        if model.model_id.trim().is_empty()
            || model.receipt_id.trim().is_empty()
            || model.evidence_class != EXPECTED_EVIDENCE_CLASS
            || model.measurement_ids.is_empty()
            || covariance_map.insert(model.model_id.clone(), model).is_some()
        {
            rejected.push("covariance receipt identity/evidence is invalid or duplicated".to_owned());
        }
    }

    for ((group_kind, _group_id), group) in dependency_groups.iter().filter(|(_, items)| items.len() > 1) {
        let group_ids: BTreeSet<_> = group.iter().map(|measurement| measurement.measurement_id.clone()).collect();
        let declared_model = group[0].covariance_model_id.as_deref();
        let same_model = declared_model.is_some()
            && group.iter().all(|measurement| measurement.covariance_model_id.as_deref() == declared_model);
        let covariance_valid = declared_model
            .and_then(|model_id| covariance_map.get(model_id))
            .is_some_and(|model| {
                let recorded: BTreeSet<_> = model.measurement_ids.iter().cloned().collect();
                recorded.len() == model.measurement_ids.len()
                    && group_ids.is_subset(&recorded)
            });
        if !same_model || !covariance_valid {
            blocked.push(format!(
                "shared {group_kind} group lacks a matched selected-set covariance receipt"
            ));
        }
    }

    for model in scenario.covariance_models.iter() {
        let ids: BTreeSet<_> = model.measurement_ids.iter().cloned().collect();
        if ids.len() != model.measurement_ids.len()
            || ids.iter().any(|id| !measurement_ids.contains(id.as_str()))
        {
            rejected.push(format!("covariance receipt {} references duplicate or unknown measurements", model.model_id));
        }
    }

    let mut required_endpoints: BTreeSet<String> = [
        "bond_strength",
        "failure_mode",
        "aged_bond_strength",
        "bond_retention",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    if profile.requires_barrier_preservation {
        required_endpoints.insert("water_vapour_barrier".to_owned());
        required_endpoints.insert("oxygen_barrier".to_owned());
    }
    for endpoint in &required_endpoints {
        if !provided.contains(endpoint) {
            blocked.push(format!("selected portfolio does not cover mandatory product endpoint {endpoint}"));
        }
    }

    let outcomes = &scenario.outcomes;
    if outcomes.evidence_class != SYNTHETIC_LABEL_CLASS || outcomes.physically_measured {
        rejected.push("synthetic fixture outcomes cannot claim physical measurement".to_owned());
    }

    let require_value = |value: Option<f64>, label: &str, blocked: &mut Vec<String>| {
        match value {
            Some(number) if number.is_finite() && number >= 0.0 => Some(number),
            _ => {
                blocked.push(format!("mandatory outcome {label} is missing or invalid"));
                None
            }
        }
    };
    let bond = require_value(outcomes.bond_strength_mpa, "bond_strength_mpa", &mut blocked);
    let aged = require_value(outcomes.aged_bond_strength_mpa, "aged_bond_strength_mpa", &mut blocked);
    let retention = require_value(outcomes.bond_retention, "bond_retention", &mut blocked);
    if outcomes.failure_mode.as_deref().is_none_or(str::is_empty) {
        blocked.push("mandatory outcome failure_mode is missing".to_owned());
    }

    if profile.requires_barrier_preservation {
        require_value(outcomes.wvtr_ratio, "wvtr_ratio", &mut blocked);
        require_value(outcomes.otr_ratio, "otr_ratio", &mut blocked);
    }

    let (audit_disposition, product_decision, reasons) = if !rejected.is_empty() {
        (AuditDisposition::Rejected, ProductDecision::Abstain, rejected)
    } else if !blocked.is_empty() {
        (AuditDisposition::Blocked, ProductDecision::Abstain, blocked)
    } else {
        let mut failures = Vec::<String>::new();
        if bond.is_some_and(|value| value + FLOAT_EPSILON < profile.min_bond_strength_mpa) {
            failures.push("initial bond strength is below the frozen application floor".to_owned());
        }
        if aged.is_some_and(|value| value + FLOAT_EPSILON < profile.min_aged_bond_strength_mpa) {
            failures.push("aged bond strength is below the frozen application floor".to_owned());
        }
        if retention.is_some_and(|value| value + FLOAT_EPSILON < profile.min_retention) {
            failures.push("bond-strength retention is below the frozen application floor".to_owned());
        }
        if outcomes.failure_mode.as_ref().is_some_and(|mode| {
            !profile.acceptable_failure_modes.iter().any(|acceptable| acceptable == mode)
        }) {
            failures.push("failure mode is outside the frozen acceptable set".to_owned());
        }
        if profile.requires_barrier_preservation {
            if outcomes.wvtr_ratio.zip(profile.max_wvtr_ratio)
                .is_some_and(|(value, maximum)| value > maximum + FLOAT_EPSILON)
            {
                failures.push("water-vapour barrier preservation limit is violated".to_owned());
            }
            if outcomes.otr_ratio.zip(profile.max_otr_ratio)
                .is_some_and(|(value, maximum)| value > maximum + FLOAT_EPSILON)
            {
                failures.push("oxygen barrier preservation limit is violated".to_owned());
            }
        }
        if failures.is_empty() {
            (AuditDisposition::Pass, ProductDecision::AcceptWindow, Vec::new())
        } else {
            (AuditDisposition::Pass, ProductDecision::RejectWindow, failures)
        }
    };

    ScenarioResult {
        scenario_id: scenario.scenario_id.clone(),
        audit_disposition,
        product_decision,
        expected_audit_disposition: scenario.expected.audit_disposition,
        expected_product_decision: scenario.expected.product_decision,
        reasons,
        selected_action_count: receipt.selected_action_ids.len(),
        measurement_count: scenario.measurements.len(),
        selected_cost_units: selected_cost.is_finite().then_some(selected_cost),
        selected_latency_hours: selected_latency.is_finite().then_some(selected_latency),
        selected_specimens: Some(selected_specimens),
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
                "  {:?} / {:?} {} actions={} measurements={} cost={:?} latency_h={:?}",
                result.audit_disposition,
                result.product_decision,
                result.scenario_id,
                result.selected_action_count,
                result.measurement_count,
                result.selected_cost_units,
                result.selected_latency_hours
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
            "plasma decision-OED fixture has {} disposition mismatches",
            report.mismatched_expectations
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(
        "../../data/research/plasma_decision_oed_fixture_v0_1.json"
    );

    fn report() -> AuditReport {
        audit_fixture_json(FIXTURE).expect("decision-OED fixture audit")
    }

    fn result_for<'a>(report: &'a AuditReport, id: &str) -> &'a ScenarioResult {
        report.results.iter().find(|item| item.scenario_id == id).expect("scenario exists")
    }

    #[test]
    fn frozen_fixture_matches_all_expected_audit_and_product_dispositions() {
        let report = report();
        assert_eq!(report.scenario_count, 10);
        assert_eq!(report.mismatched_expectations, 0);
        assert_eq!(report.matched_expectations, 10);
        assert_eq!(report.pass_count, 4);
        assert_eq!(report.blocked_count, 3);
        assert_eq!(report.rejected_count, 3);
        assert!(!report.optimizer_executed);
        assert!(!report.automatic_experiment_execution);
        assert!(!report.automatic_quality_release);
    }

    #[test]
    fn proxy_only_plan_cannot_accept_a_product() {
        let result = result_for(&report(), "proxy-only-abstain");
        assert_eq!(result.audit_disposition, AuditDisposition::Blocked);
        assert_eq!(result.product_decision, ProductDecision::Abstain);
    }

    #[test]
    fn low_contact_angle_does_not_override_weak_bond() {
        let result = result_for(&report(), "contact-angle-trap-reject");
        assert_eq!(result.audit_disposition, AuditDisposition::Pass);
        assert_eq!(result.product_decision, ProductDecision::RejectWindow);
    }

    #[test]
    fn immediate_pass_does_not_override_aging_reversal() {
        let result = result_for(&report(), "aging-reversal-reject");
        assert_eq!(result.product_decision, ProductDecision::RejectWindow);
    }

    #[test]
    fn packaging_barrier_loss_rejects_otherwise_strong_bond() {
        let result = result_for(&report(), "packaging-barrier-tradeoff-reject");
        assert_eq!(result.product_decision, ProductDecision::RejectWindow);
    }

    #[test]
    fn complete_durable_multifidelity_fixture_accepts_synthetically() {
        let result = result_for(&report(), "durable-multifidelity-accept-synthetic");
        assert_eq!(result.audit_disposition, AuditDisposition::Pass);
        assert_eq!(result.product_decision, ProductDecision::AcceptWindow);
    }

    #[test]
    fn shared_preparation_without_covariance_receipt_blocks() {
        let result = result_for(&report(), "unmodelled-shared-preparation-abstain");
        assert_eq!(result.audit_disposition, AuditDisposition::Blocked);
        assert_eq!(result.product_decision, ProductDecision::Abstain);
    }

    #[test]
    fn training_group_overlap_rejects_the_selection_receipt() {
        let result = result_for(&report(), "training-group-overlap-blocked");
        assert_eq!(result.audit_disposition, AuditDisposition::Rejected);
        assert_eq!(result.product_decision, ProductDecision::Abstain);
    }

    #[test]
    fn over_budget_selection_is_rejected() {
        let result = result_for(&report(), "over-budget-proposal-rejected");
        assert_eq!(result.audit_disposition, AuditDisposition::Rejected);
    }

    #[test]
    fn inadmissible_action_is_rejected() {
        let result = result_for(&report(), "inadmissible-action-abstain");
        assert_eq!(result.audit_disposition, AuditDisposition::Rejected);
    }

    #[test]
    fn inadequate_model_family_forces_other_and_abstention() {
        let result = result_for(&report(), "model-family-inadequate-abstain");
        assert_eq!(result.audit_disposition, AuditDisposition::Blocked);
        assert_eq!(result.product_decision, ProductDecision::Abstain);
    }

    #[test]
    fn fixture_cannot_promote_synthetic_outcomes_to_physical() {
        let value: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
        let mut changed = value;
        changed["physical_measurements"] = serde_json::json!(true);
        assert!(audit_fixture_json(&changed.to_string()).is_err());
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let value: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
        let mut changed = value;
        changed["scenarios"][0]["unexpected"] = serde_json::json!(true);
        assert!(audit_fixture_json(&changed.to_string()).is_err());
    }
}
