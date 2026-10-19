#![deny(unsafe_code)]
// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, fail-closed selection of synthetic display-metrology experiments.
//!
//! The selector ranks only a frozen candidate set using declared estimates and a
//! single declared resource-cost basis. It cannot validate the model behind a
//! predicted information gain, qualify an instrument, or prove that a real lab
//! would save time. Confirmatory candidates are never selected by this planner.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const EXPECTED_SCHEMA_VERSION: &str = "0.1.0";
pub const EXPECTED_SCHEMA_DRAFT: &str = "https://json-schema.org/draft/2020-12/schema";
pub const SYNTHETIC_ONLY: &str = "SYNTHETIC_TEST_ONLY";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExperimentKind {
    /// Calibration/check-standard experiment that cannot be traded away for utility.
    ProtectedControl,
    /// Exploratory experiment that may be ranked inside the frozen candidate set.
    Exploratory,
    /// Confirmatory experiment that belongs to a separately frozen analysis plan.
    Confirmatory,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ExperimentCandidate {
    pub id: String,
    pub kind: ExperimentKind,
    pub preregistered: bool,
    pub available: bool,
    pub hard_floors_satisfied: bool,
    pub protocol_digest: String,
    pub model_digest: String,
    pub estimator_digest: String,
    pub source_data_digest: String,
    pub cost_basis_id: String,
    /// Model-estimated information benefit in the declared model's decision units.
    /// This is not observed information gain and must not be reported as a result.
    pub predicted_information_gain: f64,
    /// Resource cost on one frozen basis (for example, bench-minutes).
    /// Time, currency and risk are not combined implicitly by this first version.
    pub cost_units: f64,
    pub covers: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ExperimentSelectionPlan {
    pub schema_version: String,
    pub plan_id: String,
    pub record_state: String,
    pub protocol_digest: String,
    pub model_digest: String,
    pub estimator_digest: String,
    pub cost_basis_id: String,
    pub resource_budget_units: f64,
    pub max_selected_experiments: usize,
    pub required_coverage: Vec<String>,
    pub candidates: Vec<ExperimentCandidate>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SelectionReason {
    ProtectedControl,
    ProtectedCoverage,
    InformationGainPerCost,
}


#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SelectedExperiment {
    pub candidate_id: String,
    pub reason: SelectionReason,
    pub predicted_information_gain: f64,
    pub cost_units: f64,
    pub marginal_coverage: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExcludedExperiment {
    pub candidate_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SelectionReport {
    pub plan_id: String,
    pub plan_sha256: String,
    pub schema_sha256: String,
    pub selection_policy: String,
    pub status: String,
    pub cost_basis_id: String,
    pub budget_units: f64,
    pub selected_cost_units: f64,
    pub remaining_budget_units: f64,
    pub selected: Vec<SelectedExperiment>,
    pub excluded: Vec<ExcludedExperiment>,
    pub satisfied_coverage: Vec<String>,
    pub required_coverage: Vec<String>,
    pub predicted_information_gain_sum: f64,
    pub confirmation_firewall: String,
    pub claim_ceiling: String,
}

impl SelectionReport {
    /// Machine-readable report with a hard separation between predictions and outcomes.
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "verifier": "odrh-experiment-selector",
            "verifier_version": env!("CARGO_PKG_VERSION"),
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "schema_sha256": self.schema_sha256,
            "selection_policy": self.selection_policy,
            "status": self.status,
            "cost_basis_id": self.cost_basis_id,
            "budget_units": self.budget_units,
            "selected_cost_units": self.selected_cost_units,
            "remaining_budget_units": self.remaining_budget_units,
            "selected": self.selected,
            "excluded": self.excluded,
            "satisfied_coverage": self.satisfied_coverage,
            "required_coverage": self.required_coverage,
            "predicted_information_gain_sum": self.predicted_information_gain_sum,
            "confirmation_firewall": self.confirmation_firewall,
            "claim_ceiling": self.claim_ceiling
        })
    }
}

/// Validate a candidate plan against JSON Schema and execute the deterministic selector.
///
/// # Errors
///
/// Rejects invalid schema/plan input, incompatible model/protocol/cost identities,
/// invalid digests, duplicate IDs, unregistered choices, unavailable required
/// controls, missing required coverage, and resource-budget infeasibility.
pub fn select_experiments_str(
    plan_text: &str,
    schema_text: &str,
) -> Result<SelectionReport, Vec<String>> {
    let schema: Value = match serde_json::from_str(schema_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("schema JSON parse error: {error}")]),
    };
    let plan: ExperimentSelectionPlan = match serde_json::from_str(plan_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("plan JSON parse error: {error}")]),
    };

    let mut errors = Vec::new();
    if schema.get("$schema").and_then(Value::as_str) != Some(EXPECTED_SCHEMA_DRAFT) {
        errors.push(format!("schema must declare {EXPECTED_SCHEMA_DRAFT}"));
    }
    match jsonschema::draft202012::new(&schema) {
        Ok(validator) => {
            let plan_value: Value = match serde_json::from_str(plan_text) {
                Ok(value) => value,
                Err(error) => return Err(vec![format!("plan JSON parse error: {error}")]),
            };
            for error in validator.iter_errors(&plan_value) {
                errors.push(format!("JSON Schema violation: {error}"));
            }
        }
        Err(error) => errors.push(format!("schema compilation failed: {error}")),
    }

    errors.extend(validate_plan(&plan));
    errors.sort();
    errors.dedup();
    if !errors.is_empty() {
        return Err(errors);
    }

    run_selection(&plan, plan_text, schema_text)
}

/// Validate the frozen plan's semantic requirements independently of its schema.
fn validate_plan(plan: &ExperimentSelectionPlan) -> Vec<String> {
    let mut errors = Vec::new();

    if plan.schema_version != EXPECTED_SCHEMA_VERSION {
        errors.push(format!(
            "plan schema_version must be {EXPECTED_SCHEMA_VERSION}"
        ));
    }
    if plan.record_state != SYNTHETIC_ONLY {
        errors.push(format!(
            "v0.1 selector only accepts {SYNTHETIC_ONLY} plans"
        ));
    }
    for (name, digest) in [
        ("protocol_digest", plan.protocol_digest.as_str()),
        ("model_digest", plan.model_digest.as_str()),
        ("estimator_digest", plan.estimator_digest.as_str()),
    ] {
        if !is_sha256_digest(digest) {
            errors.push(format!("plan {name} must be a 64-character SHA-256 digest"));
        }
    }
    if plan.plan_id.trim().is_empty() || plan.cost_basis_id.trim().is_empty() {
        errors.push("plan ID and cost basis ID must be nonempty".to_owned());
    }
    if !plan.resource_budget_units.is_finite() || plan.resource_budget_units <= 0.0 {
        errors.push("resource budget must be finite and positive".to_owned());
    }
    if plan.max_selected_experiments == 0 {
        errors.push("max_selected_experiments must be positive".to_owned());
    }
    if plan.required_coverage.is_empty() {
        errors.push("at least one protected coverage stratum is required".to_owned());
    }
    let coverage: BTreeSet<&str> = plan.required_coverage.iter().map(String::as_str).collect();
    if coverage.len() != plan.required_coverage.len() {
        errors.push("required coverage strata must be unique".to_owned());
    }
    if plan.required_coverage.iter().any(|item| item.trim().is_empty()) {
        errors.push("coverage stratum IDs must be nonempty".to_owned());
    }

    let mut ids = BTreeSet::new();
    let mut protected_control_count = 0;
    for candidate in &plan.candidates {
        if candidate.id.trim().is_empty() {
            errors.push("candidate IDs must be nonempty".to_owned());
        }
        if !ids.insert(candidate.id.as_str()) {
            errors.push(format!("duplicate candidate ID: {}", candidate.id));
        }
        if !candidate.preregistered {
            errors.push(format!(
                "candidate {} is not preregistered in the frozen options set",
                candidate.id
            ));
        }
        if candidate.protocol_digest != plan.protocol_digest
            || candidate.model_digest != plan.model_digest
            || candidate.estimator_digest != plan.estimator_digest
            || candidate.cost_basis_id != plan.cost_basis_id
        {
            errors.push(format!(
                "candidate {} has a protocol/model/estimator/cost-basis identity mismatch",
                candidate.id
            ));
        }
        for (field, digest) in [
            ("protocol_digest", candidate.protocol_digest.as_str()),
            ("model_digest", candidate.model_digest.as_str()),
            ("estimator_digest", candidate.estimator_digest.as_str()),
            ("source_data_digest", candidate.source_data_digest.as_str()),
        ] {
            if !is_sha256_digest(digest) {
                errors.push(format!(
                    "candidate {} {field} must be a 64-character SHA-256 digest",
                    candidate.id
                ));
            }
        }
        if candidate.cost_basis_id.trim().is_empty() {
            errors.push(format!("candidate {} cost basis must be nonempty", candidate.id));
        }
        if !candidate.predicted_information_gain.is_finite()
            || candidate.predicted_information_gain < 0.0
        {
            errors.push(format!(
                "candidate {} predicted information gain must be finite and nonnegative",
                candidate.id
            ));
        }
        if !candidate.cost_units.is_finite() || candidate.cost_units <= 0.0 {
            errors.push(format!(
                "candidate {} cost must be finite and positive",
                candidate.id
            ));
        }
        if candidate.cost_units.is_finite()
            && candidate.cost_units > 0.0
            && !(candidate.predicted_information_gain / candidate.cost_units).is_finite()
        {
            errors.push(format!(
                "candidate {} predicted information-gain-per-cost is non-finite",
                candidate.id
            ));
        }
        let candidate_coverage: BTreeSet<&str> =
            candidate.covers.iter().map(String::as_str).collect();
        if candidate_coverage.len() != candidate.covers.len()
            || candidate.covers.iter().any(|item| item.trim().is_empty())
        {
            errors.push(format!(
                "candidate {} coverage IDs must be nonempty and unique",
                candidate.id
            ));
        }
        match candidate.kind {
            ExperimentKind::ProtectedControl => {
                protected_control_count += 1;
                if !candidate.available {
                    errors.push(format!(
                        "protected control {} is unavailable",
                        candidate.id
                    ));
                }
                if !candidate.hard_floors_satisfied {
                    errors.push(format!(
                        "protected control {} fails a required hard floor",
                        candidate.id
                    ));
                }
            }
            ExperimentKind::Exploratory => {}
            ExperimentKind::Confirmatory => {}
        }
    }
    if protected_control_count == 0 {
        errors.push("at least one protected control must be included in the plan".to_owned());
    }
    errors
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn run_selection(
    plan: &ExperimentSelectionPlan,
    plan_text: &str,
    schema_text: &str,
) -> Result<SelectionReport, Vec<String>> {
    let mut errors = Vec::new();
    let mut selected = Vec::<SelectedExperiment>::new();
    let mut selected_ids = BTreeSet::<String>::new();
    let mut covered = BTreeSet::<String>::new();
    let mut used_budget = 0.0_f64;

    let mut controls: Vec<&ExperimentCandidate> = plan
        .candidates
        .iter()
        .filter(|candidate| candidate.kind == ExperimentKind::ProtectedControl)
        .collect();
    controls.sort_by(|left, right| left.id.cmp(&right.id));

    for candidate in controls {
        if selected.len() >= plan.max_selected_experiments
            || used_budget + candidate.cost_units > plan.resource_budget_units
        {
            errors.push(format!(
                "required protected control {} does not fit within the preregistered selection budget",
                candidate.id
            ));
            continue;
        }
        add_selection(
            candidate,
            SelectionReason::ProtectedControl,
            &mut selected,
            &mut selected_ids,
            &mut covered,
            &mut used_budget,
        );
    }
    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    let required: BTreeSet<String> = plan.required_coverage.iter().cloned().collect();
    while !required.is_subset(&covered) {
        let uncovered: BTreeSet<String> = required.difference(&covered).cloned().collect();
        let mut eligible: Vec<(&ExperimentCandidate, usize, f64, f64)> = plan
            .candidates
            .iter()
            .filter(|candidate| candidate.kind == ExperimentKind::Exploratory)
            .filter(|candidate| candidate.available && candidate.hard_floors_satisfied)
            .filter(|candidate| !selected_ids.contains(&candidate.id))
            .filter(|candidate| selected.len() < plan.max_selected_experiments)
            .filter(|candidate| used_budget + candidate.cost_units <= plan.resource_budget_units)
            .filter_map(|candidate| {
                let marginal: usize = candidate
                    .covers
                    .iter()
                    .filter(|stratum| uncovered.contains(*stratum))
                    .count();
                if marginal == 0 {
                    return None;
                }
                let coverage_per_cost = marginal as f64 / candidate.cost_units;
                let gain_per_cost = candidate.predicted_information_gain / candidate.cost_units;
                Some((candidate, marginal, coverage_per_cost, gain_per_cost))
            })
            .collect();
        eligible.sort_by(|left, right| {
            right
                .2
                .total_cmp(&left.2)
                .then_with(|| right.3.total_cmp(&left.3))
                .then_with(|| left.0.id.cmp(&right.0.id))
        });
        let Some((candidate, _, _, _)) = eligible.first().copied() else {
            return Err(vec![format!(
                "cannot satisfy protected coverage within the frozen budget; uncovered strata: {}",
                uncovered.into_iter().collect::<Vec<_>>().join(", ")
            )]);
        };
        add_selection(
            candidate,
            SelectionReason::ProtectedCoverage,
            &mut selected,
            &mut selected_ids,
            &mut covered,
            &mut used_budget,
        );
    }

    let mut optional: Vec<&ExperimentCandidate> = plan
        .candidates
        .iter()
        .filter(|candidate| candidate.kind == ExperimentKind::Exploratory)
        .filter(|candidate| candidate.available && candidate.hard_floors_satisfied)
        .filter(|candidate| candidate.predicted_information_gain > 0.0)
        .filter(|candidate| !selected_ids.contains(&candidate.id))
        .collect();
    optional.sort_by(|left, right| {
        let left_ratio = left.predicted_information_gain / left.cost_units;
        let right_ratio = right.predicted_information_gain / right.cost_units;
        right_ratio
            .total_cmp(&left_ratio)
            .then_with(|| left.id.cmp(&right.id))
    });
    for candidate in optional {
        if selected.len() >= plan.max_selected_experiments {
            break;
        }
        if used_budget + candidate.cost_units > plan.resource_budget_units {
            continue;
        }
        add_selection(
            candidate,
            SelectionReason::InformationGainPerCost,
            &mut selected,
            &mut selected_ids,
            &mut covered,
            &mut used_budget,
        );
    }

    let mut excluded = Vec::new();
    for candidate in &plan.candidates {
        if selected_ids.contains(&candidate.id) {
            continue;
        }
        let reason = match candidate.kind {
            ExperimentKind::Confirmatory => "CONFIRMATORY_FIREWALL",
            ExperimentKind::ProtectedControl => "PROTECTED_CONTROL_SELECTION_FAILED",
            ExperimentKind::Exploratory if !candidate.available => "UNAVAILABLE",
            ExperimentKind::Exploratory if !candidate.hard_floors_satisfied => {
                "HARD_FLOOR_NOT_SATISFIED"
            }
            ExperimentKind::Exploratory
                if candidate.predicted_information_gain <= 0.0 =>
            {
                "NO_POSITIVE_PREDICTED_INFORMATION_GAIN"
            }
            ExperimentKind::Exploratory => "BUDGET_OR_MAX_SELECTION_LIMIT",
        };
        excluded.push(ExcludedExperiment {
            candidate_id: candidate.id.clone(),
            reason: reason.to_owned(),
        });
    }
    excluded.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let mut coverage_sorted: Vec<String> = covered.into_iter().collect();
    coverage_sorted.sort();
    let predicted_information_gain_sum: f64 = selected
        .iter()
        .map(|selection| selection.predicted_information_gain)
        .sum();
    if !used_budget.is_finite() || !predicted_information_gain_sum.is_finite() {
        return Err(vec!["selection totals are non-finite".to_owned()]);
    }

    Ok(SelectionReport {
        plan_id: plan.plan_id.clone(),
        plan_sha256: sha256_hex(plan_text.as_bytes()),
        schema_sha256: sha256_hex(schema_text.as_bytes()),
        selection_policy: "PROTECTED_CONTROLS_THEN_COVERAGE_THEN_GREEDY_INFORMATION_GAIN_PER_COST".to_owned(),
        status: "SYNTHETIC_SELECTION_ONLY".to_owned(),
        cost_basis_id: plan.cost_basis_id.clone(),
        budget_units: plan.resource_budget_units,
        selected_cost_units: used_budget,
        remaining_budget_units: plan.resource_budget_units - used_budget,
        selected,
        excluded,
        satisfied_coverage: coverage_sorted,
        required_coverage: plan.required_coverage.clone(),
        predicted_information_gain_sum,
        confirmation_firewall: "CONFIRMATORY_CANDIDATES_ARE_NEVER_SELECTED; EXPLORATORY_SELECTION_DOES_NOT_VALIDATE A CONFIRMATORY CLAIM".to_owned(),
        claim_ceiling: "Synthetic planning heuristic only; no actual uncertainty reduction, measurement savings or physical qualification is demonstrated.".to_owned(),
    })
}

fn add_selection(
    candidate: &ExperimentCandidate,
    reason: SelectionReason,
    selected: &mut Vec<SelectedExperiment>,
    selected_ids: &mut BTreeSet<String>,
    covered: &mut BTreeSet<String>,
    used_budget: &mut f64,
) {
    let mut marginal_coverage = Vec::new();
    for stratum in &candidate.covers {
        if covered.insert(stratum.clone()) {
            marginal_coverage.push(stratum.clone());
        }
    }
    marginal_coverage.sort();
    selected.push(SelectedExperiment {
        candidate_id: candidate.id.clone(),
        reason,
        predicted_information_gain: candidate.predicted_information_gain,
        cost_units: candidate.cost_units,
        marginal_coverage,
    });
    selected_ids.insert(candidate.id.clone());
    *used_budget += candidate.cost_units;
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::{
        ExperimentCandidate, ExperimentKind, ExperimentSelectionPlan, SelectionReason,
        select_experiments_str,
    };
    use serde_json::Value;

    const PLAN_TEXT: &str =
        include_str!("../../../docs/engineering/odrh-active-experiment-plan-synthetic-v0.1.json");
    const SCHEMA_TEXT: &str =
        include_str!("../../../docs/engineering/odrh-active-experiment-plan.schema.json");

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn candidate(
        id: &str,
        kind: ExperimentKind,
        cost: f64,
        gain: f64,
        covers: &[&str],
    ) -> ExperimentCandidate {
        ExperimentCandidate {
            id: id.to_owned(),
            kind,
            preregistered: true,
            available: true,
            hard_floors_satisfied: true,
            protocol_digest: digest('a'),
            model_digest: digest('b'),
            estimator_digest: digest('c'),
            source_data_digest: digest('d'),
            cost_basis_id: "BENCH_MINUTES".to_owned(),
            predicted_information_gain: gain,
            cost_units: cost,
            covers: covers.iter().map(|item| (*item).to_owned()).collect(),
        }
    }

    fn test_plan(candidates: Vec<ExperimentCandidate>) -> ExperimentSelectionPlan {
        ExperimentSelectionPlan {
            schema_version: "0.1.0".to_owned(),
            plan_id: "ODRH-ACTIVE-TEST-001".to_owned(),
            record_state: "SYNTHETIC_TEST_ONLY".to_owned(),
            protocol_digest: digest('a'),
            model_digest: digest('b'),
            estimator_digest: digest('c'),
            cost_basis_id: "BENCH_MINUTES".to_owned(),
            resource_budget_units: 10.0,
            max_selected_experiments: 5,
            required_coverage: vec!["STATIC_APL".to_owned(), "TEMPORAL".to_owned()],
            candidates,
        }
    }

    #[test]
    fn checked_in_synthetic_plan_is_valid_and_firewalled() {
        let report = select_experiments_str(PLAN_TEXT, SCHEMA_TEXT)
            .expect("known synthetic plan must validate");
        assert_eq!(report.status, "SYNTHETIC_SELECTION_ONLY");
        assert!(report.selected.iter().all(|item| {
            !item.candidate_id.starts_with("CONFIRM")
        }));
        assert_eq!(report.satisfied_coverage, ["STATIC_APL", "TEMPORAL"]);
        assert!(report.selected_cost_units <= report.budget_units);
        assert_eq!(
            report.confirmation_firewall,
            "CONFIRMATORY_CANDIDATES_ARE_NEVER_SELECTED; EXPLORATORY_SELECTION_DOES_NOT_VALIDATE A CONFIRMATORY CLAIM"
        );
        assert_eq!(
            report.to_json()["claim_ceiling"].as_str(),
            Some("Synthetic planning heuristic only; no actual uncertainty reduction, measurement savings or physical qualification is demonstrated.")
        );
    }

    #[test]
    fn protected_controls_are_selected_before_exploratory_options() {
        let plan = test_plan(vec![
            candidate("OPTION-B", ExperimentKind::Exploratory, 1.0, 4.0, &["STATIC_APL"]),
            candidate("CAL-CHECK", ExperimentKind::ProtectedControl, 1.0, 0.0, &[]),
            candidate("OPTION-A", ExperimentKind::Exploratory, 1.0, 1.0, &["TEMPORAL"]),
        ]);
        let value = serde_json::to_string(&plan).expect("plan serializes");
        let report = select_experiments_str(&value, SCHEMA_TEXT)
            .expect("protected control and coverage fit");
        assert_eq!(report.selected[0].candidate_id, "CAL-CHECK");
        assert_eq!(report.selected[0].reason, SelectionReason::ProtectedControl);
    }

    #[test]
    fn confirmatory_candidates_are_never_selected_even_with_high_score() {
        let plan = test_plan(vec![
            candidate("CAL-CHECK", ExperimentKind::ProtectedControl, 1.0, 0.0, &[]),
            candidate("EXPLORATORY", ExperimentKind::Exploratory, 2.0, 1.0, &["STATIC_APL", "TEMPORAL"]),
            candidate("CONFIRM-HIGH-SCORE", ExperimentKind::Confirmatory, 0.1, 10_000.0, &["STATIC_APL", "TEMPORAL"]),
        ]);
        let value = serde_json::to_string(&plan).expect("plan serializes");
        let report = select_experiments_str(&value, SCHEMA_TEXT)
            .expect("exploratory option covers protected strata");
        assert!(report.selected.iter().all(|entry| entry.candidate_id != "CONFIRM-HIGH-SCORE"));
        assert!(report.excluded.iter().any(|entry| {
            entry.candidate_id == "CONFIRM-HIGH-SCORE" && entry.reason == "CONFIRMATORY_FIREWALL"
        }));
    }

    #[test]
    fn mismatched_protocol_model_or_cost_basis_is_rejected() {
        let mut plan = test_plan(vec![
            candidate("CAL-CHECK", ExperimentKind::ProtectedControl, 1.0, 0.0, &[]),
            candidate("OPTION", ExperimentKind::Exploratory, 1.0, 1.0, &["STATIC_APL", "TEMPORAL"]),
        ]);
        plan.candidates[1].cost_basis_id = "DOLLARS".to_owned();
        let value = serde_json::to_string(&plan).expect("mutated plan serializes");
        let errors = select_experiments_str(&value, SCHEMA_TEXT)
            .expect_err("incompatible cost bases cannot be ranked together");
        assert!(errors.iter().any(|e| e.contains("identity mismatch")));
    }

    #[test]
    fn inadequate_budget_cannot_skip_required_control_or_coverage() {
        let mut plan = test_plan(vec![
            candidate("CAL-CHECK", ExperimentKind::ProtectedControl, 6.0, 0.0, &[]),
            candidate("STATIC", ExperimentKind::Exploratory, 3.0, 1.0, &["STATIC_APL"]),
            candidate("TEMPORAL", ExperimentKind::Exploratory, 3.0, 1.0, &["TEMPORAL"]),
        ]);
        plan.resource_budget_units = 5.0;
        let value = serde_json::to_string(&plan).expect("mutated plan serializes");
        assert!(select_experiments_str(&value, SCHEMA_TEXT).is_err());
    }

    #[test]
    fn duplicate_candidate_ids_and_non_preregistered_options_are_rejected() {
        let mut plan = test_plan(vec![
            candidate("CAL-CHECK", ExperimentKind::ProtectedControl, 1.0, 0.0, &[]),
            candidate("OPTION", ExperimentKind::Exploratory, 2.0, 1.0, &["STATIC_APL", "TEMPORAL"]),
        ]);
        plan.candidates[1].id = "CAL-CHECK".to_owned();
        plan.candidates[1].preregistered = false;
        let value = serde_json::to_string(&plan).expect("mutated plan serializes");
        let errors = select_experiments_str(&value, SCHEMA_TEXT)
            .expect_err("duplicate and non-preregistered candidate must be rejected");
        assert!(errors.iter().any(|e| e.contains("duplicate candidate ID")));
        assert!(errors.iter().any(|e| e.contains("not preregistered")));
    }

    #[test]
    fn invalid_digest_or_model_claim_is_rejected() {
        let mut plan: Value =
            serde_json::from_str(PLAN_TEXT).expect("fixture plan must parse");
        plan["candidates"][0]["estimator_digest"] = Value::String("not-a-digest".to_owned());
        let value = serde_json::to_string(&plan).expect("mutated plan serializes");
        assert!(select_experiments_str(&value, SCHEMA_TEXT)
            .expect_err("invalid model digest must fail")
            .iter()
            .any(|error| error.contains("estimator_digest")));
    }
}
