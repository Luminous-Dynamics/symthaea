// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finite-set worst-case EVSI for uncertain priors, assay models and costs.
//!
//! This is scenario-robust optimization over an explicitly declared finite set,
//! not a general distributionally-robust optimization theorem. Scenario weights
//! are descriptive and are never used in the minimax objective.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::decision_loss_metrology_policy::{
    MetrologyCandidateEvsiReportV1, MetrologyCandidateStatusV1, MetrologyCandidateV1,
    MetrologyDecisionLossModelV1, MetrologyDecisionLossPolicyV1, validate_policy,
    score_candidate,
};

pub const ROBUST_EVSI_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.robust-evsi-input.v1";
pub const ROBUST_EVSI_SCENARIO_SCHEMA_V1: &str =
    "luminous.semi-bench.robust-evsi-scenario.v1";
pub const ROBUST_EVSI_CANDIDATE_SCHEMA_V1: &str =
    "luminous.semi-bench.robust-evsi-candidate.v1";
pub const ROBUST_EVSI_CANDIDATE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.robust-evsi-candidate-report.v1";
pub const ROBUST_EVSI_PORTFOLIO_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.robust-evsi-portfolio-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustEvsiCandidateStatusV1 {
    Eligible,
    ReadinessUnknown,
    ReadinessRejected,
    DecisionDeadlineMissed,
    IncompleteScenarioSet,
    ScenarioSetRejected,
    NonPositiveWorstCaseNetValue,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustEvsiScenarioV1 {
    pub schema: String,
    pub scenario_id: String,
    /// Descriptive metadata only in minimax mode; never used to weight the objective.
    pub scenario_weight: Option<f64>,
    pub measurement_cost: f64,
    pub delay_cost_per_second: f64,
    pub decision_loss_model: MetrologyDecisionLossModelV1,
    pub source_provenance_sha256: String,
    pub scenario_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustEvsiCandidateV1 {
    pub schema: String,
    pub candidate: MetrologyCandidateV1,
    pub central_scenario_id: String,
    pub scenarios: Vec<RobustEvsiScenarioV1>,
    pub scenario_set_complete: bool,
    pub scenario_set_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustEvsiInputV1 {
    pub schema: String,
    pub input_id: String,
    pub base_policy: MetrologyDecisionLossPolicyV1,
    pub maximum_scenarios_per_candidate: u32,
    pub uncertainty_set_complete: bool,
    pub candidates: Vec<RobustEvsiCandidateV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustEvsiScenarioResultV1 {
    pub scenario_id: String,
    pub scenario_weight: Option<f64>,
    pub source_provenance_sha256: String,
    pub scenario_sha256: String,
    pub expected_loss_without_measurement: Option<f64>,
    pub expected_loss_after_measurement: Option<f64>,
    pub gross_expected_loss_reduction: Option<f64>,
    pub measurement_cost: f64,
    pub turnaround_cost: f64,
    pub net_expected_loss_reduction: Option<f64>,
    pub status: MetrologyCandidateStatusV1,
    pub score_report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustEvsiCandidateReportV1 {
    pub schema: String,
    pub candidate_id: String,
    pub group_key: String,
    pub status: RobustEvsiCandidateStatusV1,
    pub scenario_set_sha256: String,
    pub scenario_results: Vec<RobustEvsiScenarioResultV1>,
    pub worst_case_scenario_id: Option<String>,
    pub central_scenario_id: Option<String>,
    pub best_case_scenario_id: Option<String>,
    pub worst_case_net_expected_loss_reduction: Option<f64>,
    pub central_net_expected_loss_reduction: Option<f64>,
    pub best_case_net_expected_loss_reduction: Option<f64>,
    pub reason_codes: Vec<String>,
    pub report_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustEvsiPortfolioStatusV1 {
    Feasible,
    InfeasibleCoverage,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustEvsiPortfolioReportV1 {
    pub schema: String,
    pub input_id: String,
    pub policy_sha256: String,
    pub status: RobustEvsiPortfolioStatusV1,
    pub candidate_reports: Vec<RobustEvsiCandidateReportV1>,
    pub selected_candidate_ids: Vec<String>,
    pub total_measurement_cost_units: u32,
    pub remaining_measurement_budget_units: u32,
    pub selected_group_counts: BTreeMap<String, u32>,
    pub worst_case_expected_net_loss_reduction: f64,
    pub reason_codes: Vec<String>,
    pub uncertainty_set_complete: bool,
    pub source_authenticity_proven: bool,
    pub fab_costs_empirically_established: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct RobustEvsiErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), RobustEvsiErrorV1> {
    if value.trim().is_empty() {
        return Err(RobustEvsiErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, RobustEvsiErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        RobustEvsiErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn scenario_digest(scenario: &RobustEvsiScenarioV1) -> Result<String, RobustEvsiErrorV1> {
    let mut canonical = scenario.clone();
    canonical.scenario_sha256.clear();
    json_sha256(&canonical)
}

fn scenario_set_digest(candidate: &RobustEvsiCandidateV1) -> Result<String, RobustEvsiErrorV1> {
    let mut canonical = candidate.clone();
    canonical.scenario_set_sha256.clear();
    canonical
        .scenarios
        .sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    json_sha256(&canonical)
}

fn candidate_report_digest(
    report: &RobustEvsiCandidateReportV1,
) -> Result<String, RobustEvsiErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn portfolio_report_digest(
    report: &RobustEvsiPortfolioReportV1,
) -> Result<String, RobustEvsiErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn model_identity_signature(
    model: &MetrologyDecisionLossModelV1,
) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
    (
        model.states.iter().map(|v| v.state_id.clone()).collect(),
        model.actions.iter().map(|v| v.action_id.clone()).collect(),
        model.assay.outcomes.iter().map(|v| v.outcome_id.clone()).collect(),
    )
}

/// Seals a scenario after its prior, loss matrix, assay likelihoods and costs are fixed.
pub fn seal_robust_evsi_scenario_v1(
    scenario: &RobustEvsiScenarioV1,
) -> Result<String, RobustEvsiErrorV1> {
    scenario_digest(scenario)
}

/// Seals a candidate's complete finite uncertainty set in scenario-ID order.
pub fn seal_robust_evsi_candidate_v1(
    candidate: &RobustEvsiCandidateV1,
) -> Result<String, RobustEvsiErrorV1> {
    scenario_set_digest(candidate)
}

fn unknown_candidate_report(
    candidate: &RobustEvsiCandidateV1,
    reason: &str,
) -> Result<RobustEvsiCandidateReportV1, RobustEvsiErrorV1> {
    let mut report = RobustEvsiCandidateReportV1 {
        schema: ROBUST_EVSI_CANDIDATE_REPORT_SCHEMA_V1.into(),
        candidate_id: candidate.candidate.candidate_id.clone(),
        group_key: candidate.candidate.group_key.clone(),
        status: RobustEvsiCandidateStatusV1::IncompleteScenarioSet,
        scenario_set_sha256: candidate.scenario_set_sha256.clone(),
        scenario_results: Vec::new(),
        worst_case_scenario_id: None,
        central_scenario_id: None,
        best_case_scenario_id: None,
        worst_case_net_expected_loss_reduction: None,
        central_net_expected_loss_reduction: None,
        best_case_net_expected_loss_reduction: None,
        reason_codes: vec![reason.into()],
        report_sha256: String::new(),
    };
    report.report_sha256 = candidate_report_digest(&report)?;
    Ok(report)
}

fn score_robust_candidate(
    robust: &RobustEvsiCandidateV1,
    base_policy: &MetrologyDecisionLossPolicyV1,
    max_scenarios: usize,
    uncertainty_set_complete: bool,
) -> Result<RobustEvsiCandidateReportV1, RobustEvsiErrorV1> {
    require_text(&robust.candidate.candidate_id, "candidate.candidate_id")?;
    require_text(&robust.central_scenario_id, "central_scenario_id")?;
    if robust.schema != ROBUST_EVSI_CANDIDATE_SCHEMA_V1
        || robust.scenario_set_sha256 != scenario_set_digest(robust)?
        || !valid_sha256(&robust.scenario_set_sha256)
    {
        return Err(RobustEvsiErrorV1(format!(
            "candidate {} scenario-set digest is invalid",
            robust.candidate.candidate_id
        )));
    }
    if !uncertainty_set_complete || !robust.scenario_set_complete {
        return unknown_candidate_report(robust, "uncertainty_scenario_set_incomplete");
    }
    if robust.scenarios.is_empty() || robust.scenarios.len() > max_scenarios {
        return Err(RobustEvsiErrorV1(format!(
            "candidate {} scenario count is empty or exceeds its declared cap",
            robust.candidate.candidate_id
        )));
    }

    let mut scenario_ids = BTreeSet::new();
    let mut any_weight = false;
    let mut all_weight = true;
    let mut weight_sum = 0.0;
    let mut signature = None;
    let mut results = Vec::with_capacity(robust.scenarios.len());

    for scenario in &robust.scenarios {
        if scenario.schema != ROBUST_EVSI_SCENARIO_SCHEMA_V1
            || !valid_sha256(&scenario.source_provenance_sha256)
            || !valid_sha256(&scenario.scenario_sha256)
            || scenario_digest(scenario)? != scenario.scenario_sha256
        {
            return Err(RobustEvsiErrorV1(format!(
                "candidate {} scenario {} has invalid schema/provenance/digest",
                robust.candidate.candidate_id, scenario.scenario_id
            )));
        }
        require_text(&scenario.scenario_id, "scenario.scenario_id")?;
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(RobustEvsiErrorV1(format!(
                "candidate {} has duplicate scenario IDs",
                robust.candidate.candidate_id
            )));
        }
        if !scenario.measurement_cost.is_finite()
            || scenario.measurement_cost < 0.0
            || !scenario.delay_cost_per_second.is_finite()
            || scenario.delay_cost_per_second < 0.0
        {
            return Err(RobustEvsiErrorV1(format!(
                "candidate {} scenario {} has invalid cost parameters",
                robust.candidate.candidate_id, scenario.scenario_id
            )));
        }
        match scenario.scenario_weight {
            Some(weight) if weight.is_finite() && weight >= 0.0 => {
                any_weight = true;
                weight_sum += weight;
            }
            Some(_) => {
                return Err(RobustEvsiErrorV1(format!(
                    "candidate {} scenario {} has invalid descriptive weight",
                    robust.candidate.candidate_id, scenario.scenario_id
                )));
            }
            None => all_weight = false,
        }

        let current_signature = model_identity_signature(&scenario.decision_loss_model);
        if let Some(expected) = &signature {
            if expected != &current_signature {
                return Err(RobustEvsiErrorV1(format!(
                    "candidate {} scenario {} changes state/action/outcome identities",
                    robust.candidate.candidate_id, scenario.scenario_id
                )));
            }
        } else {
            signature = Some(current_signature);
        }

        let mut scenario_candidate = robust.candidate.clone();
        scenario_candidate.measurement_cost = scenario.measurement_cost;
        scenario_candidate.decision_loss_model = scenario.decision_loss_model.clone();
        let mut scenario_policy = base_policy.clone();
        scenario_policy.delay_cost_per_second = scenario.delay_cost_per_second;
        let score = score_candidate(&scenario_candidate, &scenario_policy)
            .map_err(|error| RobustEvsiErrorV1(error.to_string()))?;

        results.push(RobustEvsiScenarioResultV1 {
            scenario_id: scenario.scenario_id.clone(),
            scenario_weight: scenario.scenario_weight,
            source_provenance_sha256: scenario.source_provenance_sha256.clone(),
            scenario_sha256: scenario.scenario_sha256.clone(),
            expected_loss_without_measurement: score.expected_loss_without_measurement,
            expected_loss_after_measurement: score.expected_loss_after_measurement,
            gross_expected_loss_reduction: score.gross_expected_loss_reduction,
            measurement_cost: score.measurement_cost,
            turnaround_cost: score.turnaround_cost,
            net_expected_loss_reduction: score.net_expected_loss_reduction,
            status: score.status,
            score_report_sha256: score.report_sha256,
        });
    }

    if any_weight && (!all_weight || (weight_sum - 1.0).abs() > 1.0e-9) {
        return Err(RobustEvsiErrorV1(
            "scenario weights must be absent for all scenarios or sum to one".into(),
        ));
    }
    if !scenario_ids.contains(robust.central_scenario_id.as_str()) {
        return Err(RobustEvsiErrorV1(format!(
            "candidate {} central scenario is absent",
            robust.candidate.candidate_id
        )));
    }

    results.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let invalid_status = results.iter().find_map(|result| match result.status {
        MetrologyCandidateStatusV1::ReadinessUnknown => {
            Some(RobustEvsiCandidateStatusV1::ReadinessUnknown)
        }
        MetrologyCandidateStatusV1::ReadinessRejected => {
            Some(RobustEvsiCandidateStatusV1::ReadinessRejected)
        }
        MetrologyCandidateStatusV1::DecisionDeadlineMissed => {
            Some(RobustEvsiCandidateStatusV1::DecisionDeadlineMissed)
        }
        _ => None,
    });
    if let Some(status) = invalid_status {
        let mut report = RobustEvsiCandidateReportV1 {
            schema: ROBUST_EVSI_CANDIDATE_REPORT_SCHEMA_V1.into(),
            candidate_id: robust.candidate.candidate_id.clone(),
            group_key: robust.candidate.group_key.clone(),
            status,
            scenario_set_sha256: robust.scenario_set_sha256.clone(),
            scenario_results: results,
            worst_case_scenario_id: None,
            central_scenario_id: Some(robust.central_scenario_id.clone()),
            best_case_scenario_id: None,
            worst_case_net_expected_loss_reduction: None,
            central_net_expected_loss_reduction: None,
            best_case_net_expected_loss_reduction: None,
            reason_codes: vec!["at_least_one_scenario_is_not_measurement_eligible".into()],
            report_sha256: String::new(),
        };
        report.report_sha256 = candidate_report_digest(&report)?;
        return Ok(report);
    }

    let mut scored: Vec<&RobustEvsiScenarioResultV1> = results
        .iter()
        .filter(|result| result.net_expected_loss_reduction.is_some())
        .collect();
    if scored.len() != results.len() {
        return Err(RobustEvsiErrorV1(
            "eligible scenario is missing a net EVSI value".into(),
        ));
    }
    scored.sort_by(|left, right| {
        left.net_expected_loss_reduction
            .unwrap()
            .total_cmp(&right.net_expected_loss_reduction.unwrap())
            .then_with(|| left.scenario_id.cmp(&right.scenario_id))
    });
    let worst = scored.first().unwrap();
    let best = scored.last().unwrap();
    let central = results
        .iter()
        .find(|result| result.scenario_id == robust.central_scenario_id)
        .unwrap();
    let worst_value = worst.net_expected_loss_reduction.unwrap();
    let status = if worst_value > 0.0 {
        RobustEvsiCandidateStatusV1::Eligible
    } else {
        RobustEvsiCandidateStatusV1::NonPositiveWorstCaseNetValue
    };
    let mut report = RobustEvsiCandidateReportV1 {
        schema: ROBUST_EVSI_CANDIDATE_REPORT_SCHEMA_V1.into(),
        candidate_id: robust.candidate.candidate_id.clone(),
        group_key: robust.candidate.group_key.clone(),
        status,
        scenario_set_sha256: robust.scenario_set_sha256.clone(),
        scenario_results: results,
        worst_case_scenario_id: Some(worst.scenario_id.clone()),
        central_scenario_id: Some(robust.central_scenario_id.clone()),
        best_case_scenario_id: Some(best.scenario_id.clone()),
        worst_case_net_expected_loss_reduction: Some(worst_value),
        central_net_expected_loss_reduction: central.net_expected_loss_reduction,
        best_case_net_expected_loss_reduction: best.net_expected_loss_reduction,
        reason_codes: if status == RobustEvsiCandidateStatusV1::Eligible {
            Vec::new()
        } else {
            vec!["worst_case_net_expected_loss_reduction_not_positive".into()]
        },
        report_sha256: String::new(),
    };
    report.report_sha256 = candidate_report_digest(&report)?;
    Ok(report)
}

/// Optimize a portfolio against the minimum net EVSI over each candidate's
/// complete finite scenario set. Scenario weights are not used as objective
/// weights.
pub fn optimize_robust_metrology_evsi_portfolio_v1(
    input: &RobustEvsiInputV1,
) -> Result<RobustEvsiPortfolioReportV1, RobustEvsiErrorV1> {
    if input.schema != ROBUST_EVSI_INPUT_SCHEMA_V1 {
        return Err(RobustEvsiErrorV1("robust EVSI input schema is unsupported".into()));
    }
    require_text(&input.input_id, "input.input_id")?;
    validate_policy(&input.base_policy).map_err(|error| RobustEvsiErrorV1(error.to_string()))?;
    if input.maximum_scenarios_per_candidate == 0 || input.maximum_scenarios_per_candidate > 32 {
        return Err(RobustEvsiErrorV1("maximum scenario count must be in 1..=32".into()));
    }
    if input.candidates.len() > input.base_policy.maximum_exact_candidates as usize
        || input.candidates.len() > 20
    {
        return Err(RobustEvsiErrorV1(
            "robust portfolio exceeds the exact candidate cap".into(),
        ));
    }

    let mut candidates = input.candidates.clone();
    candidates.sort_by(|left, right| {
        left.candidate.candidate_id.cmp(&right.candidate.candidate_id)
    });
    if candidates.windows(2).any(|pair| {
        pair[0].candidate.candidate_id == pair[1].candidate.candidate_id
    }) {
        return Err(RobustEvsiErrorV1("duplicate robust candidate ID".into()));
    }

    let mut candidate_reports = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        candidate_reports.push(score_robust_candidate(
            candidate,
            &input.base_policy,
            input.maximum_scenarios_per_candidate as usize,
            input.uncertainty_set_complete,
        )?);
    }

    if !input.uncertainty_set_complete
        || candidate_reports
            .iter()
            .any(|report| report.status == RobustEvsiCandidateStatusV1::IncompleteScenarioSet)
    {
        return make_unknown_portfolio(input, candidate_reports, "uncertainty_set_incomplete");
    }

    let n = candidate_reports.len();
    let mut found = false;
    let mut best_value = f64::NEG_INFINITY;
    let mut best_ids = Vec::new();
    let mut best_cost = 0_u32;
    let mut best_groups = BTreeMap::new();

    for mask in 0_u64..(1_u64 << n) {
        let mut value = 0.0;
        let mut cost = 0_u32;
        let mut ids = Vec::new();
        let mut groups: BTreeMap<String, u32> = BTreeMap::new();
        let mut feasible = true;

        for (index, report) in candidate_reports.iter().enumerate() {
            if (mask & (1_u64 << index)) == 0 {
                continue;
            }
            if !matches!(
                report.status,
                RobustEvsiCandidateStatusV1::Eligible
                    | RobustEvsiCandidateStatusV1::NonPositiveWorstCaseNetValue
            ) {
                feasible = false;
                break;
            }
            let Some(next_cost) = cost
                .checked_add(candidates[index].candidate.measurement_cost_units)
            else {
                feasible = false;
                break;
            };
            if next_cost > input.base_policy.measurement_budget_units {
                feasible = false;
                break;
            }
            cost = next_cost;
            value += report
                .worst_case_net_expected_loss_reduction
                .unwrap_or(0.0);
            ids.push(report.candidate_id.clone());
            *groups.entry(report.group_key.clone()).or_default() += 1;
        }

        if !feasible
            || input
                .base_policy
                .minimum_group_coverage
                .iter()
                .any(|(group, minimum)| groups.get(group).copied().unwrap_or(0) < *minimum)
        {
            continue;
        }

        if !found
            || value > best_value + 1.0e-10
            || ((value - best_value).abs() <= 1.0e-10 && ids < best_ids)
        {
            found = true;
            best_value = value;
            best_ids = ids;
            best_cost = cost;
            best_groups = groups;
        }
    }

    let mut report = RobustEvsiPortfolioReportV1 {
        schema: ROBUST_EVSI_PORTFOLIO_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        policy_sha256: json_sha256(&input.base_policy)?,
        status: if found {
            RobustEvsiPortfolioStatusV1::Feasible
        } else {
            RobustEvsiPortfolioStatusV1::InfeasibleCoverage
        },
        candidate_reports,
        selected_candidate_ids: best_ids,
        total_measurement_cost_units: best_cost,
        remaining_measurement_budget_units: input
            .base_policy
            .measurement_budget_units
            .saturating_sub(best_cost),
        selected_group_counts: best_groups,
        worst_case_expected_net_loss_reduction: if found { best_value } else { 0.0 },
        reason_codes: if found {
            Vec::new()
        } else {
            vec!["robust_portfolio_infeasible_under_budget_and_group_coverage".into()]
        },
        uncertainty_set_complete: input.uncertainty_set_complete,
        source_authenticity_proven: false,
        fab_costs_empirically_established: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = portfolio_report_digest(&report)?;
    Ok(report)
}

fn make_unknown_portfolio(
    input: &RobustEvsiInputV1,
    candidate_reports: Vec<RobustEvsiCandidateReportV1>,
    reason: &str,
) -> Result<RobustEvsiPortfolioReportV1, RobustEvsiErrorV1> {
    let mut report = RobustEvsiPortfolioReportV1 {
        schema: ROBUST_EVSI_PORTFOLIO_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        policy_sha256: json_sha256(&input.base_policy)?,
        status: MetrologyPortfolioStatusV1::InfeasibleCoverage,
        candidate_reports,
        selected_candidate_ids: Vec::new(),
        total_measurement_cost_units: 0,
        remaining_measurement_budget_units: input.base_policy.measurement_budget_units,
        selected_group_counts: BTreeMap::new(),
        worst_case_expected_net_loss_reduction: 0.0,
        reason_codes: vec![reason.into()],
        uncertainty_set_complete: false,
        source_authenticity_proven: false,
        fab_costs_empirically_established: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = portfolio_report_digest(&report)?;
    Ok(report)
}
