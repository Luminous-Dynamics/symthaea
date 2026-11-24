// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Decision-loss-aware physical-metrology acquisition.
//!
//! Scores the expected value of sample information (EVSI) under explicit
//! state priors, action losses and assay likelihoods, then solves a small
//! exact portfolio problem subject to metrology capacity and group quotas.
//! All economics are caller-declared; this module does not infer fab costs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const METROLOGY_DECISION_LOSS_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.metrology-decision-loss-policy.v1";
pub const METROLOGY_DECISION_LOSS_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.metrology-decision-loss-input.v1";
pub const METROLOGY_ASSAY_MODEL_SCHEMA_V1: &str = "luminous.semi-bench.metrology-assay-model.v1";
pub const METROLOGY_READINESS_SCHEMA_V1: &str = "luminous.semi-bench.metrology-readiness.v1";
pub const METROLOGY_CANDIDATE_SCHEMA_V1: &str = "luminous.semi-bench.metrology-candidate.v1";
pub const METROLOGY_EVSI_REPORT_SCHEMA_V1: &str = "luminous.semi-bench.metrology-evsi-report.v1";
pub const METROLOGY_PORTFOLIO_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.metrology-portfolio-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetrologyReadinessStatusV1 {
    Qualified,
    Unknown,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetrologyCandidateStatusV1 {
    Eligible,
    ReadinessUnknown,
    ReadinessRejected,
    DecisionDeadlineMissed,
    NonPositiveNetValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetrologyPortfolioStatusV1 {
    Feasible,
    InfeasibleCoverage,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyDecisionStateV1 {
    pub state_id: String,
    pub prior_probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyDecisionActionV1 {
    pub action_id: String,
    /// State ID -> loss if this action is taken while that state is true.
    pub loss_by_state: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyAssayOutcomeV1 {
    pub outcome_id: String,
    /// State ID -> P(this outcome | state).
    pub likelihood_by_state: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyAssayModelV1 {
    pub schema: String,
    pub model_id: String,
    pub revision: u64,
    pub outcomes: Vec<MetrologyAssayOutcomeV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyDecisionLossModelV1 {
    pub model_id: String,
    pub revision: u64,
    pub states: Vec<MetrologyDecisionStateV1>,
    pub actions: Vec<MetrologyDecisionActionV1>,
    pub assay: MetrologyAssayModelV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyReadinessEvidenceV1 {
    pub schema: String,
    pub status: MetrologyReadinessStatusV1,
    pub instrument_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub certificate_id: String,
    pub certificate_sha256: String,
    pub calibration_record_sha256: String,
    pub measurement_system_profile_sha256: String,
    pub valid_from_unix_seconds: i64,
    pub valid_until_unix_seconds: i64,
    pub revoked_at_unix_seconds: Option<i64>,
    pub expected_standard_uncertainty_upper_bound: f64,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyDecisionLossPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub measurement_budget_units: u32,
    /// Converts one second of turnaround into the same declared loss units.
    pub delay_cost_per_second: f64,
    pub minimum_group_coverage: BTreeMap<String, u32>,
    pub maximum_exact_candidates: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyCandidateV1 {
    pub schema: String,
    pub candidate_id: String,
    pub measurement_event_id: String,
    pub group_key: String,
    pub instrument_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub planned_measurement_at_unix_seconds: i64,
    pub decision_deadline_unix_seconds: i64,
    pub turnaround_seconds: u64,
    pub measurement_cost: f64,
    pub measurement_cost_units: u32,
    pub readiness: MetrologyReadinessEvidenceV1,
    pub decision_loss_model: MetrologyDecisionLossModelV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyCandidateEvsiReportV1 {
    pub schema: String,
    pub candidate_id: String,
    pub measurement_event_id: String,
    pub group_key: String,
    pub status: MetrologyCandidateStatusV1,
    pub expected_loss_without_measurement: Option<f64>,
    pub expected_loss_after_measurement: Option<f64>,
    pub gross_expected_loss_reduction: Option<f64>,
    pub measurement_cost: f64,
    pub turnaround_cost: f64,
    pub net_expected_loss_reduction: Option<f64>,
    pub measurement_cost_units: u32,
    pub readiness_report_sha256: String,
    pub decision_loss_model_sha256: String,
    pub reason_codes: Vec<String>,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyDecisionLossInputV1 {
    pub schema: String,
    pub input_id: String,
    pub policy: MetrologyDecisionLossPolicyV1,
    pub candidates: Vec<MetrologyCandidateV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyPortfolioReportV1 {
    pub schema: String,
    pub input_id: String,
    pub policy_sha256: String,
    pub status: MetrologyPortfolioStatusV1,
    pub candidate_reports: Vec<MetrologyCandidateEvsiReportV1>,
    pub selected_candidate_ids: Vec<String>,
    pub total_measurement_cost_units: u32,
    pub remaining_measurement_budget_units: u32,
    pub selected_group_counts: BTreeMap<String, u32>,
    pub expected_net_loss_reduction: f64,
    pub reason_codes: Vec<String>,
    pub source_authenticity_proven: bool,
    pub fab_costs_empirically_established: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct MetrologyDecisionLossErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), MetrologyDecisionLossErrorV1> {
    if value.trim().is_empty() {
        return Err(MetrologyDecisionLossErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, MetrologyDecisionLossErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        MetrologyDecisionLossErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn readiness_digest(
    evidence: &MetrologyReadinessEvidenceV1,
) -> Result<String, MetrologyDecisionLossErrorV1> {
    let mut canonical = evidence.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn candidate_report_digest(
    report: &MetrologyCandidateEvsiReportV1,
) -> Result<String, MetrologyDecisionLossErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn portfolio_report_digest(
    report: &MetrologyPortfolioReportV1,
) -> Result<String, MetrologyDecisionLossErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn validate_policy(
    policy: &MetrologyDecisionLossPolicyV1,
) -> Result<(), MetrologyDecisionLossErrorV1> {
    if policy.schema != METROLOGY_DECISION_LOSS_POLICY_SCHEMA_V1 {
        return Err(MetrologyDecisionLossErrorV1(
            "decision-loss policy schema is unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0
        || !policy.delay_cost_per_second.is_finite()
        || policy.delay_cost_per_second < 0.0
        || policy.maximum_exact_candidates == 0
        || policy.maximum_exact_candidates > 20
        || policy
            .minimum_group_coverage
            .keys()
            .any(|key| key.trim().is_empty())
    {
        return Err(MetrologyDecisionLossErrorV1(
            "decision-loss policy contains invalid parameters".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_decision_model(
    model: &MetrologyDecisionLossModelV1,
) -> Result<(), MetrologyDecisionLossErrorV1> {
    require_text(&model.model_id, "decision_loss_model.model_id")?;
    if model.revision == 0 || model.states.is_empty() || model.actions.is_empty() {
        return Err(MetrologyDecisionLossErrorV1(
            "decision model requires a revision, states and actions".into(),
        ));
    }
    require_text(&model.assay.model_id, "assay.model_id")?;
    if model.assay.schema != METROLOGY_ASSAY_MODEL_SCHEMA_V1
        || model.assay.revision == 0
        || model.assay.outcomes.is_empty()
    {
        return Err(MetrologyDecisionLossErrorV1(
            "assay model schema, revision or outcomes are invalid".into(),
        ));
    }

    let mut state_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    for state in &model.states {
        require_text(&state.state_id, "state.state_id")?;
        if !state_ids.insert(state.state_id.as_str())
            || !state.prior_probability.is_finite()
            || state.prior_probability < 0.0
        {
            return Err(MetrologyDecisionLossErrorV1(
                "decision state IDs or prior probabilities are invalid".into(),
            ));
        }
        prior_sum += state.prior_probability;
    }
    if (prior_sum - 1.0).abs() > 1.0e-9 {
        return Err(MetrologyDecisionLossErrorV1(
            "state prior probabilities must sum to one".into(),
        ));
    }

    let mut action_ids = BTreeSet::new();
    for action in &model.actions {
        require_text(&action.action_id, "action.action_id")?;
        if !action_ids.insert(action.action_id.as_str())
            || action.loss_by_state.len() != state_ids.len()
            || model.states.iter().any(|state| {
                action
                    .loss_by_state
                    .get(&state.state_id)
                    .is_none_or(|loss| !loss.is_finite() || *loss < 0.0)
            })
        {
            return Err(MetrologyDecisionLossErrorV1(
                "decision action loss matrix is incomplete, duplicated or invalid".into(),
            ));
        }
    }

    let mut outcome_ids = BTreeSet::new();
    let mut outcome_sum_by_state: BTreeMap<&str, f64> = model
        .states
        .iter()
        .map(|state| (state.state_id.as_str(), 0.0))
        .collect();
    for outcome in &model.assay.outcomes {
        require_text(&outcome.outcome_id, "assay.outcome_id")?;
        if !outcome_ids.insert(outcome.outcome_id.as_str())
            || outcome.likelihood_by_state.len() != state_ids.len()
        {
            return Err(MetrologyDecisionLossErrorV1(
                "assay outcome IDs or likelihood dimensions are invalid".into(),
            ));
        }
        for state in &model.states {
            let Some(probability) = outcome.likelihood_by_state.get(&state.state_id) else {
                return Err(MetrologyDecisionLossErrorV1(
                    "assay likelihood matrix is missing a state".into(),
                ));
            };
            if !probability.is_finite() || *probability < 0.0 || *probability > 1.0 {
                return Err(MetrologyDecisionLossErrorV1(
                    "assay likelihood must be finite and in [0,1]".into(),
                ));
            }
            *outcome_sum_by_state
                .get_mut(state.state_id.as_str())
                .unwrap() += probability;
        }
    }
    if outcome_sum_by_state
        .values()
        .any(|sum| (sum - 1.0).abs() > 1.0e-9)
    {
        return Err(MetrologyDecisionLossErrorV1(
            "assay outcome likelihoods must sum to one for every state".into(),
        ));
    }
    Ok(())
}

fn expected_optimal_loss(
    model: &MetrologyDecisionLossModelV1,
    probabilities: &BTreeMap<String, f64>,
) -> Result<f64, MetrologyDecisionLossErrorV1> {
    let mut best = f64::INFINITY;
    for action in &model.actions {
        let mut expected = 0.0;
        for state in &model.states {
            let probability = probabilities.get(&state.state_id).ok_or_else(|| {
                MetrologyDecisionLossErrorV1("posterior omits a decision state".into())
            })?;
            expected += probability * action.loss_by_state[&state.state_id];
        }
        if expected < best {
            best = expected;
        }
    }
    if !best.is_finite() {
        return Err(MetrologyDecisionLossErrorV1(
            "optimal expected decision loss is non-finite".into(),
        ));
    }
    Ok(best)
}

/// Computes the Bayes decision risk before and after observing the assay.
pub fn compute_expected_value_of_sample_information_v1(
    model: &MetrologyDecisionLossModelV1,
) -> Result<(f64, f64), MetrologyDecisionLossErrorV1> {
    validate_decision_model(model)?;
    let prior: BTreeMap<String, f64> = model
        .states
        .iter()
        .map(|state| (state.state_id.clone(), state.prior_probability))
        .collect();
    let before = expected_optimal_loss(model, &prior)?;
    let mut after = 0.0;

    for outcome in &model.assay.outcomes {
        let outcome_probability: f64 = model
            .states
            .iter()
            .map(|state| state.prior_probability * outcome.likelihood_by_state[&state.state_id])
            .sum();
        if outcome_probability <= 0.0 {
            continue;
        }
        let posterior: BTreeMap<String, f64> = model
            .states
            .iter()
            .map(|state| {
                (
                    state.state_id.clone(),
                    state.prior_probability * outcome.likelihood_by_state[&state.state_id]
                        / outcome_probability,
                )
            })
            .collect();
        after += outcome_probability * expected_optimal_loss(model, &posterior)?;
    }

    if !after.is_finite() || after > before + 1.0e-8 {
        return Err(MetrologyDecisionLossErrorV1(
            "assay expected risk is invalid or exceeds prior Bayes risk".into(),
        ));
    }
    Ok((before, after))
}

pub(crate) fn score_candidate(
    candidate: &MetrologyCandidateV1,
    policy: &MetrologyDecisionLossPolicyV1,
) -> Result<MetrologyCandidateEvsiReportV1, MetrologyDecisionLossErrorV1> {
    if candidate.schema != METROLOGY_CANDIDATE_SCHEMA_V1 {
        return Err(MetrologyDecisionLossErrorV1(
            "candidate schema is unsupported".into(),
        ));
    }
    require_text(&candidate.candidate_id, "candidate.candidate_id")?;
    require_text(
        &candidate.measurement_event_id,
        "candidate.measurement_event_id",
    )?;
    require_text(&candidate.group_key, "candidate.group_key")?;
    require_text(&candidate.instrument_id, "candidate.instrument_id")?;
    require_text(&candidate.measurand_id, "candidate.measurand_id")?;
    require_text(&candidate.unit, "candidate.unit")?;
    require_text(
        &candidate.readiness.certificate_id,
        "readiness.certificate_id",
    )?;
    require_text(
        &candidate.readiness.instrument_id,
        "readiness.instrument_id",
    )?;
    require_text(&candidate.readiness.measurand_id, "readiness.measurand_id")?;
    require_text(&candidate.readiness.unit, "readiness.unit")?;
    if !candidate.measurement_cost.is_finite()
        || candidate.measurement_cost < 0.0
        || candidate.measurement_cost_units == 0
        || candidate.turnaround_seconds > i64::MAX as u64
    {
        return Err(MetrologyDecisionLossErrorV1(
            "candidate measurement cost/turnaround is invalid".into(),
        ));
    }
    if candidate.readiness.schema != METROLOGY_READINESS_SCHEMA_V1
        || candidate.readiness.valid_from_unix_seconds
            >= candidate.readiness.valid_until_unix_seconds
        || !valid_sha256(&candidate.readiness.report_sha256)
        || readiness_digest(&candidate.readiness)? != candidate.readiness.report_sha256
    {
        return Err(MetrologyDecisionLossErrorV1(
            "candidate readiness evidence digest is invalid".into(),
        ));
    }

    let mut status = MetrologyCandidateStatusV1::Eligible;
    let mut reasons = Vec::new();
    match candidate.readiness.status {
        MetrologyReadinessStatusV1::Qualified => {}
        MetrologyReadinessStatusV1::Unknown => {
            status = MetrologyCandidateStatusV1::ReadinessUnknown;
            reasons.push("measurement_system_readiness_unknown".into());
        }
        MetrologyReadinessStatusV1::Rejected => {
            status = MetrologyCandidateStatusV1::ReadinessRejected;
            reasons.push("measurement_system_readiness_rejected".into());
        }
    }

    if candidate.readiness.instrument_id != candidate.instrument_id
        || candidate.readiness.measurand_id != candidate.measurand_id
        || candidate.readiness.unit != candidate.unit
    {
        status = MetrologyCandidateStatusV1::ReadinessRejected;
        reasons.push("measurement_system_identity_mismatch".into());
    }
    if !valid_sha256(&candidate.readiness.certificate_sha256)
        || !valid_sha256(&candidate.readiness.calibration_record_sha256)
        || !valid_sha256(&candidate.readiness.measurement_system_profile_sha256)
        || !candidate
            .readiness
            .expected_standard_uncertainty_upper_bound
            .is_finite()
        || candidate
            .readiness
            .expected_standard_uncertainty_upper_bound
            <= 0.0
    {
        status = MetrologyCandidateStatusV1::ReadinessUnknown;
        reasons.push("measurement_system_uncertainty_or_identity_incomplete".into());
    }
    if candidate.readiness.valid_from_unix_seconds > candidate.planned_measurement_at_unix_seconds
        || candidate.readiness.valid_until_unix_seconds
            < candidate.planned_measurement_at_unix_seconds
        || candidate
            .readiness
            .revoked_at_unix_seconds
            .is_some_and(|revoked| revoked <= candidate.planned_measurement_at_unix_seconds)
    {
        status = MetrologyCandidateStatusV1::ReadinessRejected;
        reasons.push("calibration_not_valid_at_planned_measurement_time".into());
    }

    let measurement_complete_at = candidate
        .planned_measurement_at_unix_seconds
        .checked_add(candidate.turnaround_seconds as i64)
        .ok_or_else(|| {
            MetrologyDecisionLossErrorV1("measurement completion time overflow".into())
        })?;
    if measurement_complete_at > candidate.decision_deadline_unix_seconds {
        if status == MetrologyCandidateStatusV1::Eligible {
            status = MetrologyCandidateStatusV1::DecisionDeadlineMissed;
        }
        reasons.push("measurement_cannot_return_before_decision_deadline".into());
    }

    let model_sha256 = json_sha256(&candidate.decision_loss_model)?;
    let mut before = None;
    let mut after = None;
    let mut gross = None;
    let mut net = None;
    let delay_cost = policy.delay_cost_per_second * candidate.turnaround_seconds as f64;
    if !delay_cost.is_finite() {
        return Err(MetrologyDecisionLossErrorV1(
            "turnaround cost overflow".into(),
        ));
    }

    if status == MetrologyCandidateStatusV1::Eligible {
        let (prior_risk, post_measurement_risk) =
            compute_expected_value_of_sample_information_v1(&candidate.decision_loss_model)?;
        let evsi = prior_risk - post_measurement_risk;
        let net_value = evsi - candidate.measurement_cost - delay_cost;
        before = Some(prior_risk);
        after = Some(post_measurement_risk);
        gross = Some(evsi.max(0.0));
        net = Some(net_value);
        if net_value <= 0.0 {
            status = MetrologyCandidateStatusV1::NonPositiveNetValue;
            reasons.push("net_expected_decision_loss_reduction_not_positive".into());
        }
    }

    let mut report = MetrologyCandidateEvsiReportV1 {
        schema: METROLOGY_EVSI_REPORT_SCHEMA_V1.into(),
        candidate_id: candidate.candidate_id.clone(),
        measurement_event_id: candidate.measurement_event_id.clone(),
        group_key: candidate.group_key.clone(),
        status,
        expected_loss_without_measurement: before,
        expected_loss_after_measurement: after,
        gross_expected_loss_reduction: gross,
        measurement_cost: candidate.measurement_cost,
        turnaround_cost: delay_cost,
        net_expected_loss_reduction: net,
        measurement_cost_units: candidate.measurement_cost_units,
        readiness_report_sha256: candidate.readiness.report_sha256.clone(),
        decision_loss_model_sha256: model_sha256,
        reason_codes: reasons,
        report_sha256: String::new(),
    };
    report.report_sha256 = candidate_report_digest(&report)?;
    Ok(report)
}

fn lexicographically_preferred(left: &[String], right: &[String]) -> bool {
    left < right
}

/// Exact 0/1 portfolio optimization for at most 20 candidates. A larger set
/// is rejected rather than silently falling back to a heuristic.
pub fn optimize_metrology_decision_loss_portfolio_v1(
    input: &MetrologyDecisionLossInputV1,
) -> Result<MetrologyPortfolioReportV1, MetrologyDecisionLossErrorV1> {
    if input.schema != METROLOGY_DECISION_LOSS_INPUT_SCHEMA_V1 {
        return Err(MetrologyDecisionLossErrorV1(
            "decision-loss input schema is unsupported".into(),
        ));
    }
    require_text(&input.input_id, "input.input_id")?;
    validate_policy(&input.policy)?;
    if input.candidates.len() > input.policy.maximum_exact_candidates as usize {
        return Err(MetrologyDecisionLossErrorV1(
            "candidate count exceeds the exact portfolio optimizer limit".into(),
        ));
    }

    let mut candidates = input.candidates.clone();
    candidates.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    if candidates
        .windows(2)
        .any(|pair| pair[0].candidate_id == pair[1].candidate_id)
    {
        return Err(MetrologyDecisionLossErrorV1(
            "duplicate metrology candidate ID".into(),
        ));
    }

    let mut candidate_reports = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        candidate_reports.push(score_candidate(candidate, &input.policy)?);
    }

    let n = candidates.len();
    let total_masks = 1_u64 << n;
    let mut best_value = f64::NEG_INFINITY;
    let mut best_ids: Vec<String> = Vec::new();
    let mut best_cost = 0_u32;
    let mut best_groups: BTreeMap<String, u32> = BTreeMap::new();
    let mut found_feasible = false;

    for mask in 0..total_masks {
        let mut used_cost = 0_u32;
        let mut value = 0.0;
        let mut ids = Vec::new();
        let mut group_counts: BTreeMap<String, u32> = BTreeMap::new();
        let mut feasible = true;

        for (index, report) in candidate_reports.iter().enumerate() {
            if (mask & (1_u64 << index)) == 0 {
                continue;
            }
            if report.status != MetrologyCandidateStatusV1::Eligible
                && report.status != MetrologyCandidateStatusV1::NonPositiveNetValue
            {
                feasible = false;
                break;
            }
            used_cost = match used_cost.checked_add(report.measurement_cost_units) {
                Some(cost) => cost,
                None => {
                    feasible = false;
                    break;
                }
            };
            if used_cost > input.policy.measurement_budget_units {
                feasible = false;
                break;
            }
            let net = report.net_expected_loss_reduction.unwrap_or(0.0);
            value += net;
            ids.push(report.candidate_id.clone());
            *group_counts.entry(report.group_key.clone()).or_default() += 1;
        }
        if !feasible {
            continue;
        }
        if input
            .policy
            .minimum_group_coverage
            .iter()
            .any(|(group, minimum)| group_counts.get(group).copied().unwrap_or(0) < *minimum)
        {
            continue;
        }

        if !found_feasible
            || value > best_value + 1.0e-10
            || ((value - best_value).abs() <= 1.0e-10
                && lexicographically_preferred(&ids, &best_ids))
        {
            found_feasible = true;
            best_value = value;
            best_ids = ids;
            best_cost = used_cost;
            best_groups = group_counts;
        }
    }

    let mut report = MetrologyPortfolioReportV1 {
        schema: METROLOGY_PORTFOLIO_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        policy_sha256: json_sha256(&input.policy)?,
        status: if found_feasible {
            MetrologyPortfolioStatusV1::Feasible
        } else {
            MetrologyPortfolioStatusV1::InfeasibleCoverage
        },
        candidate_reports,
        selected_candidate_ids: best_ids,
        total_measurement_cost_units: best_cost,
        remaining_measurement_budget_units: input
            .policy
            .measurement_budget_units
            .saturating_sub(best_cost),
        selected_group_counts: best_groups,
        expected_net_loss_reduction: if found_feasible { best_value } else { 0.0 },
        reason_codes: if found_feasible {
            Vec::new()
        } else {
            vec!["minimum_group_coverage_infeasible_under_qualified_candidate_budget".into()]
        },
        source_authenticity_proven: false,
        fab_costs_empirically_established: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = portfolio_report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn readiness(
        status: MetrologyReadinessStatusV1,
        instrument: &str,
        measurand: &str,
        unit: &str,
    ) -> MetrologyReadinessEvidenceV1 {
        let mut value = MetrologyReadinessEvidenceV1 {
            schema: METROLOGY_READINESS_SCHEMA_V1.into(),
            status,
            instrument_id: instrument.into(),
            measurand_id: measurand.into(),
            unit: unit.into(),
            certificate_id: "cal-cert-1".into(),
            certificate_sha256: digest('a'),
            calibration_record_sha256: digest('b'),
            measurement_system_profile_sha256: digest('c'),
            valid_from_unix_seconds: 0,
            valid_until_unix_seconds: 10_000,
            revoked_at_unix_seconds: None,
            expected_standard_uncertainty_upper_bound: 0.01,
            report_sha256: String::new(),
        };
        value.report_sha256 = readiness_digest(&value).unwrap();
        value
    }

    fn decision_model(evidence_quality: f64) -> MetrologyDecisionLossModelV1 {
        MetrologyDecisionLossModelV1 {
            model_id: "toy-false-accept-loss.v1".into(),
            revision: 1,
            states: vec![
                MetrologyDecisionStateV1 {
                    state_id: "good".into(),
                    prior_probability: 0.90,
                },
                MetrologyDecisionStateV1 {
                    state_id: "defective".into(),
                    prior_probability: 0.10,
                },
            ],
            actions: vec![
                MetrologyDecisionActionV1 {
                    action_id: "accept".into(),
                    loss_by_state: BTreeMap::from([
                        ("good".into(), 0.0),
                        ("defective".into(), 100.0),
                    ]),
                },
                MetrologyDecisionActionV1 {
                    action_id: "hold".into(),
                    loss_by_state: BTreeMap::from([
                        ("good".into(), 5.0),
                        ("defective".into(), 0.0),
                    ]),
                },
            ],
            assay: MetrologyAssayModelV1 {
                schema: METROLOGY_ASSAY_MODEL_SCHEMA_V1.into(),
                model_id: "assay.v1".into(),
                revision: 1,
                outcomes: vec![
                    MetrologyAssayOutcomeV1 {
                        outcome_id: "pass".into(),
                        likelihood_by_state: BTreeMap::from([
                            ("good".into(), 0.98),
                            ("defective".into(), 1.0 - evidence_quality),
                        ]),
                    },
                    MetrologyAssayOutcomeV1 {
                        outcome_id: "fail".into(),
                        likelihood_by_state: BTreeMap::from([
                            ("good".into(), 0.02),
                            ("defective".into(), evidence_quality),
                        ]),
                    },
                ],
            },
        }
    }

    fn candidate(
        id: &str,
        group: &str,
        cost_units: u32,
        measurement_cost: f64,
        readiness_status: MetrologyReadinessStatusV1,
        timing: (u64, i64),
        evidence_quality: f64,
    ) -> MetrologyCandidateV1 {
        MetrologyCandidateV1 {
            schema: METROLOGY_CANDIDATE_SCHEMA_V1.into(),
            candidate_id: id.into(),
            measurement_event_id: format!("measurement-{id}"),
            group_key: group.into(),
            instrument_id: "tool-A".into(),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            planned_measurement_at_unix_seconds: 100,
            decision_deadline_unix_seconds: timing.1,
            turnaround_seconds: timing.0,
            measurement_cost,
            measurement_cost_units: cost_units,
            readiness: readiness(readiness_status, "tool-A", "film-thickness", "nm"),
            decision_loss_model: decision_model(evidence_quality),
        }
    }

    fn policy(budget: u32) -> MetrologyDecisionLossPolicyV1 {
        MetrologyDecisionLossPolicyV1 {
            schema: METROLOGY_DECISION_LOSS_POLICY_SCHEMA_V1.into(),
            policy_id: "equal-budget-decision-loss.v1".into(),
            revision: 1,
            measurement_budget_units: budget,
            delay_cost_per_second: 0.01,
            minimum_group_coverage: BTreeMap::new(),
            maximum_exact_candidates: 20,
        }
    }

    fn input(
        candidates: Vec<MetrologyCandidateV1>,
        policy: MetrologyDecisionLossPolicyV1,
    ) -> MetrologyDecisionLossInputV1 {
        MetrologyDecisionLossInputV1 {
            schema: METROLOGY_DECISION_LOSS_INPUT_SCHEMA_V1.into(),
            input_id: "decision-loss-fixture-001".into(),
            policy,
            candidates,
        }
    }

    #[test]
    fn informative_assay_reduces_expected_decision_loss() {
        let informative = decision_model(0.95);
        let uninformative = decision_model(0.10);
        let (before, after_informative) =
            compute_expected_value_of_sample_information_v1(&informative).unwrap();
        let (_, after_uninformative) =
            compute_expected_value_of_sample_information_v1(&uninformative).unwrap();
        assert!(after_informative < after_uninformative);
        assert!(after_informative <= before);
    }

    #[test]
    fn readiness_unknown_or_rejected_candidate_never_selected() {
        let input = input(
            vec![
                candidate(
                    "unknown",
                    "chamber-1",
                    1,
                    0.1,
                    MetrologyReadinessStatusV1::Unknown,
                    (10, 1_000),
                    0.95,
                ),
                candidate(
                    "rejected",
                    "chamber-1",
                    1,
                    0.1,
                    MetrologyReadinessStatusV1::Rejected,
                    (10, 1_000),
                    0.95,
                ),
            ],
            policy(2),
        );
        let report = optimize_metrology_decision_loss_portfolio_v1(&input).unwrap();
        assert_eq!(report.selected_candidate_ids, Vec::<String>::new());
    }

    #[test]
    fn exact_portfolio_uses_budget_and_highest_net_decision_value() {
        let input = input(
            vec![
                candidate(
                    "low",
                    "chamber-1",
                    1,
                    0.5,
                    MetrologyReadinessStatusV1::Qualified,
                    (10, 1_000),
                    0.20,
                ),
                candidate(
                    "high",
                    "chamber-1",
                    1,
                    0.1,
                    MetrologyReadinessStatusV1::Qualified,
                    (10, 1_000),
                    0.95,
                ),
                candidate(
                    "mid",
                    "chamber-2",
                    1,
                    0.3,
                    MetrologyReadinessStatusV1::Qualified,
                    (10, 1_000),
                    0.65,
                ),
            ],
            policy(1),
        );
        let report = optimize_metrology_decision_loss_portfolio_v1(&input).unwrap();
        assert_eq!(report.status, MetrologyPortfolioStatusV1::Feasible);
        assert_eq!(report.selected_candidate_ids, vec!["high".to_string()]);
        assert_eq!(report.total_measurement_cost_units, 1);
    }

    #[test]
    fn group_minimums_are_hard_constraints_not_soft_bonuses() {
        let mut p = policy(2);
        p.minimum_group_coverage =
            BTreeMap::from([("chamber-1".into(), 1), ("chamber-2".into(), 1)]);
        let input = input(
            vec![
                candidate(
                    "a",
                    "chamber-1",
                    1,
                    0.1,
                    MetrologyReadinessStatusV1::Qualified,
                    (10, 1_000),
                    0.9,
                ),
                candidate(
                    "b",
                    "chamber-2",
                    1,
                    0.1,
                    MetrologyReadinessStatusV1::Qualified,
                    (10, 1_000),
                    0.9,
                ),
            ],
            p,
        );
        let report = optimize_metrology_decision_loss_portfolio_v1(&input).unwrap();
        assert_eq!(
            report.selected_candidate_ids,
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn infeasible_group_coverage_returns_explicit_infeasible_status() {
        let mut p = policy(1);
        p.minimum_group_coverage =
            BTreeMap::from([("chamber-1".into(), 1), ("chamber-2".into(), 1)]);
        let input = input(
            vec![candidate(
                "a",
                "chamber-1",
                1,
                0.1,
                MetrologyReadinessStatusV1::Qualified,
                (10, 1_000),
                0.9,
            )],
            p,
        );
        let report = optimize_metrology_decision_loss_portfolio_v1(&input).unwrap();
        assert_eq!(
            report.status,
            MetrologyPortfolioStatusV1::InfeasibleCoverage
        );
        assert!(report.selected_candidate_ids.is_empty());
    }

    #[test]
    fn measurements_that_return_after_the_decision_deadline_are_excluded() {
        let input = input(
            vec![candidate(
                "late",
                "chamber-1",
                1,
                0.1,
                MetrologyReadinessStatusV1::Qualified,
                (100, 150),
                0.95,
            )],
            policy(1),
        );
        let report = optimize_metrology_decision_loss_portfolio_v1(&input).unwrap();
        assert_eq!(report.selected_candidate_ids, Vec::<String>::new());
        assert_eq!(
            report.candidate_reports[0].status,
            MetrologyCandidateStatusV1::DecisionDeadlineMissed
        );
    }

    #[test]
    fn calibration_revoked_before_planned_measurement_excludes_candidate() {
        let mut candidate = candidate(
            "revoked",
            "chamber-1",
            1,
            0.1,
            MetrologyReadinessStatusV1::Qualified,
            (10, 1_000),
            0.95,
        );
        candidate.readiness.revoked_at_unix_seconds = Some(99);
        candidate.readiness.report_sha256 = readiness_digest(&candidate.readiness).unwrap();
        let report =
            optimize_metrology_decision_loss_portfolio_v1(&input(vec![candidate], policy(1)))
                .unwrap();
        assert!(report.selected_candidate_ids.is_empty());
        assert_eq!(
            report.candidate_reports[0].status,
            MetrologyCandidateStatusV1::ReadinessRejected
        );
    }

    #[test]
    fn negative_net_value_candidate_is_not_selected_without_a_quota() {
        let input = input(
            vec![candidate(
                "expensive",
                "chamber-1",
                1,
                1_000.0,
                MetrologyReadinessStatusV1::Qualified,
                (10, 1_000),
                0.95,
            )],
            policy(1),
        );
        let report = optimize_metrology_decision_loss_portfolio_v1(&input).unwrap();
        assert!(report.selected_candidate_ids.is_empty());
        assert_eq!(
            report.candidate_reports[0].status,
            MetrologyCandidateStatusV1::NonPositiveNetValue
        );
    }

    #[test]
    fn exact_optimizer_rejects_duplicate_candidate_ids() {
        let candidate = candidate(
            "same",
            "chamber-1",
            1,
            0.1,
            MetrologyReadinessStatusV1::Qualified,
            (10, 1_000),
            0.95,
        );
        let input = input(vec![candidate.clone(), candidate], policy(2));
        assert!(optimize_metrology_decision_loss_portfolio_v1(&input).is_err());
    }

    #[test]
    fn exact_optimizer_rejects_more_than_the_declared_candidate_limit() {
        let mut p = policy(30);
        p.maximum_exact_candidates = 2;
        let candidates = (0..3)
            .map(|i| {
                candidate(
                    &format!("c{i}"),
                    "chamber-1",
                    1,
                    0.1,
                    MetrologyReadinessStatusV1::Qualified,
                    (10, 1_000),
                    0.95,
                )
            })
            .collect();
        assert!(optimize_metrology_decision_loss_portfolio_v1(&input(candidates, p)).is_err());
    }

    #[test]
    fn invalid_assay_probability_matrix_is_rejected() {
        let mut model = decision_model(0.95);
        model.assay.outcomes[0]
            .likelihood_by_state
            .insert("good".into(), 1.5);
        assert!(compute_expected_value_of_sample_information_v1(&model).is_err());
    }

    #[test]
    fn tampered_readiness_receipt_is_rejected() {
        let mut candidate = candidate(
            "tampered",
            "chamber-1",
            1,
            0.1,
            MetrologyReadinessStatusV1::Qualified,
            (10, 1_000),
            0.95,
        );
        candidate
            .readiness
            .expected_standard_uncertainty_upper_bound = 0.50;
        assert!(
            optimize_metrology_decision_loss_portfolio_v1(&input(vec![candidate], policy(1)))
                .is_err()
        );
    }

    #[test]
    fn candidate_input_order_does_not_change_portfolio_receipt() {
        let a = candidate(
            "a",
            "chamber-1",
            1,
            0.1,
            MetrologyReadinessStatusV1::Qualified,
            (10, 1_000),
            0.90,
        );
        let b = candidate(
            "b",
            "chamber-2",
            1,
            0.1,
            MetrologyReadinessStatusV1::Qualified,
            (10, 1_000),
            0.90,
        );
        let first = optimize_metrology_decision_loss_portfolio_v1(&input(
            vec![a.clone(), b.clone()],
            policy(1),
        ))
        .unwrap();
        let second =
            optimize_metrology_decision_loss_portfolio_v1(&input(vec![b, a], policy(1))).unwrap();
        assert_eq!(first.report_sha256, second.report_sha256);
    }

    #[test]
    fn decision_loss_portfolio_receipt_replays_deterministically() {
        let input = input(
            vec![
                candidate(
                    "a",
                    "chamber-1",
                    1,
                    0.1,
                    MetrologyReadinessStatusV1::Qualified,
                    (10, 1_000),
                    0.90,
                ),
                candidate(
                    "b",
                    "chamber-2",
                    1,
                    0.1,
                    MetrologyReadinessStatusV1::Qualified,
                    (10, 1_000),
                    0.90,
                ),
            ],
            policy(1),
        );
        let a = optimize_metrology_decision_loss_portfolio_v1(&input).unwrap();
        let b = optimize_metrology_decision_loss_portfolio_v1(&input).unwrap();
        assert_eq!(a.report_sha256, b.report_sha256);
    }
}
