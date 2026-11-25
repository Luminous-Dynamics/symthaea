// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact small-portfolio information-gain and decision-value optimizer.
//!
//! This module evaluates declared likelihood models; it does not authenticate
//! those likelihoods or execute physical measurements. Conditional independence
//! across selected candidates is used only when explicitly declared, and
//! candidates sharing a declared dependency group cannot be selected together.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const CAUSAL_PORTFOLIO_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.causal-information-portfolio-input.v1";
pub const CAUSAL_PORTFOLIO_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.causal-information-portfolio-report.v1";

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const MAX_CANDIDATES_V1: usize = 10;
const MAX_JOINT_OUTCOMES_V1: usize = 4096;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HypothesisPriorV1 {
    pub hypothesis_id: String,
    pub prior_probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateOutcomeLikelihoodV1 {
    pub outcome_id: String,
    /// Conditional probability P(outcome | hypothesis).
    pub probability_by_hypothesis: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalMeasurementCandidateV1 {
    pub candidate_id: String,
    pub dependence_group_id: String,
    pub physical_measurement_cost: f64,
    pub capacity_units: usize,
    pub expected_result_delay_seconds: u64,
    pub randomized_audit: bool,
    pub audit_design_sha256: Option<String>,
    pub audit_inclusion_probability: Option<f64>,
    pub calibration_qualified: bool,
    pub source_path_qualified: bool,
    pub safety_preconditions_satisfied: bool,
    pub expected_predictive_risk_reduction: f64,
    pub likelihood_model_sha256: String,
    pub outcome_schema_sha256: String,
    pub outcomes: Vec<CandidateOutcomeLikelihoodV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionActionLossV1 {
    pub action_id: String,
    /// Non-negative loss under each true causal hypothesis.
    pub loss_by_hypothesis: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalPortfolioObjectiveWeightsV1 {
    pub information_gain_weight: f64,
    pub decision_loss_reduction_weight: f64,
    pub predictive_risk_reduction_weight: f64,
    pub physical_cost_weight: f64,
    pub delay_weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalInformationPortfolioInputV1 {
    pub schema: String,
    pub portfolio_id: String,
    pub hypotheses: Vec<HypothesisPriorV1>,
    pub decision_actions: Vec<DecisionActionLossV1>,
    pub candidates: Vec<CausalMeasurementCandidateV1>,
    pub physical_cost_budget: f64,
    pub maximum_capacity_units: usize,
    pub minimum_randomized_audit_count: usize,
    pub conditional_independence_assumption_declared: bool,
    pub conditional_independence_assumption_sha256: Option<String>,
    pub objective_weights: CausalPortfolioObjectiveWeightsV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalPortfolioStatusV1 {
    Selected,
    AuditFloorUnsatisfied,
    NoFeasiblePortfolio,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalCandidateEligibilityV1 {
    pub candidate_id: String,
    pub eligible: bool,
    pub reason_codes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalPortfolioMetricsV1 {
    pub candidate_ids: Vec<String>,
    pub expected_hypothesis_information_gain_bits: f64,
    pub expected_decision_loss_reduction: f64,
    pub expected_predictive_risk_reduction: f64,
    pub total_physical_measurement_cost: f64,
    pub total_expected_result_delay_seconds: u64,
    pub total_capacity_units: usize,
    pub randomized_audit_count: usize,
    pub audit_inclusion_probabilities: BTreeMap<String, f64>,
    pub objective_score: f64,
    pub portfolio_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalInformationPortfolioReportV1 {
    pub schema: String,
    pub portfolio_id: String,
    pub input_sha256: String,
    pub status: CausalPortfolioStatusV1,
    pub selected_portfolio: Option<CausalPortfolioMetricsV1>,
    pub pareto_frontier: Vec<CausalPortfolioMetricsV1>,
    pub candidate_eligibility: Vec<CausalCandidateEligibilityV1>,
    pub feasible_portfolio_count: usize,
    pub feasible_without_audit_floor_count: usize,
    pub population_estimation_available: bool,
    pub future_outcomes_require_k1_before_consumption: bool,
    pub conditional_independence_assumption_used: bool,
    pub source_likelihood_authenticity_proven: bool,
    pub physical_measurement_executed: bool,
    pub equipment_control_authority: bool,
    pub reason_codes: Vec<String>,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CausalInformationPortfolioErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), CausalInformationPortfolioErrorV1> {
    if value.trim().is_empty() {
        return Err(CausalInformationPortfolioErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, CausalInformationPortfolioErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        CausalInformationPortfolioErrorV1(format!(
            "canonical JSON serialization failed: {error}"
        ))
    })?;
    Ok(sha256_hex(&bytes))
}

fn canonical_input_digest(
    input: &CausalInformationPortfolioInputV1,
) -> Result<String, CausalInformationPortfolioErrorV1> {
    let mut canonical = input.clone();
    canonical.hypotheses.sort_by(|a, b| a.hypothesis_id.cmp(&b.hypothesis_id));
    canonical.candidates.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    canonical.decision_actions.sort_by(|a, b| a.action_id.cmp(&b.action_id));
    for candidate in &mut canonical.candidates {
        candidate.outcomes.sort_by(|a, b| a.outcome_id.cmp(&b.outcome_id));
    }
    json_sha256(&canonical)
}

fn portfolio_digest(
    portfolio: &CausalPortfolioMetricsV1,
) -> Result<String, CausalInformationPortfolioErrorV1> {
    let mut canonical = portfolio.clone();
    canonical.portfolio_sha256.clear();
    canonical.candidate_ids.sort();
    json_sha256(&canonical)
}

fn report_digest(
    report: &CausalInformationPortfolioReportV1,
) -> Result<String, CausalInformationPortfolioErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.log2())
        .sum()
}

fn validate_input(
    input: &CausalInformationPortfolioInputV1,
) -> Result<
    (
        BTreeMap<String, f64>,
        BTreeMap<String, Vec<f64>>,
        Vec<CausalCandidateEligibilityV1>,
    ),
    CausalInformationPortfolioErrorV1,
> {
    if input.schema != CAUSAL_PORTFOLIO_INPUT_SCHEMA_V1 {
        return Err(CausalInformationPortfolioErrorV1(
            "causal information portfolio schema is unsupported".into(),
        ));
    }
    require_text(&input.portfolio_id, "portfolio_id")?;
    if input.hypotheses.len() < 2 {
        return Err(CausalInformationPortfolioErrorV1(
            "at least two competing hypotheses are required".into(),
        ));
    }
    if input.candidates.len() > MAX_CANDIDATES_V1 {
        return Err(CausalInformationPortfolioErrorV1(format!(
            "exact V1 optimizer supports at most {MAX_CANDIDATES_V1} candidates"
        )));
    }
    if !input.physical_cost_budget.is_finite() || input.physical_cost_budget < 0.0 {
        return Err(CausalInformationPortfolioErrorV1(
            "physical measurement budget must be finite and non-negative".into(),
        ));
    }
    if input.maximum_capacity_units == 0 {
        return Err(CausalInformationPortfolioErrorV1(
            "maximum capacity units must be positive".into(),
        ));
    }
    if input.conditional_independence_assumption_declared {
        if !input
            .conditional_independence_assumption_sha256
            .as_deref()
            .is_some_and(valid_sha256)
        {
            return Err(CausalInformationPortfolioErrorV1(
                "conditional-independence declaration requires a valid protocol digest".into(),
            ));
        }
    }

    let weights = [
        input.objective_weights.information_gain_weight,
        input.objective_weights.decision_loss_reduction_weight,
        input.objective_weights.predictive_risk_reduction_weight,
        input.objective_weights.physical_cost_weight,
        input.objective_weights.delay_weight,
    ];
    if weights.iter().any(|value| !value.is_finite() || *value < 0.0)
        || weights[..3].iter().all(|value| *value == 0.0)
    {
        return Err(CausalInformationPortfolioErrorV1(
            "objective weights must be finite/non-negative with at least one benefit weight".into(),
        ));
    }

    let mut priors = BTreeMap::new();
    for hypothesis in &input.hypotheses {
        require_text(&hypothesis.hypothesis_id, "hypothesis_id")?;
        if !hypothesis.prior_probability.is_finite()
            || hypothesis.prior_probability < 0.0
            || priors
                .insert(
                    hypothesis.hypothesis_id.clone(),
                    hypothesis.prior_probability,
                )
                .is_some()
        {
            return Err(CausalInformationPortfolioErrorV1(
                "hypothesis prior is invalid or duplicated".into(),
            ));
        }
    }
    let prior_sum: f64 = priors.values().sum();
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(CausalInformationPortfolioErrorV1(
            "hypothesis prior probabilities must sum to one".into(),
        ));
    }

    let mut action_ids = BTreeSet::new();
    for action in &input.decision_actions {
        require_text(&action.action_id, "decision_action.action_id")?;
        if !action_ids.insert(action.action_id.as_str()) {
            return Err(CausalInformationPortfolioErrorV1(
                "duplicate decision action ID".into(),
            ));
        }
        if action.loss_by_hypothesis.len() != priors.len() {
            return Err(CausalInformationPortfolioErrorV1(format!(
                "decision action {} does not cover every hypothesis",
                action.action_id
            )));
        }
        for hypothesis_id in priors.keys() {
            let Some(loss) = action.loss_by_hypothesis.get(hypothesis_id) else {
                return Err(CausalInformationPortfolioErrorV1(format!(
                    "decision action {} is missing hypothesis {}",
                    action.action_id, hypothesis_id
                )));
            };
            if !loss.is_finite() || *loss < 0.0 {
                return Err(CausalInformationPortfolioErrorV1(format!(
                    "decision action {} has an invalid loss",
                    action.action_id
                )));
            }
        }
    }

    let mut candidate_ids = BTreeSet::new();
    let mut eligibility = Vec::new();
    let mut eligible_likelihoods = BTreeMap::new();

    for candidate in &input.candidates {
        require_text(&candidate.candidate_id, "candidate_id")?;
        require_text(&candidate.dependence_group_id, "dependence_group_id")?;
        if !candidate_ids.insert(candidate.candidate_id.as_str()) {
            return Err(CausalInformationPortfolioErrorV1(
                "duplicate candidate ID".into(),
            ));
        }
        if !candidate.physical_measurement_cost.is_finite()
            || candidate.physical_measurement_cost < 0.0
            || candidate.capacity_units == 0
            || !candidate.expected_predictive_risk_reduction.is_finite()
            || candidate.expected_predictive_risk_reduction < 0.0
            || !valid_sha256(&candidate.likelihood_model_sha256)
            || !valid_sha256(&candidate.outcome_schema_sha256)
        {
            return Err(CausalInformationPortfolioErrorV1(format!(
                "candidate {} has invalid cost, capacity, predictive reduction or digest",
                candidate.candidate_id
            )));
        }
        if candidate.outcomes.len() < 2
            || candidate.outcomes.len() > MAX_JOINT_OUTCOMES_V1
        {
            return Err(CausalInformationPortfolioErrorV1(format!(
                "candidate {} requires 2..={MAX_JOINT_OUTCOMES_V1} possible outcomes",
                candidate.candidate_id
            )));
        }
        let mut reasons = Vec::new();
        if candidate.randomized_audit {
            let design_digest = candidate.audit_design_sha256.as_deref().unwrap_or_default();
            let inclusion_probability = candidate.audit_inclusion_probability;
            if !valid_sha256(design_digest)
                || !inclusion_probability.is_some_and(|value| {
                    value.is_finite() && value > 0.0 && value <= 1.0
                })
            {
                return Err(CausalInformationPortfolioErrorV1(format!(
                    "randomized audit candidate {} requires a valid design digest and inclusion probability",
                    candidate.candidate_id
                )));
            }
        } else if candidate.audit_design_sha256.is_some()
            || candidate.audit_inclusion_probability.is_some()
        {
            return Err(CausalInformationPortfolioErrorV1(format!(
                "non-audit candidate {} cannot declare audit-only metadata",
                candidate.candidate_id
            )));
        }
        if !candidate.calibration_qualified {
            reasons.push("calibration_not_qualified".into());
        }
        if !candidate.source_path_qualified {
            reasons.push("source_path_not_qualified".into());
        }
        if !candidate.safety_preconditions_satisfied {
            reasons.push("safety_preconditions_not_satisfied".into());
        }

        let mut outcome_ids = BTreeSet::new();
        let mut probability_sums: BTreeMap<String, f64> =
            priors.keys().map(|id| (id.clone(), 0.0)).collect();
        let mut probabilities = Vec::with_capacity(candidate.outcomes.len());
        for outcome in &candidate.outcomes {
            require_text(&outcome.outcome_id, "candidate.outcome_id")?;
            if !outcome_ids.insert(outcome.outcome_id.as_str()) {
                return Err(CausalInformationPortfolioErrorV1(format!(
                    "candidate {} repeats outcome ID",
                    candidate.candidate_id
                )));
            }
            if outcome.probability_by_hypothesis.len() != priors.len() {
                return Err(CausalInformationPortfolioErrorV1(format!(
                    "candidate {} outcome does not cover every hypothesis",
                    candidate.candidate_id
                )));
            }
            let mut row = Vec::with_capacity(priors.len());
            for hypothesis_id in priors.keys() {
                let Some(probability) = outcome.probability_by_hypothesis.get(hypothesis_id) else {
                    return Err(CausalInformationPortfolioErrorV1(format!(
                        "candidate {} outcome is missing hypothesis {}",
                        candidate.candidate_id, hypothesis_id
                    )));
                };
                if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                    return Err(CausalInformationPortfolioErrorV1(format!(
                        "candidate {} has invalid outcome likelihood",
                        candidate.candidate_id
                    )));
                }
                *probability_sums.get_mut(hypothesis_id).unwrap() += probability;
                row.push(*probability);
            }
            probabilities.push(row);
        }
        if probability_sums
            .values()
            .any(|sum| (*sum - 1.0).abs() > PROBABILITY_TOLERANCE)
        {
            return Err(CausalInformationPortfolioErrorV1(format!(
                "candidate {} outcome likelihoods do not normalize per hypothesis",
                candidate.candidate_id
            )));
        }

        eligibility.push(CausalCandidateEligibilityV1 {
            candidate_id: candidate.candidate_id.clone(),
            eligible: reasons.is_empty(),
            reason_codes: reasons,
        });
        if eligibility.last().unwrap().eligible {
            eligible_likelihoods.insert(candidate.candidate_id.clone(), probabilities);
        }
    }

    Ok((priors, eligible_likelihoods, eligibility))
}

fn bayes_risk(
    probabilities: &[f64],
    hypothesis_ids: &[String],
    actions: &[DecisionActionLossV1],
) -> f64 {
    if actions.is_empty() {
        return 0.0;
    }
    actions
        .iter()
        .map(|action| {
            hypothesis_ids
                .iter()
                .zip(probabilities)
                .map(|(id, probability)| probability * action.loss_by_hypothesis[id])
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min)
}

#[derive(Default)]
struct OutcomeAccumulatorV1 {
    expected_posterior_entropy: f64,
    expected_posterior_decision_risk: f64,
}

fn enumerate_joint_outcomes(
    outcome_likelihoods: &[&Vec<Vec<f64>>],
    current_outcome_indices: &mut Vec<usize>,
    depth: usize,
    hypothesis_ids: &[String],
    priors: &[f64],
    actions: &[DecisionActionLossV1],
    accumulator: &mut OutcomeAccumulatorV1,
) {
    if depth < outcome_likelihoods.len() {
        for outcome_index in 0..outcome_likelihoods[depth].len() {
            current_outcome_indices.push(outcome_index);
            enumerate_joint_outcomes(
                outcome_likelihoods,
                current_outcome_indices,
                depth + 1,
                hypothesis_ids,
                priors,
                actions,
                accumulator,
            );
            current_outcome_indices.pop();
        }
        return;
    }

    let mut joint_likelihood = vec![1.0; hypothesis_ids.len()];
    for (candidate_likelihoods, outcome_index) in outcome_likelihoods
        .iter()
        .zip(current_outcome_indices.iter())
    {
        let row = &candidate_likelihoods[*outcome_index];
        for (hypothesis_index, probability) in row.iter().enumerate() {
            joint_likelihood[hypothesis_index] *= probability;
        }
    }

    let marginal_probability: f64 = priors
        .iter()
        .zip(&joint_likelihood)
        .map(|(prior, likelihood)| prior * likelihood)
        .sum();
    if marginal_probability <= 0.0 {
        return;
    }
    let posterior: Vec<f64> = priors
        .iter()
        .zip(&joint_likelihood)
        .map(|(prior, likelihood)| prior * likelihood / marginal_probability)
        .collect();
    accumulator.expected_posterior_entropy +=
        marginal_probability * entropy_bits(&posterior);
    accumulator.expected_posterior_decision_risk += marginal_probability
        * bayes_risk(&posterior, hypothesis_ids, actions);
}

fn portfolio_information(
    selected_indices: &[usize],
    candidates: &[CausalMeasurementCandidateV1],
    candidate_likelihoods: &BTreeMap<String, Vec<Vec<f64>>>,
    hypothesis_ids: &[String],
    priors: &[f64],
    actions: &[DecisionActionLossV1],
) -> (f64, f64) {
    if selected_indices.is_empty() {
        return (0.0, 0.0);
    }

    let likelihood_refs: Vec<&Vec<Vec<f64>>> = selected_indices
        .iter()
        .map(|index| &candidate_likelihoods[&candidates[*index].candidate_id])
        .collect();
    let Some(outcome_count) = likelihood_refs
        .iter()
        .try_fold(1usize, |total, rows| total.checked_mul(rows.len()))
    else {
        return (f64::NAN, f64::NAN);
    };
    if outcome_count > MAX_JOINT_OUTCOMES_V1 {
        return (f64::NAN, f64::NAN);
    }

    let mut accumulator = OutcomeAccumulatorV1::default();
    enumerate_joint_outcomes(
        &likelihood_refs,
        &mut Vec::new(),
        0,
        hypothesis_ids,
        priors,
        actions,
        &mut accumulator,
    );
    let prior_entropy = entropy_bits(priors);
    let prior_risk = bayes_risk(priors, hypothesis_ids, actions);
    let information_gain = (prior_entropy - accumulator.expected_posterior_entropy).max(0.0);
    let decision_gain =
        (prior_risk - accumulator.expected_posterior_decision_risk).max(0.0);
    (information_gain, decision_gain)
}

fn dominates(left: &CausalPortfolioMetricsV1, right: &CausalPortfolioMetricsV1) -> bool {
    let no_worse = left.expected_hypothesis_information_gain_bits
        + PROBABILITY_TOLERANCE
        >= right.expected_hypothesis_information_gain_bits
        && left.expected_decision_loss_reduction + PROBABILITY_TOLERANCE
            >= right.expected_decision_loss_reduction
        && left.expected_predictive_risk_reduction + PROBABILITY_TOLERANCE
            >= right.expected_predictive_risk_reduction
        && left.total_physical_measurement_cost
            <= right.total_physical_measurement_cost + PROBABILITY_TOLERANCE
        && left.total_expected_result_delay_seconds
            <= right.total_expected_result_delay_seconds
        && left.total_capacity_units <= right.total_capacity_units;
    let strictly_better = left.expected_hypothesis_information_gain_bits
        > right.expected_hypothesis_information_gain_bits + PROBABILITY_TOLERANCE
        || left.expected_decision_loss_reduction
            > right.expected_decision_loss_reduction + PROBABILITY_TOLERANCE
        || left.expected_predictive_risk_reduction
            > right.expected_predictive_risk_reduction + PROBABILITY_TOLERANCE
        || left.total_physical_measurement_cost + PROBABILITY_TOLERANCE
            < right.total_physical_measurement_cost
        || left.total_expected_result_delay_seconds < right.total_expected_result_delay_seconds
        || left.total_capacity_units < right.total_capacity_units;
    no_worse && strictly_better
}

fn make_portfolio(
    selected_indices: &[usize],
    candidates: &[CausalMeasurementCandidateV1],
    information_gain: f64,
    decision_gain: f64,
    objective_weights: &CausalPortfolioObjectiveWeightsV1,
) -> Result<CausalPortfolioMetricsV1, CausalInformationPortfolioErrorV1> {
    let mut candidate_ids: Vec<String> = selected_indices
        .iter()
        .map(|index| candidates[*index].candidate_id.clone())
        .collect();
    candidate_ids.sort();
    let total_physical_measurement_cost: f64 = selected_indices
        .iter()
        .map(|index| candidates[*index].physical_measurement_cost)
        .sum();
    let total_expected_result_delay_seconds = selected_indices.iter().try_fold(
        0_u64,
        |total, index| total.checked_add(candidates[*index].expected_result_delay_seconds),
    ).ok_or_else(|| {
        CausalInformationPortfolioErrorV1("portfolio result-delay sum overflow".into())
    })?;
    let total_capacity_units = selected_indices.iter().try_fold(
        0_usize,
        |total, index| total.checked_add(candidates[*index].capacity_units),
    ).ok_or_else(|| {
        CausalInformationPortfolioErrorV1("portfolio capacity sum overflow".into())
    })?;
    if !total_physical_measurement_cost.is_finite() {
        return Err(CausalInformationPortfolioErrorV1(
            "portfolio physical measurement cost overflow".into(),
        ));
    }
    let randomized_audit_count = selected_indices
        .iter()
        .filter(|index| candidates[**index].randomized_audit)
        .count();
    let expected_predictive_risk_reduction: f64 = selected_indices
        .iter()
        .map(|index| candidates[*index].expected_predictive_risk_reduction)
        .sum();
    let audit_inclusion_probabilities: BTreeMap<String, f64> = selected_indices
        .iter()
        .filter_map(|index| {
            let candidate = &candidates[*index];
            candidate.audit_inclusion_probability.map(|probability| {
                (candidate.candidate_id.clone(), probability)
            })
        })
        .collect();
    let objective_score = objective_weights.information_gain_weight * information_gain
        + objective_weights.decision_loss_reduction_weight * decision_gain
        + objective_weights.predictive_risk_reduction_weight
            * expected_predictive_risk_reduction
        - objective_weights.physical_cost_weight * total_physical_measurement_cost
        - objective_weights.delay_weight * total_expected_result_delay_seconds as f64;
    if !objective_score.is_finite() {
        return Err(CausalInformationPortfolioErrorV1(
            "portfolio objective score overflow".into(),
        ));
    }
    let mut portfolio = CausalPortfolioMetricsV1 {
        candidate_ids,
        expected_hypothesis_information_gain_bits: information_gain,
        expected_decision_loss_reduction: decision_gain,
        expected_predictive_risk_reduction,
        total_physical_measurement_cost,
        total_expected_result_delay_seconds,
        total_capacity_units,
        randomized_audit_count,
        audit_inclusion_probabilities,
        objective_score,
        portfolio_sha256: String::new(),
    };
    portfolio.portfolio_sha256 = portfolio_digest(&portfolio)?;
    Ok(portfolio)
}

fn subset_is_feasible(
    selected_indices: &[usize],
    candidates: &[CausalMeasurementCandidateV1],
    input: &CausalInformationPortfolioInputV1,
    enforce_audit_floor: bool,
) -> bool {
    let mut cost = 0.0;
    let mut capacity = 0usize;
    let mut audits = 0usize;
    let mut dependence_groups = BTreeSet::new();
    for index in selected_indices {
        let candidate = &candidates[*index];
        cost += candidate.physical_measurement_cost;
        let Some(next_capacity) = capacity.checked_add(candidate.capacity_units) else {
            return false;
        };
        capacity = next_capacity;
        audits += usize::from(candidate.randomized_audit);
        if !dependence_groups.insert(candidate.dependence_group_id.as_str()) {
            return false;
        }
    }
    if !cost.is_finite()
        || cost > input.physical_cost_budget + PROBABILITY_TOLERANCE
        || capacity > input.maximum_capacity_units
    {
        return false;
    }
    if enforce_audit_floor && audits < input.minimum_randomized_audit_count {
        return false;
    }
    if selected_indices.len() > 1 && !input.conditional_independence_assumption_declared {
        return false;
    }
    true
}

/// Finds the exact best small portfolio and returns a Pareto frontier. For
/// multiple candidates, conditional independence across declared distinct
/// dependency groups must be explicitly sealed; repeated groups are forbidden.
pub fn optimize_causal_information_portfolio_v1(
    input: &CausalInformationPortfolioInputV1,
) -> Result<CausalInformationPortfolioReportV1, CausalInformationPortfolioErrorV1> {
    let (prior_map, candidate_likelihoods, mut candidate_eligibility) = validate_input(input)?;
    candidate_eligibility.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let hypothesis_ids: Vec<String> = prior_map.keys().cloned().collect();
    let priors: Vec<f64> = hypothesis_ids
        .iter()
        .map(|id| prior_map[id])
        .collect();
    let candidates = &input.candidates;

    if candidates.iter().filter(|candidate| {
        candidate.calibration_qualified
            && candidate.source_path_qualified
            && candidate.safety_preconditions_satisfied
    }).count() > MAX_CANDIDATES_V1 {
        return Err(CausalInformationPortfolioErrorV1(
            "eligible candidate count exceeds exact V1 bound".into(),
        ));
    }
    if input.conditional_independence_assumption_declared
        && !valid_sha256(input.conditional_independence_assumption_sha256.as_deref().unwrap())
    {
        return Err(CausalInformationPortfolioErrorV1(
            "conditional independence assumption digest is invalid".into(),
        ));
    }

    let eligible_indices: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.calibration_qualified
                && candidate.source_path_qualified
                && candidate.safety_preconditions_satisfied
        })
        .map(|(index, _)| index)
        .collect();

    let subset_count = 1usize << eligible_indices.len();
    let mut feasible = Vec::new();
    let mut feasible_without_audit_floor_count = 0usize;

    for mask in 0..subset_count {
        let selected_indices: Vec<usize> = eligible_indices
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1usize << bit) != 0)
            .map(|(_, index)| *index)
            .collect();
        if selected_indices.is_empty()
            || !subset_is_feasible(&selected_indices, candidates, input, false)
        {
            continue;
        }

        let (information_gain, decision_gain) = portfolio_information(
            &selected_indices,
            candidates,
            &candidate_likelihoods,
            &hypothesis_ids,
            &priors,
            &input.decision_actions,
        );
        if !information_gain.is_finite() || !decision_gain.is_finite() {
            continue;
        }
        feasible_without_audit_floor_count += 1;
        if !subset_is_feasible(&selected_indices, candidates, input, true) {
            continue;
        }
        feasible.push(make_portfolio(
            &selected_indices,
            candidates,
            information_gain,
            decision_gain,
            &input.objective_weights,
        )?);
    }

    feasible.sort_by(|left, right| {
        right
            .objective_score
            .total_cmp(&left.objective_score)
            .then_with(|| left.candidate_ids.cmp(&right.candidate_ids))
    });
    let selected_portfolio = feasible.first().cloned();
    let pareto_frontier: Vec<CausalPortfolioMetricsV1> = feasible
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            !feasible
                .iter()
                .enumerate()
                .any(|(other_index, other)| other_index != *index && dominates(other, candidate))
        })
        .map(|(_, candidate)| candidate.clone())
        .collect();

    let status = if selected_portfolio.is_some() {
        CausalPortfolioStatusV1::Selected
    } else if feasible_without_audit_floor_count > 0
        && input.minimum_randomized_audit_count > 0
    {
        CausalPortfolioStatusV1::AuditFloorUnsatisfied
    } else {
        CausalPortfolioStatusV1::NoFeasiblePortfolio
    };
    let population_estimation_available = selected_portfolio
        .as_ref()
        .is_some_and(|portfolio| {
            input.minimum_randomized_audit_count > 0
                && portfolio.randomized_audit_count >= input.minimum_randomized_audit_count
        });
    let mut reason_codes = Vec::new();
    if status == CausalPortfolioStatusV1::AuditFloorUnsatisfied {
        reason_codes.push("minimum_randomized_audit_floor_unsatisfied".into());
    }
    if selected_portfolio.is_none() {
        reason_codes.push("no_feasible_portfolio_selected".into());
    }
    if !input.conditional_independence_assumption_declared && eligible_indices.len() > 1 {
        reason_codes.push("multi_candidate_conditional_independence_not_declared".into());
    }

    let mut report = CausalInformationPortfolioReportV1 {
        schema: CAUSAL_PORTFOLIO_REPORT_SCHEMA_V1.into(),
        portfolio_id: input.portfolio_id.clone(),
        input_sha256: canonical_input_digest(input)?,
        status,
        selected_portfolio,
        pareto_frontier,
        candidate_eligibility,
        feasible_portfolio_count: feasible.len(),
        feasible_without_audit_floor_count,
        population_estimation_available,
        future_outcomes_require_k1_before_consumption: true,
        conditional_independence_assumption_used: input
            .conditional_independence_assumption_declared
            && selected_portfolio
                .as_ref()
                .is_some_and(|portfolio| portfolio.candidate_ids.len() > 1),
        source_likelihood_authenticity_proven: false,
        physical_measurement_executed: false,
        equipment_control_authority: false,
        reason_codes,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn candidate(
        id: &str,
        dependence: &str,
        p_h1_alarm: f64,
        p_h2_alarm: f64,
        cost: f64,
        audit: bool,
    ) -> CausalMeasurementCandidateV1 {
        CausalMeasurementCandidateV1 {
            candidate_id: id.into(),
            dependence_group_id: dependence.into(),
            physical_measurement_cost: cost,
            capacity_units: 1,
            expected_result_delay_seconds: 10,
            randomized_audit: audit,
            audit_design_sha256: audit.then(|| digest('d')),
            audit_inclusion_probability: audit.then_some(0.5),
            calibration_qualified: true,
            source_path_qualified: true,
            safety_preconditions_satisfied: true,
            expected_predictive_risk_reduction: 0.0,
            likelihood_model_sha256: digest('a'),
            outcome_schema_sha256: digest('b'),
            outcomes: vec![
                CandidateOutcomeLikelihoodV1 {
                    outcome_id: "alarm".into(),
                    probability_by_hypothesis: BTreeMap::from([
                        ("h1".into(), p_h1_alarm),
                        ("h2".into(), p_h2_alarm),
                    ]),
                },
                CandidateOutcomeLikelihoodV1 {
                    outcome_id: "quiet".into(),
                    probability_by_hypothesis: BTreeMap::from([
                        ("h1".into(), 1.0 - p_h1_alarm),
                        ("h2".into(), 1.0 - p_h2_alarm),
                    ]),
                },
            ],
        }
    }

    fn input(candidates: Vec<CausalMeasurementCandidateV1>) -> CausalInformationPortfolioInputV1 {
        CausalInformationPortfolioInputV1 {
            schema: CAUSAL_PORTFOLIO_INPUT_SCHEMA_V1.into(),
            portfolio_id: "portfolio-001".into(),
            hypotheses: vec![
                HypothesisPriorV1 {
                    hypothesis_id: "h1".into(),
                    prior_probability: 0.5,
                },
                HypothesisPriorV1 {
                    hypothesis_id: "h2".into(),
                    prior_probability: 0.5,
                },
            ],
            decision_actions: Vec::new(),
            candidates,
            physical_cost_budget: 1.0,
            maximum_capacity_units: 1,
            minimum_randomized_audit_count: 0,
            conditional_independence_assumption_declared: false,
            conditional_independence_assumption_sha256: None,
            objective_weights: CausalPortfolioObjectiveWeightsV1 {
                information_gain_weight: 1.0,
                decision_loss_reduction_weight: 0.0,
                predictive_risk_reduction_weight: 0.0,
                physical_cost_weight: 0.0,
                delay_weight: 0.0,
            },
        }
    }

    #[test]
    fn strong_sensor_candidate_has_known_binary_information_gain() {
        let report = optimize_causal_information_portfolio_v1(&input(vec![candidate(
            "strong",
            "reference-a",
            0.9,
            0.1,
            1.0,
            false,
        )]))
        .unwrap();
        let selected = report.selected_portfolio.unwrap();
        let expected = 1.0 - (-(0.9_f64) * 0.9_f64.log2() - 0.1_f64 * 0.1_f64.log2());
        assert!((selected.expected_hypothesis_information_gain_bits - expected).abs() < 1.0e-12);
        assert_eq!(selected.candidate_ids, vec!["strong"]);
    }

    #[test]
    fn budget_and_capacity_are_hard_constraints() {
        let mut value = input(vec![
            candidate("strong", "ref-a", 0.9, 0.1, 1.0, false),
            candidate("weak", "ref-b", 0.6, 0.4, 1.0, false),
        ]);
        value.conditional_independence_assumption_declared = true;
        value.conditional_independence_assumption_sha256 = Some(digest('c'));
        value.physical_cost_budget = 1.0;
        value.maximum_capacity_units = 1;
        let report = optimize_causal_information_portfolio_v1(&value).unwrap();
        assert!(report.selected_portfolio.unwrap().candidate_ids.len() <= 1);
    }

    #[test]
    fn candidates_sharing_reference_dependency_group_cannot_both_be_selected() {
        let mut value = input(vec![
            candidate("a", "shared-reference", 0.9, 0.1, 1.0, false),
            candidate("b", "shared-reference", 0.9, 0.1, 1.0, false),
        ]);
        value.conditional_independence_assumption_declared = true;
        value.conditional_independence_assumption_sha256 = Some(digest('c'));
        value.physical_cost_budget = 2.0;
        value.maximum_capacity_units = 2;
        let report = optimize_causal_information_portfolio_v1(&value).unwrap();
        assert!(report.selected_portfolio.unwrap().candidate_ids.len() <= 1);
    }

    #[test]
    fn multiple_candidates_require_a_declared_conditional_independence_protocol() {
        let mut value = input(vec![
            candidate("a", "ref-a", 0.9, 0.1, 1.0, false),
            candidate("b", "ref-b", 0.9, 0.1, 1.0, false),
        ]);
        value.physical_cost_budget = 2.0;
        value.maximum_capacity_units = 2;
        let report = optimize_causal_information_portfolio_v1(&value).unwrap();
        assert!(report.selected_portfolio.unwrap().candidate_ids.len() <= 1);
        assert!(!report.conditional_independence_assumption_used);
    }

    #[test]
    fn unqualified_measurements_receive_no_information_credit() {
        let mut invalid = candidate("invalid", "ref-a", 0.99, 0.01, 1.0, false);
        invalid.calibration_qualified = false;
        let report = optimize_causal_information_portfolio_v1(&input(vec![invalid])).unwrap();
        assert_eq!(
            report.candidate_eligibility[0].reason_codes,
            vec!["calibration_not_qualified"]
        );
        assert_eq!(report.status, CausalPortfolioStatusV1::NoFeasiblePortfolio);
        assert!(report.selected_portfolio.is_none());
    }

    #[test]
    fn unqualified_source_path_receives_no_information_credit() {
        let mut candidate = candidate("source-unknown", "ref-a", 0.99, 0.01, 1.0, false);
        candidate.source_path_qualified = false;
        let report = optimize_causal_information_portfolio_v1(&input(vec![candidate])).unwrap();
        assert_eq!(report.status, CausalPortfolioStatusV1::NoFeasiblePortfolio);
        assert_eq!(
            report.candidate_eligibility[0].reason_codes,
            vec!["source_path_not_qualified"]
        );
    }

    #[test]
    fn mandatory_randomized_audit_floor_is_enforced() {
        let mut value = input(vec![candidate("diagnostic", "ref-a", 0.9, 0.1, 1.0, false)]);
        value.minimum_randomized_audit_count = 1;
        let report = optimize_causal_information_portfolio_v1(&value).unwrap();
        assert_eq!(report.status, CausalPortfolioStatusV1::AuditFloorUnsatisfied);
        assert!(!report.population_estimation_available);
    }

    #[test]
    fn randomized_audit_candidate_satisfies_audit_floor() {
        let mut value = input(vec![candidate("audit", "ref-a", 0.9, 0.1, 1.0, true)]);
        value.minimum_randomized_audit_count = 1;
        let report = optimize_causal_information_portfolio_v1(&value).unwrap();
        assert_eq!(report.status, CausalPortfolioStatusV1::Selected);
        assert!(report.population_estimation_available);
        let selected = report.selected_portfolio.unwrap();
        assert_eq!(
            selected.audit_inclusion_probabilities.get("audit"),
            Some(&0.5)
        );
    }

    #[test]
    fn randomized_audit_requires_design_digest_and_positive_inclusion_probability() {
        let mut missing_design = candidate("audit", "ref-a", 0.9, 0.1, 1.0, true);
        missing_design.audit_design_sha256 = None;
        assert!(optimize_causal_information_portfolio_v1(&input(vec![missing_design])).is_err());

        let mut zero_probability = candidate("audit", "ref-a", 0.9, 0.1, 1.0, true);
        zero_probability.audit_inclusion_probability = Some(0.0);
        assert!(optimize_causal_information_portfolio_v1(&input(vec![zero_probability])).is_err());
    }

    #[test]
    fn invalid_likelihood_normalization_is_rejected() {
        let mut invalid = candidate("invalid", "ref-a", 0.9, 0.1, 1.0, false);
        invalid.outcomes[1].probability_by_hypothesis.insert("h1".into(), 0.2);
        assert!(optimize_causal_information_portfolio_v1(&input(vec![invalid])).is_err());
    }

    #[test]
    fn input_order_does_not_change_digest_or_selected_candidates() {
        let mut first = input(vec![
            candidate("b", "ref-b", 0.6, 0.4, 1.0, false),
            candidate("a", "ref-a", 0.9, 0.1, 1.0, false),
        ]);
        first.physical_cost_budget = 1.0;
        let mut second = first.clone();
        second.candidates.reverse();
        assert_eq!(
            canonical_input_digest(&first).unwrap(),
            canonical_input_digest(&second).unwrap()
        );
        let selected_first = optimize_causal_information_portfolio_v1(&first)
            .unwrap()
            .selected_portfolio
            .unwrap()
            .candidate_ids;
        let selected_second = optimize_causal_information_portfolio_v1(&second)
            .unwrap()
            .selected_portfolio
            .unwrap()
            .candidate_ids;
        assert_eq!(selected_first, selected_second);
    }

    #[test]
    fn pareto_frontier_preserves_information_cost_tradeoffs() {
        let mut strong = candidate("strong", "ref-a", 0.9, 0.1, 2.0, false);
        strong.expected_predictive_risk_reduction = 0.1;
        let mut cheap = candidate("cheap", "ref-b", 0.7, 0.3, 0.5, false);
        cheap.expected_predictive_risk_reduction = 0.05;
        let mut value = input(vec![strong, cheap]);
        value.physical_cost_budget = 2.0;
        value.maximum_capacity_units = 2;
        value.conditional_independence_assumption_declared = true;
        value.conditional_independence_assumption_sha256 = Some(digest('c'));
        value.objective_weights.information_gain_weight = 0.0;
        value.objective_weights.predictive_risk_reduction_weight = 1.0;
        value.objective_weights.physical_cost_weight = 0.1;
        let report = optimize_causal_information_portfolio_v1(&value).unwrap();
        assert!(report.pareto_frontier.len() >= 2);
        assert!(report.pareto_frontier.iter().any(|portfolio| portfolio.candidate_ids == vec!["cheap"]));
        assert!(report.pareto_frontier.iter().any(|portfolio| portfolio.candidate_ids == vec!["strong"]));
    }

    #[test]
    fn ineligible_candidates_do_not_create_an_empty_selected_portfolio() {
        let mut invalid = candidate("invalid", "ref-a", 0.99, 0.01, 1.0, false);
        invalid.safety_preconditions_satisfied = false;
        let report = optimize_causal_information_portfolio_v1(&input(vec![invalid])).unwrap();
        assert_eq!(report.status, CausalPortfolioStatusV1::NoFeasiblePortfolio);
        assert!(report.selected_portfolio.is_none());
    }

    #[test]
    fn decision_loss_reduction_is_positive_when_measurement_changes_best_action() {
        let mut value = input(vec![candidate("strong", "ref-a", 0.9, 0.1, 1.0, false)]);
        value.decision_actions = vec![
            DecisionActionLossV1 {
                action_id: "act-h1".into(),
                loss_by_hypothesis: BTreeMap::from([
                    ("h1".into(), 0.0),
                    ("h2".into(), 1.0),
                ]),
            },
            DecisionActionLossV1 {
                action_id: "act-h2".into(),
                loss_by_hypothesis: BTreeMap::from([
                    ("h1".into(), 1.0),
                    ("h2".into(), 0.0),
                ]),
            },
        ];
        value.objective_weights.decision_loss_reduction_weight = 1.0;
        let report = optimize_causal_information_portfolio_v1(&value).unwrap();
        assert!(report.selected_portfolio.unwrap().expected_decision_loss_reduction > 0.0);
    }
}
