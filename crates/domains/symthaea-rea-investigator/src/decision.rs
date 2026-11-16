//! Decision-aware value-of-information planning under shared likelihood models.
//!
//! This module is additive to point-EIG and robust-EIG. It asks whether an
//! experiment improves a caller-defined downstream decision, not merely
//! whether it reduces hypothesis entropy. A decision policy is proposed only;
//! no experiment or terminal action is executed here.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::robust::{
    LikelihoodModelScenario, RobustExperimentProposal, ScenarioExperimentLikelihood,
};
use super::{
    Authorization, EvidenceProvenance, Hypothesis, LikelihoodRow, PlannerConfig, PlannerError,
    PlannerErrorKind, Probability, RejectionReason, TargetImpact, validate_config,
    validate_evidence_catalog, validate_experiment, validate_hypotheses,
};

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const UTILITY_TOLERANCE: f64 = 1.0e-10;
const OBJECTIVE_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionUtilityModel {
    /// Stable caller-defined identity for this utility/loss model.
    pub id: String,
    /// Person or organization responsible for eliciting the utilities.
    pub owner: String,
    /// Declared decision scope; e.g. a particular device/build/incident class.
    pub scope: String,
    /// Unit shared by every utility entry in this model.
    pub units: String,
    /// Consequences intentionally outside this model.
    pub declared_exclusions: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DecisionImpact {
    Informational,
    ReversibleChange,
    ServiceDisruption,
    DataMutation,
    PhysicalActuation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionUtility {
    pub hypothesis_id: String,
    /// Utility for taking this decision when this hypothesis is true.
    /// Negative values are permitted to express loss.
    pub utility: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TerminalDecision {
    pub id: String,
    pub utilities: Vec<DecisionUtility>,
    /// Metadata only. The planner never executes the terminal decision.
    pub impact: DecisionImpact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionRankingPolicy {
    /// Rank by the probability-weighted utility improvement of the shared
    /// outcome-to-decision policy.
    ExpectedEvsi,
    /// Rank by the least favorable scenario's shared-policy utility improvement.
    WorstCaseEvsi,
    /// Rank by the lower-tail CVaR of shared-policy utility improvement.
    LowerTailCvarEvsi,
    /// Choose a shared outcome-to-decision policy minimizing maximum expected
    /// regret over the supplied finite scenario set; rank by that regret.
    MinimaxRegret,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionPlannerConfig {
    pub base: PlannerConfig,
    pub ranking_policy: DecisionRankingPolicy,
    /// Probability mass in (0, 1] used for lower-tail CVaR.
    pub lower_tail_probability: f64,
    /// Minimum joint scenario/outcome probability mass on which the shared
    /// policy must agree with each scenario's posterior-optimal action.
    pub minimum_decision_stability: f64,
    /// Minimum expected utility improvement under the shared policy.
    pub minimum_shared_policy_evsi: f64,
    /// Upper bound for exact deterministic policy enumeration in MinimaxRegret.
    pub maximum_policy_combinations: usize,
}

impl Default for DecisionPlannerConfig {
    fn default() -> Self {
        Self {
            base: PlannerConfig::default(),
            ranking_policy: DecisionRankingPolicy::ExpectedEvsi,
            lower_tail_probability: 0.25,
            minimum_decision_stability: 0.0,
            minimum_shared_policy_evsi: OBJECTIVE_TOLERANCE,
            maximum_policy_combinations: 100_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionRejectionReason {
    AuthorizationUnknown,
    NotAuthorized,
    Irreversible,
    Destructive,
    TargetImpactExceeded,
    TimeBudgetExceeded,
    EnergyBudgetExceeded,
    RiskBudgetExceeded,
    InvalidOutcomeSpace,
    InvalidLikelihoods,
    NoDecisionValue,
    DecisionStabilityBelowThreshold,
    PolicySearchBudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRejectedExperiment {
    pub experiment_id: String,
    pub reasons: Vec<DecisionRejectionReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionAbstentionReason {
    NoCompetingHypotheses,
    NoIndependentlyGroundedHypothesis,
    NoCompetingHypotheses,
    NoEligibleExperiment,
    NoDecisionValue,
    DecisionUnstable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionClaimCeiling {
    pub evidence_producer_authenticated: bool,
    pub likelihood_scenarios_calibrated: bool,
    pub scenario_weights_authenticated: bool,
    pub utility_model_independently_reviewed: bool,
    pub utility_scope_complete: bool,
    pub causal_truth_proven: bool,
    pub experiment_executed: bool,
    pub terminal_decision_executed: bool,
    pub target_mutated: bool,
}

impl Default for DecisionClaimCeiling {
    fn default() -> Self {
        Self {
            evidence_producer_authenticated: false,
            likelihood_scenarios_calibrated: false,
            scenario_weights_authenticated: false,
            utility_model_independently_reviewed: false,
            utility_scope_complete: false,
            causal_truth_proven: false,
            experiment_executed: false,
            terminal_decision_executed: false,
            target_mutated: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOutcomeResult {
    pub outcome: String,
    pub outcome_probability: f64,
    pub posterior: Vec<Probability>,
    pub scenario_optimal_decision_id: Option<String>,
    pub scenario_optimal_decision_impact: Option<DecisionImpact>,
    pub shared_policy_decision_id: String,
    pub shared_policy_decision_impact: DecisionImpact,
    pub scenario_best_expected_utility: f64,
    pub shared_policy_expected_utility: f64,
    pub expected_regret: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionScenarioResult {
    pub scenario_id: String,
    pub scenario_probability: f64,
    /// Value of sample information if this scenario's posterior-optimal action
    /// could be selected after every outcome.
    pub scenario_optimal_evsi: f64,
    /// Utility improvement of the one shared policy actually being evaluated.
    pub shared_policy_evsi: f64,
    /// Expected regret relative to scenario-specific posterior-optimal actions.
    pub expected_regret: f64,
    pub outcomes: Vec<DecisionOutcomeResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutcomeDecision {
    pub outcome: String,
    pub decision_id: String,
    pub impact: DecisionImpact,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRankedExperiment {
    pub experiment_id: String,
    pub expected_evsi: f64,
    pub worst_case_evsi: f64,
    pub lower_tail_cvar_evsi: f64,
    pub expected_shared_policy_evsi: f64,
    pub worst_case_shared_policy_evsi: f64,
    pub lower_tail_shared_policy_cvar_evsi: f64,
    pub expected_regret: f64,
    pub worst_case_regret: f64,
    pub decision_stability: f64,
    pub selected_objective: f64,
    pub normalized_cost: f64,
    pub utility: f64,
    pub shared_policy: Vec<OutcomeDecision>,
    pub scenario_results: Vec<DecisionScenarioResult>,
    pub support_evidence_ids: Vec<String>,
    pub independent_root_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionInvestigationPlan {
    pub selected: Option<DecisionRankedExperiment>,
    pub ranked_experiments: Vec<DecisionRankedExperiment>,
    pub rejected_experiments: Vec<DecisionRejectedExperiment>,
    pub abstention: Option<DecisionAbstentionReason>,
    pub baseline_decision_id: Option<String>,
    pub baseline_expected_utility: Option<f64>,
    pub prior_entropy_bits: f64,
    pub grounded_hypothesis_ids: Vec<String>,
    pub scenario_ids: Vec<String>,
    pub utility_model_id: String,
    pub utility_model_owner: String,
    pub utility_model_scope: String,
    pub utility_model_units: String,
    pub utility_model_exclusions: Vec<String>,
    pub claim_ceiling: DecisionClaimCeiling,
}

#[derive(Debug, Clone)]
struct ScenarioOutcome {
    probability: f64,
    posterior: Vec<f64>,
    optimal_decision_index: Option<usize>,
    best_expected_utility: f64,
}

#[derive(Debug, Clone)]
struct ScenarioEvaluation {
    scenario_id: String,
    scenario_probability: f64,
    outcomes: Vec<ScenarioOutcome>,
    optimal_evsi: f64,
}

#[derive(Debug, Clone)]
struct CandidateMetrics {
    expected_evsi: f64,
    worst_case_evsi: f64,
    lower_tail_cvar_evsi: f64,
    expected_shared_policy_evsi: f64,
    worst_case_shared_policy_evsi: f64,
    lower_tail_shared_policy_cvar_evsi: f64,
    expected_regret: f64,
    worst_case_regret: f64,
    decision_stability: f64,
    scenario_results: Vec<DecisionScenarioResult>,
}

fn decision_error(kind: PlannerErrorKind, message: impl Into<String>) -> PlannerError {
    PlannerError {
        kind,
        message: message.into(),
    }
}

fn validate_decision_config(config: DecisionPlannerConfig) -> Result<(), PlannerError> {
    validate_config(config.base)?;
    let values = [
        config.lower_tail_probability,
        config.minimum_decision_stability,
        config.minimum_shared_policy_evsi,
    ];
    if values.iter().any(|value| !value.is_finite())
        || !(0.0 < config.lower_tail_probability && config.lower_tail_probability <= 1.0)
        || !(0.0..=1.0).contains(&config.minimum_decision_stability)
        || config.minimum_shared_policy_evsi < 0.0
        || config.maximum_policy_combinations == 0
    {
        return Err(decision_error(
            PlannerErrorKind::InvalidDecisionConfiguration,
            "decision planner requires finite thresholds, lower-tail mass in (0,1], stability in [0,1], non-negative minimum EVSI and a positive policy-search budget",
        ));
    }
    Ok(())
}

fn validate_utility_model(model: &DecisionUtilityModel) -> Result<(), PlannerError> {
    if model.id.trim().is_empty()
        || model.owner.trim().is_empty()
        || model.scope.trim().is_empty()
        || model.units.trim().is_empty()
        || model.declared_exclusions.is_empty()
        || model
            .declared_exclusions
            .iter()
            .any(|item| item.trim().is_empty())
    {
        return Err(decision_error(
            PlannerErrorKind::InvalidDecisionMatrix,
            "utility model requires non-empty identity, owner, scope, units and explicit exclusions",
        ));
    }
    Ok(())
}

fn validate_decisions(
    decisions: &[TerminalDecision],
    hypothesis_ids: &[&str],
) -> Result<BTreeMap<String, BTreeMap<String, f64>>, PlannerError> {
    if decisions.is_empty() {
        return Err(decision_error(
            PlannerErrorKind::InvalidDecisionMatrix,
            "at least one terminal decision is required",
        ));
    }

    let mut validated = BTreeMap::new();
    for decision in decisions {
        if decision.id.trim().is_empty() || validated.contains_key(&decision.id) {
            return Err(decision_error(
                PlannerErrorKind::InvalidDecisionMatrix,
                format!(
                    "terminal decision ID is empty or duplicated: {}",
                    decision.id
                ),
            ));
        }
        if decision.utilities.len() != hypothesis_ids.len() {
            return Err(decision_error(
                PlannerErrorKind::InvalidDecisionMatrix,
                format!(
                    "decision {} must provide exactly one utility for each hypothesis",
                    decision.id
                ),
            ));
        }
        let mut utilities = BTreeMap::new();
        for entry in &decision.utilities {
            if !entry.utility.is_finite()
                || !hypothesis_ids.contains(&entry.hypothesis_id.as_str())
                || utilities
                    .insert(entry.hypothesis_id.clone(), entry.utility)
                    .is_some()
            {
                return Err(decision_error(
                    PlannerErrorKind::InvalidDecisionMatrix,
                    format!(
                        "decision {} contains a duplicate, unknown or non-finite utility entry",
                        decision.id
                    ),
                ));
            }
        }
        if hypothesis_ids
            .iter()
            .any(|hypothesis_id| !utilities.contains_key(*hypothesis_id))
        {
            return Err(decision_error(
                PlannerErrorKind::InvalidDecisionMatrix,
                format!("decision {} has a missing hypothesis utility", decision.id),
            ));
        }
        validated.insert(decision.id.clone(), utilities);
    }
    Ok(validated)
}

fn validate_scenario_set(
    scenarios: &[LikelihoodModelScenario],
    proposals: &[RobustExperimentProposal],
) -> Result<(), PlannerError> {
    if scenarios.is_empty() {
        return Err(decision_error(
            PlannerErrorKind::InvalidDecisionScenario,
            "decision-aware planning requires at least one likelihood scenario",
        ));
    }
    let proposal_ids: HashSet<&str> = proposals.iter().map(|item| item.id.as_str()).collect();
    let mut scenario_ids = HashSet::new();
    let mut total_probability = 0.0;

    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.as_str())
            || !scenario.scenario_probability.is_finite()
            || scenario.scenario_probability <= 0.0
        {
            return Err(decision_error(
                PlannerErrorKind::InvalidDecisionScenario,
                "scenario IDs must be unique/non-empty and scenario weights finite/positive",
            ));
        }
        total_probability += scenario.scenario_probability;
        if scenario.experiments.len() != proposal_ids.len() {
            return Err(decision_error(
                PlannerErrorKind::InvalidDecisionScenario,
                format!(
                    "scenario {} must contain exactly one matrix for every proposal",
                    scenario.scenario_id
                ),
            ));
        }
        let mut found = HashSet::new();
        for matrix in &scenario.experiments {
            if !proposal_ids.contains(matrix.experiment_id.as_str())
                || !found.insert(matrix.experiment_id.as_str())
            {
                return Err(decision_error(
                    PlannerErrorKind::InvalidDecisionScenario,
                    format!(
                        "scenario {} contains an unknown or duplicate experiment matrix",
                        scenario.scenario_id
                    ),
                ));
            }
        }
    }
    if (total_probability - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(decision_error(
            PlannerErrorKind::InvalidDecisionScenario,
            format!("scenario probability mass must sum to 1.0, received {total_probability}"),
        ));
    }
    Ok(())
}

fn decision_rejection(reason: RejectionReason) -> DecisionRejectionReason {
    match reason {
        RejectionReason::AuthorizationUnknown => DecisionRejectionReason::AuthorizationUnknown,
        RejectionReason::NotAuthorized => DecisionRejectionReason::NotAuthorized,
        RejectionReason::Irreversible => DecisionRejectionReason::Irreversible,
        RejectionReason::Destructive => DecisionRejectionReason::Destructive,
        RejectionReason::TargetImpactExceeded => DecisionRejectionReason::TargetImpactExceeded,
        RejectionReason::TimeBudgetExceeded => DecisionRejectionReason::TimeBudgetExceeded,
        RejectionReason::EnergyBudgetExceeded => DecisionRejectionReason::EnergyBudgetExceeded,
        RejectionReason::RiskBudgetExceeded => DecisionRejectionReason::RiskBudgetExceeded,
        RejectionReason::InvalidOutcomeSpace => DecisionRejectionReason::InvalidOutcomeSpace,
        RejectionReason::InvalidLikelihoods => DecisionRejectionReason::InvalidLikelihoods,
        RejectionReason::NoPredictedInformationGain => DecisionRejectionReason::NoDecisionValue,
    }
}

fn utility_for_posterior(
    utilities: &BTreeMap<String, BTreeMap<String, f64>>,
    decision_id: &str,
    hypothesis_ids: &[&str],
    posterior: &[f64],
) -> f64 {
    hypothesis_ids
        .iter()
        .zip(posterior.iter())
        .map(|(hypothesis_id, probability)| probability * utilities[decision_id][*hypothesis_id])
        .sum()
}

fn best_decision(
    utilities: &BTreeMap<String, BTreeMap<String, f64>>,
    hypothesis_ids: &[&str],
    posterior: &[f64],
) -> (usize, f64) {
    let mut best_index = 0;
    let mut best_value = f64::NEG_INFINITY;
    for (index, decision_id) in utilities.keys().enumerate() {
        let value = utility_for_posterior(utilities, decision_id, hypothesis_ids, posterior);
        if value > best_value + OBJECTIVE_TOLERANCE {
            best_index = index;
            best_value = value;
        }
    }
    (best_index, best_value)
}

fn lower_tail_cvar(values: &[(f64, f64)], alpha: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut remaining = alpha;
    let mut accumulated = 0.0;
    for (value, probability) in ordered {
        if remaining <= 0.0 {
            break;
        }
        let mass = probability.min(remaining);
        accumulated += value * mass;
        remaining -= mass;
    }
    accumulated / alpha
}

fn normalized_cost(proposal: &RobustExperimentProposal, config: PlannerConfig) -> f64 {
    config.cost_weights.time * (proposal.estimated_time_seconds / config.max_time_seconds)
        + config.cost_weights.energy * (proposal.estimated_energy_joules / config.max_energy_joules)
        + config.cost_weights.risk * (proposal.estimated_risk / config.max_risk)
}

fn evaluate_scenario(
    scenario: &LikelihoodModelScenario,
    proposal: &RobustExperimentProposal,
    priors: &BTreeMap<&str, f64>,
    utilities: &BTreeMap<String, BTreeMap<String, f64>>,
) -> ScenarioEvaluation {
    let hypothesis_ids: Vec<&str> = priors.keys().copied().collect();
    let matrix = scenario
        .experiments
        .iter()
        .find(|matrix| matrix.experiment_id == proposal.id)
        .expect("scenario set was validated");

    let rows: HashMap<&str, &LikelihoodRow> = matrix
        .likelihoods
        .iter()
        .map(|row| (row.hypothesis_id.as_str(), row))
        .collect();

    let mut outcomes = Vec::with_capacity(proposal.outcomes.len());
    let mut expected_optimal_posterior_utility = 0.0;

    for (outcome_index, _) in proposal.outcomes.iter().enumerate() {
        let probability = hypothesis_ids
            .iter()
            .map(|id| priors[id] * rows[id].outcome_probabilities[outcome_index])
            .sum::<f64>();
        let posterior = hypothesis_ids
            .iter()
            .map(|id| {
                let joint = priors[id] * rows[id].outcome_probabilities[outcome_index];
                if probability > 0.0 {
                    joint / probability
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();

        let (optimal_decision_index, best_expected_utility) = if probability > 0.0 {
            let (index, value) = best_decision(utilities, &hypothesis_ids, &posterior);
            (Some(index), value)
        } else {
            (None, 0.0)
        };
        expected_optimal_posterior_utility += probability * best_expected_utility;
        outcomes.push(ScenarioOutcome {
            probability,
            posterior,
            optimal_decision_index,
            best_expected_utility,
        });
    }

    let baseline_prior_utility = {
        let prior: Vec<f64> = hypothesis_ids.iter().map(|id| priors[id]).collect();
        let (_, value) = best_decision(utilities, &hypothesis_ids, &prior);
        value
    };
    let raw_evsi = expected_optimal_posterior_utility - baseline_prior_utility;
    let optimal_evsi = if raw_evsi < 0.0 && raw_evsi > -UTILITY_TOLERANCE {
        0.0
    } else {
        raw_evsi
    };

    ScenarioEvaluation {
        scenario_id: scenario.scenario_id.clone(),
        scenario_probability: scenario.scenario_probability,
        outcomes,
        optimal_evsi,
    }
}

fn mixture_policy(
    proposal: &RobustExperimentProposal,
    evaluations: &[ScenarioEvaluation],
    utilities: &BTreeMap<String, BTreeMap<String, f64>>,
    priors: &BTreeMap<&str, f64>,
) -> Vec<usize> {
    let hypothesis_ids: Vec<&str> = priors.keys().copied().collect();
    let mut policy = Vec::with_capacity(proposal.outcomes.len());

    for outcome_index in 0..proposal.outcomes.len() {
        let mut joint = vec![0.0; hypothesis_ids.len()];
        let mut total_probability = 0.0;
        for scenario in evaluations {
            let outcome = &scenario.outcomes[outcome_index];
            total_probability += scenario.scenario_probability * outcome.probability;
            for (index, (_, probability)) in outcome.posterior.iter().enumerate() {
                joint[index] += scenario.scenario_probability * outcome.probability * probability;
            }
        }
        if total_probability > 0.0 {
            for probability in &mut joint {
                *probability /= total_probability;
            }
            policy.push(best_decision(utilities, &hypothesis_ids, &joint).0);
        } else {
            let prior: Vec<f64> = hypothesis_ids.iter().map(|id| priors[id]).collect();
            policy.push(best_decision(utilities, &hypothesis_ids, &prior).0);
        }
    }
    policy
}

fn policy_regrets(
    policy: &[usize],
    evaluations: &[ScenarioEvaluation],
    decision_ids: &[String],
    utilities: &BTreeMap<String, BTreeMap<String, f64>>,
    hypothesis_ids: &[&str],
) -> (Vec<f64>, Vec<f64>, f64) {
    let mut scenario_regrets = Vec::with_capacity(evaluations.len());
    let mut scenario_policy_values = Vec::with_capacity(evaluations.len());
    let mut stability = 0.0;

    for scenario in evaluations {
        let mut regret = 0.0;
        let mut policy_value = 0.0;
        for (outcome_index, outcome) in scenario.outcomes.iter().enumerate() {
            if outcome.probability <= 0.0 {
                continue;
            }
            let selected_id = &decision_ids[policy[outcome_index]];
            let selected_value =
                utility_for_posterior(utilities, selected_id, hypothesis_ids, &outcome.posterior);
            policy_value += outcome.probability * selected_value;
            regret +=
                outcome.probability * (outcome.best_expected_utility - selected_value).max(0.0);
            if outcome.optimal_decision_index == Some(policy[outcome_index]) {
                stability += scenario.scenario_probability * outcome.probability;
            }
        }
        scenario_regrets.push(regret);
        scenario_policy_values.push(policy_value);
    }

    (scenario_regrets, scenario_policy_values, stability)
}

fn minimax_regret_policy(
    evaluations: &[ScenarioEvaluation],
    proposal: &RobustExperimentProposal,
    decisions: &[String],
    utilities: &BTreeMap<String, BTreeMap<String, f64>>,
    priors: &BTreeMap<&str, f64>,
    maximum_combinations: usize,
) -> Result<Vec<usize>, PlannerError> {
    let combinations = (0..proposal.outcomes.len())
        .try_fold(1usize, |count, _| count.checked_mul(decisions.len()));
    let combinations = combinations.unwrap_or(usize::MAX);
    if combinations > maximum_combinations {
        return Err(decision_error(
            PlannerErrorKind::DecisionPolicySearchBudgetExceeded,
            format!(
                "minimax policy search requires {combinations} combinations; configured maximum is {maximum_combinations}"
            ),
        ));
    }

    let hypothesis_ids: Vec<&str> = priors.keys().copied().collect();
    let mut best_policy: Option<Vec<usize>> = None;
    let mut best_worst_regret = f64::INFINITY;
    let mut best_expected_regret = f64::INFINITY;
    let mut candidate = vec![0usize; proposal.outcomes.len()];

    for mut encoded in 0..combinations {
        for index in (0..candidate.len()).rev() {
            candidate[index] = encoded % decisions.len();
            encoded /= decisions.len();
        }
        let (regrets, _, _) = policy_regrets(
            &candidate,
            evaluations,
            decisions,
            utilities,
            &hypothesis_ids,
        );
        let worst = regrets.iter().copied().fold(0.0, f64::max);
        let expected = evaluations
            .iter()
            .zip(regrets.iter())
            .map(|(scenario, regret)| scenario.scenario_probability * regret)
            .sum::<f64>();
        if worst < best_worst_regret - OBJECTIVE_TOLERANCE
            || ((worst - best_worst_regret).abs() <= OBJECTIVE_TOLERANCE
                && expected < best_expected_regret - OBJECTIVE_TOLERANCE)
            || ((worst - best_worst_regret).abs() <= OBJECTIVE_TOLERANCE
                && (expected - best_expected_regret).abs() <= OBJECTIVE_TOLERANCE
                && best_policy.as_ref().is_none_or(|best| candidate < *best))
        {
            best_worst_regret = worst;
            best_expected_regret = expected;
            best_policy = Some(candidate.clone());
        }
    }

    best_policy.ok_or_else(|| {
        decision_error(
            PlannerErrorKind::DecisionPolicySearchBudgetExceeded,
            "minimax decision policy search produced no candidate",
        )
    })
}

fn candidate_metrics(
    proposal: &RobustExperimentProposal,
    evaluations: &[ScenarioEvaluation],
    lower_tail_probability: f64,
    policy: &[usize],
    decision_ids: &[String],
    utilities: &BTreeMap<String, BTreeMap<String, f64>>,
    decision_impacts: &BTreeMap<String, DecisionImpact>,
    priors: &BTreeMap<&str, f64>,
) -> CandidateMetrics {
    let hypothesis_ids: Vec<&str> = priors.keys().copied().collect();
    let baseline_prior: Vec<f64> = hypothesis_ids.iter().map(|id| priors[id]).collect();
    let baseline_utility = best_decision(utilities, &hypothesis_ids, &baseline_prior).1;
    let (regrets, policy_values, stability) = policy_regrets(
        policy,
        evaluations,
        decision_ids,
        utilities,
        &hypothesis_ids,
    );

    let scenario_evsi: Vec<(f64, f64)> = evaluations
        .iter()
        .map(|scenario| (scenario.optimal_evsi, scenario.scenario_probability))
        .collect();
    let scenario_policy_evsi: Vec<(f64, f64)> = evaluations
        .iter()
        .zip(policy_values.iter())
        .map(|(scenario, policy_value)| {
            (
                policy_value - baseline_utility,
                scenario.scenario_probability,
            )
        })
        .collect();

    let expected_evsi = scenario_evsi
        .iter()
        .map(|(value, weight)| value * weight)
        .sum();
    let worst_case_evsi = scenario_evsi
        .iter()
        .map(|(value, _)| *value)
        .fold(f64::INFINITY, f64::min);
    let lower_tail_cvar_evsi = lower_tail_cvar(&scenario_evsi, lower_tail_probability);
    let expected_shared_policy_evsi = scenario_policy_evsi
        .iter()
        .map(|(value, weight)| value * weight)
        .sum();
    let worst_case_shared_policy_evsi = scenario_policy_evsi
        .iter()
        .map(|(value, _)| *value)
        .fold(f64::INFINITY, f64::min);
    let lower_tail_shared_policy_cvar_evsi =
        lower_tail_cvar(&scenario_policy_evsi, lower_tail_probability);
    let expected_regret = evaluations
        .iter()
        .zip(regrets.iter())
        .map(|(scenario, regret)| scenario.scenario_probability * regret)
        .sum();
    let worst_case_regret = regrets.iter().copied().fold(0.0, f64::max);

    let scenario_results = evaluations
        .iter()
        .zip(policy_values.iter())
        .zip(regrets.iter())
        .map(|((scenario, policy_value), regret)| {
            let outcomes = scenario
                .outcomes
                .iter()
                .enumerate()
                .map(|(index, outcome)| {
                    let selected_id = &decision_ids[policy[index]];
                    let selected_value = utility_for_posterior(
                        utilities,
                        selected_id,
                        &hypothesis_ids,
                        &outcome.posterior,
                    );
                    let posterior = hypothesis_ids
                        .iter()
                        .zip(outcome.posterior.iter())
                        .map(|(id, probability)| Probability {
                            hypothesis_id: (*id).to_owned(),
                            probability: *probability,
                        })
                        .collect();
                    DecisionOutcomeResult {
                        outcome: proposal.outcomes[index].clone(),
                        outcome_probability: outcome.probability,
                        posterior,
                        scenario_optimal_decision_id: outcome
                            .optimal_decision_index
                            .map(|best| decision_ids[best].clone()),
                        scenario_optimal_decision_impact: outcome
                            .optimal_decision_index
                            .map(|best| decision_impacts[&decision_ids[best]]),
                        shared_policy_decision_id: selected_id.clone(),
                        shared_policy_decision_impact: decision_impacts[selected_id],
                        scenario_best_expected_utility: outcome.best_expected_utility,
                        shared_policy_expected_utility: selected_value,
                        expected_regret: (outcome.best_expected_utility - selected_value).max(0.0),
                    }
                })
                .collect();
            DecisionScenarioResult {
                scenario_id: scenario.scenario_id.clone(),
                scenario_probability: scenario.scenario_probability,
                scenario_optimal_evsi: scenario.optimal_evsi,
                shared_policy_evsi: policy_value - baseline_utility,
                expected_regret: *regret,
                outcomes,
            }
        })
        .collect();

    CandidateMetrics {
        expected_evsi,
        worst_case_evsi,
        lower_tail_cvar_evsi,
        expected_shared_policy_evsi,
        worst_case_shared_policy_evsi,
        lower_tail_shared_policy_cvar_evsi,
        expected_regret,
        worst_case_regret,
        decision_stability: stability,
        scenario_results,
    }
}

/// Plan the next experiment by downstream decision value rather than entropy alone.
///
/// The input likelihood matrices, scenario weights, and utility model are supplied
/// by the caller and are not authenticated or calibrated by this function. Every
/// proposal is checked against the same authorization, reversibility, impact and
/// resource gates used by the point-EIG planner. The function only ranks proposals.
pub fn plan_next_experiment_decision_aware(
    evidence: &[EvidenceProvenance],
    hypotheses: &[Hypothesis],
    proposals: &[RobustExperimentProposal],
    scenarios: &[LikelihoodModelScenario],
    utility_model: &DecisionUtilityModel,
    decisions: &[TerminalDecision],
    config: DecisionPlannerConfig,
) -> Result<DecisionInvestigationPlan, PlannerError> {
    validate_decision_config(config)?;
    validate_utility_model(utility_model)?;

    let mut proposal_ids = HashSet::new();
    for proposal in proposals {
        if proposal.id.trim().is_empty() || !proposal_ids.insert(proposal.id.as_str()) {
            return Err(decision_error(
                PlannerErrorKind::DuplicateExperimentId,
                format!(
                    "empty or duplicate decision-aware proposal ID: {}",
                    proposal.id
                ),
            ));
        }
    }

    let mut ordered_scenarios = scenarios.to_vec();
    ordered_scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let scenarios = ordered_scenarios.as_slice();
    if !proposals.is_empty() {
        validate_scenario_set(scenarios, proposals)?;
    }
    let catalog = validate_evidence_catalog(evidence)?;
    let validation = validate_hypotheses(hypotheses, &catalog);
    let (priors, grounded_hypothesis_ids, support_evidence_ids, independent_root_evidence_ids) =
        match validation {
            Ok(result) => result,
            Err(error) if error.kind == PlannerErrorKind::UngroundedHypothesis => {
                return Ok(DecisionInvestigationPlan {
                    selected: None,
                    ranked_experiments: Vec::new(),
                    rejected_experiments: Vec::new(),
                    abstention: Some(DecisionAbstentionReason::NoIndependentlyGroundedHypothesis),
                    baseline_decision_id: None,
                    baseline_expected_utility: None,
                    prior_entropy_bits: 0.0,
                    grounded_hypothesis_ids: Vec::new(),
                    scenario_ids: scenarios
                        .iter()
                        .map(|scenario| scenario.scenario_id.clone())
                        .collect(),
                    utility_model_id: utility_model.id.clone(),
                    utility_model_owner: utility_model.owner.clone(),
                    utility_model_scope: utility_model.scope.clone(),
                    utility_model_units: utility_model.units.clone(),
                    utility_model_exclusions: utility_model.declared_exclusions.clone(),
                    claim_ceiling: DecisionClaimCeiling::default(),
                });
            }
            Err(error) => return Err(error),
        };
    let hypothesis_ids: Vec<&str> = priors.keys().copied().collect();
    let utilities = validate_decisions(decisions, &hypothesis_ids)?;
    let decision_ids: Vec<String> = utilities.keys().cloned().collect();
    let decision_impacts: BTreeMap<String, DecisionImpact> = decisions
        .iter()
        .map(|decision| (decision.id.clone(), decision.impact))
        .collect();
    let prior_distribution: Vec<f64> = hypothesis_ids.iter().map(|id| priors[id]).collect();
    let (baseline_index, baseline_expected_utility) =
        best_decision(&utilities, &hypothesis_ids, &prior_distribution);
    let prior_entropy_bits = {
        let entropy = -prior_distribution
            .iter()
            .filter(|probability| **probability > 0.0)
            .map(|probability| probability * probability.log2())
            .sum::<f64>();
        entropy
    };

    if !baseline_expected_utility.is_finite() {
        return Err(decision_error(
            PlannerErrorKind::InvalidDecisionMatrix,
            "baseline expected utility overflowed the finite numeric range",
        ));
    }

    if hypotheses.len() < 2 {
        return Ok(DecisionInvestigationPlan {
            selected: None,
            ranked_experiments: Vec::new(),
            rejected_experiments: Vec::new(),
            abstention: Some(DecisionAbstentionReason::NoCompetingHypotheses),
            baseline_decision_id: Some(decision_ids[baseline_index].clone()),
            baseline_expected_utility: Some(baseline_expected_utility),
            prior_entropy_bits,
            grounded_hypothesis_ids,
            scenario_ids: scenarios
                .iter()
                .map(|scenario| scenario.scenario_id.clone())
                .collect(),
            utility_model_id: utility_model.id.clone(),
            utility_model_owner: utility_model.owner.clone(),
            utility_model_scope: utility_model.scope.clone(),
            utility_model_units: utility_model.units.clone(),
            utility_model_exclusions: utility_model.declared_exclusions.clone(),
            claim_ceiling: DecisionClaimCeiling::default(),
        });
    }

    if proposals.is_empty() {
        return Ok(DecisionInvestigationPlan {
            selected: None,
            ranked_experiments: Vec::new(),
            rejected_experiments: Vec::new(),
            abstention: Some(DecisionAbstentionReason::NoEligibleExperiment),
            baseline_decision_id: Some(decision_ids[baseline_index].clone()),
            baseline_expected_utility: Some(baseline_expected_utility),
            prior_entropy_bits,
            grounded_hypothesis_ids,
            scenario_ids: scenarios
                .iter()
                .map(|scenario| scenario.scenario_id.clone())
                .collect(),
            utility_model_id: utility_model.id.clone(),
            utility_model_owner: utility_model.owner.clone(),
            utility_model_scope: utility_model.scope.clone(),
            utility_model_units: utility_model.units.clone(),
            utility_model_exclusions: utility_model.declared_exclusions.clone(),
            claim_ceiling: DecisionClaimCeiling::default(),
        });
    }

    let mut ranked_experiments = Vec::new();
    let mut rejected_experiments = Vec::new();
    let mut saw_decision_value = false;
    let mut saw_unstable = false;
    let mut saw_eligible_candidate = false;

    for proposal in proposals {
        let mut rejection_reasons = Vec::new();
        let mut evaluations = Vec::with_capacity(scenarios.len());

        for scenario in scenarios {
            let matrix: &ScenarioExperimentLikelihood = scenario
                .experiments
                .iter()
                .find(|matrix| matrix.experiment_id == proposal.id)
                .expect("scenario set was validated");
            let experiment = super::ExperimentProposal {
                id: proposal.id.clone(),
                outcomes: proposal.outcomes.clone(),
                likelihoods: matrix.likelihoods.clone(),
                authorization: proposal.authorization,
                reversible: proposal.reversible,
                destructive: proposal.destructive,
                target_impact: proposal.target_impact,
                estimated_time_seconds: proposal.estimated_time_seconds,
                estimated_energy_joules: proposal.estimated_energy_joules,
                estimated_risk: proposal.estimated_risk,
            };

            if let Err(reason) = validate_experiment(&experiment, &hypothesis_ids, config.base) {
                let mapped = decision_rejection(reason);
                if !rejection_reasons.contains(&mapped) {
                    rejection_reasons.push(mapped);
                }
                continue;
            }
            evaluations.push(evaluate_scenario(scenario, proposal, &priors, &utilities));
        }

        if !rejection_reasons.is_empty() {
            rejected_experiments.push(DecisionRejectedExperiment {
                experiment_id: proposal.id.clone(),
                reasons: rejection_reasons,
            });
            continue;
        }
        if evaluations.len() != scenarios.len() {
            rejected_experiments.push(DecisionRejectedExperiment {
                experiment_id: proposal.id.clone(),
                reasons: vec![DecisionRejectionReason::InvalidLikelihoods],
            });
            continue;
        }

        let policy = if config.ranking_policy == DecisionRankingPolicy::MinimaxRegret {
            match minimax_regret_policy(
                &evaluations,
                proposal,
                &decision_ids,
                &utilities,
                &priors,
                config.maximum_policy_combinations,
            ) {
                Ok(policy) => policy,
                Err(error)
                    if error.kind == PlannerErrorKind::DecisionPolicySearchBudgetExceeded =>
                {
                    rejected_experiments.push(DecisionRejectedExperiment {
                        experiment_id: proposal.id.clone(),
                        reasons: vec![DecisionRejectionReason::PolicySearchBudgetExceeded],
                    });
                    continue;
                }
                Err(error) => return Err(error),
            }
        } else {
            mixture_policy(proposal, &evaluations, &utilities, &priors)
        };

        saw_eligible_candidate = true;
        let metrics = candidate_metrics(
            proposal,
            &evaluations,
            config.lower_tail_probability,
            &policy,
            &decision_ids,
            &utilities,
            &decision_impacts,
            &priors,
        );

        if metrics.decision_stability + OBJECTIVE_TOLERANCE < config.minimum_decision_stability {
            saw_unstable = true;
            rejected_experiments.push(DecisionRejectedExperiment {
                experiment_id: proposal.id.clone(),
                reasons: vec![DecisionRejectionReason::DecisionStabilityBelowThreshold],
            });
            continue;
        }
        let normalized_cost = normalized_cost(proposal, config.base).max(1.0e-12);
        let (selected_objective, utility, satisfies_decision_value_floor) =
            match config.ranking_policy {
                DecisionRankingPolicy::ExpectedEvsi => (
                    metrics.expected_shared_policy_evsi,
                    metrics.expected_shared_policy_evsi / normalized_cost,
                    metrics.expected_shared_policy_evsi
                        > config.minimum_shared_policy_evsi + OBJECTIVE_TOLERANCE,
                ),
                DecisionRankingPolicy::WorstCaseEvsi => (
                    metrics.worst_case_shared_policy_evsi,
                    metrics.worst_case_shared_policy_evsi / normalized_cost,
                    metrics.worst_case_shared_policy_evsi
                        > config.minimum_shared_policy_evsi + OBJECTIVE_TOLERANCE,
                ),
                DecisionRankingPolicy::LowerTailCvarEvsi => (
                    metrics.lower_tail_shared_policy_cvar_evsi,
                    metrics.lower_tail_shared_policy_cvar_evsi / normalized_cost,
                    metrics.lower_tail_shared_policy_cvar_evsi
                        > config.minimum_shared_policy_evsi + OBJECTIVE_TOLERANCE,
                ),
                DecisionRankingPolicy::MinimaxRegret => (
                    -metrics.worst_case_regret,
                    -metrics.worst_case_regret / normalized_cost,
                    metrics.expected_shared_policy_evsi
                        > config.minimum_shared_policy_evsi + OBJECTIVE_TOLERANCE,
                ),
            };
        if !satisfies_decision_value_floor {
            rejected_experiments.push(DecisionRejectedExperiment {
                experiment_id: proposal.id.clone(),
                reasons: vec![DecisionRejectionReason::NoDecisionValue],
            });
            continue;
        }
        saw_decision_value = true;

        let shared_policy = proposal
            .outcomes
            .iter()
            .zip(policy.iter())
            .map(|(outcome, decision_index)| OutcomeDecision {
                outcome: outcome.clone(),
                decision_id: decision_ids[*decision_index].clone(),
                impact: decision_impacts[&decision_ids[*decision_index]],
            })
            .collect();

        ranked_experiments.push(DecisionRankedExperiment {
            experiment_id: proposal.id.clone(),
            expected_evsi: metrics.expected_evsi,
            worst_case_evsi: metrics.worst_case_evsi,
            lower_tail_cvar_evsi: metrics.lower_tail_cvar_evsi,
            expected_shared_policy_evsi: metrics.expected_shared_policy_evsi,
            worst_case_shared_policy_evsi: metrics.worst_case_shared_policy_evsi,
            lower_tail_shared_policy_cvar_evsi: metrics.lower_tail_shared_policy_cvar_evsi,
            expected_regret: metrics.expected_regret,
            worst_case_regret: metrics.worst_case_regret,
            decision_stability: metrics.decision_stability,
            selected_objective,
            normalized_cost,
            utility,
            shared_policy,
            scenario_results: metrics.scenario_results,
            support_evidence_ids: support_evidence_ids.clone(),
            independent_root_evidence_ids: independent_root_evidence_ids.clone(),
        });
    }

    ranked_experiments.sort_by(|left, right| match config.ranking_policy {
        DecisionRankingPolicy::MinimaxRegret => right
            .utility
            .total_cmp(&left.utility)
            .then_with(|| {
                right
                    .expected_shared_policy_evsi
                    .total_cmp(&left.expected_shared_policy_evsi)
            })
            .then_with(|| left.normalized_cost.total_cmp(&right.normalized_cost))
            .then_with(|| left.experiment_id.cmp(&right.experiment_id)),
        _ => right
            .utility
            .total_cmp(&left.utility)
            .then_with(|| right.selected_objective.total_cmp(&left.selected_objective))
            .then_with(|| left.experiment_id.cmp(&right.experiment_id)),
    });
    rejected_experiments.sort_by(|left, right| left.experiment_id.cmp(&right.experiment_id));

    let selected = ranked_experiments.first().cloned();
    let abstention = if selected.is_some() {
        None
    } else if saw_unstable {
        Some(DecisionAbstentionReason::DecisionUnstable)
    } else if !saw_decision_value && saw_eligible_candidate {
        Some(DecisionAbstentionReason::NoDecisionValue)
    } else {
        Some(DecisionAbstentionReason::NoEligibleExperiment)
    };

    Ok(DecisionInvestigationPlan {
        selected,
        ranked_experiments,
        rejected_experiments,
        abstention,
        baseline_decision_id: Some(decision_ids[baseline_index].clone()),
        baseline_expected_utility: Some(baseline_expected_utility),
        prior_entropy_bits,
        grounded_hypothesis_ids,
        scenario_ids: scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.clone())
            .collect(),
        utility_model_id: utility_model.id.clone(),
        utility_model_owner: utility_model.owner.clone(),
        utility_model_scope: utility_model.scope.clone(),
        utility_model_units: utility_model.units.clone(),
        utility_model_exclusions: utility_model.declared_exclusions.clone(),
        claim_ceiling: DecisionClaimCeiling::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robust::{LikelihoodModelScenario, ScenarioExperimentLikelihood};

    fn evidence() -> Vec<EvidenceProvenance> {
        vec![
            EvidenceProvenance {
                evidence_id: format!("ev_{}", "a".repeat(64)),
                origin: super::super::EvidenceOrigin::IndependentObservation,
                parent_evidence_ids: Vec::new(),
            },
            EvidenceProvenance {
                evidence_id: format!("ev_{}", "b".repeat(64)),
                origin: super::super::EvidenceOrigin::IndependentObservation,
                parent_evidence_ids: Vec::new(),
            },
        ]
    }

    fn hypotheses(priors: &[f64]) -> Vec<Hypothesis> {
        priors
            .iter()
            .enumerate()
            .map(|(index, prior)| Hypothesis {
                id: format!("h{index}"),
                prior_probability: *prior,
                supporting_evidence_ids: vec![format!(
                    "ev_{}",
                    if index % 2 == 0 {
                        "a".repeat(64)
                    } else {
                        "b".repeat(64)
                    }
                )],
                contradicting_evidence_ids: Vec::new(),
            })
            .collect()
    }

    fn utility_model() -> DecisionUtilityModel {
        DecisionUtilityModel {
            id: "incident-utility-v1".to_owned(),
            owner: "system owner".to_owned(),
            scope: "Pixel 8 Pro / test build / incident class A".to_owned(),
            units: "utility-points".to_owned(),
            declared_exclusions: vec!["long-term user harm not measured".to_owned()],
        }
    }

    fn decisions(rows: &[(&str, &[f64])]) -> Vec<TerminalDecision> {
        rows.iter()
            .map(|(id, values)| TerminalDecision {
                id: (*id).to_owned(),
                utilities: values
                    .iter()
                    .enumerate()
                    .map(|(index, utility)| DecisionUtility {
                        hypothesis_id: format!("h{index}"),
                        utility: *utility,
                    })
                    .collect(),
                impact: DecisionImpact::Informational,
            })
            .collect()
    }

    fn proposal(id: &str, outcomes: &[&str]) -> RobustExperimentProposal {
        RobustExperimentProposal {
            id: id.to_owned(),
            outcomes: outcomes.iter().map(|item| (*item).to_owned()).collect(),
            authorization: Authorization::Authorized,
            reversible: true,
            destructive: false,
            target_impact: TargetImpact::ReadOnlyObservation,
            estimated_time_seconds: 10.0,
            estimated_energy_joules: 1.0,
            estimated_risk: 0.01,
        }
    }

    fn matrix(id: &str, rows: &[(&str, &[f64])]) -> ScenarioExperimentLikelihood {
        ScenarioExperimentLikelihood {
            experiment_id: id.to_owned(),
            likelihoods: rows
                .iter()
                .map(|(hypothesis_id, probabilities)| LikelihoodRow {
                    hypothesis_id: (*hypothesis_id).to_owned(),
                    outcome_probabilities: probabilities.to_vec(),
                })
                .collect(),
        }
    }

    fn scenario(
        id: &str,
        weight: f64,
        matrices: Vec<ScenarioExperimentLikelihood>,
    ) -> LikelihoodModelScenario {
        LikelihoodModelScenario {
            scenario_id: id.to_owned(),
            scenario_probability: weight,
            experiments: matrices,
        }
    }

    fn plan(
        hypotheses: &[Hypothesis],
        proposals: &[RobustExperimentProposal],
        scenarios: &[LikelihoodModelScenario],
        decisions: &[TerminalDecision],
        config: DecisionPlannerConfig,
    ) -> Result<DecisionInvestigationPlan, PlannerError> {
        plan_next_experiment_decision_aware(
            &evidence(),
            hypotheses,
            proposals,
            scenarios,
            &utility_model(),
            decisions,
            config,
        )
    }

    #[test]
    fn expected_evsi_matches_a_binary_known_answer() {
        let hypotheses = hypotheses(&[0.5, 0.5]);
        let decisions = decisions(&[("repair-a", &[10.0, 0.0]), ("repair-b", &[0.0, 10.0])]);
        let proposals = vec![proposal("probe", &["yes", "no"])];
        let scenarios = vec![scenario(
            "nominal",
            1.0,
            vec![matrix("probe", &[("h0", &[0.9, 0.1]), ("h1", &[0.1, 0.9])])],
        )];

        let result = plan(
            &hypotheses,
            &proposals,
            &scenarios,
            &decisions,
            DecisionPlannerConfig::default(),
        )
        .unwrap();
        let selected = result.selected.unwrap();

        assert!((result.baseline_expected_utility.unwrap() - 5.0).abs() < 1.0e-12);
        assert!((selected.expected_evsi - 4.0).abs() < 1.0e-12);
        assert!((selected.expected_shared_policy_evsi - 4.0).abs() < 1.0e-12);
        assert!(selected.expected_regret.abs() < 1.0e-12);
        assert!((selected.decision_stability - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn model_generated_support_cannot_ground_a_decision_plan() {
        let mut evidence = evidence();
        evidence[0].origin = super::super::EvidenceOrigin::ModelGenerated;
        evidence[1].origin = super::super::EvidenceOrigin::Synthetic;
        let hypotheses = hypotheses(&[0.5, 0.5]);
        let decisions = decisions(&[("a", &[10.0, 0.0]), ("b", &[0.0, 10.0])]);
        let proposals = vec![proposal("probe", &["yes", "no"])];
        let scenarios = vec![scenario(
            "nominal",
            1.0,
            vec![matrix("probe", &[("h0", &[0.9, 0.1]), ("h1", &[0.1, 0.9])])],
        )];

        let result = plan_next_experiment_decision_aware(
            &evidence,
            &hypotheses,
            &proposals,
            &scenarios,
            &utility_model(),
            &decisions,
            DecisionPlannerConfig::default(),
        )
        .unwrap();

        assert!(result.selected.is_none());
        assert_eq!(
            result.abstention,
            Some(DecisionAbstentionReason::NoIndependentlyGroundedHypothesis)
        );
        assert!(result.baseline_decision_id.is_none());
        assert!(result.baseline_expected_utility.is_none());
    }

    #[test]
    fn information_gain_without_action_change_has_zero_decision_value() {
        let hypotheses = hypotheses(&[0.5, 0.5]);
        let decisions = decisions(&[("keep", &[10.0, 10.0]), ("other", &[0.0, 0.0])]);
        let proposals = vec![proposal("probe", &["yes", "no"])];
        let scenarios = vec![scenario(
            "nominal",
            1.0,
            vec![matrix(
                "probe",
                &[("h0", &[0.99, 0.01]), ("h1", &[0.01, 0.99])],
            )],
        )];
        let plan = plan(
            &hypotheses,
            &proposals,
            &scenarios,
            &decisions,
            DecisionPlannerConfig::default(),
        )
        .unwrap();

        assert!(plan.selected.is_none());
        assert_eq!(plan.baseline_decision_id.as_deref(), Some("keep"));
        assert_eq!(
            plan.rejected_experiments[0].reasons,
            vec![DecisionRejectionReason::NoDecisionValue]
        );
    }

    #[test]
    fn low_entropy_probe_can_have_greater_decision_value_than_high_entropy_probe() {
        let hypotheses = hypotheses(&[0.49, 0.01, 0.50]);
        let decisions = decisions(&[
            ("continue", &[10.0, -500.0, 10.0]),
            ("mitigate", &[9.0, 0.0, 9.0]),
        ]);
        let proposals = vec![
            proposal("benign-detail", &["yes", "no"]),
            proposal("rare-critical-fault", &["yes", "no"]),
        ];
        let scenarios = vec![scenario(
            "nominal",
            1.0,
            vec![
                matrix(
                    "benign-detail",
                    &[
                        ("h0", &[1.0, 0.0]),
                        ("h1", &[1.0, 0.0]),
                        ("h2", &[0.0, 1.0]),
                    ],
                ),
                matrix(
                    "rare-critical-fault",
                    &[
                        ("h0", &[0.0, 1.0]),
                        ("h1", &[1.0, 0.0]),
                        ("h2", &[0.0, 1.0]),
                    ],
                ),
            ],
        )];
        let result = plan(
            &hypotheses,
            &proposals,
            &scenarios,
            &decisions,
            DecisionPlannerConfig::default(),
        )
        .unwrap();

        assert_eq!(result.baseline_decision_id.as_deref(), Some("mitigate"));
        let benign = result
            .ranked_experiments
            .iter()
            .find(|item| item.experiment_id == "benign-detail")
            .unwrap();
        let critical = result
            .ranked_experiments
            .iter()
            .find(|item| item.experiment_id == "rare-critical-fault")
            .unwrap();
        assert!(benign.expected_evsi > critical.expected_evsi);
        assert!(critical.expected_shared_policy_evsi > benign.expected_shared_policy_evsi);
        assert_eq!(
            result.selected.unwrap().experiment_id,
            "rare-critical-fault"
        );
    }

    #[test]
    fn shared_policy_minimax_regret_rejects_scenario_fragile_probe() {
        let hypotheses = hypotheses(&[0.5, 0.5]);
        let mut decisions = decisions(&[("repair-a", &[10.0, 0.0]), ("repair-b", &[0.0, 10.0])]);
        decisions[0].impact = DecisionImpact::ServiceDisruption;
        decisions[1].impact = DecisionImpact::PhysicalActuation;
        let proposals = vec![
            proposal("fragile", &["yes", "no"]),
            proposal("stable", &["yes", "no"]),
        ];
        let scenarios = vec![
            scenario(
                "model-a",
                0.5,
                vec![
                    matrix("fragile", &[("h0", &[0.9, 0.1]), ("h1", &[0.1, 0.9])]),
                    matrix("stable", &[("h0", &[0.9, 0.1]), ("h1", &[0.1, 0.9])]),
                ],
            ),
            scenario(
                "model-b",
                0.5,
                vec![
                    matrix("fragile", &[("h0", &[0.1, 0.9]), ("h1", &[0.9, 0.1])]),
                    matrix("stable", &[("h0", &[0.9, 0.1]), ("h1", &[0.1, 0.9])]),
                ],
            ),
        ];
        let result = plan(
            &hypotheses,
            &proposals,
            &scenarios,
            &decisions,
            DecisionPlannerConfig {
                ranking_policy: DecisionRankingPolicy::MinimaxRegret,
                ..DecisionPlannerConfig::default()
            },
        )
        .unwrap();

        let selected = result.selected.unwrap();
        assert_eq!(selected.experiment_id, "stable");
        assert_eq!(
            selected
                .shared_policy
                .iter()
                .map(|item| item.impact)
                .collect::<Vec<_>>(),
            vec![
                DecisionImpact::ServiceDisruption,
                DecisionImpact::PhysicalActuation
            ]
        );
        assert_eq!(
            selected.scenario_results[0].outcomes[0].shared_policy_decision_impact,
            DecisionImpact::ServiceDisruption
        );
    }

    #[test]
    fn worst_case_evsi_does_not_select_a_probe_that_is_useless_in_a_plausible_scenario() {
        let hypotheses = hypotheses(&[0.5, 0.5]);
        let decisions = decisions(&[("repair-a", &[10.0, 0.0]), ("repair-b", &[0.0, 10.0])]);
        let proposals = vec![proposal("probe", &["yes", "no"])];
        let scenarios = vec![
            scenario(
                "informative",
                0.9,
                vec![matrix(
                    "probe",
                    &[("h0", &[0.99, 0.01]), ("h1", &[0.01, 0.99])],
                )],
            ),
            scenario(
                "uninformative",
                0.1,
                vec![matrix("probe", &[("h0", &[0.5, 0.5]), ("h1", &[0.5, 0.5])])],
            ),
        ];
        let result = plan(
            &hypotheses,
            &proposals,
            &scenarios,
            &decisions,
            DecisionPlannerConfig {
                ranking_policy: DecisionRankingPolicy::WorstCaseEvsi,
                ..DecisionPlannerConfig::default()
            },
        )
        .unwrap();

        assert!(result.selected.is_none());
        assert_eq!(
            result.rejected_experiments[0].reasons,
            vec![DecisionRejectionReason::NoDecisionValue]
        );
        assert_eq!(
            result.abstention,
            Some(DecisionAbstentionReason::NoDecisionValue)
        );
    }

    #[test]
    fn minimax_policy_search_abstains_when_combinatorial_budget_is_exceeded() {
        let hypotheses = hypotheses(&[0.5, 0.5]);
        let decisions = decisions(&[("a", &[10.0, 0.0]), ("b", &[0.0, 10.0])]);
        let proposals = vec![proposal("probe", &["one", "two"])];
        let scenarios = vec![scenario(
            "nominal",
            1.0,
            vec![matrix("probe", &[("h0", &[0.9, 0.1]), ("h1", &[0.1, 0.9])])],
        )];
        let result = plan(
            &hypotheses,
            &proposals,
            &scenarios,
            &decisions,
            DecisionPlannerConfig {
                ranking_policy: DecisionRankingPolicy::MinimaxRegret,
                maximum_policy_combinations: 1,
                ..DecisionPlannerConfig::default()
            },
        )
        .unwrap();
        assert!(result.selected.is_none());
        assert_eq!(
            result.rejected_experiments[0].reasons,
            vec![DecisionRejectionReason::PolicySearchBudgetExceeded]
        );
    }

    #[test]
    fn unsafe_or_unknown_experiment_authorization_is_never_ranked() {
        let hypotheses = hypotheses(&[0.5, 0.5]);
        let decisions = decisions(&[("a", &[10.0, 0.0]), ("b", &[0.0, 10.0])]);
        let mut unauthorized = proposal("probe", &["yes", "no"]);
        unauthorized.authorization = Authorization::Unknown;
        let scenarios = vec![scenario(
            "nominal",
            1.0,
            vec![matrix("probe", &[("h0", &[0.9, 0.1]), ("h1", &[0.1, 0.9])])],
        )];
        let result = plan(
            &hypotheses,
            &[unauthorized],
            &scenarios,
            &decisions,
            DecisionPlannerConfig::default(),
        )
        .unwrap();
        assert!(result.selected.is_none());
        assert_eq!(
            result.rejected_experiments[0].reasons,
            vec![DecisionRejectionReason::AuthorizationUnknown]
        );
    }

    #[test]
    fn utility_matrix_rejects_missing_duplicate_and_non_finite_entries() {
        let hypotheses = hypotheses(&[0.5, 0.5]);
        let proposals = vec![proposal("probe", &["yes", "no"])];
        let scenarios = vec![scenario(
            "nominal",
            1.0,
            vec![matrix("probe", &[("h0", &[0.9, 0.1]), ("h1", &[0.1, 0.9])])],
        )];

        let missing = vec![TerminalDecision {
            id: "bad".to_owned(),
            utilities: vec![DecisionUtility {
                hypothesis_id: "h0".to_owned(),
                utility: 1.0,
            }],
            impact: DecisionImpact::Informational,
        }];
        assert_eq!(
            plan(
                &hypotheses,
                &proposals,
                &scenarios,
                &missing,
                DecisionPlannerConfig::default()
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::InvalidDecisionMatrix
        );

        let duplicate = vec![TerminalDecision {
            id: "bad".to_owned(),
            utilities: vec![
                DecisionUtility {
                    hypothesis_id: "h0".to_owned(),
                    utility: 1.0,
                },
                DecisionUtility {
                    hypothesis_id: "h0".to_owned(),
                    utility: 2.0,
                },
            ],
            impact: DecisionImpact::Informational,
        }];
        assert_eq!(
            plan(
                &hypotheses,
                &proposals,
                &scenarios,
                &duplicate,
                DecisionPlannerConfig::default()
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::InvalidDecisionMatrix
        );
    }

    #[test]
    fn scenario_permutation_preserves_decision_value_and_selection() {
        let hypotheses = hypotheses(&[0.5, 0.5]);
        let decisions = decisions(&[("a", &[10.0, 0.0]), ("b", &[0.0, 10.0])]);
        let proposals = vec![proposal("probe", &["yes", "no"])];
        let first = vec![
            scenario(
                "a",
                0.5,
                vec![matrix("probe", &[("h0", &[0.9, 0.1]), ("h1", &[0.1, 0.9])])],
            ),
            scenario(
                "b",
                0.5,
                vec![matrix("probe", &[("h0", &[0.8, 0.2]), ("h1", &[0.2, 0.8])])],
            ),
        ];
        let second = first.iter().cloned().rev().collect::<Vec<_>>();
        let one = plan(
            &hypotheses,
            &proposals,
            &first,
            &decisions,
            DecisionPlannerConfig::default(),
        )
        .unwrap();
        let two = plan(
            &hypotheses,
            &proposals,
            &second,
            &decisions,
            DecisionPlannerConfig::default(),
        )
        .unwrap();
        assert_eq!(
            one.selected.as_ref().unwrap().experiment_id,
            two.selected.as_ref().unwrap().experiment_id
        );
        assert!(
            (one.selected.unwrap().expected_shared_policy_evsi
                - two.selected.unwrap().expected_shared_policy_evsi)
                .abs()
                < 1.0e-12
        );
    }
}
