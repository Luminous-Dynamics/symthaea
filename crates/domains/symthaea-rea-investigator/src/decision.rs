//! Decision-aware robust experiment planning.
//!
//! This module treats information gain as an intermediate quantity and
//! evaluates whether an experiment is expected to improve a declared downstream
//! decision. Utilities and likelihood scenarios remain caller-supplied; they
//! are not authenticated or calibrated here. No decision is executed.

use std::collections::{BTreeMap, HashMap};

use super::robust::{
    LikelihoodModelScenario, RobustAbstentionReason, RobustExperimentProposal, RobustPlannerConfig,
    RobustRankingPolicy, plan_next_experiment_robust,
};
use super::{EvidenceProvenance, Hypothesis, PlannerConfig, PlannerError, PlannerErrorKind};

#[derive(Debug, Clone, PartialEq)]
pub struct HypothesisUtility {
    pub hypothesis_id: String,
    /// Utility may be negative; higher is better. It must be finite.
    pub utility: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TerminalDecision {
    pub id: String,
    pub utilities: Vec<HypothesisUtility>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionRankingPolicy {
    ExpectedEvsi,
    WorstCaseEvsi,
    LowerTailCvarEvsi,
    ExpectedEvsiMinusWorstCaseRegret,
    /// Minimize the maximum scenario-specific expected regret of one shared
    /// deterministic outcome-to-decision policy.
    MinimaxRegret,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionAwareConfig {
    pub robust: RobustPlannerConfig,
    pub ranking_policy: DecisionRankingPolicy,
    /// Probability mass under the declared scenario/outcome model on which
    /// the chosen shared policy agrees with scenario-specific optimal actions.
    pub minimum_decision_stability: f64,
    /// Optional hard limit on regret of the selected shared policy in any
    /// supplied scenario.
    pub maximum_worst_case_regret: Option<f64>,
    /// Dimensionless multiplier for worst-case regret in the adjusted policy.
    pub regret_penalty: f64,
    /// Minimum selection threshold. For MinimaxRegret this is applied to
    /// expected EVSI; for other policies it is applied to the selected score.
    pub minimum_selection_value: f64,
    /// Hard cap on deterministic outcome-to-decision policies evaluated by
    /// the exact minimax-regret solver. Exceeding it rejects that experiment.
    pub max_policy_evaluations: usize,
}

impl Default for DecisionAwareConfig {
    fn default() -> Self {
        Self {
            robust: RobustPlannerConfig::default(),
            ranking_policy: DecisionRankingPolicy::LowerTailCvarEvsi,
            minimum_decision_stability: 0.5,
            maximum_worst_case_regret: None,
            regret_penalty: 1.0,
            minimum_selection_value: 1.0e-9,
            max_policy_evaluations: 65_536,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionErrorKind {
    PlannerInput,
    InvalidDecisionSet,
    InvalidDecisionUtility,
    InconsistentScenarioOutput,
    PolicySearchBudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionError {
    pub kind: DecisionErrorKind,
    pub message: String,
}

impl DecisionError {
    fn new(kind: DecisionErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn from_planner(error: PlannerError) -> Self {
        Self {
            kind: DecisionErrorKind::PlannerInput,
            message: format!("{:?}: {}", error.kind, error.message),
        }
    }
}

impl std::fmt::Display for DecisionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for DecisionError {}

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
    NoPositiveEigCandidate,
    NoPositiveValueOfInformation,
    DecisionUnstableAcrossScenarios,
    WorstCaseRegretExceeded,
    PolicySearchBudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedDecisionExperiment {
    pub experiment_id: String,
    pub reasons: Vec<DecisionRejectionReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionAbstentionReason {
    NoCompetingHypotheses,
    NoIndependentlyGroundedHypothesis,
    NoEligibleExperiment,
    NoPositiveValueOfInformation,
    DecisionUnstableAcrossScenarios,
    WorstCaseRegretExceeded,
    PolicySearchBudgetExceeded,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOutcomeResult {
    pub outcome: String,
    pub probability: f64,
    pub scenario_optimal_decision_id: String,
    pub shared_policy_decision_id: String,
    pub scenario_optimal_expected_utility: f64,
    pub shared_policy_expected_utility: f64,
    pub decision_regret: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionScenarioResult {
    pub scenario_id: String,
    pub scenario_probability: f64,
    pub expected_posterior_utility: f64,
    pub expected_value_of_sample_information: f64,
    pub regret_of_shared_policy: f64,
    pub outcomes: Vec<DecisionOutcomeResult>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SharedPolicyOutcome {
    pub outcome: String,
    pub mixture_probability: f64,
    pub decision_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionAwareExperimentResult {
    pub experiment_id: String,
    pub baseline_optimal_decision_id: String,
    pub baseline_optimal_expected_utility: f64,
    pub expected_evsi: f64,
    pub worst_case_evsi: f64,
    pub lower_tail_cvar_evsi: f64,
    pub expected_shared_policy_regret: f64,
    pub worst_case_shared_policy_regret: f64,
    pub decision_stability: f64,
    pub selected_value: f64,
    pub normalized_cost: f64,
    pub utility: f64,
    pub shared_policy: Vec<SharedPolicyOutcome>,
    pub scenario_results: Vec<DecisionScenarioResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionClaimCeiling {
    pub evidence_producer_authenticated: bool,
    pub likelihood_scenarios_calibrated: bool,
    pub scenario_weights_authenticated: bool,
    pub decision_utilities_independently_validated: bool,
    pub causal_truth_proven: bool,
    pub selected_experiment_executed: bool,
    pub terminal_decision_executed: bool,
    pub target_mutated: bool,
}

impl Default for DecisionClaimCeiling {
    fn default() -> Self {
        Self {
            evidence_producer_authenticated: false,
            likelihood_scenarios_calibrated: false,
            scenario_weights_authenticated: false,
            decision_utilities_independently_validated: false,
            causal_truth_proven: false,
            selected_experiment_executed: false,
            terminal_decision_executed: false,
            target_mutated: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionAwarePlan {
    pub selected: Option<DecisionAwareExperimentResult>,
    pub ranked_experiments: Vec<DecisionAwareExperimentResult>,
    pub rejected_experiments: Vec<RejectedDecisionExperiment>,
    pub abstention: Option<DecisionAbstentionReason>,
    pub baseline_optimal_decision_id: String,
    pub baseline_optimal_expected_utility: f64,
    pub grounded_hypothesis_ids: Vec<String>,
    pub scenario_ids: Vec<String>,
    pub claim_ceiling: DecisionClaimCeiling,
}

fn validate_config(config: DecisionAwareConfig) -> Result<(), DecisionError> {
    let robust = config.robust;
    let finite_nonnegative = [
        robust.base.max_time_seconds,
        robust.base.max_energy_joules,
        robust.base.max_risk,
        robust.base.cost_weights.time,
        robust.base.cost_weights.energy,
        robust.base.cost_weights.risk,
        robust.base.minimum_information_gain_bits,
        robust.lower_tail_probability,
        config.regret_penalty,
        config.minimum_selection_value,
    ];
    if finite_nonnegative
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || !config.minimum_decision_stability.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_decision_stability)
        || robust.lower_tail_probability <= 0.0
        || robust.lower_tail_probability > 1.0
        || config.max_policy_evaluations == 0
        || config
            .maximum_worst_case_regret
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(DecisionError::new(
            DecisionErrorKind::PlannerInput,
            "decision-aware configuration contains invalid stability, CVaR, regret or selection thresholds",
        ));
    }
    Ok(())
}

fn validate_decisions(
    decisions: &[TerminalDecision],
    hypothesis_ids: &[&str],
) -> Result<Vec<BTreeMap<String, f64>>, DecisionError> {
    if decisions.is_empty() {
        return Err(DecisionError::new(
            DecisionErrorKind::InvalidDecisionSet,
            "at least one terminal decision is required",
        ));
    }

    let mut seen_decisions = std::collections::HashSet::new();
    let mut validated = Vec::with_capacity(decisions.len());
    for decision in decisions {
        if decision.id.trim().is_empty() || !seen_decisions.insert(decision.id.as_str()) {
            return Err(DecisionError::new(
                DecisionErrorKind::InvalidDecisionSet,
                "terminal decision IDs must be unique and non-empty",
            ));
        }
        if decision.utilities.len() != hypothesis_ids.len() {
            return Err(DecisionError::new(
                DecisionErrorKind::InvalidDecisionUtility,
                format!(
                    "decision {} must provide exactly one utility per hypothesis",
                    decision.id
                ),
            ));
        }
        let mut utilities = BTreeMap::new();
        for item in &decision.utilities {
            if !item.utility.is_finite()
                || !hypothesis_ids.contains(&item.hypothesis_id.as_str())
                || utilities
                    .insert(item.hypothesis_id.clone(), item.utility)
                    .is_some()
            {
                return Err(DecisionError::new(
                    DecisionErrorKind::InvalidDecisionUtility,
                    format!(
                        "decision {} has non-finite, duplicate or unknown hypothesis utility",
                        decision.id
                    ),
                ));
            }
        }
        if hypothesis_ids.iter().any(|id| !utilities.contains_key(*id)) {
            return Err(DecisionError::new(
                DecisionErrorKind::InvalidDecisionUtility,
                format!("decision {} omits a hypothesis utility", decision.id),
            ));
        }
        validated.push(utilities);
    }
    Ok(validated)
}

fn best_decision(
    decisions: &[TerminalDecision],
    utility_maps: &[BTreeMap<String, f64>],
    distribution: &BTreeMap<String, f64>,
) -> (String, f64) {
    let mut best_id = String::new();
    let mut best_value = f64::NEG_INFINITY;
    for (decision, utilities) in decisions.iter().zip(utility_maps) {
        let expected_utility = distribution
            .iter()
            .map(|(hypothesis_id, probability)| probability * utilities[hypothesis_id])
            .sum::<f64>();
        if expected_utility > best_value
            || (expected_utility == best_value && (best_id.is_empty() || decision.id < best_id))
        {
            best_id = decision.id.clone();
            best_value = expected_utility;
        }
    }
    (best_id, best_value)
}

fn lower_tail_cvar(samples: &[(f64, f64)], alpha: f64) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut remaining = alpha;
    let mut weighted = 0.0;
    for (value, mass) in ordered {
        if remaining <= 0.0 {
            break;
        }
        let used = mass.min(remaining);
        weighted += value * used;
        remaining -= used;
    }
    weighted / alpha
}

fn normalized_cost(
    estimated_time: f64,
    estimated_energy: f64,
    estimated_risk: f64,
    config: PlannerConfig,
) -> f64 {
    let weighted = config.cost_weights.time * estimated_time / config.max_time_seconds
        + config.cost_weights.energy * estimated_energy / config.max_energy_joules
        + config.cost_weights.risk * estimated_risk / config.max_risk;
    weighted.max(1.0e-12)
}

#[derive(Debug)]
struct CandidateDecisionData {
    result: DecisionAwareExperimentResult,
    worst_regret: f64,
}

/// Find the deterministic outcome-to-decision policy with minimum maximum
/// expected regret over the supplied finite scenario set.
///
/// This is an exact finite search, not a guarantee over environments outside
/// the supplied scenario set. The checked policy-count cap prevents accidental
/// exponential work; the caller receives an explicit rejection on overflow or
/// budget exhaustion. Ties resolve lexicographically by decision IDs in the
/// declared outcome order.
fn minimax_shared_policy(
    outcomes: &[String],
    scenario_results: &[super::robust::ScenarioInformationGain],
    decisions: &[TerminalDecision],
    utility_maps: &[BTreeMap<String, f64>],
    prior_distribution: &BTreeMap<String, f64>,
    max_policy_evaluations: usize,
) -> Result<HashMap<String, String>, DecisionError> {
    if outcomes.is_empty() || decisions.is_empty() || scenario_results.is_empty() {
        return Err(DecisionError::new(
            DecisionErrorKind::InconsistentScenarioOutput,
            "minimax-regret search requires outcomes, decisions and scenarios",
        ));
    }

    let outcome_count = u32::try_from(outcomes.len()).map_err(|_| {
        DecisionError::new(
            DecisionErrorKind::PolicySearchBudgetExceeded,
            "outcome count exceeds exact minimax policy-search indexing",
        )
    })?;
    let policy_count = decisions.len().checked_pow(outcome_count).ok_or_else(|| {
        DecisionError::new(
            DecisionErrorKind::PolicySearchBudgetExceeded,
            "minimax-regret policy count overflowed usize",
        )
    })?;
    if policy_count > max_policy_evaluations {
        return Err(DecisionError::new(
            DecisionErrorKind::PolicySearchBudgetExceeded,
            format!(
                "exact minimax-regret search requires {policy_count} policies; configured cap is {max_policy_evaluations}"
            ),
        ));
    }

    let mut decision_order: Vec<usize> = (0..decisions.len()).collect();
    decision_order.sort_by(|left, right| decisions[*left].id.cmp(&decisions[*right].id));

    let mut best_worst_regret = f64::INFINITY;
    let mut best_policy_key: Option<Vec<String>> = None;
    let mut best_policy: Option<HashMap<String, String>> = None;

    for ordinal in 0..policy_count {
        // Decode with the final outcome as the least-significant digit. With
        // decisions sorted by ID this visits policies in lexical order.
        let mut remainder = ordinal;
        let mut selected_indices = vec![0usize; outcomes.len()];
        for outcome_index in (0..outcomes.len()).rev() {
            let digit = remainder % decisions.len();
            remainder /= decisions.len();
            selected_indices[outcome_index] = decision_order[digit];
        }

        let policy_key: Vec<String> = selected_indices
            .iter()
            .map(|index| decisions[*index].id.clone())
            .collect();

        let mut worst_regret = 0.0f64;
        for scenario in scenario_results {
            if scenario.predicted_outcomes.len() != outcomes.len()
                || scenario
                    .predicted_outcomes
                    .iter()
                    .zip(outcomes)
                    .any(|(item, expected)| item.outcome != *expected)
            {
                return Err(DecisionError::new(
                    DecisionErrorKind::InconsistentScenarioOutput,
                    "scenario outcome labels/order differ during minimax-regret search",
                ));
            }

            let mut expected_regret = 0.0;
            for (outcome_index, predicted) in scenario.predicted_outcomes.iter().enumerate() {
                let posterior: BTreeMap<String, f64> =
                    predicted.posterior.iter().cloned().collect();
                if posterior.keys().ne(prior_distribution.keys()) {
                    return Err(DecisionError::new(
                        DecisionErrorKind::InconsistentScenarioOutput,
                        "scenario posterior hypothesis set differs from planner prior",
                    ));
                }

                let (_, optimal_utility) = best_decision(decisions, utility_maps, &posterior);
                let selected_index = selected_indices[outcome_index];
                let selected_utility = posterior
                    .iter()
                    .map(|(hypothesis_id, probability)| {
                        probability * utility_maps[selected_index][hypothesis_id]
                    })
                    .sum::<f64>();
                expected_regret +=
                    predicted.probability * (optimal_utility - selected_utility).max(0.0);
            }
            worst_regret = worst_regret.max(expected_regret);
        }

        let replace = match best_policy_key.as_ref() {
            None => true,
            Some(current_key) => {
                worst_regret.total_cmp(&best_worst_regret).is_lt()
                    || (worst_regret.total_cmp(&best_worst_regret).is_eq()
                        && policy_key.as_slice() < current_key.as_slice())
            }
        };
        if replace {
            best_worst_regret = worst_regret;
            best_policy_key = Some(policy_key);
            best_policy = Some(
                outcomes
                    .iter()
                    .zip(&selected_indices)
                    .map(|(outcome, index)| (outcome.clone(), decisions[*index].id.clone()))
                    .collect(),
            );
        }
    }

    best_policy.ok_or_else(|| {
        DecisionError::new(
            DecisionErrorKind::InconsistentScenarioOutput,
            "minimax-regret search produced no deterministic policy",
        )
    })
}

fn evaluate_candidate(
    experiment_id: &str,
    scenario_results: &[super::robust::ScenarioInformationGain],
    decisions: &[TerminalDecision],
    utility_maps: &[BTreeMap<String, f64>],
    prior_distribution: &BTreeMap<String, f64>,
    baseline_decision: &str,
    baseline_value: f64,
    normalized_cost: f64,
    config: DecisionAwareConfig,
) -> Result<CandidateDecisionData, DecisionError> {
    if scenario_results.is_empty() {
        return Err(DecisionError::new(
            DecisionErrorKind::InconsistentScenarioOutput,
            "robust planner returned no scenario results for an eligible experiment",
        ));
    }

    let first = &scenario_results[0];
    let outcomes: Vec<String> = first
        .predicted_outcomes
        .iter()
        .map(|outcome| outcome.outcome.clone())
        .collect();

    for scenario in scenario_results {
        if scenario.predicted_outcomes.len() != outcomes.len()
            || scenario
                .predicted_outcomes
                .iter()
                .zip(&outcomes)
                .any(|(item, expected)| item.outcome != *expected)
        {
            return Err(DecisionError::new(
                DecisionErrorKind::InconsistentScenarioOutput,
                "scenario outcome labels/order differ for the same experiment",
            ));
        }
    }

    // MinimaxRegret uses an exact bounded search over deterministic shared
    // outcome-to-decision policies. Other policies use the Bayes action under
    // the scenario-mixture posterior for each outcome.
    let minimax_policy = if config.ranking_policy == DecisionRankingPolicy::MinimaxRegret {
        Some(minimax_shared_policy(
            &outcomes,
            scenario_results,
            decisions,
            utility_maps,
            prior_distribution,
            config.max_policy_evaluations,
        )?)
    } else {
        None
    };

    // Construct the shared mixture posterior for each outcome. The mixture is
    // weighted by scenario probability times that scenario's outcome mass;
    // averaging posterior vectors directly would be mathematically incorrect.
    let mut shared_policy = Vec::with_capacity(outcomes.len());
    let mut shared_decision_by_outcome = HashMap::new();

    for (outcome_index, outcome) in outcomes.iter().enumerate() {
        let mixture_mass = scenario_results
            .iter()
            .map(|scenario| {
                scenario.scenario_probability
                    * scenario.predicted_outcomes[outcome_index].probability
            })
            .sum::<f64>();

        let mut mixture_joint: BTreeMap<String, f64> = prior_distribution
            .keys()
            .map(|hypothesis_id| (hypothesis_id.clone(), 0.0))
            .collect();
        if mixture_mass > 0.0 {
            for scenario in scenario_results {
                let predicted = &scenario.predicted_outcomes[outcome_index];
                let scenario_mass = scenario.scenario_probability * predicted.probability;
                for (hypothesis_id, posterior_probability) in &predicted.posterior {
                    let current = mixture_joint.get_mut(hypothesis_id).ok_or_else(|| {
                        DecisionError::new(
                            DecisionErrorKind::InconsistentScenarioOutput,
                            "scenario posterior references a hypothesis absent from the prior",
                        )
                    })?;
                    *current += scenario_mass * posterior_probability;
                }
            }
            for value in mixture_joint.values_mut() {
                *value /= mixture_mass;
            }
        } else {
            // Impossible under all supplied scenarios; choose the prior for a
            // deterministic but zero-mass outcome. It contributes no value.
            mixture_joint = prior_distribution.clone();
        }

        let shared_decision_id = match minimax_policy.as_ref() {
            Some(policy) => policy.get(outcome).cloned().ok_or_else(|| {
                DecisionError::new(
                    DecisionErrorKind::InconsistentScenarioOutput,
                    "minimax policy omitted a declared outcome",
                )
            })?,
            None => best_decision(decisions, utility_maps, &mixture_joint).0,
        };
        shared_decision_by_outcome.insert(outcome.clone(), shared_decision_id.clone());
        shared_policy.push(SharedPolicyOutcome {
            outcome: outcome.clone(),
            mixture_probability: mixture_mass,
            decision_id: shared_decision_id,
        });
    }

    let mut scenario_decision_results = Vec::with_capacity(scenario_results.len());
    let mut evsi_samples = Vec::with_capacity(scenario_results.len());
    let mut regret_samples = Vec::with_capacity(scenario_results.len());
    let mut decision_agreement_mass = 0.0;

    for scenario in scenario_results {
        let mut expected_posterior_utility = 0.0;
        let mut regret = 0.0;
        let mut outcome_results = Vec::with_capacity(outcomes.len());

        for predicted in &scenario.predicted_outcomes {
            let posterior: BTreeMap<String, f64> = predicted.posterior.iter().cloned().collect();
            if posterior.keys().ne(prior_distribution.keys()) {
                return Err(DecisionError::new(
                    DecisionErrorKind::InconsistentScenarioOutput,
                    "scenario posterior hypothesis set differs from planner prior",
                ));
            }

            let (scenario_best_id, scenario_best_utility) =
                best_decision(decisions, utility_maps, &posterior);
            let shared_id = shared_decision_by_outcome[&predicted.outcome].clone();
            let shared_utility = utility_maps
                .iter()
                .zip(decisions)
                .find(|(_, decision)| decision.id == shared_id)
                .map(|(utilities, _)| {
                    posterior
                        .iter()
                        .map(|(hypothesis_id, probability)| probability * utilities[hypothesis_id])
                        .sum::<f64>()
                })
                .ok_or_else(|| {
                    DecisionError::new(
                        DecisionErrorKind::InconsistentScenarioOutput,
                        "shared policy selected an unknown decision",
                    )
                })?;

            let decision_regret = (scenario_best_utility - shared_utility).max(0.0);
            expected_posterior_utility += predicted.probability * scenario_best_utility;
            regret += predicted.probability * decision_regret;

            let joint_mass = scenario.scenario_probability * predicted.probability;
            if scenario_best_id == shared_id {
                decision_agreement_mass += joint_mass;
            }

            outcome_results.push(DecisionOutcomeResult {
                outcome: predicted.outcome.clone(),
                probability: predicted.probability,
                scenario_optimal_decision_id: scenario_best_id,
                shared_policy_decision_id: shared_id,
                scenario_optimal_expected_utility: scenario_best_utility,
                shared_policy_expected_utility: shared_utility,
                decision_regret,
            });
        }

        let evsi = expected_posterior_utility - baseline_value;
        evsi_samples.push((evsi, scenario.scenario_probability));
        regret_samples.push((regret, scenario.scenario_probability));
        scenario_decision_results.push(DecisionScenarioResult {
            scenario_id: scenario.scenario_id.clone(),
            scenario_probability: scenario.scenario_probability,
            expected_posterior_utility,
            expected_value_of_sample_information: evsi,
            regret_of_shared_policy: regret,
            outcomes: outcome_results,
        });
    }

    let expected_evsi = evsi_samples
        .iter()
        .map(|(value, probability)| value * probability)
        .sum::<f64>();
    let worst_case_evsi = evsi_samples
        .iter()
        .map(|(value, _)| *value)
        .fold(f64::INFINITY, f64::min);
    let lower_tail_cvar_evsi = lower_tail_cvar(&evsi_samples, config.robust.lower_tail_probability);
    let expected_shared_policy_regret = regret_samples
        .iter()
        .map(|(value, probability)| value * probability)
        .sum::<f64>();
    let worst_case_shared_policy_regret = regret_samples
        .iter()
        .map(|(value, _)| *value)
        .fold(0.0, f64::max);

    // Probability mass for which the actual shared policy action agrees
    // with the scenario-specific optimal action. This is policy-specific;
    // it must not report the best achievable modal agreement instead.
    let decision_stability = decision_agreement_mass.clamp(0.0, 1.0);

    let selected_value = match config.ranking_policy {
        DecisionRankingPolicy::ExpectedEvsi => expected_evsi,
        DecisionRankingPolicy::WorstCaseEvsi => worst_case_evsi,
        DecisionRankingPolicy::LowerTailCvarEvsi => lower_tail_cvar_evsi,
        DecisionRankingPolicy::ExpectedEvsiMinusWorstCaseRegret => {
            expected_evsi - config.regret_penalty * worst_case_shared_policy_regret
        }
        DecisionRankingPolicy::MinimaxRegret => {
            // Positive, monotone score for reporting; candidates are actually
            // ordered by raw worst-case regret first in the caller.
            1.0 / (1.0 + worst_case_shared_policy_regret)
        }
    };

    let utility = selected_value / normalized_cost;
    Ok(CandidateDecisionData {
        result: DecisionAwareExperimentResult {
            experiment_id: experiment_id.to_owned(),
            baseline_optimal_decision_id: baseline_decision.to_owned(),
            baseline_optimal_expected_utility: baseline_value,
            expected_evsi,
            worst_case_evsi,
            lower_tail_cvar_evsi,
            expected_shared_policy_regret,
            worst_case_shared_policy_regret,
            decision_stability,
            selected_value,
            normalized_cost,
            utility,
            shared_policy,
            scenario_results: scenario_decision_results,
        },
        worst_regret: worst_case_shared_policy_regret,
    })
}

/// Select an experiment by the expected improvement and robustness of the
/// downstream decision, rather than information gain alone.
///
/// EIG remains computed by the robust planner as an intermediate diagnostic;
/// the selection objective is EVSI, a conservative EVSI criterion, or an
/// EVSI-minus-regret score. This function never executes the experiment or the
/// terminal decision.
pub fn plan_next_experiment_decision_aware(
    evidence: &[EvidenceProvenance],
    hypotheses: &[Hypothesis],
    experiments: &[RobustExperimentProposal],
    scenarios: &[LikelihoodModelScenario],
    decisions: &[TerminalDecision],
    config: DecisionAwareConfig,
) -> Result<DecisionAwarePlan, DecisionError> {
    super::validate_config(config.robust.base).map_err(DecisionError::from_planner)?;
    validate_config(config)?;

    let catalog =
        super::validate_evidence_catalog(evidence).map_err(DecisionError::from_planner)?;
    let (prior_map, grounded_ids, _qualified_support, _independent_roots) =
        match super::validate_hypotheses(hypotheses, &catalog) {
            Ok(result) => result,
            Err(error) if error.kind == PlannerErrorKind::UngroundedHypothesis => {
                return Ok(DecisionAwarePlan {
                    selected: None,
                    ranked_experiments: Vec::new(),
                    rejected_experiments: Vec::new(),
                    abstention: Some(DecisionAbstentionReason::NoIndependentlyGroundedHypothesis),
                    baseline_optimal_decision_id: String::new(),
                    baseline_optimal_expected_utility: 0.0,
                    grounded_hypothesis_ids: Vec::new(),
                    scenario_ids: Vec::new(),
                    claim_ceiling: DecisionClaimCeiling::default(),
                });
            }
            Err(error) => return Err(DecisionError::from_planner(error)),
        };

    let hypothesis_ids: Vec<&str> = prior_map.keys().copied().collect();
    let utility_maps = validate_decisions(decisions, &hypothesis_ids)?;
    let prior_distribution: BTreeMap<String, f64> = prior_map
        .iter()
        .map(|(id, probability)| ((*id).to_owned(), *probability))
        .collect();
    let (baseline_decision_id, baseline_value) =
        best_decision(decisions, &utility_maps, &prior_distribution);

    if hypotheses.len() < 2 {
        return Ok(DecisionAwarePlan {
            selected: None,
            ranked_experiments: Vec::new(),
            rejected_experiments: Vec::new(),
            abstention: Some(DecisionAbstentionReason::NoCompetingHypotheses),
            baseline_optimal_decision_id: baseline_decision_id,
            baseline_optimal_expected_utility: baseline_value,
            grounded_hypothesis_ids: grounded_ids,
            scenario_ids: Vec::new(),
            claim_ceiling: DecisionClaimCeiling::default(),
        });
    }

    // Reuse the same authorization, reversibility, target-impact and resource
    // gate. EIG is only a diagnostic here; all posterior data come from the
    // same shared scenarios and evidence roots.
    let mut robust_config = config.robust;
    robust_config.minimum_top1_stability = 0.0;
    robust_config.ranking_policy = RobustRankingPolicy::ExpectedInformationGain;
    robust_config.base.minimum_information_gain_bits = 0.0;
    let robust =
        plan_next_experiment_robust(evidence, hypotheses, experiments, scenarios, robust_config)
            .map_err(DecisionError::from_planner)?;

    if robust.abstention == Some(RobustAbstentionReason::NoIndependentlyGroundedHypothesis) {
        return Ok(DecisionAwarePlan {
            selected: None,
            ranked_experiments: Vec::new(),
            rejected_experiments: Vec::new(),
            abstention: Some(DecisionAbstentionReason::NoIndependentlyGroundedHypothesis),
            baseline_optimal_decision_id: baseline_decision_id,
            baseline_optimal_expected_utility: baseline_value,
            grounded_hypothesis_ids: grounded_ids,
            scenario_ids: robust.scenario_ids,
            claim_ceiling: DecisionClaimCeiling::default(),
        });
    }

    let experiment_lookup: HashMap<&str, &RobustExperimentProposal> = experiments
        .iter()
        .map(|item| (item.id.as_str(), item))
        .collect();
    let mut ranked = Vec::new();
    let mut rejected = Vec::new();

    for item in &robust.rejected_experiments {
        let reason = match item.reason {
            super::robust::RobustRejectionReason::AuthorizationUnknown => {
                DecisionRejectionReason::AuthorizationUnknown
            }
            super::robust::RobustRejectionReason::NotAuthorized => {
                DecisionRejectionReason::NotAuthorized
            }
            super::robust::RobustRejectionReason::Irreversible => {
                DecisionRejectionReason::Irreversible
            }
            super::robust::RobustRejectionReason::Destructive => {
                DecisionRejectionReason::Destructive
            }
            super::robust::RobustRejectionReason::TargetImpactExceeded => {
                DecisionRejectionReason::TargetImpactExceeded
            }
            super::robust::RobustRejectionReason::TimeBudgetExceeded => {
                DecisionRejectionReason::TimeBudgetExceeded
            }
            super::robust::RobustRejectionReason::EnergyBudgetExceeded => {
                DecisionRejectionReason::EnergyBudgetExceeded
            }
            super::robust::RobustRejectionReason::RiskBudgetExceeded => {
                DecisionRejectionReason::RiskBudgetExceeded
            }
            super::robust::RobustRejectionReason::InvalidOutcomeSpace => {
                DecisionRejectionReason::InvalidOutcomeSpace
            }
            super::robust::RobustRejectionReason::NoRobustInformationGain => {
                DecisionRejectionReason::NoPositiveEigCandidate
            }
            super::robust::RobustRejectionReason::RankingUnstable => {
                DecisionRejectionReason::DecisionUnstableAcrossScenarios
            }
        };
        rejected.push(RejectedDecisionExperiment {
            experiment_id: item.experiment_id.clone(),
            reasons: vec![reason],
        });
    }

    for candidate in &robust.ranked_experiments {
        let experiment = experiment_lookup[candidate.experiment_id.as_str()];
        let cost = normalized_cost(
            experiment.estimated_time_seconds,
            experiment.estimated_energy_joules,
            experiment.estimated_risk,
            config.robust.base,
        );
        let evaluated = match evaluate_candidate(
            &candidate.experiment_id,
            &candidate.scenario_results,
            decisions,
            &utility_maps,
            &prior_distribution,
            &baseline_decision_id,
            baseline_value,
            cost,
            config,
        ) {
            Ok(result) => result,
            Err(error) if error.kind == DecisionErrorKind::PolicySearchBudgetExceeded => {
                rejected.push(RejectedDecisionExperiment {
                    experiment_id: candidate.experiment_id.clone(),
                    reasons: vec![DecisionRejectionReason::PolicySearchBudgetExceeded],
                });
                continue;
            }
            Err(error) => return Err(error),
        };

        let mut reasons = Vec::new();
        let lacks_value = if config.ranking_policy == DecisionRankingPolicy::MinimaxRegret {
            evaluated.result.expected_evsi <= config.minimum_selection_value
        } else {
            evaluated.result.selected_value <= config.minimum_selection_value
        };
        if lacks_value {
            reasons.push(DecisionRejectionReason::NoPositiveValueOfInformation);
        }
        if evaluated.result.decision_stability < config.minimum_decision_stability {
            reasons.push(DecisionRejectionReason::DecisionUnstableAcrossScenarios);
        }
        if config
            .maximum_worst_case_regret
            .is_some_and(|limit| evaluated.worst_regret > limit)
        {
            reasons.push(DecisionRejectionReason::WorstCaseRegretExceeded);
        }
        if reasons.is_empty() {
            ranked.push(evaluated.result);
        } else {
            rejected.push(RejectedDecisionExperiment {
                experiment_id: candidate.experiment_id.clone(),
                reasons,
            });
        }
    }

    if config.ranking_policy == DecisionRankingPolicy::MinimaxRegret {
        ranked.sort_by(|left, right| {
            left.worst_case_shared_policy_regret
                .total_cmp(&right.worst_case_shared_policy_regret)
                .then_with(|| right.expected_evsi.total_cmp(&left.expected_evsi))
                .then_with(|| left.normalized_cost.total_cmp(&right.normalized_cost))
                .then_with(|| left.experiment_id.cmp(&right.experiment_id))
        });
    } else {
        ranked.sort_by(|left, right| {
            right
                .utility
                .total_cmp(&left.utility)
                .then_with(|| right.selected_value.total_cmp(&left.selected_value))
                .then_with(|| left.experiment_id.cmp(&right.experiment_id))
        });
    }
    rejected.sort_by(|left, right| left.experiment_id.cmp(&right.experiment_id));

    let selected = ranked.first().cloned();
    let abstention = if selected.is_some() {
        None
    } else if rejected.iter().any(|item| {
        item.reasons
            .contains(&DecisionRejectionReason::DecisionUnstableAcrossScenarios)
    }) {
        Some(DecisionAbstentionReason::DecisionUnstableAcrossScenarios)
    } else if rejected.iter().any(|item| {
        item.reasons
            .contains(&DecisionRejectionReason::WorstCaseRegretExceeded)
    }) {
        Some(DecisionAbstentionReason::WorstCaseRegretExceeded)
    } else if rejected.iter().any(|item| {
        item.reasons
            .contains(&DecisionRejectionReason::PolicySearchBudgetExceeded)
    }) {
        Some(DecisionAbstentionReason::PolicySearchBudgetExceeded)
    } else if !rejected.is_empty()
        && rejected.iter().all(|item| {
            item.reasons.iter().all(|reason| {
                matches!(
                    reason,
                    DecisionRejectionReason::NoPositiveEigCandidate
                        | DecisionRejectionReason::NoPositiveValueOfInformation
                )
            })
        })
    {
        Some(DecisionAbstentionReason::NoPositiveValueOfInformation)
    } else {
        Some(DecisionAbstentionReason::NoEligibleExperiment)
    };

    Ok(DecisionAwarePlan {
        selected,
        ranked_experiments: ranked,
        rejected_experiments: rejected,
        abstention,
        baseline_optimal_decision_id: baseline_decision_id,
        baseline_optimal_expected_utility: baseline_value,
        grounded_hypothesis_ids: grounded_ids,
        scenario_ids: robust.scenario_ids,
        claim_ceiling: DecisionClaimCeiling::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::robust::{
        LikelihoodModelScenario, RobustExperimentProposal, RobustPlannerConfig,
        ScenarioExperimentLikelihood,
    };
    use super::super::{
        Authorization, EvidenceOrigin, EvidenceProvenance, Hypothesis, LikelihoodRow, TargetImpact,
    };
    use super::*;

    fn ev_id(byte: char) -> String {
        format!("ev_{}", byte.to_string().repeat(64))
    }

    fn evidence() -> Vec<EvidenceProvenance> {
        vec![EvidenceProvenance {
            evidence_id: ev_id('a'),
            origin: EvidenceOrigin::IndependentObservation,
            parent_evidence_ids: vec![],
        }]
    }

    fn hypotheses() -> Vec<Hypothesis> {
        vec![
            Hypothesis {
                id: "cause-a".to_owned(),
                prior_probability: 0.5,
                supporting_evidence_ids: vec![ev_id('a')],
                contradicting_evidence_ids: vec![],
            },
            Hypothesis {
                id: "cause-b".to_owned(),
                prior_probability: 0.5,
                supporting_evidence_ids: vec![ev_id('a')],
                contradicting_evidence_ids: vec![],
            },
        ]
    }

    fn proposal(id: &str) -> RobustExperimentProposal {
        RobustExperimentProposal {
            id: id.to_owned(),
            outcomes: vec!["yes".to_owned(), "no".to_owned()],
            authorization: Authorization::Authorized,
            reversible: true,
            destructive: false,
            target_impact: TargetImpact::ReadOnlyObservation,
            estimated_time_seconds: 10.0,
            estimated_energy_joules: 10.0,
            estimated_risk: 0.01,
        }
    }

    fn matrix(id: &str, a_yes: f64, b_yes: f64) -> ScenarioExperimentLikelihood {
        ScenarioExperimentLikelihood {
            experiment_id: id.to_owned(),
            likelihoods: vec![
                LikelihoodRow {
                    hypothesis_id: "cause-a".to_owned(),
                    outcome_probabilities: vec![a_yes, 1.0 - a_yes],
                },
                LikelihoodRow {
                    hypothesis_id: "cause-b".to_owned(),
                    outcome_probabilities: vec![b_yes, 1.0 - b_yes],
                },
            ],
        }
    }

    fn scenario(
        id: &str,
        probability: f64,
        experiments: Vec<ScenarioExperimentLikelihood>,
    ) -> LikelihoodModelScenario {
        LikelihoodModelScenario {
            scenario_id: id.to_owned(),
            scenario_probability: probability,
            experiments,
        }
    }

    fn decisions() -> Vec<TerminalDecision> {
        vec![
            TerminalDecision {
                id: "remediate-a".to_owned(),
                utilities: vec![
                    HypothesisUtility {
                        hypothesis_id: "cause-a".to_owned(),
                        utility: 1.0,
                    },
                    HypothesisUtility {
                        hypothesis_id: "cause-b".to_owned(),
                        utility: 0.0,
                    },
                ],
            },
            TerminalDecision {
                id: "remediate-b".to_owned(),
                utilities: vec![
                    HypothesisUtility {
                        hypothesis_id: "cause-a".to_owned(),
                        utility: 0.0,
                    },
                    HypothesisUtility {
                        hypothesis_id: "cause-b".to_owned(),
                        utility: 1.0,
                    },
                ],
            },
        ]
    }

    #[test]
    fn known_answer_binary_test_has_evsi_of_point_nine_accuracy() {
        let plan = plan_next_experiment_decision_aware(
            &evidence(),
            &hypotheses(),
            &[proposal("probe")],
            &[scenario("nominal", 1.0, vec![matrix("probe", 0.9, 0.1)])],
            &decisions(),
            DecisionAwareConfig {
                minimum_decision_stability: 0.0,
                ..DecisionAwareConfig::default()
            },
        )
        .unwrap();
        let selected = plan.selected.unwrap();
        assert!((selected.baseline_optimal_expected_utility - 0.5).abs() < 1e-12);
        assert!((selected.expected_evsi - 0.4).abs() < 1e-12);
        assert!((selected.worst_case_evsi - 0.4).abs() < 1e-12);
        assert!(selected.worst_case_shared_policy_regret.abs() < 1e-12);
        assert!((selected.decision_stability - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lower_eig_probe_can_beat_high_eig_when_only_it_changes_the_decision() {
        // probe-information reveals a nuisance distinction that does not change
        // the remediation; action-probe separates the two actionable causes.
        let decisions = vec![
            TerminalDecision {
                id: "safe-default".to_owned(),
                utilities: vec![
                    HypothesisUtility {
                        hypothesis_id: "a".into(),
                        utility: 1.0,
                    },
                    HypothesisUtility {
                        hypothesis_id: "b".into(),
                        utility: 1.0,
                    },
                    HypothesisUtility {
                        hypothesis_id: "c".into(),
                        utility: 0.0,
                    },
                ],
            },
            TerminalDecision {
                id: "special-remediation".to_owned(),
                utilities: vec![
                    HypothesisUtility {
                        hypothesis_id: "a".into(),
                        utility: 0.0,
                    },
                    HypothesisUtility {
                        hypothesis_id: "b".into(),
                        utility: 0.0,
                    },
                    HypothesisUtility {
                        hypothesis_id: "c".into(),
                        utility: 1.0,
                    },
                ],
            },
        ];
        let evidence = vec![EvidenceProvenance {
            evidence_id: ev_id('a'),
            origin: EvidenceOrigin::IndependentObservation,
            parent_evidence_ids: vec![],
        }];
        let hypotheses = vec![
            Hypothesis {
                id: "a".into(),
                prior_probability: 0.25,
                supporting_evidence_ids: vec![ev_id('a')],
                contradicting_evidence_ids: vec![],
            },
            Hypothesis {
                id: "b".into(),
                prior_probability: 0.25,
                supporting_evidence_ids: vec![ev_id('a')],
                contradicting_evidence_ids: vec![],
            },
            Hypothesis {
                id: "c".into(),
                prior_probability: 0.5,
                supporting_evidence_ids: vec![ev_id('a')],
                contradicting_evidence_ids: vec![],
            },
        ];
        let high_eig_no_action_change = RobustExperimentProposal {
            id: "diagnose-nuisance".into(),
            outcomes: vec!["yes".into(), "no".into()],
            authorization: Authorization::Authorized,
            reversible: true,
            destructive: false,
            target_impact: TargetImpact::ReadOnlyObservation,
            estimated_time_seconds: 10.0,
            estimated_energy_joules: 10.0,
            estimated_risk: 0.01,
        };
        let lower_eig_action_change = proposal("identify-actionable-cause");
        let high_matrix = ScenarioExperimentLikelihood {
            experiment_id: "diagnose-nuisance".into(),
            likelihoods: vec![
                LikelihoodRow {
                    hypothesis_id: "a".into(),
                    outcome_probabilities: vec![0.99, 0.01],
                },
                LikelihoodRow {
                    hypothesis_id: "b".into(),
                    outcome_probabilities: vec![0.01, 0.99],
                },
                LikelihoodRow {
                    hypothesis_id: "c".into(),
                    outcome_probabilities: vec![0.50, 0.50],
                },
            ],
        };
        let low_matrix = ScenarioExperimentLikelihood {
            experiment_id: "identify-actionable-cause".into(),
            likelihoods: vec![
                LikelihoodRow {
                    hypothesis_id: "a".into(),
                    outcome_probabilities: vec![0.8, 0.2],
                },
                LikelihoodRow {
                    hypothesis_id: "b".into(),
                    outcome_probabilities: vec![0.8, 0.2],
                },
                LikelihoodRow {
                    hypothesis_id: "c".into(),
                    outcome_probabilities: vec![0.2, 0.8],
                },
            ],
        };
        let scenario = LikelihoodModelScenario {
            scenario_id: "nominal".into(),
            scenario_probability: 1.0,
            experiments: vec![high_matrix, low_matrix],
        };
        let point = super::super::robust::plan_next_experiment_robust(
            &evidence,
            &hypotheses,
            &[
                high_eig_no_action_change.clone(),
                lower_eig_action_change.clone(),
            ],
            &[scenario.clone()],
            RobustPlannerConfig {
                minimum_top1_stability: 0.0,
                ranking_policy: super::super::robust::RobustRankingPolicy::ExpectedInformationGain,
                ..RobustPlannerConfig::default()
            },
        )
        .unwrap();
        let high_eig = point
            .ranked_experiments
            .iter()
            .find(|x| x.experiment_id == "diagnose-nuisance")
            .unwrap()
            .expected_information_gain_bits;
        let low_eig = point
            .ranked_experiments
            .iter()
            .find(|x| x.experiment_id == "identify-actionable-cause")
            .unwrap()
            .expected_information_gain_bits;
        assert!(high_eig > low_eig);

        let decision_plan = plan_next_experiment_decision_aware(
            &evidence,
            &hypotheses,
            &[high_eig_no_action_change, lower_eig_action_change],
            &[scenario],
            &decisions,
            DecisionAwareConfig {
                minimum_decision_stability: 0.0,
                ..DecisionAwareConfig::default()
            },
        )
        .unwrap();
        assert_eq!(
            decision_plan.selected.unwrap().experiment_id,
            "identify-actionable-cause"
        );
    }

    #[test]
    fn rejects_incomplete_utility_matrix() {
        let mut invalid = decisions();
        invalid[0].utilities.pop();
        assert_eq!(
            plan_next_experiment_decision_aware(
                &evidence(),
                &hypotheses(),
                &[proposal("probe")],
                &[scenario("nominal", 1.0, vec![matrix("probe", 0.9, 0.1)])],
                &invalid,
                DecisionAwareConfig::default(),
            )
            .unwrap_err()
            .kind,
            DecisionErrorKind::InvalidDecisionUtility
        );
    }

    #[test]
    fn no_positive_decision_value_causes_abstention_even_when_eig_is_positive() {
        let decisions = vec![TerminalDecision {
            id: "always-default".into(),
            utilities: vec![
                HypothesisUtility {
                    hypothesis_id: "cause-a".into(),
                    utility: 1.0,
                },
                HypothesisUtility {
                    hypothesis_id: "cause-b".into(),
                    utility: 1.0,
                },
            ],
        }];
        let plan = plan_next_experiment_decision_aware(
            &evidence(),
            &hypotheses(),
            &[proposal("probe")],
            &[scenario("nominal", 1.0, vec![matrix("probe", 0.9, 0.1)])],
            &decisions,
            DecisionAwareConfig {
                minimum_decision_stability: 0.0,
                ..DecisionAwareConfig::default()
            },
        )
        .unwrap();
        assert!(plan.selected.is_none());
        assert_eq!(
            plan.abstention,
            Some(DecisionAbstentionReason::NoPositiveValueOfInformation)
        );
    }

    #[test]
    fn preserves_authorization_rejection_in_decision_aware_report() {
        let mut unsafe_probe = proposal("unsafe-probe");
        unsafe_probe.authorization = Authorization::Denied;
        let plan = plan_next_experiment_decision_aware(
            &evidence(),
            &hypotheses(),
            &[unsafe_probe],
            &[scenario(
                "nominal",
                1.0,
                vec![matrix("unsafe-probe", 0.9, 0.1)],
            )],
            &decisions(),
            DecisionAwareConfig::default(),
        )
        .unwrap();
        assert!(plan.selected.is_none());
        assert_eq!(
            plan.abstention,
            Some(DecisionAbstentionReason::NoEligibleExperiment)
        );
        assert_eq!(
            plan.rejected_experiments[0].reasons,
            vec![DecisionRejectionReason::NotAuthorized]
        );
    }

    #[test]
    fn minimax_regret_prefers_robust_shared_policy_over_scenario_weighted_bayes_policy() {
        let scenarios = vec![
            scenario("nominal", 0.9, vec![matrix("probe", 0.9, 0.1)]),
            scenario("shifted", 0.1, vec![matrix("probe", 0.1, 0.9)]),
        ];

        let bayes = plan_next_experiment_decision_aware(
            &evidence(),
            &hypotheses(),
            &[proposal("probe")],
            &scenarios,
            &decisions(),
            DecisionAwareConfig {
                minimum_decision_stability: 0.0,
                ranking_policy: DecisionRankingPolicy::ExpectedEvsi,
                ..DecisionAwareConfig::default()
            },
        )
        .unwrap()
        .selected
        .unwrap();
        assert_eq!(
            bayes
                .shared_policy
                .iter()
                .map(|item| item.decision_id.as_str())
                .collect::<Vec<_>>(),
            vec!["remediate-a", "remediate-b"]
        );

        let minimax_plan = plan_next_experiment_decision_aware(
            &evidence(),
            &hypotheses(),
            &[proposal("probe")],
            &scenarios,
            &decisions(),
            DecisionAwareConfig {
                minimum_decision_stability: 0.0,
                ranking_policy: DecisionRankingPolicy::MinimaxRegret,
                ..DecisionAwareConfig::default()
            },
        )
        .unwrap();
        let minimax = minimax_plan.selected.unwrap();
        assert_eq!(
            minimax
                .shared_policy
                .iter()
                .map(|item| item.decision_id.as_str())
                .collect::<Vec<_>>(),
            vec!["remediate-a", "remediate-a"],
            "exact minimax regret uses deterministic lexical tie-breaking"
        );
        assert!(
            minimax.worst_case_shared_policy_regret < bayes.worst_case_shared_policy_regret,
            "minimax policy must reduce maximum scenario regret"
        );
        assert!((minimax.expected_evsi - bayes.expected_evsi).abs() < 1e-12);
        assert!((minimax.worst_case_shared_policy_regret - 0.4).abs() < 1e-12);
        assert!((minimax.decision_stability - 0.5).abs() < 1e-12);
    }

    #[test]
    fn minimax_policy_search_abstains_when_exact_search_exceeds_declared_cap() {
        let plan = plan_next_experiment_decision_aware(
            &evidence(),
            &hypotheses(),
            &[proposal("probe")],
            &[scenario("nominal", 1.0, vec![matrix("probe", 0.9, 0.1)])],
            &decisions(),
            DecisionAwareConfig {
                minimum_decision_stability: 0.0,
                ranking_policy: DecisionRankingPolicy::MinimaxRegret,
                max_policy_evaluations: 3,
                ..DecisionAwareConfig::default()
            },
        )
        .unwrap();

        assert!(plan.selected.is_none());
        assert_eq!(
            plan.abstention,
            Some(DecisionAbstentionReason::PolicySearchBudgetExceeded)
        );
        assert_eq!(
            plan.rejected_experiments[0].reasons,
            vec![DecisionRejectionReason::PolicySearchBudgetExceeded]
        );
    }

    #[test]
    fn rejects_when_shared_policy_regret_exceeds_limit() {
        let scenarios = vec![
            scenario("world-a", 0.5, vec![matrix("probe", 0.9, 0.1)]),
            scenario("world-b", 0.5, vec![matrix("probe", 0.1, 0.9)]),
        ];
        let plan = plan_next_experiment_decision_aware(
            &evidence(),
            &hypotheses(),
            &[proposal("probe")],
            &scenarios,
            &decisions(),
            DecisionAwareConfig {
                minimum_decision_stability: 0.0,
                maximum_worst_case_regret: Some(0.01),
                ..DecisionAwareConfig::default()
            },
        )
        .unwrap();
        assert!(plan.selected.is_none());
        assert_eq!(
            plan.abstention,
            Some(DecisionAbstentionReason::WorstCaseRegretExceeded)
        );
    }
}
