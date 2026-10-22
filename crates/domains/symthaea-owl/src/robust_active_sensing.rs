// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative active sensing under an explicit finite set of model scenarios.
//!
//! Score each candidate action under every declared observation model (for
//! example, nominal room, added reverberation, channel drift, or a failed
//! microphone) and rank by minimum expected information gain. This is only
//! robust to the supplied scenario set, not to arbitrary unknown conditions.
//!
//! Scenario likelihoods require held-out calibration. The safety flag must
//! originate from an independent verified checker; this module is not a safety
//! proof, motor controller, or learned observation model.

use core::fmt;
use super::active_sensing::ActiveSensingBudget;

const TOLERANCE: f64 = 1.0e-9;

/// One environment / sensor-condition model for a candidate action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservationScenario<'a> {
    pub scenario_id: &'a str,
    /// One P(observation | hypothesis) row per source hypothesis.
    pub likelihoods_by_hypothesis: &'a [&'a [f64]],
}

/// A candidate action must provide the same scenario-ID set as all alternatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustSensingAction<'a> {
    pub action_id: &'a str,
    pub scenarios: &'a [ObservationScenario<'a>],
    pub energy_joules: f64,
    pub latency_seconds: f64,
    /// Supplied by a separate validated safety-envelope checker.
    pub safe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScenarioInformationGain<'a> {
    pub scenario_id: &'a str,
    pub information_gain_nats: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustActionScore<'a> {
    pub action_id: &'a str,
    pub scenario_scores: Vec<ScenarioInformationGain<'a>>,
    pub worst_case_information_gain_nats: f64,
    /// Arithmetic mean over scenarios; not a probability-weighted expectation.
    pub mean_scenario_information_gain_nats: f64,
    pub energy_joules: f64,
    pub latency_seconds: f64,
    pub safe: bool,
    pub within_budgets: bool,
    pub eligible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustActiveSensingPlan<'a> {
    pub prior_entropy_nats: f64,
    pub action_scores: Vec<RobustActionScore<'a>>,
    pub selected_action_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustActiveSensingError {
    EmptyHypothesisSet,
    EmptyActionSet,
    EmptyScenarioSet,
    EmptyActionId,
    EmptyScenarioId,
    DuplicateActionId,
    DuplicateScenarioId,
    ScenarioSetMismatch,
    InvalidPriorProbability,
    PriorNotNormalized,
    InvalidBudget,
    InvalidActionCost,
    HypothesisCountMismatch,
    EmptyObservationSet,
    ObservationCountMismatch,
    InvalidObservationProbability,
    ObservationRowNotNormalized,
    NonFiniteInformationGain,
}

impl fmt::Display for RobustActiveSensingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHypothesisSet => write!(f, "hypothesis prior must not be empty"),
            Self::EmptyActionSet => write!(f, "at least one action is required"),
            Self::EmptyScenarioSet => write!(f, "every action must include the full scenario set"),
            Self::EmptyActionId => write!(f, "action ID must not be blank"),
            Self::EmptyScenarioId => write!(f, "scenario ID must not be blank"),
            Self::DuplicateActionId => write!(f, "action IDs must be unique"),
            Self::DuplicateScenarioId => write!(f, "scenario IDs must be unique"),
            Self::ScenarioSetMismatch => write!(f, "all actions must use the same scenario ID set"),
            Self::InvalidPriorProbability => write!(f, "prior probabilities must be finite and within [0, 1]"),
            Self::PriorNotNormalized => write!(f, "prior probabilities must sum to one"),
            Self::InvalidBudget => write!(f, "budgets must be finite and nonnegative"),
            Self::InvalidActionCost => write!(f, "action costs must be finite and nonnegative"),
            Self::HypothesisCountMismatch => write!(f, "one likelihood row is required per hypothesis"),
            Self::EmptyObservationSet => write!(f, "observation rows must not be empty"),
            Self::ObservationCountMismatch => write!(f, "rows within a scenario must have equal lengths"),
            Self::InvalidObservationProbability => write!(f, "observation probabilities must be finite in [0, 1]"),
            Self::ObservationRowNotNormalized => write!(f, "each observation row must sum to one"),
            Self::NonFiniteInformationGain => write!(f, "information-gain calculation became non-finite"),
        }
    }
}
impl std::error::Error for RobustActiveSensingError {}

fn entropy(probabilities: &[f64]) -> f64 {
    probabilities.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum()
}

fn validate_scenario(
    scenario: &ObservationScenario<'_>,
    hypotheses: usize,
) -> Result<usize, RobustActiveSensingError> {
    if scenario.scenario_id.trim().is_empty() {
        return Err(RobustActiveSensingError::EmptyScenarioId);
    }
    if scenario.likelihoods_by_hypothesis.len() != hypotheses {
        return Err(RobustActiveSensingError::HypothesisCountMismatch);
    }
    let outcomes = scenario.likelihoods_by_hypothesis.first()
        .map(|row| row.len())
        .ok_or(RobustActiveSensingError::EmptyObservationSet)?;
    if outcomes == 0 { return Err(RobustActiveSensingError::EmptyObservationSet); }
    for row in scenario.likelihoods_by_hypothesis {
        if row.len() != outcomes { return Err(RobustActiveSensingError::ObservationCountMismatch); }
        if row.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
            return Err(RobustActiveSensingError::InvalidObservationProbability);
        }
        let sum: f64 = row.iter().sum();
        if !sum.is_finite() || (sum - 1.0).abs() > TOLERANCE {
            return Err(RobustActiveSensingError::ObservationRowNotNormalized);
        }
    }
    Ok(outcomes)
}

fn information_gain(
    prior: &[f64],
    scenario: &ObservationScenario<'_>,
    outcomes: usize,
    prior_entropy: f64,
) -> Result<f64, RobustActiveSensingError> {
    let mut expected_entropy = 0.0;
    for outcome in 0..outcomes {
        let p_outcome: f64 = prior.iter().enumerate()
            .map(|(h, p)| p * scenario.likelihoods_by_hypothesis[h][outcome])
            .sum();
        if !p_outcome.is_finite() { return Err(RobustActiveSensingError::NonFiniteInformationGain); }
        if p_outcome <= 0.0 { continue; }
        let posterior: Vec<f64> = prior.iter().enumerate()
            .map(|(h, p)| p * scenario.likelihoods_by_hypothesis[h][outcome] / p_outcome)
            .collect();
        expected_entropy += p_outcome * entropy(&posterior);
    }
    let gain = prior_entropy - expected_entropy;
    if !gain.is_finite() { return Err(RobustActiveSensingError::NonFiniteInformationGain); }
    Ok(gain.max(0.0))
}

/// Choose the action with the highest *minimum* expected information gain over
/// the same declared scenario set, subject to the standard hard budgets.
///
/// Actions must have unique IDs and identical scenario-ID sets. Each scenario
/// contains a valid P(observation | hypothesis, action, scenario) table. Ties
/// prefer higher worst-case gain, higher mean gain, lower energy, lower latency,
/// then lexical action ID. The function can return None if no action's
/// worst-case gain strictly exceeds the configured minimum threshold.
///
/// No robustness claim extends beyond the caller's finite scenario set.
pub fn plan_robust_active_sensing<'a>(
    prior_probabilities: &[f64],
    actions: &[RobustSensingAction<'a>],
    budget: ActiveSensingBudget,
) -> Result<RobustActiveSensingPlan<'a>, RobustActiveSensingError> {
    if prior_probabilities.is_empty() { return Err(RobustActiveSensingError::EmptyHypothesisSet); }
    if actions.is_empty() { return Err(RobustActiveSensingError::EmptyActionSet); }
    if prior_probabilities.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
        return Err(RobustActiveSensingError::InvalidPriorProbability);
    }
    let prior_sum: f64 = prior_probabilities.iter().sum();
    if !prior_sum.is_finite() || (prior_sum - 1.0).abs() > TOLERANCE {
        return Err(RobustActiveSensingError::PriorNotNormalized);
    }
    if !budget.max_energy_joules.is_finite()
        || !budget.max_latency_seconds.is_finite()
        || !budget.minimum_information_gain_nats.is_finite()
        || budget.max_energy_joules < 0.0
        || budget.max_latency_seconds < 0.0
        || budget.minimum_information_gain_nats < 0.0
    { return Err(RobustActiveSensingError::InvalidBudget); }

    let required = actions[0].scenarios;
    if required.is_empty() { return Err(RobustActiveSensingError::EmptyScenarioSet); }
    for (i, scenario) in required.iter().enumerate() {
        if scenario.scenario_id.trim().is_empty() { return Err(RobustActiveSensingError::EmptyScenarioId); }
        if required[..i].iter().any(|prev| prev.scenario_id.trim() == scenario.scenario_id.trim()) {
            return Err(RobustActiveSensingError::DuplicateScenarioId);
        }
    }
    for (i, action) in actions.iter().enumerate() {
        if action.action_id.trim().is_empty() { return Err(RobustActiveSensingError::EmptyActionId); }
        if actions[..i].iter().any(|prev| prev.action_id.trim() == action.action_id.trim()) {
            return Err(RobustActiveSensingError::DuplicateActionId);
        }
        if action.scenarios.is_empty() { return Err(RobustActiveSensingError::EmptyScenarioSet); }
        if action.scenarios.len() != required.len() || required.iter().any(|req| {
            !action.scenarios.iter().any(|got| got.scenario_id == req.scenario_id)
        }) { return Err(RobustActiveSensingError::ScenarioSetMismatch); }
        if !action.energy_joules.is_finite() || !action.latency_seconds.is_finite()
            || action.energy_joules < 0.0 || action.latency_seconds < 0.0 {
            return Err(RobustActiveSensingError::InvalidActionCost);
        }
    }

    let prior_entropy = entropy(prior_probabilities);
    if !prior_entropy.is_finite() { return Err(RobustActiveSensingError::NonFiniteInformationGain); }
    let mut action_scores = Vec::with_capacity(actions.len());
    for action in actions {
        let mut scenario_scores = Vec::with_capacity(required.len());
        for required_scenario in required {
            let scenario = action.scenarios.iter()
                .find(|candidate| candidate.scenario_id == required_scenario.scenario_id)
                .expect("scenario-set equality validated before scoring");
            let outcomes = validate_scenario(scenario, prior_probabilities.len())?;
            let gain = information_gain(prior_probabilities, scenario, outcomes, prior_entropy)?;
            scenario_scores.push(ScenarioInformationGain { scenario_id: scenario.scenario_id, information_gain_nats: gain });
        }
        let worst = scenario_scores.iter().map(|s| s.information_gain_nats).fold(f64::INFINITY, f64::min);
        let total: f64 = scenario_scores.iter().map(|s| s.information_gain_nats).sum();
        let mean = total / scenario_scores.len() as f64;
        if !worst.is_finite() || !mean.is_finite() { return Err(RobustActiveSensingError::NonFiniteInformationGain); }
        let within_budgets = action.energy_joules <= budget.max_energy_joules
            && action.latency_seconds <= budget.max_latency_seconds;
        action_scores.push(RobustActionScore {
            action_id: action.action_id,
            scenario_scores,
            worst_case_information_gain_nats: worst,
            mean_scenario_information_gain_nats: mean,
            energy_joules: action.energy_joules,
            latency_seconds: action.latency_seconds,
            safe: action.safe,
            within_budgets,
            eligible: action.safe && within_budgets && worst > budget.minimum_information_gain_nats,
        });
    }

    let mut selected_action_index: Option<usize> = None;
    for (index, score) in action_scores.iter().enumerate() {
        if !score.eligible { continue; }
        let better = match selected_action_index {
            None => true,
            Some(current_index) => {
                let current = &action_scores[current_index];
                score.worst_case_information_gain_nats > current.worst_case_information_gain_nats
                    || (score.worst_case_information_gain_nats == current.worst_case_information_gain_nats
                        && (score.mean_scenario_information_gain_nats > current.mean_scenario_information_gain_nats
                            || (score.mean_scenario_information_gain_nats == current.mean_scenario_information_gain_nats
                                && (score.energy_joules < current.energy_joules
                                    || (score.energy_joules == current.energy_joules
                                        && (score.latency_seconds < current.latency_seconds
                                            || (score.latency_seconds == current.latency_seconds
                                                && score.action_id < current.action_id)))))))
            }
        };
        if better { selected_action_index = Some(index); }
    }
    Ok(RobustActiveSensingPlan { prior_entropy_nats: prior_entropy, action_scores, selected_action_index })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRIOR: [f64; 2] = [0.5, 0.5];
    const NONE: [&[f64]; 2] = [&[0.5, 0.5], &[0.5, 0.5]];
    const PERFECT: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
    const MEDIUM: [&[f64]; 2] = [&[0.8, 0.2], &[0.2, 0.8]];
    const WEAK: [&[f64]; 2] = [&[0.7, 0.3], &[0.3, 0.7]];

    fn scenario<'a>(id: &'a str, rows: &'a [&'a [f64]]) -> ObservationScenario<'a> {
        ObservationScenario { scenario_id: id, likelihoods_by_hypothesis: rows }
    }
    fn action<'a>(id: &'a str, scenarios: &'a [ObservationScenario<'a>], energy: f64, latency: f64, safe: bool) -> RobustSensingAction<'a> {
        RobustSensingAction { action_id: id, scenarios, energy_joules: energy, latency_seconds: latency, safe }
    }
    fn budget() -> ActiveSensingBudget {
        ActiveSensingBudget { max_energy_joules: 1.0, max_latency_seconds: 1.0, minimum_information_gain_nats: 0.0 }
    }

    #[test]
    fn avoids_nominal_optimal_action_that_fails_under_fault() {
        let fragile = [scenario("nominal", &PERFECT), scenario("fault", &NONE)];
        let robust = [scenario("nominal", &MEDIUM), scenario("fault", &WEAK)];
        let actions = [action("nominal_trap", &fragile, 0.2, 0.1, true), action("robust_turn", &robust, 0.3, 0.1, true)];
        let plan = plan_robust_active_sensing(&PRIOR, &actions, budget()).unwrap();
        assert_eq!(plan.action_scores[0].worst_case_information_gain_nats, 0.0);
        assert!(plan.action_scores[1].worst_case_information_gain_nats > 0.0);
        assert_eq!(plan.action_scores[plan.selected_action_index.unwrap()].action_id, "robust_turn");
    }

    #[test]
    fn perfect_model_yields_ln_two_nats() {
        let models = [scenario("normal", &PERFECT), scenario("reverb", &PERFECT)];
        let actions = [action("look", &models, 0.1, 0.1, true)];
        let plan = plan_robust_active_sensing(&PRIOR, &actions, budget()).unwrap();
        assert!((plan.action_scores[0].worst_case_information_gain_nats - core::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn same_scenario_set_can_be_in_different_orders() {
        let a_models = [scenario("nominal", &MEDIUM), scenario("fault", &WEAK)];
        let b_models = [scenario("fault", &WEAK), scenario("nominal", &MEDIUM)];
        let actions = [action("a", &a_models, 0.2, 0.1, true), action("b", &b_models, 0.2, 0.1, true)];
        let plan = plan_robust_active_sensing(&PRIOR, &actions, budget()).unwrap();
        assert_eq!(plan.action_scores[0].worst_case_information_gain_nats, plan.action_scores[1].worst_case_information_gain_nats);
        assert_eq!(plan.selected_action_index.map(|i| plan.action_scores[i].action_id), Some("a"));
    }

    #[test]
    fn missing_scenario_is_rejected() {
        let complete = [scenario("normal", &PERFECT), scenario("fault", &PERFECT)];
        let missing = [scenario("normal", &MEDIUM)];
        let actions = [action("a", &complete, 0.1, 0.1, true), action("b", &missing, 0.1, 0.1, true)];
        assert_eq!(plan_robust_active_sensing(&PRIOR, &actions, budget()), Err(RobustActiveSensingError::ScenarioSetMismatch));
    }

    #[test]
    fn unsafe_action_is_never_selected() {
        let models = [scenario("normal", &PERFECT), scenario("fault", &PERFECT)];
        let actions = [action("unsafe", &models, 0.1, 0.1, false)];
        assert_eq!(plan_robust_active_sensing(&PRIOR, &actions, budget()).unwrap().selected_action_index, None);
    }

    #[test]
    fn over_budget_action_is_never_selected() {
        let models = [scenario("normal", &PERFECT), scenario("fault", &PERFECT)];
        let actions = [action("costly", &models, 9.0, 0.1, true)];
        assert_eq!(plan_robust_active_sensing(&PRIOR, &actions, budget()).unwrap().selected_action_index, None);
    }

    #[test]
    fn no_action_when_worst_case_gain_is_zero() {
        let models = [scenario("normal", &PERFECT), scenario("fault", &NONE)];
        let actions = [action("fragile", &models, 0.1, 0.1, true)];
        assert_eq!(plan_robust_active_sensing(&PRIOR, &actions, budget()).unwrap().selected_action_index, None);
    }

    #[test]
    fn invalid_likelihood_row_is_rejected() {
        let invalid = [&[0.8, 0.8][..], &[0.5, 0.5][..]];
        let models = [scenario("bad", &invalid)];
        let actions = [action("a", &models, 0.1, 0.1, true)];
        assert_eq!(plan_robust_active_sensing(&PRIOR, &actions, budget()), Err(RobustActiveSensingError::ObservationRowNotNormalized));
    }

    #[test]
    fn duplicate_scenario_ids_are_rejected() {
        let models = [scenario("same", &PERFECT), scenario("same", &PERFECT)];
        let actions = [action("a", &models, 0.1, 0.1, true)];
        assert_eq!(plan_robust_active_sensing(&PRIOR, &actions, budget()), Err(RobustActiveSensingError::DuplicateScenarioId));
    }

    #[test]
    fn action_order_tie_breaks_lexically() {
        let first_models = [scenario("normal", &MEDIUM)];
        let second_models = [scenario("normal", &MEDIUM)];
        let actions = [action("z", &first_models, 0.2, 0.1, true), action("a", &second_models, 0.2, 0.1, true)];
        let plan = plan_robust_active_sensing(&PRIOR, &actions, budget()).unwrap();
        assert_eq!(plan.action_scores[plan.selected_action_index.unwrap()].action_id, "a");
    }
}
