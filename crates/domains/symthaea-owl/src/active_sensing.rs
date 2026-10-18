// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Budget-constrained active sensing by expected information gain.
//!
//! Evaluate a finite, discrete hypothesis set and candidate actions. Each
//! action supplies P(observation | hypothesis). The planner computes expected
//! posterior-entropy reduction, then chooses the most informative action that
//! is marked safe, fits energy/latency budgets, and exceeds a minimum threshold.
//!
//! This module does not command motors, validate safety envelopes, estimate
//! likelihoods from sensor signals, or guarantee calibrated probability models.
//! An action marked safe is only as trustworthy as the upstream safety check.

use core::fmt;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;

/// Candidate movement / sensing action with a discrete observation model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensingAction<'a> {
    /// Stable identifier for deterministic tie-breaking.
    pub action_id: &'a str,
    /// For each hypothesis, P(observation outcome | hypothesis, action).
    /// Every row must have equal nonzero length and sum to approximately one.
    pub likelihoods_by_hypothesis: &'a [&'a [f64]],
    /// Predicted energy cost, including actuation and sensing overhead.
    pub energy_joules: f64,
    /// Predicted duration / observation latency in seconds.
    pub latency_seconds: f64,
    /// Must come from an independently validated safety-envelope check.
    pub safe: bool,
}

/// Hard limits and minimum useful information threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveSensingBudget {
    pub max_energy_joules: f64,
    pub max_latency_seconds: f64,
    /// An action must strictly exceed this expected information gain to be selected.
    pub minimum_information_gain_nats: f64,
}

/// Information-gain evaluation for one candidate action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionInformationScore<'a> {
    pub action_id: &'a str,
    pub expected_information_gain_nats: f64,
    pub energy_joules: f64,
    pub latency_seconds: f64,
    pub safe: bool,
    pub within_budgets: bool,
    /// Safe, within budgets, and strictly above the minimum-gain threshold.
    pub eligible: bool,
}

/// Evaluations of all actions and the selected original action index, if any.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveSensingPlan<'a> {
    pub prior_entropy_nats: f64,
    pub action_scores: Vec<ActionInformationScore<'a>>,
    pub selected_action_index: Option<usize>,
}

/// Input and numerical validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveSensingError {
    EmptyHypothesisSet,
    EmptyActionSet,
    EmptyActionId,
    DuplicateActionId,
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

impl fmt::Display for ActiveSensingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHypothesisSet => write!(f, "hypothesis prior must not be empty"),
            Self::EmptyActionSet => write!(f, "at least one candidate action is required"),
            Self::EmptyActionId => write!(f, "action identity must not be blank"),
            Self::DuplicateActionId => write!(f, "action identities must be unique"),
            Self::InvalidPriorProbability => {
                write!(f, "prior probabilities must be finite and within [0, 1]")
            }
            Self::PriorNotNormalized => {
                write!(f, "prior probabilities must sum to one within tolerance")
            }
            Self::InvalidBudget => {
                write!(f, "budgets must be finite and nonnegative")
            }
            Self::InvalidActionCost => {
                write!(f, "action energy and latency must be finite and nonnegative")
            }
            Self::HypothesisCountMismatch => {
                write!(f, "each action must define one observation row per hypothesis")
            }
            Self::EmptyObservationSet => {
                write!(f, "each action must define at least one possible observation")
            }
            Self::ObservationCountMismatch => {
                write!(f, "all hypothesis rows for an action must have the same length")
            }
            Self::InvalidObservationProbability => {
                write!(f, "observation probabilities must be finite and within [0, 1]")
            }
            Self::ObservationRowNotNormalized => {
                write!(f, "each hypothesis observation row must sum to one within tolerance")
            }
            Self::NonFiniteInformationGain => {
                write!(f, "expected information gain became non-finite")
            }
        }
    }
}

impl std::error::Error for ActiveSensingError {}

fn entropy_nats(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&probability| probability > 0.0)
        .map(|&probability| -probability * probability.ln())
        .sum()
}

fn validate_action(
    action: &SensingAction<'_>,
    hypothesis_count: usize,
) -> Result<usize, ActiveSensingError> {
    if action.action_id.trim().is_empty() {
        return Err(ActiveSensingError::EmptyActionId);
    }
    if !action.energy_joules.is_finite()
        || !action.latency_seconds.is_finite()
        || action.energy_joules < 0.0
        || action.latency_seconds < 0.0
    {
        return Err(ActiveSensingError::InvalidActionCost);
    }
    if action.likelihoods_by_hypothesis.len() != hypothesis_count {
        return Err(ActiveSensingError::HypothesisCountMismatch);
    }

    let observation_count = action
        .likelihoods_by_hypothesis
        .first()
        .map(|row| row.len())
        .ok_or(ActiveSensingError::EmptyObservationSet)?;
    if observation_count == 0 {
        return Err(ActiveSensingError::EmptyObservationSet);
    }

    for row in action.likelihoods_by_hypothesis {
        if row.len() != observation_count {
            return Err(ActiveSensingError::ObservationCountMismatch);
        }
        if row
            .iter()
            .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
        {
            return Err(ActiveSensingError::InvalidObservationProbability);
        }
        let row_sum: f64 = row.iter().sum();
        if !row_sum.is_finite() || (row_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(ActiveSensingError::ObservationRowNotNormalized);
        }
    }
    Ok(observation_count)
}

fn expected_information_gain(
    prior: &[f64],
    action: &SensingAction<'_>,
    observation_count: usize,
    prior_entropy: f64,
) -> Result<f64, ActiveSensingError> {
    let mut expected_posterior_entropy = 0.0;

    for observation_index in 0..observation_count {
        let predictive_probability: f64 = prior
            .iter()
            .enumerate()
            .map(|(hypothesis_index, prior_probability)| {
                prior_probability
                    * action.likelihoods_by_hypothesis[hypothesis_index][observation_index]
            })
            .sum();

        if !predictive_probability.is_finite() {
            return Err(ActiveSensingError::NonFiniteInformationGain);
        }
        if predictive_probability <= 0.0 {
            continue;
        }

        let posterior: Vec<f64> = prior
            .iter()
            .enumerate()
            .map(|(hypothesis_index, prior_probability)| {
                prior_probability
                    * action.likelihoods_by_hypothesis[hypothesis_index][observation_index]
                    / predictive_probability
            })
            .collect();
        expected_posterior_entropy += predictive_probability * entropy_nats(&posterior);
    }

    if !expected_posterior_entropy.is_finite() {
        return Err(ActiveSensingError::NonFiniteInformationGain);
    }
    let gain = prior_entropy - expected_posterior_entropy;
    if !gain.is_finite() {
        return Err(ActiveSensingError::NonFiniteInformationGain);
    }

    // Analytically this mutual information is nonnegative. Tiny row-normalizer
    // errors can produce a minuscule negative value numerically.
    Ok(gain.max(0.0))
}

/// Rank discrete actions by expected entropy reduction under hard budgets.
///
/// Prior probabilities must sum to one. Each action defines P(observation |
/// hypothesis, action), one row per hypothesis, and every row must sum to one.
/// Expected information gain is H(H) - E[H(H | O, action)] in nats.
///
/// Among safe actions within the energy and latency limits, choose the one with
/// the largest gain, but only when that gain strictly exceeds the configured
/// minimum. Exact ties prefer lower energy, then lower latency, then the
/// lexicographically smaller action ID. This is input-order independent.
///
/// Returns no selected action when all actions are unsafe, over budget, or not
/// sufficiently informative. The integrating robot must define a separate safe
/// hold/stop response for that case.
pub fn plan_active_sensing<'a>(
    prior_probabilities: &[f64],
    actions: &[SensingAction<'a>],
    budget: ActiveSensingBudget,
) -> Result<ActiveSensingPlan<'a>, ActiveSensingError> {
    if prior_probabilities.is_empty() {
        return Err(ActiveSensingError::EmptyHypothesisSet);
    }
    if actions.is_empty() {
        return Err(ActiveSensingError::EmptyActionSet);
    }
    if prior_probabilities
        .iter()
        .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
    {
        return Err(ActiveSensingError::InvalidPriorProbability);
    }
    let prior_sum: f64 = prior_probabilities.iter().sum();
    if !prior_sum.is_finite() || (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(ActiveSensingError::PriorNotNormalized);
    }
    if !budget.max_energy_joules.is_finite()
        || !budget.max_latency_seconds.is_finite()
        || !budget.minimum_information_gain_nats.is_finite()
        || budget.max_energy_joules < 0.0
        || budget.max_latency_seconds < 0.0
        || budget.minimum_information_gain_nats < 0.0
    {
        return Err(ActiveSensingError::InvalidBudget);
    }

    for (index, action) in actions.iter().enumerate() {
        if actions[..index]
            .iter()
            .any(|previous| previous.action_id.trim() == action.action_id.trim())
        {
            return Err(ActiveSensingError::DuplicateActionId);
        }
    }

    let prior_entropy = entropy_nats(prior_probabilities);
    if !prior_entropy.is_finite() {
        return Err(ActiveSensingError::NonFiniteInformationGain);
    }

    let mut action_scores = Vec::with_capacity(actions.len());
    for action in actions {
        let observation_count = validate_action(action, prior_probabilities.len())?;
        let gain = expected_information_gain(
            prior_probabilities,
            action,
            observation_count,
            prior_entropy,
        )?;
        let within_budgets = action.energy_joules <= budget.max_energy_joules
            && action.latency_seconds <= budget.max_latency_seconds;
        let eligible = action.safe
            && within_budgets
            && gain > budget.minimum_information_gain_nats;

        action_scores.push(ActionInformationScore {
            action_id: action.action_id,
            expected_information_gain_nats: gain,
            energy_joules: action.energy_joules,
            latency_seconds: action.latency_seconds,
            safe: action.safe,
            within_budgets,
            eligible,
        });
    }

    let mut selected_action_index: Option<usize> = None;
    for (index, score) in action_scores.iter().enumerate() {
        if !score.eligible {
            continue;
        }
        let should_select = match selected_action_index {
            None => true,
            Some(current_index) => {
                let current = &action_scores[current_index];
                score
                    .expected_information_gain_nats
                    .total_cmp(&current.expected_information_gain_nats)
                    .is_gt()
                    || (score.expected_information_gain_nats
                        == current.expected_information_gain_nats
                        && (score.energy_joules < current.energy_joules
                            || (score.energy_joules == current.energy_joules
                                && (score.latency_seconds < current.latency_seconds
                                    || (score.latency_seconds == current.latency_seconds
                                        && score.action_id < current.action_id)))))
            }
        };
        if should_select {
            selected_action_index = Some(index);
        }
    }

    Ok(ActiveSensingPlan {
        prior_entropy_nats: prior_entropy,
        action_scores,
        selected_action_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF_PRIOR: [f64; 2] = [0.5, 0.5];
    const UNINFORMATIVE: [&[f64]; 2] = [&[0.5, 0.5], &[0.5, 0.5]];
    const PERFECT: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];

    fn action<'a>(
        action_id: &'a str,
        likelihoods: &'a [&'a [f64]],
        energy_joules: f64,
        latency_seconds: f64,
        safe: bool,
    ) -> SensingAction<'a> {
        SensingAction {
            action_id,
            likelihoods_by_hypothesis: likelihoods,
            energy_joules,
            latency_seconds,
            safe,
        }
    }

    fn budget(max_energy_joules: f64, max_latency_seconds: f64) -> ActiveSensingBudget {
        ActiveSensingBudget {
            max_energy_joules,
            max_latency_seconds,
            minimum_information_gain_nats: 0.0,
        }
    }

    #[test]
    fn perfect_observation_yields_ln_two_information_gain() {
        let actions = [action("turn_left", &PERFECT, 1.0, 0.1, true)];
        let plan = plan_active_sensing(&HALF_PRIOR, &actions, budget(2.0, 1.0)).unwrap();
        let gain = plan.action_scores[0].expected_information_gain_nats;
        assert!((gain - core::f64::consts::LN_2).abs() < 1e-12);
        assert_eq!(plan.selected_action_index, Some(0));
    }

    #[test]
    fn uninformative_action_has_zero_gain_and_is_not_selected() {
        let actions = [action("hold", &UNINFORMATIVE, 0.0, 0.1, true)];
        let plan = plan_active_sensing(&HALF_PRIOR, &actions, budget(2.0, 1.0)).unwrap();
        assert!(plan.action_scores[0].expected_information_gain_nats.abs() < 1e-12);
        assert_eq!(plan.selected_action_index, None);
    }

    #[test]
    fn energy_budget_excludes_more_informative_but_expensive_action() {
        let actions = [
            action("cheap_partial", &UNINFORMATIVE, 0.1, 0.1, true),
            action("expensive_perfect", &PERFECT, 10.0, 0.5, true),
        ];
        let plan = plan_active_sensing(&HALF_PRIOR, &actions, budget(1.0, 1.0)).unwrap();
        assert_eq!(plan.selected_action_index, None);
        assert!(!plan.action_scores[1].within_budgets);
    }

    #[test]
    fn unsafe_action_never_selected_even_if_perfect() {
        let actions = [action("unsafe_turn", &PERFECT, 0.1, 0.1, false)];
        let plan = plan_active_sensing(&HALF_PRIOR, &actions, budget(1.0, 1.0)).unwrap();
        assert_eq!(plan.selected_action_index, None);
        assert!(!plan.action_scores[0].eligible);
    }

    #[test]
    fn selection_is_independent_of_action_order_for_exact_ties() {
        let rows_a = [&[1.0, 0.0][..], &[0.0, 1.0][..]];
        let rows_b = [&[1.0, 0.0][..], &[0.0, 1.0][..]];
        let a = action("turn_a", &rows_a, 0.5, 0.2, true);
        let b = action("turn_b", &rows_b, 0.5, 0.1, true);
        let first = plan_active_sensing(&HALF_PRIOR, &[a, b], budget(1.0, 1.0)).unwrap();
        let second = plan_active_sensing(&HALF_PRIOR, &[b, a], budget(1.0, 1.0)).unwrap();
        assert_eq!(
            first.action_scores[first.selected_action_index.unwrap()].action_id,
            "turn_b"
        );
        assert_eq!(
            second.action_scores[second.selected_action_index.unwrap()].action_id,
            "turn_b"
        );
    }

    #[test]
    fn non_normalized_prior_is_rejected() {
        let prior = [0.2, 0.2];
        let actions = [action("perfect", &PERFECT, 1.0, 0.1, true)];
        assert_eq!(
            plan_active_sensing(&prior, &actions, budget(2.0, 1.0)),
            Err(ActiveSensingError::PriorNotNormalized)
        );
    }

    #[test]
    fn invalid_observation_row_sum_is_rejected() {
        let invalid = [&[0.3, 0.3][..], &[0.5, 0.5][..]];
        let actions = [action("bad_model", &invalid, 0.5, 0.1, true)];
        assert_eq!(
            plan_active_sensing(&HALF_PRIOR, &actions, budget(1.0, 1.0)),
            Err(ActiveSensingError::ObservationRowNotNormalized)
        );
    }

    #[test]
    fn mismatched_hypothesis_dimension_is_rejected() {
        let one_row = [&[1.0, 0.0][..]];
        let actions = [action("bad_dims", &one_row, 0.5, 0.1, true)];
        assert_eq!(
            plan_active_sensing(&HALF_PRIOR, &actions, budget(1.0, 1.0)),
            Err(ActiveSensingError::HypothesisCountMismatch)
        );
    }

    #[test]
    fn duplicate_action_ids_are_rejected() {
        let actions = [
            action("same", &PERFECT, 0.5, 0.1, true),
            action("same", &PERFECT, 0.5, 0.1, true),
        ];
        assert_eq!(
            plan_active_sensing(&HALF_PRIOR, &actions, budget(1.0, 1.0)),
            Err(ActiveSensingError::DuplicateActionId)
        );
    }

    #[test]
    fn minimum_gain_threshold_can_decline_action() {
        let actions = [action("no_gain", &UNINFORMATIVE, 0.0, 0.1, true)];
        let mut constrained = budget(1.0, 1.0);
        constrained.minimum_information_gain_nats = 0.01;
        let plan = plan_active_sensing(&HALF_PRIOR, &actions, constrained).unwrap();
        assert_eq!(plan.selected_action_index, None);
    }

    #[test]
    fn non_finite_action_cost_is_rejected() {
        let actions = [action("bad_cost", &PERFECT, f64::NAN, 0.1, true)];
        assert_eq!(
            plan_active_sensing(&HALF_PRIOR, &actions, budget(1.0, 1.0)),
            Err(ActiveSensingError::InvalidActionCost)
        );
    }

    #[test]
    fn all_unsafe_or_over_budget_actions_yield_no_selection() {
        let actions = [
            action("unsafe", &PERFECT, 0.1, 0.1, false),
            action("too_slow", &PERFECT, 0.1, 5.0, true),
        ];
        let plan = plan_active_sensing(&HALF_PRIOR, &actions, budget(1.0, 1.0)).unwrap();
        assert_eq!(plan.selected_action_index, None);
    }
}
