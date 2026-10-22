// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Expected information gain over explicit competing generative-model hypotheses.
//!
//! This module is an isolated mathematical primitive. It is not wired into the
//! production FEP action selector yet. A caller supplies each model's predictive
//! outcome distribution for every candidate action; the result is mutual information
//! I(Model; Outcome | Action) in nats, plus per-outcome posterior model probabilities.
//!
//! A nonzero action-novelty bonus is not a substitute for this quantity, and this
//! quantity alone is not a complete expected-free-energy or control policy.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Probability normalization tolerance for accepted input distributions.
pub const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// A candidate generative model and its predictive outcome distributions.
///
/// outcome_probabilities_by_action[action][outcome] is the probability of that
/// outcome according to this model after taking the indexed action. Rows must be
/// valid probability distributions. The hypotheses' priors must sum to one.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelHypothesis {
    pub id: String,
    pub prior_probability: f64,
    pub outcome_probabilities_by_action: Vec<Vec<f64>>,
}

impl ModelHypothesis {
    pub fn new(
        id: impl Into<String>,
        prior_probability: f64,
        outcome_probabilities_by_action: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            id: id.into(),
            prior_probability,
            outcome_probabilities_by_action,
        }
    }
}

/// Action-conditional information gain about model identity.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionInformationGain {
    /// Mutual information I(Model; Outcome | Action), in nats.
    pub action_index: usize,
    pub information_gain_nats: f64,
    /// Entropy of the model prior, in nats.
    pub prior_entropy_nats: f64,
    /// Expected posterior model entropy after observing the outcome, in nats.
    pub expected_posterior_entropy_nats: f64,
    /// Predictive outcome distribution after marginalizing over model hypotheses.
    pub predicted_outcome_probabilities: Vec<f64>,
    /// Posterior model probabilities for each outcome. None marks an impossible
    /// outcome under the complete model mixture.
    pub posterior_model_probabilities_by_outcome: Vec<Option<Vec<f64>>>,
}

/// Input-contract or numerical-invariant failures.
#[derive(Debug, Clone, PartialEq)]
pub enum InformationGainError {
    EmptyHypotheses,
    EmptyHypothesisId { index: usize },
    DuplicateHypothesisId { id: String },
    InvalidPrior { hypothesis_id: String, probability: f64 },
    PriorMassNotNormalized { sum: f64 },
    EmptyActionSet,
    ActionCountMismatch { hypothesis_id: String, expected: usize, actual: usize },
    EmptyOutcomeSet { hypothesis_id: String, action_index: usize },
    OutcomeCountMismatch {
        hypothesis_id: String,
        action_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidOutcomeProbability {
        hypothesis_id: String,
        action_index: usize,
        outcome_index: usize,
        probability: f64,
    },
    OutcomeMassNotNormalized { hypothesis_id: String, action_index: usize, sum: f64 },
    InvalidActionIndex { action_index: usize, action_count: usize },
    NumericalInvariantViolation { action_index: usize, information_gain_nats: f64 },
}

impl fmt::Display for InformationGainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHypotheses => write!(f, "model hypothesis set is empty"),
            Self::EmptyHypothesisId { index } => {
                write!(f, "model hypothesis at index {index} has an empty id")
            }
            Self::DuplicateHypothesisId { id } => {
                write!(f, "duplicate model hypothesis id: {id}")
            }
            Self::InvalidPrior { hypothesis_id, probability } => write!(
                f,
                "invalid prior probability {probability} for hypothesis {hypothesis_id}"
            ),
            Self::PriorMassNotNormalized { sum } => {
                write!(f, "model prior probabilities sum to {sum}, not one")
            }
            Self::EmptyActionSet => write!(f, "candidate action set is empty"),
            Self::ActionCountMismatch { hypothesis_id, expected, actual } => write!(
                f,
                "hypothesis {hypothesis_id} has {actual} actions, expected {expected}"
            ),
            Self::EmptyOutcomeSet { hypothesis_id, action_index } => write!(
                f,
                "hypothesis {hypothesis_id}, action {action_index} has no outcomes"
            ),
            Self::OutcomeCountMismatch { hypothesis_id, action_index, expected, actual } => write!(
                f,
                "hypothesis {hypothesis_id}, action {action_index} has {actual} outcomes, expected {expected}"
            ),
            Self::InvalidOutcomeProbability {
                hypothesis_id,
                action_index,
                outcome_index,
                probability,
            } => write!(
                f,
                "invalid probability {probability} for hypothesis {hypothesis_id}, action {action_index}, outcome {outcome_index}"
            ),
            Self::OutcomeMassNotNormalized { hypothesis_id, action_index, sum } => write!(
                f,
                "outcome probabilities for hypothesis {hypothesis_id}, action {action_index} sum to {sum}, not one"
            ),
            Self::InvalidActionIndex { action_index, action_count } => write!(
                f,
                "action index {action_index} is outside candidate range 0..{action_count}"
            ),
            Self::NumericalInvariantViolation { action_index, information_gain_nats } => write!(
                f,
                "information gain for action {action_index} is materially negative ({information_gain_nats})"
            ),
        }
    }
}

impl Error for InformationGainError {}

/// Compute action-specific expected information gain about model identity.
///
/// The returned value is I(M; O | a) = H(M) - E_o[H(M | o, a)].
/// Impossible outcomes have no posterior and contribute zero expectation.
/// Inputs are validated and rejected if probability mass is materially invalid;
/// they are not silently renormalized.
pub fn compute_action_information_gains(
    hypotheses: &[ModelHypothesis],
) -> Result<Vec<ActionInformationGain>, InformationGainError> {
    let (action_count, outcome_count) = validate_hypotheses(hypotheses)?;
    let prior = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect::<Vec<_>>();
    let prior_entropy_nats = entropy(&prior);
    let mut gains = Vec::with_capacity(action_count);

    for action_index in 0..action_count {
        let mut predicted_outcome_probabilities = vec![0.0; outcome_count];
        for hypothesis in hypotheses {
            for (outcome_index, outcome_probability) in hypothesis
                .outcome_probabilities_by_action[action_index]
                .iter()
                .copied()
                .enumerate()
            {
                predicted_outcome_probabilities[outcome_index] +=
                    hypothesis.prior_probability * outcome_probability;
            }
        }

        let mut posterior_model_probabilities_by_outcome = Vec::with_capacity(outcome_count);
        let mut expected_posterior_entropy_nats = 0.0;

        for (outcome_index, marginal_probability) in predicted_outcome_probabilities
            .iter()
            .copied()
            .enumerate()
        {
            if marginal_probability <= 0.0 {
                posterior_model_probabilities_by_outcome.push(None);
                continue;
            }

            let posterior = hypotheses
                .iter()
                .map(|hypothesis| {
                    hypothesis.prior_probability
                        * hypothesis.outcome_probabilities_by_action[action_index][outcome_index]
                        / marginal_probability
                })
                .collect::<Vec<_>>();
            expected_posterior_entropy_nats +=
                marginal_probability * entropy(&posterior);
            posterior_model_probabilities_by_outcome.push(Some(posterior));
        }

        let raw_information_gain =
            prior_entropy_nats - expected_posterior_entropy_nats;
        let information_gain_nats = if raw_information_gain.abs() <= 1e-12 {
            0.0
        } else if raw_information_gain < -PROBABILITY_TOLERANCE {
            return Err(InformationGainError::NumericalInvariantViolation {
                action_index,
                information_gain_nats: raw_information_gain,
            });
        } else {
            raw_information_gain.max(0.0)
        };

        // Remove only sub-tolerance negative roundoff; do not claim precision
        // beyond f64 supports.
        expected_posterior_entropy_nats =
            (prior_entropy_nats - information_gain_nats).max(0.0);

        gains.push(ActionInformationGain {
            action_index,
            information_gain_nats,
            prior_entropy_nats,
            expected_posterior_entropy_nats,
            predicted_outcome_probabilities,
            posterior_model_probabilities_by_outcome,
        });
    }

    Ok(gains)
}

/// Compute the expected information gain for one candidate action.
///
/// The full model set is validated before the requested row is returned.
pub fn compute_action_information_gain(
    hypotheses: &[ModelHypothesis],
    action_index: usize,
) -> Result<ActionInformationGain, InformationGainError> {
    let gains = compute_action_information_gains(hypotheses)?;
    let action_count = gains.len();
    gains.get(action_index).cloned().ok_or(InformationGainError::InvalidActionIndex {
        action_index,
        action_count,
    })
}

/// Return the index of the action that most reduces uncertainty about model identity.
///
/// This selects only on epistemic value; it ignores pragmatic reward, action cost,
/// safety constraints and execution authority. Those belong in the caller's full
/// decision policy. Exact ties are resolved in favor of the lowest action index.
pub fn most_informative_action(
    hypotheses: &[ModelHypothesis],
) -> Result<usize, InformationGainError> {
    let gains = compute_action_information_gains(hypotheses)?;
    let mut best = gains.first().ok_or(InformationGainError::EmptyActionSet)?;

    for candidate in gains.iter().skip(1) {
        if candidate.information_gain_nats > best.information_gain_nats
            || (candidate.information_gain_nats == best.information_gain_nats
                && candidate.action_index < best.action_index)
        {
            best = candidate;
        }
    }
    Ok(best.action_index)
}

/// Shannon entropy in nats. Zero-probability entries contribute zero.
fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn validate_hypotheses(
    hypotheses: &[ModelHypothesis],
) -> Result<(usize, usize), InformationGainError> {
    if hypotheses.is_empty() {
        return Err(InformationGainError::EmptyHypotheses);
    }

    let mut ids = BTreeSet::new();
    for (index, hypothesis) in hypotheses.iter().enumerate() {
        if hypothesis.id.trim().is_empty() {
            return Err(InformationGainError::EmptyHypothesisId { index });
        }
        if !ids.insert(hypothesis.id.clone()) {
            return Err(InformationGainError::DuplicateHypothesisId {
                id: hypothesis.id.clone(),
            });
        }
        let prior = hypothesis.prior_probability;
        if !prior.is_finite() || !(0.0..=1.0).contains(&prior) {
            return Err(InformationGainError::InvalidPrior {
                hypothesis_id: hypothesis.id.clone(),
                probability: prior,
            });
        }
    }

    let prior_sum = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .sum::<f64>();
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(InformationGainError::PriorMassNotNormalized { sum: prior_sum });
    }

    let action_count = hypotheses[0].outcome_probabilities_by_action.len();
    if action_count == 0 {
        return Err(InformationGainError::EmptyActionSet);
    }
    let outcome_count = hypotheses[0].outcome_probabilities_by_action[0].len();
    if outcome_count == 0 {
        return Err(InformationGainError::EmptyOutcomeSet {
            hypothesis_id: hypotheses[0].id.clone(),
            action_index: 0,
        });
    }

    for hypothesis in hypotheses {
        let actual_action_count = hypothesis.outcome_probabilities_by_action.len();
        if actual_action_count != action_count {
            return Err(InformationGainError::ActionCountMismatch {
                hypothesis_id: hypothesis.id.clone(),
                expected: action_count,
                actual: actual_action_count,
            });
        }

        for (action_index, probabilities) in hypothesis
            .outcome_probabilities_by_action
            .iter()
            .enumerate()
        {
            if probabilities.is_empty() {
                return Err(InformationGainError::EmptyOutcomeSet {
                    hypothesis_id: hypothesis.id.clone(),
                    action_index,
                });
            }
            if probabilities.len() != outcome_count {
                return Err(InformationGainError::OutcomeCountMismatch {
                    hypothesis_id: hypothesis.id.clone(),
                    action_index,
                    expected: outcome_count,
                    actual: probabilities.len(),
                });
            }

            for (outcome_index, probability) in probabilities.iter().copied().enumerate() {
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err(InformationGainError::InvalidOutcomeProbability {
                        hypothesis_id: hypothesis.id.clone(),
                        action_index,
                        outcome_index,
                        probability,
                    });
                }
            }

            let sum = probabilities.iter().sum::<f64>();
            if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(InformationGainError::OutcomeMassNotNormalized {
                    hypothesis_id: hypothesis.id.clone(),
                    action_index,
                    sum,
                });
            }
        }
    }

    Ok((action_count, outcome_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discriminating_hypotheses() -> Vec<ModelHypothesis> {
        vec![
            ModelHypothesis::new(
                "rule-left",
                0.5,
                vec![vec![0.5, 0.5], vec![0.99, 0.01]],
            ),
            ModelHypothesis::new(
                "rule-right",
                0.5,
                vec![vec![0.5, 0.5], vec![0.01, 0.99]],
            ),
        ]
    }

    #[test]
    fn identical_predictions_have_zero_information_gain() {
        let hypotheses = vec![
            ModelHypothesis::new("model-a", 0.25, vec![vec![0.8, 0.2]]),
            ModelHypothesis::new("model-b", 0.75, vec![vec![0.8, 0.2]]),
        ];
        let gains = compute_action_information_gains(&hypotheses).unwrap();
        assert_eq!(gains.len(), 1);
        assert_eq!(gains[0].information_gain_nats, 0.0);
        assert!((gains[0].expected_posterior_entropy_nats - gains[0].prior_entropy_nats).abs() < 1e-12);
    }

    #[test]
    fn model_discriminating_action_has_positive_information_gain() {
        let hypotheses = discriminating_hypotheses();
        let gains = compute_action_information_gains(&hypotheses).unwrap();
        assert_eq!(gains.len(), 2);
        assert_eq!(gains[0].information_gain_nats, 0.0);
        assert!(gains[1].information_gain_nats > 0.5);
        assert_eq!(most_informative_action(&hypotheses).unwrap(), 1);
    }

    #[test]
    fn posterior_entropy_identity_holds_for_each_action() {
        for gain in compute_action_information_gains(&discriminating_hypotheses()).unwrap() {
            assert!(
                (gain.prior_entropy_nats
                    - gain.expected_posterior_entropy_nats
                    - gain.information_gain_nats)
                    .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn posteriors_normalize_for_each_possible_outcome() {
        let gains = compute_action_information_gains(&discriminating_hypotheses()).unwrap();
        for posterior in gains[1]
            .posterior_model_probabilities_by_outcome
            .iter()
            .flatten()
        {
            assert!((posterior.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn impossible_outcomes_do_not_generate_nan() {
        let hypotheses = vec![
            ModelHypothesis::new("deterministic-a", 0.5, vec![vec![1.0, 0.0, 0.0]]),
            ModelHypothesis::new("deterministic-b", 0.5, vec![vec![0.0, 1.0, 0.0]]),
        ];
        let gains = compute_action_information_gains(&hypotheses).unwrap();
        assert!((gains[0].information_gain_nats - std::f64::consts::LN_2).abs() < 1e-12);
        assert!(gains[0].posterior_model_probabilities_by_outcome[0].is_some());
        assert!(gains[0].posterior_model_probabilities_by_outcome[1].is_some());
        assert!(gains[0].posterior_model_probabilities_by_outcome[2].is_none());
    }

    #[test]
    fn zero_prior_hypothesis_does_not_contribute_information() {
        let hypotheses = vec![
            ModelHypothesis::new("active", 1.0, vec![vec![0.5, 0.5], vec![0.9, 0.1]]),
            ModelHypothesis::new("zero-prior", 0.0, vec![vec![0.1, 0.9], vec![0.1, 0.9]]),
        ];
        let gains = compute_action_information_gains(&hypotheses).unwrap();
        assert_eq!(gains[0].information_gain_nats, 0.0);
        assert_eq!(gains[1].information_gain_nats, 0.0);
    }

    #[test]
    fn model_order_permutation_preserves_action_information_gain() {
        let original = discriminating_hypotheses();
        let mut reversed = original.clone();
        reversed.reverse();
        let a = compute_action_information_gains(&original).unwrap();
        let b = compute_action_information_gains(&reversed).unwrap();
        for (left, right) in a.iter().zip(b) {
            assert!((left.information_gain_nats - right.information_gain_nats).abs() < 1e-12);
        }
    }

    #[test]
    fn action_count_mismatch_is_rejected() {
        let hypotheses = vec![
            ModelHypothesis::new("two-actions", 0.5, vec![vec![0.5, 0.5], vec![0.5, 0.5]]),
            ModelHypothesis::new("one-action", 0.5, vec![vec![0.5, 0.5]]),
        ];
        assert!(matches!(
            compute_action_information_gains(&hypotheses),
            Err(InformationGainError::ActionCountMismatch { .. })
        ));
    }

    #[test]
    fn unnormalized_outcome_distribution_is_rejected_without_silent_repair() {
        let hypotheses = vec![
            ModelHypothesis::new("model-a", 0.5, vec![vec![0.4, 0.4]]),
            ModelHypothesis::new("model-b", 0.5, vec![vec![0.5, 0.5]]),
        ];
        assert!(matches!(
            compute_action_information_gains(&hypotheses),
            Err(InformationGainError::OutcomeMassNotNormalized { .. })
        ));
    }

    #[test]
    fn unnormalized_prior_is_rejected() {
        let hypotheses = vec![
            ModelHypothesis::new("model-a", 0.3, vec![vec![0.5, 0.5]]),
            ModelHypothesis::new("model-b", 0.3, vec![vec![0.5, 0.5]]),
        ];
        assert!(matches!(
            compute_action_information_gains(&hypotheses),
            Err(InformationGainError::PriorMassNotNormalized { .. })
        ));
    }

    #[test]
    fn duplicate_hypothesis_identity_is_rejected() {
        let hypotheses = vec![
            ModelHypothesis::new("same", 0.5, vec![vec![0.5, 0.5]]),
            ModelHypothesis::new("same", 0.5, vec![vec![0.5, 0.5]]),
        ];
        assert!(matches!(
            compute_action_information_gains(&hypotheses),
            Err(InformationGainError::DuplicateHypothesisId { .. })
        ));
    }

    #[test]
    fn empty_hypothesis_set_is_rejected() {
        assert!(matches!(
            compute_action_information_gains(&[]),
            Err(InformationGainError::EmptyHypotheses)
        ));
    }

    #[test]
    fn invalid_action_index_fails_closed() {
        assert!(matches!(
            compute_action_information_gain(&discriminating_hypotheses(), 9),
            Err(InformationGainError::InvalidActionIndex {
                action_index: 9,
                action_count: 2
            })
        ));
    }

    #[test]
    fn equal_information_gain_ties_choose_lowest_action_index() {
        let hypotheses = vec![
            ModelHypothesis::new("a", 0.5, vec![vec![0.9, 0.1], vec![0.9, 0.1]]),
            ModelHypothesis::new("b", 0.5, vec![vec![0.1, 0.9], vec![0.1, 0.9]]),
        ];
        assert_eq!(most_informative_action(&hypotheses).unwrap(), 0);
    }
}
