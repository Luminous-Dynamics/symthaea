// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Hierarchical expected information gain over model, parameter, and latent-state
//! uncertainty.
//!
//! This is a discrete, model-relative primitive. Callers provide a normalized
//! prior over joint hypotheses and action-conditioned outcome likelihoods. The
//! primitive does not infer that these hypotheses are true, learn likelihoods,
//! incorporate preferences/cost/safety, or authorize actions.
//!
//! For a joint hypothesis H=(M, Θ, S) and candidate action A, it computes the
//! mutual-information chain rule:
//!
//! I(M, Θ, S; O | A)
//!   = I(M; O | A)
//!   + I(Θ; O | M, A)
//!   + I(S; O | M, Θ, A)
//!
//! All information values are in nats. The conditional terms are calculated
//! from the same posterior over joint hypotheses, not from unrelated entropy
//! heuristics. A chain-rule residual is returned for every action.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const NUMERICAL_TOLERANCE: f64 = 1e-8;

/// One joint generative hypothesis for model structure, parameters, and state.
///
/// `parameter_id` is scoped to `model_id`; `state_id` is scoped to the
/// (model, parameter) pair. Each distinct tuple represents one mutually
/// exclusive joint hypothesis in the supplied prior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalHypothesis {
    pub model_id: String,
    pub parameter_id: String,
    pub state_id: String,
    pub prior_probability: f64,
    /// `outcome_probabilities_by_action[action][outcome]` = P(outcome | M, Θ, S, action).
    pub outcome_probabilities_by_action: Vec<Vec<f64>>,
}

/// Select which epistemic component to maximize in an information-only
/// diagnostic comparison. This is not a complete expected-free-energy policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicLevel {
    ModelStructure,
    ModelParameters,
    LatentState,
    Joint,
}

/// Decomposed expected information for one candidate action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalActionInformationGain {
    pub action_index: usize,
    /// Model-averaged P(outcome | action), in the input outcome order.
    pub outcome_probabilities: Vec<f64>,
    /// I(M; O | action).
    pub model_information_gain_nats: f64,
    /// I(Θ; O | M, action).
    pub parameter_information_gain_nats: f64,
    /// I(S; O | M, Θ, action).
    pub state_information_gain_nats: f64,
    /// I((M, Θ, S); O | action).
    pub joint_information_gain_nats: f64,
    /// Absolute difference between the joint MI and the sum of conditional
    /// chain-rule terms. Must remain close to numerical precision.
    pub chain_rule_error_nats: f64,
}

/// Failure returned when a discrete hierarchical hypothesis set is malformed.
#[derive(Debug, Clone, PartialEq)]
pub enum HierarchicalInformationGainError {
    EmptyHypotheses,
    EmptyIdentifier {
        hypothesis_index: usize,
        field: &'static str,
    },
    DuplicateJointHypothesis {
        model_id: String,
        parameter_id: String,
        state_id: String,
    },
    InvalidPrior {
        hypothesis_index: usize,
        probability: f64,
    },
    PriorMassNotNormalized {
        sum: f64,
    },
    EmptyActionSet,
    EmptyOutcomeSet {
        action_index: usize,
    },
    InconsistentActionCount {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    InconsistentOutcomeCount {
        hypothesis_index: usize,
        action_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidOutcomeProbability {
        hypothesis_index: usize,
        action_index: usize,
        outcome_index: usize,
        probability: f64,
    },
    OutcomeMassNotNormalized {
        hypothesis_index: usize,
        action_index: usize,
        sum: f64,
    },
    NumericalInvariant {
        action_index: usize,
        chain_rule_error_nats: f64,
        minimum_component_nats: f64,
    },
    InvalidActionIndex {
        action_index: usize,
        action_count: usize,
    },
}

impl std::fmt::Display for HierarchicalInformationGainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyHypotheses => write!(f, "at least one joint hypothesis is required"),
            Self::EmptyIdentifier {
                hypothesis_index,
                field,
            } => write!(
                f,
                "hypothesis {hypothesis_index} has an empty {field} identifier"
            ),
            Self::DuplicateJointHypothesis {
                model_id,
                parameter_id,
                state_id,
            } => write!(
                f,
                "duplicate joint hypothesis ({model_id}, {parameter_id}, {state_id})"
            ),
            Self::InvalidPrior {
                hypothesis_index,
                probability,
            } => write!(
                f,
                "hypothesis {hypothesis_index} has invalid prior probability {probability}"
            ),
            Self::PriorMassNotNormalized { sum } => {
                write!(f, "joint hypothesis prior mass sums to {sum}, expected 1")
            }
            Self::EmptyActionSet => write!(f, "at least one candidate action is required"),
            Self::EmptyOutcomeSet { action_index } => {
                write!(f, "action {action_index} has no possible outcomes")
            }
            Self::InconsistentActionCount {
                hypothesis_index,
                expected,
                actual,
            } => write!(
                f,
                "hypothesis {hypothesis_index} has {actual} actions, expected {expected}"
            ),
            Self::InconsistentOutcomeCount {
                hypothesis_index,
                action_index,
                expected,
                actual,
            } => write!(
                f,
                "hypothesis {hypothesis_index}, action {action_index} has {actual} outcomes, expected {expected}"
            ),
            Self::InvalidOutcomeProbability {
                hypothesis_index,
                action_index,
                outcome_index,
                probability,
            } => write!(
                f,
                "hypothesis {hypothesis_index}, action {action_index}, outcome {outcome_index} has invalid probability {probability}"
            ),
            Self::OutcomeMassNotNormalized {
                hypothesis_index,
                action_index,
                sum,
            } => write!(
                f,
                "hypothesis {hypothesis_index}, action {action_index} outcome mass sums to {sum}, expected 1"
            ),
            Self::NumericalInvariant {
                action_index,
                chain_rule_error_nats,
                minimum_component_nats,
            } => write!(
                f,
                "action {action_index} violated the mutual-information chain rule: residual {chain_rule_error_nats}, minimum component {minimum_component_nats}"
            ),
            Self::InvalidActionIndex {
                action_index,
                action_count,
            } => write!(
                f,
                "action index {action_index} is outside 0..{action_count}"
            ),
        }
    }
}

impl std::error::Error for HierarchicalInformationGainError {}

/// Compute decomposed information gain for every candidate action.
///
/// Every joint-hypothesis outcome distribution must be normalized. Invalid
/// probability mass is rejected; it is never silently repaired or renormalized.
pub fn compute_hierarchical_information_gains(
    hypotheses: &[HierarchicalHypothesis],
) -> Result<Vec<HierarchicalActionInformationGain>, HierarchicalInformationGainError> {
    let (action_count, outcome_count) = validate_hypotheses(hypotheses)?;
    (0..action_count)
        .map(|action_index| compute_for_action(hypotheses, action_index, outcome_count))
        .collect()
}

/// Compute one candidate action's decomposed information gain.
pub fn compute_hierarchical_information_gain(
    hypotheses: &[HierarchicalHypothesis],
    action_index: usize,
) -> Result<HierarchicalActionInformationGain, HierarchicalInformationGainError> {
    let (action_count, outcome_count) = validate_hypotheses(hypotheses)?;
    if action_index >= action_count {
        return Err(HierarchicalInformationGainError::InvalidActionIndex {
            action_index,
            action_count,
        });
    }
    compute_for_action(hypotheses, action_index, outcome_count)
}

/// Select the action maximizing one declared epistemic component.
///
/// This is deliberately information-only. Pragmatic preference, observation
/// cost, risk, feasibility, and execution authority remain the caller's job.
/// Ties are resolved by the lowest action index.
pub fn most_informative_action_at_level(
    hypotheses: &[HierarchicalHypothesis],
    level: EpistemicLevel,
) -> Result<usize, HierarchicalInformationGainError> {
    let results = compute_hierarchical_information_gains(hypotheses)?;
    let mut best = results
        .first()
        .ok_or(HierarchicalInformationGainError::EmptyActionSet)?;
    for candidate in results.iter().skip(1) {
        if component(candidate, level) > component(best, level) {
            best = candidate;
        }
    }
    Ok(best.action_index)
}

fn component(result: &HierarchicalActionInformationGain, level: EpistemicLevel) -> f64 {
    match level {
        EpistemicLevel::ModelStructure => result.model_information_gain_nats,
        EpistemicLevel::ModelParameters => result.parameter_information_gain_nats,
        EpistemicLevel::LatentState => result.state_information_gain_nats,
        EpistemicLevel::Joint => result.joint_information_gain_nats,
    }
}

fn compute_for_action(
    hypotheses: &[HierarchicalHypothesis],
    action_index: usize,
    outcome_count: usize,
) -> Result<HierarchicalActionInformationGain, HierarchicalInformationGainError> {
    let outcome_probabilities = (0..outcome_count)
        .map(|outcome_index| {
            hypotheses
                .iter()
                .map(|hypothesis| {
                    hypothesis.prior_probability
                        * hypothesis.outcome_probabilities_by_action[action_index][outcome_index]
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();

    let mut expected_model_entropy = 0.0;
    let mut expected_parameter_entropy = 0.0;
    let mut expected_state_entropy = 0.0;
    let mut expected_joint_entropy = 0.0;

    for (outcome_index, outcome_probability) in outcome_probabilities.iter().copied().enumerate() {
        if outcome_probability <= 0.0 {
            continue;
        }
        let posterior = hypotheses
            .iter()
            .map(|hypothesis| {
                hypothesis.prior_probability
                    * hypothesis.outcome_probabilities_by_action[action_index][outcome_index]
                    / outcome_probability
            })
            .collect::<Vec<_>>();

        expected_model_entropy +=
            outcome_probability * conditional_model_entropy(hypotheses, &posterior);
        expected_parameter_entropy +=
            outcome_probability * conditional_parameter_entropy(hypotheses, &posterior);
        expected_state_entropy +=
            outcome_probability * conditional_state_entropy(hypotheses, &posterior);
        expected_joint_entropy += outcome_probability * entropy(&posterior);
    }

    let prior = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect::<Vec<_>>();
    let prior_model_entropy = conditional_model_entropy(hypotheses, &prior);
    let prior_parameter_entropy = conditional_parameter_entropy(hypotheses, &prior);
    let prior_state_entropy = conditional_state_entropy(hypotheses, &prior);
    let prior_joint_entropy = entropy(&prior);

    let model_gain = prior_model_entropy - expected_model_entropy;
    let parameter_gain = prior_parameter_entropy - expected_parameter_entropy;
    let state_gain = prior_state_entropy - expected_state_entropy;
    let joint_gain = prior_joint_entropy - expected_joint_entropy;

    let minimum_component = model_gain.min(parameter_gain).min(state_gain);
    let chain_rule_error = (joint_gain - (model_gain + parameter_gain + state_gain)).abs();

    if chain_rule_error > NUMERICAL_TOLERANCE || minimum_component < -NUMERICAL_TOLERANCE {
        return Err(HierarchicalInformationGainError::NumericalInvariant {
            action_index,
            chain_rule_error_nats: chain_rule_error,
            minimum_component_nats: minimum_component,
        });
    }

    Ok(HierarchicalActionInformationGain {
        action_index,
        outcome_probabilities,
        model_information_gain_nats: clean_information(model_gain),
        parameter_information_gain_nats: clean_information(parameter_gain),
        state_information_gain_nats: clean_information(state_gain),
        joint_information_gain_nats: clean_information(joint_gain),
        chain_rule_error_nats: chain_rule_error,
    })
}

fn validate_hypotheses(
    hypotheses: &[HierarchicalHypothesis],
) -> Result<(usize, usize), HierarchicalInformationGainError> {
    if hypotheses.is_empty() {
        return Err(HierarchicalInformationGainError::EmptyHypotheses);
    }

    let mut seen = BTreeSet::new();
    for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
        for (field, value) in [
            ("model", hypothesis.model_id.as_str()),
            ("parameter", hypothesis.parameter_id.as_str()),
            ("state", hypothesis.state_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(HierarchicalInformationGainError::EmptyIdentifier {
                    hypothesis_index,
                    field,
                });
            }
        }
        let key = (
            hypothesis.model_id.clone(),
            hypothesis.parameter_id.clone(),
            hypothesis.state_id.clone(),
        );
        if !seen.insert(key.clone()) {
            return Err(HierarchicalInformationGainError::DuplicateJointHypothesis {
                model_id: key.0,
                parameter_id: key.1,
                state_id: key.2,
            });
        }
        if !hypothesis.prior_probability.is_finite()
            || !(0.0..=1.0).contains(&hypothesis.prior_probability)
        {
            return Err(HierarchicalInformationGainError::InvalidPrior {
                hypothesis_index,
                probability: hypothesis.prior_probability,
            });
        }
    }

    let prior_sum = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .sum::<f64>();
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(HierarchicalInformationGainError::PriorMassNotNormalized { sum: prior_sum });
    }

    let action_count = hypotheses[0].outcome_probabilities_by_action.len();
    if action_count == 0 {
        return Err(HierarchicalInformationGainError::EmptyActionSet);
    }
    let outcome_count = hypotheses[0].outcome_probabilities_by_action[0].len();
    for action_index in 0..action_count {
        if hypotheses[0].outcome_probabilities_by_action[action_index].is_empty() {
            return Err(HierarchicalInformationGainError::EmptyOutcomeSet { action_index });
        }
    }

    for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
        if hypothesis.outcome_probabilities_by_action.len() != action_count {
            return Err(HierarchicalInformationGainError::InconsistentActionCount {
                hypothesis_index,
                expected: action_count,
                actual: hypothesis.outcome_probabilities_by_action.len(),
            });
        }
        for action_index in 0..action_count {
            let outcomes = &hypothesis.outcome_probabilities_by_action[action_index];
            if outcomes.len() != outcome_count {
                return Err(HierarchicalInformationGainError::InconsistentOutcomeCount {
                    hypothesis_index,
                    action_index,
                    expected: outcome_count,
                    actual: outcomes.len(),
                });
            }
            for (outcome_index, probability) in outcomes.iter().copied().enumerate() {
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err(
                        HierarchicalInformationGainError::InvalidOutcomeProbability {
                            hypothesis_index,
                            action_index,
                            outcome_index,
                            probability,
                        },
                    );
                }
            }
            let sum = outcomes.iter().sum::<f64>();
            if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(HierarchicalInformationGainError::OutcomeMassNotNormalized {
                    hypothesis_index,
                    action_index,
                    sum,
                });
            }
        }
    }

    Ok((action_count, outcome_count))
}

fn conditional_model_entropy(hypotheses: &[HierarchicalHypothesis], weights: &[f64]) -> f64 {
    let mut model_mass = BTreeMap::<&str, f64>::new();
    for (hypothesis, weight) in hypotheses.iter().zip(weights.iter().copied()) {
        *model_mass.entry(hypothesis.model_id.as_str()).or_default() += weight;
    }
    entropy(&model_mass.into_values().collect::<Vec<_>>())
}

fn conditional_parameter_entropy(hypotheses: &[HierarchicalHypothesis], weights: &[f64]) -> f64 {
    let mut model_mass = BTreeMap::<&str, f64>::new();
    let mut parameter_mass = BTreeMap::<(&str, &str), f64>::new();
    for (hypothesis, weight) in hypotheses.iter().zip(weights.iter().copied()) {
        *model_mass.entry(hypothesis.model_id.as_str()).or_default() += weight;
        *parameter_mass
            .entry((
                hypothesis.model_id.as_str(),
                hypothesis.parameter_id.as_str(),
            ))
            .or_default() += weight;
    }

    let mut value = 0.0;
    for ((model_id, _parameter_id), joint_mass) in parameter_mass {
        let parent_mass = model_mass.get(model_id).copied().unwrap_or(0.0);
        if joint_mass > 0.0 && parent_mass > 0.0 {
            let conditional = joint_mass / parent_mass;
            value -= joint_mass * conditional.ln();
        }
    }
    value
}

fn conditional_state_entropy(hypotheses: &[HierarchicalHypothesis], weights: &[f64]) -> f64 {
    let mut parent_mass = BTreeMap::<(&str, &str), f64>::new();
    let mut state_mass = BTreeMap::<(&str, &str, &str), f64>::new();
    for (hypothesis, weight) in hypotheses.iter().zip(weights.iter().copied()) {
        *parent_mass
            .entry((
                hypothesis.model_id.as_str(),
                hypothesis.parameter_id.as_str(),
            ))
            .or_default() += weight;
        *state_mass
            .entry((
                hypothesis.model_id.as_str(),
                hypothesis.parameter_id.as_str(),
                hypothesis.state_id.as_str(),
            ))
            .or_default() += weight;
    }

    let mut value = 0.0;
    for ((model_id, parameter_id, _state_id), joint_mass) in state_mass {
        let parent = parent_mass
            .get(&(model_id, parameter_id))
            .copied()
            .unwrap_or(0.0);
        if joint_mass > 0.0 && parent > 0.0 {
            let conditional = joint_mass / parent;
            value -= joint_mass * conditional.ln();
        }
    }
    value
}

fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn clean_information(value: f64) -> f64 {
    // Suppress only tiny negative round-off. Preserve genuinely small positive
    // information gains; a numerical tolerance must not become a signal floor.
    if value < 0.0 && value > -NUMERICAL_TOLERANCE {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_joint_hypotheses(
        model_count: usize,
        parameter_count: usize,
        state_count: usize,
        action_count: usize,
        outcome_count: usize,
    ) -> Vec<HierarchicalHypothesis> {
        let total = model_count * parameter_count * state_count;
        let mut hypotheses = Vec::with_capacity(total);
        for model in 0..model_count {
            for parameter in 0..parameter_count {
                for state in 0..state_count {
                    hypotheses.push(HierarchicalHypothesis {
                        model_id: format!("m{model}"),
                        parameter_id: format!("p{parameter}"),
                        state_id: format!("s{state}"),
                        prior_probability: 1.0 / total as f64,
                        outcome_probabilities_by_action: vec![
                            vec![
                                1.0 / outcome_count as f64;
                                outcome_count
                            ];
                            action_count
                        ],
                    });
                }
            }
        }
        hypotheses
    }

    #[test]
    fn non_discriminating_actions_have_zero_information_at_every_level() {
        let hypotheses = uniform_joint_hypotheses(2, 2, 2, 2, 2);
        let gains = compute_hierarchical_information_gains(&hypotheses).expect("gains");
        for gain in gains {
            assert_eq!(gain.model_information_gain_nats, 0.0);
            assert_eq!(gain.parameter_information_gain_nats, 0.0);
            assert_eq!(gain.state_information_gain_nats, 0.0);
            assert_eq!(gain.joint_information_gain_nats, 0.0);
            assert!(gain.chain_rule_error_nats < 1e-12);
        }
    }

    #[test]
    fn fully_discriminating_outcomes_satisfy_hierarchical_chain_rule() {
        let mut hypotheses = uniform_joint_hypotheses(2, 2, 2, 1, 8);
        for (index, hypothesis) in hypotheses.iter_mut().enumerate() {
            hypothesis.outcome_probabilities_by_action[0] = vec![0.0; 8];
            hypothesis.outcome_probabilities_by_action[0][index] = 1.0;
        }
        let gain = compute_hierarchical_information_gain(&hypotheses, 0).expect("gain");
        let expected = std::f64::consts::LN_2;
        assert!((gain.model_information_gain_nats - expected).abs() < 1e-10);
        assert!((gain.parameter_information_gain_nats - expected).abs() < 1e-10);
        assert!((gain.state_information_gain_nats - expected).abs() < 1e-10);
        assert!((gain.joint_information_gain_nats - 3.0 * expected).abs() < 1e-10);
        assert!(gain.chain_rule_error_nats < 1e-10);
    }

    #[test]
    fn parameter_information_is_separate_from_model_information() {
        let mut hypotheses = uniform_joint_hypotheses(1, 2, 1, 1, 2);
        for (index, hypothesis) in hypotheses.iter_mut().enumerate() {
            hypothesis.outcome_probabilities_by_action[0] = if index == 0 {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            };
        }
        let gain = compute_hierarchical_information_gain(&hypotheses, 0).expect("gain");
        assert_eq!(gain.model_information_gain_nats, 0.0);
        assert!((gain.parameter_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-10);
        assert_eq!(gain.state_information_gain_nats, 0.0);
    }

    #[test]
    fn state_information_is_separate_from_model_and_parameter_information() {
        let mut hypotheses = uniform_joint_hypotheses(1, 1, 2, 1, 2);
        for (index, hypothesis) in hypotheses.iter_mut().enumerate() {
            hypothesis.outcome_probabilities_by_action[0] = if index == 0 {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            };
        }
        let gain = compute_hierarchical_information_gain(&hypotheses, 0).expect("gain");
        assert_eq!(gain.model_information_gain_nats, 0.0);
        assert_eq!(gain.parameter_information_gain_nats, 0.0);
        assert!((gain.state_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-10);
    }

    #[test]
    fn model_information_is_separate_from_parameter_and_state_information() {
        let mut hypotheses = uniform_joint_hypotheses(2, 1, 1, 1, 2);
        for (index, hypothesis) in hypotheses.iter_mut().enumerate() {
            hypothesis.outcome_probabilities_by_action[0] = if index == 0 {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            };
        }
        let gain = compute_hierarchical_information_gain(&hypotheses, 0).expect("gain");
        assert!((gain.model_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-10);
        assert_eq!(gain.parameter_information_gain_nats, 0.0);
        assert_eq!(gain.state_information_gain_nats, 0.0);
    }

    #[test]
    fn malformed_probability_mass_is_rejected_not_normalized() {
        let mut hypotheses = uniform_joint_hypotheses(1, 1, 1, 1, 2);
        hypotheses[0].outcome_probabilities_by_action[0] = vec![0.2, 0.2];
        assert!(matches!(
            compute_hierarchical_information_gains(&hypotheses),
            Err(HierarchicalInformationGainError::OutcomeMassNotNormalized { .. })
        ));
    }

    #[test]
    fn small_positive_information_gain_is_not_thresholded_to_zero() {
        let epsilon = 1e-6;
        let hypotheses = vec![
            HierarchicalHypothesis {
                model_id: "m0".into(),
                parameter_id: "p0".into(),
                state_id: "s0".into(),
                prior_probability: 0.5,
                outcome_probabilities_by_action: vec![vec![0.5 + epsilon, 0.5 - epsilon]],
            },
            HierarchicalHypothesis {
                model_id: "m1".into(),
                parameter_id: "p0".into(),
                state_id: "s0".into(),
                prior_probability: 0.5,
                outcome_probabilities_by_action: vec![vec![0.5 - epsilon, 0.5 + epsilon]],
            },
        ];
        let gain = compute_hierarchical_information_gain(&hypotheses, 0).expect("gain");
        assert!(gain.model_information_gain_nats > 0.0);
        assert!(gain.model_information_gain_nats < NUMERICAL_TOLERANCE);
        assert!(gain.joint_information_gain_nats > 0.0);
    }

    #[test]
    fn action_selection_can_target_a_specific_epistemic_level() {
        let hypotheses = vec![
            HierarchicalHypothesis {
                model_id: "m0".into(),
                parameter_id: "p0".into(),
                state_id: "s0".into(),
                prior_probability: 0.5,
                outcome_probabilities_by_action: vec![vec![1.0, 0.0], vec![0.5, 0.5]],
            },
            HierarchicalHypothesis {
                model_id: "m1".into(),
                parameter_id: "p0".into(),
                state_id: "s0".into(),
                prior_probability: 0.5,
                outcome_probabilities_by_action: vec![vec![0.0, 1.0], vec![0.5, 0.5]],
            },
        ];
        assert_eq!(
            most_informative_action_at_level(&hypotheses, EpistemicLevel::ModelStructure)
                .expect("action"),
            0
        );
    }
}
