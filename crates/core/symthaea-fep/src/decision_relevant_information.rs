// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Decision-focused diagnostic selection over competing generative-model hypotheses.
//!
//! Entropy reduction is not automatically useful: a probe can identify a model
//! property without changing the best downstream decision. This module reports
//! model-space expected information gain (EIG) separately from expected value
//! of sample information (EVSI), measured as reduction in Bayes risk.
//!
//! The selector minimizes expected posterior Bayes risk plus explicit probe-cost
//! and safety-risk penalties. It declines to probe when no eligible action beats
//! the no-probe Bayes-risk baseline. It does not execute actions or authorize
//! repairs; callers retain those responsibilities.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::hierarchical_information_gain::{
    HierarchicalHypothesis, HierarchicalInformationGainError,
    compute_hierarchical_information_gains,
};

const RISK_TOLERANCE: f64 = 1e-10;

/// A candidate world/fault model with predictive observation distributions.
///
/// outcome_probabilities_by_action[a][o] is P(o | model, probe a).
/// loss_by_final_decision[d] is the task loss if final decision d is taken
/// when this hypothesis is the true model. Losses are non-negative and share
/// a common unit across hypotheses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionHypothesis {
    pub id: String,
    pub prior_probability: f64,
    pub outcome_probabilities_by_action: Vec<Vec<f64>>,
    pub loss_by_final_decision: Vec<f64>,
}

/// A diagnostic probe with explicit operational cost and safety risk.
///
/// safety_risk is a caller-defined value in [0, 1]. It is not a calibrated
/// probability unless the caller's risk model makes it one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticAction {
    pub id: String,
    pub observation_cost: f64,
    pub safety_risk: f64,
}

/// Cost/risk conversion and hard eligibility boundary for diagnostic probes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecisionDiagnosticConfig {
    pub observation_cost_weight: f64,
    pub safety_risk_weight: f64,
    pub maximum_safety_risk: f64,
}

impl Default for DecisionDiagnosticConfig {
    fn default() -> Self {
        Self {
            observation_cost_weight: 1.0,
            safety_risk_weight: 1.0,
            maximum_safety_risk: 1.0,
        }
    }
}

/// Prospective evaluation of one probe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticActionEvaluation {
    pub action_index: usize,
    pub action_id: String,
    pub outcome_probabilities: Vec<f64>,
    /// None denotes an outcome with zero predictive probability.
    pub posterior_bayes_risk_by_outcome: Vec<Option<f64>>,
    pub expected_posterior_bayes_risk: f64,
    pub expected_value_of_sample_information: f64,
    pub model_information_gain_nats: f64,
    /// Information gain about the identity of the optimal downstream decision.
    /// This is targeted decision information, not expected utility improvement.
    pub decision_information_gain_nats: f64,
    pub observation_cost: f64,
    pub safety_risk: f64,
    pub eligible: bool,
    /// None for an action rejected by the hard safety boundary.
    pub total_score: Option<f64>,
}

/// Result of a decision-focused probe comparison.
///
/// selected_action_index == None means that no eligible probe improves on
/// taking the best current decision without another observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionDiagnosticEvaluation {
    pub prior_bayes_risk: f64,
    pub no_probe_score: f64,
    pub selected_action_index: Option<usize>,
    pub actions: Vec<DiagnosticActionEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum DecisionDiagnosticError {
    #[error(transparent)]
    InformationGain(#[from] HierarchicalInformationGainError),
    #[error("at least one diagnostic action is required")]
    EmptyActionSet,
    #[error("at least one final decision is required")]
    EmptyDecisionSet,
    #[error("hypothesis {hypothesis_index} has an empty identifier")]
    EmptyHypothesisId { hypothesis_index: usize },
    #[error("duplicate hypothesis identifier: {0}")]
    DuplicateHypothesisId(String),
    #[error("hypothesis {hypothesis_index} has {actual} decisions; expected {expected}")]
    InconsistentDecisionCount {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("hypothesis {hypothesis_index}, decision {decision_index} has invalid loss")]
    InvalidDecisionLoss {
        hypothesis_index: usize,
        decision_index: usize,
    },
    #[error("action {action_index} has an empty identifier")]
    EmptyActionId { action_index: usize },
    #[error("duplicate action identifier: {0}")]
    DuplicateActionId(String),
    #[error("action {action_index} has an invalid observation cost")]
    InvalidObservationCost { action_index: usize },
    #[error("action {action_index} has a safety risk outside [0, 1]")]
    InvalidSafetyRisk { action_index: usize },
    #[error("received {actual} action descriptors for {expected} modelled actions")]
    ActionCountMismatch { expected: usize, actual: usize },
    #[error("decision diagnostic configuration is invalid")]
    InvalidConfig,
    #[error("action index {action_index} is outside 0..{action_count}")]
    ActionOutOfRange {
        action_index: usize,
        action_count: usize,
    },
    #[error(
        "outcome index {outcome_index} is outside 0..{outcome_count} for action {action_index}"
    )]
    OutcomeOutOfRange {
        action_index: usize,
        outcome_index: usize,
        outcome_count: usize,
    },
    #[error(
        "observation {outcome_index} has zero predictive probability under action {action_index}"
    )]
    ImpossibleObservation {
        action_index: usize,
        outcome_index: usize,
    },
    #[error("expected posterior Bayes risk {expected_risk} exceeds prior risk {prior_risk}")]
    NonMonotonicBayesRisk { prior_risk: f64, expected_risk: f64 },
    #[error("decision-risk calculation produced a non-finite value")]
    NonFiniteRisk,
    #[error(
        "decision information gain {decision_information_gain_nats} exceeds model information gain {model_information_gain_nats} for action {action_index}"
    )]
    DecisionInformationExceedsModelInformation {
        action_index: usize,
        decision_information_gain_nats: f64,
        model_information_gain_nats: f64,
    },
}

/// Evaluate all diagnostic probes and select only a probe with positive
/// decision value after explicit cost and safety penalties.
///
/// EVSI(a) = min_d E[L(d, M)]
///           - E_o[min_d E[L(d, M) | o, a]]
///
/// The model-space mutual information I(M; O | a) is retained as a separate
/// diagnostic so experiments can test whether entropy reduction predicts
/// decision improvement. A no-probe option is always included.
pub fn evaluate_decision_focused_diagnostics(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    config: DecisionDiagnosticConfig,
) -> Result<DecisionDiagnosticEvaluation, DecisionDiagnosticError> {
    if actions.is_empty() {
        return Err(DecisionDiagnosticError::EmptyActionSet);
    }
    validate_config(config)?;

    let joint = validate_hypotheses(hypotheses)?;
    let information = compute_hierarchical_information_gains(&joint)?;
    if actions.len() != information.len() {
        return Err(DecisionDiagnosticError::ActionCountMismatch {
            expected: information.len(),
            actual: actions.len(),
        });
    }
    validate_actions(actions)?;

    let decision_count = hypotheses[0].loss_by_final_decision.len();
    let prior: Vec<f64> = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect();
    let prior_bayes_risk = bayes_risk(&prior, hypotheses, decision_count)?;
    let prior_optimal_decision = optimal_decision_distribution(&prior, hypotheses, decision_count);
    let prior_decision_entropy = categorical_entropy(&prior_optimal_decision);
    let mut best_score = prior_bayes_risk;
    let mut selected_action_index = None;
    let mut evaluations = Vec::with_capacity(actions.len());

    for (action_index, action) in actions.iter().enumerate() {
        let action_information = &information[action_index];
        let mut posterior_risks =
            Vec::with_capacity(action_information.outcome_probabilities.len());
        let mut expected_posterior_bayes_risk = 0.0;

        for (outcome_index, outcome_probability) in action_information
            .outcome_probabilities
            .iter()
            .copied()
            .enumerate()
        {
            if outcome_probability == 0.0 {
                posterior_risks.push(None);
                continue;
            }
            let posterior = posterior_for_observation(
                hypotheses,
                action_index,
                outcome_index,
                outcome_probability,
            );
            let risk = bayes_risk(&posterior, hypotheses, decision_count)?;
            expected_posterior_bayes_risk += outcome_probability * risk;
            posterior_risks.push(Some(risk));
        }

        if !expected_posterior_bayes_risk.is_finite() {
            return Err(DecisionDiagnosticError::NonFiniteRisk);
        }
        if expected_posterior_bayes_risk > prior_bayes_risk + RISK_TOLERANCE {
            return Err(DecisionDiagnosticError::NonMonotonicBayesRisk {
                prior_risk: prior_bayes_risk,
                expected_risk: expected_posterior_bayes_risk,
            });
        }
        if expected_posterior_bayes_risk > prior_bayes_risk {
            expected_posterior_bayes_risk = prior_bayes_risk;
        }
        let expected_value_of_sample_information =
            (prior_bayes_risk - expected_posterior_bayes_risk).max(0.0);
        let expected_posterior_decision_entropy = action_information
            .outcome_probabilities
            .iter()
            .enumerate()
            .filter(|(_, probability)| **probability > 0.0)
            .map(|(outcome_index, probability)| {
                let posterior = posterior_for_observation(
                    hypotheses,
                    action_index,
                    outcome_index,
                    *probability,
                );
                let decision_distribution =
                    optimal_decision_distribution(&posterior, hypotheses, decision_count);
                probability * categorical_entropy(&decision_distribution)
            })
            .sum::<f64>();
        let raw_decision_information_gain =
            prior_decision_entropy - expected_posterior_decision_entropy;
        let decision_information_gain_nats = raw_decision_information_gain.max(0.0);
        if decision_information_gain_nats
            > action_information.model_information_gain_nats + RISK_TOLERANCE
        {
            return Err(
                DecisionDiagnosticError::DecisionInformationExceedsModelInformation {
                    action_index,
                    decision_information_gain_nats,
                    model_information_gain_nats: action_information.model_information_gain_nats,
                },
            );
        }
        let decision_information_gain_nats =
            decision_information_gain_nats.min(action_information.model_information_gain_nats);
        let eligible = action.safety_risk <= config.maximum_safety_risk;
        let total_score = if eligible {
            let score = expected_posterior_bayes_risk
                + config.observation_cost_weight * action.observation_cost
                + config.safety_risk_weight * action.safety_risk;
            if !score.is_finite() {
                return Err(DecisionDiagnosticError::NonFiniteRisk);
            }
            Some(score)
        } else {
            None
        };

        if let Some(score) = total_score
            && score + RISK_TOLERANCE < best_score
        {
            best_score = score;
            selected_action_index = Some(action_index);
        }

        evaluations.push(DiagnosticActionEvaluation {
            action_index,
            action_id: action.id.clone(),
            outcome_probabilities: action_information.outcome_probabilities.clone(),
            posterior_bayes_risk_by_outcome: posterior_risks,
            expected_posterior_bayes_risk,
            expected_value_of_sample_information,
            model_information_gain_nats: action_information.model_information_gain_nats,
            decision_information_gain_nats,
            observation_cost: action.observation_cost,
            safety_risk: action.safety_risk,
            eligible,
            total_score,
        });
    }

    Ok(DecisionDiagnosticEvaluation {
        prior_bayes_risk,
        no_probe_score: prior_bayes_risk,
        selected_action_index,
        actions: evaluations,
    })
}

/// Update posterior model probabilities after observing a diagnostic outcome.
///
/// The function is fail-closed: an outcome impossible under every hypothesis
/// returns an error rather than preserving or silently normalizing the prior.
pub fn posterior_model_probabilities(
    hypotheses: &[DecisionHypothesis],
    action_index: usize,
    outcome_index: usize,
) -> Result<Vec<f64>, DecisionDiagnosticError> {
    let joint = validate_hypotheses(hypotheses)?;
    let information = compute_hierarchical_information_gains(&joint)?;
    let action =
        information
            .get(action_index)
            .ok_or(DecisionDiagnosticError::ActionOutOfRange {
                action_index,
                action_count: information.len(),
            })?;
    let outcome_count = action.outcome_probabilities.len();
    if outcome_index >= outcome_count {
        return Err(DecisionDiagnosticError::OutcomeOutOfRange {
            action_index,
            outcome_index,
            outcome_count,
        });
    }
    let evidence_probability = action.outcome_probabilities[outcome_index];
    if evidence_probability == 0.0 {
        return Err(DecisionDiagnosticError::ImpossibleObservation {
            action_index,
            outcome_index,
        });
    }
    Ok(posterior_for_observation(
        hypotheses,
        action_index,
        outcome_index,
        evidence_probability,
    ))
}

fn validate_hypotheses(
    hypotheses: &[DecisionHypothesis],
) -> Result<Vec<HierarchicalHypothesis>, DecisionDiagnosticError> {
    if hypotheses.is_empty() {
        return Err(HierarchicalInformationGainError::EmptyHypotheses.into());
    }
    let decision_count = hypotheses[0].loss_by_final_decision.len();
    if decision_count == 0 {
        return Err(DecisionDiagnosticError::EmptyDecisionSet);
    }

    let mut identifiers = std::collections::BTreeSet::new();
    let mut joint = Vec::with_capacity(hypotheses.len());
    for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
        if hypothesis.id.trim().is_empty() {
            return Err(DecisionDiagnosticError::EmptyHypothesisId { hypothesis_index });
        }
        if !identifiers.insert(hypothesis.id.clone()) {
            return Err(DecisionDiagnosticError::DuplicateHypothesisId(
                hypothesis.id.clone(),
            ));
        }
        if hypothesis.loss_by_final_decision.len() != decision_count {
            return Err(DecisionDiagnosticError::InconsistentDecisionCount {
                hypothesis_index,
                expected: decision_count,
                actual: hypothesis.loss_by_final_decision.len(),
            });
        }
        for (decision_index, loss) in hypothesis.loss_by_final_decision.iter().enumerate() {
            if !loss.is_finite() || *loss < 0.0 {
                return Err(DecisionDiagnosticError::InvalidDecisionLoss {
                    hypothesis_index,
                    decision_index,
                });
            }
        }
        joint.push(HierarchicalHypothesis {
            model_id: hypothesis.id.clone(),
            parameter_id: "decision-policy-parameter".to_owned(),
            state_id: "decision-policy-state".to_owned(),
            prior_probability: hypothesis.prior_probability,
            outcome_probabilities_by_action: hypothesis.outcome_probabilities_by_action.clone(),
        });
    }
    Ok(joint)
}

fn validate_actions(actions: &[DiagnosticAction]) -> Result<(), DecisionDiagnosticError> {
    let mut identifiers = std::collections::BTreeSet::new();
    for (action_index, action) in actions.iter().enumerate() {
        if action.id.trim().is_empty() {
            return Err(DecisionDiagnosticError::EmptyActionId { action_index });
        }
        if !identifiers.insert(action.id.clone()) {
            return Err(DecisionDiagnosticError::DuplicateActionId(
                action.id.clone(),
            ));
        }
        if !action.observation_cost.is_finite() || action.observation_cost < 0.0 {
            return Err(DecisionDiagnosticError::InvalidObservationCost { action_index });
        }
        if !action.safety_risk.is_finite() || !(0.0..=1.0).contains(&action.safety_risk) {
            return Err(DecisionDiagnosticError::InvalidSafetyRisk { action_index });
        }
    }
    Ok(())
}

fn validate_config(config: DecisionDiagnosticConfig) -> Result<(), DecisionDiagnosticError> {
    if !config.observation_cost_weight.is_finite()
        || config.observation_cost_weight < 0.0
        || !config.safety_risk_weight.is_finite()
        || config.safety_risk_weight < 0.0
        || !config.maximum_safety_risk.is_finite()
        || !(0.0..=1.0).contains(&config.maximum_safety_risk)
    {
        return Err(DecisionDiagnosticError::InvalidConfig);
    }
    Ok(())
}

fn posterior_for_observation(
    hypotheses: &[DecisionHypothesis],
    action_index: usize,
    outcome_index: usize,
    evidence_probability: f64,
) -> Vec<f64> {
    hypotheses
        .iter()
        .map(|hypothesis| {
            hypothesis.prior_probability
                * hypothesis.outcome_probabilities_by_action[action_index][outcome_index]
                / evidence_probability
        })
        .collect()
}

fn optimal_decision_distribution(
    model_probabilities: &[f64],
    hypotheses: &[DecisionHypothesis],
    decision_count: usize,
) -> Vec<f64> {
    let mut distribution = vec![0.0; decision_count];
    for (hypothesis, probability) in hypotheses.iter().zip(model_probabilities) {
        let best = (0..decision_count)
            .min_by(|left, right| {
                hypothesis.loss_by_final_decision[*left]
                    .total_cmp(&hypothesis.loss_by_final_decision[*right])
            })
            .unwrap_or(0);
        distribution[best] += probability;
    }
    distribution
}

fn categorical_entropy(probabilities: &[f64]) -> f64 {
    -probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| probability * probability.ln())
        .sum::<f64>()
}

fn bayes_risk(
    model_probabilities: &[f64],
    hypotheses: &[DecisionHypothesis],
    decision_count: usize,
) -> Result<f64, DecisionDiagnosticError> {
    let risk = (0..decision_count)
        .map(|decision_index| {
            hypotheses
                .iter()
                .zip(model_probabilities)
                .map(|(hypothesis, probability)| {
                    probability * hypothesis.loss_by_final_decision[decision_index]
                })
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    if risk.is_finite() {
        Ok(risk)
    } else {
        Err(DecisionDiagnosticError::NonFiniteRisk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_diagnosis(
        same_optimal_decision: bool,
    ) -> (Vec<DecisionHypothesis>, Vec<DiagnosticAction>) {
        let loss_a = vec![0.0, 1.0];
        let loss_b = if same_optimal_decision {
            vec![0.0, 1.0]
        } else {
            vec![1.0, 0.0]
        };
        let hypotheses = vec![
            DecisionHypothesis {
                id: "fault-a".into(),
                prior_probability: 0.5,
                outcome_probabilities_by_action: vec![vec![1.0, 0.0], vec![0.5, 0.5]],
                loss_by_final_decision: loss_a,
            },
            DecisionHypothesis {
                id: "fault-b".into(),
                prior_probability: 0.5,
                outcome_probabilities_by_action: vec![vec![0.0, 1.0], vec![0.5, 0.5]],
                loss_by_final_decision: loss_b,
            },
        ];
        let actions = vec![
            DiagnosticAction {
                id: "inspect".into(),
                observation_cost: 0.0,
                safety_risk: 0.0,
            },
            DiagnosticAction {
                id: "wait".into(),
                observation_cost: 0.0,
                safety_risk: 0.0,
            },
        ];
        (hypotheses, actions)
    }

    #[test]
    fn chooses_probe_that_reduces_downstream_bayes_risk() {
        let (hypotheses, actions) = binary_diagnosis(false);
        let result = evaluate_decision_focused_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
        )
        .expect("valid decision-focused diagnostic problem");

        assert_eq!(result.selected_action_index, Some(0));
        assert!((result.prior_bayes_risk - 0.5).abs() < 1e-12);
        assert!(result.actions[0].expected_posterior_bayes_risk.abs() < 1e-12);
        assert!((result.actions[0].expected_value_of_sample_information - 0.5).abs() < 1e-12);
        assert!(
            (result.actions[0].model_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-10
        );
        assert!(
            (result.actions[0].decision_information_gain_nats - std::f64::consts::LN_2).abs()
                < 1e-10
        );
        assert_eq!(result.actions[1].expected_value_of_sample_information, 0.0);
    }

    #[test]
    fn lower_information_probe_wins_when_it_reduces_decision_risk() {
        let hypotheses: Vec<DecisionHypothesis> = (0..4)
            .map(|index| {
                let fault_a = index < 2;
                let nuisance_zero = index % 2 == 0;
                DecisionHypothesis {
                    id: format!("model-{index}"),
                    prior_probability: 0.25,
                    outcome_probabilities_by_action: vec![
                        if nuisance_zero {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        },
                        if fault_a {
                            vec![0.7, 0.3]
                        } else {
                            vec![0.3, 0.7]
                        },
                    ],
                    loss_by_final_decision: if fault_a {
                        vec![0.0, 1.0]
                    } else {
                        vec![1.0, 0.0]
                    },
                }
            })
            .collect();
        let actions = vec![
            DiagnosticAction {
                id: "reveal-irrelevant-nuisance".into(),
                observation_cost: 0.0,
                safety_risk: 0.0,
            },
            DiagnosticAction {
                id: "weakly-identify-decision-relevant-fault".into(),
                observation_cost: 0.0,
                safety_risk: 0.0,
            },
        ];

        let result = evaluate_decision_focused_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
        )
        .expect("valid decision-focused diagnostic problem");

        assert!(
            result.actions[0].model_information_gain_nats
                > result.actions[1].model_information_gain_nats
        );
        assert!(result.actions[0].expected_value_of_sample_information.abs() < 1e-12);
        assert!((result.actions[1].expected_value_of_sample_information - 0.2).abs() < 1e-12);
        assert!(result.actions[0].decision_information_gain_nats.abs() < 1e-12);
        assert!(result.actions[1].decision_information_gain_nats > 0.0);
        assert_eq!(result.selected_action_index, Some(1));
    }

    #[test]
    fn high_information_without_decision_value_does_not_trigger_a_probe() {
        let (hypotheses, actions) = binary_diagnosis(true);
        let result = evaluate_decision_focused_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
        )
        .expect("valid decision-focused diagnostic problem");

        assert_eq!(result.selected_action_index, None);
        assert!(
            (result.actions[0].model_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-10
        );
        assert!(result.actions[0].expected_value_of_sample_information.abs() < 1e-12);
        assert!(result.actions[0].decision_information_gain_nats.abs() < 1e-12);
    }

    #[test]
    fn probe_cost_can_exceed_its_expected_decision_value() {
        let (hypotheses, mut actions) = binary_diagnosis(false);
        actions[0].observation_cost = 1.0;
        let config = DecisionDiagnosticConfig {
            observation_cost_weight: 1.0,
            safety_risk_weight: 1.0,
            maximum_safety_risk: 1.0,
        };
        let result = evaluate_decision_focused_diagnostics(&hypotheses, &actions, config)
            .expect("valid cost-aware diagnostic problem");

        assert_eq!(result.selected_action_index, None);
        assert!((result.actions[0].expected_value_of_sample_information - 0.5).abs() < 1e-12);
    }

    #[test]
    fn hard_safety_boundary_excludes_an_informative_probe() {
        let (hypotheses, mut actions) = binary_diagnosis(false);
        actions[0].safety_risk = 0.8;
        let config = DecisionDiagnosticConfig {
            maximum_safety_risk: 0.2,
            ..DecisionDiagnosticConfig::default()
        };
        let result = evaluate_decision_focused_diagnostics(&hypotheses, &actions, config)
            .expect("valid safety-constrained diagnostic problem");

        assert_eq!(result.selected_action_index, None);
        assert!(!result.actions[0].eligible);
        assert_eq!(result.actions[0].total_score, None);
    }

    #[test]
    fn posterior_update_is_bayes_rule_and_sums_to_one() {
        let (hypotheses, _) = binary_diagnosis(false);
        let posterior =
            posterior_model_probabilities(&hypotheses, 0, 0).expect("possible observation");
        assert_eq!(posterior, vec![1.0, 0.0]);
        assert!((posterior.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn impossible_observations_fail_closed() {
        let mut hypotheses = binary_diagnosis(false).0;
        for hypothesis in &mut hypotheses {
            hypothesis.outcome_probabilities_by_action[1] = vec![1.0, 0.0];
        }
        assert!(matches!(
            posterior_model_probabilities(&hypotheses, 1, 1),
            Err(DecisionDiagnosticError::ImpossibleObservation {
                action_index: 1,
                outcome_index: 1
            })
        ));
    }

    #[test]
    fn invalid_config_and_negative_losses_are_rejected() {
        let (mut hypotheses, actions) = binary_diagnosis(false);
        let invalid = DecisionDiagnosticConfig {
            observation_cost_weight: f64::NAN,
            ..DecisionDiagnosticConfig::default()
        };
        assert!(matches!(
            evaluate_decision_focused_diagnostics(&hypotheses, &actions, invalid),
            Err(DecisionDiagnosticError::InvalidConfig)
        ));
        hypotheses[0].loss_by_final_decision[0] = -1.0;
        assert!(matches!(
            evaluate_decision_focused_diagnostics(
                &hypotheses,
                &actions,
                DecisionDiagnosticConfig::default()
            ),
            Err(DecisionDiagnosticError::InvalidDecisionLoss { .. })
        ));
    }
}
