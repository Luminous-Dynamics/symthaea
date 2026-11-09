// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Open-world model-adequacy monitoring and finite-TV robust diagnostics.
//!
//! This module separates uncertainty within a registered hypothesis family
//! from evidence that the predictive family itself is inadequate.
//!
//! The adequacy monitor consumes prequential categorical predictions: each
//! distribution must be frozen before its corresponding observation. A
//! zero-support observation is an immediate adequacy failure. Passing the
//! configured score thresholds means only "no detected inadequacy", never
//! proof that the model family is correct.
//!
//! The robust diagnostic selector wraps nominal EVSI with a finite categorical
//! total-variation ambiguity set around each action's predictive outcome
//! distribution. It exactly maximizes expected posterior loss over the finite
//! outcome TV ball. Outcomes assigned zero nominal probability have no
//! in-family posterior; they receive the maximum declared task loss, a
//! conservative open-support penalty.
//!
//! This is a transparent finite-outcome baseline. It is not an implementation
//! of MMD/RoBAS, a general distributionally robust POMDP solver, or an
//! action-execution adapter.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::decision_relevant_information::{
    DecisionDiagnosticConfig, DecisionDiagnosticError, DecisionHypothesis, DiagnosticAction,
    evaluate_decision_focused_diagnostics,
};

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const RISK_TOLERANCE: f64 = 1e-10;

/// One prequential categorical prediction and its subsequently observed outcome.
///
/// The prediction must be generated from evidence available before the
/// observation. This type cannot prove temporal ordering; callers must bind
/// records to source, task, and observation digests in their surrounding receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrequentialCategoricalObservation {
    pub predicted_outcome_probabilities: Vec<f64>,
    pub observed_outcome_index: usize,
}

/// Externally calibrated thresholds for a declared task and observation stream.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelAdequacyThresholds {
    pub minimum_observations: usize,
    /// Positive floor used to flag very-low-probability observations.
    pub minimum_predictive_probability: f64,
    pub maximum_mean_log_loss: f64,
    /// Multiclass Brier score is the sum over categories and lies in [0, 2].
    pub maximum_mean_brier_score: f64,
    pub maximum_low_probability_rate: f64,
}

/// Model-family adequacy state. No state asserts that a model is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelAdequacyStatus {
    /// Not enough prequential observations to evaluate the frozen thresholds.
    InsufficientEvidence,
    /// The supplied thresholds detected no failure; adequacy is not proven.
    NoDetectedInadequacy,
    /// A support violation or a preregistered score threshold was crossed.
    Inadequate,
}

/// Prequential predictive adequacy report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelAdequacyReport {
    pub status: ModelAdequacyStatus,
    pub observation_count: usize,
    pub zero_support_count: usize,
    pub low_probability_count: usize,
    pub low_probability_rate: f64,
    /// None when any observation had zero predictive probability.
    pub mean_log_loss: Option<f64>,
    pub mean_multiclass_brier_score: Option<f64>,
    pub minimum_observed_probability: Option<f64>,
    pub thresholds: ModelAdequacyThresholds,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ModelAdequacyError {
    #[error("model-adequacy thresholds are invalid")]
    InvalidThresholds,
    #[error("observation {observation_index} has no outcome categories")]
    EmptyOutcomeSet { observation_index: usize },
    #[error("observation {observation_index} has an invalid probability at category {category_index}")]
    InvalidProbability {
        observation_index: usize,
        category_index: usize,
    },
    #[error("observation {observation_index} probability mass sums to {sum}, expected 1")]
    ProbabilityMassNotNormalized { observation_index: usize, sum: f64 },
    #[error("observation {observation_index} has outcome index {outcome_index} outside 0..{outcome_count}")]
    OutcomeIndexOutOfRange {
        observation_index: usize,
        outcome_index: usize,
        outcome_count: usize,
    },
}

/// Evaluate frozen, pre-observation predictions against subsequent outcomes.
///
/// A zero-support observation is never repaired by adding epsilon or silently
/// renormalizing the registered family. Its log score is None, while the
/// support-violation count forces the Inadequate status.
pub fn evaluate_prequential_model_adequacy(
    observations: &[PrequentialCategoricalObservation],
    thresholds: ModelAdequacyThresholds,
) -> Result<ModelAdequacyReport, ModelAdequacyError> {
    validate_adequacy_thresholds(thresholds)?;

    let mut zero_support_count = 0;
    let mut low_probability_count = 0;
    let mut log_loss_sum = 0.0;
    let mut brier_sum = 0.0;
    let mut minimum_observed_probability: Option<f64> = None;

    for (observation_index, observation) in observations.iter().enumerate() {
        let probabilities = &observation.predicted_outcome_probabilities;
        if probabilities.is_empty() {
            return Err(ModelAdequacyError::EmptyOutcomeSet { observation_index });
        }
        let mut probability_sum = 0.0;
        for (category_index, probability) in probabilities.iter().copied().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(ModelAdequacyError::InvalidProbability {
                    observation_index,
                    category_index,
                });
            }
            probability_sum += probability;
        }
        if (probability_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(ModelAdequacyError::ProbabilityMassNotNormalized {
                observation_index,
                sum: probability_sum,
            });
        }
        if observation.observed_outcome_index >= probabilities.len() {
            return Err(ModelAdequacyError::OutcomeIndexOutOfRange {
                observation_index,
                outcome_index: observation.observed_outcome_index,
                outcome_count: probabilities.len(),
            });
        }

        let observed_probability = probabilities[observation.observed_outcome_index];
        minimum_observed_probability = Some(
            minimum_observed_probability
                .map_or(observed_probability, |minimum| minimum.min(observed_probability)),
        );
        if observed_probability == 0.0 {
            zero_support_count += 1;
        } else {
            log_loss_sum -= observed_probability.ln();
        }
        if observed_probability < thresholds.minimum_predictive_probability {
            low_probability_count += 1;
        }

        let brier_score = probabilities
            .iter()
            .enumerate()
            .map(|(category_index, probability)| {
                let target = if category_index == observation.observed_outcome_index {
                    1.0
                } else {
                    0.0
                };
                (probability - target).powi(2)
            })
            .sum::<f64>();
        brier_sum += brier_score;
    }

    let observation_count = observations.len();
    let low_probability_rate = if observation_count == 0 {
        0.0
    } else {
        low_probability_count as f64 / observation_count as f64
    };
    let mean_log_loss = if observation_count == 0 || zero_support_count > 0 {
        None
    } else {
        Some(log_loss_sum / observation_count as f64)
    };
    let mean_multiclass_brier_score = if observation_count == 0 {
        None
    } else {
        Some(brier_sum / observation_count as f64)
    };

    let status = if zero_support_count > 0 {
        ModelAdequacyStatus::Inadequate
    } else if observation_count < thresholds.minimum_observations {
        ModelAdequacyStatus::InsufficientEvidence
    } else if mean_log_loss.is_some_and(|score| score > thresholds.maximum_mean_log_loss)
        || mean_multiclass_brier_score
            .is_some_and(|score| score > thresholds.maximum_mean_brier_score)
        || low_probability_rate > thresholds.maximum_low_probability_rate
    {
        ModelAdequacyStatus::Inadequate
    } else {
        ModelAdequacyStatus::NoDetectedInadequacy
    };

    Ok(ModelAdequacyReport {
        status,
        observation_count,
        zero_support_count,
        low_probability_count,
        low_probability_rate,
        mean_log_loss,
        mean_multiclass_brier_score,
        minimum_observed_probability,
        thresholds,
    })
}

fn validate_adequacy_thresholds(
    thresholds: ModelAdequacyThresholds,
) -> Result<(), ModelAdequacyError> {
    if thresholds.minimum_observations == 0
        || !thresholds.minimum_predictive_probability.is_finite()
        || !(0.0..=1.0).contains(&thresholds.minimum_predictive_probability)
        || thresholds.minimum_predictive_probability == 0.0
        || !thresholds.maximum_mean_log_loss.is_finite()
        || thresholds.maximum_mean_log_loss < 0.0
        || !thresholds.maximum_mean_brier_score.is_finite()
        || !(0.0..=2.0).contains(&thresholds.maximum_mean_brier_score)
        || !thresholds.maximum_low_probability_rate.is_finite()
        || !(0.0..=1.0).contains(&thresholds.maximum_low_probability_rate)
    {
        return Err(ModelAdequacyError::InvalidThresholds);
    }
    Ok(())
}

/// Finite categorical total-variation robust diagnostic-selection result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TvRobustDecisionEvaluation {
    pub total_variation_radius: f64,
    pub no_probe_bayes_risk: f64,
    pub nominal_selected_action_index: Option<usize>,
    pub robust_selected_action_index: Option<usize>,
    pub actions: Vec<TvRobustActionEvaluation>,
}

/// Nominal and worst-case action values under a declared TV ambiguity set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TvRobustActionEvaluation {
    pub action_index: usize,
    pub action_id: String,
    pub eligible: bool,
    pub nominal_expected_posterior_bayes_risk: f64,
    pub worst_case_expected_posterior_bayes_risk: f64,
    pub nominal_total_score: Option<f64>,
    pub worst_case_total_score: Option<f64>,
    pub nominal_expected_value_of_sample_information: f64,
    pub worst_case_expected_value_of_sample_information: f64,
    pub nominal_model_information_gain_nats: f64,
    pub nominal_decision_information_gain_nats: f64,
    /// Adversarial outcome probabilities attaining the worst-case TV risk.
    pub worst_case_outcome_probabilities: Vec<f64>,
    /// Outcomes with zero nominal support use the maximum declared task loss.
    pub unsupported_outcome_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum TvRobustDecisionError {
    #[error(transparent)]
    DecisionDiagnostic(#[from] DecisionDiagnosticError),
    #[error("total-variation ambiguity radius must be finite and in [0, 1]")]
    InvalidTotalVariationRadius,
    #[error("probability and loss vectors must have the same non-zero length")]
    VectorLengthMismatch,
    #[error("probability at index {index} is invalid")]
    InvalidProbability { index: usize },
    #[error("probability mass sums to {sum}, expected 1")]
    ProbabilityMassNotNormalized { sum: f64 },
    #[error("loss at index {index} is non-finite")]
    InvalidLoss { index: usize },
    #[error("robust expected risk is non-finite")]
    NonFiniteRisk,
}

/// Evaluate decision-focused diagnostics against a total-variation ball around
/// each nominal predictive outcome distribution.
///
/// For each action a, the ambiguity set is Q_a(rho) = {q : TV(q, p_a) <= rho}.
/// The exact worst-case expectation of outcome-specific posterior Bayes risk
/// is computed by transferring up to rho probability mass from the least
/// costly outcomes to the most costly outcomes. Zero-support outcomes receive
/// the maximum declared task loss as a conservative bound.
///
/// The no-probe option retains nominal prior Bayes risk. Thus the robust
/// selector protects against outcome-distribution perturbation conditional on
/// the supplied model prior and posterior update rule; it does not robustify
/// those model priors or prove coverage outside the declared TV radius.
pub fn evaluate_tv_robust_decision_diagnostics(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    decision_config: DecisionDiagnosticConfig,
    total_variation_radius: f64,
) -> Result<TvRobustDecisionEvaluation, TvRobustDecisionError> {
    if !total_variation_radius.is_finite()
        || !(0.0..=1.0).contains(&total_variation_radius)
    {
        return Err(TvRobustDecisionError::InvalidTotalVariationRadius);
    }
    let nominal =
        evaluate_decision_focused_diagnostics(hypotheses, actions, decision_config)?;
    let maximum_declared_loss = hypotheses
        .iter()
        .flat_map(|hypothesis| hypothesis.loss_by_final_decision.iter().copied())
        .fold(0.0_f64, f64::max);
    let mut robust_selected_action_index = None;
    let mut best_robust_score = nominal.prior_bayes_risk;
    let mut robust_actions = Vec::with_capacity(nominal.actions.len());

    for action in &nominal.actions {
        let mut outcome_losses = Vec::with_capacity(action.posterior_bayes_risk_by_outcome.len());
        let mut unsupported_outcome_indices = Vec::new();
        for (outcome_index, posterior_risk) in
            action.posterior_bayes_risk_by_outcome.iter().enumerate()
        {
            match posterior_risk {
                Some(risk) => outcome_losses.push(*risk),
                None => {
                    outcome_losses.push(maximum_declared_loss);
                    unsupported_outcome_indices.push(outcome_index);
                }
            }
        }
        let worst_case_distribution = worst_case_distribution_within_tv(
            &action.outcome_probabilities,
            &outcome_losses,
            total_variation_radius,
        )?;
        let worst_case_expected_posterior_bayes_risk = worst_case_distribution
            .iter()
            .zip(&outcome_losses)
            .map(|(probability, risk)| probability * risk)
            .sum::<f64>();
        if !worst_case_expected_posterior_bayes_risk.is_finite() {
            return Err(TvRobustDecisionError::NonFiniteRisk);
        }
        let nominal_expected_value_of_sample_information =
            nominal.prior_bayes_risk - action.expected_posterior_bayes_risk;
        let worst_case_expected_value_of_sample_information =
            nominal.prior_bayes_risk - worst_case_expected_posterior_bayes_risk;

        let worst_case_total_score = if action.eligible {
            Some(
                worst_case_expected_posterior_bayes_risk
                    + decision_config.observation_cost_weight * action.observation_cost
                    + decision_config.safety_risk_weight * action.safety_risk,
            )
        } else {
            None
        };
        if let Some(score) = worst_case_total_score
            && score + RISK_TOLERANCE < best_robust_score
        {
            best_robust_score = score;
            robust_selected_action_index = Some(action.action_index);
        }

        robust_actions.push(TvRobustActionEvaluation {
            action_index: action.action_index,
            action_id: action.action_id.clone(),
            eligible: action.eligible,
            nominal_expected_posterior_bayes_risk: action.expected_posterior_bayes_risk,
            worst_case_expected_posterior_bayes_risk,
            nominal_total_score: action.total_score,
            worst_case_total_score,
            nominal_expected_value_of_sample_information,
            worst_case_expected_value_of_sample_information,
            nominal_model_information_gain_nats: action.model_information_gain_nats,
            nominal_decision_information_gain_nats: action.decision_information_gain_nats,
            worst_case_outcome_probabilities: worst_case_distribution,
            unsupported_outcome_indices,
        });
    }

    Ok(TvRobustDecisionEvaluation {
        total_variation_radius,
        no_probe_bayes_risk: nominal.prior_bayes_risk,
        nominal_selected_action_index: nominal.selected_action_index,
        robust_selected_action_index,
        actions: robust_actions,
    })
}

/// Return an adversarial distribution q maximizing E_q[loss] subject to
/// TV(q, p) <= radius. This is exact for a finite categorical outcome space.
pub fn worst_case_distribution_within_tv(
    nominal_probabilities: &[f64],
    outcome_losses: &[f64],
    radius: f64,
) -> Result<Vec<f64>, TvRobustDecisionError> {
    if !radius.is_finite() || !(0.0..=1.0).contains(&radius) {
        return Err(TvRobustDecisionError::InvalidTotalVariationRadius);
    }
    if nominal_probabilities.is_empty() || nominal_probabilities.len() != outcome_losses.len() {
        return Err(TvRobustDecisionError::VectorLengthMismatch);
    }
    let mut probability_mass = 0.0;
    for (index, probability) in nominal_probabilities.iter().copied().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(TvRobustDecisionError::InvalidProbability { index });
        }
        probability_mass += probability;
    }
    if (probability_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(TvRobustDecisionError::ProbabilityMassNotNormalized {
            sum: probability_mass,
        });
    }
    for (index, loss) in outcome_losses.iter().copied().enumerate() {
        if !loss.is_finite() {
            return Err(TvRobustDecisionError::InvalidLoss { index });
        }
    }

    let mut worst_case = nominal_probabilities.to_vec();
    let mut donors: Vec<usize> = (0..outcome_losses.len()).collect();
    let mut recipients = donors.clone();
    donors.sort_by(|left, right| {
        outcome_losses[*left]
            .total_cmp(&outcome_losses[*right])
            .then_with(|| left.cmp(right))
    });
    recipients.sort_by(|left, right| {
        outcome_losses[*right]
            .total_cmp(&outcome_losses[*left])
            .then_with(|| left.cmp(right))
    });

    let mut donor_index = 0;
    let mut recipient_index = 0;
    let mut remaining_budget = radius;
    while remaining_budget > PROBABILITY_TOLERANCE {
        while donor_index < donors.len()
            && worst_case[donors[donor_index]] <= PROBABILITY_TOLERANCE
        {
            donor_index += 1;
        }
        while recipient_index < recipients.len()
            && worst_case[recipients[recipient_index]] >= 1.0 - PROBABILITY_TOLERANCE
        {
            recipient_index += 1;
        }
        if donor_index >= donors.len() || recipient_index >= recipients.len() {
            break;
        }
        let donor = donors[donor_index];
        let recipient = recipients[recipient_index];
        if donor == recipient || outcome_losses[recipient] <= outcome_losses[donor] {
            break;
        }
        let transferable = remaining_budget
            .min(worst_case[donor])
            .min(1.0 - worst_case[recipient]);
        if transferable <= PROBABILITY_TOLERANCE {
            break;
        }
        worst_case[donor] -= transferable;
        worst_case[recipient] += transferable;
        remaining_budget -= transferable;
    }

    let final_mass = worst_case.iter().sum::<f64>();
    if worst_case.iter().any(|probability| !probability.is_finite())
        || (final_mass - 1.0).abs() > PROBABILITY_TOLERANCE
    {
        return Err(TvRobustDecisionError::NonFiniteRisk);
    }
    Ok(worst_case)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> ModelAdequacyThresholds {
        ModelAdequacyThresholds {
            minimum_observations: 4,
            minimum_predictive_probability: 0.05,
            maximum_mean_log_loss: 1.0,
            maximum_mean_brier_score: 0.5,
            maximum_low_probability_rate: 0.25,
        }
    }

    fn binary_problem() -> (Vec<DecisionHypothesis>, Vec<DiagnosticAction>) {
        (
            vec![
                DecisionHypothesis {
                    id: "fault-a".into(),
                    prior_probability: 0.5,
                    outcome_probabilities_by_action: vec![vec![0.99, 0.01]],
                    loss_by_final_decision: vec![0.0, 1.0],
                },
                DecisionHypothesis {
                    id: "fault-b".into(),
                    prior_probability: 0.5,
                    outcome_probabilities_by_action: vec![vec![0.5, 0.5]],
                    loss_by_final_decision: vec![1.0, 0.0],
                },
            ],
            vec![DiagnosticAction {
                id: "inspect".into(),
                observation_cost: 0.23,
                safety_risk: 0.0,
            }],
        )
    }

    #[test]
    fn tv_ball_moves_probability_from_low_loss_to_high_loss_exactly() {
        let nominal = vec![0.8, 0.2];
        let losses = vec![0.0, 1.0];
        let worst = worst_case_distribution_within_tv(&nominal, &losses, 0.1)
            .expect("valid finite TV ball");
        assert!((worst[0] - 0.7).abs() < 1e-12);
        assert!((worst[1] - 0.3).abs() < 1e-12);
        let tv_distance = 0.5
            * worst
                .iter()
                .zip(&nominal)
                .map(|(left, right)| (left - right).abs())
                .sum::<f64>();
        assert!((tv_distance - 0.1).abs() < 1e-12);
    }

    #[test]
    fn zero_radius_matches_nominal_evsi_selection() {
        let (hypotheses, actions) = binary_problem();
        let nominal = evaluate_decision_focused_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
        )
        .expect("nominal diagnostics");
        let robust = evaluate_tv_robust_decision_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
            0.0,
        )
        .expect("zero-radius TV diagnostics");
        assert_eq!(robust.nominal_selected_action_index, nominal.selected_action_index);
        assert_eq!(robust.robust_selected_action_index, nominal.selected_action_index);
        assert!(
            (robust.actions[0].nominal_expected_posterior_bayes_risk
                - robust.actions[0].worst_case_expected_posterior_bayes_risk)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn nominal_probe_is_rejected_when_tv_shift_erases_its_decision_value() {
        let (hypotheses, actions) = binary_problem();
        let robust = evaluate_tv_robust_decision_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
            0.2,
        )
        .expect("robust diagnostics");
        assert_eq!(robust.nominal_selected_action_index, Some(0));
        assert_eq!(robust.robust_selected_action_index, None);
        assert!(
            robust.actions[0].worst_case_total_score.unwrap()
                > robust.no_probe_bayes_risk
        );
    }

    #[test]
    fn unsupported_outcomes_receive_the_declared_maximum_loss_bound() {
        let (mut hypotheses, actions) = binary_problem();
        hypotheses[0].outcome_probabilities_by_action[0] = vec![1.0, 0.0];
        hypotheses[1].outcome_probabilities_by_action[0] = vec![1.0, 0.0];
        let robust = evaluate_tv_robust_decision_diagnostics(
            &hypotheses,
            &actions,
            DecisionDiagnosticConfig::default(),
            0.25,
        )
        .expect("robust diagnostics with an open-support outcome");
        assert_eq!(robust.actions[0].unsupported_outcome_indices, vec![1]);
        assert!(robust.actions[0].worst_case_outcome_probabilities[1] > 0.0);
        assert!(robust.actions[0].worst_case_expected_posterior_bayes_risk > 0.0);
    }

    #[test]
    fn adequacy_gate_detects_out_of_support_observations() {
        let observations = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![1.0, 0.0],
            observed_outcome_index: 1,
        }];
        let report = evaluate_prequential_model_adequacy(&observations, thresholds())
            .expect("valid prequential record");
        assert_eq!(report.status, ModelAdequacyStatus::Inadequate);
        assert_eq!(report.zero_support_count, 1);
        assert_eq!(report.mean_log_loss, None);
    }

    #[test]
    fn adequacy_gate_requires_minimum_evidence_and_never_claims_proof() {
        let observations = (0..4)
            .map(|_| PrequentialCategoricalObservation {
                predicted_outcome_probabilities: vec![0.8, 0.2],
                observed_outcome_index: 0,
            })
            .collect::<Vec<_>>();
        let report = evaluate_prequential_model_adequacy(&observations, thresholds())
            .expect("valid prequential records");
        assert_eq!(report.status, ModelAdequacyStatus::NoDetectedInadequacy);
        assert_eq!(report.observation_count, 4);
        assert!(report.mean_log_loss.unwrap() < 1.0);
        assert!(report.mean_multiclass_brier_score.unwrap() < 0.5);
    }

    #[test]
    fn adequacy_gate_abstains_when_evidence_is_insufficient() {
        let observations = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![0.8, 0.2],
            observed_outcome_index: 0,
        }];
        let report = evaluate_prequential_model_adequacy(&observations, thresholds())
            .expect("valid prequential record");
        assert_eq!(report.status, ModelAdequacyStatus::InsufficientEvidence);
    }

    #[test]
    fn malformed_predictive_mass_fails_closed() {
        let observations = vec![PrequentialCategoricalObservation {
            predicted_outcome_probabilities: vec![0.8, 0.3],
            observed_outcome_index: 0,
        }];
        assert!(matches!(
            evaluate_prequential_model_adequacy(&observations, thresholds()),
            Err(ModelAdequacyError::ProbabilityMassNotNormalized { .. })
        ));
    }

    #[test]
    fn invalid_tv_radius_fails_closed() {
        assert!(matches!(
            worst_case_distribution_within_tv(&[0.5, 0.5], &[0.0, 1.0], 1.1),
            Err(TvRobustDecisionError::InvalidTotalVariationRadius)
        ));
    }
}
