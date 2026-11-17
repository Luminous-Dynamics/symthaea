// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Standalone finite-hypothesis experimental-design reference.
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/noise_aware_eig.rs -o /tmp/isru-eig-tests
//   /tmp/isru-eig-tests

#![forbid(unsafe_code)]

//! Likelihood-based expected information gain (EIG) and expected value of
//! sample information (EVSI) for discrete experimental actions.
//!
//! This is deliberately separate from Symthaea's prediction-variance selector.
//! Every action supplies a complete p(outcome | hypothesis, action) likelihood
//! matrix. Measurement noise, censoring, and failure modes must be represented
//! in those likelihoods by the caller; they are not inferred from prediction spread.
//!
//! The finite hypotheses may represent joint explanations such as
//! (material regime, optical contamination state, metrology state). This file
//! does not assert that those factors are independent or implement temporal
//! state transitions. Shared causes must be encoded in the joint hypotheses
//! and/or likelihood construction, not counted as independent sensors.
//!
//! EIG is reported in nats. EVSI is measured in caller-supplied decision-loss
//! units. Cost is an explicitly declared scalar cost unit; actions with
//! incompatible cost definitions must not be compared.

use std::error::Error;
use std::fmt;

/// A candidate measurement or intervention and its outcome likelihood model.
///
/// likelihoods[h][y] is p(outcome y | hypothesis h, action).
/// Each row must be a finite probability distribution summing to one.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteAction {
    pub id: String,
    pub cost_units: f64,
    pub likelihoods: Vec<Vec<f64>>,
}

/// Validation errors are returned rather than repaired by implicit
/// normalization, row truncation, or replacement of missing probabilities.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelError {
    TooFewHypotheses,
    InvalidPriorEntry(usize),
    PriorNotNormalized(f64),
    EmptyActionId,
    InvalidCost,
    LikelihoodRowCount { expected: usize, actual: usize },
    EmptyOutcomeSpace,
    RaggedLikelihoodMatrix,
    InvalidLikelihoodEntry { hypothesis: usize, outcome: usize },
    LikelihoodNotNormalized { hypothesis: usize, sum: f64 },
    LossRowCount { expected: usize, actual: usize },
    EmptyDecisionSet,
    RaggedLossMatrix,
    InvalidLossEntry { hypothesis: usize, decision: usize },
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for ModelError {}

/// Decision-value and information metrics for one validated candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionEvaluation {
    pub action_id: String,
    pub cost_units: f64,
    pub expected_information_gain_nats: f64,
    pub expected_value_of_sample_information: f64,
    pub prior_bayes_risk: f64,
    pub expected_posterior_bayes_risk: f64,
    pub information_gain_per_cost: f64,
    pub decision_value_per_cost: f64,
}

/// Explicit choice of what "best next action" means.
///
/// Information gain and decision value are intentionally separate: a
/// measurement can reveal which hypothesis is true without changing the
/// engineering decision, while a low-entropy measurement can be decisive
/// for a high-consequence decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionObjective {
    InformationGain,
    InformationGainPerCost,
    DecisionValue,
    DecisionValuePerCost,
}

const NORMALIZATION_TOLERANCE: f64 = 1.0e-9;

fn validate_prior(prior: &[f64]) -> Result<(), ModelError> {
    if prior.len() < 2 {
        return Err(ModelError::TooFewHypotheses);
    }

    let mut sum = 0.0;
    for (index, probability) in prior.iter().copied().enumerate() {
        if !probability.is_finite() || probability < 0.0 {
            return Err(ModelError::InvalidPriorEntry(index));
        }
        sum += probability;
    }

    if (sum - 1.0).abs() > NORMALIZATION_TOLERANCE {
        return Err(ModelError::PriorNotNormalized(sum));
    }
    Ok(())
}

fn validate_action(hypothesis_count: usize, action: &DiscreteAction) -> Result<usize, ModelError> {
    if action.id.trim().is_empty() {
        return Err(ModelError::EmptyActionId);
    }
    if !action.cost_units.is_finite() || action.cost_units <= 0.0 {
        return Err(ModelError::InvalidCost);
    }
    if action.likelihoods.len() != hypothesis_count {
        return Err(ModelError::LikelihoodRowCount {
            expected: hypothesis_count,
            actual: action.likelihoods.len(),
        });
    }

    let outcome_count = action.likelihoods.first().map(Vec::len).unwrap_or(0);
    if outcome_count == 0 {
        return Err(ModelError::EmptyOutcomeSpace);
    }

    for (hypothesis, row) in action.likelihoods.iter().enumerate() {
        if row.len() != outcome_count {
            return Err(ModelError::RaggedLikelihoodMatrix);
        }

        let mut sum = 0.0;
        for (outcome, probability) in row.iter().copied().enumerate() {
            if !probability.is_finite() || probability < 0.0 {
                return Err(ModelError::InvalidLikelihoodEntry {
                    hypothesis,
                    outcome,
                });
            }
            sum += probability;
        }

        if (sum - 1.0).abs() > NORMALIZATION_TOLERANCE {
            return Err(ModelError::LikelihoodNotNormalized { hypothesis, sum });
        }
    }

    Ok(outcome_count)
}

fn validate_loss_matrix(hypothesis_count: usize, losses: &[Vec<f64>]) -> Result<usize, ModelError> {
    if losses.len() != hypothesis_count {
        return Err(ModelError::LossRowCount {
            expected: hypothesis_count,
            actual: losses.len(),
        });
    }

    let decision_count = losses.first().map(Vec::len).unwrap_or(0);
    if decision_count == 0 {
        return Err(ModelError::EmptyDecisionSet);
    }

    for (hypothesis, row) in losses.iter().enumerate() {
        if row.len() != decision_count {
            return Err(ModelError::RaggedLossMatrix);
        }
        for (decision, loss) in row.iter().copied().enumerate() {
            if !loss.is_finite() || loss < 0.0 {
                return Err(ModelError::InvalidLossEntry {
                    hypothesis,
                    decision,
                });
            }
        }
    }

    Ok(decision_count)
}

fn entropy_nats(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn bayes_risk(probabilities: &[f64], losses: &[Vec<f64>], decision_count: usize) -> f64 {
    (0..decision_count)
        .map(|decision| {
            probabilities
                .iter()
                .enumerate()
                .map(|(hypothesis, probability)| probability * losses[hypothesis][decision])
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Evaluate one action with the supplied prior, likelihood model, and loss table.
///
/// losses[h][d] is the loss incurred by choosing decision d when hypothesis h
/// is true. Priors and likelihood rows are validated but never silently
/// normalized. An outcome with zero predictive probability is skipped; it
/// cannot create a posterior or contribute fictitious value.
pub fn evaluate_action(
    prior: &[f64],
    action: &DiscreteAction,
    losses: &[Vec<f64>],
) -> Result<ActionEvaluation, ModelError> {
    validate_prior(prior)?;
    let outcome_count = validate_action(prior.len(), action)?;
    let decision_count = validate_loss_matrix(prior.len(), losses)?;

    let mut predictive = vec![0.0; outcome_count];
    for (hypothesis, prior_probability) in prior.iter().copied().enumerate() {
        for (outcome, likelihood) in action.likelihoods[hypothesis].iter().copied().enumerate() {
            predictive[outcome] += prior_probability * likelihood;
        }
    }

    let prior_entropy = entropy_nats(prior);
    let mut expected_posterior_entropy = 0.0;
    let prior_risk = bayes_risk(prior, losses, decision_count);
    let mut expected_posterior_risk = 0.0;

    for (outcome, outcome_probability) in predictive.iter().copied().enumerate() {
        if outcome_probability <= 0.0 {
            continue;
        }

        let posterior: Vec<f64> = prior
            .iter()
            .enumerate()
            .map(|(hypothesis, prior_probability)| {
                prior_probability * action.likelihoods[hypothesis][outcome] / outcome_probability
            })
            .collect();

        expected_posterior_entropy += outcome_probability * entropy_nats(&posterior);
        expected_posterior_risk +=
            outcome_probability * bayes_risk(&posterior, losses, decision_count);
    }

    // Bayesian information and optimal decision value are non-negative in
    // exact arithmetic. Clamp floating-point undershoots at zero.
    let information_gain = (prior_entropy - expected_posterior_entropy).max(0.0);
    let decision_value = (prior_risk - expected_posterior_risk).max(0.0);

    Ok(ActionEvaluation {
        action_id: action.id.clone(),
        cost_units: action.cost_units,
        expected_information_gain_nats: information_gain,
        expected_value_of_sample_information: decision_value,
        prior_bayes_risk: prior_risk,
        expected_posterior_bayes_risk: expected_posterior_risk,
        information_gain_per_cost: information_gain / action.cost_units,
        decision_value_per_cost: decision_value / action.cost_units,
    })
}

/// Select the first action attaining the largest requested score.
///
/// Evaluations must use the same prior, decision-loss table, evidence snapshot,
/// and comparable cost definition. Ties preserve input order for replay.
pub fn select_best_action(
    evaluations: &[ActionEvaluation],
    objective: ActionObjective,
) -> Option<&ActionEvaluation> {
    let mut best: Option<&ActionEvaluation> = None;
    let mut best_score = f64::NEG_INFINITY;

    for evaluation in evaluations {
        let score = match objective {
            ActionObjective::InformationGain => evaluation.expected_information_gain_nats,
            ActionObjective::InformationGainPerCost => evaluation.information_gain_per_cost,
            ActionObjective::DecisionValue => evaluation.expected_value_of_sample_information,
            ActionObjective::DecisionValuePerCost => evaluation.decision_value_per_cost,
        };

        if score > best_score {
            best = Some(evaluation);
            best_score = score;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_action(id: &str, cost_units: f64) -> DiscreteAction {
        DiscreteAction {
            id: id.to_string(),
            cost_units,
            likelihoods: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
        }
    }

    fn binary_losses() -> Vec<Vec<f64>> {
        vec![vec![0.0, 1.0], vec![1.0, 0.0]]
    }

    #[test]
    fn symmetric_binary_channel_matches_analytic_eig_and_evsi() {
        let evaluation = evaluate_action(
            &[0.5, 0.5],
            &binary_action("reference_measurement", 1.0),
            &binary_losses(),
        )
        .unwrap();

        let expected_eig = std::f64::consts::LN_2 + 0.9 * 0.9_f64.ln() + 0.1 * 0.1_f64.ln();
        assert!((evaluation.expected_information_gain_nats - expected_eig).abs() < 1.0e-12);
        assert!((evaluation.expected_value_of_sample_information - 0.4).abs() < 1.0e-12);
        assert!((evaluation.prior_bayes_risk - 0.5).abs() < 1.0e-12);
        assert!((evaluation.expected_posterior_bayes_risk - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn identical_likelihood_rows_have_zero_information_and_decision_value() {
        let action = DiscreteAction {
            id: "uninformative_check".to_string(),
            cost_units: 2.0,
            likelihoods: vec![vec![0.25, 0.75], vec![0.25, 0.75]],
        };
        let evaluation = evaluate_action(&[0.5, 0.5], &action, &binary_losses()).unwrap();

        assert!(evaluation.expected_information_gain_nats.abs() < 1.0e-12);
        assert!(evaluation.expected_value_of_sample_information.abs() < 1.0e-12);
    }

    #[test]
    fn decision_value_can_reverse_the_information_gain_ranking() {
        let prior = [0.49, 0.49, 0.02];
        let losses = vec![vec![0.0, 10.0], vec![0.0, 10.0], vec![100.0, 0.0]];
        let split_irrelevant_hypotheses = DiscreteAction {
            id: "distinguish_h0_from_h1".to_string(),
            cost_units: 1.0,
            likelihoods: vec![vec![0.1, 0.9], vec![0.9, 0.1], vec![0.5, 0.5]],
        };
        let detect_decision_critical_state = DiscreteAction {
            id: "detect_h2".to_string(),
            cost_units: 1.0,
            likelihoods: vec![vec![0.99, 0.01], vec![0.99, 0.01], vec![0.01, 0.99]],
        };

        let first = evaluate_action(&prior, &split_irrelevant_hypotheses, &losses).unwrap();
        let second = evaluate_action(&prior, &detect_decision_critical_state, &losses).unwrap();

        assert!(first.expected_information_gain_nats > second.expected_information_gain_nats);
        assert!(first.expected_value_of_sample_information < 1.0e-12);
        assert!(second.expected_value_of_sample_information > 1.0);
        assert_eq!(
            select_best_action(
                &[first.clone(), second.clone()],
                ActionObjective::InformationGain
            )
            .unwrap()
            .action_id,
            "distinguish_h0_from_h1"
        );
        assert_eq!(
            select_best_action(&[first, second], ActionObjective::DecisionValue)
                .unwrap()
                .action_id,
            "detect_h2"
        );
    }

    #[test]
    fn cost_normalization_changes_the_preferred_action() {
        let prior = [0.5, 0.5];
        let losses = binary_losses();
        let cheap = evaluate_action(&prior, &binary_action("cheap", 1.0), &losses).unwrap();
        let expensive = evaluate_action(&prior, &binary_action("expensive", 4.0), &losses).unwrap();

        assert_eq!(
            select_best_action(
                &[expensive.clone(), cheap.clone()],
                ActionObjective::InformationGainPerCost
            )
            .unwrap()
            .action_id,
            "cheap"
        );
        assert_eq!(
            select_best_action(&[cheap, expensive], ActionObjective::DecisionValuePerCost)
                .unwrap()
                .action_id,
            "cheap"
        );
    }

    #[test]
    fn zero_predictive_probability_outcomes_are_skipped() {
        let action = DiscreteAction {
            id: "one_supported_state".to_string(),
            cost_units: 1.0,
            likelihoods: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let evaluation = evaluate_action(&[1.0, 0.0], &action, &binary_losses()).unwrap();

        assert!(evaluation.expected_information_gain_nats.abs() < 1.0e-12);
        assert!(evaluation.expected_value_of_sample_information.abs() < 1.0e-12);
    }

    #[test]
    fn invalid_prior_is_rejected_without_renormalization() {
        let result = evaluate_action(
            &[0.4, 0.4],
            &binary_action("invalid_prior", 1.0),
            &binary_losses(),
        );
        assert!(matches!(result, Err(ModelError::PriorNotNormalized(_))));
    }

    #[test]
    fn invalid_likelihood_row_is_rejected_without_renormalization() {
        let action = DiscreteAction {
            id: "invalid_likelihood".to_string(),
            cost_units: 1.0,
            likelihoods: vec![vec![0.8, 0.3], vec![0.1, 0.9]],
        };
        let result = evaluate_action(&[0.5, 0.5], &action, &binary_losses());
        assert!(matches!(
            result,
            Err(ModelError::LikelihoodNotNormalized { hypothesis: 0, .. })
        ));
    }

    #[test]
    fn malformed_dimensions_and_non_finite_cost_are_rejected() {
        let mut action = binary_action("bad_shape", 1.0);
        action.likelihoods.pop();
        assert!(matches!(
            evaluate_action(&[0.5, 0.5], &action, &binary_losses()),
            Err(ModelError::LikelihoodRowCount { .. })
        ));

        let action = binary_action("bad_cost", f64::INFINITY);
        assert_eq!(
            evaluate_action(&[0.5, 0.5], &action, &binary_losses()),
            Err(ModelError::InvalidCost)
        );
    }

    #[test]
    fn empty_and_ragged_loss_tables_are_rejected() {
        let action = binary_action("valid", 1.0);
        assert_eq!(
            evaluate_action(&[0.5, 0.5], &action, &[]),
            Err(ModelError::LossRowCount {
                expected: 2,
                actual: 0
            })
        );
        assert_eq!(
            evaluate_action(&[0.5, 0.5], &action, &[vec![0.0, 1.0], vec![0.0]]),
            Err(ModelError::RaggedLossMatrix)
        );
    }

    #[test]
    fn stable_ties_preserve_candidate_order() {
        let prior = [0.5, 0.5];
        let losses = binary_losses();
        let first = evaluate_action(&prior, &binary_action("first", 1.0), &losses).unwrap();
        let second = evaluate_action(&prior, &binary_action("second", 1.0), &losses).unwrap();

        assert_eq!(
            select_best_action(&[first, second], ActionObjective::InformationGain)
                .unwrap()
                .action_id,
            "first"
        );
    }
}
