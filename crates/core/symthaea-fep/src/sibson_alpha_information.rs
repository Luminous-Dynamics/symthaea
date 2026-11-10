// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Sibson alpha-mutual information for robust finite-hypothesis experiment selection.
//!
//! This module implements a finite, categorical form of the robust expected
//! information-gain criterion used in maximin Bayesian experimental design.
//! For alpha in (0, 1), the Sibson information radius is less sensitive to
//! optimistic likelihood distinctions than ordinary Shannon mutual information.
//!
//! The corresponding alpha-tilted posterior is proportional to
//! prior(theta) * likelihood(outcome | theta, action)^alpha.
//! At alpha = 1, the implementation returns ordinary mutual information and
//! the ordinary Bayesian posterior.
//!
//! This is an information-only criterion. It does not establish that the
//! likelihoods are causally identified, choose task losses, account for action
//! cost/safety, or authorize a physical probe.

use serde::{Deserialize, Serialize};
use thiserror::Error;

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const INFORMATION_TOLERANCE: f64 = 1e-10;
const ALPHA_LIMIT_TOLERANCE: f64 = 1e-7;

/// A finite hypothesis and its action-conditioned categorical likelihoods.
///
/// outcome_probabilities_by_action[action][outcome] is the nominal
/// likelihood P(outcome | hypothesis, action). Hypothesis identifiers are
/// unique and priors are normalized across the supplied set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SibsonInformationHypothesis {
    pub id: String,
    pub prior_probability: f64,
    pub outcome_probabilities_by_action: Vec<Vec<f64>>,
}

/// The robust-information evaluation for one candidate experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SibsonActionInformation {
    pub action_index: usize,
    /// Nominal predictive P(outcome | action), under the supplied prior.
    pub nominal_outcome_probabilities: Vec<f64>,
    /// Ordinary Shannon mutual information I(Theta; O | action), in nats.
    pub shannon_information_nats: f64,
    /// Sibson alpha-mutual information, in nats.
    pub sibson_information_nats: f64,
    /// The robust posterior p_alpha(theta | outcome, action).
    /// None denotes an outcome with zero nominal predictive probability.
    pub alpha_tilted_posterior_by_outcome: Vec<Option<Vec<f64>>>,
}

/// Evaluation across every available experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SibsonInformationEvaluation {
    pub alpha: f64,
    pub actions: Vec<SibsonActionInformation>,
    /// The action maximizing Sibson information. Ties go to the lower index.
    pub most_robustly_informative_action: usize,
}

/// Invalid finite-hypothesis or alpha-information input.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SibsonInformationError {
    #[error("at least one hypothesis is required")]
    EmptyHypotheses,
    #[error("hypothesis {hypothesis_index} has an empty identifier")]
    EmptyHypothesisId { hypothesis_index: usize },
    #[error("duplicate hypothesis identifier: {0}")]
    DuplicateHypothesisId(String),
    #[error("hypothesis prior {hypothesis_index} is invalid")]
    InvalidPrior { hypothesis_index: usize },
    #[error("hypothesis prior mass sums to {sum}, expected 1")]
    PriorMassNotNormalized { sum: f64 },
    #[error("at least one candidate action is required")]
    EmptyActionSet,
    #[error("hypothesis {hypothesis_index} has {actual} actions, expected {expected}")]
    InconsistentActionCount {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("action {action_index} has no possible outcomes")]
    EmptyOutcomeSet { action_index: usize },
    #[error("hypothesis {hypothesis_index}, action {action_index} has {actual} outcomes, expected {expected}")]
    InconsistentOutcomeCount {
        hypothesis_index: usize,
        action_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("hypothesis {hypothesis_index}, action {action_index}, outcome {outcome_index} has an invalid likelihood")]
    InvalidLikelihood {
        hypothesis_index: usize,
        action_index: usize,
        outcome_index: usize,
    },
    #[error("hypothesis {hypothesis_index}, action {action_index} likelihood mass sums to {sum}, expected 1")]
    LikelihoodMassNotNormalized {
        hypothesis_index: usize,
        action_index: usize,
        sum: f64,
    },
    #[error("alpha must be finite and in (0, 1]")]
    InvalidAlpha,
    #[error("action index {action_index} is outside 0..{action_count}")]
    ActionIndexOutOfRange {
        action_index: usize,
        action_count: usize,
    },
    #[error("outcome index {outcome_index} is outside 0..{outcome_count} for action {action_index}")]
    OutcomeIndexOutOfRange {
        action_index: usize,
        outcome_index: usize,
        outcome_count: usize,
    },
    #[error("the selected outcome has zero nominal predictive probability")]
    ImpossibleOutcome,
    #[error("information calculation produced a non-finite value")]
    NonFiniteInformation,
    #[error("Sibson information was materially negative: {0}")]
    NegativeSibsonInformation(f64),
}

/// Evaluate Shannon EIG and Sibson alpha-information for every action.
///
/// For alpha in (0, 1), the finite categorical expression is
///
/// I^S_alpha(Theta; O | a)
///   = alpha/(alpha - 1) * log sum_o
///       (sum_theta p(theta) p(o | theta, a)^alpha)^(1/alpha).
///
/// Alpha = 1 is evaluated through its Shannon limit to avoid cancellation.
/// The robust posterior for a possible outcome is proportional to
/// p(theta) * p(outcome | theta, action)^alpha.
pub fn evaluate_sibson_alpha_information(
    hypotheses: &[SibsonInformationHypothesis],
    alpha: f64,
) -> Result<SibsonInformationEvaluation, SibsonInformationError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
        return Err(SibsonInformationError::InvalidAlpha);
    }
    let (action_count, outcome_count) = validate_hypotheses(hypotheses)?;
    let mut actions = Vec::with_capacity(action_count);

    for action_index in 0..action_count {
        let mut nominal_outcome_probabilities = vec![0.0; outcome_count];
        for hypothesis in hypotheses {
            for outcome_index in 0..outcome_count {
                nominal_outcome_probabilities[outcome_index] +=
                    hypothesis.prior_probability
                        * hypothesis.outcome_probabilities_by_action[action_index][outcome_index];
            }
        }

        let mut shannon_information_nats = 0.0;
        for hypothesis in hypotheses {
            if hypothesis.prior_probability == 0.0 {
                continue;
            }
            for outcome_index in 0..outcome_count {
                let likelihood =
                    hypothesis.outcome_probabilities_by_action[action_index][outcome_index];
                let marginal = nominal_outcome_probabilities[outcome_index];
                if likelihood > 0.0 && marginal > 0.0 {
                    shannon_information_nats += hypothesis.prior_probability
                        * likelihood
                        * (likelihood / marginal).ln();
                }
            }
        }
        if !shannon_information_nats.is_finite() {
            return Err(SibsonInformationError::NonFiniteInformation);
        }
        if shannon_information_nats < -INFORMATION_TOLERANCE {
            return Err(SibsonInformationError::NonFiniteInformation);
        }
        shannon_information_nats = shannon_information_nats.max(0.0);

        let mut alpha_tilted_posterior_by_outcome = Vec::with_capacity(outcome_count);
        let mut log_power_mass_by_outcome = Vec::with_capacity(outcome_count);
        for outcome_index in 0..outcome_count {
            let mut log_terms = Vec::with_capacity(hypotheses.len());
            for hypothesis in hypotheses {
                let prior = hypothesis.prior_probability;
                let likelihood =
                    hypothesis.outcome_probabilities_by_action[action_index][outcome_index];
                let term = if prior == 0.0 || likelihood == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    prior.ln() + alpha * likelihood.ln()
                };
                log_terms.push(term);
            }
            let log_normalizer = log_sum_exp(&log_terms);
            log_power_mass_by_outcome.push(log_normalizer);

            if nominal_outcome_probabilities[outcome_index] == 0.0 {
                alpha_tilted_posterior_by_outcome.push(None);
            } else {
                let posterior = normalize_log_weights(&log_terms)?;
                alpha_tilted_posterior_by_outcome.push(Some(posterior));
            }
        }

        let sibson_information_nats = if 1.0 - alpha <= ALPHA_LIMIT_TOLERANCE {
            // Continuous limit alpha -> 1 is Shannon mutual information.
            shannon_information_nats
        } else {
            let outer_terms: Vec<f64> = log_power_mass_by_outcome
                .iter()
                .copied()
                .map(|log_mass| log_mass / alpha)
                .collect();
            let log_radius = log_sum_exp(&outer_terms);
            if !log_radius.is_finite() {
                return Err(SibsonInformationError::NonFiniteInformation);
            }
            let value = alpha / (alpha - 1.0) * log_radius;
            if !value.is_finite() {
                return Err(SibsonInformationError::NonFiniteInformation);
            }
            if value < -INFORMATION_TOLERANCE {
                return Err(SibsonInformationError::NegativeSibsonInformation(value));
            }
            value.max(0.0)
        };

        actions.push(SibsonActionInformation {
            action_index,
            nominal_outcome_probabilities,
            shannon_information_nats,
            sibson_information_nats,
            alpha_tilted_posterior_by_outcome,
        });
    }

    let mut best_index = 0;
    for candidate in actions.iter().skip(1) {
        if candidate.sibson_information_nats
            > actions[best_index].sibson_information_nats
        {
            best_index = candidate.action_index;
        }
    }

    Ok(SibsonInformationEvaluation {
        alpha,
        actions,
        most_robustly_informative_action: best_index,
    })
}

/// Apply the alpha-tilted posterior for one observed action/outcome pair.
pub fn sibson_tilted_posterior(
    hypotheses: &[SibsonInformationHypothesis],
    alpha: f64,
    action_index: usize,
    outcome_index: usize,
) -> Result<Vec<f64>, SibsonInformationError> {
    let evaluation = evaluate_sibson_alpha_information(hypotheses, alpha)?;
    let action = evaluation.actions.get(action_index).ok_or(
        SibsonInformationError::ActionIndexOutOfRange {
            action_index,
            action_count: evaluation.actions.len(),
        },
    )?;
    let outcome_count = action.alpha_tilted_posterior_by_outcome.len();
    if outcome_index >= outcome_count {
        return Err(SibsonInformationError::OutcomeIndexOutOfRange {
            action_index,
            outcome_index,
            outcome_count,
        });
    }
    action.alpha_tilted_posterior_by_outcome[outcome_index]
        .clone()
        .ok_or(SibsonInformationError::ImpossibleOutcome)
}

fn validate_hypotheses(
    hypotheses: &[SibsonInformationHypothesis],
) -> Result<(usize, usize), SibsonInformationError> {
    if hypotheses.is_empty() {
        return Err(SibsonInformationError::EmptyHypotheses);
    }

    let mut identifiers = std::collections::BTreeSet::new();
    let mut prior_sum = 0.0;
    for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
        if hypothesis.id.trim().is_empty() {
            return Err(SibsonInformationError::EmptyHypothesisId { hypothesis_index });
        }
        if !identifiers.insert(hypothesis.id.clone()) {
            return Err(SibsonInformationError::DuplicateHypothesisId(
                hypothesis.id.clone(),
            ));
        }
        let prior = hypothesis.prior_probability;
        if !prior.is_finite() || !(0.0..=1.0).contains(&prior) {
            return Err(SibsonInformationError::InvalidPrior { hypothesis_index });
        }
        prior_sum += prior;
    }
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(SibsonInformationError::PriorMassNotNormalized { sum: prior_sum });
    }

    let action_count = hypotheses[0].outcome_probabilities_by_action.len();
    if action_count == 0 {
        return Err(SibsonInformationError::EmptyActionSet);
    }
    let outcome_count = hypotheses[0].outcome_probabilities_by_action[0].len();
    for action_index in 0..action_count {
        if hypotheses[0].outcome_probabilities_by_action[action_index].is_empty() {
            return Err(SibsonInformationError::EmptyOutcomeSet { action_index });
        }
    }

    for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
        if hypothesis.outcome_probabilities_by_action.len() != action_count {
            return Err(SibsonInformationError::InconsistentActionCount {
                hypothesis_index,
                expected: action_count,
                actual: hypothesis.outcome_probabilities_by_action.len(),
            });
        }
        for action_index in 0..action_count {
            let outcomes = &hypothesis.outcome_probabilities_by_action[action_index];
            if outcomes.len() != outcome_count {
                return Err(SibsonInformationError::InconsistentOutcomeCount {
                    hypothesis_index,
                    action_index,
                    expected: outcome_count,
                    actual: outcomes.len(),
                });
            }
            let mut sum = 0.0;
            for (outcome_index, probability) in outcomes.iter().copied().enumerate() {
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err(SibsonInformationError::InvalidLikelihood {
                        hypothesis_index,
                        action_index,
                        outcome_index,
                    });
                }
                sum += probability;
            }
            if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(SibsonInformationError::LikelihoodMassNotNormalized {
                    hypothesis_index,
                    action_index,
                    sum,
                });
            }
        }
    }

    Ok((action_count, outcome_count))
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    maximum
        + values
            .iter()
            .copied()
            .filter(|value| *value != f64::NEG_INFINITY)
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn normalize_log_weights(log_weights: &[f64]) -> Result<Vec<f64>, SibsonInformationError> {
    let normalizer = log_sum_exp(log_weights);
    if !normalizer.is_finite() {
        return Err(SibsonInformationError::NonFiniteInformation);
    }
    let probabilities: Vec<f64> = log_weights
        .iter()
        .copied()
        .map(|value| {
            if value == f64::NEG_INFINITY {
                0.0
            } else {
                (value - normalizer).exp()
            }
        })
        .collect();
    let sum = probabilities.iter().sum::<f64>();
    if !sum.is_finite() || (sum - 1.0).abs() > 1e-8 {
        return Err(SibsonInformationError::NonFiniteInformation);
    }
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_hypotheses(
        first_action: Vec<Vec<f64>>,
        second_action: Vec<Vec<f64>>,
    ) -> Vec<SibsonInformationHypothesis> {
        vec![
            SibsonInformationHypothesis {
                id: "model-a".into(),
                prior_probability: 0.5,
                outcome_probabilities_by_action: vec![
                    first_action[0].clone(),
                    second_action[0].clone(),
                ],
            },
            SibsonInformationHypothesis {
                id: "model-b".into(),
                prior_probability: 0.5,
                outcome_probabilities_by_action: vec![
                    first_action[1].clone(),
                    second_action[1].clone(),
                ],
            },
        ]
    }

    #[test]
    fn identical_likelihoods_have_zero_information_for_all_alpha() {
        let hypotheses = two_hypotheses(
            vec![vec![0.8, 0.2], vec![0.8, 0.2]],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        );
        for alpha in [0.1, 0.5, 0.9, 1.0] {
            let evaluation =
                evaluate_sibson_alpha_information(&hypotheses, alpha).expect("valid model");
            assert!(evaluation.actions.iter().all(|action| {
                action.sibson_information_nats.abs() < 1e-9
                    && action.shannon_information_nats.abs() < 1e-9
            }));
        }
    }

    #[test]
    fn perfect_discrimination_recovers_prior_entropy() {
        let hypotheses = two_hypotheses(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0.7, 0.3], vec![0.3, 0.7]],
        );
        for alpha in [0.1, 0.25, 0.5, 0.9, 1.0] {
            let evaluation =
                evaluate_sibson_alpha_information(&hypotheses, alpha).expect("valid model");
            assert!(
                (evaluation.actions[0].sibson_information_nats - std::f64::consts::LN_2).abs()
                    < 1e-8
            );
        }
    }

    #[test]
    fn alpha_one_matches_shannon_information_and_bayes_posterior() {
        let hypotheses = two_hypotheses(
            vec![vec![0.9, 0.1], vec![0.2, 0.8]],
            vec![vec![0.65, 0.35], vec![0.35, 0.65]],
        );
        let evaluation =
            evaluate_sibson_alpha_information(&hypotheses, 1.0).expect("valid model");
        for action in &evaluation.actions {
            assert!(
                (action.sibson_information_nats - action.shannon_information_nats).abs()
                    < 1e-12
            );
            for (outcome_index, posterior) in
                action.alpha_tilted_posterior_by_outcome.iter().enumerate()
            {
                let posterior = posterior.as_ref().expect("positive support");
                let marginal = action.nominal_outcome_probabilities[outcome_index];
                let expected: Vec<f64> = hypotheses
                    .iter()
                    .map(|hypothesis| {
                        hypothesis.prior_probability
                            * hypothesis.outcome_probabilities_by_action[action.action_index]
                                [outcome_index]
                            / marginal
                    })
                    .collect();
                for (actual, expected) in posterior.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn alpha_tilted_posterior_is_less_overconfident_than_nominal_bayes_update() {
        let hypotheses = two_hypotheses(
            vec![vec![0.9, 0.1], vec![0.2, 0.8]],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        );
        let robust =
            sibson_tilted_posterior(&hypotheses, 0.5, 0, 0).expect("possible outcome");
        let nominal = sibson_tilted_posterior(&hypotheses, 1.0, 0, 0).expect("possible outcome");
        assert!(robust[0] < nominal[0]);
        assert!((robust.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((nominal.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn robust_information_does_not_exceed_shannon_information() {
        let hypotheses = two_hypotheses(
            vec![vec![0.9, 0.1], vec![0.2, 0.8]],
            vec![vec![0.75, 0.25], vec![0.25, 0.75]],
        );
        let nominal =
            evaluate_sibson_alpha_information(&hypotheses, 1.0).expect("nominal information");
        for alpha in [0.1, 0.25, 0.5, 0.75, 0.95] {
            let robust =
                evaluate_sibson_alpha_information(&hypotheses, alpha).expect("robust information");
            for (robust_action, nominal_action) in robust.actions.iter().zip(&nominal.actions) {
                assert!(
                    robust_action.sibson_information_nats
                        <= nominal_action.shannon_information_nats + 1e-8
                );
            }
        }
    }

    #[test]
    fn robust_selector_chooses_the_largest_sibson_score() {
        let hypotheses = two_hypotheses(
            vec![vec![0.9, 0.1], vec![0.2, 0.8]],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        );
        let evaluation =
            evaluate_sibson_alpha_information(&hypotheses, 0.5).expect("robust information");
        assert_eq!(evaluation.most_robustly_informative_action, 0);
        assert!(
            evaluation.actions[0].sibson_information_nats
                > evaluation.actions[1].sibson_information_nats
        );
    }

    #[test]
    fn impossible_outcome_has_no_alpha_tilted_posterior() {
        let hypotheses = two_hypotheses(
            vec![vec![1.0, 0.0], vec![1.0, 0.0]],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        );
        assert!(matches!(
            sibson_tilted_posterior(&hypotheses, 0.5, 0, 1),
            Err(SibsonInformationError::ImpossibleOutcome)
        ));
    }

    #[test]
    fn malformed_probability_mass_and_alpha_fail_closed() {
        let mut hypotheses = two_hypotheses(
            vec![vec![0.8, 0.2], vec![0.2, 0.8]],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        );
        assert!(matches!(
            evaluate_sibson_alpha_information(&hypotheses, 0.0),
            Err(SibsonInformationError::InvalidAlpha)
        ));
        hypotheses[0].outcome_probabilities_by_action[0][0] = 1.2;
        assert!(matches!(
            evaluate_sibson_alpha_information(&hypotheses, 0.5),
            Err(SibsonInformationError::InvalidLikelihood { .. })
        ));
    }
}
