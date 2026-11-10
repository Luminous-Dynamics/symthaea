// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Prospective model information gain for scalar Gaussian observation channels.
//!
//! This module evaluates I(H; O | a) where a candidate action induces one
//! Gaussian observation distribution per latent hypothesis:
//!
//!   p(o | H=i, a) = Normal(mean[i], observation_std_dev^2).
//!
//! It uses deterministic one-dimensional quadrature of the Gaussian-mixture
//! mutual-information integral. It is intended for small finite hypothesis
//! spaces (for example, exact hidden-unit permutation hypotheses), not for
//! arbitrary high-dimensional continuous observations. Callers must provide
//! an observation model with a defensible noise scale; the module does not
//! infer one from data or turn deterministic scores into "information gain"
//! without an explicit probabilistic channel.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const MAX_HYPOTHESES: usize = 1024;
const MIN_QUADRATURE_POINTS: usize = 65;
const MAX_QUADRATURE_POINTS: usize = 2049;

/// One latent model/permutation hypothesis before a prospective observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianObservationHypothesis {
    pub hypothesis_id: String,
    pub prior_probability: f64,
}

/// One candidate action and its predicted scalar observation mean under each hypothesis.
///
/// The means are ordered identically to the supplied hypothesis slice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianObservationAction {
    pub action_id: String,
    pub outcome_means: Vec<f64>,
}

/// Numerical settings for the Gaussian observation channel and integral.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GaussianModelInformationGainConfig {
    /// Standard deviation of the measured scalar outcome under every hypothesis.
    pub observation_std_dev: f64,
    /// Odd number of quadrature points over the standard-normal coordinate.
    pub quadrature_points: usize,
    /// Integrate the standard-normal coordinate over [-tail_sigma, +tail_sigma].
    pub tail_sigma: f64,
}

impl Default for GaussianModelInformationGainConfig {
    fn default() -> Self {
        Self {
            observation_std_dev: 0.05,
            quadrature_points: 257,
            tail_sigma: 8.0,
        }
    }
}

/// Prospective information value of one candidate observation/action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianModelInformationGain {
    pub action_id: String,
    pub current_model_entropy_nats: f64,
    pub expected_posterior_model_entropy_nats: f64,
    pub model_information_gain_nats: f64,
    pub observation_std_dev: f64,
    pub hypothesis_count: usize,
    pub quadrature_points: usize,
}

/// Fail-closed validation and numerical errors.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GaussianModelInformationGainError {
    #[error("at least one Gaussian observation hypothesis is required")]
    EmptyHypothesisSet,
    #[error("at least one candidate observation action is required")]
    EmptyActionSet,
    #[error("hypothesis at index {index} has an empty ID")]
    EmptyHypothesisId { index: usize },
    #[error("duplicate hypothesis ID: {0}")]
    DuplicateHypothesisId(String),
    #[error("action at index {index} has an empty ID")]
    EmptyActionId { index: usize },
    #[error("duplicate action ID: {0}")]
    DuplicateActionId(String),
    #[error("hypothesis {hypothesis_id} has invalid prior probability {probability}")]
    InvalidPriorProbability {
        hypothesis_id: String,
        probability: f64,
    },
    #[error("hypothesis prior probability mass sums to {sum}, expected 1")]
    PriorMassNotNormalized { sum: f64 },
    #[error("action {action_id} has {actual} means, expected {expected}")]
    OutcomeMeanCountMismatch {
        action_id: String,
        expected: usize,
        actual: usize,
    },
    #[error("action {action_id} has a non-finite outcome mean at hypothesis index {index}")]
    NonFiniteOutcomeMean { action_id: String, index: usize },
    #[error("observation standard deviation must be positive and finite, got {0}")]
    InvalidObservationStdDev(f64),
    #[error(
        "quadrature point count must be odd and in {MIN_QUADRATURE_POINTS}..={MAX_QUADRATURE_POINTS}, got {0}"
    )]
    InvalidQuadraturePointCount(usize),
    #[error("integration tail sigma must be positive and finite, got {0}")]
    InvalidTailSigma(f64),
    #[error("hypothesis count {actual} exceeds exact Gaussian-mixture limit {maximum}")]
    TooManyHypotheses { actual: usize, maximum: usize },
    #[error("computed information gain is non-finite or outside [0, prior entropy]: {0}")]
    InvalidInformationGain(f64),
}

impl GaussianModelInformationGainError {
    fn validate_inputs(
        hypotheses: &[GaussianObservationHypothesis],
        actions: &[GaussianObservationAction],
        config: GaussianModelInformationGainConfig,
    ) -> Result<(), Self> {
        if hypotheses.is_empty() {
            return Err(Self::EmptyHypothesisSet);
        }
        if hypotheses.len() > MAX_HYPOTHESES {
            return Err(Self::TooManyHypotheses {
                actual: hypotheses.len(),
                maximum: MAX_HYPOTHESES,
            });
        }
        if actions.is_empty() {
            return Err(Self::EmptyActionSet);
        }
        if !config.observation_std_dev.is_finite() || config.observation_std_dev <= 0.0 {
            return Err(Self::InvalidObservationStdDev(config.observation_std_dev));
        }
        if config.quadrature_points < MIN_QUADRATURE_POINTS
            || config.quadrature_points > MAX_QUADRATURE_POINTS
            || config.quadrature_points % 2 == 0
        {
            return Err(Self::InvalidQuadraturePointCount(config.quadrature_points));
        }
        if !config.tail_sigma.is_finite() || config.tail_sigma <= 0.0 {
            return Err(Self::InvalidTailSigma(config.tail_sigma));
        }

        let mut hypothesis_ids = BTreeSet::new();
        let mut prior_sum = 0.0;
        for (index, hypothesis) in hypotheses.iter().enumerate() {
            if hypothesis.hypothesis_id.trim().is_empty() {
                return Err(Self::EmptyHypothesisId { index });
            }
            if !hypothesis_ids.insert(hypothesis.hypothesis_id.clone()) {
                return Err(Self::DuplicateHypothesisId(
                    hypothesis.hypothesis_id.clone(),
                ));
            }
            let probability = hypothesis.prior_probability;
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(Self::InvalidPriorProbability {
                    hypothesis_id: hypothesis.hypothesis_id.clone(),
                    probability,
                });
            }
            prior_sum += probability;
        }
        if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(Self::PriorMassNotNormalized { sum: prior_sum });
        }

        let mut action_ids = BTreeSet::new();
        for (action_index, action) in actions.iter().enumerate() {
            if action.action_id.trim().is_empty() {
                return Err(Self::EmptyActionId {
                    index: action_index,
                });
            }
            if !action_ids.insert(action.action_id.clone()) {
                return Err(Self::DuplicateActionId(action.action_id.clone()));
            }
            if action.outcome_means.len() != hypotheses.len() {
                return Err(Self::OutcomeMeanCountMismatch {
                    action_id: action.action_id.clone(),
                    expected: hypotheses.len(),
                    actual: action.outcome_means.len(),
                });
            }
            if let Some(index) = action
                .outcome_means
                .iter()
                .position(|mean| !mean.is_finite())
            {
                return Err(Self::NonFiniteOutcomeMean {
                    action_id: action.action_id.clone(),
                    index,
                });
            }
        }
        Ok(())
    }
}

/// Evaluate prospective mutual information for every candidate Gaussian observation.
///
/// For each hypothesis i, the integral is transformed to a standard-normal
/// coordinate z. The mixture likelihood is computed in log space:
///
///   log p(o | i) / p(o)
///     = -z^2/2 - log sum_j p(j) exp(-(z + (mu_i-mu_j)/sigma)^2/2).
///
/// Composite Simpson quadrature is deterministic and uses the same fixed
/// observation-noise variance for all hypotheses. Near-zero numerical negatives
/// are clamped only after checking the result against the prior entropy bound.
pub fn evaluate_gaussian_model_information_gains(
    hypotheses: &[GaussianObservationHypothesis],
    actions: &[GaussianObservationAction],
    config: GaussianModelInformationGainConfig,
) -> Result<Vec<GaussianModelInformationGain>, GaussianModelInformationGainError> {
    GaussianModelInformationGainError::validate_inputs(hypotheses, actions, config)?;

    let prior_entropy = hypotheses
        .iter()
        .filter(|hypothesis| hypothesis.prior_probability > 0.0)
        .map(|hypothesis| -hypothesis.prior_probability * hypothesis.prior_probability.ln())
        .sum::<f64>();

    let quadrature_weights =
        standard_normal_simpson_weights(config.quadrature_points, config.tail_sigma);
    let std_dev = config.observation_std_dev;

    actions
        .iter()
        .map(|action| {
            let mut information_gain = 0.0_f64;
            for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
                let prior = hypothesis.prior_probability;
                if prior == 0.0 {
                    continue;
                }
                let own_mean = action.outcome_means[hypothesis_index];
                let mut conditional_information = 0.0_f64;

                for (z, quadrature_weight) in &quadrature_weights {
                    let mut log_mixture_terms = Vec::with_capacity(hypotheses.len());
                    for (other, other_mean) in
                        hypotheses.iter().zip(action.outcome_means.iter().copied())
                    {
                        if other.prior_probability == 0.0 {
                            continue;
                        }
                        let standardized = z + (own_mean - other_mean) / std_dev;
                        log_mixture_terms
                            .push(other.prior_probability.ln() - 0.5 * standardized * standardized);
                    }
                    let log_mixture = log_sum_exp(&log_mixture_terms);
                    let log_likelihood_ratio = -0.5 * z * z - log_mixture;
                    conditional_information += quadrature_weight * log_likelihood_ratio;
                }
                information_gain += prior * conditional_information;
            }

            let tolerance = 1e-8 * prior_entropy.max(1.0);
            if !information_gain.is_finite()
                || information_gain < -tolerance
                || information_gain > prior_entropy + tolerance
            {
                return Err(GaussianModelInformationGainError::InvalidInformationGain(
                    information_gain,
                ));
            }
            let information_gain = information_gain.clamp(0.0, prior_entropy);
            Ok(GaussianModelInformationGain {
                action_id: action.action_id.clone(),
                current_model_entropy_nats: prior_entropy,
                expected_posterior_model_entropy_nats: prior_entropy - information_gain,
                model_information_gain_nats: information_gain,
                observation_std_dev: std_dev,
                hypothesis_count: hypotheses.len(),
                quadrature_points: config.quadrature_points,
            })
        })
        .collect()
}

fn standard_normal_simpson_weights(point_count: usize, tail_sigma: f64) -> Vec<(f64, f64)> {
    let lower = -tail_sigma;
    let upper = tail_sigma;
    let step = (upper - lower) / (point_count - 1) as f64;

    let mut points = Vec::with_capacity(point_count);
    let mut total_weight = 0.0_f64;
    for index in 0..point_count {
        let z = lower + index as f64 * step;
        let simpson = if index == 0 || index + 1 == point_count {
            1.0
        } else if index % 2 == 0 {
            2.0
        } else {
            4.0
        };
        let weight = simpson * (-0.5 * z * z).exp();
        total_weight += weight;
        points.push((z, weight));
    }

    for (_, weight) in &mut points {
        *weight /= total_weight;
    }
    points
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return f64::NEG_INFINITY;
    }
    maximum
        + values
            .iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hypotheses(priors: &[f64]) -> Vec<GaussianObservationHypothesis> {
        priors
            .iter()
            .copied()
            .enumerate()
            .map(|(index, prior_probability)| GaussianObservationHypothesis {
                hypothesis_id: format!("h{index}"),
                prior_probability,
            })
            .collect()
    }

    fn config(sigma: f64) -> GaussianModelInformationGainConfig {
        GaussianModelInformationGainConfig {
            observation_std_dev: sigma,
            quadrature_points: 257,
            tail_sigma: 8.0,
        }
    }

    #[test]
    fn identical_hypothesis_means_have_zero_information_gain() {
        let models = hypotheses(&[0.2, 0.3, 0.5]);
        let actions = vec![GaussianObservationAction {
            action_id: "indistinguishable".into(),
            outcome_means: vec![0.4, 0.4, 0.4],
        }];
        let gain = evaluate_gaussian_model_information_gains(&models, &actions, config(0.1))
            .expect("valid Gaussian observation model");
        assert!(gain[0].model_information_gain_nats.abs() < 1e-12);
        assert!(
            (gain[0].expected_posterior_model_entropy_nats - gain[0].current_model_entropy_nats)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn well_separated_hypotheses_reveal_nearly_all_model_information() {
        let models = hypotheses(&[0.5, 0.5]);
        let actions = vec![GaussianObservationAction {
            action_id: "separating".into(),
            outcome_means: vec![-5.0, 5.0],
        }];
        let gain = evaluate_gaussian_model_information_gains(&models, &actions, config(0.1))
            .expect("valid Gaussian observation model");
        assert!((gain[0].model_information_gain_nats - std::f64::consts::LN_2).abs() < 1e-9);
    }

    #[test]
    fn information_gain_is_bounded_by_prior_entropy_and_increases_as_noise_falls() {
        let models = hypotheses(&[0.7, 0.3]);
        let actions = vec![GaussianObservationAction {
            action_id: "moderate-separation".into(),
            outcome_means: vec![0.0, 0.5],
        }];
        let noisy = evaluate_gaussian_model_information_gains(&models, &actions, config(0.5))
            .expect("noisy model");
        let clearer = evaluate_gaussian_model_information_gains(&models, &actions, config(0.1))
            .expect("clearer model");
        assert!(clearer[0].model_information_gain_nats > noisy[0].model_information_gain_nats);
        assert!(
            clearer[0].model_information_gain_nats <= clearer[0].current_model_entropy_nats + 1e-12
        );
    }

    #[test]
    fn duplicate_hypotheses_are_not_distinguished_by_the_channel() {
        let models = hypotheses(&[0.2, 0.3, 0.5]);
        let actions = vec![GaussianObservationAction {
            action_id: "two-equivalence-classes".into(),
            outcome_means: vec![-2.0, -2.0, 2.0],
        }];
        let gain = evaluate_gaussian_model_information_gains(&models, &actions, config(0.05))
            .expect("valid Gaussian observation model");
        let class_entropy = -0.5_f64 * 0.5_f64.ln() - 0.5_f64 * 0.5_f64.ln();
        assert!((gain[0].model_information_gain_nats - class_entropy).abs() < 1e-8);
        assert!(gain[0].model_information_gain_nats < gain[0].current_model_entropy_nats);
    }

    #[test]
    fn action_order_and_hypothesis_order_do_not_change_named_action_scores() {
        let models = hypotheses(&[0.25, 0.75]);
        let actions = vec![
            GaussianObservationAction {
                action_id: "a".into(),
                outcome_means: vec![0.0, 0.0],
            },
            GaussianObservationAction {
                action_id: "b".into(),
                outcome_means: vec![-1.0, 1.0],
            },
        ];
        let first = evaluate_gaussian_model_information_gains(&models, &actions, config(0.2))
            .expect("first ordering");
        let reordered_models = vec![
            GaussianObservationHypothesis {
                hypothesis_id: "h1".into(),
                prior_probability: 0.75,
            },
            GaussianObservationHypothesis {
                hypothesis_id: "h0".into(),
                prior_probability: 0.25,
            },
        ];
        let reordered_actions = vec![
            GaussianObservationAction {
                action_id: "b".into(),
                outcome_means: vec![1.0, -1.0],
            },
            GaussianObservationAction {
                action_id: "a".into(),
                outcome_means: vec![0.0, 0.0],
            },
        ];
        let second = evaluate_gaussian_model_information_gains(
            &reordered_models,
            &reordered_actions,
            config(0.2),
        )
        .expect("reordered input");
        let score = |results: &[GaussianModelInformationGain], id: &str| {
            results
                .iter()
                .find(|result| result.action_id == id)
                .expect("action score")
                .model_information_gain_nats
        };
        assert!((score(&first, "a") - score(&second, "a")).abs() < 1e-12);
        assert!((score(&first, "b") - score(&second, "b")).abs() < 1e-12);
    }

    #[test]
    fn malformed_distributions_and_observation_models_fail_closed() {
        let models = hypotheses(&[0.4, 0.4]);
        let action = vec![GaussianObservationAction {
            action_id: "a".into(),
            outcome_means: vec![0.0, 1.0],
        }];
        assert!(matches!(
            evaluate_gaussian_model_information_gains(&[], &action, config(0.1)),
            Err(GaussianModelInformationGainError::EmptyHypothesisSet)
        ));
        assert!(matches!(
            evaluate_gaussian_model_information_gains(&models, &action, config(0.0)),
            Err(GaussianModelInformationGainError::InvalidObservationStdDev(
                _
            ))
        ));
        assert!(matches!(
            evaluate_gaussian_model_information_gains(
                &models,
                &action,
                config(0.1).clone_with_quadrature(64)
            ),
            Err(GaussianModelInformationGainError::InvalidQuadraturePointCount(64))
        ));
        assert!(matches!(
            evaluate_gaussian_model_information_gains(
                &models,
                &[GaussianObservationAction {
                    action_id: "bad".into(),
                    outcome_means: vec![0.0],
                }],
                config(0.1)
            ),
            Err(GaussianModelInformationGainError::OutcomeMeanCountMismatch { .. })
        ));
        assert!(matches!(
            evaluate_gaussian_model_information_gains(
                &hypotheses(&[0.5, 0.5]),
                &[GaussianObservationAction {
                    action_id: "non-finite".into(),
                    outcome_means: vec![0.0, f64::NAN],
                }],
                config(0.1)
            ),
            Err(GaussianModelInformationGainError::NonFiniteOutcomeMean { .. })
        ));
    }

    trait QuadratureConfigTestExt {
        fn clone_with_quadrature(self, points: usize) -> Self;
    }

    impl QuadratureConfigTestExt for GaussianModelInformationGainConfig {
        fn clone_with_quadrature(mut self, points: usize) -> Self {
            self.quadrature_points = points;
            self
        }
    }
}
