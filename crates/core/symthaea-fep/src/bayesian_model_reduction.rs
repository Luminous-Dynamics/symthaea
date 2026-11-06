// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Bayesian model reduction (BMR) for nested diagonal-Gaussian parameter priors.
//!
//! Given a Gaussian posterior under a full model and reduced models that differ
//! only in their parameter priors, this module computes
//!
//!   log p(y | reduced) / p(y | full)
//!     = log E_q(theta | y, full)[p(theta | reduced) / p(theta | full)].
//!
//! The integral is analytic for diagonal Gaussians. It is exact when the
//! supplied posterior is the exact full-model posterior; with a variational
//! Gaussian posterior it is the corresponding variational BMR approximation.
//!
//! This is a posterior model-evidence update, not prospective expected
//! information gain and not a complete expected-free-energy policy. Candidate
//! actions still need action-conditioned predictive outcome models. Reduced
//! models must share the full model's parameterization and may differ only
//! through prior restrictions represented here.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// Independent Gaussian coordinates for a parameter vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagonalGaussian {
    /// Digest of the canonical ordered parameterization: names, roles, shapes, and coordinate order.
    /// All three Gaussian distributions in a BMR comparison must carry the same digest.
    pub parameterization_digest: String,
    pub mean: Vec<f64>,
    /// Variances, not precisions. Every entry must be finite and strictly positive.
    pub variance: Vec<f64>,
}

/// One reduced model with a prior probability and diagonal-Gaussian parameter prior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReducedModelHypothesis {
    pub model_id: String,
    pub prior_probability: f64,
    pub parameter_prior: DiagonalGaussian,
}

/// Evidence and posterior weight for one reduced model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReducedModelEvidence {
    pub model_id: String,
    pub prior_probability: f64,
    /// Log marginal-likelihood ratio against the supplied full model.
    pub log_evidence_ratio: f64,
    pub posterior_probability: f64,
}

/// Posterior over reduced models after Bayesian model reduction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BayesianModelReductionResult {
    pub models: Vec<ReducedModelEvidence>,
    pub posterior_model_entropy_nats: f64,
    pub most_probable_model_id: String,
    /// Posterior log-odds of the most probable model against the runner-up.
    /// None when only one model has nonzero posterior mass.
    pub log_posterior_odds_over_runner_up: Option<f64>,
}

/// Input or numerical errors. Invalid distributions are rejected, not normalized silently.
#[derive(Debug, Clone, PartialEq)]
pub enum BayesianModelReductionError {
    EmptyParameterVector,
    InvalidParameterizationDigest {
        distribution: &'static str,
    },
    ParameterizationMismatch {
        distribution: &'static str,
        expected: String,
        actual: String,
    },
    ParameterDimensionMismatch {
        distribution: &'static str,
        mean_len: usize,
        variance_len: usize,
        expected: usize,
    },
    NonFiniteMean {
        distribution: &'static str,
        index: usize,
    },
    InvalidVariance {
        distribution: &'static str,
        index: usize,
        variance: f64,
    },
    EmptyModelSet,
    EmptyModelId {
        index: usize,
    },
    DuplicateModelId(String),
    InvalidModelPrior {
        model_id: String,
        probability: f64,
    },
    ModelPriorMassNotNormalized {
        sum: f64,
    },
    NonIntegrableEvidenceRatio {
        model_id: String,
        parameter_index: usize,
        denominator: f64,
    },
    NonFiniteEvidenceRatio {
        model_id: String,
        parameter_index: usize,
    },
    InvalidPosteriorNormalization,
}

impl std::fmt::Display for BayesianModelReductionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyParameterVector => write!(f, "parameter vectors must not be empty"),
            Self::InvalidParameterizationDigest { distribution } => {
                write!(f, "{distribution} has an invalid parameterization digest")
            }
            Self::ParameterizationMismatch {
                distribution,
                expected,
                actual,
            } => write!(
                f,
                "{distribution} parameterization digest {actual} does not match full posterior {expected}"
            ),
            Self::ParameterDimensionMismatch {
                distribution,
                mean_len,
                variance_len,
                expected,
            } => write!(
                f,
                "{distribution} has mean length {mean_len} and variance length {variance_len}; expected {expected}"
            ),
            Self::NonFiniteMean {
                distribution,
                index,
            } => {
                write!(f, "{distribution} mean at index {index} is non-finite")
            }
            Self::InvalidVariance {
                distribution,
                index,
                variance,
            } => write!(
                f,
                "{distribution} variance at index {index} must be positive and finite, got {variance}"
            ),
            Self::EmptyModelSet => write!(f, "at least one reduced model is required"),
            Self::EmptyModelId { index } => {
                write!(f, "reduced model at index {index} has an empty model ID")
            }
            Self::DuplicateModelId(id) => write!(f, "duplicate reduced model ID: {id}"),
            Self::InvalidModelPrior {
                model_id,
                probability,
            } => write!(
                f,
                "model {model_id} has invalid prior probability {probability}"
            ),
            Self::ModelPriorMassNotNormalized { sum } => {
                write!(f, "reduced-model prior mass sums to {sum}, expected 1")
            }
            Self::NonIntegrableEvidenceRatio {
                model_id,
                parameter_index,
                denominator,
            } => write!(
                f,
                "BMR ratio for model {model_id}, parameter {parameter_index} is not integrable (denominator {denominator})"
            ),
            Self::NonFiniteEvidenceRatio {
                model_id,
                parameter_index,
            } => write!(
                f,
                "BMR log-evidence ratio for model {model_id}, parameter {parameter_index} is non-finite"
            ),
            Self::InvalidPosteriorNormalization => {
                write!(f, "reduced-model posterior could not be normalized")
            }
        }
    }
}

impl std::error::Error for BayesianModelReductionError {}

/// Compare reduced models whose priors share the full model's parameterization.
///
/// The posterior and full prior define the reference full model. Each reduced
/// model supplies a prior over exactly the same parameter coordinates. The
/// returned evidence ratios are relative to the full model; prior model
/// probabilities are then incorporated using Bayes' rule.
pub fn reduce_models(
    full_posterior: &DiagonalGaussian,
    full_prior: &DiagonalGaussian,
    reduced_models: &[ReducedModelHypothesis],
) -> Result<BayesianModelReductionResult, BayesianModelReductionError> {
    let dimension = full_posterior.mean.len();
    if dimension == 0 {
        return Err(BayesianModelReductionError::EmptyParameterVector);
    }
    validate_gaussian(full_posterior, "full posterior", dimension)?;
    validate_gaussian(full_prior, "full prior", dimension)?;
    ensure_parameterization_matches(
        &full_posterior.parameterization_digest,
        "full prior",
        &full_prior.parameterization_digest,
    )?;

    if reduced_models.is_empty() {
        return Err(BayesianModelReductionError::EmptyModelSet);
    }

    let mut model_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    for (index, model) in reduced_models.iter().enumerate() {
        if model.model_id.trim().is_empty() {
            return Err(BayesianModelReductionError::EmptyModelId { index });
        }
        if !model_ids.insert(model.model_id.clone()) {
            return Err(BayesianModelReductionError::DuplicateModelId(
                model.model_id.clone(),
            ));
        }
        if !model.prior_probability.is_finite() || !(0.0..=1.0).contains(&model.prior_probability) {
            return Err(BayesianModelReductionError::InvalidModelPrior {
                model_id: model.model_id.clone(),
                probability: model.prior_probability,
            });
        }
        prior_sum += model.prior_probability;
        validate_gaussian(&model.parameter_prior, "reduced prior", dimension)?;
        ensure_parameterization_matches(
            &full_posterior.parameterization_digest,
            "reduced prior",
            &model.parameter_prior.parameterization_digest,
        )?;
    }
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(BayesianModelReductionError::ModelPriorMassNotNormalized { sum: prior_sum });
    }

    let mut log_weights = Vec::with_capacity(reduced_models.len());
    let mut log_evidence_ratios = Vec::with_capacity(reduced_models.len());
    for model in reduced_models {
        let log_ratio = diagonal_gaussian_log_evidence_ratio(
            full_posterior,
            full_prior,
            &model.parameter_prior,
            &model.model_id,
        )?;
        log_evidence_ratios.push(log_ratio);
        log_weights.push(if model.prior_probability == 0.0 {
            f64::NEG_INFINITY
        } else {
            model.prior_probability.ln() + log_ratio
        });
    }

    let max_log_weight = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_log_weight.is_finite() {
        return Err(BayesianModelReductionError::InvalidPosteriorNormalization);
    }
    let unnormalized = log_weights
        .iter()
        .map(|value| (value - max_log_weight).exp())
        .collect::<Vec<_>>();
    let normalizer = unnormalized.iter().sum::<f64>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(BayesianModelReductionError::InvalidPosteriorNormalization);
    }
    let posterior = unnormalized
        .iter()
        .map(|value| value / normalizer)
        .collect::<Vec<_>>();

    let entropy = posterior
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f64>();

    let mut order = (0..reduced_models.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| posterior[*right].total_cmp(&posterior[*left]));
    let best_index = order[0];
    let runner_up = order
        .iter()
        .copied()
        .skip(1)
        .find(|index| posterior[*index] > 0.0);
    let log_posterior_odds_over_runner_up =
        runner_up.map(|runner_index| log_weights[best_index] - log_weights[runner_index]);

    let models = reduced_models
        .iter()
        .zip(log_evidence_ratios.iter().copied())
        .zip(posterior.iter().copied())
        .map(
            |((model, log_evidence_ratio), posterior_probability)| ReducedModelEvidence {
                model_id: model.model_id.clone(),
                prior_probability: model.prior_probability,
                log_evidence_ratio,
                posterior_probability,
            },
        )
        .collect();

    Ok(BayesianModelReductionResult {
        models,
        posterior_model_entropy_nats: entropy,
        most_probable_model_id: reduced_models[best_index].model_id.clone(),
        log_posterior_odds_over_runner_up,
    })
}

/// Analytic log E_q[p_reduced(theta) / p_full(theta)] for diagonal Gaussians.
///
/// For one coordinate, with q=N(mq,vq), p0=N(m0,v0), and pr=N(mr,vr),
/// the contribution is
///
///   C - 0.5 ln(1 + vq A) + 0.5 vq B^2 / (1 + vq A),
///
/// where A=1/vr-1/v0, B=(mr-mq)/vr-(m0-mq)/v0, and
/// C=0.5 ln(v0/vr)-(mq-mr)^2/(2vr)+(mq-m0)^2/(2v0).
///
/// The product factorizes over independent coordinates. The integral exists
/// only when every denominator (1 + vq A) is strictly positive.
fn diagonal_gaussian_log_evidence_ratio(
    posterior: &DiagonalGaussian,
    full_prior: &DiagonalGaussian,
    reduced_prior: &DiagonalGaussian,
    model_id: &str,
) -> Result<f64, BayesianModelReductionError> {
    let mut total = 0.0_f64;
    for index in 0..posterior.mean.len() {
        let mq = posterior.mean[index];
        let vq = posterior.variance[index];
        let m0 = full_prior.mean[index];
        let v0 = full_prior.variance[index];
        let mr = reduced_prior.mean[index];
        let vr = reduced_prior.variance[index];

        let inverse_full = 1.0 / v0;
        let inverse_reduced = 1.0 / vr;
        let a = inverse_reduced - inverse_full;
        let b = (mr - mq) * inverse_reduced - (m0 - mq) * inverse_full;
        let denominator = 1.0 + vq * a;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(BayesianModelReductionError::NonIntegrableEvidenceRatio {
                model_id: model_id.to_owned(),
                parameter_index: index,
                denominator,
            });
        }

        let constant = 0.5 * (v0.ln() - vr.ln()) - (mq - mr).powi(2) * (0.5 * inverse_reduced)
            + (mq - m0).powi(2) * (0.5 * inverse_full);
        let contribution = constant - 0.5 * denominator.ln() + 0.5 * vq * b.powi(2) / denominator;
        if !contribution.is_finite() {
            return Err(BayesianModelReductionError::NonFiniteEvidenceRatio {
                model_id: model_id.to_owned(),
                parameter_index: index,
            });
        }
        total += contribution;
        if !total.is_finite() {
            return Err(BayesianModelReductionError::NonFiniteEvidenceRatio {
                model_id: model_id.to_owned(),
                parameter_index: index,
            });
        }
    }
    Ok(total)
}

fn validate_gaussian(
    gaussian: &DiagonalGaussian,
    distribution: &'static str,
    expected_dimension: usize,
) -> Result<(), BayesianModelReductionError> {
    if !is_sha256_digest(&gaussian.parameterization_digest) {
        return Err(BayesianModelReductionError::InvalidParameterizationDigest { distribution });
    }
    if gaussian.mean.len() != expected_dimension || gaussian.variance.len() != expected_dimension {
        return Err(BayesianModelReductionError::ParameterDimensionMismatch {
            distribution,
            mean_len: gaussian.mean.len(),
            variance_len: gaussian.variance.len(),
            expected: expected_dimension,
        });
    }
    for (index, mean) in gaussian.mean.iter().copied().enumerate() {
        if !mean.is_finite() {
            return Err(BayesianModelReductionError::NonFiniteMean {
                distribution,
                index,
            });
        }
    }
    for (index, variance) in gaussian.variance.iter().copied().enumerate() {
        if !variance.is_finite() || variance <= 0.0 || !(1.0 / variance).is_finite() {
            return Err(BayesianModelReductionError::InvalidVariance {
                distribution,
                index,
                variance,
            });
        }
    }
    Ok(())
}

fn ensure_parameterization_matches(
    expected: &str,
    distribution: &'static str,
    actual: &str,
) -> Result<(), BayesianModelReductionError> {
    if actual != expected {
        return Err(BayesianModelReductionError::ParameterizationMismatch {
            distribution,
            expected: expected.to_owned(),
            actual: actual.to_owned(),
        });
    }
    Ok(())
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn gaussian(mean: &[f64], variance: &[f64]) -> DiagonalGaussian {
        gaussian_with_digest(mean, variance, 'a')
    }

    fn gaussian_with_digest(
        mean: &[f64],
        variance: &[f64],
        parameterization: char,
    ) -> DiagonalGaussian {
        DiagonalGaussian {
            parameterization_digest: digest(parameterization),
            mean: mean.to_vec(),
            variance: variance.to_vec(),
        }
    }

    #[test]
    fn identical_reduced_and_full_priors_have_zero_log_evidence_ratio() {
        let prior = gaussian(&[0.0, 1.0], &[1.0, 2.0]);
        let posterior = gaussian(&[0.2, 0.8], &[0.4, 0.9]);
        let result = reduce_models(
            &posterior,
            &prior,
            &[ReducedModelHypothesis {
                model_id: "full".into(),
                prior_probability: 1.0,
                parameter_prior: prior.clone(),
            }],
        )
        .expect("identical priors are valid");
        assert!(result.models[0].log_evidence_ratio.abs() < 1e-12);
        assert!((result.models[0].posterior_probability - 1.0).abs() < 1e-12);
        assert_eq!(result.most_probable_model_id, "full");
    }

    #[test]
    fn gaussian_bmr_matches_independent_one_dimensional_quadrature_reference() {
        let posterior = gaussian(&[0.2], &[0.4]);
        let full_prior = gaussian(&[0.0], &[1.0]);
        let reduced_prior = gaussian(&[0.2], &[0.2]);
        let result = reduce_models(
            &posterior,
            &full_prior,
            &[ReducedModelHypothesis {
                model_id: "concentrated".into(),
                prior_probability: 1.0,
                parameter_prior: reduced_prior,
            }],
        )
        .expect("integrable Gaussian ratio");
        // Reference value independently checked by numerical integration.
        assert!((result.models[0].log_evidence_ratio - 0.3500401567802551).abs() < 1e-11);
    }

    #[test]
    fn posterior_model_weights_include_model_prior_odds() {
        let posterior = gaussian(&[0.2], &[0.4]);
        let full_prior = gaussian(&[0.0], &[1.0]);
        let result = reduce_models(
            &posterior,
            &full_prior,
            &[
                ReducedModelHypothesis {
                    model_id: "full".into(),
                    prior_probability: 0.5,
                    parameter_prior: full_prior.clone(),
                },
                ReducedModelHypothesis {
                    model_id: "concentrated".into(),
                    prior_probability: 0.5,
                    parameter_prior: gaussian(&[0.2], &[0.2]),
                },
            ],
        )
        .expect("two valid models");
        let full = &result.models[0];
        let concentrated = &result.models[1];
        assert!(concentrated.posterior_probability > full.posterior_probability);
        assert_eq!(result.most_probable_model_id, "concentrated");
        assert!(result.log_posterior_odds_over_runner_up.unwrap() > 0.0);
    }

    #[test]
    fn log_evidence_ratios_add_over_independent_coordinates() {
        let posterior = gaussian(&[0.2, -0.3], &[0.4, 0.6]);
        let full_prior = gaussian(&[0.0, 0.1], &[1.0, 0.8]);
        let reduced = gaussian(&[0.2, -0.1], &[0.2, 0.3]);
        let both = reduce_models(
            &posterior,
            &full_prior,
            &[ReducedModelHypothesis {
                model_id: "joint".into(),
                prior_probability: 1.0,
                parameter_prior: reduced.clone(),
            }],
        )
        .expect("joint reduction");
        let first = reduce_models(
            &gaussian(&[0.2], &[0.4]),
            &gaussian(&[0.0], &[1.0]),
            &[ReducedModelHypothesis {
                model_id: "first".into(),
                prior_probability: 1.0,
                parameter_prior: gaussian(&[0.2], &[0.2]),
            }],
        )
        .expect("first coordinate");
        let second = reduce_models(
            &gaussian(&[-0.3], &[0.6]),
            &gaussian(&[0.1], &[0.8]),
            &[ReducedModelHypothesis {
                model_id: "second".into(),
                prior_probability: 1.0,
                parameter_prior: gaussian(&[-0.1], &[0.3]),
            }],
        )
        .expect("second coordinate");
        let expected = first.models[0].log_evidence_ratio + second.models[0].log_evidence_ratio;
        assert!((both.models[0].log_evidence_ratio - expected).abs() < 1e-12);
    }

    #[test]
    fn non_integrable_ratio_is_rejected() {
        let posterior = gaussian(&[0.0], &[2.0]);
        let full_prior = gaussian(&[0.0], &[1.0]);
        let too_wide_reduced = gaussian(&[0.0], &[10.0]);
        let result = reduce_models(
            &posterior,
            &full_prior,
            &[ReducedModelHypothesis {
                model_id: "too-wide".into(),
                prior_probability: 1.0,
                parameter_prior: too_wide_reduced,
            }],
        );
        assert!(matches!(
            result,
            Err(BayesianModelReductionError::NonIntegrableEvidenceRatio { .. })
        ));
    }

    #[test]
    fn malformed_dimensions_variances_and_model_priors_are_rejected() {
        let posterior = gaussian(&[0.0], &[1.0]);
        let full_prior = gaussian(&[0.0], &[1.0]);
        let mismatch = reduce_models(
            &posterior,
            &full_prior,
            &[ReducedModelHypothesis {
                model_id: "bad-dimension".into(),
                prior_probability: 1.0,
                parameter_prior: gaussian(&[0.0, 1.0], &[1.0, 1.0]),
            }],
        );
        assert!(matches!(
            mismatch,
            Err(BayesianModelReductionError::ParameterDimensionMismatch { .. })
        ));

        let invalid_variance = reduce_models(
            &posterior,
            &full_prior,
            &[ReducedModelHypothesis {
                model_id: "bad-variance".into(),
                prior_probability: 1.0,
                parameter_prior: gaussian(&[0.0], &[0.0]),
            }],
        );
        assert!(matches!(
            invalid_variance,
            Err(BayesianModelReductionError::InvalidVariance { .. })
        ));

        let invalid_prior_mass = reduce_models(
            &posterior,
            &full_prior,
            &[
                ReducedModelHypothesis {
                    model_id: "a".into(),
                    prior_probability: 0.2,
                    parameter_prior: full_prior.clone(),
                },
                ReducedModelHypothesis {
                    model_id: "b".into(),
                    prior_probability: 0.2,
                    parameter_prior: full_prior.clone(),
                },
            ],
        );
        assert!(matches!(
            invalid_prior_mass,
            Err(BayesianModelReductionError::ModelPriorMassNotNormalized { .. })
        ));
    }

    #[test]
    fn parameter_order_permutation_does_not_change_evidence() {
        let result_a = reduce_models(
            &gaussian(&[0.2, -0.3], &[0.4, 0.6]),
            &gaussian(&[0.0, 0.1], &[1.0, 0.8]),
            &[ReducedModelHypothesis {
                model_id: "model".into(),
                prior_probability: 1.0,
                parameter_prior: gaussian(&[0.2, -0.1], &[0.2, 0.3]),
            }],
        )
        .expect("first coordinate order");
        let result_b = reduce_models(
            &gaussian_with_digest(&[-0.3, 0.2], &[0.6, 0.4], 'b'),
            &gaussian_with_digest(&[0.1, 0.0], &[0.8, 1.0], 'b'),
            &[ReducedModelHypothesis {
                model_id: "model".into(),
                prior_probability: 1.0,
                parameter_prior: gaussian_with_digest(&[-0.1, 0.2], &[0.3, 0.2], 'b'),
            }],
        )
        .expect("permuted coordinate order");
        assert!(
            (result_a.models[0].log_evidence_ratio - result_b.models[0].log_evidence_ratio).abs()
                < 1e-12
        );
    }

    #[test]
    fn malformed_parameterization_digest_is_rejected_before_evidence_calculation() {
        let mut posterior = gaussian(&[0.0], &[1.0]);
        posterior.parameterization_digest = "missing-or-malformed".into();
        let prior = gaussian(&[0.0], &[1.0]);
        let result = reduce_models(
            &posterior,
            &prior,
            &[ReducedModelHypothesis {
                model_id: "full".into(),
                prior_probability: 1.0,
                parameter_prior: prior.clone(),
            }],
        );
        assert!(matches!(
            result,
            Err(BayesianModelReductionError::InvalidParameterizationDigest {
                distribution: "full posterior"
            })
        ));
    }

    #[test]
    fn mismatched_full_prior_parameterization_digest_is_rejected() {
        let posterior = gaussian(&[0.0, 0.2], &[1.0, 0.5]);
        let full_prior = gaussian_with_digest(&[0.0, 0.0], &[1.0, 1.0], 'b');
        let reduced_prior = gaussian(&[0.0, 0.1], &[0.5, 0.5]);
        let result = reduce_models(
            &posterior,
            &full_prior,
            &[ReducedModelHypothesis {
                model_id: "mismatched-full-prior".into(),
                prior_probability: 1.0,
                parameter_prior: reduced_prior,
            }],
        );
        assert!(matches!(
            result,
            Err(BayesianModelReductionError::ParameterizationMismatch {
                distribution: "full prior",
                ..
            })
        ));
    }

    #[test]
    fn mismatched_parameterization_digest_is_rejected_even_when_dimensions_match() {
        let posterior = gaussian(&[0.0, 0.2], &[1.0, 0.5]);
        let full_prior = gaussian(&[0.0, 0.0], &[1.0, 1.0]);
        let reduced_prior = gaussian_with_digest(&[0.0, 0.1], &[0.5, 0.5], 'b');
        let result = reduce_models(
            &posterior,
            &full_prior,
            &[ReducedModelHypothesis {
                model_id: "same-width-different-coordinate-system".into(),
                prior_probability: 1.0,
                parameter_prior: reduced_prior,
            }],
        );
        assert!(matches!(
            result,
            Err(BayesianModelReductionError::ParameterizationMismatch {
                distribution: "reduced prior",
                ..
            })
        ));
    }

    #[test]
    fn zero_prior_model_remains_zero_posterior() {
        let posterior = gaussian(&[0.2], &[0.4]);
        let full_prior = gaussian(&[0.0], &[1.0]);
        let result = reduce_models(
            &posterior,
            &full_prior,
            &[
                ReducedModelHypothesis {
                    model_id: "zero".into(),
                    prior_probability: 0.0,
                    parameter_prior: gaussian(&[0.2], &[0.2]),
                },
                ReducedModelHypothesis {
                    model_id: "full".into(),
                    prior_probability: 1.0,
                    parameter_prior: full_prior.clone(),
                },
            ],
        )
        .expect("one positive prior mass");
        assert_eq!(result.models[0].posterior_probability, 0.0);
        assert_eq!(result.models[1].posterior_probability, 1.0);
        assert!(result.log_posterior_odds_over_runner_up.is_none());
    }
}
