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
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// One semantically named parameter group in a canonical model parameterization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGroupCoordinateDescriptor {
    /// Stable group identifier from the architecture's native parameter manifest.
    pub group_id: String,
    /// Semantic role, for example projection, bias, gate, or time constant.
    pub semantic_role: String,
    /// Tensor shape in native manifest order.
    pub shape: Vec<usize>,
    /// Stable names for every flattened scalar coordinate in native order.
    pub flattened_coordinate_names: Vec<String>,
}

/// Versioned canonical description of a model's ordered parameter coordinate space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterizationDescriptor {
    pub format_version: u32,
    /// Stable architecture identity, including topology/version where relevant.
    pub architecture_id: String,
    /// Groups are ordered; their order is part of the parameterization identity.
    pub groups: Vec<ParameterGroupCoordinateDescriptor>,
}

/// Validation errors for canonical parameterization digest construction.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ParameterizationDigestError {
    #[error("unsupported parameterization descriptor version {0}")]
    UnsupportedVersion(u32),
    #[error("architecture ID must not be empty")]
    EmptyArchitectureId,
    #[error("at least one parameter group is required")]
    EmptyGroupSet,
    #[error("parameter group at index {index} has an empty ID")]
    EmptyGroupId { index: usize },
    #[error("duplicate parameter group ID: {0}")]
    DuplicateGroupId(String),
    #[error("parameter group {group_id} has an empty semantic role")]
    EmptySemanticRole { group_id: String },
    #[error("parameter group {group_id} has an empty shape")]
    EmptyShape { group_id: String },
    #[error("parameter group {group_id} has a zero shape dimension at index {dimension_index}")]
    ZeroShapeDimension {
        group_id: String,
        dimension_index: usize,
    },
    #[error("parameter group {group_id} shape overflows usize")]
    ShapeOverflow { group_id: String },
    #[error("parameter group {group_id} has {actual} coordinate names, expected {expected}")]
    CoordinateCountMismatch {
        group_id: String,
        expected: usize,
        actual: usize,
    },
    #[error("parameter group {group_id} has an empty coordinate name at index {coordinate_index}")]
    EmptyCoordinateName {
        group_id: String,
        coordinate_index: usize,
    },
    #[error("parameter group {group_id} has duplicate coordinate name {coordinate_name}")]
    DuplicateCoordinateName {
        group_id: String,
        coordinate_name: String,
    },
}

/// Construct a stable SHA-256 digest for a parameter-coordinate space.
///
/// The digest binds architecture identity, ordered semantic groups, group roles,
/// tensor shapes, and every flattened coordinate name. A change to any field,
/// including coordinate order, produces a different digest. The encoding uses
/// domain separation and explicit little-endian length prefixes; it does not
/// depend on JSON map ordering, platform endianness, or Rust's Debug formatting.
///
/// Coordinate names must be canonical and unique within each group. Callers
/// should derive them from the same native parameter manifest that supplies
/// the model's actual flattened parameter order.
pub fn canonical_parameterization_digest(
    descriptor: &ParameterizationDescriptor,
) -> Result<String, ParameterizationDigestError> {
    if descriptor.format_version != 1 {
        return Err(ParameterizationDigestError::UnsupportedVersion(
            descriptor.format_version,
        ));
    }
    if descriptor.architecture_id.trim().is_empty() {
        return Err(ParameterizationDigestError::EmptyArchitectureId);
    }
    if descriptor.groups.is_empty() {
        return Err(ParameterizationDigestError::EmptyGroupSet);
    }

    let mut group_ids = BTreeSet::new();
    for (index, group) in descriptor.groups.iter().enumerate() {
        if group.group_id.trim().is_empty() {
            return Err(ParameterizationDigestError::EmptyGroupId { index });
        }
        if !group_ids.insert(group.group_id.clone()) {
            return Err(ParameterizationDigestError::DuplicateGroupId(
                group.group_id.clone(),
            ));
        }
        if group.semantic_role.trim().is_empty() {
            return Err(ParameterizationDigestError::EmptySemanticRole {
                group_id: group.group_id.clone(),
            });
        }
        if group.shape.is_empty() {
            return Err(ParameterizationDigestError::EmptyShape {
                group_id: group.group_id.clone(),
            });
        }
        let mut coordinate_count = 1_usize;
        for (dimension_index, dimension) in group.shape.iter().copied().enumerate() {
            if dimension == 0 {
                return Err(ParameterizationDigestError::ZeroShapeDimension {
                    group_id: group.group_id.clone(),
                    dimension_index,
                });
            }
            coordinate_count = coordinate_count.checked_mul(dimension).ok_or_else(|| {
                ParameterizationDigestError::ShapeOverflow {
                    group_id: group.group_id.clone(),
                }
            })?;
        }
        if group.flattened_coordinate_names.len() != coordinate_count {
            return Err(ParameterizationDigestError::CoordinateCountMismatch {
                group_id: group.group_id.clone(),
                expected: coordinate_count,
                actual: group.flattened_coordinate_names.len(),
            });
        }
        let mut coordinate_names = BTreeSet::new();
        for (coordinate_index, coordinate_name) in
            group.flattened_coordinate_names.iter().enumerate()
        {
            if coordinate_name.trim().is_empty() {
                return Err(ParameterizationDigestError::EmptyCoordinateName {
                    group_id: group.group_id.clone(),
                    coordinate_index,
                });
            }
            if !coordinate_names.insert(coordinate_name.clone()) {
                return Err(ParameterizationDigestError::DuplicateCoordinateName {
                    group_id: group.group_id.clone(),
                    coordinate_name: coordinate_name.clone(),
                });
            }
        }
    }

    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-fep-parameterization-descriptor-v1\0");
    hasher.update(descriptor.format_version.to_le_bytes());
    update_digest_string(&mut hasher, &descriptor.architecture_id);
    update_digest_length(&mut hasher, descriptor.groups.len());
    for group in &descriptor.groups {
        update_digest_string(&mut hasher, &group.group_id);
        update_digest_string(&mut hasher, &group.semantic_role);
        update_digest_length(&mut hasher, group.shape.len());
        for dimension in group.shape.iter().copied() {
            hasher.update((dimension as u64).to_le_bytes());
        }
        update_digest_length(&mut hasher, group.flattened_coordinate_names.len());
        for coordinate_name in &group.flattened_coordinate_names {
            update_digest_string(&mut hasher, coordinate_name);
        }
    }
    Ok(encode_lowercase_hex(&hasher.finalize()))
}

fn update_digest_length(hasher: &mut Sha256, length: usize) {
    hasher.update((length as u64).to_le_bytes());
}

fn update_digest_string(hasher: &mut Sha256, value: &str) {
    update_digest_length(hasher, value.len());
    hasher.update(value.as_bytes());
}

fn encode_lowercase_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

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

    fn log_normal_density(value: f64, mean: f64, variance: f64) -> f64 {
        -0.5 * ((2.0 * std::f64::consts::PI * variance).ln() + (value - mean).powi(2) / variance)
    }

    /// Independent Simpson integration of E_q[p_reduced(theta) / p_full(theta)].
    fn simpson_log_evidence_ratio(
        posterior: &DiagonalGaussian,
        full_prior: &DiagonalGaussian,
        reduced_prior: &DiagonalGaussian,
    ) -> f64 {
        assert_eq!(posterior.mean.len(), 1);
        assert_eq!(full_prior.mean.len(), 1);
        assert_eq!(reduced_prior.mean.len(), 1);

        let q_mean = posterior.mean[0];
        let q_variance = posterior.variance[0];
        let p0_mean = full_prior.mean[0];
        let p0_variance = full_prior.variance[0];
        let pr_mean = reduced_prior.mean[0];
        let pr_variance = reduced_prior.variance[0];

        let standard_deviation = q_variance.sqrt();
        let lower = q_mean - 9.0 * standard_deviation;
        let upper = q_mean + 9.0 * standard_deviation;
        let intervals = 20_000usize;
        let step = (upper - lower) / intervals as f64;
        let mut weighted_sum = 0.0;

        for index in 0..=intervals {
            let value = lower + index as f64 * step;
            let log_integrand = log_normal_density(value, q_mean, q_variance)
                + log_normal_density(value, pr_mean, pr_variance)
                - log_normal_density(value, p0_mean, p0_variance);
            let weight = if index == 0 || index == intervals {
                1.0
            } else if index % 2 == 0 {
                2.0
            } else {
                4.0
            };
            weighted_sum += weight * log_integrand.exp();
        }

        ((step / 3.0) * weighted_sum).ln()
    }

    fn parameterization_descriptor() -> ParameterizationDescriptor {
        ParameterizationDescriptor {
            format_version: 1,
            architecture_id: "cfc/v1/layer-1/hidden-2".into(),
            groups: vec![
                ParameterGroupCoordinateDescriptor {
                    group_id: "input_projection".into(),
                    semantic_role: "projection".into(),
                    shape: vec![2, 2],
                    flattened_coordinate_names: vec![
                        "input_projection[0,0]".into(),
                        "input_projection[0,1]".into(),
                        "input_projection[1,0]".into(),
                        "input_projection[1,1]".into(),
                    ],
                },
                ParameterGroupCoordinateDescriptor {
                    group_id: "time_constants".into(),
                    semantic_role: "time_constant".into(),
                    shape: vec![2],
                    flattened_coordinate_names: vec!["tau[0]".into(), "tau[1]".into()],
                },
            ],
        }
    }

    #[test]
    fn canonical_parameterization_digest_is_stable_and_coordinate_order_sensitive() {
        let descriptor = parameterization_descriptor();
        let first = canonical_parameterization_digest(&descriptor).expect("canonical digest");
        let replay = canonical_parameterization_digest(&descriptor).expect("replay digest");
        assert_eq!(first, replay);
        assert!(is_sha256_digest(&first));

        let mut reordered = descriptor.clone();
        reordered.groups[0].flattened_coordinate_names.swap(0, 1);
        assert_ne!(
            first,
            canonical_parameterization_digest(&reordered).expect("reordered digest")
        );

        let mut changed_role = descriptor.clone();
        changed_role.groups[0].semantic_role = "recurrent_projection".into();
        assert_ne!(
            first,
            canonical_parameterization_digest(&changed_role).expect("changed role")
        );

        let mut changed_architecture = descriptor.clone();
        changed_architecture.architecture_id = "cfc/v2/layer-1/hidden-2".into();
        assert_ne!(
            first,
            canonical_parameterization_digest(&changed_architecture).expect("changed architecture")
        );
    }

    #[test]
    fn canonical_parameterization_digest_rejects_malformed_coordinate_manifests() {
        let mut descriptor = parameterization_descriptor();
        descriptor.groups[0].flattened_coordinate_names.pop();
        assert!(matches!(
            canonical_parameterization_digest(&descriptor),
            Err(ParameterizationDigestError::CoordinateCountMismatch { .. })
        ));

        let mut descriptor = parameterization_descriptor();
        descriptor.groups[0].flattened_coordinate_names[1] =
            descriptor.groups[0].flattened_coordinate_names[0].clone();
        assert!(matches!(
            canonical_parameterization_digest(&descriptor),
            Err(ParameterizationDigestError::DuplicateCoordinateName { .. })
        ));

        let mut descriptor = parameterization_descriptor();
        descriptor.groups[0].shape = vec![2, 0];
        assert!(matches!(
            canonical_parameterization_digest(&descriptor),
            Err(ParameterizationDigestError::ZeroShapeDimension { .. })
        ));

        let mut descriptor = parameterization_descriptor();
        descriptor.groups[1].group_id = descriptor.groups[0].group_id.clone();
        assert!(matches!(
            canonical_parameterization_digest(&descriptor),
            Err(ParameterizationDigestError::DuplicateGroupId(_))
        ));
    }

    #[test]
    fn canonical_parameterization_digest_binds_an_actual_bmr_comparison() {
        let parameterization_digest =
            canonical_parameterization_digest(&parameterization_descriptor())
                .expect("parameterization digest");
        let make_gaussian = |mean: &[f64], variance: &[f64]| DiagonalGaussian {
            parameterization_digest: parameterization_digest.clone(),
            mean: mean.to_vec(),
            variance: variance.to_vec(),
        };
        let posterior = make_gaussian(&[0.2, -0.1], &[0.4, 0.7]);
        let full_prior = make_gaussian(&[0.0, 0.0], &[1.0, 1.0]);
        let reduced_prior = make_gaussian(&[0.2, -0.1], &[0.2, 0.3]);
        let result = reduce_models(
            &posterior,
            &full_prior,
            &[ReducedModelHypothesis {
                model_id: "same-parameterization-reduced-prior".into(),
                prior_probability: 1.0,
                parameter_prior: reduced_prior,
            }],
        );
        assert!(result.is_ok());
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
        // Fixed reference and an independently evaluated Simpson integral.
        let quadrature =
            simpson_log_evidence_ratio(&posterior, &full_prior, &gaussian(&[0.2], &[0.2]));
        assert!((result.models[0].log_evidence_ratio - 0.3500401567802551).abs() < 1e-11);
        assert!((result.models[0].log_evidence_ratio - quadrature).abs() < 1e-9);
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
