// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full-covariance Gaussian synthetic likelihood for whole-sprinkling signatures.
//!
//! Each calibration row must represent one independent complete causal-set
//! sprinkling. Event pairs are not independent observations: they share events
//! and inherit transitive causal dependencies. The implementation therefore
//! estimates covariance across whole-sprinkling feature vectors.
//!
//! This module provides a transparent parametric baseline, not a generally
//! valid likelihood theorem. The Gaussian summary approximation, shrinkage,
//! ridge and finite calibration corpus must be tested with held-out posterior
//! coverage and posterior-predictive diagnostics before physical inference.

use crate::joint_signature::{JointCausalSetSignature, JointSignatureError};
use std::error::Error;
use std::fmt;

/// Errors fitting or evaluating a joint synthetic-likelihood model.
#[derive(Clone, Debug, PartialEq)]
pub enum JointLikelihoodError {
    InvalidDimension { dimension: usize },
    TooFewSprinklings { samples: usize, features: usize },
    EmptyCalibration,
    FeatureCount { expected: usize, actual: usize },
    NonFiniteFeature { sample: usize, feature: usize },
    InvalidShrinkage,
    InvalidRidge,
    DegenerateCalibration,
    CovarianceNotPositiveDefinite,
    InvalidPrior { index: usize },
    DuplicateDimension { dimension: usize },
    EmptyModelSet,
    Signature(JointSignatureError),
}

impl fmt::Display for JointLikelihoodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { dimension } => {
                write!(f, "spacetime dimension {dimension} is outside 2..=4")
            }
            Self::TooFewSprinklings { samples, features } => write!(
                f,
                "received {samples} whole-sprinkling samples for {features} features; require samples > features"
            ),
            Self::EmptyCalibration => write!(f, "synthetic-likelihood calibration is empty"),
            Self::FeatureCount { expected, actual } => {
                write!(
                    f,
                    "feature vector has {actual} entries; expected {expected}"
                )
            }
            Self::NonFiniteFeature { sample, feature } => {
                write!(f, "calibration feature [{sample}][{feature}] is not finite")
            }
            Self::InvalidShrinkage => {
                write!(f, "diagonal shrinkage must be finite and in [0,1]")
            }
            Self::InvalidRidge => {
                write!(f, "relative covariance ridge must be finite and positive")
            }
            Self::DegenerateCalibration => {
                write!(
                    f,
                    "calibration has zero or non-finite average feature variance"
                )
            }
            Self::CovarianceNotPositiveDefinite => {
                write!(
                    f,
                    "regularized calibration covariance is not positive definite"
                )
            }
            Self::InvalidPrior { index } => {
                write!(
                    f,
                    "model prior at index {index} must be finite and positive"
                )
            }
            Self::DuplicateDimension { dimension } => {
                write!(f, "spacetime dimension {dimension} appears more than once")
            }
            Self::EmptyModelSet => write!(f, "no candidate dimension models were supplied"),
            Self::Signature(error) => write!(f, "joint signature failed: {error}"),
        }
    }
}

impl Error for JointLikelihoodError {}

impl From<JointSignatureError> for JointLikelihoodError {
    fn from(value: JointSignatureError) -> Self {
        Self::Signature(value)
    }
}

/// Fitted Gaussian synthetic likelihood for one candidate dimension.
///
/// The lower-triangular Cholesky factor is stored instead of a covariance
/// inverse. Log likelihood and Mahalanobis distance use triangular solves.
#[derive(Clone, Debug, PartialEq)]
pub struct JointGaussianCalibration {
    pub spacetime_dimension: usize,
    pub whole_sprinkling_count: usize,
    pub feature_count: usize,
    pub mean: Vec<f64>,
    pub cholesky_lower: Vec<Vec<f64>>,
    pub log_determinant_covariance: f64,
    pub diagonal_shrinkage: f64,
    pub relative_ridge: f64,
    pub average_unregularized_variance: f64,
}

impl JointGaussianCalibration {
    /// Fit a full-covariance Gaussian synthetic likelihood.
    ///
    /// The sample matrix has one row per independent complete causal set and
    /// one column per ILR-plus-chain summary feature. Sample covariance uses
    /// denominator R-1. Diagonal shrinkage is
    ///
    ///     Sigma_s = (1-lambda) S + lambda diag(S),
    ///
    /// followed by epsilon * mean(diag(S)) * I. The ridge is explicit and
    /// recorded in the fitted object. No pseudo-inverse is used.
    pub fn fit_whole_sprinkling_features(
        spacetime_dimension: usize,
        samples: &[Vec<f64>],
        diagonal_shrinkage: f64,
        relative_ridge: f64,
    ) -> Result<Self, JointLikelihoodError> {
        if !(2..=4).contains(&spacetime_dimension) {
            return Err(JointLikelihoodError::InvalidDimension {
                dimension: spacetime_dimension,
            });
        }
        if samples.is_empty() {
            return Err(JointLikelihoodError::EmptyCalibration);
        }
        if !diagonal_shrinkage.is_finite() || !(0.0..=1.0).contains(&diagonal_shrinkage) {
            return Err(JointLikelihoodError::InvalidShrinkage);
        }
        if !relative_ridge.is_finite() || relative_ridge <= 0.0 {
            return Err(JointLikelihoodError::InvalidRidge);
        }

        let feature_count = samples[0].len();
        if feature_count == 0 {
            return Err(JointLikelihoodError::FeatureCount {
                expected: 1,
                actual: 0,
            });
        }
        if samples.len() <= feature_count {
            return Err(JointLikelihoodError::TooFewSprinklings {
                samples: samples.len(),
                features: feature_count,
            });
        }

        for (sample_index, sample) in samples.iter().enumerate() {
            if sample.len() != feature_count {
                return Err(JointLikelihoodError::FeatureCount {
                    expected: feature_count,
                    actual: sample.len(),
                });
            }
            for (feature_index, value) in sample.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(JointLikelihoodError::NonFiniteFeature {
                        sample: sample_index,
                        feature: feature_index,
                    });
                }
            }
        }

        let sample_count = samples.len();
        let mut mean = vec![0.0; feature_count];
        for sample in samples {
            for (feature, value) in sample.iter().copied().enumerate() {
                mean[feature] += value / sample_count as f64;
            }
        }

        let mut covariance = vec![vec![0.0; feature_count]; feature_count];
        for sample in samples {
            for row in 0..feature_count {
                let left = sample[row] - mean[row];
                for column in row..feature_count {
                    covariance[row][column] +=
                        left * (sample[column] - mean[column]) / (sample_count - 1) as f64;
                }
            }
        }
        for row in 0..feature_count {
            for column in (row + 1)..feature_count {
                covariance[column][row] = covariance[row][column];
            }
        }

        let average_variance = (0..feature_count)
            .map(|index| covariance[index][index])
            .sum::<f64>()
            / feature_count as f64;
        if !average_variance.is_finite() || average_variance <= f64::MIN_POSITIVE {
            return Err(JointLikelihoodError::DegenerateCalibration);
        }

        let ridge = relative_ridge * average_variance;
        for row in 0..feature_count {
            for column in 0..feature_count {
                if row != column {
                    covariance[row][column] *= 1.0 - diagonal_shrinkage;
                }
            }
            covariance[row][row] += ridge;
        }

        let cholesky_lower = cholesky_decompose(&covariance)?;
        let log_determinant_covariance = 2.0
            * (0..feature_count)
                .map(|index| cholesky_lower[index][index].ln())
                .sum::<f64>();
        if !log_determinant_covariance.is_finite() {
            return Err(JointLikelihoodError::CovarianceNotPositiveDefinite);
        }

        Ok(Self {
            spacetime_dimension,
            whole_sprinkling_count: sample_count,
            feature_count,
            mean,
            cholesky_lower,
            log_determinant_covariance,
            diagonal_shrinkage,
            relative_ridge,
            average_unregularized_variance: average_variance,
        })
    }

    /// Squared Mahalanobis distance under the full regularized covariance.
    pub fn mahalanobis_squared(&self, features: &[f64]) -> Result<f64, JointLikelihoodError> {
        if features.len() != self.feature_count {
            return Err(JointLikelihoodError::FeatureCount {
                expected: self.feature_count,
                actual: features.len(),
            });
        }
        let mut whitened = vec![0.0; self.feature_count];
        for index in 0..self.feature_count {
            if !features[index].is_finite() {
                return Err(JointLikelihoodError::NonFiniteFeature {
                    sample: 0,
                    feature: index,
                });
            }
            let residual = features[index] - self.mean[index];
            let previous = (0..index)
                .map(|column| self.cholesky_lower[index][column] * whitened[column])
                .sum::<f64>();
            whitened[index] = (residual - previous) / self.cholesky_lower[index][index];
        }
        let distance = whitened.iter().map(|value| value * value).sum::<f64>();
        if !distance.is_finite() || distance < 0.0 {
            return Err(JointLikelihoodError::CovarianceNotPositiveDefinite);
        }
        Ok(distance)
    }

    /// Gaussian synthetic log likelihood for one whole-sprinkling signature.
    pub fn log_likelihood(&self, features: &[f64]) -> Result<f64, JointLikelihoodError> {
        let mahalanobis = self.mahalanobis_squared(features)?;
        let normalizer = self.feature_count as f64 * (2.0 * std::f64::consts::PI).ln()
            + self.log_determinant_covariance;
        let log_likelihood = -0.5 * (normalizer + mahalanobis);
        if !log_likelihood.is_finite() {
            return Err(JointLikelihoodError::CovarianceNotPositiveDefinite);
        }
        Ok(log_likelihood)
    }
}

/// One candidate dimension and its whole-sprinkling synthetic likelihood.
#[derive(Clone, Debug, PartialEq)]
pub struct JointDimensionModel {
    pub spacetime_dimension: usize,
    pub prior_probability: f64,
    pub calibration: JointGaussianCalibration,
}

/// Posterior probability for one candidate spacetime dimension.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimensionProbability {
    pub spacetime_dimension: usize,
    pub probability: f64,
    pub log_likelihood: f64,
}

/// Dimension posterior from calibrated Gaussian synthetic likelihoods.
///
/// This is a posterior over the finite supplied candidate set only. It is not
/// an out-of-family detector. A production evaluator still needs posterior
/// predictive checks and a calibrated abstention threshold from sealed
/// whole-sprinkling validation replicates.
#[derive(Clone, Debug, PartialEq)]
pub struct JointDimensionPosterior {
    pub probabilities: Vec<DimensionProbability>,
    pub map_dimension: usize,
    pub entropy_nats: f64,
}

/// Infer dimension by evaluating each model on its own candidate-D ILR feature
/// vector and normalizing prior-weighted synthetic likelihoods.
pub fn infer_joint_dimension(
    signature: &JointCausalSetSignature,
    models: &[JointDimensionModel],
) -> Result<JointDimensionPosterior, JointLikelihoodError> {
    if models.is_empty() {
        return Err(JointLikelihoodError::EmptyModelSet);
    }

    let mut seen_dimensions = Vec::with_capacity(models.len());
    let mut log_weights = Vec::with_capacity(models.len());
    let mut log_likelihoods = Vec::with_capacity(models.len());

    for (index, model) in models.iter().enumerate() {
        if !model.prior_probability.is_finite() || model.prior_probability <= 0.0 {
            return Err(JointLikelihoodError::InvalidPrior { index });
        }
        if seen_dimensions.contains(&model.spacetime_dimension) {
            return Err(JointLikelihoodError::DuplicateDimension {
                dimension: model.spacetime_dimension,
            });
        }
        if model.calibration.spacetime_dimension != model.spacetime_dimension {
            return Err(JointLikelihoodError::InvalidDimension {
                dimension: model.spacetime_dimension,
            });
        }
        seen_dimensions.push(model.spacetime_dimension);

        let features = signature.ilr_features_for_dimension(model.spacetime_dimension)?;
        let log_likelihood = model.calibration.log_likelihood(&features)?;
        log_likelihoods.push(log_likelihood);
        log_weights.push(model.prior_probability.ln() + log_likelihood);
    }

    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let unnormalized: Vec<f64> = log_weights
        .iter()
        .map(|weight| (weight - maximum).exp())
        .collect();
    let normalizer = unnormalized.iter().sum::<f64>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(JointLikelihoodError::CovarianceNotPositiveDefinite);
    }

    let probabilities: Vec<DimensionProbability> = models
        .iter()
        .enumerate()
        .map(|(index, model)| DimensionProbability {
            spacetime_dimension: model.spacetime_dimension,
            probability: unnormalized[index] / normalizer,
            log_likelihood: log_likelihoods[index],
        })
        .collect();
    let map_dimension = probabilities
        .iter()
        .max_by(|left, right| left.probability.total_cmp(&right.probability))
        .map(|entry| entry.spacetime_dimension)
        .ok_or(JointLikelihoodError::EmptyModelSet)?;
    let entropy_nats = probabilities
        .iter()
        .filter(|entry| entry.probability > 0.0)
        .map(|entry| -entry.probability * entry.probability.ln())
        .sum::<f64>();

    Ok(JointDimensionPosterior {
        probabilities,
        map_dimension,
        entropy_nats,
    })
}

fn cholesky_decompose(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, JointLikelihoodError> {
    let dimension = matrix.len();
    let mut lower = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let correction = (0..column)
                .map(|index| lower[row][index] * lower[column][index])
                .sum::<f64>();
            let residual = matrix[row][column] - correction;
            if row == column {
                if !residual.is_finite() || residual <= f64::MIN_POSITIVE {
                    return Err(JointLikelihoodError::CovarianceNotPositiveDefinite);
                }
                lower[row][column] = residual.sqrt();
            } else {
                let diagonal = lower[column][column];
                if !diagonal.is_finite() || diagonal <= 0.0 {
                    return Err(JointLikelihoodError::CovarianceNotPositiveDefinite);
                }
                lower[row][column] = residual / diagonal;
                if !lower[row][column].is_finite() {
                    return Err(JointLikelihoodError::CovarianceNotPositiveDefinite);
                }
            }
        }
    }
    Ok(lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn correlated_samples() -> Vec<Vec<f64>> {
        let mut samples = Vec::new();
        for index in 0..64 {
            let x = index as f64 / 63.0 - 0.5;
            let y = 2.0 * x + 0.03 * ((index * 7 % 13) as f64 - 6.0);
            let z = -0.4 * x + 0.02 * ((index * 11 % 17) as f64 - 8.0);
            samples.push(vec![x, y, z]);
        }
        samples
    }

    #[test]
    fn fit_uses_full_correlated_covariance_and_cholesky() {
        let calibration = JointGaussianCalibration::fit_whole_sprinkling_features(
            2,
            &correlated_samples(),
            0.1,
            1.0e-6,
        )
        .unwrap();
        assert_eq!(calibration.whole_sprinkling_count, 64);
        assert_eq!(calibration.feature_count, 3);
        assert!(calibration.cholesky_lower[1][0].abs() > 1.0e-4);
        assert!(calibration.log_determinant_covariance.is_finite());
        assert!(
            calibration
                .log_likelihood(&[0.0, 0.0, 0.0])
                .unwrap()
                .is_finite()
        );
    }

    #[test]
    fn diagonal_shrinkage_and_ridge_regularize_collinear_features() {
        let samples: Vec<Vec<f64>> = (0..32)
            .map(|index| {
                let value = index as f64 - 15.5;
                vec![value, 2.0 * value, -3.0 * value]
            })
            .collect();
        let calibration =
            JointGaussianCalibration::fit_whole_sprinkling_features(3, &samples, 0.25, 1.0e-5)
                .unwrap();
        assert!(
            calibration
                .log_likelihood(&[0.0, 0.0, 0.0])
                .unwrap()
                .is_finite()
        );
    }

    #[test]
    fn fit_rejects_too_few_whole_sprinkling_replicates() {
        assert!(matches!(
            JointGaussianCalibration::fit_whole_sprinkling_features(
                2,
                &[vec![0.0, 1.0], vec![1.0, 0.0]],
                0.2,
                1.0e-5
            ),
            Err(JointLikelihoodError::TooFewSprinklings { .. })
        ));
    }

    #[test]
    fn fit_rejects_zero_variance_calibration() {
        let samples = vec![vec![1.0, 2.0]; 16];
        assert_eq!(
            JointGaussianCalibration::fit_whole_sprinkling_features(2, &samples, 0.2, 1.0e-5),
            Err(JointLikelihoodError::DegenerateCalibration)
        );
    }

    #[test]
    fn posterior_is_normalized_and_uses_candidate_dimension_features() {
        let samples_2d = correlated_samples();
        let samples_3d: Vec<Vec<f64>> = samples_2d
            .iter()
            .map(|sample| vec![sample[0] * 1.2, sample[1] * 0.8, sample[2] + 0.2])
            .collect();
        let model_2d = JointDimensionModel {
            spacetime_dimension: 2,
            prior_probability: 0.5,
            calibration: JointGaussianCalibration::fit_whole_sprinkling_features(
                2,
                &samples_2d,
                0.2,
                1.0e-6,
            )
            .unwrap(),
        };
        let model_3d = JointDimensionModel {
            spacetime_dimension: 3,
            prior_probability: 0.5,
            calibration: JointGaussianCalibration::fit_whole_sprinkling_features(
                3,
                &samples_3d,
                0.2,
                1.0e-6,
            )
            .unwrap(),
        };
        let signature = JointCausalSetSignature {
            event_count: 64,
            profile_bin_count: 2,
            normalized_interval_profile: vec![0.2, 0.3, 0.5],
            longest_chain_length: 12,
        };
        let posterior = infer_joint_dimension(&signature, &[model_2d, model_3d]).unwrap();
        let probability_sum: f64 = posterior
            .probabilities
            .iter()
            .map(|entry| entry.probability)
            .sum();
        assert!((probability_sum - 1.0).abs() < 1.0e-12);
        assert!(posterior.entropy_nats.is_finite());
    }
}
