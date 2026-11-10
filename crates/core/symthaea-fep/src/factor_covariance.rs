// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit low-rank-plus-diagonal covariance for correlated diagnostics.
//!
//! This module represents Sigma = D + U U^T, where D is strictly positive
//! diagonal channel noise and the columns of U are declared unit-variance
//! shared latent factors. It computes quadratic forms using the Woodbury
//! identity and a Cholesky solve on the small factor-space system:
//!
//! q = delta^T D^-1 delta
//!     - v^T (I + U^T D^-1 U)^-1 v
//! v = U^T D^-1 delta.
//!
//! It never explicitly forms Sigma inverse, adds jitter, clips eigenvalues,
//! truncates factors, or silently substitutes a diagonal model. The supplied
//! factors and provenance are caller claims, not authenticated calibration
//! truth. A residual-correlation check can flag one important form of omitted
//! common-mode structure, but passing it is not proof of general adequacy.
//!
//! This is an isolated mathematical kernel. It deliberately does not depend on
//! the multivariate Gaussian decision API in PR #11391; composition is gated on
//! that PR's exact-head qualification.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

const MAX_FEATURES: usize = 4096;
const MAX_FACTORS: usize = 64;
const DIGEST_LENGTH: usize = 64;
const NUMERIC_TOLERANCE: f64 = 1e-12;

/// Declared covariance model Sigma = D + U U^T.
///
/// Rows of loadings correspond to feature_ids. Columns correspond to factor_ids.
/// Latent factors are assumed mutually independent, zero-mean and unit-variance;
/// this is part of the model semantics, not something inferred by this module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactorCovarianceModel {
    pub model_version: String,
    pub feature_ids: Vec<String>,
    pub feature_units: Vec<String>,
    pub diagonal_variances: Vec<f64>,
    pub factor_ids: Vec<String>,
    pub loadings: Vec<Vec<f64>>,
    /// Digest of caller-supplied calibration/provenance material.
    /// A well-formed digest is not proof that the referenced material is true.
    pub calibration_provenance_digest: String,
}

/// Result of a residual-correlation adequacy screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorAdequacyStatus {
    /// The registered minimum calibration sample count was not reached.
    InsufficientEvidence,
    /// No excessive residual correlation was detected by this screen only.
    NoDetectedResidualCorrelation,
    /// Residual correlation exceeds the registered tolerance.
    Inadequate,
}

/// Caller-supplied residual evidence for screening omitted common-mode structure.
///
/// The values must be computed from observations not used to fit the factor
/// model. This type does not authenticate chronology or sample independence;
/// the enclosing evidence protocol must provide those guarantees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactorResidualAdequacyEvidence {
    pub calibration_observation_count: usize,
    pub minimum_observation_count: usize,
    pub maximum_absolute_residual_correlation: f64,
    pub maximum_allowed_residual_correlation: f64,
    pub evidence_digest: String,
}

/// Explicit report; a passing result is not a universal adequacy certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactorResidualAdequacyReport {
    pub status: FactorAdequacyStatus,
    pub calibration_observation_count: usize,
    pub maximum_absolute_residual_correlation: f64,
    pub maximum_allowed_residual_correlation: f64,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum FactorCovarianceError {
    #[error("factor covariance model version must not be empty")]
    EmptyModelVersion,
    #[error("feature count must be in 1..={MAX_FEATURES}, received {actual}")]
    InvalidFeatureCount { actual: usize },
    #[error("feature identifier at index {index} is empty or duplicated")]
    InvalidFeatureId { index: usize },
    #[error("feature unit at index {index} is empty")]
    InvalidFeatureUnit { index: usize },
    #[error("diagonal variance count {actual} does not match feature count {expected}")]
    DiagonalDimensionMismatch { expected: usize, actual: usize },
    #[error("feature-unit count {actual} does not match feature count {expected}")]
    FeatureUnitCountMismatch { expected: usize, actual: usize },
    #[error("diagonal variance at index {index} must be finite and strictly positive")]
    InvalidDiagonalVariance { index: usize },
    #[error("factor count {actual} must be less than feature count {dimension} and no greater than {maximum}")]
    InvalidFactorRank { actual: usize, dimension: usize, maximum: usize },
    #[error("factor identifier at index {index} is empty or duplicated")]
    InvalidFactorId { index: usize },
    #[error("loading row count {actual} does not match feature count {expected}")]
    LoadingRowCountMismatch { expected: usize, actual: usize },
    #[error("loading row {row} has {actual} columns, expected {expected}")]
    LoadingColumnCountMismatch { row: usize, expected: usize, actual: usize },
    #[error("loading at row {row}, column {column} is non-finite")]
    NonFiniteLoading { row: usize, column: usize },
    #[error("calibration provenance digest must be 64 lowercase hexadecimal characters")]
    InvalidProvenanceDigest,
    #[error("residual vector has {actual} entries, expected {expected}")]
    ResidualDimensionMismatch { expected: usize, actual: usize },
    #[error("residual at index {index} is non-finite")]
    NonFiniteResidual { index: usize },
    #[error("intermediate covariance calculation became non-finite")]
    NonFiniteIntermediate,
    #[error("factor-space system is not numerically positive definite")]
    FactorSystemNotPositiveDefinite,
    #[error("computed Mahalanobis quadratic form is negative or non-finite")]
    InvalidQuadraticForm,
    #[error("residual adequacy sample count must be positive")]
    InvalidAdequacySampleCount,
    #[error("residual-correlation limits must be finite and lie in [0, 1]")]
    InvalidResidualCorrelationLimit,
    #[error("residual adequacy evidence digest must be 64 lowercase hexadecimal characters")]
    InvalidAdequacyEvidenceDigest,
}

impl FactorCovarianceModel {
    /// Validate dimensions, identifiers, positive diagonal variances and digests.
    pub fn validate(&self) -> Result<(), FactorCovarianceError> {
        let dimension = self.feature_ids.len();
        let rank = self.factor_ids.len();

        if self.model_version.trim().is_empty() {
            return Err(FactorCovarianceError::EmptyModelVersion);
        }
        if dimension == 0 || dimension > MAX_FEATURES {
            return Err(FactorCovarianceError::InvalidFeatureCount { actual: dimension });
        }
        if self.feature_units.len() != dimension {
            return Err(FactorCovarianceError::FeatureUnitCountMismatch {
                expected: dimension,
                actual: self.feature_units.len(),
            });
        }
        if self.diagonal_variances.len() != dimension {
            return Err(FactorCovarianceError::DiagonalDimensionMismatch {
                expected: dimension,
                actual: self.diagonal_variances.len(),
            });
        }
        if rank >= dimension || rank > MAX_FACTORS {
            return Err(FactorCovarianceError::InvalidFactorRank {
                actual: rank,
                dimension,
                maximum: MAX_FACTORS,
            });
        }
        if self.loadings.len() != dimension {
            return Err(FactorCovarianceError::LoadingRowCountMismatch {
                expected: dimension,
                actual: self.loadings.len(),
            });
        }

        let mut feature_ids = BTreeSet::new();
        for (index, identifier) in self.feature_ids.iter().enumerate() {
            if identifier.trim().is_empty() || !feature_ids.insert(identifier.as_str()) {
                return Err(FactorCovarianceError::InvalidFeatureId { index });
            }
            if self.feature_units[index].trim().is_empty() {
                return Err(FactorCovarianceError::InvalidFeatureUnit { index });
            }
            let variance = self.diagonal_variances[index];
            if !variance.is_finite() || variance <= 0.0 {
                return Err(FactorCovarianceError::InvalidDiagonalVariance { index });
            }
        }

        let mut factor_ids = BTreeSet::new();
        for (index, identifier) in self.factor_ids.iter().enumerate() {
            if identifier.trim().is_empty() || !factor_ids.insert(identifier.as_str()) {
                return Err(FactorCovarianceError::InvalidFactorId { index });
            }
        }

        for (row, values) in self.loadings.iter().enumerate() {
            if values.len() != rank {
                return Err(FactorCovarianceError::LoadingColumnCountMismatch {
                    row,
                    expected: rank,
                    actual: values.len(),
                });
            }
            for (column, value) in values.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(FactorCovarianceError::NonFiniteLoading { row, column });
                }
            }
        }

        if !is_lowercase_sha256(&self.calibration_provenance_digest) {
            return Err(FactorCovarianceError::InvalidProvenanceDigest);
        }
        Ok(())
    }

    pub fn dimension(&self) -> usize {
        self.feature_ids.len()
    }

    pub fn rank(&self) -> usize {
        self.factor_ids.len()
    }

    /// Compute the canonical covariance-model digest.
    ///
    /// The digest is invariant to feature row permutation, factor column
    /// permutation and factor sign flips. It still binds feature/factor IDs,
    /// units, diagonal variances, model version and provenance digest.
    pub fn model_digest(&self) -> Result<String, FactorCovarianceError> {
        self.validate()?;

        let mut feature_order: Vec<usize> = (0..self.dimension()).collect();
        feature_order.sort_by(|left, right| {
            self.feature_ids[*left].cmp(&self.feature_ids[*right])
        });
        let mut factor_order: Vec<usize> = (0..self.rank()).collect();
        factor_order.sort_by(|left, right| self.factor_ids[*left].cmp(&self.factor_ids[*right]));

        let mut hasher = Sha256::new();
        hasher.update(b"symthaea-factor-covariance-d-plus-uu-transpose-v1\0");
        hash_string(&mut hasher, &self.model_version);
        hash_string(&mut hasher, &self.calibration_provenance_digest);
        hasher.update((self.dimension() as u64).to_le_bytes());
        hasher.update((self.rank() as u64).to_le_bytes());

        for feature_index in &feature_order {
            hash_string(&mut hasher, &self.feature_ids[*feature_index]);
            hash_string(&mut hasher, &self.feature_units[*feature_index]);
            hasher.update(canonical_float_bits(self.diagonal_variances[*feature_index]).to_le_bytes());
        }

        for factor_index in &factor_order {
            hash_string(&mut hasher, &self.factor_ids[*factor_index]);
            let sign = canonical_factor_sign(self, *factor_index, &feature_order);
            for feature_index in &feature_order {
                let loading = self.loadings[*feature_index][*factor_index] * sign;
                hasher.update(canonical_float_bits(loading).to_le_bytes());
            }
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Materialize Sigma = D + U U^T for small-fixture reference calculations.
    ///
    /// The production Mahalanobis path below does not call this method.
    pub fn dense_covariance(&self) -> Result<Vec<Vec<f64>>, FactorCovarianceError> {
        self.validate()?;
        let dimension = self.dimension();
        let rank = self.rank();
        let mut covariance = vec![vec![0.0; dimension]; dimension];

        for index in 0..dimension {
            covariance[index][index] = self.diagonal_variances[index];
        }
        for row in 0..dimension {
            for column in 0..dimension {
                let mut shared = 0.0;
                for factor in 0..rank {
                    shared += self.loadings[row][factor] * self.loadings[column][factor];
                }
                covariance[row][column] += shared;
                if !covariance[row][column].is_finite() {
                    return Err(FactorCovarianceError::NonFiniteIntermediate);
                }
            }
        }
        Ok(covariance)
    }

    /// Compute delta^T (D + U U^T)^-1 delta via Woodbury.
    ///
    /// The only Cholesky factorization is of the r x r matrix
    /// I + U^T D^-1 U. No d x d inverse or dense covariance is constructed.
    pub fn mahalanobis_squared(
        &self,
        residual: &[f64],
    ) -> Result<f64, FactorCovarianceError> {
        self.validate()?;
        if residual.len() != self.dimension() {
            return Err(FactorCovarianceError::ResidualDimensionMismatch {
                expected: self.dimension(),
                actual: residual.len(),
            });
        }
        for (index, value) in residual.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(FactorCovarianceError::NonFiniteResidual { index });
            }
        }

        let dimension = self.dimension();
        let rank = self.rank();
        let mut inverse_diagonal_residual = vec![0.0; dimension];
        let mut diagonal_quadratic = 0.0;

        for index in 0..dimension {
            let value = residual[index] / self.diagonal_variances[index];
            inverse_diagonal_residual[index] = value;
            diagonal_quadratic += residual[index] * value;
        }
        if !diagonal_quadratic.is_finite() {
            return Err(FactorCovarianceError::NonFiniteIntermediate);
        }

        if rank == 0 {
            return Ok(diagonal_quadratic);
        }

        let mut factor_system = vec![vec![0.0; rank]; rank];
        let mut projected_residual = vec![0.0; rank];
        for row in 0..rank {
            factor_system[row][row] = 1.0;
        }

        for feature in 0..dimension {
            let inverse_variance = 1.0 / self.diagonal_variances[feature];
            for row_factor in 0..rank {
                let row_loading = self.loadings[feature][row_factor];
                projected_residual[row_factor] +=
                    row_loading * inverse_diagonal_residual[feature];
                for column_factor in row_factor..rank {
                    let contribution = row_loading
                        * self.loadings[feature][column_factor]
                        * inverse_variance;
                    factor_system[row_factor][column_factor] += contribution;
                    if row_factor != column_factor {
                        factor_system[column_factor][row_factor] += contribution;
                    }
                }
            }
        }

        if projected_residual.iter().any(|value| !value.is_finite())
            || factor_system.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(FactorCovarianceError::NonFiniteIntermediate);
        }

        let factor_cholesky = cholesky_factor(&factor_system)?;
        let solved = solve_cholesky(&factor_cholesky, &projected_residual)?;
        let correction = dot(&projected_residual, &solved);
        let result = diagonal_quadratic - correction;

        if !correction.is_finite() || !result.is_finite() || result < 0.0 {
            return Err(FactorCovarianceError::InvalidQuadraticForm);
        }
        Ok(result)
    }
}

/// Screen held-out residual correlations for one registered factor model.
///
/// This is deliberately a narrow adequacy check. It can identify a declared
/// residual-correlation failure; a pass does not prove Gaussianity, stationarity,
/// independence of samples, causal correctness or calibration of the factors.
pub fn assess_residual_correlation_adequacy(
    evidence: &FactorResidualAdequacyEvidence,
) -> Result<FactorResidualAdequacyReport, FactorCovarianceError> {
    if evidence.minimum_observation_count == 0 {
        return Err(FactorCovarianceError::InvalidAdequacySampleCount);
    }
    let observed = evidence.maximum_absolute_residual_correlation;
    let allowed = evidence.maximum_allowed_residual_correlation;
    if !observed.is_finite()
        || !allowed.is_finite()
        || !(0.0..=1.0).contains(&observed)
        || !(0.0..=1.0).contains(&allowed)
    {
        return Err(FactorCovarianceError::InvalidResidualCorrelationLimit);
    }
    if !is_lowercase_sha256(&evidence.evidence_digest) {
        return Err(FactorCovarianceError::InvalidAdequacyEvidenceDigest);
    }

    let status = if evidence.calibration_observation_count < evidence.minimum_observation_count {
        FactorAdequacyStatus::InsufficientEvidence
    } else if observed > allowed {
        FactorAdequacyStatus::Inadequate
    } else {
        FactorAdequacyStatus::NoDetectedResidualCorrelation
    };

    Ok(FactorResidualAdequacyReport {
        status,
        calibration_observation_count: evidence.calibration_observation_count,
        maximum_absolute_residual_correlation: observed,
        maximum_allowed_residual_correlation: allowed,
        evidence_digest: evidence.evidence_digest.clone(),
    })
}

fn cholesky_factor(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FactorCovarianceError> {
    let dimension = matrix.len();
    let mut lower = vec![vec![0.0; dimension]; dimension];

    for row in 0..dimension {
        if matrix[row].len() != dimension {
            return Err(FactorCovarianceError::FactorSystemNotPositiveDefinite);
        }
        for column in 0..=row {
            let mut value = matrix[row][column];
            for inner in 0..column {
                value -= lower[row][inner] * lower[column][inner];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(FactorCovarianceError::FactorSystemNotPositiveDefinite);
                }
                lower[row][column] = value.sqrt();
            } else {
                let pivot = lower[column][column];
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(FactorCovarianceError::FactorSystemNotPositiveDefinite);
                }
                lower[row][column] = value / pivot;
            }
        }
    }
    Ok(lower)
}

fn solve_cholesky(
    lower: &[Vec<f64>],
    right_hand_side: &[f64],
) -> Result<Vec<f64>, FactorCovarianceError> {
    let dimension = lower.len();
    if right_hand_side.len() != dimension {
        return Err(FactorCovarianceError::FactorSystemNotPositiveDefinite);
    }
    let mut forward = vec![0.0; dimension];
    for row in 0..dimension {
        let mut value = right_hand_side[row];
        for column in 0..row {
            value -= lower[row][column] * forward[column];
        }
        forward[row] = value / lower[row][row];
    }

    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut value = forward[row];
        for column in (row + 1)..dimension {
            value -= lower[column][row] * solution[column];
        }
        solution[row] = value / lower[row][row];
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return Err(FactorCovarianceError::NonFiniteIntermediate);
    }
    Ok(solution)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn canonical_float_bits(value: f64) -> u64 {
    if value == 0.0 { 0.0_f64.to_bits() } else { value.to_bits() }
}

fn canonical_factor_sign(
    model: &FactorCovarianceModel,
    factor_index: usize,
    feature_order: &[usize],
) -> f64 {
    for feature_index in feature_order {
        let loading = model.loadings[*feature_index][factor_index];
        if loading != 0.0 {
            return if loading < 0.0 { -1.0 } else { 1.0 };
        }
    }
    1.0
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == DIGEST_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(dimension: usize, rank: usize) -> FactorCovarianceModel {
        let feature_ids = (0..dimension)
            .map(|index| format!("sensor-{index:04}"))
            .collect::<Vec<_>>();
        let feature_units = (0..dimension)
            .map(|_| "normalized".to_owned())
            .collect::<Vec<_>>();
        let diagonal_variances = (0..dimension)
            .map(|index| ((index % 5) as f64 * 0.35 - 0.7).exp())
            .collect::<Vec<_>>();
        let factor_ids = (0..rank)
            .map(|index| format!("shared-mode-{index:03}"))
            .collect::<Vec<_>>();
        let loadings = (0..dimension)
            .map(|row| {
                (0..rank)
                    .map(|column| {
                        (((row + 1) * (column + 2)) as f64).sin() * 0.04
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        FactorCovarianceModel {
            model_version: "factor-covariance-v1".to_owned(),
            feature_ids,
            feature_units,
            diagonal_variances,
            factor_ids,
            loadings,
            calibration_provenance_digest: "a".repeat(64),
        }
    }

    fn dense_reference(model: &FactorCovarianceModel, residual: &[f64]) -> f64 {
        let dimension = model.dimension();
        let rank = model.rank();
        let mut sigma = vec![vec![0.0; dimension]; dimension];
        for row in 0..dimension {
            sigma[row][row] = model.diagonal_variances[row];
            for column in 0..dimension {
                for factor in 0..rank {
                    sigma[row][column] +=
                        model.loadings[row][factor] * model.loadings[column][factor];
                }
            }
        }

        // Independent dense Cholesky implementation for the small-fixture oracle.
        let mut lower = vec![vec![0.0; dimension]; dimension];
        for row in 0..dimension {
            for column in 0..=row {
                let mut value = sigma[row][column];
                for inner in 0..column {
                    value -= lower[row][inner] * lower[column][inner];
                }
                if row == column {
                    lower[row][column] = value.sqrt();
                } else {
                    lower[row][column] = value / lower[column][column];
                }
            }
        }

        let mut forward = vec![0.0; dimension];
        for row in 0..dimension {
            let mut value = residual[row];
            for column in 0..row {
                value -= lower[row][column] * forward[column];
            }
            forward[row] = value / lower[row][row];
        }

        let mut solution = vec![0.0; dimension];
        for row in (0..dimension).rev() {
            let mut value = forward[row];
            for column in (row + 1)..dimension {
                value -= lower[column][row] * solution[column];
            }
            solution[row] = value / lower[row][row];
        }
        residual.iter().zip(solution).map(|(a, b)| a * b).sum()
    }

    fn residual(dimension: usize) -> Vec<f64> {
        (0..dimension)
            .map(|index| ((index + 3) as f64 * 0.17).cos())
            .collect()
    }

    #[test]
    fn woodbury_matches_independent_dense_cholesky_across_dimensions_and_ranks() {
        let cases = [(2, 1), (3, 2), (8, 4), (32, 4), (128, 8), (512, 8)];
        for (dimension, rank) in cases {
            let model = fixture(dimension, rank);
            let residual = residual(dimension);
            let actual = model.mahalanobis_squared(&residual).unwrap();
            let expected = dense_reference(&model, &residual);
            let tolerance = 2e-10 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "d={dimension} r={rank}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn empty_factor_set_matches_diagonal_gaussian() {
        let model = fixture(8, 0);
        let residual = residual(8);
        let expected = residual
            .iter()
            .zip(&model.diagonal_variances)
            .map(|(value, variance)| value * value / variance)
            .sum::<f64>();
        let actual = model.mahalanobis_squared(&residual).unwrap();
        assert!((actual - expected).abs() <= NUMERIC_TOLERANCE * expected.max(1.0));
    }

    #[test]
    fn feature_permutation_preserves_mahalanobis_distance() {
        let model = fixture(8, 3);
        let residual = residual(8);
        let permutation = [5, 2, 7, 0, 6, 1, 4, 3];
        let mut permuted = model.clone();
        permuted.feature_ids = permutation.iter().map(|index| model.feature_ids[*index].clone()).collect();
        permuted.feature_units = permutation.iter().map(|index| model.feature_units[*index].clone()).collect();
        permuted.diagonal_variances = permutation.iter().map(|index| model.diagonal_variances[*index]).collect();
        permuted.loadings = permutation.iter().map(|index| model.loadings[*index].clone()).collect();
        let permuted_residual = permutation.iter().map(|index| residual[*index]).collect::<Vec<_>>();

        let expected = model.mahalanobis_squared(&residual).unwrap();
        let actual = permuted.mahalanobis_squared(&permuted_residual).unwrap();
        assert!((actual - expected).abs() <= NUMERIC_TOLERANCE * expected.max(1.0));
        assert_eq!(model.model_digest().unwrap(), permuted.model_digest().unwrap());
    }

    #[test]
    fn factor_permutation_and_sign_flips_preserve_covariance_and_canonical_digest() {
        let model = fixture(8, 3);
        let residual = residual(8);
        let permutation = [2, 0, 1];
        let mut transformed = model.clone();
        transformed.factor_ids = permutation.iter().map(|index| model.factor_ids[*index].clone()).collect();
        transformed.loadings = model.loadings.iter().map(|row| {
            permutation.iter().enumerate().map(|(new_column, old_column)| {
                let sign = if new_column == 1 { -1.0 } else { 1.0 };
                row[*old_column] * sign
            }).collect()
        }).collect();

        assert_eq!(model.model_digest().unwrap(), transformed.model_digest().unwrap());
        let original_covariance = model.dense_covariance().unwrap();
        let transformed_covariance = transformed.dense_covariance().unwrap();
        for row in 0..model.dimension() {
            for column in 0..model.dimension() {
                assert!(
                    (original_covariance[row][column] - transformed_covariance[row][column]).abs()
                        <= NUMERIC_TOLERANCE,
                    "covariance mismatch at ({row}, {column})"
                );
            }
        }
        let expected = model.mahalanobis_squared(&residual).unwrap();
        let actual = transformed.mahalanobis_squared(&residual).unwrap();
        assert!((actual - expected).abs() <= NUMERIC_TOLERANCE * expected.max(1.0));
    }

    #[test]
    fn shared_common_mode_does_not_count_correlated_channels_as_independent() {
        let correlated = FactorCovarianceModel {
            model_version: "factor-covariance-v1".to_owned(),
            feature_ids: vec!["temperature-a".to_owned(), "temperature-b".to_owned()],
            feature_units: vec!["kelvin".to_owned(), "kelvin".to_owned()],
            diagonal_variances: vec![1.0, 1.0],
            factor_ids: vec!["shared-calibration-drift".to_owned()],
            loadings: vec![vec![0.8], vec![0.8]],
            calibration_provenance_digest: "b".repeat(64),
        };
        let independent = FactorCovarianceModel {
            factor_ids: vec![],
            loadings: vec![vec![], vec![]],
            ..correlated.clone()
        };
        let residual = [1.0, 1.0];
        let correlated_distance = correlated.mahalanobis_squared(&residual).unwrap();
        let independent_distance = independent.mahalanobis_squared(&residual).unwrap();
        assert!(correlated_distance < independent_distance);
        assert!((correlated_distance - 2.0 / (1.0 + 2.0 * 0.8_f64.powi(2))).abs() < 1e-12);
    }

    #[test]
    fn omitted_common_mode_is_detected_by_held_out_residual_screen() {
        let evidence = FactorResidualAdequacyEvidence {
            calibration_observation_count: 500,
            minimum_observation_count: 100,
            maximum_absolute_residual_correlation: 0.72,
            maximum_allowed_residual_correlation: 0.15,
            evidence_digest: "c".repeat(64),
        };
        let report = assess_residual_correlation_adequacy(&evidence).unwrap();
        assert_eq!(report.status, FactorAdequacyStatus::Inadequate);
    }

    #[test]
    fn residual_screen_abstains_when_calibration_evidence_is_insufficient() {
        let evidence = FactorResidualAdequacyEvidence {
            calibration_observation_count: 12,
            minimum_observation_count: 100,
            maximum_absolute_residual_correlation: 0.02,
            maximum_allowed_residual_correlation: 0.15,
            evidence_digest: "d".repeat(64),
        };
        let report = assess_residual_correlation_adequacy(&evidence).unwrap();
        assert_eq!(report.status, FactorAdequacyStatus::InsufficientEvidence);
    }

    #[test]
    fn invalid_variance_loading_identifier_and_digest_fail_closed() {
        let mut model = fixture(4, 1);
        model.diagonal_variances[0] = 0.0;
        assert!(matches!(
            model.validate(),
            Err(FactorCovarianceError::InvalidDiagonalVariance { index: 0 })
        ));

        let mut model = fixture(4, 1);
        model.loadings[2][0] = f64::NAN;
        assert!(matches!(
            model.validate(),
            Err(FactorCovarianceError::NonFiniteLoading { row: 2, column: 0 })
        ));

        let mut model = fixture(4, 1);
        model.feature_ids[1] = model.feature_ids[0].clone();
        assert!(matches!(
            model.validate(),
            Err(FactorCovarianceError::InvalidFeatureId { index: 1 })
        ));

        let mut model = fixture(4, 1);
        model.factor_ids[0] = " ".to_owned();
        assert!(matches!(
            model.validate(),
            Err(FactorCovarianceError::InvalidFactorId { index: 0 })
        ));

        let mut model = fixture(4, 1);
        model.calibration_provenance_digest = "not-a-digest".to_owned();
        assert!(matches!(
            model.validate(),
            Err(FactorCovarianceError::InvalidProvenanceDigest)
        ));
    }

    #[test]
    fn malformed_loading_shape_and_non_lower_rank_fail_closed() {
        let mut model = fixture(4, 1);
        model.loadings.pop();
        assert!(matches!(
            model.validate(),
            Err(FactorCovarianceError::LoadingRowCountMismatch { .. })
        ));

        let mut model = fixture(4, 1);
        model.loadings[0].clear();
        assert!(matches!(
            model.validate(),
            Err(FactorCovarianceError::LoadingColumnCountMismatch { row: 0, .. })
        ));

        let mut model = fixture(4, 1);
        model.factor_ids = vec![
            "factor-a".to_owned(),
            "factor-b".to_owned(),
            "factor-c".to_owned(),
            "factor-d".to_owned(),
        ];
        model.loadings = vec![vec![0.0; 4]; 4];
        assert!(matches!(
            model.validate(),
            Err(FactorCovarianceError::InvalidFactorRank { .. })
        ));
    }

    #[test]
    fn adequate_residual_screen_is_not_a_general_adequacy_certificate() {
        let evidence = FactorResidualAdequacyEvidence {
            calibration_observation_count: 500,
            minimum_observation_count: 100,
            maximum_absolute_residual_correlation: 0.05,
            maximum_allowed_residual_correlation: 0.15,
            evidence_digest: "e".repeat(64),
        };
        let report = assess_residual_correlation_adequacy(&evidence).unwrap();
        assert_eq!(report.status, FactorAdequacyStatus::NoDetectedResidualCorrelation);
        assert_eq!(report.evidence_digest, evidence.evidence_digest);
    }

    #[test]
    fn residual_shape_and_non_finite_values_fail_closed() {
        let model = fixture(4, 1);
        assert!(matches!(
            model.mahalanobis_squared(&[1.0, 2.0]),
            Err(FactorCovarianceError::ResidualDimensionMismatch { .. })
        ));
        assert!(matches!(
            model.mahalanobis_squared(&[1.0, f64::INFINITY, 0.0, 0.0]),
            Err(FactorCovarianceError::NonFiniteResidual { index: 1 })
        ));
    }
}
