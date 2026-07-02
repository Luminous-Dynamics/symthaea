// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proof of Gradient Quality (PoGQ)
//!
//! Byzantine-resistant federated learning with up to 45% adversary tolerance.
//! This module provides the core PoGQ algorithm for validating gradient updates
//! in federated learning scenarios.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Configuration for PoGQ validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoGQConfig {
    /// Byzantine fault tolerance threshold (default: 0.45 = 45%)
    pub bft_threshold: f64,

    /// Minimum gradient quality score (0.0 - 1.0)
    pub min_quality_score: f64,

    /// Number of validation rounds
    pub validation_rounds: usize,

    /// Enable adaptive threshold adjustment
    pub adaptive_threshold: bool,
}

impl Default for PoGQConfig {
    fn default() -> Self {
        Self {
            bft_threshold: crate::DEFAULT_BFT_THRESHOLD,
            min_quality_score: 0.7,
            validation_rounds: 3,
            adaptive_threshold: true,
        }
    }
}

/// Gradient update from a federated learning participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientUpdate {
    /// Participant identifier
    pub participant_id: String,

    /// Gradient data (serialized weights)
    pub gradients: Vec<f64>,

    /// Timestamp of the update
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Proof of computation (zk-STARK or similar)
    pub proof: Option<Vec<u8>>,
}

/// Quality score for a gradient update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScore {
    /// Overall quality score (0.0 - 1.0)
    pub score: f64,

    /// Consistency with other gradients
    pub consistency: f64,

    /// Magnitude check (detects extreme outliers)
    pub magnitude_check: f64,

    /// Direction alignment with consensus
    pub direction_alignment: f64,

    /// Is this gradient considered valid?
    pub is_valid: bool,
}

/// PoGQ Validator
pub struct PoGQValidator {
    config: PoGQConfig,
}

impl PoGQValidator {
    /// Create a new PoGQ validator
    pub fn new(config: PoGQConfig) -> Self {
        Self { config }
    }

    /// Validate a batch of gradient updates
    ///
    /// Returns quality scores for each gradient and identifies Byzantine actors
    pub fn validate_gradients(
        &self,
        gradients: &[GradientUpdate],
    ) -> Result<Vec<QualityScore>> {
        if gradients.is_empty() {
            return Err(Error::PoGQ("No gradients to validate".to_string()));
        }

        // TODO: Implement full PoGQ algorithm
        // For now, return placeholder scores
        let scores = gradients
            .iter()
            .map(|_grad| {
                // Placeholder: In production, this would:
                // 1. Compare gradient with median/consensus
                // 2. Check magnitude bounds
                // 3. Validate cryptographic proofs
                // 4. Score based on consistency
                QualityScore {
                    score: 0.85,
                    consistency: 0.9,
                    magnitude_check: 0.95,
                    direction_alignment: 0.8,
                    is_valid: true,
                }
            })
            .collect();

        Ok(scores)
    }

    /// Aggregate valid gradients into a consensus update
    pub fn aggregate_gradients(
        &self,
        gradients: &[GradientUpdate],
        scores: &[QualityScore],
    ) -> Result<Vec<f64>> {
        if gradients.len() != scores.len() {
            return Err(Error::PoGQ("Mismatched gradients and scores".to_string()));
        }

        // Filter valid gradients
        let valid_gradients: Vec<&GradientUpdate> = gradients
            .iter()
            .zip(scores.iter())
            .filter(|(_, score)| score.is_valid)
            .map(|(grad, _)| grad)
            .collect();

        if valid_gradients.is_empty() {
            return Err(Error::PoGQ("No valid gradients to aggregate".to_string()));
        }

        // TODO: Implement weighted aggregation based on quality scores
        // For now, simple averaging
        let gradient_size = valid_gradients[0].gradients.len();
        let mut aggregated = vec![0.0; gradient_size];

        for grad in valid_gradients.iter() {
            for (i, &val) in grad.gradients.iter().enumerate() {
                aggregated[i] += val;
            }
        }

        let count = valid_gradients.len() as f64;
        aggregated.iter_mut().for_each(|v| *v /= count);

        Ok(aggregated)
    }

    /// Detect Byzantine actors based on quality scores
    pub fn detect_byzantine(
        &self,
        gradients: &[GradientUpdate],
        scores: &[QualityScore],
    ) -> Vec<String> {
        gradients
            .iter()
            .zip(scores.iter())
            .filter(|(_, score)| !score.is_valid || score.score < self.config.min_quality_score)
            .map(|(grad, _)| grad.participant_id.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pogq_validator_creation() {
        let config = PoGQConfig::default();
        let validator = PoGQValidator::new(config);
        assert_eq!(validator.config.bft_threshold, 0.45);
    }

    #[test]
    fn test_validate_empty_gradients() {
        let validator = PoGQValidator::new(PoGQConfig::default());
        let result = validator.validate_gradients(&[]);
        assert!(result.is_err());
    }
}
