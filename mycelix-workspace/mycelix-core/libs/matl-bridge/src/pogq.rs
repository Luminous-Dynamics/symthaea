// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proof of Gradient Quality (PoGQ) validation
//!
//! PoGQ ensures that gradient contributions in federated learning are:
//! 1. Properly noised for differential privacy (ε=10, σ=5.0)
//! 2. Within acceptable bounds (L2 norm limits)
//! 3. Not Byzantine (statistical anomaly detection)

use serde::{Deserialize, Serialize};
use mycelix_core_types::trust::PoGQMetrics;

use crate::error::{MatlError, MatlResult};

/// Default differential privacy parameters
pub const DEFAULT_EPSILON: f32 = 10.0;
pub const DEFAULT_SIGMA: f32 = 5.0;

/// Maximum allowed L2 norm for gradients
pub const MAX_GRADIENT_NORM: f32 = 10.0;

/// Minimum gradient elements for valid contribution
pub const MIN_GRADIENT_ELEMENTS: usize = 100;

/// A gradient contribution to validate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientContribution {
    /// Agent who submitted the gradient
    pub agent_id: String,
    /// Round number
    pub round: u64,
    /// Hash of the gradient values
    pub gradient_hash: String,
    /// L2 norm of the gradient
    pub l2_norm: f32,
    /// Number of gradient elements
    pub num_elements: usize,
    /// Noise added for differential privacy
    pub noise_sigma: f32,
    /// Claimed epsilon value
    pub claimed_epsilon: f32,
    /// Statistical moments for anomaly detection
    pub mean: f32,
    pub variance: f32,
    pub skewness: f32,
    pub kurtosis: f32,
}

impl GradientContribution {
    /// Create a new gradient contribution
    pub fn new(agent_id: String, round: u64, gradient_hash: String) -> Self {
        Self {
            agent_id,
            round,
            gradient_hash,
            l2_norm: 0.0,
            num_elements: 0,
            noise_sigma: DEFAULT_SIGMA,
            claimed_epsilon: DEFAULT_EPSILON,
            mean: 0.0,
            variance: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }

    /// Set gradient statistics
    pub fn with_stats(
        mut self,
        l2_norm: f32,
        num_elements: usize,
        mean: f32,
        variance: f32,
        skewness: f32,
        kurtosis: f32,
    ) -> Self {
        self.l2_norm = l2_norm;
        self.num_elements = num_elements;
        self.mean = mean;
        self.variance = variance;
        self.skewness = skewness;
        self.kurtosis = kurtosis;
        self
    }

    /// Set differential privacy parameters
    pub fn with_dp(mut self, sigma: f32, epsilon: f32) -> Self {
        self.noise_sigma = sigma;
        self.claimed_epsilon = epsilon;
        self
    }
}

/// PoGQ Oracle for validating gradient contributions
#[derive(Debug, Clone)]
pub struct PoGQOracle {
    /// Required epsilon for differential privacy
    required_epsilon: f32,
    /// Required sigma (noise) for differential privacy
    required_sigma: f32,
    /// Maximum allowed L2 norm
    max_norm: f32,
    /// Statistical thresholds for anomaly detection
    anomaly_config: AnomalyConfig,
}

impl PoGQOracle {
    /// Create a new PoGQ oracle with default settings
    pub fn new() -> Self {
        Self {
            required_epsilon: DEFAULT_EPSILON,
            required_sigma: DEFAULT_SIGMA,
            max_norm: MAX_GRADIENT_NORM,
            anomaly_config: AnomalyConfig::default(),
        }
    }

    /// Create with custom settings
    pub fn with_config(epsilon: f32, sigma: f32, max_norm: f32) -> Self {
        Self {
            required_epsilon: epsilon,
            required_sigma: sigma,
            max_norm,
            anomaly_config: AnomalyConfig::default(),
        }
    }

    /// Validate a gradient contribution
    pub fn validate(&self, contribution: &GradientContribution) -> MatlResult<PoGQMetrics> {
        // 1. Check differential privacy compliance
        let dp_compliant = self.check_differential_privacy(contribution)?;

        // 2. Check gradient bounds
        let bounds_valid = self.check_bounds(contribution)?;

        // 3. Check for statistical anomalies (Byzantine detection)
        let anomaly_score = self.detect_anomalies(contribution)?;

        // 4. Compute quality score
        let quality_score = self.compute_quality_score(
            dp_compliant,
            bounds_valid,
            anomaly_score,
            contribution,
        );

        Ok(PoGQMetrics {
            dp_compliant,
            epsilon: contribution.claimed_epsilon,
            sigma: contribution.noise_sigma,
            l2_norm: contribution.l2_norm,
            bounds_valid,
            quality_score,
        })
    }

    /// Check differential privacy compliance
    fn check_differential_privacy(&self, contribution: &GradientContribution) -> MatlResult<bool> {
        // Epsilon should be <= required (lower is more private)
        if contribution.claimed_epsilon > self.required_epsilon {
            return Err(MatlError::DPViolation {
                epsilon: contribution.claimed_epsilon,
                required: self.required_epsilon,
            });
        }

        // Sigma should be >= required (higher is more noise)
        if contribution.noise_sigma < self.required_sigma {
            return Ok(false);
        }

        Ok(true)
    }

    /// Check gradient bounds
    fn check_bounds(&self, contribution: &GradientContribution) -> MatlResult<bool> {
        // Check L2 norm
        if contribution.l2_norm > self.max_norm {
            return Ok(false);
        }

        // Check minimum elements
        if contribution.num_elements < MIN_GRADIENT_ELEMENTS {
            return Err(MatlError::InvalidGradient {
                reason: format!(
                    "Too few elements: {} < {}",
                    contribution.num_elements, MIN_GRADIENT_ELEMENTS
                ),
            });
        }

        Ok(true)
    }

    /// Detect statistical anomalies (potential Byzantine behavior)
    fn detect_anomalies(&self, contribution: &GradientContribution) -> MatlResult<f32> {
        let config = &self.anomaly_config;

        // Check skewness (should be near 0 for Gaussian noise)
        let skewness_ok = contribution.skewness.abs() < config.max_skewness;

        // Check kurtosis (should be near 3 for Gaussian, so excess near 0)
        let excess_kurtosis = (contribution.kurtosis - 3.0).abs();
        let kurtosis_ok = excess_kurtosis < config.max_excess_kurtosis;

        // Check variance (should be reasonable)
        let variance_ok = contribution.variance > config.min_variance
            && contribution.variance < config.max_variance;

        // Compute anomaly score (0 = no anomaly, 1 = definite anomaly)
        let mut anomaly_score = 0.0;
        if !skewness_ok {
            anomaly_score += 0.3;
        }
        if !kurtosis_ok {
            anomaly_score += 0.3;
        }
        if !variance_ok {
            anomaly_score += 0.4;
        }

        Ok(anomaly_score)
    }

    /// Compute overall quality score
    fn compute_quality_score(
        &self,
        dp_compliant: bool,
        bounds_valid: bool,
        anomaly_score: f32,
        contribution: &GradientContribution,
    ) -> f32 {
        if !dp_compliant || !bounds_valid {
            return 0.0;
        }

        // Base score
        let mut score = 1.0;

        // Penalize anomalies
        score -= anomaly_score * 0.5;

        // Bonus for stronger privacy (lower epsilon)
        if contribution.claimed_epsilon < self.required_epsilon {
            let privacy_bonus = (self.required_epsilon - contribution.claimed_epsilon) / self.required_epsilon;
            score += privacy_bonus * 0.1;
        }

        // Bonus for more noise (higher sigma)
        if contribution.noise_sigma > self.required_sigma {
            let noise_bonus = (contribution.noise_sigma - self.required_sigma) / self.required_sigma;
            score += noise_bonus.min(0.1) * 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    /// Validate a batch of contributions
    pub fn validate_batch(&self, contributions: &[GradientContribution]) -> Vec<MatlResult<PoGQMetrics>> {
        contributions.iter().map(|c| self.validate(c)).collect()
    }

    /// Get the percentage of valid contributions in a batch
    pub fn batch_validity_rate(&self, contributions: &[GradientContribution]) -> f32 {
        let results = self.validate_batch(contributions);
        let valid_count = results.iter().filter(|r| r.is_ok()).count();
        valid_count as f32 / contributions.len() as f32
    }
}

impl Default for PoGQOracle {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for statistical anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyConfig {
    /// Maximum allowed skewness (Gaussian should be ~0)
    pub max_skewness: f32,
    /// Maximum allowed excess kurtosis (Gaussian excess ~0)
    pub max_excess_kurtosis: f32,
    /// Minimum expected variance
    pub min_variance: f32,
    /// Maximum expected variance
    pub max_variance: f32,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            max_skewness: 1.0,
            max_excess_kurtosis: 2.0,
            min_variance: 0.001,
            max_variance: 100.0,
        }
    }
}

/// Result of batch PoGQ validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchValidationResult {
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
    pub validity_rate: f32,
    pub average_quality: f32,
    pub byzantine_suspects: Vec<String>,
}

impl BatchValidationResult {
    /// Create from validation results
    pub fn from_results(
        results: &[(String, MatlResult<PoGQMetrics>)],
    ) -> Self {
        let total = results.len();
        let valid = results.iter().filter(|(_, r)| r.is_ok()).count();
        let invalid = total - valid;

        let quality_sum: f32 = results
            .iter()
            .filter_map(|(_, r)| r.as_ref().ok())
            .map(|m| m.quality_score)
            .sum();
        let average_quality = if valid > 0 {
            quality_sum / valid as f32
        } else {
            0.0
        };

        // Identify Byzantine suspects (low quality or failed validation)
        let byzantine_suspects: Vec<String> = results
            .iter()
            .filter(|(_, r)| {
                r.as_ref()
                    .map(|m| m.quality_score < 0.5)
                    .unwrap_or(true)
            })
            .map(|(id, _)| id.clone())
            .collect();

        Self {
            total,
            valid,
            invalid,
            validity_rate: valid as f32 / total as f32,
            average_quality,
            byzantine_suspects,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_contribution() -> GradientContribution {
        GradientContribution::new(
            "agent-1".to_string(),
            1,
            "hash123".to_string(),
        )
        .with_stats(5.0, 1000, 0.0, 1.0, 0.1, 3.1)
        .with_dp(5.0, 10.0)
    }

    #[test]
    fn test_valid_contribution() {
        let oracle = PoGQOracle::new();
        let contribution = create_valid_contribution();

        let result = oracle.validate(&contribution);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.dp_compliant);
        assert!(metrics.bounds_valid);
        assert!(metrics.quality_score > 0.5);
    }

    #[test]
    fn test_dp_violation() {
        let oracle = PoGQOracle::new();
        let contribution = create_valid_contribution().with_dp(5.0, 15.0); // epsilon too high

        let result = oracle.validate(&contribution);
        assert!(matches!(result, Err(MatlError::DPViolation { .. })));
    }

    #[test]
    fn test_norm_violation() {
        let oracle = PoGQOracle::new();
        let mut contribution = create_valid_contribution();
        contribution.l2_norm = 15.0; // exceeds max norm

        let result = oracle.validate(&contribution);
        assert!(result.is_ok());
        assert!(!result.unwrap().bounds_valid);
    }

    #[test]
    fn test_anomaly_detection() {
        let oracle = PoGQOracle::new();
        let mut contribution = create_valid_contribution();
        contribution.skewness = 5.0; // highly skewed - suspicious
        contribution.kurtosis = 10.0; // abnormal kurtosis
        contribution.variance = 0.0001; // too low variance

        let result = oracle.validate(&contribution).unwrap();
        assert!(result.quality_score < 0.9); // Should be penalized (but still valid)
    }
}
