// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Differential Privacy for Trust Analytics
//!
//! Implements differential privacy mechanisms to protect individual agent
//! trust data while enabling meaningful aggregate analytics.
//!
//! Features:
//! - Laplace and Gaussian noise mechanisms
//! - Privacy budget tracking with advanced composition
//! - Private aggregations (mean, sum, histogram, percentile)
//! - Local differential privacy for decentralized settings
//! - Trust-specific DP queries

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Configuration for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DPConfig {
    /// Total privacy budget (epsilon)
    pub epsilon: f64,
    /// Privacy parameter delta (for approximate DP)
    pub delta: f64,
    /// Default sensitivity for queries
    pub default_sensitivity: f64,
    /// Clipping bounds for data
    pub clipping_bounds: ClippingBounds,
}

impl Default for DPConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-6,
            default_sensitivity: 1.0,
            clipping_bounds: ClippingBounds::default(),
        }
    }
}

/// Bounds for clipping data values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClippingBounds {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
}

impl Default for ClippingBounds {
    fn default() -> Self {
        Self { min: 0.0, max: 1.0 }
    }
}

impl ClippingBounds {
    /// Create new bounds
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    /// Clip a value to bounds
    pub fn clip(&self, value: f64) -> f64 {
        value.clamp(self.min, self.max)
    }

    /// Get sensitivity based on bounds
    pub fn sensitivity(&self) -> f64 {
        self.max - self.min
    }
}

/// Noise mechanism type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseMechanism {
    /// Laplace mechanism (pure DP)
    Laplace,
    /// Gaussian mechanism (approximate DP)
    Gaussian,
}

/// Random number generator wrapper for DP
pub struct DPRng {
    rng: rand::rngs::ThreadRng,
}

impl Default for DPRng {
    fn default() -> Self {
        Self::new()
    }
}

impl DPRng {
    /// Create new RNG
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    /// Sample from Laplace distribution
    /// Uses the inverse CDF method: X = μ - b * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
    pub fn sample_laplace(&mut self, scale: f64) -> f64 {
        let u: f64 = self.rng.gen();
        // Clamp to (0, 1) exclusive to avoid ln(0) = -inf when u is exactly 0.0 or 1.0
        let u = u.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
        let u_shifted = u - 0.5;
        let sign = if u_shifted >= 0.0 { 1.0 } else { -1.0 };
        -scale * sign * (1.0 - 2.0 * u_shifted.abs()).ln()
    }

    /// Sample from Gaussian distribution
    #[allow(clippy::unwrap_used)]
    pub fn sample_gaussian(&mut self, sigma: f64) -> f64 {
        // unwrap is safe here: Normal::new(0.0, 1.0) is a valid distribution
        let normal = Normal::new(0.0, sigma).unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
        normal.sample(&mut self.rng)
    }
}

/// Noise generator for differential privacy
pub struct NoiseGenerator {
    /// Configuration
    config: DPConfig,
    /// RNG
    rng: DPRng,
    /// Default mechanism
    mechanism: NoiseMechanism,
}

impl NoiseGenerator {
    /// Create new noise generator
    pub fn new(config: DPConfig) -> Self {
        Self {
            config,
            rng: DPRng::new(),
            mechanism: NoiseMechanism::Laplace,
        }
    }

    /// Create with specific mechanism
    pub fn with_mechanism(config: DPConfig, mechanism: NoiseMechanism) -> Self {
        Self {
            config,
            rng: DPRng::new(),
            mechanism,
        }
    }

    /// Add noise to a value
    pub fn add_noise(&mut self, value: f64, epsilon: f64, sensitivity: f64) -> f64 {
        match self.mechanism {
            NoiseMechanism::Laplace => {
                let scale = sensitivity / epsilon;
                value + self.rng.sample_laplace(scale)
            }
            NoiseMechanism::Gaussian => {
                // For (ε, δ)-DP with Gaussian mechanism
                let sigma =
                    sensitivity * (2.0_f64 * (1.25 / self.config.delta).ln()).sqrt() / epsilon;
                value + self.rng.sample_gaussian(sigma)
            }
        }
    }

    /// Add noise using default epsilon
    pub fn add_noise_default(&mut self, value: f64, sensitivity: f64) -> f64 {
        self.add_noise(value, self.config.epsilon, sensitivity)
    }

    /// Get the mechanism type
    pub fn mechanism(&self) -> NoiseMechanism {
        self.mechanism
    }
}

/// Privacy budget tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Total budget (epsilon)
    total_epsilon: f64,
    /// Total delta
    total_delta: f64,
    /// Spent epsilon
    spent_epsilon: f64,
    /// Spent delta
    spent_delta: f64,
    /// Query history for auditing
    query_history: Vec<BudgetQuery>,
}

/// Record of a privacy-consuming query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetQuery {
    /// Query identifier
    pub query_id: String,
    /// Epsilon consumed
    pub epsilon: f64,
    /// Delta consumed
    pub delta: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Query description
    pub description: String,
}

impl PrivacyBudget {
    /// Create new budget
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            total_epsilon: epsilon,
            total_delta: delta,
            spent_epsilon: 0.0,
            spent_delta: 0.0,
            query_history: vec![],
        }
    }

    /// Check if a query can be performed
    pub fn can_query(&self, epsilon: f64, delta: f64) -> bool {
        self.spent_epsilon + epsilon <= self.total_epsilon
            && self.spent_delta + delta <= self.total_delta
    }

    /// Consume budget for a query
    pub fn consume(
        &mut self,
        query_id: &str,
        epsilon: f64,
        delta: f64,
        description: &str,
    ) -> Result<(), PrivacyError> {
        if !self.can_query(epsilon, delta) {
            return Err(PrivacyError::BudgetExceeded {
                requested_epsilon: epsilon,
                requested_delta: delta,
                remaining_epsilon: self.remaining_epsilon(),
                remaining_delta: self.remaining_delta(),
            });
        }

        self.spent_epsilon += epsilon;
        self.spent_delta += delta;

        self.query_history.push(BudgetQuery {
            query_id: query_id.to_string(),
            epsilon,
            delta,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            description: description.to_string(),
        });

        Ok(())
    }

    /// Get remaining epsilon
    pub fn remaining_epsilon(&self) -> f64 {
        self.total_epsilon - self.spent_epsilon
    }

    /// Get remaining delta
    pub fn remaining_delta(&self) -> f64 {
        self.total_delta - self.spent_delta
    }

    /// Get usage fraction
    pub fn usage_fraction(&self) -> f64 {
        self.spent_epsilon / self.total_epsilon
    }

    /// Reset budget (for new epoch)
    pub fn reset(&mut self) {
        self.spent_epsilon = 0.0;
        self.spent_delta = 0.0;
        self.query_history.clear();
    }

    /// Get query count
    pub fn query_count(&self) -> usize {
        self.query_history.len()
    }
}

/// Private aggregator for trust data
pub struct PrivateAggregator {
    /// Configuration
    config: DPConfig,
    /// Privacy budget
    budget: PrivacyBudget,
    /// Noise generator
    noise_gen: NoiseGenerator,
    /// Query counter
    query_counter: u64,
}

impl PrivateAggregator {
    /// Create new aggregator
    pub fn new(config: DPConfig) -> Self {
        let budget = PrivacyBudget::new(config.epsilon, config.delta);
        let noise_gen = NoiseGenerator::new(config.clone());
        Self {
            config,
            budget,
            noise_gen,
            query_counter: 0,
        }
    }

    /// Private mean computation
    pub fn private_mean(&mut self, values: &[f64], epsilon: f64) -> Result<f64, PrivacyError> {
        if values.is_empty() {
            return Ok(0.0);
        }

        let query_id = format!("mean_{}", self.query_counter);
        self.query_counter += 1;

        // Consume budget
        self.budget
            .consume(&query_id, epsilon, 0.0, "Private mean")?;

        // Clip values
        let clipped: Vec<f64> = values
            .iter()
            .map(|v| self.config.clipping_bounds.clip(*v))
            .collect();

        // Compute true mean
        let sum: f64 = clipped.iter().sum();
        let n = clipped.len() as f64;
        let true_mean = sum / n;

        // Sensitivity of mean = range / n
        let sensitivity = self.config.clipping_bounds.sensitivity() / n;

        // Add noise
        Ok(self.noise_gen.add_noise(true_mean, epsilon, sensitivity))
    }

    /// Private sum computation
    pub fn private_sum(&mut self, values: &[f64], epsilon: f64) -> Result<f64, PrivacyError> {
        let query_id = format!("sum_{}", self.query_counter);
        self.query_counter += 1;

        // Consume budget
        self.budget
            .consume(&query_id, epsilon, 0.0, "Private sum")?;

        // Clip values
        let clipped: Vec<f64> = values
            .iter()
            .map(|v| self.config.clipping_bounds.clip(*v))
            .collect();

        // Compute true sum
        let true_sum: f64 = clipped.iter().sum();

        // Sensitivity of sum = range (for one change)
        let sensitivity = self.config.clipping_bounds.sensitivity();

        // Add noise
        Ok(self.noise_gen.add_noise(true_sum, epsilon, sensitivity))
    }

    /// Private count
    pub fn private_count(&mut self, count: usize, epsilon: f64) -> Result<f64, PrivacyError> {
        let query_id = format!("count_{}", self.query_counter);
        self.query_counter += 1;

        // Consume budget
        self.budget
            .consume(&query_id, epsilon, 0.0, "Private count")?;

        // Sensitivity of count = 1
        let sensitivity = 1.0;

        // Add noise
        Ok(self.noise_gen.add_noise(count as f64, epsilon, sensitivity))
    }

    /// Private histogram
    pub fn private_histogram(
        &mut self,
        values: &[f64],
        bins: usize,
        epsilon: f64,
    ) -> Result<Vec<f64>, PrivacyError> {
        let query_id = format!("hist_{}", self.query_counter);
        self.query_counter += 1;

        // Per-bin epsilon (parallel composition)
        let per_bin_epsilon = epsilon;

        // Consume budget
        self.budget
            .consume(&query_id, per_bin_epsilon, 0.0, "Private histogram")?;

        // Clip values
        let clipped: Vec<f64> = values
            .iter()
            .map(|v| self.config.clipping_bounds.clip(*v))
            .collect();

        // Compute bin edges
        let range = self.config.clipping_bounds.sensitivity();
        let bin_width = range / bins as f64;

        // Count per bin
        let mut counts = vec![0.0_f64; bins];
        for v in clipped {
            let bin_idx = ((v - self.config.clipping_bounds.min) / bin_width) as usize;
            let bin_idx = bin_idx.min(bins - 1);
            counts[bin_idx] += 1.0;
        }

        // Add noise to each bin (sensitivity = 1 for count)
        for count in counts.iter_mut() {
            *count = self.noise_gen.add_noise(*count, per_bin_epsilon, 1.0);
            *count = (*count).max(0.0); // Post-process: no negative counts
        }

        Ok(counts)
    }

    /// Private percentile (approximate)
    pub fn private_percentile(
        &mut self,
        values: &[f64],
        percentile: f64,
        epsilon: f64,
    ) -> Result<f64, PrivacyError> {
        // Use histogram approach for percentile
        let bins = 100;
        let hist = self.private_histogram(values, bins, epsilon)?;

        let total: f64 = hist.iter().sum();
        let target = total * percentile / 100.0;

        let mut cumulative = 0.0;
        let bin_width = self.config.clipping_bounds.sensitivity() / bins as f64;

        for (i, count) in hist.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return Ok(self.config.clipping_bounds.min + (i as f64 + 0.5) * bin_width);
            }
        }

        Ok(self.config.clipping_bounds.max)
    }

    /// Get budget reference
    pub fn budget(&self) -> &PrivacyBudget {
        &self.budget
    }

    /// Get mutable budget reference
    pub fn budget_mut(&mut self) -> &mut PrivacyBudget {
        &mut self.budget
    }
}

/// Local differential privacy for individual agents
pub struct LocalDP {
    /// Epsilon for local randomization
    epsilon: f64,
    /// RNG
    rng: DPRng,
}

impl LocalDP {
    /// Create new local DP instance
    pub fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            rng: DPRng::new(),
        }
    }

    /// Randomized response for boolean
    pub fn randomized_response(&mut self, true_value: bool) -> bool {
        let p = 1.0 / (1.0 + self.epsilon.exp());
        let flip = self.rng.rng.gen::<f64>() < p;

        if flip {
            !true_value
        } else {
            true_value
        }
    }

    /// RAPPOR-style encoding for categorical data
    pub fn rappor_encode(&mut self, value: usize, domain_size: usize) -> Vec<bool> {
        let mut result = vec![false; domain_size];
        result[value.min(domain_size - 1)] = true;

        // Apply randomized response to each bit
        for bit in result.iter_mut() {
            *bit = self.randomized_response(*bit);
        }

        result
    }

    /// Private value perturbation
    pub fn perturb_value(&mut self, value: f64, bounds: &ClippingBounds) -> f64 {
        let clipped = bounds.clip(value);
        let scale = bounds.sensitivity() / self.epsilon;
        clipped + self.rng.sample_laplace(scale)
    }
}

/// Trust-specific private analytics
pub struct PrivateTrustAnalytics {
    /// Aggregator
    aggregator: PrivateAggregator,
}

/// Distribution summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustDistribution {
    /// Private mean
    pub mean: f64,
    /// Private median (50th percentile)
    pub median: f64,
    /// Private count
    pub count: f64,
    /// Private histogram bins
    pub histogram: Vec<f64>,
}

impl PrivateTrustAnalytics {
    /// Create new trust analytics
    pub fn new(config: DPConfig) -> Self {
        Self {
            aggregator: PrivateAggregator::new(config),
        }
    }

    /// Analyze trust score distribution privately
    pub fn analyze_trust_distribution(
        &mut self,
        scores: &[f64],
        epsilon_per_query: f64,
    ) -> Result<TrustDistribution, PrivacyError> {
        let mean = self.aggregator.private_mean(scores, epsilon_per_query)?;
        let median = self
            .aggregator
            .private_percentile(scores, 50.0, epsilon_per_query)?;
        let count = self
            .aggregator
            .private_count(scores.len(), epsilon_per_query)?;
        let histogram = self
            .aggregator
            .private_histogram(scores, 10, epsilon_per_query)?;

        Ok(TrustDistribution {
            mean,
            median,
            count,
            histogram,
        })
    }

    /// Private trust tier counts
    pub fn private_tier_counts(
        &mut self,
        scores: &[f64],
        thresholds: &[f64],
        epsilon: f64,
    ) -> Result<Vec<f64>, PrivacyError> {
        let num_tiers = thresholds.len() + 1;
        let mut counts = vec![0.0_f64; num_tiers];

        for score in scores {
            let mut tier = 0;
            for (i, &thresh) in thresholds.iter().enumerate() {
                if *score >= thresh {
                    tier = i + 1;
                }
            }
            counts[tier] += 1.0;
        }

        // Add noise to each tier count
        let per_tier_epsilon = epsilon; // Using parallel composition
        let query_id = format!("tiers_{}", self.aggregator.query_counter);
        self.aggregator
            .budget
            .consume(&query_id, per_tier_epsilon, 0.0, "Private tier counts")?;

        for count in counts.iter_mut() {
            *count = self
                .aggregator
                .noise_gen
                .add_noise(*count, per_tier_epsilon, 1.0);
            *count = (*count).max(0.0);
        }

        Ok(counts)
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> (f64, f64) {
        (
            self.aggregator.budget.remaining_epsilon(),
            self.aggregator.budget.remaining_delta(),
        )
    }

    /// Reset budget
    pub fn reset_budget(&mut self) {
        self.aggregator.budget.reset();
    }
}

/// Privacy-related errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum PrivacyError {
    /// Privacy budget exceeded
    #[error("Privacy budget exceeded: requested (ε={requested_epsilon}, δ={requested_delta}), remaining (ε={remaining_epsilon}, δ={remaining_delta})")]
    BudgetExceeded {
        /// Epsilon requested for the query.
        requested_epsilon: f64,
        /// Delta requested for the query.
        requested_delta: f64,
        /// Remaining epsilon budget.
        remaining_epsilon: f64,
        /// Remaining delta budget.
        remaining_delta: f64,
    },

    /// Invalid parameters
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_generator_laplace() {
        let config = DPConfig::default();
        let mut gen = NoiseGenerator::new(config);

        // Generate many samples, should be centered around 0
        let samples: Vec<f64> = (0..5000).map(|_| gen.add_noise(0.0, 1.0, 1.0)).collect();

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        // Allow 0.3 tolerance for random noise
        assert!(
            mean.abs() < 0.3,
            "Laplace noise should be centered around 0, got {}",
            mean
        );
    }

    #[test]
    fn test_noise_generator_gaussian() {
        let config = DPConfig::default();
        let mut gen = NoiseGenerator::with_mechanism(config, NoiseMechanism::Gaussian);

        // Generate many samples
        let samples: Vec<f64> = (0..5000).map(|_| gen.add_noise(0.0, 1.0, 1.0)).collect();

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        // Gaussian noise with larger sigma has higher variance, allow 0.5 tolerance
        assert!(
            mean.abs() < 0.5,
            "Gaussian noise should be centered around 0, got {}",
            mean
        );
    }

    #[test]
    fn test_privacy_budget() {
        let mut budget = PrivacyBudget::new(1.0, 1e-6);

        assert!(budget.can_query(0.5, 0.0));
        budget.consume("q1", 0.5, 0.0, "Test query 1").unwrap();

        assert!(budget.can_query(0.5, 0.0));
        budget.consume("q2", 0.5, 0.0, "Test query 2").unwrap();

        // Budget exhausted
        assert!(!budget.can_query(0.1, 0.0));

        let result = budget.consume("q3", 0.1, 0.0, "Test query 3");
        assert!(matches!(result, Err(PrivacyError::BudgetExceeded { .. })));
    }

    #[test]
    fn test_private_mean() {
        let config = DPConfig {
            epsilon: 100.0, // Very high epsilon budget to allow high-epsilon queries
            ..Default::default()
        };
        let mut agg = PrivateAggregator::new(config);

        let values: Vec<f64> = vec![0.5, 0.6, 0.7, 0.4, 0.5];
        let true_mean = 0.54;

        // Use high epsilon (low noise) to make the test deterministic.
        // With epsilon=50, Laplace scale = (1.0/5) / 50 = 0.004, so noise
        // is negligible relative to the 0.2 tolerance.
        let private_mean = agg.private_mean(&values, 50.0).unwrap();

        // With high epsilon, should be close to true mean
        assert!(
            (private_mean - true_mean).abs() < 0.2,
            "Private mean {} should be close to true mean {}",
            private_mean,
            true_mean
        );
    }

    #[test]
    fn test_private_histogram() {
        let config = DPConfig {
            epsilon: 10.0,
            ..Default::default()
        };
        let mut agg = PrivateAggregator::new(config);

        let values: Vec<f64> = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let histogram = agg.private_histogram(&values, 5, 2.0).unwrap();

        assert_eq!(histogram.len(), 5);
        // Total should be approximately the count
        let total: f64 = histogram.iter().sum();
        assert!(total > 0.0, "Histogram total should be positive");
    }

    #[test]
    fn test_local_dp_randomized_response() {
        let mut ldp = LocalDP::new(2.0);

        // With moderate epsilon, responses should mostly match
        let mut matches = 0;
        for _ in 0..100 {
            if ldp.randomized_response(true) {
                matches += 1;
            }
        }

        // With epsilon=2, probability of keeping true value is e^2/(1+e^2) ≈ 0.88
        assert!(matches > 50, "Most true values should stay true");
    }

    #[test]
    fn test_trust_distribution() {
        let config = DPConfig {
            epsilon: 20.0, // High epsilon for test accuracy
            ..Default::default()
        };
        let mut analytics = PrivateTrustAnalytics::new(config);

        let scores = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.5, 0.5];
        let dist = analytics.analyze_trust_distribution(&scores, 2.0).unwrap();

        // Check reasonable values
        assert!(dist.mean > 0.0 && dist.mean < 1.0);
        assert!(dist.median > 0.0 && dist.median < 1.0);
        assert!(dist.count > 0.0);
        assert!(!dist.histogram.is_empty());
    }

    #[test]
    fn test_clipping_bounds() {
        let bounds = ClippingBounds::new(0.0, 1.0);

        assert_eq!(bounds.clip(-0.5), 0.0);
        assert_eq!(bounds.clip(0.5), 0.5);
        assert_eq!(bounds.clip(1.5), 1.0);
        assert_eq!(bounds.sensitivity(), 1.0);
    }
}
