//! # Differential Privacy for Trust Systems
//!
//! Privacy-preserving aggregations and analytics.
//!
//! ## Features
//!
//! - **Noise Mechanisms**: Laplace, Gaussian, Exponential
//! - **Privacy Budget**: Epsilon tracking and composition
//! - **Private Aggregations**: Mean, sum, histogram with DP guarantees
//! - **Local DP**: Per-agent privacy protection
//!
//! ## Theory
//!
//! Differential privacy provides mathematical guarantees that individual
//! contributions cannot be distinguished, even with auxiliary information.
//!
//! We use the (epsilon, delta)-differential privacy model where:
//! - epsilon controls privacy loss (smaller = more private)
//! - delta is the probability of failure (typically very small)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

/// Differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DPConfig {
    /// Default epsilon (privacy loss parameter)
    pub default_epsilon: f64,
    /// Default delta (failure probability)
    pub default_delta: f64,
    /// Total privacy budget per epoch
    pub epoch_budget: f64,
    /// Enable advanced composition
    pub advanced_composition: bool,
    /// Minimum samples for aggregation
    pub min_samples: usize,
    /// Clipping bounds for sensitivity
    pub clipping_bounds: ClippingBounds,
}

impl Default for DPConfig {
    fn default() -> Self {
        Self {
            default_epsilon: 1.0,
            default_delta: 1e-6,
            epoch_budget: 10.0,
            advanced_composition: true,
            min_samples: 10,
            clipping_bounds: ClippingBounds::default(),
        }
    }
}

/// Clipping bounds for sensitivity control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClippingBounds {
    /// Trust score bounds
    pub trust_min: f64,
    pub trust_max: f64,
    /// KREDIT bounds
    pub kredit_min: f64,
    pub kredit_max: f64,
    /// Activity count bounds
    pub activity_min: f64,
    pub activity_max: f64,
}

impl Default for ClippingBounds {
    fn default() -> Self {
        Self {
            trust_min: 0.0,
            trust_max: 1.0,
            kredit_min: 0.0,
            kredit_max: 1_000_000.0,
            activity_min: 0.0,
            activity_max: 10_000.0,
        }
    }
}

// ============================================================================
// Noise Mechanisms
// ============================================================================

/// Random number generator state (simple LCG for reproducibility)
#[derive(Debug, Clone)]
pub struct DPRng {
    state: u64,
}

impl DPRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn from_entropy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self::new(seed)
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Generate uniform random in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        self.next_f64()
    }

    /// Generate uniform random in [min, max)
    pub fn uniform_range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.uniform()
    }

    /// Generate Laplace noise with given scale
    pub fn laplace(&mut self, scale: f64) -> f64 {
        let u = self.uniform() - 0.5;
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    /// Generate Gaussian noise using Box-Muller
    pub fn gaussian(&mut self, mean: f64, stddev: f64) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + stddev * z
    }
}

/// Noise mechanism types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseMechanism {
    /// Laplace mechanism (pure DP)
    Laplace,
    /// Gaussian mechanism (approximate DP)
    Gaussian,
    /// Exponential mechanism (for categorical)
    Exponential,
}

/// Noise generator
#[derive(Debug)]
pub struct NoiseGenerator {
    rng: DPRng,
    mechanism: NoiseMechanism,
}

impl NoiseGenerator {
    pub fn new(seed: u64, mechanism: NoiseMechanism) -> Self {
        Self {
            rng: DPRng::new(seed),
            mechanism,
        }
    }

    /// Add noise to a value
    pub fn add_noise(&mut self, value: f64, sensitivity: f64, epsilon: f64) -> f64 {
        match self.mechanism {
            NoiseMechanism::Laplace => {
                let scale = sensitivity / epsilon;
                value + self.rng.laplace(scale)
            }
            NoiseMechanism::Gaussian => {
                // For (epsilon, delta)-DP with delta = 1e-6
                let sigma = sensitivity * (2.0_f64 * (1.25_f64 / 1e-6_f64).ln()).sqrt() / epsilon;
                value + self.rng.gaussian(0.0, sigma)
            }
            NoiseMechanism::Exponential => {
                // For exponential mechanism, value is a score
                // This is a simplified version
                let scale = sensitivity / epsilon;
                value + self.rng.laplace(scale)
            }
        }
    }

    /// Add noise to multiple values (vectorized)
    pub fn add_noise_vector(&mut self, values: &[f64], sensitivity: f64, epsilon: f64) -> Vec<f64> {
        values.iter()
            .map(|v| self.add_noise(*v, sensitivity, epsilon))
            .collect()
    }
}

// ============================================================================
// Privacy Budget
// ============================================================================

/// Privacy budget tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Total epsilon budget
    pub total_epsilon: f64,
    /// Total delta budget
    pub total_delta: f64,
    /// Consumed epsilon
    pub consumed_epsilon: f64,
    /// Consumed delta
    pub consumed_delta: f64,
    /// Query history
    pub query_history: Vec<BudgetQuery>,
    /// Use advanced composition
    pub advanced_composition: bool,
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
    /// Description
    pub description: String,
}

impl PrivacyBudget {
    /// Create new budget
    pub fn new(total_epsilon: f64, total_delta: f64, advanced_composition: bool) -> Self {
        Self {
            total_epsilon,
            total_delta,
            consumed_epsilon: 0.0,
            consumed_delta: 0.0,
            query_history: Vec::new(),
            advanced_composition,
        }
    }

    /// Check if budget allows query
    pub fn can_query(&self, epsilon: f64, delta: f64) -> bool {
        let (new_epsilon, new_delta) = self.compose(epsilon, delta);
        new_epsilon <= self.total_epsilon && new_delta <= self.total_delta
    }

    /// Consume budget for a query
    pub fn consume(
        &mut self,
        query_id: &str,
        epsilon: f64,
        delta: f64,
        description: &str,
        timestamp: u64,
    ) -> Result<(), PrivacyError> {
        if !self.can_query(epsilon, delta) {
            return Err(PrivacyError::BudgetExhausted {
                requested_epsilon: epsilon,
                requested_delta: delta,
                remaining_epsilon: self.remaining_epsilon(),
                remaining_delta: self.remaining_delta(),
            });
        }

        let (new_epsilon, new_delta) = self.compose(epsilon, delta);
        self.consumed_epsilon = new_epsilon;
        self.consumed_delta = new_delta;

        self.query_history.push(BudgetQuery {
            query_id: query_id.to_string(),
            epsilon,
            delta,
            timestamp,
            description: description.to_string(),
        });

        Ok(())
    }

    /// Compose privacy parameters
    fn compose(&self, epsilon: f64, delta: f64) -> (f64, f64) {
        if self.advanced_composition {
            // Advanced composition theorem
            let k = self.query_history.len() as f64 + 1.0;
            let composed_epsilon = (2.0 * k * epsilon.powi(2)).sqrt()
                + k * epsilon * (std::f64::consts::E.powf(epsilon) - 1.0);
            let composed_delta = self.consumed_delta + delta * k;
            (composed_epsilon.min(self.consumed_epsilon + epsilon * k), composed_delta)
        } else {
            // Basic composition (linear)
            (self.consumed_epsilon + epsilon, self.consumed_delta + delta)
        }
    }

    /// Remaining epsilon
    pub fn remaining_epsilon(&self) -> f64 {
        (self.total_epsilon - self.consumed_epsilon).max(0.0)
    }

    /// Remaining delta
    pub fn remaining_delta(&self) -> f64 {
        (self.total_delta - self.consumed_delta).max(0.0)
    }

    /// Reset budget (new epoch)
    pub fn reset(&mut self) {
        self.consumed_epsilon = 0.0;
        self.consumed_delta = 0.0;
        self.query_history.clear();
    }
}

// ============================================================================
// Private Aggregations
// ============================================================================

/// Private aggregation engine
#[derive(Debug)]
pub struct PrivateAggregator {
    config: DPConfig,
    budget: PrivacyBudget,
    noise_gen: NoiseGenerator,
    current_time: u64,
}

impl PrivateAggregator {
    /// Create new aggregator
    pub fn new(config: DPConfig, seed: u64) -> Self {
        let budget = PrivacyBudget::new(
            config.epoch_budget,
            config.default_delta * 10.0, // Allow 10x queries per epoch
            config.advanced_composition,
        );
        Self {
            noise_gen: NoiseGenerator::new(seed, NoiseMechanism::Laplace),
            config,
            budget,
            current_time: 0,
        }
    }

    /// Set current time
    pub fn set_time(&mut self, time: u64) {
        self.current_time = time;
    }

    /// Compute private mean
    pub fn private_mean(
        &mut self,
        values: &[f64],
        bounds: (f64, f64),
        epsilon: f64,
        query_id: &str,
    ) -> Result<f64, PrivacyError> {
        if values.len() < self.config.min_samples {
            return Err(PrivacyError::InsufficientSamples {
                required: self.config.min_samples,
                provided: values.len(),
            });
        }

        // Consume budget
        self.budget.consume(
            query_id,
            epsilon,
            self.config.default_delta,
            "private_mean",
            self.current_time,
        )?;

        // Clip values
        let clipped: Vec<f64> = values.iter()
            .map(|v| v.clamp(bounds.0, bounds.1))
            .collect();

        // Compute true mean
        let true_mean: f64 = clipped.iter().sum::<f64>() / clipped.len() as f64;

        // Sensitivity for mean = (max - min) / n
        let sensitivity = (bounds.1 - bounds.0) / clipped.len() as f64;

        // Add noise
        Ok(self.noise_gen.add_noise(true_mean, sensitivity, epsilon))
    }

    /// Compute private sum
    pub fn private_sum(
        &mut self,
        values: &[f64],
        bounds: (f64, f64),
        epsilon: f64,
        query_id: &str,
    ) -> Result<f64, PrivacyError> {
        if values.len() < self.config.min_samples {
            return Err(PrivacyError::InsufficientSamples {
                required: self.config.min_samples,
                provided: values.len(),
            });
        }

        self.budget.consume(
            query_id,
            epsilon,
            self.config.default_delta,
            "private_sum",
            self.current_time,
        )?;

        // Clip values
        let clipped: Vec<f64> = values.iter()
            .map(|v| v.clamp(bounds.0, bounds.1))
            .collect();

        // Compute true sum
        let true_sum: f64 = clipped.iter().sum();

        // Sensitivity for sum = max - min (one person's contribution)
        let sensitivity = bounds.1 - bounds.0;

        Ok(self.noise_gen.add_noise(true_sum, sensitivity, epsilon))
    }

    /// Compute private count
    pub fn private_count(
        &mut self,
        count: usize,
        epsilon: f64,
        query_id: &str,
    ) -> Result<f64, PrivacyError> {
        self.budget.consume(
            query_id,
            epsilon,
            self.config.default_delta,
            "private_count",
            self.current_time,
        )?;

        // Sensitivity for count = 1
        Ok(self.noise_gen.add_noise(count as f64, 1.0, epsilon))
    }

    /// Compute private histogram
    pub fn private_histogram(
        &mut self,
        values: &[f64],
        bins: &[f64],
        epsilon: f64,
        query_id: &str,
    ) -> Result<Vec<f64>, PrivacyError> {
        if values.len() < self.config.min_samples {
            return Err(PrivacyError::InsufficientSamples {
                required: self.config.min_samples,
                provided: values.len(),
            });
        }

        self.budget.consume(
            query_id,
            epsilon,
            self.config.default_delta,
            "private_histogram",
            self.current_time,
        )?;

        // Build histogram
        let mut histogram = vec![0.0; bins.len() - 1];

        for value in values {
            for (i, window) in bins.windows(2).enumerate() {
                if *value >= window[0] && *value < window[1] {
                    histogram[i] += 1.0;
                    break;
                }
            }
        }

        // Add noise to each bin (sensitivity = 1 per bin)
        let epsilon_per_bin = epsilon / histogram.len() as f64;
        Ok(self.noise_gen.add_noise_vector(&histogram, 1.0, epsilon_per_bin))
    }

    /// Compute private percentile
    pub fn private_percentile(
        &mut self,
        values: &[f64],
        percentile: f64,
        bounds: (f64, f64),
        epsilon: f64,
        query_id: &str,
    ) -> Result<f64, PrivacyError> {
        if values.len() < self.config.min_samples {
            return Err(PrivacyError::InsufficientSamples {
                required: self.config.min_samples,
                provided: values.len(),
            });
        }

        self.budget.consume(
            query_id,
            epsilon,
            self.config.default_delta,
            "private_percentile",
            self.current_time,
        )?;

        // Clip and sort values
        let mut sorted: Vec<f64> = values.iter()
            .map(|v| v.clamp(bounds.0, bounds.1))
            .collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find percentile index
        let idx = ((percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        let true_percentile = sorted[idx.min(sorted.len() - 1)];

        // Sensitivity for percentile ≈ (max - min) / n
        let sensitivity = (bounds.1 - bounds.0) / sorted.len() as f64;

        Ok(self.noise_gen.add_noise(true_percentile, sensitivity, epsilon))
    }

    /// Compute private variance
    pub fn private_variance(
        &mut self,
        values: &[f64],
        bounds: (f64, f64),
        epsilon: f64,
        query_id: &str,
    ) -> Result<f64, PrivacyError> {
        if values.len() < self.config.min_samples {
            return Err(PrivacyError::InsufficientSamples {
                required: self.config.min_samples,
                provided: values.len(),
            });
        }

        self.budget.consume(
            query_id,
            epsilon,
            self.config.default_delta,
            "private_variance",
            self.current_time,
        )?;

        // Clip values
        let clipped: Vec<f64> = values.iter()
            .map(|v| v.clamp(bounds.0, bounds.1))
            .collect();

        let n = clipped.len() as f64;
        let mean: f64 = clipped.iter().sum::<f64>() / n;
        let true_variance: f64 = clipped.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / n;

        // Sensitivity for variance ≈ (max - min)^2 / n
        let range = bounds.1 - bounds.0;
        let sensitivity = range * range / n;

        Ok(self.noise_gen.add_noise(true_variance, sensitivity, epsilon).max(0.0))
    }

    /// Get remaining budget
    pub fn remaining_budget(&self) -> (f64, f64) {
        (self.budget.remaining_epsilon(), self.budget.remaining_delta())
    }

    /// Reset budget (new epoch)
    pub fn reset_budget(&mut self) {
        self.budget.reset();
    }
}

// ============================================================================
// Local Differential Privacy
// ============================================================================

/// Local DP for per-agent privacy
#[derive(Debug)]
pub struct LocalDP {
    epsilon: f64,
    rng: DPRng,
}

impl LocalDP {
    pub fn new(epsilon: f64, seed: u64) -> Self {
        Self {
            epsilon,
            rng: DPRng::new(seed),
        }
    }

    /// Randomized response for binary value
    pub fn randomized_response(&mut self, true_value: bool) -> bool {
        let p = 1.0 / (1.0 + self.epsilon.exp());
        if self.rng.uniform() < p {
            !true_value // Flip
        } else {
            true_value // Keep
        }
    }

    /// Unary encoding for categorical
    pub fn unary_encoding(&mut self, category: usize, num_categories: usize) -> Vec<bool> {
        let p = self.epsilon.exp() / (1.0 + self.epsilon.exp());
        let q = 1.0 / (1.0 + self.epsilon.exp());

        (0..num_categories)
            .map(|i| {
                if i == category {
                    self.rng.uniform() < p
                } else {
                    self.rng.uniform() < q
                }
            })
            .collect()
    }

    /// Perturb numeric value with Laplace noise
    pub fn perturb_numeric(&mut self, value: f64, sensitivity: f64) -> f64 {
        let scale = sensitivity / self.epsilon;
        value + self.rng.laplace(scale)
    }
}

// ============================================================================
// Trust-Specific Aggregations
// ============================================================================

/// Private trust analytics
#[derive(Debug)]
pub struct PrivateTrustAnalytics {
    aggregator: PrivateAggregator,
}

impl PrivateTrustAnalytics {
    pub fn new(config: DPConfig, seed: u64) -> Self {
        Self {
            aggregator: PrivateAggregator::new(config, seed),
        }
    }

    /// Private average trust score
    pub fn average_trust(&mut self, trust_scores: &[f64], epsilon: f64) -> Result<f64, PrivacyError> {
        self.aggregator.private_mean(
            trust_scores,
            (0.0, 1.0),
            epsilon,
            &format!("avg_trust_{}", self.aggregator.current_time),
        )
    }

    /// Private trust distribution (histogram)
    pub fn trust_distribution(&mut self, trust_scores: &[f64], epsilon: f64) -> Result<TrustDistribution, PrivacyError> {
        let bins = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let histogram = self.aggregator.private_histogram(
            trust_scores,
            &bins,
            epsilon,
            &format!("trust_dist_{}", self.aggregator.current_time),
        )?;

        Ok(TrustDistribution {
            bins: bins.windows(2).map(|w| (w[0], w[1])).collect(),
            counts: histogram,
            epsilon_used: epsilon,
        })
    }

    /// Private agent count by trust tier
    pub fn agents_by_tier(&mut self, trust_scores: &[f64], epsilon: f64) -> Result<HashMap<String, f64>, PrivacyError> {
        let tiers: HashMap<String, (f64, f64)> = [
            ("low".to_string(), (0.0, 0.33)),
            ("medium".to_string(), (0.33, 0.67)),
            ("high".to_string(), (0.67, 1.0)),
        ].into_iter().collect();

        let mut result = HashMap::new();
        let epsilon_per_tier = epsilon / tiers.len() as f64;

        for (tier_name, (min, max)) in &tiers {
            let count = trust_scores.iter()
                .filter(|&&v| v >= *min && v < *max)
                .count();

            let private_count = self.aggregator.private_count(
                count,
                epsilon_per_tier,
                &format!("tier_{}_{}", tier_name, self.aggregator.current_time),
            )?;

            result.insert(tier_name.clone(), private_count.max(0.0));
        }

        Ok(result)
    }

    /// Private median trust
    pub fn median_trust(&mut self, trust_scores: &[f64], epsilon: f64) -> Result<f64, PrivacyError> {
        self.aggregator.private_percentile(
            trust_scores,
            50.0,
            (0.0, 1.0),
            epsilon,
            &format!("median_trust_{}", self.aggregator.current_time),
        )
    }

    /// Private trust variance
    pub fn trust_variance(&mut self, trust_scores: &[f64], epsilon: f64) -> Result<f64, PrivacyError> {
        self.aggregator.private_variance(
            trust_scores,
            (0.0, 1.0),
            epsilon,
            &format!("trust_var_{}", self.aggregator.current_time),
        )
    }

    /// Set current time
    pub fn set_time(&mut self, time: u64) {
        self.aggregator.set_time(time);
    }

    /// Get remaining budget
    pub fn remaining_budget(&self) -> (f64, f64) {
        self.aggregator.remaining_budget()
    }
}

/// Trust distribution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustDistribution {
    pub bins: Vec<(f64, f64)>,
    pub counts: Vec<f64>,
    pub epsilon_used: f64,
}

// ============================================================================
// Errors
// ============================================================================

/// Privacy-related errors
#[derive(Debug, Clone)]
pub enum PrivacyError {
    BudgetExhausted {
        requested_epsilon: f64,
        requested_delta: f64,
        remaining_epsilon: f64,
        remaining_delta: f64,
    },
    InsufficientSamples {
        required: usize,
        provided: usize,
    },
    InvalidParameters {
        message: String,
    },
}

impl std::fmt::Display for PrivacyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetExhausted {
                requested_epsilon,
                requested_delta,
                remaining_epsilon,
                remaining_delta,
            } => write!(
                f,
                "Privacy budget exhausted: requested ({}, {}), remaining ({}, {})",
                requested_epsilon, requested_delta, remaining_epsilon, remaining_delta
            ),
            Self::InsufficientSamples { required, provided } => {
                write!(f, "Insufficient samples: {} required, {} provided", required, provided)
            }
            Self::InvalidParameters { message } => write!(f, "Invalid parameters: {}", message),
        }
    }
}

impl std::error::Error for PrivacyError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_noise() {
        let mut gen = NoiseGenerator::new(42, NoiseMechanism::Laplace);

        let mut sum = 0.0;
        let n = 10000;

        for _ in 0..n {
            sum += gen.add_noise(0.0, 1.0, 1.0);
        }

        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "Laplace noise should center around 0");
    }

    #[test]
    fn test_gaussian_noise() {
        let mut gen = NoiseGenerator::new(42, NoiseMechanism::Gaussian);

        let mut sum = 0.0;
        let n = 10000;

        for _ in 0..n {
            sum += gen.add_noise(0.0, 1.0, 1.0);
        }

        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "Gaussian noise should center around 0");
    }

    #[test]
    fn test_privacy_budget() {
        let mut budget = PrivacyBudget::new(5.0, 1e-5, false);

        // Should allow query
        assert!(budget.can_query(1.0, 1e-6));

        // Consume budget
        budget.consume("q1", 1.0, 1e-6, "test", 1000).unwrap();
        assert!((budget.remaining_epsilon() - 4.0).abs() < 0.01);

        // Should deny when exhausted
        assert!(!budget.can_query(5.0, 1e-5));
    }

    #[test]
    fn test_private_mean() {
        let mut agg = PrivateAggregator::new(DPConfig {
            min_samples: 5,
            ..Default::default()
        }, 42);

        let values: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let true_mean = values.iter().sum::<f64>() / values.len() as f64;

        let private_mean = agg.private_mean(&values, (0.0, 1.0), 1.0, "test").unwrap();

        // Should be close to true mean with reasonable epsilon
        assert!((private_mean - true_mean).abs() < 0.1);
    }

    #[test]
    fn test_private_histogram() {
        let mut agg = PrivateAggregator::new(DPConfig {
            min_samples: 5,
            ..Default::default()
        }, 42);

        let values: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let bins = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        let histogram = agg.private_histogram(&values, &bins, 1.0, "test").unwrap();

        assert_eq!(histogram.len(), 4);
        // Each bin should have roughly 25 elements (±noise)
        for count in &histogram {
            assert!(*count > 10.0 && *count < 40.0);
        }
    }

    #[test]
    fn test_local_dp_randomized_response() {
        let epsilon = 1.0;
        let mut ldp = LocalDP::new(epsilon, 42);

        let mut reported_true = 0;
        let n = 10000;

        for _ in 0..n {
            let true_val = true;
            if ldp.randomized_response(true_val) {
                reported_true += 1;
            }
        }

        // With randomized response at epsilon=1.0:
        // flip probability p = 1 / (1 + e^epsilon) ≈ 0.269
        // When all inputs are true, expected reported_true rate ≈ 1 - 0.269 = 0.731
        let reported_rate = reported_true as f64 / n as f64;
        let flip_prob = 1.0 / (1.0 + epsilon.exp());
        let expected_rate = 1.0 - flip_prob;

        // Check that reported rate is close to expected (within statistical margin)
        assert!(
            (reported_rate - expected_rate).abs() < 0.05,
            "reported_rate={}, expected_rate={}", reported_rate, expected_rate
        );

        // Also verify that noise was actually added (reported differs from true)
        assert!(reported_rate < 0.95, "Expected noise to be added, but reported_rate={}", reported_rate);
    }

    #[test]
    fn test_trust_analytics() {
        let mut analytics = PrivateTrustAnalytics::new(DPConfig {
            min_samples: 5,
            ..Default::default()
        }, 42);

        let trust_scores: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();

        let avg = analytics.average_trust(&trust_scores, 1.0).unwrap();
        assert!(avg > 0.3 && avg < 0.7);

        let dist = analytics.trust_distribution(&trust_scores, 1.0).unwrap();
        assert_eq!(dist.bins.len(), 5);
    }

    #[test]
    fn test_insufficient_samples() {
        let mut agg = PrivateAggregator::new(DPConfig {
            min_samples: 10,
            ..Default::default()
        }, 42);

        let values = vec![0.5, 0.6, 0.7]; // Only 3 samples

        let result = agg.private_mean(&values, (0.0, 1.0), 1.0, "test");
        assert!(matches!(result, Err(PrivacyError::InsufficientSamples { .. })));
    }

    #[test]
    fn test_budget_exhaustion() {
        let mut agg = PrivateAggregator::new(DPConfig {
            epoch_budget: 2.0,
            min_samples: 5,
            ..Default::default()
        }, 42);

        let values: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();

        // First query should succeed
        agg.private_mean(&values, (0.0, 1.0), 1.5, "q1").unwrap();

        // Second query should fail (budget exhausted)
        let result = agg.private_mean(&values, (0.0, 1.0), 1.5, "q2");
        assert!(matches!(result, Err(PrivacyError::BudgetExhausted { .. })));
    }
}
