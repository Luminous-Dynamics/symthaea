// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federated CfC Weight Sharing Protocol
//!
//! This module implements federated learning for Closed-form Continuous-time (CfC)
//! neural network weights across the Symthaea swarm network. The protocol enables
//! distributed nodes to collaboratively improve their consciousness models while
//! preserving privacy through differential privacy and trust-weighted aggregation.
//!
//! # Key Features
//!
//! - **Trust-Weighted FedAvg**: Aggregates gradients weighted by Holochain trust scores
//! - **Differential Privacy**: Adds calibrated Gaussian noise to protect training data
//! - **Gradient Clipping**: Bounds sensitivity before adding noise
//! - **Byzantine Tolerance**: Optional trimmed mean for outlier rejection
//!
//! # Example
//!
//! ```rust,ignore
//! use symthaea::swarm::federated_cfc::{FederatedAggregator, GradientMessage};
//!
//! // Initialize with local model weights
//! let mut aggregator = FederatedAggregator::new(local_weights);
//!
//! // Receive gradients from peers
//! aggregator.receive_gradient(peer_gradient);
//!
//! // Aggregate when ready
//! let new_weights = aggregator.aggregate();
//!
//! // Export local gradient with differential privacy
//! let my_gradient = aggregator.export_local_gradient(0.1);
//! ```

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

// ============================================================================
// GRADIENT MESSAGE
// ============================================================================

/// A gradient update message for federated CfC learning
///
/// This is the core data structure exchanged between nodes during federated
/// learning rounds. It contains the gradient data along with metadata for
/// trust-weighted aggregation and privacy parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientMessage {
    /// Unique identifier for the source node (32-byte hash)
    pub source_id: [u8; 32],

    /// Gradient data as flattened f32 values
    ///
    /// For CfC networks, this typically includes gradients for:
    /// - W_in (input weights)
    /// - W_h (recurrent weights)
    /// - b_h (bias)
    /// - tau (time constants)
    /// - Output projection weights
    pub gradient_data: Vec<f32>,

    /// Trust score from Holochain DHT (0.0 to 1.0)
    ///
    /// This value is used to weight the gradient's contribution during
    /// aggregation. Higher trust = more influence on the aggregated result.
    pub trust_score: f32,

    /// Differential privacy noise scale (sigma)
    ///
    /// If > 0, Gaussian noise with this standard deviation was added to the
    /// gradients before export. Combined with gradient clipping, this provides
    /// (epsilon, delta)-differential privacy guarantees.
    pub noise_scale: f32,

    /// Timestamp when gradient was computed (ms since Unix epoch)
    pub timestamp: u64,

    /// Number of local training samples this gradient represents
    ///
    /// Used for sample-weighted aggregation (more samples = higher weight).
    pub sample_count: u64,

    /// Model version this gradient was computed against
    ///
    /// Used for stale gradient detection and version compatibility.
    pub model_version: u64,
}

impl GradientMessage {
    /// Create a new gradient message
    pub fn new(source_id: [u8; 32], gradient_data: Vec<f32>, trust_score: f32) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            source_id,
            gradient_data,
            trust_score,
            noise_scale: 0.0,
            timestamp,
            sample_count: 1,
            model_version: 0,
        }
    }

    /// Set the sample count
    pub fn with_sample_count(mut self, count: u64) -> Self {
        self.sample_count = count;
        self
    }

    /// Set the model version
    pub fn with_model_version(mut self, version: u64) -> Self {
        self.model_version = version;
        self
    }

    /// Set the noise scale (after DP has been applied)
    pub fn with_noise_scale(mut self, scale: f32) -> Self {
        self.noise_scale = scale;
        self
    }

    /// Get the dimension of the gradient
    pub fn dim(&self) -> usize {
        self.gradient_data.len()
    }
}

// ============================================================================
// DIFFERENTIAL PRIVACY PARAMETERS
// ============================================================================

/// Configuration for differential privacy
#[derive(Debug, Clone, Copy)]
pub struct DifferentialPrivacyConfig {
    /// L2 norm bound for gradient clipping
    ///
    /// Gradients are clipped to have at most this L2 norm before adding noise.
    /// Lower values = stronger privacy but potentially worse utility.
    pub clip_norm: f32,

    /// Noise multiplier (scales with clip_norm to determine sigma)
    ///
    /// sigma = clip_norm * noise_multiplier
    /// Higher values = stronger privacy guarantees.
    pub noise_multiplier: f64,
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            clip_norm: 1.0,
            noise_multiplier: 1.1,
        }
    }
}

impl DifferentialPrivacyConfig {
    /// Create a new configuration with specified parameters
    pub fn new(clip_norm: f32, noise_multiplier: f64) -> Self {
        Self {
            clip_norm,
            noise_multiplier,
        }
    }

    /// Get the noise standard deviation (sigma)
    pub fn sigma(&self) -> f64 {
        self.clip_norm as f64 * self.noise_multiplier
    }

    /// Create a high-privacy configuration (epsilon ~0.1)
    pub fn high_privacy() -> Self {
        Self {
            clip_norm: 0.5,
            noise_multiplier: 2.0,
        }
    }

    /// Create a moderate-privacy configuration (epsilon ~1.0)
    pub fn moderate_privacy() -> Self {
        Self {
            clip_norm: 1.0,
            noise_multiplier: 1.1,
        }
    }

    /// Create a low-privacy configuration (epsilon ~10.0)
    pub fn low_privacy() -> Self {
        Self {
            clip_norm: 2.0,
            noise_multiplier: 0.5,
        }
    }
}

// ============================================================================
// FEDERATED AGGREGATOR
// ============================================================================

/// Federated aggregator for CfC gradient updates
///
/// This struct manages the aggregation of gradient updates from multiple peers
/// using trust-weighted Federated Averaging (FedAvg). It supports differential
/// privacy for outgoing gradients and optional Byzantine fault tolerance.
pub struct FederatedAggregator {
    /// Current local model weights
    local_weights: Vec<f32>,

    /// Buffer of received peer contributions with their computed weights
    ///
    /// Each entry is (GradientMessage, trust_weight) where trust_weight is
    /// the computed weight for this gradient in the aggregation.
    peer_contributions: Vec<(GradientMessage, f32)>,

    /// Current aggregation round (increments after each aggregation)
    aggregation_round: u64,

    /// Local node's source ID
    local_source_id: [u8; 32],

    /// Differential privacy configuration (None = no DP)
    dp_config: Option<DifferentialPrivacyConfig>,

    /// Enable Byzantine fault tolerance via trimmed mean
    byzantine_tolerance: bool,

    /// Fraction of extreme values to trim (0.0 to 0.5)
    trim_fraction: f32,

    /// Maximum staleness in milliseconds (gradients older than this are rejected)
    max_staleness_ms: u64,
}

impl FederatedAggregator {
    /// Create a new federated aggregator with initial weights
    ///
    /// # Arguments
    ///
    /// * `initial_weights` - The initial model weights
    ///
    /// # Example
    ///
    /// ```rust
    /// use symthaea::swarm::federated_cfc::FederatedAggregator;
    ///
    /// let weights = vec![0.1, 0.2, 0.3, 0.4];
    /// let aggregator = FederatedAggregator::new(weights);
    /// assert_eq!(aggregator.local_weights().len(), 4);
    /// ```
    pub fn new(initial_weights: Vec<f32>) -> Self {
        // Generate a random source ID
        let mut source_id = [0u8; 32];
        let mut rng = rand::thread_rng();
        rng.fill(&mut source_id);

        Self {
            local_weights: initial_weights,
            peer_contributions: Vec::new(),
            aggregation_round: 0,
            local_source_id: source_id,
            dp_config: None,
            byzantine_tolerance: false,
            trim_fraction: 0.1,
            max_staleness_ms: 60_000, // 1 minute default
        }
    }

    /// Enable differential privacy with the given configuration
    pub fn with_differential_privacy(mut self, config: DifferentialPrivacyConfig) -> Self {
        self.dp_config = Some(config);
        self
    }

    /// Enable Byzantine fault tolerance with trimmed mean
    pub fn with_byzantine_tolerance(mut self, trim_fraction: f32) -> Self {
        self.byzantine_tolerance = true;
        self.trim_fraction = trim_fraction.clamp(0.0, 0.5);
        self
    }

    /// Set maximum staleness for incoming gradients
    pub fn with_max_staleness(mut self, max_staleness_ms: u64) -> Self {
        self.max_staleness_ms = max_staleness_ms;
        self
    }

    /// Set the local source ID
    pub fn with_source_id(mut self, source_id: [u8; 32]) -> Self {
        self.local_source_id = source_id;
        self
    }

    /// Get reference to local weights
    pub fn local_weights(&self) -> &[f32] {
        &self.local_weights
    }

    /// Get the current aggregation round
    pub fn aggregation_round(&self) -> u64 {
        self.aggregation_round
    }

    /// Get the number of pending peer contributions
    pub fn pending_contributions(&self) -> usize {
        self.peer_contributions.len()
    }

    /// Receive a gradient update from a peer
    ///
    /// The gradient is validated and added to the contribution buffer.
    /// Call `aggregate()` when enough gradients have been received.
    ///
    /// # Arguments
    ///
    /// * `msg` - The gradient message from a peer
    ///
    /// # Returns
    ///
    /// Returns `true` if the gradient was accepted, `false` if rejected.
    ///
    /// # Example
    ///
    /// ```rust
    /// use symthaea::swarm::federated_cfc::{FederatedAggregator, GradientMessage};
    ///
    /// let mut aggregator = FederatedAggregator::new(vec![0.0; 10]);
    ///
    /// let msg = GradientMessage::new([1u8; 32], vec![0.1; 10], 0.8);
    /// assert!(aggregator.receive_gradient(msg));
    /// assert_eq!(aggregator.pending_contributions(), 1);
    /// ```
    pub fn receive_gradient(&mut self, msg: GradientMessage) -> bool {
        // Validate dimension matches local weights
        if msg.gradient_data.len() != self.local_weights.len() {
            tracing::warn!(
                "Rejected gradient: dimension mismatch ({} vs {})",
                msg.gradient_data.len(),
                self.local_weights.len()
            );
            return false;
        }

        // Validate trust score
        if msg.trust_score < 0.0 || msg.trust_score > 1.0 {
            tracing::warn!("Rejected gradient: invalid trust score {}", msg.trust_score);
            return false;
        }

        // Check staleness
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        if now.saturating_sub(msg.timestamp) > self.max_staleness_ms {
            tracing::warn!(
                "Rejected gradient: too stale (age: {}ms, max: {}ms)",
                now.saturating_sub(msg.timestamp),
                self.max_staleness_ms
            );
            return false;
        }

        // Compute aggregation weight based on trust
        let trust_weight = msg.trust_score;

        // Add to contributions
        self.peer_contributions.push((msg, trust_weight));
        true
    }

    /// Perform trust-weighted FedAvg aggregation
    ///
    /// This computes the weighted average of all received gradients and applies
    /// them to the local weights. The weights are normalized so they sum to 1.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// w_new = Σ(trust_i * grad_i) / Σ(trust_i)
    /// local_weights += learning_rate * w_new
    /// ```
    ///
    /// # Returns
    ///
    /// The aggregated gradient as a vector, or None if no contributions are pending.
    ///
    /// # Example
    ///
    /// ```rust
    /// use symthaea::swarm::federated_cfc::{FederatedAggregator, GradientMessage};
    ///
    /// let mut aggregator = FederatedAggregator::new(vec![0.0; 4]);
    ///
    /// // Add gradients from two peers with equal trust
    /// let msg1 = GradientMessage::new([1u8; 32], vec![0.2, 0.4, 0.6, 0.8], 0.5);
    /// let msg2 = GradientMessage::new([2u8; 32], vec![0.4, 0.2, 0.4, 0.6], 0.5);
    /// aggregator.receive_gradient(msg1);
    /// aggregator.receive_gradient(msg2);
    ///
    /// // Aggregate - should return average
    /// let result = aggregator.aggregate();
    /// assert!(result.is_some());
    /// ```
    pub fn aggregate(&mut self) -> Option<Vec<f32>> {
        if self.peer_contributions.is_empty() {
            return None;
        }

        let dim = self.local_weights.len();

        // Apply Byzantine tolerance if enabled
        let contributions = if self.byzantine_tolerance && self.peer_contributions.len() >= 4 {
            self.apply_trimmed_mean()
        } else {
            self.peer_contributions.clone()
        };

        if contributions.is_empty() {
            return None;
        }

        // Normalize weights so they sum to 1
        let total_weight: f32 = contributions.iter().map(|(_, w)| w).sum();

        if total_weight <= 0.0 {
            return None;
        }

        // Compute weighted average
        let mut aggregated = vec![0.0f32; dim];

        for (msg, weight) in &contributions {
            let normalized_weight = weight / total_weight;
            for (i, &g) in msg.gradient_data.iter().enumerate() {
                aggregated[i] += g * normalized_weight;
            }
        }

        // Clear contributions and increment round
        self.peer_contributions.clear();
        self.aggregation_round += 1;

        Some(aggregated)
    }

    /// Aggregate using the unified mycelix-fl-core pipeline with plugin support.
    ///
    /// Converts Symthaea's `GradientMessage` contributions to core `GradientUpdate`
    /// types, runs the full pipeline (validate → DP → detect → trim → aggregate),
    /// and returns the aggregated gradient.
    ///
    /// This is the recommended aggregation path for production use, as it provides
    /// multi-signal Byzantine detection, hybrid BFT, and plugin extensibility.
    ///
    /// Requires the `mycelix` feature flag.
    #[cfg(feature = "mycelix")]
    pub fn aggregate_with_pipeline(
        &mut self,
        pipeline: &mut mycelix_fl_core::UnifiedPipeline,
    ) -> Option<mycelix_fl_core::PipelineResult> {
        if self.peer_contributions.is_empty() {
            return None;
        }

        // Convert GradientMessage → GradientUpdate
        let updates: Vec<mycelix_fl_core::GradientUpdate> = self
            .peer_contributions
            .iter()
            .map(|(msg, _weight)| {
                let pid = hex::encode(&msg.source_id[..8]);
                mycelix_fl_core::GradientUpdate::new(
                    pid,
                    msg.model_version,
                    msg.gradient_data.clone(),
                    msg.sample_count as u32,
                    0.5, // loss not tracked in GradientMessage
                )
            })
            .collect();

        // Build reputation map from trust scores
        let reputations: std::collections::HashMap<String, f32> = self
            .peer_contributions
            .iter()
            .map(|(msg, _weight)| {
                let pid = hex::encode(&msg.source_id[..8]);
                (pid, msg.trust_score)
            })
            .collect();

        match pipeline.aggregate(&updates, &reputations) {
            Ok(result) => {
                self.peer_contributions.clear();
                self.aggregation_round += 1;
                Some(result)
            }
            Err(_) => None,
        }
    }

    /// Apply trimmed mean for Byzantine fault tolerance
    ///
    /// This removes the top and bottom `trim_fraction` of values for each
    /// dimension before computing the mean. This provides robustness against
    /// malicious or faulty nodes sending extreme gradients.
    fn apply_trimmed_mean(&self) -> Vec<(GradientMessage, f32)> {
        let n = self.peer_contributions.len();
        let trim_count = (n as f32 * self.trim_fraction) as usize;

        if trim_count * 2 >= n {
            // Not enough samples to trim
            return self.peer_contributions.clone();
        }

        let dim = self.local_weights.len();

        // For efficiency, we sample a subset of dimensions
        let sample_step = (dim / 10).max(1);
        let sample_dims: Vec<usize> = (0..dim).step_by(sample_step).collect();
        let num_sample_dims = sample_dims.len();

        // Track which contributions to keep
        let mut outlier_counts = vec![0usize; n];

        for d in &sample_dims {
            // Get values at this dimension with their indices
            let mut values: Vec<(usize, f32)> = self
                .peer_contributions
                .iter()
                .enumerate()
                .map(|(i, (msg, _))| (i, msg.gradient_data[*d]))
                .collect();

            // Sort by value
            values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Mark extreme values as outliers
            for i in 0..trim_count {
                outlier_counts[values[i].0] += 1;
                outlier_counts[values[n - 1 - i].0] += 1;
            }
        }

        // Keep contributions that weren't frequently marked as outliers
        let threshold = num_sample_dims / 2;
        self.peer_contributions
            .iter()
            .enumerate()
            .filter(|(i, _)| outlier_counts[*i] < threshold)
            .map(|(_, c)| c.clone())
            .collect()
    }

    /// Export local gradient with differential privacy
    ///
    /// This method prepares the local gradient for sharing with peers. If
    /// differential privacy is enabled, it:
    /// 1. Clips the gradient to bound its L2 norm
    /// 2. Adds calibrated Gaussian noise
    ///
    /// # Arguments
    ///
    /// * `noise_scale` - Override the DP config noise scale (use 0.0 for config default)
    ///
    /// # Returns
    ///
    /// A `GradientMessage` ready for broadcast to peers.
    ///
    /// # Example
    ///
    /// ```rust
    /// use symthaea::swarm::federated_cfc::{FederatedAggregator, DifferentialPrivacyConfig};
    ///
    /// let aggregator = FederatedAggregator::new(vec![0.5, -0.3, 0.8, -0.2])
    ///     .with_differential_privacy(DifferentialPrivacyConfig::moderate_privacy());
    ///
    /// let msg = aggregator.export_local_gradient(0.1);
    /// assert_eq!(msg.gradient_data.len(), 4);
    /// assert!(msg.noise_scale > 0.0);
    /// ```
    pub fn export_local_gradient(&self, noise_scale: f32) -> GradientMessage {
        let mut gradient = self.local_weights.clone();

        // Determine effective noise scale
        let effective_noise_scale = if noise_scale > 0.0 {
            noise_scale
        } else if let Some(ref config) = self.dp_config {
            config.sigma() as f32
        } else {
            0.0
        };

        // Apply differential privacy if enabled
        if effective_noise_scale > 0.0 {
            // Get clip norm from config or default
            let clip_norm = self.dp_config.as_ref().map(|c| c.clip_norm).unwrap_or(1.0);

            // Step 1: Clip gradient norm
            clip_gradient(&mut gradient, clip_norm);

            // Step 2: Add Gaussian noise
            add_gaussian_noise(&mut gradient, effective_noise_scale);
        }

        GradientMessage {
            source_id: self.local_source_id,
            gradient_data: gradient,
            trust_score: 1.0, // Self-trust is always 1.0
            noise_scale: effective_noise_scale,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            sample_count: 1,
            model_version: self.aggregation_round,
        }
    }

    /// Update local weights directly
    ///
    /// This is typically called after local training to update the weights
    /// that will be exported to peers.
    pub fn update_local_weights(&mut self, weights: Vec<f32>) {
        self.local_weights = weights;
    }

    /// Apply an aggregated gradient to local weights
    ///
    /// # Arguments
    ///
    /// * `gradient` - The aggregated gradient
    /// * `learning_rate` - Learning rate for the update
    pub fn apply_gradient(&mut self, gradient: &[f32], learning_rate: f32) {
        if gradient.len() != self.local_weights.len() {
            return;
        }

        for (w, g) in self.local_weights.iter_mut().zip(gradient.iter()) {
            *w -= learning_rate * g;
        }
    }

    /// Apply weight deltas from FedAvg aggregation
    ///
    /// In FedAvg, clients compute delta = trained_weights - global_weights.
    /// After aggregation, the averaged delta should be **added** to global weights:
    /// new_weights = old_weights + aggregated_delta
    ///
    /// This is different from `apply_gradient` which subtracts.
    ///
    /// # Arguments
    ///
    /// * `delta` - The aggregated weight delta from FedAvg
    ///
    /// # Example
    ///
    /// ```rust
    /// use symthaea::swarm::federated_cfc::{FederatedAggregator, GradientMessage};
    ///
    /// let mut aggregator = FederatedAggregator::new(vec![1.0, 2.0, 3.0, 4.0]);
    ///
    /// // Simulate receiving weight deltas from clients
    /// let msg = GradientMessage::new([1u8; 32], vec![0.1, 0.2, 0.1, 0.2], 0.8);
    /// aggregator.receive_gradient(msg);
    ///
    /// if let Some(delta) = aggregator.aggregate() {
    ///     aggregator.apply_weight_delta(&delta);
    ///     // weights are now [1.1, 2.2, 3.1, 4.2]
    /// }
    /// ```
    pub fn apply_weight_delta(&mut self, delta: &[f32]) {
        if delta.len() != self.local_weights.len() {
            return;
        }

        for (w, d) in self.local_weights.iter_mut().zip(delta.iter()) {
            *w += d;
        }
    }

    /// Aggregate and apply weight deltas in one step
    ///
    /// This is the standard FedAvg update: aggregates all received weight deltas
    /// and applies them to the global model.
    ///
    /// # Returns
    ///
    /// Returns `true` if aggregation and update were successful, `false` if no
    /// contributions were pending.
    ///
    /// # Example
    ///
    /// ```rust
    /// use symthaea::swarm::federated_cfc::{FederatedAggregator, GradientMessage};
    ///
    /// let mut aggregator = FederatedAggregator::new(vec![0.0; 4]);
    ///
    /// // Receive gradients from multiple clients
    /// for i in 0..5 {
    ///     let msg = GradientMessage::new([i as u8; 32], vec![0.1; 4], 0.8);
    ///     aggregator.receive_gradient(msg);
    /// }
    ///
    /// // Aggregate and update global model
    /// let updated = aggregator.aggregate_and_apply();
    /// assert!(updated);
    /// ```
    pub fn aggregate_and_apply(&mut self) -> bool {
        if let Some(delta) = self.aggregate() {
            self.apply_weight_delta(&delta);
            true
        } else {
            false
        }
    }
}

// ============================================================================
// DIFFERENTIAL PRIVACY HELPERS
// ============================================================================

// DP helpers: delegate to mycelix-fl-core when available, standalone otherwise

#[cfg(feature = "mycelix")]
fn clip_gradient(gradient: &mut [f32], clip_norm: f32) {
    mycelix_fl_core::privacy::clip_gradient(gradient, clip_norm);
}

#[cfg(feature = "mycelix")]
fn add_gaussian_noise(gradient: &mut [f32], sigma: f32) {
    mycelix_fl_core::privacy::add_gaussian_noise(gradient, sigma);
}

#[cfg(feature = "mycelix")]
#[allow(dead_code)] // RESERVED(federated-learning): FL aggregation parameters
fn l2_norm(v: &[f32]) -> f32 {
    mycelix_fl_core::privacy::l2_norm(v)
}

#[cfg(not(feature = "mycelix"))]
fn clip_gradient(gradient: &mut [f32], clip_norm: f32) {
    let norm: f32 = gradient.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > clip_norm && norm > 0.0 {
        let scale = clip_norm / norm;
        for v in gradient.iter_mut() {
            *v *= scale;
        }
    }
}

#[cfg(not(feature = "mycelix"))]
fn add_gaussian_noise(gradient: &mut [f32], sigma: f32) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for v in gradient.iter_mut() {
        // Box-Muller transform for Gaussian noise
        let u1: f32 = rng.r#gen::<f32>().max(f32::EPSILON);
        let u2: f32 = rng.r#gen();
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        *v += sigma * normal;
    }
}

#[cfg(not(feature = "mycelix"))]
#[allow(dead_code)] // RESERVED(federated-learning): FL privacy parameters
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregation_with_equal_trust() {
        // Create aggregator with zero initial weights
        let mut aggregator = FederatedAggregator::new(vec![0.0; 4]);

        // Create two gradient messages with equal trust (0.5)
        let msg1 = GradientMessage::new([1u8; 32], vec![0.2, 0.4, 0.6, 0.8], 0.5);
        let msg2 = GradientMessage::new([2u8; 32], vec![0.4, 0.2, 0.4, 0.6], 0.5);

        // Receive both gradients
        assert!(aggregator.receive_gradient(msg1));
        assert!(aggregator.receive_gradient(msg2));
        assert_eq!(aggregator.pending_contributions(), 2);

        // Aggregate - should be simple average with equal trust
        let result = aggregator.aggregate().expect("aggregation should succeed");

        // Expected: (0.2+0.4)/2, (0.4+0.2)/2, (0.6+0.4)/2, (0.8+0.6)/2
        // = [0.3, 0.3, 0.5, 0.7]
        let expected = [0.3, 0.3, 0.5, 0.7];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-6,
                "Expected {}, got {} (diff: {})",
                e,
                r,
                (r - e).abs()
            );
        }

        // Contributions should be cleared
        assert_eq!(aggregator.pending_contributions(), 0);
        assert_eq!(aggregator.aggregation_round(), 1);
    }

    #[test]
    fn test_trust_weighting() {
        let mut aggregator = FederatedAggregator::new(vec![0.0; 2]);

        // High trust peer: [1.0, 0.0] with trust 0.8
        let msg_high = GradientMessage::new([1u8; 32], vec![1.0, 0.0], 0.8);

        // Low trust peer: [0.0, 1.0] with trust 0.2
        let msg_low = GradientMessage::new([2u8; 32], vec![0.0, 1.0], 0.2);

        aggregator.receive_gradient(msg_high);
        aggregator.receive_gradient(msg_low);

        let result = aggregator.aggregate().expect("aggregation should succeed");

        // Weights: 0.8 and 0.2, total = 1.0
        // Normalized: 0.8 and 0.2
        // Result[0] = 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        // Result[1] = 0.8 * 0.0 + 0.2 * 1.0 = 0.2
        assert!(
            (result[0] - 0.8).abs() < 1e-6,
            "Expected 0.8, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.2).abs() < 1e-6,
            "Expected 0.2, got {}",
            result[1]
        );
    }

    #[test]
    fn test_differential_privacy_noise() {
        // Create aggregator with known weights and DP enabled
        let aggregator = FederatedAggregator::new(vec![1.0, 2.0, 3.0, 4.0])
            .with_differential_privacy(DifferentialPrivacyConfig::new(1.0, 1.0));

        // Export gradient multiple times and check variance
        let mut samples: Vec<Vec<f32>> = Vec::new();
        for _ in 0..100 {
            let msg = aggregator.export_local_gradient(0.0);
            samples.push(msg.gradient_data.clone());
        }

        // Compute variance for each dimension
        for dim in 0..4 {
            let values: Vec<f32> = samples.iter().map(|s| s[dim]).collect();
            if values.is_empty() {
                continue;
            }
            let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
            let variance: f32 =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

            // With sigma=1.0, variance should be approximately 1.0
            // Allow some tolerance due to finite samples
            assert!(
                variance > 0.5 && variance < 2.0,
                "Variance {} out of expected range [0.5, 2.0] for dim {}",
                variance,
                dim
            );
        }

        // Check that noise_scale is recorded
        let msg = aggregator.export_local_gradient(0.0);
        assert!(
            msg.noise_scale > 0.0,
            "noise_scale should be positive with DP enabled"
        );
    }

    #[test]
    fn test_gradient_clipping() {
        // Create gradient with large norm
        let mut gradient = vec![3.0, 4.0]; // L2 norm = 5.0

        // Clip to norm 1.0
        clip_gradient(&mut gradient, 1.0);

        let norm = l2_norm(&gradient);
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Clipped norm should be 1.0, got {}",
            norm
        );

        // Direction should be preserved
        assert!(
            gradient[1].abs() > 1e-10,
            "gradient[1] too close to zero for ratio test"
        );
        let ratio = gradient[0] / gradient[1];
        assert!((ratio - 0.75).abs() < 1e-6, "Direction should be preserved");
    }

    #[test]
    fn test_no_clipping_when_below_threshold() {
        let mut gradient = vec![0.3, 0.4]; // L2 norm = 0.5

        clip_gradient(&mut gradient, 1.0);

        // Should not be modified
        assert!((gradient[0] - 0.3).abs() < 1e-6);
        assert!((gradient[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch_rejection() {
        let mut aggregator = FederatedAggregator::new(vec![0.0; 4]);

        // Try to add gradient with wrong dimension
        let msg = GradientMessage::new([1u8; 32], vec![0.1; 5], 0.5);
        assert!(!aggregator.receive_gradient(msg));
        assert_eq!(aggregator.pending_contributions(), 0);
    }

    #[test]
    fn test_invalid_trust_rejection() {
        let mut aggregator = FederatedAggregator::new(vec![0.0; 4]);

        // Trust > 1.0 should be rejected
        let msg = GradientMessage::new([1u8; 32], vec![0.1; 4], 1.5);
        assert!(!aggregator.receive_gradient(msg));

        // Trust < 0.0 should be rejected
        let msg = GradientMessage::new([1u8; 32], vec![0.1; 4], -0.1);
        assert!(!aggregator.receive_gradient(msg));
    }

    #[test]
    fn test_byzantine_tolerance() {
        let mut aggregator = FederatedAggregator::new(vec![0.0; 2]).with_byzantine_tolerance(0.2); // Trim 20% from each end

        // Add 5 gradients: 3 honest, 2 outliers
        let honest1 = GradientMessage::new([1u8; 32], vec![1.0, 1.0], 0.5);
        let honest2 = GradientMessage::new([2u8; 32], vec![1.1, 0.9], 0.5);
        let honest3 = GradientMessage::new([3u8; 32], vec![0.9, 1.1], 0.5);
        let outlier1 = GradientMessage::new([4u8; 32], vec![100.0, -100.0], 0.5);
        let outlier2 = GradientMessage::new([5u8; 32], vec![-100.0, 100.0], 0.5);

        aggregator.receive_gradient(honest1);
        aggregator.receive_gradient(honest2);
        aggregator.receive_gradient(honest3);
        aggregator.receive_gradient(outlier1);
        aggregator.receive_gradient(outlier2);

        let result = aggregator.aggregate().expect("aggregation should succeed");

        // With trimmed mean, outliers should be filtered out
        // Result should be close to [1.0, 1.0]
        assert!(
            result[0].abs() < 10.0,
            "Result[0] should not be dominated by outliers: {}",
            result[0]
        );
        assert!(
            result[1].abs() < 10.0,
            "Result[1] should not be dominated by outliers: {}",
            result[1]
        );
    }

    #[test]
    fn test_empty_aggregation() {
        let mut aggregator = FederatedAggregator::new(vec![0.0; 4]);

        // Aggregating with no contributions should return None
        let result = aggregator.aggregate();
        assert!(result.is_none());
    }

    #[test]
    fn test_apply_gradient() {
        let mut aggregator = FederatedAggregator::new(vec![1.0, 2.0, 3.0, 4.0]);

        let gradient = vec![0.1, 0.2, 0.3, 0.4];
        aggregator.apply_gradient(&gradient, 1.0);

        // w = w - lr * g = [1.0-0.1, 2.0-0.2, 3.0-0.3, 4.0-0.4]
        let expected = [0.9, 1.8, 2.7, 3.6];
        for (w, e) in aggregator.local_weights().iter().zip(expected.iter()) {
            assert!((w - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gradient_message_builder() {
        let msg = GradientMessage::new([0u8; 32], vec![1.0, 2.0], 0.7)
            .with_sample_count(100)
            .with_model_version(5)
            .with_noise_scale(0.5);

        assert_eq!(msg.sample_count, 100);
        assert_eq!(msg.model_version, 5);
        assert!((msg.noise_scale - 0.5).abs() < 1e-6);
        assert!(msg.timestamp > 0);
    }

    #[test]
    fn test_staleness_rejection() {
        let mut aggregator = FederatedAggregator::new(vec![0.0; 4]).with_max_staleness(1000); // 1 second

        // Create a message with old timestamp
        let mut msg = GradientMessage::new([1u8; 32], vec![0.1; 4], 0.5);
        msg.timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
            .saturating_sub(2000); // 2 seconds ago

        assert!(!aggregator.receive_gradient(msg));
    }

    #[test]
    fn test_export_without_dp() {
        let aggregator = FederatedAggregator::new(vec![1.0, 2.0, 3.0]);

        let msg = aggregator.export_local_gradient(0.0);

        // Without DP, gradient should match local weights exactly
        assert!((msg.noise_scale - 0.0).abs() < 1e-6);
        for (g, w) in msg.gradient_data.iter().zip(aggregator.local_weights()) {
            assert!((g - w).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dp_config_presets() {
        let high = DifferentialPrivacyConfig::high_privacy();
        assert!((high.clip_norm - 0.5).abs() < 1e-6);
        assert!((high.noise_multiplier - 2.0).abs() < 1e-6);

        let moderate = DifferentialPrivacyConfig::moderate_privacy();
        assert!((moderate.clip_norm - 1.0).abs() < 1e-6);
        assert!((moderate.noise_multiplier - 1.1).abs() < 1e-6);

        let low = DifferentialPrivacyConfig::low_privacy();
        assert!((low.clip_norm - 2.0).abs() < 1e-6);
        assert!((low.noise_multiplier - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_weight_delta() {
        let mut aggregator = FederatedAggregator::new(vec![1.0, 2.0, 3.0, 4.0]);

        let delta = vec![0.1, 0.2, 0.3, 0.4];
        aggregator.apply_weight_delta(&delta);

        // w = w + delta = [1.1, 2.2, 3.3, 4.4]
        let expected = [1.1, 2.2, 3.3, 4.4];
        for (w, e) in aggregator.local_weights().iter().zip(expected.iter()) {
            assert!((w - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_aggregate_and_apply() {
        let mut aggregator = FederatedAggregator::new(vec![1.0, 2.0, 3.0, 4.0]);

        // Add two deltas that should average to [0.2, 0.2, 0.2, 0.2]
        let msg1 = GradientMessage::new([1u8; 32], vec![0.1, 0.3, 0.1, 0.3], 0.5);
        let msg2 = GradientMessage::new([2u8; 32], vec![0.3, 0.1, 0.3, 0.1], 0.5);
        aggregator.receive_gradient(msg1);
        aggregator.receive_gradient(msg2);

        // Aggregate and apply
        let updated = aggregator.aggregate_and_apply();
        assert!(updated);

        // Expected: [1.0, 2.0, 3.0, 4.0] + [0.2, 0.2, 0.2, 0.2] = [1.2, 2.2, 3.2, 4.2]
        let expected = [1.2, 2.2, 3.2, 4.2];
        for (w, e) in aggregator.local_weights().iter().zip(expected.iter()) {
            assert!(
                (w - e).abs() < 1e-6,
                "Expected {}, got {} (diff: {})",
                e,
                w,
                (w - e).abs()
            );
        }
    }

    #[test]
    fn test_fedavg_convergence_simulation() {
        // Simulate a simple FedAvg scenario where all clients push weights toward 0.5
        let mut aggregator = FederatedAggregator::new(vec![0.0; 4]);

        // Run 5 rounds
        for _ in 0..5 {
            // Each client "trains" by computing delta toward target
            let target = 0.5f32;
            let current: Vec<f32> = aggregator.local_weights().to_vec();

            // Simulate 3 clients each moving 20% toward target
            for client_id in 0..3 {
                let delta: Vec<f32> = current.iter().map(|&w| (target - w) * 0.2).collect();

                let mut source_id = [0u8; 32];
                source_id[0] = client_id as u8;
                let msg = GradientMessage::new(source_id, delta, 0.8);
                aggregator.receive_gradient(msg);
            }

            aggregator.aggregate_and_apply();
        }

        // After 5 rounds, weights should be close to 0.5
        // (1 - 0.8^5) * 0.5 ≈ 0.34 (converging toward 0.5)
        for w in aggregator.local_weights() {
            assert!(*w > 0.3, "Weight {} should be > 0.3 after convergence", w);
        }
    }
}
