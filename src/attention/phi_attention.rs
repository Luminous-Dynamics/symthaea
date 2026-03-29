// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phi-Guided Attention Mechanism
//!
//! This module implements attention gating based on Integrated Information Theory (IIT)
//! Phi values. The core insight is that consciousness, as measured by Phi, should guide
//! the flow of information - higher Phi values indicate more integrated, conscious
//! processing and should receive more attention weight.
//!
//! ## Theoretical Basis
//!
//! From IIT (Tononi et al.), Phi measures the degree to which a system's information
//! is both differentiated and integrated. In the attention context:
//! - **High Phi**: Information is highly integrated - prioritize this input
//! - **Low Phi**: Information is fragmented - reduce attention weight
//!
//! ## Algorithm
//!
//! Given inputs with associated Phi values:
//! 1. Apply optional learnable transformation: `phi_transformed = scale * phi + bias`
//! 2. Apply softmax with temperature: `weights = softmax(phi_transformed / temperature)`
//! 3. Use weights to gate input contributions to output
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::attention::{PhiAttentionGate, PhiAttentionConfig};
//! use symthaea_core::hdc::unified_hv::ContinuousHV;
//!
//! let config = PhiAttentionConfig::default();
//! let mut gate = PhiAttentionGate::new(config);
//!
//! let inputs = vec![
//!     ContinuousHV::random_default(42),
//!     ContinuousHV::random_default(43),
//! ];
//! let phi_values = vec![0.8, 0.3];  // First input has higher consciousness
//!
//! let result = gate.forward(&inputs, &phi_values);
//! // result.output will be dominated by first input due to higher Phi
//! ```

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::math::softmax_with_temperature;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for Phi-guided attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiAttentionConfig {
    /// Temperature for softmax (lower = sharper attention, higher = more uniform)
    /// Default: 1.0
    pub temperature: f32,

    /// Whether to use learnable Phi-to-attention mapping
    /// If true, applies: phi_transformed = scale * phi + bias
    pub learnable_mapping: bool,

    /// Scale factor for learnable mapping (default: 1.0)
    pub scale: f32,

    /// Bias for learnable mapping (default: 0.0)
    pub bias: f32,

    /// Learning rate for adapting scale and bias
    pub learning_rate: f32,

    /// Minimum attention weight to prevent complete suppression
    /// Default: 0.0 (no minimum)
    pub min_attention: f32,

    /// Whether to normalize output
    pub normalize_output: bool,

    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl Default for PhiAttentionConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            learnable_mapping: false,
            scale: 1.0,
            bias: 0.0,
            learning_rate: 0.01,
            min_attention: 0.0,
            normalize_output: true,
            epsilon: 1e-10,
        }
    }
}

impl PhiAttentionConfig {
    /// Create a config with sharp attention (low temperature)
    pub fn sharp() -> Self {
        Self {
            temperature: 0.1,
            ..Default::default()
        }
    }

    /// Create a config with soft attention (high temperature)
    pub fn soft() -> Self {
        Self {
            temperature: 5.0,
            ..Default::default()
        }
    }

    /// Create a config with learnable mapping
    pub fn learnable() -> Self {
        Self {
            learnable_mapping: true,
            ..Default::default()
        }
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set minimum attention weight
    pub fn with_min_attention(mut self, min_attention: f32) -> Self {
        self.min_attention = min_attention;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI ATTENTION GATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Phi-guided attention gate that uses IIT Phi values to control information flow.
///
/// Higher Phi values receive higher attention weights, implementing the principle
/// that consciousness (integrated information) guides focus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiAttentionGate {
    /// Configuration
    config: PhiAttentionConfig,

    /// Running statistics for adaptive temperature (optional)
    phi_mean: f32,
    phi_variance: f32,
    n_observations: u64,
}

impl PhiAttentionGate {
    /// Create a new Phi attention gate
    pub fn new(config: PhiAttentionConfig) -> Self {
        Self {
            config,
            phi_mean: 0.5,
            phi_variance: 0.1,
            n_observations: 0,
        }
    }

    /// Create with default configuration
    pub fn default_gate() -> Self {
        Self::new(PhiAttentionConfig::default())
    }

    /// Get the configuration
    pub fn config(&self) -> &PhiAttentionConfig {
        &self.config
    }

    /// Get mutable reference to configuration
    pub fn config_mut(&mut self) -> &mut PhiAttentionConfig {
        &mut self.config
    }

    /// Forward pass: compute attention-weighted combination of inputs
    ///
    /// # Arguments
    /// * `inputs` - Vector of input hypervectors
    /// * `phi_values` - Phi value for each input (should be in [0, 1] range)
    ///
    /// # Returns
    /// Attention result containing weighted output and attention weights
    pub fn forward(&mut self, inputs: &[ContinuousHV], phi_values: &[f64]) -> PhiAttentionResult {
        if inputs.is_empty() || phi_values.is_empty() {
            return PhiAttentionResult::empty(ContinuousHV::zero(
                symthaea_core::hdc::unified_hv::HDC_DIMENSION,
            ));
        }

        // Truncate to shorter length rather than panicking on mismatch
        debug_assert_eq!(
            inputs.len(),
            phi_values.len(),
            "inputs/phi_values length mismatch in PhiAttention::forward"
        );
        let len = inputs.len().min(phi_values.len());
        let inputs = &inputs[..len];
        let phi_values = &phi_values[..len];

        // Update running statistics
        self.update_statistics(phi_values);

        // Compute attention weights
        let weights = compute_attention_weights(inputs, phi_values, &self.config);

        // Compute weighted combination
        let input_refs: Vec<&ContinuousHV> = inputs.iter().collect();
        let output = ContinuousHV::weighted_bundle(&input_refs, &weights);

        // Optionally normalize
        let output = if self.config.normalize_output {
            output.normalize()
        } else {
            output
        };

        let transformed_phi: Vec<f32> = phi_values
            .iter()
            .map(|&p| self.transform_phi(p as f32))
            .collect();
        let entropy = compute_entropy(&weights);

        PhiAttentionResult {
            output,
            weights,
            transformed_phi,
            entropy,
        }
    }

    /// Forward pass with gradient computation for learning
    ///
    /// Returns the result along with intermediate values needed for backpropagation
    pub fn forward_with_grad(
        &mut self,
        inputs: &[ContinuousHV],
        phi_values: &[f64],
    ) -> (PhiAttentionResult, PhiAttentionGradients) {
        let result = self.forward(inputs, phi_values);

        // Store values needed for gradient computation
        let gradients = PhiAttentionGradients {
            raw_phi: phi_values.iter().map(|&p| p as f32).collect(),
            transformed_phi: result.transformed_phi.clone(),
            weights: result.weights.clone(),
            pre_softmax: result
                .transformed_phi
                .iter()
                .map(|&p| p / self.config.temperature)
                .collect(),
        };

        (result, gradients)
    }

    /// Backward pass: compute gradients for scale and bias
    ///
    /// # Arguments
    /// * `grad_output` - Gradient of loss with respect to output
    /// * `inputs` - Original inputs
    /// * `gradients` - Cached values from forward pass
    ///
    /// # Returns
    /// Gradients for scale and bias parameters
    pub fn backward(
        &self,
        grad_output: &ContinuousHV,
        inputs: &[ContinuousHV],
        gradients: &PhiAttentionGradients,
    ) -> (f32, f32) {
        if !self.config.learnable_mapping || inputs.is_empty() {
            return (0.0, 0.0);
        }

        let n = inputs.len();
        let temp = self.config.temperature;

        // Compute dL/d(weights) from dL/d(output)
        // output = sum(weights[i] * inputs[i])
        // dL/d(weights[i]) = grad_output . inputs[i]
        let d_weights: Vec<f32> = inputs
            .iter()
            .map(|input| grad_output.similarity(input))
            .collect();

        // Softmax Jacobian: d(softmax_i)/d(z_j) = softmax_i * (delta_ij - softmax_j)
        // where z = transformed_phi / temperature
        let weights = &gradients.weights;

        // d(z)/d(scale) = phi_raw
        // d(z)/d(bias) = 1
        let mut d_scale = 0.0f32;
        let mut d_bias = 0.0f32;

        for i in 0..n {
            // Compute d(softmax_i)/d(z_j) for all j, then chain with d(z)/d(params)
            for j in 0..n {
                let jacobian = if i == j {
                    weights[i] * (1.0 - weights[i])
                } else {
                    -weights[i] * weights[j]
                };

                // Chain rule: dL/d(scale) += dL/d(w_i) * d(w_i)/d(z_j) * d(z_j)/d(scale)
                // d(z_j)/d(scale) = phi_raw[j] / temperature
                d_scale += d_weights[i] * jacobian * (gradients.raw_phi[j] / temp);

                // d(z_j)/d(bias) = 1 / temperature
                d_bias += d_weights[i] * jacobian * (1.0 / temp);
            }
        }

        (d_scale, d_bias)
    }

    /// Update learnable parameters using computed gradients
    pub fn update_parameters(&mut self, d_scale: f32, d_bias: f32) {
        if self.config.learnable_mapping {
            self.config.scale -= self.config.learning_rate * d_scale;
            self.config.bias -= self.config.learning_rate * d_bias;
        }
    }

    /// Transform phi value using learnable mapping
    #[inline]
    fn transform_phi(&self, phi: f32) -> f32 {
        if self.config.learnable_mapping {
            self.config.scale * phi + self.config.bias
        } else {
            phi
        }
    }

    /// Update running statistics for adaptive behavior
    fn update_statistics(&mut self, phi_values: &[f64]) {
        for &phi in phi_values {
            let phi = phi as f32;
            self.n_observations += 1;
            let n = self.n_observations as f32;

            // Welford's online algorithm for mean and variance
            let delta = phi - self.phi_mean;
            self.phi_mean += delta / n;
            let delta2 = phi - self.phi_mean;
            self.phi_variance += (delta * delta2 - self.phi_variance) / n;
        }
    }

    /// Get running mean of Phi values
    pub fn phi_mean(&self) -> f32 {
        self.phi_mean
    }

    /// Get running variance of Phi values
    pub fn phi_variance(&self) -> f32 {
        self.phi_variance
    }

    /// Get number of observations
    pub fn n_observations(&self) -> u64 {
        self.n_observations
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.phi_mean = 0.5;
        self.phi_variance = 0.1;
        self.n_observations = 0;
    }
}

impl Default for PhiAttentionGate {
    fn default() -> Self {
        Self::default_gate()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULT AND GRADIENT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of Phi attention computation
#[derive(Debug, Clone)]
pub struct PhiAttentionResult {
    /// Output hypervector (weighted combination of inputs)
    pub output: ContinuousHV,

    /// Attention weights for each input
    pub weights: Vec<f32>,

    /// Transformed Phi values (after learnable mapping if enabled)
    pub transformed_phi: Vec<f32>,

    /// Entropy of attention distribution (higher = more uniform)
    pub entropy: f32,
}

impl PhiAttentionResult {
    /// Create empty result
    pub fn empty(zero_hv: ContinuousHV) -> Self {
        Self {
            output: zero_hv,
            weights: vec![],
            transformed_phi: vec![],
            entropy: 0.0,
        }
    }

    /// Get the index with highest attention weight
    pub fn argmax(&self) -> Option<usize> {
        self.weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Get top-k attended indices
    pub fn top_k(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self.weights.iter().copied().enumerate().collect();

        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Check if attention is concentrated (low entropy)
    pub fn is_focused(&self) -> bool {
        // Threshold: entropy less than half of maximum possible
        let max_entropy = (self.weights.len() as f32).ln();
        self.entropy < max_entropy * 0.5
    }
}

/// Gradients for Phi attention parameters
#[derive(Debug, Clone)]
pub struct PhiAttentionGradients {
    /// Raw input Phi values
    pub raw_phi: Vec<f32>,

    /// Transformed Phi values
    pub transformed_phi: Vec<f32>,

    /// Computed attention weights
    pub weights: Vec<f32>,

    /// Pre-softmax values (transformed_phi / temperature)
    pub pre_softmax: Vec<f32>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// STANDALONE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute attention weights from Phi values
///
/// This is the core function implementing Phi-guided attention weighting.
///
/// # Arguments
/// * `inputs` - Input hypervectors (used only for dimension validation)
/// * `phi_values` - Phi values for each input (should be in [0, 1])
///
/// # Returns
/// Attention weights that sum to 1.0
///
/// # Algorithm
/// 1. Transform Phi: `phi' = scale * phi + bias` (if learnable)
/// 2. Apply softmax: `weights = softmax(phi' / temperature)`
/// 3. Apply minimum attention floor if configured
pub fn compute_attention_weights(
    inputs: &[ContinuousHV],
    phi_values: &[f64],
    config: &PhiAttentionConfig,
) -> Vec<f32> {
    if inputs.is_empty() || phi_values.is_empty() {
        return vec![];
    }

    let n = inputs.len().min(phi_values.len());

    // Transform Phi values
    let transformed: Vec<f32> = phi_values
        .iter()
        .take(n)
        .map(|&p| {
            let p = p as f32;
            if config.learnable_mapping {
                config.scale * p + config.bias
            } else {
                p
            }
        })
        .collect();

    // Apply softmax with temperature
    let mut weights = softmax_with_temperature(&transformed, config.temperature);

    // Apply minimum attention floor
    if config.min_attention > 0.0 {
        let floor = config.min_attention;
        let n_f = n as f32;

        // Ensure minimum weight while maintaining sum = 1
        for w in weights.iter_mut() {
            *w = *w * (1.0 - floor * n_f) + floor;
        }

        // Renormalize to handle floating point errors
        let sum: f32 = weights.iter().sum();
        if sum > config.epsilon {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
    }

    weights
}

/// Compute entropy of a probability distribution
fn compute_entropy(weights: &[f32]) -> f32 {
    -weights
        .iter()
        .filter(|&&w| w > 1e-10)
        .map(|&w| w * w.ln())
        .sum::<f32>()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_attention_weights_basic() {
        let inputs = vec![
            ContinuousHV::random(256, 42),
            ContinuousHV::random(256, 43),
            ContinuousHV::random(256, 44),
        ];
        let phi_values = vec![0.8, 0.3, 0.5];
        let config = PhiAttentionConfig::default();

        let weights = compute_attention_weights(&inputs, &phi_values, &config);

        // Weights should sum to 1
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Weights should sum to 1, got {}",
            sum
        );

        // Higher Phi should get higher weight
        assert!(
            weights[0] > weights[1],
            "Higher Phi should get higher weight"
        );
        assert!(
            weights[0] > weights[2],
            "Highest Phi should get highest weight"
        );
    }

    #[test]
    fn test_high_phi_dominates_output() {
        let mut gate = PhiAttentionGate::new(PhiAttentionConfig::sharp());

        let high_phi_input = ContinuousHV::random(256, 100);
        let low_phi_input = ContinuousHV::random(256, 200);

        let inputs = vec![high_phi_input.clone(), low_phi_input.clone()];
        let phi_values = vec![0.9, 0.1]; // First has much higher Phi

        let result = gate.forward(&inputs, &phi_values);

        // Output should be more similar to high-Phi input
        let sim_high = result.output.similarity(&high_phi_input);
        let sim_low = result.output.similarity(&low_phi_input);

        assert!(
            sim_high > sim_low,
            "Output should be more similar to high-Phi input: sim_high={}, sim_low={}",
            sim_high,
            sim_low
        );
    }

    #[test]
    fn test_softmax_temperature_effects() {
        let values = vec![0.3, 0.5, 0.8];

        // Low temperature (sharp)
        let sharp = softmax_with_temperature(&values, 0.1);
        // High temperature (soft)
        let soft = softmax_with_temperature(&values, 10.0);

        // Sharp distribution should be more peaked
        let sharp_max = sharp.iter().cloned().fold(0.0, f32::max);
        let soft_max = soft.iter().cloned().fold(0.0, f32::max);

        assert!(
            sharp_max > soft_max,
            "Lower temperature should give sharper distribution"
        );
    }

    #[test]
    fn test_learnable_mapping() {
        let config = PhiAttentionConfig {
            learnable_mapping: true,
            scale: 2.0,
            bias: 0.5,
            ..Default::default()
        };
        let mut gate = PhiAttentionGate::new(config);

        let inputs = vec![ContinuousHV::random(256, 42), ContinuousHV::random(256, 43)];
        let phi_values = vec![0.3, 0.5];

        let result = gate.forward(&inputs, &phi_values);

        // Transformed phi = scale * phi + bias = 2.0 * phi + 0.5
        let expected_transformed: Vec<f32> =
            phi_values.iter().map(|&p| 2.0 * p as f32 + 0.5).collect();

        for (actual, expected) in result
            .transformed_phi
            .iter()
            .zip(expected_transformed.iter())
        {
            assert!((actual - expected).abs() < 1e-5, "Transformed phi mismatch");
        }
    }

    #[test]
    fn test_gradient_flow() {
        let config = PhiAttentionConfig::learnable();
        let mut gate = PhiAttentionGate::new(config);

        let inputs = vec![ContinuousHV::random(256, 42), ContinuousHV::random(256, 43)];
        let phi_values = vec![0.6, 0.4];

        // Forward pass with gradient tracking
        let (result, gradients) = gate.forward_with_grad(&inputs, &phi_values);

        // Create a mock gradient (pretend output should match first input)
        let target = &inputs[0];
        let grad_output_values: Vec<f32> = result
            .output
            .values
            .iter()
            .zip(target.values.iter())
            .map(|(o, t)| o - t)
            .collect();
        let grad_output = ContinuousHV::from_vec(grad_output_values);

        // Compute gradients
        let (d_scale, d_bias) = gate.backward(&grad_output, &inputs, &gradients);

        // Gradients should be finite
        assert!(d_scale.is_finite(), "d_scale should be finite");
        assert!(d_bias.is_finite(), "d_bias should be finite");
    }

    #[test]
    fn test_min_attention_floor() {
        let config = PhiAttentionConfig {
            min_attention: 0.1,
            ..Default::default()
        };

        let inputs = vec![ContinuousHV::random(256, 42), ContinuousHV::random(256, 43)];
        let phi_values = vec![0.99, 0.01]; // Extreme difference

        let weights = compute_attention_weights(&inputs, &phi_values, &config);

        // Even the low-Phi input should have at least min_attention
        assert!(
            weights[1] >= 0.099,
            "Minimum attention should be enforced: got {}",
            weights[1]
        );
    }

    #[test]
    fn test_entropy_calculation() {
        // Uniform distribution should have high entropy
        let uniform_weights = vec![0.25, 0.25, 0.25, 0.25];
        let uniform_entropy = compute_entropy(&uniform_weights);

        // Peaked distribution should have low entropy
        let peaked_weights = vec![0.97, 0.01, 0.01, 0.01];
        let peaked_entropy = compute_entropy(&peaked_weights);

        assert!(
            uniform_entropy > peaked_entropy,
            "Uniform distribution should have higher entropy"
        );
    }

    #[test]
    fn test_empty_inputs() {
        let mut gate = PhiAttentionGate::default_gate();

        let result = gate.forward(&[], &[]);
        assert!(result.weights.is_empty());
    }

    #[test]
    fn test_statistics_tracking() {
        let mut gate = PhiAttentionGate::default_gate();

        let inputs = vec![ContinuousHV::random(256, 42)];

        // Feed multiple observations
        for i in 0..100 {
            let phi = 0.4 + 0.2 * ((i as f64) / 100.0);
            gate.forward(&inputs, &[phi]);
        }

        // Mean should be around 0.5 (center of [0.4, 0.6])
        assert!(
            (gate.phi_mean() - 0.5).abs() < 0.1,
            "Running mean should track input distribution"
        );
    }

    #[test]
    fn test_attention_result_helpers() {
        let result = PhiAttentionResult {
            output: ContinuousHV::random(256, 42),
            weights: vec![0.1, 0.6, 0.3],
            transformed_phi: vec![0.1, 0.6, 0.3],
            entropy: 0.5,
        };

        // Argmax should be index 1
        assert_eq!(result.argmax(), Some(1));

        // Top-2 should be indices 1 and 2
        let top2 = result.top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 1);
        assert_eq!(top2[1].0, 2);
    }

    #[test]
    fn test_parameter_update() {
        let config = PhiAttentionConfig {
            learnable_mapping: true,
            learning_rate: 0.1,
            scale: 1.0,
            bias: 0.0,
            ..Default::default()
        };
        let mut gate = PhiAttentionGate::new(config);

        let initial_scale = gate.config.scale;
        let initial_bias = gate.config.bias;

        // Update with some gradients
        gate.update_parameters(0.5, -0.3);

        // Parameters should have changed
        assert!(
            (gate.config.scale - initial_scale).abs() > 0.01,
            "Scale should be updated"
        );
        assert!(
            (gate.config.bias - initial_bias).abs() > 0.01,
            "Bias should be updated"
        );
    }
}
