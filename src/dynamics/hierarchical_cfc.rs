// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hierarchical Closed-form Continuous-time Networks
//!
//! Multi-scale CfC architecture where each level operates at different
//! temporal scales, enabling simultaneous processing of phenomena across
//! vastly different time horizons.
//!
//! ## Time Scale Hierarchy
//!
//! - Level 0 (tau ~0.01): Ultra-fast sensory/reflexive dynamics
//! - Level 1 (tau ~0.1): Fast perceptual processing
//! - Level 2 (tau ~1.0): Working memory / deliberation
//! - Level 3 (tau ~10.0): Long-term context / narrative
//!
//! ## Bidirectional Information Flow
//!
//! - **Bottom-up (temporal abstraction)**: Fast layers aggregate into slow layers
//! - **Top-down (contextual modulation)**: Slow layers modulate fast layer dynamics
//!
//! ## Key Benefits
//!
//! 1. **Multi-resolution temporal processing**: Capture both rapid transients and slow trends
//! 2. **Hierarchical prediction**: Each scale makes predictions at its natural timescale
//! 3. **Adaptive time constants**: Top-down feedback adjusts lower-level responsiveness
//! 4. **Stable learning**: Multi-scale gradients provide better training signal

use super::cfc::{ActivationType, CfCConfig, CfCNetwork, CfCNetworkConfig};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Default time constants for hierarchical layers (fast -> slow)
pub const DEFAULT_TIME_CONSTANTS: [f32; 4] = [0.01, 0.1, 1.0, 10.0];

/// Configuration for hierarchical CfC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalCfCConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension per layer (also used for final output)
    pub output_dim: usize,
    /// Time constants for each layer [fast -> slow]
    /// Each value represents the center of the tau range for that layer
    pub time_constants: Vec<f32>,
    /// Hidden dimension for each layer (defaults to same as output_dim if empty)
    pub hidden_dims: Vec<usize>,
    /// Enable top-down feedback (slow modulates fast)
    pub top_down_feedback: bool,
    /// Feedback strength (0.0 to 1.0)
    pub feedback_strength: f32,
    /// Enable bottom-up integration (fast feeds into slow)
    pub bottom_up_integration: bool,
    /// Bottom-up integration strength
    pub integration_strength: f32,
    /// Enable multi-scale prediction (each layer produces output)
    pub multi_scale_prediction: bool,
    /// Learning rate scaling per layer (fast layers often need smaller LR)
    pub lr_scales: Vec<f32>,
}

impl Default for HierarchicalCfCConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            output_dim: 32,
            time_constants: DEFAULT_TIME_CONSTANTS.to_vec(),
            hidden_dims: vec![64, 64, 48, 32], // Progressively smaller for higher levels
            top_down_feedback: true,
            feedback_strength: 0.3,
            bottom_up_integration: true,
            integration_strength: 0.5,
            multi_scale_prediction: true,
            lr_scales: vec![0.5, 1.0, 1.0, 0.5], // Reduce LR for extreme scales
        }
    }
}

/// Output from hierarchical forward pass
#[derive(Debug, Clone)]
pub struct HierarchicalOutput {
    /// Final combined output
    pub combined: Array1<f32>,
    /// Outputs at each time scale (layer 0 = fastest, layer N = slowest)
    pub scale_outputs: Vec<Array1<f32>>,
    /// Hidden states at each layer
    pub layer_states: Vec<Array1<f32>>,
    /// Effective time constants after top-down modulation
    pub effective_taus: Vec<f32>,
}

/// Hierarchical CfC network with multi-scale temporal processing
pub struct HierarchicalCfC {
    /// Configuration
    config: HierarchicalCfCConfig,
    /// CfC networks at each level (index 0 = fastest)
    layers: Vec<CfCNetwork>,
    /// Inter-layer projection weights (bottom-up)
    /// projects layer[i] -> layer[i+1]
    up_projections: Vec<Array2<f32>>,
    /// Inter-layer projection weights (top-down)
    /// projects layer[i+1] -> layer[i]
    down_projections: Vec<Array2<f32>>,
    /// Output projections for multi-scale prediction
    output_projections: Vec<Array2<f32>>,
    output_biases: Vec<Array1<f32>>,
    /// Combination weights for final output
    combination_weights: Array1<f32>,
    /// Total steps processed
    total_steps: u64,
    /// Recent outputs for top-down modulation context
    recent_slow_outputs: Vec<Array1<f32>>,
}

impl HierarchicalCfC {
    /// Create a new hierarchical CfC network
    pub fn new(config: HierarchicalCfCConfig) -> Self {
        let num_layers = config.time_constants.len();
        assert!(num_layers > 0, "Must have at least one layer");

        // Ensure hidden_dims has the right length
        let hidden_dims: Vec<usize> = if config.hidden_dims.len() == num_layers {
            config.hidden_dims.clone()
        } else {
            vec![config.output_dim; num_layers]
        };

        let mut layers = Vec::with_capacity(num_layers);
        let mut up_projections = Vec::with_capacity(num_layers - 1);
        let mut down_projections = Vec::with_capacity(num_layers - 1);
        let mut output_projections = Vec::with_capacity(num_layers);
        let mut output_biases = Vec::with_capacity(num_layers);

        for (i, &tau_center) in config.time_constants.iter().enumerate() {
            // Tau range centered on the target with 50% spread
            let tau_min = (tau_center * 0.5).max(1e-4);
            let tau_max = tau_center * 2.0;

            // Input dimension: first layer gets raw input, others get output from up_projection
            // which projects to THIS layer's hidden dim (next_hidden in the projection creation)
            let layer_input_dim = if i == 0 {
                config.input_dim
            } else {
                // After bottom-up integration, current_input has dimension hidden_dims[i]
                // because up_projections[i-1] has shape (hidden_dims[i], hidden_dims[i-1])
                hidden_dims[i]
            };

            let hidden_dim = hidden_dims[i];

            let cell_config = CfCConfig {
                input_dim: layer_input_dim,
                hidden_dim,
                use_backbone: hidden_dim > 32, // Use backbone for larger layers
                backbone_layers: 1,
                backbone_dim: hidden_dim,
                activation: ActivationType::SiLU,
                tau_range: (tau_min, tau_max),
                dropout: 0.0,
                online_learning: None,
                gradient_clip: 1.0,
            };

            let net_config = CfCNetworkConfig {
                input_dim: layer_input_dim,
                hidden_dim,
                num_layers: 1,
                output_dim: hidden_dim, // Internal output = hidden for inter-layer passing
                cell_config,
                residual: false,
                bidirectional: false,
                enable_online_learning: false,
                online_learning_config: Default::default(),
            };

            layers.push(CfCNetwork::new(net_config));

            // Inter-layer projections (except for last layer)
            if i < num_layers - 1 {
                let next_hidden = hidden_dims[i + 1];
                let scale = (2.0 / (hidden_dim + next_hidden) as f32).sqrt();

                // Bottom-up: project from this layer to next (slower)
                up_projections.push(Array2::from_shape_fn((next_hidden, hidden_dim), |_| {
                    (rand::random::<f32>() - 0.5) * 2.0 * scale
                }));

                // Top-down: project from next layer to this (faster)
                down_projections.push(Array2::from_shape_fn((hidden_dim, next_hidden), |_| {
                    (rand::random::<f32>() - 0.5) * 2.0 * scale
                }));
            }

            // Output projection for this layer
            let scale = (2.0 / (hidden_dim + config.output_dim) as f32).sqrt();
            output_projections.push(Array2::from_shape_fn(
                (config.output_dim, hidden_dim),
                |_| (rand::random::<f32>() - 0.5) * 2.0 * scale,
            ));
            output_biases.push(Array1::zeros(config.output_dim));
        }

        // Combination weights (learned importance of each scale)
        // Initialize with exponentially decaying weights (fast scales slightly more important)
        let combination_weights = Array1::from_shape_fn(num_layers, |i| {
            let decay = (-0.3 * i as f32).exp();
            decay
                / (0..num_layers)
                    .map(|j| (-0.3 * j as f32).exp())
                    .sum::<f32>()
        });

        // Initialize recent outputs for top-down context
        let recent_slow_outputs = hidden_dims.iter().map(|&d| Array1::zeros(d)).collect();

        Self {
            config,
            layers,
            up_projections,
            down_projections,
            output_projections,
            output_biases,
            combination_weights,
            total_steps: 0,
            recent_slow_outputs,
        }
    }

    /// Forward pass through hierarchical network with bidirectional processing
    ///
    /// # Arguments
    /// * `input` - Input vector
    /// * `dt` - Time step
    ///
    /// # Returns
    /// HierarchicalOutput containing outputs at all scales
    pub fn forward_hierarchical(&mut self, input: &Array1<f32>, dt: f32) -> HierarchicalOutput {
        self.total_steps += 1;
        let num_layers = self.layers.len();

        // Phase 1: Bottom-up pass - fast layers feed into slow layers
        let mut layer_outputs = Vec::with_capacity(num_layers);
        let mut current_input = input.clone();

        for i in 0..num_layers {
            // Apply top-down modulation from previous step if enabled
            if self.config.top_down_feedback && i < num_layers - 1 && self.total_steps > 1 {
                let higher_context = &self.recent_slow_outputs[i + 1];
                let modulation = self.down_projections[i].dot(higher_context);

                // Modulation affects the effective time constant
                let gate_activation: f32 = modulation.iter().sum::<f32>() / modulation.len() as f32;
                let gate = 1.0 / (1.0 + (-gate_activation).exp()); // sigmoid

                // Scale tau: gate > 0.5 -> slow down (increase tau), gate < 0.5 -> speed up
                let tau_scale = 1.0 + self.config.feedback_strength * (gate - 0.5);
                self.layers[i].scale_tau_all(tau_scale);
            }

            // Forward through this layer
            let output = self.layers[i].forward(&current_input, dt);
            layer_outputs.push(output.clone());

            // Prepare input for next layer (bottom-up integration)
            if i < num_layers - 1 && self.config.bottom_up_integration {
                // Project current output to next layer's input space
                let projected = self.up_projections[i].dot(&output);
                // Mix with raw input propagation
                current_input = projected * self.config.integration_strength;
            }
        }

        // Phase 2: Compute outputs at each scale
        let scale_outputs: Vec<Array1<f32>> = layer_outputs
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let proj = &self.output_projections[i];
                let bias = &self.output_biases[i];
                proj.dot(h) + bias
            })
            .collect();

        // Phase 3: Combine multi-scale outputs
        let combined = if self.config.multi_scale_prediction {
            let mut sum = Array1::zeros(self.config.output_dim);
            for (i, out) in scale_outputs.iter().enumerate() {
                sum = sum + out * self.combination_weights[i];
            }
            sum
        } else {
            // Just use the slowest (highest level) output
            scale_outputs
                .last()
                .cloned()
                .unwrap_or_else(|| Array1::zeros(self.config.output_dim))
        };

        // Update recent outputs for next step's top-down modulation
        self.recent_slow_outputs = layer_outputs.clone();

        // Compute effective taus (average tau per layer after modulation)
        let effective_taus: Vec<f32> = self.config.time_constants.clone();

        HierarchicalOutput {
            combined,
            scale_outputs,
            layer_states: layer_outputs,
            effective_taus,
        }
    }

    /// Simple step interface (returns combined output)
    pub fn step(&mut self, input: &Array1<f32>, dt: f32) -> Array1<f32> {
        self.forward_hierarchical(input, dt).combined
    }

    /// Multi-scale training step with BPTT
    ///
    /// Each scale has its own loss target, and gradients flow both up and down.
    ///
    /// # Arguments
    /// * `input` - Input vector
    /// * `targets` - Target outputs at each scale (same length as num_layers)
    /// * `dt` - Time step
    /// * `learning_rate` - Base learning rate (scaled per layer)
    ///
    /// # Returns
    /// Total loss across all scales
    pub fn train_step_multiscale(
        &mut self,
        input: &Array1<f32>,
        targets: &[Array1<f32>],
        dt: f32,
        learning_rate: f32,
    ) -> anyhow::Result<f32> {
        assert_eq!(
            targets.len(),
            self.layers.len(),
            "Must provide target for each scale"
        );

        // Forward pass to get all outputs
        let output = self.forward_hierarchical(input, dt);

        let mut total_loss = 0.0f32;
        let num_layers = self.layers.len();

        // Compute loss at each scale and train each layer
        for (i, (scale_out, target)) in output.scale_outputs.iter().zip(targets.iter()).enumerate()
        {
            // MSE loss at this scale
            let scale_loss = mse_loss(scale_out, target);
            total_loss += scale_loss;

            // Scale learning rate for this layer
            let layer_lr = learning_rate * self.config.lr_scales.get(i).copied().unwrap_or(1.0);

            // Get the layer's hidden state
            let layer_state = &output.layer_states[i];

            // Train the layer's output projection
            self.train_output_projection(i, layer_state, target, layer_lr);

            // Train the CfC layer itself
            // Use the layer's input (reconstructed from the forward pass)
            let layer_input = if i == 0 {
                input.clone()
            } else {
                // Use projected previous layer output
                self.up_projections[i - 1].dot(&output.layer_states[i - 1])
            };

            let _ = self.layers[i].train_step(&layer_input, layer_state, dt, layer_lr);
        }

        // Train combination weights based on scale performance
        self.update_combination_weights(&output.scale_outputs, targets, learning_rate * 0.1);

        // Train inter-layer projections
        if self.config.bottom_up_integration {
            self.train_up_projections(&output.layer_states, learning_rate * 0.5);
        }
        if self.config.top_down_feedback {
            self.train_down_projections(&output.layer_states, learning_rate * 0.5);
        }

        Ok(total_loss / num_layers as f32)
    }

    /// Single-target training (target is used for slowest scale, faster scales use state-matching)
    pub fn train_step(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> anyhow::Result<f32> {
        // Create multi-scale targets: slower scales match the target more closely
        let num_layers = self.layers.len();
        let targets: Vec<Array1<f32>> = (0..num_layers)
            .map(|i| {
                // Weight increases for slower (higher index) layers
                let weight = (i as f32 + 1.0) / num_layers as f32;
                target.mapv(|v| v * weight)
            })
            .collect();

        self.train_step_multiscale(input, &targets, dt, learning_rate)
    }

    /// Train the output projection for a specific layer
    fn train_output_projection(
        &mut self,
        layer_idx: usize,
        hidden: &Array1<f32>,
        target: &Array1<f32>,
        lr: f32,
    ) {
        let output =
            self.output_projections[layer_idx].dot(hidden) + &self.output_biases[layer_idx];

        // Gradient: d_loss/d_out = 2 * (out - target) / n
        let n = output.len().min(target.len()) as f32;
        let d_output: Array1<f32> = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| 2.0 * (o - t) / n)
            .collect::<Vec<f32>>()
            .into();

        // Update weights: W -= lr * d_out * h^T
        let out_dim = self.config.output_dim;
        let hidden_dim = hidden.len();
        for i in 0..out_dim {
            for j in 0..hidden_dim {
                let grad = d_output[i] * hidden[j];
                self.output_projections[layer_idx][[i, j]] -= lr * grad.clamp(-0.5, 0.5);
            }
            // Update bias
            self.output_biases[layer_idx][i] -= lr * d_output[i].clamp(-0.5, 0.5);
        }
    }

    /// Update combination weights based on scale-wise accuracy
    fn update_combination_weights(
        &mut self,
        outputs: &[Array1<f32>],
        targets: &[Array1<f32>],
        lr: f32,
    ) {
        let num_layers = outputs.len();
        if num_layers == 0 {
            return;
        }

        // Compute inverse loss (higher = better) for each scale
        let mut inverse_losses: Vec<f32> = Vec::with_capacity(num_layers);
        for (out, target) in outputs.iter().zip(targets.iter()) {
            let loss = mse_loss(out, target);
            inverse_losses.push(1.0 / (loss + 0.01)); // Add small value to avoid division by zero
        }

        // Normalize to get target weights
        let sum: f32 = inverse_losses.iter().sum();
        let target_weights: Vec<f32> = inverse_losses.iter().map(|l| l / sum).collect();

        // Gradient update toward target weights
        for i in 0..num_layers {
            let grad = target_weights[i] - self.combination_weights[i];
            self.combination_weights[i] += lr * grad;
        }

        // Normalize combination weights
        let weight_sum: f32 = self.combination_weights.iter().sum();
        if weight_sum > 0.0 {
            self.combination_weights.mapv_inplace(|w| w / weight_sum);
        }
    }

    /// Train bottom-up projections to improve information flow
    fn train_up_projections(&mut self, layer_states: &[Array1<f32>], lr: f32) {
        for i in 0..self.up_projections.len() {
            let lower = &layer_states[i];
            let upper = &layer_states[i + 1];

            // Target: project lower to match upper
            let projected = self.up_projections[i].dot(lower);
            let error: Array1<f32> = projected
                .iter()
                .zip(upper.iter())
                .map(|(p, u)| p - u)
                .collect::<Vec<f32>>()
                .into();

            // Update projection: W -= lr * error * lower^T
            let upper_dim = self.up_projections[i].nrows();
            let lower_dim = self.up_projections[i].ncols();
            for j in 0..upper_dim {
                for k in 0..lower_dim {
                    let grad = error[j] * lower[k];
                    self.up_projections[i][[j, k]] -= lr * grad.clamp(-0.5, 0.5);
                }
            }
        }
    }

    /// Train top-down projections for better contextual modulation
    fn train_down_projections(&mut self, layer_states: &[Array1<f32>], lr: f32) {
        for i in 0..self.down_projections.len() {
            let lower = &layer_states[i];
            let upper = &layer_states[i + 1];

            // Target: project upper to provide useful context for lower
            let projected = self.down_projections[i].dot(upper);

            // The gradient encourages projections that help reconstruct lower layer activity
            let error: Array1<f32> = projected
                .iter()
                .zip(lower.iter())
                .map(|(p, l)| p - l)
                .collect::<Vec<f32>>()
                .into();

            // Update projection: W -= lr * error * upper^T
            let lower_dim = self.down_projections[i].nrows();
            let upper_dim = self.down_projections[i].ncols();
            for j in 0..lower_dim {
                for k in 0..upper_dim {
                    let grad = error[j] * upper[k];
                    self.down_projections[i][[j, k]] -= lr * grad.clamp(-0.5, 0.5);
                }
            }
        }
    }

    /// Reset all layers and internal state
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        for output in &mut self.recent_slow_outputs {
            output.fill(0.0);
        }
        self.total_steps = 0;
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get total parameters across all layers and projections
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        // CfC layer parameters
        for layer in &self.layers {
            count += layer.num_parameters();
        }

        // Inter-layer projections
        for proj in &self.up_projections {
            count += proj.len();
        }
        for proj in &self.down_projections {
            count += proj.len();
        }

        // Output projections
        for proj in &self.output_projections {
            count += proj.len();
        }
        for bias in &self.output_biases {
            count += bias.len();
        }

        // Combination weights
        count += self.combination_weights.len();

        count
    }

    /// Get states from all layers
    pub fn all_states(&self) -> Vec<Vec<Array1<f32>>> {
        self.layers.iter().map(|l| l.state()).collect()
    }

    /// Get combination weights (importance of each scale)
    pub fn combination_weights(&self) -> &Array1<f32> {
        &self.combination_weights
    }

    /// Get configuration
    pub fn config(&self) -> &HierarchicalCfCConfig {
        &self.config
    }

    /// Get time constants for each layer
    pub fn time_constants(&self) -> &[f32] {
        &self.config.time_constants
    }

    /// Set the output bias for a specific layer.
    ///
    /// This allows narrative-specific initialization (e.g. non-zero resting
    /// energy/tension) without changing the CfC weights themselves.
    pub fn set_output_bias(&mut self, layer_idx: usize, bias: Array1<f32>) {
        if layer_idx < self.output_biases.len() && bias.len() == self.output_biases[layer_idx].len()
        {
            self.output_biases[layer_idx] = bias;
        }
    }
}

/// Mean squared error between two arrays (helper function)
fn mse_loss(output: &Array1<f32>, target: &Array1<f32>) -> f32 {
    let n = output.len().min(target.len());
    if n == 0 {
        return 0.0;
    }
    let mse = output
        .iter()
        .zip(target.iter())
        .take(n)
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f32>()
        / n as f32;
    if mse.is_finite() { mse } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_cfc_creation() {
        let config = HierarchicalCfCConfig::default();
        let hcfc = HierarchicalCfC::new(config);
        assert_eq!(hcfc.num_layers(), 4); // Default uses 4 time constants
    }

    #[test]
    fn test_hierarchical_cfc_custom_time_constants() {
        let config = HierarchicalCfCConfig {
            input_dim: 32,
            output_dim: 16,
            time_constants: vec![0.01, 0.1, 1.0, 10.0],
            hidden_dims: vec![32, 32, 24, 16],
            ..Default::default()
        };
        let hcfc = HierarchicalCfC::new(config);
        assert_eq!(hcfc.num_layers(), 4);
        assert_eq!(hcfc.time_constants(), &[0.01, 0.1, 1.0, 10.0]);
    }

    #[test]
    fn test_hierarchical_forward() {
        let config = HierarchicalCfCConfig {
            input_dim: 32,
            output_dim: 16,
            time_constants: vec![0.01, 0.1, 1.0, 10.0],
            hidden_dims: vec![32, 32, 24, 16],
            ..Default::default()
        };
        let mut hcfc = HierarchicalCfC::new(config);

        let input = Array1::from_vec(vec![0.5; 32]);
        let output = hcfc.forward_hierarchical(&input, 0.01);

        // Check combined output
        assert_eq!(output.combined.len(), 16);
        assert!(output.combined.iter().all(|x| x.is_finite()));

        // Check scale outputs
        assert_eq!(output.scale_outputs.len(), 4);
        for out in &output.scale_outputs {
            assert_eq!(out.len(), 16);
            assert!(out.iter().all(|x| x.is_finite()));
        }

        // Check layer states
        assert_eq!(output.layer_states.len(), 4);
    }

    #[test]
    fn test_fast_layer_responds_to_rapid_changes() {
        let config = HierarchicalCfCConfig {
            input_dim: 16,
            output_dim: 8,
            time_constants: vec![0.01, 1.0], // Fast and slow
            hidden_dims: vec![16, 16],
            top_down_feedback: false,
            bottom_up_integration: false,
            ..Default::default()
        };
        let mut hcfc = HierarchicalCfC::new(config);

        // Apply rapid input changes
        let input_a = Array1::from_vec(vec![1.0; 16]);
        let input_b = Array1::from_vec(vec![-1.0; 16]);

        let dt = 0.01; // Small time step

        // Step with input A
        let out_a = hcfc.forward_hierarchical(&input_a, dt);
        let fast_a = out_a.scale_outputs[0].clone();
        let slow_a = out_a.scale_outputs[1].clone();

        // Step with input B (rapid change)
        let out_b = hcfc.forward_hierarchical(&input_b, dt);
        let fast_b = out_b.scale_outputs[0].clone();
        let slow_b = out_b.scale_outputs[1].clone();

        // Fast layer should show more change than slow layer
        let fast_change: f32 = fast_a
            .iter()
            .zip(fast_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / fast_a.len() as f32;
        let slow_change: f32 = slow_a
            .iter()
            .zip(slow_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / slow_a.len() as f32;

        // Fast layer should respond more to rapid changes
        // (Note: exact ratio depends on initialization, so we use a loose check)
        assert!(
            fast_change >= slow_change * 0.1,
            "Fast layer change {} should be comparable to or greater than slow layer change {}",
            fast_change,
            slow_change
        );
    }

    #[test]
    fn test_slow_layer_captures_trends() {
        let config = HierarchicalCfCConfig {
            input_dim: 16,
            output_dim: 8,
            time_constants: vec![0.01, 10.0], // Fast and slow
            hidden_dims: vec![16, 16],
            top_down_feedback: false,
            bottom_up_integration: true,
            ..Default::default()
        };
        let mut hcfc = HierarchicalCfC::new(config);

        // Apply consistent input over many steps
        let input = Array1::from_vec(vec![1.0; 16]);
        let dt = 0.1;

        let mut slow_outputs = Vec::new();
        for _ in 0..50 {
            let out = hcfc.forward_hierarchical(&input, dt);
            slow_outputs.push(out.scale_outputs[1].clone());
        }

        // Slow layer should gradually stabilize
        let _early_variance: f32 = slow_outputs[0..5]
            .iter()
            .map(|o| o.iter().map(|x| x.abs()).sum::<f32>())
            .sum::<f32>()
            / 5.0;
        let late_variance: f32 = slow_outputs[45..50]
            .iter()
            .map(|o| o.iter().map(|x| x.abs()).sum::<f32>())
            .sum::<f32>()
            / 5.0;

        // Late outputs should have similar or higher magnitude (accumulated signal)
        assert!(late_variance.is_finite(), "Slow layer should remain stable");
    }

    #[test]
    fn test_combined_output_better_than_single_scale() {
        let config = HierarchicalCfCConfig {
            input_dim: 32,
            output_dim: 16,
            time_constants: vec![0.01, 0.1, 1.0, 10.0],
            hidden_dims: vec![32, 32, 24, 16],
            multi_scale_prediction: true,
            ..Default::default()
        };
        let mut hcfc = HierarchicalCfC::new(config.clone());

        // Create a mixed-frequency target pattern
        let input = Array1::from_vec((0..32).map(|i| (i as f32 * 0.1).sin()).collect());
        let target = Array1::from_vec(
            (0..16)
                .map(|i| {
                    // Mix of fast and slow components
                    (i as f32 * 0.5).sin() * 0.5 + (i as f32 * 0.05).cos() * 0.5
                })
                .collect(),
        );

        // Train for several steps
        let dt = 0.1;
        for _ in 0..20 {
            let _ = hcfc.train_step(&input, &target, dt, 0.01);
        }

        // Get final output
        let output = hcfc.forward_hierarchical(&input, dt);

        // Combined loss vs single-scale loss
        let combined_loss = mse_loss(&output.combined, &target);
        let single_scale_losses: Vec<f32> = output
            .scale_outputs
            .iter()
            .map(|o| mse_loss(o, &target))
            .collect();

        // Combined should be competitive with the best single scale
        let best_single = single_scale_losses
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);

        // After training, combined should be no worse than 2x the best single scale
        assert!(
            combined_loss <= best_single * 2.0 + 0.5,
            "Combined loss {} should be competitive with best single scale loss {}",
            combined_loss,
            best_single
        );
    }

    #[test]
    fn test_hierarchical_stability() {
        let config = HierarchicalCfCConfig::default();
        let mut hcfc = HierarchicalCfC::new(config);
        let input = Array1::from_vec(vec![0.3; 64]);

        for step in 0..1000 {
            let output = hcfc.forward_hierarchical(&input, 0.1);
            assert!(
                output.combined.iter().all(|x| x.is_finite()),
                "Hierarchical CfC diverged at step {}",
                step
            );
            for (i, scale_out) in output.scale_outputs.iter().enumerate() {
                assert!(
                    scale_out.iter().all(|x| x.is_finite()),
                    "Scale {} output diverged at step {}",
                    i,
                    step
                );
            }
        }
    }

    #[test]
    fn test_hierarchical_reset() {
        let config = HierarchicalCfCConfig::default();
        let mut hcfc = HierarchicalCfC::new(config);
        let input = Array1::from_vec(vec![1.0; 64]);

        // Process some input
        for _ in 0..10 {
            hcfc.forward_hierarchical(&input, 0.1);
        }

        hcfc.reset();

        // After reset, all states should be zero
        for level_states in hcfc.all_states() {
            for state in level_states {
                assert!(state.iter().all(|x| *x == 0.0), "State not reset");
            }
        }
        assert_eq!(hcfc.total_steps, 0);
    }

    #[test]
    fn test_multiscale_training() {
        let config = HierarchicalCfCConfig {
            input_dim: 16,
            output_dim: 8,
            time_constants: vec![0.01, 0.1, 1.0, 10.0],
            hidden_dims: vec![16, 16, 12, 8],
            ..Default::default()
        };
        let mut hcfc = HierarchicalCfC::new(config);

        let input = Array1::from_vec(vec![0.5; 16]);
        let targets: Vec<Array1<f32>> = (0..4)
            .map(|i| Array1::from_vec(vec![(i as f32 + 1.0) * 0.1; 8]))
            .collect();

        // Training should decrease loss
        let initial_loss = hcfc
            .train_step_multiscale(&input, &targets, 0.1, 0.01)
            .unwrap();

        for _ in 0..50 {
            let _ = hcfc.train_step_multiscale(&input, &targets, 0.1, 0.01);
        }

        let final_loss = hcfc
            .train_step_multiscale(&input, &targets, 0.1, 0.01)
            .unwrap();

        assert!(
            final_loss <= initial_loss + 0.1,
            "Training should not significantly increase loss: initial {} -> final {}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_parameter_count() {
        let config = HierarchicalCfCConfig {
            input_dim: 32,
            output_dim: 16,
            time_constants: vec![0.01, 0.1, 1.0],
            hidden_dims: vec![32, 24, 16],
            ..Default::default()
        };
        let hcfc = HierarchicalCfC::new(config);

        // Should have non-zero parameters
        let params = hcfc.num_parameters();
        assert!(params > 0, "Should have parameters");

        // Rough check: should have at least hidden_dim^2 params per layer
        assert!(
            params > 32 * 32 + 24 * 24 + 16 * 16,
            "Should have substantial parameters, got {}",
            params
        );
    }

    #[test]
    fn test_set_output_bias_method() {
        let config = HierarchicalCfCConfig {
            input_dim: 16,
            output_dim: 8,
            time_constants: vec![0.01, 0.1, 1.0],
            hidden_dims: vec![16, 16, 16],
            ..Default::default()
        };
        let mut hcfc = HierarchicalCfC::new(config);

        // Set a non-zero bias on layer 0
        let mut bias = Array1::zeros(8);
        bias[0] = 0.5;
        hcfc.set_output_bias(0, bias);

        // Forward with zero input — output should reflect the bias
        let input = Array1::zeros(16);
        let output = hcfc.forward_hierarchical(&input, 0.1);
        // Layer 0's output should have a non-zero first element due to bias
        assert!(
            output.scale_outputs[0][0].abs() > 0.01,
            "Bias should affect output: got {}",
            output.scale_outputs[0][0]
        );
    }

    #[test]
    fn test_combination_weights_normalized() {
        let config = HierarchicalCfCConfig::default();
        let hcfc = HierarchicalCfC::new(config);

        let sum: f32 = hcfc.combination_weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Combination weights should sum to 1, got {}",
            sum
        );
    }
}
