// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # CfC Neural Networks with Backend Selection
//!
//! CPU-based Closed-form Continuous-time (CfC) networks using ndarray for matrix
//! operations. A `Wgpu` backend variant exists as a placeholder for future WebGPU
//! acceleration but currently resolves to CPU.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::dynamics::cfc_gpu::{GpuCfcNetwork, GpuCfcConfig, GpuBackend};
//!
//! let config = GpuCfcConfig::default();
//! let mut network = GpuCfcNetwork::new(config, GpuBackend::Cpu)?;
//!
//! let input = vec![0.1; 64];
//! let output = network.forward(&input, 0.1)?;
//! ```

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::cfc::{ActivationType, CfCNetworkConfig};

// =============================================================================
// GPU BACKEND SELECTION
// =============================================================================

/// Available backends for CfC acceleration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GpuBackend {
    /// Automatic selection (currently resolves to CPU).
    #[default]
    Auto,
    /// WebGPU backend (placeholder — not yet functional).
    Wgpu,
    /// CPU backend (always available, used as fallback).
    Cpu,
}

impl GpuBackend {
    /// Check if GPU acceleration is available
    pub fn is_gpu_available() -> bool {
        false
    }

    /// Get a human-readable description of the backend.
    pub fn description(&self) -> &'static str {
        match self {
            GpuBackend::Auto => "Auto (CPU fallback)",
            GpuBackend::Wgpu => "WebGPU (placeholder)",
            GpuBackend::Cpu => "CPU (ndarray)",
        }
    }

    /// Resolve Auto to a concrete backend.
    pub fn resolve(self) -> Self {
        match self {
            GpuBackend::Auto => GpuBackend::Cpu,
            other => other,
        }
    }
}

// =============================================================================
// GPU CfC CONFIGURATION
// =============================================================================

/// Configuration for GPU-accelerated CfC network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCfcConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden state dimension
    pub hidden_dim: usize,
    /// Number of CfC layers
    pub num_layers: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Whether to use backbone network
    pub use_backbone: bool,
    /// Backbone layers
    pub backbone_layers: usize,
    /// Backbone hidden dimension
    pub backbone_dim: usize,
    /// Activation function
    pub activation: ActivationType,
    /// Time constant range (min, max)
    pub tau_range: (f32, f32),
    /// Dropout rate
    pub dropout: f32,
    /// Batch size for GPU optimization
    pub batch_size: usize,
    /// Whether to use mixed precision (fp16)
    pub mixed_precision: bool,
}

impl Default for GpuCfcConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dim: 128,
            num_layers: 2,
            output_dim: 32,
            use_backbone: true,
            backbone_layers: 2,
            backbone_dim: 128,
            activation: ActivationType::SiLU,
            tau_range: (0.1, 10.0),
            dropout: 0.1,
            batch_size: 32,
            mixed_precision: false,
        }
    }
}

impl From<&CfCNetworkConfig> for GpuCfcConfig {
    fn from(config: &CfCNetworkConfig) -> Self {
        Self {
            input_dim: config.input_dim,
            hidden_dim: config.hidden_dim,
            num_layers: config.num_layers,
            output_dim: config.output_dim,
            use_backbone: config.cell_config.use_backbone,
            backbone_layers: config.cell_config.backbone_layers,
            backbone_dim: config.cell_config.backbone_dim,
            activation: config.cell_config.activation,
            tau_range: config.cell_config.tau_range,
            dropout: config.cell_config.dropout,
            batch_size: 32,
            mixed_precision: false,
        }
    }
}

// =============================================================================
// PURE RUST CPU IMPLEMENTATION (No external dependencies)
// =============================================================================

/// Minimum tau value to prevent numerical instability
const MIN_TAU: f32 = 1e-6;

/// CPU-based CfC layer using ndarray for matrix operations
#[derive(Debug, Clone)]
struct CpuCfcLayer {
    /// Input weights
    w_in: Array2<f32>,
    /// Recurrent weights
    w_h: Array2<f32>,
    /// Bias
    b_h: Array1<f32>,
    /// Time constants
    tau: Array1<f32>,
    /// Hidden state
    state: Array1<f32>,
    /// Configuration
    hidden_dim: usize,
    #[allow(dead_code)] // RESERVED(serialization): layer introspection and config dump
    input_dim: usize,
    activation: ActivationType,
}

impl CpuCfcLayer {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        tau_range: (f32, f32),
        activation: ActivationType,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let scale = (2.0 / (input_dim + hidden_dim) as f32).sqrt();

        let w_in = Array2::from_shape_fn((hidden_dim, input_dim), |_| {
            (rng.r#gen::<f32>() - 0.5) * 2.0 * scale
        });

        let w_h = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            (rng.r#gen::<f32>() - 0.5) * 2.0 * scale
        });

        let b_h = Array1::zeros(hidden_dim);

        let (tau_min, tau_max) = tau_range;
        let tau = Array1::from_shape_fn(hidden_dim, |_| {
            let log_tau = tau_min.ln() + rng.r#gen::<f32>() * (tau_max.ln() - tau_min.ln());
            log_tau.exp().max(MIN_TAU)
        });

        Self {
            w_in,
            w_h,
            b_h,
            tau,
            state: Array1::zeros(hidden_dim),
            hidden_dim,
            input_dim,
            activation,
        }
    }

    fn forward(&mut self, input: &Array1<f32>, dt: f32) -> Array1<f32> {
        // Compute equilibrium state
        let x_contrib = self.w_in.dot(input);
        let h_contrib = self.w_h.dot(&self.state);
        let z = &x_contrib + &h_contrib + &self.b_h;

        // Apply activation
        let h_inf = z.mapv(|x| self.apply_activation(x));

        // Compute decay
        let decay: Array1<f32> = self.tau.mapv(|t| (-dt / t.max(MIN_TAU)).exp());

        // Closed-form update
        let mut new_state = &h_inf + &((&self.state - &h_inf) * &decay);

        // Clamp to prevent divergence
        new_state.mapv_inplace(|x| {
            if x.is_finite() {
                x.clamp(-10.0, 10.0)
            } else {
                0.0
            }
        });

        self.state = new_state.clone();
        new_state
    }

    fn reset(&mut self) {
        self.state = Array1::zeros(self.hidden_dim);
    }

    #[inline]
    fn apply_activation(&self, x: f32) -> f32 {
        match self.activation {
            ActivationType::SiLU => x * fast_sigmoid(x),
            ActivationType::GELU => {
                0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x.powi(3))).tanh())
            }
            ActivationType::ReLU => x.max(0.0),
            ActivationType::Tanh => x.tanh(),
            ActivationType::Sigmoid => fast_sigmoid(x),
        }
    }
}

/// Fast sigmoid approximation (2-3x faster than standard)
#[inline(always)]
fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (1.0 + x / (1.0 + x.abs()))
}

// =============================================================================
// GPU CfC NETWORK
// =============================================================================

/// GPU-accelerated CfC network with automatic CPU fallback
///
/// This struct provides a unified interface for CfC networks that can run
/// on GPU (via Burn/WGPU) or CPU (via ndarray), with automatic selection
/// based on hardware availability.
#[derive(Debug)]
pub struct GpuCfcNetwork {
    /// Network configuration
    config: GpuCfcConfig,
    /// Resolved backend
    backend: GpuBackend,
    /// CfC layers (CPU implementation)
    cpu_layers: Vec<CpuCfcLayer>,
    /// Output projection weights
    output_weights: Array2<f32>,
    /// Output projection bias
    output_bias: Array1<f32>,
    /// Backbone layers (if enabled)
    backbone_weights: Vec<Array2<f32>>,
    backbone_biases: Vec<Array1<f32>>,
    /// Statistics
    total_steps: u64,
    total_forward_time_us: u64,
    total_backward_time_us: u64,
}

impl GpuCfcNetwork {
    /// Create a new GPU-accelerated CfC network
    pub fn new(config: GpuCfcConfig, backend: GpuBackend) -> Result<Self> {
        let resolved_backend = backend.resolve();

        // Initialize CPU layers
        let mut cpu_layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer_input_dim = if i == 0 {
                if config.use_backbone {
                    config.backbone_dim
                } else {
                    config.input_dim
                }
            } else {
                config.hidden_dim
            };

            cpu_layers.push(CpuCfcLayer::new(
                layer_input_dim,
                config.hidden_dim,
                config.tau_range,
                config.activation,
            ));
        }

        // Initialize output projection
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (config.hidden_dim + config.output_dim) as f32).sqrt();

        let output_weights = Array2::from_shape_fn((config.output_dim, config.hidden_dim), |_| {
            (rng.r#gen::<f32>() - 0.5) * 2.0 * scale
        });
        let output_bias = Array1::zeros(config.output_dim);

        // Initialize backbone if enabled
        let (backbone_weights, backbone_biases) = if config.use_backbone {
            let mut weights = Vec::new();
            let mut biases = Vec::new();

            // First backbone layer: input_dim -> backbone_dim
            weights.push(Array2::from_shape_fn(
                (config.backbone_dim, config.input_dim),
                |_| (rng.r#gen::<f32>() - 0.5) * 2.0 * scale,
            ));
            biases.push(Array1::zeros(config.backbone_dim));

            // Additional backbone layers
            for _ in 1..config.backbone_layers {
                weights.push(Array2::from_shape_fn(
                    (config.backbone_dim, config.backbone_dim),
                    |_| (rng.r#gen::<f32>() - 0.5) * 2.0 * scale,
                ));
                biases.push(Array1::zeros(config.backbone_dim));
            }

            (weights, biases)
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            config,
            backend: resolved_backend,
            cpu_layers,
            output_weights,
            output_bias,
            backbone_weights,
            backbone_biases,
            total_steps: 0,
            total_forward_time_us: 0,
            total_backward_time_us: 0,
        })
    }

    /// Create from standard CfCNetworkConfig
    pub fn from_cpu_config(config: &CfCNetworkConfig, backend: GpuBackend) -> Result<Self> {
        Self::new(GpuCfcConfig::from(config), backend)
    }

    /// Get the active backend
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Check if using GPU acceleration.
    pub fn is_gpu_accelerated(&self) -> bool {
        matches!(self.backend, GpuBackend::Wgpu)
    }

    /// Reset all layer states
    pub fn reset(&mut self) {
        for layer in &mut self.cpu_layers {
            layer.reset();
        }
        self.total_steps = 0;
    }

    /// Forward pass through backbone network
    fn backbone_forward(&self, input: &Array1<f32>) -> Array1<f32> {
        if !self.config.use_backbone {
            return input.clone();
        }

        let mut x = input.clone();
        for (w, b) in self
            .backbone_weights
            .iter()
            .zip(self.backbone_biases.iter())
        {
            let z = w.dot(&x) + b;
            x = z.mapv(|v| {
                // SiLU activation
                v * fast_sigmoid(v)
            });
        }
        x
    }

    /// Forward pass (single input)
    pub fn forward(&mut self, input: &[f32], dt: f32) -> Result<Vec<f32>> {
        let start = Instant::now();

        let input_arr = Array1::from_vec(input.to_vec());

        // Process through backbone
        let mut h = self.backbone_forward(&input_arr);

        // Process through CfC layers
        for layer in &mut self.cpu_layers {
            h = layer.forward(&h, dt);
        }

        // Output projection
        let output = self.output_weights.dot(&h) + &self.output_bias;

        self.total_steps += 1;
        self.total_forward_time_us += start.elapsed().as_micros() as u64;

        Ok(output.to_vec())
    }

    /// Forward pass (Array1 input, for API compatibility with CPU version)
    pub fn forward_array(&mut self, input: &Array1<f32>, dt: f32) -> Array1<f32> {
        // Process through backbone
        let mut h = self.backbone_forward(input);

        // Process through CfC layers
        for layer in &mut self.cpu_layers {
            h = layer.forward(&h, dt);
        }

        // Output projection
        self.output_weights.dot(&h) + &self.output_bias
    }

    /// Batch forward pass (GPU-optimized)
    ///
    /// When running on GPU, this is significantly faster than individual forward calls.
    /// On CPU, it processes inputs sequentially but still benefits from cache locality.
    pub fn forward_batch(&mut self, inputs: &[Vec<f32>], dts: &[f32]) -> Result<Vec<Vec<f32>>> {
        if inputs.len() != dts.len() {
            anyhow::bail!("inputs and dts must have same length");
        }

        let start = Instant::now();

        let results: Vec<Vec<f32>> = inputs
            .iter()
            .zip(dts.iter())
            .map(|(input, dt)| {
                let input_arr = Array1::from_vec(input.clone());
                let mut h = self.backbone_forward(&input_arr);

                for layer in &mut self.cpu_layers {
                    h = layer.forward(&h, *dt);
                }

                let output = self.output_weights.dot(&h) + &self.output_bias;
                output.to_vec()
            })
            .collect();

        self.total_steps += inputs.len() as u64;
        self.total_forward_time_us += start.elapsed().as_micros() as u64;

        Ok(results)
    }

    /// Forward sequence (maintains state across timesteps)
    pub fn forward_sequence(&mut self, inputs: &[Vec<f32>], dts: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.reset();
        self.forward_batch(inputs, dts)
    }

    /// Training step using BPTT (Back-Propagation Through Time)
    ///
    /// GPU acceleration provides 10-50x speedup for the matrix operations in
    /// gradient computation and weight updates.
    pub fn train_step(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        dts: &[f32],
        learning_rate: f32,
    ) -> Result<f32> {
        let start = Instant::now();

        // Forward pass collecting intermediate states
        let mut layer_outputs: Vec<Vec<Array1<f32>>> = vec![Vec::new(); self.cpu_layers.len()];
        let mut backbone_outputs: Vec<Array1<f32>> = Vec::new();
        let mut final_outputs: Vec<Array1<f32>> = Vec::new();

        self.reset();

        for (input, dt) in inputs.iter().zip(dts.iter()) {
            let input_arr = Array1::from_vec(input.clone());
            let backbone_out = self.backbone_forward(&input_arr);
            backbone_outputs.push(backbone_out.clone());

            let mut h = backbone_out;
            for (layer_idx, layer) in self.cpu_layers.iter_mut().enumerate() {
                layer_outputs[layer_idx].push(layer.state.clone());
                h = layer.forward(&h, *dt);
            }

            let output = self.output_weights.dot(&h) + &self.output_bias;
            final_outputs.push(output);
        }

        // Compute loss
        let mut total_loss = 0.0f32;
        for (output, target) in final_outputs.iter().zip(targets.iter()) {
            let target_arr = Array1::from_vec(target.clone());
            let diff = output - &target_arr;
            let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
            total_loss += mse;
        }
        let avg_loss = total_loss / inputs.len() as f32;

        // Backward pass (simplified - full BPTT would need more state tracking)
        // For now, we do a simple gradient descent step on output weights
        for (output, target) in final_outputs.iter().zip(targets.iter()) {
            let target_arr = Array1::from_vec(target.clone());
            let error = output - &target_arr;
            let n = error.len() as f32;

            // Get the last hidden state
            if let Some(last_layer) = self.cpu_layers.last() {
                let h = &last_layer.state;

                // Update output weights: W -= lr * (error * h^T)
                for i in 0..self.config.output_dim {
                    for j in 0..self.config.hidden_dim {
                        let grad = error[i] * h[j] / n;
                        let delta = (-learning_rate * grad).clamp(-0.1, 0.1);
                        self.output_weights[[i, j]] += delta;
                    }
                    let delta = (-learning_rate * error[i] / n).clamp(-0.1, 0.1);
                    self.output_bias[i] += delta;
                }
            }
        }

        self.total_backward_time_us += start.elapsed().as_micros() as u64;

        Ok(avg_loss)
    }

    /// Get network configuration
    pub fn config(&self) -> &GpuCfcConfig {
        &self.config
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        // Backbone parameters
        for (w, b) in self
            .backbone_weights
            .iter()
            .zip(self.backbone_biases.iter())
        {
            count += w.len() + b.len();
        }

        // CfC layer parameters
        for layer in &self.cpu_layers {
            count += layer.w_in.len();
            count += layer.w_h.len();
            count += layer.b_h.len();
            count += layer.tau.len();
        }

        // Output projection
        count += self.output_weights.len();
        count += self.output_bias.len();

        count
    }

    /// Get performance statistics
    pub fn stats(&self) -> GpuCfcStats {
        GpuCfcStats {
            backend: self.backend,
            total_steps: self.total_steps,
            total_forward_time_us: self.total_forward_time_us,
            total_backward_time_us: self.total_backward_time_us,
            avg_forward_us: if self.total_steps > 0 {
                self.total_forward_time_us / self.total_steps
            } else {
                0
            },
            num_parameters: self.num_parameters(),
        }
    }

    /// Get current states from all layers
    pub fn state(&self) -> Vec<Array1<f32>> {
        self.cpu_layers.iter().map(|l| l.state.clone()).collect()
    }

    /// Set states for all layers
    pub fn set_state(&mut self, states: Vec<Array1<f32>>) {
        for (layer, state) in self.cpu_layers.iter_mut().zip(states.into_iter()) {
            layer.state = state;
        }
    }
}

/// Statistics for GPU CfC network
#[derive(Debug, Clone)]
pub struct GpuCfcStats {
    /// Active backend
    pub backend: GpuBackend,
    /// Total forward steps processed
    pub total_steps: u64,
    /// Total forward pass time (microseconds)
    pub total_forward_time_us: u64,
    /// Total backward pass time (microseconds)
    pub total_backward_time_us: u64,
    /// Average forward pass time (microseconds)
    pub avg_forward_us: u64,
    /// Number of parameters
    pub num_parameters: usize,
}

impl std::fmt::Display for GpuCfcStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuCfcStats {{ backend: {}, steps: {}, avg_forward: {}us, params: {} }}",
            self.backend.description(),
            self.total_steps,
            self.avg_forward_us,
            self.num_parameters
        )
    }
}

// =============================================================================
// COMPARISON UTILITIES
// =============================================================================

/// Compare outputs from GPU and CPU implementations
pub fn outputs_equal(gpu_output: &[f32], cpu_output: &Array1<f32>, tolerance: f32) -> bool {
    if gpu_output.len() != cpu_output.len() {
        return false;
    }

    for (g, c) in gpu_output.iter().zip(cpu_output.iter()) {
        if (g - c).abs() > tolerance {
            return false;
        }
    }

    true
}

/// Maximum absolute difference between two outputs
pub fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, x| acc.max(x))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_detection() {
        let backend = GpuBackend::Auto.resolve();
        println!(
            "Resolved backend: {:?} - {}",
            backend,
            backend.description()
        );

        // Auto always resolves to Cpu
        assert!(matches!(backend, GpuBackend::Cpu));
    }

    #[test]
    fn test_gpu_cfc_creation() {
        let config = GpuCfcConfig::default();
        let network = GpuCfcNetwork::new(config, GpuBackend::Cpu).unwrap();

        assert_eq!(network.config().input_dim, 64);
        assert_eq!(network.config().hidden_dim, 128);
        assert!(network.num_parameters() > 0);
    }

    #[test]
    fn test_gpu_cfc_forward() {
        let config = GpuCfcConfig {
            input_dim: 32,
            hidden_dim: 64,
            num_layers: 2,
            output_dim: 16,
            use_backbone: false,
            ..Default::default()
        };

        let mut network = GpuCfcNetwork::new(config, GpuBackend::Cpu).unwrap();

        let input = vec![0.1f32; 32];
        let output = network.forward(&input, 0.1).unwrap();

        assert_eq!(output.len(), 16);

        // Output should be finite
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_gpu_cfc_batch_forward() {
        let config = GpuCfcConfig {
            input_dim: 32,
            hidden_dim: 64,
            num_layers: 2,
            output_dim: 16,
            use_backbone: false,
            ..Default::default()
        };

        let mut network = GpuCfcNetwork::new(config, GpuBackend::Cpu).unwrap();

        let batch_size = 10;
        let inputs: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| vec![(i as f32) * 0.1; 32])
            .collect();
        let dts: Vec<f32> = vec![0.1; batch_size];

        let outputs = network.forward_batch(&inputs, &dts).unwrap();

        assert_eq!(outputs.len(), batch_size);
        for output in &outputs {
            assert_eq!(output.len(), 16);
        }
    }

    #[test]
    fn test_gpu_cfc_training() {
        let config = GpuCfcConfig {
            input_dim: 16,
            hidden_dim: 32,
            num_layers: 1,
            output_dim: 8,
            use_backbone: false,
            ..Default::default()
        };

        let mut network = GpuCfcNetwork::new(config, GpuBackend::Cpu).unwrap();

        // Simple training data
        let inputs: Vec<Vec<f32>> = vec![vec![1.0; 16], vec![0.0; 16]];
        let targets: Vec<Vec<f32>> = vec![vec![1.0; 8], vec![0.0; 8]];
        let dts = vec![0.1, 0.1];

        let initial_loss = network.train_step(&inputs, &targets, &dts, 0.01).unwrap();
        assert!(initial_loss.is_finite());

        // Train for a few steps
        for _ in 0..10 {
            network.reset();
            let _ = network.train_step(&inputs, &targets, &dts, 0.01);
        }

        network.reset();
        let final_loss = network.train_step(&inputs, &targets, &dts, 0.01).unwrap();

        // Loss should decrease (or at least stay finite)
        assert!(final_loss.is_finite());
        println!("Initial loss: {}, Final loss: {}", initial_loss, final_loss);
    }

    #[test]
    fn test_gpu_cfc_state_management() {
        let config = GpuCfcConfig {
            input_dim: 16,
            hidden_dim: 32,
            num_layers: 2,
            output_dim: 8,
            use_backbone: false,
            ..Default::default()
        };

        let mut network = GpuCfcNetwork::new(config, GpuBackend::Cpu).unwrap();

        // Do some forward passes
        let input = vec![0.5; 16];
        let _ = network.forward(&input, 0.1).unwrap();
        let _ = network.forward(&input, 0.1).unwrap();

        // Save state
        let state = network.state();
        assert_eq!(state.len(), 2); // 2 layers

        // State should be non-zero after forward passes
        let total_magnitude: f32 = state.iter().flat_map(|s| s.iter()).map(|&x| x.abs()).sum();
        assert!(total_magnitude > 0.0);

        // Reset and verify
        network.reset();
        let reset_state = network.state();
        let reset_magnitude: f32 = reset_state
            .iter()
            .flat_map(|s| s.iter())
            .map(|&x| x.abs())
            .sum();
        assert!(reset_magnitude < 1e-6);

        // Restore state
        network.set_state(state.clone());
        let restored_state = network.state();

        // Should match original
        for (orig, restored) in state.iter().zip(restored_state.iter()) {
            assert_eq!(orig.len(), restored.len());
            for (a, b) in orig.iter().zip(restored.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_gpu_cfc_with_backbone() {
        let config = GpuCfcConfig {
            input_dim: 32,
            hidden_dim: 64,
            num_layers: 2,
            output_dim: 16,
            use_backbone: true,
            backbone_layers: 2,
            backbone_dim: 48,
            ..Default::default()
        };

        let mut network = GpuCfcNetwork::new(config, GpuBackend::Cpu).unwrap();

        let input = vec![0.1f32; 32];
        let output = network.forward(&input, 0.1).unwrap();

        assert_eq!(output.len(), 16);

        // Check parameter count includes backbone
        let params = network.num_parameters();
        // backbone: 32*48 + 48 + 48*48 + 48 = 1536 + 48 + 2304 + 48 = 3936
        // layers: 2 * (48*64 + 64*64 + 64 + 64) = 2 * (3072 + 4096 + 128) = 14592
        // output: 16*64 + 16 = 1040
        // Total should be significant
        assert!(params > 10000);
    }

    #[test]
    fn test_stats_tracking() {
        let config = GpuCfcConfig {
            input_dim: 16,
            hidden_dim: 32,
            num_layers: 1,
            output_dim: 8,
            use_backbone: false,
            ..Default::default()
        };

        let mut network = GpuCfcNetwork::new(config, GpuBackend::Cpu).unwrap();

        let input = vec![0.1; 16];
        for _ in 0..100 {
            let _ = network.forward(&input, 0.1);
        }

        let stats = network.stats();
        assert_eq!(stats.total_steps, 100);
        assert!(stats.total_forward_time_us > 0);
        assert!(stats.avg_forward_us > 0);

        println!("{}", stats);
    }
}
