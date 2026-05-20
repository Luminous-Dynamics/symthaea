// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC Projection Layer - Learned Dimension Conversion
//!
//! This module provides differentiable projection layers for converting
//! between different HDC dimensions. Unlike the fixed BLAKE3 expansion
//! used previously, these projections can be trained end-to-end.
//!
//! ## Key Features
//!
//! - **Learned Projections**: Weights optimized during training
//! - **Bidirectional**: Support both expansion and compression
//! - **Differentiable**: Full gradient flow for end-to-end training
//! - **Similarity Preserving**: Maintains relative distances after projection
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   LEARNED PROJECTION LAYER                       │
//! │                                                                  │
//! │   Input (dim_in)          Projection          Output (dim_out)  │
//! │   ┌──────────┐           ┌─────────┐         ┌──────────┐      │
//! │   │ x ∈ ℝ^n  │  ───W──>  │ Wx + b  │  ───>   │ y ∈ ℝ^m  │      │
//! │   └──────────┘           └─────────┘         └──────────┘      │
//! │                                                                  │
//! │   Gradient: ∂L/∂W = outer(∂L/∂y, x)                             │
//! │             ∂L/∂x = W^T @ ∂L/∂y                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::projection::{LearnedProjection, BidirectionalBridge};
//! use symthaea::hdc::unified_hv::ContinuousHV;
//!
//! // Single-direction projection
//! let mut proj = LearnedProjection::new(2048, 16384);
//! let input = ContinuousHV::random(2048, 42);
//! let output = proj.forward(&input);
//!
//! // Bidirectional bridge (encoder + decoder)
//! let mut bridge = BidirectionalBridge::new(2048, 16384);
//! let encoded = bridge.encode(&input);
//! let decoded = bridge.decode(&encoded);
//! ```

use crate::hdc::config::{STT_DIMENSION, hdc_dim};
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNED PROJECTION LAYER
// ═══════════════════════════════════════════════════════════════════════════════

/// Learned linear projection between HDC dimensions
///
/// Projects vectors from one dimension to another using a learned
/// weight matrix. Supports gradient-based optimization via Adam.
///
/// # Initialization
///
/// Uses Xavier/Glorot initialization for stable gradient flow:
/// `W ~ U(-√(6/(n+m)), √(6/(n+m)))`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedProjection {
    /// Projection weight matrix [output_dim × input_dim]
    /// Stored in row-major order: weights[i * input_dim + j] = W[i,j]
    weights: Vec<f32>,

    /// Optional bias vector [output_dim]
    bias: Option<Vec<f32>>,

    /// Adam optimizer: first moment (momentum)
    momentum: Vec<f32>,

    /// Adam optimizer: second moment (velocity)
    velocity: Vec<f32>,

    /// Adam optimizer: bias momentum (if bias exists)
    bias_momentum: Option<Vec<f32>>,

    /// Adam optimizer: bias velocity (if bias exists)
    bias_velocity: Option<Vec<f32>>,

    /// Input dimension
    input_dim: usize,

    /// Output dimension
    output_dim: usize,

    /// Training step counter (for Adam bias correction)
    step: usize,
}

impl LearnedProjection {
    /// Create a new learned projection layer
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input vector dimension
    /// * `output_dim` - Output vector dimension
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self::with_bias(input_dim, output_dim, false)
    }

    /// Create a learned projection with optional bias
    pub fn with_bias(input_dim: usize, output_dim: usize, use_bias: bool) -> Self {
        // Xavier/Glorot initialization
        let scale = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let mut weights = Vec::with_capacity(output_dim * input_dim);

        // Simple deterministic initialization for reproducibility
        let mut state: u64 = 0xDEADBEEF;
        for _ in 0..(output_dim * input_dim) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            weights.push(normalized * scale);
        }

        let bias = if use_bias {
            Some(vec![0.0; output_dim])
        } else {
            None
        };

        Self {
            weights,
            bias: bias.clone(),
            momentum: vec![0.0; output_dim * input_dim],
            velocity: vec![0.0; output_dim * input_dim],
            bias_momentum: bias.as_ref().map(|_| vec![0.0; output_dim]),
            bias_velocity: bias.as_ref().map(|_| vec![0.0; output_dim]),
            input_dim,
            output_dim,
            step: 0,
        }
    }

    /// Create a projection from STT dimension to core dimension
    pub fn stt_to_core() -> Self {
        Self::new(STT_DIMENSION, hdc_dim())
    }

    /// Create a projection from core dimension to STT dimension
    pub fn core_to_stt() -> Self {
        Self::new(hdc_dim(), STT_DIMENSION)
    }

    /// Forward pass: project input to output dimension
    ///
    /// Computes `y = Wx + b` where:
    /// - W is the weight matrix [output_dim × input_dim]
    /// - x is the input vector [input_dim]
    /// - b is the optional bias [output_dim]
    pub fn forward(&self, input: &ContinuousHV) -> ContinuousHV {
        assert_eq!(
            input.dim(),
            self.input_dim,
            "Input dimension mismatch: expected {}, got {}",
            self.input_dim,
            input.dim()
        );

        let mut output = vec![0.0f32; self.output_dim];

        // Matrix-vector multiplication: y = Wx
        for i in 0..self.output_dim {
            let row_start = i * self.input_dim;
            let mut sum = 0.0f32;

            // Process in chunks for better cache utilization
            for j in 0..self.input_dim {
                sum += self.weights[row_start + j] * input.values[j];
            }

            output[i] = sum;
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for i in 0..self.output_dim {
                output[i] += bias[i];
            }
        }

        ContinuousHV::from_vec(output)
    }

    /// Forward pass with raw f32 slices (for integration with other systems)
    pub fn forward_raw(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim, "Input dimension mismatch");

        let mut output = vec![0.0f32; self.output_dim];

        for i in 0..self.output_dim {
            let row_start = i * self.input_dim;
            let mut sum = 0.0f32;

            for j in 0..self.input_dim {
                sum += self.weights[row_start + j] * input[j];
            }

            output[i] = sum;
        }

        if let Some(ref bias) = self.bias {
            for i in 0..self.output_dim {
                output[i] += bias[i];
            }
        }

        output
    }

    /// Backward pass: compute gradients and update weights
    ///
    /// # Arguments
    ///
    /// * `input` - Original input to forward pass
    /// * `grad_output` - Gradient of loss w.r.t. output
    /// * `lr` - Learning rate
    ///
    /// # Returns
    ///
    /// Gradient w.r.t. input (for backpropagation to previous layers)
    pub fn backward(&mut self, input: &ContinuousHV, grad_output: &[f32], lr: f32) -> Vec<f32> {
        assert_eq!(input.dim(), self.input_dim);
        assert_eq!(grad_output.len(), self.output_dim);

        self.step += 1;

        // Adam hyperparameters
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps: f32 = 1e-8;

        // Bias correction factors
        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);

        // Compute gradient w.r.t. weights: dW = outer(grad_output, input)
        // And update weights with Adam
        for i in 0..self.output_dim {
            let row_start = i * self.input_dim;
            for j in 0..self.input_dim {
                let idx = row_start + j;
                let grad = grad_output[i] * input.values[j];

                // Update momentum: m = β1*m + (1-β1)*g
                self.momentum[idx] = beta1 * self.momentum[idx] + (1.0 - beta1) * grad;

                // Update velocity: v = β2*v + (1-β2)*g²
                self.velocity[idx] = beta2 * self.velocity[idx] + (1.0 - beta2) * grad * grad;

                // Bias-corrected estimates
                let m_hat = self.momentum[idx] / bias_correction1;
                let v_hat = self.velocity[idx] / bias_correction2;

                // Update weights: w = w - lr * m_hat / (√v_hat + ε)
                self.weights[idx] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }

        // Update bias with Adam (if present)
        if let (Some(bias), Some(bias_m), Some(bias_v)) = (
            &mut self.bias,
            &mut self.bias_momentum,
            &mut self.bias_velocity,
        ) {
            for i in 0..self.output_dim {
                let grad = grad_output[i];

                bias_m[i] = beta1 * bias_m[i] + (1.0 - beta1) * grad;
                bias_v[i] = beta2 * bias_v[i] + (1.0 - beta2) * grad * grad;

                let m_hat = bias_m[i] / bias_correction1;
                let v_hat = bias_v[i] / bias_correction2;

                bias[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }

        // Compute gradient w.r.t. input: dx = W^T @ grad_output
        let mut grad_input = vec![0.0f32; self.input_dim];

        for j in 0..self.input_dim {
            let mut sum = 0.0f32;
            for i in 0..self.output_dim {
                sum += self.weights[i * self.input_dim + j] * grad_output[i];
            }
            grad_input[j] = sum;
        }

        grad_input
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        let weight_params = self.output_dim * self.input_dim;
        let bias_params = self.bias.as_ref().map(|b| b.len()).unwrap_or(0);
        weight_params + bias_params
    }

    /// Reset optimizer state (for transfer learning)
    pub fn reset_optimizer(&mut self) {
        self.momentum.fill(0.0);
        self.velocity.fill(0.0);
        if let Some(ref mut m) = self.bias_momentum {
            m.fill(0.0);
        }
        if let Some(ref mut v) = self.bias_velocity {
            v.fill(0.0);
        }
        self.step = 0;
    }

    /// Get weight matrix as 2D slice (row-major)
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get mutable weight matrix
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIDIRECTIONAL BRIDGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Bidirectional projection bridge between two HDC dimensions
///
/// Provides both encoding (expansion) and decoding (compression) paths
/// with optional reconstruction loss for training.
///
/// # Training Objective
///
/// The bridge can be trained with a reconstruction loss:
/// ```text
/// L = L_task + λ * ||decode(encode(x)) - x||²
/// ```
///
/// This encourages the bridge to preserve information during round-trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidirectionalBridge {
    /// Encoder: source → target (expansion)
    encoder: LearnedProjection,

    /// Decoder: target → source (compression)
    decoder: LearnedProjection,

    /// Reconstruction loss weight
    recon_weight: f32,

    /// Source dimension
    source_dim: usize,

    /// Target dimension
    target_dim: usize,
}

impl BidirectionalBridge {
    /// Create a new bidirectional bridge
    ///
    /// # Arguments
    ///
    /// * `source_dim` - Source (typically smaller) dimension
    /// * `target_dim` - Target (typically larger) dimension
    pub fn new(source_dim: usize, target_dim: usize) -> Self {
        Self {
            encoder: LearnedProjection::new(source_dim, target_dim),
            decoder: LearnedProjection::new(target_dim, source_dim),
            recon_weight: 0.1,
            source_dim,
            target_dim,
        }
    }

    /// Create a bridge from STT to core dimension
    pub fn stt_to_core() -> Self {
        Self::new(STT_DIMENSION, hdc_dim())
    }

    /// Set reconstruction loss weight
    pub fn with_recon_weight(mut self, weight: f32) -> Self {
        self.recon_weight = weight;
        self
    }

    /// Encode: project from source to target dimension
    pub fn encode(&self, input: &ContinuousHV) -> ContinuousHV {
        self.encoder.forward(input)
    }

    /// Decode: project from target to source dimension
    pub fn decode(&self, input: &ContinuousHV) -> ContinuousHV {
        self.decoder.forward(input)
    }

    /// Round-trip: encode then decode
    pub fn round_trip(&self, input: &ContinuousHV) -> ContinuousHV {
        let encoded = self.encode(input);
        self.decode(&encoded)
    }

    /// Compute reconstruction loss
    ///
    /// Returns MSE between input and round-trip reconstruction.
    pub fn reconstruction_loss(&self, input: &ContinuousHV) -> f32 {
        let reconstructed = self.round_trip(input);
        mse(&reconstructed.values, &input.values)
    }

    /// Train step with target encoding loss
    ///
    /// # Arguments
    ///
    /// * `source_hv` - Source vector to encode
    /// * `target_hv` - Target encoding to match
    /// * `lr` - Learning rate
    ///
    /// # Returns
    ///
    /// Total loss (encoding loss + reconstruction loss)
    pub fn train_step(
        &mut self,
        source_hv: &ContinuousHV,
        target_hv: &ContinuousHV,
        lr: f32,
    ) -> f32 {
        // Forward pass
        let encoded = self.encoder.forward(source_hv);
        let reconstructed = self.decoder.forward(&encoded);

        // Compute losses
        let encoding_loss = mse(&encoded.values, &target_hv.values);
        let recon_loss = mse(&reconstructed.values, &source_hv.values);
        let total_loss = encoding_loss + self.recon_weight * recon_loss;

        // Backward pass for encoder
        let grad_encoded = grad_mse(&encoded.values, &target_hv.values);

        // Also add gradient from reconstruction path
        let grad_reconstructed = grad_mse(&reconstructed.values, &source_hv.values);
        let grad_encoded_from_recon = self.decoder.backward(&encoded, &grad_reconstructed, lr);

        // Combine gradients
        let grad_encoded_total: Vec<f32> = grad_encoded
            .iter()
            .zip(grad_encoded_from_recon.iter())
            .map(|(g1, g2)| g1 + self.recon_weight * g2)
            .collect();

        // Update encoder
        let _ = self.encoder.backward(source_hv, &grad_encoded_total, lr);

        total_loss
    }

    /// Get source dimension
    pub fn source_dim(&self) -> usize {
        self.source_dim
    }

    /// Get target dimension
    pub fn target_dim(&self) -> usize {
        self.target_dim
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        self.encoder.num_params() + self.decoder.num_params()
    }

    /// Get reference to encoder
    pub fn encoder(&self) -> &LearnedProjection {
        &self.encoder
    }

    /// Get reference to decoder
    pub fn decoder(&self) -> &LearnedProjection {
        &self.decoder
    }

    /// Get mutable reference to encoder
    pub fn encoder_mut(&mut self) -> &mut LearnedProjection {
        &mut self.encoder
    }

    /// Get mutable reference to decoder
    pub fn decoder_mut(&mut self) -> &mut LearnedProjection {
        &mut self.decoder
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RANDOM PROJECTION (non-learned baseline)
// ═══════════════════════════════════════════════════════════════════════════════

/// Random projection for dimension reduction/expansion
///
/// Uses Johnson-Lindenstrauss lemma: random projections approximately
/// preserve distances with high probability.
///
/// This is a non-learned alternative for cases where training isn't available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomProjection {
    /// Projection matrix (row-major)
    matrix: Vec<f32>,

    /// Input dimension
    input_dim: usize,

    /// Output dimension
    output_dim: usize,

    /// Random seed used for generation
    seed: u64,
}

impl RandomProjection {
    /// Create a new random projection
    ///
    /// Uses Gaussian random entries scaled by 1/√output_dim for
    /// approximate distance preservation.
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let scale = 1.0 / (output_dim as f32).sqrt();
        let mut matrix = Vec::with_capacity(output_dim * input_dim);

        let mut state = seed ^ 0x9E3779B97F4A7C15; // avoid xorshift64 fixed-point at 0
        for _ in 0..(output_dim * input_dim) {
            // Box-Muller for approximate Gaussian
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u1 = (state as f32 / u64::MAX as f32).max(1e-10);

            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u2 = state as f32 / u64::MAX as f32;

            let gaussian = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            matrix.push(gaussian * scale);
        }

        Self {
            matrix,
            input_dim,
            output_dim,
            seed,
        }
    }

    /// Project a vector
    pub fn project(&self, input: &ContinuousHV) -> ContinuousHV {
        assert_eq!(input.dim(), self.input_dim);

        let mut output = vec![0.0f32; self.output_dim];

        for i in 0..self.output_dim {
            let row_start = i * self.input_dim;
            let mut sum = 0.0f32;

            for j in 0..self.input_dim {
                sum += self.matrix[row_start + j] * input.values[j];
            }

            output[i] = sum;
        }

        ContinuousHV::from_vec(output)
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Mean squared error between two vectors
fn mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    sum / a.len() as f32
}

/// Gradient of MSE w.r.t. first argument
fn grad_mse(predicted: &[f32], target: &[f32]) -> Vec<f32> {
    assert_eq!(predicted.len(), target.len());
    let n = predicted.len() as f32;
    predicted
        .iter()
        .zip(target.iter())
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learned_projection_forward() {
        let proj = LearnedProjection::new(100, 200);
        let input = ContinuousHV::random(100, 42);

        let output = proj.forward(&input);

        assert_eq!(output.dim(), 200);
    }

    #[test]
    fn test_learned_projection_backward() {
        let mut proj = LearnedProjection::new(100, 200);
        let input = ContinuousHV::random(100, 42);
        let grad_output = vec![0.1f32; 200];

        let grad_input = proj.backward(&input, &grad_output, 0.001);

        assert_eq!(grad_input.len(), 100);
    }

    #[test]
    fn test_bidirectional_bridge_roundtrip() {
        let bridge = BidirectionalBridge::new(100, 400);
        let input = ContinuousHV::random(100, 42);

        let encoded = bridge.encode(&input);
        let decoded = bridge.decode(&encoded);

        assert_eq!(encoded.dim(), 400);
        assert_eq!(decoded.dim(), 100);

        // After training, reconstruction should improve
        // For now, just check dimensions are correct
    }

    #[test]
    fn test_reconstruction_loss() {
        let bridge = BidirectionalBridge::new(100, 400);
        let input = ContinuousHV::random(100, 42);

        let loss = bridge.reconstruction_loss(&input);

        // Loss should be non-negative
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_random_projection() {
        let proj = RandomProjection::new(1000, 100, 42);
        let input = ContinuousHV::random(1000, 42);

        let output = proj.project(&input);

        assert_eq!(output.dim(), 100);
    }

    #[test]
    fn test_random_projection_distance_preservation() {
        let proj = RandomProjection::new(1000, 500, 42);

        let a = ContinuousHV::random(1000, 1);
        let b = ContinuousHV::random(1000, 2);

        let original_sim = a.similarity(&b);

        let a_proj = proj.project(&a);
        let b_proj = proj.project(&b);

        let projected_sim = a_proj.similarity(&b_proj);

        // JL lemma: distances should be approximately preserved
        // Allow 30% tolerance for this test
        let diff = (original_sim - projected_sim).abs();
        assert!(
            diff < 0.3,
            "Similarity changed too much: {} vs {}",
            original_sim,
            projected_sim
        );
    }

    #[test]
    fn test_mse_gradient() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let loss = mse(&a, &b);
        assert!((loss - 0.0).abs() < 1e-6);

        let grad = grad_mse(&a, &b);
        assert!(grad.iter().all(|&g| g.abs() < 1e-6));
    }

    #[test]
    fn test_projection_training_reduces_loss() {
        let mut bridge = BidirectionalBridge::new(50, 200);

        // Create aligned source and target
        let source = ContinuousHV::random(50, 42);
        let target = ContinuousHV::random(200, 42);

        let initial_loss = bridge.train_step(&source, &target, 0.0);

        // Train for a few steps
        for _ in 0..10 {
            let _ = bridge.train_step(&source, &target, 0.01);
        }

        let final_loss = bridge.train_step(&source, &target, 0.0);

        // Loss should decrease with training
        assert!(
            final_loss < initial_loss,
            "Loss should decrease: {} -> {}",
            initial_loss,
            final_loss
        );
    }
}
