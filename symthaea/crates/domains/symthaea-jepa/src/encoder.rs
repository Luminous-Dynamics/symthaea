// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Context encoder: projects 16,384D HDC vectors into compact latent space.
//!
//! Single linear projection + SiLU activation. The input HDC vectors already
//! have rich compositional structure from binding/bundling operations, so a
//! simple projection suffices — we're compressing, not learning features.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Context encoder: ContinuousHV (16,384D) → latent Vec<f32> (128D default).
///
/// Architecture: linear projection W×x + b, followed by SiLU activation.
/// Weights are learned via backpropagation through the predictor's loss signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEncoder {
    /// Projection weights, flattened row-major: [latent_dim × input_dim].
    pub weights: Vec<f32>,
    /// Bias vector: [latent_dim].
    pub bias: Vec<f32>,
    /// Input dimension (HDC_DIMENSION = 16,384).
    pub input_dim: usize,
    /// Output dimension (latent_dim, default 128).
    pub latent_dim: usize,
}

impl ContextEncoder {
    /// Create a new encoder with Xavier/Glorot initialization from seed.
    pub fn new(input_dim: usize, latent_dim: usize, seed: u64) -> Self {
        let scale = (2.0 / (input_dim + latent_dim) as f64).sqrt() as f32;
        let weights = init_weights(input_dim * latent_dim, seed, scale);
        let bias = vec![0.0; latent_dim];
        Self {
            weights,
            bias,
            input_dim,
            latent_dim,
        }
    }

    /// Encode a ContinuousHV into latent space.
    ///
    /// Computes: SiLU(W × x + b)
    /// where SiLU(z) = z × sigmoid(z)
    pub fn encode(&self, hv: &ContinuousHV) -> Vec<f32> {
        let x = &hv.values;
        debug_assert_eq!(x.len(), self.input_dim);
        let mut out = self.bias.clone();

        // Matrix-vector multiply: out[i] += sum_j(W[i*input_dim + j] * x[j])
        for i in 0..self.latent_dim {
            let row_start = i * self.input_dim;
            let mut acc = 0.0f32;
            for j in 0..self.input_dim {
                acc += self.weights[row_start + j] * x[j];
            }
            out[i] += acc;
        }

        // SiLU activation: z * sigmoid(z)
        for v in &mut out {
            *v = silu(*v);
        }

        out
    }

    /// Compute gradients of the encoder weights given upstream gradient.
    ///
    /// Returns (weight_grad, bias_grad, input_grad) where:
    /// - weight_grad: [latent_dim × input_dim] for encoder weight update
    /// - bias_grad: [latent_dim] for bias update
    /// - input_grad: [input_dim] for backprop (unused currently, reserved)
    pub fn backward(&self, hv: &ContinuousHV, latent_grad: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let x = &hv.values;

        // Recompute pre-activation for SiLU derivative
        let mut pre_act = self.bias.clone();
        for i in 0..self.latent_dim {
            let row_start = i * self.input_dim;
            let mut acc = 0.0f32;
            for j in 0..self.input_dim {
                acc += self.weights[row_start + j] * x[j];
            }
            pre_act[i] += acc;
        }

        // SiLU derivative: sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
        //                = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        let mut act_grad = vec![0.0f32; self.latent_dim];
        for i in 0..self.latent_dim {
            let sig = sigmoid(pre_act[i]);
            act_grad[i] = latent_grad[i] * (sig + pre_act[i] * sig * (1.0 - sig));
        }

        // Weight gradient: dL/dW[i,j] = act_grad[i] * x[j]
        let mut weight_grad = vec![0.0f32; self.input_dim * self.latent_dim];
        for i in 0..self.latent_dim {
            let row_start = i * self.input_dim;
            for j in 0..self.input_dim {
                weight_grad[row_start + j] = act_grad[i] * x[j];
            }
        }

        // Bias gradient = act_grad
        (weight_grad, act_grad)
    }

    /// Apply gradient update (SGD with optional weight decay).
    pub fn apply_gradients(
        &mut self,
        weight_grad: &[f32],
        bias_grad: &[f32],
        lr: f32,
        weight_decay: f32,
    ) {
        for (w, g) in self.weights.iter_mut().zip(weight_grad.iter()) {
            *w -= lr * (g + weight_decay * *w);
        }
        for (b, g) in self.bias.iter_mut().zip(bias_grad.iter()) {
            *b -= lr * g;
        }
    }

    /// Read weights (for EMA copy to target encoder).
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Read bias.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
}

/// SiLU activation: z * sigmoid(z).
/// Smooth, non-monotonic, better gradient flow than ReLU.
#[inline]
fn silu(z: f32) -> f32 {
    z * sigmoid(z)
}

/// Sigmoid: 1 / (1 + exp(-z)).
#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Initialize weights with deterministic xorshift64 PRNG, scaled by `scale`.
fn init_weights(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut weights = Vec::with_capacity(count);
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let normalized = ((state >> 40) as f32) * (2.0 / (1u64 << 24) as f32) - 1.0;
        weights.push(normalized * scale);
    }
    weights
}
