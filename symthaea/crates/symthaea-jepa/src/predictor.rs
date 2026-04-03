// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Latent predictor: maps (current_latent, action) → predicted next latent.
//!
//! Architecture: concat(latent, one_hot_action) → hidden(SiLU) → output
//!
//! This is the core JEPA component: it predicts the *target encoder's*
//! representation of the next state, given the current state and action.
//! The loss is cosine similarity in latent space (not reconstruction).

use serde::{Deserialize, Serialize};

/// Latent predictor network.
///
/// Two-layer MLP: input(latent_dim + num_actions) → hidden(latent_dim, SiLU) → output(latent_dim).
/// The hidden layer matches the latent dimension to avoid information bottleneck.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentPredictor {
    /// Layer 1 weights: [latent_dim × (latent_dim + num_actions)], row-major.
    w1: Vec<f32>,
    /// Layer 1 bias: [latent_dim].
    b1: Vec<f32>,
    /// Layer 2 weights: [latent_dim × latent_dim], row-major.
    w2: Vec<f32>,
    /// Layer 2 bias: [latent_dim].
    b2: Vec<f32>,
    /// Latent dimension.
    latent_dim: usize,
    /// Number of action types (8 motor commands from FEP).
    num_actions: usize,
}

impl LatentPredictor {
    /// Create with Xavier initialization.
    pub fn new(latent_dim: usize, num_actions: usize, seed: u64) -> Self {
        let input_dim = latent_dim + num_actions;
        let scale1 = (2.0 / (input_dim + latent_dim) as f64).sqrt() as f32;
        let scale2 = (2.0 / (latent_dim + latent_dim) as f64).sqrt() as f32;

        Self {
            w1: init_weights(latent_dim * input_dim, seed, scale1),
            b1: vec![0.0; latent_dim],
            w2: init_weights(latent_dim * latent_dim, seed.wrapping_add(1), scale2),
            b2: vec![0.0; latent_dim],
            latent_dim,
            num_actions,
        }
    }

    /// Predict next latent state given current latent and action.
    ///
    /// Returns predicted latent vector of size `latent_dim`.
    pub fn predict(&self, latent: &[f32], action: u8) -> Vec<f32> {
        let input = self.make_input(latent, action);
        let hidden = self.forward_layer1(&input);
        self.forward_layer2(&hidden)
    }

    /// Backpropagate through predictor and update weights.
    ///
    /// Given the gradient of the loss w.r.t. the output (from cosine loss),
    /// compute gradients for both layers and apply SGD update.
    ///
    /// Returns gradient w.r.t. input latent (for upstream backprop to encoder).
    pub fn backward(
        &mut self,
        latent: &[f32],
        action: u8,
        output_grad: &[f32],
        lr: f32,
    ) -> Vec<f32> {
        let input = self.make_input(latent, action);
        let input_dim = self.latent_dim + self.num_actions;

        // Forward pass (recompute activations for gradient)
        let hidden_pre = self.linear(&self.w1, &self.b1, &input, input_dim, self.latent_dim);
        let hidden = hidden_pre.iter().map(|&z| silu(z)).collect::<Vec<_>>();

        // Backward through layer 2: output_grad → hidden_grad
        let mut hidden_grad = vec![0.0f32; self.latent_dim];
        for i in 0..self.latent_dim {
            for j in 0..self.latent_dim {
                hidden_grad[j] += output_grad[i] * self.w2[i * self.latent_dim + j];
            }
        }
        // Update layer 2 weights
        for i in 0..self.latent_dim {
            for j in 0..self.latent_dim {
                self.w2[i * self.latent_dim + j] -= lr * output_grad[i] * hidden[j];
            }
            self.b2[i] -= lr * output_grad[i];
        }

        // Backward through SiLU activation
        let mut act_grad = vec![0.0f32; self.latent_dim];
        for i in 0..self.latent_dim {
            let sig = sigmoid(hidden_pre[i]);
            act_grad[i] = hidden_grad[i] * (sig + hidden_pre[i] * sig * (1.0 - sig));
        }

        // Backward through layer 1: act_grad → input_grad
        let mut input_grad = vec![0.0f32; input_dim];
        for i in 0..self.latent_dim {
            for j in 0..input_dim {
                input_grad[j] += act_grad[i] * self.w1[i * input_dim + j];
            }
        }
        // Update layer 1 weights
        for i in 0..self.latent_dim {
            for j in 0..input_dim {
                self.w1[i * input_dim + j] -= lr * act_grad[i] * input[j];
            }
            self.b1[i] -= lr * act_grad[i];
        }

        // Return gradient w.r.t. latent (first latent_dim elements of input_grad)
        input_grad.truncate(self.latent_dim);
        input_grad
    }

    fn make_input(&self, latent: &[f32], action: u8) -> Vec<f32> {
        let input_dim = self.latent_dim + self.num_actions;
        let mut input = Vec::with_capacity(input_dim);
        input.extend_from_slice(latent);
        // One-hot action encoding
        for a in 0..self.num_actions {
            input.push(if a == action as usize { 1.0 } else { 0.0 });
        }
        input
    }

    fn forward_layer1(&self, input: &[f32]) -> Vec<f32> {
        let input_dim = self.latent_dim + self.num_actions;
        let pre = self.linear(&self.w1, &self.b1, input, input_dim, self.latent_dim);
        pre.into_iter().map(silu).collect()
    }

    fn forward_layer2(&self, hidden: &[f32]) -> Vec<f32> {
        self.linear(&self.w2, &self.b2, hidden, self.latent_dim, self.latent_dim)
    }

    fn linear(
        &self,
        weights: &[f32],
        bias: &[f32],
        input: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let mut out = bias.to_vec();
        for i in 0..out_dim {
            let row_start = i * in_dim;
            let mut acc = 0.0f32;
            for j in 0..in_dim {
                acc += weights[row_start + j] * input[j];
            }
            out[i] += acc;
        }
        out
    }
}

#[inline]
fn silu(z: f32) -> f32 {
    z * sigmoid(z)
}

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

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
