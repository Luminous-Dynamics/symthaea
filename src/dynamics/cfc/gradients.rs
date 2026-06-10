// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gradient structures and optimizer state for CfC backpropagation.

use ndarray::{Array1, Array2};

/// Gradient accumulators for CfC backpropagation
#[derive(Debug, Clone)]
pub struct CfCGradients {
    /// Input weight gradients
    pub dw_in: Array2<f32>,
    /// Recurrent weight gradients
    pub dw_h: Array2<f32>,
    /// Bias gradients
    pub db_h: Array1<f32>,
    /// Time constant gradients
    pub dtau: Array1<f32>,
}

/// Cache for forward pass intermediate values (optimization: avoids recomputation in backward pass)
#[derive(Debug, Clone)]
pub struct CfCCellCache {
    /// Processed input (after backbone if enabled)
    pub processed_input: Array1<f32>,
    /// Pre-activation values (z = W_in * input + W_h * state + b_h)
    pub z: Array1<f32>,
    /// Post-activation equilibrium state (h_inf = activation(z))
    pub h_inf: Array1<f32>,
    /// Decay factor (exp(-dt/tau))
    pub decay: Array1<f32>,
    /// State at time of forward pass (needed for gradient computation)
    pub state_at_forward: Array1<f32>,
}

/// Adam optimizer state
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment estimates
    pub m_w_in: Array2<f32>,
    pub m_w_h: Array2<f32>,
    pub m_b_h: Array1<f32>,
    pub m_tau: Array1<f32>,
    /// Second moment estimates
    pub v_w_in: Array2<f32>,
    pub v_w_h: Array2<f32>,
    pub v_b_h: Array1<f32>,
    pub v_tau: Array1<f32>,
    /// Step counter
    pub t: u64,
    /// Hyperparameters
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl AdamState {
    pub(crate) fn new(hidden_dim: usize, input_dim: usize) -> Self {
        Self {
            m_w_in: Array2::zeros((hidden_dim, input_dim)),
            m_w_h: Array2::zeros((hidden_dim, hidden_dim)),
            m_b_h: Array1::zeros(hidden_dim),
            m_tau: Array1::zeros(hidden_dim),
            v_w_in: Array2::zeros((hidden_dim, input_dim)),
            v_w_h: Array2::zeros((hidden_dim, hidden_dim)),
            v_b_h: Array1::zeros(hidden_dim),
            v_tau: Array1::zeros(hidden_dim),
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// Adam optimizer state for the output projection layer
#[derive(Debug, Clone)]
pub struct OutputAdamState {
    pub m_w: Array2<f32>,
    pub v_w: Array2<f32>,
    pub m_b: Array1<f32>,
    pub v_b: Array1<f32>,
    pub t: u64,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl OutputAdamState {
    pub(crate) fn new(output_dim: usize, hidden_dim: usize) -> Self {
        Self {
            m_w: Array2::zeros((output_dim, hidden_dim)),
            v_w: Array2::zeros((output_dim, hidden_dim)),
            m_b: Array1::zeros(output_dim),
            v_b: Array1::zeros(output_dim),
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}
