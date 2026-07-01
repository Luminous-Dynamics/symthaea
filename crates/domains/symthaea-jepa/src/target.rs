// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Target encoder: EMA-updated copy of context encoder.
//!
//! The target encoder produces the "ground truth" latent representation
//! against which the predictor's output is compared. Its weights are an
//! exponential moving average of the context encoder — never trained
//! directly via gradients.
//!
//! This asymmetry (gradient-trained context vs. EMA-tracked target) is
//! the core mechanism preventing representation collapse in JEPA/BYOL:
//! - Grill et al. (2020): "Bootstrap Your Own Latent"
//! - Assran et al. (2023): I-JEPA

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::encoder::ContextEncoder;

/// Target encoder: EMA-tracked copy of the context encoder.
///
/// `target_w = momentum * target_w + (1 - momentum) * context_w`
///
/// With momentum=0.996, the target tracks the context over ~250 updates
/// (half-life ≈ 173 steps), providing a slowly-moving prediction target
/// that prevents the trivial solution of mapping everything to a constant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetEncoder {
    /// Projection weights (EMA copy of context encoder weights).
    weights: Vec<f32>,
    /// Bias (EMA copy of context encoder bias).
    bias: Vec<f32>,
    /// Input dimension.
    input_dim: usize,
    /// Latent dimension.
    latent_dim: usize,
}

impl TargetEncoder {
    /// Create a target encoder as an exact copy of the context encoder.
    pub fn from_context(context: &ContextEncoder) -> Self {
        Self {
            weights: context.weights().to_vec(),
            bias: context.bias().to_vec(),
            input_dim: context.input_dim,
            latent_dim: context.latent_dim,
        }
    }

    /// Update weights via EMA from context encoder.
    ///
    /// `target_w = momentum * target_w + (1 - momentum) * context_w`
    pub fn update_from_context(&mut self, context: &ContextEncoder, momentum: f32) {
        let m = momentum;
        let m_inv = 1.0 - m;
        for (tw, cw) in self.weights.iter_mut().zip(context.weights().iter()) {
            *tw = m * *tw + m_inv * cw;
        }
        for (tb, cb) in self.bias.iter_mut().zip(context.bias().iter()) {
            *tb = m * *tb + m_inv * cb;
        }
    }

    /// Encode a ContinuousHV into latent space (same math as ContextEncoder).
    ///
    /// Computes: SiLU(W × x + b)
    pub fn encode(&self, hv: &ContinuousHV) -> Vec<f32> {
        let x = &hv.values;
        debug_assert_eq!(x.len(), self.input_dim);
        let mut out = self.bias.clone();

        for i in 0..self.latent_dim {
            let row_start = i * self.input_dim;
            let mut acc = 0.0f32;
            for j in 0..self.input_dim {
                acc += self.weights[row_start + j] * x[j];
            }
            out[i] += acc;
        }

        // SiLU activation
        for v in &mut out {
            let sig = 1.0 / (1.0 + (-*v).exp());
            *v *= sig;
        }

        out
    }
}
