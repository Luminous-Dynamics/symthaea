// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Recurrent Highway Network projection layer for Broca.
//!
//! Sits between CfC output_hv and logit computation, adding depth
//! without vanishing gradients via T/C highway gates.
//!
//! Architecture: Bottleneck RHN (16,384D → 512D → 16,384D per layer)
//! - T gate (transform): T = σ(Wt·h + bt)
//! - H (transform fn): H = tanh(Wh·h + bh)
//! - Output: T·H + (1-T)·h (highway residual in projected space)
//! - Then project back: 512D → 16,384D
//!
//! Consciousness coupling: T_init = sigmoid(consciousness_level * 4 - 2)
//! At psi=0.5, T≈0.5 (equal mix). At psi=1.0, T≈0.88 (mostly transform).
//! At psi=0.0, T≈0.12 (mostly carry = pass-through identity).

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::HDC_DIMENSION;

pub const HIGHWAY_BOTTLENECK_DIM: usize = 512;
pub const HIGHWAY_FULL_DIM: usize = HDC_DIMENSION; // 16,384

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrocaHighwayProjection {
    /// Down-projection: full_dim → bottleneck_dim weights
    pub down_proj: Vec<f32>, // [bottleneck × full_dim]
    /// Transform gate weights in bottleneck space
    pub gate_weights: Vec<f32>, // [bottleneck × bottleneck]
    pub gate_bias: Vec<f32>, // [bottleneck]
    /// Transform (H) weights in bottleneck space
    pub transform_weights: Vec<f32>, // [bottleneck × bottleneck]
    pub transform_bias: Vec<f32>, // [bottleneck]
    /// Up-projection: bottleneck_dim → full_dim
    pub up_proj: Vec<f32>, // [full_dim × bottleneck]
    /// Number of highway layers (depth)
    pub n_layers: usize,
    /// Whether to modulate T gate with consciousness
    pub consciousness_coupled: bool,
}

impl BrocaHighwayProjection {
    pub fn new(genesis: &GenesisSeed, n_layers: usize) -> Self {
        // Initialize weights with scaled genesis-seeded random values
        // down_proj: scale = 1.0 / sqrt(full_dim)
        let scale_down = 1.0 / (HIGHWAY_FULL_DIM as f32).sqrt();
        let down_proj = Self::init_weights(
            genesis,
            "highway_down",
            HIGHWAY_BOTTLENECK_DIM * HIGHWAY_FULL_DIM,
            scale_down,
        );

        // gate_weights: initialized small so carry dominates initially
        let scale_gate = 0.01;
        let gate_weights = Self::init_weights(
            genesis,
            "highway_gate_w",
            HIGHWAY_BOTTLENECK_DIM * HIGHWAY_BOTTLENECK_DIM,
            scale_gate,
        );
        // gate_bias: negative to keep T gate closed initially
        let gate_bias = vec![-2.0; HIGHWAY_BOTTLENECK_DIM];

        // transform_weights: scale = 1.0 / sqrt(bottleneck_dim)
        let scale_trans = 1.0 / (HIGHWAY_BOTTLENECK_DIM as f32).sqrt();
        let transform_weights = Self::init_weights(
            genesis,
            "highway_trans_w",
            HIGHWAY_BOTTLENECK_DIM * HIGHWAY_BOTTLENECK_DIM,
            scale_trans,
        );
        let transform_bias = vec![0.0; HIGHWAY_BOTTLENECK_DIM];

        // up_proj: scale = 1.0 / sqrt(bottleneck_dim)
        let scale_up = 1.0 / (HIGHWAY_BOTTLENECK_DIM as f32).sqrt();
        let up_proj = Self::init_weights(
            genesis,
            "highway_up",
            HIGHWAY_FULL_DIM * HIGHWAY_BOTTLENECK_DIM,
            scale_up,
        );

        Self {
            down_proj,
            gate_weights,
            gate_bias,
            transform_weights,
            transform_bias,
            up_proj,
            n_layers,
            consciousness_coupled: true,
        }
    }

    fn init_weights(genesis: &GenesisSeed, label: &str, size: usize, scale: f32) -> Vec<f32> {
        let chunk_size = 16384;
        let mut weights = Vec::with_capacity(size);
        let mut chunk_idx = 0;
        while weights.len() < size {
            let chunk_label = format!("{label}::chunk{chunk_idx}");
            let hv = genesis.hv(&chunk_label, chunk_size);
            let remaining = size - weights.len();
            let take = remaining.min(chunk_size);
            weights.extend_from_slice(&hv.values[..take]);
            chunk_idx += 1;
        }
        for w in &mut weights {
            *w *= scale;
        }
        weights
    }

    /// Flatten all weights into a single vector for checkpoint serialization.
    pub fn flatten_weights(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(
            self.down_proj.len()
                + self.gate_weights.len()
                + self.gate_bias.len()
                + self.transform_weights.len()
                + self.transform_bias.len()
                + self.up_proj.len(),
        );
        flat.extend_from_slice(&self.down_proj);
        flat.extend_from_slice(&self.gate_weights);
        flat.extend_from_slice(&self.gate_bias);
        flat.extend_from_slice(&self.transform_weights);
        flat.extend_from_slice(&self.transform_bias);
        flat.extend_from_slice(&self.up_proj);
        flat
    }

    /// Restore weights from a flattened checkpoint vector.
    pub fn restore_weights(&mut self, weights: &[f32]) {
        let mut offset = 0;

        let down_len = self.down_proj.len();
        if offset + down_len <= weights.len() {
            self.down_proj
                .copy_from_slice(&weights[offset..offset + down_len]);
            offset += down_len;
        }

        let gate_len = self.gate_weights.len();
        if offset + gate_len <= weights.len() {
            self.gate_weights
                .copy_from_slice(&weights[offset..offset + gate_len]);
            offset += gate_len;
        }

        let gate_bias_len = self.gate_bias.len();
        if offset + gate_bias_len <= weights.len() {
            self.gate_bias
                .copy_from_slice(&weights[offset..offset + gate_bias_len]);
            offset += gate_bias_len;
        }

        let trans_len = self.transform_weights.len();
        if offset + trans_len <= weights.len() {
            self.transform_weights
                .copy_from_slice(&weights[offset..offset + trans_len]);
            offset += trans_len;
        }

        let trans_bias_len = self.transform_bias.len();
        if offset + trans_bias_len <= weights.len() {
            self.transform_bias
                .copy_from_slice(&weights[offset..offset + trans_bias_len]);
            offset += trans_bias_len;
        }

        let up_len = self.up_proj.len();
        if offset + up_len <= weights.len() {
            self.up_proj
                .copy_from_slice(&weights[offset..offset + up_len]);
        }
    }

    /// Apply highway projection to output_hv.
    /// psi: consciousness level for T gate modulation.
    pub fn project(&self, output_hv: &ContinuousHV, psi: f32) -> ContinuousHV {
        let x = &output_hv.values;

        // 1. Down-project: full_dim → bottleneck
        let mut h = self.matmul_vec(&self.down_proj, x, HIGHWAY_BOTTLENECK_DIM, HIGHWAY_FULL_DIM);

        for _ in 0..self.n_layers {
            // 2. Compute T gate (transform gate)
            let t_logits = self.matmul_vec(
                &self.gate_weights,
                &h,
                HIGHWAY_BOTTLENECK_DIM,
                HIGHWAY_BOTTLENECK_DIM,
            );
            let mut t: Vec<f32> = t_logits
                .iter()
                .zip(&self.gate_bias)
                .map(|(l, b)| sigmoid(l + b))
                .collect();

            // Consciousness coupling: scale T by psi
            if self.consciousness_coupled {
                let psi_scale = sigmoid(psi * 4.0 - 2.0); // 0.12..0.88
                for ti in &mut t {
                    *ti *= psi_scale;
                }
            }

            // 3. Compute H (transform function)
            let h_raw = self.matmul_vec(
                &self.transform_weights,
                &h,
                HIGHWAY_BOTTLENECK_DIM,
                HIGHWAY_BOTTLENECK_DIM,
            );
            let h_transformed: Vec<f32> = h_raw
                .iter()
                .zip(&self.transform_bias)
                .map(|(v, b)| tanh(v + b))
                .collect();

            // 4. Highway: new_h = T·H(h) + (1-T)·h
            h = t
                .iter()
                .zip(h_transformed.iter())
                .zip(h.iter())
                .map(|((ti, hi), xi)| ti * hi + (1.0 - ti) * xi)
                .collect();
        }

        // 5. Up-project: bottleneck → full_dim
        let projected =
            self.matmul_vec(&self.up_proj, &h, HIGHWAY_FULL_DIM, HIGHWAY_BOTTLENECK_DIM);

        // 6. Residual: blend with original (highway at network level too)
        let residual_scale = 0.3_f32; // Conservative — preserve CfC dynamics
        let result: Vec<f32> = projected
            .iter()
            .zip(x.iter())
            .map(|(p, o)| p * residual_scale + o * (1.0 - residual_scale))
            .collect();

        ContinuousHV::from_values(result).normalize()
    }

    fn matmul_vec(&self, w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0; out_dim];
        for i in 0..out_dim {
            let row_offset = i * in_dim;
            let mut sum = 0.0;
            for j in 0..in_dim {
                sum += w[row_offset + j] * x[j];
            }
            out[i] = sum;
        }
        out
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f32) -> f32 {
    x.tanh()
}
