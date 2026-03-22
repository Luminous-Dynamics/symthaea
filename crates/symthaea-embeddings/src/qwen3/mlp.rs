// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Qwen3 SwiGLU MLP
//!
//! Three-linear MLP with SiLU-gated activation: `down(silu(gate(x)) * up(x))`.
//! Weight keys match HuggingFace naming (gate_proj, up_proj, down_proj) for
//! direct safetensors loading.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// SwiGLU MLP block used in each Qwen3 decoder layer.
#[derive(Module, Debug)]
pub struct Qwen3Mlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

/// Configuration for [`Qwen3Mlp`].
#[derive(Debug, Clone)]
pub struct Qwen3MlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl Qwen3MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3Mlp<B> {
        let no_bias = |d_in, d_out| LinearConfig::new(d_in, d_out).with_bias(false);

        Qwen3Mlp {
            gate_proj: no_bias(self.hidden_size, self.intermediate_size).init(device),
            up_proj: no_bias(self.hidden_size, self.intermediate_size).init(device),
            down_proj: no_bias(self.intermediate_size, self.hidden_size).init(device),
        }
    }
}

impl<B: Backend> Qwen3Mlp<B> {
    /// Forward pass: `down_proj(silu(gate_proj(x)) * up_proj(x))`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}
