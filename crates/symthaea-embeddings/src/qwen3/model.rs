// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Qwen3 Transformer Model (embedding-mode)
//!
//! Full decoder-stack: `embed_tokens → N × (pre-norm → attn → residual →
//! post-norm → MLP → residual) → final_norm → hidden_states`.
//!
//! Factory methods `qwen3_06b()` / `qwen3_15b()` mirror HuggingFace configs.

use burn::nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;

use super::attention::{Qwen3Attention, Qwen3AttentionConfig};
use super::mlp::{Qwen3Mlp, Qwen3MlpConfig};

// ── Config ───────────────────────────────────────────────────────────

/// Architecture hyper-parameters for a Qwen3 model variant.
#[derive(Debug, Clone)]
pub struct Qwen3ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Qwen3ModelConfig {
    /// Qwen3-Embedding-0.6B (1024-D hidden, 16 Q / 8 KV heads)
    pub fn qwen3_06b() -> Self {
        Self {
            vocab_size: 151_669,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32_768,
        }
    }

    /// Qwen3-Embedding-1.5B (1536-D hidden, 12 Q / 2 KV heads)
    pub fn qwen3_15b() -> Self {
        Self {
            vocab_size: 151_669,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32_768,
        }
    }

    /// Tiny config for unit tests and benchmarks (fast init, negligible memory).
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 128,
        }
    }

    /// Materialise the model on `device` with random weights.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3Model<B> {
        let embed_tokens = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);

        let layers = (0..self.num_hidden_layers)
            .map(|_| {
                let attn_cfg = Qwen3AttentionConfig {
                    hidden_size: self.hidden_size,
                    num_heads: self.num_attention_heads,
                    num_kv_heads: self.num_key_value_heads,
                    head_dim: self.head_dim,
                    max_position_embeddings: self.max_position_embeddings,
                    rope_theta: self.rope_theta,
                };
                let mlp_cfg = Qwen3MlpConfig {
                    hidden_size: self.hidden_size,
                    intermediate_size: self.intermediate_size,
                };
                Qwen3DecoderLayer {
                    self_attn: attn_cfg.init(device),
                    mlp: mlp_cfg.init(device),
                    input_layernorm: RmsNormConfig::new(self.hidden_size)
                        .with_epsilon(self.rms_norm_eps)
                        .init(device),
                    post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                        .with_epsilon(self.rms_norm_eps)
                        .init(device),
                }
            })
            .collect();

        let norm = RmsNormConfig::new(self.hidden_size)
            .with_epsilon(self.rms_norm_eps)
            .init(device);

        Qwen3Model {
            embed_tokens,
            layers,
            norm,
        }
    }
}

// ── Modules ──────────────────────────────────────────────────────────

/// Full Qwen3 model (embed → layers → norm).
#[derive(Module, Debug)]
pub struct Qwen3Model<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<Qwen3DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

/// One pre-norm transformer decoder layer.
#[derive(Module, Debug)]
pub struct Qwen3DecoderLayer<B: Backend> {
    pub self_attn: Qwen3Attention<B>,
    pub mlp: Qwen3Mlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen3Model<B> {
    /// Forward pass returning hidden states `[batch, seq_len, hidden_size]`.
    ///
    /// `mask`: optional `[batch, 1, 1, seq_len]` additive attention mask
    /// (`0.0` = real, `-1e9` = padding). Pass `None` for unpadded inputs.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let mut hidden = self.embed_tokens.forward(input_ids);
        for layer in &self.layers {
            hidden = layer.forward(hidden, mask.clone());
        }
        self.norm.forward(hidden)
    }
}

impl<B: Backend> Qwen3DecoderLayer<B> {
    fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        // Pre-norm → attention → residual
        let residual = x.clone();
        let h = self
            .self_attn
            .forward(self.input_layernorm.forward(x), mask);
        let h = h + residual;

        // Post-norm → MLP → residual
        let residual = h.clone();
        let h = self.mlp.forward(self.post_attention_layernorm.forward(h));
        h + residual
    }
}
