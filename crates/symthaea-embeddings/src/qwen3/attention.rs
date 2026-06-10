// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Qwen3 Grouped-Query Attention with Rotary Position Embeddings
//!
//! Implements GQA where `num_kv_heads < num_heads` — each KV head serves
//! `num_heads / num_kv_heads` query heads. RoPE is applied to Q and K before
//! the scaled dot-product.

use burn::nn::{Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig};
use burn::prelude::*;

/// GQA + RoPE attention block used in each Qwen3 decoder layer.
#[derive(Module, Debug)]
pub struct Qwen3Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    rotary: RotaryEncoding<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

/// Configuration for [`Qwen3Attention`].
#[derive(Debug, Clone)]
pub struct Qwen3AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
}

impl Qwen3AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3Attention<B> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let no_bias = |d_in, d_out| LinearConfig::new(d_in, d_out).with_bias(false);

        Qwen3Attention {
            q_proj: no_bias(self.hidden_size, q_dim).init(device),
            k_proj: no_bias(self.hidden_size, kv_dim).init(device),
            v_proj: no_bias(self.hidden_size, kv_dim).init(device),
            o_proj: no_bias(q_dim, self.hidden_size).init(device),
            rotary: RotaryEncodingConfig::new(self.max_position_embeddings, self.head_dim)
                .with_theta(self.rope_theta)
                .init(device),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        }
    }
}

impl<B: Backend> Qwen3Attention<B> {
    /// Bidirectional attention forward (no causal mask — embedding use-case).
    ///
    /// `mask`: optional `[batch, 1, 1, seq_len]` additive mask.
    /// Use `0.0` for real tokens and `-1e9` for padding positions.
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        // Q / K / V linear projections (no bias per Qwen3 spec)
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [batch, heads, seq_len, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Rotary position embeddings on Q and K
        let q = self.rotary.forward(q);
        let k = self.rotary.forward(k);

        // GQA: expand KV heads to match query heads when num_kv_heads < num_heads
        let (k, v) = if self.num_heads == self.num_kv_heads {
            (k, v)
        } else {
            let groups = self.num_heads / self.num_kv_heads;
            let k = k
                .unsqueeze_dim::<5>(2) // [b, kv, 1, s, d]
                .repeat_dim(2, groups) // [b, kv, g, s, d]
                .reshape([batch, self.num_heads, seq_len, self.head_dim]);
            let v = v.unsqueeze_dim::<5>(2).repeat_dim(2, groups).reshape([
                batch,
                self.num_heads,
                seq_len,
                self.head_dim,
            ]);
            (k, v)
        };

        // Scaled dot-product attention: softmax(QK^T / sqrt(d) + mask) * V
        let inv_scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = q.matmul(k.swap_dims(2, 3)).mul_scalar(inv_scale);
        let attn = if let Some(m) = mask {
            attn + m // broadcasts [batch,1,1,seq] → [batch,heads,seq,seq]
        } else {
            attn
        };
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back to [batch, seq_len, hidden_size] and project
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(out)
    }
}
