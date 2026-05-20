// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vendored Mamba inference model from candle-transformers 0.8.4.
//!
//! We vendor this instead of depending on `candle-transformers` to:
//! 1. Add `forward_embeds()` — forward pass from continuous embeddings, bypassing
//!    token embedding lookup. This enables continuous latent prompting (feeding
//!    projected HDC thought chunks directly into Mamba as soft tokens).
//! 2. Drop the `candle-transformers` dependency (saves compile time + 200+ models).
//!
//! Original source: `candle-transformers-0.8.4/src/models/mamba.rs`
//!
//! See ["Mamba: Linear-Time Sequence Modeling with Selective State Spaces"](https://arxiv.org/abs/2312.00752)

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

/// Manual RMSNorm implementation using primitive tensor ops.
///
/// Replaces `candle_nn::RmsNorm` to avoid a missing CUDA kernel in candle 0.8.
/// The upstream `candle_nn::RmsNorm` uses a fused custom kernel that was not
/// ported to CUDA at the time of vendoring (candle-transformers 0.8.4).
///
/// # CUDA Safety
///
/// All operations used here have individual CUDA kernel implementations:
/// - `sqr()` — element-wise square
/// - `sum_keepdim()` — reduction with dimension preservation
/// - `broadcast_div()` — broadcasted division
/// - `sqrt()` — element-wise square root
/// - `broadcast_mul()` — broadcasted multiplication (weight scaling)
///
/// This makes the vendored RmsNorm fully CUDA-compatible without requiring
/// any custom CUDA kernels beyond what candle-core provides.
#[derive(Clone, Debug)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_sq = xs.sqr()?;
        let variance = (xs_sq.sum_keepdim(D::Minus1)? / xs.dim(D::Minus1)? as f64)?;
        let x_normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.broadcast_mul(&self.weight)
    }
}

// Local linear wrappers replacing candle-transformers' `with_tracing` variants.
fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<candle_nn::Linear> {
    candle_nn::linear(d1, d2, vb)
}
fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<candle_nn::Linear> {
    candle_nn::linear_no_bias(d1, d2, vb)
}

const D_CONV: usize = 4;
const D_STATE: usize = 16;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub d_model: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    pub pad_vocab_size_multiple: usize,
}

impl Config {
    fn vocab_size(&self) -> usize {
        let pad = self.pad_vocab_size_multiple;
        self.vocab_size.div_ceil(pad) * pad
    }

    fn dt_rank(&self) -> usize {
        (self.d_model + 15) / 16
    }

    fn d_inner(&self) -> usize {
        self.d_model * 2
    }
}

/// Per-token modulation signal from the CfC cognitive loop.
///
/// Allows the continuous-time CfC dynamics to steer the LSSM's
/// discretization at a sub-token level. Both CfC and Mamba share the
/// ODE form ḣ = Ah + Bx, so this modulation is mathematically natural.
///
/// When `delta_scale` > 1.0, the SSM takes larger temporal steps (faster
/// dynamics, more responsive to input). When < 1.0, it takes smaller steps
/// (more conservative, relies more on memory). `b_scale` modulates how
/// strongly new input enters the hidden state.
#[derive(Clone, Debug)]
pub struct CfcModulation {
    /// Multiplicative scale on Δ (step-size) after softplus.
    /// 1.0 = no effect. Range: [0.1, 10.0] clamped internally.
    pub delta_scale: f32,
    /// Multiplicative scale on B (input matrix).
    /// 1.0 = no effect. Range: [0.1, 10.0] clamped internally.
    pub b_scale: f32,
}

impl Default for CfcModulation {
    fn default() -> Self {
        Self {
            delta_scale: 1.0,
            b_scale: 1.0,
        }
    }
}

pub struct State {
    pub hs: Vec<Tensor>,
    pub prev_xs: Vec<[Tensor; D_CONV]>,
    pub pos: usize,
    /// Per-token CfC modulation signal. Applied to all layers.
    /// Reset to None after each forward pass.
    pub cfc_modulation: Option<CfcModulation>,
}

impl Clone for State {
    fn clone(&self) -> Self {
        let hs = self.hs.iter().map(|h| h.clone()).collect();
        let mut prev_xs = Vec::with_capacity(self.prev_xs.len());
        for xs in &self.prev_xs {
            prev_xs.push([xs[0].clone(), xs[1].clone(), xs[2].clone(), xs[3].clone()]);
        }

        Self {
            hs,
            prev_xs,
            pos: self.pos,
            cfc_modulation: self.cfc_modulation.clone(),
        }
    }
}

impl State {
    pub fn new(batch_size: usize, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let mut hs = Vec::with_capacity(cfg.n_layer);
        let mut prev_xs = Vec::with_capacity(cfg.n_layer);
        for _i in 0..cfg.n_layer {
            let h = Tensor::zeros((batch_size, cfg.d_inner(), D_STATE), dtype, device)?;
            let x = Tensor::zeros((batch_size, cfg.d_inner()), dtype, device)?;
            hs.push(h);
            prev_xs.push([x.clone(), x.clone(), x.clone(), x.clone()]);
        }
        Ok(Self {
            hs,
            prev_xs,
            pos: 0,
            cfc_modulation: None,
        })
    }

    /// Detach all hidden state tensors from the autograd graph.
    ///
    /// Used for rotating grad-chunk position (Improvement B): after warming up
    /// state on context chunks, detach to prevent gradient flow through the warmup.
    pub fn detach(&mut self) -> Result<()> {
        for h in &mut self.hs {
            *h = h.detach();
        }
        for xs in &mut self.prev_xs {
            for x in xs.iter_mut() {
                *x = x.detach();
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LORA ADAPTER (Improvement D)
// ═══════════════════════════════════════════════════════════════════════════════

/// Low-Rank Adaptation (LoRA) adapter for Mamba projection layers.
///
/// Adds a low-rank update to the frozen weight: `out += (x @ A^T) @ B^T * (alpha/rank)`.
/// A is initialized from N(0, 1/rank) and B from zeros, so the adapter starts as zero.
#[derive(Clone, Debug)]
pub struct LoraAdapter {
    /// Low-rank matrix A: [rank, in_dim]
    pub a: candle_core::Var,
    /// Low-rank matrix B: [out_dim, rank]
    pub b: candle_core::Var,
    pub rank: usize,
    pub alpha: f32,
}

impl LoraAdapter {
    /// Create a new LoRA adapter.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        device: &Device,
    ) -> Result<Self> {
        // A: random init scaled by 1/sqrt(rank)
        let a_scale = 1.0 / (rank as f64).sqrt();
        let a_data = Tensor::randn(0.0f32, a_scale as f32, (rank, in_dim), device)?;
        let a = candle_core::Var::from_tensor(&a_data)?;

        // B: zero init (so adapter starts as identity)
        let b_data = Tensor::zeros((out_dim, rank), DType::F32, device)?;
        let b = candle_core::Var::from_tensor(&b_data)?;

        Ok(Self { a, b, rank, alpha })
    }

    /// Forward pass: returns the LoRA delta to add to the main layer's output.
    ///
    /// `x` shape: [batch, in_dim]
    /// Returns: [batch, out_dim]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scale = self.alpha / self.rank as f32;
        // x @ A^T → [batch, rank]
        let h = x.matmul(&self.a.as_tensor().t()?)?;
        // h @ B^T → [batch, out_dim]
        let out = h.matmul(&self.b.as_tensor().t()?)?;
        out * scale as f64
    }

    /// Apply gradients from the backward pass to the LoRA parameters.
    ///
    /// IMPORTANT: We `.detach()` before `Var::set()` to break autograd graph edges.
    /// Without detach, `Var::set()` keeps a reference to the computation graph that
    /// produced `updated`, causing RSS to grow unboundedly across training steps
    /// (observed: 1.2GB → 9.5GB in one epoch with 48 LoRA adapters).
    pub fn apply_grads(&self, grads: &candle_core::backprop::GradStore, lr: f32) -> Result<()> {
        if let Some(grad_a) = grads.get(self.a.as_tensor()) {
            let updated = (self.a.as_tensor() - (grad_a * lr as f64)?)?;
            self.a.set(&updated.detach())?;
        }
        if let Some(grad_b) = grads.get(self.b.as_tensor()) {
            let updated = (self.b.as_tensor() - (grad_b * lr as f64)?)?;
            self.b.set(&updated.detach())?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct MambaBlock {
    in_proj: candle_nn::Linear,
    conv1d_bias: Tensor,
    conv1d_weights: [Tensor; D_CONV],
    x_proj: candle_nn::Linear,
    dt_proj: candle_nn::Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: candle_nn::Linear,
    dt_rank: usize,
    layer_index: usize,
    d_inner: usize,
    // LoRA adapters (Improvement D)
    pub lora_in_proj: Option<LoraAdapter>,
    pub lora_out_proj: Option<LoraAdapter>,
}

impl MambaBlock {
    pub fn new(layer_index: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let dt_rank = cfg.dt_rank();
        let in_proj = linear_no_bias(cfg.d_model, d_inner * 2, vb.pp("in_proj"))?;
        let x_proj = linear_no_bias(d_inner, dt_rank + D_STATE * 2, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;
        let a_log = vb.get((d_inner, D_STATE), "A_log")?;
        let d = vb.get(d_inner, "D")?;
        let out_proj = linear_no_bias(d_inner, cfg.d_model, vb.pp("out_proj"))?;
        let conv1d_bias = vb.get(d_inner, "conv1d.bias")?;
        let conv1d_weight = vb.get((d_inner, 1, D_CONV), "conv1d.weight")?;
        let conv1d_weights = [
            conv1d_weight.i((.., 0, 0))?,
            conv1d_weight.i((.., 0, 1))?,
            conv1d_weight.i((.., 0, 2))?,
            conv1d_weight.i((.., 0, 3))?,
        ];
        Ok(Self {
            in_proj,
            conv1d_bias,
            conv1d_weights,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            dt_rank,
            layer_index,
            d_inner,
            lora_in_proj: None,
            lora_out_proj: None,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (b_sz, _dim) = xs.dims2()?;
        let li = self.layer_index;
        let mut in_proj_out = xs.apply(&self.in_proj)?;
        // Add LoRA delta for in_proj (Improvement D)
        if let Some(ref lora) = self.lora_in_proj {
            in_proj_out = (in_proj_out + lora.forward(xs)?)?;
        }
        let mut xs = in_proj_out.chunk(2, D::Minus1)?;
        let proj_for_silu = xs.remove(1);
        state.prev_xs[li][state.pos % D_CONV] = xs.remove(0);
        let mut proj_for_conv = self.conv1d_bias.broadcast_as((b_sz, self.d_inner))?;
        for d_c in 0..D_CONV {
            proj_for_conv = (proj_for_conv
                + self.conv1d_weights[d_c]
                    .broadcast_mul(&state.prev_xs[li][(d_c + 1 + state.pos) % D_CONV])?)?;
        }
        let proj_for_conv = candle_nn::ops::silu(&proj_for_conv)?;
        // SSM + Selection (Algorithm 3.2, page 6, https://arxiv.org/pdf/2312.00752.pdf)

        let x_proj = self.x_proj.forward(&proj_for_conv)?;
        let delta = x_proj.narrow(D::Minus1, 0, self.dt_rank)?.contiguous()?;
        let b = x_proj.narrow(D::Minus1, self.dt_rank, D_STATE)?;
        let c = x_proj.narrow(D::Minus1, self.dt_rank + D_STATE, D_STATE)?;

        let delta = delta.apply(&self.dt_proj)?;
        // softplus
        let mut delta = (delta.exp()? + 1.)?.log()?;
        let a = self.a_log.to_dtype(delta.dtype())?.exp()?.neg()?;
        let d = self.d.to_dtype(delta.dtype())?;

        // Apply CfC modulation: scale Δ and B based on cognitive state
        let mut b = b;
        if let Some(ref modulation) = state.cfc_modulation {
            let ds = modulation.delta_scale.clamp(0.1, 10.0) as f64;
            if (ds - 1.0).abs() > 1e-6 {
                delta = (delta * ds)?;
            }
            let bs = modulation.b_scale.clamp(0.1, 10.0) as f64;
            if (bs - 1.0).abs() > 1e-6 {
                b = (b * bs)?;
            }
        }

        // Selective scan: h_t = Ab h_{t-1} + Bb x_t
        let delta = delta
            .unsqueeze(D::Minus1)?
            .broadcast_as((b_sz, self.d_inner, D_STATE))?;
        let a = a.broadcast_as((b_sz, self.d_inner, D_STATE))?;
        let b = b.broadcast_as((b_sz, self.d_inner, D_STATE))?;
        let proj_for_conv_b =
            proj_for_conv
                .unsqueeze(D::Minus1)?
                .broadcast_as((b_sz, self.d_inner, D_STATE))?;
        state.hs[li] = ((&state.hs[li] * (&delta * &a)?.exp()?)? + &delta * &b * &proj_for_conv_b)?;
        let ss = (state.hs[li]
            .matmul(&c.unsqueeze(D::Minus1)?)?
            .squeeze(D::Minus1)?
            + proj_for_conv.broadcast_mul(&d)?)?;

        let ys = (ss * candle_nn::ops::silu(&proj_for_silu))?;
        let mut out = ys.apply(&self.out_proj)?;
        // Add LoRA delta for out_proj (Improvement D)
        if let Some(ref lora) = self.lora_out_proj {
            out = (out + lora.forward(&ys)?)?;
        }
        Ok(out)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualBlock {
    mixer: MambaBlock,
    norm: RmsNorm,
}

impl ResidualBlock {
    pub fn new(layer_index: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = RmsNorm::new(cfg.d_model, 1e-5, vb.pp("norm"))?;
        let mixer = MambaBlock::new(layer_index, cfg, vb.pp("mixer"))?;
        Ok(Self { mixer, norm })
    }

    /// Forward pass through this residual block (residual connection around norm + mixer).
    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        self.mixer.forward(&xs.apply(&self.norm)?, state)? + xs
    }
}

// https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/model.py#L56
#[derive(Clone, Debug)]
pub struct Model {
    embedding: candle_nn::Embedding,
    layers: Vec<ResidualBlock>,
    norm_f: RmsNorm,
    lm_head: candle_nn::Linear,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size(), cfg.d_model, vb.pp("embedding"))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.n_layer {
            let layer = ResidualBlock::new(layer_idx, cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm_f = RmsNorm::new(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
        let lm_head = candle_nn::Linear::new(embedding.embeddings().clone(), None);
        Ok(Self {
            embedding,
            layers,
            norm_f,
            lm_head,
            dtype: vb.dtype(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, state: &mut State) -> Result<Tensor> {
        let _b_size = input_ids.dims1()?;
        let mut xs = self.embedding.forward(input_ids)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, state)?
        }
        state.pos += 1;
        state.cfc_modulation = None; // Consume per-token modulation
        xs.apply(&self.norm_f)?.apply(&self.lm_head)
    }

    /// Forward pass from raw embedding vectors, bypassing token embedding lookup.
    ///
    /// Used for continuous latent prompting: feed projected HDC thought chunks
    /// directly into Mamba's SSM layers as soft tokens. Each call processes one
    /// "soft token" (a continuous embedding vector) through all layers, accumulating
    /// SSM state exactly as Mamba was trained to do.
    ///
    /// This is the same mechanism multimodal models (LLaVA, Flamingo) use to feed
    /// image patch embeddings into language model layers.
    pub fn forward_embeds(&self, embeds: &Tensor, state: &mut State) -> Result<Tensor> {
        let mut xs = embeds.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, state)?;
        }
        state.pos += 1;
        state.cfc_modulation = None; // Consume per-token modulation
        xs.apply(&self.norm_f)?.apply(&self.lm_head)
    }

    /// Forward pass from raw embeddings, returning only the hidden state (no lm_head).
    ///
    /// ~2× faster than `forward_embeds()` since it skips the final RMSNorm + lm_head
    /// projection. Used during context injection where we only need to accumulate
    /// SSM state, not produce logits.
    pub fn forward_embeds_no_head(&self, embeds: &Tensor, state: &mut State) -> Result<Tensor> {
        let mut xs = embeds.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, state)?;
        }
        state.pos += 1;
        state.cfc_modulation = None; // Consume per-token modulation
        Ok(xs)
    }

    /// Inject a pre-stacked sequence of embeddings through all layers.
    ///
    /// Takes a `(seq_len, d_model)` tensor and processes each row sequentially
    /// through the SSM layers, accumulating state. Avoids per-chunk tensor
    /// allocation overhead compared to calling `forward_embeds` in a loop.
    ///
    /// Returns the final hidden state after all embeddings have been processed.
    pub fn inject_sequence(&self, sequence: &Tensor, state: &mut State) -> Result<()> {
        let seq_len = sequence.dim(0)?;
        for t in 0..seq_len {
            let embed = sequence.i(t)?.unsqueeze(0)?; // (1, d_model)
            let mut xs = embed;
            for layer in self.layers.iter() {
                xs = layer.forward(&xs, state)?;
            }
            state.pos += 1;
        }
        Ok(())
    }

    /// Pre-fill conv1d history for all layers with a summary embedding.
    ///
    /// Takes a `(1, d_model)` summary tensor and runs it through each layer's
    /// `in_proj` to produce `(1, d_inner)` tensors, then copies this into all
    /// D_CONV history slots. This gives the first few tokens meaningful conv
    /// context instead of zeros.
    ///
    /// Should be called after `State::new()` but before `inject_sequence()`.
    pub fn warmstart_conv_history(&self, summary: &Tensor, state: &mut State) -> Result<()> {
        for (li, layer) in self.layers.iter().enumerate() {
            // Run summary through in_proj to get d_inner*2, take first half (x path)
            let proj = layer.mixer.in_proj.forward(summary)?;
            let chunks = proj.chunk(2, D::Minus1)?;
            let x_proj = &chunks[0]; // (1, d_inner)
                                     // Fill all D_CONV history slots with this projection
            for slot in 0..D_CONV {
                state.prev_xs[li][slot] = x_proj.clone();
            }
        }
        Ok(())
    }

    /// Access the raw embedding table `[vocab_size, d_model]`.
    ///
    /// Used by manifold moment matching to compute per-dimension statistics
    /// of Mamba's embedding space for whitening adapter initialization.
    pub fn embedding_table(&self) -> &Tensor {
        self.embedding.embeddings()
    }

    pub fn device(&self) -> &Device {
        self.embedding.embeddings().device()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Enable LoRA adapters on all MambaBlock in_proj and out_proj layers.
    ///
    /// `rank`: LoRA rank (typically 4-16)
    /// `alpha`: scaling factor (typically 8.0)
    pub fn enable_lora(&mut self, rank: usize, alpha: f32, device: &Device) -> Result<()> {
        if rank == 0 {
            return Ok(());
        }
        for layer in &mut self.layers {
            let d_inner = layer.mixer.d_inner;
            let d_model = d_inner / 2; // d_inner = d_model * 2
                                       // in_proj: [d_model] → [d_inner * 2]
            layer.mixer.lora_in_proj =
                Some(LoraAdapter::new(d_model, d_inner * 2, rank, alpha, device)?);
            // out_proj: [d_inner] → [d_model]
            layer.mixer.lora_out_proj =
                Some(LoraAdapter::new(d_inner, d_model, rank, alpha, device)?);
        }
        Ok(())
    }

    /// Apply LoRA gradients from a GradStore to all LoRA parameters.
    pub fn apply_lora_grads(
        &self,
        grads: &candle_core::backprop::GradStore,
        lr: f32,
    ) -> Result<()> {
        for layer in &self.layers {
            if let Some(ref lora) = layer.mixer.lora_in_proj {
                lora.apply_grads(grads, lr)?;
            }
            if let Some(ref lora) = layer.mixer.lora_out_proj {
                lora.apply_grads(grads, lr)?;
            }
        }
        Ok(())
    }

    /// Whether LoRA is enabled.
    pub fn has_lora(&self) -> bool {
        self.layers
            .first()
            .map_or(false, |l| l.mixer.lora_in_proj.is_some())
    }
}
