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
use candle_nn::{RmsNorm, VarBuilder};

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

pub struct State {
    pub hs: Vec<Tensor>,
    pub prev_xs: Vec<[Tensor; D_CONV]>,
    pub pos: usize,
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
        })
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
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (b_sz, _dim) = xs.dims2()?;
        let li = self.layer_index;
        let mut xs = xs.apply(&self.in_proj)?.chunk(2, D::Minus1)?;
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
        let delta = (delta.exp()? + 1.)?.log()?;
        let a = self.a_log.to_dtype(delta.dtype())?.exp()?.neg()?;
        let d = self.d.to_dtype(delta.dtype())?;

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
        state.hs[li] =
            ((&state.hs[li] * (&delta * &a)?.exp()?)? + &delta * &b * &proj_for_conv_b)?;
        let ss = (state.hs[li]
            .matmul(&c.unsqueeze(D::Minus1)?)?
            .squeeze(D::Minus1)?
            + proj_for_conv.broadcast_mul(&d)?)?;

        let ys = (ss * candle_nn::ops::silu(&proj_for_silu))?;
        ys.apply(&self.out_proj)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualBlock {
    mixer: MambaBlock,
    norm: RmsNorm,
}

impl ResidualBlock {
    pub fn new(layer_index: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm"))?;
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
        let norm_f = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
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
        xs.apply(&self.norm_f)?.apply(&self.lm_head)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
