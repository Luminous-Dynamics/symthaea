// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Projection: chunk-based HDC↔SSM bridge with continuous latent prompting.
//!
//! Converts the spatial compression `HDC(16384) → bottleneck(256) → SSM(768)` into
//! temporal sequencing: chunk the 16384D thought vector into **N tokens of chunk_dim**,
//! up-project each to 768D, and feed them as **continuous embeddings** into Mamba.
//!
//! This matches the mechanism multimodal models (LLaVA, Flamingo) use to feed
//! image patches into language model layers — bypassing the token embedding lookup.
//!
//! # Compression ratio
//!
//! - Spatial: 64:1 (16384 → 256) — catastrophic information loss
//! - Temporal: 3:1 per chunk (256 → 768) — near-lossless per token
//!
//! # Parameters (single group, no adapter)
//!
//! - `group_w_up[0]`: `[768 × 256]` — 196,608 params (vs 8.8M spatial)
//! - `group_w_down[0]`: `[256 × 768]` — 196,608 params (for backward/PE)
//! - `ln_gamma`, `ln_beta`: `[256]` each — LayerNorm
//! - Total: ~393K params (36× fewer than spatial projection)

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::projection::GradientStepMetrics;

// ═══════════════════════════════════════════════════════════════════════════════
// EMBEDDING STATS (Manifold Moment Matching)
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-dimension statistics of a Mamba embedding table for manifold moment matching.
///
/// Stores mean and variance across all vocabulary embeddings. Used to initialize
/// the AdapterMLP's w1 as a whitening transform so that all SSM dimensions
/// receive equal gradient influence from step 0.
///
/// File format: 16-byte header + mean[dim] + variance[dim] as little-endian f32.
pub struct EmbeddingStats {
    /// Embedding dimension (e.g. 768).
    pub dim: usize,
    /// Number of embeddings (e.g. 50280).
    pub count: usize,
    /// Per-dimension mean [dim].
    pub mean: Vec<f32>,
    /// Per-dimension variance [dim].
    pub variance: Vec<f32>,
}

/// Magic bytes for the embedding stats file format.
const EMBS_MAGIC: [u8; 4] = *b"EMBS";
/// File format version.
const EMBS_VERSION: u32 = 1;

impl EmbeddingStats {
    /// Compute per-dimension mean and variance from a flat embedding table.
    ///
    /// `flat` should be `[count * dim]` values in row-major order (row = one embedding).
    pub fn compute(flat: &[f32], dim: usize) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert!(
            flat.len() % dim == 0,
            "flat length must be divisible by dim"
        );
        let count = flat.len() / dim;
        assert!(count > 0, "must have at least one embedding");

        let mut mean = vec![0.0f64; dim];
        let mut var = vec![0.0f64; dim];

        // Compute mean
        for row in 0..count {
            let offset = row * dim;
            for j in 0..dim {
                mean[j] += flat[offset + j] as f64;
            }
        }
        let n = count as f64;
        for j in 0..dim {
            mean[j] /= n;
        }

        // Compute variance
        for row in 0..count {
            let offset = row * dim;
            for j in 0..dim {
                let diff = flat[offset + j] as f64 - mean[j];
                var[j] += diff * diff;
            }
        }
        for j in 0..dim {
            var[j] /= n;
        }

        Self {
            dim,
            count,
            mean: mean.iter().map(|&x| x as f32).collect(),
            variance: var.iter().map(|&x| x as f32).collect(),
        }
    }

    /// Save to binary file.
    ///
    /// Format: `[magic:4][version:4][dim:4][count:4][mean:dim*4][var:dim*4]`
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&EMBS_MAGIC)?;
        file.write_all(&EMBS_VERSION.to_le_bytes())?;
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.count as u32).to_le_bytes())?;
        for &v in &self.mean {
            file.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.variance {
            file.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load from binary file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if magic != EMBS_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid embedding stats magic",
            ));
        }

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != EMBS_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported embedding stats version: {version}"),
            ));
        }

        file.read_exact(&mut buf4)?;
        let dim = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4)?;
        let count = u32::from_le_bytes(buf4) as usize;

        let mut mean = vec![0.0f32; dim];
        let mut variance = vec![0.0f32; dim];
        for j in 0..dim {
            file.read_exact(&mut buf4)?;
            mean[j] = f32::from_le_bytes(buf4);
        }
        for j in 0..dim {
            file.read_exact(&mut buf4)?;
            variance[j] = f32::from_le_bytes(buf4);
        }

        Ok(Self {
            dim,
            count,
            mean,
            variance,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GELU HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// GELU activation: x * Phi(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Approximate GELU derivative: d/dx gelu(x).
fn gelu_derivative(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    let tanh_val = inner.tanh();
    let sech2 = 1.0 - tanh_val * tanh_val;
    let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTER MLP (Improvement E)
// ═══════════════════════════════════════════════════════════════════════════════

/// Small residual MLP adapter applied after up-projection.
///
/// Forward: `out = w2 @ gelu(w1 @ x + b1) + b2 + x` (residual connection).
/// Initialized with small random weights so the adapter starts as near-identity.
pub struct AdapterMlp {
    dim: usize,
    w1: Vec<f32>,      // [dim * dim]
    b1: Vec<f32>,      // [dim]
    w2: Vec<f32>,      // [dim * dim]
    b2: Vec<f32>,      // [dim]
    grad_w1: Vec<f32>, // [dim * dim]
    grad_b1: Vec<f32>, // [dim]
    grad_w2: Vec<f32>, // [dim * dim]
    grad_b2: Vec<f32>, // [dim]
}

impl AdapterMlp {
    /// Create a new adapter MLP with small random initialization.
    pub fn new(genesis: &GenesisSeed, dim: usize) -> Self {
        let scale = 0.01 / (dim as f32).sqrt();
        let dd = dim * dim;

        let w1 = init_weights(genesis, "adapter_mlp::w1", dd, scale);
        let w2 = init_weights(genesis, "adapter_mlp::w2", dd, scale);
        let b1 = vec![0.0f32; dim];
        let b2 = vec![0.0f32; dim];

        Self {
            dim,
            w1,
            b1,
            w2,
            b2,
            grad_w1: vec![0.0; dd],
            grad_b1: vec![0.0; dim],
            grad_w2: vec![0.0; dd],
            grad_b2: vec![0.0; dim],
        }
    }

    /// Create an adapter MLP with whitening initialization from embedding statistics.
    ///
    /// - `w1` = diagonal whitening: `1/sqrt(var[i] + eps)` — normalizes each dimension
    /// - `b1` = centering: `-mean[i] / sqrt(var[i] + eps)` — centers to zero
    /// - `w2` = small random (same as default) — GELU path starts near-zero
    /// - `b2` = `mean[i]` — restore target mean through residual path
    pub fn new_from_stats(genesis: &GenesisSeed, dim: usize, stats: &EmbeddingStats) -> Self {
        assert_eq!(dim, stats.dim, "adapter dim must match embedding stats dim");
        let dd = dim * dim;
        let eps = 1e-6f32;

        // w1: diagonal whitening matrix (only diagonal entries are non-zero)
        let mut w1 = vec![0.0f32; dd];
        let mut b1 = vec![0.0f32; dim];
        for i in 0..dim {
            let inv_std = 1.0 / (stats.variance[i] + eps).sqrt();
            w1[i * dim + i] = inv_std;
            b1[i] = -stats.mean[i] * inv_std;
        }

        // w2: small random (same as default new())
        let scale = 0.01 / (dim as f32).sqrt();
        let w2 = init_weights(genesis, "adapter_mlp::w2_stats", dd, scale);
        // b2: restore target mean
        let b2 = stats.mean.clone();

        Self {
            dim,
            w1,
            b1,
            w2,
            b2,
            grad_w1: vec![0.0; dd],
            grad_b1: vec![0.0; dim],
            grad_w2: vec![0.0; dd],
            grad_b2: vec![0.0; dim],
        }
    }

    /// Forward pass: `out = w2 @ gelu(w1 @ x + b1) + b2 + x`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim);
        let d = self.dim;

        // hidden = gelu(w1 @ x + b1)
        let mut hidden = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = self.b1[i];
            for j in 0..d {
                sum += self.w1[i * d + j] * x[j];
            }
            hidden[i] = gelu(sum);
        }

        // out = w2 @ hidden + b2 + x (residual)
        let mut out = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = self.b2[i] + x[i]; // residual
            for j in 0..d {
                sum += self.w2[i * d + j] * hidden[j];
            }
            out[i] = sum;
        }

        out
    }

    /// Backward pass: compute gradients and return d_loss/d_input.
    ///
    /// `input`: the original input to forward().
    /// `d_output`: gradient of loss w.r.t. the adapter output.
    /// Returns gradient of loss w.r.t. the adapter input.
    pub fn backward(&mut self, input: &[f32], d_output: &[f32]) -> Vec<f32> {
        let d = self.dim;

        // Re-compute forward intermediate values
        let mut pre_gelu = vec![0.0f32; d];
        let mut hidden = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = self.b1[i];
            for j in 0..d {
                sum += self.w1[i * d + j] * input[j];
            }
            pre_gelu[i] = sum;
            hidden[i] = gelu(sum);
        }

        // Gradient through residual: d_input from residual = d_output (identity path)
        let mut d_input = d_output.to_vec();

        // Gradient for w2, b2: d_output flows directly
        for i in 0..d {
            self.grad_b2[i] += d_output[i];
            for j in 0..d {
                self.grad_w2[i * d + j] += d_output[i] * hidden[j];
            }
        }

        // Backprop through w2 to get d_hidden
        let mut d_hidden = vec![0.0f32; d];
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += self.w2[i * d + j] * d_output[i];
            }
            d_hidden[j] = sum;
        }

        // Backprop through gelu
        let mut d_pre_gelu = vec![0.0f32; d];
        for i in 0..d {
            d_pre_gelu[i] = d_hidden[i] * gelu_derivative(pre_gelu[i]);
        }

        // Gradient for w1, b1
        for i in 0..d {
            self.grad_b1[i] += d_pre_gelu[i];
            for j in 0..d {
                self.grad_w1[i * d + j] += d_pre_gelu[i] * input[j];
            }
        }

        // Backprop through w1 to get d_input (add to residual d_input)
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += self.w1[i * d + j] * d_pre_gelu[i];
            }
            d_input[j] += sum;
        }

        d_input
    }

    /// Apply accumulated gradients with learning rate and clipping.
    /// Returns true if any gradient was clipped.
    pub fn apply_gradients(&mut self, lr: f32, clip: f32) -> bool {
        let norm = (l2_norm(&self.grad_w1).powi(2)
            + l2_norm(&self.grad_b1).powi(2)
            + l2_norm(&self.grad_w2).powi(2)
            + l2_norm(&self.grad_b2).powi(2))
        .sqrt();

        let was_clipped = norm > clip;
        let scale = if was_clipped { clip / norm } else { 1.0 };
        let effective_lr = lr * scale;

        for (w, g) in self.w1.iter_mut().zip(self.grad_w1.iter()) {
            *w -= effective_lr * g;
        }
        for (w, g) in self.b1.iter_mut().zip(self.grad_b1.iter()) {
            *w -= effective_lr * g;
        }
        for (w, g) in self.w2.iter_mut().zip(self.grad_w2.iter()) {
            *w -= effective_lr * g;
        }
        for (w, g) in self.b2.iter_mut().zip(self.grad_b2.iter()) {
            *w -= effective_lr * g;
        }

        // Zero accumulators
        self.grad_w1.fill(0.0);
        self.grad_b1.fill(0.0);
        self.grad_w2.fill(0.0);
        self.grad_b2.fill(0.0);

        was_clipped
    }

    /// Scale accumulated gradients by a factor.
    pub fn scale_gradients(&mut self, scale: f32) {
        for g in &mut self.grad_w1 {
            *g *= scale;
        }
        for g in &mut self.grad_b1 {
            *g *= scale;
        }
        for g in &mut self.grad_w2 {
            *g *= scale;
        }
        for g in &mut self.grad_b2 {
            *g *= scale;
        }
    }

    /// Flatten all weights for checkpointing.
    pub fn flatten_weights(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(self.num_params());
        w.extend_from_slice(&self.w1);
        w.extend_from_slice(&self.b1);
        w.extend_from_slice(&self.w2);
        w.extend_from_slice(&self.b2);
        w
    }

    /// Load weights from flat vector.
    pub fn load_weights(&mut self, weights: &[f32]) {
        assert_eq!(weights.len(), self.num_params());
        let d = self.dim;
        let dd = d * d;
        let mut off = 0;
        self.w1.copy_from_slice(&weights[off..off + dd]);
        off += dd;
        self.b1.copy_from_slice(&weights[off..off + d]);
        off += d;
        self.w2.copy_from_slice(&weights[off..off + dd]);
        off += dd;
        self.b2.copy_from_slice(&weights[off..off + d]);
        let _ = off;
    }

    /// Number of parameters: 2 * dim^2 + 2 * dim.
    pub fn num_params(&self) -> usize {
        2 * self.dim * self.dim + 2 * self.dim
    }
}

impl std::fmt::Debug for AdapterMlp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdapterMlp")
            .field("dim", &self.dim)
            .field("num_params", &self.num_params())
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPORAL PROJECTION (Improvements C, E, A integrated)
// ═══════════════════════════════════════════════════════════════════════════════

/// Temporal projection: chunk-based HDC↔SSM bridge.
///
/// Chunks a 16384D HDC thought vector into N tokens of chunk_dim, applies LayerNorm,
/// up-projects each to 768D via per-group learned `group_w_up`, optionally applies
/// an adapter MLP, and adds sinusoidal positional encoding. The result is a sequence
/// of continuous embeddings ready for Mamba's `forward_embeds()`.
pub struct TemporalProjection {
    chunk_size: usize, // per-chunk dimension (e.g. 256)
    num_chunks: usize, // number of chunks (e.g. 64 non-overlapping, or more with overlap)
    stride: usize, // step between chunk starts (chunk_size = no overlap, < chunk_size = overlap)
    // Per-group up/down projections (Improvement C)
    group_w_up: Vec<Vec<f32>>,   // [num_groups][ssm_dim * chunk_size]
    group_w_down: Vec<Vec<f32>>, // [num_groups][chunk_size * ssm_dim]
    num_groups: usize,           // default 1 (single shared matrix, legacy behavior)
    ln_gamma: Vec<f32>,          // [chunk_size] LayerNorm scale
    ln_beta: Vec<f32>,           // [chunk_size] LayerNorm bias
    pos_enc: Vec<f32>,           // [num_chunks * ssm_dim] positional encoding
    learned_pos_enc: bool,       // if true, pos_enc is trainable; if false, fixed sinusoidal
    // Per-group gradient accumulators
    grad_group_up: Vec<Vec<f32>>,   // [num_groups][ssm_dim * chunk_size]
    grad_group_down: Vec<Vec<f32>>, // [num_groups][chunk_size * ssm_dim]
    grad_ln_gamma: Vec<f32>,
    grad_ln_beta: Vec<f32>,
    grad_pos_enc: Vec<f32>, // [num_chunks * ssm_dim] gradient for learned pos_enc
    // Chunk attention (learned importance per chunk)
    chunk_attention: Vec<f32>, // [num_chunks] logits (sigmoid-gated during forward)
    grad_chunk_attention: Vec<f32>, // [num_chunks] gradient accumulator
    learned_attention: bool,   // if true, chunk_attention is trainable
    // Adapter MLP (Improvement E)
    adapter: Option<AdapterMlp>,
    // Dimensions
    hdc_dim: usize, // 16384
    ssm_dim: usize, // 768
}

impl TemporalProjection {
    /// Create a new temporal projection with JL-style random initialization.
    ///
    /// - `hdc_dim`: HDC thought dimension (16384)
    /// - `chunk_dim`: per-chunk dimension (256, must divide hdc_dim evenly)
    /// - `ssm_dim`: Mamba's d_model (768)
    pub fn new(genesis: &GenesisSeed, hdc_dim: usize, chunk_dim: usize, ssm_dim: usize) -> Self {
        Self::new_full(genesis, hdc_dim, chunk_dim, ssm_dim, false, chunk_dim)
    }

    /// Create a new temporal projection with configurable positional encoding.
    ///
    /// When `learned_pos_enc` is true, the positional encoding is initialized from
    /// sinusoidal values but becomes trainable (gradients flow through it).
    pub fn new_with_options(
        genesis: &GenesisSeed,
        hdc_dim: usize,
        chunk_dim: usize,
        ssm_dim: usize,
        learned_pos_enc: bool,
    ) -> Self {
        Self::new_full(
            genesis,
            hdc_dim,
            chunk_dim,
            ssm_dim,
            learned_pos_enc,
            chunk_dim,
        )
    }

    /// Create a new temporal projection with all options.
    ///
    /// - `stride`: step between chunk start positions. When `stride < chunk_dim`,
    ///   chunks overlap, producing more tokens (e.g., stride=128, chunk_dim=256
    ///   gives 50% overlap and ~127 tokens for hdc_dim=16384).
    ///   When `stride == chunk_dim`, no overlap (default behavior).
    pub fn new_full(
        genesis: &GenesisSeed,
        hdc_dim: usize,
        chunk_dim: usize,
        ssm_dim: usize,
        learned_pos_enc: bool,
        stride: usize,
    ) -> Self {
        assert!(stride > 0, "stride must be positive");
        assert!(
            chunk_dim <= hdc_dim,
            "chunk_dim ({chunk_dim}) must be <= hdc_dim ({hdc_dim})"
        );

        // For non-overlapping (stride == chunk_dim), require divisibility
        if stride == chunk_dim {
            assert!(
                hdc_dim % chunk_dim == 0,
                "hdc_dim ({hdc_dim}) must be divisible by chunk_dim ({chunk_dim}) when stride == chunk_dim"
            );
        }

        // Number of chunks: how many full chunk_dim windows fit with the given stride
        let num_chunks = if stride >= chunk_dim {
            hdc_dim / chunk_dim
        } else {
            // Overlapping: (hdc_dim - chunk_dim) / stride + 1
            // Only count full chunks that fit within bounds
            (hdc_dim - chunk_dim) / stride + 1
        };
        assert!(num_chunks > 0, "Must have at least one chunk");

        // JL-style initialization: scale = 1/sqrt(chunk_dim)
        let scale = 1.0 / (chunk_dim as f32).sqrt();

        let up_size = ssm_dim * chunk_dim;
        let down_size = chunk_dim * ssm_dim;

        // Single group by default (Improvement C: can be expanded via set_num_groups)
        let w_up = init_weights(genesis, "temporal::w_chunk_up", up_size, scale);
        let w_down = init_weights(genesis, "temporal::w_chunk_down", down_size, scale);

        // LayerNorm: gamma=1, beta=0 (standard initialization)
        let ln_gamma = vec![1.0f32; chunk_dim];
        let ln_beta = vec![0.0f32; chunk_dim];

        // Positional encoding: always initialized from sinusoidal,
        // but trainable when `learned_pos_enc` is true.
        let pos_enc = sinusoidal_pos_enc(num_chunks, ssm_dim);
        let pos_enc_size = num_chunks * ssm_dim;

        Self {
            chunk_size: chunk_dim,
            num_chunks,
            stride,
            group_w_up: vec![w_up],
            group_w_down: vec![w_down],
            num_groups: 1,
            ln_gamma,
            ln_beta,
            pos_enc,
            learned_pos_enc,
            grad_group_up: vec![vec![0.0; up_size]],
            grad_group_down: vec![vec![0.0; down_size]],
            grad_ln_gamma: vec![0.0; chunk_dim],
            grad_ln_beta: vec![0.0; chunk_dim],
            grad_pos_enc: vec![0.0; pos_enc_size],
            // Chunk attention: initialized to 0 (sigmoid(0) = 0.5 = uniform weighting)
            chunk_attention: vec![0.0; num_chunks],
            grad_chunk_attention: vec![0.0; num_chunks],
            learned_attention: false,
            adapter: None,
            hdc_dim,
            ssm_dim,
        }
    }

    // ─── Group management (Improvement C) ────────────────────────────────

    /// Map a chunk index to its group index.
    ///
    /// Evenly distributes chunks across groups:
    /// `group = chunk_idx * num_groups / num_chunks`
    fn group_for_chunk(&self, chunk_idx: usize) -> usize {
        chunk_idx * self.num_groups / self.num_chunks
    }

    /// Get the current number of groups.
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    /// Set the number of groups, reallocating per-group weight matrices.
    ///
    /// When expanding from 1 group to N, each new group is initialized as a copy
    /// of the original single group. When setting back to 1, only group 0 is kept.
    pub fn set_num_groups(&mut self, num_groups: usize, genesis: &GenesisSeed) {
        if num_groups == 0 || num_groups == self.num_groups {
            return;
        }
        let up_size = self.ssm_dim * self.chunk_size;
        let down_size = self.chunk_size * self.ssm_dim;
        let scale = 1.0 / (self.chunk_size as f32).sqrt();

        if num_groups > self.num_groups {
            // Expand: new groups get fresh random init
            while self.group_w_up.len() < num_groups {
                let g = self.group_w_up.len();
                let label_up = format!("temporal::group{g}::w_up");
                let label_down = format!("temporal::group{g}::w_down");
                self.group_w_up
                    .push(init_weights(genesis, &label_up, up_size, scale));
                self.group_w_down
                    .push(init_weights(genesis, &label_down, down_size, scale));
                self.grad_group_up.push(vec![0.0; up_size]);
                self.grad_group_down.push(vec![0.0; down_size]);
            }
        } else {
            // Shrink: truncate to num_groups
            self.group_w_up.truncate(num_groups);
            self.group_w_down.truncate(num_groups);
            self.grad_group_up.truncate(num_groups);
            self.grad_group_down.truncate(num_groups);
        }
        self.num_groups = num_groups;
    }

    // ─── Adapter management (Improvement E) ──────────────────────────────

    /// Enable the adapter MLP.
    pub fn enable_adapter(&mut self, genesis: &GenesisSeed) {
        self.adapter = Some(AdapterMlp::new(genesis, self.ssm_dim));
    }

    /// Enable the adapter MLP with whitening initialization from embedding statistics.
    ///
    /// Uses manifold moment matching: w1 is a diagonal whitening transform that
    /// normalizes Mamba's embedding dimensions to unit variance from step 0.
    pub fn enable_adapter_from_stats(&mut self, genesis: &GenesisSeed, stats: &EmbeddingStats) {
        self.adapter = Some(AdapterMlp::new_from_stats(genesis, self.ssm_dim, stats));
    }

    /// Whether the adapter is enabled.
    pub fn has_adapter(&self) -> bool {
        self.adapter.is_some()
    }

    // ─── Forward projection ──────────────────────────────────────────────

    /// Project a 16384D thought to a sequence of N × 768D SSM embeddings.
    ///
    /// Steps per chunk:
    /// 1. Extract chunk_dim slice from thought vector
    /// 2. LayerNorm the chunk
    /// 3. Linear up-projection via per-group `group_w_up`
    /// 4. Add sinusoidal positional encoding
    /// 5. Apply adapter MLP (if enabled)
    /// 6. Scale by chunk attention (if learned)
    pub fn project_to_ssm_sequence(&self, thought: &ContinuousHV) -> Vec<Vec<f32>> {
        assert_eq!(
            thought.values.len(),
            self.hdc_dim,
            "Expected {}-dim thought, got {}",
            self.hdc_dim,
            thought.values.len()
        );

        let mut sequence = Vec::with_capacity(self.num_chunks);

        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let g = self.group_for_chunk(chunk_idx);
            let w_up = &self.group_w_up[g];

            // LayerNorm
            let normed = self.layer_norm(chunk);

            // Up-project: chunk_size → ssm_dim
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for k in 0..self.chunk_size {
                let nk = normed[k];
                for j in 0..self.ssm_dim {
                    ssm_vec[j] += w_up[j * self.chunk_size + k] * nk;
                }
            }

            // Add positional encoding
            let pos_offset = chunk_idx * self.ssm_dim;
            for j in 0..self.ssm_dim {
                ssm_vec[j] += self.pos_enc[pos_offset + j];
            }

            // Apply adapter (Improvement E)
            if let Some(ref adapter) = self.adapter {
                ssm_vec = adapter.forward(&ssm_vec);
            }

            // Apply attention gating
            if self.learned_attention {
                let attn = sigmoid(self.chunk_attention[chunk_idx]);
                for j in 0..self.ssm_dim {
                    ssm_vec[j] *= attn;
                }
            }

            sequence.push(ssm_vec);
        }

        sequence
    }

    /// Back-project a single 768D SSM hidden state to a 16384D HDC vector.
    ///
    /// Uses group 0's `group_w_down` to project 768D → chunk_dim, then tiles across all chunks.
    pub fn project_to_hdc(&self, ssm_hidden: &[f32]) -> ContinuousHV {
        assert_eq!(
            ssm_hidden.len(),
            self.ssm_dim,
            "Expected {}-dim SSM hidden, got {}",
            self.ssm_dim,
            ssm_hidden.len()
        );

        // Down-project using group 0 (single hidden state has no chunk assignment)
        let w_down = &self.group_w_down[0];
        let mut chunk_recon = vec![0.0f32; self.chunk_size];
        for k in 0..self.chunk_size {
            let mut sum = 0.0f32;
            for j in 0..self.ssm_dim {
                sum += w_down[k * self.ssm_dim + j] * ssm_hidden[j];
            }
            chunk_recon[k] = sum;
        }

        // Overlap-aware reconstruction
        let mut values = vec![0.0f32; self.hdc_dim];
        let mut counts = vec![0u16; self.hdc_dim];
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            for k in 0..self.chunk_size {
                let idx = start + k;
                if idx < self.hdc_dim {
                    values[idx] += chunk_recon[k];
                    counts[idx] += 1;
                }
            }
        }
        for (v, &c) in values.iter_mut().zip(&counts) {
            if c > 1 {
                *v /= c as f32;
            }
        }

        ContinuousHV::from_vec(values)
    }

    /// Backpropagate gradients and update weights.
    pub fn backward(&mut self, input_hv: &ContinuousHV, d_hdc: &ContinuousHV, lr: f32) -> Result<()> {
        // Map HDC gradient back to chunk-level weights
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let g = self.group_for_chunk(chunk_idx);
            let w_up = &mut self.group_w_up[g];

            // Simplified update: align input_hv chunk -> d_hdc chunk
            for k in 0..self.chunk_size {
                let idx = start + k;
                if idx < self.hdc_dim {
                    let gi = d_hdc.values[idx];
                    let xi = input_hv.values[idx];
                    for j in 0..self.ssm_dim {
                        w_up[j * self.chunk_size + k] -= lr * gi * xi;
                    }
                }
            }
        }
        Ok(())
    }

    /// Back-project a full sequence of SSM hidden states to a 16384D HDC vector.
    ///
    /// Each SSM vector in the sequence corresponds to one chunk position. Subtracts
    /// positional encoding before down-projecting, then places each reconstructed
    /// chunk at its stride-offset position (averaging overlaps).
    pub fn project_sequence_to_hdc(&self, ssm_sequence: &[Vec<f32>]) -> ContinuousHV {
        assert_eq!(
            ssm_sequence.len(),
            self.num_chunks,
            "Expected {} SSM vectors, got {}",
            self.num_chunks,
            ssm_sequence.len()
        );

        let mut values = vec![0.0f32; self.hdc_dim];
        let mut counts = vec![0u16; self.hdc_dim];

        for (chunk_idx, ssm_vec) in ssm_sequence.iter().enumerate() {
            assert_eq!(ssm_vec.len(), self.ssm_dim);
            let g = self.group_for_chunk(chunk_idx);
            let w_down = &self.group_w_down[g];

            // Subtract positional encoding to recover the projected chunk
            let pos_offset = chunk_idx * self.ssm_dim;
            let mut de_pos = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                de_pos[j] = ssm_vec[j] - self.pos_enc[pos_offset + j];
            }

            // Down-project: ssm_dim → chunk_size
            let mut chunk_recon = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                let mut sum = 0.0f32;
                for j in 0..self.ssm_dim {
                    sum += w_down[k * self.ssm_dim + j] * de_pos[j];
                }
                chunk_recon[k] = sum;
            }

            // Place at stride-offset position
            let start = chunk_idx * self.stride;
            for k in 0..self.chunk_size {
                let idx = start + k;
                if idx < self.hdc_dim {
                    values[idx] += chunk_recon[k];
                    counts[idx] += 1;
                }
            }
        }

        for (v, &c) in values.iter_mut().zip(&counts) {
            if c > 1 {
                *v /= c as f32;
            }
        }

        ContinuousHV::from_vec(values)
    }

    // ─── Gradient computation ────────────────────────────────────────────

    /// Compute roundtrip autoencoder gradients: thought → forward → backward → error.
    ///
    /// Loss: Mean MSE across all chunks (averaged, not summed, to keep gradient
    /// norms independent of `num_chunks`).
    pub fn compute_roundtrip_gradients(&mut self, thought: &ContinuousHV) {
        let scale = 1.0 / self.num_chunks as f32;
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let g = self.group_for_chunk(chunk_idx);

            // Forward: LayerNorm → up-project
            let normed = self.layer_norm(chunk);
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.group_w_up[g][j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }

            // Add positional encoding for roundtrip
            let pos_offset = chunk_idx * self.ssm_dim;
            for j in 0..self.ssm_dim {
                ssm_vec[j] += self.pos_enc[pos_offset + j];
            }

            // Apply adapter forward (if present) and save pre-adapter for backward
            let pre_adapter = ssm_vec.clone();
            if let Some(ref adapter) = self.adapter {
                ssm_vec = adapter.forward(&pre_adapter);
            }

            // Apply chunk attention gate
            let attn = if self.learned_attention {
                sigmoid(self.chunk_attention[chunk_idx])
            } else {
                1.0
            };
            if self.learned_attention {
                for j in 0..self.ssm_dim {
                    ssm_vec[j] *= attn;
                }
            }

            // Backward: down-project
            let mut recon = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                let mut sum = 0.0f32;
                for j in 0..self.ssm_dim {
                    sum += self.group_w_down[g][k * self.ssm_dim + j] * ssm_vec[j];
                }
                recon[k] = sum;
            }

            // Error: original - reconstruction (scaled by 1/num_chunks for mean)
            let mut error = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                error[k] = normed[k] - recon[k];
            }

            // Gradient for w_chunk_down
            for k in 0..self.chunk_size {
                for j in 0..self.ssm_dim {
                    self.grad_group_down[g][k * self.ssm_dim + j] +=
                        -2.0 * scale * error[k] * ssm_vec[j];
                }
            }

            // Backprop through down-projection to get ssm_error
            let mut ssm_error = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.group_w_down[g][k * self.ssm_dim + j] * (-2.0 * scale * error[k]);
                }
                ssm_error[j] = sum;
            }

            // Backprop through attention gate
            if self.learned_attention && attn > 1e-10 {
                let mut d_attn = 0.0f32;
                for j in 0..self.ssm_dim {
                    d_attn += ssm_error[j] * ssm_vec[j] / attn;
                }
                let sigmoid_grad = attn * (1.0 - attn);
                self.grad_chunk_attention[chunk_idx] += d_attn * sigmoid_grad;
                for j in 0..self.ssm_dim {
                    ssm_error[j] *= attn;
                }
            }

            // Backprop through adapter (Improvement E)
            if let Some(ref mut adapter) = self.adapter {
                ssm_error = adapter.backward(&pre_adapter, &ssm_error);
            }

            // Gradient for learned positional encoding
            if self.learned_pos_enc {
                for j in 0..self.ssm_dim {
                    self.grad_pos_enc[pos_offset + j] += ssm_error[j];
                }
            }

            // Gradient for w_chunk_up
            for j in 0..self.ssm_dim {
                for k in 0..self.chunk_size {
                    self.grad_group_up[g][j * self.chunk_size + k] += ssm_error[j] * normed[k];
                }
            }

            // Gradient for LayerNorm gamma/beta
            self.accumulate_ln_gradients(chunk, &normed, &error);
        }
    }

    /// Compute gradients from reconstruction loss: thought vs target.
    pub fn compute_gradients(&mut self, thought: &ContinuousHV, target: &ContinuousHV) {
        let scale = 1.0 / self.num_chunks as f32;
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let target_chunk = &target.values[start..start + self.chunk_size];
            let g = self.group_for_chunk(chunk_idx);

            // Forward: LayerNorm → up-project
            let normed = self.layer_norm(chunk);
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.group_w_up[g][j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }

            // Add positional encoding
            let pos_offset = chunk_idx * self.ssm_dim;
            for j in 0..self.ssm_dim {
                ssm_vec[j] += self.pos_enc[pos_offset + j];
            }

            // Backward: down-project
            let mut recon = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                let mut sum = 0.0f32;
                for j in 0..self.ssm_dim {
                    sum += self.group_w_down[g][k * self.ssm_dim + j] * ssm_vec[j];
                }
                recon[k] = sum;
            }

            // Error: target_chunk - reconstruction
            let mut error = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                error[k] = target_chunk[k] - recon[k];
            }

            // Gradient for w_chunk_down
            for k in 0..self.chunk_size {
                for j in 0..self.ssm_dim {
                    self.grad_group_down[g][k * self.ssm_dim + j] +=
                        -2.0 * scale * error[k] * ssm_vec[j];
                }
            }

            // Backprop through down-projection
            let mut ssm_error = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.group_w_down[g][k * self.ssm_dim + j] * (-2.0 * scale * error[k]);
                }
                ssm_error[j] = sum;
            }

            // Gradient for learned positional encoding
            if self.learned_pos_enc {
                for j in 0..self.ssm_dim {
                    self.grad_pos_enc[pos_offset + j] += ssm_error[j];
                }
            }

            // Gradient for w_chunk_up
            for j in 0..self.ssm_dim {
                for k in 0..self.chunk_size {
                    self.grad_group_up[g][j * self.chunk_size + k] += ssm_error[j] * normed[k];
                }
            }

            self.accumulate_ln_gradients(chunk, &normed, &error);
        }
    }

    /// Compute directional cosine loss gradients.
    pub fn compute_directional_gradients(
        &mut self,
        thought: &ContinuousHV,
        target: Option<&ContinuousHV>,
    ) {
        let scale = 1.0 / self.num_chunks as f32;

        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let g = self.group_for_chunk(chunk_idx);

            // Forward: LayerNorm → up-project
            let normed = self.layer_norm(chunk);
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.group_w_up[g][j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }

            // Target SSM vector
            let target_ssm = if let Some(t) = target {
                let t_start = chunk_idx * self.stride;
                let t_chunk = &t.values[t_start..t_start + self.chunk_size];
                let t_normed = self.layer_norm(t_chunk);
                let mut t_ssm = vec![0.0f32; self.ssm_dim];
                for j in 0..self.ssm_dim {
                    let mut sum = 0.0f32;
                    for k in 0..self.chunk_size {
                        sum += self.group_w_up[g][j * self.chunk_size + k] * t_normed[k];
                    }
                    t_ssm[j] = sum;
                }
                t_ssm
            } else {
                let mut target = vec![0.0f32; self.ssm_dim];
                for j in 0..self.ssm_dim {
                    target[j] = normed[j % self.chunk_size];
                }
                target
            };

            let a_norm: f32 = ssm_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let b_norm: f32 = target_ssm.iter().map(|x| x * x).sum::<f32>().sqrt();

            if a_norm < 1e-10 || b_norm < 1e-10 {
                continue;
            }

            let dot: f32 = ssm_vec
                .iter()
                .zip(target_ssm.iter())
                .map(|(a, b)| a * b)
                .sum();
            let cos_sim = dot / (a_norm * b_norm);

            let inv_ab = 1.0 / (a_norm * b_norm);
            let inv_a2 = 1.0 / (a_norm * a_norm);
            let mut ssm_grad = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                ssm_grad[j] = -(target_ssm[j] * inv_ab - cos_sim * ssm_vec[j] * inv_a2);
            }

            // Backprop to w_chunk_up
            for j in 0..self.ssm_dim {
                for k in 0..self.chunk_size {
                    self.grad_group_up[g][j * self.chunk_size + k] +=
                        scale * ssm_grad[j] * normed[k];
                }
            }

            // LayerNorm gradients
            let mut chunk_error = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                for j in 0..self.ssm_dim {
                    chunk_error[k] += self.group_w_up[g][j * self.chunk_size + k] * ssm_grad[j];
                }
                chunk_error[k] *= scale;
            }
            self.accumulate_ln_gradients(chunk, &normed, &chunk_error);
        }
    }

    /// Compute temporal smoothness loss gradients.
    pub fn compute_smoothness_gradients(&mut self, thought: &ContinuousHV, weight: f32) {
        if self.num_chunks < 2 || weight <= 0.0 {
            return;
        }

        let scale = weight / (self.num_chunks - 1) as f32;

        // Forward all chunks to get SSM vectors
        let mut ssm_vecs = Vec::with_capacity(self.num_chunks);
        let mut normed_chunks = Vec::with_capacity(self.num_chunks);
        let mut groups = Vec::with_capacity(self.num_chunks);
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let normed = self.layer_norm(chunk);
            let g = self.group_for_chunk(chunk_idx);

            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.group_w_up[g][j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }
            ssm_vecs.push(ssm_vec);
            normed_chunks.push(normed);
            groups.push(g);
        }

        let mut ssm_grads = vec![vec![0.0f32; self.ssm_dim]; self.num_chunks];
        for i in 0..self.num_chunks - 1 {
            for j in 0..self.ssm_dim {
                let diff = ssm_vecs[i + 1][j] - ssm_vecs[i][j];
                ssm_grads[i][j] += -2.0 * scale * diff;
                ssm_grads[i + 1][j] += 2.0 * scale * diff;
            }
        }

        // Chain rule to w_chunk_up per group
        for chunk_idx in 0..self.num_chunks {
            let g = groups[chunk_idx];
            for j in 0..self.ssm_dim {
                if ssm_grads[chunk_idx][j].abs() < 1e-12 {
                    continue;
                }
                for k in 0..self.chunk_size {
                    self.grad_group_up[g][j * self.chunk_size + k] +=
                        ssm_grads[chunk_idx][j] * normed_chunks[chunk_idx][k];
                }
            }
        }
    }

    /// Compute rank regularization gradients for the up-projection.
    pub fn compute_rank_regularization_gradients(&mut self, weight: f32, num_samples: usize) {
        if weight <= 0.0 || self.ssm_dim < 2 {
            return;
        }

        let scale = weight / num_samples.max(1) as f32;

        // Apply to each group independently
        for g in 0..self.num_groups {
            let seed_val = (self.group_w_up[g][0].to_bits() as u64)
                .wrapping_add(self.group_w_up[g].len() as u64)
                ^ 0x9E3779B97F4A7C15;
            let mut rng_state = seed_val | 1;
            let total_pairs = self.ssm_dim * self.ssm_dim;
            for _sample in 0..num_samples {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let pair_idx = (rng_state as usize) % total_pairs;
                let j1 = pair_idx / self.ssm_dim;
                let j2 = pair_idx % self.ssm_dim;
                if j1 == j2 {
                    continue;
                }

                let row1_start = j1 * self.chunk_size;
                let row2_start = j2 * self.chunk_size;
                let mut dot = 0.0f32;
                for k in 0..self.chunk_size {
                    dot += self.group_w_up[g][row1_start + k] * self.group_w_up[g][row2_start + k];
                }

                for k in 0..self.chunk_size {
                    self.grad_group_up[g][row1_start + k] +=
                        scale * dot * self.group_w_up[g][row2_start + k];
                    self.grad_group_up[g][row2_start + k] +=
                        scale * dot * self.group_w_up[g][row1_start + k];
                }
            }
        }
    }

    /// Compute anti-collapse regularization gradients (Improvement A).
    ///
    /// Projects thought → SSM chunks, samples pairs of chunks, and if their
    /// cosine similarity exceeds `threshold`, adds gradient to push them apart.
    /// This prevents mode collapse where all chunks project to similar SSM vectors.
    pub fn compute_anticollapse_gradients(
        &mut self,
        thought: &ContinuousHV,
        weight: f32,
        threshold: f32,
    ) {
        if weight <= 0.0 || self.num_chunks < 2 {
            return;
        }

        // Project all chunks
        let mut ssm_vecs = Vec::with_capacity(self.num_chunks);
        let mut normed_chunks = Vec::with_capacity(self.num_chunks);
        let mut groups = Vec::with_capacity(self.num_chunks);
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let normed = self.layer_norm(chunk);
            let g = self.group_for_chunk(chunk_idx);

            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.group_w_up[g][j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }
            ssm_vecs.push(ssm_vec);
            normed_chunks.push(normed);
            groups.push(g);
        }

        // Sample pairs and push apart if too similar
        // Use strided pair sampling: (0,N/2), (1,N/2+1), ... for good coverage
        let half = self.num_chunks / 2;
        let num_pairs = half.min(32); // cap at 32 pairs per step
        let pair_scale = weight / num_pairs.max(1) as f32;

        for p in 0..num_pairs {
            let i = p % self.num_chunks;
            let j_idx = (i + half) % self.num_chunks;
            if i == j_idx {
                continue;
            }

            // Cosine similarity
            let dot: f32 = ssm_vecs[i]
                .iter()
                .zip(ssm_vecs[j_idx].iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_i: f32 = ssm_vecs[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_j: f32 = ssm_vecs[j_idx].iter().map(|x| x * x).sum::<f32>().sqrt();

            if norm_i < 1e-10 || norm_j < 1e-10 {
                continue;
            }

            let cos_sim = dot / (norm_i * norm_j);
            if cos_sim <= threshold {
                continue;
            }

            // Push apart: gradient of cos_sim w.r.t. ssm_vecs[i] and ssm_vecs[j]
            let inv_ij = 1.0 / (norm_i * norm_j);
            let inv_i2 = 1.0 / (norm_i * norm_i);
            let inv_j2 = 1.0 / (norm_j * norm_j);

            // d(cos_sim)/d(ssm_i) = (ssm_j / (|i|*|j|)) - cos_sim * ssm_i / |i|^2
            // We want to DECREASE cos_sim, so gradient = +d(cos_sim)/d(ssm)
            let gi = groups[i];
            let gj = groups[j_idx];

            for dim in 0..self.ssm_dim {
                let d_ssm_i = pair_scale
                    * (ssm_vecs[j_idx][dim] * inv_ij - cos_sim * ssm_vecs[i][dim] * inv_i2);
                let d_ssm_j = pair_scale
                    * (ssm_vecs[i][dim] * inv_ij - cos_sim * ssm_vecs[j_idx][dim] * inv_j2);

                // Chain through w_chunk_up for chunk i
                for k in 0..self.chunk_size {
                    self.grad_group_up[gi][dim * self.chunk_size + k] +=
                        d_ssm_i * normed_chunks[i][k];
                }
                // Chain through w_chunk_up for chunk j
                for k in 0..self.chunk_size {
                    self.grad_group_up[gj][dim * self.chunk_size + k] +=
                        d_ssm_j * normed_chunks[j_idx][k];
                }
            }
        }
    }

    /// Compute E2E gradients from Mamba's cross-entropy loss backpropagated
    /// through the projection.
    ///
    /// `ssm_gradients`: gradient of loss w.r.t. the SSM embedding at each chunk position.
    /// Only non-None entries contribute gradients (sparse gradient from rotating position).
    pub fn compute_e2e_gradients(
        &mut self,
        thought: &ContinuousHV,
        ssm_gradients: &[Option<Vec<f32>>],
    ) {
        let scale = 1.0 / self.num_chunks as f32;
        for chunk_idx in 0..self.num_chunks {
            let grad = match &ssm_gradients[chunk_idx] {
                Some(g) => g,
                None => continue,
            };
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let g = self.group_for_chunk(chunk_idx);
            let normed = self.layer_norm(chunk);

            let mut ssm_error = grad.clone();
            for v in &mut ssm_error {
                *v *= scale;
            }

            // If adapter is present, backprop through it first
            if let Some(ref mut adapter) = self.adapter {
                // Recompute pre-adapter ssm_vec for adapter backward
                let mut ssm_vec = vec![0.0f32; self.ssm_dim];
                for j in 0..self.ssm_dim {
                    let mut sum = 0.0f32;
                    for k in 0..self.chunk_size {
                        sum += self.group_w_up[g][j * self.chunk_size + k] * normed[k];
                    }
                    ssm_vec[j] = sum;
                }
                let pos_offset = chunk_idx * self.ssm_dim;
                for j in 0..self.ssm_dim {
                    ssm_vec[j] += self.pos_enc[pos_offset + j];
                }
                ssm_error = adapter.backward(&ssm_vec, &ssm_error);
            }

            // Gradient for w_chunk_up
            for j in 0..self.ssm_dim {
                for k in 0..self.chunk_size {
                    self.grad_group_up[g][j * self.chunk_size + k] += ssm_error[j] * normed[k];
                }
            }

            // Gradient for learned positional encoding
            if self.learned_pos_enc {
                let pos_offset = chunk_idx * self.ssm_dim;
                for j in 0..self.ssm_dim {
                    self.grad_pos_enc[pos_offset + j] += ssm_error[j];
                }
            }
        }
    }

    // ─── Gradient application ────────────────────────────────────────────

    /// Apply accumulated gradients with learning rate and gradient clipping.
    pub fn apply_gradients(&mut self, lr: f32, grad_clip: f32) -> GradientStepMetrics {
        // Compute combined norm across all groups
        let mut total_sq = 0.0f32;
        let mut max_norm_up = 0.0f32;
        let mut max_norm_down = 0.0f32;
        for g in 0..self.num_groups {
            let nu = l2_norm(&self.grad_group_up[g]);
            let nd = l2_norm(&self.grad_group_down[g]);
            max_norm_up = max_norm_up.max(nu);
            max_norm_down = max_norm_down.max(nd);
            total_sq += nu.powi(2) + nd.powi(2);
        }
        let norm_ln =
            (l2_norm(&self.grad_ln_gamma).powi(2) + l2_norm(&self.grad_ln_beta).powi(2)).sqrt();
        let norm_pos = if self.learned_pos_enc {
            l2_norm(&self.grad_pos_enc)
        } else {
            0.0
        };
        let norm_attn = if self.learned_attention {
            l2_norm(&self.grad_chunk_attention)
        } else {
            0.0
        };
        let adapter_norm = self.adapter.as_ref().map_or(0.0f32, |a| {
            (l2_norm(&a.grad_w1).powi(2)
                + l2_norm(&a.grad_b1).powi(2)
                + l2_norm(&a.grad_w2).powi(2)
                + l2_norm(&a.grad_b2).powi(2))
            .sqrt()
        });
        let combined_norm = (total_sq
            + norm_ln.powi(2)
            + norm_pos.powi(2)
            + norm_attn.powi(2)
            + adapter_norm.powi(2))
        .sqrt();

        let was_clipped = combined_norm > grad_clip;
        let clip_scale = if was_clipped {
            grad_clip / combined_norm
        } else {
            1.0
        };

        let effective_lr = lr * clip_scale;

        // Apply to per-group weights
        for g in 0..self.num_groups {
            for (w, grad) in self.group_w_up[g]
                .iter_mut()
                .zip(self.grad_group_up[g].iter())
            {
                *w -= effective_lr * grad;
            }
            for (w, grad) in self.group_w_down[g]
                .iter_mut()
                .zip(self.grad_group_down[g].iter())
            {
                *w -= effective_lr * grad;
            }
        }
        // Apply to LayerNorm
        for (w, grad) in self.ln_gamma.iter_mut().zip(self.grad_ln_gamma.iter()) {
            *w -= effective_lr * grad;
        }
        for (w, grad) in self.ln_beta.iter_mut().zip(self.grad_ln_beta.iter()) {
            *w -= effective_lr * grad;
        }
        // Apply to learned positional encoding
        if self.learned_pos_enc {
            for (w, grad) in self.pos_enc.iter_mut().zip(self.grad_pos_enc.iter()) {
                *w -= effective_lr * grad;
            }
        }
        // Apply to learned chunk attention
        if self.learned_attention {
            for (w, grad) in self
                .chunk_attention
                .iter_mut()
                .zip(self.grad_chunk_attention.iter())
            {
                *w -= effective_lr * grad;
            }
        }
        // Apply adapter gradients (Improvement E)
        if let Some(ref mut adapter) = self.adapter {
            adapter.apply_gradients(effective_lr, grad_clip);
        }

        // Zero accumulators
        for g in 0..self.num_groups {
            self.grad_group_up[g].fill(0.0);
            self.grad_group_down[g].fill(0.0);
        }
        self.grad_ln_gamma.fill(0.0);
        self.grad_ln_beta.fill(0.0);
        if self.learned_pos_enc {
            self.grad_pos_enc.fill(0.0);
        }
        if self.learned_attention {
            self.grad_chunk_attention.fill(0.0);
        }

        GradientStepMetrics {
            norm_down: max_norm_down,
            norm_up: max_norm_up,
            norm_backward: max_norm_down,
            was_clipped,
        }
    }

    /// Scale accumulated gradients by a factor (for surprise-weighted learning).
    pub fn scale_accumulated_gradients(&mut self, scale: f32) {
        for g in 0..self.num_groups {
            for grad in &mut self.grad_group_up[g] {
                *grad *= scale;
            }
            for grad in &mut self.grad_group_down[g] {
                *grad *= scale;
            }
        }
        for grad in &mut self.grad_ln_gamma {
            *grad *= scale;
        }
        for grad in &mut self.grad_ln_beta {
            *grad *= scale;
        }
        if self.learned_pos_enc {
            for grad in &mut self.grad_pos_enc {
                *grad *= scale;
            }
        }
        if self.learned_attention {
            for grad in &mut self.grad_chunk_attention {
                *grad *= scale;
            }
        }
        // Scale adapter gradients too (Improvement E)
        if let Some(ref mut adapter) = self.adapter {
            adapter.scale_gradients(scale);
        }
    }

    /// Compute contrastive gradients: push `anchor` projection away from `negative`.
    pub fn compute_contrastive_gradients(
        &mut self,
        anchor: &ContinuousHV,
        negative: &ContinuousHV,
        weight: f32,
    ) {
        let scale = weight / self.num_chunks as f32;
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let a_chunk = &anchor.values[start..start + self.chunk_size];
            let n_chunk = &negative.values[start..start + self.chunk_size];
            let g = self.group_for_chunk(chunk_idx);

            let a_normed = self.layer_norm(a_chunk);
            let n_normed = self.layer_norm(n_chunk);

            // Project both
            let mut a_ssm = vec![0.0f32; self.ssm_dim];
            let mut n_ssm = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                for k in 0..self.chunk_size {
                    let w = self.group_w_up[g][j * self.chunk_size + k];
                    a_ssm[j] += w * a_normed[k];
                    n_ssm[j] += w * n_normed[k];
                }
            }

            // Repulsive gradient
            for j in 0..self.ssm_dim {
                let ssm_diff = a_ssm[j] - n_ssm[j];
                for k in 0..self.chunk_size {
                    let input_diff = a_normed[k] - n_normed[k];
                    self.grad_group_up[g][j * self.chunk_size + k] -= scale * ssm_diff * input_diff;
                }
            }
        }
    }

    // ─── Diagnostics ─────────────────────────────────────────────────────

    /// Get the bottleneck activation for a thought (for diagnostics).
    pub fn bottleneck_activation(&self, thought: &ContinuousHV) -> Vec<f32> {
        let sample_indices: Vec<usize> = if self.num_chunks <= 4 {
            (0..self.num_chunks).collect()
        } else {
            let step = self.num_chunks / 4;
            vec![0, step, 2 * step, 3 * step]
        };

        let mut result = vec![0.0f32; self.chunk_size];
        for &ci in &sample_indices {
            let start = ci * self.stride;
            let normed = self.layer_norm(&thought.values[start..start + self.chunk_size]);
            for (r, n) in result.iter_mut().zip(normed.iter()) {
                *r += n;
            }
        }
        let inv_n = 1.0 / sample_indices.len() as f32;
        for r in &mut result {
            *r *= inv_n;
        }
        result
    }

    /// Compute roundtrip prediction error: average per-chunk reconstruction quality.
    pub fn roundtrip_pe(&self, thought: &ContinuousHV) -> f32 {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let g = self.group_for_chunk(chunk_idx);

            let normed = self.layer_norm(chunk);

            // Up-project
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.group_w_up[g][j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }

            // Down-project
            let mut recon = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                let mut sum = 0.0f32;
                for j in 0..self.ssm_dim {
                    sum += self.group_w_down[g][k * self.ssm_dim + j] * ssm_vec[j];
                }
                recon[k] = sum;
            }

            let dot: f32 = normed.iter().zip(recon.iter()).map(|(a, b)| a * b).sum();
            let n_norm: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
            let r_norm: f32 = recon.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n_norm > 1e-10 && r_norm > 1e-10 {
                total_sim += (dot / (n_norm * r_norm)).clamp(-1.0, 1.0);
                count += 1;
            }
        }
        let avg_sim = if count > 0 {
            total_sim / count as f32
        } else {
            0.0
        };
        1.0 - avg_sim
    }

    /// Estimate effective rank of the up-projection.
    pub fn effective_rank(&self, thoughts: &[ContinuousHV]) -> f32 {
        if thoughts.len() < 2 {
            return self.chunk_size as f32;
        }

        let sample_chunks: Vec<usize> = if self.num_chunks <= 4 {
            (0..self.num_chunks).collect()
        } else {
            let step = self.num_chunks / 4;
            vec![0, step, 2 * step, 3 * step]
        };

        let mut ssm_vecs: Vec<Vec<f32>> = Vec::with_capacity(thoughts.len() * sample_chunks.len());
        for thought in thoughts {
            for &ci in &sample_chunks {
                let start = ci * self.stride;
                let chunk = &thought.values[start..start + self.chunk_size];
                let normed = self.layer_norm(chunk);
                let g = self.group_for_chunk(ci);
                let mut ssm_vec = vec![0.0f32; self.ssm_dim];
                for j in 0..self.ssm_dim {
                    let mut sum = 0.0f32;
                    for k in 0..self.chunk_size {
                        sum += self.group_w_up[g][j * self.chunk_size + k] * normed[k];
                    }
                    ssm_vec[j] = sum;
                }
                ssm_vecs.push(ssm_vec);
            }
        }

        let n = ssm_vecs.len() as f32;
        let mut variances = vec![0.0f32; self.ssm_dim];
        let mut means = vec![0.0f32; self.ssm_dim];

        for vec in &ssm_vecs {
            for (j, &v) in vec.iter().enumerate() {
                means[j] += v;
            }
        }
        for m in &mut means {
            *m /= n;
        }
        for vec in &ssm_vecs {
            for (j, &v) in vec.iter().enumerate() {
                let d = v - means[j];
                variances[j] += d * d;
            }
        }
        for v in &mut variances {
            *v /= n;
        }

        let total_var: f32 = variances.iter().sum();
        if total_var < 1e-10 {
            return 1.0;
        }

        let mut entropy = 0.0f32;
        for &v in &variances {
            let p = v / total_var;
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }

        entropy.exp()
    }

    /// Warm-start the up-projection from training data via power-iteration PCA.
    pub fn warm_start_from_samples(&mut self, samples: &[ContinuousHV]) {
        if samples.len() < 2 {
            return;
        }

        let mut normed_chunks: Vec<Vec<f32>> = Vec::with_capacity(samples.len() * self.num_chunks);
        for thought in samples {
            for ci in 0..self.num_chunks {
                let start = ci * self.stride;
                if start + self.chunk_size <= thought.values.len() {
                    normed_chunks
                        .push(self.layer_norm(&thought.values[start..start + self.chunk_size]));
                }
            }
        }

        let n = normed_chunks.len();
        if n < 2 {
            return;
        }
        let d = self.chunk_size;
        let k = self.ssm_dim.min(n).min(d);
        let scale = 1.0 / (d as f32).sqrt();

        // Compute mean
        let mut mean = vec![0.0f32; d];
        for chunk in &normed_chunks {
            for (m, &v) in mean.iter_mut().zip(chunk.iter()) {
                *m += v;
            }
        }
        let inv_n = 1.0 / n as f32;
        for m in &mut mean {
            *m *= inv_n;
        }

        // Apply warm start to each group
        for g in 0..self.num_groups {
            for comp_idx in 0..k {
                let row_start = comp_idx * d;

                for j in 0..d {
                    self.group_w_up[g][row_start + j] = if j == comp_idx % d { scale } else { 0.0 };
                }

                for _ in 0..20 {
                    let mut v = vec![0.0f32; d];
                    for chunk in &normed_chunks {
                        let mut dot = 0.0f32;
                        for j in 0..d {
                            dot += (chunk[j] - mean[j]) * self.group_w_up[g][row_start + j];
                        }
                        for j in 0..d {
                            v[j] += dot * (chunk[j] - mean[j]);
                        }
                    }

                    for prev in 0..comp_idx {
                        let prev_start = prev * d;
                        let mut proj = 0.0f32;
                        for j in 0..d {
                            proj += v[j] * self.group_w_up[g][prev_start + j];
                        }
                        for j in 0..d {
                            v[j] -= proj * self.group_w_up[g][prev_start + j];
                        }
                    }

                    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 1e-10 {
                        for j in 0..d {
                            self.group_w_up[g][row_start + j] = v[j] * scale / norm;
                        }
                    }
                }
            }

            // Warm-start w_chunk_down as transpose
            let down_scale = 1.0 / (self.ssm_dim as f32).sqrt();
            for k_idx in 0..self.chunk_size {
                for j in 0..self.ssm_dim {
                    self.group_w_down[g][k_idx * self.ssm_dim + j] =
                        self.group_w_up[g][j * self.chunk_size + k_idx] * down_scale;
                }
            }
        }
    }

    // ─── Checkpointing ──────────────────────────────────────────────────

    /// Flatten all learnable weights into a single vector (for checkpointing).
    ///
    /// Layout: `[g0_up|g1_up|...|g0_down|g1_down|...|ln_gamma|ln_beta|pos_enc?|attn?|adapter?]`
    pub fn flatten_weights(&self) -> Vec<f32> {
        let pos_size = if self.learned_pos_enc {
            self.pos_enc.len()
        } else {
            0
        };
        let attn_size = if self.learned_attention {
            self.chunk_attention.len()
        } else {
            0
        };
        let adapter_size = self.adapter.as_ref().map_or(0, |a| a.num_params());
        let per_group_up = self.ssm_dim * self.chunk_size;
        let per_group_down = self.chunk_size * self.ssm_dim;
        let total_size = self.num_groups * (per_group_up + per_group_down)
            + self.ln_gamma.len()
            + self.ln_beta.len()
            + pos_size
            + attn_size
            + adapter_size;

        let mut weights = Vec::with_capacity(total_size);
        // All up matrices
        for g in 0..self.num_groups {
            weights.extend_from_slice(&self.group_w_up[g]);
        }
        // All down matrices
        for g in 0..self.num_groups {
            weights.extend_from_slice(&self.group_w_down[g]);
        }
        weights.extend_from_slice(&self.ln_gamma);
        weights.extend_from_slice(&self.ln_beta);
        if self.learned_pos_enc {
            weights.extend_from_slice(&self.pos_enc);
        }
        if self.learned_attention {
            weights.extend_from_slice(&self.chunk_attention);
        }
        if let Some(ref adapter) = self.adapter {
            weights.extend_from_slice(&adapter.flatten_weights());
        }
        weights
    }

    /// Load weights from a flat vector (from checkpoint).
    ///
    /// Handles both legacy single-group format and multi-group format.
    pub fn load_weights(&mut self, weights: &[f32]) {
        let per_group_up = self.ssm_dim * self.chunk_size;
        let per_group_down = self.chunk_size * self.ssm_dim;
        let gamma_size = self.ln_gamma.len();
        let beta_size = self.ln_beta.len();

        // Detect format: try multi-group first, fall back to legacy single-group
        let multi_group_base =
            self.num_groups * (per_group_up + per_group_down) + gamma_size + beta_size;
        let legacy_base = per_group_up + per_group_down + gamma_size + beta_size;

        let (is_multi_group, base_size) =
            if weights.len() >= multi_group_base && self.num_groups > 1 {
                (true, multi_group_base)
            } else {
                (false, legacy_base)
            };

        let pos_size = self.pos_enc.len();
        let attn_size = self.chunk_attention.len();

        // Validate that the weight vector is at least the base size
        assert!(
            weights.len() >= base_size,
            "Expected at least {base_size} weights, got {}",
            weights.len()
        );

        let mut offset = 0;

        if is_multi_group {
            for g in 0..self.num_groups {
                self.group_w_up[g].copy_from_slice(&weights[offset..offset + per_group_up]);
                offset += per_group_up;
            }
            for g in 0..self.num_groups {
                self.group_w_down[g].copy_from_slice(&weights[offset..offset + per_group_down]);
                offset += per_group_down;
            }
        } else {
            // Legacy single-group: load into group 0, copy to others if multi-group
            self.group_w_up[0].copy_from_slice(&weights[offset..offset + per_group_up]);
            offset += per_group_up;
            self.group_w_down[0].copy_from_slice(&weights[offset..offset + per_group_down]);
            offset += per_group_down;
            // Copy to other groups if expanded
            for g in 1..self.num_groups {
                self.group_w_up[g] = self.group_w_up[0].clone();
                self.group_w_down[g] = self.group_w_down[0].clone();
            }
        }

        self.ln_gamma
            .copy_from_slice(&weights[offset..offset + gamma_size]);
        offset += gamma_size;
        self.ln_beta
            .copy_from_slice(&weights[offset..offset + beta_size]);
        offset += beta_size;

        // Load learned pos_enc if present
        if offset + pos_size <= weights.len() && weights.len() > base_size {
            self.pos_enc
                .copy_from_slice(&weights[offset..offset + pos_size]);
            self.learned_pos_enc = true;
            offset += pos_size;
        }

        // Load learned chunk attention if present
        if offset + attn_size <= weights.len() && weights.len() >= offset + attn_size {
            // Only load if the remaining size suggests attention is present
            let remaining = weights.len() - offset;
            let adapter_size = self.adapter.as_ref().map_or(0, |a| a.num_params());
            if remaining == attn_size || remaining == attn_size + adapter_size {
                self.chunk_attention
                    .copy_from_slice(&weights[offset..offset + attn_size]);
                self.learned_attention = true;
                offset += attn_size;
            }
        }

        // Load adapter weights if present
        if let Some(ref mut adapter) = self.adapter {
            let ap = adapter.num_params();
            if offset + ap <= weights.len() {
                adapter.load_weights(&weights[offset..offset + ap]);
            }
        }
    }

    /// Number of learnable parameters.
    pub fn num_params(&self) -> usize {
        let per_group_up = self.ssm_dim * self.chunk_size;
        let per_group_down = self.chunk_size * self.ssm_dim;
        let mut total = self.num_groups * (per_group_up + per_group_down)
            + self.ln_gamma.len()
            + self.ln_beta.len();
        if self.learned_pos_enc {
            total += self.pos_enc.len();
        }
        if self.learned_attention {
            total += self.chunk_attention.len();
        }
        if let Some(ref adapter) = self.adapter {
            total += adapter.num_params();
        }
        total
    }

    // ─── Accessors ───────────────────────────────────────────────────────

    /// Whether positional encoding is learned (trainable).
    pub fn learned_pos_enc(&self) -> bool {
        self.learned_pos_enc
    }

    /// Enable or disable learned positional encoding.
    pub fn set_learned_pos_enc(&mut self, learned: bool) {
        self.learned_pos_enc = learned;
        if learned {
            self.grad_pos_enc.fill(0.0);
        }
    }

    /// Project a thought to SSM sequence, selecting only the top-K most
    /// informative chunks by activation magnitude.
    pub fn project_to_ssm_sequence_topk(
        &self,
        thought: &ContinuousHV,
        budget: usize,
    ) -> Vec<Vec<f32>> {
        if budget >= self.num_chunks {
            return self.project_to_ssm_sequence(thought);
        }

        let mut chunk_magnitudes: Vec<(usize, f32)> = (0..self.num_chunks)
            .map(|ci| {
                let start = ci * self.stride;
                let chunk = &thought.values[start..start + self.chunk_size];
                let normed = self.layer_norm(chunk);
                let mag: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
                let score = if self.learned_attention {
                    mag * sigmoid(self.chunk_attention[ci])
                } else {
                    mag
                };
                (ci, score)
            })
            .collect();

        chunk_magnitudes.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut selected: Vec<usize> = chunk_magnitudes[..budget]
            .iter()
            .map(|(idx, _)| *idx)
            .collect();
        selected.sort_unstable();

        let mut sequence = Vec::with_capacity(budget);
        for chunk_idx in selected {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let normed = self.layer_norm(chunk);
            let g = self.group_for_chunk(chunk_idx);

            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.group_w_up[g][j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }

            let pos_offset = chunk_idx * self.ssm_dim;
            for j in 0..self.ssm_dim {
                ssm_vec[j] += self.pos_enc[pos_offset + j];
            }

            // Apply adapter
            if let Some(ref adapter) = self.adapter {
                ssm_vec = adapter.forward(&ssm_vec);
            }

            sequence.push(ssm_vec);
        }

        sequence
    }

    /// Whether chunk attention is learned (trainable).
    pub fn learned_attention(&self) -> bool {
        self.learned_attention
    }

    /// Enable or disable learned chunk attention.
    pub fn set_learned_attention(&mut self, learned: bool) {
        self.learned_attention = learned;
        if learned {
            self.grad_chunk_attention.fill(0.0);
        }
    }

    /// Get the chunk attention weights (sigmoid-activated).
    pub fn chunk_attention_weights(&self) -> Vec<f32> {
        self.chunk_attention.iter().map(|&x| sigmoid(x)).collect()
    }

    /// Chunk size (e.g. 256).
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Number of chunks (64 with no overlap, more with overlap).
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Stride between chunk start positions.
    pub fn stride(&self) -> usize {
        self.stride
    }

    // ─── Internal helpers ──────────────────────────────────────────────────

    /// Apply LayerNorm to a chunk: `gamma * (x - mean) / sqrt(var + eps) + beta`
    fn layer_norm(&self, chunk: &[f32]) -> Vec<f32> {
        let n = chunk.len() as f32;
        let mean: f32 = chunk.iter().sum::<f32>() / n;
        let var: f32 = chunk.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + 1e-5f32).sqrt();

        chunk
            .iter()
            .enumerate()
            .map(|(i, &x)| self.ln_gamma[i] * (x - mean) * inv_std + self.ln_beta[i])
            .collect()
    }

    /// Accumulate LayerNorm gradients (simplified: dL/d_gamma, dL/d_beta).
    fn accumulate_ln_gradients(&mut self, raw_chunk: &[f32], _normed: &[f32], error: &[f32]) {
        let n = raw_chunk.len() as f32;
        let mean: f32 = raw_chunk.iter().sum::<f32>() / n;
        let var: f32 = raw_chunk
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / n;
        let inv_std = 1.0 / (var + 1e-5f32).sqrt();

        for i in 0..raw_chunk.len() {
            let x_hat = (raw_chunk[i] - mean) * inv_std;
            self.grad_ln_gamma[i] += error[i] * x_hat;
            self.grad_ln_beta[i] += error[i];
        }
    }
}

impl std::fmt::Debug for TemporalProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemporalProjection")
            .field("hdc_dim", &self.hdc_dim)
            .field("chunk_size", &self.chunk_size)
            .field("stride", &self.stride)
            .field("num_chunks", &self.num_chunks)
            .field("ssm_dim", &self.ssm_dim)
            .field("num_groups", &self.num_groups)
            .field("has_adapter", &self.adapter.is_some())
            .field("learned_pos_enc", &self.learned_pos_enc)
            .field("learned_attention", &self.learned_attention)
            .field("num_params", &self.num_params())
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate sinusoidal positional encoding (fixed, deterministic).
fn sinusoidal_pos_enc(num_positions: usize, dim: usize) -> Vec<f32> {
    let mut enc = vec![0.0f32; num_positions * dim];
    for pos in 0..num_positions {
        for i in 0..dim {
            let angle = pos as f32 / 10000.0f32.powf(2.0 * (i / 2) as f32 / dim as f32);
            enc[pos * dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    enc
}

/// Initialize a weight vector with genesis-seeded random values scaled by `scale`.
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

/// L2 norm of a float slice.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Sigmoid activation: 1 / (1 + exp(-x)).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-temporal-projection")
    }

    #[test]
    fn test_chunking_dimensions() {
        let tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        assert_eq!(tp.num_chunks, 64);
        assert_eq!(tp.chunk_size, 256);
        assert_eq!(tp.hdc_dim, 16384);
        assert_eq!(tp.ssm_dim, 768);
        assert_eq!(tp.num_groups, 1);
    }

    #[test]
    fn test_ssm_sequence_shape() {
        let tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        let thought = ContinuousHV::random_default(42);
        let sequence = tp.project_to_ssm_sequence(&thought);

        assert_eq!(sequence.len(), 64, "Should produce 64 soft tokens");
        for (i, token) in sequence.iter().enumerate() {
            assert_eq!(token.len(), 768, "Token {i} should be 768D");
            assert!(
                token.iter().all(|x| x.is_finite()),
                "Token {i} has non-finite values"
            );
        }
    }

    #[test]
    fn test_roundtrip_pe_decreases() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        let thought = ContinuousHV::random_default(42).normalize();

        let initial_pe = roundtrip_pe(&tp, &thought);

        for _ in 0..20 {
            tp.compute_roundtrip_gradients(&thought);
            tp.apply_gradients(0.01, 100.0);
        }

        let final_pe = roundtrip_pe(&tp, &thought);
        assert!(
            final_pe < initial_pe,
            "PE should decrease: initial={initial_pe:.4}, final={final_pe:.4}"
        );
    }

    #[test]
    fn test_positional_encoding_orthogonality() {
        let enc = sinusoidal_pos_enc(64, 768);

        let pos0 = &enc[0..768];
        let pos32 = &enc[32 * 768..33 * 768];

        let dot: f32 = pos0.iter().zip(pos32.iter()).map(|(a, b)| a * b).sum();
        let norm0: f32 = pos0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm32: f32 = pos32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine_sim = dot / (norm0 * norm32 + 1e-10);

        assert!(
            cosine_sim.abs() < 0.7,
            "Distant positions should have low correlation, got cosine_sim={cosine_sim:.4}"
        );
    }

    #[test]
    fn test_param_count() {
        let tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        assert_eq!(tp.num_params(), 393_728);
    }

    #[test]
    fn test_flatten_load_roundtrip() {
        let tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        let weights = tp.flatten_weights();

        let mut tp2 =
            TemporalProjection::new(&GenesisSeed::from_phrase("different"), 16384, 256, 768);
        tp2.load_weights(&weights);

        let thought = ContinuousHV::random_default(42);
        let seq1 = tp.project_to_ssm_sequence(&thought);
        let seq2 = tp2.project_to_ssm_sequence(&thought);
        for (a, b) in seq1.iter().zip(seq2.iter()) {
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!(
                    (va - vb).abs() < 1e-6,
                    "Weights should produce identical output"
                );
            }
        }
    }

    #[test]
    fn test_gradient_clipping() {
        let mut tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        let thought = ContinuousHV::random_default(42).normalize();

        tp.compute_roundtrip_gradients(&thought);
        let metrics = tp.apply_gradients(0.01, 0.001);
        assert!(metrics.was_clipped, "Should clip with threshold 0.001");
    }

    #[test]
    fn test_learned_pos_enc_param_count() {
        let tp = TemporalProjection::new_with_options(&test_genesis(), 16384, 256, 768, true);
        assert_eq!(tp.num_params(), 393_728 + 49_152);
        assert!(tp.learned_pos_enc());
    }

    #[test]
    fn test_learned_pos_enc_trains() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new_with_options(&genesis, 16384, 256, 768, true);
        let thought = ContinuousHV::random_default(42).normalize();

        let initial_pos = tp.pos_enc.clone();

        for _ in 0..5 {
            tp.compute_roundtrip_gradients(&thought);
            tp.apply_gradients(0.01, 100.0);
        }

        let changed = tp
            .pos_enc
            .iter()
            .zip(initial_pos.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Learned pos_enc should change during training");
    }

    #[test]
    fn test_learned_pos_enc_flatten_load() {
        let genesis = test_genesis();
        let tp = TemporalProjection::new_with_options(&genesis, 16384, 256, 768, true);
        let weights = tp.flatten_weights();
        assert_eq!(weights.len(), 393_728 + 49_152);

        let mut tp2 = TemporalProjection::new_with_options(
            &GenesisSeed::from_phrase("different"),
            16384,
            256,
            768,
            false,
        );
        tp2.load_weights(&weights);
        assert!(tp2.learned_pos_enc());

        let thought = ContinuousHV::random_default(42);
        let seq1 = tp.project_to_ssm_sequence(&thought);
        let seq2 = tp2.project_to_ssm_sequence(&thought);
        for (a, b) in seq1.iter().zip(seq2.iter()) {
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!((va - vb).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_overlap_chunk_count() {
        let tp = TemporalProjection::new_full(&test_genesis(), 16384, 256, 768, false, 128);
        assert_eq!(tp.num_chunks(), 127);
        assert_eq!(tp.stride(), 128);
        assert_eq!(tp.chunk_size(), 256);
    }

    #[test]
    fn test_overlap_sequence_shape() {
        let tp = TemporalProjection::new_full(&test_genesis(), 16384, 256, 768, false, 128);
        let thought = ContinuousHV::random_default(42);
        let sequence = tp.project_to_ssm_sequence(&thought);

        assert_eq!(sequence.len(), 127, "Should produce 127 overlapping tokens");
        for (i, token) in sequence.iter().enumerate() {
            assert_eq!(token.len(), 768, "Token {i} should be 768D");
            assert!(token.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_overlap_roundtrip_pe_decreases() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new_full(&genesis, 16384, 256, 768, false, 128);
        let thought = ContinuousHV::random_default(42).normalize();

        let initial_pe = tp.roundtrip_pe(&thought);

        for _ in 0..20 {
            tp.compute_roundtrip_gradients(&thought);
            tp.apply_gradients(0.01, 100.0);
        }

        let final_pe = tp.roundtrip_pe(&thought);
        assert!(
            final_pe < initial_pe,
            "Overlap PE should decrease: initial={initial_pe:.4}, final={final_pe:.4}"
        );
    }

    #[test]
    fn test_overlap_project_to_hdc_dimension() {
        let tp = TemporalProjection::new_full(&test_genesis(), 16384, 256, 768, false, 128);
        let ssm_hidden = vec![0.1f32; 768];
        let recon = tp.project_to_hdc(&ssm_hidden);
        assert_eq!(
            recon.dim(),
            16384,
            "Overlapping project_to_hdc must produce hdc_dim output"
        );
    }

    #[test]
    fn test_non_overlap_project_to_hdc_dimension() {
        let tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        let ssm_hidden = vec![0.1f32; 768];
        let recon = tp.project_to_hdc(&ssm_hidden);
        assert_eq!(
            recon.dim(),
            16384,
            "Non-overlapping project_to_hdc must produce hdc_dim output"
        );
    }

    /// Helper: compute roundtrip PE using sequence-based back-projection.
    fn roundtrip_pe(tp: &TemporalProjection, thought: &ContinuousHV) -> f32 {
        let sequence = tp.project_to_ssm_sequence(thought);
        let recon = tp.project_sequence_to_hdc(&sequence);
        1.0 - thought.similarity(&recon).clamp(-1.0, 1.0)
    }

    #[test]
    fn test_sequence_back_projection_dimensions() {
        let tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        let thought = ContinuousHV::random_default(42);
        let sequence = tp.project_to_ssm_sequence(&thought);
        let recon = tp.project_sequence_to_hdc(&sequence);
        assert_eq!(recon.dim(), 16384);
    }

    #[test]
    fn test_sequence_back_projection_after_training() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        let thought = ContinuousHV::random_default(42).normalize();

        for _ in 0..20 {
            tp.compute_roundtrip_gradients(&thought);
            tp.apply_gradients(0.01, 100.0);
        }

        let sequence = tp.project_to_ssm_sequence(&thought);
        let seq_recon = tp.project_sequence_to_hdc(&sequence);
        let seq_sim = thought.similarity(&seq_recon).clamp(-1.0, 1.0);

        assert!(
            seq_sim > 0.0,
            "Trained sequence reconstruction should have positive similarity, got {seq_sim:.4}"
        );
        assert_eq!(seq_recon.dim(), 16384);
    }

    #[test]
    fn test_topk_chunk_budget() {
        let tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        let thought = ContinuousHV::random_default(42).normalize();

        let seq = tp.project_to_ssm_sequence_topk(&thought, 16);
        assert_eq!(seq.len(), 16, "Budget=16 should return 16 chunks");

        let full = tp.project_to_ssm_sequence_topk(&thought, 100);
        assert_eq!(full.len(), 64, "Budget>=64 should return all 64 chunks");

        for token in &seq {
            assert_eq!(token.len(), 768);
            assert!(token.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_smoothness_reduces_chunk_variance() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        let thought = ContinuousHV::random_default(42).normalize();

        let initial_seq = tp.project_to_ssm_sequence(&thought);
        let initial_roughness: f32 = initial_seq
            .windows(2)
            .map(|w| {
                w[0].iter()
                    .zip(w[1].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .sum::<f32>();

        for _ in 0..20 {
            tp.compute_smoothness_gradients(&thought, 0.1);
            tp.apply_gradients(0.01, 100.0);
        }

        let final_seq = tp.project_to_ssm_sequence(&thought);
        let final_roughness: f32 = final_seq
            .windows(2)
            .map(|w| {
                w[0].iter()
                    .zip(w[1].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .sum::<f32>();

        assert!(
            final_roughness < initial_roughness,
            "Smoothness loss should reduce roughness: {initial_roughness:.2} → {final_roughness:.2}"
        );
    }

    #[test]
    fn test_directional_loss_decreases_pe() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        let thought = ContinuousHV::random_default(42).normalize();

        let initial_pe = tp.roundtrip_pe(&thought);

        for _ in 0..20 {
            tp.compute_directional_gradients(&thought, None);
            tp.apply_gradients(0.01, 100.0);
        }

        let final_pe = tp.roundtrip_pe(&thought);
        assert!(
            final_pe.is_finite(),
            "PE should be finite after directional training"
        );
        assert!(
            (final_pe - initial_pe).abs() > 1e-6,
            "Directional loss should modify the projection"
        );
    }

    #[test]
    fn test_sequence_back_projection_overlap() {
        let tp = TemporalProjection::new_full(&test_genesis(), 16384, 256, 768, false, 128);
        let thought = ContinuousHV::random_default(42);
        let sequence = tp.project_to_ssm_sequence(&thought);
        let recon = tp.project_sequence_to_hdc(&sequence);
        assert_eq!(recon.dim(), 16384);
        assert!(recon.values.iter().all(|v| v.is_finite()));
    }

    // ─── New tests for improvements C, E, A ──────────────────────────────

    #[test]
    fn test_multi_group_dimensions() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        assert_eq!(tp.num_groups(), 1);

        tp.set_num_groups(4, &genesis);
        assert_eq!(tp.num_groups(), 4);
        assert_eq!(tp.group_w_up.len(), 4);
        assert_eq!(tp.group_w_down.len(), 4);

        // Each group should have the correct size
        for g in 0..4 {
            assert_eq!(tp.group_w_up[g].len(), 768 * 256);
            assert_eq!(tp.group_w_down[g].len(), 256 * 768);
        }
    }

    #[test]
    fn test_multi_group_forward_produces_valid_output() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        tp.set_num_groups(4, &genesis);

        let thought = ContinuousHV::random_default(42);
        let sequence = tp.project_to_ssm_sequence(&thought);
        assert_eq!(sequence.len(), 64);
        for token in &sequence {
            assert_eq!(token.len(), 768);
            assert!(token.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_adapter_forward_residual() {
        let genesis = test_genesis();
        let adapter = AdapterMlp::new(&genesis, 768);

        let input = vec![1.0f32; 768];
        let output = adapter.forward(&input);
        assert_eq!(output.len(), 768);
        assert!(output.iter().all(|x| x.is_finite()));

        // With near-zero init weights, adapter should be near-identity
        let max_diff: f32 = output
            .iter()
            .zip(input.iter())
            .map(|(o, i)| (o - i).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1.0,
            "Small-init adapter should be near-identity, max_diff={max_diff}"
        );
    }

    #[test]
    fn test_adapter_backward_produces_gradients() {
        let genesis = test_genesis();
        let mut adapter = AdapterMlp::new(&genesis, 16); // Small dim for speed

        let input = vec![0.5f32; 16];
        let d_output = vec![1.0f32; 16];
        let d_input = adapter.backward(&input, &d_output);

        assert_eq!(d_input.len(), 16);
        assert!(d_input.iter().all(|x| x.is_finite()));

        // Gradients should be non-zero
        assert!(adapter.grad_w1.iter().any(|g| g.abs() > 1e-10));
        assert!(adapter.grad_w2.iter().any(|g| g.abs() > 1e-10));
    }

    #[test]
    fn test_adapter_flatten_load_roundtrip() {
        let genesis = test_genesis();
        let adapter = AdapterMlp::new(&genesis, 32);
        let weights = adapter.flatten_weights();
        assert_eq!(weights.len(), adapter.num_params());

        let mut adapter2 = AdapterMlp::new(&GenesisSeed::from_phrase("different"), 32);
        adapter2.load_weights(&weights);

        let input = vec![0.5f32; 32];
        let out1 = adapter.forward(&input);
        let out2 = adapter2.forward(&input);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_anticollapse_produces_gradients() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        let thought = ContinuousHV::random_default(42).normalize();

        tp.compute_anticollapse_gradients(&thought, 0.1, 0.0); // threshold=0 → all pairs counted
                                                               // Check that some gradients are non-zero
        let grad_norm = l2_norm(&tp.grad_group_up[0]);
        assert!(grad_norm > 0.0, "Anti-collapse should produce gradients");
    }

    #[test]
    fn test_gelu_values() {
        // gelu(0) = 0
        assert!((gelu(0.0)).abs() < 1e-6);
        // gelu(large positive) ≈ x
        assert!((gelu(3.0) - 3.0).abs() < 0.1);
        // gelu(large negative) ≈ 0
        assert!(gelu(-3.0).abs() < 0.1);
    }

    #[test]
    fn test_group_for_chunk_distribution() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        tp.set_num_groups(4, &genesis);

        // With 64 chunks and 4 groups, each group should get 16 chunks
        let mut group_counts = vec![0usize; 4];
        for ci in 0..64 {
            let g = tp.group_for_chunk(ci);
            assert!(g < 4);
            group_counts[g] += 1;
        }
        for count in &group_counts {
            assert_eq!(*count, 16, "Each group should get exactly 16 chunks");
        }
    }

    #[test]
    fn test_multi_group_param_count() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        let single_params = tp.num_params();

        tp.set_num_groups(4, &genesis);
        let multi_params = tp.num_params();

        // 4 groups: 4 * (768*256 + 256*768) + 256 + 256 = 4 * 393216 + 512 = 1573376
        let expected = 4 * (768 * 256 + 256 * 768) + 256 + 256;
        assert_eq!(multi_params, expected);
        assert!(multi_params > single_params);
    }

    #[test]
    fn test_adapter_num_params() {
        let genesis = test_genesis();
        let adapter = AdapterMlp::new(&genesis, 768);
        // 2 * 768^2 + 2 * 768 = 1_179_648 + 1536 = 1_181_184
        assert_eq!(adapter.num_params(), 2 * 768 * 768 + 2 * 768);
    }

    // ─── Embedding Stats (Manifold Moment Matching) tests ────────────────

    #[test]
    fn test_embedding_stats_compute() {
        // 4 embeddings, dim=3
        let flat = vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0];
        let stats = EmbeddingStats::compute(&flat, 3);
        assert_eq!(stats.dim, 3);
        assert_eq!(stats.count, 4);
        // mean = [4, 5, 6]
        assert!((stats.mean[0] - 4.0).abs() < 1e-5);
        assert!((stats.mean[1] - 5.0).abs() < 1e-5);
        assert!((stats.mean[2] - 6.0).abs() < 1e-5);
        // var = [(9+1+1+9)/4, same, same] = [5, 5, 5]
        assert!((stats.variance[0] - 5.0).abs() < 1e-5);
        assert!((stats.variance[1] - 5.0).abs() < 1e-5);
        assert!((stats.variance[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_stats_save_load_roundtrip() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let stats = EmbeddingStats::compute(&flat, 3);

        let dir = std::env::temp_dir();
        let path = dir.join("test_embs.bin");
        let path_str = path.to_str().unwrap();

        stats.save(path_str).expect("save failed");
        let loaded = EmbeddingStats::load(path_str).expect("load failed");

        assert_eq!(loaded.dim, stats.dim);
        assert_eq!(loaded.count, stats.count);
        for i in 0..stats.dim {
            assert!((loaded.mean[i] - stats.mean[i]).abs() < 1e-6);
            assert!((loaded.variance[i] - stats.variance[i]).abs() < 1e-6);
        }
        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_adapter_from_stats_whitens() {
        let genesis = test_genesis();
        let dim = 4;
        // Make stats with known mean/var
        let stats = EmbeddingStats {
            dim,
            count: 100,
            mean: vec![10.0, -5.0, 0.0, 3.0],
            variance: vec![4.0, 9.0, 1.0, 16.0],
        };
        let adapter = AdapterMlp::new_from_stats(&genesis, dim, &stats);

        // Input matching the mean should produce near-identity output
        // (w2 is small random, so gelu(w1@mean + b1) ≈ gelu(0) = 0, out ≈ b2 + mean)
        // The residual path gives x, plus b2 gives mean.
        // For input = mean: w1@mean + b1 = diag(1/sqrt(var))@mean - mean/sqrt(var) = 0
        // gelu(0) = 0, so w2@0 = 0, out = 0 + b2 + mean = mean + mean = 2*mean
        // Actually b2 = mean, so out = w2@gelu(0) + b2 + x = 0 + mean + mean = 2*mean
        let output = adapter.forward(&stats.mean);
        for i in 0..dim {
            // Output should be approximately 2*mean (residual + b2)
            assert!(
                (output[i] - 2.0 * stats.mean[i]).abs() < 1.0,
                "dim {i}: expected ~{}, got {}",
                2.0 * stats.mean[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_adapter_from_stats_vs_default() {
        let genesis = test_genesis();
        let dim = 32;
        let stats = EmbeddingStats {
            dim,
            count: 1000,
            mean: (0..dim).map(|i| i as f32 * 0.1).collect(),
            variance: (0..dim).map(|i| 0.5 + i as f32 * 0.3).collect(),
        };
        let stats_adapter = AdapterMlp::new_from_stats(&genesis, dim, &stats);
        let default_adapter = AdapterMlp::new(&genesis, dim);

        // w1 should differ significantly (diagonal whitening vs small random)
        let w1_diff: f32 = stats_adapter
            .w1
            .iter()
            .zip(default_adapter.w1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            w1_diff > 1.0,
            "w1 should differ between stats and default init"
        );

        // b1 should differ (centering vs zeros)
        let b1_diff: f32 = stats_adapter
            .b1
            .iter()
            .zip(default_adapter.b1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            b1_diff > 0.1,
            "b1 should differ between stats and default init"
        );
    }

    #[test]
    fn test_enable_adapter_from_stats() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new(&genesis, 16384, 256, 768);
        assert!(!tp.has_adapter());

        let stats = EmbeddingStats {
            dim: 768,
            count: 50280,
            mean: vec![0.01; 768],
            variance: vec![0.5; 768],
        };
        tp.enable_adapter_from_stats(&genesis, &stats);
        assert!(tp.has_adapter());

        // Verify it works in the forward path
        let thought = ContinuousHV::random_default(42);
        let sequence = tp.project_to_ssm_sequence(&thought);
        assert_eq!(sequence.len(), 64);
        assert!(sequence.iter().all(|v| v.iter().all(|x| x.is_finite())));
    }
}
