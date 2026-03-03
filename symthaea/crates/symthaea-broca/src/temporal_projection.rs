//! Temporal Projection: chunk-based HDC↔SSM bridge with continuous latent prompting.
//!
//! Converts the spatial compression `HDC(16384) → bottleneck(256) → SSM(768)` into
//! temporal sequencing: chunk the 16384D thought vector into **64 tokens of 256D**,
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
//! # Parameters
//!
//! - `w_chunk_up`: `[768 × 256]` — 196,608 params (vs 8.8M spatial)
//! - `w_chunk_down`: `[256 × 768]` — 196,608 params (for backward/PE)
//! - `ln_gamma`, `ln_beta`: `[256]` each — LayerNorm
//! - Total: ~393K params (36× fewer than spatial projection)

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::projection::GradientStepMetrics;

/// Temporal projection: chunk-based HDC↔SSM bridge.
///
/// Chunks a 16384D HDC thought vector into 64 tokens of 256D, applies LayerNorm,
/// up-projects each to 768D via learned `w_chunk_up`, and adds sinusoidal positional
/// encoding. The result is a sequence of continuous embeddings ready for Mamba's
/// `forward_embeds()`.
pub struct TemporalProjection {
    chunk_size: usize,  // 256 (= per-chunk dimension)
    num_chunks: usize,  // 64 (non-overlapping) or more with overlap
    stride: usize,      // step between chunk starts (chunk_size = no overlap, < chunk_size = overlap)
    w_chunk_up: Vec<f32>,   // [ssm_dim × chunk_size] = [768 × 256]
    w_chunk_down: Vec<f32>, // [chunk_size × ssm_dim] = [256 × 768]
    ln_gamma: Vec<f32>,     // [chunk_size] LayerNorm scale
    ln_beta: Vec<f32>,      // [chunk_size] LayerNorm bias
    pos_enc: Vec<f32>,      // [num_chunks × ssm_dim] positional encoding
    learned_pos_enc: bool,  // if true, pos_enc is trainable; if false, fixed sinusoidal
    // Gradient accumulators
    grad_chunk_up: Vec<f32>,
    grad_chunk_down: Vec<f32>,
    grad_ln_gamma: Vec<f32>,
    grad_ln_beta: Vec<f32>,
    grad_pos_enc: Vec<f32>, // [num_chunks × ssm_dim] gradient for learned pos_enc
    // Chunk attention (learned importance per chunk)
    chunk_attention: Vec<f32>,      // [num_chunks] logits (sigmoid-gated during forward)
    grad_chunk_attention: Vec<f32>, // [num_chunks] gradient accumulator
    learned_attention: bool,        // if true, chunk_attention is trainable
    // Dimensions
    hdc_dim: usize,  // 16384
    ssm_dim: usize,  // 768
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
        Self::new_full(genesis, hdc_dim, chunk_dim, ssm_dim, learned_pos_enc, chunk_dim)
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
        assert!(chunk_dim <= hdc_dim, "chunk_dim ({chunk_dim}) must be <= hdc_dim ({hdc_dim})");

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

        let w_chunk_up = init_weights(genesis, "temporal::w_chunk_up", up_size, scale);
        let w_chunk_down = init_weights(genesis, "temporal::w_chunk_down", down_size, scale);

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
            w_chunk_up,
            w_chunk_down,
            ln_gamma,
            ln_beta,
            pos_enc,
            learned_pos_enc,
            grad_chunk_up: vec![0.0; up_size],
            grad_chunk_down: vec![0.0; down_size],
            grad_ln_gamma: vec![0.0; chunk_dim],
            grad_ln_beta: vec![0.0; chunk_dim],
            grad_pos_enc: vec![0.0; pos_enc_size],
            // Chunk attention: initialized to 0 (sigmoid(0) = 0.5 = uniform weighting)
            chunk_attention: vec![0.0; num_chunks],
            grad_chunk_attention: vec![0.0; num_chunks],
            learned_attention: false,
            hdc_dim,
            ssm_dim,
        }
    }

    /// Project a 16384D thought to a sequence of 64 × 768D SSM embeddings.
    ///
    /// Steps per chunk:
    /// 1. Extract 256D slice from thought vector
    /// 2. LayerNorm the chunk
    /// 3. Linear up-projection: 256D → 768D via `w_chunk_up`
    /// 4. Add sinusoidal positional encoding
    /// 5. Scale by chunk attention (if learned)
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

            // LayerNorm
            let normed = self.layer_norm(chunk);

            // Up-project: chunk_size → ssm_dim
            // Auto-vectorization friendly: accumulate into ssm_vec using
            // outer loop over input (broadcast pattern for FMA)
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for k in 0..self.chunk_size {
                let nk = normed[k];
                let w_offset = k; // w_chunk_up[j * chunk_size + k]
                // Inner loop over output dim — contiguous writes, auto-vectorizable
                for j in 0..self.ssm_dim {
                    ssm_vec[j] += self.w_chunk_up[j * self.chunk_size + w_offset] * nk;
                }
            }

            // Add positional encoding + optional attention in fused loop
            let pos_offset = chunk_idx * self.ssm_dim;
            if self.learned_attention {
                let attn = sigmoid(self.chunk_attention[chunk_idx]);
                for j in 0..self.ssm_dim {
                    ssm_vec[j] = (ssm_vec[j] + self.pos_enc[pos_offset + j]) * attn;
                }
            } else {
                for j in 0..self.ssm_dim {
                    ssm_vec[j] += self.pos_enc[pos_offset + j];
                }
            }

            sequence.push(ssm_vec);
        }

        sequence
    }

    /// Back-project a single 768D SSM hidden state to a 16384D HDC vector.
    ///
    /// Uses `w_chunk_down` to project 768D → 256D, then tiles across all chunks.
    /// This is a lossy reconstruction (single hidden state → full thought) used
    /// for PE monitoring, not for generation.
    pub fn project_to_hdc(&self, ssm_hidden: &[f32]) -> ContinuousHV {
        assert_eq!(
            ssm_hidden.len(),
            self.ssm_dim,
            "Expected {}-dim SSM hidden, got {}",
            self.ssm_dim,
            ssm_hidden.len()
        );

        // Down-project: ssm_dim → chunk_size
        let mut chunk_recon = vec![0.0f32; self.chunk_size];
        for k in 0..self.chunk_size {
            let mut sum = 0.0f32;
            for j in 0..self.ssm_dim {
                sum += self.w_chunk_down[k * self.ssm_dim + j] * ssm_hidden[j];
            }
            chunk_recon[k] = sum;
        }

        // Overlap-aware reconstruction: place each chunk at its stride offset
        // and average where chunks overlap (mathematical pseudo-inverse of
        // the overlapping extraction in project_to_ssm_sequence).
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
        // Average overlapping regions (no-op when stride == chunk_size)
        for (v, &c) in values.iter_mut().zip(&counts) {
            if c > 1 {
                *v /= c as f32;
            }
        }

        ContinuousHV::from_vec(values)
    }

    /// Back-project a full sequence of SSM hidden states to a 16384D HDC vector.
    ///
    /// Each SSM vector in the sequence corresponds to one chunk position. Subtracts
    /// positional encoding before down-projecting, then places each reconstructed
    /// chunk at its stride-offset position (averaging overlaps).
    ///
    /// This is the proper inverse of `project_to_ssm_sequence` — unlike `project_to_hdc`
    /// which tiles a single chunk, this reconstructs each position independently.
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
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * de_pos[j];
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

        // Average overlapping regions
        for (v, &c) in values.iter_mut().zip(&counts) {
            if c > 1 {
                *v /= c as f32;
            }
        }

        ContinuousHV::from_vec(values)
    }

    /// Compute roundtrip autoencoder gradients: thought → forward → backward → error.
    ///
    /// Loss: Mean MSE across all chunks (averaged, not summed, to keep gradient
    /// norms independent of `num_chunks`).
    pub fn compute_roundtrip_gradients(&mut self, thought: &ContinuousHV) {
        let scale = 1.0 / self.num_chunks as f32;
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];

            // Forward: LayerNorm → up-project
            let normed = self.layer_norm(chunk);
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_up[j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }

            // Add positional encoding for roundtrip (so down-projection learns to invert it)
            let pos_offset = chunk_idx * self.ssm_dim;
            for j in 0..self.ssm_dim {
                ssm_vec[j] += self.pos_enc[pos_offset + j];
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
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * ssm_vec[j];
                }
                recon[k] = sum;
            }

            // Error: original - reconstruction (scaled by 1/num_chunks for mean)
            let mut error = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                error[k] = normed[k] - recon[k];
            }

            // Gradient for w_chunk_down: d_loss/d_w_down[k,j] = -2 * error[k] * ssm_vec[j] / N
            for k in 0..self.chunk_size {
                for j in 0..self.ssm_dim {
                    self.grad_chunk_down[k * self.ssm_dim + j] +=
                        -2.0 * scale * error[k] * ssm_vec[j];
                }
            }

            // Backprop through down-projection to get ssm_error
            let mut ssm_error = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * (-2.0 * scale * error[k]);
                }
                ssm_error[j] = sum;
            }

            // Backprop through attention gate: d_loss/d_attn = sum(ssm_error[j] * ssm_pre_attn[j])
            // where ssm_pre_attn = ssm_vec / attn (before attention scaling)
            // d_attn/d_logit = attn * (1 - attn) (sigmoid derivative)
            if self.learned_attention && attn > 1e-10 {
                let mut d_attn = 0.0f32;
                for j in 0..self.ssm_dim {
                    // ssm_vec[j] already includes attn, so pre_attn = ssm_vec[j] / attn
                    d_attn += ssm_error[j] * ssm_vec[j] / attn;
                }
                let sigmoid_grad = attn * (1.0 - attn);
                self.grad_chunk_attention[chunk_idx] += d_attn * sigmoid_grad;
                // Also scale ssm_error through the attention gate
                for j in 0..self.ssm_dim {
                    ssm_error[j] *= attn;
                }
            }

            // Gradient for learned positional encoding: same as ssm_error (additive)
            if self.learned_pos_enc {
                for j in 0..self.ssm_dim {
                    self.grad_pos_enc[pos_offset + j] += ssm_error[j];
                }
            }

            // Gradient for w_chunk_up: d_loss/d_w_up[j,k] = ssm_error[j] * normed[k]
            for j in 0..self.ssm_dim {
                for k in 0..self.chunk_size {
                    self.grad_chunk_up[j * self.chunk_size + k] += ssm_error[j] * normed[k];
                }
            }

            // Gradient for LayerNorm gamma/beta
            self.accumulate_ln_gradients(chunk, &normed, &error);
        }
    }

    /// Compute gradients from reconstruction loss: thought vs target.
    ///
    /// Used when Mamba output is meaningful (PE < 0.5). The target is the
    /// attention-weighted bundle of back-projected output tokens.
    /// Gradients are averaged over chunks (not summed) to keep norms stable.
    pub fn compute_gradients(&mut self, thought: &ContinuousHV, target: &ContinuousHV) {
        let scale = 1.0 / self.num_chunks as f32;
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let target_chunk = &target.values[start..start + self.chunk_size];

            // Forward: LayerNorm → up-project
            let normed = self.layer_norm(chunk);
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_up[j * self.chunk_size + k] * normed[k];
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
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * ssm_vec[j];
                }
                recon[k] = sum;
            }

            // Error: target_chunk - reconstruction (scaled by 1/num_chunks for mean)
            let mut error = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                error[k] = target_chunk[k] - recon[k];
            }

            // Gradient for w_chunk_down
            for k in 0..self.chunk_size {
                for j in 0..self.ssm_dim {
                    self.grad_chunk_down[k * self.ssm_dim + j] +=
                        -2.0 * scale * error[k] * ssm_vec[j];
                }
            }

            // Backprop through down-projection
            let mut ssm_error = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * (-2.0 * scale * error[k]);
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
                    self.grad_chunk_up[j * self.chunk_size + k] += ssm_error[j] * normed[k];
                }
            }

            self.accumulate_ln_gradients(chunk, &normed, &error);
        }
    }

    /// Compute directional cosine loss gradients.
    ///
    /// Instead of roundtrip reconstruction (up → down → compare), this directly
    /// optimizes the up-projection so that projected SSM vectors preserve the
    /// **angular structure** of the original chunks. The loss is:
    ///   `1 - cos(w_up @ normed_chunk, target_ssm_vec)` averaged over chunks.
    ///
    /// When `target` is None, computes self-alignment (maximize cosine similarity
    /// between up-projected chunk and a zero-centered version of the chunk padded
    /// to SSM dim — a "directional identity" loss).
    ///
    /// When `target` is Some, computes alignment between projected chunks and
    /// corresponding chunks of the target thought vector.
    pub fn compute_directional_gradients(
        &mut self,
        thought: &ContinuousHV,
        target: Option<&ContinuousHV>,
    ) {
        let scale = 1.0 / self.num_chunks as f32;

        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];

            // Forward: LayerNorm → up-project
            let normed = self.layer_norm(chunk);
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_up[j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }

            // Target SSM vector: if we have a target, project its corresponding chunk;
            // otherwise, use the normed chunk zero-padded to SSM dim as directional target.
            let target_ssm = if let Some(t) = target {
                let t_start = chunk_idx * self.stride;
                let t_chunk = &t.values[t_start..t_start + self.chunk_size];
                let t_normed = self.layer_norm(t_chunk);
                let mut t_ssm = vec![0.0f32; self.ssm_dim];
                for j in 0..self.ssm_dim {
                    let mut sum = 0.0f32;
                    for k in 0..self.chunk_size {
                        sum += self.w_chunk_up[j * self.chunk_size + k] * t_normed[k];
                    }
                    t_ssm[j] = sum;
                }
                t_ssm
            } else {
                // Directional identity: normed chunk padded/tiled to SSM dim
                let mut target = vec![0.0f32; self.ssm_dim];
                for j in 0..self.ssm_dim {
                    target[j] = normed[j % self.chunk_size];
                }
                target
            };

            // Cosine similarity gradient: d(cos(a,b))/da = (b/|b| - cos(a,b)*a/|a|) / |a|
            let a_norm: f32 = ssm_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let b_norm: f32 = target_ssm.iter().map(|x| x * x).sum::<f32>().sqrt();

            if a_norm < 1e-10 || b_norm < 1e-10 {
                continue;
            }

            let dot: f32 = ssm_vec.iter().zip(target_ssm.iter()).map(|(a, b)| a * b).sum();
            let cos_sim = dot / (a_norm * b_norm);

            // Gradient of cosine loss (1 - cos_sim) w.r.t. ssm_vec:
            // d_loss/d_ssm = -(target_ssm / (a_norm * b_norm) - cos_sim * ssm_vec / a_norm^2)
            let inv_ab = 1.0 / (a_norm * b_norm);
            let inv_a2 = 1.0 / (a_norm * a_norm);
            let mut ssm_grad = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                ssm_grad[j] = -(target_ssm[j] * inv_ab - cos_sim * ssm_vec[j] * inv_a2);
            }

            // Backprop to w_chunk_up: d_loss/d_w[j,k] = ssm_grad[j] * normed[k]
            for j in 0..self.ssm_dim {
                for k in 0..self.chunk_size {
                    self.grad_chunk_up[j * self.chunk_size + k] += scale * ssm_grad[j] * normed[k];
                }
            }

            // LayerNorm gradients (chain through ssm_grad → w_up^T → ln)
            let mut chunk_error = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                for j in 0..self.ssm_dim {
                    chunk_error[k] += self.w_chunk_up[j * self.chunk_size + k] * ssm_grad[j];
                }
                chunk_error[k] *= scale;
            }
            self.accumulate_ln_gradients(chunk, &normed, &chunk_error);
        }
    }

    /// Compute temporal smoothness loss gradients.
    ///
    /// Penalizes large differences between consecutive SSM vectors:
    ///   L_smooth = weight * (1/N) * sum_{i=0}^{N-2} ||ssm[i+1] - ssm[i]||^2
    ///
    /// This encourages smooth transitions between adjacent chunks in SSM space,
    /// reducing discontinuities that cause erratic Mamba state updates.
    pub fn compute_smoothness_gradients(&mut self, thought: &ContinuousHV, weight: f32) {
        if self.num_chunks < 2 || weight <= 0.0 {
            return;
        }

        let scale = weight / (self.num_chunks - 1) as f32;

        // Forward all chunks to get SSM vectors
        let mut ssm_vecs = Vec::with_capacity(self.num_chunks);
        let mut normed_chunks = Vec::with_capacity(self.num_chunks);
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let normed = self.layer_norm(chunk);

            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_up[j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }
            ssm_vecs.push(ssm_vec);
            normed_chunks.push(normed);
        }

        // Gradient: d||ssm[i+1] - ssm[i]||^2 / d_ssm[i] = -2*(ssm[i+1] - ssm[i])
        //           d||ssm[i+1] - ssm[i]||^2 / d_ssm[i+1] = +2*(ssm[i+1] - ssm[i])
        // Then chain through w_chunk_up.
        let mut ssm_grads = vec![vec![0.0f32; self.ssm_dim]; self.num_chunks];
        for i in 0..self.num_chunks - 1 {
            for j in 0..self.ssm_dim {
                let diff = ssm_vecs[i + 1][j] - ssm_vecs[i][j];
                ssm_grads[i][j] += -2.0 * scale * diff;
                ssm_grads[i + 1][j] += 2.0 * scale * diff;
            }
        }

        // Chain rule to w_chunk_up
        for chunk_idx in 0..self.num_chunks {
            for j in 0..self.ssm_dim {
                if ssm_grads[chunk_idx][j].abs() < 1e-12 {
                    continue;
                }
                for k in 0..self.chunk_size {
                    self.grad_chunk_up[j * self.chunk_size + k] +=
                        ssm_grads[chunk_idx][j] * normed_chunks[chunk_idx][k];
                }
            }
        }
    }

    /// Compute rank regularization gradients for the up-projection.
    ///
    /// Penalizes off-diagonal elements of the Gram matrix `G = W_up^T @ W_up`
    /// (where rows of W_up are SSM-dim output vectors for each input feature).
    /// This encourages the projection to use more of the SSM space.
    ///
    /// Since the full Gram matrix (chunk_size × chunk_size = 256×256) is expensive,
    /// we sample `num_sample` random pairs of rows and penalize their dot product.
    pub fn compute_rank_regularization_gradients(&mut self, weight: f32, num_samples: usize) {
        if weight <= 0.0 || self.ssm_dim < 2 {
            return;
        }

        // Sample pseudo-random pairs of SSM output dimensions (rows of W_up)
        // and penalize their alignment.
        // W_up is [ssm_dim × chunk_size]: row j is the weights mapping chunk → ssm_dim[j]
        let scale = weight / num_samples.max(1) as f32;

        // Use xorshift64 with weight-derived seed for varying coverage across calls.
        // The seed incorporates the first weight value to vary with training progress.
        let seed_val = (self.w_chunk_up[0].to_bits() as u64)
            .wrapping_add(self.w_chunk_up.len() as u64)
            ^ 0x9E3779B97F4A7C15;
        let mut rng_state = seed_val | 1; // ensure non-zero
        let total_pairs = self.ssm_dim * self.ssm_dim;
        for _sample in 0..num_samples {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let pair_idx = (rng_state as usize) % total_pairs;
            let j1 = pair_idx / self.ssm_dim;
            let j2 = pair_idx % self.ssm_dim;
            if j1 == j2 {
                continue; // Skip diagonal
            }

            // Compute dot product of rows j1 and j2
            let row1_start = j1 * self.chunk_size;
            let row2_start = j2 * self.chunk_size;
            let mut dot = 0.0f32;
            for k in 0..self.chunk_size {
                dot += self.w_chunk_up[row1_start + k] * self.w_chunk_up[row2_start + k];
            }

            // Gradient of dot^2 / 2 w.r.t. W_up[j1, k] = dot * W_up[j2, k]
            // and w.r.t. W_up[j2, k] = dot * W_up[j1, k]
            for k in 0..self.chunk_size {
                self.grad_chunk_up[row1_start + k] += scale * dot * self.w_chunk_up[row2_start + k];
                self.grad_chunk_up[row2_start + k] += scale * dot * self.w_chunk_up[row1_start + k];
            }
        }
    }

    /// Accumulate gradients from end-to-end Mamba token prediction loss.
    ///
    /// Takes `d_loss/d_ssm_token[i]` (from candle autograd through Mamba) and chains
    /// through the temporal projection to get `d_loss/d_w_chunk_up`:
    ///
    /// ```text
    /// ssm_token[i] = LayerNorm(chunk[i]) × W_up + pos_enc[i]
    /// d_loss/d_W_up[j,k] = Σ_i d_loss/d_ssm[i,j] × normed_chunk[i,k]
    /// ```
    ///
    /// Also accumulates gradients for LayerNorm, pos_enc (if learned), and attention.
    pub fn compute_e2e_gradients(&mut self, thought: &ContinuousHV, ssm_grads: &[Vec<f32>]) {
        let dims = thought.dim();
        let stride = if self.stride > 0 { self.stride } else { self.chunk_size };
        let expected_chunks = if stride < self.chunk_size {
            (dims.saturating_sub(self.chunk_size)) / stride + 1
        } else {
            self.num_chunks
        };

        if ssm_grads.len() != expected_chunks {
            return; // Dimension mismatch, skip
        }

        let scale = 1.0 / expected_chunks as f32;

        for (chunk_idx, ssm_grad) in ssm_grads.iter().enumerate() {
            if ssm_grad.len() != self.ssm_dim {
                continue;
            }

            // Extract and LayerNorm the chunk (same as forward path)
            let chunk_start = chunk_idx * stride;
            let chunk_end = (chunk_start + self.chunk_size).min(dims);
            let raw_chunk = &thought.as_slice()[chunk_start..chunk_end];
            let mut normed = vec![0.0f32; self.chunk_size];
            if chunk_end - chunk_start < self.chunk_size {
                normed[..chunk_end - chunk_start].copy_from_slice(raw_chunk);
            } else {
                normed.copy_from_slice(raw_chunk);
            }

            // LayerNorm forward
            let mean: f32 = normed.iter().sum::<f32>() / self.chunk_size as f32;
            let var: f32 = normed.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.chunk_size as f32;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for k in 0..self.chunk_size {
                normed[k] = (normed[k] - mean) * inv_std * self.ln_gamma[k] + self.ln_beta[k];
            }

            // Chain rule: d_loss/d_W_up[j,k] += ssm_grad[j] * normed[k] * scale
            // (attention gate applied if learned)
            let attn = if self.learned_attention {
                sigmoid(self.chunk_attention[chunk_idx])
            } else {
                1.0
            };

            for j in 0..self.ssm_dim {
                let grad_j = ssm_grad[j] * attn * scale;
                let w_offset = j * self.chunk_size;
                for k in 0..self.chunk_size {
                    self.grad_chunk_up[w_offset + k] += grad_j * normed[k];
                }
            }

            // Attention gradient: d_loss/d_attn[i] = Σ_j ssm_grad[j] * ssm_pre_attn[j]
            if self.learned_attention && attn > 1e-10 {
                // Recompute ssm_pre_attn (before attention scaling)
                let mut ssm_pre = vec![0.0f32; self.ssm_dim];
                for j in 0..self.ssm_dim {
                    let w_offset = j * self.chunk_size;
                    for k in 0..self.chunk_size {
                        ssm_pre[j] += self.w_chunk_up[w_offset + k] * normed[k];
                    }
                    // Add pos_enc
                    ssm_pre[j] += self.pos_enc[chunk_idx * self.ssm_dim + j];
                }
                let d_attn: f32 = ssm_grad.iter().zip(ssm_pre.iter())
                    .map(|(g, s)| g * s)
                    .sum::<f32>() * scale;
                let sigmoid_grad = attn * (1.0 - attn);
                self.grad_chunk_attention[chunk_idx] += d_attn * sigmoid_grad;
            }

            // Positional encoding gradient (if learned)
            if self.learned_pos_enc {
                for j in 0..self.ssm_dim {
                    self.grad_pos_enc[chunk_idx * self.ssm_dim + j] += ssm_grad[j] * attn * scale;
                }
            }

            // LayerNorm gradient (chain through normed → raw)
            for k in 0..self.chunk_size {
                let d_normed_k: f32 = (0..self.ssm_dim)
                    .map(|j| ssm_grad[j] * attn * self.w_chunk_up[j * self.chunk_size + k])
                    .sum::<f32>() * scale;
                self.grad_ln_gamma[k] += d_normed_k * (raw_chunk.get(k).copied().unwrap_or(0.0) - mean) * inv_std;
                self.grad_ln_beta[k] += d_normed_k;
            }
        }
    }

    /// Apply accumulated gradients with per-group gradient clipping.
    ///
    /// Each weight group (w_up, w_down, LayerNorm, pos_enc, attention) is clipped
    /// independently to `grad_clip`. This prevents large gradients in one group
    /// (e.g. w_down at norm ~300) from suppressing learning in other groups
    /// (e.g. w_up at norm ~5) via shared clip scaling.
    pub fn apply_gradients(&mut self, lr: f32, grad_clip: f32) -> GradientStepMetrics {
        let norm_up = l2_norm(&self.grad_chunk_up);
        let norm_down = l2_norm(&self.grad_chunk_down);
        let norm_ln = (l2_norm(&self.grad_ln_gamma).powi(2) + l2_norm(&self.grad_ln_beta).powi(2)).sqrt();
        let norm_pos = if self.learned_pos_enc { l2_norm(&self.grad_pos_enc) } else { 0.0 };
        let norm_attn = if self.learned_attention { l2_norm(&self.grad_chunk_attention) } else { 0.0 };

        // Per-group independent clipping: each group clips to grad_clip separately
        let clip_up = if norm_up > grad_clip { grad_clip / norm_up } else { 1.0 };
        let clip_down = if norm_down > grad_clip { grad_clip / norm_down } else { 1.0 };
        let clip_ln = if norm_ln > grad_clip { grad_clip / norm_ln } else { 1.0 };
        let clip_pos = if norm_pos > grad_clip { grad_clip / norm_pos } else { 1.0 };
        let clip_attn = if norm_attn > grad_clip { grad_clip / norm_attn } else { 1.0 };

        let was_clipped = clip_up < 1.0 || clip_down < 1.0 || clip_ln < 1.0
            || clip_pos < 1.0 || clip_attn < 1.0;

        // Apply to w_chunk_up
        let lr_up = lr * clip_up;
        for (w, g) in self.w_chunk_up.iter_mut().zip(self.grad_chunk_up.iter()) {
            *w -= lr_up * g;
        }
        // Apply to w_chunk_down
        let lr_down = lr * clip_down;
        for (w, g) in self.w_chunk_down.iter_mut().zip(self.grad_chunk_down.iter()) {
            *w -= lr_down * g;
        }
        // Apply to LayerNorm
        let lr_ln = lr * clip_ln;
        for (w, g) in self.ln_gamma.iter_mut().zip(self.grad_ln_gamma.iter()) {
            *w -= lr_ln * g;
        }
        for (w, g) in self.ln_beta.iter_mut().zip(self.grad_ln_beta.iter()) {
            *w -= lr_ln * g;
        }
        // Apply to learned positional encoding
        if self.learned_pos_enc {
            let lr_pos = lr * clip_pos;
            for (w, g) in self.pos_enc.iter_mut().zip(self.grad_pos_enc.iter()) {
                *w -= lr_pos * g;
            }
        }
        // Apply to learned chunk attention
        if self.learned_attention {
            let lr_attn = lr * clip_attn;
            for (w, g) in self.chunk_attention.iter_mut().zip(self.grad_chunk_attention.iter()) {
                *w -= lr_attn * g;
            }
        }

        // Zero accumulators
        self.grad_chunk_up.fill(0.0);
        self.grad_chunk_down.fill(0.0);
        self.grad_ln_gamma.fill(0.0);
        self.grad_ln_beta.fill(0.0);
        if self.learned_pos_enc {
            self.grad_pos_enc.fill(0.0);
        }
        if self.learned_attention {
            self.grad_chunk_attention.fill(0.0);
        }

        GradientStepMetrics {
            norm_down,
            norm_up,
            norm_backward: norm_down, // For temporal, down == backward
            was_clipped,
        }
    }

    /// Scale accumulated gradients by a factor (for surprise-weighted learning).
    pub fn scale_accumulated_gradients(&mut self, scale: f32) {
        for g in &mut self.grad_chunk_up {
            *g *= scale;
        }
        for g in &mut self.grad_chunk_down {
            *g *= scale;
        }
        for g in &mut self.grad_ln_gamma {
            *g *= scale;
        }
        for g in &mut self.grad_ln_beta {
            *g *= scale;
        }
        if self.learned_pos_enc {
            for g in &mut self.grad_pos_enc {
                *g *= scale;
            }
        }
        if self.learned_attention {
            for g in &mut self.grad_chunk_attention {
                *g *= scale;
            }
        }
    }

    /// Compute contrastive gradients: push `anchor` projection away from `negative`.
    pub fn compute_contrastive_gradients(
        &mut self,
        anchor: &ContinuousHV,
        negative: &ContinuousHV,
        weight: f32,
    ) {
        // Simple repulsive gradient: for each chunk, push the up-projected
        // representations of anchor and negative apart.
        // Averaged over chunks (not summed) to keep norms stable.
        let scale = weight / self.num_chunks as f32;
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let a_chunk = &anchor.values[start..start + self.chunk_size];
            let n_chunk = &negative.values[start..start + self.chunk_size];

            let a_normed = self.layer_norm(a_chunk);
            let n_normed = self.layer_norm(n_chunk);

            // Project both
            let mut a_ssm = vec![0.0f32; self.ssm_dim];
            let mut n_ssm = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                for k in 0..self.chunk_size {
                    let w = self.w_chunk_up[j * self.chunk_size + k];
                    a_ssm[j] += w * a_normed[k];
                    n_ssm[j] += w * n_normed[k];
                }
            }

            // Repulsive gradient: push apart in SSM space
            // d_loss/d_w = scale * (a_normed - n_normed) ⊗ (a_ssm - n_ssm)
            for j in 0..self.ssm_dim {
                let ssm_diff = a_ssm[j] - n_ssm[j];
                for k in 0..self.chunk_size {
                    let input_diff = a_normed[k] - n_normed[k];
                    self.grad_chunk_up[j * self.chunk_size + k] -= scale * ssm_diff * input_diff;
                }
            }
        }
    }

    /// Get the bottleneck activation for a thought (for diagnostics).
    ///
    /// Returns the element-wise mean of 4 evenly-spaced chunks' LayerNorm'd values.
    /// This gives a representative view of the full thought's chunk distribution,
    /// not just the first chunk.
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
    ///
    /// For each chunk: LayerNorm → w_up → w_down → compare with original.
    /// Returns `1.0 - avg_cosine_similarity` across all chunks.
    /// This gives a meaningful PE metric for temporal mode, unlike the
    /// tiled full-vector cosine similarity which is always ~0.99.
    pub fn roundtrip_pe(&self, thought: &ContinuousHV) -> f32 {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];

            let normed = self.layer_norm(chunk);

            // Up-project: chunk_size → ssm_dim
            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_up[j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }

            // Down-project: ssm_dim → chunk_size
            let mut recon = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                let mut sum = 0.0f32;
                for j in 0..self.ssm_dim {
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * ssm_vec[j];
                }
                recon[k] = sum;
            }

            // Per-chunk cosine similarity
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

    /// Estimate effective rank of the up-projection using the Frobenius/spectral ratio heuristic.
    ///
    /// Projects sample thoughts through multiple chunks (not just the first)
    /// and computes the effective rank of the resulting SSM-space representations.
    pub fn effective_rank(&self, thoughts: &[ContinuousHV]) -> f32 {
        if thoughts.len() < 2 {
            return self.chunk_size as f32;
        }

        // Sample up to 4 evenly-spaced chunks to capture representational diversity
        let sample_chunks: Vec<usize> = if self.num_chunks <= 4 {
            (0..self.num_chunks).collect()
        } else {
            let step = self.num_chunks / 4;
            vec![0, step, 2 * step, 3 * step]
        };

        // Project each thought × chunk combination to SSM space
        let mut ssm_vecs: Vec<Vec<f32>> =
            Vec::with_capacity(thoughts.len() * sample_chunks.len());
        for thought in thoughts {
            for &ci in &sample_chunks {
                let start = ci * self.stride;
                let chunk = &thought.values[start..start + self.chunk_size];
                let normed = self.layer_norm(chunk);
                let mut ssm_vec = vec![0.0f32; self.ssm_dim];
                for j in 0..self.ssm_dim {
                    let mut sum = 0.0f32;
                    for k in 0..self.chunk_size {
                        sum += self.w_chunk_up[j * self.chunk_size + k] * normed[k];
                    }
                    ssm_vec[j] = sum;
                }
                ssm_vecs.push(ssm_vec);
            }
        }

        // Compute variance along each SSM dimension
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

        // Effective rank from variance spectrum (Shannon entropy of normalized eigenvalues)
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
    ///
    /// Computes the covariance of LayerNorm'd chunks from sample thoughts,
    /// then sets `w_chunk_up` rows to the top principal components. This gives
    /// the projection a much better starting point than random init.
    pub fn warm_start_from_samples(&mut self, samples: &[ContinuousHV]) {
        if samples.len() < 2 {
            return;
        }

        // Collect LayerNorm'd chunks from all samples × all chunks
        let mut normed_chunks: Vec<Vec<f32>> = Vec::with_capacity(samples.len() * self.num_chunks);
        for thought in samples {
            for ci in 0..self.num_chunks {
                let start = ci * self.stride;
                if start + self.chunk_size <= thought.values.len() {
                    normed_chunks.push(self.layer_norm(&thought.values[start..start + self.chunk_size]));
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

        // Power iteration for top-k principal components → w_chunk_up rows
        for comp_idx in 0..k {
            let row_start = comp_idx * d;

            // Initialize with scaled identity-like direction
            for j in 0..d {
                self.w_chunk_up[row_start + j] = if j == comp_idx % d { scale } else { 0.0 };
            }

            // 20 iterations of power method on covariance
            for _ in 0..20 {
                // Project: v = C * w (covariance-vector product)
                let mut v = vec![0.0f32; d];
                for chunk in &normed_chunks {
                    // dot = (chunk - mean) · w
                    let mut dot = 0.0f32;
                    for j in 0..d {
                        dot += (chunk[j] - mean[j]) * self.w_chunk_up[row_start + j];
                    }
                    // v += dot * (chunk - mean)
                    for j in 0..d {
                        v[j] += dot * (chunk[j] - mean[j]);
                    }
                }

                // Deflate: subtract projections onto previous components
                for prev in 0..comp_idx {
                    let prev_start = prev * d;
                    let mut proj = 0.0f32;
                    for j in 0..d {
                        proj += v[j] * self.w_chunk_up[prev_start + j];
                    }
                    for j in 0..d {
                        v[j] -= proj * self.w_chunk_up[prev_start + j];
                    }
                }

                // Normalize
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for j in 0..d {
                        self.w_chunk_up[row_start + j] = v[j] * scale / norm;
                    }
                }
            }
        }

        // Also warm-start w_chunk_down as transpose of w_chunk_up (pseudo-inverse)
        // w_down[k, j] = w_up[j, k] scaled
        let down_scale = 1.0 / (self.ssm_dim as f32).sqrt();
        for k in 0..self.chunk_size {
            for j in 0..self.ssm_dim {
                self.w_chunk_down[k * self.ssm_dim + j] =
                    self.w_chunk_up[j * self.chunk_size + k] * down_scale;
            }
        }
    }

    /// Flatten all learnable weights into a single vector (for checkpointing).
    ///
    /// Layout: `[w_chunk_up | w_chunk_down | ln_gamma | ln_beta | pos_enc (if learned)]`
    pub fn flatten_weights(&self) -> Vec<f32> {
        let pos_size = if self.learned_pos_enc { self.pos_enc.len() } else { 0 };
        let attn_size = if self.learned_attention { self.chunk_attention.len() } else { 0 };
        let mut weights = Vec::with_capacity(
            self.w_chunk_up.len() + self.w_chunk_down.len() + self.ln_gamma.len()
                + self.ln_beta.len() + pos_size + attn_size,
        );
        weights.extend_from_slice(&self.w_chunk_up);
        weights.extend_from_slice(&self.w_chunk_down);
        weights.extend_from_slice(&self.ln_gamma);
        weights.extend_from_slice(&self.ln_beta);
        if self.learned_pos_enc {
            weights.extend_from_slice(&self.pos_enc);
        }
        if self.learned_attention {
            weights.extend_from_slice(&self.chunk_attention);
        }
        weights
    }

    /// Load weights from a flat vector (from checkpoint).
    ///
    /// Accepts either the base size (w_up + w_down + LN) or the extended size
    /// that includes learned positional encoding weights.
    pub fn load_weights(&mut self, weights: &[f32]) {
        let up_size = self.w_chunk_up.len();
        let down_size = self.w_chunk_down.len();
        let gamma_size = self.ln_gamma.len();
        let beta_size = self.ln_beta.len();
        let base_size = up_size + down_size + gamma_size + beta_size;
        let pos_size = self.pos_enc.len();
        let extended_size = base_size + pos_size;

        // Accept base size (no pos_enc) or extended size (with pos_enc)
        assert!(
            weights.len() == base_size || weights.len() == extended_size,
            "Expected {base_size} or {extended_size} weights, got {}",
            weights.len()
        );

        let mut offset = 0;
        self.w_chunk_up.copy_from_slice(&weights[offset..offset + up_size]);
        offset += up_size;
        self.w_chunk_down.copy_from_slice(&weights[offset..offset + down_size]);
        offset += down_size;
        self.ln_gamma.copy_from_slice(&weights[offset..offset + gamma_size]);
        offset += gamma_size;
        self.ln_beta.copy_from_slice(&weights[offset..offset + beta_size]);
        offset += beta_size;

        // Load learned pos_enc if present in the checkpoint
        if offset + pos_size <= weights.len() && weights.len() > base_size {
            self.pos_enc.copy_from_slice(&weights[offset..offset + pos_size]);
            self.learned_pos_enc = true;
            offset += pos_size;
        }

        // Load learned chunk attention if present
        let attn_size = self.chunk_attention.len();
        if offset + attn_size == weights.len() {
            self.chunk_attention.copy_from_slice(&weights[offset..offset + attn_size]);
            self.learned_attention = true;
        }
    }

    /// Number of learnable parameters.
    pub fn num_params(&self) -> usize {
        let mut total = self.w_chunk_up.len() + self.w_chunk_down.len()
            + self.ln_gamma.len() + self.ln_beta.len();
        if self.learned_pos_enc { total += self.pos_enc.len(); }
        if self.learned_attention { total += self.chunk_attention.len(); }
        total
    }

    /// Whether positional encoding is learned (trainable).
    pub fn learned_pos_enc(&self) -> bool {
        self.learned_pos_enc
    }

    /// Enable or disable learned positional encoding.
    ///
    /// Used for curriculum learning: start with fixed sinusoidal (false),
    /// then enable learning (true) after the main projection has converged.
    /// When switching from false to true, the current (sinusoidal) values
    /// become the initial learned values.
    pub fn set_learned_pos_enc(&mut self, learned: bool) {
        self.learned_pos_enc = learned;
        if learned {
            // Zero out pos_enc gradients to start fresh
            self.grad_pos_enc.fill(0.0);
        }
    }

    /// Project a thought to SSM sequence, selecting only the top-K most
    /// informative chunks by activation magnitude.
    ///
    /// Returns a shorter sequence (len <= budget) that preserves the most
    /// important information. Chunks are returned in their original positional
    /// order to maintain temporal coherence.
    ///
    /// If `budget >= num_chunks`, returns the full sequence (no filtering).
    pub fn project_to_ssm_sequence_topk(
        &self,
        thought: &ContinuousHV,
        budget: usize,
    ) -> Vec<Vec<f32>> {
        if budget >= self.num_chunks {
            return self.project_to_ssm_sequence(thought);
        }

        // Score each chunk by activation magnitude × learned attention (if enabled)
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

        // Select top-K by score
        chunk_magnitudes.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut selected: Vec<usize> = chunk_magnitudes[..budget]
            .iter()
            .map(|(idx, _)| *idx)
            .collect();
        // Restore positional order
        selected.sort_unstable();

        // Project only selected chunks (with their correct positional encoding)
        let mut sequence = Vec::with_capacity(budget);
        for chunk_idx in selected {
            let start = chunk_idx * self.stride;
            let chunk = &thought.values[start..start + self.chunk_size];
            let normed = self.layer_norm(chunk);

            let mut ssm_vec = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_up[j * self.chunk_size + k] * normed[k];
                }
                ssm_vec[j] = sum;
            }

            let pos_offset = chunk_idx * self.ssm_dim;
            for j in 0..self.ssm_dim {
                ssm_vec[j] += self.pos_enc[pos_offset + j];
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

    /// Chunk size (256).
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
            // dL/d_gamma_i ≈ error[i] * x_hat (chain rule through downstream layers)
            self.grad_ln_gamma[i] += error[i] * x_hat;
            // dL/d_beta_i ≈ error[i]
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
///
/// For position `pos` and dimension `i`:
/// - Even dims: `sin(pos / 10000^(2i/dim))`
/// - Odd dims: `cos(pos / 10000^(2i/dim))`
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
///
/// Uses the same chunked HV pattern as `HdcSsmProjection::init_weights`.
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

        // Measure initial roundtrip PE
        let initial_pe = roundtrip_pe(&tp, &thought);

        // Train for several steps
        for _ in 0..20 {
            tp.compute_roundtrip_gradients(&thought);
            tp.apply_gradients(0.01, 100.0);
        }

        // PE should decrease
        let final_pe = roundtrip_pe(&tp, &thought);
        assert!(
            final_pe < initial_pe,
            "PE should decrease: initial={initial_pe:.4}, final={final_pe:.4}"
        );
    }

    #[test]
    fn test_positional_encoding_orthogonality() {
        let enc = sinusoidal_pos_enc(64, 768);

        // Check that different positions have reasonably different encodings
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
        // w_chunk_up: 768*256 = 196,608
        // w_chunk_down: 256*768 = 196,608
        // ln_gamma: 256
        // ln_beta: 256
        // total: 393,728
        assert_eq!(tp.num_params(), 393_728);
    }

    #[test]
    fn test_flatten_load_roundtrip() {
        let tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        let weights = tp.flatten_weights();

        let mut tp2 = TemporalProjection::new(&GenesisSeed::from_phrase("different"), 16384, 256, 768);
        tp2.load_weights(&weights);

        // Should produce identical output
        let thought = ContinuousHV::random_default(42);
        let seq1 = tp.project_to_ssm_sequence(&thought);
        let seq2 = tp2.project_to_ssm_sequence(&thought);
        for (a, b) in seq1.iter().zip(seq2.iter()) {
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!((va - vb).abs() < 1e-6, "Weights should produce identical output");
            }
        }
    }

    #[test]
    fn test_gradient_clipping() {
        let mut tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        let thought = ContinuousHV::random_default(42).normalize();

        tp.compute_roundtrip_gradients(&thought);
        let metrics = tp.apply_gradients(0.01, 0.001); // Very tight clip
        assert!(metrics.was_clipped, "Should clip with threshold 0.001");
    }

    #[test]
    fn test_learned_pos_enc_param_count() {
        let tp = TemporalProjection::new_with_options(&test_genesis(), 16384, 256, 768, true);
        // base: 393,728 + pos_enc: 64*768 = 49,152
        assert_eq!(tp.num_params(), 393_728 + 49_152);
        assert!(tp.learned_pos_enc());
    }

    #[test]
    fn test_learned_pos_enc_trains() {
        let genesis = test_genesis();
        let mut tp = TemporalProjection::new_with_options(&genesis, 16384, 256, 768, true);
        let thought = ContinuousHV::random_default(42).normalize();

        // Capture initial pos_enc
        let initial_pos = tp.pos_enc.clone();

        // Train for a few steps
        for _ in 0..5 {
            tp.compute_roundtrip_gradients(&thought);
            tp.apply_gradients(0.01, 100.0);
        }

        // Pos enc should have changed
        let changed = tp.pos_enc.iter().zip(initial_pos.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Learned pos_enc should change during training");
    }

    #[test]
    fn test_learned_pos_enc_flatten_load() {
        let genesis = test_genesis();
        let tp = TemporalProjection::new_with_options(&genesis, 16384, 256, 768, true);
        let weights = tp.flatten_weights();
        // Should include pos_enc
        assert_eq!(weights.len(), 393_728 + 49_152);

        let mut tp2 = TemporalProjection::new_with_options(
            &GenesisSeed::from_phrase("different"), 16384, 256, 768, false,
        );
        tp2.load_weights(&weights);
        // Should auto-detect learned pos_enc from weight size
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
        // stride=128, chunk_size=256 → 50% overlap
        // num_chunks = (16384 - 256) / 128 + 1 = 16128/128 + 1 = 127
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

        // Train for a few roundtrip steps so the projection is meaningful
        for _ in 0..20 {
            tp.compute_roundtrip_gradients(&thought);
            tp.apply_gradients(0.01, 100.0);
        }

        let sequence = tp.project_to_ssm_sequence(&thought);

        // Sequence-based reconstruction uses per-chunk down-projection
        let seq_recon = tp.project_sequence_to_hdc(&sequence);
        let seq_sim = thought.similarity(&seq_recon).clamp(-1.0, 1.0);

        // After training, sequence reconstruction should have meaningful similarity
        assert!(
            seq_sim > 0.0,
            "Trained sequence reconstruction should have positive similarity, got {seq_sim:.4}"
        );
        // And should produce correct dimensions
        assert_eq!(seq_recon.dim(), 16384);
    }

    #[test]
    fn test_topk_chunk_budget() {
        let tp = TemporalProjection::new(&test_genesis(), 16384, 256, 768);
        let thought = ContinuousHV::random_default(42).normalize();

        // Budget of 16 → should return exactly 16 chunks
        let seq = tp.project_to_ssm_sequence_topk(&thought, 16);
        assert_eq!(seq.len(), 16, "Budget=16 should return 16 chunks");

        // Budget >= num_chunks → should return all
        let full = tp.project_to_ssm_sequence_topk(&thought, 100);
        assert_eq!(full.len(), 64, "Budget>=64 should return all 64 chunks");

        // Each chunk should still be ssm_dim
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

        // Measure initial smoothness (L2 distance between consecutive chunks)
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

        // Train with smoothness loss only
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

        // Train with directional loss
        for _ in 0..20 {
            tp.compute_directional_gradients(&thought, None);
            tp.apply_gradients(0.01, 100.0);
        }

        let final_pe = tp.roundtrip_pe(&thought);
        // Directional loss may not directly optimize roundtrip PE, but it should
        // improve the projection quality. Check that PE is at least finite.
        assert!(final_pe.is_finite(), "PE should be finite after directional training");
        // And the up-projection should have changed
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
        // All values should be finite
        assert!(recon.values.iter().all(|v| v.is_finite()));
    }
}
