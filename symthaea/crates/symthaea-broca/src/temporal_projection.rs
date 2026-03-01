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
    chunk_size: usize,  // 256 (= hdc_dim / num_chunks)
    num_chunks: usize,  // 64
    w_chunk_up: Vec<f32>,   // [ssm_dim × chunk_size] = [768 × 256]
    w_chunk_down: Vec<f32>, // [chunk_size × ssm_dim] = [256 × 768]
    ln_gamma: Vec<f32>,     // [chunk_size] LayerNorm scale
    ln_beta: Vec<f32>,      // [chunk_size] LayerNorm bias
    pos_enc: Vec<f32>,      // [num_chunks × ssm_dim] sinusoidal positional encoding (fixed)
    // Gradient accumulators
    grad_chunk_up: Vec<f32>,
    grad_chunk_down: Vec<f32>,
    grad_ln_gamma: Vec<f32>,
    grad_ln_beta: Vec<f32>,
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
        assert!(
            hdc_dim % chunk_dim == 0,
            "hdc_dim ({hdc_dim}) must be divisible by chunk_dim ({chunk_dim})"
        );
        let num_chunks = hdc_dim / chunk_dim;

        // JL-style initialization: scale = 1/sqrt(chunk_dim)
        let scale = 1.0 / (chunk_dim as f32).sqrt();

        let up_size = ssm_dim * chunk_dim;
        let down_size = chunk_dim * ssm_dim;

        let w_chunk_up = init_weights(genesis, "temporal::w_chunk_up", up_size, scale);
        let w_chunk_down = init_weights(genesis, "temporal::w_chunk_down", down_size, scale);

        // LayerNorm: gamma=1, beta=0 (standard initialization)
        let ln_gamma = vec![1.0f32; chunk_dim];
        let ln_beta = vec![0.0f32; chunk_dim];

        // Sinusoidal positional encoding (fixed, not learned)
        let pos_enc = sinusoidal_pos_enc(num_chunks, ssm_dim);

        Self {
            chunk_size: chunk_dim,
            num_chunks,
            w_chunk_up,
            w_chunk_down,
            ln_gamma,
            ln_beta,
            pos_enc,
            grad_chunk_up: vec![0.0; up_size],
            grad_chunk_down: vec![0.0; down_size],
            grad_ln_gamma: vec![0.0; chunk_dim],
            grad_ln_beta: vec![0.0; chunk_dim],
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
            let start = chunk_idx * self.chunk_size;
            let chunk = &thought.values[start..start + self.chunk_size];

            // LayerNorm
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

            // Add positional encoding
            let pos_offset = chunk_idx * self.ssm_dim;
            for j in 0..self.ssm_dim {
                ssm_vec[j] += self.pos_enc[pos_offset + j];
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

        // Tile the reconstructed chunk across the full HDC dimension
        let mut values = Vec::with_capacity(self.hdc_dim);
        for _ in 0..self.num_chunks {
            values.extend_from_slice(&chunk_recon);
        }

        ContinuousHV::from_vec(values)
    }

    /// Compute roundtrip autoencoder gradients: thought → forward → backward → error.
    ///
    /// Loss: MSE between original chunk and reconstructed chunk for each of the 64 chunks.
    /// Gradients accumulate into `grad_chunk_up` and `grad_chunk_down`.
    pub fn compute_roundtrip_gradients(&mut self, thought: &ContinuousHV) {
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.chunk_size;
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

            // Backward: down-project
            let mut recon = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                let mut sum = 0.0f32;
                for j in 0..self.ssm_dim {
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * ssm_vec[j];
                }
                recon[k] = sum;
            }

            // Error: original - reconstruction
            let mut error = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                error[k] = normed[k] - recon[k];
            }

            // Gradient for w_chunk_down: d_loss/d_w_down[k,j] = -2 * error[k] * ssm_vec[j]
            for k in 0..self.chunk_size {
                for j in 0..self.ssm_dim {
                    self.grad_chunk_down[k * self.ssm_dim + j] += -2.0 * error[k] * ssm_vec[j];
                }
            }

            // Backprop through down-projection to get ssm_error
            let mut ssm_error = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * (-2.0 * error[k]);
                }
                ssm_error[j] = sum;
            }

            // Gradient for w_chunk_up: d_loss/d_w_up[j,k] = ssm_error[j] * normed[k]
            for j in 0..self.ssm_dim {
                for k in 0..self.chunk_size {
                    self.grad_chunk_up[j * self.chunk_size + k] += ssm_error[j] * normed[k];
                }
            }

            // Gradient for LayerNorm gamma/beta (simplified)
            self.accumulate_ln_gradients(chunk, &normed, &error);
        }
    }

    /// Compute gradients from reconstruction loss: thought vs target.
    ///
    /// Used when Mamba output is meaningful (PE < 0.5). The target is the
    /// attention-weighted bundle of back-projected output tokens.
    pub fn compute_gradients(&mut self, thought: &ContinuousHV, target: &ContinuousHV) {
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.chunk_size;
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

            // Backward: down-project
            let mut recon = vec![0.0f32; self.chunk_size];
            for k in 0..self.chunk_size {
                let mut sum = 0.0f32;
                for j in 0..self.ssm_dim {
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * ssm_vec[j];
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
                    self.grad_chunk_down[k * self.ssm_dim + j] += -2.0 * error[k] * ssm_vec[j];
                }
            }

            // Backprop through down-projection
            let mut ssm_error = vec![0.0f32; self.ssm_dim];
            for j in 0..self.ssm_dim {
                let mut sum = 0.0f32;
                for k in 0..self.chunk_size {
                    sum += self.w_chunk_down[k * self.ssm_dim + j] * (-2.0 * error[k]);
                }
                ssm_error[j] = sum;
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

    /// Apply accumulated gradients with learning rate and gradient clipping.
    pub fn apply_gradients(&mut self, lr: f32, grad_clip: f32) -> GradientStepMetrics {
        let norm_up = l2_norm(&self.grad_chunk_up);
        let norm_down = l2_norm(&self.grad_chunk_down);
        let norm_ln = (l2_norm(&self.grad_ln_gamma).powi(2) + l2_norm(&self.grad_ln_beta).powi(2)).sqrt();
        let combined_norm = (norm_up.powi(2) + norm_down.powi(2) + norm_ln.powi(2)).sqrt();

        let was_clipped = combined_norm > grad_clip;
        let clip_scale = if was_clipped {
            grad_clip / combined_norm
        } else {
            1.0
        };

        let effective_lr = lr * clip_scale;

        // Apply to w_chunk_up
        for (w, g) in self.w_chunk_up.iter_mut().zip(self.grad_chunk_up.iter()) {
            *w -= effective_lr * g;
        }
        // Apply to w_chunk_down
        for (w, g) in self.w_chunk_down.iter_mut().zip(self.grad_chunk_down.iter()) {
            *w -= effective_lr * g;
        }
        // Apply to LayerNorm
        for (w, g) in self.ln_gamma.iter_mut().zip(self.grad_ln_gamma.iter()) {
            *w -= effective_lr * g;
        }
        for (w, g) in self.ln_beta.iter_mut().zip(self.grad_ln_beta.iter()) {
            *w -= effective_lr * g;
        }

        // Zero accumulators
        self.grad_chunk_up.fill(0.0);
        self.grad_chunk_down.fill(0.0);
        self.grad_ln_gamma.fill(0.0);
        self.grad_ln_beta.fill(0.0);

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
    }

    /// Compute contrastive gradients: push `anchor` projection away from `negative`.
    pub fn compute_contrastive_gradients(
        &mut self,
        anchor: &ContinuousHV,
        negative: &ContinuousHV,
        weight: f32,
    ) {
        // Simple repulsive gradient: for each chunk, push the up-projected
        // representations of anchor and negative apart
        for chunk_idx in 0..self.num_chunks {
            let start = chunk_idx * self.chunk_size;
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
            // d_loss/d_w = weight * (a_normed - n_normed) ⊗ (a_ssm - n_ssm)
            for j in 0..self.ssm_dim {
                let ssm_diff = a_ssm[j] - n_ssm[j];
                for k in 0..self.chunk_size {
                    let input_diff = a_normed[k] - n_normed[k];
                    self.grad_chunk_up[j * self.chunk_size + k] -= weight * ssm_diff * input_diff;
                }
            }
        }
    }

    /// Get the bottleneck activation for a thought (for diagnostics).
    ///
    /// Returns the first chunk's LayerNorm'd values as a diagnostic proxy.
    pub fn bottleneck_activation(&self, thought: &ContinuousHV) -> Vec<f32> {
        let chunk = &thought.values[..self.chunk_size];
        self.layer_norm(chunk)
    }

    /// Estimate effective rank of the up-projection using the Frobenius/spectral ratio heuristic.
    ///
    /// Projects a set of sample thoughts and computes the effective rank of the
    /// resulting SSM-space representations.
    pub fn effective_rank(&self, thoughts: &[ContinuousHV]) -> f32 {
        if thoughts.len() < 2 {
            return self.chunk_size as f32;
        }

        // Project each thought's first chunk to SSM space
        let mut ssm_vecs: Vec<Vec<f32>> = Vec::with_capacity(thoughts.len());
        for thought in thoughts {
            let chunk = &thought.values[..self.chunk_size];
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

    /// Flatten all learnable weights into a single vector (for checkpointing).
    pub fn flatten_weights(&self) -> Vec<f32> {
        let mut weights = Vec::with_capacity(
            self.w_chunk_up.len() + self.w_chunk_down.len() + self.ln_gamma.len() + self.ln_beta.len(),
        );
        weights.extend_from_slice(&self.w_chunk_up);
        weights.extend_from_slice(&self.w_chunk_down);
        weights.extend_from_slice(&self.ln_gamma);
        weights.extend_from_slice(&self.ln_beta);
        weights
    }

    /// Load weights from a flat vector (from checkpoint).
    pub fn load_weights(&mut self, weights: &[f32]) {
        let up_size = self.w_chunk_up.len();
        let down_size = self.w_chunk_down.len();
        let gamma_size = self.ln_gamma.len();
        let beta_size = self.ln_beta.len();
        let expected = up_size + down_size + gamma_size + beta_size;
        assert_eq!(
            weights.len(),
            expected,
            "Expected {expected} weights, got {}",
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
    }

    /// Number of learnable parameters.
    pub fn num_params(&self) -> usize {
        self.w_chunk_up.len() + self.w_chunk_down.len() + self.ln_gamma.len() + self.ln_beta.len()
    }

    /// Chunk size (256).
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Number of chunks (64).
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
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
            .field("num_chunks", &self.num_chunks)
            .field("ssm_dim", &self.ssm_dim)
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

    /// Helper: compute roundtrip PE for a single thought.
    fn roundtrip_pe(tp: &TemporalProjection, thought: &ContinuousHV) -> f32 {
        let sequence = tp.project_to_ssm_sequence(thought);
        // Use last chunk's SSM output for back-projection
        let ssm_hidden = sequence.last().unwrap();
        let recon = tp.project_to_hdc(ssm_hidden);
        1.0 - thought.similarity(&recon).clamp(-1.0, 1.0)
    }
}
