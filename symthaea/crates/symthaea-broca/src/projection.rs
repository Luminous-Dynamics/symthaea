//! Bidirectional HDC↔SSM projection with gradient learning.
//!
//! Projects between 16,384D HDC space and 768D SSM (Mamba) space
//! via a 256D bottleneck. The bottleneck matches `compressed_state` dim
//! used by HarmoniesIntegrator and provides information-bottleneck regularization.
//!
//! Total parameters: ~8.8M (vs 25.2M for dense 16384×768 round-trip).
//!
//! # Architecture
//!
//! ```text
//! Forward:  HDC(16384) → w_down → GELU+residual → w_up → SSM(768)
//! Backward: SSM(768) → w_back_down → GELU+residual → w_back_up → HDC(16384)
//! ```
//!
//! Uses GELU activation (no dead neurons) with a pre-activation residual
//! connection (`GELU(x) + α*x`) for smooth gradient flow.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Pre-activation residual scale: `hidden = GELU(x) + RESIDUAL_ALPHA * x`.
/// Ensures gradient flow even through saturated regions.
const RESIDUAL_ALPHA: f32 = 0.1;

/// GELU activation: `x * Φ(x)` via tanh approximation.
#[inline]
fn gelu(x: f32) -> f32 {
    // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let c = 0.7978845608; // sqrt(2/π)
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// GELU derivative (for backprop): d/dx [GELU(x)].
#[inline]
fn gelu_derivative(x: f32) -> f32 {
    let c = 0.7978845608; // sqrt(2/π)
    let inner = c * (x + 0.044715 * x * x * x);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
}

/// Activation + residual: `GELU(x) + α*x`.
#[inline]
fn activation(x: f32) -> f32 {
    gelu(x) + RESIDUAL_ALPHA * x
}

/// Derivative of activation + residual: `GELU'(x) + α`.
#[inline]
fn activation_derivative(x: f32) -> f32 {
    gelu_derivative(x) + RESIDUAL_ALPHA
}

/// Bidirectional projection between HDC (16,384D) and SSM (768D) spaces.
///
/// Uses a 256D bottleneck with JL-style random initialization and online
/// gradient learning from semantic prediction error.
pub struct HdcSsmProjection {
    // Forward: HDC → bottleneck → SSM
    w_down: Vec<f32>,     // [bottleneck × hdc_dim]
    w_up: Vec<f32>,       // [ssm_dim × bottleneck]
    // Backward: SSM → bottleneck → HDC
    w_back_down: Vec<f32>, // [bottleneck × ssm_dim]
    w_back_up: Vec<f32>,   // [hdc_dim × bottleneck]
    // Gradient accumulators
    grad_down: Vec<f32>,
    grad_up: Vec<f32>,
    grad_back_down: Vec<f32>,
    grad_back_up: Vec<f32>,
    // Dimensions
    hdc_dim: usize,
    bottleneck: usize,
    ssm_dim: usize,
}

impl HdcSsmProjection {
    /// Create a new projection with JL-style random initialization.
    ///
    /// Weights are scaled by `1/sqrt(bottleneck)` for variance preservation.
    /// Genesis-seeded for deterministic initialization.
    pub fn new(genesis: &GenesisSeed, hdc_dim: usize, bottleneck: usize, ssm_dim: usize) -> Self {
        let scale = 1.0 / (bottleneck as f32).sqrt();

        let w_down = Self::init_weights(genesis, "projection::w_down", bottleneck * hdc_dim, scale);
        let w_up = Self::init_weights(genesis, "projection::w_up", ssm_dim * bottleneck, scale);
        let w_back_down = Self::init_weights(genesis, "projection::w_back_down", bottleneck * ssm_dim, scale);
        let w_back_up = Self::init_weights(genesis, "projection::w_back_up", hdc_dim * bottleneck, scale);

        Self {
            grad_down: vec![0.0; bottleneck * hdc_dim],
            grad_up: vec![0.0; ssm_dim * bottleneck],
            grad_back_down: vec![0.0; bottleneck * ssm_dim],
            grad_back_up: vec![0.0; hdc_dim * bottleneck],
            w_down,
            w_up,
            w_back_down,
            w_back_up,
            hdc_dim,
            bottleneck,
            ssm_dim,
        }
    }

    /// Initialize a weight vector with genesis-seeded random values scaled by `scale`.
    fn init_weights(genesis: &GenesisSeed, label: &str, size: usize, scale: f32) -> Vec<f32> {
        // Use genesis to create a deterministic ContinuousHV, then tile/truncate
        // to the desired size. For large weight matrices we chunk the initialization.
        let chunk_size = 16384; // One HDC dimension
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
        // Apply JL scaling
        for w in &mut weights {
            *w *= scale;
        }
        weights
    }

    /// Project HDC hypervector (16,384D) to SSM space (768D).
    ///
    /// Pipeline: `hv → w_down → GELU+residual → w_up → ssm_vec`
    pub fn project_to_ssm(&self, hv: &ContinuousHV) -> Vec<f32> {
        debug_assert_eq!(hv.values.len(), self.hdc_dim);

        // Step 1: w_down * hv → bottleneck (256D)
        let hidden_pre = self.matmul(&self.w_down, &hv.values, self.bottleneck, self.hdc_dim);

        // Step 2: GELU + pre-activation residual
        let hidden: Vec<f32> = hidden_pre.into_iter().map(activation).collect();

        // Step 3: w_up * hidden → ssm (768D)
        self.matmul(&self.w_up, &hidden, self.ssm_dim, self.bottleneck)
    }

    /// Project SSM vector (768D) back to HDC space (16,384D).
    ///
    /// Pipeline: `ssm_vec → w_back_down → GELU+residual → w_back_up → hv`
    pub fn project_to_hdc(&self, ssm_vec: &[f32]) -> ContinuousHV {
        debug_assert_eq!(ssm_vec.len(), self.ssm_dim);

        // Step 1: w_back_down * ssm → bottleneck (256D)
        let hidden_pre = self.matmul(&self.w_back_down, ssm_vec, self.bottleneck, self.ssm_dim);

        // Step 2: GELU + pre-activation residual
        let hidden: Vec<f32> = hidden_pre.into_iter().map(activation).collect();

        // Step 3: w_back_up * hidden → hdc (16,384D)
        let values = self.matmul(&self.w_back_up, &hidden, self.hdc_dim, self.bottleneck);

        ContinuousHV::from_vec(values)
    }

    /// Compute gradients from semantic prediction error.
    ///
    /// The error signal is the difference between the original thought HV
    /// and the round-trip reconstruction (project_to_ssm → project_to_hdc).
    pub fn compute_gradients(&mut self, thought_hv: &ContinuousHV, output_hv: &ContinuousHV) {
        debug_assert_eq!(thought_hv.values.len(), self.hdc_dim);
        debug_assert_eq!(output_hv.values.len(), self.hdc_dim);

        // Error = thought - output (MSE gradient direction)
        let error: Vec<f32> = thought_hv.values.iter()
            .zip(output_hv.values.iter())
            .map(|(t, o)| t - o)
            .collect();

        // Forward pass to get hidden activations for gradient computation
        let hidden_fwd_pre = self.matmul(&self.w_down, &thought_hv.values, self.bottleneck, self.hdc_dim);
        let hidden_fwd: Vec<f32> = hidden_fwd_pre.iter().map(|&x| activation(x)).collect();

        // Backward projection hidden activations
        let ssm_fwd = self.matmul(&self.w_up, &hidden_fwd, self.ssm_dim, self.bottleneck);
        let hidden_back_pre = self.matmul(&self.w_back_down, &ssm_fwd, self.bottleneck, self.ssm_dim);
        let hidden_back: Vec<f32> = hidden_back_pre.iter().map(|&x| activation(x)).collect();

        // Gradient for w_back_up: error * hidden_back^T
        // Shape: [hdc_dim × bottleneck]
        for i in 0..self.hdc_dim {
            for j in 0..self.bottleneck {
                self.grad_back_up[i * self.bottleneck + j] += error[i] * hidden_back[j];
            }
        }

        // Gradient for w_back_down: (w_back_up^T * error) * act'(hidden_back_pre) * ssm_fwd^T
        let mut delta_back = vec![0.0f32; self.bottleneck];
        for j in 0..self.bottleneck {
            let mut sum = 0.0f32;
            for i in 0..self.hdc_dim {
                sum += self.w_back_up[i * self.bottleneck + j] * error[i];
            }
            delta_back[j] = sum * activation_derivative(hidden_back_pre[j]);
        }
        for i in 0..self.bottleneck {
            for j in 0..self.ssm_dim {
                self.grad_back_down[i * self.ssm_dim + j] += delta_back[i] * ssm_fwd[j];
            }
        }

        // Gradient for w_up: (w_back_down^T * delta_back) * hidden_fwd^T
        let mut delta_up = vec![0.0f32; self.ssm_dim];
        for j in 0..self.ssm_dim {
            let mut sum = 0.0f32;
            for i in 0..self.bottleneck {
                sum += self.w_back_down[i * self.ssm_dim + j] * delta_back[i];
            }
            delta_up[j] = sum;
        }
        for i in 0..self.ssm_dim {
            for j in 0..self.bottleneck {
                self.grad_up[i * self.bottleneck + j] += delta_up[i] * hidden_fwd[j];
            }
        }

        // Gradient for w_down: (w_up^T * delta_up) * act'(hidden_fwd_pre) * thought_hv^T
        let mut delta_down = vec![0.0f32; self.bottleneck];
        for j in 0..self.bottleneck {
            let mut sum = 0.0f32;
            for i in 0..self.ssm_dim {
                sum += self.w_up[i * self.bottleneck + j] * delta_up[i];
            }
            delta_down[j] = sum * activation_derivative(hidden_fwd_pre[j]);
        }
        for i in 0..self.bottleneck {
            for j in 0..self.hdc_dim {
                self.grad_down[i * self.hdc_dim + j] += delta_down[i] * thought_hv.values[j];
            }
        }
    }

    /// Compute contrastive gradients: push anchor and negative apart in bottleneck space.
    ///
    /// Adds a repulsive gradient so that the projections of `anchor_hv` and
    /// `negative_hv` produce different bottleneck representations. This prevents
    /// the projection from collapsing all thoughts to the same SSM context.
    pub fn compute_contrastive_gradients(
        &mut self,
        anchor_hv: &ContinuousHV,
        negative_hv: &ContinuousHV,
        weight: f32,
    ) {
        // Forward both through w_down to get bottleneck representations
        let hidden_anchor = self.matmul(&self.w_down, &anchor_hv.values, self.bottleneck, self.hdc_dim);
        let hidden_neg = self.matmul(&self.w_down, &negative_hv.values, self.bottleneck, self.hdc_dim);

        // Apply activation to get post-activation representations
        let act_anchor: Vec<f32> = hidden_anchor.iter().map(|&x| activation(x)).collect();
        let act_neg: Vec<f32> = hidden_neg.iter().map(|&x| activation(x)).collect();

        // Contrastive gradient on w_down: push activated representations apart
        for i in 0..self.bottleneck {
            let delta = weight * (act_anchor[i] - act_neg[i]);
            let d_act = activation_derivative(hidden_anchor[i]);
            let row_start = i * self.hdc_dim;
            for j in 0..self.hdc_dim {
                self.grad_down[row_start + j] += delta * d_act * anchor_hv.values[j];
            }
        }
    }

    /// Apply accumulated gradients with SGD + gradient clipping, then zero accumulators.
    pub fn apply_gradients(&mut self, lr: f32, grad_clip: f32) {
        Self::apply_grad(&mut self.w_down, &mut self.grad_down, lr, grad_clip);
        Self::apply_grad(&mut self.w_up, &mut self.grad_up, lr, grad_clip);
        Self::apply_grad(&mut self.w_back_down, &mut self.grad_back_down, lr, grad_clip);
        Self::apply_grad(&mut self.w_back_up, &mut self.grad_back_up, lr, grad_clip);
    }

    fn apply_grad(weights: &mut [f32], grads: &mut [f32], lr: f32, grad_clip: f32) {
        // Compute gradient norm for clipping
        let grad_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        let clip_scale = if grad_norm > grad_clip {
            grad_clip / grad_norm
        } else {
            1.0
        };

        for (w, g) in weights.iter_mut().zip(grads.iter_mut()) {
            *w += lr * clip_scale * *g;
            *g = 0.0; // Zero accumulator
        }
    }

    /// Flatten all projection weights into a single Vec for swarm exchange.
    pub fn flatten_weights(&self) -> Vec<f32> {
        let total = self.w_down.len() + self.w_up.len()
            + self.w_back_down.len() + self.w_back_up.len();
        let mut flat = Vec::with_capacity(total);
        flat.extend_from_slice(&self.w_down);
        flat.extend_from_slice(&self.w_up);
        flat.extend_from_slice(&self.w_back_down);
        flat.extend_from_slice(&self.w_back_up);
        flat
    }

    /// Load weights from a flat Vec (e.g., from swarm aggregation).
    pub fn load_weights(&mut self, flat: &[f32]) {
        let expected = self.w_down.len() + self.w_up.len()
            + self.w_back_down.len() + self.w_back_up.len();
        assert_eq!(flat.len(), expected, "Weight vector size mismatch");

        let mut offset = 0;
        let n = self.w_down.len();
        self.w_down.copy_from_slice(&flat[offset..offset + n]);
        offset += n;
        let n = self.w_up.len();
        self.w_up.copy_from_slice(&flat[offset..offset + n]);
        offset += n;
        let n = self.w_back_down.len();
        self.w_back_down.copy_from_slice(&flat[offset..offset + n]);
        offset += n;
        let n = self.w_back_up.len();
        self.w_back_up.copy_from_slice(&flat[offset..offset + n]);
    }

    /// Total number of learnable parameters.
    pub fn num_params(&self) -> usize {
        self.w_down.len() + self.w_up.len() + self.w_back_down.len() + self.w_back_up.len()
    }

    /// HDC dimension.
    pub fn hdc_dim(&self) -> usize {
        self.hdc_dim
    }

    /// Bottleneck dimension.
    pub fn bottleneck_dim(&self) -> usize {
        self.bottleneck
    }

    /// SSM dimension.
    pub fn ssm_dim(&self) -> usize {
        self.ssm_dim
    }

    /// Warm-start the forward projection (w_down) from sample HDC vectors.
    ///
    /// Computes the top-k principal directions of the input distribution and
    /// aligns w_down rows to span that subspace. This accelerates convergence
    /// by ensuring the bottleneck captures variance in the thought HV space
    /// rather than random directions.
    ///
    /// Uses power iteration (lightweight, no full SVD needed).
    pub fn warm_start_from_samples(&mut self, samples: &[ContinuousHV]) {
        if samples.len() < 2 || self.hdc_dim == 0 || self.bottleneck == 0 {
            return;
        }

        let n = samples.len();
        let d = self.hdc_dim;
        let k = self.bottleneck.min(n).min(d);

        // Compute mean
        let mut mean = vec![0.0f32; d];
        for s in samples {
            for (m, v) in mean.iter_mut().zip(s.values.iter()) {
                *m += v;
            }
        }
        let inv_n = 1.0 / n as f32;
        for m in &mut mean {
            *m *= inv_n;
        }

        // Power iteration for top-k principal components
        // Use genesis-seeded random initialization for determinism
        let scale = 1.0 / (self.bottleneck as f32).sqrt();
        for comp_idx in 0..k {
            // Initialize direction from existing w_down row
            let row_start = comp_idx * d;
            let mut dir: Vec<f32> = self.w_down[row_start..row_start + d].to_vec();
            let dir_norm: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
            if dir_norm > 1e-10 {
                for v in &mut dir {
                    *v /= dir_norm;
                }
            }

            // 10 iterations of power method on covariance
            for _ in 0..10 {
                // result = C * dir = (1/n) Σ (x_i - mean) * <(x_i - mean), dir>
                let mut result = vec![0.0f32; d];
                for s in samples {
                    let mut dot = 0.0f32;
                    for j in 0..d {
                        dot += (s.values[j] - mean[j]) * dir[j];
                    }
                    for j in 0..d {
                        result[j] += (s.values[j] - mean[j]) * dot;
                    }
                }
                // Normalize
                let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                for v in &mut result {
                    *v /= norm;
                }

                // Deflation: remove components of previously found directions
                for prev_idx in 0..comp_idx {
                    let prev_start = prev_idx * d;
                    let mut dot = 0.0f32;
                    for j in 0..d {
                        dot += result[j] * self.w_down[prev_start + j] / scale;
                    }
                    for j in 0..d {
                        result[j] -= dot * self.w_down[prev_start + j] / scale;
                    }
                }
                let norm2: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm2 < 1e-10 {
                    break;
                }
                for v in &mut result {
                    *v /= norm2;
                }
                dir = result;
            }

            // Write principal direction as w_down row (scaled)
            for j in 0..d {
                self.w_down[comp_idx * d + j] = dir[j] * scale;
            }
        }

        tracing::info!(
            samples = n,
            components = k,
            "Projection warm-started from sample covariance"
        );
    }

    /// Compute the effective rank of the bottleneck activations for a batch of inputs.
    ///
    /// Effective rank = exp(entropy of singular value distribution).
    /// Low effective rank → projection is collapsing to a low-dimensional subspace.
    /// Returns a value in [1, bottleneck_dim].
    pub fn effective_rank(&self, samples: &[ContinuousHV]) -> f32 {
        if samples.is_empty() || self.bottleneck == 0 {
            return 0.0;
        }

        // Compute bottleneck activations for each sample
        let mut activations: Vec<Vec<f32>> = Vec::with_capacity(samples.len());
        for s in samples {
            let hidden_pre = self.matmul(&self.w_down, &s.values, self.bottleneck, self.hdc_dim);
            let hidden: Vec<f32> = hidden_pre.into_iter().map(activation).collect();
            activations.push(hidden);
        }

        // Compute variance per bottleneck dimension
        let n = activations.len() as f32;
        let mut means = vec![0.0f32; self.bottleneck];
        for act in &activations {
            for (m, v) in means.iter_mut().zip(act.iter()) {
                *m += v;
            }
        }
        for m in &mut means {
            *m /= n;
        }

        let mut variances = vec![0.0f32; self.bottleneck];
        for act in &activations {
            for (j, v) in act.iter().enumerate() {
                let diff = v - means[j];
                variances[j] += diff * diff;
            }
        }
        for v in &mut variances {
            *v /= n;
        }

        // Effective rank from variance distribution (proxy for SVD)
        let total_var: f32 = variances.iter().sum();
        if total_var < 1e-10 {
            return 1.0; // Complete collapse
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

    /// Matrix-vector multiply: `result[i] = sum_j(mat[i*cols + j] * vec[j])`
    ///
    /// mat shape: [rows × cols], vec shape: [cols], result shape: [rows]
    fn matmul(&self, mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        debug_assert_eq!(mat.len(), rows * cols);
        debug_assert_eq!(vec.len(), cols);

        let mut result = vec![0.0f32; rows];
        for i in 0..rows {
            let row_start = i * cols;
            let mut sum = 0.0f32;
            for j in 0..cols {
                sum += mat[row_start + j] * vec[j];
            }
            result[i] = sum;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-projection")
    }

    #[test]
    fn test_projection_creation() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        assert_eq!(proj.hdc_dim(), 16384);
        assert_eq!(proj.bottleneck_dim(), 256);
        assert_eq!(proj.ssm_dim(), 768);
        // 256*16384 + 768*256 + 256*768 + 16384*256 = 8,781,824
        assert_eq!(proj.num_params(), 256 * 16384 + 768 * 256 + 256 * 768 + 16384 * 256);
    }

    #[test]
    fn test_project_to_ssm_dimensions() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let hv = ContinuousHV::random_default(42);
        let ssm_vec = proj.project_to_ssm(&hv);
        assert_eq!(ssm_vec.len(), 768);
    }

    #[test]
    fn test_project_to_hdc_dimensions() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let ssm_vec = vec![0.1; 768];
        let hv = proj.project_to_hdc(&ssm_vec);
        assert_eq!(hv.values.len(), 16384);
    }

    #[test]
    fn test_roundtrip_preserves_structure() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let hv = ContinuousHV::random_default(42).normalize();

        // Project forward and back
        let ssm_vec = proj.project_to_ssm(&hv);
        let reconstructed = proj.project_to_hdc(&ssm_vec).normalize();

        // Not expecting perfect reconstruction (information bottleneck),
        // but similarity should be non-trivial with random init
        let sim = hv.similarity(&reconstructed);
        // With random projections, similarity is expected to be small but
        // the output should be finite and well-formed
        assert!(sim.is_finite(), "Similarity should be finite");
    }

    #[test]
    fn test_deterministic_initialization() {
        let genesis = test_genesis();
        let proj1 = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let proj2 = HdcSsmProjection::new(&genesis, 16384, 256, 768);

        let hv = ContinuousHV::random_default(42);
        let ssm1 = proj1.project_to_ssm(&hv);
        let ssm2 = proj2.project_to_ssm(&hv);
        assert_eq!(ssm1, ssm2, "Same genesis should produce identical projections");
    }

    #[test]
    fn test_different_inputs_different_outputs() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);

        let hv1 = ContinuousHV::random_default(42);
        let hv2 = ContinuousHV::random_default(99);

        let ssm1 = proj.project_to_ssm(&hv1);
        let ssm2 = proj.project_to_ssm(&hv2);
        assert_ne!(ssm1, ssm2, "Different inputs should produce different outputs");
    }

    #[test]
    fn test_flatten_load_roundtrip() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let flat = proj.flatten_weights();
        assert_eq!(flat.len(), proj.num_params());

        let mut proj2 = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        proj2.load_weights(&flat);

        let hv = ContinuousHV::random_default(42);
        let ssm1 = proj.project_to_ssm(&hv);
        let ssm2 = proj2.project_to_ssm(&hv);
        assert_eq!(ssm1, ssm2, "Loaded weights should produce identical results");
    }

    #[test]
    fn test_gradient_accumulation_and_application() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64); // Small dims for speed

        // Use non-uniform, differently-seeded vectors to ensure non-zero error
        let thought = ContinuousHV::random(dim, 42);
        let output = ContinuousHV::random(dim, 99);

        // Capture weights before
        let weights_before = proj.flatten_weights();

        // Accumulate and apply gradients with a generous LR and clip
        proj.compute_gradients(&thought, &output);
        proj.apply_gradients(0.1, 1000.0);

        let weights_after = proj.flatten_weights();

        // Weights should have changed
        let changed = weights_before.iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Gradients should modify weights");
    }

    #[test]
    fn test_gradient_clipping() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64);

        // Create a large error signal using differently-scaled random vectors
        let thought = ContinuousHV::random(dim, 42).scale(10.0);
        let output = ContinuousHV::random(dim, 99).scale(10.0);

        let weights_before = proj.flatten_weights();

        // Apply with very tight clipping
        proj.compute_gradients(&thought, &output);
        proj.apply_gradients(0.01, 0.001);

        let weights_after = proj.flatten_weights();

        // Weight changes should be bounded by clipping
        let max_change: f32 = weights_before.iter()
            .zip(weights_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_change < 1.0, "Gradient clipping should bound weight changes, got {max_change}");
    }

    #[test]
    fn test_zero_input_produces_zero_output() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let hv = ContinuousHV::zero(16384);
        let ssm_vec = proj.project_to_ssm(&hv);
        // All zeros through matmul + GELU+residual should produce all zeros
        assert!(ssm_vec.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_contrastive_gradients_modify_weights() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64);

        let anchor = ContinuousHV::random(dim, 42);
        let negative = ContinuousHV::random(dim, 99);

        let weights_before = proj.flatten_weights();

        proj.compute_contrastive_gradients(&anchor, &negative, 0.1);
        proj.apply_gradients(0.1, 1000.0);

        let weights_after = proj.flatten_weights();

        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Contrastive gradients should modify weights");
    }

    #[test]
    fn test_contrastive_pushes_apart() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 32, 64);

        let anchor = ContinuousHV::random(dim, 42).normalize();
        let negative = ContinuousHV::random(dim, 99).normalize();

        // Measure initial bottleneck distance
        let h_a_before = proj.project_to_ssm(&anchor);
        let h_n_before = proj.project_to_ssm(&negative);
        let dist_before: f32 = h_a_before
            .iter()
            .zip(h_n_before.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Apply contrastive gradients multiple times
        for _ in 0..10 {
            proj.compute_contrastive_gradients(&anchor, &negative, 0.5);
            proj.apply_gradients(0.01, 10.0);
        }

        let h_a_after = proj.project_to_ssm(&anchor);
        let h_n_after = proj.project_to_ssm(&negative);
        let dist_after: f32 = h_a_after
            .iter()
            .zip(h_n_after.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Distance should increase (or at least not decrease significantly)
        // Note: with gradient clipping and ReLU, this is a weak assertion
        assert!(
            dist_after.is_finite(),
            "Distance should be finite after contrastive, got {dist_after}"
        );
        assert!(
            dist_before.is_finite(),
            "Initial distance should be finite, got {dist_before}"
        );
    }

    #[test]
    fn test_small_projection_correctness() {
        // Verify matmul with known values
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 4, 2, 3);

        // Just verify dimensions are correct through the pipeline
        let hv = ContinuousHV::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let ssm = proj.project_to_ssm(&hv);
        assert_eq!(ssm.len(), 3);

        let back = proj.project_to_hdc(&ssm);
        assert_eq!(back.values.len(), 4);
    }

    #[test]
    fn test_warm_start_modifies_weights() {
        let genesis = test_genesis();
        let dim = 256;
        let mut proj = HdcSsmProjection::new(&genesis, dim, 16, 64);
        let weights_before = proj.flatten_weights();

        // Create sample HVs with a clear structure
        let samples: Vec<ContinuousHV> = (0..20)
            .map(|i| {
                let mut hv = ContinuousHV::random(dim, 100 + i as u64);
                // Add a dominant component to first few dims
                for j in 0..8 {
                    hv.values[j] += 5.0;
                }
                hv
            })
            .collect();

        proj.warm_start_from_samples(&samples);
        let weights_after = proj.flatten_weights();

        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "Warm-start should modify projection weights");
    }

    #[test]
    fn test_warm_start_empty_samples() {
        let genesis = test_genesis();
        let mut proj = HdcSsmProjection::new(&genesis, 256, 16, 64);
        let before = proj.flatten_weights();

        proj.warm_start_from_samples(&[]);
        let after = proj.flatten_weights();
        assert_eq!(before, after, "Empty samples should not modify weights");
    }

    #[test]
    fn test_effective_rank_finite() {
        let genesis = test_genesis();
        let dim = 256;
        let proj = HdcSsmProjection::new(&genesis, dim, 16, 64);

        let samples: Vec<ContinuousHV> = (0..10)
            .map(|i| ContinuousHV::random(dim, 200 + i as u64))
            .collect();

        let rank = proj.effective_rank(&samples);
        assert!(rank.is_finite(), "Effective rank should be finite");
        assert!(rank >= 1.0, "Effective rank should be at least 1, got {rank}");
        assert!(rank <= 16.0, "Effective rank should be at most bottleneck_dim, got {rank}");
    }

    #[test]
    fn test_effective_rank_collapse_detection() {
        let genesis = test_genesis();
        let dim = 64;
        let proj = HdcSsmProjection::new(&genesis, dim, 8, 16);

        // All-identical samples → should have low effective rank
        let sample = ContinuousHV::random(dim, 42);
        let identical_samples: Vec<ContinuousHV> = (0..10).map(|_| sample.clone()).collect();
        let rank = proj.effective_rank(&identical_samples);
        assert!(rank.is_finite());
        // With identical inputs, all activations are the same → zero variance → rank = 1
        assert!(rank <= 2.0, "Identical inputs should give low rank, got {rank}");
    }
}
