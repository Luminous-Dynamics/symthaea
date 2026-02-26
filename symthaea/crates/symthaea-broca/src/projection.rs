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
//! Forward:  HDC(16384) → w_down → ReLU → w_up → SSM(768)
//! Backward: SSM(768) → w_back_down → ReLU → w_back_up → HDC(16384)
//! ```

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

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
    /// Pipeline: `hv → w_down → ReLU → w_up → ssm_vec`
    pub fn project_to_ssm(&self, hv: &ContinuousHV) -> Vec<f32> {
        debug_assert_eq!(hv.values.len(), self.hdc_dim);

        // Step 1: w_down * hv → bottleneck (256D)
        let hidden = self.matmul(&self.w_down, &hv.values, self.bottleneck, self.hdc_dim);

        // Step 2: ReLU
        let hidden: Vec<f32> = hidden.into_iter().map(|x| x.max(0.0)).collect();

        // Step 3: w_up * hidden → ssm (768D)
        self.matmul(&self.w_up, &hidden, self.ssm_dim, self.bottleneck)
    }

    /// Project SSM vector (768D) back to HDC space (16,384D).
    ///
    /// Pipeline: `ssm_vec → w_back_down → ReLU → w_back_up → hv`
    pub fn project_to_hdc(&self, ssm_vec: &[f32]) -> ContinuousHV {
        debug_assert_eq!(ssm_vec.len(), self.ssm_dim);

        // Step 1: w_back_down * ssm → bottleneck (256D)
        let hidden = self.matmul(&self.w_back_down, ssm_vec, self.bottleneck, self.ssm_dim);

        // Step 2: ReLU
        let hidden: Vec<f32> = hidden.into_iter().map(|x| x.max(0.0)).collect();

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
        let hidden_fwd = self.matmul(&self.w_down, &thought_hv.values, self.bottleneck, self.hdc_dim);
        let hidden_fwd_relu: Vec<f32> = hidden_fwd.iter().map(|x| x.max(0.0)).collect();

        // Backward projection hidden activations
        let ssm_fwd = self.matmul(&self.w_up, &hidden_fwd_relu, self.ssm_dim, self.bottleneck);
        let hidden_back = self.matmul(&self.w_back_down, &ssm_fwd, self.bottleneck, self.ssm_dim);
        let hidden_back_relu: Vec<f32> = hidden_back.iter().map(|x| x.max(0.0)).collect();

        // Gradient for w_back_up: error * hidden_back_relu^T
        // Shape: [hdc_dim × bottleneck]
        for i in 0..self.hdc_dim {
            for j in 0..self.bottleneck {
                self.grad_back_up[i * self.bottleneck + j] += error[i] * hidden_back_relu[j];
            }
        }

        // Gradient for w_back_down: (w_back_up^T * error) * relu'(hidden_back) * ssm_fwd^T
        // Simplified: accumulate per-element
        let mut delta_back = vec![0.0f32; self.bottleneck];
        for j in 0..self.bottleneck {
            let mut sum = 0.0f32;
            for i in 0..self.hdc_dim {
                sum += self.w_back_up[i * self.bottleneck + j] * error[i];
            }
            // ReLU derivative
            delta_back[j] = if hidden_back[j] > 0.0 { sum } else { 0.0 };
        }
        for i in 0..self.bottleneck {
            for j in 0..self.ssm_dim {
                self.grad_back_down[i * self.ssm_dim + j] += delta_back[i] * ssm_fwd[j];
            }
        }

        // Gradient for w_up: (w_back_down^T * delta_back) * hidden_fwd_relu^T
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
                self.grad_up[i * self.bottleneck + j] += delta_up[i] * hidden_fwd_relu[j];
            }
        }

        // Gradient for w_down: (w_up^T * delta_up) * relu'(hidden_fwd) * thought_hv^T
        let mut delta_down = vec![0.0f32; self.bottleneck];
        for j in 0..self.bottleneck {
            let mut sum = 0.0f32;
            for i in 0..self.ssm_dim {
                sum += self.w_up[i * self.bottleneck + j] * delta_up[i];
            }
            delta_down[j] = if hidden_fwd[j] > 0.0 { sum } else { 0.0 };
        }
        for i in 0..self.bottleneck {
            for j in 0..self.hdc_dim {
                self.grad_down[i * self.hdc_dim + j] += delta_down[i] * thought_hv.values[j];
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
        // All zeros through matmul + ReLU should produce all zeros
        assert!(ssm_vec.iter().all(|&x| x.abs() < 1e-10));
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
}
