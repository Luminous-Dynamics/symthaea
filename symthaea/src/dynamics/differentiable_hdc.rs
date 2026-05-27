// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Differentiable HDC Encoder
==========================

**The Bridge**: Enables end-to-end gradient flow from CfC brain back into
the HDC encoder, allowing Symthaea to "learn to perceive."

# The Problem with Static HDC

Standard HDC uses fixed hash functions (Blake3) to create hypervectors:
```text
Input -> Blake3 -> Fixed HV (no learning!)
```

The brain (CfC/LTC) learns, but the encoder is frozen. Symthaea can't
learn that "dog" and "cat" should be similar because they function
similarly in thought - it's stuck with arbitrary hash similarities.

# The Solution: Straight-Through Estimator (STE)

We use a learnable dense layer followed by binarization, with STE
for gradient flow:

```text
Forward Pass:  x -> Dense -> tanh -> sign -> binary HV
Backward Pass: gradient flows through sign() as if it were identity
```

**Key Insight**: During forward pass, we binarize for HDC properties.
During backward pass, we pretend binarization didn't happen, allowing
gradients to flow through.

# Mathematical Foundation

For neuron output z = tanh(Wx + b):
- Forward: hv = sign(z) ∈ {-1, +1}
- Backward: ∂L/∂W = ∂L/∂hv · ∂z/∂W  (STE: treat sign as identity)

Reference: Bengio et al., "Estimating or Propagating Gradients Through
Stochastic Neurons for Conditional Computation" (2013)
*/

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for differentiable HDC encoder
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DifferentiableHDCConfig {
    /// Input dimension (e.g., embedding size)
    pub input_dim: usize,
    /// HDC output dimension (e.g., 16,384)
    pub hdc_dim: usize,
    /// Hidden layer size (optional bottleneck)
    pub hidden_dim: Option<usize>,
    /// Learning rate for encoder updates
    pub learning_rate: f32,
    /// Temperature for soft binarization (lower = sharper)
    pub temperature: f32,
    /// Use bipolar {-1, +1} or binary {0, 1}
    pub bipolar: bool,
}

impl Default for DifferentiableHDCConfig {
    fn default() -> Self {
        Self {
            input_dim: 768,        // Common embedding size (BERT, etc.)
            hdc_dim: 16_384,       // Standard HDC dimension
            hidden_dim: Some(512), // Bottleneck for efficiency
            learning_rate: 0.0001, // Conservative for encoder
            temperature: 1.0,      // Start soft, anneal to hard
            bipolar: true,         // {-1, +1} for standard HDC ops
        }
    }
}

/// Differentiable HDC Encoder with Straight-Through Estimator
///
/// Learns to project inputs into hypervector space while maintaining
/// the ability to backpropagate gradients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DifferentiableHDCEncoder {
    pub config: DifferentiableHDCConfig,

    // === Encoder Weights ===
    /// First layer: input -> hidden (if hidden_dim is Some)
    pub w1: Option<Array2<f32>>,
    pub b1: Option<Array1<f32>>,

    /// Final layer: hidden (or input) -> hdc_dim
    pub w_out: Array2<f32>,
    pub b_out: Array1<f32>,

    // === Training State ===
    /// Last pre-activation values (for gradient computation)
    #[serde(skip)]
    last_pre_activation: Option<Array1<f32>>,
    /// Last input (for gradient computation)
    #[serde(skip)]
    last_input: Option<Array1<f32>>,
    /// Total training steps
    pub training_steps: u64,
}

impl DifferentiableHDCEncoder {
    /// Create a new differentiable HDC encoder
    pub fn new(config: DifferentiableHDCConfig) -> Result<Self> {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // Xavier initialization
        let xavier = |rows: usize, cols: usize, rng: &mut rand_chacha::ChaCha8Rng| -> Array2<f32> {
            let scale = (6.0 / (rows + cols) as f32).sqrt();
            Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
        };

        let (w1, b1) = if let Some(hidden) = config.hidden_dim {
            (
                Some(xavier(hidden, config.input_dim, &mut rng)),
                Some(Array1::zeros(hidden)),
            )
        } else {
            (None, None)
        };

        let out_input_dim = config.hidden_dim.unwrap_or(config.input_dim);
        let w_out = xavier(config.hdc_dim, out_input_dim, &mut rng);
        let b_out = Array1::zeros(config.hdc_dim);

        Ok(Self {
            config,
            w1,
            b1,
            w_out,
            b_out,
            last_pre_activation: None,
            last_input: None,
            training_steps: 0,
        })
    }

    /// Create with default configuration
    pub fn default_encoder() -> Result<Self> {
        Self::new(DifferentiableHDCConfig::default())
    }

    /// Encode input to hypervector (forward pass)
    ///
    /// Returns both the binary HV and the continuous pre-activation
    /// (needed for gradient computation).
    pub fn encode(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let input_arr = Array1::from_vec(input.to_vec());
        self.last_input = Some(input_arr.clone());

        // Optional hidden layer
        let hidden = if let (Some(w1), Some(b1)) = (&self.w1, &self.b1) {
            let z1 = w1.dot(&input_arr) + b1;
            z1.mapv(|x| x.tanh()) // tanh activation
        } else {
            input_arr
        };

        // Output projection
        let pre_activation = self.w_out.dot(&hidden) + &self.b_out;
        self.last_pre_activation = Some(pre_activation.clone());

        // Soft-to-hard binarization with temperature
        let soft = pre_activation.mapv(|x| (x / self.config.temperature).tanh());

        // Hard binarization (STE: gradient passes through)
        let hard = if self.config.bipolar {
            soft.mapv(|x| if x >= 0.0 { 1.0 } else { -1.0 })
        } else {
            soft.mapv(|x| if x >= 0.0 { 1.0 } else { 0.0 })
        };

        Ok((hard.to_vec(), soft.to_vec()))
    }

    /// Encode input (simplified interface returning only binary HV)
    pub fn encode_binary(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let (binary, _) = self.encode(input)?;
        Ok(binary)
    }

    /// Backward pass: compute gradients and update weights
    ///
    /// Uses Straight-Through Estimator: gradient flows through sign()
    /// as if it were the identity function.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient of loss w.r.t. the output HV
    ///
    /// # Returns
    /// Gradient w.r.t. input (for further backprop if needed)
    pub fn backward(&mut self, grad_output: &[f32]) -> Result<Vec<f32>> {
        let grad_out = Array1::from_vec(grad_output.to_vec());

        // Get cached values from forward pass
        let pre_act = self
            .last_pre_activation
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Must call forward before backward"))?;
        let input = self
            .last_input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Must call forward before backward"))?;

        // STE: Gradient through sign() is passed through as identity
        // But we do apply the tanh derivative for the soft version
        let tanh_deriv = pre_act.mapv(|x| {
            let t = (x / self.config.temperature).tanh();
            (1.0 - t * t) / self.config.temperature
        });

        // Gradient at pre-activation: dL/dz = dL/dhv * d(tanh)/dz
        let grad_pre_act = &grad_out * &tanh_deriv;

        // Compute hidden layer output (for weight gradient)
        let hidden = if let (Some(w1), Some(b1)) = (&self.w1, &self.b1) {
            let z1 = w1.dot(input) + b1;
            z1.mapv(|x| x.tanh())
        } else {
            input.clone()
        };

        // Gradient for output weights: dL/dW_out = outer(dL/dz, hidden)
        let lr = self.config.learning_rate;
        for i in 0..self.w_out.nrows() {
            for j in 0..self.w_out.ncols() {
                self.w_out[[i, j]] -= lr * grad_pre_act[i] * hidden[j];
            }
            self.b_out[i] -= lr * grad_pre_act[i];
        }

        // Gradient through hidden layer (if exists)
        let grad_input = if let (Some(w1), Some(b1)) = (&mut self.w1, &mut self.b1) {
            // dL/dhidden = W_out^T * dL/dz
            let grad_hidden = self.w_out.t().dot(&grad_pre_act);

            // tanh derivative for hidden layer
            let z1 = w1.dot(input) + &*b1;
            let tanh_deriv_h = z1.mapv(|x| {
                let t = x.tanh();
                1.0 - t * t
            });
            let grad_pre_h = &grad_hidden * &tanh_deriv_h;

            // Update hidden weights
            for i in 0..w1.nrows() {
                for j in 0..w1.ncols() {
                    w1[[i, j]] -= lr * grad_pre_h[i] * input[j];
                }
                b1[i] -= lr * grad_pre_h[i];
            }

            // Gradient w.r.t. input
            w1.t().dot(&grad_pre_h).to_vec()
        } else {
            self.w_out.t().dot(&grad_pre_act).to_vec()
        };

        self.training_steps += 1;

        Ok(grad_input)
    }

    /// Anneal temperature for harder binarization
    ///
    /// Start with high temperature (soft), gradually decrease (hard).
    /// This helps training converge before enforcing strict binary.
    pub fn anneal_temperature(&mut self, factor: f32) {
        self.config.temperature *= factor;
        self.config.temperature = self.config.temperature.max(0.1); // Don't go too hard
    }

    /// Get current temperature
    pub fn temperature(&self) -> f32 {
        self.config.temperature
    }

    /// Compute similarity between two encoded vectors
    ///
    /// For bipolar {-1, +1}: cosine similarity
    /// For binary {0, 1}: Hamming similarity
    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(
            a.len(),
            b.len(),
            "DifferentiableHDC::similarity length mismatch"
        );
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Bind two hypervectors (element-wise multiplication)
    pub fn bind(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    /// Bundle hypervectors (element-wise sum + threshold)
    pub fn bundle(vectors: &[Vec<f32>], bipolar: bool) -> Vec<f32> {
        if vectors.is_empty() {
            return vec![];
        }

        let dim = vectors[0].len();
        let mut sum = vec![0.0; dim];

        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                sum[i] += val;
            }
        }

        // Threshold back to binary
        if bipolar {
            sum.iter()
                .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
                .collect()
        } else {
            sum.iter()
                .map(|&x| if x >= 0.0 { 1.0 } else { 0.0 })
                .collect()
        }
    }

    /// Serialize for persistence
    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(data)?)
    }
}

/// Combined encoder that can switch between fixed (Blake3) and learnable
#[derive(Clone, Debug)]
pub enum HDCEncoder {
    /// Fixed hash-based encoder (no learning)
    Fixed { seed: u64 },
    /// Learnable encoder with STE
    Learnable(Box<DifferentiableHDCEncoder>),
}

impl HDCEncoder {
    /// Encode input to hypervector
    pub fn encode(&mut self, input: &[f32], dim: usize) -> Result<Vec<f32>> {
        match self {
            HDCEncoder::Fixed { seed } => {
                // Use Blake3 for deterministic projection
                use blake3::Hasher;
                let mut hasher = Hasher::new();
                hasher.update(&seed.to_le_bytes());
                for &x in input {
                    hasher.update(&x.to_le_bytes());
                }

                let mut bytes = vec![0u8; dim * 4];
                let mut xof = hasher.finalize_xof();
                xof.fill(&mut bytes);

                let hv: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        if bits & 1 == 0 { 1.0 } else { -1.0 }
                    })
                    .collect();

                Ok(hv)
            }
            HDCEncoder::Learnable(encoder) => encoder.encode_binary(input),
        }
    }

    /// Check if encoder is learnable
    pub fn is_learnable(&self) -> bool {
        matches!(self, HDCEncoder::Learnable(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_produces_binary() {
        let config = DifferentiableHDCConfig {
            input_dim: 64,
            hdc_dim: 256,
            hidden_dim: None,
            ..Default::default()
        };
        let mut encoder = DifferentiableHDCEncoder::new(config).unwrap();

        let input = vec![0.5; 64];
        let (binary, _) = encoder.encode(&input).unwrap();

        assert_eq!(binary.len(), 256);
        // All values should be -1 or +1
        for &v in &binary {
            assert!(v == -1.0 || v == 1.0, "Value {} is not binary", v);
        }
    }

    #[test]
    fn test_backward_updates_weights() {
        let config = DifferentiableHDCConfig {
            input_dim: 32,
            hdc_dim: 64,
            hidden_dim: None,
            learning_rate: 0.1, // High LR for visible change
            ..Default::default()
        };
        let mut encoder = DifferentiableHDCEncoder::new(config).unwrap();

        let input = vec![0.5; 32];
        let w_out_before = encoder.w_out.clone();

        // Forward
        encoder.encode(&input).unwrap();

        // Backward with non-zero gradient
        let grad = vec![1.0; 64];
        encoder.backward(&grad).unwrap();

        // Weights should have changed
        let changed = encoder
            .w_out
            .iter()
            .zip(w_out_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(changed, "Weights should update after backward pass");
    }

    #[test]
    fn test_similar_inputs_similar_outputs() {
        let config = DifferentiableHDCConfig {
            input_dim: 64,
            hdc_dim: 256,
            hidden_dim: Some(128),
            temperature: 0.5, // Soft for similarity
            ..Default::default()
        };
        let mut encoder = DifferentiableHDCEncoder::new(config).unwrap();

        let input1 = vec![0.5; 64];
        let mut input2 = vec![0.5; 64];
        input2[0] = 0.51; // Tiny change

        let (_, soft1) = encoder.encode(&input1).unwrap();
        let (_, soft2) = encoder.encode(&input2).unwrap();

        let sim = DifferentiableHDCEncoder::similarity(&soft1, &soft2);
        assert!(sim > 0.95, "Similar inputs should produce similar outputs");
    }

    #[test]
    fn test_temperature_annealing() {
        let config = DifferentiableHDCConfig {
            temperature: 1.0,
            ..Default::default()
        };
        let mut encoder = DifferentiableHDCEncoder::new(config).unwrap();

        assert_eq!(encoder.temperature(), 1.0);

        encoder.anneal_temperature(0.5);
        assert_eq!(encoder.temperature(), 0.5);

        // Should not go below minimum
        for _ in 0..100 {
            encoder.anneal_temperature(0.5);
        }
        assert!(encoder.temperature() >= 0.1);
    }

    #[test]
    fn test_bind_and_bundle() {
        let a = vec![1.0, -1.0, 1.0, -1.0];
        let b = vec![1.0, 1.0, -1.0, -1.0];

        let bound = DifferentiableHDCEncoder::bind(&a, &b);
        assert_eq!(bound, vec![1.0, -1.0, -1.0, 1.0]); // XOR for bipolar

        let bundled = DifferentiableHDCEncoder::bundle(&[a, b], true);
        assert_eq!(bundled, vec![1.0, 1.0, 1.0, -1.0]); // Majority vote
    }
}
