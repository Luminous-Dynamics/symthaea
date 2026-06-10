// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Liquid Projection: Reservoir Computing + HDC
//!
//! Architecture: Mel -> Liquid Reservoir (LTC) -> Learned Projection -> HDC
//!
//! The reservoir provides non-linear temporal transformation.
//! The readout is a simple linear matrix trained via Ridge Regression.
//! Target phoneme hypervectors are fixed and orthogonal.

use crate::hdc::HV16;
use crate::ltc::{LtcCell, LtcConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the Liquid Projection encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidProjectionConfig {
    /// Reservoir hidden size (larger = more expressive)
    pub reservoir_size: usize,
    /// Ridge regression regularization parameter
    pub ridge_lambda: f32,
    /// Frame duration for LTC updates (seconds)
    pub frame_duration: f32,
}

impl Default for LiquidProjectionConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 256,
            ridge_lambda: 0.01,
            frame_duration: 0.010,
        }
    }
}

/// Fixed orthogonal target hypervectors for phonemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeTargets {
    /// Phoneme label -> target hypervector
    targets: HashMap<String, HV16>,
}

impl PhonemeTargets {
    /// Create random orthogonal targets for a set of phonemes
    pub fn new(phonemes: &[String]) -> Self {
        let mut targets = HashMap::new();

        for (i, phoneme) in phonemes.iter().enumerate() {
            // Generate deterministic random HV using phoneme index as seed
            let hv = Self::generate_target(i);
            targets.insert(phoneme.clone(), hv);
        }

        Self { targets }
    }

    /// Generate a deterministic random HV for a phoneme index
    fn generate_target(index: usize) -> HV16 {
        // Use blake3 hash for deterministic randomness
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"phoneme_target_");
        hasher.update(&index.to_le_bytes());
        let mut output = [0u8; 256];
        hasher.finalize_xof().fill(&mut output);

        HV16::from_bytes(&output)
    }

    /// Get target HV for a phoneme
    pub fn get(&self, phoneme: &str) -> Option<&HV16> {
        self.targets.get(phoneme)
    }

    /// Get all phoneme labels
    pub fn phonemes(&self) -> Vec<String> {
        self.targets.keys().cloned().collect()
    }

    /// Number of phonemes
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

/// Readout matrix that maps reservoir state to HV space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutMatrix {
    /// Weight matrix: `[hv_dim x reservoir_size]`
    /// Each row maps reservoir state to one dimension of the output
    weights: Vec<Vec<f32>>,
    /// Output dimension (HDC_DIM = 2048)
    output_dim: usize,
    /// Input dimension (reservoir_size)
    input_dim: usize,
}

impl ReadoutMatrix {
    /// Create a new readout matrix (initially zeros)
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: vec![vec![0.0; input_dim]; output_dim],
            output_dim,
            input_dim,
        }
    }

    /// Apply the readout to a reservoir state
    pub fn project(&self, state: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_dim];

        for (i, row) in self.weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                if j < state.len() {
                    output[i] += w * state[j];
                }
            }
        }

        output
    }

    /// Convert projected output to binary HV
    pub fn to_hv(&self, state: &[f32]) -> HV16 {
        let projected = self.project(state);

        // Convert to binary: positive -> 1, negative -> 0
        let mut hv = HV16::zero();
        for (i, &val) in projected.iter().enumerate() {
            if val > 0.0 && i < crate::hdc::HDC_DIM {
                let word_idx = i / 128;
                let bit_idx = i % 128;
                hv.words[word_idx] |= 1u128 << bit_idx;
            }
        }

        hv
    }

    /// Set weights from solved matrix
    pub fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.weights = weights;
    }
}

/// Ridge Regression accumulator for training the readout
pub struct RidgeAccumulator {
    /// X^T X matrix: `[input_dim x input_dim]`
    xtx: Vec<Vec<f64>>,
    /// X^T Y matrix: `[input_dim x output_dim]`
    xty: Vec<Vec<f64>>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Sample count
    n_samples: usize,
}

impl RidgeAccumulator {
    /// Create a new accumulator
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            xtx: vec![vec![0.0; input_dim]; input_dim],
            xty: vec![vec![0.0; output_dim]; input_dim],
            input_dim,
            output_dim,
            n_samples: 0,
        }
    }

    /// Add a sample (reservoir state, target HV as floats)
    pub fn add(&mut self, x: &[f32], y: &[f32]) {
        // Accumulate X^T X
        for i in 0..self.input_dim.min(x.len()) {
            for j in 0..self.input_dim.min(x.len()) {
                self.xtx[i][j] += (x[i] as f64) * (x[j] as f64);
            }
        }

        // Accumulate X^T Y
        for i in 0..self.input_dim.min(x.len()) {
            for j in 0..self.output_dim.min(y.len()) {
                self.xty[i][j] += (x[i] as f64) * (y[j] as f64);
            }
        }

        self.n_samples += 1;
    }

    /// Add a sample with HV target (converts to +1/-1 floats)
    pub fn add_with_hv(&mut self, x: &[f32], target_hv: &HV16) {
        // Convert HV to +1/-1 floats
        let mut y = vec![0.0f32; crate::hdc::HDC_DIM];
        for i in 0..crate::hdc::HDC_DIM {
            let word_idx = i / 128;
            let bit_idx = i % 128;
            let bit = ((target_hv.words[word_idx] >> bit_idx) & 1) as f32;
            y[i] = bit * 2.0 - 1.0; // Map 0/1 to -1/+1
        }

        self.add(x, &y);
    }

    /// Solve for W = (X^T X + λI)^{-1} X^T Y
    pub fn solve(&self, lambda: f32) -> Vec<Vec<f32>> {
        // Add regularization: X^T X + λI
        let mut a = self.xtx.clone();
        for i in 0..self.input_dim {
            a[i][i] += lambda as f64;
        }

        // Solve A * W = X^T Y using Cholesky decomposition
        // Since A is symmetric positive definite, we can use Cholesky

        // First, compute Cholesky decomposition: A = L L^T
        let l = Self::cholesky(&a);

        // Solve L * Z = X^T Y (forward substitution)
        // Then L^T * W = Z (backward substitution)

        let mut result = vec![vec![0.0f32; self.input_dim]; self.output_dim];

        for col in 0..self.output_dim {
            // Get column of X^T Y
            let b: Vec<f64> = (0..self.input_dim).map(|i| self.xty[i][col]).collect();

            // Forward substitution: L * z = b
            let z = Self::forward_substitute(&l, &b);

            // Backward substitution: L^T * w = z
            let w = Self::backward_substitute(&l, &z);

            // Store result
            for i in 0..self.input_dim {
                result[col][i] = w[i] as f32;
            }
        }

        // Transpose to get [output_dim x input_dim]
        result
    }

    /// Cholesky decomposition: A = L L^T
    fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                if j == i {
                    for k in 0..j {
                        sum += l[j][k] * l[j][k];
                    }
                    l[j][j] = (a[j][j] - sum).sqrt().max(1e-10);
                } else {
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    l[i][j] = (a[i][j] - sum) / l[j][j].max(1e-10);
                }
            }
        }

        l
    }

    /// Forward substitution: L * x = b
    fn forward_substitute(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut x = vec![0.0; n];

        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i][j] * x[j];
            }
            x[i] = sum / l[i][i].max(1e-10);
        }

        x
    }

    /// Backward substitution: L^T * x = b
    fn backward_substitute(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut x = vec![0.0; n];

        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= l[j][i] * x[j]; // L^T[i][j] = L[j][i]
            }
            x[i] = sum / l[i][i].max(1e-10);
        }

        x
    }

    /// Number of samples accumulated
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
}

/// The complete Liquid Projection encoder
#[derive(Clone, Serialize, Deserialize)]
pub struct LiquidProjection {
    /// Configuration
    pub config: LiquidProjectionConfig,
    /// Target phoneme hypervectors
    pub targets: PhonemeTargets,
    /// Trained readout matrix
    pub readout: ReadoutMatrix,
    /// Reservoir configuration (for reconstruction)
    ltc_config: LtcConfig,
}

impl LiquidProjection {
    /// Create a new untrained encoder
    pub fn new(phonemes: &[String], config: LiquidProjectionConfig) -> Self {
        let ltc_config = LtcConfig {
            hidden_size: config.reservoir_size,
            ..Default::default()
        };

        Self {
            targets: PhonemeTargets::new(phonemes),
            readout: ReadoutMatrix::new(config.reservoir_size, crate::hdc::HDC_DIM),
            config,
            ltc_config,
        }
    }

    /// Create a reservoir for this encoder
    pub fn create_reservoir(&self, input_size: usize) -> LtcCell {
        LtcCell::new(input_size, self.ltc_config.clone())
    }

    /// Encode a reservoir state to HV
    pub fn encode(&self, reservoir_state: &[f32]) -> HV16 {
        self.readout.to_hv(reservoir_state)
    }

    /// Get target HV for a phoneme
    pub fn get_target(&self, phoneme: &str) -> Option<&HV16> {
        self.targets.get(phoneme)
    }

    /// Decode: find the phoneme with highest similarity to encoded state
    pub fn decode(&self, reservoir_state: &[f32]) -> Option<(String, f32)> {
        let encoded = self.encode(reservoir_state);

        let mut best_phoneme = None;
        let mut best_sim = f32::NEG_INFINITY;

        for phoneme in self.targets.phonemes() {
            if let Some(target) = self.targets.get(&phoneme) {
                let sim = encoded.similarity(target);
                if sim > best_sim {
                    best_sim = sim;
                    best_phoneme = Some(phoneme);
                }
            }
        }

        best_phoneme.map(|p| (p, best_sim))
    }

    /// Set the trained readout weights
    pub fn set_readout_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.readout.set_weights(weights);
    }

    /// Save to file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Load from file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let encoder = serde_json::from_reader(reader)?;
        Ok(encoder)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Direct Linear Classifier (no HDC binarization)
// Maps reservoir state → phoneme logits directly via Ridge Regression
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the direct classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectClassifierConfig {
    /// Reservoir hidden size
    pub reservoir_size: usize,
    /// Ridge regression regularization
    pub ridge_lambda: f32,
    /// Frame duration for LTC updates
    pub frame_duration: f32,
    /// Number of context frames for input stacking
    pub context_frames: usize,
}

impl Default for DirectClassifierConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 1024,
            ridge_lambda: 0.01,
            frame_duration: 0.010,
            context_frames: 7,
        }
    }
}

/// Activation function for random features
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum RFActivation {
    #[default]
    Tanh,
    ReLU,
}

/// Random nonlinear feature expansion (Extreme Learning Machine / Random Kitchen Sinks)
/// Maps input x → activation(W*x + b) where W and b are fixed random matrices
#[derive(Clone, Serialize, Deserialize)]
pub struct RandomProjection {
    /// Random weight matrix: `[output_dim × input_dim]`
    pub weights: Vec<Vec<f32>>,
    /// Random bias: `[output_dim]`
    pub bias: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
    /// Activation function
    #[serde(default)]
    pub activation: RFActivation,
}

impl RandomProjection {
    /// Create a new random projection with Gaussian weights (tanh activation)
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        Self::with_activation(input_dim, output_dim, seed, RFActivation::Tanh)
    }

    /// Create a random projection with specified activation
    pub fn with_activation(
        input_dim: usize,
        output_dim: usize,
        seed: u64,
        activation: RFActivation,
    ) -> Self {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Scale initialization based on activation
        let std = match activation {
            RFActivation::Tanh => (2.0 / (input_dim + output_dim) as f64).sqrt() as f32,
            RFActivation::ReLU => (2.0 / input_dim as f64).sqrt() as f32, // He initialization
        };

        let weights: Vec<Vec<f32>> = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| {
                        // Box-Muller transform for Gaussian
                        let u1: f32 = rng.r#gen::<f32>().max(1e-7);
                        let u2: f32 = rng.r#gen();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        z * std
                    })
                    .collect()
            })
            .collect();

        let bias: Vec<f32> = (0..output_dim)
            .map(|_| {
                let u: f32 = rng.r#gen();
                (u - 0.5) * 2.0 * std
            })
            .collect();

        Self {
            weights,
            bias,
            input_dim,
            output_dim,
            activation,
        }
    }

    /// Apply projection: output = activation(W*x + b)
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut output = self.bias.clone();
        for (i, row) in self.weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                if j < x.len() {
                    output[i] += w * x[j];
                }
            }
        }
        // Apply activation
        match self.activation {
            RFActivation::Tanh => {
                for v in &mut output {
                    *v = v.tanh();
                }
            }
            RFActivation::ReLU => {
                for v in &mut output {
                    *v = v.max(0.0);
                }
            }
        }
        output
    }
}

/// Direct linear classifier: reservoir state → phoneme logits
/// No HDC binarization - purely continuous space
#[derive(Clone, Serialize, Deserialize)]
pub struct DirectClassifier {
    /// Configuration
    pub config: DirectClassifierConfig,
    /// Phoneme labels (index → label)
    pub phonemes: Vec<String>,
    /// Weight matrix: `[n_phonemes × reservoir_size]`
    pub weights: Vec<Vec<f32>>,
    /// Bias vector: `[n_phonemes]`
    pub bias: Vec<f32>,
    /// Log prior probabilities for each class (for Bayesian debiasing)
    #[serde(default)]
    pub log_priors: Vec<f32>,
    /// Scale factor for prior correction (0.0 = no correction, 1.0 = full correction)
    #[serde(default)]
    pub prior_scale: f32,
    /// Optional random nonlinear projection applied before classification
    #[serde(default)]
    pub random_projection: Option<RandomProjection>,
}

impl DirectClassifier {
    /// Create a new untrained classifier
    pub fn new(phonemes: &[String], config: DirectClassifierConfig) -> Self {
        let n_phonemes = phonemes.len();
        let reservoir_size = config.reservoir_size;

        Self {
            config,
            phonemes: phonemes.to_vec(),
            weights: vec![vec![0.0; reservoir_size]; n_phonemes],
            bias: vec![0.0; n_phonemes],
            log_priors: vec![0.0; n_phonemes],
            prior_scale: 0.0,
            random_projection: None,
        }
    }

    /// Get phoneme index
    pub fn phoneme_index(&self, phoneme: &str) -> Option<usize> {
        self.phonemes.iter().position(|p| p == phoneme)
    }

    /// Decode: compute logits, apply prior correction, return best phoneme
    pub fn decode(&self, reservoir_state: &[f32]) -> (String, f32) {
        let logits = self.compute_logits(reservoir_state);

        let mut best_idx = 0;
        let mut best_logit = f32::NEG_INFINITY;

        for (i, &logit) in logits.iter().enumerate() {
            // Subtract log prior to debias (Bayesian prior correction)
            let prior_correction = if i < self.log_priors.len() {
                self.prior_scale * self.log_priors[i]
            } else {
                0.0
            };
            let adjusted = logit - prior_correction;
            if adjusted > best_logit {
                best_logit = adjusted;
                best_idx = i;
            }
        }

        // Compute softmax probability for the best class (using adjusted logits)
        let adjusted_logits: Vec<f32> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let prior = if i < self.log_priors.len() {
                    self.prior_scale * self.log_priors[i]
                } else {
                    0.0
                };
                l - prior
            })
            .collect();
        let max_logit = adjusted_logits[best_idx];
        let exp_sum: f32 = adjusted_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let confidence = 1.0 / exp_sum;

        (self.phonemes[best_idx].clone(), confidence)
    }

    /// Set log priors from class counts (for Bayesian debiasing)
    pub fn set_log_priors(&mut self, class_counts: &[usize], prior_scale: f32) {
        let total: usize = class_counts.iter().sum();
        if total == 0 {
            return;
        }
        self.log_priors = class_counts
            .iter()
            .map(|&count| {
                if count > 0 {
                    (count as f32 / total as f32).ln()
                } else {
                    -20.0 // Effectively -inf but JSON-safe
                }
            })
            .collect();
        self.prior_scale = prior_scale;
    }

    /// Normalize weight vectors to unit norm per class
    /// This removes bias from class frequency: each class has equal "voice"
    pub fn normalize_weights(&mut self) {
        for row in &mut self.weights {
            let norm: f32 = row.iter().map(|w| w * w).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for w in row.iter_mut() {
                    *w /= norm;
                }
            }
        }
    }

    /// Compute raw logits for all phonemes
    /// If random_projection is set, applies it first: logits = W * tanh(W_rand * state + b_rand) + b
    pub fn compute_logits(&self, state: &[f32]) -> Vec<f32> {
        // Apply random projection if present
        let projected = if let Some(ref proj) = self.random_projection {
            proj.forward(state)
        } else {
            state.to_vec()
        };

        let mut logits = self.bias.clone();
        for (i, row) in self.weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                if j < projected.len() {
                    logits[i] += w * projected[j];
                }
            }
        }

        logits
    }

    /// Set weights from solved ridge regression
    pub fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.weights = weights;
    }

    /// Save to file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Load from file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let classifier = serde_json::from_reader(reader)?;
        Ok(classifier)
    }
}

/// Accumulator for direct classifier training
/// Uses Ridge Regression with one-hot phoneme targets
/// Supports class-balanced normalization to prevent frequency bias
pub struct DirectAccumulator {
    /// X^T X: flat row-major `[input_dim × input_dim]`
    xtx: Vec<f64>,
    /// X^T Y: flat row-major `[input_dim × n_classes]`
    xty: Vec<f64>,
    /// Per-class sample counts for balancing
    class_counts: Vec<usize>,
    /// Input dimension (reservoir_size)
    input_dim: usize,
    /// Output dimension (n_phonemes)
    n_classes: usize,
    /// Sample count
    pub n_samples: usize,
}

impl DirectAccumulator {
    /// Create a new accumulator
    pub fn new(input_dim: usize, n_classes: usize) -> Self {
        Self {
            xtx: vec![0.0; input_dim * input_dim],
            xty: vec![0.0; input_dim * n_classes],
            class_counts: vec![0; n_classes],
            input_dim,
            n_classes,
            n_samples: 0,
        }
    }

    /// Add a sample with one-hot target (phoneme index)
    pub fn add(&mut self, x: &[f32], target_class: usize) {
        if target_class >= self.n_classes {
            return;
        }

        let n = self.input_dim.min(x.len());

        // Pre-cast x to f64 for efficient inner loop
        let x64: Vec<f64> = x[..n].iter().map(|&v| v as f64).collect();

        // Accumulate X^T X (upper triangle only, symmetric)
        for i in 0..n {
            let xi = x64[i];
            let row_start = i * self.input_dim;
            for j in i..n {
                self.xtx[row_start + j] += xi * x64[j];
            }
        }

        // Accumulate X^T Y (one-hot: only target_class column gets xi)
        for i in 0..n {
            self.xty[i * self.n_classes + target_class] += x64[i];
        }

        self.class_counts[target_class] += 1;
        self.n_samples += 1;
    }

    /// Add a batch of samples using cache-tiled dgemm for X^TX accumulation.
    /// features_f64: row-major `[batch_size × input_dim]` pre-cast to f64
    /// targets: phoneme indices for each sample in the batch
    pub fn add_batch(&mut self, features_f64: &[f64], targets: &[usize], batch_size: usize) {
        let d = self.input_dim;
        // X^TX += B^T * B via cache-tiled dgemm
        // B is [batch_size × d] row-major
        // B^T is [d × batch_size]: same data, row_stride=1, col_stride=d
        // Result: [d × d] row-major
        unsafe {
            matrixmultiply::dgemm(
                d,
                batch_size,
                d,
                1.0,
                features_f64.as_ptr(),
                1,
                d as isize, // B^T view
                features_f64.as_ptr(),
                d as isize,
                1, // B row-major
                1.0,
                self.xtx.as_mut_ptr(),
                d as isize,
                1, // X^TX row-major
            );
        }
        // X^TY per-sample (O(D) each, negligible vs O(D^2) dgemm)
        for s in 0..batch_size {
            let target = targets[s];
            if target < self.n_classes {
                let offset = s * d;
                for i in 0..d {
                    self.xty[i * self.n_classes + target] += features_f64[offset + i];
                }
                self.class_counts[target] += 1;
            }
        }
        self.n_samples += batch_size;
    }

    /// Solve W = (X^T X + λI)^{-1} X^T Y
    /// Standard Ridge Regression (no class balancing - use prior correction at inference)
    /// Returns `[n_classes × input_dim]` weight matrix
    pub fn solve(&self, lambda: f32) -> Vec<Vec<f32>> {
        let n = self.input_dim;

        // Copy X^T X and fill lower triangle from upper (symmetric)
        let mut a = self.xtx.clone();
        for i in 0..n {
            for j in (i + 1)..n {
                a[j * n + i] = a[i * n + j];
            }
            a[i * n + i] += lambda as f64;
        }

        // Cholesky decomposition (flat array)
        let l = Self::cholesky_flat(&a, n);

        // Solve for each class
        let mut result = vec![vec![0.0f32; n]; self.n_classes];

        for cls in 0..self.n_classes {
            if self.class_counts[cls] == 0 {
                continue;
            }

            // Get column of X^T Y for this class
            let b: Vec<f64> = (0..n).map(|i| self.xty[i * self.n_classes + cls]).collect();

            // Forward/backward substitution
            let z = Self::forward_substitute_flat(&l, &b, n);
            let w = Self::backward_substitute_flat(&l, &z, n);

            for i in 0..n {
                result[cls][i] = w[i] as f32;
            }
        }

        result
    }

    /// Get per-class sample counts
    pub fn class_counts(&self) -> &[usize] {
        &self.class_counts
    }

    fn cholesky_flat(a: &[f64], n: usize) -> Vec<f64> {
        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                if j == i {
                    for k in 0..j {
                        let ljk = l[j * n + k];
                        sum += ljk * ljk;
                    }
                    l[j * n + j] = (a[j * n + j] - sum).sqrt().max(1e-10);
                } else {
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j].max(1e-10);
                }
            }
        }

        l
    }

    fn forward_substitute_flat(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut x = vec![0.0; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i * n + j] * x[j];
            }
            x[i] = sum / l[i * n + i].max(1e-10);
        }
        x
    }

    fn backward_substitute_flat(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= l[j * n + i] * x[j]; // L^T: column i of row j
            }
            x[i] = sum / l[i * n + i].max(1e-10);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phoneme_targets() {
        let phonemes = vec!["AA".to_string(), "AE".to_string(), "IY".to_string()];
        let targets = PhonemeTargets::new(&phonemes);

        assert_eq!(targets.len(), 3);

        // Check that targets are orthogonal (low similarity)
        let aa = targets.get("AA").unwrap();
        let ae = targets.get("AE").unwrap();
        let iy = targets.get("IY").unwrap();

        // Different phonemes should have low similarity
        assert!(aa.similarity(ae).abs() < 0.1);
        assert!(aa.similarity(iy).abs() < 0.1);
        assert!(ae.similarity(iy).abs() < 0.1);

        // Same phoneme should have similarity 1.0
        assert!((aa.similarity(aa) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ridge_accumulator() {
        let mut acc = RidgeAccumulator::new(4, 2);

        // Add some samples
        acc.add(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0]);
        acc.add(&[0.0, 1.0, 0.0, 0.0], &[0.0, 1.0]);
        acc.add(&[1.0, 1.0, 0.0, 0.0], &[1.0, 1.0]);

        let weights = acc.solve(0.01);

        // Check that we got a reasonable result
        assert_eq!(weights.len(), 2); // output_dim
        assert_eq!(weights[0].len(), 4); // input_dim
    }

    #[test]
    fn test_liquid_projection_config_default() {
        let config = LiquidProjectionConfig::default();
        assert_eq!(config.reservoir_size, 256);
        assert!((config.ridge_lambda - 0.01).abs() < 1e-6);
        assert!((config.frame_duration - 0.010).abs() < 1e-6);
    }

    #[test]
    fn test_phoneme_targets_deterministic() {
        let phonemes = vec!["AA".to_string(), "IY".to_string()];
        let targets1 = PhonemeTargets::new(&phonemes);
        let targets2 = PhonemeTargets::new(&phonemes);

        // Same index should produce same HV
        let aa1 = targets1.get("AA").unwrap();
        let aa2 = targets2.get("AA").unwrap();
        assert_eq!(
            aa1.words, aa2.words,
            "PhonemeTargets should be deterministic"
        );
    }

    #[test]
    fn test_phoneme_targets_empty() {
        let targets = PhonemeTargets::new(&[]);
        assert!(targets.is_empty());
        assert_eq!(targets.len(), 0);
        assert!(targets.get("AA").is_none());
    }

    #[test]
    fn test_readout_matrix_new_is_zeros() {
        let readout = ReadoutMatrix::new(4, 3);
        assert_eq!(readout.output_dim, 3);
        assert_eq!(readout.input_dim, 4);

        // All weights should be zero
        for row in &readout.weights {
            for &w in row {
                assert_eq!(w, 0.0);
            }
        }
    }

    #[test]
    fn test_readout_project_zero_weights() {
        let readout = ReadoutMatrix::new(4, 3);
        let state = vec![1.0, 2.0, 3.0, 4.0];
        let output = readout.project(&state);

        assert_eq!(output.len(), 3);
        for &val in &output {
            assert_eq!(val, 0.0, "zero weights should produce zero output");
        }
    }

    #[test]
    fn test_readout_project_identity_like() {
        let mut readout = ReadoutMatrix::new(3, 3);
        // Set up identity-like weights
        readout.weights[0][0] = 1.0;
        readout.weights[1][1] = 1.0;
        readout.weights[2][2] = 1.0;

        let state = vec![2.0, 3.0, 5.0];
        let output = readout.project(&state);

        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);
        assert!((output[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_readout_to_hv_positive_values() {
        let mut readout = ReadoutMatrix::new(2, crate::hdc::HDC_DIM);
        // Set all weights to positive so output is positive -> bits set to 1
        for row in &mut readout.weights {
            for w in row.iter_mut() {
                *w = 1.0;
            }
        }

        let state = vec![1.0, 1.0];
        let hv = readout.to_hv(&state);

        // All output values are positive, so all bits should be set
        let popcount: u32 = hv.words.iter().map(|w| w.count_ones()).sum();
        assert_eq!(popcount as usize, crate::hdc::HDC_DIM);
    }

    #[test]
    fn test_ridge_accumulator_sample_count() {
        let mut acc = RidgeAccumulator::new(3, 2);
        assert_eq!(acc.n_samples(), 0);

        acc.add(&[1.0, 0.0, 0.0], &[1.0, 0.0]);
        assert_eq!(acc.n_samples(), 1);

        acc.add(&[0.0, 1.0, 0.0], &[0.0, 1.0]);
        assert_eq!(acc.n_samples(), 2);
    }

    #[test]
    fn test_ridge_solve_identity_mapping() {
        // Train a system where x[0] -> y[0], x[1] -> y[1]
        let mut acc = RidgeAccumulator::new(2, 2);

        for _ in 0..10 {
            acc.add(&[1.0, 0.0], &[1.0, 0.0]);
            acc.add(&[0.0, 1.0], &[0.0, 1.0]);
        }

        let weights = acc.solve(0.001);
        assert_eq!(weights.len(), 2);

        // Weight matrix should approximate identity
        // weights[0] should be close to [1, 0] and weights[1] close to [0, 1]
        assert!(weights[0][0] > 0.9, "w[0][0] = {}", weights[0][0]);
        assert!(weights[0][1].abs() < 0.1, "w[0][1] = {}", weights[0][1]);
        assert!(weights[1][0].abs() < 0.1, "w[1][0] = {}", weights[1][0]);
        assert!(weights[1][1] > 0.9, "w[1][1] = {}", weights[1][1]);
    }

    #[test]
    fn test_direct_classifier_config_default() {
        let config = DirectClassifierConfig::default();
        assert_eq!(config.reservoir_size, 1024);
        assert_eq!(config.context_frames, 7);
        assert!((config.ridge_lambda - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_direct_classifier_new() {
        let phonemes = vec!["AA".to_string(), "IY".to_string(), "SIL".to_string()];
        let config = DirectClassifierConfig {
            reservoir_size: 4,
            ..Default::default()
        };
        let classifier = DirectClassifier::new(&phonemes, config);

        assert_eq!(classifier.phonemes.len(), 3);
        assert_eq!(classifier.weights.len(), 3); // n_phonemes
        assert_eq!(classifier.weights[0].len(), 4); // reservoir_size
        assert_eq!(classifier.bias.len(), 3);
    }

    #[test]
    fn test_direct_classifier_phoneme_index() {
        let phonemes = vec!["AA".to_string(), "IY".to_string(), "SIL".to_string()];
        let config = DirectClassifierConfig {
            reservoir_size: 4,
            ..Default::default()
        };
        let classifier = DirectClassifier::new(&phonemes, config);

        assert_eq!(classifier.phoneme_index("AA"), Some(0));
        assert_eq!(classifier.phoneme_index("IY"), Some(1));
        assert_eq!(classifier.phoneme_index("SIL"), Some(2));
        assert_eq!(classifier.phoneme_index("NOPE"), None);
    }

    #[test]
    fn test_direct_classifier_compute_logits() {
        let phonemes = vec!["AA".to_string(), "IY".to_string()];
        let config = DirectClassifierConfig {
            reservoir_size: 3,
            ..Default::default()
        };
        let mut classifier = DirectClassifier::new(&phonemes, config);

        // Set weights so AA responds to dim 0, IY to dim 1
        classifier.weights[0] = vec![1.0, 0.0, 0.0];
        classifier.weights[1] = vec![0.0, 1.0, 0.0];

        let logits = classifier.compute_logits(&[5.0, 3.0, 0.0]);
        assert!((logits[0] - 5.0).abs() < 1e-6, "AA logit = {}", logits[0]);
        assert!((logits[1] - 3.0).abs() < 1e-6, "IY logit = {}", logits[1]);
    }

    #[test]
    fn test_direct_classifier_decode() {
        let phonemes = vec!["AA".to_string(), "IY".to_string()];
        let config = DirectClassifierConfig {
            reservoir_size: 3,
            ..Default::default()
        };
        let mut classifier = DirectClassifier::new(&phonemes, config);

        classifier.weights[0] = vec![1.0, 0.0, 0.0];
        classifier.weights[1] = vec![0.0, 1.0, 0.0];

        let (label, conf) = classifier.decode(&[5.0, 3.0, 0.0]);
        assert_eq!(label, "AA");
        assert!(conf > 0.0 && conf <= 1.0, "confidence = {conf}");
    }

    #[test]
    fn test_direct_classifier_normalize_weights() {
        let phonemes = vec!["AA".to_string()];
        let config = DirectClassifierConfig {
            reservoir_size: 2,
            ..Default::default()
        };
        let mut classifier = DirectClassifier::new(&phonemes, config);
        classifier.weights[0] = vec![3.0, 4.0]; // norm = 5

        classifier.normalize_weights();

        let norm: f32 = classifier.weights[0]
            .iter()
            .map(|w| w * w)
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "normalized norm should be 1.0, got {norm}"
        );
    }

    #[test]
    fn test_direct_classifier_set_log_priors() {
        let phonemes = vec!["AA".to_string(), "IY".to_string()];
        let config = DirectClassifierConfig {
            reservoir_size: 2,
            ..Default::default()
        };
        let mut classifier = DirectClassifier::new(&phonemes, config);

        classifier.set_log_priors(&[100, 100], 1.0);
        // Equal counts -> equal log priors
        assert!(
            (classifier.log_priors[0] - classifier.log_priors[1]).abs() < 1e-6,
            "equal counts should give equal priors"
        );

        classifier.set_log_priors(&[90, 10], 1.0);
        // More AA should give higher log prior for AA
        assert!(
            classifier.log_priors[0] > classifier.log_priors[1],
            "AA (90 count) should have higher log prior than IY (10 count)"
        );
    }

    #[test]
    fn test_direct_accumulator_add_and_solve() {
        let mut acc = DirectAccumulator::new(3, 2);
        assert_eq!(acc.n_samples, 0);

        // Train: state [1,0,0] -> class 0, state [0,1,0] -> class 1
        for _ in 0..10 {
            acc.add(&[1.0, 0.0, 0.0], 0);
            acc.add(&[0.0, 1.0, 0.0], 1);
        }

        assert_eq!(acc.n_samples, 20);
        assert_eq!(acc.class_counts()[0], 10);
        assert_eq!(acc.class_counts()[1], 10);

        let weights = acc.solve(0.01);
        assert_eq!(weights.len(), 2); // n_classes
        assert_eq!(weights[0].len(), 3); // input_dim
    }

    #[test]
    fn test_direct_accumulator_ignores_invalid_class() {
        let mut acc = DirectAccumulator::new(2, 2);
        acc.add(&[1.0, 0.0], 5); // class 5 is out of range
        assert_eq!(acc.n_samples, 0); // Should not be counted
    }

    #[test]
    fn test_random_projection_deterministic() {
        let rp1 = RandomProjection::new(10, 5, 42);
        let rp2 = RandomProjection::new(10, 5, 42);

        // Same seed should produce same weights
        for i in 0..5 {
            for j in 0..10 {
                assert_eq!(
                    rp1.weights[i][j], rp2.weights[i][j],
                    "weights differ at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn test_random_projection_forward_tanh() {
        let rp = RandomProjection::new(3, 2, 42);
        let output = rp.forward(&[1.0, 0.0, -1.0]);

        assert_eq!(output.len(), 2);
        // Tanh output should be in [-1, 1]
        for &v in &output {
            assert!(v >= -1.0 && v <= 1.0, "tanh output out of range: {v}");
        }
    }

    #[test]
    fn test_random_projection_forward_relu() {
        let rp = RandomProjection::with_activation(3, 2, 42, RFActivation::ReLU);
        let output = rp.forward(&[1.0, 0.0, -1.0]);

        assert_eq!(output.len(), 2);
        // ReLU output should be >= 0
        for &v in &output {
            assert!(v >= 0.0, "ReLU output should be non-negative: {v}");
        }
    }

    #[test]
    fn test_random_projection_different_seeds() {
        let rp1 = RandomProjection::new(5, 3, 1);
        let rp2 = RandomProjection::new(5, 3, 2);

        // Different seeds should produce different weights
        let different = rp1
            .weights
            .iter()
            .zip(rp2.weights.iter())
            .any(|(r1, r2)| r1.iter().zip(r2.iter()).any(|(a, b)| (a - b).abs() > 1e-10));
        assert!(
            different,
            "different seeds should produce different weights"
        );
    }

    #[test]
    fn test_liquid_projection_new() {
        let phonemes = vec!["AA".to_string(), "IY".to_string()];
        let config = LiquidProjectionConfig {
            reservoir_size: 16,
            ..Default::default()
        };
        let lp = LiquidProjection::new(&phonemes, config);

        assert_eq!(lp.targets.len(), 2);
        assert!(lp.targets.get("AA").is_some());
        assert!(lp.targets.get("IY").is_some());
    }

    #[test]
    fn test_liquid_projection_encode_decode_roundtrip() {
        let phonemes = vec!["AA".to_string(), "IY".to_string()];
        let config = LiquidProjectionConfig {
            reservoir_size: 4,
            ..Default::default()
        };
        let lp = LiquidProjection::new(&phonemes, config);

        // With zero readout weights, encode should produce zero HV
        let state = vec![1.0, 2.0, 3.0, 4.0];
        let hv = lp.encode(&state);
        let popcount: u32 = hv.words.iter().map(|w| w.count_ones()).sum();
        assert_eq!(popcount, 0, "zero readout should produce zero HV");
    }
}
