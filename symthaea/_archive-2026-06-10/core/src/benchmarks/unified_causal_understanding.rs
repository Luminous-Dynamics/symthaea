// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Causal Understanding
//!
//! This module implements a deep architecture for universal causal discovery
//! based on three fundamental principles:
//!
//! 1. **Compression Asymmetry** (HDC): K(X) + K(Y|X) < K(Y) + K(X|Y) for X→Y
//! 2. **Interventional Invariance** (LTC): Causal mechanisms are stable under do()
//! 3. **Directional Integration** (Phi): Information flows from cause to effect
//!
//! Unlike heuristic ensembles that vote on features, this architecture builds
//! generative models and asks counterfactual questions.

use crate::hdc::{CausalDirection, CausalDiscoveryResult, CausalFeatures};

/// Helper function to create a CausalDiscoveryResult with all required fields
fn make_result(
    direction: CausalDirection,
    p_forward: f64,
    confidence: f64,
) -> CausalDiscoveryResult {
    CausalDiscoveryResult {
        direction,
        p_forward,
        p_backward: 1.0 - p_forward,
        confidence,
        features: CausalFeatures {
            reci_score: 0.0,
            igci_score: 0.0,
            anm_score: 0.0,
            higher_order_score: 0.0,
        },
    }
}

// ============================================================================
// PART 1: HDC COMPRESSION CODEBOOKS
// ============================================================================

/// HDC-based compression codebook for measuring description length
///
/// The key insight: if X→Y, then encoding X first and Y|X second
/// should be more efficient than encoding Y first and X|Y second.
///
/// We use HDC bundles as a form of learned compression:
/// - Build a codebook of prototype vectors for the input distribution
/// - Measure how well the codebook compresses conditional distributions
/// - Compare compression efficiency in both directions
pub struct CompressionCodebook {
    /// Dimension of hypervectors
    dim: usize,
    /// Number of codebook entries (prototypes)
    num_prototypes: usize,
    /// Codebook vectors (learned prototypes)
    prototypes: Vec<Vec<f32>>,
    /// Quantization levels for continuous values
    quantization_levels: usize,
}

impl CompressionCodebook {
    pub fn new(dim: usize, num_prototypes: usize) -> Self {
        Self {
            dim,
            num_prototypes,
            prototypes: Vec::new(),
            quantization_levels: 64,
        }
    }

    /// Learn a codebook from data using k-means-style clustering in HDC space
    pub fn learn(&mut self, data: &[f64]) {
        if data.is_empty() {
            return;
        }

        // Convert data points to hypervectors
        let hvs: Vec<Vec<f32>> = data.iter().map(|&v| self.value_to_hv(v, data)).collect();

        // Initialize prototypes using k-means++ style selection
        self.prototypes = self.initialize_prototypes(&hvs);

        // Refine prototypes using Lloyd's algorithm in HDC space
        for _ in 0..10 {
            self.refine_prototypes(&hvs);
        }
    }

    /// Encode a value as a hypervector using thermometer + random projection
    fn value_to_hv(&self, value: f64, context: &[f64]) -> Vec<f32> {
        let min_val = context.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = context.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_val - min_val).max(1e-10);
        let normalized = ((value - min_val) / range).clamp(0.0, 1.0);

        // Create thermometer encoding in HDC space
        let level = (normalized * self.quantization_levels as f64) as usize;
        let mut hv = vec![0.0f32; self.dim];

        // Deterministic pseudo-random projection based on level
        for i in 0..self.dim {
            let seed = (i as u64).wrapping_mul(31337).wrapping_add(level as u64);
            let hash = seed.wrapping_mul(0x517cc1b727220a95);
            let bit = ((hash >> 32) as f32) / (u32::MAX as f32);

            // Thermometer: positions below level are positive
            if (hash % self.quantization_levels as u64) < level as u64 {
                hv[i] = bit * 2.0 - 1.0;
            } else {
                hv[i] = -(bit * 2.0 - 1.0);
            }
        }

        self.normalize(&mut hv);
        hv
    }

    /// Initialize prototypes using k-means++ style selection
    fn initialize_prototypes(&self, hvs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if hvs.is_empty() {
            return vec![vec![0.0; self.dim]; self.num_prototypes];
        }

        let mut prototypes = Vec::with_capacity(self.num_prototypes);

        // First prototype: random selection
        prototypes.push(hvs[0].clone());

        // Subsequent prototypes: probabilistic selection based on distance
        for _ in 1..self.num_prototypes.min(hvs.len()) {
            let mut max_min_dist = 0.0f32;
            let mut best_idx = 0;

            for (i, hv) in hvs.iter().enumerate() {
                let min_dist = prototypes
                    .iter()
                    .map(|p| 1.0 - self.cosine_similarity(hv, p))
                    .fold(f32::INFINITY, f32::min);

                if min_dist > max_min_dist {
                    max_min_dist = min_dist;
                    best_idx = i;
                }
            }

            prototypes.push(hvs[best_idx].clone());
        }

        // Pad with random if needed
        while prototypes.len() < self.num_prototypes {
            let mut random_hv = vec![0.0f32; self.dim];
            for i in 0..self.dim {
                let seed = (prototypes.len() * self.dim + i) as u64;
                let hash = seed.wrapping_mul(0x517cc1b727220a95);
                random_hv[i] = ((hash >> 32) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            }
            self.normalize(&mut random_hv);
            prototypes.push(random_hv);
        }

        prototypes
    }

    /// Refine prototypes using Lloyd's algorithm
    fn refine_prototypes(&mut self, hvs: &[Vec<f32>]) {
        if hvs.is_empty() || self.prototypes.is_empty() {
            return;
        }

        // Assign each point to nearest prototype
        let mut assignments: Vec<Vec<&Vec<f32>>> = vec![Vec::new(); self.num_prototypes];

        for hv in hvs {
            let mut best_proto = 0;
            let mut best_sim = f32::NEG_INFINITY;

            for (i, proto) in self.prototypes.iter().enumerate() {
                let sim = self.cosine_similarity(hv, proto);
                if sim > best_sim {
                    best_sim = sim;
                    best_proto = i;
                }
            }

            assignments[best_proto].push(hv);
        }

        // Update prototypes to be centroid of assigned points
        for (i, assigned) in assignments.iter().enumerate() {
            if !assigned.is_empty() {
                let mut centroid = vec![0.0f32; self.dim];
                for hv in assigned {
                    for j in 0..self.dim {
                        centroid[j] += hv[j];
                    }
                }
                for j in 0..self.dim {
                    centroid[j] /= assigned.len() as f32;
                }
                self.normalize(&mut centroid);
                self.prototypes[i] = centroid;
            }
        }
    }

    /// Compute compression cost (negative log-likelihood proxy)
    /// Lower cost = better compression = data fits codebook well
    pub fn compression_cost(&self, data: &[f64], context: &[f64]) -> f64 {
        if data.is_empty() || self.prototypes.is_empty() {
            return 0.0;
        }

        let mut total_cost = 0.0;

        for &value in data {
            let hv = self.value_to_hv(value, context);

            // Find best matching prototype
            let best_sim = self
                .prototypes
                .iter()
                .map(|p| self.cosine_similarity(&hv, p))
                .fold(f32::NEG_INFINITY, f32::max);

            // Cost is negative log of similarity (higher sim = lower cost)
            // Add small epsilon to avoid log(0)
            let normalized_sim = (best_sim + 1.0) / 2.0; // Map [-1,1] to [0,1]
            total_cost -= (normalized_sim.max(0.001) as f64).ln();
        }

        total_cost / data.len() as f64
    }

    /// Compute conditional compression cost: cost of encoding Y given X's codebook
    pub fn conditional_compression_cost(&self, y: &[f64], x: &[f64]) -> f64 {
        if y.is_empty() || x.is_empty() || y.len() != x.len() {
            return 0.0;
        }

        // For each x value, learn what y values co-occur
        // Then measure how well y is predicted/compressed given x

        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let x_range = (x_max - x_min).max(1e-10);

        // Bin x values and collect corresponding y values
        let num_bins = 16;
        let mut bins: Vec<Vec<f64>> = vec![Vec::new(); num_bins];

        for i in 0..x.len() {
            let bin = (((x[i] - x_min) / x_range) * (num_bins - 1) as f64)
                .clamp(0.0, (num_bins - 1) as f64) as usize;
            bins[bin].push(y[i]);
        }

        // For each bin, measure how concentrated (compressible) the y values are
        let mut total_cost = 0.0;
        let mut total_count = 0;

        for bin_y in &bins {
            if bin_y.len() < 2 {
                continue;
            }

            // Variance as proxy for compressibility (lower variance = more compressible)
            let mean: f64 = bin_y.iter().sum::<f64>() / bin_y.len() as f64;
            let variance: f64 =
                bin_y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / bin_y.len() as f64;

            // Cost proportional to log-variance (entropy-like)
            total_cost += (variance.max(1e-10)).ln() * bin_y.len() as f64;
            total_count += bin_y.len();
        }

        if total_count > 0 {
            total_cost / total_count as f64
        } else {
            0.0
        }
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        symthaea_core::math::cosine_similarity_f32(a, b)
    }

    fn normalize(&self, v: &mut Vec<f32>) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
}

/// Compression-based causal discovery
///
/// Measures: K(X) + K(Y|X) vs K(Y) + K(X|Y)
/// The direction with lower total description length is causal
pub struct CompressionCausalDiscovery {
    dim: usize,
    num_prototypes: usize,
}

impl CompressionCausalDiscovery {
    pub fn new() -> Self {
        Self {
            dim: 512,
            num_prototypes: 16,
        }
    }

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        // Learn codebooks for X and Y marginal distributions
        let mut codebook_x = CompressionCodebook::new(self.dim, self.num_prototypes);
        let mut codebook_y = CompressionCodebook::new(self.dim, self.num_prototypes);

        codebook_x.learn(x);
        codebook_y.learn(y);

        // Compute compression costs
        let k_x = codebook_x.compression_cost(x, x); // K(X)
        let k_y = codebook_y.compression_cost(y, y); // K(Y)

        // Conditional costs
        let k_y_given_x = codebook_x.conditional_compression_cost(y, x); // K(Y|X)
        let k_x_given_y = codebook_y.conditional_compression_cost(x, y); // K(X|Y)

        // Total description lengths
        let forward_cost = k_x + k_y_given_x; // Cost if X→Y
        let backward_cost = k_y + k_x_given_y; // Cost if Y→X

        // Lower cost = more likely causal direction
        let asymmetry = backward_cost - forward_cost; // Positive if X→Y

        let p_forward = 1.0 / (1.0 + (-asymmetry * 2.0).exp());
        let confidence = (p_forward - 0.5).abs() * 2.0;

        make_result(
            if p_forward > 0.5 {
                CausalDirection::Forward
            } else {
                CausalDirection::Backward
            },
            p_forward,
            confidence,
        )
    }
}

impl Default for CompressionCausalDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 2: LTC INTERVENTIONAL DYNAMICS
// ============================================================================

/// Simplified LTC-inspired dynamics model for causal discovery
///
/// Key insight: If X→Y, then the mechanism P(Y|X) should be INVARIANT
/// when we intervene on X (change its distribution). But P(X|Y) will break
/// because it depends on the natural P(X).
pub struct InterventionalDynamics {
    /// Number of hidden units
    hidden_size: usize,
    /// Time constant range
    tau_range: (f64, f64),
    /// Learned weights (input → hidden)
    w_in: Vec<Vec<f64>>,
    /// Learned weights (hidden → output)
    w_out: Vec<Vec<f64>>,
    /// Time constants
    tau: Vec<f64>,
    /// Hidden state
    state: Vec<f64>,
}

impl InterventionalDynamics {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            tau_range: (0.5, 5.0),
            w_in: Vec::new(),
            w_out: Vec::new(),
            tau: Vec::new(),
            state: vec![0.0; hidden_size],
        }
    }

    /// Learn dynamics from paired observations
    pub fn learn(&mut self, inputs: &[f64], outputs: &[f64]) {
        if inputs.len() != outputs.len() || inputs.is_empty() {
            return;
        }

        let n = inputs.len();

        // Initialize weights randomly but deterministically
        self.w_in = vec![vec![0.0; 1]; self.hidden_size];
        self.w_out = vec![vec![0.0; self.hidden_size]; 1];
        self.tau = vec![1.0; self.hidden_size];

        for i in 0..self.hidden_size {
            let seed = i as u64 * 31337;
            self.w_in[i][0] =
                ((seed.wrapping_mul(0x517cc1b727220a95) >> 32) as f64) / (u32::MAX as f64) * 2.0
                    - 1.0;
            self.tau[i] = self.tau_range.0
                + ((seed.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as f64) / (u32::MAX as f64)
                    * (self.tau_range.1 - self.tau_range.0);
        }

        for j in 0..self.hidden_size {
            let seed = (j + self.hidden_size) as u64 * 31337;
            self.w_out[0][j] =
                ((seed.wrapping_mul(0x517cc1b727220a95) >> 32) as f64) / (u32::MAX as f64) * 2.0
                    - 1.0;
        }

        // Normalize inputs and outputs
        let in_mean: f64 = inputs.iter().sum::<f64>() / n as f64;
        let in_std: f64 = (inputs.iter().map(|x| (x - in_mean).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-10);
        let out_mean: f64 = outputs.iter().sum::<f64>() / n as f64;
        let out_std: f64 = (outputs.iter().map(|x| (x - out_mean).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-10);

        // Simple gradient descent to learn weights
        let lr = 0.01;
        let dt = 0.1;

        for _epoch in 0..50 {
            self.state = vec![0.0; self.hidden_size];
            let mut total_loss = 0.0;

            for t in 0..n {
                let x_norm = (inputs[t] - in_mean) / in_std;
                let y_target = (outputs[t] - out_mean) / out_std;

                // Forward pass with LTC dynamics
                for i in 0..self.hidden_size {
                    let input_contrib = self.w_in[i][0] * x_norm;
                    let decay = -self.state[i] / self.tau[i];
                    self.state[i] += dt * (decay + input_contrib.tanh());
                }

                // Compute output
                let mut y_pred = 0.0;
                for j in 0..self.hidden_size {
                    y_pred += self.w_out[0][j] * self.state[j].tanh();
                }

                let error = y_pred - y_target;
                total_loss += error * error;

                // Backprop (simplified)
                for j in 0..self.hidden_size {
                    let grad = error * self.state[j].tanh();
                    self.w_out[0][j] -= lr * grad;
                }
            }

            if total_loss / (n as f64) < 0.01 {
                break;
            }
        }
    }

    /// Predict output given input
    pub fn predict(&mut self, inputs: &[f64], context_in: &[f64], context_out: &[f64]) -> Vec<f64> {
        let n = inputs.len();
        if n == 0 || context_in.is_empty() || context_out.is_empty() {
            return vec![];
        }

        // Normalize using context statistics
        let in_mean: f64 = context_in.iter().sum::<f64>() / context_in.len() as f64;
        let in_std: f64 = (context_in
            .iter()
            .map(|x| (x - in_mean).powi(2))
            .sum::<f64>()
            / context_in.len() as f64)
            .sqrt()
            .max(1e-10);
        let out_mean: f64 = context_out.iter().sum::<f64>() / context_out.len() as f64;
        let out_std: f64 = (context_out
            .iter()
            .map(|x| (x - out_mean).powi(2))
            .sum::<f64>()
            / context_out.len() as f64)
            .sqrt()
            .max(1e-10);

        self.state = vec![0.0; self.hidden_size];
        let dt = 0.1;
        let mut predictions = Vec::with_capacity(n);

        for t in 0..n {
            let x_norm = (inputs[t] - in_mean) / in_std;

            // LTC dynamics
            for i in 0..self.hidden_size {
                let input_contrib = self.w_in[i][0] * x_norm;
                let decay = -self.state[i] / self.tau[i];
                self.state[i] += dt * (decay + input_contrib.tanh());
            }

            // Output
            let mut y_pred = 0.0;
            for j in 0..self.hidden_size {
                y_pred += self.w_out[0][j] * self.state[j].tanh();
            }

            // Denormalize
            predictions.push(y_pred * out_std + out_mean);
        }

        predictions
    }

    /// Test invariance under intervention
    ///
    /// If the mechanism is truly causal (X→Y), it should work even when
    /// we change the input distribution (intervention).
    pub fn intervention_invariance(&mut self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 10 {
            return 0.5;
        }

        let n = x.len();

        // Split data: train on first half, test on second half
        let train_n = n / 2;
        let test_n = n - train_n;

        // Train on natural distribution
        self.learn(&x[..train_n], &y[..train_n]);

        // Test 1: Prediction on held-out natural data
        let natural_preds = self.predict(&x[train_n..], x, y);
        let natural_mse = if natural_preds.len() == test_n {
            natural_preds
                .iter()
                .zip(y[train_n..].iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / test_n as f64
        } else {
            f64::MAX
        };

        // Test 2: Prediction on INTERVENED data (uniform x distribution)
        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Generate uniform intervention values
        let intervened_x: Vec<f64> = (0..test_n)
            .map(|i| x_min + (x_max - x_min) * (i as f64 / test_n as f64))
            .collect();

        let intervened_preds = self.predict(&intervened_x, x, y);

        // For true causal direction, predictions should still be reasonable
        // (follow the learned functional relationship)
        // We measure "reasonableness" by variance of predictions
        let pred_mean: f64 = intervened_preds.iter().sum::<f64>() / test_n as f64;
        let pred_var: f64 = intervened_preds
            .iter()
            .map(|p| (p - pred_mean).powi(2))
            .sum::<f64>()
            / test_n as f64;

        let y_var: f64 = {
            let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
            y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n as f64
        };

        // Invariance score: ratio of explained variance under intervention
        // Higher = more invariant = more likely causal
        if y_var > 0.0 && natural_mse < y_var {
            let natural_r2 = 1.0 - natural_mse / y_var;
            let intervention_ratio = (pred_var / y_var).min(2.0); // Cap at 2

            // Good causal mechanism: low natural error, reasonable intervention variance
            natural_r2 * 0.5 + intervention_ratio.min(1.0) * 0.5
        } else {
            0.0
        }
    }
}

/// LTC-based interventional causal discovery
pub struct InterventionalCausalDiscovery {
    hidden_size: usize,
}

impl InterventionalCausalDiscovery {
    pub fn new() -> Self {
        Self { hidden_size: 8 }
    }

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let mut forward_model = InterventionalDynamics::new(self.hidden_size);
        let mut backward_model = InterventionalDynamics::new(self.hidden_size);

        // Test invariance in both directions
        let forward_invariance = forward_model.intervention_invariance(x, y);
        let backward_invariance = backward_model.intervention_invariance(y, x);

        // Higher invariance = more likely to be the causal direction
        let asymmetry = forward_invariance - backward_invariance;

        let p_forward = 1.0 / (1.0 + (-asymmetry * 3.0).exp());
        let confidence = (p_forward - 0.5).abs() * 2.0;

        make_result(
            if p_forward > 0.5 {
                CausalDirection::Forward
            } else {
                CausalDirection::Backward
            },
            p_forward,
            confidence,
        )
    }
}

impl Default for InterventionalCausalDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 3: DIRECTIONAL PHI (INTEGRATED INFORMATION FLOW)
// ============================================================================

/// Directional Phi: measures asymmetric information flow via residual analysis
///
/// Key insight from IIT: The causal mechanism P(Y|X) should be "simpler" than
/// the anti-causal P(X|Y). We measure this via regression residual properties.
///
/// If X→Y (X causes Y):
/// - Y = f(X) + noise, where noise is independent of X
/// - Residuals from regressing Y on X should be uncorrelated with X
/// - The forward mechanism is simpler (lower description length)
///
/// We use RECI-style residual correlation asymmetry combined with
/// entropy-based integration measures.
pub struct DirectionalPhi {
    bandwidth: f64,
}

impl Default for DirectionalPhi {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectionalPhi {
    pub fn new() -> Self {
        Self { bandwidth: 0.5 }
    }

    /// Nadaraya-Watson kernel regression: E[Y|X=x]
    fn kernel_regression(&self, x: &[f64], y: &[f64], x_query: f64, bandwidth: f64) -> f64 {
        let mut weight_sum = 0.0;
        let mut weighted_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let diff = (xi - x_query) / bandwidth;
            let weight = (-0.5 * diff * diff).exp();
            weight_sum += weight;
            weighted_y += weight * yi;
        }

        if weight_sum > 1e-10 {
            weighted_y / weight_sum
        } else {
            y.iter().sum::<f64>() / y.len() as f64
        }
    }

    /// Compute residuals from regressing y on x
    fn compute_residuals(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        // Adaptive bandwidth based on data spread
        let x_std = std_dev(x);
        let bandwidth = x_std * self.bandwidth;
        let bw = bandwidth.max(1e-6);

        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| {
                let prediction = self.kernel_regression(x, y, xi, bw);
                yi - prediction
            })
            .collect()
    }

    /// Measure correlation between residuals and predictor
    /// Lower correlation = better causal model (noise independent of cause)
    fn residual_correlation(&self, predictor: &[f64], residuals: &[f64]) -> f64 {
        let n = predictor.len() as f64;
        if n < 3.0 {
            return 0.0;
        }

        let mean_p: f64 = predictor.iter().sum::<f64>() / n;
        let mean_r: f64 = residuals.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_p = 0.0;
        let mut var_r = 0.0;

        for (&p, &r) in predictor.iter().zip(residuals.iter()) {
            let dp = p - mean_p;
            let dr = r - mean_r;
            cov += dp * dr;
            var_p += dp * dp;
            var_r += dr * dr;
        }

        if var_p > 1e-10 && var_r > 1e-10 {
            (cov / (var_p.sqrt() * var_r.sqrt())).abs()
        } else {
            0.0
        }
    }

    /// Compute differential entropy estimate (via histogram)
    fn differential_entropy(&self, data: &[f64]) -> f64 {
        let n = data.len();
        if n < 10 {
            return 0.0;
        }

        let num_bins = ((n as f64).sqrt() as usize).max(5).min(20);
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_val - min_val).max(1e-10);
        let bin_width = range / num_bins as f64;

        let mut counts = vec![0usize; num_bins];
        for &v in data {
            let bin = (((v - min_val) / range) * (num_bins - 1) as f64) as usize;
            counts[bin.min(num_bins - 1)] += 1;
        }

        // Entropy = -sum(p * log(p)) + log(bin_width) for differential
        let n_f = n as f64;
        let mut entropy = 0.0;
        for count in counts {
            if count > 0 {
                let p = count as f64 / n_f;
                entropy -= p * p.ln();
            }
        }
        entropy + bin_width.ln()
    }

    /// Compute Phi-style causal score for direction X→Y
    ///
    /// Combines:
    /// 1. Residual independence (RECI-style)
    /// 2. Residual entropy (simpler mechanisms have lower-entropy residuals)
    pub fn directional_phi(&self, x: &[f64], y: &[f64]) -> f64 {
        let residuals = self.compute_residuals(x, y);

        // Component 1: Residual correlation (lower is better for causal direction)
        let corr = self.residual_correlation(x, &residuals);

        // Component 2: Residual entropy (causal mechanism has structured residuals)
        let res_entropy = self.differential_entropy(&residuals);

        // Combined score: low correlation + moderate entropy = good causal fit
        // We want: independent noise (low corr) but not too structured (moderate entropy)
        let independence_score = 1.0 - corr; // Higher = more independent = better

        // The causal direction should have MORE independent residuals
        // and residuals with appropriate entropy structure
        independence_score * (1.0 + res_entropy.tanh() * 0.2)
    }
}

/// Helper: compute standard deviation
fn std_dev(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let mean = data.iter().sum::<f64>() / n;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt().max(1e-10)
}

/// Phi-based directional causal discovery
pub struct DirectionalPhiDiscovery {
    phi: DirectionalPhi,
}

impl DirectionalPhiDiscovery {
    pub fn new() -> Self {
        Self {
            phi: DirectionalPhi::new(),
        }
    }

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        // Compute directional Phi in both directions
        let phi_xy = self.phi.directional_phi(x, y);
        let phi_yx = self.phi.directional_phi(y, x);

        // Higher phi_xy = better X→Y model (more independent residuals)
        // So if phi_xy > phi_yx, predict X→Y
        let asymmetry = phi_xy - phi_yx;

        let p_forward = 1.0 / (1.0 + (-asymmetry * 5.0).exp());
        let confidence = (p_forward - 0.5).abs() * 2.0;

        make_result(
            if p_forward > 0.5 {
                CausalDirection::Forward
            } else {
                CausalDirection::Backward
            },
            p_forward,
            confidence,
        )
    }
}

impl Default for DirectionalPhiDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 4: UNIFIED CAUSAL REASONING
// ============================================================================

/// Evidence from each primitive for causal direction
#[derive(Debug, Clone)]
pub struct CausalEvidence {
    /// Compression asymmetry (HDC): positive = X→Y has shorter description
    pub compression: f64,
    /// Intervention invariance (LTC): positive = X→Y mechanism is more stable
    pub invariance: f64,
    /// Directional Phi: positive = more information flows X→Y
    pub phi_flow: f64,
    /// Overall confidence in the evidence
    pub confidence: f64,
}

/// Unified Causal Reasoning: integrates all three primitives
///
/// This is NOT a voting ensemble. Instead, it asks:
/// "Given all the evidence, what causal structure best explains it?"
pub struct UnifiedCausalReasoning {
    compression: CompressionCausalDiscovery,
    interventional: InterventionalCausalDiscovery,
    phi: DirectionalPhiDiscovery,
}

impl UnifiedCausalReasoning {
    pub fn new() -> Self {
        Self {
            compression: CompressionCausalDiscovery::new(),
            interventional: InterventionalCausalDiscovery::new(),
            phi: DirectionalPhiDiscovery::new(),
        }
    }

    /// Gather evidence from all three primitives
    pub fn gather_evidence(&self, x: &[f64], y: &[f64]) -> CausalEvidence {
        let comp_result = self.compression.discover(x, y);
        let inv_result = self.interventional.discover(x, y);
        let phi_result = self.phi.discover(x, y);

        // Convert to signed asymmetries
        let compression = (comp_result.p_forward - 0.5) * 2.0 * comp_result.confidence;
        let invariance = (inv_result.p_forward - 0.5) * 2.0 * inv_result.confidence;
        let phi_flow = (phi_result.p_forward - 0.5) * 2.0 * phi_result.confidence;

        // Confidence based on agreement
        let signs_agree =
            (compression >= 0.0) == (invariance >= 0.0) && (invariance >= 0.0) == (phi_flow >= 0.0);

        let avg_magnitude = (compression.abs() + invariance.abs() + phi_flow.abs()) / 3.0;
        let confidence = if signs_agree {
            avg_magnitude.min(1.0)
        } else {
            avg_magnitude * 0.5 // Reduce confidence when primitives disagree
        };

        CausalEvidence {
            compression,
            invariance,
            phi_flow,
            confidence,
        }
    }

    /// Unified reasoning: integrate evidence into a coherent verdict
    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let evidence = self.gather_evidence(x, y);

        // Weighted combination based on theoretical soundness:
        // - Compression: Based on algorithmic information theory (fundamental)
        // - Invariance: Based on causal mechanism stability (empirically strong)
        // - Phi: Based on integrated information flow (complementary)

        let asymmetry =
            evidence.compression * 0.35 + evidence.invariance * 0.40 + evidence.phi_flow * 0.25;

        let p_forward = 1.0 / (1.0 + (-asymmetry * 3.0).exp());

        make_result(
            if p_forward > 0.5 {
                CausalDirection::Forward
            } else {
                CausalDirection::Backward
            },
            p_forward,
            evidence.confidence,
        )
    }

    /// Explain the reasoning behind the verdict
    pub fn explain(&self, x: &[f64], y: &[f64]) -> String {
        let evidence = self.gather_evidence(x, y);
        let result = self.discover(x, y);

        let direction_str = match result.direction {
            CausalDirection::Forward => "X → Y",
            CausalDirection::Backward => "Y → X",
        };

        format!(
            "Causal Verdict: {} (confidence: {:.1}%)\n\
             \n\
             Evidence:\n\
             - Compression: {:.3} ({})\n\
             - Invariance:  {:.3} ({})\n\
             - Phi Flow:    {:.3} ({})\n\
             \n\
             Reasoning: {}",
            direction_str,
            result.confidence * 100.0,
            evidence.compression,
            if evidence.compression > 0.0 {
                "supports X→Y"
            } else {
                "supports Y→X"
            },
            evidence.invariance,
            if evidence.invariance > 0.0 {
                "supports X→Y"
            } else {
                "supports Y→X"
            },
            evidence.phi_flow,
            if evidence.phi_flow > 0.0 {
                "supports X→Y"
            } else {
                "supports Y→X"
            },
            if evidence.confidence > 0.7 {
                "Strong agreement across all primitives."
            } else if evidence.confidence > 0.4 {
                "Moderate agreement; some uncertainty remains."
            } else {
                "Primitives disagree; relationship may be complex or bidirectional."
            }
        )
    }
}

impl Default for UnifiedCausalReasoning {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WRAPPER FUNCTIONS FOR BENCHMARK INTEGRATION
// ============================================================================

/// Discover using HDC compression asymmetry
pub fn discover_by_compression(x: &[f64], y: &[f64]) -> CausalDirection {
    CompressionCausalDiscovery::new().discover(x, y).direction
}

/// Discover using LTC interventional invariance
pub fn discover_by_intervention(x: &[f64], y: &[f64]) -> CausalDirection {
    InterventionalCausalDiscovery::new()
        .discover(x, y)
        .direction
}

/// Discover using directional Phi flow
pub fn discover_by_directional_phi(x: &[f64], y: &[f64]) -> CausalDirection {
    DirectionalPhiDiscovery::new().discover(x, y).direction
}

/// Discover using unified causal reasoning
pub fn discover_by_unified_reasoning(x: &[f64], y: &[f64]) -> CausalDirection {
    UnifiedCausalReasoning::new().discover(x, y).direction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_codebook() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut codebook = CompressionCodebook::new(256, 4);
        codebook.learn(&data);

        let cost = codebook.compression_cost(&data, &data);
        assert!(cost.is_finite());
    }

    #[test]
    fn test_unified_reasoning() {
        // Simple linear relationship: Y = 2X + noise
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 * xi + 0.1 * (xi * 31.337).sin())
            .collect();

        let reasoner = UnifiedCausalReasoning::new();
        let result = reasoner.discover(&x, &y);

        // Should produce a result with valid confidence
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "Confidence must be in [0,1]: {}",
            result.confidence
        );
        // Explanation should be non-empty
        let explanation = reasoner.explain(&x, &y);
        assert!(!explanation.is_empty(), "Explanation should not be empty");
    }
}
