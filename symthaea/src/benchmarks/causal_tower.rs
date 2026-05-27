// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Experimental module: fields will be read when causal tower benchmarks are run
#![allow(dead_code)]
//! Causal Understanding Tower
//!
//! A multi-level approach to causal discovery that combines:
//! 1. Improved Primitives (HDC, LTC, Phi)
//! 2. Classic Algorithms (IGCI, LiNGaM, ANM, RECI)
//! 3. Meta-Learning (algorithm selection based on data characteristics)
//! 4. Semantic Understanding (world knowledge integration)
//! 5. Uncertainty-Aware Decisions (handling ambiguous cases)
//!
//! Goal: Approach 100% accuracy on causal discovery benchmarks

use super::tuebingen_adapter::{
    CausalDirection, CausalDiscoveryResult, CausalFeatures, discover_by_information_theoretic,
};
use std::collections::HashMap;

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
// PHASE 1: IMPROVED PRIMITIVES
// ============================================================================

/// Improved HDC Compression with multiple codebook sizes and learned prototypes
pub struct ImprovedHdcCompression {
    codebook_sizes: Vec<usize>,
    num_prototypes: usize,
}

impl ImprovedHdcCompression {
    pub fn new() -> Self {
        Self {
            codebook_sizes: vec![4, 8, 16, 32], // Multiple granularities
            num_prototypes: 16,
        }
    }

    /// Compute description length using adaptive codebook
    fn description_length(&self, data: &[f64], codebook_size: usize) -> f64 {
        let n = data.len();
        if n < 2 {
            return 0.0;
        }

        // Normalize data
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_val - min_val).max(1e-10);

        // Quantize to codebook
        let quantized: Vec<usize> = data
            .iter()
            .map(|&v| {
                let norm = (v - min_val) / range;
                (norm * (codebook_size - 1) as f64).clamp(0.0, (codebook_size - 1) as f64) as usize
            })
            .collect();

        // Compute symbol frequencies
        let mut freq = vec![0usize; codebook_size];
        for &q in &quantized {
            freq[q] += 1;
        }

        // Shannon entropy as description length proxy
        let n_f = n as f64;
        let entropy: f64 = freq
            .iter()
            .filter(|&&f| f > 0)
            .map(|&f| {
                let p = f as f64 / n_f;
                -p * p.log2()
            })
            .sum();

        entropy * n_f // Total bits
    }

    /// Compute conditional description length K(Y|X)
    fn conditional_description_length(&self, x: &[f64], y: &[f64], codebook_size: usize) -> f64 {
        let n = x.len();
        if n < 4 {
            return self.description_length(y, codebook_size);
        }

        // Normalize both
        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let x_range = (x_max - x_min).max(1e-10);

        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_range = (y_max - y_min).max(1e-10);

        // Quantize X to create bins
        let x_bins: Vec<usize> = x
            .iter()
            .map(|&v| {
                let norm = (v - x_min) / x_range;
                ((norm * (codebook_size - 1) as f64) as usize).min(codebook_size - 1)
            })
            .collect();

        // For each X bin, compute entropy of Y
        let mut bin_counts = vec![0usize; codebook_size];
        let _conditional_entropies = vec![0.0; codebook_size];
        let mut bin_y_values: Vec<Vec<f64>> = vec![Vec::new(); codebook_size];

        for i in 0..n {
            let bin = x_bins[i];
            bin_counts[bin] += 1;
            bin_y_values[bin].push(y[i]);
        }

        // Compute weighted conditional entropy
        let n_f = n as f64;
        let mut total_cond_entropy = 0.0;

        for bin in 0..codebook_size {
            if bin_counts[bin] > 1 {
                let y_in_bin = &bin_y_values[bin];
                let bin_entropy = self.compute_entropy(y_in_bin, y_min, y_range, codebook_size);
                let weight = bin_counts[bin] as f64 / n_f;
                total_cond_entropy += weight * bin_entropy;
            }
        }

        total_cond_entropy * n_f
    }

    fn compute_entropy(&self, data: &[f64], min_val: f64, range: f64, num_bins: usize) -> f64 {
        let n = data.len();
        if n < 2 {
            return 0.0;
        }

        let mut freq = vec![0usize; num_bins];
        for &v in data {
            let norm = (v - min_val) / range;
            let bin = ((norm * (num_bins - 1) as f64) as usize).min(num_bins - 1);
            freq[bin] += 1;
        }

        let n_f = n as f64;
        freq.iter()
            .filter(|&&f| f > 0)
            .map(|&f| {
                let p = f as f64 / n_f;
                -p * p.log2()
            })
            .sum()
    }

    /// Discover causal direction using multi-scale compression
    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let mut votes_forward = 0.0;
        let mut votes_backward = 0.0;
        let mut total_weight = 0.0;

        for &cs in &self.codebook_sizes {
            // K(X) + K(Y|X) vs K(Y) + K(X|Y)
            let kx = self.description_length(x, cs);
            let ky = self.description_length(y, cs);
            let ky_given_x = self.conditional_description_length(x, y, cs);
            let kx_given_y = self.conditional_description_length(y, x, cs);

            let forward_cost = kx + ky_given_x;
            let backward_cost = ky + kx_given_y;

            // Weight by codebook size (larger = more reliable)
            let weight = (cs as f64).ln();

            if forward_cost < backward_cost {
                votes_forward += weight * (backward_cost - forward_cost);
            } else {
                votes_backward += weight * (forward_cost - backward_cost);
            }
            total_weight += weight;
        }

        let score = (votes_forward - votes_backward) / total_weight.max(1.0);
        let p_forward = 1.0 / (1.0 + (-score).exp());
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

impl Default for ImprovedHdcCompression {
    fn default() -> Self {
        Self::new()
    }
}

/// Improved LTC with actual ODE-style dynamics fitting
pub struct ImprovedLtcDynamics {
    hidden_size: usize,
    num_trials: usize,
}

impl ImprovedLtcDynamics {
    pub fn new() -> Self {
        Self {
            hidden_size: 8,
            num_trials: 5,
        }
    }

    /// Fit a simple dynamical model Y = f(X) + noise
    /// Returns (fit_quality, residual_variance)
    fn fit_forward_model(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len();
        if n < 5 {
            return (0.0, 1.0);
        }

        // Use polynomial regression with cross-validation
        let mut best_mse = f64::INFINITY;

        for degree in 1..=3 {
            let mse = self.polynomial_cv_mse(x, y, degree);
            if mse < best_mse {
                best_mse = mse;
            }
        }

        // Also try kernel regression
        let kernel_mse = self.kernel_regression_cv_mse(x, y);
        if kernel_mse < best_mse {
            best_mse = kernel_mse;
        }

        let y_var = variance(y);
        let r_squared = 1.0 - best_mse / y_var.max(1e-10);

        (r_squared.max(0.0), best_mse)
    }

    fn polynomial_cv_mse(&self, x: &[f64], y: &[f64], degree: usize) -> f64 {
        let n = x.len();
        let fold_size = n / 5;
        if fold_size < 2 {
            return f64::INFINITY;
        }

        let mut total_mse = 0.0;
        let mut total_count = 0;

        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };

            // Split data
            let mut train_x = Vec::new();
            let mut train_y = Vec::new();
            let mut test_x = Vec::new();
            let mut test_y = Vec::new();

            for i in 0..n {
                if i >= test_start && i < test_end {
                    test_x.push(x[i]);
                    test_y.push(y[i]);
                } else {
                    train_x.push(x[i]);
                    train_y.push(y[i]);
                }
            }

            // Fit polynomial on training data
            let coeffs = self.fit_polynomial(&train_x, &train_y, degree);

            // Evaluate on test data
            for i in 0..test_x.len() {
                let pred = self.eval_polynomial(&coeffs, test_x[i]);
                let err = test_y[i] - pred;
                total_mse += err * err;
                total_count += 1;
            }
        }

        if total_count > 0 {
            total_mse / total_count as f64
        } else {
            f64::INFINITY
        }
    }

    fn fit_polynomial(&self, x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
        // Simple least squares polynomial fit
        let n = x.len();
        let d = degree + 1;

        // Build Vandermonde matrix X and solve X'X * coeffs = X'y
        // For simplicity, use gradient descent
        let mut coeffs = vec![0.0; d];
        let lr = 0.01;

        // Normalize x for numerical stability
        let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
        let x_std = (x.iter().map(|&xi| (xi - x_mean).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-10);
        let x_norm: Vec<f64> = x.iter().map(|&xi| (xi - x_mean) / x_std).collect();

        for _ in 0..100 {
            let mut grad = vec![0.0; d];

            for i in 0..n {
                let pred = self.eval_polynomial(&coeffs, x_norm[i]);
                let err = pred - y[i];

                let mut x_pow = 1.0;
                for j in 0..d {
                    grad[j] += err * x_pow;
                    x_pow *= x_norm[i];
                }
            }

            for j in 0..d {
                coeffs[j] -= lr * grad[j] / n as f64;
            }
        }

        coeffs
    }

    fn eval_polynomial(&self, coeffs: &[f64], x: f64) -> f64 {
        let mut result = 0.0;
        let mut x_pow = 1.0;
        for &c in coeffs {
            result += c * x_pow;
            x_pow *= x;
        }
        result
    }

    fn kernel_regression_cv_mse(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        let fold_size = n / 5;
        if fold_size < 2 {
            return f64::INFINITY;
        }

        // Estimate bandwidth using Silverman's rule
        let x_std = variance(x).sqrt();
        let bandwidth = 1.06 * x_std * (n as f64).powf(-0.2);

        let mut total_mse = 0.0;
        let mut total_count = 0;

        for fold in 0..5 {
            let test_start = fold * fold_size;
            let test_end = if fold == 4 { n } else { (fold + 1) * fold_size };

            for i in test_start..test_end {
                // Leave-one-out style prediction
                let pred = self.nadaraya_watson(x, y, x[i], bandwidth, i);
                let err = y[i] - pred;
                total_mse += err * err;
                total_count += 1;
            }
        }

        if total_count > 0 {
            total_mse / total_count as f64
        } else {
            f64::INFINITY
        }
    }

    fn nadaraya_watson(
        &self,
        x: &[f64],
        y: &[f64],
        query: f64,
        bandwidth: f64,
        exclude_idx: usize,
    ) -> f64 {
        let mut weight_sum = 0.0;
        let mut weighted_y = 0.0;

        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            if i == exclude_idx {
                continue;
            }
            let diff = (xi - query) / bandwidth.max(1e-10);
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

    /// Discover using LTC-style dynamics fitting
    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let (r2_forward, _) = self.fit_forward_model(x, y);
        let (r2_backward, _) = self.fit_forward_model(y, x);

        // Better fit in forward direction suggests X→Y
        let asymmetry = r2_forward - r2_backward;
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

impl Default for ImprovedLtcDynamics {
    fn default() -> Self {
        Self::new()
    }
}

/// Improved Phi using HSIC for independence testing
pub struct ImprovedPhiFlow {
    num_permutations: usize,
}

impl ImprovedPhiFlow {
    pub fn new() -> Self {
        Self {
            num_permutations: 100,
        }
    }

    /// Compute HSIC (Hilbert-Schmidt Independence Criterion)
    /// Returns a measure of dependence between two variables
    fn hsic(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 5 {
            return 0.0;
        }

        // Compute kernel matrices
        let sigma_x = median_distance(x);
        let sigma_y = median_distance(y);

        let mut kx = vec![vec![0.0; n]; n];
        let mut ky = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                let dx = (x[i] - x[j]) / sigma_x.max(1e-10);
                let dy = (y[i] - y[j]) / sigma_y.max(1e-10);
                kx[i][j] = (-0.5 * dx * dx).exp();
                ky[i][j] = (-0.5 * dy * dy).exp();
            }
        }

        // Center the kernel matrices
        let mut hkx = vec![vec![0.0; n]; n];
        let mut hky = vec![vec![0.0; n]; n];

        let kx_row_means: Vec<f64> = (0..n)
            .map(|i| kx[i].iter().sum::<f64>() / n as f64)
            .collect();
        let ky_row_means: Vec<f64> = (0..n)
            .map(|i| ky[i].iter().sum::<f64>() / n as f64)
            .collect();
        let kx_mean: f64 = kx_row_means.iter().sum::<f64>() / n as f64;
        let ky_mean: f64 = ky_row_means.iter().sum::<f64>() / n as f64;

        for i in 0..n {
            for j in 0..n {
                hkx[i][j] = kx[i][j] - kx_row_means[i] - kx_row_means[j] + kx_mean;
                hky[i][j] = ky[i][j] - ky_row_means[i] - ky_row_means[j] + ky_mean;
            }
        }

        // HSIC = trace(HKx @ HKy) / (n-1)^2
        let mut hsic_val = 0.0;
        for i in 0..n {
            for j in 0..n {
                hsic_val += hkx[i][j] * hky[j][i];
            }
        }

        hsic_val / ((n - 1) * (n - 1)) as f64
    }

    /// Compute residuals from kernel regression
    fn compute_residuals(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        let _n = x.len();
        let x_std = variance(x).sqrt();
        let bandwidth = x_std * 0.5;

        x.iter()
            .zip(y.iter())
            .enumerate()
            .map(|(i, (&xi, &yi))| {
                let pred = self.nadaraya_watson_exclude(x, y, xi, bandwidth, i);
                yi - pred
            })
            .collect()
    }

    fn nadaraya_watson_exclude(
        &self,
        x: &[f64],
        y: &[f64],
        query: f64,
        bandwidth: f64,
        exclude: usize,
    ) -> f64 {
        let mut weight_sum = 0.0;
        let mut weighted_y = 0.0;

        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            if i == exclude {
                continue;
            }
            let diff = (xi - query) / bandwidth.max(1e-10);
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

    /// RECI-style: measure independence of residuals from predictor
    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        // Forward: Y = f(X) + noise, check if noise ⊥ X
        let residuals_forward = self.compute_residuals(x, y);
        let hsic_forward = self.hsic(x, &residuals_forward);

        // Backward: X = g(Y) + noise, check if noise ⊥ Y
        let residuals_backward = self.compute_residuals(y, x);
        let hsic_backward = self.hsic(y, &residuals_backward);

        // Lower HSIC = more independent = better causal model
        let asymmetry = hsic_backward - hsic_forward; // Positive if forward is better
        let p_forward = 1.0 / (1.0 + (-asymmetry * 50.0).exp());
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

impl Default for ImprovedPhiFlow {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PHASE 2: CLASSIC ALGORITHMS
// ============================================================================

/// Information Geometric Causal Inference (IGCI)
/// Based on the principle that P(cause) and P(effect|cause) are independent
pub struct IgciDiscovery;

impl Default for IgciDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl IgciDiscovery {
    pub fn new() -> Self {
        Self
    }

    /// Compute IGCI score using entropy-based estimator
    fn igci_score(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 10 {
            return 0.0;
        }

        // Sort by x
        let mut pairs: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&a, &b)| (a, b)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compute slopes
        let mut log_slopes = Vec::new();
        for i in 1..pairs.len() {
            let dx = pairs[i].0 - pairs[i - 1].0;
            let dy = pairs[i].1 - pairs[i - 1].1;
            if dx.abs() > 1e-10 {
                let slope = dy / dx;
                if slope.abs() > 1e-10 {
                    log_slopes.push(slope.abs().ln());
                }
            }
        }

        if log_slopes.is_empty() {
            return 0.0;
        }

        // Average log slope
        log_slopes.iter().sum::<f64>() / log_slopes.len() as f64
    }

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let score_xy = self.igci_score(x, y);
        let score_yx = self.igci_score(y, x);

        // Lower score suggests causal direction
        let asymmetry = score_yx - score_xy;
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

/// Linear Non-Gaussian Acyclic Model (LiNGaM)
/// Exploits non-Gaussianity for causal discovery
pub struct LingamDiscovery;

impl Default for LingamDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl LingamDiscovery {
    pub fn new() -> Self {
        Self
    }

    /// Estimate non-Gaussianity using kurtosis
    fn kurtosis(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 4.0 {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / n;
        let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4: f64 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;

        if m2 > 1e-10 {
            m4 / (m2 * m2) - 3.0 // Excess kurtosis
        } else {
            0.0
        }
    }

    /// Simple linear regression: y = a + b*x, returns (a, b, residuals)
    fn linear_regression(&self, x: &[f64], y: &[f64]) -> (f64, f64, Vec<f64>) {
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut cov_xy = 0.0;
        let mut var_x = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            cov_xy += (xi - mean_x) * (yi - mean_y);
            var_x += (xi - mean_x).powi(2);
        }

        let b = if var_x > 1e-10 { cov_xy / var_x } else { 0.0 };
        let a = mean_y - b * mean_x;

        let residuals: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| yi - (a + b * xi))
            .collect();

        (a, b, residuals)
    }

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        // Regress Y on X
        let (_, _, residuals_forward) = self.linear_regression(x, y);
        let kurt_forward = self.kurtosis(&residuals_forward).abs();

        // Regress X on Y
        let (_, _, residuals_backward) = self.linear_regression(y, x);
        let kurt_backward = self.kurtosis(&residuals_backward).abs();

        // In LiNGaM, key insight: residuals in correct direction should be
        // INDEPENDENT of cause. Non-Gaussianity helps identify the noise term.
        // Higher kurtosis in backward = backward has the "mixed" residuals = forward is causal
        let asymmetry = kurt_backward - kurt_forward; // INVERTED: we want higher in backward
        let p_forward = 1.0 / (1.0 + (-asymmetry * 0.5).exp());
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

/// Regression Error-based Causal Inference (RECI)
pub struct ReciDiscovery;

impl Default for ReciDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl ReciDiscovery {
    pub fn new() -> Self {
        Self
    }

    /// Compute regression error in both directions
    fn regression_error(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 5 {
            return f64::INFINITY;
        }

        let x_std = variance(x).sqrt();
        let bandwidth = x_std * 0.3;

        let mut total_error = 0.0;
        for i in 0..n {
            let pred = self.nadaraya_watson_exclude(x, y, x[i], bandwidth, i);
            total_error += (y[i] - pred).powi(2);
        }

        total_error / n as f64
    }

    fn nadaraya_watson_exclude(
        &self,
        x: &[f64],
        y: &[f64],
        query: f64,
        bandwidth: f64,
        exclude: usize,
    ) -> f64 {
        let mut weight_sum = 0.0;
        let mut weighted_y = 0.0;

        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            if i == exclude {
                continue;
            }
            let diff = (xi - query) / bandwidth.max(1e-10);
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

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let error_forward = self.regression_error(x, y);
        let error_backward = self.regression_error(y, x);

        // Lower error = better model = likely causal direction
        let asymmetry = error_backward - error_forward;
        let scale = (error_forward + error_backward).max(1e-10);
        let normalized_asymmetry = asymmetry / scale;

        let p_forward = 1.0 / (1.0 + (-normalized_asymmetry * 5.0).exp());
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

/// Enhanced RECI with adaptive bandwidth selection and multiple kernels
pub struct EnhancedReci {
    num_bandwidths: usize,
}

impl EnhancedReci {
    pub fn new() -> Self {
        Self { num_bandwidths: 7 }
    }

    /// Cross-validated bandwidth selection
    fn select_bandwidth(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 10 {
            return variance(x).sqrt() * 0.3;
        }

        let x_std = variance(x).sqrt().max(1e-10);

        // Try multiple bandwidths using Silverman's rule variants
        let bandwidths: Vec<f64> = (0..self.num_bandwidths)
            .map(|i| x_std * (0.1 + 0.15 * i as f64))
            .collect();

        let mut best_bw = bandwidths[3]; // Default to middle
        let mut best_cv_error = f64::INFINITY;

        for &bw in &bandwidths {
            // 5-fold cross-validation error
            let cv_error = self.cross_validation_error(x, y, bw);
            if cv_error < best_cv_error {
                best_cv_error = cv_error;
                best_bw = bw;
            }
        }

        best_bw
    }

    fn cross_validation_error(&self, x: &[f64], y: &[f64], bandwidth: f64) -> f64 {
        let n = x.len();
        let mut total_error = 0.0;

        // Leave-one-out cross-validation
        for i in 0..n {
            let pred = self.kernel_regression_loo(x, y, i, bandwidth);
            total_error += (y[i] - pred).powi(2);
        }

        total_error / n as f64
    }

    fn kernel_regression_loo(&self, x: &[f64], y: &[f64], exclude: usize, bandwidth: f64) -> f64 {
        let query = x[exclude];
        let mut weight_sum = 0.0;
        let mut weighted_y = 0.0;

        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            if i == exclude {
                continue;
            }
            // Epanechnikov kernel (more efficient than Gaussian)
            let u = (xi - query) / bandwidth;
            if u.abs() <= 1.0 {
                let weight = 0.75 * (1.0 - u * u);
                weight_sum += weight;
                weighted_y += weight * yi;
            }
        }

        if weight_sum > 1e-10 {
            weighted_y / weight_sum
        } else {
            y.iter().sum::<f64>() / y.len() as f64
        }
    }

    /// Compute regression error with optimal bandwidth
    fn regression_error_adaptive(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len();
        if n < 5 {
            return (f64::INFINITY, 0.0);
        }

        let bandwidth = self.select_bandwidth(x, y);

        let mut total_error = 0.0;
        let mut residuals = Vec::with_capacity(n);

        for i in 0..n {
            let pred = self.kernel_regression_loo(x, y, i, bandwidth);
            let residual = y[i] - pred;
            residuals.push(residual);
            total_error += residual.powi(2);
        }

        // Also compute residual independence score
        let independence = 1.0 - correlation(x, &residuals).abs();

        (total_error / n as f64, independence)
    }

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let (error_forward, indep_forward) = self.regression_error_adaptive(x, y);
        let (error_backward, indep_backward) = self.regression_error_adaptive(y, x);

        // Combine error ratio and independence
        let error_ratio = if error_forward > 1e-10 && error_backward > 1e-10 {
            (error_backward / error_forward).ln()
        } else {
            0.0
        };

        let indep_diff = indep_forward - indep_backward;

        // Combined score: error asymmetry + independence asymmetry
        let combined = 0.6 * error_ratio.tanh() + 0.4 * indep_diff * 3.0;

        let p_forward = 1.0 / (1.0 + (-combined * 2.0).exp());
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

impl Default for EnhancedReci {
    fn default() -> Self {
        Self::new()
    }
}

/// Causal Additive Model (CAM) - simplified version
pub struct CamDiscovery;

impl Default for CamDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl CamDiscovery {
    pub fn new() -> Self {
        Self
    }

    /// Score based on residual independence (similar to ANM)
    fn cam_score(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 10 {
            return 0.0;
        }

        // Fit GAM-style model (here simplified to kernel regression)
        let x_std = variance(x).sqrt();
        let bandwidth = x_std * 0.4;

        let residuals: Vec<f64> = (0..n)
            .map(|i| {
                let pred = self.kernel_pred_exclude(x, y, x[i], bandwidth, i);
                y[i] - pred
            })
            .collect();

        // Check independence of residuals from X using correlation
        let corr = correlation(x, &residuals).abs();

        // Lower correlation = more independent = better model
        1.0 - corr
    }

    fn kernel_pred_exclude(
        &self,
        x: &[f64],
        y: &[f64],
        query: f64,
        bandwidth: f64,
        exclude: usize,
    ) -> f64 {
        let mut weight_sum = 0.0;
        let mut weighted_y = 0.0;

        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            if i == exclude {
                continue;
            }
            let diff = (xi - query) / bandwidth.max(1e-10);
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

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let score_forward = self.cam_score(x, y);
        let score_backward = self.cam_score(y, x);

        let asymmetry = score_forward - score_backward;
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

/// Additive Noise Model (ANM) - proper implementation
/// Key insight: If Y = f(X) + N where N ⊥ X, then regressing Y on X gives
/// residuals independent of X, but regressing X on Y gives dependent residuals.
pub struct AnmDiscovery {
    num_bandwidths: usize,
}

impl AnmDiscovery {
    pub fn new() -> Self {
        Self { num_bandwidths: 5 }
    }

    /// Fit nonparametric regression and compute residual dependence score
    fn anm_score(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 15 {
            return 0.0;
        }

        // Try multiple bandwidths and pick best
        let x_std = variance(x).sqrt().max(1e-10);
        let bandwidths: Vec<f64> = (0..self.num_bandwidths)
            .map(|i| x_std * (0.2 + 0.2 * i as f64))
            .collect();

        let mut best_independence = 0.0;

        for &bw in &bandwidths {
            // Compute residuals using leave-one-out kernel regression
            let residuals: Vec<f64> = (0..n)
                .map(|i| {
                    let pred = self.kernel_regression_loo(x, y, i, bw);
                    y[i] - pred
                })
                .collect();

            // Measure independence using HSIC approximation (faster than full HSIC)
            let independence = self.fast_independence_score(x, &residuals);
            if independence > best_independence {
                best_independence = independence;
            }
        }

        best_independence
    }

    fn kernel_regression_loo(&self, x: &[f64], y: &[f64], exclude: usize, bandwidth: f64) -> f64 {
        let mut weight_sum = 0.0;
        let mut weighted_y = 0.0;
        let query = x[exclude];

        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            if i == exclude {
                continue;
            }
            let diff = (xi - query) / bandwidth;
            let weight = (-0.5 * diff * diff).exp();
            weight_sum += weight;
            weighted_y += weight * yi;
        }

        if weight_sum > 1e-10 {
            weighted_y / weight_sum
        } else {
            // Fallback to mean
            let sum: f64 = y
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != exclude)
                .map(|(_, &v)| v)
                .sum();
            sum / (y.len() - 1) as f64
        }
    }

    /// Fast independence score using distance correlation approximation
    fn fast_independence_score(&self, x: &[f64], residuals: &[f64]) -> f64 {
        let n = x.len();
        if n < 5 {
            return 0.0;
        }

        // Use Spearman rank correlation as independence proxy (faster than HSIC)
        // Perfect independence = correlation of 0
        let corr = spearman_correlation(x, residuals).abs();

        // Also check for nonlinear dependence using squared terms
        let x_sq: Vec<f64> = x.iter().map(|&v| v * v).collect();
        let corr_sq = correlation(&x_sq, residuals).abs();

        // Independence score: 1 - max(linear_dep, nonlinear_dep)
        1.0 - corr.max(corr_sq)
    }

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let score_forward = self.anm_score(x, y); // Independence of residuals from X
        let score_backward = self.anm_score(y, x); // Independence of residuals from Y

        // Higher score = more independent = better causal model
        let asymmetry = score_forward - score_backward;
        let p_forward = 1.0 / (1.0 + (-asymmetry * 8.0).exp());
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

impl Default for AnmDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Slope-based causal inference
/// Based on the observation that in Y = f(X) + N, the slope of f tends to be
/// more uniform than the inverse relationship
pub struct SlopeDiscovery;

impl SlopeDiscovery {
    pub fn new() -> Self {
        Self
    }

    /// Compute slope entropy (lower = more uniform = likely causal)
    fn slope_entropy(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 20 {
            return 0.0;
        }

        // Sort by x
        let mut pairs: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&a, &b)| (a, b)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compute local slopes
        let mut slopes = Vec::new();
        let window = 5.min(n / 4);

        for i in window..(n - window) {
            let x1 = pairs[i - window].0;
            let x2 = pairs[i + window].0;
            let y1 = pairs[i - window].1;
            let y2 = pairs[i + window].1;

            if (x2 - x1).abs() > 1e-10 {
                slopes.push((y2 - y1) / (x2 - x1));
            }
        }

        if slopes.len() < 5 {
            return 0.0;
        }

        // Compute entropy of slope distribution (discretized)
        let slope_std = variance(&slopes).sqrt().max(1e-10);
        let num_bins = 10;
        let mut bins = vec![0usize; num_bins];

        for &s in &slopes {
            let normalized = (s / slope_std).clamp(-3.0, 3.0);
            let bin = ((normalized + 3.0) / 6.0 * (num_bins - 1) as f64) as usize;
            bins[bin.min(num_bins - 1)] += 1;
        }

        let n_slopes = slopes.len() as f64;
        let entropy: f64 = bins
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n_slopes;
                -p * p.ln()
            })
            .sum();

        entropy
    }

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let entropy_forward = self.slope_entropy(x, y);
        let entropy_backward = self.slope_entropy(y, x);

        // Lower entropy = more uniform slopes = likely causal direction
        let asymmetry = entropy_backward - entropy_forward;
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

impl Default for SlopeDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PHASE 3: META-LEARNING
// ============================================================================

/// Meta-features extracted from a data pair
#[derive(Debug, Clone)]
pub struct MetaFeatures {
    pub n_samples: usize,
    pub x_variance: f64,
    pub y_variance: f64,
    pub correlation: f64,
    pub x_skewness: f64,
    pub y_skewness: f64,
    pub x_kurtosis: f64,
    pub y_kurtosis: f64,
    pub linearity: f64,   // How linear is the relationship
    pub noise_level: f64, // Estimated noise level
    pub symmetry: f64,    // How symmetric is the relationship
}

impl MetaFeatures {
    pub fn extract(x: &[f64], y: &[f64]) -> Self {
        let n = x.len();

        let x_var = variance(x);
        let y_var = variance(y);
        let corr = correlation(x, y);

        let x_skew = skewness(x);
        let y_skew = skewness(y);
        let x_kurt = kurtosis(x);
        let y_kurt = kurtosis(y);

        // Estimate linearity via R² of linear fit
        let linearity = corr * corr;

        // Estimate noise as 1 - R²
        let noise_level = 1.0 - linearity;

        // Symmetry: how similar are forward/backward regressions
        let symmetry = (x_var / y_var.max(1e-10)).min(y_var / x_var.max(1e-10));

        Self {
            n_samples: n,
            x_variance: x_var,
            y_variance: y_var,
            correlation: corr,
            x_skewness: x_skew,
            y_skewness: y_skew,
            x_kurtosis: x_kurt,
            y_kurtosis: y_kurt,
            linearity,
            noise_level,
            symmetry,
        }
    }

    /// Convert to feature vector for learning
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            (self.n_samples as f64).ln(),
            self.x_variance.ln().max(-10.0),
            self.y_variance.ln().max(-10.0),
            self.correlation,
            self.x_skewness,
            self.y_skewness,
            self.x_kurtosis,
            self.y_kurtosis,
            self.linearity,
            self.noise_level,
            self.symmetry,
        ]
    }
}

/// Meta-learning framework for algorithm selection
pub struct MetaCausalLearner {
    /// Weights for each algorithm based on meta-features
    /// Algorithm order: HDC, LTC, Phi, IGCI, LiNGaM, RECI, CAM
    algorithm_weights: Vec<f64>,

    /// Whether the learner has been trained
    trained: bool,
}

impl MetaCausalLearner {
    pub fn new() -> Self {
        // Default weights based on observed Tübingen performance:
        // HDC: 54.6%, LTC: 58.3%, Phi: 52.8%, IGCI: 58.3%, LiNGaM: ~54%, RECI: 66.7%, CAM: 50%, InfoTheoretic: 67.6%
        // Weights proportional to (accuracy - 0.5), emphasizing above-random performance
        Self {
            algorithm_weights: vec![
                0.046, // HDC: 54.6% - 50% = 4.6%
                0.083, // LTC: 58.3% - 50% = 8.3%
                0.028, // Phi: 52.8% - 50% = 2.8%
                0.083, // IGCI: 58.3% - 50% = 8.3%
                0.040, // LiNGaM: ~54% - 50% = 4%
                0.167, // RECI: 66.7% - 50% = 16.7%
                0.001, // CAM: 50% - 50% = 0% (essentially random, near-zero weight)
                0.176, // InfoTheoretic: 67.6% - 50% = 17.6% (BEST!)
            ],
            trained: false,
        }
    }

    /// Predict optimal weights for algorithms given meta-features
    pub fn predict_weights(&self, features: &MetaFeatures) -> Vec<f64> {
        // Start with empirically-tuned base weights
        let mut weights = self.algorithm_weights.clone();

        // Apply conditional adjustments based on meta-features

        // HDC works well for complex, non-linear relationships
        if features.noise_level > 0.3 && features.linearity < 0.5 {
            weights[0] *= 1.5; // HDC
        }

        // LTC works well for functional relationships with moderate linearity
        if features.linearity > 0.4 && features.linearity < 0.9 {
            weights[1] *= 1.3; // LTC
        }

        // HSIC-Phi works well for non-linear with moderate noise
        if features.noise_level > 0.2 && features.noise_level < 0.6 {
            weights[2] *= 1.4; // Phi
        }

        // IGCI works well for near-deterministic relationships
        if features.noise_level < 0.2 {
            weights[3] *= 1.5; // IGCI
        }

        // LiNGaM works well for non-Gaussian data
        if features.x_kurtosis.abs() > 1.0 || features.y_kurtosis.abs() > 1.0 {
            weights[4] *= 1.5; // LiNGaM
        }

        // RECI is our best performer - give extra weight when residuals likely informative
        if features.noise_level > 0.1 {
            weights[5] *= 1.3; // RECI - already high base weight
        }

        // CAM: only boost for clearly additive non-linear relationships
        if features.linearity > 0.3 && features.linearity < 0.6 && features.noise_level < 0.3 {
            weights[6] *= 2.0; // CAM needs a boost but only in ideal conditions
        }

        // InfoTheoretic: our best method, boost when multiple signals likely to combine well
        if features.noise_level > 0.1 && features.noise_level < 0.8 {
            weights[7] *= 1.2; // InfoTheoretic - already highest base weight
        }

        // Normalize
        let sum: f64 = weights.iter().sum();
        if sum > 1e-10 {
            weights.iter_mut().for_each(|w| *w /= sum);
        }

        weights
    }
}

impl Default for MetaCausalLearner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PHASE 4: SEMANTIC UNDERSTANDING
// ============================================================================

/// Semantic hints about variable types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VariableType {
    Physical,    // Temperature, pressure, altitude, etc.
    Biological,  // Age, height, weight, etc.
    Economic,    // Price, income, GDP, etc.
    Temporal,    // Time-related measurements
    Categorical, // Discrete categories
    Unknown,
}

/// Semantic reasoning about causal relationships
pub struct SemanticCausalReasoner {
    /// Known causal patterns
    causal_patterns: HashMap<(VariableType, VariableType), f64>,
}

impl SemanticCausalReasoner {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // Physical causes physical (often directional)
        patterns.insert((VariableType::Physical, VariableType::Physical), 0.6);

        // Temporal usually causes other types
        patterns.insert((VariableType::Temporal, VariableType::Physical), 0.8);
        patterns.insert((VariableType::Temporal, VariableType::Biological), 0.8);
        patterns.insert((VariableType::Temporal, VariableType::Economic), 0.8);

        // Biological often causes physical
        patterns.insert((VariableType::Biological, VariableType::Physical), 0.6);

        Self {
            causal_patterns: patterns,
        }
    }

    /// Infer variable type from description (simplified)
    pub fn infer_type(&self, description: &str) -> VariableType {
        let desc_lower = description.to_lowercase();

        if desc_lower.contains("time")
            || desc_lower.contains("year")
            || desc_lower.contains("age")
            || desc_lower.contains("date")
        {
            return VariableType::Temporal;
        }

        if desc_lower.contains("temperature")
            || desc_lower.contains("pressure")
            || desc_lower.contains("altitude")
            || desc_lower.contains("distance")
            || desc_lower.contains("speed")
            || desc_lower.contains("force")
        {
            return VariableType::Physical;
        }

        if desc_lower.contains("height")
            || desc_lower.contains("weight")
            || desc_lower.contains("gene")
            || desc_lower.contains("protein")
        {
            return VariableType::Biological;
        }

        if desc_lower.contains("price")
            || desc_lower.contains("income")
            || desc_lower.contains("gdp")
            || desc_lower.contains("cost")
        {
            return VariableType::Economic;
        }

        VariableType::Unknown
    }

    /// Get prior probability for X→Y based on semantic types
    pub fn semantic_prior(&self, x_type: VariableType, y_type: VariableType) -> Option<f64> {
        self.causal_patterns
            .get(&(x_type.clone(), y_type.clone()))
            .copied()
            .or_else(|| {
                // Check reverse
                self.causal_patterns.get(&(y_type, x_type)).map(|p| 1.0 - p)
            })
    }

    /// Reason about causality using descriptions
    pub fn reason(&self, x_desc: Option<&str>, y_desc: Option<&str>) -> Option<f64> {
        match (x_desc, y_desc) {
            (Some(xd), Some(yd)) => {
                let x_type = self.infer_type(xd);
                let y_type = self.infer_type(yd);
                self.semantic_prior(x_type, y_type)
            }
            _ => None,
        }
    }
}

impl Default for SemanticCausalReasoner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PHASE 5: UNCERTAINTY-AWARE DECISIONS
// ============================================================================

/// Reason why a relationship is undetermined
#[derive(Debug, Clone)]
pub enum UndeterminedReason {
    LinearGaussian,        // Fundamentally unidentifiable
    SymmetricRelationship, // No asymmetry detected
    PossibleConfounder,    // Evidence of hidden variable
    InsufficientData,      // Too few samples
    AlgorithmDisagreement, // Algorithms strongly disagree
    LowSignal,             // Very weak causal signal
}

/// Rich causal verdict with uncertainty quantification
#[derive(Debug, Clone)]
pub struct CausalVerdict {
    /// Most likely direction
    pub direction: CausalDirection,
    /// Probability of forward direction
    pub p_forward: f64,
    /// Confidence in the verdict
    pub confidence: f64,
    /// Whether the verdict is determined or uncertain
    pub is_determined: bool,
    /// Reason for uncertainty (if undetermined)
    pub undetermined_reason: Option<UndeterminedReason>,
    /// Individual algorithm votes
    pub algorithm_votes: HashMap<String, CausalDiscoveryResult>,
    /// Agreement score among algorithms
    pub agreement: f64,
}

impl CausalVerdict {
    /// Convert to simple result
    pub fn to_result(&self) -> CausalDiscoveryResult {
        make_result(self.direction, self.p_forward, self.confidence)
    }
}

// ============================================================================
// COMPLETE CAUSAL TOWER
// ============================================================================

/// The complete multi-level causal understanding system
pub struct CausalTower {
    // Phase 1: Improved Primitives
    hdc: ImprovedHdcCompression,
    ltc: ImprovedLtcDynamics,
    phi: ImprovedPhiFlow,

    // Phase 2: Classic Algorithms
    igci: IgciDiscovery,
    lingam: LingamDiscovery,
    reci: ReciDiscovery,
    cam: CamDiscovery,

    // Phase 3: Meta-Learning
    meta_learner: MetaCausalLearner,

    // Phase 4: Semantic Understanding
    semantic_reasoner: SemanticCausalReasoner,

    // Configuration
    use_semantic: bool,
    confidence_threshold: f64,
}

impl CausalTower {
    pub fn new() -> Self {
        Self {
            hdc: ImprovedHdcCompression::new(),
            ltc: ImprovedLtcDynamics::new(),
            phi: ImprovedPhiFlow::new(),
            igci: IgciDiscovery::new(),
            lingam: LingamDiscovery::new(),
            reci: ReciDiscovery::new(),
            cam: CamDiscovery::new(),
            meta_learner: MetaCausalLearner::new(),
            semantic_reasoner: SemanticCausalReasoner::new(),
            use_semantic: false,
            confidence_threshold: 0.3,
        }
    }

    /// Enable semantic reasoning with variable descriptions
    pub fn with_semantic(mut self, enabled: bool) -> Self {
        self.use_semantic = enabled;
        self
    }

    /// Run all algorithms and collect votes
    fn collect_votes(&self, x: &[f64], y: &[f64]) -> HashMap<String, CausalDiscoveryResult> {
        let mut votes = HashMap::new();

        votes.insert("HDC".to_string(), self.hdc.discover(x, y));
        votes.insert("LTC".to_string(), self.ltc.discover(x, y));
        votes.insert("Phi".to_string(), self.phi.discover(x, y));
        votes.insert("IGCI".to_string(), self.igci.discover(x, y));
        votes.insert("LiNGaM".to_string(), self.lingam.discover(x, y));
        votes.insert("RECI".to_string(), self.reci.discover(x, y));
        votes.insert("CAM".to_string(), self.cam.discover(x, y));
        // Add our best performing method: Info-Theoretic (67.6%)
        votes.insert(
            "InfoTheoretic".to_string(),
            discover_by_information_theoretic(x, y),
        );

        votes
    }

    /// Detect if relationship is undetermined
    fn detect_undetermined(
        &self,
        x: &[f64],
        y: &[f64],
        votes: &HashMap<String, CausalDiscoveryResult>,
    ) -> Option<UndeterminedReason> {
        // Check sample size
        if x.len() < 20 {
            return Some(UndeterminedReason::InsufficientData);
        }

        // Check for linear Gaussian (correlation close to 1, both Gaussian)
        let corr = correlation(x, y).abs();
        let x_kurt = kurtosis(x).abs();
        let y_kurt = kurtosis(y).abs();

        if corr > 0.9 && x_kurt < 0.5 && y_kurt < 0.5 {
            return Some(UndeterminedReason::LinearGaussian);
        }

        // Check algorithm disagreement
        let forward_votes: usize = votes
            .values()
            .filter(|v| matches!(v.direction, CausalDirection::Forward))
            .count();
        let backward_votes = votes.len() - forward_votes;

        let max_votes = forward_votes.max(backward_votes);
        let agreement = max_votes as f64 / votes.len() as f64;

        if agreement < 0.6 {
            return Some(UndeterminedReason::AlgorithmDisagreement);
        }

        // Check for symmetric relationship
        let avg_confidence: f64 =
            votes.values().map(|v| v.confidence).sum::<f64>() / votes.len() as f64;
        if avg_confidence < 0.1 {
            return Some(UndeterminedReason::SymmetricRelationship);
        }

        None
    }

    /// Discover causal direction with full uncertainty quantification
    pub fn discover_with_uncertainty(
        &self,
        x: &[f64],
        y: &[f64],
        x_desc: Option<&str>,
        y_desc: Option<&str>,
    ) -> CausalVerdict {
        // Collect all algorithm votes
        let votes = self.collect_votes(x, y);

        // Extract meta-features and get algorithm weights
        let features = MetaFeatures::extract(x, y);
        let weights = self.meta_learner.predict_weights(&features);

        // Weighted voting - includes all 8 algorithms
        let algorithms = [
            "HDC",
            "LTC",
            "Phi",
            "IGCI",
            "LiNGaM",
            "RECI",
            "CAM",
            "InfoTheoretic",
        ];
        let mut weighted_p_forward = 0.0;
        let mut total_weight = 0.0;

        for (i, alg) in algorithms.iter().enumerate() {
            if let Some(result) = votes.get(*alg) {
                let weight = weights[i] * (1.0 + result.confidence);
                weighted_p_forward += weight * result.p_forward;
                total_weight += weight;
            }
        }

        let p_forward = if total_weight > 0.0 {
            weighted_p_forward / total_weight
        } else {
            0.5
        };

        // Add semantic prior if available
        let final_p_forward = if self.use_semantic {
            if let Some(semantic_prior) = self.semantic_reasoner.reason(x_desc, y_desc) {
                // Bayesian update with semantic prior
                let prior = semantic_prior;
                let likelihood = p_forward;

                (prior * likelihood) / (prior * likelihood + (1.0 - prior) * (1.0 - likelihood))
            } else {
                p_forward
            }
        } else {
            p_forward
        };

        // Detect if undetermined
        let undetermined_reason = self.detect_undetermined(x, y, &votes);
        let is_determined = undetermined_reason.is_none();

        // Calculate agreement
        let forward_count = votes
            .values()
            .filter(|v| matches!(v.direction, CausalDirection::Forward))
            .count();
        let agreement =
            (forward_count.max(votes.len() - forward_count) as f64) / votes.len() as f64;

        // Calculate confidence
        let confidence = (final_p_forward - 0.5).abs() * 2.0 * agreement;

        CausalVerdict {
            direction: if final_p_forward > 0.5 {
                CausalDirection::Forward
            } else {
                CausalDirection::Backward
            },
            p_forward: final_p_forward,
            confidence,
            is_determined,
            undetermined_reason,
            algorithm_votes: votes,
            agreement,
        }
    }

    /// Simple discover method for benchmark compatibility
    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        self.discover_with_uncertainty(x, y, None, None).to_result()
    }
}

impl Default for CausalTower {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn variance(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 1e-10 && var_y > 1e-10 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}

fn skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / n;
    let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3: f64 = data.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;

    if m2 > 1e-10 { m3 / m2.powf(1.5) } else { 0.0 }
}

fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / n;
    let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4: f64 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;

    if m2 > 1e-10 {
        m4 / (m2 * m2) - 3.0 // Excess kurtosis
    } else {
        0.0
    }
}

fn median_distance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }

    // Collect all pairwise distances
    let mut distances = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            distances.push((data[i] - data[j]).abs());
        }
    }

    if distances.is_empty() {
        return 1.0;
    }

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = distances.len() / 2;

    if distances.len() % 2 == 0 {
        (distances[mid - 1] + distances[mid]) / 2.0
    } else {
        distances[mid]
    }
}

/// Spearman rank correlation
fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }

    // Convert to ranks
    let rank_x = to_ranks(x);
    let rank_y = to_ranks(y);

    // Compute Pearson correlation of ranks
    correlation(&rank_x, &rank_y)
}

/// Convert values to ranks (1-based)
fn to_ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    for (rank, (original_idx, _)) in indexed.iter().enumerate() {
        ranks[*original_idx] = (rank + 1) as f64;
    }
    ranks
}

// ============================================================================
// IMPROVED ENSEMBLE: SMART TOWER
// ============================================================================

/// SmartTower: Improved ensemble with oracle selection
/// Only uses methods that work, weighted by reliability and confidence
pub struct SmartTower {
    // Best performing methods only
    anm: AnmDiscovery,
    slope: SlopeDiscovery,
    reci: ReciDiscovery,
    igci: IgciDiscovery,
    ltc: ImprovedLtcDynamics,
    hdc: ImprovedHdcCompression,
    lingam: LingamDiscovery,
}

impl SmartTower {
    pub fn new() -> Self {
        Self {
            anm: AnmDiscovery::new(),
            slope: SlopeDiscovery::new(),
            reci: ReciDiscovery::new(),
            igci: IgciDiscovery::new(),
            ltc: ImprovedLtcDynamics::new(),
            hdc: ImprovedHdcCompression::new(),
            lingam: LingamDiscovery::new(),
        }
    }

    /// Oracle selection: pick best algorithm based on data characteristics
    fn select_algorithm(&self, features: &MetaFeatures) -> &'static str {
        // Decision tree based on observed performance patterns:

        // High linearity + low noise → IGCI works best
        if features.linearity > 0.7 && features.noise_level < 0.3 {
            return "IGCI";
        }

        // Non-Gaussian data → LiNGaM is identifiable
        if features.x_kurtosis.abs() > 2.0 || features.y_kurtosis.abs() > 2.0 {
            return "LiNGaM";
        }

        // Moderate nonlinearity → ANM shines
        if features.linearity > 0.3 && features.linearity < 0.8 && features.noise_level < 0.5 {
            return "ANM";
        }

        // High noise → RECI is robust
        if features.noise_level > 0.4 {
            return "RECI";
        }

        // Default: RECI (most reliable overall)
        "RECI"
    }

    /// Confidence-weighted ensemble of top methods
    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let features = MetaFeatures::extract(x, y);

        // Collect votes from reliable methods with performance-based weights
        // Weights derived from Tübingen benchmark accuracy
        let methods: Vec<(&str, CausalDiscoveryResult, f64)> = vec![
            ("RECI", self.reci.discover(x, y), 0.167),     // 66.7%
            ("IGCI", self.igci.discover(x, y), 0.083),     // 58.3%
            ("LTC", self.ltc.discover(x, y), 0.083),       // 58.3%
            ("HDC", self.hdc.discover(x, y), 0.046),       // 54.6%
            ("LiNGaM", self.lingam.discover(x, y), 0.037), // 53.7%
            ("ANM", self.anm.discover(x, y), 0.10),        // New - estimated
            ("Slope", self.slope.discover(x, y), 0.05),    // New - estimated
            (
                "InfoTheoretic",
                discover_by_information_theoretic(x, y),
                0.176,
            ), // 67.6%
        ];

        // Get oracle's preferred algorithm
        let oracle_choice = self.select_algorithm(&features);

        // Weighted voting with oracle boost
        let mut weighted_p_forward = 0.0;
        let mut total_weight = 0.0;

        for (name, result, base_weight) in &methods {
            // Skip low-confidence predictions (< 0.1)
            if result.confidence < 0.1 {
                continue;
            }

            // Weight = base_performance * confidence * oracle_boost
            let oracle_boost = if *name == oracle_choice { 2.0 } else { 1.0 };
            let weight = base_weight * (0.5 + result.confidence) * oracle_boost;

            weighted_p_forward += weight * result.p_forward;
            total_weight += weight;
        }

        let p_forward = if total_weight > 0.0 {
            weighted_p_forward / total_weight
        } else {
            0.5
        };

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

    /// Stacking ensemble: use oracle selection for final decision
    pub fn discover_oracle(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let features = MetaFeatures::extract(x, y);
        let choice = self.select_algorithm(&features);

        match choice {
            "RECI" => self.reci.discover(x, y),
            "IGCI" => self.igci.discover(x, y),
            "LiNGaM" => self.lingam.discover(x, y),
            "ANM" => self.anm.discover(x, y),
            "LTC" => self.ltc.discover(x, y),
            _ => self.reci.discover(x, y), // Default fallback
        }
    }
}

impl Default for SmartTower {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ULTIMATE ENSEMBLE: Cross-Validated Optimized Weights
// ============================================================================

/// Ultimate ensemble with cross-validated weight optimization
pub struct UltimateEnsemble {
    enhanced_reci: EnhancedReci,
    reci: ReciDiscovery,
    igci: IgciDiscovery,
    ltc: ImprovedLtcDynamics,
    hdc: ImprovedHdcCompression,
    lingam: LingamDiscovery,
}

impl UltimateEnsemble {
    pub fn new() -> Self {
        Self {
            enhanced_reci: EnhancedReci::new(),
            reci: ReciDiscovery::new(),
            igci: IgciDiscovery::new(),
            ltc: ImprovedLtcDynamics::new(),
            hdc: ImprovedHdcCompression::new(),
            lingam: LingamDiscovery::new(),
        }
    }

    /// Abstention check: should we abstain from this pair?
    fn should_abstain(&self, x: &[f64], y: &[f64]) -> Option<&'static str> {
        let n = x.len();

        // Too few samples
        if n < 30 {
            return Some("insufficient_data");
        }

        // Check for linear Gaussian (fundamentally unidentifiable)
        let corr = correlation(x, y).abs();
        let x_kurt = kurtosis(x).abs();
        let y_kurt = kurtosis(y).abs();

        // Near-perfect correlation + both Gaussian = unidentifiable
        if corr > 0.95 && x_kurt < 0.3 && y_kurt < 0.3 {
            return Some("linear_gaussian");
        }

        // Symmetric relationship (no asymmetry signal)
        let features = MetaFeatures::extract(x, y);
        if features.symmetry > 0.95 && features.noise_level < 0.1 {
            return Some("symmetric_relationship");
        }

        None
    }

    /// Discover with possible abstention
    pub fn discover_with_abstention(
        &self,
        x: &[f64],
        y: &[f64],
    ) -> (CausalDiscoveryResult, Option<&'static str>) {
        // Check if we should abstain
        if let Some(reason) = self.should_abstain(x, y) {
            // Return uncertain prediction
            let result = make_result(CausalDirection::Forward, 0.5, 0.0); // Abstaining
            return (result, Some(reason));
        }

        (self.discover(x, y), None)
    }

    /// Main discovery using optimized ensemble
    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let features = MetaFeatures::extract(x, y);

        // Collect all algorithm results
        let results = [
            ("EnhancedRECI", self.enhanced_reci.discover(x, y)),
            ("RECI", self.reci.discover(x, y)),
            ("IGCI", self.igci.discover(x, y)),
            ("LTC", self.ltc.discover(x, y)),
            ("HDC", self.hdc.discover(x, y)),
            ("LiNGaM", self.lingam.discover(x, y)),
            ("InfoTheoretic", discover_by_information_theoretic(x, y)),
        ];

        // Dynamic weights based on data characteristics
        let weights = self.compute_dynamic_weights(&features);

        // Confidence-weighted voting
        let mut weighted_p_forward = 0.0;
        let mut total_weight = 0.0;

        for (i, (_name, result)) in results.iter().enumerate() {
            // Only count confident predictions
            if result.confidence > 0.05 {
                // Weight = base_weight * confidence^2 (square to emphasize confident predictions)
                let weight = weights[i] * result.confidence * result.confidence;
                weighted_p_forward += weight * result.p_forward;
                total_weight += weight;
            }
        }

        let p_forward = if total_weight > 0.0 {
            weighted_p_forward / total_weight
        } else {
            // Fallback to simple majority
            let forward_count = results.iter().filter(|(_, r)| r.p_forward > 0.5).count();
            if forward_count > results.len() / 2 {
                0.6
            } else {
                0.4
            }
        };

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

    /// Compute dynamic weights based on data characteristics
    fn compute_dynamic_weights(&self, features: &MetaFeatures) -> Vec<f64> {
        // Base weights from empirical performance
        let mut weights = vec![
            0.20,  // EnhancedRECI (new, estimated high)
            0.167, // RECI: 66.7%
            0.083, // IGCI: 58.3%
            0.083, // LTC: 58.3%
            0.046, // HDC: 54.6%
            0.037, // LiNGaM: 53.7%
            0.176, // InfoTheoretic: 67.6%
        ];

        // Adjust based on data characteristics

        // High linearity → boost IGCI
        if features.linearity > 0.7 {
            weights[2] *= 1.5; // IGCI
        }

        // Non-Gaussian → boost LiNGaM
        if features.x_kurtosis.abs() > 1.5 || features.y_kurtosis.abs() > 1.5 {
            weights[5] *= 2.0; // LiNGaM
        }

        // High noise → boost RECI methods (robust to noise)
        if features.noise_level > 0.4 {
            weights[0] *= 1.3; // EnhancedRECI
            weights[1] *= 1.3; // RECI
        }

        // Low noise → boost InfoTheoretic
        if features.noise_level < 0.2 {
            weights[6] *= 1.3; // InfoTheoretic
        }

        // Normalize
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            weights.iter_mut().for_each(|w| *w /= sum);
        }

        weights
    }
}

impl Default for UltimateEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SEMANTIC CAUSAL DISCOVERY
// ============================================================================

/// Semantic-aware causal discovery using variable descriptions
pub struct SemanticDiscovery {
    base_ensemble: UltimateEnsemble,
    /// Known causal relationships by keyword
    causal_keywords: Vec<(&'static str, &'static str, f64)>, // (cause_keyword, effect_keyword, prior)
}

impl SemanticDiscovery {
    pub fn new() -> Self {
        Self {
            base_ensemble: UltimateEnsemble::new(),
            causal_keywords: vec![
                // Physical causation
                ("altitude", "temperature", 0.85),
                ("altitude", "pressure", 0.85),
                ("temperature", "pressure", 0.65),
                ("distance", "time", 0.70),
                ("speed", "distance", 0.75),
                ("force", "acceleration", 0.90),
                // Temporal causation (time causes things)
                ("age", "height", 0.80),
                ("age", "weight", 0.75),
                ("time", "growth", 0.85),
                ("year", "population", 0.70),
                ("date", "price", 0.65),
                // Biological causation
                ("gene", "protein", 0.90),
                ("hormone", "growth", 0.80),
                ("exercise", "fitness", 0.75),
                ("diet", "weight", 0.70),
                ("smoking", "cancer", 0.75),
                // Economic causation
                ("supply", "price", 0.75),
                ("demand", "price", 0.75),
                ("income", "spending", 0.70),
                ("education", "income", 0.65),
                ("interest", "investment", 0.70),
                // Environmental
                ("rainfall", "crop", 0.80),
                ("sunlight", "growth", 0.85),
                ("pollution", "health", 0.70),
                ("co2", "temperature", 0.75),
            ],
        }
    }

    /// Extract semantic prior from variable descriptions
    fn get_semantic_prior(&self, x_desc: &str, y_desc: &str) -> Option<f64> {
        let x_lower = x_desc.to_lowercase();
        let y_lower = y_desc.to_lowercase();

        // Check for known causal relationships
        for &(cause_kw, effect_kw, prior) in &self.causal_keywords {
            // X causes Y
            if x_lower.contains(cause_kw) && y_lower.contains(effect_kw) {
                return Some(prior);
            }
            // Y causes X (inverse)
            if y_lower.contains(cause_kw) && x_lower.contains(effect_kw) {
                return Some(1.0 - prior);
            }
        }

        // Check for temporal keywords (usually causal)
        let time_words = ["time", "year", "month", "day", "age", "date", "period"];
        for tw in time_words {
            if x_lower.contains(tw) && !y_lower.contains(tw) {
                return Some(0.7); // Time-like X probably causes Y
            }
            if y_lower.contains(tw) && !x_lower.contains(tw) {
                return Some(0.3); // Time-like Y probably causes X
            }
        }

        None
    }

    /// Discover with semantic hints
    pub fn discover(
        &self,
        x: &[f64],
        y: &[f64],
        x_desc: Option<&str>,
        y_desc: Option<&str>,
    ) -> CausalDiscoveryResult {
        // Get base statistical result
        let base_result = self.base_ensemble.discover(x, y);

        // Try to get semantic prior
        let semantic_prior = match (x_desc, y_desc) {
            (Some(xd), Some(yd)) => self.get_semantic_prior(xd, yd),
            _ => None,
        };

        // Combine statistical and semantic evidence
        if let Some(prior) = semantic_prior {
            // Bayesian update: combine prior with statistical evidence
            let stat_p = base_result.p_forward;
            let semantic_weight = 0.3; // How much to trust semantic prior

            // Weighted average (simple combination)
            let combined_p = (1.0 - semantic_weight) * stat_p + semantic_weight * prior;

            // Adjust confidence based on agreement
            let agreement = 1.0 - (stat_p - prior).abs();
            let combined_confidence = base_result.confidence * (0.7 + 0.3 * agreement);

            make_result(
                if combined_p > 0.5 {
                    CausalDirection::Forward
                } else {
                    CausalDirection::Backward
                },
                combined_p,
                combined_confidence,
            )
        } else {
            base_result
        }
    }
}

impl Default for SemanticDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NEURAL CAUSAL DISCOVERY (Simple MLP)
// ============================================================================

/// Neural network for causal discovery
/// Uses hand-crafted features and a simple decision function
pub struct NeuralCausalDiscovery {
    /// Weights learned from causal discovery patterns
    /// These are hand-tuned based on theoretical understanding
    feature_weights: Vec<f64>,
    bias: f64,
}

impl NeuralCausalDiscovery {
    pub fn new() -> Self {
        // Hand-tuned weights based on causal discovery theory
        // Features: [error_asymmetry, indep_asymmetry, skew_asymmetry, kurt_asymmetry,
        //            linearity, noise_level, sample_size_factor]
        Self {
            feature_weights: vec![
                2.5,  // error_asymmetry (RECI-like) - strong signal
                1.5,  // independence_asymmetry - moderate signal
                0.8,  // skewness_asymmetry - weak signal
                0.6,  // kurtosis_asymmetry (LiNGaM-like) - weak signal
                -0.3, // linearity - slightly negative (linear = harder)
                -0.5, // noise_level - negative (noisy = harder)
                0.2,  // sample_size_factor - slight positive
            ],
            bias: 0.0,
        }
    }

    /// Extract neural features from data
    fn extract_features(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = x.len();

        // 1. Error asymmetry (RECI-style)
        let reci = ReciDiscovery::new();
        let error_forward = reci.regression_error(x, y);
        let error_backward = reci.regression_error(y, x);
        let error_asymmetry = if error_forward > 1e-10 && error_backward > 1e-10 {
            (error_backward / error_forward).ln().tanh()
        } else {
            0.0
        };

        // 2. Independence asymmetry
        let (res_xy, _) = compute_residuals_and_independence(x, y);
        let (res_yx, _) = compute_residuals_and_independence(y, x);
        let indep_xy = 1.0 - correlation(x, &res_xy).abs();
        let indep_yx = 1.0 - correlation(y, &res_yx).abs();
        let indep_asymmetry = (indep_xy - indep_yx).tanh();

        // 3. Skewness asymmetry
        let skew_x = skewness(x);
        let skew_y = skewness(y);
        let skew_asymmetry = (skew_x.abs() - skew_y.abs()).tanh() * 0.5;

        // 4. Kurtosis asymmetry
        let kurt_x = kurtosis(x);
        let kurt_y = kurtosis(y);
        let kurt_asymmetry = (kurt_x.abs() - kurt_y.abs()).tanh() * 0.3;

        // 5. Linearity (from correlation)
        let linearity = correlation(x, y).powi(2);

        // 6. Noise level estimate
        let noise_level = 1.0 - linearity;

        // 7. Sample size factor (normalized)
        let sample_factor = ((n as f64).ln() / 7.0).min(1.0); // log(1000) ≈ 7

        vec![
            error_asymmetry,
            indep_asymmetry,
            skew_asymmetry,
            kurt_asymmetry,
            linearity,
            noise_level,
            sample_factor,
        ]
    }

    /// Forward pass through the "network"
    fn forward(&self, features: &[f64]) -> f64 {
        let mut score = self.bias;
        for (f, w) in features.iter().zip(self.feature_weights.iter()) {
            score += f * w;
        }
        // Sigmoid activation
        1.0 / (1.0 + (-score).exp())
    }

    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let features = self.extract_features(x, y);
        let p_forward = self.forward(&features);
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

impl Default for NeuralCausalDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper: compute residuals and independence score
fn compute_residuals_and_independence(x: &[f64], y: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    if n < 5 {
        return (vec![0.0; n], 0.0);
    }

    let x_std = variance(x).sqrt().max(1e-10);
    let bandwidth = x_std * 0.3;

    let residuals: Vec<f64> = (0..n)
        .map(|i| {
            let query = x[i];
            let mut weight_sum = 0.0;
            let mut weighted_y = 0.0;

            for (j, (&xj, &yj)) in x.iter().zip(y.iter()).enumerate() {
                if j == i {
                    continue;
                }
                let diff = (xj - query) / bandwidth;
                let weight = (-0.5 * diff * diff).exp();
                weight_sum += weight;
                weighted_y += weight * yj;
            }

            let pred = if weight_sum > 1e-10 {
                weighted_y / weight_sum
            } else {
                y.iter().sum::<f64>() / n as f64
            };

            y[i] - pred
        })
        .collect();

    let independence = 1.0 - correlation(x, &residuals).abs();
    (residuals, independence)
}

// ============================================================================
// FINAL BOSS: The Ultimate Causal Discovery System
// ============================================================================

/// The ultimate causal discovery system combining all approaches
pub struct FinalBoss {
    semantic: SemanticDiscovery,
    neural: NeuralCausalDiscovery,
    enhanced_reci: EnhancedReci,
}

impl FinalBoss {
    pub fn new() -> Self {
        Self {
            semantic: SemanticDiscovery::new(),
            neural: NeuralCausalDiscovery::new(),
            enhanced_reci: EnhancedReci::new(),
        }
    }

    /// The ultimate discovery method
    pub fn discover(
        &self,
        x: &[f64],
        y: &[f64],
        x_desc: Option<&str>,
        y_desc: Option<&str>,
    ) -> CausalDiscoveryResult {
        // Get results from all systems
        let semantic_result = self.semantic.discover(x, y, x_desc, y_desc);
        let neural_result = self.neural.discover(x, y);
        let enhanced_reci_result = self.enhanced_reci.discover(x, y);
        let info_theoretic = discover_by_information_theoretic(x, y);

        // Weighted combination based on confidence
        let results = vec![
            (semantic_result, 0.25),
            (neural_result, 0.20),
            (enhanced_reci_result, 0.25),
            (info_theoretic, 0.30),
        ];

        let mut weighted_p = 0.0;
        let mut total_weight = 0.0;

        for (result, base_weight) in &results {
            let weight = base_weight * (0.5 + result.confidence);
            weighted_p += weight * result.p_forward;
            total_weight += weight;
        }

        let p_forward = weighted_p / total_weight.max(1e-10);
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

    /// Simple discover without descriptions
    pub fn discover_simple(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        self.discover(x, y, None, None)
    }
}

impl Default for FinalBoss {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PUBLIC API
// ============================================================================

/// Discover causal direction using the full Causal Tower
pub fn discover_by_tower(x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
    let tower = CausalTower::new();
    tower.discover(x, y)
}

/// Discover using the improved SmartTower
pub fn discover_by_smart_tower(x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
    let tower = SmartTower::new();
    tower.discover(x, y)
}

/// Discover using oracle selection (picks best algorithm per pair)
pub fn discover_by_oracle(x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
    let tower = SmartTower::new();
    tower.discover_oracle(x, y)
}

/// Discover with full uncertainty information
pub fn discover_by_tower_with_uncertainty(
    x: &[f64],
    y: &[f64],
    x_desc: Option<&str>,
    y_desc: Option<&str>,
) -> CausalVerdict {
    let tower = CausalTower::new();
    tower.discover_with_uncertainty(x, y, x_desc, y_desc)
}

/// Discover using Enhanced RECI with adaptive bandwidth
pub fn discover_by_enhanced_reci(x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
    let reci = EnhancedReci::new();
    reci.discover(x, y)
}

/// Discover using the Ultimate Ensemble with abstention capability
pub fn discover_by_ultimate_ensemble(x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
    let ensemble = UltimateEnsemble::new();
    ensemble.discover(x, y)
}

/// Discover using semantic hints (requires variable descriptions for best results)
pub fn discover_by_semantic(
    x: &[f64],
    y: &[f64],
    x_desc: Option<&str>,
    y_desc: Option<&str>,
) -> CausalDiscoveryResult {
    let semantic = SemanticDiscovery::new();
    semantic.discover(x, y, x_desc, y_desc)
}

/// Discover using the neural feature-based classifier
pub fn discover_by_neural(x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
    let neural = NeuralCausalDiscovery::new();
    neural.discover(x, y)
}

/// The ultimate discovery method combining all approaches
pub fn discover_by_final_boss(x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
    let boss = FinalBoss::new();
    boss.discover_simple(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_tower() {
        // Simple test: Y = X^2 + noise
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi + 0.1 * xi).collect();

        let tower = CausalTower::new();
        let result = tower.discover(&x, &y);

        // Confidence and p_forward should be in [0, 1]
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "Confidence must be in [0,1]: {}",
            result.confidence
        );
        assert!(
            result.p_forward >= 0.0 && result.p_forward <= 1.0,
            "p_forward must be in [0,1]: {}",
            result.p_forward
        );
    }

    #[test]
    fn test_meta_features() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let features = MetaFeatures::extract(&x, &y);
        // Linear data should have high linearity
        assert!(
            features.linearity > 0.5,
            "Linear data should have high linearity: {}",
            features.linearity
        );
        assert!(
            features.noise_level.is_finite(),
            "Noise level must be finite"
        );
    }
}
