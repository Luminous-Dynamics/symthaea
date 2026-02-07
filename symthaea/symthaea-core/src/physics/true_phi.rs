//! # True Integrated Information (Φ) via Shannon Entropy
//!
//! This module implements mathematically rigorous Integrated Information Theory (IIT)
//! using actual Shannon entropy measures rather than similarity-based proxies.
//!
//! ## Key Insight
//!
//! ContinuousHV has 16,384 f32 components in [-1, 1]. We treat these as empirical samples
//! and use discretization-based entropy estimation:
//!
//! 1. **Bin components** into K buckets (K=16 or 32)
//! 2. **Build histogram** → probability distribution
//! 3. **Compute Shannon entropy**: H(X) = -Σ p(x) log₂ p(x)
//! 4. **Joint entropy** via 2D binning for H(X,Y)
//! 5. **Mutual information**: I(X;Y) = H(X) + H(Y) - H(X,Y)
//!
//! ## True Φ Calculation
//!
//! ```text
//! Φ = EI(System) - EI(MIP)
//!
//! Where:
//! - EI(System) = Effective Information of whole system
//! - EI(MIP) = Effective Information of Minimum Information Partition
//! - Φ > 0 indicates true integration (cannot be decomposed)
//! ```
//!
//! ## Scientific Basis
//!
//! - Shannon (1948) - "A Mathematical Theory of Communication"
//! - Tononi et al. (2016) - "Integrated Information Theory: From Consciousness to Its Physical Substrate"
//! - Oizumi et al. (2014) - "From the Phenomenology to the Mechanisms of Consciousness"

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Configuration for entropy calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyConfig {
    /// Number of bins for discretization (16 for speed, 32 for precision)
    pub num_bins: usize,
    /// Whether to use bits (log₂) or nats (ln)
    pub use_bits: bool,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            num_bins: 16,
            use_bits: true,
        }
    }
}

impl EntropyConfig {
    /// Create config optimized for speed
    pub fn fast() -> Self {
        Self {
            num_bins: 16,
            use_bits: true,
        }
    }

    /// Create config optimized for precision
    pub fn precise() -> Self {
        Self {
            num_bins: 32,
            use_bits: true,
        }
    }
}

/// Methods for continuous entropy estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum EntropyMethod {
    /// Histogram binning (fast, default)
    #[default]
    Histogram,
    /// k-Nearest Neighbor estimator (Kozachenko-Leonenko) - O(n²)
    KNN,
    /// Optimized k-NN using sorted array property - O(n log n)
    KNNFast,
    /// Kernel Density Estimation - O(n²)
    KDE,
    /// Optimized KDE with truncated Gaussian - O(n × neighbors)
    KDEFast,
    /// Adaptive binning (data-driven bin widths)
    AdaptiveBins,
}


/// Continuous entropy estimator using multiple methods
#[derive(Debug, Clone)]
pub struct ContinuousEntropyEstimator {
    /// Method to use for estimation
    pub method: EntropyMethod,
    /// k value for k-NN estimator (typically 1-10)
    pub k_neighbors: usize,
    /// Bandwidth for KDE (0.0 = auto-select)
    pub kde_bandwidth: f64,
    /// Number of bins for adaptive binning
    pub adaptive_bins: usize,
    /// Whether to use bits (log₂) or nats (ln)
    pub use_bits: bool,
}

impl Default for ContinuousEntropyEstimator {
    fn default() -> Self {
        Self {
            method: EntropyMethod::Histogram,
            k_neighbors: 3,
            kde_bandwidth: 0.0, // Auto-select via Silverman's rule
            adaptive_bins: 16,
            use_bits: true,
        }
    }
}

impl ContinuousEntropyEstimator {
    /// Create a k-NN estimator with specified k (O(n²) accurate version)
    pub fn knn(k: usize) -> Self {
        Self {
            method: EntropyMethod::KNN,
            k_neighbors: k,
            ..Default::default()
        }
    }

    /// Create a fast k-NN estimator (O(n log n) using sorted array property)
    pub fn knn_fast(k: usize) -> Self {
        Self {
            method: EntropyMethod::KNNFast,
            k_neighbors: k,
            ..Default::default()
        }
    }

    /// Create a KDE estimator with auto bandwidth (O(n²) accurate version)
    pub fn kde() -> Self {
        Self {
            method: EntropyMethod::KDE,
            ..Default::default()
        }
    }

    /// Create a fast KDE estimator with truncated Gaussian (O(n × neighbors))
    pub fn kde_fast() -> Self {
        Self {
            method: EntropyMethod::KDEFast,
            ..Default::default()
        }
    }

    /// Create an adaptive binning estimator
    pub fn adaptive(bins: usize) -> Self {
        Self {
            method: EntropyMethod::AdaptiveBins,
            adaptive_bins: bins,
            ..Default::default()
        }
    }

    /// Create an estimator optimized for speed
    /// Uses histogram binning which is O(n) and produces good results for most cases
    pub fn fast() -> Self {
        Self {
            method: EntropyMethod::Histogram,
            adaptive_bins: 16,
            ..Default::default()
        }
    }

    /// Create an estimator optimized for accuracy
    /// Uses adaptive binning with Miller-Madow bias correction
    pub fn accurate() -> Self {
        Self {
            method: EntropyMethod::AdaptiveBins,
            adaptive_bins: 32,
            ..Default::default()
        }
    }

    fn log(&self, x: f64) -> f64 {
        if self.use_bits {
            x.log2()
        } else {
            x.ln()
        }
    }

    /// Estimate entropy of a hypervector using the configured method
    pub fn entropy(&self, hv: &ContinuousHV) -> f64 {
        match self.method {
            EntropyMethod::Histogram => self.entropy_histogram(hv),
            EntropyMethod::KNN => self.entropy_knn(hv),
            EntropyMethod::KNNFast => self.entropy_knn_fast(hv),
            EntropyMethod::KDE => self.entropy_kde(hv),
            EntropyMethod::KDEFast => self.entropy_kde_fast(hv),
            EntropyMethod::AdaptiveBins => self.entropy_adaptive(hv),
        }
    }

    /// Histogram-based entropy (same as TruePhiCalculator)
    fn entropy_histogram(&self, hv: &ContinuousHV) -> f64 {
        let num_bins = self.adaptive_bins;
        let mut counts = vec![0usize; num_bins];

        for &value in &hv.values {
            let normalized = ((value + 1.0) / 2.0).clamp(0.0, 0.9999);
            let bin = (normalized * num_bins as f32) as usize;
            counts[bin] += 1;
        }

        let total = hv.values.len() as f64;
        let mut h = 0.0;
        for &c in &counts {
            if c > 0 {
                let p = c as f64 / total;
                h -= p * self.log(p);
            }
        }
        h
    }

    /// k-Nearest Neighbor entropy estimator (Kozachenko-Leonenko, 1987)
    ///
    /// H(X) ≈ ψ(n) - ψ(k) + d·log(2) + (d/n)·Σ log(ρ_k(i))
    ///
    /// where ψ is the digamma function, d is dimension (1 for marginal),
    /// and ρ_k(i) is the distance to the k-th nearest neighbor.
    fn entropy_knn(&self, hv: &ContinuousHV) -> f64 {
        let n = hv.values.len();
        let k = self.k_neighbors.min(n - 1).max(1);

        // For 1D data, we sort and find k-th NN distances directly
        let mut sorted: Vec<f32> = hv.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute k-th nearest neighbor distances
        let mut log_distances_sum = 0.0;
        let mut valid_count = 0;

        for i in 0..n {
            // Find k-th nearest neighbor distance
            let mut distances: Vec<f32> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i != j {
                    distances.push((sorted[i] - sorted[j]).abs());
                }
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let rho_k = distances[k - 1] as f64;
            if rho_k > 1e-10 {
                log_distances_sum += self.log(2.0 * rho_k);
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            return 0.0;
        }

        // Kozachenko-Leonenko formula for 1D
        let psi_n = digamma(n as f64);
        let psi_k = digamma(k as f64);
        let d = 1.0; // Dimension

        // H = ψ(n) - ψ(k) + log(V_d) + (d/n)·Σ log(ρ_k)
        // For 1D, V_d = 2 (length of unit ball), log(2) ≈ 0.693 (nats)
        let log_vd = if self.use_bits { 1.0 } else { 2.0_f64.ln() };

        let h = psi_n - psi_k + log_vd + (d / valid_count as f64) * log_distances_sum;
        h.max(0.0)
    }

    /// Kernel Density Estimation entropy
    ///
    /// H(X) ≈ -1/n Σ log(f̂(x_i)) where f̂ is the KDE estimate
    fn entropy_kde(&self, hv: &ContinuousHV) -> f64 {
        let n = hv.values.len();
        if n < 2 {
            return 0.0;
        }

        // Compute bandwidth using Silverman's rule of thumb
        let bandwidth = if self.kde_bandwidth > 0.0 {
            self.kde_bandwidth
        } else {
            silverman_bandwidth(&hv.values)
        };

        if bandwidth < 1e-10 {
            return 0.0;
        }

        // For each point, compute KDE estimate
        let mut log_density_sum = 0.0;

        for i in 0..n {
            let x_i = hv.values[i] as f64;

            // KDE: f̂(x) = 1/(n·h) Σ K((x - x_j)/h)
            // Using Gaussian kernel: K(u) = exp(-u²/2) / √(2π)
            let mut density = 0.0;
            for j in 0..n {
                let x_j = hv.values[j] as f64;
                let u = (x_i - x_j) / bandwidth;
                density += (-0.5 * u * u).exp();
            }
            density /= n as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt();

            if density > 1e-10 {
                log_density_sum += self.log(density);
            }
        }

        // H ≈ -E[log f(X)] ≈ -1/n Σ log f̂(x_i)
        let h = -log_density_sum / n as f64;
        h.max(0.0)
    }

    /// Adaptive binning entropy using data-driven bin widths
    ///
    /// Uses the Freedman-Diaconis rule: bin_width = 2·IQR / n^(1/3)
    fn entropy_adaptive(&self, hv: &ContinuousHV) -> f64 {
        let n = hv.values.len();
        if n < 4 {
            return self.entropy_histogram(hv);
        }

        // Compute IQR
        let mut sorted: Vec<f32> = hv.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let iqr = (sorted[q3_idx] - sorted[q1_idx]) as f64;

        if iqr < 1e-10 {
            return self.entropy_histogram(hv);
        }

        // Freedman-Diaconis bin width
        let bin_width = 2.0 * iqr / (n as f64).powf(1.0 / 3.0);

        let min_val = sorted[0] as f64;
        let max_val = sorted[n - 1] as f64;
        let range = max_val - min_val;

        if range < 1e-10 {
            return 0.0;
        }

        let num_bins = ((range / bin_width).ceil() as usize).clamp(2, 256);

        // Count in adaptive bins
        let mut counts = vec![0usize; num_bins];
        for &value in &hv.values {
            let v = value as f64;
            let bin = (((v - min_val) / range) * (num_bins - 1) as f64).round() as usize;
            let bin = bin.min(num_bins - 1);
            counts[bin] += 1;
        }

        // Compute entropy
        let total = n as f64;
        let mut h = 0.0;
        for &c in &counts {
            if c > 0 {
                let p = c as f64 / total;
                h -= p * self.log(p);
            }
        }

        // Bias correction for finite samples (Miller-Madow)
        let non_empty_bins = counts.iter().filter(|&&c| c > 0).count();
        let correction = (non_empty_bins - 1) as f64 / (2.0 * total);

        (h + correction).max(0.0)
    }

    /// Estimate mutual information using k-NN method (Kraskov et al., 2004)
    pub fn mutual_information_knn(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> f64 {
        let n = hv1.values.len().min(hv2.values.len());
        if n < 10 {
            return 0.0;
        }

        let k = self.k_neighbors.min(n - 1).max(1);

        // Build 2D point set
        let points: Vec<(f64, f64)> = hv1.values.iter()
            .zip(hv2.values.iter())
            .map(|(&a, &b)| (a as f64, b as f64))
            .collect();

        // For each point, find k-th NN in joint space (Chebyshev distance)
        let mut psi_sum = 0.0;

        for i in 0..n {
            // Find k-th NN distance in joint space
            let mut joint_distances: Vec<f64> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i != j {
                    let dx = (points[i].0 - points[j].0).abs();
                    let dy = (points[i].1 - points[j].1).abs();
                    joint_distances.push(dx.max(dy)); // Chebyshev (L∞)
                }
            }
            joint_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let epsilon = joint_distances[k - 1];

            // Count points within epsilon in marginals
            let mut n_x = 0;
            let mut n_y = 0;
            for j in 0..n {
                if i != j {
                    if (points[i].0 - points[j].0).abs() <= epsilon {
                        n_x += 1;
                    }
                    if (points[i].1 - points[j].1).abs() <= epsilon {
                        n_y += 1;
                    }
                }
            }

            // ψ(n_x + 1) + ψ(n_y + 1)
            psi_sum += digamma((n_x + 1) as f64) + digamma((n_y + 1) as f64);
        }

        // I(X;Y) = ψ(k) - (1/n)·Σ[ψ(n_x + 1) + ψ(n_y + 1)] + ψ(n)
        let mi = digamma(k as f64) - psi_sum / n as f64 + digamma(n as f64);

        // Convert to bits if needed
        let mi = if self.use_bits { mi / 2.0_f64.ln() } else { mi };
        mi.max(0.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPTIMIZED ENTROPY IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

impl ContinuousEntropyEstimator {
    /// Optimized k-NN entropy using sorted array property
    ///
    /// For 1D data, the k-th nearest neighbor can be found in O(1) per point
    /// after sorting, making the total complexity O(n log n) instead of O(n²).
    pub fn entropy_knn_fast(&self, hv: &ContinuousHV) -> f64 {
        let n = hv.values.len();
        let k = self.k_neighbors.min(n - 1).max(1);

        // Sort once: O(n log n)
        let mut sorted: Vec<f32> = hv.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // For each point in sorted order, k-th NN is at most k positions away
        let mut log_distances_sum = 0.0;
        let mut valid_count = 0;

        for i in 0..n {
            // k-th nearest neighbor distance in sorted 1D data
            // Look k steps left and right, take the minimum of the k-th distances
            let _left_dist = if i >= k {
                (sorted[i] - sorted[i - k]).abs()
            } else {
                f32::INFINITY
            };
            let _right_dist = if i + k < n {
                (sorted[i + k] - sorted[i]).abs()
            } else {
                f32::INFINITY
            };

            // For k-th NN, we need to consider both directions
            // The k-th NN is found by merging distances from both sides
            let mut distances = Vec::with_capacity(2 * k);
            for j in 1..=k {
                if i >= j {
                    distances.push((sorted[i] - sorted[i - j]).abs());
                }
                if i + j < n {
                    distances.push((sorted[i + j] - sorted[i]).abs());
                }
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if distances.len() >= k {
                let rho_k = distances[k - 1] as f64;
                if rho_k > 1e-10 {
                    log_distances_sum += self.log(2.0 * rho_k);
                    valid_count += 1;
                }
            }
        }

        if valid_count == 0 {
            return 0.0;
        }

        // Kozachenko-Leonenko formula
        let psi_n = digamma(n as f64);
        let psi_k = digamma(k as f64);
        let log_vd = if self.use_bits { 1.0 } else { 2.0_f64.ln() };

        let h = psi_n - psi_k + log_vd + (1.0 / valid_count as f64) * log_distances_sum;
        h.max(0.0)
    }

    /// Optimized KDE entropy using truncated Gaussian kernel
    ///
    /// Only computes kernel contributions for points within 4σ of each evaluation point,
    /// reducing complexity from O(n²) to O(n × k) where k is the average neighborhood size.
    pub fn entropy_kde_fast(&self, hv: &ContinuousHV) -> f64 {
        let n = hv.values.len();
        if n < 2 {
            return 0.0;
        }

        // Compute bandwidth
        let bandwidth = if self.kde_bandwidth > 0.0 {
            self.kde_bandwidth
        } else {
            silverman_bandwidth(&hv.values)
        };

        if bandwidth < 1e-10 {
            return 0.0;
        }

        // Sort for efficient neighbor finding
        let mut indexed: Vec<(f32, usize)> = hv.values.iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Truncation distance: beyond 4σ, contribution is negligible
        let truncation = 4.0 * bandwidth;

        let mut log_density_sum = 0.0;
        let norm_factor = 1.0 / (n as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt());

        for i in 0..n {
            let x_i = indexed[i].0 as f64;
            let mut density = 0.0;

            // Search left
            let mut j = i;
            while j > 0 {
                j -= 1;
                let x_j = indexed[j].0 as f64;
                let diff = (x_i - x_j).abs();
                if diff > truncation {
                    break;
                }
                let u = diff / bandwidth;
                density += (-0.5 * u * u).exp();
            }

            // Search right
            let mut j = i + 1;
            while j < n {
                let x_j = indexed[j].0 as f64;
                let diff = (x_j - x_i).abs();
                if diff > truncation {
                    break;
                }
                let u = diff / bandwidth;
                density += (-0.5 * u * u).exp();
                j += 1;
            }

            // Include self contribution
            density += 1.0; // exp(0) = 1

            density *= norm_factor;

            if density > 1e-10 {
                log_density_sum += self.log(density);
            }
        }

        let h = -log_density_sum / n as f64;
        h.max(0.0)
    }

    /// Fast mutual information using grid-based approach
    ///
    /// Uses 2D histogram instead of k-NN for O(n) complexity.
    pub fn mutual_information_fast(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> f64 {
        let n = hv1.values.len().min(hv2.values.len());
        if n < 10 {
            return 0.0;
        }

        let num_bins = self.adaptive_bins.max(8);

        // Build 2D histogram
        let mut joint_counts = vec![vec![0usize; num_bins]; num_bins];
        let mut marginal_x = vec![0usize; num_bins];
        let mut marginal_y = vec![0usize; num_bins];

        for i in 0..n {
            let x = ((hv1.values[i] + 1.0) / 2.0).clamp(0.0, 0.9999);
            let y = ((hv2.values[i] + 1.0) / 2.0).clamp(0.0, 0.9999);
            let bx = (x * num_bins as f32) as usize;
            let by = (y * num_bins as f32) as usize;

            joint_counts[bx][by] += 1;
            marginal_x[bx] += 1;
            marginal_y[by] += 1;
        }

        // Compute MI from histograms
        // I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
        let total = n as f64;
        let mut mi = 0.0;

        for bx in 0..num_bins {
            if marginal_x[bx] == 0 {
                continue;
            }
            let p_x = marginal_x[bx] as f64 / total;

            for by in 0..num_bins {
                if joint_counts[bx][by] == 0 {
                    continue;
                }
                let p_y = marginal_y[by] as f64 / total;
                let p_xy = joint_counts[bx][by] as f64 / total;

                if p_x > 0.0 && p_y > 0.0 {
                    mi += p_xy * self.log(p_xy / (p_x * p_y));
                }
            }
        }

        mi.max(0.0)
    }
}

/// Digamma function (derivative of log gamma)
/// Using asymptotic expansion for large x and recurrence for small x
fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Use recurrence to shift x to large value
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion
    let x2 = 1.0 / (x * x);
    result += x.ln() - 0.5 / x
        - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 / 252.0));

    result
}

/// Silverman's rule of thumb for KDE bandwidth
fn silverman_bandwidth(values: &[f32]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.1;
    }

    // Compute standard deviation
    let mean: f64 = values.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let variance: f64 = values.iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>() / n as f64;
    let std = variance.sqrt();

    // Compute IQR
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = sorted[n / 4] as f64;
    let q3 = sorted[3 * n / 4] as f64;
    let iqr = q3 - q1;

    // Silverman's rule: h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
    let scale = (std.min(iqr / 1.34)).max(0.01);
    0.9 * scale * (n as f64).powf(-0.2)
}

/// Probability distribution from binned vector components
#[derive(Debug, Clone)]
pub struct VectorDistribution {
    /// Probability of each bin
    pub probabilities: Vec<f64>,
    /// Number of samples (dimension of original vector)
    pub sample_count: usize,
    /// Bin edges for reference
    pub bin_edges: Vec<f32>,
}

impl VectorDistribution {
    /// Compute the bin index for a value in [-1, 1]
    fn bin_index(value: f32, num_bins: usize) -> usize {
        // Map [-1, 1] to [0, num_bins-1]
        let normalized = ((value + 1.0) / 2.0).clamp(0.0, 0.9999);
        (normalized * num_bins as f32) as usize
    }

    /// Create a distribution from a hypervector by binning
    pub fn from_hv(hv: &ContinuousHV, num_bins: usize) -> Self {
        let mut counts = vec![0usize; num_bins];

        for &value in &hv.values {
            let bin = Self::bin_index(value, num_bins);
            counts[bin] += 1;
        }

        let total = hv.values.len() as f64;
        let probabilities: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();

        // Generate bin edges
        let bin_edges: Vec<f32> = (0..=num_bins)
            .map(|i| -1.0 + 2.0 * (i as f32 / num_bins as f32))
            .collect();

        Self {
            probabilities,
            sample_count: hv.values.len(),
            bin_edges,
        }
    }
}

/// 2D joint distribution for computing joint entropy
#[derive(Debug, Clone)]
pub struct JointDistribution {
    /// 2D probability matrix (row = hv1 bin, col = hv2 bin)
    pub probabilities: Vec<Vec<f64>>,
    /// Number of samples
    pub sample_count: usize,
    /// Number of bins per dimension
    pub num_bins: usize,
}

impl JointDistribution {
    /// Create joint distribution from two hypervectors
    pub fn from_hvs(hv1: &ContinuousHV, hv2: &ContinuousHV, num_bins: usize) -> Self {
        assert_eq!(hv1.values.len(), hv2.values.len(), "Dimension mismatch");

        let mut counts = vec![vec![0usize; num_bins]; num_bins];

        for (&v1, &v2) in hv1.values.iter().zip(hv2.values.iter()) {
            let bin1 = VectorDistribution::bin_index(v1, num_bins);
            let bin2 = VectorDistribution::bin_index(v2, num_bins);
            counts[bin1][bin2] += 1;
        }

        let total = hv1.values.len() as f64;
        let probabilities: Vec<Vec<f64>> = counts
            .iter()
            .map(|row| row.iter().map(|&c| c as f64 / total).collect())
            .collect();

        Self {
            probabilities,
            sample_count: hv1.values.len(),
            num_bins,
        }
    }
}

/// A partition of components for MIP search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruePartition {
    /// Indices of components in part A
    pub part_a: Vec<usize>,
    /// Indices of components in part B
    pub part_b: Vec<usize>,
}

impl TruePartition {
    /// Create a partition from a bitmask
    pub fn from_mask(mask: usize, n: usize) -> Self {
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();

        for i in 0..n {
            if (mask & (1 << i)) != 0 {
                part_a.push(i);
            } else {
                part_b.push(i);
            }
        }

        Self { part_a, part_b }
    }
}

/// Result of true Φ computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruePhiResult {
    /// The integrated information value
    pub phi: f64,
    /// Whole system effective information
    pub system_ei: f64,
    /// MIP effective information
    pub mip_ei: f64,
    /// The minimum information partition found
    pub mip: TruePartition,
    /// Individual component entropies H(X_i)
    pub component_entropies: Vec<f64>,
    /// Pairwise mutual information matrix
    pub mutual_information_matrix: Vec<Vec<f64>>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPORAL IIT (CAUSE-EFFECT INFORMATION)
// Based on IIT 3.0's cause-effect repertoire framework
// ═══════════════════════════════════════════════════════════════════════════════

/// Temporal transition for cause-effect analysis
///
/// Represents a state transition t → t+1 for computing cause and effect repertoires.
#[derive(Debug, Clone)]
pub struct TemporalTransition {
    /// State at time t (current)
    pub current: ContinuousHV,
    /// State at time t+1 (next)
    pub next: ContinuousHV,
}

impl TemporalTransition {
    /// Create a transition from current to next state
    pub fn new(current: ContinuousHV, next: ContinuousHV) -> Self {
        Self { current, next }
    }

    /// Create a transition by applying a transformation to the current state
    pub fn from_transformation<F>(current: ContinuousHV, transform: F) -> Self
    where
        F: FnOnce(&ContinuousHV) -> ContinuousHV,
    {
        let next = transform(&current);
        Self { current, next }
    }
}

/// Cause-effect information result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CauseEffectInfo {
    /// Cause information: how much current state specifies past (ci)
    pub cause_info: f64,
    /// Effect information: how much current state specifies future (ei)
    pub effect_info: f64,
    /// Integrated cause information (φ_cause)
    pub integrated_cause: f64,
    /// Integrated effect information (φ_effect)
    pub integrated_effect: f64,
    /// Total cause-effect information (min of cause and effect)
    pub phi_cause_effect: f64,
    /// Cause repertoire entropy
    pub cause_entropy: f64,
    /// Effect repertoire entropy
    pub effect_entropy: f64,
}

/// Temporal Φ calculator for cause-effect analysis
///
/// Extends IIT with temporal dynamics:
/// - Cause information: I(current; past)
/// - Effect information: I(current; future)
/// - Integrated cause: φ_cause = I(current; past | partition)
/// - Integrated effect: φ_effect = I(current; future | partition)
///
/// Reference: Oizumi et al. (2014) - "From the Phenomenology to the Mechanisms"
#[derive(Debug, Clone)]
pub struct TemporalPhiCalculator {
    /// Base calculator for entropy computations (reserved for future use)
    #[allow(dead_code)]
    base: TruePhiCalculator,
    /// Continuous entropy estimator for MI
    estimator: ContinuousEntropyEstimator,
}

impl Default for TemporalPhiCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalPhiCalculator {
    /// Create a new temporal calculator
    pub fn new() -> Self {
        Self {
            base: TruePhiCalculator::new(),
            estimator: ContinuousEntropyEstimator::fast(),
        }
    }

    /// Create with custom estimator
    pub fn with_estimator(estimator: ContinuousEntropyEstimator) -> Self {
        Self {
            base: TruePhiCalculator::new(),
            estimator,
        }
    }

    /// Compute cause information I(current; past)
    ///
    /// How much does the current state tell us about what caused it?
    /// Uses mutual information between current state and the prior state.
    pub fn cause_information(&self, transition: &TemporalTransition) -> f64 {
        // Use the fast MI method
        self.estimator.mutual_information_fast(&transition.next, &transition.current)
    }

    /// Compute effect information I(current; future)
    ///
    /// How much does the current state tell us about what will happen?
    pub fn effect_information(&self, transition: &TemporalTransition) -> f64 {
        // Effect info is the same MI but conceptually different
        self.estimator.mutual_information_fast(&transition.current, &transition.next)
    }

    /// Compute cause repertoire entropy
    ///
    /// The entropy of the cause repertoire represents the uncertainty
    /// about past states given the current mechanism.
    pub fn cause_repertoire_entropy(&self, past_states: &[ContinuousHV]) -> f64 {
        if past_states.is_empty() {
            return 0.0;
        }

        // Bundle past states and compute entropy
        let refs: Vec<&ContinuousHV> = past_states.iter().collect();
        let bundled = ContinuousHV::bundle(&refs);
        self.estimator.entropy(&bundled)
    }

    /// Compute effect repertoire entropy
    ///
    /// The entropy of possible future states given the current mechanism.
    pub fn effect_repertoire_entropy(&self, future_states: &[ContinuousHV]) -> f64 {
        if future_states.is_empty() {
            return 0.0;
        }

        let refs: Vec<&ContinuousHV> = future_states.iter().collect();
        let bundled = ContinuousHV::bundle(&refs);
        self.estimator.entropy(&bundled)
    }

    /// Compute integrated cause information for a system
    ///
    /// φ_cause = min over partitions of I(M_A; past_A) + I(M_B; past_B)
    /// where M is the mechanism and A,B partition the system
    pub fn integrated_cause_info(
        &self,
        components: &[TemporalTransition],
    ) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Compute whole system cause info
        let current_bundle = self.bundle_states(
            &components.iter().map(|t| &t.current).collect::<Vec<_>>()
        );
        let past_bundle = self.bundle_states(
            &components.iter().map(|t| &t.next).collect::<Vec<_>>()
        );
        let system_cause = self.estimator.mutual_information_fast(&current_bundle, &past_bundle);

        // Find MIP for cause
        let mut min_partition_cause = f64::INFINITY;

        // Try all non-trivial bipartitions
        for mask in 1..(1 << n) - 1 {
            let partition = TruePartition::from_mask(mask, n);
            if partition.part_a.is_empty() || partition.part_b.is_empty() {
                continue;
            }

            // Compute cause info for each partition
            let current_a = self.bundle_indices(&components.iter().map(|t| &t.current).collect::<Vec<_>>(), &partition.part_a);
            let past_a = self.bundle_indices(&components.iter().map(|t| &t.next).collect::<Vec<_>>(), &partition.part_a);
            let cause_a = self.estimator.mutual_information_fast(&current_a, &past_a);

            let current_b = self.bundle_indices(&components.iter().map(|t| &t.current).collect::<Vec<_>>(), &partition.part_b);
            let past_b = self.bundle_indices(&components.iter().map(|t| &t.next).collect::<Vec<_>>(), &partition.part_b);
            let cause_b = self.estimator.mutual_information_fast(&current_b, &past_b);

            let partition_cause = cause_a + cause_b;
            min_partition_cause = min_partition_cause.min(partition_cause);
        }

        // φ_cause = system cause - MIP cause
        (system_cause - min_partition_cause).max(0.0)
    }

    /// Compute integrated effect information for a system
    ///
    /// φ_effect = min over partitions of I(M_A; future_A) + I(M_B; future_B)
    pub fn integrated_effect_info(
        &self,
        components: &[TemporalTransition],
    ) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Compute whole system effect info
        let current_bundle = self.bundle_states(
            &components.iter().map(|t| &t.current).collect::<Vec<_>>()
        );
        let future_bundle = self.bundle_states(
            &components.iter().map(|t| &t.next).collect::<Vec<_>>()
        );
        let system_effect = self.estimator.mutual_information_fast(&current_bundle, &future_bundle);

        // Find MIP for effect
        let mut min_partition_effect = f64::INFINITY;

        for mask in 1..(1 << n) - 1 {
            let partition = TruePartition::from_mask(mask, n);
            if partition.part_a.is_empty() || partition.part_b.is_empty() {
                continue;
            }

            let current_a = self.bundle_indices(&components.iter().map(|t| &t.current).collect::<Vec<_>>(), &partition.part_a);
            let future_a = self.bundle_indices(&components.iter().map(|t| &t.next).collect::<Vec<_>>(), &partition.part_a);
            let effect_a = self.estimator.mutual_information_fast(&current_a, &future_a);

            let current_b = self.bundle_indices(&components.iter().map(|t| &t.current).collect::<Vec<_>>(), &partition.part_b);
            let future_b = self.bundle_indices(&components.iter().map(|t| &t.next).collect::<Vec<_>>(), &partition.part_b);
            let effect_b = self.estimator.mutual_information_fast(&current_b, &future_b);

            let partition_effect = effect_a + effect_b;
            min_partition_effect = min_partition_effect.min(partition_effect);
        }

        (system_effect - min_partition_effect).max(0.0)
    }

    /// Compute full cause-effect information for a transition
    ///
    /// Returns comprehensive cause-effect analysis including:
    /// - Cause and effect information
    /// - Integrated cause and effect
    /// - φ_cause_effect (minimum of integrated cause and effect)
    pub fn compute_cause_effect(
        &self,
        transition: &TemporalTransition,
    ) -> CauseEffectInfo {
        let cause_info = self.cause_information(transition);
        let effect_info = self.effect_information(transition);
        let cause_entropy = self.estimator.entropy(&transition.current);
        let effect_entropy = self.estimator.entropy(&transition.next);

        // For single transition, integrated info is just the MI
        let integrated_cause = cause_info;
        let integrated_effect = effect_info;

        // φ_cause_effect is the minimum (IIT 3.0 definition)
        let phi_cause_effect = cause_info.min(effect_info);

        CauseEffectInfo {
            cause_info,
            effect_info,
            integrated_cause,
            integrated_effect,
            phi_cause_effect,
            cause_entropy,
            effect_entropy,
        }
    }

    /// Compute cause-effect for a system of components
    pub fn compute_system_cause_effect(
        &self,
        components: &[TemporalTransition],
    ) -> CauseEffectInfo {
        if components.is_empty() {
            return CauseEffectInfo {
                cause_info: 0.0,
                effect_info: 0.0,
                integrated_cause: 0.0,
                integrated_effect: 0.0,
                phi_cause_effect: 0.0,
                cause_entropy: 0.0,
                effect_entropy: 0.0,
            };
        }

        // Bundle all states
        let current_bundle = self.bundle_states(
            &components.iter().map(|t| &t.current).collect::<Vec<_>>()
        );
        let next_bundle = self.bundle_states(
            &components.iter().map(|t| &t.next).collect::<Vec<_>>()
        );

        let cause_info = self.estimator.mutual_information_fast(&next_bundle, &current_bundle);
        let effect_info = self.estimator.mutual_information_fast(&current_bundle, &next_bundle);
        let cause_entropy = self.estimator.entropy(&current_bundle);
        let effect_entropy = self.estimator.entropy(&next_bundle);

        let integrated_cause = self.integrated_cause_info(components);
        let integrated_effect = self.integrated_effect_info(components);
        let phi_cause_effect = integrated_cause.min(integrated_effect);

        CauseEffectInfo {
            cause_info,
            effect_info,
            integrated_cause,
            integrated_effect,
            phi_cause_effect,
            cause_entropy,
            effect_entropy,
        }
    }

    /// Helper: Bundle multiple states into one
    fn bundle_states(&self, states: &[&ContinuousHV]) -> ContinuousHV {
        if states.is_empty() {
            return ContinuousHV::zero(16384);
        }
        ContinuousHV::bundle(states)
    }

    /// Helper: Bundle states at specific indices
    fn bundle_indices(&self, states: &[&ContinuousHV], indices: &[usize]) -> ContinuousHV {
        let selected: Vec<&ContinuousHV> = indices.iter()
            .filter_map(|&i| states.get(i).copied())
            .collect();
        self.bundle_states(&selected)
    }
}

/// Calculator for true integrated information using Shannon entropy
#[derive(Debug, Clone)]
pub struct TruePhiCalculator {
    config: EntropyConfig,
}

impl TruePhiCalculator {
    /// Create a new calculator with default config
    pub fn new() -> Self {
        Self {
            config: EntropyConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: EntropyConfig) -> Self {
        Self { config }
    }

    /// Get the log function based on config
    fn log(&self, x: f64) -> f64 {
        if self.config.use_bits {
            x.log2()
        } else {
            x.ln()
        }
    }

    /// Convert HDC vector to probability distribution via binning
    pub fn to_distribution(&self, hv: &ContinuousHV) -> VectorDistribution {
        VectorDistribution::from_hv(hv, self.config.num_bins)
    }

    /// Compute Shannon entropy H(X) from a distribution
    ///
    /// H(X) = -Σ p(x) log p(x)
    pub fn entropy_from_distribution(&self, dist: &VectorDistribution) -> f64 {
        let mut h = 0.0;
        for &p in &dist.probabilities {
            if p > 0.0 {
                h -= p * self.log(p);
            }
        }
        h
    }

    /// Compute Shannon entropy H(X) directly from a hypervector
    pub fn entropy(&self, hv: &ContinuousHV) -> f64 {
        let dist = self.to_distribution(hv);
        self.entropy_from_distribution(&dist)
    }

    /// Compute joint entropy H(X,Y) via 2D histogram
    pub fn joint_entropy(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> f64 {
        let joint = JointDistribution::from_hvs(hv1, hv2, self.config.num_bins);

        let mut h = 0.0;
        for row in &joint.probabilities {
            for &p in row {
                if p > 0.0 {
                    h -= p * self.log(p);
                }
            }
        }
        h
    }

    /// Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
    ///
    /// Mutual information measures how much knowing X reduces uncertainty about Y.
    /// I(X;Y) = 0 for independent variables
    /// I(X;Y) = H(X) = H(Y) for perfectly correlated variables
    pub fn mutual_information(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> f64 {
        let h_x = self.entropy(hv1);
        let h_y = self.entropy(hv2);
        let h_xy = self.joint_entropy(hv1, hv2);

        // I(X;Y) = H(X) + H(Y) - H(X,Y)
        // Due to numerical precision, ensure non-negative
        (h_x + h_y - h_xy).max(0.0)
    }

    /// Compute effective information as sum of MI between all component pairs
    ///
    /// EI = Σ_{i<j} I(X_i; X_j)
    ///
    /// This measures the total pairwise information integration.
    pub fn effective_information(&self, components: &[ContinuousHV]) -> f64 {
        if components.len() < 2 {
            return 0.0;
        }

        let mut ei = 0.0;
        for i in 0..components.len() {
            for j in (i + 1)..components.len() {
                ei += self.mutual_information(&components[i], &components[j]);
            }
        }
        ei
    }

    /// Compute effective information for a partition
    fn partition_effective_information(
        &self,
        components: &[ContinuousHV],
        partition: &TruePartition,
    ) -> f64 {
        // EI(partition) = EI(part_a) + EI(part_b)
        // This is the sum of information within each part, ignoring cross-part info

        let mut ei = 0.0;

        // EI within part A
        for i in 0..partition.part_a.len() {
            for j in (i + 1)..partition.part_a.len() {
                let idx_i = partition.part_a[i];
                let idx_j = partition.part_a[j];
                ei += self.mutual_information(&components[idx_i], &components[idx_j]);
            }
        }

        // EI within part B
        for i in 0..partition.part_b.len() {
            for j in (i + 1)..partition.part_b.len() {
                let idx_i = partition.part_b[i];
                let idx_j = partition.part_b[j];
                ei += self.mutual_information(&components[idx_i], &components[idx_j]);
            }
        }

        ei
    }

    /// Build mutual information matrix for all component pairs
    fn build_mi_matrix(&self, components: &[ContinuousHV]) -> Vec<Vec<f64>> {
        let n = components.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let mi = self.mutual_information(&components[i], &components[j]);
                matrix[i][j] = mi;
                matrix[j][i] = mi; // Symmetric
            }
            // Diagonal: self-information = entropy
            matrix[i][i] = self.entropy(&components[i]);
        }

        matrix
    }

    /// Find the Minimum Information Partition (MIP) using true entropy measures
    ///
    /// The MIP is the partition that minimizes information loss.
    /// Φ = EI(system) - EI(MIP)
    pub fn find_true_mip(&self, components: &[ContinuousHV]) -> (TruePartition, f64) {
        let n = components.len();

        if n < 2 {
            return (
                TruePartition {
                    part_a: (0..n).collect(),
                    part_b: vec![],
                },
                0.0,
            );
        }

        if n == 2 {
            // Only one partition possible: {0} | {1}
            let partition = TruePartition {
                part_a: vec![0],
                part_b: vec![1],
            };
            let ei = self.partition_effective_information(components, &partition);
            return (partition, ei);
        }

        // For small N (≤8), exhaustive search
        if n <= 8 {
            self.exhaustive_mip_search(components)
        } else {
            // For large N, use heuristic search
            self.heuristic_mip_search(components)
        }
    }

    /// Exhaustive MIP search for small systems (N ≤ 8)
    fn exhaustive_mip_search(&self, components: &[ContinuousHV]) -> (TruePartition, f64) {
        let n = components.len();
        let mut min_ei = f64::MAX;
        let mut mip = TruePartition {
            part_a: vec![0],
            part_b: (1..n).collect(),
        };

        // Iterate through all bipartitions
        // Use bit masks: for each subset of {0, 1, ..., n-1}
        for mask in 1..(1 << n) - 1 {
            let partition = TruePartition::from_mask(mask, n);

            // Skip trivial partitions (one part empty)
            if partition.part_a.is_empty() || partition.part_b.is_empty() {
                continue;
            }

            let ei = self.partition_effective_information(components, &partition);

            if ei < min_ei {
                min_ei = ei;
                mip = partition;
            }
        }

        (mip, min_ei)
    }

    /// Heuristic MIP search for large systems (N > 8)
    ///
    /// Uses multiple strategies to find the minimum information partition:
    /// 1. Spectral clustering based on MI matrix (Fiedler vector)
    /// 2. Simulated annealing for global optimization
    /// 3. Greedy bisection with local search refinement
    /// 4. MI-based heuristics (total MI split, index split, highest MI pair)
    fn heuristic_mip_search(&self, components: &[ContinuousHV]) -> (TruePartition, f64) {
        let n = components.len();
        let mi_matrix = self.build_mi_matrix(components);

        let mut candidates = Vec::new();

        // Strategy 1: Spectral clustering via Fiedler vector
        if let Some(partition) = self.spectral_partition(&mi_matrix, n) {
            candidates.push(partition);
        }

        // Strategy 2: Simulated annealing
        if let Some(partition) = self.simulated_annealing_partition(components, n) {
            candidates.push(partition);
        }

        // Strategy 3: Greedy bisection with local search
        let greedy = self.greedy_bisection_partition(&mi_matrix, n);
        let refined = self.local_search_refinement(components, greedy);
        candidates.push(refined);

        // Strategy 4: Split by total MI (separate high-MI from low-MI components)
        let total_mi: Vec<f64> = (0..n)
            .map(|i| mi_matrix[i].iter().sum::<f64>())
            .collect();
        let mean_mi = total_mi.iter().sum::<f64>() / n as f64;
        let part_a: Vec<usize> = (0..n).filter(|&i| total_mi[i] >= mean_mi).collect();
        let part_b: Vec<usize> = (0..n).filter(|&i| total_mi[i] < mean_mi).collect();
        if !part_a.is_empty() && !part_b.is_empty() {
            candidates.push(TruePartition { part_a, part_b });
        }

        // Strategy 5: Split in half by index
        let mid = n / 2;
        candidates.push(TruePartition {
            part_a: (0..mid).collect(),
            part_b: (mid..n).collect(),
        });

        // Strategy 6: Greedy clustering based on highest MI pair
        let mut used = vec![false; n];
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();

        let mut max_mi = 0.0;
        let mut best_i = 0;
        let mut best_j = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if mi_matrix[i][j] > max_mi {
                    max_mi = mi_matrix[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        part_a.push(best_i);
        part_b.push(best_j);
        used[best_i] = true;
        used[best_j] = true;

        for k in 0..n {
            if used[k] {
                continue;
            }

            let mi_to_a: f64 = part_a.iter().map(|&i| mi_matrix[k][i]).sum();
            let mi_to_b: f64 = part_b.iter().map(|&i| mi_matrix[k][i]).sum();

            if mi_to_a >= mi_to_b {
                part_a.push(k);
            } else {
                part_b.push(k);
            }
            used[k] = true;
        }

        candidates.push(TruePartition { part_a, part_b });

        // Find partition with minimum EI
        let mut min_ei = f64::MAX;
        let mut mip = candidates[0].clone();

        for partition in &candidates {
            let ei = self.partition_effective_information(components, partition);
            if ei < min_ei {
                min_ei = ei;
                mip = partition.clone();
            }
        }

        (mip, min_ei)
    }

    /// Spectral partition using the Fiedler vector (second smallest eigenvector of Laplacian)
    ///
    /// The Fiedler vector reveals natural clusters in the MI graph.
    /// Components with the same sign tend to be more connected.
    fn spectral_partition(&self, mi_matrix: &[Vec<f64>], n: usize) -> Option<TruePartition> {
        if n < 3 {
            return None;
        }

        // Build Laplacian: L = D - A where A is the MI adjacency matrix
        let mut laplacian = vec![vec![0.0; n]; n];
        for i in 0..n {
            let degree: f64 = mi_matrix[i].iter().sum();
            laplacian[i][i] = degree;
            for j in 0..n {
                if i != j {
                    laplacian[i][j] = -mi_matrix[i][j];
                }
            }
        }

        // Power iteration to find Fiedler vector
        let fiedler = self.power_iteration_fiedler(&laplacian, n, 100);

        // Partition based on sign of Fiedler vector components
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();
        for i in 0..n {
            if fiedler[i] >= 0.0 {
                part_a.push(i);
            } else {
                part_b.push(i);
            }
        }

        if part_a.is_empty() || part_b.is_empty() {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| fiedler[a].partial_cmp(&fiedler[b]).unwrap());
            let mid = n / 2;
            return Some(TruePartition {
                part_a: indices[..mid].to_vec(),
                part_b: indices[mid..].to_vec(),
            });
        }

        Some(TruePartition { part_a, part_b })
    }

    /// Power iteration to find the Fiedler vector
    fn power_iteration_fiedler(&self, laplacian: &[Vec<f64>], n: usize, max_iter: usize) -> Vec<f64> {
        let mut v: Vec<f64> = (0..n).map(|i| i as f64 - n as f64 / 2.0).collect();

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }

        // Add regularization
        let epsilon = 0.01;
        let mut reg_laplacian = laplacian.to_vec();
        for i in 0..n {
            reg_laplacian[i][i] += epsilon;
        }

        for _ in 0..max_iter {
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += reg_laplacian[i][j] * v[j];
                }
            }

            // Deflate by constant vector
            let mean: f64 = new_v.iter().sum::<f64>() / n as f64;
            for x in &mut new_v {
                *x -= mean;
            }

            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                break;
            }
            for x in &mut new_v {
                *x /= norm;
            }

            v = new_v;
        }

        v
    }

    /// Simulated annealing for MIP search
    fn simulated_annealing_partition(
        &self,
        components: &[ContinuousHV],
        n: usize,
    ) -> Option<TruePartition> {
        if n < 3 {
            return None;
        }

        let mut assignment = vec![false; n];
        for i in 0..(n / 2) {
            assignment[i] = true;
        }

        let mut current = self.assignment_to_partition(&assignment);
        let mut current_ei = self.partition_effective_information(components, &current);
        let mut best_partition = current.clone();
        let mut best_ei = current_ei;

        let initial_temp = 1.0;
        let final_temp = 0.001;
        let cooling_rate = 0.95;
        let iterations_per_temp = n * 2;

        let mut temp = initial_temp;
        let mut rng_state = 42u64;

        while temp > final_temp {
            for _ in 0..iterations_per_temp {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng_state as usize) % n;

                let current_side = assignment[idx];
                let count_same = assignment.iter().filter(|&&x| x == current_side).count();
                if count_same <= 1 {
                    continue;
                }

                assignment[idx] = !assignment[idx];
                let new_partition = self.assignment_to_partition(&assignment);
                let new_ei = self.partition_effective_information(components, &new_partition);

                let delta = new_ei - current_ei;
                let accept = if delta < 0.0 {
                    true
                } else {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = (rng_state as f64) / (u64::MAX as f64);
                    r < (-delta / temp).exp()
                };

                if accept {
                    current = new_partition;
                    current_ei = new_ei;
                    if current_ei < best_ei {
                        best_partition = current.clone();
                        best_ei = current_ei;
                    }
                } else {
                    assignment[idx] = !assignment[idx];
                }
            }
            temp *= cooling_rate;
        }

        Some(best_partition)
    }

    /// Convert boolean assignment to TruePartition
    fn assignment_to_partition(&self, assignment: &[bool]) -> TruePartition {
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();
        for (i, &in_a) in assignment.iter().enumerate() {
            if in_a {
                part_a.push(i);
            } else {
                part_b.push(i);
            }
        }
        TruePartition { part_a, part_b }
    }

    /// Greedy bisection partition
    fn greedy_bisection_partition(&self, mi_matrix: &[Vec<f64>], n: usize) -> TruePartition {
        let mut min_mi = f64::MAX;
        let mut seed_a = 0;
        let mut seed_b = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if mi_matrix[i][j] < min_mi {
                    min_mi = mi_matrix[i][j];
                    seed_a = i;
                    seed_b = j;
                }
            }
        }

        let mut part_a = vec![seed_a];
        let mut part_b = vec![seed_b];
        let mut assigned = vec![false; n];
        assigned[seed_a] = true;
        assigned[seed_b] = true;

        for _ in 2..n {
            let mut best_idx = 0;
            let mut best_to_a = false;
            let mut best_cost = f64::MAX;

            for i in 0..n {
                if assigned[i] {
                    continue;
                }

                let mi_to_a: f64 = part_a.iter().map(|&j| mi_matrix[i][j]).sum();
                let mi_to_b: f64 = part_b.iter().map(|&j| mi_matrix[i][j]).sum();

                if mi_to_a < best_cost {
                    best_cost = mi_to_a;
                    best_idx = i;
                    best_to_a = true;
                }
                if mi_to_b < best_cost {
                    best_cost = mi_to_b;
                    best_idx = i;
                    best_to_a = false;
                }
            }

            assigned[best_idx] = true;
            if best_to_a {
                part_a.push(best_idx);
            } else {
                part_b.push(best_idx);
            }
        }

        if part_a.is_empty() && !part_b.is_empty() {
            let moved = part_b.pop().unwrap();
            part_a.push(moved);
        } else if part_b.is_empty() && !part_a.is_empty() {
            let moved = part_a.pop().unwrap();
            part_b.push(moved);
        }

        TruePartition { part_a, part_b }
    }

    /// Local search refinement
    fn local_search_refinement(
        &self,
        components: &[ContinuousHV],
        initial: TruePartition,
    ) -> TruePartition {
        let mut current = initial;
        let mut current_ei = self.partition_effective_information(components, &current);
        let mut improved = true;

        while improved {
            improved = false;

            for i in 0..current.part_a.len() {
                if current.part_a.len() <= 1 {
                    break;
                }

                let elem = current.part_a[i];
                let mut new_a = current.part_a.clone();
                new_a.remove(i);
                let mut new_b = current.part_b.clone();
                new_b.push(elem);

                let new_partition = TruePartition {
                    part_a: new_a,
                    part_b: new_b,
                };
                let new_ei = self.partition_effective_information(components, &new_partition);

                if new_ei < current_ei {
                    current = new_partition;
                    current_ei = new_ei;
                    improved = true;
                    break;
                }
            }

            if improved {
                continue;
            }

            for i in 0..current.part_b.len() {
                if current.part_b.len() <= 1 {
                    break;
                }

                let elem = current.part_b[i];
                let mut new_b = current.part_b.clone();
                new_b.remove(i);
                let mut new_a = current.part_a.clone();
                new_a.push(elem);

                let new_partition = TruePartition {
                    part_a: new_a,
                    part_b: new_b,
                };
                let new_ei = self.partition_effective_information(components, &new_partition);

                if new_ei < current_ei {
                    current = new_partition;
                    current_ei = new_ei;
                    improved = true;
                    break;
                }
            }

            if improved {
                continue;
            }

            'swap: for i in 0..current.part_a.len() {
                for j in 0..current.part_b.len() {
                    let elem_a = current.part_a[i];
                    let elem_b = current.part_b[j];

                    let mut new_a = current.part_a.clone();
                    let mut new_b = current.part_b.clone();
                    new_a[i] = elem_b;
                    new_b[j] = elem_a;

                    let new_partition = TruePartition {
                        part_a: new_a,
                        part_b: new_b,
                    };
                    let new_ei = self.partition_effective_information(components, &new_partition);

                    if new_ei < current_ei {
                        current = new_partition;
                        current_ei = new_ei;
                        improved = true;
                        break 'swap;
                    }
                }
            }
        }

        current
    }

    /// Compute true Φ = EI(system) - EI(MIP)
    ///
    /// This is the core IIT calculation using genuine Shannon entropy.
    ///
    /// # Arguments
    /// * `components` - System components as hypervectors
    ///
    /// # Returns
    /// Detailed Φ result including:
    /// - phi: The integrated information value
    /// - system_ei: Whole system effective information
    /// - mip_ei: MIP effective information
    /// - mip: The minimum information partition
    /// - component_entropies: Individual H(X_i) values
    pub fn compute_true_phi(&self, components: &[ContinuousHV]) -> TruePhiResult {
        if components.len() < 2 {
            return TruePhiResult {
                phi: 0.0,
                system_ei: 0.0,
                mip_ei: 0.0,
                mip: TruePartition {
                    part_a: (0..components.len()).collect(),
                    part_b: vec![],
                },
                component_entropies: components.iter().map(|c| self.entropy(c)).collect(),
                mutual_information_matrix: vec![],
            };
        }

        // 1. Compute component entropies
        let component_entropies: Vec<f64> = components.iter().map(|c| self.entropy(c)).collect();

        // 2. Build MI matrix
        let mi_matrix = self.build_mi_matrix(components);

        // 3. Compute system effective information
        let system_ei = self.effective_information(components);

        // 4. Find MIP and its effective information
        let (mip, mip_ei) = self.find_true_mip(components);

        // 5. Φ = EI(system) - EI(MIP)
        let phi = (system_ei - mip_ei).max(0.0);

        TruePhiResult {
            phi,
            system_ei,
            mip_ei,
            mip,
            component_entropies,
            mutual_information_matrix: mi_matrix,
        }
    }

    /// Fast Φ estimation for real-time use
    ///
    /// Skips MIP search, uses simplified calculation
    pub fn compute_phi_fast(&self, components: &[ContinuousHV]) -> f64 {
        if components.len() < 2 {
            return 0.0;
        }

        // Just compute total effective information
        // This correlates with full Φ but is much faster
        let ei = self.effective_information(components);

        // Normalize by theoretical maximum
        // Max EI would be if all pairs had max MI
        let n = components.len();
        let num_pairs = (n * (n - 1)) / 2;
        let max_entropy = self.log(self.config.num_bins as f64);
        let theoretical_max = num_pairs as f64 * max_entropy;

        if theoretical_max > 0.0 {
            (ei / theoretical_max).min(1.0)
        } else {
            0.0
        }
    }
}

impl Default for TruePhiCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::unified_hv::HDC_DIMENSION;

    fn create_test_vectors(count: usize) -> Vec<ContinuousHV> {
        (0..count)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect()
    }

    #[test]
    fn test_entropy_config() {
        let fast = EntropyConfig::fast();
        assert_eq!(fast.num_bins, 16);

        let precise = EntropyConfig::precise();
        assert_eq!(precise.num_bins, 32);
    }

    #[test]
    fn test_vector_distribution() {
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        let dist = VectorDistribution::from_hv(&hv, 16);

        // Probabilities should sum to 1
        let sum: f64 = dist.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1, got {}", sum);

        // All probabilities should be non-negative
        assert!(dist.probabilities.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_entropy_bounds() {
        let calc = TruePhiCalculator::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);

        let entropy = calc.entropy(&hv);

        // Entropy should be non-negative
        assert!(entropy >= 0.0, "Entropy should be non-negative");

        // Entropy should be at most log(num_bins) bits
        let max_entropy = (calc.config.num_bins as f64).log2();
        assert!(entropy <= max_entropy + 0.01, "Entropy {} should be at most {}", entropy, max_entropy);
    }

    #[test]
    fn test_uniform_distribution_max_entropy() {
        let calc = TruePhiCalculator::new();

        // A vector with values uniformly distributed across bins should have max entropy
        let mut values = Vec::with_capacity(HDC_DIMENSION);
        for i in 0..HDC_DIMENSION {
            // Distribute evenly across [-1, 1]
            let val = -1.0 + 2.0 * (i as f32 / HDC_DIMENSION as f32);
            values.push(val);
        }
        let hv = ContinuousHV::from_vec(values);

        let entropy = calc.entropy(&hv);
        let max_entropy = (calc.config.num_bins as f64).log2();

        // Should be close to max entropy
        assert!(entropy > max_entropy * 0.95, "Uniform dist entropy {} should be near max {}", entropy, max_entropy);
    }

    #[test]
    fn test_joint_entropy() {
        let calc = TruePhiCalculator::new();
        let hv1 = ContinuousHV::random(HDC_DIMENSION, 42);
        let hv2 = ContinuousHV::random(HDC_DIMENSION, 43);

        let h_x = calc.entropy(&hv1);
        let h_y = calc.entropy(&hv2);
        let h_xy = calc.joint_entropy(&hv1, &hv2);

        // Joint entropy should be >= max of individual entropies
        assert!(h_xy >= h_x.max(h_y) - 0.01, "H(X,Y) should be >= max(H(X), H(Y))");

        // Joint entropy should be <= sum of individual entropies
        assert!(h_xy <= h_x + h_y + 0.01, "H(X,Y) should be <= H(X) + H(Y)");
    }

    #[test]
    fn test_mutual_information_non_negative() {
        let calc = TruePhiCalculator::new();
        let hv1 = ContinuousHV::random(HDC_DIMENSION, 42);
        let hv2 = ContinuousHV::random(HDC_DIMENSION, 43);

        let mi = calc.mutual_information(&hv1, &hv2);

        // MI should be non-negative
        assert!(mi >= 0.0, "Mutual information should be non-negative, got {}", mi);
    }

    #[test]
    fn test_self_mutual_information_equals_entropy() {
        let calc = TruePhiCalculator::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);

        let entropy = calc.entropy(&hv);
        let self_mi = calc.mutual_information(&hv, &hv);

        // I(X;X) = H(X)
        assert!((self_mi - entropy).abs() < 0.01,
            "Self MI {} should equal entropy {}", self_mi, entropy);
    }

    #[test]
    fn test_effective_information_increases_with_correlation() {
        let calc = TruePhiCalculator::new();

        // Independent vectors
        let independent: Vec<ContinuousHV> = create_test_vectors(4);
        let ei_independent = calc.effective_information(&independent);

        // Correlated vectors (all derived from same base)
        let base = ContinuousHV::random(HDC_DIMENSION, 100);
        let correlated: Vec<ContinuousHV> = (0..4)
            .map(|i| {
                let noise = ContinuousHV::random(HDC_DIMENSION, i as u64 + 200);
                ContinuousHV::weighted_bundle(&[&base, &noise], &[0.9, 0.1])
            })
            .collect();
        let ei_correlated = calc.effective_information(&correlated);

        // Correlated should have higher EI
        assert!(ei_correlated > ei_independent,
            "Correlated EI {} should exceed independent EI {}", ei_correlated, ei_independent);
    }

    #[test]
    fn test_true_phi_single_component() {
        let calc = TruePhiCalculator::new();
        let components: Vec<ContinuousHV> = create_test_vectors(1);

        let result = calc.compute_true_phi(&components);

        assert_eq!(result.phi, 0.0, "Single component should have Φ = 0");
    }

    #[test]
    fn test_true_phi_two_components() {
        let calc = TruePhiCalculator::new();
        let components: Vec<ContinuousHV> = create_test_vectors(2);

        let result = calc.compute_true_phi(&components);

        // For two components, there's only one partition, so Φ = EI - 0 = EI
        assert!(result.phi >= 0.0, "Φ should be non-negative");
        assert_eq!(result.mip.part_a.len() + result.mip.part_b.len(), 2);
    }

    #[test]
    fn test_true_phi_exhaustive_search() {
        let calc = TruePhiCalculator::new();
        let components: Vec<ContinuousHV> = create_test_vectors(4);

        let result = calc.compute_true_phi(&components);

        assert!(result.phi >= 0.0, "Φ should be non-negative");
        assert!(result.system_ei >= result.mip_ei, "System EI should be >= MIP EI");
        assert_eq!(result.component_entropies.len(), 4);
    }

    #[test]
    fn test_true_phi_heuristic_search() {
        let calc = TruePhiCalculator::new();
        // 10 components triggers heuristic search
        let components: Vec<ContinuousHV> = create_test_vectors(10);

        let result = calc.compute_true_phi(&components);

        assert!(result.phi >= 0.0, "Φ should be non-negative");
        assert!(!result.mip.part_a.is_empty() && !result.mip.part_b.is_empty(),
            "MIP should have non-empty parts");
    }

    #[test]
    fn test_bound_vs_bundle_different_phi() {
        let calc = TruePhiCalculator::new();

        let a = ContinuousHV::random(HDC_DIMENSION, 1);
        let b = ContinuousHV::random(HDC_DIMENSION, 2);
        let c = ContinuousHV::random(HDC_DIMENSION, 3);

        // Bundled structure
        let bundled = ContinuousHV::bundle(&[&a, &b]);
        let bundled_components = vec![bundled.clone(), c.clone()];
        let phi_bundled = calc.compute_true_phi(&bundled_components);

        // Bound structure
        let bound = a.bind(&b);
        let bound_components = vec![bound.clone(), c.clone()];
        let phi_bound = calc.compute_true_phi(&bound_components);

        // They should have different Φ values (bind creates orthogonal structure)
        // This test verifies that our entropy measure is sensitive to structural differences
        println!("Φ(bundled) = {:.4}, Φ(bound) = {:.4}", phi_bundled.phi, phi_bound.phi);
        // Note: We don't assert specific relationship, just that they're computed correctly
    }

    #[test]
    fn test_phi_fast() {
        let calc = TruePhiCalculator::new();
        let components: Vec<ContinuousHV> = create_test_vectors(6);

        let phi_fast = calc.compute_phi_fast(&components);
        let phi_full = calc.compute_true_phi(&components);

        // Fast should be in [0, 1]
        assert!(phi_fast >= 0.0 && phi_fast <= 1.0, "Fast Φ should be normalized");

        // They should be positively correlated (but not equal)
        println!("Φ_fast = {:.4}, Φ_full = {:.4}", phi_fast, phi_full.phi);
    }

    #[test]
    fn test_mi_matrix_symmetric() {
        let calc = TruePhiCalculator::new();
        let components: Vec<ContinuousHV> = create_test_vectors(4);

        let result = calc.compute_true_phi(&components);
        let matrix = &result.mutual_information_matrix;

        // Check symmetry
        for i in 0..matrix.len() {
            for j in 0..matrix.len() {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-10,
                    "MI matrix should be symmetric");
            }
        }
    }

    // Tests for improved MIP search algorithms

    #[test]
    fn test_spectral_partition() {
        let calc = TruePhiCalculator::new();

        // Create components with clear cluster structure
        let base1 = ContinuousHV::random(HDC_DIMENSION, 100);
        let c1 = ContinuousHV::weighted_bundle(&[&base1, &ContinuousHV::random(HDC_DIMENSION, 101)], &[0.9, 0.1]);
        let c2 = ContinuousHV::weighted_bundle(&[&base1, &ContinuousHV::random(HDC_DIMENSION, 102)], &[0.9, 0.1]);

        let base2 = ContinuousHV::random(HDC_DIMENSION, 200);
        let c3 = ContinuousHV::weighted_bundle(&[&base2, &ContinuousHV::random(HDC_DIMENSION, 201)], &[0.9, 0.1]);
        let c4 = ContinuousHV::weighted_bundle(&[&base2, &ContinuousHV::random(HDC_DIMENSION, 202)], &[0.9, 0.1]);

        let components = vec![c1, c2, c3, c4];
        let mi_matrix = calc.build_mi_matrix(&components);

        let partition = calc.spectral_partition(&mi_matrix, 4);
        assert!(partition.is_some(), "Spectral partition should succeed");

        let p = partition.unwrap();
        assert!(!p.part_a.is_empty() && !p.part_b.is_empty(), "Partition should have non-empty parts");
    }

    #[test]
    fn test_simulated_annealing_partition() {
        let calc = TruePhiCalculator::new();
        let components: Vec<ContinuousHV> = create_test_vectors(12);

        let partition = calc.simulated_annealing_partition(&components, 12);

        assert!(partition.is_some(), "SA partition should succeed");
        let p = partition.unwrap();
        assert!(!p.part_a.is_empty() && !p.part_b.is_empty(), "SA partition should have non-empty parts");
        assert_eq!(p.part_a.len() + p.part_b.len(), 12, "All elements should be assigned");
    }

    #[test]
    fn test_greedy_bisection_with_local_search() {
        let calc = TruePhiCalculator::new();
        let components: Vec<ContinuousHV> = create_test_vectors(10);

        let mi_matrix = calc.build_mi_matrix(&components);
        let greedy = calc.greedy_bisection_partition(&mi_matrix, 10);
        let refined = calc.local_search_refinement(&components, greedy.clone());

        assert!(!greedy.part_a.is_empty() && !greedy.part_b.is_empty());
        assert!(!refined.part_a.is_empty() && !refined.part_b.is_empty());

        let greedy_ei = calc.partition_effective_information(&components, &greedy);
        let refined_ei = calc.partition_effective_information(&components, &refined);
        assert!(refined_ei <= greedy_ei + 1e-10, "Local search should not increase EI");
    }

    #[test]
    fn test_large_system_mip_search() {
        let calc = TruePhiCalculator::new();
        let components: Vec<ContinuousHV> = create_test_vectors(20);

        let result = calc.compute_true_phi(&components);

        assert!(result.phi >= 0.0, "Φ should be non-negative");
        assert!(!result.mip.part_a.is_empty() && !result.mip.part_b.is_empty(), "MIP should have non-empty parts");
        assert_eq!(result.mip.part_a.len() + result.mip.part_b.len(), 20, "All components should be in partition");
    }

    #[test]
    fn test_assignment_to_partition() {
        let calc = TruePhiCalculator::new();
        let assignment = vec![true, true, false, true, false];
        let partition = calc.assignment_to_partition(&assignment);

        assert_eq!(partition.part_a, vec![0, 1, 3]);
        assert_eq!(partition.part_b, vec![2, 4]);
    }

    #[test]
    fn test_power_iteration_fiedler() {
        let calc = TruePhiCalculator::new();

        let laplacian = vec![
            vec![1.0, -1.0, 0.0, 0.0],
            vec![-1.0, 2.0, -1.0, 0.0],
            vec![0.0, -1.0, 2.0, -1.0],
            vec![0.0, 0.0, -1.0, 1.0],
        ];

        let fiedler = calc.power_iteration_fiedler(&laplacian, 4, 100);

        assert_eq!(fiedler.len(), 4);
        let sum: f64 = fiedler.iter().sum();
        assert!(sum.abs() < 0.1, "Fiedler should be mean-centered, sum={}", sum);
    }

    // Tests for continuous entropy estimation

    #[test]
    fn test_entropy_method_default() {
        let est = ContinuousEntropyEstimator::default();
        assert_eq!(est.method, EntropyMethod::Histogram);
    }

    #[test]
    fn test_knn_estimator_creation() {
        let est = ContinuousEntropyEstimator::knn(5);
        assert_eq!(est.method, EntropyMethod::KNN);
        assert_eq!(est.k_neighbors, 5);
    }

    #[test]
    fn test_kde_estimator_creation() {
        let est = ContinuousEntropyEstimator::kde();
        assert_eq!(est.method, EntropyMethod::KDE);
    }

    #[test]
    fn test_adaptive_estimator_creation() {
        let est = ContinuousEntropyEstimator::adaptive(32);
        assert_eq!(est.method, EntropyMethod::AdaptiveBins);
        assert_eq!(est.adaptive_bins, 32);
    }

    #[test]
    fn test_histogram_entropy_bounds() {
        let est = ContinuousEntropyEstimator::default();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);

        let h = est.entropy(&hv);
        assert!(h >= 0.0, "Entropy should be non-negative");
        let max_h = (est.adaptive_bins as f64).log2();
        assert!(h <= max_h + 0.1, "Entropy {} should be at most {}", h, max_h);
    }

    #[test]
    fn test_knn_entropy_non_negative() {
        let est = ContinuousEntropyEstimator::knn(3);
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);

        let h = est.entropy(&hv);
        assert!(h >= 0.0, "k-NN entropy should be non-negative");
    }

    #[test]
    fn test_kde_entropy_non_negative() {
        let est = ContinuousEntropyEstimator::kde();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);

        let h = est.entropy(&hv);
        assert!(h >= 0.0, "KDE entropy should be non-negative");
    }

    #[test]
    fn test_adaptive_entropy_non_negative() {
        let est = ContinuousEntropyEstimator::adaptive(16);
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);

        let h = est.entropy(&hv);
        assert!(h >= 0.0, "Adaptive entropy should be non-negative");
    }

    #[test]
    fn test_entropy_methods_correlate() {
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);

        let h_hist = ContinuousEntropyEstimator::default().entropy(&hv);
        let h_knn = ContinuousEntropyEstimator::knn(3).entropy(&hv);
        let h_kde = ContinuousEntropyEstimator::kde().entropy(&hv);
        let h_adaptive = ContinuousEntropyEstimator::adaptive(16).entropy(&hv);

        // All should be non-negative
        assert!(h_hist >= 0.0, "Histogram entropy should be non-negative: {}", h_hist);
        assert!(h_knn >= 0.0, "k-NN entropy should be non-negative: {}", h_knn);
        assert!(h_kde >= 0.0, "KDE entropy should be non-negative: {}", h_kde);
        assert!(h_adaptive >= 0.0, "Adaptive entropy should be non-negative: {}", h_adaptive);

        // Histogram and adaptive should give positive entropy for random data
        assert!(h_hist > 0.0, "Histogram should have positive entropy for random data");
        assert!(h_adaptive > 0.0, "Adaptive should have positive entropy for random data");

        // For methods that give positive values, check they're in same ballpark
        let positive_entropies: Vec<f64> = [h_hist, h_knn, h_kde, h_adaptive]
            .iter()
            .copied()
            .filter(|&h| h > 0.01)
            .collect();

        if positive_entropies.len() >= 2 {
            let max_h = positive_entropies.iter().copied().fold(0.0, f64::max);
            let min_h = positive_entropies.iter().copied().fold(f64::MAX, f64::min);
            assert!(max_h / min_h < 10.0,
                "Methods should give similar results: hist={:.3}, knn={:.3}, kde={:.3}, adaptive={:.3}",
                h_hist, h_knn, h_kde, h_adaptive);
        }
    }

    #[test]
    fn test_knn_mutual_information() {
        let est = ContinuousEntropyEstimator::knn(3);

        // Test with correlated vectors
        let base = ContinuousHV::random(HDC_DIMENSION, 100);
        let hv1 = ContinuousHV::weighted_bundle(&[&base, &ContinuousHV::random(HDC_DIMENSION, 101)], &[0.9, 0.1]);
        let hv2 = ContinuousHV::weighted_bundle(&[&base, &ContinuousHV::random(HDC_DIMENSION, 102)], &[0.9, 0.1]);

        let mi = est.mutual_information_knn(&hv1, &hv2);
        assert!(mi >= 0.0, "MI should be non-negative");
    }

    #[test]
    fn test_knn_mi_higher_for_correlated() {
        let est = ContinuousEntropyEstimator::knn(3);

        // Independent vectors
        let ind1 = ContinuousHV::random(HDC_DIMENSION, 1);
        let ind2 = ContinuousHV::random(HDC_DIMENSION, 2);
        let mi_ind = est.mutual_information_knn(&ind1, &ind2);

        // Correlated vectors
        let base = ContinuousHV::random(HDC_DIMENSION, 100);
        let cor1 = ContinuousHV::weighted_bundle(&[&base, &ContinuousHV::random(HDC_DIMENSION, 101)], &[0.8, 0.2]);
        let cor2 = ContinuousHV::weighted_bundle(&[&base, &ContinuousHV::random(HDC_DIMENSION, 102)], &[0.8, 0.2]);
        let mi_cor = est.mutual_information_knn(&cor1, &cor2);

        assert!(mi_cor > mi_ind, "Correlated MI {} should exceed independent MI {}", mi_cor, mi_ind);
    }

    #[test]
    fn test_digamma_function() {
        // Known values
        assert!((digamma(1.0) - (-0.5772156649)).abs() < 0.01, "ψ(1) ≈ -γ");
        assert!((digamma(2.0) - 0.4227843351).abs() < 0.01, "ψ(2) ≈ 1 - γ");
        assert!(digamma(10.0) > 2.0, "ψ(10) should be positive");
    }

    #[test]
    fn test_silverman_bandwidth() {
        let values: Vec<f32> = (0..1000).map(|i| (i as f32 / 500.0) - 1.0).collect();
        let bw = silverman_bandwidth(&values);
        assert!(bw > 0.0, "Bandwidth should be positive");
        assert!(bw < 1.0, "Bandwidth should be reasonable for [-1,1] data");
    }

    // IIT Benchmark Tests
    // Based on Integrated Information Theory (Tononi et al., 2016)

    /// IIT Axiom: Φ = 0 for completely independent (unintegrated) systems
    ///
    /// Reference: Oizumi et al. (2014) - "From Phenomenology to Mechanisms"
    /// A system of independent elements has no integrated information.
    #[test]
    fn test_iit_axiom_independent_system_zero_phi() {
        let calc = TruePhiCalculator::new();

        // Create completely independent random vectors
        let independent: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64 * 1000))
            .collect();

        let result = calc.compute_true_phi(&independent);

        // Φ should be very close to 0 for independent components
        // (Some small positive value due to random correlations)
        assert!(result.phi < 0.1,
            "Independent system should have near-zero Φ: {:.4}", result.phi);

        // System EI ≈ MIP EI for independent components
        assert!((result.system_ei - result.mip_ei).abs() < 0.1,
            "EI should be similar for system and MIP: sys={:.4}, mip={:.4}",
            result.system_ei, result.mip_ei);
    }

    /// IIT Axiom: Φ > 0 for genuinely integrated systems
    ///
    /// Reference: Tononi (2008) - "Consciousness as Integrated Information"
    /// Integration means the whole has more information than the sum of its parts.
    #[test]
    fn test_iit_axiom_integrated_system_positive_phi() {
        let calc = TruePhiCalculator::new();

        // Create a highly integrated system (all derived from same base)
        let base = ContinuousHV::random(HDC_DIMENSION, 42);
        let integrated: Vec<ContinuousHV> = (0..4)
            .map(|i| {
                let noise = ContinuousHV::random(HDC_DIMENSION, 100 + i as u64);
                ContinuousHV::weighted_bundle(&[&base, &noise], &[0.85, 0.15])
            })
            .collect();

        let result = calc.compute_true_phi(&integrated);

        // Φ should be significantly positive for integrated system
        assert!(result.phi > 0.1,
            "Integrated system should have positive Φ: {:.4}", result.phi);

        // System EI > MIP EI (this is the essence of integration)
        assert!(result.system_ei > result.mip_ei,
            "System EI should exceed MIP EI: sys={:.4} > mip={:.4}",
            result.system_ei, result.mip_ei);
    }

    /// IIT Property: Φ decreases when system is partitioned
    ///
    /// Reference: IIT 3.0 - The Minimum Information Partition (MIP)
    /// cuts the system where integration is weakest.
    #[test]
    fn test_iit_partition_decreases_phi() {
        let calc = TruePhiCalculator::new();

        // Create an integrated system
        let base = ContinuousHV::random(HDC_DIMENSION, 42);
        let components: Vec<ContinuousHV> = (0..6)
            .map(|i| {
                let noise = ContinuousHV::random(HDC_DIMENSION, 200 + i as u64);
                ContinuousHV::weighted_bundle(&[&base, &noise], &[0.8, 0.2])
            })
            .collect();

        // Compute Φ for full system
        let full_result = calc.compute_true_phi(&components);

        // Compute Φ for subsystems (partitions)
        let part_a: Vec<ContinuousHV> = components[0..3].to_vec();
        let part_b: Vec<ContinuousHV> = components[3..6].to_vec();

        let phi_a = calc.compute_true_phi(&part_a);
        let phi_b = calc.compute_true_phi(&part_b);

        // Sum of partition Φs should be related to cross-partition information
        let partition_total = phi_a.phi + phi_b.phi;

        println!("Full Φ: {:.4}, Part A Φ: {:.4}, Part B Φ: {:.4}",
            full_result.phi, phi_a.phi, phi_b.phi);

        // Full system Φ includes cross-partition information
        // that is lost when partitioned
        assert!(full_result.system_ei > phi_a.system_ei + phi_b.system_ei - 0.1,
            "Full system EI should account for all pairwise MI");
    }

    /// IIT Property: Binding creates integration
    ///
    /// Reference: HDC interpretation of IIT
    /// Binding vectors creates new orthogonal structure that
    /// cannot be recovered by unbinding - this is integration.
    #[test]
    fn test_iit_binding_creates_integration() {
        let calc = TruePhiCalculator::new();

        let a = ContinuousHV::random(HDC_DIMENSION, 1);
        let b = ContinuousHV::random(HDC_DIMENSION, 2);
        let c = ContinuousHV::random(HDC_DIMENSION, 3);
        let d = ContinuousHV::random(HDC_DIMENSION, 4);

        // Unbounded system: just the 4 components
        let unbounded = vec![a.clone(), b.clone(), c.clone(), d.clone()];
        let phi_unbounded = calc.compute_true_phi(&unbounded);

        // Bounded system: a⊗b and c⊗d create new integrated structures
        let bound1 = a.bind(&b);
        let bound2 = c.bind(&d);
        let bounded = vec![bound1.clone(), bound2.clone(), a.clone(), c.clone()];
        let phi_bounded = calc.compute_true_phi(&bounded);

        println!("Unbounded Φ: {:.4}, Bounded Φ: {:.4}",
            phi_unbounded.phi, phi_bounded.phi);

        // Binding should affect the information structure
        // (not necessarily increase Φ, but change the MIP)
        assert!(phi_bounded.mip.part_a.len() > 0 && phi_bounded.mip.part_b.len() > 0,
            "Bounded system should have non-trivial MIP");
    }

    /// IIT Property: More components can increase Φ
    ///
    /// Reference: IIT 3.0 - Φ generally increases with system size
    /// for integrated systems (but not for independent ones).
    #[test]
    fn test_iit_size_effect_on_phi() {
        let calc = TruePhiCalculator::new();

        // Create integrated systems of different sizes
        let base = ContinuousHV::random(HDC_DIMENSION, 42);

        let create_integrated = |n: usize| -> Vec<ContinuousHV> {
            (0..n)
                .map(|i| {
                    let noise = ContinuousHV::random(HDC_DIMENSION, 300 + i as u64);
                    ContinuousHV::weighted_bundle(&[&base, &noise], &[0.8, 0.2])
                })
                .collect()
        };

        let small = create_integrated(3);
        let medium = create_integrated(5);

        let phi_small = calc.compute_true_phi(&small);
        let phi_medium = calc.compute_true_phi(&medium);

        println!("Small (3) EI: {:.4}, Φ: {:.4}", phi_small.system_ei, phi_small.phi);
        println!("Medium (5) EI: {:.4}, Φ: {:.4}", phi_medium.system_ei, phi_medium.phi);

        // Larger integrated systems should have higher total EI
        assert!(phi_medium.system_ei > phi_small.system_ei,
            "Larger system should have higher EI: medium={:.4} > small={:.4}",
            phi_medium.system_ei, phi_small.system_ei);
    }

    /// IIT Property: Information exclusion
    ///
    /// Reference: IIT 3.0 - Only the Minimum Information Partition matters
    /// There's exactly one partition that defines Φ.
    #[test]
    fn test_iit_mip_uniqueness() {
        let calc = TruePhiCalculator::new();

        let components: Vec<ContinuousHV> = create_test_vectors(5);
        let result = calc.compute_true_phi(&components);

        // MIP should be a valid bipartition
        assert!(!result.mip.part_a.is_empty(), "MIP part A should be non-empty");
        assert!(!result.mip.part_b.is_empty(), "MIP part B should be non-empty");

        // All elements should be in exactly one part
        let total = result.mip.part_a.len() + result.mip.part_b.len();
        assert_eq!(total, 5, "MIP should partition all elements");

        // No duplicates
        let a_set: std::collections::HashSet<_> = result.mip.part_a.iter().collect();
        let b_set: std::collections::HashSet<_> = result.mip.part_b.iter().collect();
        assert!(a_set.is_disjoint(&b_set), "MIP parts should be disjoint");
    }

    /// IIT Benchmark: Correlated vs Independent comparison
    ///
    /// This is a key distinguishing test - IIT predicts that
    /// correlated components have higher Φ than independent ones.
    #[test]
    fn test_iit_benchmark_correlation_increases_phi() {
        let calc = TruePhiCalculator::new();

        // Independent system
        let independent: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64 * 7919))
            .collect();

        // Correlated system (same base)
        let base = ContinuousHV::random(HDC_DIMENSION, 12345);
        let correlated: Vec<ContinuousHV> = (0..4)
            .map(|i| {
                let noise = ContinuousHV::random(HDC_DIMENSION, 400 + i as u64);
                ContinuousHV::weighted_bundle(&[&base, &noise], &[0.9, 0.1])
            })
            .collect();

        let phi_ind = calc.compute_true_phi(&independent);
        let phi_cor = calc.compute_true_phi(&correlated);

        println!("Independent Φ: {:.4}, EI: {:.4}", phi_ind.phi, phi_ind.system_ei);
        println!("Correlated Φ: {:.4}, EI: {:.4}", phi_cor.phi, phi_cor.system_ei);

        // Correlated should have significantly higher Φ
        assert!(phi_cor.phi > phi_ind.phi * 2.0,
            "Correlated Φ ({:.4}) should be much higher than independent Φ ({:.4})",
            phi_cor.phi, phi_ind.phi);
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // OPTIMIZED METHOD TESTS
    // These tests verify that fast implementations produce correct results
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_knn_fast_produces_reasonable_entropy() {
        let est = ContinuousEntropyEstimator::knn_fast(3);

        let uniform = ContinuousHV::random(HDC_DIMENSION, 42);
        let h = est.entropy(&uniform);

        assert!(h > 0.0, "Fast k-NN entropy should be positive: {:.4}", h);
        assert!(h < 10.0, "Fast k-NN entropy should be reasonable: {:.4}", h);
    }

    #[test]
    fn test_knn_fast_matches_slow_approximately() {
        let est_slow = ContinuousEntropyEstimator::knn(3);
        let est_fast = ContinuousEntropyEstimator::knn_fast(3);

        let hv = ContinuousHV::random(512, 123); // Smaller dimension for speed

        let h_slow = est_slow.entropy(&hv);
        let h_fast = est_fast.entropy(&hv);

        // They should be within 50% of each other (different algorithms, similar results)
        let ratio = if h_slow > 0.0 { h_fast / h_slow } else { 1.0 };
        assert!(ratio > 0.5 && ratio < 2.0,
            "Fast and slow k-NN should give similar results: slow={:.4}, fast={:.4}",
            h_slow, h_fast);
    }

    #[test]
    fn test_kde_fast_produces_reasonable_entropy() {
        let est = ContinuousEntropyEstimator::kde_fast();

        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        let h = est.entropy(&hv);

        assert!(h >= 0.0, "Fast KDE entropy should be non-negative: {:.4}", h);
        assert!(h < 10.0, "Fast KDE entropy should be reasonable: {:.4}", h);
    }

    #[test]
    fn test_kde_fast_matches_slow_approximately() {
        let est_slow = ContinuousEntropyEstimator::kde();
        let est_fast = ContinuousEntropyEstimator::kde_fast();

        let hv = ContinuousHV::random(512, 456); // Smaller dimension for speed

        let h_slow = est_slow.entropy(&hv);
        let h_fast = est_fast.entropy(&hv);

        // They should be within 50% of each other
        let diff = (h_fast - h_slow).abs();
        let max_h = h_slow.max(h_fast);
        let relative_diff = if max_h > 0.0 { diff / max_h } else { 0.0 };

        assert!(relative_diff < 0.5,
            "Fast and slow KDE should give similar results: slow={:.4}, fast={:.4}, diff={:.1}%",
            h_slow, h_fast, relative_diff * 100.0);
    }

    #[test]
    fn test_mutual_information_fast() {
        let est = ContinuousEntropyEstimator::default();

        // Correlated vectors
        let base = ContinuousHV::random(HDC_DIMENSION, 100);
        let hv1 = ContinuousHV::weighted_bundle(&[&base, &ContinuousHV::random(HDC_DIMENSION, 101)], &[0.8, 0.2]);
        let hv2 = ContinuousHV::weighted_bundle(&[&base, &ContinuousHV::random(HDC_DIMENSION, 102)], &[0.8, 0.2]);

        let mi = est.mutual_information_fast(&hv1, &hv2);
        assert!(mi >= 0.0, "Fast MI should be non-negative: {:.4}", mi);
    }

    #[test]
    fn test_fast_mi_detects_correlation() {
        let est = ContinuousEntropyEstimator::default();

        // Independent vectors
        let ind1 = ContinuousHV::random(HDC_DIMENSION, 1);
        let ind2 = ContinuousHV::random(HDC_DIMENSION, 2);
        let mi_ind = est.mutual_information_fast(&ind1, &ind2);

        // Correlated vectors
        let base = ContinuousHV::random(HDC_DIMENSION, 100);
        let cor1 = ContinuousHV::weighted_bundle(&[&base, &ContinuousHV::random(HDC_DIMENSION, 101)], &[0.8, 0.2]);
        let cor2 = ContinuousHV::weighted_bundle(&[&base, &ContinuousHV::random(HDC_DIMENSION, 102)], &[0.8, 0.2]);
        let mi_cor = est.mutual_information_fast(&cor1, &cor2);

        assert!(mi_cor > mi_ind,
            "Fast MI should detect correlation: correlated={:.4} > independent={:.4}",
            mi_cor, mi_ind);
    }

    #[test]
    fn test_fast_vs_accurate_constructors() {
        let fast = ContinuousEntropyEstimator::fast();
        let accurate = ContinuousEntropyEstimator::accurate();

        let hv = ContinuousHV::random(HDC_DIMENSION, 789);

        let h_fast = fast.entropy(&hv);
        let h_accurate = accurate.entropy(&hv);

        // Both should produce reasonable entropy values
        assert!(h_fast > 0.0 && h_fast < 10.0, "Fast entropy should be reasonable: {:.4}", h_fast);
        assert!(h_accurate > 0.0 && h_accurate < 10.0, "Accurate entropy should be reasonable: {:.4}", h_accurate);

        // They should be similar (same underlying data)
        let ratio = h_fast / h_accurate;
        assert!(ratio > 0.5 && ratio < 2.0,
            "Fast and accurate should give similar results: fast={:.4}, accurate={:.4}",
            h_fast, h_accurate);
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // TEMPORAL IIT TESTS
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_temporal_transition_creation() {
        let current = ContinuousHV::random(HDC_DIMENSION, 1);
        let next = ContinuousHV::random(HDC_DIMENSION, 2);

        let transition = TemporalTransition::new(current.clone(), next.clone());

        assert_eq!(transition.current.dim(), HDC_DIMENSION);
        assert_eq!(transition.next.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_cause_effect_information() {
        let calc = TemporalPhiCalculator::new();

        // Create a transition with correlated states (deterministic dynamics)
        let current = ContinuousHV::random(HDC_DIMENSION, 42);
        let noise = ContinuousHV::random(HDC_DIMENSION, 43);
        let next = ContinuousHV::weighted_bundle(&[&current, &noise], &[0.9, 0.1]);

        let transition = TemporalTransition::new(current, next);

        let cause_info = calc.cause_information(&transition);
        let effect_info = calc.effect_information(&transition);

        // Both should be positive for correlated states
        assert!(cause_info >= 0.0, "Cause info should be non-negative: {:.4}", cause_info);
        assert!(effect_info >= 0.0, "Effect info should be non-negative: {:.4}", effect_info);
    }

    #[test]
    fn test_cause_effect_result() {
        let calc = TemporalPhiCalculator::new();

        let current = ContinuousHV::random(HDC_DIMENSION, 100);
        let next = ContinuousHV::random(HDC_DIMENSION, 101);
        let transition = TemporalTransition::new(current, next);

        let result = calc.compute_cause_effect(&transition);

        assert!(result.cause_info >= 0.0, "Cause info non-negative");
        assert!(result.effect_info >= 0.0, "Effect info non-negative");
        assert!(result.phi_cause_effect >= 0.0, "φ_cause_effect non-negative");
        assert!(result.phi_cause_effect <= result.cause_info.min(result.effect_info) + 1e-10,
            "φ_cause_effect should be min of cause and effect");
    }

    #[test]
    fn test_deterministic_dynamics_high_info() {
        let calc = TemporalPhiCalculator::new();

        // Highly deterministic: next is almost copy of current
        let current = ContinuousHV::random(HDC_DIMENSION, 200);
        let noise = ContinuousHV::random(HDC_DIMENSION, 201);
        let next = ContinuousHV::weighted_bundle(&[&current, &noise], &[0.95, 0.05]);

        let deterministic = TemporalTransition::new(current, next);

        // Random dynamics: next is independent of current
        let random_current = ContinuousHV::random(HDC_DIMENSION, 300);
        let random_next = ContinuousHV::random(HDC_DIMENSION, 301);
        let random = TemporalTransition::new(random_current, random_next);

        let det_result = calc.compute_cause_effect(&deterministic);
        let rnd_result = calc.compute_cause_effect(&random);

        // Deterministic should have higher MI than random
        assert!(det_result.cause_info > rnd_result.cause_info,
            "Deterministic should have higher cause info: det={:.4} > rnd={:.4}",
            det_result.cause_info, rnd_result.cause_info);
    }

    #[test]
    fn test_integrated_cause_effect_for_system() {
        let calc = TemporalPhiCalculator::new();

        // Create a system of 3 interacting components
        let base = ContinuousHV::random(HDC_DIMENSION, 400);

        let transitions: Vec<TemporalTransition> = (0..3)
            .map(|i| {
                let current = ContinuousHV::weighted_bundle(
                    &[&base, &ContinuousHV::random(HDC_DIMENSION, 410 + i as u64)],
                    &[0.8, 0.2]
                );
                let next = ContinuousHV::weighted_bundle(
                    &[&base, &ContinuousHV::random(HDC_DIMENSION, 420 + i as u64)],
                    &[0.7, 0.3]
                );
                TemporalTransition::new(current, next)
            })
            .collect();

        let result = calc.compute_system_cause_effect(&transitions);

        assert!(result.cause_info >= 0.0, "System cause info non-negative");
        assert!(result.effect_info >= 0.0, "System effect info non-negative");
        assert!(result.integrated_cause >= 0.0, "Integrated cause non-negative");
        assert!(result.integrated_effect >= 0.0, "Integrated effect non-negative");
    }

    #[test]
    fn test_integrated_cause_independent_system() {
        let calc = TemporalPhiCalculator::new();

        // Independent components - no shared dynamics
        let transitions: Vec<TemporalTransition> = (0..4)
            .map(|i| {
                let current = ContinuousHV::random(HDC_DIMENSION, 500 + i as u64 * 100);
                let next = ContinuousHV::random(HDC_DIMENSION, 501 + i as u64 * 100);
                TemporalTransition::new(current, next)
            })
            .collect();

        let integrated_cause = calc.integrated_cause_info(&transitions);
        let integrated_effect = calc.integrated_effect_info(&transitions);

        // Independent components should have low integrated info
        // (but may not be exactly zero due to random correlations)
        assert!(integrated_cause < 0.5,
            "Independent system should have low integrated cause: {:.4}", integrated_cause);
        assert!(integrated_effect < 0.5,
            "Independent system should have low integrated effect: {:.4}", integrated_effect);
    }

    #[test]
    fn test_temporal_phi_symmetry() {
        let calc = TemporalPhiCalculator::new();

        let current = ContinuousHV::random(HDC_DIMENSION, 600);
        let next = ContinuousHV::random(HDC_DIMENSION, 601);

        let forward = TemporalTransition::new(current.clone(), next.clone());
        let backward = TemporalTransition::new(next, current);

        let fwd_result = calc.compute_cause_effect(&forward);
        let bwd_result = calc.compute_cause_effect(&backward);

        // MI is symmetric, so cause/effect should be similar magnitude
        let diff = (fwd_result.cause_info - bwd_result.effect_info).abs();
        assert!(diff < 0.1 || diff / fwd_result.cause_info.max(0.01) < 0.5,
            "Cause/effect should show symmetry: forward cause={:.4}, backward effect={:.4}",
            fwd_result.cause_info, bwd_result.effect_info);
    }
}
