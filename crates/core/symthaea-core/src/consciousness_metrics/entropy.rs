// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Continuous entropy estimation using multiple methods.
//!
//! Provides histogram, k-NN, KDE, and adaptive binning entropy estimators
//! along with probability distribution types for HDC vectors.

use crate::hdc::unified_hv::ContinuousHV;

use super::EntropyMethod;

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

    pub(crate) fn log(&self, x: f64) -> f64 {
        if self.use_bits { x.log2() } else { x.ln() }
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        let points: Vec<(f64, f64)> = hv1
            .values
            .iter()
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
            joint_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        let mut indexed: Vec<(f32, usize)> =
            hv.values.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

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
pub(crate) fn digamma(mut x: f64) -> f64 {
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
    result += x.ln() - 0.5 / x - x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 / 252.0));

    result
}

/// Silverman's rule of thumb for KDE bandwidth
pub(crate) fn silverman_bandwidth(values: &[f32]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.1;
    }

    // Compute standard deviation
    let mean: f64 = values.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let variance: f64 = values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let std = variance.sqrt();

    // Compute IQR
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
    pub(crate) fn bin_index(value: f32, num_bins: usize) -> usize {
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

#[cfg(test)]
mod histogram_entropy_tests {
    //! Direct property tests for `entropy_histogram` (via the public
    //! `entropy()` API). Added as the correctness oracle for the Tier 2
    //! algorithm-search proof-of-concept (`symthaea-forge`) — this function
    //! previously had zero direct tests despite being the crate's default
    //! ("fast") entropy estimator, so mutations to it had nothing to fail
    //! against except unrelated downstream tests.
    use super::*;

    fn histogram_estimator() -> ContinuousEntropyEstimator {
        ContinuousEntropyEstimator::fast()
    }

    #[test]
    fn entropy_is_never_negative() {
        let est = histogram_estimator();
        for seed in 0..8u64 {
            let hv = ContinuousHV::random(256, seed);
            assert!(
                est.entropy(&hv) >= 0.0,
                "entropy must be non-negative (seed {seed})"
            );
        }
    }

    #[test]
    fn entropy_of_a_constant_vector_is_zero() {
        // All mass falls in a single bin -> p=1.0 for that bin -> H = -1*log2(1) = 0.
        let est = histogram_estimator();
        let hv = ContinuousHV::from_values(vec![0.3f32; 512]);
        let h = est.entropy(&hv);
        assert!(
            h.abs() < 1e-9,
            "entropy of a constant-valued vector should be exactly 0, got {h}"
        );
    }

    #[test]
    fn entropy_never_exceeds_log2_of_bin_count() {
        // Mathematical upper bound (maximum entropy) for any distribution
        // over `adaptive_bins` categories is log2(adaptive_bins) bits.
        let est = histogram_estimator();
        let max_possible = (est.adaptive_bins as f64).log2();
        for seed in 0..8u64 {
            let hv = ContinuousHV::random(1024, seed);
            let h = est.entropy(&hv);
            assert!(
                h <= max_possible + 1e-9,
                "entropy {h} exceeded the log2(bins)={max_possible} upper bound (seed {seed})"
            );
        }
    }

    #[test]
    fn entropy_of_an_exactly_uniform_histogram_hits_the_upper_bound() {
        // Construct a vector that places exactly one sample in each of the
        // 16 default bins (values spread evenly across [-1, 1]) -- this
        // should produce entropy == log2(16) == 4.0 bits exactly, giving a
        // tight (not just approximate) correctness check on the binning
        // math itself, not just an inequality.
        let est = histogram_estimator();
        let num_bins = est.adaptive_bins;
        let values: Vec<f32> = (0..num_bins)
            .map(|i| {
                // Bin i covers normalized range [i/num_bins, (i+1)/num_bins);
                // pick the bin midpoint and invert normalized = (v+1)/2.
                let normalized_mid = (i as f32 + 0.5) / num_bins as f32;
                normalized_mid * 2.0 - 1.0
            })
            .collect();
        let hv = ContinuousHV::from_values(values);
        let h = est.entropy(&hv);
        let expected = (num_bins as f64).log2();
        assert!(
            (h - expected).abs() < 1e-9,
            "expected exactly uniform histogram to hit {expected} bits, got {h}"
        );
    }
}
