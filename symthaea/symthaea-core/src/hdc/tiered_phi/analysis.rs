// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Φ Multi-Scale and Entropy Analysis
//!
//! This module contains multi-scale pyramid analysis (#94) and entropy/complexity analysis (#95).
//!
//! ## Features
//!
//! - **Multi-Scale Pyramid**: Compute Φ at multiple spatial scales to find optimal consciousness granularity
//! - **Entropy Analysis**: Measure complexity, predictability, and information richness of consciousness states

use super::core::{ApproximationTier, TieredPhi};
use crate::hdc::binary_hv::BinaryHV;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct PhiPyramidConfig {
    /// Minimum components per scale (below this, don't compute)
    pub min_components_per_scale: usize,

    /// Maximum number of scales to compute
    pub max_scales: usize,

    /// Scale factor (components per level grows by this factor)
    pub scale_factor: usize,

    /// Whether to compute all scales in parallel
    pub parallel_scales: bool,

    /// Tier to use for Φ computation at each scale
    pub phi_tier: ApproximationTier,
}

impl Default for PhiPyramidConfig {
    fn default() -> Self {
        Self {
            min_components_per_scale: 2,
            max_scales: 8,
            scale_factor: 2, // Each level has 2x more components
            parallel_scales: true,
            phi_tier: ApproximationTier::SampledPartition,
        }
    }
}

impl PhiPyramidConfig {
    /// Fast config (fewer scales, heuristic tier)
    pub fn fast() -> Self {
        Self {
            max_scales: 4,
            parallel_scales: true,
            phi_tier: ApproximationTier::SampledPartition,
            ..Default::default()
        }
    }

    /// Research config (more scales, exact tier for small)
    pub fn research() -> Self {
        Self {
            max_scales: 12,
            parallel_scales: true,
            phi_tier: ApproximationTier::ExhaustivePartition,
            ..Default::default()
        }
    }
}

/// Result of multi-scale Φ computation
#[derive(Debug, Clone)]
pub struct PhiPyramidResult {
    /// Φ values at each scale (index 0 = most local)
    pub phi_by_scale: Vec<f64>,

    /// Number of components at each scale
    pub components_per_scale: Vec<usize>,

    /// Number of clusters/groups at each scale
    pub clusters_per_scale: Vec<usize>,

    /// Index of scale with peak Φ (optimal consciousness resolution)
    pub peak_scale: usize,

    /// Maximum Φ value across all scales
    pub peak_phi: f64,

    /// Ratio of peak Φ to global Φ (>1 means local dominates)
    pub locality_ratio: f64,

    /// Standard deviation of Φ across scales (high = scale-dependent)
    pub scale_variance: f64,

    /// Whether consciousness is hierarchical (multiple peaks) or flat (single peak)
    pub is_hierarchical: bool,

    /// Computation time in milliseconds
    pub computation_time_ms: f64,
}

impl PhiPyramidResult {
    /// Get the optimal scale descriptor
    pub fn optimal_scale_description(&self) -> &'static str {
        if self.phi_by_scale.is_empty() {
            return "unknown";
        }

        let n_scales = self.phi_by_scale.len();
        let relative_pos = self.peak_scale as f64 / n_scales as f64;

        if relative_pos < 0.25 {
            "local"
        } else if relative_pos < 0.5 {
            "micro"
        } else if relative_pos < 0.75 {
            "meso"
        } else {
            "global"
        }
    }

    /// Check if consciousness is primarily local (small-scale dominant)
    pub fn is_local_dominant(&self) -> bool {
        self.locality_ratio > 1.2
    }

    /// Check if consciousness is primarily global (large-scale dominant)
    pub fn is_global_dominant(&self) -> bool {
        self.locality_ratio < 0.8
    }

    /// Get the scale gradient (how Φ changes across scales)
    pub fn scale_gradient(&self) -> Vec<f64> {
        if self.phi_by_scale.len() < 2 {
            return vec![];
        }

        self.phi_by_scale.windows(2).map(|w| w[1] - w[0]).collect()
    }
}

/// Multi-Scale Φ Pyramid Calculator
///
/// Computes integrated information at multiple spatial scales to discover
/// the optimal granularity of consciousness and detect scale-dependent transitions.
#[derive(Debug, Clone)]
pub struct PhiPyramid {
    config: PhiPyramidConfig,
    phi_calculator: TieredPhi,
}

impl PhiPyramid {
    /// Create new pyramid with default config
    pub fn new() -> Self {
        let config = PhiPyramidConfig::default();
        Self {
            phi_calculator: TieredPhi::new(config.phi_tier),
            config,
        }
    }

    /// Create with custom config
    pub fn with_config(config: PhiPyramidConfig) -> Self {
        Self {
            phi_calculator: TieredPhi::new(config.phi_tier),
            config,
        }
    }

    /// Fast pyramid for real-time monitoring
    pub fn fast() -> Self {
        Self::with_config(PhiPyramidConfig::fast())
    }

    /// Research pyramid for detailed analysis
    pub fn research() -> Self {
        Self::with_config(PhiPyramidConfig::research())
    }

    /// Compute Φ across all scales
    ///
    /// # Algorithm
    ///
    /// 1. Determine scale levels based on component count
    /// 2. For each scale:
    ///    - Partition components into clusters of appropriate size
    ///    - Compute Φ for each cluster
    ///    - Average Φ across clusters at that scale
    /// 3. Find peak scale and compute statistics
    ///
    /// # Arguments
    ///
    /// * `components` - The full set of consciousness components
    ///
    /// # Returns
    ///
    /// PhiPyramidResult with Φ at each scale and analysis
    pub fn compute(&mut self, components: &[BinaryHV]) -> PhiPyramidResult {
        let start_time = Instant::now();
        let n = components.len();

        // Edge cases
        if n < self.config.min_components_per_scale {
            return PhiPyramidResult {
                phi_by_scale: vec![],
                components_per_scale: vec![],
                clusters_per_scale: vec![],
                peak_scale: 0,
                peak_phi: 0.0,
                locality_ratio: 1.0,
                scale_variance: 0.0,
                is_hierarchical: false,
                computation_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            };
        }

        // Step 1: Determine scales
        // Scale k has components_per_cluster = min_components * scale_factor^k
        // Continue until cluster size >= n (global level)
        let mut scales: Vec<usize> = vec![]; // Components per cluster at each scale
        let mut size = self.config.min_components_per_scale;

        // Reserve one slot for global scale if needed
        let max_intermediate_scales = if n > self.config.min_components_per_scale {
            self.config.max_scales.saturating_sub(1)
        } else {
            self.config.max_scales
        };

        while size < n && scales.len() < max_intermediate_scales {
            scales.push(size);
            size *= self.config.scale_factor;
        }

        // Always include global scale (all components) if within max_scales
        if scales.len() < self.config.max_scales {
            scales.push(n);
        }

        // Step 2: Compute Φ at each scale
        let scale_results: Vec<(f64, usize, usize)> = {
            #[cfg(feature = "parallel")]
            {
                if self.config.parallel_scales && scales.len() > 2 {
                    scales
                        .par_iter()
                        .map(|&cluster_size| self.compute_scale(components, cluster_size))
                        .collect()
                } else {
                    scales
                        .iter()
                        .map(|&cluster_size| self.compute_scale(components, cluster_size))
                        .collect()
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                scales
                    .iter()
                    .map(|&cluster_size| self.compute_scale(components, cluster_size))
                    .collect()
            }
        };

        // Extract results
        let phi_by_scale: Vec<f64> = scale_results.iter().map(|r| r.0).collect();
        let components_per_scale: Vec<usize> = scale_results.iter().map(|r| r.1).collect();
        let clusters_per_scale: Vec<usize> = scale_results.iter().map(|r| r.2).collect();

        // Step 3: Find peak
        let (peak_scale, peak_phi) = phi_by_scale
            .iter()
            .enumerate()
            .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| a.total_cmp(b))
            .map(|(i, &phi)| (i, phi))
            .unwrap_or((0, 0.0));

        // Step 4: Compute statistics
        let global_phi = phi_by_scale.last().copied().unwrap_or(0.0);
        let locality_ratio = if global_phi > 1e-10 {
            peak_phi / global_phi
        } else {
            1.0
        };

        let scale_variance = self.compute_variance(&phi_by_scale);

        // Detect hierarchy: multiple local maxima
        let is_hierarchical = self.detect_hierarchy(&phi_by_scale);

        PhiPyramidResult {
            phi_by_scale,
            components_per_scale,
            clusters_per_scale,
            peak_scale,
            peak_phi,
            locality_ratio,
            scale_variance,
            is_hierarchical,
            computation_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Compute Φ at a specific scale
    /// Returns (average_phi, components_per_cluster, num_clusters)
    fn compute_scale(&self, components: &[BinaryHV], cluster_size: usize) -> (f64, usize, usize) {
        let n = components.len();

        if cluster_size >= n {
            // Global scale: compute Φ on all components
            let mut calc = TieredPhi::new(self.config.phi_tier);
            let phi = calc.compute(components);
            return (phi, n, 1);
        }

        // Partition into clusters and compute average Φ
        let num_clusters = n.div_ceil(cluster_size); // Ceiling division
        let mut total_phi = 0.0;
        let mut valid_clusters = 0;

        for cluster_idx in 0..num_clusters {
            let start = cluster_idx * cluster_size;
            let end = (start + cluster_size).min(n);

            if end - start >= self.config.min_components_per_scale {
                let cluster = &components[start..end];
                let mut calc = TieredPhi::new(self.config.phi_tier);
                let phi = calc.compute(cluster);
                total_phi += phi;
                valid_clusters += 1;
            }
        }

        let avg_phi = if valid_clusters > 0 {
            total_phi / valid_clusters as f64
        } else {
            0.0
        };

        (avg_phi, cluster_size, num_clusters)
    }

    /// Compute variance of Φ values
    fn compute_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance.sqrt() // Return std dev for interpretability
    }

    /// Detect hierarchical structure (multiple local maxima)
    fn detect_hierarchy(&self, phi_values: &[f64]) -> bool {
        if phi_values.len() < 3 {
            return false;
        }

        // Count local maxima
        let mut local_maxima = 0;
        for i in 1..phi_values.len() - 1 {
            if phi_values[i] > phi_values[i - 1] && phi_values[i] > phi_values[i + 1] {
                local_maxima += 1;
            }
        }

        // Also check endpoints
        if phi_values.len() >= 2 {
            if phi_values[0] > phi_values[1] {
                local_maxima += 1;
            }
            if phi_values[phi_values.len() - 1] > phi_values[phi_values.len() - 2] {
                local_maxima += 1;
            }
        }

        // Hierarchical if more than one peak
        local_maxima >= 2
    }
}

impl Default for PhiPyramid {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: compute multi-scale Φ
pub fn multi_scale_phi(components: &[BinaryHV]) -> PhiPyramidResult {
    PhiPyramid::new().compute(components)
}

/// Convenience function: find optimal consciousness scale
pub fn optimal_scale(components: &[BinaryHV]) -> (usize, f64) {
    let result = PhiPyramid::new().compute(components);
    (result.peak_scale, result.peak_phi)
}

// ============================================================================
// Revolutionary Improvement #95: Φ Entropy & Complexity Analysis
// ============================================================================
//
// Measure the complexity, predictability, and information richness of
// consciousness states using entropy-based metrics.
//
// ## Core Insight
//
// Consciousness isn't just about integration (Φ) - it's also about complexity.
// A highly integrated but simple system may have high Φ but low complexity.
// True consciousness likely requires BOTH integration AND complexity.
//
// ## Metrics Implemented
//
// 1. **Shannon Entropy**: Information content of Φ distribution
// 2. **Sample Entropy**: Regularity/predictability of Φ time series
// 3. **Lempel-Ziv Complexity**: Algorithmic complexity approximation
// 4. **Multi-Scale Entropy**: Complexity at different time scales
// 5. **Integrated Complexity**: Φ × Complexity product (novel metric!)
//
// ## Scientific Foundation
//
// - Tononi (2004): Consciousness requires integrated information
// - Casali (2013): PCI (Perturbational Complexity Index) for consciousness
// - Costa (2005): Multi-Scale Entropy for physiological time series
// - This work: Novel Φ-Complexity integration for consciousness measurement

/// Configuration for entropy and complexity analysis
#[derive(Debug, Clone)]
pub struct PhiEntropyConfig {
    /// Number of bins for histogram-based entropy
    pub num_bins: usize,

    /// Embedding dimension for sample entropy
    pub embed_dim: usize,

    /// Tolerance radius for sample entropy (fraction of std)
    pub tolerance_fraction: f64,

    /// Maximum scale for multi-scale entropy
    pub max_scale: usize,

    /// Minimum samples required for reliable entropy estimation
    pub min_samples: usize,
}

impl Default for PhiEntropyConfig {
    fn default() -> Self {
        Self {
            num_bins: 10,
            embed_dim: 2,
            tolerance_fraction: 0.2,
            max_scale: 10,
            min_samples: 50,
        }
    }
}

impl PhiEntropyConfig {
    /// Fast config for real-time analysis
    pub fn fast() -> Self {
        Self {
            num_bins: 5,
            embed_dim: 2,
            max_scale: 3,
            min_samples: 20,
            ..Default::default()
        }
    }

    /// Research config for detailed analysis
    pub fn research() -> Self {
        Self {
            num_bins: 20,
            embed_dim: 3,
            max_scale: 20,
            min_samples: 100,
            ..Default::default()
        }
    }
}

/// Result of entropy and complexity analysis
#[derive(Debug, Clone)]
pub struct PhiEntropyResult {
    /// Shannon entropy of Φ distribution (bits)
    pub shannon_entropy: f64,

    /// Normalized Shannon entropy (0 = deterministic, 1 = uniform)
    pub normalized_entropy: f64,

    /// Sample entropy (regularity measure)
    /// Low = regular/predictable, High = complex/unpredictable
    pub sample_entropy: f64,

    /// Lempel-Ziv complexity (algorithmic complexity approximation)
    pub lz_complexity: f64,

    /// Normalized LZ complexity (0 = simple, 1 = random)
    pub normalized_lz: f64,

    /// Multi-scale entropy values (if computed)
    pub multi_scale_entropy: Vec<f64>,

    /// Complexity index (geometric mean of entropy measures)
    pub complexity_index: f64,

    /// Integrated complexity: Φ_mean × Complexity_index
    /// This is a novel metric combining integration and complexity
    pub integrated_complexity: f64,

    /// Predictability score (1 - normalized_entropy)
    pub predictability: f64,

    /// Number of samples analyzed
    pub sample_count: usize,
}

impl PhiEntropyResult {
    /// Check if consciousness is complex (high entropy + high Φ)
    pub fn is_complex(&self) -> bool {
        self.complexity_index > 0.5 && self.integrated_complexity > 0.3
    }

    /// Check if consciousness is simple/predictable
    pub fn is_predictable(&self) -> bool {
        self.predictability > 0.7
    }

    /// Check if consciousness is chaotic (high entropy, low Φ)
    pub fn is_chaotic(&self) -> bool {
        self.normalized_entropy > 0.8 && self.integrated_complexity < 0.2
    }

    /// Consciousness quality descriptor
    pub fn quality_description(&self) -> &'static str {
        if self.integrated_complexity > 0.6 {
            "rich" // High integration + high complexity
        } else if self.integrated_complexity > 0.4 {
            "moderate"
        } else if self.normalized_entropy > 0.7 {
            "chaotic" // High complexity but low integration
        } else if self.predictability > 0.7 {
            "simple" // Low complexity, predictable
        } else {
            "transitional"
        }
    }
}

/// Φ Entropy and Complexity Analyzer
///
/// Computes entropy-based complexity measures for consciousness analysis.
/// Combines with Φ to produce novel "integrated complexity" metric.
#[derive(Debug, Clone)]
pub struct PhiEntropyAnalyzer {
    config: PhiEntropyConfig,
}

impl PhiEntropyAnalyzer {
    /// Create analyzer with default config
    pub fn new() -> Self {
        Self {
            config: PhiEntropyConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: PhiEntropyConfig) -> Self {
        Self { config }
    }

    /// Fast analyzer for real-time monitoring
    pub fn fast() -> Self {
        Self::with_config(PhiEntropyConfig::fast())
    }

    /// Research analyzer for detailed analysis
    pub fn research() -> Self {
        Self::with_config(PhiEntropyConfig::research())
    }

    /// Compute entropy and complexity metrics for Φ time series
    ///
    /// # Arguments
    ///
    /// * `phi_values` - Time series of Φ measurements
    /// * `mean_phi` - Optional mean Φ for integrated complexity
    ///
    /// # Returns
    ///
    /// PhiEntropyResult with all complexity metrics
    pub fn analyze(&self, phi_values: &[f64], mean_phi: Option<f64>) -> PhiEntropyResult {
        let n = phi_values.len();

        // Edge case: insufficient samples
        if n < self.config.min_samples {
            return PhiEntropyResult {
                shannon_entropy: 0.0,
                normalized_entropy: 0.0,
                sample_entropy: 0.0,
                lz_complexity: 0.0,
                normalized_lz: 0.0,
                multi_scale_entropy: vec![],
                complexity_index: 0.0,
                integrated_complexity: 0.0,
                predictability: 1.0,
                sample_count: n,
            };
        }

        // 1. Shannon Entropy (histogram-based)
        let (shannon, normalized_shannon) = self.compute_shannon_entropy(phi_values);

        // 2. Sample Entropy (regularity measure)
        let sample_ent = self.compute_sample_entropy(phi_values);

        // 3. Lempel-Ziv Complexity
        let (lz, normalized_lz) = self.compute_lz_complexity(phi_values);

        // 4. Multi-Scale Entropy (optional, more expensive)
        let mse = if n >= self.config.min_samples * 2 {
            self.compute_multi_scale_entropy(phi_values)
        } else {
            vec![]
        };

        // 5. Complexity Index (geometric mean of entropy measures)
        let complexity_index = (normalized_shannon * sample_ent.max(0.01) * normalized_lz)
            .powf(1.0 / 3.0)
            .clamp(0.0, 1.0);

        // 6. Integrated Complexity (novel metric)
        let phi_mean = mean_phi.unwrap_or_else(|| phi_values.iter().sum::<f64>() / n as f64);
        let integrated_complexity = phi_mean * complexity_index;

        // 7. Predictability
        let predictability = 1.0 - normalized_shannon;

        PhiEntropyResult {
            shannon_entropy: shannon,
            normalized_entropy: normalized_shannon,
            sample_entropy: sample_ent,
            lz_complexity: lz,
            normalized_lz,
            multi_scale_entropy: mse,
            complexity_index,
            integrated_complexity,
            predictability,
            sample_count: n,
        }
    }

    /// Compute Shannon entropy using histogram
    fn compute_shannon_entropy(&self, values: &[f64]) -> (f64, f64) {
        let n = values.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        // Find min/max for binning
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Handle constant values
        if (max_val - min_val).abs() < 1e-10 {
            return (0.0, 0.0); // Zero entropy for constant signal
        }

        // Build histogram
        let bin_width = (max_val - min_val) / self.config.num_bins as f64;
        let mut counts = vec![0usize; self.config.num_bins];

        for &v in values {
            let bin = ((v - min_val) / bin_width).floor() as usize;
            let bin = bin.min(self.config.num_bins - 1);
            counts[bin] += 1;
        }

        // Compute Shannon entropy: H = -Σ p_i × log2(p_i)
        let n_f64 = n as f64;
        let entropy: f64 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n_f64;
                -p * p.log2()
            })
            .sum();

        // Normalized entropy (0 = deterministic, 1 = uniform)
        let max_entropy = (self.config.num_bins as f64).log2();
        let normalized = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        (entropy, normalized.clamp(0.0, 1.0))
    }

    /// Compute Sample Entropy (SampEn)
    ///
    /// Measures regularity/predictability of time series.
    /// Low SampEn = regular, predictable; High SampEn = complex, unpredictable
    fn compute_sample_entropy(&self, values: &[f64]) -> f64 {
        let n = values.len();
        let m = self.config.embed_dim;

        if n <= m + 1 {
            return 0.0;
        }

        // Compute standard deviation for tolerance
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0; // Constant signal
        }

        let r = self.config.tolerance_fraction * std;

        // Count template matches for dimension m and m+1
        let count_m = self.count_template_matches(values, m, r);
        let count_m1 = self.count_template_matches(values, m + 1, r);

        // Sample entropy = -ln(count_m+1 / count_m)
        if count_m == 0 || count_m1 == 0 {
            return 0.0;
        }

        let ratio = count_m1 as f64 / count_m as f64;
        if ratio > 0.0 { -ratio.ln() } else { 0.0 }
    }

    /// Count template matches for sample entropy
    fn count_template_matches(&self, values: &[f64], m: usize, r: f64) -> usize {
        let n = values.len();
        if n <= m {
            return 0;
        }

        let mut count = 0;
        let templates = n - m;

        for i in 0..templates {
            for j in (i + 1)..templates {
                // Check if templates match within tolerance
                let mut matches = true;
                for k in 0..m {
                    if (values[i + k] - values[j + k]).abs() > r {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    count += 1;
                }
            }
        }

        count
    }

    /// Compute Lempel-Ziv complexity (simplified version)
    ///
    /// Approximates Kolmogorov complexity by counting distinct patterns.
    fn compute_lz_complexity(&self, values: &[f64]) -> (f64, f64) {
        let n = values.len();
        if n < 2 {
            return (0.0, 0.0);
        }

        // Convert to binary string based on median
        let median = self.compute_median(values);
        let binary: Vec<u8> = values
            .iter()
            .map(|&v| if v >= median { 1 } else { 0 })
            .collect();

        // Count distinct patterns (simplified LZ76)
        let mut patterns = std::collections::HashSet::new();
        let mut i = 0;
        let mut len = 1;

        while i + len <= n {
            let pattern = &binary[i..i + len];
            if patterns.contains(pattern) {
                len += 1;
            } else {
                patterns.insert(pattern.to_vec());
                i += len;
                len = 1;
            }
        }

        let complexity = patterns.len() as f64;

        // Normalized by theoretical maximum (log2(n) patterns for random sequence)
        let max_complexity = (n as f64).log2() * 2.0;
        let normalized = if max_complexity > 0.0 {
            complexity / max_complexity
        } else {
            0.0
        };

        (complexity, normalized.clamp(0.0, 1.0))
    }

    /// Compute multi-scale entropy
    fn compute_multi_scale_entropy(&self, values: &[f64]) -> Vec<f64> {
        let mut mse = Vec::with_capacity(self.config.max_scale);

        for scale in 1..=self.config.max_scale {
            let coarse = self.coarse_grain(values, scale);
            if coarse.len() >= self.config.min_samples {
                let se = self.compute_sample_entropy(&coarse);
                mse.push(se);
            } else {
                break; // Not enough data for higher scales
            }
        }

        mse
    }

    /// Coarse-grain time series by averaging windows
    fn coarse_grain(&self, values: &[f64], scale: usize) -> Vec<f64> {
        if scale == 1 {
            return values.to_vec();
        }

        let n = values.len();
        let new_len = n / scale;
        let mut coarse = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let start = i * scale;
            let end = start + scale;
            let avg = values[start..end].iter().sum::<f64>() / scale as f64;
            coarse.push(avg);
        }

        coarse
    }

    /// Compute median value
    fn compute_median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let n = sorted.len();
        if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }
}

impl Default for PhiEntropyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: analyze Φ complexity
pub fn analyze_phi_complexity(phi_values: &[f64]) -> PhiEntropyResult {
    PhiEntropyAnalyzer::new().analyze(phi_values, None)
}

/// Convenience function: compute integrated complexity (Φ × Complexity)
pub fn integrated_complexity(phi_values: &[f64], mean_phi: f64) -> f64 {
    let result = PhiEntropyAnalyzer::new().analyze(phi_values, Some(mean_phi));
    result.integrated_complexity
}
