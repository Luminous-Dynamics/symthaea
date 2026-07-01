// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SIMD-Accelerated Histogram Binning
//!
//! Fast binning using vectorized operations when available.

/// SIMD-optimized histogram binning
///
/// Uses vectorized operations for fast binning of continuous values.
/// Falls back to scalar implementation when SIMD is not available.
#[derive(Debug, Clone)]
pub struct SimdHistogramBinner {
    num_bins: usize,
}

impl SimdHistogramBinner {
    /// Create a new binner with specified number of bins
    pub fn new(num_bins: usize) -> Self {
        Self { num_bins }
    }

    /// Compute histogram counts using optimized scalar operations
    ///
    /// This version uses loop unrolling for better performance.
    pub fn compute_histogram(&self, values: &[f32]) -> Vec<usize> {
        let mut counts = vec![0usize; self.num_bins];
        let num_bins_f = self.num_bins as f32;

        // Process 4 values at a time (manual unrolling)
        let chunks = values.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;

            let v0 = ((values[idx] + 1.0) * 0.5).clamp(0.0, 0.9999);
            let v1 = ((values[idx + 1] + 1.0) * 0.5).clamp(0.0, 0.9999);
            let v2 = ((values[idx + 2] + 1.0) * 0.5).clamp(0.0, 0.9999);
            let v3 = ((values[idx + 3] + 1.0) * 0.5).clamp(0.0, 0.9999);

            let b0 = (v0 * num_bins_f) as usize;
            let b1 = (v1 * num_bins_f) as usize;
            let b2 = (v2 * num_bins_f) as usize;
            let b3 = (v3 * num_bins_f) as usize;

            counts[b0] += 1;
            counts[b1] += 1;
            counts[b2] += 1;
            counts[b3] += 1;
        }

        // Handle remaining values
        for i in (chunks * 4)..values.len() {
            let v = ((values[i] + 1.0) * 0.5).clamp(0.0, 0.9999);
            let bin = (v * num_bins_f) as usize;
            counts[bin] += 1;
        }

        counts
    }

    /// Compute 2D histogram for joint distribution
    pub fn compute_joint_histogram(&self, values1: &[f32], values2: &[f32]) -> Vec<Vec<usize>> {
        let n = values1.len().min(values2.len());
        let mut counts = vec![vec![0usize; self.num_bins]; self.num_bins];
        let num_bins_f = self.num_bins as f32;

        for i in 0..n {
            let v1 = ((values1[i] + 1.0) * 0.5).clamp(0.0, 0.9999);
            let v2 = ((values2[i] + 1.0) * 0.5).clamp(0.0, 0.9999);
            let b1 = (v1 * num_bins_f) as usize;
            let b2 = (v2 * num_bins_f) as usize;
            counts[b1][b2] += 1;
        }

        counts
    }

    /// Compute entropy from histogram
    pub fn entropy_from_histogram(&self, counts: &[usize], use_bits: bool) -> f64 {
        let total: usize = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let total_f = total as f64;
        let mut h = 0.0;

        for &c in counts {
            if c > 0 {
                let p = c as f64 / total_f;
                h -= p * if use_bits { p.log2() } else { p.ln() };
            }
        }

        h
    }

    /// Compute entropy directly from values
    pub fn entropy(&self, values: &[f32], use_bits: bool) -> f64 {
        let counts = self.compute_histogram(values);
        self.entropy_from_histogram(&counts, use_bits)
    }

    /// Compute mutual information from joint histogram
    pub fn mutual_information_from_histograms(
        &self,
        joint: &[Vec<usize>],
        marginal1: &[usize],
        marginal2: &[usize],
        use_bits: bool,
    ) -> f64 {
        let total: usize = marginal1.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let total_f = total as f64;
        let mut mi = 0.0;

        for i in 0..self.num_bins {
            if marginal1[i] == 0 {
                continue;
            }
            let p_x = marginal1[i] as f64 / total_f;

            for j in 0..self.num_bins {
                if joint[i][j] == 0 {
                    continue;
                }
                let p_y = marginal2[j] as f64 / total_f;
                let p_xy = joint[i][j] as f64 / total_f;

                if p_x > 0.0 && p_y > 0.0 {
                    let log_term = if use_bits {
                        (p_xy / (p_x * p_y)).log2()
                    } else {
                        (p_xy / (p_x * p_y)).ln()
                    };
                    mi += p_xy * log_term;
                }
            }
        }

        mi.max(0.0)
    }
}
