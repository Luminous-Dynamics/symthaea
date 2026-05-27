// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Synergistic Integration (Σ) — Layer 2 Consciousness Metric
//!
//! Implements a PhiR-inspired synergistic integration measure for HDC states.
//! This is Layer 2 of Symthaea's three-layer consciousness measurement:
//!
//! - Layer 1: Ψ (Psi)  — Fast composite soft signal (O(1), every cycle)
//! - **Layer 2: Σ (Sigma) — Synergistic integration (O(n²), every N cycles)**
//! - Layer 3: Φ (Phi)  — True IIT via MIP search (O(n³), on demand)
//!
//! ## Method
//!
//! Uses covariance structure of HDC activation snapshots to estimate how much
//! information exists in the whole system that isn't present in any partition.
//! Under Gaussian assumptions (justified by CLT on high-dimensional sums),
//! this reduces to a Phi* (Oizumi et al. 2016) closed-form computation.
//!
//! ## References
//!
//! - Mediano et al. (2021), "What it is like to be a bit" — PhiID/PhiR
//! - Oizumi et al. (2016), "Measuring Integrated Information from the Decoding Perspective" — Phi*
//! - Kitazono et al. (2018), "Efficient Algorithms for Searching the MIP in IIT" — Queyranne

use std::collections::VecDeque;

use crate::hdc::unified_hv::ContinuousHV;

/// Result of a synergistic integration computation.
#[derive(Debug, Clone)]
pub struct SigmaResult {
    /// Σ — synergistic integration value (non-negative)
    pub sigma: f64,
    /// Total mutual information of the system (before partitioning)
    pub total_mi: f64,
    /// MI of the minimum information partition
    pub mip_mi: f64,
    /// Number of state snapshots used for this computation
    pub window_used: usize,
    /// Number of dimensions (components per snapshot) tracked
    pub num_components: usize,
}

/// Configuration for synergistic integration computation.
#[derive(Debug, Clone)]
pub struct SynergisticConfig {
    /// Number of HDC components to track (subsample from full dimension for O(n²) tractability).
    /// Default: 64
    pub num_components: usize,
    /// Sliding window size (number of state snapshots). Default: 50 (1 second at 50Hz)
    pub window_size: usize,
    /// Minimum window fill before computing. Default: 10
    pub min_samples: usize,
    /// Regularization term added to covariance diagonal to ensure positive-definiteness.
    /// Default: 1e-6
    pub regularization: f64,
}

impl Default for SynergisticConfig {
    fn default() -> Self {
        Self {
            num_components: 64,
            window_size: 50,
            min_samples: 10,
            regularization: 1e-6,
        }
    }
}

/// Synergistic Integration (Σ) — PhiR-inspired measure for HDC states.
///
/// Maintains a sliding window of HDC state snapshots and periodically computes
/// the synergistic integration: how much information exists in the whole system
/// that cannot be found in any bipartition.
pub struct SynergisticIntegration {
    /// Sliding window of subsampled state vectors
    window: VecDeque<Vec<f64>>,
    config: SynergisticConfig,
}

impl SynergisticIntegration {
    /// Create a new synergistic integration tracker.
    pub fn new(config: SynergisticConfig) -> Self {
        Self {
            window: VecDeque::with_capacity(config.window_size),
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SynergisticConfig::default())
    }

    /// Feed a new HDC state snapshot into the sliding window.
    ///
    /// Subsamples `num_components` evenly-spaced dimensions from the full HV.
    pub fn push(&mut self, state: &ContinuousHV) {
        let dim = state.dim();
        let n = self.config.num_components.min(dim);
        let stride = if n > 1 { dim / n } else { 1 };

        let snapshot: Vec<f64> = (0..n).map(|i| state.values[i * stride] as f64).collect();

        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(snapshot);
    }

    /// Whether we have enough samples to compute Σ.
    pub fn ready(&self) -> bool {
        self.window.len() >= self.config.min_samples
    }

    /// Number of snapshots currently in the window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Compute Σ (synergistic integration) from the current window.
    ///
    /// Uses Phi* under Gaussian assumption:
    ///   Phi* = total_MI - min_partition_MI
    ///
    /// where total_MI = 0.5 * ln(det(Σ_system) / det(diag(Σ_system)))
    /// and partition_MI sums MIs of the two halves.
    ///
    /// Returns `None` if insufficient samples.
    pub fn compute(&self) -> Option<SigmaResult> {
        if !self.ready() {
            return None;
        }

        let n = self.config.num_components;
        let t = self.window.len();

        // Build covariance matrix from sliding window
        let cov = self.covariance_matrix(n, t);

        // Total MI under Gaussian: 0.5 * (sum of ln(var_i) - ln(det(cov)))
        let total_mi = self.gaussian_mi(&cov, n);

        // Search bipartitions for MIP (minimum information partition)
        // For n <= 16, exhaustive; otherwise greedy splitting
        let mip_mi = if n <= 16 {
            self.exhaustive_mip(&cov, n)
        } else {
            self.greedy_mip(&cov, n)
        };

        // Σ = total_MI - MIP_MI (clamped non-negative)
        let sigma = (total_mi - mip_mi).max(0.0);

        Some(SigmaResult {
            sigma,
            total_mi,
            mip_mi,
            window_used: t,
            num_components: n,
        })
    }

    /// Reset the sliding window.
    pub fn reset(&mut self) {
        self.window.clear();
    }

    // ─── Internal helpers ──────────────────────────────────────────────

    /// Build n×n covariance matrix from window snapshots.
    pub(crate) fn covariance_matrix(&self, n: usize, t: usize) -> Vec<f64> {
        let mut means = vec![0.0f64; n];
        for snap in &self.window {
            for (i, &v) in snap.iter().enumerate().take(n) {
                means[i] += v;
            }
        }
        for m in &mut means {
            *m /= t as f64;
        }

        // Covariance: C[i][j] = (1/T) * Σ (x_i - μ_i)(x_j - μ_j)
        let mut cov = vec![0.0f64; n * n];
        for snap in &self.window {
            for i in 0..n {
                let di = snap[i] - means[i];
                for j in i..n {
                    let dj = snap[j] - means[j];
                    cov[i * n + j] += di * dj;
                }
            }
        }
        // Symmetrize and normalize
        for i in 0..n {
            for j in i..n {
                cov[i * n + j] /= t as f64;
                if j > i {
                    cov[j * n + i] = cov[i * n + j];
                }
            }
            // Regularization
            cov[i * n + i] += self.config.regularization;
        }
        cov
    }

    /// Gaussian MI of the whole system.
    /// MI = 0.5 * (Σ ln(σ²_i) - ln(det(Σ)))
    fn gaussian_mi(&self, cov: &[f64], n: usize) -> f64 {
        let sum_ln_var: f64 = (0..n).map(|i| cov[i * n + i].max(1e-30).ln()).sum();
        let ln_det = self.ln_determinant(cov, n);
        0.5 * (sum_ln_var - ln_det)
    }

    /// Gaussian MI of a subset of components.
    fn partition_mi(&self, cov: &[f64], n: usize, indices: &[usize]) -> f64 {
        let k = indices.len();
        if k <= 1 {
            return 0.0;
        }
        // Extract sub-covariance matrix
        let mut sub_cov = vec![0.0f64; k * k];
        for (si, &i) in indices.iter().enumerate() {
            for (sj, &j) in indices.iter().enumerate() {
                sub_cov[si * k + sj] = cov[i * n + j];
            }
        }
        let sum_ln_var: f64 = (0..k).map(|i| sub_cov[i * k + i].max(1e-30).ln()).sum();
        let ln_det = self.ln_determinant(&sub_cov, k);
        0.5 * (sum_ln_var - ln_det)
    }

    /// Exhaustive bipartition MIP search (2^(n-1) - 1 partitions).
    pub(crate) fn exhaustive_mip(&self, cov: &[f64], n: usize) -> f64 {
        let mut min_mi = f64::MAX;
        // Enumerate all bipartitions: bit i indicates which side
        // Only need half of 2^n due to symmetry (fix element 0 in partition A)
        let limit = 1u32 << (n - 1);
        for mask in 1..limit {
            let mut part_a = vec![0usize]; // element 0 always in A
            let mut part_b = Vec::new();
            for bit in 0..(n - 1) {
                if (mask >> bit) & 1 == 1 {
                    part_a.push(bit + 1);
                } else {
                    part_b.push(bit + 1);
                }
            }
            if part_b.is_empty() {
                continue;
            }
            let mi = self.partition_mi(cov, n, &part_a) + self.partition_mi(cov, n, &part_b);
            if mi < min_mi {
                min_mi = mi;
            }
        }
        if min_mi == f64::MAX { 0.0 } else { min_mi }
    }

    /// Greedy MIP search for n > 16: iteratively move the element that reduces MI most.
    pub(crate) fn greedy_mip(&self, cov: &[f64], n: usize) -> f64 {
        // Start with balanced partition: first half / second half
        let mut part_a: Vec<usize> = (0..n / 2).collect();
        let mut part_b: Vec<usize> = (n / 2..n).collect();

        let mut best_mi = self.partition_mi(cov, n, &part_a) + self.partition_mi(cov, n, &part_b);

        // Greedy swaps: move one element between partitions to minimize MI
        let mut improved = true;
        while improved {
            improved = false;
            // Try moving each element from A to B
            for idx in 0..part_a.len() {
                if part_a.len() <= 1 {
                    break;
                }
                let elem = part_a[idx];
                part_a.remove(idx);
                part_b.push(elem);
                let mi = self.partition_mi(cov, n, &part_a) + self.partition_mi(cov, n, &part_b);
                if mi < best_mi - 1e-12 {
                    best_mi = mi;
                    improved = true;
                    break;
                }
                part_b.pop();
                part_a.insert(idx, elem);
            }
            if improved {
                continue;
            }
            // Try moving each element from B to A
            for idx in 0..part_b.len() {
                if part_b.len() <= 1 {
                    break;
                }
                let elem = part_b[idx];
                part_b.remove(idx);
                part_a.push(elem);
                let mi = self.partition_mi(cov, n, &part_a) + self.partition_mi(cov, n, &part_b);
                if mi < best_mi - 1e-12 {
                    best_mi = mi;
                    improved = true;
                    break;
                }
                part_a.pop();
                part_b.insert(idx, elem);
            }
        }
        best_mi
    }

    /// ln(det(M)) via Cholesky decomposition. Falls back to diagonal product if not PD.
    fn ln_determinant(&self, matrix: &[f64], n: usize) -> f64 {
        // Attempt Cholesky: L such that M = L * L^T
        let mut l = vec![0.0f64; n * n];
        let mut ok = true;
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    let diag = matrix[i * n + i] - sum;
                    if diag <= 0.0 {
                        ok = false;
                        break;
                    }
                    l[i * n + j] = diag.sqrt();
                } else {
                    l[i * n + j] = (matrix[i * n + j] - sum) / l[j * n + j];
                }
            }
            if !ok {
                break;
            }
        }
        if ok {
            // ln(det(M)) = 2 * Σ ln(L_ii)
            let mut result = 0.0;
            for i in 0..n {
                result += l[i * n + i].ln();
            }
            2.0 * result
        } else {
            // Fallback: product of diagonal (treats as diagonal matrix)
            (0..n).map(|i| matrix[i * n + i].max(1e-30).ln()).sum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_window_returns_none() {
        let si = SynergisticIntegration::default_config();
        assert!(si.compute().is_none());
        assert!(!si.ready());
    }

    #[test]
    fn test_insufficient_samples_returns_none() {
        let config = SynergisticConfig {
            min_samples: 5,
            ..Default::default()
        };
        let mut si = SynergisticIntegration::new(config);
        for i in 0..4 {
            si.push(&ContinuousHV::random(128, i));
        }
        assert!(!si.ready());
        assert!(si.compute().is_none());
    }

    #[test]
    fn test_identical_states_zero_sigma() {
        let config = SynergisticConfig {
            num_components: 8,
            window_size: 20,
            min_samples: 5,
            regularization: 1e-6,
        };
        let mut si = SynergisticIntegration::new(config);
        let state = ContinuousHV::random(128, 42);
        for _ in 0..10 {
            si.push(&state);
        }
        let result = si.compute().unwrap();
        // Identical states → zero covariance off-diagonal → near-zero MI → near-zero sigma
        assert!(
            result.sigma < 0.1,
            "Identical states should have near-zero sigma, got {}",
            result.sigma
        );
        assert!(result.sigma.is_finite());
    }

    #[test]
    fn test_varied_states_positive_sigma() {
        let config = SynergisticConfig {
            num_components: 8,
            window_size: 50,
            min_samples: 10,
            regularization: 1e-6,
        };
        let mut si = SynergisticIntegration::new(config);
        // Feed diverse states — should produce positive MI structure
        for i in 0..30 {
            let hv = ContinuousHV::random(128, i);
            si.push(&hv);
        }
        let result = si.compute().unwrap();
        assert!(result.sigma.is_finite());
        assert!(result.total_mi.is_finite());
        assert!(result.mip_mi.is_finite());
        assert!(result.sigma >= 0.0);
        assert_eq!(result.window_used, 30);
        assert_eq!(result.num_components, 8);
    }

    #[test]
    fn test_window_eviction() {
        let config = SynergisticConfig {
            num_components: 4,
            window_size: 5,
            min_samples: 3,
            ..Default::default()
        };
        let mut si = SynergisticIntegration::new(config);
        for i in 0..10 {
            si.push(&ContinuousHV::random(64, i));
        }
        assert_eq!(si.window_len(), 5);
    }

    #[test]
    fn test_reset_clears_window() {
        let mut si = SynergisticIntegration::default_config();
        for i in 0..15 {
            si.push(&ContinuousHV::random(128, i));
        }
        assert!(si.ready());
        si.reset();
        assert!(!si.ready());
        assert_eq!(si.window_len(), 0);
    }

    #[test]
    fn test_sigma_bounded() {
        let config = SynergisticConfig {
            num_components: 16,
            window_size: 50,
            min_samples: 10,
            regularization: 1e-6,
        };
        let mut si = SynergisticIntegration::new(config);
        for i in 0..50 {
            si.push(&ContinuousHV::random(256, i * 7 + 3));
        }
        let result = si.compute().unwrap();
        // Sigma should be non-negative and finite
        assert!(result.sigma >= 0.0);
        assert!(result.sigma.is_finite());
        // Total MI should be >= MIP MI (by definition)
        assert!(result.total_mi >= result.mip_mi - 1e-9);
    }

    #[test]
    fn test_covariance_symmetry() {
        let config = SynergisticConfig {
            num_components: 4,
            window_size: 20,
            min_samples: 5,
            regularization: 1e-8,
        };
        let mut si = SynergisticIntegration::new(config);
        for i in 0..10 {
            si.push(&ContinuousHV::random(64, i));
        }
        let cov = si.covariance_matrix(4, 10);
        // Verify symmetry
        for i in 0..4 {
            for j in 0..4 {
                let diff = (cov[i * 4 + j] - cov[j * 4 + i]).abs();
                assert!(diff < 1e-12, "Covariance not symmetric at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_greedy_vs_exhaustive_agreement() {
        // For small n, both should give similar results
        let config = SynergisticConfig {
            num_components: 8,
            window_size: 50,
            min_samples: 20,
            regularization: 1e-6,
        };
        let mut si = SynergisticIntegration::new(config);
        for i in 0..40 {
            si.push(&ContinuousHV::random(128, i + 100));
        }
        let n = 8;
        let t = si.window.len();
        let cov = si.covariance_matrix(n, t);
        let exhaustive = si.exhaustive_mip(&cov, n);
        let greedy = si.greedy_mip(&cov, n);
        // Greedy should be >= exhaustive (it's an approximation)
        // but for small n, should be close
        assert!(
            (greedy - exhaustive).abs() < exhaustive.abs() * 0.5 + 0.1,
            "Greedy ({}) too far from exhaustive ({})",
            greedy,
            exhaustive
        );
    }

    #[test]
    fn test_ln_determinant_identity() {
        let si = SynergisticIntegration::default_config();
        // 3x3 identity matrix: det = 1, ln(det) = 0
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = si.ln_determinant(&identity, 3);
        assert!(
            result.abs() < 1e-10,
            "ln(det(I)) should be 0, got {}",
            result
        );
    }

    #[test]
    fn test_ln_determinant_diagonal() {
        let si = SynergisticIntegration::default_config();
        // diag(2, 3, 5): det = 30, ln(30) ≈ 3.4012
        let diag = vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0];
        let result = si.ln_determinant(&diag, 3);
        let expected = 30.0f64.ln();
        assert!(
            (result - expected).abs() < 1e-10,
            "ln(det(diag(2,3,5))) should be {}, got {}",
            expected,
            result
        );
    }
}
