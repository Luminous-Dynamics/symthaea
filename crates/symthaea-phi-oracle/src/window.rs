// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::VecDeque;

/// Recompute running sums from scratch every this many pushes to prevent
/// floating-point drift from accumulating over long-running windows.
const RECOMPUTE_INTERVAL: usize = 1000;

/// Sliding window that accumulates raw observations and builds variable-level
/// covariance matrices via running sums (O(1) amortized push).
///
/// Unlike pushing encoded HVs into SpectralMIPFinder (which computes covariance
/// across random-projection dimensions), this operates on the original system
/// variables, preserving the correlation structure.
pub(crate) struct ObservationWindow {
    window: VecDeque<Vec<f64>>,
    num_vars: usize,
    capacity: usize,
    /// Running Σ x[i] over the window.
    sum: Vec<f64>,
    /// Running Σ x[i]*x[j] over the window (flat n×n, row-major).
    cross_sum: Vec<f64>,
    /// Diagonal regularization added to covariance (prevents singular matrices).
    regularization: f64,
    /// Total pushes since creation/clear — triggers periodic recomputation.
    push_count: usize,
}

impl ObservationWindow {
    /// Create a new observation window.
    ///
    /// - `num_vars`: number of system variables per observation.
    /// - `capacity`: maximum window size (FIFO eviction).
    /// - `regularization`: diagonal regularization for covariance (e.g., 1e-6).
    pub(crate) fn new(num_vars: usize, capacity: usize, regularization: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity),
            num_vars,
            capacity,
            sum: vec![0.0; num_vars],
            cross_sum: vec![0.0; num_vars * num_vars],
            regularization,
            push_count: 0,
        }
    }

    /// Push one observation into the window. Evicts the oldest if at capacity.
    ///
    /// Panics in debug mode if `observation.len() != num_vars`.
    pub(crate) fn push(&mut self, observation: &[f64]) {
        debug_assert_eq!(observation.len(), self.num_vars);
        let n = self.num_vars;

        // Evict oldest if at capacity
        if self.window.len() >= self.capacity {
            if let Some(old) = self.window.pop_front() {
                for i in 0..n {
                    self.sum[i] -= old[i];
                    for j in 0..n {
                        self.cross_sum[i * n + j] -= old[i] * old[j];
                    }
                }
            }
        }

        // Add new observation to running sums
        for i in 0..n {
            self.sum[i] += observation[i];
            for j in 0..n {
                self.cross_sum[i * n + j] += observation[i] * observation[j];
            }
        }

        self.window.push_back(observation.to_vec());

        self.push_count += 1;
        if self.push_count % RECOMPUTE_INTERVAL == 0 {
            self.recompute_sums();
        }
    }

    /// Recompute running sums from scratch to eliminate accumulated
    /// floating-point error from incremental add/subtract cycles.
    fn recompute_sums(&mut self) {
        let n = self.num_vars;
        self.sum.fill(0.0);
        self.cross_sum.fill(0.0);
        for obs in &self.window {
            for i in 0..n {
                self.sum[i] += obs[i];
                for j in 0..n {
                    self.cross_sum[i * n + j] += obs[i] * obs[j];
                }
            }
        }
    }

    /// Number of observations currently in the window.
    pub(crate) fn len(&self) -> usize {
        self.window.len()
    }

    /// Number of system variables per observation.
    pub(crate) fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Build an n×n covariance matrix (flat row-major) from the running sums.
    ///
    /// Cov[i,j] = E[x_i * x_j] - E[x_i] * E[x_j] + δ_ij * regularization
    pub(crate) fn build_covariance(&self) -> Vec<f64> {
        let n = self.num_vars;
        let w = self.window.len() as f64;
        let mut cov = vec![0.0; n * n];

        if w < 1.0 {
            return cov;
        }

        for i in 0..n {
            for j in 0..n {
                let mean_ij = self.cross_sum[i * n + j] / w;
                let mean_i = self.sum[i] / w;
                let mean_j = self.sum[j] / w;
                cov[i * n + j] = mean_ij - mean_i * mean_j;
            }
            // Diagonal regularization
            cov[i * n + i] += self.regularization;
        }

        cov
    }

    /// Reset the window, clearing all observations and running sums.
    pub(crate) fn clear(&mut self) {
        self.window.clear();
        self.sum.fill(0.0);
        self.cross_sum.fill(0.0);
        self.push_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_len() {
        let mut w = ObservationWindow::new(3, 10, 1e-6);
        assert_eq!(w.len(), 0);
        w.push(&[1.0, 2.0, 3.0]);
        assert_eq!(w.len(), 1);
        w.push(&[4.0, 5.0, 6.0]);
        assert_eq!(w.len(), 2);
        assert_eq!(w.num_vars(), 3);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut w = ObservationWindow::new(2, 3, 0.0);
        w.push(&[1.0, 0.0]);
        w.push(&[2.0, 0.0]);
        w.push(&[3.0, 0.0]);
        assert_eq!(w.len(), 3);
        // This should evict [1.0, 0.0]
        w.push(&[4.0, 0.0]);
        assert_eq!(w.len(), 3);
        // Sum should be 2+3+4=9 for variable 0
        let cov = w.build_covariance();
        let mean_0 = (2.0 + 3.0 + 4.0) / 3.0;
        let expected_var =
            ((2.0f64 - mean_0).powi(2) + (3.0f64 - mean_0).powi(2) + (4.0f64 - mean_0).powi(2))
                / 3.0;
        assert!((cov[0] - expected_var).abs() < 1e-10, "variance mismatch");
    }

    #[test]
    fn test_covariance_correlated() {
        let mut w = ObservationWindow::new(2, 100, 0.0);
        // x0 = t, x1 = 2*t → perfectly correlated
        for t in 0..50 {
            let x = t as f64 * 0.1;
            w.push(&[x, 2.0 * x]);
        }
        let cov = w.build_covariance();
        // cov[0,1] should be positive and equal to 2 * var(x0)
        let var_x0 = cov[0]; // Cov[0,0]
        let cov_01 = cov[1]; // Cov[0,1]
        assert!(var_x0 > 0.0, "variance should be positive");
        assert!(
            (cov_01 - 2.0 * var_x0).abs() < 1e-10,
            "cov(x,2x) should be 2*var(x)"
        );
    }

    #[test]
    fn test_numerical_stability_after_many_pushes() {
        let cap = 50;
        let num_vars = 4;
        let mut w = ObservationWindow::new(num_vars, cap, 1e-6);

        // Push 5000 observations (triggers multiple recomputations)
        let mut rng: u64 = 42;
        let mut recent: Vec<Vec<f64>> = Vec::new();
        for _ in 0..5000 {
            let obs: Vec<f64> = (0..num_vars)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
                })
                .collect();
            w.push(&obs);
            recent.push(obs);
            if recent.len() > cap {
                recent.remove(0);
            }
        }

        // Build a fresh window from only the last `cap` entries
        let mut fresh = ObservationWindow::new(num_vars, cap, 1e-6);
        for obs in &recent {
            fresh.push(obs);
        }

        let cov_incremental = w.build_covariance();
        let cov_fresh = fresh.build_covariance();

        let max_diff = cov_incremental
            .iter()
            .zip(cov_fresh.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_diff < 1e-10,
            "covariance drift after 5000 pushes: max element-wise diff = {max_diff:.2e}"
        );
    }

    #[test]
    fn test_clear() {
        let mut w = ObservationWindow::new(2, 10, 1e-6);
        w.push(&[1.0, 2.0]);
        w.push(&[3.0, 4.0]);
        w.clear();
        assert_eq!(w.len(), 0);
        let cov = w.build_covariance();
        assert!(
            cov.iter().all(|&v| v == 0.0),
            "cleared window should have zero covariance"
        );
    }
}
