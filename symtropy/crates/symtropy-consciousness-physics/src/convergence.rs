// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rolling window convergence detector for scalar time series.
//!
//! Used to detect when J/Phi (joules per unit consciousness change) has
//! stabilized — indicating the system has reached a thermodynamic equilibrium.

use std::collections::VecDeque;

/// Detects convergence of a scalar time series using rolling window variance.
///
/// Convergence is declared when the rolling variance falls below a threshold
/// for a sustained period. This is conservative: brief dips don't count.
pub struct ConvergenceDetector {
    /// Rolling window of recent values.
    window: VecDeque<f64>,
    /// Maximum window size.
    window_size: usize,
    /// Variance below this = converged.
    variance_threshold: f64,
}

impl ConvergenceDetector {
    /// Create a new detector.
    ///
    /// `window_size`: number of recent values to track (e.g., 100 ticks).
    /// `variance_threshold`: convergence when rolling variance < this (e.g., 1e-4).
    pub fn new(window_size: usize, variance_threshold: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            variance_threshold,
        }
    }

    /// Push a new value and return whether the series has converged.
    pub fn push(&mut self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        self.window.push_back(value);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
        self.is_converged()
    }

    /// Whether the current window is converged.
    pub fn is_converged(&self) -> bool {
        if self.window.len() < self.window_size {
            return false; // Not enough data yet.
        }
        self.rolling_variance() < self.variance_threshold
    }

    /// Mean of the current window.
    pub fn rolling_mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.window.iter().sum();
        sum / self.window.len() as f64
    }

    /// Variance of the current window.
    pub fn rolling_variance(&self) -> f64 {
        if self.window.len() < 2 {
            return f64::MAX;
        }
        let mean = self.rolling_mean();
        let sum_sq: f64 = self.window.iter().map(|x| (x - mean) * (x - mean)).sum();
        sum_sq / (self.window.len() - 1) as f64
    }

    /// Number of values currently in the window.
    pub fn count(&self) -> usize {
        self.window.len()
    }
}

/// Mann-Whitney U test (nonparametric, no normality assumption).
/// Returns (U statistic, z-score, p-value approximation).
/// Two-tailed: tests whether samples come from different distributions.
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    if na < 1.0 || nb < 1.0 {
        return (0.0, 0.0, 1.0);
    }

    // Count how many times a[i] > b[j] for all pairs
    let mut u_a = 0.0;
    for &ai in a {
        for &bj in b {
            if ai > bj {
                u_a += 1.0;
            } else if (ai - bj).abs() < 1e-15 {
                u_a += 0.5;
            }
        }
    }
    let u_b = na * nb - u_a;
    let u = u_a.min(u_b);

    // Normal approximation for z-score (valid for n > 20)
    let mean_u = na * nb / 2.0;
    let std_u = (na * nb * (na + nb + 1.0) / 12.0).sqrt();
    let z = if std_u > 1e-15 {
        (u - mean_u) / std_u
    } else {
        0.0
    };

    // Two-tailed p-value from z-score (normal CDF approximation)
    let p = 2.0 * normal_cdf(-z.abs());
    (u, z, p)
}

/// Cohen's d effect size: (mean_a - mean_b) / pooled_std_dev.
/// |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large.
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    if na < 2.0 || nb < 2.0 {
        return 0.0;
    }

    let mean_a = a.iter().sum::<f64>() / na;
    let mean_b = b.iter().sum::<f64>() / nb;
    let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (na - 1.0);
    let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (nb - 1.0);

    let pooled_std = (((na - 1.0) * var_a + (nb - 1.0) * var_b) / (na + nb - 2.0)).sqrt();
    if pooled_std < 1e-15 {
        return 0.0;
    }
    (mean_a - mean_b) / pooled_std
}

/// Holm-Bonferroni correction for multiple comparisons.
///
/// Less conservative than Bonferroni but still controls family-wise error rate.
/// Takes a slice of (label, p-value) pairs and returns adjusted p-values with
/// significance flags.
///
/// Algorithm: sort p-values ascending, reject p_i if p_i × (m - i) < alpha,
/// where m = number of tests and i = rank (0-indexed).
///
/// Ref: Holm, S. (1979). Scandinavian Journal of Statistics, 6(2), 65-70.
pub fn holm_bonferroni<'a>(tests: &[(&'a str, f64)], alpha: f64) -> Vec<(&'a str, f64, bool)> {
    let m = tests.len();
    if m == 0 {
        return vec![];
    }

    // Sort by p-value (ascending), preserving original labels
    let mut indexed: Vec<(usize, &str, f64)> = tests
        .iter()
        .enumerate()
        .map(|(i, (label, p))| (i, *label, *p))
        .collect();
    indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Step-down: adjust p-values and determine significance
    let mut results = vec![("", 0.0, false); m];
    let mut still_rejecting = true;

    for (rank, &(orig_idx, label, p)) in indexed.iter().enumerate() {
        let correction_factor = (m - rank) as f64;
        let adjusted_p = (p * correction_factor).min(1.0);

        // Once we fail to reject, all subsequent tests also fail
        let significant = still_rejecting && adjusted_p < alpha;
        if !significant {
            still_rejecting = false;
        }

        results[orig_idx] = (label, adjusted_p, significant);
    }

    results
}

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d
        * (-x * x / 2.0).exp()
        * (t * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))));
    if x >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_series_converges() {
        let mut det = ConvergenceDetector::new(10, 1e-6);
        for _ in 0..10 {
            det.push(5.0);
        }
        assert!(
            det.is_converged(),
            "Constant series should converge immediately"
        );
        assert!((det.rolling_mean() - 5.0).abs() < 1e-10);
        assert!(det.rolling_variance() < 1e-10);
    }

    #[test]
    fn random_series_does_not_converge() {
        let mut det = ConvergenceDetector::new(10, 1e-6);
        for i in 0..10 {
            det.push(i as f64 * 7.3 % 13.0); // pseudo-random
        }
        assert!(!det.is_converged(), "Varying series should not converge");
    }

    #[test]
    fn step_function_converges_after_window() {
        let mut det = ConvergenceDetector::new(5, 1e-4);
        // First 5 values: varying.
        for i in 0..5 {
            det.push(i as f64);
        }
        assert!(!det.is_converged());
        // Next 5 values: constant (window fills with constant).
        for _ in 0..5 {
            det.push(10.0);
        }
        assert!(
            det.is_converged(),
            "Should converge after window of constants"
        );
    }

    #[test]
    fn empty_detector_not_converged() {
        let det = ConvergenceDetector::new(10, 1e-4);
        assert!(!det.is_converged());
        assert!(det.rolling_variance() == f64::MAX);
    }

    #[test]
    fn nan_values_ignored() {
        let mut det = ConvergenceDetector::new(5, 1e-4);
        for _ in 0..5 {
            det.push(3.0);
        }
        assert!(det.is_converged());
        // NaN push should not change convergence state (rejected).
        det.push(f64::NAN);
        assert!(det.is_converged());
    }

    #[test]
    fn mann_whitney_identical_samples() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_, _, p) = mann_whitney_u(&a, &b);
        assert!(p > 0.5, "Identical samples should have high p-value: {p}");
    }

    #[test]
    fn mann_whitney_different_samples() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let (_, _, p) = mann_whitney_u(&a, &b);
        assert!(
            p < 0.05,
            "Very different samples should have low p-value: {p}"
        );
    }

    #[test]
    fn cohens_d_large_effect() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let d = cohens_d(&a, &b);
        assert!(d.abs() > 0.8, "Large separation should give |d| > 0.8: {d}");
    }

    #[test]
    fn cohens_d_no_effect() {
        let a = vec![5.0, 5.0, 5.0, 5.0];
        let b = vec![5.0, 5.0, 5.0, 5.0];
        let d = cohens_d(&a, &b);
        assert!(
            (d - 0.0).abs() < 1e-10,
            "Identical means should give d=0: {d}"
        );
    }

    #[test]
    fn holm_bonferroni_basic() {
        let tests = vec![("A", 0.01), ("B", 0.04), ("C", 0.03)];
        let results = holm_bonferroni(&tests, 0.05);
        // Sorted: A(0.01), C(0.03), B(0.04)
        // A: 0.01 × 3 = 0.03 < 0.05 → sig
        // C: 0.03 × 2 = 0.06 ≥ 0.05 → not sig
        // B: 0.04 × 1 = 0.04 < 0.05 → but C failed, so B also fails (step-down)
        assert!(results[0].2, "A should be significant");
        assert!(!results[1].2, "B should NOT be significant (step-down)");
        assert!(!results[2].2, "C should NOT be significant");
    }

    #[test]
    fn holm_bonferroni_all_significant() {
        let tests = vec![("A", 0.001), ("B", 0.002), ("C", 0.003)];
        let results = holm_bonferroni(&tests, 0.05);
        // A: 0.001×3=0.003, B: 0.002×2=0.004, C: 0.003×1=0.003
        assert!(results.iter().all(|r| r.2), "All should be significant");
    }

    #[test]
    fn holm_bonferroni_none_significant() {
        let tests = vec![("A", 0.5), ("B", 0.6)];
        let results = holm_bonferroni(&tests, 0.05);
        assert!(results.iter().all(|r| !r.2), "None should be significant");
    }

    #[test]
    fn holm_bonferroni_preserves_order() {
        let tests = vec![("X", 0.04), ("Y", 0.01), ("Z", 0.03)];
        let results = holm_bonferroni(&tests, 0.05);
        assert_eq!(results[0].0, "X");
        assert_eq!(results[1].0, "Y");
        assert_eq!(results[2].0, "Z");
    }

    #[test]
    fn holm_bonferroni_empty() {
        let results = holm_bonferroni(&[], 0.05);
        assert!(results.is_empty());
    }

    #[test]
    fn normal_cdf_symmetry() {
        let p_pos = normal_cdf(1.96);
        let p_neg = normal_cdf(-1.96);
        assert!((p_pos - 0.975).abs() < 0.01, "CDF(1.96) ≈ 0.975: {p_pos}");
        assert!((p_neg - 0.025).abs() < 0.01, "CDF(-1.96) ≈ 0.025: {p_neg}");
    }
}
