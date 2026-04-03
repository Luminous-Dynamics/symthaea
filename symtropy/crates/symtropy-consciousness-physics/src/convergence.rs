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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_series_converges() {
        let mut det = ConvergenceDetector::new(10, 1e-6);
        for _ in 0..10 {
            det.push(5.0);
        }
        assert!(det.is_converged(), "Constant series should converge immediately");
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
        assert!(det.is_converged(), "Should converge after window of constants");
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
}
