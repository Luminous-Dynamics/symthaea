// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ordinary-least-squares simple linear regression.

use crate::descriptive::{covariance, mean, variance};

/// Fitted line `y = slope·x + intercept` with its coefficient of determination.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearFit {
    pub slope: f64,
    pub intercept: f64,
    /// Fraction of variance explained, R² ∈ [0, 1].
    pub r_squared: f64,
}

impl LinearFit {
    /// Predict `y` at a given `x`.
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// Fit `y ~ x` by ordinary least squares. Needs ≥ 2 points with non-zero
/// variance in `x`.
pub fn linear_regression(xs: &[f64], ys: &[f64]) -> Option<LinearFit> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let vx = variance(xs)?;
    if vx == 0.0 {
        return None;
    }
    let cov = covariance(xs, ys)?;
    let slope = cov / vx;
    let (mx, my) = (mean(xs)?, mean(ys)?);
    let intercept = my - slope * mx;
    // R² = (cov / (σx·σy))².
    let vy = variance(ys)?;
    let r_squared = if vy == 0.0 {
        // y is constant: the fit explains nothing unless slope is 0 too.
        if slope == 0.0 { 1.0 } else { 0.0 }
    } else {
        (cov * cov) / (vx * vy)
    };
    Some(LinearFit {
        slope,
        intercept,
        r_squared,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_line() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
        let f = linear_regression(&x, &y).unwrap();
        assert!((f.slope - 2.0).abs() < 1e-12);
        assert!((f.intercept - 1.0).abs() < 1e-12);
        assert!((f.r_squared - 1.0).abs() < 1e-12);
        assert!((f.predict(10.0) - 21.0).abs() < 1e-12);
    }

    #[test]
    fn noisy_fit_has_partial_r_squared() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.2, 7.8, 10.1]; // ≈ 2x with noise
        let f = linear_regression(&x, &y).unwrap();
        assert!((f.slope - 2.0).abs() < 0.2);
        assert!(f.r_squared > 0.99 && f.r_squared <= 1.0);
    }

    #[test]
    fn degenerate_x_returns_none() {
        assert!(linear_regression(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
    }
}
