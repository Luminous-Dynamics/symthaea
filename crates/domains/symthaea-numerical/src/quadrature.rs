// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Numerical integration (quadrature) of a function over `[a, b]`.

/// Composite trapezoidal rule with `n` sub-intervals (`n ≥ 1`).
pub fn trapezoidal(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = n.max(1);
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

/// Composite Simpson's rule with `n` sub-intervals (`n` is rounded up to the
/// next even number). Exact for cubics.
pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n < 2 { 2 } else { n + (n & 1) }; // even, ≥ 2
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let coeff = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += coeff * f(a + i as f64 * h);
    }
    sum * h / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn integrates_polynomial_exactly_simpson() {
        // ∫₀¹ x² dx = 1/3 (Simpson is exact for ≤ cubics).
        assert!((simpson(|x| x * x, 0.0, 1.0, 2) - 1.0 / 3.0).abs() < 1e-12);
        // ∫₀¹ x³ dx = 1/4.
        assert!((simpson(|x| x * x * x, 0.0, 1.0, 2) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn integrates_sine() {
        // ∫₀^π sin x dx = 2.
        assert!((simpson(|x| x.sin(), 0.0, PI, 100) - 2.0).abs() < 1e-6);
        assert!((trapezoidal(|x| x.sin(), 0.0, PI, 10000) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn trapezoidal_linear_exact() {
        // ∫₀² (2x+1) dx = [x²+x]₀² = 6.
        assert!((trapezoidal(|x| 2.0 * x + 1.0, 0.0, 2.0, 1) - 6.0).abs() < 1e-12);
    }
}
