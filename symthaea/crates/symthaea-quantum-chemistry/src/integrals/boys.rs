// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Boys function F_n(x) — the fundamental integral of quantum chemistry.
//!
//! F_n(x) = ∫₀¹ t^(2n) exp(-x·t²) dt
//!
//! For small x: Taylor series. For large x: asymptotic expansion.
//! The crossover at x ≈ 25 balances accuracy and speed.
//!
//! References:
//! - Boys, S. F. (1950). Proc. R. Soc. Lond. A 200, 542.
//! - Shavitt, I. (1963). Methods in Computational Physics, vol. 2.

use crate::constants::SQRT_PI;

/// Compute the Boys function F_n(x) for given order n and argument x.
///
/// Uses Taylor series for x < 25, asymptotic form for x ≥ 25.
/// Downward recursion from F_0 is used for higher orders.
pub fn boys_function(n: u32, x: f64) -> f64 {
    if x < 1e-14 {
        // F_n(0) = 1/(2n+1)
        return 1.0 / (2 * n + 1) as f64;
    }

    if x >= 25.0 {
        // Asymptotic: F_n(x) ≈ (2n-1)!! / 2^(n+1) * sqrt(π/x^(2n+1))
        // More precisely: F_n(x) ≈ (2n)! / (2^(2n+1) * n!) * sqrt(π / x^(2n+1))
        // Simplified: F_n(x) ≈ double_factorial(2n-1) / 2^(n+1) * sqrt(π) / x^(n+0.5)
        return asymptotic_boys(n, x);
    }

    // For moderate x, compute F_0(x) then use upward recursion.
    // F_0(x) = sqrt(π/x) * erf(sqrt(x)) / 2
    let f0 = f0_boys(x);

    if n == 0 {
        return f0;
    }

    // Upward recursion: F_{n+1}(x) = ((2n+1) * F_n(x) - exp(-x)) / (2x)
    let exp_neg_x = (-x).exp();
    let mut fn_val = f0;
    for k in 0..n {
        fn_val = ((2 * k + 1) as f64 * fn_val - exp_neg_x) / (2.0 * x);
    }

    fn_val
}

/// F_0(x) = sqrt(π/(4x)) * erf(sqrt(x))
fn f0_boys(x: f64) -> f64 {
    let sqrt_x = x.sqrt();
    SQRT_PI / (2.0 * sqrt_x) * erf_approx(sqrt_x)
}

/// Asymptotic form for large x.
/// F_n(x) ≈ (2n-1)!! * sqrt(π) / (2^(n+1) * x^(n+0.5))
fn asymptotic_boys(n: u32, x: f64) -> f64 {
    let mut dbl_fact = 1.0_f64;
    for k in 1..=n {
        dbl_fact *= (2 * k - 1) as f64;
    }
    // (2n-1)!! * sqrt(π) / (2^(n+1) * x^(n+0.5))
    dbl_fact * SQRT_PI / (2.0_f64.powi(n as i32 + 1) * x.powf(n as f64 + 0.5))
}

/// Approximation to the error function erf(x).
/// Abramowitz & Stegun 7.1.26, max error ≈ 1.5e-7.
fn erf_approx(x: f64) -> f64 {
    if x < 0.0 {
        return -erf_approx(-x);
    }

    let p = 0.327_591_1;
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;

    let t = 1.0 / (1.0 + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boys_f0_zero() {
        // F_0(0) = 1
        assert!((boys_function(0, 0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_boys_fn_zero() {
        // F_n(0) = 1/(2n+1)
        assert!((boys_function(1, 0.0) - 1.0 / 3.0).abs() < 1e-12);
        assert!((boys_function(2, 0.0) - 1.0 / 5.0).abs() < 1e-12);
        assert!((boys_function(3, 0.0) - 1.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_boys_f0_known_values() {
        // F_0(1) ≈ 0.74682
        let f0_1 = boys_function(0, 1.0);
        assert!(
            (f0_1 - 0.746_824_133).abs() < 1e-5,
            "F_0(1) = {}, expected ≈ 0.74682",
            f0_1
        );

        // F_0(10) ≈ 0.27995 (asymptotic region boundary)
        let f0_10 = boys_function(0, 10.0);
        assert!((f0_10 - 0.279_95).abs() < 1e-3, "F_0(10) = {}", f0_10);
    }

    #[test]
    fn test_boys_asymptotic() {
        // For large x, F_0(x) → sqrt(π/(4x))
        let x = 100.0;
        let f0 = boys_function(0, x);
        let expected = SQRT_PI / (2.0 * x.sqrt());
        assert!(
            (f0 - expected).abs() < 1e-6,
            "F_0(100) = {}, expected ≈ {}",
            f0,
            expected
        );
    }

    #[test]
    fn test_boys_f1_known() {
        // F_1(1) ≈ 0.18946
        let f1 = boys_function(1, 1.0);
        assert!(
            (f1 - 0.189_47).abs() < 1e-3,
            "F_1(1) = {}, expected ≈ 0.18947",
            f1
        );
    }

    #[test]
    fn test_boys_monotonicity() {
        // F_n(x) is monotonically decreasing in x for x > 0
        for n in 0..4 {
            let f_small = boys_function(n, 0.5);
            let f_large = boys_function(n, 5.0);
            assert!(
                f_small > f_large,
                "F_{}(0.5)={} should > F_{}(5.0)={}",
                n,
                f_small,
                n,
                f_large
            );
        }
    }

    #[test]
    fn test_boys_order_monotonicity() {
        // F_n(x) is monotonically decreasing in n for fixed x > 0
        let x = 2.0;
        let f0 = boys_function(0, x);
        let f1 = boys_function(1, x);
        let f2 = boys_function(2, x);
        assert!(f0 > f1 && f1 > f2);
    }
}
