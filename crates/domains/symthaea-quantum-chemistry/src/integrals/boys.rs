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
/// Uses a cancellation-free series for x < 25, asymptotic form for x ≥ 25.
///
/// A prior version of this function computed F_0 then used *upward*
/// recursion (F_{n+1}(x) = ((2n+1)F_n(x) - exp(-x)) / (2x)) to get higher
/// orders. That direction is textbook-unstable (Helgaker, Jorgensen & Olsen,
/// *Molecular Electronic-Structure Theory*, ch. 9): each step subtracts two
/// nearly-equal quantities and divides by 2x, amplifying rounding error
/// every iteration. It stayed unnoticed because the existing tests only ever
/// checked orders up to n=3 at x=0 or n=2 at a single moderate x -- a real
/// molecule combining two second-row p-block atoms with close orbital
/// exponents (verified: any carbon+nitrogen molecule, e.g. HCN/HNC) needs
/// higher orders at small x and drove it to ~1e24 Hartree. See
/// `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md` Phase 0 log.
pub fn boys_function(n: u32, x: f64) -> f64 {
    if x < 1e-14 {
        // F_n(0) = 1/(2n+1)
        return 1.0 / (2 * n + 1) as f64;
    }

    if x >= 25.0 {
        // Asymptotic: F_n(x) ≈ (2n-1)!! / 2^(n+1) * sqrt(π/x^(2n+1))
        return asymptotic_boys(n, x);
    }

    boys_series_stable(n, x)
}

/// F_n(x) via the incomplete-gamma series
/// F_n(x) = (e^-x / 2) * sum_{k=0}^inf x^k / [s(s+1)...(s+k)], with s = n + 1/2.
///
/// Every term is positive for x, s > 0, so unlike upward recursion this has
/// no cancellation at any order or argument in the covered range -- the
/// standard robust way to evaluate F_n directly, without going through F_0.
fn boys_series_stable(n: u32, x: f64) -> f64 {
    let s = n as f64 + 0.5;
    let mut term = 1.0 / s;
    let mut sum = term;
    let mut k = 0u32;
    loop {
        k += 1;
        term *= x / (s + k as f64);
        sum += term;
        if term < sum * 1e-16 || k > 500 {
            break;
        }
    }
    0.5 * (-x).exp() * sum
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

    /// Regression test for the upward-recursion instability found 2026-07-12
    /// via `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md` Phase 0
    /// (HCN/HNC RHF diverging to ~1e24 Hartree). F_n(x) has a closed-form
    /// bound -- 0 < F_n(x) <= F_n(0) = 1/(2n+1) -- for every n, x >= 0. The
    /// old tests never checked orders above n=3 or x below 0.5, which is
    /// exactly where the unstable upward recursion blew up; this sweeps a
    /// wide grid of both so a future regression can't hide in an untested
    /// corner again.
    #[test]
    fn test_boys_bounded_and_monotonic_wide_sweep() {
        for n in 0..=8u32 {
            let f_at_zero = 1.0 / (2 * n + 1) as f64;
            let mut prev = f_at_zero;
            for &x in &[
                0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 24.9, 25.0, 50.0, 100.0,
            ] {
                let f = boys_function(n, x);
                assert!(
                    f > 0.0 && f <= f_at_zero + 1e-12,
                    "F_{n}({x}) = {f} out of bounds (0, {f_at_zero}]"
                );
                assert!(
                    f <= prev + 1e-12,
                    "F_{n}(x) not monotonically decreasing: F_{n}({x})={f} > previous={prev}"
                );
                prev = f;
            }
        }
    }
}
