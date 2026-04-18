// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Inequality primitives (Phase 3A of IMO roadmap)
//!
//! Numerical checkers for classical IMO inequalities. These primitives
//! verify that an inequality holds for *concrete* numerical witnesses —
//! they are **not** symbolic proofs, but they are the fast-path for
//! numerical verification in the ConjectureEngine pipeline and the
//! strategic-guidance layer for Z3-based symbolic proofs.
//!
//! Shipped in Phase 3A:
//! - AM-GM: a₁ + a₂ + ⋯ + aₙ ≥ n·(a₁·a₂·⋯·aₙ)^(1/n)
//! - Cauchy–Schwarz: (Σ aᵢbᵢ)² ≤ (Σ aᵢ²)(Σ bᵢ²)
//! - Power mean inequality: M_p ≤ M_q for p ≤ q
//! - Convexity check on sample points (Jensen's inequality numerically)
//! - Schur-like symmetric-polynomial test for three variables
//!
//! **Deferred to Phase 3B** (not in this commit):
//! - SOS (sum-of-squares) decomposition — needs full polynomial
//!   manipulation layer
//! - Muirhead majorization — needs multi-index symbolic manipulation
//! - Functional equation substitution search — separate module
//!
//! Scope honesty: these are *numerical verification* primitives, not
//! theorem provers. A symbolic proof that AM-GM holds for all positive
//! reals requires Z3 (already in `z3_bridge.rs`) or a hand-crafted
//! induction. What this module provides is the fast numerical check
//! that informs conjecture generation and Z3 strategy selection.

/// Tolerance for numerical inequality comparisons. IMO-scale problems use
/// coefficients bounded by ~10⁶ so 1e-9 gives us ~6 decimal digits of
/// slack above round-off.
pub const INEQ_EPS: f64 = 1e-9;

/// Check that `x` is non-negative within tolerance.
fn nonneg(x: f64) -> bool {
    x > -INEQ_EPS
}

// ─── AM-GM ───────────────────────────────────────────────────────────────────

/// Arithmetic mean of the slice.
pub fn arithmetic_mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Geometric mean of the slice. Requires all entries ≥ 0; panics otherwise
/// because the geometric mean is undefined for negative numbers.
pub fn geometric_mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    for &x in xs {
        assert!(x >= -INEQ_EPS, "geometric_mean: negative input {}", x);
    }
    // Use log-sum for numerical stability on long slices.
    let log_sum: f64 = xs.iter().map(|&x| x.max(0.0).ln()).sum();
    (log_sum / xs.len() as f64).exp()
}

/// Verify AM ≥ GM for non-negative inputs. Returns true if the inequality
/// holds within `INEQ_EPS`, including the equality case (all values equal).
pub fn amgm_holds(xs: &[f64]) -> bool {
    if xs.iter().any(|&x| x < -INEQ_EPS) {
        return false; // AM-GM requires non-negative inputs
    }
    let am = arithmetic_mean(xs);
    let gm = geometric_mean(xs);
    nonneg(am - gm)
}

/// Numerical gap AM − GM (always ≥ 0 for non-negative inputs). A zero gap
/// is the equality case (all values equal).
pub fn amgm_gap(xs: &[f64]) -> f64 {
    arithmetic_mean(xs) - geometric_mean(xs)
}

// ─── Cauchy–Schwarz ──────────────────────────────────────────────────────────

/// Verify the Cauchy–Schwarz inequality for two equal-length slices:
///
///     (Σ aᵢbᵢ)² ≤ (Σ aᵢ²) · (Σ bᵢ²)
///
/// Equality iff the slices are proportional.
pub fn cauchy_schwarz_holds(a: &[f64], b: &[f64]) -> bool {
    assert_eq!(a.len(), b.len(), "Cauchy-Schwarz requires equal-length");
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let a2: f64 = a.iter().map(|x| x * x).sum();
    let b2: f64 = b.iter().map(|x| x * x).sum();
    // (Σab)² ≤ (Σa²)(Σb²)
    nonneg(a2 * b2 - dot * dot)
}

/// Slack in Cauchy–Schwarz: (Σa²)(Σb²) − (Σab)². Always ≥ 0.
pub fn cauchy_schwarz_slack(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let a2: f64 = a.iter().map(|x| x * x).sum();
    let b2: f64 = b.iter().map(|x| x * x).sum();
    a2 * b2 - dot * dot
}

// ─── Power mean inequality ──────────────────────────────────────────────────

/// Power mean of order `p` for non-negative inputs.
///
///     M_p(x) = (1/n · Σ xᵢᵖ)^(1/p)     for p ≠ 0
///     M_0(x) = geometric mean           (limit as p → 0)
///
/// Returns None if p = 0 and any input is 0 (GM is 0), or if p < 0 and any
/// input is 0 (harmonic-style divergence).
pub fn power_mean(xs: &[f64], p: f64) -> Option<f64> {
    if xs.is_empty() {
        return Some(0.0);
    }
    for &x in xs {
        assert!(x >= -INEQ_EPS, "power_mean: negative input");
    }
    if p.abs() < INEQ_EPS {
        // Geometric mean (limit case)
        if xs.iter().any(|&x| x < INEQ_EPS) {
            return Some(0.0);
        }
        return Some(geometric_mean(xs));
    }
    if p < 0.0 && xs.iter().any(|&x| x < INEQ_EPS) {
        return None;
    }
    let sum: f64 = xs.iter().map(|&x| x.max(0.0).powf(p)).sum();
    Some((sum / xs.len() as f64).powf(1.0 / p))
}

/// Verify the power-mean inequality: M_p ≤ M_q for p ≤ q. Returns true if
/// it holds within tolerance on the given sample, false if a counterexample
/// is found.
pub fn power_mean_inequality_holds(xs: &[f64], p: f64, q: f64) -> bool {
    if p > q + INEQ_EPS {
        return false; // caller violated the precondition
    }
    let mp = match power_mean(xs, p) {
        Some(v) => v,
        None => return false,
    };
    let mq = match power_mean(xs, q) {
        Some(v) => v,
        None => return false,
    };
    // We want M_p ≤ M_q, i.e. M_q − M_p ≥ 0.
    nonneg(mq - mp)
}

// ─── Convexity / Jensen ─────────────────────────────────────────────────────

/// Numerical convexity check on sample points: given f and weights wᵢ ≥ 0
/// summing to 1 and sample points xᵢ, verify
///
///     f(Σ wᵢxᵢ) ≤ Σ wᵢf(xᵢ)     (Jensen's inequality for convex f)
///
/// Returns true iff the inequality holds. Requires f to be callable.
pub fn jensen_convex_holds<F>(f: F, weights: &[f64], points: &[f64]) -> bool
where
    F: Fn(f64) -> f64,
{
    assert_eq!(weights.len(), points.len());
    let total: f64 = weights.iter().sum();
    assert!((total - 1.0).abs() < INEQ_EPS, "weights must sum to 1");
    assert!(
        weights.iter().all(|&w| w >= -INEQ_EPS),
        "weights must be non-negative"
    );
    let mean: f64 = weights.iter().zip(points).map(|(w, x)| w * x).sum();
    let f_mean = f(mean);
    let weighted_f: f64 = weights.iter().zip(points).map(|(w, x)| w * f(*x)).sum();
    nonneg(weighted_f - f_mean)
}

// ─── Schur-like ─────────────────────────────────────────────────────────────

/// Schur's inequality for three non-negative reals at exponent t = 1:
///
///     a(a − b)(a − c) + b(b − a)(b − c) + c(c − a)(c − b) ≥ 0
///
/// Holds for all (a, b, c) ≥ 0. Returns true if the numerical evaluation
/// is ≥ 0 within tolerance.
pub fn schur_t1_holds(a: f64, b: f64, c: f64) -> bool {
    assert!(a >= -INEQ_EPS && b >= -INEQ_EPS && c >= -INEQ_EPS);
    let s = a * (a - b) * (a - c) + b * (b - a) * (b - c) + c * (c - a) * (c - b);
    nonneg(s)
}

/// Schur's inequality at exponent t = 2:
///
///     a²(a − b)(a − c) + b²(b − a)(b − c) + c²(c − a)(c − b) ≥ 0
pub fn schur_t2_holds(a: f64, b: f64, c: f64) -> bool {
    assert!(a >= -INEQ_EPS && b >= -INEQ_EPS && c >= -INEQ_EPS);
    let s = a * a * (a - b) * (a - c)
        + b * b * (b - a) * (b - c)
        + c * c * (c - a) * (c - b);
    nonneg(s)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amgm_two_elements() {
        // a + b ≥ 2√(ab). For a=1, b=4: AM = 2.5, GM = 2.
        assert!(amgm_holds(&[1.0, 4.0]));
        assert!((amgm_gap(&[1.0, 4.0]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_amgm_equality_case() {
        // All equal → AM = GM = 3.
        assert!(amgm_holds(&[3.0, 3.0, 3.0, 3.0]));
        assert!(amgm_gap(&[3.0, 3.0, 3.0, 3.0]).abs() < 1e-9);
    }

    #[test]
    fn test_amgm_random_samples() {
        // Stress: many random positive samples should always satisfy AM-GM.
        let samples = [
            vec![0.1, 0.2, 0.3, 0.4],
            vec![1.5, 2.5, 3.5],
            vec![10.0, 20.0],
            vec![0.5, 0.5, 0.5, 0.5, 2.0],
            vec![100.0, 1.0, 1.0],
        ];
        for s in &samples {
            assert!(amgm_holds(s), "AM-GM failed on {:?}", s);
        }
    }

    #[test]
    fn test_amgm_rejects_negative() {
        assert!(!amgm_holds(&[1.0, -1.0]));
    }

    #[test]
    fn test_cauchy_schwarz_holds() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!(cauchy_schwarz_holds(&a, &b));
        // Slack: (1+4+9)(16+25+36) − (4+10+18)² = 14·77 − 32² = 1078 − 1024 = 54
        assert!((cauchy_schwarz_slack(&a, &b) - 54.0).abs() < 1e-9);
    }

    #[test]
    fn test_cauchy_schwarz_equality_case() {
        // Proportional vectors: equality holds.
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0];
        assert!(cauchy_schwarz_holds(&a, &b));
        assert!(cauchy_schwarz_slack(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn test_cauchy_schwarz_with_zeros() {
        let a = [0.0, 0.0];
        let b = [1.0, 2.0];
        assert!(cauchy_schwarz_holds(&a, &b));
    }

    #[test]
    fn test_power_mean_am_gm_hm_order() {
        // For positives: HM ≤ GM ≤ AM ≤ QM (power means at p = -1, 0, 1, 2)
        let xs = vec![1.0, 2.0, 4.0];
        let hm = power_mean(&xs, -1.0).unwrap();
        let gm = power_mean(&xs, 0.0).unwrap();
        let am = power_mean(&xs, 1.0).unwrap();
        let qm = power_mean(&xs, 2.0).unwrap();
        assert!(hm <= gm + 1e-9);
        assert!(gm <= am + 1e-9);
        assert!(am <= qm + 1e-9);
    }

    #[test]
    fn test_power_mean_inequality_holds() {
        let xs = vec![1.0, 2.0, 4.0, 8.0];
        for &(p, q) in &[(-1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (-1.0, 2.0)] {
            assert!(
                power_mean_inequality_holds(&xs, p, q),
                "power mean inequality failed for p={}, q={}",
                p,
                q
            );
        }
    }

    #[test]
    fn test_jensen_convex_x_squared() {
        // f(x) = x² is convex; Jensen holds.
        let weights = [0.3, 0.3, 0.4];
        let points = [1.0, 2.0, 3.0];
        assert!(jensen_convex_holds(|x| x * x, &weights, &points));
    }

    #[test]
    fn test_jensen_convex_exp() {
        let weights = [0.5, 0.5];
        let points = [0.0, 1.0];
        assert!(jensen_convex_holds(f64::exp, &weights, &points));
    }

    #[test]
    fn test_jensen_concave_fails() {
        // f(x) = -x² is concave → Jensen is REVERSED, so the convex
        // form f(Σwx) ≤ Σwf(x) should FAIL (except equality cases).
        let weights = [0.3, 0.3, 0.4];
        let points = [1.0, 2.0, 3.0];
        assert!(!jensen_convex_holds(|x| -x * x, &weights, &points));
    }

    #[test]
    fn test_schur_t1_positive_cases() {
        // Schur holds for all non-negative triples.
        for &(a, b, c) in &[
            (1.0, 1.0, 1.0),
            (1.0, 2.0, 3.0),
            (0.5, 0.5, 0.5),
            (0.0, 1.0, 2.0),
            (10.0, 1.0, 1.0),
            (1.0, 1.0, 10.0),
        ] {
            assert!(schur_t1_holds(a, b, c), "Schur t=1 failed on ({}, {}, {})", a, b, c);
            assert!(schur_t2_holds(a, b, c), "Schur t=2 failed on ({}, {}, {})", a, b, c);
        }
    }

    #[test]
    fn test_schur_equality_case() {
        // All equal: Schur is tight at 0.
        let a: f64 = 2.5;
        let s: f64 =
            a * (a - a) * (a - a) + a * (a - a) * (a - a) + a * (a - a) * (a - a);
        assert!(s.abs() < 1e-9);
        assert!(schur_t1_holds(a, a, a));
    }
}
