// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Functional equation primitives (Phase 3C of IMO roadmap)
//!
//! Numerical classification of canonical functional equations on the
//! reals. Given a function as **sample pairs** `(x, f(x))` the module
//! decides which of the standard IMO functional-equation families fits
//! best, with a confidence score derived from residual error.
//!
//! ## Shipped families
//!
//! - **Cauchy / additive**: `f(x + y) = f(x) + f(y)` ⇒ continuous
//!   solutions are `f(x) = c·x` (Cauchy 1821; Hamel basis solutions are
//!   pathological and not detected here)
//! - **Multiplicative**: `f(x·y) = f(x)·f(y)` on the positive reals ⇒
//!   continuous solutions are `f(x) = x^c`
//! - **Exponential**: `f(x + y) = f(x)·f(y)` ⇒ continuous solutions are
//!   `f(x) = a^x` for `a = f(1)`
//! - **Logarithmic**: `f(x·y) = f(x) + f(y)` on positives ⇒ continuous
//!   solutions are `f(x) = c·log_b(x)`
//! - **Constant**: `f` near-constant within tolerance
//! - **Identity**: `f(x) = x` (special case of Cauchy with `c = 1`)
//!
//! ## What this is NOT
//!
//! - This is **classification + verification**, not a theorem prover.
//!   "All functions satisfying f(xy) = f(x)f(y)" requires Z3 / Lean for
//!   a real proof. We give numerical evidence + the canonical fitted
//!   form.
//! - Pathological (non-measurable, Hamel-basis) solutions to Cauchy are
//!   not handled — we assume continuity.
//! - Functional equations with constraints (`f: R⁺ → R⁺`, monotone
//!   only, etc.) are validated post-hoc, not encoded in the search.
//!
//! ## Why this matters for IMO
//!
//! Roughly one IMO problem per year is "find all `f: R → R` such that
//! ...". The hard part for a numerical engine is going from samples
//! back to the canonical closed form. This module short-circuits that:
//! given enough sample pairs from any of the five canonical families,
//! it returns the family + the fitted constant + a confidence score.

/// Tolerance for residual checks. Same scale as `inequalities::INEQ_EPS`.
pub const FE_EPS: f64 = 1e-9;

/// A canonical functional-equation family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EquationKind {
    /// `f(x) = c` (constant).
    Constant,
    /// `f(x) = x` (identity, special Cauchy with c=1).
    Identity,
    /// `f(x + y) = f(x) + f(y)` ⇒ `f(x) = c·x` (continuous case).
    CauchyAdditive,
    /// `f(x·y) = f(x)·f(y)` on positive reals ⇒ `f(x) = x^c`.
    Multiplicative,
    /// `f(x + y) = f(x)·f(y)` ⇒ `f(x) = a^x`.
    Exponential,
    /// `f(x·y) = f(x) + f(y)` on positives ⇒ `f(x) = c·log_b(x)`.
    Logarithmic,
    /// None of the above fits within tolerance.
    Unknown,
}

impl EquationKind {
    /// Human-readable canonical-form description.
    pub fn canonical_form(self) -> &'static str {
        match self {
            EquationKind::Constant => "f(x) = c",
            EquationKind::Identity => "f(x) = x",
            EquationKind::CauchyAdditive => "f(x) = c·x  [from f(x+y)=f(x)+f(y)]",
            EquationKind::Multiplicative => "f(x) = x^c  [from f(xy)=f(x)f(y)]",
            EquationKind::Exponential => "f(x) = a^x  [from f(x+y)=f(x)f(y)]",
            EquationKind::Logarithmic => "f(x) = c·log(x)  [from f(xy)=f(x)+f(y)]",
            EquationKind::Unknown => "(unknown / no canonical fit)",
        }
    }
}

/// Result of a classification attempt: the kind, the fitted constant
/// (interpretation depends on `kind`), and a confidence score in [0, 1].
#[derive(Debug, Clone, Copy)]
pub struct Classification {
    pub kind: EquationKind,
    /// The fitted constant: `c` for Cauchy/Multiplicative/Logarithmic,
    /// the base `a` for Exponential, the constant value for Constant,
    /// 1.0 for Identity, NaN for Unknown.
    pub constant: f64,
    /// Confidence in [0, 1]. 1.0 = perfect fit (max residual < FE_EPS),
    /// 0.0 = poor fit (max residual ≥ tolerance band). Derived from
    /// `1 - clamp(max_residual / tol_band, 0, 1)`.
    pub confidence: f64,
}

impl Classification {
    fn unknown() -> Self {
        Self {
            kind: EquationKind::Unknown,
            constant: f64::NAN,
            confidence: 0.0,
        }
    }
}

// ─── Verification primitives ─────────────────────────────────────────────────

/// Verify Cauchy additivity `f(x + y) ≈ f(x) + f(y)` on a set of
/// sample pairs `(x, fx)`. For each pair `(i, j)` checks whether
/// `f(xᵢ + xⱼ)` was sampled and matches. Pairs whose sum is not in
/// the sample set are skipped, so the verifier needs a sample set
/// closed under at least some pairwise sums to give a non-trivial
/// answer (typically: include `0`, the inputs, and their pairwise sums).
///
/// Returns the maximum absolute residual, or `0.0` if no pairs were
/// testable (meaning the verifier is silent, not "passed").
pub fn cauchy_residual(samples: &[(f64, f64)]) -> f64 {
    let mut max_resid = 0.0f64;
    let mut tested = 0usize;
    for i in 0..samples.len() {
        for j in i..samples.len() {
            let (xi, fi) = samples[i];
            let (xj, fj) = samples[j];
            let target = xi + xj;
            if let Some(&(_, ftarget)) = samples.iter().find(|(x, _)| (x - target).abs() < FE_EPS)
            {
                let resid = (ftarget - (fi + fj)).abs();
                if resid > max_resid {
                    max_resid = resid;
                }
                tested += 1;
            }
        }
    }
    if tested == 0 {
        0.0
    } else {
        max_resid
    }
}

/// Verify multiplicativity `f(x·y) ≈ f(x)·f(y)` on positive samples.
pub fn multiplicative_residual(samples: &[(f64, f64)]) -> f64 {
    let mut max_resid = 0.0f64;
    let mut tested = 0usize;
    for i in 0..samples.len() {
        for j in i..samples.len() {
            let (xi, fi) = samples[i];
            let (xj, fj) = samples[j];
            if xi <= 0.0 || xj <= 0.0 {
                continue;
            }
            let target = xi * xj;
            if let Some(&(_, ftarget)) = samples.iter().find(|(x, _)| (x - target).abs() < FE_EPS)
            {
                let resid = (ftarget - (fi * fj)).abs();
                if resid > max_resid {
                    max_resid = resid;
                }
                tested += 1;
            }
        }
    }
    if tested == 0 {
        0.0
    } else {
        max_resid
    }
}

/// Verify exponentiation law `f(x + y) ≈ f(x)·f(y)`.
pub fn exponential_residual(samples: &[(f64, f64)]) -> f64 {
    let mut max_resid = 0.0f64;
    let mut tested = 0usize;
    for i in 0..samples.len() {
        for j in i..samples.len() {
            let (xi, fi) = samples[i];
            let (xj, fj) = samples[j];
            let target = xi + xj;
            if let Some(&(_, ftarget)) = samples.iter().find(|(x, _)| (x - target).abs() < FE_EPS)
            {
                let resid = (ftarget - (fi * fj)).abs();
                if resid > max_resid {
                    max_resid = resid;
                }
                tested += 1;
            }
        }
    }
    if tested == 0 {
        0.0
    } else {
        max_resid
    }
}

/// Verify the log law `f(x·y) ≈ f(x) + f(y)` on positives.
pub fn logarithmic_residual(samples: &[(f64, f64)]) -> f64 {
    let mut max_resid = 0.0f64;
    let mut tested = 0usize;
    for i in 0..samples.len() {
        for j in i..samples.len() {
            let (xi, fi) = samples[i];
            let (xj, fj) = samples[j];
            if xi <= 0.0 || xj <= 0.0 {
                continue;
            }
            let target = xi * xj;
            if let Some(&(_, ftarget)) = samples.iter().find(|(x, _)| (x - target).abs() < FE_EPS)
            {
                let resid = (ftarget - (fi + fj)).abs();
                if resid > max_resid {
                    max_resid = resid;
                }
                tested += 1;
            }
        }
    }
    if tested == 0 {
        0.0
    } else {
        max_resid
    }
}

// ─── Classification ──────────────────────────────────────────────────────────

/// Fit `f(x) = c·x` to samples by least-squares (forced through origin).
/// Returns the slope and the max absolute residual on the sample set.
fn fit_linear_through_origin(samples: &[(f64, f64)]) -> (f64, f64) {
    let num: f64 = samples.iter().map(|(x, y)| x * y).sum();
    let den: f64 = samples.iter().map(|(x, _)| x * x).sum();
    if den.abs() < FE_EPS {
        return (0.0, 0.0);
    }
    let c = num / den;
    let max_resid = samples
        .iter()
        .map(|(x, y)| (y - c * x).abs())
        .fold(0.0f64, f64::max);
    (c, max_resid)
}

/// Fit `f(x) = x^c` to positive samples by linear regression on
/// `log f = c·log x`. Returns `(c, max_residual_on_original_scale)`.
fn fit_power_law(samples: &[(f64, f64)]) -> Option<(f64, f64)> {
    let positive: Vec<(f64, f64)> = samples
        .iter()
        .copied()
        .filter(|(x, y)| *x > 0.0 && *y > 0.0)
        .collect();
    if positive.len() < 2 {
        return None;
    }
    let log_pairs: Vec<(f64, f64)> = positive.iter().map(|(x, y)| (x.ln(), y.ln())).collect();
    let (c, _) = fit_linear_through_origin(&log_pairs);
    let max_resid = positive
        .iter()
        .map(|(x, y)| (y - x.powf(c)).abs())
        .fold(0.0f64, f64::max);
    Some((c, max_resid))
}

/// Fit `f(x) = a^x` to samples by linear regression on `log f = x·log a`.
/// Requires all `f(x) > 0`.
fn fit_exponential(samples: &[(f64, f64)]) -> Option<(f64, f64)> {
    if samples.iter().any(|(_, y)| *y <= 0.0) {
        return None;
    }
    let log_pairs: Vec<(f64, f64)> = samples.iter().map(|(x, y)| (*x, y.ln())).collect();
    let (log_a, _) = fit_linear_through_origin(&log_pairs);
    let a = log_a.exp();
    let max_resid = samples
        .iter()
        .map(|(x, y)| (y - a.powf(*x)).abs())
        .fold(0.0f64, f64::max);
    Some((a, max_resid))
}

/// Fit `f(x) = c·log(x)` to positive samples (natural log) by least-squares
/// through origin in `(log x, f)` space.
fn fit_logarithmic(samples: &[(f64, f64)]) -> Option<(f64, f64)> {
    let positive: Vec<(f64, f64)> = samples
        .iter()
        .copied()
        .filter(|(x, _)| *x > 0.0)
        .collect();
    if positive.len() < 2 {
        return None;
    }
    let log_x_pairs: Vec<(f64, f64)> = positive.iter().map(|(x, y)| (x.ln(), *y)).collect();
    let (c, _) = fit_linear_through_origin(&log_x_pairs);
    let max_resid = positive
        .iter()
        .map(|(x, y)| (y - c * x.ln()).abs())
        .fold(0.0f64, f64::max);
    Some((c, max_resid))
}

/// Convert a max residual to a confidence in [0, 1] given a tolerance
/// band derived from the sample magnitude.
fn residual_to_confidence(max_resid: f64, scale: f64) -> f64 {
    let tol_band = (scale.abs() * 1e-3).max(1e-6);
    (1.0 - (max_resid / tol_band).min(1.0)).max(0.0)
}

/// Return the typical magnitude of the y-values in the sample set
/// (used as the scale for the confidence band).
fn y_scale(samples: &[(f64, f64)]) -> f64 {
    samples.iter().map(|(_, y)| y.abs()).fold(0.0f64, f64::max)
}

/// Try to classify a sampled function into one of the canonical
/// functional-equation families. Returns the **best fit** by confidence,
/// or `Unknown` if no family clears the minimum confidence threshold.
pub fn classify(samples: &[(f64, f64)]) -> Classification {
    if samples.is_empty() {
        return Classification::unknown();
    }
    let scale = y_scale(samples).max(1.0);

    // Constant check: max y minus min y near zero.
    let y_min = samples.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = samples
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    if (y_max - y_min).abs() < FE_EPS * scale.max(1.0) {
        return Classification {
            kind: EquationKind::Constant,
            constant: y_min,
            confidence: 1.0,
        };
    }

    let mut best = Classification::unknown();

    // Cauchy / linear-through-origin
    let (c_lin, resid_lin) = fit_linear_through_origin(samples);
    let conf_lin = residual_to_confidence(resid_lin, scale);
    if conf_lin > best.confidence {
        let kind = if (c_lin - 1.0).abs() < 1e-6 {
            EquationKind::Identity
        } else {
            EquationKind::CauchyAdditive
        };
        best = Classification {
            kind,
            constant: c_lin,
            confidence: conf_lin,
        };
    }

    // Multiplicative / power law
    if let Some((c_pow, resid_pow)) = fit_power_law(samples) {
        let conf_pow = residual_to_confidence(resid_pow, scale);
        if conf_pow > best.confidence {
            best = Classification {
                kind: EquationKind::Multiplicative,
                constant: c_pow,
                confidence: conf_pow,
            };
        }
    }

    // Exponential
    if let Some((a_exp, resid_exp)) = fit_exponential(samples) {
        let conf_exp = residual_to_confidence(resid_exp, scale);
        if conf_exp > best.confidence {
            best = Classification {
                kind: EquationKind::Exponential,
                constant: a_exp,
                confidence: conf_exp,
            };
        }
    }

    // Logarithmic
    if let Some((c_log, resid_log)) = fit_logarithmic(samples) {
        let conf_log = residual_to_confidence(resid_log, scale);
        if conf_log > best.confidence {
            best = Classification {
                kind: EquationKind::Logarithmic,
                constant: c_log,
                confidence: conf_log,
            };
        }
    }

    if best.confidence < 0.5 {
        return Classification::unknown();
    }
    best
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    /// Sample a closed-under-some-sums set: 0, 1, 2, 3, 4, 5, 6.
    fn sum_closed_grid(f: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
        (0..=6).map(|i| (i as f64, f(i as f64))).collect()
    }

    /// Sample a product-closed positive set: 1, 2, 3, 4, 6, 8, 12, 24.
    fn product_closed_grid(f: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
        [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0]
            .iter()
            .map(|x| (*x, f(*x)))
            .collect()
    }

    // ── Verification primitives ──

    #[test]
    fn test_cauchy_residual_linear_passes() {
        let s = sum_closed_grid(|x| 3.0 * x);
        assert!(cauchy_residual(&s) < FE_EPS);
    }

    #[test]
    fn test_cauchy_residual_quadratic_fails() {
        let s = sum_closed_grid(|x| x * x);
        // f(1+1)=4 vs f(1)+f(1)=2 ⇒ residual 2.0
        assert!(cauchy_residual(&s) > 1.0);
    }

    #[test]
    fn test_multiplicative_residual_power_passes() {
        let s = product_closed_grid(|x| x.powi(2));
        assert!(multiplicative_residual(&s) < 1e-9);
    }

    #[test]
    fn test_multiplicative_residual_linear_fails() {
        let s = product_closed_grid(|x| 2.0 * x);
        // f(2*3)=12 vs f(2)*f(3)=24 ⇒ residual 12
        assert!(multiplicative_residual(&s) > 1.0);
    }

    #[test]
    fn test_exponential_residual_2pow_passes() {
        let s = sum_closed_grid(|x| 2f64.powf(x));
        assert!(exponential_residual(&s) < 1e-9);
    }

    #[test]
    fn test_logarithmic_residual_log_passes() {
        let s = product_closed_grid(|x| x.ln());
        assert!(logarithmic_residual(&s) < 1e-9);
    }

    // ── Classification ──

    #[test]
    fn test_classify_identity() {
        let s = sum_closed_grid(|x| x);
        let c = classify(&s);
        assert_eq!(c.kind, EquationKind::Identity);
        assert!(approx(c.constant, 1.0, 1e-9));
        assert!(c.confidence > 0.999);
    }

    #[test]
    fn test_classify_cauchy_linear() {
        let s = sum_closed_grid(|x| 3.0 * x);
        let c = classify(&s);
        assert_eq!(c.kind, EquationKind::CauchyAdditive);
        assert!(approx(c.constant, 3.0, 1e-9));
        assert!(c.confidence > 0.999);
    }

    #[test]
    fn test_classify_constant() {
        let s = sum_closed_grid(|_| 7.5);
        let c = classify(&s);
        assert_eq!(c.kind, EquationKind::Constant);
        assert!(approx(c.constant, 7.5, 1e-9));
        assert!(c.confidence > 0.999);
    }

    #[test]
    fn test_classify_power_law() {
        // f(x) = x^2. Linear-through-origin would fit a slope but with
        // huge residual (4 vs 8 at x=2 vs x=4 etc.), so power law wins.
        let s = product_closed_grid(|x| x.powi(2));
        let c = classify(&s);
        assert_eq!(c.kind, EquationKind::Multiplicative);
        assert!(approx(c.constant, 2.0, 1e-6));
        assert!(c.confidence > 0.999);
    }

    #[test]
    fn test_classify_exponential() {
        let s = sum_closed_grid(|x| 2f64.powf(x));
        let c = classify(&s);
        assert_eq!(c.kind, EquationKind::Exponential);
        assert!(approx(c.constant, 2.0, 1e-6));
        assert!(c.confidence > 0.999);
    }

    #[test]
    fn test_classify_logarithmic() {
        let s = product_closed_grid(|x| 2.0 * x.ln());
        let c = classify(&s);
        assert_eq!(c.kind, EquationKind::Logarithmic);
        assert!(approx(c.constant, 2.0, 1e-6));
        assert!(c.confidence > 0.999);
    }

    #[test]
    fn test_classify_unknown_for_quadratic() {
        // f(x) = x² + 1 doesn't fit any of the canonical families
        // perfectly: linear-through-origin has huge residual, power
        // law fails because x=0 is excluded, exp doesn't fit, log
        // doesn't fit. Should land at Unknown (or at most very low conf).
        let s: Vec<(f64, f64)> = (1..=6).map(|i| (i as f64, (i * i + 1) as f64)).collect();
        let c = classify(&s);
        // Either Unknown OR a low-confidence fit. Confidence < 0.5
        // would have been clamped to Unknown. Allow the exact "Unknown"
        // outcome here.
        assert_eq!(
            c.kind,
            EquationKind::Unknown,
            "x²+1 should not fit a canonical functional-equation family, got {:?} c={} conf={}",
            c.kind,
            c.constant,
            c.confidence
        );
    }

    #[test]
    fn test_canonical_form_strings_present() {
        for k in [
            EquationKind::Constant,
            EquationKind::Identity,
            EquationKind::CauchyAdditive,
            EquationKind::Multiplicative,
            EquationKind::Exponential,
            EquationKind::Logarithmic,
            EquationKind::Unknown,
        ] {
            assert!(!k.canonical_form().is_empty());
        }
    }
}
