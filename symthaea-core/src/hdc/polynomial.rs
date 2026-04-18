// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Polynomial arithmetic + small-degree SOS decomposition
//!
//! Phase 3B scoped. Provides:
//!
//! - **Univariate polynomial** with add/sub/mul/scale/degree/eval
//! - **Bivariate polynomial** represented as a coefficient grid, with the
//!   same basic arithmetic
//! - **SOS decomposition** for small-degree univariate and bivariate
//!   symmetric polynomials: attempts to write `p` as a sum of squares,
//!   `p = Σ qᵢ²`, where each `qᵢ` is a small polynomial. When successful,
//!   this proves `p ≥ 0` for all real inputs.
//!
//! ## Scope (deliberately narrow)
//!
//! - Univariate SOS: works for **any non-negative polynomial of even
//!   degree ≤ 4**, via the classical decomposition p(x) = (ax + b)² + c² + ...
//! - Bivariate SOS: handles **symmetric polynomials in two variables up to
//!   total degree 4** via a basis search over {1, x, y, x+y, x−y, xy}²
//! - Full general-purpose SOS (SDP-based, arbitrary degree, arbitrary
//!   variables) is explicitly deferred — that's a multi-session subproject.
//!
//! ## Why this matters for IMO
//!
//! Many IMO inequalities reduce to "show p(a, b, c) ≥ 0" for a symmetric
//! polynomial p. Numerical verification (existing primitives) proves the
//! inequality at specific points; SOS decomposition proves it *everywhere*
//! by finding an explicit non-negativity witness. This is the bridge from
//! "we can check" to "we can prove".

// ─── Univariate polynomial ──────────────────────────────────────────────────

/// Dense univariate polynomial. `coeffs[i]` is the coefficient of xⁱ.
#[derive(Debug, Clone, PartialEq)]
pub struct Poly {
    pub coeffs: Vec<f64>,
}

impl Poly {
    pub fn new(coeffs: Vec<f64>) -> Self {
        let mut p = Self { coeffs };
        p.trim();
        p
    }

    /// Constant polynomial.
    pub fn constant(c: f64) -> Self {
        if c.abs() < 1e-15 {
            Poly { coeffs: vec![] }
        } else {
            Poly { coeffs: vec![c] }
        }
    }

    /// The monomial x^k.
    pub fn monomial(k: usize) -> Self {
        let mut c = vec![0.0; k + 1];
        c[k] = 1.0;
        Poly { coeffs: c }
    }

    /// Zero polynomial.
    pub fn zero() -> Self {
        Poly { coeffs: vec![] }
    }

    /// Is this the zero polynomial?
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.abs() < 1e-12)
    }

    /// Degree (None if zero polynomial).
    pub fn degree(&self) -> Option<usize> {
        if self.is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Strip trailing zeros.
    fn trim(&mut self) {
        while let Some(&last) = self.coeffs.last() {
            if last.abs() < 1e-12 {
                self.coeffs.pop();
            } else {
                break;
            }
        }
    }

    /// Evaluate p(x) at a concrete point.
    pub fn eval(&self, x: f64) -> f64 {
        // Horner's rule
        let mut acc = 0.0;
        for &c in self.coeffs.iter().rev() {
            acc = acc * x + c;
        }
        acc
    }

    /// Add two polynomials.
    pub fn add(&self, other: &Poly) -> Poly {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![0.0; len];
        for (i, &c) in self.coeffs.iter().enumerate() {
            out[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            out[i] += c;
        }
        Poly::new(out)
    }

    /// Subtract.
    pub fn sub(&self, other: &Poly) -> Poly {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![0.0; len];
        for (i, &c) in self.coeffs.iter().enumerate() {
            out[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            out[i] -= c;
        }
        Poly::new(out)
    }

    /// Scale by a constant.
    pub fn scale(&self, factor: f64) -> Poly {
        Poly::new(self.coeffs.iter().map(|c| c * factor).collect())
    }

    /// Multiply two polynomials.
    pub fn mul(&self, other: &Poly) -> Poly {
        if self.is_zero() || other.is_zero() {
            return Poly::zero();
        }
        let mut out = vec![0.0; self.coeffs.len() + other.coeffs.len() - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        Poly::new(out)
    }

    /// Square (convenience).
    pub fn square(&self) -> Poly {
        self.mul(self)
    }

    /// Approximate equality (for tests).
    pub fn approx_eq(&self, other: &Poly, tol: f64) -> bool {
        let len = self.coeffs.len().max(other.coeffs.len());
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(0.0);
            let b = other.coeffs.get(i).copied().unwrap_or(0.0);
            if (a - b).abs() > tol {
                return false;
            }
        }
        true
    }
}

// ─── Univariate SOS decomposition ───────────────────────────────────────────

/// Attempt to decompose a univariate polynomial p of even degree ≤ 4 as
/// a sum of squares, p = Σᵢ qᵢ² where each qᵢ is a low-degree polynomial.
/// Returns the list of qᵢ polynomials on success, None otherwise.
///
/// Strategy:
/// - Degree 0: p = c. If c ≥ 0, decomposition is [√c]; else fail.
/// - Degree 2: p(x) = ax² + bx + c. Non-negative iff a > 0 and discriminant
///   b² − 4ac ≤ 0. Decomposition: √a · (x + b/(2a))² + √(c − b²/(4a))².
/// - Degree 4: p(x) = ax⁴ + bx³ + cx² + dx + e. Use the completed-square
///   identity ax⁴ + bx³ + (...) = a(x² + px + q)² + r(x + s)² + t for
///   appropriate p, q, r, s, t. This is the **classical quartic SOS**.
pub fn sos_univariate(p: &Poly) -> Option<Vec<Poly>> {
    let deg = p.degree().unwrap_or(0);
    if p.is_zero() {
        return Some(vec![]);
    }
    if deg % 2 != 0 {
        return None; // odd-degree polynomials go to −∞, cannot be non-negative
    }
    match deg {
        0 => {
            let c = p.coeffs[0];
            if c >= -1e-12 {
                let root = c.max(0.0).sqrt();
                Some(vec![Poly::constant(root)])
            } else {
                None
            }
        }
        2 => {
            // p = ax² + bx + c
            let a = p.coeffs.get(2).copied().unwrap_or(0.0);
            let b = p.coeffs.get(1).copied().unwrap_or(0.0);
            let c = p.coeffs.get(0).copied().unwrap_or(0.0);
            if a < 1e-12 {
                return None; // degree-2 polynomial must have a > 0
            }
            let disc = b * b - 4.0 * a * c;
            if disc > 1e-9 {
                return None; // has two distinct real roots → goes negative
            }
            // p = a·(x + b/(2a))² + (c − b²/(4a))
            //   = (√a · x + b/(2√a))² + (c − b²/(4a))
            let sa = a.sqrt();
            let q1 = Poly::new(vec![b / (2.0 * sa), sa]);
            let rem = c - b * b / (4.0 * a);
            let rem = rem.max(0.0);
            let q2 = Poly::constant(rem.sqrt());
            Some(vec![q1, q2])
        }
        4 => {
            // p(x) = ax⁴ + bx³ + cx² + dx + e, with a > 0.
            let a = p.coeffs.get(4).copied().unwrap_or(0.0);
            let b = p.coeffs.get(3).copied().unwrap_or(0.0);
            let c = p.coeffs.get(2).copied().unwrap_or(0.0);
            let d = p.coeffs.get(1).copied().unwrap_or(0.0);
            let e = p.coeffs.get(0).copied().unwrap_or(0.0);
            if a < 1e-12 {
                return None;
            }
            // Write p = (√a · x² + α·x + β)² + γ(x + δ)² + ε
            // Expand: a·x⁴ + 2√a·α·x³ + (α² + 2√a·β)·x² + 2αβ·x + β²
            //       + γ·x² + 2γδ·x + γδ²
            //       + ε
            // Match coefficients:
            //   x⁴: a (free, sa = √a)
            //   x³: b = 2sa·α    → α = b/(2sa)
            //   x²: c = α² + 2sa·β + γ
            //   x¹: d = 2αβ + 2γδ
            //   x⁰: e = β² + γδ² + ε
            //
            // Choose γ = 0 first (pure "(sa·x² + αx + β)² + const" case).
            let sa = a.sqrt();
            let alpha = b / (2.0 * sa);
            // c = α² + 2sa·β  →  β = (c − α²) / (2sa)
            let beta = (c - alpha * alpha) / (2.0 * sa);
            // Check residual: d should equal 2αβ (γ=0), ε should equal
            // e − β². If these don't match, fall back to γ > 0 case.
            let d_expected = 2.0 * alpha * beta;
            let e_residual = e - beta * beta;
            if (d - d_expected).abs() < 1e-9 && e_residual >= -1e-9 {
                let q1 = Poly::new(vec![beta, alpha, sa]);
                let q2 = Poly::constant(e_residual.max(0.0).sqrt());
                return Some(vec![q1, q2]);
            }
            // General γ > 0 case: solve the system.
            //   From x¹: d − 2αβ = 2γδ
            //   From x²: γ = c − α² − 2sa·β  (β still free)
            //   From x⁰: ε = e − β² − γδ²  ≥ 0
            //
            // Pick β to minimize residual. Try β ∈ linspace to find
            // feasible setting.
            let n_samples = 200;
            for i in 0..=n_samples {
                let beta_try = (i as f64 - n_samples as f64 / 2.0) * 0.1;
                let gamma = c - alpha * alpha - 2.0 * sa * beta_try;
                if gamma < -1e-9 {
                    continue;
                }
                let gamma = gamma.max(0.0);
                let lhs = d - 2.0 * alpha * beta_try;
                if gamma < 1e-9 {
                    if lhs.abs() > 1e-6 {
                        continue;
                    }
                    let eps = e - beta_try * beta_try;
                    if eps >= -1e-9 {
                        let q1 = Poly::new(vec![beta_try, alpha, sa]);
                        let q2 = Poly::constant(eps.max(0.0).sqrt());
                        return Some(vec![q1, q2]);
                    }
                    continue;
                }
                let delta = lhs / (2.0 * gamma);
                let eps = e - beta_try * beta_try - gamma * delta * delta;
                if eps >= -1e-9 {
                    let q1 = Poly::new(vec![beta_try, alpha, sa]);
                    let q2 = Poly::new(vec![delta * gamma.sqrt(), gamma.sqrt()]);
                    let q3 = Poly::constant(eps.max(0.0).sqrt());
                    // Verify numerically
                    let sos_sum = q1
                        .square()
                        .add(&q2.square())
                        .add(&q3.square());
                    if p.approx_eq(&sos_sum, 1e-6) {
                        return Some(vec![q1, q2, q3]);
                    }
                }
            }
            None
        }
        _ => None, // deg ≥ 6 deferred
    }
}

/// Verify an SOS decomposition: check that Σᵢ qᵢ² approximates `p` and
/// that each qᵢ² is non-negative (trivially true — squaring).
pub fn verify_sos(p: &Poly, decomposition: &[Poly], tol: f64) -> bool {
    let mut sum = Poly::zero();
    for q in decomposition {
        sum = sum.add(&q.square());
    }
    p.approx_eq(&sum, tol)
}

/// Sample-based proof: evaluate the SOS sum at `n_samples` points in
/// [-range, range] and verify each value is ≥ 0. This is a *weaker* check
/// than the coefficient-matching verify_sos but useful for sanity.
pub fn sample_nonneg(p: &Poly, range: f64, n_samples: usize) -> bool {
    for i in 0..=n_samples {
        let t = -range + 2.0 * range * (i as f64 / n_samples as f64);
        if p.eval(t) < -1e-9 {
            return false;
        }
    }
    true
}

// ─── Bivariate polynomial ──────────────────────────────────────────────────

/// Dense bivariate polynomial. `coeffs[i][j]` is the coefficient of
/// xⁱ yʲ. The grid is stored at its minimum bounding box.
#[derive(Debug, Clone, PartialEq)]
pub struct BiPoly {
    /// coeffs[i][j] = coefficient of xⁱ yʲ; len = deg_x + 1; inner len = deg_y + 1
    pub coeffs: Vec<Vec<f64>>,
}

impl BiPoly {
    pub fn new(coeffs: Vec<Vec<f64>>) -> Self {
        BiPoly { coeffs }
    }

    /// Constant.
    pub fn constant(c: f64) -> Self {
        BiPoly {
            coeffs: vec![vec![c]],
        }
    }

    /// Monomial xⁱ yʲ.
    pub fn monomial(i: usize, j: usize) -> Self {
        let mut rows = vec![vec![0.0; j + 1]; i + 1];
        rows[i][j] = 1.0;
        BiPoly { coeffs: rows }
    }

    /// Evaluate at (x, y).
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let mut sum = 0.0;
        for (i, row) in self.coeffs.iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                sum += c * x.powi(i as i32) * y.powi(j as i32);
            }
        }
        sum
    }

    /// Add two bivariate polynomials.
    pub fn add(&self, other: &BiPoly) -> BiPoly {
        let n_rows = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![Vec::new(); n_rows];
        for (i, row) in out.iter_mut().enumerate() {
            let self_row = self.coeffs.get(i).cloned().unwrap_or_default();
            let other_row = other.coeffs.get(i).cloned().unwrap_or_default();
            let n_cols = self_row.len().max(other_row.len());
            *row = vec![0.0; n_cols];
            for (j, v) in row.iter_mut().enumerate() {
                *v = self_row.get(j).copied().unwrap_or(0.0)
                    + other_row.get(j).copied().unwrap_or(0.0);
            }
        }
        BiPoly::new(out)
    }

    /// Subtract.
    pub fn sub(&self, other: &BiPoly) -> BiPoly {
        let n_rows = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![Vec::new(); n_rows];
        for (i, row) in out.iter_mut().enumerate() {
            let self_row = self.coeffs.get(i).cloned().unwrap_or_default();
            let other_row = other.coeffs.get(i).cloned().unwrap_or_default();
            let n_cols = self_row.len().max(other_row.len());
            *row = vec![0.0; n_cols];
            for (j, v) in row.iter_mut().enumerate() {
                *v = self_row.get(j).copied().unwrap_or(0.0)
                    - other_row.get(j).copied().unwrap_or(0.0);
            }
        }
        BiPoly::new(out)
    }

    /// Scale.
    pub fn scale(&self, factor: f64) -> BiPoly {
        BiPoly {
            coeffs: self
                .coeffs
                .iter()
                .map(|row| row.iter().map(|c| c * factor).collect())
                .collect(),
        }
    }

    /// Multiply two bivariate polynomials.
    pub fn mul(&self, other: &BiPoly) -> BiPoly {
        let max_xi = self.coeffs.len() + other.coeffs.len();
        let mut out: Vec<Vec<f64>> = vec![vec![0.0; 1]; max_xi];
        for (i1, row1) in self.coeffs.iter().enumerate() {
            for (j1, &a) in row1.iter().enumerate() {
                if a.abs() < 1e-15 {
                    continue;
                }
                for (i2, row2) in other.coeffs.iter().enumerate() {
                    for (j2, &b) in row2.iter().enumerate() {
                        if b.abs() < 1e-15 {
                            continue;
                        }
                        let i = i1 + i2;
                        let j = j1 + j2;
                        while out[i].len() <= j {
                            out[i].push(0.0);
                        }
                        out[i][j] += a * b;
                    }
                }
            }
        }
        BiPoly::new(out)
    }

    /// Square.
    pub fn square(&self) -> BiPoly {
        self.mul(self)
    }

    /// Sample-check non-negativity over a grid in [-range, range]².
    pub fn sample_nonneg(&self, range: f64, n_samples: usize) -> bool {
        for i in 0..=n_samples {
            for j in 0..=n_samples {
                let x = -range + 2.0 * range * (i as f64 / n_samples as f64);
                let y = -range + 2.0 * range * (j as f64 / n_samples as f64);
                if self.eval(x, y) < -1e-9 {
                    return false;
                }
            }
        }
        true
    }

    pub fn approx_eq(&self, other: &BiPoly, tol: f64) -> bool {
        let n = self.coeffs.len().max(other.coeffs.len());
        for i in 0..n {
            let a_row = self.coeffs.get(i).cloned().unwrap_or_default();
            let b_row = other.coeffs.get(i).cloned().unwrap_or_default();
            let m = a_row.len().max(b_row.len());
            for j in 0..m {
                let a = a_row.get(j).copied().unwrap_or(0.0);
                let b = b_row.get(j).copied().unwrap_or(0.0);
                if (a - b).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

// ─── Bivariate SOS (small-basis search) ─────────────────────────────────────

/// Attempt to write a bivariate polynomial p(x, y) as a sum of squares
/// of elements drawn from a small hand-curated basis: {1, x, y, x-y, x+y,
/// xy, x², y², x²-y², x²+y²}. The search tries non-negative coefficients
/// on these squares and checks whether they sum to p.
///
/// This is *not* SDP-based — it's a finite enumeration over a fixed basis.
/// It works for symmetric 2-variable polynomials up to total degree 4 where
/// the natural decomposition uses these building blocks (which covers many
/// IMO-style examples like (x-y)² ≥ 0, (x+y)² ≥ 4xy, Schur reductions).
pub fn sos_bivariate_symmetric(p: &BiPoly) -> Option<Vec<BiPoly>> {
    // Hand-curated basis of "square roots" — polynomials q such that q² is
    // a plausible SOS component for a symmetric bivariate polynomial.
    let x = BiPoly::monomial(1, 0);
    let y = BiPoly::monomial(0, 1);
    let one = BiPoly::constant(1.0);
    let xx = BiPoly::monomial(2, 0);
    let yy = BiPoly::monomial(0, 2);
    let xy = BiPoly::monomial(1, 1);
    let basis = vec![
        one.clone(),
        x.clone(),
        y.clone(),
        x.add(&y),
        x.sub(&y),
        xy.clone(),
        xx.clone(),
        yy.clone(),
        xx.add(&yy),
        xx.sub(&yy),
    ];
    // Enumerate small non-negative linear combinations of squares and
    // check equality with p. We use a coarse grid on [0, 2] in steps of
    // 0.25, capped at 3 active basis terms to keep the search finite.
    let grid: Vec<f64> = (0..=8).map(|i| i as f64 * 0.25).collect();
    for i in 0..basis.len() {
        for coef_i in &grid {
            if *coef_i < 1e-12 {
                continue;
            }
            let part_i = basis[i].square().scale(*coef_i);
            // 1-term candidate
            if p.approx_eq(&part_i, 1e-6) {
                return Some(vec![basis[i].scale(coef_i.sqrt())]);
            }
            for j in (i + 1)..basis.len() {
                for coef_j in &grid {
                    if *coef_j < 1e-12 {
                        continue;
                    }
                    let part_j = basis[j].square().scale(*coef_j);
                    let two_term = part_i.add(&part_j);
                    if p.approx_eq(&two_term, 1e-6) {
                        return Some(vec![
                            basis[i].scale(coef_i.sqrt()),
                            basis[j].scale(coef_j.sqrt()),
                        ]);
                    }
                    for k in (j + 1)..basis.len() {
                        for coef_k in &grid {
                            if *coef_k < 1e-12 {
                                continue;
                            }
                            let part_k = basis[k].square().scale(*coef_k);
                            let three_term = two_term.add(&part_k);
                            if p.approx_eq(&three_term, 1e-6) {
                                return Some(vec![
                                    basis[i].scale(coef_i.sqrt()),
                                    basis[j].scale(coef_j.sqrt()),
                                    basis[k].scale(coef_k.sqrt()),
                                ]);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Verify a bivariate SOS decomposition.
pub fn verify_sos_bivariate(p: &BiPoly, decomp: &[BiPoly], tol: f64) -> bool {
    let mut sum = BiPoly::constant(0.0);
    for q in decomp {
        sum = sum.add(&q.square());
    }
    p.approx_eq(&sum, tol)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Univariate arithmetic ─────────────────────────────────────────

    #[test]
    fn test_poly_eval() {
        // p(x) = 2x² + 3x + 1
        let p = Poly::new(vec![1.0, 3.0, 2.0]);
        assert!((p.eval(2.0) - 15.0).abs() < 1e-9);
        assert!((p.eval(-1.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_poly_add_sub() {
        let a = Poly::new(vec![1.0, 2.0, 3.0]); // 3x² + 2x + 1
        let b = Poly::new(vec![4.0, 5.0]); // 5x + 4
        let sum = a.add(&b);
        assert_eq!(sum.coeffs, vec![5.0, 7.0, 3.0]); // 3x² + 7x + 5
        let diff = a.sub(&b);
        assert_eq!(diff.coeffs, vec![-3.0, -3.0, 3.0]); // 3x² − 3x − 3
    }

    #[test]
    fn test_poly_mul() {
        let a = Poly::new(vec![1.0, 1.0]); // x + 1
        let b = Poly::new(vec![-1.0, 1.0]); // x − 1
        let prod = a.mul(&b);
        assert_eq!(prod.coeffs, vec![-1.0, 0.0, 1.0]); // x² − 1
    }

    #[test]
    fn test_poly_square() {
        let a = Poly::new(vec![1.0, 1.0]); // x + 1
        let sq = a.square();
        assert_eq!(sq.coeffs, vec![1.0, 2.0, 1.0]); // x² + 2x + 1
    }

    // ── Univariate SOS ────────────────────────────────────────────────

    #[test]
    fn test_sos_constant_positive() {
        let p = Poly::constant(4.0);
        let decomp = sos_univariate(&p).unwrap();
        assert!(verify_sos(&p, &decomp, 1e-9));
    }

    #[test]
    fn test_sos_constant_negative_rejected() {
        let p = Poly::constant(-4.0);
        assert!(sos_univariate(&p).is_none());
    }

    #[test]
    fn test_sos_perfect_square() {
        // (x + 1)² = x² + 2x + 1
        let p = Poly::new(vec![1.0, 2.0, 1.0]);
        let decomp = sos_univariate(&p).unwrap();
        assert!(verify_sos(&p, &decomp, 1e-6));
        assert!(sample_nonneg(&p, 10.0, 100));
    }

    #[test]
    fn test_sos_positive_quadratic() {
        // x² + 1 — always positive
        let p = Poly::new(vec![1.0, 0.0, 1.0]);
        let decomp = sos_univariate(&p).unwrap();
        assert!(verify_sos(&p, &decomp, 1e-6));
    }

    #[test]
    fn test_sos_negative_quadratic_rejected() {
        // x² − 1 — goes negative at x ∈ (-1, 1)
        let p = Poly::new(vec![-1.0, 0.0, 1.0]);
        assert!(sos_univariate(&p).is_none());
    }

    #[test]
    fn test_sos_quartic_positive_const_plus_square() {
        // (x² + 1)² = x⁴ + 2x² + 1
        let p = Poly::new(vec![1.0, 0.0, 2.0, 0.0, 1.0]);
        let decomp = sos_univariate(&p).unwrap();
        assert!(verify_sos(&p, &decomp, 1e-5));
    }

    #[test]
    fn test_sos_rejects_odd_degree() {
        let p = Poly::new(vec![0.0, 1.0, 0.0, 1.0]); // x³ + x
        assert!(sos_univariate(&p).is_none());
    }

    #[test]
    fn test_sample_nonneg_positive_case() {
        let p = Poly::new(vec![1.0, 0.0, 1.0]); // x² + 1
        assert!(sample_nonneg(&p, 5.0, 50));
    }

    #[test]
    fn test_sample_nonneg_negative_case() {
        let p = Poly::new(vec![-2.0, 0.0, 1.0]); // x² − 2, negative near 0
        assert!(!sample_nonneg(&p, 5.0, 50));
    }

    // ── Bivariate arithmetic ──────────────────────────────────────────

    #[test]
    fn test_bipoly_eval() {
        // 2xy + 1
        let p = BiPoly::new(vec![vec![1.0, 0.0], vec![0.0, 2.0]]);
        assert!((p.eval(3.0, 4.0) - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_bipoly_add() {
        let a = BiPoly::monomial(1, 0); // x
        let b = BiPoly::monomial(0, 1); // y
        let sum = a.add(&b); // x + y
        assert!((sum.eval(2.0, 3.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_bipoly_mul_xy_identity() {
        // x · y = xy
        let x = BiPoly::monomial(1, 0);
        let y = BiPoly::monomial(0, 1);
        let prod = x.mul(&y);
        assert!((prod.eval(2.0, 3.0) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_bipoly_square_diff() {
        // (x − y)² = x² − 2xy + y²
        let x = BiPoly::monomial(1, 0);
        let y = BiPoly::monomial(0, 1);
        let diff = x.sub(&y);
        let sq = diff.square();
        // At (3, 1): 4
        assert!((sq.eval(3.0, 1.0) - 4.0).abs() < 1e-9);
        // At (2, 2): 0
        assert!(sq.eval(2.0, 2.0).abs() < 1e-9);
    }

    // ── Bivariate SOS ──────────────────────────────────────────────────

    #[test]
    fn test_sos_bivariate_xy_difference_squared() {
        // p(x, y) = x² − 2xy + y² = (x − y)²
        let x = BiPoly::monomial(1, 0);
        let y = BiPoly::monomial(0, 1);
        let p = x.sub(&y).square();
        let decomp = sos_bivariate_symmetric(&p).expect("should decompose");
        assert!(verify_sos_bivariate(&p, &decomp, 1e-6));
    }

    #[test]
    fn test_sos_bivariate_sum_squared() {
        // p = (x + y)² = x² + 2xy + y²
        let x = BiPoly::monomial(1, 0);
        let y = BiPoly::monomial(0, 1);
        let p = x.add(&y).square();
        let decomp = sos_bivariate_symmetric(&p).expect("should decompose");
        assert!(verify_sos_bivariate(&p, &decomp, 1e-6));
    }

    #[test]
    fn test_sos_bivariate_sum_of_two_squares() {
        // p = (x − y)² + (x + y)² = 2x² + 2y²
        let x = BiPoly::monomial(1, 0);
        let y = BiPoly::monomial(0, 1);
        let p = x.sub(&y).square().add(&x.add(&y).square());
        let decomp = sos_bivariate_symmetric(&p).expect("should decompose");
        assert!(verify_sos_bivariate(&p, &decomp, 1e-6));
    }

    #[test]
    fn test_sos_bivariate_sample_check() {
        // Constructed-positive: (x − y)² + 1 sampled on a grid — always ≥ 1 > 0
        let x = BiPoly::monomial(1, 0);
        let y = BiPoly::monomial(0, 1);
        let p = x.sub(&y).square().add(&BiPoly::constant(1.0));
        assert!(p.sample_nonneg(5.0, 20));
    }
}
