// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Complex Analysis for Symthaea
//!
//! Advanced complex analysis on top of the existing `Complex` type:
//! Möbius transformations, numerical contour integration, Laurent series,
//! residue theorem, power series, conformal maps, and the argument principle.
//!
//! # Import strategy
//! We use `crate::hdc::complex::Complex` as our scalar type and define a
//! local type alias `C64 = Complex` for brevity throughout.

use crate::hdc::complex::Complex;
use std::f64::consts::PI;

/// Convenience alias so formulas match the mathematical notation C64 = ℂ.
pub type C64 = Complex;

// ── Low-level arithmetic helpers ─────────────────────────────────────────────
// These thin wrappers expose the Complex operations as free functions so that
// callers can write `c_add(a, b)` instead of `(a + b)`.

#[inline]
pub fn c_add(a: C64, b: C64) -> C64 {
    a + b
}
#[inline]
pub fn c_sub(a: C64, b: C64) -> C64 {
    a - b
}
#[inline]
pub fn c_mul(a: C64, b: C64) -> C64 {
    a * b
}
#[inline]
pub fn c_div(a: C64, b: C64) -> C64 {
    a / b
}
#[inline]
pub fn c_conj(z: C64) -> C64 {
    z.conjugate()
}
#[inline]
pub fn c_abs(z: C64) -> f64 {
    z.magnitude()
}
#[inline]
pub fn c_arg(z: C64) -> f64 {
    z.phase()
}
#[inline]
pub fn c_exp(z: C64) -> C64 {
    z.exp()
}
#[inline]
pub fn c_ln(z: C64) -> C64 {
    z.ln()
}
/// z^w = e^(w * ln z).
#[inline]
pub fn c_pow(z: C64, w: C64) -> C64 {
    c_exp(c_mul(w, c_ln(z)))
}

// ═══════════════════════════════════════════════════════════════════════════
// MÖBIUS TRANSFORMATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// A Möbius (fractional linear) transformation f(z) = (az + b) / (cz + d).
///
/// Represented as a 2×2 complex matrix [[a, b], [c, d]].
/// Composition corresponds to matrix multiplication.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MobiusTransform {
    pub a: C64,
    pub b: C64,
    pub c: C64,
    pub d: C64,
}

impl MobiusTransform {
    /// Create a Möbius transform with the given matrix entries.
    pub fn new(a: C64, b: C64, c: C64, d: C64) -> Self {
        Self { a, b, c, d }
    }

    /// Identity transform: f(z) = z.
    pub fn identity() -> Self {
        Self {
            a: C64::ONE,
            b: C64::ZERO,
            c: C64::ZERO,
            d: C64::ONE,
        }
    }

    /// Translation by w: f(z) = z + w.
    pub fn translation(w: C64) -> Self {
        Self {
            a: C64::ONE,
            b: w,
            c: C64::ZERO,
            d: C64::ONE,
        }
    }

    /// Rotation by angle theta (radians): f(z) = e^{i*theta} * z.
    pub fn rotation(theta: f64) -> Self {
        let e = C64::from_polar(1.0, theta);
        Self {
            a: e,
            b: C64::ZERO,
            c: C64::ZERO,
            d: C64::ONE,
        }
    }

    /// Scaling by real factor r: f(z) = r * z.
    pub fn scaling(r: f64) -> Self {
        Self {
            a: C64::new(r, 0.0),
            b: C64::ZERO,
            c: C64::ZERO,
            d: C64::ONE,
        }
    }

    /// Inversion: f(z) = 1/z.
    pub fn inversion() -> Self {
        Self {
            a: C64::ZERO,
            b: C64::ONE,
            c: C64::ONE,
            d: C64::ZERO,
        }
    }

    /// Apply the transform to a point z.
    ///
    /// Returns (a*z + b) / (c*z + d).
    pub fn apply(&self, z: C64) -> C64 {
        let num = c_add(c_mul(self.a, z), self.b);
        let den = c_add(c_mul(self.c, z), self.d);
        c_div(num, den)
    }

    /// Compose: (self ∘ other)(z) = self(other(z)).
    /// Implemented as matrix multiplication [[a,b],[c,d]] * [[e,f],[g,h]].
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            a: c_add(c_mul(self.a, other.a), c_mul(self.b, other.c)),
            b: c_add(c_mul(self.a, other.b), c_mul(self.b, other.d)),
            c: c_add(c_mul(self.c, other.a), c_mul(self.d, other.c)),
            d: c_add(c_mul(self.c, other.b), c_mul(self.d, other.d)),
        }
    }

    /// Inverse: if f = (az+b)/(cz+d) then f⁻¹ = (dz-b)/(-cz+a).
    pub fn inverse(&self) -> Self {
        Self {
            a: self.d,
            b: C64::new(-self.b.re, -self.b.im),
            c: C64::new(-self.c.re, -self.c.im),
            d: self.a,
        }
    }

    /// Fixed points of the transform: solutions to (az+b)/(cz+d) = z, i.e.
    /// c*z² + (d-a)*z - b = 0.
    ///
    /// Returns 0, 1, or 2 fixed points (parabolic → 1 repeated root, else 2).
    pub fn fixed_points(&self) -> Vec<C64> {
        // c*z² + (d-a)*z - b = 0
        let coeff_a = self.c;
        let coeff_b = c_sub(self.d, self.a);
        let coeff_c = C64::new(-self.b.re, -self.b.im);

        // If c = 0: linear equation (d-a)*z - b = 0  →  z = b/(d-a)
        if c_abs(coeff_a) < 1e-12 {
            let da = coeff_b;
            if c_abs(da) < 1e-12 {
                return vec![]; // identity: every point fixed
            }
            return vec![c_div(self.b, da)];
        }

        // Quadratic: z = (-coeff_b ± sqrt(coeff_b² - 4*a*c)) / (2*a)
        let discriminant = c_sub(
            c_mul(coeff_b, coeff_b),
            c_mul(C64::new(4.0, 0.0), c_mul(coeff_a, coeff_c)),
        );
        let sqrt_d = complex_sqrt(discriminant);
        let two_a = c_mul(C64::new(2.0, 0.0), coeff_a);
        let neg_b = C64::new(-coeff_b.re, -coeff_b.im);

        let z1 = c_div(c_add(neg_b, sqrt_d), two_a);
        let z2 = c_div(c_sub(neg_b, sqrt_d), two_a);

        // If discriminant ≈ 0, return one root
        if c_abs(c_sub(z1, z2)) < 1e-10 {
            vec![z1]
        } else {
            vec![z1, z2]
        }
    }
}

/// Principal square root of a complex number.
fn complex_sqrt(z: C64) -> C64 {
    let r = c_abs(z);
    let theta = c_arg(z);
    C64::from_polar(r.sqrt(), theta / 2.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTOUR INTEGRATION (NUMERICAL)
// ═══════════════════════════════════════════════════════════════════════════

/// Numerically integrate f along a circular contour γ(t) = center + radius·e^{it}
/// for t ∈ [0, 2π] using the trapezoidal rule with `n_points` sample points.
///
/// Returns ∮_γ f(z) dz.
pub fn integrate_circular<F>(f: F, center: C64, radius: f64, n_points: usize) -> C64
where
    F: Fn(C64) -> C64,
{
    let n = n_points;
    let dt = 2.0 * PI / n as f64;
    let mut sum = C64::ZERO;
    for k in 0..n {
        let t = k as f64 * dt;
        let z = c_add(center, C64::from_polar(radius, t));
        // dz = i·radius·e^{it} dt
        let dz = c_mul(C64::new(0.0, radius), C64::from_polar(1.0, t));
        sum = c_add(sum, c_mul(f(z), c_mul(dz, C64::new(dt, 0.0))));
    }
    sum
}

/// Compute the winding number of a closed path around a point via the total
/// angle traversed (divided by 2π).
///
/// `path` should be a list of consecutive points; the path is closed by
/// connecting the last point back to the first.
pub fn winding_number(path: &[C64], point: C64) -> i32 {
    if path.len() < 2 {
        return 0;
    }
    let mut total_angle = 0.0f64;
    let n = path.len();
    for i in 0..n {
        let p0 = c_sub(path[i], point);
        let p1 = c_sub(path[(i + 1) % n], point);
        // Angle from p0 to p1 (via atan2)
        let cross = p0.re * p1.im - p0.im * p1.re;
        let dot = p0.re * p1.re + p0.im * p1.im;
        total_angle += cross.atan2(dot);
    }
    (total_angle / (2.0 * PI)).round() as i32
}

/// Apply Cauchy's integral formula to compute the n-th derivative of f at center.
///
/// f^(n)(center) = (n! / 2πi) ∮ f(z) / (z − center)^{n+1} dz
///
/// Uses `n_points` sample points on the circular contour.
pub fn cauchy_integral_formula<F>(f: F, center: C64, radius: f64, n: u32, n_points: usize) -> C64
where
    F: Fn(C64) -> C64,
{
    let n_fact = factorial(n as u64) as f64;
    // integrand: f(z) / (z - center)^(n+1)
    let two_pi_i = C64::new(0.0, 2.0 * PI);
    let integral = integrate_circular(
        |z| {
            let denom_base = c_sub(z, center);
            let denom = complex_pow_u(denom_base, n + 1);
            c_div(f(z), denom)
        },
        center,
        radius,
        n_points,
    );
    c_mul(C64::new(n_fact, 0.0), c_div(integral, two_pi_i))
}

/// Integer power of a complex number (non-negative exponent).
fn complex_pow_u(z: C64, n: u32) -> C64 {
    let mut result = C64::ONE;
    for _ in 0..n {
        result = c_mul(result, z);
    }
    result
}

fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

// ═══════════════════════════════════════════════════════════════════════════
// LAURENT SERIES / RESIDUES
// ═══════════════════════════════════════════════════════════════════════════

/// Numerically estimate the residue of f at a simple pole.
///
/// Uses the limit lim_{z→pole} (z − pole) · f(z), sampled on a small circle
/// of radius `epsilon` around the pole.
pub fn residue_at_simple_pole<F>(f: F, pole: C64, epsilon: f64) -> C64
where
    F: Fn(C64) -> C64,
{
    // Average (z - pole) * f(z) over sample points on the small circle
    let n = 128usize;
    let mut sum = C64::ZERO;
    for k in 0..n {
        let t = 2.0 * PI * k as f64 / n as f64;
        let z = c_add(pole, C64::from_polar(epsilon, t));
        let val = c_mul(c_sub(z, pole), f(z));
        sum = c_add(sum, val);
    }
    // The average converges to the residue for a simple pole
    C64::new(sum.re / n as f64, sum.im / n as f64)
}

/// Sum of residues multiplied by 2πi (the residue theorem).
///
/// ∮_γ f(z) dz = 2πi · Σ Res(f, z_k)
pub fn residue_theorem_sum(residues: &[C64]) -> C64 {
    let mut total = C64::ZERO;
    for &r in residues {
        total = c_add(total, r);
    }
    c_mul(C64::new(0.0, 2.0 * PI), total)
}

// ═══════════════════════════════════════════════════════════════════════════
// POWER SERIES
// ═══════════════════════════════════════════════════════════════════════════

/// A truncated power series Σ_{n=0}^{N-1} a_n * z^n.
#[derive(Debug, Clone)]
pub struct PowerSeries {
    /// Coefficients a_n for n = 0, 1, ..., coeffs.len()-1.
    pub coeffs: Vec<C64>,
}

impl PowerSeries {
    /// Create a power series from the given coefficient list.
    pub fn new(coeffs: Vec<C64>) -> Self {
        Self { coeffs }
    }

    /// Evaluate the series at z using at most `terms` coefficients (Horner's method).
    pub fn evaluate(&self, z: C64, terms: usize) -> C64 {
        let n = terms.min(self.coeffs.len());
        if n == 0 {
            return C64::ZERO;
        }
        let mut result = self.coeffs[n - 1];
        for i in (0..n - 1).rev() {
            result = c_add(c_mul(result, z), self.coeffs[i]);
        }
        result
    }
}

/// Radius of convergence via Cauchy-Hadamard formula: R = 1 / lim sup |a_n|^{1/n}.
///
/// Uses the maximum of |a_n|^{1/n} over the last half of coefficients as a
/// practical approximation of the lim sup.
pub fn radius_of_convergence(coeffs: &[C64]) -> f64 {
    if coeffs.is_empty() {
        return f64::INFINITY;
    }
    let start = coeffs.len() / 2;
    let mut lim_sup = 0.0f64;
    for (i, c) in coeffs[start..].iter().enumerate() {
        let n = (start + i) as f64;
        if n < 1.0 {
            continue;
        }
        let val = c_abs(*c).powf(1.0 / n);
        if val > lim_sup {
            lim_sup = val;
        }
    }
    if lim_sup < 1e-300 {
        f64::INFINITY
    } else {
        1.0 / lim_sup
    }
}

// ── Common power series ───────────────────────────────────────────────────

/// e^z via power series: Σ z^n / n!
pub fn exp_series(z: C64, terms: usize) -> C64 {
    let mut sum = C64::ZERO;
    let mut term = C64::ONE; // z^0 / 0! = 1
    for n in 0..terms {
        sum = c_add(sum, term);
        term = c_div(c_mul(term, z), C64::new((n + 1) as f64, 0.0));
    }
    sum
}

/// sin(z) via power series: Σ (-1)^n z^{2n+1} / (2n+1)!
pub fn sin_series(z: C64, terms: usize) -> C64 {
    let mut sum = C64::ZERO;
    let mut term = z; // z^1 / 1!
    let z2 = c_mul(z, z);
    for n in 0..terms {
        let two_n_1 = 2 * n + 1;
        sum = c_add(sum, term);
        // next term: multiply by -z^2 / ((2n+2)*(2n+3))
        let denom = C64::new(((two_n_1 + 1) * (two_n_1 + 2)) as f64, 0.0);
        term = c_div(c_mul(C64::new(-1.0, 0.0), c_mul(term, z2)), denom);
    }
    sum
}

/// cos(z) via power series: Σ (-1)^n z^{2n} / (2n)!
pub fn cos_series(z: C64, terms: usize) -> C64 {
    let mut sum = C64::ZERO;
    let mut term = C64::ONE; // z^0 / 0!
    let z2 = c_mul(z, z);
    for n in 0..terms {
        let two_n = 2 * n;
        sum = c_add(sum, term);
        // next term: multiply by -z^2 / ((2n+1)*(2n+2))
        let denom = C64::new(((two_n + 1) * (two_n + 2)) as f64, 0.0);
        term = c_div(c_mul(C64::new(-1.0, 0.0), c_mul(term, z2)), denom);
    }
    sum
}

/// ln(z) via power series: Σ_{n=1}^∞ (-1)^{n+1} (z-1)^n / n  for |z-1| < 1.
///
/// `z` should satisfy |z - 1| < 1 for convergence.
pub fn log_series(z: C64, terms: usize) -> C64 {
    // w = z - 1; ln(1+w) = w - w^2/2 + w^3/3 - ...
    let w = c_sub(z, C64::ONE);
    let mut sum = C64::ZERO;
    let mut w_pow = w; // w^1
    for n in 1..=terms {
        let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
        sum = c_add(sum, c_mul(C64::new(sign / n as f64, 0.0), w_pow));
        w_pow = c_mul(w_pow, w);
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFORMAL MAPS
// ═══════════════════════════════════════════════════════════════════════════

/// Joukowski transform: w = z + c²/z.
///
/// Maps a circle of radius > c centred near origin to an airfoil shape.
pub fn joukowski(z: C64, c: f64) -> C64 {
    let c2 = C64::new(c * c, 0.0);
    c_add(z, c_div(c2, z))
}

/// Cayley map: w = (z − i) / (z + i).
///
/// Maps the upper half-plane to the open unit disk.
pub fn cayley_map(z: C64) -> C64 {
    let i = C64::I;
    c_div(c_sub(z, i), c_add(z, i))
}

/// Inverse Cayley map: z = i·(1 + w) / (1 − w).
///
/// Maps the open unit disk to the upper half-plane.
pub fn inverse_cayley(w: C64) -> C64 {
    let i = C64::I;
    let one = C64::ONE;
    c_div(c_mul(i, c_add(one, w)), c_sub(one, w))
}

// ═══════════════════════════════════════════════════════════════════════════
// ARGUMENT PRINCIPLE
// ═══════════════════════════════════════════════════════════════════════════

/// Count (zeros − poles) of f inside the circular contour via the argument
/// principle: (1/2πi) ∮ f'(z)/f(z) dz = Z − P.
///
/// Numerically computes the winding number of the image curve f(γ) around 0.
pub fn count_zeros_minus_poles<F>(f: F, center: C64, radius: f64, n_points: usize) -> i32
where
    F: Fn(C64) -> C64,
{
    // Sample f along the contour
    let path: Vec<C64> = (0..n_points)
        .map(|k| {
            let t = 2.0 * PI * k as f64 / n_points as f64;
            let z = c_add(center, C64::from_polar(radius, t));
            f(z)
        })
        .collect();
    winding_number(&path, C64::ZERO)
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-6;
    const EPS_COARSE: f64 = 1e-3;

    fn approx_eq(a: C64, b: C64, tol: f64) -> bool {
        c_abs(c_sub(a, b)) < tol
    }

    // ── Möbius ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mobius_identity() {
        let id = MobiusTransform::identity();
        let z = C64::new(3.0, 4.0);
        assert!(approx_eq(id.apply(z), z, EPS));
    }

    #[test]
    fn test_mobius_rotation_pi() {
        // Rotation by π: f(z) = -z
        let rot = MobiusTransform::rotation(PI);
        let z = C64::new(1.0, 0.0);
        let expected = C64::new(-1.0, 0.0);
        assert!(approx_eq(rot.apply(z), expected, EPS));
    }

    #[test]
    fn test_mobius_inverse() {
        let m = MobiusTransform::new(
            C64::new(1.0, 1.0),
            C64::new(2.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(1.0, 0.0),
        );
        let mi = m.inverse();
        let z = C64::new(0.5, 0.3);
        let roundtrip = mi.apply(m.apply(z));
        assert!(approx_eq(roundtrip, z, EPS));
    }

    #[test]
    fn test_mobius_compose() {
        // Two translations compose to a single larger translation
        let t1 = MobiusTransform::translation(C64::new(1.0, 0.0));
        let t2 = MobiusTransform::translation(C64::new(2.0, 0.0));
        let t3 = t1.compose(&t2);
        let z = C64::new(0.0, 0.0);
        assert!(approx_eq(t3.apply(z), C64::new(3.0, 0.0), EPS));
    }

    #[test]
    fn test_mobius_fixed_points_identity_linear() {
        // f(z) = 2z: single fixed point at 0
        let m = MobiusTransform::scaling(2.0);
        let fps = m.fixed_points();
        // fixed points: z = 0 (since (2z - z) / 1 = 0  ↔  z = 0)
        // and z → ∞ (not returned)
        assert!(!fps.is_empty());
        assert!(fps.iter().any(|&z| c_abs(z) < EPS));
    }

    #[test]
    fn test_mobius_inversion_apply() {
        let inv = MobiusTransform::inversion();
        let z = C64::new(2.0, 0.0);
        assert!(approx_eq(inv.apply(z), C64::new(0.5, 0.0), EPS));
    }

    // ── Contour integration ──────────────────────────────────────────────────

    #[test]
    fn test_contour_integral_one_over_z() {
        // ∮ 1/z dz around the unit circle = 2πi
        let center = C64::ZERO;
        let result = integrate_circular(|z| c_div(C64::ONE, z), center, 1.0, 1024);
        let expected = C64::new(0.0, 2.0 * PI);
        assert!(approx_eq(result, expected, EPS_COARSE), "got {:?}", result);
    }

    #[test]
    fn test_contour_integral_constant() {
        // ∮ 1 dz around any closed contour = 0
        let result = integrate_circular(|_z| C64::ONE, C64::ZERO, 1.0, 256);
        assert!(c_abs(result) < EPS_COARSE, "got {:?}", result);
    }

    #[test]
    fn test_winding_number_inside() {
        // Unit circle path around origin — winding number = 1
        let n = 256usize;
        let path: Vec<C64> = (0..n)
            .map(|k| C64::from_polar(1.0, 2.0 * PI * k as f64 / n as f64))
            .collect();
        assert_eq!(winding_number(&path, C64::ZERO), 1);
    }

    #[test]
    fn test_winding_number_outside() {
        // Point at (10, 0) is far outside the unit circle — winding number = 0
        let n = 256usize;
        let path: Vec<C64> = (0..n)
            .map(|k| C64::from_polar(1.0, 2.0 * PI * k as f64 / n as f64))
            .collect();
        assert_eq!(winding_number(&path, C64::new(10.0, 0.0)), 0);
    }

    // ── Residues ─────────────────────────────────────────────────────────────

    #[test]
    fn test_residue_one_over_z() {
        // Res(1/z, 0) = 1
        let res = residue_at_simple_pole(|z| c_div(C64::ONE, z), C64::ZERO, 1e-4);
        assert!(approx_eq(res, C64::ONE, EPS_COARSE), "got {:?}", res);
    }

    #[test]
    fn test_residue_theorem_sum() {
        // 2πi * (1 + 1) = 4πi for two residues each = 1
        let residues = [C64::ONE, C64::ONE];
        let result = residue_theorem_sum(&residues);
        let expected = C64::new(0.0, 4.0 * PI);
        assert!(approx_eq(result, expected, EPS), "got {:?}", result);
    }

    // ── Power series ─────────────────────────────────────────────────────────

    #[test]
    fn test_exp_series_matches_std() {
        let z = C64::new(1.0, 0.5);
        let series = exp_series(z, 30);
        let exact = z.exp();
        assert!(
            approx_eq(series, exact, EPS_COARSE),
            "series={:?} exact={:?}",
            series,
            exact
        );
    }

    #[test]
    fn test_sin_series_matches_std() {
        let z = C64::new(0.5, 0.0);
        let series = sin_series(z, 20);
        // Real part should be sin(0.5)
        assert!((series.re - 0.5f64.sin()).abs() < EPS_COARSE);
        assert!(series.im.abs() < EPS_COARSE);
    }

    #[test]
    fn test_cos_series_matches_std() {
        let z = C64::new(0.5, 0.0);
        let series = cos_series(z, 20);
        assert!((series.re - 0.5f64.cos()).abs() < EPS_COARSE);
        assert!(series.im.abs() < EPS_COARSE);
    }

    #[test]
    fn test_radius_of_convergence_geometric() {
        // Geometric series sum x^n has coefficients 1, 1, 1, ...
        // Radius of convergence = 1
        let coeffs: Vec<C64> = (0..40).map(|_| C64::ONE).collect();
        let r = radius_of_convergence(&coeffs);
        assert!((r - 1.0).abs() < 0.05, "got R={}", r);
    }

    // ── Conformal maps ────────────────────────────────────────────────────────

    #[test]
    fn test_joukowski_real_axis() {
        // z = 2.0 (real), c = 1.0: w = 2 + 1/2 = 2.5 (real)
        let z = C64::new(2.0, 0.0);
        let w = joukowski(z, 1.0);
        assert!((w.re - 2.5).abs() < EPS, "got {:?}", w);
        assert!(w.im.abs() < EPS);
    }

    #[test]
    fn test_cayley_inverse_roundtrip() {
        let z = C64::new(0.5, 2.0);
        let w = cayley_map(z);
        let z2 = inverse_cayley(w);
        assert!(
            approx_eq(z, z2, EPS_COARSE),
            "roundtrip: {:?} vs {:?}",
            z,
            z2
        );
    }

    #[test]
    fn test_cayley_maps_i_to_zero() {
        // z = i maps to (i - i)/(i + i) = 0
        let z = C64::I;
        let w = cayley_map(z);
        assert!(c_abs(w) < EPS, "expected 0, got {:?}", w);
    }

    // ── Argument principle ───────────────────────────────────────────────────

    #[test]
    fn test_count_zeros_of_z_squared() {
        // f(z) = z^2 has a zero of order 2 at origin — count = 2
        let count = count_zeros_minus_poles(|z| c_mul(z, z), C64::ZERO, 0.5, 1024);
        assert_eq!(count, 2, "got {}", count);
    }
}
