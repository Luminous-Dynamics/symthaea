// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Algebraic Geometry (Numerical)
//!
//! Numerical algebraic geometry: affine varieties via Newton's method,
//! projective space, conics (classification, parametrization), elliptic
//! curves (Weierstrass form, chord-tangent group law, scalar multiplication),
//! and Bézout intersection counting.

// ---------------------------------------------------------------------------
// § Affine Varieties
// ---------------------------------------------------------------------------

/// A point in affine n-space.
#[derive(Debug, Clone)]
pub struct AffinePoint {
    pub coords: Vec<f64>,
}

/// Evaluate a system of functions at a point.
pub fn eval_system(polys: &[&dyn Fn(&[f64]) -> f64], point: &[f64]) -> Vec<f64> {
    polys.iter().map(|f| f(point)).collect()
}

/// Newton's method for solving F(x) = 0, F: ℝⁿ → ℝⁿ.
///
/// `f[i]` evaluates the i-th component.
/// `jac[i][j]` evaluates ∂Fᵢ/∂xⱼ.
/// Returns the solution or an error string.
pub fn newton_system(
    f: &[&dyn Fn(&[f64]) -> f64],
    jac: &[Vec<&dyn Fn(&[f64]) -> f64>],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, String> {
    let n = x0.len();
    assert_eq!(f.len(), n, "newton_system: f must have n components");
    assert_eq!(jac.len(), n, "newton_system: jac must have n rows");

    let mut x = x0.to_vec();

    for iter in 0..max_iter {
        let fx: Vec<f64> = f.iter().map(|fi| fi(&x)).collect();

        // Convergence check
        let residual: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if residual < tol {
            return Ok(x);
        }

        // Build Jacobian matrix J[i][j] = df_i/dx_j
        let mut j_mat: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|k| jac[i][k](&x)).collect())
            .collect();

        // Solve J * delta = -F via Gaussian elimination
        let delta = match solve_linear(&mut j_mat, &fx.iter().map(|v| -v).collect::<Vec<_>>()) {
            Some(d) => d,
            None => {
                return Err(format!(
                    "newton_system: singular Jacobian at iteration {}",
                    iter
                ));
            }
        };

        for i in 0..n {
            x[i] += delta[i];
        }
    }

    let residual: f64 = f.iter().map(|fi| fi(&x).powi(2)).sum::<f64>().sqrt();
    if residual < tol {
        Ok(x)
    } else {
        Err(format!(
            "newton_system: did not converge after {} iterations (residual={:.2e})",
            max_iter, residual
        ))
    }
}

/// Gaussian elimination to solve A x = b; returns x or None if singular.
fn solve_linear(a: &mut [Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut b = b.to_vec();

    for col in 0..n {
        // Partial pivoting
        let pivot = (col..n).max_by(|&r1, &r2| a[r1][col].abs().total_cmp(&a[r2][col].abs()));
        let pivot = pivot?;
        if a[pivot][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let pivot_val = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot_val;
            b[row] -= factor * b[col];
            for c in col..n {
                a[row][c] -= factor * a[col][c];
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - sum) / a[i][i];
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// § Projective Space
// ---------------------------------------------------------------------------

const PROJ_TOL: f64 = 1e-12;

/// A point in projective n-space, represented by homogeneous coordinates.
/// Normalised so the first nonzero coordinate equals 1.
#[derive(Debug, Clone)]
pub struct ProjectivePoint {
    pub homogeneous: Vec<f64>,
}

impl ProjectivePoint {
    /// Construct from homogeneous coordinates, normalising the first nonzero to 1.
    pub fn new(mut coords: Vec<f64>) -> Self {
        if let Some(first_nonzero) = coords.iter().find(|&&v| v.abs() > PROJ_TOL).copied() {
            for c in &mut coords {
                *c /= first_nonzero;
            }
        }
        Self {
            homogeneous: coords,
        }
    }

    /// Convert to affine coordinates by dividing by the last coordinate.
    pub fn to_affine(&self) -> Option<Vec<f64>> {
        let last = *self.homogeneous.last()?;
        if last.abs() <= PROJ_TOL {
            return None; // point at infinity
        }
        Some(
            self.homogeneous[..self.homogeneous.len() - 1]
                .iter()
                .map(|&v| v / last)
                .collect(),
        )
    }

    /// Embed an affine point into projective space by appending 1.
    pub fn from_affine(coords: &[f64]) -> Self {
        let mut h = coords.to_vec();
        h.push(1.0);
        Self::new(h)
    }

    /// Returns true when the last homogeneous coordinate is (approximately) zero.
    pub fn is_point_at_infinity(&self) -> bool {
        self.homogeneous
            .last()
            .map(|v| v.abs() <= PROJ_TOL)
            .unwrap_or(true)
    }

    /// Projective equality: coordinates are proportional.
    pub fn eq_projective(&self, other: &Self) -> bool {
        let n = self.homogeneous.len();
        if n != other.homogeneous.len() {
            return false;
        }
        // Find a nonzero pair and check cross-ratios
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let cross = self.homogeneous[i] * other.homogeneous[j]
                    - self.homogeneous[j] * other.homogeneous[i];
                if cross.abs() > PROJ_TOL {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// § Conics
// ---------------------------------------------------------------------------

/// A conic section: ax² + bxy + cy² + dx + ey + f = 0.
#[derive(Debug, Clone)]
pub struct Conic {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Conic {
    /// Discriminant of the quadratic part: b² − 4ac.
    pub fn discriminant(&self) -> f64 {
        self.b * self.b - 4.0 * self.a * self.c
    }

    /// Classify the conic based on the discriminant and other properties.
    pub fn conic_type(&self) -> &'static str {
        let disc = self.discriminant();
        // Degenerate check via matrix determinant
        // Full matrix: [[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]]
        let det = self.a * (self.c * self.f - (self.e / 2.0).powi(2))
            - (self.b / 2.0) * ((self.b / 2.0) * self.f - (self.e / 2.0) * (self.d / 2.0))
            + (self.d / 2.0) * ((self.b / 2.0) * (self.e / 2.0) - self.c * (self.d / 2.0));
        if det.abs() < 1e-10 {
            return "degenerate";
        }
        if disc.abs() < 1e-10 {
            return "parabola";
        }
        if disc < 0.0 {
            // ellipse or circle
            if (self.a - self.c).abs() < 1e-10 && self.b.abs() < 1e-10 {
                "circle"
            } else {
                "ellipse"
            }
        } else {
            "hyperbola"
        }
    }

    /// Find one approximate rational point on the conic (if it exists).
    /// Uses a grid search for a seed, then refines with Newton's method.
    pub fn rational_point(&self) -> Option<(f64, f64)> {
        let eval = |x: f64, y: f64| {
            self.a * x * x + self.b * x * y + self.c * y * y + self.d * x + self.e * y + self.f
        };
        // Try simple candidates first
        for &y in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            // Solve quadratic in x: a*x² + (b*y + d)*x + (c*y² + e*y + f) = 0
            let aa = self.a;
            let bb = self.b * y + self.d;
            let cc = self.c * y * y + self.e * y + self.f;
            if aa.abs() < 1e-12 {
                if bb.abs() > 1e-12 {
                    let x = -cc / bb;
                    if eval(x, y).abs() < 1e-8 {
                        return Some((x, y));
                    }
                }
            } else {
                let disc = bb * bb - 4.0 * aa * cc;
                if disc >= 0.0 {
                    let x = (-bb + disc.sqrt()) / (2.0 * aa);
                    if eval(x, y).abs() < 1e-8 {
                        return Some((x, y));
                    }
                    let x = (-bb - disc.sqrt()) / (2.0 * aa);
                    if eval(x, y).abs() < 1e-8 {
                        return Some((x, y));
                    }
                }
            }
        }
        None
    }

    /// Parametrize all rational points via lines through a base rational point.
    /// Returns a closure: t → (x(t), y(t)).
    pub fn parametrize(&self, base_point: (f64, f64)) -> Box<dyn Fn(f64) -> (f64, f64)> {
        let (x0, y0) = base_point;
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let d = self.d;
        let e = self.e;
        let f = self.f;

        Box::new(move |t: f64| {
            // Line through (x0, y0) with slope t: y = y0 + t*(x - x0)
            // Substitute into conic equation and solve for x.
            // A*x² + B*x + C = 0 where we factor out the known root x0.
            let aa = a + b * t + c * t * t;
            let bb = b * y0 - b * t * x0 + 2.0 * c * t * y0 - 2.0 * c * t * t * x0 + d + e * t;
            let _cc = c * y0 * y0 + e * y0 + f; // should be 0 at base_point

            if aa.abs() < 1e-12 {
                // Degenerate: return base point
                return (x0, y0);
            }

            // Since x0 is a root, A*x² + B*x + C = A*(x - x0)*(x - x1)
            // So x1 = -B/A - x0
            let x1 = -bb / aa - x0;
            let y1 = y0 + t * (x1 - x0);
            (x1, y1)
        })
    }
}

// ---------------------------------------------------------------------------
// § Elliptic Curves (Weierstrass form)
// ---------------------------------------------------------------------------

/// An elliptic curve y² = x³ + ax + b over ℝ.
#[derive(Debug, Clone)]
pub struct EllipticCurve {
    pub a: f64,
    pub b: f64,
}

/// A point on an elliptic curve (projective: includes the point at infinity).
#[derive(Debug, Clone)]
pub enum EllipticPoint {
    /// The identity element (point at infinity).
    Infinity,
    /// An affine point.
    Affine { x: f64, y: f64 },
}

const EC_TOL: f64 = 1e-8;

impl EllipticCurve {
    /// Discriminant: −16(4a³ + 27b²).
    pub fn discriminant(&self) -> f64 {
        -16.0 * (4.0 * self.a.powi(3) + 27.0 * self.b.powi(2))
    }

    /// Returns true if the curve is nonsingular (discriminant ≠ 0).
    pub fn is_nonsingular(&self) -> bool {
        self.discriminant().abs() > EC_TOL
    }

    /// Returns true if the given point lies on the curve.
    pub fn is_on_curve(&self, p: &EllipticPoint) -> bool {
        match p {
            EllipticPoint::Infinity => true,
            EllipticPoint::Affine { x, y } => {
                let lhs = y * y;
                let rhs = x * x * x + self.a * x + self.b;
                (lhs - rhs).abs() < EC_TOL
            }
        }
    }

    /// Chord-tangent group law: P + Q on the curve.
    pub fn add(&self, p: &EllipticPoint, q: &EllipticPoint) -> EllipticPoint {
        match (p, q) {
            (EllipticPoint::Infinity, _) => q.clone(),
            (_, EllipticPoint::Infinity) => p.clone(),
            (EllipticPoint::Affine { x: x1, y: y1 }, EllipticPoint::Affine { x: x2, y: y2 }) => {
                // P = -Q (vertical line)
                if (x1 - x2).abs() < EC_TOL && (y1 + y2).abs() < EC_TOL {
                    return EllipticPoint::Infinity;
                }

                let slope = if (x1 - x2).abs() < EC_TOL {
                    // Point doubling: tangent slope
                    if y1.abs() < EC_TOL {
                        return EllipticPoint::Infinity;
                    }
                    (3.0 * x1 * x1 + self.a) / (2.0 * y1)
                } else {
                    (y2 - y1) / (x2 - x1)
                };

                let x3 = slope * slope - x1 - x2;
                let y3 = slope * (x1 - x3) - y1;
                EllipticPoint::Affine { x: x3, y: y3 }
            }
        }
    }

    /// Negate a point: (x, y) → (x, −y).
    pub fn negate(&self, p: &EllipticPoint) -> EllipticPoint {
        match p {
            EllipticPoint::Infinity => EllipticPoint::Infinity,
            EllipticPoint::Affine { x, y } => EllipticPoint::Affine { x: *x, y: -y },
        }
    }

    /// Scalar multiplication via double-and-add.
    pub fn scalar_mult(&self, p: &EllipticPoint, n: i64) -> EllipticPoint {
        if n == 0 {
            return EllipticPoint::Infinity;
        }
        let (base, negative) = if n < 0 {
            (self.negate(p), true)
        } else {
            (p.clone(), false)
        };
        let _ = negative; // sign already handled

        let abs_n = n.unsigned_abs();
        let mut result = EllipticPoint::Infinity;
        let mut addend = if n < 0 { self.negate(p) } else { p.clone() };

        let mut k = abs_n;
        while k > 0 {
            if k & 1 == 1 {
                result = self.add(&result, &addend);
            }
            addend = self.add(&addend, &addend);
            k >>= 1;
        }
        let _ = base; // suppress unused warning
        result
    }

    /// Find the smallest n ≤ max_order such that nP = O (point at infinity),
    /// or None if no such n is found.
    pub fn torsion_points_order_check(&self, p: &EllipticPoint, max_order: usize) -> Option<usize> {
        let mut current = p.clone();
        for n in 1..=max_order {
            if matches!(current, EllipticPoint::Infinity) {
                return Some(n);
            }
            current = self.add(&current, p);
        }
        None
    }

    /// j-invariant: 1728 · 4a³ / (4a³ + 27b²).
    pub fn j_invariant(&self) -> f64 {
        let num = 4.0 * self.a.powi(3);
        let den = 4.0 * self.a.powi(3) + 27.0 * self.b.powi(2);
        if den.abs() < 1e-14 {
            return f64::INFINITY;
        }
        1728.0 * num / den
    }
}

// ---------------------------------------------------------------------------
// § Bézout's Theorem (numerical)
// ---------------------------------------------------------------------------

/// Count the approximate number of intersection points of two plane curves
/// f(x,y) = 0 and g(x,y) = 0 by grid search.
///
/// A grid point (x, y) is counted if |f(x,y)| < tol AND |g(x,y)| < tol.
/// Nearby grid points within 3·(step) are merged into one intersection.
pub fn bezout_intersection_count(
    f: &dyn Fn(f64, f64) -> f64,
    g: &dyn Fn(f64, f64) -> f64,
    x_range: (f64, f64),
    y_range: (f64, f64),
    grid_n: usize,
    tol: f64,
) -> usize {
    let dx = (x_range.1 - x_range.0) / grid_n as f64;
    let dy = (y_range.1 - y_range.0) / grid_n as f64;
    let merge_dist = 3.0 * dx.hypot(dy);

    let mut intersections: Vec<(f64, f64)> = Vec::new();

    for i in 0..=grid_n {
        for j in 0..=grid_n {
            let x = x_range.0 + i as f64 * dx;
            let y = y_range.0 + j as f64 * dy;
            if f(x, y).abs() < tol && g(x, y).abs() < tol {
                // Merge with nearby found intersections
                let is_new = intersections
                    .iter()
                    .all(|&(px, py)| (x - px).hypot(y - py) > merge_dist);
                if is_new {
                    intersections.push((x, y));
                }
            }
        }
    }
    intersections.len()
}

/// Theoretical Bézout bound for two plane curves of degrees d₁ and d₂.
pub fn bezout_bound(deg_f: usize, deg_g: usize) -> usize {
    deg_f * deg_g
}

// ---------------------------------------------------------------------------
// § Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Conic discriminant: unit circle x²+y²=1 is an ellipse (disc < 0)
    #[test]
    fn conic_circle_discriminant() {
        let circle = Conic {
            a: 1.0,
            b: 0.0,
            c: 1.0,
            d: 0.0,
            e: 0.0,
            f: -1.0,
        };
        assert!(
            circle.discriminant() < 0.0,
            "circle discriminant should be negative"
        );
        assert_eq!(circle.conic_type(), "circle");
    }

    // 2. Ellipse detection
    #[test]
    fn conic_ellipse_type() {
        // x²/4 + y²/9 = 1 → 9x² + 4y² - 36 = 0 (a=9, c=4, b=0)
        let ellipse = Conic {
            a: 9.0,
            b: 0.0,
            c: 4.0,
            d: 0.0,
            e: 0.0,
            f: -36.0,
        };
        assert!(ellipse.discriminant() < 0.0);
        assert_eq!(ellipse.conic_type(), "ellipse");
    }

    // 3. Hyperbola detection
    #[test]
    fn conic_hyperbola_type() {
        // xy = 1 → b=1, a=0, c=0, f=-1
        let hyp = Conic {
            a: 0.0,
            b: 1.0,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            f: -1.0,
        };
        assert!(hyp.discriminant() > 0.0);
        assert_eq!(hyp.conic_type(), "hyperbola");
    }

    // 4. Parabola detection
    #[test]
    fn conic_parabola_type() {
        // y = x² → x² - y = 0: a=1, b=0, c=0, e=-1, rest 0
        let par = Conic {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: -1.0,
            f: 0.0,
        };
        // discriminant = 0
        assert!(par.discriminant().abs() < 1e-10, "parabola disc = 0");
        assert_eq!(par.conic_type(), "parabola");
    }

    // 5. EllipticCurve y²=x³-x+1: nonsingular
    #[test]
    fn elliptic_curve_nonsingular() {
        let ec = EllipticCurve { a: -1.0, b: 1.0 };
        assert!(ec.is_nonsingular(), "y²=x³-x+1 should be nonsingular");
    }

    // 6. Elliptic addition: P + (-P) = Infinity
    #[test]
    fn elliptic_add_inverse() {
        let ec = EllipticCurve { a: -1.0, b: 1.0 };
        let p = EllipticPoint::Affine { x: 0.0, y: 1.0 };
        let neg_p = ec.negate(&p);
        let result = ec.add(&p, &neg_p);
        assert!(
            matches!(result, EllipticPoint::Infinity),
            "P + (-P) should be Infinity"
        );
    }

    // 7. Elliptic identity: P + O = P
    #[test]
    fn elliptic_add_identity() {
        let ec = EllipticCurve { a: -1.0, b: 1.0 };
        let p = EllipticPoint::Affine { x: 0.0, y: 1.0 };
        let result = ec.add(&p, &EllipticPoint::Infinity);
        match result {
            EllipticPoint::Affine { x, y } => {
                assert!((x - 0.0).abs() < EC_TOL);
                assert!((y - 1.0).abs() < EC_TOL);
            }
            _ => panic!("P + O should equal P"),
        }
    }

    // 8. Elliptic doubling: 2P at P=(0,1) on y²=x³-x+1
    #[test]
    fn elliptic_doubling() {
        let ec = EllipticCurve { a: -1.0, b: 1.0 };
        let p = EllipticPoint::Affine { x: 0.0, y: 1.0 };
        let two_p = ec.add(&p, &p);
        // Verify 2P is on the curve
        assert!(ec.is_on_curve(&two_p), "2P should lie on the curve");
        match &two_p {
            EllipticPoint::Affine { x, y } => {
                // slope = (3*0² + (-1)) / (2*1) = -1/2
                // x3 = (-1/2)² - 0 - 0 = 1/4
                // y3 = (-1/2)*(0 - 1/4) - 1 = 1/8 - 1 = -7/8
                assert!((x - 0.25).abs() < 1e-9, "2P x should be 0.25, got {}", x);
                assert!((y + 0.875).abs() < 1e-9, "2P y should be -0.875, got {}", y);
            }
            _ => panic!("2P should be affine"),
        }
    }

    // 9. j-invariant of y²=x³-1 (a=0, b=-1)
    #[test]
    fn j_invariant_test() {
        let ec = EllipticCurve { a: 0.0, b: -1.0 };
        // j = 1728 * 4*0 / (0 + 27) = 0
        let j = ec.j_invariant();
        assert!(
            (j - 0.0).abs() < 1e-9,
            "j-invariant of y²=x³-1 should be 0, got {}",
            j
        );
    }

    // 10. ProjectivePoint to affine
    #[test]
    fn projective_to_affine() {
        let p = ProjectivePoint::new(vec![1.0, 2.0, 1.0]);
        let affine = p.to_affine().unwrap();
        assert_eq!(affine.len(), 2);
        assert!(
            (affine[0] - 1.0).abs() < PROJ_TOL,
            "x should be 1, got {}",
            affine[0]
        );
        assert!(
            (affine[1] - 2.0).abs() < PROJ_TOL,
            "y should be 2, got {}",
            affine[1]
        );
    }

    // 11. Projective point at infinity
    #[test]
    fn projective_point_at_infinity() {
        let p = ProjectivePoint::new(vec![1.0, 0.0, 0.0]);
        assert!(p.is_point_at_infinity());
    }

    // 12. Bézout bound
    #[test]
    fn bezout_bound_test() {
        assert_eq!(bezout_bound(2, 3), 6);
        assert_eq!(bezout_bound(1, 1), 1);
        assert_eq!(bezout_bound(3, 3), 9);
    }

    // 13. Bezout intersection count: line y=x and y=-x intersect at (0,0)
    #[test]
    fn bezout_intersection_line_line() {
        let f = |x: f64, y: f64| y - x; // y = x
        let g = |x: f64, y: f64| y + x; // y = -x
        let count = bezout_intersection_count(&f, &g, (-2.0, 2.0), (-2.0, 2.0), 100, 0.05);
        assert_eq!(count, 1, "Two lines through origin intersect once");
    }

    // 14. Newton system: solve x² + y² = 1 and x = y (roots at ±1/√2)
    #[test]
    fn newton_system_circle_line() {
        let f0 = |v: &[f64]| v[0] * v[0] + v[1] * v[1] - 1.0;
        let f1 = |v: &[f64]| v[0] - v[1];
        let df0_dx = |v: &[f64]| 2.0 * v[0];
        let df0_dy = |v: &[f64]| 2.0 * v[1];
        let df1_dx = |_: &[f64]| 1.0;
        let df1_dy = |_: &[f64]| -1.0;

        let f: &[&dyn Fn(&[f64]) -> f64] = &[&f0, &f1];
        let jac: &[Vec<&dyn Fn(&[f64]) -> f64>] = &[vec![&df0_dx, &df0_dy], vec![&df1_dx, &df1_dy]];

        let result = newton_system(f, jac, &[0.6, 0.6], 50, 1e-10);
        assert!(result.is_ok(), "Newton should converge");
        let sol = result.unwrap();
        let expected = 1.0_f64 / 2.0_f64.sqrt();
        assert!(
            (sol[0] - expected).abs() < 1e-8,
            "x should be 1/√2, got {}",
            sol[0]
        );
    }

    // 15. Scalar multiplication: order test
    #[test]
    fn scalar_mult_order() {
        // On y²=x³-x (a=-1, b=0), (0,0) is a 2-torsion point: 2*(0,0) = Infinity
        let ec = EllipticCurve { a: -1.0, b: 0.0 };
        let p = EllipticPoint::Affine { x: 0.0, y: 0.0 };
        // (0,0) satisfies 0² = 0³ - 0 ✓
        assert!(ec.is_on_curve(&p));
        // 2*(0,0): tangent slope = (3*0² + (-1)) / (2*0) → undefined (y=0), so = Infinity
        let two_p = ec.scalar_mult(&p, 2);
        assert!(
            matches!(two_p, EllipticPoint::Infinity),
            "2*(0,0) should be Infinity on y²=x³-x"
        );
    }

    // 16. eval_system basic test
    #[test]
    fn eval_system_test() {
        let f0 = |v: &[f64]| v[0] + v[1];
        let f1 = |v: &[f64]| v[0] * v[1];
        let fns: &[&dyn Fn(&[f64]) -> f64] = &[&f0, &f1];
        let result = eval_system(fns, &[2.0, 3.0]);
        assert!((result[0] - 5.0).abs() < 1e-12);
        assert!((result[1] - 6.0).abs() < 1e-12);
    }
}
