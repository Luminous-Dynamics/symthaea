// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Root finding: bisection, Newton-Raphson, and the secant method.

/// Bisection on a bracket `[a, b]` where `f` changes sign. `None` if the bracket
/// is invalid (no sign change). Converges to within `tol`.
pub fn bisection(
    f: impl Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let (mut fa, fb) = (f(a), f(b));
    if fa == 0.0 {
        return Some(a);
    }
    if fb == 0.0 {
        return Some(b);
    }
    if fa * fb > 0.0 {
        return None; // no sign change → no guaranteed root
    }
    for _ in 0..max_iter {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if fm.abs() < tol || 0.5 * (b - a) < tol {
            return Some(m);
        }
        if fa * fm < 0.0 {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    Some(0.5 * (a + b))
}

/// Newton-Raphson from `x0` using the derivative `df`. `None` if a derivative
/// vanishes or it fails to converge in `max_iter` steps.
pub fn newton(
    f: impl Fn(f64) -> f64,
    df: impl Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let mut x = x0;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol {
            return Some(x);
        }
        let dfx = df(x);
        if dfx.abs() < 1e-300 {
            return None;
        }
        x -= fx / dfx;
    }
    (f(x).abs() < tol).then_some(x)
}

/// The secant method from two initial guesses (no derivative needed).
pub fn secant(
    f: impl Fn(f64) -> f64,
    mut x0: f64,
    mut x1: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let (mut f0, mut f1) = (f(x0), f(x1));
    for _ in 0..max_iter {
        if f1.abs() < tol {
            return Some(x1);
        }
        let denom = f1 - f0;
        if denom.abs() < 1e-300 {
            return None;
        }
        let x2 = x1 - f1 * (x1 - x0) / denom;
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x1);
    }
    (f1.abs() < tol).then_some(x1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_three_find_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let want = 2.0_f64.sqrt();
        assert!((bisection(f, 0.0, 2.0, 1e-12, 200).unwrap() - want).abs() < 1e-6);
        assert!((newton(f, df, 1.0, 1e-14, 100).unwrap() - want).abs() < 1e-12);
        assert!((secant(f, 1.0, 2.0, 1e-14, 100).unwrap() - want).abs() < 1e-12);
    }

    #[test]
    fn bisection_needs_sign_change() {
        // x²+1 has no real root; the bracket has no sign change.
        assert!(bisection(|x| x * x + 1.0, -1.0, 1.0, 1e-9, 100).is_none());
    }

    #[test]
    fn newton_handles_cube_root() {
        // root of x³ - 27 is 3.
        let r = newton(|x| x * x * x - 27.0, |x| 3.0 * x * x, 5.0, 1e-12, 100).unwrap();
        assert!((r - 3.0).abs() < 1e-9);
    }
}
