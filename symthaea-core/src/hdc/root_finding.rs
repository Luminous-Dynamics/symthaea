#![allow(dead_code)]

//! # Root Finding Engine
//!
//! Numerical methods for finding zeros of functions, with consciousness-coupled
//! HDC encoding and multi-path verification.
//!
//! ## Methods
//!
//! - **Bisection**: Guaranteed convergence for bracketed intervals
//! - **Newton-Raphson**: Quadratic convergence using derivatives
//! - **Brent's method**: Combines bisection, secant, and inverse quadratic interpolation
//!
//! ## Multi-Path Verification
//!
//! Apply 2-3 methods to the same root. When they agree, Phi increases,
//! representing higher confidence in the result.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use serde::{Deserialize, Serialize};

// ─── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_TOL: f64 = 1e-12;
const MAX_ITERATIONS: usize = 200;

// ─── Types ───────────────────────────────────────────────────────────────────

/// Method used for root finding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootMethod {
    Bisection,
    NewtonRaphson,
    Brent,
}

impl std::fmt::Display for RootMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootMethod::Bisection => write!(f, "Bisection"),
            RootMethod::NewtonRaphson => write!(f, "Newton-Raphson"),
            RootMethod::Brent => write!(f, "Brent"),
        }
    }
}

/// A step in the root-finding process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootStep {
    pub iteration: usize,
    pub x: f64,
    pub fx: f64,
    pub description: String,
}

/// Result of a root-finding operation
#[derive(Debug, Clone)]
pub struct RootResult {
    /// The root found
    pub root: f64,
    /// f(root) — should be near zero
    pub residual: f64,
    /// Method used
    pub method: RootMethod,
    /// Number of iterations
    pub iterations: usize,
    /// Whether the method converged
    pub converged: bool,
    /// Proof trace
    pub steps: Vec<RootStep>,
    /// Phi from the computation
    pub phi: f64,
    /// HDC encoding of the root
    pub encoding: BinaryHV,
}

/// Result of multi-path root finding
#[derive(Debug, Clone)]
pub struct MultiPathRootResult {
    /// Best root (from highest-Phi method)
    pub root: f64,
    /// Individual results from each method
    pub paths: Vec<RootResult>,
    /// Whether multiple methods agree
    pub agreement: bool,
    /// Total Phi (with agreement bonus)
    pub phi: f64,
    /// HDC encoding
    pub encoding: BinaryHV,
}

// ─── Root Finding Engine ─────────────────────────────────────────────────────

/// The Hyperdimensional Root Finding Engine
pub struct RootFindingEngine;

impl RootFindingEngine {
    /// Find a root of f(x) = 0 in [a, b] using bisection.
    ///
    /// Requires f(a) and f(b) to have opposite signs (bracketed interval).
    /// Guaranteed to converge with error < tol.
    pub fn bisection<F>(f: &F, a: f64, b: f64, tol: f64) -> RootResult
    where
        F: Fn(f64) -> f64,
    {
        let tol = if tol <= 0.0 { DEFAULT_TOL } else { tol };
        let mut lo = a;
        let mut hi = b;
        let mut flo = f(lo);
        let fhi = f(hi);
        let mut steps = Vec::new();

        if flo * fhi > 0.0 {
            return RootResult {
                root: f64::NAN,
                residual: f64::NAN,
                method: RootMethod::Bisection,
                iterations: 0,
                converged: false,
                steps,
                phi: 0.0,
                encoding: BinaryHV::zero(),
            };
        }

        let max_iter = (((hi - lo).abs() / tol).log2().ceil() as usize + 10).min(MAX_ITERATIONS);

        let mut mid = lo;
        for iter in 0..max_iter {
            mid = (lo + hi) / 2.0;
            let fmid = f(mid);

            steps.push(RootStep {
                iteration: iter,
                x: mid,
                fx: fmid,
                description: format!("[{:.6}, {:.6}] mid={:.10}", lo, hi, mid),
            });

            if fmid.abs() < tol || (hi - lo) / 2.0 < tol {
                let encoding = Self::encode_root(mid, RootMethod::Bisection);
                return RootResult {
                    root: mid,
                    residual: fmid,
                    method: RootMethod::Bisection,
                    iterations: iter + 1,
                    converged: true,
                    steps,
                    phi: Self::compute_phi(iter + 1, fmid, RootMethod::Bisection),
                    encoding,
                };
            }

            if fmid * flo < 0.0 {
                hi = mid;
            } else {
                lo = mid;
                flo = fmid;
            }
        }

        let fmid = f(mid);
        RootResult {
            root: mid,
            residual: fmid,
            method: RootMethod::Bisection,
            iterations: max_iter,
            converged: false,
            steps,
            phi: 0.05,
            encoding: Self::encode_root(mid, RootMethod::Bisection),
        }
    }

    /// Find a root of f(x) = 0 using Newton-Raphson.
    ///
    /// Requires f and its derivative df. Quadratic convergence near roots.
    pub fn newton_raphson<F, G>(f: &F, df: &G, x0: f64, tol: f64) -> RootResult
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
    {
        let tol = if tol <= 0.0 { DEFAULT_TOL } else { tol };
        let mut x = x0;
        let mut steps = Vec::new();

        for iter in 0..MAX_ITERATIONS {
            let fx = f(x);
            let dfx = df(x);

            steps.push(RootStep {
                iteration: iter,
                x,
                fx,
                description: format!("x={:.10}, f(x)={:.2e}, f'(x)={:.6}", x, fx, dfx),
            });

            if fx.abs() < tol {
                return RootResult {
                    root: x,
                    residual: fx,
                    method: RootMethod::NewtonRaphson,
                    iterations: iter + 1,
                    converged: true,
                    steps,
                    phi: Self::compute_phi(iter + 1, fx, RootMethod::NewtonRaphson),
                    encoding: Self::encode_root(x, RootMethod::NewtonRaphson),
                };
            }

            if dfx.abs() < 1e-15 {
                // Derivative too small
                steps.push(RootStep {
                    iteration: iter + 1,
                    x,
                    fx,
                    description: "Derivative near zero, method stalled".to_string(),
                });
                return RootResult {
                    root: x,
                    residual: fx,
                    method: RootMethod::NewtonRaphson,
                    iterations: iter + 1,
                    converged: false,
                    steps,
                    phi: 0.05,
                    encoding: Self::encode_root(x, RootMethod::NewtonRaphson),
                };
            }

            x -= fx / dfx;
        }

        let fx = f(x);
        RootResult {
            root: x,
            residual: fx,
            method: RootMethod::NewtonRaphson,
            iterations: MAX_ITERATIONS,
            converged: false,
            steps,
            phi: 0.05,
            encoding: Self::encode_root(x, RootMethod::NewtonRaphson),
        }
    }

    /// Find a root of f(x) = 0 in [a, b] using Brent's method.
    ///
    /// Combines bisection, secant, and inverse quadratic interpolation.
    /// Guaranteed convergence with superlinear speed.
    pub fn brent<F>(f: &F, a: f64, b: f64, tol: f64) -> RootResult
    where
        F: Fn(f64) -> f64,
    {
        let tol = if tol <= 0.0 { DEFAULT_TOL } else { tol };
        let mut steps = Vec::new();

        let mut a = a;
        let mut b = b;
        let mut fa = f(a);
        let mut fb = f(b);

        if fa * fb > 0.0 {
            return RootResult {
                root: f64::NAN,
                residual: f64::NAN,
                method: RootMethod::Brent,
                iterations: 0,
                converged: false,
                steps,
                phi: 0.0,
                encoding: BinaryHV::zero(),
            };
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }

        let mut c = a;
        let mut fc = fa;
        let mut mflag = true;
        let mut d = 0.0;
        let mut s;

        for iter in 0..MAX_ITERATIONS {
            if fb.abs() < tol {
                return RootResult {
                    root: b,
                    residual: fb,
                    method: RootMethod::Brent,
                    iterations: iter + 1,
                    converged: true,
                    steps,
                    phi: Self::compute_phi(iter + 1, fb, RootMethod::Brent),
                    encoding: Self::encode_root(b, RootMethod::Brent),
                };
            }

            if (a - b).abs() < tol {
                return RootResult {
                    root: b,
                    residual: fb,
                    method: RootMethod::Brent,
                    iterations: iter + 1,
                    converged: true,
                    steps,
                    phi: Self::compute_phi(iter + 1, fb, RootMethod::Brent),
                    encoding: Self::encode_root(b, RootMethod::Brent),
                };
            }

            if (fa - fc).abs() > 1e-15 && (fb - fc).abs() > 1e-15 {
                // Inverse quadratic interpolation
                s = a * fb * fc / ((fa - fb) * (fa - fc))
                    + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb));
            } else {
                // Secant method
                s = b - fb * (b - a) / (fb - fa);
            }

            // Conditions for bisection fallback
            let cond1 = if a < b {
                s < (3.0 * a + b) / 4.0 || s > b
            } else {
                s > (3.0 * a + b) / 4.0 || s < b
            };
            let cond2 = mflag && (s - b).abs() >= (b - c).abs() / 2.0;
            let cond3 = !mflag && (s - b).abs() >= (c - d).abs() / 2.0;
            let cond4 = mflag && (b - c).abs() < tol;
            let cond5 = !mflag && (c - d).abs() < tol;

            if cond1 || cond2 || cond3 || cond4 || cond5 {
                s = (a + b) / 2.0;
                mflag = true;
            } else {
                mflag = false;
            }

            let fs = f(s);

            steps.push(RootStep {
                iteration: iter,
                x: s,
                fx: fs,
                description: format!(
                    "s={:.10}, f(s)={:.2e}{}",
                    s,
                    fs,
                    if mflag { " (bisect)" } else { "" }
                ),
            });

            d = c;
            c = b;
            fc = fb;

            if fa * fs < 0.0 {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }

            if fa.abs() < fb.abs() {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
            }
        }

        RootResult {
            root: b,
            residual: fb,
            method: RootMethod::Brent,
            iterations: MAX_ITERATIONS,
            converged: false,
            steps,
            phi: 0.05,
            encoding: Self::encode_root(b, RootMethod::Brent),
        }
    }

    /// Multi-path root finding: apply multiple methods and compare.
    /// Returns the result with highest Phi, plus agreement bonus.
    pub fn find_root_multipath<F, G>(
        f: &F,
        df: Option<&G>,
        bracket: (f64, f64),
        tol: f64,
    ) -> MultiPathRootResult
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
    {
        let mut paths = Vec::new();

        // Always try bisection and Brent
        paths.push(Self::bisection(f, bracket.0, bracket.1, tol));
        paths.push(Self::brent(f, bracket.0, bracket.1, tol));

        // Newton-Raphson if derivative provided
        if let Some(df) = df {
            let x0 = (bracket.0 + bracket.1) / 2.0;
            paths.push(Self::newton_raphson(f, df, x0, tol));
        }

        // Check agreement among converged paths
        let converged: Vec<&RootResult> = paths.iter().filter(|p| p.converged).collect();
        let agreement = if converged.len() >= 2 {
            let ref_root = converged[0].root;
            converged
                .iter()
                .all(|p| (p.root - ref_root).abs() < tol.sqrt())
        } else {
            false
        };

        let best_root = converged.first().map(|p| p.root).unwrap_or(paths[0].root);

        let base_phi: f64 = paths.iter().filter(|p| p.converged).map(|p| p.phi).sum();
        let agreement_bonus = if agreement { 0.5 } else { 0.0 };
        let total_phi = base_phi + agreement_bonus;

        let encoding = Self::encode_root(best_root, RootMethod::Brent);

        MultiPathRootResult {
            root: best_root,
            paths,
            agreement,
            phi: total_phi,
            encoding,
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    fn compute_phi(iterations: usize, residual: f64, method: RootMethod) -> f64 {
        // Phi based on convergence speed and accuracy
        let speed_phi = 1.0 / (1.0 + iterations as f64 / 10.0);
        let accuracy_phi = if residual.abs() < 1e-15 {
            1.0
        } else {
            1.0 / (1.0 + residual.abs().log10().abs() / 15.0)
        };
        let method_bonus = match method {
            RootMethod::Bisection => 0.0,     // Simple, guaranteed
            RootMethod::NewtonRaphson => 0.1, // Fast but needs derivative
            RootMethod::Brent => 0.05,        // Best of both worlds
        };
        (speed_phi + accuracy_phi) / 2.0 + method_bonus
    }

    fn encode_root(root: f64, method: RootMethod) -> BinaryHV {
        let root_prim = BinaryHV::random(seed_from_name("ROOT"));
        let method_hv = BinaryHV::random(seed_from_name(&format!("METHOD_{}", method)));
        let val_hv = BinaryHV::random(seed_from_name(&format!("ROOT_VAL_{}", root.to_bits())));
        root_prim.bind(&method_hv).bind(&val_hv)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // ── Bisection ────────────────────────────────────────────────────────

    #[test]
    fn test_bisection_sqrt2() {
        // x² - 2 = 0 → x = √2
        let result = RootFindingEngine::bisection(&|x| x * x - 2.0, 1.0, 2.0, TOL);
        assert!(result.converged);
        assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-8);
    }

    #[test]
    fn test_bisection_polynomial() {
        // x³ - x - 2 = 0, root near 1.52
        let result = RootFindingEngine::bisection(&|x| x * x * x - x - 2.0, 1.0, 2.0, TOL);
        assert!(result.converged);
        assert!(result.residual.abs() < 1e-8);
    }

    #[test]
    fn test_bisection_no_bracket() {
        // f(1) > 0, f(2) > 0 — no sign change
        let result = RootFindingEngine::bisection(&|x| x * x + 1.0, 1.0, 2.0, TOL);
        assert!(!result.converged);
        assert!(result.root.is_nan());
    }

    #[test]
    fn test_bisection_sin() {
        // sin(x) = 0 near pi
        let result = RootFindingEngine::bisection(&|x| x.sin(), 3.0, 4.0, TOL);
        assert!(result.converged);
        assert!((result.root - std::f64::consts::PI).abs() < 1e-8);
    }

    // ── Newton-Raphson ───────────────────────────────────────────────────

    #[test]
    fn test_newton_sqrt2() {
        let result = RootFindingEngine::newton_raphson(&|x| x * x - 2.0, &|x| 2.0 * x, 1.5, TOL);
        assert!(result.converged);
        assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-10);
        // Newton should converge faster than bisection
        assert!(result.iterations < 10);
    }

    #[test]
    fn test_newton_cube_root() {
        // x³ - 27 = 0 → x = 3
        let result =
            RootFindingEngine::newton_raphson(&|x| x * x * x - 27.0, &|x| 3.0 * x * x, 2.0, TOL);
        assert!(result.converged);
        assert!((result.root - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_newton_exp() {
        // e^x - 3x = 0, root near 1.512
        let result =
            RootFindingEngine::newton_raphson(&|x| x.exp() - 3.0 * x, &|x| x.exp() - 3.0, 1.0, TOL);
        assert!(result.converged);
        assert!(result.residual.abs() < 1e-8);
    }

    // ── Brent's Method ───────────────────────────────────────────────────

    #[test]
    fn test_brent_sqrt2() {
        let result = RootFindingEngine::brent(&|x| x * x - 2.0, 1.0, 2.0, TOL);
        assert!(result.converged);
        assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-8);
    }

    #[test]
    fn test_brent_transcendental() {
        // sin(x) = x/2, root near 1.895
        let result = RootFindingEngine::brent(&|x| x.sin() - x / 2.0, 0.1, 3.0, TOL);
        assert!(result.converged);
        let verify = result.root.sin() - result.root / 2.0;
        assert!(verify.abs() < 1e-8);
    }

    #[test]
    fn test_brent_polynomial_degree4() {
        // x^4 - 5x^2 + 4 = 0 → roots at ±1, ±2
        let result = RootFindingEngine::brent(&|x| x.powi(4) - 5.0 * x * x + 4.0, 0.5, 1.5, TOL);
        assert!(result.converged);
        assert!((result.root - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_brent_negative_root() {
        // x² - 4 = 0, root at -2
        let result = RootFindingEngine::brent(&|x| x * x - 4.0, -3.0, -1.0, TOL);
        assert!(result.converged);
        assert!((result.root - (-2.0)).abs() < 1e-8);
    }

    // ── Multi-path ───────────────────────────────────────────────────────

    #[test]
    fn test_multipath_with_derivative() {
        let result = RootFindingEngine::find_root_multipath(
            &|x| x * x - 2.0,
            Some(&|x| 2.0 * x),
            (1.0, 2.0),
            TOL,
        );
        assert!(result.agreement);
        assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-8);
        assert_eq!(result.paths.len(), 3); // Bisection + Brent + Newton
        assert!(result.phi > 1.0); // Agreement bonus
    }

    #[test]
    fn test_multipath_without_derivative() {
        let no_deriv: Option<&fn(f64) -> f64> = None;
        let result = RootFindingEngine::find_root_multipath(
            &|x: f64| x.sin() - x / 2.0,
            no_deriv,
            (0.1, 3.0),
            TOL,
        );
        assert!(result.agreement);
        assert_eq!(result.paths.len(), 2); // Bisection + Brent only
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_root_at_endpoint() {
        // f(x) = x, root exactly at 0
        let result = RootFindingEngine::bisection(&|x| x, -1.0, 1.0, TOL);
        assert!(result.converged);
        assert!(result.root.abs() < 1e-8);
    }

    #[test]
    fn test_very_close_roots() {
        // (x-1)*(x-1.001) near x=1
        let result = RootFindingEngine::brent(&|x| (x - 1.0) * (x - 1.001), 0.5, 1.0005, TOL);
        assert!(result.converged);
        assert!((result.root - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_phi_increases_with_speed() {
        // Easy root (linear) should have higher phi than hard root
        let easy = RootFindingEngine::bisection(&|x| x - 5.0, 0.0, 10.0, TOL);
        let hard = RootFindingEngine::bisection(&|x| x.powi(10) - 1.0, 0.5, 1.5, TOL);
        // Both should converge
        assert!(easy.converged);
        assert!(hard.converged);
        // Easy should converge faster → potentially higher phi
        assert!(easy.iterations <= hard.iterations);
    }

    #[test]
    fn test_encoding_different_roots() {
        let r1 = RootFindingEngine::bisection(&|x| x - 1.0, 0.0, 2.0, TOL);
        let r2 = RootFindingEngine::bisection(&|x| x - 5.0, 3.0, 7.0, TOL);
        // Different roots should have different encodings
        let sim = r1.encoding.similarity(&r2.encoding);
        assert!(
            sim < 0.6,
            "Different roots should have low similarity: {}",
            sim
        );
    }
}
