// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//! - **Secant**: Superlinear convergence without derivatives
//!
//! ## Multi-Root Finding
//!
//! `find_roots_in_interval` subdivides an interval, detects sign changes, and
//! solves each bracket with Brent's method.
//!
//! ## Multi-Path Verification
//!
//! Apply 2-3 methods to the same root. When they agree, Phi increases,
//! representing higher confidence in the result.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_TOL: f64 = 1e-12;
const MAX_ITERATIONS: usize = 200;
const DEFAULT_BRACKET_EXPAND: usize = 50;
const DEFAULT_SUBDIVISIONS: usize = 100;

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors that can occur during root finding.
#[derive(Debug, Clone, PartialEq)]
pub enum RootFindingError {
    /// The interval [a, b] does not bracket a root (f(a) and f(b) have the same sign).
    NotBracketed { fa: f64, fb: f64 },
    /// The method did not converge within the maximum number of iterations.
    MaxIterations {
        last_x: f64,
        last_fx: f64,
        iterations: usize,
    },
    /// The derivative is zero (or near-zero), preventing the method from proceeding.
    ZeroDerivative { x: f64, fx: f64, dfx: f64 },
    /// No sign change found after expanding the bracket.
    BracketExpansionFailed { a: f64, b: f64 },
}

impl fmt::Display for RootFindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootFindingError::NotBracketed { fa, fb } => {
                write!(
                    f,
                    "interval not bracketed: f(a)={fa:.6e}, f(b)={fb:.6e} (same sign)"
                )
            }
            RootFindingError::MaxIterations {
                last_x,
                last_fx,
                iterations,
            } => {
                write!(
                    f,
                    "max iterations ({iterations}) reached: x={last_x:.10}, f(x)={last_fx:.6e}"
                )
            }
            RootFindingError::ZeroDerivative { x, fx, dfx } => {
                write!(
                    f,
                    "zero derivative at x={x:.10}: f(x)={fx:.6e}, f'(x)={dfx:.6e}"
                )
            }
            RootFindingError::BracketExpansionFailed { a, b } => {
                write!(
                    f,
                    "bracket expansion failed: no sign change in [{a:.6}, {b:.6}]"
                )
            }
        }
    }
}

impl std::error::Error for RootFindingError {}

// ─── Types ───────────────────────────────────────────────────────────────────

/// Method used for root finding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootMethod {
    Bisection,
    NewtonRaphson,
    Brent,
    Secant,
}

impl fmt::Display for RootMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootMethod::Bisection => write!(f, "Bisection"),
            RootMethod::NewtonRaphson => write!(f, "Newton-Raphson"),
            RootMethod::Brent => write!(f, "Brent"),
            RootMethod::Secant => write!(f, "Secant"),
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

            if dfx.abs() < 1e-12 * fx.abs().max(1.0) {
                // Derivative too small relative to function value
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

    /// Find a root of f(x) = 0 using the secant method.
    ///
    /// Uses two initial points x0 and x1 (no derivative needed).
    /// Superlinear convergence (order ~1.618, the golden ratio).
    pub fn secant<F>(
        f: &F,
        x0: f64,
        x1: f64,
        tol: f64,
        max_iter: usize,
    ) -> Result<RootResult, RootFindingError>
    where
        F: Fn(f64) -> f64,
    {
        let tol = if tol <= 0.0 { DEFAULT_TOL } else { tol };
        let max_iter = if max_iter == 0 {
            MAX_ITERATIONS
        } else {
            max_iter
        };

        let mut xprev = x0;
        let mut xcurr = x1;
        let mut fprev = f(xprev);
        let mut fcurr = f(xcurr);
        let mut steps = Vec::new();

        for iter in 0..max_iter {
            steps.push(RootStep {
                iteration: iter,
                x: xcurr,
                fx: fcurr,
                description: format!("x={:.10}, f(x)={:.2e}", xcurr, fcurr),
            });

            if fcurr.abs() < tol {
                return Ok(RootResult {
                    root: xcurr,
                    residual: fcurr,
                    method: RootMethod::Secant,
                    iterations: iter + 1,
                    converged: true,
                    steps,
                    phi: Self::compute_phi(iter + 1, fcurr, RootMethod::Secant),
                    encoding: Self::encode_root(xcurr, RootMethod::Secant),
                });
            }

            let denom = fcurr - fprev;
            if denom.abs() < 1e-12 * fcurr.abs().max(1.0) {
                return Err(RootFindingError::ZeroDerivative {
                    x: xcurr,
                    fx: fcurr,
                    dfx: denom,
                });
            }

            let xnext = xcurr - fcurr * (xcurr - xprev) / denom;
            xprev = xcurr;
            fprev = fcurr;
            xcurr = xnext;
            fcurr = f(xcurr);
        }

        Err(RootFindingError::MaxIterations {
            last_x: xcurr,
            last_fx: fcurr,
            iterations: max_iter,
        })
    }

    /// Find all roots of f(x) = 0 in [a, b] by subdividing the interval.
    ///
    /// Divides [a, b] into `n_subdivisions` sub-intervals, identifies those with
    /// sign changes, and solves each with Brent's method. Deduplicates roots
    /// that are closer than sqrt(tol).
    pub fn find_roots_in_interval<F>(
        f: &F,
        a: f64,
        b: f64,
        n_subdivisions: usize,
        tol: f64,
    ) -> Vec<RootResult>
    where
        F: Fn(f64) -> f64,
    {
        let n = if n_subdivisions == 0 {
            DEFAULT_SUBDIVISIONS
        } else {
            n_subdivisions
        };
        let tol = if tol <= 0.0 { DEFAULT_TOL } else { tol };
        let h = (b - a) / n as f64;
        let dedup_dist = tol.sqrt();

        let mut roots = Vec::new();

        for i in 0..n {
            let lo = a + i as f64 * h;
            let hi = a + (i + 1) as f64 * h;
            let flo = f(lo);
            let fhi = f(hi);

            // Check for exact zero at left endpoint (avoid duplicates)
            if flo.abs() < tol {
                let is_dup = roots
                    .last()
                    .is_some_and(|prev: &RootResult| (prev.root - lo).abs() < dedup_dist);
                if !is_dup {
                    roots.push(RootResult {
                        root: lo,
                        residual: flo,
                        method: RootMethod::Brent,
                        iterations: 0,
                        converged: true,
                        steps: vec![],
                        phi: 1.0,
                        encoding: Self::encode_root(lo, RootMethod::Brent),
                    });
                }
                continue;
            }

            // Sign change means a root exists in (lo, hi)
            if flo * fhi < 0.0 {
                let result = Self::brent(f, lo, hi, tol);
                if result.converged {
                    let is_dup = roots.last().is_some_and(|prev: &RootResult| {
                        (prev.root - result.root).abs() < dedup_dist
                    });
                    if !is_dup {
                        roots.push(result);
                    }
                }
            }
        }

        roots
    }

    /// Expand the interval [a, b] until a sign change of f is found.
    ///
    /// Geometrically expands outward (factor 1.5x each step) up to `max_expand`
    /// iterations. Returns the expanded bracket or an error if no sign change is found.
    pub fn bracket_root<F>(
        f: &F,
        a: f64,
        b: f64,
        max_expand: usize,
    ) -> Result<(f64, f64), RootFindingError>
    where
        F: Fn(f64) -> f64,
    {
        let max_expand = if max_expand == 0 {
            DEFAULT_BRACKET_EXPAND
        } else {
            max_expand
        };
        let mut lo = a;
        let mut hi = b;

        // Already bracketed?
        if f(lo) * f(hi) <= 0.0 {
            return Ok((lo, hi));
        }

        for _ in 0..max_expand {
            let width = (hi - lo).abs();
            lo -= width * 0.5;
            hi += width * 0.5;

            if f(lo) * f(hi) <= 0.0 {
                return Ok((lo, hi));
            }
        }

        Err(RootFindingError::BracketExpansionFailed { a: lo, b: hi })
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
            RootMethod::Secant => 0.07,       // Superlinear, no derivative
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

    // ── Polynomial roots: x^2 - 4 = 0 (roots at +/-2) ──────────────────

    #[test]
    fn test_x_squared_minus_4_positive_root() {
        let result = RootFindingEngine::bisection(&|x| x * x - 4.0, 0.0, 3.0, TOL);
        assert!(result.converged);
        assert!(
            (result.root - 2.0).abs() < 1e-8,
            "expected 2.0, got {}",
            result.root
        );
    }

    #[test]
    fn test_x_squared_minus_4_negative_root() {
        let result = RootFindingEngine::brent(&|x| x * x - 4.0, -3.0, -1.0, TOL);
        assert!(result.converged);
        assert!(
            (result.root - (-2.0)).abs() < 1e-8,
            "expected -2.0, got {}",
            result.root
        );
    }

    // ── Polynomial roots: x^3 - x = 0 (roots at -1, 0, 1) ─────────────

    #[test]
    fn test_x_cubed_minus_x_all_roots() {
        let f = |x: f64| x * x * x - x;
        let roots = RootFindingEngine::find_roots_in_interval(&f, -1.5, 1.5, 200, TOL);
        assert!(roots.len() >= 3, "expected 3 roots, found {}", roots.len());

        let mut found: Vec<f64> = roots.iter().map(|r| r.root).collect();
        found.sort_by(|a, b| a.total_cmp(b));
        assert!(
            (found[0] - (-1.0)).abs() < 1e-6,
            "expected -1, got {}",
            found[0]
        );
        assert!(found[1].abs() < 1e-6, "expected 0, got {}", found[1]);
        assert!(
            (found[2] - 1.0).abs() < 1e-6,
            "expected 1, got {}",
            found[2]
        );
    }

    // ── Transcendental: sin(x) = 0 ──────────────────────────────────────

    #[test]
    fn test_sin_root_at_pi() {
        let result = RootFindingEngine::bisection(&|x| x.sin(), 3.0, 4.0, TOL);
        assert!(result.converged);
        assert!((result.root - std::f64::consts::PI).abs() < 1e-8);
    }

    #[test]
    fn test_sin_multiple_roots() {
        // sin(x) has roots at 0, pi, 2pi, 3pi in [-0.5, 10]
        let roots =
            RootFindingEngine::find_roots_in_interval(&|x: f64| x.sin(), -0.5, 10.0, 500, TOL);
        assert!(
            roots.len() >= 3,
            "expected at least 3 sin roots, found {}",
            roots.len()
        );
    }

    // ── Transcendental: e^x = 3x ────────────────────────────────────────

    #[test]
    fn test_exp_equals_3x() {
        let result =
            RootFindingEngine::newton_raphson(&|x| x.exp() - 3.0 * x, &|x| x.exp() - 3.0, 1.0, TOL);
        assert!(result.converged);
        assert!(result.residual.abs() < 1e-8);
    }

    // ── Transcendental: cos(x) = x (Dottie number) ─────────────────────

    #[test]
    fn test_cos_equals_x() {
        // cos(x) - x = 0, Dottie number ~ 0.739085
        let result = RootFindingEngine::brent(&|x: f64| x.cos() - x, 0.0, 1.5, TOL);
        assert!(result.converged);
        assert!(
            (result.root - 0.739085).abs() < 1e-4,
            "Dottie number: got {}",
            result.root
        );
    }

    // ── Edge case: double root ───────────────────────────────────────────

    #[test]
    fn test_double_root() {
        // (x-3)^2 = x^2 - 6x + 9, double root at x = 3
        // No sign change, so bisection/Brent fail. Newton handles it (linear convergence).
        let result = RootFindingEngine::newton_raphson(
            &|x| (x - 3.0) * (x - 3.0),
            &|x| 2.0 * (x - 3.0),
            2.5,
            1e-6,
        );
        assert!(result.converged, "Newton should converge on double root");
        assert!(
            (result.root - 3.0).abs() < 1e-2,
            "expected 3.0, got {}",
            result.root
        );
    }

    // ── Edge case: nearly-flat function ──────────────────────────────────

    #[test]
    fn test_nearly_flat_function() {
        // f(x) = x^11 near zero — very flat derivative region
        let result = RootFindingEngine::bisection(&|x: f64| x.powi(11), -1.0, 1.0, 1e-6);
        assert!(result.converged);
        assert!(
            result.root.abs() < 1e-4,
            "flat root near 0: got {}",
            result.root
        );
    }

    // ── Convergence: Newton faster than bisection ────────────────────────

    #[test]
    fn test_newton_converges_faster_than_bisection() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let tol = 1e-12;

        let bisect = RootFindingEngine::bisection(&f, 1.0, 2.0, tol);
        let newton = RootFindingEngine::newton_raphson(&f, &df, 1.5, tol);

        assert!(bisect.converged);
        assert!(newton.converged);
        assert!(
            newton.iterations < bisect.iterations,
            "Newton ({} iters) should be faster than bisection ({} iters)",
            newton.iterations,
            bisect.iterations,
        );
    }

    // ── All methods agree on same root ───────────────────────────────────

    #[test]
    fn test_all_methods_agree() {
        let f = |x: f64| x * x * x - x - 2.0;
        let df = |x: f64| 3.0 * x * x - 1.0;
        let tol = 1e-8;

        let bisect = RootFindingEngine::bisection(&f, 1.0, 2.0, tol);
        let newton = RootFindingEngine::newton_raphson(&f, &df, 1.5, tol);
        let brent = RootFindingEngine::brent(&f, 1.0, 2.0, tol);
        let secant = RootFindingEngine::secant(&f, 1.0, 2.0, tol, 100).unwrap();

        assert!(bisect.converged);
        assert!(newton.converged);
        assert!(brent.converged);
        assert!(secant.converged);

        let roots = [bisect.root, newton.root, brent.root, secant.root];
        for i in 0..roots.len() {
            for j in (i + 1)..roots.len() {
                assert!(
                    (roots[i] - roots[j]).abs() < 1e-6,
                    "methods disagree: root[{}]={}, root[{}]={}",
                    i,
                    roots[i],
                    j,
                    roots[j],
                );
            }
        }
    }

    // ── Error cases ──────────────────────────────────────────────────────

    #[test]
    fn test_bisection_not_bracketed() {
        // x^2 + 1 > 0 everywhere
        let result = RootFindingEngine::bisection(&|x| x * x + 1.0, 1.0, 2.0, TOL);
        assert!(!result.converged);
        assert!(result.root.is_nan());
    }

    #[test]
    fn test_brent_not_bracketed() {
        let result = RootFindingEngine::brent(&|x| x * x + 1.0, 1.0, 2.0, TOL);
        assert!(!result.converged);
        assert!(result.root.is_nan());
    }

    #[test]
    fn test_newton_zero_derivative() {
        // Deliberately pass a zero derivative function
        let result = RootFindingEngine::newton_raphson(&|x: f64| x * x * x, &|_x| 0.0, 1.0, TOL);
        assert!(!result.converged);
    }

    #[test]
    fn test_secant_zero_denominator() {
        // f(x) = constant => f(x0) == f(x1), denominator is zero
        let err = RootFindingEngine::secant(&|_x| 5.0, 1.0, 2.0, TOL, 100);
        assert!(err.is_err());
        match err.unwrap_err() {
            RootFindingError::ZeroDerivative { .. } => {}
            other => panic!("expected ZeroDerivative, got {:?}", other),
        }
    }

    // ── Secant method ────────────────────────────────────────────────────

    #[test]
    fn test_secant_sqrt2() {
        let result = RootFindingEngine::secant(&|x| x * x - 2.0, 1.0, 2.0, TOL, 100).unwrap();
        assert!(result.converged);
        assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-8);
    }

    #[test]
    fn test_secant_cos_equals_x() {
        let result = RootFindingEngine::secant(&|x: f64| x.cos() - x, 0.0, 1.0, TOL, 100).unwrap();
        assert!(result.converged);
        assert!((result.root - 0.739085).abs() < 1e-4);
    }

    // ── Multi-root finding ───────────────────────────────────────────────

    #[test]
    fn test_find_roots_degree4_polynomial() {
        // x^4 - 5x^2 + 4 = (x-1)(x+1)(x-2)(x+2), roots at +/-1 and +/-2
        let f = |x: f64| x.powi(4) - 5.0 * x * x + 4.0;
        let roots = RootFindingEngine::find_roots_in_interval(&f, -3.0, 3.0, 200, TOL);
        assert!(roots.len() >= 4, "expected 4 roots, found {}", roots.len());
    }

    // ── Bracket root utility ─────────────────────────────────────────────

    #[test]
    fn test_bracket_root_already_bracketed() {
        let bracket = RootFindingEngine::bracket_root(&|x: f64| x - 5.0, 0.0, 10.0, 50);
        assert!(bracket.is_ok());
    }

    #[test]
    fn test_bracket_root_expansion() {
        // Start with [3, 4] for f(x) = x - 5 (no sign change). Should expand to include 5.
        let bracket = RootFindingEngine::bracket_root(&|x: f64| x - 5.0, 3.0, 4.0, 50);
        assert!(bracket.is_ok());
        let (lo, hi) = bracket.unwrap();
        let flo = lo - 5.0;
        let fhi = hi - 5.0;
        assert!(
            flo * fhi <= 0.0,
            "expanded bracket should contain sign change"
        );
    }

    #[test]
    fn test_bracket_root_impossible() {
        // x^2 + 1 > 0 everywhere
        let bracket = RootFindingEngine::bracket_root(&|x: f64| x * x + 1.0, 0.0, 1.0, 10);
        assert!(bracket.is_err());
        match bracket.unwrap_err() {
            RootFindingError::BracketExpansionFailed { .. } => {}
            other => panic!("expected BracketExpansionFailed, got {:?}", other),
        }
    }

    // ── Multi-path verification ──────────────────────────────────────────

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
        assert_eq!(result.paths.len(), 3);
        assert!(result.phi > 1.0);
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
        assert_eq!(result.paths.len(), 2);
    }

    // ── HDC encoding ─────────────────────────────────────────────────────

    #[test]
    fn test_encoding_different_roots() {
        let r1 = RootFindingEngine::bisection(&|x| x - 1.0, 0.0, 2.0, TOL);
        let r2 = RootFindingEngine::bisection(&|x| x - 5.0, 3.0, 7.0, TOL);
        let sim = r1.encoding.similarity(&r2.encoding);
        assert!(
            sim < 0.6,
            "Different roots should have low similarity: {}",
            sim
        );
    }

    // ── Error display ────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let err = RootFindingError::NotBracketed { fa: 1.0, fb: 2.0 };
        let msg = format!("{}", err);
        assert!(msg.contains("not bracketed"), "error message: {}", msg);

        let err = RootFindingError::MaxIterations {
            last_x: 1.5,
            last_fx: 0.01,
            iterations: 200,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("200"), "error message: {}", msg);
    }
}
