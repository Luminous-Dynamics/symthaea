// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Optimization Engine
//!
//! Numerical optimization methods with consciousness-coupled HDC encoding.
//!
//! ## Methods
//!
//! - **Gradient descent**: with backtracking Armijo line search and momentum
//! - **Nelder-Mead simplex**: derivative-free optimization (reflection, expansion, contraction, shrink)
//! - **L-BFGS**: limited-memory quasi-Newton (m=10 history) for large-scale problems
//! - **Penalty method**: constrained optimization via quadratic penalty functions
//!
//! ## ObjectiveFunction Trait
//!
//! Implement [`ObjectiveFunction`] for type-safe optimization, or use closures directly.
//! The trait provides `eval()` (required) and `gradient()` (optional, with finite-difference fallback).
//!
//! ## Convenience
//!
//! [`minimize`] dispatches to the appropriate solver based on [`OptMethod`].
//!
//! ## References
//!
//! - Nocedal & Wright (2006) — *Numerical Optimization*, Springer (L-BFGS, Armijo)
//! - Nelder & Mead (1965) — A simplex method for function minimization. *Computer Journal*.
//! - Kanerva (2009) — Hyperdimensional computing encodings

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use serde::{Deserialize, Serialize};

// ─── Constants ───────────────────────────────────────────────────────────────

const MAX_ITERATIONS: usize = 1000;
const DEFAULT_TOL: f64 = 1e-10;
const DEFAULT_LEARNING_RATE: f64 = 0.01;
const LBFGS_MEMORY: usize = 10;
const ARMIJO_C1: f64 = 1e-4;
const ARMIJO_SHRINK: f64 = 0.5;
const ARMIJO_MAX_BACKTRACKS: usize = 30;
const FINITE_DIFF_EPS: f64 = 1e-7;
/// Default gradient clipping norm. Prevents catastrophic steps in
/// ill-conditioned problems (e.g., exp(100x) near x=0).
const DEFAULT_GRADIENT_CLIP_NORM: f64 = 100.0;

// ─── ObjectiveFunction Trait ─────────────────────────────────────────────────

/// Trait for objective functions that can be optimized.
///
/// Implement `eval` (required) and optionally `gradient` for gradient-based methods.
/// If `gradient` is not provided, a finite-difference approximation is used.
pub trait ObjectiveFunction {
    /// Evaluate the objective function at point `x`.
    fn eval(&self, x: &[f64]) -> f64;

    /// Compute the gradient at point `x`. Returns `None` to use finite differences.
    fn gradient(&self, x: &[f64]) -> Option<Vec<f64>> {
        let _ = x;
        None
    }
}

/// Clip gradient vector to a maximum L2 norm, rescaling if needed.
/// Returns the (possibly clipped) gradient and its norm.
fn clip_gradient(g: &mut [f64], max_norm: f64) -> f64 {
    let norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for gi in g.iter_mut() {
            *gi *= scale;
        }
        max_norm
    } else {
        norm
    }
}

/// Compute the gradient via central finite differences.
/// Uses adaptive per-component step size: eps_i = sqrt(machine_eps) * max(|x_i|, 1)
/// for better accuracy across different variable scales (Nocedal & Wright 2006, §8.1).
fn numerical_gradient<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    let base_eps = f64::EPSILON.sqrt(); // ~1.49e-8, optimal for central differences
    for i in 0..n {
        let eps_i = base_eps * x[i].abs().max(1.0);
        x_plus[i] = x[i] + eps_i;
        x_minus[i] = x[i] - eps_i;
        grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * eps_i);
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }
    grad
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// Optimization method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptMethod {
    GradientDescent,
    NelderMead,
    LBFGS,
}

/// A step in the optimization process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptStep {
    pub iteration: usize,
    pub x: Vec<f64>,
    pub fx: f64,
    pub grad_norm: Option<f64>,
}

/// Result of an optimization
#[derive(Debug, Clone)]
pub struct OptResult {
    /// Optimal point found
    pub x: Vec<f64>,
    /// Function value at optimum
    pub fx: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether the method converged
    pub converged: bool,
    /// Method used
    pub method: OptMethod,
    /// History of optimization steps
    pub history: Vec<OptStep>,
    /// Phi measurement
    pub phi: f64,
    /// HDC encoding
    pub encoding: BinaryHV,
}

/// Convenience type alias matching the task spec naming convention.
pub type OptimizationResult = OptResult;

// ─── Box Constraints ─────────────────────────────────────────────────────────

/// Simple box constraints: lower[i] <= x[i] <= upper[i].
#[derive(Debug, Clone)]
pub struct BoxConstraints {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// Type alias for boxed constraint functions.
type ConstraintFn = Box<dyn Fn(&[f64]) -> f64>;

impl BoxConstraints {
    /// Create box constraints. Each dimension i is constrained to [lower[i], upper[i]].
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        assert_eq!(
            lower.len(),
            upper.len(),
            "Lower and upper bounds must have same length"
        );
        Self { lower, upper }
    }

    /// Project a point onto the feasible box.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| xi.max(self.lower[i]).min(self.upper[i]))
            .collect()
    }

    /// Convert to penalty-style inequality constraints: g_i(x) <= 0.
    /// For each dimension, two constraints: x[i] - upper[i] <= 0 and lower[i] - x[i] <= 0.
    pub fn to_penalty_constraints(&self) -> Vec<ConstraintFn> {
        let n = self.lower.len();
        let mut constraints: Vec<ConstraintFn> = Vec::with_capacity(2 * n);
        for i in 0..n {
            let lo = self.lower[i];
            let hi = self.upper[i];
            constraints.push(Box::new(move |x: &[f64]| x[i] - hi)); // x[i] <= upper[i]
            constraints.push(Box::new(move |x: &[f64]| lo - x[i])); // lower[i] <= x[i]
        }
        constraints
    }
}

// ─── Minimize Dispatcher ─────────────────────────────────────────────────────

/// Convenience dispatcher: minimize `f` starting from `x0` using the given method.
///
/// For gradient-based methods (GradientDescent, LBFGS), computes gradients via
/// central finite differences if not provided.
pub fn minimize<F>(f: F, x0: &[f64], method: OptMethod) -> OptResult
where
    F: Fn(&[f64]) -> f64,
{
    minimize_with_tol(f, x0, method, DEFAULT_TOL)
}

/// Like [`minimize`] but with explicit tolerance.
pub fn minimize_with_tol<F>(f: F, x0: &[f64], method: OptMethod, tol: f64) -> OptResult
where
    F: Fn(&[f64]) -> f64,
{
    match method {
        OptMethod::GradientDescent => {
            let grad = |x: &[f64]| numerical_gradient(&f, x);
            OptimizationEngine::gradient_descent(&f, &grad, x0, DEFAULT_LEARNING_RATE, 0.0, tol)
        }
        OptMethod::NelderMead => OptimizationEngine::nelder_mead(&f, x0, 1.0, tol),
        OptMethod::LBFGS => {
            let grad = |x: &[f64]| numerical_gradient(&f, x);
            OptimizationEngine::lbfgs(&f, &grad, x0, tol)
        }
    }
}

/// Minimize an [`ObjectiveFunction`] impl starting from `x0`.
pub fn minimize_objective(obj: &dyn ObjectiveFunction, x0: &[f64], method: OptMethod) -> OptResult {
    let f = |x: &[f64]| obj.eval(x);
    let has_analytic_grad = obj.gradient(x0).is_some();

    match method {
        OptMethod::GradientDescent => {
            let grad = |x: &[f64]| obj.gradient(x).unwrap_or_else(|| numerical_gradient(&f, x));
            OptimizationEngine::gradient_descent(
                &f,
                &grad,
                x0,
                DEFAULT_LEARNING_RATE,
                0.0,
                DEFAULT_TOL,
            )
        }
        OptMethod::NelderMead => OptimizationEngine::nelder_mead(&f, x0, 1.0, DEFAULT_TOL),
        OptMethod::LBFGS => {
            let grad = |x: &[f64]| {
                if has_analytic_grad {
                    obj.gradient(x).unwrap_or_else(|| numerical_gradient(&f, x))
                } else {
                    numerical_gradient(&f, x)
                }
            };
            OptimizationEngine::lbfgs(&f, &grad, x0, DEFAULT_TOL)
        }
    }
}

// ─── Optimization Engine ─────────────────────────────────────────────────────

/// The Hyperdimensional Optimization Engine
pub struct OptimizationEngine;

impl OptimizationEngine {
    /// Gradient descent with optional momentum.
    ///
    /// `f`: objective function
    /// `grad`: gradient function
    /// `x0`: initial point
    /// `lr`: learning rate
    /// `momentum`: momentum coefficient (0 = no momentum)
    pub fn gradient_descent<F, G>(
        f: &F,
        grad: &G,
        x0: &[f64],
        lr: f64,
        momentum: f64,
        tol: f64,
    ) -> OptResult
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let lr = if lr <= 0.0 { DEFAULT_LEARNING_RATE } else { lr };
        let tol = if tol <= 0.0 { DEFAULT_TOL } else { tol };
        let n = x0.len();

        let mut x = x0.to_vec();
        let mut velocity = vec![0.0; n];
        let mut history = Vec::new();

        for iter in 0..MAX_ITERATIONS {
            let fx = f(&x);
            let mut g = grad(&x);
            // Clip gradient to prevent catastrophic steps from extreme gradients
            let grad_norm = clip_gradient(&mut g, DEFAULT_GRADIENT_CLIP_NORM);

            history.push(OptStep {
                iteration: iter,
                x: x.clone(),
                fx,
                grad_norm: Some(grad_norm),
            });

            if grad_norm < tol {
                let encoding = Self::encode_point(&x, fx);
                return OptResult {
                    x,
                    fx,
                    iterations: iter + 1,
                    converged: true,
                    method: OptMethod::GradientDescent,
                    history,
                    phi: Self::compute_phi(iter + 1, grad_norm),
                    encoding,
                };
            }

            // Backtracking Armijo line search for step size (gradient direction only)
            let descent: f64 = g.iter().map(|gi| gi * gi).sum();
            let mut step = lr;
            for _ in 0..ARMIJO_MAX_BACKTRACKS {
                let x_trial: Vec<f64> = x
                    .iter()
                    .zip(g.iter())
                    .map(|(&xi, &gi)| xi - step * gi)
                    .collect();
                let f_trial = f(&x_trial);
                if f_trial <= fx - ARMIJO_C1 * step * descent {
                    break;
                }
                step *= ARMIJO_SHRINK;
            }

            // Update with momentum
            for i in 0..n {
                velocity[i] = momentum * velocity[i] - step * g[i];
                x[i] += velocity[i];
            }
        }

        let fx = f(&x);
        OptResult {
            x,
            fx,
            iterations: MAX_ITERATIONS,
            converged: false,
            method: OptMethod::GradientDescent,
            history,
            phi: 0.05,
            encoding: Self::encode_point(x0, fx),
        }
    }

    /// Nelder-Mead simplex (derivative-free).
    ///
    /// `f`: objective function
    /// `x0`: initial point
    /// `step_size`: initial simplex size
    pub fn nelder_mead<F>(f: &F, x0: &[f64], step_size: f64, tol: f64) -> OptResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let tol = if tol <= 0.0 { DEFAULT_TOL } else { tol };
        let n = x0.len();
        let step = if step_size <= 0.0 { 1.0 } else { step_size };

        // Nelder-Mead coefficients
        let alpha = 1.0; // reflection
        let gamma = 2.0; // expansion
        let rho = 0.5; // contraction
        let sigma = 0.5; // shrink

        // Initialize simplex: x0 + step * e_i
        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        simplex.push(x0.to_vec());
        for i in 0..n {
            let mut vertex = x0.to_vec();
            vertex[i] += step;
            simplex.push(vertex);
        }

        let mut values: Vec<f64> = simplex.iter().map(|x| f(x)).collect();
        let mut history = Vec::new();

        for iter in 0..MAX_ITERATIONS {
            // Sort by function value
            let mut indices: Vec<usize> = (0..=n).collect();
            indices.sort_by(|&a, &b| {
                values[a]
                    .partial_cmp(&values[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let sorted_simplex: Vec<Vec<f64>> =
                indices.iter().map(|&i| simplex[i].clone()).collect();
            let sorted_values: Vec<f64> = indices.iter().map(|&i| values[i]).collect();
            simplex = sorted_simplex;
            values = sorted_values;

            history.push(OptStep {
                iteration: iter,
                x: simplex[0].clone(),
                fx: values[0],
                grad_norm: None,
            });

            // Check convergence: simplex size
            let size: f64 = (1..=n)
                .map(|i| {
                    simplex[i]
                        .iter()
                        .zip(simplex[0].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0_f64, f64::max);

            if size < tol {
                return OptResult {
                    x: simplex[0].clone(),
                    fx: values[0],
                    iterations: iter + 1,
                    converged: true,
                    method: OptMethod::NelderMead,
                    history,
                    phi: Self::compute_phi(iter + 1, size),
                    encoding: Self::encode_point(&simplex[0], values[0]),
                };
            }

            // Centroid of all points except worst
            let mut centroid = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    centroid[j] += simplex[i][j];
                }
            }
            for j in 0..n {
                centroid[j] /= n as f64;
            }

            // Reflection
            let reflected: Vec<f64> = centroid
                .iter()
                .zip(simplex[n].iter())
                .map(|(c, w)| c + alpha * (c - w))
                .collect();
            let fr = f(&reflected);

            if fr < values[0] {
                // Try expansion
                let expanded: Vec<f64> = centroid
                    .iter()
                    .zip(simplex[n].iter())
                    .map(|(c, w)| c + gamma * (c - w))
                    .collect();
                let fe = f(&expanded);

                if fe < fr {
                    simplex[n] = expanded;
                    values[n] = fe;
                } else {
                    simplex[n] = reflected;
                    values[n] = fr;
                }
            } else if fr < values[n - 1] {
                simplex[n] = reflected;
                values[n] = fr;
            } else {
                // Contraction
                let contracted: Vec<f64> = centroid
                    .iter()
                    .zip(simplex[n].iter())
                    .map(|(c, w)| c + rho * (w - c))
                    .collect();
                let fc = f(&contracted);

                if fc < values[n] {
                    simplex[n] = contracted;
                    values[n] = fc;
                } else {
                    // Shrink
                    for i in 1..=n {
                        for j in 0..n {
                            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                        }
                        values[i] = f(&simplex[i]);
                    }
                }
            }
        }

        OptResult {
            x: simplex[0].clone(),
            fx: values[0],
            iterations: MAX_ITERATIONS,
            converged: false,
            method: OptMethod::NelderMead,
            history,
            phi: 0.05,
            encoding: Self::encode_point(&simplex[0], values[0]),
        }
    }

    /// L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno).
    ///
    /// Quasi-Newton method that approximates the inverse Hessian
    /// using the last `m` gradient differences.
    pub fn lbfgs<F, G>(f: &F, grad: &G, x0: &[f64], tol: f64) -> OptResult
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let tol = if tol <= 0.0 { DEFAULT_TOL } else { tol };
        let n = x0.len();
        let m = LBFGS_MEMORY;

        let mut x = x0.to_vec();
        let mut g = grad(&x);
        clip_gradient(&mut g, DEFAULT_GRADIENT_CLIP_NORM);
        let mut history = Vec::new();

        // Storage for L-BFGS two-loop recursion
        let mut s_history: Vec<Vec<f64>> = Vec::new(); // x_{k+1} - x_k
        let mut y_history: Vec<Vec<f64>> = Vec::new(); // g_{k+1} - g_k
        let mut rho_history: Vec<f64> = Vec::new(); // 1 / (y_k^T s_k)

        for iter in 0..MAX_ITERATIONS {
            let fx = f(&x);
            let grad_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();

            history.push(OptStep {
                iteration: iter,
                x: x.clone(),
                fx,
                grad_norm: Some(grad_norm),
            });

            if grad_norm < tol {
                let encoding = Self::encode_point(&x, fx);
                let phi = Self::compute_phi(iter + 1, grad_norm);
                return OptResult {
                    x,
                    fx,
                    iterations: iter + 1,
                    converged: true,
                    method: OptMethod::LBFGS,
                    history,
                    phi,
                    encoding,
                };
            }

            // L-BFGS two-loop recursion to compute search direction
            let mut q = g.clone();
            let k = s_history.len();
            let mut alpha_vec = vec![0.0; k];

            // First loop (backward)
            for i in (0..k).rev() {
                alpha_vec[i] = rho_history[i]
                    * s_history[i]
                        .iter()
                        .zip(q.iter())
                        .map(|(si, qi)| si * qi)
                        .sum::<f64>();
                for j in 0..n {
                    q[j] -= alpha_vec[i] * y_history[i][j];
                }
            }

            // Scale by H_0 = (s^T y) / (y^T y) * I
            let mut r = if k > 0 {
                let sy: f64 = s_history[k - 1]
                    .iter()
                    .zip(y_history[k - 1].iter())
                    .map(|(s, y)| s * y)
                    .sum();
                let yy: f64 = y_history[k - 1].iter().map(|y| y * y).sum();
                let scale = if yy > 1e-15 { sy / yy } else { 1.0 };
                q.iter().map(|qi| qi * scale).collect::<Vec<_>>()
            } else {
                q.clone()
            };

            // Second loop (forward)
            for i in 0..k {
                let beta = rho_history[i]
                    * y_history[i]
                        .iter()
                        .zip(r.iter())
                        .map(|(yi, ri)| yi * ri)
                        .sum::<f64>();
                for j in 0..n {
                    r[j] += s_history[i][j] * (alpha_vec[i] - beta);
                }
            }

            // Line search (simple backtracking Armijo)
            let mut step = 1.0;
            let descent: f64 = g.iter().zip(r.iter()).map(|(gi, ri)| gi * ri).sum();
            let c1 = 1e-4;

            for _ in 0..20 {
                let x_new: Vec<f64> = x
                    .iter()
                    .zip(r.iter())
                    .map(|(xi, ri)| xi - step * ri)
                    .collect();
                if f(&x_new) <= fx + c1 * step * (-descent) {
                    break;
                }
                step *= 0.5;
            }

            // Update
            let x_new: Vec<f64> = x
                .iter()
                .zip(r.iter())
                .map(|(xi, ri)| xi - step * ri)
                .collect();
            let mut g_new = grad(&x_new);
            clip_gradient(&mut g_new, DEFAULT_GRADIENT_CLIP_NORM);

            let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
            let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();

            // Scale curvature check relative to step and gradient magnitudes
            // to avoid rejecting valid updates on large-scale problems
            let s_norm = s.iter().map(|v| v * v).sum::<f64>().sqrt();
            let y_norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            if sy > 1e-10 * s_norm * y_norm {
                if s_history.len() >= m {
                    s_history.remove(0);
                    y_history.remove(0);
                    rho_history.remove(0);
                }
                s_history.push(s);
                y_history.push(y);
                rho_history.push(1.0 / sy);
            }

            x = x_new;
            g = g_new;
        }

        let fx = f(&x);
        OptResult {
            x,
            fx,
            iterations: MAX_ITERATIONS,
            converged: false,
            method: OptMethod::LBFGS,
            history,
            phi: 0.05,
            encoding: Self::encode_point(x0, fx),
        }
    }

    /// Constrained optimization via penalty method.
    ///
    /// Minimizes f(x) subject to g_i(x) <= 0.
    /// Converts to: minimize f(x) + penalty * Σ max(0, g_i(x))²
    pub fn penalty_method<F, G, C>(
        f: &F,
        grad: &G,
        constraints: &[C],
        x0: &[f64],
        tol: f64,
    ) -> OptResult
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
        C: Fn(&[f64]) -> f64,
    {
        let mut x = x0.to_vec();
        let mut penalty = 1.0;
        let penalty_growth = 10.0;
        let n = x0.len();

        for _ in 0..10 {
            let penalized_f = |xp: &[f64]| -> f64 {
                let mut val = f(xp);
                for c in constraints {
                    let cv = c(xp);
                    if cv > 0.0 {
                        val += penalty * cv * cv;
                    }
                }
                val
            };

            let penalized_grad = |xp: &[f64]| -> Vec<f64> {
                let mut g = grad(xp);
                // Numerical gradient of penalty terms
                let eps = 1e-7;
                for i in 0..n {
                    let mut xp_plus = xp.to_vec();
                    let mut xp_minus = xp.to_vec();
                    xp_plus[i] += eps;
                    xp_minus[i] -= eps;
                    let pen_plus: f64 = constraints
                        .iter()
                        .map(|c| {
                            let cv = c(&xp_plus);
                            if cv > 0.0 {
                                penalty * cv * cv
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    let pen_minus: f64 = constraints
                        .iter()
                        .map(|c| {
                            let cv = c(&xp_minus);
                            if cv > 0.0 {
                                penalty * cv * cv
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    g[i] += (pen_plus - pen_minus) / (2.0 * eps);
                }
                g
            };

            let result = Self::lbfgs(&penalized_f, &penalized_grad, &x, tol);
            x = result.x;

            // Check if constraints are satisfied
            let max_violation: f64 = constraints
                .iter()
                .map(|c| c(&x).max(0.0))
                .fold(0.0_f64, f64::max);

            if max_violation < tol {
                let fx = f(&x);
                let encoding = Self::encode_point(&x, fx);
                return OptResult {
                    x,
                    fx,
                    iterations: result.iterations,
                    converged: true,
                    method: OptMethod::LBFGS,
                    history: result.history,
                    phi: Self::compute_phi(result.iterations, max_violation),
                    encoding,
                };
            }

            penalty *= penalty_growth;
        }

        let fx = f(&x);
        let encoding = Self::encode_point(&x, fx);
        OptResult {
            x,
            fx,
            iterations: MAX_ITERATIONS,
            converged: false,
            method: OptMethod::LBFGS,
            history: Vec::new(),
            phi: 0.05,
            encoding,
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    fn compute_phi(iterations: usize, residual: f64) -> f64 {
        let speed = 1.0 / (1.0 + iterations as f64 / 50.0);
        let accuracy = if residual < 1e-15 {
            1.0
        } else {
            1.0 / (1.0 + residual.abs().log10().abs() / 10.0)
        };
        (speed + accuracy) / 2.0
    }

    fn encode_point(x: &[f64], fx: f64) -> BinaryHV {
        let opt_prim = BinaryHV::random(seed_from_name("OPTIMUM"));
        let val_hv = BinaryHV::random(seed_from_name(&format!("OPT_FX_{}", fx.to_bits())));
        let mut x_hv = BinaryHV::random(seed_from_name(&format!("OPT_DIM_{}", x.len())));
        for (i, xi) in x.iter().enumerate() {
            let comp = BinaryHV::random(seed_from_name(&format!("OPT_X{}_{}", i, xi.to_bits())));
            x_hv = x_hv.bind(&comp);
        }
        opt_prim.bind(&val_hv).bind(&x_hv)
    }
}

// ─── Wolfe Line Search ───────────────────────────────────────────────────────

/// Wolfe conditions line search for L-BFGS and trust-region methods.
///
/// Finds step length α satisfying the strong Wolfe conditions:
/// 1. Sufficient decrease (Armijo): f(x + α*d) ≤ f(x) + c1*α*∇f(x)ᵀd
/// 2. Curvature: |∇f(x + α*d)ᵀd| ≤ c2 * |∇f(x)ᵀd|
///
/// Algorithm: bracketing + zoom (Nocedal & Wright §3.5, 2006).
pub fn wolfe_line_search<F, G>(
    f: &F,
    grad: &G,
    x: &[f64],
    direction: &[f64],
    f0: f64,
    grad0: &[f64],
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> f64
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let derphi0: f64 = grad0.iter().zip(direction).map(|(g, d)| g * d).sum();
    if derphi0 >= 0.0 {
        return 1e-4; // direction is not a descent direction
    }

    let mut alpha_prev = 0.0;
    let mut alpha = 1.0;
    let alpha_max = 50.0;
    let mut f_prev = f0;

    let x_alpha = |a: f64| -> Vec<f64> { (0..n).map(|i| x[i] + a * direction[i]).collect() };
    let derphi = |a: f64| -> f64 {
        let xn = x_alpha(a);
        let g = grad(&xn);
        g.iter().zip(direction).map(|(gi, di)| gi * di).sum()
    };

    for _iter in 0..max_iter {
        let xa = x_alpha(alpha);
        let fa = f(&xa);

        if fa > f0 + c1 * alpha * derphi0 || (_iter > 0 && fa >= f_prev) {
            return wolfe_zoom(
                f, &derphi, x, direction, alpha_prev, alpha, f0, derphi0, c1, c2, &x_alpha,
            );
        }

        let da = derphi(alpha);
        if da.abs() <= -c2 * derphi0 {
            return alpha; // Strong Wolfe satisfied
        }

        if da >= 0.0 {
            return wolfe_zoom(
                f, &derphi, x, direction, alpha, alpha_prev, f0, derphi0, c1, c2, &x_alpha,
            );
        }

        alpha_prev = alpha;
        f_prev = fa;
        alpha = (alpha * 2.0).min(alpha_max);
    }
    alpha
}

fn wolfe_zoom<F, D, XA>(
    f: &F,
    derphi: &D,
    _x: &[f64],
    _direction: &[f64],
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    f0: f64,
    derphi0: f64,
    c1: f64,
    c2: f64,
    x_alpha: &XA,
) -> f64
where
    F: Fn(&[f64]) -> f64,
    D: Fn(f64) -> f64,
    XA: Fn(f64) -> Vec<f64>,
{
    let f_lo = f(&x_alpha(alpha_lo));
    for _ in 0..20 {
        let alpha_j = (alpha_lo + alpha_hi) / 2.0; // bisection
        let fj = f(&x_alpha(alpha_j));
        if fj > f0 + c1 * alpha_j * derphi0 || fj >= f_lo {
            alpha_hi = alpha_j;
        } else {
            let dj = derphi(alpha_j);
            if dj.abs() <= -c2 * derphi0 {
                return alpha_j;
            }
            if dj * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
            }
            alpha_lo = alpha_j;
        }
    }
    alpha_lo
}

// ─── Levenberg-Marquardt ──────────────────────────────────────────────────────

/// Result of a Levenberg-Marquardt nonlinear least squares solve.
#[derive(Debug, Clone)]
pub struct LMResult {
    /// Solution parameters.
    pub params: Vec<f64>,
    /// Residual vector at solution: r_i = y_i - f(x_i; params).
    pub residuals: Vec<f64>,
    /// Sum of squared residuals.
    pub sse: f64,
    /// Number of iterations taken.
    pub iterations: usize,
    /// Whether convergence criteria were met.
    pub converged: bool,
    /// Final damping parameter λ.
    pub lambda_final: f64,
}

/// Levenberg-Marquardt algorithm for nonlinear least squares.
///
/// Minimizes ‖r(p)‖² where r: ℝⁿ → ℝᵐ is a vector of residuals.
/// Blends Gauss-Newton (fast near solution) with gradient descent (robust far away)
/// via a damping parameter λ:
///   (JᵀJ + λI) Δp = -Jᵀr
/// λ is adapted based on the ratio of actual to predicted reduction.
pub struct LevenbergMarquardt;

impl LevenbergMarquardt {
    /// Fit a model by minimizing nonlinear least squares.
    ///
    /// # Arguments
    /// * `residual_fn` — function mapping params → residual vector
    /// * `p0` — initial parameter guess
    /// * `tol` — convergence threshold on gradient norm
    /// * `max_iter` — maximum iterations
    pub fn fit<R>(residual_fn: R, p0: &[f64], tol: f64, max_iter: usize) -> LMResult
    where
        R: Fn(&[f64]) -> Vec<f64>,
    {
        let n = p0.len();
        let mut p = p0.to_vec();
        let mut lambda = 1e-3;
        let eps = 1e-8; // finite difference step for Jacobian

        let r = residual_fn(&p);
        let m = r.len();
        let mut sse = r.iter().map(|ri| ri * ri).sum::<f64>();
        let mut converged = false;

        for iter in 0..max_iter {
            let r = residual_fn(&p);
            sse = r.iter().map(|ri| ri * ri).sum::<f64>();

            // Compute Jacobian via finite differences: J[i][j] = ∂r_i/∂p_j
            let mut jac = vec![vec![0.0f64; n]; m];
            for j in 0..n {
                let mut p_plus = p.clone();
                p_plus[j] += eps;
                let r_plus = residual_fn(&p_plus);
                for i in 0..m {
                    jac[i][j] = (r_plus[i] - r[i]) / eps;
                }
            }

            // JᵀJ (n×n) and Jᵀr (n×1)
            let mut jtj = vec![vec![0.0f64; n]; n];
            let mut jtr = vec![0.0f64; n];
            for i in 0..m {
                for j in 0..n {
                    jtr[j] += jac[i][j] * r[i];
                    for k in 0..n {
                        jtj[j][k] += jac[i][j] * jac[i][k];
                    }
                }
            }

            // Check gradient norm for convergence
            let grad_norm: f64 = jtr.iter().map(|v| v * v).sum::<f64>().sqrt();
            if grad_norm < tol {
                converged = true;
                break;
            }

            // Solve (JᵀJ + λI)Δp = -Jᵀr via Gaussian elimination
            let mut a = jtj.clone();
            for j in 0..n {
                a[j][j] += lambda;
            }
            let neg_jtr: Vec<f64> = jtr.iter().map(|v| -v).collect();
            let delta = match Self::solve_linear(&a, &neg_jtr) {
                Some(d) => d,
                None => break,
            };

            let p_new: Vec<f64> = (0..n).map(|j| p[j] + delta[j]).collect();
            let r_new = residual_fn(&p_new);
            let sse_new = r_new.iter().map(|ri| ri * ri).sum::<f64>();

            // Predicted reduction (from Gauss-Newton model)
            let predicted: f64 = delta
                .iter()
                .zip(&jtr)
                .map(|(d, g)| d * (lambda * d - g))
                .sum::<f64>();
            let rho = if predicted.abs() > 1e-15 {
                (sse - sse_new) / predicted
            } else {
                0.0
            };

            if rho > 0.0 {
                // Accept step
                p = p_new;
                sse = sse_new;
                // Reduce λ (more Gauss-Newton)
                lambda *= (1.0_f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
                lambda = lambda.max(1e-15);
            } else {
                // Reject step, increase λ (more gradient descent)
                lambda *= 10.0;
                lambda = lambda.min(1e12);
            }

            let _ = iter; // suppress warning
        }

        let r_final = residual_fn(&p);
        sse = r_final.iter().map(|ri| ri * ri).sum::<f64>();

        LMResult {
            params: p,
            residuals: r_final,
            sse,
            iterations: max_iter,
            converged,
            lambda_final: lambda,
        }
    }

    /// Gaussian elimination solver for Ax = b (in-place, n ≤ 100).
    fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let n = b.len();
        let mut aug: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = a[i].clone();
                row.push(b[i]);
                row
            })
            .collect();

        for col in 0..n {
            // Partial pivoting
            let max_row = (col..n)
                .max_by(|&r1, &r2| aug[r1][col].abs().partial_cmp(&aug[r2][col].abs()).unwrap())?;
            aug.swap(col, max_row);
            if aug[col][col].abs() < 1e-14 {
                return None;
            }
            let pivot = aug[col][col];
            for k in col..=n {
                aug[col][k] /= pivot;
            }
            for row in 0..n {
                if row != col {
                    let factor = aug[row][col];
                    for k in col..=n {
                        aug[row][k] -= factor * aug[col][k];
                    }
                }
            }
        }
        Some((0..n).map(|i| aug[i][n]).collect())
    }
}

/// Augmented Lagrangian method for constrained optimization.
///
/// Minimizes f(x) subject to equality constraints h_i(x) = 0.
/// Solves: min_x L_ρ(x, λ) = f(x) + Σ λᵢhᵢ(x) + (ρ/2)Σ hᵢ(x)²
/// Multipliers updated: λᵢ ← λᵢ + ρ * hᵢ(x*)
pub fn augmented_lagrangian<F, G, H>(
    f: F,
    grad_f: G,
    constraints: &[H],
    x0: &[f64],
    tol: f64,
    max_outer: usize,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
    H: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let m = constraints.len();
    let mut x = x0.to_vec();
    let mut lambda = vec![0.0f64; m];
    let mut rho = 1.0f64;

    for outer in 0..max_outer {
        // Minimize augmented Lagrangian via gradient descent
        let al_grad = |xk: &[f64]| -> Vec<f64> {
            let gf = grad_f(xk);
            let mut g = gf;
            for i in 0..m {
                let hi = constraints[i](xk);
                // ∂L/∂x += (λᵢ + ρ*hᵢ) * ∂hᵢ/∂x (finite diff for constraint gradient)
                let eps = 1e-7;
                for j in 0..n {
                    let mut xp = xk.to_vec();
                    xp[j] += eps;
                    let dhi = (constraints[i](&xp) - hi) / eps;
                    g[j] += (lambda[i] + rho * hi) * dhi;
                }
            }
            g
        };
        let al_f = |xk: &[f64]| -> f64 {
            let mut val = f(xk);
            for i in 0..m {
                let hi = constraints[i](xk);
                val += lambda[i] * hi + 0.5 * rho * hi * hi;
            }
            val
        };

        // Gradient descent on augmented Lagrangian
        let mut lr = 0.01;
        for _step in 0..500 {
            let g = al_grad(&x);
            let gn: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            if gn < tol {
                break;
            }
            for j in 0..n {
                x[j] -= lr * g[j] / gn.max(1e-12);
            }
            lr *= 0.999;
        }

        // Update multipliers
        let constraint_viol: f64 = (0..m)
            .map(|i| constraints[i](&x).powi(2))
            .sum::<f64>()
            .sqrt();
        for i in 0..m {
            lambda[i] += rho * constraints[i](&x);
        }
        rho = (rho * 1.5).min(1e6);

        if constraint_viol < tol {
            break;
        }
        let _ = outer;
    }

    let fx = f(&x);
    let gf = grad_f(&x);
    let grad_norm: f64 = gf.iter().map(|v| v * v).sum::<f64>().sqrt();
    let converged = grad_norm < tol * 10.0;

    OptResult {
        x: x.clone(),
        fx,
        history: vec![],
        iterations: max_outer,
        converged,
        method: OptMethod::GradientDescent,
        phi: 0.0,
        encoding: OptimizationEngine::encode_point(&x, fx),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-4;

    // ── Gradient Descent ─────────────────────────────────────────────────

    #[test]
    fn test_gd_quadratic() {
        // f(x) = x², minimum at x=0
        let result = OptimizationEngine::gradient_descent(
            &|x: &[f64]| x[0] * x[0],
            &|x: &[f64]| vec![2.0 * x[0]],
            &[5.0],
            0.1,
            0.0,
            1e-10,
        );
        assert!(result.converged);
        assert!(result.x[0].abs() < TOL);
    }

    #[test]
    fn test_gd_with_momentum() {
        // f(x,y) = x² + y², minimum at (0,0)
        let result = OptimizationEngine::gradient_descent(
            &|x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &|x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]],
            &[3.0, 4.0],
            0.1,
            0.9,
            1e-10,
        );
        assert!(result.converged);
        assert!(result.x[0].abs() < TOL);
        assert!(result.x[1].abs() < TOL);
    }

    #[test]
    fn test_gd_rosenbrock() {
        // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
        // Minimum at (1, 1). Hard for gradient descent.
        let result = OptimizationEngine::gradient_descent(
            &|x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &|x: &[f64]| {
                vec![
                    -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
                    200.0 * (x[1] - x[0] * x[0]),
                ]
            },
            &[0.0, 0.0],
            0.001,
            0.9,
            1e-8,
        );
        // GD may not converge on Rosenbrock with these params, but should make progress
        assert!(result.fx < 10.0, "Should reduce from initial f=1");
    }

    // ── Nelder-Mead ──────────────────────────────────────────────────────

    #[test]
    fn test_nm_quadratic() {
        let result = OptimizationEngine::nelder_mead(
            &|x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &[3.0, 4.0],
            1.0,
            1e-10,
        );
        assert!(result.converged);
        assert!(result.x[0].abs() < TOL);
        assert!(result.x[1].abs() < TOL);
    }

    #[test]
    fn test_nm_sphere() {
        // Sphere function in 3D
        let result = OptimizationEngine::nelder_mead(
            &|x: &[f64]| x.iter().map(|xi| xi * xi).sum(),
            &[1.0, 2.0, 3.0],
            1.0,
            1e-8,
        );
        assert!(result.converged);
        for xi in &result.x {
            assert!(xi.abs() < TOL, "x = {:?}", result.x);
        }
    }

    #[test]
    fn test_nm_rosenbrock() {
        let result = OptimizationEngine::nelder_mead(
            &|x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &[0.0, 0.0],
            1.0,
            1e-10,
        );
        // Nelder-Mead should get close to (1,1)
        assert!((result.x[0] - 1.0).abs() < 0.01, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 0.01, "x[1] = {}", result.x[1]);
    }

    // ── L-BFGS ───────────────────────────────────────────────────────────

    #[test]
    fn test_lbfgs_quadratic() {
        let result = OptimizationEngine::lbfgs(
            &|x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &|x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]],
            &[3.0, 4.0],
            1e-10,
        );
        assert!(result.converged);
        assert!(result.x[0].abs() < TOL);
        assert!(result.x[1].abs() < TOL);
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
        let result = OptimizationEngine::lbfgs(
            &|x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &|x: &[f64]| {
                vec![
                    -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
                    200.0 * (x[1] - x[0] * x[0]),
                ]
            },
            &[0.0, 0.0],
            1e-10,
        );
        assert!(result.converged, "L-BFGS should converge on Rosenbrock");
        assert!((result.x[0] - 1.0).abs() < 0.01);
        assert!((result.x[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lbfgs_high_dim() {
        // 10D sphere function
        let n = 10;
        let x0: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let result = OptimizationEngine::lbfgs(
            &|x: &[f64]| x.iter().map(|xi| xi * xi).sum(),
            &|x: &[f64]| x.iter().map(|xi| 2.0 * xi).collect(),
            &x0,
            1e-10,
        );
        assert!(result.converged);
        for xi in &result.x {
            assert!(xi.abs() < TOL);
        }
    }

    // ── Penalty Method ───────────────────────────────────────────────────

    #[test]
    fn test_constrained_optimization() {
        // Minimize x² + y² subject to x + y >= 1 (i.e. -(x+y-1) <= 0)
        // Solution: x = y = 0.5
        let result = OptimizationEngine::penalty_method(
            &|x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &|x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]],
            &[|x: &[f64]| -(x[0] + x[1] - 1.0)], // constraint: x+y >= 1
            &[0.0, 0.0],
            1e-6,
        );
        assert!((result.x[0] - 0.5).abs() < 0.05, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 0.5).abs() < 0.05, "x[1] = {}", result.x[1]);
    }

    // ── Rastrigin (multimodal) ───────────────────────────────────────────

    #[test]
    fn test_nm_rastrigin() {
        // Rastrigin: many local minima, global at (0,0)
        // Nelder-Mead from near origin should find it
        let result = OptimizationEngine::nelder_mead(
            &|x: &[f64]| {
                let n = x.len() as f64;
                10.0 * n
                    + x.iter()
                        .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                        .sum::<f64>()
            },
            &[0.1, 0.1],
            0.5,
            1e-10,
        );
        assert!(
            result.fx < 1.0,
            "Should find near-global minimum, got f={}",
            result.fx
        );
    }

    // ── GD vs Nelder-Mead comparison ────────────────────────────────────

    #[test]
    fn test_gd_vs_nm_comparison() {
        // Both should find the minimum of a quadratic bowl
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let g = |x: &[f64]| vec![2.0 * x[0], 4.0 * x[1]];

        let gd = OptimizationEngine::gradient_descent(&f, &g, &[3.0, 4.0], 0.1, 0.0, 1e-10);
        let nm = OptimizationEngine::nelder_mead(&f, &[3.0, 4.0], 1.0, 1e-10);

        assert!(gd.converged, "GD should converge");
        assert!(nm.converged, "NM should converge");
        assert!(gd.fx < TOL, "GD minimum should be near 0, got {}", gd.fx);
        assert!(nm.fx < TOL, "NM minimum should be near 0, got {}", nm.fx);
    }

    // ── L-BFGS convergence on non-trivial function ──────────────────────

    #[test]
    fn test_lbfgs_convergence_rate() {
        // L-BFGS should converge faster than GD on Rosenbrock
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let g = |x: &[f64]| {
            vec![
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ]
        };

        let lbfgs_result = OptimizationEngine::lbfgs(&f, &g, &[0.0, 0.0], 1e-10);
        assert!(
            lbfgs_result.converged,
            "L-BFGS should converge on Rosenbrock"
        );
        assert!(
            lbfgs_result.iterations < 200,
            "L-BFGS should converge quickly, took {} iterations",
            lbfgs_result.iterations
        );
    }

    // ── Box constraints ─────────────────────────────────────────────────

    #[test]
    fn test_box_constraints_projection() {
        let bounds = BoxConstraints::new(vec![-1.0, -1.0], vec![1.0, 1.0]);
        let projected = bounds.project(&[5.0, -3.0]);
        assert!((projected[0] - 1.0).abs() < 1e-15);
        assert!((projected[1] - (-1.0)).abs() < 1e-15);

        let interior = bounds.project(&[0.5, -0.5]);
        assert!((interior[0] - 0.5).abs() < 1e-15);
        assert!((interior[1] - (-0.5)).abs() < 1e-15);
    }

    #[test]
    fn test_box_constrained_optimization() {
        // Minimize (x-3)² + (y-3)² subject to 0 <= x,y <= 2
        // Unconstrained min at (3,3), constrained min at (2,2)
        // Box constraints as penalty: x[i] - 2 <= 0 and 0 - x[i] <= 0
        let constraints: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| x[0] - 2.0),
            Box::new(|x: &[f64]| -x[0]),
            Box::new(|x: &[f64]| x[1] - 2.0),
            Box::new(|x: &[f64]| -x[1]),
        ];

        let result = OptimizationEngine::penalty_method(
            &|x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2),
            &|x: &[f64]| vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] - 3.0)],
            &constraints,
            &[1.0, 1.0],
            1e-4,
        );
        assert!(
            (result.x[0] - 2.0).abs() < 0.1,
            "x[0] should be ~2.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.1,
            "x[1] should be ~2.0, got {}",
            result.x[1]
        );
    }

    // ── ObjectiveFunction trait ──────────────────────────────────────────

    #[test]
    fn test_objective_function_trait() {
        struct Sphere;
        impl ObjectiveFunction for Sphere {
            fn eval(&self, x: &[f64]) -> f64 {
                x.iter().map(|xi| xi * xi).sum()
            }
            fn gradient(&self, x: &[f64]) -> Option<Vec<f64>> {
                Some(x.iter().map(|xi| 2.0 * xi).collect())
            }
        }

        let sphere = Sphere;
        let result = minimize_objective(&sphere, &[3.0, 4.0], OptMethod::LBFGS);
        assert!(result.converged, "Trait-based LBFGS should converge");
        assert!(result.fx < TOL, "Should find minimum, got {}", result.fx);
    }

    #[test]
    fn test_objective_function_without_gradient() {
        struct Quadratic;
        impl ObjectiveFunction for Quadratic {
            fn eval(&self, x: &[f64]) -> f64 {
                x[0] * x[0] + x[1] * x[1]
            }
            // No gradient — uses finite differences
        }

        let q = Quadratic;
        let result = minimize_objective(&q, &[3.0, 4.0], OptMethod::LBFGS);
        assert!(result.converged, "Should converge with numerical gradient");
        assert!(result.fx < 0.01, "Should find minimum, got {}", result.fx);
    }

    // ── Minimize dispatcher ─────────────────────────────────────────────

    #[test]
    fn test_minimize_dispatcher_gd() {
        let result = minimize(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &[3.0, 4.0],
            OptMethod::GradientDescent,
        );
        assert!(
            result.fx < 1.0,
            "GD via minimize should make progress, got {}",
            result.fx
        );
    }

    #[test]
    fn test_minimize_dispatcher_nm() {
        let result = minimize(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &[3.0, 4.0],
            OptMethod::NelderMead,
        );
        assert!(result.converged, "NM via minimize should converge");
        assert!(result.fx < TOL);
    }

    #[test]
    fn test_minimize_dispatcher_lbfgs() {
        let result = minimize(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &[3.0, 4.0],
            OptMethod::LBFGS,
        );
        assert!(result.converged, "LBFGS via minimize should converge");
        assert!(result.fx < TOL);
    }

    // ── Numerical gradient ──────────────────────────────────────────────

    #[test]
    fn test_numerical_gradient_accuracy() {
        let f = |x: &[f64]| x[0] * x[0] + 3.0 * x[1] * x[1];
        let grad = numerical_gradient(&f, &[2.0, 3.0]);
        // Analytic: [2*x, 6*y] = [4.0, 18.0]
        assert!(
            (grad[0] - 4.0).abs() < 1e-5,
            "dfdx should be ~4.0, got {}",
            grad[0]
        );
        assert!(
            (grad[1] - 18.0).abs() < 1e-5,
            "dfdy should be ~18.0, got {}",
            grad[1]
        );
    }

    // ── Encoding ─────────────────────────────────────────────────────────

    #[test]
    fn test_encoding_different_optima() {
        let r1 =
            OptimizationEngine::nelder_mead(&|x: &[f64]| (x[0] - 1.0).powi(2), &[5.0], 1.0, 1e-8);
        let r2 =
            OptimizationEngine::nelder_mead(&|x: &[f64]| (x[0] - 10.0).powi(2), &[5.0], 1.0, 1e-8);
        let sim = r1.encoding.similarity(&r2.encoding);
        assert!(
            sim < 0.6,
            "Different optima should have different encodings: {}",
            sim
        );
    }

    // ── Levenberg-Marquardt tests ─────────────────────────────────────────

    #[test]
    fn test_lm_linear_residuals() {
        // Fit y = a*x + b to (0,1),(1,3),(2,5),(3,7) → a=2, b=1
        let xs = [0.0f64, 1.0, 2.0, 3.0];
        let ys = [1.0f64, 3.0, 5.0, 7.0];
        let residual_fn = |p: &[f64]| -> Vec<f64> {
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| y - (p[0] * x + p[1]))
                .collect()
        };
        let result = LevenbergMarquardt::fit(residual_fn, &[0.0, 0.0], 1e-8, 200);
        assert!(
            result.sse < 1e-6,
            "LM should fit linear data exactly, SSE={}",
            result.sse
        );
        assert!(
            (result.params[0] - 2.0).abs() < 0.01,
            "slope should be ~2, got {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 1.0).abs() < 0.01,
            "intercept should be ~1, got {}",
            result.params[1]
        );
    }

    #[test]
    fn test_lm_rosenbrock() {
        // Rosenbrock as nonlinear LS: r1 = 10*(x1 - x0²), r2 = 1 - x0 → min at (1,1)
        let residual_fn = |p: &[f64]| -> Vec<f64> { vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]] };
        let result = LevenbergMarquardt::fit(residual_fn, &[-1.2, 1.0], 1e-8, 500);
        assert!(
            result.sse < 1e-6,
            "LM Rosenbrock SSE should be near 0, got {}",
            result.sse
        );
    }

    #[test]
    fn test_lm_converges_from_bad_init() {
        // y = exp(-a*x): fit to data with bad initial guess
        let xs = [0.0f64, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|&x| (-0.5 * x).exp()).collect();
        let residual_fn = |p: &[f64]| -> Vec<f64> {
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| y - (-p[0] * x).exp())
                .collect()
        };
        let result = LevenbergMarquardt::fit(residual_fn, &[5.0], 1e-8, 300);
        assert!(
            (result.params[0] - 0.5).abs() < 0.01,
            "LM should recover a≈0.5, got {}",
            result.params[0]
        );
    }

    // ── Wolfe line search tests ───────────────────────────────────────────

    #[test]
    fn test_wolfe_sufficient_decrease() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let g = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let x0 = [3.0, 4.0];
        let grad0 = g(&x0);
        let dir = [-grad0[0], -grad0[1]]; // steepest descent
        let alpha = wolfe_line_search(&f, &g, &x0, &dir, f(&x0), &grad0, 1e-4, 0.9, 20);
        // After step, f should decrease
        let x_new = [x0[0] + alpha * dir[0], x0[1] + alpha * dir[1]];
        assert!(f(&x_new) < f(&x0), "Wolfe step should decrease objective");
        assert!(alpha > 0.0, "Wolfe alpha should be positive");
    }

    #[test]
    fn test_wolfe_returns_finite_alpha() {
        let f = |x: &[f64]| (x[0] - 2.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];
        let x0 = [5.0];
        let grad0 = g(&x0);
        let dir = [-grad0[0]];
        let alpha = wolfe_line_search(&f, &g, &x0, &dir, f(&x0), &grad0, 1e-4, 0.9, 20);
        assert!(
            alpha.is_finite() && alpha > 0.0,
            "Wolfe alpha should be finite and positive"
        );
    }

    // ── Augmented Lagrangian tests ────────────────────────────────────────

    #[test]
    fn test_augmented_lagrangian_equality_constraint() {
        // Minimize x² + y² subject to x + y - 1 = 0 → solution: x=y=0.5
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let gf = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let c = |x: &[f64]| x[0] + x[1] - 1.0;
        let result = augmented_lagrangian(&f, &gf, &[c], &[2.0, 0.0], 1e-4, 30);
        assert!(
            (result.x[0] - 0.5).abs() < 0.1 && (result.x[1] - 0.5).abs() < 0.1,
            "AL should find x≈y≈0.5, got ({:.3},{:.3})",
            result.x[0],
            result.x[1]
        );
    }
}
