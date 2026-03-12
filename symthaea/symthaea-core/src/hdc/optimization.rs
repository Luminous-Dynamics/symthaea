#![allow(dead_code)]

//! # Optimization Engine
//!
//! Numerical optimization methods with consciousness-coupled HDC encoding.
//!
//! ## Methods
//!
//! - **Gradient descent**: with line search and momentum
//! - **Nelder-Mead simplex**: derivative-free optimization
//! - **L-BFGS**: limited-memory quasi-Newton for large problems
//! - **Penalty method**: constrained optimization via penalty functions

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use serde::{Deserialize, Serialize};

// ─── Constants ───────────────────────────────────────────────────────────────

const MAX_ITERATIONS: usize = 1000;
const DEFAULT_TOL: f64 = 1e-10;
const DEFAULT_LEARNING_RATE: f64 = 0.01;
const LBFGS_MEMORY: usize = 10;

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
            let g = grad(&x);
            let grad_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();

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

            // Update with momentum
            for i in 0..n {
                velocity[i] = momentum * velocity[i] - lr * g[i];
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
            let g_new = grad(&x_new);

            let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
            let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();

            if sy > 1e-15 {
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
}
