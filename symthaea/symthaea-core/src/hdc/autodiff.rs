// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Reverse-Mode Automatic Differentiation
//!
//! Computes exact gradients via a computation graph (Wengert tape).
//! Forward pass records operations; reverse pass propagates adjoints.
//!
//! ## Why AD over finite differences?
//!
//! Finite differences: ∂f/∂x_i ≈ (f(x+εe_i) - f(x)) / ε
//! - O(n) function evaluations for n parameters
//! - Truncation error from ε choice
//! - Cancellation error for small ε
//!
//! Reverse-mode AD:
//! - ONE reverse pass gives ALL ∂f/∂x_i simultaneously
//! - Machine-precision gradients (no truncation)
//! - Cost: ~2-4× the forward pass (independent of n)
//!
//! This makes L-BFGS optimization practical for expressions with 100+ constants.

use std::cell::RefCell;

/// A node in the computation graph.
#[derive(Debug, Clone)]
struct TapeNode {
    /// Index of this node in the tape
    index: usize,
    /// Partial derivatives w.r.t. parents (parent_index, derivative)
    parents: Vec<(usize, f64)>,
    /// Forward-pass value
    value: f64,
}

/// The Wengert tape: records all operations during forward evaluation.
/// Thread-local to avoid synchronization overhead.
#[derive(Debug)]
pub struct Tape {
    nodes: Vec<TapeNode>,
}

thread_local! {
    static TAPE: RefCell<Option<Tape>> = RefCell::new(None);
}

impl Tape {
    fn new() -> Self {
        Tape { nodes: Vec::with_capacity(64) }
    }

    fn push_node(&mut self, value: f64, parents: Vec<(usize, f64)>) -> usize {
        let index = self.nodes.len();
        self.nodes.push(TapeNode { index, parents, value });
        index
    }
}

/// A differentiable value that records operations on the tape.
#[derive(Debug, Clone, Copy)]
pub struct Var {
    /// Index into the current tape
    index: usize,
    /// Forward-pass value
    pub value: f64,
}

impl Var {
    /// Create a new input variable (leaf node on the tape).
    pub fn new(value: f64) -> Self {
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized — call ad_begin() first");
            let index = tape.push_node(value, vec![]);
            Var { index, value }
        })
    }

    /// Create a constant (no gradient).
    pub fn constant(value: f64) -> Self {
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![]); // no parents = no gradient
            Var { index, value }
        })
    }

    // ── Elementary operations ─────────────────────────────────────────

    pub fn add(self, other: Var) -> Var {
        let value = self.value + other.value;
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![(self.index, 1.0), (other.index, 1.0)]);
            Var { index, value }
        })
    }

    pub fn sub(self, other: Var) -> Var {
        let value = self.value - other.value;
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![(self.index, 1.0), (other.index, -1.0)]);
            Var { index, value }
        })
    }

    pub fn mul(self, other: Var) -> Var {
        let value = self.value * other.value;
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![
                (self.index, other.value),  // ∂(a��b)/∂a = b
                (other.index, self.value),  // ∂(a·b)/∂b = a
            ]);
            Var { index, value }
        })
    }

    pub fn div(self, other: Var) -> Var {
        let value = if other.value.abs() > 1e-15 { self.value / other.value } else { f64::NAN };
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![
                (self.index, 1.0 / other.value),          // ∂(a/b)/∂a = 1/b
                (other.index, -self.value / (other.value * other.value)), // ∂(a/b)/∂b = -a/b²
            ]);
            Var { index, value }
        })
    }

    pub fn powf(self, exp: Var) -> Var {
        let value = self.value.powf(exp.value);
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![
                (self.index, exp.value * self.value.powf(exp.value - 1.0)), // ∂(a^b)/∂a = b·a^(b-1)
                (exp.index, value * self.value.ln()),  // ∂(a^b)/∂b = a^b·ln(a)
            ]);
            Var { index, value }
        })
    }

    // ── Transcendental functions ──────────────────────────────────────

    pub fn sin(self) -> Var {
        let value = self.value.sin();
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![(self.index, self.value.cos())]);
            Var { index, value }
        })
    }

    pub fn cos(self) -> Var {
        let value = self.value.cos();
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![(self.index, -self.value.sin())]);
            Var { index, value }
        })
    }

    pub fn exp(self) -> Var {
        let value = self.value.exp();
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![(self.index, value)]); // d/dx exp(x) = exp(x)
            Var { index, value }
        })
    }

    pub fn ln(self) -> Var {
        let value = self.value.ln();
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![(self.index, 1.0 / self.value)]);
            Var { index, value }
        })
    }

    pub fn sqrt(self) -> Var {
        let value = self.value.sqrt();
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![(self.index, 0.5 / value)]);
            Var { index, value }
        })
    }

    pub fn abs(self) -> Var {
        let value = self.value.abs();
        let grad = if self.value >= 0.0 { 1.0 } else { -1.0 };
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![(self.index, grad)]);
            Var { index, value }
        })
    }

    pub fn neg(self) -> Var {
        let value = -self.value;
        TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let tape = t.as_mut().expect("AD tape not initialized");
            let index = tape.push_node(value, vec![(self.index, -1.0)]);
            Var { index, value }
        })
    }
}

/// Initialize the AD tape for a computation.
/// Must be called before creating any Var values.
pub fn ad_begin() {
    TAPE.with(|tape| {
        *tape.borrow_mut() = Some(Tape::new());
    });
}

/// Compute gradients of `output` w.r.t. all input variables via reverse-mode AD.
///
/// Returns a vector of adjoints indexed by Var.index.
/// The gradient of `output` w.r.t. variable `v` is `adjoints[v.index]`.
pub fn ad_gradient(output: Var) -> Vec<f64> {
    TAPE.with(|tape| {
        let t = tape.borrow();
        let tape = t.as_ref().expect("AD tape not initialized");
        let n = tape.nodes.len();
        let mut adjoints = vec![0.0; n];
        adjoints[output.index] = 1.0; // seed: ∂output/∂output = 1

        // Reverse sweep: propagate adjoints from output to inputs
        for i in (0..n).rev() {
            let adj = adjoints[i];
            if adj == 0.0 { continue; }
            for &(parent_idx, local_grad) in &tape.nodes[i].parents {
                adjoints[parent_idx] += adj * local_grad;
            }
        }

        adjoints
    })
}

/// Clean up the AD tape after computation.
pub fn ad_end() {
    TAPE.with(|tape| {
        *tape.borrow_mut() = None;
    });
}

/// Compute the gradient of a scalar function f: R^n → R at point x.
///
/// This is the main entry point for optimization:
/// ```ignore
/// let grad = compute_gradient(|vars| {
///     let x = vars[0];
///     let y = vars[1];
///     x.mul(x).add(y.mul(y))  // f(x,y) = x² + y²
/// }, &[3.0, 4.0]);
/// // grad = [6.0, 8.0] (exact, machine precision)
/// ```
pub fn compute_gradient<F>(f: F, point: &[f64]) -> Vec<f64>
where
    F: Fn(&[Var]) -> Var,
{
    ad_begin();

    let vars: Vec<Var> = point.iter().map(|&v| Var::new(v)).collect();
    let output = f(&vars);
    let adjoints = ad_gradient(output);

    let grad: Vec<f64> = vars.iter().map(|v| {
        let adj = adjoints[v.index];
        if adj.is_finite() { adj } else { 0.0 }
    }).collect();

    ad_end();
    grad
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ad_linear() {
        // f(x) = 3x + 2, df/dx = 3
        let grad = compute_gradient(|vars| {
            let x = vars[0];
            let three = Var::constant(3.0);
            let two = Var::constant(2.0);
            x.mul(three).add(two)
        }, &[5.0]);

        assert!((grad[0] - 3.0).abs() < 1e-12, "df/dx of 3x+2 should be 3, got {}", grad[0]);
    }

    #[test]
    fn test_ad_quadratic() {
        // f(x) = x², df/dx = 2x
        let grad = compute_gradient(|vars| {
            let x = vars[0];
            x.mul(x)
        }, &[4.0]);

        assert!((grad[0] - 8.0).abs() < 1e-12, "df/dx of x² at x=4 should be 8, got {}", grad[0]);
    }

    #[test]
    fn test_ad_multivariate() {
        // f(x,y) = x² + y², ∇f = (2x, 2y)
        let grad = compute_gradient(|vars| {
            let x = vars[0];
            let y = vars[1];
            x.mul(x).add(y.mul(y))
        }, &[3.0, 4.0]);

        assert!((grad[0] - 6.0).abs() < 1e-12, "∂f/∂x at (3,4) should be 6, got {}", grad[0]);
        assert!((grad[1] - 8.0).abs() < 1e-12, "∂f/∂y at (3,4) should be 8, got {}", grad[1]);
    }

    #[test]
    fn test_ad_product_rule() {
        // f(x,y) = x*y, ∂f/∂x = y, ∂f/∂y = x
        let grad = compute_gradient(|vars| {
            vars[0].mul(vars[1])
        }, &[3.0, 5.0]);

        assert!((grad[0] - 5.0).abs() < 1e-12, "∂(xy)/∂x should be y=5, got {}", grad[0]);
        assert!((grad[1] - 3.0).abs() < 1e-12, "∂(xy)/∂y should be x=3, got {}", grad[1]);
    }

    #[test]
    fn test_ad_chain_rule() {
        // f(x) = sin(x²), df/dx = 2x·cos(x²)
        let x0 = 1.5;
        let grad = compute_gradient(|vars| {
            let x = vars[0];
            x.mul(x).sin()
        }, &[x0]);

        let expected = 2.0 * x0 * (x0 * x0).cos();
        assert!((grad[0] - expected).abs() < 1e-10,
            "d/dx sin(x²) at x={} should be {}, got {}", x0, expected, grad[0]);
    }

    #[test]
    fn test_ad_exp_composition() {
        // f(x) = exp(2x), df/dx = 2·exp(2x)
        let x0 = 1.0;
        let grad = compute_gradient(|vars| {
            let x = vars[0];
            let two = Var::constant(2.0);
            x.mul(two).exp()
        }, &[x0]);

        let expected = 2.0 * (2.0 * x0).exp();
        assert!((grad[0] - expected).abs() < 1e-10,
            "d/dx exp(2x) at x={} should be {}, got {}", x0, expected, grad[0]);
    }

    #[test]
    fn test_ad_rosenbrock_gradient() {
        // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
        // ∂f/∂x = -2(1-x) + 100·2(y-x²)·(-2x) = -2 + 2x - 400x(y-x²)
        // ∂f/∂y = 200(y-x²)
        let (x0, y0) = (1.0, 1.0); // at the minimum, gradient should be (0, 0)
        let grad = compute_gradient(|vars| {
            let x = vars[0];
            let y = vars[1];
            let one = Var::constant(1.0);
            let hundred = Var::constant(100.0);
            let t1 = one.sub(x).mul(one.sub(x)); // (1-x)²
            let t2 = y.sub(x.mul(x)).mul(y.sub(x.mul(x))).mul(hundred); // 100(y-x²)²
            t1.add(t2)
        }, &[x0, y0]);

        assert!(grad[0].abs() < 1e-10, "∂f/∂x at (1,1) should be 0, got {}", grad[0]);
        assert!(grad[1].abs() < 1e-10, "∂f/∂y at (1,1) should be 0, got {}", grad[1]);
    }

    #[test]
    fn test_ad_sqrt_gradient() {
        // f(x) = √x, df/dx = 1/(2√x)
        let x0 = 4.0;
        let grad = compute_gradient(|vars| vars[0].sqrt(), &[x0]);
        let expected = 0.5 / x0.sqrt();
        assert!((grad[0] - expected).abs() < 1e-12,
            "d/dx √x at x=4 should be {}, got {}", expected, grad[0]);
    }

    #[test]
    fn test_ad_many_parameters() {
        // f(x₁,...,x₁₀) = Σ xᵢ², ∂f/∂xᵢ = 2xᵢ
        let point: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let grad = compute_gradient(|vars| {
            let mut sum = vars[0].mul(vars[0]);
            for v in &vars[1..] {
                sum = sum.add(v.mul(*v));
            }
            sum
        }, &point);

        for i in 0..10 {
            let expected = 2.0 * point[i];
            assert!((grad[i] - expected).abs() < 1e-10,
                "∂f/∂x{} should be {}, got {}", i, expected, grad[i]);
        }
    }
}
