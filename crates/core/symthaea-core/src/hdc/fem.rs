// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Finite Element Method (FEM) Engine
//!
//! Numerical PDE solvers via Galerkin weak forms and element-wise assembly.
//!
//! ## Capabilities
//!
//! - **1D Poisson**: -u'' = f, with Dirichlet boundary conditions.
//! - **Linear Basis Functions**: Piecewise linear "hat" functions.
//! - **Global Assembly**: Sparse-to-dense global stiffness matrix construction.
//! - **Consciousness Coupling**: Phi computed from residual norm and energy conservation.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::linear_algebra::{HdcMatrix, HdcVector};
use crate::hdc::primitive_system::seed_from_name;

/// A 1D finite element (interval).
#[derive(Debug, Clone, Copy)]
pub struct Element1D {
    /// Indices of the two nodes forming the element.
    pub nodes: [usize; 2],
}

/// A 1D mesh.
#[derive(Debug, Clone)]
pub struct Mesh1D {
    /// Coordinates of the nodes.
    pub coords: Vec<f64>,
    /// Elements (connectivity).
    pub elements: Vec<Element1D>,
}

impl Mesh1D {
    /// Create a uniform 1D mesh on [0, L] with N interior nodes.
    pub fn uniform(length: f64, n_interior: usize) -> Self {
        let n_total = n_interior + 2;
        let dx = length / (n_total - 1) as f64;
        let coords: Vec<f64> = (0..n_total).map(|i| i as f64 * dx).collect();
        let elements: Vec<Element1D> = (0..n_total - 1)
            .map(|i| Element1D { nodes: [i, i + 1] })
            .collect();

        Mesh1D { coords, elements }
    }

    /// Create a non-uniform 1D mesh.
    pub fn non_uniform(coords: Vec<f64>) -> Result<Self, &'static str> {
        if coords.len() < 2 {
            return Err("Mesh must have at least 2 nodes");
        }
        for i in 0..coords.len() - 1 {
            if coords[i + 1] <= coords[i] {
                return Err("Coordinates must be strictly increasing");
            }
        }
        let elements: Vec<Element1D> = (0..coords.len() - 1)
            .map(|i| Element1D { nodes: [i, i + 1] })
            .collect();

        Ok(Mesh1D { coords, elements })
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.coords.len()
    }
}

/// Result of an FEM solve.
#[derive(Debug, Clone)]
pub struct FEMResult {
    /// Solution vector u at each node.
    pub u: Vec<f64>,
    /// Residual norm |Au - b|.
    pub residual: f64,
    /// Phi measure (inverse residual + energy stability).
    pub phi: f64,
    /// HDC encoding of the solution.
    pub encoding: BinaryHV,
}

/// Finite Element Method Engine.
pub struct FEMEngine;

impl FEMEngine {
    /// Solve the 1D Poisson equation -u'' = f(x) on [0, L] with u(0)=u_a, u(L)=u_b.
    ///
    /// Uses linear finite elements and a provided source function f.
    pub fn solve_poisson_1d<F>(mesh: &Mesh1D, f: F, u_a: f64, u_b: f64) -> FEMResult
    where
        F: Fn(f64) -> f64,
    {
        let n = mesh.n_nodes();
        let mut k_global = vec![0.0; n * n];
        let mut f_global = vec![0.0; n];

        // 1. Element assembly
        for elem in &mesh.elements {
            let i = elem.nodes[0];
            let j = elem.nodes[1];
            let x_i = mesh.coords[i];
            let x_j = mesh.coords[j];
            let h = x_j - x_i;

            // Element stiffness matrix k_e = 1/h * [1, -1; -1, 1]
            let k_e = [1.0 / h, -1.0 / h, -1.0 / h, 1.0 / h];

            // Element load vector f_e via midpoint quadrature
            let x_mid = (x_i + x_j) / 2.0;
            let f_val = f(x_mid);
            let f_e = [f_val * h / 2.0, f_val * h / 2.0];

            // Map to global
            k_global[i * n + i] += k_e[0];
            k_global[i * n + j] += k_e[1];
            k_global[j * n + i] += k_e[2];
            k_global[j * n + j] += k_e[3];

            f_global[i] += f_e[0];
            f_global[j] += f_e[1];
        }

        // 2. Apply Dirichlet BCs
        let bc_indices = [0, n - 1];
        let bc_values = [u_a, u_b];

        for (idx, &val) in bc_indices.iter().zip(bc_values.iter()) {
            // Zero out the row
            for j in 0..n {
                k_global[*idx * n + j] = 0.0;
            }
            k_global[*idx * n + *idx] = 1.0;
            f_global[*idx] = val;

            // Zero out the column (to maintain symmetry)
            for i in 0..n {
                if i != *idx {
                    f_global[i] -= k_global[i * n + *idx] * val;
                    k_global[i * n + *idx] = 0.0;
                }
            }
        }

        // 3. Solve Ku = f
        let matrix = HdcMatrix {
            data: k_global,
            rows: n,
            cols: n,
            encoding: BinaryHV::zero(),
            phi: 0.0,
        };

        let b = HdcVector::new(f_global.clone());
        let (u_vec, _linalg_res) = matrix.solve(&b);
        let u = u_vec.data;

        // 4. Compute Phi and Encoding
        let residual = Self::compute_residual(&matrix.data, &u, &f_global);
        let phi = 1.0 / (1.0 + residual);
        let encoding = Self::encode_fem_result(&u);

        FEMResult {
            u,
            residual,
            phi,
            encoding,
        }
    }

    fn compute_residual(k: &[f64], u: &[f64], f: &[f64]) -> f64 {
        let n = f.len();
        let mut res = 0.0;
        for i in 0..n {
            let mut ku_i = 0.0;
            for j in 0..n {
                ku_i += k[i * n + j] * u[j];
            }
            res += (ku_i - f[i]).powi(2);
        }
        res.sqrt()
    }

    fn encode_fem_result(u: &[f64]) -> BinaryHV {
        let base = BinaryHV::random(seed_from_name("FEM_RESULT"));
        let state_hash: u64 = u.iter().enumerate().fold(0u64, |acc, (i, &v)| {
            acc.wrapping_add(v.to_bits().wrapping_mul(i as u64 + 1))
        });
        let state_hv = BinaryHV::random(seed_from_name(&format!("FEM_STATE_{}", state_hash)));
        base.bind(&state_hv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_1d_uniform_load() {
        // -u'' = 1, u(0)=0, u(1)=0  ->  u(x) = 0.5 * x * (1 - x)
        let mesh = Mesh1D::uniform(1.0, 9);
        let result = FEMEngine::solve_poisson_1d(&mesh, |_| 1.0, 0.0, 0.0);

        assert_eq!(result.u.len(), 11);
        assert!((result.u[0] - 0.0).abs() < 1e-12);
        assert!((result.u[10] - 0.0).abs() < 1e-12);

        let mid = result.u[5];
        assert!((mid - 0.125).abs() < 1e-3);
    }

    #[test]
    fn test_nonzero_boundary_values() {
        // -u'' = 0, u(0)=1, u(1)=2  ->  u(x) = 1 + x
        let mesh = Mesh1D::uniform(1.0, 5);
        let result = FEMEngine::solve_poisson_1d(&mesh, |_| 0.0, 1.0, 2.0);

        assert!((result.u[0] - 1.0).abs() < 1e-12);
        assert!((result.u[6] - 2.0).abs() < 1e-12);
        assert!((result.u[3] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_convergence_rate() {
        // Use non-polynomial source to avoid nodal exactness
        // -u'' = sin(pi*x), u(0)=0, u(1)=0  ->  U(x) = (1/pi^2) * sin(pi*x)
        let pi = std::f64::consts::PI;
        let analytic = |x: f64| (1.0 / (pi * pi)) * (pi * x).sin();
        let source = |x: f64| (pi * x).sin();

        let mut errors = Vec::new();
        let ns = vec![8, 16, 32, 64];

        for &n in &ns {
            let mesh = Mesh1D::uniform(1.0, n);
            let result = FEMEngine::solve_poisson_1d(&mesh, source, 0.0, 0.0);

            let mut l2_sq = 0.0;
            for (i, &x) in mesh.coords.iter().enumerate() {
                let u_num = result.u[i];
                let u_exact = analytic(x);
                l2_sq += (u_num - u_exact).powi(2);
            }
            let l2_error = (l2_sq / mesh.coords.len() as f64).sqrt();
            errors.push(l2_error);
        }

        // L2 error should decrease as elements increase
        for i in 0..errors.len() - 1 {
            assert!(
                errors[i + 1] < errors[i],
                "L2 error did not decrease: {} vs {}",
                errors[i],
                errors[i + 1]
            );
        }

        // Ratio error(h)/error(h/2) should be ~4 for quadratic convergence
        let ratio = errors[1] / errors[2];
        assert!(ratio > 3.0, "Convergence rate too low: {}", ratio);
    }

    #[test]
    fn test_energy_norm_error() {
        // -u'' = 1, u(0)=0, u(1)=0 -> U(x) = 0.5*x*(1-x), U'(x) = 0.5 - x
        let analytic_derivative = |x: f64| 0.5 - x;

        let mut errors = Vec::new();
        let ns = vec![8, 16, 32];

        for &n in &ns {
            let mesh = Mesh1D::uniform(1.0, n);
            let result = FEMEngine::solve_poisson_1d(&mesh, |_| 1.0, 0.0, 0.0);

            // Energy error sq = integral( (u_h' - U')^2 dx )
            let mut energy_error_sq = 0.0;
            for elem in &mesh.elements {
                let x_i = mesh.coords[elem.nodes[0]];
                let x_j = mesh.coords[elem.nodes[1]];
                let h = x_j - x_i;

                // u_h' on this element is (u_j - u_i) / h
                let uh_prime = (result.u[elem.nodes[1]] - result.u[elem.nodes[0]]) / h;

                // integral_xi^xj (uh_prime - (0.5 - x))^2 dx
                // We use 2-point Gaussian quadrature for exactness on polynomials
                let g_points = [-0.5773502691896257, 0.5773502691896257];
                let g_weights = [1.0, 1.0];

                for (qp, qw) in g_points.iter().zip(g_weights.iter()) {
                    let x = 0.5 * (x_i + x_j) + 0.5 * h * qp;
                    let diff = uh_prime - analytic_derivative(x);
                    energy_error_sq += 0.5 * h * qw * diff.powi(2);
                }
            }
            errors.push(energy_error_sq.sqrt());
        }

        // Energy error should decrease as elements increase
        for i in 0..errors.len() - 1 {
            assert!(
                errors[i + 1] < errors[i],
                "Energy error did not decrease: {} vs {}",
                errors[i],
                errors[i + 1]
            );
        }

        // H1 convergence should be O(h)
        let ratio = errors[0] / errors[1];
        assert!(ratio > 1.8, "Energy convergence rate too low: {}", ratio);
    }
}
