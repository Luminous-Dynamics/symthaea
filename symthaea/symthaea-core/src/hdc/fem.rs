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
    /// Solve the 1D Poisson equation -u'' = f on [0, L] with u(0)=u_a, u(L)=u_b.
    ///
    /// Uses linear finite elements and a constant source term f.
    pub fn solve_poisson_1d(mesh: &Mesh1D, f: f64, u_a: f64, u_b: f64) -> FEMResult {
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

            // Element load vector f_e = f*h/2 * [1; 1]
            let f_e = [f * h / 2.0, f * h / 2.0];

            // Map to global
            k_global[i * n + i] += k_e[0];
            k_global[i * n + j] += k_e[1];
            k_global[j * n + i] += k_e[2];
            k_global[j * n + j] += k_e[3];

            f_global[i] += f_e[0];
            f_global[j] += f_e[1];
        }

        // 2. Apply Dirichlet BCs (Penalty method or Row/Col zeroing)
        // Here we use row/col zeroing with 1 on diagonal and BC value in f.
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
        // We use the HdcMatrix solver from linear_algebra.rs (LU decomposition)
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
        // Max value at x=0.5 is 0.5 * 0.5 * 0.5 = 0.125
        let mesh = Mesh1D::uniform(1.0, 9); // 11 nodes total, 10 elements
        let result = FEMEngine::solve_poisson_1d(&mesh, 1.0, 0.0, 0.0);

        assert_eq!(result.u.len(), 11);
        assert!((result.u[0] - 0.0).abs() < 1e-12);
        assert!((result.u[10] - 0.0).abs() < 1e-12);

        // Check midpoint value u(0.5)
        let mid = result.u[5];
        assert!((mid - 0.125).abs() < 1e-3, "Expected 0.125, got {}", mid);

        assert!(result.residual < 1e-10);
        assert!(result.phi > 0.9);
    }

    #[test]
    fn test_poisson_1d_linear() {
        // -u'' = 0, u(0)=0, u(1)=1  ->  u(x) = x
        let mesh = Mesh1D::uniform(1.0, 5);
        let result = FEMEngine::solve_poisson_1d(&mesh, 0.0, 0.0, 1.0);

        for (i, &val) in result.u.iter().enumerate() {
            let x = i as f64 / (result.u.len() - 1) as f64;
            assert!(
                (val - x).abs() < 1e-12,
                "At x={}, expected {}, got {}",
                x,
                x,
                val
            );
        }
    }
}
