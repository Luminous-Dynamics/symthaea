// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-linalg
//!
//! Pure-`std` dense linear algebra for the zero-dependency domain-crate
//! ecosystem. The *main* crate has `nalgebra`, but the pure-std domain crates
//! cannot use it, and several had begun reinventing Gaussian elimination
//! in-place (`symthaea-markov`'s fundamental matrix, ad-hoc solves). This is the
//! shared home for that.
//!
//! Zero dependencies, no `symthaea-core` link. Every routine is checked against
//! a hand-computed result.
//!
//! ## Contents
//! - [`matrix::Matrix`] — dense row-major matrix: mul, transpose, mul_vec,
//!   identity
//! - [`decomp`] — LU decomposition (partial pivoting) → determinant, `solve`,
//!   `inverse`
//! - [`eigen`] — dominant eigenpair (power iteration) and all eigenvalues of a
//!   symmetric matrix (Jacobi)
//!
//! ## Example
//!
//! ```
//! use symthaea_linalg::{Matrix, solve};
//! // 2x + y = 3 ; x + 3y = 5  →  (0.8, 1.4).
//! let a = Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 3.0]]).unwrap();
//! let x = solve(&a, &[3.0, 5.0]).unwrap();
//! assert!((x[0] - 0.8).abs() < 1e-12 && (x[1] - 1.4).abs() < 1e-12);
//! ```

pub mod decomp;
pub mod eigen;
pub mod matrix;

pub use decomp::{determinant, inverse, lu_decompose, solve};
pub use eigen::{jacobi_eigenvalues, power_iteration};
pub use matrix::Matrix;
