// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-Consistent Field (SCF) infrastructure.
//!
//! Provides the Restricted Hartree-Fock solver with DIIS convergence
//! acceleration, canonical orthogonalization, and all supporting machinery.

pub mod density;
pub mod diis;
pub mod fock;
pub mod generalized_eigen;
pub mod rhf;

pub use generalized_eigen::{
    GeneralizedEigenResult, canonical_orthogonalization, solve_generalized_eigen,
};
pub use rhf::{RhfConfig, RhfResult, restricted_hartree_fock};
