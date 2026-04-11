// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-quantum-chemistry — Ab Initio Quantum Chemistry
//!
//! Computes molecular electronic structure from first principles using
//! Gaussian basis sets and the Obara-Saika integral scheme.
//!
//! ## Capabilities
//!
//! - **Basis sets**: STO-3G (minimal), with 6-31G* and cc-pVDZ planned
//! - **Molecular integrals**: Overlap (S), Kinetic (T), Nuclear attraction (V),
//!   Electron repulsion integrals (ERIs) with Schwarz prescreening
//! - **Generalized eigenvalue solver**: Canonical orthogonalization (numerically
//!   stable against linear dependence, unlike Cholesky)
//! - **Hartree-Fock**: Restricted HF with DIIS acceleration (Phase 2)
//! - **DFT**: Kohn-Sham with LDA/PBE functionals (Phase 3)
//! - **MP2**: Møller-Plesset perturbation theory (Phase 3)
//!
//! ## Scope
//!
//! Small molecules only (< 200 basis functions). Pure Rust, WASM-compatible.
//! No external quantum chemistry library dependencies.
//!
//! ## Consciousness Coupling
//!
//! Novel research angle: molecular orbitals are encoded as HDC hypervectors,
//! enabling Phi-weighted orbital delocalization measurement and
//! consciousness-gated accuracy selection (higher Phi → better functional).
//!
//! ## References
//!
//! - Szabo & Ostlund (1996). *Modern Quantum Chemistry*. Dover.
//! - Obara & Saika (1986). J. Chem. Phys. 84, 3963.
//! - Hehre, Stewart & Pople (1969). J. Chem. Phys. 51, 2657 (STO-3G).
//! - Boys (1950). Proc. R. Soc. Lond. A 200, 542 (Boys function).

pub mod basis;
pub mod constants;
pub mod integrals;
pub mod molecule;
pub mod scf;

// Re-export key types for convenience
pub use basis::{BasisSet, BasisSetProvider, ContractedGaussian, PrimitiveGaussian, ShellType};
pub use molecule::{Atom, Molecule};
pub use scf::generalized_eigen::GeneralizedEigenResult;
pub use scf::rhf::{restricted_hartree_fock, RhfConfig, RhfResult};
