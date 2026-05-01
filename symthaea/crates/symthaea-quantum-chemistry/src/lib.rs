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
pub mod bridge;
pub mod cognitive_loop_bridge;
pub mod consciousness;
pub mod constants;
pub mod coupled_cluster;
pub mod dft;
pub mod emergent_consciousness;
pub mod geometry_opt;
pub mod integrals;
pub mod molecular_dynamics;
pub mod molecule;
pub mod multi_theory;
pub mod post_hf;
pub mod quantum_info;
pub mod reaction_consciousness;
pub mod scf;
pub mod stat_mech;
pub mod time_dependent;
pub mod validation;

// Re-export key types for convenience
pub use basis::{BasisSet, BasisSetProvider, ContractedGaussian, PrimitiveGaussian, ShellType};
pub use consciousness::{build_atom_basis_ranges, compute_orbital_phi, OrbitalPhiMeasurement};
pub use dft::{kohn_sham_dft, DftConfig, DftResult, XcFunctional};
pub use geometry_opt::{optimize_geometry, GeomOptConfig, GeomOptResult};
pub use molecule::{Atom, Molecule};
pub use post_hf::mp2::{mp2_correlation_energy, Mp2Result};
pub use scf::generalized_eigen::GeneralizedEigenResult;
pub use scf::rhf::{restricted_hartree_fock, RhfConfig, RhfResult};
