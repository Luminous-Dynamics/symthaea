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
//! - **Hartree-Fock**: Restricted HF (`scf::rhf`) with DIIS acceleration for
//!   closed-shell (multiplicity=1) molecules -- open-shell input is rejected with a
//!   clear panic rather than silently given a wrong closed-shell answer (Phase Q0,
//!   2026-07-16). Unrestricted HF (`scf::uhf`) for open-shell systems (radicals,
//!   triplets, doublets) landed Phase Q2, 2026-07-16, including spin-contamination
//!   reporting (⟨S²⟩). ROHF is still not implemented. Both RHF and UHF support two
//!   ERI strategies (Phase Q3, 2026-07-16): a dense-tensor "conventional" mode
//!   (default) and an opt-in "direct" mode (`RhfConfig::direct` / `UhfConfig::direct`)
//!   that recomputes integrals on demand instead of caching them -- see `scf::fock`'s
//!   module doc for the memory/compute tradeoff.
//! - **DFT**: Kohn-Sham self-consistent SCF with the LDA functional only
//!   (`dft::xc::XcFunctional` has a single `Lda` variant). PBE **exchange** is
//!   implemented (`dft::pbe::PbeExchange`, constants fetched from libxc, Phase Q5d
//!   2026-07-17) but only as a **post-hoc, non-self-consistent** evaluation
//!   (`dft::xc::pbe_exchange_energy_posthoc`) on an already-converged density --
//!   not wired into the SCF/Fock-matrix build, and PBE correlation isn't
//!   implemented at all. Real analytic basis-function and density gradients
//!   (`dft::xc::evaluate_basis_gradient_on_grid`,
//!   `dft::xc::compute_density_gradient_at_grid`) exist as of the same phase,
//!   verified via finite-difference cross-checks.
//! - **MP2**: Møller-Plesset perturbation theory (Phase 3), plus frozen-core and
//!   SCS-MP2 variants (Phase Q5b, 2026-07-17, `post_hf::mp2`).
//! - **CIS**: real Configuration Interaction Singles (`time_dependent::cis_excitations`,
//!   Phase Q5, 2026-07-17) -- builds and diagonalizes the actual CIS matrix, not just
//!   Koopmans'-theorem orbital-energy differences (that cheaper approximation is kept
//!   as `time_dependent::koopmans_theorem_excitations`).
//!
//! ## Experimental / not-yet-validated modules
//!
//! `coupled_cluster` (CCD) exists but is not a production implementation -- see its
//! module doc comment for its actual status as of the Phase Q0 truth-and-cleanup pass
//! (2026-07-16), not yet fixed. Three known HF energy discrepancies
//! (N2/STO-3G, H2O and CH4/6-31G) are tracked, disclosed, and not yet root-caused --
//! see `validation.rs`'s doc comments.
//!
//! ## Scope
//!
//! Small-to-moderate molecules. The dense-tensor ERI mode (`compute_eri_tensor`,
//! still the default) allocates an `O(n^4)` array, impractical much past a few dozen
//! basis functions (200^4 doubles ~= 12.8 GB). The direct mode (Phase Q3, 2026-07-16,
//! `RhfConfig`/`UhfConfig::direct`) removes that specific memory ceiling by never
//! materializing the dense tensor -- `O(n^2)` memory instead, at the cost of
//! recomputing some integrals up to 8x per SCF iteration instead of caching them
//! once. Neither mode changes the underlying `O(n^2.2)`-ish Schwarz-screened
//! computational cost; direct mode is a memory/compute tradeoff, not a complexity-
//! class improvement. Pure Rust, WASM-compatible. No external quantum chemistry
//! library dependencies.
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
pub mod element_data;
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
pub mod vibrational;

// Re-export key types for convenience
pub use basis::{BasisSet, BasisSetProvider, ContractedGaussian, PrimitiveGaussian, ShellType};
pub use consciousness::{OrbitalPhiMeasurement, build_atom_basis_ranges, compute_orbital_phi};
pub use dft::{DftConfig, DftResult, XcFunctional, kohn_sham_dft};
pub use geometry_opt::{GeomOptConfig, GeomOptResult, optimize_geometry};
pub use molecule::{Atom, Molecule};
pub use post_hf::mp2::{
    Mp2Result, mp2_correlation_energy, mp2_correlation_energy_frozen_core,
    scs_mp2_correlation_energy, total_frozen_core,
};
pub use scf::generalized_eigen::GeneralizedEigenResult;
pub use scf::rhf::{RhfConfig, RhfResult, restricted_hartree_fock};
pub use vibrational::{
    Frequency, ThermoConfig, ThermoResult, VibConfig, VibrationalResult,
    compute_thermochemistry_linear, normal_mode_analysis,
};
