// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-particle-physics — Standard Model Dynamics
//!
//! Computes particle physics observables from first principles:
//!
//! - **Cross-sections**: QED (e⁺e⁻ → μ⁺μ⁻, Compton), Z resonance, R-ratio
//! - **Decay widths**: Muon, tau, W/Z bosons, pion, top quark
//! - **Running couplings**: α_s(Q²), α_EM(Q²), gauge unification projection
//! - **Symmetry groups**: SU(2) Pauli matrices, SU(3) Gell-Mann matrices,
//!   structure constants, Casimir operators
//! - **Lattice gauge primitives**: periodic SU(3) links, plaquettes, Wilson action,
//!   Polyakov loops, and local gauge transformations
//!
//! ## Natural Units
//!
//! All calculations use natural units (ℏ = c = 1) with energy in GeV.
//! Conversion factors are provided for seconds, picobarns, etc.
//!
//! ## References
//!
//! - Peskin & Schroeder (1995). *An Introduction to Quantum Field Theory*.
//! - PDG Review of Particle Physics (2024).
//! - Georgi, H. (1999). *Lie Algebras in Particle Physics*.
//! - Wilson, K. G. (1974). Phys. Rev. D 10, 2445.

pub mod constants;
pub mod cross_sections;
pub mod decay_widths;
pub mod field_quantization;
pub mod general_relativity;
pub mod lattice_gauge;
pub mod lattice_qcd;
pub mod relativistic_qm;
pub mod renormalization;
pub mod symmetry_groups;

// Re-export key items
pub use constants::*;
pub use cross_sections::{
    Mandelstam, alpha_em_running, alpha_s_running, r_ratio, sigma_compton, sigma_ee_to_mumu,
    sigma_ee_to_mumu_with_z,
};
pub use decay_widths::{
    DecayChannel, muon_decay_width, muon_lifetime, pion_lifetime, top_decay_width,
    w_boson_channels, w_total_width, z_total_width,
};
pub use lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_dagger, su3_determinant,
    su3_determinant_error, su3_diagonal, su3_identity, su3_mul, su3_trace,
    su3_unitarity_error, validate_su3,
};
pub use renormalization::{
    BetaCoefficients, approximate_unification_scale, gauge_couplings_at_scale, lambda_qcd,
    qcd_beta, qed_beta,
};
pub use symmetry_groups::{
    Complex, gell_mann_matrix, gell_mann_trace_product, pauli_matrices, su2_casimir,
    su2_structure_constant, su3_casimir_adjoint, su3_casimir_fundamental, su3_structure_constant,
};
