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

pub mod constants;
pub mod cross_sections;
pub mod decay_widths;
pub mod field_quantization;
pub mod general_relativity;
pub mod lattice_cornell;
pub mod lattice_covariance;
pub mod lattice_linear_gls;
pub mod lattice_potential_fit_family;
pub mod lattice_qcd;
pub mod lattice_sommer_scale;
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
pub use lattice_cornell::{
    LATTICE_CORNELL_GLS_ID, LatticeCornellError, LatticeCornellFit, LatticeCornellPoint,
    fit_declared_lattice_cornell,
};
pub use lattice_covariance::{
    CorrelatedConstantFit, CovarianceError, correlated_constant_fit, sample_covariance,
};
pub use lattice_linear_gls::{CorrelatedLinearFit, correlated_linear_fit};
pub use lattice_potential_fit_family::{
    LATTICE_POTENTIAL_FIT_FAMILY_ID, LatticePotentialFitFamily, PotentialFitFamilyError,
    PotentialFitMember, UNIVERSAL_IR_COULOMB_COEFFICIENT,
    fit_declared_lattice_potential_family,
};
pub use lattice_sommer_scale::{
    FIXED_E_PI_OVER_12_FREE_L_MODEL_ID, FIXED_E_PI_OVER_12_L0_MODEL_ID,
    FREE_V0_SIGMA_E_L_MODEL_ID, R0_C, R4_C, R6_C, SOMMER_SCALE_FROM_CORRELATED_FIT_ID,
    SommerScaleError, SommerScaleEstimate, StandardSommerScales,
    sommer_scale_from_fit_member, standard_sommer_scales_from_fit_member,
};
pub use renormalization::{
    BetaCoefficients, approximate_unification_scale, gauge_couplings_at_scale, lambda_qcd,
    qcd_beta, qed_beta,
};
pub use symmetry_groups::{
    Complex, gell_mann_matrix, gell_mann_trace_product, pauli_matrices, su2_casimir,
    su2_structure_constant, su3_casimir_adjoint, su3_casimir_fundamental, su3_structure_constant,
};
