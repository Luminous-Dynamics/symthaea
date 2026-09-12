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
//! - **Lattice update primitives**: deterministic embedded-SU(2) Metropolis
//!   proposal steps with local touching-plaquette action deltas
//! - **Lattice sweep contract**: symmetric proposal transform, pluggable uniform
//!   source, lexicographic site/direction/subgroup composition, and sweep stats
//! - **Lattice RNG streams**: pinned ChaCha8 implementation with injective
//!   campaign/replica/rank stream coordinates and endpoint-free U(0,1)
//! - **SU(2) heat-bath sampling**: Kennedy-Pendleton scalar rejection against an
//!   independently qualified target density, plus an exact Haar scalar fallback
//! - **Reference SU(3) heat-bath**: five-probe local-force reconstruction,
//!   Wilson-coupling normalization, force-direction orientation, and one-link
//!   Cabibbo-Marinari subgroup conditional updates
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
//! - Cabibbo, N. & Marinari, E. (1982). Phys. Lett. B 119, 387-390.
//! - Kennedy, A. D. & Pendleton, B. J. (1985). Phys. Lett. B 156, 393-399.

pub mod constants;
pub mod cross_sections;
pub mod decay_widths;
pub mod field_quantization;
pub mod general_relativity;
pub mod lattice_gauge;
pub mod lattice_heatbath;
pub mod lattice_heatbath_su3;
pub mod lattice_metropolis;
pub mod lattice_qcd;
pub mod lattice_rng;
pub mod lattice_sweep;
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
pub use lattice_heatbath::{
    Su2HeatbathError, Su2HeatbathMethod, Su2HeatbathSample,
    draw_kennedy_pendleton_scalar, draw_su2_heatbath_quaternion,
};
pub use lattice_heatbath_su3::{
    Su3SubgroupHeatbathDraw, Su3SubgroupHeatbathError, Su3SubgroupHeatbathForce,
    Su3SubgroupHeatbathStepResult, draw_su3_subgroup_heatbath_rotation,
    heatbath_subgroup_step_reference, heatbath_touching_trace_sum,
    orient_heatbath_quaternion_to_force, probe_su3_subgroup_heatbath_force,
};
pub use lattice_metropolis::{
    LatticeMetropolisError, MetropolisStepResult, Su2Subgroup, Su2SubgroupProposal,
    affected_wilson_action, embedded_su2_rotation, metropolis_acceptance_probability,
    metropolis_subgroup_step,
};
pub use lattice_rng::{
    LATTICE_RNG_ALGORITHM, LATTICE_RNG_IMPLEMENTATION, LATTICE_RNG_VERSION,
    LatticeChaCha8Stream, LatticeRngError, LatticeStreamCoordinates, LatticeStreamDomain,
};
pub use lattice_sweep::{
    LatticeSweepError, SweepStats, SymmetricProposalConfig, Uniform01Source,
    draw_symmetric_subgroup_proposal, metropolis_sweep,
};
pub use renormalization::{
    BetaCoefficients, approximate_unification_scale, gauge_couplings_at_scale, lambda_qcd,
    qcd_beta, qed_beta,
};
pub use symmetry_groups::{
    Complex, gell_mann_matrix, gell_mann_trace_product, pauli_matrices, su2_casimir,
    su2_structure_constant, su3_casimir_adjoint, su3_casimir_fundamental, su3_structure_constant,
};
