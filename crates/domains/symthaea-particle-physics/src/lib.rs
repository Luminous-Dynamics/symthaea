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
//! - **Topology reference**: clover field strength/topological charge plus a
//!   deliberately slow finite-difference Wilson-action gradient-flow step
//! - **Optimized topology flow**: analytic six-staple Wilson gradient with
//!   parity against the finite-difference reference and degenerate fallback
//! - **Third-order topology flow**: independently qualified Lie-group RK3
//!   integration over the same six-staple Wilson-action gradient
//! - **Flowed gauge energy**: clover energy density plus ensemble-mean-only
//!   `t0`/`w0`-like crossing algebra with caller-supplied scale targets
//! - **Joint scale resampling**: chain-aware blocked delete-one resampling over
//!   complete per-configuration trajectories, preserving cross-flow covariance
//! - **Block adequacy evidence**: explicit autocorrelation coverage and
//!   caller-declared tau/block-count policy for the resampling geometry
//! - **Scale evidence**: same-ensemble flow curves whose derived uncertainty
//!   must bind joint resampling across the correlated flow-time trajectory
//! - **Flowed-topology lineage**: version-stable operator/flow identities with
//!   exact step-size, probe, step-count, and smoothing-time provenance
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
//! - Lüscher, M. (2010). arXiv:1006.4518.

pub mod constants;
pub mod cross_sections;
pub mod decay_widths;
pub mod field_quantization;
pub mod general_relativity;
pub mod lattice_flow_block_adequacy;
pub mod lattice_flow_energy;
pub mod lattice_flow_joint_evidence;
pub mod lattice_flow_joint_jackknife;
pub mod lattice_flow_scale_evidence;
pub mod lattice_gauge;
pub mod lattice_qcd;
mod lattice_su3_lie;
pub mod lattice_topology_flow;
pub mod lattice_topology_flow_rk3;
pub mod lattice_topology_flow_staple;
pub mod lattice_topology_measurement;
pub mod relativistic_qm;
pub mod renormalization;
pub mod symmetry_groups;

pub use constants::*;
pub use cross_sections::{
    Mandelstam, alpha_em_running, alpha_s_running, r_ratio, sigma_compton, sigma_ee_to_mumu,
    sigma_ee_to_mumu_with_z,
};
pub use decay_widths::{
    DecayChannel, muon_decay_width, muon_lifetime, pion_lifetime, top_decay_width,
    w_boson_channels, w_total_width, z_total_width,
};
pub use lattice_flow_block_adequacy::{
    FLOW_BLOCK_ADEQUACY_POLICY_ID, AutocorrelationEvidence, BlockAdequacyAssessment,
    BlockAdequacyError, BlockAdequacyPolicy, assess_flow_block_adequacy,
};
pub use lattice_flow_energy::{
    CLOVER_FLOW_ENERGY_ID, EnsembleFlowEnergyPoint, FlowEnergyError,
    clover_energy_density_at_site, dimensionless_flow_energy_curve,
    energy_density_from_field_strengths, mean_clover_energy_density,
    t0_like_from_ensemble_mean, w0_like_from_ensemble_mean,
};
pub use lattice_flow_joint_evidence::{
    JointJackknifeEvidenceError, JointJackknifeScaleEstimateEvidence,
    bind_joint_jackknife_scale_evidence,
};
pub use lattice_flow_joint_jackknife::{
    JOINT_BLOCKED_JACKKNIFE_ID, FlowEnergyTrajectory, JointBlockedJackknifeInput,
    JointBlockedJackknifeScale, JointFlowScaleKind, JointJackknifeError,
    joint_blocked_jackknife_scale,
};
pub use lattice_flow_scale_evidence::{
    FlowEnergyEvidenceCurve, FlowEnergyEvidencePoint, FlowScaleEstimateEvidence,
    FlowScaleEvidenceError, FlowScaleKind, bind_flow_scale_estimate,
};
pub use lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_dagger, su3_determinant,
    su3_determinant_error, su3_diagonal, su3_identity, su3_mul, su3_trace,
    su3_unitarity_error, validate_su3,
};
pub use lattice_topology_flow::{
    LatticeTopologyFlowError, ReferenceFlowStepStats, clover_field_strength, clover_sum,
    clover_topological_charge, clover_topological_density,
    finite_difference_wilson_flow_step_reference,
};
pub use lattice_topology_flow_rk3::{Rk3FlowError, Rk3FlowStepStats, rk3_wilson_flow_step};
pub use lattice_topology_flow_staple::{
    StapleFlowError, StapleFlowStepStats, analytic_link_gradient,
    link_staple_non_degenerate, staple_wilson_flow_step,
};
pub use lattice_topology_measurement::{
    CLOVER_TOPOLOGY_OPERATOR_ID, REFERENCE_WILSON_FLOW_ID, RK3_WILSON_FLOW_ID,
    FlowedTopologyDefinition, FlowedTopologyMeasurement, FlowedTopologyMeasurementError,
    ReferenceFlowSchedule, Rk3FlowSchedule, Rk3FlowedTopologyDefinition,
    Rk3FlowedTopologyMeasurement, measure_flowed_clover_topology_reference,
    measure_flowed_clover_topology_rk3,
};
pub use renormalization::{
    BetaCoefficients, approximate_unification_scale, gauge_couplings_at_scale, lambda_qcd,
    qcd_beta, qed_beta,
};
pub use symmetry_groups::{
    Complex, gell_mann_matrix, gell_mann_trace_product, pauli_matrices, su2_casimir,
    su2_structure_constant, su3_casimir_adjoint, su3_casimir_fundamental, su3_structure_constant,
};
