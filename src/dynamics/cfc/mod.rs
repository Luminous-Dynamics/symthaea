// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Closed-form Continuous-time (CfC) Neural Networks
//!
//! Implementation of CfC networks with closed-form solutions for continuous-time
//! neural dynamics. The core equation is:
//!
//! ```text
//! h(t) = h_inf + (h_0 - h_inf) * exp(-dt/tau)
//! ```
//!
//! ## Modules
//!
//! - `types` - Configuration types, activation functions, and utility functions
//! - `cell` - Single CfC cell with forward/backward passes
//! - `gradients` - Gradient accumulators and Adam optimizer state
//! - `network` - Multi-layer CfC network with BPTT and SPSA training
//! - `phi_gated` - Phi-gated attention utilities for IIT integration

pub(crate) mod cell;
mod gradients;
pub(crate) mod network;
mod phi_gated;
pub(crate) mod types;

/// Candidate-only adapter from typed WeightPrograms to transactional CfC edits.
pub mod weight_program_adapter;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

// Re-export all public API from submodules

// Types, config, and utilities
pub use types::{
    ActivationType, CfCConfig, DynamicsDiagnostic, NetworkOnlineLearningStats,
    OnlineLearningConfig, OnlineLearningStats,
};

// MIN_TAU, fast_sigmoid, sigmoid, mse_loss are pub(crate) in types.rs
// and accessed by cell.rs/network.rs via `use super::types::*`.

// Cell
pub use cell::CfCCell;

// Gradients and optimizer state
pub use gradients::{AdamState, CfCCellCache, CfCGradients, OutputAdamState};

// Network
pub use network::{
    CfCNetwork, CfCNetworkConfig, CfCNetworkFingerprint, CfCParameterGroup, CfCParameterGroupId,
    CfCParameterGroupReplacement, CfCParameterManifest, CfCParameterRole, CfCWeightLoadError,
};

// Weight-space programming
pub use weight_program_adapter::{
    CfCActivationSignatureSet, CfCAlignmentAcceptancePolicy, CfCDisambiguatingProbeSelection,
    CfCFunctionalProbe, CfCGaussianPermutationEIGSelection, CfCGaussianPermutationProbeScore,
    CfCHiddenUnitAlignment, CfCPermutationAlignmentError, CfCPermutationHypothesisPosterior,
    CfCProbeCandidateScore, CfCProjectionNoiseCalibration, CfCProjectionNoiseCalibrationMethod,
    CfCWeightProgramAdapterError, CfCWeightProgramCandidate, align_cfc_hidden_unit_signatures,
    build_cfc_hidden_unit_canonicalization_program,
    build_cfc_network_hidden_unit_canonicalization_program,
    calibrate_cfc_projection_observation_noise, cfc_hidden_unit_activation_signatures,
    cfc_weight_model_snapshot, materialize_cfc_weight_program, select_cfc_disambiguating_probe,
    select_cfc_disambiguating_probe_by_gaussian_eig, validate_cfc_alignment_for_parameter_transfer,
};

// Phi-gated attention
pub use phi_gated::{PhiGatedConfig, compute_phi_attention_weights};
// weighted_array_bundle is pub(crate) in phi_gated.rs and accessed
// by network.rs via `use super::phi_gated::weighted_array_bundle`.
