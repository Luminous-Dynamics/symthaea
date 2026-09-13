// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA — Evidence for Understanding, Reasoning, Epistemics, Knowledge and Abstraction.
//!
//! This module is an evaluation/qualification surface over existing Symthaea
//! cognition. It must not become a parallel reasoning, world-model, causal,
//! metacognitive, or authority stack.

mod action_execution;
mod analysis_plan;
mod baselines;
mod campaign;
mod campaign_manifest;
pub mod consequence;
pub mod constitution;
mod cross_family_analysis;
mod custody;
#[cfg(test)]
mod custody_identity_tests;
#[cfg(feature = "symthaea-backend")]
mod comparator_subject;
#[cfg(feature = "symthaea-backend")]
mod fep_comparator_freeze;
#[cfg(feature = "symthaea-backend")]
mod fep_development;
#[cfg(feature = "symthaea-backend")]
mod heldout_seal;
pub mod hidden_world;
mod promotion;
mod relay_triad;
#[cfg(feature = "symthaea-backend")]
mod relay_triad_comparator;
#[cfg(feature = "symthaea-backend")]
mod relay_triad_comparator_subject;
#[cfg(feature = "symthaea-backend")]
mod relay_triad_development;
#[cfg(feature = "symthaea-backend")]
mod relay_triad_heldout_seal;
mod selection;
mod target_contract;
mod target_lineage;
#[cfg(test)]
mod transition_identity_tests;

pub use consequence::{
    ConsequenceMetrics, ConsequencePrediction, ConsequenceScore, ConsequenceScoringError,
    PredictionOutcome, copy_current_state_baseline, score_consequence,
};
pub use constitution::{
    EUREKA_CLAIM_SPECS, EvidenceClass, ProtocolInvariant, ScientificDisposition,
    UnderstandingClaimFamily, UnderstandingClaimSpec, UnderstandingMaturity,
};
pub use hidden_world::{
    InterventionReceipt, InterventionRequest, InterventionStatus, PublicAction, PublicObservation,
    PublicValue, RuntimeWorld, StepReceipt, PUBLIC_SCHEMA_ID,
};
pub use target_contract::{EurekaTargetScope, FEP_TARGET_ADAPTER_REVISION};