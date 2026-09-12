// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA — Evidence for Understanding, Reasoning, Epistemics, Knowledge and Abstraction.
//!
//! This module is an evaluation/qualification surface over existing Symthaea
//! cognition. It must not become a parallel reasoning, world-model, causal,
//! metacognitive, or authority stack.

pub mod consequence;
pub mod constitution;
mod custody;
pub mod hidden_world;

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
