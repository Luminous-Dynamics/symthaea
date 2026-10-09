// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Narrow, deterministic evaluator for preregistered predictions about observable
//! external-AI behavior.
//!
//! This crate compares an operational prediction with a supplied observation. It
//! does not measure phenomenal experience, implement a consciousness detector, or
//! grant scientific or operational authority. See the AI-OBS-011 protocol for the
//! claim ceiling and integration plan.

#![deny(unsafe_code)]

pub mod theory_comparison;

pub use theory_comparison::{
    ConfoundStatus, EvaluationReason, EvaluationReceipt, ManipulationCheck,
    Observation, ObservationOutcome, ObservabilityTier, PairwiseDiscrimination,
    PredictionDisposition, PredictionRegistry, RegistryFreezeAnchor, RegistryValidationError,
    TheoryPrediction, TrialCoverage, assess_trial_coverage,
    compare_predictions, digest_bytes, evaluate_prediction,
    has_duplicate_source_trajectory_ids,
};
