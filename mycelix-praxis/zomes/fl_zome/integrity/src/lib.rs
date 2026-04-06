// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # FL Integrity Zome
//!
//! Defines entry types and validation rules for federated learning.
//! This zome is immutable - entry definitions cannot change without breaking data.

use hdi::prelude::*;
use praxis_core::{ModelHash, RoundId, RoundState};

/// Federated learning update entry
///
/// Represents a gradient or parameter update from a single participant
/// in a federated learning round.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FlUpdate {
    /// Unique identifier for the FL round
    pub round_id: RoundId,

    /// Model being updated
    pub model_id: String,

    /// Hash of the parent model this update is based on
    pub parent_model_hash: ModelHash,

    /// Commitment to the gradient/update (for privacy)
    pub grad_commitment: Vec<u8>,

    /// L2 norm of the clipped gradient
    pub clipped_l2_norm: f32,

    /// Local validation loss (for quality assessment)
    pub local_val_loss: f32,

    /// Number of local training samples
    pub sample_count: u32,

    /// Timestamp of update creation
    pub timestamp: i64,
}

/// Federated learning round entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FlRound {
    /// Unique identifier for this round
    pub round_id: RoundId,

    /// Model being trained
    pub model_id: String,

    /// Current state of the round
    pub state: RoundState,

    /// Hash of the base model for this round
    pub base_model_hash: ModelHash,

    /// Minimum number of participants required
    pub min_participants: u32,

    /// Maximum number of participants allowed
    pub max_participants: u32,

    /// Current number of participants
    pub current_participants: u32,

    /// Aggregation method to use
    pub aggregation_method: String,

    /// L2 norm clipping threshold
    pub clip_norm: f32,

    /// Round start timestamp
    pub started_at: i64,

    /// Round completion timestamp (if completed)
    pub completed_at: Option<i64>,

    /// Hash of aggregated model (if released)
    pub aggregated_model_hash: Option<ModelHash>,

    /// Privacy budget (epsilon for differential privacy)
    /// REVOLUTIONARY: Adaptive privacy budgets based on data sensitivity
    pub privacy_epsilon: Option<f32>,

    /// Privacy delta (for (ε,δ)-differential privacy)
    pub privacy_delta: Option<f32>,
}

/// Privacy parameters for a round
/// Used to configure adaptive privacy protection
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PrivacyParams {
    /// Round this applies to
    pub round_id: RoundId,

    /// Base privacy budget
    pub base_epsilon: f32,

    /// Minimum privacy budget (never go below this)
    pub min_epsilon: f32,

    /// Maximum privacy budget (never exceed this)
    pub max_epsilon: f32,

    /// Data sensitivity score (0.0 = low, 1.0 = high)
    /// Higher sensitivity = lower epsilon (stronger privacy)
    pub sensitivity_score: f32,
}

/// All entry types for this integrity zome
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    FlUpdate(FlUpdate),
    FlRound(FlRound),
    PrivacyParams(PrivacyParams),
}

/// All link types for this integrity zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Links from FlRound to FlUpdates
    RoundToUpdates,
    /// Links from model to rounds
    ModelToRounds,
    /// Links from round to privacy params
    RoundToPrivacy,
}

/// Validation function for FlUpdate entries
pub fn validate_fl_update(update: FlUpdate) -> ExternResult<ValidateCallbackResult> {
    // Validate L2 norm is non-negative
    if !update.clipped_l2_norm.is_finite() || update.clipped_l2_norm < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "L2 norm must be non-negative".to_string(),
        ));
    }

    // Validate sample count is positive
    if update.sample_count == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Sample count must be positive".to_string(),
        ));
    }

    // Validate local validation loss is reasonable
    if update.local_val_loss < 0.0 || update.local_val_loss.is_infinite() || update.local_val_loss.is_nan() {
        return Ok(ValidateCallbackResult::Invalid(
            "Local validation loss must be a valid non-negative number".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation function for FlRound entries
pub fn validate_fl_round(round: FlRound) -> ExternResult<ValidateCallbackResult> {
    // Validate participant bounds
    if round.min_participants > round.max_participants {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum participants cannot exceed maximum participants".to_string(),
        ));
    }

    if round.min_participants == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum participants must be at least 1".to_string(),
        ));
    }

    // Validate current participants is within bounds
    if round.current_participants > round.max_participants {
        return Ok(ValidateCallbackResult::Invalid(
            "Current participants exceeds maximum".to_string(),
        ));
    }

    // Validate clip norm is positive
    if !round.clip_norm.is_finite() || round.clip_norm <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Clip norm must be positive".to_string(),
        ));
    }

    // Validate privacy parameters if present
    if let Some(epsilon) = round.privacy_epsilon {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Privacy epsilon must be positive".to_string(),
            ));
        }
    }

    if let Some(delta) = round.privacy_delta {
        if !delta.is_finite() || delta < 0.0 || delta > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Privacy delta must be between 0 and 1".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation function for PrivacyParams entries
pub fn validate_privacy_params(params: PrivacyParams) -> ExternResult<ValidateCallbackResult> {
    // Validate epsilon bounds
    if !params.min_epsilon.is_finite() || !params.max_epsilon.is_finite() || !params.base_epsilon.is_finite() || params.min_epsilon <= 0.0 || params.max_epsilon <= 0.0 || params.base_epsilon <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "All epsilon values must be positive".to_string(),
        ));
    }

    if params.min_epsilon > params.base_epsilon || params.base_epsilon > params.max_epsilon {
        return Ok(ValidateCallbackResult::Invalid(
            "Epsilon bounds must satisfy: min <= base <= max".to_string(),
        ));
    }

    // Validate sensitivity score
    if !params.sensitivity_score.is_finite() || params.sensitivity_score < 0.0 || params.sensitivity_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Sensitivity score must be between 0 and 1".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => match store_entry.action.hashed.content.entry_type() {
            EntryType::App(app_entry_def) => {
                let entry = store_entry.entry;
                match EntryTypes::deserialize_from_type(
                    app_entry_def.zome_index,
                    app_entry_def.entry_index,
                    &entry,
                )? {
                    Some(EntryTypes::FlUpdate(update)) => validate_fl_update(update),
                    Some(EntryTypes::FlRound(round)) => validate_fl_round(round),
                    Some(EntryTypes::PrivacyParams(params)) => validate_privacy_params(params),
                    None => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests;
