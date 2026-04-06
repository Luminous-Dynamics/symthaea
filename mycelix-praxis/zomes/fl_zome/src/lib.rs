// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Federated Learning Zome
//!
//! Manages federated learning rounds, updates, and aggregation.

use praxis_core::{ModelHash, RoundId, RoundState};
use serde::{Deserialize, Serialize};

/// Federated learning update entry
///
/// Represents a gradient or parameter update from a single participant
/// in a federated learning round.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

// TODO: Implement HDK entry definitions and validation
// TODO: Implement zome functions for:
//   - create_round
//   - join_round
//   - submit_update
//   - aggregate_updates
//   - get_round_status
//   - list_rounds

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fl_update_creation() {
        let update = FlUpdate {
            round_id: RoundId("test-round".to_string()),
            model_id: "model-1".to_string(),
            parent_model_hash: ModelHash("parent-hash".to_string()),
            grad_commitment: vec![1, 2, 3],
            clipped_l2_norm: 1.0,
            local_val_loss: 0.5,
            sample_count: 100,
            timestamp: 1234567890,
        };

        assert_eq!(update.round_id.0, "test-round");
    }
}
