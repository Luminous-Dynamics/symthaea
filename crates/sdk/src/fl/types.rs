// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FL Types
//!
//! Core data structures for federated learning.
//!
//! # Security
//!
//! Gradient data is considered sensitive as it may reveal information about
//! training data. Structs containing gradient vectors implement `ZeroizeOnDrop`
//! to ensure secure memory cleanup.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::matl::{ProofOfGradientQuality, ReputationScore};

/// Gradient update metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientMetadata {
    /// Number of samples in this batch
    pub batch_size: usize,
    /// Training loss
    pub loss: f64,
    /// Optional accuracy metric
    pub accuracy: Option<f64>,
    /// Timestamp of the update
    pub timestamp: u64,
}

impl GradientMetadata {
    /// Create new metadata
    pub fn new(batch_size: usize, loss: f64) -> Self {
        Self {
            batch_size,
            loss,
            accuracy: None,
            timestamp: current_timestamp(),
        }
    }

    /// Create with accuracy
    pub fn with_accuracy(batch_size: usize, loss: f64, accuracy: f64) -> Self {
        Self {
            batch_size,
            loss,
            accuracy: Some(accuracy),
            timestamp: current_timestamp(),
        }
    }

    /// Validate metadata
    pub fn is_valid(&self) -> bool {
        self.batch_size > 0 && self.loss.is_finite()
    }
}

/// Gradient update from a participant
///
/// # Security
///
/// This struct contains sensitive gradient data that may reveal information
/// about the participant's training data. The `ZeroizeOnDrop` derive ensures
/// that the gradient vector is securely zeroed when the struct is dropped,
/// preventing memory scraping attacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Zeroize, ZeroizeOnDrop)]
pub struct GradientUpdate {
    /// Participant identifier
    #[zeroize(skip)]
    pub participant_id: String,
    /// Model version this update is for
    #[zeroize(skip)]
    pub model_version: u64,
    /// Gradient values (SENSITIVE - will be zeroized on drop)
    pub gradients: Vec<f64>,
    /// Update metadata
    #[zeroize(skip)]
    pub metadata: GradientMetadata,
}

impl GradientUpdate {
    /// Create a new gradient update
    pub fn new(
        participant_id: String,
        model_version: u64,
        gradients: Vec<f64>,
        batch_size: usize,
        loss: f64,
    ) -> Self {
        Self {
            participant_id,
            model_version,
            gradients,
            metadata: GradientMetadata::new(batch_size, loss),
        }
    }

    /// Create with full metadata
    pub fn with_metadata(
        participant_id: String,
        model_version: u64,
        gradients: Vec<f64>,
        metadata: GradientMetadata,
    ) -> Self {
        Self {
            participant_id,
            model_version,
            gradients,
            metadata,
        }
    }

    /// Get gradient dimension
    pub fn dimension(&self) -> usize {
        self.gradients.len()
    }

    /// Calculate L2 norm of gradients
    pub fn l2_norm(&self) -> f64 {
        self.gradients.iter().map(|g| g * g).sum::<f64>().sqrt()
    }

    /// Validate the update
    pub fn is_valid(&self) -> bool {
        !self.gradients.is_empty()
            && self.metadata.is_valid()
            && self.gradients.iter().all(|g| g.is_finite())
    }
}

/// Aggregated gradient result
///
/// # Security
///
/// This struct contains aggregated gradient data. While aggregated gradients
/// are less sensitive than individual participant gradients, they still may
/// leak information about the overall training process. The `ZeroizeOnDrop`
/// derive ensures secure memory cleanup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Zeroize, ZeroizeOnDrop)]
pub struct AggregatedGradient {
    /// Aggregated gradient values (SENSITIVE - will be zeroized on drop)
    pub gradients: Vec<f64>,
    /// Model version
    #[zeroize(skip)]
    pub model_version: u64,
    /// Number of participants included
    #[zeroize(skip)]
    pub participant_count: usize,
    /// Number of participants excluded (trust threshold, invalid, etc.)
    #[zeroize(skip)]
    pub excluded_count: usize,
    /// Aggregation method used
    #[zeroize(skip)]
    pub aggregation_method: AggregationMethod,
    /// Timestamp of aggregation
    #[zeroize(skip)]
    pub timestamp: u64,
}

impl AggregatedGradient {
    /// Create a new aggregated gradient
    pub fn new(
        gradients: Vec<f64>,
        model_version: u64,
        participant_count: usize,
        excluded_count: usize,
        aggregation_method: AggregationMethod,
    ) -> Self {
        Self {
            gradients,
            model_version,
            participant_count,
            excluded_count,
            aggregation_method,
            timestamp: current_timestamp(),
        }
    }

    /// Get gradient dimension
    pub fn dimension(&self) -> usize {
        self.gradients.len()
    }
}

/// Participant state for FL round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// Participant identifier
    pub id: String,
    /// Current reputation score
    pub reputation: ReputationScore,
    /// Latest PoGQ measurement
    pub pogq: Option<ProofOfGradientQuality>,
    /// Last gradient contribution
    pub last_contribution: Option<GradientUpdate>,
    /// Number of rounds participated
    pub rounds_participated: u64,
    /// Registration timestamp
    pub registered_at: u64,
}

impl Participant {
    /// Create a new participant
    pub fn new(id: String) -> Self {
        Self {
            reputation: ReputationScore::new(&id, "fl"),
            pogq: None,
            last_contribution: None,
            rounds_participated: 0,
            registered_at: current_timestamp(),
            id,
        }
    }

    /// Get trust score (reputation value)
    pub fn trust_score(&self) -> f64 {
        self.reputation.score
    }

    /// Check if participant meets trust threshold
    pub fn is_trusted(&self, threshold: f64) -> bool {
        self.trust_score() >= threshold
    }

    /// Record a positive contribution
    pub fn record_positive(&mut self) {
        self.reputation.record_positive();
        self.rounds_participated += 1;
    }

    /// Record a negative/suspicious contribution
    pub fn record_negative(&mut self) {
        self.reputation.record_negative();
    }
}

/// Round status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundStatus {
    /// Collecting gradient updates
    Collecting,
    /// Aggregating gradients
    Aggregating,
    /// Round completed
    Completed,
    /// Round failed (timeout, not enough participants, etc.)
    Failed,
}

/// Federated learning round state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLRound {
    /// Round identifier
    pub round_id: u64,
    /// Model version for this round
    pub model_version: u64,
    /// Participants in this round
    pub participants: HashMap<String, Participant>,
    /// Gradient updates received
    pub updates: Vec<GradientUpdate>,
    /// Round status
    pub status: RoundStatus,
    /// Start time
    pub start_time: u64,
    /// End time (if completed)
    pub end_time: Option<u64>,
    /// Aggregated result (if completed)
    pub aggregated_result: Option<AggregatedGradient>,
}

impl FLRound {
    /// Create a new round
    pub fn new(
        round_id: u64,
        model_version: u64,
        participants: HashMap<String, Participant>,
    ) -> Self {
        Self {
            round_id,
            model_version,
            participants,
            updates: Vec::new(),
            status: RoundStatus::Collecting,
            start_time: current_timestamp(),
            end_time: None,
            aggregated_result: None,
        }
    }

    /// Get number of updates received
    pub fn update_count(&self) -> usize {
        self.updates.len()
    }

    /// Check if round is still active
    pub fn is_active(&self) -> bool {
        self.status == RoundStatus::Collecting
    }

    /// Check if round is completed
    pub fn is_completed(&self) -> bool {
        self.status == RoundStatus::Completed
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_time
            .map(|end| end.saturating_sub(self.start_time) * 1000)
    }
}

/// Aggregation methods
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Standard Federated Averaging
    FedAvg,
    /// Trimmed Mean (removes outliers)
    TrimmedMean,
    /// Coordinate-wise Median
    Median,
    /// Krum algorithm (Byzantine-resistant)
    Krum,
    /// Trust-weighted aggregation (MATL integration)
    #[default]
    TrustWeighted,
}

/// FL coordinator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLConfig {
    /// Minimum participants for a round
    pub min_participants: usize,
    /// Maximum participants for a round
    pub max_participants: usize,
    /// Round timeout in milliseconds
    pub round_timeout_ms: u64,
    /// Byzantine tolerance (0.0-0.34, validated maximum)
    pub byzantine_tolerance: f64,
    /// Aggregation method to use
    pub aggregation_method: AggregationMethod,
    /// Trust threshold for participation
    pub trust_threshold: f64,
}

impl FLConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.min_participants < 1 {
            return Err("min_participants must be at least 1");
        }
        if self.max_participants < self.min_participants {
            return Err("max_participants cannot be less than min_participants");
        }
        if self.byzantine_tolerance < 0.0 || self.byzantine_tolerance > 0.34 {
            return Err("byzantine_tolerance must be between 0.0 and 0.34");
        }
        if self.trust_threshold < 0.0 || self.trust_threshold > 1.0 {
            return Err("trust_threshold must be between 0.0 and 1.0");
        }
        if self.round_timeout_ms < 1000 {
            return Err("round_timeout_ms must be at least 1000");
        }
        Ok(())
    }
}

impl Default for FLConfig {
    fn default() -> Self {
        Self {
            min_participants: super::DEFAULT_MIN_PARTICIPANTS,
            max_participants: super::DEFAULT_MAX_PARTICIPANTS,
            round_timeout_ms: super::DEFAULT_ROUND_TIMEOUT_MS,
            byzantine_tolerance: super::DEFAULT_BYZANTINE_TOLERANCE,
            aggregation_method: AggregationMethod::TrustWeighted,
            trust_threshold: super::DEFAULT_TRUST_THRESHOLD,
        }
    }
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_update() {
        let update = GradientUpdate::new(
            "participant-1".to_string(),
            1,
            vec![0.1, 0.2, 0.3],
            100,
            0.5,
        );

        assert_eq!(update.dimension(), 3);
        assert!(update.is_valid());
        assert!(update.l2_norm() > 0.0);
    }

    #[test]
    fn test_gradient_metadata() {
        let meta = GradientMetadata::new(100, 0.5);
        assert!(meta.is_valid());

        let meta_with_acc = GradientMetadata::with_accuracy(100, 0.5, 0.95);
        assert_eq!(meta_with_acc.accuracy, Some(0.95));
    }

    #[test]
    fn test_participant() {
        let mut participant = Participant::new("p1".to_string());
        assert_eq!(participant.rounds_participated, 0);

        participant.record_positive();
        assert_eq!(participant.rounds_participated, 1);
        assert!(participant.trust_score() > 0.5);
    }

    #[test]
    fn test_fl_config_validation() {
        let config = FLConfig::default();
        assert!(config.validate().is_ok());

        let bad_config = FLConfig {
            min_participants: 0,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_fl_round() {
        let participants = HashMap::new();
        let round = FLRound::new(1, 1, participants);

        assert!(round.is_active());
        assert!(!round.is_completed());
        assert_eq!(round.update_count(), 0);
    }
}
