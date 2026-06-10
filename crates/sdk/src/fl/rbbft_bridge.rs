// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RB-BFT Bridge for Federated Learning
//!
//! Integrates Reputation-Based Byzantine Fault Tolerance consensus with
//! federated learning for 34% validated Byzantine tolerance through K-Vector reputation.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────┐
//! │                  RB-BFT Federated Learning Pipeline                    │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │                                                                        │
//! │  Gradient Submission:               Consensus & Aggregation:           │
//! │  ┌────────────────┐                 ┌────────────────┐                │
//! │  │ ZK Proof       │──────────────▶  │ RB-BFT Vote    │                │
//! │  │ (Quality)      │                 │ (Reputation²)  │                │
//! │  └────────────────┘                 └────────────────┘                │
//! │         │                                  │                          │
//! │         ▼                                  ▼                          │
//! │  ┌────────────────┐                 ┌────────────────┐                │
//! │  │ K-Vector       │──────────────▶  │ Weighted Avg   │                │
//! │  │ (Trust)        │                 │ (34% Validated)│                │
//! │  └────────────────┘                 └────────────────┘                │
//! │                                            │                          │
//! │                                            ▼                          │
//! │                                     Global Model Update               │
//! └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **Reputation² Weighting**: High-reputation participants have quadratically more influence
//! - **34% Validated Byzantine Tolerance**: Improvement over classical 33% with reputation weighting
//! - **K-Vector Integration**: Trust scores from MATL drive aggregation weights
//! - **Consensus Voting**: Participants vote on whether to accept/reject round
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::fl::{RbbftFLBridge, RbbftFLConfig};
//! use mycelix_sdk::matl::kvector::KVector;
//!
//! let mut bridge = RbbftFLBridge::new(RbbftFLConfig::default());
//!
//! // Register participants with K-Vectors
//! bridge.register_participant("client-1", KVector::with_reputation(0.9));
//! bridge.register_participant("client-2", KVector::with_reputation(0.8));
//!
//! // Start round and collect votes
//! bridge.start_round(&model_hash);
//! bridge.submit_vote("client-1", true, &proof)?;
//! bridge.submit_vote("client-2", true, &proof)?;
//!
//! // Check consensus
//! if bridge.check_consensus() {
//!     let result = bridge.finalize_round()?;
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use super::types::{AggregatedGradient, AggregationMethod, GradientUpdate};
use crate::matl::{KVector, QUORUM_THRESHOLD, RBBFT_BYZANTINE_THRESHOLD};

/// Errors from the RB-BFT FL bridge
#[derive(Debug, Error)]
pub enum RbbftFLError {
    /// Participant was not found by the given identifier
    #[error("Participant not found: {0}")]
    ParticipantNotFound(String),

    /// Participant reputation is below the required minimum
    #[error("Participant not eligible: reputation {0} < {1}")]
    InsufficientReputation(f32, f32),

    /// No consensus round is currently active
    #[error("No active round")]
    NoActiveRound,

    /// A participant attempted to vote more than once in a round
    #[error("Duplicate vote from {0}")]
    DuplicateVote(String),

    /// Quorum threshold was not met for consensus
    #[error("Consensus not reached: {0:.2}% < {1:.2}%")]
    ConsensusNotReached(f32, f32),

    /// Too many Byzantine (faulty) participants detected
    #[error("Byzantine threshold exceeded: {0:.2}% > {1:.2}%")]
    ByzantineThresholdExceeded(f32, f32),

    /// Not enough participants to form a valid consensus round
    #[error("Insufficient participants: {0} < {1}")]
    InsufficientParticipants(usize, usize),

    /// A submitted proof failed validation
    #[error("Invalid proof: {0}")]
    InvalidProof(String),
}

/// Configuration for RB-BFT FL bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbbftFLConfig {
    /// Minimum reputation to participate (default 0.3)
    pub min_reputation: f32,
    /// Minimum participants for consensus (default 3)
    pub min_participants: usize,
    /// Byzantine tolerance threshold (default 0.34, validated maximum)
    pub byzantine_threshold: f32,
    /// Quorum threshold for consensus (default 0.667)
    pub quorum_threshold: f32,
    /// Whether to use quadratic reputation weighting
    pub use_quadratic_weighting: bool,
    /// Whether to require ZK proofs for votes
    pub require_proofs: bool,
}

impl Default for RbbftFLConfig {
    fn default() -> Self {
        Self {
            min_reputation: 0.3,
            min_participants: 3,
            byzantine_threshold: RBBFT_BYZANTINE_THRESHOLD,
            quorum_threshold: QUORUM_THRESHOLD,
            use_quadratic_weighting: true,
            require_proofs: true,
        }
    }
}

impl RbbftFLConfig {
    /// Production configuration with strict requirements
    pub fn production() -> Self {
        Self {
            min_reputation: 0.5,
            min_participants: 5,
            byzantine_threshold: RBBFT_BYZANTINE_THRESHOLD,
            quorum_threshold: QUORUM_THRESHOLD,
            use_quadratic_weighting: true,
            require_proofs: true,
        }
    }

    /// Testing configuration with relaxed requirements
    pub fn testing() -> Self {
        Self {
            min_reputation: 0.1,
            min_participants: 2,
            byzantine_threshold: RBBFT_BYZANTINE_THRESHOLD,
            quorum_threshold: 0.5,
            use_quadratic_weighting: true,
            require_proofs: false,
        }
    }
}

/// A participant in the RB-BFT FL system
#[derive(Debug, Clone)]
pub struct RbbftParticipant {
    /// Participant identifier
    pub id: String,
    /// Current K-Vector trust profile
    pub kvector: KVector,
    /// Total rounds participated
    pub rounds_participated: u64,
    /// Successful contributions
    pub successful_contributions: u64,
    /// Failed/rejected contributions
    pub failed_contributions: u64,
    /// Registration timestamp
    pub registered_at: u64,
}

impl RbbftParticipant {
    /// Create a new participant
    pub fn new(id: &str, kvector: KVector) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id: id.to_string(),
            kvector,
            rounds_participated: 0,
            successful_contributions: 0,
            failed_contributions: 0,
            registered_at: timestamp,
        }
    }

    /// Get voting weight (reputation squared)
    pub fn voting_weight(&self, use_quadratic: bool) -> f32 {
        if use_quadratic {
            self.kvector.k_r.powi(2)
        } else {
            self.kvector.k_r
        }
    }

    /// Check if eligible to participate
    pub fn is_eligible(&self, min_reputation: f32) -> bool {
        self.kvector.k_r >= min_reputation
    }

    /// Record successful contribution
    pub fn record_success(&mut self) {
        self.rounds_participated += 1;
        self.successful_contributions += 1;
        // Small reputation boost
        self.kvector.k_r = (self.kvector.k_r + 0.005).min(1.0);
    }

    /// Record failed contribution
    pub fn record_failure(&mut self) {
        self.rounds_participated += 1;
        self.failed_contributions += 1;
        // Reputation penalty
        self.kvector.k_r = (self.kvector.k_r - 0.02).max(0.0);
    }

    /// Get success rate
    pub fn success_rate(&self) -> f32 {
        if self.rounds_participated == 0 {
            1.0 // Assume good until proven otherwise
        } else {
            self.successful_contributions as f32 / self.rounds_participated as f32
        }
    }
}

/// A vote in the consensus round
#[derive(Debug, Clone)]
pub struct RbbftVote {
    /// Voter ID
    pub participant_id: String,
    /// Vote value (true = accept, false = reject)
    pub value: bool,
    /// Voter's K-Vector at vote time
    pub kvector: KVector,
    /// The gradient being voted on
    pub gradient: Option<GradientUpdate>,
    /// Whether proof was valid (if required)
    pub proof_valid: bool,
    /// Timestamp
    pub timestamp: u64,
}

impl RbbftVote {
    /// Create a new vote
    pub fn new(
        participant_id: &str,
        value: bool,
        kvector: KVector,
        gradient: Option<GradientUpdate>,
        proof_valid: bool,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            participant_id: participant_id.to_string(),
            value,
            kvector,
            gradient,
            proof_valid,
            timestamp,
        }
    }

    /// Get vote weight (reputation squared if valid)
    pub fn weight(&self, use_quadratic: bool) -> f32 {
        if !self.proof_valid {
            return 0.0;
        }

        if use_quadratic {
            self.kvector.k_r.powi(2)
        } else {
            self.kvector.k_r
        }
    }

    /// Get weighted vote value
    pub fn weighted_value(&self, use_quadratic: bool) -> f32 {
        if self.value {
            self.weight(use_quadratic)
        } else {
            0.0
        }
    }
}

/// State of a consensus round
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundState {
    /// Collecting votes
    Voting,
    /// Consensus reached, aggregating
    Aggregating,
    /// Round completed successfully
    Completed,
    /// Round failed (no consensus or Byzantine threshold exceeded)
    Failed,
}

/// Information about a round
#[derive(Debug, Clone)]
pub struct RoundInfo {
    /// Round number
    pub round: u64,
    /// Current state
    pub state: RoundState,
    /// Model hash for this round
    pub model_hash: [u8; 32],
    /// Start timestamp
    pub started_at: u64,
    /// End timestamp (if completed)
    pub ended_at: Option<u64>,
    /// Votes collected
    pub votes: Vec<RbbftVote>,
    /// Aggregation result (if completed)
    pub result: Option<AggregatedGradient>,
}

impl RoundInfo {
    fn new(round: u64, model_hash: [u8; 32]) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            round,
            state: RoundState::Voting,
            model_hash,
            started_at: timestamp,
            ended_at: None,
            votes: Vec::new(),
            result: None,
        }
    }
}

/// RB-BFT Federated Learning Bridge
///
/// Coordinates federated learning rounds with RB-BFT consensus.
#[derive(Debug)]
pub struct RbbftFLBridge {
    /// Configuration
    config: RbbftFLConfig,
    /// Registered participants
    participants: HashMap<String, RbbftParticipant>,
    /// Current round (if any)
    current_round: Option<RoundInfo>,
    /// Completed rounds
    completed_rounds: Vec<RoundInfo>,
    /// Next round number
    next_round: u64,
}

impl RbbftFLBridge {
    /// Create a new bridge with the given configuration
    pub fn new(config: RbbftFLConfig) -> Self {
        Self {
            config,
            participants: HashMap::new(),
            current_round: None,
            completed_rounds: Vec::new(),
            next_round: 1,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RbbftFLConfig::default())
    }

    /// Register a participant
    pub fn register_participant(&mut self, id: &str, kvector: KVector) -> bool {
        if kvector.k_r < self.config.min_reputation {
            return false;
        }

        let participant = RbbftParticipant::new(id, kvector);
        self.participants.insert(id.to_string(), participant);
        true
    }

    /// Update participant's K-Vector
    pub fn update_kvector(&mut self, id: &str, kvector: KVector) -> bool {
        if let Some(participant) = self.participants.get_mut(id) {
            participant.kvector = kvector;
            true
        } else {
            false
        }
    }

    /// Get participant info
    pub fn get_participant(&self, id: &str) -> Option<&RbbftParticipant> {
        self.participants.get(id)
    }

    /// Count eligible participants
    pub fn eligible_participant_count(&self) -> usize {
        self.participants
            .values()
            .filter(|p| p.is_eligible(self.config.min_reputation))
            .count()
    }

    /// Get total voting weight
    pub fn total_voting_weight(&self) -> f32 {
        self.participants
            .values()
            .filter(|p| p.is_eligible(self.config.min_reputation))
            .map(|p| p.voting_weight(self.config.use_quadratic_weighting))
            .sum()
    }

    /// Start a new round
    pub fn start_round(&mut self, model_hash: [u8; 32]) -> Result<u64, RbbftFLError> {
        let eligible_count = self.eligible_participant_count();
        if eligible_count < self.config.min_participants {
            return Err(RbbftFLError::InsufficientParticipants(
                eligible_count,
                self.config.min_participants,
            ));
        }

        let round_num = self.next_round;
        self.next_round += 1;

        let round = RoundInfo::new(round_num, model_hash);
        self.current_round = Some(round);

        Ok(round_num)
    }

    /// Submit a vote for the current round
    pub fn submit_vote(
        &mut self,
        participant_id: &str,
        value: bool,
        gradient: Option<GradientUpdate>,
        proof_valid: bool,
    ) -> Result<(), RbbftFLError> {
        // Validate participant
        let participant = self
            .participants
            .get(participant_id)
            .ok_or_else(|| RbbftFLError::ParticipantNotFound(participant_id.to_string()))?;

        if !participant.is_eligible(self.config.min_reputation) {
            return Err(RbbftFLError::InsufficientReputation(
                participant.kvector.k_r,
                self.config.min_reputation,
            ));
        }

        // Check for active round
        let round = self
            .current_round
            .as_mut()
            .ok_or(RbbftFLError::NoActiveRound)?;

        // Check for duplicate vote
        if round
            .votes
            .iter()
            .any(|v| v.participant_id == participant_id)
        {
            return Err(RbbftFLError::DuplicateVote(participant_id.to_string()));
        }

        // Validate proof if required
        if self.config.require_proofs && !proof_valid {
            return Err(RbbftFLError::InvalidProof(
                "Proof validation failed".to_string(),
            ));
        }

        // Create and record vote
        let vote = RbbftVote::new(
            participant_id,
            value,
            participant.kvector,
            gradient,
            proof_valid,
        );
        round.votes.push(vote);

        Ok(())
    }

    /// Check if consensus has been reached
    pub fn check_consensus(&self) -> bool {
        if let Some(ref round) = self.current_round {
            let (weighted_for, weighted_total) = self.calculate_vote_totals(&round.votes);

            if weighted_total == 0.0 {
                return false;
            }

            let ratio = weighted_for / weighted_total;
            ratio >= self.config.quorum_threshold
        } else {
            false
        }
    }

    /// Get current consensus ratio
    pub fn consensus_ratio(&self) -> Option<f32> {
        self.current_round.as_ref().map(|round| {
            let (weighted_for, weighted_total) = self.calculate_vote_totals(&round.votes);
            if weighted_total == 0.0 {
                0.0
            } else {
                weighted_for / weighted_total
            }
        })
    }

    /// Calculate vote totals
    fn calculate_vote_totals(&self, votes: &[RbbftVote]) -> (f32, f32) {
        let mut weighted_for = 0.0f32;
        let mut weighted_total = 0.0f32;

        for vote in votes {
            let weight = vote.weight(self.config.use_quadratic_weighting);
            weighted_total += weight;
            if vote.value {
                weighted_for += weight;
            }
        }

        // Include non-voters as implicit rejections
        let total_possible = self.total_voting_weight();
        if weighted_total < total_possible {
            weighted_total = total_possible;
        }

        (weighted_for, weighted_total)
    }

    /// Estimate Byzantine fraction from invalid proofs
    pub fn estimate_byzantine_fraction(&self) -> f32 {
        if let Some(ref round) = self.current_round {
            if round.votes.is_empty() {
                return 0.0;
            }

            let invalid_count = round.votes.iter().filter(|v| !v.proof_valid).count();
            invalid_count as f32 / round.votes.len() as f32
        } else {
            0.0
        }
    }

    /// Finalize the current round with aggregation
    pub fn finalize_round(&mut self) -> Result<AggregatedGradient, RbbftFLError> {
        // Check consensus
        if !self.check_consensus() {
            let ratio = self.consensus_ratio().unwrap_or(0.0) * 100.0;
            let threshold = self.config.quorum_threshold * 100.0;
            return Err(RbbftFLError::ConsensusNotReached(ratio, threshold));
        }

        // Check Byzantine threshold
        let byzantine_fraction = self.estimate_byzantine_fraction();
        if byzantine_fraction > self.config.byzantine_threshold {
            return Err(RbbftFLError::ByzantineThresholdExceeded(
                byzantine_fraction * 100.0,
                self.config.byzantine_threshold * 100.0,
            ));
        }

        let mut round = self
            .current_round
            .take()
            .ok_or(RbbftFLError::NoActiveRound)?;

        // Aggregate valid gradients with reputation weighting
        let valid_votes: Vec<&RbbftVote> = round
            .votes
            .iter()
            .filter(|v| v.value && v.proof_valid && v.gradient.is_some())
            .collect();

        if valid_votes.is_empty() {
            return Err(RbbftFLError::InsufficientParticipants(0, 1));
        }

        // Get gradient dimension
        let dimension = valid_votes[0]
            .gradient
            .as_ref()
            .ok_or(RbbftFLError::InsufficientParticipants(0, 1))?
            .gradients
            .len();

        // Compute weighted average
        let mut total_weight = 0.0f32;
        let mut aggregated_gradients = vec![0.0f64; dimension];

        for vote in &valid_votes {
            let weight = vote.weight(self.config.use_quadratic_weighting);
            total_weight += weight;

            if let Some(ref gradient) = vote.gradient {
                for (i, g) in gradient.gradients.iter().enumerate() {
                    aggregated_gradients[i] += g * weight as f64;
                }
            }
        }

        // Normalize
        if total_weight > 0.0 {
            for g in &mut aggregated_gradients {
                *g /= total_weight as f64;
            }
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let result = AggregatedGradient {
            gradients: aggregated_gradients,
            model_version: round.round,
            participant_count: valid_votes.len(),
            excluded_count: round.votes.len() - valid_votes.len(),
            aggregation_method: AggregationMethod::TrustWeighted,
            timestamp,
        };

        // Update participant stats
        for vote in &round.votes {
            if let Some(participant) = self.participants.get_mut(&vote.participant_id) {
                if vote.value && vote.proof_valid {
                    participant.record_success();
                } else {
                    participant.record_failure();
                }
            }
        }

        // Finalize round
        round.state = RoundState::Completed;
        round.ended_at = Some(timestamp / 1000);
        round.result = Some(result.clone());
        self.completed_rounds.push(round);

        Ok(result)
    }

    /// Get current round info
    pub fn current_round_info(&self) -> Option<&RoundInfo> {
        self.current_round.as_ref()
    }

    /// Get completed rounds
    pub fn completed_rounds(&self) -> &[RoundInfo] {
        &self.completed_rounds
    }

    /// Get statistics
    pub fn stats(&self) -> RbbftFLStats {
        let total_participants = self.participants.len();
        let eligible_participants = self.eligible_participant_count();
        let total_rounds = self.completed_rounds.len();
        let successful_rounds = self
            .completed_rounds
            .iter()
            .filter(|r| r.state == RoundState::Completed)
            .count();

        let avg_participation = if total_rounds > 0 {
            self.completed_rounds
                .iter()
                .map(|r| r.votes.len())
                .sum::<usize>() as f32
                / total_rounds as f32
        } else {
            0.0
        };

        RbbftFLStats {
            total_participants,
            eligible_participants,
            total_rounds,
            successful_rounds,
            avg_participation_per_round: avg_participation,
            total_voting_weight: self.total_voting_weight(),
        }
    }
}

/// Statistics for the RB-BFT FL bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbbftFLStats {
    /// Total registered participants
    pub total_participants: usize,
    /// Eligible participants (above min reputation)
    pub eligible_participants: usize,
    /// Total completed rounds
    pub total_rounds: usize,
    /// Successful rounds
    pub successful_rounds: usize,
    /// Average participation per round
    pub avg_participation_per_round: f32,
    /// Total voting weight
    pub total_voting_weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kvector(reputation: f32) -> KVector {
        KVector {
            k_r: reputation,
            k_a: 0.5,
            k_i: 0.5,
            k_p: 0.5,
            k_m: 0.5,
            k_s: 0.5,
            k_h: 0.5,
            k_topo: 0.5,
            k_v: 0.5,
            k_coherence: 0.5,
        }
    }

    fn make_gradient(size: usize, scale: f64) -> GradientUpdate {
        let gradients: Vec<f64> = (0..size)
            .map(|i| (i as f64 * 0.001).sin() * scale)
            .collect();

        GradientUpdate::new("test".to_string(), 1, gradients, 32, 0.5)
    }

    #[test]
    fn test_basic_consensus() {
        let mut bridge = RbbftFLBridge::new(RbbftFLConfig::testing());

        // Register participants
        bridge.register_participant("client-1", make_kvector(0.9));
        bridge.register_participant("client-2", make_kvector(0.8));
        bridge.register_participant("client-3", make_kvector(0.7));

        // Start round
        let round = bridge.start_round([0u8; 32]).unwrap();
        assert_eq!(round, 1);

        // Submit votes
        let grad = make_gradient(100, 0.1);
        bridge
            .submit_vote("client-1", true, Some(grad.clone()), true)
            .unwrap();
        bridge
            .submit_vote("client-2", true, Some(grad.clone()), true)
            .unwrap();

        // Check consensus
        assert!(bridge.check_consensus());
    }

    #[test]
    fn test_reputation_weighting() {
        let mut bridge = RbbftFLBridge::new(RbbftFLConfig::testing());

        // High rep vs low rep
        bridge.register_participant("high-rep", make_kvector(0.9));
        bridge.register_participant("low-rep", make_kvector(0.3));

        bridge.start_round([0u8; 32]).unwrap();

        // High rep votes yes with larger gradient
        let grad1 = make_gradient(100, 1.0);
        bridge
            .submit_vote("high-rep", true, Some(grad1), true)
            .unwrap();

        // Low rep votes yes with smaller gradient
        let grad2 = make_gradient(100, 0.1);
        bridge
            .submit_vote("low-rep", true, Some(grad2), true)
            .unwrap();

        let result = bridge.finalize_round().unwrap();

        // Result should be weighted towards high-rep's larger gradient
        // Check index 50 where sin(0.05) ≈ 0.05 gives a non-trivial value
        // High rep contributes scale=1.0, low rep scale=0.1, weighted by reputation²
        assert!(
            result.gradients[50].abs() > 0.01,
            "Weighted gradient at index 50 should be non-trivial"
        );
    }

    #[test]
    fn test_byzantine_detection() {
        let mut bridge = RbbftFLBridge::new(RbbftFLConfig::testing());

        bridge.register_participant("good-1", make_kvector(0.9));
        bridge.register_participant("good-2", make_kvector(0.8));
        bridge.register_participant("bad", make_kvector(0.7));

        bridge.start_round([0u8; 32]).unwrap();

        let grad = make_gradient(100, 0.1);
        bridge
            .submit_vote("good-1", true, Some(grad.clone()), true)
            .unwrap();
        bridge
            .submit_vote("good-2", true, Some(grad.clone()), true)
            .unwrap();
        // Bad participant with invalid proof
        bridge.submit_vote("bad", true, Some(grad), false).ok(); // Will fail but we track it

        // Byzantine fraction should reflect the invalid proof
        let byzantine = bridge.estimate_byzantine_fraction();
        assert!(byzantine > 0.0);
    }

    #[test]
    fn test_duplicate_vote_rejected() {
        let mut bridge = RbbftFLBridge::new(RbbftFLConfig::testing());

        bridge.register_participant("client-1", make_kvector(0.9));
        bridge.register_participant("client-2", make_kvector(0.8));

        bridge.start_round([0u8; 32]).unwrap();

        let grad = make_gradient(100, 0.1);
        bridge
            .submit_vote("client-1", true, Some(grad.clone()), true)
            .unwrap();

        // Try to vote again
        let result = bridge.submit_vote("client-1", true, Some(grad), true);
        assert!(matches!(result, Err(RbbftFLError::DuplicateVote(_))));
    }

    #[test]
    fn test_insufficient_reputation_rejected() {
        let mut bridge = RbbftFLBridge::new(RbbftFLConfig::default());

        // Try to register with low reputation
        assert!(!bridge.register_participant("low-rep", make_kvector(0.1)));
    }

    #[test]
    fn test_stats() {
        let mut bridge = RbbftFLBridge::new(RbbftFLConfig::testing());

        bridge.register_participant("client-1", make_kvector(0.9));
        bridge.register_participant("client-2", make_kvector(0.8));

        let stats = bridge.stats();
        assert_eq!(stats.total_participants, 2);
        assert_eq!(stats.eligible_participants, 2);
        assert!(stats.total_voting_weight > 1.0);
    }
}
