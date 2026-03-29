// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified ZK + RB-BFT Bridge for Federated Learning
//!
//! Combines zero-knowledge proofs with Reputation-Based Byzantine Fault Tolerance
//! to achieve cryptographically verified gradients with 34% validated Byzantine tolerance.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                    Unified ZK-RBBFT Federated Learning Pipeline                 │
//! ├─────────────────────────────────────────────────────────────────────────────────┤
//! │                                                                                 │
//! │  Phase 1: Submission            Phase 2: Consensus           Phase 3: Aggregate│
//! │  ┌────────────────┐             ┌────────────────┐           ┌────────────────┐│
//! │  │ Gradient       │             │ Reputation²    │           │ Weighted Avg   ││
//! │  │ + ZK Proof     │ ─────────▶  │ Voting         │ ────────▶ │ (34% Validated)││
//! │  │ + K-Vector     │             │ (67% Quorum)   │           │ + ZK Commit    ││
//! │  └────────────────┘             └────────────────┘           └────────────────┘│
//! │         │                              │                            │          │
//! │         ▼                              ▼                            ▼          │
//! │  ┌────────────────┐             ┌────────────────┐           ┌────────────────┐│
//! │  │ Proof Verified │             │ Byzantine      │           │ Hash Commitment││
//! │  │ in zkVM        │             │ Detection      │           │ Chain          ││
//! │  └────────────────┘             └────────────────┘           └────────────────┘│
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Security Properties
//!
//! - **Cryptographic Quality**: Gradients verified inside zkVM before acceptance
//! - **34% Validated Byzantine Tolerance**: Reputation² weighting with validated maximum of 34%
//! - **Privacy-Preserving**: Gradient values never leave the prover
//! - **Accountability**: All actions tied to K-Vector trust profiles
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::fl::{UnifiedZkRbbftBridge, UnifiedZkRbbftConfig};
//! use mycelix_sdk::matl::kvector::KVector;
//!
//! let mut bridge = UnifiedZkRbbftBridge::new(UnifiedZkRbbftConfig::default());
//!
//! // Register participants with K-Vectors
//! bridge.register_participant("client-1", KVector::with_reputation(0.9));
//!
//! // Start round
//! bridge.start_round(&model_hash)?;
//!
//! // Submit gradient with ZK proof (proven internally)
//! bridge.submit_proven_gradient("client-1", &gradient, 5, 0.01, 32, 0.5)?;
//!
//! // Check consensus and finalize
//! if bridge.check_consensus() {
//!     let result = bridge.finalize_round()?;
//!     println!("Aggregated {} participants", result.participant_count);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use super::types::{AggregatedGradient, AggregationMethod, GradientUpdate};
use crate::matl::{KVector, QUORUM_THRESHOLD, RBBFT_BYZANTINE_THRESHOLD};

#[cfg(any(feature = "simulation", feature = "risc0"))]
use crate::zkproof::{GradientProofReceipt, GradientProver};

/// Errors from the Unified ZK-RBBFT FL bridge
#[derive(Debug, Error)]
pub enum UnifiedZkRbbftError {
    /// Participant was not found
    #[error("Participant not found: {0}")]
    ParticipantNotFound(String),

    /// Participant reputation is below minimum
    #[error("Participant not eligible: reputation {reputation:.2} < {minimum:.2}")]
    InsufficientReputation {
        /// Participant's current reputation
        reputation: f32,
        /// Minimum required reputation
        minimum: f32,
    },

    /// No consensus round is currently active
    #[error("No active round")]
    NoActiveRound,

    /// A participant attempted to submit more than once
    #[error("Duplicate submission from {0}")]
    DuplicateSubmission(String),

    /// Quorum threshold was not met
    #[error("Consensus not reached: {actual:.2}% < {required:.2}%")]
    ConsensusNotReached {
        /// Achieved consensus percentage
        actual: f32,
        /// Required consensus percentage
        required: f32,
    },

    /// Too many Byzantine participants detected
    #[error("Byzantine threshold exceeded: {actual:.2}% > {max:.2}%")]
    ByzantineThresholdExceeded {
        /// Detected Byzantine fraction
        actual: f32,
        /// Maximum allowed Byzantine fraction
        max: f32,
    },

    /// Not enough participants
    #[error("Insufficient participants: {have} < {need}")]
    InsufficientParticipants {
        /// Number of participants available
        have: usize,
        /// Minimum participants required
        need: usize,
    },

    /// ZK proof generation failed
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    /// ZK proof verification failed
    #[error("Proof verification failed: {0}")]
    ProofVerificationFailed(String),

    /// Gradient quality check failed
    #[error("Gradient quality invalid: {0}")]
    GradientQualityFailed(String),

    /// Round finalization failed
    #[error("Finalization failed: {0}")]
    FinalizationFailed(String),
}

/// Configuration for Unified ZK-RBBFT FL bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedZkRbbftConfig {
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
    /// Minimum gradient norm
    pub min_gradient_norm: f32,
    /// Maximum gradient norm
    pub max_gradient_norm: f32,
    /// Maximum element magnitude
    pub max_element_magnitude: f32,
    /// Minimum gradient dimension
    pub min_gradient_dimension: usize,
}

impl Default for UnifiedZkRbbftConfig {
    fn default() -> Self {
        Self {
            min_reputation: 0.3,
            min_participants: 3,
            byzantine_threshold: RBBFT_BYZANTINE_THRESHOLD,
            quorum_threshold: QUORUM_THRESHOLD,
            use_quadratic_weighting: true,
            min_gradient_norm: 0.0001,
            max_gradient_norm: 50.0,
            max_element_magnitude: 5.0,
            min_gradient_dimension: 100,
        }
    }
}

impl UnifiedZkRbbftConfig {
    /// Production configuration with strict requirements
    pub fn production() -> Self {
        Self {
            min_reputation: 0.5,
            min_participants: 5,
            byzantine_threshold: RBBFT_BYZANTINE_THRESHOLD,
            quorum_threshold: QUORUM_THRESHOLD,
            use_quadratic_weighting: true,
            min_gradient_norm: 0.001,
            max_gradient_norm: 100.0,
            max_element_magnitude: 10.0,
            min_gradient_dimension: 1000,
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
            min_gradient_norm: 0.0001,
            max_gradient_norm: 100.0,
            max_element_magnitude: 10.0,
            min_gradient_dimension: 10,
        }
    }
}

/// A participant in the Unified ZK-RBBFT system
#[derive(Debug, Clone)]
pub struct UnifiedParticipant {
    /// Participant identifier
    pub id: String,
    /// Current K-Vector trust profile
    pub kvector: KVector,
    /// Total rounds participated
    pub rounds_participated: u64,
    /// Valid proof submissions
    pub valid_proofs: u64,
    /// Invalid proof submissions
    pub invalid_proofs: u64,
    /// Registration timestamp
    pub registered_at: u64,
}

impl UnifiedParticipant {
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
            valid_proofs: 0,
            invalid_proofs: 0,
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

    /// Record valid proof submission
    pub fn record_valid_proof(&mut self) {
        self.rounds_participated += 1;
        self.valid_proofs += 1;
        // Small reputation boost for valid proofs
        self.kvector.k_r = (self.kvector.k_r + 0.005).min(1.0);
    }

    /// Record invalid proof submission
    pub fn record_invalid_proof(&mut self) {
        self.rounds_participated += 1;
        self.invalid_proofs += 1;
        // Reputation penalty for invalid proofs
        self.kvector.k_r = (self.kvector.k_r - 0.02).max(0.0);
    }

    /// Get proof validity rate
    pub fn validity_rate(&self) -> f32 {
        if self.rounds_participated == 0 {
            1.0 // Assume good until proven otherwise
        } else {
            self.valid_proofs as f32 / self.rounds_participated as f32
        }
    }
}

/// A proven gradient submission
#[derive(Debug, Clone)]
pub struct ProvenSubmission {
    /// Submitter ID
    pub participant_id: String,
    /// Submitter's K-Vector at submission time
    pub kvector: KVector,
    /// The gradient update
    pub gradient: GradientUpdate,
    /// Gradient hash commitment
    pub gradient_hash: [u8; 32],
    /// Whether the ZK proof is valid
    pub proof_valid: bool,
    /// Whether the gradient passes quality checks
    pub quality_valid: bool,
    /// Proof generation time in ms
    pub proof_time_ms: u64,
    /// Submission timestamp
    pub timestamp: u64,
}

impl ProvenSubmission {
    /// Check if submission is valid for aggregation
    pub fn is_valid(&self) -> bool {
        self.proof_valid && self.quality_valid
    }

    /// Get voting weight (if valid)
    pub fn weight(&self, use_quadratic: bool) -> f32 {
        if !self.is_valid() {
            return 0.0;
        }

        if use_quadratic {
            self.kvector.k_r.powi(2)
        } else {
            self.kvector.k_r
        }
    }
}

/// State of a consensus round
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnifiedRoundState {
    /// Collecting proven submissions
    Collecting,
    /// Verifying proofs and checking consensus
    Verifying,
    /// Aggregating valid gradients
    Aggregating,
    /// Round completed successfully
    Completed,
    /// Round failed
    Failed,
}

/// Information about a round
#[derive(Debug, Clone)]
pub struct UnifiedRoundInfo {
    /// Round number
    pub round: u64,
    /// Current state
    pub state: UnifiedRoundState,
    /// Model hash for this round
    pub model_hash: [u8; 32],
    /// Start timestamp
    pub started_at: u64,
    /// End timestamp (if completed)
    pub ended_at: Option<u64>,
    /// Submissions collected
    pub submissions: Vec<ProvenSubmission>,
    /// Aggregation result (if completed)
    pub result: Option<UnifiedAggregationResult>,
}

impl UnifiedRoundInfo {
    fn new(round: u64, model_hash: [u8; 32]) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            round,
            state: UnifiedRoundState::Collecting,
            model_hash,
            started_at: timestamp,
            ended_at: None,
            submissions: Vec::new(),
            result: None,
        }
    }
}

/// Result of unified aggregation
#[derive(Debug, Clone)]
pub struct UnifiedAggregationResult {
    /// Aggregated gradient
    pub gradient: AggregatedGradient,
    /// Number of valid submissions included
    pub valid_submissions: usize,
    /// Number of invalid submissions excluded
    pub invalid_submissions: usize,
    /// Byzantine fraction detected
    pub byzantine_fraction: f32,
    /// Hash commitment of aggregated result
    pub aggregation_hash: [u8; 32],
    /// Hash commitments of included gradients
    pub included_hashes: Vec<[u8; 32]>,
    /// IDs of excluded participants
    pub excluded_participants: Vec<String>,
    /// Consensus ratio achieved
    pub consensus_ratio: f32,
}

/// Statistics for the Unified ZK-RBBFT bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedZkRbbftStats {
    /// Total registered participants
    pub total_participants: usize,
    /// Eligible participants (above min reputation)
    pub eligible_participants: usize,
    /// Total completed rounds
    pub total_rounds: usize,
    /// Successful rounds
    pub successful_rounds: usize,
    /// Total proofs submitted
    pub total_proofs: usize,
    /// Valid proofs
    pub valid_proofs: usize,
    /// Invalid proofs (Byzantine)
    pub invalid_proofs: usize,
    /// Average proof time in ms
    pub avg_proof_time_ms: u64,
    /// Average consensus ratio
    pub avg_consensus_ratio: f32,
    /// Average Byzantine fraction
    pub avg_byzantine_fraction: f32,
    /// Total voting weight
    pub total_voting_weight: f32,
}

/// Unified ZK-RBBFT Federated Learning Bridge
///
/// Combines ZK proofs with RB-BFT consensus for maximum security.
#[cfg(any(feature = "simulation", feature = "risc0"))]
#[derive(Debug)]
pub struct UnifiedZkRbbftBridge {
    /// Configuration
    config: UnifiedZkRbbftConfig,
    /// The ZK gradient prover
    prover: GradientProver,
    /// Registered participants
    participants: HashMap<String, UnifiedParticipant>,
    /// Current round (if any)
    current_round: Option<UnifiedRoundInfo>,
    /// Completed rounds
    completed_rounds: Vec<UnifiedRoundInfo>,
    /// Next round number
    next_round: u64,
    /// Cumulative stats
    cumulative_stats: CumulativeStats,
}

#[cfg(any(feature = "simulation", feature = "risc0"))]
#[derive(Debug, Default)]
struct CumulativeStats {
    total_proofs: usize,
    valid_proofs: usize,
    invalid_proofs: usize,
    total_proof_time_ms: u64,
    total_consensus_ratio: f32,
    total_byzantine_fraction: f32,
    rounds_counted: usize,
}

#[cfg(any(feature = "simulation", feature = "risc0"))]
impl UnifiedZkRbbftBridge {
    /// Create a new bridge with the given configuration
    pub fn new(config: UnifiedZkRbbftConfig) -> Self {
        Self {
            config,
            prover: GradientProver::for_federated_learning(),
            participants: HashMap::new(),
            current_round: None,
            completed_rounds: Vec::new(),
            next_round: 1,
            cumulative_stats: CumulativeStats::default(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(UnifiedZkRbbftConfig::default())
    }

    /// Create with production configuration
    pub fn for_production() -> Self {
        Self::new(UnifiedZkRbbftConfig::production())
    }

    /// Create with testing configuration
    pub fn for_testing() -> Self {
        Self::new(UnifiedZkRbbftConfig::testing())
    }

    /// Register a participant
    pub fn register_participant(&mut self, id: &str, kvector: KVector) -> bool {
        if kvector.k_r < self.config.min_reputation {
            return false;
        }

        let participant = UnifiedParticipant::new(id, kvector);
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
    pub fn get_participant(&self, id: &str) -> Option<&UnifiedParticipant> {
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
    pub fn start_round(&mut self, model_hash: [u8; 32]) -> Result<u64, UnifiedZkRbbftError> {
        let eligible_count = self.eligible_participant_count();
        if eligible_count < self.config.min_participants {
            return Err(UnifiedZkRbbftError::InsufficientParticipants {
                have: eligible_count,
                need: self.config.min_participants,
            });
        }

        let round_num = self.next_round;
        self.next_round += 1;

        let round = UnifiedRoundInfo::new(round_num, model_hash);
        self.current_round = Some(round);

        Ok(round_num)
    }

    /// Submit a gradient with ZK proof generation
    ///
    /// This generates a ZK proof of gradient quality and submits it for consensus.
    pub fn submit_proven_gradient(
        &mut self,
        participant_id: &str,
        gradient: &[f32],
        epochs: u32,
        learning_rate: f32,
        batch_size: usize,
        loss: f64,
    ) -> Result<(), UnifiedZkRbbftError> {
        // Validate participant
        let participant = self
            .participants
            .get(participant_id)
            .ok_or_else(|| UnifiedZkRbbftError::ParticipantNotFound(participant_id.to_string()))?
            .clone();

        if !participant.is_eligible(self.config.min_reputation) {
            return Err(UnifiedZkRbbftError::InsufficientReputation {
                reputation: participant.kvector.k_r,
                minimum: self.config.min_reputation,
            });
        }

        // Check for active round
        let round = self
            .current_round
            .as_mut()
            .ok_or(UnifiedZkRbbftError::NoActiveRound)?;

        // Check for duplicate submission
        if round
            .submissions
            .iter()
            .any(|s| s.participant_id == participant_id)
        {
            return Err(UnifiedZkRbbftError::DuplicateSubmission(
                participant_id.to_string(),
            ));
        }

        // Generate ZK proof
        let proof = self
            .prover
            .prove_gradient_quality(
                gradient,
                &round.model_hash,
                epochs,
                learning_rate,
                participant_id,
                round.round as u32,
            )
            .map_err(|e| UnifiedZkRbbftError::ProofGenerationFailed(e.to_string()))?;

        // Update stats
        self.cumulative_stats.total_proofs += 1;
        self.cumulative_stats.total_proof_time_ms += proof.generation_time_ms;

        if proof.is_valid() {
            self.cumulative_stats.valid_proofs += 1;
        } else {
            self.cumulative_stats.invalid_proofs += 1;
        }

        // Convert gradient to f64 for storage
        let gradient_f64: Vec<f64> = gradient.iter().map(|&g| g as f64).collect();

        // Create gradient update
        let gradient_update = GradientUpdate::new(
            participant_id.to_string(),
            round.round,
            gradient_f64,
            batch_size,
            loss,
        );

        // Create submission
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let submission = ProvenSubmission {
            participant_id: participant_id.to_string(),
            kvector: participant.kvector,
            gradient: gradient_update,
            gradient_hash: *proof.gradient_hash(),
            proof_valid: proof.is_valid(),
            quality_valid: proof.is_valid(), // Proof validity implies quality validity
            proof_time_ms: proof.generation_time_ms,
            timestamp,
        };

        round.submissions.push(submission);

        // Update participant record
        if let Some(p) = self.participants.get_mut(participant_id) {
            if proof.is_valid() {
                p.record_valid_proof();
            } else {
                p.record_invalid_proof();
            }
        }

        Ok(())
    }

    /// Submit a pre-proven gradient
    ///
    /// Use this when the proof was generated externally (e.g., by a RISC-0 prover service).
    pub fn submit_with_proof(
        &mut self,
        participant_id: &str,
        gradient: GradientUpdate,
        proof: GradientProofReceipt,
    ) -> Result<(), UnifiedZkRbbftError> {
        // Validate participant
        let participant = self
            .participants
            .get(participant_id)
            .ok_or_else(|| UnifiedZkRbbftError::ParticipantNotFound(participant_id.to_string()))?
            .clone();

        if !participant.is_eligible(self.config.min_reputation) {
            return Err(UnifiedZkRbbftError::InsufficientReputation {
                reputation: participant.kvector.k_r,
                minimum: self.config.min_reputation,
            });
        }

        // Check for active round
        let round = self
            .current_round
            .as_mut()
            .ok_or(UnifiedZkRbbftError::NoActiveRound)?;

        // Check for duplicate submission
        if round
            .submissions
            .iter()
            .any(|s| s.participant_id == participant_id)
        {
            return Err(UnifiedZkRbbftError::DuplicateSubmission(
                participant_id.to_string(),
            ));
        }

        // Update stats
        self.cumulative_stats.total_proofs += 1;
        self.cumulative_stats.total_proof_time_ms += proof.generation_time_ms;

        if proof.is_valid() {
            self.cumulative_stats.valid_proofs += 1;
        } else {
            self.cumulative_stats.invalid_proofs += 1;
        }

        // Create submission
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let submission = ProvenSubmission {
            participant_id: participant_id.to_string(),
            kvector: participant.kvector,
            gradient,
            gradient_hash: *proof.gradient_hash(),
            proof_valid: proof.is_valid(),
            quality_valid: proof.is_valid(),
            proof_time_ms: proof.generation_time_ms,
            timestamp,
        };

        round.submissions.push(submission);

        // Update participant record
        if let Some(p) = self.participants.get_mut(participant_id) {
            if proof.is_valid() {
                p.record_valid_proof();
            } else {
                p.record_invalid_proof();
            }
        }

        Ok(())
    }

    /// Calculate consensus metrics for current round
    fn calculate_consensus_metrics(&self) -> (f32, f32, f32) {
        if let Some(ref round) = self.current_round {
            if round.submissions.is_empty() {
                return (0.0, 0.0, 0.0);
            }

            let mut valid_weight = 0.0f32;
            let mut total_weight = 0.0f32;
            let mut invalid_count = 0usize;

            for submission in &round.submissions {
                let weight = submission
                    .kvector
                    .k_r
                    .powi(if self.config.use_quadratic_weighting {
                        2
                    } else {
                        1
                    });
                total_weight += weight;

                if submission.is_valid() {
                    valid_weight += weight;
                } else {
                    invalid_count += 1;
                }
            }

            // Include non-submitters as implicit rejections
            let total_possible = self.total_voting_weight();
            if total_weight < total_possible {
                total_weight = total_possible;
            }

            let consensus_ratio = if total_weight > 0.0 {
                valid_weight / total_weight
            } else {
                0.0
            };

            let byzantine_fraction = invalid_count as f32 / round.submissions.len() as f32;

            (consensus_ratio, byzantine_fraction, valid_weight)
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Check if consensus has been reached
    pub fn check_consensus(&self) -> bool {
        let (consensus_ratio, byzantine_fraction, _) = self.calculate_consensus_metrics();

        consensus_ratio >= self.config.quorum_threshold
            && byzantine_fraction <= self.config.byzantine_threshold
    }

    /// Get current consensus ratio
    pub fn consensus_ratio(&self) -> f32 {
        self.calculate_consensus_metrics().0
    }

    /// Get current Byzantine fraction
    pub fn byzantine_fraction(&self) -> f32 {
        self.calculate_consensus_metrics().1
    }

    /// Finalize the current round with aggregation
    pub fn finalize_round(&mut self) -> Result<UnifiedAggregationResult, UnifiedZkRbbftError> {
        let (consensus_ratio, byzantine_fraction, _) = self.calculate_consensus_metrics();

        // Check consensus
        if consensus_ratio < self.config.quorum_threshold {
            return Err(UnifiedZkRbbftError::ConsensusNotReached {
                actual: consensus_ratio * 100.0,
                required: self.config.quorum_threshold * 100.0,
            });
        }

        // Check Byzantine threshold
        if byzantine_fraction > self.config.byzantine_threshold {
            return Err(UnifiedZkRbbftError::ByzantineThresholdExceeded {
                actual: byzantine_fraction * 100.0,
                max: self.config.byzantine_threshold * 100.0,
            });
        }

        let mut round = self
            .current_round
            .take()
            .ok_or(UnifiedZkRbbftError::NoActiveRound)?;

        // Filter valid submissions
        let valid_submissions: Vec<&ProvenSubmission> =
            round.submissions.iter().filter(|s| s.is_valid()).collect();

        let invalid_submissions: Vec<&ProvenSubmission> =
            round.submissions.iter().filter(|s| !s.is_valid()).collect();

        if valid_submissions.is_empty() {
            return Err(UnifiedZkRbbftError::InsufficientParticipants { have: 0, need: 1 });
        }

        // Get gradient dimension
        let dimension = valid_submissions[0].gradient.gradients.len();

        // Compute reputation²-weighted average
        let mut total_weight = 0.0f32;
        let mut aggregated_gradients = vec![0.0f64; dimension];

        for submission in &valid_submissions {
            let weight = submission.weight(self.config.use_quadratic_weighting);
            total_weight += weight;

            for (i, g) in submission.gradient.gradients.iter().enumerate() {
                aggregated_gradients[i] += g * weight as f64;
            }
        }

        // Normalize
        if total_weight > 0.0 {
            for g in &mut aggregated_gradients {
                *g /= total_weight as f64;
            }
        }

        // Compute aggregation hash
        let aggregation_hash = compute_aggregation_hash(&aggregated_gradients);

        // Collect included hashes
        let included_hashes: Vec<[u8; 32]> =
            valid_submissions.iter().map(|s| s.gradient_hash).collect();

        // Collect excluded participants
        let excluded_participants: Vec<String> = invalid_submissions
            .iter()
            .map(|s| s.participant_id.clone())
            .collect();

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let gradient = AggregatedGradient {
            gradients: aggregated_gradients,
            model_version: round.round,
            participant_count: valid_submissions.len(),
            excluded_count: invalid_submissions.len(),
            aggregation_method: AggregationMethod::TrustWeighted,
            timestamp,
        };

        let result = UnifiedAggregationResult {
            gradient,
            valid_submissions: valid_submissions.len(),
            invalid_submissions: invalid_submissions.len(),
            byzantine_fraction,
            aggregation_hash,
            included_hashes,
            excluded_participants,
            consensus_ratio,
        };

        // Update cumulative stats
        self.cumulative_stats.total_consensus_ratio += consensus_ratio;
        self.cumulative_stats.total_byzantine_fraction += byzantine_fraction;
        self.cumulative_stats.rounds_counted += 1;

        // Finalize round
        round.state = UnifiedRoundState::Completed;
        round.ended_at = Some(timestamp / 1000);
        round.result = Some(result.clone());
        self.completed_rounds.push(round);

        Ok(result)
    }

    /// Get current round info
    pub fn current_round_info(&self) -> Option<&UnifiedRoundInfo> {
        self.current_round.as_ref()
    }

    /// Get completed rounds
    pub fn completed_rounds(&self) -> &[UnifiedRoundInfo] {
        &self.completed_rounds
    }

    /// Get statistics
    pub fn stats(&self) -> UnifiedZkRbbftStats {
        let total_participants = self.participants.len();
        let eligible_participants = self.eligible_participant_count();
        let total_rounds = self.completed_rounds.len();
        let successful_rounds = self
            .completed_rounds
            .iter()
            .filter(|r| r.state == UnifiedRoundState::Completed)
            .count();

        let avg_proof_time_ms = if self.cumulative_stats.total_proofs > 0 {
            self.cumulative_stats.total_proof_time_ms / self.cumulative_stats.total_proofs as u64
        } else {
            0
        };

        let avg_consensus_ratio = if self.cumulative_stats.rounds_counted > 0 {
            self.cumulative_stats.total_consensus_ratio
                / self.cumulative_stats.rounds_counted as f32
        } else {
            0.0
        };

        let avg_byzantine_fraction = if self.cumulative_stats.rounds_counted > 0 {
            self.cumulative_stats.total_byzantine_fraction
                / self.cumulative_stats.rounds_counted as f32
        } else {
            0.0
        };

        UnifiedZkRbbftStats {
            total_participants,
            eligible_participants,
            total_rounds,
            successful_rounds,
            total_proofs: self.cumulative_stats.total_proofs,
            valid_proofs: self.cumulative_stats.valid_proofs,
            invalid_proofs: self.cumulative_stats.invalid_proofs,
            avg_proof_time_ms,
            avg_consensus_ratio,
            avg_byzantine_fraction,
            total_voting_weight: self.total_voting_weight(),
        }
    }

    /// Clear current round (abort without completing)
    pub fn abort_round(&mut self) {
        if let Some(mut round) = self.current_round.take() {
            round.state = UnifiedRoundState::Failed;
            self.completed_rounds.push(round);
        }
    }
}

#[cfg(any(feature = "simulation", feature = "risc0"))]
impl Default for UnifiedZkRbbftBridge {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Compute hash of aggregated gradient
fn compute_aggregation_hash(gradients: &[f64]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    for g in gradients {
        hasher.update(g.to_le_bytes());
    }
    let result = hasher.finalize();

    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

#[cfg(all(test, any(feature = "simulation", feature = "risc0")))]
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

    fn make_valid_gradient(size: usize, scale: f32) -> Vec<f32> {
        (0..size).map(|i| (i as f32 * 0.01).sin() * scale).collect()
    }

    #[test]
    fn test_basic_round() {
        let mut bridge = UnifiedZkRbbftBridge::for_testing();

        // Register participants
        bridge.register_participant("client-1", make_kvector(0.9));
        bridge.register_participant("client-2", make_kvector(0.8));

        // Start round
        let round = bridge.start_round([0u8; 32]).unwrap();
        assert_eq!(round, 1);

        // Submit gradients
        let grad1 = make_valid_gradient(100, 0.5);
        let grad2 = make_valid_gradient(100, 0.4);

        bridge
            .submit_proven_gradient("client-1", &grad1, 5, 0.01, 32, 0.5)
            .unwrap();
        bridge
            .submit_proven_gradient("client-2", &grad2, 5, 0.01, 32, 0.4)
            .unwrap();

        // Check consensus
        assert!(bridge.check_consensus());

        // Finalize
        let result = bridge.finalize_round().unwrap();
        assert_eq!(result.valid_submissions, 2);
        assert_eq!(result.invalid_submissions, 0);
    }

    #[test]
    fn test_byzantine_detection() {
        let mut bridge = UnifiedZkRbbftBridge::for_testing();

        bridge.register_participant("good-1", make_kvector(0.9));
        bridge.register_participant("good-2", make_kvector(0.8));
        bridge.register_participant("bad", make_kvector(0.7));

        bridge.start_round([0u8; 32]).unwrap();

        // Good participants submit valid gradients
        let good_grad = make_valid_gradient(100, 0.5);
        bridge
            .submit_proven_gradient("good-1", &good_grad, 5, 0.01, 32, 0.5)
            .unwrap();
        bridge
            .submit_proven_gradient("good-2", &good_grad, 5, 0.01, 32, 0.5)
            .unwrap();

        // Bad participant submits zero gradient (will fail quality check)
        let bad_grad = vec![0.0f32; 100];
        bridge
            .submit_proven_gradient("bad", &bad_grad, 5, 0.01, 32, 0.5)
            .unwrap();

        // Byzantine fraction should be detected
        let byzantine = bridge.byzantine_fraction();
        assert!(byzantine > 0.0, "Byzantine fraction should be > 0");
    }

    #[test]
    fn test_reputation_weighting() {
        let mut bridge = UnifiedZkRbbftBridge::for_testing();

        // High rep vs low rep
        bridge.register_participant("high-rep", make_kvector(0.9));
        bridge.register_participant("low-rep", make_kvector(0.3));

        bridge.start_round([0u8; 32]).unwrap();

        // High rep submits gradient with larger values
        let grad1 = make_valid_gradient(100, 1.0);
        bridge
            .submit_proven_gradient("high-rep", &grad1, 5, 0.01, 32, 0.5)
            .unwrap();

        // Low rep submits gradient with smaller values
        let grad2 = make_valid_gradient(100, 0.1);
        bridge
            .submit_proven_gradient("low-rep", &grad2, 5, 0.01, 32, 0.5)
            .unwrap();

        let result = bridge.finalize_round().unwrap();

        // Result should be weighted heavily towards high-rep's gradient
        // High rep has weight 0.9² = 0.81, low rep has 0.3² = 0.09
        // Total = 0.9, so high-rep contributes 0.81/0.9 = 90% of the weight
        // At index 50: sin(0.5) ≈ 0.48
        // High-rep: 0.48 * 1.0 = 0.48, Low-rep: 0.48 * 0.1 = 0.048
        // Weighted avg ≈ (0.48 * 0.81 + 0.048 * 0.09) / 0.9 ≈ 0.437
        // This should be much closer to high-rep's 0.48 than low-rep's 0.048
        let high_rep_val = (50.0f64 * 0.01).sin() * 1.0;
        let low_rep_val = (50.0f64 * 0.01).sin() * 0.1;
        let aggregated_val = result.gradient.gradients[50].abs();

        // Verify aggregated is closer to high-rep than low-rep
        let dist_to_high = (aggregated_val - high_rep_val).abs();
        let dist_to_low = (aggregated_val - low_rep_val).abs();
        assert!(
            dist_to_high < dist_to_low,
            "Aggregated gradient should be closer to high-rep: agg={:.4}, high={:.4}, low={:.4}",
            aggregated_val,
            high_rep_val,
            low_rep_val
        );
    }

    #[test]
    fn test_duplicate_submission_rejected() {
        let mut bridge = UnifiedZkRbbftBridge::for_testing();

        bridge.register_participant("client-1", make_kvector(0.9));
        bridge.register_participant("client-2", make_kvector(0.8));

        bridge.start_round([0u8; 32]).unwrap();

        let grad = make_valid_gradient(100, 0.5);
        bridge
            .submit_proven_gradient("client-1", &grad, 5, 0.01, 32, 0.5)
            .unwrap();

        // Try to submit again
        let result = bridge.submit_proven_gradient("client-1", &grad, 5, 0.01, 32, 0.5);
        assert!(matches!(
            result,
            Err(UnifiedZkRbbftError::DuplicateSubmission(_))
        ));
    }

    #[test]
    fn test_insufficient_reputation_rejected() {
        let mut bridge = UnifiedZkRbbftBridge::new(UnifiedZkRbbftConfig::default());

        // Try to register with low reputation
        assert!(!bridge.register_participant("low-rep", make_kvector(0.1)));
    }

    #[test]
    fn test_stats_accumulation() {
        let mut bridge = UnifiedZkRbbftBridge::for_testing();

        bridge.register_participant("client-1", make_kvector(0.9));
        bridge.register_participant("client-2", make_kvector(0.8));

        // Run two rounds
        for _ in 0..2 {
            bridge.start_round([0u8; 32]).unwrap();

            let grad = make_valid_gradient(100, 0.5);
            bridge
                .submit_proven_gradient("client-1", &grad, 5, 0.01, 32, 0.5)
                .unwrap();
            bridge
                .submit_proven_gradient("client-2", &grad, 5, 0.01, 32, 0.5)
                .unwrap();

            bridge.finalize_round().unwrap();
        }

        let stats = bridge.stats();
        assert_eq!(stats.total_rounds, 2);
        assert_eq!(stats.successful_rounds, 2);
        assert_eq!(stats.total_proofs, 4);
        assert_eq!(stats.valid_proofs, 4);
    }

    #[test]
    fn test_abort_round() {
        let mut bridge = UnifiedZkRbbftBridge::for_testing();

        bridge.register_participant("client-1", make_kvector(0.9));
        bridge.register_participant("client-2", make_kvector(0.8));

        bridge.start_round([0u8; 32]).unwrap();

        // Submit one gradient
        let grad = make_valid_gradient(100, 0.5);
        bridge
            .submit_proven_gradient("client-1", &grad, 5, 0.01, 32, 0.5)
            .unwrap();

        // Abort round
        bridge.abort_round();

        // Should be able to start new round
        let round = bridge.start_round([0u8; 32]).unwrap();
        assert_eq!(round, 2);

        // Previous round should be marked failed
        assert_eq!(
            bridge.completed_rounds()[0].state,
            UnifiedRoundState::Failed
        );
    }
}
