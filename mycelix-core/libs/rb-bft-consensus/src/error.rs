// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error types for RB-BFT consensus

use thiserror::Error;

/// Errors that can occur during consensus operations
#[derive(Error, Debug, Clone)]
pub enum ConsensusError {
    /// Not enough validators to reach consensus
    #[error("Insufficient validators: have {have}, need {need}")]
    InsufficientValidators { have: usize, need: usize },

    /// Validator reputation too low to participate
    #[error("Reputation too low: {reputation:.3} < {minimum:.3}")]
    ReputationTooLow { reputation: f32, minimum: f32 },

    /// Round timeout exceeded
    #[error("Round {round} timed out after {elapsed_ms}ms")]
    RoundTimeout { round: u64, elapsed_ms: u64 },

    /// Invalid proposal
    #[error("Invalid proposal: {reason}")]
    InvalidProposal { reason: String },

    /// Invalid vote
    #[error("Invalid vote from {validator}: {reason}")]
    InvalidVote { validator: String, reason: String },

    /// Duplicate vote detected
    #[error("Duplicate vote from {validator} in round {round}")]
    DuplicateVote { validator: String, round: u64 },

    /// Byzantine behavior detected
    #[error("Byzantine behavior detected from {validator}: {evidence}")]
    ByzantineBehavior { validator: String, evidence: String },

    /// Consensus not reached
    #[error("Consensus not reached: weighted votes {weighted_votes:.3} < threshold {threshold:.3}")]
    ConsensusNotReached { weighted_votes: f32, threshold: f32 },

    /// Invalid round state
    #[error("Invalid round state: expected {expected}, got {actual}")]
    InvalidRoundState { expected: String, actual: String },

    /// Validator not found
    #[error("Validator not found: {validator_id}")]
    ValidatorNotFound { validator_id: String },

    /// Proposal already exists for round
    #[error("Proposal already exists for round {round}")]
    ProposalExists { round: u64 },

    /// Not the designated leader
    #[error("Not the leader for round {round}: leader is {leader}")]
    NotLeader { round: u64, leader: String },

    /// Signature verification failed (legacy)
    #[error("Signature verification failed: {reason}")]
    SignatureInvalid { reason: String },

    /// Invalid cryptographic signature
    #[error("Invalid signature: {reason}")]
    InvalidSignature { reason: String },

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Cryptographic operation failed
    #[error("Crypto error: {0}")]
    CryptoError(String),

    /// Invalid configuration
    #[error("Invalid configuration: {reason}")]
    InvalidConfiguration { reason: String },

    /// Internal error
    #[error("Internal consensus error: {0}")]
    Internal(String),
}

/// Result type for consensus operations
pub type ConsensusResult<T> = Result<T, ConsensusError>;
