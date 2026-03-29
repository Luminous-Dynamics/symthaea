// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consensus Network Layer
//!
//! This module defines the networking abstractions for the RB-BFT consensus protocol.
//! It provides message types and transport traits that can be implemented with any
//! networking library (libp2p, QUIC, TCP, etc.).
//!
//! ## Architecture
//!
//! ```text
//!                    ┌──────────────────┐
//!                    │  Consensus Core  │
//!                    └────────┬─────────┘
//!                             │ ConsensusMessage
//!                    ┌────────▼─────────┐
//!                    │  NetworkService  │ (trait)
//!                    └────────┬─────────┘
//!                             │
//!           ┌─────────────────┼─────────────────┐
//!           │                 │                 │
//!     ┌─────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐
//!     │  libp2p   │    │   QUIC      │   │    gRPC     │
//!     └───────────┘    └─────────────┘   └─────────────┘
//! ```
//!
//! ## Message Types
//!
//! - **Proposal**: Block proposal from leader
//! - **Vote**: Validator vote on proposal
//! - **Commit**: Final commit notification
//! - **Challenge**: Byzantine behavior accusation
//! - **ViewChange**: Round advancement request

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;

use crate::error::{ConsensusError, ConsensusResult};
use crate::crypto::ConsensusSignature;
use crate::proposal::Proposal;

/// Peer identifier (32-byte public key)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PeerId(pub [u8; 32]);

impl PeerId {
    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get as hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Get short identifier (first 8 chars)
    pub fn short(&self) -> String {
        self.to_hex()[..8].to_string()
    }
}

impl std::fmt::Display for PeerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.short())
    }
}

/// Consensus message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// Proposal from leader
    Proposal(ProposalMessage),
    /// Vote on a proposal
    Vote(VoteMessage),
    /// Commit notification
    Commit(CommitMessage),
    /// View change request
    ViewChange(ViewChangeMessage),
    /// Challenge against Byzantine behavior
    Challenge(ChallengeMessage),
    /// VRF output for leader election
    VrfOutput(VrfMessage),
    /// Request state sync
    SyncRequest(SyncRequestMessage),
    /// State sync response
    SyncResponse(SyncResponseMessage),
}

impl ConsensusMessage {
    /// Get the message type as a string
    pub fn message_type(&self) -> &'static str {
        match self {
            ConsensusMessage::Proposal(_) => "proposal",
            ConsensusMessage::Vote(_) => "vote",
            ConsensusMessage::Commit(_) => "commit",
            ConsensusMessage::ViewChange(_) => "view_change",
            ConsensusMessage::Challenge(_) => "challenge",
            ConsensusMessage::VrfOutput(_) => "vrf_output",
            ConsensusMessage::SyncRequest(_) => "sync_request",
            ConsensusMessage::SyncResponse(_) => "sync_response",
        }
    }

    /// Get the round number (if applicable)
    pub fn round(&self) -> Option<u64> {
        match self {
            ConsensusMessage::Proposal(m) => Some(m.round),
            ConsensusMessage::Vote(m) => Some(m.round),
            ConsensusMessage::Commit(m) => Some(m.round),
            ConsensusMessage::ViewChange(m) => Some(m.new_round),
            ConsensusMessage::Challenge(m) => Some(m.round),
            ConsensusMessage::VrfOutput(m) => Some(m.round),
            ConsensusMessage::SyncRequest(m) => m.from_round,
            ConsensusMessage::SyncResponse(_) => None,
        }
    }
}

/// Proposal message from leader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalMessage {
    /// Round number
    pub round: u64,
    /// The proposal
    pub proposal: Proposal,
    /// Leader's signature
    pub signature: ConsensusSignature,
}

/// Vote message from validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteMessage {
    /// Round number
    pub round: u64,
    /// Proposal hash being voted on
    pub proposal_hash: String,
    /// Whether approving or rejecting
    pub approve: bool,
    /// Voter's signature
    pub signature: ConsensusSignature,
    /// Voter's reputation-weighted vote power
    pub vote_weight: f32,
}

/// Commit notification after consensus reached
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitMessage {
    /// Round number
    pub round: u64,
    /// Committed proposal hash
    pub proposal_hash: String,
    /// Aggregated signatures from quorum
    pub aggregate_signature: Option<Vec<u8>>,
    /// Number of validators in quorum
    pub quorum_size: usize,
    /// Total vote weight of quorum
    pub quorum_weight: f32,
}

/// View change request to advance to next round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewChangeMessage {
    /// Current round
    pub current_round: u64,
    /// Requested new round
    pub new_round: u64,
    /// Reason for view change
    pub reason: ViewChangeReason,
    /// Requester's signature
    pub signature: ConsensusSignature,
}

/// Reasons for view change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViewChangeReason {
    /// Round timeout
    Timeout,
    /// Leader failure
    LeaderFailure,
    /// Invalid proposal
    InvalidProposal,
    /// Consensus not reached
    NoConsensus,
    /// Manual intervention
    Manual,
}

/// Challenge against Byzantine behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeMessage {
    /// Round where violation occurred
    pub round: u64,
    /// Accused validator
    pub accused: PeerId,
    /// Type of violation
    pub violation: ViolationType,
    /// Evidence (signatures, messages, etc.)
    pub evidence: Vec<u8>,
    /// Challenger's signature
    pub signature: ConsensusSignature,
}

/// Types of Byzantine violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Double voting in same round
    DoubleVote,
    /// Multiple proposals in same round
    DoublePropose,
    /// Invalid signature
    InvalidSignature,
    /// Invalid VRF proof
    InvalidVrf,
    /// Equivocation
    Equivocation,
}

/// VRF output for leader election
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrfMessage {
    /// Round number
    pub round: u64,
    /// VRF output (32 bytes)
    pub output: [u8; 32],
    /// VRF proof (64 bytes)
    pub proof: Vec<u8>,
    /// Validator's public key
    pub validator: PeerId,
}

/// State sync request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequestMessage {
    /// Request specific rounds
    pub from_round: Option<u64>,
    /// To round (inclusive)
    pub to_round: Option<u64>,
    /// Request validator set
    pub include_validators: bool,
}

/// State sync response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResponseMessage {
    /// Latest committed round
    pub latest_round: u64,
    /// Committed proposals (if requested)
    pub proposals: Vec<Proposal>,
    /// Current validator set (if requested)
    pub validators: Option<Vec<ValidatorInfo>>,
}

/// Validator information for sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    /// Validator's peer ID
    pub peer_id: PeerId,
    /// Current reputation
    pub reputation: f32,
    /// Stake amount
    pub stake: u64,
    /// Whether currently active
    pub active: bool,
}

/// Subscription to consensus events
#[derive(Debug, Clone)]
pub enum SubscriptionFilter {
    /// All messages
    All,
    /// Only messages for specific round
    Round(u64),
    /// Only specific message types
    Types(HashSet<String>),
    /// Only from specific peers
    Peers(HashSet<PeerId>),
}

/// Network service trait for consensus
///
/// Implementations can use any transport (libp2p, QUIC, gRPC, etc.)
pub trait NetworkService: Send + Sync {
    /// Get our own peer ID
    fn local_peer_id(&self) -> PeerId;

    /// Broadcast a message to all connected peers
    fn broadcast(&self, message: ConsensusMessage) -> Pin<Box<dyn Future<Output = ConsensusResult<()>> + Send + '_>>;

    /// Send a message to a specific peer
    fn send(&self, peer: &PeerId, message: ConsensusMessage) -> Pin<Box<dyn Future<Output = ConsensusResult<()>> + Send + '_>>;

    /// Send to multiple specific peers
    fn send_many(&self, peers: &[PeerId], message: ConsensusMessage) -> Pin<Box<dyn Future<Output = ConsensusResult<()>> + Send + '_>>;

    /// Get list of connected peers
    fn connected_peers(&self) -> Pin<Box<dyn Future<Output = Vec<PeerId>> + Send + '_>>;

    /// Check if a peer is connected
    fn is_connected(&self, peer: &PeerId) -> bool;

    /// Get the number of connected peers
    fn peer_count(&self) -> usize;
}

/// Received message with metadata
#[derive(Debug, Clone)]
pub struct ReceivedMessage {
    /// The sender
    pub from: PeerId,
    /// The message
    pub message: ConsensusMessage,
    /// When received (unix timestamp ms)
    pub received_at: u64,
}

/// Message receiver trait
pub trait MessageReceiver: Send + Sync {
    /// Receive the next message (blocking)
    fn recv(&self) -> Pin<Box<dyn Future<Output = Option<ReceivedMessage>> + Send + '_>>;

    /// Try to receive without blocking
    fn try_recv(&self) -> Option<ReceivedMessage>;
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Listen addresses
    pub listen_addrs: Vec<String>,
    /// Bootstrap peers
    pub bootstrap_peers: Vec<String>,
    /// Maximum connected peers
    pub max_peers: usize,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Message retry count
    pub retry_count: u32,
    /// Gossip topic for consensus messages
    pub gossip_topic: String,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addrs: vec!["/ip4/0.0.0.0/tcp/30333".to_string()],
            bootstrap_peers: Vec::new(),
            max_peers: 50,
            connection_timeout_ms: 10_000,
            retry_count: 3,
            gossip_topic: "mycelix-consensus-v1".to_string(),
        }
    }
}

/// Network statistics
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Current connected peers
    pub connected_peers: usize,
    /// Message send failures
    pub send_failures: u64,
}

impl NetworkStats {
    /// Record a sent message
    pub fn record_sent(&mut self, size: usize) {
        self.messages_sent += 1;
        self.bytes_sent += size as u64;
    }

    /// Record a received message
    pub fn record_received(&mut self, size: usize) {
        self.messages_received += 1;
        self.bytes_received += size as u64;
    }

    /// Record a send failure
    pub fn record_failure(&mut self) {
        self.send_failures += 1;
    }
}

/// Serialize a consensus message for network transmission
pub fn serialize_message(msg: &ConsensusMessage) -> ConsensusResult<Vec<u8>> {
    serde_json::to_vec(msg).map_err(|e| ConsensusError::SerializationError(e.to_string()))
}

/// Deserialize a consensus message from bytes
pub fn deserialize_message(bytes: &[u8]) -> ConsensusResult<ConsensusMessage> {
    serde_json::from_slice(bytes).map_err(|e| ConsensusError::SerializationError(e.to_string()))
}

/// Message validation result
#[derive(Debug, Clone)]
pub enum ValidationResult {
    /// Message is valid
    Valid,
    /// Message is invalid
    Invalid(String),
    /// Message needs more information to validate
    Pending(String),
}

/// Validate a received message
pub fn validate_message(msg: &ConsensusMessage, current_round: u64) -> ValidationResult {
    // Check round is not too old or too far in future
    if let Some(round) = msg.round() {
        if round + 10 < current_round {
            return ValidationResult::Invalid(format!(
                "Message round {} is too old (current: {})",
                round, current_round
            ));
        }
        if round > current_round + 5 {
            return ValidationResult::Pending(format!(
                "Message round {} is too far in future (current: {})",
                round, current_round
            ));
        }
    }

    // Message-specific validation can be added here
    ValidationResult::Valid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_id() {
        let bytes = [1u8; 32];
        let peer_id = PeerId::from_bytes(bytes);

        assert_eq!(peer_id.short().len(), 8);
        assert_eq!(peer_id.to_hex().len(), 64);
    }

    #[test]
    fn test_message_serialization() {
        let vote_msg = VoteMessage {
            round: 42,
            proposal_hash: "abc123".to_string(),
            approve: true,
            signature: ConsensusSignature {
                signature: vec![1, 2, 3],
                signer_pubkey: vec![4, 5, 6],
            },
            vote_weight: 0.8,
        };

        let msg = ConsensusMessage::Vote(vote_msg);

        let bytes = serialize_message(&msg).unwrap();
        let restored = deserialize_message(&bytes).unwrap();

        assert_eq!(msg.message_type(), restored.message_type());
        assert_eq!(msg.round(), restored.round());
    }

    #[test]
    fn test_message_validation() {
        let current_round = 100;

        // Valid message (current round)
        let msg = ConsensusMessage::Vote(VoteMessage {
            round: 100,
            proposal_hash: "test".to_string(),
            approve: true,
            signature: ConsensusSignature {
                signature: vec![],
                signer_pubkey: vec![],
            },
            vote_weight: 1.0,
        });

        assert!(matches!(validate_message(&msg, current_round), ValidationResult::Valid));

        // Old message
        let old_msg = ConsensusMessage::Vote(VoteMessage {
            round: 80,
            proposal_hash: "test".to_string(),
            approve: true,
            signature: ConsensusSignature {
                signature: vec![],
                signer_pubkey: vec![],
            },
            vote_weight: 1.0,
        });

        assert!(matches!(validate_message(&old_msg, current_round), ValidationResult::Invalid(_)));

        // Future message
        let future_msg = ConsensusMessage::Vote(VoteMessage {
            round: 110,
            proposal_hash: "test".to_string(),
            approve: true,
            signature: ConsensusSignature {
                signature: vec![],
                signer_pubkey: vec![],
            },
            vote_weight: 1.0,
        });

        assert!(matches!(validate_message(&future_msg, current_round), ValidationResult::Pending(_)));
    }

    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();

        assert_eq!(config.max_peers, 50);
        assert_eq!(config.retry_count, 3);
        assert!(!config.gossip_topic.is_empty());
    }

    #[test]
    fn test_network_stats() {
        let mut stats = NetworkStats::default();

        stats.record_sent(100);
        stats.record_sent(200);
        stats.record_received(150);
        stats.record_failure();

        assert_eq!(stats.messages_sent, 2);
        assert_eq!(stats.bytes_sent, 300);
        assert_eq!(stats.messages_received, 1);
        assert_eq!(stats.bytes_received, 150);
        assert_eq!(stats.send_failures, 1);
    }
}
