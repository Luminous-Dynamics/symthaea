// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Unified Governance Infrastructure
//!
//! A revolutionary governance framework that breaks the 33% BFT barrier through
//! reputation-weighted voting and enables seamless cross-hApp coordination.
//!
//! ## Key Features
//!
//! - **Reputation-Weighted Voting**: Achieves ~45% BFT tolerance
//! - **Hierarchical Federation**: Local → Regional → Global proposal escalation
//! - **Cross-hApp Reputation**: Unified trust propagation across ecosystem
//! - **Dynamic Quorum**: Adaptive participation requirements
//! - **Epistemic Validation**: Truth confidence scoring (E0-E4)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │         Mycelix Governance Infrastructure           │
//! ├─────────────────────────────────────────────────────┤
//! │                                                     │
//! │  Types          Validation       Aggregation       │
//! │  ├─ Proposal    ├─ Entry        ├─ Reputation      │
//! │  ├─ Vote        ├─ Link         ├─ Vote Tallies    │
//! │  ├─ Epistemic   ├─ Consensus    └─ Dynamic Quorum  │
//! │  └─ Params      └─ Boundaries                      │
//! │                                                     │
//! │  Execution           Sync                           │
//! │  ├─ Escalation      ├─ Cross-hApp                  │
//! │  ├─ Voting          ├─ Reputation                  │
//! │  └─ Resolution      └─ Events                      │
//! │                                                     │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use mycelix_governance::{HierarchicalProposal, ProposalCategory, ProposalType, EpistemicTier};
//!
//! // Create a proposal with epistemic tier validation
//! let proposal = HierarchicalProposal::new(
//!     "prop_001".to_string(),
//!     "Upgrade aggregation algorithm".to_string(),
//!     "Switch from FedAvg to FedProx for better convergence".to_string(),
//!     "agent_pubkey_123".to_string(),
//!     ProposalCategory::Technical,
//!     ProposalType::Normal,
//! );
//!
//! // Check epistemic tier requirements
//! assert_eq!(proposal.epistemic_tier, EpistemicTier::E1PersonalTestimony);
//! assert!(proposal.epistemic_tier.requires_peer_review() == false);
//!
//! // Higher tier proposals require peer review
//! let e2_tier = EpistemicTier::E2PeerVerified;
//! assert!(e2_tier.requires_peer_review());
//! ```
//!
//! ## Integration with 0TML
//!
//! This governance infrastructure builds directly on the trust algorithms from
//! Mycelix 0TML (Zero-Trust Machine Learning):
//!
//! - **PoGQ (Proof of Quality)**: 45% BFT tolerance in federated learning
//! - **TCDM (Temporal Consistency Detection)**: Behavioral consistency measurement
//! - **Entropy Score**: Behavioral predictability over time
//!
//! ## Breaking the 33% BFT Barrier
//!
//! Traditional DAOs fail at 34% malicious participation because they treat all votes equally.
//! Our reputation-weighted approach raises this to ~45%:
//!
//! ```text
//! Traditional DAO:           Reputation-Weighted DAO:
//! 1 account = 1 vote         1 account = f(reputation) votes
//! 34% malicious → failure    Need 50%+ reputation → ~45% BFT
//! Sybil: 1000 accounts ✗     Sybil: 0 reputation = 0 influence ✓
//! ```

// Public API
pub mod types;
pub mod validation;
pub mod aggregation;
pub mod execution;

#[cfg(feature = "cross_happ")]
pub mod sync;

// Re-exports for convenience
pub use types::{
    HierarchicalProposal,
    ReputationVote,
    ProposalScope,
    ProposalCategory,
    ProposalType,
    VoteChoice,
    EpistemicTier,
    GovernanceParams,
    QuadraticVote,
    QuadraticVotingParams,
    ConvictionVote,
    ConvictionParameters,
    ConvictionDecay,
};

// Phase 2B exports - Dynamic Quorum ✅
pub use aggregation::{
    DynamicQuorumCalculator,
    QuorumStatus,
};

// Phase 2C exports - Composable Vote Modifiers ✅
pub use types::{
    UnifiedVote,
    UnifiedVoteBuilder,
    QuadraticModifier,
    ConvictionModifier,
    DelegationModifier,
    DelegationScope,
};

// Future Phase 2B exports
// pub use aggregation::{
//     CompositeReputationCalculator,
//     VoteTallyAggregator,
// };

// #[cfg(feature = "reputation")]
// pub use aggregation::reputation::{
//     calculate_pogq,
//     calculate_tcdm,
//     calculate_entropy,
// };

// pub use execution::{
//     EscalationEngine,
//     ProposalExecutor,
// };

/// Result type for governance operations
pub type GovernanceResult<T> = Result<T, GovernanceError>;

/// Errors that can occur in governance operations
#[derive(Debug, thiserror::Error)]
pub enum GovernanceError {
    #[error("Insufficient reputation: required {required}, agent has {actual}")]
    InsufficientReputation { required: f64, actual: f64 },

    #[error("Proposal already exists: {0}")]
    DuplicateProposal(String),

    #[error("Agent already voted on this proposal")]
    DuplicateVote,

    #[error("Voting deadline has passed")]
    VotingClosed,

    #[error("Quorum not met: {actual}/{required}")]
    QuorumNotMet { actual: f64, required: f64 },

    #[error("Invalid proposal scope transition: {from:?} -> {to:?}")]
    InvalidEscalation { from: ProposalScope, to: ProposalScope },

    #[error("Cross-hApp operation failed: {0}")]
    CrossHAppError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Holochain error: {0}")]
    HolochainError(String),
}

impl From<serde_json::Error> for GovernanceError {
    fn from(err: serde_json::Error) -> Self {
        GovernanceError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_governance_imports() {
        // Verify all core types are accessible
        let _ = ProposalScope::Local;
        let _ = EpistemicTier::E0Null;
        let _ = VoteChoice::For;
    }
}
