//! Governance Module for Federated Learning
//!
//! Provides governance primitives for FL operations, integrating with:
//! - Mycelix governance hApp (proposals, voting, execution)
//! - MATL trust scoring for vote weight
//! - Identity system for voter eligibility
//!
//! ## Proposal Types
//!
//! - **Standard (7 days)**: Regular FL parameter changes
//! - **Emergency (24 hours)**: Critical security fixes
//! - **Constitutional (30 days)**: Fundamental protocol changes
//! - **ModelGovernance**: FL-specific model decisions
//!
//! ## Vote Weight
//!
//! ```text
//! weight = MATL_score × stake × participation_bonus
//! where participation_bonus = 1.0 + 0.1 × recent_participation_rate
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use fl_aggregator::governance::{
//!     GovernanceClient, Proposal, ProposalType, Vote, VoteChoice,
//! };
//!
//! // Create proposal for FL parameter change
//! let proposal = Proposal::new(
//!     "MIP-0042",
//!     "Increase Byzantine tolerance",
//!     "Increase f from 1 to 2 for Krum defense",
//!     ProposalType::Standard,
//!     "did:mycelix:proposer",
//! );
//!
//! // Calculate vote weight with MATL integration
//! let weight = calculate_vote_weight(matl_score, stake, participation_rate);
//!
//! // Cast vote
//! let vote = Vote::new("MIP-0042", "did:mycelix:voter", VoteChoice::For, weight);
//! ```

pub mod types;
pub mod proposals;
pub mod voting;
pub mod delegation;
pub mod eligibility;
pub mod weight;

#[cfg(feature = "proofs")]
pub mod verified;

pub use types::*;
pub use proposals::*;
pub use voting::*;
pub use delegation::*;
pub use eligibility::*;
pub use weight::*;

#[cfg(feature = "proofs")]
pub use verified::{VerifiedVoteManager, VerifiedGovernanceConfig, VoteWithProof};

use thiserror::Error;

/// Governance error types
#[derive(Error, Debug, Clone)]
pub enum GovernanceError {
    #[error("Proposal not found: {0}")]
    ProposalNotFound(String),

    #[error("Invalid proposal: {0}")]
    InvalidProposal(String),

    #[error("Voting not active for proposal: {0}")]
    VotingNotActive(String),

    #[error("Already voted: {0}")]
    AlreadyVoted(String),

    #[error("Insufficient eligibility: {0}")]
    InsufficientEligibility(String),

    #[error("Delegation error: {0}")]
    DelegationError(String),

    #[error("Quorum not reached: required {required}, got {actual}")]
    QuorumNotReached { required: f32, actual: f32 },

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Timelock active: {0}")]
    TimelockActive(String),

    #[error("Guardian veto: {0}")]
    GuardianVeto(String),
}

pub type GovernanceResult<T> = Result<T, GovernanceError>;
