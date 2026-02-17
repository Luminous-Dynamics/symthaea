//! Core types for unified governance infrastructure

pub mod proposal;
pub mod vote;
pub mod epistemic;
pub mod governance_params;
pub mod quadratic_vote;
pub mod conviction_vote;
pub mod unified_vote;  // Phase 2C: Composable Vote Modifiers ✨

// Re-exports for convenience
pub use proposal::{HierarchicalProposal, ProposalScope, ProposalCategory, ProposalType, ProposalStatus};
pub use vote::{ReputationVote, VoteChoice};
pub use epistemic::EpistemicTier;
pub use governance_params::GovernanceParams;
pub use quadratic_vote::{QuadraticVote, QuadraticVotingParams};
pub use conviction_vote::{ConvictionVote, ConvictionParameters, ConvictionDecay};

// Phase 2C: Unified composable voting system ✨
pub use unified_vote::{
    UnifiedVote,
    UnifiedVoteBuilder,
    QuadraticModifier,
    ConvictionModifier,
    DelegationModifier,
    DelegationScope,
};
