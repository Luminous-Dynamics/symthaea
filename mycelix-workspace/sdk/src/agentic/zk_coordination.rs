//! # ZK-Proven Coordination
//!
//! Integration of zero-knowledge proofs with the coordination module.
//! Enables agents to participate in groups without revealing their actual trust scores.
//!
//! ## Features
//!
//! - **Private Membership**: Join groups by proving trust threshold, not revealing score
//! - **Anonymous Voting**: Vote with proven weight without exposing K-Vector
//! - **Verifiable Improvement**: Prove trust improved without showing trajectory
//!
//! ## Example
//!
//! ```rust,ignore
//! use mycelix_sdk::agentic::zk_coordination::{
//!     ZKAgentGroup, MembershipProof, VoteProof,
//! };
//!
//! // Create ZK-enabled group
//! let mut group = ZKAgentGroup::new(ZKCoordinationConfig::default());
//!
//! // Join with proof (doesn't reveal actual trust score)
//! let proof = prover.prove_trust_threshold(&kvector, 0.5);
//! group.add_member_with_proof("agent-1", proof)?;
//!
//! // Vote with ZK proof
//! let vote_proof = prover.prove_vote_weight(&kvector, VoteType::Approve);
//! group.vote_with_proof("agent-1", &proposal_id, vote_proof)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::coordination::{
    ConsensusDecision, ConsensusResult, CoordinationConfig, Proposal, Vote, VoteType,
};
use super::zk_trust::{KVectorCommitment, ProofError, ProofStatement, TrustProof, TrustProver};
use crate::matl::KVector;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for ZK-enabled coordination
#[derive(Debug, Clone)]
pub struct ZKCoordinationConfig {
    /// Base coordination config
    pub coordination: CoordinationConfig,
    /// Require ZK proofs for membership
    pub require_membership_proof: bool,
    /// Require ZK proofs for voting
    pub require_vote_proof: bool,
    /// Allow mixed mode (some members with proof, some without)
    pub allow_mixed_mode: bool,
    /// Minimum proof freshness (ms) - older proofs are rejected
    pub max_proof_age_ms: u64,
}

impl Default for ZKCoordinationConfig {
    fn default() -> Self {
        Self {
            coordination: CoordinationConfig::default(),
            require_membership_proof: true,
            require_vote_proof: true,
            allow_mixed_mode: false,
            max_proof_age_ms: 3600_000, // 1 hour
        }
    }
}

// ============================================================================
// Membership Proof
// ============================================================================

/// Proof of eligibility to join a group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipProof {
    /// The ZK trust proof
    pub trust_proof: TrustProof,
    /// Claimed minimum trust (what was proven)
    pub min_trust: f64,
    /// Agent's commitment to their K-Vector
    pub commitment: KVectorCommitment,
}

impl MembershipProof {
    /// Create a membership proof
    pub fn create(
        prover: &TrustProver,
        kvector: &KVector,
        min_trust: f64,
        blinding: &[u8; 32],
    ) -> Result<Self, ProofError> {
        let statement = ProofStatement::trust_above(min_trust as f32);
        let trust_proof = prover.prove(kvector, &statement, blinding)?;
        let commitment = prover.commit(kvector, blinding);

        Ok(Self {
            trust_proof,
            min_trust,
            commitment,
        })
    }

    /// Verify the membership proof
    pub fn verify(&self, required_trust: f64) -> bool {
        // Check the proof proves sufficient trust
        if self.min_trust < required_trust {
            return false;
        }

        // In simulation mode, check the result
        if let Some(result) = self.trust_proof.get_result() {
            return result;
        }

        // Real ZK verification would happen here
        true
    }

    /// Check if proof is fresh enough
    pub fn is_fresh(&self, max_age_ms: u64, current_time: u64) -> bool {
        current_time.saturating_sub(self.trust_proof.created_at) <= max_age_ms
    }
}

// ============================================================================
// Vote Proof
// ============================================================================

/// Zero-knowledge proof for a vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteProof {
    /// The vote being cast
    pub vote_type: VoteType,
    /// Proof of trust (for weight calculation)
    pub trust_proof: TrustProof,
    /// Proven minimum trust (vote weight is derived from this)
    pub proven_trust: f64,
    /// Commitment to K-Vector at time of vote
    pub commitment: KVectorCommitment,
    /// Timestamp of vote
    pub timestamp: u64,
}

impl VoteProof {
    /// Create a vote proof
    pub fn create(
        prover: &TrustProver,
        kvector: &KVector,
        vote_type: VoteType,
        blinding: &[u8; 32],
    ) -> Result<Self, ProofError> {
        // Compute actual trust score for the proof
        let trust_score = compute_trust_score(kvector);

        // Prove trust is at least this value
        let statement = ProofStatement::trust_above(trust_score as f32);
        let trust_proof = prover.prove(kvector, &statement, blinding)?;
        let commitment = prover.commit(kvector, blinding);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Ok(Self {
            vote_type,
            trust_proof,
            proven_trust: trust_score,
            commitment,
            timestamp: now,
        })
    }

    /// Get the vote weight (based on proven trust)
    pub fn weight(&self, quadratic: bool) -> f64 {
        if quadratic {
            self.proven_trust.sqrt()
        } else {
            self.proven_trust
        }
    }

    /// Verify the vote proof
    pub fn verify(&self) -> bool {
        if let Some(result) = self.trust_proof.get_result() {
            return result;
        }
        true
    }
}

// ============================================================================
// ZK Member
// ============================================================================

/// A group member with ZK credentials
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ZKMember {
    /// Agent ID
    agent_id: String,
    /// Membership proof (None if joined without proof)
    membership_proof: Option<MembershipProof>,
    /// Proven minimum trust
    proven_trust: f64,
    /// Join timestamp
    joined_at: u64,
    /// Last commitment
    last_commitment: Option<KVectorCommitment>,
    /// Participation count
    participation_count: u64,
}

// ============================================================================
// ZK Proposal State
// ============================================================================

/// Proposal state with ZK votes
#[derive(Debug, Clone)]
struct ZKProposalState {
    /// The proposal
    proposal: Proposal,
    /// ZK votes (agent_id -> VoteProof)
    zk_votes: HashMap<String, VoteProof>,
    /// Regular votes (for mixed mode)
    regular_votes: HashMap<String, Vote>,
    /// Whether finalized
    finalized: bool,
    /// Result if finalized
    result: Option<ConsensusResult>,
}

// ============================================================================
// ZK Agent Group
// ============================================================================

/// A coordination group with ZK proof support
pub struct ZKAgentGroup {
    /// Group ID
    id: String,
    /// Configuration
    config: ZKCoordinationConfig,
    /// Members with ZK credentials
    members: HashMap<String, ZKMember>,
    /// Proposals
    proposals: HashMap<String, ZKProposalState>,
    /// Total proven trust
    total_proven_trust: f64,
}

impl ZKAgentGroup {
    /// Create a new ZK-enabled group
    pub fn new(config: ZKCoordinationConfig) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: format!("zk-group-{}", now),
            config,
            members: HashMap::new(),
            proposals: HashMap::new(),
            total_proven_trust: 0.0,
        }
    }

    /// Get group ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get member count
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    // -------------------------------------------------------------------------
    // Membership with ZK Proofs
    // -------------------------------------------------------------------------

    /// Add a member with ZK membership proof
    pub fn add_member_with_proof(
        &mut self,
        agent_id: &str,
        proof: MembershipProof,
    ) -> Result<(), ZKCoordinationError> {
        let now = Self::now();

        // Check proof freshness
        if !proof.is_fresh(self.config.max_proof_age_ms, now) {
            return Err(ZKCoordinationError::ProofExpired {
                agent_id: agent_id.to_string(),
            });
        }

        // Verify proof
        if !proof.verify(self.config.coordination.min_trust_threshold) {
            return Err(ZKCoordinationError::ProofVerificationFailed {
                agent_id: agent_id.to_string(),
            });
        }

        // Check group size
        if self.members.len() >= self.config.coordination.max_group_size {
            return Err(ZKCoordinationError::GroupFull);
        }

        // Check if already member
        if self.members.contains_key(agent_id) {
            return Err(ZKCoordinationError::AlreadyMember {
                agent_id: agent_id.to_string(),
            });
        }

        let member = ZKMember {
            agent_id: agent_id.to_string(),
            membership_proof: Some(proof.clone()),
            proven_trust: proof.min_trust,
            joined_at: now,
            last_commitment: Some(proof.commitment),
            participation_count: 0,
        };

        self.total_proven_trust += proof.min_trust;
        self.members.insert(agent_id.to_string(), member);

        Ok(())
    }

    /// Add a member without proof (only if mixed mode allowed)
    pub fn add_member_without_proof(
        &mut self,
        agent_id: &str,
        trust: f64,
    ) -> Result<(), ZKCoordinationError> {
        if self.config.require_membership_proof && !self.config.allow_mixed_mode {
            return Err(ZKCoordinationError::ProofRequired);
        }

        if trust < self.config.coordination.min_trust_threshold {
            return Err(ZKCoordinationError::InsufficientTrust {
                agent_id: agent_id.to_string(),
                trust,
                required: self.config.coordination.min_trust_threshold,
            });
        }

        if self.members.len() >= self.config.coordination.max_group_size {
            return Err(ZKCoordinationError::GroupFull);
        }

        if self.members.contains_key(agent_id) {
            return Err(ZKCoordinationError::AlreadyMember {
                agent_id: agent_id.to_string(),
            });
        }

        let member = ZKMember {
            agent_id: agent_id.to_string(),
            membership_proof: None,
            proven_trust: trust,
            joined_at: Self::now(),
            last_commitment: None,
            participation_count: 0,
        };

        self.total_proven_trust += trust;
        self.members.insert(agent_id.to_string(), member);

        Ok(())
    }

    /// Remove a member
    pub fn remove_member(&mut self, agent_id: &str) -> bool {
        if let Some(member) = self.members.remove(agent_id) {
            self.total_proven_trust -= member.proven_trust;
            true
        } else {
            false
        }
    }

    /// Check if agent has valid membership proof
    pub fn has_valid_proof(&self, agent_id: &str) -> bool {
        self.members
            .get(agent_id)
            .map(|m| m.membership_proof.is_some())
            .unwrap_or(false)
    }

    // -------------------------------------------------------------------------
    // Proposals
    // -------------------------------------------------------------------------

    /// Submit a proposal
    pub fn submit_proposal(&mut self, proposal: Proposal) -> String {
        let id = proposal.id().to_string();

        let state = ZKProposalState {
            proposal,
            zk_votes: HashMap::new(),
            regular_votes: HashMap::new(),
            finalized: false,
            result: None,
        };

        self.proposals.insert(id.clone(), state);
        id
    }

    // -------------------------------------------------------------------------
    // Voting with ZK Proofs
    // -------------------------------------------------------------------------

    /// Cast a vote with ZK proof
    pub fn vote_with_proof(
        &mut self,
        agent_id: &str,
        proposal_id: &str,
        proof: VoteProof,
    ) -> Result<(), ZKCoordinationError> {
        // Check membership
        if !self.members.contains_key(agent_id) {
            return Err(ZKCoordinationError::NotMember {
                agent_id: agent_id.to_string(),
            });
        }

        // Check proposal exists
        let state = self.proposals.get_mut(proposal_id).ok_or_else(|| {
            ZKCoordinationError::ProposalNotFound {
                proposal_id: proposal_id.to_string(),
            }
        })?;

        // Check not finalized
        if state.finalized {
            return Err(ZKCoordinationError::ProposalFinalized {
                proposal_id: proposal_id.to_string(),
            });
        }

        // Verify proof
        if !proof.verify() {
            return Err(ZKCoordinationError::ProofVerificationFailed {
                agent_id: agent_id.to_string(),
            });
        }

        // Store the ZK vote
        state.zk_votes.insert(agent_id.to_string(), proof);

        // Update member's last commitment
        if let Some(member) = self.members.get_mut(agent_id) {
            if let Some(vote) = state.zk_votes.get(agent_id) {
                member.last_commitment = Some(vote.commitment.clone());
            }
        }

        Ok(())
    }

    /// Cast a regular vote (only if mixed mode)
    pub fn vote_without_proof(
        &mut self,
        agent_id: &str,
        proposal_id: &str,
        vote_type: VoteType,
    ) -> Result<(), ZKCoordinationError> {
        if self.config.require_vote_proof && !self.config.allow_mixed_mode {
            return Err(ZKCoordinationError::ProofRequired);
        }

        let member = self
            .members
            .get(agent_id)
            .ok_or_else(|| ZKCoordinationError::NotMember {
                agent_id: agent_id.to_string(),
            })?;

        let state = self.proposals.get_mut(proposal_id).ok_or_else(|| {
            ZKCoordinationError::ProposalNotFound {
                proposal_id: proposal_id.to_string(),
            }
        })?;

        if state.finalized {
            return Err(ZKCoordinationError::ProposalFinalized {
                proposal_id: proposal_id.to_string(),
            });
        }

        let weight = if self.config.coordination.quadratic_voting {
            member.proven_trust.sqrt()
        } else {
            member.proven_trust
        };

        let vote = Vote {
            agent_id: agent_id.to_string(),
            vote_type,
            trust_score: member.proven_trust,
            weight,
            timestamp: Self::now(),
        };

        state.regular_votes.insert(agent_id.to_string(), vote);

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Consensus
    // -------------------------------------------------------------------------

    /// Check consensus on a proposal
    pub fn check_consensus(&mut self, proposal_id: &str) -> Option<ConsensusResult> {
        let state = self.proposals.get_mut(proposal_id)?;

        if state.finalized {
            return state.result.clone();
        }

        // Calculate weights from ZK proofs
        let mut approval_weight = 0.0;
        let mut rejection_weight = 0.0;
        let mut abstention_weight = 0.0;

        let quadratic = self.config.coordination.quadratic_voting;

        // Count ZK votes
        for proof in state.zk_votes.values() {
            let weight = proof.weight(quadratic);
            match proof.vote_type {
                VoteType::Approve => approval_weight += weight,
                VoteType::Reject => rejection_weight += weight,
                VoteType::Abstain => abstention_weight += weight,
            }
        }

        // Count regular votes (if mixed mode)
        for vote in state.regular_votes.values() {
            match vote.vote_type {
                VoteType::Approve => approval_weight += vote.weight,
                VoteType::Reject => rejection_weight += vote.weight,
                VoteType::Abstain => abstention_weight += vote.weight,
            }
        }

        let total_vote_weight = approval_weight + rejection_weight + abstention_weight;
        let total_possible = if quadratic {
            self.members.values().map(|m| m.proven_trust.sqrt()).sum()
        } else {
            self.total_proven_trust
        };

        let participation_rate = if total_possible > 0.0 {
            total_vote_weight / total_possible
        } else {
            0.0
        };

        let quorum_reached = participation_rate >= self.config.coordination.min_participation;

        let decision = if state.proposal.is_expired() {
            ConsensusDecision::Expired
        } else if !quorum_reached {
            ConsensusDecision::Pending
        } else {
            let active_weight = approval_weight + rejection_weight;
            if active_weight > 0.0 {
                let approval_ratio = approval_weight / active_weight;
                if approval_ratio >= self.config.coordination.approval_threshold {
                    ConsensusDecision::Approved
                } else {
                    ConsensusDecision::Rejected
                }
            } else {
                ConsensusDecision::Pending
            }
        };

        let voter_count = state.zk_votes.len() + state.regular_votes.len();

        let result = ConsensusResult {
            proposal_id: proposal_id.to_string(),
            decision,
            approval_weight,
            rejection_weight,
            abstention_weight,
            participation_rate,
            quorum_reached,
            voter_count,
            total_members: self.members.len(),
        };

        // Finalize if decision is final
        if matches!(
            decision,
            ConsensusDecision::Approved
                | ConsensusDecision::Rejected
                | ConsensusDecision::NoQuorum
                | ConsensusDecision::Expired
        ) {
            state.finalized = true;
            state.result = Some(result.clone());

            // Update participation counts
            for agent_id in state.zk_votes.keys().chain(state.regular_votes.keys()) {
                if let Some(member) = self.members.get_mut(agent_id) {
                    member.participation_count += 1;
                }
            }
        }

        Some(result)
    }

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Get total proven trust
    pub fn total_proven_trust(&self) -> f64 {
        self.total_proven_trust
    }

    /// Check if all members have proofs
    pub fn all_members_have_proofs(&self) -> bool {
        self.members.values().all(|m| m.membership_proof.is_some())
    }
}

// ============================================================================
// Errors
// ============================================================================

/// ZK coordination errors
#[derive(Debug, Clone)]
pub enum ZKCoordinationError {
    /// Proof verification failed.
    ProofVerificationFailed {
        /// Agent whose proof failed.
        agent_id: String,
    },
    /// Proof expired.
    ProofExpired {
        /// Agent whose proof expired.
        agent_id: String,
    },
    /// Proof required but not provided
    ProofRequired,
    /// Agent not a member.
    NotMember {
        /// Agent identifier.
        agent_id: String,
    },
    /// Agent already a member.
    AlreadyMember {
        /// Agent identifier.
        agent_id: String,
    },
    /// Group is full
    GroupFull,
    /// Insufficient trust.
    InsufficientTrust {
        /// Agent identifier.
        agent_id: String,
        /// Agent's current trust.
        trust: f64,
        /// Required trust level.
        required: f64,
    },
    /// Proposal not found.
    ProposalNotFound {
        /// Proposal identifier.
        proposal_id: String,
    },
    /// Proposal already finalized.
    ProposalFinalized {
        /// Proposal identifier.
        proposal_id: String,
    },
    /// Underlying proof error
    ProofError(String),
}

impl std::fmt::Display for ZKCoordinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProofVerificationFailed { agent_id } => {
                write!(f, "Proof verification failed for agent {}", agent_id)
            }
            Self::ProofExpired { agent_id } => {
                write!(f, "Proof expired for agent {}", agent_id)
            }
            Self::ProofRequired => write!(f, "ZK proof required"),
            Self::NotMember { agent_id } => write!(f, "Agent {} is not a member", agent_id),
            Self::AlreadyMember { agent_id } => write!(f, "Agent {} is already a member", agent_id),
            Self::GroupFull => write!(f, "Group is full"),
            Self::InsufficientTrust {
                agent_id,
                trust,
                required,
            } => {
                write!(
                    f,
                    "Agent {} has trust {} but {} required",
                    agent_id, trust, required
                )
            }
            Self::ProposalNotFound { proposal_id } => {
                write!(f, "Proposal {} not found", proposal_id)
            }
            Self::ProposalFinalized { proposal_id } => {
                write!(f, "Proposal {} already finalized", proposal_id)
            }
            Self::ProofError(e) => write!(f, "Proof error: {}", e),
        }
    }
}

impl std::error::Error for ZKCoordinationError {}

impl From<ProofError> for ZKCoordinationError {
    fn from(e: ProofError) -> Self {
        Self::ProofError(format!("{:?}", e))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute trust score from K-Vector (simplified)
fn compute_trust_score(kvector: &KVector) -> f64 {
    let values = kvector.to_array();
    let weights = [0.20, 0.10, 0.20, 0.15, 0.10, 0.05, 0.10, 0.05, 0.025, 0.025];

    values
        .iter()
        .zip(weights.iter())
        .map(|(v, w)| (*v as f64) * w)
        .sum::<f64>()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_kvector(trust: f32) -> KVector {
        KVector::new(
            trust, trust, trust, trust, trust, trust, trust, trust, trust, trust,
        )
    }

    #[test]
    fn test_zk_group_creation() {
        let group = ZKAgentGroup::new(ZKCoordinationConfig::default());
        assert_eq!(group.member_count(), 0);
    }

    #[test]
    fn test_mixed_mode_membership() {
        let mut config = ZKCoordinationConfig::default();
        config.allow_mixed_mode = true;
        config.require_membership_proof = false;

        let mut group = ZKAgentGroup::new(config);

        // Add member without proof
        group.add_member_without_proof("agent-1", 0.7).unwrap();
        assert_eq!(group.member_count(), 1);
        assert!(!group.has_valid_proof("agent-1"));
    }

    #[test]
    fn test_proof_required_enforcement() {
        let config = ZKCoordinationConfig::default();
        let mut group = ZKAgentGroup::new(config);

        // Should fail - no proof provided
        let result = group.add_member_without_proof("agent-1", 0.7);
        assert!(matches!(result, Err(ZKCoordinationError::ProofRequired)));
    }
}
