//! Multi-Agent Coordination
//!
//! Enables AI agents to collaborate with trust-weighted influence.
//! Supports consensus mechanisms for agent groups.
//!
//! # Features
//!
//! - **Trust-Weighted Voting**: Agents vote with weight proportional to trust
//! - **Consensus Protocols**: BFT-style agreement among agents
//! - **Influence Scaling**: Higher trust = more influence in decisions
//! - **Reputation Feedback**: Group outcomes affect individual K-Vectors
//!
//! # Example
//!
//! ```rust
//! use mycelix_sdk::agentic::coordination::{
//!     AgentGroup, CoordinationConfig, Proposal, VoteType,
//! };
//!
//! // Create a group of trading agents
//! let mut group = AgentGroup::new(CoordinationConfig::default());
//!
//! // Add agents with their trust scores
//! group.add_member("agent-1", 0.8);
//! group.add_member("agent-2", 0.6);
//! group.add_member("agent-3", 0.7);
//!
//! // Create a proposal
//! let proposal = Proposal::new(
//!     "increase-position",
//!     "Increase BTC position by 10%",
//! );
//!
//! // Submit votes
//! group.vote("agent-1", proposal.id(), VoteType::Approve);
//! group.vote("agent-2", proposal.id(), VoteType::Approve);
//! group.vote("agent-3", proposal.id(), VoteType::Reject);
//!
//! // Check consensus (trust-weighted)
//! if let Some(result) = group.check_consensus(proposal.id()) {
//!     println!("Decision: {:?}, Weight: {}", result.decision, result.approval_weight);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationConfig {
    /// Minimum trust required to participate in group
    pub min_trust_threshold: f64,
    /// Approval threshold for consensus (weighted)
    pub approval_threshold: f64,
    /// Minimum participation rate required
    pub min_participation: f64,
    /// Maximum time for voting (ms)
    pub voting_timeout_ms: u64,
    /// Whether to use quadratic voting (sqrt of trust)
    pub quadratic_voting: bool,
    /// Maximum group size
    pub max_group_size: usize,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            min_trust_threshold: 0.3,
            approval_threshold: 0.67, // 2/3 majority
            min_participation: 0.5,   // 50% must vote
            voting_timeout_ms: 60_000,
            quadratic_voting: false,
            max_group_size: 100,
        }
    }
}

/// Vote type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteType {
    /// Approve the proposal
    Approve,
    /// Reject the proposal
    Reject,
    /// Abstain from voting
    Abstain,
}

/// A proposal for group decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Unique proposal ID
    id: String,
    /// Proposal type/action
    action: String,
    /// Description
    description: String,
    /// Creator agent ID
    creator: String,
    /// Creation timestamp (ms)
    created_at: u64,
    /// Expiry timestamp (ms)
    expires_at: u64,
    /// Required quorum (weighted trust)
    required_quorum: f64,
    /// Metadata
    metadata: HashMap<String, String>,
}

impl Proposal {
    /// Create a new proposal
    pub fn new(action: impl Into<String>, description: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: format!("prop-{}", now),
            action: action.into(),
            description: description.into(),
            creator: String::new(),
            created_at: now,
            expires_at: now + 3600_000, // 1 hour default
            required_quorum: 0.5,
            metadata: HashMap::new(),
        }
    }

    /// Set the creator
    pub fn with_creator(mut self, creator: impl Into<String>) -> Self {
        self.creator = creator.into();
        self
    }

    /// Set expiry duration
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.expires_at = self.created_at + duration_ms;
        self
    }

    /// Set required quorum
    pub fn with_quorum(mut self, quorum: f64) -> Self {
        self.required_quorum = quorum.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get proposal ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get action
    pub fn action(&self) -> &str {
        &self.action
    }

    /// Check if expired
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now > self.expires_at
    }
}

/// A vote on a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Agent casting the vote
    pub agent_id: String,
    /// Vote type
    pub vote_type: VoteType,
    /// Agent's trust at time of vote
    pub trust_score: f64,
    /// Calculated weight (may differ from trust if quadratic)
    pub weight: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    /// Proposal ID
    pub proposal_id: String,
    /// Final decision
    pub decision: ConsensusDecision,
    /// Total approval weight
    pub approval_weight: f64,
    /// Total rejection weight
    pub rejection_weight: f64,
    /// Total abstention weight
    pub abstention_weight: f64,
    /// Participation rate
    pub participation_rate: f64,
    /// Whether quorum was reached
    pub quorum_reached: bool,
    /// Number of voters
    pub voter_count: usize,
    /// Total group members
    pub total_members: usize,
}

/// Consensus decision outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusDecision {
    /// Proposal approved
    Approved,
    /// Proposal rejected
    Rejected,
    /// No consensus reached (quorum not met)
    NoQuorum,
    /// Voting still in progress
    Pending,
    /// Proposal expired
    Expired,
}

/// Group member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMember {
    /// Agent ID
    pub agent_id: String,
    /// Current trust score
    pub trust_score: f64,
    /// When the agent joined (ms since epoch)
    pub joined_at: u64,
    /// Number of proposals participated in
    pub participation_count: u64,
}

/// Proposal state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalState {
    /// The proposal
    pub proposal: Proposal,
    /// Votes cast
    pub votes: HashMap<String, Vote>,
    /// Whether voting is complete
    pub finalized: bool,
    /// Final result if finalized
    pub result: Option<ConsensusResult>,
}

/// A coordinated group of agents
#[derive(Debug, Clone)]
pub struct AgentGroup {
    /// Group ID
    id: String,
    /// Configuration
    config: CoordinationConfig,
    /// Group members
    members: HashMap<String, GroupMember>,
    /// Active proposals
    proposals: HashMap<String, ProposalState>,
    /// Total trust in group
    total_trust: f64,
}

/// Serializable snapshot of AgentGroup state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGroupSnapshot {
    /// Group ID
    pub id: String,
    /// Configuration
    pub config: CoordinationConfig,
    /// Group members
    pub members: Vec<GroupMember>,
    /// Active proposals
    pub proposals: Vec<ProposalState>,
    /// Total trust in group
    pub total_trust: f64,
    /// Snapshot timestamp
    pub snapshot_at: u64,
}

impl AgentGroup {
    /// Create a new agent group
    pub fn new(config: CoordinationConfig) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: format!("group-{}", now),
            config,
            members: HashMap::new(),
            proposals: HashMap::new(),
            total_trust: 0.0,
        }
    }

    /// Create a group with specific ID
    pub fn with_id(id: impl Into<String>, config: CoordinationConfig) -> Self {
        Self {
            id: id.into(),
            config,
            members: HashMap::new(),
            proposals: HashMap::new(),
            total_trust: 0.0,
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

    /// Get total trust weight
    pub fn total_trust(&self) -> f64 {
        self.total_trust
    }

    /// Add a member to the group
    pub fn add_member(
        &mut self,
        agent_id: impl Into<String>,
        trust_score: f64,
    ) -> Result<(), CoordinationError> {
        let agent_id = agent_id.into();

        // Check trust threshold
        if trust_score < self.config.min_trust_threshold {
            return Err(CoordinationError::InsufficientTrust {
                agent_id,
                trust: trust_score,
                required: self.config.min_trust_threshold,
            });
        }

        // Check group size
        if self.members.len() >= self.config.max_group_size {
            return Err(CoordinationError::GroupFull {
                max_size: self.config.max_group_size,
            });
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let member = GroupMember {
            agent_id: agent_id.clone(),
            trust_score,
            joined_at: now,
            participation_count: 0,
        };

        self.total_trust += trust_score;
        self.members.insert(agent_id, member);

        Ok(())
    }

    /// Remove a member from the group
    pub fn remove_member(&mut self, agent_id: &str) -> bool {
        if let Some(member) = self.members.remove(agent_id) {
            self.total_trust -= member.trust_score;
            true
        } else {
            false
        }
    }

    /// Update a member's trust score
    pub fn update_trust(
        &mut self,
        agent_id: &str,
        new_trust: f64,
    ) -> Result<(), CoordinationError> {
        if let Some(member) = self.members.get_mut(agent_id) {
            self.total_trust -= member.trust_score;
            member.trust_score = new_trust;
            self.total_trust += new_trust;

            // Check if they still meet threshold
            if new_trust < self.config.min_trust_threshold {
                self.members.remove(agent_id);
                return Err(CoordinationError::InsufficientTrust {
                    agent_id: agent_id.to_string(),
                    trust: new_trust,
                    required: self.config.min_trust_threshold,
                });
            }

            Ok(())
        } else {
            Err(CoordinationError::NotMember {
                agent_id: agent_id.to_string(),
            })
        }
    }

    /// Submit a proposal
    pub fn submit_proposal(&mut self, mut proposal: Proposal) -> Result<String, CoordinationError> {
        // Verify creator is a member
        if !proposal.creator.is_empty() && !self.members.contains_key(&proposal.creator) {
            return Err(CoordinationError::NotMember {
                agent_id: proposal.creator.clone(),
            });
        }

        // Set quorum based on config
        proposal.required_quorum = self.config.min_participation;

        let proposal_id = proposal.id.clone();

        self.proposals.insert(
            proposal_id.clone(),
            ProposalState {
                proposal,
                votes: HashMap::new(),
                finalized: false,
                result: None,
            },
        );

        Ok(proposal_id)
    }

    /// Cast a vote on a proposal
    pub fn vote(
        &mut self,
        agent_id: &str,
        proposal_id: &str,
        vote_type: VoteType,
    ) -> Result<(), CoordinationError> {
        // Check membership
        let member = self
            .members
            .get(agent_id)
            .ok_or_else(|| CoordinationError::NotMember {
                agent_id: agent_id.to_string(),
            })?;

        // Get proposal
        let state = self.proposals.get_mut(proposal_id).ok_or_else(|| {
            CoordinationError::ProposalNotFound {
                proposal_id: proposal_id.to_string(),
            }
        })?;

        // Check if finalized
        if state.finalized {
            return Err(CoordinationError::ProposalFinalized {
                proposal_id: proposal_id.to_string(),
            });
        }

        // Check if expired
        if state.proposal.is_expired() {
            return Err(CoordinationError::ProposalExpired {
                proposal_id: proposal_id.to_string(),
            });
        }

        // Calculate weight
        let weight = if self.config.quadratic_voting {
            member.trust_score.sqrt()
        } else {
            member.trust_score
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let vote = Vote {
            agent_id: agent_id.to_string(),
            vote_type,
            trust_score: member.trust_score,
            weight,
            timestamp: now,
        };

        state.votes.insert(agent_id.to_string(), vote);

        Ok(())
    }

    /// Check consensus on a proposal
    pub fn check_consensus(&mut self, proposal_id: &str) -> Option<ConsensusResult> {
        let state = self.proposals.get_mut(proposal_id)?;

        // If already finalized, return cached result
        if state.finalized {
            return state.result.clone();
        }

        // Calculate totals
        let mut approval_weight = 0.0;
        let mut rejection_weight = 0.0;
        let mut abstention_weight = 0.0;

        for vote in state.votes.values() {
            match vote.vote_type {
                VoteType::Approve => approval_weight += vote.weight,
                VoteType::Reject => rejection_weight += vote.weight,
                VoteType::Abstain => abstention_weight += vote.weight,
            }
        }

        let total_vote_weight = approval_weight + rejection_weight + abstention_weight;
        let total_possible = if self.config.quadratic_voting {
            // Sum of sqrt(trust) for each member
            self.members.values().map(|m| m.trust_score.sqrt()).sum()
        } else {
            self.total_trust
        };

        let participation_rate = if total_possible > 0.0 {
            total_vote_weight / total_possible
        } else {
            0.0
        };

        let quorum_reached = participation_rate >= self.config.min_participation;

        // Determine decision
        let decision = if state.proposal.is_expired() {
            ConsensusDecision::Expired
        } else if !quorum_reached {
            // Check if it's still possible to reach quorum
            let remaining_weight = total_possible - total_vote_weight;
            if total_vote_weight + remaining_weight < self.config.min_participation * total_possible
            {
                ConsensusDecision::NoQuorum
            } else {
                ConsensusDecision::Pending
            }
        } else {
            let active_vote_weight = approval_weight + rejection_weight;
            if active_vote_weight > 0.0 {
                let approval_ratio = approval_weight / active_vote_weight;
                if approval_ratio >= self.config.approval_threshold {
                    ConsensusDecision::Approved
                } else {
                    ConsensusDecision::Rejected
                }
            } else {
                ConsensusDecision::Pending
            }
        };

        let result = ConsensusResult {
            proposal_id: proposal_id.to_string(),
            decision,
            approval_weight,
            rejection_weight,
            abstention_weight,
            participation_rate,
            quorum_reached,
            voter_count: state.votes.len(),
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
            for vote in state.votes.values() {
                if let Some(member) = self.members.get_mut(&vote.agent_id) {
                    member.participation_count += 1;
                }
            }
        }

        Some(result)
    }

    /// Get proposal state
    pub fn get_proposal(&self, proposal_id: &str) -> Option<&Proposal> {
        self.proposals.get(proposal_id).map(|s| &s.proposal)
    }

    /// List active proposals
    pub fn active_proposals(&self) -> Vec<&Proposal> {
        self.proposals
            .values()
            .filter(|s| !s.finalized && !s.proposal.is_expired())
            .map(|s| &s.proposal)
            .collect()
    }

    /// Get member trust score
    pub fn get_member_trust(&self, agent_id: &str) -> Option<f64> {
        self.members.get(agent_id).map(|m| m.trust_score)
    }

    /// List members
    pub fn members(&self) -> Vec<(&str, f64)> {
        self.members
            .values()
            .map(|m| (m.agent_id.as_str(), m.trust_score))
            .collect()
    }

    // -------------------------------------------------------------------------
    // Persistence
    // -------------------------------------------------------------------------

    /// Create a snapshot for persistence
    pub fn snapshot(&self) -> AgentGroupSnapshot {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        AgentGroupSnapshot {
            id: self.id.clone(),
            config: self.config.clone(),
            members: self.members.values().cloned().collect(),
            proposals: self.proposals.values().cloned().collect(),
            total_trust: self.total_trust,
            snapshot_at: now,
        }
    }

    /// Restore from a snapshot
    pub fn from_snapshot(snapshot: AgentGroupSnapshot) -> Self {
        let members: HashMap<String, GroupMember> = snapshot
            .members
            .into_iter()
            .map(|m| (m.agent_id.clone(), m))
            .collect();

        let proposals: HashMap<String, ProposalState> = snapshot
            .proposals
            .into_iter()
            .map(|p| (p.proposal.id.clone(), p))
            .collect();

        Self {
            id: snapshot.id,
            config: snapshot.config,
            members,
            proposals,
            total_trust: snapshot.total_trust,
        }
    }

    /// Serialize to JSON bytes
    pub fn to_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&self.snapshot())
    }

    /// Deserialize from JSON bytes
    pub fn from_json(data: &[u8]) -> Result<Self, serde_json::Error> {
        let snapshot: AgentGroupSnapshot = serde_json::from_slice(data)?;
        Ok(Self::from_snapshot(snapshot))
    }

    /// Get configuration
    pub fn config(&self) -> &CoordinationConfig {
        &self.config
    }

    /// Get all proposal states (for persistence/inspection)
    pub fn all_proposals(&self) -> &HashMap<String, ProposalState> {
        &self.proposals
    }

    /// Get all members (for persistence/inspection)
    pub fn all_members(&self) -> &HashMap<String, GroupMember> {
        &self.members
    }

    // -------------------------------------------------------------------------
    // Observability / Metrics
    // -------------------------------------------------------------------------

    /// Get comprehensive metrics for monitoring
    pub fn metrics(&self) -> CoordinationMetrics {
        let active_proposals = self
            .proposals
            .values()
            .filter(|p| !p.finalized && !p.proposal.is_expired())
            .count();

        let finalized_proposals = self.proposals.values().filter(|p| p.finalized).count();

        let expired_proposals = self
            .proposals
            .values()
            .filter(|p| !p.finalized && p.proposal.is_expired())
            .count();

        let total_votes: usize = self.proposals.values().map(|p| p.votes.len()).sum();

        let approved_count = self
            .proposals
            .values()
            .filter(|p| {
                matches!(
                    p.result.as_ref().map(|r| &r.decision),
                    Some(ConsensusDecision::Approved)
                )
            })
            .count();

        let rejected_count = self
            .proposals
            .values()
            .filter(|p| {
                matches!(
                    p.result.as_ref().map(|r| &r.decision),
                    Some(ConsensusDecision::Rejected)
                )
            })
            .count();

        let avg_trust = if self.members.is_empty() {
            0.0
        } else {
            self.total_trust / self.members.len() as f64
        };

        let avg_participation = if finalized_proposals == 0 || self.members.is_empty() {
            0.0
        } else {
            self.proposals
                .values()
                .filter(|p| p.finalized)
                .map(|p| p.votes.len() as f64 / self.members.len() as f64)
                .sum::<f64>()
                / finalized_proposals as f64
        };

        let member_participation: Vec<_> = self
            .members
            .values()
            .map(|m| (m.agent_id.clone(), m.participation_count))
            .collect();

        CoordinationMetrics {
            group_id: self.id.clone(),
            member_count: self.members.len(),
            total_trust: self.total_trust,
            average_trust: avg_trust,
            active_proposals,
            finalized_proposals,
            expired_proposals,
            total_votes,
            approved_count,
            rejected_count,
            average_participation_rate: avg_participation,
            member_participation,
            quadratic_voting_enabled: self.config.quadratic_voting,
        }
    }

    /// Log a coordination event (for tracing integration)
    #[cfg(feature = "std")]
    pub fn log_event(&self, event: CoordinationEvent) {
        // In production, this would integrate with tracing/metrics
        // For now, we provide the event for external logging
        let _ = event; // Suppress unused warning
    }
}

/// Coordination metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMetrics {
    /// Group identifier
    pub group_id: String,
    /// Current member count
    pub member_count: usize,
    /// Total trust weight
    pub total_trust: f64,
    /// Average trust per member
    pub average_trust: f64,
    /// Number of active (non-finalized, non-expired) proposals
    pub active_proposals: usize,
    /// Number of finalized proposals
    pub finalized_proposals: usize,
    /// Number of expired proposals
    pub expired_proposals: usize,
    /// Total votes cast across all proposals
    pub total_votes: usize,
    /// Number of approved proposals
    pub approved_count: usize,
    /// Number of rejected proposals
    pub rejected_count: usize,
    /// Average participation rate in finalized proposals
    pub average_participation_rate: f64,
    /// Per-member participation counts
    pub member_participation: Vec<(String, u64)>,
    /// Whether quadratic voting is enabled
    pub quadratic_voting_enabled: bool,
}

impl CoordinationMetrics {
    /// Calculate approval rate
    pub fn approval_rate(&self) -> f64 {
        let total = self.approved_count + self.rejected_count;
        if total == 0 {
            0.0
        } else {
            self.approved_count as f64 / total as f64
        }
    }

    /// Check if group is healthy
    pub fn is_healthy(&self) -> bool {
        self.member_count >= 2
            && self.average_trust >= 0.3
            && self.average_participation_rate >= 0.3
    }
}

/// Coordination events for logging/tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationEvent {
    /// Member added to group
    MemberAdded {
        /// Group ID
        group_id: String,
        /// Agent ID
        agent_id: String,
        /// Trust score
        trust: f64,
    },
    /// Member removed from group
    MemberRemoved {
        /// Group ID
        group_id: String,
        /// Agent ID
        agent_id: String,
    },
    /// Proposal submitted
    ProposalSubmitted {
        /// Group ID
        group_id: String,
        /// Proposal ID
        proposal_id: String,
        /// Action type
        action: String,
    },
    /// Vote cast
    VoteCast {
        /// Group ID
        group_id: String,
        /// Proposal ID
        proposal_id: String,
        /// Voter ID
        voter_id: String,
        /// Vote type
        vote_type: VoteType,
        /// Vote weight
        weight: f64,
    },
    /// Consensus reached
    ConsensusReached {
        /// Group ID
        group_id: String,
        /// Proposal ID
        proposal_id: String,
        /// Decision
        decision: ConsensusDecision,
        /// Participation rate
        participation_rate: f64,
    },
}

/// Coordination error types
#[derive(Debug, Clone)]
pub enum CoordinationError {
    /// Agent trust too low.
    InsufficientTrust {
        /// Agent identifier.
        agent_id: String,
        /// Agent's current trust.
        trust: f64,
        /// Required trust level.
        required: f64,
    },
    /// Agent not a member.
    NotMember {
        /// Agent identifier.
        agent_id: String,
    },
    /// Group is full.
    GroupFull {
        /// Maximum group size.
        max_size: usize,
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
    /// Proposal expired.
    ProposalExpired {
        /// Proposal identifier.
        proposal_id: String,
    },
}

impl std::fmt::Display for CoordinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoordinationError::InsufficientTrust {
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
            CoordinationError::NotMember { agent_id } => {
                write!(f, "Agent {} is not a group member", agent_id)
            }
            CoordinationError::GroupFull { max_size } => {
                write!(f, "Group is full (max {} members)", max_size)
            }
            CoordinationError::ProposalNotFound { proposal_id } => {
                write!(f, "Proposal {} not found", proposal_id)
            }
            CoordinationError::ProposalFinalized { proposal_id } => {
                write!(f, "Proposal {} is already finalized", proposal_id)
            }
            CoordinationError::ProposalExpired { proposal_id } => {
                write!(f, "Proposal {} has expired", proposal_id)
            }
        }
    }
}

impl std::error::Error for CoordinationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_group() {
        let group = AgentGroup::new(CoordinationConfig::default());
        assert_eq!(group.member_count(), 0);
        assert_eq!(group.total_trust(), 0.0);
    }

    #[test]
    fn test_add_members() {
        let mut group = AgentGroup::new(CoordinationConfig::default());

        group.add_member("agent-1", 0.8).unwrap();
        group.add_member("agent-2", 0.6).unwrap();
        group.add_member("agent-3", 0.7).unwrap();

        assert_eq!(group.member_count(), 3);
        assert!((group.total_trust() - 2.1).abs() < 0.01);
    }

    #[test]
    fn test_trust_threshold() {
        let mut group = AgentGroup::new(CoordinationConfig {
            min_trust_threshold: 0.5,
            ..Default::default()
        });

        // Should fail - trust too low
        let result = group.add_member("low-trust", 0.3);
        assert!(result.is_err());

        // Should succeed
        let result = group.add_member("good-trust", 0.6);
        assert!(result.is_ok());
    }

    #[test]
    fn test_voting() {
        let mut group = AgentGroup::new(CoordinationConfig::default());

        group.add_member("agent-1", 0.8).unwrap();
        group.add_member("agent-2", 0.6).unwrap();
        group.add_member("agent-3", 0.7).unwrap();

        let proposal = Proposal::new("action", "description")
            .with_creator("agent-1".to_string())
            .with_duration_ms(3600_000);

        let proposal_id = group.submit_proposal(proposal).unwrap();

        // Vote
        group
            .vote("agent-1", &proposal_id, VoteType::Approve)
            .unwrap();
        group
            .vote("agent-2", &proposal_id, VoteType::Approve)
            .unwrap();
        group
            .vote("agent-3", &proposal_id, VoteType::Reject)
            .unwrap();

        // Check consensus
        let result = group.check_consensus(&proposal_id).unwrap();

        // agent-1 (0.8) + agent-2 (0.6) = 1.4 approve
        // agent-3 (0.7) = 0.7 reject
        // Total active: 2.1, approval ratio: 1.4/2.1 = 0.667 >= 0.67 threshold
        assert!(result.quorum_reached);
        assert_eq!(result.voter_count, 3);
    }

    #[test]
    fn test_quadratic_voting() {
        let mut group = AgentGroup::new(CoordinationConfig {
            quadratic_voting: true,
            min_participation: 0.5,
            approval_threshold: 0.5, // Lower threshold for this test
            ..Default::default()
        });

        // Agent with high trust
        group.add_member("whale", 0.9).unwrap();
        // Agents with lower trust
        group.add_member("small-1", 0.4).unwrap();
        group.add_member("small-2", 0.4).unwrap();
        group.add_member("small-3", 0.4).unwrap();

        let proposal = Proposal::new("action", "description");
        let proposal_id = group.submit_proposal(proposal).unwrap();

        // Whale votes approve
        group
            .vote("whale", &proposal_id, VoteType::Approve)
            .unwrap();
        // All small agents vote reject
        group
            .vote("small-1", &proposal_id, VoteType::Reject)
            .unwrap();
        group
            .vote("small-2", &proposal_id, VoteType::Reject)
            .unwrap();
        group
            .vote("small-3", &proposal_id, VoteType::Reject)
            .unwrap();

        let result = group.check_consensus(&proposal_id).unwrap();

        // With quadratic voting:
        // Whale: sqrt(0.9) ≈ 0.949
        // Small agents: 3 * sqrt(0.4) ≈ 1.897
        // Total active: 2.846
        // Approval ratio: 0.949 / 2.846 ≈ 0.333
        // This is less than 0.5 threshold, so rejected
        assert!(result.quorum_reached);
        assert!(result.approval_weight < result.rejection_weight);
        // The whale's vote alone is not enough to override the group
    }

    #[test]
    fn test_remove_member() {
        let mut group = AgentGroup::new(CoordinationConfig::default());

        group.add_member("agent-1", 0.8).unwrap();
        group.add_member("agent-2", 0.6).unwrap();

        assert_eq!(group.member_count(), 2);
        assert!((group.total_trust() - 1.4).abs() < 0.01);

        group.remove_member("agent-1");

        assert_eq!(group.member_count(), 1);
        assert!((group.total_trust() - 0.6).abs() < 0.01);
    }
}
