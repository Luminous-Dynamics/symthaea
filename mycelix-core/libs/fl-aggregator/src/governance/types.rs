//! Core Governance Types
//!
//! Defines proposal types, vote structures, and governance states
//! compatible with Mycelix Holochain governance zomes.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Proposal types with different voting periods and requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProposalType {
    /// Standard proposals (7 day voting period)
    /// Used for: FL parameter changes, model updates, routine governance
    Standard,

    /// Emergency proposals (24 hour voting period)
    /// Used for: Security fixes, critical bugs, attack response
    /// Requires: 2/3 supermajority, guardian approval
    Emergency,

    /// Constitutional proposals (30 day voting period)
    /// Used for: Protocol changes, charter amendments, fundamental rules
    /// Requires: 3/4 supermajority, extended discussion period
    Constitutional,

    /// Model governance proposals (14 day voting period)
    /// Used for: FL model decisions, aggregation strategy, Byzantine thresholds
    ModelGovernance,

    /// Treasury proposals (7 day voting period)
    /// Used for: Fund allocation, grants, ecosystem support
    Treasury,

    /// Membership proposals (3 day voting period)
    /// Used for: Adding/removing members, role assignments
    Membership,
}

impl ProposalType {
    /// Get voting duration for this proposal type
    pub fn voting_duration(&self) -> Duration {
        match self {
            ProposalType::Standard => Duration::days(7),
            ProposalType::Emergency => Duration::hours(24),
            ProposalType::Constitutional => Duration::days(30),
            ProposalType::ModelGovernance => Duration::days(14),
            ProposalType::Treasury => Duration::days(7),
            ProposalType::Membership => Duration::days(3),
        }
    }

    /// Get discussion period before voting starts
    pub fn discussion_period(&self) -> Duration {
        match self {
            ProposalType::Standard => Duration::days(2),
            ProposalType::Emergency => Duration::hours(0), // No discussion for emergencies
            ProposalType::Constitutional => Duration::days(7),
            ProposalType::ModelGovernance => Duration::days(3),
            ProposalType::Treasury => Duration::days(2),
            ProposalType::Membership => Duration::days(1),
        }
    }

    /// Get required quorum (fraction of eligible voters)
    pub fn quorum_threshold(&self) -> f32 {
        match self {
            ProposalType::Standard => 0.10,       // 10% of eligible voters
            ProposalType::Emergency => 0.05,      // 5% for fast response
            ProposalType::Constitutional => 0.25, // 25% for major changes
            ProposalType::ModelGovernance => 0.15,
            ProposalType::Treasury => 0.15,
            ProposalType::Membership => 0.05,
        }
    }

    /// Get required approval threshold (fraction of votes cast)
    pub fn approval_threshold(&self) -> f32 {
        match self {
            ProposalType::Standard => 0.50,       // Simple majority
            ProposalType::Emergency => 0.67,      // 2/3 supermajority
            ProposalType::Constitutional => 0.75, // 3/4 supermajority
            ProposalType::ModelGovernance => 0.60,
            ProposalType::Treasury => 0.60,
            ProposalType::Membership => 0.50,
        }
    }

    /// Check if guardian approval is required
    pub fn requires_guardian_approval(&self) -> bool {
        matches!(self, ProposalType::Emergency | ProposalType::Constitutional)
    }

    /// Get timelock duration after approval (before execution)
    pub fn timelock_duration(&self) -> Duration {
        match self {
            ProposalType::Standard => Duration::hours(48),
            ProposalType::Emergency => Duration::hours(0), // Immediate execution
            ProposalType::Constitutional => Duration::days(7),
            ProposalType::ModelGovernance => Duration::hours(24),
            ProposalType::Treasury => Duration::hours(48),
            ProposalType::Membership => Duration::hours(24),
        }
    }
}

/// Proposal status lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    /// Initial draft state
    Draft,
    /// Discussion period (before voting)
    Discussion,
    /// Active voting period
    Voting,
    /// Voting ended, counting votes
    Tallying,
    /// Approved, in timelock period
    Approved,
    /// Rejected by vote
    Rejected,
    /// Vetoed by guardian
    Vetoed,
    /// Executed successfully
    Executed,
    /// Execution failed
    Failed,
    /// Cancelled by proposer
    Cancelled,
    /// Expired without reaching quorum
    Expired,
}

impl ProposalStatus {
    /// Check if proposal is in an active state
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            ProposalStatus::Draft
                | ProposalStatus::Discussion
                | ProposalStatus::Voting
                | ProposalStatus::Tallying
                | ProposalStatus::Approved
        )
    }

    /// Check if voting is currently allowed
    pub fn can_vote(&self) -> bool {
        matches!(self, ProposalStatus::Voting)
    }

    /// Check if proposal is finalized
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            ProposalStatus::Rejected
                | ProposalStatus::Vetoed
                | ProposalStatus::Executed
                | ProposalStatus::Failed
                | ProposalStatus::Cancelled
                | ProposalStatus::Expired
        )
    }
}

/// Vote choice options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoteChoice {
    /// Support the proposal
    For,
    /// Oppose the proposal
    Against,
    /// Abstain but count towards quorum
    Abstain,
    /// Signal need for more discussion
    NeedsWork,
}

/// A governance proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Unique proposal identifier (e.g., "MIP-0042")
    pub id: String,

    /// Short descriptive title
    pub title: String,

    /// Full proposal description (Markdown)
    pub description: String,

    /// Proposal type determining voting rules
    pub proposal_type: ProposalType,

    /// Current status
    pub status: ProposalStatus,

    /// DID of the proposer
    pub proposer_did: String,

    /// Co-sponsors (required for some proposal types)
    pub sponsors: Vec<String>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Discussion start time
    pub discussion_starts: Option<DateTime<Utc>>,

    /// Voting start time
    pub voting_starts: Option<DateTime<Utc>>,

    /// Voting end time
    pub voting_ends: Option<DateTime<Utc>>,

    /// Execution scheduled time (after timelock)
    pub execution_time: Option<DateTime<Utc>>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Related proposals
    pub related_proposals: Vec<String>,

    /// Executable actions (encoded)
    pub actions: Vec<ProposalAction>,

    /// Amendments history
    pub amendments: Vec<Amendment>,

    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// An executable action within a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalAction {
    /// Action type identifier
    pub action_type: ActionType,

    /// Target (contract, parameter, etc.)
    pub target: String,

    /// Action parameters
    pub params: HashMap<String, serde_json::Value>,

    /// Execution order
    pub order: u32,

    /// Whether this action is optional
    pub optional: bool,
}

/// Types of executable actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    /// Update FL aggregation parameters
    UpdateFLParameters,
    /// Modify Byzantine detection thresholds
    UpdateByzantineThresholds,
    /// Change model architecture settings
    UpdateModelConfig,
    /// Treasury transfer
    TreasuryTransfer,
    /// Update governance parameters
    UpdateGovernanceParams,
    /// Add/remove member
    MembershipChange,
    /// Execute arbitrary zome call
    ZomeCall,
    /// Custom action
    Custom(String),
}

/// Amendment to a proposal during discussion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Amendment {
    /// Amendment ID
    pub id: String,

    /// Author DID
    pub author_did: String,

    /// Description of changes
    pub description: String,

    /// Timestamp
    pub created_at: DateTime<Utc>,

    /// Whether accepted by proposer
    pub accepted: bool,
}

/// A vote on a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Proposal ID being voted on
    pub proposal_id: String,

    /// Voter DID
    pub voter_did: String,

    /// Vote choice
    pub choice: VoteChoice,

    /// Calculated vote weight
    pub weight: f32,

    /// Optional rationale
    pub rationale: Option<String>,

    /// Timestamp
    pub cast_at: DateTime<Utc>,

    /// Whether this vote was delegated
    pub delegated: bool,

    /// Original delegator if delegated
    pub delegator_did: Option<String>,

    /// Signature proving vote authenticity
    pub signature: Vec<u8>,
}

/// Vote tally result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VoteTally {
    /// Total weight of For votes
    pub for_weight: f32,

    /// Total weight of Against votes
    pub against_weight: f32,

    /// Total weight of Abstain votes
    pub abstain_weight: f32,

    /// Total weight of NeedsWork votes
    pub needs_work_weight: f32,

    /// Number of unique voters
    pub voter_count: u32,

    /// Total eligible voting weight
    pub eligible_weight: f32,

    /// Quorum reached
    pub quorum_reached: bool,

    /// Approval threshold met
    pub approval_met: bool,
}

impl VoteTally {
    /// Calculate participation rate
    pub fn participation_rate(&self) -> f32 {
        if self.eligible_weight > 0.0 {
            (self.for_weight + self.against_weight + self.abstain_weight + self.needs_work_weight)
                / self.eligible_weight
        } else {
            0.0
        }
    }

    /// Calculate approval rate (excluding abstains)
    pub fn approval_rate(&self) -> f32 {
        let total_decisive = self.for_weight + self.against_weight;
        if total_decisive > 0.0 {
            self.for_weight / total_decisive
        } else {
            0.0
        }
    }

    /// Add a vote to the tally
    pub fn add_vote(&mut self, choice: VoteChoice, weight: f32) {
        match choice {
            VoteChoice::For => self.for_weight += weight,
            VoteChoice::Against => self.against_weight += weight,
            VoteChoice::Abstain => self.abstain_weight += weight,
            VoteChoice::NeedsWork => self.needs_work_weight += weight,
        }
        self.voter_count += 1;
    }

    /// Check if quorum is reached for given threshold
    pub fn check_quorum(&mut self, threshold: f32) {
        self.quorum_reached = self.participation_rate() >= threshold;
    }

    /// Check if approval threshold is met
    pub fn check_approval(&mut self, threshold: f32) {
        self.approval_met = self.approval_rate() >= threshold;
    }
}

/// Governance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceConfig {
    /// Minimum MATL score to create proposals
    pub min_proposal_matl: f32,

    /// Minimum stake to create proposals
    pub min_proposal_stake: f32,

    /// Required sponsors for Constitutional proposals
    pub constitutional_sponsors: u32,

    /// Guardian DIDs with veto power
    pub guardians: Vec<String>,

    /// Whether delegation is enabled
    pub delegation_enabled: bool,

    /// Maximum delegation depth
    pub max_delegation_depth: u32,

    /// Proposal cooldown for same proposer
    pub proposal_cooldown: Duration,
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        Self {
            min_proposal_matl: 0.5,
            min_proposal_stake: 100.0,
            constitutional_sponsors: 5,
            guardians: Vec::new(),
            delegation_enabled: true,
            max_delegation_depth: 3,
            proposal_cooldown: Duration::days(1),
        }
    }
}

impl Proposal {
    /// Create a new proposal
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        description: impl Into<String>,
        proposal_type: ProposalType,
        proposer_did: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            title: title.into(),
            description: description.into(),
            proposal_type,
            status: ProposalStatus::Draft,
            proposer_did: proposer_did.into(),
            sponsors: Vec::new(),
            created_at: now,
            discussion_starts: None,
            voting_starts: None,
            voting_ends: None,
            execution_time: None,
            tags: Vec::new(),
            related_proposals: Vec::new(),
            actions: Vec::new(),
            amendments: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Start discussion phase
    pub fn start_discussion(&mut self) {
        let now = Utc::now();
        self.discussion_starts = Some(now);
        self.voting_starts = Some(now + self.proposal_type.discussion_period());
        self.status = ProposalStatus::Discussion;
    }

    /// Start voting phase
    pub fn start_voting(&mut self) {
        let now = Utc::now();
        self.voting_starts = Some(now);
        self.voting_ends = Some(now + self.proposal_type.voting_duration());
        self.status = ProposalStatus::Voting;
    }

    /// Check if voting period has ended
    pub fn voting_ended(&self) -> bool {
        if let Some(end) = self.voting_ends {
            Utc::now() > end
        } else {
            false
        }
    }

    /// Add an action to the proposal
    pub fn add_action(&mut self, action: ProposalAction) {
        self.actions.push(action);
    }

    /// Add a sponsor
    pub fn add_sponsor(&mut self, sponsor_did: impl Into<String>) {
        let did = sponsor_did.into();
        if !self.sponsors.contains(&did) {
            self.sponsors.push(did);
        }
    }
}

impl Vote {
    /// Create a new vote
    pub fn new(
        proposal_id: impl Into<String>,
        voter_did: impl Into<String>,
        choice: VoteChoice,
        weight: f32,
    ) -> Self {
        Self {
            proposal_id: proposal_id.into(),
            voter_did: voter_did.into(),
            choice,
            weight,
            rationale: None,
            cast_at: Utc::now(),
            delegated: false,
            delegator_did: None,
            signature: Vec::new(),
        }
    }

    /// Create a delegated vote
    pub fn delegated(
        proposal_id: impl Into<String>,
        delegate_did: impl Into<String>,
        delegator_did: impl Into<String>,
        choice: VoteChoice,
        weight: f32,
    ) -> Self {
        Self {
            proposal_id: proposal_id.into(),
            voter_did: delegate_did.into(),
            choice,
            weight,
            rationale: None,
            cast_at: Utc::now(),
            delegated: true,
            delegator_did: Some(delegator_did.into()),
            signature: Vec::new(),
        }
    }

    /// Set vote rationale
    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = Some(rationale.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposal_type_durations() {
        assert_eq!(ProposalType::Standard.voting_duration(), Duration::days(7));
        assert_eq!(ProposalType::Emergency.voting_duration(), Duration::hours(24));
        assert_eq!(ProposalType::Constitutional.voting_duration(), Duration::days(30));
    }

    #[test]
    fn test_proposal_type_thresholds() {
        assert_eq!(ProposalType::Standard.approval_threshold(), 0.50);
        assert_eq!(ProposalType::Emergency.approval_threshold(), 0.67);
        assert_eq!(ProposalType::Constitutional.approval_threshold(), 0.75);
    }

    #[test]
    fn test_proposal_creation() {
        let proposal = Proposal::new(
            "MIP-0001",
            "Test Proposal",
            "A test proposal for unit testing",
            ProposalType::Standard,
            "did:mycelix:proposer",
        );

        assert_eq!(proposal.id, "MIP-0001");
        assert_eq!(proposal.status, ProposalStatus::Draft);
        assert!(proposal.actions.is_empty());
    }

    #[test]
    fn test_vote_tally() {
        let mut tally = VoteTally::default();
        tally.eligible_weight = 100.0;

        tally.add_vote(VoteChoice::For, 30.0);
        tally.add_vote(VoteChoice::For, 20.0);
        tally.add_vote(VoteChoice::Against, 10.0);
        tally.add_vote(VoteChoice::Abstain, 5.0);

        assert_eq!(tally.for_weight, 50.0);
        assert_eq!(tally.against_weight, 10.0);
        assert_eq!(tally.voter_count, 4);
        assert!((tally.participation_rate() - 0.65).abs() < 0.01);
        assert!((tally.approval_rate() - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_proposal_status_checks() {
        assert!(ProposalStatus::Voting.can_vote());
        assert!(!ProposalStatus::Draft.can_vote());
        assert!(ProposalStatus::Executed.is_final());
        assert!(!ProposalStatus::Voting.is_final());
    }
}
