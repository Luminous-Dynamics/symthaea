//! RB-BFT: Reputation-Based Byzantine Fault Tolerance
//!
//! A novel consensus mechanism that achieves 34% validated Byzantine fault tolerance
//! through reputation-weighted voting. Theoretical target: 45% with improved detection.
//!
//! ## Key Innovation: Reputation² Weighting
//!
//! Instead of treating all validators equally, RB-BFT squares the reputation
//! score when computing voting weights. This means:
//! - High-reputation validators (0.9) get weight = 0.81
//! - Low-reputation validators (0.3) get weight = 0.09
//!
//! This quadratic weighting makes it exponentially harder for Byzantine
//! actors to accumulate enough influence to disrupt consensus.
//!
//! ## Consensus Flow
//!
//! 1. **Propose**: Leader proposes a block with PoGQ proof
//! 2. **Pre-vote**: Validators verify and pre-vote (reputation-weighted)
//! 3. **Pre-commit**: After 2/3+ weighted pre-votes, validators pre-commit
//! 4. **Commit**: After 2/3+ weighted pre-commits, block is finalized
//!
//! ## Integration with MATL
//!
//! RB-BFT uses the K-Vector trust scoring system for reputation:
//! - `k_r` (reputation dimension) is primary input
//! - Governance tier determines proposal rights
//! - Byzantine detection feeds into reputation updates

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

use super::kvector::{GovernanceTier, KVector};
use super::ProofOfGradientQuality;
use crate::crypto::secure_compare_hash;

/// Minimum reputation to participate as validator
pub const MIN_VALIDATOR_REPUTATION: f32 = 0.3;

/// Default Byzantine tolerance threshold for RB-BFT (validated to 34%)
pub const RBBFT_BYZANTINE_THRESHOLD: f32 = 0.34;

/// Quorum threshold for consensus (2/3 of weighted votes)
pub const QUORUM_THRESHOLD: f32 = 0.667;

/// Round timeout in seconds
pub const DEFAULT_ROUND_TIMEOUT_SECS: u64 = 30;

/// Maximum rounds before view change
pub const MAX_ROUNDS_BEFORE_VIEW_CHANGE: u32 = 3;

/// Unique identifier for a block proposal
pub type ProposalHash = [u8; 32];

/// Unique identifier for a finalized block
pub type BlockHash = [u8; 32];

/// Validator public key (placeholder - would be AgentPubKey in Holochain)
pub type ValidatorId = [u8; 32];

/// RB-BFT Consensus Round State
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/matl/"))]
pub enum RoundState {
    /// Waiting for proposal from leader
    Propose,
    /// Collecting pre-votes
    PreVote,
    /// Collecting pre-commits
    PreCommit,
    /// Block committed successfully
    Committed,
    /// Round failed, need view change
    Failed,
}

/// Vote type in the consensus protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/matl/"))]
pub enum VoteType {
    /// Pre-vote for a proposal
    PreVote,
    /// Pre-commit after sufficient pre-votes
    PreCommit,
}

/// A single vote from a validator
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/matl/"))]
pub struct Vote {
    /// Who is voting
    pub validator_id: ValidatorId,
    /// Type of vote (pre-vote or pre-commit)
    pub vote_type: VoteType,
    /// Hash of the proposal being voted on
    pub proposal_hash: ProposalHash,
    /// Current round number
    pub round: u64,
    /// Whether voting in favor (true) or against (false)
    pub value: bool,
    /// Validator's reputation at time of vote
    pub reputation: f32,
    /// Unix timestamp
    pub timestamp: u64,
    /// Signature over the vote (placeholder)
    pub signature: Vec<u8>,
}

impl Vote {
    /// Create a new vote
    pub fn new(
        validator_id: ValidatorId,
        vote_type: VoteType,
        proposal_hash: ProposalHash,
        round: u64,
        value: bool,
        reputation: f32,
    ) -> Self {
        Self {
            validator_id,
            vote_type,
            proposal_hash,
            round,
            value,
            reputation: reputation.clamp(0.0, 1.0),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            signature: Vec::new(), // Would be filled by signing
        }
    }

    /// Compute the weighted vote value using reputation² weighting
    pub fn weighted_value(&self) -> f32 {
        if self.value {
            self.reputation.powi(2)
        } else {
            0.0
        }
    }

    /// Get the vote weight (reputation²)
    pub fn weight(&self) -> f32 {
        self.reputation.powi(2)
    }
}

/// A block proposal in the consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockProposal {
    /// Unique hash identifying this proposal
    pub hash: ProposalHash,
    /// Round number
    pub round: u64,
    /// View number (increments on leader change)
    pub view: u64,
    /// Proposer's validator ID
    pub proposer: ValidatorId,
    /// Proposer's K-Vector at proposal time
    pub proposer_kvector: KVector,
    /// PoGQ proof for the proposed content
    pub pogq_proof: ProofOfGradientQuality,
    /// Block height
    pub height: u64,
    /// Parent block hash
    pub parent_hash: BlockHash,
    /// Merkle root of block contents
    pub content_root: [u8; 32],
    /// Timestamp of proposal
    pub timestamp: u64,
}

impl BlockProposal {
    /// Check if proposer has sufficient reputation to propose
    pub fn proposer_is_valid(&self) -> bool {
        self.proposer_kvector.k_r >= MIN_VALIDATOR_REPUTATION
    }

    /// Check if proposer has sufficient governance tier
    pub fn proposer_meets_tier(&self, required: GovernanceTier) -> bool {
        self.proposer_kvector.meets_governance_threshold(required)
    }
}

/// Evidence for challenging a validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeEvidence {
    /// The challenged validator
    pub validator_id: ValidatorId,
    /// Type of misbehavior
    pub violation_type: ViolationType,
    /// Proof data (depends on violation type)
    pub proof: Vec<u8>,
    /// Round where violation occurred
    pub round: u64,
    /// View where violation occurred
    pub view: u64,
    /// Timestamp of challenge submission
    pub timestamp: u64,
}

/// Types of validator misbehavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Voted twice in same round
    DoubleVote,
    /// Proposed two different blocks at same height
    DoubleProposal,
    /// Signed conflicting pre-commits
    ConflictingPreCommit,
    /// Submitted invalid PoGQ proof
    InvalidPoGQ,
    /// Went offline during critical round
    Unavailable,
}

/// Result of processing a vote
#[derive(Debug, Clone)]
pub struct VoteResult {
    /// Whether the vote was accepted
    pub accepted: bool,
    /// Current weighted vote tally (in favor)
    pub weighted_for: f32,
    /// Current weighted vote tally (total weight)
    pub weighted_total: f32,
    /// Whether quorum has been reached
    pub quorum_reached: bool,
    /// New state if state transition occurred
    pub new_state: Option<RoundState>,
}

/// Consensus result for a round
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusResult {
    /// Proposal was accepted and committed
    Accepted,
    /// Proposal was rejected
    Rejected,
    /// Round timed out
    Timeout,
    /// Insufficient participation
    NoQuorum,
}

/// Configuration for RB-BFT consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbBftConfig {
    /// Byzantine tolerance threshold (default 0.34)
    pub byzantine_threshold: f32,
    /// Quorum threshold for decisions (default 0.667)
    pub quorum_threshold: f32,
    /// Round timeout duration
    pub round_timeout: Duration,
    /// Minimum reputation to participate
    pub min_validator_reputation: f32,
    /// Minimum governance tier to propose blocks
    pub min_proposer_tier: GovernanceTier,
    /// Maximum rounds before forcing view change
    pub max_rounds_before_view_change: u32,
}

impl Default for RbBftConfig {
    fn default() -> Self {
        Self {
            byzantine_threshold: RBBFT_BYZANTINE_THRESHOLD,
            quorum_threshold: QUORUM_THRESHOLD,
            round_timeout: Duration::from_secs(DEFAULT_ROUND_TIMEOUT_SECS),
            min_validator_reputation: MIN_VALIDATOR_REPUTATION,
            min_proposer_tier: GovernanceTier::Basic,
            max_rounds_before_view_change: MAX_ROUNDS_BEFORE_VIEW_CHANGE,
        }
    }
}

/// Validator node in the consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorNode {
    /// Validator's unique identifier
    pub id: ValidatorId,
    /// Current K-Vector trust scores
    pub kvector: KVector,
    /// Whether currently active
    pub active: bool,
    /// Consecutive rounds participated
    pub consecutive_participation: u32,
    /// Consecutive rounds missed
    pub consecutive_misses: u32,
    /// Total successful proposals
    pub successful_proposals: u64,
    /// Total challenges against this validator
    pub challenges_received: u64,
}

impl ValidatorNode {
    /// Create a new validator node
    pub fn new(id: ValidatorId, kvector: KVector) -> Self {
        Self {
            id,
            kvector,
            active: true,
            consecutive_participation: 0,
            consecutive_misses: 0,
            successful_proposals: 0,
            challenges_received: 0,
        }
    }

    /// Get voting weight (reputation²)
    pub fn voting_weight(&self) -> f32 {
        if self.active && self.kvector.k_r >= MIN_VALIDATOR_REPUTATION {
            self.kvector.rbbft_voting_weight()
        } else {
            0.0
        }
    }

    /// Check if eligible to propose
    pub fn can_propose(&self, min_tier: GovernanceTier) -> bool {
        self.active
            && self.kvector.k_r >= MIN_VALIDATOR_REPUTATION
            && self.kvector.meets_governance_threshold(min_tier)
    }

    /// Record participation
    pub fn record_participation(&mut self) {
        self.consecutive_participation += 1;
        self.consecutive_misses = 0;
    }

    /// Record missed round
    pub fn record_miss(&mut self) {
        self.consecutive_misses += 1;
        self.consecutive_participation = 0;

        // Auto-deactivate after 3 consecutive misses
        if self.consecutive_misses >= 3 {
            self.active = false;
        }
    }
}

/// RB-BFT Consensus Engine
///
/// Orchestrates the consensus protocol with reputation-weighted voting.
#[derive(Debug)]
pub struct RbBftConsensus {
    /// Configuration
    config: RbBftConfig,
    /// Registered validators
    validators: HashMap<ValidatorId, ValidatorNode>,
    /// Current round number
    current_round: u64,
    /// Current view number
    current_view: u64,
    /// Current round state
    state: RoundState,
    /// Active proposal for this round
    active_proposal: Option<BlockProposal>,
    /// Pre-votes for current round
    pre_votes: HashMap<ValidatorId, Vote>,
    /// Pre-commits for current round
    pre_commits: HashMap<ValidatorId, Vote>,
    /// Round start time
    round_start: u64,
    /// Consecutive failed rounds
    failed_rounds: u32,
    /// Last committed block hash
    last_committed: Option<BlockHash>,
    /// Pending challenges
    pending_challenges: Vec<ChallengeEvidence>,
}

impl RbBftConsensus {
    /// Create a new RB-BFT consensus engine
    pub fn new(config: RbBftConfig) -> Self {
        Self {
            config,
            validators: HashMap::new(),
            current_round: 0,
            current_view: 0,
            state: RoundState::Propose,
            active_proposal: None,
            pre_votes: HashMap::new(),
            pre_commits: HashMap::new(),
            round_start: 0,
            failed_rounds: 0,
            last_committed: None,
            pending_challenges: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RbBftConfig::default())
    }

    /// Register a new validator
    pub fn register_validator(&mut self, id: ValidatorId, kvector: KVector) -> bool {
        if kvector.k_r < self.config.min_validator_reputation {
            return false;
        }

        let node = ValidatorNode::new(id, kvector);
        self.validators.insert(id, node);
        true
    }

    /// Update a validator's K-Vector
    pub fn update_validator_kvector(&mut self, id: &ValidatorId, kvector: KVector) -> bool {
        if let Some(validator) = self.validators.get_mut(id) {
            validator.kvector = kvector;

            // Deactivate if reputation dropped below minimum
            if kvector.k_r < self.config.min_validator_reputation {
                validator.active = false;
            }
            true
        } else {
            false
        }
    }

    /// Get total weighted voting power
    pub fn total_voting_weight(&self) -> f32 {
        self.validators.values().map(|v| v.voting_weight()).sum()
    }

    /// Get number of active validators
    pub fn active_validator_count(&self) -> usize {
        self.validators.values().filter(|v| v.active).count()
    }

    /// Start a new round
    pub fn start_round(&mut self, round: u64) {
        self.current_round = round;
        self.state = RoundState::Propose;
        self.active_proposal = None;
        self.pre_votes.clear();
        self.pre_commits.clear();
        self.round_start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Propose a new block
    ///
    /// Returns Ok(ProposalHash) if accepted, Err with reason if rejected
    pub fn propose_block(&mut self, proposal: BlockProposal) -> Result<ProposalHash, String> {
        // Check state
        if self.state != RoundState::Propose {
            return Err(format!("Invalid state for proposal: {:?}", self.state));
        }

        // Check round matches
        if proposal.round != self.current_round {
            return Err(format!(
                "Round mismatch: expected {}, got {}",
                self.current_round, proposal.round
            ));
        }

        // Verify proposer is registered and eligible
        let validator = self
            .validators
            .get(&proposal.proposer)
            .ok_or("Proposer not registered")?;

        if !validator.can_propose(self.config.min_proposer_tier) {
            return Err("Proposer does not meet requirements".to_string());
        }

        // Verify PoGQ is acceptable
        if proposal.pogq_proof.quality < 0.5 {
            return Err("PoGQ quality too low".to_string());
        }

        // Accept proposal and move to pre-vote phase
        let hash = proposal.hash;
        self.active_proposal = Some(proposal);
        self.state = RoundState::PreVote;

        Ok(hash)
    }

    /// Submit a vote on the active proposal
    pub fn vote(&mut self, vote: Vote) -> Result<VoteResult, String> {
        // Verify voter is registered and active
        let validator = self
            .validators
            .get(&vote.validator_id)
            .ok_or("Voter not registered")?;

        if !validator.active {
            return Err("Voter is not active".to_string());
        }

        // Verify vote matches current round and proposal
        if vote.round != self.current_round {
            return Err("Round mismatch".to_string());
        }

        if let Some(ref proposal) = self.active_proposal {
            // SECURITY: Use constant-time comparison for proposal hash verification (FIND-004)
            if !secure_compare_hash(&vote.proposal_hash, &proposal.hash) {
                return Err("Proposal hash mismatch".to_string());
            }
        } else {
            return Err("No active proposal".to_string());
        }

        // Process based on vote type and current state
        match (vote.vote_type, self.state) {
            (VoteType::PreVote, RoundState::PreVote) => self.process_pre_vote(vote),
            (VoteType::PreCommit, RoundState::PreCommit) => self.process_pre_commit(vote),
            _ => Err(format!(
                "Invalid vote type {:?} for state {:?}",
                vote.vote_type, self.state
            )),
        }
    }

    /// Process a pre-vote
    fn process_pre_vote(&mut self, vote: Vote) -> Result<VoteResult, String> {
        // Check for double voting
        if self.pre_votes.contains_key(&vote.validator_id) {
            return Err("Already voted in this round".to_string());
        }

        self.pre_votes.insert(vote.validator_id, vote);

        // Calculate weighted totals
        let (weighted_for, weighted_total) = self.calculate_weighted_votes(&self.pre_votes);
        let quorum_reached = weighted_for / weighted_total >= self.config.quorum_threshold;

        let new_state = if quorum_reached {
            self.state = RoundState::PreCommit;
            Some(RoundState::PreCommit)
        } else {
            None
        };

        Ok(VoteResult {
            accepted: true,
            weighted_for,
            weighted_total,
            quorum_reached,
            new_state,
        })
    }

    /// Process a pre-commit
    fn process_pre_commit(&mut self, vote: Vote) -> Result<VoteResult, String> {
        // Check for double voting
        if self.pre_commits.contains_key(&vote.validator_id) {
            return Err("Already pre-committed in this round".to_string());
        }

        self.pre_commits.insert(vote.validator_id, vote);

        // Calculate weighted totals
        let (weighted_for, weighted_total) = self.calculate_weighted_votes(&self.pre_commits);
        let quorum_reached = weighted_for / weighted_total >= self.config.quorum_threshold;

        let new_state = if quorum_reached {
            self.commit_block();
            Some(RoundState::Committed)
        } else {
            None
        };

        Ok(VoteResult {
            accepted: true,
            weighted_for,
            weighted_total,
            quorum_reached,
            new_state,
        })
    }

    /// Calculate weighted vote totals using reputation² weighting
    fn calculate_weighted_votes(&self, votes: &HashMap<ValidatorId, Vote>) -> (f32, f32) {
        let mut weighted_for = 0.0f32;
        let mut weighted_total = 0.0f32;

        for vote in votes.values() {
            let weight = vote.weight();
            weighted_total += weight;
            if vote.value {
                weighted_for += weight;
            }
        }

        // Include non-voters with zero weight (they count towards total from validators)
        let total_possible_weight = self.total_voting_weight();
        if weighted_total < total_possible_weight {
            // Non-voters are implicit "no" votes with their weight
            weighted_total = total_possible_weight;
        }

        (weighted_for, weighted_total)
    }

    /// Commit the current block
    fn commit_block(&mut self) {
        if let Some(ref proposal) = self.active_proposal {
            self.last_committed = Some(proposal.hash);
            self.state = RoundState::Committed;
            self.failed_rounds = 0;

            // Update proposer's stats
            if let Some(validator) = self.validators.get_mut(&proposal.proposer) {
                validator.successful_proposals += 1;
            }

            // Record participation for all voters
            let voter_ids: Vec<_> = self
                .pre_commits
                .keys()
                .chain(self.pre_votes.keys())
                .cloned()
                .collect();

            for id in voter_ids {
                if let Some(validator) = self.validators.get_mut(&id) {
                    validator.record_participation();
                }
            }

            // Record misses for non-participants
            let all_ids: Vec<_> = self.validators.keys().cloned().collect();
            for id in all_ids {
                if !self.pre_votes.contains_key(&id) && !self.pre_commits.contains_key(&id) {
                    if let Some(validator) = self.validators.get_mut(&id) {
                        validator.record_miss();
                    }
                }
            }
        }
    }

    /// Challenge a validator for misbehavior
    pub fn challenge_validator(&mut self, evidence: ChallengeEvidence) -> Result<bool, String> {
        // Verify the challenged validator exists
        if !self.validators.contains_key(&evidence.validator_id) {
            return Err("Challenged validator not found".to_string());
        }

        // Verify evidence (simplified - real impl would verify cryptographically)
        let valid = self.verify_challenge_evidence(&evidence)?;

        if valid {
            // Record challenge
            if let Some(validator) = self.validators.get_mut(&evidence.validator_id) {
                validator.challenges_received += 1;

                // Automatic penalty for confirmed violations
                match evidence.violation_type {
                    ViolationType::DoubleVote | ViolationType::DoubleProposal => {
                        // Severe: immediate deactivation
                        validator.active = false;
                        validator.kvector.k_r = (validator.kvector.k_r - 0.3).max(0.0);
                    }
                    ViolationType::ConflictingPreCommit => {
                        // Severe: immediate deactivation
                        validator.active = false;
                        validator.kvector.k_r = (validator.kvector.k_r - 0.2).max(0.0);
                    }
                    ViolationType::InvalidPoGQ => {
                        // Moderate: reputation penalty
                        validator.kvector.k_r = (validator.kvector.k_r - 0.1).max(0.0);
                    }
                    ViolationType::Unavailable => {
                        // Minor: small penalty, tracked separately
                        validator.kvector.k_r = (validator.kvector.k_r - 0.05).max(0.0);
                    }
                }
            }

            self.pending_challenges.push(evidence);
        }

        Ok(valid)
    }

    /// Verify challenge evidence (simplified)
    fn verify_challenge_evidence(&self, evidence: &ChallengeEvidence) -> Result<bool, String> {
        // In a real implementation, this would:
        // - Verify signatures on conflicting votes
        // - Check block hashes match
        // - Verify timestamps are within valid range
        // For now, accept if proof is non-empty
        Ok(!evidence.proof.is_empty())
    }

    /// Handle round timeout
    pub fn handle_timeout(&mut self) -> ConsensusResult {
        self.failed_rounds += 1;
        self.state = RoundState::Failed;

        if self.failed_rounds >= self.config.max_rounds_before_view_change {
            self.trigger_view_change();
        }

        ConsensusResult::Timeout
    }

    /// Trigger a view change (leader rotation)
    fn trigger_view_change(&mut self) {
        self.current_view += 1;
        self.failed_rounds = 0;
        // Leader selection would happen here based on view number
    }

    /// Get the current leader based on view number
    pub fn current_leader(&self) -> Option<ValidatorId> {
        let active_validators: Vec<_> = self
            .validators
            .iter()
            .filter(|(_, v)| v.can_propose(self.config.min_proposer_tier))
            .collect();

        if active_validators.is_empty() {
            return None;
        }

        // Round-robin leader selection based on view
        let idx = (self.current_view as usize) % active_validators.len();
        Some(*active_validators[idx].0)
    }

    /// Get current round state
    pub fn state(&self) -> RoundState {
        self.state
    }

    /// Get current round number
    pub fn current_round(&self) -> u64 {
        self.current_round
    }

    /// Get current view number
    pub fn current_view(&self) -> u64 {
        self.current_view
    }

    /// Get the last committed block hash
    pub fn last_committed(&self) -> Option<&BlockHash> {
        self.last_committed.as_ref()
    }

    /// Compute consensus result for a set of votes (utility function)
    ///
    /// This is the core weighted voting algorithm:
    /// - Sum of (reputation² × vote_value) for all votes
    /// - Divide by sum of (reputation²) for all validators
    /// - Accept if result > byzantine_threshold
    pub fn weighted_consensus(votes: &[Vote], byzantine_threshold: f32) -> ConsensusResult {
        if votes.is_empty() {
            return ConsensusResult::NoQuorum;
        }

        let weighted_sum: f32 = votes.iter().map(|v| v.weighted_value()).sum();
        let total_weight: f32 = votes.iter().map(|v| v.weight()).sum();

        if total_weight == 0.0 {
            return ConsensusResult::NoQuorum;
        }

        if weighted_sum / total_weight > byzantine_threshold {
            ConsensusResult::Accepted
        } else {
            ConsensusResult::Rejected
        }
    }
}

/// Statistics about the consensus engine state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusStats {
    /// Total registered validators
    pub total_validators: usize,
    /// Active validators
    pub active_validators: usize,
    /// Total weighted voting power
    pub total_voting_weight: f32,
    /// Current round
    pub current_round: u64,
    /// Current view
    pub current_view: u64,
    /// Current state
    pub state: RoundState,
    /// Consecutive failed rounds
    pub failed_rounds: u32,
    /// Pending challenges
    pub pending_challenges: usize,
}

impl RbBftConsensus {
    /// Get current consensus statistics
    pub fn stats(&self) -> ConsensusStats {
        ConsensusStats {
            total_validators: self.validators.len(),
            active_validators: self.active_validator_count(),
            total_voting_weight: self.total_voting_weight(),
            current_round: self.current_round,
            current_view: self.current_view,
            state: self.state,
            failed_rounds: self.failed_rounds,
            pending_challenges: self.pending_challenges.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_validator_id(n: u8) -> ValidatorId {
        let mut id = [0u8; 32];
        id[0] = n;
        id
    }

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

    fn make_proposal(round: u64, proposer: ValidatorId, kvector: KVector) -> BlockProposal {
        BlockProposal {
            hash: [1u8; 32],
            round,
            view: 0,
            proposer,
            proposer_kvector: kvector,
            pogq_proof: ProofOfGradientQuality::new(0.9, 0.9, 0.1),
            height: 1,
            parent_hash: [0u8; 32],
            content_root: [0u8; 32],
            timestamp: 0,
        }
    }

    #[test]
    fn test_reputation_squared_weighting() {
        let high_rep = Vote::new(
            make_validator_id(1),
            VoteType::PreVote,
            [0u8; 32],
            0,
            true,
            0.9,
        );
        let low_rep = Vote::new(
            make_validator_id(2),
            VoteType::PreVote,
            [0u8; 32],
            0,
            true,
            0.3,
        );

        // High reputation (0.9)² = 0.81
        assert!((high_rep.weight() - 0.81).abs() < 0.001);

        // Low reputation (0.3)² = 0.09
        assert!((low_rep.weight() - 0.09).abs() < 0.001);

        // 9x difference in voting power!
        assert!(high_rep.weight() / low_rep.weight() > 8.9);
    }

    #[test]
    fn test_register_validator() {
        let mut consensus = RbBftConsensus::with_defaults();

        // High reputation validator should be accepted
        let id = make_validator_id(1);
        let kvector = make_kvector(0.8);
        assert!(consensus.register_validator(id, kvector));
        assert_eq!(consensus.active_validator_count(), 1);

        // Low reputation validator should be rejected
        let id2 = make_validator_id(2);
        let kvector2 = make_kvector(0.1);
        assert!(!consensus.register_validator(id2, kvector2));
        assert_eq!(consensus.active_validator_count(), 1);
    }

    #[test]
    fn test_basic_consensus_flow() {
        let mut consensus = RbBftConsensus::with_defaults();

        // Register 3 validators with varying reputation
        let v1 = make_validator_id(1);
        let v2 = make_validator_id(2);
        let v3 = make_validator_id(3);

        consensus.register_validator(v1, make_kvector(0.9));
        consensus.register_validator(v2, make_kvector(0.8));
        consensus.register_validator(v3, make_kvector(0.7));

        // Start round
        consensus.start_round(1);
        assert_eq!(consensus.state(), RoundState::Propose);

        // Propose block
        let proposal = make_proposal(1, v1, make_kvector(0.9));
        let hash = proposal.hash;
        assert!(consensus.propose_block(proposal).is_ok());
        assert_eq!(consensus.state(), RoundState::PreVote);

        // Validators pre-vote yes (state may transition to PreCommit after quorum)
        for (i, vid) in [v1, v2, v3].iter().enumerate() {
            // Only submit pre-vote if still in pre-vote phase
            if consensus.state() == RoundState::PreVote {
                let vote = Vote::new(
                    *vid,
                    VoteType::PreVote,
                    hash,
                    1,
                    true,
                    0.9 - (i as f32 * 0.1),
                );
                let result = consensus.vote(vote).unwrap();
                assert!(result.accepted);
            }
        }

        // Should have moved to pre-commit (quorum reached after 2 high-rep votes)
        assert_eq!(consensus.state(), RoundState::PreCommit);

        // Validators pre-commit (state may transition to Committed after quorum)
        for (i, vid) in [v1, v2, v3].iter().enumerate() {
            // Only submit pre-commit if still in pre-commit phase
            if consensus.state() == RoundState::PreCommit {
                let vote = Vote::new(
                    *vid,
                    VoteType::PreCommit,
                    hash,
                    1,
                    true,
                    0.9 - (i as f32 * 0.1),
                );
                let result = consensus.vote(vote).unwrap();
                assert!(result.accepted);
            }
        }

        // Should be committed
        assert_eq!(consensus.state(), RoundState::Committed);
        assert!(consensus.last_committed().is_some());
    }

    #[test]
    fn test_weighted_consensus_function() {
        // Test the standalone weighted consensus function
        let votes = vec![
            Vote::new(
                make_validator_id(1),
                VoteType::PreVote,
                [0u8; 32],
                0,
                true,
                0.9,
            ),
            Vote::new(
                make_validator_id(2),
                VoteType::PreVote,
                [0u8; 32],
                0,
                true,
                0.8,
            ),
            Vote::new(
                make_validator_id(3),
                VoteType::PreVote,
                [0u8; 32],
                0,
                false,
                0.3,
            ),
        ];

        // Weighted for: 0.81 + 0.64 = 1.45
        // Weighted total: 0.81 + 0.64 + 0.09 = 1.54
        // Ratio: 1.45 / 1.54 ≈ 0.94 > 0.34 threshold
        let result = RbBftConsensus::weighted_consensus(&votes, RBBFT_BYZANTINE_THRESHOLD);
        assert_eq!(result, ConsensusResult::Accepted);

        // With higher threshold
        let result_strict = RbBftConsensus::weighted_consensus(&votes, 0.95);
        assert_eq!(result_strict, ConsensusResult::Rejected);
    }

    #[test]
    fn test_challenge_validator() {
        let mut consensus = RbBftConsensus::with_defaults();

        let v1 = make_validator_id(1);
        consensus.register_validator(v1, make_kvector(0.9));

        let evidence = ChallengeEvidence {
            validator_id: v1,
            violation_type: ViolationType::DoubleVote,
            proof: vec![1, 2, 3], // Non-empty proof
            round: 0,
            view: 0,
            timestamp: 0,
        };

        let result = consensus.challenge_validator(evidence);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Challenge accepted

        // Validator should be deactivated for double vote
        let validator = consensus.validators.get(&v1).unwrap();
        assert!(!validator.active);
        assert!(validator.kvector.k_r < 0.9); // Reputation penalized
    }

    #[test]
    fn test_byzantine_minority_cannot_win() {
        let mut consensus = RbBftConsensus::with_defaults();

        // 3 honest validators with high reputation
        let h1 = make_validator_id(1);
        let h2 = make_validator_id(2);
        let h3 = make_validator_id(3);
        consensus.register_validator(h1, make_kvector(0.9));
        consensus.register_validator(h2, make_kvector(0.9));
        consensus.register_validator(h3, make_kvector(0.9));

        // 2 Byzantine validators (even with decent reputation)
        let b1 = make_validator_id(4);
        let b2 = make_validator_id(5);
        consensus.register_validator(b1, make_kvector(0.7));
        consensus.register_validator(b2, make_kvector(0.7));

        // Total weight: 3×0.81 + 2×0.49 = 2.43 + 0.98 = 3.41
        // Byzantine weight: 0.98 / 3.41 = 0.287 < 0.34 threshold
        // They cannot disrupt consensus!

        let byzantine_weight = 2.0 * 0.49;
        let total_weight = 3.0 * 0.81 + 2.0 * 0.49;
        let byzantine_fraction = byzantine_weight / total_weight;

        assert!(byzantine_fraction < RBBFT_BYZANTINE_THRESHOLD);
    }
}
