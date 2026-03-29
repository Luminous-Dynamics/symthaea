// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Main RB-BFT consensus engine

use serde::{Deserialize, Serialize};

use crate::crypto::ValidatorKeypair;
use crate::error::{ConsensusError, ConsensusResult};
use crate::proposal::Proposal;
use crate::round::{RoundManager, RoundState};
use crate::slashing::{SlashableOffense, SlashingManager};
use crate::validator::{ValidatorNode, ValidatorSet};
use crate::vote::{Vote, VoteDecision};
use crate::{BYZANTINE_TOLERANCE, DEFAULT_ROUND_TIMEOUT_MS, MIN_VALIDATORS};

/// Result of consensus
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusOutcome {
    /// Consensus reached - proposal accepted
    Accepted {
        round: u64,
        proposal_id: String,
        result_hash: String,
    },
    /// Consensus not reached - proposal rejected
    Rejected {
        round: u64,
        proposal_id: String,
        reason: String,
    },
    /// Round timed out
    TimedOut { round: u64 },
    /// Round skipped (no proposal)
    Skipped { round: u64 },
}

/// Result of batch vote processing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchVoteResult {
    /// Number of votes accepted
    pub accepted_count: usize,
    /// Number of votes rejected
    pub rejected_count: usize,
    /// Details of rejected votes (voter_id, reason)
    pub rejected_details: Vec<(String, String)>,
}

impl BatchVoteResult {
    /// Check if all votes were accepted
    pub fn all_accepted(&self) -> bool {
        self.rejected_count == 0
    }

    /// Total votes processed
    pub fn total(&self) -> usize {
        self.accepted_count + self.rejected_count
    }
}

/// Configuration for the consensus engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Timeout for rounds in milliseconds
    pub round_timeout_ms: u64,
    /// Byzantine tolerance threshold (default 0.45)
    pub byzantine_tolerance: f32,
    /// Minimum validators required
    pub min_validators: usize,
    /// Minimum confirmations for slashing
    pub slashing_confirmations: usize,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            round_timeout_ms: DEFAULT_ROUND_TIMEOUT_MS,
            byzantine_tolerance: BYZANTINE_TOLERANCE,
            min_validators: MIN_VALIDATORS,
            slashing_confirmations: 3,
        }
    }
}

/// The main RB-BFT consensus engine
#[derive(Debug, Serialize, Deserialize)]
pub struct RbBftConsensus {
    /// Configuration
    config: ConsensusConfig,
    /// Validator set
    validators: ValidatorSet,
    /// Round manager
    rounds: RoundManager,
    /// Slashing manager
    slashing: SlashingManager,
    /// Our validator ID (if we're a validator)
    our_id: Option<String>,
}

impl RbBftConsensus {
    /// Create a new consensus engine
    pub fn new(config: ConsensusConfig) -> Self {
        Self {
            rounds: RoundManager::new(config.round_timeout_ms),
            slashing: SlashingManager::new(config.slashing_confirmations),
            config,
            validators: ValidatorSet::new(),
            our_id: None,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ConsensusConfig::default())
    }

    /// Set our validator ID
    pub fn set_our_id(&mut self, id: String) {
        self.our_id = Some(id);
    }

    /// Add a validator to the set
    pub fn add_validator(&mut self, validator: ValidatorNode) {
        self.validators.add(validator);
    }

    /// Get the validator set
    pub fn validators(&self) -> &ValidatorSet {
        &self.validators
    }

    /// Get mutable validator set
    pub fn validators_mut(&mut self) -> &mut ValidatorSet {
        &mut self.validators
    }

    /// Start a new consensus round
    pub fn start_round(&mut self) -> ConsensusResult<u64> {
        self.validators.can_reach_consensus()?;

        let round = self.rounds.start_new_round(&self.validators)?;
        Ok(round.number)
    }

    /// Check if we are the leader for the current round
    pub fn are_we_leader(&self) -> bool {
        if let (Some(ref our_id), Some(round)) = (&self.our_id, self.rounds.active_round()) {
            &round.leader == our_id
        } else {
            false
        }
    }

    /// Submit a proposal (must be leader)
    pub fn propose(&mut self, content: Vec<u8>, content_hash: String) -> ConsensusResult<Proposal> {
        let our_id = self.our_id.clone().ok_or(ConsensusError::Internal(
            "Our validator ID not set".to_string(),
        ))?;

        // Get parent hash first (immutable borrow)
        let parent_hash = self.rounds.history()
            .last()
            .and_then(|r| r.result_hash.clone())
            .unwrap_or_else(|| "genesis".to_string());

        let threshold = self.validators.consensus_threshold();

        let round = self.rounds.active_round_mut().ok_or(ConsensusError::InvalidRoundState {
            expected: "active round".to_string(),
            actual: "no active round".to_string(),
        })?;

        if round.leader != our_id {
            return Err(ConsensusError::NotLeader {
                round: round.number,
                leader: round.leader.clone(),
            });
        }

        let proposal = Proposal::new(
            round.number,
            our_id,
            content,
            content_hash,
            parent_hash,
        );

        // Note: In production deployments, call sign_proposal() separately
        // or use propose_signed() which takes a keypair
        // proposal remains unsigned here for backwards compatibility

        round.submit_proposal(proposal.clone(), threshold)?;

        Ok(proposal)
    }

    /// Create and sign a proposal as the round leader
    ///
    /// This is the production method that creates cryptographically signed proposals.
    pub fn propose_signed(
        &mut self,
        content: Vec<u8>,
        content_hash: String,
        parent_hash: String,
        keypair: &ValidatorKeypair,
    ) -> ConsensusResult<Proposal> {
        let our_id = keypair.public_key_hex();

        // Verify we're the expected leader
        let round = self.rounds.active_round().ok_or(ConsensusError::InvalidRoundState {
            expected: "active round".to_string(),
            actual: "no active round".to_string(),
        })?;

        if round.state != RoundState::WaitingForProposal {
            return Err(ConsensusError::InvalidRoundState {
                expected: "WaitingForProposal".to_string(),
                actual: format!("{:?}", round.state),
            });
        }

        let leader = self.validators.select_leader(round.number);
        if leader.map(|l| &l.id) != Some(&our_id) {
            return Err(ConsensusError::NotLeader {
                round: round.number,
                leader: leader.map(|l| l.id.clone()).unwrap_or_else(|| "none".to_string()),
            });
        }

        let threshold = self.validators.consensus_threshold();

        let round = self.rounds.active_round_mut().ok_or(ConsensusError::InvalidRoundState {
            expected: "active round".to_string(),
            actual: "no active round".to_string(),
        })?;

        let mut proposal = Proposal::new(
            round.number,
            our_id,
            content,
            content_hash,
            parent_hash,
        );

        // Sign with real ed25519 signature
        proposal.sign(keypair)?;

        round.submit_proposal(proposal.clone(), threshold)?;

        Ok(proposal)
    }

    /// Submit a proposal from an external source (for non-leader nodes)
    pub fn receive_proposal(&mut self, proposal: Proposal) -> ConsensusResult<()> {
        let round = self.rounds.active_round_mut().ok_or(ConsensusError::InvalidRoundState {
            expected: "active round".to_string(),
            actual: "no active round".to_string(),
        })?;

        let threshold = self.validators.consensus_threshold();
        round.submit_proposal(proposal, threshold)
    }

    /// Cast a vote on the current proposal
    pub fn vote(&mut self, decision: VoteDecision, reason: Option<String>) -> ConsensusResult<Vote> {
        let our_id = self.our_id.clone().ok_or(ConsensusError::Internal(
            "Our validator ID not set".to_string(),
        ))?;

        let validator = self.validators.get(&our_id).ok_or(ConsensusError::ValidatorNotFound {
            validator_id: our_id.clone(),
        })?;

        validator.can_participate()?;

        let round = self.rounds.active_round_mut().ok_or(ConsensusError::InvalidRoundState {
            expected: "active round".to_string(),
            actual: "no active round".to_string(),
        })?;

        if round.state != RoundState::Voting {
            return Err(ConsensusError::InvalidRoundState {
                expected: "Voting".to_string(),
                actual: format!("{:?}", round.state),
            });
        }

        let proposal = round.proposal.as_ref().ok_or(ConsensusError::InvalidRoundState {
            expected: "proposal exists".to_string(),
            actual: "no proposal".to_string(),
        })?;

        // Check for duplicate vote
        if round.votes.has_voted(&our_id) {
            return Err(ConsensusError::DuplicateVote {
                validator: our_id,
                round: round.number,
            });
        }

        // Create vote
        let vote = match decision {
            VoteDecision::Approve => Vote::approve(
                proposal.id.clone(),
                round.number,
                our_id,
                validator.reputation(),
            ),
            VoteDecision::Reject => Vote::reject(
                proposal.id.clone(),
                round.number,
                our_id,
                validator.reputation(),
                reason.unwrap_or_else(|| "No reason provided".to_string()),
            ),
            VoteDecision::Abstain => Vote::abstain(
                proposal.id.clone(),
                round.number,
                our_id,
                validator.reputation(),
            ),
        };

        // Note: Vote is unsigned. For production, use vote_signed() instead
        round.votes.add_vote(vote.clone());

        // Update voting state
        if let Some(ref mut vs) = round.voting_state {
            vs.weighted_votes_for = round.votes.weighted_approvals();
            vs.weighted_votes_against = round.votes.weighted_rejections();
            vs.voter_count = round.votes.vote_count();
        }

        Ok(vote)
    }

    /// Cast a cryptographically signed vote on the current proposal
    ///
    /// This is the production method that creates properly signed votes.
    pub fn vote_signed(
        &mut self,
        decision: VoteDecision,
        reason: Option<String>,
        keypair: &ValidatorKeypair,
    ) -> ConsensusResult<Vote> {
        let our_id = keypair.public_key_hex();

        let validator = self.validators.get(&our_id).ok_or(ConsensusError::ValidatorNotFound {
            validator_id: our_id.clone(),
        })?;

        validator.can_participate()?;

        let round = self.rounds.active_round_mut().ok_or(ConsensusError::InvalidRoundState {
            expected: "active round".to_string(),
            actual: "no active round".to_string(),
        })?;

        if round.state != RoundState::Voting {
            return Err(ConsensusError::InvalidRoundState {
                expected: "Voting".to_string(),
                actual: format!("{:?}", round.state),
            });
        }

        let proposal = round.proposal.as_ref().ok_or(ConsensusError::InvalidRoundState {
            expected: "proposal exists".to_string(),
            actual: "no proposal".to_string(),
        })?;

        // Check for duplicate vote
        if round.votes.has_voted(&our_id) {
            return Err(ConsensusError::DuplicateVote {
                validator: our_id,
                round: round.number,
            });
        }

        // Create and sign vote
        let mut vote = match decision {
            VoteDecision::Approve => Vote::approve(
                proposal.id.clone(),
                round.number,
                our_id,
                validator.reputation(),
            ),
            VoteDecision::Reject => Vote::reject(
                proposal.id.clone(),
                round.number,
                our_id,
                validator.reputation(),
                reason.unwrap_or_else(|| "No reason provided".to_string()),
            ),
            VoteDecision::Abstain => Vote::abstain(
                proposal.id.clone(),
                round.number,
                our_id,
                validator.reputation(),
            ),
        };

        // Sign with real ed25519 signature
        vote.sign(keypair)?;

        round.votes.add_vote(vote.clone());

        // Update voting state
        if let Some(ref mut vs) = round.voting_state {
            vs.weighted_votes_for = round.votes.weighted_approvals();
            vs.weighted_votes_against = round.votes.weighted_rejections();
            vs.voter_count = round.votes.vote_count();
        }

        Ok(vote)
    }

    /// Receive a vote from another validator
    pub fn receive_vote(&mut self, vote: Vote) -> ConsensusResult<()> {
        // Validate vote
        if !vote.is_valid() {
            return Err(ConsensusError::InvalidVote {
                validator: vote.voter.clone(),
                reason: "Vote validation failed".to_string(),
            });
        }

        // Check validator exists
        let validator = self.validators.get(&vote.voter).ok_or(ConsensusError::ValidatorNotFound {
            validator_id: vote.voter.clone(),
        })?;

        // Verify reputation matches (tight tolerance to prevent vote weight manipulation)
        // H-04: Tightened from 0.01 to 0.001 to prevent Byzantine vote weight gaming
        const REPUTATION_TOLERANCE: f32 = 0.001;
        if (vote.reputation - validator.reputation()).abs() > REPUTATION_TOLERANCE {
            return Err(ConsensusError::InvalidVote {
                validator: vote.voter.clone(),
                reason: format!(
                    "Reputation mismatch: vote={:.4}, validator={:.4}, tolerance={:.4}",
                    vote.reputation, validator.reputation(), REPUTATION_TOLERANCE
                ),
            });
        }

        let round = self.rounds.active_round_mut().ok_or(ConsensusError::InvalidRoundState {
            expected: "active round".to_string(),
            actual: "no active round".to_string(),
        })?;

        if vote.round != round.number {
            return Err(ConsensusError::InvalidVote {
                validator: vote.voter.clone(),
                reason: format!("Wrong round: expected {}, got {}", round.number, vote.round),
            });
        }

        // Check for double voting
        if round.votes.has_voted(&vote.voter) {
            // Potential double vote - check if it's actually different
            let existing = round.votes.votes().iter().find(|v| v.voter == vote.voter);
            if let Some(existing_vote) = existing {
                if existing_vote.decision != vote.decision {
                    // Double vote detected!
                    let evidence = crate::vote::DoubleVoteEvidence::new(
                        existing_vote.clone(),
                        vote.clone(),
                    );
                    if let Some(e) = evidence {
                        let offense = SlashableOffense::DoubleVoting(Box::new(e));
                        self.slashing.report_offense(
                            offense,
                            self.our_id.clone().unwrap_or_else(|| "system".to_string()),
                        );
                    }
                }
            }
            return Err(ConsensusError::DuplicateVote {
                validator: vote.voter,
                round: round.number,
            });
        }

        round.votes.add_vote(vote);

        // Update voting state
        if let Some(ref mut vs) = round.voting_state {
            vs.weighted_votes_for = round.votes.weighted_approvals();
            vs.weighted_votes_against = round.votes.weighted_rejections();
            vs.voter_count = round.votes.vote_count();
        }

        Ok(())
    }

    /// Receive multiple votes and verify them using batch verification
    ///
    /// This is more efficient than calling receive_vote() multiple times as it uses
    /// batch ed25519 signature verification which is ~2x faster.
    pub fn receive_votes_batch(&mut self, votes: Vec<Vote>) -> ConsensusResult<BatchVoteResult> {
        if votes.is_empty() {
            return Ok(BatchVoteResult::default());
        }

        let round_num = self.rounds.active_round()
            .ok_or(ConsensusError::InvalidRoundState {
                expected: "active round".to_string(),
                actual: "no active round".to_string(),
            })?.number;

        // Pre-validate votes before batch signature verification
        let mut valid_votes: Vec<Vote> = Vec::new();
        let mut rejected: Vec<(String, String)> = Vec::new();

        for vote in votes {
            // Basic validation
            if !vote.is_valid() {
                rejected.push((vote.voter.clone(), "Vote validation failed".to_string()));
                continue;
            }

            // Check validator exists
            let validator = match self.validators.get(&vote.voter) {
                Some(v) => v,
                None => {
                    rejected.push((vote.voter.clone(), "Validator not found".to_string()));
                    continue;
                }
            };

            // Verify reputation matches
            if (vote.reputation - validator.reputation()).abs() > 0.01 {
                rejected.push((vote.voter.clone(), "Reputation mismatch".to_string()));
                continue;
            }

            // Check round
            if vote.round != round_num {
                rejected.push((vote.voter.clone(), format!("Wrong round: {}", vote.round)));
                continue;
            }

            // Check for double voting (check against existing votes)
            let round = self.rounds.active_round().expect("active round exists during vote processing");
            if round.votes.has_voted(&vote.voter) {
                rejected.push((vote.voter.clone(), "Already voted".to_string()));
                continue;
            }

            valid_votes.push(vote);
        }

        // Now batch verify signatures of valid votes
        if !valid_votes.is_empty() {
            let messages: Vec<Vec<u8>> = valid_votes.iter()
                .map(|v| v.signable_bytes())
                .collect();

            let batch_result = crate::crypto_ops::batch_verify_vote_signatures(&valid_votes, &messages)?;

            // Process verification results
            let mut accepted_count = 0;
            let round = self.rounds.active_round_mut().expect("active round exists during vote processing");

            for (i, vote) in valid_votes.into_iter().enumerate() {
                if batch_result.is_valid_by_index(i) {
                    // Check again for double voting (might have been added by another valid vote)
                    if !round.votes.has_voted(&vote.voter) {
                        round.votes.add_vote(vote);
                        accepted_count += 1;
                    }
                } else {
                    let reason = batch_result.get_error_by_index(i)
                        .unwrap_or_else(|| "Signature verification failed".to_string());
                    rejected.push((vote.voter, reason));
                }
            }

            // Update voting state
            if let Some(ref mut vs) = round.voting_state {
                vs.weighted_votes_for = round.votes.weighted_approvals();
                vs.weighted_votes_against = round.votes.weighted_rejections();
                vs.voter_count = round.votes.vote_count();
            }

            return Ok(BatchVoteResult {
                accepted_count,
                rejected_count: rejected.len(),
                rejected_details: rejected,
            });
        }

        Ok(BatchVoteResult {
            accepted_count: 0,
            rejected_count: rejected.len(),
            rejected_details: rejected,
        })
    }

    /// Check if consensus has been reached in the current round
    pub fn check_consensus(&mut self) -> ConsensusResult<Option<ConsensusOutcome>> {
        let threshold = self.validators.consensus_threshold();
        let total_weight = self.validators.total_weight();

        let round = self.rounds.active_round_mut().ok_or(ConsensusError::InvalidRoundState {
            expected: "active round".to_string(),
            actual: "no active round".to_string(),
        })?;

        // Check timeout
        if round.is_timed_out() {
            let round_num = round.number;
            round.fail("timeout");
            return Ok(Some(ConsensusOutcome::TimedOut { round: round_num }));
        }

        // Need to be in voting state
        if round.state != RoundState::Voting {
            return Ok(None);
        }

        // Check if consensus reached
        if round.votes.consensus_reached(threshold) {
            let proposal = round.proposal.as_ref().ok_or(ConsensusError::InvalidRoundState {
                expected: "proposal exists for consensus".to_string(),
                actual: "no proposal in round".to_string(),
            })?;
            let result_hash = proposal.content_hash.clone();
            let proposal_id = proposal.id.clone();
            let round_num = round.number;

            round.commit(result_hash.clone());

            // Collect vote info for reputation updates
            let vote_outcomes: Vec<(String, bool)> = round.votes.votes()
                .iter()
                .map(|v| (v.voter.clone(), v.decision.is_approval()))
                .collect();

            // Update validator reputations (borrow ends when we stop using round)
            for (voter, is_approval) in vote_outcomes {
                if let Some(v) = self.validators.get_mut(&voter) {
                    v.update_reputation(is_approval);
                }
            }

            return Ok(Some(ConsensusOutcome::Accepted {
                round: round_num,
                proposal_id,
                result_hash,
            }));
        }

        // Check if consensus is impossible
        if round.votes.consensus_impossible(total_weight, threshold) {
            let proposal = round.proposal.as_ref().ok_or(ConsensusError::InvalidRoundState {
                expected: "proposal exists for rejection".to_string(),
                actual: "no proposal in round".to_string(),
            })?;
            let proposal_id = proposal.id.clone();
            let round_num = round.number;
            round.fail("consensus impossible");

            return Ok(Some(ConsensusOutcome::Rejected {
                round: round_num,
                proposal_id,
                reason: "Insufficient weighted votes".to_string(),
            }));
        }

        // Still waiting for more votes
        Ok(None)
    }

    /// Get current round number
    pub fn current_round(&self) -> u64 {
        self.rounds.current_round()
    }

    /// Get round statistics
    pub fn stats(&self) -> crate::round::RoundStats {
        self.rounds.stats()
    }

    /// Get slashing events
    pub fn slashing_events(&self) -> &[crate::slashing::SlashingEvent] {
        self.slashing.confirmed_events()
    }

    /// Process pending slashing confirmations
    pub fn process_slashing(&mut self) {
        self.slashing.process_confirmations();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::ValidatorKeypair;
    use crate::validator::ValidatorNode;

    fn create_test_validator(id: &str, reputation: f32) -> ValidatorNode {
        let mut v = ValidatorNode::new(id.to_string());
        v.k_vector.k_r = reputation;
        v
    }

    fn create_validator_with_keypair(reputation: f32) -> (ValidatorNode, ValidatorKeypair) {
        let keypair = ValidatorKeypair::generate();
        let mut v = ValidatorNode::new(keypair.public_key_hex());
        v.k_vector.k_r = reputation;
        (v, keypair)
    }

    fn setup_consensus_with_validators(n: usize) -> RbBftConsensus {
        let mut consensus = RbBftConsensus::with_defaults();

        for i in 0..n {
            let v = create_test_validator(&format!("validator-{}", i), 0.8);
            consensus.add_validator(v);
        }

        consensus.set_our_id("validator-0".to_string());
        consensus
    }

    fn setup_consensus_with_keypairs(n: usize) -> (RbBftConsensus, Vec<ValidatorKeypair>) {
        let mut consensus = RbBftConsensus::with_defaults();
        let mut keypairs = Vec::with_capacity(n);

        for _ in 0..n {
            let (v, kp) = create_validator_with_keypair(0.8);
            consensus.add_validator(v);
            keypairs.push(kp);
        }

        (consensus, keypairs)
    }

    #[test]
    fn test_start_round() {
        let mut consensus = setup_consensus_with_validators(5);
        let round = consensus.start_round().unwrap();
        assert_eq!(round, 1);
    }

    #[test]
    fn test_insufficient_validators() {
        let mut consensus = setup_consensus_with_validators(3);
        let result = consensus.start_round();
        assert!(result.is_err());
    }

    #[test]
    fn test_leader_can_propose() {
        let (mut consensus, keypairs) = setup_consensus_with_keypairs(5);
        consensus.start_round().unwrap();

        // Find who is leader and get their keypair
        let round = consensus.rounds.active_round().unwrap();
        let leader_id = round.leader.clone();

        let leader_keypair = keypairs.iter()
            .find(|kp| kp.public_key_hex() == leader_id)
            .expect("Leader keypair should exist");

        // Use the signed propose method
        let result = consensus.propose_signed(
            vec![1, 2, 3],
            "content-hash-123".to_string(),
            "parent-hash-000".to_string(),
            leader_keypair,
        );
        assert!(result.is_ok());

        // Verify the proposal has a valid signature
        let proposal = result.unwrap();
        assert!(proposal.has_valid_signature());
        assert_eq!(proposal.signer_pubkey().unwrap(), leader_id);
    }

    #[test]
    fn test_non_leader_cannot_propose() {
        let mut consensus = setup_consensus_with_validators(5);
        consensus.start_round().unwrap();

        // Set ourselves as NOT the leader
        let round = consensus.rounds.active_round().unwrap();
        let non_leader = if round.leader == "validator-0" {
            "validator-1"
        } else {
            "validator-0"
        };
        consensus.set_our_id(non_leader.to_string());

        let result = consensus.propose(
            vec![1, 2, 3],
            "content-hash-123".to_string(),
        );
        assert!(matches!(result, Err(ConsensusError::NotLeader { .. })));
    }

    #[test]
    fn test_consensus_threshold() {
        let consensus = setup_consensus_with_validators(5);
        // 5 validators with rep 0.8 each: total weight = 5 * 0.64 = 3.2
        // Threshold = 3.2 * 0.55 = 1.76
        let threshold = consensus.validators.consensus_threshold();
        assert!((threshold - 1.76).abs() < 0.1);
    }

    #[test]
    fn test_batch_vote_processing() {
        let (mut consensus, keypairs) = setup_consensus_with_keypairs(5);
        consensus.start_round().unwrap();

        // Find who is leader and get their keypair to create proposal
        let round = consensus.rounds.active_round().unwrap();
        let leader_id = round.leader.clone();

        let leader_keypair = keypairs.iter()
            .find(|kp| kp.public_key_hex() == leader_id)
            .expect("Leader keypair should exist");

        // Create and submit proposal
        consensus.propose_signed(
            vec![1, 2, 3],
            "content-hash-123".to_string(),
            "parent-hash-000".to_string(),
            leader_keypair,
        ).unwrap();

        // Get non-leader validators and create signed votes
        let mut votes = Vec::new();
        for kp in keypairs.iter() {
            if kp.public_key_hex() != leader_id {
                let proposal = consensus.rounds.active_round().unwrap().proposal.as_ref().unwrap();
                let mut vote = Vote::approve(
                    proposal.id.clone(),
                    1,
                    kp.public_key_hex(),
                    0.8,
                );
                vote.sign(kp).unwrap();
                votes.push(vote);
            }
        }

        // Process votes in batch
        let batch_result = consensus.receive_votes_batch(votes).unwrap();

        // All 4 non-leader votes should be accepted
        assert_eq!(batch_result.accepted_count, 4);
        assert_eq!(batch_result.rejected_count, 0);
        assert!(batch_result.all_accepted());
    }

    #[test]
    fn test_batch_vote_result_methods() {
        let result = BatchVoteResult {
            accepted_count: 5,
            rejected_count: 2,
            rejected_details: vec![
                ("voter1".to_string(), "Invalid signature".to_string()),
                ("voter2".to_string(), "Wrong round".to_string()),
            ],
        };

        assert!(!result.all_accepted());
        assert_eq!(result.total(), 7);
    }
}
