// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vote types for RB-BFT consensus

use serde::{Deserialize, Serialize};
use crate::crypto::{ConsensusSignature, ValidatorKeypair, domains, create_signable_bytes};
use crate::error::ConsensusResult;

/// A vote cast by a validator on a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Unique vote ID
    pub id: String,
    /// ID of the proposal being voted on
    pub proposal_id: String,
    /// Round number
    pub round: u64,
    /// ID of the voting validator (public key hex)
    pub voter: String,
    /// The vote decision
    pub decision: VoteDecision,
    /// Voter's reputation at time of vote
    pub reputation: f32,
    /// Computed voting weight (reputation²)
    pub weight: f32,
    /// Timestamp of the vote
    pub timestamp: i64,
    /// Cryptographic signature (ed25519)
    pub signature: ConsensusSignature,
    /// Optional reason for the vote (especially for rejections)
    pub reason: Option<String>,
}

impl Vote {
    /// Create a new unsigned vote
    pub fn new(
        proposal_id: String,
        round: u64,
        voter: String,
        decision: VoteDecision,
        reputation: f32,
    ) -> Self {
        Self {
            id: Self::compute_id(&proposal_id, &voter, round),
            proposal_id,
            round,
            voter,
            decision,
            reputation,
            weight: reputation.powi(2), // reputation² weighting
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            signature: ConsensusSignature::empty(),
            reason: None,
        }
    }

    /// Compute vote ID using SHA256
    fn compute_id(proposal_id: &str, voter: &str, round: u64) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(proposal_id.as_bytes());
        hasher.update(voter.as_bytes());
        hasher.update(round.to_le_bytes());
        let hash = hasher.finalize();
        format!("vote-{}", hex::encode(&hash[..12]))
    }

    /// Get the bytes that need to be signed
    fn signable_data(&self) -> SignableVote {
        SignableVote {
            id: self.id.clone(),
            proposal_id: self.proposal_id.clone(),
            round: self.round,
            voter: self.voter.clone(),
            decision: self.decision,
            reputation: self.reputation,
            weight: self.weight,
            timestamp: self.timestamp,
            reason: self.reason.clone(),
        }
    }

    /// Create an approval vote
    pub fn approve(proposal_id: String, round: u64, voter: String, reputation: f32) -> Self {
        Self::new(proposal_id, round, voter, VoteDecision::Approve, reputation)
    }

    /// Create a rejection vote
    pub fn reject(
        proposal_id: String,
        round: u64,
        voter: String,
        reputation: f32,
        reason: String,
    ) -> Self {
        let mut vote = Self::new(proposal_id, round, voter, VoteDecision::Reject, reputation);
        vote.reason = Some(reason);
        vote
    }

    /// Create an abstain vote
    pub fn abstain(proposal_id: String, round: u64, voter: String, reputation: f32) -> Self {
        Self::new(proposal_id, round, voter, VoteDecision::Abstain, reputation)
    }

    /// Sign the vote with the validator's private key
    ///
    /// This creates a real ed25519 signature over the vote data.
    pub fn sign(&mut self, keypair: &ValidatorKeypair) -> ConsensusResult<()> {
        let signable = self.signable_data();
        let bytes = create_signable_bytes(&signable)?;
        self.signature = keypair.sign_with_domain(domains::VOTE, &bytes);
        Ok(())
    }

    /// Verify the vote signature
    ///
    /// Returns Ok(()) if the signature is valid, or an error explaining why it's invalid.
    pub fn verify_signature(&self) -> ConsensusResult<()> {
        let signable = self.signable_data();
        let bytes = create_signable_bytes(&signable)?;
        self.signature.verify_with_domain(domains::VOTE, &bytes)
    }

    /// Check if vote has a valid signature
    pub fn has_valid_signature(&self) -> bool {
        self.verify_signature().is_ok()
    }

    /// Check if vote is valid
    pub fn is_valid(&self) -> bool {
        !self.voter.is_empty()
            && self.reputation >= 0.0
            && self.reputation <= 1.0
            && self.weight >= 0.0
            && self.has_valid_signature()
    }

    /// Get the weighted value of this vote (reputation² weight)
    ///
    /// This is the absolute voting weight regardless of decision.
    pub fn weighted_value(&self) -> f32 {
        self.weight
    }

    /// Get the voter ID (alias for voter field)
    pub fn voter_id(&self) -> &str {
        &self.voter
    }

    /// Get the effective weight for consensus calculation
    pub fn effective_weight(&self) -> f32 {
        match self.decision {
            VoteDecision::Approve => self.weight,
            VoteDecision::Reject => -self.weight,
            VoteDecision::Abstain => 0.0,
        }
    }

    /// Get the voter's public key from the signature
    pub fn signer_pubkey(&self) -> Option<String> {
        if self.signature.is_present() {
            Some(self.signature.signer_hex())
        } else {
            None
        }
    }

    /// Get the bytes that need to be signed/verified
    ///
    /// This returns the SHA256 hash of (domain + signable_data) which is what
    /// the signature is actually computed over.
    pub fn signable_bytes(&self) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let signable = self.signable_data();
        if let Ok(data) = create_signable_bytes(&signable) {
            let mut hasher = Sha256::new();
            hasher.update(domains::VOTE.as_bytes());
            hasher.update(&data);
            hasher.finalize().to_vec()
        } else {
            Vec::new()
        }
    }
}

/// Data structure used for signing (excludes signature field)
#[derive(Serialize)]
struct SignableVote {
    id: String,
    proposal_id: String,
    round: u64,
    voter: String,
    decision: VoteDecision,
    reputation: f32,
    weight: f32,
    timestamp: i64,
    reason: Option<String>,
}

/// The decision encoded in a vote
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteDecision {
    /// Approve the proposal
    Approve,
    /// Reject the proposal
    Reject,
    /// Abstain from voting (doesn't count toward threshold)
    Abstain,
}

impl VoteDecision {
    /// Check if this is an approval
    pub fn is_approval(&self) -> bool {
        matches!(self, VoteDecision::Approve)
    }

    /// Check if this is a rejection
    pub fn is_rejection(&self) -> bool {
        matches!(self, VoteDecision::Reject)
    }
}

/// Collection of votes for a proposal
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VoteCollection {
    /// All votes received
    votes: Vec<Vote>,
    /// Set of voters who have already voted (for duplicate detection)
    voters: std::collections::HashSet<String>,
    /// Total weighted approval votes
    weighted_approvals: f32,
    /// Total weighted rejection votes
    weighted_rejections: f32,
    /// Total weighted abstentions (H-02 remediation: track abstentions)
    weighted_abstentions: f32,
    /// Current round number (for abstention tracking)
    round: u64,
}

/// Abstention tracking across rounds (H-02 remediation)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AbstentionTracker {
    /// Maps validator ID to list of rounds they abstained
    history: std::collections::HashMap<String, Vec<u64>>,
    /// Window size for calculating abstention rate
    window_size: u64,
    /// Threshold for suspicious abstention rate (default 0.3 = 30%)
    suspicious_threshold: f32,
}

impl AbstentionTracker {
    /// Create a new abstention tracker
    pub fn new(window_size: u64, suspicious_threshold: f32) -> Self {
        Self {
            history: std::collections::HashMap::new(),
            window_size,
            suspicious_threshold,
        }
    }

    /// Record an abstention
    pub fn record_abstention(&mut self, validator: &str, round: u64) {
        self.history
            .entry(validator.to_string())
            .or_default()
            .push(round);
    }

    /// Calculate abstention rate for a validator within the window
    pub fn abstention_rate(&self, validator: &str, current_round: u64) -> f32 {
        let window_start = current_round.saturating_sub(self.window_size);

        if let Some(rounds) = self.history.get(validator) {
            let abstentions_in_window = rounds
                .iter()
                .filter(|&&r| r >= window_start && r <= current_round)
                .count();

            let window = (current_round - window_start + 1) as f32;
            abstentions_in_window as f32 / window
        } else {
            0.0
        }
    }

    /// Check if validator has suspicious abstention pattern
    pub fn is_suspicious(&self, validator: &str, current_round: u64) -> bool {
        self.abstention_rate(validator, current_round) > self.suspicious_threshold
    }

    /// Get validators with suspicious abstention patterns
    pub fn get_suspicious_validators(&self, current_round: u64) -> Vec<(String, f32)> {
        self.history
            .keys()
            .filter_map(|v| {
                let rate = self.abstention_rate(v, current_round);
                if rate > self.suspicious_threshold {
                    Some((v.clone(), rate))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Prune old history to prevent unbounded growth
    pub fn prune_old_rounds(&mut self, min_round: u64) {
        for rounds in self.history.values_mut() {
            rounds.retain(|&r| r >= min_round);
        }
        // Remove validators with no remaining history
        self.history.retain(|_, rounds| !rounds.is_empty());
    }
}

impl VoteCollection {
    /// Create a new empty vote collection
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new vote collection for a specific round
    pub fn for_round(round: u64) -> Self {
        Self {
            round,
            ..Self::default()
        }
    }

    /// Add a vote to the collection
    ///
    /// Returns true if vote was added, false if duplicate
    pub fn add_vote(&mut self, vote: Vote) -> bool {
        if self.voters.contains(&vote.voter) {
            return false;
        }

        self.voters.insert(vote.voter.clone());

        match vote.decision {
            VoteDecision::Approve => {
                self.weighted_approvals += vote.weight;
            }
            VoteDecision::Reject => {
                self.weighted_rejections += vote.weight;
            }
            VoteDecision::Abstain => {
                // H-02 remediation: track weighted abstentions
                self.weighted_abstentions += vote.weight;
            }
        }

        self.votes.push(vote);
        true
    }

    /// Get total weighted abstentions (H-02 remediation)
    pub fn weighted_abstentions(&self) -> f32 {
        self.weighted_abstentions
    }

    /// Get the current round
    pub fn round(&self) -> u64 {
        self.round
    }

    /// Calculate effective consensus weight accounting for abstentions
    ///
    /// H-02 remediation: Abstentions count as partial weight against
    /// to prevent manipulation via coordinated abstention
    ///
    /// # H-06 Integration Note
    /// This function is currently NOT integrated into consensus checks.
    /// To fully mitigate abstention manipulation, call this function in
    /// `consensus.rs::check_consensus()` instead of using raw threshold:
    /// ```ignore
    /// let effective = round.votes.effective_threshold(threshold, 0.1);
    /// if round.votes.consensus_reached(effective) { ... }
    /// ```
    ///
    /// # Arguments
    /// * `base_threshold` - The base consensus threshold (e.g., 0.67)
    /// * `abstention_penalty` - How much each unit of abstention weight increases threshold
    ///
    /// # Returns
    /// Adjusted threshold: `base_threshold + (weighted_abstentions * penalty)`
    #[allow(dead_code)] // H-06: Kept for future integration - see doc comment
    pub fn effective_threshold(&self, base_threshold: f32, abstention_penalty: f32) -> f32 {
        // Abstentions increase the effective threshold
        // This makes it harder to pass proposals via coordinated abstention
        base_threshold + (self.weighted_abstentions * abstention_penalty)
    }

    /// Check if a validator has already voted
    pub fn has_voted(&self, voter: &str) -> bool {
        self.voters.contains(voter)
    }

    /// Get total weighted approvals
    pub fn weighted_approvals(&self) -> f32 {
        self.weighted_approvals
    }

    /// Get total weighted rejections
    pub fn weighted_rejections(&self) -> f32 {
        self.weighted_rejections
    }

    /// Get number of votes
    pub fn vote_count(&self) -> usize {
        self.votes.len()
    }

    /// Get all votes
    pub fn votes(&self) -> &[Vote] {
        &self.votes
    }

    /// Check if consensus threshold is met
    pub fn consensus_reached(&self, threshold: f32) -> bool {
        self.weighted_approvals > threshold
    }

    /// Check if consensus is impossible
    pub fn consensus_impossible(&self, total_possible_weight: f32, threshold: f32) -> bool {
        // If remaining possible votes can't reach threshold
        let remaining = total_possible_weight - self.weighted_approvals - self.weighted_rejections;
        self.weighted_approvals + remaining < threshold
    }

    /// Get votes by decision type
    pub fn votes_by_decision(&self, decision: VoteDecision) -> Vec<&Vote> {
        self.votes.iter().filter(|v| v.decision == decision).collect()
    }
}

/// Evidence of double voting (Byzantine behavior)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleVoteEvidence {
    /// First vote cast
    pub vote1: Vote,
    /// Second (conflicting) vote
    pub vote2: Vote,
    /// When the double vote was detected
    pub detected_at: i64,
}

impl DoubleVoteEvidence {
    /// Create evidence from two conflicting votes
    pub fn new(vote1: Vote, vote2: Vote) -> Option<Self> {
        // Verify these are actually double votes
        if vote1.voter != vote2.voter || vote1.round != vote2.round {
            return None;
        }

        // Must be for the same proposal with different decisions
        if vote1.proposal_id != vote2.proposal_id || vote1.decision == vote2.decision {
            return None;
        }

        Some(Self {
            vote1,
            vote2,
            detected_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
        })
    }

    /// Get the Byzantine validator's ID
    pub fn byzantine_validator(&self) -> &str {
        &self.vote1.voter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vote_weight_is_reputation_squared() {
        let vote = Vote::new(
            "prop-1".to_string(),
            1,
            "voter-1".to_string(),
            VoteDecision::Approve,
            0.8,
        );
        assert!((vote.weight - 0.64).abs() < 0.001);
    }

    #[test]
    fn test_vote_signing_and_verification() {
        let keypair = ValidatorKeypair::generate();
        let mut vote = Vote::approve(
            "prop-1".to_string(),
            1,
            keypair.public_key_hex(),
            0.8,
        );

        // Sign the vote
        vote.sign(&keypair).expect("signing should succeed");

        // Verify
        assert!(vote.signature.is_present());
        assert!(vote.verify_signature().is_ok());
        assert!(vote.has_valid_signature());
        assert!(vote.is_valid());
    }

    #[test]
    fn test_vote_tamper_detection() {
        let keypair = ValidatorKeypair::generate();
        let mut vote = Vote::approve(
            "prop-1".to_string(),
            1,
            keypair.public_key_hex(),
            0.8,
        );

        vote.sign(&keypair).unwrap();
        assert!(vote.verify_signature().is_ok());

        // Tamper with the vote
        vote.decision = VoteDecision::Reject;

        // Signature should now be invalid
        assert!(vote.verify_signature().is_err());
        assert!(!vote.is_valid());
    }

    #[test]
    fn test_vote_collection_no_duplicates() {
        let mut collection = VoteCollection::new();

        let vote1 = Vote::approve("prop-1".to_string(), 1, "voter-1".to_string(), 0.8);
        let vote2 = Vote::approve("prop-1".to_string(), 1, "voter-1".to_string(), 0.8);

        assert!(collection.add_vote(vote1));
        assert!(!collection.add_vote(vote2)); // Duplicate
    }

    #[test]
    fn test_vote_collection_aggregates_weights() {
        let mut collection = VoteCollection::new();

        collection.add_vote(Vote::approve("prop-1".to_string(), 1, "voter-1".to_string(), 0.8));
        collection.add_vote(Vote::approve("prop-1".to_string(), 1, "voter-2".to_string(), 0.6));

        // 0.8² + 0.6² = 0.64 + 0.36 = 1.0
        assert!((collection.weighted_approvals() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_double_vote_evidence_with_signatures() {
        let keypair = ValidatorKeypair::generate();
        let voter_id = keypair.public_key_hex();

        let mut vote1 = Vote::approve("prop-1".to_string(), 1, voter_id.clone(), 0.8);
        let mut vote2 = Vote::reject("prop-1".to_string(), 1, voter_id.clone(), 0.8, "changed mind".to_string());

        // Sign both votes (this is the Byzantine behavior - signing conflicting votes)
        vote1.sign(&keypair).unwrap();
        vote2.sign(&keypair).unwrap();

        // Both signatures are valid individually
        assert!(vote1.verify_signature().is_ok());
        assert!(vote2.verify_signature().is_ok());

        // But together they prove Byzantine behavior
        let evidence = DoubleVoteEvidence::new(vote1, vote2);
        assert!(evidence.is_some());
        assert_eq!(evidence.unwrap().byzantine_validator(), voter_id);
    }

    #[test]
    fn test_double_vote_evidence() {
        let vote1 = Vote::approve("prop-1".to_string(), 1, "voter-1".to_string(), 0.8);
        let vote2 = Vote::reject("prop-1".to_string(), 1, "voter-1".to_string(), 0.8, "changed mind".to_string());

        let evidence = DoubleVoteEvidence::new(vote1, vote2);
        assert!(evidence.is_some());
        assert_eq!(evidence.unwrap().byzantine_validator(), "voter-1");
    }
}
