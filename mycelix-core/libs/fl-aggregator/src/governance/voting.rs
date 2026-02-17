//! Vote Casting and Tallying
//!
//! Handles vote submission, validation, tallying, and result computation.

use super::types::*;
use super::{GovernanceError, GovernanceResult};
use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet};

/// Vote manager handling vote submission and tallying
#[derive(Debug, Default)]
pub struct VoteManager {
    /// Votes indexed by proposal ID then voter DID
    votes: HashMap<String, HashMap<String, Vote>>,

    /// Vote receipts for verification
    receipts: HashMap<String, VoteReceipt>,

    /// Cached tallies
    tallies: HashMap<String, VoteTally>,
}

impl VoteManager {
    /// Create a new vote manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Cast a vote
    pub fn cast_vote(
        &mut self,
        proposal_id: impl Into<String>,
        voter_did: impl Into<String>,
        choice: VoteChoice,
        weight: f32,
        signature: Vec<u8>,
    ) -> GovernanceResult<VoteReceipt> {
        let proposal_id = proposal_id.into();
        let voter_did = voter_did.into();

        // Check for duplicate vote first
        if let Some(proposal_votes) = self.votes.get(&proposal_id) {
            if proposal_votes.contains_key(&voter_did) {
                return Err(GovernanceError::AlreadyVoted(voter_did));
            }
        }

        // Create vote
        let mut vote = Vote::new(&proposal_id, &voter_did, choice, weight);
        vote.signature = signature;

        // Generate receipt and commitment
        let commitment = Self::generate_commitment_static(&vote);
        let receipt = VoteReceipt {
            receipt_id: format!("{}:{}", proposal_id, voter_did),
            proposal_id: proposal_id.clone(),
            voter_did: voter_did.clone(),
            choice,
            weight,
            cast_at: vote.cast_at,
            commitment,
        };

        // Store vote and receipt
        self.votes
            .entry(proposal_id.clone())
            .or_default()
            .insert(voter_did, vote);
        self.receipts.insert(receipt.receipt_id.clone(), receipt.clone());

        // Invalidate cached tally
        self.tallies.remove(&proposal_id);

        Ok(receipt)
    }

    /// Cast a delegated vote
    pub fn cast_delegated_vote(
        &mut self,
        proposal_id: impl Into<String>,
        delegate_did: impl Into<String>,
        delegator_did: impl Into<String>,
        choice: VoteChoice,
        weight: f32,
    ) -> GovernanceResult<VoteReceipt> {
        let proposal_id = proposal_id.into();
        let delegate_did = delegate_did.into();
        let delegator_did = delegator_did.into();

        // Check if delegator already voted directly
        if let Some(proposal_votes) = self.votes.get(&proposal_id) {
            if proposal_votes.contains_key(&delegator_did) {
                return Err(GovernanceError::AlreadyVoted(delegator_did));
            }
        }

        // Create delegated vote
        let vote = Vote::delegated(
            &proposal_id,
            &delegate_did,
            &delegator_did,
            choice,
            weight,
        );

        // Generate receipt and commitment
        let commitment = Self::generate_commitment_static(&vote);
        let receipt = VoteReceipt {
            receipt_id: format!("{}:{}:delegated:{}", proposal_id, delegator_did, delegate_did),
            proposal_id: proposal_id.clone(),
            voter_did: delegator_did.clone(),
            choice,
            weight,
            cast_at: vote.cast_at,
            commitment,
        };

        // Store vote under delegator's DID
        self.votes
            .entry(proposal_id.clone())
            .or_default()
            .insert(delegator_did, vote);
        self.receipts.insert(receipt.receipt_id.clone(), receipt.clone());

        // Invalidate cached tally
        self.tallies.remove(&proposal_id);

        Ok(receipt)
    }

    /// Change an existing vote (if allowed by rules)
    pub fn change_vote(
        &mut self,
        proposal_id: &str,
        voter_did: &str,
        new_choice: VoteChoice,
        signature: Vec<u8>,
    ) -> GovernanceResult<VoteReceipt> {
        let proposal_votes = self
            .votes
            .get_mut(proposal_id)
            .ok_or_else(|| GovernanceError::ProposalNotFound(proposal_id.to_string()))?;

        let existing_vote = proposal_votes
            .get_mut(voter_did)
            .ok_or_else(|| GovernanceError::InvalidProposal("No existing vote to change".to_string()))?;

        // Don't allow changing delegated votes
        if existing_vote.delegated {
            return Err(GovernanceError::DelegationError(
                "Cannot change delegated vote".to_string(),
            ));
        }

        // Update vote
        existing_vote.choice = new_choice;
        existing_vote.cast_at = Utc::now();
        existing_vote.signature = signature;

        // Extract data needed for receipt before releasing borrow
        let weight = existing_vote.weight;
        let cast_at = existing_vote.cast_at;
        let commitment = Self::generate_commitment_static(existing_vote);

        let receipt = VoteReceipt {
            receipt_id: format!("{}:{}:changed", proposal_id, voter_did),
            proposal_id: proposal_id.to_string(),
            voter_did: voter_did.to_string(),
            choice: new_choice,
            weight,
            cast_at,
            commitment,
        };

        // Invalidate cached tally
        self.tallies.remove(proposal_id);

        Ok(receipt)
    }

    /// Get vote for a specific voter on a proposal
    pub fn get_vote(&self, proposal_id: &str, voter_did: &str) -> Option<&Vote> {
        self.votes.get(proposal_id)?.get(voter_did)
    }

    /// Get all votes for a proposal
    pub fn get_proposal_votes(&self, proposal_id: &str) -> Vec<&Vote> {
        self.votes
            .get(proposal_id)
            .map(|votes| votes.values().collect())
            .unwrap_or_default()
    }

    /// Tally votes for a proposal
    pub fn tally_votes(
        &mut self,
        proposal_id: &str,
        proposal_type: ProposalType,
        eligible_weight: f32,
    ) -> GovernanceResult<VoteTally> {
        // Return cached tally if available
        if let Some(tally) = self.tallies.get(proposal_id) {
            return Ok(tally.clone());
        }

        let votes = self
            .votes
            .get(proposal_id)
            .ok_or_else(|| GovernanceError::ProposalNotFound(proposal_id.to_string()))?;

        let mut tally = VoteTally {
            eligible_weight,
            ..Default::default()
        };

        for vote in votes.values() {
            tally.add_vote(vote.choice, vote.weight);
        }

        tally.check_quorum(proposal_type.quorum_threshold());
        tally.check_approval(proposal_type.approval_threshold());

        // Cache the tally
        self.tallies.insert(proposal_id.to_string(), tally.clone());

        Ok(tally)
    }

    /// Generate commitment hash for vote verification
    fn generate_commitment_static(vote: &Vote) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        vote.proposal_id.hash(&mut hasher);
        vote.voter_did.hash(&mut hasher);
        (vote.choice as u8).hash(&mut hasher);
        vote.cast_at.timestamp().hash(&mut hasher);

        format!("{:016x}", hasher.finish())
    }

    /// Verify a vote receipt
    pub fn verify_receipt(&self, receipt: &VoteReceipt) -> bool {
        if let Some(stored) = self.receipts.get(&receipt.receipt_id) {
            stored.commitment == receipt.commitment
        } else {
            false
        }
    }

    /// Get participation statistics
    pub fn get_participation_stats(&self, proposal_id: &str) -> Option<ParticipationStats> {
        let votes = self.votes.get(proposal_id)?;

        let mut stats = ParticipationStats {
            total_votes: votes.len() as u32,
            direct_votes: 0,
            delegated_votes: 0,
            unique_voters: HashSet::new(),
            choice_breakdown: HashMap::new(),
        };

        for vote in votes.values() {
            if vote.delegated {
                stats.delegated_votes += 1;
                if let Some(delegator) = &vote.delegator_did {
                    stats.unique_voters.insert(delegator.clone());
                }
            } else {
                stats.direct_votes += 1;
                stats.unique_voters.insert(vote.voter_did.clone());
            }

            *stats.choice_breakdown.entry(vote.choice).or_insert(0) += 1;
        }

        Some(stats)
    }

    /// List voters for a proposal
    pub fn list_voters(&self, proposal_id: &str) -> Vec<VoterInfo> {
        self.votes
            .get(proposal_id)
            .map(|votes| {
                votes
                    .values()
                    .map(|v| VoterInfo {
                        did: v.voter_did.clone(),
                        choice: v.choice,
                        weight: v.weight,
                        delegated: v.delegated,
                        delegator: v.delegator_did.clone(),
                        cast_at: v.cast_at,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Clear votes for a proposal (for testing/cleanup)
    pub fn clear_proposal_votes(&mut self, proposal_id: &str) {
        self.votes.remove(proposal_id);
        self.tallies.remove(proposal_id);
        // Remove associated receipts
        self.receipts.retain(|_, r| r.proposal_id != proposal_id);
    }
}

/// Vote receipt for verification
#[derive(Debug, Clone)]
pub struct VoteReceipt {
    pub receipt_id: String,
    pub proposal_id: String,
    pub voter_did: String,
    pub choice: VoteChoice,
    pub weight: f32,
    pub cast_at: DateTime<Utc>,
    pub commitment: String,
}

/// Participation statistics
#[derive(Debug, Clone)]
pub struct ParticipationStats {
    pub total_votes: u32,
    pub direct_votes: u32,
    pub delegated_votes: u32,
    pub unique_voters: HashSet<String>,
    pub choice_breakdown: HashMap<VoteChoice, u32>,
}

/// Voter info summary
#[derive(Debug, Clone)]
pub struct VoterInfo {
    pub did: String,
    pub choice: VoteChoice,
    pub weight: f32,
    pub delegated: bool,
    pub delegator: Option<String>,
    pub cast_at: DateTime<Utc>,
}

/// Vote verification result
#[derive(Debug, Clone)]
pub struct VoteVerification {
    pub valid: bool,
    pub reason: Option<String>,
    pub commitment_match: bool,
    pub signature_valid: bool,
}

/// Verify vote signature (placeholder - actual implementation depends on crypto)
pub fn verify_vote_signature(_vote: &Vote, _public_key: &[u8]) -> bool {
    // In a real implementation, this would verify the Ed25519 signature
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_vote() {
        let mut manager = VoteManager::new();

        let receipt = manager
            .cast_vote(
                "MIP-0001",
                "did:mycelix:alice",
                VoteChoice::For,
                10.0,
                vec![],
            )
            .unwrap();

        assert_eq!(receipt.proposal_id, "MIP-0001");
        assert_eq!(receipt.voter_did, "did:mycelix:alice");
        assert_eq!(receipt.choice, VoteChoice::For);
    }

    #[test]
    fn test_duplicate_vote_rejected() {
        let mut manager = VoteManager::new();

        manager
            .cast_vote("MIP-0001", "did:mycelix:alice", VoteChoice::For, 10.0, vec![])
            .unwrap();

        let result = manager.cast_vote(
            "MIP-0001",
            "did:mycelix:alice",
            VoteChoice::Against,
            10.0,
            vec![],
        );

        assert!(matches!(result, Err(GovernanceError::AlreadyVoted(_))));
    }

    #[test]
    fn test_vote_tally() {
        let mut manager = VoteManager::new();

        manager
            .cast_vote("MIP-0001", "did:mycelix:alice", VoteChoice::For, 30.0, vec![])
            .unwrap();
        manager
            .cast_vote("MIP-0001", "did:mycelix:bob", VoteChoice::For, 20.0, vec![])
            .unwrap();
        manager
            .cast_vote("MIP-0001", "did:mycelix:charlie", VoteChoice::Against, 10.0, vec![])
            .unwrap();

        let tally = manager
            .tally_votes("MIP-0001", ProposalType::Standard, 100.0)
            .unwrap();

        assert_eq!(tally.for_weight, 50.0);
        assert_eq!(tally.against_weight, 10.0);
        assert_eq!(tally.voter_count, 3);
        assert!((tally.approval_rate() - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_delegated_vote() {
        let mut manager = VoteManager::new();

        let receipt = manager
            .cast_delegated_vote(
                "MIP-0001",
                "did:mycelix:delegate",
                "did:mycelix:delegator",
                VoteChoice::For,
                15.0,
            )
            .unwrap();

        assert!(receipt.receipt_id.contains("delegated"));

        let vote = manager
            .get_vote("MIP-0001", "did:mycelix:delegator")
            .unwrap();
        assert!(vote.delegated);
        assert_eq!(vote.delegator_did, Some("did:mycelix:delegator".to_string()));
    }

    #[test]
    fn test_change_vote() {
        let mut manager = VoteManager::new();

        manager
            .cast_vote("MIP-0001", "did:mycelix:alice", VoteChoice::For, 10.0, vec![])
            .unwrap();

        let receipt = manager
            .change_vote("MIP-0001", "did:mycelix:alice", VoteChoice::Against, vec![])
            .unwrap();

        assert_eq!(receipt.choice, VoteChoice::Against);

        let vote = manager
            .get_vote("MIP-0001", "did:mycelix:alice")
            .unwrap();
        assert_eq!(vote.choice, VoteChoice::Against);
    }

    #[test]
    fn test_participation_stats() {
        let mut manager = VoteManager::new();

        manager
            .cast_vote("MIP-0001", "did:mycelix:alice", VoteChoice::For, 10.0, vec![])
            .unwrap();
        manager
            .cast_vote("MIP-0001", "did:mycelix:bob", VoteChoice::Against, 10.0, vec![])
            .unwrap();
        manager
            .cast_delegated_vote(
                "MIP-0001",
                "did:mycelix:alice",
                "did:mycelix:charlie",
                VoteChoice::For,
                5.0,
            )
            .unwrap();

        let stats = manager.get_participation_stats("MIP-0001").unwrap();

        assert_eq!(stats.total_votes, 3);
        assert_eq!(stats.direct_votes, 2);
        assert_eq!(stats.delegated_votes, 1);
    }
}
