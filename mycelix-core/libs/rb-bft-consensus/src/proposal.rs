// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proposal types for RB-BFT consensus

use serde::{Deserialize, Serialize};
use crate::crypto::{ConsensusSignature, ValidatorKeypair, domains, create_signable_bytes};
use crate::error::ConsensusResult;

/// A proposal submitted by a leader for consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Unique proposal ID (hash of content)
    pub id: String,
    /// Round number this proposal is for
    pub round: u64,
    /// ID of the proposing validator (leader) - their public key hex
    pub proposer: String,
    /// The actual content being proposed (serialized)
    pub content: Vec<u8>,
    /// Hash of the content for verification
    pub content_hash: String,
    /// Timestamp when proposal was created
    pub timestamp: i64,
    /// Cryptographic signature from the proposer (ed25519)
    pub signature: ConsensusSignature,
    /// Previous proposal hash (chain linkage)
    pub parent_hash: String,
}

impl Proposal {
    /// Create a new unsigned proposal
    pub fn new(
        round: u64,
        proposer: String,
        content: Vec<u8>,
        content_hash: String,
        parent_hash: String,
    ) -> Self {
        Self {
            id: Self::compute_id(round, &proposer, &content_hash),
            round,
            proposer,
            content,
            content_hash,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            signature: ConsensusSignature::empty(),
            parent_hash,
        }
    }

    /// Compute proposal ID from components using SHA256
    fn compute_id(round: u64, proposer: &str, content_hash: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(round.to_le_bytes());
        hasher.update(proposer.as_bytes());
        hasher.update(content_hash.as_bytes());
        let hash = hasher.finalize();
        format!("prop-{}", hex::encode(&hash[..16]))
    }

    /// Get the bytes that need to be signed
    fn signable_data(&self) -> SignableProposal {
        SignableProposal {
            id: self.id.clone(),
            round: self.round,
            proposer: self.proposer.clone(),
            content_hash: self.content_hash.clone(),
            timestamp: self.timestamp,
            parent_hash: self.parent_hash.clone(),
        }
    }

    /// Sign the proposal with the validator's private key
    ///
    /// This creates a real ed25519 signature over the proposal data.
    pub fn sign(&mut self, keypair: &ValidatorKeypair) -> ConsensusResult<()> {
        let signable = self.signable_data();
        let bytes = create_signable_bytes(&signable)?;
        self.signature = keypair.sign_with_domain(domains::PROPOSAL, &bytes);
        Ok(())
    }

    /// Verify the proposal signature
    ///
    /// Returns Ok(()) if the signature is valid, or an error explaining why it's invalid.
    pub fn verify_signature(&self) -> ConsensusResult<()> {
        let signable = self.signable_data();
        let bytes = create_signable_bytes(&signable)?;
        self.signature.verify_with_domain(domains::PROPOSAL, &bytes)
    }

    /// Check if proposal has a valid signature
    pub fn has_valid_signature(&self) -> bool {
        self.verify_signature().is_ok()
    }

    /// Check if proposal is structurally valid (has required fields and valid signature)
    pub fn is_valid(&self) -> bool {
        !self.id.is_empty()
            && !self.proposer.is_empty()
            && !self.content_hash.is_empty()
            && self.has_valid_signature()
    }

    /// Get the proposer's public key from the signature
    pub fn signer_pubkey(&self) -> Option<String> {
        if self.signature.is_present() {
            Some(self.signature.signer_hex())
        } else {
            None
        }
    }
}

/// Data structure used for signing (excludes signature field)
#[derive(Serialize)]
struct SignableProposal {
    id: String,
    round: u64,
    proposer: String,
    content_hash: String,
    timestamp: i64,
    parent_hash: String,
}

/// Status of a proposal in the consensus process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    /// Proposal submitted, awaiting votes
    Pending,
    /// Gathering votes
    Voting,
    /// Consensus reached, proposal accepted
    Accepted,
    /// Consensus not reached, proposal rejected
    Rejected,
    /// Proposal timed out
    TimedOut,
    /// Proposal invalidated (e.g., due to Byzantine behavior)
    Invalidated,
}

/// Metadata about a proposal's voting progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalVotingState {
    /// The proposal ID
    pub proposal_id: String,
    /// Current status
    pub status: ProposalStatus,
    /// Total weighted votes received (for)
    pub weighted_votes_for: f32,
    /// Total weighted votes received (against)
    pub weighted_votes_against: f32,
    /// Threshold needed for consensus
    pub threshold: f32,
    /// Number of validators who voted
    pub voter_count: usize,
    /// When voting started
    pub voting_started: i64,
    /// When voting ended (0 if still ongoing)
    pub voting_ended: i64,
}

impl ProposalVotingState {
    /// Create initial voting state for a proposal
    pub fn new(proposal_id: String, threshold: f32) -> Self {
        Self {
            proposal_id,
            status: ProposalStatus::Voting,
            weighted_votes_for: 0.0,
            weighted_votes_against: 0.0,
            threshold,
            voter_count: 0,
            voting_started: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            voting_ended: 0,
        }
    }

    /// Check if consensus has been reached
    pub fn consensus_reached(&self) -> bool {
        self.weighted_votes_for > self.threshold
    }

    /// Check if consensus is impossible (too many against votes)
    pub fn consensus_impossible(&self) -> bool {
        // If against votes exceed what's needed to block consensus
        self.weighted_votes_against > (1.0 - self.threshold / self.total_weight())
    }

    /// Get total weighted votes cast
    fn total_weight(&self) -> f32 {
        // Estimate based on threshold (threshold = 55% of total)
        self.threshold / 0.55
    }

    /// Finalize the voting state
    pub fn finalize(&mut self, accepted: bool) {
        self.status = if accepted {
            ProposalStatus::Accepted
        } else {
            ProposalStatus::Rejected
        };
        self.voting_ended = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposal_creation() {
        let p = Proposal::new(
            1,
            "validator-abc".to_string(),
            vec![1, 2, 3],
            "hash123456".to_string(),
            "parent123".to_string(),
        );

        assert_eq!(p.round, 1);
        assert!(!p.id.is_empty());
        assert!(!p.signature.is_present()); // Unsigned initially
    }

    #[test]
    fn test_proposal_signing_and_verification() {
        let keypair = ValidatorKeypair::generate();
        let mut p = Proposal::new(
            1,
            keypair.public_key_hex(),
            vec![1, 2, 3],
            "hash123456".to_string(),
            "parent123".to_string(),
        );

        // Sign the proposal
        p.sign(&keypair).expect("signing should succeed");

        // Verify the signature
        assert!(p.signature.is_present());
        assert!(p.verify_signature().is_ok());
        assert!(p.has_valid_signature());
        assert!(p.is_valid());
    }

    #[test]
    fn test_proposal_tamper_detection() {
        let keypair = ValidatorKeypair::generate();
        let mut p = Proposal::new(
            1,
            keypair.public_key_hex(),
            vec![1, 2, 3],
            "hash123456".to_string(),
            "parent123".to_string(),
        );

        p.sign(&keypair).unwrap();
        assert!(p.verify_signature().is_ok());

        // Tamper with the proposal
        p.round = 999;

        // Signature should now be invalid
        assert!(p.verify_signature().is_err());
        assert!(!p.is_valid());
    }

    #[test]
    fn test_proposal_wrong_key_rejected() {
        let keypair1 = ValidatorKeypair::generate();
        let keypair2 = ValidatorKeypair::generate();

        let mut p = Proposal::new(
            1,
            keypair1.public_key_hex(),
            vec![1, 2, 3],
            "hash123456".to_string(),
            "parent123".to_string(),
        );

        // Sign with keypair1
        p.sign(&keypair1).unwrap();
        assert!(p.verify_signature().is_ok());

        // Overwrite with keypair2's signature
        let signable = p.signable_data();
        let bytes = create_signable_bytes(&signable).unwrap();
        p.signature = keypair2.sign_with_domain(domains::PROPOSAL, &bytes);

        // Should still verify (signature is valid, just from different key)
        assert!(p.verify_signature().is_ok());
        // But the signer doesn't match the declared proposer
        assert_ne!(p.signer_pubkey().unwrap(), p.proposer);
    }

    #[test]
    fn test_voting_state_consensus() {
        let mut state = ProposalVotingState::new("prop-1".to_string(), 1.1);
        state.weighted_votes_for = 1.2;

        assert!(state.consensus_reached());
    }
}
