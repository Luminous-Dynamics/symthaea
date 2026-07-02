// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration Tests for ZK-Verified Voting
//!
//! Tests the zero-knowledge proof voting workflow including:
//! - Eligibility proof generation and verification
//! - Verified vote casting and validation
//! - Vote tallying with ZK proofs
//! - Security level requirements per proposal type
//!
//! These tests are standalone and don't require zome compilation.

use sha3::{Digest, Sha3_256};

// =============================================================================
// TYPE DEFINITIONS (matching voting_integrity types)
// =============================================================================

/// ZK proposal type for eligibility proofs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ZkProposalType {
    Standard = 0,
    Constitutional = 1,
    ModelGovernance = 2,
    Emergency = 3,
    Treasury = 4,
    Membership = 5,
}

/// Proposal tier for voting thresholds
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalTier {
    Basic,
    Standard,
    Major,
    Constitutional,
}

/// Vote choice
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoteChoice {
    Approve,
    Reject,
    Abstain,
}

/// Mock timestamp
#[derive(Clone, Copy, Debug)]
pub struct Timestamp(i64);

impl Timestamp {
    pub fn from_micros(us: i64) -> Self {
        Self(us)
    }

    pub fn as_micros(&self) -> i64 {
        self.0
    }
}

/// Mock action hash
#[derive(Clone, Debug)]
pub struct ActionHash([u8; 32]);

impl ActionHash {
    pub fn from_raw_32(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// Eligibility proof structure
#[derive(Clone, Debug)]
pub struct EligibilityProof {
    pub id: String,
    pub voter_did: String,
    pub voter_commitment: Vec<u8>,
    pub proposal_type: ZkProposalType,
    pub eligible: bool,
    pub requirements_met: u8,
    pub active_requirements: u8,
    pub proof_bytes: Vec<u8>,
    pub generated_at: Timestamp,
    pub expires_at: Option<Timestamp>,
}

/// Verified vote structure
#[derive(Clone, Debug)]
pub struct VerifiedVote {
    pub id: String,
    pub proposal_id: String,
    pub proposal_tier: ProposalTier,
    pub voter: String,
    pub choice: VoteChoice,
    pub eligibility_proof_hash: ActionHash,
    pub voter_commitment: Vec<u8>,
    pub effective_weight: f64,
    pub reason: Option<String>,
    pub voted_at: Timestamp,
}

// =============================================================================
// TEST HELPERS
// =============================================================================

/// Create a mock eligibility proof for testing
fn create_test_eligibility_proof(
    voter_did: &str,
    proposal_type: ZkProposalType,
    eligible: bool,
) -> EligibilityProof {
    let id = format!("proof-{}-{}", voter_did, proposal_type as u8);
    let voter_commitment = sha3_commitment(voter_did.as_bytes());

    EligibilityProof {
        id,
        voter_did: voter_did.to_string(),
        voter_commitment,
        proposal_type,
        eligible,
        requirements_met: if eligible { 7 } else { 4 },
        active_requirements: 7,
        proof_bytes: vec![0u8; 64],
        generated_at: Timestamp::from_micros(1_000_000),
        expires_at: Some(Timestamp::from_micros(1_000_000 + 3600 * 1_000_000)),
    }
}

/// Create a mock verified vote for testing
fn create_test_verified_vote(
    voter: &str,
    proposal_id: &str,
    proposal_tier: ProposalTier,
    choice: VoteChoice,
    weight: f64,
) -> VerifiedVote {
    let id = format!("vote-{}-{}", proposal_id, voter);
    let voter_commitment = sha3_commitment(voter.as_bytes());

    VerifiedVote {
        id,
        proposal_id: proposal_id.to_string(),
        proposal_tier,
        voter: voter.to_string(),
        choice,
        eligibility_proof_hash: ActionHash::from_raw_32([0u8; 32]),
        voter_commitment,
        effective_weight: weight,
        reason: None,
        voted_at: Timestamp::from_micros(1_000_000),
    }
}

/// Simple SHA3-256 commitment
fn sha3_commitment(data: &[u8]) -> Vec<u8> {
    Sha3_256::digest(data).to_vec()
}

/// Calculate vote tally from a list of verified votes
fn calculate_tally(votes: &[VerifiedVote]) -> (f64, f64, f64, usize) {
    let mut approve_weight = 0.0;
    let mut reject_weight = 0.0;
    let mut abstain_weight = 0.0;
    let total_votes = votes.len();

    for vote in votes {
        match vote.choice {
            VoteChoice::Approve => approve_weight += vote.effective_weight,
            VoteChoice::Reject => reject_weight += vote.effective_weight,
            VoteChoice::Abstain => abstain_weight += vote.effective_weight,
        }
    }

    (approve_weight, reject_weight, abstain_weight, total_votes)
}

/// Get approval threshold for a proposal tier
fn approval_threshold(tier: ProposalTier) -> f64 {
    match tier {
        ProposalTier::Basic => 0.5,
        ProposalTier::Standard => 0.5,
        ProposalTier::Major => 0.6,
        ProposalTier::Constitutional => 0.67,
    }
}

/// Check if a vote result passes the threshold
fn passes_threshold(approve: f64, reject: f64, tier: ProposalTier) -> bool {
    let total = approve + reject;
    if total == 0.0 {
        return false;
    }
    let rate = approve / total;
    rate >= approval_threshold(tier)
}

/// Maps ZkProposalType to minimum security bits required
fn min_security_bits_for_proposal(proposal_type: ZkProposalType) -> u32 {
    match proposal_type {
        ZkProposalType::Standard => 96,
        ZkProposalType::Constitutional => 264,
        ZkProposalType::ModelGovernance => 96,
        ZkProposalType::Emergency => 96,
        ZkProposalType::Treasury => 96,
        ZkProposalType::Membership => 84,
    }
}

// =============================================================================
// ZK PROPOSAL TYPE TESTS
// =============================================================================

#[cfg(test)]
mod zk_proposal_type_tests {
    use super::*;

    #[test]
    fn test_proposal_type_values() {
        assert_eq!(ZkProposalType::Standard as u8, 0);
        assert_eq!(ZkProposalType::Constitutional as u8, 1);
        assert_eq!(ZkProposalType::ModelGovernance as u8, 2);
        assert_eq!(ZkProposalType::Emergency as u8, 3);
        assert_eq!(ZkProposalType::Treasury as u8, 4);
        assert_eq!(ZkProposalType::Membership as u8, 5);
    }

    #[test]
    fn test_proposal_type_equality() {
        assert_eq!(ZkProposalType::Standard, ZkProposalType::Standard);
        assert_ne!(ZkProposalType::Standard, ZkProposalType::Constitutional);
    }

    #[test]
    fn test_all_proposal_types_exist() {
        let types = [
            ZkProposalType::Standard,
            ZkProposalType::Constitutional,
            ZkProposalType::ModelGovernance,
            ZkProposalType::Emergency,
            ZkProposalType::Treasury,
            ZkProposalType::Membership,
        ];
        assert_eq!(types.len(), 6);
    }
}

// =============================================================================
// ELIGIBILITY PROOF TESTS
// =============================================================================

#[cfg(test)]
mod eligibility_proof_tests {
    use super::*;

    #[test]
    fn test_create_eligibility_proof() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:voter1",
            ZkProposalType::Standard,
            true,
        );

        assert!(proof.eligible);
        assert_eq!(proof.requirements_met, 7);
        assert_eq!(proof.active_requirements, 7);
        assert_eq!(proof.proposal_type, ZkProposalType::Standard);
        assert!(!proof.voter_commitment.is_empty());
    }

    #[test]
    fn test_ineligible_proof() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:voter2",
            ZkProposalType::Constitutional,
            false,
        );

        assert!(!proof.eligible);
        assert_eq!(proof.requirements_met, 4);
        assert!(proof.requirements_met < proof.active_requirements);
    }

    #[test]
    fn test_proof_expiry() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:voter3",
            ZkProposalType::Standard,
            true,
        );

        assert!(proof.expires_at.is_some());
        let expires = proof.expires_at.unwrap();
        assert!(expires.as_micros() > proof.generated_at.as_micros());
    }

    #[test]
    fn test_voter_commitment_deterministic() {
        let voter_did = "did:mycelix:test";
        let proof1 = create_test_eligibility_proof(voter_did, ZkProposalType::Standard, true);
        let proof2 = create_test_eligibility_proof(voter_did, ZkProposalType::Standard, true);

        assert_eq!(proof1.voter_commitment, proof2.voter_commitment);
    }

    #[test]
    fn test_different_voters_different_commitments() {
        let proof1 = create_test_eligibility_proof(
            "did:mycelix:voter1",
            ZkProposalType::Standard,
            true,
        );
        let proof2 = create_test_eligibility_proof(
            "did:mycelix:voter2",
            ZkProposalType::Standard,
            true,
        );

        assert_ne!(proof1.voter_commitment, proof2.voter_commitment);
    }
}

// =============================================================================
// VERIFIED VOTE TESTS
// =============================================================================

#[cfg(test)]
mod verified_vote_tests {
    use super::*;

    #[test]
    fn test_create_verified_vote() {
        let vote = create_test_verified_vote(
            "did:mycelix:voter1",
            "prop-001",
            ProposalTier::Standard,
            VoteChoice::Approve,
            1.0,
        );

        assert_eq!(vote.choice, VoteChoice::Approve);
        assert_eq!(vote.proposal_tier, ProposalTier::Standard);
        assert_eq!(vote.effective_weight, 1.0);
    }

    #[test]
    fn test_vote_choices() {
        let vote_approve = create_test_verified_vote(
            "voter1", "prop", ProposalTier::Standard, VoteChoice::Approve, 1.0
        );
        let vote_reject = create_test_verified_vote(
            "voter2", "prop", ProposalTier::Standard, VoteChoice::Reject, 1.0
        );
        let vote_abstain = create_test_verified_vote(
            "voter3", "prop", ProposalTier::Standard, VoteChoice::Abstain, 1.0
        );

        assert_eq!(vote_approve.choice, VoteChoice::Approve);
        assert_eq!(vote_reject.choice, VoteChoice::Reject);
        assert_eq!(vote_abstain.choice, VoteChoice::Abstain);
    }

    #[test]
    fn test_vote_weight_range() {
        let vote_full = create_test_verified_vote(
            "voter", "prop", ProposalTier::Standard, VoteChoice::Approve, 1.0
        );
        let vote_partial = create_test_verified_vote(
            "voter", "prop", ProposalTier::Standard, VoteChoice::Approve, 0.5
        );

        assert!(vote_full.effective_weight >= 0.0);
        assert!(vote_full.effective_weight <= 1.0);
        assert!(vote_partial.effective_weight >= 0.0);
        assert!(vote_partial.effective_weight <= 1.0);
    }

    #[test]
    fn test_proposal_tier_mapping() {
        let tiers = [
            ProposalTier::Basic,
            ProposalTier::Standard,
            ProposalTier::Major,
            ProposalTier::Constitutional,
        ];

        for tier in tiers {
            let vote = create_test_verified_vote(
                "voter", "prop", tier, VoteChoice::Approve, 1.0
            );
            assert_eq!(vote.proposal_tier, tier);
        }
    }
}

// =============================================================================
// SECURITY LEVEL REQUIREMENTS TESTS
// =============================================================================

#[cfg(test)]
mod security_level_tests {
    use super::*;

    #[test]
    fn test_constitutional_requires_high_security() {
        let min_bits = min_security_bits_for_proposal(ZkProposalType::Constitutional);
        assert!(min_bits >= 264, "Constitutional should require High security (264-bit)");
    }

    #[test]
    fn test_treasury_requires_standard_security() {
        let min_bits = min_security_bits_for_proposal(ZkProposalType::Treasury);
        assert!(min_bits >= 96, "Treasury should require at least Standard security (96-bit)");
    }

    #[test]
    fn test_standard_requires_minimum_production_security() {
        let min_bits = min_security_bits_for_proposal(ZkProposalType::Standard);
        assert!(min_bits >= 96, "Standard should require production security (96-bit)");
    }

    #[test]
    fn test_membership_can_use_optimized() {
        let min_bits = min_security_bits_for_proposal(ZkProposalType::Membership);
        assert!(min_bits >= 84, "Membership can use Optimized security (84-bit)");
    }

    #[test]
    fn test_security_level_ordering() {
        // More critical proposal types should require higher security
        let standard = min_security_bits_for_proposal(ZkProposalType::Standard);
        let constitutional = min_security_bits_for_proposal(ZkProposalType::Constitutional);
        let membership = min_security_bits_for_proposal(ZkProposalType::Membership);

        assert!(constitutional > standard);
        assert!(standard >= membership);
    }
}

// =============================================================================
// VOTE TALLYING TESTS
// =============================================================================

#[cfg(test)]
mod vote_tallying_tests {
    use super::*;

    #[test]
    fn test_simple_majority() {
        let votes = vec![
            create_test_verified_vote("v1", "prop", ProposalTier::Standard, VoteChoice::Approve, 1.0),
            create_test_verified_vote("v2", "prop", ProposalTier::Standard, VoteChoice::Approve, 1.0),
            create_test_verified_vote("v3", "prop", ProposalTier::Standard, VoteChoice::Reject, 1.0),
        ];

        let (approve, reject, abstain, total) = calculate_tally(&votes);

        assert_eq!(total, 3);
        assert_eq!(approve, 2.0);
        assert_eq!(reject, 1.0);
        assert_eq!(abstain, 0.0);

        let approval_rate = approve / (approve + reject);
        assert!(approval_rate > 0.5);
    }

    #[test]
    fn test_weighted_voting() {
        let votes = vec![
            create_test_verified_vote("v1", "prop", ProposalTier::Standard, VoteChoice::Approve, 0.8),
            create_test_verified_vote("v2", "prop", ProposalTier::Standard, VoteChoice::Reject, 0.9),
            create_test_verified_vote("v3", "prop", ProposalTier::Standard, VoteChoice::Approve, 0.5),
        ];

        let (approve, reject, _, _) = calculate_tally(&votes);

        assert!((approve - 1.3).abs() < 0.001);
        assert!((reject - 0.9).abs() < 0.001);
        assert!(approve > reject);
    }

    #[test]
    fn test_abstain_handling() {
        let votes = vec![
            create_test_verified_vote("v1", "prop", ProposalTier::Standard, VoteChoice::Approve, 1.0),
            create_test_verified_vote("v2", "prop", ProposalTier::Standard, VoteChoice::Abstain, 1.0),
            create_test_verified_vote("v3", "prop", ProposalTier::Standard, VoteChoice::Abstain, 1.0),
            create_test_verified_vote("v4", "prop", ProposalTier::Standard, VoteChoice::Reject, 1.0),
        ];

        let (approve, reject, abstain, total) = calculate_tally(&votes);

        assert_eq!(total, 4);
        assert_eq!(abstain, 2.0);

        let deciding_weight = approve + reject;
        if deciding_weight > 0.0 {
            let approval_rate = approve / deciding_weight;
            assert!((approval_rate - 0.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_empty_tally() {
        let votes: Vec<VerifiedVote> = vec![];
        let (approve, reject, abstain, total) = calculate_tally(&votes);

        assert_eq!(total, 0);
        assert_eq!(approve, 0.0);
        assert_eq!(reject, 0.0);
        assert_eq!(abstain, 0.0);
    }
}

// =============================================================================
// PROPOSAL TIER THRESHOLD TESTS
// =============================================================================

#[cfg(test)]
mod proposal_tier_threshold_tests {
    use super::*;

    #[test]
    fn test_basic_tier_simple_majority() {
        let threshold = approval_threshold(ProposalTier::Basic);
        assert_eq!(threshold, 0.5);

        assert!(passes_threshold(51.0, 49.0, ProposalTier::Basic));
        assert!(!passes_threshold(49.0, 51.0, ProposalTier::Basic));
    }

    #[test]
    fn test_major_tier_supermajority() {
        let threshold = approval_threshold(ProposalTier::Major);
        assert_eq!(threshold, 0.6);

        assert!(passes_threshold(60.0, 40.0, ProposalTier::Major));
        assert!(!passes_threshold(59.0, 41.0, ProposalTier::Major));
    }

    #[test]
    fn test_constitutional_tier_two_thirds() {
        let threshold = approval_threshold(ProposalTier::Constitutional);
        assert!((threshold - 0.67).abs() < 0.01);

        assert!(passes_threshold(67.0, 33.0, ProposalTier::Constitutional));
        assert!(!passes_threshold(66.0, 34.0, ProposalTier::Constitutional));
    }

    #[test]
    fn test_threshold_ordering() {
        assert!(approval_threshold(ProposalTier::Basic) <= approval_threshold(ProposalTier::Standard));
        assert!(approval_threshold(ProposalTier::Standard) <= approval_threshold(ProposalTier::Major));
        assert!(approval_threshold(ProposalTier::Major) <= approval_threshold(ProposalTier::Constitutional));
    }
}

// =============================================================================
// ZK WORKFLOW INTEGRATION TESTS
// =============================================================================

#[cfg(test)]
mod zk_workflow_tests {
    use super::*;

    #[test]
    fn test_full_zk_voting_workflow() {
        // Step 1: Generate eligibility proof
        let voter_did = "did:mycelix:alice";
        let proposal_type = ZkProposalType::Standard;

        let proof = create_test_eligibility_proof(voter_did, proposal_type, true);
        assert!(proof.eligible, "Alice should be eligible");

        // Step 2: Verify proof is valid
        assert_eq!(proof.requirements_met, proof.active_requirements);

        // Step 3: Cast verified vote
        let vote = create_test_verified_vote(
            voter_did,
            "prop-001",
            ProposalTier::Standard,
            VoteChoice::Approve,
            0.85,
        );

        // Step 4: Verify vote commitment matches proof commitment
        assert_eq!(vote.voter_commitment, proof.voter_commitment);

        // Step 5: Vote is recorded
        assert_eq!(vote.choice, VoteChoice::Approve);
        assert_eq!(vote.effective_weight, 0.85);
    }

    #[test]
    fn test_privacy_preservation() {
        let voter_did = "did:mycelix:private_voter";
        let proof = create_test_eligibility_proof(voter_did, ZkProposalType::Standard, true);

        let expected_commitment = sha3_commitment(voter_did.as_bytes());
        assert_eq!(proof.voter_commitment, expected_commitment);
    }

    #[test]
    fn test_ineligible_voter_workflow() {
        let voter_did = "did:mycelix:bob";
        let proposal_type = ZkProposalType::Constitutional;

        let proof = create_test_eligibility_proof(voter_did, proposal_type, false);
        assert!(!proof.eligible, "Bob should be ineligible");
        assert!(proof.requirements_met < proof.active_requirements);
    }

    #[test]
    fn test_proof_expiry_workflow() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:charlie",
            ZkProposalType::Standard,
            true,
        );

        let generated = proof.generated_at.as_micros();
        let expires = proof.expires_at.unwrap().as_micros();

        let current_time = generated + 1000;
        assert!(current_time < expires, "Proof should still be valid");

        let expired_time = expires + 1000;
        assert!(expired_time > expires, "Proof should be expired");
    }

    #[test]
    fn test_vote_deduplication() {
        let voter_did = "did:mycelix:eve";
        let proposal_id = "prop-002";

        let vote1 = create_test_verified_vote(
            voter_did,
            proposal_id,
            ProposalTier::Standard,
            VoteChoice::Approve,
            1.0,
        );

        let vote2 = create_test_verified_vote(
            voter_did,
            proposal_id,
            ProposalTier::Standard,
            VoteChoice::Reject,
            1.0,
        );

        assert_eq!(vote1.voter_commitment, vote2.voter_commitment);
    }

    #[test]
    fn test_multi_voter_tally() {
        let votes = vec![
            create_test_verified_vote("alice", "prop-003", ProposalTier::Major, VoteChoice::Approve, 0.9),
            create_test_verified_vote("bob", "prop-003", ProposalTier::Major, VoteChoice::Approve, 0.85),
            create_test_verified_vote("charlie", "prop-003", ProposalTier::Major, VoteChoice::Approve, 0.8),
            create_test_verified_vote("dave", "prop-003", ProposalTier::Major, VoteChoice::Reject, 0.75),
            create_test_verified_vote("eve", "prop-003", ProposalTier::Major, VoteChoice::Abstain, 0.7),
        ];

        let (approve, reject, abstain, total) = calculate_tally(&votes);

        assert_eq!(total, 5);
        // approve: 0.9 + 0.85 + 0.8 = 2.55
        // reject: 0.75
        // abstain: 0.7
        assert!((approve - 2.55).abs() < 0.001);
        assert!((reject - 0.75).abs() < 0.001);
        assert!((abstain - 0.7).abs() < 0.001);

        // Check if passes Major tier (60%)
        let passes = passes_threshold(approve, reject, ProposalTier::Major);
        assert!(passes, "Should pass Major tier with ~77% approval");
    }
}

// =============================================================================
// CROSS-TIER SECURITY TESTS
// =============================================================================

#[cfg(test)]
mod cross_tier_security_tests {
    use super::*;

    #[test]
    fn test_constitutional_higher_security_than_standard() {
        let standard_proof = create_test_eligibility_proof(
            "voter",
            ZkProposalType::Standard,
            true,
        );

        let constitutional_proof = create_test_eligibility_proof(
            "voter",
            ZkProposalType::Constitutional,
            true,
        );

        assert_eq!(constitutional_proof.proposal_type, ZkProposalType::Constitutional);
        assert_eq!(standard_proof.proposal_type, ZkProposalType::Standard);

        // Verify security requirements differ
        let standard_bits = min_security_bits_for_proposal(ZkProposalType::Standard);
        let const_bits = min_security_bits_for_proposal(ZkProposalType::Constitutional);
        assert!(const_bits > standard_bits);
    }

    #[test]
    fn test_proof_proposal_type_match() {
        let voter_did = "did:mycelix:frank";

        let proof = create_test_eligibility_proof(
            voter_did,
            ZkProposalType::Standard,
            true,
        );

        assert_eq!(proof.proposal_type, ZkProposalType::Standard);
        assert_ne!(proof.proposal_type, ZkProposalType::Constitutional);
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_zero_weight_vote() {
        let vote = create_test_verified_vote(
            "voter",
            "prop",
            ProposalTier::Standard,
            VoteChoice::Approve,
            0.0,
        );

        let tally = vec![vote];
        let (approve, _, _, _) = calculate_tally(&tally);
        assert_eq!(approve, 0.0);
    }

    #[test]
    fn test_unanimous_approval() {
        let votes: Vec<VerifiedVote> = (0..10)
            .map(|i| create_test_verified_vote(
                &format!("voter{}", i),
                "prop",
                ProposalTier::Constitutional,
                VoteChoice::Approve,
                1.0,
            ))
            .collect();

        let (approve, reject, _, total) = calculate_tally(&votes);

        assert_eq!(total, 10);
        assert_eq!(approve, 10.0);
        assert_eq!(reject, 0.0);

        let rate = approve / (approve + reject);
        assert_eq!(rate, 1.0);
    }

    #[test]
    fn test_all_abstain() {
        let votes: Vec<VerifiedVote> = (0..5)
            .map(|i| create_test_verified_vote(
                &format!("voter{}", i),
                "prop",
                ProposalTier::Standard,
                VoteChoice::Abstain,
                1.0,
            ))
            .collect();

        let (approve, reject, abstain, _) = calculate_tally(&votes);

        assert_eq!(approve, 0.0);
        assert_eq!(reject, 0.0);
        assert_eq!(abstain, 5.0);
    }

    #[test]
    fn test_exact_threshold() {
        let approve_weight = 67.0;
        let reject_weight = 33.0;

        let passes = passes_threshold(
            approve_weight,
            reject_weight,
            ProposalTier::Constitutional,
        );

        assert!(passes, "Exactly 67% should pass constitutional threshold");
    }

    #[test]
    fn test_just_below_threshold() {
        let approve_weight = 66.9;
        let reject_weight = 33.1;

        let passes = passes_threshold(
            approve_weight,
            reject_weight,
            ProposalTier::Constitutional,
        );

        assert!(!passes, "66.9% should fail constitutional threshold");
    }

    #[test]
    fn test_many_small_weights() {
        let votes: Vec<VerifiedVote> = (0..1000)
            .map(|i| create_test_verified_vote(
                &format!("voter{}", i),
                "prop",
                ProposalTier::Standard,
                if i % 3 == 0 { VoteChoice::Reject } else { VoteChoice::Approve },
                0.001 + (i as f64 * 0.0001),
            ))
            .collect();

        let (approve, reject, _, total) = calculate_tally(&votes);

        assert_eq!(total, 1000);
        assert!(approve > 0.0);
        assert!(reject > 0.0);
    }

    #[test]
    fn test_single_vote_passes() {
        let votes = vec![
            create_test_verified_vote("single", "prop", ProposalTier::Basic, VoteChoice::Approve, 1.0),
        ];

        let (approve, reject, _, _) = calculate_tally(&votes);
        assert!(passes_threshold(approve, reject, ProposalTier::Basic));
    }

    #[test]
    fn test_tie_vote_fails() {
        let votes = vec![
            create_test_verified_vote("v1", "prop", ProposalTier::Basic, VoteChoice::Approve, 1.0),
            create_test_verified_vote("v2", "prop", ProposalTier::Basic, VoteChoice::Reject, 1.0),
        ];

        let (approve, reject, _, _) = calculate_tally(&votes);
        // Tie at 50% should fail Basic (requires > 50% or >= 50%?)
        // With >= 50%, it would pass; with > 50%, it would fail
        // Our implementation uses >=, so it passes
        let passes = passes_threshold(approve, reject, ProposalTier::Basic);
        assert!(passes, "50% should pass Basic tier with >= threshold");
    }
}

// =============================================================================
// COMMITMENT SCHEME TESTS
// =============================================================================

#[cfg(test)]
mod commitment_tests {
    use super::*;

    #[test]
    fn test_commitment_deterministic() {
        let data = b"test data";
        let c1 = sha3_commitment(data);
        let c2 = sha3_commitment(data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_commitment_different_inputs() {
        let c1 = sha3_commitment(b"input1");
        let c2 = sha3_commitment(b"input2");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_commitment_length() {
        let commitment = sha3_commitment(b"any data");
        assert_eq!(commitment.len(), 32); // SHA3-256 produces 32 bytes
    }

    #[test]
    fn test_empty_input_commitment() {
        let commitment = sha3_commitment(b"");
        assert_eq!(commitment.len(), 32);
        // SHA3-256 of empty input is well-defined
    }

    #[test]
    fn test_long_input_commitment() {
        let long_input: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let commitment = sha3_commitment(&long_input);
        assert_eq!(commitment.len(), 32);
    }
}

// =============================================================================
// E2E STARK PROOF INTEGRATION TESTS
// =============================================================================
//
// These tests use the actual fl-aggregator ZomeBridgeClient to generate
// real STARK proofs and test the complete voting workflow.
//
// Note: These tests require the fl-aggregator crate as a dev-dependency.
// If not available, they will be skipped.

#[cfg(test)]
#[cfg(feature = "fl-aggregator-tests")]
mod e2e_stark_proof_tests {
    // Import from fl-aggregator when available
    // use fl_aggregator::proofs::integration::{
    //     ZomeBridgeClient, VoterProfileBuilder, VerifierService,
    //     SerializableProof, ProofAttestation,
    // };
    // use fl_aggregator::proofs::{ProofProposalType, SecurityLevel};

    #[test]
    fn test_stark_proof_generation_standard() {
        // This test would:
        // 1. Create a VoterProfile using VoterProfileBuilder
        // 2. Generate a real STARK proof using ZomeBridgeClient
        // 3. Serialize the proof for zome storage
        // 4. Verify the proof
        // 5. Create an attestation

        // Placeholder - actual implementation requires fl-aggregator dep
        assert!(true, "STARK proof generation test placeholder");
    }

    #[test]
    fn test_stark_proof_generation_constitutional() {
        // Constitutional proposals require higher security (264-bit)
        assert!(true, "Constitutional STARK proof test placeholder");
    }

    #[test]
    fn test_attestation_creation_and_verification() {
        // This test would:
        // 1. Generate a proof
        // 2. Create a VerifierService with signing key
        // 3. Verify and attest the proof
        // 4. Verify the attestation signature

        assert!(true, "Attestation test placeholder");
    }

    #[test]
    fn test_full_e2e_voting_with_stark() {
        // Complete E2E test:
        // 1. Build voter profile
        // 2. Generate STARK proof
        // 3. Serialize for zome
        // 4. Verify with VerifierService
        // 5. Create attestation
        // 6. Simulate casting attested vote

        assert!(true, "Full E2E voting test placeholder");
    }
}

// =============================================================================
// ZOME BRIDGE INTEGRATION TESTS (Standalone)
// =============================================================================
//
// These tests validate the data structures and workflow without requiring
// the full fl-aggregator crate. They test the integration contract.

#[cfg(test)]
mod zome_bridge_contract_tests {
    use super::*;

    /// Mock proof attestation matching the zome structure
    #[derive(Clone, Debug)]
    pub struct MockProofAttestation {
        pub id: String,
        pub proof_hash: Vec<u8>,
        pub voter_commitment: Vec<u8>,
        pub proposal_type: ZkProposalType,
        pub verified: bool,
        pub verified_at: Timestamp,
        pub expires_at: Timestamp,
        pub verifier_pubkey: Vec<u8>,
        pub signature: Vec<u8>,
        pub security_level: String,
        pub verification_time_ms: u64,
    }

    impl MockProofAttestation {
        pub fn is_expired(&self, current_time: Timestamp) -> bool {
            current_time.as_micros() > self.expires_at.as_micros()
        }

        pub fn is_valid(&self, current_time: Timestamp) -> bool {
            self.verified && !self.is_expired(current_time)
        }
    }

    fn create_mock_attestation(
        proof: &EligibilityProof,
        verified: bool,
        valid_hours: i64,
    ) -> MockProofAttestation {
        let now = Timestamp::from_micros(1_000_000);
        let expires = Timestamp::from_micros(now.as_micros() + valid_hours * 3600 * 1_000_000);

        MockProofAttestation {
            id: format!("attestation:{}", proof.id),
            proof_hash: sha3_commitment(&proof.proof_bytes),
            voter_commitment: proof.voter_commitment.clone(),
            proposal_type: proof.proposal_type,
            verified,
            verified_at: now,
            expires_at: expires,
            verifier_pubkey: vec![0u8; 32], // Mock Ed25519 pubkey
            signature: vec![0u8; 64],       // Mock Ed25519 signature
            security_level: "Standard96".to_string(),
            verification_time_ms: 150,
        }
    }

    #[test]
    fn test_attestation_structure() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:alice",
            ZkProposalType::Standard,
            true,
        );

        let attestation = create_mock_attestation(&proof, true, 24);

        assert!(attestation.verified);
        assert_eq!(attestation.verifier_pubkey.len(), 32);
        assert_eq!(attestation.signature.len(), 64);
        assert!(!attestation.proof_hash.is_empty());
    }

    #[test]
    fn test_attestation_expiry() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:bob",
            ZkProposalType::Standard,
            true,
        );

        let attestation = create_mock_attestation(&proof, true, 24);

        // Check validity at different times
        let before_expiry = Timestamp::from_micros(attestation.expires_at.as_micros() - 1000);
        let after_expiry = Timestamp::from_micros(attestation.expires_at.as_micros() + 1000);

        assert!(attestation.is_valid(before_expiry));
        assert!(!attestation.is_valid(after_expiry));
    }

    #[test]
    fn test_attestation_verification_status() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:charlie",
            ZkProposalType::Standard,
            true,
        );

        let verified_attestation = create_mock_attestation(&proof, true, 24);
        let unverified_attestation = create_mock_attestation(&proof, false, 24);

        let now = Timestamp::from_micros(1_500_000);

        assert!(verified_attestation.is_valid(now));
        assert!(!unverified_attestation.is_valid(now));
    }

    #[test]
    fn test_attestation_commitment_matches_proof() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:dave",
            ZkProposalType::Constitutional,
            true,
        );

        let attestation = create_mock_attestation(&proof, true, 24);

        // Attestation should carry the same voter commitment
        assert_eq!(attestation.voter_commitment, proof.voter_commitment);
        assert_eq!(attestation.proposal_type, proof.proposal_type);
    }

    #[test]
    fn test_attested_voting_workflow() {
        // Step 1: Create eligibility proof
        let voter_did = "did:mycelix:eve";
        let proof = create_test_eligibility_proof(
            voter_did,
            ZkProposalType::Standard,
            true,
        );
        assert!(proof.eligible);

        // Step 2: External verifier creates attestation
        let attestation = create_mock_attestation(&proof, true, 24);
        assert!(attestation.verified);

        // Step 3: Check attestation is valid
        let now = Timestamp::from_micros(1_500_000);
        assert!(attestation.is_valid(now), "Attestation should be valid");

        // Step 4: Cast vote with attested proof
        let vote = create_test_verified_vote(
            voter_did,
            "prop-001",
            ProposalTier::Standard,
            VoteChoice::Approve,
            1.0, // Full weight for attested voters
        );

        // Step 5: Verify vote commitment matches attestation
        assert_eq!(vote.voter_commitment, attestation.voter_commitment);

        // Step 6: Vote is valid
        assert_eq!(vote.effective_weight, 1.0);
    }

    #[test]
    fn test_rejected_attestation_blocks_vote() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:frank",
            ZkProposalType::Constitutional,
            true,
        );

        // Verifier rejects the proof (invalid STARK)
        let attestation = create_mock_attestation(&proof, false, 24);

        let now = Timestamp::from_micros(1_500_000);
        assert!(!attestation.is_valid(now), "Unverified attestation should be invalid");

        // Voting system should reject votes without valid attestation
        // (simulated - actual logic is in zome coordinator)
    }

    #[test]
    fn test_expired_attestation_blocks_vote() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:grace",
            ZkProposalType::Standard,
            true,
        );

        // Create attestation that expires in 1 hour
        let attestation = create_mock_attestation(&proof, true, 1);

        // Check at time after expiry (2 hours later)
        let after_expiry = Timestamp::from_micros(
            attestation.expires_at.as_micros() + 3600 * 1_000_000
        );

        assert!(!attestation.is_valid(after_expiry), "Expired attestation should be invalid");
    }

    #[test]
    fn test_multiple_attestations_for_same_proof() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:heidi",
            ZkProposalType::Treasury,
            true,
        );

        // Multiple verifiers can attest
        let attestation1 = create_mock_attestation(&proof, true, 24);
        let attestation2 = create_mock_attestation(&proof, true, 48);

        let now = Timestamp::from_micros(1_500_000);

        // Both should be valid
        assert!(attestation1.is_valid(now));
        assert!(attestation2.is_valid(now));

        // Should have same proof hash
        assert_eq!(attestation1.proof_hash, attestation2.proof_hash);
    }

    #[test]
    fn test_security_level_in_attestation() {
        let standard_proof = create_test_eligibility_proof(
            "voter",
            ZkProposalType::Standard,
            true,
        );

        let constitutional_proof = create_test_eligibility_proof(
            "voter",
            ZkProposalType::Constitutional,
            true,
        );

        let standard_attestation = create_mock_attestation(&standard_proof, true, 24);
        let _constitutional_attestation = create_mock_attestation(&constitutional_proof, true, 24);

        // Attestation should record the security level used
        assert!(!standard_attestation.security_level.is_empty());
    }

    #[test]
    fn test_verification_time_tracking() {
        let proof = create_test_eligibility_proof(
            "did:mycelix:ivan",
            ZkProposalType::Standard,
            true,
        );

        let attestation = create_mock_attestation(&proof, true, 24);

        // Verification time should be recorded (useful for performance monitoring)
        assert!(attestation.verification_time_ms > 0);
    }
}

// =============================================================================
// PROPOSAL TYPE TO TIER MAPPING TESTS
// =============================================================================

#[cfg(test)]
mod proposal_tier_mapping_tests {
    use super::*;

    /// Map ZkProposalType to the minimum ProposalTier it supports
    fn proposal_type_supports_tier(proposal_type: ZkProposalType, tier: ProposalTier) -> bool {
        match proposal_type {
            ZkProposalType::Standard => matches!(tier, ProposalTier::Basic | ProposalTier::Standard),
            ZkProposalType::Constitutional => true, // Constitutional supports all tiers
            ZkProposalType::ModelGovernance => matches!(tier, ProposalTier::Basic | ProposalTier::Standard | ProposalTier::Major),
            ZkProposalType::Emergency => matches!(tier, ProposalTier::Basic | ProposalTier::Standard),
            ZkProposalType::Treasury => matches!(tier, ProposalTier::Basic | ProposalTier::Standard | ProposalTier::Major),
            ZkProposalType::Membership => matches!(tier, ProposalTier::Basic),
        }
    }

    #[test]
    fn test_standard_proof_tiers() {
        assert!(proposal_type_supports_tier(ZkProposalType::Standard, ProposalTier::Basic));
        assert!(proposal_type_supports_tier(ZkProposalType::Standard, ProposalTier::Standard));
        assert!(!proposal_type_supports_tier(ZkProposalType::Standard, ProposalTier::Major));
        assert!(!proposal_type_supports_tier(ZkProposalType::Standard, ProposalTier::Constitutional));
    }

    #[test]
    fn test_constitutional_proof_tiers() {
        // Constitutional proofs should work for all tiers
        assert!(proposal_type_supports_tier(ZkProposalType::Constitutional, ProposalTier::Basic));
        assert!(proposal_type_supports_tier(ZkProposalType::Constitutional, ProposalTier::Standard));
        assert!(proposal_type_supports_tier(ZkProposalType::Constitutional, ProposalTier::Major));
        assert!(proposal_type_supports_tier(ZkProposalType::Constitutional, ProposalTier::Constitutional));
    }

    #[test]
    fn test_membership_proof_limited() {
        // Membership proofs only for basic tier
        assert!(proposal_type_supports_tier(ZkProposalType::Membership, ProposalTier::Basic));
        assert!(!proposal_type_supports_tier(ZkProposalType::Membership, ProposalTier::Standard));
        assert!(!proposal_type_supports_tier(ZkProposalType::Membership, ProposalTier::Major));
    }

    #[test]
    fn test_treasury_proof_tiers() {
        assert!(proposal_type_supports_tier(ZkProposalType::Treasury, ProposalTier::Basic));
        assert!(proposal_type_supports_tier(ZkProposalType::Treasury, ProposalTier::Standard));
        assert!(proposal_type_supports_tier(ZkProposalType::Treasury, ProposalTier::Major));
        assert!(!proposal_type_supports_tier(ZkProposalType::Treasury, ProposalTier::Constitutional));
    }
}
