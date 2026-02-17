//! Proof-Verified Governance
//!
//! Integrates zkSTARK proofs with governance for verified voting eligibility.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::governance::verified::{VerifiedVoteManager, VoteWithProof};
//!
//! let mut manager = VerifiedVoteManager::new(config);
//!
//! // Cast vote with eligibility proof
//! let receipt = manager.cast_verified_vote(
//!     "MIP-0001",
//!     &voter_profile,
//!     VoteChoice::For,
//!     eligibility_proof,
//! )?;
//! ```

use super::{
    GovernanceError, GovernanceResult, ProposalType, Vote, VoteChoice, VoteManager, VoteReceipt,
    VoterProfile,
};
use crate::proofs::{
    ProofConfig, ProofError, ProofProposalType, ProofResult, ProofVoterProfile, SecurityLevel,
    VoteEligibilityProof,
};
use crate::proofs::integration::{EligibilityProofBundle, EligibilityProofGenerator};
use serde::{Deserialize, Serialize};

/// Configuration for verified governance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedGovernanceConfig {
    /// Whether proofs are required for voting
    pub require_proofs: bool,

    /// Proof configuration
    #[serde(skip)]
    pub proof_config: ProofConfig,

    /// Whether to verify proofs synchronously
    pub sync_verification: bool,
}

impl Default for VerifiedGovernanceConfig {
    fn default() -> Self {
        Self {
            require_proofs: false,
            proof_config: ProofConfig::default(),
            sync_verification: true,
        }
    }
}

impl VerifiedGovernanceConfig {
    /// Enable proof requirement
    pub fn with_proofs(mut self) -> Self {
        self.require_proofs = true;
        self
    }

    /// Use Standard96 security
    pub fn with_standard_security(mut self) -> Self {
        self.proof_config.security_level = SecurityLevel::Standard96;
        self
    }

    /// Use Standard128 security
    pub fn with_high_security(mut self) -> Self {
        self.proof_config.security_level = SecurityLevel::Standard128;
        self
    }
}

/// A vote with attached eligibility proof
#[derive(Clone)]
pub struct VoteWithProof {
    /// The underlying vote
    pub vote: Vote,

    /// Eligibility proof bundle
    pub proof: Option<EligibilityProofBundle>,

    /// Whether the proof has been verified
    pub verified: bool,
}

impl VoteWithProof {
    /// Create a vote without proof
    pub fn new(vote: Vote) -> Self {
        Self {
            vote,
            proof: None,
            verified: false,
        }
    }

    /// Create a vote with proof
    pub fn with_proof(vote: Vote, proof: EligibilityProofBundle) -> Self {
        Self {
            vote,
            proof: Some(proof),
            verified: false,
        }
    }

    /// Verify the eligibility proof
    pub fn verify(&mut self) -> ProofResult<bool> {
        if let Some(ref proof) = self.proof {
            let result = proof.verify()?;
            self.verified = result.valid && proof.is_eligible();
            Ok(self.verified)
        } else {
            Err(ProofError::InvalidPublicInputs("No proof attached".to_string()))
        }
    }

    /// Check if vote is verified eligible
    pub fn is_verified_eligible(&self) -> bool {
        self.verified && self.proof.as_ref().map(|p| p.is_eligible()).unwrap_or(false)
    }
}

/// Vote manager that verifies eligibility proofs
pub struct VerifiedVoteManager {
    inner: VoteManager,
    config: VerifiedGovernanceConfig,
}

impl VerifiedVoteManager {
    /// Create a new verified vote manager
    pub fn new(config: VerifiedGovernanceConfig) -> Self {
        Self {
            inner: VoteManager::new(),
            config,
        }
    }

    /// Cast a vote with eligibility proof
    pub fn cast_verified_vote(
        &mut self,
        proposal_id: impl Into<String>,
        voter_profile: &VoterProfile,
        proposal_type: ProposalType,
        choice: VoteChoice,
        proof: EligibilityProofBundle,
        signature: Vec<u8>,
    ) -> GovernanceResult<VoteReceipt> {
        // Verify the proof
        let result = proof.verify().map_err(|e| {
            GovernanceError::InsufficientEligibility(format!("Proof verification failed: {}", e))
        })?;

        if !result.valid {
            return Err(GovernanceError::InsufficientEligibility(
                format!("Proof is invalid: {:?}", result.details),
            ));
        }

        if !proof.is_eligible() {
            return Err(GovernanceError::InsufficientEligibility(
                format!(
                    "Voter not eligible: met {}/{} requirements",
                    proof.requirements_met(),
                    proof.active_requirements()
                ),
            ));
        }

        // Verify proof matches proposal type
        if let Some(proof_proposal_type) = proof.proposal_type() {
            let expected = convert_proposal_type(proposal_type);
            if proof_proposal_type != expected {
                return Err(GovernanceError::InsufficientEligibility(
                    format!(
                        "Proof is for {:?}, but voting on {:?}",
                        proof_proposal_type, proposal_type
                    ),
                ));
            }
        }

        // Calculate effective weight based on proof
        let weight = super::weight::calculate_vote_weight(
            voter_profile.matl_score,
            voter_profile.stake,
            voter_profile.recent_participation_rate(),
        );

        // Cast the vote
        self.inner.cast_vote(
            proposal_id,
            voter_profile.did.clone(),
            choice,
            weight,
            signature,
        )
    }

    /// Cast a vote with auto-generated proof
    pub fn cast_vote_with_auto_proof(
        &mut self,
        proposal_id: impl Into<String>,
        voter_profile: &VoterProfile,
        proposal_type: ProposalType,
        choice: VoteChoice,
        signature: Vec<u8>,
    ) -> GovernanceResult<VoteReceipt> {
        // Convert to proof voter profile
        let proof_profile = convert_voter_profile(voter_profile);
        let proof_proposal_type = convert_proposal_type(proposal_type);

        // Generate eligibility proof
        let proof = VoteEligibilityProof::generate(
            &proof_profile,
            proof_proposal_type,
            self.config.proof_config.clone(),
        )
        .map_err(|e| {
            GovernanceError::InsufficientEligibility(format!("Failed to generate proof: {}", e))
        })?;

        let bundle = EligibilityProofBundle::new(proof);

        self.cast_verified_vote(
            proposal_id,
            voter_profile,
            proposal_type,
            choice,
            bundle,
            signature,
        )
    }

    /// Cast a vote without proof (if proofs are optional)
    pub fn cast_unverified_vote(
        &mut self,
        proposal_id: impl Into<String>,
        voter_did: impl Into<String>,
        choice: VoteChoice,
        weight: f32,
        signature: Vec<u8>,
    ) -> GovernanceResult<VoteReceipt> {
        if self.config.require_proofs {
            return Err(GovernanceError::InsufficientEligibility(
                "Proofs are required for voting".to_string(),
            ));
        }

        self.inner.cast_vote(proposal_id, voter_did, choice, weight, signature)
    }

    /// Get vote for a voter on a proposal
    pub fn get_vote(&self, proposal_id: &str, voter_did: &str) -> Option<&Vote> {
        self.inner.get_vote(proposal_id, voter_did)
    }

    /// Tally votes for a proposal
    pub fn tally_votes(
        &mut self,
        proposal_id: &str,
        proposal_type: ProposalType,
        eligible_weight: f32,
    ) -> GovernanceResult<super::VoteTally> {
        self.inner.tally_votes(proposal_id, proposal_type, eligible_weight)
    }

    /// Get the inner vote manager (for read operations)
    pub fn inner(&self) -> &VoteManager {
        &self.inner
    }

    /// Check if proofs are required
    pub fn requires_proofs(&self) -> bool {
        self.config.require_proofs
    }

    /// Get governance configuration
    pub fn config(&self) -> &VerifiedGovernanceConfig {
        &self.config
    }
}

/// Convert governance ProposalType to proof ProposalType
pub fn convert_proposal_type(pt: ProposalType) -> ProofProposalType {
    match pt {
        ProposalType::Standard => ProofProposalType::Standard,
        ProposalType::Constitutional => ProofProposalType::Constitutional,
        ProposalType::ModelGovernance => ProofProposalType::ModelGovernance,
        ProposalType::Emergency => ProofProposalType::Emergency,
        ProposalType::Treasury => ProofProposalType::Treasury,
        ProposalType::Membership => ProofProposalType::Membership,
    }
}

/// Convert VoterProfile to ProofVoterProfile
pub fn convert_voter_profile(voter: &VoterProfile) -> ProofVoterProfile {
    ProofVoterProfile {
        did: voter.did.clone(),
        assurance_level: voter.assurance_level,
        matl_score: voter.matl_score,
        stake: voter.stake,
        account_age_days: voter.account_age_days(),
        participation_rate: voter.recent_participation_rate(),
        has_humanity_proof: voter.has_humanity_proof,
        fl_contributions: voter.fl_contributions,
    }
}

/// Implement EligibilityProofGenerator for VoterProfile
impl EligibilityProofGenerator for VoterProfile {
    fn get_voter_profile(&self) -> ProofVoterProfile {
        convert_voter_profile(self)
    }
}

/// Batch verify eligibility for multiple voters
pub fn batch_verify_eligibility(
    voters: &[VoterProfile],
    proposal_type: ProposalType,
    config: ProofConfig,
) -> Vec<(String, bool, Option<EligibilityProofBundle>)> {
    let proof_type = convert_proposal_type(proposal_type);

    voters.iter().map(|voter| {
        let profile = convert_voter_profile(voter);
        match VoteEligibilityProof::generate(&profile, proof_type, config.clone()) {
            Ok(proof) => {
                let bundle = EligibilityProofBundle::new(proof);
                let eligible = bundle.verify().map(|r| r.valid && bundle.is_eligible()).unwrap_or(false);
                (voter.did.clone(), eligible, Some(bundle))
            }
            Err(_) => (voter.did.clone(), false, None),
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use std::collections::HashMap;

    fn test_config() -> VerifiedGovernanceConfig {
        VerifiedGovernanceConfig {
            require_proofs: true,
            proof_config: ProofConfig {
                security_level: SecurityLevel::Standard96,
                parallel: false,
                max_proof_size: 0,
            },
            sync_verification: true,
        }
    }

    fn eligible_voter() -> VoterProfile {
        VoterProfile {
            did: "did:mycelix:voter".to_string(),
            assurance_level: 3,
            matl_score: 0.8,
            stake: 600.0,
            account_created: Utc::now() - Duration::days(100),
            has_humanity_proof: true,
            fl_contributions: 25,
            recent_votes: 8,
            recent_eligible_proposals: 10,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_verified_vote_manager_creation() {
        let config = test_config();
        let manager = VerifiedVoteManager::new(config);
        assert!(manager.requires_proofs());
    }

    #[test]
    fn test_cast_vote_with_auto_proof() {
        let config = test_config();
        let mut manager = VerifiedVoteManager::new(config);
        let voter = eligible_voter();

        let receipt = manager.cast_vote_with_auto_proof(
            "MIP-0001",
            &voter,
            ProposalType::Standard,
            VoteChoice::For,
            vec![],
        ).unwrap();

        assert_eq!(receipt.proposal_id, "MIP-0001");
        assert_eq!(receipt.voter_did, "did:mycelix:voter");
    }

    #[test]
    fn test_unverified_vote_rejected_when_proofs_required() {
        let config = test_config();
        let mut manager = VerifiedVoteManager::new(config);

        let result = manager.cast_unverified_vote(
            "MIP-0001",
            "did:mycelix:voter",
            VoteChoice::For,
            10.0,
            vec![],
        );

        assert!(matches!(result, Err(GovernanceError::InsufficientEligibility(_))));
    }

    #[test]
    fn test_unverified_vote_allowed_when_proofs_optional() {
        let mut config = test_config();
        config.require_proofs = false;
        let mut manager = VerifiedVoteManager::new(config);

        let receipt = manager.cast_unverified_vote(
            "MIP-0001",
            "did:mycelix:voter",
            VoteChoice::For,
            10.0,
            vec![],
        ).unwrap();

        assert_eq!(receipt.proposal_id, "MIP-0001");
    }

    #[test]
    fn test_ineligible_voter_rejected() {
        let config = test_config();
        let mut manager = VerifiedVoteManager::new(config);

        // Low assurance voter
        let voter = VoterProfile {
            did: "did:mycelix:ineligible".to_string(),
            assurance_level: 0,  // Too low for Standard (requires E1)
            matl_score: 0.1,
            stake: 0.0,
            account_created: Utc::now(),
            has_humanity_proof: false,
            fl_contributions: 0,
            recent_votes: 0,
            recent_eligible_proposals: 0,
            metadata: HashMap::new(),
        };

        let result = manager.cast_vote_with_auto_proof(
            "MIP-0001",
            &voter,
            ProposalType::Standard,
            VoteChoice::For,
            vec![],
        );

        assert!(matches!(result, Err(GovernanceError::InsufficientEligibility(_))));
    }

    #[test]
    fn test_proposal_type_conversion() {
        assert_eq!(convert_proposal_type(ProposalType::Standard), ProofProposalType::Standard);
        assert_eq!(convert_proposal_type(ProposalType::Constitutional), ProofProposalType::Constitutional);
        assert_eq!(convert_proposal_type(ProposalType::Emergency), ProofProposalType::Emergency);
    }

    #[test]
    fn test_voter_profile_conversion() {
        let voter = eligible_voter();
        let proof_profile = convert_voter_profile(&voter);

        assert_eq!(proof_profile.did, voter.did);
        assert_eq!(proof_profile.assurance_level, voter.assurance_level);
        assert_eq!(proof_profile.matl_score, voter.matl_score);
        assert_eq!(proof_profile.stake, voter.stake);
    }

    #[test]
    fn test_batch_verify_eligibility() {
        let voters = vec![
            eligible_voter(),
            VoterProfile {
                did: "did:mycelix:ineligible".to_string(),
                assurance_level: 0,
                ..eligible_voter()
            },
        ];

        let results = batch_verify_eligibility(
            &voters,
            ProposalType::Standard,
            ProofConfig {
                security_level: SecurityLevel::Standard96,
                parallel: false,
                max_proof_size: 0,
            },
        );

        assert_eq!(results.len(), 2);
        // First voter should be eligible
        assert!(results[0].1);
        // Second voter should be ineligible
        assert!(!results[1].1);
    }
}
