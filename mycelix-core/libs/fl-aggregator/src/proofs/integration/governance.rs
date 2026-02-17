//! Governance Proof Integration
//!
//! Integrates vote eligibility proofs with the governance system.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::EligibilityProofGenerator;
//!
//! // Generate a proof that voter is eligible for constitutional vote
//! let bundle = voter.generate_eligibility_proof(
//!     ProofProposalType::Constitutional,
//!     config,
//! )?;
//!
//! // Verify the proof
//! let result = bundle.verify()?;
//! assert!(result.valid && bundle.is_eligible());
//! ```

use crate::proofs::{
    VoteEligibilityProof, ProofProposalType, ProofVoterProfile,
    ProofConfig, ProofResult, VerificationResult, compute_voter_commitment,
};
use serde::{Deserialize, Serialize};

/// Bundle of proofs for vote eligibility
#[derive(Clone)]
pub struct EligibilityProofBundle {
    /// Proof that voter meets requirements
    pub eligibility_proof: VoteEligibilityProof,
}

impl EligibilityProofBundle {
    /// Create a new proof bundle
    pub fn new(eligibility_proof: VoteEligibilityProof) -> Self {
        Self { eligibility_proof }
    }

    /// Verify the eligibility proof
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        self.eligibility_proof.verify()
    }

    /// Check if voter is eligible
    pub fn is_eligible(&self) -> bool {
        self.eligibility_proof.is_eligible()
    }

    /// Get proposal type
    pub fn proposal_type(&self) -> Option<ProofProposalType> {
        self.eligibility_proof.proposal_type()
    }

    /// Get requirements met count
    pub fn requirements_met(&self) -> u8 {
        self.eligibility_proof.requirements_met()
    }

    /// Get total active requirements
    pub fn active_requirements(&self) -> u8 {
        self.eligibility_proof.active_requirements()
    }

    /// Get proof size in bytes
    pub fn size(&self) -> usize {
        self.eligibility_proof.size()
    }
}

/// Trait for types that can generate eligibility proofs
pub trait EligibilityProofGenerator {
    /// Get the voter profile for proof generation
    fn get_voter_profile(&self) -> ProofVoterProfile;

    /// Generate an eligibility proof for a proposal type
    fn generate_eligibility_proof(
        &self,
        proposal_type: ProofProposalType,
        config: ProofConfig,
    ) -> ProofResult<EligibilityProofBundle> {
        let profile = self.get_voter_profile();

        let proof = VoteEligibilityProof::generate(
            &profile,
            proposal_type,
            config,
        )?;

        Ok(EligibilityProofBundle::new(proof))
    }

    /// Compute commitment to this voter
    fn compute_commitment(&self) -> [u8; 32] {
        let profile = self.get_voter_profile();
        compute_voter_commitment(&profile)
    }
}

/// Simple voter representation for proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofableVoter {
    /// DID of the voter
    pub did: String,
    /// Assurance level (0-4)
    pub assurance_level: u8,
    /// MATL score (0.0-1.0)
    pub matl_score: f32,
    /// Stake amount
    pub stake: f32,
    /// Account age in days
    pub account_age_days: u32,
    /// Recent participation rate (0.0-1.0)
    pub participation_rate: f32,
    /// Has humanity proof
    pub has_humanity_proof: bool,
    /// FL contributions count
    pub fl_contributions: u32,
}

impl ProofableVoter {
    /// Create a new proofable voter
    pub fn new(did: String) -> Self {
        Self {
            did,
            assurance_level: 0,
            matl_score: 0.0,
            stake: 0.0,
            account_age_days: 0,
            participation_rate: 0.0,
            has_humanity_proof: false,
            fl_contributions: 0,
        }
    }

    /// Set assurance level
    pub fn with_assurance_level(mut self, level: u8) -> Self {
        self.assurance_level = level.min(4);
        self
    }

    /// Set MATL score
    pub fn with_matl_score(mut self, score: f32) -> Self {
        self.matl_score = score.clamp(0.0, 1.0);
        self
    }

    /// Set stake
    pub fn with_stake(mut self, stake: f32) -> Self {
        self.stake = stake.max(0.0);
        self
    }

    /// Set account age
    pub fn with_account_age(mut self, days: u32) -> Self {
        self.account_age_days = days;
        self
    }

    /// Set participation rate
    pub fn with_participation(mut self, rate: f32) -> Self {
        self.participation_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set humanity proof
    pub fn with_humanity_proof(mut self, has_proof: bool) -> Self {
        self.has_humanity_proof = has_proof;
        self
    }

    /// Set FL contributions
    pub fn with_fl_contributions(mut self, count: u32) -> Self {
        self.fl_contributions = count;
        self
    }
}

impl EligibilityProofGenerator for ProofableVoter {
    fn get_voter_profile(&self) -> ProofVoterProfile {
        ProofVoterProfile {
            did: self.did.clone(),
            assurance_level: self.assurance_level,
            matl_score: self.matl_score,
            stake: self.stake,
            account_age_days: self.account_age_days,
            participation_rate: self.participation_rate,
            has_humanity_proof: self.has_humanity_proof,
            fl_contributions: self.fl_contributions,
        }
    }
}

/// Helper to quickly verify a voter's eligibility
pub fn verify_voter_eligibility(
    profile: &ProofVoterProfile,
    proposal_type: ProofProposalType,
    config: ProofConfig,
) -> ProofResult<bool> {
    let proof = VoteEligibilityProof::generate(profile, proposal_type, config)?;
    let result = proof.verify()?;
    Ok(result.valid && proof.is_eligible())
}

/// Check eligibility for multiple proposal types
pub fn check_all_eligibility(
    profile: &ProofVoterProfile,
    config: ProofConfig,
) -> Vec<(ProofProposalType, bool)> {
    let types = [
        ProofProposalType::Standard,
        ProofProposalType::Constitutional,
        ProofProposalType::ModelGovernance,
        ProofProposalType::Emergency,
        ProofProposalType::Treasury,
        ProofProposalType::Membership,
    ];

    types.iter().map(|&pt| {
        let eligible = verify_voter_eligibility(profile, pt, config.clone()).unwrap_or(false);
        (pt, eligible)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::SecurityLevel;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    fn eligible_voter() -> ProofableVoter {
        ProofableVoter::new("did:mycelix:voter".to_string())
            .with_assurance_level(3)
            .with_matl_score(0.8)
            .with_stake(600.0)
            .with_account_age(100)
            .with_participation(0.5)
            .with_humanity_proof(true)
            .with_fl_contributions(25)
    }

    #[test]
    fn test_proofable_voter_builder() {
        let voter = ProofableVoter::new("did:test".to_string())
            .with_assurance_level(2)
            .with_matl_score(0.7);

        assert_eq!(voter.assurance_level, 2);
        assert_eq!(voter.matl_score, 0.7);
    }

    #[test]
    fn test_eligibility_proof_generation() {
        let voter = eligible_voter();

        let bundle = voter.generate_eligibility_proof(
            ProofProposalType::Standard,
            test_config(),
        ).unwrap();

        assert!(bundle.is_eligible());
        let result = bundle.verify().unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_eligibility_for_different_types() {
        let voter = eligible_voter();

        // Standard - should pass
        let bundle = voter.generate_eligibility_proof(
            ProofProposalType::Standard,
            test_config(),
        ).unwrap();
        assert!(bundle.is_eligible());

        // Constitutional - should pass (has humanity proof)
        let bundle = voter.generate_eligibility_proof(
            ProofProposalType::Constitutional,
            test_config(),
        ).unwrap();
        assert!(bundle.is_eligible());

        // ModelGovernance - should pass (has FL contributions)
        let bundle = voter.generate_eligibility_proof(
            ProofProposalType::ModelGovernance,
            test_config(),
        ).unwrap();
        assert!(bundle.is_eligible());
    }

    #[test]
    fn test_voter_commitment() {
        let voter = eligible_voter();

        let commitment1 = voter.compute_commitment();
        let commitment2 = voter.compute_commitment();

        assert_eq!(commitment1, commitment2);

        // Different voter -> different commitment
        let voter2 = ProofableVoter::new("did:other".to_string());
        let commitment3 = voter2.compute_commitment();
        assert_ne!(commitment1, commitment3);
    }

    #[test]
    fn test_verify_voter_eligibility_helper() {
        let voter = eligible_voter();
        let profile = voter.get_voter_profile();

        let result = verify_voter_eligibility(
            &profile,
            ProofProposalType::Standard,
            test_config(),
        ).unwrap();

        assert!(result);
    }

    #[test]
    fn test_ineligible_voter() {
        let voter = ProofableVoter::new("did:test".to_string())
            .with_assurance_level(0); // Too low

        let bundle = voter.generate_eligibility_proof(
            ProofProposalType::Standard, // Requires E1
            test_config(),
        ).unwrap();

        assert!(!bundle.is_eligible());
    }
}
