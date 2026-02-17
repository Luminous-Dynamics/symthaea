//! Identity Proof Integration
//!
//! Integrates identity assurance proofs with the identity system.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::IdentityProofGenerator;
//!
//! // Generate a proof that identity meets E2 assurance
//! let bundle = identity.generate_assurance_proof(ProofAssuranceLevel::E2, config)?;
//!
//! // Verify the proof
//! let result = bundle.verify()?;
//! assert!(result.valid);
//! ```

use crate::proofs::{
    IdentityAssuranceProof, ProofAssuranceLevel, ProofIdentityFactor,
    ProofConfig, ProofResult, VerificationResult, compute_identity_commitment,
};
use serde::{Deserialize, Serialize};

/// Bundle of proofs for identity verification
#[derive(Clone)]
pub struct IdentityProofBundle {
    /// Proof that identity meets assurance level
    pub assurance_proof: IdentityAssuranceProof,
}

impl IdentityProofBundle {
    /// Create a new proof bundle
    pub fn new(assurance_proof: IdentityAssuranceProof) -> Self {
        Self { assurance_proof }
    }

    /// Verify the identity proof
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        self.assurance_proof.verify()
    }

    /// Check if identity meets the required level
    pub fn meets_threshold(&self) -> bool {
        self.assurance_proof.meets_threshold()
    }

    /// Get the final score
    pub fn final_score(&self) -> u64 {
        self.assurance_proof.final_score()
    }

    /// Get proof size in bytes
    pub fn size(&self) -> usize {
        self.assurance_proof.size()
    }
}

/// Trait for types that can generate identity assurance proofs
pub trait IdentityProofGenerator {
    /// Get the DID for this identity
    fn did(&self) -> &str;

    /// Get identity factors for proof generation
    fn get_proof_factors(&self) -> Vec<ProofIdentityFactor>;

    /// Generate an assurance level proof
    fn generate_assurance_proof(
        &self,
        min_level: ProofAssuranceLevel,
        config: ProofConfig,
    ) -> ProofResult<IdentityProofBundle> {
        let factors = self.get_proof_factors();

        let proof = IdentityAssuranceProof::generate(
            self.did(),
            &factors,
            min_level,
            config,
        )?;

        Ok(IdentityProofBundle::new(proof))
    }

    /// Compute commitment to this identity
    fn compute_commitment(&self) -> [u8; 32] {
        let factors = self.get_proof_factors();
        compute_identity_commitment(self.did(), &factors)
    }
}

/// Simple identity representation for proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofableIdentity {
    /// Decentralized identifier
    pub did: String,

    /// Identity factors
    pub factors: Vec<SerializableIdentityFactor>,
}

/// Serializable version of identity factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableIdentityFactor {
    /// Factor contribution to score (0.0-1.0)
    pub contribution: f32,
    /// Factor category (0-4)
    pub category: u8,
    /// Whether the factor is active
    pub is_active: bool,
}

impl From<&SerializableIdentityFactor> for ProofIdentityFactor {
    fn from(factor: &SerializableIdentityFactor) -> Self {
        ProofIdentityFactor::new(factor.contribution, factor.category, factor.is_active)
    }
}

impl ProofableIdentity {
    /// Create a new proofable identity
    pub fn new(did: String) -> Self {
        Self {
            did,
            factors: Vec::new(),
        }
    }

    /// Add a factor
    pub fn add_factor(&mut self, contribution: f32, category: u8, is_active: bool) {
        self.factors.push(SerializableIdentityFactor {
            contribution,
            category,
            is_active,
        });
    }

    /// Create with factors
    pub fn with_factors(did: String, factors: Vec<SerializableIdentityFactor>) -> Self {
        Self { did, factors }
    }
}

impl IdentityProofGenerator for ProofableIdentity {
    fn did(&self) -> &str {
        &self.did
    }

    fn get_proof_factors(&self) -> Vec<ProofIdentityFactor> {
        self.factors.iter().map(|f| f.into()).collect()
    }
}

/// Helper to quickly verify an identity meets a level
pub fn verify_identity_level(
    did: &str,
    factors: &[ProofIdentityFactor],
    min_level: ProofAssuranceLevel,
    config: ProofConfig,
) -> ProofResult<bool> {
    let proof = IdentityAssuranceProof::generate(did, factors, min_level, config)?;
    let result = proof.verify()?;
    Ok(result.valid && proof.meets_threshold())
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

    #[test]
    fn test_proofable_identity() {
        let mut identity = ProofableIdentity::new("did:mycelix:test".to_string());
        identity.add_factor(0.5, 0, true);
        identity.add_factor(0.3, 1, true);

        let factors = identity.get_proof_factors();
        assert_eq!(factors.len(), 2);
    }

    #[test]
    fn test_identity_proof_generation() {
        let mut identity = ProofableIdentity::new("did:mycelix:test".to_string());
        identity.add_factor(0.5, 0, true);
        identity.add_factor(0.3, 1, true);

        let bundle = identity.generate_assurance_proof(
            ProofAssuranceLevel::E2,
            test_config(),
        ).unwrap();

        assert!(bundle.meets_threshold());
        let result = bundle.verify().unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_identity_commitment() {
        let mut identity = ProofableIdentity::new("did:mycelix:test".to_string());
        identity.add_factor(0.5, 0, true);

        let commitment1 = identity.compute_commitment();
        let commitment2 = identity.compute_commitment();

        assert_eq!(commitment1, commitment2);

        // Different identity -> different commitment
        let identity2 = ProofableIdentity::new("did:mycelix:other".to_string());
        let commitment3 = identity2.compute_commitment();
        assert_ne!(commitment1, commitment3);
    }

    #[test]
    fn test_verify_identity_level() {
        let factors = vec![
            ProofIdentityFactor::new(0.5, 0, true),
            ProofIdentityFactor::new(0.3, 1, true),
        ];

        let result = verify_identity_level(
            "did:mycelix:test",
            &factors,
            ProofAssuranceLevel::E2,
            test_config(),
        ).unwrap();

        assert!(result);

        // Should fail for higher level (lower score)
        let low_factors = vec![
            ProofIdentityFactor::new(0.3, 0, true),
        ];

        let result = verify_identity_level(
            "did:mycelix:test",
            &low_factors,
            ProofAssuranceLevel::E4, // Requires 900
            test_config(),
        ).unwrap();

        // Score is 300 + 50 (diversity) = 350, not enough for E4
        assert!(!result);
    }
}
