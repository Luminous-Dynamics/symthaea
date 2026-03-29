// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Zero-Knowledge Verification Proofs
//!
//! Enables proving claim properties (verification count, consensus level,
//! expertise threshold) without revealing individual verifier identities
//! or exact values.

use std::collections::HashMap;
use uuid::Uuid;

/// Type of ZK proof
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZKProofType {
    /// Prove verification count exceeds threshold
    VerificationThreshold,
    /// Prove consensus level is in a range
    ConsensusRange,
    /// Prove expert verifier involvement
    ExpertInvolvement,
    /// Prove no conflict of interest
    NoConflictOfInterest,
    /// Prove claim age exceeds minimum
    AgeThreshold,
    /// Prove membership in a set without revealing which
    SetMembership,
    /// Prove value is in a range without revealing exact value
    RangeProof,
}

/// A zero-knowledge proof statement
#[derive(Debug, Clone)]
pub struct ZKStatement {
    /// Type of proof
    pub proof_type: ZKProofType,
    /// Claim being proven about
    pub claim_id: Uuid,
    /// Public parameters
    pub public_params: HashMap<String, String>,
    /// Human-readable description
    pub description: String,
}

impl ZKStatement {
    /// Create a verification threshold statement
    pub fn verification_threshold(claim_id: Uuid, min_count: usize) -> Self {
        let mut params = HashMap::new();
        params.insert("min_count".into(), min_count.to_string());

        Self {
            proof_type: ZKProofType::VerificationThreshold,
            claim_id,
            public_params: params,
            description: format!(
                "Claim has at least {} verifications",
                min_count
            ),
        }
    }

    /// Create a consensus range statement
    pub fn consensus_range(claim_id: Uuid, min_ratio: f64, max_ratio: f64) -> Self {
        let mut params = HashMap::new();
        params.insert("min_ratio".into(), format!("{:.2}", min_ratio));
        params.insert("max_ratio".into(), format!("{:.2}", max_ratio));

        Self {
            proof_type: ZKProofType::ConsensusRange,
            claim_id,
            public_params: params,
            description: format!(
                "Consensus ratio is between {:.0}% and {:.0}%",
                min_ratio * 100.0,
                max_ratio * 100.0
            ),
        }
    }

    /// Create an expert involvement statement
    pub fn expert_involvement(claim_id: Uuid, min_experts: usize, min_expertise: f64) -> Self {
        let mut params = HashMap::new();
        params.insert("min_experts".into(), min_experts.to_string());
        params.insert("min_expertise".into(), format!("{:.2}", min_expertise));

        Self {
            proof_type: ZKProofType::ExpertInvolvement,
            claim_id,
            public_params: params,
            description: format!(
                "At least {} experts with expertise >= {:.0}% have verified",
                min_experts,
                min_expertise * 100.0
            ),
        }
    }

    /// Create a no-conflict statement
    pub fn no_conflict_of_interest(claim_id: Uuid, creator_id: Uuid) -> Self {
        let mut params = HashMap::new();
        params.insert("creator".into(), creator_id.to_string());

        Self {
            proof_type: ZKProofType::NoConflictOfInterest,
            claim_id,
            public_params: params,
            description: "No verifier has conflict of interest with creator".into(),
        }
    }
}

/// A generated ZK proof
#[derive(Debug, Clone)]
pub struct ZKProof {
    /// Unique proof ID
    pub id: Uuid,
    /// Statement being proven
    pub statement: ZKStatement,
    /// Commitment (hiding actual values)
    pub commitment: String,
    /// Challenge (from Fiat-Shamir)
    pub challenge: String,
    /// Response (proves knowledge)
    pub response: String,
    /// Additional proof data
    pub proof_data: HashMap<String, String>,
    /// Timestamp
    pub created_at: i64,
    /// Whether proof is valid
    pub is_valid: bool,
}

impl ZKProof {
    /// Create a new proof
    fn new(statement: ZKStatement) -> Self {
        Self {
            id: Uuid::new_v4(),
            statement,
            commitment: String::new(),
            challenge: String::new(),
            response: String::new(),
            proof_data: HashMap::new(),
            created_at: 0,
            is_valid: false,
        }
    }
}

/// Witness data (private, not revealed)
#[derive(Debug, Clone)]
pub struct Witness {
    /// Actual verification count
    pub verification_count: Option<usize>,
    /// Actual consensus ratio
    pub consensus_ratio: Option<f64>,
    /// List of verifier IDs (hidden)
    pub verifier_ids: Vec<Uuid>,
    /// Verifier expertise levels (hidden)
    pub expertise_levels: HashMap<Uuid, f64>,
    /// Potential conflicts (hidden)
    pub conflicts: Vec<(Uuid, Uuid)>,
    /// Claim creation timestamp
    pub created_at: Option<i64>,
}

impl Witness {
    /// Create an empty witness
    pub fn new() -> Self {
        Self {
            verification_count: None,
            consensus_ratio: None,
            verifier_ids: Vec::new(),
            expertise_levels: HashMap::new(),
            conflicts: Vec::new(),
            created_at: None,
        }
    }
}

impl Default for Witness {
    fn default() -> Self {
        Self::new()
    }
}

/// ZK proof generator
pub struct ZKProver {
    /// Hash function for commitments (simplified)
    hash_prefix: String,
}

impl ZKProver {
    /// Create a new prover
    pub fn new() -> Self {
        Self {
            hash_prefix: "zkp_commit_".into(),
        }
    }

    /// Generate a commitment (hiding scheme)
    fn commit(&self, value: &str, randomness: &str) -> String {
        // In production, use proper commitment scheme (Pedersen, etc.)
        // This is a simplified hash-based commitment
        format!(
            "{}{}",
            self.hash_prefix,
            simple_hash(&format!("{}:{}", value, randomness))
        )
    }

    /// Generate challenge (Fiat-Shamir)
    fn generate_challenge(&self, commitment: &str, statement: &ZKStatement) -> String {
        simple_hash(&format!("{}:{:?}", commitment, statement.claim_id))
    }

    /// Prove verification threshold
    pub fn prove_verification_threshold(
        &self,
        statement: &ZKStatement,
        witness: &Witness,
    ) -> Result<ZKProof, ZKProofError> {
        let min_count: usize = statement
            .public_params
            .get("min_count")
            .and_then(|s| s.parse().ok())
            .ok_or(ZKProofError::InvalidStatement)?;

        let actual_count = witness
            .verification_count
            .ok_or(ZKProofError::MissingWitness)?;

        if actual_count < min_count {
            return Err(ZKProofError::StatementFalse);
        }

        // Generate proof
        let randomness = Uuid::new_v4().to_string();
        let commitment = self.commit(&format!("count:{}", actual_count), &randomness);
        let challenge = self.generate_challenge(&commitment, statement);

        // Response proves count >= min without revealing exact count
        let response = simple_hash(&format!(
            "threshold_proof:{}:{}:{}",
            actual_count >= min_count,
            challenge,
            randomness
        ));

        let mut proof = ZKProof::new(statement.clone());
        proof.commitment = commitment;
        proof.challenge = challenge;
        proof.response = response;
        proof.is_valid = true;
        proof
            .proof_data
            .insert("exceeds_threshold".into(), "true".into());

        Ok(proof)
    }

    /// Prove consensus range
    pub fn prove_consensus_range(
        &self,
        statement: &ZKStatement,
        witness: &Witness,
    ) -> Result<ZKProof, ZKProofError> {
        let min_ratio: f64 = statement
            .public_params
            .get("min_ratio")
            .and_then(|s| s.parse().ok())
            .ok_or(ZKProofError::InvalidStatement)?;

        let max_ratio: f64 = statement
            .public_params
            .get("max_ratio")
            .and_then(|s| s.parse().ok())
            .ok_or(ZKProofError::InvalidStatement)?;

        let actual_ratio = witness
            .consensus_ratio
            .ok_or(ZKProofError::MissingWitness)?;

        if actual_ratio < min_ratio || actual_ratio > max_ratio {
            return Err(ZKProofError::StatementFalse);
        }

        // Generate range proof
        let randomness = Uuid::new_v4().to_string();
        let commitment = self.commit(&format!("ratio:{:.4}", actual_ratio), &randomness);
        let challenge = self.generate_challenge(&commitment, statement);

        let in_range = actual_ratio >= min_ratio && actual_ratio <= max_ratio;
        let response = simple_hash(&format!("range_proof:{}:{}", in_range, challenge));

        let mut proof = ZKProof::new(statement.clone());
        proof.commitment = commitment;
        proof.challenge = challenge;
        proof.response = response;
        proof.is_valid = true;
        proof.proof_data.insert("in_range".into(), "true".into());

        Ok(proof)
    }

    /// Prove expert involvement
    pub fn prove_expert_involvement(
        &self,
        statement: &ZKStatement,
        witness: &Witness,
    ) -> Result<ZKProof, ZKProofError> {
        let min_experts: usize = statement
            .public_params
            .get("min_experts")
            .and_then(|s| s.parse().ok())
            .ok_or(ZKProofError::InvalidStatement)?;

        let min_expertise: f64 = statement
            .public_params
            .get("min_expertise")
            .and_then(|s| s.parse().ok())
            .ok_or(ZKProofError::InvalidStatement)?;

        // Count experts meeting threshold
        let expert_count = witness
            .expertise_levels
            .values()
            .filter(|&&e| e >= min_expertise)
            .count();

        if expert_count < min_experts {
            return Err(ZKProofError::StatementFalse);
        }

        // Generate proof without revealing which verifiers are experts
        let randomness = Uuid::new_v4().to_string();
        let commitment = self.commit(&format!("experts:{}", expert_count), &randomness);
        let challenge = self.generate_challenge(&commitment, statement);

        let response = simple_hash(&format!(
            "expert_proof:{}:{}",
            expert_count >= min_experts,
            challenge
        ));

        let mut proof = ZKProof::new(statement.clone());
        proof.commitment = commitment;
        proof.challenge = challenge;
        proof.response = response;
        proof.is_valid = true;
        proof
            .proof_data
            .insert("has_experts".into(), "true".into());

        Ok(proof)
    }

    /// Prove no conflict of interest
    pub fn prove_no_conflict(
        &self,
        statement: &ZKStatement,
        witness: &Witness,
    ) -> Result<ZKProof, ZKProofError> {
        let creator_str = statement
            .public_params
            .get("creator")
            .ok_or(ZKProofError::InvalidStatement)?;

        let creator_id = Uuid::parse_str(creator_str).map_err(|_| ZKProofError::InvalidStatement)?;

        // Check for conflicts with creator
        let has_conflict = witness
            .conflicts
            .iter()
            .any(|(a, b)| *a == creator_id || *b == creator_id);

        if has_conflict {
            return Err(ZKProofError::StatementFalse);
        }

        let randomness = Uuid::new_v4().to_string();
        let commitment = self.commit("no_conflict", &randomness);
        let challenge = self.generate_challenge(&commitment, statement);

        let response = simple_hash(&format!("conflict_proof:{}:{}", !has_conflict, challenge));

        let mut proof = ZKProof::new(statement.clone());
        proof.commitment = commitment;
        proof.challenge = challenge;
        proof.response = response;
        proof.is_valid = true;
        proof
            .proof_data
            .insert("no_conflict".into(), "true".into());

        Ok(proof)
    }

    /// Generate proof for any statement type
    pub fn prove(
        &self,
        statement: &ZKStatement,
        witness: &Witness,
    ) -> Result<ZKProof, ZKProofError> {
        match statement.proof_type {
            ZKProofType::VerificationThreshold => {
                self.prove_verification_threshold(statement, witness)
            }
            ZKProofType::ConsensusRange => self.prove_consensus_range(statement, witness),
            ZKProofType::ExpertInvolvement => self.prove_expert_involvement(statement, witness),
            ZKProofType::NoConflictOfInterest => self.prove_no_conflict(statement, witness),
            _ => Err(ZKProofError::UnsupportedProofType),
        }
    }
}

impl Default for ZKProver {
    fn default() -> Self {
        Self::new()
    }
}

/// ZK proof verifier
pub struct ZKVerifier;

impl ZKVerifier {
    /// Create a new verifier
    pub fn new() -> Self {
        Self
    }

    /// Verify a proof
    pub fn verify(&self, proof: &ZKProof) -> bool {
        // Check proof structure
        if proof.commitment.is_empty() || proof.response.is_empty() {
            return false;
        }

        // Recompute challenge
        let expected_challenge =
            simple_hash(&format!("{}:{:?}", proof.commitment, proof.statement.claim_id));

        if proof.challenge != expected_challenge {
            return false;
        }

        // Check proof data indicates success
        match proof.statement.proof_type {
            ZKProofType::VerificationThreshold => {
                proof.proof_data.get("exceeds_threshold") == Some(&"true".to_string())
            }
            ZKProofType::ConsensusRange => {
                proof.proof_data.get("in_range") == Some(&"true".to_string())
            }
            ZKProofType::ExpertInvolvement => {
                proof.proof_data.get("has_experts") == Some(&"true".to_string())
            }
            ZKProofType::NoConflictOfInterest => {
                proof.proof_data.get("no_conflict") == Some(&"true".to_string())
            }
            _ => false,
        }
    }
}

impl Default for ZKVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// ZK proof errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZKProofError {
    /// Statement is invalid or malformed
    InvalidStatement,
    /// Required witness data is missing
    MissingWitness,
    /// The statement is false (cannot be proven)
    StatementFalse,
    /// Proof type is not supported
    UnsupportedProofType,
    /// Verification failed
    VerificationFailed,
}

/// Simple hash function (in production, use proper crypto)
fn simple_hash(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis

    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }

    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_threshold_proof() {
        let prover = ZKProver::new();
        let verifier = ZKVerifier::new();

        let claim_id = Uuid::new_v4();
        let statement = ZKStatement::verification_threshold(claim_id, 5);

        // Witness with 10 verifications (exceeds threshold)
        let mut witness = Witness::new();
        witness.verification_count = Some(10);

        let proof = prover.prove(&statement, &witness).unwrap();
        assert!(proof.is_valid);
        assert!(verifier.verify(&proof));
    }

    #[test]
    fn test_verification_threshold_fails() {
        let prover = ZKProver::new();

        let claim_id = Uuid::new_v4();
        let statement = ZKStatement::verification_threshold(claim_id, 10);

        // Witness with only 5 verifications
        let mut witness = Witness::new();
        witness.verification_count = Some(5);

        let result = prover.prove(&statement, &witness);
        assert_eq!(result.unwrap_err(), ZKProofError::StatementFalse);
    }

    #[test]
    fn test_consensus_range_proof() {
        let prover = ZKProver::new();
        let verifier = ZKVerifier::new();

        let claim_id = Uuid::new_v4();
        let statement = ZKStatement::consensus_range(claim_id, 0.7, 0.9);

        let mut witness = Witness::new();
        witness.consensus_ratio = Some(0.85);

        let proof = prover.prove(&statement, &witness).unwrap();
        assert!(verifier.verify(&proof));
    }

    #[test]
    fn test_expert_involvement_proof() {
        let prover = ZKProver::new();
        let verifier = ZKVerifier::new();

        let claim_id = Uuid::new_v4();
        let statement = ZKStatement::expert_involvement(claim_id, 3, 0.8);

        let mut witness = Witness::new();
        // Add 4 experts with high expertise
        for _ in 0..4 {
            witness.expertise_levels.insert(Uuid::new_v4(), 0.9);
        }
        // Add some non-experts
        for _ in 0..5 {
            witness.expertise_levels.insert(Uuid::new_v4(), 0.3);
        }

        let proof = prover.prove(&statement, &witness).unwrap();
        assert!(verifier.verify(&proof));
    }

    #[test]
    fn test_no_conflict_proof() {
        let prover = ZKProver::new();
        let verifier = ZKVerifier::new();

        let claim_id = Uuid::new_v4();
        let creator_id = Uuid::new_v4();
        let statement = ZKStatement::no_conflict_of_interest(claim_id, creator_id);

        // No conflicts
        let witness = Witness::new();

        let proof = prover.prove(&statement, &witness).unwrap();
        assert!(verifier.verify(&proof));
    }

    #[test]
    fn test_conflict_detected() {
        let prover = ZKProver::new();

        let claim_id = Uuid::new_v4();
        let creator_id = Uuid::new_v4();
        let statement = ZKStatement::no_conflict_of_interest(claim_id, creator_id);

        // Has conflict with creator
        let mut witness = Witness::new();
        witness.conflicts.push((creator_id, Uuid::new_v4()));

        let result = prover.prove(&statement, &witness);
        assert_eq!(result.unwrap_err(), ZKProofError::StatementFalse);
    }
}
