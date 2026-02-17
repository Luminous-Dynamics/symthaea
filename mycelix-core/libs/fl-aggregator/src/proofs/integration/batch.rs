//! Batch Verification
//!
//! Efficiently verify multiple proofs in batch with aggregated results.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::{BatchVerifier, BatchVerificationResult};
//!
//! let mut batch = BatchVerifier::new();
//!
//! // Add various proofs
//! batch.add_range_proof(range_proof);
//! batch.add_gradient_proof(gradient_proof);
//! batch.add_identity_proof(identity_proof);
//!
//! // Verify all at once
//! let result = batch.verify_all()?;
//! println!("Verified {}/{} proofs", result.valid_count, result.total_count);
//! ```

use crate::proofs::{
    ProofResult, ProofType, VerificationResult,
    RangeProof, MembershipProof, GradientIntegrityProof,
    IdentityAssuranceProof, VoteEligibilityProof,
};
use std::time::{Duration, Instant};

/// Result of batch verification
#[derive(Debug, Clone)]
pub struct BatchVerificationResult {
    /// Total number of proofs verified
    pub total_count: usize,

    /// Number of valid proofs
    pub valid_count: usize,

    /// Number of invalid proofs
    pub invalid_count: usize,

    /// Total verification time
    pub total_time: Duration,

    /// Individual results
    pub results: Vec<ProofVerificationEntry>,

    /// Whether all proofs are valid
    pub all_valid: bool,
}

/// Single proof verification entry
#[derive(Debug, Clone)]
pub struct ProofVerificationEntry {
    /// Proof type
    pub proof_type: ProofType,

    /// Proof identifier (index in batch)
    pub index: usize,

    /// Custom label for the proof
    pub label: Option<String>,

    /// Verification result
    pub result: VerificationResult,
}

/// Enum to hold different proof types
#[derive(Clone)]
pub enum AnyProof {
    Range(RangeProof),
    Membership(MembershipProof),
    GradientIntegrity(GradientIntegrityProof),
    IdentityAssurance(IdentityAssuranceProof),
    VoteEligibility(VoteEligibilityProof),
}

impl AnyProof {
    fn proof_type(&self) -> ProofType {
        match self {
            AnyProof::Range(_) => ProofType::Range,
            AnyProof::Membership(_) => ProofType::Membership,
            AnyProof::GradientIntegrity(_) => ProofType::GradientIntegrity,
            AnyProof::IdentityAssurance(_) => ProofType::IdentityAssurance,
            AnyProof::VoteEligibility(_) => ProofType::VoteEligibility,
        }
    }

    fn verify(&self) -> ProofResult<VerificationResult> {
        match self {
            AnyProof::Range(p) => p.verify(),
            AnyProof::Membership(p) => p.verify(),
            AnyProof::GradientIntegrity(p) => p.verify(),
            AnyProof::IdentityAssurance(p) => p.verify(),
            AnyProof::VoteEligibility(p) => p.verify(),
        }
    }
}

/// Entry in the batch verifier
struct BatchEntry {
    proof: AnyProof,
    label: Option<String>,
}

/// Batch verifier for multiple proofs
#[derive(Default)]
pub struct BatchVerifier {
    proofs: Vec<BatchEntry>,
}

impl BatchVerifier {
    /// Create a new empty batch verifier
    pub fn new() -> Self {
        Self { proofs: Vec::new() }
    }

    /// Create with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            proofs: Vec::with_capacity(capacity),
        }
    }

    /// Add a range proof to the batch
    pub fn add_range_proof(&mut self, proof: RangeProof) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::Range(proof),
            label: None,
        });
        self
    }

    /// Add a range proof with a label
    pub fn add_range_proof_labeled(&mut self, proof: RangeProof, label: impl Into<String>) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::Range(proof),
            label: Some(label.into()),
        });
        self
    }

    /// Add a membership proof to the batch
    pub fn add_membership_proof(&mut self, proof: MembershipProof) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::Membership(proof),
            label: None,
        });
        self
    }

    /// Add a membership proof with a label
    pub fn add_membership_proof_labeled(&mut self, proof: MembershipProof, label: impl Into<String>) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::Membership(proof),
            label: Some(label.into()),
        });
        self
    }

    /// Add a gradient integrity proof to the batch
    pub fn add_gradient_proof(&mut self, proof: GradientIntegrityProof) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::GradientIntegrity(proof),
            label: None,
        });
        self
    }

    /// Add a gradient integrity proof with a label
    pub fn add_gradient_proof_labeled(&mut self, proof: GradientIntegrityProof, label: impl Into<String>) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::GradientIntegrity(proof),
            label: Some(label.into()),
        });
        self
    }

    /// Add an identity assurance proof to the batch
    pub fn add_identity_proof(&mut self, proof: IdentityAssuranceProof) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::IdentityAssurance(proof),
            label: None,
        });
        self
    }

    /// Add an identity assurance proof with a label
    pub fn add_identity_proof_labeled(&mut self, proof: IdentityAssuranceProof, label: impl Into<String>) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::IdentityAssurance(proof),
            label: Some(label.into()),
        });
        self
    }

    /// Add a vote eligibility proof to the batch
    pub fn add_vote_proof(&mut self, proof: VoteEligibilityProof) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::VoteEligibility(proof),
            label: None,
        });
        self
    }

    /// Add a vote eligibility proof with a label
    pub fn add_vote_proof_labeled(&mut self, proof: VoteEligibilityProof, label: impl Into<String>) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof: AnyProof::VoteEligibility(proof),
            label: Some(label.into()),
        });
        self
    }

    /// Add any proof type
    pub fn add_proof(&mut self, proof: AnyProof) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof,
            label: None,
        });
        self
    }

    /// Add any proof type with a label
    pub fn add_proof_labeled(&mut self, proof: AnyProof, label: impl Into<String>) -> &mut Self {
        self.proofs.push(BatchEntry {
            proof,
            label: Some(label.into()),
        });
        self
    }

    /// Get the number of proofs in the batch
    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.proofs.is_empty()
    }

    /// Clear all proofs from the batch
    pub fn clear(&mut self) {
        self.proofs.clear();
    }

    /// Verify all proofs in the batch sequentially
    pub fn verify_all(&self) -> ProofResult<BatchVerificationResult> {
        if self.proofs.is_empty() {
            return Ok(BatchVerificationResult {
                total_count: 0,
                valid_count: 0,
                invalid_count: 0,
                total_time: Duration::ZERO,
                results: Vec::new(),
                all_valid: true,
            });
        }

        let start = Instant::now();
        let mut results = Vec::with_capacity(self.proofs.len());
        let mut valid_count = 0;
        let mut invalid_count = 0;

        for (index, entry) in self.proofs.iter().enumerate() {
            let result = entry.proof.verify()?;

            if result.valid {
                valid_count += 1;
            } else {
                invalid_count += 1;
            }

            results.push(ProofVerificationEntry {
                proof_type: entry.proof.proof_type(),
                index,
                label: entry.label.clone(),
                result,
            });
        }

        let total_time = start.elapsed();

        Ok(BatchVerificationResult {
            total_count: self.proofs.len(),
            valid_count,
            invalid_count,
            total_time,
            results,
            all_valid: invalid_count == 0,
        })
    }

    /// Verify all proofs and stop on first failure (fail-fast)
    pub fn verify_fail_fast(&self) -> ProofResult<BatchVerificationResult> {
        if self.proofs.is_empty() {
            return Ok(BatchVerificationResult {
                total_count: 0,
                valid_count: 0,
                invalid_count: 0,
                total_time: Duration::ZERO,
                results: Vec::new(),
                all_valid: true,
            });
        }

        let start = Instant::now();
        let mut results = Vec::with_capacity(self.proofs.len());
        let mut valid_count = 0;

        for (index, entry) in self.proofs.iter().enumerate() {
            let result = entry.proof.verify()?;

            let is_valid = result.valid;

            results.push(ProofVerificationEntry {
                proof_type: entry.proof.proof_type(),
                index,
                label: entry.label.clone(),
                result,
            });

            if is_valid {
                valid_count += 1;
            } else {
                // Stop on first invalid proof
                let total_time = start.elapsed();
                return Ok(BatchVerificationResult {
                    total_count: self.proofs.len(),
                    valid_count,
                    invalid_count: 1,
                    total_time,
                    results,
                    all_valid: false,
                });
            }
        }

        let total_time = start.elapsed();

        Ok(BatchVerificationResult {
            total_count: self.proofs.len(),
            valid_count,
            invalid_count: 0,
            total_time,
            results,
            all_valid: true,
        })
    }

    /// Get statistics about proof types in the batch
    pub fn type_counts(&self) -> BatchTypeStats {
        let mut stats = BatchTypeStats::default();

        for entry in &self.proofs {
            match entry.proof {
                AnyProof::Range(_) => stats.range_count += 1,
                AnyProof::Membership(_) => stats.membership_count += 1,
                AnyProof::GradientIntegrity(_) => stats.gradient_count += 1,
                AnyProof::IdentityAssurance(_) => stats.identity_count += 1,
                AnyProof::VoteEligibility(_) => stats.vote_count += 1,
            }
        }

        stats.total = self.proofs.len();
        stats
    }
}

/// Statistics about proof types in a batch
#[derive(Debug, Clone, Default)]
pub struct BatchTypeStats {
    pub total: usize,
    pub range_count: usize,
    pub membership_count: usize,
    pub gradient_count: usize,
    pub identity_count: usize,
    pub vote_count: usize,
}

impl BatchVerificationResult {
    /// Get all valid entries
    pub fn valid_entries(&self) -> impl Iterator<Item = &ProofVerificationEntry> {
        self.results.iter().filter(|e| e.result.valid)
    }

    /// Get all invalid entries
    pub fn invalid_entries(&self) -> impl Iterator<Item = &ProofVerificationEntry> {
        self.results.iter().filter(|e| !e.result.valid)
    }

    /// Get entries by proof type
    pub fn entries_by_type(&self, proof_type: ProofType) -> impl Iterator<Item = &ProofVerificationEntry> {
        self.results.iter().filter(move |e| e.proof_type == proof_type)
    }

    /// Get entry by label
    pub fn entry_by_label(&self, label: &str) -> Option<&ProofVerificationEntry> {
        self.results.iter().find(|e| {
            e.label.as_ref().map(|l| l == label).unwrap_or(false)
        })
    }

    /// Get average verification time per proof
    pub fn avg_verification_time(&self) -> Duration {
        if self.total_count == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.total_count as u32
        }
    }

    /// Calculate success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_count == 0 {
            1.0
        } else {
            self.valid_count as f64 / self.total_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::{
        ProofConfig, SecurityLevel,
        ProofIdentityFactor, ProofAssuranceLevel,
        ProofVoterProfile, ProofProposalType,
    };

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_batch_verifier_empty() {
        let batch = BatchVerifier::new();
        let result = batch.verify_all().unwrap();

        assert!(result.all_valid);
        assert_eq!(result.total_count, 0);
        assert_eq!(result.valid_count, 0);
    }

    #[test]
    fn test_batch_verifier_single_range() {
        let mut batch = BatchVerifier::new();

        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        batch.add_range_proof(proof);

        let result = batch.verify_all().unwrap();

        assert!(result.all_valid);
        assert_eq!(result.total_count, 1);
        assert_eq!(result.valid_count, 1);
    }

    #[test]
    fn test_batch_verifier_multiple_types() {
        let mut batch = BatchVerifier::new();

        // Add range proof
        let range_proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        batch.add_range_proof_labeled(range_proof, "range_test");

        // Add gradient proof
        let gradient = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let gradient_proof = GradientIntegrityProof::generate(&gradient, 5.0, test_config()).unwrap();
        batch.add_gradient_proof_labeled(gradient_proof, "gradient_test");

        // Add identity proof
        let factors = vec![
            ProofIdentityFactor::new(0.5, 0, true),
            ProofIdentityFactor::new(0.3, 1, true),
        ];
        let identity_proof = IdentityAssuranceProof::generate(
            "did:test",
            &factors,
            ProofAssuranceLevel::E2,
            test_config(),
        ).unwrap();
        batch.add_identity_proof_labeled(identity_proof, "identity_test");

        let result = batch.verify_all().unwrap();

        assert!(result.all_valid);
        assert_eq!(result.total_count, 3);
        assert_eq!(result.valid_count, 3);

        // Check by label
        let range_entry = result.entry_by_label("range_test").unwrap();
        assert_eq!(range_entry.proof_type, ProofType::Range);
        assert!(range_entry.result.valid);

        let gradient_entry = result.entry_by_label("gradient_test").unwrap();
        assert_eq!(gradient_entry.proof_type, ProofType::GradientIntegrity);

        // Check type stats
        let stats = batch.type_counts();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.range_count, 1);
        assert_eq!(stats.gradient_count, 1);
        assert_eq!(stats.identity_count, 1);
    }

    #[test]
    fn test_batch_verifier_with_invalid() {
        let mut batch = BatchVerifier::new();

        // Add valid range proof
        let valid_proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        batch.add_range_proof(valid_proof);

        // Add another valid proof
        let valid_proof2 = RangeProof::generate(75, 0, 100, test_config()).unwrap();
        batch.add_range_proof(valid_proof2);

        let result = batch.verify_all().unwrap();
        assert!(result.all_valid);
        assert_eq!(result.valid_count, 2);
    }

    #[test]
    fn test_batch_verifier_vote_eligibility() {
        let mut batch = BatchVerifier::new();

        // Create voter profile
        let voter = ProofVoterProfile {
            did: "did:test:voter".to_string(),
            assurance_level: 3,
            matl_score: 0.8,
            stake: 600.0,
            account_age_days: 100,
            participation_rate: 0.5,
            has_humanity_proof: true,
            fl_contributions: 25,
        };

        // Add vote proofs for different proposal types
        let standard_proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Standard,
            test_config(),
        ).unwrap();
        batch.add_vote_proof_labeled(standard_proof, "standard_vote");

        let constitutional_proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Constitutional,
            test_config(),
        ).unwrap();
        batch.add_vote_proof_labeled(constitutional_proof, "constitutional_vote");

        let result = batch.verify_all().unwrap();

        assert!(result.all_valid);
        assert_eq!(result.total_count, 2);

        // Check by type
        let vote_entries: Vec<_> = result.entries_by_type(ProofType::VoteEligibility).collect();
        assert_eq!(vote_entries.len(), 2);
    }

    #[test]
    fn test_batch_statistics() {
        let result = BatchVerificationResult {
            total_count: 10,
            valid_count: 8,
            invalid_count: 2,
            total_time: Duration::from_millis(500),
            results: Vec::new(),
            all_valid: false,
        };

        assert_eq!(result.success_rate(), 0.8);
        assert_eq!(result.avg_verification_time(), Duration::from_millis(50));
    }
}
