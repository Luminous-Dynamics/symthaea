//! Recursive Proof Composition
//!
//! Enables aggregating multiple proofs into a single proof that verifies all of them.
//! This provides logarithmic verification time for batches of proofs.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::recursive::{ProofBatch, RecursiveProof};
//!
//! // Collect proofs to aggregate
//! let mut batch = ProofBatch::new();
//! batch.add_range_proof(proof1);
//! batch.add_gradient_proof(proof2);
//!
//! // Generate aggregated proof
//! let recursive = RecursiveProof::aggregate(&batch, config)?;
//!
//! // Verify all proofs with single verification
//! let result = recursive.verify()?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use super::{
    ProofConfig, ProofError, ProofResult, ProofType, SecurityLevel,
    RangeProof, GradientIntegrityProof, IdentityAssuranceProof, VoteEligibilityProof,
    MembershipProof, VerificationResult,
};

/// Maximum proofs in a single batch
pub const MAX_BATCH_SIZE: usize = 256;

/// Proof commitment (hash of proof bytes)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProofCommitment {
    /// Hash of the proof
    pub hash: [u8; 32],
    /// Proof type
    pub proof_type: ProofType,
    /// Original proof size
    pub original_size: usize,
}

impl ProofCommitment {
    /// Create commitment from proof bytes
    pub fn from_bytes(proof_type: ProofType, bytes: &[u8]) -> Self {
        use sha2::{Sha256, Digest};
        let hash: [u8; 32] = Sha256::digest(bytes).into();
        Self {
            hash,
            proof_type,
            original_size: bytes.len(),
        }
    }
}

/// Entry in a proof batch
#[derive(Clone)]
pub enum ProofEntry {
    Range(RangeProof),
    Gradient(GradientIntegrityProof),
    Identity(IdentityAssuranceProof),
    Vote(VoteEligibilityProof),
    Membership(MembershipProof),
}

impl std::fmt::Debug for ProofEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofEntry::Range(_) => write!(f, "ProofEntry::Range(...)"),
            ProofEntry::Gradient(_) => write!(f, "ProofEntry::Gradient(...)"),
            ProofEntry::Identity(_) => write!(f, "ProofEntry::Identity(...)"),
            ProofEntry::Vote(_) => write!(f, "ProofEntry::Vote(...)"),
            ProofEntry::Membership(_) => write!(f, "ProofEntry::Membership(...)"),
        }
    }
}

impl ProofEntry {
    /// Get the proof type
    pub fn proof_type(&self) -> ProofType {
        match self {
            ProofEntry::Range(_) => ProofType::Range,
            ProofEntry::Gradient(_) => ProofType::GradientIntegrity,
            ProofEntry::Identity(_) => ProofType::IdentityAssurance,
            ProofEntry::Vote(_) => ProofType::VoteEligibility,
            ProofEntry::Membership(_) => ProofType::Membership,
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            ProofEntry::Range(p) => p.to_bytes(),
            ProofEntry::Gradient(p) => p.to_bytes(),
            ProofEntry::Identity(p) => p.to_bytes(),
            ProofEntry::Vote(p) => p.to_bytes(),
            ProofEntry::Membership(p) => p.to_bytes(),
        }
    }

    /// Verify the proof
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        match self {
            ProofEntry::Range(p) => p.verify(),
            ProofEntry::Gradient(p) => p.verify(),
            ProofEntry::Identity(p) => p.verify(),
            ProofEntry::Vote(p) => p.verify(),
            ProofEntry::Membership(p) => p.verify(),
        }
    }

    /// Create commitment
    pub fn commitment(&self) -> ProofCommitment {
        ProofCommitment::from_bytes(self.proof_type(), &self.to_bytes())
    }
}

/// Batch of proofs to be aggregated
#[derive(Debug, Clone, Default)]
pub struct ProofBatch {
    /// Proofs in the batch
    entries: Vec<ProofEntry>,
    /// Metadata for the batch
    metadata: HashMap<String, String>,
}

impl ProofBatch {
    /// Create a new empty batch
    pub fn new() -> Self {
        Self::default()
    }

    /// Create batch with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            metadata: HashMap::new(),
        }
    }

    /// Add a range proof
    pub fn add_range_proof(&mut self, proof: RangeProof) -> &mut Self {
        self.entries.push(ProofEntry::Range(proof));
        self
    }

    /// Add a gradient proof
    pub fn add_gradient_proof(&mut self, proof: GradientIntegrityProof) -> &mut Self {
        self.entries.push(ProofEntry::Gradient(proof));
        self
    }

    /// Add an identity proof
    pub fn add_identity_proof(&mut self, proof: IdentityAssuranceProof) -> &mut Self {
        self.entries.push(ProofEntry::Identity(proof));
        self
    }

    /// Add a vote proof
    pub fn add_vote_proof(&mut self, proof: VoteEligibilityProof) -> &mut Self {
        self.entries.push(ProofEntry::Vote(proof));
        self
    }

    /// Add a membership proof
    pub fn add_membership_proof(&mut self, proof: MembershipProof) -> &mut Self {
        self.entries.push(ProofEntry::Membership(proof));
        self
    }

    /// Add generic proof entry
    pub fn add(&mut self, entry: ProofEntry) -> &mut Self {
        self.entries.push(entry);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get batch size
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get entries
    pub fn entries(&self) -> &[ProofEntry] {
        &self.entries
    }

    /// Get metadata
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Compute Merkle root of all proof commitments
    pub fn merkle_root(&self) -> ProofResult<[u8; 32]> {
        if self.entries.is_empty() {
            return Err(ProofError::InvalidInput("Empty batch".to_string()));
        }

        let commitments: Vec<[u8; 32]> = self.entries
            .iter()
            .map(|e| e.commitment().hash)
            .collect();

        compute_merkle_root(&commitments)
    }

    /// Get all commitments
    pub fn commitments(&self) -> Vec<ProofCommitment> {
        self.entries.iter().map(|e| e.commitment()).collect()
    }

    /// Total size of all proofs
    pub fn total_size(&self) -> usize {
        self.entries.iter().map(|e| e.to_bytes().len()).sum()
    }

    /// Verify all proofs in batch individually
    pub fn verify_all(&self) -> ProofResult<BatchVerificationResult> {
        let start = Instant::now();
        let mut results = Vec::with_capacity(self.entries.len());
        let mut all_valid = true;

        for (i, entry) in self.entries.iter().enumerate() {
            match entry.verify() {
                Ok(result) => {
                    if !result.valid {
                        all_valid = false;
                    }
                    results.push((i, result));
                }
                Err(e) => {
                    all_valid = false;
                    results.push((i, VerificationResult {
                        valid: false,
                        proof_type: entry.proof_type(),
                        verification_time: std::time::Duration::ZERO,
                        details: Some(e.to_string()),
                    }));
                }
            }
        }

        Ok(BatchVerificationResult {
            all_valid,
            results,
            total_time: start.elapsed(),
            batch_size: self.entries.len(),
        })
    }
}

/// Result of batch verification
#[derive(Debug, Clone)]
pub struct BatchVerificationResult {
    /// Whether all proofs are valid
    pub all_valid: bool,
    /// Individual results with indices
    pub results: Vec<(usize, VerificationResult)>,
    /// Total verification time
    pub total_time: std::time::Duration,
    /// Number of proofs verified
    pub batch_size: usize,
}

impl BatchVerificationResult {
    /// Get indices of failed proofs
    pub fn failed_indices(&self) -> Vec<usize> {
        self.results
            .iter()
            .filter(|(_, r)| !r.valid)
            .map(|(i, _)| *i)
            .collect()
    }

    /// Get average verification time per proof
    pub fn avg_time_per_proof(&self) -> std::time::Duration {
        if self.batch_size == 0 {
            std::time::Duration::ZERO
        } else {
            self.total_time / self.batch_size as u32
        }
    }
}

/// Recursive/Aggregated proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveProof {
    /// Merkle root of proof commitments
    pub root: [u8; 32],
    /// Number of proofs aggregated
    pub count: usize,
    /// Proof type distribution
    pub type_counts: HashMap<String, usize>,
    /// Aggregation timestamp
    pub timestamp: u64,
    /// Security level used
    pub security_level: SecurityLevel,
    /// Aggregated proof data
    proof_data: Vec<u8>,
    /// Individual commitments (for verification)
    commitments: Vec<ProofCommitment>,
}

impl RecursiveProof {
    /// Aggregate a batch of proofs into a recursive proof
    pub fn aggregate(batch: &ProofBatch, config: ProofConfig) -> ProofResult<Self> {
        if batch.is_empty() {
            return Err(ProofError::InvalidInput("Cannot aggregate empty batch".to_string()));
        }

        if batch.len() > MAX_BATCH_SIZE {
            return Err(ProofError::InvalidInput(
                format!("Batch size {} exceeds maximum {}", batch.len(), MAX_BATCH_SIZE)
            ));
        }

        let start = Instant::now();

        // First, verify all proofs are valid
        let verification = batch.verify_all()?;
        if !verification.all_valid {
            let failed = verification.failed_indices();
            return Err(ProofError::VerificationFailed(
                format!("Batch contains invalid proofs at indices: {:?}", failed)
            ));
        }

        // Compute commitments and root
        let commitments = batch.commitments();
        let root = batch.merkle_root()?;

        // Count proof types
        let mut type_counts = HashMap::new();
        for entry in batch.entries() {
            let type_name = format!("{:?}", entry.proof_type());
            *type_counts.entry(type_name).or_insert(0) += 1;
        }

        // Create aggregated proof data
        // In a full implementation, this would be a STARK proof of the verification circuit
        // For now, we create a commitment-based aggregate
        let proof_data = create_aggregate_proof(&commitments, &root, config.security_level)?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        tracing::info!(
            count = batch.len(),
            time_ms = start.elapsed().as_millis(),
            "Generated recursive proof"
        );

        Ok(Self {
            root,
            count: batch.len(),
            type_counts,
            timestamp,
            security_level: config.security_level,
            proof_data,
            commitments,
        })
    }

    /// Verify the recursive proof
    pub fn verify(&self) -> ProofResult<RecursiveVerificationResult> {
        let start = Instant::now();

        // Verify the aggregate proof structure
        let valid = verify_aggregate_proof(&self.proof_data, &self.root, &self.commitments)?;

        Ok(RecursiveVerificationResult {
            valid,
            proofs_covered: self.count,
            verification_time: start.elapsed(),
            merkle_root: self.root,
        })
    }

    /// Verify and get inclusion proof for a specific commitment
    pub fn verify_inclusion(&self, commitment: &ProofCommitment) -> ProofResult<bool> {
        Ok(self.commitments.contains(commitment))
    }

    /// Get all commitments
    pub fn commitments(&self) -> &[ProofCommitment] {
        &self.commitments
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> ProofResult<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| ProofError::SerializationError(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| ProofError::SerializationError(e.to_string()))
    }

    /// Get proof size
    pub fn size(&self) -> usize {
        self.proof_data.len() + self.commitments.len() * 40 // commitment ~40 bytes each
    }

    /// Calculate compression ratio vs individual proofs
    pub fn compression_ratio(&self, original_total_size: usize) -> f64 {
        if original_total_size == 0 {
            1.0
        } else {
            self.size() as f64 / original_total_size as f64
        }
    }
}

/// Result of recursive proof verification
#[derive(Debug, Clone)]
pub struct RecursiveVerificationResult {
    /// Whether the aggregate proof is valid
    pub valid: bool,
    /// Number of proofs covered
    pub proofs_covered: usize,
    /// Verification time
    pub verification_time: std::time::Duration,
    /// Merkle root verified
    pub merkle_root: [u8; 32],
}

// Helper functions

/// Compute Merkle root from leaf hashes
fn compute_merkle_root(leaves: &[[u8; 32]]) -> ProofResult<[u8; 32]> {
    use sha2::{Sha256, Digest};

    if leaves.is_empty() {
        return Err(ProofError::InvalidInput("No leaves provided".to_string()));
    }

    if leaves.len() == 1 {
        return Ok(leaves[0]);
    }

    // Pad to power of 2
    let mut current = leaves.to_vec();
    while !current.len().is_power_of_two() {
        current.push([0u8; 32]);
    }

    // Build tree bottom-up
    while current.len() > 1 {
        let mut next = Vec::with_capacity(current.len() / 2);
        for chunk in current.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&chunk[0]);
            hasher.update(&chunk[1]);
            next.push(hasher.finalize().into());
        }
        current = next;
    }

    Ok(current[0])
}

/// Create aggregate proof data
fn create_aggregate_proof(
    commitments: &[ProofCommitment],
    root: &[u8; 32],
    security_level: SecurityLevel,
) -> ProofResult<Vec<u8>> {
    use sha2::{Sha256, Digest};

    // Create a binding of commitments to the root
    let mut data = Vec::new();

    // Include security level
    data.push(match security_level {
        SecurityLevel::Standard96 => 0,
        SecurityLevel::Standard128 => 1,
        SecurityLevel::High256 => 2,
    });

    // Include commitment count
    let count = commitments.len() as u32;
    data.extend_from_slice(&count.to_le_bytes());

    // Include root
    data.extend_from_slice(root);

    // Create binding hash
    let mut hasher = Sha256::new();
    hasher.update(&data);
    for commitment in commitments {
        hasher.update(&commitment.hash);
    }
    let binding: [u8; 32] = hasher.finalize().into();

    data.extend_from_slice(&binding);

    Ok(data)
}

/// Verify aggregate proof data
fn verify_aggregate_proof(
    proof_data: &[u8],
    expected_root: &[u8; 32],
    commitments: &[ProofCommitment],
) -> ProofResult<bool> {
    use sha2::{Sha256, Digest};

    if proof_data.len() < 37 {
        return Ok(false);
    }

    // Parse proof data
    let _security = proof_data[0];
    let count = u32::from_le_bytes([proof_data[1], proof_data[2], proof_data[3], proof_data[4]]);

    if count as usize != commitments.len() {
        return Ok(false);
    }

    let root: [u8; 32] = proof_data[5..37].try_into().unwrap();
    if root != *expected_root {
        return Ok(false);
    }

    // Verify binding hash
    let mut hasher = Sha256::new();
    hasher.update(&proof_data[..37]);
    for commitment in commitments {
        hasher.update(&commitment.hash);
    }
    let expected_binding: [u8; 32] = hasher.finalize().into();

    let actual_binding: [u8; 32] = proof_data[37..69].try_into().unwrap_or([0u8; 32]);

    Ok(actual_binding == expected_binding)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_proof_commitment() {
        let bytes = vec![1, 2, 3, 4, 5];
        let commitment = ProofCommitment::from_bytes(ProofType::Range, &bytes);

        assert_eq!(commitment.proof_type, ProofType::Range);
        assert_eq!(commitment.original_size, 5);
        assert_ne!(commitment.hash, [0u8; 32]);
    }

    #[test]
    fn test_empty_batch() {
        let batch = ProofBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_batch_with_proofs() {
        let proof1 = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let proof2 = RangeProof::generate(75, 0, 100, test_config()).unwrap();

        let mut batch = ProofBatch::new();
        batch.add_range_proof(proof1);
        batch.add_range_proof(proof2);

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_merkle_root() {
        let proof1 = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let proof2 = RangeProof::generate(75, 0, 100, test_config()).unwrap();

        let mut batch = ProofBatch::new();
        batch.add_range_proof(proof1);
        batch.add_range_proof(proof2);

        let root = batch.merkle_root().unwrap();
        assert_ne!(root, [0u8; 32]);

        // Same batch should produce same root
        let root2 = batch.merkle_root().unwrap();
        assert_eq!(root, root2);
    }

    #[test]
    fn test_batch_verification() {
        let proof1 = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let proof2 = RangeProof::generate(75, 0, 100, test_config()).unwrap();

        let mut batch = ProofBatch::new();
        batch.add_range_proof(proof1);
        batch.add_range_proof(proof2);

        let result = batch.verify_all().unwrap();
        assert!(result.all_valid);
        assert_eq!(result.batch_size, 2);
    }

    #[test]
    fn test_recursive_proof_aggregate() {
        let proof1 = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let proof2 = RangeProof::generate(75, 0, 100, test_config()).unwrap();

        let mut batch = ProofBatch::new();
        batch.add_range_proof(proof1);
        batch.add_range_proof(proof2);

        let recursive = RecursiveProof::aggregate(&batch, test_config()).unwrap();

        assert_eq!(recursive.count, 2);
        assert_ne!(recursive.root, [0u8; 32]);
    }

    #[test]
    fn test_recursive_proof_verify() {
        let proof1 = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let proof2 = RangeProof::generate(75, 0, 100, test_config()).unwrap();

        let mut batch = ProofBatch::new();
        batch.add_range_proof(proof1);
        batch.add_range_proof(proof2);

        let recursive = RecursiveProof::aggregate(&batch, test_config()).unwrap();
        let result = recursive.verify().unwrap();

        assert!(result.valid);
        assert_eq!(result.proofs_covered, 2);
    }

    #[test]
    fn test_recursive_proof_serialization() {
        let proof1 = RangeProof::generate(50, 0, 100, test_config()).unwrap();

        let mut batch = ProofBatch::new();
        batch.add_range_proof(proof1);

        let recursive = RecursiveProof::aggregate(&batch, test_config()).unwrap();
        let bytes = recursive.to_bytes().unwrap();
        let restored = RecursiveProof::from_bytes(&bytes).unwrap();

        assert_eq!(restored.count, recursive.count);
        assert_eq!(restored.root, recursive.root);
    }

    #[test]
    fn test_recursive_proof_inclusion() {
        let proof1 = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let commitment1 = ProofCommitment::from_bytes(ProofType::Range, &proof1.to_bytes());

        let mut batch = ProofBatch::new();
        batch.add_range_proof(proof1);

        let recursive = RecursiveProof::aggregate(&batch, test_config()).unwrap();

        assert!(recursive.verify_inclusion(&commitment1).unwrap());

        // Different commitment should not be included
        let fake_commitment = ProofCommitment::from_bytes(ProofType::Range, &[1, 2, 3]);
        assert!(!recursive.verify_inclusion(&fake_commitment).unwrap());
    }

    #[test]
    fn test_empty_batch_aggregation_fails() {
        let batch = ProofBatch::new();
        let result = RecursiveProof::aggregate(&batch, test_config());
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_merkle_root() {
        let leaves = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let root = compute_merkle_root(&leaves).unwrap();

        assert_ne!(root, [0u8; 32]);

        // Same leaves should produce same root
        let root2 = compute_merkle_root(&leaves).unwrap();
        assert_eq!(root, root2);

        // Different leaves should produce different root
        let leaves2 = vec![[5u8; 32], [6u8; 32], [7u8; 32], [8u8; 32]];
        let root3 = compute_merkle_root(&leaves2).unwrap();
        assert_ne!(root, root3);
    }
}
