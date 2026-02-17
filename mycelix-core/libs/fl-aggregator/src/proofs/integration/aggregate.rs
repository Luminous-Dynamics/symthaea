//! Aggregate Proofs
//!
//! Combines multiple proofs into aggregate structures for efficient verification
//! and compact representation.
//!
//! ## Overview
//!
//! Aggregate proofs allow:
//! - Combining multiple FL submissions into a single verified aggregate
//! - Creating Merkle commitments over verified proofs
//! - Efficiently attesting that a set of proofs were all verified
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::{
//!     GradientAggregateProof, AggregateBuilder, VerifiedGradientSubmission,
//! };
//!
//! // Build aggregate from verified submissions
//! let mut builder = AggregateBuilder::new();
//! for submission in verified_submissions {
//!     builder.add_submission(submission)?;
//! }
//!
//! // Finalize into aggregate proof
//! let aggregate = builder.finalize()?;
//!
//! // Verify the aggregate
//! let is_valid = aggregate.verify()?;
//! println!("Round {} with {} participants", aggregate.round_id(), aggregate.participant_count());
//! ```

use crate::proofs::{
    ProofError, ProofResult, ProofConfig, VerificationResult, ProofType,
    build_merkle_tree, MerklePathNode,
};
use crate::NodeId;
use serde::{Deserialize, Serialize};
use blake3::Hasher;
use std::time::Instant;

/// Aggregate proof for multiple FL gradient submissions
#[derive(Clone)]
pub struct GradientAggregateProof {
    /// Round identifier
    round_id: u64,

    /// Merkle root of all submission commitments
    commitment_root: [u8; 32],

    /// Number of participants
    participant_count: usize,

    /// Individual submission metadata
    submissions: Vec<AggregateSubmissionMeta>,

    /// Total proof size (sum of individual proofs)
    total_proof_size: usize,

    /// Aggregation timestamp
    timestamp: u64,

    /// Configuration used (reserved for future recursive proof generation)
    #[allow(dead_code)]
    config: ProofConfig,
}

/// Metadata for a submission in the aggregate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregateSubmissionMeta {
    /// Node ID
    pub node_id: NodeId,

    /// Gradient commitment
    pub commitment: [u8; 32],

    /// Gradient dimension
    pub gradient_dim: usize,

    /// Proof size in bytes
    pub proof_size: usize,

    /// Merkle path to root (for membership verification)
    pub merkle_path: Vec<MerklePathNode>,

    /// Position in the aggregate
    pub position: usize,
}

impl GradientAggregateProof {
    /// Get the round ID
    pub fn round_id(&self) -> u64 {
        self.round_id
    }

    /// Get the commitment root
    pub fn commitment_root(&self) -> [u8; 32] {
        self.commitment_root
    }

    /// Get the participant count
    pub fn participant_count(&self) -> usize {
        self.participant_count
    }

    /// Get submission metadata
    pub fn submissions(&self) -> &[AggregateSubmissionMeta] {
        &self.submissions
    }

    /// Get total proof size
    pub fn total_proof_size(&self) -> usize {
        self.total_proof_size
    }

    /// Get timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Verify the aggregate structure
    ///
    /// This verifies that:
    /// - The Merkle root is correctly computed
    /// - All merkle paths are valid
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        let start = Instant::now();

        // Collect all commitment leaves
        let leaves: Vec<[u8; 32]> = self.submissions
            .iter()
            .map(|s| s.commitment)
            .collect();

        if leaves.is_empty() {
            return Ok(VerificationResult::failure(
                ProofType::GradientIntegrity,
                start.elapsed(),
                "Empty aggregate",
            ));
        }

        // Recompute Merkle root
        let (computed_root, _) = build_merkle_tree(&leaves);

        if computed_root != self.commitment_root {
            return Ok(VerificationResult::failure(
                ProofType::GradientIntegrity,
                start.elapsed(),
                "Merkle root mismatch",
            ));
        }

        // Verify each submission's Merkle path
        for submission in &self.submissions {
            if !verify_merkle_path(
                &submission.commitment,
                &submission.merkle_path,
                &self.commitment_root,
            ) {
                return Ok(VerificationResult::failure(
                    ProofType::GradientIntegrity,
                    start.elapsed(),
                    &format!("Invalid Merkle path for node {}", submission.node_id),
                ));
            }
        }

        Ok(VerificationResult::success(
            ProofType::GradientIntegrity,
            start.elapsed(),
        ))
    }

    /// Get submission by node ID
    pub fn get_submission(&self, node_id: &str) -> Option<&AggregateSubmissionMeta> {
        self.submissions.iter().find(|s| s.node_id == node_id)
    }

    /// Verify membership of a specific commitment
    pub fn verify_membership(&self, commitment: &[u8; 32], path: &[MerklePathNode]) -> bool {
        verify_merkle_path(commitment, path, &self.commitment_root)
    }

    /// Serialize the aggregate proof to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Round ID (8 bytes)
        bytes.extend_from_slice(&self.round_id.to_le_bytes());

        // Commitment root (32 bytes)
        bytes.extend_from_slice(&self.commitment_root);

        // Participant count (8 bytes)
        bytes.extend_from_slice(&(self.participant_count as u64).to_le_bytes());

        // Timestamp (8 bytes)
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());

        // Total proof size (8 bytes)
        bytes.extend_from_slice(&(self.total_proof_size as u64).to_le_bytes());

        // Submissions (variable)
        bytes.extend_from_slice(&(self.submissions.len() as u64).to_le_bytes());
        for submission in &self.submissions {
            bytes.extend_from_slice(&serialize_submission_meta(submission));
        }

        bytes
    }

    /// Get the size of the aggregate proof structure in bytes
    pub fn size(&self) -> usize {
        8 + 32 + 8 + 8 + 8 + // Fixed fields
        self.submissions.len() * (32 + 8 + 8 + 8) + // Per-submission base
        self.submissions.iter().map(|s| s.node_id.len() + s.merkle_path.len() * 33).sum::<usize>()
    }
}

/// Builder for creating aggregate proofs
pub struct AggregateBuilder {
    /// Round ID
    round_id: u64,

    /// Collected submissions (node_id -> commitment, proof_size, gradient_dim)
    submissions: Vec<(NodeId, [u8; 32], usize, usize)>,

    /// Configuration
    config: ProofConfig,
}

impl AggregateBuilder {
    /// Create a new aggregate builder
    pub fn new() -> Self {
        Self {
            round_id: 0,
            submissions: Vec::new(),
            config: ProofConfig::default(),
        }
    }

    /// Create with a specific round ID
    pub fn with_round_id(round_id: u64) -> Self {
        Self {
            round_id,
            submissions: Vec::new(),
            config: ProofConfig::default(),
        }
    }

    /// Set the proof configuration
    pub fn with_config(mut self, config: ProofConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a verified submission to the aggregate
    pub fn add_submission(
        &mut self,
        node_id: NodeId,
        commitment: [u8; 32],
        gradient_dim: usize,
        proof_size: usize,
    ) -> &mut Self {
        self.submissions.push((node_id, commitment, proof_size, gradient_dim));
        self
    }

    /// Add a submission from gradient data
    pub fn add_gradient(
        &mut self,
        node_id: NodeId,
        gradient: &[f32],
        proof_size: usize,
    ) -> &mut Self {
        let commitment = compute_gradient_commitment(gradient);
        self.submissions.push((node_id, commitment, proof_size, gradient.len()));
        self
    }

    /// Get current submission count
    pub fn len(&self) -> usize {
        self.submissions.len()
    }

    /// Check if builder is empty
    pub fn is_empty(&self) -> bool {
        self.submissions.is_empty()
    }

    /// Finalize the builder into an aggregate proof
    pub fn finalize(self) -> ProofResult<GradientAggregateProof> {
        if self.submissions.is_empty() {
            return Err(ProofError::InvalidPublicInputs(
                "Cannot create aggregate from empty submissions".to_string()
            ));
        }

        // Build commitment leaves
        let leaves: Vec<[u8; 32]> = self.submissions
            .iter()
            .map(|(_, commitment, _, _)| *commitment)
            .collect();

        // Build Merkle tree
        let (root, all_paths) = build_merkle_tree(&leaves);

        // Create submission metadata with paths
        let mut submissions = Vec::with_capacity(self.submissions.len());
        let mut total_proof_size = 0;

        for (i, (node_id, commitment, proof_size, gradient_dim)) in self.submissions.into_iter().enumerate() {
            total_proof_size += proof_size;

            submissions.push(AggregateSubmissionMeta {
                node_id,
                commitment,
                gradient_dim,
                proof_size,
                merkle_path: all_paths[i].clone(),
                position: i,
            });
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(GradientAggregateProof {
            round_id: self.round_id,
            commitment_root: root,
            participant_count: submissions.len(),
            submissions,
            total_proof_size,
            timestamp,
            config: self.config,
        })
    }
}

impl Default for AggregateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute gradient commitment (hash of gradient values)
fn compute_gradient_commitment(gradient: &[f32]) -> [u8; 32] {
    let mut hasher = Hasher::new();

    for value in gradient {
        hasher.update(&value.to_le_bytes());
    }

    *hasher.finalize().as_bytes()
}

/// Verify a Merkle path (must match build_merkle_tree which uses blake3)
fn verify_merkle_path(leaf: &[u8; 32], path: &[MerklePathNode], root: &[u8; 32]) -> bool {
    let mut current = *leaf;

    for node in path {
        let mut hasher = Hasher::new();
        if node.position {
            // We're the right child, sibling is on left
            hasher.update(&node.sibling);
            hasher.update(&current);
        } else {
            // We're the left child, sibling is on right
            hasher.update(&current);
            hasher.update(&node.sibling);
        }
        current = *hasher.finalize().as_bytes();
    }

    current == *root
}

/// Serialize submission metadata
fn serialize_submission_meta(meta: &AggregateSubmissionMeta) -> Vec<u8> {
    let mut bytes = Vec::new();

    // Node ID length and bytes
    bytes.extend_from_slice(&(meta.node_id.len() as u32).to_le_bytes());
    bytes.extend_from_slice(meta.node_id.as_bytes());

    // Commitment
    bytes.extend_from_slice(&meta.commitment);

    // Gradient dim
    bytes.extend_from_slice(&(meta.gradient_dim as u64).to_le_bytes());

    // Proof size
    bytes.extend_from_slice(&(meta.proof_size as u64).to_le_bytes());

    // Position
    bytes.extend_from_slice(&(meta.position as u64).to_le_bytes());

    // Merkle path
    bytes.extend_from_slice(&(meta.merkle_path.len() as u32).to_le_bytes());
    for node in &meta.merkle_path {
        bytes.extend_from_slice(&node.sibling);
        bytes.push(if node.position { 1 } else { 0 });
    }

    bytes
}

/// Aggregate statistics
#[derive(Debug, Clone, Default)]
pub struct AggregateStats {
    /// Total number of submissions
    pub submission_count: usize,

    /// Total gradient elements across all submissions
    pub total_gradient_elements: usize,

    /// Total proof size in bytes
    pub total_proof_bytes: usize,

    /// Average proof size per submission
    pub avg_proof_size: usize,

    /// Average gradient dimension
    pub avg_gradient_dim: usize,
}

impl GradientAggregateProof {
    /// Get aggregate statistics
    pub fn stats(&self) -> AggregateStats {
        let total_gradient_elements: usize = self.submissions
            .iter()
            .map(|s| s.gradient_dim)
            .sum();

        let avg_proof_size = if self.submissions.is_empty() {
            0
        } else {
            self.total_proof_size / self.submissions.len()
        };

        let avg_gradient_dim = if self.submissions.is_empty() {
            0
        } else {
            total_gradient_elements / self.submissions.len()
        };

        AggregateStats {
            submission_count: self.participant_count,
            total_gradient_elements,
            total_proof_bytes: self.total_proof_size,
            avg_proof_size,
            avg_gradient_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_builder_empty() {
        let builder = AggregateBuilder::new();
        let result = builder.finalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_aggregate_builder_single() {
        let mut builder = AggregateBuilder::with_round_id(1);

        let gradient = vec![0.1, 0.2, 0.3, 0.4];
        builder.add_gradient("node_1".to_string(), &gradient, 1000);

        let aggregate = builder.finalize().unwrap();

        assert_eq!(aggregate.round_id(), 1);
        assert_eq!(aggregate.participant_count(), 1);
        assert_eq!(aggregate.submissions().len(), 1);
        assert_eq!(aggregate.submissions()[0].node_id, "node_1");
        assert_eq!(aggregate.submissions()[0].gradient_dim, 4);
    }

    #[test]
    fn test_aggregate_builder_multiple() {
        let mut builder = AggregateBuilder::with_round_id(42);

        for i in 0..8 {
            let gradient: Vec<f32> = (0..100).map(|j| (i * 100 + j) as f32 * 0.001).collect();
            builder.add_gradient(format!("node_{}", i), &gradient, 15000);
        }

        let aggregate = builder.finalize().unwrap();

        assert_eq!(aggregate.round_id(), 42);
        assert_eq!(aggregate.participant_count(), 8);
        assert_eq!(aggregate.total_proof_size(), 15000 * 8);

        // Verify the aggregate
        let result = aggregate.verify().unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_aggregate_verification() {
        let mut builder = AggregateBuilder::with_round_id(1);

        // Add 4 submissions (power of 2 for clean Merkle tree)
        for i in 0..4 {
            let commitment = [i as u8; 32];
            builder.add_submission(
                format!("node_{}", i),
                commitment,
                100,
                5000,
            );
        }

        let aggregate = builder.finalize().unwrap();

        // Verify overall structure
        let result = aggregate.verify().unwrap();
        assert!(result.valid);

        // Verify individual membership
        for submission in aggregate.submissions() {
            assert!(aggregate.verify_membership(
                &submission.commitment,
                &submission.merkle_path,
            ));
        }
    }

    #[test]
    fn test_aggregate_stats() {
        let mut builder = AggregateBuilder::new();

        builder.add_submission("node_1".to_string(), [1u8; 32], 1000, 10000);
        builder.add_submission("node_2".to_string(), [2u8; 32], 2000, 20000);
        builder.add_submission("node_3".to_string(), [3u8; 32], 3000, 30000);

        let aggregate = builder.finalize().unwrap();
        let stats = aggregate.stats();

        assert_eq!(stats.submission_count, 3);
        assert_eq!(stats.total_gradient_elements, 6000);
        assert_eq!(stats.total_proof_bytes, 60000);
        assert_eq!(stats.avg_proof_size, 20000);
        assert_eq!(stats.avg_gradient_dim, 2000);
    }

    #[test]
    fn test_aggregate_get_submission() {
        let mut builder = AggregateBuilder::new();

        builder.add_submission("alice".to_string(), [1u8; 32], 100, 5000);
        builder.add_submission("bob".to_string(), [2u8; 32], 200, 6000);

        let aggregate = builder.finalize().unwrap();

        let alice = aggregate.get_submission("alice").unwrap();
        assert_eq!(alice.gradient_dim, 100);
        assert_eq!(alice.proof_size, 5000);

        let bob = aggregate.get_submission("bob").unwrap();
        assert_eq!(bob.gradient_dim, 200);

        assert!(aggregate.get_submission("charlie").is_none());
    }

    #[test]
    fn test_aggregate_serialization() {
        let mut builder = AggregateBuilder::with_round_id(123);

        builder.add_submission("node_1".to_string(), [1u8; 32], 100, 5000);
        builder.add_submission("node_2".to_string(), [2u8; 32], 200, 6000);

        let aggregate = builder.finalize().unwrap();

        let bytes = aggregate.to_bytes();
        assert!(!bytes.is_empty());

        // Verify size calculation
        let size = aggregate.size();
        assert!(size > 0);
    }

    #[test]
    fn test_merkle_path_verification() {
        // Create 4 leaves
        let leaves: Vec<[u8; 32]> = (0..4)
            .map(|i| {
                let mut leaf = [0u8; 32];
                leaf[0] = i;
                leaf
            })
            .collect();

        let (root, all_paths) = build_merkle_tree(&leaves);

        // Verify each leaf's path
        for (i, leaf) in leaves.iter().enumerate() {
            assert!(
                verify_merkle_path(leaf, &all_paths[i], &root),
                "Path verification failed for leaf {}",
                i
            );
        }

        // Verify invalid leaf fails
        let invalid_leaf = [255u8; 32];
        assert!(!verify_merkle_path(&invalid_leaf, &all_paths[0], &root));
    }
}
