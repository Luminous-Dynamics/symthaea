// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proof aggregation for efficient batch verification.
//!
//! This module provides functionality to aggregate multiple tax proofs
//! into a single verifiable proof, reducing verification costs.
//!
//! # Use Cases
//!
//! - **Batch verification**: Verify many proofs with a single check
//! - **On-chain efficiency**: Reduce gas costs for blockchain anchoring
//! - **Privacy**: Aggregate proofs without revealing individual details
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_zk_tax::aggregation::{ProofAggregator, AggregatedProof};
//!
//! let aggregator = ProofAggregator::new();
//!
//! // Add proofs from multiple taxpayers
//! aggregator.add_proof(proof1);
//! aggregator.add_proof(proof2);
//! aggregator.add_proof(proof3);
//!
//! // Create aggregated proof
//! let aggregated = aggregator.aggregate()?;
//!
//! // Verify all proofs at once
//! assert!(aggregated.verify().is_ok());
//!
//! // Verify a specific proof is included
//! assert!(aggregated.contains(&proof1.commitment));
//! ```

use crate::{Error, Result, TaxBracketProof, BracketCommitment};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

// =============================================================================
// Aggregated Proof
// =============================================================================

/// An aggregated proof combining multiple individual proofs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregatedProof {
    /// Number of individual proofs
    pub proof_count: usize,

    /// Merkle root of all proof commitments
    pub merkle_root: [u8; 32],

    /// Aggregated commitment (XOR of all commitments)
    pub aggregate_commitment: [u8; 32],

    /// Minimum tax year in the batch
    pub min_year: u32,

    /// Maximum tax year in the batch
    pub max_year: u32,

    /// Sum of all bracket indices (for statistics)
    pub bracket_sum: u64,

    /// Bitmap of jurisdictions included (up to 64)
    pub jurisdiction_bitmap: u64,

    /// Individual commitments (for verification)
    pub commitments: Vec<[u8; 32]>,

    /// Merkle proofs for each commitment
    pub merkle_proofs: Vec<Vec<[u8; 32]>>,

    /// Aggregated ZK receipt (if using SNARK aggregation)
    pub aggregated_receipt: Option<Vec<u8>>,

    /// Timestamp of aggregation
    pub aggregated_at: u64,
}

impl AggregatedProof {
    /// Verify the aggregated proof.
    pub fn verify(&self) -> Result<()> {
        // Verify proof count matches commitments
        if self.commitments.len() != self.proof_count {
            return Err(Error::invalid_proof(format!(
                "Commitment count mismatch: expected {}, got {}",
                self.proof_count,
                self.commitments.len()
            )));
        }

        // Verify Merkle root
        let computed_root = compute_merkle_root(&self.commitments);
        if computed_root != self.merkle_root {
            return Err(Error::CommitmentMismatch {
                expected: hex::encode(self.merkle_root),
                actual: hex::encode(computed_root),
            });
        }

        // Verify aggregate commitment
        let computed_aggregate = compute_aggregate(&self.commitments);
        if computed_aggregate != self.aggregate_commitment {
            return Err(Error::invalid_proof("Aggregate commitment mismatch"));
        }

        Ok(())
    }

    /// Check if a specific commitment is included.
    pub fn contains(&self, commitment: &BracketCommitment) -> bool {
        let bytes = commitment.to_bytes();
        self.commitments.iter().any(|c| *c == bytes)
    }

    /// Get a Merkle proof for a specific commitment.
    pub fn merkle_proof_for(&self, commitment: &BracketCommitment) -> Option<&Vec<[u8; 32]>> {
        let bytes = commitment.to_bytes();
        self.commitments
            .iter()
            .position(|c| *c == bytes)
            .map(|i| &self.merkle_proofs[i])
    }

    /// Verify a specific commitment is in the tree.
    pub fn verify_inclusion(&self, commitment: &BracketCommitment) -> bool {
        let bytes = commitment.to_bytes();

        if let Some(idx) = self.commitments.iter().position(|c| *c == bytes) {
            if idx < self.merkle_proofs.len() {
                return verify_merkle_proof(
                    &bytes,
                    &self.merkle_proofs[idx],
                    idx,
                    &self.merkle_root,
                );
            }
        }

        false
    }

    /// Get summary statistics.
    pub fn summary(&self) -> AggregationSummary {
        let avg_bracket = if self.proof_count > 0 {
            self.bracket_sum as f64 / self.proof_count as f64
        } else {
            0.0
        };

        let jurisdiction_count = self.jurisdiction_bitmap.count_ones();

        AggregationSummary {
            proof_count: self.proof_count,
            year_range: (self.min_year, self.max_year),
            average_bracket: avg_bracket,
            jurisdiction_count: jurisdiction_count as usize,
        }
    }
}

/// Summary of an aggregated proof.
#[derive(Clone, Debug, Serialize)]
pub struct AggregationSummary {
    pub proof_count: usize,
    pub year_range: (u32, u32),
    pub average_bracket: f64,
    pub jurisdiction_count: usize,
}

// =============================================================================
// Proof Aggregator
// =============================================================================

/// Builder for creating aggregated proofs.
#[derive(Default)]
pub struct ProofAggregator {
    proofs: Vec<TaxBracketProof>,
}

impl ProofAggregator {
    /// Create a new aggregator.
    pub fn new() -> Self {
        Self { proofs: Vec::new() }
    }

    /// Add a proof to the aggregation.
    pub fn add_proof(&mut self, proof: TaxBracketProof) -> &mut Self {
        self.proofs.push(proof);
        self
    }

    /// Add multiple proofs.
    pub fn add_proofs(&mut self, proofs: impl IntoIterator<Item = TaxBracketProof>) -> &mut Self {
        self.proofs.extend(proofs);
        self
    }

    /// Get the number of proofs added.
    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.proofs.is_empty()
    }

    /// Create the aggregated proof.
    pub fn aggregate(&self) -> Result<AggregatedProof> {
        if self.proofs.is_empty() {
            return Err(Error::proof_generation("Cannot aggregate empty proof set"));
        }

        // Extract commitments
        let commitments: Vec<[u8; 32]> = self.proofs
            .iter()
            .map(|p| p.commitment.to_bytes())
            .collect();

        // Compute Merkle tree
        let merkle_root = compute_merkle_root(&commitments);

        // Compute aggregate commitment
        let aggregate_commitment = compute_aggregate(&commitments);

        // Compute statistics
        let min_year = self.proofs.iter().map(|p| p.tax_year).min().unwrap();
        let max_year = self.proofs.iter().map(|p| p.tax_year).max().unwrap();
        let bracket_sum: u64 = self.proofs.iter().map(|p| p.bracket_index as u64).sum();

        // Build jurisdiction bitmap
        let mut jurisdiction_bitmap: u64 = 0;
        for proof in &self.proofs {
            let j_idx = proof.jurisdiction as u64;
            if j_idx < 64 {
                jurisdiction_bitmap |= 1 << j_idx;
            }
        }

        // Generate Merkle proofs for each commitment
        let merkle_proofs = generate_all_merkle_proofs(&commitments);

        // Get current timestamp
        let aggregated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(AggregatedProof {
            proof_count: self.proofs.len(),
            merkle_root,
            aggregate_commitment,
            min_year,
            max_year,
            bracket_sum,
            jurisdiction_bitmap,
            commitments,
            merkle_proofs,
            aggregated_receipt: None, // Would use SNARK in production
            aggregated_at,
        })
    }

    /// Clear all proofs.
    pub fn clear(&mut self) {
        self.proofs.clear();
    }
}

// =============================================================================
// Merkle Tree Utilities
// =============================================================================

/// Compute Merkle root of commitments.
fn compute_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut current_level = leaves.to_vec();

    while current_level.len() > 1 {
        let mut next_level = Vec::new();

        for chunk in current_level.chunks(2) {
            let hash = if chunk.len() == 2 {
                hash_pair(&chunk[0], &chunk[1])
            } else {
                hash_pair(&chunk[0], &chunk[0])
            };
            next_level.push(hash);
        }

        current_level = next_level;
    }

    current_level[0]
}

/// Hash two nodes together.
fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);

    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

/// Compute aggregate commitment (XOR of all).
fn compute_aggregate(commitments: &[[u8; 32]]) -> [u8; 32] {
    let mut aggregate = [0u8; 32];

    for commitment in commitments {
        for (i, byte) in commitment.iter().enumerate() {
            aggregate[i] ^= byte;
        }
    }

    aggregate
}

/// Generate Merkle proofs for all leaves.
fn generate_all_merkle_proofs(leaves: &[[u8; 32]]) -> Vec<Vec<[u8; 32]>> {
    let mut proofs = Vec::with_capacity(leaves.len());

    for i in 0..leaves.len() {
        proofs.push(generate_merkle_proof(leaves, i));
    }

    proofs
}

/// Generate Merkle proof for a specific leaf.
fn generate_merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<[u8; 32]> {
    if leaves.len() <= 1 {
        return Vec::new();
    }

    let mut proof = Vec::new();
    let mut current_level = leaves.to_vec();
    let mut current_index = index;

    while current_level.len() > 1 {
        let sibling_index = if current_index % 2 == 0 {
            current_index + 1
        } else {
            current_index - 1
        };

        if sibling_index < current_level.len() {
            proof.push(current_level[sibling_index]);
        } else {
            proof.push(current_level[current_index]);
        }

        // Build next level
        let mut next_level = Vec::new();
        for chunk in current_level.chunks(2) {
            let hash = if chunk.len() == 2 {
                hash_pair(&chunk[0], &chunk[1])
            } else {
                hash_pair(&chunk[0], &chunk[0])
            };
            next_level.push(hash);
        }

        current_level = next_level;
        current_index /= 2;
    }

    proof
}

/// Verify a Merkle proof.
fn verify_merkle_proof(
    leaf: &[u8; 32],
    proof: &[[u8; 32]],
    index: usize,
    root: &[u8; 32],
) -> bool {
    let mut current = *leaf;
    let mut current_index = index;

    for sibling in proof {
        current = if current_index % 2 == 0 {
            hash_pair(&current, sibling)
        } else {
            hash_pair(sibling, &current)
        };
        current_index /= 2;
    }

    current == *root
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Jurisdiction, FilingStatus, TaxBracketProver};

    fn create_test_proofs(count: usize) -> Vec<TaxBracketProof> {
        let prover = TaxBracketProver::dev_mode();
        (0..count)
            .map(|i| {
                let income = 50_000 + (i as u64 * 10_000);
                prover.prove(income, Jurisdiction::US, FilingStatus::Single, 2024).unwrap()
            })
            .collect()
    }

    #[test]
    fn test_aggregation() {
        let proofs = create_test_proofs(5);

        let mut aggregator = ProofAggregator::new();
        for proof in &proofs {
            aggregator.add_proof(proof.clone());
        }

        let aggregated = aggregator.aggregate().unwrap();

        assert_eq!(aggregated.proof_count, 5);
        assert!(aggregated.verify().is_ok());
    }

    #[test]
    fn test_inclusion() {
        let proofs = create_test_proofs(3);

        let mut aggregator = ProofAggregator::new();
        for proof in &proofs {
            aggregator.add_proof(proof.clone());
        }

        let aggregated = aggregator.aggregate().unwrap();

        // All proofs should be included
        for proof in &proofs {
            assert!(aggregated.contains(&proof.commitment));
            assert!(aggregated.verify_inclusion(&proof.commitment));
        }
    }

    #[test]
    fn test_merkle_proof() {
        let leaves: Vec<[u8; 32]> = (0..4)
            .map(|i| {
                let mut arr = [0u8; 32];
                arr[0] = i;
                arr
            })
            .collect();

        let root = compute_merkle_root(&leaves);

        // Verify each proof
        for (i, leaf) in leaves.iter().enumerate() {
            let proof = generate_merkle_proof(&leaves, i);
            assert!(verify_merkle_proof(leaf, &proof, i, &root));
        }
    }

    #[test]
    fn test_summary() {
        let proofs = create_test_proofs(10);

        let mut aggregator = ProofAggregator::new();
        aggregator.add_proofs(proofs);

        let aggregated = aggregator.aggregate().unwrap();
        let summary = aggregated.summary();

        assert_eq!(summary.proof_count, 10);
        assert_eq!(summary.year_range, (2024, 2024));
        assert!(summary.jurisdiction_count >= 1);
    }
}
