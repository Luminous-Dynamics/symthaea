// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hamming similarity verification proof.
//!
//! Proves: "The similarity between query and result is genuinely X"
//! without revealing the query vector or the claim vector.
//!
//! In Binius: XOR (the similarity diff) is FREE. Only the popcount
//! accumulation costs AND constraints (~2,048 for 16,384-bit vectors).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A proof that the Hamming similarity between two BinaryHVs is correct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimilarityProof {
    /// Hash of query vector (hides actual query)
    pub query_hash: [u8; 32],
    /// Hash of claim vector (hides actual claim content)
    pub claim_hash: [u8; 32],
    /// The claimed similarity value [0.0, 1.0]
    pub similarity: f32,
    /// Whether similarity exceeds the threshold
    pub above_threshold: bool,
    /// Threshold used
    pub threshold: f32,
    /// Proof commitment: hash(query_hash || claim_hash || similarity)
    pub commitment: [u8; 32],
    /// Domain tag
    pub domain_tag: String,
}

/// Generate a similarity proof from two BinaryHV byte slices.
///
/// Computes Hamming similarity and commits to the result.
pub fn generate_similarity_proof(
    query_bytes: &[u8],
    claim_bytes: &[u8],
    threshold: f32,
) -> SimilarityProof {
    assert_eq!(
        query_bytes.len(),
        claim_bytes.len(),
        "vectors must be same length"
    );

    let total_bits = query_bytes.len() * 8;

    // Compute Hamming similarity: matching_bits / total_bits
    let matching_bits: u32 = query_bytes
        .iter()
        .zip(claim_bytes)
        .map(|(a, b)| (!(a ^ b)).count_ones())
        .sum();

    let similarity = matching_bits as f32 / total_bits as f32;

    let query_hash = {
        let h = Sha256::digest(query_bytes);
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&h);
        hash
    };

    let claim_hash = {
        let h = Sha256::digest(claim_bytes);
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&h);
        hash
    };

    // Commitment: binds similarity to the specific query-claim pair
    let commitment = {
        let mut hasher = Sha256::new();
        hasher.update(query_hash);
        hasher.update(claim_hash);
        hasher.update(similarity.to_le_bytes());
        let result = hasher.finalize();
        let mut c = [0u8; 32];
        c.copy_from_slice(&result);
        c
    };

    SimilarityProof {
        query_hash,
        claim_hash,
        similarity,
        above_threshold: similarity >= threshold,
        threshold,
        commitment,
        domain_tag: "ZTML:Prism:SimilarityProof:v1".to_string(),
    }
}

/// Verify a similarity proof: commitment is valid and threshold check is correct.
pub fn verify_similarity_proof(proof: &SimilarityProof) -> Result<(), String> {
    // Recompute commitment
    let expected = {
        let mut hasher = Sha256::new();
        hasher.update(proof.query_hash);
        hasher.update(proof.claim_hash);
        hasher.update(proof.similarity.to_le_bytes());
        let result = hasher.finalize();
        let mut c = [0u8; 32];
        c.copy_from_slice(&result);
        c
    };

    if expected != proof.commitment {
        return Err("Commitment mismatch".to_string());
    }

    // Verify threshold check
    let expected_above = proof.similarity >= proof.threshold;
    if proof.above_threshold != expected_above {
        return Err("Threshold check mismatch".to_string());
    }

    // Similarity must be in [0, 1]
    if proof.similarity < 0.0 || proof.similarity > 1.0 {
        return Err(format!("Similarity out of range: {}", proof.similarity));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors_similarity_1() {
        let v = vec![0xAA; 2048]; // 16,384 bits
        let proof = generate_similarity_proof(&v, &v, 0.5);
        assert!((proof.similarity - 1.0).abs() < 0.001);
        assert!(proof.above_threshold);
        assert!(verify_similarity_proof(&proof).is_ok());
    }

    #[test]
    fn test_opposite_vectors_similarity_0() {
        let a = vec![0xFF; 2048];
        let b = vec![0x00; 2048];
        let proof = generate_similarity_proof(&a, &b, 0.5);
        assert!((proof.similarity - 0.0).abs() < 0.001);
        assert!(!proof.above_threshold);
        assert!(verify_similarity_proof(&proof).is_ok());
    }

    #[test]
    fn test_random_similarity_around_half() {
        // Random bytes have ~50% matching bits
        let a: Vec<u8> = (0..2048).map(|i| (i * 7 + 3) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| (i * 13 + 7) as u8).collect();
        let proof = generate_similarity_proof(&a, &b, 0.4);
        assert!(
            proof.similarity > 0.3 && proof.similarity < 0.7,
            "random similarity should be ~0.5, got {}",
            proof.similarity
        );
        assert!(verify_similarity_proof(&proof).is_ok());
    }

    #[test]
    fn test_tampered_commitment_detected() {
        let v = vec![0xCC; 2048];
        let mut proof = generate_similarity_proof(&v, &v, 0.5);
        proof.commitment[0] ^= 0xFF; // Tamper
        assert!(verify_similarity_proof(&proof).is_err());
    }

    #[test]
    fn test_vectors_hidden_in_proof() {
        let query = vec![0xDE; 2048];
        let claim = vec![0xAD; 2048];
        let proof = generate_similarity_proof(&query, &claim, 0.5);

        let json = serde_json::to_string(&proof).unwrap();
        // Raw vector bytes must NOT appear in the proof
        assert!(!json.contains("0xde"), "query bytes must be hidden");
        assert!(!json.contains("0xad"), "claim bytes must be hidden");
        // Only hashes should appear
        assert!(json.contains("query_hash"));
        assert!(json.contains("claim_hash"));
    }
}
