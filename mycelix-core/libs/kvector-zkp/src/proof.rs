// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! High-level proof API for K-Vector range proofs
//!
//! Provides a simple interface for generating and verifying proofs.

use serde::{Deserialize, Serialize};
use winterfell::Proof;

use crate::air::NUM_COMPONENTS;
use crate::error::{ZkpError, ZkpResult};
use crate::prover::{KVectorProver, KVectorTrace, KVectorWitness};
use crate::verifier::verify_kvector_proof;

/// A complete K-Vector range proof with commitment and targets
#[derive(Clone, Debug)]
pub struct KVectorRangeProof {
    /// The STARK proof
    pub proof: Proof,
    /// Commitment to the K-Vector values
    pub commitment: [u8; 32],
    /// Scaled target values (public)
    pub targets: [u64; NUM_COMPONENTS],
}

impl KVectorRangeProof {
    /// Generate a proof that all K-Vector components are in [0, 1]
    pub fn prove(witness: &KVectorWitness) -> ZkpResult<Self> {
        use winterfell::Prover;

        // Validate the witness first
        witness.validate()?;

        // Build the execution trace
        let trace = KVectorTrace::new(witness)?;

        // Create prover and generate proof
        let prover = KVectorProver::new();
        let proof = prover
            .prove(trace)
            .map_err(|e| ZkpError::ProofGenerationFailed(format!("{:?}", e)))?;

        Ok(Self {
            proof,
            commitment: witness.commitment(),
            targets: witness.scaled_values(),
        })
    }

    /// Verify this proof
    pub fn verify(&self) -> ZkpResult<()> {
        verify_kvector_proof(self.proof.clone(), self.commitment, self.targets)
    }

    /// Serialize the proof to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let proof_bytes = self.proof.to_bytes();
        let mut result = Vec::with_capacity(32 + 8 * NUM_COMPONENTS + 8 + proof_bytes.len());

        // Commitment (32 bytes)
        result.extend_from_slice(&self.commitment);

        // Targets (8 bytes each × NUM_COMPONENTS)
        for &target in &self.targets {
            result.extend_from_slice(&target.to_le_bytes());
        }

        // Proof length (8 bytes, little-endian)
        result.extend_from_slice(&(proof_bytes.len() as u64).to_le_bytes());

        // Proof data
        result.extend_from_slice(&proof_bytes);

        result
    }

    /// Deserialize a proof from bytes
    pub fn from_bytes(bytes: &[u8]) -> ZkpResult<Self> {
        let header_size = 32 + 8 * NUM_COMPONENTS + 8; // commitment + targets + length
        if bytes.len() < header_size {
            return Err(ZkpError::InvalidProof("Proof too short".into()));
        }

        // Extract commitment
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&bytes[0..32]);

        // Extract targets
        let mut targets = [0u64; NUM_COMPONENTS];
        for i in 0..NUM_COMPONENTS {
            let start = 32 + i * 8;
            let mut target_bytes = [0u8; 8];
            target_bytes.copy_from_slice(&bytes[start..start + 8]);
            targets[i] = u64::from_le_bytes(target_bytes);
        }

        // Extract proof length
        let len_start = 32 + 8 * NUM_COMPONENTS;
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&bytes[len_start..len_start + 8]);
        let proof_len = u64::from_le_bytes(len_bytes) as usize;

        let proof_start = header_size;
        if bytes.len() < proof_start + proof_len {
            return Err(ZkpError::InvalidProof("Proof data truncated".into()));
        }

        // Extract and parse proof
        let proof = Proof::from_bytes(&bytes[proof_start..proof_start + proof_len])
            .map_err(|e| ZkpError::InvalidProof(format!("{:?}", e)))?;

        Ok(Self {
            proof,
            commitment,
            targets,
        })
    }

    /// Get the size of the serialized proof in bytes
    pub fn size(&self) -> usize {
        32 + 8 * NUM_COMPONENTS + 8 + self.proof.to_bytes().len()
    }
}

/// Serializable proof representation for storage/transmission
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableProof {
    /// Commitment (hex encoded)
    pub commitment: String,
    /// Target values
    pub targets: Vec<u64>,
    /// Proof data (base64 encoded)
    pub proof_data: String,
}

impl From<&KVectorRangeProof> for SerializableProof {
    fn from(proof: &KVectorRangeProof) -> Self {
        Self {
            commitment: hex_encode(&proof.commitment),
            targets: proof.targets.to_vec(),
            proof_data: base64_encode(&proof.proof.to_bytes()),
        }
    }
}

impl TryFrom<&SerializableProof> for KVectorRangeProof {
    type Error = ZkpError;

    fn try_from(serializable: &SerializableProof) -> ZkpResult<Self> {
        let commitment =
            hex_decode(&serializable.commitment).map_err(ZkpError::SerializationError)?;

        if commitment.len() != 32 {
            return Err(ZkpError::InvalidProof("Invalid commitment length".into()));
        }

        let mut commitment_arr = [0u8; 32];
        commitment_arr.copy_from_slice(&commitment);

        if serializable.targets.len() != NUM_COMPONENTS {
            return Err(ZkpError::InvalidProof("Invalid targets length".into()));
        }

        let mut targets = [0u64; NUM_COMPONENTS];
        targets.copy_from_slice(&serializable.targets);

        let proof_bytes =
            base64_decode(&serializable.proof_data).map_err(ZkpError::SerializationError)?;

        let proof = Proof::from_bytes(&proof_bytes)
            .map_err(|e| ZkpError::InvalidProof(format!("{:?}", e)))?;

        Ok(Self {
            proof,
            commitment: commitment_arr,
            targets,
        })
    }
}

// Simple hex encoding
fn hex_encode(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(data.len() * 2);
    for byte in data {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0xf) as usize] as char);
    }
    result
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("Odd length hex string".into());
    }

    let mut result = Vec::with_capacity(s.len() / 2);
    let mut chars = s.chars();

    while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
        let high = a.to_digit(16).ok_or("Invalid hex character")?;
        let low = b.to_digit(16).ok_or("Invalid hex character")?;
        result.push((high << 4 | low) as u8);
    }

    Ok(result)
}

// Simple base64 encoding/decoding
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::new();
    let mut chunks = data.chunks_exact(3);

    for chunk in chunks.by_ref() {
        let n = (chunk[0] as u32) << 16 | (chunk[1] as u32) << 8 | (chunk[2] as u32);
        result.push(ALPHABET[(n >> 18 & 63) as usize] as char);
        result.push(ALPHABET[(n >> 12 & 63) as usize] as char);
        result.push(ALPHABET[(n >> 6 & 63) as usize] as char);
        result.push(ALPHABET[(n & 63) as usize] as char);
    }

    let remainder = chunks.remainder();
    if !remainder.is_empty() {
        let mut n = (remainder[0] as u32) << 16;
        if remainder.len() > 1 {
            n |= (remainder[1] as u32) << 8;
        }

        result.push(ALPHABET[(n >> 18 & 63) as usize] as char);
        result.push(ALPHABET[(n >> 12 & 63) as usize] as char);

        if remainder.len() > 1 {
            result.push(ALPHABET[(n >> 6 & 63) as usize] as char);
        } else {
            result.push('=');
        }
        result.push('=');
    }

    result
}

fn base64_decode(data: &str) -> Result<Vec<u8>, String> {
    const DECODE: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1,
        -1, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4,
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1,
        -1, -1, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,
    ];

    let data = data.trim_end_matches('=');
    let mut result = Vec::with_capacity(data.len() * 3 / 4);

    let mut buffer = 0u32;
    let mut bits = 0u32;

    for c in data.bytes() {
        if c >= 128 {
            return Err("Invalid base64 character".into());
        }
        let val = DECODE[c as usize];
        if val < 0 {
            return Err("Invalid base64 character".into());
        }

        buffer = buffer << 6 | val as u32;
        bits += 6;

        if bits >= 8 {
            bits -= 8;
            result.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_roundtrip() {
        let original = b"Hello, World!";
        let encoded = base64_encode(original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(original.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_hex_roundtrip() {
        let original = [0xde, 0xad, 0xbe, 0xef];
        let encoded = hex_encode(&original);
        let decoded = hex_decode(&encoded).unwrap();
        assert_eq!(original.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_serializable_proof_structure() {
        let commitment =
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
        let proof_data = "SGVsbG8="; // "Hello" in base64

        let serializable = SerializableProof {
            commitment: commitment.to_string(),
            targets: vec![5000; NUM_COMPONENTS],
            proof_data: proof_data.to_string(),
        };

        assert_eq!(serializable.commitment.len(), 64); // 32 bytes * 2 hex chars
        assert_eq!(serializable.targets.len(), NUM_COMPONENTS);
    }

    // ============================================================
    // End-to-End Proof Tests
    // ============================================================

    #[test]
    fn test_e2e_prove_verify_typical_values() {
        // Typical K-Vector values representing a moderately trusted node
        let witness = KVectorWitness {
            k_r: 0.75,   // Reputation
            k_a: 0.60,   // Activity
            k_i: 0.85,   // Integrity
            k_p: 0.70,   // Performance
            k_m: 0.50,   // Membership duration
            k_s: 0.40,   // Stake weight
            k_h: 0.80,   // Historical consistency
            k_topo: 0.65, // Network topology
        };

        // Generate proof
        let proof = KVectorRangeProof::prove(&witness)
            .expect("Proof generation should succeed");

        // Verify proof
        proof.verify().expect("Proof verification should succeed");

        // Check commitment matches
        assert_eq!(proof.commitment, witness.commitment());

        // Check targets match scaled values
        assert_eq!(proof.targets, witness.scaled_values());
    }

    #[test]
    fn test_e2e_prove_verify_minimum_values() {
        // Near-minimum values with sufficient diversity (M-05 requires variance >= 0.01)
        let witness = KVectorWitness {
            k_r: 0.01,
            k_a: 0.02,
            k_i: 0.03,
            k_p: 0.04,
            k_m: 0.05,
            k_s: 0.06,
            k_h: 0.07,
            k_topo: 0.08,
        };

        let proof = KVectorRangeProof::prove(&witness)
            .expect("Proof generation should succeed for near-minimum values");

        proof.verify().expect("Proof verification should succeed for near-minimum values");

        // Targets should be small scaled values
        assert!(proof.targets.iter().all(|&t| t <= 1000));
    }

    #[test]
    fn test_e2e_reject_all_zeros() {
        // All zeros is a degenerate case that causes trivial constraint polynomials
        let witness = KVectorWitness {
            k_r: 0.0,
            k_a: 0.0,
            k_i: 0.0,
            k_p: 0.0,
            k_m: 0.0,
            k_s: 0.0,
            k_h: 0.0,
            k_topo: 0.0,
        };

        let result = KVectorRangeProof::prove(&witness);
        assert!(result.is_err(), "Should reject all-zeros K-Vector");
    }

    #[test]
    fn test_e2e_prove_verify_maximum_values() {
        // Near-maximum values with diversity (M-05 requires variance >= 0.01)
        let witness = KVectorWitness {
            k_r: 0.92,
            k_a: 0.93,
            k_i: 0.94,
            k_p: 0.95,
            k_m: 0.96,
            k_s: 0.97,
            k_h: 0.98,
            k_topo: 0.99,
        };

        let proof = KVectorRangeProof::prove(&witness)
            .expect("Proof generation should succeed for near-max values");

        proof.verify().expect("Proof verification should succeed for near-max values");

        // All targets should be high values (> 9000)
        assert!(proof.targets.iter().all(|&t| t >= 9000));
    }

    #[test]
    fn test_e2e_prove_verify_mixed_extremes() {
        // Alternating zeros and ones
        let witness = KVectorWitness {
            k_r: 0.0,
            k_a: 1.0,
            k_i: 0.0,
            k_p: 1.0,
            k_m: 0.0,
            k_s: 1.0,
            k_h: 0.0,
            k_topo: 1.0,
        };

        let proof = KVectorRangeProof::prove(&witness)
            .expect("Proof generation should succeed");

        proof.verify().expect("Proof verification should succeed");
    }

    #[test]
    fn test_e2e_proof_serialization_roundtrip() {
        let witness = KVectorWitness {
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        let original_proof = KVectorRangeProof::prove(&witness)
            .expect("Proof generation should succeed");

        // Serialize to bytes
        let bytes = original_proof.to_bytes();

        // Deserialize from bytes
        let restored_proof = KVectorRangeProof::from_bytes(&bytes)
            .expect("Deserialization should succeed");

        // Verify restored proof
        restored_proof.verify().expect("Restored proof should verify");

        // Check fields match
        assert_eq!(original_proof.commitment, restored_proof.commitment);
        assert_eq!(original_proof.targets, restored_proof.targets);
    }

    #[test]
    fn test_e2e_serializable_proof_roundtrip() {
        let witness = KVectorWitness {
            k_r: 0.55,
            k_a: 0.65,
            k_i: 0.75,
            k_p: 0.45,
            k_m: 0.35,
            k_s: 0.25,
            k_h: 0.95,
            k_topo: 0.15,
        };

        let original_proof = KVectorRangeProof::prove(&witness)
            .expect("Proof generation should succeed");

        // Convert to serializable format
        let serializable = SerializableProof::from(&original_proof);

        // Convert back
        let restored_proof = KVectorRangeProof::try_from(&serializable)
            .expect("Conversion should succeed");

        // Verify restored proof
        restored_proof.verify().expect("Restored proof should verify");
    }

    #[test]
    fn test_e2e_reject_out_of_range_high() {
        // Value > 1.0 should be rejected
        let witness = KVectorWitness {
            k_r: 1.5, // Out of range!
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        let result = KVectorRangeProof::prove(&witness);
        assert!(result.is_err(), "Should reject k_r > 1.0");
    }

    #[test]
    fn test_e2e_reject_out_of_range_negative() {
        // Negative value should be rejected
        let witness = KVectorWitness {
            k_r: 0.8,
            k_a: -0.1, // Out of range!
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        let result = KVectorRangeProof::prove(&witness);
        assert!(result.is_err(), "Should reject k_a < 0.0");
    }

    #[test]
    fn test_e2e_tampered_commitment_rejected() {
        let witness = KVectorWitness {
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        let mut proof = KVectorRangeProof::prove(&witness)
            .expect("Proof generation should succeed");

        // Tamper with commitment
        proof.commitment[0] ^= 0xFF;

        // Verification should fail
        let result = proof.verify();
        assert!(result.is_err(), "Tampered commitment should be rejected");
    }

    #[test]
    fn test_e2e_tampered_targets_rejected() {
        let witness = KVectorWitness {
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        let mut proof = KVectorRangeProof::prove(&witness)
            .expect("Proof generation should succeed");

        // Tamper with targets
        proof.targets[0] = 9999; // Change from 8000

        // Verification should fail
        let result = proof.verify();
        assert!(result.is_err(), "Tampered targets should be rejected");
    }

    #[test]
    fn test_e2e_proof_size_reasonable() {
        let witness = KVectorWitness {
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        let proof = KVectorRangeProof::prove(&witness)
            .expect("Proof generation should succeed");

        let size = proof.size();

        // Proof should be reasonably sized (< 100KB for this simple circuit)
        assert!(size < 100_000, "Proof size {} bytes is too large", size);

        // But also not trivially small (> 1KB)
        assert!(size > 1_000, "Proof size {} bytes is suspiciously small", size);
    }

    #[test]
    fn test_e2e_different_witnesses_different_commitments() {
        // Diverse values satisfying M-05 variance requirements
        let witness1 = KVectorWitness {
            k_r: 0.45,
            k_a: 0.50,
            k_i: 0.55,
            k_p: 0.60,
            k_m: 0.65,
            k_s: 0.70,
            k_h: 0.75,
            k_topo: 0.80,
        };

        let witness2 = KVectorWitness {
            k_r: 0.46, // Different values
            k_a: 0.51,
            k_i: 0.56,
            k_p: 0.61,
            k_m: 0.66,
            k_s: 0.71,
            k_h: 0.76,
            k_topo: 0.81,
        };

        let proof1 = KVectorRangeProof::prove(&witness1).unwrap();
        let proof2 = KVectorRangeProof::prove(&witness2).unwrap();

        // Different witnesses should produce different commitments
        assert_ne!(proof1.commitment, proof2.commitment);
    }
}
