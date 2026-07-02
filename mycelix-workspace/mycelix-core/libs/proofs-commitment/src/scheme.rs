// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core commitment scheme traits and implementations

use sha3::{Digest, Sha3_256};

use crate::CommitmentHash;

/// Trait for commitment schemes
///
/// A commitment scheme allows binding to a value without revealing it,
/// then later proving the commitment corresponds to that value.
pub trait CommitmentScheme {
    /// The input type for this commitment scheme
    type Input: ?Sized;

    /// Compute a commitment to the input
    fn commit(&self, input: &Self::Input) -> CommitmentHash;

    /// Verify that input matches the commitment
    ///
    /// Uses constant-time comparison to prevent timing attacks.
    fn verify(&self, input: &Self::Input, commitment: &CommitmentHash) -> bool {
        let computed = self.commit(input);
        constant_time_eq(&computed, commitment)
    }
}

/// SHA3-256 based commitment scheme for raw bytes
#[derive(Clone, Debug, Default)]
pub struct Sha3Commitment;

impl CommitmentScheme for Sha3Commitment {
    type Input = [u8];

    fn commit(&self, input: &[u8]) -> CommitmentHash {
        let mut hasher = Sha3_256::new();
        hasher.update(input);
        hasher.finalize().into()
    }
}

impl Sha3Commitment {
    /// Create a new SHA3 commitment scheme
    pub fn new() -> Self {
        Self
    }
}

/// Builder for composite commitments over multiple fields
///
/// Useful for committing to complex structures with multiple components.
///
/// # Example
///
/// ```
/// use proofs_commitment::CompositeCommitment;
///
/// let hash = CompositeCommitment::new()
///     .field(b"user_id")
///     .field_u64(1234567890)
///     .field(b"score_data")
///     .finalize();
/// ```
#[derive(Clone, Debug)]
pub struct CompositeCommitment {
    hasher: Sha3_256,
}

impl Default for CompositeCommitment {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositeCommitment {
    /// Create a new composite commitment builder
    pub fn new() -> Self {
        Self {
            hasher: Sha3_256::new(),
        }
    }

    /// Add a byte slice field to the commitment
    pub fn field(mut self, data: &[u8]) -> Self {
        // Include length prefix to prevent concatenation attacks
        self.hasher.update((data.len() as u64).to_le_bytes());
        self.hasher.update(data);
        self
    }

    /// Add a u64 field to the commitment
    pub fn field_u64(mut self, value: u64) -> Self {
        self.hasher.update(value.to_le_bytes());
        self
    }

    /// Add a u32 field to the commitment
    pub fn field_u32(mut self, value: u32) -> Self {
        self.hasher.update(value.to_le_bytes());
        self
    }

    /// Add multiple u64 values to the commitment
    pub fn field_u64_slice(mut self, values: &[u64]) -> Self {
        // Include count for safety
        self.hasher.update((values.len() as u64).to_le_bytes());
        for &v in values {
            self.hasher.update(v.to_le_bytes());
        }
        self
    }

    /// Finalize and return the commitment hash
    pub fn finalize(self) -> CommitmentHash {
        self.hasher.finalize().into()
    }

    /// Finalize and verify against an expected commitment
    ///
    /// Uses constant-time comparison.
    pub fn verify(self, expected: &CommitmentHash) -> bool {
        let computed = self.finalize();
        constant_time_eq(&computed, expected)
    }
}

/// Constant-time byte array comparison
///
/// Prevents timing attacks by always comparing all bytes regardless of
/// where differences occur.
pub fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha3_commitment_deterministic() {
        let scheme = Sha3Commitment::new();
        let data = b"test data";

        let hash1 = scheme.commit(data);
        let hash2 = scheme.commit(data);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_sha3_commitment_different_inputs() {
        let scheme = Sha3Commitment::new();

        let hash1 = scheme.commit(b"data1");
        let hash2 = scheme.commit(b"data2");

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_composite_length_prefix_prevents_attacks() {
        // Without length prefix, "ab" + "cd" == "a" + "bcd"
        // With length prefix, they should differ
        let hash1 = CompositeCommitment::new()
            .field(b"ab")
            .field(b"cd")
            .finalize();

        let hash2 = CompositeCommitment::new()
            .field(b"a")
            .field(b"bcd")
            .finalize();

        assert_ne!(
            hash1, hash2,
            "Length prefix should prevent concatenation attacks"
        );
    }

    #[test]
    fn test_constant_time_eq() {
        let a = [1u8; 32];
        let b = [1u8; 32];
        let c = [2u8; 32];

        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
    }

    #[test]
    fn test_composite_verify() {
        let hash = CompositeCommitment::new()
            .field(b"test")
            .field_u64(42)
            .finalize();

        // Same construction should verify
        let valid = CompositeCommitment::new()
            .field(b"test")
            .field_u64(42)
            .verify(&hash);

        assert!(valid);

        // Different construction should fail
        let invalid = CompositeCommitment::new()
            .field(b"test")
            .field_u64(43)
            .verify(&hash);

        assert!(!invalid);
    }
}
