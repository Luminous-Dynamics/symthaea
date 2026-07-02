// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared Commitment Schemes for Mycelix ZK Proof Systems
//!
//! This library provides reusable commitment schemes used across various ZK proof
//! circuits in the Mycelix ecosystem, including:
//!
//! - K-Vector range proofs
//! - Gradient integrity proofs
//! - Identity assurance proofs
//! - Vote eligibility proofs
//!
//! # Design Principles
//!
//! 1. **Consistency**: All circuits use the same commitment algorithm
//! 2. **Security**: SHA3-256 with constant-time verification
//! 3. **Composability**: Support for multi-field commitments
//! 4. **Efficiency**: Fixed-point scaling for floating-point values
//!
//! # Example
//!
//! ```
//! use proofs_commitment::{FixedPointCommitment, CommitmentScheme};
//!
//! // Create a commitment to 8 values (like K-Vector)
//! let values = [0.8f32, 0.7, 0.9, 0.6, 0.5, 0.4, 0.85, 0.75];
//! let commitment = FixedPointCommitment::default();
//!
//! let hash = commitment.commit_f32_slice(&values);
//!
//! // Verify the commitment
//! assert!(commitment.verify_f32_slice(&values, &hash));
//! ```

mod error;
mod fixed_point;
mod scheme;

pub use error::{CommitmentError, CommitmentResult};
pub use fixed_point::{FixedPointCommitment, ScalingConfig, KVECTOR_SCALE_FACTOR};
pub use scheme::{CommitmentScheme, CompositeCommitment, Sha3Commitment};

/// Standard commitment hash type (32 bytes = 256 bits)
pub type CommitmentHash = [u8; 32];

/// Number of K-Vector components (for compatibility)
pub const KVECTOR_COMPONENTS: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_commitment() {
        let values = [0.8f32, 0.7, 0.9, 0.6, 0.5, 0.4, 0.85, 0.75];
        let commitment = FixedPointCommitment::default();

        let hash = commitment.commit_f32_slice(&values);
        assert!(commitment.verify_f32_slice(&values, &hash));

        // Tampered values should fail
        let tampered = [0.81f32, 0.7, 0.9, 0.6, 0.5, 0.4, 0.85, 0.75];
        assert!(!commitment.verify_f32_slice(&tampered, &hash));
    }

    #[test]
    fn test_sha3_commitment() {
        let data = b"test data for commitment";
        let commitment = Sha3Commitment::default();

        let hash = commitment.commit(data);
        assert!(commitment.verify(data, &hash));

        let tampered = b"tampered data for commitment";
        assert!(!commitment.verify(tampered, &hash));
    }

    #[test]
    fn test_composite_commitment() {
        let commitment = CompositeCommitment::new();

        // Commit multiple fields
        let hash = commitment
            .field(b"field1_data")
            .field(b"field2_data")
            .field_u64(12345)
            .finalize();

        // Same inputs should produce same hash
        let hash2 = commitment
            .field(b"field1_data")
            .field(b"field2_data")
            .field_u64(12345)
            .finalize();

        assert_eq!(hash, hash2);

        // Different inputs should produce different hash
        let hash3 = commitment
            .field(b"field1_data")
            .field(b"field2_DIFFERENT")
            .field_u64(12345)
            .finalize();

        assert_ne!(hash, hash3);
    }
}
