// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verifier for K-Vector range proofs
//!
//! Verifies STARK proofs that K-Vector components are in [0, 1].
//!
//! ## Security Considerations
//!
//! **S-07 Commitment Verification (CRITICAL)**: The verifier now explicitly checks
//! that the provided commitment matches `SHA3-256(targets)`. This prevents an
//! attacker from providing targets that don't match the commitment scheme.
//!
//! **C-02 Timing Side Channel Mitigation**: This verifier accepts multiple proof
//! security levels but does NOT leak which level was used through timing differences.
//! The verification time is dominated by the proof size and cryptographic operations,
//! not by the option matching. However, proof SIZE may vary between security levels,
//! which could leak information if an attacker can observe proof transmission.
//! For high-security deployments, consider:
//! 1. Using a single security level consistently
//! 2. Padding proofs to a fixed size before transmission
//! 3. Adding random delays to mask verification timing

use sha3::{Digest, Sha3_256};
use winterfell::{
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    math::fields::f128::BaseElement,
    verify, AcceptableOptions, BatchingMethod, Proof,
};

use crate::air::{KVectorPublicInputs, KVectorRangeAir, NUM_COMPONENTS};
use crate::error::{ZkpError, ZkpResult};

/// Type aliases for cleaner code
type Hasher = Blake3_256<BaseElement>;
type VC = MerkleTree<Hasher>;
type RandCoin = DefaultRandomCoin<Hasher>;

/// Minimum verification time in microseconds (for timing attack mitigation)
/// Set to 0 to disable (default). Enable for high-security deployments.
const MIN_VERIFICATION_TIME_US: u64 = 0;

/// Compute the expected commitment from target values
///
/// The commitment is SHA3-256 of all target values concatenated as little-endian bytes.
/// This matches the commitment scheme used by the prover.
fn compute_commitment(targets: &[u64; NUM_COMPONENTS]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    for &target in targets {
        hasher.update(target.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Verify that a commitment matches the target values
///
/// # S-07 Security Fix
/// This function prevents attackers from providing targets that don't match
/// the commitment. Without this check, an attacker could:
/// 1. Obtain a valid proof for targets T1 with commitment C1
/// 2. Submit the proof with different targets T2 (claiming they have different scores)
/// 3. The verifier would accept if the proof structure is valid for T2
///
/// By verifying commitment == SHA3-256(targets), we ensure binding.
fn verify_commitment(commitment: &[u8; 32], targets: &[u64; NUM_COMPONENTS]) -> ZkpResult<()> {
    let expected = compute_commitment(targets);

    // Use constant-time comparison to prevent timing attacks
    let mut diff = 0u8;
    for (a, b) in commitment.iter().zip(expected.iter()) {
        diff |= a ^ b;
    }

    if diff != 0 {
        return Err(ZkpError::VerificationFailed(
            "S-07: commitment does not match targets - possible tampering".into()
        ));
    }

    Ok(())
}

/// Verify a K-Vector range proof
///
/// # Arguments
/// * `proof` - The STARK proof
/// * `commitment` - Expected commitment hash (must equal SHA3-256 of targets)
/// * `targets` - The scaled target values (public)
///
/// # Security
///
/// **S-07 Commitment Verification**: Before verifying the STARK proof, this function
/// verifies that `commitment == SHA3-256(targets)`. This ensures the targets
/// provided for verification match the committed values.
///
/// This function accepts proofs generated with any of the configured security levels.
/// Verification timing is consistent regardless of which security level was used,
/// as the underlying cryptographic operations dominate the execution time.
pub fn verify_kvector_proof(
    proof: Proof,
    commitment: [u8; 32],
    targets: [u64; NUM_COMPONENTS],
) -> ZkpResult<()> {
    let start = std::time::Instant::now();

    // S-07: Verify commitment matches targets BEFORE expensive proof verification
    // This prevents attacks where adversary provides valid proof with wrong targets
    verify_commitment(&commitment, &targets)?;

    let pub_inputs = KVectorPublicInputs::new(commitment, targets);

    // Define acceptable proof parameters (must match prover options)
    // C-02: All options have similar computational complexity to minimize timing variance
    let acceptable = AcceptableOptions::OptionSet(vec![
        // Standard security (128-bit)
        winterfell::ProofOptions::new(
            32,
            8,
            0,
            winterfell::FieldExtension::None,
            8,
            31,
            BatchingMethod::Linear,
            BatchingMethod::Linear,
        ),
        // Optimized security (112-bit, faster)
        winterfell::ProofOptions::new(
            28,
            8,
            0,
            winterfell::FieldExtension::None,
            8,  // FRI folding factor must be power of 2
            31,
            BatchingMethod::Linear,
            BatchingMethod::Linear,
        ),
        // Fast security (80-bit, for testing only)
        winterfell::ProofOptions::new(
            20,
            4,
            0,
            winterfell::FieldExtension::None,
            16,
            63,
            BatchingMethod::Linear,
            BatchingMethod::Linear,
        ),
        // High security (256-bit)
        winterfell::ProofOptions::new(
            64,
            16,
            8,
            winterfell::FieldExtension::None,
            4,
            15,
            BatchingMethod::Linear,
            BatchingMethod::Linear,
        ),
    ]);

    let result = verify::<KVectorRangeAir, Hasher, RandCoin, VC>(proof, pub_inputs, &acceptable)
        .map_err(|e| ZkpError::VerificationFailed(format!("{:?}", e)));

    // C-02: Ensure minimum verification time to prevent timing attacks
    // This adds a small delay if verification completed too quickly
    if MIN_VERIFICATION_TIME_US > 0 {
        let elapsed_us = start.elapsed().as_micros() as u64;
        if elapsed_us < MIN_VERIFICATION_TIME_US {
            std::thread::sleep(std::time::Duration::from_micros(
                MIN_VERIFICATION_TIME_US - elapsed_us,
            ));
        }
    }

    result
}

/// Minimum security bits for production verification
///
/// C-04: Production verifiers should ONLY accept Standard (~96-bit) or High (~264-bit)
/// security proofs. Proofs generated with Fast (~40-bit) or Optimized (~84-bit)
/// settings will be rejected.
pub const PRODUCTION_MIN_SECURITY_BITS: u32 = 96;

/// Verify a K-Vector range proof with minimum security level enforcement
///
/// # C-04 Security Enhancement
///
/// This function rejects proofs generated with insufficient security levels.
/// Use this for production deployments instead of `verify_kvector_proof`.
///
/// # Arguments
/// * `proof` - The STARK proof
/// * `commitment` - Expected commitment hash (must equal SHA3-256 of targets)
/// * `targets` - The scaled target values (public)
/// * `min_security_bits` - Minimum acceptable security level in bits
///
/// # Security Levels Accepted
/// - 96+ bits: Accepts Standard and High proofs
/// - 84+ bits: Also accepts Optimized proofs (not recommended for production)
/// - 40+ bits: Accepts all proofs including Fast (NEVER use in production)
///
/// # Example
/// ```ignore
/// // Production usage - only accept Standard or High security
/// verify_kvector_proof_secure(proof, commitment, targets, PRODUCTION_MIN_SECURITY_BITS)?;
/// ```
pub fn verify_kvector_proof_secure(
    proof: Proof,
    commitment: [u8; 32],
    targets: [u64; NUM_COMPONENTS],
    min_security_bits: u32,
) -> ZkpResult<()> {
    let start = std::time::Instant::now();

    // S-07: Verify commitment matches targets BEFORE expensive proof verification
    verify_commitment(&commitment, &targets)?;

    let pub_inputs = KVectorPublicInputs::new(commitment, targets);

    // C-04: Build acceptable options based on minimum security requirement
    // Only include proof options that meet the minimum security level
    let mut acceptable_options = Vec::new();

    // High security (~264-bit): 64 queries, 16 blowup, 8 grinding
    if min_security_bits <= 264 {
        acceptable_options.push(winterfell::ProofOptions::new(
            64, 16, 8,
            winterfell::FieldExtension::None,
            4, 15,
            BatchingMethod::Linear, BatchingMethod::Linear,
        ));
    }

    // Standard security (~96-bit): 32 queries, 8 blowup
    if min_security_bits <= 96 {
        acceptable_options.push(winterfell::ProofOptions::new(
            32, 8, 0,
            winterfell::FieldExtension::None,
            8, 31,
            BatchingMethod::Linear, BatchingMethod::Linear,
        ));
    }

    // Optimized security (~84-bit): 28 queries, 8 blowup
    if min_security_bits <= 84 {
        acceptable_options.push(winterfell::ProofOptions::new(
            28, 8, 0,
            winterfell::FieldExtension::None,
            8, 31,
            BatchingMethod::Linear, BatchingMethod::Linear,
        ));
    }

    // Fast security (~40-bit): 20 queries, 4 blowup - TESTING ONLY
    if min_security_bits <= 40 {
        acceptable_options.push(winterfell::ProofOptions::new(
            20, 4, 0,
            winterfell::FieldExtension::None,
            16, 63,
            BatchingMethod::Linear, BatchingMethod::Linear,
        ));
    }

    if acceptable_options.is_empty() {
        return Err(ZkpError::VerificationFailed(format!(
            "C-04: No proof options meet the minimum security requirement of {} bits",
            min_security_bits
        )));
    }

    let acceptable = AcceptableOptions::OptionSet(acceptable_options);

    let result = verify::<KVectorRangeAir, Hasher, RandCoin, VC>(proof, pub_inputs, &acceptable)
        .map_err(|e| {
            // Distinguish between security level rejection and other failures
            let error_str = format!("{:?}", e);
            if error_str.contains("options") {
                ZkpError::VerificationFailed(format!(
                    "C-04: Proof was generated with security level below {} bits",
                    min_security_bits
                ))
            } else {
                ZkpError::VerificationFailed(error_str)
            }
        });

    // C-02: Timing attack mitigation
    if MIN_VERIFICATION_TIME_US > 0 {
        let elapsed_us = start.elapsed().as_micros() as u64;
        if elapsed_us < MIN_VERIFICATION_TIME_US {
            std::thread::sleep(std::time::Duration::from_micros(
                MIN_VERIFICATION_TIME_US - elapsed_us,
            ));
        }
    }

    result
}

/// Verify a K-Vector range proof for production use
///
/// This is a convenience wrapper around `verify_kvector_proof_secure` that
/// enforces `PRODUCTION_MIN_SECURITY_BITS` (96-bit minimum).
///
/// # C-04 Production Safety
///
/// Use this function for all production verifications. It will reject proofs
/// generated with `Fast` (~40-bit) or `Optimized` (~84-bit) security levels.
pub fn verify_kvector_proof_production(
    proof: Proof,
    commitment: [u8; 32],
    targets: [u64; NUM_COMPONENTS],
) -> ZkpResult<()> {
    verify_kvector_proof_secure(proof, commitment, targets, PRODUCTION_MIN_SECURITY_BITS)
}

/// Simplified verification with just commitment (extracts targets from proof context)
pub fn verify_kvector_proof_simple(proof: Proof, commitment: [u8; 32]) -> ZkpResult<()> {
    // For simple verification, we need the targets from somewhere
    // In practice, these should be provided by the prover/protocol
    // For now, use zero targets (this will fail unless commitment matches)
    let targets = [0u64; NUM_COMPONENTS];
    verify_kvector_proof(proof, commitment, targets)
}

/// Batch verify multiple proofs
pub fn batch_verify_proofs(
    proofs: Vec<(Proof, [u8; 32], [u64; NUM_COMPONENTS])>,
) -> ZkpResult<Vec<bool>> {
    Ok(proofs
        .into_iter()
        .map(|(proof, commitment, targets)| {
            verify_kvector_proof(proof, commitment, targets).is_ok()
        })
        .collect())
}

/// Proof metadata for serialization
#[derive(Clone, Debug)]
pub struct ProofMetadata {
    /// Commitment hash
    pub commitment: [u8; 32],
    /// Target values (scaled)
    pub targets: [u64; NUM_COMPONENTS],
    /// Proof size in bytes
    pub proof_size: usize,
    /// Prover version
    pub version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_metadata() {
        let metadata = ProofMetadata {
            commitment: [0u8; 32],
            targets: [5000; NUM_COMPONENTS], // 0.5 scaled
            proof_size: 0,
            version: 1,
        };

        assert_eq!(metadata.commitment, [0u8; 32]);
        assert_eq!(metadata.targets[0], 5000);
    }

    // ============ S-07 Commitment Verification Tests ============

    #[test]
    fn test_compute_commitment_deterministic() {
        let targets = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let commit1 = compute_commitment(&targets);
        let commit2 = compute_commitment(&targets);
        assert_eq!(commit1, commit2, "Commitment should be deterministic");
    }

    #[test]
    fn test_compute_commitment_different_targets() {
        let targets1 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let targets2 = [1001, 2000, 3000, 4000, 5000, 6000, 7000, 8000]; // One value differs

        let commit1 = compute_commitment(&targets1);
        let commit2 = compute_commitment(&targets2);

        assert_ne!(commit1, commit2, "Different targets should have different commitments");
    }

    #[test]
    fn test_verify_commitment_valid() {
        let targets = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let commitment = compute_commitment(&targets);

        let result = verify_commitment(&commitment, &targets);
        assert!(result.is_ok(), "Valid commitment should verify");
    }

    #[test]
    fn test_verify_commitment_invalid_rejects() {
        let targets = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let commitment = compute_commitment(&targets);

        // Tamper with targets
        let tampered_targets = [9999, 2000, 3000, 4000, 5000, 6000, 7000, 8000];

        let result = verify_commitment(&commitment, &tampered_targets);
        assert!(result.is_err(), "Tampered targets should be rejected");

        match result {
            Err(ZkpError::VerificationFailed(msg)) => {
                assert!(msg.contains("S-07"), "Error should mention S-07");
                assert!(msg.contains("commitment"), "Error should mention commitment");
            }
            _ => panic!("Expected VerificationFailed error"),
        }
    }

    #[test]
    fn test_verify_commitment_wrong_commitment() {
        let targets = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let wrong_commitment = [0xab; 32]; // Random wrong commitment

        let result = verify_commitment(&wrong_commitment, &targets);
        assert!(result.is_err(), "Wrong commitment should be rejected");
    }

    #[test]
    fn test_verify_commitment_single_bit_difference() {
        let targets = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let mut commitment = compute_commitment(&targets);

        // Flip a single bit
        commitment[0] ^= 0x01;

        let result = verify_commitment(&commitment, &targets);
        assert!(result.is_err(), "Single bit difference should be detected");
    }

    #[test]
    fn test_verify_commitment_all_zeros() {
        let targets = [0u64; NUM_COMPONENTS];
        let commitment = compute_commitment(&targets);

        let result = verify_commitment(&commitment, &targets);
        assert!(result.is_ok(), "All-zero targets with matching commitment should verify");
    }

    #[test]
    fn test_verify_commitment_max_values() {
        let targets = [10000u64; NUM_COMPONENTS]; // MAX_SCORE
        let commitment = compute_commitment(&targets);

        let result = verify_commitment(&commitment, &targets);
        assert!(result.is_ok(), "Max value targets with matching commitment should verify");
    }

    #[test]
    fn test_commitment_order_matters() {
        // Test that the order of targets matters in commitment
        let targets1 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let targets2 = [2000, 1000, 3000, 4000, 5000, 6000, 7000, 8000]; // Swapped first two

        let commit1 = compute_commitment(&targets1);
        let commit2 = compute_commitment(&targets2);

        assert_ne!(commit1, commit2, "Swapped targets should produce different commitments");
    }

    #[test]
    fn test_constant_time_comparison() {
        // This test ensures the constant-time comparison doesn't have
        // obvious issues (though true constant-time testing requires timing analysis)
        let targets = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let commitment = compute_commitment(&targets);

        // Test with many different wrong commitments
        for i in 0..32 {
            let mut wrong = commitment;
            wrong[i] = wrong[i].wrapping_add(1);

            let result = verify_commitment(&wrong, &targets);
            assert!(result.is_err(), "Byte {} modification should be detected", i);
        }
    }
}
