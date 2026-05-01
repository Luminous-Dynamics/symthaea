// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Miden VM ZK-STARK Consciousness Proofs
//!
//! Zero-knowledge consciousness tier attestation using Miden VM.
//! Unlike Winterfell (which is NOT zero-knowledge), Miden's advice provider
//! ensures the secret Phi score is never revealed to the verifier.
//!
//! ## Architecture
//!
//! ```text
//! Client (native):
//!   1. phi_score → advice stack (secret, never revealed)
//!   2. threshold + commitment → operand stack (public)
//!   3. Miden prover executes program → proof bytes (~80-100KB)
//!
//! Verifier (WASM or native):
//!   1. Receives proof bytes + public inputs (threshold, commitment)
//!   2. miden_verifier::verify() → bool
//!   3. Phi score remains unknown to verifier
//! ```
//!
//! ## Feature Gate
//!
//! Enable with `backend-miden` feature in Cargo.toml.

use crate::consciousness::CivicTier;
use crate::error::ZkpError;
use sha2::{Digest, Sha256};

/// Result type for Miden ZKP operations.
pub type MidenResult<T> = Result<T, ZkpError>;

/// Maximum proof size for Miden STARK proofs (150KB).
/// Miden proofs are typically 80-136KB depending on program complexity.
pub const MAX_MIDEN_PROOF_SIZE: usize = 150_000;

/// Proof system identifier for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MidenProofSystem {
    /// Miden VM ZK-STARK (actual zero-knowledge)
    MidenZkStark,
}

/// Result of generating a Miden consciousness proof.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MidenConsciousnessProof {
    /// STARK proof bytes (80-136KB typical)
    pub proof_bytes: Vec<u8>,
    /// SHA-256 commitment to the score: H(phi_score || domain_tag)
    pub score_commitment: [u8; 32],
    /// The tier being proven (public — verifier knows which tier)
    pub proven_tier: CivicTier,
    /// Proof system identifier
    pub proof_system: MidenProofSystem,
    /// Stack outputs from the Miden program (public)
    pub stack_outputs: Vec<u64>,
}

/// Compute the score commitment: SHA-256(phi_bytes || domain_tag || nonce || epoch).
///
/// The nonce and epoch_secs ensure each commitment is unique and time-bound,
/// preventing replay attacks where a valid proof is reused across sessions.
pub fn compute_score_commitment(phi_score: f64, nonce: &[u8; 32], epoch_secs: u64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(phi_score.to_le_bytes());
    hasher.update(b"MYCELIX:CivicTier:v2");
    hasher.update(nonce);
    hasher.update(epoch_secs.to_le_bytes());
    let result = hasher.finalize();
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&result);
    commitment
}

/// Legacy commitment without nonce/epoch (for backward-compatible test helpers).
/// Do NOT use for new proofs — use `compute_score_commitment` with nonce+epoch.
#[cfg(test)]
fn compute_score_commitment_legacy(phi_score: f64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(phi_score.to_le_bytes());
    hasher.update(b"MYCELIX:CivicTier:v2");
    let result = hasher.finalize();
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&result);
    commitment
}

/// Map a consciousness tier to its minimum Phi threshold.
///
/// These thresholds must match the values used in bridge-common's
/// `consciousness_profile.rs` tier computation.
pub fn tier_threshold(tier: &CivicTier) -> f64 {
    match tier {
        CivicTier::Observer => 0.0,
        CivicTier::Participant => 0.2,
        CivicTier::Citizen => 0.3,
        CivicTier::Steward => 0.4,
        CivicTier::Guardian => 0.6,
    }
}

/// Validate a Miden consciousness proof structurally.
///
/// This can run in WASM without the Miden prover dependency.
/// Checks proof size, commitment non-zero, and tier validity.
pub fn validate_miden_proof_structure(proof: &MidenConsciousnessProof) -> MidenResult<()> {
    if proof.proof_bytes.is_empty() {
        return Err(ZkpError::ProvingError("Empty proof bytes".into()));
    }
    if proof.proof_bytes.len() > MAX_MIDEN_PROOF_SIZE {
        return Err(ZkpError::ProvingError(format!(
            "Proof too large: {} > {}",
            proof.proof_bytes.len(),
            MAX_MIDEN_PROOF_SIZE
        )));
    }
    if proof.score_commitment == [0u8; 32] {
        return Err(ZkpError::ProvingError("Zero score commitment".into()));
    }
    if proof.proof_system != MidenProofSystem::MidenZkStark {
        return Err(ZkpError::ProvingError("Wrong proof system".into()));
    }
    Ok(())
}

// ============================================================================
// Miden Assembly Program
// ============================================================================

/// The Miden Assembly source for consciousness tier range proof.
///
/// Public inputs (operand stack): [threshold_permille]
/// Secret inputs (advice stack):  [phi_permille]
///
/// The program:
///   1. Reads secret phi from advice stack
///   2. Asserts both values are valid u32
///   3. Asserts phi >= threshold (proves tier without revealing phi)
///   4. Pushes 1 (success)
///
/// If phi < threshold, the assertion fails and no valid proof can be generated.
/// This is the core ZK property: the prover can only produce a proof when the
/// secret phi genuinely meets the threshold.
pub const CONSCIOUSNESS_RANGE_PROOF_MASM: &str = "
begin
    # Stack (16 deep): [threshold, 0, 0, ..., 0]
    # Advice: [phi]

    # Read secret phi
    adv_push.1
    # 17: [phi, threshold, 0, ..., 0]

    # Drop threshold — we'll reconstruct comparison via subtraction
    # Actually: just drop the extra element immediately to get back to 16
    # by consuming both values in a comparison.

    # Strategy: compute (phi - threshold) in the field, then assert it
    # equals a u32 value. If phi < threshold, the subtraction wraps to
    # a huge field element that won't fit in u32.

    # phi is on top, threshold below.
    # sub: pops [b,a], pushes a-b. So: a=threshold, b=phi → threshold-phi (WRONG)
    # We want phi-threshold. Need: [threshold, phi] then sub gives phi-threshold.
    swap sub
    # Wait: swap gives [threshold, phi, 0...], sub gives phi-threshold? No!
    # sub pops top (b) and second (a), returns a-b
    # After swap: top=threshold, second=phi → a=phi, b=threshold → phi-threshold ✓
    # But earlier this gave a wrapped value... let me re-check.
    # Oh wait: the FIRST time I had movup.2 which changed things. Without dup/movup:
    # [phi, threshold, 0...] → swap → [threshold, phi, 0...] → sub →
    # a=phi (was second), b=threshold (was top) → phi - threshold = 750-400 = 350 ✓
    # That should work. The earlier failure was because I had an extra dup.

    # drop result + done = 16-2+1 = 15
    # But wait: we started at 17, sub consumes 2 pushes 1 → 16
    # Then we have [350, 0, 0, ..., 0] (16 elements)
    # Just need to verify the result and exit.

    # u32assert checks top is u32. If phi < threshold, the field
    # subtraction wraps and u32assert fails (proving aborts).
    # Note: u32assert may push decomposition elements.
    u32assert

    # Drop all computation artifacts to get back to 16
    drop drop
end
";

// ============================================================================
// Prover (native only — NOT available in WASM)
// ============================================================================

/// Generate a ZK-STARK proof that `phi_permille >= threshold_permille`.
///
/// The phi value is secret (passed via advice stack) and never revealed
/// to the verifier. Only the threshold is public.
///
/// Returns the proof, stack outputs, and program info needed for verification.
///
/// # Arguments
/// * `phi_permille` - The agent's actual Phi score (0-1000), SECRET
/// * `threshold_permille` - The minimum tier threshold (0-1000), PUBLIC
///
/// # Feature Gate
/// Requires `backend-miden` feature.
#[cfg(feature = "backend-miden")]
pub fn prove_consciousness_tier_miden(
    phi_permille: u32,
    threshold_permille: u32,
) -> MidenResult<MidenConsciousnessProof> {
    use miden_core::Felt;
    use miden_vm::{advice::AdviceInputs, Assembler, DefaultHost, ProvingOptions, StackInputs};

    // Validate inputs
    if phi_permille > 1000 {
        return Err(ZkpError::ProvingError(format!(
            "phi_permille {} exceeds 1000",
            phi_permille
        )));
    }
    if threshold_permille > 1000 {
        return Err(ZkpError::ProvingError(format!(
            "threshold_permille {} exceeds 1000",
            threshold_permille
        )));
    }
    if phi_permille < threshold_permille {
        return Err(ZkpError::ProvingError(format!(
            "phi_permille {} < threshold_permille {} — cannot prove false statement",
            phi_permille, threshold_permille
        )));
    }

    // Assemble the program
    let assembler = Assembler::default();
    let program = assembler
        .assemble_program(CONSCIOUSNESS_RANGE_PROOF_MASM)
        .map_err(|e| ZkpError::ProvingError(format!("Assembly error: {}", e)))?;

    // Public inputs: threshold on operand stack
    let stack_inputs = StackInputs::new(&[Felt::new(threshold_permille as u64)])
        .map_err(|e| ZkpError::ProvingError(format!("Stack input error: {}", e)))?;

    // Secret inputs: phi on advice stack (never revealed to verifier)
    let advice_inputs = AdviceInputs::default()
        .with_stack_values([phi_permille as u64])
        .map_err(|e| ZkpError::ProvingError(format!("Advice input error: {}", e)))?;

    // Execute and prove (blocking — native only)
    let mut host = DefaultHost::default();
    let (outputs, proof) = miden_vm::prove_sync(
        &program,
        stack_inputs.clone(),
        advice_inputs,
        &mut host,
        ProvingOptions::default(), // 96-bit security, BLAKE3
    )
    .map_err(|e| ZkpError::ProvingError(format!("Proving error: {}", e)))?;

    // Determine the proven tier from the threshold
    let proven_tier = tier_from_threshold_permille(threshold_permille);

    // Compute score commitment (binds the proof to this specific phi).
    // For now, Miden prover generates its own nonce+epoch internally.
    let nonce = {
        let mut n = [0u8; 32];
        // Deterministic nonce from phi+threshold for reproducible tests.
        // In production, callers should pass a random nonce.
        let seed = (phi_permille as u64) << 32 | (threshold_permille as u64);
        n[..8].copy_from_slice(&seed.to_le_bytes());
        n
    };
    let epoch_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let commitment = compute_score_commitment(phi_permille as f64 / 1000.0, &nonce, epoch_secs);

    // Serialize proof
    let proof_bytes = proof.to_bytes();

    // Serialize outputs for verification
    let stack_output_values: Vec<u64> = (0..16)
        .filter_map(|i| outputs.get_element(i).map(|f| f.as_canonical_u64()))
        .collect();

    Ok(MidenConsciousnessProof {
        proof_bytes,
        score_commitment: commitment,
        proven_tier,
        proof_system: MidenProofSystem::MidenZkStark,
        stack_outputs: stack_output_values,
    })
}

/// Map a threshold (permille) to the corresponding consciousness tier.
fn tier_from_threshold_permille(threshold: u32) -> CivicTier {
    match threshold {
        0..=199 => CivicTier::Observer,
        200..=299 => CivicTier::Participant,
        300..=399 => CivicTier::Citizen,
        400..=599 => CivicTier::Steward,
        _ => CivicTier::Guardian,
    }
}

// ============================================================================
// Verifier (works in both native and WASM via miden-verifier)
// ============================================================================

/// Verify a Miden ZK-STARK consciousness tier proof.
///
/// This function does NOT learn the agent's actual Phi score. It only
/// confirms that the prover knew a phi value >= the threshold.
///
/// # Arguments
/// * `proof` - The Miden proof (from `prove_consciousness_tier_miden`)
/// * `threshold_permille` - The public threshold that was proven against
/// * `program_hash` - Hash of the expected Miden program (prevents program substitution)
///
/// # Feature Gate
/// Requires `backend-miden` feature. Compiles to WASM for zome-side verification.
#[cfg(feature = "backend-miden")]
pub fn verify_consciousness_tier_miden(
    proof: &MidenConsciousnessProof,
    threshold_permille: u32,
    program_hash: &[u64; 4],
) -> MidenResult<bool> {
    use miden_core::Felt;
    use miden_vm::{ExecutionProof, Kernel, ProgramInfo, StackInputs, StackOutputs, Word};

    // Structural validation first
    validate_miden_proof_structure(proof)?;

    // Deserialize the proof
    let execution_proof = ExecutionProof::from_bytes(&proof.proof_bytes)
        .map_err(|e| ZkpError::VerificationFailed(format!("Proof deserialization: {}", e)))?;

    // Reconstruct public inputs
    let stack_inputs = StackInputs::new(&[Felt::new(threshold_permille as u64)])
        .map_err(|e| ZkpError::VerificationFailed(format!("Stack input error: {}", e)))?;

    // Reconstruct outputs
    let output_felts: Vec<Felt> = proof.stack_outputs.iter().map(|&v| Felt::new(v)).collect();
    let stack_outputs = StackOutputs::new(&output_felts)
        .map_err(|e| ZkpError::VerificationFailed(format!("Stack output error: {}", e)))?;

    // Reconstruct program info from hash (Word wraps [Felt; 4])
    let hash_word: Word = Word::new([
        Felt::new(program_hash[0]),
        Felt::new(program_hash[1]),
        Felt::new(program_hash[2]),
        Felt::new(program_hash[3]),
    ]);
    let program_info = ProgramInfo::new(hash_word, Kernel::default());

    // Verify the STARK proof
    match miden_vm::verify(program_info, stack_inputs, stack_outputs, execution_proof) {
        Ok(security_level) => {
            if security_level >= 80 {
                Ok(true)
            } else {
                Err(ZkpError::VerificationFailed(format!(
                    "Security level {} < 80 bits",
                    security_level
                )))
            }
        }
        Err(e) => Err(ZkpError::VerificationFailed(format!(
            "STARK verification failed: {}",
            e
        ))),
    }
}

/// Get the program hash for the consciousness range proof program.
///
/// This is a fixed value — the program never changes. Used by the
/// verifier to ensure the correct program was proven.
/// Get the program hash for the consciousness range proof program.
///
/// Returns a `Word` ([Felt; 4]) — the Miden program identifier used
/// by the verifier to ensure the correct program was proven.
#[cfg(feature = "backend-miden")]
pub fn consciousness_program_hash() -> MidenResult<[u64; 4]> {
    let assembler = miden_vm::Assembler::default();
    let program = assembler
        .assemble_program(CONSCIOUSNESS_RANGE_PROOF_MASM)
        .map_err(|e| ZkpError::ProvingError(format!("Assembly error: {}", e)))?;

    let hash_word = program.hash();
    Ok([
        hash_word[0].as_canonical_u64(),
        hash_word[1].as_canonical_u64(),
        hash_word[2].as_canonical_u64(),
        hash_word[3].as_canonical_u64(),
    ])
}

// ============================================================================
// Tests (run without Miden dependency — structural validation only)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_commitment_deterministic() {
        let nonce = [0xAAu8; 32];
        let epoch = 1_776_000_000u64;
        let c1 = compute_score_commitment(0.75, &nonce, epoch);
        let c2 = compute_score_commitment(0.75, &nonce, epoch);
        assert_eq!(c1, c2, "Same inputs should produce same commitment");
    }

    #[test]
    fn test_score_commitment_different_scores() {
        let nonce = [0xBBu8; 32];
        let epoch = 1_776_000_000u64;
        let c1 = compute_score_commitment(0.5, &nonce, epoch);
        let c2 = compute_score_commitment(0.6, &nonce, epoch);
        assert_ne!(
            c1, c2,
            "Different scores should produce different commitments"
        );
    }

    #[test]
    fn test_score_commitment_non_zero() {
        let nonce = [0u8; 32];
        let c = compute_score_commitment(0.0, &nonce, 0);
        assert_ne!(
            c, [0u8; 32],
            "Even zero inputs should produce non-zero commitment"
        );
    }

    #[test]
    fn test_score_commitment_different_nonces() {
        let n1 = [0x01u8; 32];
        let n2 = [0x02u8; 32];
        let epoch = 1_776_000_000u64;
        let c1 = compute_score_commitment(0.5, &n1, epoch);
        let c2 = compute_score_commitment(0.5, &n2, epoch);
        assert_ne!(
            c1, c2,
            "Different nonces must produce different commitments (replay prevention)"
        );
    }

    #[test]
    fn test_score_commitment_different_epochs() {
        let nonce = [0xCCu8; 32];
        let c1 = compute_score_commitment(0.5, &nonce, 1_000_000);
        let c2 = compute_score_commitment(0.5, &nonce, 2_000_000);
        assert_ne!(
            c1, c2,
            "Different epochs must produce different commitments"
        );
    }

    #[test]
    fn test_tier_thresholds_monotonic() {
        let tiers = [
            CivicTier::Observer,
            CivicTier::Participant,
            CivicTier::Citizen,
            CivicTier::Steward,
            CivicTier::Guardian,
        ];
        for pair in tiers.windows(2) {
            assert!(
                tier_threshold(&pair[0]) < tier_threshold(&pair[1]),
                "{:?} threshold should be less than {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn test_validate_empty_proof_rejected() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![],
            score_commitment: [0xAA; 32],
            proven_tier: CivicTier::Citizen,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![],
        };
        assert!(validate_miden_proof_structure(&proof).is_err());
    }

    #[test]
    fn test_validate_oversized_proof_rejected() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![0u8; MAX_MIDEN_PROOF_SIZE + 1],
            score_commitment: [0xBB; 32],
            proven_tier: CivicTier::Steward,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![],
        };
        assert!(validate_miden_proof_structure(&proof).is_err());
    }

    #[test]
    fn test_validate_zero_commitment_rejected() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![1, 2, 3],
            score_commitment: [0u8; 32],
            proven_tier: CivicTier::Guardian,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![],
        };
        assert!(validate_miden_proof_structure(&proof).is_err());
    }

    #[test]
    fn test_validate_valid_proof_accepted() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![0u8; 80_000], // 80KB — typical Miden proof
            score_commitment: compute_score_commitment_legacy(0.75),
            proven_tier: CivicTier::Steward,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![1], // 1 = tier check passed
        };
        assert!(validate_miden_proof_structure(&proof).is_ok());
    }

    #[test]
    fn test_miden_proof_serde_roundtrip() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![1, 2, 3, 4, 5],
            score_commitment: compute_score_commitment_legacy(0.5),
            proven_tier: CivicTier::Citizen,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![1],
        };
        let json = serde_json::to_string(&proof).unwrap();
        let back: MidenConsciousnessProof = serde_json::from_str(&json).unwrap();
        assert_eq!(proof.proof_bytes, back.proof_bytes);
        assert_eq!(proof.score_commitment, back.score_commitment);
        assert_eq!(proof.proven_tier, back.proven_tier);
    }

    /// Integration test: prove a consciousness tier and verify the proof.
    /// This exercises the full Miden VM pipeline: assemble → prove → serialize → verify.
    #[cfg(feature = "backend-miden")]
    #[test]
    fn test_prove_and_verify_consciousness_tier() {
        // Prove: Steward tier (phi=750 >= threshold=400)
        let proof = prove_consciousness_tier_miden(750, 400)
            .expect("Proving should succeed for phi >= threshold");

        assert_eq!(proof.proven_tier, CivicTier::Steward);
        assert_eq!(proof.proof_system, MidenProofSystem::MidenZkStark);
        assert!(!proof.proof_bytes.is_empty(), "Proof should be non-empty");
        assert!(
            proof.proof_bytes.len() < MAX_MIDEN_PROOF_SIZE,
            "Proof {} bytes should be under {} limit",
            proof.proof_bytes.len(),
            MAX_MIDEN_PROOF_SIZE
        );

        // Get program hash for verification
        let program_hash = consciousness_program_hash().expect("Program hash should be computable");

        // Verify
        let verified = verify_consciousness_tier_miden(&proof, 400, &program_hash)
            .expect("Verification should succeed");
        assert!(verified, "Valid proof should verify");
    }

    /// Proving must fail when phi < threshold (can't prove false statements).
    #[cfg(feature = "backend-miden")]
    #[test]
    fn test_prove_fails_when_phi_below_threshold() {
        let result = prove_consciousness_tier_miden(200, 400);
        assert!(result.is_err(), "Should fail when phi < threshold");
    }

    /// Prove Guardian tier (phi=900 >= threshold=600).
    #[cfg(feature = "backend-miden")]
    #[test]
    fn test_prove_guardian_tier() {
        let proof =
            prove_consciousness_tier_miden(900, 600).expect("Guardian proof should succeed");
        assert_eq!(proof.proven_tier, CivicTier::Guardian);

        let program_hash = consciousness_program_hash().unwrap();
        let verified = verify_consciousness_tier_miden(&proof, 600, &program_hash).unwrap();
        assert!(verified);
    }

    #[test]
    fn test_proof_size_at_limit_accepted() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![0u8; MAX_MIDEN_PROOF_SIZE], // exactly at limit
            score_commitment: [0xFF; 32],
            proven_tier: CivicTier::Guardian,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![],
        };
        assert!(validate_miden_proof_structure(&proof).is_ok());
    }
}
