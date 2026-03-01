//! mycelix-attribution-verifier — Off-chain ZK-STARK verification for usage attestations.
//!
//! Workflow:
//!   1. Read unverified UsageAttestation from a JSON file or stdin
//!   2. Verify the ZK-STARK proof (witness_commitment + proof_bytes)
//!   3. Sign the verification result with Ed25519
//!   4. Output a VerifyAttestationInput JSON payload for submission to the DHT
//!
//! The verifier runs off-chain (not in WASM) because ZK-STARK verification
//! is computationally expensive and requires full crypto libraries.
//!
//! Proof format (to be finalized with Winterfell integration):
//!   - witness_commitment: Blake3 hash of the secret witness (32 bytes)
//!   - proof_bytes: Winterfell STARK proof serialized bytes
//!
//! For now, this binary implements:
//!   - Witness commitment verification (Blake3)
//!   - Ed25519 signing of verification results
//!   - Proof format validation (size checks, structure)
//!   - JSON I/O compatible with the DHT's verify_usage_attestation extern

use clap::Parser;
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

// ── CLI Arguments ────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "mycelix-attribution-verifier")]
#[command(about = "Verify ZK-STARK usage attestation proofs and sign verification results")]
struct Args {
    /// Path to attestation JSON file (or - for stdin)
    #[arg(short, long)]
    attestation: PathBuf,

    /// Path to Ed25519 signing key (32 bytes hex or raw)
    #[arg(short, long)]
    signing_key: PathBuf,

    /// Output format: json or submit-payload
    #[arg(short, long, default_value = "json")]
    format: OutputFormat,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum OutputFormat {
    /// Full verification report
    Json,
    /// VerifyAttestationInput payload for DHT submission
    SubmitPayload,
}

// ── Types ────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct AttestationInput {
    /// Original action hash from the DHT (hex-encoded)
    original_action_hash: String,
    id: String,
    dependency_id: String,
    user_did: String,
    /// Blake3 hash of the secret witness (hex-encoded, 32 bytes)
    witness_commitment: String,
    /// STARK proof bytes (hex-encoded)
    proof_bytes: String,
    /// Public inputs for Winterfell STARK verification (8 u64 values).
    /// If absent, falls back to structural validation only.
    public_inputs: Option<Vec<u64>>,
}

#[derive(Serialize, Debug)]
struct VerificationReport {
    attestation_id: String,
    dependency_id: String,
    user_did: String,
    commitment_valid: bool,
    proof_valid: bool,
    proof_size_bytes: usize,
    verifier_pubkey_hex: String,
    signature_hex: String,
    verdict: String,
}

#[derive(Serialize, Debug)]
struct SubmitPayload {
    /// Hex-encoded action hash (must be decoded to bytes for DHT submission)
    original_action_hash: String,
    /// Ed25519 public key (32 bytes, hex-encoded)
    verifier_pubkey: String,
    /// Ed25519 signature (64 bytes, hex-encoded)
    verifier_signature: String,
}

#[derive(Error, Debug)]
enum VerifierError {
    #[error("Invalid witness commitment: expected 32 bytes, got {0}")]
    InvalidCommitment(usize),
    #[error("Proof too small: {0} bytes (minimum 32)")]
    ProofTooSmall(usize),
    #[error("Proof too large: {0} bytes (maximum 512000)")]
    ProofTooLarge(usize),
    #[error("Proof deserialization failed: {0}")]
    DeserializationFailed(String),
}

// ── Verification Logic ──────────────────────────────────────────────

/// Verify that commitment bytes match the commitment limbs in public inputs.
/// Returns false if they don't match, true if they match or can't be checked.
fn cross_validate_commitment(commitment_bytes: &[u8], public_inputs: Option<&[u64]>) -> bool {
    let Some(pi) = public_inputs else {
        return true; // Can't check without public inputs
    };
    if pi.len() != 8 || commitment_bytes.len() != 32 {
        return true; // Let other checks handle these errors
    }

    let commitment_arr: [u8; 32] = commitment_bytes.try_into().unwrap();
    let expected = mycelix_attribution_stark_common::commitment_to_limbs(&commitment_arr);
    let actual = [pi[4], pi[5], pi[6], pi[7]];
    expected == actual
}

fn verify_commitment(commitment_bytes: &[u8]) -> Result<bool, VerifierError> {
    if commitment_bytes.len() != 32 {
        return Err(VerifierError::InvalidCommitment(commitment_bytes.len()));
    }
    // Commitment format is valid (32-byte Blake3 hash)
    // Actual witness verification would check: blake3(witness) == commitment
    // But we don't have the witness (that's the point of ZK)
    Ok(true)
}

fn verify_proof(
    proof_bytes: &[u8],
    public_inputs_raw: Option<&[u64]>,
) -> Result<bool, VerifierError> {
    if proof_bytes.len() < 32 {
        return Err(VerifierError::ProofTooSmall(proof_bytes.len()));
    }
    if proof_bytes.len() > 512_000 {
        return Err(VerifierError::ProofTooLarge(proof_bytes.len()));
    }

    // Full Winterfell STARK verification requires public inputs
    let Some(pi_vals) = public_inputs_raw else {
        eprintln!(
            "  REJECTED: No public_inputs provided. STARK verification requires public inputs."
        );
        return Ok(false);
    };

    if pi_vals.len() != 8 {
        eprintln!(
            "  REJECTED: Expected 8 public inputs, got {}.",
            pi_vals.len()
        );
        return Ok(false);
    }

    use mycelix_attribution_stark_common::{
        default_proof_options, PublicInputs, UsageAttestationAir,
    };
    use winter_crypto::hashers::Blake3_256;
    use winter_math::fields::f128::BaseElement;
    use winterfell::crypto::{DefaultRandomCoin, MerkleTree};

    let pub_inputs = PublicInputs {
        dep_hash_lo: pi_vals[0],
        dep_hash_hi: pi_vals[1],
        did_hash_lo: pi_vals[2],
        did_hash_hi: pi_vals[3],
        witness_commitment: [pi_vals[4], pi_vals[5], pi_vals[6], pi_vals[7]],
    };

    let proof = winterfell::Proof::from_bytes(proof_bytes).map_err(|e| {
        eprintln!("  Proof deserialization failed: {}", e);
        VerifierError::DeserializationFailed(e.to_string())
    })?;

    let acceptable = winterfell::AcceptableOptions::OptionSet(vec![default_proof_options()]);

    match winterfell::verify::<
        UsageAttestationAir,
        Blake3_256<BaseElement>,
        DefaultRandomCoin<Blake3_256<BaseElement>>,
        MerkleTree<Blake3_256<BaseElement>>,
    >(proof, pub_inputs, &acceptable)
    {
        Ok(_) => {
            eprintln!("  STARK verification: PASSED");
            Ok(true)
        }
        Err(e) => {
            eprintln!("  STARK verification: FAILED ({})", e);
            Ok(false)
        }
    }
}

/// Sign a verification result. The signed message includes the original_action_hash
/// to prevent replay attacks across different attestations.
fn sign_verification(
    signing_key: &SigningKey,
    attestation_id: &str,
    dependency_id: &str,
    original_action_hash: &str,
    verdict: bool,
) -> (Vec<u8>, Vec<u8>) {
    let message = format!(
        "mycelix-attribution:verify:{}:{}:{}:{}",
        attestation_id, dependency_id, original_action_hash, verdict
    );
    let signature = signing_key.sign(message.as_bytes());

    let pubkey = signing_key.verifying_key().to_bytes().to_vec();
    let sig_bytes = signature.to_bytes().to_vec();

    (pubkey, sig_bytes)
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // Read attestation
    let att_content = if args.attestation.to_str() == Some("-") {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .unwrap_or_else(|e| {
                eprintln!("Failed to read stdin: {}", e);
                std::process::exit(1);
            });
        buf
    } else {
        std::fs::read_to_string(&args.attestation).unwrap_or_else(|e| {
            eprintln!("Failed to read {}: {}", args.attestation.display(), e);
            std::process::exit(1);
        })
    };

    let att: AttestationInput = serde_json::from_str(&att_content).unwrap_or_else(|e| {
        eprintln!("Failed to parse attestation JSON: {}", e);
        std::process::exit(1);
    });

    // Read signing key
    let key_content = std::fs::read(&args.signing_key).unwrap_or_else(|e| {
        eprintln!(
            "Failed to read signing key {}: {}",
            args.signing_key.display(),
            e
        );
        std::process::exit(1);
    });

    let key_bytes: [u8; 32] = if key_content.len() == 64 {
        // Hex-encoded
        let decoded = hex::decode(&key_content).unwrap_or_else(|e| {
            eprintln!("Failed to decode hex signing key: {}", e);
            std::process::exit(1);
        });
        decoded.try_into().unwrap_or_else(|_| {
            eprintln!("Signing key must be exactly 32 bytes");
            std::process::exit(1);
        })
    } else if key_content.len() == 32 {
        // Raw bytes
        key_content.try_into().unwrap_or_else(|_| {
            eprintln!("Signing key must be exactly 32 bytes");
            std::process::exit(1);
        })
    } else {
        eprintln!(
            "Invalid signing key: expected 32 bytes (raw) or 64 bytes (hex), got {}",
            key_content.len()
        );
        std::process::exit(1);
    };

    let signing_key = SigningKey::from_bytes(&key_bytes);

    // Decode attestation fields
    let commitment_bytes = hex::decode(&att.witness_commitment).unwrap_or_else(|e| {
        eprintln!("Invalid witness_commitment hex: {}", e);
        std::process::exit(1);
    });
    let proof_bytes = hex::decode(&att.proof_bytes).unwrap_or_else(|e| {
        eprintln!("Invalid proof_bytes hex: {}", e);
        std::process::exit(1);
    });

    eprintln!("Verifying attestation {}...", att.id);
    eprintln!("  dependency: {}", att.dependency_id);
    eprintln!("  user: {}", att.user_did);
    eprintln!("  commitment: {} bytes", commitment_bytes.len());
    eprintln!("  proof: {} bytes", proof_bytes.len());

    // Verify
    let commitment_valid = match verify_commitment(&commitment_bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  Commitment verification failed: {}", e);
            false
        }
    };

    // Cross-validate: commitment bytes must match commitment limbs in public_inputs
    let commitment_matches =
        cross_validate_commitment(&commitment_bytes, att.public_inputs.as_deref());
    if !commitment_matches {
        eprintln!("  REJECTED: witness_commitment hex does not match public_inputs[4..8]");
    }

    let proof_valid = match verify_proof(&proof_bytes, att.public_inputs.as_deref()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  Proof verification failed: {}", e);
            false
        }
    };

    let verdict = commitment_valid && commitment_matches && proof_valid;

    // Sign
    let (pubkey, signature) = sign_verification(
        &signing_key,
        &att.id,
        &att.dependency_id,
        &att.original_action_hash,
        verdict,
    );

    eprintln!(
        "  Verdict: {} (commitment={}, proof={})",
        if verdict { "VERIFIED" } else { "REJECTED" },
        commitment_valid,
        proof_valid
    );

    match args.format {
        OutputFormat::Json => {
            let report = VerificationReport {
                attestation_id: att.id,
                dependency_id: att.dependency_id,
                user_did: att.user_did,
                commitment_valid,
                proof_valid,
                proof_size_bytes: proof_bytes.len(),
                verifier_pubkey_hex: hex::encode(&pubkey),
                signature_hex: hex::encode(&signature),
                verdict: if verdict {
                    "VERIFIED".to_string()
                } else {
                    "REJECTED".to_string()
                },
            };
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        OutputFormat::SubmitPayload => {
            let payload = SubmitPayload {
                original_action_hash: att.original_action_hash,
                verifier_pubkey: hex::encode(&pubkey),
                verifier_signature: hex::encode(&signature),
            };
            println!("{}", serde_json::to_string_pretty(&payload).unwrap());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_commitment_valid() {
        let commitment = vec![0xAB; 32];
        assert!(verify_commitment(&commitment).unwrap());
    }

    #[test]
    fn test_verify_commitment_wrong_size() {
        let commitment = vec![0xAB; 16];
        assert!(verify_commitment(&commitment).is_err());
    }

    #[test]
    fn test_verify_proof_no_public_inputs_rejected() {
        let proof = vec![0x01; 256];
        // Proofs without public_inputs are rejected (not silently accepted)
        assert!(!verify_proof(&proof, None).unwrap());
    }

    #[test]
    fn test_verify_proof_wrong_public_inputs_count() {
        let proof = vec![0x01; 256];
        // Wrong number of public inputs (need exactly 8)
        assert!(!verify_proof(&proof, Some(&[1, 2, 3])).unwrap());
    }

    #[test]
    fn test_verify_proof_too_small() {
        let proof = vec![0x01; 16];
        assert!(verify_proof(&proof, None).is_err());
    }

    #[test]
    fn test_verify_proof_too_large() {
        let proof = vec![0x01; 600_000];
        assert!(verify_proof(&proof, None).is_err());
    }

    #[test]
    fn test_sign_verification() {
        let key = SigningKey::from_bytes(&[42u8; 32]);
        let (pubkey, sig) =
            sign_verification(&key, "attest-001", "crate:serde:1.0", "abc123", true);
        assert_eq!(pubkey.len(), 32);
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn test_sign_verification_deterministic() {
        let key = SigningKey::from_bytes(&[42u8; 32]);
        let (_, sig1) = sign_verification(&key, "a", "b", "hash1", true);
        let (_, sig2) = sign_verification(&key, "a", "b", "hash1", true);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_sign_verification_replay_protection() {
        let key = SigningKey::from_bytes(&[42u8; 32]);
        // Same attestation ID & dep, different action hash → different signature
        let (_, sig1) = sign_verification(&key, "a", "b", "hash1", true);
        let (_, sig2) = sign_verification(&key, "a", "b", "hash2", true);
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_sign_verification_crypto_valid() {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};
        let key = SigningKey::from_bytes(&[42u8; 32]);
        let (pubkey_bytes, sig_bytes) =
            sign_verification(&key, "attest-001", "crate:serde:1.0", "action-hash", true);
        let vk = VerifyingKey::from_bytes(&pubkey_bytes.try_into().unwrap()).unwrap();
        let sig = Signature::from_bytes(&sig_bytes.try_into().unwrap());
        let msg = "mycelix-attribution:verify:attest-001:crate:serde:1.0:action-hash:true";
        assert!(vk.verify(msg.as_bytes(), &sig).is_ok());
    }

    #[test]
    fn test_cross_validate_commitment_match() {
        use mycelix_attribution_stark_common::{commitment_to_limbs, compute_witness_commitment};
        let commitment = compute_witness_commitment(2, "crate:serde:1.0", "did:abc", "Org");
        let limbs = commitment_to_limbs(&commitment);
        let pi = vec![0, 0, 0, 0, limbs[0], limbs[1], limbs[2], limbs[3]];
        assert!(cross_validate_commitment(&commitment, Some(&pi)));
    }

    #[test]
    fn test_cross_validate_commitment_mismatch() {
        use mycelix_attribution_stark_common::compute_witness_commitment;
        let commitment = compute_witness_commitment(2, "crate:serde:1.0", "did:abc", "Org");
        // Different limbs in public inputs
        let pi = vec![0, 0, 0, 0, 999, 888, 777, 666];
        assert!(!cross_validate_commitment(&commitment, Some(&pi)));
    }

    /// End-to-end: generate a real STARK proof via Winterfell, then verify it
    /// through the verifier's `verify_proof` function.
    #[test]
    fn test_e2e_prove_then_verify() {
        use mycelix_attribution_stark_common::{
            commitment_to_limbs, compute_witness_commitment, default_proof_options,
            hash_to_u64_pair, PublicInputs, UsageAttestationAir, TRACE_LENGTH, TRACE_WIDTH,
        };
        use winter_crypto::hashers::Blake3_256;
        use winter_math::fields::f128::BaseElement;
        use winter_math::FieldElement;
        use winterfell::{
            crypto::{DefaultRandomCoin, MerkleTree},
            matrix::ColMatrix,
            DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde,
            PartitionOptions, ProofOptions, Prover, StarkDomain, TraceInfo, TraceTable,
        };

        // -- Inline prover (mirrors prover binary) --
        struct TestProver {
            options: ProofOptions,
            pub_inputs: PublicInputs,
        }
        impl Prover for TestProver {
            type BaseField = BaseElement;
            type Air = UsageAttestationAir;
            type Trace = TraceTable<Self::BaseField>;
            type HashFn = Blake3_256<Self::BaseField>;
            type VC = MerkleTree<Self::HashFn>;
            type RandomCoin = DefaultRandomCoin<Self::HashFn>;
            type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
                DefaultTraceLde<E, Self::HashFn, Self::VC>;
            type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
                DefaultConstraintEvaluator<'a, Self::Air, E>;
            type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
                DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;

            fn get_pub_inputs(&self, _trace: &Self::Trace) -> PublicInputs {
                self.pub_inputs.clone()
            }
            fn options(&self) -> &ProofOptions {
                &self.options
            }
            fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
                &self,
                trace_info: &TraceInfo,
                main_trace: &ColMatrix<Self::BaseField>,
                domain: &StarkDomain<Self::BaseField>,
                partition_options: PartitionOptions,
            ) -> (Self::TraceLde<E>, winterfell::TracePolyTable<E>) {
                DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
            }
            fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
                &self,
                air: &'a Self::Air,
                aux_rand_elements: Option<winterfell::AuxRandElements<E>>,
                composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
            ) -> Self::ConstraintEvaluator<'a, E> {
                DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
            }
            fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
                &self,
                composition_poly_trace: winterfell::CompositionPolyTrace<E>,
                num_constraint_composition_columns: usize,
                domain: &StarkDomain<Self::BaseField>,
                partition_options: PartitionOptions,
            ) -> (
                Self::ConstraintCommitment<E>,
                winterfell::CompositionPoly<E>,
            ) {
                DefaultConstraintCommitment::new(
                    composition_poly_trace,
                    num_constraint_composition_columns,
                    domain,
                    partition_options,
                )
            }
        }

        // Generate proof
        let dep_id = "crate:tokio:1.0";
        let user_did = "did:mycelix:e2etest";
        let scale: u8 = 3; // Large

        let (dep_lo, dep_hi) = hash_to_u64_pair(dep_id.as_bytes());
        let (did_lo, did_hi) = hash_to_u64_pair(user_did.as_bytes());
        let commitment = compute_witness_commitment(scale, dep_id, user_did, "");
        let commitment_limbs = commitment_to_limbs(&commitment);

        let pub_inputs = PublicInputs {
            dep_hash_lo: dep_lo,
            dep_hash_hi: dep_hi,
            did_hash_lo: did_lo,
            did_hash_hi: did_hi,
            witness_commitment: commitment_limbs,
        };

        let scale_cycle: [u64; 4] = [1, 2, 3, 4];
        let mut trace = TraceTable::new(TRACE_WIDTH, TRACE_LENGTH);
        trace.fill(
            |state| {
                state[0] = BaseElement::from(scale as u64);
                state[1] = BaseElement::from(dep_lo);
                state[2] = BaseElement::from(dep_hi);
                state[3] = BaseElement::from(did_lo);
                state[4] = BaseElement::from(did_hi);
                state[5] = BaseElement::from(commitment_limbs[0]);
                state[6] = BaseElement::from(commitment_limbs[1]);
                state[7] = BaseElement::from(commitment_limbs[2]);
                state[8] = BaseElement::from(commitment_limbs[3]);
            },
            |step, state| {
                state[0] = BaseElement::from(scale_cycle[step % 4]);
            },
        );

        let prover = TestProver {
            options: default_proof_options(),
            pub_inputs: pub_inputs.clone(),
        };
        let proof = prover.prove(trace).expect("proof generation failed");
        let proof_bytes = proof.to_bytes();

        // Verify through verifier's verify_proof()
        let pi_vec: Vec<u64> = vec![
            pub_inputs.dep_hash_lo,
            pub_inputs.dep_hash_hi,
            pub_inputs.did_hash_lo,
            pub_inputs.did_hash_hi,
            pub_inputs.witness_commitment[0],
            pub_inputs.witness_commitment[1],
            pub_inputs.witness_commitment[2],
            pub_inputs.witness_commitment[3],
        ];

        let result = verify_proof(&proof_bytes, Some(&pi_vec));
        assert!(
            result.is_ok(),
            "verify_proof returned error: {:?}",
            result.err()
        );
        assert!(result.unwrap(), "STARK proof should verify successfully");
    }
}
