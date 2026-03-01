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
}

// ── Verification Logic ──────────────────────────────────────────────

fn verify_commitment(commitment_bytes: &[u8]) -> Result<bool, VerifierError> {
    if commitment_bytes.len() != 32 {
        return Err(VerifierError::InvalidCommitment(commitment_bytes.len()));
    }
    // Commitment format is valid (32-byte Blake3 hash)
    // Actual witness verification would check: blake3(witness) == commitment
    // But we don't have the witness (that's the point of ZK)
    Ok(true)
}

fn verify_proof(proof_bytes: &[u8]) -> Result<bool, VerifierError> {
    if proof_bytes.len() < 32 {
        return Err(VerifierError::ProofTooSmall(proof_bytes.len()));
    }
    if proof_bytes.len() > 512_000 {
        return Err(VerifierError::ProofTooLarge(proof_bytes.len()));
    }

    // TODO: Integrate Winterfell STARK verification here.
    //
    // The proof format will be:
    //   1. Deserialize proof_bytes into winterfell::StarkProof
    //   2. Define the AIR (Algebraic Intermediate Representation) for usage attestation:
    //      - Public inputs: dependency_id hash, user_did hash
    //      - Private inputs: usage scale, organization details
    //      - Constraints: usage_scale ∈ {Small, Medium, Large, Enterprise}
    //   3. Call winterfell::verify(proof, pub_inputs, air_params)
    //   4. Return verification result
    //
    // For now: structural validation only (non-empty, within size limits)
    eprintln!(
        "  Warning: Full Winterfell STARK verification not yet integrated."
    );
    eprintln!(
        "  Performing structural validation only ({} bytes).",
        proof_bytes.len()
    );

    Ok(true)
}

fn sign_verification(
    signing_key: &SigningKey,
    attestation_id: &str,
    dependency_id: &str,
    verdict: bool,
) -> (Vec<u8>, Vec<u8>) {
    let message = format!(
        "mycelix-attribution:verify:{}:{}:{}",
        attestation_id, dependency_id, verdict
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
            eprintln!(
                "Failed to read {}: {}",
                args.attestation.display(),
                e
            );
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
        let decoded = hex::decode(&key_content)
            .unwrap_or_else(|e| {
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

    let proof_valid = match verify_proof(&proof_bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  Proof verification failed: {}", e);
            false
        }
    };

    let verdict = commitment_valid && proof_valid;

    // Sign
    let (pubkey, signature) = sign_verification(
        &signing_key,
        &att.id,
        &att.dependency_id,
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
    fn test_verify_proof_valid() {
        let proof = vec![0x01; 256];
        assert!(verify_proof(&proof).unwrap());
    }

    #[test]
    fn test_verify_proof_too_small() {
        let proof = vec![0x01; 16];
        assert!(verify_proof(&proof).is_err());
    }

    #[test]
    fn test_verify_proof_too_large() {
        let proof = vec![0x01; 600_000];
        assert!(verify_proof(&proof).is_err());
    }

    #[test]
    fn test_sign_verification() {
        let key = SigningKey::from_bytes(&[42u8; 32]);
        let (pubkey, sig) = sign_verification(
            &key,
            "attest-001",
            "crate:serde:1.0",
            true,
        );
        assert_eq!(pubkey.len(), 32);
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn test_sign_verification_deterministic() {
        let key = SigningKey::from_bytes(&[42u8; 32]);
        let (_, sig1) = sign_verification(&key, "a", "b", true);
        let (_, sig2) = sign_verification(&key, "a", "b", true);
        assert_eq!(sig1, sig2);
    }
}
