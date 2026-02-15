//! Zero-Knowledge Proof Bridge
//!
//! Tauri commands for generating and verifying LUCID ZK proofs.
//!
//! This bridge exposes the `lucid-zkp` crate to the frontend, enabling:
//! - Anonymous belief attestation
//! - Reputation range proofs
//! - Proof verification

use lucid_zkp::{
    create_commitment, hash_belief, AnonymousBeliefInput, AnonymousBeliefProof,
    ProofGenerator, ProofVerifier, ReputationRangeInput, ReputationRangeProof, ZkpError,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;

/// State for ZK proof generation
pub struct ZkpState {
    generator: Arc<Mutex<ProofGenerator>>,
}

impl ZkpState {
    pub fn new() -> Self {
        Self {
            generator: Arc::new(Mutex::new(ProofGenerator::new())),
        }
    }
}

impl Default for ZkpState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TYPES FOR IPC
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct AnonymousBeliefProofInput {
    /// Hash of the belief content (hex-encoded or raw bytes)
    pub belief_hash: String,
    /// The agent's secret key (hex-encoded)
    pub agent_secret: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnonymousBeliefProofOutput {
    /// The serialized proof (JSON)
    pub proof_json: String,
    /// The author commitment (hex-encoded)
    pub author_commitment: String,
    /// Belief hash (hex-encoded)
    pub belief_hash: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReputationRangeProofInput {
    /// The actual reputation value
    pub actual_reputation: u64,
    /// The minimum threshold to prove
    pub min_threshold: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReputationRangeProofOutput {
    /// The serialized proof (JSON)
    pub proof_json: String,
    /// The reputation commitment (hex-encoded)
    pub reputation_commitment: String,
    /// The threshold that was proven
    pub min_threshold: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VerifyProofInput {
    /// The proof type ("anonymous_belief" or "reputation_range")
    pub proof_type: String,
    /// The serialized proof (JSON)
    pub proof_json: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VerifyProofOutput {
    /// Whether the proof is valid
    pub valid: bool,
    /// Error message if invalid
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitmentOutput {
    /// The commitment (hex-encoded)
    pub commitment: String,
    /// The nonce used (hex-encoded)
    pub nonce: String,
}

// ============================================================================
// TAURI COMMANDS
// ============================================================================

/// Generate an anonymous belief proof
///
/// This proves that the caller knows the secret identity behind a belief
/// commitment without revealing that identity.
#[tauri::command]
pub async fn generate_anonymous_belief_proof(
    input: AnonymousBeliefProofInput,
    state: State<'_, ZkpState>,
) -> Result<AnonymousBeliefProofOutput, String> {
    let generator = state.generator.lock().await;

    // Parse inputs
    let belief_hash = hex::decode(&input.belief_hash)
        .map_err(|e| format!("Invalid belief_hash hex: {}", e))?;
    let agent_secret = hex::decode(&input.agent_secret)
        .map_err(|e| format!("Invalid agent_secret hex: {}", e))?;

    // Generate a random nonce
    let nonce = ProofGenerator::generate_nonce();

    // Generate the proof
    let proof = generator
        .generate_anonymous_belief_proof(belief_hash, agent_secret, nonce)
        .map_err(|e| format!("Proof generation failed: {}", e))?;

    // Serialize and return
    let proof_json = proof.to_json().map_err(|e| format!("Serialization failed: {}", e))?;

    Ok(AnonymousBeliefProofOutput {
        proof_json,
        author_commitment: proof.author_commitment.to_hex(),
        belief_hash: hex::encode(&proof.belief_hash),
    })
}

/// Generate a reputation range proof
///
/// This proves that the caller's reputation exceeds a threshold
/// without revealing the exact reputation value.
#[tauri::command]
pub async fn generate_reputation_range_proof(
    input: ReputationRangeProofInput,
    state: State<'_, ZkpState>,
) -> Result<ReputationRangeProofOutput, String> {
    let generator = state.generator.lock().await;

    // Generate random blinding
    let blinding = ProofGenerator::generate_blinding();

    // Generate the proof
    let proof = generator
        .generate_reputation_range_proof(input.actual_reputation, input.min_threshold, blinding)
        .map_err(|e| format!("Proof generation failed: {}", e))?;

    // Serialize and return
    let proof_json = proof.to_json().map_err(|e| format!("Serialization failed: {}", e))?;

    Ok(ReputationRangeProofOutput {
        proof_json,
        reputation_commitment: proof.reputation_commitment.to_hex(),
        min_threshold: proof.min_threshold,
    })
}

/// Verify a proof
///
/// Verifies either an anonymous belief proof or a reputation range proof.
#[tauri::command]
pub async fn verify_proof(input: VerifyProofInput) -> Result<VerifyProofOutput, String> {
    match input.proof_type.as_str() {
        "anonymous_belief" => {
            let proof: AnonymousBeliefProof = serde_json::from_str(&input.proof_json)
                .map_err(|e| format!("Invalid proof JSON: {}", e))?;

            match ProofVerifier::verify_anonymous_belief(&proof) {
                Ok(valid) => Ok(VerifyProofOutput { valid, error: None }),
                Err(e) => Ok(VerifyProofOutput {
                    valid: false,
                    error: Some(e.to_string()),
                }),
            }
        }
        "reputation_range" => {
            let proof: ReputationRangeProof = serde_json::from_str(&input.proof_json)
                .map_err(|e| format!("Invalid proof JSON: {}", e))?;

            match ProofVerifier::verify_reputation_range(&proof) {
                Ok(valid) => Ok(VerifyProofOutput { valid, error: None }),
                Err(e) => Ok(VerifyProofOutput {
                    valid: false,
                    error: Some(e.to_string()),
                }),
            }
        }
        _ => Err(format!("Unknown proof type: {}", input.proof_type)),
    }
}

/// Create a commitment to a value
///
/// Returns a commitment and the nonce used so the value can be revealed later if needed.
#[tauri::command]
pub async fn create_value_commitment(value: String) -> Result<CommitmentOutput, String> {
    let value_bytes = value.as_bytes();
    let nonce = ProofGenerator::generate_nonce();

    let commitment = create_commitment(value_bytes, &nonce);

    Ok(CommitmentOutput {
        commitment: commitment.to_hex(),
        nonce: hex::encode(&nonce),
    })
}

/// Hash belief content
///
/// Creates a deterministic hash of belief content for use in proofs.
#[tauri::command]
pub async fn hash_belief_content(content: String) -> Result<String, String> {
    let hash = hash_belief(&content);
    Ok(hex::encode(&hash))
}

/// Check if ZKP is ready
#[tauri::command]
pub async fn zkp_ready() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hash_belief_content() {
        let result = hash_belief_content("Test belief".to_string()).await;
        assert!(result.is_ok());

        let hash = result.unwrap();
        assert!(!hash.is_empty());
        assert!(hex::decode(&hash).is_ok()); // Valid hex
    }

    #[tokio::test]
    async fn test_create_value_commitment() {
        let result = create_value_commitment("my_secret_value".to_string()).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.commitment.is_empty());
        assert!(!output.nonce.is_empty());
    }
}
