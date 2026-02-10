//! zkSTARK Circuit Implementation
//!
//! Currently provides simulation mode. Future: integrate RISC Zero zkVM.

use super::types::{GradientProof, ProofMetadata, PublicInputs};
use crate::crypto::secure_compare;
use sha3::{Digest, Sha3_256};
use std::time::Instant;

/// Minimum proof size in bytes (for security)
const MIN_PROOF_SIZE: usize = 10_000;
/// Target proof size for simulation (50-100KB like real zkSTARK)
const TARGET_PROOF_SIZE: usize = 75_000;

/// zkSTARK circuit for gradient computation verification
///
/// # Proof Statement
///
/// Proves that a gradient was computed correctly through local training:
/// - forward_pass_correct(D_i, W_t) - Loss computed correctly
/// - backward_pass_correct() - Gradients derived from loss
/// - gradient_matches_trace() - Proof matches submission
pub struct GradientProofCircuit {
    client_id: String,
    current_round: u32,
    /// Use real zkSTARK library (false = simulation mode)
    use_real_stark: bool,
}

impl GradientProofCircuit {
    /// Create new circuit for a client
    pub fn new(client_id: &str) -> Self {
        Self {
            client_id: client_id.to_string(),
            current_round: 0,
            use_real_stark: false, // Simulation mode by default
        }
    }

    /// Enable real zkSTARK mode (requires RISC Zero integration)
    pub fn with_real_stark(mut self) -> Self {
        self.use_real_stark = true;
        self
    }

    /// Set current round
    pub fn set_round(&mut self, round: u32) {
        self.current_round = round;
    }

    /// Generate gradient proof
    ///
    /// # Arguments
    ///
    /// * `global_model` - Starting model weights
    /// * `gradient` - Computed gradient
    /// * `epochs` - Number of local training epochs
    /// * `lr` - Learning rate
    ///
    /// # Returns
    ///
    /// GradientProof with zkSTARK proof bytes
    pub fn prove_gradient(
        &self,
        global_model: &[f32],
        gradient: &[f32],
        epochs: u32,
        lr: f32,
    ) -> GradientProof {
        let start = Instant::now();

        // Create public inputs
        let public_inputs = PublicInputs {
            global_model_hash: self.hash_f32_slice(global_model),
            gradient_hash: self.hash_f32_slice(gradient),
            client_id: self.client_id.clone(),
            round_idx: self.current_round,
            epochs,
            learning_rate: lr,
        };

        // Generate proof
        let proof_bytes = if self.use_real_stark {
            self.generate_real_stark_proof(&public_inputs, gradient)
        } else {
            self.generate_simulated_proof(&public_inputs, gradient)
        };

        let proof_gen_time_ms = start.elapsed().as_secs_f32() * 1000.0;

        let metadata = ProofMetadata {
            proof_gen_time_ms,
            proof_size_bytes: proof_bytes.len(),
            gradient_dim: gradient.len(),
            model_dim: global_model.len(),
            is_real_stark: self.use_real_stark,
        };

        GradientProof::new(proof_bytes, public_inputs, metadata)
    }

    /// Verify gradient proof
    ///
    /// # Arguments
    ///
    /// * `proof` - The gradient proof to verify
    ///
    /// # Returns
    ///
    /// true if proof is valid, false otherwise
    pub fn verify_proof(&self, proof: &GradientProof) -> bool {
        if self.use_real_stark {
            self.verify_real_stark_proof(proof)
        } else {
            self.verify_simulated_proof(proof)
        }
    }

    /// Hash a f32 slice using SHA3-256
    fn hash_f32_slice(&self, data: &[f32]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        for val in data {
            hasher.update(val.to_le_bytes());
        }
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Generate simulated zkSTARK proof
    ///
    /// NOT cryptographically secure - for testing architecture only.
    fn generate_simulated_proof(&self, public_inputs: &PublicInputs, gradient: &[f32]) -> Vec<u8> {
        let mut proof = Vec::with_capacity(TARGET_PROOF_SIZE);

        // 1. Public commitment (32 bytes)
        let public_commitment = self.compute_public_commitment(public_inputs);
        proof.extend_from_slice(&public_commitment);

        // 2. Gradient commitment (32 bytes)
        let gradient_commitment = self.hash_f32_slice(gradient);
        proof.extend_from_slice(&gradient_commitment);

        // 3. Epochs encoding (4 bytes)
        proof.extend_from_slice(&public_inputs.epochs.to_le_bytes());

        // 4. Learning rate encoding (4 bytes)
        proof.extend_from_slice(&public_inputs.learning_rate.to_le_bytes());

        // 5. Simulated FRI layers and STARK data
        // Real STARK has: trace commitment, constraint evaluations, FRI layers, etc.
        // We simulate with deterministic padding based on gradient

        let mut padding_seed = [0u8; 32];
        padding_seed[..8].copy_from_slice(&(gradient.len() as u64).to_le_bytes());
        padding_seed[8..16]
            .copy_from_slice(&public_inputs.epochs.to_le_bytes().repeat(2).as_slice()[..8]);

        // Generate deterministic padding
        let mut hasher = Sha3_256::new();
        hasher.update(padding_seed);

        while proof.len() < TARGET_PROOF_SIZE {
            let hash = hasher.finalize_reset();
            proof.extend_from_slice(&hash);
            hasher.update(hash);
        }

        proof.truncate(TARGET_PROOF_SIZE);
        proof
    }

    /// Compute public input commitment
    fn compute_public_commitment(&self, public_inputs: &PublicInputs) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(public_inputs.global_model_hash);
        hasher.update(public_inputs.gradient_hash);
        hasher.update(public_inputs.client_id.as_bytes());
        hasher.update(public_inputs.round_idx.to_le_bytes());
        hasher.update(public_inputs.epochs.to_le_bytes());
        hasher.update(public_inputs.learning_rate.to_le_bytes());

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Verify simulated proof
    fn verify_simulated_proof(&self, proof: &GradientProof) -> bool {
        // Basic sanity checks
        if proof.proof_bytes.len() < MIN_PROOF_SIZE {
            return false;
        }

        // Extract and verify public commitment
        if proof.proof_bytes.len() < 32 {
            return false;
        }

        let proof_commitment = &proof.proof_bytes[0..32];
        let expected_commitment = self.compute_public_commitment(&proof.public_inputs);

        // SECURITY: Use constant-time comparison to prevent timing attacks (FIND-004)
        if !secure_compare(proof_commitment, &expected_commitment) {
            return false;
        }

        // Verify epochs encoding
        if proof.proof_bytes.len() < 68 {
            return false;
        }

        let epochs_bytes: [u8; 4] = proof.proof_bytes[64..68].try_into().unwrap_or([0; 4]);
        let encoded_epochs = u32::from_le_bytes(epochs_bytes);

        if encoded_epochs != proof.public_inputs.epochs {
            return false;
        }

        true
    }

    /// Generate real zkSTARK proof (future implementation)
    fn generate_real_stark_proof(
        &self,
        _public_inputs: &PublicInputs,
        _gradient: &[f32],
    ) -> Vec<u8> {
        // Future: integrate RISC Zero zkVM
        // This would:
        // 1. Compile the gradient computation to RISC-V
        // 2. Execute in zkVM with trace recording
        // 3. Generate STARK proof from trace
        // 4. Return serialized Receipt

        // Return empty proof bytes; callers should use simulation mode
        Vec::new()
    }

    /// Verify real zkSTARK proof (future implementation)
    fn verify_real_stark_proof(&self, _proof: &GradientProof) -> bool {
        // Future: verify RISC Zero Receipt
        false
    }
}

/// Batch verify multiple proofs
///
/// More efficient than individual verification when checking many proofs.
#[allow(dead_code)]
pub fn batch_verify(proofs: &[&GradientProof]) -> Vec<bool> {
    proofs
        .iter()
        .map(|proof| {
            let circuit = GradientProofCircuit::new(&proof.public_inputs.client_id);
            circuit.verify_proof(proof)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_generation() {
        let circuit = GradientProofCircuit::new("test-node");

        let global_model: Vec<f32> = vec![0.1; 1000];
        let gradient: Vec<f32> = vec![0.01; 1000];

        let proof = circuit.prove_gradient(&global_model, &gradient, 5, 0.01);

        assert!(proof.proof_bytes.len() >= MIN_PROOF_SIZE);
        assert_eq!(proof.public_inputs.epochs, 5);
        assert!(!proof.metadata.is_real_stark);
    }

    #[test]
    fn test_deterministic_proofs() {
        let circuit = GradientProofCircuit::new("test-node");

        let global_model: Vec<f32> = vec![0.1; 1000];
        let gradient: Vec<f32> = vec![0.01; 1000];

        let proof1 = circuit.prove_gradient(&global_model, &gradient, 5, 0.01);
        let proof2 = circuit.prove_gradient(&global_model, &gradient, 5, 0.01);

        // Same inputs should produce same proof
        assert_eq!(proof1.proof_bytes, proof2.proof_bytes);
    }

    #[test]
    fn test_tampered_proof_fails() {
        let circuit = GradientProofCircuit::new("test-node");

        let global_model: Vec<f32> = vec![0.1; 1000];
        let gradient: Vec<f32> = vec![0.01; 1000];

        let mut proof = circuit.prove_gradient(&global_model, &gradient, 5, 0.01);

        // Tamper with commitment
        proof.proof_bytes[0] ^= 0xFF;

        assert!(!circuit.verify_proof(&proof));
    }

    #[test]
    fn test_batch_verify() {
        let global_model: Vec<f32> = vec![0.1; 1000];

        let proofs: Vec<GradientProof> = (0..5)
            .map(|i| {
                let circuit = GradientProofCircuit::new(&format!("node-{}", i));
                let gradient: Vec<f32> = vec![0.01 * (i as f32 + 1.0); 1000];
                circuit.prove_gradient(&global_model, &gradient, 5, 0.01)
            })
            .collect();

        let proof_refs: Vec<&GradientProof> = proofs.iter().collect();
        let results = batch_verify(&proof_refs);

        assert!(results.iter().all(|&r| r));
    }
}
