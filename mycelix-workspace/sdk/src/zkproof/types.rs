//! zkSTARK Proof Types

use serde::{Deserialize, Serialize};

/// Public inputs for gradient proof verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicInputs {
    /// Hash of the starting global model
    pub global_model_hash: [u8; 32],
    /// Hash of the computed gradient
    pub gradient_hash: [u8; 32],
    /// Client identifier
    pub client_id: String,
    /// Training round index
    pub round_idx: u32,
    /// Number of local training epochs
    pub epochs: u32,
    /// Learning rate used
    pub learning_rate: f32,
}

/// Metadata about proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Proof generation time in milliseconds
    pub proof_gen_time_ms: f32,
    /// Proof size in bytes
    pub proof_size_bytes: usize,
    /// Gradient dimension
    pub gradient_dim: usize,
    /// Model dimension
    pub model_dim: usize,
    /// Whether using real zkSTARK (vs simulation)
    pub is_real_stark: bool,
}

/// Gradient proof containing proof bytes and public inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientProof {
    /// zkSTARK proof bytes
    pub proof_bytes: Vec<u8>,
    /// Public inputs (for verification)
    pub public_inputs: PublicInputs,
    /// Proof metadata
    pub metadata: ProofMetadata,
}

impl GradientProof {
    /// Create a new gradient proof
    pub fn new(proof_bytes: Vec<u8>, public_inputs: PublicInputs, metadata: ProofMetadata) -> Self {
        Self {
            proof_bytes,
            public_inputs,
            metadata,
        }
    }

    /// Proof size in KB
    pub fn size_kb(&self) -> f32 {
        self.proof_bytes.len() as f32 / 1024.0
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_serialization() {
        let public_inputs = PublicInputs {
            global_model_hash: [0u8; 32],
            gradient_hash: [1u8; 32],
            client_id: "test-node".to_string(),
            round_idx: 1,
            epochs: 5,
            learning_rate: 0.01,
        };

        let metadata = ProofMetadata {
            proof_gen_time_ms: 100.0,
            proof_size_bytes: 50000,
            gradient_dim: 1000,
            model_dim: 1000,
            is_real_stark: false,
        };

        let proof = GradientProof::new(vec![0u8; 50000], public_inputs, metadata);

        let bytes = proof.to_bytes().unwrap();
        let restored = GradientProof::from_bytes(&bytes).unwrap();

        assert_eq!(
            proof.public_inputs.client_id,
            restored.public_inputs.client_id
        );
        assert_eq!(proof.proof_bytes.len(), restored.proof_bytes.len());
    }
}
