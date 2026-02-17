//! Proof Serialization
//!
//! Efficient serialization/deserialization of proofs for network transport and storage.
//!
//! ## Features
//!
//! - Compact binary format for proofs
//! - Version-tagged serialization for forward compatibility
//! - Compression support for large proofs
//! - Validation on deserialization
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::{SerializableProof, ProofEnvelope};
//!
//! // Serialize a proof
//! let envelope = ProofEnvelope::from_range_proof(proof)?;
//! let bytes = envelope.to_bytes()?;
//!
//! // Deserialize
//! let restored = ProofEnvelope::from_bytes(&bytes)?;
//! let proof = restored.into_range_proof()?;
//! ```

use crate::proofs::{
    ProofError, ProofResult, ProofType, SecurityLevel,
    RangeProof, GradientIntegrityProof, IdentityAssuranceProof, VoteEligibilityProof,
    GradientPublicInputs, IdentityPublicInputs, VotePublicInputs,
};
use serde::{Deserialize, Serialize};

/// Serializable version of range public inputs
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializableRangeInputs {
    min: u64,
    max: u64,
    adjusted_offset: u64,
}

/// Current serialization format version
pub const SERIALIZATION_VERSION: u8 = 1;

/// Magic bytes for proof envelope identification
pub const MAGIC_BYTES: [u8; 4] = [0x50, 0x52, 0x4F, 0x46]; // "PROF"

/// Serialized proof envelope with metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofEnvelope {
    /// Format version
    pub version: u8,

    /// Proof type
    pub proof_type: ProofType,

    /// Security level used
    pub security_level: SecurityLevel,

    /// Serialized public inputs
    pub public_inputs: Vec<u8>,

    /// Serialized proof bytes
    pub proof_bytes: Vec<u8>,

    /// Optional compression flag
    pub compressed: bool,

    /// Checksum for integrity
    pub checksum: [u8; 4],
}

impl ProofEnvelope {
    /// Create envelope from range proof
    pub fn from_range_proof(proof: &RangeProof) -> ProofResult<Self> {
        let public_inputs = proof.public_inputs();
        let serializable = SerializableRangeInputs {
            min: public_inputs.min,
            max: public_inputs.max,
            adjusted_offset: public_inputs.adjusted_offset,
        };
        let public_inputs_bytes = serde_json::to_vec(&serializable)
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        let proof_bytes = proof.to_bytes();

        let checksum = compute_checksum(&proof_bytes);

        Ok(Self {
            version: SERIALIZATION_VERSION,
            proof_type: ProofType::Range,
            security_level: SecurityLevel::Standard128, // Default
            public_inputs: public_inputs_bytes,
            proof_bytes,
            compressed: false,
            checksum,
        })
    }

    /// Create envelope from gradient proof
    pub fn from_gradient_proof(proof: &GradientIntegrityProof) -> ProofResult<Self> {
        let public_inputs = proof.public_inputs();
        let public_inputs_bytes = serde_json::to_vec(&SerializableGradientInputs::from(public_inputs.clone()))
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        let proof_bytes = proof.to_bytes();
        let checksum = compute_checksum(&proof_bytes);

        Ok(Self {
            version: SERIALIZATION_VERSION,
            proof_type: ProofType::GradientIntegrity,
            security_level: SecurityLevel::Standard128,
            public_inputs: public_inputs_bytes,
            proof_bytes,
            compressed: false,
            checksum,
        })
    }

    /// Create envelope from identity proof
    pub fn from_identity_proof(proof: &IdentityAssuranceProof) -> ProofResult<Self> {
        let public_inputs = proof.public_inputs();
        let public_inputs_bytes = serde_json::to_vec(&SerializableIdentityInputs::from(public_inputs.clone()))
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        let proof_bytes = proof.to_bytes();
        let checksum = compute_checksum(&proof_bytes);

        Ok(Self {
            version: SERIALIZATION_VERSION,
            proof_type: ProofType::IdentityAssurance,
            security_level: SecurityLevel::Standard128,
            public_inputs: public_inputs_bytes,
            proof_bytes,
            compressed: false,
            checksum,
        })
    }

    /// Create envelope from vote proof
    pub fn from_vote_proof(proof: &VoteEligibilityProof) -> ProofResult<Self> {
        let public_inputs = proof.public_inputs();
        let public_inputs_bytes = serde_json::to_vec(&SerializableVoteInputs::from(public_inputs.clone()))
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        let proof_bytes = proof.to_bytes();
        let checksum = compute_checksum(&proof_bytes);

        Ok(Self {
            version: SERIALIZATION_VERSION,
            proof_type: ProofType::VoteEligibility,
            security_level: SecurityLevel::Standard128,
            public_inputs: public_inputs_bytes,
            proof_bytes,
            compressed: false,
            checksum,
        })
    }

    /// Serialize envelope to bytes
    pub fn to_bytes(&self) -> ProofResult<Vec<u8>> {
        let mut bytes = Vec::new();

        // Magic bytes
        bytes.extend_from_slice(&MAGIC_BYTES);

        // Serialize envelope
        let envelope_bytes = serde_json::to_vec(self)
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        // Length prefix
        bytes.extend_from_slice(&(envelope_bytes.len() as u32).to_le_bytes());

        // Envelope data
        bytes.extend_from_slice(&envelope_bytes);

        Ok(bytes)
    }

    /// Deserialize envelope from bytes
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        if bytes.len() < 8 {
            return Err(ProofError::InvalidProofFormat("Too short".to_string()));
        }

        // Check magic bytes
        if &bytes[0..4] != &MAGIC_BYTES {
            return Err(ProofError::InvalidProofFormat("Invalid magic bytes".to_string()));
        }

        // Read length
        let len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;

        if bytes.len() < 8 + len {
            return Err(ProofError::InvalidProofFormat("Truncated data".to_string()));
        }

        // Deserialize envelope
        let envelope: ProofEnvelope = serde_json::from_slice(&bytes[8..8 + len])
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        // Verify checksum
        let computed_checksum = compute_checksum(&envelope.proof_bytes);
        if computed_checksum != envelope.checksum {
            return Err(ProofError::InvalidProofFormat("Checksum mismatch".to_string()));
        }

        // Version check
        if envelope.version > SERIALIZATION_VERSION {
            return Err(ProofError::InvalidProofFormat(
                format!("Unsupported version: {}", envelope.version)
            ));
        }

        Ok(envelope)
    }

    /// Get the proof type
    pub fn proof_type(&self) -> ProofType {
        self.proof_type
    }

    /// Get the raw proof bytes
    pub fn proof_bytes(&self) -> &[u8] {
        &self.proof_bytes
    }

    /// Get the size of the serialized envelope
    pub fn size(&self) -> usize {
        8 + // Magic + length
        self.public_inputs.len() +
        self.proof_bytes.len() +
        32 // Overhead
    }

    /// Compress the proof bytes using zstd (requires `proofs-compressed` feature)
    ///
    /// Returns a new envelope with compressed proof bytes.
    /// Typically achieves 40-60% size reduction.
    #[cfg(feature = "proofs-compressed")]
    pub fn compress(&self) -> ProofResult<Self> {
        if self.compressed {
            return Ok(self.clone()); // Already compressed
        }

        let compressed_bytes = zstd::encode_all(self.proof_bytes.as_slice(), 3)
            .map_err(|e| ProofError::SerializationError(format!("Compression failed: {}", e)))?;

        Ok(Self {
            version: self.version,
            proof_type: self.proof_type,
            security_level: self.security_level,
            public_inputs: self.public_inputs.clone(),
            proof_bytes: compressed_bytes,
            compressed: true,
            checksum: self.checksum, // Keep original checksum for verification after decompression
        })
    }

    /// Decompress the proof bytes (requires `proofs-compressed` feature)
    ///
    /// Returns a new envelope with decompressed proof bytes.
    #[cfg(feature = "proofs-compressed")]
    pub fn decompress(&self) -> ProofResult<Self> {
        if !self.compressed {
            return Ok(self.clone()); // Already decompressed
        }

        let decompressed_bytes = zstd::decode_all(self.proof_bytes.as_slice())
            .map_err(|e| ProofError::SerializationError(format!("Decompression failed: {}", e)))?;

        // Verify checksum matches the decompressed data
        let computed_checksum = compute_checksum(&decompressed_bytes);
        if computed_checksum != self.checksum {
            return Err(ProofError::InvalidProofFormat("Checksum mismatch after decompression".to_string()));
        }

        Ok(Self {
            version: self.version,
            proof_type: self.proof_type,
            security_level: self.security_level,
            public_inputs: self.public_inputs.clone(),
            proof_bytes: decompressed_bytes,
            compressed: false,
            checksum: self.checksum,
        })
    }

    /// Get compression ratio (compressed size / original size)
    ///
    /// Returns None if the proof isn't compressed.
    #[cfg(feature = "proofs-compressed")]
    pub fn compression_ratio(&self, original_size: usize) -> Option<f32> {
        if self.compressed && original_size > 0 {
            Some(self.proof_bytes.len() as f32 / original_size as f32)
        } else {
            None
        }
    }

    /// Check if the envelope is compressed
    pub fn is_compressed(&self) -> bool {
        self.compressed
    }
}

/// Serializable version of gradient public inputs
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializableGradientInputs {
    num_elements: usize,
    max_norm_squared: u64,
    commitment: [u8; 32],
    final_norm_squared: u64,
}

impl From<GradientPublicInputs> for SerializableGradientInputs {
    fn from(inputs: GradientPublicInputs) -> Self {
        Self {
            num_elements: inputs.num_elements,
            max_norm_squared: inputs.max_norm_squared,
            commitment: inputs.commitment,
            final_norm_squared: inputs.final_norm_squared,
        }
    }
}

/// Serializable version of identity public inputs
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializableIdentityInputs {
    identity_commitment: [u8; 32],
    min_level: u8,
    meets_threshold: bool,
    final_score: u64,
}

impl From<IdentityPublicInputs> for SerializableIdentityInputs {
    fn from(inputs: IdentityPublicInputs) -> Self {
        Self {
            identity_commitment: inputs.identity_commitment,
            min_level: inputs.min_level,
            meets_threshold: inputs.meets_threshold,
            final_score: inputs.final_score,
        }
    }
}

/// Serializable version of vote public inputs
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializableVoteInputs {
    voter_commitment: [u8; 32],
    proposal_type: u8,
    eligible: bool,
    requirements_met: u8,
    active_requirements: u8,
}

impl From<VotePublicInputs> for SerializableVoteInputs {
    fn from(inputs: VotePublicInputs) -> Self {
        Self {
            voter_commitment: inputs.voter_commitment,
            proposal_type: inputs.proposal_type,
            eligible: inputs.eligible,
            requirements_met: inputs.requirements_met,
            active_requirements: inputs.active_requirements,
        }
    }
}

/// Compute a simple checksum for integrity verification
fn compute_checksum(data: &[u8]) -> [u8; 4] {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let hash = hasher.finish();

    [
        (hash >> 24) as u8,
        (hash >> 16) as u8,
        (hash >> 8) as u8,
        hash as u8,
    ]
}

/// Batch serialization for multiple proofs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofBatch {
    /// Batch version
    pub version: u8,

    /// Number of proofs
    pub count: usize,

    /// Individual envelopes
    pub envelopes: Vec<ProofEnvelope>,

    /// Batch checksum
    pub batch_checksum: [u8; 4],
}

impl ProofBatch {
    /// Create a new empty batch
    pub fn new() -> Self {
        Self {
            version: SERIALIZATION_VERSION,
            count: 0,
            envelopes: Vec::new(),
            batch_checksum: [0; 4],
        }
    }

    /// Add a proof envelope to the batch
    pub fn add(&mut self, envelope: ProofEnvelope) {
        self.envelopes.push(envelope);
        self.count = self.envelopes.len();
        self.update_checksum();
    }

    /// Serialize the batch
    pub fn to_bytes(&self) -> ProofResult<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| ProofError::SerializationError(e.to_string()))
    }

    /// Deserialize a batch
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        let batch: ProofBatch = serde_json::from_slice(bytes)
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        // Verify checksum
        let mut temp = batch.clone();
        temp.update_checksum();
        if temp.batch_checksum != batch.batch_checksum {
            return Err(ProofError::InvalidProofFormat("Batch checksum mismatch".to_string()));
        }

        Ok(batch)
    }

    /// Update the batch checksum
    fn update_checksum(&mut self) {
        let mut data = Vec::new();
        for envelope in &self.envelopes {
            data.extend_from_slice(&envelope.checksum);
        }
        self.batch_checksum = compute_checksum(&data);
    }

    /// Get total size of the batch
    pub fn total_size(&self) -> usize {
        self.envelopes.iter().map(|e| e.size()).sum()
    }
}

impl Default for ProofBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::{ProofConfig, SecurityLevel};

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_range_proof_serialization() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();

        // Serialize
        let envelope = ProofEnvelope::from_range_proof(&proof).unwrap();
        let bytes = envelope.to_bytes().unwrap();

        // Deserialize
        let restored = ProofEnvelope::from_bytes(&bytes).unwrap();

        assert_eq!(restored.proof_type, ProofType::Range);
        assert_eq!(restored.checksum, envelope.checksum);
    }

    #[test]
    fn test_gradient_proof_serialization() {
        let gradient = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let proof = GradientIntegrityProof::generate(&gradient, 5.0, test_config()).unwrap();

        let envelope = ProofEnvelope::from_gradient_proof(&proof).unwrap();
        let bytes = envelope.to_bytes().unwrap();

        let restored = ProofEnvelope::from_bytes(&bytes).unwrap();
        assert_eq!(restored.proof_type, ProofType::GradientIntegrity);
    }

    #[test]
    fn test_batch_serialization() {
        let mut batch = ProofBatch::new();

        // Add multiple proofs
        let proof1 = RangeProof::generate(25, 0, 50, test_config()).unwrap();
        let proof2 = RangeProof::generate(75, 50, 100, test_config()).unwrap();

        batch.add(ProofEnvelope::from_range_proof(&proof1).unwrap());
        batch.add(ProofEnvelope::from_range_proof(&proof2).unwrap());

        assert_eq!(batch.count, 2);

        // Serialize and deserialize
        let bytes = batch.to_bytes().unwrap();
        let restored = ProofBatch::from_bytes(&bytes).unwrap();

        assert_eq!(restored.count, 2);
        assert_eq!(restored.batch_checksum, batch.batch_checksum);
    }

    #[test]
    fn test_invalid_magic_bytes() {
        let invalid_bytes = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = ProofEnvelope::from_bytes(&invalid_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_data() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC_BYTES);
        bytes.extend_from_slice(&100u32.to_le_bytes()); // Claims 100 bytes
        bytes.extend_from_slice(&[0u8; 10]); // Only 10 bytes

        let result = ProofEnvelope::from_bytes(&bytes);
        assert!(result.is_err());
    }
}
