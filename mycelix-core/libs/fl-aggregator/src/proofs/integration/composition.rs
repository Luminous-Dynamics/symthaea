//! Proof Composition
//!
//! Lightweight alternative to recursive STARKs for combining multiple proofs.
//!
//! # Recursive STARKs: Considerations
//!
//! ## What Are Recursive STARKs?
//!
//! Recursive STARKs allow proving that a STARK proof is valid within another STARK proof.
//! This enables:
//!
//! - **Proof aggregation**: Combine N proofs into 1 proof of constant size
//! - **Incremental computation**: Prove long computations in steps
//! - **Proof compression**: Reduce verification cost for many proofs
//!
//! ## Why We Don't Use Full Recursion (Yet)
//!
//! 1. **Complexity**: Implementing a STARK verifier as an AIR circuit is complex
//!    - Must implement field arithmetic in constraints
//!    - Must implement polynomial evaluation and FRI verification
//!    - Trace size grows significantly (millions of rows)
//!
//! 2. **Performance**: Recursive proof generation is expensive
//!    - Inner proof verification adds ~1M constraints
//!    - Generation time increases 10-100x
//!    - Memory requirements grow substantially
//!
//! 3. **Winterfell Limitations**: No native recursion support
//!    - Would need custom STARK-friendly hash (Rescue/Poseidon)
//!    - Would need to implement verifier logic as AIR
//!
//! ## Our Approach: Proof Composition
//!
//! Instead of full recursion, we use **commitment-based composition**:
//!
//! 1. Generate individual proofs normally
//! 2. Create a Merkle tree over proof commitments
//! 3. Sign/attest to the Merkle root
//! 4. Verify: check proofs + Merkle membership + signature
//!
//! This provides:
//! - Efficient batching (verify proofs in parallel)
//! - Compact attestation (single signature over root)
//! - Incremental updates (add proofs to tree)
//!
//! ## Future: True Recursion
//!
//! When Winterfell adds recursion support, or we migrate to a recursive-native
//! system (Plonky2, Nova), we can implement:
//!
//! ```text
//! InnerProof₁ ──┐
//! InnerProof₂ ──┼──► RecursiveProof ──► SingleVerification
//! InnerProof₃ ──┘
//! ```
//!
//! For now, our composition achieves similar practical benefits without the
//! implementation complexity.
//!
//! # Post-Quantum Cryptography
//!
//! With the `proofs-pq` feature, attestations can use CRYSTALS-Dilithium
//! (NIST FIPS 204) for post-quantum security. This is recommended for
//! applications requiring long-term security against quantum attacks.

use crate::proofs::{
    ProofError, ProofResult, ProofType, VerificationResult,
    build_merkle_tree, MerklePathNode,
};
use super::serialization::ProofEnvelope;
use blake3::Hasher;
use ed25519_dalek::{Signature, SigningKey, VerifyingKey, Signer, Verifier};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// Post-quantum cryptography (optional)
#[cfg(feature = "proofs-pq")]
use pqcrypto_dilithium::dilithium3;
#[cfg(feature = "proofs-pq")]
use pqcrypto_traits::sign::{
    PublicKey as PqPublicKey,
    SignedMessage as PqSignedMessage,
};

/// Composed proof structure
///
/// Combines multiple proof envelopes with a Merkle commitment and attestation.
#[derive(Clone)]
pub struct ComposedProof {
    /// Individual proof envelopes
    proofs: Vec<ProofEnvelope>,

    /// Merkle root over proof commitments
    merkle_root: [u8; 32],

    /// Merkle paths for each proof
    merkle_paths: Vec<Vec<MerklePathNode>>,

    /// Attestation signature over the Merkle root (Ed25519 - classical)
    attestation: Option<CompositionAttestation>,

    /// Post-quantum attestation (Dilithium - quantum-resistant)
    #[cfg(feature = "proofs-pq")]
    pq_attestation: Option<PqAttestation>,

    /// Composition metadata
    metadata: CompositionMetadata,
}

/// Attestation for composed proofs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositionAttestation {
    /// Signer's public key
    pub public_key: [u8; 32],

    /// Signature over (merkle_root || metadata_hash) - stored as Vec for serde
    #[serde(with = "signature_bytes")]
    pub signature: [u8; 64],

    /// Unix timestamp
    pub timestamp: u64,

    /// Optional context string
    pub context: Option<String>,
}

/// Custom serde module for [u8; 64] signature
mod signature_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        bytes.to_vec().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<u8>::deserialize(deserializer)?;
        if vec.len() != 64 {
            return Err(serde::de::Error::custom("signature must be 64 bytes"));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&vec);
        Ok(arr)
    }
}

/// Post-quantum attestation for composed proofs using CRYSTALS-Dilithium
#[cfg(feature = "proofs-pq")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PqAttestation {
    /// Signer's public key (Dilithium3: 1952 bytes)
    pub public_key: Vec<u8>,

    /// Signature over (merkle_root || metadata_hash) (Dilithium3: 3293 bytes)
    pub signature: Vec<u8>,

    /// Unix timestamp
    pub timestamp: u64,

    /// Optional context string
    pub context: Option<String>,

    /// Algorithm identifier for forward compatibility
    pub algorithm: PqAlgorithm,
}

/// Post-quantum algorithm identifier
#[cfg(feature = "proofs-pq")]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PqAlgorithm {
    /// CRYSTALS-Dilithium Level 3 (NIST FIPS 204)
    Dilithium3,
}

/// Metadata about the composition
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompositionMetadata {
    /// Number of proofs
    pub proof_count: usize,

    /// Proof types included
    pub proof_types: Vec<ProofType>,

    /// Total proof size in bytes
    pub total_proof_bytes: usize,

    /// Composition timestamp
    pub created_at: u64,

    /// Optional round ID for FL
    pub round_id: Option<u64>,

    /// Optional batch ID
    pub batch_id: Option<String>,
}

impl ComposedProof {
    /// Create a new composed proof from envelopes
    pub fn compose(proofs: Vec<ProofEnvelope>) -> ProofResult<Self> {
        if proofs.is_empty() {
            return Err(ProofError::InvalidPublicInputs(
                "Cannot compose empty proof list".to_string()
            ));
        }

        // Create commitment for each proof (hash of proof bytes)
        let commitments: Vec<[u8; 32]> = proofs
            .iter()
            .map(|p| {
                let mut hasher = Hasher::new();
                hasher.update(&p.proof_bytes);
                hasher.update(&p.public_inputs);
                *hasher.finalize().as_bytes()
            })
            .collect();

        // Build Merkle tree
        let (merkle_root, merkle_paths) = build_merkle_tree(&commitments);

        // Collect metadata
        let metadata = CompositionMetadata {
            proof_count: proofs.len(),
            proof_types: proofs.iter().map(|p| p.proof_type).collect(),
            total_proof_bytes: proofs.iter().map(|p| p.proof_bytes.len()).sum(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            round_id: None,
            batch_id: None,
        };

        Ok(Self {
            proofs,
            merkle_root,
            merkle_paths,
            attestation: None,
            #[cfg(feature = "proofs-pq")]
            pq_attestation: None,
            metadata,
        })
    }

    /// Add Ed25519 attestation signature (classical cryptography)
    ///
    /// For post-quantum security, use `attest_pq` with the `proofs-pq` feature.
    pub fn attest(&mut self, signing_key: &SigningKey, context: Option<String>) -> ProofResult<()> {
        // Create message: merkle_root || metadata_hash
        let metadata_bytes = serde_json::to_vec(&self.metadata)
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        let mut hasher = Hasher::new();
        hasher.update(&self.merkle_root);
        hasher.update(&metadata_bytes);
        let message = hasher.finalize();

        // Sign
        let signature = signing_key.sign(message.as_bytes());

        self.attestation = Some(CompositionAttestation {
            public_key: signing_key.verifying_key().to_bytes(),
            signature: signature.to_bytes(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            context,
        });

        Ok(())
    }

    /// Add post-quantum attestation signature using CRYSTALS-Dilithium
    ///
    /// This provides quantum-resistant security (NIST FIPS 204).
    /// Signatures are larger (~3KB) but secure against quantum attacks.
    #[cfg(feature = "proofs-pq")]
    pub fn attest_pq(
        &mut self,
        secret_key: &dilithium3::SecretKey,
        public_key: &dilithium3::PublicKey,
        context: Option<String>,
    ) -> ProofResult<()> {
        // Create message: merkle_root || metadata_hash
        let metadata_bytes = serde_json::to_vec(&self.metadata)
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        let mut hasher = Hasher::new();
        hasher.update(&self.merkle_root);
        hasher.update(&metadata_bytes);
        let message = hasher.finalize();

        // Sign with Dilithium
        let signed_msg = dilithium3::sign(message.as_bytes(), secret_key);
        let signature_bytes = signed_msg.as_bytes().to_vec();

        self.pq_attestation = Some(PqAttestation {
            public_key: public_key.as_bytes().to_vec(),
            signature: signature_bytes,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            context,
            algorithm: PqAlgorithm::Dilithium3,
        });

        Ok(())
    }

    /// Verify the composed proof structure
    ///
    /// This verifies:
    /// 1. Merkle root is correctly computed
    /// 2. Each proof's Merkle path is valid
    /// 3. Attestation signature is valid (if present)
    /// 4. PQ attestation signature is valid (if present and feature enabled)
    ///
    /// Note: This does NOT verify the individual proofs themselves.
    /// Call `verify_all_proofs()` for full verification.
    pub fn verify_structure(&self) -> ProofResult<CompositionVerificationResult> {
        let start = Instant::now();
        let mut issues = Vec::new();

        // Recompute commitments
        let commitments: Vec<[u8; 32]> = self.proofs
            .iter()
            .map(|p| {
                let mut hasher = Hasher::new();
                hasher.update(&p.proof_bytes);
                hasher.update(&p.public_inputs);
                *hasher.finalize().as_bytes()
            })
            .collect();

        // Verify Merkle root
        let (computed_root, _) = build_merkle_tree(&commitments);
        if computed_root != self.merkle_root {
            issues.push("Merkle root mismatch".to_string());
        }

        // Verify each path
        for (i, (commitment, path)) in commitments.iter().zip(&self.merkle_paths).enumerate() {
            if !verify_merkle_path(commitment, path, &self.merkle_root) {
                issues.push(format!("Invalid Merkle path for proof {}", i));
            }
        }

        // Verify Ed25519 attestation if present
        let attestation_valid = if let Some(ref attestation) = self.attestation {
            let metadata_bytes = serde_json::to_vec(&self.metadata)
                .map_err(|e| ProofError::SerializationError(e.to_string()))?;

            let mut hasher = Hasher::new();
            hasher.update(&self.merkle_root);
            hasher.update(&metadata_bytes);
            let message = hasher.finalize();

            let public_key = VerifyingKey::from_bytes(&attestation.public_key)
                .map_err(|e| ProofError::VerificationFailed(e.to_string()))?;

            let signature = Signature::from_bytes(&attestation.signature);

            match public_key.verify(message.as_bytes(), &signature) {
                Ok(_) => true,
                Err(_) => {
                    issues.push("Invalid Ed25519 attestation signature".to_string());
                    false
                }
            }
        } else {
            true // No attestation to verify
        };

        // Verify PQ attestation if present
        #[cfg(feature = "proofs-pq")]
        let pq_attestation_valid = if let Some(ref pq_att) = self.pq_attestation {
            let metadata_bytes = serde_json::to_vec(&self.metadata)
                .map_err(|e| ProofError::SerializationError(e.to_string()))?;

            let mut hasher = Hasher::new();
            hasher.update(&self.merkle_root);
            hasher.update(&metadata_bytes);
            let message = hasher.finalize();

            // Verify based on algorithm
            match pq_att.algorithm {
                PqAlgorithm::Dilithium3 => {
                    let public_key = dilithium3::PublicKey::from_bytes(&pq_att.public_key)
                        .map_err(|_| ProofError::VerificationFailed("Invalid Dilithium public key".to_string()))?;

                    // The signature includes the message in pqcrypto format
                    let signed_msg = dilithium3::SignedMessage::from_bytes(&pq_att.signature)
                        .map_err(|_| ProofError::VerificationFailed("Invalid Dilithium signature format".to_string()))?;

                    match dilithium3::open(&signed_msg, &public_key) {
                        Ok(verified_msg) => {
                            // Verify the message matches
                            if verified_msg == message.as_bytes() {
                                true
                            } else {
                                issues.push("PQ attestation message mismatch".to_string());
                                false
                            }
                        }
                        Err(_) => {
                            issues.push("Invalid PQ (Dilithium) attestation signature".to_string());
                            false
                        }
                    }
                }
            }
        } else {
            true // No PQ attestation to verify
        };

        #[cfg(not(feature = "proofs-pq"))]
        let pq_attestation_valid = true;

        let verification_time = start.elapsed();

        Ok(CompositionVerificationResult {
            structure_valid: issues.is_empty(),
            attestation_valid: attestation_valid && pq_attestation_valid,
            proof_count: self.proofs.len(),
            verification_time,
            issues,
        })
    }

    /// Verify all individual proofs
    pub fn verify_all_proofs(&self) -> ProofResult<Vec<(usize, VerificationResult)>> {
        // Note: This would require deserializing and verifying each proof
        // For now, we return a placeholder indicating proofs need external verification
        Err(ProofError::VerificationFailed(
            "Individual proof verification requires proof type-specific logic. \
             Use verify_structure() for composition verification, then verify \
             individual proofs externally.".to_string()
        ))
    }

    /// Get the Merkle root
    pub fn merkle_root(&self) -> [u8; 32] {
        self.merkle_root
    }

    /// Get the number of proofs
    pub fn proof_count(&self) -> usize {
        self.proofs.len()
    }

    /// Get the metadata
    pub fn metadata(&self) -> &CompositionMetadata {
        &self.metadata
    }

    /// Set round ID
    pub fn with_round_id(mut self, round_id: u64) -> Self {
        self.metadata.round_id = Some(round_id);
        self
    }

    /// Set batch ID
    pub fn with_batch_id(mut self, batch_id: impl Into<String>) -> Self {
        self.metadata.batch_id = Some(batch_id.into());
        self
    }

    /// Get proof at index
    pub fn get_proof(&self, index: usize) -> Option<&ProofEnvelope> {
        self.proofs.get(index)
    }

    /// Get Merkle proof for a specific proof
    /// Returns the Merkle root and path for the proof at the given index
    pub fn get_merkle_proof(&self, index: usize) -> Option<(&[u8; 32], &[MerklePathNode])> {
        if index >= self.proofs.len() {
            return None;
        }

        // Verify path exists and return root + path
        // Note: Commitment is computed on-the-fly, not stored
        self.merkle_paths.get(index).map(|path| {
            (&self.merkle_root, path.as_slice())
        })
    }

    /// Check if Ed25519 attestation is present
    pub fn has_attestation(&self) -> bool {
        self.attestation.is_some()
    }

    /// Get Ed25519 attestation if present
    pub fn attestation(&self) -> Option<&CompositionAttestation> {
        self.attestation.as_ref()
    }

    /// Check if post-quantum attestation is present
    #[cfg(feature = "proofs-pq")]
    pub fn has_pq_attestation(&self) -> bool {
        self.pq_attestation.is_some()
    }

    /// Get post-quantum attestation if present
    #[cfg(feature = "proofs-pq")]
    pub fn pq_attestation(&self) -> Option<&PqAttestation> {
        self.pq_attestation.as_ref()
    }

    /// Check if any attestation (classical or PQ) is present
    pub fn has_any_attestation(&self) -> bool {
        #[cfg(feature = "proofs-pq")]
        {
            self.attestation.is_some() || self.pq_attestation.is_some()
        }
        #[cfg(not(feature = "proofs-pq"))]
        {
            self.attestation.is_some()
        }
    }
}

/// Result of composition verification
#[derive(Clone, Debug)]
pub struct CompositionVerificationResult {
    /// Whether the Merkle structure is valid
    pub structure_valid: bool,

    /// Whether the attestation is valid (true if no attestation)
    pub attestation_valid: bool,

    /// Number of proofs in composition
    pub proof_count: usize,

    /// Time taken for verification
    pub verification_time: std::time::Duration,

    /// List of issues found
    pub issues: Vec<String>,
}

impl CompositionVerificationResult {
    /// Check if verification passed
    pub fn is_valid(&self) -> bool {
        self.structure_valid && self.attestation_valid
    }
}

/// Verify a Merkle path (matches build_merkle_tree's blake3 hashing)
fn verify_merkle_path(leaf: &[u8; 32], path: &[MerklePathNode], root: &[u8; 32]) -> bool {
    let mut current = *leaf;

    for node in path {
        let mut hasher = Hasher::new();
        if node.position {
            hasher.update(&node.sibling);
            hasher.update(&current);
        } else {
            hasher.update(&current);
            hasher.update(&node.sibling);
        }
        current = *hasher.finalize().as_bytes();
    }

    current == *root
}

/// Builder for composed proofs
pub struct CompositionBuilder {
    proofs: Vec<ProofEnvelope>,
    round_id: Option<u64>,
    batch_id: Option<String>,
}

impl CompositionBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            proofs: Vec::new(),
            round_id: None,
            batch_id: None,
        }
    }

    /// Add a proof envelope
    pub fn add_proof(mut self, envelope: ProofEnvelope) -> Self {
        self.proofs.push(envelope);
        self
    }

    /// Add multiple proof envelopes
    pub fn add_proofs(mut self, envelopes: impl IntoIterator<Item = ProofEnvelope>) -> Self {
        self.proofs.extend(envelopes);
        self
    }

    /// Set round ID
    pub fn with_round_id(mut self, round_id: u64) -> Self {
        self.round_id = Some(round_id);
        self
    }

    /// Set batch ID
    pub fn with_batch_id(mut self, batch_id: impl Into<String>) -> Self {
        self.batch_id = Some(batch_id.into());
        self
    }

    /// Build the composed proof
    pub fn build(self) -> ProofResult<ComposedProof> {
        let mut composed = ComposedProof::compose(self.proofs)?;

        if let Some(round_id) = self.round_id {
            composed.metadata.round_id = Some(round_id);
        }

        if let Some(batch_id) = self.batch_id {
            composed.metadata.batch_id = Some(batch_id);
        }

        Ok(composed)
    }
}

impl Default for CompositionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::{RangeProof, ProofConfig, SecurityLevel};
    use rand::rngs::OsRng;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    fn make_envelope() -> ProofEnvelope {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        ProofEnvelope::from_range_proof(&proof).unwrap()
    }

    #[test]
    fn test_compose_single_proof() {
        let envelope = make_envelope();
        let composed = ComposedProof::compose(vec![envelope]).unwrap();

        assert_eq!(composed.proof_count(), 1);
        assert!(!composed.merkle_root().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_compose_multiple_proofs() {
        let envelopes = vec![
            make_envelope(),
            make_envelope(),
            make_envelope(),
        ];

        let composed = ComposedProof::compose(envelopes).unwrap();

        assert_eq!(composed.proof_count(), 3);
        assert_eq!(composed.metadata().proof_count, 3);
    }

    #[test]
    fn test_verify_structure() {
        let envelopes = vec![make_envelope(), make_envelope()];
        let composed = ComposedProof::compose(envelopes).unwrap();

        let result = composed.verify_structure().unwrap();

        assert!(result.is_valid());
        assert!(result.structure_valid);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_attestation() {
        use rand::RngCore;

        let envelopes = vec![make_envelope()];
        let mut composed = ComposedProof::compose(envelopes).unwrap();

        // Generate a signing key from random bytes
        let mut secret_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut secret_bytes);
        let signing_key = SigningKey::from_bytes(&secret_bytes);

        // Attest
        composed.attest(&signing_key, Some("test context".to_string())).unwrap();

        assert!(composed.has_attestation());

        // Verify
        let result = composed.verify_structure().unwrap();
        assert!(result.attestation_valid);
    }

    #[test]
    fn test_builder() {
        let composed = CompositionBuilder::new()
            .add_proof(make_envelope())
            .add_proof(make_envelope())
            .with_round_id(42)
            .with_batch_id("test_batch")
            .build()
            .unwrap();

        assert_eq!(composed.proof_count(), 2);
        assert_eq!(composed.metadata().round_id, Some(42));
        assert_eq!(composed.metadata().batch_id, Some("test_batch".to_string()));
    }

    #[test]
    fn test_empty_composition_fails() {
        let result = ComposedProof::compose(vec![]);
        assert!(result.is_err());
    }

    /// Test post-quantum attestation with Dilithium
    #[cfg(feature = "proofs-pq")]
    #[test]
    fn test_pq_attestation() {
        let envelopes = vec![make_envelope()];
        let mut composed = ComposedProof::compose(envelopes).unwrap();

        // Generate Dilithium keypair
        let (public_key, secret_key) = dilithium3::keypair();

        // Attest with PQ signature
        composed.attest_pq(&secret_key, &public_key, Some("PQ test context".to_string())).unwrap();

        assert!(composed.has_pq_attestation());
        assert!(composed.has_any_attestation());

        // Verify
        let result = composed.verify_structure().unwrap();
        assert!(result.attestation_valid);
        assert!(result.is_valid());

        // Check PQ attestation details
        let pq_att = composed.pq_attestation().unwrap();
        assert_eq!(pq_att.algorithm, PqAlgorithm::Dilithium3);
        assert_eq!(pq_att.context, Some("PQ test context".to_string()));
        // Dilithium3 signature is ~3293 bytes
        assert!(pq_att.signature.len() > 3000);
        // Dilithium3 public key is 1952 bytes
        assert_eq!(pq_att.public_key.len(), 1952);
    }

    /// Test hybrid attestation (both Ed25519 and Dilithium)
    #[cfg(feature = "proofs-pq")]
    #[test]
    fn test_hybrid_attestation() {
        use rand::RngCore;

        let envelopes = vec![make_envelope()];
        let mut composed = ComposedProof::compose(envelopes).unwrap();

        // Add classical Ed25519 attestation
        let mut secret_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut secret_bytes);
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        composed.attest(&signing_key, Some("Ed25519 context".to_string())).unwrap();

        // Add PQ Dilithium attestation
        let (public_key, secret_key) = dilithium3::keypair();
        composed.attest_pq(&secret_key, &public_key, Some("Dilithium context".to_string())).unwrap();

        // Both attestations present
        assert!(composed.has_attestation());
        assert!(composed.has_pq_attestation());

        // Verify both
        let result = composed.verify_structure().unwrap();
        assert!(result.attestation_valid);
        assert!(result.is_valid());
    }
}
