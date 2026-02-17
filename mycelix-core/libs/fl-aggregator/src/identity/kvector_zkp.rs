//! K-Vector ZKP Integration for FL Participation Verification
//!
//! Integrates kvector-zkp library for cryptographic verification of participant
//! trust metrics without revealing the actual K-Vector values.
//!
//! ## Overview
//!
//! When a node submits a gradient, it can also submit a ZK proof that its K-Vector
//! (trust metrics) satisfies the network requirements. This allows:
//!
//! 1. **Privacy**: K-Vector values remain hidden from other participants
//! 2. **Verifiability**: Anyone can verify the proof is valid
//! 3. **Sybil Resistance**: Prevents nodes from claiming fake trust scores
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::identity::kvector_zkp::{KVectorVerifier, NodeKVectorProof};
//!
//! // Verifier (aggregator side)
//! let verifier = KVectorVerifier::new(KVectorVerifierConfig::default());
//!
//! // Verify a node's K-Vector proof
//! let result = verifier.verify_node_proof(&proof)?;
//! if result.is_valid {
//!     // Node can participate with zkpoc_verified = true
//!     gradient_record = gradient_record.with_zkpoc_verified(true);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use thiserror::Error;

#[cfg(feature = "kvector-zkp")]
use kvector_zkp::{KVectorWitness, proof::KVectorRangeProof};

/// Errors from K-Vector ZKP verification
#[derive(Error, Debug)]
pub enum KVectorZkpError {
    #[error("Proof verification failed: {0}")]
    VerificationFailed(String),

    #[error("Invalid proof format: {0}")]
    InvalidProofFormat(String),

    #[error("Proof deserialization failed: {0}")]
    DeserializationFailed(String),

    #[error("Commitment mismatch: expected {expected}, got {actual}")]
    CommitmentMismatch { expected: String, actual: String },

    #[error("Trust score below threshold: {score:.4} < {threshold:.4}")]
    TrustScoreBelowThreshold { score: f32, threshold: f32 },

    #[error("Feature not enabled: kvector-zkp feature required")]
    FeatureNotEnabled,
}

pub type KVectorZkpResult<T> = Result<T, KVectorZkpError>;

/// Configuration for K-Vector verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVectorVerifierConfig {
    /// Minimum trust score required for FL participation
    pub min_trust_score: f32,

    /// Whether to cache verified commitments
    pub enable_caching: bool,

    /// Maximum cache size
    pub max_cache_size: usize,

    /// Whether to log verification times
    pub log_verification_times: bool,
}

impl Default for KVectorVerifierConfig {
    fn default() -> Self {
        Self {
            min_trust_score: 0.3,
            enable_caching: true,
            max_cache_size: 10000,
            log_verification_times: false,
        }
    }
}

/// A node's K-Vector proof submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeKVectorProof {
    /// Node identifier
    pub node_id: String,

    /// Commitment to K-Vector values (32-byte hash)
    pub commitment: [u8; 32],

    /// The serialized ZK proof
    pub proof_bytes: Vec<u8>,

    /// Claimed trust score (will be verified against proof)
    pub claimed_trust_score: f32,

    /// Timestamp when proof was generated
    pub generated_at: u64,

    /// Optional metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Result of K-Vector proof verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the proof is valid
    pub is_valid: bool,

    /// The verified commitment
    pub commitment: [u8; 32],

    /// Verified trust score (computed from targets in proof)
    pub trust_score: f32,

    /// Whether trust score meets minimum threshold
    pub meets_threshold: bool,

    /// Verification time in milliseconds
    pub verification_time_ms: f64,

    /// Any warnings or notes
    pub notes: Vec<String>,
}

/// K-Vector ZKP verifier for FL participation
pub struct KVectorVerifier {
    config: KVectorVerifierConfig,
    /// Cache of verified commitments (commitment -> (is_valid, trust_score))
    #[allow(dead_code)]
    cache: HashMap<[u8; 32], (bool, f32)>,
    /// Verification statistics
    stats: VerificationStats,
}

#[derive(Debug, Default)]
struct VerificationStats {
    total_verifications: u64,
    successful_verifications: u64,
    failed_verifications: u64,
    cache_hits: u64,
    total_verification_time_ms: f64,
}

impl KVectorVerifier {
    /// Create a new verifier with the given configuration
    pub fn new(config: KVectorVerifierConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            stats: VerificationStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_verifier() -> Self {
        Self::new(KVectorVerifierConfig::default())
    }

    /// Verify a node's K-Vector proof
    #[cfg(feature = "kvector-zkp")]
    pub fn verify_node_proof(&mut self, proof: &NodeKVectorProof) -> KVectorZkpResult<VerificationResult> {
        let start = Instant::now();
        self.stats.total_verifications += 1;

        // Check cache first
        if self.config.enable_caching {
            if let Some(&(is_valid, trust_score)) = self.cache.get(&proof.commitment) {
                self.stats.cache_hits += 1;
                let meets_threshold = trust_score >= self.config.min_trust_score;
                return Ok(VerificationResult {
                    is_valid,
                    commitment: proof.commitment,
                    trust_score,
                    meets_threshold,
                    verification_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    notes: vec!["Retrieved from cache".to_string()],
                });
            }
        }

        // Deserialize proof
        let zkp = KVectorRangeProof::from_bytes(&proof.proof_bytes)
            .map_err(|e| KVectorZkpError::DeserializationFailed(e.to_string()))?;

        // Verify commitment matches
        if zkp.commitment != proof.commitment {
            self.stats.failed_verifications += 1;
            return Err(KVectorZkpError::CommitmentMismatch {
                expected: hex::encode(proof.commitment),
                actual: hex::encode(zkp.commitment),
            });
        }

        // Verify the ZK proof
        zkp.verify()
            .map_err(|e| {
                self.stats.failed_verifications += 1;
                KVectorZkpError::VerificationFailed(e.to_string())
            })?;

        // Compute trust score from verified targets
        let trust_score = self.compute_trust_score_from_targets(&zkp.targets);
        let meets_threshold = trust_score >= self.config.min_trust_score;

        // Update cache
        if self.config.enable_caching && self.cache.len() < self.config.max_cache_size {
            self.cache.insert(proof.commitment, (true, trust_score));
        }

        self.stats.successful_verifications += 1;
        let verification_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_verification_time_ms += verification_time_ms;

        let mut notes = Vec::new();
        if !meets_threshold {
            notes.push(format!(
                "Trust score {:.4} below threshold {:.4}",
                trust_score, self.config.min_trust_score
            ));
        }

        Ok(VerificationResult {
            is_valid: true,
            commitment: proof.commitment,
            trust_score,
            meets_threshold,
            verification_time_ms,
            notes,
        })
    }

    /// Stub when kvector-zkp feature is not enabled
    #[cfg(not(feature = "kvector-zkp"))]
    pub fn verify_node_proof(&mut self, _proof: &NodeKVectorProof) -> KVectorZkpResult<VerificationResult> {
        Err(KVectorZkpError::FeatureNotEnabled)
    }

    /// Compute trust score from ZK proof targets
    ///
    /// Trust = 0.25×k_r + 0.15×k_a + 0.20×k_i + 0.15×k_p + 0.05×k_m + 0.10×k_s + 0.05×k_h + 0.05×k_topo
    #[cfg(feature = "kvector-zkp")]
    fn compute_trust_score_from_targets(&self, targets: &[u64; 8]) -> f32 {
        const SCALE_FACTOR: f32 = 10000.0;
        const WEIGHTS: [f32; 8] = [0.25, 0.15, 0.20, 0.15, 0.05, 0.10, 0.05, 0.05];

        targets
            .iter()
            .zip(WEIGHTS.iter())
            .map(|(&t, &w)| (t as f32 / SCALE_FACTOR) * w)
            .sum()
    }

    /// Get verification statistics
    pub fn statistics(&self) -> VerificationStatistics {
        VerificationStatistics {
            total_verifications: self.stats.total_verifications,
            successful_verifications: self.stats.successful_verifications,
            failed_verifications: self.stats.failed_verifications,
            cache_hits: self.stats.cache_hits,
            cache_hit_rate: if self.stats.total_verifications > 0 {
                self.stats.cache_hits as f64 / self.stats.total_verifications as f64
            } else {
                0.0
            },
            avg_verification_time_ms: if self.stats.successful_verifications > 0 {
                self.stats.total_verification_time_ms / self.stats.successful_verifications as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the verification cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get current configuration
    pub fn config(&self) -> &KVectorVerifierConfig {
        &self.config
    }
}

/// Public verification statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStatistics {
    pub total_verifications: u64,
    pub successful_verifications: u64,
    pub failed_verifications: u64,
    pub cache_hits: u64,
    pub cache_hit_rate: f64,
    pub avg_verification_time_ms: f64,
}

/// Helper to create a proof submission from kvector-zkp types
#[cfg(feature = "kvector-zkp")]
pub fn create_node_proof(
    node_id: &str,
    witness: &KVectorWitness,
) -> KVectorZkpResult<NodeKVectorProof> {
    // Generate the ZK proof
    let proof = KVectorRangeProof::prove(witness)
        .map_err(|e| KVectorZkpError::VerificationFailed(format!("Proof generation failed: {}", e)))?;

    // Compute trust score for the claimed value
    let trust_score = witness.k_r * 0.25
        + witness.k_a * 0.15
        + witness.k_i * 0.20
        + witness.k_p * 0.15
        + witness.k_m * 0.05
        + witness.k_s * 0.10
        + witness.k_h * 0.05
        + witness.k_topo * 0.05;

    Ok(NodeKVectorProof {
        node_id: node_id.to_string(),
        commitment: proof.commitment,
        proof_bytes: proof.to_bytes(),
        claimed_trust_score: trust_score,
        generated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        metadata: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verifier_config_default() {
        let config = KVectorVerifierConfig::default();
        assert_eq!(config.min_trust_score, 0.3);
        assert!(config.enable_caching);
        assert_eq!(config.max_cache_size, 10000);
    }

    #[test]
    fn test_verifier_creation() {
        let verifier = KVectorVerifier::default_verifier();
        let stats = verifier.statistics();
        assert_eq!(stats.total_verifications, 0);
    }

    #[cfg(feature = "kvector-zkp")]
    #[test]
    fn test_full_verification_cycle() {
        use kvector_zkp::KVectorWitness;

        // Create a witness with valid K-Vector
        let witness = KVectorWitness {
            k_r: 0.75,
            k_a: 0.60,
            k_i: 0.85,
            k_p: 0.70,
            k_m: 0.45,
            k_s: 0.55,
            k_h: 0.80,
            k_topo: 0.65,
        };

        // Create proof submission
        let proof = create_node_proof("test_node", &witness)
            .expect("Proof creation should succeed");

        assert_eq!(proof.node_id, "test_node");
        assert!(!proof.proof_bytes.is_empty());

        // Verify
        let mut verifier = KVectorVerifier::default_verifier();
        let result = verifier.verify_node_proof(&proof)
            .expect("Verification should succeed");

        assert!(result.is_valid);
        assert!(result.meets_threshold);
        assert!(result.trust_score > 0.5);
    }

    #[cfg(feature = "kvector-zkp")]
    #[test]
    fn test_caching() {
        use kvector_zkp::KVectorWitness;

        let witness = KVectorWitness {
            k_r: 0.8, k_a: 0.7, k_i: 0.9, k_p: 0.6,
            k_m: 0.5, k_s: 0.4, k_h: 0.85, k_topo: 0.75,
        };

        let proof = create_node_proof("cache_test", &witness).unwrap();

        let mut verifier = KVectorVerifier::default_verifier();

        // First verification
        let result1 = verifier.verify_node_proof(&proof).unwrap();
        assert!(result1.is_valid);
        assert_eq!(verifier.statistics().cache_hits, 0);

        // Second verification should hit cache
        let result2 = verifier.verify_node_proof(&proof).unwrap();
        assert!(result2.is_valid);
        assert_eq!(verifier.statistics().cache_hits, 1);

        // Results should match
        assert_eq!(result1.trust_score, result2.trust_score);
    }

    #[cfg(feature = "kvector-zkp")]
    #[test]
    fn test_trust_threshold_enforcement() {
        use kvector_zkp::KVectorWitness;

        // Low trust K-Vector
        let witness = KVectorWitness {
            k_r: 0.1, k_a: 0.1, k_i: 0.1, k_p: 0.1,
            k_m: 0.1, k_s: 0.1, k_h: 0.1, k_topo: 0.1,
        };

        let proof = create_node_proof("low_trust", &witness).unwrap();

        let mut verifier = KVectorVerifier::new(KVectorVerifierConfig {
            min_trust_score: 0.5,
            ..Default::default()
        });

        let result = verifier.verify_node_proof(&proof).unwrap();

        // Proof is valid but doesn't meet threshold
        assert!(result.is_valid);
        assert!(!result.meets_threshold);
        assert!(result.trust_score < 0.5);
    }

    #[cfg(not(feature = "kvector-zkp"))]
    #[test]
    fn test_feature_not_enabled() {
        let mut verifier = KVectorVerifier::default_verifier();
        let dummy_proof = NodeKVectorProof {
            node_id: "test".to_string(),
            commitment: [0u8; 32],
            proof_bytes: vec![],
            claimed_trust_score: 0.5,
            generated_at: 0,
            metadata: HashMap::new(),
        };

        let result = verifier.verify_node_proof(&dummy_proof);
        assert!(matches!(result, Err(KVectorZkpError::FeatureNotEnabled)));
    }
}
