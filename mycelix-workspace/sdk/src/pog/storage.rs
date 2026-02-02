//! # Storage Infrastructure PoG
//!
//! Decentralized storage verification for Proof of Grounding.

use super::{PogScore, VerificationLevel};
use serde::{Deserialize, Serialize};

/// Storage provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageType {
    /// IPFS/Filecoin pinning
    IpfsFilecoin,
    /// Holochain DHT (native bonus)
    HolochainDht,
    /// Arweave permaweb
    Arweave,
    /// S3-compatible custom
    S3Compatible,
}

impl StorageType {
    /// Get PoG multiplier
    pub fn multiplier(&self) -> f64 {
        match self {
            StorageType::IpfsFilecoin => 1.3,
            StorageType::HolochainDht => 1.4, // Native bonus
            StorageType::Arweave => 1.2,
            StorageType::S3Compatible => 1.1,
        }
    }
}

/// Storage attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageAttestation {
    /// Provider DID
    pub provider_did: String,
    /// Storage commitment in GB
    pub storage_commitment_gb: u64,
    /// Storage type
    pub storage_type: StorageType,
    /// Replication proofs
    pub replication_proofs: Vec<ReplicationProof>,
    /// Last successful challenge response
    pub last_challenge_response: u64,
    /// Availability score (0.0 - 1.0)
    pub availability_score: f64,
    /// Verification level
    pub verification_level: VerificationLevel,
}

/// Proof of data replication/availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationProof {
    /// Challenge ID
    pub challenge_id: String,
    /// Merkle root of stored data
    pub merkle_root: Vec<u8>,
    /// Response time in milliseconds
    pub response_time_ms: u32,
    /// Verification timestamp
    pub verified_at: u64,
}

impl StorageAttestation {
    /// Calculate PoG score for this attestation
    pub fn calculate_pog(&self) -> PogScore {
        const MIN_STORAGE_GB: u64 = 100;

        if self.storage_commitment_gb < MIN_STORAGE_GB {
            return PogScore::zero();
        }

        // Logarithmic scaling (1 TB = ~3x score of 100 GB)
        let storage_factor = (self.storage_commitment_gb as f64 / MIN_STORAGE_GB as f64).ln() + 1.0;

        // Availability multiplier
        let availability = self.availability_score.clamp(0.0, 1.0);

        // Storage type multiplier
        let type_mult = self.storage_type.multiplier();

        // Verification weight
        let verification = self.verification_level.weight();

        let score = (storage_factor * availability * type_mult * verification * 0.15).min(1.0);

        PogScore::new(score)
    }

    /// Check if attestation is fresh (within 24 hours)
    pub fn is_fresh(&self, current_time: u64) -> bool {
        current_time.saturating_sub(self.last_challenge_response) < 24 * 60 * 60
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_pog() {
        let attestation = StorageAttestation {
            provider_did: "did:test:1".to_string(),
            storage_commitment_gb: 500,
            storage_type: StorageType::HolochainDht,
            replication_proofs: vec![],
            last_challenge_response: 1000,
            availability_score: 0.99,
            verification_level: VerificationLevel::OracleVerified,
        };

        let pog = attestation.calculate_pog();
        assert!(pog.value() > 0.0);
    }

    #[test]
    fn test_minimum_storage() {
        let attestation = StorageAttestation {
            provider_did: "did:test:1".to_string(),
            storage_commitment_gb: 50, // Below minimum
            storage_type: StorageType::IpfsFilecoin,
            replication_proofs: vec![],
            last_challenge_response: 1000,
            availability_score: 0.99,
            verification_level: VerificationLevel::HardwareAttested,
        };

        let pog = attestation.calculate_pog();
        assert!((pog.value() - 0.0).abs() < 0.001);
    }
}
