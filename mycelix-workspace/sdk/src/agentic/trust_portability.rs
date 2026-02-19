//! # Trust Portability Protocol
//!
//! Cross-chain trust attestations and K-Vector proofs.
//!
//! ## Features
//!
//! - **Portable K-Vectors**: Carry trust across chain boundaries
//! - **Cross-Chain Attestations**: Verifiable trust claims on other chains
//! - **Bridge Adapters**: Protocol adapters for different chains
//! - **Proof Generation**: Generate portable trust proofs
//!
//! ## Security Model
//!
//! Trust portability uses a "verify, then trust" model:
//! - Proofs are generated on the source chain
//! - Proofs are verified on the destination chain
//! - Trust is discounted based on source chain reputation
//! - Freshness requirements prevent stale trust claims

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::matl::KVector;

// ============================================================================
// Configuration
// ============================================================================

/// Trust portability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortabilityConfig {
    /// Maximum proof age (ms)
    pub max_proof_age_ms: u64,
    /// Trust discount factor for cross-chain (0.0-1.0)
    pub cross_chain_discount: f64,
    /// Minimum source chain reputation
    pub min_source_reputation: f64,
    /// Require ZK proofs
    pub require_zk: bool,
    /// Allow partial K-Vector transfer
    pub allow_partial: bool,
    /// Maximum dimensions to transfer
    pub max_dimensions: usize,
}

impl Default for PortabilityConfig {
    fn default() -> Self {
        Self {
            max_proof_age_ms: 3600_000, // 1 hour
            cross_chain_discount: 0.8,  // 20% discount
            min_source_reputation: 0.5,
            require_zk: true,
            allow_partial: true,
            max_dimensions: 10,
        }
    }
}

// ============================================================================
// Chain Identity
// ============================================================================

/// Chain identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChainId(pub String);

impl ChainId {
    /// Create a new chain identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Chain profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainProfile {
    /// Chain ID
    pub id: ChainId,
    /// Chain name
    pub name: String,
    /// Chain type
    pub chain_type: ChainType,
    /// Reputation score (0.0-1.0)
    pub reputation: f64,
    /// Supported proof types
    pub supported_proofs: Vec<ProofType>,
    /// Bridge contract/address
    pub bridge_address: String,
    /// Last synced block/height
    pub last_synced: u64,
}

/// Types of chains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChainType {
    /// EVM-compatible (Ethereum, Polygon, etc.)
    EVM,
    /// Cosmos SDK
    Cosmos,
    /// Substrate/Polkadot
    Substrate,
    /// Solana
    Solana,
    /// Near Protocol
    Near,
    /// Custom/Unknown
    Custom,
}

/// Supported proof types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofType {
    /// Zero-knowledge proof
    ZK,
    /// Merkle proof
    Merkle,
    /// Multi-signature attestation
    MultiSig,
    /// Light client proof
    LightClient,
    /// Optimistic (fraud proof based)
    Optimistic,
}

// ============================================================================
// Portable Trust
// ============================================================================

/// Portable trust package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableTrust {
    /// Package ID
    pub id: String,
    /// Agent ID (cross-chain identifier)
    pub agent_id: String,
    /// Source chain
    pub source_chain: ChainId,
    /// K-Vector snapshot
    pub kvector: KVector,
    /// Selected dimensions (if partial)
    pub dimensions: Vec<KVectorDimension>,
    /// Trust score at time of export
    pub trust_score: f64,
    /// Proof of trust
    pub proof: TrustProof,
    /// Generation timestamp
    pub timestamp: u64,
    /// Expiry timestamp
    pub expires_at: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// K-Vector dimensions for selective export
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KVectorDimension {
    /// Reputation dimension (k_r).
    Reputation,
    /// Activity dimension (k_a).
    Activity,
    /// Integrity dimension (k_i).
    Integrity,
    /// Performance dimension (k_p).
    Performance,
    /// Margin dimension (k_m).
    Margin,
    /// Stake dimension (k_s).
    Stake,
    /// History dimension (k_h).
    History,
    /// Topology dimension (k_topo).
    Topology,
    /// Validity dimension (k_v).
    Validity,
    /// Composite Phi dimension (k_coherence).
    Phi,
}

/// Trust proof (portable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustProof {
    /// Proof type
    pub proof_type: ProofType,
    /// Serialized proof data
    pub proof_data: Vec<u8>,
    /// Public inputs
    pub public_inputs: Vec<u8>,
    /// Verification key ID
    pub vk_id: String,
    /// Signatures (for multi-sig)
    pub signatures: Vec<ProofSignature>,
}

/// Signature on a proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSignature {
    /// Signer (validator/oracle)
    pub signer: String,
    /// Signature bytes
    pub signature: Vec<u8>,
    /// Signer's stake/weight
    pub weight: f64,
}

// ============================================================================
// Import/Export
// ============================================================================

/// Export result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    /// Portable trust package
    pub package: PortableTrust,
    /// Export transaction hash (on source chain)
    pub tx_hash: String,
    /// Status
    pub status: ExportStatus,
}

/// Export status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportStatus {
    /// Export is pending confirmation.
    Pending,
    /// Export confirmed on source chain.
    Confirmed,
    /// Export failed.
    Failed,
}

/// Import result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    /// Package ID
    pub package_id: String,
    /// Applied trust score (after discount)
    pub applied_trust: f64,
    /// Discount applied
    pub discount: f64,
    /// Import status
    pub status: ImportStatus,
    /// Verification result
    pub verification: VerificationResult,
}

/// Import status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportStatus {
    /// Import verified and applied.
    Verified,
    /// Import pending verification.
    Pending,
    /// Import rejected.
    Rejected,
    /// Import package expired.
    Expired,
}

/// Proof verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Is proof valid?
    pub valid: bool,
    /// Verification method used
    pub method: ProofType,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
    /// Errors if invalid
    pub errors: Vec<String>,
}

// ============================================================================
// Bridge Adapter
// ============================================================================

/// Bridge adapter trait
pub trait BridgeAdapter {
    /// Export trust to this chain
    fn export(
        &self,
        agent_id: &str,
        kvector: &KVector,
        dimensions: &[KVectorDimension],
    ) -> Result<ExportResult, BridgeError>;

    /// Import trust from another chain
    fn import(&self, package: &PortableTrust) -> Result<ImportResult, BridgeError>;

    /// Verify a proof
    fn verify(&self, proof: &TrustProof) -> Result<VerificationResult, BridgeError>;

    /// Get chain profile
    fn chain_profile(&self) -> &ChainProfile;

    /// Check if chain is synced
    fn is_synced(&self) -> bool;
}

/// Bridge errors
#[derive(Debug, Clone)]
pub enum BridgeError {
    /// Target chain is not supported.
    ChainNotSupported(
        /// Chain ID.
        ChainId,
    ),
    /// Proof type not supported by target chain.
    ProofTypeNotSupported(
        /// Proof type.
        ProofType,
    ),
    /// Proof verification failed.
    VerificationFailed(
        /// Error message.
        String,
    ),
    /// Trust package has expired.
    PackageExpired,
    /// Source chain reputation is too low.
    InsufficientReputation {
        /// Required reputation.
        required: f64,
        /// Actual reputation.
        actual: f64,
    },
    /// Network communication error.
    NetworkError(
        /// Error message.
        String,
    ),
    /// Proof data is invalid.
    InvalidProof(
        /// Error message.
        String,
    ),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChainNotSupported(id) => write!(f, "Chain not supported: {}", id.as_str()),
            Self::ProofTypeNotSupported(pt) => write!(f, "Proof type not supported: {:?}", pt),
            Self::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            Self::PackageExpired => write!(f, "Trust package expired"),
            Self::InsufficientReputation { required, actual } => {
                write!(
                    f,
                    "Insufficient reputation: {} required, {} actual",
                    required, actual
                )
            }
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::InvalidProof(msg) => write!(f, "Invalid proof: {}", msg),
        }
    }
}

impl std::error::Error for BridgeError {}

// ============================================================================
// Mock Bridge (for testing)
// ============================================================================

/// Mock bridge adapter for testing
#[derive(Debug)]
#[allow(dead_code)]
pub struct MockBridgeAdapter {
    profile: ChainProfile,
    config: PortabilityConfig,
    exported: Vec<PortableTrust>,
    imported: Vec<String>,
}

impl MockBridgeAdapter {
    /// Create a new mock bridge adapter for testing.
    pub fn new(chain_id: &str, chain_type: ChainType) -> Self {
        Self {
            profile: ChainProfile {
                id: ChainId::new(chain_id),
                name: format!("Mock {}", chain_id),
                chain_type,
                reputation: 0.8,
                supported_proofs: vec![ProofType::ZK, ProofType::MultiSig],
                bridge_address: "0x1234567890abcdef".to_string(),
                last_synced: 0,
            },
            config: PortabilityConfig::default(),
            exported: Vec::new(),
            imported: Vec::new(),
        }
    }
}

impl BridgeAdapter for MockBridgeAdapter {
    fn export(
        &self,
        agent_id: &str,
        kvector: &KVector,
        dimensions: &[KVectorDimension],
    ) -> Result<ExportResult, BridgeError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let package = PortableTrust {
            id: format!("pkg-{}-{}", agent_id, timestamp),
            agent_id: agent_id.to_string(),
            source_chain: self.profile.id.clone(),
            kvector: *kvector,
            dimensions: dimensions.to_vec(),
            trust_score: kvector.trust_score() as f64,
            proof: TrustProof {
                proof_type: ProofType::ZK,
                proof_data: vec![1, 2, 3, 4], // Mock proof
                public_inputs: vec![],
                vk_id: "mock-vk".to_string(),
                signatures: vec![],
            },
            timestamp,
            expires_at: timestamp + self.config.max_proof_age_ms,
            metadata: HashMap::new(),
        };

        Ok(ExportResult {
            package,
            tx_hash: format!("0x{:x}", timestamp),
            status: ExportStatus::Confirmed,
        })
    }

    fn import(&self, package: &PortableTrust) -> Result<ImportResult, BridgeError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Check expiry
        if timestamp > package.expires_at {
            return Err(BridgeError::PackageExpired);
        }

        // Verify proof
        let verification = self.verify(&package.proof)?;

        if !verification.valid {
            return Err(BridgeError::VerificationFailed(
                verification.errors.join(", "),
            ));
        }

        // Apply discount
        let applied_trust = package.trust_score * self.config.cross_chain_discount;

        Ok(ImportResult {
            package_id: package.id.clone(),
            applied_trust,
            discount: 1.0 - self.config.cross_chain_discount,
            status: ImportStatus::Verified,
            verification,
        })
    }

    fn verify(&self, proof: &TrustProof) -> Result<VerificationResult, BridgeError> {
        // Mock verification - just check proof data exists
        let valid = !proof.proof_data.is_empty();

        Ok(VerificationResult {
            valid,
            method: proof.proof_type,
            confidence: if valid { 0.95 } else { 0.0 },
            errors: if valid {
                vec![]
            } else {
                vec!["Empty proof data".to_string()]
            },
        })
    }

    fn chain_profile(&self) -> &ChainProfile {
        &self.profile
    }

    fn is_synced(&self) -> bool {
        true
    }
}

// ============================================================================
// Portability Engine
// ============================================================================

/// Main trust portability engine
#[derive(Debug)]
pub struct PortabilityEngine {
    config: PortabilityConfig,
    /// Registered chains
    chains: HashMap<ChainId, ChainProfile>,
    /// Pending exports
    pending_exports: Vec<ExportResult>,
    /// Completed imports
    imported_trust: HashMap<String, ImportResult>,
    /// Export history
    export_history: Vec<ExportResult>,
}

impl PortabilityEngine {
    /// Create new portability engine
    pub fn new(config: PortabilityConfig) -> Self {
        Self {
            config,
            chains: HashMap::new(),
            pending_exports: Vec::new(),
            imported_trust: HashMap::new(),
            export_history: Vec::new(),
        }
    }

    /// Register a chain
    pub fn register_chain(&mut self, profile: ChainProfile) {
        self.chains.insert(profile.id.clone(), profile);
    }

    /// Get chain by ID
    pub fn get_chain(&self, chain_id: &ChainId) -> Option<&ChainProfile> {
        self.chains.get(chain_id)
    }

    /// Prepare trust for export
    pub fn prepare_export(
        &self,
        agent_id: &str,
        kvector: &KVector,
        target_chain: &ChainId,
        dimensions: Option<Vec<KVectorDimension>>,
    ) -> Result<PortableTrust, BridgeError> {
        let target = self
            .chains
            .get(target_chain)
            .ok_or_else(|| BridgeError::ChainNotSupported(target_chain.clone()))?;

        // Check target chain reputation
        if target.reputation < self.config.min_source_reputation {
            return Err(BridgeError::InsufficientReputation {
                required: self.config.min_source_reputation,
                actual: target.reputation,
            });
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let dims = dimensions.unwrap_or_else(|| {
            vec![
                KVectorDimension::Reputation,
                KVectorDimension::Activity,
                KVectorDimension::Integrity,
                KVectorDimension::Performance,
            ]
        });

        // Generate proof (simplified)
        let proof = self.generate_proof(kvector, &dims)?;

        Ok(PortableTrust {
            id: format!("export-{}-{}", agent_id, timestamp),
            agent_id: agent_id.to_string(),
            source_chain: ChainId::new("mycelix"), // Assuming we're on Mycelix
            kvector: *kvector,
            dimensions: dims,
            trust_score: kvector.trust_score() as f64,
            proof,
            timestamp,
            expires_at: timestamp + self.config.max_proof_age_ms,
            metadata: HashMap::new(),
        })
    }

    fn generate_proof(
        &self,
        kvector: &KVector,
        dimensions: &[KVectorDimension],
    ) -> Result<TrustProof, BridgeError> {
        // Simplified proof generation
        // Real implementation would generate actual ZK proof

        let proof_data = serde_json::to_vec(kvector).unwrap_or_default(); // Serialize K-Vector
        let public_inputs = dimensions.iter().map(|d| *d as u8).collect();

        Ok(TrustProof {
            proof_type: if self.config.require_zk {
                ProofType::ZK
            } else {
                ProofType::MultiSig
            },
            proof_data,
            public_inputs,
            vk_id: "mycelix-trust-v1".to_string(),
            signatures: vec![],
        })
    }

    /// Verify and import trust
    pub fn import_trust(&mut self, package: PortableTrust) -> Result<ImportResult, BridgeError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Check expiry
        if timestamp > package.expires_at {
            return Err(BridgeError::PackageExpired);
        }

        // Get source chain reputation
        let source_reputation = self
            .chains
            .get(&package.source_chain)
            .map(|c| c.reputation)
            .unwrap_or(0.5);

        // Verify proof
        let verification = self.verify_proof(&package.proof)?;

        if !verification.valid {
            return Err(BridgeError::InvalidProof(verification.errors.join(", ")));
        }

        // Calculate discounted trust
        let discount = self.config.cross_chain_discount * source_reputation;
        let applied_trust = package.trust_score * discount;

        let result = ImportResult {
            package_id: package.id.clone(),
            applied_trust,
            discount: 1.0 - discount,
            status: ImportStatus::Verified,
            verification,
        };

        self.imported_trust
            .insert(package.agent_id.clone(), result.clone());

        Ok(result)
    }

    fn verify_proof(&self, proof: &TrustProof) -> Result<VerificationResult, BridgeError> {
        // Simplified verification
        // Real implementation would verify actual ZK/Merkle/MultiSig proof

        let valid = !proof.proof_data.is_empty();

        Ok(VerificationResult {
            valid,
            method: proof.proof_type,
            confidence: if valid { 0.9 } else { 0.0 },
            errors: if valid {
                vec![]
            } else {
                vec!["Invalid proof data".to_string()]
            },
        })
    }

    /// Get imported trust for an agent
    pub fn get_imported_trust(&self, agent_id: &str) -> Option<&ImportResult> {
        self.imported_trust.get(agent_id)
    }

    /// Get total cross-chain trust for an agent
    pub fn total_cross_chain_trust(&self, agent_id: &str) -> f64 {
        self.imported_trust
            .get(agent_id)
            .map(|r| r.applied_trust)
            .unwrap_or(0.0)
    }

    /// Get statistics
    pub fn stats(&self) -> PortabilityStats {
        PortabilityStats {
            registered_chains: self.chains.len(),
            pending_exports: self.pending_exports.len(),
            completed_imports: self.imported_trust.len(),
            total_exports: self.export_history.len(),
        }
    }
}

/// Portability statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortabilityStats {
    /// Number of registered chains.
    pub registered_chains: usize,
    /// Number of pending exports.
    pub pending_exports: usize,
    /// Number of completed imports.
    pub completed_imports: usize,
    /// Total exports processed.
    pub total_exports: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_registration() {
        let mut engine = PortabilityEngine::new(PortabilityConfig::default());

        let profile = ChainProfile {
            id: ChainId::new("ethereum"),
            name: "Ethereum".to_string(),
            chain_type: ChainType::EVM,
            reputation: 0.9,
            supported_proofs: vec![ProofType::ZK, ProofType::Merkle],
            bridge_address: "0x1234".to_string(),
            last_synced: 1000,
        };

        engine.register_chain(profile);

        assert!(engine.get_chain(&ChainId::new("ethereum")).is_some());
    }

    #[test]
    fn test_export_preparation() {
        let mut engine = PortabilityEngine::new(PortabilityConfig::default());

        // Register target chain
        engine.register_chain(ChainProfile {
            id: ChainId::new("polygon"),
            name: "Polygon".to_string(),
            chain_type: ChainType::EVM,
            reputation: 0.8,
            supported_proofs: vec![ProofType::ZK],
            bridge_address: "0x5678".to_string(),
            last_synced: 1000,
        });

        let kvector = KVector::new(0.8, 0.7, 0.9, 0.6, 0.5, 0.4, 0.7, 0.5, 0.8, 0.6);

        let package = engine.prepare_export("agent-1", &kvector, &ChainId::new("polygon"), None);

        assert!(package.is_ok());
        let pkg = package.unwrap();
        assert_eq!(pkg.agent_id, "agent-1");
        assert!(!pkg.proof.proof_data.is_empty());
    }

    #[test]
    fn test_import_trust() {
        let mut engine = PortabilityEngine::new(PortabilityConfig {
            cross_chain_discount: 0.8,
            ..Default::default()
        });

        // Register source chain
        engine.register_chain(ChainProfile {
            id: ChainId::new("ethereum"),
            name: "Ethereum".to_string(),
            chain_type: ChainType::EVM,
            reputation: 0.9,
            supported_proofs: vec![ProofType::ZK],
            bridge_address: "0x1234".to_string(),
            last_synced: 1000,
        });

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let package = PortableTrust {
            id: "test-pkg".to_string(),
            agent_id: "agent-1".to_string(),
            source_chain: ChainId::new("ethereum"),
            kvector: KVector::new(0.8, 0.7, 0.9, 0.6, 0.5, 0.4, 0.7, 0.5, 0.8, 0.6),
            dimensions: vec![KVectorDimension::Reputation],
            trust_score: 0.7,
            proof: TrustProof {
                proof_type: ProofType::ZK,
                proof_data: vec![1, 2, 3],
                public_inputs: vec![],
                vk_id: "test-vk".to_string(),
                signatures: vec![],
            },
            timestamp,
            expires_at: timestamp + 3600_000,
            metadata: HashMap::new(),
        };

        let result = engine.import_trust(package);
        assert!(result.is_ok());

        let import = result.unwrap();
        assert_eq!(import.status, ImportStatus::Verified);
        assert!(import.applied_trust < 0.7); // Should be discounted
    }

    #[test]
    fn test_expired_package_rejection() {
        let mut engine = PortabilityEngine::new(PortabilityConfig::default());

        let package = PortableTrust {
            id: "test-pkg".to_string(),
            agent_id: "agent-1".to_string(),
            source_chain: ChainId::new("ethereum"),
            kvector: KVector::new_participant(),
            dimensions: vec![],
            trust_score: 0.5,
            proof: TrustProof {
                proof_type: ProofType::ZK,
                proof_data: vec![1],
                public_inputs: vec![],
                vk_id: "test".to_string(),
                signatures: vec![],
            },
            timestamp: 0,
            expires_at: 1, // Already expired
            metadata: HashMap::new(),
        };

        let result = engine.import_trust(package);
        assert!(matches!(result, Err(BridgeError::PackageExpired)));
    }

    #[test]
    fn test_mock_bridge_adapter() {
        let adapter = MockBridgeAdapter::new("test-chain", ChainType::EVM);

        let kvector = KVector::new_participant();
        let result = adapter.export("agent-1", &kvector, &[KVectorDimension::Reputation]);

        assert!(result.is_ok());
        let export = result.unwrap();
        assert_eq!(export.status, ExportStatus::Confirmed);

        // Import the exported package
        let import = adapter.import(&export.package);
        assert!(import.is_ok());
    }

    #[test]
    fn test_stats() {
        let mut engine = PortabilityEngine::new(PortabilityConfig::default());

        engine.register_chain(ChainProfile {
            id: ChainId::new("eth"),
            name: "Ethereum".to_string(),
            chain_type: ChainType::EVM,
            reputation: 0.9,
            supported_proofs: vec![ProofType::ZK],
            bridge_address: "0x".to_string(),
            last_synced: 0,
        });

        let stats = engine.stats();
        assert_eq!(stats.registered_chains, 1);
    }
}
