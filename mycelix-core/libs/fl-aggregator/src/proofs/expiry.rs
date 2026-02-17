//! Proof Expiry and Revocation System
//!
//! Time-limited proofs with Certificate Revocation List (CRL) support.
//!
//! ## Features
//!
//! - Time-limited proof validity windows
//! - Proof revocation with CRL
//! - Revocation reason tracking
//! - Bloom filter for efficient revocation checks
//! - Persistence support for revocation lists
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::expiry::{
//!     ExpiringProof, RevocationRegistry, RevocationReason,
//! };
//!
//! // Create expiring proof (valid for 1 hour)
//! let proof = ExpiringProof::new(proof_bytes, Duration::from_secs(3600));
//!
//! // Check validity
//! if proof.is_valid() {
//!     println!("Proof is valid until: {:?}", proof.expires_at());
//! }
//!
//! // Revocation registry
//! let registry = RevocationRegistry::new(RevocationConfig::default());
//!
//! // Revoke a proof
//! registry.revoke(proof_id, RevocationReason::Superseded).await?;
//!
//! // Check if revoked
//! if registry.is_revoked(proof_id).await {
//!     println!("Proof has been revoked");
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::proofs::{ProofError, ProofResult, ProofType};
use crate::proofs::types::ByteReader;

// ============================================================================
// Expiring Proof
// ============================================================================

/// A proof with an expiration time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpiringProof {
    /// Proof ID
    pub id: String,
    /// Proof type
    pub proof_type: ProofType,
    /// Raw proof bytes
    pub proof_bytes: Vec<u8>,
    /// Creation timestamp (Unix ms)
    pub created_at: u64,
    /// Expiration timestamp (Unix ms)
    pub expires_at: u64,
    /// Issuer ID
    pub issuer: Option<String>,
    /// Proof commitment (hash)
    pub commitment: [u8; 32],
    /// Extension data
    pub extensions: HashMap<String, String>,
}

impl ExpiringProof {
    /// Create a new expiring proof
    pub fn new(id: &str, proof_type: ProofType, proof_bytes: Vec<u8>, validity: Duration) -> Self {
        let now = current_timestamp_ms();
        let commitment = blake3::hash(&proof_bytes);

        Self {
            id: id.to_string(),
            proof_type,
            proof_bytes,
            created_at: now,
            expires_at: now + validity.as_millis() as u64,
            issuer: None,
            commitment: *commitment.as_bytes(),
            extensions: HashMap::new(),
        }
    }

    /// Create with absolute expiration
    pub fn with_expiration(
        id: &str,
        proof_type: ProofType,
        proof_bytes: Vec<u8>,
        expires_at: u64,
    ) -> Self {
        let commitment = blake3::hash(&proof_bytes);

        Self {
            id: id.to_string(),
            proof_type,
            proof_bytes,
            created_at: current_timestamp_ms(),
            expires_at,
            issuer: None,
            commitment: *commitment.as_bytes(),
            extensions: HashMap::new(),
        }
    }

    /// Set issuer
    pub fn with_issuer(mut self, issuer: &str) -> Self {
        self.issuer = Some(issuer.to_string());
        self
    }

    /// Add extension
    pub fn with_extension(mut self, key: &str, value: &str) -> Self {
        self.extensions.insert(key.to_string(), value.to_string());
        self
    }

    /// Check if proof is expired
    pub fn is_expired(&self) -> bool {
        current_timestamp_ms() >= self.expires_at
    }

    /// Check if proof is valid (not expired)
    pub fn is_valid(&self) -> bool {
        !self.is_expired()
    }

    /// Get remaining validity duration
    pub fn remaining(&self) -> Duration {
        let now = current_timestamp_ms();
        if now >= self.expires_at {
            Duration::ZERO
        } else {
            Duration::from_millis(self.expires_at - now)
        }
    }

    /// Get time until expiration as human readable string
    pub fn remaining_human(&self) -> String {
        let remaining = self.remaining();
        if remaining.is_zero() {
            "expired".to_string()
        } else if remaining.as_secs() < 60 {
            format!("{}s", remaining.as_secs())
        } else if remaining.as_secs() < 3600 {
            format!("{}m", remaining.as_secs() / 60)
        } else if remaining.as_secs() < 86400 {
            format!("{}h", remaining.as_secs() / 3600)
        } else {
            format!("{}d", remaining.as_secs() / 86400)
        }
    }

    /// Verify commitment matches proof bytes
    pub fn verify_commitment(&self) -> bool {
        let computed = blake3::hash(&self.proof_bytes);
        computed.as_bytes() == &self.commitment
    }

    /// Extend validity
    pub fn extend(&mut self, additional: Duration) {
        self.expires_at += additional.as_millis() as u64;
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Magic header
        bytes.extend_from_slice(b"EXPR");

        // Version
        bytes.push(1);

        // Proof type
        bytes.push(self.proof_type as u8);

        // Timestamps
        bytes.extend_from_slice(&self.created_at.to_le_bytes());
        bytes.extend_from_slice(&self.expires_at.to_le_bytes());

        // Commitment
        bytes.extend_from_slice(&self.commitment);

        // ID length + ID
        let id_bytes = self.id.as_bytes();
        bytes.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        bytes.extend_from_slice(id_bytes);

        // Proof bytes length + proof bytes
        bytes.extend_from_slice(&(self.proof_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.proof_bytes);

        bytes
    }

    /// Deserialize from bytes
    ///
    /// Uses safe bounds-checked parsing to prevent panics on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        let mut reader = ByteReader::new(bytes);

        // Read and validate magic bytes
        let magic = reader.read_bytes(4).ok_or_else(|| {
            ProofError::SerializationError("Missing magic bytes".to_string())
        })?;
        if magic != b"EXPR" {
            return Err(ProofError::SerializationError("Invalid magic".to_string()));
        }

        let version = reader.read_u8().ok_or_else(|| {
            ProofError::SerializationError("Missing version byte".to_string())
        })?;
        if version != 1 {
            return Err(ProofError::SerializationError(format!(
                "Unsupported version: {}",
                version
            )));
        }

        let proof_type_byte = reader.read_u8().ok_or_else(|| {
            ProofError::SerializationError("Missing proof type byte".to_string())
        })?;
        let proof_type = match proof_type_byte {
            0 => ProofType::Range,
            1 => ProofType::GradientIntegrity,
            2 => ProofType::IdentityAssurance,
            3 => ProofType::VoteEligibility,
            4 => ProofType::Membership,
            _ => {
                return Err(ProofError::SerializationError("Invalid proof type".to_string()));
            }
        };

        let created_at = reader.read_u64_le().ok_or_else(|| {
            ProofError::SerializationError("Missing created_at timestamp".to_string())
        })?;

        let expires_at = reader.read_u64_le().ok_or_else(|| {
            ProofError::SerializationError("Missing expires_at timestamp".to_string())
        })?;

        let commitment = reader.read_32_bytes().ok_or_else(|| {
            ProofError::SerializationError("Missing commitment hash".to_string())
        })?;

        let id_len = reader.read_u16_le().ok_or_else(|| {
            ProofError::SerializationError("Missing id length".to_string())
        })? as usize;

        let id_bytes = reader.read_bytes(id_len).ok_or_else(|| {
            ProofError::SerializationError("Truncated id string".to_string())
        })?;
        let id = String::from_utf8_lossy(id_bytes).to_string();

        let proof_len = reader.read_u32_le().ok_or_else(|| {
            ProofError::SerializationError("Missing proof length".to_string())
        })? as usize;

        let proof_bytes = reader.read_bytes(proof_len).ok_or_else(|| {
            ProofError::SerializationError("Truncated proof bytes".to_string())
        })?.to_vec();

        Ok(Self {
            id,
            proof_type,
            proof_bytes,
            created_at,
            expires_at,
            issuer: None,
            commitment,
            extensions: HashMap::new(),
        })
    }
}

// ============================================================================
// Revocation
// ============================================================================

/// Reason for proof revocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RevocationReason {
    /// Proof was superseded by newer version
    Superseded,
    /// Key compromise detected
    KeyCompromise,
    /// Affiliation changed
    AffiliationChanged,
    /// Proof should not have been issued
    PrivilegeWithdrawn,
    /// Issuer stopped operating
    CessationOfOperation,
    /// Certificate hold (temporary)
    CertificateHold,
    /// Removed from CRL (unrevoke)
    RemoveFromCRL,
    /// Unspecified reason
    Unspecified,
}

impl RevocationReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            RevocationReason::Superseded => "superseded",
            RevocationReason::KeyCompromise => "key_compromise",
            RevocationReason::AffiliationChanged => "affiliation_changed",
            RevocationReason::PrivilegeWithdrawn => "privilege_withdrawn",
            RevocationReason::CessationOfOperation => "cessation_of_operation",
            RevocationReason::CertificateHold => "certificate_hold",
            RevocationReason::RemoveFromCRL => "remove_from_crl",
            RevocationReason::Unspecified => "unspecified",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "superseded" => Some(RevocationReason::Superseded),
            "key_compromise" => Some(RevocationReason::KeyCompromise),
            "affiliation_changed" => Some(RevocationReason::AffiliationChanged),
            "privilege_withdrawn" => Some(RevocationReason::PrivilegeWithdrawn),
            "cessation_of_operation" => Some(RevocationReason::CessationOfOperation),
            "certificate_hold" => Some(RevocationReason::CertificateHold),
            "remove_from_crl" => Some(RevocationReason::RemoveFromCRL),
            "unspecified" => Some(RevocationReason::Unspecified),
            _ => None,
        }
    }
}

/// Revocation entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationEntry {
    /// Proof ID or commitment
    pub proof_id: String,
    /// Revocation reason
    pub reason: RevocationReason,
    /// Revocation timestamp
    pub revoked_at: u64,
    /// Who performed the revocation
    pub revoked_by: Option<String>,
    /// Optional notes
    pub notes: Option<String>,
    /// Invalidity date (when the issue occurred, may differ from revoked_at)
    pub invalidity_date: Option<u64>,
}

/// Revocation registry configuration
#[derive(Debug, Clone)]
pub struct RevocationConfig {
    /// Use bloom filter for fast negative lookups
    pub use_bloom_filter: bool,
    /// Bloom filter size (bits)
    pub bloom_filter_size: usize,
    /// Bloom filter hash count
    pub bloom_filter_hashes: u32,
    /// Maximum entries to keep
    pub max_entries: usize,
    /// Auto-cleanup interval in seconds
    pub cleanup_interval_secs: u64,
}

impl Default for RevocationConfig {
    fn default() -> Self {
        Self {
            use_bloom_filter: true,
            bloom_filter_size: 10_000_000, // ~1.2MB
            bloom_filter_hashes: 7,
            max_entries: 1_000_000,
            cleanup_interval_secs: 3600,
        }
    }
}

/// Revocation registry (Certificate Revocation List)
pub struct RevocationRegistry {
    config: RevocationConfig,
    entries: Arc<RwLock<HashMap<String, RevocationEntry>>>,
    bloom_filter: Arc<RwLock<BloomFilter>>,
    stats: Arc<RwLock<RevocationStats>>,
}

impl RevocationRegistry {
    /// Create a new registry
    pub fn new(config: RevocationConfig) -> Self {
        let bloom = BloomFilter::new(config.bloom_filter_size, config.bloom_filter_hashes);

        Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            bloom_filter: Arc::new(RwLock::new(bloom)),
            stats: Arc::new(RwLock::new(RevocationStats::default())),
        }
    }

    /// Revoke a proof
    pub async fn revoke(&self, proof_id: &str, reason: RevocationReason) -> ProofResult<()> {
        self.revoke_with_details(proof_id, reason, None, None).await
    }

    /// Revoke with full details
    pub async fn revoke_with_details(
        &self,
        proof_id: &str,
        reason: RevocationReason,
        revoked_by: Option<&str>,
        notes: Option<&str>,
    ) -> ProofResult<()> {
        let entry = RevocationEntry {
            proof_id: proof_id.to_string(),
            reason,
            revoked_at: current_timestamp_ms(),
            revoked_by: revoked_by.map(|s| s.to_string()),
            notes: notes.map(|s| s.to_string()),
            invalidity_date: None,
        };

        // Add to entries
        {
            let mut entries = self.entries.write().await;
            if entries.len() >= self.config.max_entries {
                return Err(ProofError::InvalidInput(
                    "Revocation registry full".to_string(),
                ));
            }
            entries.insert(proof_id.to_string(), entry);
        }

        // Add to bloom filter
        if self.config.use_bloom_filter {
            let mut bloom = self.bloom_filter.write().await;
            bloom.insert(proof_id.as_bytes());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_revocations += 1;
            *stats.by_reason.entry(reason).or_insert(0) += 1;
        }

        Ok(())
    }

    /// Check if a proof is revoked
    pub async fn is_revoked(&self, proof_id: &str) -> bool {
        // Fast path with bloom filter
        if self.config.use_bloom_filter {
            let bloom = self.bloom_filter.read().await;
            if !bloom.may_contain(proof_id.as_bytes()) {
                return false; // Definitely not revoked
            }
        }

        // Check actual entries
        let entries = self.entries.read().await;
        entries.contains_key(proof_id)
    }

    /// Get revocation details
    pub async fn get_revocation(&self, proof_id: &str) -> Option<RevocationEntry> {
        let entries = self.entries.read().await;
        entries.get(proof_id).cloned()
    }

    /// Unrevoke a proof (for certificate hold)
    pub async fn unrevoke(&self, proof_id: &str) -> ProofResult<bool> {
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.get(proof_id) {
            if entry.reason != RevocationReason::CertificateHold {
                return Err(ProofError::InvalidInput(
                    "Can only unrevoke certificate holds".to_string(),
                ));
            }
        }

        Ok(entries.remove(proof_id).is_some())
    }

    /// Get all revoked proof IDs
    pub async fn list_revoked(&self) -> Vec<String> {
        let entries = self.entries.read().await;
        entries.keys().cloned().collect()
    }

    /// Get revocations by reason
    pub async fn list_by_reason(&self, reason: RevocationReason) -> Vec<RevocationEntry> {
        let entries = self.entries.read().await;
        entries
            .values()
            .filter(|e| e.reason == reason)
            .cloned()
            .collect()
    }

    /// Get registry statistics
    pub async fn stats(&self) -> RevocationStats {
        self.stats.read().await.clone()
    }

    /// Export CRL as bytes
    pub async fn export_crl(&self) -> Vec<u8> {
        let entries = self.entries.read().await;
        let crl = CRL {
            version: 1,
            this_update: current_timestamp_ms(),
            next_update: current_timestamp_ms() + 86400000, // +24h
            entries: entries.values().cloned().collect(),
        };

        serde_json::to_vec(&crl).unwrap_or_default()
    }

    /// Import CRL from bytes
    pub async fn import_crl(&self, bytes: &[u8]) -> ProofResult<usize> {
        let crl: CRL = serde_json::from_slice(bytes)
            .map_err(|e| ProofError::SerializationError(e.to_string()))?;

        let mut entries = self.entries.write().await;
        let mut bloom = self.bloom_filter.write().await;
        let mut count = 0;

        for entry in crl.entries {
            if !entries.contains_key(&entry.proof_id) {
                bloom.insert(entry.proof_id.as_bytes());
                entries.insert(entry.proof_id.clone(), entry);
                count += 1;
            }
        }

        Ok(count)
    }

    /// Clear all revocations
    pub async fn clear(&self) {
        let mut entries = self.entries.write().await;
        let mut bloom = self.bloom_filter.write().await;
        let mut stats = self.stats.write().await;

        entries.clear();
        *bloom = BloomFilter::new(self.config.bloom_filter_size, self.config.bloom_filter_hashes);
        *stats = RevocationStats::default();
    }
}

/// CRL structure for export/import
#[derive(Debug, Serialize, Deserialize)]
struct CRL {
    version: u8,
    this_update: u64,
    next_update: u64,
    entries: Vec<RevocationEntry>,
}

/// Revocation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RevocationStats {
    pub total_revocations: u64,
    pub by_reason: HashMap<RevocationReason, u64>,
}

// ============================================================================
// Bloom Filter
// ============================================================================

/// Simple bloom filter for efficient revocation checks
struct BloomFilter {
    bits: Vec<bool>,
    num_hashes: u32,
}

impl BloomFilter {
    fn new(size: usize, num_hashes: u32) -> Self {
        Self {
            bits: vec![false; size],
            num_hashes,
        }
    }

    fn insert(&mut self, item: &[u8]) {
        for i in 0..self.num_hashes {
            let index = self.hash(item, i) % self.bits.len();
            self.bits[index] = true;
        }
    }

    fn may_contain(&self, item: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let index = self.hash(item, i) % self.bits.len();
            if !self.bits[index] {
                return false;
            }
        }
        true
    }

    fn hash(&self, item: &[u8], seed: u32) -> usize {
        let mut hasher = blake3::Hasher::new();
        hasher.update(item);
        hasher.update(&seed.to_le_bytes());
        let hash = hasher.finalize();
        let bytes = hash.as_bytes();

        // Use first 8 bytes as usize
        usize::from_le_bytes(bytes[0..8].try_into().unwrap())
    }
}

// ============================================================================
// Validity Checker
// ============================================================================

/// Combined validity checker for expiry and revocation
pub struct ValidityChecker {
    registry: Arc<RevocationRegistry>,
}

impl ValidityChecker {
    /// Create a new validity checker
    pub fn new(registry: Arc<RevocationRegistry>) -> Self {
        Self { registry }
    }

    /// Check if an expiring proof is valid
    pub async fn check(&self, proof: &ExpiringProof) -> ValidityStatus {
        // Check expiration
        if proof.is_expired() {
            return ValidityStatus::Expired {
                expired_at: proof.expires_at,
            };
        }

        // Check revocation
        if self.registry.is_revoked(&proof.id).await {
            if let Some(entry) = self.registry.get_revocation(&proof.id).await {
                return ValidityStatus::Revoked {
                    reason: entry.reason,
                    revoked_at: entry.revoked_at,
                };
            }
        }

        // Check commitment
        if !proof.verify_commitment() {
            return ValidityStatus::Invalid {
                reason: "Commitment mismatch".to_string(),
            };
        }

        ValidityStatus::Valid {
            remaining_ms: proof.remaining().as_millis() as u64,
        }
    }

    /// Batch check multiple proofs
    pub async fn check_batch(&self, proofs: &[ExpiringProof]) -> Vec<ValidityStatus> {
        let mut results = Vec::with_capacity(proofs.len());
        for proof in proofs {
            results.push(self.check(proof).await);
        }
        results
    }
}

/// Validity status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidityStatus {
    /// Proof is valid
    Valid { remaining_ms: u64 },
    /// Proof has expired
    Expired { expired_at: u64 },
    /// Proof has been revoked
    Revoked {
        reason: RevocationReason,
        revoked_at: u64,
    },
    /// Proof is invalid
    Invalid { reason: String },
}

impl ValidityStatus {
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidityStatus::Valid { .. })
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expiring_proof_creation() {
        let proof = ExpiringProof::new(
            "test-id",
            ProofType::Range,
            vec![1, 2, 3],
            Duration::from_secs(3600),
        );

        assert!(!proof.is_expired());
        assert!(proof.is_valid());
        assert!(proof.remaining() > Duration::ZERO);
    }

    #[test]
    fn test_expiring_proof_expired() {
        let mut proof = ExpiringProof::new(
            "test-id",
            ProofType::Range,
            vec![1, 2, 3],
            Duration::from_secs(0),
        );

        // Force expiration
        proof.expires_at = proof.created_at;

        assert!(proof.is_expired());
        assert!(!proof.is_valid());
    }

    #[test]
    fn test_expiring_proof_serialization() {
        let proof = ExpiringProof::new(
            "test-id",
            ProofType::Range,
            vec![1, 2, 3, 4, 5],
            Duration::from_secs(3600),
        );

        let bytes = proof.to_bytes();
        let restored = ExpiringProof::from_bytes(&bytes).unwrap();

        assert_eq!(proof.id, restored.id);
        assert_eq!(proof.proof_bytes, restored.proof_bytes);
        assert_eq!(proof.commitment, restored.commitment);
    }

    #[test]
    fn test_expiring_proof_commitment() {
        let proof = ExpiringProof::new(
            "test-id",
            ProofType::Range,
            vec![1, 2, 3],
            Duration::from_secs(3600),
        );

        assert!(proof.verify_commitment());
    }

    #[test]
    fn test_revocation_reason() {
        assert_eq!(
            RevocationReason::from_str("superseded"),
            Some(RevocationReason::Superseded)
        );
        assert_eq!(
            RevocationReason::KeyCompromise.as_str(),
            "key_compromise"
        );
    }

    #[tokio::test]
    async fn test_revocation_registry_revoke() {
        let registry = RevocationRegistry::new(RevocationConfig::default());

        registry
            .revoke("proof-1", RevocationReason::Superseded)
            .await
            .unwrap();

        assert!(registry.is_revoked("proof-1").await);
        assert!(!registry.is_revoked("proof-2").await);
    }

    #[tokio::test]
    async fn test_revocation_registry_details() {
        let registry = RevocationRegistry::new(RevocationConfig::default());

        registry
            .revoke_with_details(
                "proof-1",
                RevocationReason::KeyCompromise,
                Some("admin"),
                Some("Compromised key detected"),
            )
            .await
            .unwrap();

        let entry = registry.get_revocation("proof-1").await.unwrap();
        assert_eq!(entry.reason, RevocationReason::KeyCompromise);
        assert_eq!(entry.revoked_by, Some("admin".to_string()));
    }

    #[tokio::test]
    async fn test_revocation_registry_unrevoke() {
        let registry = RevocationRegistry::new(RevocationConfig::default());

        // Can only unrevoke certificate holds
        registry
            .revoke("proof-1", RevocationReason::CertificateHold)
            .await
            .unwrap();

        let result = registry.unrevoke("proof-1").await.unwrap();
        assert!(result);
        assert!(!registry.is_revoked("proof-1").await);
    }

    #[tokio::test]
    async fn test_revocation_registry_stats() {
        let registry = RevocationRegistry::new(RevocationConfig::default());

        registry
            .revoke("proof-1", RevocationReason::Superseded)
            .await
            .unwrap();
        registry
            .revoke("proof-2", RevocationReason::Superseded)
            .await
            .unwrap();
        registry
            .revoke("proof-3", RevocationReason::KeyCompromise)
            .await
            .unwrap();

        let stats = registry.stats().await;
        assert_eq!(stats.total_revocations, 3);
        assert_eq!(stats.by_reason.get(&RevocationReason::Superseded), Some(&2));
    }

    #[tokio::test]
    async fn test_validity_checker() {
        let registry = Arc::new(RevocationRegistry::new(RevocationConfig::default()));
        let checker = ValidityChecker::new(registry.clone());

        let proof = ExpiringProof::new(
            "proof-1",
            ProofType::Range,
            vec![1, 2, 3],
            Duration::from_secs(3600),
        );

        // Valid proof
        let status = checker.check(&proof).await;
        assert!(status.is_valid());

        // Revoke and check again
        registry
            .revoke("proof-1", RevocationReason::Superseded)
            .await
            .unwrap();

        let status = checker.check(&proof).await;
        assert!(!status.is_valid());
        assert!(matches!(status, ValidityStatus::Revoked { .. }));
    }

    #[test]
    fn test_bloom_filter() {
        let mut bloom = BloomFilter::new(10000, 7);

        bloom.insert(b"item1");
        bloom.insert(b"item2");

        assert!(bloom.may_contain(b"item1"));
        assert!(bloom.may_contain(b"item2"));
        // May have false positives, but false negatives are not allowed
    }

    #[tokio::test]
    async fn test_crl_export_import() {
        let registry1 = RevocationRegistry::new(RevocationConfig::default());
        registry1
            .revoke("proof-1", RevocationReason::Superseded)
            .await
            .unwrap();
        registry1
            .revoke("proof-2", RevocationReason::KeyCompromise)
            .await
            .unwrap();

        let crl_bytes = registry1.export_crl().await;

        let registry2 = RevocationRegistry::new(RevocationConfig::default());
        let imported = registry2.import_crl(&crl_bytes).await.unwrap();

        assert_eq!(imported, 2);
        assert!(registry2.is_revoked("proof-1").await);
        assert!(registry2.is_revoked("proof-2").await);
    }
}
