//! Timestamped Proofs with Expiration
//!
//! Adds temporal validity to proofs to prevent replay attacks and ensure freshness.
//!
//! ## Security Properties
//!
//! - **Replay Prevention**: Proofs can only be used within their validity window
//! - **Freshness Guarantee**: Old proofs automatically expire
//! - **Nonce Binding**: Optional nonce prevents proof reuse across contexts
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::{RangeProof, ProofConfig};
//! use fl_aggregator::proofs::timestamped::{TimestampedProof, TimestampConfig};
//! use std::time::Duration;
//!
//! // Generate a proof
//! let proof = RangeProof::generate(50, 0, 100, ProofConfig::default())?;
//!
//! // Wrap with 1-hour validity
//! let config = TimestampConfig::default().with_validity(Duration::from_secs(3600));
//! let timestamped = TimestampedProof::new(proof, config);
//!
//! // Check validity before using
//! if timestamped.is_valid() {
//!     let result = timestamped.verify()?;
//! }
//! ```

use crate::proofs::{ProofError, ProofResult, ProofType, VerificationResult};
use crate::proofs::types::{ByteReader, ByteWriter};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Default proof validity duration (1 hour)
pub const DEFAULT_VALIDITY_SECS: u64 = 3600;

/// Maximum allowed clock skew (5 minutes)
pub const MAX_CLOCK_SKEW_SECS: u64 = 300;

/// Configuration for timestamped proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampConfig {
    /// How long the proof remains valid
    pub validity_duration: Duration,

    /// Optional nonce to bind proof to specific context
    pub nonce: Option<[u8; 32]>,

    /// Whether to allow clock skew tolerance
    pub allow_clock_skew: bool,

    /// Maximum clock skew to tolerate
    pub max_clock_skew: Duration,

    /// Issuer identifier (e.g., node ID, DID)
    pub issuer: Option<String>,

    /// Audience/recipient identifier
    pub audience: Option<String>,
}

impl Default for TimestampConfig {
    fn default() -> Self {
        Self {
            validity_duration: Duration::from_secs(DEFAULT_VALIDITY_SECS),
            nonce: None,
            allow_clock_skew: true,
            max_clock_skew: Duration::from_secs(MAX_CLOCK_SKEW_SECS),
            issuer: None,
            audience: None,
        }
    }
}

impl TimestampConfig {
    /// Set validity duration
    pub fn with_validity(mut self, duration: Duration) -> Self {
        self.validity_duration = duration;
        self
    }

    /// Set nonce for context binding
    pub fn with_nonce(mut self, nonce: [u8; 32]) -> Self {
        self.nonce = Some(nonce);
        self
    }

    /// Generate random nonce
    pub fn with_random_nonce(mut self) -> Self {
        let mut nonce = [0u8; 32];
        // Use timestamp + counter as simple entropy source
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        nonce[..16].copy_from_slice(&timestamp.to_le_bytes());
        // Fill rest with hash of timestamp
        let hash = Sha256::digest(&timestamp.to_le_bytes());
        nonce[16..].copy_from_slice(&hash[..16]);
        self.nonce = Some(nonce);
        self
    }

    /// Set issuer
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = Some(issuer.into());
        self
    }

    /// Set audience
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = Some(audience.into());
        self
    }

    /// Disable clock skew tolerance
    pub fn strict_time(mut self) -> Self {
        self.allow_clock_skew = false;
        self
    }

    /// Short-lived proof (5 minutes)
    pub fn short_lived() -> Self {
        Self::default().with_validity(Duration::from_secs(300))
    }

    /// Long-lived proof (24 hours)
    pub fn long_lived() -> Self {
        Self::default().with_validity(Duration::from_secs(86400))
    }

    /// Single-use proof (requires nonce)
    pub fn single_use() -> Self {
        Self::default()
            .with_validity(Duration::from_secs(300))
            .with_random_nonce()
    }
}

/// Timestamp metadata for a proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofTimestamp {
    /// When the proof was created (Unix timestamp in seconds)
    pub created_at: u64,

    /// When the proof expires (Unix timestamp in seconds)
    pub expires_at: u64,

    /// Optional nonce for replay prevention
    pub nonce: Option<[u8; 32]>,

    /// Issuer identifier
    pub issuer: Option<String>,

    /// Intended audience
    pub audience: Option<String>,

    /// Binding hash (commits to proof + timestamp)
    pub binding: [u8; 32],
}

impl ProofTimestamp {
    /// Create new timestamp metadata
    pub fn new(proof_hash: &[u8], config: &TimestampConfig) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let created_at = now;
        let expires_at = now + config.validity_duration.as_secs();

        // Compute binding hash
        let binding = Self::compute_binding(
            proof_hash,
            created_at,
            expires_at,
            config.nonce.as_ref(),
            config.issuer.as_deref(),
            config.audience.as_deref(),
        );

        Self {
            created_at,
            expires_at,
            nonce: config.nonce,
            issuer: config.issuer.clone(),
            audience: config.audience.clone(),
            binding,
        }
    }

    /// Compute binding hash that commits to all timestamp fields
    fn compute_binding(
        proof_hash: &[u8],
        created_at: u64,
        expires_at: u64,
        nonce: Option<&[u8; 32]>,
        issuer: Option<&str>,
        audience: Option<&str>,
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(proof_hash);
        hasher.update(&created_at.to_le_bytes());
        hasher.update(&expires_at.to_le_bytes());
        if let Some(n) = nonce {
            hasher.update(n);
        }
        if let Some(i) = issuer {
            hasher.update(i.as_bytes());
        }
        if let Some(a) = audience {
            hasher.update(a.as_bytes());
        }
        let result = hasher.finalize();
        let mut binding = [0u8; 32];
        binding.copy_from_slice(&result);
        binding
    }

    /// Verify the binding hash
    pub fn verify_binding(&self, proof_hash: &[u8]) -> bool {
        let expected = Self::compute_binding(
            proof_hash,
            self.created_at,
            self.expires_at,
            self.nonce.as_ref(),
            self.issuer.as_deref(),
            self.audience.as_deref(),
        );
        self.binding == expected
    }

    /// Check if timestamp is currently valid
    pub fn is_valid(&self) -> bool {
        self.is_valid_with_skew(Duration::from_secs(MAX_CLOCK_SKEW_SECS))
    }

    /// Check validity with custom clock skew tolerance
    pub fn is_valid_with_skew(&self, max_skew: Duration) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let skew_secs = max_skew.as_secs();

        // Not yet valid (created in future beyond skew tolerance)
        if self.created_at > now + skew_secs {
            return false;
        }

        // Expired
        if now > self.expires_at + skew_secs {
            return false;
        }

        true
    }

    /// Check if timestamp is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now > self.expires_at
    }

    /// Get remaining validity duration
    pub fn remaining_validity(&self) -> Option<Duration> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if now >= self.expires_at {
            None
        } else {
            Some(Duration::from_secs(self.expires_at - now))
        }
    }

    /// Get age of the proof
    pub fn age(&self) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if now >= self.created_at {
            Duration::from_secs(now - self.created_at)
        } else {
            Duration::ZERO
        }
    }

    /// Serialize timestamp metadata
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut writer = ByteWriter::with_capacity(128);

        writer.write_u64_le(self.created_at);
        writer.write_u64_le(self.expires_at);

        // Nonce (1 byte flag + optional 32 bytes)
        if let Some(nonce) = &self.nonce {
            writer.write_u8(1);
            writer.write_bytes(nonce);
        } else {
            writer.write_u8(0);
        }

        // Issuer (length-prefixed string)
        if let Some(issuer) = &self.issuer {
            writer.write_string_u16(issuer);
        } else {
            writer.write_u16_le(0);
        }

        // Audience (length-prefixed string)
        if let Some(audience) = &self.audience {
            writer.write_string_u16(audience);
        } else {
            writer.write_u16_le(0);
        }

        // Binding hash
        writer.write_bytes(&self.binding);

        writer.finish()
    }

    /// Deserialize timestamp metadata
    ///
    /// Uses safe bounds-checked parsing to prevent panics on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        let mut reader = ByteReader::new(bytes);

        let created_at = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated created_at timestamp".to_string())
        })?;

        let expires_at = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated expires_at timestamp".to_string())
        })?;

        let has_nonce = reader.read_u8().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing nonce flag".to_string())
        })? == 1;

        let nonce = if has_nonce {
            let n = reader.read_32_bytes().ok_or_else(|| {
                ProofError::InvalidProofFormat("Missing or truncated nonce".to_string())
            })?;
            Some(n)
        } else {
            None
        };

        let issuer_len = reader.read_u16_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing issuer length".to_string())
        })? as usize;

        let issuer = if issuer_len > 0 {
            let issuer_bytes = reader.read_bytes(issuer_len).ok_or_else(|| {
                ProofError::InvalidProofFormat("Truncated issuer string".to_string())
            })?;
            let s = String::from_utf8(issuer_bytes.to_vec())
                .map_err(|_| ProofError::InvalidProofFormat("Invalid issuer UTF-8".to_string()))?;
            Some(s)
        } else {
            None
        };

        let audience_len = reader.read_u16_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing audience length".to_string())
        })? as usize;

        let audience = if audience_len > 0 {
            let audience_bytes = reader.read_bytes(audience_len).ok_or_else(|| {
                ProofError::InvalidProofFormat("Truncated audience string".to_string())
            })?;
            let s = String::from_utf8(audience_bytes.to_vec())
                .map_err(|_| ProofError::InvalidProofFormat("Invalid audience UTF-8".to_string()))?;
            Some(s)
        } else {
            None
        };

        let binding = reader.read_32_bytes().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated binding hash".to_string())
        })?;

        Ok(Self {
            created_at,
            expires_at,
            nonce,
            issuer,
            audience,
            binding,
        })
    }
}

/// A proof wrapped with timestamp and expiration
#[derive(Clone)]
pub struct TimestampedProof<P> {
    /// The underlying proof
    pub proof: P,

    /// Timestamp metadata
    pub timestamp: ProofTimestamp,

    /// Proof type for verification
    proof_type: ProofType,
}

impl<P> TimestampedProof<P>
where
    P: Clone + TimestampableProof,
{
    /// Create a new timestamped proof
    pub fn new(proof: P, config: TimestampConfig) -> Self {
        let proof_hash = proof.proof_hash();
        let timestamp = ProofTimestamp::new(&proof_hash, &config);
        let proof_type = proof.proof_type();

        Self {
            proof,
            timestamp,
            proof_type,
        }
    }

    /// Create with custom timestamp (for testing or migration)
    pub fn with_timestamp(proof: P, timestamp: ProofTimestamp) -> Self {
        let proof_type = proof.proof_type();
        Self {
            proof,
            timestamp,
            proof_type,
        }
    }

    /// Check if the proof is currently valid (not expired)
    pub fn is_valid(&self) -> bool {
        self.timestamp.is_valid()
    }

    /// Check validity with custom clock skew
    pub fn is_valid_with_skew(&self, max_skew: Duration) -> bool {
        self.timestamp.is_valid_with_skew(max_skew)
    }

    /// Check if the proof is expired
    pub fn is_expired(&self) -> bool {
        self.timestamp.is_expired()
    }

    /// Verify the proof (checks timestamp and runs proof verification)
    pub fn verify(&self) -> ProofResult<TimestampedVerificationResult> {
        // First check timestamp validity
        if !self.timestamp.is_valid() {
            return Ok(TimestampedVerificationResult {
                inner: VerificationResult::failure(
                    self.proof_type,
                    Duration::ZERO,
                    "Proof timestamp invalid or expired",
                ),
                timestamp_valid: false,
                binding_valid: false,
                expired: self.timestamp.is_expired(),
                remaining_validity: None,
            });
        }

        // Verify binding
        let proof_hash = self.proof.proof_hash();
        if !self.timestamp.verify_binding(&proof_hash) {
            return Ok(TimestampedVerificationResult {
                inner: VerificationResult::failure(
                    self.proof_type,
                    Duration::ZERO,
                    "Timestamp binding verification failed",
                ),
                timestamp_valid: true,
                binding_valid: false,
                expired: false,
                remaining_validity: self.timestamp.remaining_validity(),
            });
        }

        // Run proof verification
        let inner = self.proof.verify_proof()?;

        Ok(TimestampedVerificationResult {
            timestamp_valid: true,
            binding_valid: true,
            expired: false,
            remaining_validity: self.timestamp.remaining_validity(),
            inner,
        })
    }

    /// Verify with audience check
    pub fn verify_for_audience(&self, expected_audience: &str) -> ProofResult<TimestampedVerificationResult> {
        // Check audience first
        if let Some(ref audience) = self.timestamp.audience {
            if audience != expected_audience {
                return Ok(TimestampedVerificationResult {
                    inner: VerificationResult::failure(
                        self.proof_type,
                        Duration::ZERO,
                        &format!("Audience mismatch: expected {}, got {}", expected_audience, audience),
                    ),
                    timestamp_valid: true,
                    binding_valid: true,
                    expired: false,
                    remaining_validity: self.timestamp.remaining_validity(),
                });
            }
        }

        self.verify()
    }

    /// Get remaining validity duration
    pub fn remaining_validity(&self) -> Option<Duration> {
        self.timestamp.remaining_validity()
    }

    /// Get proof age
    pub fn age(&self) -> Duration {
        self.timestamp.age()
    }

    /// Get the nonce if present
    pub fn nonce(&self) -> Option<&[u8; 32]> {
        self.timestamp.nonce.as_ref()
    }

    /// Get issuer if present
    pub fn issuer(&self) -> Option<&str> {
        self.timestamp.issuer.as_deref()
    }

    /// Get audience if present
    pub fn audience(&self) -> Option<&str> {
        self.timestamp.audience.as_deref()
    }

    /// Serialize the timestamped proof
    pub fn to_bytes(&self) -> Vec<u8> {
        let proof_bytes = self.proof.to_bytes();
        let timestamp_bytes = self.timestamp.to_bytes();

        let mut bytes = Vec::new();
        // Magic bytes for timestamped proof
        bytes.extend_from_slice(b"TSTP");
        // Version
        bytes.push(1);
        // Proof type
        bytes.push(self.proof_type as u8);
        // Timestamp length
        bytes.extend_from_slice(&(timestamp_bytes.len() as u32).to_le_bytes());
        // Timestamp
        bytes.extend_from_slice(&timestamp_bytes);
        // Proof length
        bytes.extend_from_slice(&(proof_bytes.len() as u32).to_le_bytes());
        // Proof
        bytes.extend_from_slice(&proof_bytes);

        bytes
    }
}

/// Trait for proofs that can be timestamped
pub trait TimestampableProof: Clone {
    /// Get a hash of the proof for binding
    fn proof_hash(&self) -> Vec<u8>;

    /// Get the proof type
    fn proof_type(&self) -> ProofType;

    /// Verify the underlying proof
    fn verify_proof(&self) -> ProofResult<VerificationResult>;

    /// Serialize to bytes
    fn to_bytes(&self) -> Vec<u8>;
}

/// Verification result with timestamp information
#[derive(Debug, Clone)]
pub struct TimestampedVerificationResult {
    /// Inner verification result
    pub inner: VerificationResult,

    /// Whether the timestamp was valid
    pub timestamp_valid: bool,

    /// Whether the binding was valid
    pub binding_valid: bool,

    /// Whether the proof has expired
    pub expired: bool,

    /// Remaining validity duration
    pub remaining_validity: Option<Duration>,
}

impl TimestampedVerificationResult {
    /// Check if the entire verification passed
    pub fn is_valid(&self) -> bool {
        self.inner.valid && self.timestamp_valid && self.binding_valid && !self.expired
    }

    /// Get summary of verification status
    pub fn summary(&self) -> String {
        if self.is_valid() {
            format!(
                "Valid (expires in {:?})",
                self.remaining_validity.unwrap_or_default()
            )
        } else if self.expired {
            "Expired".to_string()
        } else if !self.timestamp_valid {
            "Invalid timestamp".to_string()
        } else if !self.binding_valid {
            "Binding verification failed".to_string()
        } else {
            format!("Proof invalid: {:?}", self.inner.details)
        }
    }
}

/// Nonce registry for tracking used nonces (replay prevention)
#[derive(Debug, Default)]
pub struct NonceRegistry {
    /// Used nonces with expiration time
    used_nonces: std::collections::HashMap<[u8; 32], u64>,

    /// Maximum number of nonces to track
    max_entries: usize,
}

impl NonceRegistry {
    /// Create a new nonce registry
    pub fn new(max_entries: usize) -> Self {
        Self {
            used_nonces: std::collections::HashMap::new(),
            max_entries,
        }
    }

    /// Check and register a nonce (returns false if already used)
    pub fn check_and_register(&mut self, nonce: &[u8; 32], expires_at: u64) -> bool {
        // Clean expired entries periodically
        if self.used_nonces.len() >= self.max_entries {
            self.cleanup_expired();
        }

        if self.used_nonces.contains_key(nonce) {
            return false;
        }

        self.used_nonces.insert(*nonce, expires_at);
        true
    }

    /// Check if a nonce has been used
    pub fn is_used(&self, nonce: &[u8; 32]) -> bool {
        self.used_nonces.contains_key(nonce)
    }

    /// Remove expired nonces
    pub fn cleanup_expired(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.used_nonces.retain(|_, &mut expires_at| expires_at > now);
    }

    /// Get number of tracked nonces
    pub fn len(&self) -> usize {
        self.used_nonces.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.used_nonces.is_empty()
    }
}

// Implement TimestampableProof for our proof types
use crate::proofs::{RangeProof, GradientIntegrityProof, MembershipProof, IdentityAssuranceProof, VoteEligibilityProof};

impl TimestampableProof for RangeProof {
    fn proof_hash(&self) -> Vec<u8> {
        let bytes = self.to_bytes();
        Sha256::digest(&bytes).to_vec()
    }

    fn proof_type(&self) -> ProofType {
        ProofType::Range
    }

    fn verify_proof(&self) -> ProofResult<VerificationResult> {
        self.verify()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
}

impl TimestampableProof for GradientIntegrityProof {
    fn proof_hash(&self) -> Vec<u8> {
        let bytes = self.to_bytes();
        Sha256::digest(&bytes).to_vec()
    }

    fn proof_type(&self) -> ProofType {
        ProofType::GradientIntegrity
    }

    fn verify_proof(&self) -> ProofResult<VerificationResult> {
        self.verify()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
}

impl TimestampableProof for MembershipProof {
    fn proof_hash(&self) -> Vec<u8> {
        let bytes = self.to_bytes();
        Sha256::digest(&bytes).to_vec()
    }

    fn proof_type(&self) -> ProofType {
        ProofType::Membership
    }

    fn verify_proof(&self) -> ProofResult<VerificationResult> {
        self.verify()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
}

impl TimestampableProof for IdentityAssuranceProof {
    fn proof_hash(&self) -> Vec<u8> {
        let bytes = self.to_bytes();
        Sha256::digest(&bytes).to_vec()
    }

    fn proof_type(&self) -> ProofType {
        ProofType::IdentityAssurance
    }

    fn verify_proof(&self) -> ProofResult<VerificationResult> {
        self.verify()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
}

impl TimestampableProof for VoteEligibilityProof {
    fn proof_hash(&self) -> Vec<u8> {
        let bytes = self.to_bytes();
        Sha256::digest(&bytes).to_vec()
    }

    fn proof_type(&self) -> ProofType {
        ProofType::VoteEligibility
    }

    fn verify_proof(&self) -> ProofResult<VerificationResult> {
        self.verify()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::{ProofConfig, SecurityLevel};
    use sha2::Digest;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_timestamp_config_builder() {
        let config = TimestampConfig::default()
            .with_validity(Duration::from_secs(600))
            .with_issuer("test-issuer")
            .with_audience("test-audience");

        assert_eq!(config.validity_duration, Duration::from_secs(600));
        assert_eq!(config.issuer, Some("test-issuer".to_string()));
        assert_eq!(config.audience, Some("test-audience".to_string()));
    }

    #[test]
    fn test_timestamp_config_presets() {
        let short = TimestampConfig::short_lived();
        assert_eq!(short.validity_duration, Duration::from_secs(300));

        let long = TimestampConfig::long_lived();
        assert_eq!(long.validity_duration, Duration::from_secs(86400));

        let single = TimestampConfig::single_use();
        assert!(single.nonce.is_some());
    }

    #[test]
    fn test_timestamped_range_proof() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let timestamped = TimestampedProof::new(proof, TimestampConfig::default());

        assert!(timestamped.is_valid());
        assert!(!timestamped.is_expired());

        let result = timestamped.verify().unwrap();
        assert!(result.is_valid());
        assert!(result.timestamp_valid);
        assert!(result.binding_valid);
    }

    #[test]
    fn test_expired_proof() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();

        // Create timestamp manually with past expiration (already expired)
        let proof_hash = proof.proof_hash();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let timestamp = ProofTimestamp {
            created_at: now - 3600,  // Created 1 hour ago
            expires_at: now - 1800,  // Expired 30 minutes ago
            nonce: None,
            issuer: None,
            audience: None,
            binding: {
                let mut hasher = sha2::Sha256::new();
                hasher.update(&proof_hash);
                hasher.update(&(now - 3600u64).to_le_bytes());
                hasher.update(&(now - 1800u64).to_le_bytes());
                let result = hasher.finalize();
                let mut b = [0u8; 32];
                b.copy_from_slice(&result);
                b
            },
        };

        let timestamped = TimestampedProof::with_timestamp(proof, timestamp);

        assert!(timestamped.is_expired());
        assert!(!timestamped.is_valid());

        let result = timestamped.verify().unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_nonce_binding() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let config = TimestampConfig::single_use();
        let timestamped = TimestampedProof::new(proof, config);

        assert!(timestamped.nonce().is_some());

        let result = timestamped.verify().unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_audience_verification() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let config = TimestampConfig::default().with_audience("aggregator-1");
        let timestamped = TimestampedProof::new(proof, config);

        // Correct audience
        let result = timestamped.verify_for_audience("aggregator-1").unwrap();
        assert!(result.is_valid());

        // Wrong audience
        let result = timestamped.verify_for_audience("aggregator-2").unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_timestamp_serialization() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let config = TimestampConfig::default()
            .with_nonce([42u8; 32])
            .with_issuer("test-issuer")
            .with_audience("test-audience");

        let timestamped = TimestampedProof::new(proof, config);

        // Serialize timestamp
        let timestamp_bytes = timestamped.timestamp.to_bytes();

        // Deserialize
        let restored = ProofTimestamp::from_bytes(&timestamp_bytes).unwrap();

        assert_eq!(restored.created_at, timestamped.timestamp.created_at);
        assert_eq!(restored.expires_at, timestamped.timestamp.expires_at);
        assert_eq!(restored.nonce, timestamped.timestamp.nonce);
        assert_eq!(restored.issuer, timestamped.timestamp.issuer);
        assert_eq!(restored.audience, timestamped.timestamp.audience);
        assert_eq!(restored.binding, timestamped.timestamp.binding);
    }

    #[test]
    fn test_nonce_registry() {
        let mut registry = NonceRegistry::new(100);

        let nonce1 = [1u8; 32];
        let nonce2 = [2u8; 32];
        let future_expiry = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;

        // First use should succeed
        assert!(registry.check_and_register(&nonce1, future_expiry));
        assert!(!registry.is_empty());

        // Second use should fail (replay)
        assert!(!registry.check_and_register(&nonce1, future_expiry));

        // Different nonce should succeed
        assert!(registry.check_and_register(&nonce2, future_expiry));

        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_nonce_registry_cleanup() {
        let mut registry = NonceRegistry::new(100);

        let nonce1 = [1u8; 32];
        let past_expiry = 0; // Already expired

        registry.check_and_register(&nonce1, past_expiry);
        assert_eq!(registry.len(), 1);

        registry.cleanup_expired();
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_remaining_validity() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let config = TimestampConfig::default().with_validity(Duration::from_secs(3600));
        let timestamped = TimestampedProof::new(proof, config);

        let remaining = timestamped.remaining_validity().unwrap();
        assert!(remaining.as_secs() > 3500); // Should be close to 1 hour
        assert!(remaining.as_secs() <= 3600);
    }

    #[test]
    fn test_proof_age() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let timestamped = TimestampedProof::new(proof, TimestampConfig::default());

        let age = timestamped.age();
        assert!(age.as_secs() < 5); // Should be very young
    }

    #[test]
    fn test_timestamped_gradient_proof() {
        let gradients = vec![0.1, 0.2, -0.1, 0.05, 0.15];
        let proof = GradientIntegrityProof::generate(&gradients, 10.0, test_config()).unwrap();
        let timestamped = TimestampedProof::new(proof, TimestampConfig::default());

        let result = timestamped.verify().unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_binding_tampering_detection() {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        let config = TimestampConfig::default();
        let mut timestamped = TimestampedProof::new(proof, config);

        // Tamper with the binding
        timestamped.timestamp.binding[0] ^= 0xFF;

        let result = timestamped.verify().unwrap();
        assert!(!result.is_valid());
        assert!(!result.binding_valid);
    }

    #[test]
    fn test_malformed_timestamp_bytes_empty() {
        // Empty bytes should return error, not panic
        let result = ProofTimestamp::from_bytes(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ProofError::InvalidProofFormat(_)));
    }

    #[test]
    fn test_malformed_timestamp_bytes_truncated() {
        // Truncated bytes (only partial timestamp) should return error
        let truncated = [0u8; 10]; // Only 10 bytes, need at least created_at + expires_at
        let result = ProofTimestamp::from_bytes(&truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_malformed_timestamp_bytes_bad_nonce_flag() {
        // Valid header but truncated nonce data
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&100u64.to_le_bytes()); // created_at
        bytes.extend_from_slice(&200u64.to_le_bytes()); // expires_at
        bytes.push(1); // has_nonce = true
        // But no nonce data follows - should error gracefully
        let result = ProofTimestamp::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_malformed_timestamp_bytes_bad_string_length() {
        // Valid header but claims impossibly long string
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&100u64.to_le_bytes()); // created_at
        bytes.extend_from_slice(&200u64.to_le_bytes()); // expires_at
        bytes.push(0); // has_nonce = false
        bytes.extend_from_slice(&1000u16.to_le_bytes()); // issuer_len = 1000 (but no data)

        let result = ProofTimestamp::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_malformed_timestamp_bytes_invalid_utf8() {
        // Valid structure but invalid UTF-8 in issuer
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&100u64.to_le_bytes()); // created_at
        bytes.extend_from_slice(&200u64.to_le_bytes()); // expires_at
        bytes.push(0); // has_nonce = false
        bytes.extend_from_slice(&4u16.to_le_bytes()); // issuer_len = 4
        bytes.extend_from_slice(&[0xFF, 0xFE, 0x00, 0x01]); // Invalid UTF-8
        bytes.extend_from_slice(&0u16.to_le_bytes()); // audience_len = 0
        bytes.extend_from_slice(&[0u8; 32]); // binding

        let result = ProofTimestamp::from_bytes(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ProofError::InvalidProofFormat(msg) if msg.contains("UTF-8")));
    }
}
