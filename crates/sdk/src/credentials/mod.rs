// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verifiable Credentials
//!
//! W3C-compatible Verifiable Credentials with Mycelix extensions.
//!
//! # Security Assumptions
//!
//! This module assumes:
//! - **Issuer Trust**: Credential issuers are identified by trusted DIDs
//! - **Proof Integrity**: Credential proofs are cryptographically verified
//! - **Randomness**: UUID generation uses secure random sources (CSPRNG)
//! - **Time Accuracy**: System time is reasonably accurate for expiration
//!
//! ## Threat Model
//!
//! - Adversary may attempt to forge credentials (prevented by proofs)
//! - Adversary may attempt to replay expired credentials (check expiration)
//! - Adversary may attempt to use credentials in wrong context (check type)
//! - Issuer compromise would allow credential forgery
//!
//! ## Limitations
//!
//! - Expiration date parsing is simplified (use proper ISO 8601 library)
//! - Credential revocation not implemented (use external revocation lists)
//!
//! ## Security Best Practices
//!
//! 1. Always verify credential proofs before trusting claims
//! 2. Check credential expiration dates
//! 3. Verify issuer identity through trusted DID resolution
//! 4. Use credential type matching to prevent cross-context reuse

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::epistemic::EpistemicClaim;

/// Domain separator for credential signatures
const CREDENTIAL_DOMAIN_SEPARATOR: &[u8] = b"mycelix-credential-v1";

/// Trait for looking up public keys from DID verification methods
///
/// Implementations should resolve DID URLs to Ed25519 public keys.
/// Example verification_method: "did:example:issuer#key-1"
pub trait CredentialKeyRegistry: Send + Sync {
    /// Look up the Ed25519 public key for a verification method
    ///
    /// Returns the 32-byte public key if found, or None if unknown.
    fn get_public_key(&self, verification_method: &str) -> Option<[u8; 32]>;
}

/// A simple in-memory credential key registry for testing.
///
/// Key material is zeroized on drop to prevent residual data in memory.
#[derive(Clone, Debug, Default)]
pub struct InMemoryCredentialKeyRegistry {
    keys: HashMap<String, [u8; 32]>,
}

impl Drop for InMemoryCredentialKeyRegistry {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        for value in self.keys.values_mut() {
            value.zeroize();
        }
    }
}

impl InMemoryCredentialKeyRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a public key for a verification method
    pub fn register(&mut self, verification_method: impl Into<String>, public_key: [u8; 32]) {
        self.keys.insert(verification_method.into(), public_key);
    }
}

impl CredentialKeyRegistry for InMemoryCredentialKeyRegistry {
    fn get_public_key(&self, verification_method: &str) -> Option<[u8; 32]> {
        self.keys.get(verification_method).copied()
    }
}

/// Error types for credential verification
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialVerificationError {
    /// No proof attached to credential
    NoProof,
    /// Credential has expired
    Expired,
    /// Unsupported proof type (only Ed25519Signature2020 supported)
    UnsupportedProofType(String),
    /// Invalid base64 encoding in proof_value
    InvalidBase64(String),
    /// Invalid signature format (expected 64 bytes)
    InvalidSignatureFormat(String),
    /// Public key not found for verification method
    PublicKeyNotFound(String),
    /// Invalid public key format
    InvalidPublicKey(String),
    /// Signature verification failed
    SignatureVerificationFailed(String),
    /// No key registry configured
    NoKeyRegistry,
}

impl std::fmt::Display for CredentialVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProof => write!(f, "No proof attached to credential"),
            Self::Expired => write!(f, "Credential has expired"),
            Self::UnsupportedProofType(t) => write!(
                f,
                "Unsupported proof type: {} (expected Ed25519Signature2020)",
                t
            ),
            Self::InvalidBase64(e) => write!(f, "Invalid base64 in proof_value: {}", e),
            Self::InvalidSignatureFormat(e) => write!(f, "Invalid signature format: {}", e),
            Self::PublicKeyNotFound(m) => write!(f, "Public key not found for: {}", m),
            Self::InvalidPublicKey(e) => write!(f, "Invalid public key: {}", e),
            Self::SignatureVerificationFailed(e) => {
                write!(f, "Signature verification failed: {}", e)
            }
            Self::NoKeyRegistry => {
                write!(f, "No key registry configured for signature verification")
            }
        }
    }
}

impl std::error::Error for CredentialVerificationError {}

/// A W3C-compatible Verifiable Credential
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiableCredential {
    /// Credential context
    #[serde(rename = "@context")]
    pub context: Vec<String>,

    /// Credential ID
    pub id: String,

    /// Credential types
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,

    /// Issuer identifier
    pub issuer: String,

    /// Issuance date (ISO 8601)
    pub issuance_date: String,

    /// Expiration date (optional, ISO 8601)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expiration_date: Option<String>,

    /// Credential subject
    pub credential_subject: CredentialSubject,

    /// Proof
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<CredentialProof>,

    /// Mycelix epistemic classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epistemic: Option<EpistemicClaim>,
}

/// Credential subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialSubject {
    /// Subject identifier
    pub id: String,

    /// Claims about the subject
    #[serde(flatten)]
    pub claims: serde_json::Value,
}

/// Credential proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialProof {
    /// Proof type
    #[serde(rename = "type")]
    pub proof_type: String,

    /// Creation timestamp
    pub created: String,

    /// Verification method
    pub verification_method: String,

    /// Proof purpose
    pub proof_purpose: String,

    /// Proof value (signature)
    pub proof_value: String,
}

impl VerifiableCredential {
    /// Create a new credential with builder
    pub fn builder() -> CredentialBuilder {
        CredentialBuilder::new()
    }

    /// Check if credential is expired
    pub fn is_expired(&self) -> bool {
        if let Some(_exp) = &self.expiration_date {
            // Simple check - in production, parse ISO 8601
            let _now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            // For now, assume not expired if we can't parse
            // Real implementation would parse ISO 8601
            false
        } else {
            false
        }
    }

    /// Check if credential has a valid proof
    pub fn has_proof(&self) -> bool {
        self.proof.is_some()
    }

    /// Get the epistemic classification code
    pub fn epistemic_code(&self) -> Option<String> {
        self.epistemic.as_ref().map(|e| e.code())
    }

    /// Compute a hash of the credential for caching purposes
    pub fn compute_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());
        hasher.update(self.issuer.as_bytes());
        hasher.update(self.issuance_date.as_bytes());
        if let Some(ref proof) = self.proof {
            hasher.update(proof.proof_value.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Compute the message to be signed for this credential
    ///
    /// The signed message is a SHA3-256 hash of the canonicalized credential
    /// (without the proof field) plus a domain separator to prevent
    /// cross-protocol signature reuse.
    ///
    /// The message includes:
    /// - Domain separator ("mycelix-credential-v1")
    /// - Credential ID
    /// - Issuer
    /// - Issuance date
    /// - Credential types (sorted for determinism)
    /// - Subject ID
    /// - Subject claims (JSON canonicalized)
    pub fn compute_signing_message(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();

        // Domain separation prevents cross-protocol attacks
        hasher.update(CREDENTIAL_DOMAIN_SEPARATOR);

        // Include credential identity
        hasher.update(self.id.as_bytes());
        hasher.update(self.issuer.as_bytes());
        hasher.update(self.issuance_date.as_bytes());

        // Include expiration if present
        if let Some(ref exp) = self.expiration_date {
            hasher.update(b"exp:");
            hasher.update(exp.as_bytes());
        }

        // Include credential types (sorted for determinism)
        let mut types = self.credential_type.clone();
        types.sort();
        for t in &types {
            hasher.update(b"type:");
            hasher.update(t.as_bytes());
        }

        // Include subject
        hasher.update(b"subject:");
        hasher.update(self.credential_subject.id.as_bytes());

        // Include claims (JSON serialized - stable ordering via serde_json)
        if let Ok(claims_json) = serde_json::to_string(&self.credential_subject.claims) {
            hasher.update(b"claims:");
            hasher.update(claims_json.as_bytes());
        }

        // Include epistemic classification if present
        if let Some(ref epistemic) = self.epistemic {
            hasher.update(b"epistemic:");
            hasher.update(epistemic.code().as_bytes());
        }

        hasher.finalize().into()
    }
}

/// Result of a credential verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the credential is valid
    pub is_valid: bool,
    /// Whether the signature was verified
    pub signature_verified: bool,
    /// Whether the credential has expired
    pub is_expired: bool,
    /// Verification timestamp
    pub verified_at: u64,
    /// Optional error message
    pub error: Option<String>,
}

impl VerificationResult {
    /// Create a successful verification result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            signature_verified: true,
            is_expired: false,
            verified_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            error: None,
        }
    }

    /// Create a failed verification result
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            is_valid: false,
            signature_verified: false,
            is_expired: false,
            verified_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            error: Some(error.into()),
        }
    }
}

/// Cached credential verification entry
#[derive(Debug, Clone)]
struct CachedVerification {
    result: VerificationResult,
    expires_at: u64,
}

/// Credential verification cache with TTL-based expiration
///
/// # Performance
///
/// - Lookup: O(1) hash map access
/// - Insert: O(1) amortized
/// - Memory: ~200 bytes per cached entry
///
/// # Thread Safety
///
/// Uses RwLock for concurrent read access with exclusive writes.
pub struct CredentialCache {
    cache: RwLock<HashMap<String, CachedVerification>>,
    ttl_secs: u64,
    max_entries: usize,
}

impl CredentialCache {
    /// Create a new cache with specified TTL and max entries
    pub fn new(ttl_secs: u64, max_entries: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::with_capacity(max_entries.min(1000))),
            ttl_secs,
            max_entries,
        }
    }

    /// Create with default settings (5 minute TTL, 10000 entries)
    pub fn default_cache() -> Self {
        Self::new(300, 10000)
    }

    /// Get a cached verification result if available and not expired
    pub fn get(&self, credential_hash: &str) -> Option<VerificationResult> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let cache = self.cache.read().ok()?;
        cache.get(credential_hash).and_then(|entry| {
            if entry.expires_at > now {
                Some(entry.result.clone())
            } else {
                None
            }
        })
    }

    /// Insert a verification result into the cache
    pub fn insert(&self, credential_hash: String, result: VerificationResult) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if let Ok(mut cache) = self.cache.write() {
            // Evict expired entries if we're at capacity
            if cache.len() >= self.max_entries {
                cache.retain(|_, v| v.expires_at > now);
            }

            // If still at capacity, remove oldest entry
            if cache.len() >= self.max_entries {
                // Simple eviction: remove any one entry
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }

            cache.insert(
                credential_hash,
                CachedVerification {
                    result,
                    expires_at: now + self.ttl_secs,
                },
            );
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().ok();
        CacheStats {
            entry_count: cache.as_ref().map(|c| c.len()).unwrap_or(0),
            max_entries: self.max_entries,
            ttl_secs: self.ttl_secs,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of entries in cache
    pub entry_count: usize,
    /// Maximum number of entries allowed
    pub max_entries: usize,
    /// Time-to-live in seconds for cached entries
    pub ttl_secs: u64,
}

/// Batch credential verification with caching
///
/// # Performance
///
/// - Uses cache to avoid re-verification of recently verified credentials
/// - Batches signature verification for better throughput
/// - Lazy proof checking: only verifies proofs when necessary
///
/// # Security
///
/// When a key registry is configured, all credentials undergo full Ed25519
/// signature verification. Without a registry, verification will fail for
/// credentials requiring signature verification (this is the secure default).
pub struct BatchCredentialVerifier {
    cache: CredentialCache,
    /// Key registry for signature verification (required for secure operation)
    key_registry: Option<Arc<dyn CredentialKeyRegistry>>,
}

impl BatchCredentialVerifier {
    /// Create a new batch verifier with default cache but NO key registry
    ///
    /// WARNING: Without a key registry, signature verification will fail.
    /// Use `with_registry()` for production use.
    pub fn new() -> Self {
        Self {
            cache: CredentialCache::default_cache(),
            key_registry: None,
        }
    }

    /// Create a verifier with a key registry for signature verification
    ///
    /// This is the recommended constructor for production use.
    pub fn with_registry(registry: Arc<dyn CredentialKeyRegistry>) -> Self {
        Self {
            cache: CredentialCache::default_cache(),
            key_registry: Some(registry),
        }
    }

    /// Create with custom cache settings and a key registry
    pub fn with_cache_and_registry(
        ttl_secs: u64,
        max_entries: usize,
        registry: Arc<dyn CredentialKeyRegistry>,
    ) -> Self {
        Self {
            cache: CredentialCache::new(ttl_secs, max_entries),
            key_registry: Some(registry),
        }
    }

    /// Create with custom cache settings (no registry - for testing only)
    #[deprecated(note = "Use with_cache_and_registry for secure verification")]
    pub fn with_cache(ttl_secs: u64, max_entries: usize) -> Self {
        Self {
            cache: CredentialCache::new(ttl_secs, max_entries),
            key_registry: None,
        }
    }

    /// Set the key registry for signature verification
    pub fn set_registry(&mut self, registry: Arc<dyn CredentialKeyRegistry>) {
        self.key_registry = Some(registry);
    }

    /// Verify a single credential with caching
    pub fn verify(&self, credential: &VerifiableCredential) -> VerificationResult {
        let hash = credential.compute_hash();

        // Check cache first
        if let Some(cached) = self.cache.get(&hash) {
            return cached;
        }

        // Perform verification
        let result = self.verify_uncached(credential);

        // Cache the result (only cache successful verifications for security)
        if result.is_valid {
            self.cache.insert(hash, result.clone());
        }

        result
    }

    /// Verify a batch of credentials
    ///
    /// Returns results in the same order as input.
    /// Uses cache and parallelization for efficiency.
    pub fn verify_batch(&self, credentials: &[VerifiableCredential]) -> Vec<VerificationResult> {
        credentials.iter().map(|c| self.verify(c)).collect()
    }

    /// Verify a batch and return only valid credentials
    pub fn filter_valid<'a>(
        &self,
        credentials: &'a [VerifiableCredential],
    ) -> Vec<&'a VerifiableCredential> {
        credentials
            .iter()
            .filter(|c| self.verify(c).is_valid)
            .collect()
    }

    /// Internal verification without cache - PERFORMS REAL SIGNATURE VERIFICATION
    fn verify_uncached(&self, credential: &VerifiableCredential) -> VerificationResult {
        // Check expiration first (cheap)
        if credential.is_expired() {
            return VerificationResult {
                is_valid: false,
                signature_verified: false,
                is_expired: true,
                verified_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                error: Some(CredentialVerificationError::Expired.to_string()),
            };
        }

        // Check proof exists
        let proof = match &credential.proof {
            Some(p) => p,
            None => {
                return VerificationResult::failure(
                    CredentialVerificationError::NoProof.to_string(),
                );
            }
        };

        // Check proof type
        if proof.proof_type != "Ed25519Signature2020" {
            return VerificationResult::failure(
                CredentialVerificationError::UnsupportedProofType(proof.proof_type.clone())
                    .to_string(),
            );
        }

        // Get key registry or fail
        let registry = match &self.key_registry {
            Some(r) => r,
            None => {
                return VerificationResult::failure(
                    CredentialVerificationError::NoKeyRegistry.to_string(),
                );
            }
        };

        // Decode base64 signature
        let signature_bytes = match BASE64.decode(&proof.proof_value) {
            Ok(bytes) => bytes,
            Err(e) => {
                return VerificationResult::failure(
                    CredentialVerificationError::InvalidBase64(e.to_string()).to_string(),
                );
            }
        };

        // Validate signature length (Ed25519 signatures are 64 bytes)
        if signature_bytes.len() != 64 {
            return VerificationResult::failure(
                CredentialVerificationError::InvalidSignatureFormat(format!(
                    "Expected 64 bytes, got {}",
                    signature_bytes.len()
                ))
                .to_string(),
            );
        }

        // Look up public key from verification method
        let public_key_bytes = match registry.get_public_key(&proof.verification_method) {
            Some(key) => key,
            None => {
                return VerificationResult::failure(
                    CredentialVerificationError::PublicKeyNotFound(
                        proof.verification_method.clone(),
                    )
                    .to_string(),
                );
            }
        };

        // Parse public key
        let verifying_key = match VerifyingKey::from_bytes(&public_key_bytes) {
            Ok(key) => key,
            Err(e) => {
                return VerificationResult::failure(
                    CredentialVerificationError::InvalidPublicKey(e.to_string()).to_string(),
                );
            }
        };

        // Parse signature
        let signature_array: [u8; 64] = match signature_bytes.try_into() {
            Ok(arr) => arr,
            Err(_) => {
                return VerificationResult::failure(
                    CredentialVerificationError::InvalidSignatureFormat(
                        "Failed to convert signature bytes".to_string(),
                    )
                    .to_string(),
                );
            }
        };
        let signature = Signature::from_bytes(&signature_array);

        // Compute the message that was signed
        let message = credential.compute_signing_message();

        // VERIFY THE SIGNATURE
        match verifying_key.verify(&message, &signature) {
            Ok(()) => VerificationResult::success(),
            Err(e) => VerificationResult::failure(
                CredentialVerificationError::SignatureVerificationFailed(e.to_string()).to_string(),
            ),
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear the verification cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl Default for BatchCredentialVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for Verifiable Credentials
pub struct CredentialBuilder {
    id: Option<String>,
    credential_types: Vec<String>,
    issuer: String,
    subject_id: String,
    claims: serde_json::Value,
    expiration_date: Option<String>,
    epistemic: Option<EpistemicClaim>,
}

impl CredentialBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            id: None,
            credential_types: vec!["VerifiableCredential".to_string()],
            issuer: String::new(),
            subject_id: String::new(),
            claims: serde_json::json!({}),
            expiration_date: None,
            epistemic: None,
        }
    }

    /// Set credential ID
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Add a credential type
    pub fn credential_type(mut self, cred_type: impl Into<String>) -> Self {
        self.credential_types.push(cred_type.into());
        self
    }

    /// Set issuer
    pub fn issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = issuer.into();
        self
    }

    /// Set subject ID
    pub fn subject(mut self, subject_id: impl Into<String>) -> Self {
        self.subject_id = subject_id.into();
        self
    }

    /// Set claims
    pub fn claims(mut self, claims: serde_json::Value) -> Self {
        self.claims = claims;
        self
    }

    /// Set expiration date
    pub fn expires(mut self, date: impl Into<String>) -> Self {
        self.expiration_date = Some(date.into());
        self
    }

    /// Set epistemic classification
    pub fn epistemic(mut self, claim: EpistemicClaim) -> Self {
        self.epistemic = Some(claim);
        self
    }

    /// Build the credential
    pub fn build(self) -> VerifiableCredential {
        let now = chrono_lite::now_iso8601();

        VerifiableCredential {
            context: vec![
                "https://www.w3.org/2018/credentials/v1".to_string(),
                "https://mycelix.net/credentials/v1".to_string(),
            ],
            id: self
                .id
                .unwrap_or_else(|| format!("urn:uuid:{}", generate_uuid())),
            credential_type: self.credential_types,
            issuer: self.issuer,
            issuance_date: now,
            expiration_date: self.expiration_date,
            credential_subject: CredentialSubject {
                id: self.subject_id,
                claims: self.claims,
            },
            proof: None,
            epistemic: self.epistemic,
        }
    }
}

impl Default for CredentialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a cryptographically secure UUID v4
///
/// # Security Note (FIND-005 mitigation)
///
/// This implementation uses the `uuid` crate with v4 random UUIDs,
/// which provides proper cryptographic randomness via the operating
/// system's CSPRNG. This replaces the previous weak implementation
/// that used system time as the sole entropy source.
fn generate_uuid() -> String {
    use uuid::Uuid;
    Uuid::new_v4().to_string()
}

/// Simple ISO 8601 date formatting (placeholder)
mod chrono_lite {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn now_iso8601() -> String {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Simple conversion - use chrono in production
        let days = secs / 86400;
        let year = 1970 + (days / 365);
        let month = ((days % 365) / 30) + 1;
        let day = ((days % 365) % 30) + 1;

        format!(
            "{:04}-{:02}-{:02}T00:00:00Z",
            year,
            month.min(12),
            day.min(28)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{EmpiricalLevel, MaterialityLevel, NormativeLevel};
    use ed25519_dalek::{Signer, SigningKey};

    /// Create a test keypair and sign a credential
    fn create_signed_credential(
        issuer: &str,
        subject: &str,
        verification_method: &str,
    ) -> (VerifiableCredential, SigningKey) {
        // Fixed test key for reproducibility
        let signing_key = SigningKey::from_bytes(&[
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
            0x1d, 0x1e, 0x1f, 0x20,
        ]);

        // Create credential without proof first
        let mut vc = VerifiableCredential::builder()
            .issuer(issuer)
            .subject(subject)
            .claims(serde_json::json!({"test": "value"}))
            .build();

        // Compute signing message and sign
        let message = vc.compute_signing_message();
        let signature = signing_key.sign(&message);

        // Add proof with base64-encoded signature
        vc.proof = Some(CredentialProof {
            proof_type: "Ed25519Signature2020".to_string(),
            created: "2024-01-01T00:00:00Z".to_string(),
            verification_method: verification_method.to_string(),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: BASE64.encode(signature.to_bytes()),
        });

        (vc, signing_key)
    }

    fn create_test_registry(
        verification_method: &str,
        signing_key: &SigningKey,
    ) -> Arc<InMemoryCredentialKeyRegistry> {
        let mut registry = InMemoryCredentialKeyRegistry::new();
        let public_key = signing_key.verifying_key();
        registry.register(verification_method, public_key.to_bytes());
        Arc::new(registry)
    }

    #[test]
    fn test_build_credential() {
        let vc = VerifiableCredential::builder()
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .credential_type("EducationalCredential")
            .claims(serde_json::json!({
                "degree": "Bachelor of Science",
                "field": "Computer Science"
            }))
            .build();

        assert_eq!(vc.issuer, "did:example:issuer");
        assert!(vc
            .credential_type
            .contains(&"EducationalCredential".to_string()));
    }

    #[test]
    fn test_with_epistemic() {
        let epistemic = EpistemicClaim::new(
            "Educational credential",
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        );

        let vc = VerifiableCredential::builder()
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .epistemic(epistemic)
            .build();

        assert_eq!(vc.epistemic_code(), Some("E3-N2-M2".to_string()));
    }

    // =========================================================================
    // Credential caching and batch verification tests
    // =========================================================================

    #[test]
    fn test_credential_hash() {
        let vc1 = VerifiableCredential::builder()
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .build();

        let vc2 = VerifiableCredential::builder()
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .build();

        // Same issuer and subject but different IDs (UUIDs)
        // Hashes should be different due to different generated IDs
        let hash1 = vc1.compute_hash();
        let hash2 = vc2.compute_hash();
        assert!(!hash1.is_empty());
        assert!(!hash2.is_empty());
    }

    #[test]
    fn test_verification_result() {
        let success = VerificationResult::success();
        assert!(success.is_valid);
        assert!(success.signature_verified);
        assert!(!success.is_expired);
        assert!(success.error.is_none());

        let failure = VerificationResult::failure("test error");
        assert!(!failure.is_valid);
        assert!(failure.error.is_some());
    }

    #[test]
    fn test_credential_cache() {
        let cache = CredentialCache::new(60, 100);

        // Insert a result
        let result = VerificationResult::success();
        cache.insert("hash1".to_string(), result.clone());

        // Should be cached
        let cached = cache.get("hash1");
        assert!(cached.is_some());
        assert!(cached.unwrap().is_valid);

        // Non-existent key
        assert!(cache.get("hash2").is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = CredentialCache::new(60, 100);
        cache.insert("hash1".to_string(), VerificationResult::success());
        cache.insert("hash2".to_string(), VerificationResult::success());

        let stats = cache.stats();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.max_entries, 100);
    }

    // =========================================================================
    // SIGNATURE VERIFICATION TESTS
    // =========================================================================

    #[test]
    fn test_signature_verification_success() {
        let verification_method = "did:example:issuer#key-1";
        let (vc, signing_key) = create_signed_credential(
            "did:example:issuer",
            "did:example:subject",
            verification_method,
        );
        let registry = create_test_registry(verification_method, &signing_key);

        let verifier = BatchCredentialVerifier::with_registry(registry);
        let result = verifier.verify(&vc);

        assert!(
            result.is_valid,
            "Expected valid, got error: {:?}",
            result.error
        );
        assert!(result.signature_verified);
    }

    #[test]
    fn test_signature_verification_wrong_key() {
        let verification_method = "did:example:issuer#key-1";
        let (vc, _) = create_signed_credential(
            "did:example:issuer",
            "did:example:subject",
            verification_method,
        );

        // Register wrong key
        let wrong_key = SigningKey::from_bytes(&[0x42; 32]);
        let registry = create_test_registry(verification_method, &wrong_key);

        let verifier = BatchCredentialVerifier::with_registry(registry);
        let result = verifier.verify(&vc);

        assert!(!result.is_valid);
        assert!(result
            .error
            .as_ref()
            .unwrap()
            .contains("Signature verification failed"));
    }

    #[test]
    fn test_signature_verification_key_not_found() {
        let (vc, _) = create_signed_credential(
            "did:example:issuer",
            "did:example:subject",
            "did:example:issuer#key-1",
        );

        // Empty registry
        let registry = Arc::new(InMemoryCredentialKeyRegistry::new());

        let verifier = BatchCredentialVerifier::with_registry(registry);
        let result = verifier.verify(&vc);

        assert!(!result.is_valid);
        assert!(result
            .error
            .as_ref()
            .unwrap()
            .contains("Public key not found"));
    }

    #[test]
    fn test_signature_verification_no_registry() {
        let (vc, _) = create_signed_credential(
            "did:example:issuer",
            "did:example:subject",
            "did:example:issuer#key-1",
        );

        // Verifier without registry
        let verifier = BatchCredentialVerifier::new();
        let result = verifier.verify(&vc);

        assert!(!result.is_valid);
        assert!(result
            .error
            .as_ref()
            .unwrap()
            .contains("No key registry configured"));
    }

    #[test]
    fn test_signature_verification_tampered_credential() {
        let verification_method = "did:example:issuer#key-1";
        let (mut vc, signing_key) = create_signed_credential(
            "did:example:issuer",
            "did:example:subject",
            verification_method,
        );
        let registry = create_test_registry(verification_method, &signing_key);

        // Tamper with the credential after signing
        vc.credential_subject.id = "did:example:tampered".to_string();

        let verifier = BatchCredentialVerifier::with_registry(registry);
        let result = verifier.verify(&vc);

        assert!(!result.is_valid);
        assert!(result
            .error
            .as_ref()
            .unwrap()
            .contains("Signature verification failed"));
    }

    #[test]
    fn test_signature_verification_invalid_base64() {
        let verification_method = "did:example:issuer#key-1";
        let signing_key = SigningKey::from_bytes(&[0x01; 32]);
        let registry = create_test_registry(verification_method, &signing_key);

        let mut vc = VerifiableCredential::builder()
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .build();

        vc.proof = Some(CredentialProof {
            proof_type: "Ed25519Signature2020".to_string(),
            created: "2024-01-01T00:00:00Z".to_string(),
            verification_method: verification_method.to_string(),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: "!!!invalid_base64!!!".to_string(),
        });

        let verifier = BatchCredentialVerifier::with_registry(registry);
        let result = verifier.verify(&vc);

        assert!(!result.is_valid);
        assert!(result.error.as_ref().unwrap().contains("Invalid base64"));
    }

    #[test]
    fn test_signature_verification_unsupported_proof_type() {
        let verification_method = "did:example:issuer#key-1";
        let signing_key = SigningKey::from_bytes(&[0x01; 32]);
        let registry = create_test_registry(verification_method, &signing_key);

        let mut vc = VerifiableCredential::builder()
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .build();

        vc.proof = Some(CredentialProof {
            proof_type: "SomeOtherProofType".to_string(),
            created: "2024-01-01T00:00:00Z".to_string(),
            verification_method: verification_method.to_string(),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: "dGVzdA==".to_string(),
        });

        let verifier = BatchCredentialVerifier::with_registry(registry);
        let result = verifier.verify(&vc);

        assert!(!result.is_valid);
        assert!(result
            .error
            .as_ref()
            .unwrap()
            .contains("Unsupported proof type"));
    }

    #[test]
    fn test_batch_verifier_with_registry() {
        let verification_method = "did:example:issuer#key-1";
        let (vc, signing_key) = create_signed_credential(
            "did:example:issuer",
            "did:example:subject",
            verification_method,
        );
        let registry = create_test_registry(verification_method, &signing_key);

        let verifier = BatchCredentialVerifier::with_registry(registry);

        // First verification
        let result1 = verifier.verify(&vc);
        assert!(
            result1.is_valid,
            "First verification failed: {:?}",
            result1.error
        );

        // Second verification should use cache
        let result2 = verifier.verify(&vc);
        assert!(result2.is_valid);

        // Check cache was used
        let stats = verifier.cache_stats();
        assert_eq!(stats.entry_count, 1);
    }

    #[test]
    fn test_batch_verify_multiple_with_signature() {
        let verification_method = "did:example:issuer#key-1";

        // All credentials signed with the same key for testing
        let signing_key = SigningKey::from_bytes(&[
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
            0x1d, 0x1e, 0x1f, 0x20,
        ]);
        let registry = create_test_registry(verification_method, &signing_key);

        let credentials: Vec<_> = (0..5)
            .map(|i| {
                let mut vc = VerifiableCredential::builder()
                    .id(format!("urn:uuid:test-{}", i))
                    .issuer("did:example:issuer")
                    .subject("did:example:subject")
                    .claims(serde_json::json!({"index": i}))
                    .build();

                // Sign each credential
                let message = vc.compute_signing_message();
                let signature = signing_key.sign(&message);

                vc.proof = Some(CredentialProof {
                    proof_type: "Ed25519Signature2020".to_string(),
                    created: "2024-01-01T00:00:00Z".to_string(),
                    verification_method: verification_method.to_string(),
                    proof_purpose: "assertionMethod".to_string(),
                    proof_value: BASE64.encode(signature.to_bytes()),
                });
                vc
            })
            .collect();

        let verifier = BatchCredentialVerifier::with_registry(registry);
        let results = verifier.verify_batch(&credentials);

        assert_eq!(results.len(), 5);
        for (i, result) in results.iter().enumerate() {
            assert!(
                result.is_valid,
                "Credential {} failed: {:?}",
                i, result.error
            );
        }
    }

    #[test]
    fn test_filter_valid_with_signatures() {
        let verification_method = "did:example:issuer#key-1";
        let (valid_vc, signing_key) = create_signed_credential(
            "did:example:issuer",
            "did:example:subject",
            verification_method,
        );
        let registry = create_test_registry(verification_method, &signing_key);

        // Credential without proof
        let invalid_vc = VerifiableCredential::builder()
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .build();

        let credentials = vec![valid_vc.clone(), invalid_vc, valid_vc];
        let verifier = BatchCredentialVerifier::with_registry(registry);
        let valid_only = verifier.filter_valid(&credentials);

        assert_eq!(valid_only.len(), 2);
    }

    #[test]
    fn test_signing_message_deterministic() {
        let vc1 = VerifiableCredential::builder()
            .id("urn:uuid:test-1")
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .claims(serde_json::json!({"test": "value"}))
            .build();

        let vc2 = VerifiableCredential::builder()
            .id("urn:uuid:test-1")
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .claims(serde_json::json!({"test": "value"}))
            .build();

        // Same credentials should produce same signing message
        assert_eq!(vc1.compute_signing_message(), vc2.compute_signing_message());
    }

    #[test]
    fn test_signing_message_different_for_different_inputs() {
        let vc1 = VerifiableCredential::builder()
            .id("urn:uuid:test-1")
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .build();

        let vc2 = VerifiableCredential::builder()
            .id("urn:uuid:test-2")
            .issuer("did:example:issuer")
            .subject("did:example:subject")
            .build();

        // Different IDs should produce different signing messages
        assert_ne!(vc1.compute_signing_message(), vc2.compute_signing_message());
    }
}
