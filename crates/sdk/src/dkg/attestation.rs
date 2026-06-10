// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DKG Attestation System
//!
//! Attestations are claims that a validator vouches for the accuracy of a triple.
//! They include:
//! - The attester's identifier (DID or AgentPubKey)
//! - The attester's reputation at time of attestation
//! - The type of attestation (endorsement, challenge, neutral)
//! - Optional evidence supporting the attestation
//! - Cryptographic signature proving authenticity (MANDATORY)
//!
//! # Security Model
//!
//! All attestations MUST be cryptographically signed using Ed25519. The signature
//! covers:
//! - Triple hash (SHA3-256 commitment)
//! - Attester identifier
//! - Timestamp
//! - Attestation type
//!
//! This prevents:
//! - Attestation forgery
//! - Sybil attacks
//! - Historical attestation manipulation
//!
//! # Integration with MATL
//!
//! Attestations integrate with the Mycelix Adaptive Trust Layer:
//! - Reputation scores are derived from K-Vector dimensions
//! - Endorsement weight uses reputation squared voting
//! - Challenges trigger RB-BFT consensus verification

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Domain separator for attestation signatures
const ATTESTATION_DOMAIN_SEPARATOR: &[u8] = b"mycelix-attestation-v1";

/// Error types for attestation operations
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttestationError {
    /// Triple hash cannot be empty
    EmptyTripleHash,
    /// Attester identifier cannot be empty
    EmptyAttester,
    /// Too many supporting sources
    TooManySources(usize),
    /// Evidence is too long
    EvidenceTooLong(usize),
    /// Signature is empty
    EmptySignature,
    /// Invalid signature format
    InvalidSignatureFormat(String),
    /// Signature verification failed
    SignatureVerificationFailed(String),
    /// Public key not found for attester
    PublicKeyNotFound(String),
    /// Invalid public key format
    InvalidPublicKey(String),
    /// Rate limited - too many attestations in short time period
    RateLimited {
        /// Time to wait before next attestation is allowed
        retry_after: Duration,
    },
}

impl std::fmt::Display for AttestationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTripleHash => write!(f, "Triple hash cannot be empty"),
            Self::EmptyAttester => write!(f, "Attester cannot be empty"),
            Self::TooManySources(n) => write!(f, "Too many supporting sources: {} (max 10)", n),
            Self::EvidenceTooLong(n) => write!(f, "Evidence too long: {} chars (max 2000)", n),
            Self::EmptySignature => write!(f, "Signature cannot be empty"),
            Self::InvalidSignatureFormat(e) => write!(f, "Invalid signature format: {}", e),
            Self::SignatureVerificationFailed(e) => {
                write!(f, "Signature verification failed: {}", e)
            }
            Self::PublicKeyNotFound(id) => write!(f, "Public key not found for attester: {}", id),
            Self::InvalidPublicKey(e) => write!(f, "Invalid public key format: {}", e),
            Self::RateLimited { retry_after } => {
                write!(f, "Rate limited: retry after {:?}", retry_after)
            }
        }
    }
}

impl std::error::Error for AttestationError {}

/// Trait for looking up attester public keys
///
/// Implementations should integrate with the existing identity system
/// (e.g., Holochain agent keys, DID resolution, etc.)
pub trait PublicKeyRegistry: Send + Sync {
    /// Look up the Ed25519 public key for an attester
    ///
    /// Returns the 32-byte public key if found, or None if the attester is unknown.
    fn get_public_key(&self, attester_id: &str) -> Option<[u8; 32]>;
}

/// A simple in-memory public key registry for testing
#[derive(Clone, Debug, Default)]
pub struct InMemoryKeyRegistry {
    keys: std::collections::HashMap<String, [u8; 32]>,
}

impl InMemoryKeyRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a public key for an attester
    pub fn register(&mut self, attester_id: impl Into<String>, public_key: [u8; 32]) {
        self.keys.insert(attester_id.into(), public_key);
    }
}

impl PublicKeyRegistry for InMemoryKeyRegistry {
    fn get_public_key(&self, attester_id: &str) -> Option<[u8; 32]> {
        self.keys.get(attester_id).copied()
    }
}

/// Types of attestations
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationType {
    /// Strong agreement: attester believes claim is accurate
    Endorse,
    /// Strong disagreement: attester believes claim is false
    Challenge,
    /// Neutral: attester has seen the claim but takes no position
    Acknowledge,
    /// Partial: attester agrees with some aspects but not all
    Partial {
        /// Agreement level from 0-100 percentage
        agreement_level: u8,
    },
}

impl AttestationType {
    /// Get weight multiplier for confidence calculation
    ///
    /// - Endorse: +1.0 (full positive)
    /// - Challenge: -1.0 (full negative, creates contradiction)
    /// - Acknowledge: +0.1 (small positive)
    /// - Partial: scaled by agreement level
    pub fn weight(&self) -> f64 {
        match self {
            AttestationType::Endorse => 1.0,
            AttestationType::Challenge => -1.0,
            AttestationType::Acknowledge => 0.1,
            AttestationType::Partial { agreement_level } => {
                (*agreement_level as f64 / 100.0) * 2.0 - 1.0 // Maps 0-100 to -1.0 to +1.0
            }
        }
    }

    /// Check if this attestation type creates a contradiction
    pub fn is_contradiction(&self) -> bool {
        matches!(self, AttestationType::Challenge)
    }

    /// Get a byte representation for signing
    fn to_signing_bytes(&self) -> Vec<u8> {
        match self {
            AttestationType::Endorse => vec![0x01],
            AttestationType::Challenge => vec![0x02],
            AttestationType::Acknowledge => vec![0x03],
            AttestationType::Partial { agreement_level } => vec![0x04, *agreement_level],
        }
    }
}

/// An attestation record with mandatory cryptographic signature
///
/// # Security
///
/// This struct contains a cryptographic signature which is sensitive key material.
/// The `ZeroizeOnDrop` derive ensures that the signature bytes are securely
/// zeroed when the struct is dropped, preventing memory scraping attacks
/// from recovering signature data.
#[derive(Clone, Debug, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct Attestation {
    /// SHA3-256 commitment hash of the triple being attested
    #[zeroize(skip)]
    pub triple_hash: String,
    /// Identifier of the attester (DID or AgentPubKey)
    #[zeroize(skip)]
    pub attester: String,
    /// Type of attestation
    #[zeroize(skip)]
    pub attestation_type: AttestationType,
    /// Attester's reputation at time of attestation
    #[zeroize(skip)]
    pub attester_reputation: f64,
    /// Optional evidence/reasoning
    #[zeroize(skip)]
    pub evidence: Option<String>,
    /// Optional sources supporting the attestation
    #[zeroize(skip)]
    pub supporting_sources: Vec<String>,
    /// Unix timestamp of attestation
    #[zeroize(skip)]
    pub created_at: u64,
    /// Ed25519 signature proving the attestation is from the claimed attester (SENSITIVE - will be zeroized on drop)
    pub signature: Vec<u8>,
}

impl Attestation {
    /// Create a new endorsement attestation
    ///
    /// Note: The signature must be provided via `with_signature()` before validation.
    pub fn endorse(
        triple_hash: impl Into<String>,
        attester: impl Into<String>,
        signature: Vec<u8>,
    ) -> Self {
        Self {
            triple_hash: triple_hash.into(),
            attester: attester.into(),
            attestation_type: AttestationType::Endorse,
            attester_reputation: 0.0,
            evidence: None,
            supporting_sources: Vec::new(),
            created_at: 0,
            signature,
        }
    }

    /// Create a new challenge attestation
    pub fn challenge(
        triple_hash: impl Into<String>,
        attester: impl Into<String>,
        signature: Vec<u8>,
    ) -> Self {
        Self {
            triple_hash: triple_hash.into(),
            attester: attester.into(),
            attestation_type: AttestationType::Challenge,
            attester_reputation: 0.0,
            evidence: None,
            supporting_sources: Vec::new(),
            created_at: 0,
            signature,
        }
    }

    /// Create a new acknowledgment attestation
    pub fn acknowledge(
        triple_hash: impl Into<String>,
        attester: impl Into<String>,
        signature: Vec<u8>,
    ) -> Self {
        Self {
            triple_hash: triple_hash.into(),
            attester: attester.into(),
            attestation_type: AttestationType::Acknowledge,
            attester_reputation: 0.0,
            evidence: None,
            supporting_sources: Vec::new(),
            created_at: 0,
            signature,
        }
    }

    /// Builder: set reputation
    pub fn with_reputation(mut self, rep: f64) -> Self {
        self.attester_reputation = rep.clamp(0.0, 1.0);
        self
    }

    /// Builder: set evidence
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence = Some(evidence.into());
        self
    }

    /// Builder: add supporting source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.supporting_sources.push(source.into());
        self
    }

    /// Builder: set timestamp
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.created_at = ts;
        self
    }

    /// Builder: set signature
    pub fn with_signature(mut self, sig: Vec<u8>) -> Self {
        self.signature = sig;
        self
    }

    /// Compute the message to be signed for this attestation
    ///
    /// The signed message includes:
    /// - Domain separator ("mycelix-attestation-v1")
    /// - Triple hash (as UTF-8 bytes)
    /// - Attester ID (as UTF-8 bytes)
    /// - Timestamp (8 bytes, little-endian)
    /// - Attestation type (variable length encoding)
    ///
    /// The result is SHA3-256 hashed to produce a fixed-size message.
    pub fn compute_signing_message(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();

        // Domain separation prevents cross-protocol attacks
        hasher.update(ATTESTATION_DOMAIN_SEPARATOR);

        // Include triple hash (cryptographic commitment to the triple)
        hasher.update(self.triple_hash.as_bytes());

        // Include attester identity
        hasher.update(self.attester.as_bytes());

        // Include timestamp (prevents replay with different times)
        hasher.update(self.created_at.to_le_bytes());

        // Include attestation type
        hasher.update(self.attestation_type.to_signing_bytes());

        hasher.finalize().into()
    }

    /// Compute SHA3-256 commitment hash for a triple
    ///
    /// This should be used to generate the `triple_hash` field from the actual triple data.
    /// The commitment binds the attestation to a specific triple.
    pub fn compute_triple_commitment(
        subject: &str,
        predicate: &str,
        object: &str,
        creator: &str,
        created_at: u64,
    ) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(b"mycelix-triple-commitment-v1");
        hasher.update(subject.as_bytes());
        hasher.update(predicate.as_bytes());
        hasher.update(object.as_bytes());
        hasher.update(creator.as_bytes());
        hasher.update(created_at.to_le_bytes());

        // Return hex-encoded hash
        let hash = hasher.finalize();
        hex::encode(hash)
    }

    /// Verify the signature against the attester's public key
    ///
    /// Returns Ok(()) if verification succeeds, or an error describing the failure.
    pub fn verify_signature(
        &self,
        registry: &dyn PublicKeyRegistry,
    ) -> Result<(), AttestationError> {
        // Look up the attester's public key
        let public_key_bytes = registry
            .get_public_key(&self.attester)
            .ok_or_else(|| AttestationError::PublicKeyNotFound(self.attester.clone()))?;

        // Parse the public key
        let verifying_key = VerifyingKey::from_bytes(&public_key_bytes)
            .map_err(|e| AttestationError::InvalidPublicKey(e.to_string()))?;

        // Parse the signature
        if self.signature.len() != 64 {
            return Err(AttestationError::InvalidSignatureFormat(format!(
                "Expected 64 bytes, got {}",
                self.signature.len()
            )));
        }
        let signature_bytes: [u8; 64] = self.signature.clone().try_into().map_err(|_| {
            AttestationError::InvalidSignatureFormat("Invalid signature length".into())
        })?;
        let signature = Signature::from_bytes(&signature_bytes);

        // Compute the message that was signed
        let message = self.compute_signing_message();

        // Verify the signature
        verifying_key
            .verify(&message, &signature)
            .map_err(|e| AttestationError::SignatureVerificationFailed(e.to_string()))
    }

    /// Validate the attestation has required fields (basic validation without signature check)
    ///
    /// For full validation including signature verification, use `validate_with_registry()`.
    pub fn validate(&self) -> Result<(), AttestationError> {
        if self.triple_hash.is_empty() {
            return Err(AttestationError::EmptyTripleHash);
        }
        if self.attester.is_empty() {
            return Err(AttestationError::EmptyAttester);
        }
        if self.supporting_sources.len() > 10 {
            return Err(AttestationError::TooManySources(
                self.supporting_sources.len(),
            ));
        }
        if let Some(ref evidence) = self.evidence {
            if evidence.len() > 2000 {
                return Err(AttestationError::EvidenceTooLong(evidence.len()));
            }
        }
        if self.signature.is_empty() {
            return Err(AttestationError::EmptySignature);
        }
        Ok(())
    }

    /// Full validation including signature verification
    ///
    /// This should be called before accepting any attestation.
    pub fn validate_with_registry(
        &self,
        registry: &dyn PublicKeyRegistry,
    ) -> Result<(), AttestationError> {
        // Basic field validation
        self.validate()?;

        // Signature verification
        self.verify_signature(registry)?;

        Ok(())
    }

    /// Get the weighted contribution of this attestation
    ///
    /// Weight = attestation_type.weight() * reputation
    pub fn weighted_contribution(&self) -> f64 {
        self.attestation_type.weight() * self.attester_reputation
    }
}

/// Default rate limit duration (60 seconds)
pub const DEFAULT_RATE_LIMIT_SECS: u64 = 60;

/// Collection of attestations for a triple with rate limiting (FIND-008)
#[derive(Clone, Debug)]
pub struct AttestationSet {
    /// All attestations for this triple
    pub attestations: Vec<Attestation>,
    /// Last update time per attester for rate limiting (transient - not serialized)
    last_update: HashMap<String, Instant>,
    /// Rate limit duration - minimum time between attestations from same attester
    rate_limit: Duration,
}

impl Default for AttestationSet {
    fn default() -> Self {
        Self {
            attestations: Vec::new(),
            last_update: HashMap::new(),
            rate_limit: Duration::from_secs(DEFAULT_RATE_LIMIT_SECS),
        }
    }
}

// Custom serialization to skip the transient rate limiting state
impl Serialize for AttestationSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("AttestationSet", 1)?;
        state.serialize_field("attestations", &self.attestations)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for AttestationSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct AttestationSetHelper {
            attestations: Vec<Attestation>,
        }
        let helper = AttestationSetHelper::deserialize(deserializer)?;
        Ok(Self {
            attestations: helper.attestations,
            last_update: HashMap::new(),
            rate_limit: Duration::from_secs(DEFAULT_RATE_LIMIT_SECS),
        })
    }
}

impl AttestationSet {
    /// Create a new empty attestation set with default rate limiting
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new attestation set with custom rate limit
    pub fn with_rate_limit(rate_limit: Duration) -> Self {
        Self {
            attestations: Vec::new(),
            last_update: HashMap::new(),
            rate_limit,
        }
    }

    /// Set the rate limit duration
    pub fn set_rate_limit(&mut self, rate_limit: Duration) {
        self.rate_limit = rate_limit;
    }

    /// Get the current rate limit duration
    pub fn rate_limit(&self) -> Duration {
        self.rate_limit
    }

    /// Check if an attester is currently rate limited
    ///
    /// Returns Some(remaining_time) if rate limited, None if allowed.
    pub fn is_rate_limited(&self, attester_id: &str) -> Option<Duration> {
        if let Some(last) = self.last_update.get(attester_id) {
            let elapsed = last.elapsed();
            if elapsed < self.rate_limit {
                return Some(self.rate_limit - elapsed);
            }
        }
        None
    }

    /// Add an attestation after validating its signature with rate limiting (FIND-008)
    ///
    /// Returns an error if:
    /// - Validation fails
    /// - Signature verification fails
    /// - Attester is rate limited (must wait before submitting another attestation)
    pub fn add_verified(
        &mut self,
        attestation: Attestation,
        registry: &dyn PublicKeyRegistry,
    ) -> Result<(), AttestationError> {
        // Check rate limiting first (FIND-008)
        if let Some(retry_after) = self.is_rate_limited(&attestation.attester) {
            return Err(AttestationError::RateLimited { retry_after });
        }

        // Validate with signature verification
        attestation.validate_with_registry(registry)?;

        // Update rate limit tracking
        self.last_update
            .insert(attestation.attester.clone(), Instant::now());

        // Remove any existing attestation from the same attester
        self.attestations
            .retain(|a| a.attester != attestation.attester);
        self.attestations.push(attestation);
        Ok(())
    }

    /// Add an attestation with rate limiting but without signature verification
    ///
    /// WARNING: Only use this when you've already verified the signature elsewhere.
    /// Prefer `add_verified()` for normal operation.
    ///
    /// Returns an error if the attester is rate limited.
    pub fn add_rate_limited(&mut self, attestation: Attestation) -> Result<(), AttestationError> {
        // Check rate limiting (FIND-008)
        if let Some(retry_after) = self.is_rate_limited(&attestation.attester) {
            return Err(AttestationError::RateLimited { retry_after });
        }

        // Update rate limit tracking
        self.last_update
            .insert(attestation.attester.clone(), Instant::now());

        // Remove any existing attestation from the same attester
        self.attestations
            .retain(|a| a.attester != attestation.attester);
        self.attestations.push(attestation);
        Ok(())
    }

    /// Add an attestation without signature verification or rate limiting (for internal use only)
    ///
    /// WARNING: Only use this when you've already verified the signature elsewhere
    /// AND you're handling rate limiting at a higher level.
    /// Prefer `add_verified()` or `add_rate_limited()` for normal operation.
    pub fn add(&mut self, attestation: Attestation) {
        // Remove any existing attestation from the same attester
        self.attestations
            .retain(|a| a.attester != attestation.attester);
        self.attestations.push(attestation);
    }

    /// Clear rate limiting state for all attesters
    ///
    /// Useful for testing or after significant time has passed.
    pub fn clear_rate_limits(&mut self) {
        self.last_update.clear();
    }

    /// Clear rate limiting state for a specific attester
    pub fn clear_rate_limit_for(&mut self, attester_id: &str) {
        self.last_update.remove(attester_id);
    }

    /// Get count of each attestation type
    pub fn counts(&self) -> AttestationCounts {
        let mut counts = AttestationCounts::default();
        for att in &self.attestations {
            match att.attestation_type {
                AttestationType::Endorse => counts.endorsements += 1,
                AttestationType::Challenge => counts.challenges += 1,
                AttestationType::Acknowledge => counts.acknowledgments += 1,
                AttestationType::Partial { .. } => counts.partial += 1,
            }
        }
        counts
    }

    /// Get all endorsement reputations
    pub fn endorsement_reputations(&self) -> Vec<f64> {
        self.attestations
            .iter()
            .filter(|a| matches!(a.attestation_type, AttestationType::Endorse))
            .map(|a| a.attester_reputation)
            .collect()
    }

    /// Get total contradiction weight (sum of challenge reputations)
    pub fn contradiction_weight(&self) -> f64 {
        self.attestations
            .iter()
            .filter(|a| a.attestation_type.is_contradiction())
            .map(|a| a.attester_reputation)
            .sum()
    }

    /// Calculate net reputation-weighted score
    ///
    /// Positive = more endorsements, negative = more challenges
    pub fn net_weighted_score(&self) -> f64 {
        self.attestations
            .iter()
            .map(|a| a.weighted_contribution())
            .sum()
    }

    /// Get unique attester count
    pub fn unique_attester_count(&self) -> usize {
        let mut attesters: Vec<&String> = self.attestations.iter().map(|a| &a.attester).collect();
        attesters.sort();
        attesters.dedup();
        attesters.len()
    }

    /// Check if a specific attester has already attested
    pub fn has_attested(&self, attester: &str) -> bool {
        self.attestations.iter().any(|a| a.attester == attester)
    }

    /// Get the most recent attestation timestamp
    pub fn latest_timestamp(&self) -> u64 {
        self.attestations
            .iter()
            .map(|a| a.created_at)
            .max()
            .unwrap_or(0)
    }
}

/// Counts of attestation types
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AttestationCounts {
    /// Number of endorsements
    pub endorsements: u32,
    /// Number of challenges
    pub challenges: u32,
    /// Number of acknowledgments
    pub acknowledgments: u32,
    /// Number of partial agreements
    pub partial: u32,
}

impl AttestationCounts {
    /// Total number of attestations
    pub fn total(&self) -> u32 {
        self.endorsements + self.challenges + self.acknowledgments + self.partial
    }

    /// Ratio of endorsements to challenges
    pub fn endorsement_ratio(&self) -> f64 {
        if self.challenges == 0 {
            if self.endorsements > 0 {
                f64::INFINITY
            } else {
                1.0
            }
        } else {
            self.endorsements as f64 / self.challenges as f64
        }
    }
}

/// Helper for hex encoding (minimal implementation to avoid adding dependency)
mod hex {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        let bytes = bytes.as_ref();
        let mut result = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            result.push(HEX_CHARS[(b >> 4) as usize] as char);
            result.push(HEX_CHARS[(b & 0x0f) as usize] as char);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    /// Create a test keypair and sign an attestation
    fn create_signed_attestation(
        triple_hash: &str,
        attester: &str,
        attestation_type: AttestationType,
        timestamp: u64,
    ) -> (Attestation, SigningKey) {
        let signing_key = SigningKey::from_bytes(&[
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
            0x1d, 0x1e, 0x1f, 0x20,
        ]);

        // Create attestation without signature first to compute the message
        let mut attestation = Attestation {
            triple_hash: triple_hash.to_string(),
            attester: attester.to_string(),
            attestation_type,
            attester_reputation: 0.0,
            evidence: None,
            supporting_sources: Vec::new(),
            created_at: timestamp,
            signature: Vec::new(),
        };

        // Compute and sign the message
        let message = attestation.compute_signing_message();
        let signature = signing_key.sign(&message);
        attestation.signature = signature.to_bytes().to_vec();

        (attestation, signing_key)
    }

    fn create_test_registry(attester: &str, signing_key: &SigningKey) -> InMemoryKeyRegistry {
        let mut registry = InMemoryKeyRegistry::new();
        let public_key = signing_key.verifying_key();
        registry.register(attester, public_key.to_bytes());
        registry
    }

    #[test]
    fn test_attestation_type_weights() {
        assert!((AttestationType::Endorse.weight() - 1.0).abs() < 0.01);
        assert!((AttestationType::Challenge.weight() + 1.0).abs() < 0.01);
        assert!((AttestationType::Acknowledge.weight() - 0.1).abs() < 0.01);

        let partial_50 = AttestationType::Partial {
            agreement_level: 50,
        };
        assert!(partial_50.weight().abs() < 0.01); // 50% = neutral

        let partial_100 = AttestationType::Partial {
            agreement_level: 100,
        };
        assert!((partial_100.weight() - 1.0).abs() < 0.01);

        let partial_0 = AttestationType::Partial { agreement_level: 0 };
        assert!((partial_0.weight() + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_attestation_validation_with_signature() {
        let (attestation, signing_key) =
            create_signed_attestation("hash123", "attester1", AttestationType::Endorse, 1000);
        let registry = create_test_registry("attester1", &signing_key);

        // Basic validation should pass
        assert!(attestation.validate().is_ok());

        // Full validation with signature should pass
        assert!(attestation.validate_with_registry(&registry).is_ok());
    }

    #[test]
    fn test_attestation_validation_empty_hash() {
        let (mut attestation, _) =
            create_signed_attestation("", "attester1", AttestationType::Endorse, 1000);
        attestation.triple_hash = String::new();

        assert!(matches!(
            attestation.validate(),
            Err(AttestationError::EmptyTripleHash)
        ));
    }

    #[test]
    fn test_attestation_validation_empty_attester() {
        let (mut attestation, _) =
            create_signed_attestation("hash123", "", AttestationType::Endorse, 1000);
        attestation.attester = String::new();

        assert!(matches!(
            attestation.validate(),
            Err(AttestationError::EmptyAttester)
        ));
    }

    #[test]
    fn test_attestation_validation_empty_signature() {
        let (mut attestation, _) =
            create_signed_attestation("hash123", "attester1", AttestationType::Endorse, 1000);
        attestation.signature = Vec::new();

        assert!(matches!(
            attestation.validate(),
            Err(AttestationError::EmptySignature)
        ));
    }

    #[test]
    fn test_signature_verification_success() {
        let (attestation, signing_key) =
            create_signed_attestation("hash123", "alice", AttestationType::Endorse, 1700000000);
        let registry = create_test_registry("alice", &signing_key);

        assert!(attestation.verify_signature(&registry).is_ok());
    }

    #[test]
    fn test_signature_verification_wrong_key() {
        let (attestation, _) =
            create_signed_attestation("hash123", "alice", AttestationType::Endorse, 1700000000);

        // Create registry with different key
        let wrong_key = SigningKey::from_bytes(&[0x42; 32]);
        let registry = create_test_registry("alice", &wrong_key);

        assert!(matches!(
            attestation.verify_signature(&registry),
            Err(AttestationError::SignatureVerificationFailed(_))
        ));
    }

    #[test]
    fn test_signature_verification_key_not_found() {
        let (attestation, _) =
            create_signed_attestation("hash123", "alice", AttestationType::Endorse, 1700000000);
        let registry = InMemoryKeyRegistry::new(); // Empty registry

        assert!(matches!(
            attestation.verify_signature(&registry),
            Err(AttestationError::PublicKeyNotFound(_))
        ));
    }

    #[test]
    fn test_signature_verification_tampered_data() {
        let (mut attestation, signing_key) =
            create_signed_attestation("hash123", "alice", AttestationType::Endorse, 1700000000);
        let registry = create_test_registry("alice", &signing_key);

        // Tamper with the triple hash
        attestation.triple_hash = "tampered_hash".to_string();

        assert!(matches!(
            attestation.verify_signature(&registry),
            Err(AttestationError::SignatureVerificationFailed(_))
        ));
    }

    #[test]
    fn test_weighted_contribution() {
        let (mut attestation, _) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        attestation = attestation.with_reputation(0.9);
        assert!((attestation.weighted_contribution() - 0.9).abs() < 0.01);

        let (mut challenge, _) =
            create_signed_attestation("hash", "bob", AttestationType::Challenge, 1000);
        challenge = challenge.with_reputation(0.3);
        assert!((challenge.weighted_contribution() + 0.3).abs() < 0.01);
    }

    #[test]
    fn test_attestation_set_with_verified_add() {
        let mut set = AttestationSet::new();

        let (attestation1, signing_key1) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let registry1 = create_test_registry("alice", &signing_key1);

        assert!(set
            .add_verified(attestation1.with_reputation(0.8), &registry1)
            .is_ok());
        assert_eq!(set.attestations.len(), 1);
    }

    #[test]
    fn test_attestation_set_rejects_invalid_signature() {
        let mut set = AttestationSet::new();

        let (mut attestation, _) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        // Corrupt the signature
        attestation.signature[0] ^= 0xFF;

        let signing_key = SigningKey::from_bytes(&[0x01; 32]);
        let registry = create_test_registry("alice", &signing_key);

        // Should fail due to invalid signature (wrong key)
        assert!(set.add_verified(attestation, &registry).is_err());
        assert!(set.attestations.is_empty());
    }

    #[test]
    fn test_attestation_set_deduplication() {
        let mut set = AttestationSet::new();

        // Use same attester for both
        let (attestation1, signing_key) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);

        // Create second attestation (challenge) with same key
        let mut attestation2 = Attestation {
            triple_hash: "hash".to_string(),
            attester: "alice".to_string(),
            attestation_type: AttestationType::Challenge,
            attester_reputation: 0.9,
            evidence: None,
            supporting_sources: Vec::new(),
            created_at: 2000,
            signature: Vec::new(),
        };
        let message2 = attestation2.compute_signing_message();
        let signature2 = signing_key.sign(&message2);
        attestation2.signature = signature2.to_bytes().to_vec();

        let registry = create_test_registry("alice", &signing_key);

        assert!(set
            .add_verified(attestation1.with_reputation(0.8), &registry)
            .is_ok());

        // Clear rate limit for alice to test deduplication behavior (not rate limiting)
        set.clear_rate_limit_for("alice");

        assert!(set.add_verified(attestation2, &registry).is_ok());

        // Should only have one attestation from alice (the challenge replaces the endorse)
        assert_eq!(set.attestations.len(), 1);
        assert!(matches!(
            set.attestations[0].attestation_type,
            AttestationType::Challenge
        ));
    }

    #[test]
    fn test_attestation_set_counts() {
        let mut set = AttestationSet::new();

        let (att1, key1) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let (att2, key2) = create_signed_attestation("hash", "bob", AttestationType::Endorse, 1000);
        let (att3, key3) =
            create_signed_attestation("hash", "eve", AttestationType::Challenge, 1000);

        let mut registry = InMemoryKeyRegistry::new();
        registry.register("alice", key1.verifying_key().to_bytes());
        registry.register("bob", key2.verifying_key().to_bytes());
        registry.register("eve", key3.verifying_key().to_bytes());

        set.add_verified(att1, &registry).unwrap();
        set.add_verified(att2, &registry).unwrap();
        set.add_verified(att3, &registry).unwrap();

        let counts = set.counts();
        assert_eq!(counts.endorsements, 2);
        assert_eq!(counts.challenges, 1);
        assert_eq!(counts.total(), 3);
    }

    #[test]
    fn test_attestation_set_contradiction_weight() {
        let mut set = AttestationSet::new();

        let (att1, key1) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let (att2, key2) =
            create_signed_attestation("hash", "eve", AttestationType::Challenge, 1000);
        let (att3, key3) =
            create_signed_attestation("hash", "mallory", AttestationType::Challenge, 1000);

        let mut registry = InMemoryKeyRegistry::new();
        registry.register("alice", key1.verifying_key().to_bytes());
        registry.register("eve", key2.verifying_key().to_bytes());
        registry.register("mallory", key3.verifying_key().to_bytes());

        set.add_verified(att1.with_reputation(0.9), &registry)
            .unwrap();
        set.add_verified(att2.with_reputation(0.3), &registry)
            .unwrap();
        set.add_verified(att3.with_reputation(0.4), &registry)
            .unwrap();

        assert!((set.contradiction_weight() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_net_weighted_score() {
        let mut set = AttestationSet::new();

        let (att1, key1) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let (att2, key2) = create_signed_attestation("hash", "bob", AttestationType::Endorse, 1000);
        let (att3, key3) =
            create_signed_attestation("hash", "eve", AttestationType::Challenge, 1000);

        let mut registry = InMemoryKeyRegistry::new();
        registry.register("alice", key1.verifying_key().to_bytes());
        registry.register("bob", key2.verifying_key().to_bytes());
        registry.register("eve", key3.verifying_key().to_bytes());

        set.add_verified(att1.with_reputation(0.9), &registry)
            .unwrap();
        set.add_verified(att2.with_reputation(0.8), &registry)
            .unwrap();
        set.add_verified(att3.with_reputation(0.5), &registry)
            .unwrap();

        // Net = 0.9 + 0.8 - 0.5 = 1.2
        assert!((set.net_weighted_score() - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_endorsement_ratio() {
        let mut set = AttestationSet::new();

        let (att1, key1) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let (att2, key2) = create_signed_attestation("hash", "bob", AttestationType::Endorse, 1000);
        let (att3, key3) =
            create_signed_attestation("hash", "eve", AttestationType::Challenge, 1000);

        let mut registry = InMemoryKeyRegistry::new();
        registry.register("alice", key1.verifying_key().to_bytes());
        registry.register("bob", key2.verifying_key().to_bytes());
        registry.register("eve", key3.verifying_key().to_bytes());

        set.add_verified(att1, &registry).unwrap();
        set.add_verified(att2, &registry).unwrap();
        set.add_verified(att3, &registry).unwrap();

        let counts = set.counts();
        assert!((counts.endorsement_ratio() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_triple_commitment() {
        let commitment = Attestation::compute_triple_commitment(
            "subject:test",
            "predicate:has_color",
            "blue",
            "alice",
            1700000000,
        );

        // Should be a valid hex string of 64 characters (32 bytes)
        assert_eq!(commitment.len(), 64);
        assert!(commitment.chars().all(|c| c.is_ascii_hexdigit()));

        // Same inputs should produce same commitment
        let commitment2 = Attestation::compute_triple_commitment(
            "subject:test",
            "predicate:has_color",
            "blue",
            "alice",
            1700000000,
        );
        assert_eq!(commitment, commitment2);

        // Different inputs should produce different commitment
        let commitment3 = Attestation::compute_triple_commitment(
            "subject:test",
            "predicate:has_color",
            "green", // Different object
            "alice",
            1700000000,
        );
        assert_ne!(commitment, commitment3);
    }

    #[test]
    fn test_signing_message_deterministic() {
        let (attestation1, _) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let (attestation2, _) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);

        assert_eq!(
            attestation1.compute_signing_message(),
            attestation2.compute_signing_message()
        );
    }

    #[test]
    fn test_signing_message_different_for_different_inputs() {
        let (attestation1, _) =
            create_signed_attestation("hash1", "alice", AttestationType::Endorse, 1000);
        let (attestation2, _) =
            create_signed_attestation("hash2", "alice", AttestationType::Endorse, 1000);

        assert_ne!(
            attestation1.compute_signing_message(),
            attestation2.compute_signing_message()
        );
    }

    // =========================================================================
    // Rate Limiting Tests (FIND-008)
    // =========================================================================

    #[test]
    fn test_rate_limiting_blocks_rapid_attestations() {
        // Use a short rate limit for testing
        let mut set = AttestationSet::with_rate_limit(Duration::from_millis(100));

        let (att1, key1) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let registry = create_test_registry("alice", &key1);

        // First attestation should succeed
        assert!(set
            .add_verified(att1.with_reputation(0.8), &registry)
            .is_ok());

        // Second attestation from same attester immediately should fail
        let (att2, _) =
            create_signed_attestation("hash2", "alice", AttestationType::Challenge, 2000);
        let result = set.add_verified(att2, &registry);
        assert!(matches!(result, Err(AttestationError::RateLimited { .. })));
    }

    #[test]
    fn test_rate_limiting_allows_different_attesters() {
        let mut set = AttestationSet::with_rate_limit(Duration::from_secs(60));

        let (att1, key1) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let (att2, key2) = create_signed_attestation("hash", "bob", AttestationType::Endorse, 1000);

        let mut registry = InMemoryKeyRegistry::new();
        registry.register("alice", key1.verifying_key().to_bytes());
        registry.register("bob", key2.verifying_key().to_bytes());

        // Both should succeed as they are different attesters
        assert!(set.add_verified(att1, &registry).is_ok());
        assert!(set.add_verified(att2, &registry).is_ok());
        assert_eq!(set.attestations.len(), 2);
    }

    #[test]
    fn test_rate_limit_clear_allows_new_attestation() {
        let mut set = AttestationSet::with_rate_limit(Duration::from_secs(60));

        let (att1, key1) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let registry = create_test_registry("alice", &key1);

        // First attestation succeeds
        assert!(set
            .add_verified(att1.with_reputation(0.8), &registry)
            .is_ok());

        // Clear rate limit
        set.clear_rate_limit_for("alice");

        // Now second attestation should succeed
        let mut attestation2 = Attestation {
            triple_hash: "hash2".to_string(),
            attester: "alice".to_string(),
            attestation_type: AttestationType::Challenge,
            attester_reputation: 0.0,
            evidence: None,
            supporting_sources: Vec::new(),
            created_at: 2000,
            signature: Vec::new(),
        };
        let message = attestation2.compute_signing_message();
        let signature = key1.sign(&message);
        attestation2.signature = signature.to_bytes().to_vec();

        assert!(set.add_verified(attestation2, &registry).is_ok());
    }

    #[test]
    fn test_is_rate_limited_returns_remaining_time() {
        let mut set = AttestationSet::with_rate_limit(Duration::from_millis(200));

        // Before any attestations, no one is rate limited
        assert!(set.is_rate_limited("alice").is_none());

        let (att1, key1) =
            create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        let registry = create_test_registry("alice", &key1);

        set.add_verified(att1, &registry).unwrap();

        // Now alice should be rate limited
        let remaining = set.is_rate_limited("alice");
        assert!(remaining.is_some());
        assert!(remaining.unwrap() <= Duration::from_millis(200));
    }

    #[test]
    fn test_add_rate_limited_without_signature_verification() {
        let mut set = AttestationSet::with_rate_limit(Duration::from_millis(100));

        let (att1, _) = create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);

        // First add should succeed
        assert!(set.add_rate_limited(att1).is_ok());
        assert_eq!(set.attestations.len(), 1);

        // Second add should fail due to rate limiting
        let (att2, _) =
            create_signed_attestation("hash2", "alice", AttestationType::Challenge, 2000);
        let result = set.add_rate_limited(att2);
        assert!(matches!(result, Err(AttestationError::RateLimited { .. })));
    }

    #[test]
    fn test_rate_limit_serialization_preserves_attestations() {
        let mut set = AttestationSet::with_rate_limit(Duration::from_secs(30));
        let (att1, _) = create_signed_attestation("hash", "alice", AttestationType::Endorse, 1000);
        set.add(att1);

        // Serialize and deserialize
        let json = serde_json::to_string(&set).unwrap();
        let deserialized: AttestationSet = serde_json::from_str(&json).unwrap();

        // Attestations should be preserved
        assert_eq!(deserialized.attestations.len(), 1);
        // Rate limit state is transient and not serialized, so it should use default
        assert_eq!(
            deserialized.rate_limit,
            Duration::from_secs(DEFAULT_RATE_LIMIT_SECS)
        );
    }
}
