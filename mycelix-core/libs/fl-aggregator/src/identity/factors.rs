//! Identity Factors for Multi-Factor Decentralized Identity
//!
//! Implements the 9 identity factor types across 5 categories:
//! - Cryptographic: CryptoKeyFactor, HardwareKeyFactor
//! - Biometric: BiometricFactor
//! - Social: SocialRecoveryFactor, ReputationAttestationFactor
//! - External: GitcoinPassportFactor, VerifiableCredentialFactor
//! - Knowledge: RecoveryPhraseFactor, SecurityQuestionsFactor

use crate::identity::{FactorCategory, FactorStatus, IdentityError, IdentityResult};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
use hmac::Hmac;
use pbkdf2::pbkdf2_hmac;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha512};
use std::fmt::Debug;
use rand::RngCore;

/// Trait for identity factors
pub trait IdentityFactor: Send + Sync + Debug {
    /// Unique factor ID
    fn factor_id(&self) -> &str;

    /// Factor type name
    fn factor_type(&self) -> &str;

    /// Factor category
    fn category(&self) -> FactorCategory;

    /// Current status
    fn status(&self) -> FactorStatus;

    /// Set status
    fn set_status(&mut self, status: FactorStatus);

    /// Get contribution to assurance level (0.0-1.0)
    fn contribution(&self) -> f32;

    /// When the factor was created
    fn created_at(&self) -> DateTime<Utc>;

    /// When the factor was last verified
    fn last_verified(&self) -> Option<DateTime<Utc>>;

    /// Update last verified timestamp
    fn mark_verified(&mut self);

    /// Clone the factor (for boxed trait objects)
    fn clone_box(&self) -> Box<dyn IdentityFactor>;

    /// Serialize to JSON
    fn to_json(&self) -> serde_json::Value;
}

impl Clone for Box<dyn IdentityFactor> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Primary cryptographic key pair (Ed25519)
#[derive(Debug, Clone)]
pub struct CryptoKeyFactor {
    pub factor_id: String,
    pub public_key: VerifyingKey,
    signing_key: Option<SigningKey>,
    pub device_fingerprint: Option<String>,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}

impl CryptoKeyFactor {
    /// Generate a new key pair
    pub fn generate() -> IdentityResult<Self> {
        let mut csprng = rand::rngs::OsRng;
        let mut secret_bytes = [0u8; 32];
        csprng.fill_bytes(&mut secret_bytes);
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let public_key = signing_key.verifying_key();

        Ok(Self {
            factor_id: format!("crypto-{}", uuid_short()),
            public_key,
            signing_key: Some(signing_key),
            device_fingerprint: None,
            status: FactorStatus::Pending,
            created_at: Utc::now(),
            last_verified: None,
        })
    }

    /// Create from existing public key (no signing capability)
    pub fn from_public_key(public_key: VerifyingKey) -> Self {
        Self {
            factor_id: format!("crypto-{}", uuid_short()),
            public_key,
            signing_key: None,
            device_fingerprint: None,
            status: FactorStatus::Pending,
            created_at: Utc::now(),
            last_verified: None,
        }
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> IdentityResult<Signature> {
        let signing_key = self
            .signing_key
            .as_ref()
            .ok_or_else(|| IdentityError::CryptoError("No private key available".to_string()))?;

        use ed25519_dalek::Signer;
        Ok(signing_key.sign(message))
    }

    /// Verify a signature
    pub fn verify(&self, message: &[u8], signature: &Signature) -> bool {
        use ed25519_dalek::Verifier;
        self.public_key.verify(message, signature).is_ok()
    }

    /// Get DID from public key
    pub fn to_did(&self) -> String {
        let hash = Sha256::digest(self.public_key.as_bytes());
        format!("did:mycelix:{}", hex::encode(&hash[..16]))
    }
}

impl IdentityFactor for CryptoKeyFactor {
    fn factor_id(&self) -> &str {
        &self.factor_id
    }

    fn factor_type(&self) -> &str {
        "CryptoKey"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Cryptographic
    }

    fn status(&self) -> FactorStatus {
        self.status
    }

    fn set_status(&mut self, status: FactorStatus) {
        self.status = status;
    }

    fn contribution(&self) -> f32 {
        if self.status == FactorStatus::Active {
            0.5 // Primary key contributes 50%
        } else {
            0.0
        }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn last_verified(&self) -> Option<DateTime<Utc>> {
        self.last_verified
    }

    fn mark_verified(&mut self) {
        self.last_verified = Some(Utc::now());
        self.status = FactorStatus::Active;
    }

    fn clone_box(&self) -> Box<dyn IdentityFactor> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "factor_id": self.factor_id,
            "factor_type": "CryptoKey",
            "category": "Cryptographic",
            "status": format!("{:?}", self.status),
            "has_private_key": self.signing_key.is_some(),
            "created_at": self.created_at.to_rfc3339(),
            "last_verified": self.last_verified.map(|t| t.to_rfc3339()),
        })
    }
}

/// Gitcoin Passport factor for Sybil resistance
#[derive(Debug, Clone)]
pub struct GitcoinPassportFactor {
    pub factor_id: String,
    pub passport_address: String,
    pub score: f32,
    pub stamps: Vec<String>,
    pub expiry: Option<DateTime<Utc>>,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}

impl GitcoinPassportFactor {
    /// Create a new Gitcoin Passport factor
    pub fn new(passport_address: &str, score: f32) -> IdentityResult<Self> {
        if passport_address.is_empty() {
            return Err(IdentityError::InvalidFactor(
                "Passport address cannot be empty".to_string(),
            ));
        }

        let status = if score >= 20.0 {
            FactorStatus::Active
        } else {
            FactorStatus::Pending
        };

        Ok(Self {
            factor_id: format!("gitcoin-{}", uuid_short()),
            passport_address: passport_address.to_lowercase(),
            score,
            stamps: Vec::new(),
            expiry: None,
            status,
            created_at: Utc::now(),
            last_verified: if score >= 20.0 { Some(Utc::now()) } else { None },
        })
    }

    /// Add stamps from verification
    pub fn add_stamps(&mut self, stamps: Vec<String>) {
        self.stamps.extend(stamps);
    }

    /// Check if passport meets minimum threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.score >= threshold && self.status == FactorStatus::Active
    }
}

impl IdentityFactor for GitcoinPassportFactor {
    fn factor_id(&self) -> &str {
        &self.factor_id
    }

    fn factor_type(&self) -> &str {
        "GitcoinPassport"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::ExternalVerification
    }

    fn status(&self) -> FactorStatus {
        self.status
    }

    fn set_status(&mut self, status: FactorStatus) {
        self.status = status;
    }

    fn contribution(&self) -> f32 {
        if self.status != FactorStatus::Active {
            return 0.0;
        }

        // Score >= 50: 0.4, Score >= 20: 0.3
        if self.score >= 50.0 {
            0.4
        } else if self.score >= 20.0 {
            0.3
        } else {
            0.0
        }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn last_verified(&self) -> Option<DateTime<Utc>> {
        self.last_verified
    }

    fn mark_verified(&mut self) {
        self.last_verified = Some(Utc::now());
        if self.score >= 20.0 {
            self.status = FactorStatus::Active;
        }
    }

    fn clone_box(&self) -> Box<dyn IdentityFactor> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "factor_id": self.factor_id,
            "factor_type": "GitcoinPassport",
            "category": "ExternalVerification",
            "status": format!("{:?}", self.status),
            "passport_address": self.passport_address,
            "score": self.score,
            "stamps": self.stamps,
            "created_at": self.created_at.to_rfc3339(),
            "last_verified": self.last_verified.map(|t| t.to_rfc3339()),
        })
    }
}

/// Social recovery guardians using Shamir Secret Sharing
#[derive(Debug, Clone)]
pub struct SocialRecoveryFactor {
    pub factor_id: String,
    pub guardian_dids: Vec<String>,
    pub threshold: u8,
    pub total_shares: u8,
    pub recovery_key_hash: Option<String>,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}

impl SocialRecoveryFactor {
    /// Create a new social recovery factor
    pub fn new(threshold: u8) -> Self {
        Self {
            factor_id: format!("recovery-{}", uuid_short()),
            guardian_dids: Vec::new(),
            threshold,
            total_shares: 0,
            recovery_key_hash: None,
            status: FactorStatus::Pending,
            created_at: Utc::now(),
            last_verified: None,
        }
    }

    /// Add a guardian
    pub fn add_guardian(&mut self, guardian_did: &str) {
        if !self.guardian_dids.contains(&guardian_did.to_string()) {
            self.guardian_dids.push(guardian_did.to_string());
            self.total_shares = self.guardian_dids.len() as u8;
        }
    }

    /// Remove a guardian
    pub fn remove_guardian(&mut self, guardian_did: &str) {
        self.guardian_dids.retain(|d| d != guardian_did);
        self.total_shares = self.guardian_dids.len() as u8;
    }

    /// Check if recovery configuration is valid
    pub fn is_configured(&self) -> bool {
        self.guardian_dids.len() >= self.threshold as usize && self.threshold >= 2
    }
}

impl IdentityFactor for SocialRecoveryFactor {
    fn factor_id(&self) -> &str {
        &self.factor_id
    }

    fn factor_type(&self) -> &str {
        "SocialRecovery"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::SocialProof
    }

    fn status(&self) -> FactorStatus {
        self.status
    }

    fn set_status(&mut self, status: FactorStatus) {
        self.status = status;
    }

    fn contribution(&self) -> f32 {
        if self.status != FactorStatus::Active {
            return 0.0;
        }

        // Well-configured (5+ guardians, threshold >= 3): 0.3
        // Basic (3+ guardians, threshold >= 2): 0.2
        if self.guardian_dids.len() >= 5 && self.threshold >= 3 {
            0.3
        } else if self.guardian_dids.len() >= 3 && self.threshold >= 2 {
            0.2
        } else {
            0.1
        }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn last_verified(&self) -> Option<DateTime<Utc>> {
        self.last_verified
    }

    fn mark_verified(&mut self) {
        self.last_verified = Some(Utc::now());
        if self.is_configured() {
            self.status = FactorStatus::Active;
        }
    }

    fn clone_box(&self) -> Box<dyn IdentityFactor> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "factor_id": self.factor_id,
            "factor_type": "SocialRecovery",
            "category": "SocialProof",
            "status": format!("{:?}", self.status),
            "guardian_count": self.guardian_dids.len(),
            "threshold": self.threshold,
            "is_configured": self.is_configured(),
            "created_at": self.created_at.to_rfc3339(),
            "last_verified": self.last_verified.map(|t| t.to_rfc3339()),
        })
    }
}

/// Biometric identity factor (hash only, never raw data)
#[derive(Debug, Clone)]
pub struct BiometricFactor {
    pub factor_id: String,
    pub biometric_type: BiometricType,
    pub template_hash: String,
    pub liveness_verified: bool,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}

/// Biometric types supported
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BiometricType {
    Face,
    Fingerprint,
    Iris,
    Voice,
    Behavioral,
}

impl BiometricFactor {
    /// Create a new biometric factor from template hash
    pub fn new(biometric_type: BiometricType, template_hash: &str) -> Self {
        Self {
            factor_id: format!("bio-{}", uuid_short()),
            biometric_type,
            template_hash: template_hash.to_string(),
            liveness_verified: false,
            status: FactorStatus::Pending,
            created_at: Utc::now(),
            last_verified: None,
        }
    }

    /// Mark liveness as verified
    pub fn set_liveness_verified(&mut self, verified: bool) {
        self.liveness_verified = verified;
    }
}

impl IdentityFactor for BiometricFactor {
    fn factor_id(&self) -> &str {
        &self.factor_id
    }

    fn factor_type(&self) -> &str {
        "Biometric"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Biometric
    }

    fn status(&self) -> FactorStatus {
        self.status
    }

    fn set_status(&mut self, status: FactorStatus) {
        self.status = status;
    }

    fn contribution(&self) -> f32 {
        if self.status != FactorStatus::Active {
            return 0.0;
        }

        // With liveness: 0.3, Without: 0.2
        if self.liveness_verified {
            0.3
        } else {
            0.2
        }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn last_verified(&self) -> Option<DateTime<Utc>> {
        self.last_verified
    }

    fn mark_verified(&mut self) {
        self.last_verified = Some(Utc::now());
        self.status = FactorStatus::Active;
    }

    fn clone_box(&self) -> Box<dyn IdentityFactor> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "factor_id": self.factor_id,
            "factor_type": "Biometric",
            "category": "Biometric",
            "biometric_type": format!("{:?}", self.biometric_type),
            "status": format!("{:?}", self.status),
            "liveness_verified": self.liveness_verified,
            "has_template": !self.template_hash.is_empty(),
            "created_at": self.created_at.to_rfc3339(),
            "last_verified": self.last_verified.map(|t| t.to_rfc3339()),
        })
    }
}

/// Hardware security key (YubiKey, Ledger, etc.)
#[derive(Debug, Clone)]
pub struct HardwareKeyFactor {
    pub factor_id: String,
    pub device_type: String,
    pub device_id: Option<String>,
    pub public_key: Option<Vec<u8>>,
    pub counter: u32,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}

impl HardwareKeyFactor {
    /// Create a new hardware key factor
    pub fn new(device_type: &str) -> Self {
        Self {
            factor_id: format!("hw-{}", uuid_short()),
            device_type: device_type.to_string(),
            device_id: None,
            public_key: None,
            counter: 0,
            status: FactorStatus::Pending,
            created_at: Utc::now(),
            last_verified: None,
        }
    }

    /// Register device with attestation
    pub fn register_device(&mut self, device_id: &str, public_key: Vec<u8>) {
        self.device_id = Some(device_id.to_string());
        self.public_key = Some(public_key);
    }

    /// Update counter (prevents replay attacks)
    pub fn update_counter(&mut self, new_counter: u32) -> bool {
        if new_counter > self.counter {
            self.counter = new_counter;
            true
        } else {
            false
        }
    }
}

impl IdentityFactor for HardwareKeyFactor {
    fn factor_id(&self) -> &str {
        &self.factor_id
    }

    fn factor_type(&self) -> &str {
        "HardwareKey"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Cryptographic
    }

    fn status(&self) -> FactorStatus {
        self.status
    }

    fn set_status(&mut self, status: FactorStatus) {
        self.status = status;
    }

    fn contribution(&self) -> f32 {
        if self.status == FactorStatus::Active {
            0.3 // High contribution due to physical device
        } else {
            0.0
        }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn last_verified(&self) -> Option<DateTime<Utc>> {
        self.last_verified
    }

    fn mark_verified(&mut self) {
        self.last_verified = Some(Utc::now());
        if self.public_key.is_some() {
            self.status = FactorStatus::Active;
        }
    }

    fn clone_box(&self) -> Box<dyn IdentityFactor> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "factor_id": self.factor_id,
            "factor_type": "HardwareKey",
            "category": "Cryptographic",
            "device_type": self.device_type,
            "device_id": self.device_id,
            "status": format!("{:?}", self.status),
            "counter": self.counter,
            "has_public_key": self.public_key.is_some(),
            "created_at": self.created_at.to_rfc3339(),
            "last_verified": self.last_verified.map(|t| t.to_rfc3339()),
        })
    }
}

/// Recovery phrase factor (BIP39 mnemonic)
#[derive(Debug, Clone)]
pub struct RecoveryPhraseFactor {
    pub factor_id: String,
    pub phrase_hash: String,
    pub word_count: u8,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}

impl RecoveryPhraseFactor {
    /// Create from phrase hash (never store plaintext)
    pub fn new(phrase_hash: &str, word_count: u8) -> Self {
        Self {
            factor_id: format!("phrase-{}", uuid_short()),
            phrase_hash: phrase_hash.to_string(),
            word_count,
            status: FactorStatus::Pending,
            created_at: Utc::now(),
            last_verified: None,
        }
    }

    /// Hash a phrase for storage
    pub fn hash_phrase(phrase: &str) -> String {
        let hash = Sha256::digest(phrase.as_bytes());
        hex::encode(hash)
    }

    /// Verify phrase matches stored hash
    pub fn verify_phrase(&self, phrase: &str) -> bool {
        Self::hash_phrase(phrase) == self.phrase_hash
    }
}

impl IdentityFactor for RecoveryPhraseFactor {
    fn factor_id(&self) -> &str {
        &self.factor_id
    }

    fn factor_type(&self) -> &str {
        "RecoveryPhrase"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Knowledge
    }

    fn status(&self) -> FactorStatus {
        self.status
    }

    fn set_status(&mut self, status: FactorStatus) {
        self.status = status;
    }

    fn contribution(&self) -> f32 {
        if self.status == FactorStatus::Active {
            0.25 // Backup factor
        } else {
            0.0
        }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn last_verified(&self) -> Option<DateTime<Utc>> {
        self.last_verified
    }

    fn mark_verified(&mut self) {
        self.last_verified = Some(Utc::now());
        self.status = FactorStatus::Active;
    }

    fn clone_box(&self) -> Box<dyn IdentityFactor> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "factor_id": self.factor_id,
            "factor_type": "RecoveryPhrase",
            "category": "Knowledge",
            "status": format!("{:?}", self.status),
            "word_count": self.word_count,
            "created_at": self.created_at.to_rfc3339(),
            "last_verified": self.last_verified.map(|t| t.to_rfc3339()),
        })
    }
}

// Helper function for generating short UUIDs
fn uuid_short() -> String {
    use sha2::{Digest, Sha256};
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let hash = Sha256::digest(now.to_le_bytes());
    hex::encode(&hash[..4])
}

// ============================================================================
// NEW MFDI IDENTITY FACTORS
// ============================================================================

/// Attestation category for reputation attestations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttestationCategory {
    General,
    Professional,
    Community,
}

impl std::fmt::Display for AttestationCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttestationCategory::General => write!(f, "General"),
            AttestationCategory::Professional => write!(f, "Professional"),
            AttestationCategory::Community => write!(f, "Community"),
        }
    }
}

/// Single attestation from a peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attestation {
    pub attester_did: String,
    pub attestation_hash: String,
    pub trust_score: f32,
    pub category: AttestationCategory,
    pub expiry: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

impl Attestation {
    /// Check if attestation is still valid (not expired)
    pub fn is_valid(&self) -> bool {
        if let Some(expiry) = self.expiry {
            Utc::now() < expiry
        } else {
            true
        }
    }
}

/// Reputation attestation factor for peer-based identity verification
///
/// Allows peers to vouch for an identity, contributing to trust scoring.
/// Integrates with MATL for enhanced reputation weighting.
#[derive(Debug, Clone)]
pub struct ReputationAttestationFactor {
    pub factor_id: String,
    pub attestations: Vec<Attestation>,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}

impl ReputationAttestationFactor {
    /// Create a new reputation attestation factor
    pub fn new() -> Self {
        Self {
            factor_id: format!("rep-attest-{}", uuid_short()),
            attestations: Vec::new(),
            status: FactorStatus::Pending,
            created_at: Utc::now(),
            last_verified: None,
        }
    }

    /// Add an attestation from a peer
    pub fn add_attestation(
        &mut self,
        attester_did: &str,
        attestation_hash: &str,
        trust_score: f32,
        category: AttestationCategory,
        expiry: Option<DateTime<Utc>>,
    ) {
        // Prevent duplicate attestations from the same attester
        if self.attestations.iter().any(|a| a.attester_did == attester_did) {
            return;
        }

        self.attestations.push(Attestation {
            attester_did: attester_did.to_string(),
            attestation_hash: attestation_hash.to_string(),
            trust_score: trust_score.clamp(0.0, 1.0),
            category,
            expiry,
            created_at: Utc::now(),
        });

        // Auto-activate if we have at least one valid attestation
        if self.valid_attestation_count() >= 1 && self.status == FactorStatus::Pending {
            self.status = FactorStatus::Active;
            self.last_verified = Some(Utc::now());
        }
    }

    /// Remove an attestation by attester DID
    pub fn remove_attestation(&mut self, attester_did: &str) {
        self.attestations.retain(|a| a.attester_did != attester_did);
    }

    /// Get valid (non-expired) attestations
    pub fn valid_attestations(&self) -> Vec<&Attestation> {
        self.attestations.iter().filter(|a| a.is_valid()).collect()
    }

    /// Count valid attestations
    pub fn valid_attestation_count(&self) -> usize {
        self.valid_attestations().len()
    }

    /// Get unique categories from valid attestations
    pub fn unique_categories(&self) -> std::collections::HashSet<&AttestationCategory> {
        self.valid_attestations()
            .iter()
            .map(|a| &a.category)
            .collect()
    }

    /// Calculate average trust score from MATL-weighted attestations
    /// Uses reputation-squared weighting consistent with MATL scoring
    pub fn average_trust_score(&self) -> f32 {
        let valid = self.valid_attestations();
        if valid.is_empty() {
            return 0.0;
        }

        // Weight by trust_score^2 (MATL-style reputation weighting)
        let weighted_sum: f32 = valid.iter().map(|a| a.trust_score * a.trust_score).sum();
        let total_weight: f32 = valid.iter().map(|a| a.trust_score).sum();

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Check if attestations are diverse (from multiple categories)
    pub fn has_diverse_attestations(&self) -> bool {
        self.unique_categories().len() >= 2
    }
}

impl Default for ReputationAttestationFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl IdentityFactor for ReputationAttestationFactor {
    fn factor_id(&self) -> &str {
        &self.factor_id
    }

    fn factor_type(&self) -> &str {
        "ReputationAttestation"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::SocialProof
    }

    fn status(&self) -> FactorStatus {
        self.status
    }

    fn set_status(&mut self, status: FactorStatus) {
        self.status = status;
    }

    fn contribution(&self) -> f32 {
        if self.status != FactorStatus::Active {
            return 0.0;
        }

        let count = self.valid_attestation_count();
        let categories = self.unique_categories().len();

        // Base contribution based on attestation count
        // 1 attestation: 0.2
        // 3+ attestations: 0.25
        // 5+ attestations from diverse categories: 0.3
        if count >= 5 && categories >= 2 {
            0.3
        } else if count >= 3 {
            0.25
        } else if count >= 1 {
            0.2
        } else {
            0.0
        }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn last_verified(&self) -> Option<DateTime<Utc>> {
        self.last_verified
    }

    fn mark_verified(&mut self) {
        self.last_verified = Some(Utc::now());
        if self.valid_attestation_count() >= 1 {
            self.status = FactorStatus::Active;
        }
    }

    fn clone_box(&self) -> Box<dyn IdentityFactor> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "factor_id": self.factor_id,
            "factor_type": "ReputationAttestation",
            "category": "SocialProof",
            "status": format!("{:?}", self.status),
            "attestation_count": self.attestations.len(),
            "valid_attestation_count": self.valid_attestation_count(),
            "unique_categories": self.unique_categories().len(),
            "average_trust_score": self.average_trust_score(),
            "has_diverse_attestations": self.has_diverse_attestations(),
            "created_at": self.created_at.to_rfc3339(),
            "last_verified": self.last_verified.map(|t| t.to_rfc3339()),
        })
    }
}

/// Security questions factor for knowledge-based identity verification
///
/// Provides backup authentication using pre-configured question/answer pairs.
/// Includes anti-brute-force protection with lockout after failed attempts.
#[derive(Debug, Clone)]
pub struct SecurityQuestionsFactor {
    pub factor_id: String,
    /// SHA-256 hashes of question+answer combinations
    pub question_hashes: Vec<String>,
    pub question_count: u8,
    pub required_correct: u8,
    pub attempts: u32,
    pub locked_until: Option<DateTime<Utc>>,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}

impl SecurityQuestionsFactor {
    /// Create a new security questions factor
    pub fn new(required_correct: u8) -> Self {
        Self {
            factor_id: format!("secq-{}", uuid_short()),
            question_hashes: Vec::new(),
            question_count: 0,
            required_correct: required_correct.max(1),
            attempts: 0,
            locked_until: None,
            status: FactorStatus::Pending,
            created_at: Utc::now(),
            last_verified: None,
        }
    }

    /// Hash a question+answer pair using SHA-256
    pub fn hash_qa(question: &str, answer: &str) -> String {
        let normalized_question = question.trim().to_lowercase();
        let normalized_answer = answer.trim().to_lowercase();
        let combined = format!("{}:{}", normalized_question, normalized_answer);
        let hash = Sha256::digest(combined.as_bytes());
        hex::encode(hash)
    }

    /// Add a security question (stores only hash, never plaintext)
    pub fn add_question(&mut self, question: &str, answer: &str) {
        let hash = Self::hash_qa(question, answer);
        if !self.question_hashes.contains(&hash) {
            self.question_hashes.push(hash);
            self.question_count = self.question_hashes.len() as u8;
        }

        // Activate if we have enough questions
        if self.question_count >= self.required_correct && self.status == FactorStatus::Pending {
            self.status = FactorStatus::Active;
        }
    }

    /// Check if the factor is currently locked due to failed attempts
    pub fn is_locked(&self) -> bool {
        if let Some(locked_until) = self.locked_until {
            Utc::now() < locked_until
        } else {
            false
        }
    }

    /// Verify answers against stored hashes
    /// Returns true if >= required_correct answers match
    pub fn verify_answers(&mut self, answers: &[(String, String)]) -> bool {
        // Check if locked
        if self.is_locked() {
            return false;
        }

        let mut correct_count = 0;
        for (question, answer) in answers {
            let hash = Self::hash_qa(question, answer);
            if self.question_hashes.contains(&hash) {
                correct_count += 1;
            }
        }

        let success = correct_count >= self.required_correct;

        if success {
            self.attempts = 0;
            self.locked_until = None;
            self.last_verified = Some(Utc::now());
        } else {
            self.attempts += 1;
            // Lock for 10 minutes after 5 failed attempts
            if self.attempts >= 5 {
                self.locked_until = Some(Utc::now() + chrono::Duration::minutes(10));
            }
        }

        success
    }

    /// Reset failed attempts (admin function)
    pub fn reset_attempts(&mut self) {
        self.attempts = 0;
        self.locked_until = None;
    }

    /// Get remaining lockout time in seconds
    pub fn lockout_remaining_seconds(&self) -> Option<i64> {
        self.locked_until.map(|until| {
            let remaining = until - Utc::now();
            remaining.num_seconds().max(0)
        })
    }
}

impl IdentityFactor for SecurityQuestionsFactor {
    fn factor_id(&self) -> &str {
        &self.factor_id
    }

    fn factor_type(&self) -> &str {
        "SecurityQuestions"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Knowledge
    }

    fn status(&self) -> FactorStatus {
        self.status
    }

    fn set_status(&mut self, status: FactorStatus) {
        self.status = status;
    }

    fn contribution(&self) -> f32 {
        if self.status != FactorStatus::Active {
            return 0.0;
        }

        // 3 questions: 0.15
        // 5+ questions: 0.2
        if self.question_count >= 5 {
            0.2
        } else if self.question_count >= 3 {
            0.15
        } else {
            0.1
        }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn last_verified(&self) -> Option<DateTime<Utc>> {
        self.last_verified
    }

    fn mark_verified(&mut self) {
        self.last_verified = Some(Utc::now());
        if self.question_count >= self.required_correct {
            self.status = FactorStatus::Active;
        }
    }

    fn clone_box(&self) -> Box<dyn IdentityFactor> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "factor_id": self.factor_id,
            "factor_type": "SecurityQuestions",
            "category": "Knowledge",
            "status": format!("{:?}", self.status),
            "question_count": self.question_count,
            "required_correct": self.required_correct,
            "attempts": self.attempts,
            "is_locked": self.is_locked(),
            "lockout_remaining_seconds": self.lockout_remaining_seconds(),
            "created_at": self.created_at.to_rfc3339(),
            "last_verified": self.last_verified.map(|t| t.to_rfc3339()),
        })
    }
}

/// Credential types for verifiable credentials factor
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CredentialType {
    VerifiedHumanity,
    KYC,
    Professional,
    Academic,
    Government,
}

impl std::fmt::Display for CredentialType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CredentialType::VerifiedHumanity => write!(f, "VerifiedHumanity"),
            CredentialType::KYC => write!(f, "KYC"),
            CredentialType::Professional => write!(f, "Professional"),
            CredentialType::Academic => write!(f, "Academic"),
            CredentialType::Government => write!(f, "Government"),
        }
    }
}

/// Additional credential for multi-issuer verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdditionalCredential {
    pub issuer_did: String,
    pub credential_type: CredentialType,
    pub credential_hash: String,
    pub expiry: Option<DateTime<Utc>>,
    pub revoked: bool,
    pub created_at: DateTime<Utc>,
}

impl AdditionalCredential {
    /// Check if credential is valid (not expired and not revoked)
    pub fn is_valid(&self) -> bool {
        if self.revoked {
            return false;
        }
        if let Some(expiry) = self.expiry {
            Utc::now() < expiry
        } else {
            true
        }
    }
}

/// Verifiable Credential identity factor
///
/// Represents external credentials from trusted issuers that verify
/// aspects of an identity (humanity, KYC status, professional licenses, etc.)
#[derive(Debug, Clone)]
pub struct VerifiableCredentialFactor {
    pub factor_id: String,
    pub credential_type: CredentialType,
    pub issuer_did: String,
    pub subject_did: String,
    pub credential_hash: String,
    pub expiry: Option<DateTime<Utc>>,
    pub revoked: bool,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
    /// Additional credentials from different issuers (for bonus contribution)
    additional_credentials: Vec<AdditionalCredential>,
}

impl VerifiableCredentialFactor {
    /// Create a new verifiable credential factor
    pub fn new(
        credential_type: CredentialType,
        issuer_did: &str,
        subject_did: &str,
        credential_hash: &str,
        expiry: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            factor_id: format!("vc-{}", uuid_short()),
            credential_type,
            issuer_did: issuer_did.to_string(),
            subject_did: subject_did.to_string(),
            credential_hash: credential_hash.to_string(),
            expiry,
            revoked: false,
            status: FactorStatus::Pending,
            created_at: Utc::now(),
            last_verified: None,
            additional_credentials: Vec::new(),
        }
    }

    /// Check if the primary credential is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expiry) = self.expiry {
            Utc::now() >= expiry
        } else {
            false
        }
    }

    /// Check if the primary credential is valid (not revoked and not expired)
    pub fn verify_not_revoked(&self) -> bool {
        !self.revoked && !self.is_expired()
    }

    /// Revoke the primary credential
    pub fn revoke(&mut self) {
        self.revoked = true;
        self.status = FactorStatus::Revoked;
    }

    /// Add an additional credential from a different issuer
    pub fn add_credential(
        &mut self,
        issuer_did: &str,
        credential_type: CredentialType,
        credential_hash: &str,
        expiry: Option<DateTime<Utc>>,
    ) {
        // Don't add duplicate credentials from same issuer
        if self.issuer_did == issuer_did
            || self.additional_credentials.iter().any(|c| c.issuer_did == issuer_did)
        {
            return;
        }

        self.additional_credentials.push(AdditionalCredential {
            issuer_did: issuer_did.to_string(),
            credential_type,
            credential_hash: credential_hash.to_string(),
            expiry,
            revoked: false,
            created_at: Utc::now(),
        });
    }

    /// Get count of unique valid issuers (including primary)
    pub fn unique_issuer_count(&self) -> usize {
        let mut count = if self.verify_not_revoked() { 1 } else { 0 };
        count += self
            .additional_credentials
            .iter()
            .filter(|c| c.is_valid())
            .count();
        count
    }

    /// Check if credential is high-trust type (KYC or Government)
    pub fn is_high_trust_type(&self) -> bool {
        matches!(
            self.credential_type,
            CredentialType::KYC | CredentialType::Government
        )
    }

    /// Get all valid additional credentials
    pub fn valid_credentials(&self) -> Vec<&AdditionalCredential> {
        self.additional_credentials
            .iter()
            .filter(|c| c.is_valid())
            .collect()
    }
}

impl IdentityFactor for VerifiableCredentialFactor {
    fn factor_id(&self) -> &str {
        &self.factor_id
    }

    fn factor_type(&self) -> &str {
        "VerifiableCredential"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::ExternalVerification
    }

    fn status(&self) -> FactorStatus {
        self.status
    }

    fn set_status(&mut self, status: FactorStatus) {
        self.status = status;
    }

    fn contribution(&self) -> f32 {
        if self.status != FactorStatus::Active {
            return 0.0;
        }

        if !self.verify_not_revoked() {
            return 0.0;
        }

        let unique_issuers = self.unique_issuer_count();

        // Multiple VCs from different issuers: 0.35
        // KYC/Government: 0.3
        // Basic VC: 0.2
        if unique_issuers >= 2 {
            0.35
        } else if self.is_high_trust_type() {
            0.3
        } else {
            0.2
        }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn last_verified(&self) -> Option<DateTime<Utc>> {
        self.last_verified
    }

    fn mark_verified(&mut self) {
        self.last_verified = Some(Utc::now());
        if self.verify_not_revoked() {
            self.status = FactorStatus::Active;
        }
    }

    fn clone_box(&self) -> Box<dyn IdentityFactor> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "factor_id": self.factor_id,
            "factor_type": "VerifiableCredential",
            "category": "ExternalVerification",
            "status": format!("{:?}", self.status),
            "credential_type": self.credential_type.to_string(),
            "issuer_did": self.issuer_did,
            "subject_did": self.subject_did,
            "is_expired": self.is_expired(),
            "is_revoked": self.revoked,
            "is_valid": self.verify_not_revoked(),
            "unique_issuer_count": self.unique_issuer_count(),
            "is_high_trust_type": self.is_high_trust_type(),
            "additional_credentials_count": self.additional_credentials.len(),
            "created_at": self.created_at.to_rfc3339(),
            "last_verified": self.last_verified.map(|t| t.to_rfc3339()),
            "expiry": self.expiry.map(|t| t.to_rfc3339()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_key_factor() {
        let factor = CryptoKeyFactor::generate().unwrap();
        assert_eq!(factor.factor_type(), "CryptoKey");
        assert_eq!(factor.category(), FactorCategory::Cryptographic);
        assert_eq!(factor.status(), FactorStatus::Pending);
        assert_eq!(factor.contribution(), 0.0);

        let mut factor = factor;
        factor.mark_verified();
        assert_eq!(factor.status(), FactorStatus::Active);
        assert_eq!(factor.contribution(), 0.5);
    }

    #[test]
    fn test_crypto_key_signing() {
        let factor = CryptoKeyFactor::generate().unwrap();
        let message = b"test message";
        let signature = factor.sign(message).unwrap();
        assert!(factor.verify(message, &signature));
        assert!(!factor.verify(b"wrong message", &signature));
    }

    #[test]
    fn test_gitcoin_passport_factor() {
        let factor = GitcoinPassportFactor::new("0x1234", 25.0).unwrap();
        assert_eq!(factor.factor_type(), "GitcoinPassport");
        assert_eq!(factor.status(), FactorStatus::Active);
        assert_eq!(factor.contribution(), 0.3);

        let high_score = GitcoinPassportFactor::new("0x5678", 55.0).unwrap();
        assert_eq!(high_score.contribution(), 0.4);

        let low_score = GitcoinPassportFactor::new("0x9abc", 15.0).unwrap();
        assert_eq!(low_score.status(), FactorStatus::Pending);
        assert_eq!(low_score.contribution(), 0.0);
    }

    #[test]
    fn test_social_recovery_factor() {
        let mut factor = SocialRecoveryFactor::new(3);
        assert!(!factor.is_configured());

        factor.add_guardian("did:mycelix:g1");
        factor.add_guardian("did:mycelix:g2");
        factor.add_guardian("did:mycelix:g3");
        assert!(factor.is_configured());

        factor.add_guardian("did:mycelix:g4");
        factor.add_guardian("did:mycelix:g5");
        factor.mark_verified();

        assert_eq!(factor.contribution(), 0.3);
    }

    #[test]
    fn test_biometric_factor() {
        let mut factor = BiometricFactor::new(BiometricType::Face, "hash123");
        factor.mark_verified();
        assert_eq!(factor.contribution(), 0.2);

        factor.set_liveness_verified(true);
        assert_eq!(factor.contribution(), 0.3);
    }

    #[test]
    fn test_recovery_phrase_hashing() {
        let phrase = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let hash = RecoveryPhraseFactor::hash_phrase(phrase);
        let factor = RecoveryPhraseFactor::new(&hash, 12);

        assert!(factor.verify_phrase(phrase));
        assert!(!factor.verify_phrase("wrong phrase"));
    }

    // ========================================================================
    // Tests for ReputationAttestationFactor
    // ========================================================================

    #[test]
    fn test_reputation_attestation_factor_creation() {
        let factor = ReputationAttestationFactor::new();
        assert_eq!(factor.factor_type(), "ReputationAttestation");
        assert_eq!(factor.category(), FactorCategory::SocialProof);
        assert_eq!(factor.status(), FactorStatus::Pending);
        assert_eq!(factor.contribution(), 0.0);
        assert_eq!(factor.valid_attestation_count(), 0);
    }

    #[test]
    fn test_reputation_attestation_add_attestation() {
        let mut factor = ReputationAttestationFactor::new();

        factor.add_attestation(
            "did:mycelix:attester1",
            "hash123",
            0.8,
            AttestationCategory::General,
            None,
        );

        assert_eq!(factor.attestations.len(), 1);
        assert_eq!(factor.valid_attestation_count(), 1);
        assert_eq!(factor.status(), FactorStatus::Active);
        assert_eq!(factor.contribution(), 0.2);
    }

    #[test]
    fn test_reputation_attestation_duplicate_prevention() {
        let mut factor = ReputationAttestationFactor::new();

        factor.add_attestation(
            "did:mycelix:attester1",
            "hash123",
            0.8,
            AttestationCategory::General,
            None,
        );
        factor.add_attestation(
            "did:mycelix:attester1",
            "hash456",
            0.9,
            AttestationCategory::Professional,
            None,
        );

        // Should only have one attestation (duplicate attester rejected)
        assert_eq!(factor.attestations.len(), 1);
    }

    #[test]
    fn test_reputation_attestation_contribution_scaling() {
        let mut factor = ReputationAttestationFactor::new();

        // 1 attestation: 0.2
        factor.add_attestation("did:mycelix:a1", "h1", 0.8, AttestationCategory::General, None);
        assert_eq!(factor.contribution(), 0.2);

        // 3 attestations: 0.25
        factor.add_attestation("did:mycelix:a2", "h2", 0.7, AttestationCategory::General, None);
        factor.add_attestation("did:mycelix:a3", "h3", 0.9, AttestationCategory::General, None);
        assert_eq!(factor.contribution(), 0.25);

        // 5+ attestations with diverse categories: 0.3
        factor.add_attestation("did:mycelix:a4", "h4", 0.6, AttestationCategory::Professional, None);
        factor.add_attestation("did:mycelix:a5", "h5", 0.8, AttestationCategory::Community, None);
        assert_eq!(factor.contribution(), 0.3);
        assert!(factor.has_diverse_attestations());
    }

    #[test]
    fn test_reputation_attestation_expiry() {
        let mut factor = ReputationAttestationFactor::new();

        // Add expired attestation
        let expired = Utc::now() - chrono::Duration::days(1);
        factor.add_attestation(
            "did:mycelix:expired",
            "hash",
            0.8,
            AttestationCategory::General,
            Some(expired),
        );

        // Attestation exists but is not valid
        assert_eq!(factor.attestations.len(), 1);
        assert_eq!(factor.valid_attestation_count(), 0);
    }

    #[test]
    fn test_reputation_attestation_matl_weighted_score() {
        let mut factor = ReputationAttestationFactor::new();

        factor.add_attestation("did:mycelix:a1", "h1", 0.9, AttestationCategory::General, None);
        factor.add_attestation("did:mycelix:a2", "h2", 0.3, AttestationCategory::General, None);

        // MATL-weighted: higher trust scores contribute more
        let avg = factor.average_trust_score();
        // (0.9^2 + 0.3^2) / (0.9 + 0.3) = (0.81 + 0.09) / 1.2 = 0.75
        assert!((avg - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_reputation_attestation_remove() {
        let mut factor = ReputationAttestationFactor::new();

        factor.add_attestation("did:mycelix:a1", "h1", 0.8, AttestationCategory::General, None);
        factor.add_attestation("did:mycelix:a2", "h2", 0.7, AttestationCategory::General, None);

        factor.remove_attestation("did:mycelix:a1");
        assert_eq!(factor.attestations.len(), 1);
        assert_eq!(factor.attestations[0].attester_did, "did:mycelix:a2");
    }

    // ========================================================================
    // Tests for SecurityQuestionsFactor
    // ========================================================================

    #[test]
    fn test_security_questions_creation() {
        let factor = SecurityQuestionsFactor::new(2);
        assert_eq!(factor.factor_type(), "SecurityQuestions");
        assert_eq!(factor.category(), FactorCategory::Knowledge);
        assert_eq!(factor.status(), FactorStatus::Pending);
        assert_eq!(factor.required_correct, 2);
        assert_eq!(factor.question_count, 0);
    }

    #[test]
    fn test_security_questions_hash_qa() {
        let hash1 = SecurityQuestionsFactor::hash_qa("What is your pet's name?", "Fluffy");
        let hash2 = SecurityQuestionsFactor::hash_qa("what is your pet's name?", "fluffy");
        let hash3 = SecurityQuestionsFactor::hash_qa("What is your pet's name?", "Max");

        // Same Q&A normalized should match
        assert_eq!(hash1, hash2);
        // Different answers should not match
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_security_questions_add_and_activate() {
        let mut factor = SecurityQuestionsFactor::new(2);

        factor.add_question("Pet name?", "Fluffy");
        assert_eq!(factor.question_count, 1);
        assert_eq!(factor.status(), FactorStatus::Pending);

        factor.add_question("City born?", "Dallas");
        assert_eq!(factor.question_count, 2);
        assert_eq!(factor.status(), FactorStatus::Active);
    }

    #[test]
    fn test_security_questions_verify_answers() {
        let mut factor = SecurityQuestionsFactor::new(2);

        factor.add_question("Pet name?", "Fluffy");
        factor.add_question("City born?", "Dallas");
        factor.add_question("Favorite color?", "Blue");

        // Correct answers
        let answers = vec![
            ("Pet name?".to_string(), "Fluffy".to_string()),
            ("City born?".to_string(), "Dallas".to_string()),
        ];
        assert!(factor.verify_answers(&answers));

        // Wrong answers
        let wrong_answers = vec![
            ("Pet name?".to_string(), "Max".to_string()),
            ("City born?".to_string(), "Houston".to_string()),
        ];
        assert!(!factor.verify_answers(&wrong_answers));
    }

    #[test]
    fn test_security_questions_partial_correct() {
        let mut factor = SecurityQuestionsFactor::new(2);

        factor.add_question("Pet name?", "Fluffy");
        factor.add_question("City born?", "Dallas");
        factor.add_question("Favorite color?", "Blue");

        // Only 1 correct (need 2)
        let partial = vec![
            ("Pet name?".to_string(), "Fluffy".to_string()),
            ("City born?".to_string(), "Wrong".to_string()),
        ];
        assert!(!factor.verify_answers(&partial));
    }

    #[test]
    fn test_security_questions_lockout() {
        let mut factor = SecurityQuestionsFactor::new(2);

        factor.add_question("Pet name?", "Fluffy");
        factor.add_question("City born?", "Dallas");

        let wrong_answers = vec![
            ("Pet name?".to_string(), "Wrong".to_string()),
            ("City born?".to_string(), "Wrong".to_string()),
        ];

        // 5 failed attempts should trigger lockout
        for _ in 0..5 {
            factor.verify_answers(&wrong_answers);
        }

        assert!(factor.is_locked());
        assert_eq!(factor.attempts, 5);

        // Even correct answers should fail while locked
        let correct = vec![
            ("Pet name?".to_string(), "Fluffy".to_string()),
            ("City born?".to_string(), "Dallas".to_string()),
        ];
        assert!(!factor.verify_answers(&correct));
    }

    #[test]
    fn test_security_questions_reset_attempts() {
        let mut factor = SecurityQuestionsFactor::new(2);

        factor.add_question("Pet name?", "Fluffy");
        factor.add_question("City born?", "Dallas");

        // Simulate failed attempts
        factor.attempts = 5;
        factor.locked_until = Some(Utc::now() + chrono::Duration::minutes(10));

        assert!(factor.is_locked());

        factor.reset_attempts();
        assert!(!factor.is_locked());
        assert_eq!(factor.attempts, 0);
    }

    #[test]
    fn test_security_questions_contribution_scaling() {
        let mut factor = SecurityQuestionsFactor::new(2);

        // Less than 3 questions: 0.1
        factor.add_question("Q1?", "A1");
        factor.add_question("Q2?", "A2");
        factor.mark_verified();
        assert_eq!(factor.contribution(), 0.1);

        // 3 questions: 0.15
        factor.add_question("Q3?", "A3");
        assert_eq!(factor.contribution(), 0.15);

        // 5+ questions: 0.2
        factor.add_question("Q4?", "A4");
        factor.add_question("Q5?", "A5");
        assert_eq!(factor.contribution(), 0.2);
    }

    // ========================================================================
    // Tests for VerifiableCredentialFactor
    // ========================================================================

    #[test]
    fn test_verifiable_credential_creation() {
        let factor = VerifiableCredentialFactor::new(
            CredentialType::VerifiedHumanity,
            "did:mycelix:issuer",
            "did:mycelix:subject",
            "credential_hash_123",
            None,
        );

        assert_eq!(factor.factor_type(), "VerifiableCredential");
        assert_eq!(factor.category(), FactorCategory::ExternalVerification);
        assert_eq!(factor.status(), FactorStatus::Pending);
        assert!(!factor.revoked);
        assert!(!factor.is_expired());
        assert!(factor.verify_not_revoked());
    }

    #[test]
    fn test_verifiable_credential_basic_contribution() {
        let mut factor = VerifiableCredentialFactor::new(
            CredentialType::VerifiedHumanity,
            "did:mycelix:issuer",
            "did:mycelix:subject",
            "hash",
            None,
        );

        factor.mark_verified();
        // Basic VC: 0.2
        assert_eq!(factor.contribution(), 0.2);
    }

    #[test]
    fn test_verifiable_credential_kyc_contribution() {
        let mut factor = VerifiableCredentialFactor::new(
            CredentialType::KYC,
            "did:mycelix:issuer",
            "did:mycelix:subject",
            "hash",
            None,
        );

        factor.mark_verified();
        assert!(factor.is_high_trust_type());
        // KYC: 0.3
        assert_eq!(factor.contribution(), 0.3);
    }

    #[test]
    fn test_verifiable_credential_government_contribution() {
        let mut factor = VerifiableCredentialFactor::new(
            CredentialType::Government,
            "did:mycelix:gov_issuer",
            "did:mycelix:subject",
            "hash",
            None,
        );

        factor.mark_verified();
        assert!(factor.is_high_trust_type());
        // Government: 0.3
        assert_eq!(factor.contribution(), 0.3);
    }

    #[test]
    fn test_verifiable_credential_multi_issuer() {
        let mut factor = VerifiableCredentialFactor::new(
            CredentialType::VerifiedHumanity,
            "did:mycelix:issuer1",
            "did:mycelix:subject",
            "hash1",
            None,
        );
        factor.mark_verified();

        // Add second credential from different issuer
        factor.add_credential(
            "did:mycelix:issuer2",
            CredentialType::Professional,
            "hash2",
            None,
        );

        assert_eq!(factor.unique_issuer_count(), 2);
        // Multiple issuers: 0.35
        assert_eq!(factor.contribution(), 0.35);
    }

    #[test]
    fn test_verifiable_credential_duplicate_issuer_prevention() {
        let mut factor = VerifiableCredentialFactor::new(
            CredentialType::VerifiedHumanity,
            "did:mycelix:issuer1",
            "did:mycelix:subject",
            "hash1",
            None,
        );

        // Try to add same issuer
        factor.add_credential(
            "did:mycelix:issuer1",
            CredentialType::Professional,
            "hash2",
            None,
        );

        assert_eq!(factor.unique_issuer_count(), 1);
        assert!(factor.additional_credentials.is_empty());
    }

    #[test]
    fn test_verifiable_credential_expiry() {
        let expired = Utc::now() - chrono::Duration::days(1);
        let factor = VerifiableCredentialFactor::new(
            CredentialType::VerifiedHumanity,
            "did:mycelix:issuer",
            "did:mycelix:subject",
            "hash",
            Some(expired),
        );

        assert!(factor.is_expired());
        assert!(!factor.verify_not_revoked());
    }

    #[test]
    fn test_verifiable_credential_revocation() {
        let mut factor = VerifiableCredentialFactor::new(
            CredentialType::VerifiedHumanity,
            "did:mycelix:issuer",
            "did:mycelix:subject",
            "hash",
            None,
        );

        factor.mark_verified();
        assert_eq!(factor.contribution(), 0.2);

        factor.revoke();
        assert!(factor.revoked);
        assert!(!factor.verify_not_revoked());
        assert_eq!(factor.status(), FactorStatus::Revoked);
        assert_eq!(factor.contribution(), 0.0);
    }

    #[test]
    fn test_verifiable_credential_additional_credential_validity() {
        let mut factor = VerifiableCredentialFactor::new(
            CredentialType::VerifiedHumanity,
            "did:mycelix:issuer1",
            "did:mycelix:subject",
            "hash1",
            None,
        );
        factor.mark_verified();

        // Add valid credential
        factor.add_credential(
            "did:mycelix:issuer2",
            CredentialType::Professional,
            "hash2",
            Some(Utc::now() + chrono::Duration::days(365)),
        );

        // Add expired credential
        factor.add_credential(
            "did:mycelix:issuer3",
            CredentialType::Academic,
            "hash3",
            Some(Utc::now() - chrono::Duration::days(1)),
        );

        // Only primary + 1 valid additional = 2 unique issuers
        assert_eq!(factor.unique_issuer_count(), 2);
        assert_eq!(factor.valid_credentials().len(), 1);
    }

    #[test]
    fn test_verifiable_credential_to_json() {
        let factor = VerifiableCredentialFactor::new(
            CredentialType::KYC,
            "did:mycelix:issuer",
            "did:mycelix:subject",
            "hash123",
            None,
        );

        let json = factor.to_json();
        assert_eq!(json["factor_type"], "VerifiableCredential");
        assert_eq!(json["credential_type"], "KYC");
        assert_eq!(json["is_high_trust_type"], true);
        assert_eq!(json["is_valid"], true);
    }
}
