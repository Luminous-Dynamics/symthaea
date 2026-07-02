// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Type definitions for Mycelix Identity Client

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Assurance levels following eIDAS-inspired tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum AssuranceLevel {
    /// Unverified (self-attested only)
    #[default]
    E0,
    /// Email verified
    E1,
    /// Phone verified + social recovery configured
    E2,
    /// Government ID verified (KYC)
    E3,
    /// Biometric + multi-factor (high-value transactions)
    E4,
}

impl AssuranceLevel {
    /// Get numeric value for comparisons
    pub fn value(&self) -> u8 {
        match self {
            AssuranceLevel::E0 => 0,
            AssuranceLevel::E1 => 1,
            AssuranceLevel::E2 => 2,
            AssuranceLevel::E3 => 3,
            AssuranceLevel::E4 => 4,
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "E0" => Some(AssuranceLevel::E0),
            "E1" => Some(AssuranceLevel::E1),
            "E2" => Some(AssuranceLevel::E2),
            "E3" => Some(AssuranceLevel::E3),
            "E4" => Some(AssuranceLevel::E4),
            _ => None,
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            AssuranceLevel::E0 => "Unverified",
            AssuranceLevel::E1 => "Email Verified",
            AssuranceLevel::E2 => "Phone + Recovery",
            AssuranceLevel::E3 => "Government ID",
            AssuranceLevel::E4 => "Biometric + MFA",
        }
    }
}

impl std::fmt::Display for AssuranceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// DID Document structure (W3C DID Core compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidDocument {
    /// DID identifier (did:mycelix:...)
    pub id: String,
    /// Controller agent pub key
    pub controller: String,
    /// Verification methods (keys)
    pub verification_method: Vec<VerificationMethod>,
    /// Authentication methods
    pub authentication: Vec<String>,
    /// Service endpoints
    pub service: Vec<ServiceEndpoint>,
    /// Creation timestamp
    pub created: i64,
    /// Last update timestamp
    pub updated: i64,
    /// Document version
    pub version: u32,
}

impl Default for DidDocument {
    fn default() -> Self {
        Self {
            id: String::new(),
            controller: String::new(),
            verification_method: Vec::new(),
            authentication: Vec::new(),
            service: Vec::new(),
            created: 0,
            updated: 0,
            version: 1,
        }
    }
}

/// Verification method (key) in a DID document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMethod {
    /// Method ID
    pub id: String,
    /// Key type
    #[serde(rename = "type")]
    pub type_: String,
    /// Controller DID
    pub controller: String,
    /// Public key in multibase format
    pub public_key_multibase: String,
}

/// Service endpoint in a DID document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    /// Service ID
    pub id: String,
    /// Service type
    #[serde(rename = "type")]
    pub type_: String,
    /// Service endpoint URL
    pub service_endpoint: String,
}

/// Result of DID resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidResolutionResult {
    /// Whether resolution was successful
    pub success: bool,
    /// The resolved DID document (if found)
    pub did_document: Option<DidDocument>,
    /// Error message if resolution failed
    pub error: Option<String>,
    /// Whether this is from cache
    pub cached: bool,
    /// Resolution metadata
    pub metadata: Option<ResolutionMetadata>,
}

impl Default for DidResolutionResult {
    fn default() -> Self {
        Self {
            success: false,
            did_document: None,
            error: None,
            cached: false,
            metadata: None,
        }
    }
}

/// Resolution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionMetadata {
    /// Time taken to resolve (ms)
    pub resolve_time_ms: u64,
    /// Source of resolution
    pub source: ResolutionSource,
}

/// Source of DID resolution
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ResolutionSource {
    Local,
    Dht,
    Fallback,
}

/// Verifiable Credential structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiableCredential {
    /// Credential context
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    /// Credential ID
    pub id: String,
    /// Credential types
    #[serde(rename = "type")]
    pub type_: Vec<String>,
    /// Issuer DID
    pub issuer: String,
    /// Issuance date
    pub issuance_date: String,
    /// Expiration date (optional)
    pub expiration_date: Option<String>,
    /// Credential subject (claims)
    pub credential_subject: HashMap<String, serde_json::Value>,
    /// Credential status
    pub credential_status: Option<CredentialStatus>,
    /// Proof/signature
    pub proof: Option<CredentialProof>,
}

/// Credential status for revocation checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialStatus {
    /// Status ID
    pub id: String,
    /// Status type
    #[serde(rename = "type")]
    pub type_: String,
    /// Revocation list index
    pub revocation_list_index: Option<String>,
    /// Revocation list credential
    pub revocation_list_credential: Option<String>,
}

/// Credential proof/signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialProof {
    /// Proof type
    #[serde(rename = "type")]
    pub type_: String,
    /// Creation timestamp
    pub created: String,
    /// Verification method used
    pub verification_method: String,
    /// Proof purpose
    pub proof_purpose: String,
    /// Proof value (signature)
    pub proof_value: String,
}

/// Result of credential verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialVerificationResult {
    /// Whether verification was successful
    pub valid: bool,
    /// Verification checks performed
    pub checks: VerificationChecks,
    /// Error message if verification failed
    pub error: Option<String>,
    /// Assurance level derived from credential
    pub assurance_level: Option<AssuranceLevel>,
}

impl Default for CredentialVerificationResult {
    fn default() -> Self {
        Self {
            valid: false,
            checks: VerificationChecks::default(),
            error: None,
            assurance_level: None,
        }
    }
}

/// Verification checks performed
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationChecks {
    pub signature_valid: bool,
    pub not_expired: bool,
    pub not_revoked: bool,
    pub issuer_trusted: bool,
    pub schema_valid: bool,
}

/// Revocation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationStatus {
    /// Credential ID
    pub credential_id: String,
    /// Current status
    pub status: RevocationState,
    /// Reason for revocation/suspension
    pub reason: Option<String>,
    /// When status was checked
    pub checked_at: i64,
}

/// Revocation state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevocationState {
    Active,
    Revoked,
    Suspended,
}

impl Default for RevocationState {
    fn default() -> Self {
        RevocationState::Active
    }
}

/// Identity verification request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityVerificationRequest {
    /// DID to verify
    pub did: String,
    /// Minimum assurance level required
    pub min_assurance_level: Option<AssuranceLevel>,
    /// Specific credentials to check
    pub required_credentials: Option<Vec<String>>,
    /// Requesting hApp identifier
    pub source_happ: String,
}

/// Identity verification response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityVerificationResponse {
    /// Verification ID
    pub id: String,
    /// DID that was verified
    pub did: String,
    /// Whether the DID exists and is active
    pub is_valid: bool,
    /// Whether the DID is deactivated
    pub is_deactivated: bool,
    /// Current assurance level
    pub assurance_level: AssuranceLevel,
    /// MATL reputation score (0.0 - 1.0)
    pub matl_score: f64,
    /// Total credentials held
    pub credential_count: u32,
    /// When DID was created
    pub did_created: Option<i64>,
    /// Verification timestamp
    pub verified_at: i64,
}

impl Default for IdentityVerificationResponse {
    fn default() -> Self {
        Self {
            id: String::new(),
            did: String::new(),
            is_valid: false,
            is_deactivated: false,
            assurance_level: AssuranceLevel::E0,
            matl_score: 0.5,
            credential_count: 0,
            did_created: None,
            verified_at: 0,
        }
    }
}

/// Cross-hApp reputation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossHappReputation {
    /// DID this reputation is for
    pub did: String,
    /// Reputation scores from various hApps
    pub scores: Vec<HappReputationScore>,
    /// Aggregated overall score
    pub aggregate_score: f64,
    /// Last update timestamp
    pub last_updated: i64,
}

/// Reputation score from a single hApp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HappReputationScore {
    /// hApp identifier
    pub happ_id: String,
    /// hApp name
    pub happ_name: String,
    /// Reputation score (0.0 - 1.0)
    pub score: f64,
    /// Number of interactions
    pub interactions: u64,
    /// When updated
    pub updated_at: i64,
}

/// Configuration for identity client
#[derive(Debug, Clone)]
pub struct IdentityClientConfig {
    /// Conductor URL for Holochain communication
    pub conductor_url: String,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Fallback mode
    pub fallback_mode: FallbackMode,
    /// Identity DNA role name
    pub identity_role_name: String,
}

impl Default for IdentityClientConfig {
    fn default() -> Self {
        Self {
            conductor_url: "ws://localhost:8888".to_string(),
            enable_cache: true,
            cache_ttl_secs: 300,
            fallback_mode: FallbackMode::Warn,
            identity_role_name: "identity".to_string(),
        }
    }
}

impl IdentityClientConfig {
    /// Create a new config with specified conductor URL
    pub fn with_conductor_url(conductor_url: impl Into<String>) -> Self {
        Self {
            conductor_url: conductor_url.into(),
            ..Default::default()
        }
    }
}

/// Fallback behavior when identity hApp is unavailable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackMode {
    /// Log warning and return defaults
    Warn,
    /// Return error
    Error,
    /// Silently return defaults
    Silent,
}

/// High-value transaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighValueTransactionConfig {
    /// Minimum value threshold
    pub threshold: f64,
    /// Currency
    pub currency: String,
    /// Required assurance level
    pub required_assurance_level: AssuranceLevel,
    /// Required credentials
    pub required_credentials: Option<Vec<String>>,
}

impl Default for HighValueTransactionConfig {
    fn default() -> Self {
        Self {
            threshold: 1000.0,
            currency: "USD".to_string(),
            required_assurance_level: AssuranceLevel::E2,
            required_credentials: None,
        }
    }
}

/// Guardian information for identity recovery and endorsement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guardian {
    /// Guardian's DID
    pub did: String,
    /// Guardian's display name (if available)
    pub name: Option<String>,
    /// Type of guardian relationship
    pub guardian_type: GuardianType,
    /// When the guardian relationship was established
    pub established_at: i64,
    /// Whether the guardian has endorsed this identity
    pub has_endorsed: bool,
    /// Endorsement timestamp (if endorsed)
    pub endorsed_at: Option<i64>,
    /// Trust weight of this guardian (0.0 - 1.0)
    pub trust_weight: f64,
}

impl Default for Guardian {
    fn default() -> Self {
        Self {
            did: String::new(),
            name: None,
            guardian_type: GuardianType::Social,
            established_at: 0,
            has_endorsed: false,
            endorsed_at: None,
            trust_weight: 1.0,
        }
    }
}

/// Type of guardian relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardianType {
    /// Social guardian (friend/family)
    Social,
    /// Institutional guardian (organization)
    Institutional,
    /// Hardware guardian (HSM/hardware wallet)
    Hardware,
    /// Recovery guardian (specifically for recovery)
    Recovery,
}

impl Default for GuardianType {
    fn default() -> Self {
        GuardianType::Social
    }
}

/// Result of guardian endorsement check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardianEndorsementResult {
    /// DID that was checked
    pub did: String,
    /// List of guardians
    pub guardians: Vec<Guardian>,
    /// Number of endorsements received
    pub endorsement_count: u32,
    /// Total number of guardians
    pub total_guardians: u32,
    /// Whether enough endorsements for recovery (usually 2/3)
    pub meets_recovery_threshold: bool,
    /// Aggregate trust score from guardians
    pub aggregate_trust: f64,
    /// When the check was performed
    pub checked_at: i64,
}
