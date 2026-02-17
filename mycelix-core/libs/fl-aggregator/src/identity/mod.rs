//! Multi-Factor Decentralized Identity (MFDI) System
//!
//! Provides graduated identity verification with multi-factor authentication,
//! Gitcoin Passport integration, MATL trust scoring, and FL participation gating.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use fl_aggregator::identity::{
//!     IdentityManager, AssuranceLevel, FactorCategory,
//!     CryptoKeyFactor, GitcoinPassportFactor,
//! };
//!
//! // Create identity manager
//! let mut manager = IdentityManager::new("did:mycelix:alice");
//!
//! // Add factors
//! manager.add_factor(Box::new(CryptoKeyFactor::generate()?));
//! manager.add_factor(Box::new(GitcoinPassportFactor::new("0x123...", 25.0)?));
//!
//! // Calculate assurance level
//! let result = manager.calculate_assurance();
//! assert!(result.level >= AssuranceLevel::E2);
//!
//! // Check FL participation eligibility
//! let requirements = FLParticipationRequirements::standard();
//! let admission = manager.check_fl_eligibility(&requirements)?;
//! ```
//!
//! ## Identity Factors (9 Types)
//!
//! - **Cryptographic**: Primary key pair (Ed25519), Hardware keys
//! - **Biometric**: Face, fingerprint, iris (hash only)
//! - **Social**: Recovery guardians (Shamir Secret Sharing), Peer attestations
//! - **External**: Gitcoin Passport, Verifiable Credentials
//! - **Knowledge**: Recovery phrases, Security questions

pub mod types;
pub mod factors;
pub mod assurance;
pub mod gitcoin_passport;
pub mod matl;
pub mod fl_participation;
pub mod freshness;
pub mod kvector_zkp;

pub use types::{
    AssuranceLevel, FactorCategory, FactorStatus,
    IdentityScope, Capability as IdentityCapability,
};
pub use factors::*;
pub use assurance::{
    AssuranceResult, calculate_assurance_level, get_required_level,
    has_capability, get_capabilities_summary, Capability,
};
pub use gitcoin_passport::{
    GitcoinPassportClient, GitcoinPassportConfig, PassportScore,
    Stamp, StampProvider, PassportError,
};
pub use matl::{IdentityMATLBridge, IdentityTrustSignal, EnhancedMATLScore, IdentityRiskLevel};
pub use fl_participation::{
    FLParticipationRequirements, FLAdmissionResult, FLDenialReason,
    FLCapabilities, FLParticipationRecord,
};
pub use freshness::{FactorFreshness, FactorDecayConfig};
pub use kvector_zkp::{
    KVectorVerifier, KVectorVerifierConfig, NodeKVectorProof,
    VerificationResult as KVectorVerificationResult, VerificationStatistics,
    KVectorZkpError, KVectorZkpResult,
};
#[cfg(feature = "kvector-zkp")]
pub use kvector_zkp::create_node_proof;

use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};
use thiserror::Error;

/// Identity error types
#[derive(Error, Debug)]
pub enum IdentityError {
    #[error("Factor not found: {0}")]
    FactorNotFound(String),

    #[error("Invalid factor: {0}")]
    InvalidFactor(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Insufficient assurance level: required {required:?}, actual {actual:?}")]
    InsufficientAssurance {
        required: AssuranceLevel,
        actual: AssuranceLevel,
    },

    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type IdentityResult<T> = Result<T, IdentityError>;

/// Core Mycelix Identity structure
#[derive(Debug, Clone)]
pub struct MycelixIdentity {
    /// W3C DID (did:mycelix:{agent_pub_key})
    pub did: String,

    /// Agent type classification
    pub agent_type: AgentType,

    /// Multi-factor authentication state
    pub mfa_state: MFAState,

    /// Verifiable credentials
    pub credentials: Vec<VerifiableCredential>,

    /// Recovery configuration
    pub recovery_config: RecoveryConfig,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last activity timestamp
    pub last_active: DateTime<Utc>,
}

/// Agent type classification
#[derive(Debug, Clone, PartialEq)]
pub enum AgentType {
    /// Human member with optional biometric proof
    HumanMember {
        biometric_hash: Option<String>,
        humanity_proofs: Vec<HumanityProof>,
    },

    /// AI/Instrumental actor with operator accountability
    InstrumentalActor {
        model_type: String,
        version: String,
        operator_did: String,
    },

    /// DAO collective with governance contract
    DAOCollective {
        member_dids: Vec<String>,
        governance_contract: String,
        threshold: u8,
    },
}

/// Humanity proof types
#[derive(Debug, Clone, PartialEq)]
pub enum HumanityProof {
    GitcoinPassport { score: f32, stamps: Vec<String> },
    VerifiedHumanity { credential_id: String },
    Biometric { proof_type: String, timestamp: DateTime<Utc> },
    SocialVouch { voucher_dids: Vec<String>, threshold: u8 },
}

/// Multi-factor authentication state
#[derive(Debug, Clone, Default)]
pub struct MFAState {
    /// Active identity factors
    pub factors: Vec<Box<dyn IdentityFactor>>,

    /// Recovery guardians configuration
    pub recovery_guardians: Vec<GuardianConfig>,

    /// Verification event history
    pub verification_history: Vec<VerificationEvent>,

    /// Current assurance level (cached)
    pub assurance_level: AssuranceLevel,
}

/// Guardian configuration for social recovery
#[derive(Debug, Clone)]
pub struct GuardianConfig {
    pub guardian_did: String,
    pub share_hash: String,
    pub trust_score: f32,
    pub relationship_type: RelationshipType,
    pub added_at: DateTime<Utc>,
    pub can_initiate_recovery: bool,
}

/// Relationship type for guardians
#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipType {
    Family,
    Friend,
    Colleague,
    Service,
    Unknown,
}

/// Verification event record
#[derive(Debug, Clone)]
pub struct VerificationEvent {
    pub factor_id: String,
    pub verified_at: DateTime<Utc>,
    pub success: bool,
    pub method: String,
    pub metadata: HashMap<String, String>,
}

/// Recovery configuration
#[derive(Debug, Clone, Default)]
pub struct RecoveryConfig {
    pub guardians_threshold: u8,
    pub total_guardians: u8,
    pub recovery_key_hash: Option<String>,
    pub recovery_phrase_hash: Option<String>,
    pub security_questions_configured: bool,
}

/// Verifiable Credential
#[derive(Debug, Clone)]
pub struct VerifiableCredential {
    pub id: String,
    pub issuer_did: String,
    pub subject_did: String,
    pub credential_type: VCType,
    pub claims: HashMap<String, serde_json::Value>,
    pub issuance_date: DateTime<Utc>,
    pub expiration_date: Option<DateTime<Utc>>,
    pub proof: VCProof,
    pub status: VCStatus,
}

/// Verifiable Credential types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VCType {
    VerifiedHumanity,
    GitcoinPassport,
    KYCBasic,
    KYCEnhanced,
    ProfessionalLicense,
    EducationDegree,
    MembershipProof,
    AgeOver18,
    ResidencyProof,
    Custom(String),
}

/// VC proof structure
#[derive(Debug, Clone)]
pub struct VCProof {
    pub proof_type: String,
    pub created: DateTime<Utc>,
    pub verification_method: String,
    pub signature: Vec<u8>,
}

/// VC status
#[derive(Debug, Clone, PartialEq)]
pub enum VCStatus {
    Active,
    Revoked,
    Expired,
    Suspended,
}

impl VerifiableCredential {
    /// Check if credential is currently valid
    pub fn is_valid(&self) -> bool {
        if self.status != VCStatus::Active {
            return false;
        }

        if let Some(expiry) = self.expiration_date {
            if Utc::now() > expiry {
                return false;
            }
        }

        true
    }
}

impl MycelixIdentity {
    /// Create a new identity
    pub fn new(did: String, agent_type: AgentType) -> Self {
        let now = Utc::now();
        Self {
            did,
            agent_type,
            mfa_state: MFAState::default(),
            credentials: Vec::new(),
            recovery_config: RecoveryConfig::default(),
            created_at: now,
            last_active: now,
        }
    }

    /// Add an identity factor
    pub fn add_factor(&mut self, factor: Box<dyn IdentityFactor>) {
        self.mfa_state.factors.push(factor);
        self.recalculate_assurance();
    }

    /// Remove a factor by ID
    pub fn remove_factor(&mut self, factor_id: &str) -> Option<Box<dyn IdentityFactor>> {
        if let Some(pos) = self.mfa_state.factors.iter().position(|f| f.factor_id() == factor_id) {
            let factor = self.mfa_state.factors.remove(pos);
            self.recalculate_assurance();
            Some(factor)
        } else {
            None
        }
    }

    /// Get active factors
    pub fn active_factors(&self) -> Vec<&dyn IdentityFactor> {
        self.mfa_state
            .factors
            .iter()
            .filter(|f| f.status() == FactorStatus::Active)
            .map(|f| f.as_ref())
            .collect()
    }

    /// Check if identity has a humanity proof
    pub fn has_humanity_proof(&self) -> bool {
        if let AgentType::HumanMember { humanity_proofs, .. } = &self.agent_type {
            !humanity_proofs.is_empty()
        } else {
            false
        }
    }

    /// Add a humanity proof
    pub fn add_humanity_proof(&mut self, proof: HumanityProof) {
        if let AgentType::HumanMember { humanity_proofs, .. } = &mut self.agent_type {
            humanity_proofs.push(proof);
        }
    }

    /// Recalculate assurance level from current factors
    pub fn recalculate_assurance(&mut self) {
        let factors: Vec<&dyn IdentityFactor> = self.mfa_state
            .factors
            .iter()
            .map(|f| f.as_ref())
            .collect();

        let result = calculate_assurance_level(&factors);
        self.mfa_state.assurance_level = result.level;
    }

    /// Get current assurance level
    pub fn assurance_level(&self) -> AssuranceLevel {
        self.mfa_state.assurance_level
    }

    /// Check if identity meets a required assurance level
    pub fn meets_assurance(&self, required: AssuranceLevel) -> bool {
        self.mfa_state.assurance_level >= required
    }

    /// Get factor categories present
    pub fn factor_categories(&self) -> HashSet<FactorCategory> {
        self.mfa_state
            .factors
            .iter()
            .filter(|f| f.status() == FactorStatus::Active)
            .map(|f| f.category())
            .collect()
    }

    /// Add a verifiable credential
    pub fn add_credential(&mut self, credential: VerifiableCredential) {
        self.credentials.push(credential);
    }

    /// Get valid credentials of a specific type
    pub fn credentials_of_type(&self, vc_type: &VCType) -> Vec<&VerifiableCredential> {
        self.credentials
            .iter()
            .filter(|vc| &vc.credential_type == vc_type && vc.is_valid())
            .collect()
    }

    /// Record a verification event
    pub fn record_verification(&mut self, factor_id: &str, success: bool, method: &str) {
        self.mfa_state.verification_history.push(VerificationEvent {
            factor_id: factor_id.to_string(),
            verified_at: Utc::now(),
            success,
            method: method.to_string(),
            metadata: HashMap::new(),
        });
        self.last_active = Utc::now();
    }

    /// Get identity age in days
    pub fn age_days(&self) -> i64 {
        (Utc::now() - self.created_at).num_days()
    }
}

impl Default for AssuranceLevel {
    fn default() -> Self {
        AssuranceLevel::E0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_creation() {
        let identity = MycelixIdentity::new(
            "did:mycelix:test123".to_string(),
            AgentType::HumanMember {
                biometric_hash: None,
                humanity_proofs: vec![],
            },
        );

        assert_eq!(identity.did, "did:mycelix:test123");
        assert_eq!(identity.assurance_level(), AssuranceLevel::E0);
        assert!(identity.mfa_state.factors.is_empty());
    }

    #[test]
    fn test_agent_types() {
        let human = AgentType::HumanMember {
            biometric_hash: Some("hash123".to_string()),
            humanity_proofs: vec![HumanityProof::GitcoinPassport {
                score: 25.0,
                stamps: vec!["github".to_string()],
            }],
        };

        let ai = AgentType::InstrumentalActor {
            model_type: "claude".to_string(),
            version: "3.5".to_string(),
            operator_did: "did:mycelix:operator".to_string(),
        };

        let dao = AgentType::DAOCollective {
            member_dids: vec!["did:mycelix:m1".to_string(), "did:mycelix:m2".to_string()],
            governance_contract: "contract_hash".to_string(),
            threshold: 2,
        };

        assert!(matches!(human, AgentType::HumanMember { .. }));
        assert!(matches!(ai, AgentType::InstrumentalActor { .. }));
        assert!(matches!(dao, AgentType::DAOCollective { .. }));
    }

    #[test]
    fn test_verifiable_credential_validity() {
        let valid_vc = VerifiableCredential {
            id: "vc-1".to_string(),
            issuer_did: "did:mycelix:issuer".to_string(),
            subject_did: "did:mycelix:subject".to_string(),
            credential_type: VCType::VerifiedHumanity,
            claims: HashMap::new(),
            issuance_date: Utc::now(),
            expiration_date: Some(Utc::now() + chrono::Duration::days(365)),
            proof: VCProof {
                proof_type: "Ed25519Signature2020".to_string(),
                created: Utc::now(),
                verification_method: "did:mycelix:issuer#key-1".to_string(),
                signature: vec![],
            },
            status: VCStatus::Active,
        };

        assert!(valid_vc.is_valid());

        let expired_vc = VerifiableCredential {
            status: VCStatus::Active,
            expiration_date: Some(Utc::now() - chrono::Duration::days(1)),
            ..valid_vc.clone()
        };

        assert!(!expired_vc.is_valid());

        let revoked_vc = VerifiableCredential {
            status: VCStatus::Revoked,
            ..valid_vc
        };

        assert!(!revoked_vc.is_valid());
    }
}
