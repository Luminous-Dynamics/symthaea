// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe mirror types for the Mycelix Identity cluster.
//!
//! These types mirror the identity zome entry types but use only `serde` —
//! no HDI/HDK dependencies. AgentPubKey → String (base64),
//! ActionHash → String (base64), Timestamp → i64 (microseconds since epoch).

use serde::{Deserialize, Serialize};

// ============================================================================
// Enums
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssuranceLevel {
    Anonymous,
    Basic,
    Verified,
    HighlyAssured,
    ConstitutionallyCritical,
}

impl AssuranceLevel {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Anonymous => "Anonymous",
            Self::Basic => "Basic",
            Self::Verified => "Verified",
            Self::HighlyAssured => "Highly Assured",
            Self::ConstitutionallyCritical => "Constitutional",
        }
    }

    pub fn ordinal(&self) -> u8 {
        match self {
            Self::Anonymous => 0,
            Self::Basic => 1,
            Self::Verified => 2,
            Self::HighlyAssured => 3,
            Self::ConstitutionallyCritical => 4,
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Anonymous => "assurance-anonymous",
            Self::Basic => "assurance-basic",
            Self::Verified => "assurance-verified",
            Self::HighlyAssured => "assurance-high",
            Self::ConstitutionallyCritical => "assurance-constitutional",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorType {
    PrimaryKeyPair,
    HardwareKey,
    Biometric,
    SocialRecovery,
    ReputationAttestation,
    GitcoinPassport,
    VerifiableCredential,
    RecoveryPhrase,
    SecurityQuestions,
}

impl FactorType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::PrimaryKeyPair => "Primary Key",
            Self::HardwareKey => "Hardware Key",
            Self::Biometric => "Biometric",
            Self::SocialRecovery => "Social Recovery",
            Self::ReputationAttestation => "Reputation",
            Self::GitcoinPassport => "Gitcoin Passport",
            Self::VerifiableCredential => "Verifiable Credential",
            Self::RecoveryPhrase => "Recovery Phrase",
            Self::SecurityQuestions => "Security Questions",
        }
    }

    pub fn category(&self) -> &'static str {
        match self {
            Self::PrimaryKeyPair | Self::HardwareKey => "Cryptographic",
            Self::Biometric => "Biometric",
            Self::SocialRecovery | Self::ReputationAttestation => "Social Proof",
            Self::GitcoinPassport | Self::VerifiableCredential => "External Verification",
            Self::RecoveryPhrase | Self::SecurityQuestions => "Knowledge",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::PrimaryKeyPair => "\u{1F511}",
            Self::HardwareKey => "\u{1F4DF}",
            Self::Biometric => "\u{1F9EC}",
            Self::SocialRecovery => "\u{1F91D}",
            Self::ReputationAttestation => "\u{2B50}",
            Self::GitcoinPassport => "\u{1F30D}",
            Self::VerifiableCredential => "\u{1F4DC}",
            Self::RecoveryPhrase => "\u{1F4DD}",
            Self::SecurityQuestions => "\u{2753}",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustTier {
    Observer,
    Basic,
    Standard,
    Elevated,
    Guardian,
}

impl TrustTier {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Observer => "Observer",
            Self::Basic => "Basic",
            Self::Standard => "Standard",
            Self::Elevated => "Elevated",
            Self::Guardian => "Guardian",
        }
    }

    pub fn min_score(&self) -> f64 {
        match self {
            Self::Observer => 0.0,
            Self::Basic => 0.3,
            Self::Standard => 0.4,
            Self::Elevated => 0.6,
            Self::Guardian => 0.8,
        }
    }

    pub fn css_color(&self) -> &'static str {
        match self {
            Self::Observer => "#6b7280",
            Self::Basic => "#60a5fa",
            Self::Standard => "#34d399",
            Self::Elevated => "#a78bfa",
            Self::Guardian => "#fbbf24",
        }
    }

    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 { Self::Guardian }
        else if score >= 0.6 { Self::Elevated }
        else if score >= 0.4 { Self::Standard }
        else if score >= 0.3 { Self::Basic }
        else { Self::Observer }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStatus {
    Pending,
    Approved,
    ReadyToExecute,
    Completed,
    Rejected,
    Cancelled,
}

impl RecoveryStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pending => "Awaiting Votes",
            Self::Approved => "Approved",
            Self::ReadyToExecute => "Ready",
            Self::Completed => "Completed",
            Self::Rejected => "Rejected",
            Self::Cancelled => "Cancelled",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Pending | Self::Approved | Self::ReadyToExecute)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevocationStatus {
    Active,
    Suspended,
    Revoked,
}

// ============================================================================
// View Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMethodView {
    pub id: String,
    pub type_name: String,
    pub controller: String,
    pub public_key_multibase: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpointView {
    pub id: String,
    pub type_name: String,
    pub endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidDocumentView {
    pub id: String,
    pub controller: String,
    pub verification_methods: Vec<VerificationMethodView>,
    pub key_agreements: Vec<String>,
    pub services: Vec<ServiceEndpointView>,
    pub created: i64,
    pub updated: i64,
    pub version: u32,
    pub active: bool,
}

impl DidDocumentView {
    pub fn short_id(&self) -> String {
        if self.id.len() > 24 {
            format!("{}...{}", &self.id[..16], &self.id[self.id.len()-8..])
        } else {
            self.id.clone()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorView {
    pub factor_type: FactorType,
    pub factor_id: String,
    pub enrolled_at: i64,
    pub last_verified: i64,
    pub effective_strength: f32,
    pub active: bool,
    pub metadata: String,
}

impl FactorView {
    /// Strength decays over time. Returns visual percentage (0-100).
    pub fn visual_strength(&self, now_secs: i64) -> f32 {
        let age_days = (now_secs - self.last_verified) as f32 / 86400.0;
        let decay = (-age_days / 90.0).exp(); // 90-day half-life
        (self.effective_strength * decay * 100.0).clamp(0.0, 100.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaStateView {
    pub did: String,
    pub factors: Vec<FactorView>,
    pub assurance_level: AssuranceLevel,
    pub effective_strength: f32,
    pub category_count: u8,
    pub updated: i64,
}

impl MfaStateView {
    pub fn active_factors(&self) -> Vec<&FactorView> {
        self.factors.iter().filter(|f| f.active).collect()
    }

    pub fn factors_needed_for_next_level(&self) -> Option<(AssuranceLevel, u8)> {
        match self.assurance_level {
            AssuranceLevel::Anonymous => Some((AssuranceLevel::Basic, 1)),
            AssuranceLevel::Basic => Some((AssuranceLevel::Verified, 3)),
            AssuranceLevel::Verified => Some((AssuranceLevel::HighlyAssured, 5)),
            AssuranceLevel::HighlyAssured => Some((AssuranceLevel::ConstitutionallyCritical, 0)),
            AssuranceLevel::ConstitutionallyCritical => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfigView {
    pub did: String,
    #[serde(alias = "trustee_dids")]
    pub trustees: Vec<String>,
    pub threshold: u32,
    pub time_lock_secs: u64,
    pub active: bool,
    pub created: i64,
}

// =============================================================================
// Progressive Recovery Types
// =============================================================================

/// Verification anchor type for the frontend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationAnchorType {
    Phone,
    Email,
    Passkey,
    Device,
    Biometric,
}

impl VerificationAnchorType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Phone => "Phone Number",
            Self::Email => "Email Address",
            Self::Passkey => "Passkey",
            Self::Device => "Device",
            Self::Biometric => "Biometric",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Phone => "\u{1F4F1}",
            Self::Email => "\u{1F4E7}",
            Self::Passkey => "\u{1F511}",
            Self::Device => "\u{1F4BB}",
            Self::Biometric => "\u{1F9EC}",
        }
    }
}

/// A verification anchor as displayed in the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationAnchorView {
    pub anchor_type: VerificationAnchorType,
    pub enrolled_at: i64,
    pub masked_identifier: String,
}

/// Self-recovery configuration for the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfRecoveryConfigView {
    pub did: String,
    pub anchors: Vec<VerificationAnchorView>,
    pub anchor_threshold: u32,
    pub time_lock_secs: u64,
    pub active: bool,
    pub superseded_by_social: bool,
    pub created: i64,
}

/// Unified recovery strength indicator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecoveryTier {
    None,
    Pending,
    SelfRecovery { anchor_count: u32 },
    SocialRecovery { trustee_count: u32, threshold: u32 },
}

impl RecoveryTier {
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "Unprotected",
            Self::Pending => "Setup Needed",
            Self::SelfRecovery { .. } => "Self-Recovery",
            Self::SocialRecovery { .. } => "Social Recovery",
        }
    }

    pub fn strength(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Pending => 0.15,
            Self::SelfRecovery { anchor_count } => {
                0.3 + (*anchor_count as f64 * 0.15).min(0.3)
            }
            Self::SocialRecovery { .. } => 0.9,
        }
    }

    pub fn css_color(&self) -> &'static str {
        match self {
            Self::None => "#ff4444",
            Self::Pending => "#ffaa44",
            Self::SelfRecovery { .. } => "#44aaff",
            Self::SocialRecovery { .. } => "#44ff88",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryVoteView {
    pub trustee_did: String,
    pub vote: String,
    pub comment: Option<String>,
    pub voted_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryRequestView {
    pub id: String,
    pub did: String,
    pub status: RecoveryStatus,
    pub initiated_by: String,
    pub reason: String,
    pub votes: Vec<RecoveryVoteView>,
    pub time_lock_expires: Option<i64>,
    pub created: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialView {
    pub id: String,
    pub subject_did: String,
    pub issuer_did: String,
    pub credential_type: Vec<String>,
    pub claims: serde_json::Value,
    pub issued_at: i64,
    pub expires_at: Option<i64>,
    pub revoked: bool,
    pub schema_id: Option<String>,
}

impl CredentialView {
    pub fn is_expired(&self, now_secs: i64) -> bool {
        self.expires_at.map(|exp| now_secs > exp).unwrap_or(false)
    }

    pub fn status_label(&self, now_secs: i64) -> &'static str {
        if self.revoked { "Revoked" }
        else if self.is_expired(now_secs) { "Expired" }
        else { "Active" }
    }

    pub fn primary_type(&self) -> &str {
        self.credential_type.last().map(|s| s.as_str()).unwrap_or("Credential")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustCredentialView {
    pub id: String,
    pub subject_did: String,
    pub issuer_did: String,
    pub trust_tier: TrustTier,
    pub issued_at: i64,
    pub expires_at: Option<i64>,
    pub revoked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NameRegistryView {
    pub canonical: String,
    pub owner_did: String,
    pub registered_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainScoreView {
    pub domain: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessProfileView {
    pub identity: f64,
    pub reputation: f64,
    pub community: f64,
    pub engagement: f64,
}

impl ConsciousnessProfileView {
    pub fn combined_score(&self) -> f64 {
        (self.identity + self.reputation + self.community + self.engagement) / 4.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationView {
    pub agent: String,
    pub composite_score: f64,
    pub domain_scores: Vec<DomainScoreView>,
    pub consciousness_profile: ConsciousnessProfileView,
    pub trust_tier: TrustTier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationRequestView {
    pub id: String,
    pub requester_did: String,
    pub subject_did: String,
    pub purpose: String,
    pub expires_at: i64,
    pub status: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_tier_from_score() {
        assert_eq!(TrustTier::from_score(0.0), TrustTier::Observer);
        assert_eq!(TrustTier::from_score(0.3), TrustTier::Basic);
        assert_eq!(TrustTier::from_score(0.4), TrustTier::Standard);
        assert_eq!(TrustTier::from_score(0.6), TrustTier::Elevated);
        assert_eq!(TrustTier::from_score(0.8), TrustTier::Guardian);
        assert_eq!(TrustTier::from_score(1.0), TrustTier::Guardian);
    }

    #[test]
    fn test_did_short_id() {
        let did = DidDocumentView {
            id: "did:mycelix:uhCAk4YSfRTHgq0P0LfxR9ip-DPO4FcD".into(),
            controller: String::new(),
            verification_methods: vec![],
            key_agreements: vec![],
            services: vec![],
            created: 0, updated: 0, version: 1, active: true,
        };
        assert!(did.short_id().len() < did.id.len());
        assert!(did.short_id().contains("..."));
    }

    #[test]
    fn test_factor_visual_strength_decay() {
        let factor = FactorView {
            factor_type: FactorType::Biometric,
            factor_id: "bio-1".into(),
            enrolled_at: 0,
            last_verified: 1_000_000,
            effective_strength: 0.9,
            active: true,
            metadata: String::new(),
        };
        // Verified just now
        let fresh = factor.visual_strength(1_000_000);
        assert!(fresh > 85.0);
        // Verified 180 days ago
        let stale = factor.visual_strength(1_000_000 + 180 * 86400);
        assert!(stale < 20.0);
    }

    #[test]
    fn test_credential_status() {
        let cred = CredentialView {
            id: "c1".into(),
            subject_did: "did:me".into(),
            issuer_did: "did:them".into(),
            credential_type: vec!["EducationCredential".into()],
            claims: serde_json::json!({}),
            issued_at: 1_000_000,
            expires_at: Some(2_000_000),
            revoked: false,
            schema_id: None,
        };
        assert_eq!(cred.status_label(1_500_000), "Active");
        assert_eq!(cred.status_label(2_500_000), "Expired");
    }

    #[test]
    fn test_assurance_level_ordering() {
        assert!(AssuranceLevel::Anonymous.ordinal() < AssuranceLevel::Basic.ordinal());
        assert!(AssuranceLevel::Basic.ordinal() < AssuranceLevel::Verified.ordinal());
        assert!(AssuranceLevel::Verified.ordinal() < AssuranceLevel::HighlyAssured.ordinal());
    }

    #[test]
    fn test_consciousness_profile_combined() {
        let profile = ConsciousnessProfileView {
            identity: 0.8,
            reputation: 0.6,
            community: 0.4,
            engagement: 0.2,
        };
        assert!((profile.combined_score() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_serde_roundtrip() {
        let did = DidDocumentView {
            id: "did:mycelix:test".into(),
            controller: "agent".into(),
            verification_methods: vec![VerificationMethodView {
                id: "key-1".into(),
                type_name: "Ed25519".into(),
                controller: "did:mycelix:test".into(),
                public_key_multibase: "zBase58Key".into(),
            }],
            key_agreements: vec![],
            services: vec![],
            created: 1711900000, updated: 1711900000, version: 1, active: true,
        };
        let json = serde_json::to_string(&did).unwrap();
        let round: DidDocumentView = serde_json::from_str(&json).unwrap();
        assert_eq!(round.id, did.id);
    }
}
