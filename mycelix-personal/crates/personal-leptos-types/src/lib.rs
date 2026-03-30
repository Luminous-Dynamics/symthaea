// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! WASM-safe mirror types for the Mycelix Personal cluster.
//!
//! These types mirror `personal-types` but use only `serde` — no HDI/HDK
//! dependencies. Holochain `Timestamp` is replaced with `i64` (microseconds).

use serde::{Deserialize, Serialize};

// ============================================================================
// Vault Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VaultVisibility {
    Private,
    Public,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisclosureScope {
    ExistenceOnly,
    SelectedFields(Vec<String>),
    Full,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredentialType {
    Identity,
    Health,
    FederatedLearning,
    Governance,
    Domain(String),
}

impl CredentialType {
    pub fn label(&self) -> &str {
        match self {
            Self::Identity => "Identity",
            Self::Health => "Health",
            Self::FederatedLearning => "Federated Learning",
            Self::Governance => "Governance",
            Self::Domain(s) => s.as_str(),
        }
    }
}

// ============================================================================
// K-Vector Trust Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrustTier {
    Observer,
    Basic,
    Standard,
    Elevated,
    Guardian,
}

impl TrustTier {
    pub fn min_score(&self) -> f64 {
        match self {
            Self::Observer => 0.0,
            Self::Basic => 0.3,
            Self::Standard => 0.4,
            Self::Elevated => 0.6,
            Self::Guardian => 0.8,
        }
    }

    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            Self::Guardian
        } else if score >= 0.6 {
            Self::Elevated
        } else if score >= 0.4 {
            Self::Standard
        } else if score >= 0.3 {
            Self::Basic
        } else {
            Self::Observer
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Observer => "Observer",
            Self::Basic => "Basic",
            Self::Standard => "Standard",
            Self::Elevated => "Elevated",
            Self::Guardian => "Guardian",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Observer => "tier-observer",
            Self::Basic => "tier-basic",
            Self::Standard => "tier-standard",
            Self::Elevated => "tier-elevated",
            Self::Guardian => "tier-guardian",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrustScoreRange {
    pub lower: f32,
    pub upper: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KVectorComponent {
    Reputation,
    Activity,
    Integrity,
    Performance,
    Membership,
    Stake,
    History,
    Topology,
}

impl KVectorComponent {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Reputation => "Reputation",
            Self::Activity => "Activity",
            Self::Integrity => "Integrity",
            Self::Performance => "Performance",
            Self::Membership => "Membership",
            Self::Stake => "Stake",
            Self::History => "History",
            Self::Topology => "Topology",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationStatus {
    Pending,
    Fulfilled,
    Declined,
    Expired,
    Cancelled,
}

// ============================================================================
// Presentation Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialPresentation {
    pub credential_type: CredentialType,
    pub disclosed_data: String,
    pub scope: DisclosureScope,
    pub presented_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresentationRequest {
    pub credential_type: CredentialType,
    pub scope: DisclosureScope,
    pub context: Option<String>,
}

// ============================================================================
// View Types (UI-facing structs with String IDs)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileView {
    pub display_name: String,
    pub avatar: Option<String>,
    pub bio: Option<String>,
    pub metadata: std::collections::HashMap<String, String>,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterKeyView {
    pub label: String,
    pub purpose: String,
    pub public_key_hex: String,
    pub active: bool,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthRecordView {
    pub hash: String,
    pub record_type: String,
    pub data: String,
    pub source: String,
    pub event_date: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricView {
    pub hash: String,
    pub metric_type: String,
    pub value: f64,
    pub unit: String,
    pub measured_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentGrantView {
    pub hash: String,
    pub grantee: String,
    pub record_types: Vec<String>,
    pub expires_at: Option<i64>,
    pub active: bool,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredCredentialView {
    pub hash: String,
    pub credential_type: CredentialType,
    pub issuer: String,
    pub issued_at: i64,
    pub expires_at: Option<i64>,
    pub revoked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustCredentialView {
    pub id: String,
    pub subject_did: String,
    pub issuer_did: String,
    pub trust_tier: TrustTier,
    pub trust_score_range: TrustScoreRange,
    pub issued_at: i64,
    pub expires_at: Option<i64>,
    pub revoked: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vault_visibility_serde_roundtrip() {
        let vis = VaultVisibility::Private;
        let json = serde_json::to_string(&vis).unwrap();
        let back: VaultVisibility = serde_json::from_str(&json).unwrap();
        assert_eq!(back, VaultVisibility::Private);
    }

    #[test]
    fn disclosure_scope_selected_fields() {
        let scope = DisclosureScope::SelectedFields(vec!["name".into(), "email".into()]);
        let json = serde_json::to_string(&scope).unwrap();
        let back: DisclosureScope = serde_json::from_str(&json).unwrap();
        assert_eq!(back, scope);
    }

    #[test]
    fn credential_type_domain() {
        let ct = CredentialType::Domain("water_steward".into());
        let json = serde_json::to_string(&ct).unwrap();
        let back: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ct);
    }

    #[test]
    fn trust_tier_from_score_boundaries() {
        assert_eq!(TrustTier::from_score(0.0), TrustTier::Observer);
        assert_eq!(TrustTier::from_score(0.29), TrustTier::Observer);
        assert_eq!(TrustTier::from_score(0.3), TrustTier::Basic);
        assert_eq!(TrustTier::from_score(0.4), TrustTier::Standard);
        assert_eq!(TrustTier::from_score(0.6), TrustTier::Elevated);
        assert_eq!(TrustTier::from_score(0.8), TrustTier::Guardian);
        assert_eq!(TrustTier::from_score(1.0), TrustTier::Guardian);
    }

    #[test]
    fn trust_tier_min_score_consistent() {
        for tier in &[
            TrustTier::Observer,
            TrustTier::Basic,
            TrustTier::Standard,
            TrustTier::Elevated,
            TrustTier::Guardian,
        ] {
            assert_eq!(TrustTier::from_score(tier.min_score()), *tier);
        }
    }

    #[test]
    fn profile_view_serde_roundtrip() {
        let view = ProfileView {
            display_name: "Alice".into(),
            avatar: Some("https://example.com/avatar.png".into()),
            bio: Some("Hello world".into()),
            metadata: [("role".into(), "researcher".into())].into(),
            updated_at: 1_700_000_000,
        };
        let json = serde_json::to_string(&view).unwrap();
        let back: ProfileView = serde_json::from_str(&json).unwrap();
        assert_eq!(back.display_name, "Alice");
    }

    #[test]
    fn kvector_labels() {
        assert_eq!(KVectorComponent::Reputation.label(), "Reputation");
        assert_eq!(KVectorComponent::Topology.label(), "Topology");
    }

    #[test]
    fn trust_tier_css_classes() {
        assert_eq!(TrustTier::Observer.css_class(), "tier-observer");
        assert_eq!(TrustTier::Guardian.css_class(), "tier-guardian");
    }
}
