//! Shared types for the Mycelix Personal (Sovereign) cluster.
//!
//! These types are used across all Personal zomes for consistent
//! data modeling and cross-zome communication.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

/// Visibility scope for vault entries.
///
/// Determines whether an entry is stored only on the agent's source chain
/// (private) or published to the DHT (public).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VaultVisibility {
    /// Stored only on the agent's source chain. Never published to DHT.
    Private,
    /// Published to the DHT. Visible to other agents on the same network.
    Public,
}

/// Scope of a selective disclosure request.
///
/// Controls what data the personal bridge reveals to other clusters
/// when handling cross-cluster queries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisclosureScope {
    /// Disclose only the existence of a credential (yes/no).
    ExistenceOnly,
    /// Disclose specific named fields from a credential or record.
    SelectedFields(Vec<String>),
    /// Full disclosure of the requested record.
    Full,
}

/// Type of verifiable credential stored in the wallet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredentialType {
    /// Identity credential (e.g., DID, government ID).
    Identity,
    /// Health credential (e.g., vaccination record, allergy info).
    Health,
    /// FL participation credential (e.g., Phi attestation, K-vector).
    FederatedLearning,
    /// Governance participation credential (e.g., voting eligibility).
    Governance,
    /// Domain-specific credential from Commons or Civic clusters.
    Domain(String),
}

// ============================================================================
// K-Vector Trust Types
// ============================================================================

/// Trust tiers for governance participation.
///
/// Derived from K-Vector `trust_score()` with defined thresholds.
/// Used by the credential wallet for trust credential issuance and
/// by governance for voting eligibility checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustTier {
    /// Trust score < 0.3 — Observer only, cannot vote.
    Observer,
    /// Trust score >= 0.3 — Basic participation.
    Basic,
    /// Trust score >= 0.4 — Can vote on major proposals.
    Standard,
    /// Trust score >= 0.6 — Can propose and vote on constitutional changes.
    Elevated,
    /// Trust score >= 0.8 — Full governance rights including emergency powers.
    Guardian,
}

impl TrustTier {
    /// Get the minimum trust score for this tier.
    pub fn min_score(&self) -> f64 {
        match self {
            TrustTier::Observer => 0.0,
            TrustTier::Basic => 0.3,
            TrustTier::Standard => 0.4,
            TrustTier::Elevated => 0.6,
            TrustTier::Guardian => 0.8,
        }
    }

    /// Determine tier from trust score.
    ///
    /// Accepts f64 to avoid precision loss at tier boundaries when computing
    /// the midpoint of an f32 range.
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            TrustTier::Guardian
        } else if score >= 0.6 {
            TrustTier::Elevated
        } else if score >= 0.4 {
            TrustTier::Standard
        } else if score >= 0.3 {
            TrustTier::Basic
        } else {
            TrustTier::Observer
        }
    }
}

/// Privacy-preserving trust score range.
///
/// Proves the trust score falls within a range without revealing the exact value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrustScoreRange {
    /// Lower bound (inclusive).
    pub lower: f32,
    /// Upper bound (inclusive).
    pub upper: f32,
}

/// K-Vector component identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KVectorComponent {
    /// k_r: Reputation
    Reputation,
    /// k_a: Activity
    Activity,
    /// k_i: Integrity
    Integrity,
    /// k_p: Performance
    Performance,
    /// k_m: Membership duration
    Membership,
    /// k_s: Stake weight
    Stake,
    /// k_h: Historical consistency
    History,
    /// k_topo: Network topology contribution
    Topology,
}

/// Status of an attestation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationStatus {
    /// Waiting for response.
    Pending,
    /// Attestation provided.
    Fulfilled,
    /// Request was declined.
    Declined,
    /// Request expired.
    Expired,
    /// Request was cancelled.
    Cancelled,
}

// ============================================================================
// Presentation Types
// ============================================================================

/// A signed attestation that can be presented via the personal bridge.
///
/// This is the output format for credential presentations — the bridge
/// wraps raw credentials into this structure before sending cross-cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialPresentation {
    /// Type of credential being presented.
    pub credential_type: CredentialType,
    /// The disclosed payload (filtered by DisclosureScope).
    pub disclosed_data: String,
    /// Scope used for this presentation.
    pub scope: DisclosureScope,
    /// Timestamp of the presentation.
    pub presented_at: Timestamp,
}

/// Input for requesting a credential presentation from the personal bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresentationRequest {
    /// What type of credential to present.
    pub credential_type: CredentialType,
    /// How much to disclose.
    pub scope: DisclosureScope,
    /// Optional context (e.g., "governance_vote:proposal_42").
    pub context: Option<String>,
}

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
    fn disclosure_scope_selected_fields_serde() {
        let scope = DisclosureScope::SelectedFields(vec!["name".into(), "email".into()]);
        let json = serde_json::to_string(&scope).unwrap();
        let back: DisclosureScope = serde_json::from_str(&json).unwrap();
        assert_eq!(back, scope);
    }

    #[test]
    fn credential_type_domain_serde() {
        let ct = CredentialType::Domain("water_steward".into());
        let json = serde_json::to_string(&ct).unwrap();
        let back: CredentialType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ct);
    }

    #[test]
    fn presentation_request_serde_roundtrip() {
        let req = PresentationRequest {
            credential_type: CredentialType::FederatedLearning,
            scope: DisclosureScope::ExistenceOnly,
            context: Some("governance_vote:prop_42".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: PresentationRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.credential_type, CredentialType::FederatedLearning);
        assert_eq!(back.scope, DisclosureScope::ExistenceOnly);
        assert_eq!(back.context.as_deref(), Some("governance_vote:prop_42"));
    }

    #[test]
    fn credential_presentation_serde_roundtrip() {
        let pres = CredentialPresentation {
            credential_type: CredentialType::Identity,
            disclosed_data: r#"{"name":"Alice"}"#.into(),
            scope: DisclosureScope::SelectedFields(vec!["name".into()]),
            presented_at: Timestamp::from_micros(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&pres).unwrap();
        let back: CredentialPresentation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.credential_type, CredentialType::Identity);
        assert!(back.disclosed_data.contains("Alice"));
    }

    #[test]
    fn trust_tier_from_score_all_boundaries() {
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
            let score = tier.min_score();
            assert_eq!(TrustTier::from_score(score), *tier);
        }
    }

    #[test]
    fn trust_tier_serde_roundtrip() {
        let tier = TrustTier::Elevated;
        let json = serde_json::to_string(&tier).unwrap();
        let back: TrustTier = serde_json::from_str(&json).unwrap();
        assert_eq!(back, TrustTier::Elevated);
    }

    #[test]
    fn trust_score_range_serde_roundtrip() {
        let range = TrustScoreRange {
            lower: 0.3,
            upper: 0.7,
        };
        let json = serde_json::to_string(&range).unwrap();
        let back: TrustScoreRange = serde_json::from_str(&json).unwrap();
        assert_eq!(back, range);
    }

    #[test]
    fn kvector_component_serde_roundtrip() {
        let comp = KVectorComponent::Topology;
        let json = serde_json::to_string(&comp).unwrap();
        let back: KVectorComponent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, KVectorComponent::Topology);
    }

    #[test]
    fn attestation_status_serde_roundtrip() {
        let status = AttestationStatus::Fulfilled;
        let json = serde_json::to_string(&status).unwrap();
        let back: AttestationStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, AttestationStatus::Fulfilled);
    }

    #[test]
    fn all_credential_types_are_distinct() {
        let types = vec![
            CredentialType::Identity,
            CredentialType::Health,
            CredentialType::FederatedLearning,
            CredentialType::Governance,
            CredentialType::Domain("test".into()),
        ];
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Credential types at {} and {} should differ", i, j);
                }
            }
        }
    }
}
