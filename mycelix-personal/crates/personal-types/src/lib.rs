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
