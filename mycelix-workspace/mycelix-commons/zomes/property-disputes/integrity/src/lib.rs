// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property Disputes Integrity Zome
use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PropertyDispute {
    pub id: String,
    pub property_id: String,
    pub dispute_type: DisputeType,
    pub claimant_did: String,
    pub respondent_did: String,
    pub description: String,
    pub evidence_ids: Vec<String>,
    pub status: DisputeStatus,
    pub justice_case_id: Option<String>,
    pub filed: Timestamp,
    pub resolved: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeType {
    Boundary,
    Ownership,
    Encumbrance,
    Easement,
    Trespass,
    Damage,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeStatus {
    Filed,
    UnderReview,
    Mediation,
    Arbitration,
    Resolved,
    Dismissed,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OwnershipClaim {
    pub id: String,
    pub property_id: String,
    pub claimant_did: String,
    pub claim_basis: ClaimBasis,
    pub supporting_documents: Vec<String>,
    pub status: ClaimStatus,
    pub filed: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ClaimBasis {
    PriorOwnership,
    Inheritance,
    AdversePossession,
    FraudulentTransfer,
    DocumentaryEvidence,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ClaimStatus {
    Pending,
    UnderInvestigation,
    Validated,
    Rejected,
    Superseded,
}

/// Anchor entry for deterministic link bases from strings
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    PropertyDispute(PropertyDispute),
    OwnershipClaim(OwnershipClaim),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    PropertyToDisputes,
    ClaimantToDisputes,
    PropertyToClaims,
    DisputeToJustice,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::PropertyDispute(dispute) => {
                    validate_create_property_dispute(EntryCreationAction::Create(action), dispute)
                }
                EntryTypes::OwnershipClaim(claim) => {
                    validate_create_ownership_claim(EntryCreationAction::Create(action), claim)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::PropertyDispute(dispute) => {
                    validate_update_property_dispute(action, dispute)
                }
                EntryTypes::OwnershipClaim(claim) => validate_update_ownership_claim(action, claim),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Invalid(
                    "Anchors cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::PropertyToDisputes => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PropertyToDisputes link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ClaimantToDisputes => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ClaimantToDisputes link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PropertyToClaims => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PropertyToClaims link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::DisputeToJustice => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "DisputeToJustice link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_create_property_dispute(
    _action: EntryCreationAction,
    dispute: PropertyDispute,
) -> ExternResult<ValidateCallbackResult> {
    // --- Empty string checks (required fields) ---
    if dispute.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "PropertyDispute id cannot be empty".into(),
        ));
    }
    if dispute.property_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "PropertyDispute property_id cannot be empty".into(),
        ));
    }
    if dispute.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "PropertyDispute description cannot be empty".into(),
        ));
    }
    // --- String length limits ---
    if dispute.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "PropertyDispute id too long (max 256 chars)".into(),
        ));
    }
    if dispute.property_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "PropertyDispute property_id too long (max 256 chars)".into(),
        ));
    }
    if dispute.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "PropertyDispute description too long (max 4096 chars)".into(),
        ));
    }
    for evidence_id in &dispute.evidence_ids {
        if evidence_id.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "PropertyDispute evidence_id too long (max 256 chars)".into(),
            ));
        }
    }
    if let Some(ref justice_case_id) = dispute.justice_case_id {
        if justice_case_id.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "PropertyDispute justice_case_id too long (max 256 chars)".into(),
            ));
        }
    }
    // --- Existing validation ---
    if !dispute.claimant_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Claimant must be a valid DID".into(),
        ));
    }
    if !dispute.respondent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Respondent must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_property_dispute(
    _action: Update,
    _dispute: PropertyDispute,
) -> ExternResult<ValidateCallbackResult> {
    // Status can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_ownership_claim(
    _action: EntryCreationAction,
    claim: OwnershipClaim,
) -> ExternResult<ValidateCallbackResult> {
    if !claim.claimant_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Claimant must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_ownership_claim(
    _action: Update,
    _claim: OwnershipClaim,
) -> ExternResult<ValidateCallbackResult> {
    // Status can be updated
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Factory Functions
    // ============================================================================

    fn mock_agent_pub_key() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn mock_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn mock_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000) // 2024-01-01 00:00:00 UTC
    }

    fn mock_create_action() -> Create {
        Create {
            author: mock_agent_pub_key(),
            timestamp: mock_timestamp(),
            action_seq: 0,
            prev_action: mock_action_hash(),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    fn mock_update_action() -> Update {
        Update {
            author: mock_agent_pub_key(),
            timestamp: mock_timestamp(),
            action_seq: 1,
            prev_action: mock_action_hash(),
            original_action_address: mock_action_hash(),
            original_entry_address: EntryHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    fn factory_property_dispute(claimant_did: &str, respondent_did: &str) -> PropertyDispute {
        PropertyDispute {
            id: "dispute_001".to_string(),
            property_id: "prop_123".to_string(),
            dispute_type: DisputeType::Boundary,
            claimant_did: claimant_did.to_string(),
            respondent_did: respondent_did.to_string(),
            description: "Boundary line dispute".to_string(),
            evidence_ids: vec!["evidence_1".to_string()],
            status: DisputeStatus::Filed,
            justice_case_id: None,
            filed: mock_timestamp(),
            resolved: None,
        }
    }

    fn factory_ownership_claim(claimant_did: &str) -> OwnershipClaim {
        OwnershipClaim {
            id: "claim_001".to_string(),
            property_id: "prop_456".to_string(),
            claimant_did: claimant_did.to_string(),
            claim_basis: ClaimBasis::PriorOwnership,
            supporting_documents: vec!["doc_1".to_string()],
            status: ClaimStatus::Pending,
            filed: mock_timestamp(),
        }
    }

    // ============================================================================
    // PropertyDispute Create Validation Tests
    // ============================================================================

    #[test]
    fn test_create_property_dispute_valid() {
        let dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_property_dispute_claimant_empty() {
        let dispute = factory_property_dispute("", "did:key:def456");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_property_dispute_claimant_no_prefix() {
        let dispute = factory_property_dispute("key:abc123", "did:key:def456");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_property_dispute_claimant_invalid_format() {
        let dispute = factory_property_dispute("not-a-did", "did:key:def456");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_property_dispute_respondent_empty() {
        let dispute = factory_property_dispute("did:key:abc123", "");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_property_dispute_respondent_no_prefix() {
        let dispute = factory_property_dispute("did:key:abc123", "key:def456");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_property_dispute_respondent_invalid_format() {
        let dispute = factory_property_dispute("did:key:abc123", "not-a-did");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_property_dispute_both_invalid() {
        let dispute = factory_property_dispute("invalid1", "invalid2");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_property_dispute_with_justice_case_id() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.justice_case_id = Some("justice_case_789".to_string());
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_property_dispute_with_resolved_timestamp() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.resolved = Some(mock_timestamp());
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ============================================================================
    // PropertyDispute Update Validation Tests
    // ============================================================================

    #[test]
    fn test_update_property_dispute_always_valid() {
        let dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        let action = mock_update_action();
        let result = validate_update_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_property_dispute_invalid_did_still_valid() {
        // Update validation doesn't check DIDs
        let dispute = factory_property_dispute("invalid", "also-invalid");
        let action = mock_update_action();
        let result = validate_update_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_property_dispute_status_change() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.status = DisputeStatus::Resolved;
        dispute.resolved = Some(mock_timestamp());
        let action = mock_update_action();
        let result = validate_update_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ============================================================================
    // OwnershipClaim Create Validation Tests
    // ============================================================================

    #[test]
    fn test_create_ownership_claim_valid() {
        let claim = factory_ownership_claim("did:key:abc123");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_ownership_claim(action, claim).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_create_ownership_claim_claimant_empty() {
        let claim = factory_ownership_claim("");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_ownership_claim(action, claim).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_ownership_claim_claimant_no_prefix() {
        let claim = factory_ownership_claim("key:abc123");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_ownership_claim(action, claim).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_ownership_claim_claimant_invalid_format() {
        let claim = factory_ownership_claim("not-a-did");
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_ownership_claim(action, claim).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_create_ownership_claim_different_did_methods() {
        let valid_dids = vec![
            "did:key:z6Mk...",
            "did:web:example.com",
            "did:pkh:eth:0x...",
            "did:ion:EiD...",
        ];

        for did in valid_dids {
            let claim = factory_ownership_claim(did);
            let action = EntryCreationAction::Create(mock_create_action());
            let result = validate_create_ownership_claim(action, claim).unwrap();
            assert_eq!(
                result,
                ValidateCallbackResult::Valid,
                "Failed for DID: {}",
                did
            );
        }
    }

    // ============================================================================
    // OwnershipClaim Update Validation Tests
    // ============================================================================

    #[test]
    fn test_update_ownership_claim_always_valid() {
        let claim = factory_ownership_claim("did:key:abc123");
        let action = mock_update_action();
        let result = validate_update_ownership_claim(action, claim).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_ownership_claim_invalid_did_still_valid() {
        // Update validation doesn't check DIDs
        let claim = factory_ownership_claim("invalid");
        let action = mock_update_action();
        let result = validate_update_ownership_claim(action, claim).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_ownership_claim_status_change() {
        let mut claim = factory_ownership_claim("did:key:abc123");
        claim.status = ClaimStatus::Validated;
        let action = mock_update_action();
        let result = validate_update_ownership_claim(action, claim).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ============================================================================
    // Anchor Tests
    // ============================================================================

    #[test]
    fn test_anchor_creation() {
        let anchor = Anchor("property_disputes".to_string());
        assert_eq!(anchor.0, "property_disputes");
    }

    #[test]
    fn test_anchor_clone() {
        let anchor1 = Anchor("test".to_string());
        let anchor2 = anchor1.clone();
        assert_eq!(anchor1, anchor2);
    }

    // ============================================================================
    // DisputeType Enum Tests
    // ============================================================================

    #[test]
    fn test_dispute_type_boundary() {
        let dt = DisputeType::Boundary;
        assert_eq!(dt, DisputeType::Boundary);
    }

    #[test]
    fn test_dispute_type_ownership() {
        let dt = DisputeType::Ownership;
        assert_eq!(dt, DisputeType::Ownership);
    }

    #[test]
    fn test_dispute_type_encumbrance() {
        let dt = DisputeType::Encumbrance;
        assert_eq!(dt, DisputeType::Encumbrance);
    }

    #[test]
    fn test_dispute_type_easement() {
        let dt = DisputeType::Easement;
        assert_eq!(dt, DisputeType::Easement);
    }

    #[test]
    fn test_dispute_type_trespass() {
        let dt = DisputeType::Trespass;
        assert_eq!(dt, DisputeType::Trespass);
    }

    #[test]
    fn test_dispute_type_damage() {
        let dt = DisputeType::Damage;
        assert_eq!(dt, DisputeType::Damage);
    }

    #[test]
    fn test_dispute_type_other() {
        let dt = DisputeType::Other("Custom dispute".to_string());
        assert_eq!(dt, DisputeType::Other("Custom dispute".to_string()));
    }

    #[test]
    fn test_dispute_type_serde() {
        let dt = DisputeType::Boundary;
        let serialized = serde_json::to_string(&dt).unwrap();
        let deserialized: DisputeType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(dt, deserialized);
    }

    #[test]
    fn test_dispute_type_other_serde() {
        let dt = DisputeType::Other("Noise complaint".to_string());
        let serialized = serde_json::to_string(&dt).unwrap();
        let deserialized: DisputeType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(dt, deserialized);
    }

    // ============================================================================
    // DisputeStatus Enum Tests
    // ============================================================================

    #[test]
    fn test_dispute_status_filed() {
        let ds = DisputeStatus::Filed;
        assert_eq!(ds, DisputeStatus::Filed);
    }

    #[test]
    fn test_dispute_status_under_review() {
        let ds = DisputeStatus::UnderReview;
        assert_eq!(ds, DisputeStatus::UnderReview);
    }

    #[test]
    fn test_dispute_status_mediation() {
        let ds = DisputeStatus::Mediation;
        assert_eq!(ds, DisputeStatus::Mediation);
    }

    #[test]
    fn test_dispute_status_arbitration() {
        let ds = DisputeStatus::Arbitration;
        assert_eq!(ds, DisputeStatus::Arbitration);
    }

    #[test]
    fn test_dispute_status_resolved() {
        let ds = DisputeStatus::Resolved;
        assert_eq!(ds, DisputeStatus::Resolved);
    }

    #[test]
    fn test_dispute_status_dismissed() {
        let ds = DisputeStatus::Dismissed;
        assert_eq!(ds, DisputeStatus::Dismissed);
    }

    #[test]
    fn test_dispute_status_serde() {
        let ds = DisputeStatus::Mediation;
        let serialized = serde_json::to_string(&ds).unwrap();
        let deserialized: DisputeStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(ds, deserialized);
    }

    // ============================================================================
    // ClaimBasis Enum Tests
    // ============================================================================

    #[test]
    fn test_claim_basis_prior_ownership() {
        let cb = ClaimBasis::PriorOwnership;
        assert_eq!(cb, ClaimBasis::PriorOwnership);
    }

    #[test]
    fn test_claim_basis_inheritance() {
        let cb = ClaimBasis::Inheritance;
        assert_eq!(cb, ClaimBasis::Inheritance);
    }

    #[test]
    fn test_claim_basis_adverse_possession() {
        let cb = ClaimBasis::AdversePossession;
        assert_eq!(cb, ClaimBasis::AdversePossession);
    }

    #[test]
    fn test_claim_basis_fraudulent_transfer() {
        let cb = ClaimBasis::FraudulentTransfer;
        assert_eq!(cb, ClaimBasis::FraudulentTransfer);
    }

    #[test]
    fn test_claim_basis_documentary_evidence() {
        let cb = ClaimBasis::DocumentaryEvidence;
        assert_eq!(cb, ClaimBasis::DocumentaryEvidence);
    }

    #[test]
    fn test_claim_basis_other() {
        let cb = ClaimBasis::Other("Tribal rights".to_string());
        assert_eq!(cb, ClaimBasis::Other("Tribal rights".to_string()));
    }

    #[test]
    fn test_claim_basis_serde() {
        let cb = ClaimBasis::Inheritance;
        let serialized = serde_json::to_string(&cb).unwrap();
        let deserialized: ClaimBasis = serde_json::from_str(&serialized).unwrap();
        assert_eq!(cb, deserialized);
    }

    #[test]
    fn test_claim_basis_other_serde() {
        let cb = ClaimBasis::Other("Customary law".to_string());
        let serialized = serde_json::to_string(&cb).unwrap();
        let deserialized: ClaimBasis = serde_json::from_str(&serialized).unwrap();
        assert_eq!(cb, deserialized);
    }

    // ============================================================================
    // ClaimStatus Enum Tests
    // ============================================================================

    #[test]
    fn test_claim_status_pending() {
        let cs = ClaimStatus::Pending;
        assert_eq!(cs, ClaimStatus::Pending);
    }

    #[test]
    fn test_claim_status_under_investigation() {
        let cs = ClaimStatus::UnderInvestigation;
        assert_eq!(cs, ClaimStatus::UnderInvestigation);
    }

    #[test]
    fn test_claim_status_validated() {
        let cs = ClaimStatus::Validated;
        assert_eq!(cs, ClaimStatus::Validated);
    }

    #[test]
    fn test_claim_status_rejected() {
        let cs = ClaimStatus::Rejected;
        assert_eq!(cs, ClaimStatus::Rejected);
    }

    #[test]
    fn test_claim_status_superseded() {
        let cs = ClaimStatus::Superseded;
        assert_eq!(cs, ClaimStatus::Superseded);
    }

    #[test]
    fn test_claim_status_serde() {
        let cs = ClaimStatus::UnderInvestigation;
        let serialized = serde_json::to_string(&cs).unwrap();
        let deserialized: ClaimStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(cs, deserialized);
    }

    // ============================================================================
    // PropertyDispute Struct Tests
    // ============================================================================

    #[test]
    fn test_property_dispute_clone() {
        let dispute1 = factory_property_dispute("did:key:abc123", "did:key:def456");
        let dispute2 = dispute1.clone();
        assert_eq!(dispute1, dispute2);
    }

    #[test]
    fn test_property_dispute_with_all_fields() {
        let dispute = PropertyDispute {
            id: "dispute_full".to_string(),
            property_id: "prop_999".to_string(),
            dispute_type: DisputeType::Other("Water rights".to_string()),
            claimant_did: "did:key:claimant".to_string(),
            respondent_did: "did:key:respondent".to_string(),
            description: "Full description".to_string(),
            evidence_ids: vec!["ev1".to_string(), "ev2".to_string(), "ev3".to_string()],
            status: DisputeStatus::Arbitration,
            justice_case_id: Some("justice_999".to_string()),
            filed: mock_timestamp(),
            resolved: Some(mock_timestamp()),
        };
        assert_eq!(dispute.evidence_ids.len(), 3);
        assert!(dispute.justice_case_id.is_some());
        assert!(dispute.resolved.is_some());
    }

    // ============================================================================
    // OwnershipClaim Struct Tests
    // ============================================================================

    #[test]
    fn test_ownership_claim_clone() {
        let claim1 = factory_ownership_claim("did:key:abc123");
        let claim2 = claim1.clone();
        assert_eq!(claim1, claim2);
    }

    #[test]
    fn test_ownership_claim_with_all_fields() {
        let claim = OwnershipClaim {
            id: "claim_full".to_string(),
            property_id: "prop_888".to_string(),
            claimant_did: "did:key:claimant123".to_string(),
            claim_basis: ClaimBasis::Other("Treaty rights".to_string()),
            supporting_documents: vec![
                "deed1.pdf".to_string(),
                "survey.pdf".to_string(),
                "affidavit.pdf".to_string(),
            ],
            status: ClaimStatus::Validated,
            filed: mock_timestamp(),
        };
        assert_eq!(claim.supporting_documents.len(), 3);
        assert_eq!(claim.status, ClaimStatus::Validated);
    }

    // ============================================================================
    // String Length & Empty Validation Tests
    // ============================================================================

    #[test]
    fn test_property_dispute_empty_id() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.id = "".to_string();
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_property_dispute_whitespace_id() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.id = "   ".to_string();
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_property_dispute_id_too_long() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.id = "x".repeat(257);
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_property_dispute_id_at_limit() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.id = "x".repeat(64);
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_property_dispute_empty_property_id() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.property_id = "".to_string();
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_property_dispute_property_id_too_long() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.property_id = "x".repeat(257);
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_property_dispute_empty_description() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.description = "".to_string();
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_property_dispute_description_too_long() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.description = "x".repeat(4097);
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_property_dispute_description_at_limit() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.description = "x".repeat(4096);
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_property_dispute_evidence_id_too_long() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.evidence_ids = vec!["x".repeat(257)];
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_property_dispute_evidence_id_at_limit() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.evidence_ids = vec!["x".repeat(64)];
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_property_dispute_justice_case_id_too_long() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.justice_case_id = Some("x".repeat(257));
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_property_dispute_justice_case_id_at_limit() {
        let mut dispute = factory_property_dispute("did:key:abc123", "did:key:def456");
        dispute.justice_case_id = Some("x".repeat(64));
        let action = EntryCreationAction::Create(mock_create_action());
        let result = validate_create_property_dispute(action, dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }
}
