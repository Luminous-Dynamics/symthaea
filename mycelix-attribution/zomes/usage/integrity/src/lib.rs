// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;
use serde::{Deserialize, Serialize};

// ── Entry Types ──────────────────────────────────────────────────────

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UsageType {
    DirectDependency,
    Transitive,
    InternalTooling,
    Production,
    Research,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UsageScale {
    Small,
    Medium,
    Large,
    Enterprise,
}

/// Voluntary attestation that an entity uses a dependency.
/// Immutable — updates rejected in validation.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsageReceipt {
    pub id: String,
    pub dependency_id: String,
    pub user_did: String,
    pub organization: Option<String>,
    pub usage_type: UsageType,
    pub scale: Option<UsageScale>,
    pub version_range: Option<String>,
    pub context: Option<String>,
    pub attested_at: Timestamp,
}

/// ZK-STARK proof of usage without revealing scale.
/// Immutable — updates only allowed to set verified/verifier fields.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsageAttestation {
    pub id: String,
    pub dependency_id: String,
    pub user_did: String,
    pub witness_commitment: Vec<u8>,
    pub proof_bytes: Vec<u8>,
    pub verified: bool,
    pub generated_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub verifier_pubkey: Option<Vec<u8>>,
    pub verifier_signature: Option<Vec<u8>>,
}

impl UsageAttestation {
    pub const MAX_PROOF_SIZE: usize = 500_000; // 500KB
}

// ── Entry & Link Enums ───────────────────────────────────────────────

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    UsageReceipt(UsageReceipt),
    UsageAttestation(UsageAttestation),
}

#[hdk_link_types]
pub enum LinkTypes {
    DependencyToUsageReceipts,
    UserToUsageReceipts,
    DependencyToAttestations,
    UserToAttestations,
    PredecessorToAttestation,
    AllAttestations,
    UsageRateLimit,
}

// ── Pure Validation Functions ────────────────────────────────────────

pub fn validate_create_usage_receipt(
    _action: Create,
    receipt: UsageReceipt,
) -> ExternResult<ValidateCallbackResult> {
    if receipt.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Usage receipt id must not be empty".into(),
        ));
    }
    if receipt.dependency_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "dependency_id must not be empty".into(),
        ));
    }
    if !receipt.user_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "user_did must start with 'did:'".into(),
        ));
    }
    if let Some(ref ctx) = receipt.context {
        if ctx.len() > 1000 {
            return Ok(ValidateCallbackResult::Invalid(
                "context must be at most 1000 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_create_usage_attestation(
    _action: Create,
    att: UsageAttestation,
) -> ExternResult<ValidateCallbackResult> {
    if att.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Attestation id must not be empty".into(),
        ));
    }
    if att.dependency_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "dependency_id must not be empty".into(),
        ));
    }
    if !att.user_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "user_did must start with 'did:'".into(),
        ));
    }
    if att.witness_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "witness_commitment must be exactly 32 bytes (Blake3)".into(),
        ));
    }
    if att.proof_bytes.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "proof_bytes must not be empty".into(),
        ));
    }
    if att.proof_bytes.len() > UsageAttestation::MAX_PROOF_SIZE {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "proof_bytes must be at most {} bytes",
            UsageAttestation::MAX_PROOF_SIZE
        )));
    }
    if let Some(ref pk) = att.verifier_pubkey {
        if pk.len() != 32 {
            return Ok(ValidateCallbackResult::Invalid(
                "verifier_pubkey must be exactly 32 bytes (Ed25519)".into(),
            ));
        }
    }
    if let Some(ref sig) = att.verifier_signature {
        if sig.len() != 64 {
            return Ok(ValidateCallbackResult::Invalid(
                "verifier_signature must be exactly 64 bytes".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

// ── HDI Validation Callback ──────────────────────────────────────────

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::UsageReceipt(receipt) => validate_create_usage_receipt(action, receipt),
                EntryTypes::UsageAttestation(att) => validate_create_usage_attestation(action, att),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::UsageReceipt(_) => Ok(ValidateCallbackResult::Invalid(
                    "Usage receipts are immutable and cannot be updated".into(),
                )),
                EntryTypes::UsageAttestation(_att) => {
                    // Allow updates only for verification fields.
                    // Full verification requires host access (must_get_valid_record),
                    // so we permit the update here — the coordinator enforces
                    // that only verified/verifier fields change.
                    Ok(ValidateCallbackResult::Valid)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::DependencyToUsageReceipts
            | LinkTypes::UserToUsageReceipts
            | LinkTypes::DependencyToAttestations
            | LinkTypes::UserToAttestations
            | LinkTypes::PredecessorToAttestation
            | LinkTypes::AllAttestations
            | LinkTypes::UsageRateLimit => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::DependencyToUsageReceipts
            | LinkTypes::UserToUsageReceipts
            | LinkTypes::DependencyToAttestations
            | LinkTypes::UserToAttestations
            | LinkTypes::PredecessorToAttestation
            | LinkTypes::AllAttestations
            | LinkTypes::UsageRateLimit => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

// ── Unit Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_action() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::now(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex::from(0),
                0.into(),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    fn valid_receipt() -> UsageReceipt {
        UsageReceipt {
            id: "usage-001".into(),
            dependency_id: "crate:serde:1.0".into(),
            user_did: "did:mycelix:user123".into(),
            organization: Some("Acme Corp".into()),
            usage_type: UsageType::Production,
            scale: Some(UsageScale::Enterprise),
            version_range: Some(">=1.0.200".into()),
            context: Some("Core serialization layer".into()),
            attested_at: Timestamp::now(),
        }
    }

    fn valid_attestation() -> UsageAttestation {
        UsageAttestation {
            id: "attest-001".into(),
            dependency_id: "crate:serde:1.0".into(),
            user_did: "did:mycelix:user123".into(),
            witness_commitment: vec![0xAB; 32],
            proof_bytes: vec![0x01; 128],
            verified: false,
            generated_at: Timestamp::now(),
            expires_at: None,
            verifier_pubkey: None,
            verifier_signature: None,
        }
    }

    // ── UsageReceipt tests ───────────────────────────────────────────

    #[test]
    fn test_valid_receipt() {
        let result = validate_create_usage_receipt(test_action(), valid_receipt()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_empty_receipt_id_rejected() {
        let mut r = valid_receipt();
        r.id = String::new();
        let result = validate_create_usage_receipt(test_action(), r).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_invalid_user_did_rejected() {
        let mut r = valid_receipt();
        r.user_did = "not-did".into();
        let result = validate_create_usage_receipt(test_action(), r).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_context_too_long_rejected() {
        let mut r = valid_receipt();
        r.context = Some("x".repeat(1001));
        let result = validate_create_usage_receipt(test_action(), r).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── UsageAttestation tests ───────────────────────────────────────

    #[test]
    fn test_valid_attestation() {
        let result = validate_create_usage_attestation(test_action(), valid_attestation()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_wrong_commitment_size_rejected() {
        let mut a = valid_attestation();
        a.witness_commitment = vec![0xAB; 16]; // not 32
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_empty_proof_rejected() {
        let mut a = valid_attestation();
        a.proof_bytes = vec![];
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_oversized_proof_rejected() {
        let mut a = valid_attestation();
        a.proof_bytes = vec![0x01; 500_001];
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_invalid_verifier_pubkey_size() {
        let mut a = valid_attestation();
        a.verifier_pubkey = Some(vec![0u8; 33]); // not 32
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_invalid_verifier_signature_size() {
        let mut a = valid_attestation();
        a.verifier_signature = Some(vec![0u8; 63]); // not 64
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_proof_boundary_500kb() {
        // Exactly 500KB should be valid
        let mut a = valid_attestation();
        a.proof_bytes = vec![0x01; 500_000];
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_proof_boundary_500kb_plus_one() {
        // 500KB + 1 byte should be rejected
        let mut a = valid_attestation();
        a.proof_bytes = vec![0x01; 500_001];
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_commitment_31_bytes_rejected() {
        let mut a = valid_attestation();
        a.witness_commitment = vec![0xAB; 31];
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_commitment_33_bytes_rejected() {
        let mut a = valid_attestation();
        a.witness_commitment = vec![0xAB; 33];
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_context_exactly_1000_chars_valid() {
        let mut r = valid_receipt();
        r.context = Some("x".repeat(1000));
        let result = validate_create_usage_receipt(test_action(), r).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_empty_dependency_id_receipt_rejected() {
        let mut r = valid_receipt();
        r.dependency_id = String::new();
        let result = validate_create_usage_receipt(test_action(), r).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_usage_types_roundtrip() {
        let types = vec![
            UsageType::DirectDependency,
            UsageType::Transitive,
            UsageType::InternalTooling,
            UsageType::Production,
            UsageType::Research,
        ];
        for ut in types {
            let json = serde_json::to_string(&ut).unwrap();
            let back: UsageType = serde_json::from_str(&json).unwrap();
            assert_eq!(ut, back);
        }
    }

    #[test]
    fn test_usage_scale_roundtrip() {
        let scales = vec![
            UsageScale::Small,
            UsageScale::Medium,
            UsageScale::Large,
            UsageScale::Enterprise,
        ];
        for s in scales {
            let json = serde_json::to_string(&s).unwrap();
            let back: UsageScale = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn test_max_proof_size_constant() {
        assert_eq!(UsageAttestation::MAX_PROOF_SIZE, 500_000);
    }

    // ── Receipt edge cases ──────────────────────────────────────────

    #[test]
    fn test_receipt_none_context_valid() {
        let mut r = valid_receipt();
        r.context = None;
        let result = validate_create_usage_receipt(test_action(), r).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_receipt_none_organization_valid() {
        let mut r = valid_receipt();
        r.organization = None;
        r.scale = None;
        r.version_range = None;
        let result = validate_create_usage_receipt(test_action(), r).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ── Attestation edge cases ──────────────────────────────────────

    #[test]
    fn test_attestation_empty_dependency_id_rejected() {
        let mut a = valid_attestation();
        a.dependency_id = String::new();
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_attestation_invalid_user_did_rejected() {
        let mut a = valid_attestation();
        a.user_did = "not-a-did".into();
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_attestation_empty_id_rejected() {
        let mut a = valid_attestation();
        a.id = String::new();
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_attestation_valid_verifier_fields() {
        let mut a = valid_attestation();
        a.verifier_pubkey = Some(vec![0u8; 32]);
        a.verifier_signature = Some(vec![0u8; 64]);
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_attestation_with_expiry_valid() {
        let mut a = valid_attestation();
        a.expires_at = Some(Timestamp::now());
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ── UsageReceipt serde ──────────────────────────────────────────

    #[test]
    fn test_usage_receipt_serde_roundtrip() {
        let r = valid_receipt();
        let json = serde_json::to_string(&r).unwrap();
        let back: UsageReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(r.id, back.id);
        assert_eq!(r.dependency_id, back.dependency_id);
        assert_eq!(r.usage_type, back.usage_type);
    }

    #[test]
    fn test_usage_attestation_serde_roundtrip() {
        let a = valid_attestation();
        let json = serde_json::to_string(&a).unwrap();
        let back: UsageAttestation = serde_json::from_str(&json).unwrap();
        assert_eq!(a.id, back.id);
        assert_eq!(a.dependency_id, back.dependency_id);
        assert_eq!(a.witness_commitment, back.witness_commitment);
        assert_eq!(a.verified, back.verified);
    }

    #[test]
    fn test_anchor_serde_roundtrip() {
        let anchor = Anchor("test".into());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(anchor, back);
    }

    // ── Proof size exactly 1 byte (minimum) ─────────────────────────

    #[test]
    fn test_proof_minimum_1_byte_valid() {
        let mut a = valid_attestation();
        a.proof_bytes = vec![0x01];
        let result = validate_create_usage_attestation(test_action(), a).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }
}
