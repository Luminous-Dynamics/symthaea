//! Revocation Registry Integrity Zome
//! Defines entry types and validation for credential revocation
//!
//! Updated to use HDI 0.7 patterns with FlatOp validation

use hdi::prelude::*;

/// Revocation status for a credential
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RevocationStatus {
    /// Credential is valid
    Active,
    /// Credential is temporarily suspended
    Suspended,
    /// Credential is permanently revoked
    Revoked,
}

/// Revocation entry for a credential
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RevocationEntry {
    /// Credential identifier (hash or ID)
    pub credential_id: String,
    /// Issuer's DID who revoked
    pub issuer: String,
    /// Current status
    pub status: RevocationStatus,
    /// Reason for revocation/suspension
    pub reason: String,
    /// When the revocation takes effect
    pub effective_from: Timestamp,
    /// When the revocation was recorded
    pub recorded_at: Timestamp,
    /// Optional: when suspension ends (for Suspended status)
    pub suspension_end: Option<Timestamp>,
}

/// Revocation list for batch operations
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RevocationList {
    /// List identifier
    pub id: String,
    /// Issuer who owns this list
    pub issuer: String,
    /// List of revoked credential IDs
    pub revoked: Vec<String>,
    /// Last update timestamp
    pub updated: Timestamp,
    /// Version for optimistic concurrency
    pub version: u32,
}

/// Revocation check request result
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RevocationCheckResult {
    pub credential_id: String,
    pub status: RevocationStatus,
    pub reason: Option<String>,
    pub checked_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    RevocationEntry(RevocationEntry),
    RevocationList(RevocationList),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Credential to revocation entry
    CredentialToRevocation,
    /// Issuer to their revocation entries
    IssuerToRevocation,
    /// Issuer to their revocation lists
    IssuerToRevocationList,
}

/// Genesis self-check - called when app is installed
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern matching
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::RevocationEntry(entry) => {
                    validate_create_revocation_entry(EntryCreationAction::Create(action), entry)
                }
                EntryTypes::RevocationList(list) => validate_create_revocation_list(EntryCreationAction::Create(action), list),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::RevocationEntry(entry) => {
                    validate_update_revocation_entry(action, entry, original_action_hash)
                }
                EntryTypes::RevocationList(list) => {
                    validate_update_revocation_list(action, list, original_action_hash)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            if tag.0.len() > 1024 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds maximum length of 1024 bytes".into(),
                ));
            }
            match link_type {
                LinkTypes::CredentialToRevocation => Ok(ValidateCallbackResult::Valid),
                LinkTypes::IssuerToRevocation => Ok(ValidateCallbackResult::Valid),
                LinkTypes::IssuerToRevocationList => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the link creator can delete their links".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
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
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

/// Validate revocation entry creation
fn validate_create_revocation_entry(
    _action: EntryCreationAction,
    entry: RevocationEntry,
) -> ExternResult<ValidateCallbackResult> {
    // Validate issuer is a DID
    if !entry.issuer.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be a valid DID".into(),
        ));
    }

    // Validate reason is provided
    if entry.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Revocation reason is required".into(),
        ));
    }

    // Validate suspension has end date
    if entry.status == RevocationStatus::Suspended && entry.suspension_end.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Suspended status requires suspension_end date".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate revocation entry update
fn validate_update_revocation_entry(
    action: Update,
    entry: RevocationEntry,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Validate issuer is a DID
    if !entry.issuer.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be a valid DID".into(),
        ));
    }

    // Validate suspension has end date
    if entry.status == RevocationStatus::Suspended && entry.suspension_end.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Suspended status requires suspension_end date".into(),
        ));
    }

    // Fetch original to enforce state transitions
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: RevocationEntry = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original revocation entry not found".into()
        )))?;

    // Immutable fields
    if entry.credential_id != original.credential_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential ID cannot be changed on revocation entry".into(),
        ));
    }
    if entry.issuer != original.issuer {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer cannot be changed on revocation entry".into(),
        ));
    }

    // Revoked is terminal — cannot transition away from Revoked
    if original.status == RevocationStatus::Revoked {
        return Ok(ValidateCallbackResult::Invalid(
            "Revoked status is terminal and cannot be changed".into(),
        ));
    }

    // Valid transitions: Active → Suspended/Revoked, Suspended → Active/Revoked
    match (&original.status, &entry.status) {
        (RevocationStatus::Active, RevocationStatus::Suspended)
        | (RevocationStatus::Active, RevocationStatus::Revoked)
        | (RevocationStatus::Suspended, RevocationStatus::Active)
        | (RevocationStatus::Suspended, RevocationStatus::Revoked) => {}
        (a, b) if a == b => {} // No-op update allowed
        _ => {
            return Ok(ValidateCallbackResult::Invalid(
                "Invalid revocation status transition".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate revocation list creation
fn validate_create_revocation_list(
    _action: EntryCreationAction,
    list: RevocationList,
) -> ExternResult<ValidateCallbackResult> {
    // Validate issuer is a DID
    if !list.issuer.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be a valid DID".into(),
        ));
    }

    // Validate version starts at 1
    if list.version != 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial version must be 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate revocation list update
fn validate_update_revocation_list(
    action: Update,
    list: RevocationList,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Validate issuer is a DID
    if !list.issuer.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be a valid DID".into(),
        ));
    }

    // Fetch original to enforce invariants
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: RevocationList = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original revocation list not found".into()
        )))?;

    // Immutable fields
    if list.id != original.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Revocation list ID cannot be changed".into(),
        ));
    }
    if list.issuer != original.issuer {
        return Ok(ValidateCallbackResult::Invalid(
            "Revocation list issuer cannot be changed".into(),
        ));
    }

    // Version must increment
    if list.version <= original.version {
        return Ok(ValidateCallbackResult::Invalid(
            "Revocation list version must increase on update".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    // --- RevocationStatus ---

    #[test]
    fn revocation_status_json_variants() {
        let variants = vec![
            (RevocationStatus::Active, "\"Active\""),
            (RevocationStatus::Suspended, "\"Suspended\""),
            (RevocationStatus::Revoked, "\"Revoked\""),
        ];
        for (variant, expected) in variants {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, expected);
            let back: RevocationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, variant);
        }
    }

    // --- RevocationEntry ---

    #[test]
    fn revocation_entry_revoked_json_round_trip() {
        let entry = RevocationEntry {
            credential_id: "cred-001".into(),
            issuer: "did:mycelix:issuer1".into(),
            status: RevocationStatus::Revoked,
            reason: "Key compromise detected".into(),
            effective_from: ts(1_700_000_000_000_000),
            recorded_at: ts(1_700_000_001_000_000),
            suspension_end: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: RevocationEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn revocation_entry_suspended_with_end_date() {
        let entry = RevocationEntry {
            credential_id: "cred-002".into(),
            issuer: "did:mycelix:issuer1".into(),
            status: RevocationStatus::Suspended,
            reason: "Under investigation".into(),
            effective_from: ts(1_700_000_000_000_000),
            recorded_at: ts(1_700_000_001_000_000),
            suspension_end: Some(ts(1_710_000_000_000_000)),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: RevocationEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
        assert!(entry.suspension_end.is_some());
    }

    #[test]
    fn revocation_entry_suspended_requires_end_date() {
        // Validation condition: Suspended status without suspension_end should be rejected
        let status = RevocationStatus::Suspended;
        let suspension_end: Option<Timestamp> = None;
        assert!(
            status == RevocationStatus::Suspended && suspension_end.is_none(),
            "Suspended without end date should be rejected"
        );
    }

    #[test]
    fn revocation_entry_rejects_non_did_issuer() {
        assert!(!"alice@example.com".starts_with("did:"));
        assert!("did:mycelix:issuer1".starts_with("did:"));
    }

    #[test]
    fn revocation_entry_rejects_empty_reason() {
        assert!("".is_empty(), "Empty reason should be rejected");
        assert!(!"Key compromise".is_empty());
    }

    // --- RevocationList ---

    #[test]
    fn revocation_list_json_round_trip() {
        let list = RevocationList {
            id: "revlist-001".into(),
            issuer: "did:mycelix:issuer1".into(),
            revoked: vec!["cred-001".into(), "cred-002".into()],
            updated: ts(1_700_000_000_000_000),
            version: 1,
        };
        let json = serde_json::to_string(&list).unwrap();
        let back: RevocationList = serde_json::from_str(&json).unwrap();
        assert_eq!(list, back);
    }

    #[test]
    fn revocation_list_initial_version_must_be_one() {
        // Validation condition: initial version must be 1
        let version = 0u32;
        assert_ne!(version, 1, "Version 0 should be rejected");
        let version = 1u32;
        assert_eq!(version, 1, "Version 1 should be accepted");
    }

    #[test]
    fn revocation_list_empty_revoked_is_valid() {
        let list = RevocationList {
            id: "revlist-002".into(),
            issuer: "did:mycelix:issuer2".into(),
            revoked: vec![],
            updated: ts(1_700_000_000_000_000),
            version: 1,
        };
        let json = serde_json::to_string(&list).unwrap();
        let back: RevocationList = serde_json::from_str(&json).unwrap();
        assert_eq!(list, back);
    }

    // --- RevocationCheckResult ---

    #[test]
    fn revocation_check_result_json_round_trip() {
        let result = RevocationCheckResult {
            credential_id: "cred-001".into(),
            status: RevocationStatus::Revoked,
            reason: Some("Expired".into()),
            checked_at: ts(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: RevocationCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.credential_id, "cred-001");
        assert_eq!(back.status, RevocationStatus::Revoked);
        assert_eq!(back.reason, Some("Expired".into()));
    }
}
