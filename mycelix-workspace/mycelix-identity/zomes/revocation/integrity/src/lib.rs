// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Revocation Registry Integrity Zome
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
                EntryTypes::RevocationList(list) => {
                    validate_create_revocation_list(EntryCreationAction::Create(action), list)
                }
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
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::CredentialToRevocation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::IssuerToRevocation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::IssuerToRevocationList => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
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
    _action: Update,
    entry: RevocationEntry,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Basic validation - cannot un-revoke once permanently revoked
    // Note: Full validation would require fetching original entry

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
    _action: Update,
    list: RevocationList,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Validate issuer is a DID
    if !list.issuer.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be a valid DID".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
