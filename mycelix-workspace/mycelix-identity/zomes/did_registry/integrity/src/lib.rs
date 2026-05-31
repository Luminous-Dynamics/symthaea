// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! DID Registry Integrity Zome
//! Defines entry types and validation for DID:mycelix identifiers
//!
//! Updated to use HDI 0.7 patterns with FlatOp validation

use hdi::prelude::*;

/// DID Document entry type
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DidDocument {
    /// The DID identifier (did:mycelix:<agent_pub_key>)
    pub id: String,
    /// Controller of this DID (usually self)
    pub controller: AgentPubKey,
    /// Verification methods (public keys)
    pub verification_method: Vec<VerificationMethod>,
    /// Authentication methods
    pub authentication: Vec<String>,
    /// Service endpoints
    pub service: Vec<ServiceEndpoint>,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last update timestamp
    pub updated: Timestamp,
    /// Version number for updates
    pub version: u32,
}

/// Verification method for cryptographic operations
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct VerificationMethod {
    pub id: String,
    pub type_: String,
    pub controller: String,
    pub public_key_multibase: String,
}

/// Service endpoint for discovery
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub id: String,
    pub type_: String,
    pub service_endpoint: String,
}

/// DID Deactivation record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DidDeactivation {
    pub did: String,
    pub reason: String,
    pub deactivated_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    DidDocument(DidDocument),
    DidDeactivation(DidDeactivation),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToDid,
    DidToVerificationMethod,
    DidToService,
    DidHistory,
    DidToDeactivation,
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
                EntryTypes::DidDocument(did_doc) => {
                    validate_create_did_document(EntryCreationAction::Create(action), did_doc)
                }
                EntryTypes::DidDeactivation(deactivation) => validate_create_did_deactivation(
                    EntryCreationAction::Create(action),
                    deactivation,
                ),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::DidDocument(did_doc) => validate_update_did_document(action, did_doc),
                EntryTypes::DidDeactivation(_) => Ok(ValidateCallbackResult::Invalid(
                    "Deactivation records cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::AgentToDid => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DidToVerificationMethod => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DidToService => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DidHistory => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DidToDeactivation => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => {
            // Links can be deleted
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate DID document creation
fn validate_create_did_document(
    action: EntryCreationAction,
    did_doc: DidDocument,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !did_doc.id.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate controller matches author
    let author = action.author();
    if did_doc.controller != *author {
        return Ok(ValidateCallbackResult::Invalid(
            "DID controller must be the author".into(),
        ));
    }

    // Validate at least one verification method
    if did_doc.verification_method.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must have at least one verification method".into(),
        ));
    }

    // Validate version starts at 1
    if did_doc.version != 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial DID version must be 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate DID document update
fn validate_update_did_document(
    action: Update,
    did_doc: DidDocument,
) -> ExternResult<ValidateCallbackResult> {
    // Validate author is controller
    if did_doc.controller != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Only controller can update DID".into(),
        ));
    }

    // Version validation would require fetching original - skip for now
    // More complex validation can be added later

    Ok(ValidateCallbackResult::Valid)
}

/// Validate DID deactivation creation
fn validate_create_did_deactivation(
    _action: EntryCreationAction,
    deactivation: DidDeactivation,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !deactivation.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate reason provided
    if deactivation.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Deactivation reason is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
