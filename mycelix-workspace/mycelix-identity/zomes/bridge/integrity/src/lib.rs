// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Identity Bridge Integrity Zome
//!
//! Entry types and validation for cross-hApp identity operations.
//! Enables other hApps to query and verify identities via the Bridge protocol.
//!
//! Updated to use HDI 0.7 patterns with FlatOp validation

use hdi::prelude::*;

/// hApp registration with the identity bridge
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HappRegistration {
    /// Unique identifier for the hApp
    pub happ_id: String,
    /// Human-readable name
    pub happ_name: String,
    /// Capabilities this hApp supports
    pub capabilities: Vec<String>,
    /// MATL trust score for this hApp (0.0 - 1.0)
    pub matl_score: f64,
    /// When this hApp was registered
    pub registered_at: Timestamp,
}

/// Identity query request from another hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IdentityQuery {
    /// Query ID for tracking
    pub id: String,
    /// DID being queried
    pub did: String,
    /// Requesting hApp
    pub source_happ: String,
    /// What fields are requested (selective disclosure)
    pub requested_fields: Vec<String>,
    /// Query timestamp
    pub queried_at: Timestamp,
}

/// Identity verification result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IdentityVerification {
    /// Verification ID
    pub id: String,
    /// DID that was verified
    pub did: String,
    /// Whether the DID exists and is active
    pub is_valid: bool,
    /// Whether the DID is deactivated
    pub is_deactivated: bool,
    /// MATL reputation score (0.0 - 1.0)
    pub matl_score: f64,
    /// Total credentials issued to this DID
    pub credential_count: u32,
    /// When the DID was created
    pub did_created: Option<Timestamp>,
    /// Verification timestamp
    pub verified_at: Timestamp,
}

/// Bridge event for pub/sub
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BridgeEvent {
    /// Event ID
    pub id: String,
    /// Event type (e.g., "did_created", "did_updated", "did_deactivated")
    pub event_type: BridgeEventType,
    /// Agent/DID this event relates to
    pub subject: String,
    /// Event payload (JSON-encoded)
    pub payload: String,
    /// Source hApp
    pub source_happ: String,
    /// Event timestamp
    pub timestamp: Timestamp,
}

/// Types of bridge events
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BridgeEventType {
    /// A new DID was created
    DidCreated,
    /// A DID was updated (keys rotated, services changed)
    DidUpdated,
    /// A DID was deactivated
    DidDeactivated,
    /// A credential was issued to a DID
    CredentialIssued,
    /// A credential was revoked
    CredentialRevoked,
    /// Recovery was initiated for a DID
    RecoveryInitiated,
    /// Recovery was completed for a DID
    RecoveryCompleted,
    /// hApp registered with bridge
    HappRegistered,
    /// Custom event type
    Custom(String),
}

/// Cross-hApp reputation record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IdentityReputation {
    /// DID this reputation is for
    pub did: String,
    /// hApp this reputation came from
    pub source_happ: String,
    /// Reputation score (0.0 - 1.0)
    pub score: f64,
    /// Number of interactions in the source hApp
    pub interactions: u64,
    /// Last update timestamp
    pub last_updated: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    HappRegistration(HappRegistration),
    IdentityQuery(IdentityQuery),
    IdentityVerification(IdentityVerification),
    BridgeEvent(BridgeEvent),
    IdentityReputation(IdentityReputation),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all registered hApps
    RegisteredHapps,
    /// DID to its reputation scores from various hApps
    DidToReputations,
    /// Anchor to recent bridge events
    RecentEvents,
    /// Event type to events (for filtering)
    EventTypeToEvents,
    /// hApp to its queries
    HappToQueries,
    /// DID to queries about it
    DidToQueries,
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
                EntryTypes::HappRegistration(registration) => validate_create_happ_registration(
                    EntryCreationAction::Create(action),
                    registration,
                ),
                EntryTypes::IdentityQuery(query) => {
                    validate_create_identity_query(EntryCreationAction::Create(action), query)
                }
                EntryTypes::IdentityVerification(verification) => {
                    validate_create_identity_verification(
                        EntryCreationAction::Create(action),
                        verification,
                    )
                }
                EntryTypes::BridgeEvent(event) => {
                    validate_create_bridge_event(EntryCreationAction::Create(action), event)
                }
                EntryTypes::IdentityReputation(reputation) => validate_create_identity_reputation(
                    EntryCreationAction::Create(action),
                    reputation,
                ),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::HappRegistration(registration) => {
                    validate_update_happ_registration(action, registration)
                }
                EntryTypes::IdentityReputation(reputation) => {
                    validate_update_identity_reputation(action, reputation)
                }
                // Queries, verifications, and events are immutable
                EntryTypes::IdentityQuery(_) => Ok(ValidateCallbackResult::Invalid(
                    "Identity queries cannot be updated".into(),
                )),
                EntryTypes::IdentityVerification(_) => Ok(ValidateCallbackResult::Invalid(
                    "Identity verifications cannot be updated".into(),
                )),
                EntryTypes::BridgeEvent(_) => Ok(ValidateCallbackResult::Invalid(
                    "Bridge events cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::RegisteredHapps => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DidToReputations => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecentEvents => Ok(ValidateCallbackResult::Valid),
            LinkTypes::EventTypeToEvents => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HappToQueries => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DidToQueries => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate hApp registration creation
fn validate_create_happ_registration(
    _action: EntryCreationAction,
    registration: HappRegistration,
) -> ExternResult<ValidateCallbackResult> {
    // hApp ID must be non-empty
    if registration.happ_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "hApp ID cannot be empty".into(),
        ));
    }

    // hApp name must be non-empty
    if registration.happ_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "hApp name cannot be empty".into(),
        ));
    }

    // MATL score must be valid
    if !(0.0..=1.0).contains(&registration.matl_score) {
        return Ok(ValidateCallbackResult::Invalid(
            "MATL score must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate hApp registration update
fn validate_update_happ_registration(
    _action: Update,
    registration: HappRegistration,
) -> ExternResult<ValidateCallbackResult> {
    // MATL score must be valid
    if !(0.0..=1.0).contains(&registration.matl_score) {
        return Ok(ValidateCallbackResult::Invalid(
            "MATL score must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate identity query creation
fn validate_create_identity_query(
    _action: EntryCreationAction,
    query: IdentityQuery,
) -> ExternResult<ValidateCallbackResult> {
    // DID must be valid format
    if !query.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must use did:mycelix format".into(),
        ));
    }

    // Source hApp must be specified
    if query.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp must be specified".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate identity verification creation
fn validate_create_identity_verification(
    _action: EntryCreationAction,
    verification: IdentityVerification,
) -> ExternResult<ValidateCallbackResult> {
    // DID must be valid format
    if !verification.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must use did:mycelix format".into(),
        ));
    }

    // MATL score must be valid
    if !(0.0..=1.0).contains(&verification.matl_score) {
        return Ok(ValidateCallbackResult::Invalid(
            "MATL score must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate bridge event creation
fn validate_create_bridge_event(
    _action: EntryCreationAction,
    event: BridgeEvent,
) -> ExternResult<ValidateCallbackResult> {
    // Subject must be non-empty
    if event.subject.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Event subject cannot be empty".into(),
        ));
    }

    // Source hApp must be specified
    if event.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp must be specified".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate identity reputation creation
fn validate_create_identity_reputation(
    _action: EntryCreationAction,
    reputation: IdentityReputation,
) -> ExternResult<ValidateCallbackResult> {
    // DID must be valid format
    if !reputation.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must use did:mycelix format".into(),
        ));
    }

    // Source hApp must be specified
    if reputation.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp must be specified".into(),
        ));
    }

    // Score must be valid
    if !(0.0..=1.0).contains(&reputation.score) {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation score must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate identity reputation update
fn validate_update_identity_reputation(
    _action: Update,
    reputation: IdentityReputation,
) -> ExternResult<ValidateCallbackResult> {
    // Score must be valid
    if !(0.0..=1.0).contains(&reputation.score) {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation score must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
