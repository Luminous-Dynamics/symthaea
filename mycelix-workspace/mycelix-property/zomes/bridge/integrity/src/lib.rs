// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Property Bridge Integrity Zome
//!
//! Entry types for cross-hApp property queries, ownership verification, and collateral pledging.

use hdi::prelude::*;

/// Property ownership query from another hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OwnershipQuery {
    pub id: String,
    pub property_id: String,
    pub source_happ: String,
    pub purpose: OwnershipQueryPurpose,
    pub queried_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum OwnershipQueryPurpose {
    CollateralVerification,
    TransferVerification,
    DisputeResolution,
    TaxAssessment,
}

/// Property ownership verification result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OwnershipResult {
    pub id: String,
    pub property_id: String,
    pub owner_did: String,
    pub ownership_type: OwnershipType,
    pub share_percentage: f64,
    pub encumbrances: Vec<Encumbrance>,
    pub verified_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum OwnershipType {
    Sole,
    Joint,
    Fractional,
    Corporate,
    Trust,
    Commons,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Encumbrance {
    pub encumbrance_type: EncumbranceType,
    pub holder_did: String,
    pub amount: Option<u64>,
    pub expires_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EncumbranceType {
    Mortgage,
    Lien,
    Easement,
    Lease,
    CollateralPledge,
}

/// Cross-hApp collateral pledge
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CollateralPledge {
    pub id: String,
    pub property_id: String,
    pub owner_did: String,
    pub pledged_to_happ: String,
    pub pledged_for_loan: String,
    pub value: u64,
    pub currency: String,
    pub status: PledgeStatus,
    pub pledged_at: Timestamp,
    pub released_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PledgeStatus {
    Active,
    Released,
    Foreclosed,
}

/// Property event for broadcasting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PropertyBridgeEvent {
    pub id: String,
    pub event_type: PropertyEventType,
    pub property_id: String,
    pub subject_did: String,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PropertyEventType {
    OwnershipTransferred,
    PropertyRegistered,
    CollateralPledged,
    CollateralReleased,
    EncumbranceAdded,
    EncumbranceRemoved,
    DisputeFiled,
}

/// Anchor entry for deterministic link bases from strings
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    OwnershipQuery(OwnershipQuery),
    OwnershipResult(OwnershipResult),
    CollateralPledge(CollateralPledge),
    PropertyBridgeEvent(PropertyBridgeEvent),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    PropertyToOwnership,
    DidToProperties,
    ActivePledges,
    RecentEvents,
}

#[hdk_extern]
pub fn validate_create_entry_ownership_query(
    _action: EntryCreationAction,
    query: OwnershipQuery,
) -> ExternResult<ValidateCallbackResult> {
    if query.property_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Property ID required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate_create_entry_ownership_result(
    _action: EntryCreationAction,
    result: OwnershipResult,
) -> ExternResult<ValidateCallbackResult> {
    if result.share_percentage < 0.0 || result.share_percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Share must be 0-100%".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate_create_entry_collateral_pledge(
    _action: EntryCreationAction,
    pledge: CollateralPledge,
) -> ExternResult<ValidateCallbackResult> {
    if !pledge.owner_did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must have valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate_create_entry_property_bridge_event(
    _action: EntryCreationAction,
    event: PropertyBridgeEvent,
) -> ExternResult<ValidateCallbackResult> {
    if event.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate_create_entry_anchor(
    _action: EntryCreationAction,
    _anchor: Anchor,
) -> ExternResult<ValidateCallbackResult> {
    // Anchors are always valid - they're just string wrappers for link bases
    Ok(ValidateCallbackResult::Valid)
}
