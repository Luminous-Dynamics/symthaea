// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Emergency Bridge Integrity Zome
//! Cross-hApp integration types for emergency coordination

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A cross-hApp query for emergency data
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencyQuery {
    pub query_id: String,
    pub source_happ: String,
    pub query_type: QueryType,
    pub parameters: String,
    pub requested_by: AgentPubKey,
    pub requested_at: Timestamp,
    pub response: Option<String>,
    pub responded_at: Option<Timestamp>,
}

/// Types of cross-hApp queries
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum QueryType {
    ActiveDisasters,
    ResourceAvailability,
    ShelterCapacity,
    PersonnelStatus,
    HealthRecords,
    IdentityVerification,
    SupplyChainStatus,
}

/// An event broadcast to other hApps
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencyEvent {
    pub event_id: String,
    pub event_type: EventType,
    pub disaster_hash: Option<ActionHash>,
    pub payload: String,
    pub emitted_by: AgentPubKey,
    pub emitted_at: Timestamp,
    pub target_happs: Vec<String>,
}

/// Types of emergency events
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EventType {
    DisasterDeclared,
    DisasterClosed,
    EvacuationOrdered,
    ResourceRequested,
    MassCasualtyDeclared,
    ShelterOpened,
    ShelterFull,
    AllClear,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    EmergencyQuery(EmergencyQuery),
    EmergencyEvent(EmergencyEvent),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllQueries,
    AllEvents,
    DisasterToEvent,
    HappToQuery,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencyQuery(query) => validate_create_query(action, query),
                EntryTypes::EmergencyEvent(event) => validate_create_event(action, event),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencyQuery(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencyEvent(_) => Ok(ValidateCallbackResult::Invalid(
                    "Emergency events are immutable".into(),
                )),
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
            LinkTypes::AllQueries => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllEvents => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DisasterToEvent => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HappToQuery => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_query(
    _action: Create,
    query: EmergencyQuery,
) -> ExternResult<ValidateCallbackResult> {
    if query.query_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Query ID cannot be empty".into(),
        ));
    }
    if query.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp cannot be empty".into(),
        ));
    }
    if query.parameters.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Query parameters cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_event(
    _action: Create,
    event: EmergencyEvent,
) -> ExternResult<ValidateCallbackResult> {
    if event.event_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Event ID cannot be empty".into(),
        ));
    }
    if event.payload.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Event payload cannot be empty".into(),
        ));
    }
    if event.target_happs.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Event must target at least one hApp".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
