// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Housing Bridge Integrity Zome
//! Entry types for cross-hApp queries and events.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Type of housing query for cross-hApp integration
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum HousingQueryType {
    MemberVerification,
    UnitAvailability,
    EquityStatus,
    MaintenanceHistory,
    AffordabilityCheck,
    LeaseStatus,
    GovernanceStatus,
}

/// A cross-hApp query about housing
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HousingQuery {
    pub query_type: HousingQueryType,
    pub requester: AgentPubKey,
    pub target_agent: Option<AgentPubKey>,
    pub target_hash: Option<ActionHash>,
    pub parameters: String,
    pub response: Option<String>,
    pub queried_at: Timestamp,
    pub responded_at: Option<Timestamp>,
}

/// Type of housing event for broadcasting
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum HousingEventType {
    MemberJoined,
    MemberLeft,
    UnitAvailable,
    UnitOccupied,
    MaintenanceEmergency,
    ResolutionPassed,
    ElectionCompleted,
    LeaseTransferred,
    EquityMilestone,
    AffordabilityReportPublished,
}

/// A housing event broadcast to other hApps
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HousingEvent {
    pub event_type: HousingEventType,
    pub source_agent: AgentPubKey,
    pub data: String,
    pub occurred_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    HousingQuery(HousingQuery),
    HousingEvent(HousingEvent),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All queries anchor
    AllQueries,
    /// All events anchor
    AllEvents,
    /// Event type index
    EventTypeToEvent,
    /// Agent to their queries
    AgentToQuery,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::HousingQuery(query) => validate_create_query(action, query),
                EntryTypes::HousingEvent(event) => validate_create_event(action, event),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::HousingQuery(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::HousingEvent(_) => Ok(ValidateCallbackResult::Invalid(
                    "Events are immutable".into(),
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
            LinkTypes::EventTypeToEvent => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToQuery => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_query(
    _action: Create,
    query: HousingQuery,
) -> ExternResult<ValidateCallbackResult> {
    if query.parameters.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Query parameters must be at most 4096 characters".into(),
        ));
    }
    if query.response.is_some() {
        return Ok(ValidateCallbackResult::Invalid(
            "New queries should not have a response".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_event(
    _action: Create,
    event: HousingEvent,
) -> ExternResult<ValidateCallbackResult> {
    if event.data.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Event data cannot be empty".into(),
        ));
    }
    if event.data.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Event data must be at most 8192 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
