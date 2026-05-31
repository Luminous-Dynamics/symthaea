// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Care Bridge Integrity Zome
//! Defines entry types for cross-hApp integration queries and events.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Type of cross-hApp care query
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CareQueryType {
    /// Check if an agent has care credentials
    CredentialCheck,
    /// Query available care providers in an area
    ProviderSearch,
    /// Request care plan coordination across hApps
    PlanCoordination,
    /// Query health-related needs from the health hApp
    HealthNeeds,
    /// Verify identity through the identity hApp
    IdentityVerification,
    /// Check governance standing
    GovernanceStanding,
}

/// A cross-hApp query record for auditability
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareQuery {
    /// Type of query
    pub query_type: CareQueryType,
    /// The agent initiating the query
    pub requester: AgentPubKey,
    /// Target hApp or zome
    pub target_happ: String,
    /// Query parameters (JSON)
    pub parameters: String,
    /// Query result (JSON, filled after response)
    pub result: Option<String>,
    /// When the query was made
    pub created_at: Timestamp,
    /// When the result was received
    pub resolved_at: Option<Timestamp>,
    /// Whether the query succeeded
    pub success: Option<bool>,
}

/// Type of care event broadcast
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CareEventType {
    /// A new care plan was created
    PlanCreated,
    /// A care plan was updated
    PlanUpdated,
    /// A care session was logged
    SessionLogged,
    /// A new provider joined the network
    ProviderJoined,
    /// A credential was verified
    CredentialVerified,
    /// An urgent care request was posted
    UrgentRequest,
    /// A care circle was formed
    CircleFormed,
    /// A time exchange was completed
    ExchangeCompleted,
}

/// A care event broadcast to other hApps
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareEvent {
    /// Type of event
    pub event_type: CareEventType,
    /// Agent that triggered the event
    pub source_agent: AgentPubKey,
    /// Event payload (JSON)
    pub payload: String,
    /// When the event occurred
    pub created_at: Timestamp,
    /// Related entry hashes for cross-referencing
    pub related_hashes: Vec<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CareQuery(CareQuery),
    CareEvent(CareEvent),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All queries anchor
    AllQueries,
    /// Agent to their queries
    AgentToQuery,
    /// All events anchor
    AllEvents,
    /// Event type to events
    EventTypeToEvent,
    /// Agent to events they triggered
    AgentToEvent,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareQuery(query) => validate_create_query(action, query),
                EntryTypes::CareEvent(event) => validate_create_event(action, event),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareQuery(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareEvent(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
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
    query: CareQuery,
) -> ExternResult<ValidateCallbackResult> {
    if query.target_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Target hApp cannot be empty".into(),
        ));
    }
    if query.target_happ.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Target hApp must be 256 characters or fewer".into(),
        ));
    }
    if query.parameters.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Parameters must be 8192 characters or fewer".into(),
        ));
    }
    if !query.parameters.is_empty() {
        if serde_json::from_str::<serde_json::Value>(&query.parameters).is_err() {
            return Ok(ValidateCallbackResult::Invalid(
                "Parameters must be valid JSON".into(),
            ));
        }
    }
    if let Some(ref result) = query.result {
        if result.len() > 8192 {
            return Ok(ValidateCallbackResult::Invalid(
                "Result must be 8192 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_event(
    _action: Create,
    event: CareEvent,
) -> ExternResult<ValidateCallbackResult> {
    if event.payload.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Payload must be 8192 characters or fewer".into(),
        ));
    }
    if !event.payload.is_empty() {
        if serde_json::from_str::<serde_json::Value>(&event.payload).is_err() {
            return Ok(ValidateCallbackResult::Invalid(
                "Payload must be valid JSON".into(),
            ));
        }
    }
    if event.related_hashes.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 20 related hashes".into(),
        ));
    }
    for hash in &event.related_hashes {
        if hash.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each related hash must be 256 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}
