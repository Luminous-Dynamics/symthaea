//! Commons Bridge Integrity Zome
//!
//! Unified bridge for cross-domain integration within the Commons cluster.
//! Replaces the 5 separate bridge zomes from property, housing, care,
//! mutualaid, and water hApps.

use hdi::prelude::*;
use commons_types::{CommonsQuery, CommonsEvent};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A cross-domain query stored on the DHT for auditability
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StoredQuery {
    pub domain: String,
    pub query_type: String,
    pub requester: AgentPubKey,
    pub params: String,
    pub result: Option<String>,
    pub created_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
    pub success: Option<bool>,
}

impl From<CommonsQuery> for StoredQuery {
    fn from(q: CommonsQuery) -> Self {
        Self {
            domain: q.domain,
            query_type: q.query_type,
            requester: q.requester,
            params: q.params,
            result: q.result,
            created_at: q.created_at,
            resolved_at: q.resolved_at,
            success: q.success,
        }
    }
}

/// A cross-domain event stored on the DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StoredEvent {
    pub domain: String,
    pub event_type: String,
    pub source_agent: AgentPubKey,
    pub payload: String,
    pub created_at: Timestamp,
    pub related_hashes: Vec<String>,
}

impl From<CommonsEvent> for StoredEvent {
    fn from(e: CommonsEvent) -> Self {
        Self {
            domain: e.domain,
            event_type: e.event_type,
            source_agent: e.source_agent,
            payload: e.payload,
            created_at: e.created_at,
            related_hashes: e.related_hashes,
        }
    }
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Query(StoredQuery),
    Event(StoredEvent),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All queries anchor
    AllQueries,
    /// Agent to their queries
    AgentToQuery,
    /// Domain to queries
    DomainToQuery,
    /// All events anchor
    AllEvents,
    /// Event type to events
    EventTypeToEvent,
    /// Agent to events they triggered
    AgentToEvent,
    /// Domain to events
    DomainToEvent,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action: _ } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Query(query) => validate_query(&query),
                EntryTypes::Event(event) => validate_event(&event),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

const VALID_DOMAINS: &[&str] = &["property", "housing", "care", "mutualaid", "water"];

fn validate_query(query: &StoredQuery) -> ExternResult<ValidateCallbackResult> {
    if !VALID_DOMAINS.contains(&query.domain.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Invalid domain '{}'. Must be one of: {:?}", query.domain, VALID_DOMAINS),
        ));
    }
    if query.params.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Parameters must be 8192 characters or fewer".into(),
        ));
    }
    if !query.params.is_empty() {
        if serde_json::from_str::<serde_json::Value>(&query.params).is_err() {
            return Ok(ValidateCallbackResult::Invalid(
                "Parameters must be valid JSON".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_event(event: &StoredEvent) -> ExternResult<ValidateCallbackResult> {
    if !VALID_DOMAINS.contains(&event.domain.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Invalid domain '{}'. Must be one of: {:?}", event.domain, VALID_DOMAINS),
        ));
    }
    if event.payload.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Payload must be 8192 characters or fewer".into(),
        ));
    }
    if event.related_hashes.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 20 related hashes".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
