//! Civic Bridge Integrity Zome
//!
//! Unified bridge for cross-domain integration within the Civic cluster.
//! Replaces the 3 separate bridge zomes from justice, emergency, and media.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A cross-domain query stored on the DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CivicQueryEntry {
    pub domain: String,
    pub query_type: String,
    pub requester: AgentPubKey,
    pub params: String,
    pub result: Option<String>,
    pub created_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
    pub success: Option<bool>,
}

/// A cross-domain event stored on the DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CivicEventEntry {
    pub domain: String,
    pub event_type: String,
    pub source_agent: AgentPubKey,
    pub payload: String,
    pub created_at: Timestamp,
    pub related_hashes: Vec<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Query(CivicQueryEntry),
    Event(CivicEventEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllQueries,
    AgentToQuery,
    DomainToQuery,
    AllEvents,
    EventTypeToEvent,
    AgentToEvent,
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

const VALID_DOMAINS: &[&str] = &["justice", "emergency", "media"];

fn validate_query(query: &CivicQueryEntry) -> ExternResult<ValidateCallbackResult> {
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
    Ok(ValidateCallbackResult::Valid)
}

fn validate_event(event: &CivicEventEntry) -> ExternResult<ValidateCallbackResult> {
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
