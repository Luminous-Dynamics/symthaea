//! Commons Bridge Integrity Zome
//!
//! Unified bridge for cross-domain integration within the Commons cluster.
//! Replaces the 5 separate bridge zomes from property, housing, care,
//! mutualaid, and water hApps.
//!
//! Entry struct definitions come from `mycelix-bridge-entry-types` (shared
//! with the Civic bridge). The `EntryTypes` enum and validation are local.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{
    BridgeQueryEntry, BridgeEventEntry,
    validate_query_fields, validate_event_fields,
};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Backward-compatible type alias — code that references `StoredQuery` still compiles.
pub type StoredQuery = BridgeQueryEntry;

/// Backward-compatible type alias.
pub type StoredEvent = BridgeEventEntry;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Query(BridgeQueryEntry),
    Event(BridgeEventEntry),
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
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, action: _ }) => match app_entry {
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            EntryTypes::Query(query) => validate_query(&query),
            EntryTypes::Event(event) => validate_event(&event),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

const VALID_DOMAINS: &[&str] = &["property", "housing", "care", "mutualaid", "water"];

fn validate_query(query: &BridgeQueryEntry) -> ExternResult<ValidateCallbackResult> {
    match validate_query_fields(query, VALID_DOMAINS) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

fn validate_event(event: &BridgeEventEntry) -> ExternResult<ValidateCallbackResult> {
    match validate_event_fields(event, VALID_DOMAINS) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}
