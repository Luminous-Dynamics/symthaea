#![deny(unsafe_code)]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mail Bridge Integrity Zome
//!
//! Entry types for cross-cluster dispatch in the unified Mycelix hApp.

use hdi::prelude::*;

/// Cross-cluster query entry for audit trail
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MailBridgeQuery {
    /// Domain being queried
    pub domain: String,
    /// Query type
    pub query_type: String,
    /// Requester agent
    pub requester: AgentPubKey,
    /// Serialized parameters
    pub params: Vec<u8>,
    /// Result (filled after execution)
    pub result: Option<Vec<u8>>,
    /// When queried
    pub queried_at: Timestamp,
    /// Whether successful
    pub success: bool,
}

/// Cross-cluster event notification
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MailBridgeEvent {
    /// Event type
    pub event_type: String,
    /// Source domain
    pub source: String,
    /// Serialized payload
    pub payload: Vec<u8>,
    /// Timestamp
    pub timestamp: Timestamp,
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Agent to their bridge queries
    AgentToQueries,
    /// Anchor to events
    AnchorToEvents,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 1)]
    MailBridgeQuery(MailBridgeQuery),
    #[entry_type(required_validations = 1)]
    MailBridgeEvent(MailBridgeEvent),
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
                EntryTypes::MailBridgeQuery(query) => {
                    if query.requester != action.author {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Requester must match author".to_string(),
                        ));
                    }
                    if query.domain.is_empty() || query.query_type.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Domain and query_type cannot be empty".to_string(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
