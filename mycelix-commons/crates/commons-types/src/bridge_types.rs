//! Bridge types for cross-domain communication within the Commons cluster
//!
//! These types are used by the unified commons-bridge zome to route
//! queries and events between the 5 domain groups.

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

/// A cross-domain query within the Commons cluster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CommonsQuery {
    /// Domain: "property", "housing", "care", "mutualaid", "water"
    pub domain: String,
    /// Query type within the domain
    pub query_type: String,
    /// The agent initiating the query
    pub requester: AgentPubKey,
    /// Query parameters (JSON)
    pub params: String,
    /// Query result (JSON, filled after response)
    pub result: Option<String>,
    /// When the query was made
    pub created_at: Timestamp,
    /// When the result was received
    pub resolved_at: Option<Timestamp>,
    /// Whether the query succeeded
    pub success: Option<bool>,
}

/// A cross-domain event broadcast within the Commons cluster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CommonsEvent {
    /// Domain: "property", "housing", "care", "mutualaid", "water"
    pub domain: String,
    /// Event type within the domain
    pub event_type: String,
    /// Agent that triggered the event
    pub source_agent: AgentPubKey,
    /// Event payload (JSON)
    pub payload: String,
    /// When the event occurred
    pub created_at: Timestamp,
    /// Related entry hashes for cross-referencing
    pub related_hashes: Vec<String>,
}

/// Health status for the bridge
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeHealthStatus {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}
