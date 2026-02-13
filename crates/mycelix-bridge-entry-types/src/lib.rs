//! Mycelix Bridge Entry Types — Shared DHT entry structs for cluster bridges
//!
//! Provides the canonical entry types stored on the DHT by both the Commons
//! and Civic bridge integrity zomes. Each integrity zome wraps these in its
//! own `#[hdk_entry_types]` enum (required by Holochain), but the underlying
//! struct definitions live here — single source of truth.
//!
//! Also provides domain-independent validation helpers.

use hdi::prelude::*;

// ============================================================================
// Entry types
// ============================================================================

/// Cross-domain query entry stored on the DHT.
///
/// Used by both commons-bridge and civic-bridge integrity zomes
/// as a variant in their `EntryTypes` enum.
///
/// We derive `SerializedBytes` + manually implement `TryFrom<&Entry>` instead
/// of using `#[hdk_entry_helper]`. The helper macro generates code referencing
/// `From<&T> for SerializedBytes` which is incompatible with `derive(SerializedBytes)`
/// in HDI 0.7 — a known macro mismatch. Our manual impl achieves the same result.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug, SerializedBytes)]
pub struct BridgeQueryEntry {
    /// Domain within the cluster (e.g., "property", "justice")
    pub domain: String,
    /// Query type / function name
    pub query_type: String,
    /// The agent initiating the query
    pub requester: AgentPubKey,
    /// Query parameters (JSON string)
    pub params: String,
    /// Result payload (filled after resolution)
    pub result: Option<String>,
    /// When the query was created
    pub created_at: Timestamp,
    /// When the query was resolved
    pub resolved_at: Option<Timestamp>,
    /// Whether the query succeeded
    pub success: Option<bool>,
}

impl TryFrom<&Entry> for BridgeQueryEntry {
    type Error = WasmError;
    fn try_from(entry: &Entry) -> Result<Self, Self::Error> {
        match entry {
            Entry::App(bytes) => {
                let sb = SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()));
                <Self as TryFrom<SerializedBytes>>::try_from(sb)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))
            }
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                "Not an app entry".into(),
            ))),
        }
    }
}

/// Cross-domain event entry stored on the DHT.
///
/// Used by both commons-bridge and civic-bridge integrity zomes.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug, SerializedBytes)]
pub struct BridgeEventEntry {
    /// Domain within the cluster (e.g., "housing", "emergency")
    pub domain: String,
    /// Event type identifier
    pub event_type: String,
    /// Agent that triggered the event
    pub source_agent: AgentPubKey,
    /// Event payload (JSON string)
    pub payload: String,
    /// When the event occurred
    pub created_at: Timestamp,
    /// Related entry hashes for cross-referencing
    pub related_hashes: Vec<String>,
}

impl TryFrom<&Entry> for BridgeEventEntry {
    type Error = WasmError;
    fn try_from(entry: &Entry) -> Result<Self, Self::Error> {
        match entry {
            Entry::App(bytes) => {
                let sb = SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()));
                <Self as TryFrom<SerializedBytes>>::try_from(sb)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))
            }
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                "Not an app entry".into(),
            ))),
        }
    }
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate a bridge query entry against a domain allowlist.
///
/// Returns `Ok(())` if valid, or `Err(reason)` if invalid.
/// Each integrity zome calls this with its own `VALID_DOMAINS` list.
pub fn validate_query_fields(
    query: &BridgeQueryEntry,
    valid_domains: &[&str],
) -> Result<(), String> {
    if !valid_domains.contains(&query.domain.as_str()) {
        return Err(format!(
            "Invalid domain '{}'. Must be one of: {:?}",
            query.domain, valid_domains
        ));
    }
    if query.params.len() > 8192 {
        return Err("Parameters must be 8192 characters or fewer".into());
    }
    if !query.params.is_empty() {
        if serde_json::from_str::<serde_json::Value>(&query.params).is_err() {
            return Err("Parameters must be valid JSON".into());
        }
    }
    Ok(())
}

/// Validate a bridge event entry against a domain allowlist.
///
/// Returns `Ok(())` if valid, or `Err(reason)` if invalid.
pub fn validate_event_fields(
    event: &BridgeEventEntry,
    valid_domains: &[&str],
) -> Result<(), String> {
    if !valid_domains.contains(&event.domain.as_str()) {
        return Err(format!(
            "Invalid domain '{}'. Must be one of: {:?}",
            event.domain, valid_domains
        ));
    }
    if event.payload.len() > 8192 {
        return Err("Payload must be 8192 characters or fewer".into());
    }
    if event.related_hashes.len() > 20 {
        return Err("Cannot have more than 20 related hashes".into());
    }
    Ok(())
}
