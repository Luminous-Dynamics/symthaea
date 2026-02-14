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

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn make_query(domain: &str, params: &str) -> BridgeQueryEntry {
        BridgeQueryEntry {
            domain: domain.into(),
            query_type: "test_query".into(),
            requester: fake_agent(),
            params: params.into(),
            result: None,
            created_at: Timestamp::from_micros(0),
            resolved_at: None,
            success: None,
        }
    }

    fn make_event(domain: &str, payload: &str) -> BridgeEventEntry {
        BridgeEventEntry {
            domain: domain.into(),
            event_type: "test_event".into(),
            source_agent: fake_agent(),
            payload: payload.into(),
            created_at: Timestamp::from_micros(0),
            related_hashes: vec![],
        }
    }

    const DOMAINS: &[&str] = &[
        "property", "housing", "care", "justice",
        "mutualaid", "water", "food", "transport",
    ];

    // ---- validate_query_fields ----

    #[test]
    fn query_valid_domain_accepted() {
        let q = make_query("property", "{}");
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_all_valid_domains_accepted() {
        for domain in DOMAINS {
            let q = make_query(domain, "{}");
            assert!(validate_query_fields(&q, DOMAINS).is_ok(), "domain '{}' should be valid", domain);
        }
    }

    #[test]
    fn query_invalid_domain_rejected() {
        let q = make_query("invalid", "{}");
        let err = validate_query_fields(&q, DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
        assert!(err.contains("invalid"));
    }

    #[test]
    fn query_empty_domain_rejected() {
        let q = make_query("", "{}");
        assert!(validate_query_fields(&q, DOMAINS).is_err());
    }

    #[test]
    fn query_empty_params_accepted() {
        let q = make_query("property", "");
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_valid_json_params_accepted() {
        let q = make_query("property", r#"{"key":"value","nested":{"a":1}}"#);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_invalid_json_params_rejected() {
        let q = make_query("property", "{not json at all");
        let err = validate_query_fields(&q, DOMAINS).unwrap_err();
        assert!(err.contains("valid JSON"));
    }

    #[test]
    fn query_params_at_limit_accepted() {
        // Exactly 8192 chars of valid JSON: {"k":"xxx..."}
        let wrapper_len = r#"{"k":""}"#.len(); // 8 chars
        let padding = "x".repeat(8192 - wrapper_len);
        let json = format!(r#"{{"k":"{}"}}"#, padding);
        assert_eq!(json.len(), 8192);
        let q = make_query("property", &json);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_params_over_limit_rejected() {
        let big = "x".repeat(8193);
        let q = make_query("property", &big);
        let err = validate_query_fields(&q, DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    // ---- validate_event_fields ----

    #[test]
    fn event_valid_domain_accepted() {
        let e = make_event("housing", "{}");
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn event_invalid_domain_rejected() {
        let e = make_event("unknown", "{}");
        let err = validate_event_fields(&e, DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_payload_at_limit_accepted() {
        let payload = "x".repeat(8192);
        let e = make_event("property", &payload);
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn event_payload_over_limit_rejected() {
        let payload = "x".repeat(8193);
        let e = make_event("property", &payload);
        let err = validate_event_fields(&e, DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    #[test]
    fn event_exactly_20_hashes_accepted() {
        let mut e = make_event("property", "{}");
        e.related_hashes = (0..20).map(|i| format!("hash_{}", i)).collect();
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn event_21_hashes_rejected() {
        let mut e = make_event("property", "{}");
        e.related_hashes = (0..21).map(|i| format!("hash_{}", i)).collect();
        let err = validate_event_fields(&e, DOMAINS).unwrap_err();
        assert!(err.contains("20 related hashes"));
    }

    #[test]
    fn event_zero_hashes_accepted() {
        let e = make_event("care", "{}");
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    // ---- Serde roundtrip ----

    #[test]
    fn query_serde_roundtrip() {
        let q = make_query("property", r#"{"key":"value"}"#);
        let bytes = serde_json::to_vec(&q).unwrap();
        let q2: BridgeQueryEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(q, q2);
    }

    #[test]
    fn query_with_result_serde_roundtrip() {
        let mut q = make_query("housing", "{}");
        q.result = Some("resolved data".into());
        q.resolved_at = Some(Timestamp::from_micros(1_000_000));
        q.success = Some(true);
        let bytes = serde_json::to_vec(&q).unwrap();
        let q2: BridgeQueryEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(q, q2);
    }

    #[test]
    fn event_serde_roundtrip() {
        let mut e = make_event("care", r#"{"action":"help"}"#);
        e.related_hashes = vec!["hash_a".into(), "hash_b".into()];
        let bytes = serde_json::to_vec(&e).unwrap();
        let e2: BridgeEventEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(e, e2);
    }

    // ---- Food & transport domain validation ----

    #[test]
    fn query_food_domain_accepted() {
        let q = make_query("food", r#"{"product":"tomatoes"}"#);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_transport_domain_accepted() {
        let q = make_query("transport", r#"{"route_id":"r1"}"#);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn event_food_domain_accepted() {
        let e = make_event("food", r#"{"event":"harvest_recorded"}"#);
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn event_transport_domain_accepted() {
        let mut e = make_event("transport", r#"{"event":"trip_logged"}"#);
        e.related_hashes = vec!["vehicle_hash_1".into()];
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn query_all_domains_accepted() {
        for domain in DOMAINS {
            let q = make_query(domain, "{}");
            assert!(validate_query_fields(&q, DOMAINS).is_ok(), "domain '{}' should be valid", domain);
        }
    }

    #[test]
    fn event_all_domains_accepted() {
        for domain in DOMAINS {
            let e = make_event(domain, "{}");
            assert!(validate_event_fields(&e, DOMAINS).is_ok(), "domain '{}' should be valid", domain);
        }
    }

    // ---- Empty domain list ----

    #[test]
    fn empty_domain_list_rejects_everything() {
        let q = make_query("property", "{}");
        assert!(validate_query_fields(&q, &[]).is_err());

        let e = make_event("property", "{}");
        assert!(validate_event_fields(&e, &[]).is_err());
    }
}
