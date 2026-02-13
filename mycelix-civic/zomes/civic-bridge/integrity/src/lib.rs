//! Civic Bridge Integrity Zome
//!
//! Unified bridge for cross-domain integration within the Civic cluster.
//! Replaces the 3 separate bridge zomes from justice, emergency, and media.
//!
//! Entry struct definitions come from `mycelix-bridge-entry-types` (shared
//! with the Commons bridge). The `EntryTypes` enum and validation are local.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{
    BridgeQueryEntry, BridgeEventEntry,
    validate_query_fields, validate_event_fields,
};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Backward-compatible type alias — code that references `CivicQueryEntry` still compiles.
pub type CivicQueryEntry = BridgeQueryEntry;

/// Backward-compatible type alias.
pub type CivicEventEntry = BridgeEventEntry;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Query(BridgeQueryEntry),
    Event(BridgeEventEntry),
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
    /// Rate limit tracking: agent → anchor per dispatch call
    DispatchRateLimit,
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

const VALID_DOMAINS: &[&str] = &["justice", "emergency", "media"];

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

    // ---- VALID_DOMAINS ----

    #[test]
    fn valid_domains_covers_all_civic_domains() {
        assert!(VALID_DOMAINS.contains(&"justice"), "missing justice");
        assert!(VALID_DOMAINS.contains(&"emergency"), "missing emergency");
        assert!(VALID_DOMAINS.contains(&"media"), "missing media");
    }

    #[test]
    fn valid_domains_has_expected_count() {
        assert_eq!(VALID_DOMAINS.len(), 3, "expected 3 civic domains");
    }

    // ---- Query validation ----

    #[test]
    fn query_justice_domain_valid() {
        let q = make_query("justice", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_emergency_domain_valid() {
        let q = make_query("emergency", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_media_domain_valid() {
        let q = make_query("media", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_invalid_domain_rejected() {
        let q = make_query("property", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
        assert!(err.contains("property"));
    }

    #[test]
    fn query_oversized_params_rejected() {
        let big = "x".repeat(8193);
        let q = make_query("justice", &big);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_err());
    }

    #[test]
    fn query_invalid_json_params_rejected() {
        let q = make_query("justice", "{not json");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("valid JSON"));
    }

    // ---- Event validation ----

    #[test]
    fn event_justice_domain_valid() {
        let e = make_event("justice", "{}");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_emergency_domain_valid() {
        let e = make_event("emergency", "{}");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_media_domain_valid() {
        let e = make_event("media", "{}");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_invalid_domain_rejected() {
        let e = make_event("housing", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_oversized_payload_rejected() {
        let big = "x".repeat(8193);
        let e = make_event("emergency", &big);
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_err());
    }

    #[test]
    fn event_too_many_related_hashes_rejected() {
        let mut e = make_event("media", "{}");
        e.related_hashes = (0..21).map(|i| format!("hash_{}", i)).collect();
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("20 related hashes"));
    }

    // ---- Type alias ----

    #[test]
    fn civic_query_alias_is_bridge_query() {
        let q = make_query("justice", "{}");
        let _civic: CivicQueryEntry = q;
    }

    #[test]
    fn civic_event_alias_is_bridge_event() {
        let e = make_event("justice", "{}");
        let _civic: CivicEventEntry = e;
    }
}
