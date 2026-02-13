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

const VALID_DOMAINS: &[&str] = &["property", "housing", "care", "mutualaid", "water", "food", "transport"];

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
    fn valid_domains_covers_all_commons_domains() {
        assert!(VALID_DOMAINS.contains(&"property"), "missing property");
        assert!(VALID_DOMAINS.contains(&"housing"), "missing housing");
        assert!(VALID_DOMAINS.contains(&"care"), "missing care");
        assert!(VALID_DOMAINS.contains(&"mutualaid"), "missing mutualaid");
        assert!(VALID_DOMAINS.contains(&"water"), "missing water");
        assert!(VALID_DOMAINS.contains(&"food"), "missing food");
        assert!(VALID_DOMAINS.contains(&"transport"), "missing transport");
    }

    #[test]
    fn valid_domains_has_expected_count() {
        assert_eq!(VALID_DOMAINS.len(), 7, "expected 7 commons domains");
    }

    // ---- Query validation per domain ----

    #[test]
    fn query_all_commons_domains_valid() {
        for domain in VALID_DOMAINS {
            let q = make_query(domain, "{}");
            assert!(
                validate_query_fields(&q, VALID_DOMAINS).is_ok(),
                "domain '{}' should be valid for queries",
                domain
            );
        }
    }

    #[test]
    fn query_civic_domain_rejected() {
        let q = make_query("justice", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
        assert!(err.contains("justice"));
    }

    #[test]
    fn query_oversized_params_rejected() {
        let big = "x".repeat(8193);
        let q = make_query("property", &big);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_err());
    }

    #[test]
    fn query_invalid_json_params_rejected() {
        let q = make_query("housing", "{bad json");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("valid JSON"));
    }

    // ---- Event validation per domain ----

    #[test]
    fn event_all_commons_domains_valid() {
        for domain in VALID_DOMAINS {
            let e = make_event(domain, "{}");
            assert!(
                validate_event_fields(&e, VALID_DOMAINS).is_ok(),
                "domain '{}' should be valid for events",
                domain
            );
        }
    }

    #[test]
    fn event_civic_domain_rejected() {
        let e = make_event("emergency", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_oversized_payload_rejected() {
        let big = "x".repeat(8193);
        let e = make_event("water", &big);
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_err());
    }

    #[test]
    fn event_too_many_related_hashes_rejected() {
        let mut e = make_event("food", "{}");
        e.related_hashes = (0..21).map(|i| format!("hash_{}", i)).collect();
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("20 related hashes"));
    }

    // ---- Type alias ----

    #[test]
    fn stored_query_alias_is_bridge_query() {
        let q = make_query("property", "{}");
        let _stored: StoredQuery = q;
    }

    #[test]
    fn stored_event_alias_is_bridge_event() {
        let e = make_event("property", "{}");
        let _stored: StoredEvent = e;
    }
}
