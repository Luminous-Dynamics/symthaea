//! Commons Bridge Integrity Zome
//!
//! Unified bridge for cross-domain integration within the Commons cluster.
//! Replaces the 5 separate bridge zomes from property, housing, care,
//! mutualaid, and water hApps.
//!
//! Entry struct definitions come from `mycelix-bridge-entry-types` (shared
//! with the Civic bridge). The `EntryTypes` enum and validation are local.

use hdi::prelude::*;
pub use mycelix_bridge_entry_types::CachedCredentialEntry;
use mycelix_bridge_entry_types::{
    validate_cached_credential, validate_event_fields, validate_query_fields, BridgeEventEntry,
    BridgeQueryEntry,
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
    CachedCredential(CachedCredentialEntry),
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
    /// Agent → cached consciousness credential
    AgentToCredentialCache,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry {
            app_entry,
            action: _,
        }) => match app_entry {
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            EntryTypes::Query(query) => validate_query(&query),
            EntryTypes::Event(event) => validate_event(&event),
            EntryTypes::CachedCredential(cred) => validate_credential_cache(&cred),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            EntryTypes::Query(query) => validate_query(&query),
            EntryTypes::Event(event) => validate_event(&event),
            EntryTypes::CachedCredential(cred) => validate_credential_cache(&cred),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

const VALID_DOMAINS: &[&str] = &[
    "property",
    "housing",
    "care",
    "mutualaid",
    "water",
    "food",
    "transport",
    "support",
    "space",
    "governance_gate",
];

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

fn validate_credential_cache(cred: &CachedCredentialEntry) -> ExternResult<ValidateCallbackResult> {
    match validate_cached_credential(cred) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

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

    // ── VALID_DOMAINS ───────────────────────────────────────────────────

    #[test]
    fn valid_domains_covers_all_commons_domains() {
        assert!(VALID_DOMAINS.contains(&"property"), "missing property");
        assert!(VALID_DOMAINS.contains(&"housing"), "missing housing");
        assert!(VALID_DOMAINS.contains(&"care"), "missing care");
        assert!(VALID_DOMAINS.contains(&"mutualaid"), "missing mutualaid");
        assert!(VALID_DOMAINS.contains(&"water"), "missing water");
        assert!(VALID_DOMAINS.contains(&"food"), "missing food");
        assert!(VALID_DOMAINS.contains(&"transport"), "missing transport");
        assert!(VALID_DOMAINS.contains(&"support"), "missing support");
        assert!(VALID_DOMAINS.contains(&"space"), "missing space");
    }

    #[test]
    fn valid_domains_has_expected_count() {
        assert_eq!(
            VALID_DOMAINS.len(),
            10,
            "expected 10 commons domains (9 + governance_gate)"
        );
    }

    #[test]
    fn valid_domains_does_not_contain_civic_domains() {
        assert!(
            !VALID_DOMAINS.contains(&"justice"),
            "justice is a civic domain"
        );
        assert!(
            !VALID_DOMAINS.contains(&"emergency"),
            "emergency is a civic domain"
        );
        assert!(!VALID_DOMAINS.contains(&"media"), "media is a civic domain");
    }

    // ── Query validation: domain ────────────────────────────────────────

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
    fn query_civic_domain_justice_rejected() {
        let q = make_query("justice", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
        assert!(err.contains("justice"));
    }

    #[test]
    fn query_civic_domain_emergency_rejected() {
        let q = make_query("emergency", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn query_civic_domain_media_rejected() {
        let q = make_query("media", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn query_empty_domain_rejected() {
        let q = make_query("", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_err());
    }

    #[test]
    fn query_nonexistent_domain_rejected() {
        let q = make_query("finance", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn query_case_sensitive_domain_rejected() {
        let q = make_query("Property", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    // ── Query validation: params ────────────────────────────────────────

    #[test]
    fn query_empty_params_accepted() {
        let q = make_query("property", "");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_valid_json_params_accepted() {
        let q = make_query("property", r#"{"key":"value","nested":{"a":1}}"#);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_invalid_json_params_rejected() {
        let q = make_query("housing", "{bad json");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("valid JSON"));
    }

    #[test]
    fn query_params_at_boundary_8192_accepted() {
        let wrapper_len = r#"{"k":""}"#.len();
        let padding = "x".repeat(8192 - wrapper_len);
        let json = format!(r#"{{"k":"{}"}}"#, padding);
        assert_eq!(json.len(), 8192);
        let q = make_query("property", &json);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_oversized_params_rejected() {
        let big = "x".repeat(8193);
        let q = make_query("property", &big);
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    #[test]
    fn query_json_array_params_accepted() {
        let q = make_query("care", r#"[1, 2, 3]"#);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_json_string_params_accepted() {
        let q = make_query("water", r#""just a string""#);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    // ── Query validation: optional fields ───────────────────────────────

    #[test]
    fn query_with_result_and_success() {
        let mut q = make_query("property", "{}");
        q.result = Some("resolved data".into());
        q.resolved_at = Some(Timestamp::from_micros(1_000_000));
        q.success = Some(true);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_with_failed_result() {
        let mut q = make_query("housing", "{}");
        q.result = Some("error: not found".into());
        q.success = Some(false);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    // ── Event validation: domain ────────────────────────────────────────

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
    fn event_civic_domain_emergency_rejected() {
        let e = make_event("emergency", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_civic_domain_justice_rejected() {
        let e = make_event("justice", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_civic_domain_media_rejected() {
        let e = make_event("media", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_empty_domain_rejected() {
        let e = make_event("", "{}");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_err());
    }

    #[test]
    fn event_nonexistent_domain_rejected() {
        let e = make_event("governance", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_case_sensitive_domain_rejected() {
        let e = make_event("Water", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    // ── Event validation: payload ───────────────────────────────────────

    #[test]
    fn event_empty_payload_accepted() {
        let e = make_event("property", "");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_payload_at_boundary_8192_accepted() {
        let payload = "x".repeat(8192);
        let e = make_event("property", &payload);
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_oversized_payload_rejected() {
        let big = "x".repeat(8193);
        let e = make_event("water", &big);
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    // ── Event validation: related_hashes ────────────────────────────────

    #[test]
    fn event_zero_hashes_accepted() {
        let e = make_event("care", "{}");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_exactly_20_hashes_accepted() {
        let mut e = make_event("care", "{}");
        e.related_hashes = (0..20).map(|i| format!("hash_{}", i)).collect();
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_21_hashes_rejected() {
        let mut e = make_event("food", "{}");
        e.related_hashes = (0..21).map(|i| format!("hash_{}", i)).collect();
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("20 related hashes"));
    }

    #[test]
    fn event_one_hash_accepted() {
        let mut e = make_event("transport", "{}");
        e.related_hashes = vec!["single_hash".into()];
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_query() {
        let q = make_query("property", r#"{"key":"value"}"#);
        let json = serde_json::to_string(&q).unwrap();
        let back: BridgeQueryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, q);
    }

    #[test]
    fn serde_roundtrip_query_with_result() {
        let mut q = make_query("housing", "{}");
        q.result = Some("data here".into());
        q.resolved_at = Some(Timestamp::from_micros(5_000_000));
        q.success = Some(true);
        let json = serde_json::to_string(&q).unwrap();
        let back: BridgeQueryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, q);
    }

    #[test]
    fn serde_roundtrip_event() {
        let mut e = make_event("care", r#"{"action":"help"}"#);
        e.related_hashes = vec!["hash_a".into(), "hash_b".into()];
        let json = serde_json::to_string(&e).unwrap();
        let back: BridgeEventEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn serde_roundtrip_event_no_hashes() {
        let e = make_event("mutualaid", "{}");
        let json = serde_json::to_string(&e).unwrap();
        let back: BridgeEventEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    // ── Type alias tests ────────────────────────────────────────────────

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

    // ── Anchor test ─────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip_anchor() {
        let a = Anchor("all_queries".to_string());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    // ── Update validation: Query ─────────────────────────────────────────

    #[test]
    fn test_update_query_invalid_domain_rejected() {
        let q = make_query("finance", "{}");
        let result = validate_query(&q);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(msg.contains("Invalid domain"), "got: {msg}");
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn test_update_query_invalid_empty_domain_rejected() {
        let q = make_query("", "{}");
        let result = validate_query(&q);
        match result {
            Ok(ValidateCallbackResult::Invalid(_)) => {}
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn test_update_query_invalid_oversized_params_rejected() {
        let big = "x".repeat(8193);
        let q = make_query("property", &big);
        let result = validate_query(&q);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(msg.contains("8192"), "got: {msg}");
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Update validation: Event ─────────────────────────────────────────

    #[test]
    fn test_update_event_invalid_domain_rejected() {
        let e = make_event("justice", "{}");
        let result = validate_event(&e);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(msg.contains("Invalid domain"), "got: {msg}");
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn test_update_event_invalid_oversized_payload_rejected() {
        let big = "x".repeat(8193);
        let e = make_event("water", &big);
        let result = validate_event(&e);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(msg.contains("8192"), "got: {msg}");
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn test_update_event_invalid_too_many_hashes_rejected() {
        let mut e = make_event("food", "{}");
        e.related_hashes = (0..21).map(|i| format!("hash_{}", i)).collect();
        let result = validate_event(&e);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(msg.contains("20 related hashes"), "got: {msg}");
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }
}
