// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Civic Bridge Integrity Zome
//!
//! Unified bridge for cross-domain integration within the Civic cluster.
//! Replaces the 3 separate bridge zomes from justice, emergency, and media.
//!
//! Entry struct definitions come from `mycelix-bridge-entry-types` (shared
//! with the Commons bridge). The `EntryTypes` enum and validation are local.

use hdi::prelude::*;
pub use mycelix_bridge_entry_types::CachedCredentialEntry;
use mycelix_bridge_entry_types::{
    check_author_match, check_link_author_match, validate_cached_credential,
    validate_event_fields, validate_query_fields, BridgeEventEntry, BridgeQueryEntry,
    CrossClusterNotification,
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
    CachedCredential(CachedCredentialEntry),
    Notification(CrossClusterNotification),
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
    /// Agent → cached consciousness credential
    AgentToCredentialCache,
    /// Agent → their notifications
    AgentToNotification,
    /// Global notifications anchor
    AllNotifications,
    /// Agent → notification subscription preferences
    NotificationSubscription,
    /// Agent → saga workflows
    AgentToSaga,
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
            EntryTypes::Notification(_) => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            EntryTypes::Query(query) => validate_query(&query),
            EntryTypes::Event(event) => validate_event(&event),
            EntryTypes::CachedCredential(cred) => validate_credential_cache(&cred),
            EntryTypes::Notification(_) => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

const VALID_DOMAINS: &[&str] = &["justice", "emergency", "media", "governance_gate"];

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
    fn valid_domains_covers_all_civic_domains() {
        assert!(VALID_DOMAINS.contains(&"justice"), "missing justice");
        assert!(VALID_DOMAINS.contains(&"emergency"), "missing emergency");
        assert!(VALID_DOMAINS.contains(&"media"), "missing media");
    }

    #[test]
    fn valid_domains_has_expected_count() {
        assert_eq!(
            VALID_DOMAINS.len(),
            4,
            "expected 4 civic domains (3 + governance_gate)"
        );
    }

    #[test]
    fn valid_domains_does_not_contain_commons_domains() {
        assert!(
            !VALID_DOMAINS.contains(&"property"),
            "property is a commons domain"
        );
        assert!(
            !VALID_DOMAINS.contains(&"housing"),
            "housing is a commons domain"
        );
        assert!(!VALID_DOMAINS.contains(&"care"), "care is a commons domain");
        assert!(
            !VALID_DOMAINS.contains(&"mutualaid"),
            "mutualaid is a commons domain"
        );
        assert!(
            !VALID_DOMAINS.contains(&"water"),
            "water is a commons domain"
        );
        assert!(!VALID_DOMAINS.contains(&"food"), "food is a commons domain");
        assert!(
            !VALID_DOMAINS.contains(&"transport"),
            "transport is a commons domain"
        );
    }

    // ── Query validation: domain ────────────────────────────────────────

    #[test]
    fn query_all_civic_domains_valid() {
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
    fn query_commons_domain_property_rejected() {
        let q = make_query("property", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
        assert!(err.contains("property"));
    }

    #[test]
    fn query_commons_domain_housing_rejected() {
        let q = make_query("housing", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn query_commons_domain_care_rejected() {
        let q = make_query("care", "{}");
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
        let q = make_query("Justice", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn query_whitespace_domain_rejected() {
        let q = make_query(" justice", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    // ── Query validation: params ────────────────────────────────────────

    #[test]
    fn query_empty_params_accepted() {
        let q = make_query("justice", "");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_valid_json_params_accepted() {
        let q = make_query("justice", r#"{"key":"value","nested":{"a":1}}"#);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_invalid_json_params_rejected() {
        let q = make_query("justice", "{bad json");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("valid JSON"));
    }

    #[test]
    fn query_params_at_boundary_8192_accepted() {
        let wrapper_len = r#"{"k":""}"#.len();
        let padding = "x".repeat(8192 - wrapper_len);
        let json = format!(r#"{{"k":"{}"}}"#, padding);
        assert_eq!(json.len(), 8192);
        let q = make_query("justice", &json);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_oversized_params_rejected() {
        let big = "x".repeat(8193);
        let q = make_query("justice", &big);
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    #[test]
    fn query_json_array_params_accepted() {
        let q = make_query("emergency", r#"[1, 2, 3]"#);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_json_string_params_accepted() {
        let q = make_query("media", r#""just a string""#);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_json_number_params_accepted() {
        let q = make_query("justice", "42");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_json_null_params_accepted() {
        let q = make_query("justice", "null");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_json_boolean_params_accepted() {
        let q = make_query("emergency", "true");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    // ── Query validation: optional fields ───────────────────────────────

    #[test]
    fn query_with_result_and_success() {
        let mut q = make_query("justice", "{}");
        q.result = Some("resolved data".into());
        q.resolved_at = Some(Timestamp::from_micros(1_000_000));
        q.success = Some(true);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_with_failed_result() {
        let mut q = make_query("emergency", "{}");
        q.result = Some("error: not found".into());
        q.success = Some(false);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_with_only_result_no_success() {
        let mut q = make_query("media", "{}");
        q.result = Some("partial data".into());
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    // ── Event validation: domain ────────────────────────────────────────

    #[test]
    fn event_all_civic_domains_valid() {
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
    fn event_commons_domain_housing_rejected() {
        let e = make_event("housing", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_commons_domain_water_rejected() {
        let e = make_event("water", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_commons_domain_property_rejected() {
        let e = make_event("property", "{}");
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
        let e = make_event("Emergency", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_whitespace_domain_rejected() {
        let e = make_event("media ", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    // ── Event validation: payload ───────────────────────────────────────

    #[test]
    fn event_empty_payload_accepted() {
        let e = make_event("justice", "");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_payload_at_boundary_8192_accepted() {
        let payload = "x".repeat(8192);
        let e = make_event("justice", &payload);
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_oversized_payload_rejected() {
        let big = "x".repeat(8193);
        let e = make_event("emergency", &big);
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    #[test]
    fn event_payload_just_over_boundary_8193_rejected() {
        let payload = "y".repeat(8193);
        let e = make_event("media", &payload);
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_err());
    }

    // ── Event validation: related_hashes ────────────────────────────────

    #[test]
    fn event_zero_hashes_accepted() {
        let e = make_event("justice", "{}");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_one_hash_accepted() {
        let mut e = make_event("emergency", "{}");
        e.related_hashes = vec!["single_hash".into()];
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_exactly_20_hashes_accepted() {
        let mut e = make_event("media", "{}");
        e.related_hashes = (0..20).map(|i| format!("hash_{}", i)).collect();
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_21_hashes_rejected() {
        let mut e = make_event("justice", "{}");
        e.related_hashes = (0..21).map(|i| format!("hash_{}", i)).collect();
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("20 related hashes"));
    }

    #[test]
    fn event_100_hashes_rejected() {
        let mut e = make_event("emergency", "{}");
        e.related_hashes = (0..100).map(|i| format!("hash_{}", i)).collect();
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("20 related hashes"));
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_query() {
        let q = make_query("justice", r#"{"case_id":"c-001"}"#);
        let json = serde_json::to_string(&q).unwrap();
        let back: BridgeQueryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, q);
    }

    #[test]
    fn serde_roundtrip_query_with_result() {
        let mut q = make_query("emergency", "{}");
        q.result = Some("dispatch confirmed".into());
        q.resolved_at = Some(Timestamp::from_micros(5_000_000));
        q.success = Some(true);
        let json = serde_json::to_string(&q).unwrap();
        let back: BridgeQueryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, q);
    }

    #[test]
    fn serde_roundtrip_query_with_failed_result() {
        let mut q = make_query("media", "{}");
        q.result = Some("not found".into());
        q.success = Some(false);
        let json = serde_json::to_string(&q).unwrap();
        let back: BridgeQueryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, q);
    }

    #[test]
    fn serde_roundtrip_event() {
        let mut e = make_event("justice", r#"{"action":"filing"}"#);
        e.related_hashes = vec!["hash_a".into(), "hash_b".into()];
        let json = serde_json::to_string(&e).unwrap();
        let back: BridgeEventEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn serde_roundtrip_event_no_hashes() {
        let e = make_event("emergency", "{}");
        let json = serde_json::to_string(&e).unwrap();
        let back: BridgeEventEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn serde_roundtrip_anchor() {
        let a = Anchor("all_queries".to_string());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn serde_roundtrip_anchor_empty() {
        let a = Anchor(String::new());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    // ── Type alias tests ────────────────────────────────────────────────

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

    // ── validate_query / validate_event wrappers ────────────────────────

    #[test]
    fn validate_query_wrapper_valid_returns_valid() {
        let q = make_query("justice", "{}");
        let result = validate_query(&q).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn validate_query_wrapper_invalid_returns_invalid() {
        let q = make_query("property", "{}");
        let result = validate_query(&q).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Invalid domain"));
            }
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn validate_event_wrapper_valid_returns_valid() {
        let e = make_event("emergency", "{}");
        let result = validate_event(&e).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn validate_event_wrapper_invalid_returns_invalid() {
        let e = make_event("housing", "{}");
        let result = validate_event(&e).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Invalid domain"));
            }
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ── Anchor tests ────────────────────────────────────────────────────

    #[test]
    fn anchor_clone_equality() {
        let a = Anchor("civic_queries".to_string());
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn anchor_different_values_not_equal() {
        let a = Anchor("queries".to_string());
        let b = Anchor("events".to_string());
        assert_ne!(a, b);
    }

    // ── Update validation: Query ─────────────────────────────────────────

    #[test]
    fn test_update_query_invalid_domain_rejected() {
        let q = make_query("property", "{}");
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
        let q = make_query("justice", &big);
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
        let e = make_event("property", "{}");
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
        let e = make_event("emergency", &big);
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
        let mut e = make_event("media", "{}");
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
