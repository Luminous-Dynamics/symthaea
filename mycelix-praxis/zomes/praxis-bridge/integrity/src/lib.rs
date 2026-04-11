// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EduNet Bridge Integrity Zome
//!
//! Unified bridge for cross-domain integration within the EduNet cluster.
//! Covers 10 domains: learning, fl, credential, dao, srs, gamification,
//! adaptive, pods, knowledge, integration.
//!
//! Entry struct definitions come from `mycelix-bridge-entry-types` (shared
//! across all cluster bridges). The `EntryTypes` enum and validation are local.

#![deny(unsafe_code)]

use hdi::prelude::*;
pub use mycelix_bridge_entry_types::CachedCredentialEntry;
use mycelix_bridge_entry_types::{
    check_author_match, check_link_author_match, validate_cached_credential,
    validate_event_fields, validate_query_fields, BridgeEventEntry, BridgeQueryEntry,
};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Backward-compatible type alias for edunet queries.
pub type EdunetQueryEntry = BridgeQueryEntry;

/// Backward-compatible type alias for edunet events.
pub type EdunetEventEntry = BridgeEventEntry;

/// Trust tier alias — maps to the shared consciousness tier system.
/// Observer → Participant → Citizen → Steward → Guardian
// Note: CivicTier lives in mycelix-bridge-common (coordinator dep).
// Re-exported here as documentation; coordinator zome provides the runtime gating.
pub type TrustTier = u8;

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
    AllQueries,
    AgentToQuery,
    DomainToQuery,
    AllEvents,
    EventTypeToEvent,
    AgentToEvent,
    DomainToEvent,
    /// Rate limit tracking: agent -> anchor per dispatch call
    DispatchRateLimit,
    /// Agent -> cached consciousness credential
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

const VALID_DOMAINS: &[&str] = &[
    "learning",
    "fl",
    "credential",
    "dao",
    "srs",
    "gamification",
    "adaptive",
    "pods",
    "knowledge",
    "integration",
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

    // -- Helpers --

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn make_query(domain: &str, params: &str) -> BridgeQueryEntry {
        BridgeQueryEntry {
            schema_version: 1,
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
            schema_version: 1,
            domain: domain.into(),
            event_type: "test_event".into(),
            source_agent: fake_agent(),
            payload: payload.into(),
            created_at: Timestamp::from_micros(0),
            related_hashes: vec![],
        }
    }

    // -- VALID_DOMAINS --

    #[test]
    fn valid_domains_covers_all_edunet_domains() {
        assert!(VALID_DOMAINS.contains(&"learning"), "missing learning");
        assert!(VALID_DOMAINS.contains(&"fl"), "missing fl");
        assert!(VALID_DOMAINS.contains(&"credential"), "missing credential");
        assert!(VALID_DOMAINS.contains(&"dao"), "missing dao");
        assert!(VALID_DOMAINS.contains(&"srs"), "missing srs");
        assert!(VALID_DOMAINS.contains(&"gamification"), "missing gamification");
        assert!(VALID_DOMAINS.contains(&"adaptive"), "missing adaptive");
        assert!(VALID_DOMAINS.contains(&"pods"), "missing pods");
        assert!(VALID_DOMAINS.contains(&"knowledge"), "missing knowledge");
        assert!(VALID_DOMAINS.contains(&"integration"), "missing integration");
    }

    #[test]
    fn valid_domains_has_expected_count() {
        assert_eq!(
            VALID_DOMAINS.len(),
            11,
            "expected 11 edunet domains (10 + governance_gate)"
        );
    }

    #[test]
    fn valid_domains_does_not_contain_civic_domains() {
        assert!(!VALID_DOMAINS.contains(&"justice"), "justice is a civic domain");
        assert!(!VALID_DOMAINS.contains(&"emergency"), "emergency is a civic domain");
        assert!(!VALID_DOMAINS.contains(&"media"), "media is a civic domain");
    }

    #[test]
    fn valid_domains_does_not_contain_commons_domains() {
        assert!(!VALID_DOMAINS.contains(&"property"), "property is a commons domain");
        assert!(!VALID_DOMAINS.contains(&"housing"), "housing is a commons domain");
        assert!(!VALID_DOMAINS.contains(&"water"), "water is a commons domain");
    }

    // -- Query validation: domain --

    #[test]
    fn query_all_edunet_domains_valid() {
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
    fn query_learning_domain_valid() {
        let q = make_query("learning", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_fl_domain_valid() {
        let q = make_query("fl", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_credential_domain_valid() {
        let q = make_query("credential", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_civic_domain_rejected() {
        let q = make_query("justice", "{}");
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
        let q = make_query("Learning", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn query_whitespace_domain_rejected() {
        let q = make_query(" learning", "{}");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    // -- Query validation: params --

    #[test]
    fn query_empty_params_accepted() {
        let q = make_query("learning", "");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_valid_json_params_accepted() {
        let q = make_query("fl", r#"{"key":"value","nested":{"a":1}}"#);
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_invalid_json_params_rejected() {
        let q = make_query("credential", "{bad json");
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("valid JSON"));
    }

    #[test]
    fn query_oversized_params_rejected() {
        let big = "x".repeat(8193);
        let q = make_query("dao", &big);
        let err = validate_query_fields(&q, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    // -- Event validation: domain --

    #[test]
    fn event_all_edunet_domains_valid() {
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
    fn event_learning_domain_valid() {
        let e = make_event("learning", "{}");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_civic_domain_rejected() {
        let e = make_event("justice", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_empty_domain_rejected() {
        let e = make_event("", "{}");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_err());
    }

    // -- Event validation: payload --

    #[test]
    fn event_empty_payload_accepted() {
        let e = make_event("srs", "");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_oversized_payload_rejected() {
        let big = "x".repeat(8193);
        let e = make_event("gamification", &big);
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    // -- Event validation: related_hashes --

    #[test]
    fn event_exactly_20_hashes_accepted() {
        let mut e = make_event("adaptive", "{}");
        e.related_hashes = (0..20).map(|i| format!("hash_{}", i)).collect();
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_21_hashes_rejected() {
        let mut e = make_event("pods", "{}");
        e.related_hashes = (0..21).map(|i| format!("hash_{}", i)).collect();
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("20 related hashes"));
    }

    // -- Serde roundtrip --

    #[test]
    fn serde_roundtrip_query() {
        let q = make_query("learning", r#"{"course_id":"c-001"}"#);
        let json = serde_json::to_string(&q).unwrap();
        let back: BridgeQueryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, q);
    }

    #[test]
    fn serde_roundtrip_event() {
        let mut e = make_event("fl", r#"{"action":"round_complete"}"#);
        e.related_hashes = vec!["hash_a".into(), "hash_b".into()];
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

    // -- Type alias tests --

    #[test]
    fn edunet_query_alias_is_bridge_query() {
        let q = make_query("learning", "{}");
        let _edunet: EdunetQueryEntry = q;
    }

    #[test]
    fn edunet_event_alias_is_bridge_event() {
        let e = make_event("learning", "{}");
        let _edunet: EdunetEventEntry = e;
    }

    // -- Validate wrappers --

    #[test]
    fn validate_query_wrapper_valid_returns_valid() {
        let q = make_query("learning", "{}");
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
        let e = make_event("fl", "{}");
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

    // -- Anchor tests --

    #[test]
    fn anchor_clone_equality() {
        let a = Anchor("edunet_queries".to_string());
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn anchor_different_values_not_equal() {
        let a = Anchor("queries".to_string());
        let b = Anchor("events".to_string());
        assert_ne!(a, b);
    }
}
