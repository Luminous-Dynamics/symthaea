//! Personal Bridge Integrity Zome
//!
//! Validates bridge queries and events for the Sovereign tier.
//! Uses shared bridge entry types from mycelix-bridge-entry-types.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{
    BridgeQueryEntry, BridgeEventEntry,
    validate_query_fields, validate_event_fields,
};

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Backward-compatible type alias.
pub type PersonalQueryEntry = BridgeQueryEntry;

/// Backward-compatible type alias.
pub type PersonalEventEntry = BridgeEventEntry;

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
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            EntryTypes::Query(query) => validate_query(&query),
            EntryTypes::Event(event) => validate_event(&event),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

const VALID_DOMAINS: &[&str] = &["identity", "health", "credential"];

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

    #[test]
    fn valid_domains_covers_all_personal_domains() {
        assert!(VALID_DOMAINS.contains(&"identity"));
        assert!(VALID_DOMAINS.contains(&"health"));
        assert!(VALID_DOMAINS.contains(&"credential"));
    }

    #[test]
    fn valid_domains_has_expected_count() {
        assert_eq!(VALID_DOMAINS.len(), 3);
    }

    #[test]
    fn valid_domains_does_not_contain_civic_or_commons() {
        assert!(!VALID_DOMAINS.contains(&"justice"));
        assert!(!VALID_DOMAINS.contains(&"property"));
        assert!(!VALID_DOMAINS.contains(&"emergency"));
    }

    #[test]
    fn query_identity_domain_valid() {
        let q = make_query("identity", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_health_domain_valid() {
        let q = make_query("health", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_credential_domain_valid() {
        let q = make_query("credential", "{}");
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn query_justice_domain_rejected() {
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
    fn event_identity_domain_valid() {
        let e = make_event("identity", "{}");
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_ok());
    }

    #[test]
    fn event_property_domain_rejected() {
        let e = make_event("property", "{}");
        let err = validate_event_fields(&e, VALID_DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_oversized_payload_rejected() {
        let e = make_event("health", &"x".repeat(8193));
        assert!(validate_event_fields(&e, VALID_DOMAINS).is_err());
    }

    #[test]
    fn query_oversized_params_rejected() {
        let q = make_query("credential", &"x".repeat(8193));
        assert!(validate_query_fields(&q, VALID_DOMAINS).is_err());
    }

    #[test]
    fn anchor_clone_equality() {
        let a = Anchor("personal_queries".to_string());
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn serde_roundtrip_query() {
        let q = make_query("identity", r#"{"field":"display_name"}"#);
        let json = serde_json::to_string(&q).unwrap();
        let back: BridgeQueryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, q);
    }

    #[test]
    fn serde_roundtrip_event() {
        let e = make_event("health", r#"{"type":"biometric_update"}"#);
        let json = serde_json::to_string(&e).unwrap();
        let back: BridgeEventEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn validate_query_wrapper_valid() {
        let q = make_query("identity", "{}");
        let result = validate_query(&q).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn validate_query_wrapper_invalid() {
        let q = make_query("property", "{}");
        let result = validate_query(&q).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn validate_event_wrapper_valid() {
        let e = make_event("credential", "{}");
        let result = validate_event(&e).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn validate_event_wrapper_invalid() {
        let e = make_event("housing", "{}");
        let result = validate_event(&e).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn type_alias_compatibility() {
        let q = make_query("identity", "{}");
        let _personal: PersonalQueryEntry = q;
        let e = make_event("health", "{}");
        let _personal_event: PersonalEventEntry = e;
    }
}
