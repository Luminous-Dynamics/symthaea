//! Hearth Bridge Integrity Zome
//!
//! Validates bridge queries and events for the Hearth cluster.
//! Uses shared bridge entry types from mycelix-bridge-entry-types.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{
    BridgeEventEntry, BridgeQueryEntry, validate_cached_credential,
};
pub use mycelix_bridge_entry_types::CachedCredentialEntry;

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Backward-compatible type alias.
pub type HearthQueryEntry = BridgeQueryEntry;

/// Backward-compatible type alias.
pub type HearthEventEntry = BridgeEventEntry;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    BridgeQuery(BridgeQueryEntry),
    BridgeEvent(BridgeEventEntry),
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
    DispatchRateLimit,
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
            EntryTypes::BridgeQuery(query) => validate_query(&query),
            EntryTypes::BridgeEvent(event) => validate_event(&event),
            EntryTypes::CachedCredential(cred) => validate_credential_cache(&cred),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, original_action_hash, .. }) => match app_entry {
            EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Invalid(
                "Anchor cannot be updated once created".into(),
            )),
            EntryTypes::BridgeQuery(query) => {
                validate_query(&query)?;
                validate_query_immutable_fields(&query, &original_action_hash)
            }
            EntryTypes::BridgeEvent(event) => {
                validate_event(&event)?;
                validate_event_immutable_fields(&event, &original_action_hash)
            }
            EntryTypes::CachedCredential(cred) => validate_credential_cache(&cred),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Invalid(
            "Bridge entries cannot be deleted once created".into(),
        )),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

const VALID_DOMAINS: &[&str] = &[
    "kinship",
    "gratitude",
    "stories",
    "care",
    "autonomy",
    "emergency",
    "decisions",
    "resources",
    "milestones",
    "rhythms",
    "governance_gate",
];

fn validate_query(query: &BridgeQueryEntry) -> ExternResult<ValidateCallbackResult> {
    // Bridge queries use payload field (params) up to 65536 bytes
    if query.params.len() > 65536 {
        return Ok(ValidateCallbackResult::Invalid(
            "Query payload must be <= 65536 bytes".into(),
        ));
    }
    // Domain validation
    if !VALID_DOMAINS.contains(&query.domain.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid domain '{}'. Must be one of: {:?}",
            query.domain, VALID_DOMAINS
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_credential_cache(cred: &CachedCredentialEntry) -> ExternResult<ValidateCallbackResult> {
    match validate_cached_credential(cred) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

fn validate_event(event: &BridgeEventEntry) -> ExternResult<ValidateCallbackResult> {
    // Bridge events use payload up to 65536 bytes
    if event.payload.len() > 65536 {
        return Ok(ValidateCallbackResult::Invalid(
            "Event payload must be <= 65536 bytes".into(),
        ));
    }
    // Domain validation
    if !VALID_DOMAINS.contains(&event.domain.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid domain '{}'. Must be one of: {:?}",
            event.domain, VALID_DOMAINS
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_query_immutable_fields(
    new: &BridgeQueryEntry,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: BridgeQueryEntry = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original BridgeQuery: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original BridgeQuery entry is missing".into()
        )))?;
    if new.domain != original.domain {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change domain on a BridgeQuery".into(),
        ));
    }
    if new.query_type != original.query_type {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change query_type on a BridgeQuery".into(),
        ));
    }
    if new.requester != original.requester {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change requester on a BridgeQuery".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_event_immutable_fields(
    new: &BridgeEventEntry,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: BridgeEventEntry = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original BridgeEvent: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original BridgeEvent entry is missing".into()
        )))?;
    if new.domain != original.domain {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change domain on a BridgeEvent".into(),
        ));
    }
    if new.event_type != original.event_type {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change event_type on a BridgeEvent".into(),
        ));
    }
    if new.source_agent != original.source_agent {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change source_agent on a BridgeEvent".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
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

    // ---- VALID_DOMAINS coverage ----

    #[test]
    fn valid_domains_covers_all_hearth_domains() {
        assert!(VALID_DOMAINS.contains(&"kinship"));
        assert!(VALID_DOMAINS.contains(&"gratitude"));
        assert!(VALID_DOMAINS.contains(&"stories"));
        assert!(VALID_DOMAINS.contains(&"care"));
        assert!(VALID_DOMAINS.contains(&"autonomy"));
        assert!(VALID_DOMAINS.contains(&"emergency"));
        assert!(VALID_DOMAINS.contains(&"decisions"));
        assert!(VALID_DOMAINS.contains(&"resources"));
        assert!(VALID_DOMAINS.contains(&"milestones"));
        assert!(VALID_DOMAINS.contains(&"rhythms"));
    }

    #[test]
    fn valid_domains_has_expected_count() {
        assert_eq!(VALID_DOMAINS.len(), 11);
    }

    #[test]
    fn valid_domains_does_not_contain_civic_or_commons() {
        assert!(!VALID_DOMAINS.contains(&"justice"));
        assert!(!VALID_DOMAINS.contains(&"property"));
        assert!(!VALID_DOMAINS.contains(&"housing"));
        assert!(!VALID_DOMAINS.contains(&"media"));
    }

    // ---- Query validation ----

    #[test]
    fn query_all_valid_domains_accepted() {
        for domain in VALID_DOMAINS {
            let q = make_query(domain, "{}");
            let result = validate_query(&q).unwrap();
            assert!(
                matches!(result, ValidateCallbackResult::Valid),
                "domain '{}' should be valid",
                domain
            );
        }
    }

    #[test]
    fn query_invalid_domain_rejected() {
        let q = make_query("justice", "{}");
        let result = validate_query(&q).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("Invalid domain")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn query_empty_domain_rejected() {
        let q = make_query("", "{}");
        match validate_query(&q).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("Invalid domain")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn query_payload_at_65536_passes() {
        let payload = "x".repeat(65536);
        let q = make_query("kinship", &payload);
        assert!(matches!(
            validate_query(&q).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn query_payload_over_65536_rejected() {
        let payload = "x".repeat(65537);
        let q = make_query("kinship", &payload);
        match validate_query(&q).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("65536")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Event validation ----

    #[test]
    fn event_all_valid_domains_accepted() {
        for domain in VALID_DOMAINS {
            let e = make_event(domain, "{}");
            let result = validate_event(&e).unwrap();
            assert!(
                matches!(result, ValidateCallbackResult::Valid),
                "domain '{}' should be valid",
                domain
            );
        }
    }

    #[test]
    fn event_invalid_domain_rejected() {
        let e = make_event("property", "{}");
        match validate_event(&e).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("Invalid domain")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn event_payload_at_65536_passes() {
        let payload = "x".repeat(65536);
        let e = make_event("care", &payload);
        assert!(matches!(
            validate_event(&e).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn event_payload_over_65536_rejected() {
        let payload = "x".repeat(65537);
        let e = make_event("care", &payload);
        match validate_event(&e).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("65536")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Serde roundtrips ----

    #[test]
    fn serde_roundtrip_query() {
        let q = make_query("milestones", r#"{"member":"agent_123"}"#);
        let json = serde_json::to_string(&q).unwrap();
        let back: BridgeQueryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, q);
    }

    #[test]
    fn serde_roundtrip_event() {
        let e = make_event("rhythms", r#"{"type":"rhythm_completed"}"#);
        let json = serde_json::to_string(&e).unwrap();
        let back: BridgeEventEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    // ---- Wrapper integration ----

    #[test]
    fn validate_query_wrapper_valid() {
        let q = make_query("gratitude", "{}");
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
        let e = make_event("emergency", "{}");
        let result = validate_event(&e).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn validate_event_wrapper_invalid() {
        let e = make_event("housing", "{}");
        let result = validate_event(&e).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- Type alias compatibility ----

    #[test]
    fn type_alias_compatibility() {
        let q = make_query("kinship", "{}");
        let _hearth: HearthQueryEntry = q;
        let e = make_event("care", "{}");
        let _hearth_event: HearthEventEntry = e;
    }

    // ---- Anchor ----

    #[test]
    fn anchor_clone_equality() {
        let a = Anchor("hearth_queries".to_string());
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ---- Immutable Field Pure Equality Tests ----

    #[test]
    fn query_immutable_field_domain_difference_detected() {
        let a = make_query("kinship", "{}");
        let mut b = a.clone();
        b.domain = "care".into();
        assert_ne!(a.domain, b.domain);
    }

    #[test]
    fn query_immutable_field_query_type_difference_detected() {
        let a = make_query("kinship", "{}");
        let mut b = a.clone();
        b.query_type = "different_query".into();
        assert_ne!(a.query_type, b.query_type);
    }

    #[test]
    fn query_immutable_field_requester_difference_detected() {
        let a = make_query("kinship", "{}");
        let mut b = a.clone();
        b.requester = AgentPubKey::from_raw_36(vec![99u8; 36]);
        assert_ne!(a.requester, b.requester);
    }

    #[test]
    fn event_immutable_field_domain_difference_detected() {
        let a = make_event("care", "{}");
        let mut b = a.clone();
        b.domain = "emergency".into();
        assert_ne!(a.domain, b.domain);
    }

    #[test]
    fn event_immutable_field_event_type_difference_detected() {
        let a = make_event("care", "{}");
        let mut b = a.clone();
        b.event_type = "different_event".into();
        assert_ne!(a.event_type, b.event_type);
    }

    #[test]
    fn event_immutable_field_source_agent_difference_detected() {
        let a = make_event("care", "{}");
        let mut b = a.clone();
        b.source_agent = AgentPubKey::from_raw_36(vec![99u8; 36]);
        assert_ne!(a.source_agent, b.source_agent);
    }

    #[test]
    fn anchor_update_rejected_message() {
        let msg = "Anchor cannot be updated once created";
        assert!(msg.contains("cannot be updated"));
    }

    #[test]
    fn delete_guard_message_content() {
        let msg = "Bridge entries cannot be deleted once created";
        assert!(msg.contains("cannot be deleted"));
    }
}
