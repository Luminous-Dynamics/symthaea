//! Mycelix Bridge Common — Shared dispatch types and utilities
//!
//! Provides the cross-domain dispatch primitives used by both the
//! Commons and Civic cluster bridge zomes. Each cluster's bridge
//! coordinator imports these types and calls `dispatch_call_checked()`
//! with its own allowlist.

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Dispatch types
// ============================================================================

/// Input for dispatching a call to any domain zome within a cluster DNA.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DispatchInput {
    /// Target zome name (e.g., "property_registry", "justice_cases").
    /// Must be in the cluster's allowed zomes list.
    pub zome: String,
    /// Target function name (e.g., "verify_ownership", "get_property").
    pub fn_name: String,
    /// MessagePack-serialized input payload. Use `()` serialized for no-arg functions.
    pub payload: Vec<u8>,
}

/// Result of a dispatched cross-domain call.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DispatchResult {
    /// Whether the call succeeded.
    pub success: bool,
    /// MessagePack-serialized response payload (on success).
    pub response: Option<Vec<u8>>,
    /// Error message (on failure).
    pub error: Option<String>,
}

/// Input for resolving a query with a result.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResolveQueryInput {
    pub query_hash: ActionHash,
    pub result: String,
    pub success: bool,
}

/// Query for events by type within a domain.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EventTypeQuery {
    pub domain: String,
    pub event_type: String,
}

/// Health status for a cluster bridge.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

// ============================================================================
// Dispatch logic
// ============================================================================

/// Dispatch a synchronous call to a domain zome, with allowlist validation.
///
/// This is the core cross-domain integration primitive. It validates the
/// target zome against the provided allowlist, then uses
/// `call(CallTargetCell::Local, ...)` to invoke the function directly
/// within the same DNA.
///
/// The `payload` field in `DispatchInput` must already be MessagePack-encoded.
/// We bypass `ExternIO::encode()` to avoid double-serialization.
pub fn dispatch_call_checked(
    input: &DispatchInput,
    allowed_zomes: &[&str],
) -> ExternResult<DispatchResult> {
    if !allowed_zomes.contains(&input.zome.as_str()) {
        return Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!(
                "Zome '{}' is not in the allowed dispatch list. Valid zomes: {:?}",
                input.zome, allowed_zomes
            )),
        });
    }

    let payload = ExternIO(input.payload.clone());

    let result = HDK.with(|h| {
        h.borrow().call(vec![Call::new(
            CallTarget::ConductorCell(CallTargetCell::Local),
            ZomeName::from(input.zome.as_str()),
            FunctionName::from(input.fn_name.as_str()),
            None,
            payload,
        )])
    });

    match result {
        Ok(responses) => match responses.into_iter().next() {
            Some(ZomeCallResponse::Ok(extern_io)) => Ok(DispatchResult {
                success: true,
                response: Some(extern_io.0),
                error: None,
            }),
            Some(ZomeCallResponse::NetworkError(err)) => Ok(DispatchResult {
                success: false,
                response: None,
                error: Some(format!("Network error: {}", err)),
            }),
            Some(other) => Ok(DispatchResult {
                success: false,
                response: None,
                error: Some(format!("Zome call rejected: {:?}", other)),
            }),
            None => Ok(DispatchResult {
                success: false,
                response: None,
                error: Some("No response from zome call".into()),
            }),
        },
        Err(e) => Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!("Call failed: {:?}", e)),
        }),
    }
}

// ============================================================================
// Utilities
// ============================================================================

/// Convert links to their target records, skipping any that have been deleted.
pub fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    // dispatch_call_checked: the disallowed-zome path returns before
    // touching HDK, so we can test it without a running conductor.

    #[test]
    fn dispatch_rejects_disallowed_zome() {
        let input = DispatchInput {
            zome: "evil_zome".into(),
            fn_name: "steal_data".into(),
            payload: vec![],
        };
        let allowed = &["property_registry", "housing_units"];
        let result = dispatch_call_checked(&input, allowed).unwrap();
        assert!(!result.success);
        assert!(result.response.is_none());
        let err = result.error.unwrap();
        assert!(err.contains("not in the allowed dispatch list"));
        assert!(err.contains("evil_zome"));
    }

    #[test]
    fn dispatch_rejects_empty_allowlist() {
        let input = DispatchInput {
            zome: "property_registry".into(),
            fn_name: "get_property".into(),
            payload: vec![],
        };
        let result = dispatch_call_checked(&input, &[]).unwrap();
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn dispatch_rejects_similar_zome_name() {
        let input = DispatchInput {
            zome: "property_registry_evil".into(),
            fn_name: "get_property".into(),
            payload: vec![],
        };
        let allowed = &["property_registry"];
        let result = dispatch_call_checked(&input, allowed).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn dispatch_error_lists_valid_zomes() {
        let input = DispatchInput {
            zome: "bad".into(),
            fn_name: "fn".into(),
            payload: vec![],
        };
        let allowed = &["alpha", "beta", "gamma"];
        let result = dispatch_call_checked(&input, allowed).unwrap();
        let err = result.error.unwrap();
        assert!(err.contains("alpha"));
        assert!(err.contains("beta"));
        assert!(err.contains("gamma"));
    }

    // Type serde roundtrips

    #[test]
    fn dispatch_input_serde_roundtrip() {
        let input = DispatchInput {
            zome: "property_registry".into(),
            fn_name: "get_property".into(),
            payload: vec![1, 2, 3, 4],
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: DispatchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input.zome, input2.zome);
        assert_eq!(input.fn_name, input2.fn_name);
        assert_eq!(input.payload, input2.payload);
    }

    #[test]
    fn dispatch_result_success_serde_roundtrip() {
        let result = DispatchResult {
            success: true,
            response: Some(vec![10, 20, 30]),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: DispatchResult = serde_json::from_str(&json).unwrap();
        assert!(r2.success);
        assert_eq!(r2.response, Some(vec![10, 20, 30]));
        assert!(r2.error.is_none());
    }

    #[test]
    fn dispatch_result_error_serde_roundtrip() {
        let result = DispatchResult {
            success: false,
            response: None,
            error: Some("something failed".into()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: DispatchResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.success);
        assert!(r2.response.is_none());
        assert_eq!(r2.error.as_deref(), Some("something failed"));
    }

    #[test]
    fn event_type_query_serde_roundtrip() {
        let q = EventTypeQuery {
            domain: "housing".into(),
            event_type: "lease_created".into(),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: EventTypeQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.domain, q2.domain);
        assert_eq!(q.event_type, q2.event_type);
    }

    #[test]
    fn bridge_health_serde_roundtrip() {
        let h = BridgeHealth {
            healthy: true,
            agent: "uhCAk_test_agent".into(),
            total_events: 42,
            total_queries: 7,
            domains: vec!["property".into(), "housing".into()],
        };
        let json = serde_json::to_string(&h).unwrap();
        let h2: BridgeHealth = serde_json::from_str(&json).unwrap();
        assert!(h2.healthy);
        assert_eq!(h2.total_events, 42);
        assert_eq!(h2.total_queries, 7);
        assert_eq!(h2.domains.len(), 2);
    }
}
