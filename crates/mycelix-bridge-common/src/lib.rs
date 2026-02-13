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
// Cross-cluster dispatch (inter-DNA within the same hApp)
// ============================================================================

/// Input for dispatching a call to a zome in another DNA within the same hApp.
///
/// Used for commons↔civic cross-cluster communication.  The `role` field
/// identifies the target DNA by its hApp role name (e.g., `"commons"` or
/// `"civic"`).  The call is routed via `CallTargetCell::OtherRole`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrossClusterDispatchInput {
    /// hApp role name of the target DNA (e.g., "commons" or "civic").
    pub role: String,
    /// Target zome name within the other DNA.
    pub zome: String,
    /// Target function name.
    pub fn_name: String,
    /// MessagePack-serialized input payload.
    pub payload: Vec<u8>,
}

/// Dispatch a synchronous call to a zome in another DNA, with allowlist
/// validation.
///
/// This is the cross-cluster counterpart of [`dispatch_call_checked`].
/// Instead of `CallTargetCell::Local`, it uses
/// `CallTargetCell::OtherRole(role)` to reach a different DNA within the
/// same installed hApp.  The target zome must be in `allowed_zomes`.
pub fn dispatch_call_cross_cluster(
    input: &CrossClusterDispatchInput,
    allowed_zomes: &[&str],
) -> ExternResult<DispatchResult> {
    if !allowed_zomes.contains(&input.zome.as_str()) {
        return Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!(
                "Zome '{}' is not in the allowed cross-cluster dispatch list. Valid zomes: {:?}",
                input.zome, allowed_zomes
            )),
        });
    }

    let payload = ExternIO(input.payload.clone());

    let result = HDK.with(|h| {
        h.borrow().call(vec![Call::new(
            CallTarget::ConductorCell(CallTargetCell::OtherRole(input.role.clone())),
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
                error: Some(format!("Cross-cluster network error: {}", err)),
            }),
            Some(other) => Ok(DispatchResult {
                success: false,
                response: None,
                error: Some(format!("Cross-cluster call rejected: {:?}", other)),
            }),
            None => Ok(DispatchResult {
                success: false,
                response: None,
                error: Some("No response from cross-cluster call".into()),
            }),
        },
        Err(e) => Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!("Cross-cluster call failed: {:?}", e)),
        }),
    }
}

// ============================================================================
// Rate limiting constants
// ============================================================================

/// Maximum dispatch calls per agent within the rate limit window.
pub const RATE_LIMIT_MAX_DISPATCH: usize = 100;

/// Rate limit window in seconds.
pub const RATE_LIMIT_WINDOW_SECS: i64 = 60;

/// Check whether the number of recent dispatches exceeds the rate limit.
///
/// Returns `Ok(())` if within limits, or an error string if exceeded.
/// This is a pure validation function — the caller is responsible for
/// counting recent dispatches (via `get_links` on the agent's rate-limit
/// links) and passing the count here.
pub fn check_rate_limit_count(recent_count: usize) -> Result<(), String> {
    if recent_count >= RATE_LIMIT_MAX_DISPATCH {
        Err(format!(
            "Rate limit exceeded: {} dispatches in {}s (max {})",
            recent_count, RATE_LIMIT_WINDOW_SECS, RATE_LIMIT_MAX_DISPATCH
        ))
    } else {
        Ok(())
    }
}

// ============================================================================
// Typed cross-domain dispatch helpers
// ============================================================================

/// Input for verifying property ownership (commons: housing → property)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PropertyOwnershipQuery {
    pub property_id: String,
    pub requester_did: String,
}

/// Result of a property ownership verification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PropertyOwnershipResult {
    pub is_owner: bool,
    pub owner_did: Option<String>,
    pub error: Option<String>,
}

/// Input for querying care provider availability (commons: mutualaid → care)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CareAvailabilityQuery {
    pub skill_needed: String,
    pub location: Option<String>,
}

/// Result of a care availability query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CareAvailabilityResult {
    pub available_count: u32,
    pub recommendation: String,
    pub error: Option<String>,
}

/// Input for checking active cases in an area (civic: emergency → justice)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JusticeAreaQuery {
    pub area: String,
    pub case_type: Option<String>,
}

/// Result of an area case query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JusticeAreaResult {
    pub active_cases: u32,
    pub recommendation: String,
    pub error: Option<String>,
}

/// Input for checking factcheck status (civic: justice → media)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FactcheckStatusQuery {
    pub claim_id: String,
}

/// Result of a factcheck status query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FactcheckStatusResult {
    pub has_factcheck: bool,
    pub verdict: Option<String>,
    pub error: Option<String>,
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

    // Cross-cluster dispatch validation tests

    #[test]
    fn cross_cluster_rejects_disallowed_zome() {
        let input = CrossClusterDispatchInput {
            role: "civic".into(),
            zome: "evil_zome".into(),
            fn_name: "steal_data".into(),
            payload: vec![],
        };
        let allowed = &["justice_cases", "emergency_incidents"];
        let result = dispatch_call_cross_cluster(&input, allowed).unwrap();
        assert!(!result.success);
        assert!(result.response.is_none());
        let err = result.error.unwrap();
        assert!(err.contains("not in the allowed cross-cluster dispatch list"));
        assert!(err.contains("evil_zome"));
    }

    #[test]
    fn cross_cluster_rejects_empty_allowlist() {
        let input = CrossClusterDispatchInput {
            role: "commons".into(),
            zome: "property_registry".into(),
            fn_name: "get_property".into(),
            payload: vec![],
        };
        let result = dispatch_call_cross_cluster(&input, &[]).unwrap();
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn cross_cluster_rejects_similar_zome_name() {
        let input = CrossClusterDispatchInput {
            role: "civic".into(),
            zome: "justice_cases_evil".into(),
            fn_name: "get_case".into(),
            payload: vec![],
        };
        let allowed = &["justice_cases"];
        let result = dispatch_call_cross_cluster(&input, allowed).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn cross_cluster_error_lists_valid_zomes() {
        let input = CrossClusterDispatchInput {
            role: "civic".into(),
            zome: "bad".into(),
            fn_name: "fn".into(),
            payload: vec![],
        };
        let allowed = &["justice_cases", "emergency_incidents", "media_publication"];
        let result = dispatch_call_cross_cluster(&input, allowed).unwrap();
        let err = result.error.unwrap();
        assert!(err.contains("justice_cases"));
        assert!(err.contains("emergency_incidents"));
        assert!(err.contains("media_publication"));
    }

    #[test]
    fn cross_cluster_dispatch_input_serde_roundtrip() {
        let input = CrossClusterDispatchInput {
            role: "civic".into(),
            zome: "justice_cases".into(),
            fn_name: "get_case".into(),
            payload: vec![5, 6, 7],
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CrossClusterDispatchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input.role, input2.role);
        assert_eq!(input.zome, input2.zome);
        assert_eq!(input.fn_name, input2.fn_name);
        assert_eq!(input.payload, input2.payload);
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

    // Rate limit tests

    #[test]
    fn rate_limit_zero_calls_passes() {
        assert!(check_rate_limit_count(0).is_ok());
    }

    #[test]
    fn rate_limit_under_max_passes() {
        assert!(check_rate_limit_count(99).is_ok());
    }

    #[test]
    fn rate_limit_at_max_rejects() {
        let err = check_rate_limit_count(RATE_LIMIT_MAX_DISPATCH).unwrap_err();
        assert!(err.contains("Rate limit exceeded"));
    }

    #[test]
    fn rate_limit_over_max_rejects() {
        let err = check_rate_limit_count(1000).unwrap_err();
        assert!(err.contains("Rate limit exceeded"));
        assert!(err.contains("1000"));
    }

    #[test]
    fn rate_limit_error_includes_window() {
        let err = check_rate_limit_count(200).unwrap_err();
        assert!(err.contains(&format!("{}s", RATE_LIMIT_WINDOW_SECS)));
    }

    // Typed helper serde tests

    #[test]
    fn property_ownership_query_serde_roundtrip() {
        let q = PropertyOwnershipQuery {
            property_id: "PROP-001".into(),
            requester_did: "did:mycelix:abc".into(),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: PropertyOwnershipQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.property_id, q2.property_id);
        assert_eq!(q.requester_did, q2.requester_did);
    }

    #[test]
    fn property_ownership_result_serde_roundtrip() {
        let r = PropertyOwnershipResult {
            is_owner: true,
            owner_did: Some("did:mycelix:owner".into()),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: PropertyOwnershipResult = serde_json::from_str(&json).unwrap();
        assert!(r2.is_owner);
        assert_eq!(r2.owner_did, Some("did:mycelix:owner".into()));
    }

    #[test]
    fn care_availability_query_serde_roundtrip() {
        let q = CareAvailabilityQuery {
            skill_needed: "nursing".into(),
            location: None,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: CareAvailabilityQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.skill_needed, q2.skill_needed);
        assert!(q2.location.is_none());
    }

    #[test]
    fn justice_area_query_serde_roundtrip() {
        let q = JusticeAreaQuery {
            area: "north-side".into(),
            case_type: Some("civil".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: JusticeAreaQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.area, q2.area);
        assert_eq!(q.case_type, q2.case_type);
    }

    #[test]
    fn factcheck_status_query_serde_roundtrip() {
        let q = FactcheckStatusQuery {
            claim_id: "CL-42".into(),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: FactcheckStatusQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.claim_id, q2.claim_id);
    }

    #[test]
    fn factcheck_status_result_serde_roundtrip() {
        let r = FactcheckStatusResult {
            has_factcheck: true,
            verdict: Some("verified".into()),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: FactcheckStatusResult = serde_json::from_str(&json).unwrap();
        assert!(r2.has_factcheck);
        assert_eq!(r2.verdict, Some("verified".into()));
    }
}
