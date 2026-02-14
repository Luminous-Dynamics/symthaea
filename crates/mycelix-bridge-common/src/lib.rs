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

/// Input for querying food availability (commons: emergency → food, mutualaid → food)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FoodAvailabilityQuery {
    pub product_name: Option<String>,
    pub market_type: Option<String>,
    pub max_distance_km: Option<f64>,
}

/// Result of a food availability query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FoodAvailabilityResult {
    pub available_listings: u32,
    pub nearest_market: Option<String>,
    pub error: Option<String>,
}

/// Input for querying transport routes (commons: mutualaid → transport, care → transport)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransportRouteQuery {
    pub origin_lat: f64,
    pub origin_lon: f64,
    pub destination_lat: f64,
    pub destination_lon: f64,
    pub mode: Option<String>,
}

/// Result of a transport route query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransportRouteResult {
    pub route_count: u32,
    pub estimated_minutes: Option<u32>,
    pub estimated_emissions_kg_co2: Option<f64>,
    pub error: Option<String>,
}

/// Input for querying carbon credits (commons: property → transport)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CarbonCreditQuery {
    pub agent_did: String,
}

/// Result of a carbon credit query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CarbonCreditResult {
    pub total_credits_kg_co2: f64,
    pub trips_logged: u32,
    pub error: Option<String>,
}

// ============================================================================
// Audit trail query types
// ============================================================================

/// Input for querying events within a time range, optionally filtered by domain and type.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AuditTrailQuery {
    /// Start of the time range (inclusive), as microseconds since epoch.
    pub from_us: i64,
    /// End of the time range (inclusive), as microseconds since epoch.
    pub to_us: i64,
    /// Optional domain filter (e.g., "property", "justice").
    pub domain: Option<String>,
    /// Optional event type filter (e.g., "ownership_transferred").
    pub event_type: Option<String>,
}

/// Summary of a single audit trail entry (lightweight, no full record).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AuditTrailEntry {
    pub domain: String,
    pub event_type: String,
    pub source_agent: String,
    pub payload_preview: String,
    pub created_at_us: i64,
    pub action_hash: ActionHash,
}

/// Result of an audit trail query.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AuditTrailResult {
    pub entries: Vec<AuditTrailEntry>,
    pub total_matched: u32,
    pub query_from_us: i64,
    pub query_to_us: i64,
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

    // Audit trail type serde tests

    #[test]
    fn audit_trail_query_full_serde() {
        let q = AuditTrailQuery {
            from_us: 1_700_000_000_000_000,
            to_us: 1_700_001_000_000_000,
            domain: Some("property".into()),
            event_type: Some("ownership_transferred".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: AuditTrailQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.from_us, 1_700_000_000_000_000);
        assert_eq!(q2.domain.as_deref(), Some("property"));
        assert_eq!(q2.event_type.as_deref(), Some("ownership_transferred"));
    }

    #[test]
    fn audit_trail_query_no_filters() {
        let q = AuditTrailQuery {
            from_us: 0,
            to_us: i64::MAX,
            domain: None,
            event_type: None,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: AuditTrailQuery = serde_json::from_str(&json).unwrap();
        assert!(q2.domain.is_none());
        assert!(q2.event_type.is_none());
    }

    #[test]
    fn audit_trail_entry_serde() {
        let e = AuditTrailEntry {
            domain: "justice".into(),
            event_type: "case_filed".into(),
            source_agent: "uhCAk_agent1".into(),
            payload_preview: "{\"case_id\":\"CASE-1\"}".into(),
            created_at_us: 1_700_000_500_000_000,
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&e).unwrap();
        let e2: AuditTrailEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(e2.domain, "justice");
        assert_eq!(e2.event_type, "case_filed");
    }

    #[test]
    fn audit_trail_result_serde() {
        let r = AuditTrailResult {
            entries: vec![],
            total_matched: 0,
            query_from_us: 0,
            query_to_us: 1_000_000,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: AuditTrailResult = serde_json::from_str(&json).unwrap();
        assert!(r2.entries.is_empty());
        assert_eq!(r2.total_matched, 0);
    }

    // Food/Transport/Carbon typed helper serde tests

    #[test]
    fn food_availability_query_serde_roundtrip() {
        let q = FoodAvailabilityQuery {
            product_name: Some("tomatoes".into()),
            market_type: Some("FarmersMarket".into()),
            max_distance_km: Some(15.0),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: FoodAvailabilityQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.product_name.as_deref(), Some("tomatoes"));
        assert_eq!(q2.market_type.as_deref(), Some("FarmersMarket"));
        assert_eq!(q2.max_distance_km, Some(15.0));
    }

    #[test]
    fn food_availability_query_no_filters() {
        let q = FoodAvailabilityQuery {
            product_name: None,
            market_type: None,
            max_distance_km: None,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: FoodAvailabilityQuery = serde_json::from_str(&json).unwrap();
        assert!(q2.product_name.is_none());
    }

    #[test]
    fn food_availability_result_serde_roundtrip() {
        let r = FoodAvailabilityResult {
            available_listings: 12,
            nearest_market: Some("Southside Farmers Market".into()),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: FoodAvailabilityResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.available_listings, 12);
        assert_eq!(r2.nearest_market.as_deref(), Some("Southside Farmers Market"));
        assert!(r2.error.is_none());
    }

    #[test]
    fn transport_route_query_serde_roundtrip() {
        let q = TransportRouteQuery {
            origin_lat: 32.9483,
            origin_lon: -96.7299,
            destination_lat: 32.7767,
            destination_lon: -96.7970,
            mode: Some("Cycling".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: TransportRouteQuery = serde_json::from_str(&json).unwrap();
        assert!((q2.origin_lat - 32.9483).abs() < 1e-4);
        assert_eq!(q2.mode.as_deref(), Some("Cycling"));
    }

    #[test]
    fn transport_route_result_serde_roundtrip() {
        let r = TransportRouteResult {
            route_count: 3,
            estimated_minutes: Some(45),
            estimated_emissions_kg_co2: Some(0.0),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: TransportRouteResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.route_count, 3);
        assert_eq!(r2.estimated_minutes, Some(45));
        assert_eq!(r2.estimated_emissions_kg_co2, Some(0.0));
    }

    #[test]
    fn carbon_credit_query_serde_roundtrip() {
        let q = CarbonCreditQuery {
            agent_did: "did:mycelix:agent123".into(),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: CarbonCreditQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.agent_did, "did:mycelix:agent123");
    }

    #[test]
    fn carbon_credit_result_serde_roundtrip() {
        let r = CarbonCreditResult {
            total_credits_kg_co2: 127.5,
            trips_logged: 34,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: CarbonCreditResult = serde_json::from_str(&json).unwrap();
        assert!((r2.total_credits_kg_co2 - 127.5).abs() < 1e-6);
        assert_eq!(r2.trips_logged, 34);
        assert!(r2.error.is_none());
    }
}
