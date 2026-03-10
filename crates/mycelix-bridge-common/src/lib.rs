//! Mycelix Bridge Common — Shared dispatch types and utilities
//!
//! Provides the cross-domain dispatch primitives used by both the
//! Commons and Civic cluster bridge zomes. Each cluster's bridge
//! coordinator imports these types and calls `dispatch_call_checked()`
//! with its own allowlist.
//!
//! ## Consciousness Thresholds (always available)
//!
//! The `consciousness_thresholds` module is the canonical source of truth for all
//! consciousness-related thresholds across the Mycelix ecosystem. It does not
//! depend on HDK and can be used by any Rust crate.

pub mod consciousness_thresholds;
/// Backward-compatible module alias — allows `mycelix_bridge_common::phi_thresholds::*` paths.
pub use consciousness_thresholds as phi_thresholds;
pub use consciousness_thresholds::{ConsciousnessThresholds, PhiThresholds};

pub mod consciousness_profile;
pub use consciousness_profile::{
    bootstrap_credential, evaluate_bootstrap_governance, evaluate_governance, gate_consciousness,
    is_bootstrap_eligible, needs_refresh, requirement_for_basic, requirement_for_constitutional,
    requirement_for_guardian, requirement_for_proposal, requirement_for_voting, should_audit,
    ConsciousnessCredential, ConsciousnessProfile, ConsciousnessTier, GateAuditInput,
    GovernanceAuditFilter, GovernanceAuditResult, GovernanceEligibility, GovernanceRequirement,
    GRACE_PERIOD_US, REFRESH_WINDOW_US,
};

pub mod collective_phi;
pub use collective_phi::{
    AgentConsciousnessVector, CollectivePhiEngine, CollectivePhiResult, COLLECTIVE_PHI_MAX_SYNC,
};

pub mod routing;
pub use routing::{
    resolve_civic_zome, resolve_commons_zome, BridgeDomain, CivicZome, CommonsZome,
    CrossClusterRole, CIVIC_DOMAINS, COMMONS_DOMAINS,
};

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

/// Wrapper for cross-cluster dispatch with audit correlation.
///
/// When a coordinator initiates a cross-cluster action, it generates a
/// correlation ID and wraps the dispatch so both sides can log the same
/// ID in their audit trail.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CorrelatedDispatch {
    /// Unique correlation ID linking audit events across clusters.
    /// Format: "<agent_hex_prefix>:<timestamp_us>"
    pub correlation_id: String,
    /// Target zome in the other cluster.
    pub target_zome: String,
    /// Target function name.
    pub target_fn: String,
    /// JSON-serialized payload.
    pub payload: String,
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

/// Cross-cluster dispatch to commons with automatic sub-cluster role resolution.
///
/// Instead of using a fixed `"commons"` role, this resolves the target zome
/// to either `"commons_land"` or `"commons_care"` based on which sub-cluster
/// DNA contains that zome.
///
/// This is needed because the commons cluster is split into two DNA roles
/// in the unified hApp to fit under Holochain's 16MB DNA limit.
pub fn dispatch_call_cross_cluster_commons(
    input: &CrossClusterDispatchInput,
    allowed_zomes: &[&str],
) -> ExternResult<DispatchResult> {
    // Resolve which sub-cluster this zome belongs to
    let role = CommonsZome::resolve_role(&input.zome).unwrap_or("commons_land");

    let routed_input = CrossClusterDispatchInput {
        role: role.to_string(),
        zome: input.zome.clone(),
        fn_name: input.fn_name.clone(),
        payload: input.payload.clone(),
    };
    dispatch_call_cross_cluster(&routed_input, allowed_zomes)
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
// Cross-cluster emergency↔commons query types
// ============================================================================

/// Input for querying water safety in a disaster zone (emergency → water)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WaterSafetyQuery {
    pub area_lat: f64,
    pub area_lon: f64,
    pub radius_km: f64,
}

/// Result of a water safety query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WaterSafetyResult {
    pub safe_sources: u32,
    pub contaminated_sources: u32,
    pub total_sources: u32,
}

/// Input for querying food availability during an emergency (emergency → food)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmergencyFoodQuery {
    pub area_lat: f64,
    pub area_lon: f64,
    pub radius_km: f64,
    pub people_count: u32,
}

/// Result of an emergency food availability query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmergencyFoodResult {
    pub available_kg: f64,
    pub distribution_points: u32,
    pub estimated_days_supply: f64,
}

/// Input for querying shelter capacity during an emergency (emergency → housing)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ShelterCapacityQuery {
    pub area_lat: f64,
    pub area_lon: f64,
    pub radius_km: f64,
    pub beds_needed: u32,
}

/// Result of a shelter capacity query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ShelterCapacityResult {
    pub available_beds: u32,
    pub total_shelters: u32,
    pub nearest_shelter_km: f64,
}

/// Input for querying available care providers during an emergency (emergency → care)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmergencyCareQuery {
    pub area_lat: f64,
    pub area_lon: f64,
    pub skill_needed: String,
    pub urgency_level: u8,
}

/// Result of an emergency care provider query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmergencyCareResult {
    pub available_providers: u32,
    pub nearest_provider_km: f64,
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
// Typed hearth↔other cluster query/result helpers
// ============================================================================

/// Input for querying hearth membership (civic/commons → hearth)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HearthMemberQuery {
    pub hearth_hash: ActionHash,
    pub agent: AgentPubKey,
}

/// Result of a hearth membership query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HearthMemberResult {
    pub is_member: bool,
    pub role: Option<String>,
    pub display_name: Option<String>,
    pub error: Option<String>,
}

/// Input for querying hearth care availability (commons → hearth)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HearthCareQuery {
    pub hearth_hash: ActionHash,
    pub care_type: Option<String>,
}

/// Result of a hearth care query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HearthCareResult {
    pub available_caregivers: u32,
    pub active_schedules: u32,
    pub error: Option<String>,
}

/// Input for querying hearth emergency status (civic → hearth)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HearthEmergencyQuery {
    pub hearth_hash: ActionHash,
}

/// Result of a hearth emergency status query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HearthEmergencyResult {
    pub has_active_alerts: bool,
    pub active_alert_count: u32,
    pub members_checked_in: u32,
    pub members_missing: u32,
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
        assert_eq!(
            r2.nearest_market.as_deref(),
            Some("Southside Farmers Market")
        );
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

    // Emergency↔Commons cross-cluster type serde tests

    #[test]
    fn water_safety_query_serde_roundtrip() {
        let q = WaterSafetyQuery {
            area_lat: 32.9483,
            area_lon: -96.7299,
            radius_km: 10.0,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: WaterSafetyQuery = serde_json::from_str(&json).unwrap();
        assert!((q2.area_lat - 32.9483).abs() < 1e-4);
        assert!((q2.area_lon - (-96.7299)).abs() < 1e-4);
        assert!((q2.radius_km - 10.0).abs() < 1e-6);
    }

    #[test]
    fn water_safety_result_serde_roundtrip() {
        let r = WaterSafetyResult {
            safe_sources: 8,
            contaminated_sources: 2,
            total_sources: 10,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: WaterSafetyResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.safe_sources, 8);
        assert_eq!(r2.contaminated_sources, 2);
        assert_eq!(r2.total_sources, 10);
    }

    #[test]
    fn water_safety_result_all_contaminated() {
        let r = WaterSafetyResult {
            safe_sources: 0,
            contaminated_sources: 5,
            total_sources: 5,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: WaterSafetyResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.safe_sources, 0);
        assert_eq!(r2.contaminated_sources, 5);
    }

    #[test]
    fn emergency_food_query_serde_roundtrip() {
        let q = EmergencyFoodQuery {
            area_lat: 29.7604,
            area_lon: -95.3698,
            radius_km: 25.0,
            people_count: 500,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: EmergencyFoodQuery = serde_json::from_str(&json).unwrap();
        assert!((q2.area_lat - 29.7604).abs() < 1e-4);
        assert!((q2.area_lon - (-95.3698)).abs() < 1e-4);
        assert!((q2.radius_km - 25.0).abs() < 1e-6);
        assert_eq!(q2.people_count, 500);
    }

    #[test]
    fn emergency_food_result_serde_roundtrip() {
        let r = EmergencyFoodResult {
            available_kg: 2500.5,
            distribution_points: 4,
            estimated_days_supply: 3.5,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: EmergencyFoodResult = serde_json::from_str(&json).unwrap();
        assert!((r2.available_kg - 2500.5).abs() < 1e-6);
        assert_eq!(r2.distribution_points, 4);
        assert!((r2.estimated_days_supply - 3.5).abs() < 1e-6);
    }

    #[test]
    fn emergency_food_result_zero_supply() {
        let r = EmergencyFoodResult {
            available_kg: 0.0,
            distribution_points: 0,
            estimated_days_supply: 0.0,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: EmergencyFoodResult = serde_json::from_str(&json).unwrap();
        assert!((r2.available_kg).abs() < 1e-6);
        assert_eq!(r2.distribution_points, 0);
        assert!((r2.estimated_days_supply).abs() < 1e-6);
    }

    #[test]
    fn shelter_capacity_query_serde_roundtrip() {
        let q = ShelterCapacityQuery {
            area_lat: 30.2672,
            area_lon: -97.7431,
            radius_km: 15.0,
            beds_needed: 200,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: ShelterCapacityQuery = serde_json::from_str(&json).unwrap();
        assert!((q2.area_lat - 30.2672).abs() < 1e-4);
        assert!((q2.area_lon - (-97.7431)).abs() < 1e-4);
        assert!((q2.radius_km - 15.0).abs() < 1e-6);
        assert_eq!(q2.beds_needed, 200);
    }

    #[test]
    fn shelter_capacity_result_serde_roundtrip() {
        let r = ShelterCapacityResult {
            available_beds: 150,
            total_shelters: 3,
            nearest_shelter_km: 2.4,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: ShelterCapacityResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.available_beds, 150);
        assert_eq!(r2.total_shelters, 3);
        assert!((r2.nearest_shelter_km - 2.4).abs() < 1e-6);
    }

    #[test]
    fn shelter_capacity_result_no_shelters() {
        let r = ShelterCapacityResult {
            available_beds: 0,
            total_shelters: 0,
            nearest_shelter_km: 0.0,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: ShelterCapacityResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.available_beds, 0);
        assert_eq!(r2.total_shelters, 0);
    }

    #[test]
    fn emergency_care_query_serde_roundtrip() {
        let q = EmergencyCareQuery {
            area_lat: 32.7767,
            area_lon: -96.7970,
            skill_needed: "trauma_surgeon".into(),
            urgency_level: 5,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: EmergencyCareQuery = serde_json::from_str(&json).unwrap();
        assert!((q2.area_lat - 32.7767).abs() < 1e-4);
        assert!((q2.area_lon - (-96.7970)).abs() < 1e-4);
        assert_eq!(q2.skill_needed, "trauma_surgeon");
        assert_eq!(q2.urgency_level, 5);
    }

    #[test]
    fn emergency_care_query_low_urgency() {
        let q = EmergencyCareQuery {
            area_lat: 0.0,
            area_lon: 0.0,
            skill_needed: "first_aid".into(),
            urgency_level: 1,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: EmergencyCareQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.urgency_level, 1);
        assert_eq!(q2.skill_needed, "first_aid");
    }

    #[test]
    fn emergency_care_result_serde_roundtrip() {
        let r = EmergencyCareResult {
            available_providers: 7,
            nearest_provider_km: 1.2,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: EmergencyCareResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.available_providers, 7);
        assert!((r2.nearest_provider_km - 1.2).abs() < 1e-6);
    }

    #[test]
    fn emergency_care_result_no_providers() {
        let r = EmergencyCareResult {
            available_providers: 0,
            nearest_provider_km: 0.0,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: EmergencyCareResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.available_providers, 0);
    }

    // Boundary validation tests for emergency↔commons types

    #[test]
    fn water_safety_query_extreme_coordinates() {
        let q = WaterSafetyQuery {
            area_lat: 90.0,
            area_lon: 180.0,
            radius_km: 0.001,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: WaterSafetyQuery = serde_json::from_str(&json).unwrap();
        assert!((q2.area_lat - 90.0).abs() < 1e-6);
        assert!((q2.area_lon - 180.0).abs() < 1e-6);
    }

    #[test]
    fn water_safety_query_negative_coordinates() {
        let q = WaterSafetyQuery {
            area_lat: -90.0,
            area_lon: -180.0,
            radius_km: 100.0,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: WaterSafetyQuery = serde_json::from_str(&json).unwrap();
        assert!((q2.area_lat - (-90.0)).abs() < 1e-6);
        assert!((q2.area_lon - (-180.0)).abs() < 1e-6);
    }

    #[test]
    fn shelter_capacity_query_zero_beds_needed() {
        let q = ShelterCapacityQuery {
            area_lat: 0.0,
            area_lon: 0.0,
            radius_km: 1.0,
            beds_needed: 0,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: ShelterCapacityQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.beds_needed, 0);
    }

    #[test]
    fn emergency_food_query_zero_people() {
        let q = EmergencyFoodQuery {
            area_lat: 0.0,
            area_lon: 0.0,
            radius_km: 1.0,
            people_count: 0,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: EmergencyFoodQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.people_count, 0);
    }

    #[test]
    fn emergency_care_query_max_urgency_level() {
        let q = EmergencyCareQuery {
            area_lat: 0.0,
            area_lon: 0.0,
            skill_needed: "any".into(),
            urgency_level: 255,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: EmergencyCareQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.urgency_level, 255);
    }

    #[test]
    fn emergency_care_query_empty_skill() {
        let q = EmergencyCareQuery {
            area_lat: 0.0,
            area_lon: 0.0,
            skill_needed: "".into(),
            urgency_level: 3,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: EmergencyCareQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.skill_needed, "");
    }

    // Hearth↔cluster typed helper serde tests

    #[test]
    fn hearth_member_query_serde_roundtrip() {
        let q = HearthMemberQuery {
            hearth_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            agent: AgentPubKey::from_raw_36(vec![2u8; 36]),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: HearthMemberQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.hearth_hash, q2.hearth_hash);
        assert_eq!(q.agent, q2.agent);
    }

    #[test]
    fn hearth_member_result_found() {
        let r = HearthMemberResult {
            is_member: true,
            role: Some("Adult".into()),
            display_name: Some("Alice".into()),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: HearthMemberResult = serde_json::from_str(&json).unwrap();
        assert!(r2.is_member);
        assert_eq!(r2.role.as_deref(), Some("Adult"));
        assert_eq!(r2.display_name.as_deref(), Some("Alice"));
        assert!(r2.error.is_none());
    }

    #[test]
    fn hearth_member_result_not_found() {
        let r = HearthMemberResult {
            is_member: false,
            role: None,
            display_name: None,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: HearthMemberResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.is_member);
        assert!(r2.role.is_none());
    }

    #[test]
    fn hearth_care_query_serde_roundtrip() {
        let q = HearthCareQuery {
            hearth_hash: ActionHash::from_raw_36(vec![3u8; 36]),
            care_type: Some("Childcare".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: HearthCareQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.hearth_hash, q2.hearth_hash);
        assert_eq!(q2.care_type.as_deref(), Some("Childcare"));
    }

    #[test]
    fn hearth_care_query_no_filter() {
        let q = HearthCareQuery {
            hearth_hash: ActionHash::from_raw_36(vec![4u8; 36]),
            care_type: None,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: HearthCareQuery = serde_json::from_str(&json).unwrap();
        assert!(q2.care_type.is_none());
    }

    #[test]
    fn hearth_care_result_serde_roundtrip() {
        let r = HearthCareResult {
            available_caregivers: 3,
            active_schedules: 7,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: HearthCareResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.available_caregivers, 3);
        assert_eq!(r2.active_schedules, 7);
        assert!(r2.error.is_none());
    }

    #[test]
    fn hearth_emergency_query_serde_roundtrip() {
        let q = HearthEmergencyQuery {
            hearth_hash: ActionHash::from_raw_36(vec![5u8; 36]),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: HearthEmergencyQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.hearth_hash, q2.hearth_hash);
    }

    #[test]
    fn hearth_emergency_result_active() {
        let r = HearthEmergencyResult {
            has_active_alerts: true,
            active_alert_count: 2,
            members_checked_in: 4,
            members_missing: 1,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: HearthEmergencyResult = serde_json::from_str(&json).unwrap();
        assert!(r2.has_active_alerts);
        assert_eq!(r2.active_alert_count, 2);
        assert_eq!(r2.members_checked_in, 4);
        assert_eq!(r2.members_missing, 1);
        assert!(r2.error.is_none());
    }

    #[test]
    fn hearth_emergency_result_clear() {
        let r = HearthEmergencyResult {
            has_active_alerts: false,
            active_alert_count: 0,
            members_checked_in: 5,
            members_missing: 0,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: HearthEmergencyResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.has_active_alerts);
        assert_eq!(r2.active_alert_count, 0);
        assert_eq!(r2.members_missing, 0);
    }
}
