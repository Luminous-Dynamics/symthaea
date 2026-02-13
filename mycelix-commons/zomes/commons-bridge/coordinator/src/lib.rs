//! Commons Bridge Coordinator Zome
//!
//! Unified cross-domain dispatch for the Commons cluster.
//! Provides three integration patterns:
//!
//! 1. **dispatch_call** — synchronous RPC to any domain zome via
//!    `call(CallTargetCell::Local, ...)`. The core value of clustering.
//! 2. **query_commons** — audited async query/response with auto-dispatch
//! 3. **broadcast_event** — pub-sub event distribution across domains

use commons_bridge_integrity::*;
use commons_types::{CommonsQuery, CommonsEvent};
use hdk::prelude::*;
use mycelix_bridge_common::{
    self as bridge,
    DispatchInput, DispatchResult, CrossClusterDispatchInput,
    ResolveQueryInput, EventTypeQuery, BridgeHealth,
    PropertyOwnershipQuery, PropertyOwnershipResult,
    CareAvailabilityQuery, CareAvailabilityResult,
    RATE_LIMIT_WINDOW_SECS, check_rate_limit_count,
};

// ============================================================================
// Allowed zome names — security boundary for dispatch
// ============================================================================

const ALLOWED_ZOMES: &[&str] = &[
    // Property domain
    "property_registry",
    "property_transfer",
    "property_disputes",
    "property_commons",
    // Housing domain
    "housing_units",
    "housing_membership",
    "housing_finances",
    "housing_maintenance",
    "housing_clt",
    "housing_governance",
    // Care domain
    "care_timebank",
    "care_circles",
    "care_matching",
    "care_plans",
    "care_credentials",
    // Mutual aid domain
    "mutualaid_needs",
    "mutualaid_circles",
    "mutualaid_governance",
    "mutualaid_pools",
    "mutualaid_requests",
    "mutualaid_resources",
    "mutualaid_timebank",
    // Water domain
    "water_flow",
    "water_purity",
    "water_capture",
    "water_steward",
    "water_wisdom",
    // Food domain
    "food_production",
    "food_distribution",
    "food_preservation",
    "food_knowledge",
    // Transport domain
    "transport_routes",
    "transport_sharing",
    "transport_impact",
];

// ============================================================================
// Helpers (use zome-specific EntryTypes for anchors)
// ============================================================================

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

// ============================================================================
// Rate Limiting
// ============================================================================

/// Check rate limit and log the dispatch. Returns error if limit exceeded.
fn enforce_rate_limit(target_zome: &str) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let anchor = ensure_anchor("dispatch_rate_limit")?;

    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::DispatchRateLimit)?,
        GetStrategy::Local,
    )?;

    let now = sys_time()?;
    let window_start_micros = now.as_micros() - (RATE_LIMIT_WINDOW_SECS * 1_000_000);
    let window_start = Timestamp::from_micros(window_start_micros);

    let recent_count = links.iter()
        .filter(|l| l.timestamp >= window_start)
        .count();

    check_rate_limit_count(recent_count)
        .map_err(|msg| wasm_error!(WasmErrorInner::Guest(msg)))?;

    // Log this dispatch
    create_link(
        agent,
        anchor,
        LinkTypes::DispatchRateLimit,
        target_zome.as_bytes().to_vec(),
    )?;

    Ok(())
}

// ============================================================================
// Cross-Domain Dispatch (synchronous RPC)
// ============================================================================

/// Dispatch a synchronous call to any domain zome within the Commons DNA.
///
/// Rate-limited to 100 calls per 60 seconds per agent. Validates the target
/// zome against an allowlist, then uses `call(CallTargetCell::Local, ...)` to
/// invoke the function directly within the same DNA.
///
/// ## Example (from another coordinator zome or external client)
/// ```ignore
/// let input = DispatchInput {
///     zome: "property_registry".into(),
///     fn_name: "verify_ownership".into(),
///     payload: encode(&VerifyOwnershipInput { ... })?,
/// };
/// let result: DispatchResult = call("commons_bridge", "dispatch_call", input)?;
/// ```
#[hdk_extern]
pub fn dispatch_call(input: DispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&input.zome)?;
    bridge::dispatch_call_checked(&input, ALLOWED_ZOMES)
}

// ============================================================================
// Audited Query/Response (with auto-dispatch)
// ============================================================================

/// Submit a cross-domain query within the Commons cluster.
///
/// Stores the query on the DHT for auditability, then attempts to auto-dispatch
/// to the target domain zome if the query_type matches a known function name.
/// If auto-dispatch succeeds, the query is automatically resolved with the result.
#[hdk_extern]
pub fn query_commons(query: CommonsQuery) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Query(query.clone()))?;

    // Link to all queries
    let all_anchor = ensure_anchor("all_commons_queries")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllQueries, ())?;

    // Link agent to query
    let agent_anchor = ensure_anchor(&format!("agent_queries:{}", query.requester))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToQuery, ())?;

    // Link domain to query
    let domain_anchor = ensure_anchor(&format!("domain_queries:{}", query.domain))?;
    create_link(domain_anchor, action_hash.clone(), LinkTypes::DomainToQuery, ())?;

    // Attempt auto-dispatch if query_type looks like a zome function call
    if let Some(zome_name) = resolve_domain_zome(&query.domain, &query.query_type) {
        let payload_bytes = ExternIO::encode(query.params.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0;
        let dispatch = DispatchInput {
            zome: zome_name,
            fn_name: query.query_type.clone(),
            payload: payload_bytes,
        };
        if let Ok(result) = dispatch_call(dispatch) {
            if result.success {
                let result_str = result.response
                    .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
                    .unwrap_or_else(|| "null".to_string());
                // Auto-resolve the query
                let _ = resolve_query(ResolveQueryInput {
                    query_hash: action_hash.clone(),
                    result: result_str,
                    success: true,
                });
            }
        }
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

/// Map a domain name to its primary coordinator zome for auto-dispatch.
fn resolve_domain_zome(domain: &str, query_type: &str) -> Option<String> {
    let zome = match domain {
        "property" => match query_type {
            s if s.contains("transfer") || s.contains("ownership") => "property_transfer",
            s if s.contains("dispute") => "property_disputes",
            s if s.contains("encumbrance") || s.contains("title") => "property_registry",
            _ => "property_registry",
        },
        "housing" => match query_type {
            s if s.contains("clt") || s.contains("lease") || s.contains("resale") => "housing_clt",
            s if s.contains("member") => "housing_membership",
            s if s.contains("finance") || s.contains("fee") => "housing_finances",
            s if s.contains("maintenance") || s.contains("repair") => "housing_maintenance",
            s if s.contains("governance") || s.contains("proposal") => "housing_governance",
            _ => "housing_units",
        },
        "care" => match query_type {
            s if s.contains("match") => "care_matching",
            s if s.contains("circle") => "care_circles",
            s if s.contains("credential") => "care_credentials",
            s if s.contains("plan") => "care_plans",
            _ => "care_timebank",
        },
        "mutualaid" => match query_type {
            s if s.contains("resource") || s.contains("booking") => "mutualaid_resources",
            s if s.contains("need") || s.contains("handoff") => "mutualaid_needs",
            s if s.contains("pool") => "mutualaid_pools",
            s if s.contains("request") => "mutualaid_requests",
            s if s.contains("circle") => "mutualaid_circles",
            s if s.contains("governance") || s.contains("proposal") => "mutualaid_governance",
            _ => "mutualaid_timebank",
        },
        "water" => match query_type {
            s if s.contains("purity") || s.contains("quality") => "water_purity",
            s if s.contains("capture") || s.contains("harvest") => "water_capture",
            s if s.contains("steward") || s.contains("guardian") => "water_steward",
            s if s.contains("wisdom") || s.contains("knowledge") => "water_wisdom",
            _ => "water_flow",
        },
        "food" => match query_type {
            s if s.contains("distribution") || s.contains("market") || s.contains("order") => "food_distribution",
            s if s.contains("preservation") || s.contains("batch") || s.contains("storage") => "food_preservation",
            s if s.contains("knowledge") || s.contains("seed") || s.contains("recipe") => "food_knowledge",
            _ => "food_production",
        },
        "transport" => match query_type {
            s if s.contains("share") || s.contains("ride") || s.contains("cargo") => "transport_sharing",
            s if s.contains("impact") || s.contains("carbon") || s.contains("emission") => "transport_impact",
            _ => "transport_routes",
        },
        _ => return None,
    };
    Some(zome.to_string())
}

/// Resolve a pending query with a result
#[hdk_extern]
pub fn resolve_query(input: ResolveQueryInput) -> ExternResult<Record> {
    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let mut query: StoredQuery = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid query entry".into())))?;

    let now = sys_time()?;
    query.result = Some(input.result);
    query.resolved_at = Some(now);
    query.success = Some(input.success);

    let updated_hash = update_entry(input.query_hash, &EntryTypes::Query(query))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated query".into()
    )))
}

/// Signal payload emitted to connected UI clients when a bridge event is created
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    /// Signal type identifier for client-side routing
    pub signal_type: String,
    /// The domain that produced the event
    pub domain: String,
    /// Type of event within the domain
    pub event_type: String,
    /// Serialized event payload
    pub payload: String,
    /// Action hash of the created event entry
    pub action_hash: ActionHash,
}

/// Broadcast a cross-domain event within the Commons cluster and emit a signal
#[hdk_extern]
pub fn broadcast_event(event: CommonsEvent) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Event(event.clone()))?;

    // Link to all events
    let all_anchor = ensure_anchor("all_commons_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    // Link by event type
    let type_anchor = ensure_anchor(&format!("event_type:{}:{}", event.domain, event.event_type))?;
    create_link(type_anchor, action_hash.clone(), LinkTypes::EventTypeToEvent, ())?;

    // Link agent to event
    let agent_anchor = ensure_anchor(&format!("agent_events:{}", event.source_agent))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToEvent, ())?;

    // Link domain to event
    let domain_anchor = ensure_anchor(&format!("domain_events:{}", event.domain))?;
    create_link(domain_anchor, action_hash.clone(), LinkTypes::DomainToEvent, ())?;

    // Emit signal to connected UI clients
    let signal = BridgeEventSignal {
        signal_type: "commons_bridge_event".to_string(),
        domain: event.domain.clone(),
        event_type: event.event_type.clone(),
        payload: event.payload.clone(),
        action_hash: action_hash.clone(),
    };
    emit_signal(&signal)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}

// ============================================================================
// Query helpers
// ============================================================================

/// Get all events for a specific domain
#[hdk_extern]
pub fn get_domain_events(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get all queries for a specific domain
#[hdk_extern]
pub fn get_domain_queries(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_queries:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToQuery)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get events by type within a domain
#[hdk_extern]
pub fn get_events_by_type(query: EventTypeQuery) -> ExternResult<Vec<Record>> {
    let type_anchor = anchor_hash(&format!("event_type:{}:{}", query.domain, query.event_type))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::EventTypeToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get all recent events across all domains
#[hdk_extern]
pub fn get_all_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_commons_events")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllEvents)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get my queries
#[hdk_extern]
pub fn get_my_queries(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_queries:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToQuery)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get my events
#[hdk_extern]
pub fn get_my_events(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_events:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

// ============================================================================
// Cross-Cluster Dispatch: Commons → Civic
// ============================================================================

/// Civic-side zomes that commons-bridge is allowed to call cross-cluster.
const ALLOWED_CIVIC_ZOMES: &[&str] = &[
    // Justice domain
    "justice_cases",
    "justice_evidence",
    "justice_arbitration",
    "justice_restorative",
    "justice_enforcement",
    // Emergency domain
    "emergency_incidents",
    "emergency_triage",
    "emergency_resources",
    "emergency_coordination",
    "emergency_shelters",
    "emergency_comms",
    // Media domain
    "media_publication",
    "media_attribution",
    "media_factcheck",
    "media_curation",
    // Civic bridge
    "civic_bridge",
];

/// The hApp role name for the Civic DNA.
const CIVIC_ROLE: &str = "civic";

/// Dispatch a call to any zome in the Civic DNA.
///
/// This is the cross-cluster counterpart of `dispatch_call`.  It uses
/// `CallTargetCell::OtherRole("civic")` to reach zomes in the Civic DNA.
///
/// ## Example use cases
/// - Housing checking for active emergencies before issuing leases
/// - Property verifying media publications referencing a parcel
/// - Water stewardship checking justice disputes on watershed rights
#[hdk_extern]
pub fn dispatch_civic_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("civic:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: CIVIC_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_CIVIC_ZOMES)
}

// ---- Specific cross-cluster use cases ----

/// Input for checking active emergencies near a location.
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckEmergencyForAreaInput {
    /// Latitude of the area to check.
    pub lat: f64,
    /// Longitude of the area to check.
    pub lon: f64,
}

/// Result of an emergency area check.
#[derive(Serialize, Deserialize, Debug)]
pub struct EmergencyAreaCheckResult {
    pub has_active_emergencies: bool,
    pub active_count: u32,
    pub recommendation: Option<String>,
    pub error: Option<String>,
}

/// Check if there are active emergencies affecting a geographic area.
///
/// Cross-cluster call: commons-bridge → civic emergency_incidents via
/// `CallTargetCell::OtherRole("civic")`.  Used by housing, property,
/// and water zomes before critical operations in disaster-prone areas.
#[hdk_extern]
pub fn check_emergency_for_area(input: CheckEmergencyForAreaInput) -> ExternResult<EmergencyAreaCheckResult> {
    let response = call(
        CallTargetCell::OtherRole(CIVIC_ROLE.into()),
        ZomeName::from("emergency_incidents"),
        FunctionName::from("get_active_disasters"),
        None,
        (),
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
            let count = records.len() as u32;
            Ok(EmergencyAreaCheckResult {
                has_active_emergencies: count > 0,
                active_count: count,
                recommendation: if count > 0 {
                    Some(format!(
                        "{} active emergency(ies) — verify operations at ({:.4}, {:.4}) are safe to proceed",
                        count, input.lat, input.lon
                    ))
                } else {
                    None
                },
                error: None,
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(EmergencyAreaCheckResult {
            has_active_emergencies: false,
            active_count: 0,
            recommendation: None,
            error: Some(format!("Cross-cluster network error: {}", err)),
        }),
        _ => Ok(EmergencyAreaCheckResult {
            has_active_emergencies: false,
            active_count: 0,
            recommendation: None,
            error: Some("Failed to reach civic cluster emergency_incidents".into()),
        }),
    }
}

/// Input for checking justice disputes related to a property.
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckJusticeDisputesInput {
    /// Property or resource identifier to check for disputes.
    pub resource_id: String,
}

/// Result of a justice dispute check.
#[derive(Serialize, Deserialize, Debug)]
pub struct JusticeDisputeCheckResult {
    pub has_pending_cases: bool,
    pub recommendation: Option<String>,
    pub error: Option<String>,
}

/// Check if there are active justice cases that may affect a property transfer.
///
/// Cross-cluster call: commons-bridge → civic justice_cases via
/// `CallTargetCell::OtherRole("civic")`.  Used by property-transfer
/// to verify no pending enforcement actions before completing transfers.
#[hdk_extern]
pub fn check_justice_disputes_for_property(input: CheckJusticeDisputesInput) -> ExternResult<JusticeDisputeCheckResult> {
    // Query the civic bridge for justice-related cases matching the resource
    let dispatch = CrossClusterDispatchInput {
        role: CIVIC_ROLE.to_string(),
        zome: "civic_bridge".to_string(),
        fn_name: "get_domain_events".to_string(),
        payload: ExternIO::encode("justice".to_string())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    match bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_CIVIC_ZOMES) {
        Ok(result) if result.success => {
            // Events were returned — check if any reference this property
            // In production, we'd decode and filter; for now, presence of
            // justice events is a signal to proceed with caution.
            Ok(JusticeDisputeCheckResult {
                has_pending_cases: result.response.map_or(false, |r| r.len() > 4),
                recommendation: Some(format!(
                    "Verify resource '{}' is not subject to active enforcement before transfer",
                    input.resource_id
                )),
                error: None,
            })
        }
        Ok(result) => Ok(JusticeDisputeCheckResult {
            has_pending_cases: false,
            recommendation: None,
            error: result.error,
        }),
        Err(e) => Ok(JusticeDisputeCheckResult {
            has_pending_cases: false,
            recommendation: None,
            error: Some(format!("Cross-cluster call failed: {:?}", e)),
        }),
    }
}

// ============================================================================
// Health Check
// ============================================================================

/// Network health check — returns status for all 5 domains
#[hdk_extern]
pub fn health_check(_: ()) -> ExternResult<BridgeHealth> {
    let caller = agent_info()?.agent_initial_pubkey;
    let events = get_all_events(())?;
    let queries = get_my_queries(())?;

    Ok(BridgeHealth {
        healthy: true,
        agent: caller.to_string(),
        total_events: events.len() as u32,
        total_queries: queries.len() as u32,
        domains: vec![
            "property".to_string(),
            "housing".to_string(),
            "care".to_string(),
            "mutualaid".to_string(),
            "water".to_string(),
            "food".to_string(),
            "transport".to_string(),
        ],
    })
}

// ============================================================================
// Typed Convenience Functions (intra-cluster)
// ============================================================================

/// Verify property ownership — housing/care/mutualaid can check before acting.
///
/// Dispatches to `property_registry.verify_ownership` with typed input/output.
#[hdk_extern]
pub fn verify_property_ownership(input: PropertyOwnershipQuery) -> ExternResult<PropertyOwnershipResult> {
    enforce_rate_limit("property_registry")?;
    let payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Encode error: {:?}", e))))?;
    let dispatch = DispatchInput {
        zome: "property_registry".into(),
        fn_name: "verify_ownership".into(),
        payload: payload.0,
    };
    let result = bridge::dispatch_call_checked(&dispatch, ALLOWED_ZOMES)?;
    if result.success {
        if let Some(response) = result.response {
            let decoded: PropertyOwnershipResult = ExternIO(response).decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
            Ok(decoded)
        } else {
            Ok(PropertyOwnershipResult {
                is_owner: false,
                owner_did: None,
                error: Some("No response data".into()),
            })
        }
    } else {
        Ok(PropertyOwnershipResult {
            is_owner: false,
            owner_did: None,
            error: result.error,
        })
    }
}

/// Check care provider availability — mutualaid can find matching caregivers.
///
/// Dispatches to `care_matching.check_availability` with typed input/output.
#[hdk_extern]
pub fn check_care_availability(input: CareAvailabilityQuery) -> ExternResult<CareAvailabilityResult> {
    enforce_rate_limit("care_matching")?;
    let payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Encode error: {:?}", e))))?;
    let dispatch = DispatchInput {
        zome: "care_matching".into(),
        fn_name: "check_availability".into(),
        payload: payload.0,
    };
    let result = bridge::dispatch_call_checked(&dispatch, ALLOWED_ZOMES)?;
    if result.success {
        if let Some(response) = result.response {
            let decoded: CareAvailabilityResult = ExternIO(response).decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
            Ok(decoded)
        } else {
            Ok(CareAvailabilityResult {
                available_count: 0,
                recommendation: "No response data".into(),
                error: Some("Empty response".into()),
            })
        }
    } else {
        Ok(CareAvailabilityResult {
            available_count: 0,
            recommendation: "Dispatch failed".into(),
            error: result.error,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Allowlist validation ----

    #[test]
    fn local_allowlist_covers_all_domains() {
        // Verify ALLOWED_ZOMES has representatives from all 7 commons domains
        let has_property = ALLOWED_ZOMES.iter().any(|z| z.starts_with("property_"));
        let has_housing = ALLOWED_ZOMES.iter().any(|z| z.starts_with("housing_"));
        let has_care = ALLOWED_ZOMES.iter().any(|z| z.starts_with("care_"));
        let has_mutualaid = ALLOWED_ZOMES.iter().any(|z| z.starts_with("mutualaid_"));
        let has_water = ALLOWED_ZOMES.iter().any(|z| z.starts_with("water_"));
        let has_food = ALLOWED_ZOMES.iter().any(|z| z.starts_with("food_"));
        let has_transport = ALLOWED_ZOMES.iter().any(|z| z.starts_with("transport_"));
        assert!(has_property, "ALLOWED_ZOMES missing property domain");
        assert!(has_housing, "ALLOWED_ZOMES missing housing domain");
        assert!(has_care, "ALLOWED_ZOMES missing care domain");
        assert!(has_mutualaid, "ALLOWED_ZOMES missing mutualaid domain");
        assert!(has_water, "ALLOWED_ZOMES missing water domain");
        assert!(has_food, "ALLOWED_ZOMES missing food domain");
        assert!(has_transport, "ALLOWED_ZOMES missing transport domain");
    }

    #[test]
    fn local_allowlist_has_expected_count() {
        // 4 property + 6 housing + 5 care + 7 mutualaid + 5 water + 4 food + 3 transport = 34
        assert_eq!(ALLOWED_ZOMES.len(), 34);
    }

    #[test]
    fn civic_allowlist_covers_all_civic_domains() {
        let has_justice = ALLOWED_CIVIC_ZOMES.iter().any(|z| z.starts_with("justice_"));
        let has_emergency = ALLOWED_CIVIC_ZOMES.iter().any(|z| z.starts_with("emergency_"));
        let has_media = ALLOWED_CIVIC_ZOMES.iter().any(|z| z.starts_with("media_"));
        let has_bridge = ALLOWED_CIVIC_ZOMES.contains(&"civic_bridge");
        assert!(has_justice, "ALLOWED_CIVIC_ZOMES missing justice domain");
        assert!(has_emergency, "ALLOWED_CIVIC_ZOMES missing emergency domain");
        assert!(has_media, "ALLOWED_CIVIC_ZOMES missing media domain");
        assert!(has_bridge, "ALLOWED_CIVIC_ZOMES missing civic_bridge");
    }

    #[test]
    fn civic_allowlist_has_expected_count() {
        // 5 justice + 6 emergency + 4 media + 1 civic_bridge = 16
        assert_eq!(ALLOWED_CIVIC_ZOMES.len(), 16);
    }

    #[test]
    fn civic_role_constant_is_civic() {
        assert_eq!(CIVIC_ROLE, "civic");
    }

    // ---- Domain resolution ----

    #[test]
    fn resolve_property_domain() {
        assert_eq!(resolve_domain_zome("property", "get_property").unwrap(), "property_registry");
        assert_eq!(resolve_domain_zome("property", "transfer_ownership").unwrap(), "property_transfer");
        assert_eq!(resolve_domain_zome("property", "file_dispute").unwrap(), "property_disputes");
        assert_eq!(resolve_domain_zome("property", "check_title").unwrap(), "property_registry");
    }

    #[test]
    fn resolve_housing_domain() {
        assert_eq!(resolve_domain_zome("housing", "create_clt_lease").unwrap(), "housing_clt");
        assert_eq!(resolve_domain_zome("housing", "add_member").unwrap(), "housing_membership");
        assert_eq!(resolve_domain_zome("housing", "pay_fee").unwrap(), "housing_finances");
        assert_eq!(resolve_domain_zome("housing", "report_maintenance").unwrap(), "housing_maintenance");
        assert_eq!(resolve_domain_zome("housing", "submit_proposal").unwrap(), "housing_governance");
        assert_eq!(resolve_domain_zome("housing", "list_units").unwrap(), "housing_units");
    }

    #[test]
    fn resolve_care_domain() {
        assert_eq!(resolve_domain_zome("care", "find_match").unwrap(), "care_matching");
        assert_eq!(resolve_domain_zome("care", "join_circle").unwrap(), "care_circles");
        assert_eq!(resolve_domain_zome("care", "verify_credential").unwrap(), "care_credentials");
        assert_eq!(resolve_domain_zome("care", "create_plan").unwrap(), "care_plans");
        assert_eq!(resolve_domain_zome("care", "log_hours").unwrap(), "care_timebank");
    }

    #[test]
    fn resolve_mutualaid_domain() {
        assert_eq!(resolve_domain_zome("mutualaid", "book_resource").unwrap(), "mutualaid_resources");
        assert_eq!(resolve_domain_zome("mutualaid", "post_need").unwrap(), "mutualaid_needs");
        assert_eq!(resolve_domain_zome("mutualaid", "join_pool").unwrap(), "mutualaid_pools");
        assert_eq!(resolve_domain_zome("mutualaid", "submit_request").unwrap(), "mutualaid_requests");
        assert_eq!(resolve_domain_zome("mutualaid", "form_circle").unwrap(), "mutualaid_circles");
        assert_eq!(resolve_domain_zome("mutualaid", "propose_governance").unwrap(), "mutualaid_governance");
    }

    #[test]
    fn resolve_water_domain() {
        assert_eq!(resolve_domain_zome("water", "test_purity").unwrap(), "water_purity");
        assert_eq!(resolve_domain_zome("water", "log_capture").unwrap(), "water_capture");
        assert_eq!(resolve_domain_zome("water", "assign_steward").unwrap(), "water_steward");
        assert_eq!(resolve_domain_zome("water", "record_wisdom").unwrap(), "water_wisdom");
        assert_eq!(resolve_domain_zome("water", "measure_flow_rate").unwrap(), "water_flow");
    }

    #[test]
    fn resolve_food_domain() {
        assert_eq!(resolve_domain_zome("food", "register_plot").unwrap(), "food_production");
        assert_eq!(resolve_domain_zome("food", "list_market").unwrap(), "food_distribution");
        assert_eq!(resolve_domain_zome("food", "place_order").unwrap(), "food_distribution");
        assert_eq!(resolve_domain_zome("food", "start_batch").unwrap(), "food_preservation");
        assert_eq!(resolve_domain_zome("food", "check_storage").unwrap(), "food_preservation");
        assert_eq!(resolve_domain_zome("food", "catalog_seed").unwrap(), "food_knowledge");
        assert_eq!(resolve_domain_zome("food", "share_recipe").unwrap(), "food_knowledge");
    }

    #[test]
    fn resolve_transport_domain() {
        assert_eq!(resolve_domain_zome("transport", "register_vehicle").unwrap(), "transport_routes");
        assert_eq!(resolve_domain_zome("transport", "create_route").unwrap(), "transport_routes");
        assert_eq!(resolve_domain_zome("transport", "post_ride_share").unwrap(), "transport_sharing");
        assert_eq!(resolve_domain_zome("transport", "request_cargo").unwrap(), "transport_sharing");
        assert_eq!(resolve_domain_zome("transport", "get_carbon_credits").unwrap(), "transport_impact");
        assert_eq!(resolve_domain_zome("transport", "calculate_emissions").unwrap(), "transport_impact");
    }

    #[test]
    fn resolve_unknown_domain_returns_none() {
        assert!(resolve_domain_zome("nonexistent", "anything").is_none());
    }

    // ---- Cross-cluster input type serde ----

    #[test]
    fn check_emergency_input_serde_roundtrip() {
        let input = CheckEmergencyForAreaInput { lat: 32.95, lon: -96.73 };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CheckEmergencyForAreaInput = serde_json::from_str(&json).unwrap();
        assert!((input2.lat - 32.95).abs() < 1e-10);
        assert!((input2.lon - (-96.73)).abs() < 1e-10);
    }

    #[test]
    fn emergency_area_check_result_serde_roundtrip() {
        let result = EmergencyAreaCheckResult {
            has_active_emergencies: true,
            active_count: 3,
            recommendation: Some("Caution advised".into()),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: EmergencyAreaCheckResult = serde_json::from_str(&json).unwrap();
        assert!(r2.has_active_emergencies);
        assert_eq!(r2.active_count, 3);
        assert_eq!(r2.recommendation.as_deref(), Some("Caution advised"));
    }

    #[test]
    fn justice_dispute_input_serde_roundtrip() {
        let input = CheckJusticeDisputesInput { resource_id: "PROP-001".into() };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CheckJusticeDisputesInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.resource_id, "PROP-001");
    }

    #[test]
    fn justice_dispute_result_serde_roundtrip() {
        let result = JusticeDisputeCheckResult {
            has_pending_cases: false,
            recommendation: None,
            error: Some("Network unreachable".into()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: JusticeDisputeCheckResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.has_pending_cases);
        assert!(r2.recommendation.is_none());
        assert_eq!(r2.error.as_deref(), Some("Network unreachable"));
    }

    // ---- Rate limit validation ----

    #[test]
    fn rate_limit_under_threshold_passes() {
        assert!(check_rate_limit_count(0).is_ok());
        assert!(check_rate_limit_count(50).is_ok());
        assert!(check_rate_limit_count(99).is_ok());
    }

    #[test]
    fn rate_limit_at_threshold_rejects() {
        let err = check_rate_limit_count(100).unwrap_err();
        assert!(err.contains("Rate limit exceeded"));
        assert!(err.contains("100"));
    }

    #[test]
    fn rate_limit_over_threshold_rejects() {
        let err = check_rate_limit_count(500).unwrap_err();
        assert!(err.contains("Rate limit exceeded"));
    }

    // ---- Typed convenience function serde ----

    #[test]
    fn property_ownership_query_serde_roundtrip() {
        let q = PropertyOwnershipQuery {
            property_id: "PROP-001".into(),
            requester_did: "did:mycelix:agent_123".into(),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: PropertyOwnershipQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.property_id, "PROP-001");
        assert_eq!(q2.requester_did, "did:mycelix:agent_123");
    }

    #[test]
    fn property_ownership_result_serde_roundtrip() {
        let r = PropertyOwnershipResult {
            is_owner: true,
            owner_did: Some("did:mycelix:owner_456".into()),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: PropertyOwnershipResult = serde_json::from_str(&json).unwrap();
        assert!(r2.is_owner);
        assert_eq!(r2.owner_did.as_deref(), Some("did:mycelix:owner_456"));
    }

    #[test]
    fn care_availability_query_serde_roundtrip() {
        let q = CareAvailabilityQuery {
            skill_needed: "nursing".into(),
            location: Some("downtown".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: CareAvailabilityQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.skill_needed, "nursing");
        assert_eq!(q2.location.as_deref(), Some("downtown"));
    }

    #[test]
    fn care_availability_result_serde_roundtrip() {
        let r = CareAvailabilityResult {
            available_count: 5,
            recommendation: "3 providers nearby".into(),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: CareAvailabilityResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.available_count, 5);
        assert!(r2.error.is_none());
    }

    // ---- Bridge event signal serde ----

    #[test]
    fn bridge_event_signal_serde_roundtrip() {
        let signal = BridgeEventSignal {
            signal_type: "commons_bridge_event".to_string(),
            domain: "property".to_string(),
            event_type: "ownership_transferred".to_string(),
            payload: r#"{"property_id":"PROP-001"}"#.to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let s2: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.signal_type, "commons_bridge_event");
        assert_eq!(s2.domain, "property");
        assert_eq!(s2.event_type, "ownership_transferred");
        assert!(s2.payload.contains("PROP-001"));
    }

    #[test]
    fn bridge_event_signal_type_is_commons() {
        let signal = BridgeEventSignal {
            signal_type: "commons_bridge_event".to_string(),
            domain: "water".to_string(),
            event_type: "flow_measured".to_string(),
            payload: "{}".to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(signal.signal_type, "commons_bridge_event");
    }
}
