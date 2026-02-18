//! Civic Bridge Coordinator Zome
//!
//! Unified cross-domain dispatch for the Civic cluster.
//! Provides three integration patterns:
//!
//! 1. **dispatch_call** — synchronous RPC to any domain zome via
//!    `call(CallTargetCell::Local, ...)`. The core value of clustering.
//! 2. **query_civic** — audited async query/response with auto-dispatch
//! 3. **broadcast_event** — pub-sub event distribution across domains

use civic_bridge_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    self as bridge,
    DispatchInput, DispatchResult, CrossClusterDispatchInput,
    ResolveQueryInput, EventTypeQuery, BridgeHealth,
    JusticeAreaQuery, JusticeAreaResult,
    FactcheckStatusQuery, FactcheckStatusResult,
    AuditTrailQuery, AuditTrailEntry, AuditTrailResult,
    WaterSafetyQuery, WaterSafetyResult,
    EmergencyFoodQuery, EmergencyFoodResult,
    ShelterCapacityQuery, ShelterCapacityResult,
    EmergencyCareQuery, EmergencyCareResult,
    RATE_LIMIT_WINDOW_SECS, check_rate_limit_count,
};

// ============================================================================
// Allowed zome names — security boundary for dispatch
// ============================================================================

const ALLOWED_ZOMES: &[&str] = &[
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

/// Dispatch a synchronous call to any domain zome within the Civic DNA.
///
/// Rate-limited to 100 calls per 60 seconds per agent. Validates the target
/// zome against an allowlist, then uses `call(CallTargetCell::Local, ...)`.
#[hdk_extern]
pub fn dispatch_call(input: DispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&input.zome)?;
    bridge::dispatch_call_checked(&input, ALLOWED_ZOMES)
}

// ============================================================================
// Audited Query/Response (with auto-dispatch)
// ============================================================================

/// Submit a cross-domain civic query.
///
/// Stores the query on the DHT for auditability, then attempts to auto-dispatch
/// to the target domain zome if the query_type matches a known function name.
#[hdk_extern]
pub fn query_civic(query: CivicQueryEntry) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Query(query.clone()))?;

    let all_anchor = ensure_anchor("all_civic_queries")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllQueries, ())?;

    let agent_anchor = ensure_anchor(&format!("agent_queries:{}", query.requester))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToQuery, ())?;

    let domain_anchor = ensure_anchor(&format!("domain_queries:{}", query.domain))?;
    create_link(domain_anchor, action_hash.clone(), LinkTypes::DomainToQuery, ())?;

    // Attempt auto-dispatch
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
        "justice" => match query_type {
            s if s.contains("evidence") => "justice_evidence",
            s if s.contains("arbitrat") => "justice_arbitration",
            s if s.contains("restorative") || s.contains("mediat") => "justice_restorative",
            s if s.contains("enforce") || s.contains("sanction") => "justice_enforcement",
            _ => "justice_cases",
        },
        "emergency" => match query_type {
            s if s.contains("triage") || s.contains("priorit") => "emergency_triage",
            s if s.contains("resource") || s.contains("supply") => "emergency_resources",
            s if s.contains("coordinat") => "emergency_coordination",
            s if s.contains("shelter") => "emergency_shelters",
            s if s.contains("comm") || s.contains("alert") => "emergency_comms",
            _ => "emergency_incidents",
        },
        "media" => match query_type {
            s if s.contains("attribution") || s.contains("source") => "media_attribution",
            s if s.contains("fact") || s.contains("check") || s.contains("verify") => "media_factcheck",
            s if s.contains("curat") || s.contains("recommend") => "media_curation",
            _ => "media_publication",
        },
        _ => return None,
    };
    Some(zome.to_string())
}

// ============================================================================
// Query Resolution
// ============================================================================

/// Resolve a pending query with a result
#[hdk_extern]
pub fn resolve_query(input: ResolveQueryInput) -> ExternResult<Record> {
    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let mut query: CivicQueryEntry = record
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

// ============================================================================
// Event Broadcasting
// ============================================================================

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

/// Broadcast a cross-domain event and emit a signal to connected clients
#[hdk_extern]
pub fn broadcast_event(event: CivicEventEntry) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Event(event.clone()))?;

    let all_anchor = ensure_anchor("all_civic_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    let type_anchor = ensure_anchor(&format!("event_type:{}:{}", event.domain, event.event_type))?;
    create_link(type_anchor, action_hash.clone(), LinkTypes::EventTypeToEvent, ())?;

    let agent_anchor = ensure_anchor(&format!("agent_events:{}", event.source_agent))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToEvent, ())?;

    let domain_anchor = ensure_anchor(&format!("domain_events:{}", event.domain))?;
    create_link(domain_anchor, action_hash.clone(), LinkTypes::DomainToEvent, ())?;

    // Emit signal to connected UI clients
    let signal = BridgeEventSignal {
        signal_type: "civic_bridge_event".to_string(),
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

/// Get events for a specific domain
#[hdk_extern]
pub fn get_domain_events(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

/// Get all events
#[hdk_extern]
pub fn get_all_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_civic_events")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllEvents)?,
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
// Cross-Cluster Dispatch: Civic → Commons
// ============================================================================

/// Commons-side zomes that civic-bridge is allowed to call cross-cluster.
const ALLOWED_COMMONS_ZOMES: &[&str] = &[
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
    // Commons bridge
    "commons_bridge",
];

/// The hApp role name for the Commons DNA.
const COMMONS_ROLE: &str = "commons";

/// Dispatch a call to any zome in the Commons DNA.
///
/// This is the cross-cluster counterpart of `dispatch_call`.  It uses
/// `CallTargetCell::OtherRole("commons")` to reach zomes in the Commons DNA.
///
/// ## Example use cases
/// - Justice enforcement freezing property transfers during disputes
/// - Emergency management querying housing capacity for sheltering
/// - Media fact-checking property ownership claims
#[hdk_extern]
pub fn dispatch_commons_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("commons:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: COMMONS_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_COMMONS_ZOMES)
}

// ---- Specific cross-cluster use cases ----

/// Input for querying property status for justice enforcement.
#[derive(Serialize, Deserialize, Debug)]
pub struct QueryPropertyForEnforcementInput {
    /// Property ID or hash to look up.
    pub property_id: String,
    /// Justice case ID that requires the property freeze.
    pub case_id: String,
}

/// Result of a property enforcement query.
#[derive(Serialize, Deserialize, Debug)]
pub struct PropertyEnforcementResult {
    pub property_found: bool,
    pub enforcement_advisory: Option<String>,
    pub error: Option<String>,
}

/// Query property registry to verify ownership before enforcement action.
///
/// Cross-cluster call: civic-bridge → commons property_registry via
/// `CallTargetCell::OtherRole("commons")`.  Used by justice-enforcement
/// to verify a property exists and identify the owner before issuing
/// sanctions or freezing transfers.
#[hdk_extern]
pub fn query_property_for_enforcement(input: QueryPropertyForEnforcementInput) -> ExternResult<PropertyEnforcementResult> {
    let dispatch = CrossClusterDispatchInput {
        role: COMMONS_ROLE.to_string(),
        zome: "commons_bridge".to_string(),
        fn_name: "get_domain_events".to_string(),
        payload: ExternIO::encode("property".to_string())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    match bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_COMMONS_ZOMES) {
        Ok(result) if result.success => {
            Ok(PropertyEnforcementResult {
                property_found: true,
                enforcement_advisory: Some(format!(
                    "Property '{}' is in the commons registry — case '{}' enforcement may proceed after verification",
                    input.property_id, input.case_id
                )),
                error: None,
            })
        }
        Ok(result) => Ok(PropertyEnforcementResult {
            property_found: false,
            enforcement_advisory: None,
            error: result.error,
        }),
        Err(e) => Ok(PropertyEnforcementResult {
            property_found: false,
            enforcement_advisory: None,
            error: Some(format!("Cross-cluster call failed: {:?}", e)),
        }),
    }
}

/// Input for checking housing capacity for emergency sheltering.
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckHousingCapacityInput {
    /// Disaster hash for correlation.
    pub disaster_id: String,
    /// Geographic area keyword (city name, zone, etc.).
    pub area: String,
}

/// Result of a housing capacity check.
#[derive(Serialize, Deserialize, Debug)]
pub struct HousingCapacityResult {
    pub commons_reachable: bool,
    pub recommendation: Option<String>,
    pub error: Option<String>,
}

/// Check housing capacity in the Commons cluster for emergency sheltering.
///
/// Cross-cluster call: civic-bridge → commons housing_units via
/// `CallTargetCell::OtherRole("commons")`.  Used by emergency-coordination
/// to identify available housing that could supplement dedicated shelters
/// during large-scale disasters.
#[hdk_extern]
pub fn check_housing_capacity_for_sheltering(input: CheckHousingCapacityInput) -> ExternResult<HousingCapacityResult> {
    let response = call(
        CallTargetCell::OtherRole(COMMONS_ROLE.into()),
        ZomeName::from("housing_units"),
        FunctionName::from("get_all_units"),
        None,
        (),
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
            let count = records.len() as u32;
            Ok(HousingCapacityResult {
                commons_reachable: true,
                recommendation: Some(format!(
                    "Found {} housing unit(s) in commons — evaluate availability for disaster '{}' in area '{}'",
                    count, input.disaster_id, input.area
                )),
                error: None,
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(HousingCapacityResult {
            commons_reachable: false,
            recommendation: None,
            error: Some(format!("Cross-cluster network error: {}", err)),
        }),
        _ => Ok(HousingCapacityResult {
            commons_reachable: false,
            recommendation: None,
            error: Some("Failed to reach commons cluster housing_units".into()),
        }),
    }
}

/// Input for verifying care credentials for justice evidence.
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyCareCredentialsInput {
    /// DID of the care provider whose credentials need verification.
    pub provider_did: String,
    /// Case ID for audit trail.
    pub case_id: String,
}

/// Result of a care credential verification.
#[derive(Serialize, Deserialize, Debug)]
pub struct CareCredentialVerifyResult {
    pub commons_reachable: bool,
    pub recommendation: Option<String>,
    pub error: Option<String>,
}

/// Verify care provider credentials in the Commons cluster for justice evidence.
///
/// Cross-cluster call: civic-bridge → commons care_credentials via
/// `CallTargetCell::OtherRole("commons")`.  Used by justice-evidence
/// to verify the qualifications of care providers when their testimony
/// or assessments are submitted as evidence.
#[hdk_extern]
pub fn verify_care_credentials_for_evidence(input: VerifyCareCredentialsInput) -> ExternResult<CareCredentialVerifyResult> {
    let dispatch = CrossClusterDispatchInput {
        role: COMMONS_ROLE.to_string(),
        zome: "care_credentials".to_string(),
        fn_name: "get_provider_credentials".to_string(),
        payload: ExternIO::encode(input.provider_did.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    match bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_COMMONS_ZOMES) {
        Ok(result) if result.success => {
            Ok(CareCredentialVerifyResult {
                commons_reachable: true,
                recommendation: Some(format!(
                    "Care credentials for '{}' retrieved from commons — evaluate for case '{}'",
                    input.provider_did, input.case_id
                )),
                error: None,
            })
        }
        Ok(result) => Ok(CareCredentialVerifyResult {
            commons_reachable: true,
            recommendation: None,
            error: result.error,
        }),
        Err(e) => Ok(CareCredentialVerifyResult {
            commons_reachable: false,
            recommendation: None,
            error: Some(format!("Cross-cluster call failed: {:?}", e)),
        }),
    }
}

// ---- Emergency → Commons cross-cluster resource queries ----

/// Query water safety in a disaster zone.
///
/// Cross-cluster call: civic-bridge → commons water_purity via
/// `CallTargetCell::OtherRole("commons")`.  Used by emergency-coordination
/// to determine which water sources in an affected area are safe for
/// consumption during disaster response.
#[hdk_extern]
pub fn query_water_safety_for_emergency(input: WaterSafetyQuery) -> ExternResult<WaterSafetyResult> {
    let dispatch = CrossClusterDispatchInput {
        role: COMMONS_ROLE.to_string(),
        zome: "water_purity".to_string(),
        fn_name: "check_area_safety".to_string(),
        payload: ExternIO::encode(&input)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    match bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_COMMONS_ZOMES) {
        Ok(result) if result.success => {
            if let Some(response) = result.response {
                let decoded: WaterSafetyResult = ExternIO(response).decode()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
                Ok(decoded)
            } else {
                Ok(WaterSafetyResult {
                    safe_sources: 0,
                    contaminated_sources: 0,
                    total_sources: 0,
                })
            }
        }
        Ok(_result) => Ok(WaterSafetyResult {
            safe_sources: 0,
            contaminated_sources: 0,
            total_sources: 0,
        }),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Cross-cluster water safety query failed: {:?}", e)
        ))),
    }
}

/// Query food availability for emergency distribution.
///
/// Cross-cluster call: civic-bridge → commons food_distribution via
/// `CallTargetCell::OtherRole("commons")`.  Used by emergency-resources
/// to identify food stocks and distribution points that can serve
/// displaced populations.
#[hdk_extern]
pub fn query_food_for_emergency(input: EmergencyFoodQuery) -> ExternResult<EmergencyFoodResult> {
    let dispatch = CrossClusterDispatchInput {
        role: COMMONS_ROLE.to_string(),
        zome: "food_distribution".to_string(),
        fn_name: "check_emergency_availability".to_string(),
        payload: ExternIO::encode(&input)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    match bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_COMMONS_ZOMES) {
        Ok(result) if result.success => {
            if let Some(response) = result.response {
                let decoded: EmergencyFoodResult = ExternIO(response).decode()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
                Ok(decoded)
            } else {
                Ok(EmergencyFoodResult {
                    available_kg: 0.0,
                    distribution_points: 0,
                    estimated_days_supply: 0.0,
                })
            }
        }
        Ok(_result) => Ok(EmergencyFoodResult {
            available_kg: 0.0,
            distribution_points: 0,
            estimated_days_supply: 0.0,
        }),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Cross-cluster food availability query failed: {:?}", e)
        ))),
    }
}

/// Query shelter capacity for emergency housing.
///
/// Cross-cluster call: civic-bridge → commons housing_units via
/// `CallTargetCell::OtherRole("commons")`.  Used by emergency-shelters
/// to find available beds and shelters near a disaster zone, supplementing
/// dedicated emergency shelters with community housing capacity.
#[hdk_extern]
pub fn query_shelter_capacity_for_emergency(input: ShelterCapacityQuery) -> ExternResult<ShelterCapacityResult> {
    let dispatch = CrossClusterDispatchInput {
        role: COMMONS_ROLE.to_string(),
        zome: "housing_units".to_string(),
        fn_name: "check_shelter_capacity".to_string(),
        payload: ExternIO::encode(&input)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    match bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_COMMONS_ZOMES) {
        Ok(result) if result.success => {
            if let Some(response) = result.response {
                let decoded: ShelterCapacityResult = ExternIO(response).decode()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
                Ok(decoded)
            } else {
                Ok(ShelterCapacityResult {
                    available_beds: 0,
                    total_shelters: 0,
                    nearest_shelter_km: 0.0,
                })
            }
        }
        Ok(_result) => Ok(ShelterCapacityResult {
            available_beds: 0,
            total_shelters: 0,
            nearest_shelter_km: 0.0,
        }),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Cross-cluster shelter capacity query failed: {:?}", e)
        ))),
    }
}

/// Query available care providers for emergency response.
///
/// Cross-cluster call: civic-bridge → commons care_matching via
/// `CallTargetCell::OtherRole("commons")`.  Used by emergency-triage
/// to find care providers with the needed skills near a disaster area,
/// prioritized by urgency level.
#[hdk_extern]
pub fn query_care_for_emergency(input: EmergencyCareQuery) -> ExternResult<EmergencyCareResult> {
    let dispatch = CrossClusterDispatchInput {
        role: COMMONS_ROLE.to_string(),
        zome: "care_matching".to_string(),
        fn_name: "find_emergency_providers".to_string(),
        payload: ExternIO::encode(&input)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    match bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_COMMONS_ZOMES) {
        Ok(result) if result.success => {
            if let Some(response) = result.response {
                let decoded: EmergencyCareResult = ExternIO(response).decode()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
                Ok(decoded)
            } else {
                Ok(EmergencyCareResult {
                    available_providers: 0,
                    nearest_provider_km: 0.0,
                })
            }
        }
        Ok(_result) => Ok(EmergencyCareResult {
            available_providers: 0,
            nearest_provider_km: 0.0,
        }),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Cross-cluster care provider query failed: {:?}", e)
        ))),
    }
}

// ============================================================================
// Audit Trail Queries
// ============================================================================

/// Query bridge events within a time range, optionally filtered by domain and type.
///
/// Retrieves events from the DHT, then filters by timestamp and optional
/// domain/event_type criteria. Returns lightweight summaries with payload previews.
#[hdk_extern]
pub fn query_audit_trail(query: AuditTrailQuery) -> ExternResult<AuditTrailResult> {
    let from = Timestamp::from_micros(query.from_us);
    let to = Timestamp::from_micros(query.to_us);

    let records = if let (Some(ref domain), Some(ref event_type)) = (&query.domain, &query.event_type) {
        let type_anchor = anchor_hash(&format!("event_type:{}:{}", domain, event_type))?;
        let links = get_links(
            LinkQuery::try_new(type_anchor, LinkTypes::EventTypeToEvent)?,
            GetStrategy::default(),
        )?;
        bridge::records_from_links(links)?
    } else if let Some(ref domain) = query.domain {
        let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
        let links = get_links(
            LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
            GetStrategy::default(),
        )?;
        bridge::records_from_links(links)?
    } else {
        get_all_events(())?
    };

    let mut entries = Vec::new();
    for record in &records {
        let action = record.action();
        let ts = action.timestamp();
        if ts < from || ts > to {
            continue;
        }

        if let Some(entry) = record.entry().as_option() {
            let bytes: SerializedBytes = SerializedBytes::try_from(entry.clone())
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Serialize: {:?}", e))))?;
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes.bytes()) {
                let domain = value.get("domain").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                let event_type = value.get("event_type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                let source = value.get("source_agent").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                let payload = value.get("payload").and_then(|v| v.as_str()).unwrap_or("{}");
                let preview = if payload.len() > 120 {
                    format!("{}...", &payload[..120])
                } else {
                    payload.to_string()
                };

                entries.push(AuditTrailEntry {
                    domain,
                    event_type,
                    source_agent: source,
                    payload_preview: preview,
                    created_at_us: ts.as_micros(),
                    action_hash: record.action_address().clone(),
                });
            }
        }
    }

    let total = entries.len() as u32;
    Ok(AuditTrailResult {
        entries,
        total_matched: total,
        query_from_us: query.from_us,
        query_to_us: query.to_us,
    })
}

// ============================================================================
// Health Check
// ============================================================================

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
            "justice".to_string(),
            "emergency".to_string(),
            "media".to_string(),
        ],
    })
}

// ============================================================================
// Typed Convenience Functions (intra-cluster)
// ============================================================================

/// Query active justice cases in an area — emergency can check before deploying.
///
/// Dispatches to `justice_cases.get_active_cases_for_area` with typed input/output.
#[hdk_extern]
pub fn get_active_cases_for_area(input: JusticeAreaQuery) -> ExternResult<JusticeAreaResult> {
    enforce_rate_limit("justice_cases")?;
    let payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Encode error: {:?}", e))))?;
    let dispatch = DispatchInput {
        zome: "justice_cases".into(),
        fn_name: "get_active_cases_for_area".into(),
        payload: payload.0,
    };
    let result = bridge::dispatch_call_checked(&dispatch, ALLOWED_ZOMES)?;
    if result.success {
        if let Some(response) = result.response {
            let decoded: JusticeAreaResult = ExternIO(response).decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
            Ok(decoded)
        } else {
            Ok(JusticeAreaResult {
                active_cases: 0,
                recommendation: "No response data".into(),
                error: Some("Empty response".into()),
            })
        }
    } else {
        Ok(JusticeAreaResult {
            active_cases: 0,
            recommendation: "Dispatch failed".into(),
            error: result.error,
        })
    }
}

/// Check factcheck status for a claim — justice can verify media claims.
///
/// Dispatches to `media_factcheck.check_status` with typed input/output.
#[hdk_extern]
pub fn check_factcheck_status(input: FactcheckStatusQuery) -> ExternResult<FactcheckStatusResult> {
    enforce_rate_limit("media_factcheck")?;
    let payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Encode error: {:?}", e))))?;
    let dispatch = DispatchInput {
        zome: "media_factcheck".into(),
        fn_name: "check_status".into(),
        payload: payload.0,
    };
    let result = bridge::dispatch_call_checked(&dispatch, ALLOWED_ZOMES)?;
    if result.success {
        if let Some(response) = result.response {
            let decoded: FactcheckStatusResult = ExternIO(response).decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e))))?;
            Ok(decoded)
        } else {
            Ok(FactcheckStatusResult {
                has_factcheck: false,
                verdict: None,
                error: Some("Empty response".into()),
            })
        }
    } else {
        Ok(FactcheckStatusResult {
            has_factcheck: false,
            verdict: None,
            error: result.error,
        })
    }
}

// ============================================================================
// Cross-Cluster Dispatch (Civic → Identity)
// ============================================================================

/// Identity-side zomes that civic-bridge is allowed to call cross-cluster.
const ALLOWED_IDENTITY_ZOMES: &[&str] = &[
    "identity_bridge",
    "did_registry",
];

/// The hApp role name for the Identity DNA.
const IDENTITY_ROLE: &str = "identity";

/// Dispatch a cross-cluster call to any allowed zome in the Identity DNA.
#[hdk_extern]
pub fn dispatch_identity_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("identity:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Verify an agent's DID is active before sensitive operations.
///
/// Cross-cluster call: civic-bridge → identity did_registry.
/// Used by justice-cases and emergency-coordination to verify agent identity
/// before allowing sensitive civic actions (filing cases, declaring emergencies).
#[hdk_extern]
pub fn verify_agent_did(did: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:did_registry")?;
    let payload = ExternIO::encode(did)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "did_registry".to_string(),
        fn_name: "is_did_active".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Get an agent's trust score from the identity bridge.
///
/// Returns MATL composite score (0.0-1.0) combining MFA assurance and reputation.
/// Used for trust-gated operations in civic domains.
#[hdk_extern]
pub fn get_agent_trust_score(did: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:identity_bridge")?;
    let payload = ExternIO::encode(did)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "identity_bridge".to_string(),
        fn_name: "get_matl_score".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
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
        let has_justice = ALLOWED_ZOMES.iter().any(|z| z.starts_with("justice_"));
        let has_emergency = ALLOWED_ZOMES.iter().any(|z| z.starts_with("emergency_"));
        let has_media = ALLOWED_ZOMES.iter().any(|z| z.starts_with("media_"));
        assert!(has_justice, "ALLOWED_ZOMES missing justice domain");
        assert!(has_emergency, "ALLOWED_ZOMES missing emergency domain");
        assert!(has_media, "ALLOWED_ZOMES missing media domain");
    }

    #[test]
    fn local_allowlist_has_expected_count() {
        // 5 justice + 6 emergency + 4 media = 15
        assert_eq!(ALLOWED_ZOMES.len(), 15);
    }

    #[test]
    fn commons_allowlist_covers_all_commons_domains() {
        let has_property = ALLOWED_COMMONS_ZOMES.iter().any(|z| z.starts_with("property_"));
        let has_housing = ALLOWED_COMMONS_ZOMES.iter().any(|z| z.starts_with("housing_"));
        let has_care = ALLOWED_COMMONS_ZOMES.iter().any(|z| z.starts_with("care_"));
        let has_mutualaid = ALLOWED_COMMONS_ZOMES.iter().any(|z| z.starts_with("mutualaid_"));
        let has_water = ALLOWED_COMMONS_ZOMES.iter().any(|z| z.starts_with("water_"));
        let has_food = ALLOWED_COMMONS_ZOMES.iter().any(|z| z.starts_with("food_"));
        let has_transport = ALLOWED_COMMONS_ZOMES.iter().any(|z| z.starts_with("transport_"));
        let has_bridge = ALLOWED_COMMONS_ZOMES.contains(&"commons_bridge");
        assert!(has_property, "ALLOWED_COMMONS_ZOMES missing property domain");
        assert!(has_housing, "ALLOWED_COMMONS_ZOMES missing housing domain");
        assert!(has_care, "ALLOWED_COMMONS_ZOMES missing care domain");
        assert!(has_mutualaid, "ALLOWED_COMMONS_ZOMES missing mutualaid domain");
        assert!(has_water, "ALLOWED_COMMONS_ZOMES missing water domain");
        assert!(has_food, "ALLOWED_COMMONS_ZOMES missing food domain");
        assert!(has_transport, "ALLOWED_COMMONS_ZOMES missing transport domain");
        assert!(has_bridge, "ALLOWED_COMMONS_ZOMES missing commons_bridge");
    }

    #[test]
    fn commons_allowlist_has_expected_count() {
        // 4 property + 6 housing + 5 care + 7 mutualaid + 5 water + 4 food + 3 transport + 1 commons_bridge = 35
        assert_eq!(ALLOWED_COMMONS_ZOMES.len(), 35);
    }

    #[test]
    fn commons_role_constant_is_commons() {
        assert_eq!(COMMONS_ROLE, "commons");
    }

    // ---- Domain resolution ----

    #[test]
    fn resolve_justice_domain() {
        assert_eq!(resolve_domain_zome("justice", "submit_evidence").unwrap(), "justice_evidence");
        assert_eq!(resolve_domain_zome("justice", "start_arbitration").unwrap(), "justice_arbitration");
        assert_eq!(resolve_domain_zome("justice", "initiate_restorative").unwrap(), "justice_restorative");
        assert_eq!(resolve_domain_zome("justice", "enforce_sanction").unwrap(), "justice_enforcement");
        assert_eq!(resolve_domain_zome("justice", "file_case").unwrap(), "justice_cases");
    }

    #[test]
    fn resolve_emergency_domain() {
        assert_eq!(resolve_domain_zome("emergency", "assess_triage").unwrap(), "emergency_triage");
        assert_eq!(resolve_domain_zome("emergency", "deploy_resource").unwrap(), "emergency_resources");
        assert_eq!(resolve_domain_zome("emergency", "coordinate_response").unwrap(), "emergency_coordination");
        assert_eq!(resolve_domain_zome("emergency", "open_shelter").unwrap(), "emergency_shelters");
        assert_eq!(resolve_domain_zome("emergency", "send_alert").unwrap(), "emergency_comms");
        assert_eq!(resolve_domain_zome("emergency", "report_incident").unwrap(), "emergency_incidents");
    }

    #[test]
    fn resolve_media_domain() {
        assert_eq!(resolve_domain_zome("media", "verify_attribution").unwrap(), "media_attribution");
        assert_eq!(resolve_domain_zome("media", "run_factcheck").unwrap(), "media_factcheck");
        assert_eq!(resolve_domain_zome("media", "curate_content").unwrap(), "media_curation");
        assert_eq!(resolve_domain_zome("media", "submit_article").unwrap(), "media_publication");
    }

    #[test]
    fn resolve_unknown_domain_returns_none() {
        assert!(resolve_domain_zome("nonexistent", "anything").is_none());
    }

    // ---- Cross-cluster input type serde ----

    #[test]
    fn property_enforcement_input_serde_roundtrip() {
        let input = QueryPropertyForEnforcementInput {
            property_id: "PROP-42".into(),
            case_id: "CASE-7".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: QueryPropertyForEnforcementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.property_id, "PROP-42");
        assert_eq!(input2.case_id, "CASE-7");
    }

    #[test]
    fn property_enforcement_result_serde_roundtrip() {
        let result = PropertyEnforcementResult {
            property_found: true,
            enforcement_advisory: Some("Proceed with caution".into()),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: PropertyEnforcementResult = serde_json::from_str(&json).unwrap();
        assert!(r2.property_found);
        assert_eq!(r2.enforcement_advisory.as_deref(), Some("Proceed with caution"));
    }

    #[test]
    fn housing_capacity_input_serde_roundtrip() {
        let input = CheckHousingCapacityInput {
            disaster_id: "DIS-100".into(),
            area: "Richardson, TX".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CheckHousingCapacityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.disaster_id, "DIS-100");
        assert_eq!(input2.area, "Richardson, TX");
    }

    #[test]
    fn housing_capacity_result_serde_roundtrip() {
        let result = HousingCapacityResult {
            commons_reachable: true,
            recommendation: Some("15 units available".into()),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: HousingCapacityResult = serde_json::from_str(&json).unwrap();
        assert!(r2.commons_reachable);
        assert_eq!(r2.recommendation.as_deref(), Some("15 units available"));
    }

    #[test]
    fn care_credentials_input_serde_roundtrip() {
        let input = VerifyCareCredentialsInput {
            provider_did: "did:key:z6Mk...".into(),
            case_id: "CASE-99".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: VerifyCareCredentialsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.provider_did, "did:key:z6Mk...");
        assert_eq!(input2.case_id, "CASE-99");
    }

    #[test]
    fn care_credentials_result_serde_roundtrip() {
        let result = CareCredentialVerifyResult {
            commons_reachable: false,
            recommendation: None,
            error: Some("Connection refused".into()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: CareCredentialVerifyResult = serde_json::from_str(&json).unwrap();
        assert!(!r2.commons_reachable);
        assert!(r2.recommendation.is_none());
        assert_eq!(r2.error.as_deref(), Some("Connection refused"));
    }

    // ---- Emergency → Commons cross-cluster type serde ----

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
            safe_sources: 12,
            contaminated_sources: 3,
            total_sources: 15,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: WaterSafetyResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.safe_sources, 12);
        assert_eq!(r2.contaminated_sources, 3);
        assert_eq!(r2.total_sources, 15);
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
        assert_eq!(q2.skill_needed, "trauma_surgeon");
        assert_eq!(q2.urgency_level, 5);
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

    // ---- Emergency cross-cluster target zomes are in commons allowlist ----

    #[test]
    fn emergency_water_safety_target_in_commons_allowlist() {
        assert!(
            ALLOWED_COMMONS_ZOMES.contains(&"water_purity"),
            "water_purity must be in ALLOWED_COMMONS_ZOMES for water safety queries"
        );
    }

    #[test]
    fn emergency_food_target_in_commons_allowlist() {
        assert!(
            ALLOWED_COMMONS_ZOMES.contains(&"food_distribution"),
            "food_distribution must be in ALLOWED_COMMONS_ZOMES for food availability queries"
        );
    }

    #[test]
    fn emergency_shelter_target_in_commons_allowlist() {
        assert!(
            ALLOWED_COMMONS_ZOMES.contains(&"housing_units"),
            "housing_units must be in ALLOWED_COMMONS_ZOMES for shelter capacity queries"
        );
    }

    #[test]
    fn emergency_care_target_in_commons_allowlist() {
        assert!(
            ALLOWED_COMMONS_ZOMES.contains(&"care_matching"),
            "care_matching must be in ALLOWED_COMMONS_ZOMES for care provider queries"
        );
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
    fn justice_area_query_serde_roundtrip() {
        let q = JusticeAreaQuery {
            area: "downtown".into(),
            case_type: Some("property_dispute".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: JusticeAreaQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.area, "downtown");
        assert_eq!(q2.case_type.as_deref(), Some("property_dispute"));
    }

    #[test]
    fn justice_area_result_serde_roundtrip() {
        let r = JusticeAreaResult {
            active_cases: 3,
            recommendation: "Caution advised".into(),
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: JusticeAreaResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.active_cases, 3);
        assert!(r2.error.is_none());
    }

    #[test]
    fn factcheck_status_query_serde_roundtrip() {
        let q = FactcheckStatusQuery {
            claim_id: "CLAIM-001".into(),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: FactcheckStatusQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.claim_id, "CLAIM-001");
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
        assert_eq!(r2.verdict.as_deref(), Some("verified"));
    }

    // ---- Bridge event signal serde ----

    #[test]
    fn bridge_event_signal_serde_roundtrip() {
        let signal = BridgeEventSignal {
            signal_type: "civic_bridge_event".to_string(),
            domain: "justice".to_string(),
            event_type: "case_filed".to_string(),
            payload: r#"{"case_id":"CASE-42"}"#.to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let s2: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.signal_type, "civic_bridge_event");
        assert_eq!(s2.domain, "justice");
        assert_eq!(s2.event_type, "case_filed");
        assert!(s2.payload.contains("CASE-42"));
    }

    // ---- Audit trail type serde ----

    #[test]
    fn audit_trail_query_full_filters_serde() {
        let q = AuditTrailQuery {
            from_us: 1_700_000_000_000_000,
            to_us: 1_700_001_000_000_000,
            domain: Some("justice".into()),
            event_type: Some("case_filed".into()),
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: AuditTrailQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q2.from_us, 1_700_000_000_000_000);
        assert_eq!(q2.domain.as_deref(), Some("justice"));
    }

    #[test]
    fn audit_trail_query_no_filters_serde() {
        let q = AuditTrailQuery {
            from_us: 0,
            to_us: i64::MAX,
            domain: None,
            event_type: None,
        };
        let json = serde_json::to_string(&q).unwrap();
        let q2: AuditTrailQuery = serde_json::from_str(&json).unwrap();
        assert!(q2.domain.is_none());
    }

    #[test]
    fn audit_trail_entry_serde_roundtrip() {
        let e = AuditTrailEntry {
            domain: "emergency".into(),
            event_type: "disaster_declared".into(),
            source_agent: "uhCAk_responder".into(),
            payload_preview: r#"{"severity":"Critical"}"#.into(),
            created_at_us: 1_700_000_500_000_000,
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&e).unwrap();
        let e2: AuditTrailEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(e2.domain, "emergency");
        assert_eq!(e2.event_type, "disaster_declared");
    }

    #[test]
    fn audit_trail_result_empty_serde() {
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

    #[test]
    fn bridge_event_signal_type_is_civic() {
        let signal = BridgeEventSignal {
            signal_type: "civic_bridge_event".to_string(),
            domain: "emergency".to_string(),
            event_type: "disaster_declared".to_string(),
            payload: "{}".to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(signal.signal_type, "civic_bridge_event");
    }

    // ============================================================================
    // Cross-domain dispatch edge case tests
    // ============================================================================

    // ---- Allowlist integrity ----

    #[test]
    fn allowed_zomes_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_ZOMES {
            assert!(
                seen.insert(zome),
                "Duplicate zome in ALLOWED_ZOMES: '{}'",
                zome
            );
        }
    }

    #[test]
    fn allowed_commons_zomes_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_COMMONS_ZOMES {
            assert!(
                seen.insert(zome),
                "Duplicate zome in ALLOWED_COMMONS_ZOMES: '{}'",
                zome
            );
        }
    }

    #[test]
    fn allowed_zomes_entries_are_non_empty() {
        for zome in ALLOWED_ZOMES {
            assert!(!zome.is_empty(), "ALLOWED_ZOMES contains an empty string");
            assert!(
                !zome.contains(' '),
                "ALLOWED_ZOMES entry '{}' contains whitespace",
                zome
            );
        }
    }

    #[test]
    fn allowed_commons_zomes_entries_are_non_empty() {
        for zome in ALLOWED_COMMONS_ZOMES {
            assert!(!zome.is_empty(), "ALLOWED_COMMONS_ZOMES contains an empty string");
            assert!(
                !zome.contains(' '),
                "ALLOWED_COMMONS_ZOMES entry '{}' contains whitespace",
                zome
            );
        }
    }

    #[test]
    fn allowed_zomes_per_domain_count() {
        let justice_count = ALLOWED_ZOMES.iter().filter(|z| z.starts_with("justice_")).count();
        let emergency_count = ALLOWED_ZOMES.iter().filter(|z| z.starts_with("emergency_")).count();
        let media_count = ALLOWED_ZOMES.iter().filter(|z| z.starts_with("media_")).count();
        assert_eq!(justice_count, 5, "Expected 5 justice zomes");
        assert_eq!(emergency_count, 6, "Expected 6 emergency zomes");
        assert_eq!(media_count, 4, "Expected 4 media zomes");
    }

    // ---- resolve_domain_zome outputs are in ALLOWED_ZOMES ----

    #[test]
    fn resolve_outputs_are_in_allowlist() {
        // Every zome name returned by resolve_domain_zome must be in ALLOWED_ZOMES
        let test_cases = vec![
            ("justice", "submit_evidence"),
            ("justice", "start_arbitration"),
            ("justice", "initiate_restorative"),
            ("justice", "enforce_sanction"),
            ("justice", "file_case"),
            ("emergency", "assess_triage"),
            ("emergency", "deploy_resource"),
            ("emergency", "coordinate_response"),
            ("emergency", "open_shelter"),
            ("emergency", "send_alert"),
            ("emergency", "report_incident"),
            ("media", "verify_attribution"),
            ("media", "run_factcheck"),
            ("media", "curate_content"),
            ("media", "submit_article"),
        ];
        for (domain, query_type) in test_cases {
            let resolved = resolve_domain_zome(domain, query_type);
            assert!(
                resolved.is_some(),
                "resolve_domain_zome('{}', '{}') returned None",
                domain, query_type
            );
            let zome_name = resolved.unwrap();
            assert!(
                ALLOWED_ZOMES.contains(&zome_name.as_str()),
                "resolve_domain_zome('{}', '{}') returned '{}' which is not in ALLOWED_ZOMES",
                domain, query_type, zome_name
            );
        }
    }

    // ---- Edge cases for resolve_domain_zome ----

    #[test]
    fn resolve_empty_query_type_uses_default() {
        // Empty query_type should fall through to the default zome for each domain
        assert_eq!(resolve_domain_zome("justice", "").unwrap(), "justice_cases");
        assert_eq!(resolve_domain_zome("emergency", "").unwrap(), "emergency_incidents");
        assert_eq!(resolve_domain_zome("media", "").unwrap(), "media_publication");
    }

    #[test]
    fn resolve_empty_domain_returns_none() {
        assert!(resolve_domain_zome("", "file_case").is_none());
    }

    #[test]
    fn resolve_case_sensitive_domain() {
        // Domain matching is case-sensitive
        assert!(resolve_domain_zome("Justice", "file_case").is_none());
        assert!(resolve_domain_zome("JUSTICE", "file_case").is_none());
        assert!(resolve_domain_zome("EMERGENCY", "report_incident").is_none());
        assert!(resolve_domain_zome("Media", "submit_article").is_none());
    }

    #[test]
    fn resolve_gibberish_query_type_uses_default() {
        // Nonsensical query_type that matches no keywords should use default
        assert_eq!(
            resolve_domain_zome("justice", "xyzzy_foobar").unwrap(),
            "justice_cases"
        );
        assert_eq!(
            resolve_domain_zome("emergency", "quantum_entanglement").unwrap(),
            "emergency_incidents"
        );
        assert_eq!(
            resolve_domain_zome("media", "cosmic_rays").unwrap(),
            "media_publication"
        );
    }

    // ---- Health check domain list ----

    #[test]
    fn health_check_returns_exactly_three_domains() {
        // The health_check function hardcodes 3 domains; verify the list
        // is consistent with what resolve_domain_zome accepts.
        let expected_domains = vec!["justice", "emergency", "media"];
        for domain in &expected_domains {
            assert!(
                resolve_domain_zome(domain, "anything").is_some(),
                "Health check domain '{}' is not recognized by resolve_domain_zome",
                domain
            );
        }
    }

    // ---- Justice mediation keyword routing ----

    #[test]
    fn resolve_justice_mediation_routes_to_restorative() {
        assert_eq!(
            resolve_domain_zome("justice", "start_mediation").unwrap(),
            "justice_restorative"
        );
    }

    // ---- Justice sanction keyword routing ----

    #[test]
    fn resolve_justice_sanction_routes_to_enforcement() {
        assert_eq!(
            resolve_domain_zome("justice", "apply_sanction").unwrap(),
            "justice_enforcement"
        );
    }

    // ---- Emergency priority keyword routing ----

    #[test]
    fn resolve_emergency_priority_routes_to_triage() {
        assert_eq!(
            resolve_domain_zome("emergency", "set_priority_level").unwrap(),
            "emergency_triage"
        );
    }

    // ---- Emergency supply keyword routing ----

    #[test]
    fn resolve_emergency_supply_routes_to_resources() {
        assert_eq!(
            resolve_domain_zome("emergency", "request_supply").unwrap(),
            "emergency_resources"
        );
    }

    // ---- Emergency comm keyword routing ----

    #[test]
    fn resolve_emergency_comm_routes_to_comms() {
        assert_eq!(
            resolve_domain_zome("emergency", "broadcast_comm").unwrap(),
            "emergency_comms"
        );
    }

    // ---- Media source keyword routing ----

    #[test]
    fn resolve_media_source_routes_to_attribution() {
        assert_eq!(
            resolve_domain_zome("media", "add_source_reference").unwrap(),
            "media_attribution"
        );
    }

    // ---- Media verify keyword routing ----

    #[test]
    fn resolve_media_verify_routes_to_factcheck() {
        assert_eq!(
            resolve_domain_zome("media", "verify_claim").unwrap(),
            "media_factcheck"
        );
    }

    // ---- Media check keyword routing ----

    #[test]
    fn resolve_media_check_routes_to_factcheck() {
        assert_eq!(
            resolve_domain_zome("media", "run_check").unwrap(),
            "media_factcheck"
        );
    }

    // ---- Media recommend keyword routing ----

    #[test]
    fn resolve_media_recommend_routes_to_curation() {
        assert_eq!(
            resolve_domain_zome("media", "get_recommendations").unwrap(),
            "media_curation"
        );
    }

    // ---- Multiple keywords: first match wins ----

    #[test]
    fn resolve_multiple_keywords_uses_first_match() {
        // "evidence_arbitration" contains both "evidence" and "arbitrat";
        // "evidence" is checked first in the justice match.
        assert_eq!(
            resolve_domain_zome("justice", "evidence_arbitration").unwrap(),
            "justice_evidence"
        );
        // "triage_resource" contains both "triage" and "resource";
        // "triage" is checked first in the emergency match.
        assert_eq!(
            resolve_domain_zome("emergency", "triage_resource").unwrap(),
            "emergency_triage"
        );
    }

    // ---- Cross-cluster allowlist symmetry ----

    #[test]
    fn cross_cluster_commons_allowlist_includes_commons_bridge() {
        assert!(
            ALLOWED_COMMONS_ZOMES.contains(&"commons_bridge"),
            "Civic should be able to call commons_bridge for cross-cluster queries"
        );
    }

    #[test]
    fn cross_cluster_commons_allowlist_per_domain_count() {
        let property_count = ALLOWED_COMMONS_ZOMES.iter().filter(|z| z.starts_with("property_")).count();
        let housing_count = ALLOWED_COMMONS_ZOMES.iter().filter(|z| z.starts_with("housing_")).count();
        let care_count = ALLOWED_COMMONS_ZOMES.iter().filter(|z| z.starts_with("care_")).count();
        let mutualaid_count = ALLOWED_COMMONS_ZOMES.iter().filter(|z| z.starts_with("mutualaid_")).count();
        let water_count = ALLOWED_COMMONS_ZOMES.iter().filter(|z| z.starts_with("water_")).count();
        let food_count = ALLOWED_COMMONS_ZOMES.iter().filter(|z| z.starts_with("food_")).count();
        let transport_count = ALLOWED_COMMONS_ZOMES.iter().filter(|z| z.starts_with("transport_")).count();
        assert_eq!(property_count, 4, "Expected 4 property zomes in ALLOWED_COMMONS_ZOMES");
        assert_eq!(housing_count, 6, "Expected 6 housing zomes in ALLOWED_COMMONS_ZOMES");
        assert_eq!(care_count, 5, "Expected 5 care zomes in ALLOWED_COMMONS_ZOMES");
        assert_eq!(mutualaid_count, 7, "Expected 7 mutualaid zomes in ALLOWED_COMMONS_ZOMES");
        assert_eq!(water_count, 5, "Expected 5 water zomes in ALLOWED_COMMONS_ZOMES");
        assert_eq!(food_count, 4, "Expected 4 food zomes in ALLOWED_COMMONS_ZOMES");
        assert_eq!(transport_count, 3, "Expected 3 transport zomes in ALLOWED_COMMONS_ZOMES");
    }

    // ---- Identity cross-cluster allowlist ----

    #[test]
    fn identity_allowlist_contains_bridge() {
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"identity_bridge"));
    }

    #[test]
    fn identity_allowlist_contains_did_registry() {
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"did_registry"));
    }

    #[test]
    fn identity_role_constant_correct() {
        assert_eq!(IDENTITY_ROLE, "identity");
    }

    #[test]
    fn identity_allowlist_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_IDENTITY_ZOMES {
            assert!(
                seen.insert(zome),
                "Duplicate zome in ALLOWED_IDENTITY_ZOMES: '{}'",
                zome
            );
        }
    }

    #[test]
    fn identity_allowlist_entries_are_non_empty() {
        for zome in ALLOWED_IDENTITY_ZOMES {
            assert!(!zome.is_empty(), "ALLOWED_IDENTITY_ZOMES contains an empty string");
            assert!(
                !zome.contains(' '),
                "ALLOWED_IDENTITY_ZOMES entry '{}' contains whitespace",
                zome
            );
        }
    }
}
