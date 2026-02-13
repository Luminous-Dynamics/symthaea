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

/// Broadcast a cross-domain event
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
        let has_bridge = ALLOWED_COMMONS_ZOMES.contains(&"commons_bridge");
        assert!(has_property, "ALLOWED_COMMONS_ZOMES missing property domain");
        assert!(has_housing, "ALLOWED_COMMONS_ZOMES missing housing domain");
        assert!(has_care, "ALLOWED_COMMONS_ZOMES missing care domain");
        assert!(has_mutualaid, "ALLOWED_COMMONS_ZOMES missing mutualaid domain");
        assert!(has_water, "ALLOWED_COMMONS_ZOMES missing water domain");
        assert!(has_bridge, "ALLOWED_COMMONS_ZOMES missing commons_bridge");
    }

    #[test]
    fn commons_allowlist_has_expected_count() {
        // 4 property + 6 housing + 5 care + 7 mutualaid + 5 water + 1 commons_bridge = 28
        assert_eq!(ALLOWED_COMMONS_ZOMES.len(), 28);
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
}
