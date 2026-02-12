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

fn is_allowed_zome(zome: &str) -> bool {
    ALLOWED_ZOMES.contains(&zome)
}

// ============================================================================
// Helpers
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

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
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

// ============================================================================
// Cross-Domain Dispatch (synchronous RPC)
// ============================================================================

/// Input for dispatching a call to any domain zome within the Civic DNA.
#[derive(Serialize, Deserialize, Debug)]
pub struct DispatchInput {
    /// Target zome name (e.g., "justice_cases", "emergency_incidents").
    /// Must be in the ALLOWED_ZOMES list.
    pub zome: String,
    /// Target function name (e.g., "report_incident", "publish").
    pub fn_name: String,
    /// MessagePack-serialized input payload.
    pub payload: Vec<u8>,
}

/// Result of a dispatched cross-domain call.
#[derive(Serialize, Deserialize, Debug)]
pub struct DispatchResult {
    /// Whether the call succeeded.
    pub success: bool,
    /// MessagePack-serialized response payload (on success).
    pub response: Option<Vec<u8>>,
    /// Error message (on failure).
    pub error: Option<String>,
}

/// Dispatch a synchronous call to any domain zome within the Civic DNA.
///
/// Validates the target zome against an allowlist, then uses
/// `call(CallTargetCell::Local, ...)` to invoke the function directly.
#[hdk_extern]
pub fn dispatch_call(input: DispatchInput) -> ExternResult<DispatchResult> {
    if !is_allowed_zome(&input.zome) {
        return Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!(
                "Zome '{}' is not in the allowed dispatch list. Valid zomes: {:?}",
                input.zome, ALLOWED_ZOMES
            )),
        });
    }

    let payload = ExternIO(input.payload);

    match call(
        CallTargetCell::Local,
        ZomeName::from(input.zome.as_str()),
        FunctionName::from(input.fn_name.as_str()),
        None,
        payload,
    ) {
        Ok(ZomeCallResponse::Ok(extern_io)) => Ok(DispatchResult {
            success: true,
            response: Some(extern_io.0),
            error: None,
        }),
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!("Network error: {}", err)),
        }),
        Ok(other) => Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!("Zome call rejected: {:?}", other)),
        }),
        Err(e) => Ok(DispatchResult {
            success: false,
            response: None,
            error: Some(format!("Call failed: {:?}", e)),
        }),
    }
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
        let payload_bytes = query.params.as_bytes().to_vec();
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
#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveQueryInput {
    pub query_hash: ActionHash,
    pub result: String,
    pub success: bool,
}

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
// Query Helpers
// ============================================================================

/// Get events for a specific domain
#[hdk_extern]
pub fn get_domain_events(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all events
#[hdk_extern]
pub fn get_all_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_civic_events")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllEvents)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
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
    records_from_links(links)
}

// ============================================================================
// Health Check
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CivicBridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

#[hdk_extern]
pub fn health_check(_: ()) -> ExternResult<CivicBridgeHealth> {
    let caller = agent_info()?.agent_initial_pubkey;
    let events = get_all_events(())?;
    let queries = get_my_queries(())?;

    Ok(CivicBridgeHealth {
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
