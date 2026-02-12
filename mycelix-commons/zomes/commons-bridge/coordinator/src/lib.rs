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
    DispatchInput, DispatchResult, ResolveQueryInput, EventTypeQuery, BridgeHealth,
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
// Cross-Domain Dispatch (synchronous RPC)
// ============================================================================

/// Dispatch a synchronous call to any domain zome within the Commons DNA.
///
/// This is the core cross-domain integration primitive. It validates the target
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
    let stored: StoredQuery = query.clone().into();
    let action_hash = create_entry(&EntryTypes::Query(stored))?;

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

/// Broadcast a cross-domain event within the Commons cluster
#[hdk_extern]
pub fn broadcast_event(event: CommonsEvent) -> ExternResult<Record> {
    let stored: StoredEvent = event.clone().into();
    let action_hash = create_entry(&EntryTypes::Event(stored))?;

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
        ],
    })
}
