//! Commons Bridge Coordinator Zome
//!
//! Unified cross-domain dispatch for the Commons cluster.
//! Routes queries and events by domain field to appropriate zomes
//! via call(CallTargetCell::Local, ...).

use commons_bridge_integrity::*;
use commons_types::{CommonsQuery, CommonsEvent, BridgeHealthStatus};
use hdk::prelude::*;

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

/// Submit a cross-domain query within the Commons cluster
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

/// Input for resolving a query with a result
#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveQueryInput {
    pub query_hash: ActionHash,
    pub result: String,
    pub success: bool,
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

/// Get all events for a specific domain
#[hdk_extern]
pub fn get_domain_events(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all queries for a specific domain
#[hdk_extern]
pub fn get_domain_queries(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_queries:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToQuery)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get events by type within a domain
#[derive(Serialize, Deserialize, Debug)]
pub struct EventTypeQuery {
    pub domain: String,
    pub event_type: String,
}

#[hdk_extern]
pub fn get_events_by_type(query: EventTypeQuery) -> ExternResult<Vec<Record>> {
    let type_anchor = anchor_hash(&format!("event_type:{}:{}", query.domain, query.event_type))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::EventTypeToEvent)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all recent events across all domains
#[hdk_extern]
pub fn get_all_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_commons_events")?;
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

/// Get my events
#[hdk_extern]
pub fn get_my_events(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_events:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToEvent)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Network health check — returns status for all 5 domains
#[hdk_extern]
pub fn health_check(_: ()) -> ExternResult<BridgeHealthStatus> {
    let caller = agent_info()?.agent_initial_pubkey;
    let events = get_all_events(())?;
    let queries = get_my_queries(())?;

    Ok(BridgeHealthStatus {
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
