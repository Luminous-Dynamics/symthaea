// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Care Bridge Coordinator Zome
//! Cross-hApp query dispatch, event broadcasting, and integration functions.

use care_bridge_integrity::*;
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

/// Submit a cross-hApp care query
#[hdk_extern]
pub fn query_care(query: CareQuery) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::CareQuery(query.clone()))?;

    // Link to all queries
    let all_anchor = ensure_anchor("all_care_queries")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllQueries, ())?;

    // Link agent to query
    let agent_anchor = ensure_anchor(&format!("agent_queries:{}", query.requester))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToQuery,
        (),
    )?;

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

    let mut query: CareQuery = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid query entry".into()
        )))?;

    let now = sys_time()?;
    query.result = Some(input.result);
    query.resolved_at = Some(now);
    query.success = Some(input.success);

    let updated_hash = update_entry(input.query_hash, &EntryTypes::CareQuery(query))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated query".into()
    )))
}

/// Broadcast a care event to the network
#[hdk_extern]
pub fn broadcast_event(event: CareEvent) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::CareEvent(event.clone()))?;

    // Link to all events
    let all_anchor = ensure_anchor("all_care_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    // Link by event type
    let type_key = match &event.event_type {
        CareEventType::PlanCreated => "plan_created",
        CareEventType::PlanUpdated => "plan_updated",
        CareEventType::SessionLogged => "session_logged",
        CareEventType::ProviderJoined => "provider_joined",
        CareEventType::CredentialVerified => "credential_verified",
        CareEventType::UrgentRequest => "urgent_request",
        CareEventType::CircleFormed => "circle_formed",
        CareEventType::ExchangeCompleted => "exchange_completed",
    };
    let type_anchor = ensure_anchor(&format!("event_type:{}", type_key))?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::EventTypeToEvent,
        (),
    )?;

    // Link agent to event
    let agent_anchor = ensure_anchor(&format!("agent_events:{}", event.source_agent))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToEvent,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}

/// Verify identity link - queries the identity hApp to verify an agent's identity
#[hdk_extern]
pub fn verify_identity_link(target_agent: AgentPubKey) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let query = CareQuery {
        query_type: CareQueryType::IdentityVerification,
        requester: caller,
        target_happ: "mycelix-identity".to_string(),
        parameters: serde_json::json!({
            "agent": target_agent.to_string(),
            "verification_type": "basic"
        })
        .to_string(),
        result: None,
        created_at: now,
        resolved_at: None,
        success: None,
    };

    query_care(query)
}

/// Check health needs - queries the health hApp for care-relevant health information
#[hdk_extern]
pub fn check_health_needs(target_agent: AgentPubKey) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let query = CareQuery {
        query_type: CareQueryType::HealthNeeds,
        requester: caller,
        target_happ: "mycelix-health".to_string(),
        parameters: serde_json::json!({
            "agent": target_agent.to_string(),
            "query": "care_needs"
        })
        .to_string(),
        result: None,
        created_at: now,
        resolved_at: None,
        success: None,
    };

    query_care(query)
}

/// Get all events of a specific type
#[hdk_extern]
pub fn get_events_by_type(event_type: CareEventType) -> ExternResult<Vec<Record>> {
    let type_key = match &event_type {
        CareEventType::PlanCreated => "plan_created",
        CareEventType::PlanUpdated => "plan_updated",
        CareEventType::SessionLogged => "session_logged",
        CareEventType::ProviderJoined => "provider_joined",
        CareEventType::CredentialVerified => "credential_verified",
        CareEventType::UrgentRequest => "urgent_request",
        CareEventType::CircleFormed => "circle_formed",
        CareEventType::ExchangeCompleted => "exchange_completed",
    };
    let type_anchor = anchor_hash(&format!("event_type:{}", type_key))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::EventTypeToEvent)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all recent events
#[hdk_extern]
pub fn get_all_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_care_events")?;
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

/// Network health check - returns basic status
#[derive(Serialize, Deserialize, Debug)]
pub struct BridgeHealthStatus {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
}

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
    })
}
