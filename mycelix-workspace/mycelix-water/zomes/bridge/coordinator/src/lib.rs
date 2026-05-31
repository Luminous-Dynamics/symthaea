// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Water Bridge Coordinator Zome
//! Cross-hApp integration: queries, events, property rights checks, identity verification

use hdk::prelude::*;
use water_bridge_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
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
// WATER QUERIES
// ============================================================================

/// Submit a cross-hApp water query
#[hdk_extern]
pub fn query_water(input: QueryWaterInput) -> ExternResult<Record> {
    // Validate params is valid JSON
    if serde_json::from_str::<serde_json::Value>(&input.params).is_err() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query params must be valid JSON".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let query = WaterQuery {
        query_type: input.query_type,
        requester: agent_info.agent_initial_pubkey.clone(),
        params: input.params,
        requested_at: now,
        response: None,
        answered: false,
    };

    let action_hash = create_entry(&EntryTypes::WaterQuery(query))?;

    // Link to all queries
    create_entry(&EntryTypes::Anchor(Anchor("all_queries".to_string())))?;
    create_link(
        anchor_hash("all_queries")?,
        action_hash.clone(),
        LinkTypes::AllQueries,
        (),
    )?;

    // Link requester to query
    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::RequesterToQuery,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryWaterInput {
    pub query_type: WaterQueryType,
    pub params: String,
}

/// Answer a pending water query
#[hdk_extern]
pub fn answer_query(input: AnswerQueryInput) -> ExternResult<Record> {
    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;
    let mut query: WaterQuery = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid query entry".into()
        )))?;

    if query.answered {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query has already been answered".into()
        )));
    }

    // Validate response is valid JSON
    if serde_json::from_str::<serde_json::Value>(&input.response).is_err() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Response must be valid JSON".into()
        )));
    }

    query.response = Some(input.response);
    query.answered = true;

    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::WaterQuery(query),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated query".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AnswerQueryInput {
    pub query_hash: ActionHash,
    pub response: String,
}

// ============================================================================
// WATER EVENTS
// ============================================================================

/// Broadcast a water event to other hApps
#[hdk_extern]
pub fn broadcast_event(input: BroadcastEventInput) -> ExternResult<Record> {
    // Validate payload is valid JSON
    if serde_json::from_str::<serde_json::Value>(&input.payload).is_err() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event payload must be valid JSON".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let event = WaterEvent {
        event_type: input.event_type.clone(),
        payload: input.payload,
        triggered_by: agent_info.agent_initial_pubkey,
        occurred_at: now,
        reference_hash: input.reference_hash,
    };

    let action_hash = create_entry(&EntryTypes::WaterEvent(event))?;

    // Link to all events
    create_entry(&EntryTypes::Anchor(Anchor("all_events".to_string())))?;
    create_link(
        anchor_hash("all_events")?,
        action_hash.clone(),
        LinkTypes::AllEvents,
        (),
    )?;

    // Link event type to event
    let type_anchor = format!("event_type:{:?}", input.event_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::EventTypeToEvent,
        (),
    )?;

    // Link to recent events
    create_entry(&EntryTypes::Anchor(Anchor("recent_events".to_string())))?;
    create_link(
        anchor_hash("recent_events")?,
        action_hash.clone(),
        LinkTypes::RecentEvents,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastEventInput {
    pub event_type: WaterEventType,
    pub payload: String,
    pub reference_hash: Option<ActionHash>,
}

/// Get recent water events (for polling by other hApps)
#[hdk_extern]
pub fn get_recent_events(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("recent_events")?, LinkTypes::RecentEvents)?,
        GetStrategy::default(),
    )?;
    let mut records = records_from_links(links)?;
    records.sort_by(|a, b| b.action().timestamp().cmp(&a.action().timestamp()));
    // Return most recent 50 events
    records.truncate(50);
    Ok(records)
}

/// Get events by type
#[hdk_extern]
pub fn get_events_by_type(event_type: WaterEventType) -> ExternResult<Vec<Record>> {
    let type_anchor = format!("event_type:{:?}", event_type);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&type_anchor)?, LinkTypes::EventTypeToEvent)?,
        GetStrategy::default(),
    )?;
    let mut records = records_from_links(links)?;
    records.sort_by(|a, b| b.action().timestamp().cmp(&a.action().timestamp()));
    Ok(records)
}

// ============================================================================
// CROSS-HAPP INTEGRATION HELPERS
// ============================================================================

/// Check property rights for a location (calls property hApp via bridge)
/// Returns a JSON string with the property rights status
#[hdk_extern]
pub fn check_property_rights(
    input: PropertyRightsCheckInput,
) -> ExternResult<PropertyRightsResult> {
    // In production, this would call the property hApp via bridge call.
    // For now, we record the check and return a stub that can be replaced
    // when the property hApp bridge is configured.
    let agent_info = agent_info()?;
    let now = sys_time()?;

    let query_params = serde_json::json!({
        "check_type": "property_rights",
        "location_lat": input.location_lat,
        "location_lon": input.location_lon,
        "agent": agent_info.agent_initial_pubkey.to_string()
    });

    let query = WaterQuery {
        query_type: WaterQueryType::AgentRights,
        requester: agent_info.agent_initial_pubkey,
        params: query_params.to_string(),
        requested_at: now,
        response: None,
        answered: false,
    };

    let action_hash = create_entry(&EntryTypes::WaterQuery(query))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_queries".to_string())))?;
    create_link(
        anchor_hash("all_queries")?,
        action_hash,
        LinkTypes::AllQueries,
        (),
    )?;

    Ok(PropertyRightsResult {
        has_rights: false,
        message: "Property rights check pending - awaiting bridge to property hApp".to_string(),
        query_recorded: true,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PropertyRightsCheckInput {
    pub location_lat: f64,
    pub location_lon: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PropertyRightsResult {
    pub has_rights: bool,
    pub message: String,
    pub query_recorded: bool,
}

/// Verify steward identity (calls identity hApp via bridge)
#[hdk_extern]
pub fn verify_steward_identity(agent: AgentPubKey) -> ExternResult<IdentityVerificationResult> {
    // In production, this would call the identity hApp via bridge call.
    // For now, we verify the agent exists on the network.
    let agent_info_result = agent_info()?;
    let now = sys_time()?;

    let query_params = serde_json::json!({
        "check_type": "identity_verification",
        "target_agent": agent.to_string()
    });

    let query = WaterQuery {
        query_type: WaterQueryType::AgentRights,
        requester: agent_info_result.agent_initial_pubkey,
        params: query_params.to_string(),
        requested_at: now,
        response: None,
        answered: false,
    };

    let action_hash = create_entry(&EntryTypes::WaterQuery(query))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_queries".to_string())))?;
    create_link(
        anchor_hash("all_queries")?,
        action_hash,
        LinkTypes::AllQueries,
        (),
    )?;

    Ok(IdentityVerificationResult {
        is_verified: false,
        message: "Identity verification pending - awaiting bridge to identity hApp".to_string(),
        query_recorded: true,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IdentityVerificationResult {
    pub is_verified: bool,
    pub message: String,
    pub query_recorded: bool,
}
