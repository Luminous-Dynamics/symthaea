// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Housing Bridge Coordinator Zome
//! Cross-hApp integration for housing queries and event broadcasting.

use hdk::prelude::*;
use housing_bridge_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryHousingInput {
    pub query_type: HousingQueryType,
    pub target_agent: Option<AgentPubKey>,
    pub target_hash: Option<ActionHash>,
    pub parameters: String,
}

/// Query housing data for cross-hApp integration
#[hdk_extern]
pub fn query_housing(input: QueryHousingInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let now = sys_time()?;

    let query = HousingQuery {
        query_type: input.query_type,
        requester: agent_info.agent_initial_pubkey.clone(),
        target_agent: input.target_agent,
        target_hash: input.target_hash,
        parameters: input.parameters,
        response: None,
        queried_at: now,
        responded_at: None,
    };

    let action_hash = create_entry(&EntryTypes::HousingQuery(query))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_queries".to_string())))?;
    create_link(
        anchor_hash("all_queries")?,
        action_hash.clone(),
        LinkTypes::AllQueries,
        (),
    )?;

    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToQuery,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RespondToQueryInput {
    pub query_hash: ActionHash,
    pub response: String,
}

/// Respond to a housing query
#[hdk_extern]
pub fn respond_to_query(input: RespondToQueryInput) -> ExternResult<Record> {
    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let mut query: HousingQuery = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid query entry".into()
        )))?;

    if query.response.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query has already been responded to".into()
        )));
    }

    let now = sys_time()?;
    query.response = Some(input.response);
    query.responded_at = Some(now);

    let new_hash = update_entry(input.query_hash, &EntryTypes::HousingQuery(query))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated query".into()
    )))
}

/// Broadcast a housing event to other hApps
#[hdk_extern]
pub fn broadcast_event(event: HousingEvent) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::HousingEvent(event.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_events".to_string())))?;
    create_link(
        anchor_hash("all_events")?,
        action_hash.clone(),
        LinkTypes::AllEvents,
        (),
    )?;

    // Index by event type
    let type_anchor = format!("event_type:{:?}", event.event_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::EventTypeToEvent,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}

/// Verify member identity via cross-hApp bridge to mycelix-identity
#[hdk_extern]
pub fn verify_member_identity(member: AgentPubKey) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let now = sys_time()?;

    let query = HousingQuery {
        query_type: HousingQueryType::MemberVerification,
        requester: agent_info.agent_initial_pubkey.clone(),
        target_agent: Some(member),
        target_hash: None,
        parameters: serde_json::json!({
            "action": "verify_identity",
            "source_happ": "mycelix-housing"
        })
        .to_string(),
        response: None,
        queried_at: now,
        responded_at: None,
    };

    let action_hash = create_entry(&EntryTypes::HousingQuery(query))?;

    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToQuery,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

/// Check property registry via cross-hApp bridge to mycelix-property
#[hdk_extern]
pub fn check_property_registry(property_hash: ActionHash) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let now = sys_time()?;

    let query = HousingQuery {
        query_type: HousingQueryType::LeaseStatus,
        requester: agent_info.agent_initial_pubkey.clone(),
        target_agent: None,
        target_hash: Some(property_hash),
        parameters: serde_json::json!({
            "action": "check_property_registry",
            "source_happ": "mycelix-housing"
        })
        .to_string(),
        response: None,
        queried_at: now,
        responded_at: None,
    };

    let action_hash = create_entry(&EntryTypes::HousingQuery(query))?;

    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToQuery,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

/// Request dispute resolution via cross-hApp bridge to mycelix-justice
#[hdk_extern]
pub fn request_dispute_resolution(dispute_data: String) -> ExternResult<Record> {
    if dispute_data.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispute data cannot be empty".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let event = HousingEvent {
        event_type: HousingEventType::MaintenanceEmergency,
        source_agent: agent_info.agent_initial_pubkey.clone(),
        data: serde_json::json!({
            "action": "request_dispute_resolution",
            "source_happ": "mycelix-housing",
            "dispute": dispute_data
        })
        .to_string(),
        occurred_at: now,
    };

    let action_hash = create_entry(&EntryTypes::HousingEvent(event))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_events".to_string())))?;
    create_link(
        anchor_hash("all_events")?,
        action_hash.clone(),
        LinkTypes::AllEvents,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}
