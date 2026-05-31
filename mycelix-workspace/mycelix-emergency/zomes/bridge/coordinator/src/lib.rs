// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Emergency Bridge Coordinator Zome
//! Cross-hApp integration for emergency coordination

use emergency_bridge_integrity::*;
use hdk::prelude::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Submit a cross-hApp query for emergency data
#[hdk_extern]
pub fn query_emergency(input: QueryEmergencyInput) -> ExternResult<Record> {
    if input.source_happ.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Source hApp cannot be empty".into()
        )));
    }
    if input.parameters.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Parameters cannot be empty".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let query = EmergencyQuery {
        query_id: format!("eq:{}:{}", input.source_happ, now.as_micros()),
        source_happ: input.source_happ.clone(),
        query_type: input.query_type,
        parameters: input.parameters,
        requested_by: agent_info.agent_initial_pubkey,
        requested_at: now,
        response: None,
        responded_at: None,
    };

    let action_hash = create_entry(&EntryTypes::EmergencyQuery(query))?;

    // Link to all queries
    create_entry(&EntryTypes::Anchor(Anchor("all_queries".to_string())))?;
    create_link(
        anchor_hash("all_queries")?,
        action_hash.clone(),
        LinkTypes::AllQueries,
        (),
    )?;

    // Link by source hApp
    let happ_anchor = format!("happ_queries:{}", input.source_happ);
    create_entry(&EntryTypes::Anchor(Anchor(happ_anchor.clone())))?;
    create_link(
        anchor_hash(&happ_anchor)?,
        action_hash.clone(),
        LinkTypes::HappToQuery,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

/// Input for querying emergency data
#[derive(Serialize, Deserialize, Debug)]
pub struct QueryEmergencyInput {
    pub source_happ: String,
    pub query_type: QueryType,
    pub parameters: String,
}

/// Broadcast an emergency event to other hApps
#[hdk_extern]
pub fn broadcast_event(input: BroadcastEventInput) -> ExternResult<Record> {
    if input.payload.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Event payload cannot be empty".into()
        )));
    }
    if input.target_happs.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Must target at least one hApp".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let event = EmergencyEvent {
        event_id: format!("ee:{}:{}", agent_info.agent_initial_pubkey, now.as_micros()),
        event_type: input.event_type,
        disaster_hash: input.disaster_hash.clone(),
        payload: input.payload,
        emitted_by: agent_info.agent_initial_pubkey,
        emitted_at: now,
        target_happs: input.target_happs,
    };

    let action_hash = create_entry(&EntryTypes::EmergencyEvent(event))?;

    // Link to all events
    create_entry(&EntryTypes::Anchor(Anchor("all_events".to_string())))?;
    create_link(
        anchor_hash("all_events")?,
        action_hash.clone(),
        LinkTypes::AllEvents,
        (),
    )?;

    // Link disaster to event
    if let Some(disaster_hash) = input.disaster_hash {
        create_link(
            disaster_hash,
            action_hash.clone(),
            LinkTypes::DisasterToEvent,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}

/// Input for broadcasting an emergency event
#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastEventInput {
    pub event_type: EventType,
    pub disaster_hash: Option<ActionHash>,
    pub payload: String,
    pub target_happs: Vec<String>,
}

/// Request health records for emergency treatment (cross-hApp)
#[hdk_extern]
pub fn request_health_records(input: HealthRecordsRequestInput) -> ExternResult<Record> {
    let query_input = QueryEmergencyInput {
        source_happ: "mycelix-health".to_string(),
        query_type: QueryType::HealthRecords,
        parameters: serde_json::json!({
            "patient_agent": input.patient_agent.to_string(),
            "requesting_disaster": input.disaster_hash.to_string(),
            "reason": input.reason,
        })
        .to_string(),
    };

    query_emergency(query_input)
}

/// Input for requesting health records
#[derive(Serialize, Deserialize, Debug)]
pub struct HealthRecordsRequestInput {
    pub patient_agent: AgentPubKey,
    pub disaster_hash: ActionHash,
    pub reason: String,
}

/// Verify a volunteer's identity (cross-hApp)
#[hdk_extern]
pub fn verify_volunteer_identity(input: VerifyVolunteerInput) -> ExternResult<Record> {
    let query_input = QueryEmergencyInput {
        source_happ: "mycelix-identity".to_string(),
        query_type: QueryType::IdentityVerification,
        parameters: serde_json::json!({
            "volunteer_agent": input.volunteer_agent.to_string(),
            "disaster_hash": input.disaster_hash.to_string(),
            "required_credentials": input.required_credentials,
        })
        .to_string(),
    };

    query_emergency(query_input)
}

/// Input for verifying volunteer identity
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyVolunteerInput {
    pub volunteer_agent: AgentPubKey,
    pub disaster_hash: ActionHash,
    pub required_credentials: Vec<String>,
}
