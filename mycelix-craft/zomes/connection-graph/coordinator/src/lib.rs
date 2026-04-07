#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Connection Graph Coordinator Zome
//!
//! Craft network connections and recommendations.
//! Connection counts use link-counting (never mutable entry counters).
//! Bidirectional links created on accept ensure both parties can query.

use hdk::prelude::*;
use connection_graph_integrity::{
    ConnectionRequest, ConnectionStatus, EntryTypes, LinkTypes, Recommendation,
};

// ============== Helpers ==============

fn load_connection_requests(links: Vec<Link>) -> ExternResult<Vec<(ActionHash, ConnectionRequest)>> {
    let mut items = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(req) = ConnectionRequest::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        items.push((hash, req));
                    }
                }
            }
        }
    }
    Ok(items)
}

fn load_recommendations(links: Vec<Link>) -> ExternResult<Vec<Recommendation>> {
    let mut items = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(rec) = Recommendation::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        items.push(rec);
                    }
                }
            }
        }
    }
    Ok(items)
}

// ============== Input Types ==============

#[derive(Serialize, Deserialize, Debug)]
pub struct SendConnectionRequestInput {
    pub target: AgentPubKey,
    pub message: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WriteRecommendationInput {
    pub recommended_agent: AgentPubKey,
    pub relationship: String,
    pub text: String,
}

// ============== Extern Functions ==============

/// Send a connection request to another agent.
#[hdk_extern]
pub fn send_connection_request(input: SendConnectionRequestInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let request = ConnectionRequest {
        sender: agent.clone(),
        target: input.target.clone(),
        message: input.message,
        created_at: now,
        status: ConnectionStatus::Pending,
    };

    let action_hash = create_entry(EntryTypes::ConnectionRequest(request))?;

    // Link: sender -> outgoing request
    let sender_hash: AnyDhtHash = agent.into();
    create_link(sender_hash, action_hash.clone(), LinkTypes::AgentToOutgoingRequest, vec![])?;

    // Link: target -> incoming request
    let target_hash: AnyDhtHash = input.target.into();
    create_link(target_hash, action_hash.clone(), LinkTypes::AgentToIncomingRequest, vec![])?;

    Ok(action_hash)
}

/// Accept a connection request — creates bidirectional AgentToConnection links.
#[hdk_extern]
pub fn accept_connection(request_hash: ActionHash) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Fetch the request
    let record = get(request_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;

    let request: ConnectionRequest = match record.entry().as_option() {
        Some(Entry::App(bytes)) => ConnectionRequest::try_from(
            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
        ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?,
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Not an entry".into()))),
    };

    // Only the target can accept
    if request.target != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the target of a connection request can accept it".into()
        )));
    }

    if request.status != ConnectionStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Connection request is not pending".into()
        )));
    }

    // Update status to Accepted
    let mut accepted = request.clone();
    accepted.status = ConnectionStatus::Accepted;
    let new_hash = update_entry(request_hash, &accepted)?;

    // Create bidirectional connection links
    // Sender -> Target (as connection)
    create_link(
        request.sender.clone(),
        agent.clone(),
        LinkTypes::AgentToConnection,
        vec![],
    )?;

    // Target -> Sender (as connection)
    create_link(
        agent,
        request.sender,
        LinkTypes::AgentToConnection,
        vec![],
    )?;

    Ok(new_hash)
}

/// Decline a connection request.
#[hdk_extern]
pub fn decline_connection(request_hash: ActionHash) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let record = get(request_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;

    let request: ConnectionRequest = match record.entry().as_option() {
        Some(Entry::App(bytes)) => ConnectionRequest::try_from(
            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
        ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?,
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Not an entry".into()))),
    };

    if request.target != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the target can decline a connection request".into()
        )));
    }

    let mut declined = request;
    declined.status = ConnectionStatus::Declined;
    update_entry(request_hash, &declined)
}

/// List all accepted connections for the calling agent (link-counting safe).
#[hdk_extern]
pub fn list_connections(_: ()) -> ExternResult<Vec<AgentPubKey>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToConnection)?,
        GetStrategy::Local,
    )?;

    Ok(links
        .into_iter()
        .filter_map(|link| link.target.into_agent_pub_key())
        .collect())
}

/// List pending incoming connection requests.
#[hdk_extern]
pub fn list_pending_requests(_: ()) -> ExternResult<Vec<(ActionHash, ConnectionRequest)>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToIncomingRequest)?,
        GetStrategy::Local,
    )?;

    let all = load_connection_requests(links)?;
    Ok(all
        .into_iter()
        .filter(|(_, req)| req.status == ConnectionStatus::Pending)
        .collect())
}

/// Get connection count for any agent (link-counting, never stored as counter).
#[hdk_extern]
pub fn get_connection_count(agent: AgentPubKey) -> ExternResult<u32> {
    let agent_hash: AnyDhtHash = agent.into();
    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToConnection)?,
        GetStrategy::Local,
    )?;
    Ok(links.len() as u32)
}

/// Write a recommendation for another agent.
#[hdk_extern]
pub fn write_recommendation(input: WriteRecommendationInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let rec = Recommendation {
        recommender: agent.clone(),
        recommended_agent: input.recommended_agent.clone(),
        relationship: input.relationship,
        text: input.text,
        created_at: now,
        staked_sap: None,
        stake_id: None,
    };

    let action_hash = create_entry(EntryTypes::Recommendation(rec))?;

    // Link: recommender -> recommendation (authored by)
    let recommender_hash: AnyDhtHash = agent.into();
    create_link(recommender_hash, action_hash.clone(), LinkTypes::AgentToRecommendation, vec![])?;

    // Link: recommended agent -> recommendation (received)
    let recommended_hash: AnyDhtHash = input.recommended_agent.into();
    create_link(recommended_hash, action_hash.clone(), LinkTypes::RecommendedAgentToRecommendation, vec![])?;

    Ok(action_hash)
}

/// List recommendations received by an agent.
#[hdk_extern]
pub fn list_recommendations(agent: AgentPubKey) -> ExternResult<Vec<Recommendation>> {
    let agent_hash: AnyDhtHash = agent.into();
    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::RecommendedAgentToRecommendation)?,
        GetStrategy::Local,
    )?;
    load_recommendations(links)
}

/// List recommendations received by the calling agent.
#[hdk_extern]
pub fn list_my_recommendations(_: ()) -> ExternResult<Vec<Recommendation>> {
    let agent = agent_info()?.agent_initial_pubkey;
    list_recommendations(agent)
}
