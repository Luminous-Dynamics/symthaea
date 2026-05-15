// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Connection Graph Coordinator Zome
//!
//! Craft network connections and recommendations.
//! Connection counts use link-counting (never mutable entry counters).
//! Bidirectional links created on accept ensure both parties can query.

use connection_graph_integrity::*;
use hdk::prelude::*;
use mycelix_zome_helpers as _;

#[hdk_extern]
pub fn request_connection(target: AgentPubKey) -> ExternResult<ActionHash> {
    let my_pubkey = agent_info()?.agent_initial_pubkey;
    if my_pubkey == target {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot connect to self".into()
        )));
    }

    // Create a request link
    create_link(my_pubkey, target, LinkTypes::AgentToOutgoingRequest, ())
}

#[hdk_extern]
pub fn accept_connection(request_action: ActionHash) -> ExternResult<ActionHash> {
    let record = get(request_action.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Request not found".into())
    ))?;
    let action = record.action();

    if let Action::CreateLink(create_link_action) = action {
        let requester = create_link_action
            .base_address
            .clone()
            .into_agent_pub_key()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Base address is not an AgentPubKey".into()
            )))?;
        let me = agent_info()?.agent_initial_pubkey;

        // Create bidirectional connection links
        create_link(
            me.clone(),
            requester.clone(),
            LinkTypes::AgentToConnection,
            (),
        )?;
        create_link(requester, me, LinkTypes::AgentToConnection, ())
    } else {
        Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid request action".into()
        )))
    }
}
