// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;

/// Simple message structure for broadcasting
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct MessageInput {
    pub content: String,
}

/// Message with metadata for returning to clients
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub content: String,
    pub timestamp: Timestamp,
    pub author: AgentPubKey,
}

/// Broadcast a message (simplified - just returns confirmation for now)
///
/// In a full implementation, this would create a DHT entry.
/// For now, we're creating a working stub to test the architecture.
#[hdk_extern]
pub fn broadcast_message(input: MessageInput) -> ExternResult<Message> {
    let info = agent_info()?;

    // Create message response
    let message = Message {
        content: input.content,
        timestamp: sys_time()?,
        author: info.agent_initial_pubkey,
    };

    Ok(message)
}

/// Get all messages (stub implementation)
///
/// Returns an empty list for now. Full DHT query implementation
/// will be added once basic architecture is working.
#[hdk_extern]
pub fn get_all_messages(_: ()) -> ExternResult<Vec<Message>> {
    Ok(vec![])
}

/// Get messages from a specific author (stub implementation)
#[hdk_extern]
pub fn get_messages_by_author(_author: AgentPubKey) -> ExternResult<Vec<Message>> {
    Ok(vec![])
}

/// Get the current agent's public key (utility function)
#[hdk_extern]
pub fn whoami(_: ()) -> ExternResult<AgentPubKey> {
    agent_info().map(|info| info.agent_initial_pubkey)
}
