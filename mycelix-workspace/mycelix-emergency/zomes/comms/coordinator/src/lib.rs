// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Communications Coordinator Zome
//! Offline-first emergency messaging with store-and-forward semantics

use comms_integrity::*;
use hdk::prelude::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Send an emergency message (works offline - stored locally, synced later)
#[hdk_extern]
pub fn send_message(input: SendMessageInput) -> ExternResult<Record> {
    if input.content.is_empty() || input.content.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Content must be 1-4096 characters".into()
        )));
    }
    if input.ttl_hours == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "TTL must be at least 1 hour".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let message = EmergencyMessage {
        sender: agent_info.agent_initial_pubkey.clone(),
        channel_hash: input.channel_hash.clone(),
        priority: input.priority,
        content: input.content,
        location: input.location,
        created_at: now,
        ttl_hours: input.ttl_hours,
        hop_count: 0,
        synced: false,
    };

    let action_hash = create_entry(&EntryTypes::EmergencyMessage(message))?;

    // Link to channel if provided
    if let Some(channel_hash) = input.channel_hash {
        create_link(
            channel_hash,
            action_hash.clone(),
            LinkTypes::ChannelToMessage,
            (),
        )?;
    }

    // Link agent to message
    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToMessage,
        (),
    )?;

    // Track as unsynced for offline-first operation
    create_entry(&EntryTypes::Anchor(Anchor("unsynced_messages".to_string())))?;
    create_link(
        anchor_hash("unsynced_messages")?,
        action_hash.clone(),
        LinkTypes::UnsyncedMessages,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created message".into()
    )))
}

/// Input for sending a message
#[derive(Serialize, Deserialize, Debug)]
pub struct SendMessageInput {
    pub channel_hash: Option<ActionHash>,
    pub priority: MessagePriority,
    pub content: String,
    pub location: Option<(f64, f64)>,
    pub ttl_hours: u8,
}

/// Create an emergency communication channel
#[hdk_extern]
pub fn create_channel(input: CreateChannelInput) -> ExternResult<Record> {
    if input.name.is_empty() || input.name.len() > 128 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Channel name must be 1-128 characters".into()
        )));
    }
    if input.participants.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Channel must have at least one participant".into()
        )));
    }

    let agent_info = agent_info()?;

    let channel = EmergencyChannel {
        name: input.name,
        disaster_hash: input.disaster_hash.clone(),
        channel_type: input.channel_type,
        participants: input.participants,
        created_by: agent_info.agent_initial_pubkey,
    };

    let action_hash = create_entry(&EntryTypes::EmergencyChannel(channel))?;

    // Link disaster to channel
    create_link(
        input.disaster_hash,
        action_hash.clone(),
        LinkTypes::DisasterToChannel,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created channel".into()
    )))
}

/// Input for creating a channel
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateChannelInput {
    pub name: String,
    pub disaster_hash: ActionHash,
    pub channel_type: ChannelType,
    pub participants: Vec<AgentPubKey>,
}

/// Issue an emergency broadcast
#[hdk_extern]
pub fn broadcast(input: BroadcastInput) -> ExternResult<Record> {
    if input.content.is_empty() || input.content.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Broadcast content must be 1-2048 characters".into()
        )));
    }

    let agent_info = agent_info()?;

    let broadcast = Broadcast {
        disaster_hash: input.disaster_hash.clone(),
        broadcast_type: input.broadcast_type,
        content: input.content,
        target_area: input.target_area,
        issued_by: agent_info.agent_initial_pubkey,
        expires_at: input.expires_at,
    };

    let action_hash = create_entry(&EntryTypes::Broadcast(broadcast))?;

    // Link disaster to broadcast
    create_link(
        input.disaster_hash,
        action_hash.clone(),
        LinkTypes::DisasterToBroadcast,
        (),
    )?;

    // Link to active broadcasts
    create_entry(&EntryTypes::Anchor(Anchor("active_broadcasts".to_string())))?;
    create_link(
        anchor_hash("active_broadcasts")?,
        action_hash.clone(),
        LinkTypes::ActiveBroadcasts,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created broadcast".into()
    )))
}

/// Input for issuing a broadcast
#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastInput {
    pub disaster_hash: ActionHash,
    pub broadcast_type: BroadcastType,
    pub content: String,
    pub target_area: (f64, f64, f32),
    pub expires_at: Timestamp,
}

/// Get messages for a channel
#[hdk_extern]
pub fn get_channel_messages(channel_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(channel_hash, LinkTypes::ChannelToMessage)?,
        GetStrategy::default(),
    )?;

    let mut messages = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            messages.push(record);
        }
    }

    messages.sort_by(|a, b| a.action().timestamp().cmp(&b.action().timestamp()));
    Ok(messages)
}

/// Get active broadcasts (non-expired)
#[hdk_extern]
pub fn get_active_broadcasts(_: ()) -> ExternResult<Vec<Record>> {
    let now = sys_time()?;

    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("active_broadcasts")?,
            LinkTypes::ActiveBroadcasts,
        )?,
        GetStrategy::default(),
    )?;

    let mut broadcasts = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
            if let Some(broadcast) = record
                .entry()
                .to_app_option::<Broadcast>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if broadcast.expires_at > now {
                    broadcasts.push(record);
                } else {
                    // Clean up expired broadcast link
                    delete_link(link.create_link_hash, GetOptions::default())?;
                }
            }
        }
    }

    Ok(broadcasts)
}

/// Get all unsynced messages (for offline-first sync)
#[hdk_extern]
pub fn get_unsynced_messages(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("unsynced_messages")?,
            LinkTypes::UnsyncedMessages,
        )?,
        GetStrategy::default(),
    )?;

    let mut messages = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            messages.push(record);
        }
    }

    messages.sort_by(|a, b| a.action().timestamp().cmp(&b.action().timestamp()));
    Ok(messages)
}

/// Mark a message as synced (remove from unsynced list)
#[hdk_extern]
pub fn mark_synced(message_hash: ActionHash) -> ExternResult<Record> {
    // Get and update the message
    let current_record = get(message_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Message not found".into())
    ))?;

    let current_message: EmergencyMessage = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid message entry".into()
        )))?;

    let updated_message = EmergencyMessage {
        synced: true,
        ..current_message
    };

    let new_action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::EmergencyMessage(updated_message),
    )?;

    // Remove from unsynced list
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("unsynced_messages")?,
            LinkTypes::UnsyncedMessages,
        )?,
        GetStrategy::default(),
    )?;
    for link in links {
        let target = ActionHash::try_from(link.target.clone());
        if let Ok(target_hash) = target {
            if target_hash == message_hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    get(new_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated message".into()
    )))
}
