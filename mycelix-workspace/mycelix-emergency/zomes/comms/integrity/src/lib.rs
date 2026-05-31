// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Communications Integrity Zome
//! Offline-first emergency messaging with store-and-forward

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// An emergency message (offline-first capable)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencyMessage {
    pub sender: AgentPubKey,
    pub channel_hash: Option<ActionHash>,
    pub priority: MessagePriority,
    pub content: String,
    pub location: Option<(f64, f64)>,
    pub created_at: Timestamp,
    pub ttl_hours: u8,
    pub hop_count: u8,
    pub synced: bool,
}

/// Message priority levels (NATO-aligned)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MessagePriority {
    Flash,
    Immediate,
    Priority,
    Routine,
}

/// An emergency communication channel
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencyChannel {
    pub name: String,
    pub disaster_hash: ActionHash,
    pub channel_type: ChannelType,
    pub participants: Vec<AgentPubKey>,
    pub created_by: AgentPubKey,
}

/// Types of communication channels
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ChannelType {
    Command,
    Operations,
    Logistics,
    Medical,
    Public,
    Volunteer,
}

/// A broadcast message to an area
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Broadcast {
    pub disaster_hash: ActionHash,
    pub broadcast_type: BroadcastType,
    pub content: String,
    pub target_area: (f64, f64, f32),
    pub issued_by: AgentPubKey,
    pub expires_at: Timestamp,
}

/// Types of emergency broadcasts
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BroadcastType {
    Evacuation,
    ShelterInPlace,
    AllClear,
    ResourceDrop,
    MedicalAlert,
    WeatherWarning,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    EmergencyMessage(EmergencyMessage),
    EmergencyChannel(EmergencyChannel),
    Broadcast(Broadcast),
}

#[hdk_link_types]
pub enum LinkTypes {
    ChannelToMessage,
    DisasterToChannel,
    DisasterToBroadcast,
    AgentToMessage,
    UnsyncedMessages,
    ActiveBroadcasts,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencyMessage(msg) => validate_create_message(action, msg),
                EntryTypes::EmergencyChannel(channel) => validate_create_channel(action, channel),
                EntryTypes::Broadcast(broadcast) => validate_create_broadcast(action, broadcast),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencyMessage(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EmergencyChannel(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Broadcast(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ChannelToMessage => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DisasterToChannel => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DisasterToBroadcast => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToMessage => Ok(ValidateCallbackResult::Valid),
            LinkTypes::UnsyncedMessages => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveBroadcasts => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::UnsyncedMessages => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveBroadcasts => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_message(
    _action: Create,
    msg: EmergencyMessage,
) -> ExternResult<ValidateCallbackResult> {
    if msg.content.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Message content cannot be empty".into(),
        ));
    }
    if msg.content.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Message content cannot exceed 4096 bytes".into(),
        ));
    }
    if msg.ttl_hours == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "TTL must be at least 1 hour".into(),
        ));
    }
    if let Some((lat, lon)) = msg.location {
        if lat < -90.0 || lat > 90.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Latitude must be between -90 and 90".into(),
            ));
        }
        if lon < -180.0 || lon > 180.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Longitude must be between -180 and 180".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_channel(
    _action: Create,
    channel: EmergencyChannel,
) -> ExternResult<ValidateCallbackResult> {
    if channel.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Channel name cannot be empty".into(),
        ));
    }
    if channel.participants.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Channel must have at least one participant".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_broadcast(
    _action: Create,
    broadcast: Broadcast,
) -> ExternResult<ValidateCallbackResult> {
    if broadcast.content.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Broadcast content cannot be empty".into(),
        ));
    }
    let (lat, lon, radius) = broadcast.target_area;
    if lat < -90.0 || lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Target area latitude must be between -90 and 90".into(),
        ));
    }
    if lon < -180.0 || lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Target area longitude must be between -180 and 180".into(),
        ));
    }
    if radius <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Target area radius must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
