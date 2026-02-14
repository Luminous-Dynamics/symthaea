//! Communications Coordinator Zome
//! Offline-first emergency messaging with store-and-forward semantics

use emergency_comms_integrity::*;
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

    messages.sort_by_key(|a| a.action().timestamp());
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

    messages.sort_by_key(|a| a.action().timestamp());
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

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Coordinator input struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn send_message_input_serde_roundtrip() {
        let input = SendMessageInput {
            channel_hash: Some(ActionHash::from_raw_36(vec![0u8; 36])),
            priority: MessagePriority::Flash,
            content: "Evacuation notice for Zone A".to_string(),
            location: Some((29.76, -95.37)),
            ttl_hours: 24,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SendMessageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.priority, MessagePriority::Flash);
        assert_eq!(decoded.content, "Evacuation notice for Zone A");
        assert!(decoded.channel_hash.is_some());
        assert_eq!(decoded.location, Some((29.76, -95.37)));
        assert_eq!(decoded.ttl_hours, 24);
    }

    #[test]
    fn send_message_input_minimal_serde() {
        let input = SendMessageInput {
            channel_hash: None,
            priority: MessagePriority::Routine,
            content: "Status check".to_string(),
            location: None,
            ttl_hours: 1,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SendMessageInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.channel_hash.is_none());
        assert!(decoded.location.is_none());
        assert_eq!(decoded.ttl_hours, 1);
    }

    #[test]
    fn create_channel_input_serde_roundtrip() {
        let input = CreateChannelInput {
            name: "Medical Ops Channel".to_string(),
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            channel_type: ChannelType::Medical,
            participants: vec![
                AgentPubKey::from_raw_36(vec![1u8; 36]),
                AgentPubKey::from_raw_36(vec![2u8; 36]),
            ],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateChannelInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Medical Ops Channel");
        assert_eq!(decoded.channel_type, ChannelType::Medical);
        assert_eq!(decoded.participants.len(), 2);
    }

    #[test]
    fn broadcast_input_serde_roundtrip() {
        let input = BroadcastInput {
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            broadcast_type: BroadcastType::Evacuation,
            content: "Evacuate Zone B immediately".to_string(),
            target_area: (29.76, -95.37, 10.0),
            expires_at: Timestamp::from_micros(1000000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: BroadcastInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.broadcast_type, BroadcastType::Evacuation);
        assert_eq!(decoded.content, "Evacuate Zone B immediately");
        assert_eq!(decoded.target_area.0, 29.76);
        assert_eq!(decoded.target_area.1, -95.37);
        assert_eq!(decoded.target_area.2, 10.0);
    }

    // ========================================================================
    // Integrity enum serde tests (all variants)
    // ========================================================================

    #[test]
    fn message_priority_all_variants_serde() {
        let variants = vec![
            MessagePriority::Flash,
            MessagePriority::Immediate,
            MessagePriority::Priority,
            MessagePriority::Routine,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: MessagePriority = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn channel_type_all_variants_serde() {
        let variants = vec![
            ChannelType::Command,
            ChannelType::Operations,
            ChannelType::Logistics,
            ChannelType::Medical,
            ChannelType::Public,
            ChannelType::Volunteer,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: ChannelType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn broadcast_type_all_variants_serde() {
        let variants = vec![
            BroadcastType::Evacuation,
            BroadcastType::ShelterInPlace,
            BroadcastType::AllClear,
            BroadcastType::ResourceDrop,
            BroadcastType::MedicalAlert,
            BroadcastType::WeatherWarning,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: BroadcastType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // Entry type serde roundtrip tests
    // ========================================================================

    #[test]
    fn emergency_message_full_serde_roundtrip() {
        let msg = EmergencyMessage {
            sender: AgentPubKey::from_raw_36(vec![1u8; 36]),
            channel_hash: Some(ActionHash::from_raw_36(vec![2u8; 36])),
            priority: MessagePriority::Flash,
            content: "Flood warning for district 5".to_string(),
            location: Some((40.7128, -74.0060)),
            created_at: Timestamp::from_micros(1_700_000_000),
            ttl_hours: 48,
            hop_count: 3,
            synced: true,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: EmergencyMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.sender, msg.sender);
        assert_eq!(decoded.channel_hash, msg.channel_hash);
        assert_eq!(decoded.priority, MessagePriority::Flash);
        assert_eq!(decoded.content, "Flood warning for district 5");
        assert_eq!(decoded.location, Some((40.7128, -74.0060)));
        assert_eq!(decoded.ttl_hours, 48);
        assert_eq!(decoded.hop_count, 3);
        assert!(decoded.synced);
    }

    #[test]
    fn emergency_message_none_options_serde() {
        let msg = EmergencyMessage {
            sender: AgentPubKey::from_raw_36(vec![0u8; 36]),
            channel_hash: None,
            priority: MessagePriority::Routine,
            content: "Test".to_string(),
            location: None,
            created_at: Timestamp::from_micros(0),
            ttl_hours: 1,
            hop_count: 0,
            synced: false,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: EmergencyMessage = serde_json::from_str(&json).unwrap();
        assert!(decoded.channel_hash.is_none());
        assert!(decoded.location.is_none());
        assert!(!decoded.synced);
    }

    #[test]
    fn emergency_channel_serde_roundtrip() {
        let channel = EmergencyChannel {
            name: "Command Post Alpha".to_string(),
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            channel_type: ChannelType::Command,
            participants: vec![
                AgentPubKey::from_raw_36(vec![1u8; 36]),
                AgentPubKey::from_raw_36(vec![2u8; 36]),
            ],
            created_by: AgentPubKey::from_raw_36(vec![1u8; 36]),
        };
        let json = serde_json::to_string(&channel).unwrap();
        let decoded: EmergencyChannel = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Command Post Alpha");
        assert_eq!(decoded.channel_type, ChannelType::Command);
        assert_eq!(decoded.participants.len(), 2);
    }

    #[test]
    fn broadcast_entry_serde_roundtrip() {
        let broadcast = Broadcast {
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            broadcast_type: BroadcastType::WeatherWarning,
            content: "Tornado warning for county area".to_string(),
            target_area: (35.0, -97.0, 25.0),
            issued_by: AgentPubKey::from_raw_36(vec![1u8; 36]),
            expires_at: Timestamp::from_micros(9_999_999),
        };
        let json = serde_json::to_string(&broadcast).unwrap();
        let decoded: Broadcast = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.broadcast_type, BroadcastType::WeatherWarning);
        assert_eq!(decoded.content, "Tornado warning for county area");
        assert_eq!(decoded.target_area, (35.0, -97.0, 25.0));
    }

    // ========================================================================
    // Clone/equality tests
    // ========================================================================

    #[test]
    fn emergency_message_clone_equals_original() {
        let msg = EmergencyMessage {
            sender: AgentPubKey::from_raw_36(vec![0u8; 36]),
            channel_hash: Some(ActionHash::from_raw_36(vec![0u8; 36])),
            priority: MessagePriority::Immediate,
            content: "Clone test".to_string(),
            location: Some((0.0, 0.0)),
            created_at: Timestamp::from_micros(0),
            ttl_hours: 1,
            hop_count: 0,
            synced: false,
        };
        let cloned = msg.clone();
        assert_eq!(msg, cloned);
    }

    #[test]
    fn emergency_message_ne_different_content() {
        let a = EmergencyMessage {
            sender: AgentPubKey::from_raw_36(vec![0u8; 36]),
            channel_hash: None,
            priority: MessagePriority::Routine,
            content: "A".to_string(),
            location: None,
            created_at: Timestamp::from_micros(0),
            ttl_hours: 1,
            hop_count: 0,
            synced: false,
        };
        let mut b = a.clone();
        b.content = "B".to_string();
        assert_ne!(a, b);
    }

    #[test]
    fn emergency_channel_clone_equals_original() {
        let channel = EmergencyChannel {
            name: "Test".to_string(),
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            channel_type: ChannelType::Public,
            participants: vec![AgentPubKey::from_raw_36(vec![0u8; 36])],
            created_by: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(channel, channel.clone());
    }

    #[test]
    fn broadcast_clone_equals_original() {
        let broadcast = Broadcast {
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            broadcast_type: BroadcastType::AllClear,
            content: "All clear".to_string(),
            target_area: (0.0, 0.0, 1.0),
            issued_by: AgentPubKey::from_raw_36(vec![0u8; 36]),
            expires_at: Timestamp::from_micros(0),
        };
        assert_eq!(broadcast, broadcast.clone());
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn send_message_input_unicode_content_serde() {
        let input = SendMessageInput {
            channel_hash: None,
            priority: MessagePriority::Flash,
            content: "\u{1F6A8} \u{7D27}\u{6025}\u{901A}\u{4FE1} \u{043D}\u{0435}\u{043E}\u{0442}\u{043B}\u{043E}\u{0436}\u{043D}\u{044B}\u{0439}".to_string(),
            location: None,
            ttl_hours: 12,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SendMessageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.content, input.content);
    }

    #[test]
    fn send_message_input_max_ttl_serde() {
        let input = SendMessageInput {
            channel_hash: None,
            priority: MessagePriority::Routine,
            content: "Max ttl".to_string(),
            location: None,
            ttl_hours: u8::MAX,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SendMessageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.ttl_hours, u8::MAX);
    }

    #[test]
    fn broadcast_input_negative_coordinates_serde() {
        let input = BroadcastInput {
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            broadcast_type: BroadcastType::ShelterInPlace,
            content: "Stay indoors".to_string(),
            target_area: (-33.8688, 151.2093, 5.5),
            expires_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: BroadcastInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.target_area.0, -33.8688);
        assert_eq!(decoded.target_area.1, 151.2093);
        assert_eq!(decoded.target_area.2, 5.5);
    }

    #[test]
    fn emergency_message_boundary_hop_count() {
        let msg = EmergencyMessage {
            sender: AgentPubKey::from_raw_36(vec![0u8; 36]),
            channel_hash: None,
            priority: MessagePriority::Routine,
            content: "Hop test".to_string(),
            location: None,
            created_at: Timestamp::from_micros(0),
            ttl_hours: 1,
            hop_count: u8::MAX,
            synced: false,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: EmergencyMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hop_count, u8::MAX);
    }

    #[test]
    fn create_channel_input_empty_name_roundtrip() {
        // The struct itself allows empty name; validation rejects it
        let input = CreateChannelInput {
            name: "".to_string(),
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            channel_type: ChannelType::Volunteer,
            participants: vec![AgentPubKey::from_raw_36(vec![0u8; 36])],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateChannelInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "");
    }

    #[test]
    fn emergency_message_location_boundary_values_serde() {
        let msg = EmergencyMessage {
            sender: AgentPubKey::from_raw_36(vec![0u8; 36]),
            channel_hash: None,
            priority: MessagePriority::Priority,
            content: "Boundary".to_string(),
            location: Some((-90.0, 180.0)),
            created_at: Timestamp::from_micros(0),
            ttl_hours: 1,
            hop_count: 0,
            synced: false,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: EmergencyMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.location, Some((-90.0, 180.0)));
    }

    #[test]
    fn emergency_channel_empty_participants_vec_serde() {
        // Struct allows empty participants; coordinator validation rejects
        let channel = EmergencyChannel {
            name: "Empty".to_string(),
            disaster_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            channel_type: ChannelType::Logistics,
            participants: vec![],
            created_by: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&channel).unwrap();
        let decoded: EmergencyChannel = serde_json::from_str(&json).unwrap();
        assert!(decoded.participants.is_empty());
    }
}
