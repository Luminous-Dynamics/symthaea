//! Communications Integrity Zome
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
            tag,
            action: _,
        } => {
            let tag_len = tag.0.len();
            match link_type {
                LinkTypes::ChannelToMessage => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::DisasterToChannel => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::DisasterToBroadcast => {
                    // Broadcast links may carry area metadata in tags
                    if tag_len > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 512 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::AgentToMessage => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::UnsyncedMessages => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ActiveBroadcasts => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
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
    if msg.content.trim().is_empty() {
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
        if !lat.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "location latitude must be a finite number".into(),
            ));
        }
        if !lon.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "location longitude must be a finite number".into(),
            ));
        }
        if !(-90.0..=90.0).contains(&lat) {
            return Ok(ValidateCallbackResult::Invalid(
                "Latitude must be between -90 and 90".into(),
            ));
        }
        if !(-180.0..=180.0).contains(&lon) {
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
    if channel.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Channel name cannot be empty".into(),
        ));
    }
    if channel.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Channel name too long (max 256 chars)".into(),
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
    if broadcast.content.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Broadcast content cannot be empty".into(),
        ));
    }
    if broadcast.content.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Broadcast content too long (max 4096 bytes)".into(),
        ));
    }
    let (lat, lon, radius) = broadcast.target_area;
    if !lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "target_area latitude must be a finite number".into(),
        ));
    }
    if !lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "target_area longitude must be a finite number".into(),
        ));
    }
    if !radius.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "target_area radius must be a finite number".into(),
        ));
    }
    if !(-90.0..=90.0).contains(&lat) {
        return Ok(ValidateCallbackResult::Invalid(
            "Target area latitude must be between -90 and 90".into(),
        ));
    }
    if !(-180.0..=180.0).contains(&lon) {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RESULT HELPERS
    // ========================================================================

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    // ========================================================================
    // CONSTRUCTION HELPERS
    // ========================================================================

    fn fake_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(0)
    }

    // ========================================================================
    // FACTORY FUNCTIONS (valid defaults)
    // ========================================================================

    fn make_message() -> EmergencyMessage {
        EmergencyMessage {
            sender: fake_agent(),
            channel_hash: Some(fake_action_hash()),
            priority: MessagePriority::Immediate,
            content: "Requesting immediate evacuation at sector 7".into(),
            location: Some((32.9483, -96.7299)),
            created_at: fake_timestamp(),
            ttl_hours: 24,
            hop_count: 0,
            synced: false,
        }
    }

    fn make_channel() -> EmergencyChannel {
        EmergencyChannel {
            name: "Operations Channel Alpha".into(),
            disaster_hash: fake_action_hash(),
            channel_type: ChannelType::Operations,
            participants: vec![fake_agent()],
            created_by: fake_agent(),
        }
    }

    fn make_broadcast() -> Broadcast {
        Broadcast {
            disaster_hash: fake_action_hash(),
            broadcast_type: BroadcastType::Evacuation,
            content: "Evacuate all residents within 5km radius".into(),
            target_area: (32.9483, -96.7299, 5.0),
            issued_by: fake_agent(),
            expires_at: Timestamp::from_micros(3_600_000_000),
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP: Anchor
    // ========================================================================

    #[test]
    fn anchor_serde_roundtrip() {
        let anchor = Anchor("all_channels".into());
        let json = serde_json::to_string(&anchor).expect("serialize");
        let parsed: Anchor = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(anchor, parsed);
    }

    #[test]
    fn anchor_empty_string_serde_roundtrip() {
        let anchor = Anchor("".into());
        let json = serde_json::to_string(&anchor).expect("serialize");
        let parsed: Anchor = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(anchor, parsed);
    }

    #[test]
    fn anchor_unicode_serde_roundtrip() {
        let anchor = Anchor("emergencias_\u{00e9}\u{00e7}\u{00f1}".into());
        let json = serde_json::to_string(&anchor).expect("serialize");
        let parsed: Anchor = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(anchor, parsed);
    }

    // ========================================================================
    // SERDE ROUNDTRIP: MessagePriority
    // ========================================================================

    #[test]
    fn message_priority_all_variants_serde_roundtrip() {
        let variants = vec![
            MessagePriority::Flash,
            MessagePriority::Immediate,
            MessagePriority::Priority,
            MessagePriority::Routine,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: MessagePriority = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP: ChannelType
    // ========================================================================

    #[test]
    fn channel_type_all_variants_serde_roundtrip() {
        let variants = vec![
            ChannelType::Command,
            ChannelType::Operations,
            ChannelType::Logistics,
            ChannelType::Medical,
            ChannelType::Public,
            ChannelType::Volunteer,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: ChannelType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP: BroadcastType
    // ========================================================================

    #[test]
    fn broadcast_type_all_variants_serde_roundtrip() {
        let variants = vec![
            BroadcastType::Evacuation,
            BroadcastType::ShelterInPlace,
            BroadcastType::AllClear,
            BroadcastType::ResourceDrop,
            BroadcastType::MedicalAlert,
            BroadcastType::WeatherWarning,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: BroadcastType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP: EmergencyMessage
    // ========================================================================

    #[test]
    fn emergency_message_serde_roundtrip() {
        let msg = make_message();
        let json = serde_json::to_string(&msg).expect("serialize");
        let parsed: EmergencyMessage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(msg, parsed);
    }

    #[test]
    fn emergency_message_no_channel_serde_roundtrip() {
        let mut msg = make_message();
        msg.channel_hash = None;
        let json = serde_json::to_string(&msg).expect("serialize");
        let parsed: EmergencyMessage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(msg, parsed);
    }

    #[test]
    fn emergency_message_no_location_serde_roundtrip() {
        let mut msg = make_message();
        msg.location = None;
        let json = serde_json::to_string(&msg).expect("serialize");
        let parsed: EmergencyMessage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(msg, parsed);
    }

    // ========================================================================
    // SERDE ROUNDTRIP: EmergencyChannel
    // ========================================================================

    #[test]
    fn emergency_channel_serde_roundtrip() {
        let channel = make_channel();
        let json = serde_json::to_string(&channel).expect("serialize");
        let parsed: EmergencyChannel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(channel, parsed);
    }

    #[test]
    fn emergency_channel_multiple_participants_serde_roundtrip() {
        let mut channel = make_channel();
        channel.participants = vec![
            AgentPubKey::from_raw_36(vec![1u8; 36]),
            AgentPubKey::from_raw_36(vec![2u8; 36]),
            AgentPubKey::from_raw_36(vec![3u8; 36]),
        ];
        let json = serde_json::to_string(&channel).expect("serialize");
        let parsed: EmergencyChannel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(channel, parsed);
    }

    // ========================================================================
    // SERDE ROUNDTRIP: Broadcast
    // ========================================================================

    #[test]
    fn broadcast_serde_roundtrip() {
        let broadcast = make_broadcast();
        let json = serde_json::to_string(&broadcast).expect("serialize");
        let parsed: Broadcast = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(broadcast, parsed);
    }

    // ========================================================================
    // validate_create_message: VALID CASES
    // ========================================================================

    #[test]
    fn create_message_valid_passes() {
        let result = validate_create_message(fake_create(), make_message());
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_all_priorities_pass() {
        let priorities = vec![
            MessagePriority::Flash,
            MessagePriority::Immediate,
            MessagePriority::Priority,
            MessagePriority::Routine,
        ];
        for priority in priorities {
            let mut msg = make_message();
            msg.priority = priority;
            let result = validate_create_message(fake_create(), msg);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn create_message_no_channel_hash_passes() {
        let mut msg = make_message();
        msg.channel_hash = None;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_no_location_passes() {
        let mut msg = make_message();
        msg.location = None;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_synced_true_passes() {
        let mut msg = make_message();
        msg.synced = true;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_max_hop_count_passes() {
        let mut msg = make_message();
        msg.hop_count = u8::MAX;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_max_ttl_passes() {
        let mut msg = make_message();
        msg.ttl_hours = u8::MAX;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_ttl_one_passes() {
        let mut msg = make_message();
        msg.ttl_hours = 1;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_content_exactly_4096_bytes_passes() {
        let mut msg = make_message();
        msg.content = "a".repeat(4096);
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_single_char_content_passes() {
        let mut msg = make_message();
        msg.content = "!".into();
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_unicode_content_passes() {
        let mut msg = make_message();
        msg.content = "\u{1F6A8} Emergencia: evacuar inmediatamente \u{1F6B6}".into();
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // validate_create_message: CONTENT ERRORS
    // ========================================================================

    #[test]
    fn create_message_empty_content_rejected() {
        let mut msg = make_message();
        msg.content = "".into();
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Message content cannot be empty");
    }

    #[test]
    fn create_message_content_exceeds_4096_bytes_rejected() {
        let mut msg = make_message();
        msg.content = "x".repeat(4097);
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Message content cannot exceed 4096 bytes"
        );
    }

    #[test]
    fn create_message_content_way_over_limit_rejected() {
        let mut msg = make_message();
        msg.content = "z".repeat(100_000);
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Message content cannot exceed 4096 bytes"
        );
    }

    // ========================================================================
    // validate_create_message: TTL ERRORS
    // ========================================================================

    #[test]
    fn create_message_ttl_zero_rejected() {
        let mut msg = make_message();
        msg.ttl_hours = 0;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "TTL must be at least 1 hour");
    }

    // ========================================================================
    // validate_create_message: LOCATION ERRORS
    // ========================================================================

    #[test]
    fn create_message_latitude_below_negative_90_rejected() {
        let mut msg = make_message();
        msg.location = Some((-90.1, 0.0));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn create_message_latitude_above_90_rejected() {
        let mut msg = make_message();
        msg.location = Some((90.1, 0.0));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn create_message_longitude_below_negative_180_rejected() {
        let mut msg = make_message();
        msg.location = Some((0.0, -180.1));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn create_message_longitude_above_180_rejected() {
        let mut msg = make_message();
        msg.location = Some((0.0, 180.1));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn create_message_latitude_boundary_values_pass() {
        for lat in [-90.0, 0.0, 90.0] {
            let mut msg = make_message();
            msg.location = Some((lat, 0.0));
            let result = validate_create_message(fake_create(), msg);
            assert!(is_valid(&result), "lat={} should pass", lat);
        }
    }

    #[test]
    fn create_message_longitude_boundary_values_pass() {
        for lon in [-180.0, 0.0, 180.0] {
            let mut msg = make_message();
            msg.location = Some((0.0, lon));
            let result = validate_create_message(fake_create(), msg);
            assert!(is_valid(&result), "lon={} should pass", lon);
        }
    }

    #[test]
    fn create_message_extreme_invalid_lat_rejected() {
        let mut msg = make_message();
        msg.location = Some((f64::MAX, 0.0));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
    }

    #[test]
    fn create_message_extreme_invalid_lon_rejected() {
        let mut msg = make_message();
        msg.location = Some((0.0, f64::MIN));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
    }

    #[test]
    fn create_message_nan_latitude_rejected() {
        let mut msg = make_message();
        msg.location = Some((f64::NAN, 0.0));
        let result = validate_create_message(fake_create(), msg);
        // NaN fails the contains() check because NaN != NaN
        assert!(is_invalid(&result));
    }

    #[test]
    fn create_message_nan_longitude_rejected() {
        let mut msg = make_message();
        msg.location = Some((0.0, f64::NAN));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
    }

    #[test]
    fn create_message_infinity_latitude_rejected() {
        let mut msg = make_message();
        msg.location = Some((f64::INFINITY, 0.0));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
    }

    #[test]
    fn create_message_neg_infinity_longitude_rejected() {
        let mut msg = make_message();
        msg.location = Some((0.0, f64::NEG_INFINITY));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // validate_create_channel: VALID CASES
    // ========================================================================

    #[test]
    fn create_channel_valid_passes() {
        let result = validate_create_channel(fake_create(), make_channel());
        assert!(is_valid(&result));
    }

    #[test]
    fn create_channel_all_types_pass() {
        let channel_types = vec![
            ChannelType::Command,
            ChannelType::Operations,
            ChannelType::Logistics,
            ChannelType::Medical,
            ChannelType::Public,
            ChannelType::Volunteer,
        ];
        for ct in channel_types {
            let mut ch = make_channel();
            ch.channel_type = ct;
            let result = validate_create_channel(fake_create(), ch);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn create_channel_multiple_participants_passes() {
        let mut ch = make_channel();
        ch.participants = vec![
            AgentPubKey::from_raw_36(vec![1u8; 36]),
            AgentPubKey::from_raw_36(vec![2u8; 36]),
            AgentPubKey::from_raw_36(vec![3u8; 36]),
        ];
        let result = validate_create_channel(fake_create(), ch);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_channel_single_participant_passes() {
        let mut ch = make_channel();
        ch.participants = vec![fake_agent()];
        let result = validate_create_channel(fake_create(), ch);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_channel_unicode_name_passes() {
        let mut ch = make_channel();
        ch.name =
            "\u{7D27}\u{6025}\u{901A}\u{4FE1} \u{30C1}\u{30E3}\u{30F3}\u{30CD}\u{30EB}".into();
        let result = validate_create_channel(fake_create(), ch);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_channel_name_at_limit_passes() {
        let mut ch = make_channel();
        ch.name = "A".repeat(256);
        let result = validate_create_channel(fake_create(), ch);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_channel_name_over_limit_rejected() {
        let mut ch = make_channel();
        ch.name = "A".repeat(257);
        let result = validate_create_channel(fake_create(), ch);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Channel name too long (max 256 chars)"
        );
    }

    #[test]
    fn create_channel_very_long_name_rejected() {
        let mut ch = make_channel();
        ch.name = "A".repeat(1000);
        let result = validate_create_channel(fake_create(), ch);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Channel name too long (max 256 chars)"
        );
    }

    #[test]
    fn create_broadcast_content_at_limit_passes() {
        let mut b = make_broadcast();
        b.content = "c".repeat(4096);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_broadcast_content_over_limit_rejected() {
        let mut b = make_broadcast();
        b.content = "c".repeat(4097);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Broadcast content too long (max 4096 bytes)"
        );
    }

    // ========================================================================
    // validate_create_channel: ERROR PATHS
    // ========================================================================

    #[test]
    fn create_channel_empty_name_rejected() {
        let mut ch = make_channel();
        ch.name = "".into();
        let result = validate_create_channel(fake_create(), ch);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Channel name cannot be empty");
    }

    #[test]
    fn create_channel_empty_participants_rejected() {
        let mut ch = make_channel();
        ch.participants = vec![];
        let result = validate_create_channel(fake_create(), ch);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Channel must have at least one participant"
        );
    }

    #[test]
    fn create_channel_both_errors_returns_first() {
        // When both name and participants are invalid, the name check runs first
        let mut ch = make_channel();
        ch.name = "".into();
        ch.participants = vec![];
        let result = validate_create_channel(fake_create(), ch);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Channel name cannot be empty");
    }

    // ========================================================================
    // validate_create_broadcast: VALID CASES
    // ========================================================================

    #[test]
    fn create_broadcast_valid_passes() {
        let result = validate_create_broadcast(fake_create(), make_broadcast());
        assert!(is_valid(&result));
    }

    #[test]
    fn create_broadcast_all_types_pass() {
        let broadcast_types = vec![
            BroadcastType::Evacuation,
            BroadcastType::ShelterInPlace,
            BroadcastType::AllClear,
            BroadcastType::ResourceDrop,
            BroadcastType::MedicalAlert,
            BroadcastType::WeatherWarning,
        ];
        for bt in broadcast_types {
            let mut b = make_broadcast();
            b.broadcast_type = bt;
            let result = validate_create_broadcast(fake_create(), b);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn create_broadcast_large_radius_passes() {
        let mut b = make_broadcast();
        b.target_area = (0.0, 0.0, f32::MAX);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_broadcast_tiny_radius_passes() {
        let mut b = make_broadcast();
        b.target_area = (0.0, 0.0, 0.001);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_broadcast_unicode_content_passes() {
        let mut b = make_broadcast();
        b.content = "\u{26A0}\u{FE0F} \u{0412}\u{043D}\u{0438}\u{043C}\u{0430}\u{043D}\u{0438}\u{0435}: \u{044D}\u{0432}\u{0430}\u{043A}\u{0443}\u{0430}\u{0446}\u{0438}\u{044F}".into();
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_broadcast_lat_boundary_values_pass() {
        for lat in [-90.0_f64, 0.0, 90.0] {
            let mut b = make_broadcast();
            b.target_area = (lat, 0.0, 1.0);
            let result = validate_create_broadcast(fake_create(), b);
            assert!(is_valid(&result), "lat={} should pass", lat);
        }
    }

    #[test]
    fn create_broadcast_lon_boundary_values_pass() {
        for lon in [-180.0_f64, 0.0, 180.0] {
            let mut b = make_broadcast();
            b.target_area = (0.0, lon, 1.0);
            let result = validate_create_broadcast(fake_create(), b);
            assert!(is_valid(&result), "lon={} should pass", lon);
        }
    }

    // ========================================================================
    // validate_create_broadcast: CONTENT ERRORS
    // ========================================================================

    #[test]
    fn create_broadcast_empty_content_rejected() {
        let mut b = make_broadcast();
        b.content = "".into();
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Broadcast content cannot be empty");
    }

    // ========================================================================
    // validate_create_broadcast: LATITUDE ERRORS
    // ========================================================================

    #[test]
    fn create_broadcast_lat_below_negative_90_rejected() {
        let mut b = make_broadcast();
        b.target_area = (-90.1, 0.0, 1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Target area latitude must be between -90 and 90"
        );
    }

    #[test]
    fn create_broadcast_lat_above_90_rejected() {
        let mut b = make_broadcast();
        b.target_area = (90.1, 0.0, 1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Target area latitude must be between -90 and 90"
        );
    }

    #[test]
    fn create_broadcast_nan_lat_rejected() {
        let mut b = make_broadcast();
        b.target_area = (f64::NAN, 0.0, 1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
    }

    #[test]
    fn create_broadcast_infinity_lat_rejected() {
        let mut b = make_broadcast();
        b.target_area = (f64::INFINITY, 0.0, 1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // validate_create_broadcast: LONGITUDE ERRORS
    // ========================================================================

    #[test]
    fn create_broadcast_lon_below_negative_180_rejected() {
        let mut b = make_broadcast();
        b.target_area = (0.0, -180.1, 1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Target area longitude must be between -180 and 180"
        );
    }

    #[test]
    fn create_broadcast_lon_above_180_rejected() {
        let mut b = make_broadcast();
        b.target_area = (0.0, 180.1, 1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Target area longitude must be between -180 and 180"
        );
    }

    #[test]
    fn create_broadcast_nan_lon_rejected() {
        let mut b = make_broadcast();
        b.target_area = (0.0, f64::NAN, 1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
    }

    #[test]
    fn create_broadcast_neg_infinity_lon_rejected() {
        let mut b = make_broadcast();
        b.target_area = (0.0, f64::NEG_INFINITY, 1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // validate_create_broadcast: RADIUS ERRORS
    // ========================================================================

    #[test]
    fn create_broadcast_zero_radius_rejected() {
        let mut b = make_broadcast();
        b.target_area = (0.0, 0.0, 0.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Target area radius must be positive");
    }

    #[test]
    fn create_broadcast_negative_radius_rejected() {
        let mut b = make_broadcast();
        b.target_area = (0.0, 0.0, -5.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Target area radius must be positive");
    }

    #[test]
    fn create_broadcast_neg_infinity_radius_rejected() {
        let mut b = make_broadcast();
        b.target_area = (0.0, 0.0, f32::NEG_INFINITY);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
    }

    #[test]
    fn create_broadcast_nan_radius_rejected() {
        let mut b = make_broadcast();
        b.target_area = (0.0, 0.0, f32::NAN);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "target_area radius must be a finite number"
        );
    }

    // ========================================================================
    // validate_create_broadcast: VALIDATION ORDER
    // ========================================================================

    #[test]
    fn create_broadcast_empty_content_checked_before_coordinates() {
        let mut b = make_broadcast();
        b.content = "".into();
        b.target_area = (999.0, 999.0, -1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        // Content is checked first
        assert_eq!(invalid_msg(&result), "Broadcast content cannot be empty");
    }

    #[test]
    fn create_broadcast_lat_checked_before_lon() {
        let mut b = make_broadcast();
        b.target_area = (999.0, 999.0, 1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Target area latitude must be between -90 and 90"
        );
    }

    #[test]
    fn create_broadcast_lon_checked_before_radius() {
        let mut b = make_broadcast();
        b.target_area = (0.0, 999.0, -1.0);
        let result = validate_create_broadcast(fake_create(), b);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Target area longitude must be between -180 and 180"
        );
    }

    // ========================================================================
    // validate_create_message: VALIDATION ORDER
    // ========================================================================

    #[test]
    fn create_message_empty_content_checked_before_ttl() {
        let mut msg = make_message();
        msg.content = "".into();
        msg.ttl_hours = 0;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Message content cannot be empty");
    }

    #[test]
    fn create_message_content_length_checked_before_ttl() {
        let mut msg = make_message();
        msg.content = "x".repeat(5000);
        msg.ttl_hours = 0;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Message content cannot exceed 4096 bytes"
        );
    }

    #[test]
    fn create_message_ttl_checked_before_location() {
        let mut msg = make_message();
        msg.ttl_hours = 0;
        msg.location = Some((999.0, 999.0));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "TTL must be at least 1 hour");
    }

    #[test]
    fn create_message_lat_checked_before_lon() {
        let mut msg = make_message();
        msg.location = Some((999.0, 999.0));
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    // ========================================================================
    // EDGE CASES: Multi-byte unicode and content length
    // ========================================================================

    #[test]
    fn create_message_multibyte_unicode_at_boundary() {
        // .len() counts bytes, not characters. 4-byte emoji chars push past the limit faster.
        let mut msg = make_message();
        // Each emoji is 4 bytes, so 1024 emojis = 4096 bytes exactly
        msg.content = "\u{1F525}".repeat(1024);
        assert_eq!(msg.content.len(), 4096);
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_message_multibyte_unicode_over_boundary() {
        let mut msg = make_message();
        // 1025 four-byte emojis = 4100 bytes, over the 4096 limit
        msg.content = "\u{1F525}".repeat(1025);
        assert!(msg.content.len() > 4096);
        let result = validate_create_message(fake_create(), msg);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // EDGE CASES: Zero-length hop count
    // ========================================================================

    #[test]
    fn create_message_hop_count_zero_passes() {
        let mut msg = make_message();
        msg.hop_count = 0;
        let result = validate_create_message(fake_create(), msg);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag_len = tag.len();
        match link_type {
            LinkTypes::DisasterToBroadcast => {
                if tag_len > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            _ => {
                if tag_len > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    fn validate_delete_link_tag(tag: Vec<u8>) -> ExternResult<ValidateCallbackResult> {
        if tag.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Delete link tag too long (max 256 bytes)".into(),
            ));
        }
        Ok(ValidateCallbackResult::Valid)
    }

    // -- ChannelToMessage (256 limit) --

    #[test]
    fn channel_to_message_link_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::ChannelToMessage, vec![0u8; 100]);
        assert!(is_valid(&result));
    }

    #[test]
    fn channel_to_message_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::ChannelToMessage, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn channel_to_message_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::ChannelToMessage, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // -- DisasterToBroadcast (512 limit) --

    #[test]
    fn disaster_to_broadcast_link_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::DisasterToBroadcast, vec![0u8; 400]);
        assert!(is_valid(&result));
    }

    #[test]
    fn disaster_to_broadcast_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::DisasterToBroadcast, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn disaster_to_broadcast_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::DisasterToBroadcast, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    // -- UnsyncedMessages (256 limit) --

    #[test]
    fn unsynced_messages_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::UnsyncedMessages, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn unsynced_messages_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::UnsyncedMessages, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // -- Large tag DoS prevention --

    #[test]
    fn massive_link_tag_rejected_all_types() {
        let huge_tag = vec![0xFFu8; 10_000];
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::ChannelToMessage,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::DisasterToChannel,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::DisasterToBroadcast,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::AgentToMessage,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::UnsyncedMessages,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::ActiveBroadcasts,
            huge_tag.clone()
        )));
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_valid() {
        let result = validate_delete_link_tag(vec![0u8; 128]);
        assert!(is_valid(&result));
    }

    #[test]
    fn delete_link_tag_at_limit() {
        let result = validate_delete_link_tag(vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn delete_link_tag_over_limit() {
        let result = validate_delete_link_tag(vec![0u8; 257]);
        assert!(is_invalid(&result));
    }
}
