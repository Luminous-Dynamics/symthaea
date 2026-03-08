//! Triage Integrity Zome
//! Mass casualty triage records using START protocol

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A triage assessment record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TriageRecord {
    pub disaster_hash: ActionHash,
    pub patient_id: String,
    pub patient_hash: Option<ActionHash>,
    pub category: TriageCategory,
    pub injuries: String,
    pub location: String,
    pub timestamp: Timestamp,
    pub triaged_by: AgentPubKey,
    pub transport_priority: TransportPriority,
    pub notes: String,
}

/// START triage categories
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TriageCategory {
    /// Immediate - life-threatening, salvageable
    Immediate,
    /// Delayed - serious but can wait
    Delayed,
    /// Minor - walking wounded
    Minor,
    /// Expectant - unlikely to survive
    Expectant,
    /// Deceased
    Dead,
}

/// Transport priority
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransportPriority {
    Urgent,
    Priority,
    Routine,
    None,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    TriageRecord(TriageRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    DisasterToTriage,
    CategoryToTriage,
    PatientToTriage,
    AgentToTriage,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::TriageRecord(record) => validate_create_triage(action, record),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::TriageRecord(record) => validate_update_triage(action, record),
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
                LinkTypes::DisasterToTriage => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::CategoryToTriage => {
                    // Category links may store serialized category metadata
                    if tag_len > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 512 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::PatientToTriage => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::AgentToTriage => {
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

fn validate_triage_fields(
    record: &TriageRecord,
    require_location: bool,
) -> ExternResult<ValidateCallbackResult> {
    if record.patient_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Patient ID cannot be empty".into(),
        ));
    }
    if record.patient_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Patient ID must be 256 characters or fewer".into(),
        ));
    }
    if require_location && record.location.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Location cannot be empty".into(),
        ));
    }
    if record.location.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Location must be 512 characters or fewer".into(),
        ));
    }
    if record.injuries.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Injuries description must be 4096 characters or fewer".into(),
        ));
    }
    if record.notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Notes must be 4096 characters or fewer".into(),
        ));
    }
    // Transport priority consistency: Dead patients should not have Urgent transport
    if record.category == TriageCategory::Dead
        && record.transport_priority == TransportPriority::Urgent
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Dead patients cannot have Urgent transport priority".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_triage(
    _action: Create,
    record: TriageRecord,
) -> ExternResult<ValidateCallbackResult> {
    validate_triage_fields(&record, true)
}

fn validate_update_triage(
    _action: Update,
    record: TriageRecord,
) -> ExternResult<ValidateCallbackResult> {
    // Updates allow empty location (field may be cleared or unknown during re-triage)
    validate_triage_fields(&record, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper functions ──────────────────────────────────────────────

    fn fake_create() -> Create {
        Create {
            author: fake_agent_pub_key(),
            timestamp: fake_timestamp(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    fn fake_update() -> Update {
        Update {
            author: fake_agent_pub_key(),
            timestamp: fake_timestamp(),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0u8; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    fn fake_agent_pub_key() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn is_valid(result: ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_message(result: ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg,
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    // ── Factory functions ─────────────────────────────────────────────

    fn valid_triage_record() -> TriageRecord {
        TriageRecord {
            disaster_hash: fake_action_hash(),
            patient_id: "PAT-001".into(),
            patient_hash: None,
            category: TriageCategory::Immediate,
            injuries: "Compound fracture left tibia".into(),
            location: "Zone A, Sector 3".into(),
            timestamp: fake_timestamp(),
            triaged_by: fake_agent_pub_key(),
            transport_priority: TransportPriority::Urgent,
            notes: "Requires immediate surgical intervention".into(),
        }
    }

    // ── Serde roundtrip: Anchor ───────────────────────────────────────

    #[test]
    fn test_anchor_serde_roundtrip() {
        let anchor = Anchor("all_triage".into());
        let serialized = serde_json::to_string(&anchor).unwrap();
        let deserialized: Anchor = serde_json::from_str(&serialized).unwrap();
        assert_eq!(anchor, deserialized);
    }

    #[test]
    fn test_anchor_empty_string() {
        let anchor = Anchor("".into());
        let serialized = serde_json::to_string(&anchor).unwrap();
        let deserialized: Anchor = serde_json::from_str(&serialized).unwrap();
        assert_eq!(anchor, deserialized);
    }

    #[test]
    fn test_anchor_unicode() {
        let anchor = Anchor("triage_\u{1F6D1}_urgente".into());
        let serialized = serde_json::to_string(&anchor).unwrap();
        let deserialized: Anchor = serde_json::from_str(&serialized).unwrap();
        assert_eq!(anchor, deserialized);
    }

    // ── Serde roundtrip: TriageCategory ───────────────────────────────

    #[test]
    fn test_triage_category_serde_all_variants() {
        let variants = vec![
            TriageCategory::Immediate,
            TriageCategory::Delayed,
            TriageCategory::Minor,
            TriageCategory::Expectant,
            TriageCategory::Dead,
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: TriageCategory = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized, "Failed roundtrip for {:?}", variant);
        }
    }

    // ── Serde roundtrip: TransportPriority ────────────────────────────

    #[test]
    fn test_transport_priority_serde_all_variants() {
        let variants = vec![
            TransportPriority::Urgent,
            TransportPriority::Priority,
            TransportPriority::Routine,
            TransportPriority::None,
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: TransportPriority = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized, "Failed roundtrip for {:?}", variant);
        }
    }

    // ── Serde roundtrip: TriageRecord ─────────────────────────────────

    #[test]
    fn test_triage_record_serde_roundtrip() {
        let record = valid_triage_record();
        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: TriageRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(record, deserialized);
    }

    #[test]
    fn test_triage_record_with_patient_hash() {
        let mut record = valid_triage_record();
        record.patient_hash = Some(fake_action_hash());
        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: TriageRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(record, deserialized);
    }

    #[test]
    fn test_triage_record_without_patient_hash() {
        let record = valid_triage_record();
        assert_eq!(record.patient_hash, None);
        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: TriageRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.patient_hash, None);
    }

    // ── validate_create_triage: valid cases ───────────────────────────

    #[test]
    fn test_validate_create_triage_valid() {
        let record = valid_triage_record();
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_all_categories() {
        let categories = vec![
            TriageCategory::Immediate,
            TriageCategory::Delayed,
            TriageCategory::Minor,
            TriageCategory::Expectant,
            TriageCategory::Dead,
        ];

        for category in categories {
            let mut record = valid_triage_record();
            // Dead + Urgent is invalid, so use None for Dead
            if category == TriageCategory::Dead {
                record.transport_priority = TransportPriority::None;
            }
            record.category = category.clone();
            let result = validate_create_triage(fake_create(), record);
            assert!(
                is_valid(result),
                "Expected valid for category {:?}",
                category
            );
        }
    }

    #[test]
    fn test_validate_create_triage_all_transport_priorities() {
        let priorities = vec![
            TransportPriority::Urgent,
            TransportPriority::Priority,
            TransportPriority::Routine,
            TransportPriority::None,
        ];

        for priority in priorities {
            let mut record = valid_triage_record();
            record.transport_priority = priority.clone();
            let result = validate_create_triage(fake_create(), record);
            assert!(
                is_valid(result),
                "Expected valid for priority {:?}",
                priority
            );
        }
    }

    #[test]
    fn test_validate_create_triage_with_patient_hash() {
        let mut record = valid_triage_record();
        record.patient_hash = Some(fake_action_hash());
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_empty_injuries() {
        let mut record = valid_triage_record();
        record.injuries = "".into();
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_empty_notes() {
        let mut record = valid_triage_record();
        record.notes = "".into();
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    // ── validate_create_triage: invalid patient_id ────────────────────

    #[test]
    fn test_validate_create_triage_empty_patient_id() {
        let mut record = valid_triage_record();
        record.patient_id = "".into();
        let result = validate_create_triage(fake_create(), record);
        let msg = invalid_message(result);
        assert!(
            msg.contains("Patient ID"),
            "Error message should mention Patient ID, got: {}",
            msg
        );
    }

    // ── validate_create_triage: invalid location ──────────────────────

    #[test]
    fn test_validate_create_triage_empty_location() {
        let mut record = valid_triage_record();
        record.location = "".into();
        let result = validate_create_triage(fake_create(), record);
        let msg = invalid_message(result);
        assert!(
            msg.contains("Location"),
            "Error message should mention Location, got: {}",
            msg
        );
    }

    // ── validate_create_triage: both empty triggers patient_id first ──

    #[test]
    fn test_validate_create_triage_both_empty_patient_id_checked_first() {
        let mut record = valid_triage_record();
        record.patient_id = "".into();
        record.location = "".into();
        let result = validate_create_triage(fake_create(), record);
        let msg = invalid_message(result);
        assert!(
            msg.contains("Patient ID"),
            "patient_id should be validated before location, got: {}",
            msg
        );
    }

    // ── validate_update_triage: valid cases ───────────────────────────

    #[test]
    fn test_validate_update_triage_valid() {
        let record = valid_triage_record();
        let result = validate_update_triage(fake_update(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_update_triage_all_categories() {
        let categories = vec![
            TriageCategory::Immediate,
            TriageCategory::Delayed,
            TriageCategory::Minor,
            TriageCategory::Expectant,
            TriageCategory::Dead,
        ];

        for category in categories {
            let mut record = valid_triage_record();
            // Dead + Urgent is invalid, so use None for Dead
            if category == TriageCategory::Dead {
                record.transport_priority = TransportPriority::None;
            }
            record.category = category.clone();
            let result = validate_update_triage(fake_update(), record);
            assert!(
                is_valid(result),
                "Expected valid update for category {:?}",
                category
            );
        }
    }

    #[test]
    fn test_validate_update_triage_allows_empty_location() {
        let mut record = valid_triage_record();
        record.location = "".into();
        let result = validate_update_triage(fake_update(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_update_triage_allows_empty_injuries() {
        let mut record = valid_triage_record();
        record.injuries = "".into();
        let result = validate_update_triage(fake_update(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_update_triage_allows_empty_notes() {
        let mut record = valid_triage_record();
        record.notes = "".into();
        let result = validate_update_triage(fake_update(), record);
        assert!(is_valid(result));
    }

    // ── validate_update_triage: invalid patient_id ────────────────────

    #[test]
    fn test_validate_update_triage_empty_patient_id() {
        let mut record = valid_triage_record();
        record.patient_id = "".into();
        let result = validate_update_triage(fake_update(), record);
        let msg = invalid_message(result);
        assert!(
            msg.contains("Patient ID"),
            "Error message should mention Patient ID, got: {}",
            msg
        );
    }

    // ── Edge cases: Unicode content ───────────────────────────────────

    #[test]
    fn test_validate_create_triage_unicode_patient_id() {
        let mut record = valid_triage_record();
        record.patient_id = "\u{60A3}\u{8005}-\u{4E00}\u{53F7}".into(); // Japanese: patient-one
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_unicode_location() {
        let mut record = valid_triage_record();
        record.location = "\u{6771}\u{4EAC}\u{90FD}\u{65B0}\u{5BBF}\u{533A}".into(); // Tokyo Shinjuku
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_emoji_in_notes() {
        let mut record = valid_triage_record();
        record.notes = "Critical condition \u{1F6D1}\u{1F6A8}".into();
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_rtl_text() {
        let mut record = valid_triage_record();
        record.patient_id = "\u{0645}\u{0631}\u{064A}\u{0636}-001".into(); // Arabic: patient-001
        record.location = "\u{0628}\u{063A}\u{062F}\u{0627}\u{062F}".into(); // Baghdad
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    // ── Edge cases: Long strings ──────────────────────────────────────

    #[test]
    fn test_validate_create_triage_very_long_patient_id() {
        let mut record = valid_triage_record();
        record.patient_id = "P".repeat(10000);
        let result = validate_create_triage(fake_create(), record);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_triage_very_long_location() {
        let mut record = valid_triage_record();
        record.location = "L".repeat(10000);
        let result = validate_create_triage(fake_create(), record);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_triage_very_long_injuries() {
        let mut record = valid_triage_record();
        record.injuries = "Detailed injury description. ".repeat(500);
        let result = validate_create_triage(fake_create(), record);
        assert!(is_invalid(result));
    }

    #[test]
    fn test_validate_create_triage_very_long_notes() {
        let mut record = valid_triage_record();
        record.notes = "Note. ".repeat(5000);
        let result = validate_create_triage(fake_create(), record);
        assert!(is_invalid(result));
    }

    // ── Edge cases: Whitespace-only strings ───────────────────────────

    #[test]
    fn test_validate_create_triage_whitespace_patient_id_rejected() {
        let mut record = valid_triage_record();
        record.patient_id = "   ".into();
        // Whitespace-only is rejected by trim().is_empty()
        let result = validate_create_triage(fake_create(), record);
        assert!(!is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_whitespace_location_rejected() {
        let mut record = valid_triage_record();
        record.location = "   ".into();
        // Whitespace-only is rejected by trim().is_empty()
        let result = validate_create_triage(fake_create(), record);
        assert!(!is_valid(result));
    }

    // ── Edge cases: Single-character strings ──────────────────────────

    #[test]
    fn test_validate_create_triage_single_char_patient_id() {
        let mut record = valid_triage_record();
        record.patient_id = "X".into();
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_single_char_location() {
        let mut record = valid_triage_record();
        record.location = "A".into();
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    // ── Edge cases: Special characters ────────────────────────────────

    #[test]
    fn test_validate_create_triage_special_chars_patient_id() {
        let mut record = valid_triage_record();
        record.patient_id = "PAT-001/B\t\n\"<>&".into();
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_newlines_in_location() {
        let mut record = valid_triage_record();
        record.location = "Floor 3\nRoom 301\nBed 2".into();
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    // ── Edge cases: Null byte in strings ──────────────────────────────

    #[test]
    fn test_validate_create_triage_null_byte_patient_id() {
        let mut record = valid_triage_record();
        record.patient_id = "PAT\0001".into();
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    // ── Serde: TriageRecord with all fields populated ─────────────────

    #[test]
    fn test_triage_record_full_serde_roundtrip() {
        let record = TriageRecord {
            disaster_hash: fake_action_hash(),
            patient_id: "FULL-PAT-999".into(),
            patient_hash: Some(ActionHash::from_raw_36(vec![1u8; 36])),
            category: TriageCategory::Expectant,
            injuries: "Multiple trauma".into(),
            location: "Building C, Floor 2".into(),
            timestamp: Timestamp::from_micros(1_700_000_000_000_000),
            triaged_by: AgentPubKey::from_raw_36(vec![2u8; 36]),
            transport_priority: TransportPriority::Priority,
            notes: "Re-triage in 15 min".into(),
        };
        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: TriageRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(record, deserialized);
    }

    // ── Serde: TriageRecord with empty optional fields ────────────────

    #[test]
    fn test_triage_record_minimal_serde_roundtrip() {
        let record = TriageRecord {
            disaster_hash: fake_action_hash(),
            patient_id: "M".into(),
            patient_hash: None,
            category: TriageCategory::Minor,
            injuries: "".into(),
            location: "X".into(),
            timestamp: Timestamp::from_micros(0),
            triaged_by: fake_agent_pub_key(),
            transport_priority: TransportPriority::None,
            notes: "".into(),
        };
        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: TriageRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(record, deserialized);
    }

    // ── Clone/PartialEq for TriageRecord ──────────────────────────────

    #[test]
    fn test_triage_record_clone_eq() {
        let record = valid_triage_record();
        let cloned = record.clone();
        assert_eq!(record, cloned);
    }

    #[test]
    fn test_triage_record_ne_different_patient_id() {
        let a = valid_triage_record();
        let mut b = valid_triage_record();
        b.patient_id = "PAT-999".into();
        assert_ne!(a, b);
    }

    #[test]
    fn test_triage_record_ne_different_category() {
        let a = valid_triage_record();
        let mut b = valid_triage_record();
        b.category = TriageCategory::Dead;
        assert_ne!(a, b);
    }

    // ── Clone/PartialEq for enums ─────────────────────────────────────

    #[test]
    fn test_triage_category_clone_eq() {
        let cat = TriageCategory::Delayed;
        assert_eq!(cat, cat.clone());
    }

    #[test]
    fn test_transport_priority_clone_eq() {
        let prio = TransportPriority::Routine;
        assert_eq!(prio, prio.clone());
    }

    // ── Asymmetry between create and update validation ────────────────

    #[test]
    fn test_create_rejects_empty_location_but_update_allows_it() {
        let mut record = valid_triage_record();
        record.location = "".into();

        let create_result = validate_create_triage(fake_create(), record.clone());
        assert!(
            is_invalid(create_result),
            "create should reject empty location"
        );

        let update_result = validate_update_triage(fake_update(), record);
        assert!(
            is_valid(update_result),
            "update should allow empty location"
        );
    }

    #[test]
    fn test_both_create_and_update_reject_empty_patient_id() {
        let mut record = valid_triage_record();
        record.patient_id = "".into();

        let create_result = validate_create_triage(fake_create(), record.clone());
        assert!(
            is_invalid(create_result),
            "create should reject empty patient_id"
        );

        let update_result = validate_update_triage(fake_update(), record);
        assert!(
            is_invalid(update_result),
            "update should reject empty patient_id"
        );
    }

    // ── Serde: enum Debug output ──────────────────────────────────────

    #[test]
    fn test_triage_category_debug() {
        assert_eq!(format!("{:?}", TriageCategory::Immediate), "Immediate");
        assert_eq!(format!("{:?}", TriageCategory::Delayed), "Delayed");
        assert_eq!(format!("{:?}", TriageCategory::Minor), "Minor");
        assert_eq!(format!("{:?}", TriageCategory::Expectant), "Expectant");
        assert_eq!(format!("{:?}", TriageCategory::Dead), "Dead");
    }

    #[test]
    fn test_transport_priority_debug() {
        assert_eq!(format!("{:?}", TransportPriority::Urgent), "Urgent");
        assert_eq!(format!("{:?}", TransportPriority::Priority), "Priority");
        assert_eq!(format!("{:?}", TransportPriority::Routine), "Routine");
        assert_eq!(format!("{:?}", TransportPriority::None), "None");
    }

    // ── Serde: deserialize invalid JSON ───────────────────────────────

    #[test]
    fn test_triage_category_reject_invalid_variant() {
        let result = serde_json::from_str::<TriageCategory>("\"NonExistent\"");
        assert!(result.is_err());
    }

    #[test]
    fn test_transport_priority_reject_invalid_variant() {
        let result = serde_json::from_str::<TransportPriority>("\"SuperUrgent\"");
        assert!(result.is_err());
    }

    #[test]
    fn test_triage_category_reject_integer() {
        let result = serde_json::from_str::<TriageCategory>("42");
        assert!(result.is_err());
    }

    #[test]
    fn test_transport_priority_reject_null() {
        let result = serde_json::from_str::<TransportPriority>("null");
        assert!(result.is_err());
    }

    // ── Serde: case sensitivity ───────────────────────────────────────

    #[test]
    fn test_triage_category_case_sensitive() {
        let result = serde_json::from_str::<TriageCategory>("\"immediate\"");
        assert!(result.is_err(), "Enum variants should be case-sensitive");
    }

    #[test]
    fn test_transport_priority_case_sensitive() {
        let result = serde_json::from_str::<TransportPriority>("\"urgent\"");
        assert!(result.is_err(), "Enum variants should be case-sensitive");
    }

    // ── Timestamp edge cases ──────────────────────────────────────────

    #[test]
    fn test_validate_create_triage_max_timestamp() {
        let mut record = valid_triage_record();
        record.timestamp = Timestamp::from_micros(i64::MAX);
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    #[test]
    fn test_validate_create_triage_negative_timestamp() {
        let mut record = valid_triage_record();
        record.timestamp = Timestamp::from_micros(-1_000_000);
        let result = validate_create_triage(fake_create(), record);
        assert!(is_valid(result));
    }

    // ── Combination: valid record for each category x priority ────────

    #[test]
    fn test_validate_create_triage_category_priority_matrix() {
        let categories = vec![
            TriageCategory::Immediate,
            TriageCategory::Delayed,
            TriageCategory::Minor,
            TriageCategory::Expectant,
            TriageCategory::Dead,
        ];
        let priorities = vec![
            TransportPriority::Urgent,
            TransportPriority::Priority,
            TransportPriority::Routine,
            TransportPriority::None,
        ];

        for cat in &categories {
            for prio in &priorities {
                let mut record = valid_triage_record();
                record.category = cat.clone();
                record.transport_priority = prio.clone();
                let result = validate_create_triage(fake_create(), record);
                // Dead + Urgent is the only invalid combination
                if *cat == TriageCategory::Dead && *prio == TransportPriority::Urgent {
                    assert!(is_invalid(result), "Expected invalid for Dead x Urgent");
                } else {
                    assert!(
                        is_valid(result),
                        "Expected valid for {:?} x {:?}",
                        cat,
                        prio
                    );
                }
            }
        }
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_create_link_tag(link_type: &LinkTypes, tag: &LinkTag) -> ValidateCallbackResult {
        let tag_len = tag.0.len();
        match link_type {
            LinkTypes::DisasterToTriage | LinkTypes::PatientToTriage | LinkTypes::AgentToTriage => {
                if tag_len > 256 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 256 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
            LinkTypes::CategoryToTriage => {
                if tag_len > 512 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 512 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
        }
    }

    fn validate_delete_link_tag(tag: &LinkTag) -> ValidateCallbackResult {
        if tag.0.len() > 256 {
            ValidateCallbackResult::Invalid("Delete link tag too long (max 256 bytes)".into())
        } else {
            ValidateCallbackResult::Valid
        }
    }

    // -- DisasterToTriage (256-byte limit) boundary tests --

    #[test]
    fn link_tag_disaster_to_triage_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::DisasterToTriage, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_disaster_to_triage_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_create_link_tag(&LinkTypes::DisasterToTriage, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_disaster_to_triage_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_create_link_tag(&LinkTypes::DisasterToTriage, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- CategoryToTriage (512-byte limit) boundary tests --

    #[test]
    fn link_tag_category_to_triage_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::CategoryToTriage, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_category_to_triage_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 512]);
        let result = validate_create_link_tag(&LinkTypes::CategoryToTriage, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_category_to_triage_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 513]);
        let result = validate_create_link_tag(&LinkTypes::CategoryToTriage, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- PatientToTriage (256-byte limit) boundary tests --

    #[test]
    fn link_tag_patient_to_triage_at_limit_valid() {
        let tag = LinkTag::new(vec![0xCC; 256]);
        let result = validate_create_link_tag(&LinkTypes::PatientToTriage, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_patient_to_triage_over_limit_invalid() {
        let tag = LinkTag::new(vec![0xCC; 257]);
        let result = validate_create_link_tag(&LinkTypes::PatientToTriage, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- DoS prevention: massive tags rejected for all link types --

    #[test]
    fn link_tag_dos_prevention_all_types() {
        let massive_tag = LinkTag::new(vec![0xFF; 10_000]);
        let all_types = [
            LinkTypes::DisasterToTriage,
            LinkTypes::CategoryToTriage,
            LinkTypes::PatientToTriage,
            LinkTypes::AgentToTriage,
        ];
        for lt in &all_types {
            let result = validate_create_link_tag(lt, &massive_tag);
            assert!(
                matches!(result, ValidateCallbackResult::Invalid(_)),
                "Massive tag should be rejected for {:?}",
                lt
            );
        }
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // --- Triage hardening tests ---

    #[test]
    fn triage_dead_patient_urgent_transport_rejected() {
        let mut r = valid_triage_record();
        r.category = TriageCategory::Dead;
        r.transport_priority = TransportPriority::Urgent;
        let result = validate_create_triage(fake_create(), r);
        assert!(is_invalid(result));
    }

    #[test]
    fn triage_dead_patient_none_transport_accepted() {
        let mut r = valid_triage_record();
        r.category = TriageCategory::Dead;
        r.transport_priority = TransportPriority::None;
        let result = validate_create_triage(fake_create(), r);
        assert!(is_valid(result));
    }

    #[test]
    fn triage_immediate_patient_urgent_transport_accepted() {
        let mut r = valid_triage_record();
        r.category = TriageCategory::Immediate;
        r.transport_priority = TransportPriority::Urgent;
        let result = validate_create_triage(fake_create(), r);
        assert!(is_valid(result));
    }

    #[test]
    fn triage_patient_id_too_long_rejected() {
        let mut r = valid_triage_record();
        r.patient_id = "A".repeat(257);
        let result = validate_create_triage(fake_create(), r);
        assert!(is_invalid(result));
    }

    #[test]
    fn triage_patient_id_at_limit_accepted() {
        let mut r = valid_triage_record();
        r.patient_id = "A".repeat(256);
        let result = validate_create_triage(fake_create(), r);
        assert!(is_valid(result));
    }

    #[test]
    fn triage_location_too_long_rejected() {
        let mut r = valid_triage_record();
        r.location = "A".repeat(513);
        let result = validate_create_triage(fake_create(), r);
        assert!(is_invalid(result));
    }

    #[test]
    fn triage_injuries_too_long_rejected() {
        let mut r = valid_triage_record();
        r.injuries = "A".repeat(4097);
        let result = validate_create_triage(fake_create(), r);
        assert!(is_invalid(result));
    }

    #[test]
    fn triage_notes_too_long_rejected() {
        let mut r = valid_triage_record();
        r.notes = "A".repeat(4097);
        let result = validate_create_triage(fake_create(), r);
        assert!(is_invalid(result));
    }

    #[test]
    fn triage_notes_at_limit_accepted() {
        let mut r = valid_triage_record();
        r.notes = "A".repeat(4096);
        let result = validate_create_triage(fake_create(), r);
        assert!(is_valid(result));
    }

    #[test]
    fn triage_empty_injuries_accepted() {
        let mut r = valid_triage_record();
        r.injuries = String::new();
        let result = validate_create_triage(fake_create(), r);
        assert!(is_valid(result));
    }

    #[test]
    fn triage_expectant_with_all_transport_priorities() {
        for prio in [
            TransportPriority::Urgent,
            TransportPriority::Priority,
            TransportPriority::Routine,
            TransportPriority::None,
        ] {
            let mut r = valid_triage_record();
            r.category = TriageCategory::Expectant;
            r.transport_priority = prio;
            let result = validate_create_triage(fake_create(), r);
            assert!(is_valid(result));
        }
    }
}
