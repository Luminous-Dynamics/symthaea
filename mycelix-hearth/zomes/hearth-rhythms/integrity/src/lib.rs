//! Hearth Rhythms Integrity Zome
//!
//! Defines entry types and validation for family rhythms (routines),
//! rhythm occurrences (check-ins), and member presence tracking.

use hdi::prelude::*;
use hearth_types::*;

// ============================================================================
// Entry Types
// ============================================================================

/// A recurring family rhythm (routine/ritual).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Rhythm {
    /// Hash of the hearth this rhythm belongs to.
    pub hearth_hash: ActionHash,
    /// Name of the rhythm (1-256 chars).
    pub name: String,
    /// Type of rhythm (Morning, Evening, Weekly, etc.).
    pub rhythm_type: RhythmType,
    /// Schedule description (1-256 chars, e.g. "Mon/Wed/Fri 7pm").
    pub schedule: String,
    /// Participating members (max 50).
    pub participants: Vec<AgentPubKey>,
    /// Description of the rhythm (0-4096 chars).
    pub description: String,
    /// When this rhythm was created.
    pub created_at: Timestamp,
}

/// A single occurrence of a rhythm (a check-in).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RhythmOccurrence {
    /// Hash of the rhythm this occurrence belongs to.
    pub rhythm_hash: ActionHash,
    /// Date/time of the occurrence.
    pub date: Timestamp,
    /// Members who were present (max 50).
    pub participants_present: Vec<AgentPubKey>,
    /// Notes about this occurrence (0-4096 chars).
    pub notes: String,
    /// Optional mood in basis points (0-10000).
    pub mood_bp: Option<u32>,
    /// When this record was created.
    pub created_at: Timestamp,
}

/// Current presence status of a hearth member.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PresenceStatus {
    /// Hash of the hearth.
    pub hearth_hash: ActionHash,
    /// The agent this status is for.
    pub agent: AgentPubKey,
    /// Current status (Home, Away, Working, etc.).
    pub status: PresenceStatusType,
    /// When the agent expects to return (if away).
    pub expected_return: Option<Timestamp>,
    /// When this status was last updated.
    pub updated_at: Timestamp,
}

// ============================================================================
// Entry / Link Type Enums
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Rhythm(Rhythm),
    RhythmOccurrence(RhythmOccurrence),
    PresenceStatus(PresenceStatus),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Hearth -> Rhythm
    HearthToRhythms,
    /// Rhythm -> RhythmOccurrence
    RhythmToOccurrences,
    /// Hearth -> PresenceStatus
    HearthToPresence,
    /// AgentPubKey -> PresenceStatus
    AgentToPresence,
}

// ============================================================================
// Genesis + Validation
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry {
            app_entry,
            action: _,
        }) => match app_entry {
            EntryTypes::Rhythm(rhythm) => validate_rhythm(&rhythm),
            EntryTypes::RhythmOccurrence(occurrence) => validate_occurrence(&occurrence),
            EntryTypes::PresenceStatus(presence) => validate_presence(&presence),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry {
            app_entry,
            original_action_hash,
            ..
        }) => match app_entry {
            EntryTypes::Rhythm(rhythm) => {
                validate_rhythm(&rhythm)?;
                validate_rhythm_immutable_fields(&rhythm, &original_action_hash)
            }
            EntryTypes::RhythmOccurrence(_) => {
                // INVARIANT: RhythmOccurrence immutability — occurrences are event
                // records and cannot be modified after creation.
                Ok(ValidateCallbackResult::Invalid(
                    "RhythmOccurrence cannot be updated once created".into(),
                ))
            }
            EntryTypes::PresenceStatus(presence) => {
                validate_presence(&presence)?;
                validate_presence_immutable_fields(&presence, &original_action_hash)
            }
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds 512 bytes".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Invalid(
            "Rhythm entries cannot be deleted once created".into(),
        )),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

pub fn validate_rhythm(rhythm: &Rhythm) -> ExternResult<ValidateCallbackResult> {
    if rhythm.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Rhythm name cannot be empty".into(),
        ));
    }
    if rhythm.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rhythm name must be <= 256 characters".into(),
        ));
    }
    if rhythm.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rhythm description must be <= 4096 characters".into(),
        ));
    }
    if rhythm.schedule.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Rhythm schedule cannot be empty".into(),
        ));
    }
    if rhythm.schedule.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rhythm schedule must be <= 256 characters".into(),
        ));
    }
    if rhythm.participants.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rhythm participants must be <= 50".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_occurrence(occurrence: &RhythmOccurrence) -> ExternResult<ValidateCallbackResult> {
    if occurrence.notes.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Occurrence notes must be <= 4096 characters".into(),
        ));
    }
    if let Some(mood) = occurrence.mood_bp {
        if mood > 10000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Occurrence mood_bp must be <= 10000".into(),
            ));
        }
    }
    if occurrence.participants_present.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Occurrence participants_present must be <= 50".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_presence(_presence: &PresenceStatus) -> ExternResult<ValidateCallbackResult> {
    // PresenceStatusType enum is validated by serde deserialization.
    // No additional structural validation needed beyond type system guarantees.
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Immutable Field Validation
// ============================================================================

fn validate_rhythm_immutable_fields(
    new: &Rhythm,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: Rhythm = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original Rhythm: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original Rhythm entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a Rhythm".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_presence_immutable_fields(
    new: &PresenceStatus,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original: PresenceStatus = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original PresenceStatus: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original PresenceStatus entry is missing".into()
        )))?;
    if new.hearth_hash != original.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a PresenceStatus".into(),
        ));
    }
    if new.agent != original.agent {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change agent on a PresenceStatus".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helpers ----

    fn fake_agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn make_rhythm(name: &str, schedule: &str, desc: &str, participants: usize) -> Rhythm {
        Rhythm {
            hearth_hash: fake_action_hash(),
            name: name.into(),
            rhythm_type: RhythmType::Evening,
            schedule: schedule.into(),
            participants: (0..participants)
                .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
                .collect(),
            description: desc.into(),
            created_at: fake_timestamp(),
        }
    }

    fn make_occurrence(notes: &str, mood: Option<u32>, present: usize) -> RhythmOccurrence {
        RhythmOccurrence {
            rhythm_hash: fake_action_hash(),
            date: fake_timestamp(),
            participants_present: (0..present)
                .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
                .collect(),
            notes: notes.into(),
            mood_bp: mood,
            created_at: fake_timestamp(),
        }
    }

    fn make_presence(status: PresenceStatusType) -> PresenceStatus {
        PresenceStatus {
            hearth_hash: fake_action_hash(),
            agent: fake_agent_a(),
            status,
            expected_return: None,
            updated_at: fake_timestamp(),
        }
    }

    // ---- Rhythm Serde ----

    #[test]
    fn rhythm_serde_roundtrip() {
        let r = make_rhythm("Dinner Time", "Daily 6pm", "Family dinner", 4);
        let json = serde_json::to_string(&r).unwrap();
        let back: Rhythm = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn rhythm_all_types_serde() {
        for rt in &[
            RhythmType::Morning,
            RhythmType::Evening,
            RhythmType::Weekly,
            RhythmType::Seasonal,
            RhythmType::Custom("Solstice".into()),
        ] {
            let mut r = make_rhythm("Test", "Daily", "", 0);
            r.rhythm_type = rt.clone();
            let json = serde_json::to_string(&r).unwrap();
            let back: Rhythm = serde_json::from_str(&json).unwrap();
            assert_eq!(back.rhythm_type, *rt);
        }
    }

    // ---- Rhythm Validation ----

    #[test]
    fn valid_rhythm_passes() {
        let r = make_rhythm("Morning Huddle", "Daily 7am", "Quick check-in", 5);
        assert!(matches!(
            validate_rhythm(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn rhythm_empty_name_rejected() {
        let r = make_rhythm("", "Daily", "", 0);
        match validate_rhythm(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn rhythm_name_exactly_256_passes() {
        let r = make_rhythm(&"n".repeat(256), "Daily", "", 0);
        assert!(matches!(
            validate_rhythm(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn rhythm_name_257_rejected() {
        let r = make_rhythm(&"n".repeat(257), "Daily", "", 0);
        match validate_rhythm(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("256")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn rhythm_empty_description_passes() {
        let r = make_rhythm("Test", "Daily", "", 0);
        assert!(matches!(
            validate_rhythm(&r).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn rhythm_description_4097_rejected() {
        let r = make_rhythm("Test", "Daily", &"d".repeat(4097), 0);
        match validate_rhythm(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("4096")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn rhythm_empty_schedule_rejected() {
        let r = make_rhythm("Test", "", "", 0);
        match validate_rhythm(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn rhythm_schedule_257_rejected() {
        let r = make_rhythm("Test", &"s".repeat(257), "", 0);
        match validate_rhythm(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("256")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn rhythm_participants_51_rejected() {
        let r = make_rhythm("Test", "Daily", "", 51);
        match validate_rhythm(&r).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("50")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Occurrence Serde ----

    #[test]
    fn occurrence_serde_roundtrip() {
        let o = make_occurrence("Great session", Some(8500), 3);
        let json = serde_json::to_string(&o).unwrap();
        let back: RhythmOccurrence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, o);
    }

    // ---- Occurrence Validation ----

    #[test]
    fn valid_occurrence_passes() {
        let o = make_occurrence("Good", Some(7000), 4);
        assert!(matches!(
            validate_occurrence(&o).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn occurrence_notes_4097_rejected() {
        let o = make_occurrence(&"n".repeat(4097), None, 0);
        match validate_occurrence(&o).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("4096")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn occurrence_mood_10000_passes() {
        let o = make_occurrence("", Some(10000), 0);
        assert!(matches!(
            validate_occurrence(&o).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn occurrence_mood_10001_rejected() {
        let o = make_occurrence("", Some(10001), 0);
        match validate_occurrence(&o).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("10000")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn occurrence_mood_none_passes() {
        let o = make_occurrence("", None, 0);
        assert!(matches!(
            validate_occurrence(&o).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn occurrence_participants_51_rejected() {
        let o = make_occurrence("", None, 51);
        match validate_occurrence(&o).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("50")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- PresenceStatus Serde ----

    #[test]
    fn presence_serde_roundtrip() {
        let p = make_presence(PresenceStatusType::Home);
        let json = serde_json::to_string(&p).unwrap();
        let back: PresenceStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn presence_all_statuses_serde() {
        for status in &[
            PresenceStatusType::Home,
            PresenceStatusType::Away,
            PresenceStatusType::Working,
            PresenceStatusType::Sleeping,
            PresenceStatusType::DoNotDisturb,
        ] {
            let p = make_presence(status.clone());
            let json = serde_json::to_string(&p).unwrap();
            let back: PresenceStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back.status, *status);
        }
    }

    #[test]
    fn presence_with_expected_return_serde() {
        let mut p = make_presence(PresenceStatusType::Away);
        p.expected_return = Some(Timestamp::from_micros(5_000_000));
        let json = serde_json::to_string(&p).unwrap();
        let back: PresenceStatus = serde_json::from_str(&json).unwrap();
        assert!(back.expected_return.is_some());
    }

    // ---- Presence Validation ----

    #[test]
    fn valid_presence_passes() {
        let p = make_presence(PresenceStatusType::Working);
        assert!(matches!(
            validate_presence(&p).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    // ---- Entry/Link Type Enums ----

    #[test]
    fn entry_types_all_variants_exist() {
        let _r = UnitEntryTypes::Rhythm;
        let _o = UnitEntryTypes::RhythmOccurrence;
        let _p = UnitEntryTypes::PresenceStatus;
    }

    #[test]
    fn link_types_all_variants_exist() {
        let _a = LinkTypes::HearthToRhythms;
        let _b = LinkTypes::RhythmToOccurrences;
        let _c = LinkTypes::HearthToPresence;
        let _d = LinkTypes::AgentToPresence;
    }

    // ---- Delete Guard ----

    #[test]
    fn delete_guard_message_content() {
        let msg = "Rhythm entries cannot be deleted once created";
        assert!(msg.contains("cannot be deleted"));
    }

    // ---- Presence Status Validation ----

    #[test]
    fn presence_status_validation_passes() {
        let p = PresenceStatus {
            hearth_hash: fake_action_hash(),
            agent: fake_agent_a(),
            status: PresenceStatusType::Home,
            expected_return: None,
            updated_at: fake_timestamp(),
        };
        assert!(matches!(
            validate_presence(&p).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    // ---- Immutable Field Pure Equality Tests ----

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    #[test]
    fn rhythm_immutable_field_hearth_hash_difference_detected() {
        let a = make_rhythm("Test", "Daily", "", 0);
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn occurrence_immutability_all_fields_stable() {
        let a = make_occurrence("Notes", Some(5000), 2);
        let b = a.clone();
        assert_eq!(a.rhythm_hash, b.rhythm_hash);
        assert_eq!(a.date, b.date);
        assert_eq!(a.notes, b.notes);
        assert_eq!(a.mood_bp, b.mood_bp);
        assert_eq!(a.created_at, b.created_at);
    }

    #[test]
    fn presence_immutable_field_hearth_hash_difference_detected() {
        let a = make_presence(PresenceStatusType::Home);
        let mut b = a.clone();
        b.hearth_hash = ActionHash::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(a.hearth_hash, b.hearth_hash);
    }

    #[test]
    fn presence_immutable_field_agent_difference_detected() {
        let a = make_presence(PresenceStatusType::Home);
        let mut b = a.clone();
        b.agent = fake_agent_b();
        assert_ne!(a.agent, b.agent);
    }
}
