//! Hearth Rhythms Coordinator Zome
//!
//! Business logic for managing family rhythms (routines/rituals),
//! logging rhythm occurrences, and tracking member presence.

use hdk::prelude::*;
use hearth_rhythms_integrity::*;
use hearth_types::*;

// ============================================================================
// Input Types
// ============================================================================

/// Input for creating a new family rhythm.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateRhythmInput {
    pub hearth_hash: ActionHash,
    pub name: String,
    pub rhythm_type: RhythmType,
    pub schedule: String,
    pub participants: Vec<AgentPubKey>,
    pub description: String,
}

/// Input for logging a rhythm occurrence (check-in).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogOccurrenceInput {
    pub rhythm_hash: ActionHash,
    pub participants_present: Vec<AgentPubKey>,
    pub notes: String,
    pub mood_bp: Option<u32>,
}

/// Input for setting presence status.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SetPresenceInput {
    pub hearth_hash: ActionHash,
    pub status: PresenceStatusType,
    pub expected_return: Option<Timestamp>,
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create a new family rhythm and link it to the hearth.
#[hdk_extern]
pub fn create_rhythm(input: CreateRhythmInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let rhythm = Rhythm {
        hearth_hash: input.hearth_hash.clone(),
        name: input.name,
        rhythm_type: input.rhythm_type,
        schedule: input.schedule,
        participants: input.participants,
        description: input.description,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::Rhythm(rhythm))?;

    // Link hearth -> rhythm
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToRhythms,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created rhythm".into()
    )))
}

/// Log a rhythm occurrence (check-in) and emit a RhythmOccurred signal.
#[hdk_extern]
pub fn log_occurrence(input: LogOccurrenceInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let occurrence = RhythmOccurrence {
        rhythm_hash: input.rhythm_hash.clone(),
        date: now,
        participants_present: input.participants_present.clone(),
        notes: input.notes,
        mood_bp: input.mood_bp,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::RhythmOccurrence(occurrence))?;

    // Link rhythm -> occurrence
    create_link(
        input.rhythm_hash.clone(),
        action_hash.clone(),
        LinkTypes::RhythmToOccurrences,
        (),
    )?;

    // Emit real-time signal
    let signal = HearthSignal::RhythmOccurred {
        rhythm_hash: input.rhythm_hash,
        participants: input.participants_present,
    };
    emit_signal(&signal)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created rhythm occurrence".into()
    )))
}

/// Set the calling agent's presence status within a hearth.
/// Creates or updates a PresenceStatus entry and emits a PresenceChanged signal.
#[hdk_extern]
pub fn set_presence(input: SetPresenceInput) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let presence = PresenceStatus {
        hearth_hash: input.hearth_hash.clone(),
        agent: caller.clone(),
        status: input.status.clone(),
        expected_return: input.expected_return,
        updated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::PresenceStatus(presence))?;

    // Link hearth -> presence
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToPresence,
        (),
    )?;

    // Link agent -> presence
    create_link(
        caller.clone(),
        action_hash.clone(),
        LinkTypes::AgentToPresence,
        (),
    )?;

    // Emit real-time signal
    let signal = HearthSignal::PresenceChanged {
        agent: caller,
        status: input.status,
    };
    emit_signal(&signal)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created presence status".into()
    )))
}

/// Get all rhythms for a given hearth.
#[hdk_extern]
pub fn get_hearth_rhythms(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToRhythms)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all occurrences for a given rhythm.
#[hdk_extern]
pub fn get_rhythm_occurrences(rhythm_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(rhythm_hash, LinkTypes::RhythmToOccurrences)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all presence statuses for a given hearth.
#[hdk_extern]
pub fn get_hearth_presence(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToPresence)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Create a rhythm digest for the given epoch window.
/// Two-hop query: HearthToRhythms → RhythmToOccurrences, filter by created_at
/// within epoch, aggregate per rhythm into RhythmSummary.
#[hdk_extern]
pub fn create_rhythm_digest(input: DigestEpochInput) -> ExternResult<Vec<RhythmSummary>> {
    if input.epoch_start >= input.epoch_end {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "epoch_start must be before epoch_end".into()
        )));
    }

    let epoch_start_micros = input.epoch_start.as_micros();
    let epoch_end_micros = input.epoch_end.as_micros();

    // Get all rhythms linked to this hearth
    let rhythm_links = get_links(
        LinkQuery::try_new(input.hearth_hash, LinkTypes::HearthToRhythms)?,
        GetStrategy::default(),
    )?;

    let mut summaries = Vec::new();

    for rhythm_link in rhythm_links {
        let rhythm_hash = ActionHash::try_from(rhythm_link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid rhythm link target".into())))?;

        // Get the rhythm record to know expected participant count
        let rhythm_record = match get(rhythm_hash.clone(), GetOptions::default())? {
            Some(r) => r,
            None => continue,
        };
        let rhythm: Rhythm = rhythm_record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid rhythm entry".into()
            )))?;

        let expected_participants = rhythm.participants.len() as u32;

        // Get all occurrences for this rhythm
        let occurrence_links = get_links(
            LinkQuery::try_new(rhythm_hash.clone(), LinkTypes::RhythmToOccurrences)?,
            GetStrategy::default(),
        )?;

        let mut occurrence_count: u32 = 0;
        let mut participation_sum: u64 = 0;

        for occ_link in occurrence_links {
            let occ_hash = ActionHash::try_from(occ_link.target).map_err(|_| {
                wasm_error!(WasmErrorInner::Guest(
                    "Invalid occurrence link target".into()
                ))
            })?;

            let occ_record = match get(occ_hash, GetOptions::default())? {
                Some(r) => r,
                None => continue,
            };
            let occurrence: RhythmOccurrence = occ_record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid rhythm occurrence entry".into()
                )))?;

            let created_micros = occurrence.created_at.as_micros();
            if created_micros >= epoch_start_micros && created_micros < epoch_end_micros {
                occurrence_count += 1;
                let participation_bp = if expected_participants > 0 {
                    ((occurrence.participants_present.len() as u64) * 10000
                        / expected_participants as u64) as u32
                } else {
                    10000
                };
                participation_sum += participation_bp as u64;
            }
        }

        if occurrence_count > 0 {
            let avg_participation_bp = (participation_sum / occurrence_count as u64) as u32;
            summaries.push(RhythmSummary {
                rhythm_hash,
                occurrences: occurrence_count,
                avg_participation_bp: avg_participation_bp.min(10000),
            });
        }
    }

    Ok(summaries)
}

// ============================================================================
// Helpers
// ============================================================================

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    // ---- CreateRhythmInput serde ----

    #[test]
    fn create_rhythm_input_serde_roundtrip() {
        let input = CreateRhythmInput {
            hearth_hash: fake_action_hash(),
            name: "Family Dinner".to_string(),
            rhythm_type: RhythmType::Evening,
            schedule: "Daily 6pm".to_string(),
            participants: vec![fake_agent(), fake_agent_b()],
            description: "Eat together every evening".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRhythmInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Family Dinner");
        assert_eq!(decoded.participants.len(), 2);
    }

    #[test]
    fn create_rhythm_input_all_types() {
        for rt in &[
            RhythmType::Morning,
            RhythmType::Evening,
            RhythmType::Weekly,
            RhythmType::Seasonal,
            RhythmType::Custom("Harvest".into()),
        ] {
            let input = CreateRhythmInput {
                hearth_hash: fake_action_hash(),
                name: "Test".into(),
                rhythm_type: rt.clone(),
                schedule: "Daily".into(),
                participants: vec![],
                description: "".into(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: CreateRhythmInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.rhythm_type, *rt);
        }
    }

    // ---- LogOccurrenceInput serde ----

    #[test]
    fn log_occurrence_input_serde_roundtrip() {
        let input = LogOccurrenceInput {
            rhythm_hash: fake_action_hash(),
            participants_present: vec![fake_agent()],
            notes: "Great dinner!".to_string(),
            mood_bp: Some(8500),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LogOccurrenceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.notes, "Great dinner!");
        assert_eq!(decoded.mood_bp, Some(8500));
    }

    #[test]
    fn log_occurrence_input_no_mood() {
        let input = LogOccurrenceInput {
            rhythm_hash: fake_action_hash(),
            participants_present: vec![],
            notes: "".to_string(),
            mood_bp: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LogOccurrenceInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.mood_bp.is_none());
        assert!(decoded.participants_present.is_empty());
    }

    // ---- SetPresenceInput serde ----

    #[test]
    fn set_presence_input_serde_roundtrip() {
        let input = SetPresenceInput {
            hearth_hash: fake_action_hash(),
            status: PresenceStatusType::Away,
            expected_return: Some(Timestamp::from_micros(5_000_000)),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SetPresenceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, PresenceStatusType::Away);
        assert!(decoded.expected_return.is_some());
    }

    #[test]
    fn set_presence_input_all_statuses() {
        for status in &[
            PresenceStatusType::Home,
            PresenceStatusType::Away,
            PresenceStatusType::Working,
            PresenceStatusType::Sleeping,
            PresenceStatusType::DoNotDisturb,
        ] {
            let input = SetPresenceInput {
                hearth_hash: fake_action_hash(),
                status: status.clone(),
                expected_return: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: SetPresenceInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, *status);
        }
    }

    // ---- Entry serde roundtrips ----

    #[test]
    fn rhythm_entry_serde_roundtrip() {
        let r = Rhythm {
            hearth_hash: fake_action_hash(),
            name: "Morning Yoga".into(),
            rhythm_type: RhythmType::Morning,
            schedule: "Mon/Wed/Fri 6am".into(),
            participants: vec![fake_agent()],
            description: "Sunrise yoga session".into(),
            created_at: Timestamp::from_micros(1_000_000),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: Rhythm = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    // ---- DigestEpochInput serde ----

    #[test]
    fn digest_epoch_input_serde_roundtrip() {
        let input = DigestEpochInput {
            hearth_hash: fake_action_hash(),
            epoch_start: Timestamp::from_micros(0),
            epoch_end: Timestamp::from_micros(604_800_000_000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: DigestEpochInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hearth_hash, input.hearth_hash);
        assert_eq!(back.epoch_start, input.epoch_start);
        assert_eq!(back.epoch_end, input.epoch_end);
    }

    // ---- Entry serde roundtrips ----

    #[test]
    fn rhythm_occurrence_entry_serde_roundtrip() {
        let o = RhythmOccurrence {
            rhythm_hash: fake_action_hash(),
            date: Timestamp::from_micros(2_000_000),
            participants_present: vec![fake_agent(), fake_agent_b()],
            notes: "Everyone enjoyed it".into(),
            mood_bp: Some(9000),
            created_at: Timestamp::from_micros(2_000_000),
        };
        let json = serde_json::to_string(&o).unwrap();
        let back: RhythmOccurrence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, o);
    }
}
