//! Hearth Rhythms Coordinator Zome
//!
//! Business logic for managing family rhythms (routines/rituals),
//! logging rhythm occurrences, and tracking member presence.

use hdk::prelude::*;
use hearth_coordinator_common::{get_latest_record, records_from_links, require_membership};
use hearth_rhythms_integrity::*;
use hearth_types::*;
use mycelix_bridge_common::{
    GovernanceEligibility, GovernanceRequirement, gate_consciousness,
    requirement_for_basic,
};

// ============================================================================
// Consciousness Gating
// ============================================================================

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("hearth_bridge", requirement, action_name)
}

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
// Pure Helpers
// ============================================================================

/// Check whether the caller is authorized to set presence for the target agent.
/// Self-set is always allowed. Guardians may set presence for other members.
fn can_set_presence_for(caller: &AgentPubKey, target: &AgentPubKey, role: &MemberRole) -> bool {
    if caller == target {
        return true;
    }
    role.is_guardian()
}

/// Compute participation rate in basis points (0-10000) for an occurrence.
/// Returns 10000 if expected_participants is 0 (vacuously full). Caps at 10000.
#[allow(dead_code)]
fn participation_bp(actual_present: usize, expected_participants: u32) -> u32 {
    if expected_participants == 0 {
        return 10000;
    }
    ((actual_present as u64 * 10000) / expected_participants as u64).min(10000) as u32
}

/// Check whether a mood basis-point value is within valid range (0-10000).
#[allow(dead_code)]
fn is_valid_mood_bp(mood_bp: Option<u32>) -> bool {
    match mood_bp {
        None => true,
        Some(v) => v <= 10000,
    }
}

/// Check whether the given timestamp falls within an epoch window [start, end).
#[allow(dead_code)]
fn is_in_epoch(ts: &Timestamp, epoch_start: &Timestamp, epoch_end: &Timestamp) -> bool {
    ts >= epoch_start && ts < epoch_end
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create a new family rhythm and link it to the hearth.
#[hdk_extern]
pub fn create_rhythm(input: CreateRhythmInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_rhythm")?;
    require_membership(&input.hearth_hash)?;
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
/// Reads the rhythm to obtain its hearth_hash, then validates membership.
#[hdk_extern]
pub fn log_occurrence(input: LogOccurrenceInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "log_occurrence")?;
    // Read the rhythm to get the hearth_hash for membership validation
    let rhythm_record = get(input.rhythm_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Rhythm not found".into())),
    )?;
    let rhythm: Rhythm = rhythm_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid rhythm entry".into()
        )))?;

    require_membership(&rhythm.hearth_hash)?;

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
/// The caller can only set their own presence unless they are a guardian
/// (Founder, Elder, or Adult).
#[hdk_extern]
pub fn set_presence(input: SetPresenceInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "set_presence")?;
    let caller = agent_info()?.agent_initial_pubkey;
    let role = require_membership(&input.hearth_hash)?;

    // The entry's agent is always the caller (self-set).
    // Guardians could set for others via a separate function in the future.
    let target = caller.clone();

    if !can_set_presence_for(&caller, &target, &role) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the agent themselves or a guardian can set presence".into()
        )));
    }

    let now = sys_time()?;

    let presence = PresenceStatus {
        hearth_hash: input.hearth_hash.clone(),
        agent: target.clone(),
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
        target.clone(),
        action_hash.clone(),
        LinkTypes::AgentToPresence,
        (),
    )?;

    // Emit real-time signal
    let signal = HearthSignal::PresenceChanged {
        agent: target,
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
/// Two-hop query: HearthToRhythms -> RhythmToOccurrences, filter by created_at
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
        let rhythm_record = match get_latest_record(rhythm_hash.clone())? {
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

            let occ_record = match get_latest_record(occ_hash)? {
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
                let p_bp = if expected_participants > 0 {
                    ((occurrence.participants_present.len() as u64) * 10000
                        / expected_participants as u64) as u32
                } else {
                    10000
                };
                participation_sum += p_bp as u64;
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

    fn fake_agent_c() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![3u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash_b() -> ActionHash {
        ActionHash::from_raw_36(vec![1u8; 36])
    }

    fn make_rhythm(name: &str, rt: RhythmType) -> Rhythm {
        Rhythm {
            hearth_hash: fake_action_hash(),
            name: name.into(),
            rhythm_type: rt,
            schedule: "Daily 6pm".into(),
            participants: vec![fake_agent(), fake_agent_b()],
            description: "A family rhythm".into(),
            created_at: Timestamp::from_micros(1_000_000),
        }
    }

    fn make_occurrence(notes: &str, mood: Option<u32>, present: usize) -> RhythmOccurrence {
        RhythmOccurrence {
            rhythm_hash: fake_action_hash(),
            date: Timestamp::from_micros(2_000_000),
            participants_present: (0..present)
                .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
                .collect(),
            notes: notes.into(),
            mood_bp: mood,
            created_at: Timestamp::from_micros(2_000_000),
        }
    }

    fn make_presence(agent: AgentPubKey, status: PresenceStatusType) -> PresenceStatus {
        PresenceStatus {
            hearth_hash: fake_action_hash(),
            agent,
            status,
            expected_return: None,
            updated_at: Timestamp::from_micros(1_000_000),
        }
    }

    // ========================================================================
    // CreateRhythmInput serde
    // ========================================================================

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

    // ========================================================================
    // LogOccurrenceInput serde
    // ========================================================================

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

    // ========================================================================
    // SetPresenceInput serde
    // ========================================================================

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

    // ========================================================================
    // Entry serde roundtrips
    // ========================================================================

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

    #[test]
    fn presence_status_entry_serde_roundtrip() {
        let p = make_presence(fake_agent(), PresenceStatusType::Home);
        let json = serde_json::to_string(&p).unwrap();
        let back: PresenceStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn presence_status_with_return_serde_roundtrip() {
        let mut p = make_presence(fake_agent(), PresenceStatusType::Away);
        p.expected_return = Some(Timestamp::from_micros(5_000_000));
        let json = serde_json::to_string(&p).unwrap();
        let back: PresenceStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back.expected_return, Some(Timestamp::from_micros(5_000_000)));
        assert_eq!(back.status, PresenceStatusType::Away);
    }

    // ========================================================================
    // DigestEpochInput serde
    // ========================================================================

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

    // ========================================================================
    // Pure helper: can_set_presence_for
    // ========================================================================

    #[test]
    fn presence_self_set_allowed_for_all_roles() {
        let agent = fake_agent();
        for role in &[
            MemberRole::Founder,
            MemberRole::Elder,
            MemberRole::Adult,
            MemberRole::Youth,
            MemberRole::Child,
            MemberRole::Guest,
            MemberRole::Ancestor,
        ] {
            assert!(
                can_set_presence_for(&agent, &agent, role),
                "Self-set should be allowed for {:?}",
                role
            );
        }
    }

    #[test]
    fn presence_other_set_allowed_for_guardian_roles() {
        let caller = fake_agent();
        let target = fake_agent_b();

        // Guardian roles: Founder, Elder, Adult
        assert!(can_set_presence_for(&caller, &target, &MemberRole::Founder));
        assert!(can_set_presence_for(&caller, &target, &MemberRole::Elder));
        assert!(can_set_presence_for(&caller, &target, &MemberRole::Adult));
    }

    #[test]
    fn presence_other_set_rejected_for_non_guardian_roles() {
        let caller = fake_agent();
        let target = fake_agent_b();

        // Non-guardian roles: Youth, Child, Guest, Ancestor
        assert!(!can_set_presence_for(&caller, &target, &MemberRole::Youth));
        assert!(!can_set_presence_for(&caller, &target, &MemberRole::Child));
        assert!(!can_set_presence_for(&caller, &target, &MemberRole::Guest));
        assert!(!can_set_presence_for(
            &caller,
            &target,
            &MemberRole::Ancestor
        ));
    }

    #[test]
    fn presence_self_set_reflexive() {
        // Same key constructed independently should still match
        let a1 = AgentPubKey::from_raw_36(vec![42u8; 36]);
        let a2 = AgentPubKey::from_raw_36(vec![42u8; 36]);
        assert!(can_set_presence_for(&a1, &a2, &MemberRole::Youth));
    }

    // ========================================================================
    // Pure helper: participation_bp
    // ========================================================================

    #[test]
    fn participation_bp_full_attendance() {
        assert_eq!(participation_bp(4, 4), 10000);
    }

    #[test]
    fn participation_bp_half_attendance() {
        assert_eq!(participation_bp(2, 4), 5000);
    }

    #[test]
    fn participation_bp_zero_expected() {
        // Vacuously full
        assert_eq!(participation_bp(0, 0), 10000);
    }

    #[test]
    fn participation_bp_over_attendance_caps() {
        // More present than expected (new members joined mid-rhythm)
        assert_eq!(participation_bp(10, 4), 10000);
    }

    #[test]
    fn participation_bp_no_one_present() {
        assert_eq!(participation_bp(0, 5), 0);
    }

    #[test]
    fn participation_bp_one_of_many() {
        assert_eq!(participation_bp(1, 10), 1000);
    }

    // ========================================================================
    // Pure helper: is_valid_mood_bp
    // ========================================================================

    #[test]
    fn valid_mood_bp_none() {
        assert!(is_valid_mood_bp(None));
    }

    #[test]
    fn valid_mood_bp_zero() {
        assert!(is_valid_mood_bp(Some(0)));
    }

    #[test]
    fn valid_mood_bp_max() {
        assert!(is_valid_mood_bp(Some(10000)));
    }

    #[test]
    fn invalid_mood_bp_over_max() {
        assert!(!is_valid_mood_bp(Some(10001)));
    }

    #[test]
    fn invalid_mood_bp_way_over() {
        assert!(!is_valid_mood_bp(Some(u32::MAX)));
    }

    // ========================================================================
    // Pure helper: is_in_epoch
    // ========================================================================

    #[test]
    fn in_epoch_inside() {
        let start = Timestamp::from_micros(100);
        let end = Timestamp::from_micros(200);
        let ts = Timestamp::from_micros(150);
        assert!(is_in_epoch(&ts, &start, &end));
    }

    #[test]
    fn in_epoch_at_start() {
        let start = Timestamp::from_micros(100);
        let end = Timestamp::from_micros(200);
        assert!(is_in_epoch(&start, &start, &end));
    }

    #[test]
    fn in_epoch_at_end_excluded() {
        let start = Timestamp::from_micros(100);
        let end = Timestamp::from_micros(200);
        assert!(!is_in_epoch(&end, &start, &end));
    }

    #[test]
    fn in_epoch_before() {
        let start = Timestamp::from_micros(100);
        let end = Timestamp::from_micros(200);
        let ts = Timestamp::from_micros(50);
        assert!(!is_in_epoch(&ts, &start, &end));
    }

    #[test]
    fn in_epoch_after() {
        let start = Timestamp::from_micros(100);
        let end = Timestamp::from_micros(200);
        let ts = Timestamp::from_micros(300);
        assert!(!is_in_epoch(&ts, &start, &end));
    }

    // ========================================================================
    // Signal serde roundtrips
    // ========================================================================

    #[test]
    fn signal_rhythm_occurred_serde_roundtrip() {
        let sig = HearthSignal::RhythmOccurred {
            rhythm_hash: fake_action_hash(),
            participants: vec![fake_agent(), fake_agent_b()],
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("RhythmOccurred"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::RhythmOccurred {
                rhythm_hash,
                participants,
            } => {
                assert_eq!(rhythm_hash, fake_action_hash());
                assert_eq!(participants.len(), 2);
            }
            _ => panic!("Expected RhythmOccurred signal"),
        }
    }

    #[test]
    fn signal_rhythm_occurred_empty_participants() {
        let sig = HearthSignal::RhythmOccurred {
            rhythm_hash: fake_action_hash(),
            participants: vec![],
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::RhythmOccurred { participants, .. } => {
                assert!(participants.is_empty());
            }
            _ => panic!("Expected RhythmOccurred signal"),
        }
    }

    #[test]
    fn signal_presence_changed_serde_roundtrip() {
        let sig = HearthSignal::PresenceChanged {
            agent: fake_agent(),
            status: PresenceStatusType::Working,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("PresenceChanged"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::PresenceChanged { agent, status } => {
                assert_eq!(agent, fake_agent());
                assert_eq!(status, PresenceStatusType::Working);
            }
            _ => panic!("Expected PresenceChanged signal"),
        }
    }

    #[test]
    fn signal_presence_changed_all_statuses() {
        for status in &[
            PresenceStatusType::Home,
            PresenceStatusType::Away,
            PresenceStatusType::Working,
            PresenceStatusType::Sleeping,
            PresenceStatusType::DoNotDisturb,
        ] {
            let sig = HearthSignal::PresenceChanged {
                agent: fake_agent(),
                status: status.clone(),
            };
            let json = serde_json::to_string(&sig).unwrap();
            let back: HearthSignal = serde_json::from_str(&json).unwrap();
            match back {
                HearthSignal::PresenceChanged {
                    status: decoded_status,
                    ..
                } => {
                    assert_eq!(decoded_status, *status);
                }
                _ => panic!("Expected PresenceChanged signal"),
            }
        }
    }

    // ========================================================================
    // RhythmType variant tests
    // ========================================================================

    #[test]
    fn rhythm_type_morning_serde() {
        let rt = RhythmType::Morning;
        let json = serde_json::to_string(&rt).unwrap();
        let back: RhythmType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RhythmType::Morning);
    }

    #[test]
    fn rhythm_type_evening_serde() {
        let rt = RhythmType::Evening;
        let json = serde_json::to_string(&rt).unwrap();
        let back: RhythmType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RhythmType::Evening);
    }

    #[test]
    fn rhythm_type_weekly_serde() {
        let rt = RhythmType::Weekly;
        let json = serde_json::to_string(&rt).unwrap();
        let back: RhythmType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RhythmType::Weekly);
    }

    #[test]
    fn rhythm_type_seasonal_serde() {
        let rt = RhythmType::Seasonal;
        let json = serde_json::to_string(&rt).unwrap();
        let back: RhythmType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RhythmType::Seasonal);
    }

    #[test]
    fn rhythm_type_custom_serde() {
        let rt = RhythmType::Custom("Solstice".to_string());
        let json = serde_json::to_string(&rt).unwrap();
        let back: RhythmType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RhythmType::Custom("Solstice".to_string()));
    }

    #[test]
    fn rhythm_type_custom_empty_string() {
        let rt = RhythmType::Custom("".to_string());
        let json = serde_json::to_string(&rt).unwrap();
        let back: RhythmType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RhythmType::Custom("".to_string()));
    }

    // ========================================================================
    // PresenceStatusType variant tests
    // ========================================================================

    #[test]
    fn presence_home_serde() {
        let s = PresenceStatusType::Home;
        let json = serde_json::to_string(&s).unwrap();
        let back: PresenceStatusType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, PresenceStatusType::Home);
    }

    #[test]
    fn presence_away_serde() {
        let s = PresenceStatusType::Away;
        let json = serde_json::to_string(&s).unwrap();
        let back: PresenceStatusType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, PresenceStatusType::Away);
    }

    #[test]
    fn presence_working_serde() {
        let s = PresenceStatusType::Working;
        let json = serde_json::to_string(&s).unwrap();
        let back: PresenceStatusType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, PresenceStatusType::Working);
    }

    #[test]
    fn presence_sleeping_serde() {
        let s = PresenceStatusType::Sleeping;
        let json = serde_json::to_string(&s).unwrap();
        let back: PresenceStatusType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, PresenceStatusType::Sleeping);
    }

    #[test]
    fn presence_dnd_serde() {
        let s = PresenceStatusType::DoNotDisturb;
        let json = serde_json::to_string(&s).unwrap();
        let back: PresenceStatusType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, PresenceStatusType::DoNotDisturb);
    }

    // ========================================================================
    // Recurrence variant tests (used in CreateRhythmInput context)
    // ========================================================================

    #[test]
    fn recurrence_daily_serde() {
        let r = Recurrence::Daily;
        let json = serde_json::to_string(&r).unwrap();
        let back: Recurrence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Recurrence::Daily);
    }

    #[test]
    fn recurrence_weekly_serde() {
        let r = Recurrence::Weekly;
        let json = serde_json::to_string(&r).unwrap();
        let back: Recurrence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Recurrence::Weekly);
    }

    #[test]
    fn recurrence_monthly_serde() {
        let r = Recurrence::Monthly;
        let json = serde_json::to_string(&r).unwrap();
        let back: Recurrence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Recurrence::Monthly);
    }

    #[test]
    fn recurrence_custom_serde() {
        let r = Recurrence::Custom("Every equinox".to_string());
        let json = serde_json::to_string(&r).unwrap();
        let back: Recurrence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Recurrence::Custom("Every equinox".to_string()));
    }

    // ========================================================================
    // Scenario: rhythm lifecycle (create -> log -> digest)
    // ========================================================================

    #[test]
    fn scenario_rhythm_entry_preserves_fields() {
        let rhythm = make_rhythm("Evening Prayer", RhythmType::Evening);
        assert_eq!(rhythm.name, "Evening Prayer");
        assert_eq!(rhythm.rhythm_type, RhythmType::Evening);
        assert_eq!(rhythm.participants.len(), 2);
        assert_eq!(rhythm.schedule, "Daily 6pm");
    }

    #[test]
    fn scenario_occurrence_captures_mood_and_participants() {
        let occ = make_occurrence("Everyone present and happy", Some(9500), 4);
        assert_eq!(occ.notes, "Everyone present and happy");
        assert_eq!(occ.mood_bp, Some(9500));
        assert_eq!(occ.participants_present.len(), 4);
    }

    #[test]
    fn scenario_occurrence_without_mood() {
        let occ = make_occurrence("Quick check-in", None, 2);
        assert!(occ.mood_bp.is_none());
        assert_eq!(occ.participants_present.len(), 2);
    }

    // ========================================================================
    // Scenario: presence lifecycle
    // ========================================================================

    #[test]
    fn scenario_presence_transitions() {
        // Simulates: Home -> Away -> Working -> Home
        let agent = fake_agent();
        let p1 = make_presence(agent.clone(), PresenceStatusType::Home);
        assert_eq!(p1.status, PresenceStatusType::Home);

        let p2 = make_presence(agent.clone(), PresenceStatusType::Away);
        assert_eq!(p2.status, PresenceStatusType::Away);

        let p3 = make_presence(agent.clone(), PresenceStatusType::Working);
        assert_eq!(p3.status, PresenceStatusType::Working);

        let p4 = make_presence(agent, PresenceStatusType::Home);
        assert_eq!(p4.status, PresenceStatusType::Home);
    }

    #[test]
    fn scenario_presence_with_expected_return() {
        let mut p = make_presence(fake_agent(), PresenceStatusType::Away);
        assert!(p.expected_return.is_none());
        p.expected_return = Some(Timestamp::from_micros(10_000_000));
        assert_eq!(
            p.expected_return,
            Some(Timestamp::from_micros(10_000_000))
        );
    }

    // ========================================================================
    // Scenario: membership + guardian role checks (pure logic)
    // ========================================================================

    #[test]
    fn scenario_guardian_roles_checked_correctly() {
        // Guardian = Founder, Elder, Adult
        assert!(MemberRole::Founder.is_guardian());
        assert!(MemberRole::Elder.is_guardian());
        assert!(MemberRole::Adult.is_guardian());
        assert!(!MemberRole::Youth.is_guardian());
        assert!(!MemberRole::Child.is_guardian());
        assert!(!MemberRole::Guest.is_guardian());
        assert!(!MemberRole::Ancestor.is_guardian());
    }

    #[test]
    fn scenario_presence_auth_guardian_can_set_for_child() {
        let guardian = fake_agent();
        let child = fake_agent_b();
        assert!(can_set_presence_for(
            &guardian,
            &child,
            &MemberRole::Founder
        ));
    }

    #[test]
    fn scenario_presence_auth_youth_cannot_set_for_sibling() {
        let youth = fake_agent();
        let sibling = fake_agent_b();
        assert!(!can_set_presence_for(&youth, &sibling, &MemberRole::Youth));
    }

    // ========================================================================
    // Edge cases: three distinct agents
    // ========================================================================

    #[test]
    fn presence_auth_three_agents_guardian() {
        let founder = fake_agent();
        let target_a = fake_agent_b();
        let target_b = fake_agent_c();

        assert!(can_set_presence_for(
            &founder,
            &target_a,
            &MemberRole::Founder
        ));
        assert!(can_set_presence_for(
            &founder,
            &target_b,
            &MemberRole::Founder
        ));
    }

    #[test]
    fn presence_auth_three_agents_guest() {
        let guest = fake_agent();
        let target_a = fake_agent_b();
        let target_b = fake_agent_c();

        assert!(!can_set_presence_for(
            &guest,
            &target_a,
            &MemberRole::Guest
        ));
        assert!(!can_set_presence_for(
            &guest,
            &target_b,
            &MemberRole::Guest
        ));
        // Self-set always works
        assert!(can_set_presence_for(&guest, &guest, &MemberRole::Guest));
    }

    // ========================================================================
    // Multiple input serde tests (additional coverage)
    // ========================================================================

    #[test]
    fn create_rhythm_input_empty_participants() {
        let input = CreateRhythmInput {
            hearth_hash: fake_action_hash(),
            name: "Solo practice".into(),
            rhythm_type: RhythmType::Morning,
            schedule: "Daily 5am".into(),
            participants: vec![],
            description: "Meditation alone".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateRhythmInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.participants.is_empty());
    }

    #[test]
    fn log_occurrence_input_max_mood() {
        let input = LogOccurrenceInput {
            rhythm_hash: fake_action_hash(),
            participants_present: vec![fake_agent()],
            notes: "Perfect session".into(),
            mood_bp: Some(10000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LogOccurrenceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.mood_bp, Some(10000));
    }

    #[test]
    fn set_presence_input_no_return_serde() {
        let input = SetPresenceInput {
            hearth_hash: fake_action_hash(),
            status: PresenceStatusType::Home,
            expected_return: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SetPresenceInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.expected_return.is_none());
    }

    // ========================================================================
    // RhythmSummary serde
    // ========================================================================

    #[test]
    fn rhythm_summary_serde_roundtrip() {
        let summary = RhythmSummary {
            rhythm_hash: fake_action_hash(),
            occurrences: 7,
            avg_participation_bp: 8500,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: RhythmSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.occurrences, 7);
        assert_eq!(back.avg_participation_bp, 8500);
    }

    #[test]
    fn rhythm_summary_zero_values() {
        let summary = RhythmSummary {
            rhythm_hash: fake_action_hash(),
            occurrences: 0,
            avg_participation_bp: 0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: RhythmSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.occurrences, 0);
        assert_eq!(back.avg_participation_bp, 0);
    }

    // ========================================================================
    // Rhythm entry: all RhythmType variants in context
    // ========================================================================

    #[test]
    fn rhythm_entry_morning_type() {
        let r = make_rhythm("Sunrise Stretch", RhythmType::Morning);
        let json = serde_json::to_string(&r).unwrap();
        let back: Rhythm = serde_json::from_str(&json).unwrap();
        assert_eq!(back.rhythm_type, RhythmType::Morning);
    }

    #[test]
    fn rhythm_entry_seasonal_type() {
        let r = make_rhythm("Harvest Feast", RhythmType::Seasonal);
        let json = serde_json::to_string(&r).unwrap();
        let back: Rhythm = serde_json::from_str(&json).unwrap();
        assert_eq!(back.rhythm_type, RhythmType::Seasonal);
    }

    #[test]
    fn rhythm_entry_custom_type() {
        let r = make_rhythm("Full Moon Ceremony", RhythmType::Custom("Lunar".into()));
        let json = serde_json::to_string(&r).unwrap();
        let back: Rhythm = serde_json::from_str(&json).unwrap();
        assert_eq!(back.rhythm_type, RhythmType::Custom("Lunar".into()));
    }

    // ========================================================================
    // Occurrence with different hearth hash
    // ========================================================================

    #[test]
    fn occurrence_with_different_rhythm_hash() {
        let mut o = make_occurrence("test", None, 1);
        o.rhythm_hash = fake_action_hash_b();
        let json = serde_json::to_string(&o).unwrap();
        let back: RhythmOccurrence = serde_json::from_str(&json).unwrap();
        assert_eq!(back.rhythm_hash, fake_action_hash_b());
    }
}
