//! Hearth Milestones Coordinator Zome
//!
//! Business logic for recording life milestones and managing liminal transitions.
//! Milestones are immutable records; transitions progress forward through phases.

use hdk::prelude::*;
use hearth_milestones_integrity::*;
use hearth_types::*;

// ============================================================================
// Input Types
// ============================================================================

/// Input for recording a new life milestone.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecordMilestoneInput {
    pub hearth_hash: ActionHash,
    pub member: AgentPubKey,
    pub milestone_type: MilestoneType,
    pub date: Timestamp,
    pub description: String,
    pub witnesses: Vec<AgentPubKey>,
    pub media_hashes: Vec<ActionHash>,
}

/// Input for beginning a new life transition.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BeginTransitionInput {
    pub hearth_hash: ActionHash,
    pub member: AgentPubKey,
    pub transition_type: TransitionType,
    pub supporting_members: Vec<AgentPubKey>,
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Record a new life milestone for a hearth member.
/// Creates the Milestone entry and links it to both the hearth and the member.
#[hdk_extern]
pub fn record_milestone(input: RecordMilestoneInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let milestone = Milestone {
        hearth_hash: input.hearth_hash.clone(),
        member: input.member.clone(),
        milestone_type: input.milestone_type,
        date: input.date,
        description: input.description,
        witnesses: input.witnesses,
        media_hashes: input.media_hashes,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::Milestone(milestone))?;

    // Link hearth -> milestone
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToMilestones,
        (),
    )?;

    // Link member -> milestone
    create_link(
        input.member,
        action_hash.clone(),
        LinkTypes::AgentToMilestones,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created milestone".into()
    )))
}

/// Begin a new life transition for a hearth member.
/// Starts in the PreLiminal phase with recategorization blocked.
#[hdk_extern]
pub fn begin_transition(input: BeginTransitionInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let transition = LifeTransition {
        hearth_hash: input.hearth_hash.clone(),
        member: input.member.clone(),
        transition_type: input.transition_type,
        current_phase: TransitionPhase::PreLiminal,
        supporting_members: input.supporting_members,
        recategorization_blocked: true,
        started_at: now,
        completed_at: None,
    };

    let action_hash = create_entry(&EntryTypes::LifeTransition(transition))?;

    // Link hearth -> transition
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToTransitions,
        (),
    )?;

    // Link member -> transition
    create_link(
        input.member,
        action_hash.clone(),
        LinkTypes::AgentToTransitions,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created transition".into()
    )))
}

/// Advance a transition to the next phase.
///
/// Phase progression: PreLiminal -> Liminal -> PostLiminal -> Integrated.
/// Sets `recategorization_blocked = false` only when reaching Integrated.
/// Returns an error if the transition is already Integrated.
#[hdk_extern]
pub fn advance_transition(transition_hash: ActionHash) -> ExternResult<Record> {
    let record = get(transition_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Transition not found".into())
    ))?;

    let mut transition: LifeTransition = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid transition entry".into()
        )))?;

    let next_phase = match transition.current_phase {
        TransitionPhase::PreLiminal => TransitionPhase::Liminal,
        TransitionPhase::Liminal => TransitionPhase::PostLiminal,
        TransitionPhase::PostLiminal => TransitionPhase::Integrated,
        TransitionPhase::Integrated => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Transition is already Integrated and cannot be advanced further".into()
            )));
        }
    };

    transition.current_phase = next_phase.clone();

    // Only unblock recategorization when fully integrated
    if next_phase == TransitionPhase::Integrated {
        transition.recategorization_blocked = false;
    }

    let updated_hash = update_entry(transition_hash, &EntryTypes::LifeTransition(transition))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated transition".into()
    )))
}

/// Complete a transition by setting completed_at to now.
/// The transition must already be in the Integrated phase.
#[hdk_extern]
pub fn complete_transition(transition_hash: ActionHash) -> ExternResult<Record> {
    let now = sys_time()?;

    let record = get(transition_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Transition not found".into())
    ))?;

    let mut transition: LifeTransition = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid transition entry".into()
        )))?;

    if transition.current_phase != TransitionPhase::Integrated {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Transition must be in Integrated phase before completion".into()
        )));
    }

    if transition.completed_at.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Transition is already completed".into()
        )));
    }

    transition.completed_at = Some(now);

    let updated_hash = update_entry(transition_hash, &EntryTypes::LifeTransition(transition))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated transition".into()
    )))
}

/// Get the full family timeline of milestones for a hearth.
#[hdk_extern]
pub fn get_family_timeline(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToMilestones)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all milestones for a specific member.
#[hdk_extern]
pub fn get_member_milestones(member: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(member, LinkTypes::AgentToMilestones)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get active (non-completed) transitions for a hearth.
#[hdk_extern]
pub fn get_active_transitions(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToTransitions)?,
        GetStrategy::default(),
    )?;
    let all_records = records_from_links(links)?;

    let mut active = Vec::new();
    for record in all_records {
        if let Some(transition) = record
            .entry()
            .to_app_option::<LifeTransition>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if transition.completed_at.is_none() {
                active.push(record);
            }
        }
    }
    Ok(active)
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

    // ---- RecordMilestoneInput serde ----

    #[test]
    fn record_milestone_input_serde_roundtrip() {
        let input = RecordMilestoneInput {
            hearth_hash: fake_action_hash(),
            member: fake_agent(),
            milestone_type: MilestoneType::Birth,
            date: Timestamp::from_micros(1_000_000),
            description: "Baby born!".to_string(),
            witnesses: vec![fake_agent_b()],
            media_hashes: vec![fake_action_hash()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordMilestoneInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.description, "Baby born!");
        assert_eq!(decoded.witnesses.len(), 1);
        assert_eq!(decoded.media_hashes.len(), 1);
    }

    #[test]
    fn record_milestone_input_all_types() {
        for mt in &[
            MilestoneType::Birth,
            MilestoneType::Graduation,
            MilestoneType::Marriage,
            MilestoneType::Passing,
            MilestoneType::Custom("Custom Event".into()),
        ] {
            let input = RecordMilestoneInput {
                hearth_hash: fake_action_hash(),
                member: fake_agent(),
                milestone_type: mt.clone(),
                date: Timestamp::from_micros(0),
                description: "Test".into(),
                witnesses: vec![],
                media_hashes: vec![],
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: RecordMilestoneInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.milestone_type, *mt);
        }
    }

    // ---- BeginTransitionInput serde ----

    #[test]
    fn begin_transition_input_serde_roundtrip() {
        let input = BeginTransitionInput {
            hearth_hash: fake_action_hash(),
            member: fake_agent(),
            transition_type: TransitionType::ComingOfAge,
            supporting_members: vec![fake_agent_b()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: BeginTransitionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.transition_type, TransitionType::ComingOfAge);
        assert_eq!(decoded.supporting_members.len(), 1);
    }

    #[test]
    fn begin_transition_input_all_types() {
        for tt in &[
            TransitionType::JoiningHearth,
            TransitionType::LeavingHearth,
            TransitionType::ComingOfAge,
            TransitionType::Retirement,
            TransitionType::Bereavement,
            TransitionType::Custom("Adoption".into()),
        ] {
            let input = BeginTransitionInput {
                hearth_hash: fake_action_hash(),
                member: fake_agent(),
                transition_type: tt.clone(),
                supporting_members: vec![],
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: BeginTransitionInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.transition_type, *tt);
        }
    }

    // ---- Milestone entry serde ----

    #[test]
    fn milestone_entry_serde_roundtrip() {
        let m = Milestone {
            hearth_hash: fake_action_hash(),
            member: fake_agent(),
            milestone_type: MilestoneType::Graduation,
            date: Timestamp::from_micros(2_000_000),
            description: "Graduated!".into(),
            witnesses: vec![fake_agent_b()],
            media_hashes: vec![],
            created_at: Timestamp::from_micros(3_000_000),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: Milestone = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    // ---- LifeTransition entry serde ----

    #[test]
    fn life_transition_entry_serde_roundtrip() {
        let t = LifeTransition {
            hearth_hash: fake_action_hash(),
            member: fake_agent(),
            transition_type: TransitionType::JoiningHearth,
            current_phase: TransitionPhase::Liminal,
            supporting_members: vec![fake_agent_b()],
            recategorization_blocked: true,
            started_at: Timestamp::from_micros(1_000_000),
            completed_at: None,
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: LifeTransition = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn life_transition_completed_serde_roundtrip() {
        let t = LifeTransition {
            hearth_hash: fake_action_hash(),
            member: fake_agent(),
            transition_type: TransitionType::ComingOfAge,
            current_phase: TransitionPhase::Integrated,
            supporting_members: vec![],
            recategorization_blocked: false,
            started_at: Timestamp::from_micros(1_000_000),
            completed_at: Some(Timestamp::from_micros(5_000_000)),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: LifeTransition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.current_phase, TransitionPhase::Integrated);
        assert!(!back.recategorization_blocked);
        assert!(back.completed_at.is_some());
    }

    // ---- Phase ordering ----

    #[test]
    fn transition_phases_ordered_correctly() {
        assert!(TransitionPhase::PreLiminal < TransitionPhase::Liminal);
        assert!(TransitionPhase::Liminal < TransitionPhase::PostLiminal);
        assert!(TransitionPhase::PostLiminal < TransitionPhase::Integrated);
    }

    // ---- Empty collections ----

    #[test]
    fn record_milestone_input_empty_collections() {
        let input = RecordMilestoneInput {
            hearth_hash: fake_action_hash(),
            member: fake_agent(),
            milestone_type: MilestoneType::NewHome,
            date: Timestamp::from_micros(0),
            description: "Moved in!".into(),
            witnesses: vec![],
            media_hashes: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordMilestoneInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.witnesses.is_empty());
        assert!(decoded.media_hashes.is_empty());
    }
}
