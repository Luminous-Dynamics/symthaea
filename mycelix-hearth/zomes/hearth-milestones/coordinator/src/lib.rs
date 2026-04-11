// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hearth Milestones Coordinator Zome
//!
//! Business logic for recording life milestones and managing liminal transitions.
//! Milestones are immutable records; transitions progress forward through phases.

use hdk::prelude::*;
use hearth_coordinator_common::{records_from_links, require_guardian, require_membership};
use hearth_milestones_integrity::*;
use hearth_types::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, GovernanceEligibility,
};

// ============================================================================
// Consciousness Gating
// ============================================================================


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
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "record_milestone")?;
    require_membership(&input.hearth_hash)?;
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

    let milestone_type_clone = milestone.milestone_type.clone();
    let action_hash = create_entry(&EntryTypes::Milestone(milestone))?;

    // Link hearth -> milestone
    create_link(
        input.hearth_hash.clone(),
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

    emit_signal(&HearthSignal::MilestoneRecorded {
        hearth_hash: input.hearth_hash,
        milestone_hash: action_hash.clone(),
        milestone_type: milestone_type_clone,
    })?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created milestone".into()
    )))
}

/// Begin a new life transition for a hearth member.
/// Starts in the PreLiminal phase with recategorization blocked.
#[hdk_extern]
pub fn begin_transition(input: BeginTransitionInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "begin_transition")?;
    require_membership(&input.hearth_hash)?;
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
/// Only guardians (Founder, Elder, Adult) can advance transitions.
#[hdk_extern]
pub fn advance_transition(transition_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "advance_transition")?;
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

    // Guardian auth: only guardians can advance transitions
    require_guardian(&transition.hearth_hash)?;

    // Compute next phase (errors if already Integrated)
    let next_phase = next_phase_for(&transition.current_phase)?;

    transition.current_phase = next_phase.clone();

    // Only unblock recategorization when fully integrated
    if next_phase == TransitionPhase::Integrated {
        transition.recategorization_blocked = false;
    }

    let updated_hash = update_entry(
        transition_hash.clone(),
        &EntryTypes::LifeTransition(transition),
    )?;

    emit_signal(&HearthSignal::TransitionAdvanced {
        transition_hash,
        new_phase: next_phase,
    })?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated transition".into()
    )))
}

/// Complete a transition by setting completed_at to now.
/// The transition must already be in the Integrated phase (PostLiminal must be
/// completed first). Only guardians can complete transitions.
#[hdk_extern]
pub fn complete_transition(transition_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "complete_transition")?;
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

    // Guardian auth: only guardians can complete transitions
    require_guardian(&transition.hearth_hash)?;

    if !is_transition_completable(&transition.current_phase) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Transition must be in PostLiminal phase before completion".into()
        )));
    }

    if transition.completed_at.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Transition is already completed".into()
        )));
    }

    transition.current_phase = TransitionPhase::Integrated;
    transition.recategorization_blocked = false;
    transition.completed_at = Some(now);

    let updated_hash = update_entry(
        transition_hash.clone(),
        &EntryTypes::LifeTransition(transition),
    )?;

    emit_signal(&HearthSignal::TransitionAdvanced {
        transition_hash,
        new_phase: TransitionPhase::Integrated,
    })?;

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

/// Check if a phase progression is valid: only forward, one step at a time.
/// PreLiminal -> Liminal -> PostLiminal -> Integrated.
#[allow(dead_code)]
fn is_valid_phase_progression(current: &TransitionPhase, next: &TransitionPhase) -> bool {
    matches!(
        (current, next),
        (TransitionPhase::PreLiminal, TransitionPhase::Liminal)
            | (TransitionPhase::Liminal, TransitionPhase::PostLiminal)
            | (TransitionPhase::PostLiminal, TransitionPhase::Integrated)
    )
}

/// Check if a transition can be completed (must be in PostLiminal phase).
fn is_transition_completable(phase: &TransitionPhase) -> bool {
    *phase == TransitionPhase::PostLiminal
}

/// Get the next phase for a given current phase. Returns an error if already Integrated.
fn next_phase_for(current: &TransitionPhase) -> ExternResult<TransitionPhase> {
    match current {
        TransitionPhase::PreLiminal => Ok(TransitionPhase::Liminal),
        TransitionPhase::Liminal => Ok(TransitionPhase::PostLiminal),
        TransitionPhase::PostLiminal => Ok(TransitionPhase::Integrated),
        TransitionPhase::Integrated => Err(wasm_error!(WasmErrorInner::Guest(
            "Transition is already Integrated and cannot be advanced further".into()
        ))),
    }
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

    // ---- Phase progression helpers ----

    #[test]
    fn phase_progression_prelim_to_liminal_valid() {
        assert!(is_valid_phase_progression(
            &TransitionPhase::PreLiminal,
            &TransitionPhase::Liminal
        ));
    }

    #[test]
    fn phase_progression_liminal_to_postliminal_valid() {
        assert!(is_valid_phase_progression(
            &TransitionPhase::Liminal,
            &TransitionPhase::PostLiminal
        ));
    }

    #[test]
    fn phase_progression_postliminal_to_integrated_valid() {
        assert!(is_valid_phase_progression(
            &TransitionPhase::PostLiminal,
            &TransitionPhase::Integrated
        ));
    }

    #[test]
    fn phase_progression_backward_liminal_to_prelim_invalid() {
        assert!(!is_valid_phase_progression(
            &TransitionPhase::Liminal,
            &TransitionPhase::PreLiminal
        ));
    }

    #[test]
    fn phase_progression_backward_postliminal_to_liminal_invalid() {
        assert!(!is_valid_phase_progression(
            &TransitionPhase::PostLiminal,
            &TransitionPhase::Liminal
        ));
    }

    #[test]
    fn phase_progression_backward_integrated_to_postliminal_invalid() {
        assert!(!is_valid_phase_progression(
            &TransitionPhase::Integrated,
            &TransitionPhase::PostLiminal
        ));
    }

    #[test]
    fn phase_progression_skip_prelim_to_postliminal_invalid() {
        assert!(!is_valid_phase_progression(
            &TransitionPhase::PreLiminal,
            &TransitionPhase::PostLiminal
        ));
    }

    #[test]
    fn phase_progression_skip_prelim_to_integrated_invalid() {
        assert!(!is_valid_phase_progression(
            &TransitionPhase::PreLiminal,
            &TransitionPhase::Integrated
        ));
    }

    #[test]
    fn phase_progression_skip_liminal_to_integrated_invalid() {
        assert!(!is_valid_phase_progression(
            &TransitionPhase::Liminal,
            &TransitionPhase::Integrated
        ));
    }

    #[test]
    fn phase_progression_same_phase_invalid() {
        assert!(!is_valid_phase_progression(
            &TransitionPhase::Liminal,
            &TransitionPhase::Liminal
        ));
    }

    #[test]
    fn phase_progression_integrated_to_integrated_invalid() {
        assert!(!is_valid_phase_progression(
            &TransitionPhase::Integrated,
            &TransitionPhase::Integrated
        ));
    }

    // ---- Transition completability ----

    #[test]
    fn completable_only_postliminal() {
        assert!(!is_transition_completable(&TransitionPhase::PreLiminal));
        assert!(!is_transition_completable(&TransitionPhase::Liminal));
        assert!(is_transition_completable(&TransitionPhase::PostLiminal));
        assert!(!is_transition_completable(&TransitionPhase::Integrated));
    }

    // ---- Signal serde roundtrips ----

    #[test]
    fn signal_milestone_recorded_serde_roundtrip() {
        let sig = HearthSignal::MilestoneRecorded {
            hearth_hash: fake_action_hash(),
            milestone_hash: ActionHash::from_raw_36(vec![3u8; 36]),
            milestone_type: MilestoneType::Birth,
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::MilestoneRecorded { milestone_type, .. } => {
                assert_eq!(milestone_type, MilestoneType::Birth);
            }
            _ => panic!("Expected MilestoneRecorded signal"),
        }
    }

    #[test]
    fn signal_milestone_recorded_custom_type_serde() {
        let sig = HearthSignal::MilestoneRecorded {
            hearth_hash: fake_action_hash(),
            milestone_hash: ActionHash::from_raw_36(vec![4u8; 36]),
            milestone_type: MilestoneType::Custom("Adoption Day".into()),
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::MilestoneRecorded { milestone_type, .. } => {
                assert_eq!(milestone_type, MilestoneType::Custom("Adoption Day".into()));
            }
            _ => panic!("Expected MilestoneRecorded signal"),
        }
    }

    #[test]
    fn signal_transition_advanced_serde_roundtrip() {
        let sig = HearthSignal::TransitionAdvanced {
            transition_hash: fake_action_hash(),
            new_phase: TransitionPhase::Liminal,
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::TransitionAdvanced { new_phase, .. } => {
                assert_eq!(new_phase, TransitionPhase::Liminal);
            }
            _ => panic!("Expected TransitionAdvanced signal"),
        }
    }

    #[test]
    fn signal_transition_advanced_all_phases_serde() {
        for phase in &[
            TransitionPhase::PreLiminal,
            TransitionPhase::Liminal,
            TransitionPhase::PostLiminal,
            TransitionPhase::Integrated,
        ] {
            let sig = HearthSignal::TransitionAdvanced {
                transition_hash: fake_action_hash(),
                new_phase: phase.clone(),
            };
            let json = serde_json::to_string(&sig).unwrap();
            let back: HearthSignal = serde_json::from_str(&json).unwrap();
            match back {
                HearthSignal::TransitionAdvanced { new_phase, .. } => {
                    assert_eq!(new_phase, *phase);
                }
                _ => panic!("Expected TransitionAdvanced signal"),
            }
        }
    }

    // ---- next_phase_for helper ----

    #[test]
    fn next_phase_for_preliminal() {
        assert_eq!(
            next_phase_for(&TransitionPhase::PreLiminal).unwrap(),
            TransitionPhase::Liminal
        );
    }

    #[test]
    fn next_phase_for_liminal() {
        assert_eq!(
            next_phase_for(&TransitionPhase::Liminal).unwrap(),
            TransitionPhase::PostLiminal
        );
    }

    #[test]
    fn next_phase_for_postliminal() {
        assert_eq!(
            next_phase_for(&TransitionPhase::PostLiminal).unwrap(),
            TransitionPhase::Integrated
        );
    }

    #[test]
    fn next_phase_for_integrated_errors() {
        assert!(next_phase_for(&TransitionPhase::Integrated).is_err());
    }
}
