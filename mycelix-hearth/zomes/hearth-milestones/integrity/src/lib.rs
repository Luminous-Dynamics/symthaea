//! Hearth Milestones Integrity Zome
//!
//! Defines entry types and validation for life milestones and liminal transitions.
//! Milestones mark significant family events; transitions model the forward-only
//! progression through liminal phases (PreLiminal -> Liminal -> PostLiminal -> Integrated).

use hdi::prelude::*;
use hearth_types::*;

// ============================================================================
// Entry Types
// ============================================================================

/// A life milestone recorded for a hearth member.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Milestone {
    /// Hash of the hearth this milestone belongs to.
    pub hearth_hash: ActionHash,
    /// The member this milestone is for.
    pub member: AgentPubKey,
    /// Type of milestone (Birth, Graduation, etc.).
    pub milestone_type: MilestoneType,
    /// Date of the milestone event.
    pub date: Timestamp,
    /// Description of the milestone (1-4096 chars).
    pub description: String,
    /// Agents who witnessed this milestone (max 50).
    pub witnesses: Vec<AgentPubKey>,
    /// Action hashes of attached media (max 20).
    pub media_hashes: Vec<ActionHash>,
    /// When this record was created.
    pub created_at: Timestamp,
}

/// A life transition modeled as a liminal process.
///
/// Transitions progress forward-only through phases:
/// PreLiminal -> Liminal -> PostLiminal -> Integrated.
///
/// While a transition is in progress (not yet Integrated),
/// `recategorization_blocked` is true to prevent premature role changes.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LifeTransition {
    /// Hash of the hearth this transition belongs to.
    pub hearth_hash: ActionHash,
    /// The member undergoing the transition.
    pub member: AgentPubKey,
    /// Type of transition (JoiningHearth, ComingOfAge, etc.).
    pub transition_type: TransitionType,
    /// Current phase in the liminal process.
    pub current_phase: TransitionPhase,
    /// Members supporting this transition (max 50).
    pub supporting_members: Vec<AgentPubKey>,
    /// Whether role recategorization is blocked during this transition.
    pub recategorization_blocked: bool,
    /// When the transition started.
    pub started_at: Timestamp,
    /// When the transition completed (None if still in progress).
    pub completed_at: Option<Timestamp>,
}

// ============================================================================
// Entry / Link Type Enums
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Milestone(Milestone),
    LifeTransition(LifeTransition),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Hearth -> Milestone
    HearthToMilestones,
    /// AgentPubKey -> Milestone
    AgentToMilestones,
    /// Hearth -> LifeTransition
    HearthToTransitions,
    /// AgentPubKey -> LifeTransition
    AgentToTransitions,
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
            EntryTypes::Milestone(milestone) => validate_milestone(&milestone),
            EntryTypes::LifeTransition(transition) => validate_transition(&transition),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Milestone(milestone) => validate_milestone(&milestone),
            EntryTypes::LifeTransition(transition) => validate_transition(&transition),
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
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

pub fn validate_milestone(milestone: &Milestone) -> ExternResult<ValidateCallbackResult> {
    if milestone.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Milestone description cannot be empty".into(),
        ));
    }
    if milestone.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Milestone description must be <= 4096 characters".into(),
        ));
    }
    if milestone.witnesses.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Milestone witnesses must be <= 50".into(),
        ));
    }
    if milestone.media_hashes.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Milestone media_hashes must be <= 20".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_transition(transition: &LifeTransition) -> ExternResult<ValidateCallbackResult> {
    if transition.supporting_members.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transition supporting_members must be <= 50".into(),
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

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn make_milestone(description: &str, witnesses: usize, media: usize) -> Milestone {
        Milestone {
            hearth_hash: fake_action_hash(),
            member: fake_agent_a(),
            milestone_type: MilestoneType::Birthday,
            date: fake_timestamp(),
            description: description.into(),
            witnesses: (0..witnesses)
                .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
                .collect(),
            media_hashes: (0..media)
                .map(|i| ActionHash::from_raw_36(vec![i as u8; 36]))
                .collect(),
            created_at: fake_timestamp(),
        }
    }

    fn make_transition(supporting: usize) -> LifeTransition {
        LifeTransition {
            hearth_hash: fake_action_hash(),
            member: fake_agent_a(),
            transition_type: TransitionType::ComingOfAge,
            current_phase: TransitionPhase::PreLiminal,
            supporting_members: (0..supporting)
                .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
                .collect(),
            recategorization_blocked: true,
            started_at: fake_timestamp(),
            completed_at: None,
        }
    }

    // ---- Milestone Serde ----

    #[test]
    fn milestone_serde_roundtrip() {
        let m = make_milestone("First birthday!", 2, 1);
        let json = serde_json::to_string(&m).unwrap();
        let back: Milestone = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn milestone_all_types_serde() {
        for mt in &[
            MilestoneType::Birth,
            MilestoneType::Birthday,
            MilestoneType::FirstStep,
            MilestoneType::SchoolStart,
            MilestoneType::Graduation,
            MilestoneType::Engagement,
            MilestoneType::Marriage,
            MilestoneType::NewHome,
            MilestoneType::Retirement,
            MilestoneType::Passing,
            MilestoneType::Custom("Baptism".into()),
        ] {
            let mut m = make_milestone("Test", 0, 0);
            m.milestone_type = mt.clone();
            let json = serde_json::to_string(&m).unwrap();
            let back: Milestone = serde_json::from_str(&json).unwrap();
            assert_eq!(back.milestone_type, *mt);
        }
    }

    // ---- Milestone Validation ----

    #[test]
    fn valid_milestone_passes() {
        let m = make_milestone("A wonderful day", 3, 2);
        assert!(matches!(
            validate_milestone(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn milestone_empty_description_rejected() {
        let m = make_milestone("", 0, 0);
        match validate_milestone(&m).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn milestone_description_exactly_4096_passes() {
        let m = make_milestone(&"d".repeat(4096), 0, 0);
        assert!(matches!(
            validate_milestone(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn milestone_description_4097_rejected() {
        let m = make_milestone(&"d".repeat(4097), 0, 0);
        match validate_milestone(&m).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("4096")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn milestone_witnesses_exactly_50_passes() {
        let m = make_milestone("OK", 50, 0);
        assert!(matches!(
            validate_milestone(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn milestone_witnesses_51_rejected() {
        let m = make_milestone("OK", 51, 0);
        match validate_milestone(&m).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("50")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn milestone_media_exactly_20_passes() {
        let m = make_milestone("OK", 0, 20);
        assert!(matches!(
            validate_milestone(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn milestone_media_21_rejected() {
        let m = make_milestone("OK", 0, 21);
        match validate_milestone(&m).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("20")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- LifeTransition Serde ----

    #[test]
    fn life_transition_serde_roundtrip() {
        let t = make_transition(3);
        let json = serde_json::to_string(&t).unwrap();
        let back: LifeTransition = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn life_transition_completed_serde_roundtrip() {
        let mut t = make_transition(1);
        t.current_phase = TransitionPhase::Integrated;
        t.recategorization_blocked = false;
        t.completed_at = Some(Timestamp::from_micros(2_000_000));
        let json = serde_json::to_string(&t).unwrap();
        let back: LifeTransition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.current_phase, TransitionPhase::Integrated);
        assert!(back.completed_at.is_some());
        assert!(!back.recategorization_blocked);
    }

    // ---- Transition Validation ----

    #[test]
    fn valid_transition_passes() {
        let t = make_transition(5);
        assert!(matches!(
            validate_transition(&t).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn transition_supporting_exactly_50_passes() {
        let t = make_transition(50);
        assert!(matches!(
            validate_transition(&t).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn transition_supporting_51_rejected() {
        let t = make_transition(51);
        match validate_transition(&t).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("50")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Entry/Link Type Enums ----

    #[test]
    fn entry_types_milestone_variant_exists() {
        let _v = UnitEntryTypes::Milestone;
    }

    #[test]
    fn entry_types_life_transition_variant_exists() {
        let _v = UnitEntryTypes::LifeTransition;
    }

    #[test]
    fn link_types_all_variants_exist() {
        let _a = LinkTypes::HearthToMilestones;
        let _b = LinkTypes::AgentToMilestones;
        let _c = LinkTypes::HearthToTransitions;
        let _d = LinkTypes::AgentToTransitions;
    }

    // ---- Edge Cases ----

    #[test]
    fn milestone_single_char_description_passes() {
        let m = make_milestone("X", 0, 0);
        assert!(matches!(
            validate_milestone(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn milestone_unicode_description_passes() {
        let m = make_milestone("Le premier pas de bebe", 0, 0);
        assert!(matches!(
            validate_milestone(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn transition_zero_supporting_members_passes() {
        let t = make_transition(0);
        assert!(matches!(
            validate_transition(&t).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn milestone_zero_witnesses_and_media_passes() {
        let m = make_milestone("Simple milestone", 0, 0);
        assert!(matches!(
            validate_milestone(&m).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }
}
