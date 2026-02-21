//! Hearth Decisions Integrity Zome
//!
//! Defines entry types and validation for family/household decisions,
//! votes, and decision outcomes.

use hdi::prelude::*;
use hearth_types::*;

// ============================================================================
// Entry Types
// ============================================================================

/// A decision proposed within a hearth for members to vote on.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Decision {
    /// The hearth this decision belongs to.
    pub hearth_hash: ActionHash,
    /// Short title describing the decision.
    pub title: String,
    /// Detailed description of the decision context.
    pub description: String,
    /// Method for reaching the decision.
    pub decision_type: DecisionType,
    /// Which roles are eligible to vote.
    pub eligible_roles: Vec<MemberRole>,
    /// The options to choose from.
    pub options: Vec<String>,
    /// When voting closes.
    pub deadline: Timestamp,
    /// Current status of the decision.
    pub status: DecisionStatus,
    /// Agent who created this decision.
    pub created_by: AgentPubKey,
    /// When the decision was created.
    pub created_at: Timestamp,
}

/// A vote cast by a member on a decision.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Vote {
    /// The decision this vote is for.
    pub decision_hash: ActionHash,
    /// The agent casting the vote.
    pub voter: AgentPubKey,
    /// Index into the decision's options Vec.
    pub choice: u32,
    /// Vote weight in basis points (0-10000).
    pub weight_bp: u32,
    /// Optional reasoning for the vote.
    pub reasoning: Option<String>,
    /// When the vote was cast.
    pub created_at: Timestamp,
}

/// The recorded outcome of a finalized decision.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DecisionOutcome {
    /// The decision this outcome is for.
    pub decision_hash: ActionHash,
    /// Index of the winning option.
    pub chosen_option: u32,
    /// Participation rate in basis points (0-10000).
    pub participation_rate_bp: u32,
    /// When the decision was resolved.
    pub resolved_at: Timestamp,
}

// ============================================================================
// Entry & Link Type Registration
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Decision(Decision),
    Vote(Vote),
    DecisionOutcome(DecisionOutcome),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Hearth -> Decision
    HearthToDecisions,
    /// Decision -> Vote
    DecisionToVotes,
    /// AgentPubKey -> Vote
    AgentToVotes,
    /// Decision -> DecisionOutcome
    DecisionToOutcome,
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
            EntryTypes::Decision(decision) => validate_decision(&decision),
            EntryTypes::Vote(vote) => validate_vote(&vote),
            EntryTypes::DecisionOutcome(outcome) => validate_outcome(&outcome),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Decision(decision) => validate_decision(&decision),
            EntryTypes::Vote(_) => {
                // INVARIANT: Vote immutability — once a vote is cast on a decision,
                // it cannot be modified or retracted. This ensures that tallied results
                // remain stable and that members cannot retroactively change outcomes.
                Ok(ValidateCallbackResult::Invalid(
                    "Votes cannot be updated once cast".into(),
                ))
            }
            EntryTypes::DecisionOutcome(outcome) => validate_outcome(&outcome),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

pub fn validate_decision(decision: &Decision) -> ExternResult<ValidateCallbackResult> {
    if decision.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision title cannot be empty".into(),
        ));
    }
    if decision.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision title must be <= 256 characters".into(),
        ));
    }
    if decision.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision description must be <= 4096 characters".into(),
        ));
    }
    if decision.options.len() < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision must have at least 2 options".into(),
        ));
    }
    if decision.options.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision must have <= 20 options".into(),
        ));
    }
    if decision.eligible_roles.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision must have at least 1 eligible role".into(),
        ));
    }
    // Validate individual option strings
    for opt in &decision.options {
        if opt.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Decision option cannot be empty".into(),
            ));
        }
        if opt.len() > 1024 {
            return Ok(ValidateCallbackResult::Invalid(
                "Decision option must be <= 1024 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_vote(vote: &Vote) -> ExternResult<ValidateCallbackResult> {
    if vote.weight_bp > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote weight_bp must be <= 10000".into(),
        ));
    }
    // We cannot fully validate choice < options.len() at integrity level
    // because we don't have access to the decision entry. We do a soft
    // upper bound check: no decision should have > 20 options.
    if vote.choice >= 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote choice must be < 20".into(),
        ));
    }
    if let Some(ref reasoning) = vote.reasoning {
        if reasoning.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Vote reasoning must be <= 4096 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_outcome(outcome: &DecisionOutcome) -> ExternResult<ValidateCallbackResult> {
    if outcome.participation_rate_bp > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "DecisionOutcome participation_rate_bp must be <= 10000".into(),
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

    // ---- Helper Constructors ----

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xAAu8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xABu8; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn make_decision(title: &str, options: Vec<&str>) -> Decision {
        Decision {
            hearth_hash: fake_action_hash(),
            title: title.into(),
            description: "A test decision".into(),
            decision_type: DecisionType::MajorityVote,
            eligible_roles: vec![MemberRole::Adult, MemberRole::Elder],
            options: options.into_iter().map(String::from).collect(),
            deadline: Timestamp::from_micros(2_000_000),
            status: DecisionStatus::Open,
            created_by: fake_agent(),
            created_at: fake_timestamp(),
        }
    }

    fn make_vote(choice: u32, weight_bp: u32) -> Vote {
        Vote {
            decision_hash: fake_action_hash(),
            voter: fake_agent(),
            choice,
            weight_bp,
            reasoning: None,
            created_at: fake_timestamp(),
        }
    }

    fn make_outcome(chosen: u32, participation_bp: u32) -> DecisionOutcome {
        DecisionOutcome {
            decision_hash: fake_action_hash(),
            chosen_option: chosen,
            participation_rate_bp: participation_bp,
            resolved_at: fake_timestamp(),
        }
    }

    // ---- Decision Validation ----

    #[test]
    fn valid_decision_passes() {
        let d = make_decision("Where to eat?", vec!["Pizza", "Tacos"]);
        assert!(matches!(
            validate_decision(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn decision_empty_title_rejected() {
        let d = make_decision("", vec!["A", "B"]);
        match validate_decision(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("cannot be empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn decision_title_exactly_256_passes() {
        let d = make_decision(&"t".repeat(256), vec!["A", "B"]);
        assert!(matches!(
            validate_decision(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn decision_title_257_rejected() {
        let d = make_decision(&"t".repeat(257), vec!["A", "B"]);
        match validate_decision(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 256")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn decision_description_exactly_4096_passes() {
        let mut d = make_decision("Title", vec!["A", "B"]);
        d.description = "d".repeat(4096);
        assert!(matches!(
            validate_decision(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn decision_description_4097_rejected() {
        let mut d = make_decision("Title", vec!["A", "B"]);
        d.description = "d".repeat(4097);
        match validate_decision(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 4096")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn decision_1_option_rejected() {
        let d = make_decision("Title", vec!["Only option"]);
        match validate_decision(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("at least 2")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn decision_0_options_rejected() {
        let d = make_decision("Title", vec![]);
        match validate_decision(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("at least 2")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn decision_20_options_passes() {
        let opts: Vec<&str> = (0..20).map(|_| "Option").collect();
        let d = make_decision("Title", opts);
        assert!(matches!(
            validate_decision(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn decision_21_options_rejected() {
        let opts: Vec<&str> = (0..21).map(|_| "Option").collect();
        let d = make_decision("Title", opts);
        match validate_decision(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 20")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn decision_empty_eligible_roles_rejected() {
        let mut d = make_decision("Title", vec!["A", "B"]);
        d.eligible_roles = vec![];
        match validate_decision(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("at least 1 eligible role"))
            }
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn decision_empty_option_string_rejected() {
        let d = make_decision("Title", vec!["Good", ""]);
        match validate_decision(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("option cannot be empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Vote Validation ----

    #[test]
    fn valid_vote_passes() {
        let v = make_vote(0, 10000);
        assert!(matches!(
            validate_vote(&v).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn vote_weight_0_passes() {
        let v = make_vote(0, 0);
        assert!(matches!(
            validate_vote(&v).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn vote_weight_10000_passes() {
        let v = make_vote(0, 10000);
        assert!(matches!(
            validate_vote(&v).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn vote_weight_10001_rejected() {
        let v = make_vote(0, 10001);
        match validate_vote(&v).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 10000")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn vote_choice_19_passes() {
        let v = make_vote(19, 5000);
        assert!(matches!(
            validate_vote(&v).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn vote_choice_20_rejected() {
        let v = make_vote(20, 5000);
        match validate_vote(&v).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("< 20")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn vote_reasoning_at_max_passes() {
        let mut v = make_vote(0, 10000);
        v.reasoning = Some("r".repeat(4096));
        assert!(matches!(
            validate_vote(&v).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn vote_reasoning_exceeds_max_rejected() {
        let mut v = make_vote(0, 10000);
        v.reasoning = Some("r".repeat(4097));
        match validate_vote(&v).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 4096")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- DecisionOutcome Validation ----

    #[test]
    fn valid_outcome_passes() {
        let o = make_outcome(0, 8500);
        assert!(matches!(
            validate_outcome(&o).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn outcome_participation_0_passes() {
        let o = make_outcome(0, 0);
        assert!(matches!(
            validate_outcome(&o).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn outcome_participation_10000_passes() {
        let o = make_outcome(0, 10000);
        assert!(matches!(
            validate_outcome(&o).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn outcome_participation_10001_rejected() {
        let o = make_outcome(0, 10001);
        match validate_outcome(&o).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 10000")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ---- Serde Roundtrips ----

    #[test]
    fn decision_serde_roundtrip() {
        let d = make_decision("Vacation spot?", vec!["Beach", "Mountains", "City"]);
        let json = serde_json::to_string(&d).unwrap();
        let back: Decision = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn vote_serde_roundtrip() {
        let v = make_vote(1, 10000);
        let json = serde_json::to_string(&v).unwrap();
        let back: Vote = serde_json::from_str(&json).unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn outcome_serde_roundtrip() {
        let o = make_outcome(2, 9500);
        let json = serde_json::to_string(&o).unwrap();
        let back: DecisionOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(back, o);
    }

    // ---- Entry / Link Type Enums ----

    #[test]
    fn entry_types_all_variants_exist() {
        let _decision = UnitEntryTypes::Decision;
        let _vote = UnitEntryTypes::Vote;
        let _outcome = UnitEntryTypes::DecisionOutcome;
    }

    #[test]
    fn link_types_all_variants_exist() {
        let _decisions = LinkTypes::HearthToDecisions;
        let _votes = LinkTypes::DecisionToVotes;
        let _agent_votes = LinkTypes::AgentToVotes;
        let _outcome = LinkTypes::DecisionToOutcome;
    }

    // ---- All DecisionType variants ----

    #[test]
    fn decision_all_types_valid() {
        let types = vec![
            DecisionType::Consensus,
            DecisionType::MajorityVote,
            DecisionType::ElderDecision,
            DecisionType::GuardianDecision,
        ];
        for dt in types {
            let mut d = make_decision("Test", vec!["A", "B"]);
            d.decision_type = dt;
            assert!(matches!(
                validate_decision(&d).unwrap(),
                ValidateCallbackResult::Valid
            ));
        }
    }

    // ---- All DecisionStatus variants ----

    #[test]
    fn decision_all_statuses_valid() {
        let statuses = vec![
            DecisionStatus::Open,
            DecisionStatus::Closed,
            DecisionStatus::Finalized,
        ];
        for status in statuses {
            let mut d = make_decision("Test", vec!["A", "B"]);
            d.status = status;
            assert!(matches!(
                validate_decision(&d).unwrap(),
                ValidateCallbackResult::Valid
            ));
        }
    }
}
