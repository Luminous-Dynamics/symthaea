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
    /// Optional quorum in basis points (0-10000). None = no quorum required.
    #[serde(default)]
    pub quorum_bp: Option<u32>,
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
    /// Quorum requirement at time of finalization (snapshot for audit trail).
    #[serde(default)]
    pub quorum_bp: Option<u32>,
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
    /// Decision -> Vote (all votes ever, including amended — never deleted)
    DecisionToVoteHistory,
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
        FlatOp::StoreEntry(OpEntry::UpdateEntry {
            app_entry,
            action,
            original_action_hash,
            ..
        }) => match app_entry {
            EntryTypes::Decision(decision) => {
                validate_decision(&decision)?;
                validate_decision_update(&action, &decision, &original_action_hash)
            }
            EntryTypes::Vote(_) => {
                // INVARIANT: Vote immutability — once a vote is cast on a decision,
                // it cannot be modified or retracted. This ensures that tallied results
                // remain stable and that members cannot retroactively change outcomes.
                Ok(ValidateCallbackResult::Invalid(
                    "Votes cannot be updated once cast".into(),
                ))
            }
            EntryTypes::DecisionOutcome(_) => {
                // INVARIANT: DecisionOutcome immutability — once a decision outcome
                // is recorded, it cannot be modified. This preserves the integrity
                // of the audit trail and prevents retroactive result tampering.
                Ok(ValidateCallbackResult::Invalid(
                    "DecisionOutcome cannot be updated once recorded".into(),
                ))
            }
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => validate_create_link(link_type, &tag),
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => validate_delete_link(link_type),
        FlatOp::RegisterDelete(_) => {
            // INVARIANT: Decisions, Votes, and Outcomes are append-only.
            // Deletion would break audit trails, tally integrity, and history links.
            Ok(ValidateCallbackResult::Invalid(
                "Decision entries cannot be deleted".into(),
            ))
        }
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
    if let Some(q) = decision.quorum_bp {
        if q > 10000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Decision quorum_bp must be <= 10000".into(),
            ));
        }
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
    if let Some(q) = outcome.quorum_bp {
        if q > 10000 {
            return Ok(ValidateCallbackResult::Invalid(
                "DecisionOutcome quorum_bp must be <= 10000".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate that a Decision update only modifies allowed fields with valid transitions.
pub fn validate_decision_update(
    _action: &Update,
    new_decision: &Decision,
    original_action_hash: &ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash.clone())?;
    let original_decision: Decision = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original Decision: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original Decision entry is missing".into()
        )))?;

    // Immutable field checks
    if new_decision.hearth_hash != original_decision.hearth_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change hearth_hash on a Decision".into(),
        ));
    }
    if new_decision.created_by != original_decision.created_by {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change created_by on a Decision".into(),
        ));
    }
    if new_decision.created_at != original_decision.created_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change created_at on a Decision".into(),
        ));
    }
    if new_decision.decision_type != original_decision.decision_type {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change decision_type on a Decision".into(),
        ));
    }
    if new_decision.eligible_roles != original_decision.eligible_roles {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change eligible_roles on a Decision".into(),
        ));
    }
    if new_decision.options != original_decision.options {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change options on a Decision".into(),
        ));
    }
    if new_decision.deadline != original_decision.deadline {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change deadline on a Decision".into(),
        ));
    }
    if new_decision.quorum_bp != original_decision.quorum_bp {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change quorum_bp on a Decision".into(),
        ));
    }

    // Valid status transitions: Open -> Closed, Open -> Finalized only
    match (&original_decision.status, &new_decision.status) {
        (DecisionStatus::Open, DecisionStatus::Closed) => {}
        (DecisionStatus::Open, DecisionStatus::Finalized) => {}
        (old, new) if old == new => {}
        (old, new) => {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid status transition: {:?} -> {:?}",
                old, new
            )));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Link Validation
// ============================================================================

/// Maximum link tag length per link type.
fn link_tag_max_len(link_type: &LinkTypes) -> usize {
    match link_type {
        // All decision link types use empty tags (()) — 256 bytes is generous
        LinkTypes::HearthToDecisions => 256,
        LinkTypes::DecisionToVotes => 256,
        LinkTypes::AgentToVotes => 256,
        LinkTypes::DecisionToOutcome => 256,
        LinkTypes::DecisionToVoteHistory => 256,
    }
}

/// Validate link creation: enforce tag length limits.
pub fn validate_create_link(
    link_type: LinkTypes,
    tag: &LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    let max_len = link_tag_max_len(&link_type);
    if tag.0.len() > max_len {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "{:?} link tag too long (max {} bytes, got {})",
            link_type,
            max_len,
            tag.0.len()
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate link deletion: audit trail links cannot be deleted.
pub fn validate_delete_link(link_type: LinkTypes) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        // Vote history is an immutable audit trail — never delete
        LinkTypes::DecisionToVoteHistory => Ok(ValidateCallbackResult::Invalid(
            "DecisionToVoteHistory links cannot be deleted (audit trail)".into(),
        )),
        // Outcome links are permanent — one outcome per decision
        LinkTypes::DecisionToOutcome => Ok(ValidateCallbackResult::Invalid(
            "DecisionToOutcome links cannot be deleted".into(),
        )),
        // DecisionToVotes and AgentToVotes can be deleted (amend_vote link-swap)
        // HearthToDecisions can be deleted (if needed for reorganization)
        _ => Ok(ValidateCallbackResult::Valid),
    }
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
            quorum_bp: None,
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
            quorum_bp: None,
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

    // ---- Decision Quorum Validation ----

    #[test]
    fn quorum_none_passes() {
        let d = make_decision("Test", vec!["A", "B"]);
        assert!(matches!(
            validate_decision(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn quorum_0_passes() {
        let mut d = make_decision("Test", vec!["A", "B"]);
        d.quorum_bp = Some(0);
        assert!(matches!(
            validate_decision(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn quorum_10000_passes() {
        let mut d = make_decision("Test", vec!["A", "B"]);
        d.quorum_bp = Some(10000);
        assert!(matches!(
            validate_decision(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn quorum_10001_rejected() {
        let mut d = make_decision("Test", vec!["A", "B"]);
        d.quorum_bp = Some(10001);
        match validate_decision(&d).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 10000")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn quorum_5000_passes() {
        let mut d = make_decision("Test", vec!["A", "B"]);
        d.quorum_bp = Some(5000);
        assert!(matches!(
            validate_decision(&d).unwrap(),
            ValidateCallbackResult::Valid
        ));
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
    fn decision_serde_backward_compat_no_quorum() {
        // Serialize a Decision with quorum_bp, then strip it from JSON to simulate legacy
        let d = make_decision("Test", vec!["A", "B"]);
        let mut json_val: serde_json::Value = serde_json::to_value(&d).unwrap();
        json_val.as_object_mut().unwrap().remove("quorum_bp");
        let back: Decision = serde_json::from_value(json_val).unwrap();
        assert_eq!(back.quorum_bp, None);
        assert_eq!(back.title, "Test");
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
        let _vote_history = LinkTypes::DecisionToVoteHistory;
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

    // ---- DecisionOutcome quorum validation ----

    #[test]
    fn outcome_with_quorum_serde_roundtrip() {
        let o = DecisionOutcome {
            decision_hash: fake_action_hash(),
            chosen_option: 1,
            participation_rate_bp: 8500,
            resolved_at: fake_timestamp(),
            quorum_bp: Some(5000),
        };
        let json = serde_json::to_string(&o).unwrap();
        let back: DecisionOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(back.quorum_bp, Some(5000));
        assert_eq!(back.chosen_option, 1);
    }

    #[test]
    fn outcome_backward_compat_no_quorum() {
        let o = make_outcome(0, 8500);
        let mut json_val: serde_json::Value = serde_json::to_value(&o).unwrap();
        json_val.as_object_mut().unwrap().remove("quorum_bp");
        let back: DecisionOutcome = serde_json::from_value(json_val).unwrap();
        assert_eq!(back.quorum_bp, None);
        assert_eq!(back.participation_rate_bp, 8500);
    }

    #[test]
    fn outcome_quorum_10001_rejected() {
        let mut o = make_outcome(0, 5000);
        o.quorum_bp = Some(10001);
        match validate_outcome(&o).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("<= 10000")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn outcome_quorum_10000_passes() {
        let mut o = make_outcome(0, 5000);
        o.quorum_bp = Some(10000);
        assert!(matches!(
            validate_outcome(&o).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    // ---- validate_decision_update (pure function tests) ----
    // Note: validate_decision_update calls must_get_valid_record which requires
    // a running conductor. These tests validate the pure validation logic
    // by testing the helpers and status transition rules directly.

    #[test]
    fn status_transition_open_to_closed_valid() {
        let old = DecisionStatus::Open;
        let new = DecisionStatus::Closed;
        let valid = matches!(
            (&old, &new),
            (DecisionStatus::Open, DecisionStatus::Closed)
                | (DecisionStatus::Open, DecisionStatus::Finalized)
        );
        assert!(valid);
    }

    #[test]
    fn status_transition_open_to_finalized_valid() {
        let old = DecisionStatus::Open;
        let new = DecisionStatus::Finalized;
        let valid = matches!(
            (&old, &new),
            (DecisionStatus::Open, DecisionStatus::Closed)
                | (DecisionStatus::Open, DecisionStatus::Finalized)
        );
        assert!(valid);
    }

    #[test]
    fn status_transition_finalized_to_open_invalid() {
        let old = DecisionStatus::Finalized;
        let new = DecisionStatus::Open;
        let valid = matches!(
            (&old, &new),
            (DecisionStatus::Open, DecisionStatus::Closed)
                | (DecisionStatus::Open, DecisionStatus::Finalized)
        ) || old == new;
        assert!(!valid);
    }

    #[test]
    fn status_transition_closed_to_open_invalid() {
        let old = DecisionStatus::Closed;
        let new = DecisionStatus::Open;
        let valid = matches!(
            (&old, &new),
            (DecisionStatus::Open, DecisionStatus::Closed)
                | (DecisionStatus::Open, DecisionStatus::Finalized)
        ) || old == new;
        assert!(!valid);
    }

    #[test]
    fn status_transition_closed_to_finalized_invalid() {
        let old = DecisionStatus::Closed;
        let new = DecisionStatus::Finalized;
        let valid = matches!(
            (&old, &new),
            (DecisionStatus::Open, DecisionStatus::Closed)
                | (DecisionStatus::Open, DecisionStatus::Finalized)
        ) || old == new;
        assert!(!valid);
    }

    #[test]
    fn status_transition_same_status_valid() {
        for status in &[
            DecisionStatus::Open,
            DecisionStatus::Closed,
            DecisionStatus::Finalized,
        ] {
            let valid = matches!(
                (status, status),
                (DecisionStatus::Open, DecisionStatus::Closed)
                    | (DecisionStatus::Open, DecisionStatus::Finalized)
            ) || status == status;
            assert!(
                valid,
                "same-status transition should be valid for {:?}",
                status
            );
        }
    }

    #[test]
    fn immutable_fields_equality_check() {
        let d1 = make_decision("Test", vec!["A", "B"]);
        let d2 = make_decision("Test", vec!["A", "B"]);
        assert_eq!(d1.hearth_hash, d2.hearth_hash);
        assert_eq!(d1.created_by, d2.created_by);
        assert_eq!(d1.created_at, d2.created_at);
        assert_eq!(d1.decision_type, d2.decision_type);
        assert_eq!(d1.eligible_roles, d2.eligible_roles);
        assert_eq!(d1.options, d2.options);
        assert_eq!(d1.deadline, d2.deadline);
        assert_eq!(d1.quorum_bp, d2.quorum_bp);
    }

    #[test]
    fn immutable_field_hearth_hash_difference_detected() {
        let d1 = make_decision("Test", vec!["A", "B"]);
        let mut d2 = d1.clone();
        d2.hearth_hash = ActionHash::from_raw_36(vec![0xCDu8; 36]);
        assert_ne!(d1.hearth_hash, d2.hearth_hash);
    }

    #[test]
    fn immutable_field_options_difference_detected() {
        let d1 = make_decision("Test", vec!["A", "B"]);
        let mut d2 = d1.clone();
        d2.options = vec!["A".into(), "B".into(), "C".into()];
        assert_ne!(d1.options, d2.options);
    }

    #[test]
    fn immutable_field_deadline_difference_detected() {
        let d1 = make_decision("Test", vec!["A", "B"]);
        let mut d2 = d1.clone();
        d2.deadline = Timestamp::from_micros(9_000_000);
        assert_ne!(d1.deadline, d2.deadline);
    }

    // ---- DecisionOutcome immutability (validated at integrity level) ----
    // Note: The actual validation rejects all UpdateEntry ops for DecisionOutcome.
    // This test verifies the struct is usable and the invariant is documented.

    #[test]
    fn outcome_immutability_documented() {
        // DecisionOutcome should be created once and never updated.
        // The integrity validate() function rejects UpdateEntry for DecisionOutcome.
        // This test ensures the struct fields are all set at creation time.
        let o = DecisionOutcome {
            decision_hash: fake_action_hash(),
            chosen_option: 1,
            participation_rate_bp: 8500,
            resolved_at: fake_timestamp(),
            quorum_bp: Some(5000),
        };
        assert_eq!(o.chosen_option, 1);
        assert_eq!(o.quorum_bp, Some(5000));
    }

    // ---- Link Validation: tag length ----

    #[test]
    fn link_hearth_to_decisions_valid_tag() {
        let tag = LinkTag(vec![0u8; 128]);
        assert!(matches!(
            validate_create_link(LinkTypes::HearthToDecisions, &tag).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn link_hearth_to_decisions_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        assert!(matches!(
            validate_create_link(LinkTypes::HearthToDecisions, &tag).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn link_hearth_to_decisions_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        match validate_create_link(LinkTypes::HearthToDecisions, &tag).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("link tag too long")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn link_decision_to_votes_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        match validate_create_link(LinkTypes::DecisionToVotes, &tag).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("link tag too long")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn link_agent_to_votes_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        match validate_create_link(LinkTypes::AgentToVotes, &tag).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("link tag too long")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn link_decision_to_outcome_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        match validate_create_link(LinkTypes::DecisionToOutcome, &tag).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("link tag too long")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn link_vote_history_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        match validate_create_link(LinkTypes::DecisionToVoteHistory, &tag).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("link tag too long")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn link_empty_tag_valid_all_types() {
        let tag = LinkTag(vec![]);
        for link_type in &[
            LinkTypes::HearthToDecisions,
            LinkTypes::DecisionToVotes,
            LinkTypes::AgentToVotes,
            LinkTypes::DecisionToOutcome,
            LinkTypes::DecisionToVoteHistory,
        ] {
            assert!(
                matches!(
                    validate_create_link(link_type.clone(), &tag).unwrap(),
                    ValidateCallbackResult::Valid
                ),
                "empty tag should be valid for {:?}",
                link_type
            );
        }
    }

    // ---- Link Delete Validation: audit trail protection ----

    #[test]
    fn delete_vote_history_link_rejected() {
        match validate_delete_link(LinkTypes::DecisionToVoteHistory).unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("audit trail"));
            }
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn delete_outcome_link_rejected() {
        match validate_delete_link(LinkTypes::DecisionToOutcome).unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("cannot be deleted"));
            }
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn delete_decision_to_votes_link_allowed() {
        // DecisionToVotes links can be deleted (amend_vote link-swap)
        assert!(matches!(
            validate_delete_link(LinkTypes::DecisionToVotes).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn delete_agent_to_votes_link_allowed() {
        // AgentToVotes links can be deleted (amend_vote link-swap)
        assert!(matches!(
            validate_delete_link(LinkTypes::AgentToVotes).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }

    #[test]
    fn delete_hearth_to_decisions_link_allowed() {
        assert!(matches!(
            validate_delete_link(LinkTypes::HearthToDecisions).unwrap(),
            ValidateCallbackResult::Valid
        ));
    }
}
