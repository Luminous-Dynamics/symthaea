//! Hearth Decisions Coordinator Zome
//!
//! Provides CRUD operations for decisions, voting, tallying, and finalization.

use hdk::prelude::*;
use hearth_decisions_integrity::*;
use hearth_types::*;

// ============================================================================
// Input Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateDecisionInput {
    pub hearth_hash: ActionHash,
    pub title: String,
    pub description: String,
    pub decision_type: DecisionType,
    pub eligible_roles: Vec<MemberRole>,
    pub options: Vec<String>,
    pub deadline: Timestamp,
    pub quorum_bp: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastVoteInput {
    pub decision_hash: ActionHash,
    pub choice: u32,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalizeDecisionInput {
    pub decision_hash: ActionHash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloseDecisionInput {
    pub decision_hash: ActionHash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmendVoteInput {
    pub decision_hash: ActionHash,
    pub choice: u32,
    pub reasoning: Option<String>,
}

// ============================================================================
// Helpers
// ============================================================================

/// Decode a typed value from a ZomeCallResponse.
fn decode_zome_response<T: serde::de::DeserializeOwned + std::fmt::Debug>(
    response: ZomeCallResponse,
    context: &str,
) -> ExternResult<T> {
    match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode {} response: {}",
                context, e
            )))
        }),
        other => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cross-zome call to {} failed: {:?}",
            context, other
        )))),
    }
}

/// Check whether a choice index is within the valid range of options.
fn is_choice_valid(choice: u32, option_count: usize) -> bool {
    (choice as usize) < option_count
}

/// Check whether a role is in the list of eligible roles.
fn is_role_eligible(role: &MemberRole, eligible_roles: &[MemberRole]) -> bool {
    eligible_roles.contains(role)
}

/// Check whether a decision is in a votable state (Open).
fn is_decision_votable(status: &DecisionStatus) -> bool {
    *status == DecisionStatus::Open
}

/// Check whether a decision can be manually closed (must be Open).
fn is_decision_closeable(status: &DecisionStatus) -> bool {
    *status == DecisionStatus::Open
}

/// Check whether the current time is at or past the deadline.
fn is_deadline_passed(now: &Timestamp, deadline: &Timestamp) -> bool {
    now >= deadline
}

/// Compute participation rate in basis points (0–10000).
/// Returns 0 if active_members is 0. Capped at 10000.
fn participation_rate_bp(voter_count: u32, active_members: u32) -> u32 {
    if active_members == 0 {
        return 0;
    }
    ((voter_count as u64 * 10000) / active_members as u64).min(10000) as u32
}

/// Find an existing vote by this agent on the given decision.
/// Searches the agent's vote links and returns the vote's ActionHash if found.
fn find_existing_vote_for_decision(
    _agent: &AgentPubKey,
    decision_hash: &ActionHash,
    agent_vote_links: &[Link],
) -> ExternResult<Option<ActionHash>> {
    for link in agent_vote_links {
        let target =
            link.target
                .clone()
                .into_action_hash()
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Link target is not an ActionHash".into()
                )))?;
        if let Some(record) = get(target.clone(), GetOptions::default())? {
            let existing_vote: Vote = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Failed to deserialize vote: {e}"
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Vote entry is missing".into()
                )))?;
            if existing_vote.decision_hash == *decision_hash {
                return Ok(Some(target));
            }
        }
    }
    Ok(None)
}

/// Check whether the quorum requirement is met.
/// Returns true if no quorum is required (None) or if participation meets the threshold.
fn is_quorum_met(participation_rate_bp: u32, quorum_bp: Option<u32>) -> bool {
    match quorum_bp {
        None => true,
        Some(q) => participation_rate_bp >= q,
    }
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create a new decision for a hearth.
/// Links the decision from the hearth via HearthToDecisions.
#[hdk_extern]
pub fn create_decision(input: CreateDecisionInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;

    let decision = Decision {
        hearth_hash: input.hearth_hash.clone(),
        title: input.title,
        description: input.description,
        decision_type: input.decision_type,
        eligible_roles: input.eligible_roles,
        options: input.options,
        deadline: input.deadline,
        quorum_bp: input.quorum_bp,
        status: DecisionStatus::Open,
        created_by: agent,
        created_at: now,
    };

    let decision_hash = create_entry(&EntryTypes::Decision(decision))?;

    create_link(
        input.hearth_hash,
        decision_hash.clone(),
        LinkTypes::HearthToDecisions,
        (),
    )?;

    let record = get(decision_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created Decision".into())
    ))?;

    Ok(record)
}

/// Cast a vote on a decision.
/// Vote weight is determined by the caller's role via cross-zome call to kinship.
/// Links the vote from the decision and from the agent.
#[hdk_extern]
pub fn cast_vote(input: CastVoteInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;

    // Get the decision to find its hearth_hash
    let decision_record = get(input.decision_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Decision not found".into())),
    )?;
    let decision: Decision = decision_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize decision: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Decision entry is missing".into()
        )))?;

    // 1. Decision must be Open
    if !is_decision_votable(&decision.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot vote on a decision with status {:?} (must be Open)",
            decision.status
        ))));
    }

    // 2. Deadline must not have passed
    if is_deadline_passed(&now, &decision.deadline) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot vote: the decision deadline has passed".into()
        )));
    }

    // 3. Choice must be within bounds
    if !is_choice_valid(input.choice, decision.options.len()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid choice {}: decision has {} options (0..{})",
            input.choice,
            decision.options.len(),
            decision.options.len().saturating_sub(1)
        ))));
    }

    // 4. No duplicate votes — check if this agent already voted on this decision
    let my_vote_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToVotes)?,
        GetStrategy::default(),
    )?;
    if find_existing_vote_for_decision(&agent, &input.decision_hash, &my_vote_links)?.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "You have already voted on this decision".into()
        )));
    }

    // 5. Eligible role — get caller's role via kinship cross-zome call
    let caller_role: Option<MemberRole> = decode_zome_response(
        call(
            CallTargetCell::Local,
            ZomeName::new("hearth_kinship"),
            FunctionName::new("get_caller_role"),
            None,
            decision.hearth_hash,
        )?,
        "get_caller_role",
    )?;

    let role = caller_role.ok_or(wasm_error!(WasmErrorInner::Guest(
        "You are not an active member of this hearth".into()
    )))?;

    if !is_role_eligible(&role, &decision.eligible_roles) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Your role {:?} is not eligible for this decision (eligible: {:?})",
            role, decision.eligible_roles
        ))));
    }

    let weight_bp = role.default_vote_weight_bp();

    let vote = Vote {
        decision_hash: input.decision_hash.clone(),
        voter: agent.clone(),
        choice: input.choice,
        weight_bp,
        reasoning: input.reasoning,
        created_at: now,
    };

    let vote_hash = create_entry(&EntryTypes::Vote(vote))?;

    create_link(
        input.decision_hash,
        vote_hash.clone(),
        LinkTypes::DecisionToVotes,
        (),
    )?;

    create_link(agent, vote_hash.clone(), LinkTypes::AgentToVotes, ())?;

    let record = get(vote_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created Vote".into())
    ))?;

    Ok(record)
}

/// Tally all votes for a decision.
/// Returns Vec<(option_index, total_weight_bp)> sorted by option index.
#[hdk_extern]
pub fn tally_votes(decision_hash: ActionHash) -> ExternResult<Vec<(u32, u32)>> {
    let links = get_links(
        LinkQuery::try_new(decision_hash, LinkTypes::DecisionToVotes)?,
        GetStrategy::default(),
    )?;

    // Collect all votes
    let mut tallies: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

    for link in links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        if let Some(record) = get(target, GetOptions::default())? {
            let vote: Vote = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Failed to deserialize vote: {e}"
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Vote entry is missing".into()
                )))?;

            let current = tallies.entry(vote.choice).or_insert(0);
            *current = current.saturating_add(vote.weight_bp);
        }
    }

    let mut result: Vec<(u32, u32)> = tallies.into_iter().collect();
    result.sort_by_key(|(idx, _)| *idx);

    Ok(result)
}

/// Finalize a decision: tally votes, create DecisionOutcome, update decision status.
#[hdk_extern]
pub fn finalize_decision(input: FinalizeDecisionInput) -> ExternResult<Record> {
    let now = sys_time()?;

    // Get the decision
    let decision_record = get(input.decision_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Decision not found".into())),
    )?;
    let mut decision: Decision = decision_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize decision: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Decision entry is missing".into()
        )))?;

    // 1. Decision must be Open (prevents re-finalization)
    if !is_decision_votable(&decision.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot finalize a decision with status {:?} (must be Open)",
            decision.status
        ))));
    }

    // 2. Deadline must have passed (prevents premature finalization)
    if !is_deadline_passed(&now, &decision.deadline) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot finalize: the decision deadline has not yet passed".into()
        )));
    }

    // Tally votes
    let tallies = tally_votes(input.decision_hash.clone())?;

    // Find the winning option (highest total weight)
    let chosen_option = tallies
        .iter()
        .max_by_key(|(_, weight)| *weight)
        .map(|(idx, _)| *idx)
        .unwrap_or(0);

    // Calculate participation rate: voter count / active member count
    let vote_links = get_links(
        LinkQuery::try_new(input.decision_hash.clone(), LinkTypes::DecisionToVotes)?,
        GetStrategy::default(),
    )?;
    let voter_count = vote_links.len() as u32;

    // Get active member count via kinship cross-zome call
    let active_members: u32 = decode_zome_response(
        call(
            CallTargetCell::Local,
            ZomeName::new("hearth_kinship"),
            FunctionName::new("get_active_member_count"),
            None,
            decision.hearth_hash.clone(),
        )?,
        "get_active_member_count",
    )?;

    let participation_rate_bp = participation_rate_bp(voter_count, active_members);

    // 3. Quorum must be met (if set)
    if !is_quorum_met(participation_rate_bp, decision.quorum_bp) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot finalize: quorum not met (participation {}bp < required {}bp)",
            participation_rate_bp,
            decision.quorum_bp.unwrap_or(0)
        ))));
    }

    // Create the outcome
    let outcome = DecisionOutcome {
        decision_hash: input.decision_hash.clone(),
        chosen_option,
        participation_rate_bp,
        resolved_at: now,
    };

    let outcome_hash = create_entry(&EntryTypes::DecisionOutcome(outcome))?;

    create_link(
        input.decision_hash.clone(),
        outcome_hash.clone(),
        LinkTypes::DecisionToOutcome,
        (),
    )?;

    // Update decision status to Finalized
    decision.status = DecisionStatus::Finalized;
    update_entry(input.decision_hash, &decision)?;

    let record = get(outcome_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created DecisionOutcome".into())
    ))?;

    Ok(record)
}

/// Close a decision before its deadline.
/// Auth: the decision creator OR any guardian-level member of the hearth.
#[hdk_extern]
pub fn close_decision(input: CloseDecisionInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;

    let decision_record = get(input.decision_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Decision not found".into())),
    )?;
    let mut decision: Decision = decision_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize decision: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Decision entry is missing".into()
        )))?;

    // 1. Decision must be Open
    if !is_decision_closeable(&decision.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot close a decision with status {:?} (must be Open)",
            decision.status
        ))));
    }

    // 2. Auth: creator OR guardian
    if agent != decision.created_by {
        let caller_role: Option<MemberRole> = decode_zome_response(
            call(
                CallTargetCell::Local,
                ZomeName::new("hearth_kinship"),
                FunctionName::new("get_caller_role"),
                None,
                decision.hearth_hash.clone(),
            )?,
            "get_caller_role",
        )?;

        let role = caller_role.ok_or(wasm_error!(WasmErrorInner::Guest(
            "You are not an active member of this hearth".into()
        )))?;

        if !role.is_guardian() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only the decision creator or a guardian can close a decision".into()
            )));
        }
    }

    // 3. Close the decision
    decision.status = DecisionStatus::Closed;
    update_entry(input.decision_hash.clone(), &decision)?;

    let record = get(input.decision_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the updated Decision".into())
    ))?;

    Ok(record)
}

/// Amend an existing vote on a decision (link-swap pattern).
/// The old vote entry remains on the DHT as an audit trail; its links are deleted
/// and replaced with links to the new vote entry.
#[hdk_extern]
pub fn amend_vote(input: AmendVoteInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;

    // Get the decision
    let decision_record = get(input.decision_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Decision not found".into())),
    )?;
    let decision: Decision = decision_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize decision: {e}"
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Decision entry is missing".into()
        )))?;

    // 1. Decision must be Open
    if !is_decision_votable(&decision.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot amend vote on a decision with status {:?} (must be Open)",
            decision.status
        ))));
    }

    // 2. Deadline must not have passed
    if is_deadline_passed(&now, &decision.deadline) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot amend vote: the decision deadline has passed".into()
        )));
    }

    // 3. Choice must be within bounds
    if !is_choice_valid(input.choice, decision.options.len()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid choice {}: decision has {} options (0..{})",
            input.choice,
            decision.options.len(),
            decision.options.len().saturating_sub(1)
        ))));
    }

    // 4. Must have an existing vote to amend
    let my_vote_links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToVotes)?,
        GetStrategy::default(),
    )?;
    let _existing_vote_hash =
        find_existing_vote_for_decision(&agent, &input.decision_hash, &my_vote_links)?.ok_or(
            wasm_error!(WasmErrorInner::Guest(
                "No existing vote to amend on this decision".into()
            )),
        )?;

    // 5. Delete old links (DecisionToVotes and AgentToVotes)
    let decision_vote_links = get_links(
        LinkQuery::try_new(input.decision_hash.clone(), LinkTypes::DecisionToVotes)?,
        GetStrategy::default(),
    )?;
    for link in &decision_vote_links {
        if let Some(target) = link.target.clone().into_action_hash() {
            if target == _existing_vote_hash {
                delete_link(link.create_link_hash.clone(), GetOptions::default())?;
            }
        }
    }
    for link in &my_vote_links {
        if let Some(target) = link.target.clone().into_action_hash() {
            if target == _existing_vote_hash {
                delete_link(link.create_link_hash.clone(), GetOptions::default())?;
            }
        }
    }

    // 6. Get caller role + eligible check
    let caller_role: Option<MemberRole> = decode_zome_response(
        call(
            CallTargetCell::Local,
            ZomeName::new("hearth_kinship"),
            FunctionName::new("get_caller_role"),
            None,
            decision.hearth_hash,
        )?,
        "get_caller_role",
    )?;

    let role = caller_role.ok_or(wasm_error!(WasmErrorInner::Guest(
        "You are not an active member of this hearth".into()
    )))?;

    if !is_role_eligible(&role, &decision.eligible_roles) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Your role {:?} is not eligible for this decision (eligible: {:?})",
            role, decision.eligible_roles
        ))));
    }

    let weight_bp = role.default_vote_weight_bp();

    // 7. Create new vote entry + links
    let vote = Vote {
        decision_hash: input.decision_hash.clone(),
        voter: agent.clone(),
        choice: input.choice,
        weight_bp,
        reasoning: input.reasoning,
        created_at: now,
    };

    let vote_hash = create_entry(&EntryTypes::Vote(vote))?;

    create_link(
        input.decision_hash,
        vote_hash.clone(),
        LinkTypes::DecisionToVotes,
        (),
    )?;

    create_link(agent, vote_hash.clone(), LinkTypes::AgentToVotes, ())?;

    let record = get(vote_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not find the newly created Vote".into())
    ))?;

    Ok(record)
}

/// Get all decisions for a hearth.
#[hdk_extern]
pub fn get_hearth_decisions(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToDecisions)?,
        GetStrategy::default(),
    )?;

    let mut decisions = Vec::new();
    for link in links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        if let Some(record) = get(target, GetOptions::default())? {
            decisions.push(record);
        }
    }

    Ok(decisions)
}

/// Get all votes for a decision.
#[hdk_extern]
pub fn get_decision_votes(decision_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(decision_hash, LinkTypes::DecisionToVotes)?,
        GetStrategy::default(),
    )?;

    let mut votes = Vec::new();
    for link in links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        if let Some(record) = get(target, GetOptions::default())? {
            votes.push(record);
        }
    }

    Ok(votes)
}

/// Get decisions in a hearth where the calling agent has not yet voted.
#[hdk_extern]
pub fn get_my_pending_votes(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Get all agent's vote links
    let my_vote_links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToVotes)?,
        GetStrategy::default(),
    )?;

    // Collect the decision hashes this agent has already voted on
    let mut voted_decision_hashes: std::collections::HashSet<ActionHash> =
        std::collections::HashSet::new();
    for link in my_vote_links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        if let Some(record) = get(target, GetOptions::default())? {
            let vote: Vote = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Failed to deserialize vote: {e}"
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Vote entry is missing".into()
                )))?;
            voted_decision_hashes.insert(vote.decision_hash);
        }
    }

    // Get all hearth decisions
    let decision_links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToDecisions)?,
        GetStrategy::default(),
    )?;

    let mut pending = Vec::new();
    for link in decision_links {
        let target = link
            .target
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Link target is not an ActionHash".into()
            )))?;

        // Skip decisions the agent already voted on
        if voted_decision_hashes.contains(&target) {
            continue;
        }

        if let Some(record) = get(target, GetOptions::default())? {
            let decision: Decision = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Failed to deserialize decision: {e}"
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Decision entry is missing".into()
                )))?;

            // Only include open decisions
            if decision.status == DecisionStatus::Open {
                pending.push(record);
            }
        }
    }

    Ok(pending)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Input Type Serde ----

    #[test]
    fn create_decision_input_serde_roundtrip() {
        let input = CreateDecisionInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            title: "Where to eat?".into(),
            description: "Family dinner vote".into(),
            decision_type: DecisionType::MajorityVote,
            eligible_roles: vec![MemberRole::Adult, MemberRole::Elder],
            options: vec!["Pizza".into(), "Tacos".into()],
            deadline: Timestamp::from_micros(2_000_000),
            quorum_bp: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateDecisionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.title, "Where to eat?");
        assert_eq!(back.options.len(), 2);
    }

    #[test]
    fn cast_vote_input_serde_roundtrip() {
        let input = CastVoteInput {
            decision_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            choice: 1,
            reasoning: Some("I prefer tacos".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CastVoteInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.choice, 1);
        assert_eq!(back.reasoning.unwrap(), "I prefer tacos");
    }

    #[test]
    fn cast_vote_input_no_reasoning() {
        let input = CastVoteInput {
            decision_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            choice: 0,
            reasoning: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CastVoteInput = serde_json::from_str(&json).unwrap();
        assert!(back.reasoning.is_none());
    }

    #[test]
    fn finalize_decision_input_serde_roundtrip() {
        let input = FinalizeDecisionInput {
            decision_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        let _back: FinalizeDecisionInput = serde_json::from_str(&json).unwrap();
    }

    // ---- AmendVoteInput Serde ----

    #[test]
    fn amend_vote_input_serde_roundtrip() {
        let input = AmendVoteInput {
            decision_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            choice: 2,
            reasoning: Some("Changed my mind".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: AmendVoteInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.choice, 2);
        assert_eq!(back.reasoning.unwrap(), "Changed my mind");
    }

    #[test]
    fn amend_vote_input_no_reasoning() {
        let input = AmendVoteInput {
            decision_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            choice: 0,
            reasoning: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: AmendVoteInput = serde_json::from_str(&json).unwrap();
        assert!(back.reasoning.is_none());
    }

    #[test]
    fn amend_vote_input_with_reasoning() {
        let input = AmendVoteInput {
            decision_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            choice: 5,
            reasoning: Some("After further thought".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: AmendVoteInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.choice, 5);
        assert!(back.reasoning.is_some());
    }

    // ---- CloseDecisionInput Serde ----

    #[test]
    fn close_decision_input_serde_roundtrip() {
        let input = CloseDecisionInput {
            decision_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CloseDecisionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.decision_hash, input.decision_hash);
    }

    // ---- Pure helper: is_decision_closeable ----

    #[test]
    fn closeable_open() {
        assert!(is_decision_closeable(&DecisionStatus::Open));
    }

    #[test]
    fn not_closeable_closed() {
        assert!(!is_decision_closeable(&DecisionStatus::Closed));
    }

    #[test]
    fn not_closeable_finalized() {
        assert!(!is_decision_closeable(&DecisionStatus::Finalized));
    }

    #[test]
    fn closeable_matches_votable() {
        // Closeable and votable have the same predicate (Open)
        for status in &[
            DecisionStatus::Open,
            DecisionStatus::Closed,
            DecisionStatus::Finalized,
        ] {
            assert_eq!(
                is_decision_closeable(status),
                is_decision_votable(status),
                "mismatch for {:?}",
                status
            );
        }
    }

    #[test]
    fn create_decision_input_all_types() {
        let types = vec![
            DecisionType::Consensus,
            DecisionType::MajorityVote,
            DecisionType::ElderDecision,
            DecisionType::GuardianDecision,
        ];
        for dt in types {
            let input = CreateDecisionInput {
                hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
                title: "Test".into(),
                description: "".into(),
                decision_type: dt,
                eligible_roles: vec![MemberRole::Adult],
                options: vec!["A".into(), "B".into()],
                deadline: Timestamp::from_micros(2_000_000),
                quorum_bp: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let _back: CreateDecisionInput = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn create_decision_input_all_roles() {
        let roles = vec![
            MemberRole::Founder,
            MemberRole::Elder,
            MemberRole::Adult,
            MemberRole::Youth,
            MemberRole::Child,
            MemberRole::Guest,
            MemberRole::Ancestor,
        ];
        let input = CreateDecisionInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            title: "Test".into(),
            description: "".into(),
            decision_type: DecisionType::Consensus,
            eligible_roles: roles,
            options: vec!["A".into(), "B".into()],
            deadline: Timestamp::from_micros(2_000_000),
            quorum_bp: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateDecisionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.eligible_roles.len(), 7);
    }

    #[test]
    fn create_decision_input_many_options() {
        let options: Vec<String> = (0..20).map(|i| format!("Option {i}")).collect();
        let input = CreateDecisionInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            title: "Big vote".into(),
            description: "Many options".into(),
            decision_type: DecisionType::MajorityVote,
            eligible_roles: vec![MemberRole::Adult],
            options,
            deadline: Timestamp::from_micros(2_000_000),
            quorum_bp: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateDecisionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.options.len(), 20);
    }

    #[test]
    fn cast_vote_input_high_choice() {
        let input = CastVoteInput {
            decision_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            choice: 19,
            reasoning: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CastVoteInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.choice, 19);
    }

    // ---- Pure helper: is_choice_valid ----

    #[test]
    fn choice_valid_first_option() {
        assert!(is_choice_valid(0, 3));
    }

    #[test]
    fn choice_valid_last_option() {
        assert!(is_choice_valid(2, 3));
    }

    #[test]
    fn choice_invalid_out_of_bounds() {
        assert!(!is_choice_valid(3, 3));
    }

    #[test]
    fn choice_invalid_empty_options() {
        assert!(!is_choice_valid(0, 0));
    }

    #[test]
    fn choice_invalid_large_index() {
        assert!(!is_choice_valid(100, 5));
    }

    // ---- Pure helper: is_role_eligible ----

    #[test]
    fn role_eligible_adult_in_list() {
        let eligible = vec![MemberRole::Adult, MemberRole::Elder];
        assert!(is_role_eligible(&MemberRole::Adult, &eligible));
    }

    #[test]
    fn role_not_eligible_youth_excluded() {
        let eligible = vec![MemberRole::Adult, MemberRole::Elder];
        assert!(!is_role_eligible(&MemberRole::Youth, &eligible));
    }

    #[test]
    fn role_eligible_all_roles() {
        let eligible = vec![
            MemberRole::Founder,
            MemberRole::Elder,
            MemberRole::Adult,
            MemberRole::Youth,
            MemberRole::Child,
        ];
        assert!(is_role_eligible(&MemberRole::Youth, &eligible));
    }

    #[test]
    fn role_not_eligible_empty_list() {
        assert!(!is_role_eligible(&MemberRole::Adult, &[]));
    }

    // ---- Pure helper: is_decision_votable ----

    #[test]
    fn decision_votable_open() {
        assert!(is_decision_votable(&DecisionStatus::Open));
    }

    #[test]
    fn decision_not_votable_closed() {
        assert!(!is_decision_votable(&DecisionStatus::Closed));
    }

    #[test]
    fn decision_not_votable_finalized() {
        assert!(!is_decision_votable(&DecisionStatus::Finalized));
    }

    // ---- Pure helper: is_deadline_passed ----

    #[test]
    fn deadline_not_passed_before() {
        let now = Timestamp::from_micros(1_000_000);
        let deadline = Timestamp::from_micros(2_000_000);
        assert!(!is_deadline_passed(&now, &deadline));
    }

    #[test]
    fn deadline_passed_exact() {
        let t = Timestamp::from_micros(2_000_000);
        assert!(is_deadline_passed(&t, &t));
    }

    #[test]
    fn deadline_passed_after() {
        let now = Timestamp::from_micros(3_000_000);
        let deadline = Timestamp::from_micros(2_000_000);
        assert!(is_deadline_passed(&now, &deadline));
    }

    // ---- Pure helper: participation_rate_bp ----

    #[test]
    fn participation_full() {
        assert_eq!(participation_rate_bp(5, 5), 10000);
    }

    #[test]
    fn participation_half() {
        assert_eq!(participation_rate_bp(5, 10), 5000);
    }

    #[test]
    fn participation_zero_voters() {
        assert_eq!(participation_rate_bp(0, 10), 0);
    }

    #[test]
    fn participation_zero_members() {
        assert_eq!(participation_rate_bp(5, 0), 0);
    }

    #[test]
    fn participation_capped_at_10000() {
        // More voters than members shouldn't exceed 10000
        assert_eq!(participation_rate_bp(15, 10), 10000);
    }

    // ---- Pure helper: is_quorum_met ----

    #[test]
    fn quorum_none_always_met() {
        assert!(is_quorum_met(0, None));
        assert!(is_quorum_met(5000, None));
        assert!(is_quorum_met(10000, None));
    }

    #[test]
    fn quorum_exact_threshold_met() {
        assert!(is_quorum_met(5000, Some(5000)));
    }

    #[test]
    fn quorum_exceeds_threshold() {
        assert!(is_quorum_met(7500, Some(5000)));
    }

    #[test]
    fn quorum_not_met() {
        assert!(!is_quorum_met(4999, Some(5000)));
    }

    #[test]
    fn quorum_zero_threshold_always_met() {
        assert!(is_quorum_met(0, Some(0)));
        assert!(is_quorum_met(10000, Some(0)));
    }

    #[test]
    fn quorum_full_participation_required() {
        assert!(is_quorum_met(10000, Some(10000)));
        assert!(!is_quorum_met(9999, Some(10000)));
    }

    #[test]
    fn create_decision_input_with_quorum_serde() {
        let input = CreateDecisionInput {
            hearth_hash: ActionHash::from_raw_36(vec![0xABu8; 36]),
            title: "Quorum vote".into(),
            description: "Needs 50% participation".into(),
            decision_type: DecisionType::MajorityVote,
            eligible_roles: vec![MemberRole::Adult],
            options: vec!["Yes".into(), "No".into()],
            deadline: Timestamp::from_micros(2_000_000),
            quorum_bp: Some(5000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateDecisionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.quorum_bp, Some(5000));
    }

    // ========================================================================
    // Scenario-Based Integration Tests (pure function decision tree)
    // ========================================================================

    // ---- Status Lifecycle ----

    #[test]
    fn lifecycle_open_is_votable_and_closeable() {
        let status = DecisionStatus::Open;
        assert!(is_decision_votable(&status));
        assert!(is_decision_closeable(&status));
    }

    #[test]
    fn lifecycle_closed_blocks_voting_and_closing() {
        let status = DecisionStatus::Closed;
        assert!(!is_decision_votable(&status));
        assert!(!is_decision_closeable(&status));
    }

    #[test]
    fn lifecycle_finalized_blocks_everything() {
        let status = DecisionStatus::Finalized;
        assert!(!is_decision_votable(&status));
        assert!(!is_decision_closeable(&status));
    }

    // ---- Combined Validation Scenarios ----

    #[test]
    fn scenario_vote_valid_open_before_deadline_eligible() {
        let status = DecisionStatus::Open;
        let now = Timestamp::from_micros(1_000_000);
        let deadline = Timestamp::from_micros(2_000_000);
        let role = MemberRole::Adult;
        let eligible = vec![MemberRole::Adult, MemberRole::Elder];

        assert!(is_decision_votable(&status));
        assert!(!is_deadline_passed(&now, &deadline));
        assert!(is_choice_valid(0, 3));
        assert!(is_role_eligible(&role, &eligible));
    }

    #[test]
    fn scenario_vote_rejected_past_deadline_even_if_open() {
        let status = DecisionStatus::Open;
        let now = Timestamp::from_micros(3_000_000);
        let deadline = Timestamp::from_micros(2_000_000);

        assert!(is_decision_votable(&status));
        assert!(is_deadline_passed(&now, &deadline));
        // Vote should be rejected despite Open status
    }

    #[test]
    fn scenario_vote_rejected_closed_before_deadline() {
        let status = DecisionStatus::Closed;
        let now = Timestamp::from_micros(1_000_000);
        let deadline = Timestamp::from_micros(2_000_000);

        assert!(!is_decision_votable(&status));
        assert!(!is_deadline_passed(&now, &deadline));
        // Vote should be rejected despite deadline not passed
    }

    #[test]
    fn scenario_finalize_requires_deadline_passed() {
        let status = DecisionStatus::Open;
        let now = Timestamp::from_micros(3_000_000);
        let deadline = Timestamp::from_micros(2_000_000);

        assert!(is_decision_votable(&status));
        assert!(is_deadline_passed(&now, &deadline));
        // Finalization should proceed
    }

    #[test]
    fn scenario_finalize_rejects_pre_deadline() {
        let status = DecisionStatus::Open;
        let now = Timestamp::from_micros(1_000_000);
        let deadline = Timestamp::from_micros(2_000_000);

        assert!(is_decision_votable(&status));
        assert!(!is_deadline_passed(&now, &deadline));
        // Finalization should be rejected
    }

    #[test]
    fn scenario_finalize_quorum_met() {
        // 8 of 10 members voted = 8000bp, quorum = 5000bp
        let rate = participation_rate_bp(8, 10);
        assert_eq!(rate, 8000);
        assert!(is_quorum_met(rate, Some(5000)));
    }

    #[test]
    fn scenario_finalize_quorum_not_met() {
        // 2 of 10 members voted = 2000bp, quorum = 5000bp
        let rate = participation_rate_bp(2, 10);
        assert_eq!(rate, 2000);
        assert!(!is_quorum_met(rate, Some(5000)));
    }

    #[test]
    fn scenario_finalize_no_quorum_always_passes() {
        // Even with 0 voters, no quorum means it passes
        let rate = participation_rate_bp(0, 10);
        assert_eq!(rate, 0);
        assert!(is_quorum_met(rate, None));
    }

    #[test]
    fn scenario_all_roles_weight_check() {
        let roles = vec![
            (MemberRole::Founder, 10000u32),
            (MemberRole::Elder, 10000),
            (MemberRole::Adult, 10000),
            (MemberRole::Youth, 5000),
            (MemberRole::Child, 0),
            (MemberRole::Guest, 0),
            (MemberRole::Ancestor, 0),
        ];
        for (role, expected_weight) in roles {
            assert_eq!(
                role.default_vote_weight_bp(),
                expected_weight,
                "unexpected weight for {:?}",
                role
            );
        }
    }

    #[test]
    fn scenario_choice_boundary_max_options() {
        // Decision with exactly 20 options (max allowed)
        assert!(is_choice_valid(0, 20));
        assert!(is_choice_valid(19, 20));
        assert!(!is_choice_valid(20, 20));
    }

    #[test]
    fn scenario_participation_edge_cases() {
        // Single voter, single member = 100%
        assert_eq!(participation_rate_bp(1, 1), 10000);
        // 1 voter, 3 members = 33.33% ≈ 3333bp
        assert_eq!(participation_rate_bp(1, 3), 3333);
        // 2 voters, 3 members = 66.66% ≈ 6666bp
        assert_eq!(participation_rate_bp(2, 3), 6666);
        // All edge: 0 voters, 0 members = 0
        assert_eq!(participation_rate_bp(0, 0), 0);
    }
}
