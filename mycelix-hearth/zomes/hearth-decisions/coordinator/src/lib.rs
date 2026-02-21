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
/// For simplicity, uses a constant weight of 10000 since we cannot
/// easily read the caller's role from another zome without a bridge call.
/// Links the vote from the decision and from the agent.
#[hdk_extern]
pub fn cast_vote(input: CastVoteInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;

    let vote = Vote {
        decision_hash: input.decision_hash.clone(),
        voter: agent.clone(),
        choice: input.choice,
        weight_bp: 10000,
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

    // Tally votes
    let tallies = tally_votes(input.decision_hash.clone())?;

    // Find the winning option (highest total weight)
    let chosen_option = tallies
        .iter()
        .max_by_key(|(_, weight)| *weight)
        .map(|(idx, _)| *idx)
        .unwrap_or(0);

    // Calculate participation rate: total votes / eligible member count
    // Since we can't query membership from this zome, we use total_weight / (voters * 10000)
    // as a rough proxy. For now, count distinct voters.
    let vote_links = get_links(
        LinkQuery::try_new(input.decision_hash.clone(), LinkTypes::DecisionToVotes)?,
        GetStrategy::default(),
    )?;
    let voter_count = vote_links.len() as u32;

    // Participation rate: we don't know total eligible members, so we store
    // the voter count scaled. If no voters, participation is 0.
    // For a proper implementation this would query the kinship zome.
    // Here we just store the voter count as a raw basis-point value capped at 10000.
    let participation_rate_bp = voter_count.min(10000);

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
}
