// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # DAO Coordinator Zome
//!
//! Implements business logic for decentralized governance.
//! This zome is upgradeable - business logic can change without breaking data.
//!
//! ## Core Functions:
//! - Create and manage proposals
//! - Cast and tally votes
//! - Execute approved proposals
//! - Query proposal status and history

use dao_integrity::*;
use hdk::prelude::*;
use mycelix_zome_helpers as _;

/// Create a new governance proposal
#[hdk_extern]
pub fn create_proposal(input: CreateProposalInput) -> ExternResult<ActionHash> {
    // Trust tier gate: requires Participant tier with identity >= 0.25
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_proposal(),
        "create_proposal",
    )?;

    // Get proposer agent info
    let agent_info = agent_info()?;
    let proposer_pubkey = agent_info.agent_initial_pubkey;

    let now = sys_time()?.as_micros() / 1000;
    let voting_deadline = voting_deadline_fn(now as i64, &input.proposal_type);

    // Serialize actions to JSON string
    let actions_json = serde_json::to_string(&input.actions).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to serialize actions: {}",
            e
        )))
    })?;

    // Create proposal entry
    let proposal = Proposal {
        proposal_id: input.proposal_id.clone(),
        title: input.title,
        description: input.description,
        proposer: proposer_pubkey.to_string(),
        proposal_type: input.proposal_type,
        category: input.category,
        actions_json,
        created_at: now as i64,
        voting_deadline,
        status: ProposalStatus::Active,
        for_votes: 0,
        against_votes: 0,
        abstain_votes: 0,
        weighted_for: 0,
        weighted_against: 0,
        voting_mode: VotingMode::Simple,
        executed_at: None,
    };

    let action_hash = create_entry(EntryTypes::Proposal(proposal))?;

    // Link from anchor to proposal
    let path = Path::from(PROPOSALS_ANCHOR);
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllProposals,
        (),
    )?;

    // Link from proposer to proposal
    create_link(
        proposer_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToProposals,
        (),
    )?;

    Ok(action_hash)
}

/// Cast a vote on a proposal
#[hdk_extern]
pub fn cast_vote(input: CastVoteInput) -> ExternResult<ActionHash> {
    // Trust tier gate: requires Participant tier
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_basic(),
        "cast_vote",
    )?;

    let agent_info = agent_info()?;
    let voter_pubkey = agent_info.agent_initial_pubkey;

    // Check if voter already voted
    let existing_votes = get_links(
        LinkQuery::new(
            voter_pubkey.clone(),
            LinkTypes::AgentToVotes.try_into_filter()?,
        )
        .tag_prefix(LinkTag::new(input.proposal_hash.to_string())),
        GetStrategy::Local,
    )?;

    if !existing_votes.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Already voted on this proposal".into()
        )));
    }

    // Create vote entry
    let vote = Vote {
        proposal_id: input.proposal_hash.to_string(),
        voter: voter_pubkey.to_string(),
        choice: input.choice,
        vote_weight: Some(input.weight as u64),
        justification: input.rationale,
        timestamp: (sys_time()?.as_micros() / 1000) as i64,
        reputation_allocated: None,
    };

    let action_hash = create_entry(EntryTypes::Vote(vote))?;

    // Link from proposal to vote
    create_link(
        input.proposal_hash.clone(),
        action_hash.clone(),
        LinkTypes::ProposalToVotes,
        (),
    )?;

    // Link from voter to vote
    create_link(
        voter_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToVotes,
        LinkTag::new(input.proposal_hash.to_string()),
    )?;

    Ok(action_hash)
}

/// Get a proposal by its action hash
#[hdk_extern]
pub fn get_proposal(proposal_hash: ActionHash) -> ExternResult<Option<Proposal>> {
    let record = get(proposal_hash, GetOptions::default())?;
    match record {
        Some(record) => {
            let proposal: Proposal = record
                .entry()
                .to_app_option::<Proposal>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid proposal entry".into()
                )))?;
            Ok(Some(proposal))
        }
        None => Ok(None),
    }
}

/// List all active proposals
#[hdk_extern]
pub fn list_active_proposals(_: ()) -> ExternResult<Vec<Record>> {
    let path = Path::from(PROPOSALS_ANCHOR);
    let links = get_links(
        LinkQuery::new(
            path.path_entry_hash()?,
            LinkTypes::AllProposals.try_into_filter()?,
        ),
        GetStrategy::Local,
    )?;

    let mut proposals = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                proposals.push(record);
            }
        }
    }

    Ok(proposals)
}

// ============== Internal Helpers ==============

const PROPOSALS_ANCHOR: &str = "all_proposals";

fn voting_deadline_fn(now: i64, proposal_type: &ProposalType) -> i64 {
    let duration = match proposal_type {
        ProposalType::Fast => 2 * 24 * 60 * 60,   // 2 days
        ProposalType::Normal => 7 * 24 * 60 * 60, // 7 days
        ProposalType::Slow => 14 * 24 * 60 * 60,  // 14 days
    };
    now + duration
}

/// Input for creating a proposal
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateProposalInput {
    pub proposal_id: String,
    pub title: String,
    pub description: String,
    pub proposal_type: ProposalType,
    pub category: ProposalCategory,
    pub actions: Vec<ProposalAction>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProposalAction {
    pub action_type: String,
    pub params_json: String,
}

/// Input for casting a vote
#[derive(Serialize, Deserialize, Debug)]
pub struct CastVoteInput {
    pub proposal_hash: ActionHash,
    pub choice: VoteChoice,
    pub weight: u32,
    pub rationale: Option<String>,
}

// ============== Reputation & Scoring ==============

/// Input for updating member reputation
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateReputationInput {
    pub member: AgentPubKey,
    pub delta: i32,
    pub reason: String,
}

/// Update a member's local reputation score
#[hdk_extern]
pub fn update_member_reputation(input: UpdateReputationInput) -> ExternResult<ActionHash> {
    // Only Stewards or higher can update reputation manually
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_basic(), // TODO: Make this Steward specific
        "update_reputation",
    )?;

    let current_rep = get_member_reputation(input.member.clone())?;
    let new_level = (current_rep as i32 + input.delta).max(0).min(1000) as u64;

    let random_bytes = random_bytes(32)?;
    let source = ActionHash::from_raw_32(random_bytes.into_vec());

    let reputation = Reputation {
        agent: input.member.clone(),
        subject: "general".into(),
        level_permille: new_level,
        updated_at: (sys_time()?.as_micros() / 1000) as i64,
        source,
    };

    let action_hash = create_entry(EntryTypes::Reputation(reputation))?;

    // Link from agent to reputation record
    let path = Path::from(format!("reputation.{}", input.member));
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AgentToReputation,
        (),
    )?;

    Ok(action_hash)
}

/// Get current reputation score for a member
#[hdk_extern]
pub fn get_member_reputation(member: AgentPubKey) -> ExternResult<u16> {
    let path = Path::from(format!("reputation.{}", member));
    let links = get_links(
        LinkQuery::new(
            path.path_entry_hash()?,
            LinkTypes::AgentToReputation.try_into_filter()?,
        ),
        GetStrategy::Local,
    )?;

    if let Some(link) = links.last() {
        if let Some(hash) = link.target.clone().into_entry_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let rep: Reputation = record
                    .entry()
                    .to_app_option::<Reputation>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid entry".into())))?;
                return Ok(rep.level_permille as u16);
            }
        }
    }
    Ok(0)
}
