//! RB-BFT Coordinator Zome
//!
//! Implements the Reputation-Based Byzantine Fault Tolerance consensus protocol.
//! Achieves 45% Byzantine tolerance through reputation² weighted voting.

use hdk::prelude::*;
use rbbft_integrity::*;

/// Register as a validator with initial K-Vector
#[hdk_extern]
pub fn register_validator(kvector: KVector) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let registration = ValidatorRegistration {
        agent: agent.clone(),
        kvector,
        registered_at: sys_time()?,
        stake_proof: None,
    };

    let action_hash = create_entry(EntryTypes::ValidatorRegistration(registration))?;

    // Link to validator index
    let index_anchor = anchor("validators".into(), "all".into())?;
    create_link(
        index_anchor,
        action_hash.clone(),
        LinkTypes::ValidatorIndex,
        (),
    )?;

    // Link agent to their registration
    create_link(
        agent,
        action_hash.clone(),
        LinkTypes::ValidatorToRegistration,
        (),
    )?;

    Ok(action_hash)
}

/// Get all registered validators
#[hdk_extern]
pub fn get_validators(_: ()) -> ExternResult<Vec<ValidatorRegistration>> {
    let index_anchor = anchor("validators".into(), "all".into())?;
    let links = get_links(
        LinkQuery::try_new(index_anchor, LinkTypes::ValidatorIndex)?,
        GetStrategy::default()
    )?;

    let mut validators = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(reg) = record.entry().to_app_option::<ValidatorRegistration>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))? {
                    validators.push(reg);
                }
            }
        }
    }

    Ok(validators)
}

/// Propose a new block
#[hdk_extern]
pub fn propose_block(input: ProposeBlockInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Get proposer's K-Vector from their registration
    let kvector = get_my_kvector()?;

    let proposal = BlockProposal {
        round: input.round,
        view: input.view,
        proposer: agent,
        proposer_kvector: kvector,
        height: input.height,
        parent_hash: input.parent_hash,
        content_root: input.content_root,
        timestamp: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::BlockProposal(proposal))?;

    // Link to round index
    let round_anchor = anchor("rounds".into(), input.round.to_string().into())?;
    create_link(
        round_anchor,
        action_hash.clone(),
        LinkTypes::RoundToProposals,
        (),
    )?;

    Ok(action_hash)
}

/// Input for proposing a block
#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeBlockInput {
    pub round: u64,
    pub view: u64,
    pub height: u64,
    pub parent_hash: Option<ActionHash>,
    pub content_root: Vec<u8>,
}

/// Vote on a proposal
#[hdk_extern]
pub fn vote_on_proposal(input: VoteInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    // Get voter's reputation
    let kvector = get_my_kvector()?;

    let vote = Vote {
        voter: agent,
        vote_type: input.vote_type,
        proposal_hash: input.proposal_hash.clone(),
        round: input.round,
        value: input.value,
        reputation: kvector.k_r,
        timestamp: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::Vote(vote))?;

    // Link to proposal
    create_link(
        input.proposal_hash,
        action_hash.clone(),
        LinkTypes::ProposalToVotes,
        (),
    )?;

    Ok(action_hash)
}

/// Input for voting
#[derive(Serialize, Deserialize, Debug)]
pub struct VoteInput {
    pub proposal_hash: ActionHash,
    pub round: u64,
    pub vote_type: VoteType,
    pub value: bool,
}

/// Get votes for a proposal
#[hdk_extern]
pub fn get_proposal_votes(proposal_hash: ActionHash) -> ExternResult<Vec<Vote>> {
    let links = get_links(
        LinkQuery::try_new(proposal_hash, LinkTypes::ProposalToVotes)?,
        GetStrategy::default()
    )?;

    let mut votes = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(vote) = record.entry().to_app_option::<Vote>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))? {
                    votes.push(vote);
                }
            }
        }
    }

    Ok(votes)
}

/// Calculate weighted consensus result for a proposal
#[hdk_extern]
pub fn calculate_consensus(proposal_hash: ActionHash) -> ExternResult<ConsensusResult> {
    let votes = get_proposal_votes(proposal_hash)?;

    if votes.is_empty() {
        return Ok(ConsensusResult {
            weighted_for: 0.0,
            weighted_total: 0.0,
            quorum_reached: false,
            accepted: false,
        });
    }

    let mut weighted_for = 0.0f32;
    let mut weighted_total = 0.0f32;

    for vote in &votes {
        let weight = vote.weight();
        weighted_total += weight;
        if vote.value {
            weighted_for += weight;
        }
    }

    let quorum_reached = weighted_total > 0.0 &&
        (weighted_for / weighted_total) >= QUORUM_THRESHOLD;

    Ok(ConsensusResult {
        weighted_for,
        weighted_total,
        quorum_reached,
        accepted: quorum_reached,
    })
}

/// Consensus calculation result
#[derive(Serialize, Deserialize, Debug)]
pub struct ConsensusResult {
    pub weighted_for: f32,
    pub weighted_total: f32,
    pub quorum_reached: bool,
    pub accepted: bool,
}

/// Commit a block after consensus
#[hdk_extern]
pub fn commit_block(input: CommitBlockInput) -> ExternResult<ActionHash> {
    let consensus = calculate_consensus(input.proposal_hash.clone())?;

    if !consensus.accepted {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot commit block - consensus not reached".into()
        )));
    }

    let committed = CommittedBlock {
        proposal_hash: input.proposal_hash,
        round: input.round,
        height: input.height,
        weighted_votes_for: consensus.weighted_for,
        weighted_votes_total: consensus.weighted_total,
        committed_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::CommittedBlock(committed))?;

    // Link to block chain
    let height_anchor = anchor("blocks".into(), input.height.to_string().into())?;
    create_link(
        height_anchor,
        action_hash.clone(),
        LinkTypes::BlockChain,
        (),
    )?;

    Ok(action_hash)
}

/// Input for committing a block
#[derive(Serialize, Deserialize, Debug)]
pub struct CommitBlockInput {
    pub proposal_hash: ActionHash,
    pub round: u64,
    pub height: u64,
}

/// Challenge a validator for misbehavior
#[hdk_extern]
pub fn challenge_validator(input: ChallengeInput) -> ExternResult<ActionHash> {
    let challenger = agent_info()?.agent_initial_pubkey;

    let evidence = ChallengeEvidence {
        validator: input.validator.clone(),
        violation_type: input.violation_type,
        proof: input.proof,
        round: input.round,
        challenger,
        timestamp: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::ChallengeEvidence(evidence))?;

    // Link to validator
    create_link(
        input.validator,
        action_hash.clone(),
        LinkTypes::ValidatorToChallenges,
        (),
    )?;

    Ok(action_hash)
}

/// Input for challenging a validator
#[derive(Serialize, Deserialize, Debug)]
pub struct ChallengeInput {
    pub validator: AgentPubKey,
    pub violation_type: ViolationType,
    pub proof: Vec<u8>,
    pub round: u64,
}

/// Get the current agent's K-Vector from their registration
fn get_my_kvector() -> ExternResult<KVector> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::ValidatorToRegistration)?,
        GetStrategy::default()
    )?;

    if let Some(link) = links.first() {
        if let Some(target) = link.target.clone().into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(reg) = record.entry().to_app_option::<ValidatorRegistration>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))? {
                    return Ok(reg.kvector);
                }
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "Agent is not registered as a validator".into()
    )))
}

/// Update validator's K-Vector (e.g., after reputation change)
#[hdk_extern]
pub fn update_kvector(kvector: KVector) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::ValidatorToRegistration)?,
        GetStrategy::default()
    )?;

    let original_hash = links
        .first()
        .and_then(|l| l.target.clone().into_action_hash())
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
            "No registration found for agent".into()
        )))?;

    let original = get(original_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
            "Registration not found".into()
        )))?;

    let mut reg = original
        .entry()
        .to_app_option::<ValidatorRegistration>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
            "Invalid registration entry".into()
        )))?;

    reg.kvector = kvector;

    update_entry(original_hash, EntryTypes::ValidatorRegistration(reg))
}

/// Create an anchor for indexing
fn anchor(anchor_type: String, anchor_text: String) -> ExternResult<EntryHash> {
    // Use a deterministic hash based on anchor type and text
    let path = Path::from(format!("{}.{}", anchor_type, anchor_text));
    path.path_entry_hash()
}
