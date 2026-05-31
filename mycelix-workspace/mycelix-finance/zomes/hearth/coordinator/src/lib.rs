// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! HEARTH (Commons Pool) Coordinator Zome
//!
//! Implements Commons Charter Article II, Section 2 - Commons Hearths
//!
//! A HEARTH is a community-governed pool of resources.
//! - Anyone can contribute
//! - Community votes on allocation requests
//! - Transparent governance (all activity visible)
//!
//! Philosophy: "From each according to ability, to each according to need"
//! but with democratic accountability.

use hdk::prelude::*;

// Re-export integrity types for external use
pub use hearth_integrity::*;

// =============================================================================
// INPUT/OUTPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreatePoolInput {
    pub name: String,
    pub description: String,
    pub dao_did: String,
    pub resource_type: ResourceType,
    pub cultural_alias: Option<String>,
    pub governance: Option<PoolGovernanceInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PoolGovernanceInput {
    pub allocation_threshold: Option<f32>,
    pub voting_period_hours: Option<u32>,
    pub contribution_weighted_voting: Option<bool>,
    pub min_civ_to_request: Option<f32>,
    pub max_allocation_percent: Option<f32>,
    pub require_justification: Option<bool>,
    pub request_cooldown_hours: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ContributeInput {
    pub pool_id: String,
    pub amount: f64,
    pub message: Option<String>,
    pub anonymous: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestAllocationInput {
    pub pool_id: String,
    pub recipient_did: Option<String>, // If None, requester is recipient
    pub amount: f64,
    pub justification: String,
    pub need_category: NeedCategory,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VoteInput {
    pub request_id: String,
    pub vote: bool, // true = for, false = against
    pub comment: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PoolSummary {
    pub id: String,
    pub name: String,
    pub description: String,
    pub dao_did: String,
    pub resource_type: ResourceType,
    pub total_resources: f64,
    pub contributor_count: u32,
    pub status: PoolStatus,
    pub pending_requests: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestSummary {
    pub id: String,
    pub pool_id: String,
    pub requester_did: String,
    pub recipient_did: String,
    pub amount: f64,
    pub justification: String,
    pub need_category: NeedCategory,
    pub status: RequestStatus,
    pub votes_for: u32,
    pub votes_against: u32,
    pub voting_deadline: Timestamp,
}

// =============================================================================
// POOL MANAGEMENT
// =============================================================================

/// Create a new commons pool
#[hdk_extern]
pub fn create_pool(input: CreatePoolInput) -> ExternResult<PoolSummary> {
    // Input validation
    if input.name.is_empty() || input.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pool name must be 1-256 characters".into()
        )));
    }
    if input.description.is_empty() || input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pool description must be 1-4096 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    if let Some(ref alias) = input.cultural_alias {
        if alias.is_empty() || alias.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cultural alias must be 1-256 characters".into()
            )));
        }
    }
    if let Some(ref gov) = input.governance {
        if let Some(threshold) = gov.allocation_threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Allocation threshold must be between 0.0 and 1.0".into()
                )));
            }
        }
        if let Some(percent) = gov.max_allocation_percent {
            if !(0.0..=1.0).contains(&percent) {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Max allocation percent must be between 0.0 and 1.0".into()
                )));
            }
        }
    }

    let now = sys_time()?;

    // Set default governance or use provided
    let governance = if let Some(gov) = input.governance {
        PoolGovernance {
            allocation_threshold: gov.allocation_threshold.unwrap_or(0.5),
            voting_period_hours: gov
                .voting_period_hours
                .unwrap_or(DEFAULT_VOTING_PERIOD_HOURS),
            contribution_weighted_voting: gov.contribution_weighted_voting.unwrap_or(false),
            min_civ_to_request: gov.min_civ_to_request,
            max_allocation_percent: gov.max_allocation_percent.unwrap_or(MAX_ALLOCATION_PERCENT),
            require_justification: gov.require_justification.unwrap_or(true),
            request_cooldown_hours: gov.request_cooldown_hours.unwrap_or(24),
        }
    } else {
        PoolGovernance {
            allocation_threshold: 0.5,
            voting_period_hours: DEFAULT_VOTING_PERIOD_HOURS,
            contribution_weighted_voting: false,
            min_civ_to_request: None,
            max_allocation_percent: MAX_ALLOCATION_PERCENT,
            require_justification: true,
            request_cooldown_hours: 24,
        }
    };

    let pool_id = format!("pool:{}:{}", input.dao_did, now.as_micros());
    let pool = CommonsPool {
        id: pool_id.clone(),
        name: input.name.clone(),
        description: input.description.clone(),
        dao_did: input.dao_did.clone(),
        resource_type: input.resource_type.clone(),
        cultural_alias: input.cultural_alias,
        total_resources: 0.0,
        lifetime_contributions: 0.0,
        lifetime_allocations: 0.0,
        contributor_count: 0,
        status: PoolStatus::Proposed,
        governance,
        created: now,
    };

    let pool_hash = create_entry(&EntryTypes::CommonsPool(pool.clone()))?;

    // Link to DAO
    create_link(
        anchor_hash(&format!("dao_pools:{}", input.dao_did))?,
        pool_hash,
        LinkTypes::DaoToPools,
        (),
    )?;

    Ok(PoolSummary {
        id: pool_id,
        name: input.name,
        description: input.description,
        dao_did: input.dao_did,
        resource_type: input.resource_type,
        total_resources: 0.0,
        contributor_count: 0,
        status: PoolStatus::Proposed,
        pending_requests: 0,
    })
}

/// Get all pools for a DAO
#[hdk_extern]
pub fn get_dao_pools(dao_did: String) -> ExternResult<Vec<PoolSummary>> {
    if dao_did.is_empty() || dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dao_pools:{}", dao_did))?,
            LinkTypes::DaoToPools,
        )?,
        GetStrategy::default(),
    )?;

    let mut pools = Vec::new();
    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(pool) = record.entry().to_app_option::<CommonsPool>().ok().flatten() {
                pools.push(pool_to_summary(&pool));
            }
        }
    }

    Ok(pools)
}

/// Get a specific pool
#[hdk_extern]
pub fn get_pool(_pool_id: String) -> ExternResult<Option<CommonsPool>> {
    if _pool_id.is_empty() || _pool_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pool ID must be 1-256 characters".into()
        )));
    }
    // In production: would use index
    Ok(None)
}

// =============================================================================
// CONTRIBUTIONS
// =============================================================================

/// Contribute to a commons pool
#[hdk_extern]
pub fn contribute(input: ContributeInput) -> ExternResult<PoolContribution> {
    // Input validation
    if input.pool_id.is_empty() || input.pool_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pool ID must be 1-256 characters".into()
        )));
    }
    if input.amount <= 0.0 || !input.amount.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount must be a positive finite number".into()
        )));
    }
    if let Some(ref message) = input.message {
        if message.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Message must not exceed 4096 characters".into()
            )));
        }
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let contributor_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    // Create contribution
    let contribution_id = format!(
        "contrib:{}:{}:{}",
        input.pool_id,
        contributor_did,
        now.as_micros()
    );
    let contribution = PoolContribution {
        id: contribution_id,
        pool_id: input.pool_id.clone(),
        contributor_did: contributor_did.clone(),
        amount: input.amount,
        message: input.message,
        anonymous: input.anonymous,
        timestamp: now,
    };

    let contrib_hash = create_entry(&EntryTypes::PoolContribution(contribution.clone()))?;

    // Link to pool
    create_link(
        anchor_hash(&format!("pool_contribs:{}", input.pool_id))?,
        contrib_hash.clone(),
        LinkTypes::PoolToContributions,
        (),
    )?;

    // Link to contributor
    create_link(
        anchor_hash(&format!("my_contribs:{}", contributor_did))?,
        contrib_hash,
        LinkTypes::ContributorToContributions,
        (),
    )?;

    // Update pool totals
    update_pool_after_contribution(&input.pool_id, input.amount, &contributor_did)?;

    Ok(contribution)
}

/// Get all contributions to a pool
#[hdk_extern]
pub fn get_pool_contributions(pool_id: String) -> ExternResult<Vec<PoolContribution>> {
    if pool_id.is_empty() || pool_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pool ID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("pool_contribs:{}", pool_id))?,
            LinkTypes::PoolToContributions,
        )?,
        GetStrategy::default(),
    )?;

    let mut contributions = Vec::new();
    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(contrib) = record
                .entry()
                .to_app_option::<PoolContribution>()
                .ok()
                .flatten()
            {
                contributions.push(contrib);
            }
        }
    }

    // Sort by timestamp (newest first)
    contributions.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(contributions)
}

/// Get my contributions
#[hdk_extern]
pub fn get_my_contributions(_: ()) -> ExternResult<Vec<PoolContribution>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let contributor_did = format!("did:mycelix:{}", caller);

    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("my_contribs:{}", contributor_did))?,
            LinkTypes::ContributorToContributions,
        )?,
        GetStrategy::default(),
    )?;

    let mut contributions = Vec::new();
    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(contrib) = record
                .entry()
                .to_app_option::<PoolContribution>()
                .ok()
                .flatten()
            {
                contributions.push(contrib);
            }
        }
    }

    Ok(contributions)
}

// =============================================================================
// ALLOCATION REQUESTS
// =============================================================================

/// Request an allocation from a pool
#[hdk_extern]
pub fn request_allocation(input: RequestAllocationInput) -> ExternResult<RequestSummary> {
    // Input validation
    if input.pool_id.is_empty() || input.pool_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pool ID must be 1-256 characters".into()
        )));
    }
    if input.amount <= 0.0 || !input.amount.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount must be a positive finite number".into()
        )));
    }
    if input.justification.is_empty() || input.justification.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Justification must be 1-4096 characters".into()
        )));
    }
    if let Some(ref recipient) = input.recipient_did {
        if recipient.is_empty() || recipient.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Recipient DID must be 1-256 characters".into()
            )));
        }
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let requester_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    // Recipient defaults to requester if not specified
    let recipient_did = input.recipient_did.unwrap_or(requester_did.clone());

    // Calculate voting deadline (default 72 hours)
    let deadline = Timestamp::from_micros(
        now.as_micros() as i64 + (DEFAULT_VOTING_PERIOD_HOURS as i64 * 3600 * 1_000_000),
    );

    let request_id = format!(
        "request:{}:{}:{}",
        input.pool_id,
        requester_did,
        now.as_micros()
    );
    let request = AllocationRequest {
        id: request_id.clone(),
        pool_id: input.pool_id.clone(),
        requester_did: requester_did.clone(),
        recipient_did: recipient_did.clone(),
        amount: input.amount,
        justification: input.justification.clone(),
        need_category: input.need_category.clone(),
        status: RequestStatus::Voting,
        voting_deadline: deadline,
        votes_for: 0,
        votes_against: 0,
        total_vote_weight: 0.0,
        created: now,
        resolved: None,
    };

    let request_hash = create_entry(&EntryTypes::AllocationRequest(request.clone()))?;

    // Link to pool
    create_link(
        anchor_hash(&format!("pool_requests:{}", input.pool_id))?,
        request_hash.clone(),
        LinkTypes::PoolToRequests,
        (),
    )?;

    // Link to requester
    create_link(
        anchor_hash(&format!("my_requests:{}", requester_did))?,
        request_hash,
        LinkTypes::MemberToRequests,
        (),
    )?;

    Ok(RequestSummary {
        id: request_id,
        pool_id: input.pool_id,
        requester_did,
        recipient_did,
        amount: input.amount,
        justification: input.justification,
        need_category: input.need_category,
        status: RequestStatus::Voting,
        votes_for: 0,
        votes_against: 0,
        voting_deadline: deadline,
    })
}

/// Get all pending requests for a pool
#[hdk_extern]
pub fn get_pool_requests(pool_id: String) -> ExternResult<Vec<RequestSummary>> {
    if pool_id.is_empty() || pool_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pool ID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("pool_requests:{}", pool_id))?,
            LinkTypes::PoolToRequests,
        )?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(request) = record
                .entry()
                .to_app_option::<AllocationRequest>()
                .ok()
                .flatten()
            {
                requests.push(request_to_summary(&request));
            }
        }
    }

    Ok(requests)
}

// =============================================================================
// VOTING
// =============================================================================

/// Vote on an allocation request
#[hdk_extern]
pub fn vote_on_request(input: VoteInput) -> ExternResult<AllocationVote> {
    // Input validation
    if input.request_id.is_empty() || input.request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    if let Some(ref comment) = input.comment {
        if comment.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Comment must not exceed 4096 characters".into()
            )));
        }
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let voter_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    // Default vote weight = 1.0 (can be weighted by contribution in future)
    let weight = 1.0;

    let vote = AllocationVote {
        request_id: input.request_id.clone(),
        voter_did: voter_did.clone(),
        vote: input.vote,
        weight,
        comment: input.comment,
        timestamp: now,
    };

    let vote_hash = create_entry(&EntryTypes::AllocationVote(vote.clone()))?;

    // Link to request
    create_link(
        anchor_hash(&format!("request_votes:{}", input.request_id))?,
        vote_hash,
        LinkTypes::RequestToVotes,
        (),
    )?;

    // Update request tally
    update_request_after_vote(&input.request_id, input.vote, weight)?;

    Ok(vote)
}

/// Get votes for a request
#[hdk_extern]
pub fn get_request_votes(request_id: String) -> ExternResult<Vec<AllocationVote>> {
    if request_id.is_empty() || request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("request_votes:{}", request_id))?,
            LinkTypes::RequestToVotes,
        )?,
        GetStrategy::default(),
    )?;

    let mut votes = Vec::new();
    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(vote) = record
                .entry()
                .to_app_option::<AllocationVote>()
                .ok()
                .flatten()
            {
                votes.push(vote);
            }
        }
    }

    Ok(votes)
}

/// Finalize a request (check if voting period ended and apply result)
#[hdk_extern]
pub fn finalize_request(_request_id: String) -> ExternResult<RequestSummary> {
    if _request_id.is_empty() || _request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    // In production: check deadline, calculate result, update status, disburse if approved
    // This is a placeholder
    Err(wasm_error!(WasmErrorInner::Guest(
        "Not yet implemented".into()
    )))
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn anchor_hash(anchor: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(anchor.to_string()))
}

fn pool_to_summary(pool: &CommonsPool) -> PoolSummary {
    PoolSummary {
        id: pool.id.clone(),
        name: pool.name.clone(),
        description: pool.description.clone(),
        dao_did: pool.dao_did.clone(),
        resource_type: pool.resource_type.clone(),
        total_resources: pool.total_resources,
        contributor_count: pool.contributor_count,
        status: pool.status.clone(),
        pending_requests: 0, // Would need to query
    }
}

fn request_to_summary(request: &AllocationRequest) -> RequestSummary {
    RequestSummary {
        id: request.id.clone(),
        pool_id: request.pool_id.clone(),
        requester_did: request.requester_did.clone(),
        recipient_did: request.recipient_did.clone(),
        amount: request.amount,
        justification: request.justification.clone(),
        need_category: request.need_category.clone(),
        status: request.status.clone(),
        votes_for: request.votes_for,
        votes_against: request.votes_against,
        voting_deadline: request.voting_deadline,
    }
}

fn update_pool_after_contribution(
    _pool_id: &str,
    _amount: f64,
    _contributor_did: &str,
) -> ExternResult<()> {
    // In production: find pool entry and update totals
    // This is a placeholder
    Ok(())
}

fn update_request_after_vote(_request_id: &str, _vote_for: bool, _weight: f64) -> ExternResult<()> {
    // In production: find request entry and update vote tally
    // This is a placeholder
    Ok(())
}

// =============================================================================
// EXPORTS FOR OTHER ZOMES
// =============================================================================

/// Get commons contribution for reputation calculation
#[hdk_extern]
pub fn get_commons_reputation_input(_member_did: String) -> ExternResult<f32> {
    if _member_did.is_empty() || _member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    let contributions = get_my_contributions(())?;

    // Normalize based on total contributed (across all pools)
    let total: f64 = contributions.iter().map(|c| c.amount).sum();

    // Cap at 100 units for max score
    let normalized = (total / 100.0).min(1.0) as f32;

    // Commons contribution could factor into reputation
    // But per constitutional principles, should be optional and capped
    Ok(normalized * 0.05) // 5% max weight
}
