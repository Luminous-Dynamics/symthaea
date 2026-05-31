// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Pools Coordinator Zome - Mutual aid pool management
//!
//! This zome provides the coordination logic for creating and managing
//! mutual aid pools, contributions, and disbursements.

use hdk::prelude::*;
use pools_integrity::{
    Anchor as PoolsAnchor, Contribution, ContributionRule, Disbursement, DisbursementRule,
    DisbursementStatus, EntryTypes, LinkTypes, MemberRole, MutualAidPool, PoolMembership,
    PoolStatus,
};

/// Input for creating a new pool
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreatePoolInput {
    pub name: String,
    pub description: String,
    pub creator_did: String,
    pub contribution_rules: Option<ContributionRule>,
    pub disbursement_rules: Option<DisbursementRule>,
}

/// Input for updating pool rules
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdatePoolRulesInput {
    pub pool_hash: ActionHash,
    pub contribution_rules: Option<ContributionRule>,
    pub disbursement_rules: Option<DisbursementRule>,
}

/// Input for adding a member to a pool
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AddMemberInput {
    pub pool_hash: ActionHash,
    pub pool_id: String,
    pub member_did: String,
    pub role: MemberRole,
}

/// Input for making a contribution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContributeInput {
    pub pool_hash: ActionHash,
    pub pool_id: String,
    pub member_did: String,
    pub amount: u64,
    pub note: Option<String>,
}

/// Input for requesting a disbursement
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestDisbursementInput {
    pub pool_hash: ActionHash,
    pub pool_id: String,
    pub recipient_did: String,
    pub amount: u64,
    pub reason: String,
    pub is_emergency: bool,
}

/// Input for voting on a disbursement
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoteDisbursementInput {
    pub disbursement_hash: ActionHash,
    pub voter_did: String,
    pub approve: bool,
}

/// Input for updating pool status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdatePoolStatusInput {
    pub pool_hash: ActionHash,
    pub status: PoolStatus,
}

/// Output containing a pool with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolWithHash {
    pub hash: ActionHash,
    pub pool: MutualAidPool,
}

/// Output containing a contribution with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContributionWithHash {
    pub hash: ActionHash,
    pub contribution: Contribution,
}

/// Output containing a disbursement with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisbursementWithHash {
    pub hash: ActionHash,
    pub disbursement: Disbursement,
}

/// Output containing a membership with its action hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MembershipWithHash {
    pub hash: ActionHash,
    pub membership: PoolMembership,
}

/// Pool statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolStats {
    pub total_members: u32,
    pub total_contributed: u64,
    pub total_disbursed: u64,
    pub pending_disbursements: u32,
    pub current_balance: u64,
}

/// Generate a unique ID based on timestamp and agent
fn generate_id(prefix: &str) -> ExternResult<String> {
    let now = sys_time()?;
    let agent = agent_info()?.agent_initial_pubkey;
    // Create unique ID from timestamp and agent pubkey truncated hash
    let agent_str = format!("{:?}", agent);
    let short_hash = &agent_str[agent_str.len().saturating_sub(8)..];
    Ok(format!(
        "{}_{:x}_{}",
        prefix,
        now.as_micros() as u64,
        short_hash
    ))
}

/// Get or create the main pools anchor
fn get_pools_anchor() -> ExternResult<EntryHash> {
    let anchor = PoolsAnchor::new("all_pools");
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create a pool-specific anchor
fn get_pool_anchor(pool_id: &str) -> ExternResult<EntryHash> {
    let anchor = PoolsAnchor::new(format!("pool_{}", pool_id));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Get or create a member-specific anchor
fn get_member_anchor(did: &str) -> ExternResult<EntryHash> {
    let anchor = PoolsAnchor::new(format!("member_{}", did));
    let entry_hash = hash_entry(&anchor)?;
    create_entry(EntryTypes::Anchor(anchor))?;
    Ok(entry_hash)
}

/// Create a new mutual aid pool
#[hdk_extern]
pub fn create_pool(input: CreatePoolInput) -> ExternResult<PoolWithHash> {
    let now = sys_time()?;
    let id = generate_id("pool")?;

    let pool = MutualAidPool {
        id: id.clone(),
        name: input.name,
        description: input.description,
        members: vec![input.creator_did.clone()],
        contribution_rules: input.contribution_rules.unwrap_or_default(),
        disbursement_rules: input.disbursement_rules.unwrap_or_default(),
        balance: 0,
        status: PoolStatus::Active,
        created_at: now,
        updated_at: now,
    };

    // Create the pool entry
    let pool_hash = create_entry(EntryTypes::MutualAidPool(pool.clone()))?;

    // Link from main anchor
    let pools_anchor = get_pools_anchor()?;
    create_link(
        pools_anchor,
        pool_hash.clone(),
        LinkTypes::AnchorToPool,
        id.as_bytes().to_vec(),
    )?;

    // Create membership for creator (as admin)
    let membership = PoolMembership {
        pool_id: id.clone(),
        member_did: input.creator_did.clone(),
        role: MemberRole::Admin,
        joined_at: now,
        total_contributed: 0,
        total_received: 0,
        last_contribution: None,
        last_disbursement: None,
    };
    let membership_hash = create_entry(EntryTypes::PoolMembership(membership))?;

    // Link pool to membership
    create_link(
        pool_hash.clone(),
        membership_hash.clone(),
        LinkTypes::PoolToMembership,
        (),
    )?;

    // Link member to membership
    let member_anchor = get_member_anchor(&input.creator_did)?;
    create_link(
        member_anchor,
        membership_hash,
        LinkTypes::MemberToMembership,
        (),
    )?;

    Ok(PoolWithHash {
        hash: pool_hash,
        pool,
    })
}

/// Get a pool by its action hash
#[hdk_extern]
pub fn get_pool(action_hash: ActionHash) -> ExternResult<Option<PoolWithHash>> {
    match get(action_hash.clone(), GetOptions::default())? {
        Some(record) => {
            let pool: MutualAidPool = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("No MutualAidPool found".to_string()))
                })?;
            Ok(Some(PoolWithHash {
                hash: action_hash,
                pool,
            }))
        }
        None => Ok(None),
    }
}

/// Get all pools
#[hdk_extern]
pub fn get_all_pools(_: ()) -> ExternResult<Vec<PoolWithHash>> {
    let pools_anchor = get_pools_anchor()?;
    let links = get_links(
        LinkQuery::try_new(pools_anchor, LinkTypes::AnchorToPool)?,
        GetStrategy::default(),
    )?;

    let mut pools = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(pool_with_hash) = get_pool(action_hash)? {
                // Only include active pools
                if pool_with_hash.pool.status == PoolStatus::Active {
                    pools.push(pool_with_hash);
                }
            }
        }
    }

    Ok(pools)
}

/// Add a member to a pool
#[hdk_extern]
pub fn add_member(input: AddMemberInput) -> ExternResult<MembershipWithHash> {
    let now = sys_time()?;

    // Get the current pool
    let record = get(input.pool_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Pool not found".to_string())))?;

    let mut pool: MutualAidPool = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No MutualAidPool found".to_string())))?;

    // Check if member is already in pool
    if pool.members.contains(&input.member_did) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member already in pool".to_string()
        )));
    }

    // Add member to pool
    pool.members.push(input.member_did.clone());
    pool.updated_at = now;

    // Update pool entry
    let new_pool_hash = update_entry(input.pool_hash.clone(), EntryTypes::MutualAidPool(pool))?;

    // Create membership
    let membership = PoolMembership {
        pool_id: input.pool_id,
        member_did: input.member_did.clone(),
        role: input.role,
        joined_at: now,
        total_contributed: 0,
        total_received: 0,
        last_contribution: None,
        last_disbursement: None,
    };
    let membership_hash = create_entry(EntryTypes::PoolMembership(membership.clone()))?;

    // Link pool to membership
    create_link(
        new_pool_hash,
        membership_hash.clone(),
        LinkTypes::PoolToMembership,
        (),
    )?;

    // Link member to membership
    let member_anchor = get_member_anchor(&input.member_did)?;
    create_link(
        member_anchor,
        membership_hash.clone(),
        LinkTypes::MemberToMembership,
        (),
    )?;

    Ok(MembershipWithHash {
        hash: membership_hash,
        membership,
    })
}

/// Make a contribution to a pool
#[hdk_extern]
pub fn contribute(input: ContributeInput) -> ExternResult<ContributionWithHash> {
    let now = sys_time()?;
    let id = generate_id("contrib")?;

    // Get the current pool
    let record = get(input.pool_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Pool not found".to_string())))?;

    let mut pool: MutualAidPool = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No MutualAidPool found".to_string())))?;

    // Verify member is in pool
    if !pool.members.contains(&input.member_did) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Not a member of this pool".to_string()
        )));
    }

    // Update pool balance
    pool.balance = pool.balance.saturating_add(input.amount);
    pool.updated_at = now;

    // Update pool entry
    let new_pool_hash = update_entry(input.pool_hash.clone(), EntryTypes::MutualAidPool(pool))?;

    // Create contribution
    let contribution = Contribution {
        id: id.clone(),
        pool_id: input.pool_id,
        member_did: input.member_did.clone(),
        amount: input.amount,
        note: input.note,
        timestamp: now,
    };
    let contribution_hash = create_entry(EntryTypes::Contribution(contribution.clone()))?;

    // Link pool to contribution
    create_link(
        new_pool_hash,
        contribution_hash.clone(),
        LinkTypes::PoolToContribution,
        id.as_bytes().to_vec(),
    )?;

    // Link member to contribution
    let member_anchor = get_member_anchor(&input.member_did)?;
    create_link(
        member_anchor,
        contribution_hash.clone(),
        LinkTypes::MemberToContribution,
        (),
    )?;

    Ok(ContributionWithHash {
        hash: contribution_hash,
        contribution,
    })
}

/// Request a disbursement from a pool
#[hdk_extern]
pub fn request_disbursement(input: RequestDisbursementInput) -> ExternResult<DisbursementWithHash> {
    let now = sys_time()?;
    let id = generate_id("disb")?;

    // Get the current pool
    let record = get(input.pool_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Pool not found".to_string())))?;

    let pool: MutualAidPool = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No MutualAidPool found".to_string())))?;

    // Verify recipient is in pool
    if !pool.members.contains(&input.recipient_did) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Not a member of this pool".to_string()
        )));
    }

    // Check if amount exceeds balance
    if input.amount > pool.balance {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient pool balance".to_string()
        )));
    }

    // Check max disbursement rule
    if input.amount > pool.disbursement_rules.max_disbursement {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount exceeds maximum disbursement".to_string()
        )));
    }

    // Create disbursement request
    let disbursement = Disbursement {
        id: id.clone(),
        pool_id: input.pool_id,
        recipient_did: input.recipient_did.clone(),
        amount: input.amount,
        reason: input.reason,
        approved_by: Vec::new(),
        rejected_by: Vec::new(),
        status: DisbursementStatus::Pending,
        is_emergency: input.is_emergency,
        requested_at: now,
        processed_at: None,
    };
    let disbursement_hash = create_entry(EntryTypes::Disbursement(disbursement.clone()))?;

    // Link pool to disbursement
    create_link(
        input.pool_hash.clone(),
        disbursement_hash.clone(),
        LinkTypes::PoolToDisbursement,
        id.as_bytes().to_vec(),
    )?;

    // Link to pending disbursements
    create_link(
        input.pool_hash,
        disbursement_hash.clone(),
        LinkTypes::PoolToPendingDisbursement,
        (),
    )?;

    // Link member to disbursement
    let member_anchor = get_member_anchor(&input.recipient_did)?;
    create_link(
        member_anchor,
        disbursement_hash.clone(),
        LinkTypes::MemberToDisbursement,
        (),
    )?;

    Ok(DisbursementWithHash {
        hash: disbursement_hash,
        disbursement,
    })
}

/// Vote on a disbursement request
#[hdk_extern]
pub fn vote_disbursement(input: VoteDisbursementInput) -> ExternResult<DisbursementWithHash> {
    let now = sys_time()?;

    // Get the current disbursement
    let record = get(input.disbursement_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Disbursement not found".to_string())))?;

    let mut disbursement: Disbursement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No Disbursement found".to_string())))?;

    // Check if already voted
    if disbursement.approved_by.contains(&input.voter_did)
        || disbursement.rejected_by.contains(&input.voter_did)
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Already voted on this disbursement".to_string()
        )));
    }

    // Check if still pending
    if disbursement.status != DisbursementStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Disbursement is no longer pending".to_string()
        )));
    }

    // Record vote
    if input.approve {
        disbursement.approved_by.push(input.voter_did);
    } else {
        disbursement.rejected_by.push(input.voter_did);
    }

    // Update entry
    let new_hash = update_entry(
        input.disbursement_hash.clone(),
        EntryTypes::Disbursement(disbursement.clone()),
    )?;

    Ok(DisbursementWithHash {
        hash: new_hash,
        disbursement,
    })
}

/// Process a disbursement (approve and transfer funds)
#[hdk_extern]
pub fn process_disbursement(disbursement_hash: ActionHash) -> ExternResult<DisbursementWithHash> {
    let now = sys_time()?;

    // Get the disbursement
    let disb_record = get(disbursement_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Disbursement not found".to_string())))?;

    let mut disbursement: Disbursement = disb_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No Disbursement found".to_string())))?;

    // Must be pending
    if disbursement.status != DisbursementStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Disbursement is not pending".to_string()
        )));
    }

    // Note: In a full implementation, we would:
    // 1. Get the pool and check approval threshold
    // 2. Deduct from pool balance
    // 3. Update membership records
    // For now, just mark as completed

    disbursement.status = DisbursementStatus::Completed;
    disbursement.processed_at = Some(now);

    let new_hash = update_entry(
        disbursement_hash,
        EntryTypes::Disbursement(disbursement.clone()),
    )?;

    Ok(DisbursementWithHash {
        hash: new_hash,
        disbursement,
    })
}

/// Get all contributions for a pool
#[hdk_extern]
pub fn get_pool_contributions(pool_hash: ActionHash) -> ExternResult<Vec<ContributionWithHash>> {
    let links = get_links(
        LinkQuery::try_new(pool_hash, LinkTypes::PoolToContribution)?,
        GetStrategy::default(),
    )?;

    let mut contributions = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Ok(Some(contribution)) = record.entry().to_app_option::<Contribution>() {
                    contributions.push(ContributionWithHash {
                        hash: action_hash,
                        contribution,
                    });
                }
            }
        }
    }

    Ok(contributions)
}

/// Get all disbursements for a pool
#[hdk_extern]
pub fn get_pool_disbursements(pool_hash: ActionHash) -> ExternResult<Vec<DisbursementWithHash>> {
    let links = get_links(
        LinkQuery::try_new(pool_hash, LinkTypes::PoolToDisbursement)?,
        GetStrategy::default(),
    )?;

    let mut disbursements = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Ok(Some(disbursement)) = record.entry().to_app_option::<Disbursement>() {
                    disbursements.push(DisbursementWithHash {
                        hash: action_hash,
                        disbursement,
                    });
                }
            }
        }
    }

    Ok(disbursements)
}

/// Get pending disbursements for a pool
#[hdk_extern]
pub fn get_pending_disbursements(pool_hash: ActionHash) -> ExternResult<Vec<DisbursementWithHash>> {
    let links = get_links(
        LinkQuery::try_new(pool_hash, LinkTypes::PoolToPendingDisbursement)?,
        GetStrategy::default(),
    )?;

    let mut disbursements = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Ok(Some(disbursement)) = record.entry().to_app_option::<Disbursement>() {
                    if disbursement.status == DisbursementStatus::Pending {
                        disbursements.push(DisbursementWithHash {
                            hash: action_hash,
                            disbursement,
                        });
                    }
                }
            }
        }
    }

    Ok(disbursements)
}

/// Get pools a member belongs to
#[hdk_extern]
pub fn get_member_pools(member_did: String) -> ExternResult<Vec<PoolWithHash>> {
    let member_anchor = get_member_anchor(&member_did)?;
    let links = get_links(
        LinkQuery::try_new(member_anchor, LinkTypes::MemberToMembership)?,
        GetStrategy::default(),
    )?;

    let mut pools = Vec::new();
    let mut seen_pools = std::collections::HashSet::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Ok(Some(membership)) = record.entry().to_app_option::<PoolMembership>() {
                    // Avoid duplicates
                    if !seen_pools.contains(&membership.pool_id) {
                        seen_pools.insert(membership.pool_id.clone());

                        // Find the pool by ID
                        let all_pools = get_all_pools(())?;
                        for pool_with_hash in all_pools {
                            if pool_with_hash.pool.id == membership.pool_id {
                                pools.push(pool_with_hash);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(pools)
}

/// Get pool statistics
#[hdk_extern]
pub fn get_pool_stats(pool_hash: ActionHash) -> ExternResult<PoolStats> {
    let pool = get_pool(pool_hash.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Pool not found".to_string())))?;

    let contributions = get_pool_contributions(pool_hash.clone())?;
    let disbursements = get_pool_disbursements(pool_hash.clone())?;
    let pending = get_pending_disbursements(pool_hash)?;

    let total_contributed: u64 = contributions.iter().map(|c| c.contribution.amount).sum();
    let total_disbursed: u64 = disbursements
        .iter()
        .filter(|d| d.disbursement.status == DisbursementStatus::Completed)
        .map(|d| d.disbursement.amount)
        .sum();

    Ok(PoolStats {
        total_members: pool.pool.members.len() as u32,
        total_contributed,
        total_disbursed,
        pending_disbursements: pending.len() as u32,
        current_balance: pool.pool.balance,
    })
}

/// Update pool status (pause, close, reactivate)
#[hdk_extern]
pub fn update_pool_status(input: UpdatePoolStatusInput) -> ExternResult<PoolWithHash> {
    let now = sys_time()?;

    let record = get(input.pool_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Pool not found".to_string())))?;

    let mut pool: MutualAidPool = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No MutualAidPool found".to_string())))?;

    pool.status = input.status;
    pool.updated_at = now;

    let new_hash = update_entry(input.pool_hash, EntryTypes::MutualAidPool(pool.clone()))?;

    Ok(PoolWithHash {
        hash: new_hash,
        pool,
    })
}
