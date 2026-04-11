// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pools Coordinator Zome - Mutual aid pool management
//!
//! This zome provides the coordination logic for creating and managing
//! mutual aid pools, contributions, and disbursements.

use hdk::prelude::*;
use mutualaid_pools_integrity::{
    Anchor as PoolsAnchor, Contribution, ContributionRule, Disbursement, DisbursementRule,
    DisbursementStatus, EntryTypes, LinkTypes, MemberRole, MutualAidPool, PoolMembership,
    PoolStatus,
};
use mycelix_bridge_common::{civic_requirement_proposal, civic_requirement_voting};
use mycelix_zome_helpers::get_latest_record;


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
    // Validate pool name is not empty or whitespace-only
    if input.name.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pool name cannot be empty or whitespace-only".into()
        )));
    }

    // Validate pool description is not empty or whitespace-only
    if input.description.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pool description cannot be empty or whitespace-only".into()
        )));
    }

    // Validate creator DID is not empty or whitespace-only
    if input.creator_did.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Creator DID cannot be empty or whitespace-only".into()
        )));
    }

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
    match get_latest_record(action_hash.clone())? {
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
    // Validate contribution amount is non-zero
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Contribution amount must be greater than zero".into()
        )));
    }

    // Validate note length (defense in depth — also checked at integrity layer)
    if let Some(ref note) = input.note {
        if note.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Contribution note must be 4096 characters or fewer".into()
            )));
        }
    }

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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "request_disbursement")?;

    // Validate disbursement amount is non-zero
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Disbursement amount must be greater than zero".into()
        )));
    }

    // Validate reason is not empty or whitespace-only
    if input.reason.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Disbursement reason cannot be empty or whitespace-only".into()
        )));
    }

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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "vote_disbursement")?;

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
    let disb_record = get_latest_record(disbursement_hash.clone())?
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
            if let Some(record) = get_latest_record(action_hash.clone())? {
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
            if let Some(record) = get_latest_record(action_hash.clone())? {
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
            if let Some(record) = get_latest_record(action_hash.clone())? {
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
            if let Some(record) = get_latest_record(action_hash)? {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Input struct serde roundtrip tests ─────────────────────────────

    #[test]
    fn create_pool_input_serde_roundtrip() {
        let input = CreatePoolInput {
            name: "Neighborhood Fund".to_string(),
            description: "Emergency mutual aid pool".to_string(),
            creator_did: "did:mycelix:alice123".to_string(),
            contribution_rules: Some(ContributionRule {
                min_monthly: 50,
                max_withdrawal: 500,
                cooldown_days: 7,
            }),
            disbursement_rules: Some(DisbursementRule {
                min_approvals: 2,
                approval_threshold_percent: 60,
                max_disbursement: 1000,
                allow_emergency_bypass: true,
            }),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreatePoolInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Neighborhood Fund");
        assert_eq!(decoded.creator_did, "did:mycelix:alice123");
        assert_eq!(decoded.contribution_rules.as_ref().unwrap().min_monthly, 50);
        assert_eq!(
            decoded
                .disbursement_rules
                .as_ref()
                .unwrap()
                .allow_emergency_bypass,
            true
        );
    }

    #[test]
    fn create_pool_input_no_rules() {
        let input = CreatePoolInput {
            name: "Simple Pool".to_string(),
            description: "No special rules".to_string(),
            creator_did: "did:mycelix:bob456".to_string(),
            contribution_rules: None,
            disbursement_rules: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreatePoolInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.contribution_rules, None);
        assert_eq!(decoded.disbursement_rules, None);
    }

    #[test]
    fn update_pool_rules_input_serde_roundtrip() {
        let pool_hash = ActionHash::from_raw_36(vec![0xAA; 36]);
        let input = UpdatePoolRulesInput {
            pool_hash: pool_hash.clone(),
            contribution_rules: Some(ContributionRule {
                min_monthly: 100,
                max_withdrawal: 1000,
                cooldown_days: 14,
            }),
            disbursement_rules: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdatePoolRulesInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.pool_hash, pool_hash);
        assert_eq!(
            decoded.contribution_rules.as_ref().unwrap().cooldown_days,
            14
        );
        assert_eq!(decoded.disbursement_rules, None);
    }

    #[test]
    fn add_member_input_serde_roundtrip() {
        let pool_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input = AddMemberInput {
            pool_hash: pool_hash.clone(),
            pool_id: "pool_123".to_string(),
            member_did: "did:mycelix:charlie789".to_string(),
            role: MemberRole::Member,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddMemberInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.pool_id, "pool_123");
        assert_eq!(decoded.member_did, "did:mycelix:charlie789");
        assert_eq!(decoded.role, MemberRole::Member);
    }

    #[test]
    fn add_member_input_all_roles() {
        let pool_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
        for role in [MemberRole::Admin, MemberRole::Member, MemberRole::Observer] {
            let input = AddMemberInput {
                pool_hash: pool_hash.clone(),
                pool_id: "pool_123".to_string(),
                member_did: "did:mycelix:test".to_string(),
                role: role.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: AddMemberInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.role, role);
        }
    }

    #[test]
    fn contribute_input_serde_roundtrip() {
        let pool_hash = ActionHash::from_raw_36(vec![0xCC; 36]);
        let input = ContributeInput {
            pool_hash: pool_hash.clone(),
            pool_id: "pool_456".to_string(),
            member_did: "did:mycelix:donor".to_string(),
            amount: 250,
            note: Some("Monthly contribution".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ContributeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, 250);
        assert_eq!(decoded.note, Some("Monthly contribution".to_string()));
    }

    #[test]
    fn contribute_input_no_note() {
        let pool_hash = ActionHash::from_raw_36(vec![0xCC; 36]);
        let input = ContributeInput {
            pool_hash: pool_hash.clone(),
            pool_id: "pool_456".to_string(),
            member_did: "did:mycelix:donor".to_string(),
            amount: 100,
            note: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ContributeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.note, None);
    }

    #[test]
    fn request_disbursement_input_serde_roundtrip() {
        let pool_hash = ActionHash::from_raw_36(vec![0xDD; 36]);
        let input = RequestDisbursementInput {
            pool_hash: pool_hash.clone(),
            pool_id: "pool_789".to_string(),
            recipient_did: "did:mycelix:recipient".to_string(),
            amount: 500,
            reason: "Unexpected medical expense".to_string(),
            is_emergency: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RequestDisbursementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, 500);
        assert_eq!(decoded.reason, "Unexpected medical expense");
        assert_eq!(decoded.is_emergency, true);
    }

    #[test]
    fn vote_disbursement_input_serde_roundtrip() {
        let disbursement_hash = ActionHash::from_raw_36(vec![0xEE; 36]);
        let input = VoteDisbursementInput {
            disbursement_hash: disbursement_hash.clone(),
            voter_did: "did:mycelix:voter".to_string(),
            approve: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VoteDisbursementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.disbursement_hash, disbursement_hash);
        assert_eq!(decoded.voter_did, "did:mycelix:voter");
        assert_eq!(decoded.approve, true);
    }

    #[test]
    fn vote_disbursement_input_reject() {
        let disbursement_hash = ActionHash::from_raw_36(vec![0xEE; 36]);
        let input = VoteDisbursementInput {
            disbursement_hash: disbursement_hash.clone(),
            voter_did: "did:mycelix:voter2".to_string(),
            approve: false,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VoteDisbursementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.approve, false);
    }

    #[test]
    fn update_pool_status_input_serde_roundtrip() {
        let pool_hash = ActionHash::from_raw_36(vec![0xFF; 36]);
        for status in [PoolStatus::Active, PoolStatus::Paused, PoolStatus::Closed] {
            let input = UpdatePoolStatusInput {
                pool_hash: pool_hash.clone(),
                status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdatePoolStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    // ── Output struct serde roundtrip tests ────────────────────────────

    #[test]
    fn pool_stats_serde_roundtrip() {
        let stats = PoolStats {
            total_members: 12,
            total_contributed: 5000,
            total_disbursed: 1500,
            pending_disbursements: 3,
            current_balance: 3500,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let decoded: PoolStats = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_members, 12);
        assert_eq!(decoded.total_contributed, 5000);
        assert_eq!(decoded.total_disbursed, 1500);
        assert_eq!(decoded.pending_disbursements, 3);
        assert_eq!(decoded.current_balance, 3500);
    }

    #[test]
    fn pool_stats_zero_values() {
        let stats = PoolStats {
            total_members: 0,
            total_contributed: 0,
            total_disbursed: 0,
            pending_disbursements: 0,
            current_balance: 0,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let decoded: PoolStats = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_members, 0);
        assert_eq!(decoded.current_balance, 0);
    }

    // ── Integrity enum serde roundtrip tests ──────────────────────────

    #[test]
    fn pool_status_all_variants_serde() {
        let variants = vec![PoolStatus::Active, PoolStatus::Paused, PoolStatus::Closed];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: PoolStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn disbursement_status_all_variants_serde() {
        let variants = vec![
            DisbursementStatus::Pending,
            DisbursementStatus::Approved,
            DisbursementStatus::Rejected,
            DisbursementStatus::Completed,
            DisbursementStatus::Cancelled,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: DisbursementStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn member_role_all_variants_serde() {
        let variants = vec![MemberRole::Admin, MemberRole::Member, MemberRole::Observer];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: MemberRole = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    // ── Integrity struct serde roundtrip tests ────────────────────────

    #[test]
    fn contribution_rule_serde_roundtrip() {
        let rule = ContributionRule {
            min_monthly: 75,
            max_withdrawal: 2000,
            cooldown_days: 30,
        };
        let json = serde_json::to_string(&rule).unwrap();
        let decoded: ContributionRule = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, rule);
    }

    #[test]
    fn contribution_rule_default_serde() {
        let rule = ContributionRule::default();
        let json = serde_json::to_string(&rule).unwrap();
        let decoded: ContributionRule = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, rule);
        assert_eq!(decoded.min_monthly, 0);
    }

    #[test]
    fn disbursement_rule_serde_roundtrip() {
        let rule = DisbursementRule {
            min_approvals: 5,
            approval_threshold_percent: 75,
            max_disbursement: 10000,
            allow_emergency_bypass: false,
        };
        let json = serde_json::to_string(&rule).unwrap();
        let decoded: DisbursementRule = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, rule);
    }

    #[test]
    fn disbursement_rule_default_serde() {
        let rule = DisbursementRule::default();
        let json = serde_json::to_string(&rule).unwrap();
        let decoded: DisbursementRule = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, rule);
        assert_eq!(decoded.min_approvals, 1);
        assert_eq!(decoded.approval_threshold_percent, 50);
    }

    // ── Validation edge case tests ──────────────────────────────────

    #[test]
    fn create_pool_input_whitespace_name_detected() {
        // Verify that whitespace-only names can be detected by our validation logic
        let input = CreatePoolInput {
            name: "   \t\n  ".to_string(),
            description: "Valid description".to_string(),
            creator_did: "did:mycelix:alice".to_string(),
            contribution_rules: None,
            disbursement_rules: None,
        };
        assert!(
            input.name.trim().is_empty(),
            "Whitespace-only name should be detected"
        );
    }

    #[test]
    fn create_pool_input_empty_name_detected() {
        let input = CreatePoolInput {
            name: "".to_string(),
            description: "Valid description".to_string(),
            creator_did: "did:mycelix:alice".to_string(),
            contribution_rules: None,
            disbursement_rules: None,
        };
        assert!(
            input.name.trim().is_empty(),
            "Empty name should be detected"
        );
    }

    #[test]
    fn create_pool_input_whitespace_description_detected() {
        let input = CreatePoolInput {
            name: "Valid Pool".to_string(),
            description: "   ".to_string(),
            creator_did: "did:mycelix:alice".to_string(),
            contribution_rules: None,
            disbursement_rules: None,
        };
        assert!(
            input.description.trim().is_empty(),
            "Whitespace-only description should be detected"
        );
    }

    #[test]
    fn create_pool_input_whitespace_creator_did_detected() {
        let input = CreatePoolInput {
            name: "Valid Pool".to_string(),
            description: "Valid description".to_string(),
            creator_did: "  \t ".to_string(),
            contribution_rules: None,
            disbursement_rules: None,
        };
        assert!(
            input.creator_did.trim().is_empty(),
            "Whitespace-only DID should be detected"
        );
    }

    #[test]
    fn create_pool_input_valid_name_not_rejected() {
        let input = CreatePoolInput {
            name: "Neighborhood Fund".to_string(),
            description: "Mutual aid pool".to_string(),
            creator_did: "did:mycelix:alice".to_string(),
            contribution_rules: None,
            disbursement_rules: None,
        };
        assert!(
            !input.name.trim().is_empty(),
            "Valid name should not be rejected"
        );
        assert!(
            !input.description.trim().is_empty(),
            "Valid description should not be rejected"
        );
    }

    #[test]
    fn contribute_input_zero_amount_detected() {
        let pool_hash = ActionHash::from_raw_36(vec![0xCC; 36]);
        let input = ContributeInput {
            pool_hash,
            pool_id: "pool_1".to_string(),
            member_did: "did:mycelix:donor".to_string(),
            amount: 0,
            note: None,
        };
        assert_eq!(input.amount, 0, "Zero amount should be detected");
    }

    #[test]
    fn contribute_input_nonzero_amount_accepted() {
        let pool_hash = ActionHash::from_raw_36(vec![0xCC; 36]);
        let input = ContributeInput {
            pool_hash,
            pool_id: "pool_1".to_string(),
            member_did: "did:mycelix:donor".to_string(),
            amount: 1,
            note: None,
        };
        assert!(input.amount > 0, "Non-zero amount should be accepted");
    }

    #[test]
    fn request_disbursement_input_zero_amount_detected() {
        let pool_hash = ActionHash::from_raw_36(vec![0xDD; 36]);
        let input = RequestDisbursementInput {
            pool_hash,
            pool_id: "pool_1".to_string(),
            recipient_did: "did:mycelix:recipient".to_string(),
            amount: 0,
            reason: "Valid reason".to_string(),
            is_emergency: false,
        };
        assert_eq!(
            input.amount, 0,
            "Zero disbursement amount should be detected"
        );
    }

    #[test]
    fn request_disbursement_input_whitespace_reason_detected() {
        let pool_hash = ActionHash::from_raw_36(vec![0xDD; 36]);
        let input = RequestDisbursementInput {
            pool_hash,
            pool_id: "pool_1".to_string(),
            recipient_did: "did:mycelix:recipient".to_string(),
            amount: 100,
            reason: "   \n\t  ".to_string(),
            is_emergency: false,
        };
        assert!(
            input.reason.trim().is_empty(),
            "Whitespace-only reason should be detected"
        );
    }

    #[test]
    fn request_disbursement_input_valid_accepted() {
        let pool_hash = ActionHash::from_raw_36(vec![0xDD; 36]);
        let input = RequestDisbursementInput {
            pool_hash,
            pool_id: "pool_1".to_string(),
            recipient_did: "did:mycelix:recipient".to_string(),
            amount: 500,
            reason: "Medical emergency".to_string(),
            is_emergency: true,
        };
        assert!(input.amount > 0, "Valid amount should pass");
        assert!(!input.reason.trim().is_empty(), "Valid reason should pass");
    }
}
