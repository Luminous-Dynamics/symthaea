// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Treasury Coordinator Zome
use hdk::prelude::*;
use treasury_integrity::*;

/// Helper to create anchor hash from string
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    anchor_str.hash(&mut hasher);
    let h1 = hasher.finish();
    hasher.write_u64(h1);
    let h2 = hasher.finish();
    hasher.write_u64(h2);
    let h3 = hasher.finish();
    hasher.write_u64(h3);
    let h4 = hasher.finish();

    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&h1.to_le_bytes());
    result[8..16].copy_from_slice(&h2.to_le_bytes());
    result[16..24].copy_from_slice(&h3.to_le_bytes());
    result[24..32].copy_from_slice(&h4.to_le_bytes());

    Ok(EntryHash::from_raw_32(result.to_vec()))
}

#[hdk_extern]
pub fn create_treasury(input: CreateTreasuryInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let treasury = Treasury {
        id: format!(
            "treasury:{}:{}",
            input.name.replace(' ', "_"),
            now.as_micros()
        ),
        name: input.name,
        description: input.description,
        currency: input.currency,
        balance: 0.0,
        reserve_ratio: input.reserve_ratio,
        managers: input.managers.clone(),
        created: now,
        last_updated: now,
    };

    let action_hash = create_entry(&EntryTypes::Treasury(treasury))?;
    for manager in input.managers {
        create_link(
            anchor_hash(&manager)?,
            action_hash.clone(),
            LinkTypes::ManagerToTreasury,
            (),
        )?;
    }
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateTreasuryInput {
    pub name: String,
    pub description: String,
    pub currency: String,
    pub reserve_ratio: f64,
    pub managers: Vec<String>,
}

#[hdk_extern]
pub fn contribute(input: ContributeInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let contribution = Contribution {
        id: format!("contrib:{}:{}", input.contributor_did, now.as_micros()),
        treasury_id: input.treasury_id.clone(),
        contributor_did: input.contributor_did.clone(),
        amount: input.amount,
        currency: input.currency,
        contribution_type: input.contribution_type,
        timestamp: now,
    };

    let action_hash = create_entry(&EntryTypes::Contribution(contribution))?;
    create_link(
        anchor_hash(&input.treasury_id)?,
        action_hash.clone(),
        LinkTypes::TreasuryToContributions,
        (),
    )?;
    create_link(
        anchor_hash(&input.contributor_did)?,
        action_hash.clone(),
        LinkTypes::ContributorToContributions,
        (),
    )?;

    // Update treasury balance
    update_treasury_balance(&input.treasury_id, input.amount)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ContributeInput {
    pub treasury_id: String,
    pub contributor_did: String,
    pub amount: f64,
    pub currency: String,
    pub contribution_type: ContributionType,
}

fn update_treasury_balance(treasury_id: &str, delta: f64) -> ExternResult<()> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Treasury,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(treasury) = record.entry().to_app_option::<Treasury>().ok().flatten() {
            if treasury.id == treasury_id {
                let now = sys_time()?;
                let updated = Treasury {
                    balance: treasury.balance + delta,
                    last_updated: now,
                    ..treasury
                };
                update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Treasury(updated),
                )?;
                return Ok(());
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Treasury not found".into()
    )))
}

#[hdk_extern]
pub fn propose_allocation(input: ProposeAllocationInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let allocation = Allocation {
        id: format!("alloc:{}:{}", input.treasury_id, now.as_micros()),
        treasury_id: input.treasury_id.clone(),
        proposal_id: input.proposal_id,
        recipient_did: input.recipient_did,
        amount: input.amount,
        currency: input.currency,
        purpose: input.purpose,
        status: AllocationStatus::Proposed,
        approved_by: Vec::new(),
        created: now,
        executed: None,
    };

    let action_hash = create_entry(&EntryTypes::Allocation(allocation))?;
    create_link(
        anchor_hash(&input.treasury_id)?,
        action_hash.clone(),
        LinkTypes::TreasuryToAllocations,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeAllocationInput {
    pub treasury_id: String,
    pub proposal_id: Option<String>,
    pub recipient_did: String,
    pub amount: f64,
    pub currency: String,
    pub purpose: String,
}

#[hdk_extern]
pub fn execute_allocation(allocation_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Allocation,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(alloc) = record.entry().to_app_option::<Allocation>().ok().flatten() {
            if alloc.id == allocation_id && alloc.status == AllocationStatus::Approved {
                let now = sys_time()?;
                let executed = Allocation {
                    status: AllocationStatus::Executed,
                    executed: Some(now),
                    ..alloc.clone()
                };
                update_treasury_balance(&alloc.treasury_id, -alloc.amount)?;
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Allocation(executed),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Allocation not found or not approved".into()
    )))
}

#[hdk_extern]
pub fn create_savings_pool(input: CreatePoolInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let pool = SavingsPool {
        id: format!("pool:{}:{}", input.treasury_id, now.as_micros()),
        treasury_id: input.treasury_id.clone(),
        name: input.name,
        target_amount: input.target_amount,
        current_amount: 0.0,
        currency: input.currency,
        members: input.initial_members.clone(),
        yield_rate: input.yield_rate,
        created: now,
    };

    let action_hash = create_entry(&EntryTypes::SavingsPool(pool))?;
    create_link(
        anchor_hash(&input.treasury_id)?,
        action_hash.clone(),
        LinkTypes::TreasuryToPools,
        (),
    )?;
    for member in input.initial_members {
        create_link(
            anchor_hash(&member)?,
            action_hash.clone(),
            LinkTypes::MemberToPool,
            (),
        )?;
    }
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreatePoolInput {
    pub treasury_id: String,
    pub name: String,
    pub target_amount: f64,
    pub currency: String,
    pub initial_members: Vec<String>,
    pub yield_rate: f64,
}

#[hdk_extern]
pub fn get_treasury(treasury_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Treasury,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(treasury) = record.entry().to_app_option::<Treasury>().ok().flatten() {
            if treasury.id == treasury_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Approve an allocation (manager only)
#[hdk_extern]
pub fn approve_allocation(input: ApproveAllocationInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Allocation,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(alloc) = record.entry().to_app_option::<Allocation>().ok().flatten() {
            if alloc.id == input.allocation_id {
                if alloc.status != AllocationStatus::Proposed {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Can only approve proposed allocations".into()
                    )));
                }

                // Verify approver is a treasury manager
                let treasury = get_treasury(alloc.treasury_id.clone())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest("Treasury not found".into())
                ))?;
                let treasury_data = treasury
                    .entry()
                    .to_app_option::<Treasury>()
                    .ok()
                    .flatten()
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Invalid treasury data".into()
                    )))?;

                if !treasury_data.managers.contains(&input.approver_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only managers can approve allocations".into()
                    )));
                }

                // Add approver and check if we have enough approvals
                let mut approved_by = alloc.approved_by.clone();
                if !approved_by.contains(&input.approver_did) {
                    approved_by.push(input.approver_did);
                }

                // Require majority of managers to approve
                let required_approvals = (treasury_data.managers.len() / 2) + 1;
                let new_status = if approved_by.len() >= required_approvals {
                    AllocationStatus::Approved
                } else {
                    AllocationStatus::Proposed
                };

                let updated = Allocation {
                    status: new_status,
                    approved_by,
                    ..alloc
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Allocation(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Allocation not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApproveAllocationInput {
    pub allocation_id: String,
    pub approver_did: String,
}

/// Reject an allocation (manager only)
#[hdk_extern]
pub fn reject_allocation(input: RejectAllocationInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Allocation,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(alloc) = record.entry().to_app_option::<Allocation>().ok().flatten() {
            if alloc.id == input.allocation_id {
                if alloc.status != AllocationStatus::Proposed {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Can only reject proposed allocations".into()
                    )));
                }

                // Verify rejector is a treasury manager
                let treasury = get_treasury(alloc.treasury_id.clone())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest("Treasury not found".into())
                ))?;
                let treasury_data = treasury
                    .entry()
                    .to_app_option::<Treasury>()
                    .ok()
                    .flatten()
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Invalid treasury data".into()
                    )))?;

                if !treasury_data.managers.contains(&input.rejector_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only managers can reject allocations".into()
                    )));
                }

                let rejected = Allocation {
                    status: AllocationStatus::Rejected,
                    ..alloc
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Allocation(rejected),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Allocation not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RejectAllocationInput {
    pub allocation_id: String,
    pub rejector_did: String,
}

/// Get all contributions for a treasury
#[hdk_extern]
pub fn get_treasury_contributions(treasury_id: String) -> ExternResult<Vec<Record>> {
    let mut contributions = Vec::new();
    let query = LinkQuery::try_new(
        anchor_hash(&treasury_id)?,
        LinkTypes::TreasuryToContributions,
    )?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            contributions.push(record);
        }
    }
    Ok(contributions)
}

/// Get all allocations for a treasury
#[hdk_extern]
pub fn get_treasury_allocations(treasury_id: String) -> ExternResult<Vec<Record>> {
    let mut allocations = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&treasury_id)?, LinkTypes::TreasuryToAllocations)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            allocations.push(record);
        }
    }
    Ok(allocations)
}

/// Add a manager to treasury (requires existing manager approval)
#[hdk_extern]
pub fn add_manager(input: AddManagerInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Treasury,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(treasury) = record.entry().to_app_option::<Treasury>().ok().flatten() {
            if treasury.id == input.treasury_id {
                // Verify caller is a manager
                if !treasury.managers.contains(&input.added_by_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only managers can add new managers".into()
                    )));
                }

                // Check if already a manager
                if treasury.managers.contains(&input.new_manager_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "DID is already a manager".into()
                    )));
                }

                let now = sys_time()?;
                let mut managers = treasury.managers.clone();
                managers.push(input.new_manager_did.clone());

                let updated = Treasury {
                    managers,
                    last_updated: now,
                    ..treasury
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Treasury(updated),
                )?;
                create_link(
                    anchor_hash(&input.new_manager_did)?,
                    action_hash.clone(),
                    LinkTypes::ManagerToTreasury,
                    (),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Treasury not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddManagerInput {
    pub treasury_id: String,
    pub new_manager_did: String,
    pub added_by_did: String,
}

/// Remove a manager from treasury (requires existing manager approval, cannot remove last manager)
#[hdk_extern]
pub fn remove_manager(input: RemoveManagerInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Treasury,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(treasury) = record.entry().to_app_option::<Treasury>().ok().flatten() {
            if treasury.id == input.treasury_id {
                // Verify caller is a manager
                if !treasury.managers.contains(&input.removed_by_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only managers can remove managers".into()
                    )));
                }

                // Cannot remove last manager
                if treasury.managers.len() <= 1 {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Cannot remove last manager".into()
                    )));
                }

                // Check if target is a manager
                if !treasury.managers.contains(&input.manager_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "DID is not a manager".into()
                    )));
                }

                let now = sys_time()?;
                let managers: Vec<String> = treasury
                    .managers
                    .iter()
                    .filter(|m| *m != &input.manager_did)
                    .cloned()
                    .collect();

                let updated = Treasury {
                    managers,
                    last_updated: now,
                    ..treasury
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Treasury(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Treasury not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveManagerInput {
    pub treasury_id: String,
    pub manager_did: String,
    pub removed_by_did: String,
}

/// Get all treasuries managed by a DID
#[hdk_extern]
pub fn get_manager_treasuries(manager_did: String) -> ExternResult<Vec<Record>> {
    let mut treasuries = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&manager_did)?, LinkTypes::ManagerToTreasury)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            treasuries.push(record);
        }
    }
    Ok(treasuries)
}

/// Get all savings pools for a treasury
#[hdk_extern]
pub fn get_treasury_pools(treasury_id: String) -> ExternResult<Vec<Record>> {
    let mut pools = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&treasury_id)?, LinkTypes::TreasuryToPools)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            pools.push(record);
        }
    }
    Ok(pools)
}

/// Get a specific savings pool by ID
#[hdk_extern]
pub fn get_savings_pool(pool_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::SavingsPool,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pool) = record.entry().to_app_option::<SavingsPool>().ok().flatten() {
            if pool.id == pool_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Join a savings pool
#[hdk_extern]
pub fn join_savings_pool(input: JoinPoolInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::SavingsPool,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pool) = record.entry().to_app_option::<SavingsPool>().ok().flatten() {
            if pool.id == input.pool_id {
                if pool.members.contains(&input.member_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Already a member".into()
                    )));
                }

                let mut members = pool.members.clone();
                members.push(input.member_did.clone());

                let updated = SavingsPool { members, ..pool };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::SavingsPool(updated),
                )?;
                create_link(
                    anchor_hash(&input.member_did)?,
                    action_hash.clone(),
                    LinkTypes::MemberToPool,
                    (),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Pool not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct JoinPoolInput {
    pub pool_id: String,
    pub member_did: String,
}

/// Contribute to a savings pool
#[hdk_extern]
pub fn contribute_to_pool(input: PoolContributionInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::SavingsPool,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pool) = record.entry().to_app_option::<SavingsPool>().ok().flatten() {
            if pool.id == input.pool_id {
                if !pool.members.contains(&input.contributor_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only members can contribute".into()
                    )));
                }

                let updated = SavingsPool {
                    current_amount: pool.current_amount + input.amount,
                    ..pool
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::SavingsPool(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Pool not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PoolContributionInput {
    pub pool_id: String,
    pub contributor_did: String,
    pub amount: f64,
}

/// Get all pools a member belongs to
#[hdk_extern]
pub fn get_member_pools(member_did: String) -> ExternResult<Vec<Record>> {
    let mut pools = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&member_did)?, LinkTypes::MemberToPool)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            pools.push(record);
        }
    }
    Ok(pools)
}

/// Get contributor's contribution history
#[hdk_extern]
pub fn get_contributor_history(contributor_did: String) -> ExternResult<Vec<Record>> {
    let mut contributions = Vec::new();
    let query = LinkQuery::try_new(
        anchor_hash(&contributor_did)?,
        LinkTypes::ContributorToContributions,
    )?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            contributions.push(record);
        }
    }
    Ok(contributions)
}

/// Cancel a proposed allocation (proposer or manager)
#[hdk_extern]
pub fn cancel_allocation(input: CancelAllocationInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Allocation,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(alloc) = record.entry().to_app_option::<Allocation>().ok().flatten() {
            if alloc.id == input.allocation_id {
                if alloc.status != AllocationStatus::Proposed {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Can only cancel proposed allocations".into()
                    )));
                }

                // Verify caller is a treasury manager
                let treasury = get_treasury(alloc.treasury_id.clone())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest("Treasury not found".into())
                ))?;
                let treasury_data = treasury
                    .entry()
                    .to_app_option::<Treasury>()
                    .ok()
                    .flatten()
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Invalid treasury data".into()
                    )))?;

                if !treasury_data.managers.contains(&input.cancelled_by_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only managers can cancel allocations".into()
                    )));
                }

                let cancelled = Allocation {
                    status: AllocationStatus::Cancelled,
                    ..alloc
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Allocation(cancelled),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Allocation not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CancelAllocationInput {
    pub allocation_id: String,
    pub cancelled_by_did: String,
}

/// Get allocations by status
#[hdk_extern]
pub fn get_allocations_by_status(input: AllocationStatusQuery) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Allocation,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(alloc) = record.entry().to_app_option::<Allocation>().ok().flatten() {
            if alloc.treasury_id == input.treasury_id && alloc.status == input.status {
                results.push(record);
            }
        }
    }
    Ok(results)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AllocationStatusQuery {
    pub treasury_id: String,
    pub status: AllocationStatus,
}
