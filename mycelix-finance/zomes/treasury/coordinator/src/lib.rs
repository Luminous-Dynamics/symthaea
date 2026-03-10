#![deny(unsafe_code)]
//! Treasury Coordinator Zome
use hdk::prelude::*;
use mycelix_finance_shared::{anchor_hash, follow_update_chain, links_to_records, validate_id, verify_caller_is_did};
use treasury_integrity::*;

const DEFAULT_LIST_LIMIT: usize = 100;

/// Maximum retries for optimistic-locking read-modify-write loops (RC-6 through RC-8).
const MAX_RETRIES: u8 = 3;

/// Generic list input with an ID and optional limit for pagination.
#[derive(Serialize, Deserialize, Debug)]
pub struct ListInput {
    pub id: String,
    pub limit: Option<usize>,
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
        balance: 0,
        reserve_ratio: input.reserve_ratio,
        managers: input.managers.clone(),
        created: now,
        last_updated: now,
    };

    // Prevent duplicate IDs
    let existing = get_links(
        LinkQuery::try_new(anchor_hash(&treasury.id)?, LinkTypes::TreasuryIdToTreasury)?,
        GetStrategy::default(),
    )?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Treasury with this ID already exists".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Treasury(treasury.clone()))?;
    // ID-based index link
    create_link(
        anchor_hash(&treasury.id)?,
        action_hash.clone(),
        LinkTypes::TreasuryIdToTreasury,
        (),
    )?;
    for manager in input.managers {
        create_link(
            anchor_hash(&manager)?,
            action_hash.clone(),
            LinkTypes::ManagerToTreasury,
            (),
        )?;
    }
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Treasury record not found after creation for treasury {}",
            treasury.id
        ))))
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
    verify_caller_is_did(&input.contributor_did)?;
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
    credit_treasury(&input.treasury_id, input.amount)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Contribution record not found after creation for treasury {}",
            input.treasury_id
        ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ContributeInput {
    pub treasury_id: String,
    pub contributor_did: String,
    pub amount: u64,
    pub currency: String,
    pub contribution_type: ContributionType,
}

fn credit_treasury(treasury_id: &str, amount: u64) -> ExternResult<()> {
    for attempt in 0..=MAX_RETRIES {
        let (record, treasury) = get_treasury_record(treasury_id)?;
        let now = sys_time()?;
        let new_balance = treasury.balance.checked_add(amount).ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest("Treasury balance overflow".into()))
        })?;
        let updated = Treasury {
            balance: new_balance,
            last_updated: now,
            ..treasury
        };
        update_entry(
            record.action_address().clone(),
            &EntryTypes::Treasury(updated),
        )?;

        // Verify: re-read and check if our update won
        let (_, verify) = get_treasury_record(treasury_id)?;
        if verify.balance == new_balance {
            return Ok(());
        }
        if attempt == MAX_RETRIES {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "credit_treasury: concurrent update conflict after retries".into()
            )));
        }
    }
    unreachable!()
}

fn debit_treasury(treasury_id: &str, amount: u64) -> ExternResult<()> {
    for attempt in 0..=MAX_RETRIES {
        let (record, treasury) = get_treasury_record(treasury_id)?;
        let now = sys_time()?;
        let new_balance = treasury.balance.checked_sub(amount).ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest(
                "Insufficient treasury balance".into()
            ))
        })?;
        let updated = Treasury {
            balance: new_balance,
            last_updated: now,
            ..treasury
        };
        update_entry(
            record.action_address().clone(),
            &EntryTypes::Treasury(updated),
        )?;

        // Verify: re-read and check if our update won
        let (_, verify) = get_treasury_record(treasury_id)?;
        if verify.balance == new_balance {
            return Ok(());
        }
        if attempt == MAX_RETRIES {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "debit_treasury: concurrent update conflict after retries".into()
            )));
        }
    }
    unreachable!()
}

/// Internal helper: fetch a treasury Record + deserialized entry by ID via link index.
/// Follows the update chain to return the latest version.
fn get_treasury_record(treasury_id: &str) -> ExternResult<(Record, Treasury)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(treasury_id)?, LinkTypes::TreasuryIdToTreasury)?,
        GetStrategy::default(),
    )?;
    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Treasury not found".into()
        )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
    let treasury = record
        .entry()
        .to_app_option::<Treasury>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Treasury deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Treasury entry missing".into()
        )))?;
    Ok((record, treasury))
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

    // Prevent duplicate IDs
    let existing = get_links(
        LinkQuery::try_new(
            anchor_hash(&allocation.id)?,
            LinkTypes::AllocationIdToAllocation,
        )?,
        GetStrategy::default(),
    )?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Allocation with this ID already exists".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Allocation(allocation.clone()))?;
    create_link(
        anchor_hash(&input.treasury_id)?,
        action_hash.clone(),
        LinkTypes::TreasuryToAllocations,
        (),
    )?;
    // ID-based index link
    create_link(
        anchor_hash(&allocation.id)?,
        action_hash.clone(),
        LinkTypes::AllocationIdToAllocation,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Allocation record not found after creation for allocation {}",
            allocation.id
        ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeAllocationInput {
    pub treasury_id: String,
    pub proposal_id: Option<String>,
    pub recipient_did: String,
    pub amount: u64,
    pub currency: String,
    pub purpose: String,
}

#[hdk_extern]
pub fn execute_allocation(allocation_id: String) -> ExternResult<Record> {
    validate_id(&allocation_id, "allocation_id")?;
    let (record, alloc) = get_allocation_record(&allocation_id)?;
    if alloc.status != AllocationStatus::Approved {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Allocation not found or not approved".into()
        )));
    }
    let now = sys_time()?;
    let executed = Allocation {
        status: AllocationStatus::Executed,
        executed: Some(now),
        ..alloc.clone()
    };
    debit_treasury(&alloc.treasury_id, alloc.amount)?;
    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::Allocation(executed),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Allocation record not found after execution for allocation {}",
            allocation_id
        ))))
}

/// Internal helper: fetch an allocation Record + deserialized entry by ID via link index.
/// Follows the update chain to return the latest version.
fn get_allocation_record(allocation_id: &str) -> ExternResult<(Record, Allocation)> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(allocation_id)?,
            LinkTypes::AllocationIdToAllocation,
        )?,
        GetStrategy::default(),
    )?;
    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Allocation not found".into()
        )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
    let alloc = record
        .entry()
        .to_app_option::<Allocation>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Allocation deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Allocation entry missing".into()
        )))?;
    Ok((record, alloc))
}

/// Internal helper: fetch a SavingsPool Record + deserialized entry by ID via link index.
/// Follows the update chain to return the latest version.
fn get_savings_pool_record(pool_id: &str) -> ExternResult<(Record, SavingsPool)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(pool_id)?, LinkTypes::PoolIdToPool)?,
        GetStrategy::default(),
    )?;
    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Savings pool not found".into()
        )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
    let pool = record
        .entry()
        .to_app_option::<SavingsPool>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "SavingsPool deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "SavingsPool entry missing".into()
        )))?;
    Ok((record, pool))
}

/// Internal helper: fetch a CommonsPool Record + deserialized entry by ID via link index.
/// Follows the update chain to return the latest version.
fn get_commons_pool_record(pool_id: &str) -> ExternResult<(Record, CommonsPool)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(pool_id)?, LinkTypes::CommonsPoolIdToPool)?,
        GetStrategy::default(),
    )?;
    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Commons pool not found".into()
        )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
    let pool = record
        .entry()
        .to_app_option::<CommonsPool>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "CommonsPool deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "CommonsPool entry missing".into()
        )))?;
    Ok((record, pool))
}

#[hdk_extern]
pub fn create_savings_pool(input: CreatePoolInput) -> ExternResult<Record> {
    if !input.yield_rate.is_finite() || input.yield_rate < 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Yield rate must be a finite non-negative number".into()
        )));
    }
    let now = sys_time()?;
    let pool = SavingsPool {
        id: format!("pool:{}:{}", input.treasury_id, now.as_micros()),
        treasury_id: input.treasury_id.clone(),
        name: input.name,
        target_amount: input.target_amount,
        current_amount: 0,
        currency: input.currency,
        members: input.initial_members.clone(),
        yield_rate: input.yield_rate,
        created: now,
    };

    // Prevent duplicate IDs
    let existing = get_links(
        LinkQuery::try_new(anchor_hash(&pool.id)?, LinkTypes::PoolIdToPool)?,
        GetStrategy::default(),
    )?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Savings pool with this ID already exists".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::SavingsPool(pool.clone()))?;
    // ID-based index link for O(1) lookups
    create_link(
        anchor_hash(&pool.id)?,
        action_hash.clone(),
        LinkTypes::PoolIdToPool,
        (),
    )?;
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
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Savings pool record not found after creation for pool {}",
            pool.id
        ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreatePoolInput {
    pub treasury_id: String,
    pub name: String,
    pub target_amount: u64,
    pub currency: String,
    pub initial_members: Vec<String>,
    pub yield_rate: f64,
}

#[hdk_extern]
pub fn get_treasury(treasury_id: String) -> ExternResult<Option<Record>> {
    validate_id(&treasury_id, "treasury_id")?;
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&treasury_id)?, LinkTypes::TreasuryIdToTreasury)?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return Ok(Some(follow_update_chain(hash)?));
    }
    Ok(None)
}

/// Approve an allocation (manager only)
#[hdk_extern]
pub fn approve_allocation(input: ApproveAllocationInput) -> ExternResult<Record> {
    let (record, alloc) = get_allocation_record(&input.allocation_id)?;

    if alloc.status != AllocationStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only approve proposed allocations".into()
        )));
    }

    // Verify caller is the approver
    verify_caller_is_did(&input.approver_did)?;

    // Verify approver is a treasury manager
    let treasury = get_treasury(alloc.treasury_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Treasury not found".into())
    ))?;
    let treasury_data = treasury
        .entry()
        .to_app_option::<Treasury>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Treasury deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Treasury entry missing".into()
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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Allocation record not found after approval for allocation {}",
            input.allocation_id
        ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApproveAllocationInput {
    pub allocation_id: String,
    pub approver_did: String,
}

/// Reject an allocation (manager only)
#[hdk_extern]
pub fn reject_allocation(input: RejectAllocationInput) -> ExternResult<Record> {
    let (record, alloc) = get_allocation_record(&input.allocation_id)?;

    if alloc.status != AllocationStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only reject proposed allocations".into()
        )));
    }

    // Verify caller is the rejector
    verify_caller_is_did(&input.rejector_did)?;

    // Verify rejector is a treasury manager
    let treasury = get_treasury(alloc.treasury_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Treasury not found".into())
    ))?;
    let treasury_data = treasury
        .entry()
        .to_app_option::<Treasury>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Treasury deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Treasury entry missing".into()
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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Allocation record not found after rejection for allocation {}",
            input.allocation_id
        ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RejectAllocationInput {
    pub allocation_id: String,
    pub rejector_did: String,
}

/// Get contributions for a treasury (paginated)
///
/// Batch-optimized: Contribution entries are immutable (write-once records of
/// SAP contributed), so bare get() via links_to_records is equivalent to
/// follow_update_chain.
#[hdk_extern]
pub fn get_treasury_contributions(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::TreasuryToContributions)?;
    let links: Vec<_> = get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
        .collect();
    links_to_records(links)
}

/// Get allocations for a treasury (paginated)
///
/// Uses follow_update_chain because allocations are mutable (status transitions).
#[hdk_extern]
pub fn get_treasury_allocations(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut allocations = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::TreasuryToAllocations)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
        allocations.push(follow_update_chain(hash)?);
    }
    Ok(allocations)
}

/// Add a manager to treasury (requires existing manager approval)
#[hdk_extern]
pub fn add_manager(input: AddManagerInput) -> ExternResult<Record> {
    let (record, treasury) = get_treasury_record(&input.treasury_id)?;

    // Verify caller is the adding manager
    verify_caller_is_did(&input.added_by_did)?;

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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Treasury record not found after adding manager {} to treasury {}",
            input.new_manager_did, input.treasury_id
        ))))
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
    let (record, treasury) = get_treasury_record(&input.treasury_id)?;

    // Verify caller is the removing manager
    verify_caller_is_did(&input.removed_by_did)?;

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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Treasury record not found after removing manager {} from treasury {}",
            input.manager_did, input.treasury_id
        ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveManagerInput {
    pub treasury_id: String,
    pub manager_did: String,
    pub removed_by_did: String,
}

/// Get treasuries managed by a DID (paginated)
///
/// Uses follow_update_chain because treasuries are mutable (balance, managers).
#[hdk_extern]
pub fn get_manager_treasuries(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut treasuries = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::ManagerToTreasury)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
        treasuries.push(follow_update_chain(hash)?);
    }
    Ok(treasuries)
}

/// Get savings pools for a treasury (paginated)
///
/// Uses follow_update_chain because savings pools are mutable (members, balance).
#[hdk_extern]
pub fn get_treasury_pools(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut pools = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::TreasuryToPools)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
        pools.push(follow_update_chain(hash)?);
    }
    Ok(pools)
}

/// Get a specific savings pool by ID (O(1) link-based lookup)
#[hdk_extern]
pub fn get_savings_pool(pool_id: String) -> ExternResult<Option<Record>> {
    validate_id(&pool_id, "pool_id")?;
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&pool_id)?, LinkTypes::PoolIdToPool)?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return Ok(Some(follow_update_chain(hash)?));
    }
    Ok(None)
}

/// Join a savings pool
#[hdk_extern]
pub fn join_savings_pool(input: JoinPoolInput) -> ExternResult<Record> {
    verify_caller_is_did(&input.member_did)?;
    let (record, pool) = get_savings_pool_record(&input.pool_id)?;

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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Savings pool record not found after member {} joined pool {}",
            input.member_did, input.pool_id
        ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct JoinPoolInput {
    pub pool_id: String,
    pub member_did: String,
}

/// Contribute to a savings pool
#[hdk_extern]
pub fn contribute_to_pool(input: PoolContributionInput) -> ExternResult<Record> {
    for attempt in 0..=MAX_RETRIES {
        let (record, pool) = get_savings_pool_record(&input.pool_id)?;

        if !pool.members.contains(&input.contributor_did) {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only members can contribute".into()
            )));
        }

        let new_amount = pool.current_amount.checked_add(input.amount).ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest("Pool balance overflow".into()))
        })?;
        let updated = SavingsPool {
            current_amount: new_amount,
            ..pool
        };

        let action_hash = update_entry(
            record.action_address().clone(),
            &EntryTypes::SavingsPool(updated),
        )?;

        // Verify: re-read and check if our update won
        let (_, verify) = get_savings_pool_record(&input.pool_id)?;
        if verify.current_amount == new_amount {
            return get(action_hash, GetOptions::default())?
                .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
                    "Savings pool record not found after contribution to pool {}",
                    input.pool_id
                ))));
        }
        if attempt == MAX_RETRIES {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "contribute_to_pool: concurrent update conflict after retries".into()
            )));
        }
    }
    unreachable!()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PoolContributionInput {
    pub pool_id: String,
    pub contributor_did: String,
    pub amount: u64,
}

/// Get pools a member belongs to (paginated)
///
/// Uses follow_update_chain because savings pools are mutable (members, balance).
#[hdk_extern]
pub fn get_member_pools(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut pools = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::MemberToPool)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
        pools.push(follow_update_chain(hash)?);
    }
    Ok(pools)
}

/// Get contributor's contribution history (paginated)
///
/// Batch-optimized: Contribution entries are immutable (write-once records of
/// SAP contributed), so bare get() via links_to_records is equivalent to
/// follow_update_chain.
#[hdk_extern]
pub fn get_contributor_history(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let query = LinkQuery::try_new(
        anchor_hash(&input.id)?,
        LinkTypes::ContributorToContributions,
    )?;
    let links: Vec<_> = get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
        .collect();
    links_to_records(links)
}

/// Cancel a proposed allocation (proposer or manager)
#[hdk_extern]
pub fn cancel_allocation(input: CancelAllocationInput) -> ExternResult<Record> {
    let (record, alloc) = get_allocation_record(&input.allocation_id)?;

    if alloc.status != AllocationStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only cancel proposed allocations".into()
        )));
    }

    // Verify caller is the cancelling manager
    verify_caller_is_did(&input.cancelled_by_did)?;

    let treasury = get_treasury(alloc.treasury_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Treasury not found".into())
    ))?;
    let treasury_data = treasury
        .entry()
        .to_app_option::<Treasury>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Treasury deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Treasury entry missing".into()
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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Allocation record not found after cancellation for allocation {}",
            input.allocation_id
        ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CancelAllocationInput {
    pub allocation_id: String,
    pub cancelled_by_did: String,
}

/// Get allocations by status (uses TreasuryToAllocations links + in-memory status filter)
///
// NOTE: Sequential follow_update_chain required — entries are mutable (status transitions)
#[hdk_extern]
pub fn get_allocations_by_status(input: AllocationStatusQuery) -> ExternResult<Vec<Record>> {
    let query = LinkQuery::try_new(
        anchor_hash(&input.treasury_id)?,
        LinkTypes::TreasuryToAllocations,
    )?;
    let mut results = Vec::new();
    for link in get_links(query, GetStrategy::default())? {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
        let record = follow_update_chain(hash)?;
        if let Some(alloc) = record
            .entry()
            .to_app_option::<Allocation>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Allocation deserialization error: {:?}",
                    e
                )))
            })?
        {
            if alloc.status == input.status {
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

// ---------------------------------------------------------------------------
// Commons Pool functions
// ---------------------------------------------------------------------------

/// Create a new commons pool for a DAO
#[hdk_extern]
pub fn create_commons_pool(input: CreateCommonsPoolInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let pool = CommonsPool {
        id: format!(
            "commons:{}:{}",
            input.dao_did.replace(':', "_"),
            now.as_micros()
        ),
        dao_did: input.dao_did.clone(),
        inalienable_reserve: 0,
        available_balance: 0,
        demurrage_exempt: true, // Constitutional -- always true
        created_at: now,
        last_activity: now,
    };

    // Prevent duplicate IDs
    let existing = get_links(
        LinkQuery::try_new(anchor_hash(&pool.id)?, LinkTypes::CommonsPoolIdToPool)?,
        GetStrategy::default(),
    )?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Commons pool with this ID already exists".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::CommonsPool(pool.clone()))?;
    // ID-based index link for O(1) lookups
    create_link(
        anchor_hash(&pool.id)?,
        action_hash.clone(),
        LinkTypes::CommonsPoolIdToPool,
        (),
    )?;
    create_link(
        anchor_hash(&input.dao_did)?,
        action_hash.clone(),
        LinkTypes::DaoToCommonsPool,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Commons pool record not found after creation for pool {}",
            pool.id
        ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCommonsPoolInput {
    pub dao_did: String,
}

/// Contribute SAP to a commons pool.
/// 25% goes to the inalienable reserve, 75% goes to available balance.
#[hdk_extern]
pub fn contribute_to_commons(input: ContributeToCommonsInput) -> ExternResult<Record> {
    verify_caller_is_did(&input.contributor_did)?;

    for attempt in 0..=MAX_RETRIES {
        let (record, pool) = get_commons_pool_record(&input.commons_pool_id)?;
        let now = sys_time()?;

        // Split: 25% inalienable reserve, 75% available (integer math)
        let to_reserve = input.amount / 4;
        let to_available = input.amount - to_reserve;

        let new_reserve = pool.inalienable_reserve.checked_add(to_reserve).ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest("Reserve overflow".into()))
        })?;
        let new_available = pool.available_balance.checked_add(to_available).ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest("Available balance overflow".into()))
        })?;

        let updated = CommonsPool {
            inalienable_reserve: new_reserve,
            available_balance: new_available,
            last_activity: now,
            ..pool
        };

        let action_hash = update_entry(
            record.action_address().clone(),
            &EntryTypes::CommonsPool(updated),
        )?;

        // Verify: re-read and check if our update won
        let (_, verify) = get_commons_pool_record(&input.commons_pool_id)?;
        if verify.inalienable_reserve == new_reserve && verify.available_balance == new_available {
            return get(action_hash, GetOptions::default())?
                .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
                    "Commons pool record not found after contribution to pool {}",
                    input.commons_pool_id
                ))));
        }
        if attempt == MAX_RETRIES {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "contribute_to_commons: concurrent update conflict after retries".into()
            )));
        }
    }
    unreachable!()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ContributeToCommonsInput {
    pub commons_pool_id: String,
    pub contributor_did: String,
    pub amount: u64,
}

/// Receive demurrage redistribution (compost) into the commons pool available balance.
#[hdk_extern]
pub fn receive_compost(input: ReceiveCompostInput) -> ExternResult<Record> {
    // Record the compost receival once (idempotent side-effect outside retry loop)
    let now_receipt = sys_time()?;
    let receival = CompostReceival {
        id: format!("compost:{}:{}", input.commons_pool_id, now_receipt.as_micros()),
        commons_pool_id: input.commons_pool_id.clone(),
        amount: input.amount,
        source_member_did: input.source_member_did.clone(),
        timestamp: now_receipt,
    };
    let receival_hash = create_entry(&EntryTypes::CompostReceival(receival))?;
    create_link(
        anchor_hash(&input.commons_pool_id)?,
        receival_hash,
        LinkTypes::CommonsPoolToCompost,
        (),
    )?;

    for attempt in 0..=MAX_RETRIES {
        let (record, pool) = get_commons_pool_record(&input.commons_pool_id)?;
        let now = sys_time()?;

        // Add to available balance (compost goes to available, not reserve)
        let new_available = pool.available_balance.checked_add(input.amount).ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest("Available balance overflow".into()))
        })?;
        let updated = CommonsPool {
            available_balance: new_available,
            last_activity: now,
            ..pool
        };

        let action_hash = update_entry(
            record.action_address().clone(),
            &EntryTypes::CommonsPool(updated),
        )?;

        // Verify: re-read and check if our update won
        let (_, verify) = get_commons_pool_record(&input.commons_pool_id)?;
        if verify.available_balance == new_available {
            return get(action_hash, GetOptions::default())?
                .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
                    "Commons pool record not found after receiving compost for pool {}",
                    input.commons_pool_id
                ))));
        }
        if attempt == MAX_RETRIES {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "receive_compost: concurrent update conflict after retries".into()
            )));
        }
    }
    unreachable!()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReceiveCompostInput {
    pub commons_pool_id: String,
    pub amount: u64,
    pub source_member_did: String,
}

/// Request allocation from commons pool available balance only.
/// Validates that the inalienable reserve is never touched and that the
/// reserve ratio remains at or above 25% after the allocation.
#[hdk_extern]
pub fn request_allocation(input: RequestCommonsAllocationInput) -> ExternResult<Record> {
    for attempt in 0..=MAX_RETRIES {
        let (record, pool) = get_commons_pool_record(&input.commons_pool_id)?;

        // Validate: allocation comes only from available balance
        if input.amount > pool.available_balance {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Allocation exceeds available balance (inalienable reserve is untouchable)"
                    .into()
            )));
        }

        let new_available = pool.available_balance - input.amount;
        let new_total = pool.inalienable_reserve + new_available;

        // Validate reserve ratio stays >= 25% (or total is 0)
        if new_total > 0 {
            let reserve_pct = pool.inalienable_reserve as u128 * 100;
            let threshold = new_total as u128 * 25;
            if reserve_pct < threshold {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Allocation would drop reserve ratio below 25% constitutional minimum"
                        .into()
                )));
            }
        }

        let now = sys_time()?;
        let updated = CommonsPool {
            available_balance: new_available,
            last_activity: now,
            ..pool
        };

        let action_hash = update_entry(
            record.action_address().clone(),
            &EntryTypes::CommonsPool(updated),
        )?;

        // Verify: re-read and check if our update won
        let (_, verify) = get_commons_pool_record(&input.commons_pool_id)?;
        if verify.available_balance == new_available {
            return get(action_hash, GetOptions::default())?
                .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
                    "Commons pool record not found after allocation from pool {}",
                    input.commons_pool_id
                ))));
        }
        if attempt == MAX_RETRIES {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "request_allocation: concurrent update conflict after retries".into()
            )));
        }
    }
    unreachable!()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestCommonsAllocationInput {
    pub commons_pool_id: String,
    pub requester_did: String,
    pub amount: u64,
    pub purpose: String,
}

/// Get a commons pool by its ID (O(1) link-based lookup)
#[hdk_extern]
pub fn get_commons_pool(pool_id: String) -> ExternResult<Option<Record>> {
    validate_id(&pool_id, "pool_id")?;
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&pool_id)?, LinkTypes::CommonsPoolIdToPool)?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return Ok(Some(follow_update_chain(hash)?));
    }
    Ok(None)
}
