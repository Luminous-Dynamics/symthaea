//! Treasury Coordinator Zome
use hdk::prelude::*;
use mycelix_finance_shared::{anchor_hash, verify_caller_is_did};
use treasury_integrity::*;

const DEFAULT_LIST_LIMIT: usize = 100;

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
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let (record, treasury) = get_treasury_record(treasury_id)?;
    let now = sys_time()?;
    let new_balance = treasury
        .balance
        .checked_add(amount)
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Treasury balance overflow".into())))?;
    let updated = Treasury {
        balance: new_balance,
        last_updated: now,
        ..treasury
    };
    update_entry(
        record.action_address().clone(),
        &EntryTypes::Treasury(updated),
    )?;
    Ok(())
}

fn debit_treasury(treasury_id: &str, amount: u64) -> ExternResult<()> {
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
    Ok(())
}

/// Internal helper: fetch a treasury Record + deserialized entry by ID via link index.
fn get_treasury_record(treasury_id: &str) -> ExternResult<(Record, Treasury)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(treasury_id)?, LinkTypes::TreasuryIdToTreasury)?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            let treasury = record
                .entry()
                .to_app_option::<Treasury>()
                .ok()
                .flatten()
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("Invalid treasury data".into()))
                })?;
            return Ok((record, treasury));
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
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Internal helper: fetch an allocation Record + deserialized entry by ID via link index.
fn get_allocation_record(allocation_id: &str) -> ExternResult<(Record, Allocation)> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(allocation_id)?,
            LinkTypes::AllocationIdToAllocation,
        )?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            let alloc = record
                .entry()
                .to_app_option::<Allocation>()
                .ok()
                .flatten()
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("Invalid allocation data".into()))
                })?;
            return Ok((record, alloc));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Allocation not found".into()
    )))
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
    pub target_amount: u64,
    pub currency: String,
    pub initial_members: Vec<String>,
    pub yield_rate: f64,
}

#[hdk_extern]
pub fn get_treasury(treasury_id: String) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&treasury_id)?, LinkTypes::TreasuryIdToTreasury)?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(hash, GetOptions::default());
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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RejectAllocationInput {
    pub allocation_id: String,
    pub rejector_did: String,
}

/// Get contributions for a treasury (paginated)
#[hdk_extern]
pub fn get_treasury_contributions(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut contributions = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::TreasuryToContributions)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
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

/// Get allocations for a treasury (paginated)
#[hdk_extern]
pub fn get_treasury_allocations(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut allocations = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::TreasuryToAllocations)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
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
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveManagerInput {
    pub treasury_id: String,
    pub manager_did: String,
    pub removed_by_did: String,
}

/// Get treasuries managed by a DID (paginated)
#[hdk_extern]
pub fn get_manager_treasuries(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut treasuries = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::ManagerToTreasury)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
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

/// Get savings pools for a treasury (paginated)
#[hdk_extern]
pub fn get_treasury_pools(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut pools = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::TreasuryToPools)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
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
    verify_caller_is_did(&input.member_did)?;
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

                let new_amount =
                    pool.current_amount
                        .checked_add(input.amount)
                        .ok_or_else(|| {
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
    pub amount: u64,
}

/// Get pools a member belongs to (paginated)
#[hdk_extern]
pub fn get_member_pools(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut pools = Vec::new();
    let query = LinkQuery::try_new(anchor_hash(&input.id)?, LinkTypes::MemberToPool)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
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

/// Get contributor's contribution history (paginated)
#[hdk_extern]
pub fn get_contributor_history(input: ListInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    let mut contributions = Vec::new();
    let query = LinkQuery::try_new(
        anchor_hash(&input.id)?,
        LinkTypes::ContributorToContributions,
    )?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(limit)
    {
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
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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

    let action_hash = create_entry(&EntryTypes::CommonsPool(pool))?;
    create_link(
        anchor_hash(&input.dao_did)?,
        action_hash.clone(),
        LinkTypes::DaoToCommonsPool,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CommonsPool,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pool) = record.entry().to_app_option::<CommonsPool>().ok().flatten() {
            if pool.id == input.commons_pool_id {
                let now = sys_time()?;

                // Split: 25% inalienable reserve, 75% available (integer math)
                let to_reserve = input.amount / 4;
                let to_available = input.amount - to_reserve;

                let new_reserve = pool
                    .inalienable_reserve
                    .checked_add(to_reserve)
                    .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Reserve overflow".into())))?;
                let new_available = pool
                    .available_balance
                    .checked_add(to_available)
                    .ok_or_else(|| {
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
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Commons pool not found".into()
    )))
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
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CommonsPool,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pool) = record.entry().to_app_option::<CommonsPool>().ok().flatten() {
            if pool.id == input.commons_pool_id {
                let now = sys_time()?;

                // Record the compost receival
                let receival = CompostReceival {
                    id: format!("compost:{}:{}", input.commons_pool_id, now.as_micros()),
                    commons_pool_id: input.commons_pool_id.clone(),
                    amount: input.amount,
                    source_member_did: input.source_member_did.clone(),
                    timestamp: now,
                };
                let receival_hash = create_entry(&EntryTypes::CompostReceival(receival))?;
                create_link(
                    anchor_hash(&input.commons_pool_id)?,
                    receival_hash,
                    LinkTypes::CommonsPoolToCompost,
                    (),
                )?;

                // Add to available balance (compost goes to available, not reserve)
                let new_available = pool
                    .available_balance
                    .checked_add(input.amount)
                    .ok_or_else(|| {
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
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Commons pool not found".into()
    )))
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
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CommonsPool,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pool) = record.entry().to_app_option::<CommonsPool>().ok().flatten() {
            if pool.id == input.commons_pool_id {
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
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Commons pool not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestCommonsAllocationInput {
    pub commons_pool_id: String,
    pub requester_did: String,
    pub amount: u64,
    pub purpose: String,
}

/// Get a commons pool by its ID
#[hdk_extern]
pub fn get_commons_pool(pool_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CommonsPool,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(pool) = record.entry().to_app_option::<CommonsPool>().ok().flatten() {
            if pool.id == pool_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}
