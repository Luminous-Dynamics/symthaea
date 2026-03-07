//! Treasury Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

// =============================================================================
// STRING LENGTH LIMITS — Prevent DHT bloat attacks
// =============================================================================

const MAX_DID_LEN: usize = 256;
const MAX_ID_LEN: usize = 256;
const MAX_NAME_LEN: usize = 200;
const MAX_DESCRIPTION_LEN: usize = 4096;
const MAX_PURPOSE_LEN: usize = 1024;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Treasury {
    pub id: String,
    pub name: String,
    pub description: String,
    pub currency: String,
    pub balance: u64,
    pub reserve_ratio: f64,
    pub managers: Vec<String>,
    pub created: Timestamp,
    pub last_updated: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Contribution {
    pub id: String,
    pub treasury_id: String,
    pub contributor_did: String,
    pub amount: u64,
    pub currency: String,
    pub contribution_type: ContributionType,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContributionType {
    Deposit,
    Yield,
    Fee,
    Grant,
    Other(String),
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Allocation {
    pub id: String,
    pub treasury_id: String,
    pub proposal_id: Option<String>,
    pub recipient_did: String,
    pub amount: u64,
    pub currency: String,
    pub purpose: String,
    pub status: AllocationStatus,
    pub approved_by: Vec<String>,
    pub created: Timestamp,
    pub executed: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AllocationStatus {
    Proposed,
    Approved,
    Executed,
    Rejected,
    Cancelled,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SavingsPool {
    pub id: String,
    pub treasury_id: String,
    pub name: String,
    pub target_amount: u64,
    pub current_amount: u64,
    pub currency: String,
    pub members: Vec<String>,
    pub yield_rate: f64,
    pub created: Timestamp,
}

/// Commons pool for a DAO -- inalienable reserve is constitutionally protected
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CommonsPool {
    pub id: String,
    pub dao_did: String,
    pub inalienable_reserve: u64,
    pub available_balance: u64,
    pub demurrage_exempt: bool, // Always true -- constitutional
    pub created_at: Timestamp,
    pub last_activity: Timestamp,
}

/// Record of demurrage redistribution received into a commons pool
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CompostReceival {
    pub id: String,
    pub commons_pool_id: String,
    pub amount: u64,
    pub source_member_did: String,
    pub timestamp: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Treasury(Treasury),
    Contribution(Contribution),
    Allocation(Allocation),
    SavingsPool(SavingsPool),
    CommonsPool(CommonsPool),
    CompostReceival(CompostReceival),
}

#[hdk_link_types]
pub enum LinkTypes {
    TreasuryToContributions,
    TreasuryToAllocations,
    TreasuryToPools,
    ManagerToTreasury,
    ContributorToContributions,
    MemberToPool,
    DaoToCommonsPool,
    CommonsPoolToCompost,
    TreasuryIdToTreasury,
    AllocationIdToAllocation,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Treasury(treasury) => {
                    validate_create_treasury(EntryCreationAction::Create(action), treasury)
                }
                EntryTypes::Contribution(contribution) => {
                    validate_create_contribution(EntryCreationAction::Create(action), contribution)
                }
                EntryTypes::Allocation(allocation) => {
                    validate_create_allocation(EntryCreationAction::Create(action), allocation)
                }
                EntryTypes::SavingsPool(pool) => {
                    validate_create_savings_pool(EntryCreationAction::Create(action), pool)
                }
                EntryTypes::CommonsPool(pool) => {
                    validate_create_commons_pool(EntryCreationAction::Create(action), pool)
                }
                EntryTypes::CompostReceival(receival) => {
                    validate_create_compost_receival(EntryCreationAction::Create(action), receival)
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Treasury(treasury) => validate_update_treasury(action, treasury),
                EntryTypes::Contribution(_) => Ok(ValidateCallbackResult::Invalid(
                    "Contributions cannot be updated".into(),
                )),
                EntryTypes::Allocation(allocation) => {
                    validate_update_allocation(action, allocation)
                }
                EntryTypes::SavingsPool(pool) => validate_update_savings_pool(action, pool),
                EntryTypes::CommonsPool(pool) => validate_update_commons_pool(action, pool),
                EntryTypes::CompostReceival(_) => Ok(ValidateCallbackResult::Invalid(
                    "Compost receivals cannot be updated -- they are immutable records".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            ..
        } => {
            // Validate hash lengths (39 bytes for Holochain hashes)
            let base_valid = base_address.as_ref().len() == 39;
            let target_valid = target_address.as_ref().len() == 39;

            match link_type {
                LinkTypes::TreasuryToContributions
                | LinkTypes::TreasuryToAllocations
                | LinkTypes::TreasuryToPools => {
                    // Entry to entry links (treasury to related entries)
                    if !base_valid || !target_valid {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Treasury links must connect valid entry hashes".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ManagerToTreasury
                | LinkTypes::ContributorToContributions
                | LinkTypes::MemberToPool => {
                    // Agent to entry links
                    if !base_valid || !target_valid {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link must connect valid agent and entry hashes".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::DaoToCommonsPool => {
                    if !base_valid || !target_valid {
                        return Ok(ValidateCallbackResult::Invalid(
                            "DaoToCommonsPool link must connect valid hashes".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::CommonsPoolToCompost => {
                    if !base_valid || !target_valid {
                        return Ok(ValidateCallbackResult::Invalid(
                            "CommonsPoolToCompost link must connect valid hashes".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::TreasuryIdToTreasury | LinkTypes::AllocationIdToAllocation => {
                    // Anchor-to-entry links for ID-based lookups
                    if !base_valid || !target_valid {
                        return Ok(ValidateCallbackResult::Invalid(
                            "ID index link must connect valid hashes".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            match link_type {
                // Contribution links are immutable (audit trail)
                LinkTypes::TreasuryToContributions | LinkTypes::ContributorToContributions => {
                    Ok(ValidateCallbackResult::Invalid(
                        "Contribution links cannot be deleted - audit trail must be preserved"
                            .into(),
                    ))
                }
                // Manager links require governance process (not direct deletion)
                LinkTypes::ManagerToTreasury => Ok(ValidateCallbackResult::Invalid(
                    "Manager links cannot be directly deleted - use governance process".into(),
                )),
                // Compost receival links are immutable (audit trail)
                LinkTypes::CommonsPoolToCompost => Ok(ValidateCallbackResult::Invalid(
                    "Compost receival links cannot be deleted - audit trail must be preserved"
                        .into(),
                )),
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_treasury(
    _action: EntryCreationAction,
    treasury: Treasury,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if treasury.name.len() > MAX_NAME_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Treasury name exceeds maximum length of 200".into(),
        ));
    }
    if treasury.description.len() > MAX_DESCRIPTION_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Treasury description exceeds maximum length of 4096".into(),
        ));
    }
    if treasury.id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Treasury ID exceeds maximum length".into(),
        ));
    }
    for manager in &treasury.managers {
        if manager.len() > MAX_DID_LEN {
            return Ok(ValidateCallbackResult::Invalid(
                "Manager DID exceeds maximum length".into(),
            ));
        }
    }

    if treasury.managers.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Treasury must have at least one manager".into(),
        ));
    }
    for manager in &treasury.managers {
        if !manager.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "Managers must be valid DIDs".into(),
            ));
        }
    }
    if treasury.reserve_ratio < 0.0 || treasury.reserve_ratio > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Reserve ratio must be between 0 and 1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_treasury(
    _action: Update,
    treasury: Treasury,
) -> ExternResult<ValidateCallbackResult> {
    if treasury.reserve_ratio < 0.0 || treasury.reserve_ratio > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Reserve ratio must be between 0 and 1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_contribution(
    _action: EntryCreationAction,
    contribution: Contribution,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if contribution.contributor_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if contribution.id.len() > MAX_ID_LEN || contribution.treasury_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "ID exceeds maximum length".into(),
        ));
    }

    if !contribution.contributor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Contributor must be a valid DID".into(),
        ));
    }
    if contribution.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Contribution amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_allocation(
    _action: EntryCreationAction,
    allocation: Allocation,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if allocation.recipient_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if allocation.purpose.len() > MAX_PURPOSE_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Purpose exceeds maximum length of 1024".into(),
        ));
    }
    if allocation.id.len() > MAX_ID_LEN || allocation.treasury_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "ID exceeds maximum length".into(),
        ));
    }
    for approver in &allocation.approved_by {
        if approver.len() > MAX_DID_LEN {
            return Ok(ValidateCallbackResult::Invalid(
                "Approver DID exceeds maximum length".into(),
            ));
        }
    }

    if !allocation.recipient_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Recipient must be a valid DID".into(),
        ));
    }
    if allocation.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Allocation amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_allocation(
    _action: Update,
    allocation: Allocation,
) -> ExternResult<ValidateCallbackResult> {
    if allocation.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Allocation amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_savings_pool(
    _action: EntryCreationAction,
    pool: SavingsPool,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if pool.name.len() > MAX_NAME_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Pool name exceeds maximum length of 200".into(),
        ));
    }
    if pool.id.len() > MAX_ID_LEN || pool.treasury_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "ID exceeds maximum length".into(),
        ));
    }
    for member in &pool.members {
        if member.len() > MAX_DID_LEN {
            return Ok(ValidateCallbackResult::Invalid(
                "Member DID exceeds maximum length".into(),
            ));
        }
    }

    if pool.target_amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Target amount must be positive".into(),
        ));
    }
    if pool.yield_rate < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Yield rate cannot be negative".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_savings_pool(
    _action: Update,
    pool: SavingsPool,
) -> ExternResult<ValidateCallbackResult> {
    if pool.target_amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Target amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate CommonsPool: reserve ratio must never drop below 25%.
/// inalienable_reserve / (inalienable_reserve + available_balance) >= 0.25
/// Exception: total is 0 (empty pool is valid).
fn validate_commons_pool_reserve_ratio(pool: &CommonsPool) -> ExternResult<ValidateCallbackResult> {
    let total = pool.inalienable_reserve + pool.available_balance;
    if total > 0 {
        // Use integer math to avoid floating point: reserve * 100 >= total * 25
        let reserve_pct = pool.inalienable_reserve as u128 * 100;
        let threshold = total as u128 * 25;
        if reserve_pct < threshold {
            return Ok(ValidateCallbackResult::Invalid(
                "Commons pool reserve ratio must be at least 25% (inalienable_reserve / total >= 0.25)".into()
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_commons_pool(
    _action: EntryCreationAction,
    pool: CommonsPool,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if pool.dao_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if pool.id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Pool ID exceeds maximum length".into(),
        ));
    }

    if !pool.dao_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DAO DID must be a valid DID".into(),
        ));
    }
    if !pool.demurrage_exempt {
        return Ok(ValidateCallbackResult::Invalid(
            "Commons pool must be demurrage exempt (constitutional requirement)".into(),
        ));
    }
    validate_commons_pool_reserve_ratio(&pool)
}

fn validate_update_commons_pool(
    _action: Update,
    pool: CommonsPool,
) -> ExternResult<ValidateCallbackResult> {
    if !pool.demurrage_exempt {
        return Ok(ValidateCallbackResult::Invalid(
            "Commons pool must remain demurrage exempt (constitutional requirement)".into(),
        ));
    }
    validate_commons_pool_reserve_ratio(&pool)
}

fn validate_create_compost_receival(
    _action: EntryCreationAction,
    receival: CompostReceival,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if receival.source_member_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if receival.id.len() > MAX_ID_LEN || receival.commons_pool_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "ID exceeds maximum length".into(),
        ));
    }

    if receival.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Compost receival amount must be positive".into(),
        ));
    }
    if !receival.source_member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Source member must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
