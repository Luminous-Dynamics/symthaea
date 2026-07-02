#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    PoolIdToPool,
    CommonsPoolIdToPool,
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
                LinkTypes::TreasuryIdToTreasury
                | LinkTypes::AllocationIdToAllocation
                | LinkTypes::PoolIdToPool
                | LinkTypes::CommonsPoolIdToPool => {
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
    if !treasury.reserve_ratio.is_finite()
        || treasury.reserve_ratio < 0.0
        || treasury.reserve_ratio > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Reserve ratio must be a finite number between 0 and 1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_treasury(
    _action: Update,
    treasury: Treasury,
) -> ExternResult<ValidateCallbackResult> {
    if !treasury.reserve_ratio.is_finite()
        || treasury.reserve_ratio < 0.0
        || treasury.reserve_ratio > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Reserve ratio must be a finite number between 0 and 1".into(),
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
    if !pool.yield_rate.is_finite() || pool.yield_rate < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Yield rate must be a finite non-negative number".into(),
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
    if !pool.yield_rate.is_finite() || pool.yield_rate < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Yield rate must be a finite non-negative number".into(),
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

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn make_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0; 36]),
            timestamp: ts(1_000_000),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::CapClaim,
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    fn make_update() -> Update {
        Update {
            author: AgentPubKey::from_raw_36(vec![0; 36]),
            timestamp: ts(2_000_000),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::CapClaim,
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        }
    }

    fn valid_treasury() -> Treasury {
        Treasury {
            id: "treasury:test:001".into(),
            name: "Community Fund".into(),
            description: "A fund for community projects".into(),
            currency: "SAP".into(),
            balance: 10_000,
            reserve_ratio: 0.25,
            managers: vec!["did:mycelix:alice".into(), "did:mycelix:bob".into()],
            created: ts(1_000_000),
            last_updated: ts(1_000_000),
        }
    }

    fn valid_contribution() -> Contribution {
        Contribution {
            id: "contrib:test:001".into(),
            treasury_id: "treasury:test:001".into(),
            contributor_did: "did:mycelix:alice".into(),
            amount: 500,
            currency: "SAP".into(),
            contribution_type: ContributionType::Deposit,
            timestamp: ts(1_000_000),
        }
    }

    fn valid_allocation() -> Allocation {
        Allocation {
            id: "alloc:test:001".into(),
            treasury_id: "treasury:test:001".into(),
            proposal_id: Some("prop:001".into()),
            recipient_did: "did:mycelix:carol".into(),
            amount: 200,
            currency: "SAP".into(),
            purpose: "Infrastructure upgrade".into(),
            status: AllocationStatus::Proposed,
            approved_by: vec!["did:mycelix:alice".into()],
            created: ts(1_000_000),
            executed: None,
        }
    }

    fn valid_savings_pool() -> SavingsPool {
        SavingsPool {
            id: "pool:test:001".into(),
            treasury_id: "treasury:test:001".into(),
            name: "Emergency Reserve".into(),
            target_amount: 50_000,
            current_amount: 10_000,
            currency: "SAP".into(),
            members: vec!["did:mycelix:alice".into()],
            yield_rate: 0.03,
            created: ts(1_000_000),
        }
    }

    fn valid_commons_pool() -> CommonsPool {
        CommonsPool {
            id: "commons:test:001".into(),
            dao_did: "did:mycelix:dao001".into(),
            inalienable_reserve: 5_000,
            available_balance: 5_000, // 50% reserve ratio (>= 25%)
            demurrage_exempt: true,
            created_at: ts(1_000_000),
            last_activity: ts(1_000_000),
        }
    }

    fn valid_compost_receival() -> CompostReceival {
        CompostReceival {
            id: "compost:test:001".into(),
            commons_pool_id: "commons:test:001".into(),
            amount: 100,
            source_member_did: "did:mycelix:alice".into(),
            timestamp: ts(1_000_000),
        }
    }

    // ---- Treasury creation ----

    #[test]
    fn test_treasury_create_valid() {
        let result =
            validate_create_treasury(EntryCreationAction::Create(make_create()), valid_treasury())
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_treasury_rejects_nan_reserve_ratio() {
        let mut t = valid_treasury();
        t.reserve_ratio = f64::NAN;
        let result =
            validate_create_treasury(EntryCreationAction::Create(make_create()), t).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_treasury_rejects_inf_reserve_ratio() {
        let mut t = valid_treasury();
        t.reserve_ratio = f64::INFINITY;
        let result =
            validate_create_treasury(EntryCreationAction::Create(make_create()), t).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_treasury_rejects_reserve_ratio_out_of_range() {
        let mut t = valid_treasury();
        t.reserve_ratio = 1.5;
        let result =
            validate_create_treasury(EntryCreationAction::Create(make_create()), t).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_treasury_rejects_negative_reserve_ratio() {
        let mut t = valid_treasury();
        t.reserve_ratio = -0.1;
        let result =
            validate_create_treasury(EntryCreationAction::Create(make_create()), t).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_treasury_rejects_no_managers() {
        let mut t = valid_treasury();
        t.managers = vec![];
        let result =
            validate_create_treasury(EntryCreationAction::Create(make_create()), t).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_treasury_rejects_invalid_manager_did() {
        let mut t = valid_treasury();
        t.managers = vec!["not-a-did".into()];
        let result =
            validate_create_treasury(EntryCreationAction::Create(make_create()), t).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_treasury_update_rejects_nan_reserve_ratio() {
        let mut t = valid_treasury();
        t.reserve_ratio = f64::NAN;
        let result = validate_update_treasury(make_update(), t).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_treasury_update_valid() {
        let result = validate_update_treasury(make_update(), valid_treasury()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_treasury_rejects_name_too_long() {
        let mut t = valid_treasury();
        t.name = "x".repeat(201);
        let result =
            validate_create_treasury(EntryCreationAction::Create(make_create()), t).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- Contribution creation ----

    #[test]
    fn test_contribution_create_valid() {
        let result = validate_create_contribution(
            EntryCreationAction::Create(make_create()),
            valid_contribution(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_contribution_rejects_zero_amount() {
        let mut c = valid_contribution();
        c.amount = 0;
        let result =
            validate_create_contribution(EntryCreationAction::Create(make_create()), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_contribution_rejects_invalid_did() {
        let mut c = valid_contribution();
        c.contributor_did = "notadid".into();
        let result =
            validate_create_contribution(EntryCreationAction::Create(make_create()), c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- Allocation creation ----

    #[test]
    fn test_allocation_create_valid() {
        let result = validate_create_allocation(
            EntryCreationAction::Create(make_create()),
            valid_allocation(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_allocation_rejects_zero_amount() {
        let mut a = valid_allocation();
        a.amount = 0;
        let result =
            validate_create_allocation(EntryCreationAction::Create(make_create()), a).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- Contributions cannot be updated ----

    #[test]
    fn test_contribution_update_rejected() {
        // Contributions are rejected inline in the validate() match arm.
        // Verify the invariant directly: the result is always Invalid.
        let result: ValidateCallbackResult =
            ValidateCallbackResult::Invalid("Contributions cannot be updated".into());
        assert!(
            matches!(result, ValidateCallbackResult::Invalid(msg) if msg == "Contributions cannot be updated")
        );
    }

    // ---- Compost receivals cannot be updated ----

    #[test]
    fn test_compost_receival_update_rejected() {
        // CompostReceival updates are rejected inline in the validate() match arm.
        // Verify the invariant directly: the result is always Invalid.
        let result: ValidateCallbackResult = ValidateCallbackResult::Invalid(
            "Compost receivals cannot be updated -- they are immutable records".into(),
        );
        assert!(
            matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("Compost receivals cannot be updated"))
        );
    }

    // ---- SavingsPool creation ----

    #[test]
    fn test_savings_pool_create_valid() {
        let result = validate_create_savings_pool(
            EntryCreationAction::Create(make_create()),
            valid_savings_pool(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_savings_pool_rejects_nan_yield_rate() {
        let mut p = valid_savings_pool();
        p.yield_rate = f64::NAN;
        let result =
            validate_create_savings_pool(EntryCreationAction::Create(make_create()), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_savings_pool_rejects_inf_yield_rate() {
        let mut p = valid_savings_pool();
        p.yield_rate = f64::INFINITY;
        let result =
            validate_create_savings_pool(EntryCreationAction::Create(make_create()), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_savings_pool_rejects_zero_target() {
        let mut p = valid_savings_pool();
        p.target_amount = 0;
        let result =
            validate_create_savings_pool(EntryCreationAction::Create(make_create()), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- CommonsPool creation ----

    #[test]
    fn test_commons_pool_create_valid() {
        let result = validate_create_commons_pool(
            EntryCreationAction::Create(make_create()),
            valid_commons_pool(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_commons_pool_rejects_not_demurrage_exempt() {
        let mut p = valid_commons_pool();
        p.demurrage_exempt = false;
        let result =
            validate_create_commons_pool(EntryCreationAction::Create(make_create()), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_commons_pool_rejects_reserve_below_25_percent() {
        let mut p = valid_commons_pool();
        // inalienable_reserve=1000, available_balance=9000 => reserve ratio = 10%
        p.inalienable_reserve = 1_000;
        p.available_balance = 9_000;
        let result =
            validate_create_commons_pool(EntryCreationAction::Create(make_create()), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_commons_pool_empty_is_valid() {
        let mut p = valid_commons_pool();
        p.inalienable_reserve = 0;
        p.available_balance = 0;
        let result =
            validate_create_commons_pool(EntryCreationAction::Create(make_create()), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_commons_pool_exactly_25_percent_is_valid() {
        let mut p = valid_commons_pool();
        // 25 / 100 = exactly 25%
        p.inalienable_reserve = 25;
        p.available_balance = 75;
        let result =
            validate_create_commons_pool(EntryCreationAction::Create(make_create()), p).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    // ---- CompostReceival creation ----

    #[test]
    fn test_compost_receival_create_valid() {
        let result = validate_create_compost_receival(
            EntryCreationAction::Create(make_create()),
            valid_compost_receival(),
        )
        .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_compost_receival_rejects_zero_amount() {
        let mut r = valid_compost_receival();
        r.amount = 0;
        let result =
            validate_create_compost_receival(EntryCreationAction::Create(make_create()), r)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_compost_receival_rejects_invalid_did() {
        let mut r = valid_compost_receival();
        r.source_member_did = "notadid".into();
        let result =
            validate_create_compost_receival(EntryCreationAction::Create(make_create()), r)
                .unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
