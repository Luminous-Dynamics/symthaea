// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Treasury Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Treasury {
    pub id: String,
    pub name: String,
    pub description: String,
    pub currency: String,
    pub balance: f64,
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
    pub amount: f64,
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
    pub amount: f64,
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
    pub target_amount: f64,
    pub current_amount: f64,
    pub currency: String,
    pub members: Vec<String>,
    pub yield_rate: f64,
    pub created: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Treasury(Treasury),
    Contribution(Contribution),
    Allocation(Allocation),
    SavingsPool(SavingsPool),
}

#[hdk_link_types]
pub enum LinkTypes {
    TreasuryToContributions,
    TreasuryToAllocations,
    TreasuryToPools,
    ManagerToTreasury,
    ContributorToContributions,
    MemberToPool,
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
    if !contribution.contributor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Contributor must be a valid DID".into(),
        ));
    }
    if contribution.amount <= 0.0 {
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
    if !allocation.recipient_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Recipient must be a valid DID".into(),
        ));
    }
    if allocation.amount <= 0.0 {
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
    if allocation.amount <= 0.0 {
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
    if pool.target_amount <= 0.0 {
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
    if pool.target_amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Target amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
