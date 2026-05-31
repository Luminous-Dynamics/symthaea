// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Circles Integrity Zome
//!
//! This zome defines entry types and validation rules for community credit circles
//! in the Mycelix Mutual Aid hApp. Implements mutual credit with automatic clearing.

use hdi::prelude::*;
use mutualaid_common::*;

/// Entry types for the circles zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A mutual credit circle
    #[entry_type(visibility = "public")]
    CreditCircle(CreditCircle),
    /// A member's credit line within a circle
    #[entry_type(visibility = "public")]
    CreditLine(CreditLine),
    /// A credit transaction
    #[entry_type(visibility = "public")]
    CreditTransaction(CreditTransaction),
    /// Balance snapshot
    #[entry_type(visibility = "public")]
    Balance(Balance),
}

/// Link types for the circles zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from circle to its members
    CircleToMembers,
    /// Link from member to their circles
    MemberToCircles,
    /// Link from member to their credit lines
    MemberToCreditLines,
    /// Link from circle to transactions
    CircleToTransactions,
    /// Link from member to their transactions
    MemberToTransactions,
    /// Link for all circles discovery
    AllCircles,
    /// Link from circle to latest balances
    CircleToBalances,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            tag,
            ..
        } => validate_create_link(link_type, base_address, target_address, tag),
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            let _ = link_type;
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate entry creation
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::CreditCircle(circle) => validate_credit_circle(circle),
        EntryTypes::CreditLine(line) => validate_credit_line(line),
        EntryTypes::CreditTransaction(tx) => validate_credit_transaction(tx),
        EntryTypes::Balance(balance) => validate_balance(balance),
    }
}

/// Validate a credit circle
fn validate_credit_circle(circle: CreditCircle) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if circle.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle ID cannot be empty".to_string(),
        ));
    }

    // Name must not be empty
    if circle.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name cannot be empty".to_string(),
        ));
    }

    // Name length limit
    if circle.name.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name cannot exceed 100 characters".to_string(),
        ));
    }

    // Description length limit
    if circle.description.len() > 2000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle description cannot exceed 2000 characters".to_string(),
        ));
    }

    // Currency name must not be empty
    if circle.currency_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency name cannot be empty".to_string(),
        ));
    }

    // Currency symbol must not be empty
    if circle.currency_symbol.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency symbol cannot be empty".to_string(),
        ));
    }

    // Default credit limit must be positive
    if circle.default_credit_limit <= 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Default credit limit must be positive".to_string(),
        ));
    }

    // Max credit limit must be >= default
    if circle.max_credit_limit < circle.default_credit_limit {
        return Ok(ValidateCallbackResult::Invalid(
            "Max credit limit cannot be less than default".to_string(),
        ));
    }

    // Transaction fee must be non-negative
    if circle.transaction_fee_percent < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction fee cannot be negative".to_string(),
        ));
    }

    // Fee must be reasonable (max 10%)
    if circle.transaction_fee_percent > 10.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction fee cannot exceed 10%".to_string(),
        ));
    }

    // Demurrage rate must be non-negative
    if circle.demurrage_rate_percent < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Demurrage rate cannot be negative".to_string(),
        ));
    }

    // Must have at least one founder
    if circle.founders.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle must have at least one founder".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a credit line
fn validate_credit_line(line: CreditLine) -> ExternResult<ValidateCallbackResult> {
    // Credit limit must be positive
    if line.credit_limit <= 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit limit must be positive".to_string(),
        ));
    }

    // Balance cannot exceed credit limit (too negative)
    if line.balance < -(line.credit_limit as i64) {
        return Ok(ValidateCallbackResult::Invalid(
            "Balance exceeds credit limit".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a credit transaction
fn validate_credit_transaction(tx: CreditTransaction) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if tx.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction ID cannot be empty".to_string(),
        ));
    }

    // Amount must be positive (transfer direction determined by from/to)
    if tx.amount <= 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction amount must be positive".to_string(),
        ));
    }

    // From and to must be different
    if tx.from == tx.to {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transfer to yourself".to_string(),
        ));
    }

    // Memo length limit
    if tx.memo.len() > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction memo cannot exceed 500 characters".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a balance entry
fn validate_balance(balance: Balance) -> ExternResult<ValidateCallbackResult> {
    // Credit available cannot be negative
    if balance.credit_available < 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit available cannot be negative".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    _tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::CircleToMembers => Ok(ValidateCallbackResult::Valid),
        LinkTypes::MemberToCircles => Ok(ValidateCallbackResult::Valid),
        LinkTypes::MemberToCreditLines => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CircleToTransactions => Ok(ValidateCallbackResult::Valid),
        LinkTypes::MemberToTransactions => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllCircles => Ok(ValidateCallbackResult::Valid),
        LinkTypes::CircleToBalances => Ok(ValidateCallbackResult::Valid),
    }
}
