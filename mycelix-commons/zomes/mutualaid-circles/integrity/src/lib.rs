// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Circles Integrity Zome
//!
//! This zome defines entry types and validation rules for community credit circles
//! in the Mycelix Mutual Aid hApp. Implements mutual credit with automatic clearing.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};
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
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
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
    if circle.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle ID cannot be empty".to_string(),
        ));
    }

    // ID length limit
    if circle.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle ID exceeds 256 character limit".to_string(),
        ));
    }

    // Name must not be empty
    if circle.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name cannot be empty".to_string(),
        ));
    }

    // Name length limit
    if circle.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name exceeds 256 character limit".to_string(),
        ));
    }

    // Description length limit
    if circle.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle description exceeds 4096 character limit".to_string(),
        ));
    }

    // Currency name must not be empty
    if circle.currency_name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency name cannot be empty".to_string(),
        ));
    }

    // Currency name length limit
    if circle.currency_name.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency name exceeds 128 character limit".to_string(),
        ));
    }

    // Currency symbol must not be empty
    if circle.currency_symbol.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency symbol cannot be empty".to_string(),
        ));
    }

    // Currency symbol length limit
    if circle.currency_symbol.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency symbol exceeds 128 character limit".to_string(),
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

    // Geographic scope length limit
    if let Some(ref scope) = circle.geographic_scope {
        if scope.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Geographic scope exceeds 256 character limit".to_string(),
            ));
        }
    }

    // Must have at least one founder
    if circle.founders.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle must have at least one founder".to_string(),
        ));
    }

    // Founders limit
    if circle.founders.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 100 founders".to_string(),
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
    if line.balance < -line.credit_limit {
        return Ok(ValidateCallbackResult::Invalid(
            "Balance exceeds credit limit".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a credit transaction
fn validate_credit_transaction(tx: CreditTransaction) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if tx.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction ID cannot be empty".to_string(),
        ));
    }

    // ID length limit
    if tx.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction ID exceeds 256 character limit".to_string(),
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
    if tx.memo.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction memo exceeds 4096 character limit".to_string(),
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
    tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::CircleToMembers => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "CircleToMembers link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::MemberToCircles => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "MemberToCircles link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::MemberToCreditLines => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "MemberToCreditLines link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::CircleToTransactions => {
            // Transactions may carry metadata in tags
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "CircleToTransactions link tag too long (max 512 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::MemberToTransactions => {
            if tag.0.len() > 512 {
                return Ok(ValidateCallbackResult::Invalid(
                    "MemberToTransactions link tag too long (max 512 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::AllCircles => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "AllCircles link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::CircleToBalances => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "CircleToBalances link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // FACTORY FUNCTIONS
    // =============================================================================

    fn agent1() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdb; 36])
    }

    fn agent2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdc; 36])
    }

    fn action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xab; 36])
    }

    fn timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn valid_credit_circle() -> CreditCircle {
        CreditCircle {
            id: "circle-1".to_string(),
            name: "Community Credits".to_string(),
            description: "A local credit circle".to_string(),
            currency_name: "Credits".to_string(),
            currency_symbol: "CR".to_string(),
            default_credit_limit: 1000,
            max_credit_limit: 5000,
            transaction_fee_percent: 0.5,
            demurrage_rate_percent: 1.0,
            geographic_scope: Some("Local".to_string()),
            founders: vec![agent1()],
            rules_hash: None,
            created_at: timestamp(),
            active: true,
        }
    }

    fn valid_credit_line() -> CreditLine {
        CreditLine {
            circle_hash: action_hash(),
            member: agent1(),
            credit_limit: 1000,
            balance: 0,
            total_credit_extended: 0,
            total_credit_received: 0,
            joined_at: timestamp(),
            status: mutualaid_common::CreditLineStatus::Active,
            last_activity: timestamp(),
        }
    }

    fn valid_credit_transaction() -> CreditTransaction {
        CreditTransaction {
            id: "tx-1".to_string(),
            circle_hash: action_hash(),
            from: agent1(),
            to: agent2(),
            amount: 100,
            transaction_type: mutualaid_common::TransactionType::Payment,
            memo: "Payment for service".to_string(),
            related_exchange_hash: None,
            created_at: timestamp(),
            confirmed: false,
        }
    }

    fn valid_balance() -> Balance {
        Balance {
            member: agent1(),
            circle_hash: action_hash(),
            balance: 0,
            credit_available: 1000,
            as_of: timestamp(),
        }
    }

    // =============================================================================
    // CREDIT CIRCLE VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_valid_credit_circle() {
        let circle = valid_credit_circle();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_circle_empty_id() {
        let mut circle = valid_credit_circle();
        circle.id = "".to_string();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_empty_name() {
        let mut circle = valid_credit_circle();
        circle.name = "".to_string();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_name_too_long() {
        let mut circle = valid_credit_circle();
        circle.name = "a".repeat(257);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_name_at_limit() {
        let mut circle = valid_credit_circle();
        circle.name = "a".repeat(256);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_circle_description_too_long() {
        let mut circle = valid_credit_circle();
        circle.description = "a".repeat(4097);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_description_at_limit() {
        let mut circle = valid_credit_circle();
        circle.description = "a".repeat(4096);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_circle_empty_description() {
        let mut circle = valid_credit_circle();
        circle.description = "".to_string();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_circle_empty_currency_name() {
        let mut circle = valid_credit_circle();
        circle.currency_name = "".to_string();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_empty_currency_symbol() {
        let mut circle = valid_credit_circle();
        circle.currency_symbol = "".to_string();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_zero_default_limit() {
        let mut circle = valid_credit_circle();
        circle.default_credit_limit = 0;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_negative_default_limit() {
        let mut circle = valid_credit_circle();
        circle.default_credit_limit = -100;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_default_limit_one() {
        let mut circle = valid_credit_circle();
        circle.default_credit_limit = 1;
        circle.max_credit_limit = 1;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_circle_max_less_than_default() {
        let mut circle = valid_credit_circle();
        circle.default_credit_limit = 1000;
        circle.max_credit_limit = 999;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_max_equals_default() {
        let mut circle = valid_credit_circle();
        circle.default_credit_limit = 1000;
        circle.max_credit_limit = 1000;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_circle_negative_fee() {
        let mut circle = valid_credit_circle();
        circle.transaction_fee_percent = -0.1;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_fee_zero() {
        let mut circle = valid_credit_circle();
        circle.transaction_fee_percent = 0.0;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_circle_fee_exactly_10() {
        let mut circle = valid_credit_circle();
        circle.transaction_fee_percent = 10.0;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_circle_fee_over_10() {
        let mut circle = valid_credit_circle();
        circle.transaction_fee_percent = 10.001;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_fee_exactly_10_plus_epsilon() {
        let mut circle = valid_credit_circle();
        circle.transaction_fee_percent = 10.1;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_negative_demurrage() {
        let mut circle = valid_credit_circle();
        circle.demurrage_rate_percent = -0.1;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_demurrage_zero() {
        let mut circle = valid_credit_circle();
        circle.demurrage_rate_percent = 0.0;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_circle_no_founders() {
        let mut circle = valid_credit_circle();
        circle.founders = vec![];
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_multiple_founders() {
        let mut circle = valid_credit_circle();
        circle.founders = vec![agent1(), agent2()];
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    // =============================================================================
    // CREDIT LINE VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_valid_credit_line() {
        let line = valid_credit_line();
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_line_zero_limit() {
        let mut line = valid_credit_line();
        line.credit_limit = 0;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_line_negative_limit() {
        let mut line = valid_credit_line();
        line.credit_limit = -100;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_line_limit_one() {
        let mut line = valid_credit_line();
        line.credit_limit = 1;
        line.balance = 0;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_line_positive_balance() {
        let mut line = valid_credit_line();
        line.balance = 500;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_line_negative_balance_within_limit() {
        let mut line = valid_credit_line();
        line.balance = -500;
        line.credit_limit = 1000;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_line_balance_exactly_at_limit() {
        let mut line = valid_credit_line();
        line.balance = -1000;
        line.credit_limit = 1000;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_line_balance_exceeds_limit() {
        let mut line = valid_credit_line();
        line.balance = -1001;
        line.credit_limit = 1000;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_line_balance_exceeds_limit_by_one() {
        let mut line = valid_credit_line();
        line.balance = -1000 - 1;
        line.credit_limit = 1000;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_line_very_negative_balance() {
        let mut line = valid_credit_line();
        line.balance = -10000;
        line.credit_limit = 1000;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =============================================================================
    // CREDIT TRANSACTION VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_valid_credit_transaction() {
        let tx = valid_credit_transaction();
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_transaction_empty_id() {
        let mut tx = valid_credit_transaction();
        tx.id = "".to_string();
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_transaction_zero_amount() {
        let mut tx = valid_credit_transaction();
        tx.amount = 0;
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_transaction_negative_amount() {
        let mut tx = valid_credit_transaction();
        tx.amount = -100;
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_transaction_amount_one() {
        let mut tx = valid_credit_transaction();
        tx.amount = 1;
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_transaction_self_transfer() {
        let mut tx = valid_credit_transaction();
        tx.from = agent1();
        tx.to = agent1();
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_transaction_different_agents() {
        let mut tx = valid_credit_transaction();
        tx.from = agent1();
        tx.to = agent2();
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_transaction_empty_memo() {
        let mut tx = valid_credit_transaction();
        tx.memo = "".to_string();
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_transaction_memo_too_long() {
        let mut tx = valid_credit_transaction();
        tx.memo = "a".repeat(4097);
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_transaction_memo_at_limit() {
        let mut tx = valid_credit_transaction();
        tx.memo = "a".repeat(4096);
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_transaction_large_amount() {
        let mut tx = valid_credit_transaction();
        tx.amount = i64::MAX;
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    // =============================================================================
    // BALANCE VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_valid_balance() {
        let balance = valid_balance();
        let result = validate_balance(balance).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_balance_negative_credit_available() {
        let mut balance = valid_balance();
        balance.credit_available = -1;
        let result = validate_balance(balance).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_balance_zero_credit_available() {
        let mut balance = valid_balance();
        balance.credit_available = 0;
        let result = validate_balance(balance).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_balance_large_credit_available() {
        let mut balance = valid_balance();
        balance.credit_available = i64::MAX;
        let result = validate_balance(balance).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_balance_negative_balance_positive_credit() {
        let mut balance = valid_balance();
        balance.balance = -500;
        balance.credit_available = 500;
        let result = validate_balance(balance).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_balance_positive_balance() {
        let mut balance = valid_balance();
        balance.balance = 1000;
        balance.credit_available = 2000;
        let result = validate_balance(balance).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    // =============================================================================
    // EDGE CASE AND BOUNDARY TESTS
    // =============================================================================

    #[test]
    fn test_credit_circle_minimum_valid() {
        let circle = CreditCircle {
            id: "x".to_string(),
            name: "x".to_string(),
            description: "".to_string(),
            currency_name: "x".to_string(),
            currency_symbol: "x".to_string(),
            default_credit_limit: 1,
            max_credit_limit: 1,
            transaction_fee_percent: 0.0,
            demurrage_rate_percent: 0.0,
            geographic_scope: None,
            founders: vec![agent1()],
            rules_hash: None,
            created_at: timestamp(),
            active: true,
        };
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_transaction_reversed_agents() {
        let tx1 = valid_credit_transaction();

        let mut tx2 = valid_credit_transaction();
        tx2.from = agent2();
        tx2.to = agent1();

        let result1 = validate_credit_transaction(tx1).unwrap();
        let result2 = validate_credit_transaction(tx2).unwrap();

        assert!(matches!(result1, ValidateCallbackResult::Valid));
        assert!(matches!(result2, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_credit_line_extreme_negative_balance() {
        let mut line = valid_credit_line();
        line.balance = i64::MIN;
        line.credit_limit = 1000;
        let result = validate_credit_line(line).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_credit_circle_fee_boundary_values() {
        // Test 0.0
        let mut circle = valid_credit_circle();
        circle.transaction_fee_percent = 0.0;
        assert!(matches!(
            validate_credit_circle(circle.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Test 5.0 (mid-range)
        circle.transaction_fee_percent = 5.0;
        assert!(matches!(
            validate_credit_circle(circle.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Test 9.9999
        circle.transaction_fee_percent = 9.9999;
        assert!(matches!(
            validate_credit_circle(circle.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Test 10.0 (exactly at limit)
        circle.transaction_fee_percent = 10.0;
        assert!(matches!(
            validate_credit_circle(circle.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Test 10.0001 (just over limit)
        circle.transaction_fee_percent = 10.0001;
        assert!(matches!(
            validate_credit_circle(circle).unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_credit_line_various_balances() {
        let mut line = valid_credit_line();
        line.credit_limit = 1000;

        // Zero balance
        line.balance = 0;
        assert!(matches!(
            validate_credit_line(line.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Positive balance
        line.balance = 500;
        assert!(matches!(
            validate_credit_line(line.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Max positive
        line.balance = i64::MAX;
        assert!(matches!(
            validate_credit_line(line.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // At limit (negative)
        line.balance = -1000;
        assert!(matches!(
            validate_credit_line(line.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Just over limit
        line.balance = -1001;
        assert!(matches!(
            validate_credit_line(line).unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    #[test]
    fn test_balance_credit_available_edge_cases() {
        let mut balance = valid_balance();

        // Exactly zero
        balance.credit_available = 0;
        assert!(matches!(
            validate_balance(balance.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Small positive
        balance.credit_available = 1;
        assert!(matches!(
            validate_balance(balance.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Large positive
        balance.credit_available = 1_000_000_000;
        assert!(matches!(
            validate_balance(balance.clone()).unwrap(),
            ValidateCallbackResult::Valid
        ));

        // Small negative
        balance.credit_available = -1;
        assert!(matches!(
            validate_balance(balance).unwrap(),
            ValidateCallbackResult::Invalid(_)
        ));
    }

    // =============================================================================
    // LINK TAG VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_link_circle_to_members_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CircleToMembers, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_circle_to_members_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CircleToMembers, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_circle_to_transactions_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 512]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CircleToTransactions, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_circle_to_transactions_tag_too_long() {
        let tag = LinkTag(vec![0u8; 513]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CircleToTransactions, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_all_circles_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AllCircles, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_all_circles_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::AllCircles, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_member_to_circles_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::MemberToCircles, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_member_to_circles_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::MemberToCircles, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_member_to_credit_lines_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::MemberToCreditLines, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_member_to_credit_lines_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::MemberToCreditLines, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_member_to_transactions_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 512]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::MemberToTransactions, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_member_to_transactions_tag_too_long() {
        let tag = LinkTag(vec![0u8; 513]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::MemberToTransactions, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_circle_to_balances_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CircleToBalances, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_circle_to_balances_tag_too_long() {
        let tag = LinkTag(vec![0u8; 257]);
        let base = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let target = AnyLinkableHash::from(ActionHash::from_raw_36(vec![0u8; 36]));
        let result = validate_create_link(LinkTypes::CircleToBalances, base, target, tag);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // =============================================================================
    // STRING LENGTH LIMIT BOUNDARY TESTS
    // =============================================================================

    // -- Circle ID --

    #[test]
    fn test_validate_circle_id_at_limit() {
        let mut circle = valid_credit_circle();
        circle.id = "x".repeat(64);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_validate_circle_id_too_long() {
        let mut circle = valid_credit_circle();
        circle.id = "x".repeat(257);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_circle_id_whitespace() {
        let mut circle = valid_credit_circle();
        circle.id = "   ".to_string();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- Currency name --

    #[test]
    fn test_validate_circle_currency_name_at_limit() {
        let mut circle = valid_credit_circle();
        circle.currency_name = "c".repeat(128);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_validate_circle_currency_name_too_long() {
        let mut circle = valid_credit_circle();
        circle.currency_name = "c".repeat(129);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_circle_currency_name_whitespace() {
        let mut circle = valid_credit_circle();
        circle.currency_name = "  \t ".to_string();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- Currency symbol --

    #[test]
    fn test_validate_circle_currency_symbol_at_limit() {
        let mut circle = valid_credit_circle();
        circle.currency_symbol = "s".repeat(128);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_validate_circle_currency_symbol_too_long() {
        let mut circle = valid_credit_circle();
        circle.currency_symbol = "s".repeat(129);
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_circle_currency_symbol_whitespace() {
        let mut circle = valid_credit_circle();
        circle.currency_symbol = "  ".to_string();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- Geographic scope --

    #[test]
    fn test_validate_circle_geographic_scope_at_limit() {
        let mut circle = valid_credit_circle();
        circle.geographic_scope = Some("g".repeat(256));
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_validate_circle_geographic_scope_too_long() {
        let mut circle = valid_credit_circle();
        circle.geographic_scope = Some("g".repeat(257));
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_circle_geographic_scope_none() {
        let mut circle = valid_credit_circle();
        circle.geographic_scope = None;
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    // -- Founders Vec limit --

    #[test]
    fn test_validate_circle_founders_at_limit() {
        let mut circle = valid_credit_circle();
        circle.founders = (0..100)
            .map(|i| AgentPubKey::from_raw_36(vec![(i & 0xff) as u8; 36]))
            .collect();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_validate_circle_founders_too_many() {
        let mut circle = valid_credit_circle();
        circle.founders = (0..101)
            .map(|i| AgentPubKey::from_raw_36(vec![(i & 0xff) as u8; 36]))
            .collect();
        let result = validate_credit_circle(circle).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- Transaction ID --

    #[test]
    fn test_validate_transaction_id_at_limit() {
        let mut tx = valid_credit_transaction();
        tx.id = "t".repeat(64);
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_validate_transaction_id_too_long() {
        let mut tx = valid_credit_transaction();
        tx.id = "t".repeat(257);
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_transaction_id_whitespace() {
        let mut tx = valid_credit_transaction();
        tx.id = "  \t ".to_string();
        let result = validate_credit_transaction(tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
