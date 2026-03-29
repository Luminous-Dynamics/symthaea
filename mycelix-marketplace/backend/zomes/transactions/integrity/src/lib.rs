// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;

/// Transaction entry - represents a purchase in the marketplace
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Transaction {
    /// Buyer agent
    pub buyer: AgentPubKey,

    /// Seller agent
    pub seller: AgentPubKey,

    /// Listing being purchased
    pub listing_hash: ActionHash,

    /// Quantity purchased
    pub quantity: u32,

    /// Total price in cents (quantity * unit_price)
    pub total_price_cents: u64,

    /// Current status
    pub status: TransactionStatus,

    /// Creation timestamp
    pub created_at: Timestamp,

    /// Last update timestamp
    pub updated_at: Timestamp,

    /// Delivery tracking (optional)
    pub tracking_info: Option<String>,

    /// Epistemic classification
    /// Transactions are N1 (communal) agreements between buyer-seller
    pub epistemic: EpistemicClassification,
}

/// Transaction lifecycle states
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TransactionStatus {
    /// Buyer created, awaiting seller confirmation
    Pending,

    /// Seller confirmed, payment/arrangement made
    Confirmed,

    /// Seller marked as shipped
    Shipped,

    /// Buyer confirmed delivery
    Delivered,

    /// Transaction completed successfully (triggers MATL update)
    Completed,

    /// Disputed by either party
    Disputed,

    /// Cancelled before completion
    Cancelled,
}

/// Epistemic classification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicClassification {
    pub empirical: EmpiricalLevel,
    pub normative: NormativeLevel,
    pub materiality: MaterialityLevel,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EmpiricalLevel {
    E0Null,
    E1Testimonial,
    E2PrivateVerify,
    E3Cryptographic,
    E4PublicRepro,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum NormativeLevel {
    N0Personal,
    N1Communal,
    N2Network,
    N3Axiomatic,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MaterialityLevel {
    M0Ephemeral,
    M1Temporal,
    M2Persistent,
    M3Foundational,
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Buyer -> Transactions
    BuyerToTransactions,

    /// Seller -> Transactions
    SellerToTransactions,

    /// Listing -> Transactions
    ListingToTransactions,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Transaction(Transaction),
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Transaction(transaction) => validate_transaction(&transaction),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_transaction(transaction: &Transaction) -> ExternResult<ValidateCallbackResult> {
    // Quantity validation
    if transaction.quantity == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be at least 1".into(),
        ));
    }

    // Price validation
    if transaction.total_price_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total price must be greater than zero".into(),
        ));
    }

    // Buyer and seller must be different
    if transaction.buyer == transaction.seller {
        return Ok(ValidateCallbackResult::Invalid(
            "Buyer and seller must be different agents".into(),
        ));
    }

    // Transactions should be N1 (communal - buyer-seller agreement)
    if transaction.epistemic.normative != NormativeLevel::N1Communal {
        return Ok(ValidateCallbackResult::Invalid(
            "Transactions must be N1 (communal agreement)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_agent(byte: u8) -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![byte; 36])
    }

    fn valid_transaction() -> Transaction {
        Transaction {
            buyer: mock_agent(1),
            seller: mock_agent(2),
            listing_hash: ActionHash::from_raw_36(vec![3u8; 36]),
            quantity: 1,
            total_price_cents: 1999,
            status: TransactionStatus::Pending,
            created_at: Timestamp::from_micros(1000000),
            updated_at: Timestamp::from_micros(1000000),
            tracking_info: None,
            epistemic: EpistemicClassification {
                empirical: EmpiricalLevel::E1Testimonial,
                normative: NormativeLevel::N1Communal,
                materiality: MaterialityLevel::M1Temporal,
            },
        }
    }

    #[test]
    fn test_validate_transaction_valid() {
        let tx = valid_transaction();
        let result = validate_transaction(&tx).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_transaction_zero_quantity() {
        let tx = Transaction { quantity: 0, ..valid_transaction() };
        let result = validate_transaction(&tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_transaction_zero_price() {
        let tx = Transaction { total_price_cents: 0, ..valid_transaction() };
        let result = validate_transaction(&tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_transaction_same_buyer_seller() {
        let tx = Transaction { buyer: mock_agent(1), seller: mock_agent(1), ..valid_transaction() };
        let result = validate_transaction(&tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_transaction_wrong_normative_level() {
        let tx = Transaction {
            epistemic: EpistemicClassification {
                empirical: EmpiricalLevel::E1Testimonial,
                normative: NormativeLevel::N0Personal,
                materiality: MaterialityLevel::M1Temporal,
            },
            ..valid_transaction()
        };
        let result = validate_transaction(&tx).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_all_transaction_statuses() {
        let statuses = vec![
            TransactionStatus::Pending, TransactionStatus::Confirmed,
            TransactionStatus::Shipped, TransactionStatus::Delivered,
            TransactionStatus::Completed, TransactionStatus::Disputed,
            TransactionStatus::Cancelled,
        ];
        assert_eq!(statuses.len(), 7);
    }

    #[test]
    fn test_transaction_serde_roundtrip() {
        let tx = valid_transaction();
        let json = serde_json::to_string(&tx).unwrap();
        let parsed: Transaction = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.quantity, tx.quantity);
        assert_eq!(parsed.total_price_cents, tx.total_price_cents);
    }

    #[test]
    fn test_validate_transaction_large_quantity() {
        let tx = Transaction { quantity: 1_000_000, total_price_cents: 1_000_000_000, ..valid_transaction() };
        let result = validate_transaction(&tx).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_transaction_min_valid() {
        let tx = Transaction { quantity: 1, total_price_cents: 1, ..valid_transaction() };
        let result = validate_transaction(&tx).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }
}
