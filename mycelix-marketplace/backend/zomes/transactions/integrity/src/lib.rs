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
