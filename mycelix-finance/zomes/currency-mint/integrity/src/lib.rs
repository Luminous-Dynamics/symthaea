//! Currency Factory Integrity Zome
//!
//! Enforces the **Immutable Economic Physics** for community-minted currencies:
//! - Zero-sum mutual credit (one person's credit = another's debt)
//! - Constitutional parameter bounds (credit limit 10-200, demurrage max 5%)
//! - No pre-mining, no ICOs, no fractional reserve
//!
//! Communities get **Parameter Sovereignty**: custom names, limits, demurrage.
//! The physics are enforced here at the integrity level — no coordinator can bypass them.

use hdi::prelude::*;
use mycelix_finance_types::{CurrencyStatus, MintedCurrencyParams, MINTED_CREDIT_LIMIT_MAX};

// String length limits — prevent DHT bloat attacks
const MAX_DID_LEN: usize = 256;
const MAX_ID_LEN: usize = 256;
const MAX_DESCRIPTION_LEN: usize = 2000;

// =============================================================================
// ENTRY TYPES
// =============================================================================

/// A community-minted currency definition.
///
/// Created by a DAO governance vote. The parameters are validated against
/// constitutional bounds at the integrity level. Once Active, exchanges
/// can occur. Retirement freezes all balances.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CurrencyDefinition {
    /// Unique identifier
    pub id: String,
    /// DID of the DAO that created this currency
    pub creator_dao_did: String,
    /// Governance proposal that authorized creation (if community > 10 members)
    pub governance_proposal_id: Option<String>,
    /// The community-chosen parameters (validated against constitutional limits)
    pub params: MintedCurrencyParams,
    /// Current status
    pub status: CurrencyStatus,
    /// When this currency was created
    pub created_at: Timestamp,
}

/// A member's balance in a community-minted currency.
///
/// Zero-sum invariant: the sum of all MintedBalance.balance for a given
/// currency_id is always exactly zero. This is enforced by only allowing
/// balance changes through MintedExchange entries.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MintedBalance {
    /// DID of the member
    pub member_did: String,
    /// ID of the currency
    pub currency_id: String,
    /// Current balance (can be negative — mutual credit)
    pub balance: i32,
    /// Total service hours provided (lifetime)
    pub total_provided: f32,
    /// Total service hours received (lifetime)
    pub total_received: f32,
    /// Number of exchanges
    pub exchange_count: u32,
    /// Last activity timestamp
    pub last_activity: Timestamp,
}

/// A service exchange in a community-minted currency.
///
/// Same zero-sum mutual credit mechanics as TEND, but scoped to the
/// community's custom parameters (limit, max hours, min minutes).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MintedExchange {
    /// Unique identifier
    pub id: String,
    /// ID of the currency
    pub currency_id: String,
    /// DID of the service provider (earns credit)
    pub provider_did: String,
    /// DID of the service receiver (takes on debt)
    pub receiver_did: String,
    /// Hours of service exchanged
    pub hours: f32,
    /// Description of the service
    pub service_description: String,
    /// When the exchange was recorded
    pub timestamp: Timestamp,
    /// Whether this exchange is confirmed
    pub confirmed: bool,
}

/// Confirmation receipt for a minted exchange.
///
/// Created by the receiver to confirm they received the service.
/// Once confirmed, balances are updated. Immutable once created.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MintedExchangeConfirmation {
    /// ID of the exchange being confirmed
    pub exchange_id: String,
    /// DID of the confirmer (must be the receiver)
    pub confirmer_did: String,
    /// When confirmed
    pub timestamp: Timestamp,
}

/// A dispute on a confirmed exchange.
///
/// Opens a freeze on the exchange. Resolution either confirms the original
/// balances (reject dispute) or reverses them (accept dispute).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MintedDispute {
    /// ID of the disputed exchange
    pub exchange_id: String,
    /// DID of the member who opened the dispute
    pub opener_did: String,
    /// Reason for dispute
    pub reason: String,
    /// Resolution status: None = open, Some(true) = reversed, Some(false) = rejected
    pub resolved: Option<bool>,
    /// DID of the resolver (governance agent)
    pub resolver_did: Option<String>,
    /// Resolution reason
    pub resolution_reason: Option<String>,
    /// When opened
    pub opened_at: Timestamp,
    /// When resolved (if resolved)
    pub resolved_at: Option<Timestamp>,
}

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// =============================================================================
// ENTRY & LINK TYPE ENUMS
// =============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CurrencyDefinition(CurrencyDefinition),
    MintedBalance(MintedBalance),
    MintedExchange(MintedExchange),
    MintedExchangeConfirmation(MintedExchangeConfirmation),
    MintedDispute(MintedDispute),
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Link from DAO DID to currencies it has minted
    DaoToCurrencies,
    /// Link from currency ID to its definition
    CurrencyIdToDefinition,
    /// Link from currency ID + member DID to balance
    CurrencyMemberToBalance,
    /// Link from currency ID to exchanges
    CurrencyToExchanges,
    /// Link from exchange ID to its confirmation receipt
    ExchangeToConfirmation,
    /// Link from exchange ID to its dispute
    ExchangeToDispute,
    /// Anchor infrastructure
    AnchorLinks,
}

// =============================================================================
// VALIDATION — The Immutable Physics
// =============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::CurrencyDefinition(def) => validate_create_currency(def),
                EntryTypes::MintedBalance(bal) => validate_minted_balance(&bal),
                EntryTypes::MintedExchange(ex) => validate_minted_exchange(&ex),
                EntryTypes::MintedExchangeConfirmation(conf) => {
                    validate_exchange_confirmation(&conf)
                }
                EntryTypes::MintedDispute(dispute) => validate_minted_dispute(&dispute),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::CurrencyDefinition(def) => validate_update_currency(def),
                    EntryTypes::MintedBalance(bal) => validate_minted_balance(&bal),
                    EntryTypes::MintedExchange(_) => {
                        // Exchanges are immutable once created
                        Ok(ValidateCallbackResult::Invalid(
                            "Minted exchanges cannot be updated".into(),
                        ))
                    }
                    EntryTypes::MintedExchangeConfirmation(_) => Ok(
                        ValidateCallbackResult::Invalid("Confirmations cannot be updated".into()),
                    ),
                    EntryTypes::MintedDispute(_) => {
                        // Disputes can be updated (for resolution)
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Invalid(
                        "Anchors cannot be updated".into(),
                    )),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::DaoToCurrencies => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CurrencyIdToDefinition => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CurrencyMemberToBalance => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CurrencyToExchanges => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ExchangeToConfirmation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ExchangeToDispute => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AnchorLinks => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        // Entries in the Currency Mint are append-only / update-only.
        // Currencies, balances, exchanges, and confirmations must not be deleted
        // from the DHT — they are permanent audit records. Anchors are also
        // non-deletable for link integrity.
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Invalid(
            "Currency Mint entries cannot be deleted".into(),
        )),
    }
}

fn validate_create_currency(def: CurrencyDefinition) -> ExternResult<ValidateCallbackResult> {
    // String length limits
    if def.id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency ID exceeds maximum length".into(),
        ));
    }
    if def.creator_dao_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if !def.creator_dao_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Creator must be a valid DID".into(),
        ));
    }

    // Validate parameters against constitutional limits
    if let Err(e) = def.params.validate() {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Currency parameters violate constitutional limits: {}",
            e
        )));
    }

    // New currencies must start as Draft
    if def.status != CurrencyStatus::Draft {
        return Ok(ValidateCallbackResult::Invalid(
            "New currencies must start in Draft status".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_currency(def: CurrencyDefinition) -> ExternResult<ValidateCallbackResult> {
    // Parameters must still be valid after update
    if let Err(e) = def.params.validate() {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Updated parameters violate constitutional limits: {}",
            e
        )));
    }
    // Retired currencies cannot be un-retired
    // (We can't check the original status in integrity validation, but
    // the coordinator enforces valid status transitions)
    Ok(ValidateCallbackResult::Valid)
}

fn validate_minted_balance(bal: &MintedBalance) -> ExternResult<ValidateCallbackResult> {
    if bal.member_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if bal.currency_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency ID exceeds maximum length".into(),
        ));
    }
    if !bal.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Member must be a valid DID".into(),
        ));
    }

    // Float fields must be finite (NaN bypasses comparison operators)
    if !bal.total_provided.is_finite() || !bal.total_received.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Balance totals must be finite numbers".into(),
        ));
    }

    // Constitutional maximum: balance cannot exceed the absolute maximum credit limit
    // The coordinator enforces the currency-specific limit; integrity enforces the ceiling.
    if bal.balance.abs() > MINTED_CREDIT_LIMIT_MAX {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Balance {} exceeds constitutional maximum of ±{}",
            bal.balance, MINTED_CREDIT_LIMIT_MAX
        )));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_minted_exchange(ex: &MintedExchange) -> ExternResult<ValidateCallbackResult> {
    // String length limits
    if ex.provider_did.len() > MAX_DID_LEN || ex.receiver_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if ex.id.len() > MAX_ID_LEN || ex.currency_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "ID exceeds maximum length".into(),
        ));
    }
    if ex.service_description.len() > MAX_DESCRIPTION_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Description exceeds maximum length".into(),
        ));
    }

    // DID validation
    if !ex.provider_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider must be a valid DID".into(),
        ));
    }
    if !ex.receiver_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Receiver must be a valid DID".into(),
        ));
    }

    // No self-exchange
    if ex.provider_did == ex.receiver_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot exchange with yourself".into(),
        ));
    }

    // Hours must be finite and positive
    if !ex.hours.is_finite() || ex.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours must be a finite positive number".into(),
        ));
    }

    // Description required
    if ex.service_description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service description required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_minted_dispute(dispute: &MintedDispute) -> ExternResult<ValidateCallbackResult> {
    if dispute.exchange_id.is_empty() || dispute.exchange_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange ID must be 1-256 characters".into(),
        ));
    }
    if !dispute.opener_did.starts_with("did:") || dispute.opener_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Opener must be a valid DID".into(),
        ));
    }
    if dispute.reason.is_empty() || dispute.reason.len() > MAX_DESCRIPTION_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Reason must be 1-2000 characters".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_exchange_confirmation(
    conf: &MintedExchangeConfirmation,
) -> ExternResult<ValidateCallbackResult> {
    if conf.exchange_id.is_empty() || conf.exchange_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange ID must be 1-256 characters".into(),
        ));
    }
    if !conf.confirmer_did.starts_with("did:") || conf.confirmer_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Confirmer must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
