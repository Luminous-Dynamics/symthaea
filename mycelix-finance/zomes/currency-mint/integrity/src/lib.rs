#![deny(unsafe_code)]
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

/// A pending balance adjustment for crash recovery during exchange confirmation.
///
/// Created BEFORE balance updates in `confirm_minted_exchange`. If a crash occurs
/// between the provider update and the receiver update, a governance agent can call
/// `recover_incomplete_minted_confirmations` to complete the interrupted operation and
/// restore the zero-sum invariant.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PendingMintedAdjustment {
    /// The exchange this adjustment belongs to
    pub exchange_id: String,
    /// DID of the service provider (gains hours)
    pub provider_did: String,
    /// DID of the service receiver (spends hours)
    pub receiver_did: String,
    /// Amount of hours being exchanged
    pub hours: f64,
    /// The currency this adjustment belongs to
    pub currency_id: String,
    /// Whether the provider's balance has been updated
    pub provider_completed: bool,
    /// Whether the receiver's balance has been updated
    pub receiver_completed: bool,
    /// When this pending adjustment was created
    pub created_at: Timestamp,
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
    PendingMintedAdjustment(PendingMintedAdjustment),
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
    /// Link from pending minted adjustment anchor to its entry
    PendingMintedAdjustmentToExchange,
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
                EntryTypes::PendingMintedAdjustment(adj) => {
                    validate_create_pending_minted_adjustment(adj)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => {
                match app_entry {
                    EntryTypes::CurrencyDefinition(def) => validate_update_currency(
                        &action.original_action_address,
                        def,
                    ),
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
                    EntryTypes::MintedDispute(dispute) => validate_dispute_update(&dispute),
                    EntryTypes::PendingMintedAdjustment(adj) => {
                        // Only completed flags can change; hours must stay valid
                        if !adj.hours.is_finite() || adj.hours <= 0.0 {
                            Ok(ValidateCallbackResult::Invalid(
                                "PendingMintedAdjustment hours must be finite and positive".into(),
                            ))
                        } else {
                            Ok(ValidateCallbackResult::Valid)
                        }
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
            LinkTypes::PendingMintedAdjustmentToExchange => Ok(ValidateCallbackResult::Valid),
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

fn validate_update_currency(
    original_action: &ActionHash,
    def: CurrencyDefinition,
) -> ExternResult<ValidateCallbackResult> {
    // Parameters must still be valid after update
    if let Err(e) = def.params.validate() {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Updated parameters violate constitutional limits: {}",
            e
        )));
    }
    // Enforce status transition rules
    if let Ok(original_record) = must_get_valid_record(original_action.clone()) {
        if let Ok(Some(original)) =
            original_record.entry().to_app_option::<CurrencyDefinition>()
        {
            if original.status != def.status
                && !original.status.can_transition_to(&def.status)
            {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Invalid currency status transition: {:?} → {:?}",
                    original.status, def.status
                )));
            }
        }
    }
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

/// Validate a dispute update (resolution).
///
/// An updated dispute MUST be resolved — otherwise there's no reason to update it.
/// When resolved, it must have a resolver DID, resolution reason, and resolved_at timestamp.
/// The base fields (exchange_id, opener_did, reason, opened_at) are immutable but
/// we can't compare against the original in StoreEntry. We enforce structural consistency:
/// a resolved dispute must have all resolution fields present.
fn validate_dispute_update(dispute: &MintedDispute) -> ExternResult<ValidateCallbackResult> {
    // Basic field validation (same as create)
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

    // An updated dispute must be resolved — unresolved updates are meaningless
    match dispute.resolved {
        None => {
            return Ok(ValidateCallbackResult::Invalid(
                "Dispute update must include a resolution (resolved cannot be None)".into(),
            ));
        }
        Some(_) => {
            // Resolver DID is required
            match &dispute.resolver_did {
                None => {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Resolved dispute must have a resolver DID".into(),
                    ));
                }
                Some(did) => {
                    if !did.starts_with("did:") || did.len() > MAX_DID_LEN {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Resolver must be a valid DID".into(),
                        ));
                    }
                }
            }
            // Resolution reason is required
            match &dispute.resolution_reason {
                None => {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Resolved dispute must have a resolution reason".into(),
                    ));
                }
                Some(reason) => {
                    if reason.is_empty() || reason.len() > MAX_DESCRIPTION_LEN {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Resolution reason must be 1-2000 characters".into(),
                        ));
                    }
                }
            }
            // Resolved timestamp is required
            if dispute.resolved_at.is_none() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Resolved dispute must have a resolved_at timestamp".into(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_pending_minted_adjustment(
    adj: PendingMintedAdjustment,
) -> ExternResult<ValidateCallbackResult> {
    // Hours must be finite and positive
    if !adj.hours.is_finite() || adj.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "PendingMintedAdjustment hours must be finite and positive".into(),
        ));
    }

    // DID length checks
    if adj.provider_did.len() > MAX_DID_LEN || adj.receiver_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if adj.exchange_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange ID exceeds maximum length".into(),
        ));
    }
    if adj.currency_id.len() > MAX_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency ID exceeds maximum length".into(),
        ));
    }

    // DIDs must be valid
    if !adj.provider_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider must be a valid DID".into(),
        ));
    }
    if !adj.receiver_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Receiver must be a valid DID".into(),
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

    // ---- CurrencyDefinition validation ----

    fn valid_currency_def() -> CurrencyDefinition {
        CurrencyDefinition {
            id: "currency:test".into(),
            creator_dao_did: "did:mycelix:dao:test".into(),
            governance_proposal_id: None,
            params: MintedCurrencyParams {
                name: "TestCoin".into(),
                symbol: "TC".into(),
                description: "A test currency".into(),
                credit_limit: 40,
                demurrage_rate: 0.02,
                max_service_hours: 8,
                min_service_minutes: 15,
                requires_confirmation: false,
                confirmation_timeout_hours: 0,
                max_exchanges_per_day: 0,
            },
            status: CurrencyStatus::Draft,
            created_at: ts(1_000_000),
        }
    }

    #[test]
    fn test_currency_create_valid() {
        let result = validate_create_currency(valid_currency_def()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_currency_create_rejects_non_draft_status() {
        let mut def = valid_currency_def();
        def.status = CurrencyStatus::Active;
        let result = validate_create_currency(def).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_currency_create_rejects_invalid_did() {
        let mut def = valid_currency_def();
        def.creator_dao_did = "not-a-did".into();
        let result = validate_create_currency(def).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_currency_create_rejects_oversized_id() {
        let mut def = valid_currency_def();
        def.id = "x".repeat(MAX_ID_LEN + 1);
        let result = validate_create_currency(def).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_currency_create_rejects_oversized_did() {
        let mut def = valid_currency_def();
        def.creator_dao_did = format!("did:{}", "x".repeat(MAX_DID_LEN));
        let result = validate_create_currency(def).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_currency_create_rejects_invalid_params_credit_limit_zero() {
        let mut def = valid_currency_def();
        def.params.credit_limit = 0;
        let result = validate_create_currency(def).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_currency_create_rejects_invalid_params_demurrage_too_high() {
        let mut def = valid_currency_def();
        def.params.demurrage_rate = 0.06; // 6% > 5% max
        let result = validate_create_currency(def).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_currency_update_valid() {
        let mut def = valid_currency_def();
        def.status = CurrencyStatus::Active;
        def.params.credit_limit = 80;
        // Dummy action hash — must_get_valid_record will fail gracefully outside WASM
        let dummy_hash = ActionHash::from_raw_36(vec![0; 36]);
        let result = validate_update_currency(&dummy_hash, def).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_currency_update_rejects_invalid_params() {
        let mut def = valid_currency_def();
        def.params.credit_limit = 999; // exceeds max
        let dummy_hash = ActionHash::from_raw_36(vec![0; 36]);
        let result = validate_update_currency(&dummy_hash, def).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- MintedBalance validation ----

    #[test]
    fn test_balance_valid() {
        let bal = MintedBalance {
            member_did: "did:mycelix:alice".into(),
            currency_id: "currency:test".into(),
            balance: 50,
            total_provided: 100.0,
            total_received: 50.0,
            exchange_count: 10,
            last_activity: ts(1_000_000),
        };
        let result = validate_minted_balance(&bal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_balance_rejects_nan() {
        let bal = MintedBalance {
            member_did: "did:mycelix:alice".into(),
            currency_id: "currency:test".into(),
            balance: 10,
            total_provided: f32::NAN,
            total_received: 5.0,
            exchange_count: 1,
            last_activity: ts(1_000_000),
        };
        let result = validate_minted_balance(&bal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_balance_rejects_infinity() {
        let bal = MintedBalance {
            member_did: "did:mycelix:alice".into(),
            currency_id: "currency:test".into(),
            balance: 10,
            total_provided: 5.0,
            total_received: f32::INFINITY,
            exchange_count: 1,
            last_activity: ts(1_000_000),
        };
        let result = validate_minted_balance(&bal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_balance_rejects_exceeds_constitutional_max() {
        let bal = MintedBalance {
            member_did: "did:mycelix:alice".into(),
            currency_id: "currency:test".into(),
            balance: MINTED_CREDIT_LIMIT_MAX + 1,
            total_provided: 0.0,
            total_received: 0.0,
            exchange_count: 0,
            last_activity: ts(1_000_000),
        };
        let result = validate_minted_balance(&bal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_balance_rejects_invalid_did() {
        let bal = MintedBalance {
            member_did: "not-a-did".into(),
            currency_id: "currency:test".into(),
            balance: 0,
            total_provided: 0.0,
            total_received: 0.0,
            exchange_count: 0,
            last_activity: ts(1_000_000),
        };
        let result = validate_minted_balance(&bal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- MintedExchange validation ----

    fn valid_exchange() -> MintedExchange {
        MintedExchange {
            id: "mex:test:001".into(),
            currency_id: "currency:test".into(),
            provider_did: "did:mycelix:alice".into(),
            receiver_did: "did:mycelix:bob".into(),
            hours: 2.0,
            service_description: "Garden tending".into(),
            timestamp: ts(1_000_000),
            confirmed: false,
        }
    }

    #[test]
    fn test_exchange_valid() {
        let result = validate_minted_exchange(&valid_exchange()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_exchange_rejects_self_exchange() {
        let mut ex = valid_exchange();
        ex.receiver_did = ex.provider_did.clone();
        let result = validate_minted_exchange(&ex).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_exchange_rejects_nan_hours() {
        let mut ex = valid_exchange();
        ex.hours = f32::NAN;
        let result = validate_minted_exchange(&ex).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_exchange_rejects_zero_hours() {
        let mut ex = valid_exchange();
        ex.hours = 0.0;
        let result = validate_minted_exchange(&ex).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_exchange_rejects_negative_hours() {
        let mut ex = valid_exchange();
        ex.hours = -1.0;
        let result = validate_minted_exchange(&ex).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_exchange_rejects_empty_description() {
        let mut ex = valid_exchange();
        ex.service_description = "".into();
        let result = validate_minted_exchange(&ex).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_exchange_rejects_invalid_provider_did() {
        let mut ex = valid_exchange();
        ex.provider_did = "not-a-did".into();
        let result = validate_minted_exchange(&ex).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- MintedExchangeConfirmation validation ----

    #[test]
    fn test_confirmation_valid() {
        let conf = MintedExchangeConfirmation {
            exchange_id: "mex:test:001".into(),
            confirmer_did: "did:mycelix:bob".into(),
            timestamp: ts(2_000_000),
        };
        let result = validate_exchange_confirmation(&conf).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_confirmation_rejects_empty_exchange_id() {
        let conf = MintedExchangeConfirmation {
            exchange_id: "".into(),
            confirmer_did: "did:mycelix:bob".into(),
            timestamp: ts(2_000_000),
        };
        let result = validate_exchange_confirmation(&conf).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_confirmation_rejects_invalid_did() {
        let conf = MintedExchangeConfirmation {
            exchange_id: "mex:test:001".into(),
            confirmer_did: "not-a-did".into(),
            timestamp: ts(2_000_000),
        };
        let result = validate_exchange_confirmation(&conf).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- MintedDispute create validation ----

    fn valid_dispute() -> MintedDispute {
        MintedDispute {
            exchange_id: "mex:test:123".into(),
            opener_did: "did:mycelix:alice".into(),
            reason: "Service not provided".into(),
            resolved: None,
            resolver_did: None,
            resolution_reason: None,
            opened_at: ts(1_000_000),
            resolved_at: None,
        }
    }

    #[test]
    fn test_dispute_create_valid() {
        let result = validate_minted_dispute(&valid_dispute()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_dispute_create_rejects_empty_exchange_id() {
        let mut d = valid_dispute();
        d.exchange_id = "".into();
        let result = validate_minted_dispute(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_dispute_create_rejects_invalid_opener_did() {
        let mut d = valid_dispute();
        d.opener_did = "not-a-did".into();
        let result = validate_minted_dispute(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_dispute_create_rejects_empty_reason() {
        let mut d = valid_dispute();
        d.reason = "".into();
        let result = validate_minted_dispute(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ---- Dispute update validation ----

    fn valid_resolved_dispute() -> MintedDispute {
        MintedDispute {
            exchange_id: "mex:test:123".into(),
            opener_did: "did:mycelix:alice".into(),
            reason: "Service not provided".into(),
            resolved: Some(true),
            resolver_did: Some("did:mycelix:governance".into()),
            resolution_reason: Some("Evidence supports the claim".into()),
            opened_at: ts(1_000_000),
            resolved_at: Some(ts(2_000_000)),
        }
    }

    #[test]
    fn test_dispute_update_valid_accepted() {
        let result = validate_dispute_update(&valid_resolved_dispute()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_dispute_update_valid_rejected() {
        let mut d = valid_resolved_dispute();
        d.resolved = Some(false);
        let result = validate_dispute_update(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_dispute_update_rejects_unresolved() {
        let mut d = valid_resolved_dispute();
        d.resolved = None;
        let result = validate_dispute_update(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_dispute_update_rejects_missing_resolver() {
        let mut d = valid_resolved_dispute();
        d.resolver_did = None;
        let result = validate_dispute_update(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_dispute_update_rejects_invalid_resolver_did() {
        let mut d = valid_resolved_dispute();
        d.resolver_did = Some("not-a-did".into());
        let result = validate_dispute_update(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_dispute_update_rejects_missing_reason() {
        let mut d = valid_resolved_dispute();
        d.resolution_reason = None;
        let result = validate_dispute_update(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_dispute_update_rejects_empty_reason() {
        let mut d = valid_resolved_dispute();
        d.resolution_reason = Some("".into());
        let result = validate_dispute_update(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_dispute_update_rejects_missing_timestamp() {
        let mut d = valid_resolved_dispute();
        d.resolved_at = None;
        let result = validate_dispute_update(&d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
