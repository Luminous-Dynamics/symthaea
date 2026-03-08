//! Currency Factory Coordinator Zome
//!
//! Enables communities to create their own mutual credit currencies with
//! **Parameter Sovereignty** (custom names, limits, demurrage) while the
//! **Immutable Economic Physics** (zero-sum, balance limits, no pre-mining)
//! are enforced at the integrity level.
//!
//! Lifecycle: Draft → Active → Suspended → Active → Retired (terminal)
//!
//! Communities with >10 members require a governance proposal to create a currency.
//! The proposal ID is recorded in the CurrencyDefinition for audit.
//!
//! ## API Surface (30 externs)
//!
//! ### Lifecycle (7)
//! - `create_currency(CreateCurrencyInput) -> CurrencyDefinition`
//! - `activate_currency(ActivateCurrencyInput) -> CurrencyDefinition`
//! - `suspend_currency(String) -> CurrencyDefinition`
//! - `reactivate_currency(String) -> CurrencyDefinition`
//! - `retire_currency(String) -> CurrencyDefinition`
//! - `get_currency(String) -> Option<CurrencyDefinition>`
//! - `get_dao_currencies(String) -> Vec<CurrencyDefinition>`
//!
//! ### Discovery (2)
//! - `list_active_currencies(()) -> Vec<CurrencyDefinition>`
//! - `search_currencies(String) -> Vec<CurrencyDefinition>`
//!
//! ### Exchanges (7)
//! - `record_minted_exchange(RecordMintedExchangeInput) -> MintedExchange`
//! - `confirm_minted_exchange(String) -> MintedExchange`
//! - `cancel_expired_exchange(String) -> bool`
//! - `list_pending_exchanges(String) -> Vec<MintedExchange>`
//! - `list_pending_for_receiver(String) -> Vec<MintedExchange>`
//! - `get_currency_exchanges(PaginatedCurrencyInput) -> Vec<MintedExchange>`
//! - `get_exchange(String) -> Option<MintedExchange>`
//!
//! ### Balances & Portfolio (3)
//! - `get_minted_balance(GetMintedBalanceInput) -> MintedBalanceInfo`
//! - `get_member_portfolio(String) -> Vec<MintedBalanceInfo>`
//! - `get_member_exchanges(GetMemberExchangesInput) -> Vec<MintedExchange>`
//!
//! ### Stats & Reports (4)
//! - `get_currency_stats(String) -> CurrencyStats`
//! - `get_demurrage_report(String) -> DemurrageReport`
//! - `get_compost_balance(String) -> CompostBalance`
//! - `list_currency_members(String) -> Vec<String>`
//!
//! ### Demurrage & Compost (3)
//! - `apply_minted_demurrage(ApplyDemurrageInput) -> DemurrageResult`
//! - `apply_demurrage_all(String) -> Vec<DemurrageResult>`
//! - `redistribute_compost(String) -> RedistributeCompostResult`
//!
//! ### Disputes (3)
//! - `open_minted_dispute(OpenDisputeInput) -> MintedDispute`
//! - `resolve_minted_dispute(ResolveDisputeInput) -> MintedDispute`
//! - `get_dispute(String) -> Option<MintedDispute>`
//!
//! ### Governance (1)
//! - `amend_currency_params(AmendCurrencyParamsInput) -> CurrencyDefinition`

use hdk::prelude::*;
use mycelix_finance_types::{CurrencyStatus, MintedCurrencyParams};

// Internal helpers (pub(crate) visibility)
mod helpers;

// Extern modules — re-exported so Holochain can discover all #[hdk_extern] functions
mod balances;
pub mod demurrage;
mod disputes;
mod exchanges;
mod lifecycle;
mod stats;

pub use balances::*;
pub use demurrage::*;
pub use disputes::*;
pub use exchanges::*;
pub use lifecycle::*;
pub use stats::*;

// =============================================================================
// INPUT/OUTPUT TYPES (shared across modules)
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCurrencyInput {
    /// DID of the creating DAO/community
    pub dao_did: String,
    /// Community-chosen parameters
    pub params: MintedCurrencyParams,
    /// Governance proposal that authorized this (required for communities >10 members)
    pub governance_proposal_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ActivateCurrencyInput {
    pub currency_id: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordMintedExchangeInput {
    pub currency_id: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetMintedBalanceInput {
    pub currency_id: String,
    pub member_did: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MintedBalanceInfo {
    pub member_did: String,
    pub currency_id: String,
    pub currency_name: String,
    pub currency_symbol: String,
    pub balance: i32,
    pub credit_limit: i32,
    pub can_provide: bool,
    pub can_receive: bool,
    pub total_provided: f32,
    pub total_received: f32,
    pub exchange_count: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetMemberExchangesInput {
    pub currency_id: String,
    pub member_did: String,
    pub limit: Option<usize>,
    /// Only return exchanges with timestamp > this value (cursor for forward pagination)
    pub after_timestamp: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PaginatedCurrencyInput {
    pub currency_id: String,
    pub limit: Option<usize>,
    /// Only return exchanges with timestamp >= this value (cursor for forward pagination)
    pub after_timestamp: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DemurrageReportEntry {
    pub member_did: String,
    pub current_balance: i32,
    pub pending_deduction: i32,
    pub effective_balance: i32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DemurrageReport {
    pub currency_id: String,
    pub currency_name: String,
    pub demurrage_rate: f64,
    pub total_pending_demurrage: i64,
    pub affected_members: u32,
    pub entries: Vec<DemurrageReportEntry>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenDisputeInput {
    pub exchange_id: String,
    pub reason: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDisputeInput {
    pub exchange_id: String,
    /// true = accept dispute (reverse balances), false = reject (keep balances)
    pub accept: bool,
    pub resolution_reason: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AmendCurrencyParamsInput {
    pub currency_id: String,
    pub new_params: MintedCurrencyParams,
    pub governance_proposal_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CurrencyStats {
    pub currency_id: String,
    pub currency_name: String,
    pub currency_symbol: String,
    pub status: CurrencyStatus,
    pub member_count: u32,
    pub total_credit: i64,
    pub total_debt: i64,
    pub compost_balance: i64,
    pub net_sum: i64,
    pub total_exchanges: u64,
    pub confirmed_exchanges: u64,
    pub pending_exchanges: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApplyDemurrageInput {
    pub currency_id: String,
    pub member_did: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DemurrageResult {
    pub member_did: String,
    pub currency_id: String,
    pub previous_balance: i32,
    pub deduction: i32,
    pub new_balance: i32,
    pub demurrage_rate: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompostBalance {
    pub currency_id: String,
    pub accumulated: i32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RedistributeCompostResult {
    pub currency_id: String,
    pub total_redistributed: i32,
    pub recipients: u32,
    pub per_member_amount: i32,
    pub remainder_kept: i32,
}
