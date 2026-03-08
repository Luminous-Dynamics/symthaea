//! Shared mirror types and helper functions for Currency Mint sweettests.
//!
//! Mirror types avoid linking the coordinator crate (which generates conflicting
//! #[no_mangle] symbols with other coordinators).

use holochain::prelude::*;
use holochain::sweettest::*;
use mycelix_finance_types::MintedCurrencyParams;

// ── Mirror types ────────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCurrencyInput {
    pub dao_did: String,
    pub params: MintedCurrencyParams,
    pub governance_proposal_id: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ActivateCurrencyInput {
    pub currency_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordMintedExchangeInput {
    pub currency_id: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetMintedBalanceInput {
    pub currency_id: String,
    pub member_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MintedExchange {
    pub id: String,
    pub currency_id: String,
    pub provider_did: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub timestamp: Timestamp,
    pub confirmed: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConfirmMintedExchangeInput {
    pub exchange_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginatedCurrencyInput {
    pub currency_id: String,
    pub limit: Option<usize>,
    pub after_timestamp: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetMemberExchangesInput {
    pub currency_id: String,
    pub member_did: String,
    pub limit: Option<usize>,
    pub after_timestamp: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompostBalance {
    pub currency_id: String,
    pub accumulated: i32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OpenDisputeInput {
    pub exchange_id: String,
    pub reason: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MintedDispute {
    pub exchange_id: String,
    pub opener_did: String,
    pub reason: String,
    pub resolved: Option<bool>,
    pub resolver_did: Option<String>,
    pub resolution_reason: Option<String>,
    pub opened_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CurrencyStats {
    pub currency_id: String,
    pub currency_name: String,
    pub currency_symbol: String,
    pub status: mycelix_finance_types::CurrencyStatus,
    pub member_count: u32,
    pub total_credit: i64,
    pub total_debt: i64,
    pub compost_balance: i64,
    pub net_sum: i64,
    pub total_exchanges: u64,
    pub confirmed_exchanges: u64,
    pub pending_exchanges: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResolveDisputeInput {
    pub exchange_id: String,
    pub accept: bool,
    pub resolution_reason: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AmendCurrencyParamsInput {
    pub currency_id: String,
    pub new_params: MintedCurrencyParams,
    pub governance_proposal_id: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ApplyDemurrageInput {
    pub currency_id: String,
    pub member_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DemurrageResult {
    pub member_did: String,
    pub currency_id: String,
    pub previous_balance: i32,
    pub deduction: i32,
    pub new_balance: i32,
    pub demurrage_rate: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DemurrageReport {
    pub currency_id: String,
    pub currency_name: String,
    pub demurrage_rate: f64,
    pub total_pending_demurrage: i64,
    pub affected_members: u32,
    pub entries: Vec<DemurrageReportEntry>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DemurrageReportEntry {
    pub member_did: String,
    pub current_balance: i32,
    pub pending_deduction: i32,
    pub effective_balance: i32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RedistributeCompostResult {
    pub currency_id: String,
    pub total_redistributed: i32,
    pub recipients: u32,
    pub per_member_amount: i32,
    pub remainder_kept: i32,
}

// ── Helper functions ────────────────────────────────────────────────────────

pub fn test_params(name: &str, symbol: &str) -> MintedCurrencyParams {
    MintedCurrencyParams {
        name: name.into(),
        symbol: symbol.into(),
        description: format!("Test currency: {}", name),
        credit_limit: 40,
        demurrage_rate: 0.02,
        max_service_hours: 8,
        min_service_minutes: 15,
        requires_confirmation: false,
        confirmation_timeout_hours: 0,
        max_exchanges_per_day: 0,
    }
}

pub fn test_params_with_confirmation(name: &str, symbol: &str) -> MintedCurrencyParams {
    let mut p = test_params(name, symbol);
    p.requires_confirmation = true;
    p.confirmation_timeout_hours = 48;
    p
}

pub fn test_params_with_rate_limit(
    name: &str,
    symbol: &str,
    max_per_day: u8,
) -> MintedCurrencyParams {
    let mut p = test_params(name, symbol);
    p.max_exchanges_per_day = max_per_day;
    p
}

// ── Conductor setup helper ────────────────────────────────────────────────

/// Set up a SweetConductor with `agent_count` agents and the finance DNA installed.
///
/// Returns (conductor, agents, apps). Access cells via `apps[i].cells()[0]`.
pub async fn setup_finance_conductor(
    agent_count: usize,
) -> (SweetConductor, Vec<AgentPubKey>, Vec<SweetApp>) {
    let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path)
        .await
        .expect("Load DNA");
    let mut conductor = SweetConductor::from_standard_config().await;
    let agents = SweetAgents::get(conductor.keystore(), agent_count).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-finance", &agents, &[dna])
        .await
        .expect("Install app");
    (conductor, agents, apps.into())
}
