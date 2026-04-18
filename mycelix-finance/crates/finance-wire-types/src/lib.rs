// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wire types for Mycelix Finance.
//!
//! External-facing types for cross-cluster calls, test fixtures, and Leptos frontends.
//! These mirror the integrity entry types but are usable without HDK dependencies.
//! All hashes/keys are `String`, timestamps are `u64` (microseconds), opaque data is `Vec<u8>`.

use serde::{Deserialize, Serialize};

// =============================================================================
// ASSET & COLLATERAL
// =============================================================================

/// Asset type for collateral registration.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AssetType {
    RealEstate,
    Vehicle,
    Cryptocurrency,
    EnergyAsset,
    Equipment,
    Other(String),
}

/// Input for registering collateral against a loan or stake.
///
/// Cross-hApp-addressable via `source_happ` + `asset_id` so assets registered
/// in e.g. `mycelix-property` can back financial obligations without the
/// finance cluster needing to know that hApp's internal entry model.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RegisterCollateralInput {
    pub owner_did: String,
    /// Source hApp ID that authoritatively owns this asset entry
    /// (e.g. "mycelix-property", "mycelix-energy").
    pub source_happ: String,
    pub asset_type: AssetType,
    /// Opaque, `source_happ`-local asset identifier (e.g. "property:lot:42").
    pub asset_id: String,
    /// Estimated value in `currency` micro-units.
    pub value_estimate: u64,
    /// Currency of `value_estimate` (typically "SAP").
    pub currency: String,
}

/// Response for SAP balance queries.
///
/// Exposes both the raw stored balance and the effective balance after
/// pending demurrage has been netted out, so callers can display both
/// "what's on file" and "what's currently spendable" without calling twice.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SapBalanceResponse {
    pub member_did: String,
    /// Balance as last persisted (before pending demurrage).
    pub raw_balance: u64,
    /// Effective balance = `raw_balance - pending_demurrage`.
    pub effective_balance: u64,
    /// Demurrage accrued since `last_demurrage_at`, not yet persisted.
    pub pending_demurrage: u64,
    /// Microsecond timestamp of last persisted demurrage application.
    /// Signed to match Holochain's `Timestamp::as_micros()` return type.
    pub last_demurrage_at: i64,
}

/// Generic balance response used by the finance bridge for any currency.
///
/// Distinguished from [`SapBalanceResponse`] because the bridge may serve
/// TEND / MYCEL / SAP uniformly — this shape works for all three.
/// `available=false` signals the underlying zome was unreachable and the
/// returned balance is a permissive fallback (bootstrap mode).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BalanceResponse {
    pub member_did: String,
    pub currency: String,
    pub balance: u64,
    pub available: bool,
}

// =============================================================================
// PAYMENTS
// =============================================================================

/// Payment type classification.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentType {
    Direct,
    /// Treasury contribution (treasury_id)
    TreasuryContribution(String),
    /// Commons contribution (commons_pool_id)
    CommonsContribution(String),
    /// Escrow-backed payment (escrow_id)
    Escrow(String),
    /// Recurring payment with configuration
    Recurring(RecurringConfig),
}

/// Configuration for recurring payments.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RecurringConfig {
    pub frequency_days: u32,
    /// End date as microsecond timestamp (None = indefinite)
    pub end_date: Option<u64>,
    /// Remaining payment count (None = unlimited)
    pub remaining: Option<u32>,
}

/// Transfer status for payments.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransferStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
    Cancelled,
    Refunded,
}

/// A payment record.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Payment {
    pub id: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub fee: u64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub status: TransferStatus,
    pub memo: Option<String>,
    /// Created timestamp (microseconds)
    pub created: u64,
    /// Completed timestamp (microseconds), None if not yet completed
    pub completed: Option<u64>,
}

/// Input for sending a payment.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SendPaymentInput {
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub memo: Option<String>,
}

/// Input for querying payment history.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetPaymentHistoryInput {
    pub did: String,
    pub limit: Option<u32>,
}

// =============================================================================
// TEND (TIME-EXCHANGE NETWORK for DISTRIBUTED economies)
// =============================================================================

/// Service categories for TEND exchanges.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ServiceCategory {
    CareWork,
    HomeServices,
    FoodServices,
    Transportation,
    Education,
    GeneralAssistance,
    Administrative,
    Creative,
    TechSupport,
    Wellness,
    Gardening,
    Custom(String),
}

/// Exchange status for TEND time-banking records.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ExchangeStatus {
    Proposed,
    Confirmed,
    Disputed,
    Cancelled,
    Resolved,
}

/// Input for recording a TEND exchange.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecordExchangeInput {
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub cultural_alias: Option<String>,
    pub dao_did: String,
    /// Service date as microsecond timestamp
    pub service_date: Option<u64>,
}

/// A recorded TEND exchange.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExchangeRecord {
    pub id: String,
    pub provider_did: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub status: ExchangeStatus,
    /// Timestamp in microseconds
    pub timestamp: u64,
}

/// TEND oracle state response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OracleStateResponse {
    pub vitality: u32,
    pub tier_name: String,
    pub limit: i32,
}

/// TEND balance information for a member within a DAO.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BalanceInfo {
    pub member_did: String,
    pub dao_did: String,
    pub balance: i32,
    pub can_provide: bool,
    pub can_receive: bool,
    pub total_provided: f32,
    pub total_received: f32,
    pub exchange_count: u32,
}

/// Input for querying a member's TEND balance.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetBalanceInput {
    pub member_did: String,
    pub dao_did: String,
}

/// Paginated input scoped to a DAO.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedDaoInput {
    pub dao_did: String,
    pub limit: Option<usize>,
}

/// Paginated input scoped to a DID.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedDidInput {
    pub did: String,
    pub limit: Option<usize>,
}

// =============================================================================
// RECOGNITION (MYCEL reputation)
// =============================================================================

/// Contribution types for recognition events.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContributionType {
    Technical,
    Community,
    Care,
    Governance,
    Creative,
    Education,
    General,
}

/// Input for recognizing a member's contribution.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecognizeMemberInput {
    pub recipient_did: String,
    pub contribution_type: ContributionType,
    pub cycle_id: String,
}

/// A recognition event from one member to another.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecognitionEvent {
    /// DID of the member giving recognition
    pub recognizer_did: String,
    /// DID of the member receiving recognition
    pub recipient_did: String,
    /// Computed weight: recognizer's MYCEL x base_weight
    pub weight: f64,
    /// Type of contribution being recognized
    pub contribution_type: ContributionType,
    /// Monthly cycle ID (e.g., "2026-02")
    pub cycle_id: String,
    /// Recognizer's MYCEL score at time of recognition
    pub recognizer_mycel: f64,
    /// Timestamp in microseconds
    pub timestamp: u64,
}

/// Input for querying recognition events.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetRecognitionsInput {
    pub member_did: String,
    pub cycle_id: Option<String>,
    /// Maximum number of recognition events to return (default 100)
    pub limit: Option<usize>,
}

/// A member's MYCEL state -- soulbound, non-transferable reputation.
///
/// MYCEL score is 0.0-1.0, computed from 4 weighted components:
/// - Participation (40%): tx activity, governance voting, commons engagement
/// - Recognition (20%): weighted recognition events from other members
/// - Validation (20%): quality of work as validator/contributor
/// - Longevity (20%): time active, capped at 24 months
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemberMycelState {
    /// DID of the member
    pub member_did: String,
    /// Composite MYCEL score (0.0 - 1.0)
    pub mycel_score: f64,
    /// Participation component (0.0 - 1.0)
    pub participation: f64,
    /// Recognition component (0.0 - 1.0)
    pub recognition: f64,
    /// Validation component (0.0 - 1.0)
    pub validation: f64,
    /// Longevity component (0.0 - 1.0)
    pub longevity: f64,
    /// Active months count (for longevity calculation)
    pub active_months: u32,
    /// Whether this member is in apprentice mode
    pub is_apprentice: bool,
    /// DID of the vouching mentor (if apprentice)
    pub mentor_did: Option<String>,
    /// Total recognitions given in current cycle
    pub recognitions_given_this_cycle: u32,
    /// Current cycle ID
    pub current_cycle_id: String,
    /// When the MYCEL state was last updated (microseconds)
    pub last_updated: u64,
}

// =============================================================================
// STAKING
// =============================================================================

/// Stake status for collateral positions.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum StakeStatus {
    /// Active and earning rewards
    Active,
    /// In unbonding period (21 days)
    Unbonding,
    /// Withdrawn after unbonding
    Withdrawn,
    /// Partially slashed
    Slashed,
    /// Fully slashed (jailed)
    Jailed,
}

/// Input for creating a collateral stake.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateStakeInput {
    pub staker_did: String,
    pub sap_amount: u64,
}

/// A collateral stake position.
///
/// Represents a validator's stake using SAP collateral with MYCEL weighting.
/// Stake weight = 1.0 + mycel_score (range 1.0-2.0).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CollateralStake {
    /// Unique stake identifier
    pub id: String,
    /// Staker's DID
    pub staker_did: String,
    /// SAP collateral amount
    pub sap_amount: u64,
    /// MYCEL score at time of staking (affects weight)
    pub mycel_score: f32,
    /// Stake weight = 1.0 + mycel_score (range 1.0-2.0)
    pub stake_weight: f32,
    /// Stake creation timestamp (microseconds)
    pub staked_at: u64,
    /// Unbonding period end (None if not unbonding, microseconds)
    pub unbonding_until: Option<u64>,
    /// Current stake status
    pub status: StakeStatus,
    /// Accumulated pending rewards
    pub pending_rewards: u64,
    /// Last reward claim timestamp (microseconds)
    pub last_reward_claim: u64,
}

// =============================================================================
// TREASURY
// =============================================================================

/// A community treasury.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Treasury {
    pub id: String,
    pub name: String,
    pub description: String,
    pub currency: String,
    pub balance: u64,
    pub reserve_ratio: f64,
    pub managers: Vec<String>,
    /// Created timestamp (microseconds)
    pub created: u64,
    /// Last updated timestamp (microseconds)
    pub last_updated: u64,
}

/// Commons pool for a DAO -- inalienable reserve is constitutionally protected.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommonsPool {
    pub id: String,
    pub dao_did: String,
    pub inalienable_reserve: u64,
    pub available_balance: u64,
    /// Always true -- constitutional requirement
    pub demurrage_exempt: bool,
    /// Created timestamp (microseconds)
    pub created_at: u64,
    /// Last activity timestamp (microseconds)
    pub last_activity: u64,
}

/// Generic list input with an ID and optional limit for pagination.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ListInput {
    pub id: String,
    pub limit: Option<usize>,
}

// =============================================================================
// RUNTIME DISCOVERY
// =============================================================================

/// Runtime discovery response from the finance bridge zome.
///
/// Provides the caller's member context and resolved resource IDs
/// for the connected DAO/treasury/commons pool.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct FinanceRuntimeDiscovery {
    /// The authenticated member's DID.
    pub member_did: String,
    /// Default DAO for this member (first entry of `dao_dids`, or None).
    pub default_dao_did: Option<String>,
    /// All DAOs the member is a confirmed participant in.
    pub dao_dids: Vec<String>,
    /// Resolved treasury ID for the member's DAO.
    pub treasury_id: Option<String>,
    /// Resolved commons pool ID for the member's DAO.
    pub commons_pool_id: Option<String>,
}

// =============================================================================
// BRIDGE + HEALTH + CROSS-CLUSTER I/O
// =============================================================================

/// Standard bridge health response.
///
/// Returned from `finance_bridge::health_check` for cross-cluster monitoring.
/// Mirrors the health-response shape used by other cluster bridges.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FinanceBridgeHealth {
    pub healthy: bool,
    /// DID of the agent running this bridge.
    pub agent: String,
    /// Names of coordinator zomes this bridge can dispatch to locally.
    pub zomes: Vec<String>,
}

/// Input for the finance bridge's cross-hApp payment entrypoint.
///
/// `source_happ` is the caller's hApp ID (for attribution + fee tiering);
/// `reference` is a free-form memo (invoice ID, rent period, etc.).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProcessPaymentInput {
    pub source_happ: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub reference: String,
}

/// Input for the simpler collateral-deposit path (raw collateral-type string
/// instead of the richer [`RegisterCollateralInput`] + `AssetType` enum).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DepositCollateralInput {
    pub depositor_did: String,
    /// Collateral asset identifier (e.g. "ETH", "USDC").
    pub collateral_type: String,
    /// Collateral amount in micro-units.
    pub collateral_amount: u64,
    /// Oracle exchange rate used for SAP minting (collateral → SAP).
    pub oracle_rate: f64,
}

/// Input for recomputing a collateral position's LTV + health status.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateCollateralHealthInput {
    pub collateral_id: String,
    /// Current obligation the collateral backs (in SAP micro-units).
    pub obligation_amount: u64,
}

/// Member fee-tier lookup response (derived from MYCEL recognition score).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FeeTierResponse {
    pub member_did: String,
    /// MYCEL recognition score; f64 to match the oracle return type.
    pub mycel_score: f64,
    /// Human-readable tier name (Bronze/Silver/Gold/...).
    pub tier_name: String,
    /// Fee rate applied to this tier, as a fraction (0.0–1.0).
    pub base_fee_rate: f64,
}

// =============================================================================
// DEMURRAGE + GOVERNANCE MINTING
// =============================================================================

/// Input for applying demurrage to a single member's balance.
///
/// When deduction > 0, the deducted amount is redistributed as compost
/// to local (70%) / regional (20%) / global (10%) commons pools per the
/// constitutional compost rule. The caller supplies the target pool IDs
/// so the caller (not this zome) controls which pools receive the compost.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ApplyDemurrageInput {
    pub member_did: String,
    /// Currency to apply demurrage to (typically "SAP").
    pub currency_id: String,
    /// Local commons pool for 70% of deducted compost. None ⇒ queue until set.
    pub local_commons_pool_id: Option<String>,
    /// Regional commons pool for 20% of deducted compost.
    pub regional_commons_pool_id: Option<String>,
    /// Global commons pool for 10% of deducted compost.
    pub global_commons_pool_id: Option<String>,
}

/// Result of a demurrage application (production path).
///
/// `deducted` is the amount removed from the member's balance (0 when
/// the balance is below the exempt floor). `redistributed` is true iff
/// all three target pools accepted their compost share; false means one
/// or more shares were queued for retry.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DemurrageResult {
    pub deducted: u64,
    pub redistributed: bool,
}

/// Input for minting new SAP authorized by a governance proposal.
///
/// The ONLY path for new SAP to enter circulation outside collateral deposits.
/// `proposal_id` is the governance proposal that authorized this mint;
/// `recipient_did` is the member receiving the new SAP.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MintSapFromGovernanceInput {
    pub amount: u64,
    pub proposal_id: String,
    pub recipient_did: String,
}
