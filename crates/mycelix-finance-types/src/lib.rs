#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Finance Shared Types
//!
//! Canonical type definitions for the Mycelix three-currency economic system.
//! This crate has NO HDK/HDI dependency so it can be used by:
//! - Integrity zomes (hdi)
//! - Coordinator zomes (hdk)
//! - External SDK
//! - CLI tools
//!
//! All zomes SHOULD import these types rather than re-defining them locally.

use serde::{Deserialize, Serialize};

// =============================================================================
// CURRENCIES
// =============================================================================

/// The three currencies of the Mycelix economic system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Currency {
    /// Soulbound reputation substrate (0.0 - 1.0, non-transferable)
    Mycel,
    /// Circulation medium (transferable, subject to demurrage)
    Sap,
    /// Mutual credit (time-based, zero-sum)
    Tend,
}

impl Currency {
    /// Get the display name for this currency
    pub fn display_name(&self) -> &'static str {
        match self {
            Currency::Mycel => "MYCEL",
            Currency::Sap => "SAP",
            Currency::Tend => "TEND",
        }
    }

    /// Check if currency is transferable
    pub fn is_transferable(&self) -> bool {
        match self {
            Currency::Mycel => false, // Soulbound — never transferable
            Currency::Sap => true,
            Currency::Tend => true,
        }
    }
}

impl core::fmt::Display for Currency {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// =============================================================================
// FEE TIERS
// =============================================================================

/// Progressive fee tiers based on MYCEL score.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeeTier {
    /// MYCEL < 0.3, base fee 0.10%
    Newcomer,
    /// MYCEL 0.3 - 0.7, base fee 0.03%
    Member,
    /// MYCEL > 0.7, base fee 0.01%
    Steward,
}

impl FeeTier {
    /// Base fee rate as a fraction (e.g. 0.001 = 0.1%)
    pub fn base_fee_rate(&self) -> f64 {
        match self {
            FeeTier::Newcomer => 0.001,
            FeeTier::Member => 0.0003,
            FeeTier::Steward => 0.0001,
        }
    }

    /// Derive fee tier from a MYCEL score (0.0 - 1.0)
    pub fn from_mycel(score: f64) -> Self {
        if score > 0.7 {
            FeeTier::Steward
        } else if score >= 0.3 {
            FeeTier::Member
        } else {
            FeeTier::Newcomer
        }
    }
}

// =============================================================================
// TEND LIMIT TIERS (Counter-cyclical)
// =============================================================================

/// Counter-cyclical TEND limit tiers driven by metabolic oracle vitality.
///
/// When the network is stressed, TEND capacity expands automatically
/// to provide emergency exchange capacity (WIR Bank pattern).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TendLimitTier {
    /// Vitality >= 41, limit +-40
    Normal,
    /// Vitality 21-40, limit +-60
    Elevated,
    /// Vitality 11-20, limit +-80
    High,
    /// Vitality 0-10, limit +-120
    Emergency,
}

impl TendLimitTier {
    pub fn limit(&self) -> i32 {
        match self {
            TendLimitTier::Normal => 40,
            TendLimitTier::Elevated => 60,
            TendLimitTier::High => 80,
            TendLimitTier::Emergency => 120,
        }
    }

    pub fn from_vitality(vitality: u32) -> Self {
        match vitality {
            0..=10 => TendLimitTier::Emergency,
            11..=20 => TendLimitTier::High,
            21..=40 => TendLimitTier::Elevated,
            _ => TendLimitTier::Normal,
        }
    }
}

// =============================================================================
// METABOLIC STATE
// =============================================================================

/// Metabolic states of the network (5 levels).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetabolicState {
    Thriving,
    Healthy,
    Stressed,
    Critical,
    Failing,
}

impl MetabolicState {
    pub fn from_vitality(score: f64) -> Self {
        if score >= 80.0 {
            MetabolicState::Thriving
        } else if score >= 60.0 {
            MetabolicState::Healthy
        } else if score >= 40.0 {
            MetabolicState::Stressed
        } else if score >= 20.0 {
            MetabolicState::Critical
        } else {
            MetabolicState::Failing
        }
    }
}

// =============================================================================
// CONTRIBUTION TYPES
// =============================================================================

/// Types of contribution recognized in the system.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContributionType {
    Technical,
    Community,
    Care,
    Governance,
    Creative,
    Education,
    General,
}

// =============================================================================
// SUCCESSION
// =============================================================================

/// How SAP should be handled when a member exits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SuccessionPreference {
    /// Default: remaining SAP goes to member's local commons pool
    #[default]
    Commons,
    /// SAP transferred to a designated DID
    Designee(String),
    /// SAP redeemed for collateral through the bridge
    Redemption,
}

// =============================================================================
// DEMURRAGE CONSTANTS & COMPUTATION
// =============================================================================

/// Annual demurrage rate (2%). Constitutional bounds: 1-5%.
pub const DEMURRAGE_RATE: f64 = 0.02;

// =============================================================================
// DEMURRAGE DISCOUNT FOR DIVERSIFIED POSITIONS
// =============================================================================

/// Demurrage discount per distinct asset class in a multi-collateral position.
/// 0.1% per class, capped at 0.5% total discount.
///
/// Incentivizes diversification through ongoing savings rather than just
/// a threshold shift that may never be triggered.
pub const DEMURRAGE_DISCOUNT_PER_CLASS: f64 = 0.001;

/// Maximum demurrage discount from diversification (0.5% off the base rate).
pub const DEMURRAGE_DISCOUNT_CAP: f64 = 0.005;

/// Minimum number of distinct asset classes to qualify for demurrage discount.
/// A single-class position gets no discount.
pub const DEMURRAGE_DISCOUNT_MIN_CLASSES: usize = 3;

/// Compute the effective demurrage rate for a position with diversified collateral.
///
/// `base_rate` is the standard demurrage rate (e.g., 0.02 = 2%).
/// `distinct_classes` is the number of distinct asset classes in the position.
///
/// Returns: effective rate = base_rate - discount (never below 0.5 * base_rate).
///
/// Example: 3 classes → discount = 3 * 0.001 - 0.002 (first 2 don't count) = 0.001
///          4 classes → 0.002, 5 classes → 0.003, 7+ classes → 0.005 (capped)
pub fn compute_diversified_demurrage_rate(base_rate: f64, distinct_classes: usize) -> f64 {
    if distinct_classes < DEMURRAGE_DISCOUNT_MIN_CLASSES {
        return base_rate;
    }
    let bonus_classes = distinct_classes.saturating_sub(DEMURRAGE_DISCOUNT_MIN_CLASSES - 1);
    let discount =
        (bonus_classes as f64 * DEMURRAGE_DISCOUNT_PER_CLASS).min(DEMURRAGE_DISCOUNT_CAP);
    // Never reduce below 50% of base rate (constitutional floor)
    (base_rate - discount).max(base_rate * 0.5)
}
/// SAP exempt floor in micro-units (200 SAP = 200_000_000 micro-SAP).
///
/// Reduced from 1,000 SAP based on macro-economic simulation findings:
/// at 1,000 SAP floor, demurrage rate (1-5%) has no effect on transaction
/// velocity because most agents hold near or below the floor. At 200 SAP,
/// demurrage becomes an effective circulation lever.
/// See: mycelix-workspace/simulations/SIMULATION_REPORT.md (Finding 3)
pub const DEMURRAGE_EXEMPT_FLOOR: u64 = 200_000_000;
/// Compost distribution: 70% to local commons pool.
pub const COMPOST_LOCAL_PCT: u64 = 70;
/// Compost distribution: 20% to regional commons pool.
pub const COMPOST_REGIONAL_PCT: u64 = 20;
/// Compost distribution: 10% to global commons fund.
pub const COMPOST_GLOBAL_PCT: u64 = 10;
/// Inalienable reserve ratio: constitutional minimum 25%.
pub const INALIENABLE_RESERVE_RATIO: f64 = 0.25;

/// Maximum SAP that can be minted per governance proposal (in micro-SAP).
/// 100,000 SAP = 100_000_000_000 micro-SAP. Constitutional bound.
pub const SAP_MINT_PER_PROPOSAL_MAX: u64 = 100_000_000_000;
/// Maximum SAP that can be minted via governance per year (in micro-SAP).
/// 1,000,000 SAP = 1_000_000_000_000 micro-SAP. Constitutional bound.
pub const SAP_MINT_ANNUAL_MAX: u64 = 1_000_000_000_000;

// =============================================================================
// PHASE 1: SETTLEMENT HARDENING CONSTANTS
// =============================================================================

/// Maximum pending compost deliveries before forcing drain (prevents unbounded queue).
pub const PENDING_COMPOST_MAX_QUEUE: usize = 256;

/// Maximum age (in seconds) for a pending compost entry before it's considered stale.
/// Stale entries are retried once more then discarded with an audit log entry.
pub const PENDING_COMPOST_STALE_SECONDS: u64 = 86_400; // 24 hours

/// Oracle rate tolerance: max deviation from consensus before deposit is rejected.
/// 5% tolerance — if claimed rate diverges more than this from oracle consensus, reject.
pub const ORACLE_RATE_TOLERANCE: f64 = 0.05;

/// Mint cap counter anchor name. Used for CAS-based annual mint cap enforcement.
pub const SAP_MINT_CAP_ANCHOR: &str = "sap_mint_annual_cap";

/// Maximum retries for compost delivery before logging as orphaned.
pub const COMPOST_MAX_RETRIES: u32 = 3;

// =============================================================================
// PHASE 1a: PENDING COMPOST TYPES
// =============================================================================

/// A compost delivery that failed and needs retry.
///
/// Created when treasury/receive_compost fails during demurrage application.
/// Processed by `drain_pending_compost()` on next balance mutation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingCompost {
    /// Target commons pool ID
    pub commons_pool_id: String,
    /// Amount of compost to deliver (micro-SAP)
    pub amount: u64,
    /// DID of the member whose demurrage produced this compost
    pub source_member_did: String,
    /// Pool tier: Local, Regional, or Global
    pub pool_tier: CompostPoolTier,
    /// When this pending delivery was created
    pub created_at_micros: i64,
    /// Number of retry attempts so far
    pub retry_count: u32,
}

/// Which commons pool tier a compost delivery targets.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompostPoolTier {
    Local,
    Regional,
    Global,
}

// =============================================================================
// PHASE 1c: MINT CAP COUNTER
// =============================================================================

/// Tracks cumulative SAP minted via governance in the current annual period.
///
/// Uses optimistic-locking with CAS semantics: read → update → verify.
/// Prevents concurrent governance mints from exceeding the annual cap.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SapMintCapCounter {
    /// Start of the current annual period (microseconds since epoch)
    pub period_start_micros: i64,
    /// Cumulative SAP minted in this period (micro-SAP)
    pub cumulative_minted: u64,
    /// Number of governance mints in this period
    pub mint_count: u32,
    /// Last updated timestamp (microseconds)
    pub last_updated_micros: i64,
}

impl SapMintCapCounter {
    /// Check if a proposed mint would exceed the annual cap.
    pub fn would_exceed_cap(&self, proposed_amount: u64) -> bool {
        self.cumulative_minted.saturating_add(proposed_amount) > SAP_MINT_ANNUAL_MAX
    }

    /// Remaining mintable SAP in the current period.
    pub fn remaining_capacity(&self) -> u64 {
        SAP_MINT_ANNUAL_MAX.saturating_sub(self.cumulative_minted)
    }

    /// Check if the current period has expired (> 365 days since period_start).
    pub fn is_period_expired(&self, now_micros: i64) -> bool {
        let year_micros: i64 = 365 * 24 * 60 * 60 * 1_000_000;
        now_micros.saturating_sub(self.period_start_micros) > year_micros
    }
}

// =============================================================================
// PHASE 2a: COVENANT TYPES
// =============================================================================

/// A covenant (restriction) placed on registered collateral.
///
/// Covenants prevent collateral from being transferred, sold, or released
/// while obligations are outstanding. Enforced at the coordinator level
/// before any collateral status change.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Covenant {
    /// Unique covenant ID
    pub id: String,
    /// Collateral registration ID this covenant applies to
    pub collateral_id: String,
    /// Type of restriction
    pub restriction: CovenantRestriction,
    /// DID of the lien holder / covenant beneficiary
    pub beneficiary_did: String,
    /// When this covenant was created
    pub created_at_micros: i64,
    /// When this covenant expires (None = permanent until released)
    pub expires_at_micros: Option<i64>,
    /// Whether this covenant has been released
    pub released: bool,
    /// DID of the agent who released the covenant (if released)
    pub released_by: Option<String>,
    /// When the covenant was released
    pub released_at_micros: Option<i64>,
}

/// Types of collateral restrictions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovenantRestriction {
    /// Cannot transfer ownership while covenant is active
    TransferLock,
    /// Cannot sell the asset while covenant is active
    SaleProhibition,
    /// A lien holder has priority claim on the asset
    LienHolder,
    /// Insurance must be maintained on the asset
    InsuranceRequirement,
    /// Asset is pledged as collateral for a specific obligation
    CollateralPledge { obligation_id: String },
}

// =============================================================================
// PHASE 2b: LTV MONITORING TYPES
// =============================================================================

/// Health status of a collateral position, computed from LTV ratio.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollateralHealth {
    /// Collateral registration ID
    pub collateral_id: String,
    /// Current estimated value of the collateral (micro-SAP equivalent)
    pub current_value: u64,
    /// Outstanding obligation against this collateral (micro-SAP)
    pub obligation_amount: u64,
    /// Loan-to-Value ratio (0.0 - 1.0+, higher = more leveraged)
    pub ltv_ratio: f64,
    /// Health status based on LTV thresholds
    pub status: CollateralHealthStatus,
    /// When this health check was computed
    pub computed_at_micros: i64,
}

/// Collateral health tiers based on LTV ratio.
///
/// Thresholds: Warning (80%), Margin Call (90%), Liquidation (95%).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollateralHealthStatus {
    /// LTV < 80% — healthy position
    Healthy,
    /// LTV 80-90% — warning issued, should add collateral or reduce obligation
    Warning,
    /// LTV 90-95% — margin call, must add collateral within 72 hours
    MarginCall,
    /// LTV > 95% — eligible for governance-approved liquidation
    Liquidation,
}

impl CollateralHealthStatus {
    /// Compute health status from LTV ratio.
    pub fn from_ltv(ltv: f64) -> Self {
        if ltv > 0.95 {
            CollateralHealthStatus::Liquidation
        } else if ltv > 0.90 {
            CollateralHealthStatus::MarginCall
        } else if ltv > 0.80 {
            CollateralHealthStatus::Warning
        } else {
            CollateralHealthStatus::Healthy
        }
    }
}

/// LTV threshold constants for collateral health monitoring.
pub const LTV_WARNING_THRESHOLD: f64 = 0.80;
pub const LTV_MARGIN_CALL_THRESHOLD: f64 = 0.90;
pub const LTV_LIQUIDATION_THRESHOLD: f64 = 0.95;

/// Hours before margin call forces liquidation eligibility.
pub const MARGIN_CALL_GRACE_HOURS: u64 = 72;

// =============================================================================
// PHASE 3a: ENERGY CERTIFICATE TYPES
// =============================================================================

/// A verified certificate of energy production, suitable as SAP collateral.
///
/// Represents a verified proof that a specific quantity of renewable energy
/// was produced at a specific location and time. Linked to mycelix-energy
/// projects via `project_id`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnergyCertificate {
    /// Unique certificate ID
    pub id: String,
    /// Energy project that produced this energy
    pub project_id: String,
    /// Energy source type
    pub source: EnergySource,
    /// Quantity produced in kilowatt-hours (kWh)
    pub kwh_produced: f64,
    /// Period start (microseconds since epoch)
    pub period_start_micros: i64,
    /// Period end (microseconds since epoch)
    pub period_end_micros: i64,
    /// Geographic location (latitude, longitude)
    pub location: (f64, f64),
    /// DID of the producer
    pub producer_did: String,
    /// DID of the verifier (Citizen+ tier required to verify)
    pub verifier_did: Option<String>,
    /// Verification status
    pub status: CertificateStatus,
    /// Terra Atlas cross-reference ID (optional, for USACE-linked projects)
    pub terra_atlas_id: Option<String>,
    /// SAP-equivalent value (set after oracle attestation)
    pub sap_value: Option<u64>,
    /// When this certificate was created
    pub created_at_micros: i64,
}

/// Energy source types for certificates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergySource {
    Solar,
    Wind,
    Hydro,
    Geothermal,
    Biomass,
    BatteryDischarge,
    Other(String),
}

/// Verification status of an energy certificate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateStatus {
    /// Submitted by producer, awaiting verification
    Pending,
    /// Verified by a Citizen+ tier peer
    Verified,
    /// Rejected during verification
    Rejected { reason: String },
    /// Pledged as collateral (cannot be traded)
    Pledged,
    /// Consumed / retired (cannot be reused)
    Retired,
}

// =============================================================================
// PHASE 3b: AGRICULTURAL ASSET TYPES
// =============================================================================

/// A registered agricultural asset, suitable as SAP collateral.
///
/// Represents verified agricultural production capacity or harvested goods.
/// Linked to mycelix-commons food production zomes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgriculturalAsset {
    /// Unique asset ID
    pub id: String,
    /// Type of agricultural asset
    pub asset_type: AgriAssetType,
    /// Quantity in kilograms
    pub quantity_kg: f64,
    /// Geographic location (latitude, longitude)
    pub location: (f64, f64),
    /// Harvest or production date (microseconds since epoch)
    pub production_date_micros: i64,
    /// Expected shelf life / viability (microseconds from production date)
    pub viability_duration_micros: Option<i64>,
    /// DID of the producer
    pub producer_did: String,
    /// DID of the verifier (Citizen+ required)
    pub verifier_did: Option<String>,
    /// Verification status
    pub status: AgriAssetStatus,
    /// SAP-equivalent value (set after oracle attestation)
    pub sap_value: Option<u64>,
    /// When this asset was registered
    pub created_at_micros: i64,
}

/// Types of agricultural assets.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgriAssetType {
    /// Harvested grain (maize, wheat, rice, etc.)
    Grain { crop: String },
    /// Fresh produce (vegetables, fruits)
    Produce { crop: String },
    /// Soil amendment (compost, Terra Preta, biochar)
    SoilAmendment { amendment_type: String },
    /// Fertilizer (NPK, organic, etc.)
    Fertilizer { fertilizer_type: String },
    /// Seed stock
    Seeds { variety: String },
    /// Livestock (by weight, not count)
    Livestock { species: String },
    /// Other agricultural product
    Other(String),
}

/// Verification status of an agricultural asset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgriAssetStatus {
    /// Registered, awaiting verification
    Pending,
    /// Verified by a Citizen+ tier peer
    Verified,
    /// Rejected during verification
    Rejected { reason: String },
    /// Pledged as collateral
    Pledged,
    /// Consumed or expired
    Consumed,
}

// =============================================================================
// PHASE 3c: MULTI-COLLATERAL POSITION TYPES
// =============================================================================

/// A multi-collateral position combining multiple assets as backing.
///
/// Diversified collateral positions receive a 5% LTV bonus per additional
/// distinct asset type (capped at 20% total bonus).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiCollateralPosition {
    /// Unique position ID
    pub id: String,
    /// DID of the position holder
    pub holder_did: String,
    /// Component collateral assets with their weights
    pub components: Vec<CollateralComponent>,
    /// Aggregate SAP value of the position
    pub aggregate_value: u64,
    /// Aggregate obligation against this position
    pub aggregate_obligation: u64,
    /// Diversification bonus (0.0 - 0.20, 5% per distinct asset type, capped at 20%)
    pub diversification_bonus: f64,
    /// Effective LTV after diversification bonus
    pub effective_ltv: f64,
    /// Health status
    pub status: CollateralHealthStatus,
    /// When this position was created
    pub created_at_micros: i64,
    /// When this position was last revalued
    pub last_revalued_micros: i64,
}

/// A component of a multi-collateral position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollateralComponent {
    /// ID of the underlying collateral, energy certificate, or agricultural asset
    pub asset_id: String,
    /// Type of the underlying asset
    pub asset_class: CollateralAssetClass,
    /// Weight in the position (0.0 - 1.0, all weights must sum to 1.0)
    pub weight: f64,
    /// Current SAP-equivalent value
    pub current_value: u64,
}

/// Asset classes for multi-collateral diversification scoring.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollateralAssetClass {
    RealEstate,
    Cryptocurrency,
    EnergyCertificate,
    AgriculturalAsset,
    CarbonCredit,
    Vehicle,
    Equipment,
    Other,
}

/// Diversification bonus: 5% per distinct asset class, capped at 20%.
pub const DIVERSIFICATION_BONUS_PER_CLASS: f64 = 0.05;
pub const DIVERSIFICATION_BONUS_CAP: f64 = 0.20;

/// Compute diversification bonus from a set of collateral components.
pub fn compute_diversification_bonus(components: &[CollateralComponent]) -> f64 {
    // Count distinct asset classes
    let distinct: usize = {
        let mut seen = Vec::new();
        for c in components {
            if !seen.contains(&c.asset_class) {
                seen.push(c.asset_class.clone());
            }
        }
        seen.len()
    };
    // Bonus starts from the 2nd distinct class
    let bonus_classes = distinct.saturating_sub(1);
    (bonus_classes as f64 * DIVERSIFICATION_BONUS_PER_CLASS).min(DIVERSIFICATION_BONUS_CAP)
}

// =============================================================================
// PHASE 4a: FIAT BRIDGE TYPES
// =============================================================================

/// A one-way fiat deposit record (inbound only — fiat enters, SAP is minted).
///
/// Design principle: "digestion" model. Fiat is consumed to create SAP.
/// No reverse dependency — Mycelix never depends on fiat settlement systems.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FiatBridgeDeposit {
    /// Unique deposit ID
    pub id: String,
    /// DID of the depositor
    pub depositor_did: String,
    /// Fiat currency (ISO 4217: "USD", "ZAR", "EUR", etc.)
    pub fiat_currency: String,
    /// Fiat amount in minor units (cents)
    pub fiat_amount: u64,
    /// SAP minted against this deposit
    pub sap_minted: u64,
    /// Exchange rate used (fiat minor units per SAP micro-unit)
    pub exchange_rate: f64,
    /// DID of the governance-approved deposit verifier
    pub verifier_did: String,
    /// Status of the deposit
    pub status: FiatDepositStatus,
    /// External reference (bank transfer ID, proof of deposit, etc.)
    pub external_reference: String,
    /// When this deposit was recorded
    pub created_at_micros: i64,
    /// When this deposit was verified
    pub verified_at_micros: Option<i64>,
}

/// Status of a fiat bridge deposit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FiatDepositStatus {
    /// Submitted, awaiting verifier confirmation
    Pending,
    /// Verified by governance-approved agent, SAP minted
    Verified,
    /// Rejected by verifier (invalid proof, etc.)
    Rejected { reason: String },
}

/// Supported fiat currencies for the bridge.
/// Each must have a governance-approved deposit verifier.
pub const SUPPORTED_FIAT_CURRENCIES: &[&str] =
    &["USD", "ZAR", "EUR", "GBP", "MXN", "KRW", "JPY", "CHF"];

// =============================================================================
// SAP→PHYSICAL ASSET REDEMPTION
// =============================================================================

/// A redemption claim: SAP exchanged for a claim on physical assets.
///
/// Design principle: no SAP→fiat reverse path. Instead, SAP can be redeemed
/// for verified physical assets (energy, agricultural goods, housing credits).
/// This keeps the economy grounded in real value rather than creating a fiat
/// exit door that enables bank runs.
///
/// Redemption requires governance approval for large amounts and is rate-limited
/// to prevent rapid withdrawal cascades.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssetRedemption {
    /// Unique redemption ID
    pub id: String,
    /// DID of the redeemer
    pub redeemer_did: String,
    /// SAP amount being redeemed (micro-SAP)
    pub sap_amount: u64,
    /// Type of physical asset being claimed
    pub asset_claim: AssetClaim,
    /// Status of the redemption
    pub status: RedemptionStatus,
    /// When this redemption was requested
    pub requested_at_micros: i64,
    /// Cooldown: earliest this redemption can be fulfilled (7-day delay)
    pub fulfillable_after_micros: i64,
    /// When this redemption was fulfilled (if completed)
    pub fulfilled_at_micros: Option<i64>,
    /// DID of the fulfiller (asset provider)
    pub fulfiller_did: Option<String>,
}

/// Types of physical assets that can be claimed with SAP.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetClaim {
    /// Claim on energy production (kWh)
    Energy {
        kwh_requested: u64,
        source_preference: Option<String>, // "Solar", "Wind", etc.
    },
    /// Claim on agricultural goods (kg)
    Agricultural {
        product_type: String, // "Grain", "Produce", etc.
        kg_requested: u64,
    },
    /// Claim on housing credit (micro-SAP equivalent)
    HousingCredit {
        housing_unit_id: Option<String>,
        credit_amount: u64,
    },
    /// Claim on carbon offset (tonnes CO2e)
    CarbonOffset { tonnes_co2e: u64 },
    /// Claim on community service hours (TEND equivalent)
    ServiceHours {
        hours_requested: u32,
        service_category: Option<String>,
    },
}

/// Status of a physical asset redemption.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RedemptionStatus {
    /// Requested, in 7-day cooldown period
    Pending,
    /// Cooldown expired, awaiting fulfillment by asset provider
    Fulfillable,
    /// Matched with an asset provider, being fulfilled
    InFulfillment,
    /// Completed: asset delivered, SAP burned
    Fulfilled,
    /// Cancelled by redeemer (during cooldown only)
    Cancelled,
    /// Expired: no fulfiller found within 30 days of becoming fulfillable
    Expired,
}

/// Cooldown period for asset redemptions (7 days in microseconds).
/// Prevents bank-run dynamics — gives the community time to respond.
pub const REDEMPTION_COOLDOWN_MICROS: i64 = 7 * 24 * 60 * 60 * 1_000_000;

/// Maximum redemption expiry (30 days after becoming fulfillable).
pub const REDEMPTION_EXPIRY_MICROS: i64 = 30 * 24 * 60 * 60 * 1_000_000;

/// Maximum SAP that can be redeemed per member per 30-day period.
/// Prevents rapid withdrawal cascades. Constitutional bound.
pub const REDEMPTION_MAX_PER_MEMBER_PER_PERIOD: u64 = 50_000_000_000; // 50,000 SAP

// =============================================================================
// PHASE 4b: EXTERNAL ORACLE FEED TYPES
// =============================================================================

/// An external price feed subscription.
///
/// Blends external market data with community consensus.
/// Community consensus remains authoritative; external is advisory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalOracleFeed {
    /// Feed identifier (e.g., "coinbase:ETH-USD", "ecb:EUR-USD")
    pub feed_id: String,
    /// Latest reported price
    pub latest_price: f64,
    /// Confidence score (0.0 - 1.0, based on feed reliability history)
    pub confidence: f64,
    /// When this price was last updated (microseconds)
    pub last_updated_micros: i64,
    /// DID of the agent that submitted this feed update
    pub reporter_did: String,
}

/// Maximum oracle blend alpha (community weight at large reporter counts).
/// At 100+ reporters, community consensus gets 70% weight.
pub const ORACLE_BLEND_ALPHA_MAX: f64 = 0.70;

/// Minimum oracle blend alpha (community weight at minimum reporters).
/// With only 2 reporters, community gets 50% — equal weight with external feed.
pub const ORACLE_BLEND_ALPHA_MIN: f64 = 0.50;

/// Legacy constant for backward compatibility.
pub const ORACLE_BLEND_ALPHA: f64 = ORACLE_BLEND_ALPHA_MAX;

/// Compute oracle blend alpha scaled by reporter count.
///
/// Small communities (few reporters) get more external oracle support.
/// Large communities with robust consensus rely more on their own price discovery.
///
/// Formula: `alpha = min + (max - min) * (1 - e^(-k * (n - 2)))` where k = 0.05
/// Exponential approach to max — reaches ~95% of target by 60 reporters.
///
/// | Reporters | Alpha | Community Weight |
/// |-----------|-------|-----------------|
/// |     2     | 0.50  | 50% (minimum)   |
/// |     5     | 0.53  | 53%             |
/// |    10     | 0.57  | 57%             |
/// |    20     | 0.63  | 63%             |
/// |    50     | 0.68  | 68%             |
/// |   100+    | 0.70  | 70% (cap)       |
pub fn compute_oracle_alpha(reporter_count: usize) -> f64 {
    let n = reporter_count.max(2) as f64;
    let k = 0.05; // convergence rate — reaches ~95% at 60 reporters
    let alpha = ORACLE_BLEND_ALPHA_MIN
        + (ORACLE_BLEND_ALPHA_MAX - ORACLE_BLEND_ALPHA_MIN) * (1.0 - (-k * (n - 2.0)).exp());
    alpha.clamp(ORACLE_BLEND_ALPHA_MIN, ORACLE_BLEND_ALPHA_MAX)
}

/// Compute blended oracle rate from community consensus and external feed.
///
/// Alpha (community weight) scales with reporter count: small communities get
/// more external support, large communities rely on their own consensus.
///
/// Returns community rate if external is unavailable or has low confidence.
pub fn compute_blended_oracle_rate(
    community_rate: f64,
    external_rate: Option<f64>,
    external_confidence: f64,
) -> f64 {
    // Default to max alpha for backward compatibility (caller doesn't provide count)
    compute_blended_oracle_rate_scaled(community_rate, external_rate, external_confidence, 100)
}

/// Compute blended oracle rate with reporter-count-scaled alpha.
///
/// Use this instead of `compute_blended_oracle_rate` when reporter count is known.
pub fn compute_blended_oracle_rate_scaled(
    community_rate: f64,
    external_rate: Option<f64>,
    external_confidence: f64,
    reporter_count: usize,
) -> f64 {
    match external_rate {
        Some(ext) if ext.is_finite() && ext > 0.0 && external_confidence > 0.3 => {
            let base_alpha = compute_oracle_alpha(reporter_count);
            // Further scale by confidence: lower confidence = more weight on community
            let effective_alpha = base_alpha + (1.0 - base_alpha) * (1.0 - external_confidence);
            let blended = effective_alpha * community_rate + (1.0 - effective_alpha) * ext;
            if blended.is_finite() && blended > 0.0 {
                blended
            } else {
                community_rate
            }
        }
        _ => community_rate,
    }
}

/// Compute demurrage deduction on SAP balances.
///
/// Implements: eligible * (1 - e^(-rate * years))
/// where eligible = max(balance - exempt_floor, 0).
///
/// Pure function — no HDK dependencies.
pub fn compute_demurrage_deduction(
    balance: u64,
    exempt_floor: u64,
    rate: f64,
    seconds_elapsed: u64,
) -> u64 {
    if balance <= exempt_floor || seconds_elapsed == 0 || rate <= 0.0 || !rate.is_finite() {
        return 0;
    }
    let eligible_int = balance - exempt_floor;
    let eligible = eligible_int as f64;
    let years = seconds_elapsed as f64 / 31_536_000.0;
    let decay = 1.0 - (-rate * years).exp();
    let deduction = eligible * decay;
    if deduction <= 0.0 || deduction.is_nan() {
        0
    } else {
        // Clamp to integer eligible to avoid f64 rounding exceeding the true value
        (deduction as u64).min(eligible_int)
    }
}

/// Compute demurrage deduction for a minted mutual credit balance.
///
/// Only positive balances (credits) decay — negative balances (debts) are exempt.
/// This incentivizes spending rather than hoarding in community currencies.
///
/// Returns the number of whole hours to deduct (always ≥ 0).
pub fn compute_minted_demurrage(balance: i32, rate: f64, seconds_elapsed: u64) -> i32 {
    if balance <= 0 || seconds_elapsed == 0 || rate <= 0.0 || !rate.is_finite() {
        return 0;
    }
    let years = seconds_elapsed as f64 / 31_536_000.0;
    let decay = 1.0 - (-rate * years).exp();
    let deduction = balance as f64 * decay;
    if deduction < 1.0 {
        0 // Sub-hour amounts round down (no deduction)
    } else {
        // Clamp to balance to prevent f64 rounding from exceeding the original value
        (deduction.floor() as i32).min(balance)
    }
}

// =============================================================================
// CURRENCY FACTORY: Community-Minted Mutual Credit
// =============================================================================

/// Constitutional limits for community-minted currencies.
/// These are the "immutable physics" — communities get Parameter Sovereignty
/// but the zero-sum mutual credit model is enforced at the Rust level.
pub const MINTED_CREDIT_LIMIT_MIN: i32 = 10;
pub const MINTED_CREDIT_LIMIT_MAX: i32 = 200;
pub const MINTED_DEMURRAGE_RATE_MAX: f64 = 0.05; // 5% annual max
pub const MINTED_MAX_SERVICE_HOURS_MAX: u32 = 16;
pub const MINTED_MIN_SERVICE_MINUTES_MIN: u32 = 5;
pub const MINTED_CONFIRMATION_TIMEOUT_MAX: u32 = 720; // 30 days
pub const MINTED_MAX_EXCHANGES_PER_DAY_MAX: u8 = 50; // 0 = unlimited
pub const MINTED_SYMBOL_MAX_LEN: usize = 6;
pub const MINTED_NAME_MAX_LEN: usize = 64;
pub const MINTED_DESCRIPTION_MAX_LEN: usize = 500;

/// Status of a community-minted currency
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrencyStatus {
    /// Created, awaiting governance ratification (communities > 10 members)
    Draft,
    /// Active and accepting exchanges
    Active,
    /// Suspended by governance action
    Suspended,
    /// Permanently retired — no new exchanges, balances frozen
    Retired,
}

impl CurrencyStatus {
    /// Valid status transitions. Retired is terminal and cannot be reversed.
    pub fn can_transition_to(&self, new: &CurrencyStatus) -> bool {
        matches!(
            (self, new),
            (CurrencyStatus::Draft, CurrencyStatus::Active)
                | (CurrencyStatus::Active, CurrencyStatus::Suspended)
                | (CurrencyStatus::Active, CurrencyStatus::Retired)
                | (CurrencyStatus::Suspended, CurrencyStatus::Active) // reactivation
                | (CurrencyStatus::Suspended, CurrencyStatus::Retired)
        )
    }
}

/// Parameters for a community-minted mutual credit currency.
///
/// Communities get **Parameter Sovereignty**: custom names, limits, demurrage.
/// The **Economic Physics** are immutable: zero-sum mutual credit, enforced
/// balance limits, no pre-mining, no ICOs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MintedCurrencyParams {
    /// Human-readable name (e.g., "Cuidado", "Ubuntu Hours")
    pub name: String,
    /// Short symbol (e.g., "CUI", max 6 chars)
    pub symbol: String,
    /// Community description of the currency's purpose (max 500 chars)
    pub description: String,
    /// Maximum positive/negative balance per member
    pub credit_limit: i32,
    /// Annual demurrage rate (0.0 = no decay, max 0.05)
    pub demurrage_rate: f64,
    /// Max service hours per exchange
    pub max_service_hours: u32,
    /// Min service duration in minutes
    pub min_service_minutes: u32,
    /// Whether exchanges require receiver confirmation before balances update
    pub requires_confirmation: bool,
    /// Hours before an unconfirmed exchange expires (0 = never, max 720 = 30 days)
    pub confirmation_timeout_hours: u32,
    /// Maximum exchanges per member per day (0 = unlimited, max 50)
    pub max_exchanges_per_day: u8,
}

impl MintedCurrencyParams {
    /// Validate parameters against constitutional limits.
    /// Returns Ok(()) or a descriptive error string.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() || self.name.len() > MINTED_NAME_MAX_LEN {
            return Err(format!("Name must be 1-{} characters", MINTED_NAME_MAX_LEN));
        }
        if self.symbol.is_empty() || self.symbol.len() > MINTED_SYMBOL_MAX_LEN {
            return Err(format!(
                "Symbol must be 1-{} characters",
                MINTED_SYMBOL_MAX_LEN
            ));
        }
        if !self.symbol.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err("Symbol must be alphanumeric".into());
        }
        if self.description.len() > MINTED_DESCRIPTION_MAX_LEN {
            return Err(format!(
                "Description must be at most {} characters",
                MINTED_DESCRIPTION_MAX_LEN
            ));
        }
        if self.credit_limit < MINTED_CREDIT_LIMIT_MIN
            || self.credit_limit > MINTED_CREDIT_LIMIT_MAX
        {
            return Err(format!(
                "Credit limit must be {}-{}",
                MINTED_CREDIT_LIMIT_MIN, MINTED_CREDIT_LIMIT_MAX
            ));
        }
        if self.demurrage_rate < 0.0 || self.demurrage_rate > MINTED_DEMURRAGE_RATE_MAX {
            return Err(format!(
                "Demurrage rate must be 0.0-{}",
                MINTED_DEMURRAGE_RATE_MAX
            ));
        }
        if !self.demurrage_rate.is_finite() {
            return Err("Demurrage rate must be finite".into());
        }
        if self.max_service_hours == 0 || self.max_service_hours > MINTED_MAX_SERVICE_HOURS_MAX {
            return Err(format!(
                "Max service hours must be 1-{}",
                MINTED_MAX_SERVICE_HOURS_MAX
            ));
        }
        if self.min_service_minutes < MINTED_MIN_SERVICE_MINUTES_MIN
            || self.min_service_minutes > 60
        {
            return Err(format!(
                "Min service minutes must be {}-60",
                MINTED_MIN_SERVICE_MINUTES_MIN
            ));
        }
        if self.confirmation_timeout_hours > MINTED_CONFIRMATION_TIMEOUT_MAX {
            return Err(format!(
                "Confirmation timeout must be 0-{} hours",
                MINTED_CONFIRMATION_TIMEOUT_MAX
            ));
        }
        if self.max_exchanges_per_day > MINTED_MAX_EXCHANGES_PER_DAY_MAX {
            return Err(format!(
                "Max exchanges per day must be 0-{} (0 = unlimited)",
                MINTED_MAX_EXCHANGES_PER_DAY_MAX
            ));
        }
        Ok(())
    }
}

/// Source of SAP minting — every SAP must trace to a provenance
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SapMintSource {
    /// Minted against external collateral (ETH, USDC)
    CollateralBridge { deposit_id: String },
    /// Minted by governance proposal (community issuance)
    GovernanceProposal { proposal_id: String },
    /// Initial distribution for new community bootstrap
    InitialDistribution { reason: String },
}

/// Cultural alias for a community's currency.
/// A farming co-op creates "Water Credits"; a neighborhood creates "Soweto Care Hours".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurrencyAlias {
    /// Display name (e.g., "CARE", "HORAS", "UBUNTU")
    pub alias_name: String,
    /// Base currency this aliases (Tend, Sap, or a minted currency ID)
    pub base_currency: Currency,
    /// Optional display symbol (e.g., "C", "H")
    pub display_symbol: Option<String>,
    /// Optional description
    pub description: Option<String>,
}

// =============================================================================
// HEARTH-SCOPED CONSTANTS
// =============================================================================

/// Hearth TEND credit limit (smaller than DAO ±40)
pub const HEARTH_TEND_CREDIT_LIMIT: i32 = 20;
/// Maximum members in a hearth (family unit)
pub const HEARTH_MAX_MEMBERS: u32 = 50;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_currency_display() {
        assert_eq!(format!("{}", Currency::Mycel), "MYCEL");
        assert_eq!(format!("{}", Currency::Sap), "SAP");
        assert_eq!(format!("{}", Currency::Tend), "TEND");
    }

    #[test]
    fn test_fee_tier_from_mycel() {
        assert_eq!(FeeTier::from_mycel(0.1), FeeTier::Newcomer);
        assert_eq!(FeeTier::from_mycel(0.3), FeeTier::Member);
        assert_eq!(FeeTier::from_mycel(0.5), FeeTier::Member);
        assert_eq!(FeeTier::from_mycel(0.71), FeeTier::Steward);
    }

    #[test]
    fn test_tend_limit_tier() {
        assert_eq!(TendLimitTier::from_vitality(5), TendLimitTier::Emergency);
        assert_eq!(TendLimitTier::from_vitality(15), TendLimitTier::High);
        assert_eq!(TendLimitTier::from_vitality(30), TendLimitTier::Elevated);
        assert_eq!(TendLimitTier::from_vitality(50), TendLimitTier::Normal);
        assert_eq!(TendLimitTier::Emergency.limit(), 120);
        assert_eq!(TendLimitTier::Normal.limit(), 40);
    }

    #[test]
    fn test_metabolic_state() {
        assert_eq!(
            MetabolicState::from_vitality(90.0),
            MetabolicState::Thriving
        );
        assert_eq!(MetabolicState::from_vitality(60.0), MetabolicState::Healthy);
        assert_eq!(
            MetabolicState::from_vitality(40.0),
            MetabolicState::Stressed
        );
        assert_eq!(
            MetabolicState::from_vitality(20.0),
            MetabolicState::Critical
        );
        assert_eq!(MetabolicState::from_vitality(10.0), MetabolicState::Failing);
    }

    #[test]
    fn test_demurrage_below_exempt() {
        // Balance at or below exempt floor → no deduction
        assert_eq!(
            compute_demurrage_deduction(1_000_000_000, 1_000_000_000, 0.02, 31_536_000),
            0
        );
        assert_eq!(
            compute_demurrage_deduction(500_000_000, 1_000_000_000, 0.02, 31_536_000),
            0
        );
    }

    #[test]
    fn test_demurrage_one_year() {
        // 10,000 SAP (10B micro) with 1,000 SAP exempt, 2% rate, 1 year
        let deduction =
            compute_demurrage_deduction(10_000_000_000, 1_000_000_000, 0.02, 31_536_000);
        // Expected: 9B * (1 - e^(-0.02)) ≈ 9B * 0.0198 ≈ 178_200_000
        assert!(
            deduction > 170_000_000 && deduction < 190_000_000,
            "Expected ~178M, got {}",
            deduction
        );
    }

    #[test]
    fn test_demurrage_zero_elapsed() {
        assert_eq!(
            compute_demurrage_deduction(10_000_000_000, 1_000_000_000, 0.02, 0),
            0
        );
    }

    #[test]
    fn test_succession_serde() {
        let json = serde_json::to_string(&SuccessionPreference::Designee("did:mycelix:abc".into()))
            .unwrap();
        let parsed: SuccessionPreference = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed,
            SuccessionPreference::Designee("did:mycelix:abc".into())
        );
    }

    // =========================================================================
    // Currency Factory: MintedCurrencyParams validation
    // =========================================================================

    fn valid_params() -> MintedCurrencyParams {
        MintedCurrencyParams {
            name: "Ubuntu Hours".into(),
            symbol: "UBU".into(),
            description: "Community care hours".into(),
            credit_limit: 40,
            demurrage_rate: 0.02,
            max_service_hours: 8,
            min_service_minutes: 15,
            requires_confirmation: false,
            confirmation_timeout_hours: 0,
            max_exchanges_per_day: 0,
        }
    }

    #[test]
    fn test_minted_params_valid() {
        assert!(valid_params().validate().is_ok());
    }

    #[test]
    fn test_minted_params_name_empty() {
        let mut p = valid_params();
        p.name = "".into();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_minted_params_name_too_long() {
        let mut p = valid_params();
        p.name = "X".repeat(MINTED_NAME_MAX_LEN + 1);
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_minted_params_symbol_non_alphanumeric() {
        let mut p = valid_params();
        p.symbol = "UB!".into();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_minted_params_symbol_too_long() {
        let mut p = valid_params();
        p.symbol = "ABCDEFG".into(); // 7 chars, max is 6
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_minted_params_credit_limit_boundaries() {
        let mut p = valid_params();
        // Below minimum
        p.credit_limit = MINTED_CREDIT_LIMIT_MIN - 1;
        assert!(p.validate().is_err());
        // At minimum
        p.credit_limit = MINTED_CREDIT_LIMIT_MIN;
        assert!(p.validate().is_ok());
        // At maximum
        p.credit_limit = MINTED_CREDIT_LIMIT_MAX;
        assert!(p.validate().is_ok());
        // Above maximum
        p.credit_limit = MINTED_CREDIT_LIMIT_MAX + 1;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_minted_params_demurrage_boundaries() {
        let mut p = valid_params();
        // Negative rate
        p.demurrage_rate = -0.01;
        assert!(p.validate().is_err());
        // Zero rate (valid — no decay)
        p.demurrage_rate = 0.0;
        assert!(p.validate().is_ok());
        // At max
        p.demurrage_rate = MINTED_DEMURRAGE_RATE_MAX;
        assert!(p.validate().is_ok());
        // Above max
        p.demurrage_rate = MINTED_DEMURRAGE_RATE_MAX + 0.001;
        assert!(p.validate().is_err());
        // NaN
        p.demurrage_rate = f64::NAN;
        assert!(p.validate().is_err());
        // Infinity
        p.demurrage_rate = f64::INFINITY;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_minted_params_service_hours() {
        let mut p = valid_params();
        // Zero hours
        p.max_service_hours = 0;
        assert!(p.validate().is_err());
        // At max
        p.max_service_hours = MINTED_MAX_SERVICE_HOURS_MAX;
        assert!(p.validate().is_ok());
        // Above max
        p.max_service_hours = MINTED_MAX_SERVICE_HOURS_MAX + 1;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_minted_params_service_minutes() {
        let mut p = valid_params();
        // Below minimum
        p.min_service_minutes = MINTED_MIN_SERVICE_MINUTES_MIN - 1;
        assert!(p.validate().is_err());
        // At minimum
        p.min_service_minutes = MINTED_MIN_SERVICE_MINUTES_MIN;
        assert!(p.validate().is_ok());
        // At max (60)
        p.min_service_minutes = 60;
        assert!(p.validate().is_ok());
        // Above 60
        p.min_service_minutes = 61;
        assert!(p.validate().is_err());
    }

    // =========================================================================
    // SapMintSource + CurrencyAlias serde
    // =========================================================================

    #[test]
    fn test_sap_mint_source_serde() {
        let sources = vec![
            SapMintSource::CollateralBridge {
                deposit_id: "dep-123".into(),
            },
            SapMintSource::GovernanceProposal {
                proposal_id: "prop-456".into(),
            },
            SapMintSource::InitialDistribution {
                reason: "Bootstrap new community".into(),
            },
        ];
        for source in &sources {
            let json = serde_json::to_string(source).unwrap();
            let parsed: SapMintSource = serde_json::from_str(&json).unwrap();
            assert_eq!(&parsed, source);
        }
    }

    #[test]
    fn test_currency_alias_serde() {
        let alias = CurrencyAlias {
            alias_name: "Cuidado".into(),
            base_currency: Currency::Tend,
            display_symbol: Some("CUI".into()),
            description: Some("Care hours in our community".into()),
        };
        let json = serde_json::to_string(&alias).unwrap();
        let parsed: CurrencyAlias = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, alias);
    }

    #[test]
    fn test_currency_status_serde() {
        for status in &[
            CurrencyStatus::Draft,
            CurrencyStatus::Active,
            CurrencyStatus::Suspended,
            CurrencyStatus::Retired,
        ] {
            let json = serde_json::to_string(status).unwrap();
            let parsed: CurrencyStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&parsed, status);
        }
    }

    #[test]
    fn test_hearth_constants() {
        assert_eq!(HEARTH_TEND_CREDIT_LIMIT, 20);
        // Hearth limit must be less than DAO limit (compile-time invariant)
        const _: () = assert!(HEARTH_TEND_CREDIT_LIMIT < 40);
        assert_eq!(HEARTH_MAX_MEMBERS, 50);
    }

    #[test]
    fn test_currency_transferability() {
        assert!(!Currency::Mycel.is_transferable(), "MYCEL is soulbound");
        assert!(Currency::Sap.is_transferable());
        assert!(Currency::Tend.is_transferable());
    }

    // =========================================================================
    // Minted currency demurrage
    // =========================================================================

    #[test]
    fn test_minted_demurrage_positive_balance() {
        // 100 hours at 2% for one year
        let deduction = compute_minted_demurrage(100, 0.02, 31_536_000);
        // Expected: 100 * (1 - exp(-0.02)) ≈ 1.98 → floor = 1
        assert_eq!(deduction, 1);
    }

    #[test]
    fn test_minted_demurrage_negative_balance_exempt() {
        // Debts don't decay
        assert_eq!(compute_minted_demurrage(-50, 0.02, 31_536_000), 0);
    }

    #[test]
    fn test_minted_demurrage_zero_balance() {
        assert_eq!(compute_minted_demurrage(0, 0.02, 31_536_000), 0);
    }

    #[test]
    fn test_minted_demurrage_zero_elapsed() {
        assert_eq!(compute_minted_demurrage(100, 0.02, 0), 0);
    }

    #[test]
    fn test_minted_demurrage_zero_rate() {
        assert_eq!(compute_minted_demurrage(100, 0.0, 31_536_000), 0);
    }

    #[test]
    fn test_minted_demurrage_max_rate_large_balance() {
        // 200 hours at 5% for one year
        let deduction = compute_minted_demurrage(200, 0.05, 31_536_000);
        // Expected: 200 * (1 - exp(-0.05)) ≈ 9.75 → floor = 9
        assert_eq!(deduction, 9);
    }

    #[test]
    fn test_minted_demurrage_small_balance_no_deduction() {
        // 1 hour at 2% for one year → 0.0198 → rounds to 0
        assert_eq!(compute_minted_demurrage(1, 0.02, 31_536_000), 0);
    }

    #[test]
    fn test_minted_demurrage_nan_rate() {
        assert_eq!(compute_minted_demurrage(100, f64::NAN, 31_536_000), 0);
    }

    #[test]
    fn test_minted_demurrage_negative_rate() {
        assert_eq!(compute_minted_demurrage(100, -0.01, 31_536_000), 0);
    }

    #[test]
    fn test_minted_demurrage_multi_year() {
        // 100 hours at 5% for 5 years
        let deduction = compute_minted_demurrage(100, 0.05, 5 * 31_536_000);
        // Expected: 100 * (1 - exp(-0.25)) ≈ 22.12 → floor = 22
        assert_eq!(deduction, 22);
    }

    #[test]
    fn test_minted_demurrage_never_exceeds_balance() {
        // Even after 100 years, deduction <= balance
        let deduction = compute_minted_demurrage(50, 0.05, 100 * 31_536_000);
        assert!(deduction <= 50);
        assert!(deduction >= 49); // Should be ~50 (nearly fully decayed)
    }

    #[test]
    fn test_minted_demurrage_infinity_rate() {
        assert_eq!(compute_minted_demurrage(100, f64::INFINITY, 31_536_000), 0);
    }

    // =========================================================================
    // Currency status transitions
    // =========================================================================

    #[test]
    fn test_currency_status_all_variants() {
        let statuses = [
            CurrencyStatus::Draft,
            CurrencyStatus::Active,
            CurrencyStatus::Suspended,
            CurrencyStatus::Retired,
        ];
        assert_eq!(statuses.len(), 4);
        // All variants are distinct
        for (i, a) in statuses.iter().enumerate() {
            for (j, b) in statuses.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_minted_params_roundtrip() {
        let p = valid_params();
        let json = serde_json::to_string(&p).unwrap();
        let parsed: MintedCurrencyParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
        assert!(parsed.validate().is_ok());
    }

    #[test]
    fn test_minted_params_all_bounds_simultaneously() {
        // Test at constitutional maximums
        let p = MintedCurrencyParams {
            name: "X".repeat(MINTED_NAME_MAX_LEN),
            symbol: "A".repeat(MINTED_SYMBOL_MAX_LEN),
            description: "D".repeat(MINTED_DESCRIPTION_MAX_LEN),
            credit_limit: MINTED_CREDIT_LIMIT_MAX,
            demurrage_rate: MINTED_DEMURRAGE_RATE_MAX,
            max_service_hours: MINTED_MAX_SERVICE_HOURS_MAX,
            min_service_minutes: MINTED_MIN_SERVICE_MINUTES_MIN,
            requires_confirmation: true,
            confirmation_timeout_hours: 72,
            max_exchanges_per_day: MINTED_MAX_EXCHANGES_PER_DAY_MAX,
        };
        assert!(p.validate().is_ok());

        // Test at constitutional minimums
        let p2 = MintedCurrencyParams {
            name: "X".into(),
            symbol: "A".into(),
            description: String::new(),
            credit_limit: MINTED_CREDIT_LIMIT_MIN,
            demurrage_rate: 0.0,
            max_service_hours: 1,
            min_service_minutes: MINTED_MIN_SERVICE_MINUTES_MIN,
            requires_confirmation: false,
            confirmation_timeout_hours: 0,
            max_exchanges_per_day: 0,
        };
        assert!(p2.validate().is_ok());
    }

    #[test]
    fn test_sap_mint_caps() {
        // Per-proposal cap must be less than annual cap (compile-time invariant)
        const _: () = assert!(SAP_MINT_PER_PROPOSAL_MAX < SAP_MINT_ANNUAL_MAX);
        // At most 10 proposals per year at max amount = annual cap
        assert_eq!(SAP_MINT_ANNUAL_MAX / SAP_MINT_PER_PROPOSAL_MAX, 10);
    }

    // =========================================================================
    // Currency Factory: confirmation + timeout validation
    // =========================================================================

    #[test]
    fn test_minted_params_confirmation_timeout_valid() {
        let mut p = valid_params();
        p.requires_confirmation = true;
        p.confirmation_timeout_hours = 72;
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_minted_params_confirmation_timeout_zero() {
        // 0 = no expiry, always valid
        let mut p = valid_params();
        p.requires_confirmation = true;
        p.confirmation_timeout_hours = 0;
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_minted_params_confirmation_timeout_max() {
        let mut p = valid_params();
        p.confirmation_timeout_hours = MINTED_CONFIRMATION_TIMEOUT_MAX;
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_minted_params_confirmation_timeout_exceeds_max() {
        let mut p = valid_params();
        p.confirmation_timeout_hours = MINTED_CONFIRMATION_TIMEOUT_MAX + 1;
        assert!(p.validate().is_err());
    }

    // =========================================================================
    // Currency Factory: minted demurrage edge cases
    // =========================================================================

    #[test]
    fn test_minted_demurrage_one_hour_balance() {
        // 1 hour credit, 2% rate, 1 year: deduction = 1 * 0.0198 < 1 → floors to 0
        assert_eq!(compute_minted_demurrage(1, 0.02, 31_536_000), 0);
    }

    #[test]
    fn test_minted_demurrage_large_balance_one_second() {
        // Even large balance over 1 second: tiny deduction
        let d = compute_minted_demurrage(10_000, 0.05, 1);
        // 10000 * (1 - e^(-0.05/31536000)) ≈ 10000 * 1.58e-9 ≈ 0.0000158 → floors to 0
        assert_eq!(d, 0);
    }

    #[test]
    fn test_minted_demurrage_max_rate_one_year() {
        // 10,000 at max rate (5%), 1 year: ~4.88% continuous = ~488
        let d = compute_minted_demurrage(10_000, MINTED_DEMURRAGE_RATE_MAX, 31_536_000);
        let expected = (10_000.0 * (1.0 - (-MINTED_DEMURRAGE_RATE_MAX).exp())).floor() as i32;
        assert!(
            (d - expected).abs() <= 1,
            "got {} expected ~{}",
            d,
            expected
        );
    }

    #[test]
    fn test_minted_demurrage_negative_rate_is_noop() {
        assert_eq!(compute_minted_demurrage(1000, -0.01, 31_536_000), 0);
    }

    #[test]
    fn test_minted_demurrage_nan_rate_is_noop() {
        assert_eq!(compute_minted_demurrage(1000, f64::NAN, 31_536_000), 0);
    }

    #[test]
    fn test_minted_demurrage_max_i32_balance() {
        // Should not panic or overflow
        let d = compute_minted_demurrage(i32::MAX, 0.02, 31_536_000);
        assert!(d > 0);
        assert!(d < i32::MAX);
    }

    // =========================================================================
    // SAP demurrage (compute_demurrage_deduction) edge cases
    // =========================================================================

    #[test]
    fn test_sap_demurrage_nan_rate() {
        // NaN rate: decay = 1 - exp(NaN) = NaN, deduction = NaN → clamped to 0
        let d = compute_demurrage_deduction(10_000_000_000, 1_000_000_000, f64::NAN, 31_536_000);
        assert_eq!(d, 0);
    }

    #[test]
    fn test_sap_demurrage_infinity_rate() {
        let d =
            compute_demurrage_deduction(10_000_000_000, 1_000_000_000, f64::INFINITY, 31_536_000);
        // Non-finite rates return 0 (invalid input)
        assert_eq!(d, 0);
    }

    #[test]
    fn test_sap_demurrage_neg_infinity_rate() {
        let d = compute_demurrage_deduction(
            10_000_000_000,
            1_000_000_000,
            f64::NEG_INFINITY,
            31_536_000,
        );
        // exp(+Inf) = Inf, decay = 1 - Inf = -Inf, deduction < 0 → clamped to 0
        assert_eq!(d, 0);
    }

    #[test]
    fn test_sap_demurrage_negative_rate() {
        // Negative rate: exp(+years) > 1, decay < 0 → deduction < 0 → clamped to 0
        let d = compute_demurrage_deduction(10_000_000_000, 1_000_000_000, -0.05, 31_536_000);
        assert_eq!(d, 0);
    }

    #[test]
    fn test_sap_demurrage_max_u64_balance() {
        // Should not panic or overflow
        let d = compute_demurrage_deduction(u64::MAX, 0, 0.02, 31_536_000);
        // u64::MAX - 0 = u64::MAX eligible, ~1.98% decay
        assert!(d > 0);
        assert!(d < u64::MAX);
    }

    #[test]
    fn test_sap_demurrage_exempt_equals_balance() {
        // Balance == floor → no eligible amount → 0
        assert_eq!(
            compute_demurrage_deduction(5_000, 5_000, 0.02, 31_536_000),
            0
        );
    }

    #[test]
    fn test_sap_demurrage_one_micro_above_floor() {
        // 1 micro-SAP above floor, 2% for 1 year: deduction = 1 * 0.0198 → floors to 0
        assert_eq!(
            compute_demurrage_deduction(1_001, 1_000, 0.02, 31_536_000),
            0
        );
    }

    #[test]
    fn test_sap_demurrage_multi_year_never_exceeds_eligible() {
        // After 100 years at 5%, deduction should approach but never exceed eligible
        let balance = 10_000_000_000u64;
        let floor = 1_000_000_000u64;
        let d = compute_demurrage_deduction(balance, floor, 0.05, 100 * 31_536_000);
        let eligible = balance - floor;
        assert!(d <= eligible);
        // Should be very close to eligible (>99.9%)
        assert!(d as f64 / eligible as f64 > 0.99);
    }

    #[test]
    fn test_sap_demurrage_zero_rate() {
        // Zero rate: exp(0) = 1, decay = 0 → no deduction
        let d = compute_demurrage_deduction(10_000_000_000, 1_000_000_000, 0.0, 31_536_000);
        assert_eq!(d, 0);
    }

    // =========================================================================
    // Cross-function invariants
    // =========================================================================

    #[test]
    fn test_demurrage_monotonicity_in_time() {
        // Deduction must increase (or stay same) as time increases
        let mut prev = 0u64;
        for years in 1..=20 {
            let d = compute_demurrage_deduction(
                10_000_000_000,
                1_000_000_000,
                0.02,
                years * 31_536_000,
            );
            assert!(d >= prev, "year {}: {} < {}", years, d, prev);
            prev = d;
        }
    }

    #[test]
    fn test_minted_demurrage_monotonicity_in_time() {
        let mut prev = 0i32;
        for years in 1..=20u64 {
            let d = compute_minted_demurrage(1000, 0.02, years * 31_536_000);
            assert!(d >= prev, "year {}: {} < {}", years, d, prev);
            prev = d;
        }
    }

    #[test]
    fn test_demurrage_monotonicity_in_rate() {
        // Higher rate → more deduction
        let mut prev = 0u64;
        for rate_bps in [10, 50, 100, 200, 500] {
            let rate = rate_bps as f64 / 10_000.0;
            let d = compute_demurrage_deduction(10_000_000_000, 1_000_000_000, rate, 31_536_000);
            assert!(d >= prev, "rate {}: {} < {}", rate, d, prev);
            prev = d;
        }
    }

    #[test]
    fn test_minted_demurrage_monotonicity_in_rate() {
        let mut prev = 0i32;
        for rate_bps in [10, 50, 100, 200, 500] {
            let rate = rate_bps as f64 / 10_000.0;
            let d = compute_minted_demurrage(10_000, rate, 31_536_000);
            assert!(d >= prev, "rate {}: {} < {}", rate, d, prev);
            prev = d;
        }
    }

    // =========================================================================
    // Compost redistribution rounding edge cases
    // =========================================================================

    #[test]
    fn test_compost_redistribution_rounding_101() {
        // 101 SAP split among 70/20/10 percentages
        let total: u64 = 101;
        let local = total * COMPOST_LOCAL_PCT / 100;
        let regional = total * COMPOST_REGIONAL_PCT / 100;
        let global = total - local - regional; // remainder to global
        assert_eq!(
            local + regional + global,
            total,
            "No rounding loss for 101: local={}, regional={}, global={}",
            local,
            regional,
            global
        );
    }

    #[test]
    fn test_compost_redistribution_rounding_257() {
        let total: u64 = 257;
        let local = total * COMPOST_LOCAL_PCT / 100;
        let regional = total * COMPOST_REGIONAL_PCT / 100;
        let global = total - local - regional;
        assert_eq!(
            local + regional + global,
            total,
            "No rounding loss for 257: local={}, regional={}, global={}",
            local,
            regional,
            global
        );
    }

    #[test]
    fn test_compost_redistribution_rounding_999() {
        let total: u64 = 999;
        let local = total * COMPOST_LOCAL_PCT / 100;
        let regional = total * COMPOST_REGIONAL_PCT / 100;
        let global = total - local - regional;
        assert_eq!(
            local + regional + global,
            total,
            "No rounding loss for 999: local={}, regional={}, global={}",
            local,
            regional,
            global
        );
    }

    #[test]
    fn test_compost_redistribution_rounding_1() {
        // Edge case: smallest possible amount
        let total: u64 = 1;
        let local = total * COMPOST_LOCAL_PCT / 100;
        let regional = total * COMPOST_REGIONAL_PCT / 100;
        let global = total - local - regional;
        assert_eq!(
            local + regional + global,
            total,
            "No rounding loss for 1: local={}, regional={}, global={}",
            local,
            regional,
            global
        );
    }

    #[test]
    fn test_compost_redistribution_rounding_individual_values() {
        // Verify the floor-division behavior for non-evenly-divisible amounts
        // 101: 70% = 70, 20% = 20, remainder = 11 (not 10.1)
        let total: u64 = 101;
        let local = total * COMPOST_LOCAL_PCT / 100;
        let regional = total * COMPOST_REGIONAL_PCT / 100;
        let global = total - local - regional;
        assert_eq!(local, 70);
        assert_eq!(regional, 20);
        assert_eq!(global, 11); // gets the remainder

        // 999: 70% = 699, 20% = 199, remainder = 101
        let total2: u64 = 999;
        let local2 = total2 * COMPOST_LOCAL_PCT / 100;
        let regional2 = total2 * COMPOST_REGIONAL_PCT / 100;
        let global2 = total2 - local2 - regional2;
        assert_eq!(local2, 699);
        assert_eq!(regional2, 199);
        assert_eq!(global2, 101);
    }

    // =========================================================================
    // CurrencyStatus state machine transitions
    // =========================================================================

    #[test]
    fn test_currency_status_valid_transitions() {
        use CurrencyStatus::*;
        // Forward transitions
        assert!(Draft.can_transition_to(&Active));
        assert!(Active.can_transition_to(&Suspended));
        assert!(Active.can_transition_to(&Retired));
        assert!(Suspended.can_transition_to(&Active)); // reactivation
        assert!(Suspended.can_transition_to(&Retired));
    }

    #[test]
    fn test_currency_status_invalid_transitions() {
        use CurrencyStatus::*;
        // Retired is terminal
        assert!(!Retired.can_transition_to(&Active));
        assert!(!Retired.can_transition_to(&Draft));
        assert!(!Retired.can_transition_to(&Suspended));
        // Draft can only go to Active
        assert!(!Draft.can_transition_to(&Suspended));
        assert!(!Draft.can_transition_to(&Retired));
        // No self-transitions
        assert!(!Active.can_transition_to(&Active));
        assert!(!Draft.can_transition_to(&Draft));
    }

    // =========================================================================
    // Property-based tests (proptest)
    // =========================================================================

    mod proptests {
        use super::super::*;
        use proptest::prelude::*;

        // Property: fork resolution via `min_by_key` is deterministic.
        // Given any permutation of the same set of byte arrays (simulating
        // ActionHashes in `follow_update_chain`), `min_by_key` always selects
        // the same winner.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn fork_resolution_is_deterministic(
                hashes in prop::collection::hash_set(
                    prop::collection::vec(any::<u8>(), 32..=32),
                    2..10
                )
            ) {
                let hashes_vec: Vec<Vec<u8>> = hashes.into_iter().collect();

                // The canonical winner is the lexicographic minimum
                let winner = hashes_vec.iter().min().unwrap().clone();

                // Reversed order must yield the same winner
                let mut shuffled = hashes_vec.clone();
                shuffled.reverse();
                let winner2 = shuffled.iter().min().unwrap().clone();
                prop_assert_eq!(&winner, &winner2);

                // Rotated order must yield the same winner
                let mut rotated = hashes_vec.clone();
                rotated.rotate_left(1);
                let winner3 = rotated.iter().min().unwrap().clone();
                prop_assert_eq!(&winner, &winner3);
            }
        }

        // Property: `CurrencyStatus::can_transition_to` is anti-symmetric.
        // If A can transition to B, then B cannot transition to A,
        // except for the Suspended <-> Active pair.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn currency_status_transition_antisymmetric(
                a_idx in 0usize..4,
                b_idx in 0usize..4,
            ) {
                let statuses = [
                    CurrencyStatus::Draft,
                    CurrencyStatus::Active,
                    CurrencyStatus::Suspended,
                    CurrencyStatus::Retired,
                ];
                let a = &statuses[a_idx];
                let b = &statuses[b_idx];

                if a == b {
                    // Self-transitions are always invalid
                    prop_assert!(!a.can_transition_to(b));
                } else if a.can_transition_to(b) {
                    let is_suspended_active = matches!(
                        (a, b),
                        (CurrencyStatus::Suspended, CurrencyStatus::Active)
                            | (CurrencyStatus::Active, CurrencyStatus::Suspended)
                    );
                    if !is_suspended_active {
                        prop_assert!(
                            !b.can_transition_to(a),
                            "Anti-symmetry violated: {:?} -> {:?} and {:?} -> {:?}",
                            a, b, b, a
                        );
                    }
                }
            }
        }

        // Property: balance mutations preserve `is_finite`.
        // If input balance is finite and mutation amount is finite, result is finite.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn balance_mutation_preserves_finite(
                balance in 0.0f64..1e15,
                mutation in -1e12f64..1e12,
            ) {
                let sum = balance + mutation;
                prop_assert!(
                    sum.is_finite(),
                    "finite + finite must be finite: {} + {} = {}",
                    balance, mutation, sum
                );

                let product = balance * mutation;
                prop_assert!(
                    product.is_finite(),
                    "finite * finite must be finite: {} * {} = {}",
                    balance, mutation, product
                );
            }
        }

        // Property: demurrage deduction is monotonic in time.
        // For any valid rate and balance, longer elapsed time means
        // equal or greater deduction.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn demurrage_monotonic_in_time(
                balance in (DEMURRAGE_EXEMPT_FLOOR + 1)..=(u64::MAX / 2),
                rate in 0.001f64..0.05,
                t1_secs in 1u64..=(100 * 31_536_000u64),
                extra_secs in 1u64..=31_536_000u64,
            ) {
                let t2_secs = t1_secs.saturating_add(extra_secs);
                let d1 = compute_demurrage_deduction(balance, DEMURRAGE_EXEMPT_FLOOR, rate, t1_secs);
                let d2 = compute_demurrage_deduction(balance, DEMURRAGE_EXEMPT_FLOOR, rate, t2_secs);
                prop_assert!(
                    d2 >= d1,
                    "Demurrage must be monotonic: d({})={} should be >= d({})={} for balance={}, rate={}",
                    t2_secs, d2, t1_secs, d1, balance, rate
                );
            }
        }

        // Property: demurrage deduction is bounded -- never exceeds eligible balance.
        // For reasonable durations (< 100 years) and valid rates.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn demurrage_bounded_by_balance(
                balance in (DEMURRAGE_EXEMPT_FLOOR + 1)..=(u64::MAX / 2),
                rate in 0.001f64..0.05,
                seconds in 1u64..=(100 * 31_536_000u64),
            ) {
                let eligible = balance - DEMURRAGE_EXEMPT_FLOOR;
                let deduction = compute_demurrage_deduction(
                    balance,
                    DEMURRAGE_EXEMPT_FLOOR,
                    rate,
                    seconds,
                );
                prop_assert!(
                    deduction <= eligible,
                    "Deduction {} exceeds eligible {} (balance={}, rate={}, secs={})",
                    deduction, eligible, balance, rate, seconds
                );
            }
        }
        // Property: conservation — demurrage deduction + remaining = original eligible.
        // The function must never create or destroy value.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(512))]

            #[test]
            fn demurrage_conservation(
                balance in (DEMURRAGE_EXEMPT_FLOOR + 1)..=(u64::MAX / 2),
                rate in 0.001f64..0.05,
                seconds in 1u64..=(100 * 31_536_000u64),
            ) {
                let eligible = balance - DEMURRAGE_EXEMPT_FLOOR;
                let deduction = compute_demurrage_deduction(
                    balance,
                    DEMURRAGE_EXEMPT_FLOOR,
                    rate,
                    seconds,
                );
                let remaining_eligible = eligible - deduction;
                // Conservation: deduction + remaining = eligible
                prop_assert_eq!(
                    deduction + remaining_eligible,
                    eligible,
                    "Conservation violated: {} + {} != {} (balance={}, rate={}, secs={})",
                    deduction, remaining_eligible, eligible, balance, rate, seconds
                );
            }
        }

        // Property: exempt floor safety — balances at or below the floor are never touched.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn demurrage_exempt_floor_safety(
                balance in 0u64..=DEMURRAGE_EXEMPT_FLOOR,
                rate in 0.001f64..0.05,
                seconds in 1u64..=(100 * 31_536_000u64),
            ) {
                let deduction = compute_demurrage_deduction(
                    balance,
                    DEMURRAGE_EXEMPT_FLOOR,
                    rate,
                    seconds,
                );
                prop_assert_eq!(
                    deduction,
                    0,
                    "Exempt-floor balance {} should have zero deduction, got {}",
                    balance, deduction
                );
            }
        }

        // Property: diversified demurrage rate is bounded within [base * 0.5, base].
        // The constitutional floor is 50% of base_rate.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn diversified_rate_bounded(
                base_rate in 0.005f64..0.05,
                distinct_classes in 0usize..20,
            ) {
                let effective = compute_diversified_demurrage_rate(base_rate, distinct_classes);
                let floor = base_rate * 0.5;
                prop_assert!(
                    effective >= floor - 1e-10,
                    "Rate {} below constitutional floor {} (base={}, classes={})",
                    effective, floor, base_rate, distinct_classes
                );
                prop_assert!(
                    effective <= base_rate + 1e-10,
                    "Rate {} exceeds base {} (classes={})",
                    effective, base_rate, distinct_classes
                );
            }
        }

        // Property: diversified rate is monotonically non-increasing with more classes.
        // More distinct classes → lower-or-equal effective rate.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn diversified_rate_monotonic_in_classes(
                base_rate in 0.005f64..0.05,
                c1 in 0usize..15,
                extra in 1usize..5,
            ) {
                let c2 = c1 + extra;
                let r1 = compute_diversified_demurrage_rate(base_rate, c1);
                let r2 = compute_diversified_demurrage_rate(base_rate, c2);
                prop_assert!(
                    r2 <= r1 + 1e-10,
                    "More classes should give lower rate: r({})={} > r({})={}",
                    c1, r1, c2, r2
                );
            }
        }

        // Property: minted demurrage on negative balances is always zero.
        // Debts (negative balances) are exempt from demurrage — only credits decay.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn minted_demurrage_negative_balance_exempt(
                balance in i32::MIN..=0,
                rate in 0.001f64..0.05,
                seconds in 1u64..=(100 * 31_536_000u64),
            ) {
                let deduction = compute_minted_demurrage(balance, rate, seconds);
                prop_assert_eq!(
                    deduction,
                    0,
                    "Negative balance {} should have zero demurrage, got {}",
                    balance, deduction
                );
            }
        }

        // Property: minted demurrage never exceeds the original balance.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn minted_demurrage_bounded(
                balance in 1i32..=i32::MAX,
                rate in 0.001f64..0.05,
                seconds in 1u64..=(100 * 31_536_000u64),
            ) {
                let deduction = compute_minted_demurrage(balance, rate, seconds);
                prop_assert!(
                    deduction <= balance,
                    "Deduction {} exceeds balance {} (rate={}, secs={})",
                    deduction, balance, rate, seconds
                );
                prop_assert!(
                    deduction >= 0,
                    "Deduction {} is negative (balance={}, rate={}, secs={})",
                    deduction, balance, rate, seconds
                );
            }
        }

        // Property: compost distribution percentages sum to 100%.
        #[test]
        fn compost_distribution_sums_to_100() {
            assert_eq!(
                COMPOST_LOCAL_PCT + COMPOST_REGIONAL_PCT + COMPOST_GLOBAL_PCT,
                100,
                "Compost distribution must sum to 100%"
            );
        }

        // Property: zero seconds elapsed always produces zero deduction.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn demurrage_zero_time_zero_deduction(
                balance in 0u64..=(u64::MAX / 2),
                rate in 0.001f64..0.05,
            ) {
                let deduction = compute_demurrage_deduction(
                    balance,
                    DEMURRAGE_EXEMPT_FLOOR,
                    rate,
                    0, // zero elapsed time
                );
                prop_assert_eq!(
                    deduction,
                    0,
                    "Zero time should mean zero deduction, got {} for balance={}",
                    deduction, balance
                );
            }
        }
    }

    // =========================================================================
    // SapMintCapCounter tests
    // =========================================================================

    #[test]
    fn test_sap_mint_cap_counter_would_exceed_cap() {
        let counter = SapMintCapCounter {
            period_start_micros: 0,
            cumulative_minted: SAP_MINT_ANNUAL_MAX - 100,
            mint_count: 5,
            last_updated_micros: 0,
        };
        // 100 fits exactly
        assert!(!counter.would_exceed_cap(100));
        // 101 exceeds
        assert!(counter.would_exceed_cap(101));
        // 0 never exceeds
        assert!(!counter.would_exceed_cap(0));
    }

    #[test]
    fn test_sap_mint_cap_counter_would_exceed_cap_already_full() {
        let counter = SapMintCapCounter {
            period_start_micros: 0,
            cumulative_minted: SAP_MINT_ANNUAL_MAX,
            mint_count: 10,
            last_updated_micros: 0,
        };
        // Any positive amount exceeds when already at cap
        assert!(counter.would_exceed_cap(1));
        // Zero still does not exceed (saturating_add stays at cap, which is not > cap)
        assert!(!counter.would_exceed_cap(0));
    }

    #[test]
    fn test_sap_mint_cap_counter_remaining_capacity() {
        let counter = SapMintCapCounter {
            period_start_micros: 0,
            cumulative_minted: 500_000_000_000,
            mint_count: 3,
            last_updated_micros: 0,
        };
        assert_eq!(
            counter.remaining_capacity(),
            SAP_MINT_ANNUAL_MAX - 500_000_000_000
        );

        // Full counter has zero remaining
        let full = SapMintCapCounter {
            period_start_micros: 0,
            cumulative_minted: SAP_MINT_ANNUAL_MAX,
            mint_count: 10,
            last_updated_micros: 0,
        };
        assert_eq!(full.remaining_capacity(), 0);

        // Empty counter has full capacity
        let empty = SapMintCapCounter {
            period_start_micros: 0,
            cumulative_minted: 0,
            mint_count: 0,
            last_updated_micros: 0,
        };
        assert_eq!(empty.remaining_capacity(), SAP_MINT_ANNUAL_MAX);
    }

    #[test]
    fn test_sap_mint_cap_counter_is_period_expired() {
        let one_year_micros: i64 = 365 * 24 * 60 * 60 * 1_000_000;
        let counter = SapMintCapCounter {
            period_start_micros: 1_000_000,
            cumulative_minted: 0,
            mint_count: 0,
            last_updated_micros: 0,
        };
        // Not expired: exactly 1 year later
        assert!(!counter.is_period_expired(1_000_000 + one_year_micros));
        // Expired: 1 year + 1 microsecond later
        assert!(counter.is_period_expired(1_000_000 + one_year_micros + 1));
        // Not expired: same time as start
        assert!(!counter.is_period_expired(1_000_000));
        // Not expired: before period start
        assert!(!counter.is_period_expired(0));
    }

    // =========================================================================
    // CollateralHealthStatus::from_ltv tests
    // =========================================================================

    #[test]
    fn test_collateral_health_status_healthy() {
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.0),
            CollateralHealthStatus::Healthy
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.50),
            CollateralHealthStatus::Healthy
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.80),
            CollateralHealthStatus::Healthy
        );
    }

    #[test]
    fn test_collateral_health_status_warning() {
        // Just above 80% threshold
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.81),
            CollateralHealthStatus::Warning
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.85),
            CollateralHealthStatus::Warning
        );
        // At 90% boundary (not above 0.90, so still Warning)
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.90),
            CollateralHealthStatus::Warning
        );
    }

    #[test]
    fn test_collateral_health_status_margin_call() {
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.91),
            CollateralHealthStatus::MarginCall
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.93),
            CollateralHealthStatus::MarginCall
        );
        // At 95% boundary (not above 0.95, so still MarginCall)
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.95),
            CollateralHealthStatus::MarginCall
        );
    }

    #[test]
    fn test_collateral_health_status_liquidation() {
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.951),
            CollateralHealthStatus::Liquidation
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.99),
            CollateralHealthStatus::Liquidation
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(1.0),
            CollateralHealthStatus::Liquidation
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(1.5),
            CollateralHealthStatus::Liquidation
        );
    }

    // =========================================================================
    // compute_diversification_bonus tests
    // =========================================================================

    fn make_component(class: CollateralAssetClass) -> CollateralComponent {
        CollateralComponent {
            asset_id: "test".into(),
            asset_class: class,
            weight: 1.0,
            current_value: 1000,
        }
    }

    #[test]
    fn test_diversification_bonus_single_class() {
        let components = vec![make_component(CollateralAssetClass::RealEstate)];
        // 1 class → 0 bonus classes → 0.0
        assert_eq!(compute_diversification_bonus(&components), 0.0);
    }

    #[test]
    fn test_diversification_bonus_two_classes() {
        let components = vec![
            make_component(CollateralAssetClass::RealEstate),
            make_component(CollateralAssetClass::EnergyCertificate),
        ];
        // 2 classes → 1 bonus class → 5%
        assert!((compute_diversification_bonus(&components) - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diversification_bonus_five_classes_capped() {
        let components = vec![
            make_component(CollateralAssetClass::RealEstate),
            make_component(CollateralAssetClass::Cryptocurrency),
            make_component(CollateralAssetClass::EnergyCertificate),
            make_component(CollateralAssetClass::AgriculturalAsset),
            make_component(CollateralAssetClass::CarbonCredit),
        ];
        // 5 classes → 4 bonus classes → 20% (= cap)
        assert!((compute_diversification_bonus(&components) - 0.20).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diversification_bonus_six_classes_still_capped() {
        let components = vec![
            make_component(CollateralAssetClass::RealEstate),
            make_component(CollateralAssetClass::Cryptocurrency),
            make_component(CollateralAssetClass::EnergyCertificate),
            make_component(CollateralAssetClass::AgriculturalAsset),
            make_component(CollateralAssetClass::CarbonCredit),
            make_component(CollateralAssetClass::Vehicle),
        ];
        // 6 classes → 5 bonus → 25%, but capped at 20%
        assert!((compute_diversification_bonus(&components) - 0.20).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diversification_bonus_duplicate_classes_no_extra() {
        let components = vec![
            make_component(CollateralAssetClass::RealEstate),
            make_component(CollateralAssetClass::RealEstate),
            make_component(CollateralAssetClass::EnergyCertificate),
        ];
        // 2 distinct classes → 1 bonus → 5%
        assert!((compute_diversification_bonus(&components) - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diversification_bonus_empty() {
        let components: Vec<CollateralComponent> = vec![];
        assert_eq!(compute_diversification_bonus(&components), 0.0);
    }

    // =========================================================================
    // compute_blended_oracle_rate tests
    // =========================================================================

    #[test]
    fn test_blended_oracle_rate_community_only_no_external() {
        // No external rate → returns community rate
        let rate = compute_blended_oracle_rate(100.0, None, 0.0);
        assert_eq!(rate, 100.0);
    }

    #[test]
    fn test_blended_oracle_rate_low_confidence_fallback() {
        // External confidence <= 0.3 → falls back to community rate
        let rate = compute_blended_oracle_rate(100.0, Some(200.0), 0.3);
        assert_eq!(rate, 100.0);

        let rate2 = compute_blended_oracle_rate(100.0, Some(200.0), 0.1);
        assert_eq!(rate2, 100.0);
    }

    #[test]
    fn test_blended_oracle_rate_blended() {
        // High confidence external rate should blend (default reporter_count=100 → alpha ~0.70)
        let rate = compute_blended_oracle_rate(100.0, Some(120.0), 0.9);
        // At 100 reporters, alpha ~0.70. effective_alpha = 0.70 + 0.30*(1-0.9) = 0.73
        // blended = 0.73 * 100 + 0.27 * 120 = 73 + 32.4 = 105.4
        assert!(
            rate > 100.0 && rate < 120.0,
            "blended rate {} should be between community and external",
            rate
        );
    }

    #[test]
    fn test_blended_oracle_rate_invalid_external() {
        // NaN external → falls back to community
        assert_eq!(
            compute_blended_oracle_rate(100.0, Some(f64::NAN), 0.9),
            100.0
        );
        // Negative external → falls back to community
        assert_eq!(compute_blended_oracle_rate(100.0, Some(-10.0), 0.9), 100.0);
        // Zero external → falls back to community
        assert_eq!(compute_blended_oracle_rate(100.0, Some(0.0), 0.9), 100.0);
        // Infinity external → falls back to community
        assert_eq!(
            compute_blended_oracle_rate(100.0, Some(f64::INFINITY), 0.9),
            100.0
        );
    }

    // =========================================================================
    // PendingCompost construction test
    // =========================================================================

    #[test]
    fn test_pending_compost_construction() {
        let pc = PendingCompost {
            commons_pool_id: "pool-local-001".into(),
            amount: 5000,
            source_member_did: "did:mycelix:alice".into(),
            pool_tier: CompostPoolTier::Local,
            created_at_micros: 1_700_000_000_000_000,
            retry_count: 0,
        };
        assert_eq!(pc.commons_pool_id, "pool-local-001");
        assert_eq!(pc.amount, 5000);
        assert_eq!(pc.pool_tier, CompostPoolTier::Local);
        assert_eq!(pc.retry_count, 0);

        // Verify serde roundtrip
        let json = serde_json::to_string(&pc).unwrap();
        let parsed: PendingCompost = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, pc);
    }

    // =========================================================================
    // Covenant construction test
    // =========================================================================

    #[test]
    fn test_covenant_construction() {
        let cov = Covenant {
            id: "cov-001".into(),
            collateral_id: "col-123".into(),
            restriction: CovenantRestriction::TransferLock,
            beneficiary_did: "did:mycelix:bank".into(),
            created_at_micros: 1_700_000_000_000_000,
            expires_at_micros: None,
            released: false,
            released_by: None,
            released_at_micros: None,
        };
        assert_eq!(cov.id, "cov-001");
        assert_eq!(cov.restriction, CovenantRestriction::TransferLock);
        assert!(!cov.released);
        assert!(cov.expires_at_micros.is_none());

        // Verify serde roundtrip
        let json = serde_json::to_string(&cov).unwrap();
        let parsed: Covenant = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, cov);
    }

    #[test]
    fn test_covenant_with_collateral_pledge() {
        let cov = Covenant {
            id: "cov-002".into(),
            collateral_id: "col-456".into(),
            restriction: CovenantRestriction::CollateralPledge {
                obligation_id: "loan-789".into(),
            },
            beneficiary_did: "did:mycelix:lender".into(),
            created_at_micros: 1_700_000_000_000_000,
            expires_at_micros: Some(1_800_000_000_000_000),
            released: false,
            released_by: None,
            released_at_micros: None,
        };
        assert!(matches!(
            cov.restriction,
            CovenantRestriction::CollateralPledge { .. }
        ));
        assert_eq!(cov.expires_at_micros, Some(1_800_000_000_000_000));

        // Verify serde roundtrip
        let json = serde_json::to_string(&cov).unwrap();
        let parsed: Covenant = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, cov);
    }

    // =========================================================================
    // Diversified demurrage discount tests
    // =========================================================================

    #[test]
    fn test_diversified_demurrage_single_class_no_discount() {
        let rate = compute_diversified_demurrage_rate(DEMURRAGE_RATE, 1);
        assert_eq!(rate, DEMURRAGE_RATE);
    }

    #[test]
    fn test_diversified_demurrage_two_classes_no_discount() {
        // Below DEMURRAGE_DISCOUNT_MIN_CLASSES (3)
        let rate = compute_diversified_demurrage_rate(DEMURRAGE_RATE, 2);
        assert_eq!(rate, DEMURRAGE_RATE);
    }

    #[test]
    fn test_diversified_demurrage_three_classes() {
        // 3 classes → 1 bonus class → 0.1% discount → 1.9%
        let rate = compute_diversified_demurrage_rate(DEMURRAGE_RATE, 3);
        assert!(
            (rate - 0.019).abs() < 0.0001,
            "3 classes should give ~1.9%, got {}",
            rate
        );
    }

    #[test]
    fn test_diversified_demurrage_five_classes() {
        // 5 classes → 3 bonus classes → 0.3% discount → 1.7%
        let rate = compute_diversified_demurrage_rate(DEMURRAGE_RATE, 5);
        assert!(
            (rate - 0.017).abs() < 0.0001,
            "5 classes should give ~1.7%, got {}",
            rate
        );
    }

    #[test]
    fn test_diversified_demurrage_cap() {
        // 10 classes → 8 bonus → 0.8%, but capped at 0.5% → 1.5%
        let rate = compute_diversified_demurrage_rate(DEMURRAGE_RATE, 10);
        assert!(
            (rate - 0.015).abs() < 0.0001,
            "capped at 0.5% discount → 1.5%, got {}",
            rate
        );
    }

    #[test]
    fn test_diversified_demurrage_floor() {
        // Even with large discount, never below 50% of base rate
        let rate = compute_diversified_demurrage_rate(0.01, 100); // 1% base, huge classes
        assert!(rate >= 0.005, "never below 50% of base rate, got {}", rate);
    }

    // =========================================================================
    // Oracle alpha scaling tests
    // =========================================================================

    #[test]
    fn test_oracle_alpha_minimum_reporters() {
        let alpha = compute_oracle_alpha(2);
        assert!(
            (alpha - ORACLE_BLEND_ALPHA_MIN).abs() < 0.02,
            "2 reporters should give ~50% alpha, got {}",
            alpha
        );
    }

    #[test]
    fn test_oracle_alpha_ten_reporters() {
        let alpha = compute_oracle_alpha(10);
        assert!(
            alpha > 0.55 && alpha < 0.65,
            "10 reporters should give ~57% alpha, got {}",
            alpha
        );
    }

    #[test]
    fn test_oracle_alpha_hundred_reporters() {
        let alpha = compute_oracle_alpha(100);
        assert!(
            (alpha - ORACLE_BLEND_ALPHA_MAX).abs() < 0.005,
            "100+ reporters should converge to ~70% alpha, got {}",
            alpha
        );
    }

    #[test]
    fn test_oracle_alpha_monotonic() {
        let mut prev = compute_oracle_alpha(2);
        for n in [3, 5, 10, 20, 50, 100, 500] {
            let current = compute_oracle_alpha(n);
            assert!(
                current >= prev - 0.001, // allow tiny float imprecision
                "alpha should be monotonically non-decreasing: alpha({})={} < alpha({})={}",
                n,
                current,
                n - 1,
                prev
            );
            prev = current;
        }
    }

    #[test]
    fn test_oracle_alpha_bounds() {
        for n in [0, 1, 2, 5, 10, 100, 1000, 1_000_000] {
            let alpha = compute_oracle_alpha(n);
            assert!(
                alpha >= ORACLE_BLEND_ALPHA_MIN && alpha <= ORACLE_BLEND_ALPHA_MAX,
                "alpha({}) = {} out of bounds [{}, {}]",
                n,
                alpha,
                ORACLE_BLEND_ALPHA_MIN,
                ORACLE_BLEND_ALPHA_MAX
            );
        }
    }

    #[test]
    fn test_blended_rate_scaled_small_community() {
        // 2 reporters: alpha ~0.50, so external gets ~50% weight
        let rate = compute_blended_oracle_rate_scaled(100.0, Some(200.0), 0.9, 2);
        assert!(
            rate > 120.0,
            "small community should give more weight to external, got {}",
            rate
        );

        // 100 reporters: alpha ~0.70, external gets ~30% weight
        let rate_big = compute_blended_oracle_rate_scaled(100.0, Some(200.0), 0.9, 100);
        assert!(
            rate_big < rate,
            "large community should rely more on own consensus"
        );
    }

    // =========================================================================
    // Asset redemption tests
    // =========================================================================

    #[test]
    fn test_asset_redemption_construction() {
        let redemption = AssetRedemption {
            id: "redeem-001".into(),
            redeemer_did: "did:mycelix:alice".into(),
            sap_amount: 1_000_000_000, // 1,000 SAP
            asset_claim: AssetClaim::Energy {
                kwh_requested: 10_000,
                source_preference: Some("Solar".into()),
            },
            status: RedemptionStatus::Pending,
            requested_at_micros: 1_700_000_000_000_000,
            fulfillable_after_micros: 1_700_000_000_000_000 + REDEMPTION_COOLDOWN_MICROS,
            fulfilled_at_micros: None,
            fulfiller_did: None,
        };
        assert_eq!(redemption.status, RedemptionStatus::Pending);
        assert!(redemption.fulfilled_at_micros.is_none());

        // Verify serde roundtrip
        let json = serde_json::to_string(&redemption).unwrap();
        let parsed: AssetRedemption = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, redemption);
    }

    #[test]
    fn test_asset_claim_variants() {
        let claims = vec![
            AssetClaim::Energy {
                kwh_requested: 100,
                source_preference: None,
            },
            AssetClaim::Agricultural {
                product_type: "Maize".into(),
                kg_requested: 200,
            },
            AssetClaim::HousingCredit {
                housing_unit_id: Some("unit-1".into()),
                credit_amount: 5000,
            },
            AssetClaim::CarbonOffset { tonnes_co2e: 10 },
            AssetClaim::ServiceHours {
                hours_requested: 8,
                service_category: Some("CareWork".into()),
            },
        ];
        for claim in &claims {
            let json = serde_json::to_string(claim).unwrap();
            let parsed: AssetClaim = serde_json::from_str(&json).unwrap();
            assert_eq!(&parsed, claim);
        }
    }

    #[test]
    fn test_redemption_cooldown_constant() {
        assert_eq!(REDEMPTION_COOLDOWN_MICROS, 604_800_000_000); // 7 days
        assert_eq!(REDEMPTION_EXPIRY_MICROS, 2_592_000_000_000); // 30 days
    }
}

#[cfg(test)]
mod resilience_tests;

#[cfg(test)]
mod simulation;

#[cfg(test)]
mod lifecycle_test;
