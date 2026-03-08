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
/// SAP exempt floor in micro-units (1,000 SAP = 1_000_000_000 micro-SAP).
pub const DEMURRAGE_EXEMPT_FLOOR: u64 = 1_000_000_000;
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
    if balance <= exempt_floor || seconds_elapsed == 0 {
        return 0;
    }
    let eligible = (balance - exempt_floor) as f64;
    let years = seconds_elapsed as f64 / 31_536_000.0;
    let decay = 1.0 - (-rate * years).exp();
    let deduction = eligible * decay;
    if deduction < 0.0 {
        0
    } else if deduction > eligible {
        eligible as u64
    } else {
        deduction as u64
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
        deduction.floor() as i32
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
}
