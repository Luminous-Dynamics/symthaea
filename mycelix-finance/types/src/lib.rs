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
        if score >= 80.0 { MetabolicState::Thriving }
        else if score >= 60.0 { MetabolicState::Healthy }
        else if score >= 40.0 { MetabolicState::Stressed }
        else if score >= 20.0 { MetabolicState::Critical }
        else { MetabolicState::Failing }
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuccessionPreference {
    /// Default: remaining SAP goes to member's local commons pool
    Commons,
    /// SAP transferred to a designated DID
    Designee(String),
    /// SAP redeemed for collateral through the bridge
    Redemption,
}

impl Default for SuccessionPreference {
    fn default() -> Self {
        SuccessionPreference::Commons
    }
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

/// Compute demurrage deduction on SAP balances.
///
/// Implements: eligible * (1 - e^(-rate * years))
/// where eligible = max(balance - exempt_floor, 0).
///
/// Pure function — no HDK dependencies.
pub fn compute_demurrage_deduction(balance: u64, exempt_floor: u64, rate: f64, seconds_elapsed: u64) -> u64 {
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
        assert_eq!(MetabolicState::from_vitality(90.0), MetabolicState::Thriving);
        assert_eq!(MetabolicState::from_vitality(60.0), MetabolicState::Healthy);
        assert_eq!(MetabolicState::from_vitality(40.0), MetabolicState::Stressed);
        assert_eq!(MetabolicState::from_vitality(20.0), MetabolicState::Critical);
        assert_eq!(MetabolicState::from_vitality(10.0), MetabolicState::Failing);
    }

    #[test]
    fn test_demurrage_below_exempt() {
        // Balance at or below exempt floor → no deduction
        assert_eq!(compute_demurrage_deduction(1_000_000_000, 1_000_000_000, 0.02, 31_536_000), 0);
        assert_eq!(compute_demurrage_deduction(500_000_000, 1_000_000_000, 0.02, 31_536_000), 0);
    }

    #[test]
    fn test_demurrage_one_year() {
        // 10,000 SAP (10B micro) with 1,000 SAP exempt, 2% rate, 1 year
        let deduction = compute_demurrage_deduction(10_000_000_000, 1_000_000_000, 0.02, 31_536_000);
        // Expected: 9B * (1 - e^(-0.02)) ≈ 9B * 0.0198 ≈ 178_200_000
        assert!(deduction > 170_000_000 && deduction < 190_000_000,
            "Expected ~178M, got {}", deduction);
    }

    #[test]
    fn test_demurrage_zero_elapsed() {
        assert_eq!(compute_demurrage_deduction(10_000_000_000, 1_000_000_000, 0.02, 0), 0);
    }

    #[test]
    fn test_succession_serde() {
        let json = serde_json::to_string(&SuccessionPreference::Designee("did:mycelix:abc".into())).unwrap();
        let parsed: SuccessionPreference = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, SuccessionPreference::Designee("did:mycelix:abc".into()));
    }
}
