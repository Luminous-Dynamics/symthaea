// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Economics Module
//!
//! Three-Currency Economic System:
//! - **MYCEL**: Soulbound reputation substrate (non-transferable, 0.0-1.0)
//! - **SAP**: Circulation medium with continuous demurrage
//! - **TEND**: Mutual credit time exchange (zero-sum)
//!
//! Anti-reflexivity: SAP value never depends on MYCEL, MYCEL never computed
//! from SAP balance, TEND never convertible to SAP at fixed rate.

pub mod commons;
pub mod decay_garden;
pub mod metabolic_oracle;
pub mod poc;
pub mod recognition;

pub use commons::{CommonsContribution, CommonsPool, CommonsResult};
pub use decay_garden::{
    calculate_demurrage, CompostAllocation, CompostDistribution, CompostEvent, DemurrageConfig,
};
pub use metabolic_oracle::{
    MetabolicOracle, MetabolicState, PolicyAdjustment, PolicyBounds, TendLimitTier, VitalityIndex,
};
pub use poc::{
    calculate_mycel_score, jubilee_normalize, GamingDetection, GamingRecommendation,
    MycelCalculation, MycelComponent, MycelScore,
};
pub use recognition::{
    calculate_recognition_score, ContributionType, RecognitionConfig, RecognitionEvent,
};

use serde::{Deserialize, Serialize};

// Re-export canonical types from shared types crate (used by both SDK and Holochain zomes)
pub use mycelix_finance_types::{
    compute_demurrage_deduction, Currency, FeeTier, SuccessionPreference, COMPOST_GLOBAL_PCT,
    COMPOST_LOCAL_PCT, COMPOST_REGIONAL_PCT, DEMURRAGE_EXEMPT_FLOOR, DEMURRAGE_RATE,
    INALIENABLE_RESERVE_RATIO,
};

/// Extension trait for Currency with SDK-specific convenience methods
pub trait CurrencyExt {
    /// Get the technical identifier (same as display_name for our currencies)
    fn technical_id(&self) -> &'static str;
}

impl CurrencyExt for Currency {
    fn technical_id(&self) -> &'static str {
        self.display_name()
    }
}

/// Extension trait for FeeTier with SDK-specific progressive scaling
pub trait FeeTierExt {
    /// Alias for base_fee_rate() (backward compatibility)
    fn base_rate(&self) -> f64;
    /// Calculate effective fee with anti-plutocracy scaling.
    /// Large transactions pay progressively higher rates.
    fn effective_rate(&self, tx_value: u64, median_tx: u64) -> f64;
}

impl FeeTierExt for FeeTier {
    fn base_rate(&self) -> f64 {
        self.base_fee_rate()
    }

    fn effective_rate(&self, tx_value: u64, median_tx: u64) -> f64 {
        let base = self.base_fee_rate();
        if tx_value <= median_tx || median_tx == 0 {
            base
        } else {
            // Progressive scaling for high-value transactions
            let ratio = tx_value as f64 / median_tx as f64;
            base * (1.0 + ratio.log10())
        }
    }
}

/// Member economic state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberEconomics {
    /// Decentralized identifier
    pub did: String,
    /// MYCEL reputation score (computed from 4 components)
    pub mycel_score: MycelScore,
    /// SAP balance (raw — demurrage computed on read)
    pub sap_balance: u64,
    /// SAP locked as collateral
    pub sap_locked: u64,
    /// TEND time credits (signed, mutual credit)
    pub tend_balance: i32,
    /// Timestamp of last demurrage computation
    pub last_demurrage_at: u64,
    /// Whether member is in apprentice mode
    pub is_apprentice: bool,
    /// Mentor DID (if apprentice)
    pub mentor_did: Option<String>,
    /// SAP succession preference on exit/death
    pub succession_preference: SuccessionPreference,
    /// Local DAO membership
    pub local_dao_id: Option<String>,
}

impl MemberEconomics {
    /// Create new member with initial apprentice state
    pub fn new(did: String, mentor_did: Option<String>) -> Self {
        Self {
            did,
            mycel_score: MycelScore::apprentice(),
            sap_balance: 0,
            sap_locked: 0,
            tend_balance: 0,
            last_demurrage_at: 0,
            is_apprentice: true,
            mentor_did,
            succession_preference: SuccessionPreference::default(),
            local_dao_id: None,
        }
    }

    /// Get current fee tier
    pub fn fee_tier(&self) -> FeeTier {
        FeeTier::from_mycel(self.mycel_score.composite)
    }

    /// Calculate effective SAP balance after demurrage
    pub fn effective_sap_balance(&self, config: &DemurrageConfig, current_time: u64) -> u64 {
        let seconds_elapsed = current_time.saturating_sub(self.last_demurrage_at);
        if seconds_elapsed == 0 || self.sap_balance <= config.exempt_floor {
            return self.sap_balance;
        }
        let decayed = calculate_demurrage(
            self.sap_balance,
            config.exempt_floor,
            config.annual_rate,
            seconds_elapsed,
        );
        self.sap_balance - decayed
    }

    /// Apply demurrage and return the amount decayed
    pub fn apply_demurrage(&mut self, config: &DemurrageConfig, current_time: u64) -> u64 {
        let seconds_elapsed = current_time.saturating_sub(self.last_demurrage_at);
        if seconds_elapsed == 0 || self.sap_balance <= config.exempt_floor {
            self.last_demurrage_at = current_time;
            return 0;
        }
        let decayed = calculate_demurrage(
            self.sap_balance,
            config.exempt_floor,
            config.annual_rate,
            seconds_elapsed,
        );
        self.sap_balance -= decayed;
        self.last_demurrage_at = current_time;
        decayed
    }

    /// Graduate from apprentice to newcomer (MYCEL >= 0.3)
    pub fn graduate(&mut self) {
        self.is_apprentice = false;
        self.mentor_did = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fee_tier_from_mycel() {
        assert_eq!(FeeTier::from_mycel(0.1), FeeTier::Newcomer);
        assert_eq!(FeeTier::from_mycel(0.29), FeeTier::Newcomer);
        assert_eq!(FeeTier::from_mycel(0.3), FeeTier::Member);
        assert_eq!(FeeTier::from_mycel(0.5), FeeTier::Member);
        assert_eq!(FeeTier::from_mycel(0.7), FeeTier::Member);
        assert_eq!(FeeTier::from_mycel(0.71), FeeTier::Steward);
        assert_eq!(FeeTier::from_mycel(0.95), FeeTier::Steward);
    }

    #[test]
    fn test_progressive_fee() {
        let tier = FeeTier::Member;
        let median = 1000;

        // Normal transaction pays base rate
        assert!((tier.effective_rate(500, median) - 0.0003).abs() < 0.0001);

        // 10x median pays higher rate
        let rate_10x = tier.effective_rate(10000, median);
        assert!(rate_10x > 0.0003);
    }

    #[test]
    fn test_member_demurrage() {
        let config = DemurrageConfig::default();
        let mut member = MemberEconomics::new("did:example:123".to_string(), None);
        member.sap_balance = 10_000;
        member.last_demurrage_at = 0;

        // One year later
        let one_year = 365 * 24 * 60 * 60;

        // Balance at exempt floor — no decay
        member.sap_balance = 1_000;
        let effective = member.effective_sap_balance(&config, one_year);
        assert_eq!(effective, 1_000);

        // Balance above exempt floor — decays
        member.sap_balance = 10_000;
        let effective = member.effective_sap_balance(&config, one_year);
        // 9000 above exempt decays ~2%/year = ~180 decay, effective ~9820
        assert!(effective < 10_000);
        assert!(effective > 9_700);
    }

    #[test]
    fn test_succession_default() {
        let member = MemberEconomics::new("did:example:alice".to_string(), None);
        assert_eq!(member.succession_preference, SuccessionPreference::Commons);
        assert!(member.is_apprentice);
    }
}
