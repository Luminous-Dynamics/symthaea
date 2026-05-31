// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Economics Module
//!
//! Implementation of MIP-E-002: Metabolic Bridge Amendment
//!
//! Core economic primitives for circulation-based value flows:
//! - **Proof of Contribution (PoC)**: Behavioral + Physical contribution scoring
//! - **Metabolic Oracle**: Autopoietic parameter self-adjustment
//! - **HEARTH**: Liquid commons pools
//! - **Decay Garden**: Wealth circulation mechanism

pub mod decay_garden;
pub mod hearth;
pub mod metabolic_oracle;
pub mod poc;

pub use decay_garden::{CompostDistribution, CompostEvent, DecayGarden, DormancyCheck};
pub use hearth::{HarvestRequest, Hearth, HearthPool, TendingAction, WarmingAction};
pub use metabolic_oracle::{
    MetabolicOracle, MetabolicState, PolicyAdjustment, PolicyBounds, VitalityIndex,
};
pub use poc::{
    BehavioralMetrics, ContributionScore, PoCWeights, ProofOfContribution, calculate_poc_score,
};

use serde::{Deserialize, Serialize};

/// Currency identifiers per MIP-E-002 naming conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Currency {
    /// CIV / MYCELIUM - Reputation substrate (non-transferable)
    Civ,
    /// CGC / SPORE - Gift circulation credits
    Cgc,
    /// FLOW / SAP - Utility/transaction token
    Flow,
    /// TEND - Time exchange credits
    Tend,
    /// HEARTH - Liquid commons pool tokens
    Hearth,
    /// ROOT - Stewardship/guardianship tokens
    Root,
}

impl Currency {
    /// Get the metabolic name for this currency
    pub fn metabolic_name(&self) -> &'static str {
        match self {
            Currency::Civ => "MYCELIUM",
            Currency::Cgc => "SPORE",
            Currency::Flow => "SAP",
            Currency::Tend => "TEND",
            Currency::Hearth => "HEARTH",
            Currency::Root => "ROOT",
        }
    }

    /// Get the technical identifier
    pub fn technical_id(&self) -> &'static str {
        match self {
            Currency::Civ => "CIV",
            Currency::Cgc => "CGC",
            Currency::Flow => "FLOW",
            Currency::Tend => "TEND",
            Currency::Hearth => "HEARTH",
            Currency::Root => "ROOT",
        }
    }

    /// Check if currency is transferable
    pub fn is_transferable(&self) -> bool {
        match self {
            Currency::Civ => false, // Reputation is non-transferable
            _ => true,
        }
    }
}

/// Progressive fee tier based on CIV score (MIP-E-002 Article III)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FeeTier {
    /// CIV < 0.3: 0.15% base fee
    Newcomer,
    /// CIV 0.3-0.5: 0.10% base fee
    Participant,
    /// CIV 0.5-0.7: 0.05% base fee
    Contributor,
    /// CIV 0.7-0.9: 0.025% base fee
    Steward,
    /// CIV > 0.9: 0.01% base fee
    Guardian,
}

impl FeeTier {
    /// Determine fee tier from CIV score
    pub fn from_civ(civ: f64) -> Self {
        match civ {
            c if c < 0.3 => FeeTier::Newcomer,
            c if c < 0.5 => FeeTier::Participant,
            c if c < 0.7 => FeeTier::Contributor,
            c if c < 0.9 => FeeTier::Steward,
            _ => FeeTier::Guardian,
        }
    }

    /// Get base fee rate (as fraction, not percentage)
    pub fn base_rate(&self) -> f64 {
        match self {
            FeeTier::Newcomer => 0.0015,    // 0.15%
            FeeTier::Participant => 0.0010, // 0.10%
            FeeTier::Contributor => 0.0005, // 0.05%
            FeeTier::Steward => 0.00025,    // 0.025%
            FeeTier::Guardian => 0.0001,    // 0.01%
        }
    }

    /// Calculate effective fee with anti-plutocracy scaling
    /// Large transactions pay progressively higher rates
    pub fn effective_rate(&self, tx_value: u64, median_tx: u64) -> f64 {
        let base = self.base_rate();
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
    /// CIV reputation score (0.0 - 1.0)
    pub civ_score: f64,
    /// SAP/FLOW balance
    pub sap_balance: u64,
    /// SAP locked as collateral
    pub sap_locked: u64,
    /// CGC/SPORE monthly allocation
    pub cgc_allocation: u64,
    /// TEND time credits
    pub tend_balance: u64,
    /// Last activity timestamp (for dormancy)
    pub last_activity: u64,
    /// Local DAO membership
    pub local_dao_id: Option<String>,
    /// Active agent count (for agentic economy)
    pub active_agent_count: u32,
}

impl MemberEconomics {
    /// Create new member with initial state
    pub fn new(did: String) -> Self {
        Self {
            did,
            civ_score: 0.3, // Starting CIV
            sap_balance: 0,
            sap_locked: 0,
            cgc_allocation: 10, // Default 10 SPORE/month
            tend_balance: 0,
            last_activity: 0,
            local_dao_id: None,
            active_agent_count: 0,
        }
    }

    /// Get current fee tier
    pub fn fee_tier(&self) -> FeeTier {
        FeeTier::from_civ(self.civ_score)
    }

    /// Check if member is dormant (180+ days inactive)
    pub fn is_dormant(&self, current_time: u64) -> bool {
        const DORMANCY_THRESHOLD: u64 = 180 * 24 * 60 * 60; // 180 days in seconds
        current_time.saturating_sub(self.last_activity) > DORMANCY_THRESHOLD
    }

    /// Update last activity timestamp
    pub fn record_activity(&mut self, timestamp: u64) {
        self.last_activity = timestamp;
    }

    /// Calculate maximum loan based on reputation-collateralized lending
    pub fn max_loan(&self, hearth_available: u64) -> u64 {
        (self.civ_score * hearth_available as f64 * 0.3) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fee_tier_from_civ() {
        assert_eq!(FeeTier::from_civ(0.1), FeeTier::Newcomer);
        assert_eq!(FeeTier::from_civ(0.4), FeeTier::Participant);
        assert_eq!(FeeTier::from_civ(0.6), FeeTier::Contributor);
        assert_eq!(FeeTier::from_civ(0.8), FeeTier::Steward);
        assert_eq!(FeeTier::from_civ(0.95), FeeTier::Guardian);
    }

    #[test]
    fn test_progressive_fee() {
        let tier = FeeTier::Contributor;
        let median = 1000;

        // Normal transaction pays base rate
        assert!((tier.effective_rate(500, median) - 0.0005).abs() < 0.0001);

        // 10x median pays ~2x base rate
        let rate_10x = tier.effective_rate(10000, median);
        assert!(rate_10x > 0.0005 && rate_10x < 0.0015);
    }

    #[test]
    fn test_member_dormancy() {
        let mut member = MemberEconomics::new("did:example:123".to_string());
        member.last_activity = 1000;

        // Not dormant after 100 days
        let time_100_days = 1000 + 100 * 24 * 60 * 60;
        assert!(!member.is_dormant(time_100_days));

        // Dormant after 200 days
        let time_200_days = 1000 + 200 * 24 * 60 * 60;
        assert!(member.is_dormant(time_200_days));
    }

    #[test]
    fn test_max_loan() {
        let mut member = MemberEconomics::new("did:example:123".to_string());
        member.civ_score = 0.7;

        let hearth_available = 100_000;
        let max_loan = member.max_loan(hearth_available);

        // 0.7 * 100,000 * 0.3 = 21,000
        assert_eq!(max_loan, 21_000);
    }
}
