// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Continuous Demurrage — Wealth Circulation
//!
//! SAP balances above the exempt floor decay continuously at 2% annual.
//! No dormancy threshold. No clock to game. Active participants naturally
//! counter decay through earning.
//!
//! Decayed SAP is redistributed ("composted") to commons pools — never destroyed.
//! Constitutional: demurrage must exist (rate adjustable 1-5%, existence not).

use serde::{Deserialize, Serialize};

/// Demurrage configuration (replaces DecayGarden)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemurrageConfig {
    /// Annual demurrage rate (default 2%, constitutional bounds: 1-5%)
    pub annual_rate: f64,
    /// Balance exempt from demurrage per member (default 1,000 SAP)
    pub exempt_floor: u64,
    /// Compost distribution ratios
    pub distribution: CompostDistribution,
}

impl Default for DemurrageConfig {
    fn default() -> Self {
        Self {
            annual_rate: 0.02,   // 2%
            exempt_floor: 1_000, // 1,000 SAP protected
            distribution: CompostDistribution::default(),
        }
    }
}

/// How composted (decayed) SAP is distributed to commons pools
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CompostDistribution {
    /// Ratio to member's local commons pool
    pub local_commons_ratio: f64,
    /// Ratio to regional commons pool
    pub regional_commons_ratio: f64,
    /// Ratio to global commons fund
    pub global_commons_ratio: f64,
}

impl Default for CompostDistribution {
    fn default() -> Self {
        Self {
            local_commons_ratio: 0.70,    // 70%
            regional_commons_ratio: 0.20, // 20%
            global_commons_ratio: 0.10,   // 10%
        }
    }
}

impl CompostDistribution {
    /// Validate distribution ratios sum to 1.0
    pub fn is_valid(&self) -> bool {
        let sum =
            self.local_commons_ratio + self.regional_commons_ratio + self.global_commons_ratio;
        (sum - 1.0).abs() < 0.001
    }

    /// Calculate distribution amounts
    pub fn distribute(&self, total: u64) -> CompostAllocation {
        CompostAllocation {
            local_commons: (total as f64 * self.local_commons_ratio) as u64,
            regional_commons: (total as f64 * self.regional_commons_ratio) as u64,
            global_commons: (total as f64 * self.global_commons_ratio) as u64,
        }
    }
}

/// Allocation of composted SAP to commons pools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompostAllocation {
    /// Amount to local commons pool
    pub local_commons: u64,
    /// Amount to regional commons pool
    pub regional_commons: u64,
    /// Amount to global commons fund
    pub global_commons: u64,
}

impl CompostAllocation {
    /// Total composted
    pub fn total(&self) -> u64 {
        self.local_commons + self.regional_commons + self.global_commons
    }
}

/// Calculate continuous demurrage on a balance.
///
/// Formula: decay = (balance - exempt_floor) × (1 - e^(-rate × years))
/// Applied on every balance read — no dormancy clock.
///
/// Returns the amount that has decayed (to be redistributed).
pub fn calculate_demurrage(
    balance: u64,
    exempt_floor: u64,
    annual_rate: f64,
    seconds_elapsed: u64,
) -> u64 {
    if balance <= exempt_floor || annual_rate <= 0.0 || seconds_elapsed == 0 {
        return 0;
    }

    let eligible = balance - exempt_floor;
    let years = seconds_elapsed as f64 / (365.25 * 24.0 * 60.0 * 60.0);

    // Continuous exponential decay: effective = eligible × e^(-rate × years)
    // Decayed amount = eligible - effective = eligible × (1 - e^(-rate × years))
    let decay_factor = 1.0 - (-annual_rate * years).exp();
    let decayed = (eligible as f64 * decay_factor) as u64;

    decayed.min(eligible) // Never decay more than eligible
}

/// Compost event for audit/gratitude ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompostEvent {
    /// Member whose SAP was composted
    pub member_did: String,
    /// Total amount composted
    pub amount: u64,
    /// Distribution breakdown
    pub allocation: CompostAllocation,
    /// Timestamp
    pub timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_demurrage_below_exempt() {
        // Balance at or below exempt floor — zero decay
        assert_eq!(calculate_demurrage(1_000, 1_000, 0.02, 365 * 86400), 0);
        assert_eq!(calculate_demurrage(500, 1_000, 0.02, 365 * 86400), 0);
    }

    #[test]
    fn test_demurrage_one_year() {
        // 10,000 SAP, 1,000 exempt, 2% rate, 1 year
        let one_year = (365.25 * 86400.0) as u64;
        let decayed = calculate_demurrage(10_000, 1_000, 0.02, one_year);

        // 9,000 eligible × (1 - e^(-0.02)) ≈ 9,000 × 0.0198 ≈ 178
        assert!(
            decayed > 150 && decayed < 200,
            "Expected ~178 decay, got {}",
            decayed
        );
    }

    #[test]
    fn test_demurrage_five_years() {
        // Long-term: 10,000 SAP over 5 years at 2%
        let five_years = (5.0 * 365.25 * 86400.0) as u64;
        let decayed = calculate_demurrage(10_000, 1_000, 0.02, five_years);

        // 9,000 × (1 - e^(-0.10)) ≈ 9,000 × 0.0952 ≈ 857
        assert!(
            decayed > 800 && decayed < 950,
            "Expected ~857 decay, got {}",
            decayed
        );
    }

    #[test]
    fn test_compost_distribution() {
        let dist = CompostDistribution::default();
        assert!(dist.is_valid());

        let allocation = dist.distribute(10_000);
        assert_eq!(allocation.local_commons, 7_000);
        assert_eq!(allocation.regional_commons, 2_000);
        assert_eq!(allocation.global_commons, 1_000);
        assert_eq!(allocation.total(), 10_000);
    }

    #[test]
    fn test_zero_elapsed() {
        assert_eq!(calculate_demurrage(10_000, 1_000, 0.02, 0), 0);
    }

    #[test]
    fn test_zero_rate() {
        let one_year = (365.25 * 86400.0) as u64;
        assert_eq!(calculate_demurrage(10_000, 1_000, 0.0, one_year), 0);
    }
}
