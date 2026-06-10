// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Temporal Economics
//!
//! Implementation of MIP-E-005: Temporal Economics Framework
//!
//! Aligns economic incentives with long-term thinking through time-locked
//! commitments, covenant bonds, and intergenerational trusts.

pub mod commitment;
pub mod covenant;
pub mod patience;

pub use commitment::{CommitmentStatus, CommitmentTier, EarlyExitResult, TemporalCommitment};
pub use covenant::{BeneficiarySpec, Covenant};
pub use patience::{calculate_patience_coefficient, PatienceCoefficient};

use serde::{Deserialize, Serialize};

/// Unique commitment ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommitmentId(String);

impl CommitmentId {
    /// Generate new commitment ID
    pub fn generate() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self(format!("commit-{:x}", timestamp))
    }

    /// Create from string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Unique covenant ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CovenantId(String);

impl CovenantId {
    /// Generate new covenant ID
    pub fn generate() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self(format!("covenant-{:x}", timestamp))
    }

    /// Get string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Unique trust ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrustId(String);

impl TrustId {
    /// Generate new trust ID
    pub fn generate() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self(format!("trust-{:x}", timestamp))
    }
}

/// Duration tier for time-locked commitments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurationTier {
    /// 1-6 months
    Sprout,
    /// 7 months - 2 years
    Sapling,
    /// 2-7 years
    Tree,
    /// 7-14 years
    Grove,
    /// 14+ years
    Forest,
}

impl DurationTier {
    /// Get tier from duration in epochs (1 epoch = 30 days)
    pub fn from_epochs(epochs: u32) -> Self {
        match epochs {
            0..=6 => DurationTier::Sprout,
            7..=24 => DurationTier::Sapling,
            25..=84 => DurationTier::Tree,
            85..=168 => DurationTier::Grove,
            _ => DurationTier::Forest,
        }
    }

    /// Get governance multiplier for tier
    pub fn governance_multiplier(&self) -> f64 {
        match self {
            DurationTier::Sprout => 1.0,
            DurationTier::Sapling => 1.5,
            DurationTier::Tree => 2.5,
            DurationTier::Grove => 4.0,
            DurationTier::Forest => 7.0,
        }
    }

    /// Get TEND earning multiplier for tier
    pub fn tend_multiplier(&self) -> f64 {
        match self {
            DurationTier::Sprout => 1.0,
            DurationTier::Sapling => 1.25,
            DurationTier::Tree => 1.75,
            DurationTier::Grove => 2.5,
            DurationTier::Forest => 4.0,
        }
    }

    /// Get early exit penalty percentage
    pub fn exit_penalty_pct(&self) -> f64 {
        match self {
            DurationTier::Sprout => 0.05,
            DurationTier::Sapling => 0.10,
            DurationTier::Tree => 0.15,
            DurationTier::Grove => 0.20,
            DurationTier::Forest => 0.25,
        }
    }
}

/// Covenant type for additional multipliers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovenantType {
    /// No covenant attached
    None,
    /// Named beneficiaries
    Named,
    /// Local DAO or bioregion
    LocalCommunity,
    /// Future generations
    FutureGenerations,
    /// Ecological entity
    Ecological,
    /// Universal (all Mycelix members)
    Universal,
}

impl CovenantType {
    /// Get additional governance multiplier
    pub fn governance_bonus(&self) -> f64 {
        match self {
            CovenantType::None => 1.0,
            CovenantType::Named => 1.1,
            CovenantType::LocalCommunity => 1.25,
            CovenantType::FutureGenerations => 1.5,
            CovenantType::Ecological => 1.5,
            CovenantType::Universal => 1.75,
        }
    }

    /// Get additional TEND multiplier
    pub fn tend_bonus(&self) -> f64 {
        match self {
            CovenantType::None => 1.0,
            CovenantType::Named => 1.1,
            CovenantType::LocalCommunity => 1.2,
            CovenantType::FutureGenerations => 1.5,
            CovenantType::Ecological => 1.5,
            CovenantType::Universal => 1.75,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_from_epochs() {
        assert_eq!(DurationTier::from_epochs(3), DurationTier::Sprout);
        assert_eq!(DurationTier::from_epochs(12), DurationTier::Sapling);
        assert_eq!(DurationTier::from_epochs(48), DurationTier::Tree);
        assert_eq!(DurationTier::from_epochs(120), DurationTier::Grove);
        assert_eq!(DurationTier::from_epochs(200), DurationTier::Forest);
    }

    #[test]
    fn test_governance_multiplier() {
        assert!((DurationTier::Sprout.governance_multiplier() - 1.0).abs() < 0.01);
        assert!((DurationTier::Forest.governance_multiplier() - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_covenant_bonus() {
        assert!((CovenantType::Universal.governance_bonus() - 1.75).abs() < 0.01);
        assert!((CovenantType::None.governance_bonus() - 1.0).abs() < 0.01);
    }
}
