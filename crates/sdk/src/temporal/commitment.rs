// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Time-Locked Commitments
//!
//! SAP locked for specified durations with graduated governance weight.

use super::{CommitmentId, CovenantId, CovenantType, DurationTier};
use serde::{Deserialize, Serialize};

/// Time-locked SAP commitment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCommitment {
    /// Unique commitment ID
    pub commitment_id: CommitmentId,
    /// Member DID
    pub member_did: String,
    /// SAP amount locked
    pub sap_locked: u64,
    /// Lock start timestamp
    pub lock_start: u64,
    /// Lock duration in epochs (1 epoch = 30 days)
    pub duration_epochs: u32,
    /// Unlock timestamp (computed)
    pub unlock_at: u64,
    /// Associated covenant (optional)
    pub covenant_id: Option<CovenantId>,
    /// Covenant type for multipliers
    pub covenant_type: CovenantType,
    /// Computed tier
    pub tier: DurationTier,
    /// Status
    pub status: CommitmentStatus,
}

/// Commitment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentStatus {
    /// Active and locked
    Active,
    /// Matured (can be unlocked)
    Matured,
    /// Exited early (penalty applied)
    ExitedEarly,
    /// Fully unlocked
    Unlocked,
}

/// Commitment tier summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentTier {
    /// Tier name
    pub tier: DurationTier,
    /// Governance multiplier
    pub governance_multiplier: f64,
    /// TEND multiplier
    pub tend_multiplier: f64,
    /// Exit penalty percentage
    pub exit_penalty_pct: f64,
    /// With covenant bonuses applied
    pub with_covenant: CovenantBonuses,
}

/// Covenant bonuses applied to commitment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovenantBonuses {
    /// Total governance multiplier (tier × covenant)
    pub total_governance_multiplier: f64,
    /// Total TEND multiplier
    pub total_tend_multiplier: f64,
}

impl TemporalCommitment {
    /// Create a new time-locked commitment
    pub fn new(
        member_did: String,
        sap_amount: u64,
        duration_epochs: u32,
        covenant_id: Option<CovenantId>,
        covenant_type: CovenantType,
        timestamp: u64,
    ) -> Self {
        let tier = DurationTier::from_epochs(duration_epochs);
        let unlock_at = timestamp + (duration_epochs as u64 * 30 * 24 * 60 * 60); // epochs to seconds

        Self {
            commitment_id: CommitmentId::generate(),
            member_did,
            sap_locked: sap_amount,
            lock_start: timestamp,
            duration_epochs,
            unlock_at,
            covenant_id,
            covenant_type,
            tier,
            status: CommitmentStatus::Active,
        }
    }

    /// Get effective governance multiplier
    pub fn governance_multiplier(&self) -> f64 {
        self.tier.governance_multiplier() * self.covenant_type.governance_bonus()
    }

    /// Get effective TEND multiplier
    pub fn tend_multiplier(&self) -> f64 {
        self.tier.tend_multiplier() * self.covenant_type.tend_bonus()
    }

    /// Check if commitment has matured
    pub fn is_matured(&self, current_time: u64) -> bool {
        current_time >= self.unlock_at
    }

    /// Get tier summary
    pub fn tier_summary(&self) -> CommitmentTier {
        CommitmentTier {
            tier: self.tier,
            governance_multiplier: self.tier.governance_multiplier(),
            tend_multiplier: self.tier.tend_multiplier(),
            exit_penalty_pct: self.tier.exit_penalty_pct(),
            with_covenant: CovenantBonuses {
                total_governance_multiplier: self.governance_multiplier(),
                total_tend_multiplier: self.tend_multiplier(),
            },
        }
    }
}

/// Result of early exit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyExitResult {
    /// SAP returned to member
    pub sap_returned: u64,
    /// Penalty amount
    pub penalty_amount: u64,
    /// Amount going to HEARTH
    pub to_hearth: u64,
    /// Amount going to Global Commons
    pub to_commons: u64,
    /// CIV adjustment
    pub civ_adjustment: f64,
    /// Cooldown end timestamp
    pub cooldown_until: u64,
}

/// Process early exit from commitment
pub fn process_early_exit(
    commitment: &mut TemporalCommitment,
    current_time: u64,
) -> EarlyExitResult {
    let penalty_pct = commitment.tier.exit_penalty_pct();
    let penalty_amount = (commitment.sap_locked as f64 * penalty_pct) as u64;
    let sap_returned = commitment.sap_locked - penalty_amount;

    // Split penalty: 50% HEARTH, 50% Commons
    let to_hearth = penalty_amount / 2;
    let to_commons = penalty_amount - to_hearth;

    // CIV penalty
    let civ_adjustment = -0.02;

    // 6 epoch cooldown (same tier)
    let cooldown_until = current_time + (6 * 30 * 24 * 60 * 60);

    commitment.status = CommitmentStatus::ExitedEarly;

    EarlyExitResult {
        sap_returned,
        penalty_amount,
        to_hearth,
        to_commons,
        civ_adjustment,
        cooldown_until,
    }
}

/// Calculate effective vote weight from commitments
pub fn calculate_vote_weight(
    base_civ: f64,
    commitments: &[TemporalCommitment],
    total_sap: u64,
) -> f64 {
    if total_sap == 0 {
        return base_civ;
    }

    let weighted_sum: f64 = commitments
        .iter()
        .filter(|c| c.status == CommitmentStatus::Active)
        .map(|c| c.sap_locked as f64 * c.governance_multiplier())
        .sum();

    let avg_multiplier = weighted_sum / total_sap as f64;

    base_civ * avg_multiplier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_commitment() {
        let commit = TemporalCommitment::new(
            "did:test:alice".to_string(),
            10_000,
            48, // 4 years = Tree tier
            None,
            CovenantType::None,
            1_000_000,
        );

        assert_eq!(commit.tier, DurationTier::Tree);
        assert!((commit.governance_multiplier() - 2.5).abs() < 0.01);
        assert_eq!(commit.status, CommitmentStatus::Active);
    }

    #[test]
    fn test_covenant_bonus() {
        let commit = TemporalCommitment::new(
            "did:test:alice".to_string(),
            10_000,
            200, // Forest tier
            Some(CovenantId::generate()),
            CovenantType::Universal,
            1_000_000,
        );

        // Forest (7.0) × Universal (1.75) = 12.25
        assert!((commit.governance_multiplier() - 12.25).abs() < 0.01);
    }

    #[test]
    fn test_early_exit() {
        let mut commit = TemporalCommitment::new(
            "did:test:alice".to_string(),
            10_000,
            48, // Tree tier = 15% penalty
            None,
            CovenantType::None,
            1_000_000,
        );

        let result = process_early_exit(&mut commit, 2_000_000);

        assert_eq!(result.penalty_amount, 1_500); // 15% of 10,000
        assert_eq!(result.sap_returned, 8_500);
        assert_eq!(result.to_hearth, 750);
        assert_eq!(result.to_commons, 750);
        assert!((result.civ_adjustment - (-0.02)).abs() < 0.001);
        assert_eq!(commit.status, CommitmentStatus::ExitedEarly);
    }

    #[test]
    fn test_vote_weight() {
        let commits = vec![
            TemporalCommitment::new(
                "did:test:alice".to_string(),
                5_000,
                200, // Forest = 7.0x
                None,
                CovenantType::None,
                1_000_000,
            ),
            TemporalCommitment::new(
                "did:test:alice".to_string(),
                5_000,
                3, // Sprout = 1.0x
                None,
                CovenantType::None,
                1_000_000,
            ),
        ];

        // Weighted average: (5000*7 + 5000*1) / 10000 = 4.0
        let weight = calculate_vote_weight(0.8, &commits, 10_000);
        // 0.8 × 4.0 = 3.2
        assert!((weight - 3.2).abs() < 0.01);
    }
}
