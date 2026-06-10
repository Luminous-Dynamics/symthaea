// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Patience Coefficient
//!
//! Calculates the patience coefficient from temporal commitments,
//! used as a multiplier throughout the protocol.

use super::{CommitmentStatus, CovenantType, TemporalCommitment};
use serde::{Deserialize, Serialize};

/// Patience coefficient for a member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatienceCoefficient {
    /// Member's weighted average commitment duration (epochs)
    pub avg_duration_epochs: f64,
    /// Longest active commitment (epochs)
    pub max_duration: u32,
    /// Covenant depth (how purpose-bound, 0.0 to 1.0)
    pub covenant_depth: f64,
    /// Computed coefficient (0.5 to 3.0)
    pub coefficient: f64,
}

impl PatienceCoefficient {
    /// Minimum coefficient (no commitments or very short)
    pub const MIN: f64 = 0.5;
    /// Maximum coefficient (very long + universal covenant)
    pub const MAX: f64 = 3.0;
    /// Neutral coefficient (moderate commitment)
    pub const NEUTRAL: f64 = 1.0;
}

/// Calculate patience coefficient from commitments
pub fn calculate_patience_coefficient(commitments: &[TemporalCommitment]) -> PatienceCoefficient {
    // Filter to active commitments
    let active: Vec<_> = commitments
        .iter()
        .filter(|c| c.status == CommitmentStatus::Active)
        .collect();

    if active.is_empty() {
        return PatienceCoefficient {
            avg_duration_epochs: 0.0,
            max_duration: 0,
            covenant_depth: 0.0,
            coefficient: PatienceCoefficient::MIN,
        };
    }

    // Calculate weighted average duration
    let total_sap: u64 = active.iter().map(|c| c.sap_locked).sum();
    let weighted_duration: f64 = active
        .iter()
        .map(|c| c.duration_epochs as f64 * c.sap_locked as f64)
        .sum::<f64>()
        / total_sap as f64;

    // Find max duration
    let max_duration = active.iter().map(|c| c.duration_epochs).max().unwrap_or(0);

    // Calculate covenant depth (weighted by SAP)
    let covenant_depth: f64 = active
        .iter()
        .map(|c| {
            let depth = covenant_type_to_depth(&c.covenant_type);
            depth * c.sap_locked as f64
        })
        .sum::<f64>()
        / total_sap as f64;

    // Calculate coefficient
    // Duration component: log scale, 1 epoch = 0.5, 168 epochs (14 years) = 2.0
    let duration_factor = (weighted_duration.ln() / 168_f64.ln()).clamp(0.0, 1.0);
    let duration_component = 0.5 + duration_factor * 1.5;

    // Covenant component: adds up to 0.5 based on depth
    let covenant_component = covenant_depth * 0.5;

    let coefficient = (duration_component + covenant_component)
        .clamp(PatienceCoefficient::MIN, PatienceCoefficient::MAX);

    PatienceCoefficient {
        avg_duration_epochs: weighted_duration,
        max_duration,
        covenant_depth,
        coefficient,
    }
}

/// Convert covenant type to depth score (0.0 to 1.0)
fn covenant_type_to_depth(covenant_type: &CovenantType) -> f64 {
    match covenant_type {
        CovenantType::None => 0.0,
        CovenantType::Named => 0.2,
        CovenantType::LocalCommunity => 0.5,
        CovenantType::FutureGenerations => 0.8,
        CovenantType::Ecological => 0.8,
        CovenantType::Universal => 1.0,
    }
}

/// Apply patience coefficient to a base score
pub fn apply_patience(base_score: f64, patience: &PatienceCoefficient) -> f64 {
    base_score * patience.coefficient
}

/// Calculate effective PoC with patience
/// PoC_Score = (Behavioral_50% + PoG_30% + Intention_20%) × Patience_Coefficient
pub fn calculate_poc_with_patience(
    behavioral_score: f64,
    pog_score: f64,
    intention_score: f64,
    patience: &PatienceCoefficient,
) -> f64 {
    let base_poc = behavioral_score * 0.50 + pog_score * 0.30 + intention_score * 0.20;
    base_poc * patience.coefficient
}

#[cfg(test)]
mod tests {
    use super::super::{CommitmentId, DurationTier};
    use super::*;

    fn make_commitment(duration: u32, sap: u64, covenant_type: CovenantType) -> TemporalCommitment {
        TemporalCommitment {
            commitment_id: CommitmentId::generate(),
            member_did: "did:test:alice".to_string(),
            sap_locked: sap,
            lock_start: 0,
            duration_epochs: duration,
            unlock_at: duration as u64 * 30 * 24 * 60 * 60,
            covenant_id: None,
            covenant_type,
            tier: DurationTier::from_epochs(duration),
            status: CommitmentStatus::Active,
        }
    }

    #[test]
    fn test_no_commitments() {
        let patience = calculate_patience_coefficient(&[]);
        assert!((patience.coefficient - PatienceCoefficient::MIN).abs() < 0.01);
    }

    #[test]
    fn test_short_commitment() {
        let commits = vec![make_commitment(3, 1000, CovenantType::None)];
        let patience = calculate_patience_coefficient(&commits);

        // Short duration, no covenant = low coefficient
        assert!(patience.coefficient < PatienceCoefficient::NEUTRAL);
    }

    #[test]
    fn test_long_commitment_with_covenant() {
        let commits = vec![make_commitment(168, 1000, CovenantType::Universal)];
        let patience = calculate_patience_coefficient(&commits);

        // Long duration + universal covenant = high coefficient
        assert!(patience.coefficient > 2.0);
        assert!((patience.covenant_depth - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mixed_commitments() {
        let commits = vec![
            make_commitment(168, 5000, CovenantType::Universal),
            make_commitment(6, 5000, CovenantType::None),
        ];
        let patience = calculate_patience_coefficient(&commits);

        // Mixed: (168*5000 + 6*5000) / 10000 = 87 avg duration
        assert!(patience.avg_duration_epochs > 80.0);
        assert!(patience.avg_duration_epochs < 90.0);

        // Covenant depth: (1.0*5000 + 0.0*5000) / 10000 = 0.5
        assert!((patience.covenant_depth - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_poc_with_patience() {
        let patience = PatienceCoefficient {
            avg_duration_epochs: 100.0,
            max_duration: 100,
            covenant_depth: 0.5,
            coefficient: 1.5,
        };

        let poc = calculate_poc_with_patience(0.8, 0.7, 0.6, &patience);

        // Base: 0.8*0.5 + 0.7*0.3 + 0.6*0.2 = 0.4 + 0.21 + 0.12 = 0.73
        // With patience: 0.73 * 1.5 = 1.095
        assert!((poc - 1.095).abs() < 0.01);
    }
}
