// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Yamanaka factor iPSC reprogramming protocol.
//!
//! Models the 28-day reprogramming process from somatic cells to induced
//! pluripotent stem cells (iPSCs) via Oct4, Sox2, Klf4, and c-Myc expression.

use serde::{Deserialize, Serialize};

use crate::types::{CellState, CellType, CultureEnvironment, ReprogrammingStage, YamanakaFactors};

/// iPSC reprogramming protocol state machine.
///
/// Tracks the multi-day reprogramming process, automatically adjusting Yamanaka
/// factor levels per stage and computing success probability based on culture
/// conditions and cell viability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReprogrammingProtocol {
    /// Current stage of the 28-day reprogramming timeline.
    pub stage: ReprogrammingStage,
    /// Number of days elapsed since protocol initiation.
    pub day: u32,
    /// Current Yamanaka factor expression levels (adjusted per stage).
    pub factors: YamanakaFactors,
    /// Cell being reprogrammed; updated each day with pluripotency gains.
    pub cell_state: CellState,
    /// Cumulative stress from suboptimal conditions (reduces success probability).
    stress_accumulator: f64,
}

impl ReprogrammingProtocol {
    /// Create a new reprogramming protocol from a somatic cell.
    pub fn new() -> Self {
        Self {
            stage: ReprogrammingStage::Initial,
            day: 0,
            factors: Self::optimal_factor_levels(ReprogrammingStage::Initial),
            cell_state: CellState::new_somatic(),
            stress_accumulator: 0.0,
        }
    }

    /// Create from a specific donor cell.
    pub fn from_donor(donor: CellState) -> Self {
        Self {
            stage: ReprogrammingStage::Initial,
            day: 0,
            factors: Self::optimal_factor_levels(ReprogrammingStage::Initial),
            cell_state: donor,
            stress_accumulator: 0.0,
        }
    }

    /// Advance the protocol by one day.
    ///
    /// Updates the stage based on the current day, adjusts factor levels,
    /// and modifies cell state according to reprogramming dynamics.
    pub fn advance_day(&mut self, culture: &CultureEnvironment) -> ReprogrammingStage {
        if self.stage == ReprogrammingStage::Failed || self.stage == ReprogrammingStage::Complete {
            return self.stage;
        }

        self.day += 1;

        // Accumulate stress from suboptimal conditions
        let deviation = culture.deviation_from_standard();
        self.stress_accumulator += deviation;

        // Check for failure conditions
        if self.cell_state.viability < 0.3 {
            self.stage = ReprogrammingStage::Failed;
            return self.stage;
        }

        // Stage transitions based on day
        self.stage = match self.day {
            0 => ReprogrammingStage::Initial,
            1..=3 => ReprogrammingStage::EarlyResponse,
            4..=10 => ReprogrammingStage::MidReprogramming,
            11..=20 => ReprogrammingStage::LateReprogramming,
            21..=28 => ReprogrammingStage::Stabilization,
            _ => ReprogrammingStage::Complete,
        };

        // Update factor levels for the current stage
        self.factors = Self::optimal_factor_levels(self.stage);

        // Update cell state based on stage
        self.update_cell_state();

        self.stage
    }

    /// Whether the protocol has completed successfully.
    pub fn is_complete(&self) -> bool {
        self.stage == ReprogrammingStage::Complete
    }

    /// Whether the protocol has failed.
    pub fn is_failed(&self) -> bool {
        self.stage == ReprogrammingStage::Failed
    }

    /// Compute current success probability based on cell state, stress, and stage.
    pub fn success_probability(&self) -> f64 {
        if self.stage == ReprogrammingStage::Failed {
            return 0.0;
        }
        if self.stage == ReprogrammingStage::Complete {
            return 1.0;
        }

        // Base success rate depends on stage progression
        let stage_progress = match self.stage {
            ReprogrammingStage::Initial => 0.01,
            ReprogrammingStage::EarlyResponse => 0.05,
            ReprogrammingStage::MidReprogramming => 0.15,
            ReprogrammingStage::LateReprogramming => 0.40,
            ReprogrammingStage::Stabilization => 0.70,
            ReprogrammingStage::Complete => 1.0,
            ReprogrammingStage::Failed => 0.0,
        };

        // Modulate by viability and stress
        let viability_factor = self.cell_state.viability.clamp(0.0, 1.0);
        let stress_penalty = (1.0 - self.stress_accumulator * 0.1).clamp(0.0, 1.0);
        let factor_quality = self.factors.balance_score();

        (stage_progress * viability_factor * stress_penalty * factor_quality).clamp(0.0, 1.0)
    }

    /// Return optimal Yamanaka factor levels for a given stage.
    pub fn optimal_factor_levels(stage: ReprogrammingStage) -> YamanakaFactors {
        match stage {
            ReprogrammingStage::Initial => YamanakaFactors {
                oct4: 0.9,
                sox2: 0.9,
                klf4: 0.8,
                c_myc: 0.7,
            },
            ReprogrammingStage::EarlyResponse => YamanakaFactors {
                oct4: 0.85,
                sox2: 0.85,
                klf4: 0.75,
                c_myc: 0.6,
            },
            ReprogrammingStage::MidReprogramming => YamanakaFactors {
                oct4: 0.8,
                sox2: 0.8,
                klf4: 0.7,
                c_myc: 0.5,
            },
            ReprogrammingStage::LateReprogramming => YamanakaFactors {
                oct4: 0.7,
                sox2: 0.75,
                klf4: 0.6,
                c_myc: 0.3, // c-Myc reduced late to avoid oncogenesis
            },
            ReprogrammingStage::Stabilization => YamanakaFactors {
                oct4: 0.5,
                sox2: 0.5,
                klf4: 0.4,
                c_myc: 0.1,
            },
            ReprogrammingStage::Complete | ReprogrammingStage::Failed => YamanakaFactors {
                oct4: 0.0,
                sox2: 0.0,
                klf4: 0.0,
                c_myc: 0.0,
            },
        }
    }

    /// Update cell state based on current stage and factors.
    fn update_cell_state(&mut self) {
        let cell = &mut self.cell_state;

        // Gradual pluripotency gain
        match self.stage {
            ReprogrammingStage::EarlyResponse => {
                cell.pluripotency_score += 0.02;
                cell.methylation_score -= 0.01;
            }
            ReprogrammingStage::MidReprogramming => {
                cell.pluripotency_score += 0.05;
                cell.methylation_score -= 0.03;
            }
            ReprogrammingStage::LateReprogramming => {
                cell.pluripotency_score += 0.08;
                cell.methylation_score -= 0.02;
            }
            ReprogrammingStage::Stabilization => {
                cell.pluripotency_score += 0.03;
                cell.methylation_score -= 0.005;
            }
            ReprogrammingStage::Complete => {
                cell.cell_type = CellType::Pluripotent;
                cell.pluripotency_score = cell.pluripotency_score.max(0.95);
            }
            _ => {}
        }

        // Clamp values
        cell.pluripotency_score = cell.pluripotency_score.clamp(0.0, 1.0);
        cell.methylation_score = cell.methylation_score.clamp(0.0, 1.0);

        // Viability decreases slightly from stress
        cell.viability -= 0.005 * self.stress_accumulator.min(1.0);
        cell.viability = cell.viability.clamp(0.0, 1.0);
        cell.age_hours += 24.0;
        cell.division_count += 1;
    }
}

impl Default for ReprogrammingProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_stage() {
        let protocol = ReprogrammingProtocol::new();
        assert_eq!(protocol.stage, ReprogrammingStage::Initial);
        assert_eq!(protocol.day, 0);
    }

    #[test]
    fn test_stage_progression() {
        let mut protocol = ReprogrammingProtocol::new();
        let culture = CultureEnvironment::standard();

        // Day 1-3: Early Response
        protocol.advance_day(&culture);
        assert_eq!(protocol.stage, ReprogrammingStage::EarlyResponse);

        // Day 4-10: Mid Reprogramming
        for _ in 0..3 {
            protocol.advance_day(&culture);
        }
        assert_eq!(protocol.stage, ReprogrammingStage::MidReprogramming);

        // Day 11-20: Late Reprogramming
        for _ in 0..7 {
            protocol.advance_day(&culture);
        }
        assert_eq!(protocol.stage, ReprogrammingStage::LateReprogramming);

        // Day 21-28: Stabilization
        for _ in 0..10 {
            protocol.advance_day(&culture);
        }
        assert_eq!(protocol.stage, ReprogrammingStage::Stabilization);

        // Day 29+: Complete
        for _ in 0..8 {
            protocol.advance_day(&culture);
        }
        assert_eq!(protocol.stage, ReprogrammingStage::Complete);
    }

    #[test]
    fn test_factor_levels_change_per_stage() {
        let initial = ReprogrammingProtocol::optimal_factor_levels(ReprogrammingStage::Initial);
        let late =
            ReprogrammingProtocol::optimal_factor_levels(ReprogrammingStage::LateReprogramming);

        // c-Myc should decrease in late stages (oncogenesis risk)
        assert!(
            late.c_myc < initial.c_myc,
            "c-Myc should decrease in late reprogramming"
        );
    }

    #[test]
    fn test_success_probability_bounded() {
        let protocol = ReprogrammingProtocol::new();
        let prob = protocol.success_probability();
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_success_probability_increases_with_stage() {
        let mut protocol = ReprogrammingProtocol::new();
        let culture = CultureEnvironment::standard();

        let prob_initial = protocol.success_probability();

        for _ in 0..15 {
            protocol.advance_day(&culture);
        }
        let prob_late = protocol.success_probability();

        assert!(
            prob_late > prob_initial,
            "Success probability should increase: initial={}, late={}",
            prob_initial,
            prob_late
        );
    }

    #[test]
    fn test_complete_protocol() {
        let mut protocol = ReprogrammingProtocol::new();
        let culture = CultureEnvironment::standard();

        for _ in 0..35 {
            protocol.advance_day(&culture);
        }

        assert!(protocol.is_complete());
        assert!(!protocol.is_failed());
        assert!((protocol.success_probability() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_failed_protocol_low_viability() {
        let mut protocol = ReprogrammingProtocol::new();
        protocol.cell_state.viability = 0.2; // Below threshold

        let culture = CultureEnvironment::standard();
        protocol.advance_day(&culture);

        assert!(protocol.is_failed());
        assert!((protocol.success_probability()).abs() < 1e-9);
    }

    #[test]
    fn test_pluripotency_increases_over_time() {
        let mut protocol = ReprogrammingProtocol::new();
        let culture = CultureEnvironment::standard();

        let initial_pluri = protocol.cell_state.pluripotency_score;

        for _ in 0..20 {
            protocol.advance_day(&culture);
        }

        assert!(
            protocol.cell_state.pluripotency_score > initial_pluri,
            "Pluripotency should increase during reprogramming"
        );
    }

    #[test]
    fn test_methylation_decreases_over_time() {
        let mut protocol = ReprogrammingProtocol::new();
        let culture = CultureEnvironment::standard();

        let initial_meth = protocol.cell_state.methylation_score;

        for _ in 0..15 {
            protocol.advance_day(&culture);
        }

        assert!(
            protocol.cell_state.methylation_score < initial_meth,
            "Methylation should decrease during reprogramming (epigenetic erasure)"
        );
    }
}
