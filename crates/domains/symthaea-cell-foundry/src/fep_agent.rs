// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Free Energy Principle (FEP) active inference agent for cell culture control.
//!
//! The agent selects actions to minimize free energy (prediction error)
//! by maintaining cell state close to expected/desired trajectories.

use serde::{Deserialize, Serialize};

use crate::cell_encoder::encode_cell_state;
use crate::types::{CellState, CultureEnvironment};

/// Discrete actions the culture FEP agent can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CultureAction {
    /// Correct incubator temperature toward setpoint.
    AdjustTemp,
    /// Correct CO₂ level toward setpoint.
    AdjustCo2,
    /// Correct media pH toward setpoint.
    AdjustPh,
    /// Replenish culture media volume.
    AddMedia,
    /// Continue incubation without intervention.
    ExtendIncubation,
    /// Subculture (passage) cells to maintain healthy density.
    PassageCells,
    /// Initiate in vitro gametogenesis on the current iPSC population.
    InitiateIvg,
    /// Abort the protocol due to unrecoverable conditions.
    AbortProtocol,
}

impl CultureAction {
    /// All possible actions as a fixed-size array.
    pub const ALL: [CultureAction; 8] = [
        CultureAction::AdjustTemp,
        CultureAction::AdjustCo2,
        CultureAction::AdjustPh,
        CultureAction::AddMedia,
        CultureAction::ExtendIncubation,
        CultureAction::PassageCells,
        CultureAction::InitiateIvg,
        CultureAction::AbortProtocol,
    ];

    /// Convert action to index.
    pub fn to_index(self) -> usize {
        match self {
            CultureAction::AdjustTemp => 0,
            CultureAction::AdjustCo2 => 1,
            CultureAction::AdjustPh => 2,
            CultureAction::AddMedia => 3,
            CultureAction::ExtendIncubation => 4,
            CultureAction::PassageCells => 5,
            CultureAction::InitiateIvg => 6,
            CultureAction::AbortProtocol => 7,
        }
    }

    /// Convert index to action.
    pub fn from_index(idx: usize) -> Self {
        CultureAction::ALL[idx.min(7)]
    }
}

/// FEP-based active inference agent for cell culture decision-making.
///
/// Selects actions that minimize the discrepancy between observed cell state
/// and expected (desired) cell state, using HDC-encoded state representations.
pub struct CultureFepAgent {
    /// Complete action repertoire available to the agent.
    pub actions: [CultureAction; 8],
    /// Free energy (prediction error) above which the agent considers the state surprising.
    pub surprise_threshold: f64,
}

impl CultureFepAgent {
    /// Create a new FEP agent with default configuration.
    pub fn new() -> Self {
        Self {
            actions: CultureAction::ALL,
            surprise_threshold: 0.5,
        }
    }

    /// Create with a custom surprise threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            actions: CultureAction::ALL,
            surprise_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Select the best action given current cell state and environment.
    ///
    /// The agent evaluates the cell state and environment to determine which
    /// action would most reduce free energy (prediction error).
    pub fn select_action(
        &self,
        cell_state: &CellState,
        environment: &CultureEnvironment,
    ) -> CultureAction {
        // If cell viability is critical, abort before considering recovery actions.
        if !cell_state.viability.is_finite() || cell_state.viability < 0.2 {
            return CultureAction::AbortProtocol;
        }

        // Environmental deviations drive corrective actions. Compare normalized
        // control error rather than raw values with incompatible units (degrees C,
        // percentage points, and pH units).
        let std = CultureEnvironment::standard();
        let temp_score = (environment.temperature_celsius - std.temperature_celsius).abs() / 0.5;
        let co2_score = (environment.co2_percent - std.co2_percent).abs() / 0.3;
        let ph_score = (environment.ph - std.ph).abs() / 0.2;
        let media_score = if environment.media_volume_ml.is_finite() {
            if environment.media_volume_ml < 5.0 {
                1.0 + (5.0 - environment.media_volume_ml) / 5.0
            } else {
                0.0
            }
        } else {
            f64::INFINITY
        };

        let corrective_scores = [
            (CultureAction::AdjustTemp, temp_score),
            (CultureAction::AdjustCo2, co2_score),
            (CultureAction::AdjustPh, ph_score),
            (CultureAction::AddMedia, media_score),
        ];

        let (best_action, best_score) = corrective_scores
            .into_iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("fixed corrective action set is non-empty");

        if !best_score.is_finite() {
            CultureAction::AbortProtocol
        } else if best_score > 1.0 {
            best_action
        } else if environment.passage_number > 15 {
            CultureAction::PassageCells
        } else {
            CultureAction::ExtendIncubation
        }
    }

    /// Compute free energy (prediction error) between observed and expected cell states.
    ///
    /// Uses HDC cosine distance: free_energy = 1 - similarity(observed, expected).
    /// Lower free energy means the cell state matches expectations.
    pub fn compute_free_energy(&self, cell_state: &CellState, expected: &CellState) -> f64 {
        let hv_observed = encode_cell_state(cell_state);
        let hv_expected = encode_cell_state(expected);
        let similarity = hv_observed.similarity(&hv_expected) as f64;
        // Free energy = 1 - similarity (range [0, 2])
        (1.0 - similarity).max(0.0)
    }

    /// Whether the current state is "surprising" (high free energy).
    pub fn is_surprised(&self, cell_state: &CellState, expected: &CellState) -> bool {
        self.compute_free_energy(cell_state, expected) > self.surprise_threshold
    }
}

impl Default for CultureFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_index_roundtrip() {
        for action in &CultureAction::ALL {
            let idx = action.to_index();
            let recovered = CultureAction::from_index(idx);
            assert_eq!(*action, recovered);
        }
    }

    #[test]
    fn test_select_action_abort_low_viability() {
        let agent = CultureFepAgent::new();
        let mut cell = CellState::new_somatic();
        cell.viability = 0.1;
        let env = CultureEnvironment::standard();
        let action = agent.select_action(&cell, &env);
        assert_eq!(action, CultureAction::AbortProtocol);
    }

    #[test]
    fn test_select_action_adjust_temp() {
        let agent = CultureFepAgent::new();
        let cell = CellState::new_somatic();
        let mut env = CultureEnvironment::standard();
        env.temperature_celsius = 39.0; // 2 degrees off
        let action = agent.select_action(&cell, &env);
        assert_eq!(action, CultureAction::AdjustTemp);
    }

    #[test]
    fn test_select_action_adjust_co2() {
        let agent = CultureFepAgent::new();
        let cell = CellState::new_somatic();
        let mut env = CultureEnvironment::standard();
        env.co2_percent = 8.0; // Way off
        let action = agent.select_action(&cell, &env);
        assert_eq!(action, CultureAction::AdjustCo2);
    }

    #[test]
    fn test_select_action_adjust_ph() {
        let agent = CultureFepAgent::new();
        let cell = CellState::new_somatic();
        let mut env = CultureEnvironment::standard();
        env.ph = 7.8;
        let action = agent.select_action(&cell, &env);
        assert_eq!(action, CultureAction::AdjustPh);
    }

    #[test]
    fn test_select_action_extend_incubation_at_equilibrium() {
        let agent = CultureFepAgent::new();
        let cell = CellState::new_somatic();
        let env = CultureEnvironment::standard();
        let action = agent.select_action(&cell, &env);
        assert_eq!(action, CultureAction::ExtendIncubation);
    }

    #[test]
    fn test_select_action_add_media_even_when_other_conditions_are_standard() {
        let agent = CultureFepAgent::new();
        let cell = CellState::new_somatic();
        let mut env = CultureEnvironment::standard();
        env.media_volume_ml = 1.0;
        let action = agent.select_action(&cell, &env);
        assert_eq!(action, CultureAction::AddMedia);
    }

    #[test]
    fn test_select_action_uses_normalized_control_error() {
        let agent = CultureFepAgent::new();
        let cell = CellState::new_somatic();
        let mut env = CultureEnvironment::standard();
        env.temperature_celsius += 0.6; // score 1.2
        env.co2_percent += 0.6; // score 2.0
        let action = agent.select_action(&cell, &env);
        assert_eq!(action, CultureAction::AdjustCo2);
    }

    #[test]
    fn test_select_action_aborts_on_non_finite_viability() {
        let agent = CultureFepAgent::new();
        let mut cell = CellState::new_somatic();
        cell.viability = f64::NAN;
        let action = agent.select_action(&cell, &CultureEnvironment::standard());
        assert_eq!(action, CultureAction::AbortProtocol);
    }

    #[test]
    fn test_free_energy_zero_for_identical() {
        let agent = CultureFepAgent::new();
        let cell = CellState::new_somatic();
        let fe = agent.compute_free_energy(&cell, &cell);
        assert!(
            fe < 0.01,
            "Free energy should be ~0 for identical states, got {}",
            fe
        );
    }

    #[test]
    fn test_free_energy_positive_for_different() {
        let agent = CultureFepAgent::new();
        let somatic = CellState::new_somatic();
        let ipsc = CellState::new_ipsc();
        let fe = agent.compute_free_energy(&somatic, &ipsc);
        assert!(
            fe > 0.0,
            "Free energy should be positive for different states, got {}",
            fe
        );
    }

    #[test]
    fn test_not_surprised_at_match() {
        let agent = CultureFepAgent::new();
        let cell = CellState::new_somatic();
        assert!(!agent.is_surprised(&cell, &cell));
    }

    #[test]
    fn test_surprised_at_mismatch() {
        let agent = CultureFepAgent::with_threshold(0.1);
        let somatic = CellState::new_somatic();
        let ipsc = CellState::new_ipsc();
        assert!(agent.is_surprised(&somatic, &ipsc));
    }

    #[test]
    fn test_free_energy_bounded() {
        let agent = CultureFepAgent::new();
        let cell1 = CellState::new_somatic();
        let cell2 = CellState::new_ipsc();
        let fe = agent.compute_free_energy(&cell1, &cell2);
        assert!(fe >= 0.0 && fe <= 2.0);
    }
}
