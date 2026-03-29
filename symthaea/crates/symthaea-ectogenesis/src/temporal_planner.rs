// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! O(1) temporal planning via CfC closed-form evolution.
//!
//! Uses `HdcLtcUnifiedNeuron::evolve_closed_form` to predict fetal state
//! at any future gestational week in constant time, regardless of the
//! temporal gap. This enables instant "what-if" analysis for interventions.

use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::SECONDS_PER_WEEK;

/// Temporal planner using CfC closed-form O(1) jumps.
///
/// Given the current fetal state HV, predicts the state at any future week
/// by evolving a unified HDC-LTC neuron through the time gap in one step.
pub struct TemporalPlanner {
    /// The CfC neuron used for temporal evolution.
    neuron: HdcLtcUnifiedNeuron,
}

impl TemporalPlanner {
    /// Create a new temporal planner with default neuron configuration.
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 10.0, // Slower dynamics for gestational timescale
            backbone_tau: 0.3,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xECE0_FE1A),
        }
    }

    /// Predict fetal state HV at a target week given the current state.
    ///
    /// Uses O(1) closed-form evolution: `x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt/tau)`.
    /// The cost is identical whether predicting 1 week ahead or 30 weeks ahead.
    pub fn predict_at_week(
        &mut self,
        current_state: &ContinuousHV,
        current_week: u32,
        target_week: u32,
    ) -> ContinuousHV {
        if target_week <= current_week {
            return current_state.clone();
        }

        let dt = (target_week - current_week) as f32 * SECONDS_PER_WEEK;

        // Set neuron state to current fetal state
        self.neuron.set_state(current_state.clone());

        // Evolve using O(1) closed-form solution
        self.neuron.evolve_closed_form(dt, current_state);

        // Return the predicted state
        self.neuron.state().clone()
    }

    /// Perform what-if analysis: predict outcomes at key future weeks
    /// after applying an intervention (represented as a perturbation HV).
    ///
    /// Returns predictions at weeks 24, 30, 34, 37, and 40 (or remaining
    /// weeks if current_week is past those milestones).
    pub fn what_if_analysis(
        &mut self,
        current_state: &ContinuousHV,
        current_week: u32,
        intervention: &ContinuousHV,
    ) -> Vec<(u32, ContinuousHV)> {
        // Apply intervention by bundling with current state
        let intervened = ContinuousHV::bundle(&[current_state, intervention]);

        let key_weeks = [24, 30, 34, 37, 40];
        let mut results = Vec::new();

        for &target_week in &key_weeks {
            if target_week > current_week {
                let predicted = self.predict_at_week(&intervened, current_week, target_week);
                results.push((target_week, predicted));
            }
        }

        results
    }

    /// Compare two intervention scenarios: returns per-week similarity
    /// between the predicted trajectories.
    pub fn compare_interventions(
        &mut self,
        current_state: &ContinuousHV,
        current_week: u32,
        intervention_a: &ContinuousHV,
        intervention_b: &ContinuousHV,
    ) -> Vec<(u32, f32)> {
        let results_a = self.what_if_analysis(current_state, current_week, intervention_a);
        let results_b = self.what_if_analysis(current_state, current_week, intervention_b);

        results_a
            .iter()
            .zip(results_b.iter())
            .map(|((week, hv_a), (_, hv_b))| (*week, hv_a.similarity(hv_b)))
            .collect()
    }

    /// Reset the neuron state (for reuse across multiple predictions).
    pub fn reset(&mut self) {
        self.neuron.reset();
    }
}

impl Default for TemporalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gestational_encoder::encode_normative_trajectory;
    use crate::types::GestationalWeek;

    #[test]
    fn test_predict_same_week_returns_clone() {
        let mut planner = TemporalPlanner::new();
        let state = encode_normative_trajectory(GestationalWeek::new(20));
        let predicted = planner.predict_at_week(&state, 20, 20);
        let sim = state.similarity(&predicted);
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "Same-week prediction should return clone, sim={sim}"
        );
    }

    #[test]
    fn test_predict_future_week_differs() {
        let mut planner = TemporalPlanner::new();
        let state = encode_normative_trajectory(GestationalWeek::new(10));
        let predicted = planner.predict_at_week(&state, 10, 30);
        let sim = state.similarity(&predicted);
        assert!(
            sim < 0.99,
            "Future prediction should differ from current, sim={sim}"
        );
    }

    #[test]
    fn test_predict_dimension_preserved() {
        let mut planner = TemporalPlanner::new();
        let state = encode_normative_trajectory(GestationalWeek::new(15));
        let predicted = planner.predict_at_week(&state, 15, 35);
        assert_eq!(predicted.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_what_if_analysis_produces_results() {
        let mut planner = TemporalPlanner::new();
        let state = encode_normative_trajectory(GestationalWeek::new(10));
        let intervention = ContinuousHV::random(HDC_DIMENSION, 0xBEEF_CAFE);
        let results = planner.what_if_analysis(&state, 10, &intervention);
        assert!(!results.is_empty(), "What-if should produce results");
        // All target weeks should be > 10
        for (week, _) in &results {
            assert!(*week > 10);
        }
    }

    #[test]
    fn test_what_if_different_interventions_differ() {
        let mut planner = TemporalPlanner::new();
        let state = encode_normative_trajectory(GestationalWeek::new(10));
        let intervention_a = ContinuousHV::random(HDC_DIMENSION, 0xAAAA_BBBB);
        let intervention_b = ContinuousHV::random(HDC_DIMENSION, 0xCCCC_DDDD);

        let results_a = planner.what_if_analysis(&state, 10, &intervention_a);
        planner.reset();
        let results_b = planner.what_if_analysis(&state, 10, &intervention_b);

        // At least some predictions should differ
        let mut any_different = false;
        for ((_, hv_a), (_, hv_b)) in results_a.iter().zip(results_b.iter()) {
            let sim = hv_a.similarity(hv_b);
            if sim < 0.99 {
                any_different = true;
            }
        }
        assert!(
            any_different,
            "Different interventions should produce different outcomes"
        );
    }

    #[test]
    fn test_compare_interventions() {
        let mut planner = TemporalPlanner::new();
        let state = encode_normative_trajectory(GestationalWeek::new(10));
        let int_a = ContinuousHV::random(HDC_DIMENSION, 0xFACE_1234);
        let int_b = ContinuousHV::random(HDC_DIMENSION, 0xDEAD_5678);

        let comparisons = planner.compare_interventions(&state, 10, &int_a, &int_b);
        assert!(!comparisons.is_empty());

        for (week, sim) in &comparisons {
            assert!(*week > 10);
            assert!(*sim >= -1.0 && *sim <= 1.0);
        }
    }

    #[test]
    fn test_o1_consistency() {
        // Predicting week 30 from week 10 should give consistent results
        let mut planner = TemporalPlanner::new();
        let state = encode_normative_trajectory(GestationalWeek::new(10));
        let predicted1 = planner.predict_at_week(&state, 10, 30);
        planner.reset();
        let predicted2 = planner.predict_at_week(&state, 10, 30);
        let sim = predicted1.similarity(&predicted2);
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "Same prediction should be deterministic, sim={sim}"
        );
    }
}
