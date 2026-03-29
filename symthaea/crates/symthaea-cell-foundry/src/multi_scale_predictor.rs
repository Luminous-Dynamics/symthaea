// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-scale biological prediction using O(1) closed-form temporal jumps.
//!
//! A single HdcLtcUnifiedNeuron predicts cell state at ALL timescales
//! from 1 hour to 9 months. The closed-form CfC solution means each
//! prediction costs the same regardless of the time horizon.

use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::BIO_HORIZONS;

/// Multi-scale predictor: single LTC neuron predicts cell state at any horizon.
///
/// The key innovation is that `evolve_closed_form(horizon, input)` costs O(1)
/// regardless of whether `horizon` is 1 hour or 9 months. This is because the
/// closed-form solution computes:
///
/// ```text
/// x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt/tau)
/// ```
///
/// which requires no integration steps.
pub struct MultiScalePredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl MultiScalePredictor {
    /// Create a new predictor with default configuration.
    ///
    /// Uses a large tau_base suitable for biological timescales.
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 3600.0,  // 1 hour base time constant
            backbone_tau: 0.1, // Mild state-dependent scaling
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xB10_BEAD),
        }
    }

    /// Create with a custom configuration.
    pub fn with_config(config: UnifiedConfig) -> Self {
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xB10_BEAD),
        }
    }

    /// Predict cell state at a specific time horizon (in seconds).
    ///
    /// Clones the neuron to avoid mutating the predictor's internal state,
    /// then performs a single O(1) closed-form temporal jump.
    ///
    /// Returns the predicted state as a 16,384D hypervector.
    pub fn predict_at_horizon(
        &self,
        current_state: &ContinuousHV,
        horizon_seconds: f32,
    ) -> ContinuousHV {
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(horizon_seconds, current_state);
        neuron_copy.state().clone()
    }

    /// Predict cell state at all biological horizons (1h to 9 months).
    ///
    /// Returns a vector of (horizon_seconds, predicted_state) tuples.
    /// Each prediction is O(1), so the total cost is O(N_HORIZONS).
    pub fn predict_all_horizons(&self, current_state: &ContinuousHV) -> Vec<(f32, ContinuousHV)> {
        BIO_HORIZONS
            .iter()
            .map(|&h| {
                let predicted = self.predict_at_horizon(current_state, h);
                (h, predicted)
            })
            .collect()
    }

    /// Feed an observation to update the predictor's internal state.
    ///
    /// This evolves the neuron forward by `dt_seconds` with the observed
    /// cell state, allowing the predictor to learn temporal dynamics.
    pub fn observe(&mut self, observed_state: &ContinuousHV, dt_seconds: f32) {
        self.neuron.evolve_closed_form(dt_seconds, observed_state);
    }

    /// Get the predictor's current internal state.
    pub fn state(&self) -> &ContinuousHV {
        self.neuron.state()
    }

    /// Reset the predictor to its initial state.
    pub fn reset(&mut self) {
        let config = self.neuron.state().dim(); // preserve dimension
        let new_config = UnifiedConfig {
            tau_base: 3600.0,
            backbone_tau: 0.1,
            dimension: config,
            ..UnifiedConfig::default()
        };
        self.neuron = HdcLtcUnifiedNeuron::new(new_config, 0xB10_BEAD);
    }
}

impl Default for MultiScalePredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_prediction_dimension() {
        let predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let predicted = predictor.predict_at_horizon(&input, 3600.0);
        assert_eq!(predicted.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_prediction_nonzero() {
        let predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let predicted = predictor.predict_at_horizon(&input, 3600.0);
        assert!(predicted.norm() > 0.0, "Prediction should be non-zero");
    }

    #[test]
    fn test_different_horizons_different_predictions() {
        // Use a short tau so small and large dt produce visibly different decay
        let config = UnifiedConfig {
            tau_base: 1.0, // 1-second tau makes 1s vs 100s very different
            backbone_tau: 0.1,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        let mut predictor = MultiScalePredictor::with_config(config);
        // Prime with one input to move state away from zero
        let prime_input = ContinuousHV::random(HDC_DIMENSION, 99);
        predictor.observe(&prime_input, 100.0);

        // Predict with a DIFFERENT input so x_inf differs from current state
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        // dt=0.01: exp(-0.01/1) ≈ 0.99 (state barely changes)
        // dt=1000: exp(-1000/1) ≈ 0 (fully converged to x_inf)
        let pred_short = predictor.predict_at_horizon(&input, 0.01);
        let pred_long = predictor.predict_at_horizon(&input, 1000.0);

        let sim = pred_short.similarity(&pred_long);
        assert!(
            sim < 0.999,
            "Different horizons should give different predictions, similarity={}",
            sim
        );
    }

    #[test]
    fn test_same_horizon_same_prediction() {
        let predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let pred1 = predictor.predict_at_horizon(&input, 3600.0);
        let pred2 = predictor.predict_at_horizon(&input, 3600.0);

        let sim = pred1.similarity(&pred2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Same horizon should give same prediction"
        );
    }

    #[test]
    fn test_predict_all_horizons_count() {
        let predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let predictions = predictor.predict_all_horizons(&input);
        assert_eq!(predictions.len(), BIO_HORIZONS.len());
    }

    #[test]
    fn test_predict_all_horizons_ordered() {
        let predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let predictions = predictor.predict_all_horizons(&input);
        for i in 1..predictions.len() {
            assert!(
                predictions[i].0 > predictions[i - 1].0,
                "Horizons should be in increasing order"
            );
        }
    }

    #[test]
    fn test_o1_property_same_cost() {
        // Both predictions should take roughly the same time
        let predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let start_1h = Instant::now();
        for _ in 0..100 {
            let _ = predictor.predict_at_horizon(&input, 3600.0);
        }
        let time_1h = start_1h.elapsed();

        let start_9mo = Instant::now();
        for _ in 0..100 {
            let _ = predictor.predict_at_horizon(&input, 23_328_000.0);
        }
        let time_9mo = start_9mo.elapsed();

        // Allow 5x tolerance for timing jitter
        let ratio = time_9mo.as_nanos() as f64 / time_1h.as_nanos().max(1) as f64;
        assert!(
            ratio < 5.0 && ratio > 0.2,
            "O(1) property: 1h took {:?}, 9mo took {:?}, ratio={}",
            time_1h,
            time_9mo,
            ratio
        );
    }

    #[test]
    fn test_observe_mutates_state() {
        let mut predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let state_before = predictor.state().clone();
        predictor.observe(&input, 100.0);
        let state_after = predictor.state().clone();

        let sim = state_before.similarity(&state_after);
        assert!(
            sim < 0.999,
            "observe() should change internal state, similarity={}",
            sim
        );
    }

    #[test]
    fn test_predict_does_not_mutate_state() {
        let mut predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        predictor.observe(&input, 1.0); // Prime with non-zero state

        let state_before = predictor.state().clone();
        let _pred = predictor.predict_at_horizon(&input, 3600.0);
        let state_after = predictor.state().clone();

        let sim = state_before.similarity(&state_after);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "predict_at_horizon should not mutate internal state"
        );
    }

    #[test]
    fn test_reset() {
        let mut predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        predictor.observe(&input, 100.0);
        let state_evolved = predictor.state().clone();

        predictor.reset();
        let state_reset = predictor.state().clone();

        // After reset, state should be zero (initial state)
        let sim = state_evolved.similarity(&state_reset);
        assert!(
            sim < 0.5,
            "Reset should produce different state from evolved, similarity={}",
            sim
        );
    }

    #[test]
    fn test_1h_embedded_in_1d_consistency() {
        // Prediction at 1 hour should be "on the way" to prediction at 1 day
        let predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let pred_1h = predictor.predict_at_horizon(&input, 3600.0);
        let pred_1d = predictor.predict_at_horizon(&input, 86400.0);

        // The 1h prediction should be more similar to the input than the 1d prediction
        let sim_1h_input = pred_1h.similarity(&input);
        let sim_1d_input = pred_1d.similarity(&input);

        // 1h prediction should generally be closer to current state than 1d
        // (due to exponential decay in CfC dynamics)
        assert!(
            sim_1h_input >= sim_1d_input - 0.1,
            "1h prediction should be at least as close to input as 1d: sim_1h={}, sim_1d={}",
            sim_1h_input,
            sim_1d_input
        );
    }

    #[test]
    fn test_bounded_predictions() {
        let predictor = MultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        for &horizon in BIO_HORIZONS {
            let pred = predictor.predict_at_horizon(&input, horizon);
            assert!(
                pred.norm().is_finite(),
                "Prediction at horizon {} should be finite",
                horizon
            );
            assert!(
                pred.norm() < 100.0,
                "Prediction at horizon {} should be bounded, norm={}",
                horizon,
                pred.norm()
            );
        }
    }
}
