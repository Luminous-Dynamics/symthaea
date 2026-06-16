// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LTC-based continuous control for cell culture environments.
//!
//! Uses an HdcLtcUnifiedNeuron to maintain culture parameters at target values,
//! with O(1) closed-form temporal jumps for prediction at arbitrary future times.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::cell_encoder::encode_environment;
use crate::types::CultureEnvironment;

// ═══════════════════════════════════════════════════════════════════════════════
// CULTURE ADJUSTMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Adjustments to apply to the culture environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CultureAdjustment {
    /// Temperature correction to apply (°C; positive = heat, negative = cool).
    pub delta_temp: f64,
    /// CO₂ concentration correction (%; positive = add, negative = reduce).
    pub delta_co2: f64,
    /// pH correction (positive = alkalinise, negative = acidify).
    pub delta_ph: f64,
    /// True when culture media should be exchanged at the next opportunity.
    pub media_change_needed: bool,
}

impl CultureAdjustment {
    /// No adjustment needed.
    pub fn none() -> Self {
        Self {
            delta_temp: 0.0,
            delta_co2: 0.0,
            delta_ph: 0.0,
            media_change_needed: false,
        }
    }

    /// Total magnitude of adjustment.
    pub fn magnitude(&self) -> f64 {
        self.delta_temp.abs() + self.delta_co2.abs() + self.delta_ph.abs()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTROLLER
// ═══════════════════════════════════════════════════════════════════════════════

/// LTC-based controller for cell culture environment.
///
/// The neuron state tracks the culture environment over time. The closed-form
/// solution allows O(1) prediction of future environment states.
pub struct CultureController {
    neuron: HdcLtcUnifiedNeuron,
    target_env: CultureEnvironment,
}

impl CultureController {
    /// Create a new controller targeting the given environment.
    pub fn new(target: CultureEnvironment) -> Self {
        // Use a configuration with longer time constants suitable for biological processes
        let config = UnifiedConfig {
            tau_base: 10.0, // 10 second base — biological control is slow
            backbone_tau: 0.3,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xC0C1_0001),
            target_env: target,
        }
    }

    /// Create with a specific UnifiedConfig.
    pub fn with_config(target: CultureEnvironment, config: UnifiedConfig) -> Self {
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xC0C1_0001),
            target_env: target,
        }
    }

    /// Step the controller forward by dt_seconds and compute adjustments.
    ///
    /// Encodes the current environment into an HDC vector, evolves the LTC
    /// neuron state, and decodes the neuron's output into environment adjustments.
    pub fn step(&mut self, current_env: &CultureEnvironment, dt_seconds: f32) -> CultureAdjustment {
        let env_hv = encode_environment(current_env);
        self.neuron.evolve_closed_form(dt_seconds, &env_hv);
        self.decode_adjustment(current_env)
    }

    /// O(1) prediction of environment state at a future time horizon.
    ///
    /// Clones the neuron to avoid mutating controller state, then performs
    /// a single closed-form temporal jump.
    pub fn predict_at(
        &self,
        current_env: &CultureEnvironment,
        future_seconds: f32,
    ) -> ContinuousHV {
        let env_hv = encode_environment(current_env);
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(future_seconds, &env_hv);
        neuron_copy.state().clone()
    }

    /// Get the current target environment.
    pub fn target(&self) -> &CultureEnvironment {
        &self.target_env
    }

    /// Set a new target environment.
    pub fn set_target(&mut self, target: CultureEnvironment) {
        self.target_env = target;
    }

    /// Decode the neuron state into a culture adjustment based on deviation from target.
    fn decode_adjustment(&self, current: &CultureEnvironment) -> CultureAdjustment {
        let neuron_state = self.neuron.state();
        let state_norm = neuron_state.norm() as f64;

        // Scale factor: use neuron state norm to modulate response magnitude
        let response_scale = (state_norm / 100.0).min(1.0);

        // Error signals (target - current)
        let temp_error = self.target_env.temperature_celsius - current.temperature_celsius;
        let co2_error = self.target_env.co2_percent - current.co2_percent;
        let ph_error = self.target_env.ph - current.ph;

        // Proportional correction modulated by neuron dynamics
        let gain = 0.3 * (1.0 + response_scale);
        let delta_temp = temp_error * gain;
        let delta_co2 = co2_error * gain;
        let delta_ph = ph_error * gain;

        // Media change needed if pH drift is large or culture is old
        let media_change_needed = ph_error.abs() > 0.3 || current.passage_number > 20;

        CultureAdjustment {
            delta_temp,
            delta_co2,
            delta_ph,
            media_change_needed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_reduces_deviation() {
        let target = CultureEnvironment::standard();
        let mut controller = CultureController::new(target);

        // Start with a deviated environment
        let mut current = CultureEnvironment {
            temperature_celsius: 38.5,
            co2_percent: 6.0,
            o2_percent: 20.0,
            ph: 7.6,
            humidity_percent: 95.0,
            media_volume_ml: 10.0,
            passage_number: 0,
        };

        let initial_deviation = current.deviation_from_standard();

        // Apply controller adjustments for several steps
        for _ in 0..10 {
            let adj = controller.step(&current, 1.0);
            current.temperature_celsius += adj.delta_temp;
            current.co2_percent += adj.delta_co2;
            current.ph += adj.delta_ph;
        }

        let final_deviation = current.deviation_from_standard();
        assert!(
            final_deviation < initial_deviation,
            "Controller should reduce deviation: initial={}, final={}",
            initial_deviation,
            final_deviation
        );
    }

    #[test]
    fn test_controller_no_adjustment_at_target() {
        let target = CultureEnvironment::standard();
        let mut controller = CultureController::new(target.clone());
        let adj = controller.step(&target, 1.0);
        // Adjustments should be very small when at target
        assert!(
            adj.magnitude() < 0.5,
            "At target, adjustments should be small, got magnitude={}",
            adj.magnitude()
        );
    }

    #[test]
    fn test_prediction_is_o1() {
        let target = CultureEnvironment::standard();
        let controller = CultureController::new(target);
        let current = CultureEnvironment::standard();

        // Both predictions should produce valid HVs (point: same computational cost)
        let pred_1h = controller.predict_at(&current, 3600.0);
        let pred_9mo = controller.predict_at(&current, 23_328_000.0);

        assert!(pred_1h.norm() > 0.0);
        assert!(pred_9mo.norm() > 0.0);
        assert_eq!(pred_1h.dim(), HDC_DIMENSION);
        assert_eq!(pred_9mo.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_prediction_does_not_mutate_controller() {
        let target = CultureEnvironment::standard();
        let mut controller = CultureController::new(target);
        let current = CultureEnvironment::standard();
        controller.step(&current, 1.0); // Prime with non-zero state

        let state_before = controller.neuron.state().clone();
        let _pred = controller.predict_at(&current, 3600.0);
        let state_after = controller.neuron.state().clone();

        let sim = state_before.similarity(&state_after);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "predict_at should not mutate controller state"
        );
    }

    #[test]
    fn test_media_change_for_large_ph_drift() {
        let target = CultureEnvironment::standard();
        let mut controller = CultureController::new(target);
        let mut current = CultureEnvironment::standard();
        current.ph = 8.0; // large drift
        let adj = controller.step(&current, 1.0);
        assert!(
            adj.media_change_needed,
            "Large pH drift should trigger media change"
        );
    }

    #[test]
    fn test_culture_adjustment_none() {
        let adj = CultureAdjustment::none();
        assert!(adj.magnitude() < 1e-9);
        assert!(!adj.media_change_needed);
    }
}
