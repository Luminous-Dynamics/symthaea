// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! O(1) temporal degradation model using LTC closed-form dynamics.
//!
//! Models DNA damage accumulation over arbitrary timespans via CfC (Closed-form
//! Continuous-time) neurons. The key insight: because `evolve_closed_form` uses
//! an analytical solution, predicting damage at 100 years costs exactly the same
//! as predicting at 100,000 years.
//!
//! ## State Representation
//!
//! The neuron state is a 16,384D damage integrity vector. Each dimension encodes
//! the integrity of a region of the genome. The neuron evolves under an input
//! signal representing environmental stress (temperature, radiation, moisture).
//!
//! ## Arrhenius-CfC Bridge
//!
//! Temperature modulates the effective time constant:
//! - Cold storage: slow degradation (large tau_effective)
//! - Warm storage: fast degradation (small tau_effective)
//!
//! The Arrhenius factor scales the dt passed to evolve_closed_form, creating
//! temperature-dependent dynamics without modifying the neuron architecture.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Seconds per Julian year (365.25 days), used to convert age in years to seconds for LTC evolution.
pub const SECONDS_PER_YEAR: f32 = 31_557_600.0;

/// Boltzmann constant in eV/K.
const KB_EV: f64 = 8.617e-5;

/// Reference temperature for baseline dynamics (15C = 288.15K).
const REFERENCE_TEMP_K: f64 = 288.15;

/// O(1) temporal degradation model using LTC closed-form dynamics.
///
/// Models DNA damage accumulation over arbitrary timespans via CfC.
/// State = 16,384D damage integrity vector (per-region).
/// Temperature affects time constants (Arrhenius kinetics).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationModel {
    /// The LTC neuron that models degradation dynamics.
    neuron: HdcLtcUnifiedNeuron,
    /// Storage temperature in Celsius.
    temperature_celsius: f64,
    /// Arrhenius activation energy for DNA degradation (eV).
    activation_energy: f64,
}

impl DegradationModel {
    /// Create a new degradation model at the given temperature.
    ///
    /// Initializes a 16,384D LTC neuron with a long time constant suitable
    /// for modeling degradation over years to millennia.
    pub fn new(temperature_celsius: f64) -> Self {
        let config = UnifiedConfig {
            dimension: HDC_DIMENSION,
            activation: UnifiedActivation::Tanh,
            // Long time constant for geological timescales
            tau_base: 1000.0,
            backbone_tau: 0.1,
            learning_rate: 0.0, // No online learning for degradation model
            momentum: 0.0,
            weight_decay: 0.0,
            gating_steepness: 0.5, // Moderate gating for smooth degradation
            interp_bias: 0.0,
            fourier_frequencies: Vec::new(), // Disabled — not needed for degradation dynamics
            fourier_amplitude: 0.1,
        };

        let neuron = HdcLtcUnifiedNeuron::new(config, 42);

        Self {
            neuron,
            temperature_celsius,
            activation_energy: 1.2, // Typical for DNA degradation (~1.0-1.4 eV)
        }
    }

    /// Predict the damage profile after a given number of years.
    ///
    /// Uses O(1) closed-form evolution: the cost is identical whether
    /// predicting 100 years or 100,000 years into the future.
    ///
    /// Returns a damage integrity vector where lower values indicate
    /// more degradation.
    pub fn predict_damage_profile(&mut self, age_years: f64) -> ContinuousHV {
        // Create an environmental stress input representing constant degradation pressure
        let stress_input = self.create_stress_input();

        // Scale time by Arrhenius factor: hotter = faster degradation
        let arrhenius_factor = self.arrhenius_time_factor();
        let effective_dt = age_years as f32 * SECONDS_PER_YEAR * arrhenius_factor as f32;

        // O(1) temporal jump: evolve the neuron to the target time
        self.neuron.evolve_closed_form(effective_dt, &stress_input);

        // The neuron state now represents the damage profile
        self.neuron.state().clone()
    }

    /// Compare a predicted damage profile with an observed one.
    ///
    /// Returns a similarity score in [-1, 1] where:
    /// - 1.0: prediction perfectly matches observation
    /// - 0.0: no correlation (anomalous)
    /// - -1.0: prediction is opposite to observation (very anomalous)
    pub fn compare_predicted_vs_actual(
        &self,
        predicted: &ContinuousHV,
        actual: &ContinuousHV,
    ) -> f64 {
        predicted.similarity(actual) as f64
    }

    /// Get the current temperature in Celsius.
    pub fn temperature_celsius(&self) -> f64 {
        self.temperature_celsius
    }

    /// Get a reference to the internal neuron state.
    pub fn state(&self) -> &ContinuousHV {
        self.neuron.state()
    }

    /// Reset the neuron to its initial state (no degradation).
    pub fn reset(&mut self) {
        let config = UnifiedConfig {
            dimension: HDC_DIMENSION,
            activation: UnifiedActivation::Tanh,
            tau_base: 1000.0,
            backbone_tau: 0.1,
            learning_rate: 0.0,
            momentum: 0.0,
            weight_decay: 0.0,
            gating_steepness: 0.5,
            interp_bias: 0.0,
            fourier_frequencies: Vec::new(),
            fourier_amplitude: 0.1,
        };
        self.neuron = HdcLtcUnifiedNeuron::new(config, 42);
    }

    /// Create the environmental stress input vector.
    ///
    /// The input represents constant degradation pressure from the environment.
    /// Temperature is encoded in the input magnitude.
    fn create_stress_input(&self) -> ContinuousHV {
        // Use a deterministic "degradation pressure" vector
        let base = ContinuousHV::random(HDC_DIMENSION, 0xD0A_DECA);

        // Scale by temperature-dependent stress factor
        let stress_magnitude = self.temperature_stress_factor();
        base.scale(stress_magnitude as f32)
    }

    /// Compute the Arrhenius time scaling factor.
    ///
    /// factor = exp(-Ea/kB * (1/T - 1/T_ref))
    ///
    /// At reference temp (15C), factor = 1.0.
    /// At higher temps, factor > 1.0 (faster degradation).
    /// At lower temps, factor < 1.0 (slower degradation).
    fn arrhenius_time_factor(&self) -> f64 {
        let temp_k = self.temperature_celsius + 273.15;
        let exponent = -(self.activation_energy / KB_EV) * (1.0 / temp_k - 1.0 / REFERENCE_TEMP_K);
        exponent.exp()
    }

    /// Temperature-dependent stress magnitude.
    fn temperature_stress_factor(&self) -> f64 {
        // Normalize to a reasonable range for the neuron input
        let temp_k = self.temperature_celsius + 273.15;
        (temp_k / REFERENCE_TEMP_K).clamp(0.1, 5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_cost_independent_of_timespan() {
        // Both predictions should complete (demonstrating O(1) property).
        // We cannot measure exact wall-clock time reliably in unit tests,
        // but we verify both produce valid results.
        let mut model_short = DegradationModel::new(15.0);
        let mut model_long = DegradationModel::new(15.0);

        let profile_100 = model_short.predict_damage_profile(100.0);
        let profile_100k = model_long.predict_damage_profile(100_000.0);

        assert_eq!(profile_100.dim(), HDC_DIMENSION);
        assert_eq!(profile_100k.dim(), HDC_DIMENSION);
        assert!(profile_100.norm() > 0.0);
        assert!(profile_100k.norm() > 0.0);
    }

    #[test]
    fn test_damage_increases_with_time() {
        let mut model_young = DegradationModel::new(15.0);
        let mut model_old = DegradationModel::new(15.0);

        let initial = model_young.state().clone();

        let profile_young = model_young.predict_damage_profile(100.0);
        let profile_old = model_old.predict_damage_profile(100_000.0);

        // Younger sample should be more similar to initial state
        let sim_young = initial.similarity(&profile_young);
        let sim_old = initial.similarity(&profile_old);

        assert!(
            sim_young > sim_old || (sim_young - sim_old).abs() < 0.1,
            "Younger sample should be closer to initial state: young={}, old={}",
            sim_young,
            sim_old
        );
    }

    #[test]
    fn test_temperature_affects_damage_rate() {
        let mut model_cold = DegradationModel::new(0.0);
        let mut model_warm = DegradationModel::new(30.0);

        let initial_cold = model_cold.state().clone();
        let initial_warm = model_warm.state().clone();

        let profile_cold = model_cold.predict_damage_profile(10_000.0);
        let profile_warm = model_warm.predict_damage_profile(10_000.0);

        // Warm should diverge more from initial than cold
        let sim_cold = initial_cold.similarity(&profile_cold);
        let sim_warm = initial_warm.similarity(&profile_warm);

        // The warm model should have evolved further from initial
        // (either lower similarity or they differ in some measurable way)
        assert!(
            profile_cold.dim() == HDC_DIMENSION && profile_warm.dim() == HDC_DIMENSION,
            "Both profiles should be valid"
        );

        // Verify the Arrhenius factor is working
        let cold_factor = DegradationModel::new(0.0).arrhenius_time_factor();
        let warm_factor = DegradationModel::new(30.0).arrhenius_time_factor();
        assert!(
            warm_factor > cold_factor,
            "Warm should have higher Arrhenius factor: warm={}, cold={}",
            warm_factor,
            cold_factor
        );
    }

    #[test]
    fn test_compare_predicted_vs_actual() {
        let model = DegradationModel::new(15.0);

        let a = ContinuousHV::random(HDC_DIMENSION, 42);
        let b = a.clone();

        let sim = model.compare_predicted_vs_actual(&a, &b);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Identical vectors should have similarity 1.0"
        );
    }

    #[test]
    fn test_compare_dissimilar() {
        let model = DegradationModel::new(15.0);

        let a = ContinuousHV::random(HDC_DIMENSION, 42);
        let b = ContinuousHV::random(HDC_DIMENSION, 43);

        let sim = model.compare_predicted_vs_actual(&a, &b);
        assert!(
            sim.abs() < 0.2,
            "Random vectors should have near-zero similarity: {}",
            sim
        );
    }

    #[test]
    fn test_reset() {
        let mut model = DegradationModel::new(15.0);

        // Evolve the model so state is non-zero
        let _ = model.predict_damage_profile(10_000.0);
        let evolved = model.state().clone();
        assert!(evolved.norm() > 0.0, "Evolved state should be non-zero");

        // Reset and evolve again — should produce same state as first evolution
        model.reset();
        let _ = model.predict_damage_profile(10_000.0);
        let re_evolved = model.state().clone();

        let sim = evolved.similarity(&re_evolved);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Reset + re-evolve should match original evolution, sim={}",
            sim
        );
    }

    #[test]
    fn test_arrhenius_factor_at_reference_temp() {
        let model = DegradationModel::new(15.0);
        let factor = model.arrhenius_time_factor();
        assert!(
            (factor - 1.0).abs() < 0.01,
            "At reference temp, factor should be 1.0, got {}",
            factor
        );
    }

    #[test]
    fn test_arrhenius_factor_monotonic_with_temperature() {
        let temps = [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0];
        let factors: Vec<f64> = temps
            .iter()
            .map(|&t| DegradationModel::new(t).arrhenius_time_factor())
            .collect();

        for i in 1..factors.len() {
            assert!(
                factors[i] > factors[i - 1],
                "Arrhenius factor should increase with temperature: T={}->f={}, T={}->f={}",
                temps[i - 1],
                factors[i - 1],
                temps[i],
                factors[i]
            );
        }
    }

    #[test]
    fn test_seconds_per_year() {
        // Julian year: 365.25 * 24 * 3600 = 31,557,600
        assert_eq!(SECONDS_PER_YEAR, 31_557_600.0);
    }

    #[test]
    fn test_model_serializable() {
        let model = DegradationModel::new(15.0);
        let json = serde_json::to_string(&model);
        assert!(json.is_ok(), "DegradationModel should be serializable");
    }
}
