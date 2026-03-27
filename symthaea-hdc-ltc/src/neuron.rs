// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # HDC-LTC Unified Neuron
//!
//! A single neuron whose state is a hypervector that evolves through
//! Liquid Time-Constant dynamics with a closed-form (O(1)) solution.
//!
//! ## Core Equation
//!
//! ```text
//! dx/dt = (-x + f(W . x + U . u)) / tau(||x||)
//! ```
//!
//! Where `.` is HDC binding (element-wise multiply), `+` is bundling (average),
//! and `tau` is state-dependent. The closed-form solution enables arbitrary
//! temporal jumps at constant cost:
//!
//! ```text
//! x(t + dt) = sigma * x_inf + (1 - sigma) * x(t)
//! ```

use crate::config::{fast_tanh, Activation, NeuronConfig};
use crate::continuous_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Unified HDC-LTC Neuron with closed-form temporal evolution.
///
/// The neuron state is a hypervector (typically 16,384 dimensions) that evolves
/// continuously in time. Weight operations use HDC binding (element-wise multiply)
/// instead of matrix multiplication, yielding O(D) per step rather than O(D^2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcUnifiedNeuron {
    /// Current state (hypervector).
    state: ContinuousHV,
    /// Weight hypervector for state transformation (W).
    weight_hv: ContinuousHV,
    /// Input mask hypervector (U).
    input_mask: ContinuousHV,
    /// Time constant modulator HV (for input-dependent tau adjustment).
    tau_modulator: ContinuousHV,
    /// Gating function weight HV (for closed-form interpolation factor sigma).
    gate_weight: ContinuousHV,
    /// Gating bias HV.
    gate_bias: ContinuousHV,
    /// Configuration.
    config: NeuronConfig,
    /// Momentum accumulator for weight updates.
    weight_momentum: ContinuousHV,
    /// Momentum accumulator for input mask updates.
    input_momentum: ContinuousHV,
    /// Total time evolved (seconds).
    total_time: f64,
    /// Number of evolution steps taken.
    update_count: u64,
}

impl HdcLtcUnifiedNeuron {
    /// Create a new neuron with the given configuration and deterministic seed.
    ///
    /// Initializes the 5 internal HVs via modified Gram-Schmidt orthogonalization
    /// so they have minimal interference at startup.
    pub fn new(config: NeuronConfig, seed: u64) -> Self {
        let dim = config.dim;
        let ortho = ContinuousHV::orthogonal_set(dim, 5, seed);

        Self {
            state: ContinuousHV::new(dim),
            weight_hv: ortho[0].clone(),
            input_mask: ortho[1].clone(),
            tau_modulator: ortho[2].clone(),
            gate_weight: ortho[3].clone(),
            gate_bias: ortho[4].scale(0.1),
            weight_momentum: ContinuousHV::new(dim),
            input_momentum: ContinuousHV::new(dim),
            total_time: 0.0,
            update_count: 0,
            config,
        }
    }

    /// Get the current state hypervector.
    pub fn state(&self) -> &ContinuousHV {
        &self.state
    }

    /// Get a mutable reference to the state.
    pub fn state_mut(&mut self) -> &mut ContinuousHV {
        &mut self.state
    }

    /// Set the state directly.
    pub fn set_state(&mut self, state: ContinuousHV) {
        self.state = state;
    }

    /// Reset the neuron state to zero and clear counters.
    pub fn reset(&mut self) {
        self.state = ContinuousHV::new(self.config.dim);
        self.total_time = 0.0;
        self.update_count = 0;
    }

    /// Get the total time evolved.
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Get the update count.
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Get the configuration.
    pub fn config(&self) -> &NeuronConfig {
        &self.config
    }

    /// Get the effective time constant for the current state and input.
    pub fn effective_tau(&self, input: &ContinuousHV) -> f32 {
        self.compute_tau(input)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Core dynamics
    // ─────────────────────────────────────────────────────────────────────

    /// Compute equilibrium state x_inf = f(W . x + U . u).
    #[inline]
    fn compute_equilibrium(&self, input: &ContinuousHV) -> ContinuousHV {
        let transformed_state = self.weight_hv.bind(&self.state);
        let masked_input = self.input_mask.bind(input);
        let combined = ContinuousHV::bundle(&[&transformed_state, &masked_input]);
        self.config.activation.apply(&combined)
    }

    /// Compute effective time constant tau(||x||, u).
    #[inline]
    fn compute_tau(&self, input: &ContinuousHV) -> f32 {
        let state_norm = self.state.norm();
        let input_adjustment = input.similarity(&self.tau_modulator);
        let tau = self.config.tau_base
            * (1.0 + self.config.backbone_tau * state_norm)
            * (1.0 + 0.2 * input_adjustment);
        tau.clamp(0.01, 10.0)
    }

    /// Compute the gating/interpolation factor sigma for the closed-form solution.
    #[inline]
    fn compute_gating(&self, input: &ContinuousHV, dt: f32) -> f32 {
        let tau = self.compute_tau(input);
        let dim = self.config.dim;
        let inv_dim = 1.0 / dim as f32;

        // Fused bundle+similarity to avoid allocating a full HV.
        let mut dot = 0.0f32;
        let mut bundle_norm_sq = 0.0f32;
        let mut gw_norm_sq = 0.0f32;
        let mut bias_sum = 0.0f32;

        for i in 0..dim {
            let b = (self.state.values[i] + input.values[i]) * 0.5;
            let g = self.gate_weight.values[i];
            dot += b * g;
            bundle_norm_sq += b * b;
            gw_norm_sq += g * g;
            bias_sum += self.gate_bias.values[i];
        }

        let denom = (bundle_norm_sq * gw_norm_sq).sqrt();
        let sim = if denom < 1e-10 { 0.0 } else { dot / denom };
        let gate_activation = sim + bias_sum * inv_dim;

        // Sigmoid gating with steepness control
        let sigma_base =
            1.0 / (1.0 + (-gate_activation * self.config.gating_steepness).exp());

        // Time-scaled gating: larger dt means more interpolation toward equilibrium
        // Clamp exponent to prevent f32 underflow
        let decay = (-dt / tau).max(-87.0).exp();
        let sigma = 1.0 - decay * (1.0 - sigma_base);
        sigma.clamp(0.0, 1.0)
    }

    /// **Closed-form evolution** -- O(1) temporal jump to any time horizon.
    ///
    /// This is the primary evolution method. It computes the equilibrium state
    /// x_inf and an adaptive gating factor sigma, then interpolates:
    ///
    /// ```text
    /// x(t + dt) = sigma * x_inf + (1 - sigma) * x(t)
    /// ```
    ///
    /// Cost is O(D) regardless of dt, enabling jumps of 1 ms or 100 s at equal cost.
    pub fn evolve_closed_form(&mut self, dt: f32, input: &ContinuousHV) {
        let x_inf = self.compute_equilibrium(input);
        let sigma = self.compute_gating(input, dt);
        self.state.lerp_in_place(&x_inf, sigma);
        self.apply_state_bounds();
        self.update_stats(dt);
    }

    /// **Fused closed-form evolution** -- zero intermediate allocations.
    ///
    /// Combines equilibrium computation and interpolation into a single pass
    /// through the dimension. Saves 4 x D x 4 bytes of intermediate allocations
    /// per call (256 KB at D=16,384).
    ///
    /// Uses `fast_tanh` rational approximation (max error ~0.004).
    pub fn evolve_closed_form_fused(&mut self, dt: f32, input: &ContinuousHV) {
        let sigma = self.compute_gating(input, dt);
        let one_minus_sigma = 1.0 - sigma;
        let dim = self.config.dim;

        match self.config.activation {
            Activation::Tanh => {
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let x_inf = fast_tanh(
                        (self.weight_hv.values[i] * state_i
                            + self.input_mask.values[i] * input.values[i])
                            * 0.5,
                    );
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
            Activation::Sigmoid => {
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let combined = (self.weight_hv.values[i] * state_i
                        + self.input_mask.values[i] * input.values[i])
                        * 0.5;
                    let x_inf = 1.0 / (1.0 + (-combined).exp());
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
            Activation::SiLU => {
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let combined = (self.weight_hv.values[i] * state_i
                        + self.input_mask.values[i] * input.values[i])
                        * 0.5;
                    let x_inf = combined / (1.0 + (-combined).exp());
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
            Activation::Identity => {
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let x_inf = (self.weight_hv.values[i] * state_i
                        + self.input_mask.values[i] * input.values[i])
                        * 0.5;
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
            Activation::BoundedTanh { bound } => {
                for i in 0..dim {
                    let state_i = self.state.values[i];
                    let x_inf = fast_tanh(
                        (self.weight_hv.values[i] * state_i
                            + self.input_mask.values[i] * input.values[i])
                            * 0.5
                            * bound,
                    );
                    self.state.values[i] = one_minus_sigma * state_i + sigma * x_inf;
                }
            }
        }

        self.apply_state_bounds();
        self.update_stats(dt);
    }

    /// Simple Hebbian learning: "what fires together wires together".
    ///
    /// Updates weight_hv based on the correlation between input and state,
    /// with momentum and weight decay.
    pub fn hebbian_update(&mut self, input: &ContinuousHV) {
        let lr = self.config.learning_rate;
        let correlation = input.bind(&self.state);

        let m = self.config.momentum;
        self.weight_momentum = self.weight_momentum.scale(m).add(&correlation.scale(lr));

        let decay = self.config.weight_decay;
        self.weight_hv = self
            .weight_hv
            .scale(1.0 - decay)
            .add(&self.weight_momentum);

        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Apply soft state bounds to prevent numerical explosion.
    #[inline]
    fn apply_state_bounds(&mut self) {
        let norm = self.state.norm();
        if norm > 5.0 {
            self.state.scale_in_place(5.0 / norm);
        }
    }

    /// Update running statistics.
    #[inline]
    fn update_stats(&mut self, dt: f32) {
        self.total_time += dt as f64;
        self.update_count += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> NeuronConfig {
        NeuronConfig {
            dim: 256,
            ..NeuronConfig::default()
        }
    }

    #[test]
    fn test_creation() {
        let neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        assert_eq!(neuron.state().dim(), 256);
        assert_eq!(neuron.total_time(), 0.0);
        assert_eq!(neuron.update_count(), 0);
    }

    #[test]
    fn test_evolution_changes_state() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        let before = neuron.state().clone();
        neuron.evolve_closed_form(0.1, &input);
        assert_ne!(neuron.state().values, before.values);
        assert_eq!(neuron.update_count(), 1);
    }

    #[test]
    fn test_large_dt_no_crash() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        // Should handle arbitrarily large dt without NaN or panic
        neuron.evolve_closed_form(1_000_000.0, &input);
        assert!(neuron.state().norm().is_finite());
    }

    #[test]
    fn test_tau_is_state_dependent() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        let tau_zero = neuron.effective_tau(&input);

        // Evolve to change state, then check tau changed
        for _ in 0..10 {
            neuron.evolve_closed_form(0.1, &input);
        }
        let tau_after = neuron.effective_tau(&input);
        assert!(
            (tau_zero - tau_after).abs() > 1e-6,
            "tau should change: {} vs {}",
            tau_zero,
            tau_after
        );
    }

    #[test]
    fn test_reset() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        neuron.evolve_closed_form(0.1, &input);
        assert!(neuron.state().norm() > 0.0);

        neuron.reset();
        assert!((neuron.state().norm()).abs() < 1e-10);
        assert_eq!(neuron.total_time(), 0.0);
        assert_eq!(neuron.update_count(), 0);
    }

    #[test]
    fn test_fused_matches_standard() {
        let config = small_config();
        let mut n1 = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        let mut n2 = HdcLtcUnifiedNeuron::new(config, 42);
        let input = ContinuousHV::new_random(256, 100);

        n1.evolve_closed_form(0.05, &input);
        n2.evolve_closed_form_fused(0.05, &input);

        // Should be very close (fast_tanh vs tanh difference)
        let sim = n1.state().similarity(n2.state());
        assert!(
            sim > 0.99,
            "Fused and standard should match closely: sim = {}",
            sim
        );
    }

    #[test]
    fn test_hebbian_update() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        // Evolve first so state is non-zero
        neuron.evolve_closed_form(0.1, &input);
        // Hebbian update should not crash
        neuron.hebbian_update(&input);
    }

    #[test]
    fn test_state_bounds() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        // Manually set a huge state
        neuron.set_state(ContinuousHV::from_values(vec![100.0; 256]));
        let input = ContinuousHV::new_random(256, 100);
        neuron.evolve_closed_form(0.01, &input);
        // State should be bounded
        assert!(neuron.state().norm() <= 5.01);
    }
}
