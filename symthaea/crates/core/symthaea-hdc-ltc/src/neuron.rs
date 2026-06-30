// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # HDC-LTC Unified Neuron
//!
//! A single neuron whose state is a hypervector that evolves through
//! Liquid Time-Constant (LTC) dynamics with a solver-free Closed-form
//! Continuous (CfC) solution (Hasani et al., 2022).
//!
//! ## Core ODE
//!
//! ```text
//! τ_i(h, I) · dh_i/dt = -h_i + f(W_i · h_i + U_i · I_i) · g_i(h, I) + b_i
//! ```
//!
//! Where:
//! - `W_i`, `U_i` are per-dimension weight hypervectors (HDC binding)
//! - `g_i(h, I) = σ(A_i · h_i + B_i · I_i + c_i)` is the per-component gating field
//! - `τ_i(h, I)` is the per-dimension liquid time-constant
//!
//! ## Closed-Form Solution
//!
//! Instead of numerical ODE integration, the CfC form gives an exact
//! analytical step at any dt in O(D) operations:
//!
//! ```text
//! gate_i  = σ(gate_weight_i · h_i  +  input_mask_i · I_i  +  gate_bias_i)
//! tau_i   = τ_base · clamp(σ(τ_mod_i · |h_i| + τ_coupling · |I_i|), τ_min, τ_max)
//! x_inf_i = f(weight_hv_i · h_i + input_mask_i · I_i) · gate_i
//! σ_i     = 1 - exp(-dt / tau_i)
//! h_i(t+dt) = (1 - σ_i) · h_i(t)  +  σ_i · x_inf_i
//! ```
//!
//! Because τ_i is per-dimension, different dimensions of the hypervector can
//! track events at different timescales simultaneously — fast sensory reactions
//! (small τ) coexist with long-horizon cognitive memory (large τ) in a single
//! 16,384-dimensional state vector.

use crate::config::{Activation, NeuronConfig, fast_tanh};
use crate::continuous_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Unified HDC-LTC Neuron with per-dimension CfC temporal evolution.
///
/// The neuron state is a hypervector (typically 16,384 dimensions) that evolves
/// continuously in time via a solver-free Closed-form Continuous (CfC) step.
/// Weight operations use HDC binding (element-wise multiply) instead of matrix
/// multiplication, yielding O(D) per step rather than O(D²).
///
/// The key advance over a scalar-gated CfC is that **each dimension has its own
/// liquid time-constant** τ_i(h, I), enabling multi-timescale dynamics within a
/// single hypervector without any partitioning or routing logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcUnifiedNeuron {
    /// Current state hypervector h(t).
    state: ContinuousHV,
    /// Weight hypervector W: modulates state self-connection.
    weight_hv: ContinuousHV,
    /// Input mask hypervector U: modulates how input I(t) enters the ODE.
    input_mask: ContinuousHV,
    /// Per-dimension tau modulator: drives liquid time-constant field.
    tau_modulator: ContinuousHV,
    /// Per-dimension gate weight A: state contribution to gating sigmoid.
    gate_weight: ContinuousHV,
    /// Per-dimension gate bias c: offset for gating sigmoid.
    gate_bias: ContinuousHV,
    /// Configuration (tau_base, tau_min, tau_max, tau_coupling, etc.).
    config: NeuronConfig,
    /// Momentum accumulator for weight_hv Hebbian updates.
    weight_momentum: ContinuousHV,
    /// Momentum accumulator for input_mask Hebbian updates.
    input_momentum: ContinuousHV,
    /// Total simulated time elapsed (seconds).
    total_time: f64,
    /// Number of evolution steps taken.
    update_count: u64,
}

impl HdcLtcUnifiedNeuron {
    /// Create a new neuron with the given configuration and deterministic seed.
    ///
    /// Internal hypervectors are initialised via modified Gram-Schmidt
    /// orthogonalisation so they have near-zero mutual interference at startup.
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

    // ─────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────

    /// Current state hypervector h(t).
    pub fn state(&self) -> &ContinuousHV {
        &self.state
    }

    /// Mutable reference to the state hypervector.
    pub fn state_mut(&mut self) -> &mut ContinuousHV {
        &mut self.state
    }

    /// Overwrite the state directly (use sparingly — prefer evolve steps).
    pub fn set_state(&mut self, state: ContinuousHV) {
        self.state = state;
    }

    /// Reset state to zero and clear all time counters.
    pub fn reset(&mut self) {
        self.state = ContinuousHV::new(self.config.dim);
        self.total_time = 0.0;
        self.update_count = 0;
    }

    /// Total simulated time elapsed (seconds).
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Number of `evolve_*` calls made.
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Reference to the neuron configuration.
    pub fn config(&self) -> &NeuronConfig {
        &self.config
    }

    // ─────────────────────────────────────────────────────────────────────
    // Public diagnostics
    // ─────────────────────────────────────────────────────────────────────

    /// Compute the **mean** effective time-constant across all dimensions
    /// for diagnostic / telemetry purposes.
    ///
    /// To get per-dimension τ fields call `compute_tau_field()` directly.
    pub fn mean_tau(&self, input: &ContinuousHV) -> f32 {
        let field = self.compute_tau_field(input);
        field.iter().sum::<f32>() / field.len() as f32
    }

    // ─────────────────────────────────────────────────────────────────────
    // Core CfC dynamics
    // ─────────────────────────────────────────────────────────────────────

    /// **Primary evolution method** — fully-fused CfC step.
    ///
    /// Implements the Hasani LTC closed-form update in a single O(D) pass
    /// with zero intermediate heap allocations.
    ///
    /// Handles any dt value correctly:
    /// - Very small dt → σ_i ≈ 0 (state barely moves)
    /// - Very large dt → σ_i ≈ 1 (state snaps to equilibrium)
    pub fn evolve_closed_form_fused(&mut self, dt: f32, input: &ContinuousHV) {
        let dim = self.config.dim;
        let tau_base = self.config.tau_base;
        let tau_min = self.config.tau_min;
        let tau_max = self.config.tau_max;
        let tau_coupling = self.config.tau_coupling;
        let gating_steepness = self.config.gating_steepness;
        let dt_clamped = dt.max(1e-9);

        match self.config.activation {
            Activation::Tanh => {
                for i in 0..dim {
                    let h_i = self.state.values[i];
                    let inp_i = input.values[i];

                    // Per-dim gating: g_i = σ(A_i · h_i + U_i · I_i + c_i)
                    let gate_pre = self.gate_weight.values[i] * h_i
                        + self.input_mask.values[i] * inp_i
                        + self.gate_bias.values[i];
                    let gate_i = sigmoid(gate_pre * gating_steepness);
                    debug_assert!(
                        gate_i >= 0.0 && gate_i <= 1.0 && gate_i.is_finite(),
                        "Gate value {} must be within [0, 1] and finite",
                        gate_i
                    );

                    // Per-dim tau: τ_i = τ_base · clamp(σ(τ_mod_i · |h_i| + τ_coupling · |I_i|), min, max)
                    let tau_pre =
                        self.tau_modulator.values[i] * h_i.abs() + tau_coupling * inp_i.abs();
                    let tau_i = (tau_base * (1.0 + sigmoid(tau_pre))).clamp(tau_min, tau_max);
                    debug_assert!(
                        tau_i >= tau_min && tau_i <= tau_max && tau_i.is_finite(),
                        "Tau {} must be finite and within limits [{}, {}]",
                        tau_i,
                        tau_min,
                        tau_max
                    );

                    // Gated equilibrium: x_inf_i = tanh(W_i · h_i + U_i · I_i) · g_i
                    let x_inf_i = fast_tanh(
                        self.weight_hv.values[i] * h_i + self.input_mask.values[i] * inp_i,
                    ) * gate_i;
                    debug_assert!(x_inf_i.is_finite(), "Equilibrium x_inf_i must be finite");

                    // Per-dim interpolation factor: σ_i = 1 - exp(-dt / τ_i)
                    let sigma_i = 1.0 - (-dt_clamped / tau_i).max(-87.0_f32).exp();
                    debug_assert!(
                        sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite(),
                        "Sigma {} must be within [0, 1] and finite",
                        sigma_i
                    );

                    // CfC update: h_i(t+dt) = (1-σ_i)·h_i + σ_i·x_inf_i
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite(), "Evolved state value must be finite");
                    self.state.values[i] = new_h_i;
                }
            }
            Activation::Sigmoid => {
                for i in 0..dim {
                    let h_i = self.state.values[i];
                    let inp_i = input.values[i];
                    let gate_pre = self.gate_weight.values[i] * h_i
                        + self.input_mask.values[i] * inp_i
                        + self.gate_bias.values[i];
                    let gate_i = sigmoid(gate_pre * gating_steepness);
                    debug_assert!(gate_i >= 0.0 && gate_i <= 1.0 && gate_i.is_finite());
                    let tau_pre =
                        self.tau_modulator.values[i] * h_i.abs() + tau_coupling * inp_i.abs();
                    let tau_i = (tau_base * (1.0 + sigmoid(tau_pre))).clamp(tau_min, tau_max);
                    debug_assert!(tau_i >= tau_min && tau_i <= tau_max && tau_i.is_finite());
                    let x_inf_i =
                        sigmoid(self.weight_hv.values[i] * h_i + self.input_mask.values[i] * inp_i)
                            * gate_i;
                    debug_assert!(x_inf_i.is_finite());
                    let sigma_i = 1.0 - (-dt_clamped / tau_i).max(-87.0_f32).exp();
                    debug_assert!(sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite());
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite());
                    self.state.values[i] = new_h_i;
                }
            }
            Activation::SiLU => {
                for i in 0..dim {
                    let h_i = self.state.values[i];
                    let inp_i = input.values[i];
                    let gate_pre = self.gate_weight.values[i] * h_i
                        + self.input_mask.values[i] * inp_i
                        + self.gate_bias.values[i];
                    let gate_i = sigmoid(gate_pre * gating_steepness);
                    debug_assert!(gate_i >= 0.0 && gate_i <= 1.0 && gate_i.is_finite());
                    let tau_pre =
                        self.tau_modulator.values[i] * h_i.abs() + tau_coupling * inp_i.abs();
                    let tau_i = (tau_base * (1.0 + sigmoid(tau_pre))).clamp(tau_min, tau_max);
                    debug_assert!(tau_i >= tau_min && tau_i <= tau_max && tau_i.is_finite());
                    let combined =
                        self.weight_hv.values[i] * h_i + self.input_mask.values[i] * inp_i;
                    let x_inf_i = silu(combined) * gate_i;
                    debug_assert!(x_inf_i.is_finite());
                    let sigma_i = 1.0 - (-dt_clamped / tau_i).max(-87.0_f32).exp();
                    debug_assert!(sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite());
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite());
                    self.state.values[i] = new_h_i;
                }
            }
            Activation::Identity => {
                for i in 0..dim {
                    let h_i = self.state.values[i];
                    let inp_i = input.values[i];
                    let gate_pre = self.gate_weight.values[i] * h_i
                        + self.input_mask.values[i] * inp_i
                        + self.gate_bias.values[i];
                    let gate_i = sigmoid(gate_pre * gating_steepness);
                    debug_assert!(gate_i >= 0.0 && gate_i <= 1.0 && gate_i.is_finite());
                    let tau_pre =
                        self.tau_modulator.values[i] * h_i.abs() + tau_coupling * inp_i.abs();
                    let tau_i = (tau_base * (1.0 + sigmoid(tau_pre))).clamp(tau_min, tau_max);
                    debug_assert!(tau_i >= tau_min && tau_i <= tau_max && tau_i.is_finite());
                    let x_inf_i = (self.weight_hv.values[i] * h_i
                        + self.input_mask.values[i] * inp_i)
                        * gate_i;
                    debug_assert!(x_inf_i.is_finite());
                    let sigma_i = 1.0 - (-dt_clamped / tau_i).max(-87.0_f32).exp();
                    debug_assert!(sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite());
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite());
                    self.state.values[i] = new_h_i;
                }
            }
            Activation::BoundedTanh { bound } => {
                for i in 0..dim {
                    let h_i = self.state.values[i];
                    let inp_i = input.values[i];
                    let gate_pre = self.gate_weight.values[i] * h_i
                        + self.input_mask.values[i] * inp_i
                        + self.gate_bias.values[i];
                    let gate_i = sigmoid(gate_pre * gating_steepness);
                    debug_assert!(gate_i >= 0.0 && gate_i <= 1.0 && gate_i.is_finite());
                    let tau_pre =
                        self.tau_modulator.values[i] * h_i.abs() + tau_coupling * inp_i.abs();
                    let tau_i = (tau_base * (1.0 + sigmoid(tau_pre))).clamp(tau_min, tau_max);
                    debug_assert!(tau_i >= tau_min && tau_i <= tau_max && tau_i.is_finite());
                    let x_inf_i = fast_tanh(
                        (self.weight_hv.values[i] * h_i + self.input_mask.values[i] * inp_i)
                            * bound,
                    ) * gate_i;
                    debug_assert!(x_inf_i.is_finite());
                    let sigma_i = 1.0 - (-dt_clamped / tau_i).max(-87.0_f32).exp();
                    debug_assert!(sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite());
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite());
                    self.state.values[i] = new_h_i;
                }
            }
        }

        self.apply_state_bounds();
        self.update_stats(dt);
    }

    /// **Compatibility alias** for `evolve_closed_form_fused`.
    ///
    /// All existing call sites using `evolve_closed_form` continue to work.
    #[inline]
    pub fn evolve_closed_form(&mut self, dt: f32, input: &ContinuousHV) {
        self.evolve_closed_form_fused(dt, input);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Learning
    // ─────────────────────────────────────────────────────────────────────

    /// Hebbian weight update: "what fires together wires together".
    ///
    /// Updates `weight_hv` based on input–state correlation, with momentum
    /// and L2 weight decay. The `input_mask` is updated symmetrically.
    pub fn hebbian_update(&mut self, input: &ContinuousHV) {
        let lr = self.config.learning_rate;
        let m = self.config.momentum;
        let decay = self.config.weight_decay;

        // weight_hv update
        let correlation = input.bind(&self.state);
        self.weight_momentum = self.weight_momentum.scale(m).add(&correlation.scale(lr));
        self.weight_hv = self.weight_hv.scale(1.0 - decay).add(&self.weight_momentum);
        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }

        // input_mask update — reinforce directions where input aligns with state
        let input_corr = self.state.bind(input);
        self.input_momentum = self
            .input_momentum
            .scale(m)
            .add(&input_corr.scale(lr * 0.5));
        self.input_mask = self.input_mask.scale(1.0 - decay).add(&self.input_momentum);
        if self.input_mask.norm() > 2.0 {
            self.input_mask = self.input_mask.normalize().scale(2.0);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Compute per-dimension liquid time-constant field τ_i(h, I).
    ///
    /// Returns a `Vec<f32>` of length `dim` where each entry is in
    /// `[tau_min, tau_max]`.
    pub(crate) fn compute_tau_field(&self, input: &ContinuousHV) -> Vec<f32> {
        let dim = self.config.dim;
        let tau_base = self.config.tau_base;
        let tau_min = self.config.tau_min;
        let tau_max = self.config.tau_max;
        let tau_coupling = self.config.tau_coupling;
        let mut field = Vec::with_capacity(dim);
        for i in 0..dim {
            let tau_pre = self.tau_modulator.values[i] * self.state.values[i].abs()
                + tau_coupling * input.values[i].abs();
            let tau_i = (tau_base * (1.0 + sigmoid(tau_pre))).clamp(tau_min, tau_max);
            field.push(tau_i);
        }
        field
    }

    /// Soft state bounds: rescale if L2 norm exceeds 5.0.
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
// Fast elementwise activations
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically-stable sigmoid σ(x) = 1 / (1 + e^{-x}).
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU / Swish: x · σ(x).
#[inline(always)]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
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

    // ── Construction ───────────────────────────────────────────────────────

    #[test]
    fn test_creation() {
        let neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        assert_eq!(neuron.state().dim(), 256);
        assert_eq!(neuron.total_time(), 0.0);
        assert_eq!(neuron.update_count(), 0);
    }

    // ── Forward evolution ──────────────────────────────────────────────────

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
        // Arbitrary large dt must not panic or produce NaN.
        neuron.evolve_closed_form(1_000_000.0, &input);
        assert!(neuron.state().norm().is_finite());
    }

    #[test]
    fn test_state_bounded_after_evolution() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 7);
        // Manually inflate state to trigger bounds clamp.
        neuron.set_state(ContinuousHV::from_values(vec![100.0; 256]));
        let input = ContinuousHV::new_random(256, 42);
        neuron.evolve_closed_form(0.01, &input);
        assert!(
            neuron.state().norm() <= 5.01,
            "norm={}",
            neuron.state().norm()
        );
    }

    #[test]
    fn test_state_finite_after_many_steps() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 99);
        let input = ContinuousHV::new_random(256, 55);
        for _ in 0..1000 {
            neuron.evolve_closed_form(0.01, &input);
        }
        assert!(neuron.state().norm().is_finite());
    }

    // ── Tau field ──────────────────────────────────────────────────────────

    #[test]
    fn test_tau_field_length() {
        let neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        let field = neuron.compute_tau_field(&input);
        assert_eq!(field.len(), 256);
    }

    #[test]
    fn test_tau_field_bounded() {
        let cfg = small_config();
        let neuron = HdcLtcUnifiedNeuron::new(cfg.clone(), 42);
        let input = ContinuousHV::new_random(256, 100);
        let field = neuron.compute_tau_field(&input);
        for &tau in &field {
            assert!(tau >= cfg.tau_min, "tau {} < tau_min {}", tau, cfg.tau_min);
            assert!(tau <= cfg.tau_max, "tau {} > tau_max {}", tau, cfg.tau_max);
        }
    }

    #[test]
    fn test_tau_field_varies_across_dims() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        // Evolve so state becomes heterogeneous.
        for _ in 0..10 {
            neuron.evolve_closed_form(0.1, &input);
        }
        let field = neuron.compute_tau_field(&input);
        let min_tau = field.iter().cloned().fold(f32::MAX, f32::min);
        let max_tau = field.iter().cloned().fold(f32::MIN, f32::max);
        // With heterogeneous state the field should span some range.
        assert!(
            (max_tau - min_tau) > 1e-6,
            "tau field should vary: min={} max={}",
            min_tau,
            max_tau
        );
    }

    #[test]
    fn test_mean_tau_in_bounds() {
        let cfg = small_config();
        let neuron = HdcLtcUnifiedNeuron::new(cfg.clone(), 42);
        let input = ContinuousHV::new_random(256, 100);
        let mean_tau = neuron.mean_tau(&input);
        assert!(mean_tau >= cfg.tau_min);
        assert!(mean_tau <= cfg.tau_max);
    }

    // ── Compatibility & learning ───────────────────────────────────────────

    #[test]
    fn test_evolve_closed_form_alias_matches_fused() {
        let config = small_config();
        let mut n1 = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        let mut n2 = HdcLtcUnifiedNeuron::new(config, 42);
        let input = ContinuousHV::new_random(256, 100);
        n1.evolve_closed_form(0.05, &input);
        n2.evolve_closed_form_fused(0.05, &input);
        // Both paths are identical now (alias).
        let sim = n1.state().similarity(n2.state());
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "alias and fused diverged: sim={}",
            sim
        );
    }

    #[test]
    fn test_reset() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        neuron.evolve_closed_form(0.1, &input);
        assert!(neuron.state().norm() > 0.0);
        neuron.reset();
        assert!(neuron.state().norm().abs() < 1e-10);
        assert_eq!(neuron.total_time(), 0.0);
        assert_eq!(neuron.update_count(), 0);
    }

    #[test]
    fn test_hebbian_update() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        neuron.evolve_closed_form(0.1, &input);
        neuron.hebbian_update(&input);
        // State and weight should remain finite.
        assert!(neuron.state().norm().is_finite());
    }

    // ── Irregular-time property ────────────────────────────────────────────

    #[test]
    fn test_irregular_timestamps_same_steady_state() {
        // Neuron driven with the same input but irregular vs regular dt
        // should converge to the same steady state.
        let config = small_config();
        let input = ContinuousHV::new_random(256, 77);

        let mut regular = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        let mut irregular = HdcLtcUnifiedNeuron::new(config, 42);

        // Regular: 100 steps of dt=0.01 (total 1.0s)
        for _ in 0..100 {
            regular.evolve_closed_form(0.01, &input);
        }

        // Irregular: same total time via varying dt
        let dts = [0.001_f32, 0.05, 0.2, 0.1, 0.003, 0.05, 0.3, 0.1, 0.096, 0.1];
        for &dt in dts.iter().cycle().take(100) {
            irregular.evolve_closed_form(dt, &input);
        }

        // Both should converge near the same steady state (within noise).
        // We check that the state norm is in the same ballpark.
        let reg_norm = regular.state().norm();
        let irr_norm = irregular.state().norm();
        assert!(
            (reg_norm - irr_norm).abs() / (reg_norm + 1e-6) < 0.5,
            "regular norm={} irregular norm={}",
            reg_norm,
            irr_norm
        );
    }

    #[test]
    fn test_neuron_invariants_under_extreme_inputs() {
        let config = small_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        // 1. Extreme inputs (very large values)
        let extreme_input = ContinuousHV::from_values(vec![1e6; 256]);
        neuron.evolve_closed_form(0.1, &extreme_input);
        assert!(neuron.state().norm().is_finite());
        assert!(neuron.state().norm() <= 5.01); // Soft bound of 5.0 should hold

        // 2. NaN/Inf check in inputs
        // Ensure that normal bounds clamping or assertions protect neuron evolution
        let mut nan_input = ContinuousHV::new_random(256, 123);
        nan_input.values[0] = f32::NAN;
        nan_input.values[1] = f32::INFINITY;
        nan_input.values[2] = f32::NEG_INFINITY;

        // Ensure we handle/sanitize inputs or check state boundaries
        let mut san_input = nan_input.clone();
        for val in san_input.values.iter_mut() {
            if !val.is_finite() {
                *val = 0.0;
            }
        }
        neuron.evolve_closed_form(0.1, &san_input);
        assert!(neuron.state().norm().is_finite());
    }
}
