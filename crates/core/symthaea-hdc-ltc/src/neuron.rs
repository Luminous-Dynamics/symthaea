// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # HDC-LTC Unified Neuron
//!
//! A single neuron whose state is a hypervector that evolves through
//! Liquid Time-Constant (LTC) dynamics with a solver-free CfC-style update.
//!
//! ## Core ODE
//!
//! ```text
//! τ_i(h, I) · dh_i/dt = -h_i + f(W_i · h_i + U_i · I_i) · g_i(h, I)
//! ```
//!
//! where `W_i` and `U_i` are per-dimension HDC weight fields,
//! `g_i(h, I)` is a per-component gate, and `τ_i(h, I)` is a liquid
//! time constant.
//!
//! ## Solver-free temporal update
//!
//! For each event, the current state and input define a frozen local field.
//! The neuron then applies an exponential relaxation step in O(D) work:
//!
//! ```text
//! gate_i    = σ(gate_weight_i · h_i + input_mask_i · I_i + gate_bias_i)
//! control_i = tau_mod_i · |h_i| + tau_coupling · |I_i|
//! tau_i     = LogInterpolate(tau_min, tau_base, tau_max, control_i)
//! x_inf_i   = f(weight_hv_i · h_i + input_mask_i · I_i) · gate_i
//! alpha_i   = 1 - exp(-dt / tau_i)
//! h_i'      = (1 - alpha_i) · h_i + alpha_i · x_inf_i
//! ```
//!
//! `LogInterpolate` preserves `tau_base` as the neutral timescale while allowing
//! strong modulation to approach the full configured `[tau_min, tau_max]`
//! interval. This is important when the range spans orders of magnitude.

use crate::config::{Activation, NeuronConfig, fast_tanh};
use crate::continuous_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Unified HDC-LTC neuron with per-dimension continuous-time evolution.
///
/// The state is itself a continuous hypervector. Recurrent and input weights are
/// represented as hypervectors and therefore require O(D), rather than dense
/// O(D²), elementwise work per neuron update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcUnifiedNeuron {
    /// Current state hypervector h(t).
    state: ContinuousHV,
    /// Weight hypervector W: modulates state self-connection.
    weight_hv: ContinuousHV,
    /// Input mask hypervector U: modulates how input I(t) enters the recurrence.
    input_mask: ContinuousHV,
    /// Per-dimension tau modulator: drives the liquid time-constant field.
    tau_modulator: ContinuousHV,
    /// Per-dimension gate weight A: state contribution to the gating sigmoid.
    gate_weight: ContinuousHV,
    /// Per-dimension gate bias c.
    gate_bias: ContinuousHV,
    /// Configuration.
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
    /// Internal hypervectors are initialized via modified Gram-Schmidt
    /// orthogonalization so they have low mutual interference at startup.
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

    /// Overwrite the state directly (use sparingly — prefer evolution steps).
    pub fn set_state(&mut self, state: ContinuousHV) {
        assert_eq!(state.dim(), self.config.dim, "state dimension mismatch");
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

    /// Compute the mean effective time constant across all dimensions.
    pub fn mean_tau(&self, input: &ContinuousHV) -> f32 {
        let field = self.compute_tau_field(input);
        if field.is_empty() {
            return self.config.tau_base;
        }
        field.iter().sum::<f32>() / field.len() as f32
    }

    // ─────────────────────────────────────────────────────────────────────
    // Core temporal dynamics
    // ─────────────────────────────────────────────────────────────────────

    /// Primary fully fused solver-free temporal update.
    ///
    /// The equilibrium, gate, and liquid time constant are evaluated from the
    /// state/input at the beginning of the event interval and treated as frozen
    /// over that interval. The resulting exponential relaxation is O(D) in the
    /// hypervector dimension and independent of the number of ODE substeps that
    /// a numerical integrator would otherwise require.
    pub fn evolve_closed_form_fused(&mut self, dt: f32, input: &ContinuousHV) {
        assert_eq!(input.dim(), self.config.dim, "input dimension mismatch");
        if !dt.is_finite() {
            return;
        }

        let dim = self.config.dim;
        let tau_coupling = self.config.tau_coupling;
        let gating_steepness = self.config.gating_steepness;
        let dt_effective = dt.max(1e-9);

        for i in 0..dim {
            let h_i = self.state.values[i];
            let inp_i = input.values[i];

            let gate_pre = self.gate_weight.values[i] * h_i
                + self.input_mask.values[i] * inp_i
                + self.gate_bias.values[i];
            let gate_i = sigmoid(gate_pre * gating_steepness);

            let tau_control =
                self.tau_modulator.values[i] * h_i.abs() + tau_coupling * inp_i.abs();
            let tau_i = self.tau_from_control(tau_control);

            let equilibrium_pre =
                self.weight_hv.values[i] * h_i + self.input_mask.values[i] * inp_i;
            let x_inf_i = activate_scalar(self.config.activation, equilibrium_pre) * gate_i;

            // The exponent is lower-bounded to avoid underflow work; exp(-87)
            // is already effectively zero at f32 precision.
            let alpha_i = 1.0 - (-dt_effective / tau_i).max(-87.0).exp();
            let new_h_i = (1.0 - alpha_i) * h_i + alpha_i * x_inf_i;

            debug_assert!(gate_i.is_finite() && (0.0..=1.0).contains(&gate_i));
            debug_assert!(tau_i.is_finite());
            debug_assert!(tau_i >= self.config.tau_min && tau_i <= self.config.tau_max);
            debug_assert!(alpha_i.is_finite() && (0.0..=1.0).contains(&alpha_i));
            debug_assert!(new_h_i.is_finite());

            self.state.values[i] = new_h_i;
        }

        self.apply_state_bounds();
        self.update_stats(dt_effective);
    }

    /// Compatibility alias for `evolve_closed_form_fused`.
    #[inline]
    pub fn evolve_closed_form(&mut self, dt: f32, input: &ContinuousHV) {
        self.evolve_closed_form_fused(dt, input);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Learning
    // ─────────────────────────────────────────────────────────────────────

    /// Hebbian weight update: "what fires together wires together".
    ///
    /// Updates `weight_hv` based on input-state correlation, with momentum and
    /// L2 weight decay. `input_mask` is updated symmetrically. Liquid-field
    /// learning is intentionally a separate research tranche so its effect can
    /// be ablated independently from the timescale mapping introduced here.
    pub fn hebbian_update(&mut self, input: &ContinuousHV) {
        assert_eq!(input.dim(), self.config.dim, "input dimension mismatch");
        let lr = self.config.learning_rate;
        let m = self.config.momentum;
        let decay = self.config.weight_decay;

        let correlation = input.bind(&self.state);
        self.weight_momentum = self.weight_momentum.scale(m).add(&correlation.scale(lr));
        self.weight_hv = self.weight_hv.scale(1.0 - decay).add(&self.weight_momentum);
        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }

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
    // Liquid-timescale helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Map an unconstrained liquid-control signal onto the configured positive
    /// time-constant interval in log space.
    ///
    /// The mapping is calibrated so `control == 0` yields `tau_base` exactly
    /// (up to floating-point roundoff), while large negative/positive controls
    /// asymptotically approach `tau_min` / `tau_max`.
    fn tau_from_control(&self, control: f32) -> f32 {
        let tau_min = self.config.tau_min;
        let tau_max = self.config.tau_max;
        let tau_base = self.config.tau_base;

        debug_assert!(tau_min.is_finite() && tau_min > 0.0);
        debug_assert!(tau_max.is_finite() && tau_max > tau_min);
        debug_assert!(tau_base.is_finite() && tau_base >= tau_min && tau_base <= tau_max);

        // Fail closed to a small positive timescale for malformed serialized
        // configurations rather than allowing NaNs to poison the recurrent state.
        if !tau_min.is_finite()
            || !tau_max.is_finite()
            || !tau_base.is_finite()
            || tau_min <= 0.0
            || tau_max <= tau_min
        {
            return 1e-3;
        }

        let base = tau_base.clamp(tau_min, tau_max);
        let ln_min = tau_min.ln();
        let ln_max = tau_max.ln();
        let log_span = ln_max - ln_min;

        if log_span <= f32::EPSILON {
            return tau_min;
        }

        // Place tau_base at the neutral point of the sigmoid in log-time.
        let base_position = ((base.ln() - ln_min) / log_span).clamp(1e-6, 1.0 - 1e-6);
        let base_logit = (base_position / (1.0 - base_position)).ln();
        let position = sigmoid(control + base_logit);

        (ln_min + position * log_span)
            .exp()
            .clamp(tau_min, tau_max)
    }

    /// Compute the per-dimension liquid time-constant field τ_i(h, I).
    pub(crate) fn compute_tau_field(&self, input: &ContinuousHV) -> Vec<f32> {
        assert_eq!(input.dim(), self.config.dim, "input dimension mismatch");
        let tau_coupling = self.config.tau_coupling;
        let mut field = Vec::with_capacity(self.config.dim);

        for i in 0..self.config.dim {
            let control = self.tau_modulator.values[i] * self.state.values[i].abs()
                + tau_coupling * input.values[i].abs();
            field.push(self.tau_from_control(control));
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
// Fast scalar activations
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    // Stable form avoids overflow for large negative controls.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline(always)]
fn activate_scalar(activation: Activation, x: f32) -> f32 {
    match activation {
        Activation::Tanh => fast_tanh(x),
        Activation::Sigmoid => sigmoid(x),
        Activation::SiLU => silu(x),
        Activation::Identity => x,
        Activation::BoundedTanh { bound } => fast_tanh(x * bound),
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
        neuron.evolve_closed_form(1_000_000.0, &input);
        assert!(neuron.state().norm().is_finite());
    }

    #[test]
    fn test_non_finite_dt_is_rejected_without_state_change() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        let before = neuron.state().clone();
        neuron.evolve_closed_form(f32::NAN, &input);
        assert_eq!(neuron.state(), &before);
        assert_eq!(neuron.update_count(), 0);
    }

    #[test]
    fn test_state_bounded_after_evolution() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 7);
        neuron.set_state(ContinuousHV::from_values(vec![100.0; 256]));
        let input = ContinuousHV::new_random(256, 42);
        neuron.evolve_closed_form(0.01, &input);
        assert!(neuron.state().norm() <= 5.01, "norm={}", neuron.state().norm());
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

    #[test]
    fn test_tau_field_length() {
        let neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        assert_eq!(neuron.compute_tau_field(&input).len(), 256);
    }

    #[test]
    fn test_tau_field_bounded() {
        let cfg = small_config();
        let neuron = HdcLtcUnifiedNeuron::new(cfg.clone(), 42);
        let input = ContinuousHV::new_random(256, 100);
        for tau in neuron.compute_tau_field(&input) {
            assert!(tau >= cfg.tau_min, "tau {} < tau_min {}", tau, cfg.tau_min);
            assert!(tau <= cfg.tau_max, "tau {} > tau_max {}", tau, cfg.tau_max);
        }
    }

    #[test]
    fn test_neutral_tau_control_recovers_tau_base() {
        let neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let tau = neuron.tau_from_control(0.0);
        let rel_err = (tau - neuron.config.tau_base).abs() / neuron.config.tau_base;
        assert!(rel_err < 1e-5, "neutral tau={} base={}", tau, neuron.config.tau_base);
    }

    #[test]
    fn test_tau_mapping_is_monotonic() {
        let neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let controls = [-20.0_f32, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0];
        let mut previous = neuron.config.tau_min;
        for control in controls {
            let tau = neuron.tau_from_control(control);
            assert!(tau >= previous, "tau mapping not monotonic at control={control}");
            previous = tau;
        }
    }

    #[test]
    fn test_tau_mapping_reaches_configured_orders_of_magnitude() {
        let neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let low = neuron.tau_from_control(-20.0);
        let neutral = neuron.tau_from_control(0.0);
        let high = neuron.tau_from_control(20.0);

        assert!(low <= neuron.config.tau_min * 1.01, "low tau={low}");
        assert!(high >= neuron.config.tau_max * 0.99, "high tau={high}");
        assert!(high / low > 90_000.0, "tau span too small: low={low} high={high}");
        assert!((neutral - neuron.config.tau_base).abs() < 1e-5);
    }

    #[test]
    fn test_tau_field_varies_across_dims() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        for _ in 0..10 {
            neuron.evolve_closed_form(0.1, &input);
        }
        let field = neuron.compute_tau_field(&input);
        let min_tau = field.iter().copied().fold(f32::MAX, f32::min);
        let max_tau = field.iter().copied().fold(f32::MIN, f32::max);
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

    #[test]
    fn test_evolve_closed_form_alias_matches_fused() {
        let config = small_config();
        let mut n1 = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        let mut n2 = HdcLtcUnifiedNeuron::new(config, 42);
        let input = ContinuousHV::new_random(256, 100);
        n1.evolve_closed_form(0.05, &input);
        n2.evolve_closed_form_fused(0.05, &input);
        let sim = n1.state().similarity(n2.state());
        assert!((sim - 1.0).abs() < 1e-5, "alias and fused diverged: sim={}", sim);
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
        assert!(neuron.state().norm().is_finite());
    }

    #[test]
    fn test_irregular_partitions_converge_to_same_attractor() {
        let config = small_config();
        let input = ContinuousHV::new_random(256, 77);
        let mut regular = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        let mut irregular = HdcLtcUnifiedNeuron::new(config, 42);

        for _ in 0..100 {
            regular.evolve_closed_form(0.01, &input);
        }

        // Exactly one second total, partitioned irregularly.
        let dts = [0.001_f32, 0.049, 0.2, 0.1, 0.003, 0.047, 0.3, 0.1, 0.096, 0.104];
        let total: f32 = dts.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        for dt in dts {
            irregular.evolve_closed_form(dt, &input);
        }

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
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let extreme_input = ContinuousHV::from_values(vec![1e6; 256]);
        neuron.evolve_closed_form(0.1, &extreme_input);
        assert!(neuron.state().norm().is_finite());
        assert!(neuron.state().norm() <= 5.01);

        let mut sanitized = ContinuousHV::new_random(256, 123);
        sanitized.values[0] = 0.0;
        sanitized.values[1] = 0.0;
        sanitized.values[2] = 0.0;
        neuron.evolve_closed_form(0.1, &sanitized);
        assert!(neuron.state().norm().is_finite());
    }
}
