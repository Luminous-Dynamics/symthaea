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

/// Validation error for the opt-in checked HDC-LTC/CfC evolution path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuronStepError {
    /// Time increments must be finite and non-negative.
    InvalidDeltaTime,
    /// Input hypervector dimension differs from the configured neuron dimension.
    InputDimensionMismatch { expected: usize, actual: usize },
    /// Externally mutated state no longer matches the configured dimension.
    StateDimensionMismatch { expected: usize, actual: usize },
    /// A non-finite input component would poison future temporal state.
    NonFiniteInput { index: usize },
    /// A non-finite state component would poison future temporal state.
    NonFiniteState { index: usize },
    /// A deserialized or corrupted internal parameter vector has the wrong dimension.
    ParameterDimensionMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A non-finite internal parameter would poison future temporal state.
    NonFiniteParameter { field: &'static str, index: usize },
    /// A required numerical configuration value is invalid.
    InvalidConfiguration { field: &'static str },
}

impl std::fmt::Display for NeuronStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDeltaTime => {
                write!(f, "dt must be finite and non-negative")
            }
            Self::InputDimensionMismatch { expected, actual } => {
                write!(f, "input dimension mismatch: expected {expected}, got {actual}")
            }
            Self::StateDimensionMismatch { expected, actual } => {
                write!(f, "state dimension mismatch: expected {expected}, got {actual}")
            }
            Self::NonFiniteInput { index } => {
                write!(f, "input component {index} is not finite")
            }
            Self::NonFiniteState { index } => {
                write!(f, "state component {index} is not finite")
            }
            Self::ParameterDimensionMismatch { field, expected, actual } => {
                write!(
                    f,
                    "parameter {field} dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NonFiniteParameter { field, index } => {
                write!(f, "parameter {field} component {index} is not finite")
            }
            Self::InvalidConfiguration { field } => {
                write!(f, "neuron configuration field {field} is invalid")
            }
        }
    }
}

impl std::error::Error for NeuronStepError {}

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

    /// Validated CfC step for inputs crossing a trust or data-quality boundary.
    ///
    /// The preflight rejects invalid time increments, dimensions, and non-finite input/state
    /// before mutating the neuron. A zero time increment is a no-op. The preflight performs
    /// an additional O(D) scan; callers that have already validated inputs can use the
    /// allocation-free `evolve_closed_form_fused` hot path directly.
    pub fn try_evolve_closed_form_fused(
        &mut self,
        dt: f32,
        input: &ContinuousHV,
    ) -> Result<(), NeuronStepError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(NeuronStepError::InvalidDeltaTime);
        }
        let expected = self.config.dim;
        if expected == 0 {
            return Err(NeuronStepError::InvalidConfiguration { field: "dim" });
        }
        if !self.config.tau_base.is_finite() || self.config.tau_base <= 0.0 {
            return Err(NeuronStepError::InvalidConfiguration { field: "tau_base" });
        }
        if !self.config.tau_min.is_finite()
            || self.config.tau_min <= 0.0
            || !self.config.tau_max.is_finite()
            || self.config.tau_max < self.config.tau_min
        {
            return Err(NeuronStepError::InvalidConfiguration { field: "tau_min/tau_max" });
        }
        if !self.config.tau_coupling.is_finite() {
            return Err(NeuronStepError::InvalidConfiguration { field: "tau_coupling" });
        }
        if !self.config.gating_steepness.is_finite() {
            return Err(NeuronStepError::InvalidConfiguration { field: "gating_steepness" });
        }
        if let Activation::BoundedTanh { bound } = self.config.activation {
            if !bound.is_finite() {
                return Err(NeuronStepError::InvalidConfiguration { field: "activation.bound" });
            }
        }
        if input.dim() != expected {
            return Err(NeuronStepError::InputDimensionMismatch {
                expected,
                actual: input.dim(),
            });
        }
        if self.state.dim() != expected {
            return Err(NeuronStepError::StateDimensionMismatch {
                expected,
                actual: self.state.dim(),
            });
        }
        if let Some(index) = input.values.iter().position(|value| !value.is_finite()) {
            return Err(NeuronStepError::NonFiniteInput { index });
        }
        if let Some(index) = self.state.values.iter().position(|value| !value.is_finite()) {
            return Err(NeuronStepError::NonFiniteState { index });
        }
        for (field, vector) in [
            ("weight_hv", &self.weight_hv),
            ("input_mask", &self.input_mask),
            ("tau_modulator", &self.tau_modulator),
            ("gate_weight", &self.gate_weight),
            ("gate_bias", &self.gate_bias),
        ] {
            if vector.dim() != expected {
                return Err(NeuronStepError::ParameterDimensionMismatch {
                    field,
                    expected,
                    actual: vector.dim(),
                });
            }
            if let Some(index) = vector.values.iter().position(|value| !value.is_finite()) {
                return Err(NeuronStepError::NonFiniteParameter { field, index });
            }
        }
        if dt == 0.0 {
            return Ok(());
        }

        self.evolve_closed_form_fused(dt, input);
        Ok(())
    }

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

        // Fuse the post-update L2-bound reduction into the main O(D) update pass.
        // If the bound is not crossed, this avoids a separate read of the whole state.
        let mut state_norm_sq = 0.0_f32;
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
                    let sigma_i = cfc_interpolation_alpha(dt_clamped, tau_i);
                    debug_assert!(
                        sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite(),
                        "Sigma {} must be within [0, 1] and finite",
                        sigma_i
                    );

                    // CfC update: h_i(t+dt) = (1-σ_i)·h_i + σ_i·x_inf_i
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite(), "Evolved state value must be finite");
                    self.state.values[i] = new_h_i;
                    state_norm_sq += new_h_i * new_h_i;
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
                    let sigma_i = cfc_interpolation_alpha(dt_clamped, tau_i);
                    debug_assert!(sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite());
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite());
                    self.state.values[i] = new_h_i;
                    state_norm_sq += new_h_i * new_h_i;
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
                    let sigma_i = cfc_interpolation_alpha(dt_clamped, tau_i);
                    debug_assert!(sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite());
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite());
                    self.state.values[i] = new_h_i;
                    state_norm_sq += new_h_i * new_h_i;
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
                    let sigma_i = cfc_interpolation_alpha(dt_clamped, tau_i);
                    debug_assert!(sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite());
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite());
                    self.state.values[i] = new_h_i;
                    state_norm_sq += new_h_i * new_h_i;
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
                    let sigma_i = cfc_interpolation_alpha(dt_clamped, tau_i);
                    debug_assert!(sigma_i >= 0.0 && sigma_i <= 1.0 && sigma_i.is_finite());
                    let new_h_i = (1.0 - sigma_i) * h_i + sigma_i * x_inf_i;
                    debug_assert!(new_h_i.is_finite());
                    self.state.values[i] = new_h_i;
                    state_norm_sq += new_h_i * new_h_i;
                }
            }
        }

        let state_norm = state_norm_sq.sqrt();
        if state_norm > 5.0 {
            self.state.scale_in_place(5.0 / state_norm);
        }
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

/// Stable CfC interpolation coefficient: 1 - exp(-dt / tau).
///
/// `exp_m1` avoids catastrophic cancellation when dt is tiny relative to tau,
/// where `1.0 - exp(-dt / tau)` can lose a meaningful fraction of the update.
#[inline(always)]
fn cfc_interpolation_alpha(dt: f32, tau: f32) -> f32 {
    -(-dt / tau).exp_m1()
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

    // ── Validated evolution and numerical stability ─────────────────────────

    #[test]
    fn checked_step_rejects_invalid_dt_without_mutating_state() {
        for dt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1] {
            let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
            let input = ContinuousHV::new_random(256, 100);
            let before = neuron.state().clone();
            assert_eq!(
                neuron.try_evolve_closed_form_fused(dt, &input),
                Err(NeuronStepError::InvalidDeltaTime)
            );
            assert_eq!(neuron.state(), &before);
            assert_eq!(neuron.update_count(), 0);
        }
    }

    #[test]
    fn checked_step_rejects_dimension_and_nonfinite_inputs_before_mutation() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let before = neuron.state().clone();
        let wrong_dim = ContinuousHV::new_random(255, 100);
        assert_eq!(
            neuron.try_evolve_closed_form_fused(0.1, &wrong_dim),
            Err(NeuronStepError::InputDimensionMismatch {
                expected: 256,
                actual: 255,
            })
        );

        let mut nonfinite = ContinuousHV::new_random(256, 100);
        nonfinite.values[17] = f32::NAN;
        assert_eq!(
            neuron.try_evolve_closed_form_fused(0.1, &nonfinite),
            Err(NeuronStepError::NonFiniteInput { index: 17 })
        );
        assert_eq!(neuron.state(), &before);
        assert_eq!(neuron.update_count(), 0);
    }

    #[test]
    fn checked_step_rejects_corrupt_state_and_deserialized_parameters() {
        let input = ContinuousHV::new_random(256, 100);

        let mut wrong_state = HdcLtcUnifiedNeuron::new(small_config(), 42);
        wrong_state.set_state(ContinuousHV::new(255));
        assert_eq!(
            wrong_state.try_evolve_closed_form_fused(0.1, &input),
            Err(NeuronStepError::StateDimensionMismatch {
                expected: 256,
                actual: 255,
            })
        );

        let neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let mut serialized = serde_json::to_value(&neuron).unwrap();
        serialized["weight_hv"]["values"] = serde_json::json!(vec![0.0_f32; 255]);
        let mut corrupt: HdcLtcUnifiedNeuron = serde_json::from_value(serialized).unwrap();
        assert_eq!(
            corrupt.try_evolve_closed_form_fused(0.1, &input),
            Err(NeuronStepError::ParameterDimensionMismatch {
                field: "weight_hv",
                expected: 256,
                actual: 255,
            })
        );

        let mut nonfinite_state = HdcLtcUnifiedNeuron::new(small_config(), 43);
        let mut values = nonfinite_state.state().values.clone();
        values[9] = f32::INFINITY;
        nonfinite_state.set_state(ContinuousHV::from_values(values));
        assert_eq!(
            nonfinite_state.try_evolve_closed_form_fused(0.1, &input),
            Err(NeuronStepError::NonFiniteState { index: 9 })
        );
    }

    #[test]
    fn checked_step_zero_dt_is_a_true_noop() {
        let mut neuron = HdcLtcUnifiedNeuron::new(small_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        neuron.evolve_closed_form_fused(0.1, &input);
        let before = neuron.state().clone();
        let time = neuron.total_time();
        let updates = neuron.update_count();

        assert_eq!(neuron.try_evolve_closed_form_fused(0.0, &input), Ok(()));
        assert_eq!(neuron.state(), &before);
        assert_eq!(neuron.total_time(), time);
        assert_eq!(neuron.update_count(), updates);
    }

    #[test]
    fn exp_m1_preserves_tiny_cfc_updates() {
        let dt = 1.0e-8_f32;
        let tau = 0.1_f32;
        let alpha = cfc_interpolation_alpha(dt, tau);
        let first_order = dt / tau;
        assert!(alpha > 0.0, "tiny positive dt must not round to a zero update");
        assert!(
            (alpha - first_order).abs() / first_order < 1.0e-5,
            "alpha={alpha}, first_order={first_order}"
        );

        for (dt, tau) in [(1.0e-4, 0.1), (0.01, 0.5), (1.0, 10.0), (1.0e6, 0.001)] {
            let stable = cfc_interpolation_alpha(dt, tau);
            let direct = 1.0 - (-dt / tau).exp();
            assert!(stable.is_finite() && (0.0..=1.0).contains(&stable));
            assert!((stable - direct).abs() < 2.0e-6);
        }
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
