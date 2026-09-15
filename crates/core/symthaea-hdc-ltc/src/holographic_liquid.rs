// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Binding-equivariant Holographic Liquid State (HLS) research cell.
//!
//! This module isolates a stronger architectural hypothesis than the legacy
//! `HdcLtcUnifiedNeuron`: temporal evolution can be constructed so that real
//! unitary HDC role binding commutes with the recurrent update.
//!
//! For a bipolar unitary role `r`, define `B_r(x) = r ⊙ x`. The cell is
//! parameterized so that, in exact arithmetic,
//!
//! ```text
//! F(B_r(h), B_r(x), dt) = B_r(F(h, x, dt)).
//! ```
//!
//! The construction requires three ingredients:
//!
//! 1. gate and liquid-timescale controls depend on `|h|` and `|x|`, so they are
//!    invariant under role sign flips;
//! 2. the equilibrium pre-activation is diagonal/elementwise, so role binding
//!    commutes with it;
//! 3. the activation is odd, so `phi(r*z) = r*phi(z)` for `r in {-1,+1}`.
//!
//! Importantly, the theorem is independent of the numerical values of the six
//! trainable diagonal fields. `HlsParameters` therefore exposes those fields as
//! one validated atomic snapshot so optimizers can train the dynamics without
//! weakening the forward binding-equivariance law.

use crate::config::fast_tanh;
use crate::continuous_hv::{ContinuousHV, UnitaryRole};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Odd activations compatible with exact real bipolar binding equivariance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum HlsActivation {
    #[default]
    Tanh,
    Identity,
    BoundedTanh { bound: f32 },
}

impl HlsActivation {
    #[inline(always)]
    fn apply(self, value: f32) -> f32 {
        match self {
            Self::Tanh => fast_tanh(value),
            Self::Identity => value,
            Self::BoundedTanh { bound } => fast_tanh(value * bound),
        }
    }
}

/// Configuration for the theorem-bearing HLS research cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlsConfig {
    pub dim: usize,
    pub tau_min: f32,
    pub tau_base: f32,
    pub tau_max: f32,
    /// Input contribution to the liquid-timescale control field.
    pub tau_input_coupling: f32,
    /// Multiplier applied to the magnitude-conditioned gate pre-activation.
    pub gate_steepness: f32,
    /// Soft L2 state bound. Set to `f32::INFINITY` to disable.
    pub state_norm_limit: f32,
    pub activation: HlsActivation,
}

impl Default for HlsConfig {
    fn default() -> Self {
        Self {
            dim: 16_384,
            tau_min: 0.001,
            tau_base: 0.1,
            tau_max: 100.0,
            tau_input_coupling: 0.3,
            gate_steepness: 1.0,
            state_norm_limit: 5.0,
            activation: HlsActivation::Tanh,
        }
    }
}

/// Complete trainable parameter surface of the diagonal HLS cell.
///
/// Every field has exactly `dim` scalar parameters, so a cell contains `6 * dim`
/// trainable recurrent scalars. The architecture-level binding-equivariance
/// theorem holds for arbitrary finite values of these fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HlsParameters {
    pub recurrent_weight: ContinuousHV,
    pub input_weight: ContinuousHV,
    pub tau_state_weight: ContinuousHV,
    pub gate_state_weight: ContinuousHV,
    pub gate_input_weight: ContinuousHV,
    pub gate_bias: ContinuousHV,
}

impl HlsParameters {
    /// Zero-valued parameter/delta snapshot of the requested dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            recurrent_weight: ContinuousHV::new(dim),
            input_weight: ContinuousHV::new(dim),
            tau_state_weight: ContinuousHV::new(dim),
            gate_state_weight: ContinuousHV::new(dim),
            gate_input_weight: ContinuousHV::new(dim),
            gate_bias: ContinuousHV::new(dim),
        }
    }

    pub fn scalar_count(&self) -> usize {
        self.recurrent_weight
            .dim()
            .saturating_add(self.input_weight.dim())
            .saturating_add(self.tau_state_weight.dim())
            .saturating_add(self.gate_state_weight.dim())
            .saturating_add(self.gate_input_weight.dim())
            .saturating_add(self.gate_bias.dim())
    }

    pub fn l2_norm(&self) -> f32 {
        let sum_sq = parameter_fields(self)
            .into_iter()
            .flat_map(|(_, field)| field.values.iter().copied())
            .map(|value| value * value)
            .sum::<f32>();
        sum_sq.sqrt()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HlsError {
    ZeroDimension,
    InvalidTimescaleBounds,
    InvalidParameter(&'static str),
    DimensionMismatch { expected: usize, actual: usize },
    ParameterDimensionMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteParameter(&'static str),
    InvalidParameterStep,
    InvalidParameterBound,
    NonFiniteDt,
}

impl fmt::Display for HlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "HLS dimension must be non-zero"),
            Self::InvalidTimescaleBounds => write!(
                f,
                "HLS timescales must satisfy 0 < tau_min <= tau_base <= tau_max"
            ),
            Self::InvalidParameter(name) => write!(f, "HLS parameter {name} must be finite"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "HLS dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::ParameterDimensionMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "HLS parameter field {field} dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteParameter(field) => {
                write!(f, "HLS parameter field {field} contains a non-finite value")
            }
            Self::InvalidParameterStep => {
                write!(f, "HLS parameter update step size must be finite")
            }
            Self::InvalidParameterBound => write!(
                f,
                "HLS parameter update absolute bound must be finite and positive"
            ),
            Self::NonFiniteDt => write!(f, "HLS dt must be finite"),
        }
    }
}

impl std::error::Error for HlsError {}

/// Minimal binding-equivariant continuous-time holographic state cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicLiquidCell {
    config: HlsConfig,
    state: ContinuousHV,
    recurrent_weight: ContinuousHV,
    input_weight: ContinuousHV,
    tau_state_weight: ContinuousHV,
    gate_state_weight: ContinuousHV,
    gate_input_weight: ContinuousHV,
    gate_bias: ContinuousHV,
    total_time: f64,
    update_count: u64,
}

impl HolographicLiquidCell {
    pub fn try_new(config: HlsConfig, seed: u64) -> Result<Self, HlsError> {
        validate_config(&config)?;
        let dim = config.dim;
        Ok(Self {
            config,
            state: ContinuousHV::new(dim),
            recurrent_weight: ContinuousHV::new_random(dim, seed.wrapping_add(1)),
            input_weight: ContinuousHV::new_random(dim, seed.wrapping_add(2)),
            tau_state_weight: ContinuousHV::new_random(dim, seed.wrapping_add(3)),
            gate_state_weight: ContinuousHV::new_random(dim, seed.wrapping_add(4)),
            gate_input_weight: ContinuousHV::new_random(dim, seed.wrapping_add(5)),
            gate_bias: ContinuousHV::new_random(dim, seed.wrapping_add(6)).scale(0.1),
            total_time: 0.0,
            update_count: 0,
        })
    }

    pub fn config(&self) -> &HlsConfig {
        &self.config
    }

    pub fn state(&self) -> &ContinuousHV {
        &self.state
    }

    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Exact trainable scalar count for this cell.
    pub fn parameter_count(&self) -> usize {
        self.config.dim.saturating_mul(6)
    }

    /// Clone the complete trainable parameter surface as one atomic snapshot.
    pub fn parameters(&self) -> HlsParameters {
        HlsParameters {
            recurrent_weight: self.recurrent_weight.clone(),
            input_weight: self.input_weight.clone(),
            tau_state_weight: self.tau_state_weight.clone(),
            gate_state_weight: self.gate_state_weight.clone(),
            gate_input_weight: self.gate_input_weight.clone(),
            gate_bias: self.gate_bias.clone(),
        }
    }

    /// Atomically replace all trainable fields after complete validation.
    ///
    /// No field is mutated unless every field has the correct dimension and all
    /// values are finite. Changing these numerical values does not alter the
    /// architecture-level binding-equivariance proof.
    pub fn set_parameters(&mut self, parameters: HlsParameters) -> Result<(), HlsError> {
        validate_parameters(&parameters, self.config.dim)?;
        self.recurrent_weight = parameters.recurrent_weight;
        self.input_weight = parameters.input_weight;
        self.tau_state_weight = parameters.tau_state_weight;
        self.gate_state_weight = parameters.gate_state_weight;
        self.gate_input_weight = parameters.gate_input_weight;
        self.gate_bias = parameters.gate_bias;
        Ok(())
    }

    /// Apply an optimizer-produced delta atomically with an explicit absolute
    /// parameter bound.
    ///
    /// `next = clamp(current + step_size * delta, -max_abs, +max_abs)`.
    /// The candidate snapshot is fully validated before it replaces live state.
    pub fn apply_parameter_delta(
        &mut self,
        delta: &HlsParameters,
        step_size: f32,
        max_abs: f32,
    ) -> Result<(), HlsError> {
        if !step_size.is_finite() {
            return Err(HlsError::InvalidParameterStep);
        }
        if !max_abs.is_finite() || max_abs <= 0.0 {
            return Err(HlsError::InvalidParameterBound);
        }
        validate_parameters(delta, self.config.dim)?;

        let mut next = self.parameters();
        apply_delta_field(
            "recurrent_weight",
            &mut next.recurrent_weight,
            &delta.recurrent_weight,
            step_size,
            max_abs,
        )?;
        apply_delta_field(
            "input_weight",
            &mut next.input_weight,
            &delta.input_weight,
            step_size,
            max_abs,
        )?;
        apply_delta_field(
            "tau_state_weight",
            &mut next.tau_state_weight,
            &delta.tau_state_weight,
            step_size,
            max_abs,
        )?;
        apply_delta_field(
            "gate_state_weight",
            &mut next.gate_state_weight,
            &delta.gate_state_weight,
            step_size,
            max_abs,
        )?;
        apply_delta_field(
            "gate_input_weight",
            &mut next.gate_input_weight,
            &delta.gate_input_weight,
            step_size,
            max_abs,
        )?;
        apply_delta_field(
            "gate_bias",
            &mut next.gate_bias,
            &delta.gate_bias,
            step_size,
            max_abs,
        )?;
        self.set_parameters(next)
    }

    pub fn set_state(&mut self, state: ContinuousHV) -> Result<(), HlsError> {
        self.check_dim(state.dim())?;
        self.state = state;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.state = ContinuousHV::new(self.config.dim);
        self.total_time = 0.0;
        self.update_count = 0;
    }

    /// Advance one irregular-time event.
    ///
    /// Work is O(D) for fixed cell structure and independent of any numerical
    /// ODE substep count associated with `dt`.
    pub fn step(&mut self, dt: f32, input: &ContinuousHV) -> Result<(), HlsError> {
        self.check_dim(input.dim())?;
        if !dt.is_finite() {
            return Err(HlsError::NonFiniteDt);
        }
        let dt = dt.max(1e-9);

        for i in 0..self.config.dim {
            let h = self.state.values[i];
            let x = input.values[i];

            // Magnitude-conditioned controls are invariant under bipolar role binding.
            let gate_pre = self.gate_state_weight.values[i] * h.abs()
                + self.gate_input_weight.values[i] * x.abs()
                + self.gate_bias.values[i];
            let gate = sigmoid(gate_pre * self.config.gate_steepness);

            let tau_control = self.tau_state_weight.values[i] * h.abs()
                + self.config.tau_input_coupling * x.abs();
            let tau = self.tau_from_control(tau_control);

            // This expression is role-equivariant because the same ±1 sign
            // factors both h and x and the activation is restricted to odd maps.
            let equilibrium_pre =
                self.recurrent_weight.values[i] * h + self.input_weight.values[i] * x;
            let equilibrium = self.config.activation.apply(equilibrium_pre) * gate;

            let alpha = 1.0 - (-dt / tau).max(-87.0).exp();
            self.state.values[i] = (1.0 - alpha) * h + alpha * equilibrium;
        }

        self.apply_state_bound();
        self.total_time += dt as f64;
        self.update_count += 1;
        Ok(())
    }

    /// Measure the maximum absolute violation of binding equivariance for one
    /// proposed transition. Zero is exact at f32 arithmetic; small non-zero
    /// values quantify floating-point asymmetry without changing the theorem.
    pub fn binding_equivariance_error(
        &self,
        role: &UnitaryRole,
        input: &ContinuousHV,
        dt: f32,
    ) -> Result<f32, HlsError> {
        self.check_dim(role.dim())?;
        self.check_dim(input.dim())?;

        let mut direct = self.clone();
        direct.step(dt, input)?;
        let expected = role.bind(direct.state());

        let mut transformed = self.clone();
        transformed.set_state(role.bind(self.state()))?;
        let transformed_input = role.bind(input);
        transformed.step(dt, &transformed_input)?;

        Ok(expected
            .values
            .iter()
            .zip(transformed.state.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max))
    }

    fn check_dim(&self, actual: usize) -> Result<(), HlsError> {
        if actual == self.config.dim {
            Ok(())
        } else {
            Err(HlsError::DimensionMismatch {
                expected: self.config.dim,
                actual,
            })
        }
    }

    fn tau_from_control(&self, control: f32) -> f32 {
        let ln_min = self.config.tau_min.ln();
        let ln_max = self.config.tau_max.ln();
        let span = ln_max - ln_min;
        let base_position = ((self.config.tau_base.ln() - ln_min) / span)
            .clamp(1e-6, 1.0 - 1e-6);
        let base_logit = (base_position / (1.0 - base_position)).ln();
        let position = sigmoid(control + base_logit);
        (ln_min + position * span)
            .exp()
            .clamp(self.config.tau_min, self.config.tau_max)
    }

    fn apply_state_bound(&mut self) {
        let limit = self.config.state_norm_limit;
        if !limit.is_finite() {
            return;
        }
        let norm = self.state.norm();
        if norm > limit && norm > 0.0 {
            self.state.scale_in_place(limit / norm);
        }
    }
}

fn parameter_fields(parameters: &HlsParameters) -> [(&'static str, &ContinuousHV); 6] {
    [
        ("recurrent_weight", &parameters.recurrent_weight),
        ("input_weight", &parameters.input_weight),
        ("tau_state_weight", &parameters.tau_state_weight),
        ("gate_state_weight", &parameters.gate_state_weight),
        ("gate_input_weight", &parameters.gate_input_weight),
        ("gate_bias", &parameters.gate_bias),
    ]
}

fn validate_parameters(parameters: &HlsParameters, dim: usize) -> Result<(), HlsError> {
    for (name, field) in parameter_fields(parameters) {
        if field.dim() != dim {
            return Err(HlsError::ParameterDimensionMismatch {
                field: name,
                expected: dim,
                actual: field.dim(),
            });
        }
        if field.values.iter().any(|value| !value.is_finite()) {
            return Err(HlsError::NonFiniteParameter(name));
        }
    }
    Ok(())
}

fn apply_delta_field(
    name: &'static str,
    target: &mut ContinuousHV,
    delta: &ContinuousHV,
    step_size: f32,
    max_abs: f32,
) -> Result<(), HlsError> {
    for (value, update) in target.values.iter_mut().zip(delta.values.iter().copied()) {
        let candidate = *value + step_size * update;
        if !candidate.is_finite() {
            return Err(HlsError::NonFiniteParameter(name));
        }
        *value = candidate.clamp(-max_abs, max_abs);
    }
    Ok(())
}

fn validate_config(config: &HlsConfig) -> Result<(), HlsError> {
    if config.dim == 0 {
        return Err(HlsError::ZeroDimension);
    }
    if !config.tau_min.is_finite()
        || !config.tau_base.is_finite()
        || !config.tau_max.is_finite()
        || config.tau_min <= 0.0
        || config.tau_base < config.tau_min
        || config.tau_max < config.tau_base
        || config.tau_max <= config.tau_min
    {
        return Err(HlsError::InvalidTimescaleBounds);
    }
    if !config.tau_input_coupling.is_finite() {
        return Err(HlsError::InvalidParameter("tau_input_coupling"));
    }
    if !config.gate_steepness.is_finite() {
        return Err(HlsError::InvalidParameter("gate_steepness"));
    }
    if config.state_norm_limit.is_nan() || config.state_norm_limit <= 0.0 {
        return Err(HlsError::InvalidParameter("state_norm_limit"));
    }
    if let HlsActivation::BoundedTanh { bound } = config.activation
        && !bound.is_finite()
    {
        return Err(HlsError::InvalidParameter("activation.bound"));
    }
    Ok(())
}

#[inline(always)]
fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> HlsConfig {
        HlsConfig {
            dim: 512,
            ..HlsConfig::default()
        }
    }

    #[test]
    fn rejects_invalid_timescale_ordering() {
        let mut config = test_config();
        config.tau_min = 1.0;
        config.tau_base = 0.1;
        assert_eq!(
            HolographicLiquidCell::try_new(config, 1).unwrap_err(),
            HlsError::InvalidTimescaleBounds
        );
    }

    #[test]
    fn transition_is_finite_and_bounded() {
        let mut cell = HolographicLiquidCell::try_new(test_config(), 42).unwrap();
        let input = ContinuousHV::new_random(512, 43);
        for _ in 0..1000 {
            cell.step(0.01, &input).unwrap();
        }
        assert!(cell.state().norm().is_finite());
        assert!(cell.state().norm() <= cell.config().state_norm_limit + 1e-5);
    }

    #[test]
    fn parameter_count_matches_snapshot() {
        let cell = HolographicLiquidCell::try_new(test_config(), 42).unwrap();
        assert_eq!(cell.parameter_count(), 6 * 512);
        assert_eq!(cell.parameters().scalar_count(), cell.parameter_count());
    }

    #[test]
    fn parameter_replacement_is_atomic_on_validation_failure() {
        let mut cell = HolographicLiquidCell::try_new(test_config(), 42).unwrap();
        let before = cell.parameters();
        let mut malformed = before.clone();
        malformed.input_weight.values.pop();
        assert!(matches!(
            cell.set_parameters(malformed),
            Err(HlsError::ParameterDimensionMismatch {
                field: "input_weight",
                ..
            })
        ));
        assert_eq!(cell.parameters(), before);
    }

    #[test]
    fn arbitrary_finite_parameter_values_preserve_binding_equivariance() {
        let mut cell = HolographicLiquidCell::try_new(test_config(), 42).unwrap();
        let mut parameters = cell.parameters();
        for (field_index, (_, field)) in parameter_fields(&parameters).into_iter().enumerate() {
            // Read-only pass establishes deterministic field lengths before mutation.
            assert_eq!(field.dim(), 512, "field {field_index}");
        }
        for (index, value) in parameters.recurrent_weight.values.iter_mut().enumerate() {
            *value = ((index % 17) as f32 - 8.0) * 0.11;
        }
        for (index, value) in parameters.input_weight.values.iter_mut().enumerate() {
            *value = ((index % 13) as f32 - 6.0) * 0.09;
        }
        for (index, value) in parameters.tau_state_weight.values.iter_mut().enumerate() {
            *value = ((index % 11) as f32 - 5.0) * 0.17;
        }
        for (index, value) in parameters.gate_state_weight.values.iter_mut().enumerate() {
            *value = ((index % 7) as f32 - 3.0) * 0.21;
        }
        for (index, value) in parameters.gate_input_weight.values.iter_mut().enumerate() {
            *value = ((index % 19) as f32 - 9.0) * 0.07;
        }
        for (index, value) in parameters.gate_bias.values.iter_mut().enumerate() {
            *value = ((index % 5) as f32 - 2.0) * 0.05;
        }
        cell.set_parameters(parameters).unwrap();
        cell.set_state(ContinuousHV::new_random(512, 100)).unwrap();
        let input = ContinuousHV::new_random(512, 101);
        let role = UnitaryRole::new(512, 102);
        let error = cell
            .binding_equivariance_error(&role, &input, 0.137)
            .unwrap();
        assert!(error <= 1e-6, "trained-parameter equivariance error={error}");
    }

    #[test]
    fn bounded_parameter_delta_clamps_candidate_without_partial_update() {
        let mut cell = HolographicLiquidCell::try_new(test_config(), 42).unwrap();
        let mut delta = HlsParameters::zeros(512);
        delta.recurrent_weight.values.fill(100.0);
        delta.gate_bias.values.fill(-100.0);
        cell.apply_parameter_delta(&delta, 1.0, 0.25).unwrap();
        let parameters = cell.parameters();
        assert!(parameters
            .recurrent_weight
            .values
            .iter()
            .all(|value| value.abs() <= 0.25));
        assert!(parameters
            .gate_bias
            .values
            .iter()
            .all(|value| value.abs() <= 0.25));
    }

    #[test]
    fn binding_equivariance_holds_from_nonzero_state() {
        let mut cell = HolographicLiquidCell::try_new(test_config(), 42).unwrap();
        cell.set_state(ContinuousHV::new_random(512, 100)).unwrap();
        let input = ContinuousHV::new_random(512, 101);
        let role = UnitaryRole::new(512, 102);
        let error = cell
            .binding_equivariance_error(&role, &input, 0.137)
            .unwrap();
        assert!(error <= 1e-6, "binding equivariance error={error}");
    }

    #[test]
    fn binding_equivariance_survives_large_dt() {
        let mut cell = HolographicLiquidCell::try_new(test_config(), 77).unwrap();
        cell.set_state(ContinuousHV::new_random(512, 78)).unwrap();
        let input = ContinuousHV::new_random(512, 79);
        let role = UnitaryRole::new(512, 80);
        let error = cell
            .binding_equivariance_error(&role, &input, 100.0)
            .unwrap();
        assert!(error <= 1e-6, "large-dt equivariance error={error}");
    }
}
