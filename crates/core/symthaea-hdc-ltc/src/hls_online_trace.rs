// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact online forward sensitivities for the diagonal HLS recurrence.
//!
//! This is an RTRL-style eligibility trace specialized to the theorem-bearing
//! diagonal `HolographicLiquidCell`. Because coordinate `i` only depends on the
//! previous state at coordinate `i` (before optional global norm limiting), each
//! coordinate can propagate derivatives with respect to its own six parameters:
//!
//! `e_p(t+1) = (dF/dh) * e_p(t) + dF/dp`.
//!
//! The trace therefore stores `6D` scalars and advances in `O(D)` work for the
//! fixed six-field HLS parameterization, independent of sequence length.
//!
//! Exactness boundary: the global L2 state limiter couples coordinates when it
//! activates. `step_with_eligibility` detects that case before mutating the cell
//! and fails closed rather than silently degrading an exact trace into an
//! approximation.

use crate::config::fast_tanh;
use crate::continuous_hv::ContinuousHV;
use crate::holographic_liquid::{
    HlsActivation, HlsConfig, HlsError, HlsParameters, HolographicLiquidCell,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Per-parameter online derivative of the current hidden state.
///
/// For example `recurrent_weight[i]` stores
/// `d h_i(current) / d recurrent_weight_i`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HlsEligibilityTrace {
    pub recurrent_weight: ContinuousHV,
    pub input_weight: ContinuousHV,
    pub tau_state_weight: ContinuousHV,
    pub gate_state_weight: ContinuousHV,
    pub gate_input_weight: ContinuousHV,
    pub gate_bias: ContinuousHV,
}

impl HlsEligibilityTrace {
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

    pub fn dim(&self) -> usize {
        self.recurrent_weight.dim()
    }

    pub fn scalar_count(&self) -> usize {
        self.dim().saturating_mul(6)
    }

    pub fn reset(&mut self) {
        let dim = self.dim();
        *self = Self::zeros(dim);
    }

    /// Contract the current-state learning signal `dL/dh` with the eligibility
    /// traces to produce `dL/dtheta` for all six local parameter fields.
    pub fn parameter_gradient(
        &self,
        learning_signal: &ContinuousHV,
    ) -> Result<HlsParameters, HlsTraceError> {
        self.validate()?;
        if learning_signal.dim() != self.dim() {
            return Err(HlsTraceError::DimensionMismatch {
                expected: self.dim(),
                actual: learning_signal.dim(),
            });
        }
        if learning_signal
            .values
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(HlsTraceError::NonFiniteLearningSignal);
        }

        Ok(HlsParameters {
            recurrent_weight: contract(&self.recurrent_weight, learning_signal),
            input_weight: contract(&self.input_weight, learning_signal),
            tau_state_weight: contract(&self.tau_state_weight, learning_signal),
            gate_state_weight: contract(&self.gate_state_weight, learning_signal),
            gate_input_weight: contract(&self.gate_input_weight, learning_signal),
            gate_bias: contract(&self.gate_bias, learning_signal),
        })
    }

    fn validate(&self) -> Result<(), HlsTraceError> {
        let dim = self.dim();
        for (name, field) in trace_fields(self) {
            if field.dim() != dim {
                return Err(HlsTraceError::TraceDimensionMismatch {
                    field: name,
                    expected: dim,
                    actual: field.dim(),
                });
            }
            if field.values.iter().any(|value| !value.is_finite()) {
                return Err(HlsTraceError::NonFiniteTrace(name));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HlsTraceError {
    Cell(HlsError),
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    TraceDimensionMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteTrace(&'static str),
    NonFiniteLearningSignal,
    GlobalNormBoundWouldActivate {
        candidate_norm: f32,
        limit: f32,
    },
}

impl fmt::Display for HlsTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cell(error) => write!(f, "HLS trace cell error: {error}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "HLS trace dimension mismatch: expected {expected}, got {actual}")
            }
            Self::TraceDimensionMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "HLS trace field {field} dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteTrace(field) => {
                write!(f, "HLS trace field {field} contains a non-finite value")
            }
            Self::NonFiniteLearningSignal => {
                write!(f, "HLS trace learning signal contains a non-finite value")
            }
            Self::GlobalNormBoundWouldActivate {
                candidate_norm,
                limit,
            } => write!(
                f,
                "exact local HLS trace is invalid when global norm bound activates: candidate norm {candidate_norm}, limit {limit}"
            ),
        }
    }
}

impl std::error::Error for HlsTraceError {}

impl From<HlsError> for HlsTraceError {
    fn from(value: HlsError) -> Self {
        Self::Cell(value)
    }
}

/// Advance the HLS cell and its exact diagonal RTRL eligibility trace together.
///
/// The method computes the candidate transition and derivative recurrence first.
/// If the global L2 limiter would activate, it returns an error without mutating
/// either the live cell or trace because that limiter introduces cross-coordinate
/// Jacobian terms not represented by this local trace.
pub fn step_with_eligibility(
    cell: &mut HolographicLiquidCell,
    trace: &mut HlsEligibilityTrace,
    dt: f32,
    input: &ContinuousHV,
) -> Result<(), HlsTraceError> {
    let dim = cell.config().dim;
    trace.validate()?;
    if trace.dim() != dim {
        return Err(HlsTraceError::DimensionMismatch {
            expected: dim,
            actual: trace.dim(),
        });
    }
    if input.dim() != dim {
        return Err(HlsTraceError::DimensionMismatch {
            expected: dim,
            actual: input.dim(),
        });
    }
    if !dt.is_finite() {
        return Err(HlsError::NonFiniteDt.into());
    }
    let dt = dt.max(1e-9);

    let config = cell.config().clone();
    let parameters = cell.parameters();
    let state = cell.state().clone();
    let mut candidate_state = vec![0.0_f32; dim];
    let mut next_trace = trace.clone();

    for i in 0..dim {
        let local = local_transition_derivatives(
            &config,
            dt,
            state.values[i],
            input.values[i],
            parameters.recurrent_weight.values[i],
            parameters.input_weight.values[i],
            parameters.tau_state_weight.values[i],
            parameters.gate_state_weight.values[i],
            parameters.gate_input_weight.values[i],
            parameters.gate_bias.values[i],
        );

        candidate_state[i] = local.next_state;
        next_trace.recurrent_weight.values[i] = local.d_state_d_previous
            * trace.recurrent_weight.values[i]
            + local.d_state_d_recurrent_weight;
        next_trace.input_weight.values[i] = local.d_state_d_previous
            * trace.input_weight.values[i]
            + local.d_state_d_input_weight;
        next_trace.tau_state_weight.values[i] = local.d_state_d_previous
            * trace.tau_state_weight.values[i]
            + local.d_state_d_tau_state_weight;
        next_trace.gate_state_weight.values[i] = local.d_state_d_previous
            * trace.gate_state_weight.values[i]
            + local.d_state_d_gate_state_weight;
        next_trace.gate_input_weight.values[i] = local.d_state_d_previous
            * trace.gate_input_weight.values[i]
            + local.d_state_d_gate_input_weight;
        next_trace.gate_bias.values[i] = local.d_state_d_previous
            * trace.gate_bias.values[i]
            + local.d_state_d_gate_bias;
    }

    next_trace.validate()?;
    let candidate_norm = candidate_state
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let limit = config.state_norm_limit;
    if limit.is_finite() && candidate_norm > limit {
        return Err(HlsTraceError::GlobalNormBoundWouldActivate {
            candidate_norm,
            limit,
        });
    }

    // The live forward call uses the same recurrence. Since the limiter is known
    // not to activate, the local derivative factorization remains exact.
    cell.step(dt, input)?;
    *trace = next_trace;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct LocalDerivatives {
    next_state: f32,
    d_state_d_previous: f32,
    d_state_d_recurrent_weight: f32,
    d_state_d_input_weight: f32,
    d_state_d_tau_state_weight: f32,
    d_state_d_gate_state_weight: f32,
    d_state_d_gate_input_weight: f32,
    d_state_d_gate_bias: f32,
}

#[allow(clippy::too_many_arguments)]
fn local_transition_derivatives(
    config: &HlsConfig,
    dt: f32,
    h: f32,
    x: f32,
    recurrent_weight: f32,
    input_weight: f32,
    tau_state_weight: f32,
    gate_state_weight: f32,
    gate_input_weight: f32,
    gate_bias: f32,
) -> LocalDerivatives {
    let h_abs = h.abs();
    let x_abs = x.abs();
    let sign_h = if h > 0.0 {
        1.0
    } else if h < 0.0 {
        -1.0
    } else {
        0.0
    };

    let gate_linear = gate_state_weight * h_abs + gate_input_weight * x_abs + gate_bias;
    let gate_pre = gate_linear * config.gate_steepness;
    let gate = sigmoid(gate_pre);
    let gate_common = gate * (1.0 - gate) * config.gate_steepness;
    let d_gate_d_h = gate_common * gate_state_weight * sign_h;
    let d_gate_d_gate_state_weight = gate_common * h_abs;
    let d_gate_d_gate_input_weight = gate_common * x_abs;
    let d_gate_d_gate_bias = gate_common;

    let tau_control = tau_state_weight * h_abs + config.tau_input_coupling * x_abs;
    let (tau, d_ln_tau_d_control) = tau_and_log_derivative(config, tau_control);

    let unclipped_exponent = -dt / tau;
    let exponent = unclipped_exponent.max(-87.0);
    let decay = exponent.exp();
    let alpha = 1.0 - decay;
    let d_alpha_d_control = if unclipped_exponent > -87.0 {
        -decay * (dt / tau) * d_ln_tau_d_control
    } else {
        0.0
    };
    let d_alpha_d_h = d_alpha_d_control * tau_state_weight * sign_h;
    let d_alpha_d_tau_state_weight = d_alpha_d_control * h_abs;

    let equilibrium_pre = recurrent_weight * h + input_weight * x;
    let activation = activation_value(config.activation, equilibrium_pre);
    let activation_derivative = activation_derivative(config.activation, equilibrium_pre);
    let equilibrium = activation * gate;

    let d_equilibrium_d_h =
        activation_derivative * recurrent_weight * gate + activation * d_gate_d_h;
    let d_equilibrium_d_recurrent_weight = activation_derivative * h * gate;
    let d_equilibrium_d_input_weight = activation_derivative * x * gate;
    let d_equilibrium_d_gate_state_weight = activation * d_gate_d_gate_state_weight;
    let d_equilibrium_d_gate_input_weight = activation * d_gate_d_gate_input_weight;
    let d_equilibrium_d_gate_bias = activation * d_gate_d_gate_bias;

    let delta = equilibrium - h;
    let next_state = h + alpha * delta;
    let d_state_d_previous =
        1.0 - alpha + alpha * d_equilibrium_d_h + d_alpha_d_h * delta;

    LocalDerivatives {
        next_state,
        d_state_d_previous,
        d_state_d_recurrent_weight: alpha * d_equilibrium_d_recurrent_weight,
        d_state_d_input_weight: alpha * d_equilibrium_d_input_weight,
        d_state_d_tau_state_weight: d_alpha_d_tau_state_weight * delta,
        d_state_d_gate_state_weight: alpha * d_equilibrium_d_gate_state_weight,
        d_state_d_gate_input_weight: alpha * d_equilibrium_d_gate_input_weight,
        d_state_d_gate_bias: alpha * d_equilibrium_d_gate_bias,
    }
}

fn tau_and_log_derivative(config: &HlsConfig, control: f32) -> (f32, f32) {
    let ln_min = config.tau_min.ln();
    let ln_max = config.tau_max.ln();
    let span = ln_max - ln_min;
    let base_position = ((config.tau_base.ln() - ln_min) / span).clamp(1e-6, 1.0 - 1e-6);
    let base_logit = (base_position / (1.0 - base_position)).ln();
    let position = sigmoid(control + base_logit);
    let tau = (ln_min + position * span)
        .exp()
        .clamp(config.tau_min, config.tau_max);
    let d_ln_tau_d_control = span * position * (1.0 - position);
    (tau, d_ln_tau_d_control)
}

#[inline(always)]
fn activation_value(activation: HlsActivation, value: f32) -> f32 {
    match activation {
        HlsActivation::Tanh => fast_tanh(value),
        HlsActivation::Identity => value,
        HlsActivation::BoundedTanh { bound } => fast_tanh(value * bound),
    }
}

#[inline(always)]
fn activation_derivative(activation: HlsActivation, value: f32) -> f32 {
    match activation {
        HlsActivation::Tanh => fast_tanh_derivative(value),
        HlsActivation::Identity => 1.0,
        HlsActivation::BoundedTanh { bound } => bound * fast_tanh_derivative(value * bound),
    }
}

/// Derivative of the exact piecewise `fast_tanh` implementation used by HLS,
/// excluding non-differentiable branch/clamp boundaries where zero is selected.
#[inline(always)]
fn fast_tanh_derivative(value: f32) -> f32 {
    if value.abs() > 4.97 {
        return 0.0;
    }
    let x2 = value * value;
    let numerator = value * (27.0 + x2);
    let denominator = 27.0 + 9.0 * x2;
    let raw = numerator / denominator;
    if !(-1.0..1.0).contains(&raw) {
        return 0.0;
    }
    let numerator_prime = 27.0 + 3.0 * x2;
    let denominator_prime = 18.0 * value;
    (numerator_prime * denominator - numerator * denominator_prime)
        / (denominator * denominator)
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

fn contract(trace: &ContinuousHV, learning_signal: &ContinuousHV) -> ContinuousHV {
    ContinuousHV::from_values(
        trace
            .values
            .iter()
            .zip(learning_signal.values.iter())
            .map(|(eligibility, signal)| eligibility * signal)
            .collect(),
    )
}

fn trace_fields(trace: &HlsEligibilityTrace) -> [(&'static str, &ContinuousHV); 6] {
    [
        ("recurrent_weight", &trace.recurrent_weight),
        ("input_weight", &trace.input_weight),
        ("tau_state_weight", &trace.tau_state_weight),
        ("gate_state_weight", &trace.gate_state_weight),
        ("gate_input_weight", &trace.gate_input_weight),
        ("gate_bias", &trace.gate_bias),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(dim: usize) -> HlsConfig {
        HlsConfig {
            dim,
            state_norm_limit: f32::INFINITY,
            activation: HlsActivation::Tanh,
            ..HlsConfig::default()
        }
    }

    fn scaled_random(dim: usize, seed: u64, scale: f32) -> ContinuousHV {
        ContinuousHV::new_random(dim, seed).scale(scale)
    }

    #[test]
    fn trace_matches_finite_difference_through_irregular_sequence() {
        let dim = 12;
        let mut initial = HolographicLiquidCell::try_new(config(dim), 42).unwrap();
        initial.set_state(scaled_random(dim, 43, 0.2)).unwrap();
        let inputs = [
            scaled_random(dim, 50, 0.2),
            scaled_random(dim, 51, 0.2),
            scaled_random(dim, 52, 0.2),
            scaled_random(dim, 53, 0.2),
        ];
        let dts = [0.003_f32, 0.071, 0.4, 1.3];

        let mut traced = initial.clone();
        let mut trace = HlsEligibilityTrace::zeros(dim);
        for (&dt, input) in dts.iter().zip(inputs.iter()) {
            step_with_eligibility(&mut traced, &mut trace, dt, input).unwrap();
        }
        let learning_signal = scaled_random(dim, 60, 0.3);
        let analytic = trace.parameter_gradient(&learning_signal).unwrap();

        let epsilon = 1e-3_f32;
        let coordinate = 4_usize;
        let field_names = [
            "recurrent_weight",
            "input_weight",
            "tau_state_weight",
            "gate_state_weight",
            "gate_input_weight",
            "gate_bias",
        ];

        for field in field_names {
            let plus = sequence_loss_with_perturbation(
                &initial,
                &inputs,
                &dts,
                &learning_signal,
                field,
                coordinate,
                epsilon,
            );
            let minus = sequence_loss_with_perturbation(
                &initial,
                &inputs,
                &dts,
                &learning_signal,
                field,
                coordinate,
                -epsilon,
            );
            let numerical = (plus - minus) / (2.0 * epsilon);
            let predicted = parameter_field_value(&analytic, field, coordinate);
            let tolerance = 5e-3_f32.max(0.03 * numerical.abs());
            assert!(
                (predicted - numerical).abs() <= tolerance,
                "field={field} analytic={predicted} numerical={numerical} tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn active_global_norm_bound_fails_closed() {
        let mut cell = HolographicLiquidCell::try_new(
            HlsConfig {
                dim: 8,
                state_norm_limit: 0.1,
                ..HlsConfig::default()
            },
            7,
        )
        .unwrap();
        cell.set_state(ContinuousHV::from_values(vec![0.5; 8]))
            .unwrap();
        let before = cell.state().clone();
        let input = ContinuousHV::new(8);
        let mut trace = HlsEligibilityTrace::zeros(8);
        assert!(matches!(
            step_with_eligibility(&mut cell, &mut trace, 1e-6, &input),
            Err(HlsTraceError::GlobalNormBoundWouldActivate { .. })
        ));
        assert_eq!(cell.state(), &before);
        assert_eq!(trace, HlsEligibilityTrace::zeros(8));
    }

    #[test]
    fn parameter_gradient_is_coordinate_local_contraction() {
        let mut trace = HlsEligibilityTrace::zeros(4);
        trace.recurrent_weight.values = vec![1.0, 2.0, 3.0, 4.0];
        let signal = ContinuousHV::from_values(vec![0.5, -1.0, 2.0, 0.25]);
        let gradient = trace.parameter_gradient(&signal).unwrap();
        assert_eq!(
            gradient.recurrent_weight.values,
            vec![0.5, -2.0, 6.0, 1.0]
        );
    }

    fn sequence_loss_with_perturbation(
        initial: &HolographicLiquidCell,
        inputs: &[ContinuousHV],
        dts: &[f32],
        learning_signal: &ContinuousHV,
        field: &str,
        coordinate: usize,
        perturbation: f32,
    ) -> f32 {
        let mut cell = initial.clone();
        let mut parameters = cell.parameters();
        *parameter_field_value_mut(&mut parameters, field, coordinate) += perturbation;
        cell.set_parameters(parameters).unwrap();
        for (&dt, input) in dts.iter().zip(inputs.iter()) {
            cell.step(dt, input).unwrap();
        }
        cell.state().dot(learning_signal)
    }

    fn parameter_field_value(parameters: &HlsParameters, field: &str, coordinate: usize) -> f32 {
        match field {
            "recurrent_weight" => parameters.recurrent_weight.values[coordinate],
            "input_weight" => parameters.input_weight.values[coordinate],
            "tau_state_weight" => parameters.tau_state_weight.values[coordinate],
            "gate_state_weight" => parameters.gate_state_weight.values[coordinate],
            "gate_input_weight" => parameters.gate_input_weight.values[coordinate],
            "gate_bias" => parameters.gate_bias.values[coordinate],
            _ => unreachable!(),
        }
    }

    fn parameter_field_value_mut<'a>(
        parameters: &'a mut HlsParameters,
        field: &str,
        coordinate: usize,
    ) -> &'a mut f32 {
        match field {
            "recurrent_weight" => &mut parameters.recurrent_weight.values[coordinate],
            "input_weight" => &mut parameters.input_weight.values[coordinate],
            "tau_state_weight" => &mut parameters.tau_state_weight.values[coordinate],
            "gate_state_weight" => &mut parameters.gate_state_weight.values[coordinate],
            "gate_input_weight" => &mut parameters.gate_input_weight.values[coordinate],
            "gate_bias" => &mut parameters.gate_bias.values[coordinate],
            _ => unreachable!(),
        }
    }
}
