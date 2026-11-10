// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metric-aware sampled Lie-derivative checks for explicitly admitted linear actions.
//!
//! The caller supplies frozen sample states in the report's declared variable order.
//! For G = s L L^T, the evaluator whitens the covector and orbit tangent as
//! q = L^-1 grad(f), w = L^T A x, then measures (q_hat dot w_hat)^2.
//! This is a finite-sample diagnostic, not an exact identity or proof.

use crate::linear_group_action::AdmittedLinearGroupActionV1;
use serde::{Deserialize, Serialize};
use std::fmt;
use symthaea_core::hdc::conjecture_engine::Expr;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearActionNumericProfileV1 {
    pub sample_count: usize,
    pub maximum_samples: usize,
    pub maximum_variables: usize,
    pub minimum_informative_samples_per_generator: usize,
    pub absolute_finite_difference_step: f64,
    pub relative_finite_difference_step: f64,
    pub squared_cosine_tolerance: f64,
}

impl Default for LinearActionNumericProfileV1 {
    fn default() -> Self {
        Self {
            sample_count: 32,
            maximum_samples: 256,
            maximum_variables: 64,
            minimum_informative_samples_per_generator: 24,
            absolute_finite_difference_step: 1e-6,
            relative_finite_difference_step: 1e-6,
            squared_cosine_tolerance: 1e-6,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LinearActionNumericErrorV1 {
    InvalidProfile,
    SampleCountMismatch { expected: usize, actual: usize },
    SampleBudgetExceeded { actual: usize, maximum: usize },
    SampleDimensionMismatch { sample_index: usize, expected: usize, actual: usize },
    NonFiniteSample { sample_index: usize, coordinate_index: usize },
    NonFiniteExpressionEvaluation { sample_index: usize, variable_index: usize },
    NonFiniteGradient { sample_index: usize, variable_index: usize },
    NonFiniteOrbitTangent { generator_id: String, sample_index: usize },
    NonFiniteMetricWhitening { generator_id: String, sample_index: usize },
}

impl fmt::Display for LinearActionNumericErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for LinearActionNumericErrorV1 {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionGeneratorNumericStatisticsV1 {
    pub generator_id: String,
    pub informative_sample_count: usize,
    pub skipped_zero_gradient_count: usize,
    pub skipped_zero_orbit_tangent_count: usize,
    pub maximum_normalized_squared_residual: f64,
    pub mean_normalized_squared_residual: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionNumericReportV1 {
    pub action_id: String,
    pub coordinate_basis_id: String,
    pub expression_variables: Vec<String>,
    pub profile: LinearActionNumericProfileV1,
    /// Stable FNV-1a sample fingerprint; not a cryptographic integrity digest.
    pub sample_fingerprint_fnv1a64: u64,
    pub generator_statistics: Vec<LinearActionGeneratorNumericStatisticsV1>,
}

impl LinearActionNumericReportV1 {
    /// Finite-sample qualification only. Insufficient coverage fails closed.
    pub fn qualifies_under_profile(&self) -> bool {
        !self.generator_statistics.is_empty()
            && self.generator_statistics.iter().all(|statistics| {
                statistics.informative_sample_count
                    >= self.profile.minimum_informative_samples_per_generator
                    && statistics.maximum_normalized_squared_residual
                        <= self.profile.squared_cosine_tolerance
            })
    }
}

/// Check a scalar expression under every generator of an admitted action.
///
/// Samples are ordered as expression variables in first-seen AST order, followed
/// by declared action coordinates missing from the expression. Other expression
/// variables are fixed parameters. For covariance experiments, transform sample
/// states together with the expression, action and metric.
pub fn check_linear_action_numeric_invariance(
    expr: &Expr,
    action: &AdmittedLinearGroupActionV1,
    sample_states: &[Vec<f64>],
    profile: LinearActionNumericProfileV1,
) -> Result<LinearActionNumericReportV1, LinearActionNumericErrorV1> {
    if profile.sample_count == 0
        || profile.maximum_samples == 0
        || profile.maximum_variables == 0
        || profile.sample_count > profile.maximum_samples
        || profile.minimum_informative_samples_per_generator == 0
        || profile.minimum_informative_samples_per_generator > profile.sample_count
        || !profile.absolute_finite_difference_step.is_finite()
        || profile.absolute_finite_difference_step <= 0.0
        || !profile.relative_finite_difference_step.is_finite()
        || !(0.0..=1.0).contains(&profile.relative_finite_difference_step)
        || !profile.squared_cosine_tolerance.is_finite()
        || !(0.0..=1.0).contains(&profile.squared_cosine_tolerance)
    {
        return Err(LinearActionNumericErrorV1::InvalidProfile);
    }
    if sample_states.len() > profile.maximum_samples {
        return Err(LinearActionNumericErrorV1::SampleBudgetExceeded {
            actual: sample_states.len(), maximum: profile.maximum_samples,
        });
    }
    if sample_states.len() != profile.sample_count {
        return Err(LinearActionNumericErrorV1::SampleCountMismatch {
            expected: profile.sample_count, actual: sample_states.len(),
        });
    }

    let declaration = action.action();
    let mut variable_names = Vec::new();
    collect_variable_names(expr, &mut variable_names);
    for name in &declaration.coordinate_names {
        if !variable_names.contains(name) {
            variable_names.push(name.clone());
        }
    }
    if variable_names.len() > profile.maximum_variables {
        return Err(LinearActionNumericErrorV1::InvalidProfile);
    }
    for (sample_index, sample) in sample_states.iter().enumerate() {
        if sample.len() != variable_names.len() {
            return Err(LinearActionNumericErrorV1::SampleDimensionMismatch {
                sample_index, expected: variable_names.len(), actual: sample.len(),
            });
        }
        for (coordinate_index, value) in sample.iter().enumerate() {
            if !value.is_finite() {
                return Err(LinearActionNumericErrorV1::NonFiniteSample {
                    sample_index, coordinate_index,
                });
            }
        }
    }

    let coordinate_indices = declaration.coordinate_names.iter().map(|name| {
        variable_names.iter().position(|candidate| candidate == name)
            .expect("declared action coordinates were appended")
    }).collect::<Vec<_>>();
    let lower = action.scaled_metric_cholesky_lower();
    let mut generator_statistics = Vec::with_capacity(declaration.generators.len());

    for generator in &declaration.generators {
        let mut informative = 0usize;
        let mut skipped_gradient = 0usize;
        let mut skipped_tangent = 0usize;
        let mut maximum = 0.0_f64;
        let mut sum = 0.0_f64;

        for (sample_index, state) in sample_states.iter().enumerate() {
            let coordinate_gradient = central_difference_coordinate_gradient(
                expr, state, &variable_names, &coordinate_indices, profile, sample_index,
            )?;
            let mut orbit_tangent = vec![0.0_f64; coordinate_indices.len()];
            for (row_index, component) in orbit_tangent.iter_mut().enumerate() {
                for (column_index, &state_index) in coordinate_indices.iter().enumerate() {
                    *component += generator.matrix[row_index][column_index] * state[state_index];
                }
            }
            if orbit_tangent.iter().any(|value| !value.is_finite()) {
                return Err(LinearActionNumericErrorV1::NonFiniteOrbitTangent {
                    generator_id: generator.id.clone(), sample_index,
                });
            }

            let whitened_covector = forward_solve_lower(lower, &coordinate_gradient);
            let whitened_tangent = multiply_lower_transpose(lower, &orbit_tangent);
            if whitened_covector.iter().chain(whitened_tangent.iter()).any(|value| !value.is_finite()) {
                return Err(LinearActionNumericErrorV1::NonFiniteMetricWhitening {
                    generator_id: generator.id.clone(), sample_index,
                });
            }
            let gradient_norm = stable_norm(&whitened_covector);
            let tangent_norm = stable_norm(&whitened_tangent);
            if !gradient_norm.is_finite() || !tangent_norm.is_finite() {
                return Err(LinearActionNumericErrorV1::NonFiniteMetricWhitening {
                    generator_id: generator.id.clone(), sample_index,
                });
            }
            if gradient_norm == 0.0 {
                skipped_gradient = skipped_gradient.saturating_add(1);
                continue;
            }
            if tangent_norm == 0.0 {
                skipped_tangent = skipped_tangent.saturating_add(1);
                continue;
            }

            let cosine = whitened_covector.iter().zip(whitened_tangent.iter())
                .map(|(g, v)| (g / gradient_norm) * (v / tangent_norm))
                .sum::<f64>();
            if !cosine.is_finite() {
                return Err(LinearActionNumericErrorV1::NonFiniteMetricWhitening {
                    generator_id: generator.id.clone(), sample_index,
                });
            }
            let residual = (cosine * cosine).min(1.0);
            informative = informative.saturating_add(1);
            sum += residual;
            maximum = maximum.max(residual);
        }

        let mean = if informative == 0 { f64::INFINITY } else { sum / informative as f64 };
        generator_statistics.push(LinearActionGeneratorNumericStatisticsV1 {
            generator_id: generator.id.clone(),
            informative_sample_count: informative,
            skipped_zero_gradient_count: skipped_gradient,
            skipped_zero_orbit_tangent_count: skipped_tangent,
            maximum_normalized_squared_residual: if informative == 0 { f64::INFINITY } else { maximum },
            mean_normalized_squared_residual: mean,
        });
    }

    Ok(LinearActionNumericReportV1 {
        action_id: declaration.action_id.clone(),
        coordinate_basis_id: declaration.coordinate_basis_id.clone(),
        expression_variables: variable_names,
        profile,
        sample_fingerprint_fnv1a64: fingerprint_samples(sample_states),
        generator_statistics,
    })
}

fn collect_variable_names(expr: &Expr, output: &mut Vec<String>) {
    match expr {
        Expr::Var(name) => { if !output.contains(name) { output.push(name.clone()); } }
        Expr::Const(_) => {}
        Expr::BinOp(_, left, right) => {
            collect_variable_names(left, output);
            collect_variable_names(right, output);
        }
        Expr::Func(_, argument) => collect_variable_names(argument, output),
        Expr::Sum(body, variable) => {
            collect_variable_names(body, output);
            if !output.contains(variable) { output.push(variable.clone()); }
        }
    }
}

fn central_difference_coordinate_gradient(
    expr: &Expr,
    state: &[f64],
    variable_names: &[String],
    coordinate_indices: &[usize],
    profile: LinearActionNumericProfileV1,
    sample_index: usize,
) -> Result<Vec<f64>, LinearActionNumericErrorV1> {
    let mut gradient = Vec::with_capacity(coordinate_indices.len());
    for &variable_index in coordinate_indices {
        let step = profile.absolute_finite_difference_step
            + profile.relative_finite_difference_step * state[variable_index].abs();
        if !step.is_finite() || step <= 0.0 {
            return Err(LinearActionNumericErrorV1::NonFiniteGradient { sample_index, variable_index });
        }
        let mut plus = state.to_vec();
        let mut minus = state.to_vec();
        plus[variable_index] += step;
        minus[variable_index] -= step;
        let plus_value = evaluate_expression(expr, variable_names, &plus);
        let minus_value = evaluate_expression(expr, variable_names, &minus);
        let (Some(plus_value), Some(minus_value)) = (plus_value, minus_value) else {
            return Err(LinearActionNumericErrorV1::NonFiniteExpressionEvaluation { sample_index, variable_index });
        };
        let derivative = (plus_value - minus_value) / (2.0 * step);
        if !derivative.is_finite() {
            return Err(LinearActionNumericErrorV1::NonFiniteGradient { sample_index, variable_index });
        }
        gradient.push(derivative);
    }
    Ok(gradient)
}

fn evaluate_expression(expr: &Expr, names: &[String], state: &[f64]) -> Option<f64> {
    let bindings = names.iter().map(String::as_str).zip(state.iter().copied()).collect::<Vec<_>>();
    let value = expr.eval(&bindings);
    value.is_finite().then_some(value)
}

fn forward_solve_lower(lower: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let mut solution: Vec<f64> = Vec::with_capacity(rhs.len());
    for (row_index, row) in lower.iter().enumerate() {
        let correction = row.iter().take(row_index).zip(solution.iter())
            .map(|(matrix_value, solution_value)| *matrix_value * *solution_value)
            .sum::<f64>();
        solution.push((rhs[row_index] - correction) / row[row_index]);
    }
    solution
}

fn multiply_lower_transpose(lower: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(vector.len());
    for (column_index, _) in vector.iter().enumerate() {
        let value = lower.iter().zip(vector.iter())
            .map(|(row, component)| row[column_index] * component)
            .sum::<f64>();
        result.push(value);
    }
    result
}

fn stable_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |norm, value| norm.hypot(*value))
}

fn fingerprint_samples(samples: &[Vec<f64>]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for sample in samples {
        hash ^= sample.len() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        for value in sample {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
