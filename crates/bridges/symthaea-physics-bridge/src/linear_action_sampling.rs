// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded, metric-aware sampled invariance checks for declared linear actions.
//!
//! Sample states and finite-difference direction bases are explicit inputs. This
//! makes coordinate-covariance tests replay the same geometric experiment rather
//! than independently sampling unrelated coordinate representations.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use symthaea_core::hdc::conjecture_engine::Expr;

use crate::linear_group_action::AdmittedLinearGroupActionV1;

/// One frozen evaluation state and a full basis of finite-difference directions.
///
/// Each direction is a vector in the action's declared coordinate representation.
/// The vectors are columns of the direction matrix. Under x' = D x, callers
/// transform both state and every direction by D. Expression variables outside
/// the action coordinates remain fixed parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionSampleV1 {
    /// Stable identity used to pair the same geometric sample across representations.
    pub sample_id: String,
    /// Values for expression variables in first-seen order, followed by any
    /// action coordinates absent from the expression.
    pub state: Vec<f64>,
    /// Full finite-difference direction basis; one vector per column.
    pub direction_basis: Vec<Vec<f64>>,
}

/// Numerical policy for sampled linear-action invariance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearActionSamplingProfileV1 {
    /// Central-difference displacement along each supplied direction.
    pub finite_difference_step: f64,
    /// Maximum squared metric cosine between df and the group-orbit tangent.
    pub squared_cosine_tolerance: f64,
    /// Maximum number of frozen sample states in one evaluation.
    pub maximum_samples: usize,
    /// Minimum informative samples required for every generator.
    pub minimum_informative_samples: usize,
    /// Maximum allowed spectral condition number of each direction basis.
    pub maximum_direction_basis_condition: f64,
}

/// Hard ceiling independent of caller configuration. This prevents a profile
/// from accidentally removing the sampling resource bound entirely.
const HARD_MAX_LINEAR_ACTION_SAMPLES: usize = 4096;

impl Default for LinearActionSamplingProfileV1 {
    fn default() -> Self {
        Self {
            finite_difference_step: 1e-5,
            squared_cosine_tolerance: 1e-6,
            maximum_samples: 256,
            minimum_informative_samples: 8,
            maximum_direction_basis_condition: 1e8,
        }
    }
}

/// A per-generator summary of the sampled metric directional residual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionGeneratorSampleStatisticsV1 {
    pub generator_id: String,
    pub checked_sample_count: usize,
    pub informative_sample_count: usize,
    pub skipped_sample_count: usize,
    pub mean_normalized_squared_residual: f64,
    pub maximum_normalized_squared_residual: f64,
    pub within_tolerance: bool,
}

/// Evidence report for a sampled scalar-expression invariance check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionSampleConditionDiagnosticV1 {
    pub sample_id: String,
    pub direction_basis_condition_number: f64,
}

/// Evidence report for a sampled scalar-expression invariance check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionSampledInvarianceReportV1 {
    pub action_id: String,
    pub coordinate_basis_id: String,
    pub profile: LinearActionSamplingProfileV1,
    pub sample_ids: Vec<String>,
    pub direction_basis_diagnostics: Vec<LinearActionSampleConditionDiagnosticV1>,
    pub generator_statistics: Vec<LinearActionGeneratorSampleStatisticsV1>,
}

impl LinearActionSampledInvarianceReportV1 {
    /// True only when every generator passed and each had enough informative samples.
    pub fn invariant_within_tolerance(&self) -> bool {
        !self.generator_statistics.is_empty()
            && self
                .generator_statistics
                .iter()
                .all(|statistics| statistics.within_tolerance)
    }
}

/// Fail-closed errors for sampled linear-action evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearActionSamplingErrorV1 {
    InvalidProfile,
    NoSamples,
    SampleBudgetExceeded { actual: usize, maximum: usize },
    EmptySampleId { index: usize },
    DuplicateSampleId { sample_id: String },
    StateDimensionMismatch {
        sample_id: String,
        expected: usize,
        actual: usize,
    },
    NonFiniteState { sample_id: String, index: usize },
    DirectionBasisDimensionMismatch {
        sample_id: String,
        expected: usize,
        actual: usize,
    },
    DirectionVectorDimensionMismatch {
        sample_id: String,
        direction_index: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteDirection {
        sample_id: String,
        direction_index: usize,
        coordinate_index: usize,
    },
    ZeroDirectionBasis { sample_id: String },
    DirectionBasisIllConditioned {
        sample_id: String,
        condition_number: f64,
    },
    NonFiniteExpressionEvaluation { sample_id: String },
    GradientReconstructionFailed { sample_id: String },
    NonFiniteOrbitTangent {
        sample_id: String,
        generator_id: String,
    },
    NonFiniteMetricWhitening {
        sample_id: String,
        generator_id: String,
    },
    InsufficientInformativeSamples {
        generator_id: String,
        found: usize,
        required: usize,
    },
}

/// Evaluate the scalar expression using the declared variable order.
fn evaluate_expression(expr: &Expr, variable_names: &[String], state: &[f64]) -> f64 {
    let bindings: Vec<(&str, f64)> = variable_names
        .iter()
        .map(String::as_str)
        .zip(state.iter().copied())
        .collect();
    expr.eval(&bindings)
}

/// Collect variables in deterministic first-seen order.
fn collect_variable_names(expr: &Expr, output: &mut Vec<String>) {
    use symthaea_core::hdc::conjecture_engine::Expr;

    match expr {
        Expr::Var(name) => {
            if !output.contains(name) {
                output.push(name.clone());
            }
        }
        Expr::Const(_) => {}
        Expr::BinOp(_, left, right) => {
            collect_variable_names(left, output);
            collect_variable_names(right, output);
        }
        Expr::Func(_, argument) => collect_variable_names(argument, output),
        Expr::Sum(body, variable) => {
            collect_variable_names(body, output);
            if !output.contains(variable) {
                output.push(variable.clone());
            }
        }
    }
}

/// Reconstruct the action-coordinate gradient from directional finite differences.
///
/// If V contains the supplied direction vectors as columns, then the observed
/// directional derivatives satisfy q = V^T p. We solve this system by SVD after
/// scale-normalizing V, and reject bases above the declared condition budget.
fn finite_difference_gradient(
    expr: &Expr,
    variable_names: &[String],
    action_coordinate_indices: &[usize],
    sample: &LinearActionSampleV1,
    profile: LinearActionSamplingProfileV1,
) -> Result<(Vec<f64>, f64), LinearActionSamplingErrorV1> {
    let dimension = action_coordinate_indices.len();
    let direction_scale = sample
        .direction_basis
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if !direction_scale.is_finite() || direction_scale == 0.0 {
        return Err(LinearActionSamplingErrorV1::ZeroDirectionBasis {
            sample_id: sample.sample_id.clone(),
        });
    }

    let direction_matrix = DMatrix::from_fn(dimension, dimension, |row, column| {
        sample.direction_basis[column][row] / direction_scale
    });
    let transpose = direction_matrix.transpose();
    let decomposition = transpose.svd(true, true);
    let maximum_singular_value = decomposition
        .singular_values
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let minimum_singular_value = decomposition
        .singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let condition_number = maximum_singular_value / minimum_singular_value;
    if !condition_number.is_finite()
        || minimum_singular_value <= 0.0
        || condition_number > profile.maximum_direction_basis_condition
    {
        return Err(LinearActionSamplingErrorV1::DirectionBasisIllConditioned {
            sample_id: sample.sample_id.clone(),
            condition_number,
        });
    }

    let mut observed = Vec::with_capacity(dimension);
    for direction in &sample.direction_basis {
        let mut plus = sample.state.clone();
        let mut minus = sample.state.clone();
        for (action_index, &variable_index) in action_coordinate_indices.iter().enumerate() {
            let displacement = profile.finite_difference_step * direction[action_index];
            plus[variable_index] += displacement;
            minus[variable_index] -= displacement;
        }
        let plus_value = evaluate_expression(expr, variable_names, &plus);
        let minus_value = evaluate_expression(expr, variable_names, &minus);
        if !plus_value.is_finite() || !minus_value.is_finite() {
            return Err(LinearActionSamplingErrorV1::NonFiniteExpressionEvaluation {
                sample_id: sample.sample_id.clone(),
            });
        }
        let directional_derivative =
            (plus_value - minus_value) / (2.0 * profile.finite_difference_step);
        if !directional_derivative.is_finite() {
            return Err(LinearActionSamplingErrorV1::NonFiniteExpressionEvaluation {
                sample_id: sample.sample_id.clone(),
            });
        }
        observed.push(directional_derivative / direction_scale);
    }

    let solution = decomposition
        .solve(&DVector::from_vec(observed), f64::EPSILON * (dimension as f64))
        .map_err(|_| LinearActionSamplingErrorV1::GradientReconstructionFailed {
            sample_id: sample.sample_id.clone(),
        })?;
    let gradient: Vec<f64> = solution.iter().copied().collect();
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(LinearActionSamplingErrorV1::GradientReconstructionFailed {
            sample_id: sample.sample_id.clone(),
        });
    }
    Ok((gradient, condition_number))
}

/// Forward solve L y = p for lower-triangular L.
fn solve_lower_triangular(lower: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let dimension = rhs.len();
    let mut solution = Vec::with_capacity(dimension);
    for (row_index, rhs_value) in rhs.iter().enumerate() {
        let correction = solution
            .iter()
            .zip(lower[row_index].iter())
            .map(|(value, coefficient)| *value * *coefficient)
            .sum::<f64>();
        let pivot = lower[row_index][row_index];
        let value = (*rhs_value - correction) / pivot;
        if !value.is_finite() || pivot <= 0.0 {
            return None;
        }
        solution.push(value);
    }
    Some(solution)
}

/// Compute L^T v for lower-triangular L.
fn multiply_lower_transpose(lower: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let dimension = vector.len();
    let mut result = vec![0.0_f64; dimension];
    for (column, output) in result.iter_mut().enumerate() {
        *output = lower
            .iter()
            .zip(vector.iter())
            .map(|(row, value)| row[column] * *value)
            .sum();
    }
    result
}

/// Stable Euclidean norm using hypot accumulation.
fn stable_norm(vector: &[f64]) -> f64 {
    vector
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value))
}

/// Compute the squared metric cosine between covector p and tangent vector v.
///
/// With G = s L L^T, the dual and tangent whitenings are L^-1 p and L^T v.
/// Their ordinary Euclidean cosine equals the metric/dual-metric cosine, and
/// the common positive metric scale cancels.
fn metric_squared_cosine(
    gradient: &[f64],
    tangent: &[f64],
    lower: &[Vec<f64>],
) -> Option<f64> {
    let whitened_gradient = solve_lower_triangular(lower, gradient)?;
    let whitened_tangent = multiply_lower_transpose(lower, tangent);
    if whitened_tangent.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let gradient_norm = stable_norm(&whitened_gradient);
    let tangent_norm = stable_norm(&whitened_tangent);
    if !gradient_norm.is_finite() || !tangent_norm.is_finite() {
        return None;
    }
    if gradient_norm == 0.0 || tangent_norm == 0.0 {
        return Some(f64::NAN);
    }
    let cosine = whitened_gradient
        .iter()
        .zip(whitened_tangent.iter())
        .map(|(left, right)| (left / gradient_norm) * (right / tangent_norm))
        .sum::<f64>();
    if !cosine.is_finite() {
        return None;
    }
    Some((cosine * cosine).clamp(0.0, 1.0))
}

/// Evaluate an already admitted linear action on explicit, paired sample states.
///
/// The squared metric cosine is computed from the finite-difference covector and
/// the generator orbit tangent. Samples with zero gradient or zero orbit tangent
/// are skipped as uninformative. Every generator must have at least the declared
/// minimum informative sample count; otherwise the result fails closed.
pub fn evaluate_linear_action_samples(
    expr: &Expr,
    action: &AdmittedLinearGroupActionV1,
    samples: &[LinearActionSampleV1],
    profile: LinearActionSamplingProfileV1,
) -> Result<LinearActionSampledInvarianceReportV1, LinearActionSamplingErrorV1> {
    if !profile.finite_difference_step.is_finite()
        || profile.finite_difference_step <= 0.0
        || !profile.squared_cosine_tolerance.is_finite()
        || !(0.0..=1.0).contains(&profile.squared_cosine_tolerance)
        || profile.maximum_samples == 0
        || profile.maximum_samples > HARD_MAX_LINEAR_ACTION_SAMPLES
        || profile.minimum_informative_samples == 0
        || profile.minimum_informative_samples > profile.maximum_samples
        || !profile.maximum_direction_basis_condition.is_finite()
        || profile.maximum_direction_basis_condition < 1.0
    {
        return Err(LinearActionSamplingErrorV1::InvalidProfile);
    }
    if samples.is_empty() {
        return Err(LinearActionSamplingErrorV1::NoSamples);
    }
    if samples.len() > profile.maximum_samples {
        return Err(LinearActionSamplingErrorV1::SampleBudgetExceeded {
            actual: samples.len(),
            maximum: profile.maximum_samples,
        });
    }

    let declaration = action.action();
    let dimension = declaration.coordinate_names.len();
    let mut variable_names = Vec::new();
    collect_variable_names(expr, &mut variable_names);
    for coordinate_name in &declaration.coordinate_names {
        if !variable_names.contains(coordinate_name) {
            variable_names.push(coordinate_name.clone());
        }
    }
    let mut action_coordinate_indices = Vec::with_capacity(dimension);
    for coordinate_name in &declaration.coordinate_names {
        action_coordinate_indices.push(
            variable_names
                .iter()
                .position(|name| name == coordinate_name)
                .expect("declared coordinate was appended to variable names"),
        );
    }

    let mut sample_ids = BTreeSet::new();
    for (sample_index, sample) in samples.iter().enumerate() {
        if sample.sample_id.trim().is_empty() {
            return Err(LinearActionSamplingErrorV1::EmptySampleId { index: sample_index });
        }
        if !sample_ids.insert(sample.sample_id.as_str()) {
            return Err(LinearActionSamplingErrorV1::DuplicateSampleId {
                sample_id: sample.sample_id.clone(),
            });
        }
        if sample.state.len() != variable_names.len() {
            return Err(LinearActionSamplingErrorV1::StateDimensionMismatch {
                sample_id: sample.sample_id.clone(),
                expected: variable_names.len(),
                actual: sample.state.len(),
            });
        }
        if let Some(index) = sample.state.iter().position(|value| !value.is_finite()) {
            return Err(LinearActionSamplingErrorV1::NonFiniteState {
                sample_id: sample.sample_id.clone(),
                index,
            });
        }
        if sample.direction_basis.len() != dimension {
            return Err(LinearActionSamplingErrorV1::DirectionBasisDimensionMismatch {
                sample_id: sample.sample_id.clone(),
                expected: dimension,
                actual: sample.direction_basis.len(),
            });
        }
        for (direction_index, direction) in sample.direction_basis.iter().enumerate() {
            if direction.len() != dimension {
                return Err(LinearActionSamplingErrorV1::DirectionVectorDimensionMismatch {
                    sample_id: sample.sample_id.clone(),
                    direction_index,
                    expected: dimension,
                    actual: direction.len(),
                });
            }
            if let Some(coordinate_index) =
                direction.iter().position(|value| !value.is_finite())
            {
                return Err(LinearActionSamplingErrorV1::NonFiniteDirection {
                    sample_id: sample.sample_id.clone(),
                    direction_index,
                    coordinate_index,
                });
            }
        }
    }

    // Reconstruct each gradient once and retain the condition diagnostics.
    // Multiple generators reuse the same paired function evaluations.
    let mut sample_gradients = Vec::with_capacity(samples.len());
    let mut direction_basis_diagnostics = Vec::with_capacity(samples.len());
    for sample in samples {
        let (gradient, condition_number) = finite_difference_gradient(
            expr,
            &variable_names,
            &action_coordinate_indices,
            sample,
            profile,
        )?;
        sample_gradients.push(gradient);
        direction_basis_diagnostics.push(LinearActionSampleConditionDiagnosticV1 {
            sample_id: sample.sample_id.clone(),
            direction_basis_condition_number: condition_number,
        });
    }

    let mut generator_statistics = Vec::with_capacity(declaration.generators.len());
    for generator in &declaration.generators {
        let mut informative_sample_count = 0usize;
        let mut skipped_sample_count = 0usize;
        let mut residual_sum = 0.0_f64;
        let mut maximum_residual = 0.0_f64;

        for (sample, gradient) in samples.iter().zip(sample_gradients.iter()) {
            let mut tangent = vec![0.0_f64; dimension];
            for (tangent_component, generator_row) in
                tangent.iter_mut().zip(generator.matrix.iter())
            {
                *tangent_component = generator_row
                    .iter()
                    .zip(action_coordinate_indices.iter())
                    .map(|(coefficient, &state_index)| coefficient * sample.state[state_index])
                    .sum();
            }
            if tangent.iter().any(|value| !value.is_finite()) {
                return Err(LinearActionSamplingErrorV1::NonFiniteOrbitTangent {
                    sample_id: sample.sample_id.clone(),
                    generator_id: generator.id.clone(),
                });
            }

            let residual = metric_squared_cosine(
                gradient,
                &tangent,
                action.scaled_metric_cholesky_lower(),
            )
            .ok_or_else(|| LinearActionSamplingErrorV1::NonFiniteMetricWhitening {
                sample_id: sample.sample_id.clone(),
                generator_id: generator.id.clone(),
            })?;
            if residual.is_nan() {
                skipped_sample_count = skipped_sample_count.saturating_add(1);
                continue;
            }
            informative_sample_count = informative_sample_count.saturating_add(1);
            residual_sum += residual;
            maximum_residual = maximum_residual.max(residual);
        }

        if informative_sample_count < profile.minimum_informative_samples {
            return Err(LinearActionSamplingErrorV1::InsufficientInformativeSamples {
                generator_id: generator.id.clone(),
                found: informative_sample_count,
                required: profile.minimum_informative_samples,
            });
        }

        generator_statistics.push(LinearActionGeneratorSampleStatisticsV1 {
            generator_id: generator.id.clone(),
            checked_sample_count: samples.len(),
            informative_sample_count,
            skipped_sample_count,
            mean_normalized_squared_residual: residual_sum / informative_sample_count as f64,
            maximum_normalized_squared_residual: maximum_residual,
            within_tolerance: maximum_residual <= profile.squared_cosine_tolerance,
        });
    }

    Ok(LinearActionSampledInvarianceReportV1 {
        action_id: declaration.action_id.clone(),
        coordinate_basis_id: declaration.coordinate_basis_id.clone(),
        profile,
        sample_ids: samples.iter().map(|sample| sample.sample_id.clone()).collect(),
        direction_basis_diagnostics,
        generator_statistics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_group_action::{
        LinearGeneratorV1, LinearGroupActionAdmissionProfileV1, LinearGroupActionV1,
    };
    use symthaea_core::hdc::conjecture_engine::BinOp;

    fn add(left: Expr, right: Expr) -> Expr {
        Expr::BinOp(BinOp::Add, Box::new(left), Box::new(right))
    }

    fn sub(left: Expr, right: Expr) -> Expr {
        Expr::BinOp(BinOp::Sub, Box::new(left), Box::new(right))
    }

    fn mul(left: Expr, right: Expr) -> Expr {
        Expr::BinOp(BinOp::Mul, Box::new(left), Box::new(right))
    }

    fn square(name: &str) -> Expr {
        Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var(name.to_string())),
            Box::new(Expr::Const(2.0)),
        )
    }

    fn canonical_action() -> AdmittedLinearGroupActionV1 {
        LinearGroupActionV1 {
            action_id: "sampled-so2".to_string(),
            coordinate_basis_id: "xy".to_string(),
            coordinate_names: vec!["x".to_string(), "y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generators: vec![LinearGeneratorV1 {
                id: "J".to_string(),
                matrix: vec![vec![0.0, -1.0], vec![1.0, 0.0]],
            }],
        }
        .try_admit(LinearGroupActionAdmissionProfileV1::default())
        .unwrap()
    }

    fn samples(
        states: &[[f64; 2]],
        direction_basis: [[f64; 2]; 2],
    ) -> Vec<LinearActionSampleV1> {
        states
            .iter()
            .enumerate()
            .map(|(index, state)| LinearActionSampleV1 {
                sample_id: format!("paired-{index}"),
                state: state.to_vec(),
                direction_basis: direction_basis.map(|direction| direction.to_vec()).to_vec(),
            })
            .collect()
    }

    fn sampling_profile() -> LinearActionSamplingProfileV1 {
        LinearActionSamplingProfileV1 {
            finite_difference_step: 1e-5,
            squared_cosine_tolerance: 1e-8,
            maximum_samples: 32,
            minimum_informative_samples: 4,
            maximum_direction_basis_condition: 1e6,
        }
    }

    #[test]
    fn sampled_linear_action_accepts_radial_expression() {
        let states = [
            [0.7, 1.1],
            [1.2, 0.8],
            [1.7, 2.1],
            [2.3, 1.4],
            [0.9, 2.5],
            [2.6, 2.1],
        ];
        let report = evaluate_linear_action_samples(
            &add(square("x"), square("y")),
            &canonical_action(),
            &samples(&states, [[1.0, 0.0], [0.0, 1.0]]),
            sampling_profile(),
        )
        .unwrap();
        assert!(report.invariant_within_tolerance());
        assert_eq!(report.generator_statistics[0].informative_sample_count, states.len());
        assert!(
            report.generator_statistics[0].maximum_normalized_squared_residual
                <= sampling_profile().squared_cosine_tolerance
        );
        assert_eq!(report.direction_basis_diagnostics.len(), states.len());
        assert!(report.direction_basis_diagnostics.iter().all(|diagnostic| {
            diagnostic.direction_basis_condition_number >= 1.0
        }));
    }

    #[test]
    fn sampled_linear_action_rejects_anisotropic_expression() {
        let states = [
            [0.7, 1.1],
            [1.2, 0.8],
            [1.7, 2.1],
            [2.3, 1.4],
            [0.9, 2.5],
            [2.6, 2.1],
        ];
        let expression = add(square("x"), mul(Expr::Const(2.0), square("y")));
        let report = evaluate_linear_action_samples(
            &expression,
            &canonical_action(),
            &samples(&states, [[1.0, 0.0], [0.0, 1.0]]),
            sampling_profile(),
        )
        .unwrap();
        assert!(!report.invariant_within_tolerance());
        assert!(report.generator_statistics[0].maximum_normalized_squared_residual > 0.01);
    }

    #[test]
    fn sampled_residual_is_covariant_under_paired_nonorthogonal_coordinate_change() {
        // D = [[1, 1], [0, 1]]. States, action, metric and finite-difference
        // direction vectors are all transformed together.
        let transform = nalgebra::Matrix2::new(1.0, 1.0, 0.0, 1.0);
        let inverse = transform.try_inverse().unwrap();
        let canonical_states = [
            nalgebra::Vector2::new(0.7, 1.1),
            nalgebra::Vector2::new(1.2, 0.8),
            nalgebra::Vector2::new(1.7, 2.1),
            nalgebra::Vector2::new(2.3, 1.4),
            nalgebra::Vector2::new(0.9, 2.5),
            nalgebra::Vector2::new(2.6, 2.1),
        ];
        let expression = add(square("x"), mul(Expr::Const(1.2), square("y")));
        let canonical_samples = samples(
            &canonical_states.map(|state| [state[0], state[1]]),
            [[1.0, 0.0], [0.0, 1.0]],
        );
        let canonical_report = evaluate_linear_action_samples(
            &expression,
            &canonical_action(),
            &canonical_samples,
            sampling_profile(),
        )
        .unwrap();

        let transformed_metric = inverse.transpose() * nalgebra::Matrix2::identity() * inverse;
        let transformed_generator =
            transform * nalgebra::Matrix2::new(0.0, -1.0, 1.0, 0.0) * inverse;
        let transformed_action = LinearGroupActionV1 {
            action_id: "sampled-so2-sheared".to_string(),
            coordinate_basis_id: "XY".to_string(),
            coordinate_names: vec!["X".to_string(), "Y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: transformed_metric
                .row_iter()
                .map(|row| row.iter().copied().collect())
                .collect(),
            generators: vec![LinearGeneratorV1 {
                id: "J-sheared".to_string(),
                matrix: transformed_generator
                    .row_iter()
                    .map(|row| row.iter().copied().collect())
                    .collect(),
            }],
        }
        .try_admit(LinearGroupActionAdmissionProfileV1::default())
        .unwrap();
        let transformed_expression = sub(
            add(square("X"), mul(Expr::Const(2.2), square("Y"))),
            mul(Expr::Const(2.0), mul(Expr::Var("X".to_string()), Expr::Var("Y".to_string()))),
        );
        let transformed_directions = transform;
        let transformed_samples: Vec<LinearActionSampleV1> = canonical_states
            .iter()
            .enumerate()
            .map(|(index, state)| {
                let transformed_state = transform * state;
                LinearActionSampleV1 {
                    sample_id: format!("paired-{index}"),
                    state: vec![transformed_state[0], transformed_state[1]],
                    direction_basis: (0..2)
                        .map(|column| {
                            vec![
                                transformed_directions[(0, column)],
                                transformed_directions[(1, column)],
                            ]
                        })
                        .collect(),
                }
            })
            .collect();
        let transformed_report = evaluate_linear_action_samples(
            &transformed_expression,
            &transformed_action,
            &transformed_samples,
            sampling_profile(),
        )
        .unwrap();
        let canonical_residual =
            canonical_report.generator_statistics[0].maximum_normalized_squared_residual;
        let transformed_residual =
            transformed_report.generator_statistics[0].maximum_normalized_squared_residual;
        assert!((canonical_residual - transformed_residual).abs() < 2.0e-9);
        assert_eq!(
            canonical_report.invariant_within_tolerance(),
            transformed_report.invariant_within_tolerance()
        );
    }

    #[test]
    fn sampled_linear_action_rejects_ill_conditioned_direction_basis() {
        let states = [[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]];
        let profile = LinearActionSamplingProfileV1 {
            maximum_direction_basis_condition: 10.0,
            ..sampling_profile()
        };
        assert!(matches!(
            evaluate_linear_action_samples(
                &add(square("x"), square("y")),
                &canonical_action(),
                &samples(&states, [[1.0, 1.0], [1.0, 1.000001]]),
                profile
            ),
            Err(LinearActionSamplingErrorV1::DirectionBasisIllConditioned { .. })
        ));
    }

    #[test]
    fn sampled_linear_action_rejects_insufficient_informative_samples() {
        let states = [[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]];
        let expression = Expr::Const(3.0);
        assert!(matches!(
            evaluate_linear_action_samples(
                &expression,
                &canonical_action(),
                &samples(&states, [[1.0, 0.0], [0.0, 1.0]]),
                sampling_profile()
            ),
            Err(LinearActionSamplingErrorV1::InsufficientInformativeSamples { .. })
        ));
    }

    #[test]
    fn sampled_linear_action_rejects_profile_that_exceeds_hard_sample_ceiling() {
        let states = [[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]];
        let profile = LinearActionSamplingProfileV1 {
            maximum_samples: HARD_MAX_LINEAR_ACTION_SAMPLES + 1,
            minimum_informative_samples: 1,
            ..sampling_profile()
        };
        assert!(matches!(
            evaluate_linear_action_samples(
                &add(square("x"), square("y")),
                &canonical_action(),
                &samples(&states, [[1.0, 0.0], [0.0, 1.0]]),
                profile
            ),
            Err(LinearActionSamplingErrorV1::InvalidProfile)
        ));
    }

    #[test]
    fn sampled_linear_action_rejects_sample_budget_excess() {
        let states = [[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]];
        let profile = LinearActionSamplingProfileV1 {
            maximum_samples: 2,
            minimum_informative_samples: 1,
            ..sampling_profile()
        };
        assert!(matches!(
            evaluate_linear_action_samples(
                &add(square("x"), square("y")),
                &canonical_action(),
                &samples(&states, [[1.0, 0.0], [0.0, 1.0]]),
                profile
            ),
            Err(LinearActionSamplingErrorV1::SampleBudgetExceeded { .. })
        ));
    }
}
