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

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};

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
    /// Values for free expression variables in first-seen order, followed by
    /// any action coordinates absent from the expression. Sum-bound variables
    /// are supplied by lexical evaluation and are not part of this state vector.
    pub state: Vec<f64>,
    /// Finite-difference direction probes. They must span the action-coordinate
    /// tangent space; overcomplete probe sets test local directional consistency.
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
    /// Maximum number of expression variables, including action coordinates.
    pub maximum_expression_variables: usize,
    /// Maximum estimated AST node evaluations across the complete sample batch.
    pub maximum_expression_evaluation_cost: usize,
    /// Minimum informative samples required for every generator.
    pub minimum_informative_samples: usize,
    /// Maximum allowed spectral condition number of each direction basis.
    pub maximum_direction_basis_condition: f64,
    /// Maximum relative residual when reconstructing the gradient from directional differences.
    pub maximum_directional_reconstruction_relative_residual: f64,
}

/// Hard ceiling independent of caller configuration. This prevents a profile
/// from accidentally removing the sampling resource bound entirely.
const HARD_MAX_LINEAR_ACTION_SAMPLES: usize = 4096;
const HARD_MAX_LINEAR_ACTION_EXPRESSION_VARIABLES: usize = 512;
const HARD_MAX_LINEAR_ACTION_DIMENSION: usize = 32;
const HARD_MAX_EXPRESSION_EVALUATION_COST: usize = 50_000_000;
const HARD_MAX_LINEAR_ALGEBRA_OPERATION_COST: usize = 100_000_000;
const HARD_MAX_LINEAR_ACTION_GENERATORS: usize = 128;
const HARD_MAX_LINEAR_ACTION_DIRECTIONS: usize = 128;
const HARD_MAX_DIRECTION_BASIS_CONDITION: f64 = 1e12;

impl Default for LinearActionSamplingProfileV1 {
    fn default() -> Self {
        Self {
            finite_difference_step: 1e-5,
            squared_cosine_tolerance: 1e-6,
            maximum_samples: 256,
            maximum_expression_variables: 64,
            maximum_expression_evaluation_cost: 10_000_000,
            minimum_informative_samples: 8,
            maximum_direction_basis_condition: 1e8,
            maximum_directional_reconstruction_relative_residual: 1e-8,
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
    pub median_normalized_squared_residual: f64,
    pub p95_normalized_squared_residual: f64,
    pub standard_deviation_normalized_squared_residual: f64,
    pub maximum_normalized_squared_residual: f64,
    pub within_tolerance: bool,
    pub sample_residuals: Vec<LinearActionSampleResidualV1>,
}

/// Numerical conditioning diagnostics for one sample's finite-difference basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionSampleConditionDiagnosticV1 {
    pub sample_id: String,
    pub direction_basis_condition_number: f64,
    pub relative_directional_reconstruction_residual: f64,
}

/// Per-sample residual retained in the evidence receipt. None means the
/// sample was uninformative because its gradient or orbit tangent was zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionSampleResidualV1 {
    pub sample_id: String,
    pub normalized_squared_residual: Option<f64>,
}

/// Evidence report for a sampled scalar-expression invariance check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearActionSampledInvarianceReportV1 {
    pub action_id: String,
    pub coordinate_basis_id: String,
    pub profile: LinearActionSamplingProfileV1,
    /// BLAKE3 of the canonical expression AST encoding; identifier, not a signature.
    pub expression_blake3: String,
    /// BLAKE3 of the exact action declaration, including metric and generators.
    pub declared_action_blake3: String,
    /// BLAKE3 of the A1 admission profile that produced the admitted action.
    pub action_admission_profile_blake3: String,
    /// BLAKE3 of ordered variable names, sample IDs, states and directions.
    pub sample_batch_blake3: String,
    /// BLAKE3 of the complete sampling profile.
    pub sampling_profile_blake3: String,
    /// Conservative upper bound on AST node evaluations for this batch.
    pub estimated_expression_evaluation_cost: usize,
    /// Conservative operation budget for direction SVDs and generator tangents.
    pub estimated_linear_algebra_operation_cost: usize,
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
    ExpressionVariableBudgetExceeded { actual: usize, maximum: usize },
    ActionDimensionLimitExceeded { actual: usize, maximum: usize },
    ExpressionEvaluationBudgetExceeded { estimated: usize, maximum: usize },
    GeneratorCountLimitExceeded { actual: usize, maximum: usize },
    LinearAlgebraBudgetExceeded { estimated: usize, maximum: usize },
    NonFiniteState { sample_id: String, index: usize },
    DirectionCountOutsideBounds {
        sample_id: String,
        minimum: usize,
        maximum: usize,
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
    NonFinitePerturbation {
        sample_id: String,
        direction_index: usize,
        coordinate_index: usize,
    },
    ZeroDisplacementDirection { sample_id: String, direction_index: usize },
    ZeroDirectionBasis { sample_id: String },
    DirectionBasisIllConditioned {
        sample_id: String,
        condition_number: f64,
    },
    NonFiniteExpressionEvaluation { sample_id: String },
    GradientReconstructionFailed { sample_id: String },
    GradientReconstructionResidualExceeded {
        sample_id: String,
        relative_residual: f64,
        maximum: f64,
    },
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

/// Append a length-prefixed byte string to a canonical BLAKE3 input.
fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_usize(hasher: &mut blake3::Hasher, value: usize) {
    hasher.update(&(value as u64).to_le_bytes());
}

fn hash_f64(hasher: &mut blake3::Hasher, value: f64) {
    hasher.update(&value.to_bits().to_le_bytes());
}

/// Canonical structural encoding of the expression tree. Constants use their
/// exact IEEE-754 bit patterns; this is an identity digest, not a semantic hash.
fn hash_expression_node(hasher: &mut blake3::Hasher, expr: &Expr) {
    match expr {
        Expr::Var(name) => {
            hasher.update(&[0]);
            hash_str(hasher, name);
        }
        Expr::Const(value) => {
            hasher.update(&[1]);
            hash_f64(hasher, *value);
        }
        Expr::BinOp(operation, left, right) => {
            hasher.update(&[2]);
            let tag = match operation {
                BinOp::Add => 0,
                BinOp::Sub => 1,
                BinOp::Mul => 2,
                BinOp::Div => 3,
                BinOp::Pow => 4,
            };
            hasher.update(&[tag]);
            hash_expression_node(hasher, left);
            hash_expression_node(hasher, right);
        }
        Expr::Func(function, argument) => {
            hasher.update(&[3]);
            let tag = match function {
                UnaryFn::Sqrt => 0,
                UnaryFn::Log => 1,
                UnaryFn::Exp => 2,
                UnaryFn::Sin => 3,
                UnaryFn::Cos => 4,
                UnaryFn::Abs => 5,
                UnaryFn::Floor => 6,
            };
            hasher.update(&[tag]);
            hash_expression_node(hasher, argument);
        }
        Expr::Sum(body, variable) => {
            hasher.update(&[4]);
            hash_str(hasher, variable);
            hash_expression_node(hasher, body);
        }
    }
}

fn expression_blake3(expr: &Expr) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-linear-action-expression-v1");
    hash_expression_node(&mut hasher, expr);
    hasher.finalize().to_hex().to_string()
}

/// Hash the exact caller-supplied action declaration, including ordered IDs,
/// units, matrix dimensions and raw IEEE-754 entries.
fn action_declaration_blake3(action: &crate::linear_group_action::LinearGroupActionV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-linear-group-action-declaration-v1");
    hash_str(&mut hasher, &action.action_id);
    hash_str(&mut hasher, &action.coordinate_basis_id);
    hash_usize(&mut hasher, action.coordinate_names.len());
    for name in &action.coordinate_names {
        hash_str(&mut hasher, name);
    }
    hash_usize(&mut hasher, action.coordinate_unit_ids.len());
    for unit in &action.coordinate_unit_ids {
        hash_str(&mut hasher, unit);
    }
    hash_usize(&mut hasher, action.metric.len());
    for row in &action.metric {
        hash_usize(&mut hasher, row.len());
        for value in row {
            hash_f64(&mut hasher, *value);
        }
    }
    hash_usize(&mut hasher, action.generators.len());
    for generator in &action.generators {
        hash_str(&mut hasher, &generator.id);
        hash_usize(&mut hasher, generator.matrix.len());
        for row in &generator.matrix {
            hash_usize(&mut hasher, row.len());
            for value in row {
                hash_f64(&mut hasher, *value);
            }
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn action_admission_profile_blake3(
    profile: crate::linear_group_action::LinearGroupActionAdmissionProfileV1,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-linear-group-action-admission-profile-v1");
    hash_usize(&mut hasher, profile.maximum_dimension);
    hash_usize(&mut hasher, profile.maximum_generators);
    hash_f64(&mut hasher, profile.metric_symmetry_relative_tolerance);
    hash_f64(&mut hasher, profile.metric_skew_relative_tolerance);
    hasher.finalize().to_hex().to_string()
}

fn sample_batch_blake3(
    variable_names: &[String],
    samples: &[LinearActionSampleV1],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-linear-action-sample-batch-v1");
    hash_usize(&mut hasher, variable_names.len());
    for name in variable_names {
        hash_str(&mut hasher, name);
    }
    hash_usize(&mut hasher, samples.len());
    for sample in samples {
        hash_str(&mut hasher, &sample.sample_id);
        hash_usize(&mut hasher, sample.state.len());
        for value in &sample.state {
            hash_f64(&mut hasher, *value);
        }
        hash_usize(&mut hasher, sample.direction_basis.len());
        for direction in &sample.direction_basis {
            hash_usize(&mut hasher, direction.len());
            for value in direction {
                hash_f64(&mut hasher, *value);
            }
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn sampling_profile_blake3(profile: LinearActionSamplingProfileV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-linear-action-sampling-profile-v1");
    hash_f64(&mut hasher, profile.finite_difference_step);
    hash_f64(&mut hasher, profile.squared_cosine_tolerance);
    hash_usize(&mut hasher, profile.maximum_samples);
    hash_usize(&mut hasher, profile.maximum_expression_variables);
    hash_usize(&mut hasher, profile.maximum_expression_evaluation_cost);
    hash_usize(&mut hasher, profile.minimum_informative_samples);
    hash_f64(&mut hasher, profile.maximum_direction_basis_condition);
    hash_f64(
        &mut hasher,
        profile.maximum_directional_reconstruction_relative_residual,
    );
    hasher.finalize().to_hex().to_string()
}

/// Upper bound on recursive expression-node visits for one evaluation.
/// A Sum evaluates its body at most 101 times (the core Expr evaluator caps n at 100).
fn expression_evaluation_cost(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Var(_) | Expr::Const(_) => Some(1),
        Expr::BinOp(_, left, right) => 1usize
            .checked_add(expression_evaluation_cost(left)?)?
            .checked_add(expression_evaluation_cost(right)?),
        Expr::Func(_, argument) => 1usize.checked_add(expression_evaluation_cost(argument)?),
        Expr::Sum(body, _) => 1usize.checked_add(expression_evaluation_cost(body)?.checked_mul(101)?),
    }
}

/// Lexically scoped expression evaluation. Unlike the legacy Expr evaluator,
/// an inner Sum binding shadows an outer binding with the same variable name.
fn evaluate_expression_node(expr: &Expr, variables: &[(&str, f64)]) -> f64 {
    match expr {
        Expr::Var(name) => variables
            .iter()
            .rev()
            .find(|(candidate, _)| *candidate == name.as_str())
            .map(|(_, value)| *value)
            .unwrap_or(f64::NAN),
        Expr::Const(value) => *value,
        Expr::BinOp(operation, left, right) => {
            let left = evaluate_expression_node(left, variables);
            let right = evaluate_expression_node(right, variables);
            match operation {
                BinOp::Add => left + right,
                BinOp::Sub => left - right,
                BinOp::Mul => left * right,
                BinOp::Div => {
                    if right.abs() > 1e-15 {
                        left / right
                    } else {
                        f64::NAN
                    }
                }
                BinOp::Pow => left.powf(right),
            }
        }
        Expr::Func(function, argument) => {
            let value = evaluate_expression_node(argument, variables);
            match function {
                UnaryFn::Sqrt => {
                    if value >= 0.0 { value.sqrt() } else { f64::NAN }
                }
                UnaryFn::Log => {
                    if value > 0.0 { value.ln() } else { f64::NAN }
                }
                UnaryFn::Exp => value.exp(),
                UnaryFn::Sin => value.sin(),
                UnaryFn::Cos => value.cos(),
                UnaryFn::Abs => value.abs(),
                UnaryFn::Floor => value.floor(),
            }
        }
        Expr::Sum(body, variable) => {
            let upper = variables
                .iter()
                .rev()
                .find(|(name, _)| *name == "n")
                .map(|(_, value)| *value as usize)
                .unwrap_or(0)
                .min(100);
            let mut sum = 0.0_f64;
            for index in 0..=upper {
                let mut inner = variables.to_vec();
                inner.push((variable.as_str(), index as f64));
                sum += evaluate_expression_node(body, &inner);
                if !sum.is_finite() {
                    return f64::NAN;
                }
            }
            sum
        }
    }
}

fn evaluate_expression(expr: &Expr, variable_names: &[String], state: &[f64]) -> f64 {
    let bindings: Vec<(&str, f64)> = variable_names
        .iter()
        .map(String::as_str)
        .zip(state.iter().copied())
        .collect();
    evaluate_expression_node(expr, &bindings)
}

/// Collect free variables in deterministic first-seen order. Sum-bound variables
/// are excluded from the sample state and are supplied by the lexical evaluator.
fn collect_variable_names_scoped(
    expr: &Expr,
    bound: &mut Vec<String>,
    output: &mut Vec<String>,
) {
    match expr {
        Expr::Var(name) => {
            if !bound.contains(name) && !output.contains(name) {
                output.push(name.clone());
            }
        }
        Expr::Const(_) => {}
        Expr::BinOp(_, left, right) => {
            collect_variable_names_scoped(left, bound, output);
            collect_variable_names_scoped(right, bound, output);
        }
        Expr::Func(_, argument) => collect_variable_names_scoped(argument, bound, output),
        Expr::Sum(body, variable) => {
            bound.push(variable.clone());
            collect_variable_names_scoped(body, bound, output);
            bound.pop();
        }
    }
}

/// Reconstruct the metric-whitened action-coordinate gradient from directional differences.
///
/// If V contains the supplied direction vectors as columns, then q = V^T p.
/// With G = s L L^T, define V_w = L^T V and p_w = L^-1 p. Then
/// q = V_w^T p_w. The SVD condition number is therefore measured in the
/// intrinsic metric geometry. Under x' = D x and G' = D^-T G D^-1,
/// L'^T D differs from L^T only by an orthogonal factor, so the condition
/// number is invariant to valid non-orthogonal coordinate re-representations.
fn finite_difference_gradient(
    expr: &Expr,
    variable_names: &[String],
    action_coordinate_indices: &[usize],
    sample: &LinearActionSampleV1,
    lower: &[Vec<f64>],
    profile: LinearActionSamplingProfileV1,
) -> Result<(Vec<f64>, f64, f64), LinearActionSamplingErrorV1> {
    let dimension = action_coordinate_indices.len();
    let direction_count = sample.direction_basis.len();
    let whitened_directions: Vec<Vec<f64>> = sample
        .direction_basis
        .iter()
        .map(|direction| multiply_lower_transpose(lower, direction))
        .collect();
    if whitened_directions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(LinearActionSamplingErrorV1::ZeroDirectionBasis {
            sample_id: sample.sample_id.clone(),
        });
    }
    let direction_scale = whitened_directions
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if !direction_scale.is_finite() || direction_scale == 0.0 {
        return Err(LinearActionSamplingErrorV1::ZeroDirectionBasis {
            sample_id: sample.sample_id.clone(),
        });
    }

    // Each row is one directional probe in whitened coordinates. With more
    // probes than dimensions, the SVD computes a least-squares gradient and
    // the reconstruction residual measures whether all finite differences
    // are consistent with one local covector.
    let measurement_matrix = DMatrix::from_fn(direction_count, dimension, |row, column| {
        whitened_directions[row][column] / direction_scale
    });
    let decomposition = measurement_matrix.clone().svd(true, true);
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

    // Use a common scale for every plus/minus function value. The common
    // finite-difference factor 1/(2h) and this expression scale both cancel
    // from the gradient direction required by a cosine residual. This avoids
    // overflowing a quotient when h is tiny or when f has a huge amplitude.
    let mut function_pairs = Vec::with_capacity(direction_count);
    let mut expression_value_scale = 0.0_f64;
    for (direction_index, direction) in sample.direction_basis.iter().enumerate() {
        let mut plus = sample.state.clone();
        let mut minus = sample.state.clone();
        let mut moved = false;
        for (action_index, &variable_index) in action_coordinate_indices.iter().enumerate() {
            let displacement = profile.finite_difference_step * direction[action_index];
            if !displacement.is_finite() {
                return Err(LinearActionSamplingErrorV1::NonFinitePerturbation {
                    sample_id: sample.sample_id.clone(),
                    direction_index,
                    coordinate_index: action_index,
                });
            }
            let plus_value = plus[variable_index] + displacement;
            let minus_value = minus[variable_index] - displacement;
            if !plus_value.is_finite() || !minus_value.is_finite() {
                return Err(LinearActionSamplingErrorV1::NonFinitePerturbation {
                    sample_id: sample.sample_id.clone(),
                    direction_index,
                    coordinate_index: action_index,
                });
            }
            moved |= plus_value != plus[variable_index] || minus_value != minus[variable_index];
            plus[variable_index] = plus_value;
            minus[variable_index] = minus_value;
        }
        if !moved {
            return Err(LinearActionSamplingErrorV1::ZeroDisplacementDirection {
                sample_id: sample.sample_id.clone(),
                direction_index,
            });
        }
        let plus_value = evaluate_expression(expr, variable_names, &plus);
        let minus_value = evaluate_expression(expr, variable_names, &minus);
        if !plus_value.is_finite() || !minus_value.is_finite() {
            return Err(LinearActionSamplingErrorV1::NonFiniteExpressionEvaluation {
                sample_id: sample.sample_id.clone(),
            });
        }
        expression_value_scale = expression_value_scale
            .max(plus_value.abs())
            .max(minus_value.abs());
        function_pairs.push((plus_value, minus_value));
    }

    // Prefer raw pair differences: subtracting first preserves small changes
    // on top of a large common function offset. If an opposite-sign pair would
    // overflow, fall back to the common value scaling above.
    let raw_differences: Vec<f64> = function_pairs
        .iter()
        .map(|(plus, minus)| plus - minus)
        .collect();
    let observed = if raw_differences.iter().all(|value| value.is_finite()) {
        raw_differences
    } else if expression_value_scale == 0.0 {
        vec![0.0_f64; direction_count]
    } else {
        function_pairs
            .iter()
            .map(|(plus, minus)| plus / expression_value_scale - minus / expression_value_scale)
            .collect()
    };
    let observed_scale = observed
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let normalized_observed: Vec<f64> = if observed_scale == 0.0 {
        vec![0.0_f64; direction_count]
    } else {
        observed.iter().map(|value| value / observed_scale).collect()
    };

    // The direction matrix is normalized in metric-whitened coordinates and
    // the observed directional differences are normalized by one common
    // positive scale. The recovered vector is therefore a positive scalar
    // multiple of L^-1 df, exactly the dual-whitened covector needed below.
    let observed_vector = DVector::from_vec(normalized_observed);
    let solution = decomposition
        .solve(&observed_vector, f64::EPSILON * dimension as f64)
        .map_err(|_| LinearActionSamplingErrorV1::GradientReconstructionFailed {
            sample_id: sample.sample_id.clone(),
        })?;
    let gradient: Vec<f64> = solution.iter().copied().collect();
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(LinearActionSamplingErrorV1::GradientReconstructionFailed {
            sample_id: sample.sample_id.clone(),
        });
    }
    let reconstructed = &measurement_matrix * &solution;
    let residual_norm = reconstructed
        .iter()
        .zip(observed_vector.iter())
        .fold(0.0_f64, |norm, (left, right)| norm.hypot(left - right));
    let observed_norm = stable_norm(observed_vector.as_slice());
    let relative_reconstruction_residual = if observed_norm == 0.0 {
        residual_norm
    } else {
        residual_norm / observed_norm
    };
    if !relative_reconstruction_residual.is_finite() {
        return Err(LinearActionSamplingErrorV1::GradientReconstructionFailed {
            sample_id: sample.sample_id.clone(),
        });
    }
    if relative_reconstruction_residual
        > profile.maximum_directional_reconstruction_relative_residual
    {
        return Err(LinearActionSamplingErrorV1::GradientReconstructionResidualExceeded {
            sample_id: sample.sample_id.clone(),
            relative_residual: relative_reconstruction_residual,
            maximum: profile.maximum_directional_reconstruction_relative_residual,
        });
    }
    Ok((gradient, condition_number, relative_reconstruction_residual))
}

/// Forward solve L y = p for lower-triangular L.
fn solve_lower_triangular(lower: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let mut solution = Vec::with_capacity(rhs.len());
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

/// Compute squared Euclidean cosine between already metric-whitened vectors.
///
/// The finite-difference path reconstructs L^-1 df directly, while the orbit
/// tangent is transformed by L^T. Thus no second whitening is required here.
fn whitened_squared_cosine(
    whitened_gradient: &[f64],
    whitened_tangent: &[f64],
) -> Option<f64> {
    let gradient_scale = whitened_gradient
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let tangent_scale = whitened_tangent
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if !gradient_scale.is_finite() || !tangent_scale.is_finite() {
        return None;
    }
    if gradient_scale == 0.0 || tangent_scale == 0.0 {
        return Some(f64::NAN);
    }
    let normalized_gradient: Vec<f64> = whitened_gradient
        .iter()
        .map(|value| *value / gradient_scale)
        .collect();
    let normalized_tangent: Vec<f64> = whitened_tangent
        .iter()
        .map(|value| *value / tangent_scale)
        .collect();
    let gradient_norm = stable_norm(&normalized_gradient);
    let tangent_norm = stable_norm(&normalized_tangent);
    if !gradient_norm.is_finite() || !tangent_norm.is_finite() {
        return None;
    }
    if gradient_norm == 0.0 || tangent_norm == 0.0 {
        return Some(f64::NAN);
    }
    let cosine = normalized_gradient
        .iter()
        .zip(normalized_tangent.iter())
        .map(|(left, right)| (left / gradient_norm) * (right / tangent_norm))
        .sum::<f64>();
    if !cosine.is_finite() {
        return None;
    }
    Some((cosine * cosine).clamp(0.0, 1.0))
}

fn median_and_p95(sorted_values: &[f64]) -> (f64, f64) {
    let count = sorted_values.len();
    let median = if count % 2 == 0 {
        0.5 * (sorted_values[count / 2 - 1] + sorted_values[count / 2])
    } else {
        sorted_values[count / 2]
    };
    let p95_index = ((95 * count).div_ceil(100)).saturating_sub(1).min(count - 1);
    (median, sorted_values[p95_index])
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
        || profile.maximum_expression_variables == 0
        || profile.maximum_expression_variables > HARD_MAX_LINEAR_ACTION_EXPRESSION_VARIABLES
        || profile.maximum_expression_evaluation_cost == 0
        || profile.maximum_expression_evaluation_cost > HARD_MAX_EXPRESSION_EVALUATION_COST
        || profile.minimum_informative_samples == 0
        || profile.minimum_informative_samples > profile.maximum_samples
        || !profile.maximum_direction_basis_condition.is_finite()
        || profile.maximum_direction_basis_condition < 1.0
        || profile.maximum_direction_basis_condition > HARD_MAX_DIRECTION_BASIS_CONDITION
        || !profile
            .maximum_directional_reconstruction_relative_residual
            .is_finite()
        || !(0.0..=1.0).contains(
            &profile.maximum_directional_reconstruction_relative_residual
        )
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
    if dimension > HARD_MAX_LINEAR_ACTION_DIMENSION {
        return Err(LinearActionSamplingErrorV1::ActionDimensionLimitExceeded {
            actual: dimension,
            maximum: HARD_MAX_LINEAR_ACTION_DIMENSION,
        });
    }

    let mut variable_names = Vec::new();
    collect_variable_names_scoped(expr, &mut Vec::new(), &mut variable_names);
    for coordinate_name in &declaration.coordinate_names {
        if !variable_names.contains(coordinate_name) {
            variable_names.push(coordinate_name.clone());
        }
    }
    if variable_names.len() > profile.maximum_expression_variables {
        return Err(LinearActionSamplingErrorV1::ExpressionVariableBudgetExceeded {
            actual: variable_names.len(),
            maximum: profile.maximum_expression_variables,
        });
    }

    if declaration.generators.len() > HARD_MAX_LINEAR_ACTION_GENERATORS {
        return Err(LinearActionSamplingErrorV1::GeneratorCountLimitExceeded {
            actual: declaration.generators.len(),
            maximum: HARD_MAX_LINEAR_ACTION_GENERATORS,
        });
    }

    // Count actual directional probes. Overcomplete probe sets improve
    // finite-difference consistency checks, so both AST and linear-algebra
    // budgets charge every supplied direction rather than only the dimension.
    let total_directional_probes = samples.iter().fold(0usize, |total, sample| {
        total.checked_add(sample.direction_basis.len()).unwrap_or(usize::MAX)
    });

    // Bound the overdetermined SVD work plus per-generator tangent products.
    // The multiplier is a conservative operation proxy, not a measured runtime.
    let dimension_squared = dimension.checked_mul(dimension).unwrap_or(usize::MAX);
    let direction_factorization_cost = dimension_squared
        .checked_mul(total_directional_probes)
        .and_then(|cost| cost.checked_mul(10))
        .unwrap_or(usize::MAX);
    let tangent_cost = dimension_squared
        .checked_mul(declaration.generators.len())
        .and_then(|cost| cost.checked_mul(samples.len()))
        .unwrap_or(usize::MAX);
    let estimated_linear_algebra_operation_cost = direction_factorization_cost
        .checked_add(tangent_cost)
        .unwrap_or(usize::MAX);
    if estimated_linear_algebra_operation_cost > HARD_MAX_LINEAR_ALGEBRA_OPERATION_COST {
        return Err(LinearActionSamplingErrorV1::LinearAlgebraBudgetExceeded {
            estimated: estimated_linear_algebra_operation_cost,
            maximum: HARD_MAX_LINEAR_ALGEBRA_OPERATION_COST,
        });
    }

    // Sum nodes are conservatively charged for their maximum 101 body
    // evaluations, so a small AST cannot hide an expensive summation.
    let per_evaluation_cost = expression_evaluation_cost(expr).unwrap_or(usize::MAX);
    let estimated_expression_evaluation_cost = per_evaluation_cost
        .checked_mul(2)
        .and_then(|cost| cost.checked_mul(total_directional_probes))
        .unwrap_or(usize::MAX);
    if estimated_expression_evaluation_cost > profile.maximum_expression_evaluation_cost {
        return Err(LinearActionSamplingErrorV1::ExpressionEvaluationBudgetExceeded {
            estimated: estimated_expression_evaluation_cost,
            maximum: profile.maximum_expression_evaluation_cost,
        });
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
        if sample.direction_basis.len() < dimension
            || sample.direction_basis.len() > HARD_MAX_LINEAR_ACTION_DIRECTIONS
        {
            return Err(LinearActionSamplingErrorV1::DirectionCountOutsideBounds {
                sample_id: sample.sample_id.clone(),
                minimum: dimension,
                maximum: HARD_MAX_LINEAR_ACTION_DIRECTIONS,
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
        let (gradient, condition_number, reconstruction_residual) = finite_difference_gradient(
            expr,
            &variable_names,
            &action_coordinate_indices,
            sample,
            action.scaled_metric_cholesky_lower(),
            profile,
        )?;
        sample_gradients.push(gradient);
        direction_basis_diagnostics.push(LinearActionSampleConditionDiagnosticV1 {
            sample_id: sample.sample_id.clone(),
            direction_basis_condition_number: condition_number,
            relative_directional_reconstruction_residual: reconstruction_residual,
        });
    }

    let mut generator_statistics = Vec::with_capacity(declaration.generators.len());
    for generator in &declaration.generators {
        let mut informative_sample_count = 0usize;
        let mut skipped_sample_count = 0usize;
        let mut residual_sum = 0.0_f64;
        let mut maximum_residual = 0.0_f64;
        let mut residual_values = Vec::with_capacity(samples.len());
        let mut sample_residuals = Vec::with_capacity(samples.len());

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

            let whitened_tangent = multiply_lower_transpose(
                action.scaled_metric_cholesky_lower(),
                &tangent,
            );
            let residual = whitened_squared_cosine(gradient, &whitened_tangent)
            .ok_or_else(|| LinearActionSamplingErrorV1::NonFiniteMetricWhitening {
                sample_id: sample.sample_id.clone(),
                generator_id: generator.id.clone(),
            })?;
            if residual.is_nan() {
                skipped_sample_count = skipped_sample_count.saturating_add(1);
                sample_residuals.push(LinearActionSampleResidualV1 {
                    sample_id: sample.sample_id.clone(),
                    normalized_squared_residual: None,
                });
                continue;
            }
            sample_residuals.push(LinearActionSampleResidualV1 {
                sample_id: sample.sample_id.clone(),
                normalized_squared_residual: Some(residual),
            });
            informative_sample_count = informative_sample_count.saturating_add(1);
            residual_sum += residual;
            maximum_residual = maximum_residual.max(residual);
            residual_values.push(residual);
        }

        if informative_sample_count < profile.minimum_informative_samples {
            return Err(LinearActionSamplingErrorV1::InsufficientInformativeSamples {
                generator_id: generator.id.clone(),
                found: informative_sample_count,
                required: profile.minimum_informative_samples,
            });
        }

        residual_values.sort_by(f64::total_cmp);
        let (median_residual, p95_residual) = median_and_p95(&residual_values);
        let mean_residual = residual_sum / informative_sample_count as f64;
        let variance = residual_values
            .iter()
            .map(|value| {
                let difference = *value - mean_residual;
                difference * difference
            })
            .sum::<f64>()
            / informative_sample_count as f64;
        generator_statistics.push(LinearActionGeneratorSampleStatisticsV1 {
            generator_id: generator.id.clone(),
            checked_sample_count: samples.len(),
            informative_sample_count,
            skipped_sample_count,
            mean_normalized_squared_residual: mean_residual,
            median_normalized_squared_residual: median_residual,
            p95_normalized_squared_residual: p95_residual,
            standard_deviation_normalized_squared_residual: variance.sqrt(),
            maximum_normalized_squared_residual: maximum_residual,
            within_tolerance: maximum_residual <= profile.squared_cosine_tolerance,
            sample_residuals,
        });
    }

    Ok(LinearActionSampledInvarianceReportV1 {
        action_id: declaration.action_id.clone(),
        coordinate_basis_id: declaration.coordinate_basis_id.clone(),
        profile,
        expression_blake3: expression_blake3(expr),
        declared_action_blake3: action_declaration_blake3(action.declared_action()),
        action_admission_profile_blake3: action_admission_profile_blake3(*action.profile()),
        sample_batch_blake3: sample_batch_blake3(&variable_names, samples),
        sampling_profile_blake3: sampling_profile_blake3(profile),
        estimated_expression_evaluation_cost,
        estimated_linear_algebra_operation_cost,
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
            maximum_expression_variables: 64,
            maximum_expression_evaluation_cost: 1_000_000,
            minimum_informative_samples: 4,
            maximum_direction_basis_condition: 1e6,
            maximum_directional_reconstruction_relative_residual: 1e-8,
        }
    }

    #[test]
    fn sampled_linear_action_is_invariant_to_expression_amplitude() {
        let states = [
            [0.7, 1.1],
            [1.2, 0.8],
            [1.7, 2.1],
            [2.3, 1.4],
            [0.9, 2.5],
            [2.6, 2.1],
        ];
        let radial = add(square("x"), square("y"));
        let mut baseline: Option<f64> = None;
        for amplitude in [1e-200, 1.0, 1e200] {
            let expression = mul(Expr::Const(amplitude), radial.clone());
            let report = evaluate_linear_action_samples(
                &expression,
                &canonical_action(),
                &samples(&states, [[1.0, 0.0], [0.0, 1.0]]),
                sampling_profile(),
            )
            .unwrap();
            let maximum = report.generator_statistics[0].maximum_normalized_squared_residual;
            assert!(report.invariant_within_tolerance());
            if let Some(expected) = baseline {
                assert!(
                    (maximum - expected).abs() < 1e-10,
                    "residual changed under expression amplitude {amplitude}"
                );
            } else {
                baseline = Some(maximum);
            }
        }
    }

    #[test]
    fn sampled_linear_action_hashes_expression_action_samples_and_profile() {
        let states = [
            [0.7, 1.1],
            [1.2, 0.8],
            [1.7, 2.1],
            [2.3, 1.4],
        ];
        let expression = add(square("x"), square("y"));
        let action = canonical_action();
        let sample_batch = samples(&states, [[1.0, 0.0], [0.0, 1.0]]);
        let first = evaluate_linear_action_samples(
            &expression,
            &action,
            &sample_batch,
            sampling_profile(),
        )
        .unwrap();
        let repeated = evaluate_linear_action_samples(
            &expression,
            &action,
            &sample_batch,
            sampling_profile(),
        )
        .unwrap();

        assert_eq!(first.expression_blake3.len(), 64);
        assert_eq!(first.declared_action_blake3.len(), 64);
        assert_eq!(first.action_admission_profile_blake3.len(), 64);
        assert_eq!(first.sample_batch_blake3.len(), 64);
        assert_eq!(first.sampling_profile_blake3.len(), 64);
        assert_eq!(first.expression_blake3, repeated.expression_blake3);
        assert_eq!(first.declared_action_blake3, repeated.declared_action_blake3);
        assert_eq!(
            first.action_admission_profile_blake3,
            repeated.action_admission_profile_blake3
        );
        assert_eq!(first.sample_batch_blake3, repeated.sample_batch_blake3);
        assert_eq!(first.sampling_profile_blake3, repeated.sampling_profile_blake3);

        let mut changed_samples = sample_batch.clone();
        changed_samples[0].state[0] = 0.7000000000000001;
        let changed = evaluate_linear_action_samples(
            &expression,
            &action,
            &changed_samples,
            sampling_profile(),
        )
        .unwrap();
        assert_ne!(first.sample_batch_blake3, changed.sample_batch_blake3);

        let mut changed_directions = sample_batch.clone();
        changed_directions[0].direction_basis[0][0] = 1.0000000000000002;
        let changed = evaluate_linear_action_samples(
            &expression,
            &action,
            &changed_directions,
            sampling_profile(),
        )
        .unwrap();
        assert_ne!(first.sample_batch_blake3, changed.sample_batch_blake3);

        let changed_expression = add(square("x"), mul(Expr::Const(2.0), square("y")));
        let changed = evaluate_linear_action_samples(
            &changed_expression,
            &action,
            &sample_batch,
            sampling_profile(),
        )
        .unwrap();
        assert_ne!(first.expression_blake3, changed.expression_blake3);

        let changed_profile = LinearActionSamplingProfileV1 {
            finite_difference_step: 2e-5,
            ..sampling_profile()
        };
        let changed = evaluate_linear_action_samples(
            &expression,
            &action,
            &sample_batch,
            changed_profile,
        )
        .unwrap();
        assert_ne!(first.sampling_profile_blake3, changed.sampling_profile_blake3);
    }

    #[test]
    fn sampled_linear_action_lexically_scopes_sum_variables() {
        use symthaea_core::hdc::conjecture_engine::Expr;

        // (n+1)(x²+y²)+n is SO(2)-invariant in x,y. k is a bound
        // summation variable and must not be copied into the sampled state.
        let sum = Expr::Sum(Box::new(add(square("x"), square("y"))), "k".to_string());
        let expression = add(sum, Expr::Var("n".to_string()));
        let sample_batch: Vec<LinearActionSampleV1> = [
            [0.7, 1.1, 3.0],
            [1.2, 0.8, 3.0],
            [1.7, 2.1, 3.0],
            [2.3, 1.4, 3.0],
            [0.9, 2.5, 3.0],
            [2.6, 2.1, 3.0],
        ]
        .iter()
        .enumerate()
        .map(|(index, state)| LinearActionSampleV1 {
            sample_id: format!("sum-{index}"),
            state: state.to_vec(),
            direction_basis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        })
        .collect();
        let report = evaluate_linear_action_samples(
            &expression,
            &canonical_action(),
            &sample_batch,
            sampling_profile(),
        )
        .unwrap();
        assert!(report.invariant_within_tolerance());
        assert_eq!(
            report.generator_statistics[0].informative_sample_count,
            sample_batch.len()
        );
    }

    #[test]
    fn sampled_linear_action_rejects_nonfinite_and_zero_fd_perturbations() {
        let states = [[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]];
        let large_step = LinearActionSamplingProfileV1 {
            finite_difference_step: 1e308,
            minimum_informative_samples: 1,
            ..sampling_profile()
        };
        assert!(matches!(
            evaluate_linear_action_samples(
                &add(square("x"), square("y")),
                &canonical_action(),
                &samples(&states, [[2.0, 0.0], [0.0, 1.0]]),
                large_step
            ),
            Err(LinearActionSamplingErrorV1::NonFinitePerturbation { .. })
        ));

        let tiny_direction = [
            LinearActionSampleV1 {
                sample_id: "tiny-0".to_string(),
                state: states[0].to_vec(),
                direction_basis: vec![vec![1e-320, 0.0], vec![0.0, 1e-320]],
            },
            LinearActionSampleV1 {
                sample_id: "tiny-1".to_string(),
                state: states[1].to_vec(),
                direction_basis: vec![vec![1e-320, 0.0], vec![0.0, 1.0]],
            },
            LinearActionSampleV1 {
                sample_id: "tiny-2".to_string(),
                state: states[2].to_vec(),
                direction_basis: vec![vec![1e-320, 0.0], vec![0.0, 1.0]],
            },
            LinearActionSampleV1 {
                sample_id: "tiny-3".to_string(),
                state: states[3].to_vec(),
                direction_basis: vec![vec![1e-320, 0.0], vec![0.0, 1.0]],
            },
        ];
        assert!(matches!(
            evaluate_linear_action_samples(
                &add(square("x"), square("y")),
                &canonical_action(),
                &tiny_direction,
                sampling_profile()
            ),
            Err(LinearActionSamplingErrorV1::ZeroDisplacementDirection { .. })
        ));
    }

    #[test]
    fn sampled_linear_action_enforces_ast_evaluation_budget() {
        let states = [[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]];
        let expression = Expr::Sum(
            Box::new(add(square("x"), square("y"))),
            "k".to_string(),
        );
        let profile = LinearActionSamplingProfileV1 {
            maximum_expression_evaluation_cost: 100,
            minimum_informative_samples: 1,
            ..sampling_profile()
        };
        assert!(matches!(
            evaluate_linear_action_samples(
                &expression,
                &canonical_action(),
                &samples(&states, [[1.0, 0.0], [0.0, 1.0]]),
                profile
            ),
            Err(LinearActionSamplingErrorV1::ExpressionEvaluationBudgetExceeded { .. })
        ));
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
        let statistics = &report.generator_statistics[0];
        assert!(statistics.median_normalized_squared_residual
            <= statistics.p95_normalized_squared_residual);
        assert!(statistics.p95_normalized_squared_residual
            <= statistics.maximum_normalized_squared_residual);
        assert!(statistics.standard_deviation_normalized_squared_residual.is_finite());
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

    fn quadratic_form_expression(
        matrix: &nalgebra::Matrix2<f64>,
        first: &str,
        second: &str,
    ) -> Expr {
        add(
            add(
                mul(Expr::Const(matrix[(0, 0)]), square(first)),
                mul(Expr::Const(matrix[(1, 1)]), square(second)),
            ),
            mul(
                Expr::Const(matrix[(0, 1)] + matrix[(1, 0)]),
                mul(Expr::Var(first.to_string()), Expr::Var(second.to_string())),
            ),
        )
    }

    #[test]
    fn sampled_residual_is_covariant_across_paired_nonorthogonal_coordinate_changes() {
        // Replay the exact same geometric samples and their finite-difference
        // directions in six valid non-orthogonal coordinate representations.
        let transforms = [
            nalgebra::Matrix2::new(1.0, 1.0, 0.0, 1.0),
            nalgebra::Matrix2::new(2.0, 1.0, 1.0, 1.0),
            nalgebra::Matrix2::new(1.0, 0.5, -0.25, 2.0),
            nalgebra::Matrix2::new(0.5, -1.0, 1.0, 1.0),
            nalgebra::Matrix2::new(2.0, -1.0, 0.5, 1.0),
            nalgebra::Matrix2::new(1.0, 0.0, 3.0, 1.0),
        ];
        let canonical_states = [
            nalgebra::Vector2::new(0.7, 1.1),
            nalgebra::Vector2::new(1.2, 0.8),
            nalgebra::Vector2::new(1.7, 2.1),
            nalgebra::Vector2::new(2.3, 1.4),
            nalgebra::Vector2::new(0.9, 2.5),
            nalgebra::Vector2::new(2.6, 2.1),
        ];
        let canonical_quadratic = nalgebra::Matrix2::new(1.0, 0.0, 0.0, 1.2);
        let canonical_generator = nalgebra::Matrix2::new(0.0, -1.0, 1.0, 0.0);
        let canonical_expression = Expr::Func(
            UnaryFn::Sqrt,
            Box::new(quadratic_form_expression(&canonical_quadratic, "x", "y")),
        );
        let canonical_directions = [
            nalgebra::Vector2::new(1.0, 0.0),
            nalgebra::Vector2::new(0.0, 1.0),
            nalgebra::Vector2::new(1.0, 1.0),
        ];
        let canonical_samples: Vec<LinearActionSampleV1> = canonical_states
            .iter()
            .enumerate()
            .map(|(sample_index, state)| LinearActionSampleV1 {
                sample_id: format!("paired-{sample_index}"),
                state: vec![state[0], state[1]],
                direction_basis: canonical_directions
                    .iter()
                    .map(|direction| vec![direction[0], direction[1]])
                    .collect(),
            })
            .collect();
        let canonical_report = evaluate_linear_action_samples(
            &canonical_expression,
            &canonical_action(),
            &canonical_samples,
            sampling_profile(),
        )
        .unwrap();

        for (basis_index, transform) in transforms.iter().enumerate() {
            let inverse = transform.try_inverse().expect("frozen coordinate transform");
            let transformed_metric =
                inverse.transpose() * nalgebra::Matrix2::identity() * inverse;
            let transformed_generator = transform * canonical_generator * inverse;
            let transformed_quadratic = inverse.transpose() * canonical_quadratic * inverse;

            let transformed_action = LinearGroupActionV1 {
                action_id: format!("sampled-so2-basis-{basis_index}"),
                coordinate_basis_id: format!("basis-{basis_index}"),
                coordinate_names: vec!["X".to_string(), "Y".to_string()],
                coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
                metric: transformed_metric
                    .row_iter()
                    .map(|row| row.iter().copied().collect())
                    .collect(),
                generators: vec![LinearGeneratorV1 {
                    id: "J-transformed".to_string(),
                    matrix: transformed_generator
                        .row_iter()
                        .map(|row| row.iter().copied().collect())
                        .collect(),
                }],
            }
            .try_admit(LinearGroupActionAdmissionProfileV1::default())
            .unwrap();
            let transformed_expression = Expr::Func(
                UnaryFn::Sqrt,
                Box::new(quadratic_form_expression(&transformed_quadratic, "X", "Y")),
            );

            let transformed_samples: Vec<LinearActionSampleV1> = canonical_states
                .iter()
                .enumerate()
                .map(|(sample_index, state)| {
                    let transformed_state = transform * state;
                    LinearActionSampleV1 {
                        sample_id: format!("paired-{sample_index}"),
                        state: vec![transformed_state[0], transformed_state[1]],
                        direction_basis: canonical_directions
                            .iter()
                            .map(|direction| {
                                let transformed_direction = transform * direction;
                                vec![transformed_direction[0], transformed_direction[1]]
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

            let canonical_statistics = &canonical_report.generator_statistics[0];
            let transformed_statistics = &transformed_report.generator_statistics[0];
            assert!(
                (canonical_statistics.maximum_normalized_squared_residual
                    - transformed_statistics.maximum_normalized_squared_residual)
                    .abs()
                    < 2.0e-8,
                "maximum residual changed under basis {basis_index}"
            );
            assert_eq!(
                canonical_report.invariant_within_tolerance(),
                transformed_report.invariant_within_tolerance()
            );

            for (canonical_diagnostic, transformed_diagnostic) in canonical_report
                .direction_basis_diagnostics
                .iter()
                .zip(transformed_report.direction_basis_diagnostics.iter())
            {
                assert_eq!(canonical_diagnostic.sample_id, transformed_diagnostic.sample_id);
                assert!(
                    (canonical_diagnostic.direction_basis_condition_number
                        - transformed_diagnostic.direction_basis_condition_number)
                        .abs()
                        < 2.0e-10,
                    "metric-whitened direction-basis condition changed under basis {basis_index}: {} vs {}",
                    canonical_diagnostic.direction_basis_condition_number,
                    transformed_diagnostic.direction_basis_condition_number
                );
            }
            for (canonical_sample, transformed_sample) in canonical_statistics
                .sample_residuals
                .iter()
                .zip(transformed_statistics.sample_residuals.iter())
            {
                assert_eq!(canonical_sample.sample_id, transformed_sample.sample_id);
                assert!(
                    (canonical_sample.normalized_squared_residual.unwrap()
                        - transformed_sample.normalized_squared_residual.unwrap())
                        .abs()
                        < 2.0e-8,
                    "paired residual changed under basis {basis_index}"
                );
            }
            assert_eq!(transformed_report.sample_ids, canonical_report.sample_ids);
            assert!(
                transformed_report.direction_basis_diagnostics.iter().all(|diagnostic| {
                    diagnostic.relative_directional_reconstruction_residual < 1.0e-10
                })
            );
        }
    }

    #[test]
    fn sampled_linear_action_rejects_linear_algebra_budget_excess() {
        let dimension = HARD_MAX_LINEAR_ACTION_DIMENSION;
        let names: Vec<String> = (0..dimension).map(|index| format!("x{index}")).collect();
        let mut generator = vec![vec![0.0_f64; dimension]; dimension];
        for plane in (0..dimension).step_by(2) {
            generator[plane][plane + 1] = -1.0;
            generator[plane + 1][plane] = 1.0;
        }
        let action = LinearGroupActionV1 {
            action_id: "budget-so2-product".to_string(),
            coordinate_basis_id: "x0-through-x31".to_string(),
            coordinate_unit_ids: vec!["1".to_string(); dimension],
            coordinate_names: names,
            metric: (0..dimension)
                .map(|row| {
                    (0..dimension)
                        .map(|column| if row == column { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect(),
            generators: vec![LinearGeneratorV1 {
                id: "block-so2".to_string(),
                matrix: generator,
            }],
        }
        .try_admit(LinearGroupActionAdmissionProfileV1::default())
        .unwrap();

        let direction_basis: Vec<Vec<f64>> = (0..dimension)
            .map(|column| {
                (0..dimension)
                    .map(|row| if row == column { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let samples: Vec<LinearActionSampleV1> = (0..400)
            .map(|index| LinearActionSampleV1 {
                sample_id: format!("budget-{index}"),
                state: vec![1.0; dimension],
                direction_basis: direction_basis.clone(),
            })
            .collect();
        assert!(matches!(
            evaluate_linear_action_samples(
                &Expr::Var("x0".to_string()),
                &action,
                &samples,
                LinearActionSamplingProfileV1 {
                    maximum_samples: HARD_MAX_LINEAR_ACTION_SAMPLES,
                    minimum_informative_samples: 1,
                    ..LinearActionSamplingProfileV1::default()
                }
            ),
            Err(LinearActionSamplingErrorV1::LinearAlgebraBudgetExceeded { .. })
        ));
    }

    #[test]
    fn sampled_linear_action_rejects_action_dimension_above_hard_ceiling() {
        let dimension = HARD_MAX_LINEAR_ACTION_DIMENSION + 1;
        let mut generator = vec![vec![0.0_f64; dimension]; dimension];
        generator[0][1] = -1.0;
        generator[1][0] = 1.0;
        let action = LinearGroupActionV1 {
            action_id: "over-dimension-cap".to_string(),
            coordinate_basis_id: "large-basis".to_string(),
            coordinate_names: (0..dimension).map(|index| format!("x{index}")).collect(),
            coordinate_unit_ids: vec!["1".to_string(); dimension],
            metric: (0..dimension)
                .map(|row| {
                    (0..dimension)
                        .map(|column| if row == column { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect(),
            generators: vec![LinearGeneratorV1 {
                id: "J".to_string(),
                matrix: generator,
            }],
        }
        .try_admit(LinearGroupActionAdmissionProfileV1 {
            maximum_dimension: dimension,
            ..LinearGroupActionAdmissionProfileV1::default()
        })
        .unwrap();
        let sample = LinearActionSampleV1 {
            sample_id: "dimension-cap".to_string(),
            state: vec![1.0; dimension],
            direction_basis: (0..dimension)
                .map(|column| {
                    (0..dimension)
                        .map(|row| if row == column { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect(),
        };
        assert!(matches!(
            evaluate_linear_action_samples(
                &Expr::Var("x0".to_string()),
                &action,
                &[sample],
                LinearActionSamplingProfileV1 {
                    maximum_samples: 1,
                    minimum_informative_samples: 1,
                    ..LinearActionSamplingProfileV1::default()
                }
            ),
            Err(LinearActionSamplingErrorV1::ActionDimensionLimitExceeded {
                actual: 33,
                maximum: 32
            })
        ));
    }

    #[test]
    fn sampled_linear_action_rejects_generator_count_above_sampling_ceiling() {
        let action = LinearGroupActionV1 {
            action_id: "too-many-generators".to_string(),
            coordinate_basis_id: "xy".to_string(),
            coordinate_names: vec!["x".to_string(), "y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generators: (0..=HARD_MAX_LINEAR_ACTION_GENERATORS)
                .map(|index| LinearGeneratorV1 {
                    id: format!("J-{index}"),
                    matrix: vec![vec![0.0, -1.0], vec![1.0, 0.0]],
                })
                .collect(),
        };
        let action = action
            .try_admit(LinearGroupActionAdmissionProfileV1 {
                maximum_generators: HARD_MAX_LINEAR_ACTION_GENERATORS + 1,
                ..LinearGroupActionAdmissionProfileV1::default()
            })
            .unwrap();
        let sample_batch = samples(
            &[[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]],
            [[1.0, 0.0], [0.0, 1.0]],
        );
        assert!(matches!(
            evaluate_linear_action_samples(
                &add(square("x"), square("y")),
                &action,
                &sample_batch,
                sampling_profile()
            ),
            Err(LinearActionSamplingErrorV1::GeneratorCountLimitExceeded {
                actual: 129,
                maximum: 128
            })
        ));
    }

    #[test]
    fn overcomplete_direction_probes_detect_inconsistent_finite_difference_gradient() {
        // A cubic cross-term has direction-dependent central-difference bias.
        // Two orthogonal probes alone always define a square solvable system;
        // a third independent probe makes the inconsistency observable.
        let expression = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Var("y".to_string())),
            )),
            Box::new(Expr::Const(3.0)),
        );
        let sample = LinearActionSampleV1 {
            sample_id: "overcomplete-cubic".to_string(),
            state: vec![1.0, 1.0],
            direction_basis: vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ],
        };
        let profile = LinearActionSamplingProfileV1 {
            finite_difference_step: 0.1,
            maximum_samples: 1,
            minimum_informative_samples: 1,
            maximum_directional_reconstruction_relative_residual: 1.0e-8,
            ..sampling_profile()
        };
        assert!(matches!(
            evaluate_linear_action_samples(
                &expression,
                &canonical_action(),
                &[sample],
                profile
            ),
            Err(LinearActionSamplingErrorV1::GradientReconstructionResidualExceeded { .. })
        ));
    }

    #[test]
    fn overcomplete_zero_gradient_remains_uninformative() {
        let sample = LinearActionSampleV1 {
            sample_id: "overcomplete-constant".to_string(),
            state: vec![1.0, 2.0],
            direction_basis: vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ],
        };
        let profile = LinearActionSamplingProfileV1 {
            maximum_samples: 1,
            minimum_informative_samples: 1,
            ..sampling_profile()
        };
        assert!(matches!(
            evaluate_linear_action_samples(
                &Expr::Const(3.0),
                &canonical_action(),
                &[sample],
                profile
            ),
            Err(LinearActionSamplingErrorV1::InsufficientInformativeSamples { .. })
        ));
    }

    #[test]
    fn sampled_linear_action_rejects_too_few_direction_probes() {
        let states = [[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]];
        let mut batch = samples(&states, [[1.0, 0.0], [0.0, 1.0]]);
        for sample in &mut batch {
            sample.direction_basis.pop();
        }
        assert!(matches!(
            evaluate_linear_action_samples(
                &add(square("x"), square("y")),
                &canonical_action(),
                &batch,
                sampling_profile()
            ),
            Err(LinearActionSamplingErrorV1::DirectionCountOutsideBounds {
                minimum: 2,
                actual: 1,
                ..
            })
        ));
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
    fn sampled_linear_action_rejects_invalid_reconstruction_residual_budget() {
        let states = [[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]];
        let profile = LinearActionSamplingProfileV1 {
            maximum_directional_reconstruction_relative_residual: 1.01,
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
    fn sampled_linear_action_rejects_expression_variable_budget_excess() {
        let states = [[0.7, 1.1], [1.2, 0.8], [1.7, 2.1], [2.3, 1.4]];
        let profile = LinearActionSamplingProfileV1 {
            maximum_expression_variables: 1,
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
            Err(LinearActionSamplingErrorV1::ExpressionVariableBudgetExceeded {
                actual: 2,
                maximum: 1
            })
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
