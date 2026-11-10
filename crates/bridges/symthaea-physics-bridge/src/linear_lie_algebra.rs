// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded numerical qualification of admitted infinitesimal generator families.
//!
//! Every supplied generator is first whitened by the admitted metric and
//! Frobenius-normalized. Rank, commutator closure and Jacobi checks are then
//! performed in that orthonormalized metric representation. The returned
//! structure constants refer to this explicitly reported normalized basis,
//! not silently to the caller's unnormalized matrices.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::linear_group_action::AdmittedLinearGroupActionV1;

/// Numerical policy for qualifying a finite-dimensional Lie-algebra basis.
///
/// The default generator cap (32) and corresponding pair / triple-operation
/// caps keep dense SVD and Jacobi work bounded independently of caller data.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearLieAlgebraProfileV1 {
    pub maximum_generators: usize,
    pub maximum_pairwise_brackets: usize,
    pub maximum_jacobi_scalar_operations: usize,
    /// Singular-value cutoff relative to the largest singular value of
    /// Frobenius-normalized metric-whitened generator columns.
    pub relative_rank_tolerance: f64,
    pub maximum_basis_condition_number: f64,
    pub maximum_bracket_closure_relative_defect: f64,
    pub maximum_jacobi_relative_defect: f64,
}

const HARD_MAX_LIE_GENERATORS: usize = 32;
const HARD_MAX_PAIRWISE_BRACKETS: usize = 496;
const HARD_MAX_JACOBI_SCALAR_OPERATIONS: usize = 25_000_000;

impl Default for LinearLieAlgebraProfileV1 {
    fn default() -> Self {
        Self {
            maximum_generators: 32,
            maximum_pairwise_brackets: 496,
            maximum_jacobi_scalar_operations: 25_000_000,
            relative_rank_tolerance: 1e-10,
            maximum_basis_condition_number: 1e8,
            maximum_bracket_closure_relative_defect: 1e-9,
            maximum_jacobi_relative_defect: 1e-8,
        }
    }
}

/// Fail-closed reasons why a supplied generator family was not qualified.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearLieAlgebraQualificationErrorV1 {
    InvalidProfile,
    NoGenerators,
    GeneratorBudgetExceeded { actual: usize, maximum: usize },
    PairBudgetExceeded { actual: usize, maximum: usize },
    JacobiBudgetExceeded { actual: usize, maximum: usize },
    BasisDimensionTooSmall { coordinate_dimension: usize, generator_count: usize },
    GeneratorWhiteningFailed { generator_id: String },
    ZeroWhitenedGenerator { generator_id: String },
    NonFiniteWhitenedGenerator { generator_id: String },
    BasisRankDeficient {
        estimated_rank: usize,
        generator_count: usize,
        smallest_relative_singular_value: f64,
    },
    BasisIllConditioned { condition_number: f64, maximum: f64 },
    BasisFactorizationFailed,
    BracketProjectionFailed { left_generator_id: String, right_generator_id: String },
    BracketNotClosed {
        left_generator_id: String,
        right_generator_id: String,
        relative_defect: f64,
        maximum: f64,
    },
    JacobiIdentityFailed {
        first_generator_id: String,
        second_generator_id: String,
        third_generator_id: String,
        relative_defect: f64,
        maximum: f64,
    },
}

impl fmt::Display for LinearLieAlgebraQualificationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LinearLieAlgebraQualificationErrorV1 {}

/// Scale factors that map a normalized whitened generator back to its
/// original metric-whitened magnitude:
///
/// B_i = input_scale_i * whitened_max_abs_i * scaled_frobenius_norm_i * C_i,
/// where B_i = L^T A_i L^-T. Keeping the factors separate avoids overflow
/// in their product.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearLieGeneratorNormalizationV1 {
    pub generator_id: String,
    /// Maximum absolute entry of the caller-supplied A_i before whitening.
    pub input_generator_maximum_absolute_entry: f64,
    /// Maximum absolute entry after whitening A_i / input_scale_i.
    pub whitened_maximum_absolute_entry: f64,
    /// Frobenius norm after dividing the whitened matrix by its maximum entry.
    pub whitened_scaled_frobenius_norm: f64,
}

/// Structure constants for one ordered pair i < j, in the returned normalized
/// metric-whitened basis C_k:
///
/// [C_i, C_j] = sum_k coefficients[k] * C_k.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearLieBracketStructureConstantsV1 {
    pub left_generator_id: String,
    pub right_generator_id: String,
    pub coefficients_in_normalized_basis: Vec<f64>,
    pub relative_closure_defect: f64,
}

/// Jacobi residual for one strictly ordered generator triple.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearLieJacobiDiagnosticV1 {
    pub first_generator_id: String,
    pub second_generator_id: String,
    pub third_generator_id: String,
    pub absolute_residual_norm: f64,
    pub normalization_term_norm: f64,
    pub relative_defect: f64,
}

/// Aggregate evidence for the numerical basis / closure / Jacobi qualification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearLieAlgebraDiagnosticsV1 {
    pub generator_count: usize,
    pub estimated_basis_rank: usize,
    pub singular_values: Vec<f64>,
    pub relative_rank_threshold: f64,
    pub estimated_basis_condition_number: f64,
    pub pairwise_bracket_count: usize,
    pub jacobi_triples_checked: usize,
    pub maximum_bracket_closure_relative_defect: f64,
    pub maximum_jacobi_relative_defect: f64,
}

/// An admitted, independent and numerically bracket-closed Lie-algebra
/// representation under the supplied tolerances.
///
/// The basis matrices are metric-whitened and Frobenius-normalized. The
/// structure constants are expressed relative to those exact returned
/// normalized basis matrices. This object is numerical evidence; it is not
/// an exact proof of a global Lie group or physical symmetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdmittedLinearLieAlgebraV1 {
    pub action_id: String,
    pub coordinate_basis_id: String,
    pub coordinate_names: Vec<String>,
    pub profile: LinearLieAlgebraProfileV1,
    pub generator_ids: Vec<String>,
    pub generator_normalizations: Vec<LinearLieGeneratorNormalizationV1>,
    /// Row-major normalized metric-whitened generators C_i.
    pub normalized_whitened_generators: Vec<Vec<Vec<f64>>>,
    pub structure_constants: Vec<LinearLieBracketStructureConstantsV1>,
    pub jacobi_diagnostics: Vec<LinearLieJacobiDiagnosticV1>,
    pub diagnostics: LinearLieAlgebraDiagnosticsV1,
}

impl AdmittedLinearLieAlgebraV1 {
    pub fn generator_count(&self) -> usize {
        self.generator_ids.len()
    }

    pub fn is_numerically_abelian(&self) -> bool {
        self.structure_constants
            .iter()
            .all(|bracket| {
                bracket
                    .coefficients_in_normalized_basis
                    .iter()
                    .all(|value| value.abs() <= self.profile.maximum_bracket_closure_relative_defect)
            })
    }
}

/// Qualify the already-admitted generators as an independent, numerically
/// bracket-closed finite-dimensional Lie-algebra basis.
impl AdmittedLinearGroupActionV1 {
    pub fn try_qualify_lie_algebra(
        &self,
        profile: LinearLieAlgebraProfileV1,
    ) -> Result<AdmittedLinearLieAlgebraV1, LinearLieAlgebraQualificationErrorV1> {
        validate_profile(profile)?;

        let declaration = self.action();
        let dimension = declaration.coordinate_names.len();
        let generator_count = declaration.generators.len();
        if generator_count == 0 {
            return Err(LinearLieAlgebraQualificationErrorV1::NoGenerators);
        }
        if generator_count > profile.maximum_generators {
            return Err(LinearLieAlgebraQualificationErrorV1::GeneratorBudgetExceeded {
                actual: generator_count,
                maximum: profile.maximum_generators,
            });
        }
        if generator_count > dimension.saturating_mul(dimension) {
            return Err(LinearLieAlgebraQualificationErrorV1::BasisDimensionTooSmall {
                coordinate_dimension: dimension,
                generator_count,
            });
        }

        let pair_count = generator_count
            .checked_mul(generator_count.saturating_sub(1))
            .map(|value| value / 2)
            .ok_or(LinearLieAlgebraQualificationErrorV1::PairBudgetExceeded {
                actual: usize::MAX,
                maximum: profile.maximum_pairwise_brackets,
            })?;
        if pair_count > profile.maximum_pairwise_brackets {
            return Err(LinearLieAlgebraQualificationErrorV1::PairBudgetExceeded {
                actual: pair_count,
                maximum: profile.maximum_pairwise_brackets,
            });
        }

        let jacobi_triple_count = choose_three(generator_count);
        let jacobi_operations = jacobi_triple_count
            .checked_mul(generator_count)
            .and_then(|value| value.checked_mul(generator_count))
            .and_then(|value| value.checked_mul(3))
            .ok_or(LinearLieAlgebraQualificationErrorV1::JacobiBudgetExceeded {
                actual: usize::MAX,
                maximum: profile.maximum_jacobi_scalar_operations,
            })?;
        if jacobi_operations > profile.maximum_jacobi_scalar_operations {
            return Err(LinearLieAlgebraQualificationErrorV1::JacobiBudgetExceeded {
                actual: jacobi_operations,
                maximum: profile.maximum_jacobi_scalar_operations,
            });
        }

        let mut normalized_generators = Vec::with_capacity(generator_count);
        let mut normalizations = Vec::with_capacity(generator_count);
        for generator in &declaration.generators {
            let (whitened, input_generator_maximum_absolute_entry) = whiten_generator(
                &generator.matrix,
                self.scaled_metric_cholesky_lower(),
            )
            .ok_or_else(|| LinearLieAlgebraQualificationErrorV1::GeneratorWhiteningFailed {
                generator_id: generator.id.clone(),
            })?;
            let whitened_maximum_absolute_entry = matrix_max_abs(&whitened);
            if !whitened_maximum_absolute_entry.is_finite() {
                return Err(LinearLieAlgebraQualificationErrorV1::NonFiniteWhitenedGenerator {
                    generator_id: generator.id.clone(),
                });
            }
            if whitened_maximum_absolute_entry == 0.0 {
                return Err(LinearLieAlgebraQualificationErrorV1::ZeroWhitenedGenerator {
                    generator_id: generator.id.clone(),
                });
            }
            let whitened_scaled_frobenius_norm =
                matrix_scaled_frobenius_norm(&whitened, whitened_maximum_absolute_entry);
            if !whitened_scaled_frobenius_norm.is_finite()
                || whitened_scaled_frobenius_norm == 0.0
            {
                return Err(LinearLieAlgebraQualificationErrorV1::NonFiniteWhitenedGenerator {
                    generator_id: generator.id.clone(),
                });
            }
            let normalized = whitened
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|value| {
                            (value / whitened_maximum_absolute_entry)
                                / whitened_scaled_frobenius_norm
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            normalized_generators.push(normalized);
            normalizations.push(LinearLieGeneratorNormalizationV1 {
                generator_id: generator.id.clone(),
                input_generator_maximum_absolute_entry,
                whitened_maximum_absolute_entry,
                whitened_scaled_frobenius_norm,
            });
        }

        let flat_dimension = dimension
            .checked_mul(dimension)
            .ok_or(LinearLieAlgebraQualificationErrorV1::BasisDimensionTooSmall {
                coordinate_dimension: dimension,
                generator_count,
            })?;
        let design = DMatrix::from_fn(flat_dimension, generator_count, |row, column| {
            let matrix_row = row / dimension;
            let matrix_column = row % dimension;
            normalized_generators[column][matrix_row][matrix_column]
        });

        let svd = design.clone().svd(true, true);
        let singular_values = svd.singular_values.iter().copied().collect::<Vec<_>>();
        if singular_values.len() != generator_count {
            return Err(LinearLieAlgebraQualificationErrorV1::BasisRankDeficient {
                estimated_rank: singular_values.len(),
                generator_count,
                smallest_relative_singular_value: 0.0,
            });
        }
        let maximum_singular_value = singular_values.iter().copied().fold(0.0_f64, f64::max);
        let minimum_singular_value = singular_values.iter().copied().fold(f64::INFINITY, f64::min);
        let smallest_relative_singular_value = if maximum_singular_value == 0.0 {
            0.0
        } else {
            minimum_singular_value / maximum_singular_value
        };
        let relative_rank_threshold = profile.relative_rank_tolerance;
        let estimated_basis_condition_number = maximum_singular_value / minimum_singular_value;

        if !estimated_basis_condition_number.is_finite()
            || smallest_relative_singular_value <= relative_rank_threshold
        {
            let estimated_rank = singular_values
                .iter()
                .filter(|value| **value > maximum_singular_value * relative_rank_threshold)
                .count();
            return Err(LinearLieAlgebraQualificationErrorV1::BasisRankDeficient {
                estimated_rank,
                generator_count,
                smallest_relative_singular_value,
            });
        }
        if estimated_basis_condition_number > profile.maximum_basis_condition_number {
            return Err(LinearLieAlgebraQualificationErrorV1::BasisIllConditioned {
                condition_number: estimated_basis_condition_number,
                maximum: profile.maximum_basis_condition_number,
            });
        }

        let solve_tolerance = maximum_singular_value * relative_rank_threshold;
        let mut bracket_tensor = vec![
            vec![vec![0.0_f64; generator_count]; generator_count];
            generator_count
        ];
        let mut structure_constants = Vec::with_capacity(pair_count);
        let mut maximum_bracket_closure_relative_defect = 0.0_f64;

        for (left_index, left_generator) in normalized_generators.iter().enumerate() {
            for (right_index, right_generator) in
                normalized_generators.iter().enumerate().skip(left_index + 1)
            {
                let commutator = matrix_commutator(left_generator, right_generator);
                let commutator_vector = flatten_matrix(&commutator);
                let commutator_norm = stable_norm(&commutator_vector);
                let coefficients = if commutator_norm == 0.0 {
                    DVector::zeros(generator_count)
                } else {
                    svd.solve(&DVector::from_vec(commutator_vector.clone()), solve_tolerance)
                        .map_err(|_| LinearLieAlgebraQualificationErrorV1::BracketProjectionFailed {
                            left_generator_id: declaration.generators[left_index].id.clone(),
                            right_generator_id: declaration.generators[right_index].id.clone(),
                        })?
                };
                let reconstructed = &design * &coefficients;
                let mut residual_norm = 0.0_f64;
                for (actual, estimated) in commutator_vector.iter().zip(reconstructed.iter()) {
                    residual_norm = residual_norm.hypot(actual - estimated);
                }
                let relative_defect = if commutator_norm == 0.0 {
                    0.0
                } else {
                    residual_norm / commutator_norm
                };
                if !relative_defect.is_finite()
                    || relative_defect > profile.maximum_bracket_closure_relative_defect
                {
                    return Err(LinearLieAlgebraQualificationErrorV1::BracketNotClosed {
                        left_generator_id: declaration.generators[left_index].id.clone(),
                        right_generator_id: declaration.generators[right_index].id.clone(),
                        relative_defect,
                        maximum: profile.maximum_bracket_closure_relative_defect,
                    });
                }
                maximum_bracket_closure_relative_defect =
                    maximum_bracket_closure_relative_defect.max(relative_defect);

                let coefficients_in_normalized_basis = coefficients.iter().copied().collect::<Vec<_>>();
                for (basis_index, coefficient) in coefficients_in_normalized_basis.iter().enumerate() {
                    bracket_tensor[left_index][right_index][basis_index] = *coefficient;
                    bracket_tensor[right_index][left_index][basis_index] = -*coefficient;
                }
                structure_constants.push(LinearLieBracketStructureConstantsV1 {
                    left_generator_id: declaration.generators[left_index].id.clone(),
                    right_generator_id: declaration.generators[right_index].id.clone(),
                    coefficients_in_normalized_basis,
                    relative_closure_defect: relative_defect,
                });
            }
        }

        let mut jacobi_diagnostics = Vec::with_capacity(jacobi_triple_count);
        let mut maximum_jacobi_relative_defect = 0.0_f64;
        for (first, first_row) in bracket_tensor.iter().enumerate() {
            for (second, _) in first_row.iter().enumerate().skip(first + 1) {
                for (third, _) in first_row.iter().enumerate().skip(second + 1) {
                    let mut jacobi_norm = 0.0_f64;
                    let mut terms_norm = 0.0_f64;
                    for (output, _) in declaration.generators.iter().enumerate() {
                        let mut jacobi_component = 0.0_f64;
                        for (inner, _) in declaration.generators.iter().enumerate() {
                            let term_one = bracket_tensor[second][third][inner]
                                * bracket_tensor[first][inner][output];
                            let term_two = bracket_tensor[third][first][inner]
                                * bracket_tensor[second][inner][output];
                            let term_three = bracket_tensor[first][second][inner]
                                * bracket_tensor[third][inner][output];
                            jacobi_component += term_one + term_two + term_three;
                            terms_norm = terms_norm.hypot(term_one);
                            terms_norm = terms_norm.hypot(term_two);
                            terms_norm = terms_norm.hypot(term_three);
                        }
                        jacobi_norm = jacobi_norm.hypot(jacobi_component);
                    }
                    let relative_defect = if terms_norm == 0.0 {
                        0.0
                    } else {
                        jacobi_norm / terms_norm
                    };
                    if !relative_defect.is_finite()
                        || relative_defect > profile.maximum_jacobi_relative_defect
                    {
                        return Err(LinearLieAlgebraQualificationErrorV1::JacobiIdentityFailed {
                            first_generator_id: declaration.generators[first].id.clone(),
                            second_generator_id: declaration.generators[second].id.clone(),
                            third_generator_id: declaration.generators[third].id.clone(),
                            relative_defect,
                            maximum: profile.maximum_jacobi_relative_defect,
                        });
                    }
                    maximum_jacobi_relative_defect =
                        maximum_jacobi_relative_defect.max(relative_defect);
                    jacobi_diagnostics.push(LinearLieJacobiDiagnosticV1 {
                        first_generator_id: declaration.generators[first].id.clone(),
                        second_generator_id: declaration.generators[second].id.clone(),
                        third_generator_id: declaration.generators[third].id.clone(),
                        absolute_residual_norm: jacobi_norm,
                        normalization_term_norm: terms_norm,
                        relative_defect,
                    });
                }
            }
        }

        Ok(AdmittedLinearLieAlgebraV1 {
            action_id: declaration.action_id.clone(),
            coordinate_basis_id: declaration.coordinate_basis_id.clone(),
            coordinate_names: declaration.coordinate_names.clone(),
            profile,
            generator_ids: declaration.generators.iter().map(|generator| generator.id.clone()).collect(),
            generator_normalizations: normalizations,
            normalized_whitened_generators: normalized_generators,
            structure_constants,
            jacobi_diagnostics,
            diagnostics: LinearLieAlgebraDiagnosticsV1 {
                generator_count,
                estimated_basis_rank: generator_count,
                singular_values,
                relative_rank_threshold,
                estimated_basis_condition_number,
                pairwise_bracket_count: pair_count,
                jacobi_triples_checked: jacobi_triple_count,
                maximum_bracket_closure_relative_defect,
                maximum_jacobi_relative_defect,
            },
        })
    }
}

fn validate_profile(profile: LinearLieAlgebraProfileV1) -> Result<(), LinearLieAlgebraQualificationErrorV1> {
    if profile.maximum_generators == 0
        || profile.maximum_generators > HARD_MAX_LIE_GENERATORS
        || profile.maximum_pairwise_brackets == 0
        || profile.maximum_pairwise_brackets > HARD_MAX_PAIRWISE_BRACKETS
        || profile.maximum_jacobi_scalar_operations == 0
        || profile.maximum_jacobi_scalar_operations > HARD_MAX_JACOBI_SCALAR_OPERATIONS
        || !profile.relative_rank_tolerance.is_finite()
        || !(0.0..=1.0).contains(&profile.relative_rank_tolerance)
        || profile.relative_rank_tolerance == 0.0
        || !profile.maximum_basis_condition_number.is_finite()
        || profile.maximum_basis_condition_number < 1.0
        || !profile.maximum_bracket_closure_relative_defect.is_finite()
        || !(0.0..=1.0).contains(&profile.maximum_bracket_closure_relative_defect)
        || !profile.maximum_jacobi_relative_defect.is_finite()
        || !(0.0..=1.0).contains(&profile.maximum_jacobi_relative_defect)
    {
        return Err(LinearLieAlgebraQualificationErrorV1::InvalidProfile);
    }
    Ok(())
}

fn choose_three(value: usize) -> usize {
    if value < 3 {
        0
    } else {
        value
            .saturating_mul(value - 1)
            .saturating_mul(value - 2)
            / 6
    }
}

/// Metric-whiten A as L^T A L^-T, where G / scale = L L^T.
fn whiten_generator(
    generator: &[Vec<f64>],
    lower: &[Vec<f64>],
) -> Option<(Vec<Vec<f64>>, f64)> {
    let dimension = generator.len();
    let generator_scale = generator
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if generator_scale == 0.0 || !generator_scale.is_finite() {
        return None;
    }

    let mut left_product = vec![vec![0.0_f64; dimension]; dimension];
    for (row_index, output_row) in left_product.iter_mut().enumerate() {
        for (column_index, output) in output_row.iter_mut().enumerate() {
            let mut value = 0.0_f64;
            for (inner_index, generator_row) in generator.iter().enumerate() {
                value += lower[inner_index][row_index] * (generator_row[column_index] / generator_scale);
            }
            if !value.is_finite() {
                return None;
            }
            *output = value;
        }
    }

    let mut whitened = vec![vec![0.0_f64; dimension]; dimension];
    for (row_index, left_row) in left_product.iter().enumerate() {
        let mut solved_row = Vec::with_capacity(dimension);
        for (column_index, left_entry) in left_row.iter().enumerate() {
            let correction = solved_row
                .iter()
                .zip(lower[column_index].iter())
                .take(column_index)
                .map(|(value, lower_value)| *value * *lower_value)
                .sum::<f64>();
            let value = (*left_entry - correction) / lower[column_index][column_index];
            if !value.is_finite() {
                return None;
            }
            solved_row.push(value);
        }
        whitened[row_index] = solved_row;
    }
    Some((whitened, generator_scale))
}

fn matrix_max_abs(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
}

fn matrix_scaled_frobenius_norm(matrix: &[Vec<f64>], scale: f64) -> f64 {
    matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |norm, value| norm.hypot(value / scale))
}

fn matrix_commutator(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let dimension = left.len();
    let mut result = vec![vec![0.0_f64; dimension]; dimension];
    for (row_index, result_row) in result.iter_mut().enumerate() {
        for (column_index, result_entry) in result_row.iter_mut().enumerate() {
            let mut value = 0.0_f64;
            for (inner_index, left_row_entry) in left[row_index].iter().enumerate() {
                value += *left_row_entry * right[inner_index][column_index]
                    - right[row_index][inner_index] * left[inner_index][column_index];
            }
            *result_entry = value;
        }
    }
    result
}

fn flatten_matrix(matrix: &[Vec<f64>]) -> Vec<f64> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}

fn stable_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_group_action::{
        LinearGeneratorV1, LinearGroupActionAdmissionProfileV1, LinearGroupActionV1,
    };
    use nalgebra::Matrix3;

    fn action(
        action_id: &str,
        coordinate_names: Vec<&str>,
        metric: Vec<Vec<f64>>,
        generators: Vec<(&str, Vec<Vec<f64>>)>,
    ) -> AdmittedLinearGroupActionV1 {
        LinearGroupActionV1 {
            action_id: action_id.to_string(),
            coordinate_basis_id: format!("{action_id}-basis"),
            coordinate_unit_ids: coordinate_names.iter().map(|_| "1".to_string()).collect(),
            coordinate_names: coordinate_names.iter().map(|name| (*name).to_string()).collect(),
            metric,
            generators: generators
                .into_iter()
                .map(|(id, matrix)| LinearGeneratorV1 { id: id.to_string(), matrix })
                .collect(),
        }
        .try_admit(LinearGroupActionAdmissionProfileV1::default())
        .unwrap()
    }

    fn so2_action() -> AdmittedLinearGroupActionV1 {
        action(
            "so2",
            vec!["x", "y"],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![("J", vec![vec![0.0, -1.0], vec![1.0, 0.0]])],
        )
    }

    fn so3_generators() -> Vec<(&'static str, Vec<Vec<f64>>)> {
        vec![
            ("Jx", vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, -1.0], vec![0.0, 1.0, 0.0]]),
            ("Jy", vec![vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 0.0], vec![-1.0, 0.0, 0.0]]),
            ("Jz", vec![vec![0.0, -1.0, 0.0], vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]),
        ]
    }

    fn so3_action() -> AdmittedLinearGroupActionV1 {
        action(
            "so3",
            vec!["x", "y", "z"],
            vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
            so3_generators(),
        )
    }

    fn to_vec(matrix: &Matrix3<f64>) -> Vec<Vec<f64>> {
        matrix.row_iter().map(|row| row.iter().copied().collect()).collect()
    }

    #[test]
    fn qualifies_one_dimensional_so2_algebra() {
        let qualified = so2_action()
            .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default())
            .unwrap();
        assert_eq!(qualified.diagnostics.estimated_basis_rank, 1);
        assert_eq!(qualified.structure_constants.len(), 0);
        assert_eq!(qualified.jacobi_diagnostics.len(), 0);
        assert!(qualified.is_numerically_abelian());
    }

    #[test]
    fn so3_structure_constants_use_conventional_commutator_orientation() {
        let qualified = so3_action()
            .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default())
            .unwrap();
        let bracket = &qualified.structure_constants[0];
        assert_eq!(bracket.left_generator_id, "Jx");
        assert_eq!(bracket.right_generator_id, "Jy");
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (bracket.coefficients_in_normalized_basis[2] - expected).abs() < 1e-12
        );
    }

    #[test]
    fn qualifies_so3_and_reports_structure_constants_and_jacobi() {
        let qualified = so3_action()
            .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default())
            .unwrap();
        assert_eq!(qualified.diagnostics.estimated_basis_rank, 3);
        assert_eq!(qualified.structure_constants.len(), 3);
        assert_eq!(qualified.jacobi_diagnostics.len(), 1);
        assert!(qualified.diagnostics.maximum_bracket_closure_relative_defect < 1e-12);
        assert!(qualified.diagnostics.maximum_jacobi_relative_defect < 1e-12);
    }

    #[test]
    fn jacobi_check_measures_nonzero_terms_that_cancel() {
        // A non-orthogonal generator basis of so(3) makes the nested Jacobi
        // terms individually nonzero; the three-term sum must still vanish.
        let generators = vec![
            so3_generators()[0].clone(),
            (
                "Jy-plus-Jz",
                {
                    let jy = so3_generators()[1].1.clone();
                    let jz = so3_generators()[2].1.clone();
                    jy.iter()
                        .zip(jz.iter())
                        .map(|(left, right)| {
                            left.iter().zip(right.iter()).map(|(a, b)| a + b).collect()
                        })
                        .collect()
                },
            ),
            so3_generators()[2].clone(),
        ];
        let qualified = action(
            "so3-mixed-generator-basis",
            vec!["x", "y", "z"],
            vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
            generators,
        )
        .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default())
        .unwrap();
        assert_eq!(qualified.jacobi_diagnostics.len(), 1);
        let jacobi = &qualified.jacobi_diagnostics[0];
        assert!(jacobi.normalization_term_norm > 0.1);
        assert!(jacobi.absolute_residual_norm < 1e-12);
        assert!(jacobi.relative_defect < 1e-12);
    }

    #[test]
    fn qualifies_direct_sum_so2_plus_so2() {
        let generators = vec![
            ("Jxy", vec![
                vec![0.0, -1.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ]),
            ("Jzw", vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, -1.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ]),
        ];
        let qualified = action(
            "so2-plus-so2",
            vec!["x", "y", "z", "w"],
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
            generators,
        )
        .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default())
        .unwrap();
        assert!(qualified.is_abelian());
        assert_eq!(qualified.structure_constants.len(), 1);
        assert!(qualified.structure_constants[0]
            .coefficients_in_normalized_basis
            .iter()
            .all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn generator_normalization_handles_extreme_input_scales() {
        for scale in [1.0e-300, 1.0e300] {
            let scaled = action(
                "scaled-so2",
                vec!["x", "y"],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                vec![(
                    "J-scaled",
                    vec![vec![0.0, -scale], vec![scale, 0.0]],
                )],
            )
            .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default())
            .unwrap();
            assert!(
                (scaled.generator_normalizations[0]
                    .input_generator_maximum_absolute_entry
                    - scale)
                    .abs()
                    / scale
                    < 1e-12
            );
            assert!(scaled.is_numerically_abelian());
        }
    }

    #[test]
    fn rejects_ill_conditioned_but_numerically_full_rank_generator_basis() {
        let mut generators = so3_generators();
        let mut near_copy = generators[0].1.clone();
        for (row, near_row) in near_copy.iter_mut().enumerate() {
            for (column, value) in near_row.iter_mut().enumerate() {
                *value += 1.0e-4 * generators[1].1[row][column];
            }
        }
        generators[2] = ("near-copy-1e-4", near_copy);
        let admitted = action(
            "ill-conditioned-so3",
            vec!["x", "y", "z"],
            vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
            generators,
        );
        let mut profile = LinearLieAlgebraProfileV1::default();
        profile.maximum_basis_condition_number = 1.0e3;
        assert!(matches!(
            admitted.try_qualify_lie_algebra(profile),
            Err(LinearLieAlgebraQualificationErrorV1::BasisIllConditioned { .. })
        ));
    }

    #[test]
    fn rejects_metric_skew_generators_that_are_not_bracket_closed() {
        let mut generators = so3_generators();
        generators.pop();
        assert!(matches!(
            action(
                "nonclosed-two-generator-plane",
                vec!["x", "y", "z"],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
                generators,
            )
            .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default()),
            Err(LinearLieAlgebraQualificationErrorV1::BracketNotClosed { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_generator_basis() {
        let mut generators = so3_generators();
        let duplicate = generators[0].1.clone();
        generators.push(("Jx-copy", duplicate));
        assert!(matches!(
            action(
                "duplicate-so3",
                vec!["x", "y", "z"],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
                generators,
            )
            .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default()),
            Err(LinearLieAlgebraQualificationErrorV1::BasisRankDeficient { .. })
        ));
    }

    #[test]
    fn rejects_nearly_dependent_generator_basis() {
        let mut generators = so3_generators();
        let mut near_copy = generators[0].1.clone();
        for row in 0..3 {
            for column in 0..3 {
                near_copy[row][column] += 1e-12 * generators[1].1[row][column];
            }
        }
        generators[2] = ("Jx-near-copy", near_copy);
        assert!(matches!(
            action(
                "near-dependent-so3",
                vec!["x", "y", "z"],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
                generators,
            )
            .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default()),
            Err(LinearLieAlgebraQualificationErrorV1::BasisRankDeficient { .. })
        ));
    }

    #[test]
    fn so3_basis_rank_closure_and_jacobi_are_covariant_under_nonorthogonal_coordinates() {
        let transform = Matrix3::new(
            2.0, 1.0, 0.5,
            0.0, 1.5, 0.25,
            0.0, 0.0, 0.75,
        );
        let inverse = transform.try_inverse().unwrap();
        let transformed_metric = inverse.transpose() * Matrix3::identity() * inverse;
        let transformed_generators = so3_generators()
            .into_iter()
            .map(|(id, matrix)| {
                let original = Matrix3::from_row_slice(&matrix.iter().flatten().copied().collect::<Vec<_>>());
                (id, to_vec(&(transform * original * inverse)))
            })
            .collect::<Vec<_>>();
        let canonical = so3_action()
            .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default())
            .unwrap();
        let transformed = action(
            "so3-sheared",
            vec!["X", "Y", "Z"],
            to_vec(&transformed_metric),
            transformed_generators,
        )
        .try_qualify_lie_algebra(LinearLieAlgebraProfileV1::default())
        .unwrap();

        assert_eq!(canonical.diagnostics.estimated_basis_rank, transformed.diagnostics.estimated_basis_rank);
        assert!(
            (canonical.diagnostics.maximum_bracket_closure_relative_defect
                - transformed.diagnostics.maximum_bracket_closure_relative_defect)
                .abs()
                < 1e-9
        );
        assert!(
            (canonical.diagnostics.maximum_jacobi_relative_defect
                - transformed.diagnostics.maximum_jacobi_relative_defect)
                .abs()
                < 1e-9
        );
        for (canonical_pair, transformed_pair) in canonical
            .structure_constants
            .iter()
            .zip(transformed.structure_constants.iter())
        {
            assert!(
                canonical_pair.coefficients_in_normalized_basis
                    .iter()
                    .zip(transformed_pair.coefficients_in_normalized_basis.iter())
                    .all(|(left, right)| (left - right).abs() < 1e-8)
            );
        }
    }

    #[test]
    fn rejects_profile_and_computational_budgets_before_factorization() {
        let action = so3_action();
        let mut invalid = LinearLieAlgebraProfileV1::default();
        invalid.relative_rank_tolerance = 0.0;
        assert!(matches!(
            action.try_qualify_lie_algebra(invalid),
            Err(LinearLieAlgebraQualificationErrorV1::InvalidProfile)
        ));

        let mut pair_limited = LinearLieAlgebraProfileV1::default();
        pair_limited.maximum_pairwise_brackets = 2;
        assert!(matches!(
            action.try_qualify_lie_algebra(pair_limited),
            Err(LinearLieAlgebraQualificationErrorV1::PairBudgetExceeded { .. })
        ));

        let mut jacobi_limited = LinearLieAlgebraProfileV1::default();
        jacobi_limited.maximum_jacobi_scalar_operations = 1;
        assert!(matches!(
            action.try_qualify_lie_algebra(jacobi_limited),
            Err(LinearLieAlgebraQualificationErrorV1::JacobiBudgetExceeded { .. })
        ));

        let mut over_hard_generator_ceiling = LinearLieAlgebraProfileV1::default();
        over_hard_generator_ceiling.maximum_generators = HARD_MAX_LIE_GENERATORS + 1;
        assert!(matches!(
            action.try_qualify_lie_algebra(over_hard_generator_ceiling),
            Err(LinearLieAlgebraQualificationErrorV1::InvalidProfile)
        ));

        let mut over_hard_jacobi_ceiling = LinearLieAlgebraProfileV1::default();
        over_hard_jacobi_ceiling.maximum_jacobi_scalar_operations =
            HARD_MAX_JACOBI_SCALAR_OPERATIONS + 1;
        assert!(matches!(
            action.try_qualify_lie_algebra(over_hard_jacobi_ceiling),
            Err(LinearLieAlgebraQualificationErrorV1::InvalidProfile)
        ));
    }
}
