// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Resource-bounded qualification of finite-dimensional matrix Lie-algebra bases.
//!
//! This module consumes an already admitted metric-isometric linear action. It
//! does not discover generators, integrate a global Lie group, identify
//! disconnected components, or establish physical / discrete-time symmetry.
//!
//! The numerical workspace whitens each generator with the admitted Cholesky
//! factor, then normalizes its Frobenius norm. Independence is qualified by
//! deterministic column-pivoted, twice-reorthogonalized modified Gram-Schmidt.
//! Bracket closure is checked by projection and reconstruction in that basis.
//! Jacobi is checked independently on the resulting structure constants.

use crate::linear_group_action::AdmittedLinearGroupActionV1;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Explicit numerical and resource policy for Lie-basis qualification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearLieAlgebraQualificationProfileV1 {
    /// Relative pivot threshold for independence of metric-whitened generators.
    pub rank_relative_tolerance: f64,
    /// Relative Frobenius residual allowed for each matrix commutator.
    pub bracket_closure_relative_tolerance: f64,
    /// Relative structure-constant residual allowed by the Jacobi check.
    pub jacobi_relative_tolerance: f64,
    /// A bracket below this normalized norm is recorded as numerically zero.
    pub zero_bracket_relative_tolerance: f64,
    /// Maximum number of unordered generator pairs to inspect.
    pub maximum_pair_checks: usize,
    /// Maximum number of unordered generator triples for Jacobi checks.
    pub maximum_jacobi_checks: usize,
    /// Maximum deterministic operation proxy for the complete qualification.
    pub maximum_work_units: usize,
}

impl Default for LinearLieAlgebraQualificationProfileV1 {
    fn default() -> Self {
        Self {
            rank_relative_tolerance: 1.0e-10,
            bracket_closure_relative_tolerance: 1.0e-9,
            jacobi_relative_tolerance: 1.0e-9,
            zero_bracket_relative_tolerance: 1.0e-14,
            maximum_pair_checks: 8_128,
            maximum_jacobi_checks: 50_000,
            maximum_work_units: 100_000_000,
        }
    }
}

/// Fail-closed preflight or numerical failures.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearLieAlgebraQualificationErrorV1 {
    InvalidProfile,
    PairBudgetExceeded {
        required: usize,
        maximum: usize,
    },
    JacobiBudgetExceeded {
        required: usize,
        maximum: usize,
    },
    WorkBudgetExceeded {
        required: usize,
        maximum: usize,
    },
    WorkEstimateOverflow,
    RankDeficient {
        numerical_rank: usize,
        generator_count: usize,
        unresolved_relative_pivot: f64,
    },
    NonFiniteWhitenedGenerator {
        generator_id: String,
    },
    ZeroWhitenedGenerator {
        generator_id: String,
    },
    NonFiniteBracket {
        left_generator_id: String,
        right_generator_id: String,
    },
    NonFiniteStructureConstant,
    NonFiniteJacobiResidual,
}

impl fmt::Display for LinearLieAlgebraQualificationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LinearLieAlgebraQualificationErrorV1 {}

/// Result status. Success is numerical evidence under the supplied profile,
/// not a formal proof or a global group-integration result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearLieAlgebraQualificationStatusV1 {
    NumericallyClosedAndJacobiConsistent,
    BracketClosureFailed,
    JacobiFailed,
}

/// Normalization and reconstruction details for one input generator.
///
/// Structure constants are expressed in the Frobenius-unit, metric-whitened
/// generator basis. The two scales below make this basis transformation
/// explicit and reproducible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LieGeneratorNormalizationV1 {
    pub generator_id: String,
    pub input_max_abs_scale: f64,
    pub metric_whitened_scaled_frobenius_norm: f64,
    pub whitened_metric_skew_relative_defect: f64,
}

/// One checked bracket [A_i, A_j].
///
/// structure_constants[k] is the coefficient of the normalized generator
/// whose identity is report.generator_ids[k].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LieBracketDiagnosticV1 {
    pub left_generator_id: String,
    pub right_generator_id: String,
    pub normalized_bracket_frobenius_norm: f64,
    pub relative_closure_defect: f64,
    pub classified_as_numerically_zero: bool,
    pub structure_constants: Vec<f64>,
}

/// Jacobi identity result for one ordered triple i < j < k.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LieJacobiDiagnosticV1 {
    pub first_generator_id: String,
    pub second_generator_id: String,
    pub third_generator_id: String,
    /// Local defect normalized by the magnitudes of the three nested brackets,
    /// unless those magnitudes are at the floating-point roundoff floor.
    pub relative_defect: f64,
    /// Sum of the norms of the three nested-bracket terms.
    pub local_term_scale: f64,
    /// Defect against the square of the largest structure constant. This
    /// remains interpretable when the three local terms are all near zero.
    pub global_scale_relative_defect: f64,
    /// True when local normalization would amplify roundoff; in that case
    /// relative_defect uses the global algebra scale and this field is true.
    pub used_global_scale_fallback: bool,
}

/// Aggregate diagnostics for the complete admitted generator family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearLieAlgebraDiagnosticsV1 {
    pub generator_count: usize,
    pub numerical_rank: usize,
    /// Pivot residuals in deterministic pivot order. Initial columns have unit
    /// Frobenius norm, so these values are dimensionless.
    pub relative_rank_pivots: Vec<f64>,
    /// Ratio of largest to smallest accepted pivot. This is a pivot diagnostic,
    /// not a spectral condition number.
    pub pivot_condition_estimate: f64,
    pub generator_normalizations: Vec<LieGeneratorNormalizationV1>,
    pub bracket_pair_count: usize,
    pub maximum_bracket_closure_defect: f64,
    pub jacobi_triple_count: usize,
    pub maximum_jacobi_defect: Option<f64>,
    pub estimated_work_units: usize,
}

/// Auditable numerical qualification of the declared generator basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearLieAlgebraQualificationReportV1 {
    pub schema_version: u32,
    pub action_id: String,
    pub coordinate_basis_id: String,
    /// Original caller order; all coefficient vectors use this order.
    pub generator_ids: Vec<String>,
    pub profile: LinearLieAlgebraQualificationProfileV1,
    pub status: LinearLieAlgebraQualificationStatusV1,
    pub diagnostics: LinearLieAlgebraDiagnosticsV1,
    pub brackets: Vec<LieBracketDiagnosticV1>,
    pub jacobi_checks: Vec<LieJacobiDiagnosticV1>,
}

/// Qualify independence, bracket closure, and Jacobi structure for an admitted
/// finite-dimensional linear action.
///
/// Whitening uses G / s = L L^T and B_i = L^T A_i L^-T. Each B_i is then
/// normalized to unit Frobenius norm. Returned structure constants are for
/// these normalized generators, not silently for the original matrix scales.
///
/// The routine rejects rank ambiguity rather than deleting dependent
/// generators. Bracket closure is measured for every unordered pair;
/// antisymmetry supplies the reverse ordered pair. Jacobi is evaluated from
/// reconstructed structure constants, not from a second matrix-commutator
/// calculation.
pub fn qualify_linear_lie_algebra(
    admitted: &AdmittedLinearGroupActionV1,
    profile: LinearLieAlgebraQualificationProfileV1,
) -> Result<LinearLieAlgebraQualificationReportV1, LinearLieAlgebraQualificationErrorV1> {
    validate_profile(profile)?;

    let action = admitted.action();
    let dimension = admitted.dimension();
    let generator_count = action.generators.len();
    let pair_count = generator_count
        .checked_mul(generator_count.saturating_sub(1))
        .ok_or(LinearLieAlgebraQualificationErrorV1::WorkEstimateOverflow)?
        / 2;
    let jacobi_count = generator_count
        .checked_mul(generator_count.saturating_sub(1))
        .and_then(|value| value.checked_mul(generator_count.saturating_sub(2)))
        .ok_or(LinearLieAlgebraQualificationErrorV1::WorkEstimateOverflow)?
        / 6;

    if pair_count > profile.maximum_pair_checks {
        return Err(LinearLieAlgebraQualificationErrorV1::PairBudgetExceeded {
            required: pair_count,
            maximum: profile.maximum_pair_checks,
        });
    }
    if jacobi_count > profile.maximum_jacobi_checks {
        return Err(LinearLieAlgebraQualificationErrorV1::JacobiBudgetExceeded {
            required: jacobi_count,
            maximum: profile.maximum_jacobi_checks,
        });
    }

    let estimated_work_units =
        estimate_work_units(dimension, generator_count, pair_count, jacobi_count)
            .ok_or(LinearLieAlgebraQualificationErrorV1::WorkEstimateOverflow)?;
    if estimated_work_units > profile.maximum_work_units {
        return Err(LinearLieAlgebraQualificationErrorV1::WorkBudgetExceeded {
            required: estimated_work_units,
            maximum: profile.maximum_work_units,
        });
    }

    let lower = admitted.scaled_metric_cholesky_lower();
    let mut normalized_generators = Vec::with_capacity(generator_count);
    let mut generator_normalizations = Vec::with_capacity(generator_count);

    for generator in &action.generators {
        let (normalized, input_max_abs_scale, whitened_norm, skew_defect) =
            whiten_and_normalize_generator(&generator.matrix, lower, &generator.id)?;
        normalized_generators.push(normalized);
        generator_normalizations.push(LieGeneratorNormalizationV1 {
            generator_id: generator.id.clone(),
            input_max_abs_scale,
            metric_whitened_scaled_frobenius_norm: whitened_norm,
            whitened_metric_skew_relative_defect: skew_defect,
        });
    }

    let (orthonormal_basis, triangular, permutation, relative_rank_pivots) =
        pivoted_modified_gram_schmidt(&normalized_generators, profile.rank_relative_tolerance)?;
    let minimum_pivot = relative_rank_pivots
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let maximum_pivot = relative_rank_pivots.iter().copied().fold(0.0_f64, f64::max);
    let pivot_condition_estimate = maximum_pivot / minimum_pivot;

    let generator_ids: Vec<String> = action
        .generators
        .iter()
        .map(|generator| generator.id.clone())
        .collect();

    let mut structure =
        vec![vec![vec![0.0_f64; generator_count]; generator_count]; generator_count];
    let mut brackets = Vec::with_capacity(pair_count);
    let mut maximum_bracket_closure_defect = 0.0_f64;

    for left_index in 0..generator_count {
        for right_index in (left_index + 1)..generator_count {
            let bracket = matrix_commutator(
                &normalized_generators[left_index],
                &normalized_generators[right_index],
                dimension,
            );
            let bracket_norm = vector_norm(&bracket);
            if !bracket_norm.is_finite() {
                return Err(LinearLieAlgebraQualificationErrorV1::NonFiniteBracket {
                    left_generator_id: generator_ids[left_index].clone(),
                    right_generator_id: generator_ids[right_index].clone(),
                });
            }

            let (coefficients, relative_defect) = project_and_solve(
                &bracket,
                &orthonormal_basis,
                &triangular,
                &permutation,
                profile.zero_bracket_relative_tolerance,
            );
            if !relative_defect.is_finite()
                || coefficients
                    .iter()
                    .any(|coefficient| !coefficient.is_finite())
            {
                return Err(LinearLieAlgebraQualificationErrorV1::NonFiniteStructureConstant);
            }
            let classified_as_numerically_zero =
                bracket_norm <= profile.zero_bracket_relative_tolerance;

            maximum_bracket_closure_defect = maximum_bracket_closure_defect.max(relative_defect);
            for (basis_index, coefficient) in coefficients.iter().enumerate() {
                structure[left_index][right_index][basis_index] = *coefficient;
                structure[right_index][left_index][basis_index] = -*coefficient;
            }

            brackets.push(LieBracketDiagnosticV1 {
                left_generator_id: generator_ids[left_index].clone(),
                right_generator_id: generator_ids[right_index].clone(),
                normalized_bracket_frobenius_norm: bracket_norm,
                relative_closure_defect: relative_defect,
                classified_as_numerically_zero,
                structure_constants: coefficients,
            });
        }
    }

    let closure_passed =
        maximum_bracket_closure_defect <= profile.bracket_closure_relative_tolerance;
    let maximum_structure_constant = structure
        .iter()
        .flat_map(|plane| plane.iter())
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let global_jacobi_scale = maximum_structure_constant * maximum_structure_constant;
    if !global_jacobi_scale.is_finite() {
        return Err(LinearLieAlgebraQualificationErrorV1::NonFiniteStructureConstant);
    }
    let jacobi_roundoff_floor =
        64.0 * f64::EPSILON * generator_count.max(1) as f64 * global_jacobi_scale;
    let mut jacobi_checks = Vec::new();
    let mut maximum_jacobi_defect = None;

    let status = if !closure_passed {
        LinearLieAlgebraQualificationStatusV1::BracketClosureFailed
    } else {
        jacobi_checks.reserve(jacobi_count);
        let mut maximum_defect = 0.0_f64;

        for first in 0..generator_count {
            for second in (first + 1)..generator_count {
                for third in (second + 1)..generator_count {
                    let first_term = jacobi_term(&structure, generator_count, first, second, third);
                    let second_term =
                        jacobi_term(&structure, generator_count, second, third, first);
                    let third_term = jacobi_term(&structure, generator_count, third, first, second);

                    let mut residual = vec![0.0_f64; generator_count];
                    for component in 0..generator_count {
                        residual[component] =
                            first_term[component] + second_term[component] + third_term[component];
                    }

                    let local_term_scale = vector_norm(&first_term)
                        + vector_norm(&second_term)
                        + vector_norm(&third_term);
                    let numerator = vector_norm(&residual);
                    let global_scale_relative_defect = if global_jacobi_scale == 0.0 {
                        if numerator == 0.0 { 0.0 } else { f64::INFINITY }
                    } else {
                        numerator / global_jacobi_scale
                    };
                    // A local ratio is unstable when all three nested brackets
                    // are at roundoff scale. Preserve the raw local scale, but
                    // normalize that case against the largest admitted
                    // structure constant squared. This avoids reporting a
                    // unit-sized Jacobi defect for an algebra whose exact
                    // Jacobi terms are zero and whose floating residue is tiny.
                    let used_global_scale_fallback = local_term_scale <= jacobi_roundoff_floor;
                    let defect = if used_global_scale_fallback {
                        global_scale_relative_defect
                    } else {
                        numerator / local_term_scale
                    };
                    if !defect.is_finite() || !global_scale_relative_defect.is_finite() {
                        return Err(LinearLieAlgebraQualificationErrorV1::NonFiniteJacobiResidual);
                    }
                    maximum_defect = maximum_defect.max(defect);
                    jacobi_checks.push(LieJacobiDiagnosticV1 {
                        first_generator_id: generator_ids[first].clone(),
                        second_generator_id: generator_ids[second].clone(),
                        third_generator_id: generator_ids[third].clone(),
                        relative_defect: defect,
                        local_term_scale,
                        global_scale_relative_defect,
                        used_global_scale_fallback,
                    });
                }
            }
        }

        maximum_jacobi_defect = Some(maximum_defect);
        if maximum_defect <= profile.jacobi_relative_tolerance {
            LinearLieAlgebraQualificationStatusV1::NumericallyClosedAndJacobiConsistent
        } else {
            LinearLieAlgebraQualificationStatusV1::JacobiFailed
        }
    };

    Ok(LinearLieAlgebraQualificationReportV1 {
        schema_version: 1,
        action_id: action.action_id.clone(),
        coordinate_basis_id: action.coordinate_basis_id.clone(),
        generator_ids,
        profile,
        status,
        diagnostics: LinearLieAlgebraDiagnosticsV1 {
            generator_count,
            numerical_rank: generator_count,
            relative_rank_pivots,
            pivot_condition_estimate,
            generator_normalizations,
            bracket_pair_count: pair_count,
            maximum_bracket_closure_defect,
            jacobi_triple_count: jacobi_checks.len(),
            maximum_jacobi_defect,
            estimated_work_units,
        },
        brackets,
        jacobi_checks,
    })
}

const HARD_MAXIMUM_PAIR_CHECKS: usize = 8_128;
const HARD_MAXIMUM_JACOBI_CHECKS: usize = 341_376;
const HARD_MAXIMUM_WORK_UNITS: usize = 100_000_000;

fn validate_profile(
    profile: LinearLieAlgebraQualificationProfileV1,
) -> Result<(), LinearLieAlgebraQualificationErrorV1> {
    let valid_tolerance = |value: f64| value.is_finite() && (0.0..=1.0).contains(&value);
    if !valid_tolerance(profile.rank_relative_tolerance)
        || !valid_tolerance(profile.bracket_closure_relative_tolerance)
        || !valid_tolerance(profile.jacobi_relative_tolerance)
        || !valid_tolerance(profile.zero_bracket_relative_tolerance)
        || profile.rank_relative_tolerance == 0.0
        || profile.maximum_pair_checks > HARD_MAXIMUM_PAIR_CHECKS
        || profile.maximum_jacobi_checks > HARD_MAXIMUM_JACOBI_CHECKS
        || profile.maximum_work_units == 0
        || profile.maximum_work_units > HARD_MAXIMUM_WORK_UNITS
    {
        return Err(LinearLieAlgebraQualificationErrorV1::InvalidProfile);
    }
    Ok(())
}

fn estimate_work_units(
    dimension: usize,
    generator_count: usize,
    pair_count: usize,
    jacobi_count: usize,
) -> Option<usize> {
    let n2 = dimension.checked_mul(dimension)?;
    let n3 = n2.checked_mul(dimension)?;
    let m2 = generator_count.checked_mul(generator_count)?;
    let m3 = m2.checked_mul(generator_count)?;

    let whitening = generator_count.checked_mul(n3)?;
    let rank = m2.checked_mul(n2)?.checked_mul(4)?;
    let bracket_projection = pair_count.checked_mul(n3)?.checked_mul(2)?;
    let bracket_reconstruction = pair_count.checked_mul(m2)?.checked_mul(2)?;
    let brackets = bracket_projection.checked_add(bracket_reconstruction)?;
    let jacobi = jacobi_count.checked_mul(m2)?.checked_mul(6)?;
    whitening
        .checked_add(rank)?
        .checked_add(brackets)?
        .checked_add(jacobi)?
        .checked_add(m3)
}

fn vector_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

fn whiten_and_normalize_generator(
    generator: &[Vec<f64>],
    lower: &[Vec<f64>],
    generator_id: &str,
) -> Result<(Vec<f64>, f64, f64, f64), LinearLieAlgebraQualificationErrorV1> {
    let dimension = generator.len();
    let input_scale = generator
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if !input_scale.is_finite() || input_scale == 0.0 {
        return Err(
            LinearLieAlgebraQualificationErrorV1::ZeroWhitenedGenerator {
                generator_id: generator_id.to_string(),
            },
        );
    }

    // Compute L^T (A / input_scale), then solve X L^T = left_product.
    // This avoids explicitly constructing L^-T.
    let mut left_product = vec![vec![0.0_f64; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            let mut value = 0.0_f64;
            for inner in 0..dimension {
                value += lower[inner][row] * (generator[inner][column] / input_scale);
            }
            left_product[row][column] = value;
        }
    }

    let mut whitened = vec![vec![0.0_f64; dimension]; dimension];
    for row in 0..dimension {
        let mut solved_row = vec![0.0_f64; dimension];
        for column in 0..dimension {
            let mut correction = 0.0_f64;
            for inner in 0..column {
                correction += solved_row[inner] * lower[column][inner];
            }
            solved_row[column] = (left_product[row][column] - correction) / lower[column][column];
        }
        whitened[row] = solved_row;
    }

    if whitened.iter().flatten().any(|value| !value.is_finite()) {
        return Err(
            LinearLieAlgebraQualificationErrorV1::NonFiniteWhitenedGenerator {
                generator_id: generator_id.to_string(),
            },
        );
    }

    let whitened_entries: Vec<f64> = whitened.iter().flatten().copied().collect();
    let whitened_norm = vector_norm(&whitened_entries);
    if !whitened_norm.is_finite() {
        return Err(
            LinearLieAlgebraQualificationErrorV1::NonFiniteWhitenedGenerator {
                generator_id: generator_id.to_string(),
            },
        );
    }
    if whitened_norm == 0.0 {
        return Err(
            LinearLieAlgebraQualificationErrorV1::ZeroWhitenedGenerator {
                generator_id: generator_id.to_string(),
            },
        );
    }

    let mut skew_norm = 0.0_f64;
    let mut normalized = Vec::with_capacity(dimension * dimension);
    for (row, row_values) in whitened.iter().take(dimension).enumerate() {
        for (column, value) in row_values.iter().take(dimension).enumerate() {
            skew_norm = skew_norm.hypot(*value + whitened[column][row]);
            normalized.push(*value / whitened_norm);
        }
    }
    let skew_defect = skew_norm / (2.0 * whitened_norm);
    if !skew_defect.is_finite() {
        return Err(
            LinearLieAlgebraQualificationErrorV1::NonFiniteWhitenedGenerator {
                generator_id: generator_id.to_string(),
            },
        );
    }

    Ok((normalized, input_scale, whitened_norm, skew_defect))
}

/// Output of deterministic column-pivoted MGS:
/// (orthonormal basis, triangular factor, pivot permutation, relative pivots).
type PivotedMgsResultV1 = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>, Vec<f64>);

/// Deterministic column-pivoted MGS with two reorthogonalization passes.
/// Original generator columns are unit norm, so pivot residuals are relative.
fn pivoted_modified_gram_schmidt(
    columns: &[Vec<f64>],
    rank_tolerance: f64,
) -> Result<PivotedMgsResultV1, LinearLieAlgebraQualificationErrorV1> {
    let count = columns.len();
    let mut residuals = columns.to_vec();
    let mut permutation: Vec<usize> = (0..count).collect();
    let mut triangular = vec![vec![0.0_f64; count]; count];
    let mut orthonormal = Vec::with_capacity(count);
    let mut pivots = Vec::with_capacity(count);

    for pivot_index in 0..count {
        let mut selected = pivot_index;
        let mut selected_norm = vector_norm(&residuals[selected]);
        for (candidate, residual) in residuals
            .iter()
            .enumerate()
            .skip(pivot_index + 1)
        {
            let candidate_norm = vector_norm(residual);
            if candidate_norm > selected_norm {
                selected = candidate;
                selected_norm = candidate_norm;
            }
        }

        if selected != pivot_index {
            residuals.swap(pivot_index, selected);
            permutation.swap(pivot_index, selected);
            for row_values in triangular.iter_mut().take(pivot_index) {
                row_values.swap(pivot_index, selected);
            }
        }

        let pivot = vector_norm(&residuals[pivot_index]);
        pivots.push(pivot);
        if !pivot.is_finite() || pivot <= rank_tolerance {
            return Err(LinearLieAlgebraQualificationErrorV1::RankDeficient {
                numerical_rank: pivot_index,
                generator_count: count,
                unresolved_relative_pivot: pivot,
            });
        }

        let basis_vector: Vec<f64> = residuals[pivot_index]
            .iter()
            .map(|value| value / pivot)
            .collect();
        orthonormal.push(basis_vector.clone());
        triangular[pivot_index][pivot_index] = pivot;

        for candidate in (pivot_index + 1)..count {
            let first_projection = dot(&basis_vector, &residuals[candidate]);
            for (value, basis_value) in residuals[candidate].iter_mut().zip(basis_vector.iter()) {
                *value -= first_projection * basis_value;
            }

            let second_projection = dot(&basis_vector, &residuals[candidate]);
            for (value, basis_value) in residuals[candidate].iter_mut().zip(basis_vector.iter()) {
                *value -= second_projection * basis_value;
            }
            triangular[pivot_index][candidate] = first_projection + second_projection;
        }
    }

    Ok((orthonormal, triangular, permutation, pivots))
}

fn matrix_commutator(left: &[f64], right: &[f64], dimension: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; dimension * dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            let mut value = 0.0_f64;
            for inner in 0..dimension {
                value += left[row * dimension + inner] * right[inner * dimension + column]
                    - right[row * dimension + inner] * left[inner * dimension + column];
            }
            result[row * dimension + column] = value;
        }
    }
    result
}

fn project_and_solve(
    vector: &[f64],
    orthonormal_basis: &[Vec<f64>],
    triangular: &[Vec<f64>],
    permutation: &[usize],
    denominator_floor: f64,
) -> (Vec<f64>, f64) {
    let count = orthonormal_basis.len();
    let projections: Vec<f64> = orthonormal_basis
        .iter()
        .map(|basis_vector| dot(basis_vector, vector))
        .collect();

    let mut residual = vector.to_vec();
    for (basis_vector, coefficient) in orthonormal_basis.iter().zip(projections.iter()) {
        for (value, basis_value) in residual.iter_mut().zip(basis_vector.iter()) {
            *value -= coefficient * basis_value;
        }
    }
    let input_norm = vector_norm(vector);
    let relative_defect = vector_norm(&residual) / input_norm.max(denominator_floor);

    // Solve R c_pivoted = Q^T vector by upper-triangular back substitution.
    let mut pivoted_coefficients = vec![0.0_f64; count];
    for row in (0..count).rev() {
        let mut correction = 0.0_f64;
        for column in (row + 1)..count {
            correction += triangular[row][column] * pivoted_coefficients[column];
        }
        pivoted_coefficients[row] = (projections[row] - correction) / triangular[row][row];
    }

    let mut coefficients = vec![0.0_f64; count];
    for (pivot_index, original_index) in permutation.iter().enumerate() {
        coefficients[*original_index] = pivoted_coefficients[pivot_index];
    }

    (coefficients, relative_defect)
}

fn jacobi_term(
    structure: &[Vec<Vec<f64>>],
    generator_count: usize,
    outer: usize,
    middle_left: usize,
    middle_right: usize,
) -> Vec<f64> {
    let mut result = vec![0.0_f64; generator_count];
    for (inner, coefficient) in structure[middle_left][middle_right]
        .iter()
        .take(generator_count)
        .enumerate()
    {
        for (output, value) in result.iter_mut().enumerate() {
            *value += coefficient * structure[outer][inner][output];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_group_action::{
        LinearGeneratorV1, LinearGroupActionAdmissionProfileV1, LinearGroupActionV1,
    };

    fn generator(id: &str, matrix: Vec<Vec<f64>>) -> LinearGeneratorV1 {
        LinearGeneratorV1 {
            id: id.to_string(),
            matrix,
        }
    }

    fn so3_generators() -> Vec<LinearGeneratorV1> {
        vec![
            generator(
                "Jx",
                vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, -1.0],
                    vec![0.0, 1.0, 0.0],
                ],
            ),
            generator(
                "Jy",
                vec![
                    vec![0.0, 0.0, 1.0],
                    vec![0.0, 0.0, 0.0],
                    vec![-1.0, 0.0, 0.0],
                ],
            ),
            generator(
                "Jz",
                vec![
                    vec![0.0, -1.0, 0.0],
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                ],
            ),
        ]
    }

    fn action(
        generators: Vec<LinearGeneratorV1>,
        metric: Vec<Vec<f64>>,
    ) -> AdmittedLinearGroupActionV1 {
        let dimension = metric.len();
        let declared = LinearGroupActionV1 {
            action_id: "lie-basis-test".to_string(),
            coordinate_basis_id: "declared-coordinates".to_string(),
            coordinate_names: (0..dimension).map(|index| format!("x{index}")).collect(),
            coordinate_unit_ids: vec!["1".to_string(); dimension],
            metric,
            generators,
        };
        declared
            .try_admit(LinearGroupActionAdmissionProfileV1::default())
            .expect("test action must pass A1 admission")
    }

    fn identity(dimension: usize) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0_f64; dimension]; dimension];
        for index in 0..dimension {
            result[index][index] = 1.0;
        }
        result
    }

    fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let dimension = matrix.len();
        let mut result = vec![vec![0.0_f64; dimension]; dimension];
        for row in 0..dimension {
            for column in 0..dimension {
                result[row][column] = matrix[column][row];
            }
        }
        result
    }

    fn multiply(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let dimension = left.len();
        let mut result = vec![vec![0.0_f64; dimension]; dimension];
        for row in 0..dimension {
            for column in 0..dimension {
                for inner in 0..dimension {
                    result[row][column] += left[row][inner] * right[inner][column];
                }
            }
        }
        result
    }

    fn conjugated_so3_generators() -> Vec<LinearGeneratorV1> {
        let transform = vec![
            vec![1.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let inverse = vec![
            vec![1.0, -1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        so3_generators()
            .into_iter()
            .map(|item| {
                let matrix = multiply(&multiply(&transform, &item.matrix), &inverse);
                generator(&item.id, matrix)
            })
            .collect()
    }

    #[test]
    fn admits_so3_generator_basis_and_checks_jacobi() {
        let admitted = action(so3_generators(), identity(3));
        let report = qualify_linear_lie_algebra(
            &admitted,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        assert_eq!(
            report.status,
            LinearLieAlgebraQualificationStatusV1::NumericallyClosedAndJacobiConsistent
        );
        assert_eq!(report.diagnostics.numerical_rank, 3);
        assert_eq!(report.diagnostics.bracket_pair_count, 3);
        assert_eq!(report.diagnostics.jacobi_triple_count, 1);
        assert!(report.diagnostics.maximum_bracket_closure_defect < 1.0e-12);
        assert!(report.diagnostics.maximum_jacobi_defect.unwrap() < 1.0e-12);

        // Independent closed-form so(3) reference: [Jx, Jy] = Jz.
        // The reported basis is Frobenius-normalized, hence the coefficient is
        // 1/sqrt(2), not 1.0.
        let xy = report
            .brackets
            .iter()
            .find(|item| item.left_generator_id == "Jx" && item.right_generator_id == "Jy")
            .expect("Jx/Jy bracket must be present");
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((xy.structure_constants[2] - expected).abs() < 1.0e-12);
        assert!(xy.structure_constants[0].abs() < 1.0e-12);
        assert!(xy.structure_constants[1].abs() < 1.0e-12);
    }

    #[test]
    fn rejects_individually_admitted_generators_whose_bracket_leaves_the_span() {
        let mut generators = so3_generators();
        generators.pop();
        let admitted = action(generators, identity(3));
        let report = qualify_linear_lie_algebra(
            &admitted,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        assert_eq!(
            report.status,
            LinearLieAlgebraQualificationStatusV1::BracketClosureFailed
        );
        assert!(report.diagnostics.maximum_bracket_closure_defect > 0.9);
        assert!(report.jacobi_checks.is_empty());
        assert_eq!(report.diagnostics.maximum_jacobi_defect, None);
    }

    #[test]
    fn detects_duplicate_generators_without_silently_reducing_the_basis() {
        let mut duplicate = so3_generators()[0].clone();
        duplicate.id = "Jx-duplicate".to_string();
        let generators = vec![
            so3_generators()[0].clone(),
            duplicate,
            so3_generators()[2].clone(),
        ];
        let admitted = action(generators, identity(3));
        let error = qualify_linear_lie_algebra(
            &admitted,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            LinearLieAlgebraQualificationErrorV1::RankDeficient {
                numerical_rank: 2,
                generator_count: 3,
                ..
            }
        ));
    }

    #[test]
    fn closure_rank_and_jacobi_are_covariant_under_a_nonorthogonal_basis_change() {
        let canonical = action(so3_generators(), identity(3));
        let canonical_report = qualify_linear_lie_algebra(
            &canonical,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        let inverse = vec![
            vec![1.0, -1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let transformed_metric = multiply(&transpose(&inverse), &inverse);
        let transformed = action(conjugated_so3_generators(), transformed_metric);
        let transformed_report = qualify_linear_lie_algebra(
            &transformed,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        assert_eq!(canonical_report.status, transformed_report.status);
        assert_eq!(
            canonical_report.diagnostics.numerical_rank,
            transformed_report.diagnostics.numerical_rank
        );
        assert!(
            (canonical_report.diagnostics.maximum_bracket_closure_defect
                - transformed_report
                    .diagnostics
                    .maximum_bracket_closure_defect)
                .abs()
                < 1.0e-12
        );
        assert!(
            (canonical_report.diagnostics.maximum_jacobi_defect.unwrap()
                - transformed_report
                    .diagnostics
                    .maximum_jacobi_defect
                    .unwrap())
            .abs()
                < 1.0e-12
        );
        for (canonical_bracket, transformed_bracket) in canonical_report
            .brackets
            .iter()
            .zip(transformed_report.brackets.iter())
        {
            for (canonical_coefficient, transformed_coefficient) in canonical_bracket
                .structure_constants
                .iter()
                .zip(transformed_bracket.structure_constants.iter())
            {
                assert!(
                    (canonical_coefficient - transformed_coefficient).abs() < 1.0e-12,
                    "structure constants changed under coordinate re-representation"
                );
            }
        }
        for (left, right) in canonical_report
            .diagnostics
            .relative_rank_pivots
            .iter()
            .zip(transformed_report.diagnostics.relative_rank_pivots.iter())
        {
            assert!((left - right).abs() < 1.0e-12);
        }
    }

    #[test]
    fn so2_is_accepted_with_vacuous_jacobi_triple_set() {
        let admitted = action(
            vec![generator("J", vec![vec![0.0, -1.0], vec![1.0, 0.0]])],
            identity(2),
        );
        let report = qualify_linear_lie_algebra(
            &admitted,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        assert_eq!(
            report.status,
            LinearLieAlgebraQualificationStatusV1::NumericallyClosedAndJacobiConsistent
        );
        assert_eq!(report.diagnostics.numerical_rank, 1);
        assert_eq!(report.diagnostics.bracket_pair_count, 0);
        assert_eq!(report.diagnostics.jacobi_triple_count, 0);
        assert_eq!(report.diagnostics.maximum_jacobi_defect, Some(0.0));
    }

    #[test]
    fn detects_nearly_dependent_generators_and_abstains() {
        let mut generators = so3_generators();
        let jx = generators[0].matrix.clone();
        let jy = generators[1].matrix.clone();
        for row in 0..3 {
            for column in 0..3 {
                generators[1].matrix[row][column] = jx[row][column] + 1.0e-12 * jy[row][column];
            }
        }
        generators[1].id = "Jx-plus-small-Jy".to_string();
        let admitted = action(generators, identity(3));
        let error = qualify_linear_lie_algebra(
            &admitted,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            LinearLieAlgebraQualificationErrorV1::RankDeficient {
                numerical_rank: 2,
                generator_count: 3,
                ..
            }
        ));
    }

    #[test]
    fn generator_rescaling_and_permutation_preserve_qualification() {
        let canonical = action(so3_generators(), identity(3));
        let canonical_report = qualify_linear_lie_algebra(
            &canonical,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        let mut generators = so3_generators();
        for (generator, scale) in generators.iter_mut().zip([7.0, -2.0, 0.125]) {
            for value in generator.matrix.iter_mut().flatten() {
                *value *= scale;
            }
        }
        generators.reverse();
        let transformed = action(generators, identity(3));
        let transformed_report = qualify_linear_lie_algebra(
            &transformed,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        assert_eq!(canonical_report.status, transformed_report.status);
        assert!(
            (canonical_report.diagnostics.maximum_bracket_closure_defect
                - transformed_report
                    .diagnostics
                    .maximum_bracket_closure_defect)
                .abs()
                < 1.0e-12
        );
        assert!(
            (canonical_report.diagnostics.maximum_jacobi_defect.unwrap()
                - transformed_report
                    .diagnostics
                    .maximum_jacobi_defect
                    .unwrap())
            .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn accepts_invertible_mixed_and_rescaled_so3_generator_basis() {
        let original = so3_generators();
        let basis_change = [[2.0, 1.0, 0.0], [0.0, 1.0, 3.0], [1.0, 0.0, -1.0]];
        let mixed = basis_change
            .iter()
            .enumerate()
            .map(|(index, coefficients)| {
                let mut matrix = vec![vec![0.0_f64; 3]; 3];
                for (generator_index, coefficient) in coefficients.iter().enumerate() {
                    for row in 0..3 {
                        for column in 0..3 {
                            matrix[row][column] +=
                                coefficient * original[generator_index].matrix[row][column];
                        }
                    }
                }
                generator(&format!("mixed-{index}"), matrix)
            })
            .collect();
        let admitted = action(mixed, identity(3));
        let report = qualify_linear_lie_algebra(
            &admitted,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        assert_eq!(
            report.status,
            LinearLieAlgebraQualificationStatusV1::NumericallyClosedAndJacobiConsistent
        );
        assert_eq!(report.diagnostics.numerical_rank, 3);
        assert!(report.diagnostics.maximum_bracket_closure_defect < 1.0e-12);
        assert!(report.diagnostics.maximum_jacobi_defect.unwrap() < 1.0e-12);
    }

    #[test]
    fn jacobi_normalization_does_not_amplify_zero_nested_bracket_roundoff() {
        let admitted = action(so3_generators(), identity(3));
        let report = qualify_linear_lie_algebra(
            &admitted,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        let jacobi = &report.jacobi_checks[0];
        assert!(jacobi.used_global_scale_fallback);
        assert!(jacobi.global_scale_relative_defect < 1.0e-12);
        assert!(jacobi.relative_defect < 1.0e-12);
        assert!(jacobi.local_term_scale < 1.0e-12);
    }

    #[test]
    fn rejects_pair_and_jacobi_work_budgets_before_allocation() {
        let admitted = action(so3_generators(), identity(3));
        let mut pair_profile = LinearLieAlgebraQualificationProfileV1::default();
        pair_profile.maximum_pair_checks = 2;
        assert!(matches!(
            qualify_linear_lie_algebra(&admitted, pair_profile),
            Err(LinearLieAlgebraQualificationErrorV1::PairBudgetExceeded {
                required: 3,
                maximum: 2
            })
        ));

        let mut jacobi_profile = LinearLieAlgebraQualificationProfileV1::default();
        jacobi_profile.maximum_jacobi_checks = 0;
        assert!(matches!(
            qualify_linear_lie_algebra(&admitted, jacobi_profile),
            Err(LinearLieAlgebraQualificationErrorV1::JacobiBudgetExceeded {
                required: 1,
                maximum: 0
            })
        ));
    }

    #[test]
    fn accepts_direct_sum_of_two_commuting_so2_algebras() {
        let generators = vec![
            generator(
                "J01",
                vec![
                    vec![0.0, -1.0, 0.0, 0.0],
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, 0.0],
                ],
            ),
            generator(
                "J23",
                vec![
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, -1.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                ],
            ),
        ];
        let admitted = action(generators, identity(4));
        let report = qualify_linear_lie_algebra(
            &admitted,
            LinearLieAlgebraQualificationProfileV1::default(),
        )
        .unwrap();

        assert_eq!(
            report.status,
            LinearLieAlgebraQualificationStatusV1::NumericallyClosedAndJacobiConsistent
        );
        assert_eq!(report.diagnostics.numerical_rank, 2);
        assert_eq!(report.diagnostics.bracket_pair_count, 1);
        assert!(report.brackets[0].classified_as_numerically_zero);
    }

    #[test]
    fn rejects_excessive_work_before_allocating_the_qualification_workspace() {
        let admitted = action(so3_generators(), identity(3));
        let mut profile = LinearLieAlgebraQualificationProfileV1::default();
        profile.maximum_work_units = 1;

        assert!(matches!(
            qualify_linear_lie_algebra(&admitted, profile),
            Err(LinearLieAlgebraQualificationErrorV1::WorkBudgetExceeded { .. })
        ));
    }

    #[test]
    fn rejects_profile_values_above_hard_resource_ceilings() {
        let admitted = action(so3_generators(), identity(3));
        let mut profile = LinearLieAlgebraQualificationProfileV1::default();
        profile.maximum_work_units = HARD_MAXIMUM_WORK_UNITS + 1;

        assert!(matches!(
            qualify_linear_lie_algebra(&admitted, profile),
            Err(LinearLieAlgebraQualificationErrorV1::InvalidProfile)
        ));
    }
}
