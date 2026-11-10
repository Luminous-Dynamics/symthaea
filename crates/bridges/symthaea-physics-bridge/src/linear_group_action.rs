// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned declarations and fail-closed admission for explicit linear group actions.
//!
//! This module admits an infinitesimal linear action against an explicit symmetric
//! positive-definite metric. It does not infer an action from an expression and does
//! not prove that a supplied generator collection is a closed, independent Lie-algebra
//! basis. Those are separate discovery / algebra-qualification obligations.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// One declared infinitesimal generator in an explicit coordinate representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearGeneratorV1 {
    /// Stable identity within the action declaration.
    pub id: String,
    /// Row-major square matrix acting on column coordinate vectors.
    pub matrix: Vec<Vec<f64>>,
}

/// Explicit coordinate representation of a candidate linear group action.
///
/// All coordinates must already be expressed in the declared compatible basis and
/// unit convention. Unit identifiers are preserved as metadata; this admission layer
/// does not perform dimensional analysis or silently convert units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearGroupActionV1 {
    /// Stable identity for this exact action declaration.
    pub action_id: String,
    /// Identity of the coordinate basis / chart in which matrices are expressed.
    pub coordinate_basis_id: String,
    /// Ordered coordinate identities. Matrix rows and columns use this ordering.
    pub coordinate_names: Vec<String>,
    /// Unit identifiers for the corresponding normalized coordinate values.
    /// Use an explicit dimensionless identifier (for example, "1") where appropriate.
    pub coordinate_unit_ids: Vec<String>,
    /// Symmetric positive-definite metric in this coordinate representation.
    pub metric: Vec<Vec<f64>>,
    /// Candidate infinitesimal generators in the declared representation.
    pub generators: Vec<LinearGeneratorV1>,
}

/// Numerical admission policy for a declared linear action.
///
/// The tolerances are dimensionless relative Frobenius defects. No implicit metric,
/// coordinate scale, or tolerance is selected by the claim-bearing admission call.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearGroupActionAdmissionProfileV1 {
    /// Maximum number of coordinates admitted in one action declaration.
    pub maximum_dimension: usize,
    /// Maximum number of infinitesimal generators admitted in one declaration.
    pub maximum_generators: usize,
    /// Metric-whitened RMS Frobenius defect of the antisymmetric metric part.
    pub metric_symmetry_relative_tolerance: f64,
    /// Relative defect allowed for A^T G + G A = 0, generator by generator.
    pub metric_skew_relative_tolerance: f64,
}

impl Default for LinearGroupActionAdmissionProfileV1 {
    fn default() -> Self {
        Self {
            maximum_dimension: 32,
            maximum_generators: 128,
            metric_symmetry_relative_tolerance: 1e-12,
            metric_skew_relative_tolerance: 1e-12,
        }
    }
}

/// Fail-closed reasons why a linear action declaration was not admitted.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearGroupActionAdmissionError {
    InvalidProfile,
    EmptyActionId,
    EmptyCoordinateBasisId,
    NoCoordinates,
    DimensionLimitExceeded { actual: usize, maximum: usize },
    GeneratorCountLimitExceeded { actual: usize, maximum: usize },
    CoordinateUnitCountMismatch { expected: usize, actual: usize },
    EmptyCoordinateName { index: usize },
    EmptyCoordinateUnitId { index: usize },
    DuplicateCoordinateName { name: String },
    MetricRowCountMismatch { expected: usize, actual: usize },
    MetricColumnCountMismatch { row: usize, expected: usize, actual: usize },
    GeneratorMatrixRowCountMismatch {
        generator_id: String,
        expected: usize,
        actual: usize,
    },
    GeneratorMatrixColumnCountMismatch {
        generator_id: String,
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteMatrixEntry {
        matrix_id: String,
        row: usize,
        column: usize,
    },
    MetricNotSymmetric { relative_defect: f64 },
    MetricNotPositiveDefinite { pivot_index: usize },
    MetricFactorizationFailed { pivot_index: usize },
    NoGenerators,
    EmptyGeneratorId { index: usize },
    DuplicateGeneratorId { generator_id: String },
    ZeroGenerator { generator_id: String },
    GeneratorNotMetricSkew {
        generator_id: String,
        relative_defect: f64,
    },
}

impl fmt::Display for LinearGroupActionAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LinearGroupActionAdmissionError {}

/// Measured metric-skew residual for one admitted generator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratorMetricSkewDiagnosticV1 {
    pub generator_id: String,
    pub relative_defect: f64,
}

/// Measured diagnostics retained by a successful action admission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearGroupActionAdmissionDiagnosticsV1 {
    pub metric_symmetry_relative_defect: f64,
    /// Minimum diagonal entry of the Cholesky factor of G / metric_scale.
    /// This is a pivot-scale diagnostic, not a condition-number estimate.
    pub minimum_scaled_cholesky_diagonal: f64,
    /// Maximum diagonal entry of the Cholesky factor of G / metric_scale.
    pub maximum_scaled_cholesky_diagonal: f64,
    pub generator_metric_skew: Vec<GeneratorMetricSkewDiagnosticV1>,
}

/// An immutable action declaration that passed the local structural / metric checks.
///
/// The Cholesky factor is for the scaled metric G / metric_scale. Keeping the scale
/// separately avoids overflow from factoring a valid metric with very large units.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedLinearGroupActionV1 {
    /// Effective declaration; its metric is the symmetric part admitted by the profile.
    action: LinearGroupActionV1,
    /// Exact unmodified input declaration, retained for audit.
    declared_action: LinearGroupActionV1,
    profile: LinearGroupActionAdmissionProfileV1,
    diagnostics: LinearGroupActionAdmissionDiagnosticsV1,
    metric_scale: f64,
    scaled_metric_cholesky_lower: Vec<Vec<f64>>,
}

impl AdmittedLinearGroupActionV1 {
    /// Effective admitted declaration. Its metric is exactly symmetric.
    ///
    /// When a slightly asymmetric input metric is admitted within tolerance,
    /// this returns its symmetric part; see declared_action() for the exact input.
    pub fn action(&self) -> &LinearGroupActionV1 {
        &self.action
    }

    /// The exact, unmodified declaration supplied to try_admit().
    pub fn declared_action(&self) -> &LinearGroupActionV1 {
        &self.declared_action
    }

    /// Number of coordinates in the declared representation.
    pub fn dimension(&self) -> usize {
        self.action.coordinate_names.len()
    }

    /// Exact admission profile used to validate this action.
    pub fn profile(&self) -> &LinearGroupActionAdmissionProfileV1 {
        &self.profile
    }

    /// Measured diagnostics that justified admission.
    pub fn diagnostics(&self) -> &LinearGroupActionAdmissionDiagnosticsV1 {
        &self.diagnostics
    }

    /// Positive scale satisfying G = metric_scale * L * L^T.
    pub fn metric_scale(&self) -> f64 {
        self.metric_scale
    }

    /// Lower-triangular Cholesky factor of G / metric_scale.
    pub fn scaled_metric_cholesky_lower(&self) -> &[Vec<f64>] {
        &self.scaled_metric_cholesky_lower
    }
}

impl LinearGroupActionV1 {
    /// Admit the declaration only if its coordinate metadata, metric and every
    /// generator satisfy the declared relative tolerances.
    ///
    /// This establishes:
    /// - a complete, ordered coordinate identity / unit declaration;
    /// - a finite, symmetric positive-definite metric;
    /// - finite, nonzero square generator matrices;
    /// - metric-skewness A^T G + G A = 0 for each generator.
    ///
    /// It does not establish Lie-bracket closure, linear independence of generators,
    /// global exponentiation behavior, expression invariance, dynamical equivariance,
    /// conservation, or a Noether momentum map.
    pub fn try_admit(
        &self,
        profile: LinearGroupActionAdmissionProfileV1,
    ) -> Result<AdmittedLinearGroupActionV1, LinearGroupActionAdmissionError> {
        if profile.maximum_dimension == 0
            || profile.maximum_generators == 0
            || !profile.metric_symmetry_relative_tolerance.is_finite()
            || !(0.0..=1.0).contains(&profile.metric_symmetry_relative_tolerance)
            || !profile.metric_skew_relative_tolerance.is_finite()
            || !(0.0..=1.0).contains(&profile.metric_skew_relative_tolerance)
        {
            return Err(LinearGroupActionAdmissionError::InvalidProfile);
        }
        if self.action_id.trim().is_empty() {
            return Err(LinearGroupActionAdmissionError::EmptyActionId);
        }
        if self.coordinate_basis_id.trim().is_empty() {
            return Err(LinearGroupActionAdmissionError::EmptyCoordinateBasisId);
        }

        let dimension = self.coordinate_names.len();
        if dimension == 0 {
            return Err(LinearGroupActionAdmissionError::NoCoordinates);
        }
        if dimension > profile.maximum_dimension {
            return Err(LinearGroupActionAdmissionError::DimensionLimitExceeded {
                actual: dimension,
                maximum: profile.maximum_dimension,
            });
        }
        if self.coordinate_unit_ids.len() != dimension {
            return Err(LinearGroupActionAdmissionError::CoordinateUnitCountMismatch {
                expected: dimension,
                actual: self.coordinate_unit_ids.len(),
            });
        }

        let mut coordinate_names = BTreeSet::new();
        for (index, name) in self.coordinate_names.iter().enumerate() {
            if name.trim().is_empty() {
                return Err(LinearGroupActionAdmissionError::EmptyCoordinateName { index });
            }
            if !coordinate_names.insert(name) {
                return Err(LinearGroupActionAdmissionError::DuplicateCoordinateName {
                    name: name.clone(),
                });
            }
        }
        for (index, unit_id) in self.coordinate_unit_ids.iter().enumerate() {
            if unit_id.trim().is_empty() {
                return Err(LinearGroupActionAdmissionError::EmptyCoordinateUnitId { index });
            }
        }

        validate_square_matrix(&self.metric, dimension, "metric")?;
        if self.generators.is_empty() {
            return Err(LinearGroupActionAdmissionError::NoGenerators);
        }
        if self.generators.len() > profile.maximum_generators {
            return Err(LinearGroupActionAdmissionError::GeneratorCountLimitExceeded {
                actual: self.generators.len(),
                maximum: profile.maximum_generators,
            });
        }

        let metric_scale = max_abs(&self.metric);
        if metric_scale == 0.0 {
            return Err(LinearGroupActionAdmissionError::MetricNotPositiveDefinite {
                pivot_index: 0,
            });
        }
        // Factor the symmetric part first. This provides the intrinsic
        // metric scale used to measure the antisymmetric component.
        let scaled_metric_cholesky_lower =
            cholesky_lower_scaled_metric(&self.metric, metric_scale)?;
        let metric_symmetry_defect = relative_symmetry_defect(
            &self.metric,
            metric_scale,
            &scaled_metric_cholesky_lower,
        );
        if !metric_symmetry_defect.is_finite()
            || metric_symmetry_defect > profile.metric_symmetry_relative_tolerance
        {
            return Err(LinearGroupActionAdmissionError::MetricNotSymmetric {
                relative_defect: metric_symmetry_defect,
            });
        }
        let mut minimum_scaled_cholesky_diagonal = f64::INFINITY;
        let mut maximum_scaled_cholesky_diagonal = 0.0_f64;
        for (index, row) in scaled_metric_cholesky_lower.iter().enumerate() {
            let diagonal = row[index];
            minimum_scaled_cholesky_diagonal = minimum_scaled_cholesky_diagonal.min(diagonal);
            maximum_scaled_cholesky_diagonal = maximum_scaled_cholesky_diagonal.max(diagonal);
        }

        // Use the exact symmetric part as the admitted metric. A small
        // antisymmetric input component is tolerated only as measurement /
        // serialization noise; it is never propagated as a physical metric.
        let mut admitted_action = self.clone();
        for (row_index, row) in admitted_action.metric.iter_mut().enumerate() {
            for (column_index, value) in row.iter_mut().enumerate() {
                let symmetric_scaled = 0.5
                    * (self.metric[row_index][column_index] / metric_scale
                        + self.metric[column_index][row_index] / metric_scale);
                *value = symmetric_scaled * metric_scale;
            }
        }

        let mut generator_ids = BTreeSet::new();
        let mut generator_metric_skew = Vec::with_capacity(self.generators.len());
        for (index, generator) in self.generators.iter().enumerate() {
            if generator.id.trim().is_empty() {
                return Err(LinearGroupActionAdmissionError::EmptyGeneratorId { index });
            }
            if !generator_ids.insert(generator.id.as_str()) {
                return Err(LinearGroupActionAdmissionError::DuplicateGeneratorId {
                    generator_id: generator.id.clone(),
                });
            }
            validate_generator_matrix(&generator.matrix, dimension, &generator.id)?;
            let generator_scale = max_abs(&generator.matrix);
            if generator_scale == 0.0 {
                return Err(LinearGroupActionAdmissionError::ZeroGenerator {
                    generator_id: generator.id.clone(),
                });
            }

            let relative_defect = metric_skew_relative_defect(
                &generator.matrix,
                generator_scale,
                &scaled_metric_cholesky_lower,
            );
            if !relative_defect.is_finite()
                || relative_defect > profile.metric_skew_relative_tolerance
            {
                return Err(LinearGroupActionAdmissionError::GeneratorNotMetricSkew {
                    generator_id: generator.id.clone(),
                    relative_defect,
                });
            }
            generator_metric_skew.push(GeneratorMetricSkewDiagnosticV1 {
                generator_id: generator.id.clone(),
                relative_defect,
            });
        }

        Ok(AdmittedLinearGroupActionV1 {
            action: admitted_action,
            declared_action: self.clone(),
            profile,
            diagnostics: LinearGroupActionAdmissionDiagnosticsV1 {
                metric_symmetry_relative_defect: metric_symmetry_defect,
                minimum_scaled_cholesky_diagonal,
                maximum_scaled_cholesky_diagonal,
                generator_metric_skew,
            },
            metric_scale,
            scaled_metric_cholesky_lower,
        })
    }
}

fn validate_square_matrix(
    matrix: &[Vec<f64>],
    dimension: usize,
    matrix_id: &str,
) -> Result<(), LinearGroupActionAdmissionError> {
    if matrix.len() != dimension {
        return Err(LinearGroupActionAdmissionError::MetricRowCountMismatch {
            expected: dimension,
            actual: matrix.len(),
        });
    }
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != dimension {
            return Err(LinearGroupActionAdmissionError::MetricColumnCountMismatch {
                row: row_index,
                expected: dimension,
                actual: row.len(),
            });
        }
        for (column_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(LinearGroupActionAdmissionError::NonFiniteMatrixEntry {
                    matrix_id: matrix_id.to_string(),
                    row: row_index,
                    column: column_index,
                });
            }
        }
    }
    Ok(())
}

fn validate_generator_matrix(
    matrix: &[Vec<f64>],
    dimension: usize,
    generator_id: &str,
) -> Result<(), LinearGroupActionAdmissionError> {
    if matrix.len() != dimension {
        return Err(
            LinearGroupActionAdmissionError::GeneratorMatrixRowCountMismatch {
                generator_id: generator_id.to_string(),
                expected: dimension,
                actual: matrix.len(),
            },
        );
    }
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != dimension {
            return Err(
                LinearGroupActionAdmissionError::GeneratorMatrixColumnCountMismatch {
                    generator_id: generator_id.to_string(),
                    row: row_index,
                    expected: dimension,
                    actual: row.len(),
                },
            );
        }
        for (column_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(LinearGroupActionAdmissionError::NonFiniteMatrixEntry {
                    matrix_id: format!("generator:{generator_id}"),
                    row: row_index,
                    column: column_index,
                });
            }
        }
    }
    Ok(())
}

fn max_abs(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
}

fn scaled_frobenius_norm(matrix: &[Vec<f64>], scale: f64) -> f64 {
    matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |norm, value| norm.hypot(value / scale))
}

fn relative_symmetry_defect(
    metric: &[Vec<f64>],
    scale: f64,
    scaled_metric_cholesky_lower: &[Vec<f64>],
) -> f64 {
    // Split G into symmetric S and antisymmetric W. Cholesky-whiten the
    // symmetric part S = scale * L * L^T and measure
    // L^-1 W L^-T. Under a coordinate congruence, this whitened
    // antisymmetric form changes only by orthogonal conjugation, making its
    // Frobenius norm independent of the chosen coordinate representation.
    let dimension = metric.len();
    let mut antisymmetric = vec![vec![0.0_f64; dimension]; dimension];
    for (row_index, row) in metric.iter().enumerate() {
        for (column_index, value) in row.iter().enumerate() {
            antisymmetric[row_index][column_index] =
                0.5 * (*value / scale - metric[column_index][row_index] / scale);
        }
    }

    // First solve L * X = W one column at a time.
    let mut left_whitened = vec![vec![0.0_f64; dimension]; dimension];
    for (column_index, _) in metric.iter().enumerate() {
        let mut solved_column = Vec::with_capacity(dimension);
        for (row_index, _) in metric.iter().enumerate() {
            let correction = solved_column
                .iter()
                .enumerate()
                .map(|(inner_index, value)| {
                    scaled_metric_cholesky_lower[row_index][inner_index] * *value
                })
                .sum::<f64>();
            let pivot = scaled_metric_cholesky_lower[row_index][row_index];
            let value = (antisymmetric[row_index][column_index] - correction) / pivot;
            if !value.is_finite() {
                return f64::INFINITY;
            }
            solved_column.push(value);
        }
        for (row_index, value) in solved_column.iter().enumerate() {
            left_whitened[row_index][column_index] = *value;
        }
    }

    // Then solve Y * L^T = X one row at a time.
    let mut whitened = vec![vec![0.0_f64; dimension]; dimension];
    for (row_index, left_row) in left_whitened.iter().enumerate() {
        let mut solved_row = Vec::with_capacity(dimension);
        for (column_index, left_entry) in left_row.iter().enumerate() {
            let correction = solved_row
                .iter()
                .zip(scaled_metric_cholesky_lower[column_index].iter())
                .take(column_index)
                .map(|(value, lower_value)| *value * *lower_value)
                .sum::<f64>();
            let pivot = scaled_metric_cholesky_lower[column_index][column_index];
            let value = (*left_entry - correction) / pivot;
            if !value.is_finite() {
                return f64::INFINITY;
            }
            solved_row.push(value);
        }
        whitened[row_index] = solved_row;
    }

    scaled_frobenius_norm(&whitened, 1.0) / (dimension as f64).sqrt()
}

fn cholesky_lower_scaled_metric(
    metric: &[Vec<f64>],
    scale: f64,
) -> Result<Vec<Vec<f64>>, LinearGroupActionAdmissionError> {
    let dimension = metric.len();
    let mut lower = vec![vec![0.0_f64; dimension]; dimension];

    for (row_index, metric_row) in metric.iter().enumerate() {
        for (column_index, metric_entry) in metric_row.iter().take(row_index + 1).enumerate() {
            // Average only after the relative symmetry check; normalize first to
            // avoid overflowing the addition for large but finite metric entries.
            let value = 0.5 * (*metric_entry / scale + metric[column_index][row_index] / scale);
            let mut residual = value;
            for (left, right) in lower[row_index][..column_index]
                .iter()
                .zip(lower[column_index][..column_index].iter())
            {
                residual -= left * right;
            }
            if row_index == column_index {
                if !residual.is_finite() || residual <= 0.0 {
                    return Err(LinearGroupActionAdmissionError::MetricNotPositiveDefinite {
                        pivot_index: row_index,
                    });
                }
                lower[row_index][column_index] = residual.sqrt();
            } else {
                let pivot = lower[column_index][column_index];
                let entry = residual / pivot;
                if !entry.is_finite() {
                    return Err(LinearGroupActionAdmissionError::MetricFactorizationFailed {
                        pivot_index: column_index,
                    });
                }
                lower[row_index][column_index] = entry;
            }
        }
    }
    Ok(lower)
}

fn metric_skew_relative_defect(
    generator: &[Vec<f64>],
    generator_scale: f64,
    scaled_metric_cholesky_lower: &[Vec<f64>],
) -> f64 {
    // Let G / metric_scale = L L^T. The metric-skew condition
    // A^T G + G A = 0 becomes B^T + B = 0 for
    // B = L^T A L^-T. Under a coordinate change, valid Cholesky
    // whitenings differ by an orthogonal factor, so this Frobenius
    // residual is representation-invariant.
    //
    // Normalize A before multiplication and compute B with a
    // right-triangular solve instead of explicitly forming L^-T.
    let dimension = generator.len();
    let mut left_product = vec![vec![0.0_f64; dimension]; dimension];
    for (row_index, output_row) in left_product.iter_mut().enumerate() {
        for (column_index, output) in output_row.iter_mut().enumerate() {
            let mut value = 0.0_f64;
            for (inner_index, generator_row) in generator.iter().enumerate() {
                value += scaled_metric_cholesky_lower[inner_index][row_index]
                    * (generator_row[column_index] / generator_scale);
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
                .zip(scaled_metric_cholesky_lower[column_index].iter())
                .take(column_index)
                .map(|(value, lower_value)| value * lower_value)
                .sum::<f64>();
            let pivot = scaled_metric_cholesky_lower[column_index][column_index];
            let value = (*left_entry - correction) / pivot;
            if !value.is_finite() {
                return f64::INFINITY;
            }
            solved_row.push(value);
        }
        whitened[row_index] = solved_row;
    }

    let whitened_scale = max_abs(&whitened);
    if !whitened_scale.is_finite() || whitened_scale == 0.0 {
        return f64::INFINITY;
    }
    let generator_norm = scaled_frobenius_norm(&whitened, whitened_scale);
    let mut defect_norm = 0.0_f64;
    for (row_index, row) in whitened.iter().enumerate() {
        for (column_index, value) in row.iter().enumerate() {
            let residual =
                *value / whitened_scale + whitened[column_index][row_index] / whitened_scale;
            defect_norm = defect_norm.hypot(residual);
        }
    }

    defect_norm / (2.0 * generator_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> LinearGroupActionAdmissionProfileV1 {
        LinearGroupActionAdmissionProfileV1::default()
    }

    fn canonical_so2() -> LinearGroupActionV1 {
        LinearGroupActionV1 {
            action_id: "canonical-so2".to_string(),
            coordinate_basis_id: "xy".to_string(),
            coordinate_names: vec!["x".to_string(), "y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generators: vec![LinearGeneratorV1 {
                id: "J".to_string(),
                matrix: vec![vec![0.0, -1.0], vec![1.0, 0.0]],
            }],
        }
    }

    #[test]
    fn admits_canonical_euclidean_so2_action() {
        let admitted = canonical_so2().try_admit(profile()).unwrap();
        assert_eq!(admitted.dimension(), 2);
        assert_eq!(admitted.action().action_id, "canonical-so2");
        assert!((admitted.metric_scale() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn admits_rational_nonorthogonal_representation_with_transformed_metric() {
        let action = LinearGroupActionV1 {
            action_id: "so2-rational-nonorthogonal".to_string(),
            coordinate_basis_id: "XY".to_string(),
            coordinate_names: vec!["X".to_string(), "Y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![0.4, -0.4], vec![-0.4, 0.8]],
            generators: vec![LinearGeneratorV1 {
                id: "conjugated-J".to_string(),
                matrix: vec![vec![1.0, -2.0], vec![1.0, -1.0]],
            }],
        };
        let admitted = action.try_admit(profile()).unwrap();
        assert_eq!(admitted.dimension(), 2);
    }

    #[test]
    fn admitted_action_canonicalizes_metric_but_preserves_exact_declaration() {
        let mut action = canonical_so2();
        action.metric = vec![vec![1.0, 5.0e-13], vec![-5.0e-13, 1.0]];
        let admitted = action.try_admit(profile()).unwrap();

        assert_eq!(admitted.declared_action().metric, action.metric);
        assert_eq!(admitted.action().metric[0][1], admitted.action().metric[1][0]);
        assert!((admitted.diagnostics().metric_symmetry_relative_defect - 5.0e-13).abs() < 1.0e-15);
        assert_eq!(admitted.action().metric, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    }

    #[test]
    fn admission_receipt_preserves_profile_and_measured_defects() {
        let selected_profile = profile();
        let admitted = canonical_so2().try_admit(selected_profile).unwrap();
        assert_eq!(admitted.profile(), &selected_profile);
        assert_eq!(admitted.diagnostics().metric_symmetry_relative_defect, 0.0);
        assert_eq!(admitted.diagnostics().generator_metric_skew.len(), 1);
        assert_eq!(
            admitted.diagnostics().generator_metric_skew[0].generator_id,
            "J"
        );
        assert_eq!(
            admitted.diagnostics().generator_metric_skew[0].relative_defect,
            0.0
        );
        assert_eq!(
            admitted.diagnostics().minimum_scaled_cholesky_diagonal,
            1.0
        );
        assert_eq!(
            admitted.diagnostics().maximum_scaled_cholesky_diagonal,
            1.0
        );
    }

    #[test]
    fn metric_symmetry_admission_is_invariant_under_nonorthogonal_basis_change() {
        // D = [[1, 1], [0, 1]] and G' = D^-T G D^-1. The same
        // small antisymmetric perturbation must receive the same whitened
        // metric-symmetry defect in both coordinate representations.
        let tolerance = 5.0e-4;
        let mut canonical = canonical_so2();
        canonical.metric = vec![vec![1.0, 0.001], vec![-0.001, 1.0]];
        let mut canonical_profile = profile();
        canonical_profile.metric_symmetry_relative_tolerance = tolerance;
        let transformed = LinearGroupActionV1 {
            action_id: "so2-sheared-nonsymmetric-metric".to_string(),
            coordinate_basis_id: "xy-sheared".to_string(),
            coordinate_names: vec!["X".to_string(), "Y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, -0.999], vec![-1.001, 2.0]],
            generators: vec![LinearGeneratorV1 {
                id: "conjugated-J".to_string(),
                matrix: vec![vec![1.0, -2.0], vec![1.0, -1.0]],
            }],
        };
        let canonical_error = canonical.try_admit(canonical_profile).unwrap_err();
        let transformed_error = transformed.try_admit(canonical_profile).unwrap_err();

        let relative_defect = |error: LinearGroupActionAdmissionError| match error {
            LinearGroupActionAdmissionError::MetricNotSymmetric { relative_defect } => {
                relative_defect
            }
            other => panic!("expected metric-symmetry rejection, got {other:?}"),
        };
        let canonical_defect = relative_defect(canonical_error);
        let transformed_defect = relative_defect(transformed_error);
        assert!((canonical_defect - transformed_defect).abs() < 1.0e-12);
        assert!(canonical_defect > tolerance);
        assert!(transformed_defect > tolerance);
    }

    #[test]
    fn metric_skew_admission_is_invariant_under_nonorthogonal_basis_change() {
        // D = [[1, 1], [0, 1]], A' = D A D^-1, and G' = D^-T D^-1.
        // The epsilon perturbation is intentionally not metric-skew. Admission
        // must make the same decision and report the same normalized defect in
        // both representations.
        let tolerance = 5.0e-4;
        let mut canonical = canonical_so2();
        canonical.generators[0].matrix =
            vec![vec![0.001, -1.0], vec![1.0, -0.001]];
        let mut canonical_profile = profile();
        canonical_profile.metric_skew_relative_tolerance = tolerance;

        let transformed = LinearGroupActionV1 {
            action_id: "so2-sheared-approximate".to_string(),
            coordinate_basis_id: "xy-sheared".to_string(),
            coordinate_names: vec!["X".to_string(), "Y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, -1.0], vec![-1.0, 2.0]],
            generators: vec![LinearGeneratorV1 {
                id: "conjugated-J-perturbed".to_string(),
                matrix: vec![vec![1.001, -2.002], vec![1.0, -1.001]],
            }],
        };
        let transformed_profile = canonical_profile;
        let canonical_error = canonical
            .try_admit(canonical_profile)
            .unwrap_err();
        let transformed_error = transformed
            .try_admit(transformed_profile)
            .unwrap_err();

        let relative_defect = |error: LinearGroupActionAdmissionError| match error {
            LinearGroupActionAdmissionError::GeneratorNotMetricSkew {
                relative_defect,
                ..
            } => relative_defect,
            other => panic!("expected metric-skew rejection, got {other:?}"),
        };
        let canonical_defect = relative_defect(canonical_error);
        let transformed_defect = relative_defect(transformed_error);
        assert!((canonical_defect - transformed_defect).abs() < 1.0e-12);
        assert!(canonical_defect > tolerance);
        assert!(transformed_defect > tolerance);
    }

    #[test]
    fn rejects_conjugated_generator_under_wrong_identity_metric() {
        let mut action = canonical_so2();
        action.generators[0].matrix = vec![vec![1.0, -2.0], vec![1.0, -1.0]];
        assert!(matches!(
            action.try_admit(profile()),
            Err(LinearGroupActionAdmissionError::GeneratorNotMetricSkew { .. })
        ));
    }

    #[test]
    fn rejects_indefinite_metric() {
        let mut action = canonical_so2();
        action.metric = vec![vec![1.0, 0.0], vec![0.0, -1.0]];
        assert!(matches!(
            action.try_admit(profile()),
            Err(LinearGroupActionAdmissionError::MetricNotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn rejects_nonsymmetric_metric_before_factorization() {
        let mut action = canonical_so2();
        action.metric = vec![vec![1.0, 0.1], vec![0.0, 1.0]];
        assert!(matches!(
            action.try_admit(profile()),
            Err(LinearGroupActionAdmissionError::MetricNotSymmetric { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_coordinate_names() {
        let mut action = canonical_so2();
        action.coordinate_names[1] = "x".to_string();
        assert!(matches!(
            action.try_admit(profile()),
            Err(LinearGroupActionAdmissionError::DuplicateCoordinateName { .. })
        ));
    }

    #[test]
    fn rejects_generator_dimension_mismatch() {
        let mut action = canonical_so2();
        action.generators[0].matrix = vec![vec![0.0, -1.0]];
        assert!(matches!(
            action.try_admit(profile()),
            Err(LinearGroupActionAdmissionError::GeneratorMatrixRowCountMismatch { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_generator_entries() {
        let mut action = canonical_so2();
        action.generators[0].matrix[0][1] = f64::INFINITY;
        assert!(matches!(
            action.try_admit(profile()),
            Err(LinearGroupActionAdmissionError::NonFiniteMatrixEntry { .. })
        ));
    }

    #[test]
    fn rejects_zero_generator() {
        let mut action = canonical_so2();
        action.generators[0].matrix = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        assert!(matches!(
            action.try_admit(profile()),
            Err(LinearGroupActionAdmissionError::ZeroGenerator { .. })
        ));
    }

    #[test]
    fn admits_large_metric_scale_without_intermediate_overflow() {
        let mut action = canonical_so2();
        action.metric = vec![vec![1e300, 0.0], vec![0.0, 1e300]];
        let admitted = action.try_admit(profile()).unwrap();
        assert_eq!(admitted.metric_scale(), 1e300);
    }

    #[test]
    fn admits_tiny_nonzero_generator_without_absolute_magnitude_cutoff() {
        let mut action = canonical_so2();
        action.generators[0].matrix = vec![vec![0.0, -1e-300], vec![1e-300, 0.0]];
        let admitted = action.try_admit(profile()).unwrap();
        assert_eq!(admitted.dimension(), 2);
    }

    #[test]
    fn rejects_dimension_over_profile_budget() {
        let action = canonical_so2();
        let mut bounded = profile();
        bounded.maximum_dimension = 1;
        assert!(matches!(
            action.try_admit(bounded),
            Err(LinearGroupActionAdmissionError::DimensionLimitExceeded {
                actual: 2,
                maximum: 1
            })
        ));
    }

    #[test]
    fn rejects_generator_count_over_profile_budget() {
        let mut action = canonical_so2();
        action.generators.push(LinearGeneratorV1 {
            id: "J2".to_string(),
            matrix: vec![vec![0.0, -1.0], vec![1.0, 0.0]],
        });
        let mut bounded = profile();
        bounded.maximum_generators = 1;
        assert!(matches!(
            action.try_admit(bounded),
            Err(LinearGroupActionAdmissionError::GeneratorCountLimitExceeded {
                actual: 2,
                maximum: 1
            })
        ));
    }

    #[test]
    fn rejects_non_finite_or_negative_admission_tolerances() {
        let mut invalid = profile();
        invalid.metric_skew_relative_tolerance = f64::NAN;
        assert!(matches!(
            canonical_so2().try_admit(invalid),
            Err(LinearGroupActionAdmissionError::InvalidProfile)
        ));

        let mut negative = profile();
        negative.metric_symmetry_relative_tolerance = -1.0;
        assert!(matches!(
            canonical_so2().try_admit(negative),
            Err(LinearGroupActionAdmissionError::InvalidProfile)
        ));

        let mut excessive = profile();
        excessive.metric_skew_relative_tolerance = 1.01;
        assert!(matches!(
            canonical_so2().try_admit(excessive),
            Err(LinearGroupActionAdmissionError::InvalidProfile)
        ));
    }
}
