// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit, fail-closed admission for finite linear group representations.
//!
//! Finite transformations are represented directly by their matrices and complete
//! multiplication table. This is intentionally separate from infinitesimal linear
//! actions: reflections and isolated finite components do not need to be represented
//! by Lie-algebra generators.

use crate::linear_group_action::{
    LinearGroupActionAdmissionError, cholesky_lower_scaled_metric, max_abs,
    relative_symmetry_defect, scaled_frobenius_norm, validate_square_matrix,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One finite group element represented by a matrix acting on column coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FiniteGroupElementV1 {
    /// Stable identity within the finite group declaration.
    pub id: String,
    /// Row-major square matrix in the declared coordinate basis.
    pub matrix: Vec<Vec<f64>>,
}

/// One ordered product in the supplied finite-group multiplication table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteGroupMultiplicationV1 {
    pub left_element_id: String,
    pub right_element_id: String,
    pub product_element_id: String,
}

/// Explicit finite linear group action with a declared metric and full Cayley table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FiniteLinearGroupActionV1 {
    pub action_id: String,
    pub coordinate_basis_id: String,
    /// Matrix rows and columns follow this exact order.
    pub coordinate_names: Vec<String>,
    pub coordinate_unit_ids: Vec<String>,
    /// Metric on column coordinate vectors.
    pub metric: Vec<Vec<f64>>,
    pub identity_element_id: String,
    pub elements: Vec<FiniteGroupElementV1>,
    /// Must contain exactly one entry for every ordered element pair.
    pub multiplication_table: Vec<FiniteGroupMultiplicationV1>,
}

/// Resource-bounded numerical policy for finite group admission.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FiniteGroupAdmissionProfileV1 {
    pub maximum_dimension: usize,
    pub maximum_elements: usize,
    pub maximum_table_entries: usize,
    pub maximum_associativity_checks: usize,
    /// Metric-whitened RMS defect of the antisymmetric metric component.
    pub metric_symmetry_relative_tolerance: f64,
    /// RMS defect of U_g^T U_g - I in metric-whitened coordinates.
    pub element_isometry_relative_tolerance: f64,
    /// RMS defect of U_g U_h - U_(gh) in metric-whitened coordinates.
    pub multiplication_relative_tolerance: f64,
}

impl Default for FiniteGroupAdmissionProfileV1 {
    fn default() -> Self {
        Self {
            maximum_dimension: 32,
            maximum_elements: 128,
            maximum_table_entries: 16_384,
            maximum_associativity_checks: 2_097_152,
            metric_symmetry_relative_tolerance: 1e-10,
            element_isometry_relative_tolerance: 1e-10,
            multiplication_relative_tolerance: 1e-10,
        }
    }
}

const HARD_MAX_FINITE_GROUP_DIMENSION: usize = 32;
const HARD_MAX_FINITE_GROUP_ELEMENTS: usize = 128;
const HARD_MAX_FINITE_GROUP_TABLE_ENTRIES: usize = 16_384;
const HARD_MAX_FINITE_GROUP_ASSOCIATIVITY_CHECKS: usize = 2_097_152;

/// Fail-closed reasons why a finite group declaration was not admitted.
#[derive(Debug, Clone, PartialEq)]
pub enum FiniteGroupAdmissionErrorV1 {
    InvalidProfile,
    EmptyActionId,
    EmptyCoordinateBasisId,
    EmptyIdentityElementId,
    NoCoordinates,
    DimensionLimitExceeded { actual: usize, maximum: usize },
    CoordinateUnitCountMismatch { expected: usize, actual: usize },
    EmptyCoordinateName { index: usize },
    DuplicateCoordinateName { name: String },
    EmptyCoordinateUnitId { index: usize },
    MetricAdmission(LinearGroupActionAdmissionError),
    NoElements,
    ElementCountLimitExceeded { actual: usize, maximum: usize },
    EmptyElementId { index: usize },
    DuplicateElementId { element_id: String },
    IdentityElementMissing { element_id: String },
    MatrixAdmission {
        element_id: String,
        source: LinearGroupActionAdmissionError,
    },
    NonFiniteWhitenedElement { element_id: String },
    ElementNotMetricIsometry { element_id: String, relative_defect: f64 },
    IdentityMatrixMismatch { relative_defect: f64 },
    TableEntryBudgetExceeded { actual: usize, maximum: usize },
    IncompleteMultiplicationTable { expected: usize, actual: usize },
    DuplicateMultiplicationEntry { left: String, right: String },
    UnknownMultiplicationOperand { element_id: String },
    UnknownMultiplicationProduct { element_id: String },
    IdentityTableViolation { side: &'static str, element_id: String },
    MissingTwoSidedInverse { element_id: String },
    NonAssociativeMultiplicationTable {
        left: String,
        middle: String,
        right: String,
    },
    AssociativityBudgetExceeded { actual: usize, maximum: usize },
    MultiplicationRepresentationMismatch {
        left: String,
        right: String,
        product: String,
        relative_defect: f64,
    },
}

impl fmt::Display for FiniteGroupAdmissionErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FiniteGroupAdmissionErrorV1 {}

/// Per-element metric-isometry diagnostic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FiniteGroupElementDiagnosticV1 {
    pub element_id: String,
    pub metric_isometry_relative_defect: f64,
}

/// Evidence retained by a successful finite-group admission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FiniteGroupAdmissionDiagnosticsV1 {
    pub metric_symmetry_relative_defect: f64,
    pub identity_matrix_relative_defect: f64,
    pub maximum_element_isometry_relative_defect: f64,
    pub maximum_multiplication_relative_defect: f64,
    pub checked_multiplication_pairs: usize,
    pub checked_associativity_triples: usize,
    pub element_diagnostics: Vec<FiniteGroupElementDiagnosticV1>,
}

/// Immutable finite group representation that passed table and metric admission.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedFiniteLinearGroupActionV1 {
    action: FiniteLinearGroupActionV1,
    declared_action: FiniteLinearGroupActionV1,
    profile: FiniteGroupAdmissionProfileV1,
    diagnostics: FiniteGroupAdmissionDiagnosticsV1,
    metric_scale: f64,
    scaled_metric_cholesky_lower: Vec<Vec<f64>>,
    multiplication_lookup: BTreeMap<(String, String), String>,
}

impl AdmittedFiniteLinearGroupActionV1 {
    /// Effective declaration with the exactly symmetric admitted metric.
    pub fn action(&self) -> &FiniteLinearGroupActionV1 {
        &self.action
    }

    /// Exact unmodified declaration supplied by the caller.
    pub fn declared_action(&self) -> &FiniteLinearGroupActionV1 {
        &self.declared_action
    }

    pub fn profile(&self) -> &FiniteGroupAdmissionProfileV1 {
        &self.profile
    }

    pub fn diagnostics(&self) -> &FiniteGroupAdmissionDiagnosticsV1 {
        &self.diagnostics
    }

    pub fn dimension(&self) -> usize {
        self.action.coordinate_names.len()
    }

    pub fn order(&self) -> usize {
        self.action.elements.len()
    }

    pub fn metric_scale(&self) -> f64 {
        self.metric_scale
    }

    /// Lower-triangular factor L satisfying G_effective = scale * L * L^T.
    pub fn scaled_metric_cholesky_lower(&self) -> &[Vec<f64>] {
        &self.scaled_metric_cholesky_lower
    }

    /// Matrix of a declared element in the effective coordinate representation.
    pub fn element_matrix(&self, element_id: &str) -> Option<&[Vec<f64>]> {
        self.action
            .elements
            .iter()
            .find(|element| element.id == element_id)
            .map(|element| element.matrix.as_slice())
    }

    /// Declared product g*h, if both element IDs are present.
    pub fn product_of(&self, left: &str, right: &str) -> Option<&str> {
        self.multiplication_lookup
            .get(&(left.to_string(), right.to_string()))
            .map(String::as_str)
    }
}

fn matrix_product(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let dimension = left.len();
    let mut result = vec![vec![0.0_f64; dimension]; dimension];
    for (row_index, left_row) in left.iter().enumerate() {
        for (column_index, output) in result[row_index].iter_mut().enumerate() {
            *output = left_row
                .iter()
                .zip(right.iter())
                .map(|(left_value, right_row)| *left_value * right_row[column_index])
                .sum();
        }
    }
    result
}

fn matrix_transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let dimension = matrix.len();
    let mut result = vec![vec![0.0_f64; dimension]; dimension];
    for (row_index, row) in matrix.iter().enumerate() {
        for (column_index, value) in row.iter().enumerate() {
            result[column_index][row_index] = *value;
        }
    }
    result
}

/// Relative RMS Frobenius defect in a fixed dimension. Group-whitened matrices
/// are orthogonal, so an absolute dimension-normalized defect is well scaled.
fn matrix_rms_defect(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
    let dimension = left.len();
    let mut norm = 0.0_f64;
    for (left_row, right_row) in left.iter().zip(right.iter()) {
        for (left_value, right_value) in left_row.iter().zip(right_row.iter()) {
            norm = norm.hypot(left_value - right_value);
        }
    }
    norm / (dimension as f64).sqrt()
}

fn identity_matrix(dimension: usize) -> Vec<Vec<f64>> {
    (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| if row == column { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

/// Compute U = L^T M L^-T without explicitly forming L^-T.
///
/// The matrix is normalized before the left product, then rescaled after the
/// triangular solve. This keeps conjugated finite elements numerically usable
/// when the coordinate basis has large but finite scaling.
fn metric_whiten_matrix(
    matrix: &[Vec<f64>],
    lower: &[Vec<f64>],
) -> Option<Vec<Vec<f64>>> {
    let dimension = matrix.len();
    let matrix_scale = max_abs(matrix);
    if !matrix_scale.is_finite() || matrix_scale == 0.0 {
        return None;
    }

    let mut left_product = vec![vec![0.0_f64; dimension]; dimension];
    for (row_index, output_row) in left_product.iter_mut().enumerate() {
        for (column_index, output) in output_row.iter_mut().enumerate() {
            let value = matrix
                .iter()
                .enumerate()
                .map(|(inner_index, matrix_row)| {
                    lower[inner_index][row_index] * (matrix_row[column_index] / matrix_scale)
                })
                .sum::<f64>();
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
        for (column_index, solved_value) in solved_row.iter().enumerate() {
            let value = *solved_value * matrix_scale;
            if !value.is_finite() {
                return None;
            }
            whitened[row_index][column_index] = value;
        }
    }
    Some(whitened)
}

impl FiniteLinearGroupActionV1 {
    /// Admit a finite matrix-group representation only when the metric, element
    /// isometries, complete abstract table, and numerical matrix products agree.
    pub fn try_admit(
        &self,
        profile: FiniteGroupAdmissionProfileV1,
    ) -> Result<AdmittedFiniteLinearGroupActionV1, FiniteGroupAdmissionErrorV1> {
        if profile.maximum_dimension == 0
            || profile.maximum_dimension > HARD_MAX_FINITE_GROUP_DIMENSION
            || profile.maximum_elements == 0
            || profile.maximum_elements > HARD_MAX_FINITE_GROUP_ELEMENTS
            || profile.maximum_table_entries == 0
            || profile.maximum_table_entries > HARD_MAX_FINITE_GROUP_TABLE_ENTRIES
            || profile.maximum_associativity_checks == 0
            || profile.maximum_associativity_checks > HARD_MAX_FINITE_GROUP_ASSOCIATIVITY_CHECKS
            || !profile.metric_symmetry_relative_tolerance.is_finite()
            || !(0.0..=1.0).contains(&profile.metric_symmetry_relative_tolerance)
            || !profile.element_isometry_relative_tolerance.is_finite()
            || !(0.0..=1.0).contains(&profile.element_isometry_relative_tolerance)
            || !profile.multiplication_relative_tolerance.is_finite()
            || !(0.0..=1.0).contains(&profile.multiplication_relative_tolerance)
        {
            return Err(FiniteGroupAdmissionErrorV1::InvalidProfile);
        }
        if self.action_id.trim().is_empty() {
            return Err(FiniteGroupAdmissionErrorV1::EmptyActionId);
        }
        if self.coordinate_basis_id.trim().is_empty() {
            return Err(FiniteGroupAdmissionErrorV1::EmptyCoordinateBasisId);
        }
        if self.identity_element_id.trim().is_empty() {
            return Err(FiniteGroupAdmissionErrorV1::EmptyIdentityElementId);
        }

        let dimension = self.coordinate_names.len();
        if dimension == 0 {
            return Err(FiniteGroupAdmissionErrorV1::NoCoordinates);
        }
        if dimension > profile.maximum_dimension {
            return Err(FiniteGroupAdmissionErrorV1::DimensionLimitExceeded {
                actual: dimension,
                maximum: profile.maximum_dimension,
            });
        }
        if self.coordinate_unit_ids.len() != dimension {
            return Err(FiniteGroupAdmissionErrorV1::CoordinateUnitCountMismatch {
                expected: dimension,
                actual: self.coordinate_unit_ids.len(),
            });
        }

        let mut coordinate_names = BTreeSet::new();
        for (index, name) in self.coordinate_names.iter().enumerate() {
            if name.trim().is_empty() {
                return Err(FiniteGroupAdmissionErrorV1::EmptyCoordinateName { index });
            }
            if !coordinate_names.insert(name.as_str()) {
                return Err(FiniteGroupAdmissionErrorV1::DuplicateCoordinateName {
                    name: name.clone(),
                });
            }
        }
        for (index, unit_id) in self.coordinate_unit_ids.iter().enumerate() {
            if unit_id.trim().is_empty() {
                return Err(FiniteGroupAdmissionErrorV1::EmptyCoordinateUnitId { index });
            }
        }

        validate_square_matrix(&self.metric, dimension, "finite-group metric")
            .map_err(FiniteGroupAdmissionErrorV1::MetricAdmission)?;
        if self.elements.is_empty() {
            return Err(FiniteGroupAdmissionErrorV1::NoElements);
        }
        if self.elements.len() > profile.maximum_elements {
            return Err(FiniteGroupAdmissionErrorV1::ElementCountLimitExceeded {
                actual: self.elements.len(),
                maximum: profile.maximum_elements,
            });
        }
        let expected_table_entries = self
            .elements
            .len()
            .checked_mul(self.elements.len())
            .ok_or(FiniteGroupAdmissionErrorV1::InvalidProfile)?;
        if expected_table_entries > profile.maximum_table_entries {
            return Err(FiniteGroupAdmissionErrorV1::TableEntryBudgetExceeded {
                actual: expected_table_entries,
                maximum: profile.maximum_table_entries,
            });
        }
        let associativity_checks = expected_table_entries
            .checked_mul(self.elements.len())
            .ok_or(FiniteGroupAdmissionErrorV1::InvalidProfile)?;
        if associativity_checks > profile.maximum_associativity_checks {
            return Err(FiniteGroupAdmissionErrorV1::AssociativityBudgetExceeded {
                actual: associativity_checks,
                maximum: profile.maximum_associativity_checks,
            });
        }
        if self.multiplication_table.len() != expected_table_entries {
            return Err(FiniteGroupAdmissionErrorV1::IncompleteMultiplicationTable {
                expected: expected_table_entries,
                actual: self.multiplication_table.len(),
            });
        }

        let metric_scale = max_abs(&self.metric);
        if !metric_scale.is_finite() || metric_scale == 0.0 {
            return Err(FiniteGroupAdmissionErrorV1::MetricAdmission(
                LinearGroupActionAdmissionError::MetricNotPositiveDefinite { pivot_index: 0 },
            ));
        }
        let lower = cholesky_lower_scaled_metric(&self.metric, metric_scale)
            .map_err(FiniteGroupAdmissionErrorV1::MetricAdmission)?;
        let metric_symmetry_defect = relative_symmetry_defect(
            &self.metric,
            metric_scale,
            &lower,
        );
        if !metric_symmetry_defect.is_finite()
            || metric_symmetry_defect > profile.metric_symmetry_relative_tolerance
        {
            return Err(FiniteGroupAdmissionErrorV1::MetricAdmission(
                LinearGroupActionAdmissionError::MetricNotSymmetric {
                    relative_defect: metric_symmetry_defect,
                },
            ));
        }

        // As in A1, retain the caller's exact input and propagate only the
        // exactly symmetric metric that passed the declared noise tolerance.
        let mut effective_action = self.clone();
        for (row_index, row) in effective_action.metric.iter_mut().enumerate() {
            for (column_index, value) in row.iter_mut().enumerate() {
                *value = 0.5
                    * (self.metric[row_index][column_index] / metric_scale
                        + self.metric[column_index][row_index] / metric_scale)
                    * metric_scale;
            }
        }

        let mut element_ids = BTreeSet::new();
        let mut whitened_elements = BTreeMap::new();
        let mut element_diagnostics = Vec::with_capacity(self.elements.len());
        let identity = identity_matrix(dimension);
        let mut maximum_isometry_defect = 0.0_f64;
        for (index, element) in self.elements.iter().enumerate() {
            if element.id.trim().is_empty() {
                return Err(FiniteGroupAdmissionErrorV1::EmptyElementId { index });
            }
            if !element_ids.insert(element.id.as_str()) {
                return Err(FiniteGroupAdmissionErrorV1::DuplicateElementId {
                    element_id: element.id.clone(),
                });
            }
            validate_square_matrix(
                &element.matrix,
                dimension,
                &format!("finite-group element:{}", element.id),
            )
            .map_err(|source| FiniteGroupAdmissionErrorV1::MatrixAdmission {
                element_id: element.id.clone(),
                source,
            })?;

            let whitened = metric_whiten_matrix(&element.matrix, &lower).ok_or_else(|| {
                FiniteGroupAdmissionErrorV1::NonFiniteWhitenedElement {
                    element_id: element.id.clone(),
                }
            })?;
            let gram = matrix_product(&matrix_transpose(&whitened), &whitened);
            let isometry_defect = matrix_rms_defect(&gram, &identity);
            if !isometry_defect.is_finite()
                || isometry_defect > profile.element_isometry_relative_tolerance
            {
                return Err(FiniteGroupAdmissionErrorV1::ElementNotMetricIsometry {
                    element_id: element.id.clone(),
                    relative_defect: isometry_defect,
                });
            }
            maximum_isometry_defect = maximum_isometry_defect.max(isometry_defect);
            element_diagnostics.push(FiniteGroupElementDiagnosticV1 {
                element_id: element.id.clone(),
                metric_isometry_relative_defect: isometry_defect,
            });
            whitened_elements.insert(element.id.clone(), whitened);
        }

        if !element_ids.contains(self.identity_element_id.as_str()) {
            return Err(FiniteGroupAdmissionErrorV1::IdentityElementMissing {
                element_id: self.identity_element_id.clone(),
            });
        }
        let identity_whitened = &whitened_elements[&self.identity_element_id];
        let identity_defect = matrix_rms_defect(identity_whitened, &identity);
        if !identity_defect.is_finite()
            || identity_defect > profile.multiplication_relative_tolerance
        {
            return Err(FiniteGroupAdmissionErrorV1::IdentityMatrixMismatch {
                relative_defect: identity_defect,
            });
        }

        let mut multiplication_lookup = BTreeMap::new();
        for entry in &self.multiplication_table {
            if !element_ids.contains(entry.left_element_id.as_str()) {
                return Err(FiniteGroupAdmissionErrorV1::UnknownMultiplicationOperand {
                    element_id: entry.left_element_id.clone(),
                });
            }
            if !element_ids.contains(entry.right_element_id.as_str()) {
                return Err(FiniteGroupAdmissionErrorV1::UnknownMultiplicationOperand {
                    element_id: entry.right_element_id.clone(),
                });
            }
            if !element_ids.contains(entry.product_element_id.as_str()) {
                return Err(FiniteGroupAdmissionErrorV1::UnknownMultiplicationProduct {
                    element_id: entry.product_element_id.clone(),
                });
            }
            let key = (entry.left_element_id.clone(), entry.right_element_id.clone());
            if multiplication_lookup
                .insert(key, entry.product_element_id.clone())
                .is_some()
            {
                return Err(FiniteGroupAdmissionErrorV1::DuplicateMultiplicationEntry {
                    left: entry.left_element_id.clone(),
                    right: entry.right_element_id.clone(),
                });
            }
        }
        if multiplication_lookup.len() != expected_table_entries {
            return Err(FiniteGroupAdmissionErrorV1::IncompleteMultiplicationTable {
                expected: expected_table_entries,
                actual: multiplication_lookup.len(),
            });
        }

        let element_names: Vec<String> =
            self.elements.iter().map(|element| element.id.clone()).collect();
        for element_id in &element_names {
            if multiplication_lookup
                .get(&(element_id.clone(), self.identity_element_id.clone()))
                .is_none_or(|product| product != element_id)
            {
                return Err(FiniteGroupAdmissionErrorV1::IdentityTableViolation {
                    side: "right",
                    element_id: element_id.clone(),
                });
            }
            if multiplication_lookup
                .get(&(self.identity_element_id.clone(), element_id.clone()))
                .is_none_or(|product| product != element_id)
            {
                return Err(FiniteGroupAdmissionErrorV1::IdentityTableViolation {
                    side: "left",
                    element_id: element_id.clone(),
                });
            }
            let has_two_sided_inverse = element_names.iter().any(|candidate| {
                multiplication_lookup
                    .get(&(element_id.clone(), candidate.clone()))
                    .is_some_and(|product| product == &self.identity_element_id)
                    && multiplication_lookup
                        .get(&(candidate.clone(), element_id.clone()))
                        .is_some_and(|product| product == &self.identity_element_id)
            });
            if !has_two_sided_inverse {
                return Err(FiniteGroupAdmissionErrorV1::MissingTwoSidedInverse {
                    element_id: element_id.clone(),
                });
            }
        }

        let mut checked_associativity_triples = 0usize;
        for left in &element_names {
            for middle in &element_names {
                for right in &element_names {
                    checked_associativity_triples =
                        checked_associativity_triples.saturating_add(1);
                    let left_middle = multiplication_lookup
                        .get(&(left.clone(), middle.clone()))
                        .expect("complete table checked above");
                    let middle_right = multiplication_lookup
                        .get(&(middle.clone(), right.clone()))
                        .expect("complete table checked above");
                    let left_associated = multiplication_lookup
                        .get(&(left_middle.clone(), right.clone()))
                        .expect("complete table checked above");
                    let right_associated = multiplication_lookup
                        .get(&(left.clone(), middle_right.clone()))
                        .expect("complete table checked above");
                    if left_associated != right_associated {
                        return Err(FiniteGroupAdmissionErrorV1::NonAssociativeMultiplicationTable {
                            left: left.clone(),
                            middle: middle.clone(),
                            right: right.clone(),
                        });
                    }
                }
            }
        }

        let mut maximum_multiplication_defect = 0.0_f64;
        for entry in &self.multiplication_table {
            let left = &whitened_elements[&entry.left_element_id];
            let right = &whitened_elements[&entry.right_element_id];
            let product = &whitened_elements[&entry.product_element_id];
            let computed = matrix_product(left, right);
            let defect = matrix_rms_defect(&computed, product);
            if !defect.is_finite()
                || defect > profile.multiplication_relative_tolerance
            {
                return Err(
                    FiniteGroupAdmissionErrorV1::MultiplicationRepresentationMismatch {
                        left: entry.left_element_id.clone(),
                        right: entry.right_element_id.clone(),
                        product: entry.product_element_id.clone(),
                        relative_defect: defect,
                    },
                );
            }
            maximum_multiplication_defect = maximum_multiplication_defect.max(defect);
        }

        // Retain the exact symmetric metric in the admitted effective declaration.
        let effective_action = effective_action;
        Ok(AdmittedFiniteLinearGroupActionV1 {
            action: effective_action,
            declared_action: self.clone(),
            profile,
            diagnostics: FiniteGroupAdmissionDiagnosticsV1 {
                metric_symmetry_relative_defect: metric_symmetry_defect,
                identity_matrix_relative_defect: identity_defect,
                maximum_element_isometry_relative_defect: maximum_isometry_defect,
                maximum_multiplication_relative_defect: maximum_multiplication_defect,
                checked_multiplication_pairs: expected_table_entries,
                checked_associativity_triples,
                element_diagnostics,
            },
            metric_scale,
            scaled_metric_cholesky_lower: lower,
            multiplication_lookup,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix2;

    fn quarter_turn(power: usize) -> Vec<Vec<f64>> {
        match power % 4 {
            0 => vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            1 => vec![vec![0.0, -1.0], vec![1.0, 0.0]],
            2 => vec![vec![-1.0, 0.0], vec![0.0, -1.0]],
            _ => vec![vec![0.0, 1.0], vec![-1.0, 0.0]],
        }
    }

    fn matrix_product_rows(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
        matrix_product(left, right)
    }

    fn c4_action() -> FiniteLinearGroupActionV1 {
        let elements: Vec<FiniteGroupElementV1> = (0..4)
            .map(|power| FiniteGroupElementV1 {
                id: format!("r{power}"),
                matrix: quarter_turn(power),
            })
            .collect();
        let multiplication_table = (0..4)
            .flat_map(|left| {
                (0..4).map(move |right| FiniteGroupMultiplicationV1 {
                    left_element_id: format!("r{left}"),
                    right_element_id: format!("r{right}"),
                    product_element_id: format!("r{}", (left + right) % 4),
                })
            })
            .collect();
        FiniteLinearGroupActionV1 {
            action_id: "C4".to_string(),
            coordinate_basis_id: "xy".to_string(),
            coordinate_names: vec!["x".to_string(), "y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            identity_element_id: "r0".to_string(),
            elements,
            multiplication_table,
        }
    }

    fn d4_product(left: &str, right: &str) -> String {
        let left_reflection = left.starts_with('f');
        let right_reflection = right.starts_with('f');
        let left_power = left[1..].parse::<usize>().unwrap();
        let right_power = right[1..].parse::<usize>().unwrap();
        match (left_reflection, right_reflection) {
            (false, false) => format!("r{}", (left_power + right_power) % 4),
            (false, true) => format!("f{}", (left_power + right_power) % 4),
            (true, false) => format!("f{}", (left_power + 4 - right_power) % 4),
            (true, true) => format!("r{}", (left_power + 4 - right_power) % 4),
        }
    }

    fn d4_action() -> FiniteLinearGroupActionV1 {
        let reflection = vec![vec![1.0, 0.0], vec![0.0, -1.0]];
        let mut elements = Vec::with_capacity(8);
        for power in 0..4 {
            elements.push(FiniteGroupElementV1 {
                id: format!("r{power}"),
                matrix: quarter_turn(power),
            });
        }
        for power in 0..4 {
            elements.push(FiniteGroupElementV1 {
                id: format!("f{power}"),
                matrix: matrix_product_rows(&quarter_turn(power), &reflection),
            });
        }
        let ids: Vec<String> = elements.iter().map(|element| element.id.clone()).collect();
        let multiplication_table = ids
            .iter()
            .flat_map(|left| {
                ids.iter().map(move |right| FiniteGroupMultiplicationV1 {
                    left_element_id: left.clone(),
                    right_element_id: right.clone(),
                    product_element_id: d4_product(left, right),
                })
            })
            .collect();
        FiniteLinearGroupActionV1 {
            action_id: "D4".to_string(),
            coordinate_basis_id: "xy".to_string(),
            coordinate_names: vec!["x".to_string(), "y".to_string()],
            coordinate_unit_ids: vec!["1".to_string(), "1".to_string()],
            metric: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            identity_element_id: "r0".to_string(),
            elements,
            multiplication_table,
        }
    }

    fn transformed_d4() -> FiniteLinearGroupActionV1 {
        let mut action = d4_action();
        let transform = Matrix2::new(1.0, 1.0, 0.0, 1.0);
        let inverse = transform.try_inverse().unwrap();
        let metric = inverse.transpose() * Matrix2::identity() * inverse;
        action.action_id = "D4-sheared".to_string();
        action.coordinate_basis_id = "xy-sheared".to_string();
        action.coordinate_names = vec!["X".to_string(), "Y".to_string()];
        action.metric = metric
            .row_iter()
            .map(|row| row.iter().copied().collect())
            .collect();
        for element in &mut action.elements {
            let matrix = Matrix2::from_row_slice(&[
                element.matrix[0][0],
                element.matrix[0][1],
                element.matrix[1][0],
                element.matrix[1][1],
            ]);
            let transformed = transform * matrix * inverse;
            element.matrix = transformed
                .row_iter()
                .map(|row| row.iter().copied().collect())
                .collect();
        }
        action
    }

    #[test]
    fn finite_group_admission_accepts_c4_and_retains_complete_cayley_table() {
        let admitted = c4_action()
            .try_admit(FiniteGroupAdmissionProfileV1::default())
            .unwrap();
        assert_eq!(admitted.order(), 4);
        assert_eq!(admitted.product_of("r1", "r3"), Some("r0"));
        assert_eq!(admitted.product_of("r2", "r2"), Some("r0"));
        assert_eq!(admitted.diagnostics().checked_multiplication_pairs, 16);
        assert_eq!(admitted.diagnostics().checked_associativity_triples, 64);
        assert!(admitted.diagnostics().maximum_multiplication_relative_defect < 1e-12);
    }

    #[test]
    fn finite_group_admission_accepts_d4_with_reflections_as_first_class_elements() {
        let admitted = d4_action()
            .try_admit(FiniteGroupAdmissionProfileV1::default())
            .unwrap();
        assert_eq!(admitted.order(), 8);
        assert_eq!(admitted.product_of("f1", "f3"), Some("r2"));
        let reflection = admitted.element_matrix("f0").unwrap();
        let determinant =
            reflection[0][0] * reflection[1][1] - reflection[0][1] * reflection[1][0];
        assert_eq!(determinant, -1.0);
        assert_eq!(admitted.diagnostics().checked_multiplication_pairs, 64);
        assert_eq!(admitted.diagnostics().checked_associativity_triples, 512);
    }

    #[test]
    fn finite_group_admission_is_covariant_under_nonorthogonal_basis_change() {
        let canonical = d4_action()
            .try_admit(FiniteGroupAdmissionProfileV1::default())
            .unwrap();
        let transformed = transformed_d4()
            .try_admit(FiniteGroupAdmissionProfileV1::default())
            .unwrap();
        assert_eq!(canonical.order(), transformed.order());
        assert_eq!(
            canonical.action().multiplication_table,
            transformed.action().multiplication_table
        );
        assert!(
            (canonical.diagnostics().maximum_element_isometry_relative_defect
                - transformed.diagnostics().maximum_element_isometry_relative_defect)
                .abs()
                < 1e-12
        );
        assert!(
            (canonical.diagnostics().maximum_multiplication_relative_defect
                - transformed.diagnostics().maximum_multiplication_relative_defect)
                .abs()
                < 1e-12
        );
        for element in &canonical.action().elements {
            assert!(
                transformed
                    .element_matrix(&element.id)
                    .is_some()
            );
        }
    }

    #[test]
    fn finite_group_admission_rejects_incomplete_and_duplicate_tables() {
        let mut missing = c4_action();
        missing.multiplication_table.pop();
        assert!(matches!(
            missing.try_admit(FiniteGroupAdmissionProfileV1::default()),
            Err(FiniteGroupAdmissionErrorV1::IncompleteMultiplicationTable {
                expected: 16,
                actual: 15
            })
        ));

        let mut duplicate = c4_action();
        duplicate.multiplication_table[1] = duplicate.multiplication_table[0].clone();
        assert!(matches!(
            duplicate.try_admit(FiniteGroupAdmissionProfileV1::default()),
            Err(FiniteGroupAdmissionErrorV1::DuplicateMultiplicationEntry { .. })
        ));
    }

    #[test]
    fn finite_group_admission_rejects_matrix_table_mismatch() {
        let mut action = c4_action();
        let entry = action
            .multiplication_table
            .iter_mut()
            .find(|entry| entry.left_element_id == "r1" && entry.right_element_id == "r1")
            .unwrap();
        entry.product_element_id = "r3".to_string();
        assert!(matches!(
            action.try_admit(FiniteGroupAdmissionProfileV1::default()),
            Err(
                FiniteGroupAdmissionErrorV1::NonAssociativeMultiplicationTable { .. }
                    | FiniteGroupAdmissionErrorV1::MultiplicationRepresentationMismatch { .. }
            )
        ));
    }

    #[test]
    fn finite_group_admission_rejects_non_isometric_element() {
        let mut action = c4_action();
        action.elements[1].matrix = vec![vec![2.0, 0.0], vec![0.0, 1.0]];
        assert!(matches!(
            action.try_admit(FiniteGroupAdmissionProfileV1::default()),
            Err(FiniteGroupAdmissionErrorV1::ElementNotMetricIsometry { .. })
        ));
    }

    #[test]
    fn finite_group_admission_rejects_wrong_identity_and_missing_inverse() {
        let mut wrong_identity = c4_action();
        wrong_identity.identity_element_id = "r1".to_string();
        assert!(matches!(
            wrong_identity.try_admit(FiniteGroupAdmissionProfileV1::default()),
            Err(FiniteGroupAdmissionErrorV1::IdentityMatrixMismatch { .. })
                | Err(FiniteGroupAdmissionErrorV1::IdentityTableViolation { .. })
        ));

        let mut no_inverse = c4_action();
        no_inverse.multiplication_table = no_inverse
            .multiplication_table
            .into_iter()
            .map(|mut entry| {
                if entry.left_element_id == "r1" && entry.right_element_id == "r3" {
                    entry.product_element_id = "r1".to_string();
                }
                entry
            })
            .collect();
        assert!(matches!(
            no_inverse.try_admit(FiniteGroupAdmissionProfileV1 {
                multiplication_relative_tolerance: 1.0,
                ..FiniteGroupAdmissionProfileV1::default()
            }),
            Err(FiniteGroupAdmissionErrorV1::IdentityTableViolation { .. })
                | Err(FiniteGroupAdmissionErrorV1::MissingTwoSidedInverse { .. })
                | Err(FiniteGroupAdmissionErrorV1::NonAssociativeMultiplicationTable { .. })
                | Err(FiniteGroupAdmissionErrorV1::MultiplicationRepresentationMismatch { .. })
        ));
    }

    #[test]
    fn finite_group_admission_rejects_budget_excess_before_table_validation() {
        let mut profile = FiniteGroupAdmissionProfileV1::default();
        profile.maximum_elements = 3;
        assert!(matches!(
            c4_action().try_admit(profile),
            Err(FiniteGroupAdmissionErrorV1::ElementCountLimitExceeded {
                actual: 4,
                maximum: 3
            })
        ));
    }
}
