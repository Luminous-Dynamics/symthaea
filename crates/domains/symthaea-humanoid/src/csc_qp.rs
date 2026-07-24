// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical compressed-sparse-column representation for bounded equality QPs.
//!
//! The control stack historically exchanged dense row-major equality matrices.
//! This module introduces one validated sparse representation that can be handed
//! to in-process OSQP/ProxQP/qpOASES adapters without silently changing variable
//! or constraint ordering. Zero pruning is deterministic and structural
//! fingerprints depend on sparsity only, never on state-dependent values.

use serde::{Deserialize, Serialize};

use crate::equality_qp::DenseEqualityQuadraticProgram;

pub const CSC_QP_SCHEMA_VERSION: u32 = 1;
pub const CSC_CANONICAL_ZERO_TOLERANCE: f64 = 1.0e-14;

/// Canonical sparsity fingerprint shared by every dense, CSC, in-process, and
/// external solver path. Numeric values are deliberately excluded.
pub fn canonical_qp_structure_fingerprint(problem: &DenseEqualityQuadraticProgram) -> u64 {
    if !problem.validate() {
        return 0;
    }
    let mut hash = 0xcbf29ce484222325u64;
    for value in [
        problem.diagonal_hessian.len() as u64,
        problem.equality_matrix.len() as u64,
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for row in &problem.equality_matrix {
        for (column, value) in row.iter().enumerate() {
            if value.abs() > 1.0e-14 {
                hash ^= column as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash ^= u64::MAX;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash.max(1)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CscMatrix {
    pub rows: usize,
    pub columns: usize,
    pub column_offsets: Vec<usize>,
    pub row_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl CscMatrix {
    pub fn validate(&self) -> bool {
        self.columns > 0
            && self.column_offsets.len() == self.columns + 1
            && self.column_offsets.first().copied() == Some(0)
            && self.column_offsets.last().copied() == Some(self.values.len())
            && self.row_indices.len() == self.values.len()
            && self.values.iter().all(|value| value.is_finite())
            && self
                .column_offsets
                .windows(2)
                .all(|window| window[0] <= window[1] && window[1] <= self.values.len())
            && (0..self.columns).all(|column| {
                let start = self.column_offsets[column];
                let end = self.column_offsets[column + 1];
                self.row_indices[start..end]
                    .iter()
                    .all(|row| *row < self.rows)
                    && self.row_indices[start..end]
                        .windows(2)
                        .all(|window| window[0] < window[1])
            })
    }

    pub fn from_dense_rows(rows: &[Vec<f64>], columns: usize, zero_tolerance: f64) -> Option<Self> {
        if columns == 0
            || !zero_tolerance.is_finite()
            || zero_tolerance < 0.0
            || rows
                .iter()
                .any(|row| row.len() != columns || row.iter().any(|value| !value.is_finite()))
        {
            return None;
        }
        let mut column_offsets = Vec::with_capacity(columns + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        column_offsets.push(0);
        for column in 0..columns {
            for (row_index, row) in rows.iter().enumerate() {
                let value = row[column];
                if value.abs() > zero_tolerance {
                    row_indices.push(row_index);
                    values.push(value);
                }
            }
            column_offsets.push(values.len());
        }
        let matrix = Self {
            rows: rows.len(),
            columns,
            column_offsets,
            row_indices,
            values,
        };
        matrix.validate().then_some(matrix)
    }

    pub fn multiply(&self, vector: &[f64]) -> Option<Vec<f64>> {
        if !self.validate()
            || vector.len() != self.columns
            || vector.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let mut result = vec![0.0; self.rows];
        for (column, value) in vector.iter().copied().enumerate() {
            let start = self.column_offsets[column];
            let end = self.column_offsets[column + 1];
            for index in start..end {
                result[self.row_indices[index]] += self.values[index] * value;
            }
        }
        Some(result)
    }

    pub fn transpose_multiply(&self, vector: &[f64]) -> Option<Vec<f64>> {
        if !self.validate()
            || vector.len() != self.rows
            || vector.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let mut result = vec![0.0; self.columns];
        for column in 0..self.columns {
            let start = self.column_offsets[column];
            let end = self.column_offsets[column + 1];
            result[column] = (start..end)
                .map(|index| self.values[index] * vector[self.row_indices[index]])
                .sum();
        }
        Some(result)
    }

    /// Fingerprint of the serialized CSC storage layout. This is not the
    /// canonical cross-backend QP sparsity identity.
    pub fn storage_layout_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for value in [self.rows as u64, self.columns as u64] {
            hash ^= value;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for value in self
            .column_offsets
            .iter()
            .chain(self.row_indices.iter())
            .map(|value| *value as u64)
        {
            for byte in value.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash.max(1)
    }

    #[deprecated(
        note = "use storage_layout_fingerprint; solver identities use canonical QP fingerprints"
    )]
    pub fn structure_fingerprint(&self) -> u64 {
        self.storage_layout_fingerprint()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CscEqualityQuadraticProgram {
    pub schema_version: u32,
    pub diagonal_hessian: Vec<f64>,
    pub linear_term: Vec<f64>,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub equality_matrix: CscMatrix,
    pub equality_target: Vec<f64>,
    pub structure_fingerprint: u64,
}

impl CscEqualityQuadraticProgram {
    pub fn from_dense(
        problem: &DenseEqualityQuadraticProgram,
        zero_tolerance: f64,
    ) -> Option<Self> {
        if !problem.validate()
            || !zero_tolerance.is_finite()
            || (zero_tolerance - CSC_CANONICAL_ZERO_TOLERANCE).abs() > f64::EPSILON
        {
            return None;
        }
        let equality_matrix = CscMatrix::from_dense_rows(
            &problem.equality_matrix,
            problem.diagonal_hessian.len(),
            zero_tolerance,
        )?;
        let structure_fingerprint = canonical_qp_structure_fingerprint(problem);
        let sparse = Self {
            schema_version: CSC_QP_SCHEMA_VERSION,
            diagonal_hessian: problem.diagonal_hessian.clone(),
            linear_term: problem.linear_term.clone(),
            lower_bounds: problem.lower_bounds.clone(),
            upper_bounds: problem.upper_bounds.clone(),
            equality_matrix,
            equality_target: problem.equality_target.clone(),
            structure_fingerprint,
        };
        sparse.validate().then_some(sparse)
    }

    pub fn canonical_structure_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for value in [
            self.diagonal_hessian.len() as u64,
            self.equality_matrix.rows as u64,
        ] {
            hash ^= value;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for row in 0..self.equality_matrix.rows {
            for column in 0..self.equality_matrix.columns {
                let start = self.equality_matrix.column_offsets[column];
                let end = self.equality_matrix.column_offsets[column + 1];
                if self.equality_matrix.row_indices[start..end]
                    .binary_search(&row)
                    .is_ok()
                {
                    hash ^= column as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
            hash ^= u64::MAX;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash.max(1)
    }

    pub fn validate(&self) -> bool {
        let variables = self.diagonal_hessian.len();
        self.schema_version == CSC_QP_SCHEMA_VERSION
            && variables > 0
            && self.linear_term.len() == variables
            && self.lower_bounds.len() == variables
            && self.upper_bounds.len() == variables
            && self.equality_matrix.columns == variables
            && self.equality_matrix.rows == self.equality_target.len()
            && self.equality_matrix.validate()
            && self.structure_fingerprint == self.canonical_structure_fingerprint()
            && self
                .diagonal_hessian
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && self.linear_term.iter().all(|value| value.is_finite())
            && self
                .lower_bounds
                .iter()
                .zip(self.upper_bounds.iter())
                .all(|(lower, upper)| lower.is_finite() && upper.is_finite() && lower <= upper)
            && self.equality_target.iter().all(|value| value.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_problem() -> DenseEqualityQuadraticProgram {
        DenseEqualityQuadraticProgram {
            diagonal_hessian: vec![1.0, 2.0, 3.0],
            linear_term: vec![0.0; 3],
            lower_bounds: vec![-1.0; 3],
            upper_bounds: vec![1.0; 3],
            equality_matrix: vec![vec![1.0, 0.0, 2.0], vec![0.0, -1.0, 0.0]],
            equality_target: vec![0.5, -0.25],
        }
    }

    #[test]
    fn dense_conversion_preserves_products() {
        let sparse = CscEqualityQuadraticProgram::from_dense(&dense_problem(), 1.0e-14).unwrap();
        assert!(sparse.validate());
        assert_eq!(
            sparse.equality_matrix.multiply(&[0.5, 0.25, -0.5]).unwrap(),
            vec![-0.5, -0.25]
        );
    }

    #[test]
    fn dense_and_csc_use_one_canonical_structure_identity() {
        let dense = dense_problem();
        let sparse = CscEqualityQuadraticProgram::from_dense(&dense, 1.0e-14).unwrap();
        assert_eq!(
            sparse.structure_fingerprint,
            canonical_qp_structure_fingerprint(&dense)
        );
        assert_eq!(
            sparse.structure_fingerprint,
            sparse.canonical_structure_fingerprint()
        );
    }

    #[test]
    fn structural_fingerprint_ignores_numeric_values() {
        let first = CscEqualityQuadraticProgram::from_dense(&dense_problem(), 1.0e-14).unwrap();
        let mut changed = dense_problem();
        changed.equality_matrix[0][0] = 9.0;
        changed.equality_matrix[0][2] = -3.0;
        let second = CscEqualityQuadraticProgram::from_dense(&changed, 1.0e-14).unwrap();
        assert_eq!(first.structure_fingerprint, second.structure_fingerprint);
    }

    #[test]
    fn noncanonical_pruning_tolerance_is_rejected() {
        assert!(CscEqualityQuadraticProgram::from_dense(&dense_problem(), 1.0e-6).is_none());
    }

    #[test]
    fn bounds_only_qp_has_valid_empty_constraint_matrix() {
        let problem = DenseEqualityQuadraticProgram {
            diagonal_hessian: vec![1.0, 2.0],
            linear_term: vec![0.0, 0.0],
            lower_bounds: vec![-1.0, -1.0],
            upper_bounds: vec![1.0, 1.0],
            equality_matrix: Vec::new(),
            equality_target: Vec::new(),
        };
        let sparse = CscEqualityQuadraticProgram::from_dense(&problem, 1.0e-14).unwrap();
        assert!(sparse.validate());
        assert_eq!(sparse.equality_matrix.rows, 0);
        assert_eq!(
            sparse.equality_matrix.multiply(&[0.0, 0.0]).unwrap(),
            Vec::<f64>::new()
        );
    }

    #[test]
    fn row_indices_must_be_strictly_sorted_per_column() {
        let mut matrix = CscMatrix {
            rows: 2,
            columns: 1,
            column_offsets: vec![0, 2],
            row_indices: vec![1, 1],
            values: vec![1.0, 2.0],
        };
        assert!(!matrix.validate());
        matrix.row_indices = vec![0, 1];
        assert!(matrix.validate());
    }
}
