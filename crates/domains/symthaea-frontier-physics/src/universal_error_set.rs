// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![allow(clippy::needless_range_loop)]
// Dense, fixed-index quantum-information contractions are explicit here.

//! Generic finite-dimensional code/error-set QEC matrix construction.
//!
//! This module builds A[(i,k),(j,l)] = <c_i|E_k^dagger E_l|c_j> from an
//! explicit code isometry and a finite list of physical error operators.
//! It reports the exact Knill-Laflamme subspace projection. It does not yet
//! compute the PSD-constrained Hellinger distance or diamond environment
//! leakage metric required by approximate universal-recovery theorems.

use num_complex::Complex64;

use crate::black_hole_information::QuantumInformationError;

/// Dense QEC matrix limit for the combined dimension K * |E|.
pub const MAX_DENSE_QEC_MATRIX_DIMENSION: usize = 1024;

/// A code isometry V: C^K -> C^d, stored row-major as V[physical, logical].
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorSetCode {
    physical_dimension: usize,
    logical_dimension: usize,
    isometry: Vec<Complex64>,
}

impl ErrorSetCode {
    pub fn new(
        physical_dimension: usize,
        logical_dimension: usize,
        isometry: Vec<Complex64>,
        tolerance: f64,
    ) -> Result<Self, QuantumInformationError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(QuantumInformationError::InvalidTolerance(tolerance));
        }
        if physical_dimension == 0 {
            return Err(QuantumInformationError::InvalidDimension(
                physical_dimension,
            ));
        }
        if logical_dimension == 0 || logical_dimension > physical_dimension {
            return Err(QuantumInformationError::InvalidDimension(logical_dimension));
        }
        let expected = physical_dimension
            .checked_mul(logical_dimension)
            .ok_or(QuantumInformationError::DimensionOverflow)?;
        if isometry.len() != expected {
            return Err(QuantumInformationError::DimensionMismatch {
                expected,
                actual: isometry.len(),
            });
        }
        for (index, value) in isometry.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(QuantumInformationError::NonFiniteMatrix {
                    row: index / logical_dimension,
                    column: index % logical_dimension,
                });
            }
        }

        let mut residual: f64 = 0.0;
        for left in 0..logical_dimension {
            for right in 0..logical_dimension {
                let mut overlap = Complex64::new(0.0, 0.0);
                for physical in 0..physical_dimension {
                    overlap += isometry[physical * logical_dimension + left].conj()
                        * isometry[physical * logical_dimension + right];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                residual = residual.max((overlap - Complex64::new(expected, 0.0)).norm());
            }
        }
        if residual > tolerance {
            return Err(QuantumInformationError::InformationIdentityError {
                residual,
                tolerance,
            });
        }

        Ok(Self {
            physical_dimension,
            logical_dimension,
            isometry,
        })
    }

    pub const fn physical_dimension(&self) -> usize {
        self.physical_dimension
    }

    pub const fn logical_dimension(&self) -> usize {
        self.logical_dimension
    }

    pub fn isometry(&self) -> &[Complex64] {
        &self.isometry
    }

    /// Construct the generic dense QEC matrix for a finite physical error set.
    ///
    /// Error operators need not be unitary. This is code/error-set algebra,
    /// not a channel CPTP check. Physical channel coefficients and TP
    /// constraints are validated separately.
    pub fn qec_matrix(
        &self,
        error_set: &FiniteErrorSet,
    ) -> Result<ErrorSetQecMatrix, QuantumInformationError> {
        if self.physical_dimension != error_set.physical_dimension {
            return Err(QuantumInformationError::DimensionMismatch {
                expected: self.physical_dimension,
                actual: error_set.physical_dimension,
            });
        }
        let dimension = self
            .logical_dimension
            .checked_mul(error_set.len())
            .ok_or(QuantumInformationError::DimensionOverflow)?;
        if dimension > MAX_DENSE_QEC_MATRIX_DIMENSION {
            return Err(QuantumInformationError::DenseDimensionLimit {
                requested: dimension,
                maximum: MAX_DENSE_QEC_MATRIX_DIMENSION,
            });
        }
        let matrix_entries = dimension
            .checked_mul(dimension)
            .ok_or(QuantumInformationError::DimensionOverflow)?;
        let mut entries = vec![Complex64::new(0.0, 0.0); matrix_entries];

        // Cache each E_k V column so all overlaps use identical code inputs.
        let mut error_codewords = vec![
            vec![Complex64::new(0.0, 0.0); self.physical_dimension * self.logical_dimension];
            error_set.len()
        ];
        for (error_index, operator) in error_set.operators.iter().enumerate() {
            for output in 0..self.physical_dimension {
                for logical in 0..self.logical_dimension {
                    let mut value = Complex64::new(0.0, 0.0);
                    for input in 0..self.physical_dimension {
                        value += operator[output * self.physical_dimension + input]
                            * self.isometry[input * self.logical_dimension + logical];
                    }
                    error_codewords[error_index][output * self.logical_dimension + logical] = value;
                }
            }
        }

        for logical_row in 0..self.logical_dimension {
            for error_row in 0..error_set.len() {
                for logical_column in 0..self.logical_dimension {
                    for error_column in 0..error_set.len() {
                        let mut overlap = Complex64::new(0.0, 0.0);
                        for physical in 0..self.physical_dimension {
                            let left = error_codewords[error_row]
                                [physical * self.logical_dimension + logical_row];
                            let right = error_codewords[error_column]
                                [physical * self.logical_dimension + logical_column];
                            overlap += left.conj() * right;
                        }
                        let row = logical_row * error_set.len() + error_row;
                        let column = logical_column * error_set.len() + error_column;
                        entries[row * dimension + column] = overlap;
                    }
                }
            }
        }

        Ok(ErrorSetQecMatrix {
            logical_dimension: self.logical_dimension,
            error_count: error_set.len(),
            entries,
        })
    }
}

/// A finite set of physical error operators E_k, each stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct FiniteErrorSet {
    physical_dimension: usize,
    operators: Vec<Vec<Complex64>>,
}

impl FiniteErrorSet {
    pub fn new(
        physical_dimension: usize,
        operators: Vec<Vec<Complex64>>,
    ) -> Result<Self, QuantumInformationError> {
        if physical_dimension == 0 {
            return Err(QuantumInformationError::InvalidDimension(physical_dimension));
        }
        if operators.is_empty() {
            return Err(QuantumInformationError::InvalidDimension(0));
        }
        let operator_entries = physical_dimension
            .checked_mul(physical_dimension)
            .ok_or(QuantumInformationError::DimensionOverflow)?;
        for (operator_index, operator) in operators.iter().enumerate() {
            if operator.len() != operator_entries {
                return Err(QuantumInformationError::DimensionMismatch {
                    expected: operator_entries,
                    actual: operator.len(),
                });
            }
            for (entry_index, value) in operator.iter().enumerate() {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(QuantumInformationError::NonFiniteMatrix {
                        row: operator_index,
                        column: entry_index,
                    });
                }
            }
        }
        Ok(Self {
            physical_dimension,
            operators,
        })
    }

    pub const fn physical_dimension(&self) -> usize {
        self.physical_dimension
    }

    pub fn len(&self) -> usize {
        self.operators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }

    pub fn operators(&self) -> &[Vec<Complex64>] {
        &self.operators
    }
}

/// Generic QEC matrix with flat index logical_index * error_count + error_index.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorSetQecMatrix {
    logical_dimension: usize,
    error_count: usize,
    entries: Vec<Complex64>,
}

/// Frobenius projection onto the exact Knill-Laflamme subspace.
#[derive(Debug, Clone, PartialEq)]
pub struct KnillLaflammeDiagnostics {
    /// lambda_kl = (1/K) sum_i A[(i,k),(i,l)].
    pub lambda: Vec<Complex64>,
    pub maximum_exact_subspace_residual: f64,
    pub frobenius_exact_subspace_residual: f64,
    pub hermiticity_residual: f64,
}

impl ErrorSetQecMatrix {
    pub fn logical_dimension(&self) -> usize {
        self.logical_dimension
    }

    pub fn error_count(&self) -> usize {
        self.error_count
    }

    pub fn dimension(&self) -> usize {
        self.logical_dimension * self.error_count
    }

    pub fn entries(&self) -> &[Complex64] {
        &self.entries
    }

    pub fn get(
        &self,
        logical_row: usize,
        error_row: usize,
        logical_column: usize,
        error_column: usize,
    ) -> Option<Complex64> {
        if logical_row >= self.logical_dimension
            || logical_column >= self.logical_dimension
            || error_row >= self.error_count
            || error_column >= self.error_count
        {
            return None;
        }
        let dimension = self.dimension();
        let row = logical_row * self.error_count + error_row;
        let column = logical_column * self.error_count + error_column;
        Some(self.entries[row * dimension + column])
    }

    pub fn hermiticity_residual(&self) -> f64 {
        let dimension = self.dimension();
        let mut maximum: f64 = 0.0;
        for row in 0..dimension {
            for column in 0..dimension {
                maximum = maximum.max(
                    (self.entries[row * dimension + column]
                        - self.entries[column * dimension + row].conj())
                    .norm(),
                );
            }
        }
        maximum
    }

    /// Project onto I_K tensor lambda using the Frobenius inner product.
    ///
    /// This is a linear-subspace projection. It does not solve the
    /// positive-semidefinite-constrained minimization required by the
    /// approximate Knill-Laflamme Hellinger metric.
    pub fn knill_laflamme_diagnostics(&self) -> KnillLaflammeDiagnostics {
        let mut lambda = vec![Complex64::new(0.0, 0.0); self.error_count * self.error_count];
        for error_row in 0..self.error_count {
            for error_column in 0..self.error_count {
                for logical in 0..self.logical_dimension {
                    lambda[error_row * self.error_count + error_column] +=
                        self.get(logical, error_row, logical, error_column).unwrap();
                }
                lambda[error_row * self.error_count + error_column] /=
                    self.logical_dimension as f64;
            }
        }

        let mut maximum: f64 = 0.0;
        let mut frobenius_squared = 0.0;
        for logical_row in 0..self.logical_dimension {
            for error_row in 0..self.error_count {
                for logical_column in 0..self.logical_dimension {
                    for error_column in 0..self.error_count {
                        let actual = self
                            .get(logical_row, error_row, logical_column, error_column)
                            .unwrap();
                        let expected = if logical_row == logical_column {
                            lambda[error_row * self.error_count + error_column]
                        } else {
                            Complex64::new(0.0, 0.0)
                        };
                        let residual = (actual - expected).norm();
                        maximum = maximum.max(residual);
                        frobenius_squared += residual * residual;
                    }
                }
            }
        }

        KnillLaflammeDiagnostics {
            lambda,
            maximum_exact_subspace_residual: maximum,
            frobenius_exact_subspace_residual: frobenius_squared.sqrt(),
            hermiticity_residual: self.hermiticity_residual(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorSetCode, FiniteErrorSet, MAX_DENSE_QEC_MATRIX_DIMENSION};
    use crate::black_hole_information::QuantumInformationError;
    use num_complex::Complex64;

    const TOLERANCE: f64 = 1.0e-12;

    fn repetition_code_and_error_set() -> (ErrorSetCode, FiniteErrorSet) {
        let mut isometry = vec![Complex64::new(0.0, 0.0); 8 * 2];
        isometry[0] = Complex64::new(1.0, 0.0);
        isometry[7 * 2 + 1] = Complex64::new(1.0, 0.0);
        let code = ErrorSetCode::new(8, 2, isometry, TOLERANCE).unwrap();

        let masks = [0_usize, 4, 2, 1];
        let mut operators = Vec::new();
        for mask in masks {
            let mut matrix = vec![Complex64::new(0.0, 0.0); 8 * 8];
            for input in 0..8 {
                matrix[(input ^ mask) * 8 + input] = Complex64::new(1.0, 0.0);
            }
            operators.push(matrix);
        }
        let errors = FiniteErrorSet::new(8, operators).unwrap();
        (code, errors)
    }

    #[test]
    fn generic_qec_matrix_reproduces_exact_repetition_code_kl_identity() {
        let (code, errors) = repetition_code_and_error_set();
        let qec = code.qec_matrix(&errors).unwrap();
        assert_eq!(qec.logical_dimension(), 2);
        assert_eq!(qec.error_count(), 4);
        assert_eq!(qec.dimension(), 8);
        assert!(qec.hermiticity_residual() <= TOLERANCE);

        for row in 0..8 {
            for column in 0..8 {
                let expected = if row == column { 1.0 } else { 0.0 };
                let actual = qec.entries()[row * 8 + column];
                assert!((actual - Complex64::new(expected, 0.0)).norm() <= TOLERANCE);
            }
        }

        let diagnostics = qec.knill_laflamme_diagnostics();
        assert!(diagnostics.maximum_exact_subspace_residual <= TOLERANCE);
        assert!(diagnostics.frobenius_exact_subspace_residual <= TOLERANCE);
        assert!(diagnostics.hermiticity_residual <= TOLERANCE);
        for row in 0..4 {
            for column in 0..4 {
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!(
                    (diagnostics.lambda[row * 4 + column] - Complex64::new(expected, 0.0)).norm()
                        <= TOLERANCE
                );
            }
        }
    }

    #[test]
    fn generic_qec_matrix_handles_complex_nonunitary_error_operators() {
        let code = ErrorSetCode::new(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            TOLERANCE,
        )
        .unwrap();
        let errors = FiniteErrorSet::new(
            2,
            vec![
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 2.0),
                ],
            ],
        )
        .unwrap();
        let qec = code.qec_matrix(&errors).unwrap();
        assert!(qec.hermiticity_residual() <= TOLERANCE);
        assert!(qec.get(1, 0, 1, 1).unwrap().im.abs() > 0.0);
        assert!(
            qec.knill_laflamme_diagnostics()
                .maximum_exact_subspace_residual
                > 0.0
        );
    }

    #[test]
    fn generic_qec_constructors_reject_invalid_code_and_error_matrices() {
        assert!(matches!(
            ErrorSetCode::new(
                2,
                1,
                vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
                TOLERANCE
            ),
            Err(QuantumInformationError::InformationIdentityError { .. })
        ));
        assert!(matches!(
            ErrorSetCode::new(2, 1, vec![Complex64::new(f64::NAN, 0.0); 2], TOLERANCE),
            Err(QuantumInformationError::NonFiniteMatrix { .. })
        ));
        assert!(matches!(
            FiniteErrorSet::new(2, Vec::new()),
            Err(QuantumInformationError::InvalidDimension(0))
        ));
        assert!(matches!(
            FiniteErrorSet::new(2, vec![vec![Complex64::new(f64::INFINITY, 0.0); 4]]),
            Err(QuantumInformationError::NonFiniteMatrix { .. })
        ));
    }

    #[test]
    fn generic_qec_matrix_rejects_dimension_mismatch_and_dense_limit() {
        let code = ErrorSetCode::new(
            2,
            1,
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            TOLERANCE,
        )
        .unwrap();
        let mismatched = FiniteErrorSet::new(3, vec![vec![Complex64::new(0.0, 0.0); 9]]).unwrap();
        assert!(matches!(
            code.qec_matrix(&mismatched),
            Err(QuantumInformationError::DimensionMismatch { .. })
        ));

        let one_dimensional_code =
            ErrorSetCode::new(1, 1, vec![Complex64::new(1.0, 0.0)], TOLERANCE).unwrap();
        let too_many_errors = FiniteErrorSet::new(
            1,
            vec![vec![Complex64::new(1.0, 0.0)]; MAX_DENSE_QEC_MATRIX_DIMENSION + 1],
        )
        .unwrap();
        assert!(matches!(
            one_dimensional_code.qec_matrix(&too_many_errors),
            Err(QuantumInformationError::DenseDimensionLimit { .. })
        ));
    }
}
