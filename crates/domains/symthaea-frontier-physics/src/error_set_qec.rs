// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![allow(clippy::needless_range_loop)]
// Dense small-instance error-set contractions use explicit register-index loops.

//! General finite-dimensional Knill-Laflamme QEC-matrix construction.
//!
//! Given an isometric code embedding V and a declared finite error set E_k,
//! this module constructs
//!
//! A[(i,k),(j,l)] = <c_i| E_k^dagger E_l |c_j>,
//!
//! using the Gram vectors E_k |c_i>. It computes the best Hilbert-Schmidt
//! projection onto the exact Knill-Laflamme form I_K tensor lambda. The Gram
//! construction makes A positive semidefinite in exact arithmetic; this module
//! deliberately does not claim a floating-point minimum-eigenvalue certificate,
//! a Hellinger distance, an environment-leakage diamond norm, or an approximate
//! universal Petz recovery map.

use num_complex::Complex64;

/// A row-major dense complex matrix used by the small-instance QEC oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexMatrix {
    rows: usize,
    columns: usize,
    entries: Vec<Complex64>,
}

/// Validation or numerical failure while constructing a QEC matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorSetQecError {
    EmptyMatrix,
    DimensionOverflow,
    EntryCount { expected: usize, actual: usize },
    NonFiniteEntry { index: usize },
    InvalidTolerance(f64),
    EmptyErrorSet,
    InvalidErrorOperatorShape {
        index: usize,
        rows: usize,
        columns: usize,
        expected: usize,
    },
    CodeIsometryResidual { residual: f64, tolerance: f64 },
    NonFiniteDiagnostic(&'static str),
}

impl core::fmt::Display for ErrorSetQecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "matrix dimensions must be nonzero"),
            Self::DimensionOverflow => write!(f, "matrix element count overflows usize"),
            Self::EntryCount { expected, actual } => {
                write!(f, "matrix needs {expected} entries, received {actual}")
            }
            Self::NonFiniteEntry { index } => {
                write!(f, "matrix entry {index} contains a non-finite component")
            }
            Self::InvalidTolerance(value) => {
                write!(f, "tolerance must be positive and finite, got {value}")
            }
            Self::EmptyErrorSet => write!(f, "the declared error set must be nonempty"),
            Self::InvalidErrorOperatorShape {
                index,
                rows,
                columns,
                expected,
            } => write!(
                f,
                "error operator {index} has shape {rows}x{columns}, expected {expected}x{expected}"
            ),
            Self::CodeIsometryResidual { residual, tolerance } => write!(
                f,
                "code embedding is not isometric: residual {residual} exceeds {tolerance}"
            ),
            Self::NonFiniteDiagnostic(name) => {
                write!(f, "computed QEC diagnostic {name} is non-finite")
            }
        }
    }
}

impl std::error::Error for ErrorSetQecError {}

impl ComplexMatrix {
    /// Construct a finite, nonempty row-major complex matrix.
    pub fn new(
        rows: usize,
        columns: usize,
        entries: Vec<Complex64>,
    ) -> Result<Self, ErrorSetQecError> {
        if rows == 0 || columns == 0 {
            return Err(ErrorSetQecError::EmptyMatrix);
        }
        let expected = rows
            .checked_mul(columns)
            .ok_or(ErrorSetQecError::DimensionOverflow)?;
        if entries.len() != expected {
            return Err(ErrorSetQecError::EntryCount {
                expected,
                actual: entries.len(),
            });
        }
        for (index, value) in entries.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(ErrorSetQecError::NonFiniteEntry { index });
            }
        }
        Ok(Self {
            rows,
            columns,
            entries,
        })
    }

    /// Build a matrix from a row-major real-valued fixture.
    pub fn from_real(
        rows: usize,
        columns: usize,
        entries: &[f64],
    ) -> Result<Self, ErrorSetQecError> {
        Self::new(
            rows,
            columns,
            entries.iter().copied().map(|value| Complex64::new(value, 0.0)).collect(),
        )
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub fn entries(&self) -> &[Complex64] {
        &self.entries
    }

    pub fn get(&self, row: usize, column: usize) -> Complex64 {
        self.entries[row * self.columns + column]
    }

    fn frobenius_norm(&self) -> f64 {
        self.entries
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }
}

/// QEC-matrix diagnostics for a code and a fixed finite error set.
///
/// Index convention is row=(logical_i,error_k) and
/// column=(logical_j,error_l), flattened as logical * error_count + error.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorSetQecDiagnostics {
    pub code_dimension: usize,
    pub physical_dimension: usize,
    pub error_count: usize,
    pub qec_matrix: ComplexMatrix,
    /// Error-space matrix lambda minimizing ||A - I_K tensor lambda||_F.
    pub knill_laflamme_lambda: ComplexMatrix,
    /// Maximum absolute A_ab - conjugate(A_ba).
    pub hermiticity_residual: f64,
    /// Frobenius norm of A - I_K tensor lambda.
    pub kl_projection_frobenius_residual: f64,
    /// Relative KL projection residual, normalized by max(1, ||A||_F).
    pub kl_projection_relative_residual: f64,
    /// Maximum residual of V^dagger V - I_K.
    pub code_isometry_residual: f64,
}

impl ErrorSetQecDiagnostics {
    /// Whether the supplied tolerance accepts the exact Knill-Laflamme form.
    ///
    /// This is an exact-condition diagnostic for the supplied finite error set,
    /// not an approximate-universal-recovery theorem certificate.
    pub fn exact_kl_within(&self, tolerance: f64) -> bool {
        tolerance.is_finite()
            && tolerance > 0.0
            && self.kl_projection_relative_residual <= tolerance
            && self.hermiticity_residual <= tolerance
            && self.code_isometry_residual <= tolerance
    }
}

/// Construct the complex QEC matrix and its exact-KL projection.
///
/// code_isometry has shape physical_dimension x code_dimension. Each error
/// operator must be a square physical_dimension matrix. The code embedding is
/// checked against V^dagger V = I before the Gram matrix is computed.
///
/// No error operator is silently normalized, orthogonalized, or repaired.
pub fn compute_error_set_qec_diagnostics(
    code_isometry: &ComplexMatrix,
    error_operators: &[ComplexMatrix],
    tolerance: f64,
) -> Result<ErrorSetQecDiagnostics, ErrorSetQecError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(ErrorSetQecError::InvalidTolerance(tolerance));
    }
    if error_operators.is_empty() {
        return Err(ErrorSetQecError::EmptyErrorSet);
    }

    let physical_dimension = code_isometry.rows();
    let code_dimension = code_isometry.columns();
    let error_count = error_operators.len();
    if code_dimension > physical_dimension {
        return Err(ErrorSetQecError::CodeIsometryResidual {
            residual: f64::INFINITY,
            tolerance,
        });
    }

    for (index, error) in error_operators.iter().enumerate() {
        if error.rows() != physical_dimension || error.columns() != physical_dimension {
            return Err(ErrorSetQecError::InvalidErrorOperatorShape {
                index,
                rows: error.rows(),
                columns: error.columns(),
                expected: physical_dimension,
            });
        }
    }

    // Validate the code embedding before using its columns as normalized code states.
    let mut code_isometry_residual: f64 = 0.0;
    for logical_row in 0..code_dimension {
        for logical_column in 0..code_dimension {
            let mut overlap = Complex64::new(0.0, 0.0);
            for physical in 0..physical_dimension {
                overlap += code_isometry.get(physical, logical_row).conj()
                    * code_isometry.get(physical, logical_column);
            }
            let expected = if logical_row == logical_column {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            code_isometry_residual = code_isometry_residual.max((overlap - expected).norm());
        }
    }
    if !code_isometry_residual.is_finite() {
        return Err(ErrorSetQecError::NonFiniteDiagnostic("code isometry"));
    }
    if code_isometry_residual > tolerance {
        return Err(ErrorSetQecError::CodeIsometryResidual {
            residual: code_isometry_residual,
            tolerance,
        });
    }

    // Precompute the Gram vectors E_k V[:,i], indexed [error][logical][physical].
    let mut error_encoded = vec![
        vec![vec![Complex64::new(0.0, 0.0); physical_dimension]; code_dimension];
        error_count
    ];
    for error_index in 0..error_count {
        let error = &error_operators[error_index];
        for logical in 0..code_dimension {
            for output in 0..physical_dimension {
                let mut value = Complex64::new(0.0, 0.0);
                for input in 0..physical_dimension {
                    value += error.get(output, input) * code_isometry.get(input, logical);
                }
                error_encoded[error_index][logical][output] = value;
            }
        }
    }

    let qec_dimension = code_dimension
        .checked_mul(error_count)
        .ok_or(ErrorSetQecError::DimensionOverflow)?;
    let qec_entries = qec_dimension
        .checked_mul(qec_dimension)
        .ok_or(ErrorSetQecError::DimensionOverflow)?;
    let mut qec_values = vec![Complex64::new(0.0, 0.0); qec_entries];

    for logical_row in 0..code_dimension {
        for error_row in 0..error_count {
            let row = logical_row * error_count + error_row;
            for logical_column in 0..code_dimension {
                for error_column in 0..error_count {
                    let column = logical_column * error_count + error_column;
                    let mut overlap = Complex64::new(0.0, 0.0);
                    for physical in 0..physical_dimension {
                        overlap += error_encoded[error_row][logical_row][physical].conj()
                            * error_encoded[error_column][logical_column][physical];
                    }
                    qec_values[row * qec_dimension + column] = overlap;
                }
            }
        }
    }

    let qec_matrix = ComplexMatrix::new(qec_dimension, qec_dimension, qec_values)?;
    let mut hermiticity_residual: f64 = 0.0;
    for row in 0..qec_dimension {
        for column in 0..qec_dimension {
            hermiticity_residual = hermiticity_residual.max(
                (qec_matrix.get(row, column) - qec_matrix.get(column, row).conj()).norm(),
            );
        }
    }

    // Orthogonal projection onto A = I_K tensor lambda in Frobenius geometry:
    // lambda_(k,l) = (1/K) sum_i A_((i,k),(i,l)).
    let mut lambda_values = vec![Complex64::new(0.0, 0.0); error_count * error_count];
    for error_row in 0..error_count {
        for error_column in 0..error_count {
            let mut value = Complex64::new(0.0, 0.0);
            for logical in 0..code_dimension {
                let row = logical * error_count + error_row;
                let column = logical * error_count + error_column;
                value += qec_matrix.get(row, column);
            }
            lambda_values[error_row * error_count + error_column] =
                value / code_dimension as f64;
        }
    }
    let knill_laflamme_lambda =
        ComplexMatrix::new(error_count, error_count, lambda_values)?;

    let mut kl_squared = 0.0;
    for logical_row in 0..code_dimension {
        for error_row in 0..error_count {
            let row = logical_row * error_count + error_row;
            for logical_column in 0..code_dimension {
                for error_column in 0..error_count {
                    let column = logical_column * error_count + error_column;
                    let target = if logical_row == logical_column {
                        knill_laflamme_lambda.get(error_row, error_column)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    kl_squared += (qec_matrix.get(row, column) - target).norm_sqr();
                }
            }
        }
    }
    let kl_projection_frobenius_residual = kl_squared.sqrt();
    let kl_projection_relative_residual =
        kl_projection_frobenius_residual / qec_matrix.frobenius_norm().max(1.0);

    for (name, value) in [
        ("Hermiticity", hermiticity_residual),
        ("KL projection", kl_projection_frobenius_residual),
        ("relative KL projection", kl_projection_relative_residual),
    ] {
        if !value.is_finite() {
            return Err(ErrorSetQecError::NonFiniteDiagnostic(name));
        }
    }

    Ok(ErrorSetQecDiagnostics {
        code_dimension,
        physical_dimension,
        error_count,
        qec_matrix,
        knill_laflamme_lambda,
        hermiticity_residual,
        kl_projection_frobenius_residual,
        kl_projection_relative_residual,
        code_isometry_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::{ComplexMatrix, ErrorSetQecError, compute_error_set_qec_diagnostics};
    use num_complex::Complex64;

    fn repetition_code_isometry() -> ComplexMatrix {
        let mut entries = vec![Complex64::new(0.0, 0.0); 8 * 2];
        entries[0] = Complex64::new(1.0, 0.0);
        entries[7 * 2 + 1] = Complex64::new(1.0, 0.0);
        ComplexMatrix::new(8, 2, entries).unwrap()
    }

    fn bit_flip_operator(mask: usize) -> ComplexMatrix {
        let mut entries = vec![Complex64::new(0.0, 0.0); 8 * 8];
        for column in 0..8 {
            let row = column ^ mask;
            entries[row * 8 + column] = Complex64::new(1.0, 0.0);
        }
        ComplexMatrix::new(8, 8, entries).unwrap()
    }

    #[test]
    fn repetition_code_bit_flip_error_set_satisfies_exact_kl_conditions() {
        let code = repetition_code_isometry();
        let errors = [
            bit_flip_operator(0),
            bit_flip_operator(4),
            bit_flip_operator(2),
            bit_flip_operator(1),
        ];
        let diagnostics =
            compute_error_set_qec_diagnostics(&code, &errors, 1.0e-12).unwrap();
        assert!(diagnostics.exact_kl_within(1.0e-12));
        assert_eq!(diagnostics.error_count, 4);
        assert_eq!(diagnostics.code_dimension, 2);
        assert_eq!(diagnostics.hermiticity_residual, 0.0);
        assert_eq!(diagnostics.kl_projection_frobenius_residual, 0.0);
        for row in 0..4 {
            for column in 0..4 {
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert_eq!(diagnostics.knill_laflamme_lambda.get(row, column), expected);
            }
        }
    }

    #[test]
    fn phase_error_on_repetition_code_is_not_misclassified_as_correctable() {
        let code = repetition_code_isometry();
        let identity = bit_flip_operator(0);
        let mut phase = vec![Complex64::new(0.0, 0.0); 8 * 8];
        for basis in 0..8 {
            let sign = if basis & 4 == 0 { 1.0 } else { -1.0 };
            phase[basis * 8 + basis] = Complex64::new(sign, 0.0);
        }
        let phase = ComplexMatrix::new(8, 8, phase).unwrap();
        let diagnostics =
            compute_error_set_qec_diagnostics(&code, &[identity, phase], 1.0e-12).unwrap();
        assert!(diagnostics.kl_projection_relative_residual > 0.1);
        assert!(!diagnostics.exact_kl_within(1.0e-12));
    }

    #[test]
    fn nonorthogonal_error_set_retains_its_rank_deficient_lambda() {
        let code = repetition_code_isometry();
        let identity = bit_flip_operator(0);
        let duplicated_identity = bit_flip_operator(0);
        let diagnostics = compute_error_set_qec_diagnostics(
            &code,
            &[identity, duplicated_identity],
            1.0e-12,
        )
        .unwrap();
        assert!(diagnostics.exact_kl_within(1.0e-12));
        assert_eq!(
            diagnostics.knill_laflamme_lambda.get(0, 1),
            Complex64::new(1.0, 0.0)
        );
        assert_eq!(
            diagnostics.knill_laflamme_lambda.get(1, 0),
            Complex64::new(1.0, 0.0)
        );
    }

    #[test]
    fn rejects_nonisometric_code_embeddings_and_bad_error_shapes() {
        let invalid_code = ComplexMatrix::from_real(2, 2, &[1.0, 0.0, 1.0, 0.0]).unwrap();
        let identity = ComplexMatrix::from_real(2, 2, &[1.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(matches!(
            compute_error_set_qec_diagnostics(&invalid_code, &[identity.clone()], 1.0e-12),
            Err(ErrorSetQecError::CodeIsometryResidual { .. })
        ));
        let code = ComplexMatrix::from_real(2, 1, &[1.0, 0.0]).unwrap();
        let wrong = ComplexMatrix::from_real(2, 1, &[1.0, 0.0]).unwrap();
        assert!(matches!(
            compute_error_set_qec_diagnostics(&code, &[wrong], 1.0e-12),
            Err(ErrorSetQecError::InvalidErrorOperatorShape { .. })
        ));
    }

    #[test]
    fn rejects_nonfinite_matrix_entries_and_invalid_tolerances() {
        assert!(matches!(
            ComplexMatrix::new(1, 1, vec![Complex64::new(f64::NAN, 0.0)]),
            Err(ErrorSetQecError::NonFiniteEntry { .. })
        ));
        let code = repetition_code_isometry();
        let identity = bit_flip_operator(0);
        assert_eq!(
            compute_error_set_qec_diagnostics(&code, &[identity], 0.0),
            Err(ErrorSetQecError::InvalidTolerance(0.0))
        );
    }
}
