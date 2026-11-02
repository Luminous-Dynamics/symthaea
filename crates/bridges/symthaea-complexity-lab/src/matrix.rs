//! Exact rational matrix certificates for the ETH-005 auxiliary-matrix step.
//!
//! This module verifies a supplied factorization I - M0 = U * V, which is a
//! certificate that rank(I - M0) <= inner_dimension(U,V). It does not construct
//! M0, search for U/V, or prove the paper's generic matrix-existence theorem.

use std::error::Error;
use std::fmt;

/// Exact rational matrix with one positive common denominator.
///
/// Numerators are row-major. Fields are private so every matrix is dimension-
/// checked and has a positive denominator at construction time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RationalMatrix {
    rows: usize,
    columns: usize,
    numerators: Vec<i128>,
    denominator: i128,
}

impl RationalMatrix {
    pub fn new(
        rows: usize,
        columns: usize,
        numerators: Vec<i128>,
        denominator: i128,
    ) -> Result<Self, MatrixCertificateError> {
        if denominator <= 0 {
            return Err(MatrixCertificateError::NonPositiveDenominator { denominator });
        }
        let expected = rows
            .checked_mul(columns)
            .ok_or(MatrixCertificateError::DimensionOverflow)?;
        if numerators.len() != expected {
            return Err(MatrixCertificateError::DataLengthMismatch {
                expected,
                actual: numerators.len(),
            });
        }
        Ok(Self {
            rows,
            columns,
            numerators,
            denominator,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn denominator(&self) -> i128 {
        self.denominator
    }

    pub fn numerators(&self) -> &[i128] {
        &self.numerators
    }

    fn numerator_at(&self, row: usize, column: usize) -> i128 {
        self.numerators[row * self.columns + column]
    }

    fn identity_minus(&self) -> Result<Self, MatrixCertificateError> {
        if self.rows != self.columns {
            return Err(MatrixCertificateError::NotSquare {
                rows: self.rows,
                columns: self.columns,
            });
        }

        let mut numerators = Vec::new();
        numerators
            .try_reserve_exact(self.numerators.len())
            .map_err(|_| MatrixCertificateError::AllocationFailed {
                entries: self.numerators.len(),
            })?;

        for row in 0..self.rows {
            for column in 0..self.columns {
                let identity_numerator = if row == column {
                    self.denominator
                } else {
                    0
                };
                let value = identity_numerator
                    .checked_sub(self.numerator_at(row, column))
                    .ok_or(MatrixCertificateError::ArithmeticOverflow {
                        row,
                        column,
                        operation: "forming I - M0",
                    })?;
                numerators.push(value);
            }
        }

        Self::new(
            self.rows,
            self.columns,
            numerators,
            self.denominator,
        )
    }
}

/// Errors in exact rational matrix certificate verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatrixCertificateError {
    DimensionOverflow,
    AllocationFailed { entries: usize },
    DataLengthMismatch { expected: usize, actual: usize },
    NonPositiveDenominator { denominator: i128 },
    NotSquare { rows: usize, columns: usize },
    FactorShapeMismatch {
        target_rows: usize,
        target_columns: usize,
        left_rows: usize,
        left_columns: usize,
        right_rows: usize,
        right_columns: usize,
    },
    ArithmeticOverflow {
        row: usize,
        column: usize,
        operation: &'static str,
    },
}

impl fmt::Display for MatrixCertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOverflow => write!(f, "matrix dimensions overflow usize"),
            Self::AllocationFailed { entries } => {
                write!(f, "unable to reserve {entries} matrix entries")
            }
            Self::DataLengthMismatch { expected, actual } => write!(
                f,
                "matrix contains {actual} numerators; expected {expected}"
            ),
            Self::NonPositiveDenominator { denominator } => write!(
                f,
                "rational matrix denominator must be positive, got {denominator}"
            ),
            Self::NotSquare { rows, columns } => {
                write!(f, "I - M0 requires a square matrix, got {rows}x{columns}")
            }
            Self::FactorShapeMismatch {
                target_rows,
                target_columns,
                left_rows,
                left_columns,
                right_rows,
                right_columns,
            } => write!(
                f,
                "cannot multiply {left_rows}x{left_columns} by {right_rows}x{right_columns} \
                 to obtain {target_rows}x{target_columns}"
            ),
            Self::ArithmeticOverflow { row, column, operation } => write!(
                f,
                "checked matrix arithmetic overflow at ({row},{column}) while {operation}"
            ),
        }
    }
}

impl Error for MatrixCertificateError {}

/// Verify exactly that target = left * right over Q.
///
/// Each matrix uses its own common positive denominator. The verifier performs
/// checked cross multiplication; it never converts to floating point. A false
/// factorization returns Ok(false), while malformed dimensions or overflow
/// return an explicit error.
pub fn verify_rank_factorization(
    target: &RationalMatrix,
    left: &RationalMatrix,
    right: &RationalMatrix,
) -> Result<bool, MatrixCertificateError> {
    if left.rows != target.rows
        || right.columns != target.columns
        || left.columns != right.rows
    {
        return Err(MatrixCertificateError::FactorShapeMismatch {
            target_rows: target.rows,
            target_columns: target.columns,
            left_rows: left.rows,
            left_columns: left.columns,
            right_rows: right.rows,
            right_columns: right.columns,
        });
    }

    let factor_denominator = left
        .denominator
        .checked_mul(right.denominator)
        .ok_or(MatrixCertificateError::ArithmeticOverflow {
            row: 0,
            column: 0,
            operation: "multiplying factor denominators",
        })?;

    for row in 0..target.rows {
        for column in 0..target.columns {
            let mut product_numerator = 0i128;
            for inner in 0..left.columns {
                let product = left
                    .numerator_at(row, inner)
                    .checked_mul(right.numerator_at(inner, column))
                    .ok_or(MatrixCertificateError::ArithmeticOverflow {
                        row,
                        column,
                        operation: "multiplying factor entries",
                    })?;
                product_numerator = product_numerator.checked_add(product).ok_or(
                    MatrixCertificateError::ArithmeticOverflow {
                        row,
                        column,
                        operation: "accumulating factor entries",
                    },
                )?;
            }

            let lhs = product_numerator
                .checked_mul(target.denominator)
                .ok_or(MatrixCertificateError::ArithmeticOverflow {
                    row,
                    column,
                    operation: "cross-multiplying product by target denominator",
                })?;
            let rhs = target
                .numerator_at(row, column)
                .checked_mul(factor_denominator)
                .ok_or(MatrixCertificateError::ArithmeticOverflow {
                    row,
                    column,
                    operation: "cross-multiplying target by factor denominators",
                })?;

            if lhs != rhs {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Verify a supplied factorization I - M0 = U * V over Q.
///
/// On success, the certificate establishes rank(I - M0) <= U.columns.
/// It does not establish the separate paper obligation that a suitable M0
/// exists with the required polynomial, endpoint, or support properties.
pub fn verify_identity_minus_rank_factorization(
    m0: &RationalMatrix,
    left: &RationalMatrix,
    right: &RationalMatrix,
) -> Result<bool, MatrixCertificateError> {
    let target = m0.identity_minus()?;
    verify_rank_factorization(&target, left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize, numerators: &[i128], denominator: i128) -> RationalMatrix {
        RationalMatrix::new(rows, columns, numerators.to_vec(), denominator).unwrap()
    }

    #[test]
    fn accepts_rank_one_certificate_for_identity_minus_m0() {
        // M0 = [[1, 0], [0, 0]], hence I-M0 = [[0,0],[0,1]] = [0,1]^T [0,1].
        let m0 = matrix(2, 2, &[1, 0, 0, 0], 1);
        let left = matrix(2, 1, &[0, 1], 1);
        let right = matrix(1, 2, &[0, 1], 1);
        assert!(verify_identity_minus_rank_factorization(&m0, &left, &right).unwrap());
    }

    #[test]
    fn rejects_mutated_rank_factorization_entry() {
        let m0 = matrix(2, 2, &[1, 0, 0, 0], 1);
        let left = matrix(2, 1, &[0, 1], 1);
        let mutated_right = matrix(1, 2, &[1, 1], 1);
        assert!(!verify_identity_minus_rank_factorization(&m0, &left, &mutated_right).unwrap());
    }

    #[test]
    fn accepts_factorization_with_distinct_rational_denominators() {
        // M0 = I/2, so I-M0 = I/2 = (I/2) * I.
        let m0 = matrix(2, 2, &[1, 0, 0, 1], 2);
        let left = matrix(2, 2, &[1, 0, 0, 1], 2);
        let right = matrix(2, 2, &[1, 0, 0, 1], 1);
        assert!(verify_identity_minus_rank_factorization(&m0, &left, &right).unwrap());
    }

    #[test]
    fn rejects_non_square_m0() {
        let m0 = matrix(1, 2, &[1, 0], 1);
        let left = matrix(1, 1, &[1], 1);
        let right = matrix(1, 2, &[1, 0], 1);
        assert!(matches!(
            verify_identity_minus_rank_factorization(&m0, &left, &right),
            Err(MatrixCertificateError::NotSquare { .. })
        ));
    }

    #[test]
    fn rejects_factor_shape_mismatch() {
        let target = matrix(2, 2, &[1, 0, 0, 1], 1);
        let left = matrix(2, 1, &[1, 0], 1);
        let right = matrix(2, 2, &[1, 0, 0, 1], 1);
        assert!(matches!(
            verify_rank_factorization(&target, &left, &right),
            Err(MatrixCertificateError::FactorShapeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_fixed_width_denominator_overflow() {
        let target = matrix(1, 1, &[0], 1);
        let left = matrix(1, 1, &[0], i128::MAX);
        let right = matrix(1, 1, &[0], 2);
        assert!(matches!(
            verify_rank_factorization(&target, &left, &right),
            Err(MatrixCertificateError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn rejects_nonpositive_matrix_denominator() {
        assert!(matches!(
            RationalMatrix::new(1, 1, vec![0], 0),
            Err(MatrixCertificateError::NonPositiveDenominator { .. })
        ));
    }
}
