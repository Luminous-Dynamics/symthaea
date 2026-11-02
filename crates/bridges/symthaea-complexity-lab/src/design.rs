//! Small independent certificate checks for exact linear-algebra claims.
//!
//! The first supported witness is lambda^T * C = target, where C is an integer
//! measurement matrix and lambda/target are rational vectors represented with
//! one positive common denominator each.
//!
//! This module verifies supplied witnesses; it does not search for them.

use std::error::Error;
use std::fmt;

/// A rational vector represented exactly as numerators / one positive denominator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RationalVector {
    pub numerators: Vec<i128>,
    pub denominator: i128,
}

impl RationalVector {
    pub fn new(numerators: Vec<i128>, denominator: i128) -> Result<Self, RowSpaceError> {
        if denominator <= 0 {
            return Err(RowSpaceError::NonPositiveDenominator { denominator });
        }
        Ok(Self { numerators, denominator })
    }
}

/// The first exact certificate primitive for the ETH measurement-design verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSpaceCertificate {
    /// Coefficients lambda in lambda^T C = target.
    pub multipliers: RationalVector,
    /// The target row restricted to one non-owner group.
    pub target: RationalVector,
}

/// Fail-closed validation errors. A mathematically false witness is returned as
/// Ok(false); malformed input or arithmetic overflow is an explicit error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowSpaceError {
    EmptyMatrix,
    RaggedMatrix { row: usize, expected_columns: usize, actual_columns: usize },
    MultiplierDimensionMismatch { expected: usize, actual: usize },
    TargetDimensionMismatch { expected: usize, actual: usize },
    NonPositiveDenominator { denominator: i128 },
    ArithmeticOverflow { column: usize, operation: &'static str },
}

impl fmt::Display for RowSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "measurement matrix must contain at least one row"),
            Self::RaggedMatrix { row, expected_columns, actual_columns } => write!(
                f,
                "measurement matrix row {row} has {actual_columns} columns; expected {expected_columns}"
            ),
            Self::MultiplierDimensionMismatch { expected, actual } => write!(
                f,
                "witness has {actual} multipliers; measurement matrix has {expected} rows"
            ),
            Self::TargetDimensionMismatch { expected, actual } => write!(
                f,
                "target has {actual} coordinates; measurement matrix has {expected} columns"
            ),
            Self::NonPositiveDenominator { denominator } => {
                write!(f, "rational denominator must be positive, got {denominator}")
            }
            Self::ArithmeticOverflow { column, operation } => write!(
                f,
                "checked arithmetic overflow in {operation} at target column {column}"
            ),
        }
    }
}

impl Error for RowSpaceError {}

/// Verify exactly that lambda^T * C equals target over Q.
///
/// matrix is row-major with integer entries. Multipliers and target may use
/// different common denominators. For column j the verifier checks:
///
///   (sum_i lambda_num[i] * C[i][j]) * target_den
///       == target_num[j] * lambda_den.
///
/// This is an independent exact witness checker. It is intentionally small and
/// contains no candidate-search logic. It fails closed on malformed dimensions,
/// non-positive denominators, and i128 overflow.
pub fn verify_row_space_witness(
    matrix: &[Vec<i64>],
    certificate: &RowSpaceCertificate,
) -> Result<bool, RowSpaceError> {
    let first_row = matrix.first().ok_or(RowSpaceError::EmptyMatrix)?;
    let columns = first_row.len();

    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != columns {
            return Err(RowSpaceError::RaggedMatrix {
                row: row_index,
                expected_columns: columns,
                actual_columns: row.len(),
            });
        }
    }

    if certificate.multipliers.denominator <= 0 {
        return Err(RowSpaceError::NonPositiveDenominator {
            denominator: certificate.multipliers.denominator,
        });
    }
    if certificate.target.denominator <= 0 {
        return Err(RowSpaceError::NonPositiveDenominator {
            denominator: certificate.target.denominator,
        });
    }

    if certificate.multipliers.numerators.len() != matrix.len() {
        return Err(RowSpaceError::MultiplierDimensionMismatch {
            expected: matrix.len(),
            actual: certificate.multipliers.numerators.len(),
        });
    }
    if certificate.target.numerators.len() != columns {
        return Err(RowSpaceError::TargetDimensionMismatch {
            expected: columns,
            actual: certificate.target.numerators.len(),
        });
    }

    for column in 0..columns {
        let mut lhs_numerator = 0i128;
        for (row_index, row) in matrix.iter().enumerate() {
            let product = certificate.multipliers.numerators[row_index]
                .checked_mul(i128::from(row[column]))
                .ok_or(RowSpaceError::ArithmeticOverflow {
                    column,
                    operation: "multiplier-by-matrix multiplication",
                })?;
            lhs_numerator = lhs_numerator.checked_add(product).ok_or(
                RowSpaceError::ArithmeticOverflow {
                    column,
                    operation: "row-space witness accumulation",
                },
            )?;
        }

        let lhs = lhs_numerator
            .checked_mul(certificate.target.denominator)
            .ok_or(RowSpaceError::ArithmeticOverflow {
                column,
                operation: "left denominator cross multiplication",
            })?;
        let rhs = certificate.target.numerators[column]
            .checked_mul(certificate.multipliers.denominator)
            .ok_or(RowSpaceError::ArithmeticOverflow {
                column,
                operation: "right denominator cross multiplication",
            })?;

        if lhs != rhs {
            return Ok(false);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rational(numerators: &[i128], denominator: i128) -> RationalVector {
        RationalVector::new(numerators.to_vec(), denominator).unwrap()
    }

    #[test]
    fn accepts_exact_rational_row_space_witness() {
        let matrix = vec![vec![1, 0, 1], vec![0, 1, 1]];
        let certificate = RowSpaceCertificate {
            multipliers: rational(&[1, 1], 2),
            target: rational(&[1, 1, 2], 2),
        };
        assert!(verify_row_space_witness(&matrix, &certificate).unwrap());
    }

    #[test]
    fn accepts_different_common_denominators_exactly() {
        let matrix = vec![vec![1, 0], vec![0, 1]];
        let certificate = RowSpaceCertificate {
            multipliers: rational(&[1, 2], 3),
            target: rational(&[2, 4], 6),
        };
        assert!(verify_row_space_witness(&matrix, &certificate).unwrap());
    }

    #[test]
    fn rejects_one_coordinate_mutation() {
        let matrix = vec![vec![1, 0, 1], vec![0, 1, 1]];
        let certificate = RowSpaceCertificate {
            multipliers: rational(&[1, 1], 2),
            target: rational(&[1, 1, 3], 2),
        };
        assert!(!verify_row_space_witness(&matrix, &certificate).unwrap());
    }

    #[test]
    fn accepts_zero_measurement_rows_for_zero_target() {
        let matrix = vec![vec![0, 0, 0]];
        let certificate = RowSpaceCertificate {
            multipliers: rational(&[7], 3),
            target: rational(&[0, 0, 0], 11),
        };
        assert!(verify_row_space_witness(&matrix, &certificate).unwrap());
    }

    #[test]
    fn rejects_multiplier_dimension_mismatch() {
        let matrix = vec![vec![1, 0], vec![0, 1]];
        let certificate = RowSpaceCertificate {
            multipliers: rational(&[1], 1),
            target: rational(&[1, 0], 1),
        };
        assert!(matches!(
            verify_row_space_witness(&matrix, &certificate),
            Err(RowSpaceError::MultiplierDimensionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_target_dimension_mismatch() {
        let matrix = vec![vec![1, 0]];
        let certificate = RowSpaceCertificate {
            multipliers: rational(&[1], 1),
            target: rational(&[1], 1),
        };
        assert!(matches!(
            verify_row_space_witness(&matrix, &certificate),
            Err(RowSpaceError::TargetDimensionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_ragged_matrix_and_empty_matrix() {
        let certificate = RowSpaceCertificate {
            multipliers: rational(&[1, 1], 1),
            target: rational(&[0, 0], 1),
        };
        assert!(matches!(
            verify_row_space_witness(&[vec![1, 0], vec![0]], &certificate),
            Err(RowSpaceError::RaggedMatrix { .. })
        ));
        assert!(matches!(
            verify_row_space_witness(&[], &certificate),
            Err(RowSpaceError::EmptyMatrix)
        ));
    }

    #[test]
    fn rejects_nonpositive_denominators() {
        assert!(matches!(
            RationalVector::new(vec![1], 0),
            Err(RowSpaceError::NonPositiveDenominator { .. })
        ));
        let matrix = vec![vec![1]];
        let certificate = RowSpaceCertificate {
            multipliers: rational(&[1], 1),
            target: RationalVector {
                numerators: vec![1],
                denominator: -1,
            },
        };
        assert!(matches!(
            verify_row_space_witness(&matrix, &certificate),
            Err(RowSpaceError::NonPositiveDenominator { .. })
        ));
    }

    #[test]
    fn arithmetic_overflow_is_rejected_not_wrapped() {
        let matrix = vec![vec![i64::MAX]];
        let certificate = RowSpaceCertificate {
            multipliers: rational(&[i128::MAX], 1),
            target: rational(&[0], 1),
        };
        assert!(matches!(
            verify_row_space_witness(&matrix, &certificate),
            Err(RowSpaceError::ArithmeticOverflow { .. })
        ));
    }
}
