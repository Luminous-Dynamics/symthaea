//! Independent exact-rational verification for linear-programming certificates.
//!
//! The v1 contract uses the standard-form maximization problem
//!
//!     maximize c^T x   subject to A x <= b, x >= 0.
//!
//! The optimizer is untrusted. A feasible primal vector and feasible dual vector
//! with equal exact rational objectives prove optimality by weak duality. A
//! Farkas multiplier vector proves infeasibility when y >= 0, A^T y >= 0 and
//! b^T y < 0. This module checks certificates; it does not implement a simplex
//! optimizer or establish that the caller's LP correctly models a real system.

use crate::semiring::CheckedRational;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

/// A canonical rational LP in standard-form maximization convention:
/// maximize c^T x subject to A x <= b and x >= 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RationalLp {
    a: Vec<Vec<CheckedRational>>,
    b: Vec<CheckedRational>,
    c: Vec<CheckedRational>,
}

impl RationalLp {
    pub fn new(
        a: Vec<Vec<CheckedRational>>,
        b: Vec<CheckedRational>,
        c: Vec<CheckedRational>,
    ) -> Result<Self, LpCertificateError> {
        if a.len() != b.len() {
            return Err(LpCertificateError::RowCountMismatch {
                matrix_rows: a.len(),
                rhs_rows: b.len(),
            });
        }
        for (row, coefficients) in a.iter().enumerate() {
            if coefficients.len() != c.len() {
                return Err(LpCertificateError::MatrixRowWidthMismatch {
                    row,
                    expected: c.len(),
                    actual: coefficients.len(),
                });
            }
        }
        Ok(Self { a, b, c })
    }

    pub fn row_count(&self) -> usize {
        self.b.len()
    }

    pub fn column_count(&self) -> usize {
        self.c.len()
    }

    pub fn matrix(&self) -> &[Vec<CheckedRational>] {
        &self.a
    }

    pub fn rhs(&self) -> &[CheckedRational] {
        &self.b
    }

    pub fn objective(&self) -> &[CheckedRational] {
        &self.c
    }
}

/// Exact primal and dual candidate for a standard-form maximization LP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimalDualCertificate {
    /// Primal candidate x, required to be componentwise nonnegative.
    pub primal: Vec<CheckedRational>,
    /// Dual candidate y, required to be componentwise nonnegative.
    pub dual: Vec<CheckedRational>,
}

/// Compact result returned only after exact primal/dual verification succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedLpOptimum {
    pub objective: CheckedRational,
}

/// Farkas infeasibility witness for A x <= b, x >= 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FarkasCertificate {
    /// Nonnegative row multipliers y with A^T y >= 0 and b^T y < 0.
    pub multipliers: Vec<CheckedRational>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LpCertificateError {
    RowCountMismatch {
        matrix_rows: usize,
        rhs_rows: usize,
    },
    MatrixRowWidthMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    VectorLengthMismatch {
        vector: &'static str,
        expected: usize,
        actual: usize,
    },
    NegativePrimalVariable {
        column: usize,
    },
    NegativeDualMultiplier {
        row: usize,
    },
    NegativeFarkasMultiplier {
        row: usize,
    },
    PrimalConstraintViolation {
        row: usize,
    },
    DualConstraintViolation {
        column: usize,
    },
    PrimalDualObjectiveMismatch,
    FarkasDoesNotContradict,
    ArithmeticOverflow,
    ComparisonOverflow,
}

impl fmt::Display for LpCertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowCountMismatch { matrix_rows, rhs_rows } => write!(
                f,
                "LP matrix has {matrix_rows} rows but RHS has {rhs_rows} entries"
            ),
            Self::MatrixRowWidthMismatch { row, expected, actual } => write!(
                f,
                "LP matrix row {row} has {actual} columns; expected {expected}"
            ),
            Self::VectorLengthMismatch { vector, expected, actual } => write!(
                f,
                "{vector} certificate has length {actual}; expected {expected}"
            ),
            Self::NegativePrimalVariable { column } => {
                write!(f, "primal variable {column} is negative")
            }
            Self::NegativeDualMultiplier { row } => {
                write!(f, "dual multiplier {row} is negative")
            }
            Self::NegativeFarkasMultiplier { row } => {
                write!(f, "Farkas multiplier {row} is negative")
            }
            Self::PrimalConstraintViolation { row } => {
                write!(f, "primal candidate violates LP row {row}")
            }
            Self::DualConstraintViolation { column } => {
                write!(f, "dual candidate violates column constraint {column}")
            }
            Self::PrimalDualObjectiveMismatch => {
                write!(f, "primal and dual rational objectives do not match")
            }
            Self::FarkasDoesNotContradict => {
                write!(f, "Farkas multipliers do not certify infeasibility")
            }
            Self::ArithmeticOverflow => {
                write!(f, "checked rational arithmetic overflow during certificate replay")
            }
            Self::ComparisonOverflow => {
                write!(f, "checked rational comparison overflow during certificate replay")
            }
        }
    }
}

impl Error for LpCertificateError {}

/// Verify exact primal feasibility, dual feasibility and equality of objectives.
///
/// For the declared standard-form maximization problem, the dual is
/// minimize b^T y subject to A^T y >= c and y >= 0. Exact primal and dual
/// feasibility with equal objective values proves global optimality.
pub fn verify_primal_dual_optimality(
    lp: &RationalLp,
    certificate: &PrimalDualCertificate,
) -> Result<VerifiedLpOptimum, LpCertificateError> {
    let rows = lp.row_count();
    let columns = lp.column_count();
    if certificate.primal.len() != columns {
        return Err(LpCertificateError::VectorLengthMismatch {
            vector: "primal",
            expected: columns,
            actual: certificate.primal.len(),
        });
    }
    if certificate.dual.len() != rows {
        return Err(LpCertificateError::VectorLengthMismatch {
            vector: "dual",
            expected: rows,
            actual: certificate.dual.len(),
        });
    }

    for (column, value) in certificate.primal.iter().enumerate() {
        if value.is_negative() {
            return Err(LpCertificateError::NegativePrimalVariable { column });
        }
    }
    for (row, value) in certificate.dual.iter().enumerate() {
        if value.is_negative() {
            return Err(LpCertificateError::NegativeDualMultiplier { row });
        }
    }

    for row in 0..rows {
        let lhs = dot(&lp.a[row], &certificate.primal)?;
        if compare_rational(lhs, lp.b[row])? == Ordering::Greater {
            return Err(LpCertificateError::PrimalConstraintViolation { row });
        }
    }

    for column in 0..columns {
        let mut lhs = CheckedRational::zero();
        for row in 0..rows {
            let term = lp.a[row][column]
                .checked_mul(certificate.dual[row])
                .map_err(|_| LpCertificateError::ArithmeticOverflow)?;
            lhs = lhs.checked_add(term)
                .map_err(|_| LpCertificateError::ArithmeticOverflow)?;
        }
        if compare_rational(lhs, lp.c[column])? == Ordering::Less {
            return Err(LpCertificateError::DualConstraintViolation { column });
        }
    }

    let primal_objective = dot(&lp.c, &certificate.primal)?;
    let dual_objective = dot(&lp.b, &certificate.dual)?;
    if primal_objective != dual_objective {
        return Err(LpCertificateError::PrimalDualObjectiveMismatch);
    }

    Ok(VerifiedLpOptimum {
        objective: primal_objective,
    })
}

/// Verify a Farkas witness for infeasibility of A x <= b, x >= 0.
///
/// A valid y satisfies y >= 0, A^T y >= 0, and b^T y < 0. Any feasible
/// x would imply 0 <= (A^T y)^T x <= b^T y < 0, a contradiction.
pub fn verify_farkas_infeasibility(
    lp: &RationalLp,
    certificate: &FarkasCertificate,
) -> Result<(), LpCertificateError> {
    let rows = lp.row_count();
    let columns = lp.column_count();
    if certificate.multipliers.len() != rows {
        return Err(LpCertificateError::VectorLengthMismatch {
            vector: "Farkas",
            expected: rows,
            actual: certificate.multipliers.len(),
        });
    }

    for (row, multiplier) in certificate.multipliers.iter().enumerate() {
        if multiplier.is_negative() {
            return Err(LpCertificateError::NegativeFarkasMultiplier { row });
        }
    }

    for column in 0..columns {
        let mut lhs = CheckedRational::zero();
        for row in 0..rows {
            let term = lp.a[row][column]
                .checked_mul(certificate.multipliers[row])
                .map_err(|_| LpCertificateError::ArithmeticOverflow)?;
            lhs = lhs.checked_add(term)
                .map_err(|_| LpCertificateError::ArithmeticOverflow)?;
        }
        if compare_rational(lhs, CheckedRational::zero())? == Ordering::Less {
            return Err(LpCertificateError::FarkasDoesNotContradict);
        }
    }

    let contradiction = dot(&lp.b, &certificate.multipliers)?;
    if compare_rational(contradiction, CheckedRational::zero())? != Ordering::Less {
        return Err(LpCertificateError::FarkasDoesNotContradict);
    }
    Ok(())
}

fn dot(
    left: &[CheckedRational],
    right: &[CheckedRational],
) -> Result<CheckedRational, LpCertificateError> {
    if left.len() != right.len() {
        return Err(LpCertificateError::VectorLengthMismatch {
            vector: "dot-product",
            expected: left.len(),
            actual: right.len(),
        });
    }
    let mut sum = CheckedRational::zero();
    for (&a, &b) in left.iter().zip(right) {
        let product = a.checked_mul(b)
            .map_err(|_| LpCertificateError::ArithmeticOverflow)?;
        sum = sum.checked_add(product)
            .map_err(|_| LpCertificateError::ArithmeticOverflow)?;
    }
    Ok(sum)
}

fn compare_rational(
    left: CheckedRational,
    right: CheckedRational,
) -> Result<Ordering, LpCertificateError> {
    // Checked cross-products preserve signed rational ordering; overflow is a
    // verifier failure, never a floating-point fallback.
    let lhs = left.numerator()
        .checked_mul(right.denominator())
        .ok_or(LpCertificateError::ComparisonOverflow)?;
    let rhs = right.numerator()
        .checked_mul(left.denominator())
        .ok_or(LpCertificateError::ComparisonOverflow)?;
    Ok(lhs.cmp(&rhs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(numerator: i128, denominator: i128) -> CheckedRational {
        CheckedRational::new(numerator, denominator).unwrap()
    }

    #[test]
    fn verifies_exact_primal_dual_optimality() {
        // max 3x + 2y
        // s.t. x + y <= 4, x <= 2, y <= 3, x,y >= 0.
        let lp = RationalLp::new(
            vec![
                vec![q(1, 1), q(1, 1)],
                vec![q(1, 1), q(0, 1)],
                vec![q(0, 1), q(1, 1)],
            ],
            vec![q(4, 1), q(2, 1), q(3, 1)],
            vec![q(3, 1), q(2, 1)],
        ).unwrap();
        let certificate = PrimalDualCertificate {
            primal: vec![q(2, 1), q(2, 1)],
            dual: vec![q(2, 1), q(1, 1), q(0, 1)],
        };
        let verified = verify_primal_dual_optimality(&lp, &certificate).unwrap();
        assert_eq!(verified.objective, q(10, 1));
    }

    #[test]
    fn rejects_mutated_primal_or_dual_certificates() {
        let lp = RationalLp::new(
            vec![
                vec![q(1, 1), q(1, 1)],
                vec![q(1, 1), q(0, 1)],
                vec![q(0, 1), q(1, 1)],
            ],
            vec![q(4, 1), q(2, 1), q(3, 1)],
            vec![q(3, 1), q(2, 1)],
        ).unwrap();

        let bad_primal = PrimalDualCertificate {
            primal: vec![q(3, 1), q(1, 1)],
            dual: vec![q(2, 1), q(1, 1), q(0, 1)],
        };
        assert_eq!(
            verify_primal_dual_optimality(&lp, &bad_primal),
            Err(LpCertificateError::PrimalConstraintViolation { row: 1 })
        );

        let bad_dual = PrimalDualCertificate {
            primal: vec![q(2, 1), q(2, 1)],
            dual: vec![q(1, 1), q(0, 1), q(0, 1)],
        };
        assert!(matches!(
            verify_primal_dual_optimality(&lp, &bad_dual),
            Err(LpCertificateError::DualConstraintViolation { .. })
        ));
    }

    #[test]
    fn verifies_rational_farkas_infeasibility_certificate() {
        // x <= 0 and -x <= -1 cannot both hold for x >= 0.
        let lp = RationalLp::new(
            vec![vec![q(1, 1)], vec![q(-1, 1)]],
            vec![q(0, 1), q(-1, 1)],
            vec![q(0, 1)],
        ).unwrap();
        let certificate = FarkasCertificate {
            multipliers: vec![q(1, 1), q(1, 1)],
        };
        assert_eq!(verify_farkas_infeasibility(&lp, &certificate), Ok(()));
    }

    #[test]
    fn rejects_farkas_witness_that_does_not_contradict() {
        let lp = RationalLp::new(
            vec![vec![q(1, 1)], vec![q(-1, 1)]],
            vec![q(0, 1), q(-1, 1)],
            vec![q(0, 1)],
        ).unwrap();
        let certificate = FarkasCertificate {
            multipliers: vec![q(1, 1), q(0, 1)],
        };
        assert_eq!(
            verify_farkas_infeasibility(&lp, &certificate),
            Err(LpCertificateError::FarkasDoesNotContradict)
        );
    }
}
