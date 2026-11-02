//! Exact certificates for rational polynomials in commuting permutations.
//!
//! The verifier reconstructs M0 = sum_t c_t * P_1^(e_t1) ... P_s^(e_ts)
//! from explicit permutation maps and rational coefficients. It rejects
//! non-permutations and non-commuting generators. The output can then be fed
//! to matrix::verify_identity_minus_rank_factorization.
//!
//! This module verifies a supplied representation; it does not search for the
//! permutation generators, polynomial, sparsity pattern, or low-rank factors.

use std::error::Error;
use std::fmt;

use crate::matrix::{MatrixCertificateError, RationalMatrix, verify_rational_matrix_equality};

/// One monomial term c * P_1^e1 * ... * P_s^es.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutationMonomial {
    pub exponents: Vec<usize>,
    pub coefficient_numerator: i128,
    pub coefficient_denominator: i128,
}

/// A polynomial in a declared family of commuting permutations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutationPolynomialCertificate {
    /// Each generator maps a source point to its destination point.
    pub generators: Vec<Vec<usize>>,
    pub terms: Vec<PermutationMonomial>,
}

/// Fail-closed errors in the permutation-polynomial certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PermutationPolynomialError {
    InvalidPermutationLength {
        generator: usize,
        expected: usize,
        actual: usize,
    },
    PermutationDestinationOutOfRange {
        generator: usize,
        point: usize,
        destination: usize,
        dimension: usize,
    },
    DuplicatePermutationDestination {
        generator: usize,
        destination: usize,
    },
    NonCommutingGenerators {
        left_generator: usize,
        right_generator: usize,
        point: usize,
    },
    ExponentDimensionMismatch {
        term: usize,
        expected: usize,
        actual: usize,
    },
    NonPositiveCoefficientDenominator {
        term: usize,
        denominator: i128,
    },
    DenominatorLcmOverflow,
    CoefficientScaleOverflow {
        term: usize,
    },
    EntryAccumulationOverflow {
        term: usize,
        source: usize,
        destination: usize,
    },
    DimensionOverflow,
    AllocationFailed {
        entries: usize,
    },
    Matrix(MatrixCertificateError),
}

impl fmt::Display for PermutationPolynomialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPermutationLength { generator, expected, actual } => write!(
                f,
                "permutation generator {generator} has length {actual}; expected {expected}"
            ),
            Self::PermutationDestinationOutOfRange { generator, point, destination, dimension } => write!(
                f,
                "permutation generator {generator} maps point {point} to {destination}, outside 0..{dimension}"
            ),
            Self::DuplicatePermutationDestination { generator, destination } => write!(
                f,
                "permutation generator {generator} maps more than one point to {destination}"
            ),
            Self::NonCommutingGenerators { left_generator, right_generator, point } => write!(
                f,
                "permutation generators {left_generator} and {right_generator} do not commute at point {point}"
            ),
            Self::ExponentDimensionMismatch { term, expected, actual } => write!(
                f,
                "polynomial term {term} has {actual} exponents; expected {expected}"
            ),
            Self::NonPositiveCoefficientDenominator { term, denominator } => write!(
                f,
                "polynomial term {term} has non-positive coefficient denominator {denominator}"
            ),
            Self::DenominatorLcmOverflow => write!(f, "common coefficient denominator overflows i128"),
            Self::CoefficientScaleOverflow { term } => {
                write!(f, "coefficient scaling overflows in polynomial term {term}")
            }
            Self::EntryAccumulationOverflow { term, source, destination } => write!(
                f,
                "matrix entry accumulation overflows in term {term}, at ({source},{destination})"
            ),
            Self::DimensionOverflow => write!(f, "polynomial matrix dimension overflows usize"),
            Self::AllocationFailed { entries } => {
                write!(f, "unable to reserve {entries} polynomial matrix entries")
            }
            Self::Matrix(error) => write!(f, "matrix verification error: {error}"),
        }
    }
}

impl Error for PermutationPolynomialError {}

impl From<MatrixCertificateError> for PermutationPolynomialError {
    fn from(value: MatrixCertificateError) -> Self {
        Self::Matrix(value)
    }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

fn checked_lcm(a: i128, b: i128) -> Result<i128, PermutationPolynomialError> {
    let divisor = gcd(a, b);
    (a / divisor)
        .checked_mul(b)
        .ok_or(PermutationPolynomialError::DenominatorLcmOverflow)
}

fn validate_permutations(
    dimension: usize,
    generators: &[Vec<usize>],
) -> Result<(), PermutationPolynomialError> {
    for (generator_index, generator) in generators.iter().enumerate() {
        if generator.len() != dimension {
            return Err(PermutationPolynomialError::InvalidPermutationLength {
                generator: generator_index,
                expected: dimension,
                actual: generator.len(),
            });
        }

        let mut seen = Vec::new();
        seen.try_reserve_exact(dimension)
            .map_err(|_| PermutationPolynomialError::AllocationFailed {
                entries: dimension,
            })?;
        seen.resize(dimension, false);

        for (point, &destination) in generator.iter().enumerate() {
            if destination >= dimension {
                return Err(PermutationPolynomialError::PermutationDestinationOutOfRange {
                    generator: generator_index,
                    point,
                    destination,
                    dimension,
                });
            }
            if seen[destination] {
                return Err(PermutationPolynomialError::DuplicatePermutationDestination {
                    generator: generator_index,
                    destination,
                });
            }
            seen[destination] = true;
        }
    }

    for left in 0..generators.len() {
        for right in (left + 1)..generators.len() {
            for point in 0..dimension {
                let left_after_right = generators[left][generators[right][point]];
                let right_after_left = generators[right][generators[left][point]];
                if left_after_right != right_after_left {
                    return Err(PermutationPolynomialError::NonCommutingGenerators {
                        left_generator: left,
                        right_generator: right,
                        point,
                    });
                }
            }
        }
    }

    Ok(())
}

fn monomial_destination(
    source: usize,
    exponents: &[usize],
    generators: &[Vec<usize>],
) -> usize {
    let mut current = source;
    for (generator, &exponent) in generators.iter().zip(exponents.iter()) {
        // Reduce the exponent modulo this point's cycle length. Since each
        // generator has already been validated as a permutation, the orbit
        // returns to the current point after at most dimension steps. This
        // prevents a hostile certificate from making evaluation proportional
        // to an arbitrarily large usize exponent.
        let mut cursor = generator[current];
        let mut cycle_length = 1usize;
        while cursor != current {
            cursor = generator[cursor];
            cycle_length += 1;
        }
        let steps = exponent % cycle_length;
        for _ in 0..steps {
            current = generator[current];
        }
    }
    current
}

/// Evaluate the supplied polynomial exactly as a rational matrix.
///
/// Each permutation generator is checked to be bijective and pairwise commuting.
/// All coefficients are lifted to a positive common denominator with checked
/// lcm/scaling/accumulation. Duplicate monomials are allowed and add exactly.
pub fn evaluate_permutation_polynomial(
    dimension: usize,
    certificate: &PermutationPolynomialCertificate,
) -> Result<RationalMatrix, PermutationPolynomialError> {
    validate_permutations(dimension, &certificate.generators)?;

    let mut common_denominator = 1i128;
    for (term_index, term) in certificate.terms.iter().enumerate() {
        if term.exponents.len() != certificate.generators.len() {
            return Err(PermutationPolynomialError::ExponentDimensionMismatch {
                term: term_index,
                expected: certificate.generators.len(),
                actual: term.exponents.len(),
            });
        }
        if term.coefficient_denominator <= 0 {
            return Err(PermutationPolynomialError::NonPositiveCoefficientDenominator {
                term: term_index,
                denominator: term.coefficient_denominator,
            });
        }
        common_denominator = checked_lcm(common_denominator, term.coefficient_denominator)?;
    }

    let entries = dimension
        .checked_mul(dimension)
        .ok_or(PermutationPolynomialError::DimensionOverflow)?;
    let mut numerators = Vec::new();
    numerators
        .try_reserve_exact(entries)
        .map_err(|_| PermutationPolynomialError::AllocationFailed { entries })?;
    numerators.resize(entries, 0i128);

    for (term_index, term) in certificate.terms.iter().enumerate() {
        let scale = common_denominator / term.coefficient_denominator;
        let coefficient = term
            .coefficient_numerator
            .checked_mul(scale)
            .ok_or(PermutationPolynomialError::CoefficientScaleOverflow {
                term: term_index,
            })?;

        for source in 0..dimension {
            let destination = monomial_destination(
                source,
                &term.exponents,
                &certificate.generators,
            );
            let index = source * dimension + destination;
            numerators[index] = numerators[index]
                .checked_add(coefficient)
                .ok_or(PermutationPolynomialError::EntryAccumulationOverflow {
                    term: term_index,
                    source,
                    destination,
                })?;
        }
    }

    Ok(RationalMatrix::new(
        dimension,
        dimension,
        numerators,
        common_denominator,
    )?)
}

/// Verify that a claimed M0 is exactly the polynomial in the supplied
/// commuting permutation generators.
pub fn verify_permutation_polynomial(
    claimed_m0: &RationalMatrix,
    certificate: &PermutationPolynomialCertificate,
) -> Result<bool, PermutationPolynomialError> {
    if claimed_m0.rows() != claimed_m0.columns() {
        return Ok(false);
    }
    let evaluated = evaluate_permutation_polynomial(claimed_m0.rows(), certificate)?;
    Ok(verify_rational_matrix_equality(&evaluated, claimed_m0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize, numerators: &[i128], denominator: i128) -> RationalMatrix {
        RationalMatrix::new(rows, columns, numerators.to_vec(), denominator).unwrap()
    }

    #[test]
    fn accepts_polynomial_in_identity_and_three_cycle() {
        // P=(0 1 2), M0 = (1/2) I + (3/2) P.
        let claimed_m0 = matrix(3, 3, &[1, 3, 0, 0, 1, 3, 3, 0, 1], 2);
        let certificate = PermutationPolynomialCertificate {
            generators: vec![vec![1, 2, 0]],
            terms: vec![
                PermutationMonomial {
                    exponents: vec![0],
                    coefficient_numerator: 1,
                    coefficient_denominator: 2,
                },
                PermutationMonomial {
                    exponents: vec![1],
                    coefficient_numerator: 3,
                    coefficient_denominator: 2,
                },
            ],
        };
        assert!(verify_permutation_polynomial(&claimed_m0, &certificate).unwrap());
    }

    #[test]
    fn rejects_mutated_claimed_matrix() {
        let claimed_m0 = matrix(3, 3, &[1, 3, 0, 0, 1, 3, 3, 0, 1], 2);
        let certificate = PermutationPolynomialCertificate {
            generators: vec![vec![1, 2, 0]],
            terms: vec![
                PermutationMonomial {
                    exponents: vec![0],
                    coefficient_numerator: 1,
                    coefficient_denominator: 2,
                },
                PermutationMonomial {
                    exponents: vec![1],
                    coefficient_numerator: 3,
                    coefficient_denominator: 2,
                },
            ],
        };
        let mutated = matrix(3, 3, &[2, 3, 0, 0, 1, 3, 3, 0, 1], 2);
        assert!(!verify_permutation_polynomial(&mutated, &certificate).unwrap());
        assert!(verify_permutation_polynomial(&claimed_m0, &certificate).unwrap());
    }

    #[test]
    fn rejects_non_bijective_generator() {
        let claimed_m0 = matrix(3, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1], 1);
        let certificate = PermutationPolynomialCertificate {
            generators: vec![vec![1, 1, 0]],
            terms: vec![PermutationMonomial {
                exponents: vec![0],
                coefficient_numerator: 1,
                coefficient_denominator: 1,
            }],
        };
        assert!(matches!(
            verify_permutation_polynomial(&claimed_m0, &certificate),
            Err(PermutationPolynomialError::DuplicatePermutationDestination { .. })
        ));
    }

    #[test]
    fn rejects_noncommuting_generators() {
        let claimed_m0 = matrix(3, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1], 1);
        let certificate = PermutationPolynomialCertificate {
            generators: vec![vec![1, 0, 2], vec![0, 2, 1]],
            terms: vec![PermutationMonomial {
                exponents: vec![0, 0],
                coefficient_numerator: 1,
                coefficient_denominator: 1,
            }],
        };
        assert!(matches!(
            verify_permutation_polynomial(&claimed_m0, &certificate),
            Err(PermutationPolynomialError::NonCommutingGenerators { .. })
        ));
    }

    #[test]
    fn rejects_exponent_dimension_mismatch() {
        let claimed_m0 = matrix(2, 2, &[1, 0, 0, 1], 1);
        let certificate = PermutationPolynomialCertificate {
            generators: vec![vec![1, 0]],
            terms: vec![PermutationMonomial {
                exponents: vec![],
                coefficient_numerator: 1,
                coefficient_denominator: 1,
            }],
        };
        assert!(matches!(
            verify_permutation_polynomial(&claimed_m0, &certificate),
            Err(PermutationPolynomialError::ExponentDimensionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_nonpositive_coefficients_and_denominator_lcm_overflow() {
        let claimed_m0 = matrix(1, 1, &[1], 1);
        let zero_denominator = PermutationPolynomialCertificate {
            generators: vec![],
            terms: vec![PermutationMonomial {
                exponents: vec![],
                coefficient_numerator: 1,
                coefficient_denominator: 0,
            }],
        };
        assert!(matches!(
            verify_permutation_polynomial(&claimed_m0, &zero_denominator),
            Err(PermutationPolynomialError::NonPositiveCoefficientDenominator { .. })
        ));

        let overflow = PermutationPolynomialCertificate {
            generators: vec![],
            terms: vec![
                PermutationMonomial {
                    exponents: vec![],
                    coefficient_numerator: 1,
                    coefficient_denominator: i128::MAX,
                },
                PermutationMonomial {
                    exponents: vec![],
                    coefficient_numerator: 1,
                    coefficient_denominator: i128::MAX - 1,
                },
            ],
        };
        assert!(matches!(
            evaluate_permutation_polynomial(1, &overflow),
            Err(PermutationPolynomialError::DenominatorLcmOverflow)
        ));
    }

    #[test]
    fn huge_exponents_reduce_by_permutation_cycle() {
        // On a 3-cycle, (usize::MAX - 1) mod 3 = 2 on supported 32/64-bit targets.
        let claimed_p_squared = matrix(3, 3, &[0, 0, 1, 1, 0, 0, 0, 1, 0], 1);
        let certificate = PermutationPolynomialCertificate {
            generators: vec![vec![1, 2, 0]],
            terms: vec![PermutationMonomial {
                exponents: vec![usize::MAX - 1],
                coefficient_numerator: 1,
                coefficient_denominator: 1,
            }],
        };
        assert!(verify_permutation_polynomial(&claimed_p_squared, &certificate).unwrap());
    }

    #[test]
    fn accepts_empty_polynomial_as_zero_matrix() {
        let zero = matrix(2, 2, &[0, 0, 0, 0], 1);
        let certificate = PermutationPolynomialCertificate {
            generators: vec![],
            terms: vec![],
        };
        assert!(verify_permutation_polynomial(&zero, &certificate).unwrap());
    }
}
