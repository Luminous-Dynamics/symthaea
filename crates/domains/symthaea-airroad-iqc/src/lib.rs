#![forbid(unsafe_code)]

//! Minimal, dependency-free complex-Hermitian primitives for AIRROAD's IQC
//! backend boundary.
//!
//! The crate separates Hermitian validation, complex-to-real conversion, and
//! Clarabel PSD-triangle packing. It does not yet construct IQC multipliers
//! or invoke an SDP solver.

use std::fmt;

/// A scalar in the complex field, represented by finite IEEE-754 components.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}

impl Complex64 {
    /// Construct a complex scalar.
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Return the complex conjugate.
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Multiply two complex scalars.
    #[must_use]
    pub fn multiply(self, rhs: Self) -> Self {
        Self {
            re: self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            im: self.re.mul_add(rhs.im, self.im * rhs.re),
        }
    }

    /// Add two complex scalars.
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

/// Errors produced by matrix validation and conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// Matrices of order zero are not admitted.
    ZeroDimension,
    /// The requested dense matrix size overflows usize.
    DimensionOverflow,
    /// The number of supplied entries is not order squared.
    LengthMismatch {
        /// Required number of entries.
        expected: usize,
        /// Actual number of entries.
        actual: usize,
    },
    /// A matrix entry contains NaN or infinity.
    NonFiniteEntry {
        /// Row-major entry index.
        index: usize,
    },
    /// A Hermitian diagonal entry has a nonzero imaginary component.
    NonRealHermitianDiagonal {
        /// Diagonal index.
        index: usize,
    },
    /// An off-diagonal pair is not exactly conjugate symmetric.
    NonHermitianPair {
        /// Upper-triangular row.
        row: usize,
        /// Upper-triangular column.
        col: usize,
    },
    /// A real matrix is not exactly symmetric.
    NonSymmetricPair {
        /// Upper-triangular row.
        row: usize,
        /// Upper-triangular column.
        col: usize,
    },
    /// Scaling an entry for Clarabel's PSD triangle produced a non-finite value.
    NonFinitePackedEntry {
        /// Index in the packed column-major vector.
        index: usize,
    },
    /// A permutation does not contain exactly order indices.
    InvalidPermutationLength {
        /// Required permutation length.
        expected: usize,
        /// Actual permutation length.
        actual: usize,
    },
    /// A permutation index is outside the matrix order.
    PermutationIndexOutOfRange {
        /// Invalid index.
        index: usize,
        /// Matrix order.
        order: usize,
    },
    /// A permutation repeats an index.
    DuplicatePermutationIndex {
        /// Repeated index.
        index: usize,
    },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "matrix order must be nonzero"),
            Self::DimensionOverflow => write!(formatter, "matrix dimension overflows usize"),
            Self::LengthMismatch { expected, actual } => {
                write!(formatter, "expected {expected} entries, received {actual}")
            }
            Self::NonFiniteEntry { index } => {
                write!(formatter, "matrix entry {index} is not finite")
            }
            Self::NonRealHermitianDiagonal { index } => {
                write!(formatter, "Hermitian diagonal entry {index} is not real")
            }
            Self::NonHermitianPair { row, col } => {
                write!(formatter, "entry ({row},{col}) is not the conjugate of its transpose")
            }
            Self::NonSymmetricPair { row, col } => {
                write!(formatter, "real entry ({row},{col}) is not symmetric")
            }
            Self::NonFinitePackedEntry { index } => {
                write!(formatter, "packed PSD entry {index} is not finite")
            }
            Self::InvalidPermutationLength { expected, actual } => {
                write!(formatter, "expected permutation length {expected}, received {actual}")
            }
            Self::PermutationIndexOutOfRange { index, order } => {
                write!(formatter, "permutation index {index} is outside order {order}")
            }
            Self::DuplicatePermutationIndex { index } => {
                write!(formatter, "permutation index {index} occurs more than once")
            }
        }
    }
}

impl std::error::Error for MatrixError {}

/// A dense row-major complex Hermitian matrix.
///
/// Construction is exact with respect to the supplied floating-point values:
/// diagonal imaginary components must be zero and each off-diagonal pair must
/// be exactly conjugate. No tolerance-based projection or silent symmetrizing
/// is performed. An approximate-import API must expose its residual bound and
/// retain the original input separately.
#[derive(Clone, Debug, PartialEq)]
pub struct HermitianMatrix {
    order: usize,
    entries: Vec<Complex64>,
}

impl HermitianMatrix {
    /// Validate and construct a dense row-major Hermitian matrix.
    pub fn try_new(order: usize, entries: Vec<Complex64>) -> Result<Self, MatrixError> {
        let expected = checked_square(order)?;
        if entries.len() != expected {
            return Err(MatrixError::LengthMismatch {
                expected,
                actual: entries.len(),
            });
        }

        for (index, value) in entries.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(MatrixError::NonFiniteEntry { index });
            }
        }

        for row in 0..order {
            if entries[row * order + row].im != 0.0 {
                return Err(MatrixError::NonRealHermitianDiagonal { index: row });
            }
            for col in (row + 1)..order {
                let upper = entries[row * order + col];
                let lower = entries[col * order + row];
                if upper.re != lower.re || upper.im != -lower.im {
                    return Err(MatrixError::NonHermitianPair { row, col });
                }
            }
        }

        Ok(Self { order, entries })
    }

    /// Return the matrix order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Return a matrix entry, or None for an out-of-range index.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<Complex64> {
        if row >= self.order || col >= self.order {
            return None;
        }
        Some(self.entries[row * self.order + col])
    }

    /// Return the dense row-major entries.
    #[must_use]
    pub fn as_slice(&self) -> &[Complex64] {
        &self.entries
    }

    /// Reorder rows and columns together.
    ///
    /// new_to_old[i] identifies the old row/column placed at new index i.
    /// The input must be a complete permutation; duplicated or omitted ports
    /// are rejected instead of being silently accepted.
    pub fn permute(&self, new_to_old: &[usize]) -> Result<Self, MatrixError> {
        validate_permutation(self.order, new_to_old)?;
        let mut entries = Vec::with_capacity(self.entries.len());
        for &old_row in new_to_old {
            for &old_col in new_to_old {
                entries.push(self.entries[old_row * self.order + old_col]);
            }
        }
        Self::try_new(self.order, entries)
    }

    /// Map H = A + iB to the real symmetric matrix [A -B; B A],
    /// where A is real symmetric and B is real skew-symmetric.
    ///
    /// The mapping preserves positive semidefiniteness and duplicates each
    /// eigenvalue of H in the real representation.
    pub fn realify(&self) -> Result<RealSymmetricMatrix, MatrixError> {
        let real_order = self
            .order
            .checked_mul(2)
            .ok_or(MatrixError::DimensionOverflow)?;
        let real_len = checked_square(real_order)?;
        let mut entries = vec![0.0; real_len];

        for row in 0..real_order {
            for col in 0..real_order {
                let value = if row < self.order && col < self.order {
                    self.entries[row * self.order + col].re
                } else if row < self.order && col >= self.order {
                    -self.entries[row * self.order + (col - self.order)].im
                } else if row >= self.order && col < self.order {
                    self.entries[(row - self.order) * self.order + col].im
                } else {
                    self.entries[(row - self.order) * self.order + (col - self.order)].re
                };
                entries[row * real_order + col] = value;
            }
        }

        RealSymmetricMatrix::try_new(real_order, entries)
    }
}

/// A dense row-major real symmetric matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct RealSymmetricMatrix {
    order: usize,
    entries: Vec<f64>,
}

impl RealSymmetricMatrix {
    /// Validate and construct a dense row-major real symmetric matrix.
    pub fn try_new(order: usize, entries: Vec<f64>) -> Result<Self, MatrixError> {
        let expected = checked_square(order)?;
        if entries.len() != expected {
            return Err(MatrixError::LengthMismatch {
                expected,
                actual: entries.len(),
            });
        }

        for (index, value) in entries.iter().enumerate() {
            if !value.is_finite() {
                return Err(MatrixError::NonFiniteEntry { index });
            }
        }

        for row in 0..order {
            for col in (row + 1)..order {
                if entries[row * order + col] != entries[col * order + row] {
                    return Err(MatrixError::NonSymmetricPair { row, col });
                }
            }
        }

        Ok(Self { order, entries })
    }

    /// Return the matrix order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Return an entry, or None for an out-of-range index.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.order || col >= self.order {
            return None;
        }
        Some(self.entries[row * self.order + col])
    }

    /// Return the dense row-major entries.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.entries
    }

    /// Pack the upper triangle in column-major order using Clarabel's PSD
    /// cone convention: off-diagonal entries are multiplied by sqrt(2).
    ///
    /// The Euclidean dot product of two packed symmetric matrices equals
    /// their Frobenius inner product. This is a packing conversion only: it
    /// does not prove positive semidefiniteness or invoke an SDP solver.
    pub fn to_clarabel_psd_triangle(&self) -> Result<Vec<f64>, MatrixError> {
        let capacity = checked_triangle(self.order)?;
        let mut packed = Vec::with_capacity(capacity);
        let sqrt_two = std::f64::consts::SQRT_2;

        for col in 0..self.order {
            for row in 0..=col {
                let mut value = self.entries[row * self.order + col];
                if row != col {
                    value *= sqrt_two;
                }
                if !value.is_finite() {
                    return Err(MatrixError::NonFinitePackedEntry {
                        index: packed.len(),
                    });
                }
                packed.push(value);
            }
        }
        Ok(packed)
    }

    /// Reconstruct a real symmetric matrix from Clarabel's packed PSD triangle.
    ///
    /// The input must follow the same column-major upper-triangle ordering as
    /// to_clarabel_psd_triangle, including sqrt(2) scaling of off-diagonal
    /// entries. Non-finite values, zero order, and incorrect vector lengths
    /// are rejected.
    pub fn from_clarabel_psd_triangle(
        order: usize,
        packed: &[f64],
    ) -> Result<Self, MatrixError> {
        let expected = checked_triangle(order)?;
        if packed.len() != expected {
            return Err(MatrixError::LengthMismatch {
                expected,
                actual: packed.len(),
            });
        }

        let mut entries = vec![0.0; checked_square(order)?];
        let sqrt_two = std::f64::consts::SQRT_2;
        let mut index = 0;

        for col in 0..order {
            for row in 0..=col {
                let encoded = packed[index];
                if !encoded.is_finite() {
                    return Err(MatrixError::NonFinitePackedEntry { index });
                }
                let value = if row == col {
                    encoded
                } else {
                    encoded / sqrt_two
                };
                entries[row * order + col] = value;
                entries[col * order + row] = value;
                index += 1;
            }
        }

        Self::try_new(order, entries)
    }
}

fn checked_triangle(order: usize) -> Result<usize, MatrixError> {
    if order == 0 {
        return Err(MatrixError::ZeroDimension);
    }
    let next_order = order.checked_add(1).ok_or(MatrixError::DimensionOverflow)?;
    let (left, right) = if order & 1 == 0 {
        (order / 2, next_order)
    } else {
        (order, next_order / 2)
    };
    left.checked_mul(right).ok_or(MatrixError::DimensionOverflow)
}

fn checked_square(order: usize) -> Result<usize, MatrixError> {
    if order == 0 {
        return Err(MatrixError::ZeroDimension);
    }
    order.checked_mul(order).ok_or(MatrixError::DimensionOverflow)
}

fn validate_permutation(order: usize, permutation: &[usize]) -> Result<(), MatrixError> {
    if permutation.len() != order {
        return Err(MatrixError::InvalidPermutationLength {
            expected: order,
            actual: permutation.len(),
        });
    }

    let mut seen = vec![false; order];
    for &index in permutation {
        if index >= order {
            return Err(MatrixError::PermutationIndexOutOfRange { index, order });
        }
        if seen[index] {
            return Err(MatrixError::DuplicatePermutationIndex { index });
        }
        seen[index] = true;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Complex64 as C, HermitianMatrix, MatrixError, RealSymmetricMatrix};

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    fn two_by_two_hermitian() -> HermitianMatrix {
        HermitianMatrix::try_new(
            2,
            vec![c(2.0, 0.0), c(1.0, 2.0), c(1.0, -2.0), c(5.0, 0.0)],
        )
        .expect("fixture is Hermitian")
    }

    fn real_quadratic(matrix: &RealSymmetricMatrix, vector: &[f64]) -> f64 {
        let mut total = 0.0;
        for row in 0..matrix.order() {
            for col in 0..matrix.order() {
                total += vector[row] * matrix.get(row, col).expect("in range") * vector[col];
            }
        }
        total
    }

    fn complex_quadratic(matrix: &HermitianMatrix, vector: &[C]) -> C {
        let mut total = C::new(0.0, 0.0);
        for row in 0..matrix.order() {
            for col in 0..matrix.order() {
                let h = matrix.get(row, col).expect("in range");
                total = total.add(vector[row].conjugate().multiply(h).multiply(vector[col]));
            }
        }
        total
    }

    #[test]
    fn realification_matches_the_block_definition() {
        let matrix = two_by_two_hermitian();
        let real = matrix.realify().expect("valid realification");
        assert_eq!(real.order(), 4);
        assert_eq!(
            real.as_slice(),
            &[
                2.0, 1.0, 0.0, -2.0,
                1.0, 5.0, 2.0, 0.0,
                0.0, 2.0, 2.0, 1.0,
                -2.0, 0.0, 1.0, 5.0,
            ]
        );
    }

    #[test]
    fn realification_is_symmetric_for_complex_hermitian_input() {
        let real = two_by_two_hermitian().realify().expect("valid realification");
        for row in 0..real.order() {
            for col in 0..real.order() {
                assert_eq!(
                    real.get(row, col),
                    real.get(col, row),
                    "entry ({row},{col})"
                );
            }
        }
    }

    #[test]
    fn realification_preserves_the_quadratic_form() {
        let matrix = two_by_two_hermitian();
        let real = matrix.realify().expect("valid realification");
        let z = [c(1.0, 2.0), c(-1.0, 3.0)];
        let x = [1.0, -1.0, 2.0, 3.0];
        let lhs = complex_quadratic(&matrix, &z);
        let rhs = real_quadratic(&real, &x);
        assert_eq!(lhs.im, 0.0);
        assert!((lhs.re - rhs).abs() < 1.0e-12);
    }

    #[test]
    fn diagonal_eigenvalue_cases_are_duplicated() {
        let positive = HermitianMatrix::try_new(
            2,
            vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(2.0, 0.0)],
        )
        .expect("Hermitian diagonal");
        assert_eq!(
            positive.realify().expect("realification").as_slice(),
            &[
                1.0, 0.0, 0.0, 0.0,
                0.0, 2.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 2.0,
            ]
        );

        let singular = HermitianMatrix::try_new(1, vec![c(0.0, 0.0)]).expect("Hermitian zero");
        assert_eq!(
            singular.realify().expect("realification").as_slice(),
            &[0.0, 0.0, 0.0, 0.0]
        );

        let indefinite = HermitianMatrix::try_new(
            2,
            vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)],
        )
        .expect("Hermitian indefinite");
        let real_indefinite = indefinite.realify().expect("realification");
        assert!(real_quadratic(&real_indefinite, &[0.0, 1.0, 0.0, 0.0]) < 0.0);
    }

    #[test]
    fn clarabel_triangle_uses_columnwise_upper_triangle_and_sqrt_two() {
        let matrix = RealSymmetricMatrix::try_new(
            3,
            vec![
                1.0, 2.0, 3.0,
                2.0, 4.0, 5.0,
                3.0, 5.0, 6.0,
            ],
        )
        .expect("symmetric fixture");
        let packed = matrix
            .to_clarabel_psd_triangle()
            .expect("finite Clarabel PSD vector");
        let expected = [
            1.0,
            2.0_f64.sqrt() * 2.0,
            4.0,
            2.0_f64.sqrt() * 3.0,
            2.0_f64.sqrt() * 5.0,
            6.0,
        ];
        assert_eq!(packed.len(), expected.len());
        for (actual, expected) in packed.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn clarabel_triangle_preserves_frobenius_inner_product() {
        let x = RealSymmetricMatrix::try_new(2, vec![1.0, 2.0, 2.0, 3.0])
            .expect("symmetric x");
        let y = RealSymmetricMatrix::try_new(2, vec![4.0, 5.0, 5.0, 6.0])
            .expect("symmetric y");
        let sx = x
            .to_clarabel_psd_triangle()
            .expect("finite x PSD vector");
        let sy = y
            .to_clarabel_psd_triangle()
            .expect("finite y PSD vector");
        let packed_dot: f64 = sx.iter().zip(&sy).map(|(a, b)| a * b).sum();
        let matrix_inner_product = x
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        assert!((packed_dot - matrix_inner_product).abs() < 1.0e-12);
        assert!((packed_dot - 42.0).abs() < 1.0e-12);
    }

    #[test]
    fn clarabel_triangle_round_trips_through_the_documented_scaling() {
        let original = RealSymmetricMatrix::try_new(
            3,
            vec![
                2.0, 1.0, -3.0,
                1.0, 4.0, 5.0,
                -3.0, 5.0, 6.0,
            ],
        )
        .expect("symmetric fixture");
        let packed = original
            .to_clarabel_psd_triangle()
            .expect("finite packed triangle");
        let reconstructed =
            RealSymmetricMatrix::from_clarabel_psd_triangle(3, &packed)
                .expect("valid packed triangle");
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn clarabel_triangle_rejects_bad_inverse_inputs() {
        assert_eq!(
            RealSymmetricMatrix::from_clarabel_psd_triangle(2, &[1.0]),
            Err(MatrixError::LengthMismatch {
                expected: 3,
                actual: 1,
            })
        );
        assert_eq!(
            RealSymmetricMatrix::from_clarabel_psd_triangle(1, &[f64::NAN]),
            Err(MatrixError::NonFinitePackedEntry { index: 0 })
        );
    }

    #[test]
    fn clarabel_triangle_rejects_scaling_overflow() {
        let matrix = RealSymmetricMatrix::try_new(
            2,
            vec![0.0, f64::MAX, f64::MAX, 0.0],
        )
        .expect("finite symmetric input");
        assert_eq!(
            matrix.to_clarabel_psd_triangle(),
            Err(MatrixError::NonFinitePackedEntry { index: 1 })
        );
    }

    #[test]
    fn port_permutation_commutes_with_realification() {
        let matrix = two_by_two_hermitian();
        let permuted = matrix.permute(&[1, 0]).expect("valid permutation");
        let actual = permuted.realify().expect("permuted realification");
        let original = matrix.realify().expect("original realification");
        let q = [1, 0, 3, 2];
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(
                    actual.get(row, col),
                    original.get(q[row], q[col]),
                    "entry ({row},{col})"
                );
            }
        }
    }

    #[test]
    fn rejects_non_hermitian_inputs_without_repair() {
        let diagonal = HermitianMatrix::try_new(1, vec![c(1.0, 1.0)]);
        assert_eq!(
            diagonal,
            Err(MatrixError::NonRealHermitianDiagonal { index: 0 })
        );

        let off_diagonal = HermitianMatrix::try_new(
            2,
            vec![c(1.0, 0.0), c(1.0, 2.0), c(1.0, 2.0), c(1.0, 0.0)],
        );
        assert_eq!(
            off_diagonal,
            Err(MatrixError::NonHermitianPair { row: 0, col: 1 })
        );
    }

    #[test]
    fn rejects_zero_dimension_bad_length_and_non_finite_values() {
        assert_eq!(
            HermitianMatrix::try_new(0, vec![]),
            Err(MatrixError::ZeroDimension)
        );
        assert_eq!(
            HermitianMatrix::try_new(2, vec![c(0.0, 0.0)]),
            Err(MatrixError::LengthMismatch {
                expected: 4,
                actual: 1
            })
        );
        assert_eq!(
            HermitianMatrix::try_new(1, vec![c(f64::NAN, 0.0)]),
            Err(MatrixError::NonFiniteEntry { index: 0 })
        );
        assert_eq!(
            RealSymmetricMatrix::try_new(1, vec![f64::INFINITY]),
            Err(MatrixError::NonFiniteEntry { index: 0 })
        );
    }

    #[test]
    fn rejects_invalid_port_permutations() {
        let matrix = two_by_two_hermitian();
        assert_eq!(
            matrix.permute(&[0]),
            Err(MatrixError::InvalidPermutationLength {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            matrix.permute(&[0, 0]),
            Err(MatrixError::DuplicatePermutationIndex { index: 0 })
        );
        assert_eq!(
            matrix.permute(&[0, 2]),
            Err(MatrixError::PermutationIndexOutOfRange { index: 2, order: 2 })
        );
    }
}
