#![forbid(unsafe_code)]

//! Exact finite-grid transforms and weighted-profile experiments for complexity research.

pub mod candidate_successor;
pub mod design;
pub mod matrix;
pub mod permutation_polynomial;
pub mod profile;
pub mod semiring;

use std::error::Error;
use std::fmt;

/// The arithmetic operation that overflowed while transforming a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithmeticStage {
    ForwardMultiply,
    ForwardNegate,
    ForwardAdd,
    InverseMultiply,
    InverseAdd,
}

/// Fail-closed errors from finite-grid construction, indexing, and transforms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GridError {
    SideLengthOverflow,
    PointCountOverflow,
    AllocationFailed { entries: usize },
    InvalidTableLength { expected: usize, actual: usize },
    CoordinateDimensionMismatch { expected: usize, actual: usize },
    CoordinateOutOfRange { axis: usize, value: usize, maximum: usize },
    IndexOutOfRange { index: usize, entries: usize },
    BinomialOverflow { n: usize, k: usize },
    ArithmeticOverflow {
        stage: ArithmeticStage,
        axis: usize,
        output_index: usize,
    },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SideLengthOverflow => write!(f, "alphabet side length overflow"),
            Self::PointCountOverflow => write!(f, "finite-grid point count overflow"),
            Self::AllocationFailed { entries } => {
                write!(f, "unable to reserve a table of {entries} entries")
            }
            Self::InvalidTableLength { expected, actual } => {
                write!(f, "table length {actual} does not match expected length {expected}")
            }
            Self::CoordinateDimensionMismatch { expected, actual } => write!(
                f,
                "coordinate has {actual} dimensions; expected {expected}"
            ),
            Self::CoordinateOutOfRange { axis, value, maximum } => write!(
                f,
                "coordinate {value} on axis {axis} exceeds maximum {maximum}"
            ),
            Self::IndexOutOfRange { index, entries } => {
                write!(f, "flat index {index} is outside table length {entries}")
            }
            Self::BinomialOverflow { n, k } => {
                write!(f, "binomial coefficient C({n},{k}) exceeds u128")
            }
            Self::ArithmeticOverflow { stage, axis, output_index } => write!(
                f,
                "checked arithmetic overflow at {stage:?}, axis {axis}, flat index {output_index}"
            ),
        }
    }
}

impl Error for GridError {}

/// Shape of the product grid {0, ..., degree}^variables.
///
/// Flattening is little-endian in the coordinates: axis 0 is the least
/// significant base-(degree + 1) digit. The empty product has one point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridShape {
    degree: usize,
    variables: usize,
    side: usize,
    entries: usize,
}

impl GridShape {
    /// Construct a grid, checking both the alphabet size and total point count.
    pub fn new(degree: usize, variables: usize) -> Result<Self, GridError> {
        let side = degree.checked_add(1).ok_or(GridError::SideLengthOverflow)?;
        let mut entries = 1usize;
        for _ in 0..variables {
            entries = entries
                .checked_mul(side)
                .ok_or(GridError::PointCountOverflow)?;
        }
        Ok(Self {
            degree,
            variables,
            side,
            entries,
        })
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn variables(&self) -> usize {
        self.variables
    }

    pub fn alphabet_size(&self) -> usize {
        self.side
    }

    pub fn point_count(&self) -> usize {
        self.entries
    }

    /// Map a coordinate tuple to its flat table index.
    pub fn index_of(&self, coordinates: &[usize]) -> Result<usize, GridError> {
        if coordinates.len() != self.variables {
            return Err(GridError::CoordinateDimensionMismatch {
                expected: self.variables,
                actual: coordinates.len(),
            });
        }

        let mut index = 0usize;
        let mut stride = 1usize;
        for (axis, &value) in coordinates.iter().enumerate() {
            if value > self.degree {
                return Err(GridError::CoordinateOutOfRange {
                    axis,
                    value,
                    maximum: self.degree,
                });
            }
            let contribution = value
                .checked_mul(stride)
                .ok_or(GridError::PointCountOverflow)?;
            index = index
                .checked_add(contribution)
                .ok_or(GridError::PointCountOverflow)?;
            stride = stride
                .checked_mul(self.side)
                .ok_or(GridError::PointCountOverflow)?;
        }
        Ok(index)
    }

    /// Decode a flat table index into its coordinate tuple.
    pub fn coordinates_of(&self, index: usize) -> Result<Vec<usize>, GridError> {
        if index >= self.entries {
            return Err(GridError::IndexOutOfRange {
                index,
                entries: self.entries,
            });
        }
        let mut coordinates = Vec::new();
        coordinates
            .try_reserve_exact(self.variables)
            .map_err(|_| GridError::AllocationFailed {
                entries: self.variables,
            })?;
        let mut remainder = index;
        for _ in 0..self.variables {
            coordinates.push(remainder % self.side);
            remainder /= self.side;
        }
        Ok(coordinates)
    }

    /// Apply the tensor-product forward-difference transform.
    ///
    /// For every multi-index alpha:
    ///
    /// Delta^alpha f(0) =
    ///   sum_{u <= alpha} (-1)^(|alpha|-|u|)
    ///     product_i C(alpha_i, u_i) * f(u).
    ///
    /// Every intermediate operation is checked. Overflow is an error, never
    /// wrapping or saturation.
    pub fn forward_differences(&self, values: &[i128]) -> Result<Vec<i128>, GridError> {
        self.check_table(values)?;
        let mut current = copy_table(values)?;
        let mut stride = 1usize;

        for axis in 0..self.variables {
            current = self.transform_axis(&current, axis, stride, false)?;
            stride = stride
                .checked_mul(self.side)
                .ok_or(GridError::PointCountOverflow)?;
        }
        Ok(current)
    }

    /// Reconstruct the table from its tensor-product Newton coefficients.
    ///
    /// f(x) = sum_{alpha <= x} product_i C(x_i, alpha_i) * Delta^alpha f(0).
    pub fn reconstruct(&self, coefficients: &[i128]) -> Result<Vec<i128>, GridError> {
        self.check_table(coefficients)?;
        let mut current = copy_table(coefficients)?;
        let mut stride = 1usize;

        for axis in 0..self.variables {
            current = self.transform_axis(&current, axis, stride, true)?;
            stride = stride
                .checked_mul(self.side)
                .ok_or(GridError::PointCountOverflow)?;
        }
        Ok(current)
    }

    fn check_table(&self, table: &[i128]) -> Result<(), GridError> {
        if table.len() != self.entries {
            return Err(GridError::InvalidTableLength {
                expected: self.entries,
                actual: table.len(),
            });
        }
        Ok(())
    }

    fn transform_axis(
        &self,
        source: &[i128],
        axis: usize,
        stride: usize,
        inverse: bool,
    ) -> Result<Vec<i128>, GridError> {
        let width = stride
            .checked_mul(self.side)
            .ok_or(GridError::PointCountOverflow)?;
        let blocks = self.entries / width;
        let mut destination = zeroed_table(self.entries)?;

        for block in 0..blocks {
            let base = block * width;
            for offset in 0..stride {
                for target_coordinate in 0..self.side {
                    let output_index = base + offset + target_coordinate * stride;
                    let mut accumulator = 0i128;

                    for source_coordinate in 0..=target_coordinate {
                        let binomial = checked_binomial(target_coordinate, source_coordinate)
                            .map_err(|_| GridError::BinomialOverflow {
                                n: target_coordinate,
                                k: source_coordinate,
                            })?;
                        let weight = i128::try_from(binomial).map_err(|_| {
                            GridError::BinomialOverflow {
                                n: target_coordinate,
                                k: source_coordinate,
                            }
                        })?;
                        let input_index = base + offset + source_coordinate * stride;
                        let mut term = source[input_index].checked_mul(weight).ok_or(
                            GridError::ArithmeticOverflow {
                                stage: if inverse {
                                    ArithmeticStage::InverseMultiply
                                } else {
                                    ArithmeticStage::ForwardMultiply
                                },
                                axis,
                                output_index,
                            },
                        )?;

                        if !inverse && (target_coordinate - source_coordinate) % 2 == 1 {
                            term = term.checked_neg().ok_or(GridError::ArithmeticOverflow {
                                stage: ArithmeticStage::ForwardNegate,
                                axis,
                                output_index,
                            })?;
                        }

                        accumulator = accumulator.checked_add(term).ok_or(
                            GridError::ArithmeticOverflow {
                                stage: if inverse {
                                    ArithmeticStage::InverseAdd
                                } else {
                                    ArithmeticStage::ForwardAdd
                                },
                                axis,
                                output_index,
                            },
                        )?;
                    }

                    destination[output_index] = accumulator;
                }
            }
        }

        Ok(destination)
    }
}

fn zeroed_table(entries: usize) -> Result<Vec<i128>, GridError> {
    let mut table = Vec::new();
    table
        .try_reserve_exact(entries)
        .map_err(|_| GridError::AllocationFailed { entries })?;
    table.resize(entries, 0);
    Ok(table)
}

fn copy_table(values: &[i128]) -> Result<Vec<i128>, GridError> {
    let mut table = zeroed_table(values.len())?;
    table.copy_from_slice(values);
    Ok(table)
}

/// Compute C(n,k) exactly in u128, reducing factors before multiplication.
fn checked_binomial(n: usize, k: usize) -> Result<u128, GridError> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut result = 1u128;

    for i in 1..=k {
        let mut numerator = (n - k + i) as u128;
        let mut denominator = i as u128;

        let first_gcd = gcd(numerator, denominator);
        numerator /= first_gcd;
        denominator /= first_gcd;

        let second_gcd = gcd(result, denominator);
        result /= second_gcd;
        denominator /= second_gcd;

        // The recurrence is integral; a remaining denominator means the
        // cancellation invariant was violated, so fail closed.
        if denominator != 1 {
            return Err(GridError::BinomialOverflow { n, k });
        }
        result = result
            .checked_mul(numerator)
            .ok_or(GridError::BinomialOverflow { n, k })?;
    }

    Ok(result)
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Exact number of forward-difference terms for one tensor transform.
///
/// This reports work for the finite-table reference transform, not the
/// complexity of the OpenAI algorithm.
pub fn forward_term_count(shape: &GridShape) -> Option<u128> {
    if shape.variables == 0 {
        return Some(0);
    }
    let mut n = shape.variables as u128;
    let mut entries = shape.entries as u128;
    let mut side_plus_one = (shape.side as u128).checked_add(1)?;

    // Divide one even factor before multiplying. This avoids returning None
    // merely because the unreduced intermediate product overflows u128 while
    // the mathematically divided count would still fit.
    if n % 2 == 0 {
        n /= 2;
    } else if entries % 2 == 0 {
        entries /= 2;
    } else {
        side_plus_one /= 2;
    }

    n.checked_mul(entries)?.checked_mul(side_plus_one)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_value(state: &mut u64) -> i128 {
        // Fixed xorshift64* stream: deterministic across test runs.
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let value = (*state).wrapping_mul(0x2545_F491_4F6C_DD1D);
        (value % 101) as i128 - 50
    }

    #[test]
    fn round_trip_random_integer_tables_on_small_grids() {
        let mut seed = 0xE7_00_02_11_2026u64;
        let mut cases = 0usize;

        for degree in 0..=3 {
            for variables in 0..=4 {
                let shape = GridShape::new(degree, variables).unwrap();
                for _ in 0..16 {
                    let mut values = Vec::with_capacity(shape.point_count());
                    for _ in 0..shape.point_count() {
                        values.push(next_value(&mut seed));
                    }
                    let coefficients = shape.forward_differences(&values).unwrap();
                    let reconstructed = shape.reconstruct(&coefficients).unwrap();
                    assert_eq!(reconstructed, values);
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 320);
    }

    #[test]
    fn boolean_two_variable_coefficients_match_subset_mobius() {
        let shape = GridShape::new(1, 2).unwrap();
        // Axis 0 is the least significant coordinate:
        // f(0,0)=3, f(1,0)=5, f(0,1)=-1, f(1,1)=8.
        let values = vec![3i128, 5, -1, 8];
        assert_eq!(shape.forward_differences(&values).unwrap(), vec![3i128, 2, -4, 7]);
    }

    #[test]
    fn qary_newton_coefficients_reconstruct_a_quadratic() {
        let shape = GridShape::new(2, 1).unwrap();
        // f(x)=x^2 on {0,1,2}; Newton coefficients are [0,1,2].
        let values = vec![0i128, 1, 4];
        let coefficients = shape.forward_differences(&values).unwrap();
        assert_eq!(coefficients, vec![0i128, 1, 2]);
        assert_eq!(shape.reconstruct(&coefficients).unwrap(), values);
    }

    #[test]
    fn zero_variables_and_zero_degree_are_well_defined() {
        let empty_product = GridShape::new(3, 0).unwrap();
        assert_eq!(empty_product.point_count(), 1);
        assert_eq!(empty_product.forward_differences(&[17]).unwrap(), vec![17i128]);
        assert_eq!(empty_product.reconstruct(&[17]).unwrap(), vec![17i128]);

        let singleton_alphabet = GridShape::new(0, 5).unwrap();
        assert_eq!(singleton_alphabet.point_count(), 1);
        assert_eq!(singleton_alphabet.forward_differences(&[23]).unwrap(), vec![23i128]);
        assert_eq!(singleton_alphabet.reconstruct(&[23]).unwrap(), vec![23i128]);
    }

    #[test]
    fn coordinate_encoding_is_bijective_on_small_grids() {
        for degree in 0..=3 {
            for variables in 0..=4 {
                let shape = GridShape::new(degree, variables).unwrap();
                for index in 0..shape.point_count() {
                    let coordinates = shape.coordinates_of(index).unwrap();
                    assert_eq!(shape.index_of(&coordinates).unwrap(), index);
                }
            }
        }
    }

    #[test]
    fn invalid_lengths_and_coordinates_are_rejected() {
        let shape = GridShape::new(2, 2).unwrap();
        assert!(matches!(
            shape.forward_differences(&[1, 2]),
            Err(GridError::InvalidTableLength { .. })
        ));
        assert!(matches!(
            shape.index_of(&[0]),
            Err(GridError::CoordinateDimensionMismatch { .. })
        ));
        assert!(matches!(
            shape.index_of(&[0, 3]),
            Err(GridError::CoordinateOutOfRange { axis: 1, .. })
        ));
        assert!(matches!(
            shape.coordinates_of(shape.point_count()),
            Err(GridError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn arithmetic_overflow_fails_closed() {
        let shape = GridShape::new(1, 1).unwrap();
        let result = shape.forward_differences(&[i128::MIN, 0]);
        assert!(matches!(
            result,
            Err(GridError::ArithmeticOverflow {
                stage: ArithmeticStage::ForwardNegate,
                axis: 0,
                ..
            })
        ));
    }

    #[test]
    fn point_count_overflow_is_reported() {
        assert!(matches!(
            GridShape::new(1, usize::BITS as usize),
            Err(GridError::PointCountOverflow)
        ));
    }

    #[test]
    fn binomial_computation_uses_exact_cancellation() {
        assert_eq!(
            checked_binomial(67, 33).unwrap(),
            14_226_520_737_620_288_370u128
        );
        assert_eq!(checked_binomial(8, 0).unwrap(), 1);
        assert_eq!(checked_binomial(8, 9).unwrap(), 0);
        assert!(checked_binomial(200, 100).is_err());
    }

    #[test]
    fn forward_term_count_matches_explicit_axis_work() {
        for degree in 0..=3 {
            for variables in 0..=4 {
                let shape = GridShape::new(degree, variables).unwrap();
                let mut explicit = 0u128;
                let mut stride = 1usize;

                for _axis in 0..variables {
                    let width = stride * shape.alphabet_size();
                    let blocks = shape.point_count() / width;
                    for _block in 0..blocks {
                        for _offset in 0..stride {
                            for target_coordinate in 0..shape.alphabet_size() {
                                explicit += (target_coordinate + 1) as u128;
                            }
                        }
                    }
                    stride *= shape.alphabet_size();
                }

                assert_eq!(forward_term_count(&shape), Some(explicit));
            }
        }
    }
}
