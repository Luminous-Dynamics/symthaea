// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent exact oracle for the first finite symmetry benchmark fixtures.
//!
//! This module validates finite group closure, exact point-set stabilizers and
//! polynomial invariance using integer arithmetic. It does not implement or
//! benchmark any learned discovery method.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub type Matrix2 = [[i8; 2]; 2];
pub type Point2 = [i32; 2];
pub type Exponent2 = (u8, u8);
pub type Polynomial2 = Vec<(Exponent2, i64)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupElementV1 {
    pub id: &'static str,
    pub matrix: Matrix2,
}

pub const D4_V1: [GroupElementV1; 8] = [
    GroupElementV1 { id: "e", matrix: [[1, 0], [0, 1]] },
    GroupElementV1 { id: "r90", matrix: [[0, -1], [1, 0]] },
    GroupElementV1 { id: "r180", matrix: [[-1, 0], [0, -1]] },
    GroupElementV1 { id: "r270", matrix: [[0, 1], [-1, 0]] },
    GroupElementV1 { id: "fx", matrix: [[1, 0], [0, -1]] },
    GroupElementV1 { id: "fy", matrix: [[-1, 0], [0, 1]] },
    GroupElementV1 { id: "fdiag", matrix: [[0, 1], [1, 0]] },
    GroupElementV1 { id: "fanti", matrix: [[0, -1], [-1, 0]] },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleErrorV1 {
    ArithmeticOverflow,
    DuplicateMonomial { exponent: Exponent2 },
}

impl fmt::Display for OracleErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for OracleErrorV1 {}

pub fn multiply_matrices(left: Matrix2, right: Matrix2) -> Matrix2 {
    let mut output = [[0_i8; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            output[row][column] =
                left[row][0] * right[0][column] + left[row][1] * right[1][column];
        }
    }
    output
}

pub fn validate_finite_group(group: &[GroupElementV1]) -> bool {
    if group.is_empty() {
        return false;
    }
    let identity = [[1, 0], [0, 1]];
    if group.iter().filter(|element| element.matrix == identity).count() != 1 {
        return false;
    }
    for left in group {
        for right in group {
            let product = multiply_matrices(left.matrix, right.matrix);
            if !group.iter().any(|candidate| candidate.matrix == product) {
                return false;
            }
        }
        let has_two_sided_inverse = group.iter().any(|candidate| {
            multiply_matrices(left.matrix, candidate.matrix) == identity
                && multiply_matrices(candidate.matrix, left.matrix) == identity
        });
        if !has_two_sided_inverse {
            return false;
        }
    }
    true
}

pub fn apply_matrix(matrix: Matrix2, point: Point2) -> Point2 {
    [
        i32::from(matrix[0][0]) * point[0] + i32::from(matrix[0][1]) * point[1],
        i32::from(matrix[1][0]) * point[0] + i32::from(matrix[1][1]) * point[1],
    ]
}

/// Exact unordered point-set stabilizer; point order and repeated input order
/// do not affect the result. The caller supplies integer coordinates.
pub fn point_set_stabilizer(
    group: &[GroupElementV1],
    points: &[Point2],
) -> Vec<&'static str> {
    let original: BTreeSet<Point2> = points.iter().copied().collect();
    let mut result = group
        .iter()
        .filter(|element| {
            let transformed: BTreeSet<Point2> = points
                .iter()
                .map(|point| apply_matrix(element.matrix, *point))
                .collect();
            transformed == original
        })
        .map(|element| element.id)
        .collect::<Vec<_>>();
    result.sort_unstable();
    result
}

fn checked_pow(base: i64, exponent: u8) -> Result<i64, OracleErrorV1> {
    let mut value = 1_i64;
    for _ in 0..exponent {
        value = value
            .checked_mul(base)
            .ok_or(OracleErrorV1::ArithmeticOverflow)?;
    }
    Ok(value)
}

fn checked_binomial(n: u8, k: u8) -> Result<i64, OracleErrorV1> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut value = 1_i64;
    for index in 0..k {
        value = value
            .checked_mul(i64::from(n - index))
            .ok_or(OracleErrorV1::ArithmeticOverflow)?;
        value /= i64::from(index + 1);
    }
    Ok(value)
}

fn checked_term_product(values: &[i64]) -> Result<i64, OracleErrorV1> {
    values.iter().try_fold(1_i64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(OracleErrorV1::ArithmeticOverflow)
    })
}

/// Substitute x -> a*x+b*y and y -> c*x+d*y, then collect coefficients.
/// All operations are exact checked i64 arithmetic.
pub fn transform_polynomial(
    polynomial: &Polynomial2,
    matrix: Matrix2,
) -> Result<BTreeMap<Exponent2, i64>, OracleErrorV1> {
    let mut output = BTreeMap::<Exponent2, i64>::new();
    let a = i64::from(matrix[0][0]);
    let b = i64::from(matrix[0][1]);
    let c = i64::from(matrix[1][0]);
    let d = i64::from(matrix[1][1]);

    for ((power_x, power_y), coefficient) in polynomial {
        for choose_x in 0..=*power_x {
            for choose_y in 0..=*power_y {
                let coefficient_term = checked_term_product(&[
                    *coefficient,
                    checked_binomial(*power_x, choose_x)?,
                    checked_pow(a, choose_x)?,
                    checked_pow(b, *power_x - choose_x)?,
                    checked_binomial(*power_y, choose_y)?,
                    checked_pow(c, choose_y)?,
                    checked_pow(d, *power_y - choose_y)?,
                ])?;
                let exponent = (
                    choose_x + choose_y,
                    (*power_x - choose_x) + (*power_y - choose_y),
                );
                let previous = output.get(&exponent).copied().unwrap_or(0);
                let updated = previous
                    .checked_add(coefficient_term)
                    .ok_or(OracleErrorV1::ArithmeticOverflow)?;
                output.insert(exponent, updated);
            }
        }
    }
    output.retain(|_, coefficient| *coefficient != 0);
    Ok(output)
}

pub fn polynomial_coefficients(
    polynomial: &Polynomial2,
) -> Result<BTreeMap<Exponent2, i64>, OracleErrorV1> {
    let mut output = BTreeMap::new();
    for (exponent, coefficient) in polynomial {
        let previous = output.get(exponent).copied().unwrap_or(0_i64);
        let updated = previous
            .checked_add(*coefficient)
            .ok_or(OracleErrorV1::ArithmeticOverflow)?;
        output.insert(*exponent, updated);
    }
    output.retain(|_, coefficient| *coefficient != 0);
    Ok(output)
}

pub fn polynomial_stabilizer(
    group: &[GroupElementV1],
    polynomial: &Polynomial2,
) -> Result<Vec<&'static str>, OracleErrorV1> {
    let original = polynomial_coefficients(polynomial)?;
    let mut result = Vec::new();
    for element in group {
        if transform_polynomial(polynomial, element.matrix)? == original {
            result.push(element.id);
        }
    }
    result.sort_unstable();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids_equal(actual: &[&str], expected: &[&str]) -> bool {
        let actual: BTreeSet<&str> = actual.iter().copied().collect();
        let expected: BTreeSet<&str> = expected.iter().copied().collect();
        actual == expected
    }

    #[test]
    fn d4_table_is_a_closed_group_with_identity_and_inverses() {
        assert_eq!(D4_V1.len(), 8);
        assert!(validate_finite_group(&D4_V1));
    }

    #[test]
    fn square_point_set_has_exact_d4_stabilizer() {
        let square = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
        assert!(ids_equal(
            &point_set_stabilizer(&D4_V1, &square),
            &["e", "r90", "r180", "r270", "fx", "fy", "fdiag", "fanti"]
        ));
    }

    #[test]
    fn chiral_four_point_orbit_has_c4_stabilizer_inside_d4() {
        let orbit = [[1, 2], [-2, 1], [-1, -2], [2, -1]];
        assert!(ids_equal(
            &point_set_stabilizer(&D4_V1, &orbit),
            &["e", "r90", "r180", "r270"]
        ));
    }

    #[test]
    fn near_square_perturbation_has_only_identity_stabilizer() {
        // The benchmark manifest's decimal coordinates are multiplied by 100.
        let near_square = [[100, 101], [-100, 100], [-100, -100], [100, -100]];
        assert!(ids_equal(
            &point_set_stabilizer(&D4_V1, &near_square),
            &["e"]
        ));
    }

    #[test]
    fn point_stabilizer_is_invariant_to_input_order_and_group_rotation() {
        let square = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
        let permuted = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
        let rotated = square.map(|point| apply_matrix(D4_V1[1].matrix, point));
        assert!(ids_equal(
            &point_set_stabilizer(&D4_V1, &square),
            &point_set_stabilizer(&D4_V1, &permuted)
        ));
        assert!(ids_equal(
            &point_set_stabilizer(&D4_V1, &square),
            &point_set_stabilizer(&D4_V1, &rotated)
        ));
    }

    #[test]
    fn x4_plus_y4_is_d4_invariant() {
        let polynomial = vec![((4, 0), 1), ((0, 4), 1)];
        assert!(ids_equal(
            &polynomial_stabilizer(&D4_V1, &polynomial).unwrap(),
            &["e", "r90", "r180", "r270", "fx", "fy", "fdiag", "fanti"]
        ));
    }

    #[test]
    fn mixed_quartic_is_c4_invariant_but_reflection_breaking() {
        let polynomial = vec![
            ((4, 0), 1),
            ((0, 4), 1),
            ((3, 1), 1),
            ((1, 3), -1),
        ];
        assert!(ids_equal(
            &polynomial_stabilizer(&D4_V1, &polynomial).unwrap(),
            &["e", "r90", "r180", "r270"]
        ));
    }

    #[test]
    fn polynomial_duplicate_terms_are_collected_exactly() {
        let polynomial = vec![((4, 0), 1), ((4, 0), 2), ((0, 4), 1)];
        assert_eq!(
            polynomial_coefficients(&polynomial).unwrap().get(&(4, 0)),
            Some(&3)
        );
    }

    #[test]
    fn checked_integer_expansion_fails_closed_on_overflow() {
        let polynomial = vec![((16, 0), i64::MAX)];
        assert_eq!(
            transform_polynomial(&polynomial, D4_V1[0].matrix),
            Err(OracleErrorV1::ArithmeticOverflow)
        );
    }
}
