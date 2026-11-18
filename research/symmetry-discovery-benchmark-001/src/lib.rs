// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact finite symmetry benchmark oracles.
//!
//! This crate composes with the finite quotient compiler but independently
//! reconstructs the D4 action, structural stabilizers, polynomial stabilizers,
//! exhaustive candidate orbits, and lifted minimizers.

use std::collections::{BTreeMap, BTreeSet};

use symmetry_quotient_reference::{
    CandidateSemanticsV1, CompileOutcomeV1, FiniteActionV1, FiniteGroupV1,
    MultiEnvironmentProblemV1, QualificationBudgetV1, QuotientErrorV1, SearchProblemV1,
    TaskEnvironmentV1, compile_contextual_multi_environment_quotient, compile_quotient,
    lift_representative_solution, verify_certificate,
    verify_contextual_multi_environment_quotient,
};

pub type Matrix2 = [[i64; 2]; 2];
pub type Point2 = [i64; 2];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolynomialTermV1 {
    pub x_degree: u32,
    pub y_degree: u32,
    pub coefficient: i64,
}

pub const D4_MATRICES: [Matrix2; 8] = [
    [[1, 0], [0, 1]],
    [[0, -1], [1, 0]],
    [[-1, 0], [0, -1]],
    [[0, 1], [-1, 0]],
    [[1, 0], [0, -1]],
    [[-1, 0], [0, 1]],
    [[0, 1], [1, 0]],
    [[0, -1], [-1, 0]],
];

pub const SQUARE_POINTS: [Point2; 4] = [
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
];

pub const CHIRAL_C4_POINTS: [Point2; 4] = [
    [1, 2],
    [-2, 1],
    [-1, -2],
    [2, -1],
];

pub const NEAR_D4_POINTS_SCALED: [Point2; 4] = [
    [100, 101],
    [-100, 100],
    [-100, -100],
    [100, -100],
];

pub const D4_POLYNOMIAL: [PolynomialTermV1; 2] = [
    PolynomialTermV1 { x_degree: 4, y_degree: 0, coefficient: 1 },
    PolynomialTermV1 { x_degree: 0, y_degree: 4, coefficient: 1 },
];

pub const C4_POLYNOMIAL: [PolynomialTermV1; 4] = [
    PolynomialTermV1 { x_degree: 4, y_degree: 0, coefficient: 1 },
    PolynomialTermV1 { x_degree: 0, y_degree: 4, coefficient: 1 },
    PolynomialTermV1 { x_degree: 3, y_degree: 1, coefficient: 1 },
    PolynomialTermV1 { x_degree: 1, y_degree: 3, coefficient: -1 },
];

pub fn multiply_matrix(left: Matrix2, right: Matrix2) -> Matrix2 {
    let mut product = [[0_i64; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            for inner in 0..2 {
                product[row][column] += left[row][inner] * right[inner][column];
            }
        }
    }
    product
}

pub fn apply_matrix(matrix: Matrix2, point: Point2) -> Point2 {
    [
        matrix[0][0] * point[0] + matrix[0][1] * point[1],
        matrix[1][0] * point[0] + matrix[1][1] * point[1],
    ]
}

pub fn d4_group_and_action() -> FiniteActionV1 {
    let mut multiplication = vec![vec![0_usize; 8]; 8];
    for left in 0..8 {
        for right in 0..8 {
            let product = multiply_matrix(D4_MATRICES[left], D4_MATRICES[right]);
            multiplication[left][right] = D4_MATRICES
                .iter()
                .position(|candidate| *candidate == product)
                .expect("D4 matrices must be closed");
        }
    }

    let point_indices: BTreeMap<Point2, usize> = SQUARE_POINTS
        .iter()
        .copied()
        .enumerate()
        .map(|(index, point)| (point, index))
        .collect();
    let vertex_permutations: Vec<Vec<usize>> = D4_MATRICES
        .iter()
        .map(|matrix| {
            SQUARE_POINTS
                .iter()
                .map(|point| {
                    *point_indices
                        .get(&apply_matrix(*matrix, *point))
                        .expect("D4 must preserve square vertices")
                })
                .collect()
        })
        .collect();

    let pair_permutations = vertex_permutations
        .iter()
        .map(|permutation| {
            let mut result = vec![0_usize; 16];
            for source in 0..4 {
                for target in 0..4 {
                    result[source * 4 + target] =
                        permutation[source] * 4 + permutation[target];
                }
            }
            result
        })
        .collect();

    FiniteActionV1 {
        action_id: "benchmark-d4-ordered-pair-action-v0.1".to_owned(),
        group: FiniteGroupV1 {
            group_id: "benchmark-D4-matrix-group-v0.1".to_owned(),
            identity: 0,
            multiplication,
        },
        permutations: pair_permutations,
    }
}

pub fn structural_stabilizer(points: &[Point2]) -> Vec<usize> {
    let point_set: BTreeSet<Point2> = points.iter().copied().collect();
    D4_MATRICES
        .iter()
        .enumerate()
        .filter_map(|(element, matrix)| {
            let image: BTreeSet<Point2> = points
                .iter()
                .map(|point| apply_matrix(*matrix, *point))
                .collect();
            (image == point_set).then_some(element)
        })
        .collect()
}

fn binomial(n: u32, k: u32) -> i64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut value = 1_i64;
    for step in 1..=k {
        value = value * i64::from(n - step + 1) / i64::from(step);
    }
    value
}

fn integer_power(base: i64, exponent: u32) -> i64 {
    (0..exponent).fold(1_i64, |value, _| value * base)
}

fn polynomial_map(terms: &[PolynomialTermV1]) -> BTreeMap<(u32, u32), i64> {
    let mut result = BTreeMap::new();
    for term in terms {
        *result
            .entry((term.x_degree, term.y_degree))
            .or_insert(0_i64) += term.coefficient;
    }
    result.retain(|_, coefficient| *coefficient != 0);
    result
}

/// Substitute (x',y') = M(x,y), expand exactly over integer coefficients,
/// and return the canonical polynomial coefficient map.
pub fn transformed_polynomial(
    terms: &[PolynomialTermV1],
    matrix: Matrix2,
) -> BTreeMap<(u32, u32), i64> {
    let mut result = BTreeMap::new();
    for term in terms {
        for x_from_x in 0..=term.x_degree {
            for y_from_x in 0..=term.y_degree {
                let coefficient = term.coefficient
                    * binomial(term.x_degree, x_from_x)
                    * integer_power(matrix[0][0], x_from_x)
                    * integer_power(matrix[0][1], term.x_degree - x_from_x)
                    * binomial(term.y_degree, y_from_x)
                    * integer_power(matrix[1][0], y_from_x)
                    * integer_power(matrix[1][1], term.y_degree - y_from_x);
                let degrees = (
                    x_from_x + y_from_x,
                    (term.x_degree - x_from_x) + (term.y_degree - y_from_x),
                );
                *result.entry(degrees).or_insert(0_i64) += coefficient;
            }
        }
    }
    result.retain(|_, coefficient| *coefficient != 0);
    result
}

pub fn polynomial_stabilizer(terms: &[PolynomialTermV1]) -> Vec<usize> {
    let original = polynomial_map(terms);
    D4_MATRICES
        .iter()
        .enumerate()
        .filter_map(|(element, matrix)| {
            (transformed_polynomial(terms, *matrix) == original).then_some(element)
        })
        .collect()
}

pub fn square_ordered_pair_problem() -> SearchProblemV1 {
    let mut candidate_ids = Vec::with_capacity(16);
    let mut semantics = Vec::with_capacity(16);
    for source in 0..4 {
        for target in 0..4 {
            let dx = SQUARE_POINTS[source][0] - SQUARE_POINTS[target][0];
            let dy = SQUARE_POINTS[source][1] - SQUARE_POINTS[target][1];
            let squared_distance = dx * dx + dy * dy;
            candidate_ids.push(format!("p{source}{target}"));
            semantics.push(CandidateSemanticsV1 {
                in_search_domain: true,
                admissible: true,
                feasible: true,
                objective_cost: squared_distance,
                constraint_signature: vec![squared_distance],
                boundary_condition_id: "square-boundary-v1".to_owned(),
                output_equivalence_id: "squared-distance-output-v1".to_owned(),
            });
        }
    }
    SearchProblemV1 {
        problem_id: "benchmark-square-ordered-pairs-v0.1".to_owned(),
        candidate_ids,
        semantics,
    }
}

pub fn exhaustive_orbits(action: &FiniteActionV1, candidate_count: usize) -> Vec<Vec<usize>> {
    let mut remaining: BTreeSet<usize> = (0..candidate_count).collect();
    let mut result = Vec::new();
    while let Some(representative) = remaining.iter().next().copied() {
        let orbit: BTreeSet<usize> = action
            .permutations
            .iter()
            .map(|permutation| permutation[representative])
            .collect();
        for member in &orbit {
            remaining.remove(member);
        }
        result.push(orbit.into_iter().collect());
    }
    result.sort_by_key(|orbit| orbit[0]);
    result
}

fn fixture_fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_json_manifest_matches_noncryptographic_byte_fingerprint() {
        let manifest = include_bytes!("../fixtures/v0.1/cases.json");
        assert_eq!(manifest.len(), 136_742);
        assert_eq!(fixture_fnv1a64(manifest), 0x9022_19de_12ff_50d1);
    }

    #[test]
    fn d4_matrices_form_an_exact_group_and_square_action() {
        let action = d4_group_and_action();
        let group_order = D4_MATRICES.len();
        assert_eq!(group_order, 8);
        for left in 0..group_order {
            for right in 0..group_order {
                let product = action.group.multiplication[left][right];
                assert!(product < group_order);
                assert_eq!(
                    multiply_matrix(D4_MATRICES[left], D4_MATRICES[right]),
                    D4_MATRICES[product]
                );
            }
        }
        for element in 0..group_order {
            assert_eq!(action.group.multiplication[0][element], element);
            assert_eq!(action.group.multiplication[element][0], element);
            assert!(action.permutations[element].iter().all(|index| *index < 16));
        }
    }

    #[test]
    fn square_chiral_and_near_symmetry_stabilizers_match_frozen_labels() {
        assert_eq!(structural_stabilizer(&SQUARE_POINTS), (0..8).collect::<Vec<_>>());
        assert_eq!(structural_stabilizer(&CHIRAL_C4_POINTS), vec![0, 1, 2, 3]);
        assert_eq!(structural_stabilizer(&NEAR_D4_POINTS_SCALED), vec![0]);
    }


    #[test]
    fn all_frozen_structural_fixtures_match_exact_d4_stabilizers() {
        // Geometry is scaled by a positive scalar only to encode decimal
        // near-symmetry fixtures as exact integer points.
        let cases: &[(&str, &[Point2], &[usize])] = &[
            (
                "D4-SQUARE-POINTSET-001",
                &[[1, 1], [-1, 1], [-1, -1], [1, -1]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "C4-CHIRAL-ORBIT-001",
                &[[1, 2], [-2, 1], [-1, -2], [2, -1]],
                &[0, 1, 2, 3],
            ),
            (
                "D4-NEAR-SYMMETRY-BREAK-001",
                &[[100, 101], [-1, 1], [-1, -1], [1, -1]],
                &[0],
            ),
            (
                "GENERATED-D4-01",
                &[[-7, -4], [-7, 4], [-4, -7], [-4, 7], [4, -7], [4, 7], [7, -4], [7, 4]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-02",
                &[[-31, -25], [-31, 25], [-25, -31], [-25, 31], [25, -31], [25, 31], [31, -25], [31, 25]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-03",
                &[[-38, -21], [-38, 21], [-21, -38], [-21, 38], [21, -38], [21, 38], [38, -21], [38, 21]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-04",
                &[[-35, -32], [-35, 32], [-32, -35], [-32, 35], [32, -35], [32, 35], [35, -32], [35, 32]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-05",
                &[[-39, -38], [-39, 38], [-38, -39], [-38, 39], [38, -39], [38, 39], [39, -38], [39, 38]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-06",
                &[[-28, -23], [-28, 23], [-23, -28], [-23, 28], [23, -28], [23, 28], [28, -23], [28, 23]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-07",
                &[[-45, -21], [-45, 21], [-21, -45], [-21, 45], [21, -45], [21, 45], [45, -21], [45, 21]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-08",
                &[[-50, -22], [-50, 22], [-22, -50], [-22, 50], [22, -50], [22, 50], [50, -22], [50, 22]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-09",
                &[[-48, -41], [-48, 41], [-41, -48], [-41, 48], [41, -48], [41, 48], [48, -41], [48, 41]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-10",
                &[[-50, -26], [-50, 26], [-26, -50], [-26, 50], [26, -50], [26, 50], [50, -26], [50, 26]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-11",
                &[[-38, -22], [-38, 22], [-22, -38], [-22, 38], [22, -38], [22, 38], [38, -22], [38, 22]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-12",
                &[[-41, -23], [-41, 23], [-23, -41], [-23, 41], [23, -41], [23, 41], [41, -23], [41, 23]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-13",
                &[[-42, -32], [-42, 32], [-32, -42], [-32, 42], [32, -42], [32, 42], [42, -32], [42, 32]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-14",
                &[[-12, -11], [-12, 11], [-11, -12], [-11, 12], [11, -12], [11, 12], [12, -11], [12, 11]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-15",
                &[[-40, -5], [-40, 5], [-5, -40], [-5, 40], [5, -40], [5, 40], [40, -5], [40, 5]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-16",
                &[[-40, -7], [-40, 7], [-7, -40], [-7, 40], [7, -40], [7, 40], [40, -7], [40, 7]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-17",
                &[[-40, -3], [-40, 3], [-3, -40], [-3, 40], [3, -40], [3, 40], [40, -3], [40, 3]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-18",
                &[[-42, -28], [-42, 28], [-28, -42], [-28, 42], [28, -42], [28, 42], [42, -28], [42, 28]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-19",
                &[[-43, -33], [-43, 33], [-33, -43], [-33, 43], [33, -43], [33, 43], [43, -33], [43, 33]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-20",
                &[[-27, -2], [-27, 2], [-2, -27], [-2, 27], [2, -27], [2, 27], [27, -2], [27, 2]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-21",
                &[[-21, -10], [-21, 10], [-10, -21], [-10, 21], [10, -21], [10, 21], [21, -10], [21, 10]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-22",
                &[[-21, -3], [-21, 3], [-3, -21], [-3, 21], [3, -21], [3, 21], [21, -3], [21, 3]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-23",
                &[[-25, -1], [-25, 1], [-1, -25], [-1, 25], [1, -25], [1, 25], [25, -1], [25, 1]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-24",
                &[[-28, -15], [-28, 15], [-15, -28], [-15, 28], [15, -28], [15, 28], [28, -15], [28, 15]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-25",
                &[[-24, -11], [-24, 11], [-11, -24], [-11, 24], [11, -24], [11, 24], [24, -11], [24, 11]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-26",
                &[[-18, -6], [-18, 6], [-6, -18], [-6, 18], [6, -18], [6, 18], [18, -6], [18, 6]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-27",
                &[[-40, -30], [-40, 30], [-30, -40], [-30, 40], [30, -40], [30, 40], [40, -30], [40, 30]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-28",
                &[[-17, -15], [-17, 15], [-15, -17], [-15, 17], [15, -17], [15, 17], [17, -15], [17, 15]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-29",
                &[[-48, -5], [-48, 5], [-5, -48], [-5, 48], [5, -48], [5, 48], [48, -5], [48, 5]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-D4-30",
                &[[-46, -43], [-46, 43], [-43, -46], [-43, 46], [43, -46], [43, 46], [46, -43], [46, 43]],
                &[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            (
                "GENERATED-C4-01",
                &[[-43, -42], [-42, 43], [42, -43], [43, 42]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-02",
                &[[-44, 4], [-4, -44], [4, 44], [44, -4]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-03",
                &[[-49, 5], [-5, -49], [5, 49], [49, -5]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-04",
                &[[-12, -1], [-1, 12], [1, -12], [12, 1]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-05",
                &[[-34, 11], [-11, -34], [11, 34], [34, -11]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-06",
                &[[-50, 22], [-22, -50], [22, 50], [50, -22]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-07",
                &[[-43, 11], [-11, -43], [11, 43], [43, -11]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-08",
                &[[-32, -25], [-25, 32], [25, -32], [32, 25]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-09",
                &[[-16, -13], [-13, 16], [13, -16], [16, 13]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-10",
                &[[-19, -7], [-7, 19], [7, -19], [19, 7]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-11",
                &[[-34, -13], [-13, 34], [13, -34], [34, 13]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-12",
                &[[-46, -13], [-13, 46], [13, -46], [46, 13]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-13",
                &[[-16, -9], [-9, 16], [9, -16], [16, 9]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-14",
                &[[-31, 2], [-2, -31], [2, 31], [31, -2]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-15",
                &[[-39, -34], [-34, 39], [34, -39], [39, 34]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-16",
                &[[-37, -7], [-7, 37], [7, -37], [37, 7]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-17",
                &[[-37, 27], [-27, -37], [27, 37], [37, -27]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-18",
                &[[-36, -32], [-32, 36], [32, -36], [36, 32]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-19",
                &[[-48, 27], [-27, -48], [27, 48], [48, -27]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-20",
                &[[-42, 4], [-4, -42], [4, 42], [42, -4]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-21",
                &[[-49, 24], [-24, -49], [24, 49], [49, -24]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-22",
                &[[-38, -6], [-6, 38], [6, -38], [38, 6]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-23",
                &[[-32, 18], [-18, -32], [18, 32], [32, -18]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-24",
                &[[-39, 6], [-6, -39], [6, 39], [39, -6]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-25",
                &[[-46, -11], [-11, 46], [11, -46], [46, 11]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-26",
                &[[-37, -17], [-17, 37], [17, -37], [37, 17]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-27",
                &[[-41, -26], [-26, 41], [26, -41], [41, 26]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-28",
                &[[-20, -2], [-2, 20], [2, -20], [20, 2]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-29",
                &[[-47, -36], [-36, 47], [36, -47], [47, 36]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C4-30",
                &[[-50, -11], [-11, 50], [11, -50], [50, 11]],
                &[0, 1, 2, 3],
            ),
            (
                "GENERATED-C2_R180-01",
                &[[-49, -37], [49, 37]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-02",
                &[[-42, -43], [42, 43]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-03",
                &[[-14, 19], [14, -19]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-04",
                &[[-34, 23], [34, -23]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-05",
                &[[-46, -32], [46, 32]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-06",
                &[[-5, -49], [5, 49]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-07",
                &[[-1, -3], [1, 3]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-08",
                &[[-23, -35], [23, 35]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-09",
                &[[-24, 48], [24, -48]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-10",
                &[[-17, -27], [17, 27]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-11",
                &[[-20, 15], [20, -15]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-12",
                &[[-36, 19], [36, -19]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-13",
                &[[-49, 2], [49, -2]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-14",
                &[[-21, -18], [21, 18]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-15",
                &[[-34, 48], [34, -48]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-16",
                &[[-17, -30], [17, 30]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-17",
                &[[-24, -9], [24, 9]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-18",
                &[[-5, 42], [5, -42]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-19",
                &[[-36, 46], [36, -46]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-20",
                &[[-33, -20], [33, 20]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-21",
                &[[-44, -36], [44, 36]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-22",
                &[[-37, -29], [37, 29]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-23",
                &[[-2, 8], [2, -8]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-24",
                &[[-16, -41], [16, 41]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-25",
                &[[-6, 20], [6, -20]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-26",
                &[[-48, -26], [48, 26]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-27",
                &[[-16, -48], [16, 48]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-28",
                &[[-5, 1], [5, -1]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-29",
                &[[-49, -19], [49, 19]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_R180-30",
                &[[-20, 29], [20, -29]],
                &[0, 2],
            ),
            (
                "GENERATED-C2_FX-01",
                &[[-39, -21], [-39, 21]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-02",
                &[[20, -28], [20, 28]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-03",
                &[[29, -14], [29, 14]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-04",
                &[[-44, -36], [-44, 36]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-05",
                &[[-8, -9], [-8, 9]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-06",
                &[[-13, -9], [-13, 9]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-07",
                &[[-9, -40], [-9, 40]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-08",
                &[[-14, -40], [-14, 40]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-09",
                &[[-40, -14], [-40, 14]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-10",
                &[[-16, -17], [-16, 17]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-11",
                &[[18, -12], [18, 12]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-12",
                &[[-36, -21], [-36, 21]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-13",
                &[[29, -20], [29, 20]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-14",
                &[[43, -49], [43, 49]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-15",
                &[[-29, -46], [-29, 46]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-16",
                &[[-27, -47], [-27, 47]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-17",
                &[[-18, -12], [-18, 12]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-18",
                &[[23, -27], [23, 27]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-19",
                &[[-17, -11], [-17, 11]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-20",
                &[[-5, -38], [-5, 38]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-21",
                &[[4, -15], [4, 15]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-22",
                &[[-36, -27], [-36, 27]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-23",
                &[[46, -45], [46, 45]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-24",
                &[[-15, -38], [-15, 38]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-25",
                &[[20, -50], [20, 50]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-26",
                &[[-4, -28], [-4, 28]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-27",
                &[[5, -30], [5, 30]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-28",
                &[[39, -27], [39, 27]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-29",
                &[[-28, -27], [-28, 27]],
                &[0, 4],
            ),
            (
                "GENERATED-C2_FX-30",
                &[[24, -48], [24, 48]],
                &[0, 4],
            ),
            (
                "GENERATED-C1-01",
                &[[36, -10]],
                &[0],
            ),
            (
                "GENERATED-C1-02",
                &[[-48, 41]],
                &[0],
            ),
            (
                "GENERATED-C1-03",
                &[[39, 17]],
                &[0],
            ),
            (
                "GENERATED-C1-04",
                &[[-11, -16]],
                &[0],
            ),
            (
                "GENERATED-C1-05",
                &[[2, -45]],
                &[0],
            ),
            (
                "GENERATED-C1-06",
                &[[20, -28]],
                &[0],
            ),
            (
                "GENERATED-C1-07",
                &[[1, -17]],
                &[0],
            ),
            (
                "GENERATED-C1-08",
                &[[18, -25]],
                &[0],
            ),
            (
                "GENERATED-C1-09",
                &[[-1, 32]],
                &[0],
            ),
            (
                "GENERATED-C1-10",
                &[[42, 26]],
                &[0],
            ),
            (
                "GENERATED-C1-11",
                &[[-37, 48]],
                &[0],
            ),
            (
                "GENERATED-C1-12",
                &[[-24, -38]],
                &[0],
            ),
            (
                "GENERATED-C1-13",
                &[[-26, 2]],
                &[0],
            ),
            (
                "GENERATED-C1-14",
                &[[-46, 1]],
                &[0],
            ),
            (
                "GENERATED-C1-15",
                &[[17, -21]],
                &[0],
            ),
            (
                "GENERATED-C1-16",
                &[[23, 5]],
                &[0],
            ),
            (
                "GENERATED-C1-17",
                &[[-48, -3]],
                &[0],
            ),
            (
                "GENERATED-C1-18",
                &[[16, 17]],
                &[0],
            ),
            (
                "GENERATED-C1-19",
                &[[-42, 40]],
                &[0],
            ),
            (
                "GENERATED-C1-20",
                &[[-19, -9]],
                &[0],
            ),
            (
                "GENERATED-C1-21",
                &[[19, -32]],
                &[0],
            ),
            (
                "GENERATED-C1-22",
                &[[-22, 44]],
                &[0],
            ),
            (
                "GENERATED-C1-23",
                &[[21, -13]],
                &[0],
            ),
            (
                "GENERATED-C1-24",
                &[[-41, 11]],
                &[0],
            ),
            (
                "GENERATED-C1-25",
                &[[-34, -21]],
                &[0],
            ),
            (
                "GENERATED-C1-26",
                &[[19, 41]],
                &[0],
            ),
            (
                "GENERATED-C1-27",
                &[[18, 49]],
                &[0],
            ),
            (
                "GENERATED-C1-28",
                &[[36, -22]],
                &[0],
            ),
            (
                "GENERATED-C1-29",
                &[[12, -4]],
                &[0],
            ),
            (
                "GENERATED-C1-30",
                &[[-33, 31]],
                &[0],
            ),
        ];
        let frozen_manifest = include_str!("../fixtures/v0.1/cases.json");
        assert_eq!(cases.len(), 153);
        for (case_id, points, expected_stabilizer) in cases {
            assert!(frozen_manifest.contains(case_id), "fixture missing: {case_id}");
            let actual_stabilizer = structural_stabilizer(points);
            assert_eq!(
                actual_stabilizer.as_slice(),
                *expected_stabilizer,
                "stabilizer mismatch for fixture {case_id}"
            );
        }
    }

    #[test]
    fn exact_polynomial_stabilizers_match_d4_and_c4_fixtures() {
        assert_eq!(polynomial_stabilizer(&D4_POLYNOMIAL), (0..8).collect::<Vec<_>>());
        assert_eq!(polynomial_stabilizer(&C4_POLYNOMIAL), vec![0, 1, 2, 3]);
    }

    #[test]
    fn quotient_orbits_match_an_independent_exhaustive_oracle() {
        let action = d4_group_and_action();
        let problem = square_ordered_pair_problem();
        let oracle = exhaustive_orbits(&action, problem.candidate_ids.len());
        let result = compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = result else {
            panic!("the exact D4 distance task should be invariant");
        };
        let actual: Vec<Vec<usize>> = certificate
            .orbits
            .iter()
            .map(|orbit| orbit.members.iter().map(|member| member.candidate_index).collect())
            .collect();
        assert_eq!(actual, oracle);
        assert_eq!(actual.iter().map(Vec::len).collect::<Vec<_>>(), vec![4, 8, 4]);
        verify_certificate(&problem, &action, &certificate, QualificationBudgetV1::default())
            .unwrap();
    }

    #[test]
    fn quotient_lift_recovers_every_original_optimal_candidate() {
        let action = d4_group_and_action();
        let problem = square_ordered_pair_problem();
        let result = compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = result else {
            panic!("expected a verified D4 quotient");
        };

        let best_orbit = certificate
            .orbits
            .iter()
            .find(|orbit| {
                problem.semantics[orbit.representative_index].objective_cost
                    == problem
                        .semantics
                        .iter()
                        .map(|semantics| semantics.objective_cost)
                        .min()
                        .unwrap()
            })
            .unwrap();
        let lifted = lift_representative_solution(
            &problem,
            &action,
            &certificate,
            best_orbit.representative_index,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        let lifted_ids: BTreeSet<String> =
            lifted.iter().map(|candidate| candidate.candidate_id.clone()).collect();
        let exhaustive_minima: BTreeSet<String> = problem
            .candidate_ids
            .iter()
            .zip(problem.semantics.iter())
            .filter(|(_, semantics)| semantics.objective_cost == 0)
            .map(|(candidate_id, _)| candidate_id.clone())
            .collect();
        assert_eq!(lifted_ids, exhaustive_minima);
        assert_eq!(lifted.len(), 4);
    }

    #[test]
    fn broken_boundary_semantics_blocks_hard_pruning() {
        let action = d4_group_and_action();
        let mut problem = square_ordered_pair_problem();
        problem.semantics[0].boundary_condition_id = "special-fixed-boundary".to_owned();
        assert!(matches!(
            compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap(),
            CompileOutcomeV1::Blocked(_)
        ));
    }

    #[test]
    fn contextual_atlas_qualifies_d4_c4_and_c1_without_globalizing_authority() {
        let action = d4_group_and_action();
        let mut candidate_ids = Vec::with_capacity(16);
        let mut d4_semantics = Vec::with_capacity(16);
        let mut c4_semantics = Vec::with_capacity(16);
        let mut c1_semantics = Vec::with_capacity(16);

        for source in 0_usize..4 {
            for target in 0_usize..4 {
                let delta = (target + 4 - source) % 4;
                candidate_ids.push(format!("p{source}{target}"));
                d4_semantics.push(CandidateSemanticsV1 {
                    in_search_domain: true,
                    admissible: true,
                    feasible: true,
                    objective_cost: 0,
                    constraint_signature: vec![0],
                    boundary_condition_id: "same-boundary".to_owned(),
                    output_equivalence_id: "same-output".to_owned(),
                });
                c4_semantics.push(CandidateSemanticsV1 {
                    in_search_domain: true,
                    admissible: true,
                    feasible: true,
                    objective_cost: delta as i64,
                    constraint_signature: vec![delta as i64],
                    boundary_condition_id: "rotating-boundary".to_owned(),
                    output_equivalence_id: "oriented-output".to_owned(),
                });
                c1_semantics.push(CandidateSemanticsV1 {
                    in_search_domain: true,
                    admissible: true,
                    feasible: true,
                    objective_cost: (source * 4 + target) as i64,
                    constraint_signature: vec![0],
                    boundary_condition_id: "asymmetric-boundary".to_owned(),
                    output_equivalence_id: "unique-output".to_owned(),
                });
            }
        }

        let problem = MultiEnvironmentProblemV1 {
            problem_id: "benchmark-contextual-d4-c4-c1-atlas-v0.1".to_owned(),
            candidate_ids,
            environments: vec![
                TaskEnvironmentV1 {
                    environment_id: "isotropic-context".to_owned(),
                    semantics: d4_semantics,
                },
                TaskEnvironmentV1 {
                    environment_id: "rotating-context".to_owned(),
                    semantics: c4_semantics,
                },
                TaskEnvironmentV1 {
                    environment_id: "asymmetric-context".to_owned(),
                    semantics: c1_semantics,
                },
            ],
        };

        let mut atlas = compile_contextual_multi_environment_quotient(
            &problem,
            &action,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(
            atlas
                .environment_quotients
                .iter()
                .map(|entry| {
                    entry
                        .task_preserving_subgroup
                        .retained_ambient_elements
                        .clone()
                })
                .collect::<Vec<_>>(),
            vec![vec![0, 1, 2, 3, 4, 5, 6, 7], vec![0, 1, 2, 3], vec![0]]
        );

        let receipt = verify_contextual_multi_environment_quotient(
            &problem,
            &action,
            &atlas,
            QualificationBudgetV1::default(),
        )
        .unwrap();
        assert_eq!(
            receipt
                .environments
                .iter()
                .map(|entry| (
                    entry.context_id.as_str(),
                    entry.preserving_subgroup_order,
                    entry.orbit_count
                ))
                .collect::<Vec<_>>(),
            vec![
                ("isotropic-context", 8, 3),
                ("rotating-context", 4, 4),
                ("asymmetric-context", 1, 16)
            ]
        );

        atlas.environment_quotients[0].context_id = "rotating-context".to_owned();
        assert_eq!(
            verify_contextual_multi_environment_quotient(
                &problem,
                &action,
                &atlas,
                QualificationBudgetV1::default()
            ),
            Err(QuotientErrorV1::EnvironmentCertificateMismatch)
        );
    }

    #[test]
    fn polynomial_representation_obeys_matrix_composition_exactly() {
        let source = polynomial_map(&C4_POLYNOMIAL);
        assert_eq!(
            transformed_polynomial(&C4_POLYNOMIAL, D4_MATRICES[0]),
            source
        );
        for left in 0..8 {
            for right in 0..8 {
                let right_then_left = multiply_matrix(D4_MATRICES[right], D4_MATRICES[left]);
                let composed = transformed_polynomial(&C4_POLYNOMIAL, right_then_left);

                let right_terms: Vec<PolynomialTermV1> = transformed_polynomial(
                    &C4_POLYNOMIAL,
                    D4_MATRICES[right],
                )
                .into_iter()
                .map(|((x_degree, y_degree), coefficient)| PolynomialTermV1 {
                    x_degree,
                    y_degree,
                    coefficient,
                })
                .collect();
                let sequential =
                    transformed_polynomial(&right_terms, D4_MATRICES[left]);
                assert_eq!(sequential, composed);
            }
        }
    }
}
