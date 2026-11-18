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
    CompileOutcomeV1, FiniteActionV1, FiniteGroupV1, CandidateSemanticsV1,
    QualificationBudgetV1, SearchProblemV1, compile_quotient, lift_representative_solution,
    verify_certificate,
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(actual.iter().map(Vec::len).collect::<Vec<_>>(), vec![4, 4, 8]);
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
    fn polynomial_representation_obeys_matrix_composition_exactly() {
        let source = polynomial_map(&C4_POLYNOMIAL);
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
        assert_eq!(source, polynomial_map(&C4_POLYNOMIAL));
    }
}
