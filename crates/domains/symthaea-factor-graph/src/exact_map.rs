//! Bounded exhaustive MAP reference for discrete factor graphs.
//!
//! This module is deliberately a reference/oracle path, not a scalable MAP
//! solver. It enumerates every assignment up to a caller-supplied hard limit
//! and evaluates the product of factor potentials in the log domain.
//!
//! The result is exhaustive over the supplied finite model, while numerical
//! comparisons use f64 log scores. It is not arbitrary-precision symbolic
//! inference. It is useful for small-instance differential tests.

use super::FactorGraph;
use std::fmt;

/// Exhaustive MAP result for a bounded discrete factor graph.
///
/// joint_log_potential is the log of the unnormalized product of all factor
/// potentials at the selected assignment. It is not a normalized log
/// posterior unless the model's partition function is separately known.
#[derive(Debug, Clone, PartialEq)]
pub struct ExhaustiveMapResult {
    /// Lexicographically first assignment attaining the maximum score.
    pub assignment: Vec<usize>,
    /// Maximum unnormalized joint log potential.
    pub joint_log_potential: f64,
    /// Number of assignments attaining the same f64 log score.
    pub maximizer_count: usize,
    /// Number of complete assignments evaluated.
    pub evaluated_assignments: usize,
}

/// Validation, resource-bound, or support failure from exhaustive MAP.
#[derive(Debug, Clone, PartialEq)]
pub enum ExhaustiveMapError {
    ZeroAssignmentLimit,
    ZeroCardinality { variable_index: usize },
    InvalidVariableIdentity { index: usize, id: usize },
    InvalidFactorIdentity { index: usize, id: usize },
    UnknownVariableInFactor { factor_id: usize, variable_id: usize },
    DuplicateVariableInFactor { factor_id: usize, variable_id: usize },
    FactorTableSizeOverflow { factor_id: usize },
    FactorTableLengthMismatch {
        factor_id: usize,
        expected: usize,
        actual: usize,
    },
    InvalidPotential {
        factor_id: usize,
        table_index: usize,
        value: f64,
    },
    AssignmentCountOverflow,
    AssignmentLimitExceeded { required: usize, limit: usize },
    NoPositiveJointAssignment,
    NonFiniteJointLogPotential,
}

impl fmt::Display for ExhaustiveMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroAssignmentLimit => write!(f, "assignment limit must be greater than zero"),
            Self::ZeroCardinality { variable_index } => {
                write!(f, "variable {variable_index} has zero cardinality")
            }
            Self::InvalidVariableIdentity { index, id } => {
                write!(f, "variable at position {index} has non-canonical id {id}")
            }
            Self::InvalidFactorIdentity { index, id } => {
                write!(f, "factor at position {index} has non-canonical id {id}")
            }
            Self::UnknownVariableInFactor { factor_id, variable_id } => write!(
                f,
                "factor {factor_id} references unknown variable {variable_id}"
            ),
            Self::DuplicateVariableInFactor { factor_id, variable_id } => write!(
                f,
                "factor {factor_id} contains variable {variable_id} more than once"
            ),
            Self::FactorTableSizeOverflow { factor_id } => {
                write!(f, "factor {factor_id} table cardinality overflows usize")
            }
            Self::FactorTableLengthMismatch { factor_id, expected, actual } => write!(
                f,
                "factor {factor_id} table has {actual} entries; expected {expected}"
            ),
            Self::InvalidPotential { factor_id, table_index, value } => write!(
                f,
                "factor {factor_id} potential at {table_index} is invalid: {value}"
            ),
            Self::AssignmentCountOverflow => write!(f, "joint assignment count overflows usize"),
            Self::AssignmentLimitExceeded { required, limit } => write!(
                f,
                "exhaustive MAP requires {required} assignments; limit is {limit}"
            ),
            Self::NoPositiveJointAssignment => {
                write!(f, "all assignments have zero joint potential")
            }
            Self::NonFiniteJointLogPotential => {
                write!(f, "joint log potential is not finite")
            }
        }
    }
}

impl std::error::Error for ExhaustiveMapError {}

/// Exhaustively find the joint MAP assignment if the complete state space
/// does not exceed max_assignments.
///
/// Factor tables use the same row-major convention as FactorGraph: the last
/// variable in each factor scope changes fastest. Exact zeros are permitted
/// and eliminate assignments from support. Negative and non-finite potentials
/// are rejected. An all-zero joint model returns NoPositiveJointAssignment;
/// it never fabricates a uniform distribution.
///
/// The function reads only variable cardinalities, factor scopes, and factor
/// tables. It does not trust cached beliefs or the message store.
///
/// Time is O(N * F), where N is the joint assignment count and F is the total
/// factor-scope work per assignment. This is intended for bounded references,
/// not large models.
pub fn exhaustive_map_bounded(
    graph: &FactorGraph,
    max_assignments: usize,
) -> Result<ExhaustiveMapResult, ExhaustiveMapError> {
    if max_assignments == 0 {
        return Err(ExhaustiveMapError::ZeroAssignmentLimit);
    }

    for (index, variable) in graph.variables.iter().enumerate() {
        if variable.id != index {
            return Err(ExhaustiveMapError::InvalidVariableIdentity {
                index,
                id: variable.id,
            });
        }
        if variable.cardinality == 0 {
            return Err(ExhaustiveMapError::ZeroCardinality {
                variable_index: index,
            });
        }
    }

    // Validate the entire factor model before evaluating any assignment.
    for (index, factor) in graph.factors.iter().enumerate() {
        if factor.id != index {
            return Err(ExhaustiveMapError::InvalidFactorIdentity {
                index,
                id: factor.id,
            });
        }
        let mut expected = 1usize;
        let mut seen = std::collections::BTreeSet::new();
        for &variable_id in &factor.variables {
            let Some(variable) = graph.variables.get(variable_id) else {
                return Err(ExhaustiveMapError::UnknownVariableInFactor {
                    factor_id: factor.id,
                    variable_id,
                });
            };
            if !seen.insert(variable_id) {
                return Err(ExhaustiveMapError::DuplicateVariableInFactor {
                    factor_id: factor.id,
                    variable_id,
                });
            }
            expected = expected.checked_mul(variable.cardinality).ok_or(
                ExhaustiveMapError::FactorTableSizeOverflow {
                    factor_id: factor.id,
                },
            )?;
        }
        if factor.table.len() != expected {
            return Err(ExhaustiveMapError::FactorTableLengthMismatch {
                factor_id: factor.id,
                expected,
                actual: factor.table.len(),
            });
        }
        for (table_index, &value) in factor.table.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(ExhaustiveMapError::InvalidPotential {
                    factor_id: factor.id,
                    table_index,
                    value,
                });
            }
        }
    }

    let assignment_count = graph
        .variables
        .iter()
        .try_fold(1usize, |count, variable| {
            count.checked_mul(variable.cardinality)
        })
        .ok_or(ExhaustiveMapError::AssignmentCountOverflow)?;

    if assignment_count > max_assignments {
        return Err(ExhaustiveMapError::AssignmentLimitExceeded {
            required: assignment_count,
            limit: max_assignments,
        });
    }

    let cardinalities: Vec<usize> = graph
        .variables
        .iter()
        .map(|variable| variable.cardinality)
        .collect();

    let mut best_assignment: Option<Vec<usize>> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut maximizer_count = 0usize;

    for flat_index in 0..assignment_count {
        let assignment = decode_assignment(flat_index, &cardinalities);
        let mut log_terms = Vec::with_capacity(graph.factors.len());
        let mut supported = true;

        for factor in &graph.factors {
            let table_index = factor
                .variables
                .iter()
                .fold(0usize, |index, &variable_id| {
                    index * cardinalities[variable_id] + assignment[variable_id]
                });
            let potential = factor.table[table_index];

            if potential == 0.0 {
                supported = false;
                break;
            }
            log_terms.push(potential.ln());
        }

        if !supported {
            continue;
        }

        // Sort to make the summation order independent of factor insertion
        // order, then use Neumaier compensation to reduce accumulation error.
        log_terms.sort_by(f64::total_cmp);
        let score = compensated_sum(&log_terms);
        if !score.is_finite() {
            return Err(ExhaustiveMapError::NonFiniteJointLogPotential);
        }

        match score.total_cmp(&best_score) {
            std::cmp::Ordering::Greater => {
                best_score = score;
                best_assignment = Some(assignment);
                maximizer_count = 1;
            }
            std::cmp::Ordering::Equal => {
                maximizer_count += 1;
            }
            std::cmp::Ordering::Less => {}
        }
    }

    let assignment = best_assignment.ok_or(ExhaustiveMapError::NoPositiveJointAssignment)?;
    Ok(ExhaustiveMapResult {
        assignment,
        joint_log_potential: best_score,
        maximizer_count,
        evaluated_assignments: assignment_count,
    })
}

fn decode_assignment(mut flat_index: usize, cardinalities: &[usize]) -> Vec<usize> {
    let mut assignment = vec![0usize; cardinalities.len()];
    for position in (0..cardinalities.len()).rev() {
        assignment[position] = flat_index % cardinalities[position];
        flat_index /= cardinalities[position];
    }
    assignment
}

fn compensated_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for &value in values {
        let next = sum + value;
        if sum.abs() >= value.abs() {
            correction += (sum - next) + value;
        } else {
            correction += (value - next) + sum;
        }
        sum = next;
    }
    sum + correction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_map_counterexample_recovers_a_valid_global_assignment() {
        // A == B and B != C. Exactly two assignments have positive potential:
        // [0, 0, 1] and [1, 1, 0]. Every variable's max-marginal is tied.
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let c = graph.add_variable("C", 2);
        graph.add_factor(&[a, b], vec![1.0, 0.0, 0.0, 1.0]);
        graph.add_factor(&[b, c], vec![0.0, 1.0, 1.0, 0.0]);

        let result = exhaustive_map_bounded(&graph, 8).unwrap();
        assert_eq!(result.assignment, vec![0, 0, 1]);
        assert_eq!(result.maximizer_count, 2);
        assert_eq!(result.evaluated_assignments, 8);
        assert_eq!(result.joint_log_potential, 0.0);
    }

    #[test]
    fn mixed_cardinality_factor_uses_last_scope_variable_as_fastest() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 3);
        // Scope [A,B]; B varies fastest. The unique maximum is A=1, B=2.
        graph.add_factor(&[a, b], vec![1.0, 2.0, 3.0, 4.0, 5.0, 9.0]);

        let result = exhaustive_map_bounded(&graph, 6).unwrap();
        assert_eq!(result.assignment, vec![1, 2]);
        assert_eq!(result.maximizer_count, 1);
        assert!((result.joint_log_potential - 9.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn factor_insertion_order_does_not_change_the_map_result() {
        fn build(reverse: bool) -> FactorGraph {
            let mut graph = FactorGraph::new();
            let a = graph.add_variable("A", 2);
            let b = graph.add_variable("B", 2);
            let left = (vec![a], vec![0.3, 0.7]);
            let right = (vec![b], vec![0.4, 0.6]);
            let factors = if reverse { vec![right, left] } else { vec![left, right] };
            for (scope, table) in factors {
                graph.add_factor(&scope, table);
            }
            graph
        }

        let forward = exhaustive_map_bounded(&build(false), 4).unwrap();
        let reverse = exhaustive_map_bounded(&build(true), 4).unwrap();
        assert_eq!(forward.assignment, reverse.assignment);
        assert_eq!(forward.joint_log_potential, reverse.joint_log_potential);
        assert_eq!(forward.maximizer_count, reverse.maximizer_count);
    }

    #[test]
    fn zero_joint_support_is_not_replaced_with_a_uniform_result() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        graph.add_factor(&[a], vec![0.0, 0.0]);

        assert_eq!(
            exhaustive_map_bounded(&graph, 2),
            Err(ExhaustiveMapError::NoPositiveJointAssignment)
        );
    }

    #[test]
    fn malformed_non_finite_and_negative_potentials_are_rejected() {
        for invalid in [f64::NAN, f64::INFINITY, -0.1] {
            let mut graph = FactorGraph::new();
            let a = graph.add_variable("A", 2);
            graph.add_factor(&[a], vec![1.0, invalid]);

            assert!(matches!(
                exhaustive_map_bounded(&graph, 2),
                Err(ExhaustiveMapError::InvalidPotential { .. })
            ));
        }
    }

    #[test]
    fn exhaustive_search_limit_is_a_hard_failure() {
        let mut graph = FactorGraph::new();
        graph.add_variable("A", 2);
        graph.add_variable("B", 3);

        assert_eq!(
            exhaustive_map_bounded(&graph, 5),
            Err(ExhaustiveMapError::AssignmentLimitExceeded {
                required: 6,
                limit: 5
            })
        );
    }

    #[test]
    fn disconnected_variables_count_all_global_ties() {
        let mut graph = FactorGraph::new();
        graph.add_variable("A", 2);
        graph.add_variable("B", 3);

        let result = exhaustive_map_bounded(&graph, 6).unwrap();
        assert_eq!(result.assignment, vec![0, 0]);
        assert_eq!(result.maximizer_count, 6);
        assert_eq!(result.joint_log_potential, 0.0);
    }

    #[test]
    fn product_underflow_does_not_destroy_the_map_ordering() {
        let mut graph = FactorGraph::new();
        for index in 0..20 {
            let variable = graph.add_variable(&format!("X{index}"), 2);
            graph.add_factor(&[variable], vec![1.0e-200, 2.0e-200]);
        }

        let result = exhaustive_map_bounded(&graph, 1usize << 20).unwrap();
        assert_eq!(result.assignment, vec![1; 20]);
        assert!(result.joint_log_potential.is_finite());
    }
}
