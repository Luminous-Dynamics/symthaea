//! Bounded exhaustive posterior-marginal reference for discrete factor graphs.
//!
//! This is a small-model oracle for differential testing, not a scalable
//! inference algorithm. It enumerates the complete assignment space under a
//! caller-provided hard limit and computes normalized marginals and log Z
//! using log-domain factor scores.
//!
//! The arithmetic is f64. Extremely small relative assignment weights can
//! underflow during the final exponentiation; the result reports how many
//! relative weights underflowed instead of hiding that numerical limitation.

use super::exact_map::{exhaustive_map_bounded, ExhaustiveMapError};
use super::{BPConfig, FactorGraph};

/// Exhaustive posterior marginals for a finite factor graph.
#[derive(Debug, Clone, PartialEq)]
pub struct ExhaustiveMarginalResult {
    /// Posterior marginal for each variable, in graph variable order.
    pub marginals: Vec<Vec<f64>>,
    /// Log partition function: log(sum_x product_f f(x_scope)).
    pub log_partition_function: f64,
    /// Maximum unnormalized joint log potential.
    pub max_log_joint_potential: f64,
    /// Number of assignments with strictly positive joint potential.
    pub supported_assignments: usize,
    /// Number of complete assignments evaluated.
    pub evaluated_assignments: usize,
    /// Positive relative weights that underflowed to f64 zero.
    pub underflowed_relative_weights: usize,
}

/// Compute exact-enumeration posterior marginals within a strict search bound.
///
/// This first calls the bounded exhaustive MAP reference to validate the
/// model, enforce the assignment limit, and obtain the maximum log score.
/// It then enumerates the same state space to compute the log partition
/// function and each posterior marginal.
///
/// Exact zero potentials remove assignments from support. If no assignment
/// has positive joint potential, the function returns
/// ExhaustiveMapError::NoPositiveJointAssignment. Negative/non-finite values,
/// malformed scopes, and oversized state spaces are rejected by the shared
/// MAP validator.
///
/// This is exhaustive over the finite assignment space but uses f64 log
/// arithmetic and exp for normalized weights. Underflowed relative weights
/// are counted and reported in the result.
pub fn exhaustive_marginals_bounded(
    graph: &FactorGraph,
    max_assignments: usize,
) -> Result<ExhaustiveMarginalResult, ExhaustiveMapError> {
    let map = exhaustive_map_bounded(graph, max_assignments)?;
    let cardinalities: Vec<usize> = graph
        .variables
        .iter()
        .map(|variable| variable.cardinality)
        .collect();

    let mut total_mass = 0.0;
    let mut total_correction = 0.0;
    let mut marginal_mass = cardinalities
        .iter()
        .map(|&cardinality| vec![0.0; cardinality])
        .collect::<Vec<_>>();
    let mut marginal_correction = marginal_mass.clone();
    let mut supported_assignments = 0usize;
    let mut underflowed_relative_weights = 0usize;

    for flat_index in 0..map.evaluated_assignments {
        let assignment = decode_assignment(flat_index, &cardinalities);
        let Some(log_score) = joint_log_potential(graph, &assignment) else {
            continue;
        };
        supported_assignments += 1;

        let relative_log_weight = log_score - map.joint_log_potential;
        let weight = relative_log_weight.exp();
        if weight == 0.0 {
            underflowed_relative_weights += 1;
            continue;
        }

        neumaier_add(&mut total_mass, &mut total_correction, weight);
        for (variable_index, &state) in assignment.iter().enumerate() {
            neumaier_add(
                &mut marginal_mass[variable_index][state],
                &mut marginal_correction[variable_index][state],
                weight,
            );
        }
    }

    let normalizer = total_mass + total_correction;
    if !(normalizer.is_finite() && normalizer > 0.0) {
        return Err(ExhaustiveMapError::NonFiniteJointLogPotential);
    }

    let marginals = marginal_mass
        .into_iter()
        .zip(marginal_correction)
        .map(|(masses, corrections)| {
            masses
                .into_iter()
                .zip(corrections)
                .map(|(mass, correction)| (mass + correction) / normalizer)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let log_partition_function = map.joint_log_potential + normalizer.ln();
    if !log_partition_function.is_finite() {
        return Err(ExhaustiveMapError::NonFiniteJointLogPotential);
    }

    Ok(ExhaustiveMarginalResult {
        marginals,
        log_partition_function,
        max_log_joint_potential: map.joint_log_potential,
        supported_assignments,
        evaluated_assignments: map.evaluated_assignments,
        underflowed_relative_weights,
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

/// Returns None exactly when one factor assigns zero potential.
fn joint_log_potential(graph: &FactorGraph, assignment: &[usize]) -> Option<f64> {
    let mut log_terms = Vec::with_capacity(graph.factors.len());
    for factor in &graph.factors {
        let table_index = factor
            .variables
            .iter()
            .fold(0usize, |index, &variable_id| {
                index * graph.variables[variable_id].cardinality + assignment[variable_id]
            });
        let potential = factor.table[table_index];
        if potential == 0.0 {
            return None;
        }
        log_terms.push(potential.ln());
    }
    log_terms.sort_by(f64::total_cmp);
    Some(compensated_sum(&log_terms))
}

fn neumaier_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let next = *sum + value;
    if sum.abs() >= value.abs() {
        *correction += (*sum - next) + value;
    } else {
        *correction += (value - next) + *sum;
    }
    *sum = next;
}

fn compensated_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for &value in values {
        neumaier_add(&mut sum, &mut correction, value);
    }
    sum + correction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_graph() -> FactorGraph {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let c = graph.add_variable("C", 2);
        graph.add_factor(&[a, b], vec![0.4, 0.1, 0.2, 0.3]);
        graph.add_factor(&[b, c], vec![0.5, 0.5, 0.3, 0.7]);
        graph
    }

    #[test]
    fn tree_sum_product_matches_exhaustive_posterior_marginals() {
        let graph = tree_graph();
        let exact = exhaustive_marginals_bounded(&graph, 8).unwrap();
        let mut bp_graph = graph.clone();
        let bp = bp_graph.run_bp(&BPConfig::default());

        assert!(bp.converged);
        assert_eq!(exact.evaluated_assignments, 8);
        assert_eq!(exact.supported_assignments, 8);
        for (variable_index, expected) in exact.marginals.iter().enumerate() {
            for (state, expected_probability) in expected.iter().enumerate() {
                assert!(
                    (bp.beliefs[variable_index][state] - expected_probability).abs() < 1.0e-10,
                    "variable {variable_index}, state {state}: BP={}, exact={}",
                    bp.beliefs[variable_index][state],
                    expected_probability
                );
            }
        }
    }

    #[test]
    fn equality_constraint_produces_correct_zero_potential_support() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        graph.add_factor(&[a, b], vec![1.0, 0.0, 0.0, 1.0]);

        let exact = exhaustive_marginals_bounded(&graph, 4).unwrap();
        assert_eq!(exact.supported_assignments, 2);
        assert_eq!(exact.marginals, vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
        assert!((exact.log_partition_function - 2.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn disconnected_variables_have_uniform_marginals_and_known_log_partition() {
        let mut graph = FactorGraph::new();
        graph.add_variable("A", 2);
        graph.add_variable("B", 3);

        let exact = exhaustive_marginals_bounded(&graph, 6).unwrap();
        assert_eq!(exact.marginals, vec![vec![0.5, 0.5], vec![1.0 / 3.0; 3]]);
        assert!((exact.log_partition_function - 6.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn extreme_factor_products_keep_finite_log_partition_and_report_underflow() {
        let mut graph = FactorGraph::new();
        for index in 0..20 {
            let variable = graph.add_variable(&format!("X{index}"), 2);
            graph.add_factor(&[variable], vec![1.0e-200, 2.0e-200]);
        }

        let exact = exhaustive_marginals_bounded(&graph, 1usize << 20).unwrap();
        assert!(exact.log_partition_function.is_finite());
        assert!(exact.underflowed_relative_weights > 0);
        for marginal in &exact.marginals {
            assert!((marginal[0] - 1.0 / 3.0).abs() < 1.0e-12);
            assert!((marginal[1] - 2.0 / 3.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn zero_joint_mass_is_a_typed_failure() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        graph.add_factor(&[a], vec![0.0, 0.0]);
        assert_eq!(
            exhaustive_marginals_bounded(&graph, 2),
            Err(ExhaustiveMapError::NoPositiveJointAssignment)
        );
    }
}
