//! Fail-closed log-domain sum-product belief propagation.
//!
//! This module is a non-breaking checked alternative to the legacy methods in
//! the parent module. It validates the public graph representation, ignores
//! any caller-supplied cached messages, computes messages in log space, and
//! returns typed errors when support is empty. It deliberately does not return
//! a Bethe free-energy value or a joint MAP assignment.
//!
//! An undamped, converged result on an acyclic factor graph is algorithmically
//! exact for the declared finite factorization (with ordinary f64 rounding).
//! Damping can preserve mass on states that have exact-zero support and is
//! therefore labelled approximate even on a forest. A converged cyclic graph
//! is always labelled approximate.

use super::{BPConfig, FactorGraph, MessageSchedule};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

/// Input validation, numerical, support, or execution failure.
#[derive(Clone, Debug, PartialEq)]
pub enum CheckedBPError {
    ZeroCardinality { variable: usize },
    NonCanonicalVariableId { index: usize, id: usize },
    NonCanonicalFactorId { index: usize, id: usize },
    InvalidVariableReference { factor: usize, variable: usize },
    DuplicateVariableInScope { factor: usize, variable: usize },
    TableSizeOverflow { factor: usize },
    TableLengthMismatch { factor: usize, expected: usize, actual: usize },
    InvalidAdjacency { variable: usize },
    InvalidFactorAdjacency { factor: usize },
    InvalidPotential { factor: usize, table_index: usize, value: f64 },
    AllZeroFactor { factor: usize },
    InvalidIterationLimit,
    InvalidTolerance(f64),
    InvalidDamping(f64),
    ZeroSupport { context: &'static str },
    NonFiniteMessage { context: &'static str },
}

impl fmt::Display for CheckedBPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCardinality { variable } => write!(f, "variable {variable} has zero cardinality"),
            Self::NonCanonicalVariableId { index, id } => write!(f, "variable at index {index} has ID {id}"),
            Self::NonCanonicalFactorId { index, id } => write!(f, "factor at index {index} has ID {id}"),
            Self::InvalidVariableReference { factor, variable } => write!(f, "factor {factor} references unknown variable {variable}"),
            Self::DuplicateVariableInScope { factor, variable } => write!(f, "factor {factor} repeats variable {variable}"),
            Self::TableSizeOverflow { factor } => write!(f, "factor {factor} table size overflows usize"),
            Self::TableLengthMismatch { factor, expected, actual } => write!(f, "factor {factor} has {actual} entries; expected {expected}"),
            Self::InvalidAdjacency { variable } => write!(f, "variable {variable} adjacency does not match factor scopes"),
            Self::InvalidFactorAdjacency { factor } => write!(f, "factor {factor} adjacency differs from its scope"),
            Self::InvalidPotential { factor, table_index, value } => write!(f, "factor {factor} potential {table_index} is invalid: {value}"),
            Self::AllZeroFactor { factor } => write!(f, "factor {factor} has no positive potential"),
            Self::InvalidIterationLimit => write!(f, "max_iterations must be greater than zero"),
            Self::InvalidTolerance(value) => write!(f, "tolerance must be finite and positive, got {value}"),
            Self::InvalidDamping(value) => write!(f, "damping must be finite and in (0, 1], got {value}"),
            Self::ZeroSupport { context } => write!(f, "zero probability support while computing {context}"),
            Self::NonFiniteMessage { context } => write!(f, "non-finite message while computing {context}"),
        }
    }
}
impl Error for CheckedBPError {}

/// Structure of the validated bipartite variable/factor graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphStructure {
    Forest,
    Loopy,
}

/// Inference guarantee after applying the configured update schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceStatus {
    /// Undamped sum-product converged on a forest. Exact algorithmically;
    /// returned f64 values still have ordinary floating-point rounding.
    ExactForest,
    /// A damped forest run converged to the tolerance. This is an iterative
    /// approximation because damping can preserve mass on exact-zero states.
    ApproximateDampedForest,
    /// A loopy BP fixed point converged. It remains approximate.
    ApproximateLoopy,
    /// The iteration limit was reached before the residual tolerance.
    NotConverged,
}

/// Checked sum-product result.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBPResult {
    /// Normalized state probabilities. Extremely small values may underflow
    /// when exponentiating; use log_beliefs to retain the full f64 log range.
    pub beliefs: Vec<Vec<f64>>,
    /// Normalized log probabilities, preserving states whose probabilities
    /// underflow when converted into ordinary f64 probability space.
    pub log_beliefs: Vec<Vec<f64>>,
    pub converged: bool,
    pub iterations: usize,
    /// Maximum absolute probability-message change on the final iteration.
    pub max_message_delta: f64,
    /// Maximum absolute finite log-message change. Infinity indicates a
    /// support change (finite log mass becoming exact zero or vice versa).
    pub max_log_message_delta: f64,
    pub graph_structure: GraphStructure,
    pub inference_status: InferenceStatus,
}

/// Run validated sum-product belief propagation in the log domain.
///
/// The graph's public factor/adjacency fields are validated before use. Existing
/// messages stored on the graph are ignored; fresh normalized log messages are
/// initialized for every call. Zero support is never replaced by a uniform
/// vector.
///
/// Parallel updates are synchronous. Sequential updates use each newly
/// computed message immediately. Damping mixes normalized message
/// distributions in probability space, implemented stably with log-add-exp.
pub fn run_sum_product(
    graph: &FactorGraph,
    config: &BPConfig,
) -> Result<CheckedBPResult, CheckedBPError> {
    validate_graph(graph)?;
    validate_config(config)?;

    let n_variables = graph.variables.len();
    let structure = graph_structure(graph);
    let mut messages: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

    for factor in &graph.factors {
        for &variable in &factor.variables {
            let cardinality = graph.variables[variable].cardinality;
            let log_uniform = -(cardinality as f64).ln();
            let initial = vec![log_uniform; cardinality];
            messages.insert(var_to_factor_key(n_variables, variable, factor.id), initial.clone());
            messages.insert(factor_to_var_key(n_variables, factor.id, variable), initial);
        }
    }

    let mut converged = false;
    let mut iterations = 0usize;
    let mut final_delta = f64::INFINITY;
    let mut final_log_delta = f64::INFINITY;

    for iteration in 0..config.max_iterations {
        iterations = iteration + 1;
        let mut max_delta = 0.0f64;
        let mut max_log_delta = 0.0f64;

        match config.schedule {
            MessageSchedule::Parallel => {
                let mut pending = Vec::with_capacity(messages.len());

                // All variable-to-factor messages are computed from the old map.
                for variable in 0..n_variables {
                    for &factor in &graph.variables[variable].neighbors {
                        let key = var_to_factor_key(n_variables, variable, factor);
                        let new_message = compute_var_to_factor(graph, &messages, n_variables, variable, factor)?;
                        let old_message = messages.get(&key).expect("validated message initialization");
                        let damped = damp_message(&new_message, old_message, config.damping)?;
                        max_delta = max_delta.max(probability_delta(&damped, old_message));
                        max_log_delta = max_log_delta.max(log_message_delta(&damped, old_message));
                        pending.push((key, damped));
                    }
                }

                // Factor-to-variable messages are also computed from the old map,
                // preserving a genuinely parallel schedule.
                for factor in 0..graph.factors.len() {
                    for &variable in &graph.factors[factor].variables {
                        let key = factor_to_var_key(n_variables, factor, variable);
                        let new_message = compute_factor_to_var(graph, &messages, n_variables, factor, variable)?;
                        let old_message = messages.get(&key).expect("validated message initialization");
                        let damped = damp_message(&new_message, old_message, config.damping)?;
                        max_delta = max_delta.max(probability_delta(&damped, old_message));
                        max_log_delta = max_log_delta.max(log_message_delta(&damped, old_message));
                        pending.push((key, damped));
                    }
                }

                for (key, message) in pending {
                    messages.insert(key, message);
                }
            }
            MessageSchedule::Sequential => {
                for variable in 0..n_variables {
                    for &factor in &graph.variables[variable].neighbors {
                        let key = var_to_factor_key(n_variables, variable, factor);
                        let new_message = compute_var_to_factor(graph, &messages, n_variables, variable, factor)?;
                        let old_message = messages.get(&key).expect("validated message initialization");
                        let damped = damp_message(&new_message, old_message, config.damping)?;
                        max_delta = max_delta.max(probability_delta(&damped, old_message));
                        max_log_delta = max_log_delta.max(log_message_delta(&damped, old_message));
                        messages.insert(key, damped);
                    }
                }

                for factor in 0..graph.factors.len() {
                    for &variable in &graph.factors[factor].variables {
                        let key = factor_to_var_key(n_variables, factor, variable);
                        let new_message = compute_factor_to_var(graph, &messages, n_variables, factor, variable)?;
                        let old_message = messages.get(&key).expect("validated message initialization");
                        let damped = damp_message(&new_message, old_message, config.damping)?;
                        max_delta = max_delta.max(probability_delta(&damped, old_message));
                        max_log_delta = max_log_delta.max(log_message_delta(&damped, old_message));
                        messages.insert(key, damped);
                    }
                }
            }
        }

        final_delta = max_delta;
        final_log_delta = max_log_delta;
        if max_delta < config.tolerance && max_log_delta < config.tolerance {
            converged = true;
            break;
        }
    }

    let mut beliefs = Vec::with_capacity(n_variables);
    let mut log_beliefs = Vec::with_capacity(n_variables);
    for variable in 0..n_variables {
        let cardinality = graph.variables[variable].cardinality;
        let mut log_belief = vec![0.0; cardinality];
        for &factor in &graph.variables[variable].neighbors {
            let message = messages
                .get(&factor_to_var_key(n_variables, factor, variable))
                .expect("validated message initialization");
            for state in 0..cardinality {
                log_belief[state] += message[state];
            }
        }
        normalize_log(&mut log_belief, "variable belief")?;
        beliefs.push(log_belief.iter().map(|value| value.exp()).collect());
        log_beliefs.push(log_belief);
    }

    let inference_status = if !converged {
        InferenceStatus::NotConverged
    } else {
        match (structure, config.damping == 1.0) {
            (GraphStructure::Forest, true) => InferenceStatus::ExactForest,
            (GraphStructure::Forest, false) => InferenceStatus::ApproximateDampedForest,
            (GraphStructure::Loopy, _) => InferenceStatus::ApproximateLoopy,
        }
    };

    Ok(CheckedBPResult {
        beliefs,
        log_beliefs,
        converged,
        iterations,
        max_message_delta: final_delta,
        max_log_message_delta: final_log_delta,
        graph_structure: structure,
        inference_status,
    })
}

fn validate_config(config: &BPConfig) -> Result<(), CheckedBPError> {
    if config.max_iterations == 0 {
        return Err(CheckedBPError::InvalidIterationLimit);
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(CheckedBPError::InvalidTolerance(config.tolerance));
    }
    if !(config.damping.is_finite() && config.damping > 0.0 && config.damping <= 1.0) {
        return Err(CheckedBPError::InvalidDamping(config.damping));
    }
    Ok(())
}

fn validate_graph(graph: &FactorGraph) -> Result<(), CheckedBPError> {
    let mut expected_adjacency = vec![Vec::new(); graph.variables.len()];

    for (index, variable) in graph.variables.iter().enumerate() {
        if variable.id != index {
            return Err(CheckedBPError::NonCanonicalVariableId { index, id: variable.id });
        }
        if variable.cardinality == 0 {
            return Err(CheckedBPError::ZeroCardinality { variable: index });
        }
    }

    for (index, factor) in graph.factors.iter().enumerate() {
        if factor.id != index {
            return Err(CheckedBPError::NonCanonicalFactorId { index, id: factor.id });
        }
        if factor.neighbors != factor.variables {
            return Err(CheckedBPError::InvalidFactorAdjacency { factor: index });
        }

        let mut expected = 1usize;
        let mut seen = HashSet::new();
        for &variable in &factor.variables {
            let Some(node) = graph.variables.get(variable) else {
                return Err(CheckedBPError::InvalidVariableReference { factor: index, variable });
            };
            if !seen.insert(variable) {
                return Err(CheckedBPError::DuplicateVariableInScope { factor: index, variable });
            }
            expected = expected.checked_mul(node.cardinality)
                .ok_or(CheckedBPError::TableSizeOverflow { factor: index })?;
            expected_adjacency[variable].push(index);
        }
        if factor.table.len() != expected {
            return Err(CheckedBPError::TableLengthMismatch {
                factor: index,
                expected,
                actual: factor.table.len(),
            });
        }

        let mut has_positive = false;
        for (table_index, &potential) in factor.table.iter().enumerate() {
            if !potential.is_finite() || potential < 0.0 {
                return Err(CheckedBPError::InvalidPotential {
                    factor: index,
                    table_index,
                    value: potential,
                });
            }
            has_positive |= potential > 0.0;
        }
        if !has_positive {
            return Err(CheckedBPError::AllZeroFactor { factor: index });
        }
    }

    for (index, variable) in graph.variables.iter().enumerate() {
        if variable.neighbors != expected_adjacency[index] {
            return Err(CheckedBPError::InvalidAdjacency { variable: index });
        }
    }
    Ok(())
}

fn graph_structure(graph: &FactorGraph) -> GraphStructure {
    let n_variables = graph.variables.len();
    let total_nodes = n_variables + graph.factors.len();
    let mut parent: Vec<usize> = (0..total_nodes).collect();
    let mut rank = vec![0usize; total_nodes];

    fn find(parent: &mut [usize], node: usize) -> usize {
        if parent[node] != node {
            let next = parent[node];
            let root = find(parent, next);
            parent[node] = root;
        }
        parent[node]
    }

    for factor in &graph.factors {
        let factor_node = n_variables + factor.id;
        for &variable in &factor.variables {
            let left = find(&mut parent, variable);
            let right = find(&mut parent, factor_node);
            if left == right {
                return GraphStructure::Loopy;
            }
            if rank[left] < rank[right] {
                parent[left] = right;
            } else if rank[left] > rank[right] {
                parent[right] = left;
            } else {
                parent[right] = left;
                rank[left] += 1;
            }
        }
    }
    GraphStructure::Forest
}

fn var_to_factor_key(n_variables: usize, variable: usize, factor: usize) -> (usize, usize) {
    (variable, n_variables + factor)
}

fn factor_to_var_key(n_variables: usize, factor: usize, variable: usize) -> (usize, usize) {
    (n_variables + factor, variable)
}

fn compute_var_to_factor(
    graph: &FactorGraph,
    messages: &HashMap<(usize, usize), Vec<f64>>,
    n_variables: usize,
    variable: usize,
    excluded_factor: usize,
) -> Result<Vec<f64>, CheckedBPError> {
    let cardinality = graph.variables[variable].cardinality;
    let mut result = vec![0.0; cardinality];
    for &factor in &graph.variables[variable].neighbors {
        if factor == excluded_factor {
            continue;
        }
        let incoming = messages
            .get(&factor_to_var_key(n_variables, factor, variable))
            .expect("validated message initialization");
        for state in 0..cardinality {
            result[state] += incoming[state];
        }
    }
    normalize_log(&mut result, "variable-to-factor message")?;
    Ok(result)
}

fn compute_factor_to_var(
    graph: &FactorGraph,
    messages: &HashMap<(usize, usize), Vec<f64>>,
    n_variables: usize,
    factor_index: usize,
    target_variable: usize,
) -> Result<Vec<f64>, CheckedBPError> {
    let factor = &graph.factors[factor_index];
    let cards: Vec<usize> = factor.variables.iter()
        .map(|&variable| graph.variables[variable].cardinality)
        .collect();
    let target_position = factor.variables.iter()
        .position(|&variable| variable == target_variable)
        .expect("validated factor scope");
    let total = cards.iter().try_fold(1usize, |acc, &card| acc.checked_mul(card))
        .ok_or(CheckedBPError::TableSizeOverflow { factor: factor_index })?;
    let mut result = vec![f64::NEG_INFINITY; cards[target_position]];

    for flat_index in 0..total {
        let assignment = decode_assignment(flat_index, &cards);
        let potential = factor.table[flat_index];
        if potential == 0.0 {
            continue;
        }

        let mut term = potential.ln();
        for (position, &variable) in factor.variables.iter().enumerate() {
            if position == target_position {
                continue;
            }
            let message = messages
                .get(&var_to_factor_key(n_variables, variable, factor_index))
                .expect("validated message initialization");
            term += message[assignment[position]];
        }
        if !term.is_finite() && term != f64::NEG_INFINITY {
            return Err(CheckedBPError::NonFiniteMessage { context: "factor-to-variable term" });
        }

        let state = assignment[target_position];
        result[state] = log_add_exp(result[state], term);
    }

    normalize_log(&mut result, "factor-to-variable message")?;
    Ok(result)
}

fn decode_assignment(mut flat_index: usize, cardinalities: &[usize]) -> Vec<usize> {
    let mut result = vec![0usize; cardinalities.len()];
    for position in (0..cardinalities.len()).rev() {
        result[position] = flat_index % cardinalities[position];
        flat_index /= cardinalities[position];
    }
    result
}

fn normalize_log(values: &mut [f64], context: &'static str) -> Result<(), CheckedBPError> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return Err(CheckedBPError::ZeroSupport { context });
    }
    if !maximum.is_finite() {
        return Err(CheckedBPError::NonFiniteMessage { context });
    }
    let total = values.iter().map(|value| (value - maximum).exp()).sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return Err(CheckedBPError::NonFiniteMessage { context });
    }
    let log_normalizer = maximum + total.ln();
    if !log_normalizer.is_finite() {
        return Err(CheckedBPError::NonFiniteMessage { context });
    }
    for value in values {
        *value -= log_normalizer;
    }
    Ok(())
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn damp_message(
    new_message: &[f64],
    old_message: &[f64],
    damping: f64,
) -> Result<Vec<f64>, CheckedBPError> {
    if damping == 1.0 {
        return Ok(new_message.to_vec());
    }
    let log_new_weight = damping.ln();
    let log_old_weight = (1.0 - damping).ln();
    let mut result = new_message.iter().zip(old_message)
        .map(|(&new, &old)| {
            log_add_exp(log_new_weight + new, log_old_weight + old)
        })
        .collect::<Vec<_>>();
    normalize_log(&mut result, "damped message")?;
    Ok(result)
}

fn probability_delta(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right)
        .map(|(&a, &b)| (a.exp() - b.exp()).abs())
        .fold(0.0, f64::max)
}

fn log_message_delta(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right)
        .map(|(&a, &b)| {
            if a == b {
                0.0
            } else if !a.is_finite() || !b.is_finite() {
                f64::INFINITY
            } else {
                (a - b).abs()
            }
        })
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_node_tree() -> FactorGraph {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        graph.add_factor(&[a, b], vec![0.3, 0.1, 0.2, 0.4]);
        graph
    }

    #[test]
    fn forest_marginals_match_direct_enumeration() {
        let graph = two_node_tree();
        let result = run_sum_product(&graph, &BPConfig::default()).unwrap();
        assert!(result.converged);
        assert_eq!(result.graph_structure, GraphStructure::Forest);
        assert_eq!(result.inference_status, InferenceStatus::ExactForest);
        assert!((result.beliefs[0][0] - 0.4).abs() < 1.0e-10);
        assert!((result.beliefs[0][1] - 0.6).abs() < 1.0e-10);
        assert!((result.beliefs[1][0] - 0.5).abs() < 1.0e-10);
        assert!((result.beliefs[1][1] - 0.5).abs() < 1.0e-10);
    }

    #[test]
    fn three_variable_forest_matches_joint_enumeration() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let c = graph.add_variable("C", 2);
        graph.add_factor(&[a, b], vec![0.4, 0.1, 0.2, 0.3]);
        graph.add_factor(&[b, c], vec![0.5, 0.5, 0.3, 0.7]);

        let result = run_sum_product(&graph, &BPConfig::default()).unwrap();
        assert!(result.converged);
        assert_eq!(result.inference_status, InferenceStatus::ExactForest);

        let f1 = [[0.4, 0.1], [0.2, 0.3]];
        let f2 = [[0.5, 0.5], [0.3, 0.7]];
        let mut z = 0.0;
        let mut exact = [[0.0; 2]; 3];
        for av in 0..2 {
            for bv in 0..2 {
                for cv in 0..2 {
                    let weight = f1[av][bv] * f2[bv][cv];
                    z += weight;
                    exact[0][av] += weight;
                    exact[1][bv] += weight;
                    exact[2][cv] += weight;
                }
            }
        }
        for variable in 0..3 {
            for state in 0..2 {
                let expected = exact[variable][state] / z;
                assert!(
                    (result.beliefs[variable][state] - expected).abs() < 1.0e-10,
                    "variable {variable}, state {state}: checked={}, enumeration={expected}",
                    result.beliefs[variable][state]
                );
            }
        }
    }

    #[test]
    fn disconnected_and_isolated_variables_are_uniform() {
        let mut graph = FactorGraph::new();
        graph.add_variable("A", 3);
        graph.add_variable("B", 2);
        graph.add_factor(&[], vec![2.0]);
        let result = run_sum_product(&graph, &BPConfig::default()).unwrap();
        assert_eq!(result.graph_structure, GraphStructure::Forest);
        assert_eq!(result.beliefs[0], vec![1.0 / 3.0; 3]);
        assert_eq!(result.beliefs[1], vec![0.5; 2]);
    }

    #[test]
    fn contradictory_unary_evidence_returns_zero_support() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        graph.add_factor(&[a], vec![1.0, 0.0]);
        graph.add_factor(&[a], vec![0.0, 1.0]);
        assert!(matches!(
            run_sum_product(&graph, &BPConfig::default()),
            Err(CheckedBPError::ZeroSupport { .. })
        ));
    }

    #[test]
    fn invalid_potentials_and_zero_cardinalities_are_rejected() {
        for invalid in [f64::NAN, f64::INFINITY, -0.1, f64::NEG_INFINITY] {
            let mut graph = FactorGraph::new();
            let a = graph.add_variable("A", 2);
            graph.add_factor(&[a], vec![1.0, invalid]);
            assert!(matches!(
                run_sum_product(&graph, &BPConfig::default()),
                Err(CheckedBPError::InvalidPotential { .. })
            ));
        }

        let mut graph = FactorGraph::new();
        graph.add_variable("A", 0);
        assert!(matches!(
            run_sum_product(&graph, &BPConfig::default()),
            Err(CheckedBPError::ZeroCardinality { variable: 0 })
        ));
    }

    #[test]
    fn all_zero_factor_is_rejected_before_message_passing() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        graph.add_factor(&[a], vec![0.0, 0.0]);
        assert_eq!(
            run_sum_product(&graph, &BPConfig::default()),
            Err(CheckedBPError::AllZeroFactor { factor: 0 })
        );
    }

    #[test]
    fn malformed_scopes_and_adjacency_are_rejected() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        graph.add_factor(&[a, a], vec![1.0; 4]);
        assert!(matches!(
            run_sum_product(&graph, &BPConfig::default()),
            Err(CheckedBPError::DuplicateVariableInScope { factor: 0, variable: 0 })
        ));

        let mut graph = two_node_tree();
        graph.variables[0].neighbors.clear();
        assert!(matches!(
            run_sum_product(&graph, &BPConfig::default()),
            Err(CheckedBPError::InvalidAdjacency { variable: 0 })
        ));
    }

    #[test]
    fn extreme_dynamic_range_does_not_fall_back_to_uniform() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        for _ in 0..1200 {
            graph.add_factor(&[a], vec![1.0e-200, 2.0e-200]);
        }

        let result = run_sum_product(&graph, &BPConfig::default()).unwrap();
        assert_eq!(result.beliefs[0][0], 0.0);
        assert!(result.log_beliefs[0][0].is_finite());
        assert!(result.log_beliefs[0][0] < -800.0);
        assert!((result.beliefs[0][1] - 1.0).abs() < 1.0e-12);
        assert!(result.max_message_delta < BPConfig::default().tolerance);
        assert!(result.max_log_message_delta < BPConfig::default().tolerance);
    }

    #[test]
    fn parallel_and_sequential_schedules_agree_on_tree() {
        let graph = two_node_tree();
        let parallel = run_sum_product(&graph, &BPConfig::default()).unwrap();
        let sequential = run_sum_product(&graph, &BPConfig {
            schedule: MessageSchedule::Sequential,
            ..BPConfig::default()
        }).unwrap();
        for variable in 0..graph.variables.len() {
            for state in 0..graph.variables[variable].cardinality {
                assert!((parallel.beliefs[variable][state] - sequential.beliefs[variable][state]).abs() < 1.0e-10);
            }
        }
    }

    #[test]
    fn bipartite_cycle_is_labelled_loopy() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let c = graph.add_variable("C", 2);
        graph.add_factor(&[a, b], vec![0.9, 0.1, 0.1, 0.9]);
        graph.add_factor(&[b, c], vec![0.9, 0.1, 0.1, 0.9]);
        graph.add_factor(&[a, c], vec![0.9, 0.1, 0.1, 0.9]);
        let result = run_sum_product(&graph, &BPConfig::default()).unwrap();
        assert_eq!(result.graph_structure, GraphStructure::Loopy);
        assert_eq!(result.inference_status, InferenceStatus::ApproximateLoopy);
    }

    #[test]
    fn damped_forest_is_not_mislabeled_exact() {
        let graph = two_node_tree();
        let result = run_sum_product(&graph, &BPConfig {
            damping: 0.5,
            ..BPConfig::default()
        }).unwrap();
        assert!(result.converged);
        assert_eq!(result.graph_structure, GraphStructure::Forest);
        assert_eq!(result.inference_status, InferenceStatus::ApproximateDampedForest);
    }

    #[test]
    fn iteration_exhaustion_is_not_labelled_exact() {
        let graph = two_node_tree();
        let result = run_sum_product(&graph, &BPConfig {
            max_iterations: 1,
            tolerance: 1.0e-30,
            damping: 1.0,
            schedule: MessageSchedule::Parallel,
        }).unwrap();
        assert!(!result.converged);
        assert_eq!(result.inference_status, InferenceStatus::NotConverged);
    }

    #[test]
    fn stale_or_corrupt_cached_messages_are_ignored() {
        let graph = two_node_tree();
        let expected = run_sum_product(&graph, &BPConfig::default()).unwrap();

        let mut corrupted = graph.clone();
        corrupted.messages.insert((0, 1), vec![f64::NAN, f64::INFINITY]);
        corrupted.messages.insert((usize::MAX, 0), vec![]);
        let actual = run_sum_product(&corrupted, &BPConfig::default()).unwrap();

        assert_eq!(actual.beliefs, expected.beliefs);
        assert_eq!(actual.inference_status, expected.inference_status);
    }

    #[test]
    fn zero_iteration_limit_is_rejected() {
        let graph = two_node_tree();
        let invalid = BPConfig {
            max_iterations: 0,
            ..BPConfig::default()
        };
        assert_eq!(
            run_sum_product(&graph, &invalid),
            Err(CheckedBPError::InvalidIterationLimit)
        );
    }

    #[test]
    fn invalid_configuration_is_rejected_before_inference() {
        let graph = two_node_tree();
        let invalid = BPConfig { damping: 0.0, ..BPConfig::default() };
        assert!(matches!(
            run_sum_product(&graph, &invalid),
            Err(CheckedBPError::InvalidDamping(0.0))
        ));
    }
}
