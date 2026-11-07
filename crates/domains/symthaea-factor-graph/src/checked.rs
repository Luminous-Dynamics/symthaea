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
    /// A particular message/belief update has no support under the current
    /// messages. On a forest this implies the corresponding component has no
    /// consistent assignment; on a loopy graph it is not a global
    /// unsatisfiability certificate.
    ZeroSupport { context: &'static str },
    NonFiniteMessage { context: &'static str },
    LoopyGraphUnsupported,
    MessagePassingDidNotConverge { max_passes: usize },
    InconsistentTraceback { variable: usize, existing: usize, proposed: usize },
    BeliefShapeMismatch { context: &'static str },
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
            Self::LoopyGraphUnsupported => write!(f, "exact forest MAP inference does not accept loopy graphs"),
            Self::MessagePassingDidNotConverge { max_passes } => write!(f, "forest max-product messages did not stabilize within {max_passes} passes"),
            Self::InconsistentTraceback { variable, existing, proposed } => write!(f, "MAP traceback assigned variable {variable} both {existing} and {proposed}"),
            Self::BeliefShapeMismatch { context } => write!(f, "belief result shape or normalization is invalid: {context}"),
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
    /// Undamped forest messages reached a nonzero residual that is below
    /// the configured tolerance. A converged tolerance test is not, by itself,
    /// proof that the finite message schedule has reached its exact fixed point.
    ApproximateForest,
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
    /// True factor marginals from factor potential times incoming cavity
    /// messages. Each row is in the corresponding factor's table order.
    pub factor_beliefs: Vec<Vec<f64>>,
    /// Log-domain factor marginals, preserving support under exponentiation
    /// underflow. Each row follows the corresponding factor's table order.
    pub log_factor_beliefs: Vec<Vec<f64>>,
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

    let mut factor_beliefs = Vec::with_capacity(graph.factors.len());
    let mut log_factor_beliefs = Vec::with_capacity(graph.factors.len());
    for factor in &graph.factors {
        let cards = factor.variables.iter()
            .map(|&variable| graph.variables[variable].cardinality)
            .collect::<Vec<_>>();
        let total = cards.iter().try_fold(1usize, |count, &cardinality| {
            count.checked_mul(cardinality)
        }).ok_or(CheckedBPError::TableSizeOverflow { factor: factor.id })?;
        let mut log_belief = vec![f64::NEG_INFINITY; total];

        for flat_index in 0..total {
            let potential = factor.table[flat_index];
            if potential == 0.0 {
                continue;
            }
            let config = decode_assignment(flat_index, &cards);
            let mut terms = vec![potential.ln()];
            let mut unsupported = false;
            for (position, &variable) in factor.variables.iter().enumerate() {
                let message = messages
                    .get(&var_to_factor_key(n_variables, variable, factor.id))
                    .expect("validated message initialization");
                let value = message[config[position]];
                if value == f64::NEG_INFINITY {
                    unsupported = true;
                    break;
                }
                terms.push(value);
            }
            if unsupported {
                continue;
            }
            terms.sort_by(f64::total_cmp);
            let score = compensated_sum(&terms);
            if !score.is_finite() {
                if score == f64::NEG_INFINITY {
                    continue;
                }
                return Err(CheckedBPError::NonFiniteMessage { context: "factor belief" });
            }
            log_belief[flat_index] = score;
        }

        normalize_log(&mut log_belief, "factor belief")?;
        factor_beliefs.push(log_belief.iter().map(|value| value.exp()).collect());
        log_factor_beliefs.push(log_belief);
    }

    let inference_status = if !converged {
        InferenceStatus::NotConverged
    } else {
        match (structure, config.damping == 1.0, final_delta == 0.0 && final_log_delta == 0.0) {
            (GraphStructure::Forest, true, true) => InferenceStatus::ExactForest,
            (GraphStructure::Forest, true, false) => InferenceStatus::ApproximateForest,
            (GraphStructure::Forest, false, _) => InferenceStatus::ApproximateDampedForest,
            (GraphStructure::Loopy, _, _) => InferenceStatus::ApproximateLoopy,
        }
    };

    Ok(CheckedBPResult {
        beliefs,
        log_beliefs,
        factor_beliefs,
        log_factor_beliefs,
        converged,
        iterations,
        max_message_delta: final_delta,
        max_log_message_delta: final_log_delta,
        graph_structure: structure,
        inference_status,
    })
}

/// Exact joint-MAP result on a validated forest.
///
/// Ties use deterministic lowest-state / first-row-major backpointer selection.
/// The returned score is recomputed directly from the original factor tables;
/// it is not a sum of singleton max-marginals.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedMAPResult {
    pub assignment: Vec<usize>,
    /// Unnormalized joint log potential of the returned complete assignment.
    pub joint_log_potential: f64,
    pub message_passes: usize,
    pub graph_structure: GraphStructure,
}

/// Exact joint MAP/MPE by max-product messages plus global traceback.
///
/// Only validated forests are accepted. Loopy graphs are rejected rather than
/// returning a local-belief argmax as though it were globally optimal.
///
/// Messages are max-normalized in log space. Factor-to-variable messages
/// retain a deterministic maximizing factor configuration for every supported
/// target state. A root state is selected independently per connected
/// component, then those backpointers reconstruct a globally consistent
/// assignment. Finally, the original factors are reevaluated to verify the
/// assignment and compute its joint log potential.
///
/// Complexity is exponential in the largest factor scope (each factor message
/// enumerates its local table), and linear in the number of message passes
/// times factor-table work. This is an exact small/medium forest algorithm,
/// not an arbitrary-graph MAP solver.
pub fn run_max_product_forest(
    graph: &FactorGraph,
) -> Result<CheckedMAPResult, CheckedBPError> {
    validate_graph(graph)?;
    let structure = graph_structure(graph);
    if structure != GraphStructure::Forest {
        return Err(CheckedBPError::LoopyGraphUnsupported);
    }

    let n_variables = graph.variables.len();
    let n_factors = graph.factors.len();
    let total_nodes = n_variables.checked_add(n_factors)
        .ok_or(CheckedBPError::TableSizeOverflow { factor: usize::MAX })?;
    let max_passes = total_nodes.saturating_add(1);

    let mut variable_to_factor: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    let mut factor_to_variable: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    let mut backpointers: HashMap<(usize, usize), Vec<Option<Vec<usize>>>> = HashMap::new();

    // Max-normalized uniform initialization is the all-zero log message.
    for factor in &graph.factors {
        for &variable in &factor.variables {
            let cardinality = graph.variables[variable].cardinality;
            variable_to_factor.insert((variable, factor.id), vec![0.0; cardinality]);
            factor_to_variable.insert((factor.id, variable), vec![0.0; cardinality]);
        }
    }

    let mut stable = false;
    let mut message_passes = 0usize;
    for pass in 0..max_passes {
        message_passes = pass + 1;

        let mut next_variable_to_factor = HashMap::with_capacity(variable_to_factor.len());
        for variable in 0..n_variables {
            for &factor in &graph.variables[variable].neighbors {
                let message = compute_var_to_factor_max(
                    graph,
                    &factor_to_variable,
                    variable,
                    factor,
                )?;
                next_variable_to_factor.insert((variable, factor), message);
            }
        }

        let mut next_factor_to_variable = HashMap::with_capacity(factor_to_variable.len());
        let mut next_backpointers = HashMap::with_capacity(backpointers.len());
        for factor in 0..n_factors {
            for &variable in &graph.factors[factor].variables {
                let (message, pointers) = compute_factor_to_var_max(
                    graph,
                    &variable_to_factor,
                    factor,
                    variable,
                )?;
                next_factor_to_variable.insert((factor, variable), message);
                next_backpointers.insert((factor, variable), pointers);
            }
        }

        stable = next_variable_to_factor == variable_to_factor
            && next_factor_to_variable == factor_to_variable;
        variable_to_factor = next_variable_to_factor;
        factor_to_variable = next_factor_to_variable;
        backpointers = next_backpointers;
        if stable {
            break;
        }
    }
    if !stable {
        return Err(CheckedBPError::MessagePassingDidNotConverge { max_passes });
    }

    let mut assignment = vec![None; n_variables];
    let mut visited_factors = vec![false; n_factors];

    // Root each disconnected variable-containing component at its lowest
    // unassigned variable ID. Isolated variables deterministically choose 0.
    for root in 0..n_variables {
        if assignment[root].is_some() {
            continue;
        }

        let mut root_scores = vec![0.0; graph.variables[root].cardinality];
        for state in 0..root_scores.len() {
            let mut terms = Vec::with_capacity(graph.variables[root].neighbors.len());
            let mut unsupported = false;
            for &factor in &graph.variables[root].neighbors {
                let incoming = factor_to_variable
                    .get(&(factor, root))
                    .expect("converged forest has every factor-to-variable message");
                if incoming[state] == f64::NEG_INFINITY {
                    unsupported = true;
                    break;
                }
                terms.push(incoming[state]);
            }
            if unsupported {
                root_scores[state] = f64::NEG_INFINITY;
            } else {
                terms.sort_by(f64::total_cmp);
                root_scores[state] = compensated_sum(&terms);
            }
        }
        let root_state = argmax_lowest(&root_scores)
            .ok_or(CheckedBPError::ZeroSupport { context: "forest MAP root" })?;
        assignment[root] = Some(root_state);

        let mut queue = std::collections::VecDeque::new();
        queue.push_back((root, None::<usize>));
        while let Some((variable, parent_factor)) = queue.pop_front() {
            let state = assignment[variable]
                .ok_or(CheckedBPError::ZeroSupport { context: "forest MAP traceback" })?;

            for &factor in &graph.variables[variable].neighbors {
                if Some(factor) == parent_factor || visited_factors[factor] {
                    continue;
                }
                visited_factors[factor] = true;

                let pointers = backpointers
                    .get(&(factor, variable))
                    .expect("converged forest has factor backpointers");
                let chosen = pointers.get(state)
                    .and_then(|pointer| pointer.as_ref())
                    .ok_or(CheckedBPError::ZeroSupport { context: "forest MAP backpointer" })?;

                for (position, &child) in graph.factors[factor].variables.iter().enumerate() {
                    if child == variable {
                        continue;
                    }
                    let proposed = chosen[position];
                    match assignment[child] {
                        Some(existing) if existing != proposed => {
                            return Err(CheckedBPError::InconsistentTraceback {
                                variable: child,
                                existing,
                                proposed,
                            });
                        }
                        Some(_) => {}
                        None => {
                            assignment[child] = Some(proposed);
                            queue.push_back((child, Some(factor)));
                        }
                    }
                }
            }
        }
    }

    let assignment = assignment.into_iter()
        .enumerate()
        .map(|(variable, state)| state.ok_or(CheckedBPError::ZeroSupport {
            context: if variable < n_variables { "incomplete forest MAP assignment" } else { "forest MAP assignment" },
        }))
        .collect::<Result<Vec<_>, _>>()?;

    // Verify the complete assignment against original factor tables and
    // recompute the score. Do not derive a joint score from singleton beliefs.
    let mut factor_log_terms = Vec::with_capacity(graph.factors.len());
    for factor in &graph.factors {
        let table_index = factor.variables.iter().fold(0usize, |index, &variable| {
            index * graph.variables[variable].cardinality + assignment[variable]
        });
        let potential = factor.table[table_index];
        if potential == 0.0 {
            return Err(CheckedBPError::ZeroSupport { context: "joint MAP assignment verification" });
        }
        factor_log_terms.push(potential.ln());
    }
    factor_log_terms.sort_by(f64::total_cmp);
    let joint_log_potential = compensated_sum(&factor_log_terms);
    if !joint_log_potential.is_finite() {
        return Err(CheckedBPError::NonFiniteMessage { context: "joint MAP score" });
    }

    Ok(CheckedMAPResult {
        assignment,
        joint_log_potential,
        message_passes,
        graph_structure: structure,
    })
}

fn compute_var_to_factor_max(
    graph: &FactorGraph,
    messages: &HashMap<(usize, usize), Vec<f64>>,
    variable: usize,
    excluded_factor: usize,
) -> Result<Vec<f64>, CheckedBPError> {
    let cardinality = graph.variables[variable].cardinality;
    let mut result = vec![0.0; cardinality];
    for state in 0..cardinality {
        let mut terms = Vec::with_capacity(graph.variables[variable].neighbors.len());
        let mut unsupported = false;
        for &factor in &graph.variables[variable].neighbors {
            if factor == excluded_factor {
                continue;
            }
            let incoming = messages.get(&(factor, variable))
                .expect("initialized factor-to-variable message");
            if incoming[state] == f64::NEG_INFINITY {
                unsupported = true;
                break;
            }
            terms.push(incoming[state]);
        }
        terms.sort_by(f64::total_cmp);
        result[state] = if unsupported { f64::NEG_INFINITY } else { compensated_sum(&terms) };
    }
    normalize_max(&mut result, "max-product variable-to-factor message")?;
    Ok(result)
}

fn compute_factor_to_var_max(
    graph: &FactorGraph,
    messages: &HashMap<(usize, usize), Vec<f64>>,
    factor_index: usize,
    target_variable: usize,
) -> Result<(Vec<f64>, Vec<Option<Vec<usize>>>), CheckedBPError> {
    let factor = &graph.factors[factor_index];
    let cardinalities = factor.variables.iter()
        .map(|&variable| graph.variables[variable].cardinality)
        .collect::<Vec<_>>();
    let target_position = factor.variables.iter()
        .position(|&variable| variable == target_variable)
        .expect("validated factor scope");
    let total = cardinalities.iter().try_fold(1usize, |count, &cardinality| {
        count.checked_mul(cardinality)
    }).ok_or(CheckedBPError::TableSizeOverflow { factor: factor_index })?;

    let mut result = vec![f64::NEG_INFINITY; cardinalities[target_position]];
    let mut pointers = vec![None; cardinalities[target_position]];

    for flat_index in 0..total {
        let config = decode_assignment(flat_index, &cardinalities);
        let potential = factor.table[flat_index];
        if potential == 0.0 {
            continue;
        }
        let mut terms = vec![potential.ln()];
        let mut supported = true;
        for (position, &variable) in factor.variables.iter().enumerate() {
            if position == target_position {
                continue;
            }
            let incoming = messages.get(&(variable, factor_index))
                .expect("initialized variable-to-factor message");
            let value = incoming[config[position]];
            if value == f64::NEG_INFINITY {
                supported = false;
                break;
            }
            terms.push(value);
        }
        if !supported {
            continue;
        }
        terms.sort_by(f64::total_cmp);
        let score = compensated_sum(&terms);
        if !score.is_finite() {
            return Err(CheckedBPError::NonFiniteMessage { context: "max-product factor score" });
        }

        let target_state = config[target_position];
        if score > result[target_state] || (score == result[target_state] && pointers[target_state].is_none()) {
            result[target_state] = score;
            pointers[target_state] = Some(config);
        }
    }

    normalize_max(&mut result, "max-product factor-to-variable message")?;
    Ok((result, pointers))
}

fn normalize_max(values: &mut [f64], context: &'static str) -> Result<(), CheckedBPError> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return Err(CheckedBPError::ZeroSupport { context });
    }
    if !maximum.is_finite() {
        return Err(CheckedBPError::NonFiniteMessage { context });
    }
    for value in values {
        *value -= maximum;
    }
    Ok(())
}

fn argmax_lowest(values: &[f64]) -> Option<usize> {
    let mut best_state = None;
    let mut best_value = f64::NEG_INFINITY;
    for (state, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_state = Some(state);
        }
    }
    best_state
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

/// Bethe free-energy evaluation status.
///
/// On a forest with exact undamped BP beliefs, the Bethe log-partition equals
/// the exact log partition. On loopy graphs, it is a variational approximation.
/// If BP did not converge, this is simply the Bethe functional evaluated at
/// the returned local beliefs, not a stationary-point estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetheStatus {
    ExactForest,
    /// Converged damped-forest or loopy fixed-point approximation.
    ApproximateFixedPoint,
    /// BP stopped at a tolerance or iteration boundary without an exact or
    /// converged fixed point.
    NonStationary,
}

/// Bethe functional and its sign-converted log-partition estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct BetheResult {
    pub free_energy: f64,
    pub log_partition_estimate: f64,
    pub status: BetheStatus,
}

/// Evaluate the Bethe functional using true cavity-derived factor beliefs.
///
/// F_B = sum_a E_{b_a}[log b_a - log f_a]
///       + sum_i (1 - degree(i)) E_{b_i}[log b_i].
///
/// This function consumes only beliefs emitted by the checked sum-product API.
/// It does not use the legacy singleton-product factor approximation or an
/// arbitrary penalty for impossible factor entries.
pub fn bethe_free_energy(
    graph: &FactorGraph,
    inference: &CheckedBPResult,
) -> Result<BetheResult, CheckedBPError> {
    validate_graph(graph)?;
    if inference.factor_beliefs.len() != graph.factors.len()
        || inference.log_factor_beliefs.len() != graph.factors.len()
        || inference.beliefs.len() != graph.variables.len()
        || inference.log_beliefs.len() != graph.variables.len()
        || inference.graph_structure != graph_structure(graph)
    {
        return Err(CheckedBPError::BeliefShapeMismatch { context: "top-level result dimensions or graph structure" });
    }

    for (variable_index, variable) in graph.variables.iter().enumerate() {
        let probabilities = &inference.beliefs[variable_index];
        let log_probabilities = &inference.log_beliefs[variable_index];
        if probabilities.len() != variable.cardinality
            || log_probabilities.len() != variable.cardinality
        {
            return Err(CheckedBPError::BeliefShapeMismatch { context: "variable marginal cardinality" });
        }
        if probabilities.iter().any(|value| !value.is_finite() || *value < 0.0)
            || log_probabilities.iter().any(|value| value.is_nan() || *value == f64::INFINITY)
        {
            return Err(CheckedBPError::BeliefShapeMismatch { context: "non-finite or negative variable marginal" });
        }
        for (&probability, &log_probability) in probabilities.iter().zip(log_probabilities) {
            let reconstructed = log_probability.exp();
            if (probability - reconstructed).abs() > 1.0e-10 {
                return Err(CheckedBPError::BeliefShapeMismatch { context: "variable probability/log-probability mismatch" });
            }
        }
        let probability_sum = probabilities.iter().sum::<f64>();
        let log_sum = log_sum_exp(log_probabilities);
        if (probability_sum - 1.0).abs() > 1.0e-8 || !log_sum.is_finite() || log_sum.abs() > 1.0e-8 {
            return Err(CheckedBPError::BeliefShapeMismatch { context: "variable marginal is not normalized" });
        }
    }

    for (factor_index, factor) in graph.factors.iter().enumerate() {
        let probabilities = &inference.factor_beliefs[factor_index];
        let log_probabilities = &inference.log_factor_beliefs[factor_index];
        if probabilities.len() != factor.table.len()
            || log_probabilities.len() != factor.table.len()
        {
            return Err(CheckedBPError::BeliefShapeMismatch { context: "factor marginal table length" });
        }
        if probabilities.iter().any(|value| !value.is_finite() || *value < 0.0)
            || log_probabilities.iter().any(|value| value.is_nan() || *value == f64::INFINITY)
        {
            return Err(CheckedBPError::BeliefShapeMismatch { context: "non-finite or negative factor marginal" });
        }
        for (&probability, &log_probability) in probabilities.iter().zip(log_probabilities) {
            let reconstructed = log_probability.exp();
            if (probability - reconstructed).abs() > 1.0e-10 {
                return Err(CheckedBPError::BeliefShapeMismatch { context: "factor probability/log-probability mismatch" });
            }
        }
        let probability_sum = probabilities.iter().sum::<f64>();
        let log_sum = log_sum_exp(log_probabilities);
        if (probability_sum - 1.0).abs() > 1.0e-8 || !log_sum.is_finite() || log_sum.abs() > 1.0e-8 {
            return Err(CheckedBPError::BeliefShapeMismatch { context: "factor marginal is not normalized" });
        }
    }

    let mut energy_terms = Vec::new();
    for (factor_index, factor) in graph.factors.iter().enumerate() {
        let log_beliefs = &inference.log_factor_beliefs[factor_index];
        if log_beliefs.len() != factor.table.len() {
            return Err(CheckedBPError::TableLengthMismatch {
                factor: factor_index,
                expected: factor.table.len(),
                actual: log_beliefs.len(),
            });
        }
        for (table_index, &log_belief) in log_beliefs.iter().enumerate() {
            if log_belief == f64::NEG_INFINITY {
                continue;
            }
            let potential = factor.table[table_index];
            if potential <= 0.0 {
                return Err(CheckedBPError::ZeroSupport { context: "Bethe factor energy" });
            }
            let probability = log_belief.exp();
            if probability == 0.0 {
                continue;
            }
            energy_terms.push(probability * (log_belief - potential.ln()));
        }
    }

    for (variable_index, variable) in graph.variables.iter().enumerate() {
        let degree = variable.neighbors.len() as f64;
        let log_belief = &inference.log_beliefs[variable_index];
        if log_belief.len() != variable.cardinality {
            return Err(CheckedBPError::TableLengthMismatch {
                factor: usize::MAX,
                expected: variable.cardinality,
                actual: log_belief.len(),
            });
        }
        for &log_probability in log_belief {
            if log_probability == f64::NEG_INFINITY {
                continue;
            }
            let probability = log_probability.exp();
            if probability > 0.0 {
                energy_terms.push((1.0 - degree) * probability * log_probability);
            }
        }
    }

    energy_terms.sort_by(f64::total_cmp);
    let free_energy = compensated_sum(&energy_terms);
    let log_partition_estimate = -free_energy;
    if !(free_energy.is_finite() && log_partition_estimate.is_finite()) {
        return Err(CheckedBPError::NonFiniteMessage { context: "Bethe free energy" });
    }

    let status = if !inference.converged {
        BetheStatus::NonStationary
    } else {
        match (inference.inference_status, inference.graph_structure) {
            (InferenceStatus::ExactForest, GraphStructure::Forest) => BetheStatus::ExactForest,
            (InferenceStatus::NotConverged, _) => BetheStatus::NonStationary,
            _ => BetheStatus::ApproximateFixedPoint,
        }
    };

    Ok(BetheResult {
        free_energy,
        log_partition_estimate,
        status,
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

fn log_sum_exp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if !maximum.is_finite() {
        return f64::NAN;
    }
    let sum = values.iter().map(|value| (*value - maximum).exp()).sum::<f64>();
    maximum + sum.ln()
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
    fn tolerance_convergence_without_exact_fixed_point_is_not_labelled_exact() {
        let graph = two_node_tree();
        let result = run_sum_product(&graph, &BPConfig {
            max_iterations: 1,
            tolerance: 10.0,
            damping: 1.0,
            schedule: MessageSchedule::Parallel,
        }).unwrap();

        assert!(result.converged);
        assert!(result.max_message_delta > 0.0);
        assert!(result.max_log_message_delta > 0.0);
        assert_eq!(result.inference_status, InferenceStatus::ApproximateForest);
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
    fn factor_beliefs_match_direct_factor_marginals_on_a_tree() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        graph.add_factor(&[a, b], vec![0.3, 0.1, 0.2, 0.4]);

        let inference = run_sum_product(&graph, &BPConfig::default()).unwrap();
        assert_eq!(inference.factor_beliefs.len(), 1);
        let expected = [0.3_f64 / 1.0, 0.1, 0.2, 0.4];
        for (actual, expected) in inference.factor_beliefs[0].iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        let bethe = bethe_free_energy(&graph, &inference).unwrap();
        assert_eq!(bethe.status, BetheStatus::ExactForest);
        assert!((bethe.log_partition_estimate - 0.0).abs() < 1.0e-10);
    }

    #[test]
    fn factor_belief_uses_cavity_potential_not_product_of_singleton_marginals() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let c = graph.add_variable("C", 2);
        graph.add_factor(&[a, b], vec![0.9, 0.1, 0.1, 0.9]);
        graph.add_factor(&[b, c], vec![0.9, 0.1, 0.1, 0.9]);
        graph.add_factor(&[a, c], vec![0.9, 0.1, 0.1, 0.9]);

        let inference = run_sum_product(&graph, &BPConfig::default()).unwrap();
        // Symmetry makes singleton marginals 0.5/0.5, whose product is 0.25.
        // The true cavity-derived factor belief is proportional to the factor:
        // [0.45, 0.05, 0.05, 0.45].
        let expected = [0.45, 0.05, 0.05, 0.45];
        for (actual, expected) in inference.factor_beliefs[0].iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        assert!((inference.factor_beliefs[0][0] - 0.25).abs() > 0.1);
        let bethe = bethe_free_energy(&graph, &inference).unwrap();
        assert_eq!(bethe.status, BetheStatus::ApproximateFixedPoint);
    }

    #[test]
    fn bethe_log_partition_matches_direct_enumeration_on_a_three_node_tree() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let c = graph.add_variable("C", 2);
        graph.add_factor(&[a, b], vec![0.4, 0.1, 0.2, 0.3]);
        graph.add_factor(&[b, c], vec![0.5, 0.5, 0.3, 0.7]);

        let inference = run_sum_product(&graph, &BPConfig::default()).unwrap();
        let mut z = 0.0;
        for av in 0..2 {
            for bv in 0..2 {
                for cv in 0..2 {
                    let left = [[0.4, 0.1], [0.2, 0.3]][av][bv];
                    let right = [[0.5, 0.5], [0.3, 0.7]][bv][cv];
                    z += left * right;
                }
            }
        }
        let bethe = bethe_free_energy(&graph, &inference).unwrap();
        assert_eq!(bethe.status, BetheStatus::ExactForest);
        assert!((bethe.log_partition_estimate - z.ln()).abs() < 1.0e-10);
    }

    #[test]
    fn chain_factor_beliefs_match_direct_joint_marginals() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let c = graph.add_variable("C", 2);
        let f1 = [0.4, 0.1, 0.2, 0.3];
        let f2 = [0.5, 0.5, 0.3, 0.7];
        graph.add_factor(&[a, b], f1.to_vec());
        graph.add_factor(&[b, c], f2.to_vec());

        let inference = run_sum_product(&graph, &BPConfig::default()).unwrap();
        let mut z = 0.0;
        let mut exact_f1 = [0.0; 4];
        let mut exact_f2 = [0.0; 4];
        for av in 0..2 {
            for bv in 0..2 {
                for cv in 0..2 {
                    let weight = f1[av * 2 + bv] * f2[bv * 2 + cv];
                    z += weight;
                    exact_f1[av * 2 + bv] += weight;
                    exact_f2[bv * 2 + cv] += weight;
                }
            }
        }
        for index in 0..4 {
            assert!((inference.factor_beliefs[0][index] - exact_f1[index] / z).abs() < 1.0e-10);
            assert!((inference.factor_beliefs[1][index] - exact_f2[index] / z).abs() < 1.0e-10);
        }
    }

    #[test]
    fn bethe_rejects_mutated_factor_belief_dimensions_and_inconsistent_probability_views() {
        let graph = two_node_tree();
        let inference = run_sum_product(&graph, &BPConfig::default()).unwrap();

        let mut wrong_shape = inference.clone();
        wrong_shape.log_factor_beliefs[0].pop();
        assert!(matches!(
            bethe_free_energy(&graph, &wrong_shape),
            Err(CheckedBPError::BeliefShapeMismatch { .. })
        ));

        let mut wrong_values = inference.clone();
        wrong_values.factor_beliefs[0][0] = 0.9;
        assert!(matches!(
            bethe_free_energy(&graph, &wrong_values),
            Err(CheckedBPError::BeliefShapeMismatch { .. })
        ));
    }

    #[test]
    fn loose_tolerance_bethe_is_not_labelled_exact() {
        let graph = two_node_tree();
        let inference = run_sum_product(&graph, &BPConfig {
            max_iterations: 1,
            tolerance: 10.0,
            ..BPConfig::default()
        }).unwrap();
        let bethe = bethe_free_energy(&graph, &inference).unwrap();
        assert_eq!(bethe.status, BetheStatus::NonStationary);
    }

    #[test]
    fn forest_joint_map_traceback_resolves_incompatible_local_ties() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let c = graph.add_variable("C", 2);
        // A=B and B!=C. Valid global maxima: (0,0,1) and (1,1,0).
        graph.add_factor(&[a, b], vec![1.0, 0.0, 0.0, 1.0]);
        graph.add_factor(&[b, c], vec![0.0, 1.0, 1.0, 0.0]);

        let result = run_max_product_forest(&graph).unwrap();
        assert_eq!(result.assignment, vec![0, 0, 1]);
        assert_eq!(result.joint_log_potential, 0.0);
        assert_eq!(result.graph_structure, GraphStructure::Forest);
        assert!(result.message_passes <= graph.variables.len() + graph.factors.len() + 1);
    }

    #[test]
    fn forest_joint_map_handles_mixed_cardinality_and_verifies_joint_score() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 3);
        graph.add_factor(&[a, b], vec![1.0, 2.0, 3.0, 4.0, 5.0, 9.0]);

        let result = run_max_product_forest(&graph).unwrap();
        assert_eq!(result.assignment, vec![1, 2]);
        assert!((result.joint_log_potential - 9.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn forest_joint_map_combines_disconnected_components_and_isolated_variables() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let isolated = graph.add_variable("I", 3);
        graph.add_factor(&[a], vec![0.1, 0.9]);
        graph.add_factor(&[b], vec![0.8, 0.2]);

        let result = run_max_product_forest(&graph).unwrap();
        assert_eq!(result.assignment, vec![1, 0, 0]);
        assert!((result.joint_log_potential - (0.9_f64 * 0.8).ln()).abs() < 1.0e-12);
    }

    #[test]
    fn exact_joint_map_refuses_loopy_graphs() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        let b = graph.add_variable("B", 2);
        let c = graph.add_variable("C", 2);
        graph.add_factor(&[a, b], vec![0.9, 0.1, 0.1, 0.9]);
        graph.add_factor(&[b, c], vec![0.9, 0.1, 0.1, 0.9]);
        graph.add_factor(&[a, c], vec![0.9, 0.1, 0.1, 0.9]);
        assert_eq!(
            run_max_product_forest(&graph),
            Err(CheckedBPError::LoopyGraphUnsupported)
        );
    }

    #[test]
    fn forest_joint_map_returns_zero_support_on_contradictory_factors() {
        let mut graph = FactorGraph::new();
        let a = graph.add_variable("A", 2);
        graph.add_factor(&[a], vec![1.0, 0.0]);
        graph.add_factor(&[a], vec![0.0, 1.0]);
        assert!(matches!(
            run_max_product_forest(&graph),
            Err(CheckedBPError::ZeroSupport { .. })
        ));
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
