// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Complete finite game-value profiles and win-whenever-possible QBF functions.
//!
//! This module implements the bounded, explicit game-value layer underneath
//! QBF synthesis. For each prefix position i, g_i maps every assignment to
//! the first i moves to the truth value of the remaining quantified suffix.
//!
//! This is deliberately distinct from WinningStrategy:
//! - a winning strategy only needs to win from the initial position;
//! - a Skolem/Herbrand function provides the correct choice from every
//!   partial position where that player can still force its preferred result.
//!
//! The construction reads a verified residual proof DAG. The independent
//! verifier checks the complete layer recurrence against the original CNF.
//! This is a finite reference interface, not a general SynQBF proof parser.

use super::decomposition::CnfProblem;
use super::quantified_game_dag::{
    QbfGameDagBudget, QbfGameDagError, QbfGameDagNodeKind, QbfGameProofDag,
    verify_qbf_game_proof_dag,
};
use super::quantified_scope::{Quantifier, ValidatedQuantifierPrefix};
use std::collections::BTreeMap;

/// A complete game-value function after a fixed number of prefix moves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameValueLayer {
    pub prefix_position: usize,
    /// Original variable IDs in exact quantifier-prefix order.
    pub prefix_variables: Vec<usize>,
    /// Keys contain assigned values of prefix_variables in that order.
    pub values: BTreeMap<Vec<bool>, bool>,
}

/// A full-history Skolem or Herbrand function for one quantified variable.
///
/// Unlike restricted winning-strategy functions, dependencies include every
/// variable preceding the controlled variable in the ordered prefix. This
/// matches full-history Skolem/Herbrand synthesis, not opponent-only strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullHistoryStrategyFunction {
    pub variable: usize,
    pub quantifier: Quantifier,
    pub dependencies: Vec<usize>,
    /// A total table over dependencies, in prefix order.
    pub table: BTreeMap<Vec<bool>, bool>,
}

/// Both full-history strategy families and all intermediate game values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantifiedGameValueProfile {
    pub variable_count: usize,
    pub prefix_order: Vec<(Quantifier, usize)>,
    /// Exactly variable_count + 1 layers, from no moves through all moves.
    pub layers: Vec<GameValueLayer>,
    /// One full-history Skolem function for every existential variable.
    pub skolem_functions: Vec<FullHistoryStrategyFunction>,
    /// One full-history Herbrand function for every universal variable.
    pub herbrand_functions: Vec<FullHistoryStrategyFunction>,
    pub truth_value: bool,
}

/// Deterministic limits for finite game-value extraction and checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameValueProfileBudget {
    pub max_variables: usize,
    pub max_total_variable_name_bytes: usize,
    pub max_clauses: usize,
    pub max_literals: usize,
    pub max_total_truth_table_entries: u128,
    pub max_dag_path_steps: u128,
    pub max_matrix_literal_checks: u128,
}

impl Default for GameValueProfileBudget {
    fn default() -> Self {
        Self {
            max_variables: 16,
            max_total_variable_name_bytes: 4 * 1024,
            max_clauses: 1 << 16,
            max_literals: 1 << 20,
            max_total_truth_table_entries: 1 << 24,
            max_dag_path_steps: 1 << 28,
            max_matrix_literal_checks: 1 << 28,
        }
    }
}

/// Fail-closed construction and verification errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameValueProfileError {
    DagVerification(QbfGameDagError),
    MatrixPrefixVariableCountMismatch {
        matrix: usize,
        prefix: usize,
    },
    VariableBudgetExceeded {
        required: usize,
        limit: usize,
    },
    VariableNameBudgetExceeded {
        required: usize,
        limit: usize,
    },
    EmptyVariableName {
        index: usize,
    },
    DuplicateVariableName(String),
    ClauseBudgetExceeded {
        required: usize,
        limit: usize,
    },
    LiteralBudgetExceeded {
        required: usize,
        limit: usize,
    },
    LiteralVariableOutOfRange {
        clause: usize,
        literal: usize,
        variable: usize,
        variable_count: usize,
    },
    TruthTableEntryBudgetExceeded {
        required: u128,
        limit: u128,
    },
    DagPathStepBudgetExceeded {
        required: u128,
        limit: u128,
    },
    MatrixLiteralCheckBudgetExceeded {
        required: u128,
        limit: u128,
    },
    PrefixBindingMismatch,
    MissingDagNode {
        node: usize,
    },
    NodePrefixPositionMismatch {
        node: usize,
        expected: usize,
        actual: usize,
    },
    LayerCountMismatch {
        expected: usize,
        actual: usize,
    },
    LayerPrefixPositionMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    LayerVariableScopeMismatch {
        layer: usize,
    },
    LayerEntryCountMismatch {
        layer: usize,
        expected: u128,
        actual: usize,
    },
    LayerKeyWidthMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    MissingLayerAssignment {
        layer: usize,
        assignment: Vec<bool>,
    },
    MatrixValueMismatch {
        assignment: Vec<bool>,
        expected: bool,
        actual: bool,
    },
    GameValueRecurrenceMismatch {
        prefix_position: usize,
        assignment: Vec<bool>,
        expected: bool,
        actual: bool,
    },
    StrategyFunctionCountMismatch {
        owner: Quantifier,
        expected: usize,
        actual: usize,
    },
    StrategyFunctionVariableMismatch {
        owner: Quantifier,
        index: usize,
        expected: usize,
        actual: usize,
    },
    StrategyFunctionScopeMismatch {
        variable: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    StrategyFunctionEntryCountMismatch {
        variable: usize,
        expected: u128,
        actual: usize,
    },
    MissingStrategyFunctionRow {
        variable: usize,
        assignment: Vec<bool>,
    },
    StrategyFunctionValueMismatch {
        variable: usize,
        assignment: Vec<bool>,
        expected: bool,
        actual: bool,
    },
    TruthValueMismatch {
        expected: bool,
        actual: bool,
    },
    ResourceAccountingOverflow,
}

/// Work independently derived by the profile verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GameValueProfileVerificationReport {
    pub layers_checked: usize,
    pub layer_assignments_checked: u128,
    pub recurrence_checks: u128,
    pub strategy_function_rows_checked: u128,
    pub matrix_evaluations: u128,
    pub matrix_literal_checks: u128,
    pub truth_value: bool,
}

fn checked_power_of_two(exponent: usize) -> Result<u128, GameValueProfileError> {
    let shift =
        u32::try_from(exponent).map_err(|_| GameValueProfileError::ResourceAccountingOverflow)?;
    1u128
        .checked_shl(shift)
        .ok_or(GameValueProfileError::ResourceAccountingOverflow)
}

fn ordered_prefix(prefix: &ValidatedQuantifierPrefix) -> Vec<(Quantifier, usize)> {
    prefix
        .blocks()
        .iter()
        .flat_map(|block| {
            block
                .variables
                .iter()
                .copied()
                .map(move |variable| (block.quantifier, variable))
        })
        .collect()
}

fn profile_entry_bound(variable_count: usize) -> Result<u128, GameValueProfileError> {
    let final_layer = checked_power_of_two(variable_count)?;
    let layers = final_layer
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
    let one_function_family = final_layer
        .checked_sub(1)
        .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
    // Exactly one full-history function is emitted for each quantified
    // variable, regardless of whether it is existential or universal.
    layers
        .checked_add(one_function_family)
        .ok_or(GameValueProfileError::ResourceAccountingOverflow)
}

fn dag_path_step_bound(variable_count: usize) -> Result<u128, GameValueProfileError> {
    // Sum_i (i + 1) * 2^i for i=0..n. Each lookup visits at most i+1 nodes.
    let mut total = 0u128;
    for position in 0..=variable_count {
        let assignments = checked_power_of_two(position)?;
        let steps = assignments
            .checked_mul(
                u128::try_from(position + 1)
                    .map_err(|_| GameValueProfileError::ResourceAccountingOverflow)?,
            )
            .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
        total = total
            .checked_add(steps)
            .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
    }
    Ok(total)
}

fn validate_profile_source(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: GameValueProfileBudget,
) -> Result<Vec<(Quantifier, usize)>, GameValueProfileError> {
    if problem.variables.len() != prefix.variable_count() {
        return Err(GameValueProfileError::MatrixPrefixVariableCountMismatch {
            matrix: problem.variables.len(),
            prefix: prefix.variable_count(),
        });
    }
    if problem.variables.len() > budget.max_variables {
        return Err(GameValueProfileError::VariableBudgetExceeded {
            required: problem.variables.len(),
            limit: budget.max_variables,
        });
    }

    let name_bytes = problem.variables.iter().try_fold(0usize, |total, name| {
        total
            .checked_add(name.len())
            .ok_or(GameValueProfileError::ResourceAccountingOverflow)
    })?;
    if name_bytes > budget.max_total_variable_name_bytes {
        return Err(GameValueProfileError::VariableNameBudgetExceeded {
            required: name_bytes,
            limit: budget.max_total_variable_name_bytes,
        });
    }
    let mut names = std::collections::BTreeSet::new();
    for (index, name) in problem.variables.iter().enumerate() {
        if name.is_empty() {
            return Err(GameValueProfileError::EmptyVariableName { index });
        }
        if !names.insert(name.as_str()) {
            return Err(GameValueProfileError::DuplicateVariableName(name.clone()));
        }
    }
    if problem.clauses.len() > budget.max_clauses {
        return Err(GameValueProfileError::ClauseBudgetExceeded {
            required: problem.clauses.len(),
            limit: budget.max_clauses,
        });
    }
    let literal_total = problem.clauses.iter().try_fold(0usize, |total, clause| {
        total
            .checked_add(clause.len())
            .ok_or(GameValueProfileError::ResourceAccountingOverflow)
    })?;
    if literal_total > budget.max_literals {
        return Err(GameValueProfileError::LiteralBudgetExceeded {
            required: literal_total,
            limit: budget.max_literals,
        });
    }
    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        for (literal_index, literal) in clause.iter().enumerate() {
            if literal.variable >= problem.variables.len() {
                return Err(GameValueProfileError::LiteralVariableOutOfRange {
                    clause: clause_index,
                    literal: literal_index,
                    variable: literal.variable,
                    variable_count: problem.variables.len(),
                });
            }
        }
    }

    let entries = profile_entry_bound(problem.variables.len())?;
    if entries > budget.max_total_truth_table_entries {
        return Err(GameValueProfileError::TruthTableEntryBudgetExceeded {
            required: entries,
            limit: budget.max_total_truth_table_entries,
        });
    }
    let path_steps = dag_path_step_bound(problem.variables.len())?;
    if path_steps > budget.max_dag_path_steps {
        return Err(GameValueProfileError::DagPathStepBudgetExceeded {
            required: path_steps,
            limit: budget.max_dag_path_steps,
        });
    }

    let assignments = checked_power_of_two(problem.variables.len())?;
    let literal_count = problem.clauses.iter().try_fold(0u128, |total, clause| {
        total
            .checked_add(
                u128::try_from(clause.len())
                    .map_err(|_| GameValueProfileError::ResourceAccountingOverflow)?,
            )
            .ok_or(GameValueProfileError::ResourceAccountingOverflow)
    })?;
    let literal_checks = assignments
        .checked_mul(literal_count)
        .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
    if literal_checks > budget.max_matrix_literal_checks {
        return Err(GameValueProfileError::MatrixLiteralCheckBudgetExceeded {
            required: literal_checks,
            limit: budget.max_matrix_literal_checks,
        });
    }

    Ok(ordered_prefix(prefix))
}

fn prefix_assignment(mask: u128, width: usize) -> Vec<bool> {
    (0..width).map(|index| ((mask >> index) & 1) != 0).collect()
}

fn dag_value_after_prefix(
    dag: &QbfGameProofDag,
    assignment: &[bool],
    requested_position: usize,
    step_counter: &mut u128,
    budget: GameValueProfileBudget,
) -> Result<bool, GameValueProfileError> {
    let mut node_id = dag.root;
    loop {
        *step_counter = step_counter
            .checked_add(1)
            .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
        if *step_counter > budget.max_dag_path_steps {
            return Err(GameValueProfileError::DagPathStepBudgetExceeded {
                required: *step_counter,
                limit: budget.max_dag_path_steps,
            });
        }
        let node = dag
            .nodes
            .get(node_id)
            .ok_or(GameValueProfileError::MissingDagNode { node: node_id })?;

        if node.prefix_position == requested_position {
            return Ok(node.value);
        }
        if node.prefix_position > requested_position {
            return Err(GameValueProfileError::NodePrefixPositionMismatch {
                node: node_id,
                expected: requested_position,
                actual: node.prefix_position,
            });
        }

        match &node.kind {
            QbfGameDagNodeKind::TrueTerminal | QbfGameDagNodeKind::FalseTerminal => {
                return Ok(node.value);
            }
            QbfGameDagNodeKind::ExistsDecision {
                low_child,
                high_child,
                ..
            }
            | QbfGameDagNodeKind::ForAllDecision {
                low_child,
                high_child,
                ..
            } => {
                let branch = *assignment.get(node.prefix_position).ok_or(
                    GameValueProfileError::NodePrefixPositionMismatch {
                        node: node_id,
                        expected: node.prefix_position + 1,
                        actual: assignment.len(),
                    },
                )?;
                node_id = if branch { *high_child } else { *low_child };
            }
        }
    }
}

/// Extract every intermediate game-value layer and both full-history
/// strategy-function families from an independently verified residual DAG.
///
/// The input DAG verifier is run before extraction. Output tables are complete
/// over every prefix assignment, even where the DAG has shared states or
/// decisive terminal subtrees.
pub fn extract_game_value_profile_from_dag(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    dag: &QbfGameProofDag,
    dag_budget: QbfGameDagBudget,
    budget: GameValueProfileBudget,
) -> Result<QuantifiedGameValueProfile, GameValueProfileError> {
    let prefix_order = validate_profile_source(problem, prefix, budget)?;
    verify_qbf_game_proof_dag(problem, prefix, dag, dag_budget)
        .map_err(GameValueProfileError::DagVerification)?;
    if dag.prefix_order != prefix_order {
        return Err(GameValueProfileError::PrefixBindingMismatch);
    }

    let mut layers = Vec::with_capacity(prefix_order.len() + 1);
    let mut path_steps = 0u128;
    for position in 0..=prefix_order.len() {
        let variables: Vec<usize> = prefix_order[..position]
            .iter()
            .map(|(_, variable)| *variable)
            .collect();
        let rows = checked_power_of_two(position)?;
        let mut values = BTreeMap::new();
        for mask in 0..rows {
            let key = prefix_assignment(mask, position);
            let value = dag_value_after_prefix(dag, &key, position, &mut path_steps, budget)?;
            values.insert(key, value);
        }
        layers.push(GameValueLayer {
            prefix_position: position,
            prefix_variables: variables,
            values,
        });
    }

    let mut skolem_functions = Vec::new();
    let mut herbrand_functions = Vec::new();
    for (position, &(quantifier, variable)) in prefix_order.iter().enumerate() {
        let dependencies: Vec<usize> = prefix_order[..position]
            .iter()
            .map(|(_, previous_variable)| *previous_variable)
            .collect();
        let chosen_value = matches!(quantifier, Quantifier::Exists);
        let next_layer = &layers[position + 1];
        let rows = checked_power_of_two(position)?;
        let mut table = BTreeMap::new();
        for mask in 0..rows {
            let history = prefix_assignment(mask, position);
            let mut child_key = history.clone();
            child_key.push(chosen_value);
            let Some(&value) = next_layer.values.get(&child_key) else {
                return Err(GameValueProfileError::MissingLayerAssignment {
                    layer: position + 1,
                    assignment: child_key,
                });
            };
            table.insert(history, value);
        }
        let function = FullHistoryStrategyFunction {
            variable,
            quantifier,
            dependencies,
            table,
        };
        match quantifier {
            Quantifier::Exists => skolem_functions.push(function),
            Quantifier::ForAll => herbrand_functions.push(function),
        }
    }

    let truth_value = *layers
        .first()
        .and_then(|layer| layer.values.get(&vec![]))
        .ok_or(GameValueProfileError::MissingLayerAssignment {
            layer: 0,
            assignment: vec![],
        })?;

    Ok(QuantifiedGameValueProfile {
        variable_count: problem.variables.len(),
        prefix_order,
        layers,
        skolem_functions,
        herbrand_functions,
        truth_value,
    })
}

fn matrix_value(
    problem: &CnfProblem,
    assignment: &[bool],
    literal_checks: &mut u128,
) -> Result<bool, GameValueProfileError> {
    for clause in &problem.clauses {
        let mut clause_satisfied = false;
        for literal in clause {
            *literal_checks = literal_checks
                .checked_add(1)
                .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
            if assignment[literal.variable] == literal.positive {
                clause_satisfied = true;
                break;
            }
        }
        if !clause_satisfied {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Independently validate a complete game-value profile against the source
/// matrix and exact prefix recurrence. This does not build or query a proof DAG.
pub fn verify_game_value_profile(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    profile: &QuantifiedGameValueProfile,
    budget: GameValueProfileBudget,
) -> Result<GameValueProfileVerificationReport, GameValueProfileError> {
    let expected_order = validate_profile_source(problem, prefix, budget)?;
    if profile.variable_count != problem.variables.len() || profile.prefix_order != expected_order {
        return Err(GameValueProfileError::PrefixBindingMismatch);
    }

    let expected_layer_count = expected_order.len() + 1;
    if profile.layers.len() != expected_layer_count {
        return Err(GameValueProfileError::LayerCountMismatch {
            expected: expected_layer_count,
            actual: profile.layers.len(),
        });
    }

    let mut report = GameValueProfileVerificationReport::default();
    for (position, layer) in profile.layers.iter().enumerate() {
        if layer.prefix_position != position {
            return Err(GameValueProfileError::LayerPrefixPositionMismatch {
                layer: position,
                expected: position,
                actual: layer.prefix_position,
            });
        }
        let expected_variables: Vec<usize> = expected_order[..position]
            .iter()
            .map(|(_, variable)| *variable)
            .collect();
        if layer.prefix_variables != expected_variables {
            return Err(GameValueProfileError::LayerVariableScopeMismatch { layer: position });
        }
        let expected_entries = checked_power_of_two(position)?;
        if layer.values.len() as u128 != expected_entries {
            return Err(GameValueProfileError::LayerEntryCountMismatch {
                layer: position,
                expected: expected_entries,
                actual: layer.values.len(),
            });
        }
        for key in layer.values.keys() {
            if key.len() != position {
                return Err(GameValueProfileError::LayerKeyWidthMismatch {
                    layer: position,
                    expected: position,
                    actual: key.len(),
                });
            }
        }
    }

    let final_layer = profile
        .layers
        .last()
        .ok_or(GameValueProfileError::LayerCountMismatch {
            expected: expected_layer_count,
            actual: 0,
        })?;
    let full_assignments = checked_power_of_two(expected_order.len())?;
    let mut literal_checks = 0u128;
    for mask in 0..full_assignments {
        let key = prefix_assignment(mask, expected_order.len());
        let mut assignment = vec![false; problem.variables.len()];
        for (position, &(_, variable)) in expected_order.iter().enumerate() {
            assignment[variable] = key[position];
        }
        let expected = matrix_value(problem, &assignment, &mut literal_checks)?;
        let actual = *final_layer.values.get(&key).ok_or_else(|| {
            GameValueProfileError::MissingLayerAssignment {
                layer: expected_order.len(),
                assignment: key.clone(),
            }
        })?;
        if expected != actual {
            return Err(GameValueProfileError::MatrixValueMismatch {
                assignment: key,
                expected,
                actual,
            });
        }
        report.matrix_evaluations = report
            .matrix_evaluations
            .checked_add(1)
            .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
    }

    for position in (0..expected_order.len()).rev() {
        let (quantifier, _) = expected_order[position];
        let current = &profile.layers[position];
        let next = &profile.layers[position + 1];
        let rows = checked_power_of_two(position)?;
        for mask in 0..rows {
            let history = prefix_assignment(mask, position);
            let mut low_key = history.clone();
            low_key.push(false);
            let mut high_key = history.clone();
            high_key.push(true);
            let low = *next.values.get(&low_key).ok_or_else(|| {
                GameValueProfileError::MissingLayerAssignment {
                    layer: position + 1,
                    assignment: low_key.clone(),
                }
            })?;
            let high = *next.values.get(&high_key).ok_or_else(|| {
                GameValueProfileError::MissingLayerAssignment {
                    layer: position + 1,
                    assignment: high_key.clone(),
                }
            })?;
            let expected = match quantifier {
                Quantifier::Exists => low || high,
                Quantifier::ForAll => low && high,
            };
            let actual = *current.values.get(&history).ok_or_else(|| {
                GameValueProfileError::MissingLayerAssignment {
                    layer: position,
                    assignment: history.clone(),
                }
            })?;
            if actual != expected {
                return Err(GameValueProfileError::GameValueRecurrenceMismatch {
                    prefix_position: position,
                    assignment: history,
                    expected,
                    actual,
                });
            }
            report.recurrence_checks = report
                .recurrence_checks
                .checked_add(1)
                .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
        }
    }

    let expected_skolem_count = expected_order
        .iter()
        .filter(|(quantifier, _)| *quantifier == Quantifier::Exists)
        .count();
    let expected_herbrand_count = expected_order.len() - expected_skolem_count;
    if profile.skolem_functions.len() != expected_skolem_count {
        return Err(GameValueProfileError::StrategyFunctionCountMismatch {
            owner: Quantifier::Exists,
            expected: expected_skolem_count,
            actual: profile.skolem_functions.len(),
        });
    }
    if profile.herbrand_functions.len() != expected_herbrand_count {
        return Err(GameValueProfileError::StrategyFunctionCountMismatch {
            owner: Quantifier::ForAll,
            expected: expected_herbrand_count,
            actual: profile.herbrand_functions.len(),
        });
    }

    let mut skolem_index = 0usize;
    let mut herbrand_index = 0usize;
    for (position, &(quantifier, variable)) in expected_order.iter().enumerate() {
        let function = match quantifier {
            Quantifier::Exists => {
                let function = &profile.skolem_functions[skolem_index];
                skolem_index += 1;
                function
            }
            Quantifier::ForAll => {
                let function = &profile.herbrand_functions[herbrand_index];
                herbrand_index += 1;
                function
            }
        };
        if function.variable != variable || function.quantifier != quantifier {
            return Err(GameValueProfileError::StrategyFunctionVariableMismatch {
                owner: quantifier,
                index: position,
                expected: variable,
                actual: function.variable,
            });
        }
        let expected_dependencies: Vec<usize> = expected_order[..position]
            .iter()
            .map(|(_, previous_variable)| *previous_variable)
            .collect();
        if function.dependencies != expected_dependencies {
            return Err(GameValueProfileError::StrategyFunctionScopeMismatch {
                variable,
                expected: expected_dependencies,
                actual: function.dependencies.clone(),
            });
        }
        let expected_entries = checked_power_of_two(position)?;
        if function.table.len() as u128 != expected_entries {
            return Err(GameValueProfileError::StrategyFunctionEntryCountMismatch {
                variable,
                expected: expected_entries,
                actual: function.table.len(),
            });
        }

        let layer = &profile.layers[position + 1];
        let chosen_value = quantifier == Quantifier::Exists;
        for mask in 0..expected_entries {
            let history = prefix_assignment(mask, position);
            let mut child_key = history.clone();
            child_key.push(chosen_value);
            let expected = *layer.values.get(&child_key).ok_or_else(|| {
                GameValueProfileError::MissingLayerAssignment {
                    layer: position + 1,
                    assignment: child_key,
                }
            })?;
            let actual = *function.table.get(&history).ok_or_else(|| {
                GameValueProfileError::MissingStrategyFunctionRow {
                    variable,
                    assignment: history.clone(),
                }
            })?;
            if expected != actual {
                return Err(GameValueProfileError::StrategyFunctionValueMismatch {
                    variable,
                    assignment: history,
                    expected,
                    actual,
                });
            }
            report.strategy_function_rows_checked = report
                .strategy_function_rows_checked
                .checked_add(1)
                .ok_or(GameValueProfileError::ResourceAccountingOverflow)?;
        }
    }

    let root_value = *profile.layers[0].values.get(&vec![]).ok_or(
        GameValueProfileError::MissingLayerAssignment {
            layer: 0,
            assignment: vec![],
        },
    )?;
    if profile.truth_value != root_value {
        return Err(GameValueProfileError::TruthValueMismatch {
            expected: root_value,
            actual: profile.truth_value,
        });
    }

    report.layers_checked = profile.layers.len();
    report.layer_assignments_checked = profile.layers.iter().try_fold(0u128, |total, layer| {
        total
            .checked_add(layer.values.len() as u128)
            .ok_or(GameValueProfileError::ResourceAccountingOverflow)
    })?;
    report.matrix_literal_checks = literal_checks;
    report.truth_value = root_value;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::super::decomposition::Literal;
    use super::super::quantified_game_dag::build_qbf_game_proof_dag;
    use super::super::quantified_scope::QuantifierBlock;
    use super::*;

    fn block(quantifier: Quantifier, variables: &[usize]) -> QuantifierBlock {
        QuantifierBlock {
            quantifier,
            variables: variables.to_vec(),
        }
    }

    fn prefix(variable_count: usize, blocks: Vec<QuantifierBlock>) -> ValidatedQuantifierPrefix {
        ValidatedQuantifierPrefix::new(variable_count, blocks, Default::default()).unwrap()
    }

    fn dag_budget() -> QbfGameDagBudget {
        QbfGameDagBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 128,
            max_literals: 512,
            max_proof_nodes: 1 << 12,
            max_builder_state_visits: 1 << 12,
            max_verifier_node_visits: 1 << 12,
            max_residual_literal_visits: 1 << 18,
            max_stored_residual_literals: 1 << 18,
            max_stored_residual_clauses: 1 << 16,
        }
    }

    fn profile_budget() -> GameValueProfileBudget {
        GameValueProfileBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 128,
            max_literals: 512,
            max_total_truth_table_entries: 1 << 12,
            max_dag_path_steps: 1 << 14,
            max_matrix_literal_checks: 1 << 14,
        }
    }

    fn two_variable_formulas() -> Vec<CnfProblem> {
        let literals = [
            Literal::positive(0),
            Literal::negative(0),
            Literal::positive(1),
            Literal::negative(1),
        ];
        let mut formulas = vec![CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![],
        }];
        for left_mask in 0..16usize {
            let left: Vec<_> = (0..4)
                .filter(|index| ((left_mask >> index) & 1) != 0)
                .map(|index| literals[index])
                .collect();
            formulas.push(CnfProblem {
                variables: vec!["x".into(), "y".into()],
                clauses: vec![left.clone()],
            });
            for right_mask in 0..16usize {
                let right: Vec<_> = (0..4)
                    .filter(|index| ((right_mask >> index) & 1) != 0)
                    .map(|index| literals[index])
                    .collect();
                formulas.push(CnfProblem {
                    variables: vec!["x".into(), "y".into()],
                    clauses: vec![left.clone(), right],
                });
            }
        }
        formulas
    }

    fn independent_matrix_value(problem: &CnfProblem, assignment: &[bool]) -> bool {
        problem.clauses.iter().all(|clause| {
            clause
                .iter()
                .any(|literal| assignment[literal.variable] == literal.positive)
        })
    }

    fn independent_game_value(
        problem: &CnfProblem,
        order: &[(Quantifier, usize)],
        assignment: &mut [bool],
        position: usize,
    ) -> bool {
        if position == order.len() {
            return independent_matrix_value(problem, assignment);
        }
        let (quantifier, variable) = order[position];
        assignment[variable] = false;
        let low = independent_game_value(problem, order, assignment, position + 1);
        assignment[variable] = true;
        let high = independent_game_value(problem, order, assignment, position + 1);
        match quantifier {
            Quantifier::Exists => low || high,
            Quantifier::ForAll => low && high,
        }
    }

    fn independent_value_after_prefix(
        problem: &CnfProblem,
        order: &[(Quantifier, usize)],
        history: &[bool],
    ) -> bool {
        let mut assignment = vec![false; problem.variables.len()];
        for (position, &(_, variable)) in order.iter().take(history.len()).enumerate() {
            assignment[variable] = history[position];
        }
        independent_game_value(problem, order, &mut assignment, history.len())
    }

    fn equality_problem() -> CnfProblem {
        CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![
                vec![Literal::negative(0), Literal::positive(1)],
                vec![Literal::positive(0), Literal::negative(1)],
            ],
        }
    }

    #[test]
    fn builds_all_layers_and_both_full_history_function_families() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget()).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget(),
            profile_budget(),
        )
        .unwrap();
        let report =
            verify_game_value_profile(&problem, &prefix, &profile, profile_budget()).unwrap();

        assert!(profile.truth_value);
        assert_eq!(profile.layers.len(), 3);
        assert_eq!(profile.layers[0].values.get(&vec![]), Some(&true));
        assert_eq!(profile.layers[1].values.len(), 2);
        assert_eq!(profile.layers[2].values.len(), 4);
        assert_eq!(profile.skolem_functions.len(), 1);
        assert_eq!(profile.herbrand_functions.len(), 1);
        assert_eq!(profile.skolem_functions[0].dependencies, vec![0]);
        assert_eq!(profile.herbrand_functions[0].dependencies, vec![]);
        assert_eq!(report.layers_checked, 3);
        assert_eq!(report.recurrence_checks, 3);
        assert_eq!(report.matrix_evaluations, 4);
    }

    #[test]
    fn paper_example_separates_win_whenever_possible_from_winning_strategy() {
        // forall x1 exists x2 forall x3:
        // (x1 OR x3) AND (!x1 OR !x2 OR !x3).
        let problem = CnfProblem {
            variables: vec!["x1".into(), "x2".into(), "x3".into()],
            clauses: vec![
                vec![Literal::positive(0), Literal::positive(2)],
                vec![
                    Literal::negative(0),
                    Literal::negative(1),
                    Literal::negative(2),
                ],
            ],
        };
        let prefix = prefix(
            3,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[2]),
            ],
        );
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget()).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget(),
            profile_budget(),
        )
        .unwrap();
        verify_game_value_profile(&problem, &prefix, &profile, profile_budget()).unwrap();

        assert!(!profile.truth_value);
        assert_eq!(
            profile.layers[1].values,
            BTreeMap::from([(vec![false], false), (vec![true], true)])
        );
        assert_eq!(
            profile.layers[2].values,
            BTreeMap::from([
                (vec![false, false], false),
                (vec![false, true], false),
                (vec![true, false], true),
                (vec![true, true], false),
            ])
        );
        // Full-history Skolem/Herbrand functions cover every partial play.
        assert_eq!(
            profile.skolem_functions[0].table,
            BTreeMap::from([(vec![false], false), (vec![true], false)])
        );
        assert_eq!(
            profile.herbrand_functions[0].table,
            BTreeMap::from([
                (vec![false, false], false),
                (vec![false, true], false),
                (vec![true, false], true),
                (vec![true, true], true),
            ])
        );
        assert_eq!(profile.herbrand_functions[0].dependencies, vec![0, 1]);
    }

    #[test]
    fn terminal_subtrees_expand_to_total_game_value_layers() {
        let problem = CnfProblem {
            variables: vec!["u".into(), "x".into(), "v".into()],
            clauses: vec![vec![Literal::positive(1)]],
        };
        let prefix = prefix(
            3,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[2]),
            ],
        );
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget()).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget(),
            profile_budget(),
        )
        .unwrap();
        let report =
            verify_game_value_profile(&problem, &prefix, &profile, profile_budget()).unwrap();
        assert_eq!(profile.layers[3].values.len(), 8);
        assert_eq!(report.matrix_evaluations, 8);
        assert_eq!(profile.skolem_functions[0].table.len(), 2);
        assert_eq!(profile.skolem_functions[1].table.len(), 4);
        assert_eq!(profile.herbrand_functions[0].table.len(), 1);
    }

    #[test]
    fn verifier_rejects_mutated_intermediate_game_value() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget()).unwrap();
        let mut profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget(),
            profile_budget(),
        )
        .unwrap();
        profile.layers[1].values.insert(vec![false], true);
        assert!(matches!(
            verify_game_value_profile(&problem, &prefix, &profile, profile_budget()),
            Err(GameValueProfileError::GameValueRecurrenceMismatch { .. })
        ));
    }

    #[test]
    fn verifier_rejects_mutated_full_history_function_row() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget()).unwrap();
        let mut profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget(),
            profile_budget(),
        )
        .unwrap();
        profile.skolem_functions[0].table.insert(vec![true], false);
        assert!(matches!(
            verify_game_value_profile(&problem, &prefix, &profile, profile_budget()),
            Err(GameValueProfileError::StrategyFunctionValueMismatch { .. })
        ));
    }

    #[test]
    fn rejects_profile_budget_before_dag_traversal() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget()).unwrap();
        let tight = GameValueProfileBudget {
            max_total_truth_table_entries: 9,
            ..profile_budget()
        };
        assert!(matches!(
            extract_game_value_profile_from_dag(&problem, &prefix, &dag, dag_budget(), tight),
            Err(GameValueProfileError::TruthTableEntryBudgetExceeded { .. })
        ));
    }

    #[test]
    fn full_history_skolem_and_herbrand_functions_keep_same_player_history() {
        let problem = equality_problem();
        let existential_prefix = prefix(
            2,
            vec![
                block(Quantifier::Exists, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (existential_dag, _) =
            build_qbf_game_proof_dag(&problem, &existential_prefix, dag_budget()).unwrap();
        let existential_profile = extract_game_value_profile_from_dag(
            &problem,
            &existential_prefix,
            &existential_dag,
            dag_budget(),
            profile_budget(),
        )
        .unwrap();
        verify_game_value_profile(
            &problem,
            &existential_prefix,
            &existential_profile,
            profile_budget(),
        )
        .unwrap();
        assert_eq!(
            existential_profile.skolem_functions[1].dependencies,
            vec![0]
        );
        assert_eq!(
            existential_profile.skolem_functions[1].table,
            BTreeMap::from([(vec![false], false), (vec![true], true)])
        );

        let universal_prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::ForAll, &[1]),
            ],
        );
        let (universal_dag, _) =
            build_qbf_game_proof_dag(&problem, &universal_prefix, dag_budget()).unwrap();
        let universal_profile = extract_game_value_profile_from_dag(
            &problem,
            &universal_prefix,
            &universal_dag,
            dag_budget(),
            profile_budget(),
        )
        .unwrap();
        verify_game_value_profile(
            &problem,
            &universal_prefix,
            &universal_profile,
            profile_budget(),
        )
        .unwrap();
        assert_eq!(
            universal_profile.herbrand_functions[1].dependencies,
            vec![0]
        );
        assert_eq!(
            universal_profile.herbrand_functions[1].table,
            BTreeMap::from([(vec![false], true), (vec![true], false)])
        );
    }

    #[test]
    fn exhaustive_two_variable_qbf_layers_match_independent_game_oracle() {
        let formulas = two_variable_formulas();
        assert_eq!(formulas.len(), 273);
        let orders = [[0usize, 1usize], [1usize, 0usize]];

        for (formula_index, problem) in formulas.iter().enumerate() {
            for order in orders {
                for quantifier_mask in 0..4usize {
                    let blocks: Vec<_> = order
                        .iter()
                        .enumerate()
                        .map(|(position, &variable)| {
                            block(
                                if ((quantifier_mask >> position) & 1) == 0 {
                                    Quantifier::ForAll
                                } else {
                                    Quantifier::Exists
                                },
                                &[variable],
                            )
                        })
                        .collect();
                    let prefix = prefix(2, blocks);
                    let dag_budget = dag_budget();
                    let (dag, _) = build_qbf_game_proof_dag(problem, &prefix, dag_budget).unwrap();
                    let profile = extract_game_value_profile_from_dag(
                        problem,
                        &prefix,
                        &dag,
                        dag_budget,
                        profile_budget(),
                    )
                    .unwrap();
                    verify_game_value_profile(problem, &prefix, &profile, profile_budget())
                        .unwrap();

                    let ordered: Vec<_> = order
                        .iter()
                        .enumerate()
                        .map(|(position, &variable)| {
                            (
                                if ((quantifier_mask >> position) & 1) == 0 {
                                    Quantifier::ForAll
                                } else {
                                    Quantifier::Exists
                                },
                                variable,
                            )
                        })
                        .collect();
                    for (layer_index, layer) in profile.layers.iter().enumerate() {
                        let row_count = checked_power_of_two(layer_index).unwrap();
                        for mask in 0..row_count {
                            let history = prefix_assignment(mask, layer_index);
                            let expected =
                                independent_value_after_prefix(problem, &ordered, &history);
                            assert_eq!(
                                layer.values.get(&history),
                                Some(&expected),
                                "formula {formula_index}, order {order:?}, quantifier mask {quantifier_mask}, layer {layer_index}, history {history:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_out_of_range_source_literal_before_matrix_indexing() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(1)]],
        };
        let prefix = prefix(1, vec![block(Quantifier::Exists, &[0])]);
        assert_eq!(
            validate_profile_source(&problem, &prefix, profile_budget()).unwrap_err(),
            GameValueProfileError::LiteralVariableOutOfRange {
                clause: 0,
                literal: 0,
                variable: 1,
                variable_count: 1,
            }
        );
    }

    #[test]
    fn profile_entry_bound_counts_one_function_per_prefix_variable() {
        // n=2: layers have 1+2+4 entries, functions have 1+2 entries.
        assert_eq!(profile_entry_bound(2).unwrap(), 10);
    }

    #[test]
    fn rejects_excessive_source_clause_count() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![], vec![]],
        };
        let prefix = prefix(1, vec![block(Quantifier::Exists, &[0])]);
        let tight = GameValueProfileBudget {
            max_clauses: 1,
            ..profile_budget()
        };
        assert_eq!(
            validate_profile_source(&problem, &prefix, tight).unwrap_err(),
            GameValueProfileError::ClauseBudgetExceeded {
                required: 2,
                limit: 1,
            }
        );
    }

    #[test]
    fn empty_prefix_has_one_empty_game_value_row_and_no_functions() {
        let problem = CnfProblem {
            variables: vec![],
            clauses: vec![],
        };
        let prefix = prefix(0, vec![]);
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget()).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget(),
            profile_budget(),
        )
        .unwrap();
        verify_game_value_profile(&problem, &prefix, &profile, profile_budget()).unwrap();
        assert!(profile.truth_value);
        assert_eq!(profile.layers.len(), 1);
        assert_eq!(profile.layers[0].values, BTreeMap::from([(vec![], true)]));
        assert!(profile.skolem_functions.is_empty());
        assert!(profile.herbrand_functions.is_empty());
    }
}
