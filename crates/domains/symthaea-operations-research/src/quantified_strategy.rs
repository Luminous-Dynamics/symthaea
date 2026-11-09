// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded Skolem/Herbrand strategy synthesis and independent verification.
//!
//! This is an exhaustive reference engine for small prenex QBF instances. It
//! returns explicit winning functions, not merely a truth value:
//! - TRUE QBF: existential Skolem functions;
//! - FALSE QBF: universal Herbrand functions.
//!
//! It is not a scalable QBF solver, DQBF solver, #QBF counter, or proof system.
//! The verifier enumerates all opponent assignments and checks the original
//! CNF matrix independently of the recursive strategy producer.

use super::decomposition::CnfProblem;
use super::quantified_scope::{
    Quantifier, QuantifierBlock, QuantifierScopeError, ValidatedQuantifierPrefix,
};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// The player whose strategy is being represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyOwner {
    Existential,
    Universal,
}

impl StrategyOwner {
    fn quantifier(self) -> Quantifier {
        match self {
            Self::Existential => Quantifier::Exists,
            Self::Universal => Quantifier::ForAll,
        }
    }

    fn matrix_value(self) -> bool {
        matches!(self, Self::Existential)
    }
}

/// A total Boolean function over an explicitly ordered dependency scope.
///
/// Each table key is an assignment to the declared dependencies, in exactly
/// that order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanStrategyFunction {
    pub variable: usize,
    pub dependencies: Vec<usize>,
    pub table: BTreeMap<Vec<bool>, bool>,
}

/// A finite winning strategy for one player.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WinningStrategy {
    pub owner: StrategyOwner,
    /// Ordered by variable index for deterministic serialization.
    pub functions: Vec<BooleanStrategyFunction>,
}

/// Explicit deterministic resource limits for the bounded QBF reference path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantifiedEvaluationBudget {
    pub max_variables: usize,
    pub max_total_variable_name_bytes: usize,
    pub max_clauses: usize,
    pub max_literals: usize,
    pub max_quantifier_node_visits: u128,
    pub max_opponent_assignments: u128,
    pub max_strategy_table_entries: u128,
}

impl Default for QuantifiedEvaluationBudget {
    fn default() -> Self {
        Self {
            max_variables: 16,
            max_total_variable_name_bytes: 4 * 1024,
            max_clauses: 1 << 16,
            max_literals: 1 << 20,
            max_quantifier_node_visits: 1 << 18,
            max_opponent_assignments: 1 << 16,
            max_strategy_table_entries: 1 << 17,
        }
    }
}

/// Fail-closed errors. A budget failure is never converted into TRUE or FALSE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantifiedStrategyError {
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
    QuantifierNodeBudgetExceeded {
        required: u128,
        limit: u128,
    },
    OpponentAssignmentBudgetExceeded {
        required: u128,
        limit: u128,
    },
    StrategyTableBudgetExceeded {
        required: u128,
        limit: u128,
    },
    DuplicateStrategyFunction {
        variable: usize,
    },
    StrategyFunctionCountMismatch {
        expected: usize,
        actual: usize,
    },
    StrategyFunctionsNotCanonical {
        previous: usize,
        current: usize,
    },
    MissingStrategyFunction {
        variable: usize,
    },
    UnexpectedStrategyFunction {
        variable: usize,
        owner: StrategyOwner,
    },
    FunctionTableEntryCountMismatch {
        variable: usize,
        expected: u128,
        actual: usize,
    },
    FunctionTableKeyWidthMismatch {
        variable: usize,
        expected: usize,
        actual: usize,
    },
    MissingFunctionTableRow {
        variable: usize,
        key: Vec<bool>,
    },
    Counterexample {
        opponent_assignment: Vec<bool>,
        matrix_value: bool,
        required_matrix_value: bool,
    },
    ConflictingStrategyChoice {
        variable: usize,
        dependency_assignment: Vec<bool>,
    },
    InvalidStrategyScope(QuantifierScopeError),
    ResourceAccountingOverflow,
}

/// Resource counters from recursive search, kept separate from verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QuantifiedSearchStats {
    pub quantifier_node_visits: u128,
    pub partial_matrix_checks: u128,
    pub early_true_prunes: u128,
    pub early_false_prunes: u128,
    pub matrix_evaluations: u128,
}

/// Evidence emitted by the independent strategy checker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StrategyVerificationReport {
    pub opponent_assignments_checked: u128,
    pub matrix_evaluations: u128,
    pub strategy_table_entries_checked: u128,
}

/// A bounded QBF result with an explicit winning strategy and checker report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QbfEvaluationResult {
    pub value: bool,
    pub strategy: WinningStrategy,
    pub search: QuantifiedSearchStats,
    pub verification: StrategyVerificationReport,
}

#[derive(Debug, Clone)]
struct PartialEvaluation {
    value: bool,
    existential_functions: BTreeMap<usize, BooleanStrategyFunction>,
    universal_functions: BTreeMap<usize, BooleanStrategyFunction>,
}

fn checked_power_of_two(exponent: usize) -> Result<u128, QuantifiedStrategyError> {
    let shift =
        u32::try_from(exponent).map_err(|_| QuantifiedStrategyError::ResourceAccountingOverflow)?;
    1u128
        .checked_shl(shift)
        .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)
}

fn validate_matrix(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: QuantifiedEvaluationBudget,
) -> Result<(), QuantifiedStrategyError> {
    if problem.variables.len() != prefix.variable_count() {
        return Err(QuantifiedStrategyError::MatrixPrefixVariableCountMismatch {
            matrix: problem.variables.len(),
            prefix: prefix.variable_count(),
        });
    }
    if problem.variables.len() > budget.max_variables {
        return Err(QuantifiedStrategyError::VariableBudgetExceeded {
            required: problem.variables.len(),
            limit: budget.max_variables,
        });
    }

    let name_bytes = problem.variables.iter().try_fold(0usize, |total, name| {
        total
            .checked_add(name.len())
            .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)
    })?;
    if name_bytes > budget.max_total_variable_name_bytes {
        return Err(QuantifiedStrategyError::VariableNameBudgetExceeded {
            required: name_bytes,
            limit: budget.max_total_variable_name_bytes,
        });
    }

    let mut names = HashSet::with_capacity(problem.variables.len());
    for (index, name) in problem.variables.iter().enumerate() {
        if name.is_empty() {
            return Err(QuantifiedStrategyError::EmptyVariableName { index });
        }
        if !names.insert(name.as_str()) {
            return Err(QuantifiedStrategyError::DuplicateVariableName(name.clone()));
        }
    }

    if problem.clauses.len() > budget.max_clauses {
        return Err(QuantifiedStrategyError::ClauseBudgetExceeded {
            required: problem.clauses.len(),
            limit: budget.max_clauses,
        });
    }
    let literal_count = problem.clauses.iter().try_fold(0usize, |total, clause| {
        total
            .checked_add(clause.len())
            .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)
    })?;
    if literal_count > budget.max_literals {
        return Err(QuantifiedStrategyError::LiteralBudgetExceeded {
            required: literal_count,
            limit: budget.max_literals,
        });
    }

    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        for (literal_index, literal) in clause.iter().enumerate() {
            if literal.variable >= problem.variables.len() {
                return Err(QuantifiedStrategyError::LiteralVariableOutOfRange {
                    clause: clause_index,
                    literal: literal_index,
                    variable: literal.variable,
                    variable_count: problem.variables.len(),
                });
            }
        }
    }
    Ok(())
}

fn strategy_table_entry_bound(
    prefix: &ValidatedQuantifierPrefix,
    owner: StrategyOwner,
) -> Result<u128, QuantifiedStrategyError> {
    let target = owner.quantifier();
    prefix.blocks().iter().try_fold(0u128, |total, block| {
        if block.quantifier != target {
            return Ok(total);
        }
        block
            .variables
            .iter()
            .try_fold(total, |subtotal, &variable| {
                let dependencies = match owner {
                    StrategyOwner::Existential => prefix.existential_dependencies(variable),
                    StrategyOwner::Universal => prefix.universal_dependencies(variable),
                }
                .map_err(QuantifiedStrategyError::InvalidStrategyScope)?;
                subtotal
                    .checked_add(checked_power_of_two(dependencies.len())?)
                    .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)
            })
    })
}

fn opponent_assignment_bound(
    prefix: &ValidatedQuantifierPrefix,
    owner: StrategyOwner,
) -> Result<u128, QuantifiedStrategyError> {
    let opponent = match owner {
        StrategyOwner::Existential => Quantifier::ForAll,
        StrategyOwner::Universal => Quantifier::Exists,
    };
    let count = prefix
        .blocks()
        .iter()
        .filter(|block| block.quantifier == opponent)
        .try_fold(0usize, |total, block| {
            total
                .checked_add(block.variables.len())
                .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)
        })?;
    checked_power_of_two(count)
}

fn preflight(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: QuantifiedEvaluationBudget,
) -> Result<(), QuantifiedStrategyError> {
    validate_matrix(problem, prefix, budget)?;

    // The truth value is not known before search. Bound both possible witness
    // directions before entering the recursive producer.
    let existential_entries = strategy_table_entry_bound(prefix, StrategyOwner::Existential)?;
    let universal_entries = strategy_table_entry_bound(prefix, StrategyOwner::Universal)?;
    let required_entries = existential_entries.max(universal_entries);
    if required_entries > budget.max_strategy_table_entries {
        return Err(QuantifiedStrategyError::StrategyTableBudgetExceeded {
            required: required_entries,
            limit: budget.max_strategy_table_entries,
        });
    }

    let existential_opponents = opponent_assignment_bound(prefix, StrategyOwner::Existential)?;
    let universal_opponents = opponent_assignment_bound(prefix, StrategyOwner::Universal)?;
    let required_opponents = existential_opponents.max(universal_opponents);
    if required_opponents > budget.max_opponent_assignments {
        return Err(QuantifiedStrategyError::OpponentAssignmentBudgetExceeded {
            required: required_opponents,
            limit: budget.max_opponent_assignments,
        });
    }
    Ok(())
}

fn matrix_satisfied(problem: &CnfProblem, assignment: &[bool]) -> bool {
    problem.clauses.iter().all(|clause| {
        clause
            .iter()
            .any(|literal| assignment[literal.variable] == literal.positive)
    })
}

/// Return a decisive matrix value when a partial prefix assignment already
/// satisfies every clause or permanently falsifies one clause. None means
/// that some clause still has an unassigned literal and no clause is false.
fn partial_matrix_value(
    problem: &CnfProblem,
    assignment: &[bool],
    variable_positions: &[usize],
    assigned_prefix_len: usize,
) -> Option<bool> {
    let mut every_clause_satisfied = true;
    for clause in &problem.clauses {
        let mut clause_satisfied = false;
        let mut has_unassigned_literal = false;
        for literal in clause {
            if variable_positions[literal.variable] >= assigned_prefix_len {
                has_unassigned_literal = true;
                continue;
            }
            if assignment[literal.variable] == literal.positive {
                clause_satisfied = true;
                break;
            }
        }
        if clause_satisfied {
            continue;
        }
        if !has_unassigned_literal {
            return Some(false);
        }
        every_clause_satisfied = false;
    }
    every_clause_satisfied.then_some(true)
}

/// Build a constant strategy table for a suffix variable, but populate only
/// rows consistent with the opponent assignments already fixed on this search
/// path. Unassigned dependencies are expanded across both values, allowing
/// sibling branches to merge into a total table without overlapping rows.
fn constant_suffix_function(
    variable: usize,
    dependencies: Vec<usize>,
    assignment: &[bool],
    variable_positions: &[usize],
    assigned_prefix_len: usize,
    value: bool,
    budget: QuantifiedEvaluationBudget,
) -> Result<BooleanStrategyFunction, QuantifiedStrategyError> {
    let free_dependency_positions: Vec<usize> = dependencies
        .iter()
        .enumerate()
        .filter_map(|(index, &dependency)| {
            (variable_positions[dependency] >= assigned_prefix_len).then_some(index)
        })
        .collect();
    let row_count = checked_power_of_two(free_dependency_positions.len())?;
    if row_count > budget.max_strategy_table_entries {
        return Err(QuantifiedStrategyError::StrategyTableBudgetExceeded {
            required: row_count,
            limit: budget.max_strategy_table_entries,
        });
    }

    let mut table = BTreeMap::new();
    for mask in 0..row_count {
        let mut key = vec![false; dependencies.len()];
        for (dependency_index, &dependency) in dependencies.iter().enumerate() {
            if variable_positions[dependency] < assigned_prefix_len {
                key[dependency_index] = assignment[dependency];
            }
        }
        for (free_index, &dependency_index) in free_dependency_positions.iter().enumerate() {
            key[dependency_index] = ((mask >> free_index) & 1) != 0;
        }
        table.insert(key, value);
    }

    Ok(BooleanStrategyFunction {
        variable,
        dependencies,
        table,
    })
}

/// Fill the winning player's strategy functions for a suffix whose remaining
/// matrix value is already fixed by the current partial assignment.
fn constant_strategy_for_suffix(
    position: usize,
    order: &[(Quantifier, usize)],
    prefix: &ValidatedQuantifierPrefix,
    owner: StrategyOwner,
    assignment: &[bool],
    variable_positions: &[usize],
    value: bool,
    budget: QuantifiedEvaluationBudget,
) -> Result<BTreeMap<usize, BooleanStrategyFunction>, QuantifiedStrategyError> {
    let mut functions = BTreeMap::new();
    for &(quantifier, variable) in order.iter().skip(position) {
        if quantifier != owner.quantifier() {
            continue;
        }
        let dependencies = match owner {
            StrategyOwner::Existential => prefix.existential_dependencies(variable),
            StrategyOwner::Universal => prefix.universal_dependencies(variable),
        }
        .map_err(QuantifiedStrategyError::InvalidStrategyScope)?;
        let function = constant_suffix_function(
            variable,
            dependencies,
            assignment,
            variable_positions,
            position,
            value,
            budget,
        )?;
        functions.insert(variable, function);
    }
    Ok(functions)
}

fn dependency_key(dependencies: &[usize], assignment: &[bool]) -> Vec<bool> {
    dependencies
        .iter()
        .map(|&dependency| assignment[dependency])
        .collect()
}

fn record_choice(
    functions: &mut BTreeMap<usize, BooleanStrategyFunction>,
    variable: usize,
    dependencies: Vec<usize>,
    assignment: &[bool],
    value: bool,
) -> Result<(), QuantifiedStrategyError> {
    let key = dependency_key(&dependencies, assignment);
    let mut table = BTreeMap::new();
    table.insert(key, value);
    let function = BooleanStrategyFunction {
        variable,
        dependencies,
        table,
    };
    if functions.insert(variable, function).is_some() {
        return Err(QuantifiedStrategyError::DuplicateStrategyFunction { variable });
    }
    Ok(())
}

fn merge_functions(
    mut left: BTreeMap<usize, BooleanStrategyFunction>,
    right: BTreeMap<usize, BooleanStrategyFunction>,
) -> Result<BTreeMap<usize, BooleanStrategyFunction>, QuantifiedStrategyError> {
    for (variable, right_function) in right {
        match left.get_mut(&variable) {
            None => {
                left.insert(variable, right_function);
            }
            Some(left_function) => {
                if left_function.dependencies != right_function.dependencies {
                    return Err(QuantifiedStrategyError::ConflictingStrategyChoice {
                        variable,
                        dependency_assignment: Vec::new(),
                    });
                }
                for (key, value) in right_function.table {
                    if let Some(existing) = left_function.table.get(&key) {
                        if *existing != value {
                            return Err(QuantifiedStrategyError::ConflictingStrategyChoice {
                                variable,
                                dependency_assignment: key,
                            });
                        }
                    } else {
                        left_function.table.insert(key, value);
                    }
                }
            }
        }
    }
    Ok(left)
}

fn search(
    position: usize,
    order: &[(Quantifier, usize)],
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    assignment: &mut [bool],
    variable_positions: &[usize],
    budget: QuantifiedEvaluationBudget,
    stats: &mut QuantifiedSearchStats,
) -> Result<PartialEvaluation, QuantifiedStrategyError> {
    stats.quantifier_node_visits = stats
        .quantifier_node_visits
        .checked_add(1)
        .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)?;
    if stats.quantifier_node_visits > budget.max_quantifier_node_visits {
        return Err(QuantifiedStrategyError::QuantifierNodeBudgetExceeded {
            required: stats.quantifier_node_visits,
            limit: budget.max_quantifier_node_visits,
        });
    }

    if position < order.len() {
        stats.partial_matrix_checks = stats
            .partial_matrix_checks
            .checked_add(1)
            .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)?;
        if let Some(value) = partial_matrix_value(problem, assignment, variable_positions, position)
        {
            if value {
                stats.early_true_prunes = stats
                    .early_true_prunes
                    .checked_add(1)
                    .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)?;
                let existential_functions = constant_strategy_for_suffix(
                    position,
                    order,
                    prefix,
                    StrategyOwner::Existential,
                    assignment,
                    variable_positions,
                    false,
                    budget,
                )?;
                return Ok(PartialEvaluation {
                    value: true,
                    existential_functions,
                    universal_functions: BTreeMap::new(),
                });
            }

            stats.early_false_prunes = stats
                .early_false_prunes
                .checked_add(1)
                .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)?;
            let universal_functions = constant_strategy_for_suffix(
                position,
                order,
                prefix,
                StrategyOwner::Universal,
                assignment,
                variable_positions,
                false,
                budget,
            )?;
            return Ok(PartialEvaluation {
                value: false,
                existential_functions: BTreeMap::new(),
                universal_functions,
            });
        }
    }

    if position == order.len() {
        stats.matrix_evaluations = stats
            .matrix_evaluations
            .checked_add(1)
            .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)?;
        return Ok(PartialEvaluation {
            value: matrix_satisfied(problem, assignment),
            existential_functions: BTreeMap::new(),
            universal_functions: BTreeMap::new(),
        });
    }

    let (quantifier, variable) = order[position];
    match quantifier {
        Quantifier::Exists => {
            assignment[variable] = false;
            let mut left = search(
                position + 1,
                order,
                problem,
                prefix,
                assignment,
                variable_positions,
                budget,
                stats,
            )?;
            if left.value {
                let dependencies = prefix
                    .existential_dependencies(variable)
                    .map_err(QuantifiedStrategyError::InvalidStrategyScope)?;
                record_choice(
                    &mut left.existential_functions,
                    variable,
                    dependencies,
                    assignment,
                    false,
                )?;
                return Ok(left);
            }

            assignment[variable] = true;
            let mut right = search(
                position + 1,
                order,
                problem,
                prefix,
                assignment,
                variable_positions,
                budget,
                stats,
            )?;
            if right.value {
                let dependencies = prefix
                    .existential_dependencies(variable)
                    .map_err(QuantifiedStrategyError::InvalidStrategyScope)?;
                record_choice(
                    &mut right.existential_functions,
                    variable,
                    dependencies,
                    assignment,
                    true,
                )?;
                return Ok(right);
            }

            let universal_functions =
                merge_functions(left.universal_functions, right.universal_functions)?;
            Ok(PartialEvaluation {
                value: false,
                existential_functions: BTreeMap::new(),
                universal_functions,
            })
        }
        Quantifier::ForAll => {
            assignment[variable] = false;
            let mut left = search(
                position + 1,
                order,
                problem,
                prefix,
                assignment,
                variable_positions,
                budget,
                stats,
            )?;
            if !left.value {
                let dependencies = prefix
                    .universal_dependencies(variable)
                    .map_err(QuantifiedStrategyError::InvalidStrategyScope)?;
                record_choice(
                    &mut left.universal_functions,
                    variable,
                    dependencies,
                    assignment,
                    false,
                )?;
                return Ok(left);
            }

            assignment[variable] = true;
            let mut right = search(
                position + 1,
                order,
                problem,
                prefix,
                assignment,
                variable_positions,
                budget,
                stats,
            )?;
            if !right.value {
                let dependencies = prefix
                    .universal_dependencies(variable)
                    .map_err(QuantifiedStrategyError::InvalidStrategyScope)?;
                record_choice(
                    &mut right.universal_functions,
                    variable,
                    dependencies,
                    assignment,
                    true,
                )?;
                return Ok(right);
            }

            let existential_functions =
                merge_functions(left.existential_functions, right.existential_functions)?;
            Ok(PartialEvaluation {
                value: true,
                existential_functions,
                universal_functions: BTreeMap::new(),
            })
        }
    }
}

fn ordered_prefix(prefix: &ValidatedQuantifierPrefix) -> Vec<(Quantifier, usize)> {
    prefix
        .blocks()
        .iter()
        .flat_map(|block: &QuantifierBlock| {
            block
                .variables
                .iter()
                .copied()
                .map(move |variable| (block.quantifier, variable))
        })
        .collect()
}

fn variable_positions(variable_count: usize, order: &[(Quantifier, usize)]) -> Vec<usize> {
    let mut positions = vec![usize::MAX; variable_count];
    for (position, &(_, variable)) in order.iter().enumerate() {
        positions[variable] = position;
    }
    positions
}

/// Evaluate a bounded prenex QBF and extract an explicit winning strategy.
///
/// Search uses recursive finite-game semantics. Before returning, this
/// function calls the independent exhaustive checker on the extracted
/// Skolem/Herbrand functions. A budget or verification failure returns an
/// error rather than a truth value.
pub fn evaluate_qbf_with_strategy(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: QuantifiedEvaluationBudget,
) -> Result<QbfEvaluationResult, QuantifiedStrategyError> {
    preflight(problem, prefix, budget)?;
    let order = ordered_prefix(prefix);
    let positions = variable_positions(problem.variables.len(), &order);
    let mut assignment = vec![false; problem.variables.len()];
    let mut search_stats = QuantifiedSearchStats::default();

    let partial = search(
        0,
        &order,
        problem,
        prefix,
        &mut assignment,
        &positions,
        budget,
        &mut search_stats,
    )?;

    let (owner, functions) = if partial.value {
        (
            StrategyOwner::Existential,
            partial.existential_functions.into_values().collect(),
        )
    } else {
        (
            StrategyOwner::Universal,
            partial.universal_functions.into_values().collect(),
        )
    };

    let strategy = WinningStrategy { owner, functions };
    let verification = verify_winning_strategy(problem, prefix, &strategy, budget)?;

    Ok(QbfEvaluationResult {
        value: partial.value,
        strategy,
        search: search_stats,
        verification,
    })
}

/// Independently verify a Skolem or Herbrand strategy against the source CNF.
///
/// The checker enumerates every assignment to the opposing player and fills
/// controlled variables in prefix order. It does not call recursive search.
pub fn verify_winning_strategy(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    strategy: &WinningStrategy,
    budget: QuantifiedEvaluationBudget,
) -> Result<StrategyVerificationReport, QuantifiedStrategyError> {
    validate_matrix(problem, prefix, budget)?;

    let owner_quantifier = strategy.owner.quantifier();
    let opponent_quantifier = match strategy.owner {
        StrategyOwner::Existential => Quantifier::ForAll,
        StrategyOwner::Universal => Quantifier::Exists,
    };

    let mut expected_controlled = Vec::new();
    let mut opponent_variables = Vec::new();
    for block in prefix.blocks() {
        for &variable in &block.variables {
            if block.quantifier == owner_quantifier {
                expected_controlled.push(variable);
            } else if block.quantifier == opponent_quantifier {
                opponent_variables.push(variable);
            }
        }
    }

    if strategy.functions.len() > expected_controlled.len() {
        return Err(QuantifiedStrategyError::StrategyFunctionCountMismatch {
            expected: expected_controlled.len(),
            actual: strategy.functions.len(),
        });
    }

    let mut functions = BTreeMap::new();
    let mut previous_variable = None;
    for function in &strategy.functions {
        if let Some(previous) = previous_variable {
            if function.variable <= previous {
                return Err(QuantifiedStrategyError::StrategyFunctionsNotCanonical {
                    previous,
                    current: function.variable,
                });
            }
        }
        previous_variable = Some(function.variable);
        let variable = function.variable;
        let actual_quantifier = prefix
            .blocks()
            .iter()
            .find(|block| block.variables.contains(&variable))
            .map(|block| block.quantifier);
        if actual_quantifier != Some(owner_quantifier) {
            return Err(QuantifiedStrategyError::UnexpectedStrategyFunction {
                variable,
                owner: strategy.owner,
            });
        }
        if functions.insert(variable, function).is_some() {
            return Err(QuantifiedStrategyError::DuplicateStrategyFunction { variable });
        }

        match strategy.owner {
            StrategyOwner::Existential => {
                prefix.validate_skolem_dependencies(variable, &function.dependencies)
            }
            StrategyOwner::Universal => {
                prefix.validate_herbrand_dependencies(variable, &function.dependencies)
            }
        }
        .map_err(QuantifiedStrategyError::InvalidStrategyScope)?;
    }

    for &variable in &expected_controlled {
        if !functions.contains_key(&variable) {
            return Err(QuantifiedStrategyError::MissingStrategyFunction { variable });
        }
    }

    let mut table_entry_count = 0u128;
    for (&variable, function) in &functions {
        let expected_entries = checked_power_of_two(function.dependencies.len())?;
        if function.table.len() as u128 != expected_entries {
            return Err(QuantifiedStrategyError::FunctionTableEntryCountMismatch {
                variable,
                expected: expected_entries,
                actual: function.table.len(),
            });
        }
        for key in function.table.keys() {
            if key.len() != function.dependencies.len() {
                return Err(QuantifiedStrategyError::FunctionTableKeyWidthMismatch {
                    variable,
                    expected: function.dependencies.len(),
                    actual: key.len(),
                });
            }
        }
        table_entry_count = table_entry_count
            .checked_add(expected_entries)
            .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)?;
    }
    if table_entry_count > budget.max_strategy_table_entries {
        return Err(QuantifiedStrategyError::StrategyTableBudgetExceeded {
            required: table_entry_count,
            limit: budget.max_strategy_table_entries,
        });
    }

    let opponent_assignments = checked_power_of_two(opponent_variables.len())?;
    if opponent_assignments > budget.max_opponent_assignments {
        return Err(QuantifiedStrategyError::OpponentAssignmentBudgetExceeded {
            required: opponent_assignments,
            limit: budget.max_opponent_assignments,
        });
    }

    let mut used_rows: BTreeMap<usize, BTreeSet<Vec<bool>>> = BTreeMap::new();
    let mut assignment = vec![false; problem.variables.len()];
    let mut matrix_evaluations = 0u128;

    for mask in 0..opponent_assignments {
        for (bit, &variable) in opponent_variables.iter().enumerate() {
            assignment[variable] = ((mask >> bit) & 1) == 1;
        }

        for block in prefix.blocks() {
            if block.quantifier == owner_quantifier {
                for &variable in &block.variables {
                    let function = functions
                        .get(&variable)
                        .ok_or(QuantifiedStrategyError::MissingStrategyFunction { variable })?;
                    let key = dependency_key(&function.dependencies, &assignment);
                    let Some(&choice) = function.table.get(&key) else {
                        return Err(QuantifiedStrategyError::MissingFunctionTableRow {
                            variable,
                            key,
                        });
                    };
                    used_rows.entry(variable).or_default().insert(key);
                    assignment[variable] = choice;
                }
            }
        }

        matrix_evaluations = matrix_evaluations
            .checked_add(1)
            .ok_or(QuantifiedStrategyError::ResourceAccountingOverflow)?;
        let matrix_value = matrix_satisfied(problem, &assignment);
        let required_value = strategy.owner.matrix_value();
        if matrix_value != required_value {
            let opponent_assignment = opponent_variables
                .iter()
                .map(|&variable| assignment[variable])
                .collect();
            return Err(QuantifiedStrategyError::Counterexample {
                opponent_assignment,
                matrix_value,
                required_matrix_value: required_value,
            });
        }
    }

    for (&variable, function) in &functions {
        let used = used_rows.get(&variable).cloned().unwrap_or_default();
        let declared: BTreeSet<Vec<bool>> = function.table.keys().cloned().collect();
        if used != declared {
            let missing_or_extra = declared
                .symmetric_difference(&used)
                .next()
                .cloned()
                .unwrap_or_default();
            return Err(QuantifiedStrategyError::MissingFunctionTableRow {
                variable,
                key: missing_or_extra,
            });
        }
    }

    Ok(StrategyVerificationReport {
        opponent_assignments_checked: opponent_assignments,
        matrix_evaluations,
        strategy_table_entries_checked: table_entry_count,
    })
}

#[cfg(test)]
mod tests {
    use super::super::decomposition::Literal;
    use super::*;

    fn budget() -> QuantifiedEvaluationBudget {
        QuantifiedEvaluationBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 64,
            max_literals: 256,
            max_quantifier_node_visits: 1 << 12,
            max_opponent_assignments: 1 << 8,
            max_strategy_table_entries: 1 << 10,
        }
    }

    fn block(quantifier: Quantifier, variables: &[usize]) -> QuantifierBlock {
        QuantifierBlock {
            quantifier,
            variables: variables.to_vec(),
        }
    }

    fn make_prefix(blocks: Vec<QuantifierBlock>, variables: usize) -> ValidatedQuantifierPrefix {
        ValidatedQuantifierPrefix::new(variables, blocks, Default::default()).unwrap()
    }

    fn equality_problem() -> CnfProblem {
        // x <-> y, represented as (!x OR y) AND (x OR !y).
        CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![
                vec![Literal::negative(0), Literal::positive(1)],
                vec![Literal::positive(0), Literal::negative(1)],
            ],
        }
    }

    fn matrix_oracle(problem: &CnfProblem, mask: usize) -> bool {
        problem.clauses.iter().all(|clause| {
            clause.iter().any(|literal| {
                let value = ((mask >> literal.variable) & 1) != 0;
                value == literal.positive
            })
        })
    }

    fn independent_qbf_oracle(problem: &CnfProblem, prefix: &ValidatedQuantifierPrefix) -> bool {
        let order: Vec<(Quantifier, usize)> = prefix
            .blocks()
            .iter()
            .flat_map(|block| {
                block
                    .variables
                    .iter()
                    .copied()
                    .map(move |variable| (block.quantifier, variable))
            })
            .collect();

        fn visit(
            position: usize,
            order: &[(Quantifier, usize)],
            problem: &CnfProblem,
            assignment: &mut [bool],
        ) -> bool {
            if position == order.len() {
                let mask = assignment
                    .iter()
                    .enumerate()
                    .fold(0usize, |value, (index, bit)| {
                        value | (usize::from(*bit) << index)
                    });
                return matrix_oracle(problem, mask);
            }
            let (quantifier, variable) = order[position];
            assignment[variable] = false;
            let low = visit(position + 1, order, problem, assignment);
            assignment[variable] = true;
            let high = visit(position + 1, order, problem, assignment);
            match quantifier {
                Quantifier::Exists => low || high,
                Quantifier::ForAll => low && high,
            }
        }

        visit(
            0,
            &order,
            problem,
            &mut vec![false; problem.variables.len()],
        )
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
        for left_mask in 0..16 {
            let left: Vec<_> = (0..4)
                .filter(|index| ((left_mask >> index) & 1) != 0)
                .map(|index| literals[index])
                .collect();
            formulas.push(CnfProblem {
                variables: vec!["x".into(), "y".into()],
                clauses: vec![left.clone()],
            });
            for right_mask in 0..16 {
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

    #[test]
    fn partial_matrix_pruning_extracts_total_suffix_strategy_tables() {
        let problem = CnfProblem {
            variables: vec!["u".into(), "x".into(), "v".into(), "z".into()],
            clauses: vec![vec![Literal::positive(1)]],
        };
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[2]),
                block(Quantifier::Exists, &[3]),
            ],
            4,
        );
        let result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        assert!(result.value);
        assert_eq!(result.search.early_true_prunes, 2);
        assert_eq!(result.search.early_false_prunes, 2);
        assert!(result.search.quantifier_node_visits < 31);
        let x = result
            .strategy
            .functions
            .iter()
            .find(|function| function.variable == 1)
            .unwrap();
        let z = result
            .strategy
            .functions
            .iter()
            .find(|function| function.variable == 3)
            .unwrap();
        assert_eq!(x.dependencies, vec![0]);
        assert_eq!(
            x.table,
            BTreeMap::from([(vec![false], true), (vec![true], true)])
        );
        assert_eq!(z.dependencies, vec![0, 2]);
        assert_eq!(
            z.table,
            BTreeMap::from([
                (vec![false, false], false),
                (vec![false, true], false),
                (vec![true, false], false),
                (vec![true, true], false),
            ])
        );
        assert_eq!(result.verification.opponent_assignments_checked, 4);
    }

    #[test]
    fn forall_exists_equality_extracts_skolem_identity() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        assert!(result.value);
        assert_eq!(result.strategy.owner, StrategyOwner::Existential);
        let function = result
            .strategy
            .functions
            .iter()
            .find(|function| function.variable == 1)
            .unwrap();
        assert_eq!(function.dependencies, vec![0]);
        assert_eq!(function.table.get(&vec![false]), Some(&false));
        assert_eq!(function.table.get(&vec![true]), Some(&true));
        assert_eq!(result.verification.opponent_assignments_checked, 2);
    }

    #[test]
    fn exists_forall_equality_extracts_herbrand_identity() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[0]),
            ],
            2,
        );
        let result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        assert!(!result.value);
        assert_eq!(result.strategy.owner, StrategyOwner::Universal);
        let function = result
            .strategy
            .functions
            .iter()
            .find(|function| function.variable == 0)
            .unwrap();
        assert_eq!(function.dependencies, vec![1]);
        assert_eq!(function.table.get(&vec![false]), Some(&true));
        assert_eq!(function.table.get(&vec![true]), Some(&false));
    }

    #[test]
    fn forall_forall_equality_returns_constant_herbrand_counterstrategy() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::ForAll, &[1]),
            ],
            2,
        );
        let result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        assert!(!result.value);
        assert_eq!(result.strategy.owner, StrategyOwner::Universal);
        let x = result
            .strategy
            .functions
            .iter()
            .find(|f| f.variable == 0)
            .unwrap();
        let y = result
            .strategy
            .functions
            .iter()
            .find(|f| f.variable == 1)
            .unwrap();
        assert_eq!(x.table.get(&vec![]), Some(&false));
        assert_eq!(y.table.get(&vec![]), Some(&true));
    }

    #[test]
    fn exists_exists_equality_returns_constant_skolem_witness() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::Exists, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        assert!(result.value);
        let x = result
            .strategy
            .functions
            .iter()
            .find(|f| f.variable == 0)
            .unwrap();
        let y = result
            .strategy
            .functions
            .iter()
            .find(|f| f.variable == 1)
            .unwrap();
        assert_eq!(x.table.get(&vec![]), Some(&false));
        assert_eq!(y.table.get(&vec![]), Some(&false));
    }

    #[test]
    fn multi_variable_universal_block_is_a_joint_dependency_scope() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into(), "z".into()],
            // z = x XOR y.
            clauses: vec![
                vec![
                    Literal::positive(0),
                    Literal::positive(1),
                    Literal::negative(2),
                ],
                vec![
                    Literal::negative(0),
                    Literal::negative(1),
                    Literal::negative(2),
                ],
                vec![
                    Literal::positive(0),
                    Literal::negative(1),
                    Literal::positive(2),
                ],
                vec![
                    Literal::negative(0),
                    Literal::positive(1),
                    Literal::positive(2),
                ],
            ],
        };
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0, 1]),
                block(Quantifier::Exists, &[2]),
            ],
            3,
        );
        let result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        assert!(result.value);
        let z = result
            .strategy
            .functions
            .iter()
            .find(|f| f.variable == 2)
            .unwrap();
        assert_eq!(z.dependencies, vec![0, 1]);
        assert_eq!(z.table.len(), 4);
    }

    #[test]
    fn empty_prefix_over_empty_universe_has_empty_skolem_strategy() {
        let problem = CnfProblem {
            variables: vec![],
            clauses: vec![],
        };
        let prefix = make_prefix(vec![], 0);
        let result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        assert!(result.value);
        assert!(result.strategy.functions.is_empty());
        assert_eq!(result.verification.opponent_assignments_checked, 1);
    }

    #[test]
    fn empty_clause_is_false_and_produces_herbrand_strategy() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![]],
        };
        let prefix = make_prefix(vec![block(Quantifier::ForAll, &[0])], 1);
        let result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        assert!(!result.value);
        assert_eq!(result.strategy.owner, StrategyOwner::Universal);
        assert_eq!(
            result.strategy.functions[0].table.get(&vec![]),
            Some(&false)
        );
    }

    #[test]
    fn independent_checker_rejects_tampered_skolem_function() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let mut result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        let y = result
            .strategy
            .functions
            .iter_mut()
            .find(|function| function.variable == 1)
            .unwrap();
        y.table.insert(vec![false], true);
        y.table.insert(vec![true], false);
        assert!(matches!(
            verify_winning_strategy(&problem, &prefix, &result.strategy, budget()),
            Err(QuantifiedStrategyError::Counterexample { .. })
        ));
    }

    #[test]
    fn independent_checker_rejects_incomplete_strategy_table() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let mut result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        let y = result
            .strategy
            .functions
            .iter_mut()
            .find(|function| function.variable == 1)
            .unwrap();
        y.table.remove(&vec![true]);
        assert!(matches!(
            verify_winning_strategy(&problem, &prefix, &result.strategy, budget()),
            Err(QuantifiedStrategyError::FunctionTableEntryCountMismatch { .. })
        ));
    }

    #[test]
    fn independent_checker_rejects_missing_strategy_function() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let mut result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        result.strategy.functions.clear();
        assert_eq!(
            verify_winning_strategy(&problem, &prefix, &result.strategy, budget()).unwrap_err(),
            QuantifiedStrategyError::MissingStrategyFunction { variable: 1 }
        );
    }

    #[test]
    fn independent_checker_rejects_unexpected_opponent_function() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let mut result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        result.strategy.functions[0].variable = 0;
        assert_eq!(
            verify_winning_strategy(&problem, &prefix, &result.strategy, budget()).unwrap_err(),
            QuantifiedStrategyError::UnexpectedStrategyFunction {
                variable: 0,
                owner: StrategyOwner::Existential
            }
        );
    }

    #[test]
    fn independent_checker_rejects_oversized_strategy_function_list() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let mut result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        let duplicate = result.strategy.functions[0].clone();
        result.strategy.functions.push(duplicate);
        assert_eq!(
            verify_winning_strategy(&problem, &prefix, &result.strategy, budget()).unwrap_err(),
            QuantifiedStrategyError::StrategyFunctionCountMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn independent_checker_rejects_noncanonical_function_order() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![],
        };
        let prefix = make_prefix(vec![block(Quantifier::Exists, &[1, 0])], 2);
        let mut result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
        result.strategy.functions.reverse();
        assert!(matches!(
            verify_winning_strategy(&problem, &prefix, &result.strategy, budget()),
            Err(QuantifiedStrategyError::StrategyFunctionsNotCanonical { .. })
        ));
    }

    #[test]
    fn generated_three_variable_qbf_prefixes_match_independent_game_oracle() {
        let mut state = 0xbb67_ae85_84ca_a73bu64;
        let mut next = || {
            state = state
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            state
        };
        let literals = [
            Literal::positive(0),
            Literal::negative(0),
            Literal::positive(1),
            Literal::negative(1),
            Literal::positive(2),
            Literal::negative(2),
        ];
        let orders = [
            [0usize, 1usize, 2usize],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];

        for formula_index in 0..64 {
            let clause_count = (next() % 5) as usize;
            let mut clauses = Vec::with_capacity(clause_count);
            for _ in 0..clause_count {
                let literal_count = (next() % 5) as usize;
                let mut clause = Vec::with_capacity(literal_count);
                for _ in 0..literal_count {
                    clause.push(literals[(next() % literals.len() as u64) as usize]);
                }
                clauses.push(clause);
            }
            let problem = CnfProblem {
                variables: vec!["x".into(), "y".into(), "z".into()],
                clauses,
            };

            for order in orders {
                for quantifier_mask in 0..8usize {
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
                    let prefix = make_prefix(blocks, 3);
                    let expected = independent_qbf_oracle(&problem, &prefix);
                    let result = evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap();
                    assert_eq!(
                        result.value, expected,
                        "formula {formula_index}, order {order:?}, quantifier mask {quantifier_mask}"
                    );
                    let checked =
                        verify_winning_strategy(&problem, &prefix, &result.strategy, budget())
                            .unwrap();
                    assert_eq!(
                        checked.opponent_assignments_checked,
                        result.verification.opponent_assignments_checked
                    );
                }
            }
        }
    }

    #[test]
    fn quantifier_node_budget_exhaustion_is_not_a_truth_value() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let tight = QuantifiedEvaluationBudget {
            max_quantifier_node_visits: 1,
            ..budget()
        };
        assert!(matches!(
            evaluate_qbf_with_strategy(&problem, &prefix, tight),
            Err(QuantifiedStrategyError::QuantifierNodeBudgetExceeded { .. })
        ));
    }

    #[test]
    fn strategy_table_budget_exhaustion_is_rejected_before_search() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let tight = QuantifiedEvaluationBudget {
            max_strategy_table_entries: 1,
            ..budget()
        };
        assert!(matches!(
            evaluate_qbf_with_strategy(&problem, &prefix, tight),
            Err(QuantifiedStrategyError::StrategyTableBudgetExceeded { .. })
        ));
    }

    #[test]
    fn opponent_assignment_budget_exhaustion_is_rejected() {
        let problem = equality_problem();
        let prefix = make_prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let tight = QuantifiedEvaluationBudget {
            max_opponent_assignments: 1,
            ..budget()
        };
        assert!(matches!(
            evaluate_qbf_with_strategy(&problem, &prefix, tight),
            Err(QuantifiedStrategyError::OpponentAssignmentBudgetExceeded { .. })
        ));
    }

    #[test]
    fn matrix_prefix_variable_universe_mismatch_is_rejected() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![],
        };
        let prefix = make_prefix(vec![], 0);
        assert_eq!(
            evaluate_qbf_with_strategy(&problem, &prefix, budget()).unwrap_err(),
            QuantifiedStrategyError::MatrixPrefixVariableCountMismatch {
                matrix: 1,
                prefix: 0
            }
        );
    }

    #[test]
    fn exhaustive_small_cnf_and_quantifier_prefixes_match_independent_oracle() {
        let formulas = two_variable_formulas();
        assert_eq!(formulas.len(), 273);

        for (formula_index, problem) in formulas.iter().enumerate() {
            for order in [[0usize, 1usize], [1usize, 0usize]] {
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
                    let prefix = make_prefix(blocks, 2);
                    let expected = independent_qbf_oracle(problem, &prefix);
                    let result = evaluate_qbf_with_strategy(problem, &prefix, budget()).unwrap();
                    assert_eq!(
                        result.value, expected,
                        "formula {formula_index}, order {order:?}, quantifier mask {quantifier_mask}"
                    );
                    let checked =
                        verify_winning_strategy(problem, &prefix, &result.strategy, budget())
                            .unwrap();
                    assert_eq!(
                        checked.opponent_assignments_checked,
                        result.verification.opponent_assignments_checked
                    );
                    assert_eq!(
                        checked.matrix_evaluations,
                        checked.opponent_assignments_checked
                    );
                }
            }
        }
    }
}
