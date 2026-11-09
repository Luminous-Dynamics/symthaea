// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact bounded #QBF counting by enumerating complete Skolem strategy
//! profiles. This is intentionally distinct from QBF decision, #SAT model
//! counting and projected model counting.
//!
//! For each existential variable, one total truth table is defined over the
//! preceding universal variables in the validated prenex prefix. A profile is
//! winning iff the original CNF matrix is true for every universal assignment.
//! Each distinct tuple of truth tables is counted once.
//!
//! This is a small-instance reference oracle, not a scalable #QBF solver or a
//! proof system. Candidate-profile and universal-assignment spaces are
//! preflighted before enumeration, and budget exhaustion never returns a
//! partial count.

use super::decomposition::CnfProblem;
use super::quantified_scope::{Quantifier, QuantifierScopeError, ValidatedQuantifierPrefix};

/// Explicit resource bounds for finite #QBF reference enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QbfStrategyCountBudget {
    pub max_variables: usize,
    pub max_total_variable_name_bytes: usize,
    pub max_clauses: usize,
    pub max_literals: usize,
    pub max_total_strategy_table_bits: usize,
    pub max_candidate_profiles: u128,
    pub max_universal_assignments: u128,
    pub max_total_matrix_evaluations: u128,
}

impl Default for QbfStrategyCountBudget {
    fn default() -> Self {
        Self {
            max_variables: 16,
            max_total_variable_name_bytes: 4 * 1024,
            max_clauses: 1 << 16,
            max_literals: 1 << 20,
            max_total_strategy_table_bits: 20,
            max_candidate_profiles: 1 << 20,
            max_universal_assignments: 1 << 16,
            max_total_matrix_evaluations: 1 << 24,
        }
    }
}

/// An exact count over the finite candidate-profile space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QbfStrategyCountReport {
    /// Number of distinct winning existential strategy profiles.
    pub winning_strategy_profiles: u128,
    /// Size of the complete candidate profile space, before matrix checks.
    pub candidate_strategy_profiles: u128,
    /// Candidate profiles completely considered (equal to candidate space on
    /// every successful return).
    pub candidate_profiles_checked: u128,
    /// Profiles rejected after encountering their first losing universal row.
    pub profiles_rejected_early: u128,
    /// Total universal assignments checked across candidate profiles.
    pub universal_assignments_checked: u128,
    /// Number of original CNF evaluations (one per checked universal row).
    pub matrix_evaluations: u128,
    /// Sum of the truth-table bits for all existential strategy functions.
    pub strategy_table_bits: usize,
    /// Number of universal variables in the source prefix.
    pub universal_variable_count: usize,
    /// Number of existential variables in the source prefix.
    pub existential_variable_count: usize,
}

/// Fail-closed errors. No approximate or partial count is returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QbfStrategyCountError {
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
    StrategyTableBitBudgetExceeded {
        required: usize,
        limit: usize,
    },
    CandidateProfileSpaceOverflow {
        table_bits: usize,
    },
    CandidateProfileBudgetExceeded {
        required: u128,
        limit: u128,
    },
    UniversalAssignmentSpaceOverflow {
        variable_count: usize,
    },
    UniversalAssignmentBudgetExceeded {
        required: u128,
        limit: u128,
    },
    MatrixEvaluationBudgetExceeded {
        required: u128,
        limit: u128,
    },
    InvalidStrategyScope(QuantifierScopeError),
    ResourceAccountingOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StrategyFunctionLayout {
    variable: usize,
    dependencies: Vec<usize>,
    table_entries: u128,
    bit_offset: usize,
}

fn checked_power_of_two(exponent: usize) -> Result<u128, QbfStrategyCountError> {
    let shift = u32::try_from(exponent).map_err(|_| {
        QbfStrategyCountError::CandidateProfileSpaceOverflow {
            table_bits: exponent,
        }
    })?;
    1u128
        .checked_shl(shift)
        .ok_or(QbfStrategyCountError::CandidateProfileSpaceOverflow {
            table_bits: exponent,
        })
}

fn validate_source(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: QbfStrategyCountBudget,
) -> Result<(), QbfStrategyCountError> {
    if problem.variables.len() != prefix.variable_count() {
        return Err(QbfStrategyCountError::MatrixPrefixVariableCountMismatch {
            matrix: problem.variables.len(),
            prefix: prefix.variable_count(),
        });
    }
    if problem.variables.len() > budget.max_variables {
        return Err(QbfStrategyCountError::VariableBudgetExceeded {
            required: problem.variables.len(),
            limit: budget.max_variables,
        });
    }

    let name_bytes = problem.variables.iter().try_fold(0usize, |total, name| {
        total
            .checked_add(name.len())
            .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)
    })?;
    if name_bytes > budget.max_total_variable_name_bytes {
        return Err(QbfStrategyCountError::VariableNameBudgetExceeded {
            required: name_bytes,
            limit: budget.max_total_variable_name_bytes,
        });
    }

    let mut names = std::collections::BTreeSet::new();
    for (index, name) in problem.variables.iter().enumerate() {
        if name.is_empty() {
            return Err(QbfStrategyCountError::EmptyVariableName { index });
        }
        if !names.insert(name.as_str()) {
            return Err(QbfStrategyCountError::DuplicateVariableName(name.clone()));
        }
    }

    if problem.clauses.len() > budget.max_clauses {
        return Err(QbfStrategyCountError::ClauseBudgetExceeded {
            required: problem.clauses.len(),
            limit: budget.max_clauses,
        });
    }
    let literal_count = problem.clauses.iter().try_fold(0usize, |total, clause| {
        total
            .checked_add(clause.len())
            .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)
    })?;
    if literal_count > budget.max_literals {
        return Err(QbfStrategyCountError::LiteralBudgetExceeded {
            required: literal_count,
            limit: budget.max_literals,
        });
    }
    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        for (literal_index, literal) in clause.iter().enumerate() {
            if literal.variable >= problem.variables.len() {
                return Err(QbfStrategyCountError::LiteralVariableOutOfRange {
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

fn strategy_layout(
    prefix: &ValidatedQuantifierPrefix,
) -> Result<(Vec<StrategyFunctionLayout>, Vec<usize>, usize), QbfStrategyCountError> {
    let mut functions = Vec::new();
    let mut universals = Vec::new();
    let mut table_bit_offset = 0usize;

    for block in prefix.blocks() {
        for &variable in &block.variables {
            match block.quantifier {
                Quantifier::ForAll => universals.push(variable),
                Quantifier::Exists => {
                    let dependencies = prefix
                        .existential_dependencies(variable)
                        .map_err(QbfStrategyCountError::InvalidStrategyScope)?;
                    let table_entries = checked_power_of_two(dependencies.len())?;
                    let table_bits = usize::try_from(table_entries)
                        .map_err(|_| QbfStrategyCountError::ResourceAccountingOverflow)?;
                    let bit_offset = table_bit_offset;
                    table_bit_offset = table_bit_offset
                        .checked_add(table_bits)
                        .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)?;
                    functions.push(StrategyFunctionLayout {
                        variable,
                        dependencies,
                        table_entries,
                        bit_offset,
                    });
                }
            }
        }
    }

    Ok((functions, universals, table_bit_offset))
}

fn matrix_satisfied(problem: &CnfProblem, assignment: &[bool]) -> bool {
    problem.clauses.iter().all(|clause| {
        clause
            .iter()
            .any(|literal| assignment[literal.variable] == literal.positive)
    })
}

fn dependency_row_index(
    dependencies: &[usize],
    assignment: &[bool],
) -> Result<usize, QbfStrategyCountError> {
    let mut key = 0usize;
    for (bit, &dependency) in dependencies.iter().enumerate() {
        if assignment[dependency] {
            let shift = u32::try_from(bit)
                .map_err(|_| QbfStrategyCountError::ResourceAccountingOverflow)?;
            key |= 1usize
                .checked_shl(shift)
                .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)?;
        }
    }
    Ok(key)
}

fn function_value_for_profile(
    profile_mask: u128,
    layout: &StrategyFunctionLayout,
    assignment: &[bool],
) -> Result<bool, QbfStrategyCountError> {
    let row = dependency_row_index(&layout.dependencies, assignment)?;
    let bit_index = layout
        .bit_offset
        .checked_add(row)
        .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)?;
    let shift =
        u32::try_from(bit_index).map_err(|_| QbfStrategyCountError::ResourceAccountingOverflow)?;
    Ok(((profile_mask >> shift) & 1) != 0)
}

/// Count all distinct winning Skolem strategy profiles by bounded exhaustive
/// enumeration.
///
/// Every existential function table is total over precisely the preceding
/// universal variables. A profile is counted once iff it satisfies the
/// original CNF matrix for every assignment to all universal variables.
pub fn count_qbf_winning_strategy_profiles(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: QbfStrategyCountBudget,
) -> Result<QbfStrategyCountReport, QbfStrategyCountError> {
    validate_source(problem, prefix, budget)?;
    let (functions, universal_variables, strategy_table_bits) = strategy_layout(prefix)?;

    if strategy_table_bits > budget.max_total_strategy_table_bits {
        return Err(QbfStrategyCountError::StrategyTableBitBudgetExceeded {
            required: strategy_table_bits,
            limit: budget.max_total_strategy_table_bits,
        });
    }

    let candidate_profiles = checked_power_of_two(strategy_table_bits)?;
    if candidate_profiles > budget.max_candidate_profiles {
        return Err(QbfStrategyCountError::CandidateProfileBudgetExceeded {
            required: candidate_profiles,
            limit: budget.max_candidate_profiles,
        });
    }

    let universal_assignments = checked_power_of_two(universal_variables.len()).map_err(|_| {
        QbfStrategyCountError::UniversalAssignmentSpaceOverflow {
            variable_count: universal_variables.len(),
        }
    })?;
    if universal_assignments > budget.max_universal_assignments {
        return Err(QbfStrategyCountError::UniversalAssignmentBudgetExceeded {
            required: universal_assignments,
            limit: budget.max_universal_assignments,
        });
    }

    let work_bound = candidate_profiles
        .checked_mul(universal_assignments)
        .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)?;
    if work_bound > budget.max_total_matrix_evaluations {
        return Err(QbfStrategyCountError::MatrixEvaluationBudgetExceeded {
            required: work_bound,
            limit: budget.max_total_matrix_evaluations,
        });
    }

    let mut winning_strategy_profiles = 0u128;
    let mut candidate_profiles_checked = 0u128;
    let mut profiles_rejected_early = 0u128;
    let mut universal_assignments_checked = 0u128;
    let mut matrix_evaluations = 0u128;
    let mut assignment = vec![false; problem.variables.len()];

    for profile_mask in 0..candidate_profiles {
        candidate_profiles_checked = candidate_profiles_checked
            .checked_add(1)
            .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)?;
        let mut profile_wins = true;

        for universal_mask in 0..universal_assignments {
            for (bit, &variable) in universal_variables.iter().enumerate() {
                let shift = u32::try_from(bit)
                    .map_err(|_| QbfStrategyCountError::ResourceAccountingOverflow)?;
                assignment[variable] = ((universal_mask >> shift) & 1) != 0;
            }

            for function in &functions {
                assignment[function.variable] =
                    function_value_for_profile(profile_mask, function, &assignment)?;
            }

            universal_assignments_checked = universal_assignments_checked
                .checked_add(1)
                .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)?;
            matrix_evaluations = matrix_evaluations
                .checked_add(1)
                .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)?;

            if !matrix_satisfied(problem, &assignment) {
                profile_wins = false;
                profiles_rejected_early = profiles_rejected_early
                    .checked_add(1)
                    .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)?;
                break;
            }
        }

        if profile_wins {
            winning_strategy_profiles = winning_strategy_profiles
                .checked_add(1)
                .ok_or(QbfStrategyCountError::ResourceAccountingOverflow)?;
        }
    }

    Ok(QbfStrategyCountReport {
        winning_strategy_profiles,
        candidate_strategy_profiles: candidate_profiles,
        candidate_profiles_checked,
        profiles_rejected_early,
        universal_assignments_checked,
        matrix_evaluations,
        strategy_table_bits,
        universal_variable_count: universal_variables.len(),
        existential_variable_count: functions.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::decomposition::Literal;
    use super::super::quantified_scope::QuantifierBlock;
    use super::super::quantified_strategy::{
        QuantifiedEvaluationBudget, evaluate_qbf_with_strategy,
    };
    use super::*;

    fn budget() -> QbfStrategyCountBudget {
        QbfStrategyCountBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 64,
            max_literals: 256,
            max_total_strategy_table_bits: 16,
            max_candidate_profiles: 1 << 16,
            max_universal_assignments: 1 << 8,
            max_total_matrix_evaluations: 1 << 20,
        }
    }

    fn block(quantifier: Quantifier, variables: &[usize]) -> QuantifierBlock {
        QuantifierBlock {
            quantifier,
            variables: variables.to_vec(),
        }
    }

    fn prefix(variables: usize, blocks: Vec<QuantifierBlock>) -> ValidatedQuantifierPrefix {
        ValidatedQuantifierPrefix::new(variables, blocks, Default::default()).unwrap()
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

    fn independent_profile_count(problem: &CnfProblem, prefix: &ValidatedQuantifierPrefix) -> u128 {
        let mut layouts = Vec::new();
        let mut universal_variables = Vec::new();
        for block in prefix.blocks() {
            for &variable in &block.variables {
                match block.quantifier {
                    Quantifier::ForAll => universal_variables.push(variable),
                    Quantifier::Exists => {
                        layouts.push((variable, prefix.existential_dependencies(variable).unwrap()))
                    }
                }
            }
        }

        fn check_profile(
            problem: &CnfProblem,
            layouts: &[(usize, Vec<usize>)],
            tables: &[Vec<bool>],
            universal_variables: &[usize],
        ) -> bool {
            let universal_rows = 1usize << universal_variables.len();
            for universal_mask in 0..universal_rows {
                let mut assignment = vec![false; problem.variables.len()];
                for (bit, &variable) in universal_variables.iter().enumerate() {
                    assignment[variable] = ((universal_mask >> bit) & 1) != 0;
                }
                for (function_index, (variable, dependencies)) in layouts.iter().enumerate() {
                    let mut row_index = 0usize;
                    for (bit, &dependency) in dependencies.iter().enumerate() {
                        if assignment[dependency] {
                            row_index |= 1usize << bit;
                        }
                    }
                    assignment[*variable] = tables[function_index][row_index];
                }
                if !problem.clauses.iter().all(|clause| {
                    clause
                        .iter()
                        .any(|literal| assignment[literal.variable] == literal.positive)
                }) {
                    return false;
                }
            }
            true
        }

        fn enumerate(
            function_index: usize,
            problem: &CnfProblem,
            layouts: &[(usize, Vec<usize>)],
            universal_variables: &[usize],
            tables: &mut Vec<Vec<bool>>,
        ) -> u128 {
            if function_index == layouts.len() {
                return u128::from(check_profile(problem, layouts, tables, universal_variables));
            }
            let entries = 1usize << layouts[function_index].1.len();
            let mut winning = 0u128;
            for function_mask in 0..(1usize << entries) {
                let table: Vec<bool> = (0..entries)
                    .map(|row| ((function_mask >> row) & 1) != 0)
                    .collect();
                tables.push(table);
                winning += enumerate(
                    function_index + 1,
                    problem,
                    layouts,
                    universal_variables,
                    tables,
                );
                tables.pop();
            }
            winning
        }

        enumerate(0, problem, &layouts, &universal_variables, &mut Vec::new())
    }

    fn assert_count(problem: &CnfProblem, prefix: &ValidatedQuantifierPrefix, expected: u128) {
        let report = count_qbf_winning_strategy_profiles(problem, prefix, budget()).unwrap();
        assert_eq!(report.winning_strategy_profiles, expected);
        let decision =
            evaluate_qbf_with_strategy(problem, prefix, QuantifiedEvaluationBudget::default())
                .unwrap();
        assert_eq!(report.winning_strategy_profiles > 0, decision.value);
        assert_eq!(
            report.candidate_profiles_checked,
            report.candidate_strategy_profiles
        );
    }

    #[test]
    fn forall_exists_equality_has_one_winning_skolem_function() {
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        assert_count(&equality_problem(), &prefix, 1);
    }

    #[test]
    fn forall_exists_or_has_two_winning_skolem_functions() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![vec![Literal::positive(0), Literal::positive(1)]],
        };
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        assert_count(&problem, &prefix, 2);
    }

    #[test]
    fn exists_forall_equality_has_no_winning_constant_strategy() {
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[0]),
            ],
        );
        assert_count(&equality_problem(), &prefix, 0);
    }

    #[test]
    fn exists_forall_or_has_one_winning_constant_strategy() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![vec![Literal::positive(0), Literal::positive(1)]],
        };
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::Exists, &[0]),
                block(Quantifier::ForAll, &[1]),
            ],
        );
        assert_count(&problem, &prefix, 1);
    }

    #[test]
    fn exists_forall_and_has_no_winning_constant_strategy() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![vec![Literal::positive(0)], vec![Literal::positive(1)]],
        };
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::Exists, &[0]),
                block(Quantifier::ForAll, &[1]),
            ],
        );
        assert_count(&problem, &prefix, 0);
    }

    #[test]
    fn two_existential_equality_has_two_constant_strategy_profiles() {
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::Exists, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        assert_count(&equality_problem(), &prefix, 2);
    }

    #[test]
    fn universal_only_true_formula_has_one_empty_strategy() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0), Literal::negative(0)]],
        };
        let prefix = prefix(1, vec![block(Quantifier::ForAll, &[0])]);
        assert_count(&problem, &prefix, 1);
    }

    #[test]
    fn universal_only_false_formula_has_zero_empty_strategies() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };
        let prefix = prefix(1, vec![block(Quantifier::ForAll, &[0])]);
        assert_count(&problem, &prefix, 0);
    }

    #[test]
    fn no_universals_count_matches_number_of_satisfying_assignments() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![vec![Literal::positive(0), Literal::positive(1)]],
        };
        let prefix = prefix(2, vec![block(Quantifier::Exists, &[0, 1])]);
        let report = count_qbf_winning_strategy_profiles(&problem, &prefix, budget()).unwrap();
        assert_eq!(report.winning_strategy_profiles, 3);
    }

    #[test]
    fn empty_prefix_and_empty_matrix_has_one_empty_profile() {
        let problem = CnfProblem {
            variables: vec![],
            clauses: vec![],
        };
        let prefix = prefix(0, vec![]);
        assert_count(&problem, &prefix, 1);
    }

    #[test]
    fn multivariable_universal_block_allows_joint_skolem_dependency() {
        // z = x XOR y.
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into(), "z".into()],
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
        let prefix = prefix(
            3,
            vec![
                block(Quantifier::ForAll, &[0, 1]),
                block(Quantifier::Exists, &[2]),
            ],
        );
        assert_count(&problem, &prefix, 1);
    }

    #[test]
    fn candidate_profile_budget_fails_closed() {
        let prefix = prefix(1, vec![block(Quantifier::Exists, &[0])]);
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![],
        };
        let tight = QbfStrategyCountBudget {
            max_candidate_profiles: 1,
            ..budget()
        };
        assert_eq!(
            count_qbf_winning_strategy_profiles(&problem, &prefix, tight).unwrap_err(),
            QbfStrategyCountError::CandidateProfileBudgetExceeded {
                required: 2,
                limit: 1
            }
        );
    }

    #[test]
    fn total_matrix_evaluation_budget_fails_before_counting() {
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let tight = QbfStrategyCountBudget {
            max_total_matrix_evaluations: 4,
            ..budget()
        };
        assert_eq!(
            count_qbf_winning_strategy_profiles(&equality_problem(), &prefix, tight).unwrap_err(),
            QbfStrategyCountError::MatrixEvaluationBudgetExceeded {
                required: 8,
                limit: 4
            }
        );
    }

    #[test]
    fn exhaustive_two_variable_exact_counts_match_independent_profile_enumerator() {
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
                    let prefix = prefix(2, blocks);
                    let actual =
                        count_qbf_winning_strategy_profiles(problem, &prefix, budget()).unwrap();
                    let expected = independent_profile_count(problem, &prefix);
                    assert_eq!(
                        actual.winning_strategy_profiles, expected,
                        "formula {formula_index}, order {order:?}, quantifier mask {quantifier_mask}"
                    );
                }
            }
        }
    }

    #[test]
    fn generated_three_variable_exact_counts_match_independent_profile_enumerator() {
        let mut state = 0x510e_527f_ade6_82d1u64;
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

        for formula_index in 0..32 {
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
                    let prefix = prefix(3, blocks);
                    let actual =
                        count_qbf_winning_strategy_profiles(&problem, &prefix, budget()).unwrap();
                    let expected = independent_profile_count(&problem, &prefix);
                    assert_eq!(
                        actual.winning_strategy_profiles, expected,
                        "formula {formula_index}, order {order:?}, quantifier mask {quantifier_mask}"
                    );
                    assert_eq!(
                        actual.candidate_profiles_checked,
                        actual.candidate_strategy_profiles
                    );
                }
            }
        }
    }

    #[test]
    fn exhaustive_two_variable_profiles_are_positive_iff_qbf_is_true() {
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
                    let prefix = prefix(2, blocks);
                    let report =
                        count_qbf_winning_strategy_profiles(problem, &prefix, budget()).unwrap();
                    let decision = evaluate_qbf_with_strategy(
                        problem,
                        &prefix,
                        QuantifiedEvaluationBudget::default(),
                    )
                    .unwrap();
                    assert_eq!(
                        report.winning_strategy_profiles > 0,
                        decision.value,
                        "formula {formula_index}, order {order:?}, quantifier mask {quantifier_mask}"
                    );
                    assert_eq!(
                        report.candidate_profiles_checked,
                        report.candidate_strategy_profiles
                    );
                }
            }
        }
    }
}
