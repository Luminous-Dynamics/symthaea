// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded MaxQBF optimization over the residual-CNF proof DAG.
//!
//! Given a FALSE prenex QBF with source clauses C_i, MaxQBF asks for a
//! maximum-cardinality subset of clauses whose quantified conjunction is
//! TRUE. This module applies the blocking-variable reduction and computes the
//! minimum blocker cost bottom-up over the existing verified residual-CNF
//! game DAG. An independent direct minimax/subset oracle checks feasibility
//! and optimality on small instances.
//!
//! The output is a residual-CNF game-DAG certificate plus finite optimality
//! checks. It is not a formal cert-TQD implementation or a scalable MaxQBF
//! solver.

use super::decomposition::{CnfProblem, Literal};
use super::quantified_game_dag::{
    QbfGameDagBudget, QbfGameDagError, QbfGameDagNodeKind, QbfGameDagVerificationReport,
    QbfGameProofDag, build_qbf_game_proof_dag, verify_qbf_game_proof_dag,
};
use super::quantified_scope::{
    Quantifier, QuantifierBlock, QuantifierPrefixBudget, QuantifierScopeError,
    ValidatedQuantifierPrefix,
};
use std::collections::BTreeSet;

/// Deterministic bounds for the DAG optimizer and independent finite oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxQbfBudget {
    pub max_source_clauses: usize,
    pub max_blocking_variables: usize,
    pub max_dp_node_visits: u128,
    pub max_direct_subset_masks: u128,
    pub max_direct_game_node_visits: u128,
    pub max_direct_matrix_evaluations: u128,
    pub game_dag: QbfGameDagBudget,
}

impl Default for MaxQbfBudget {
    fn default() -> Self {
        Self {
            max_source_clauses: 16,
            max_blocking_variables: 16,
            max_dp_node_visits: 1 << 20,
            max_direct_subset_masks: 1 << 16,
            max_direct_game_node_visits: 1 << 26,
            max_direct_matrix_evaluations: 1 << 24,
            game_dag: QbfGameDagBudget::default(),
        }
    }
}

/// The selected source-clause subset and its minimum blocker objective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxQbfOptimizationReport {
    /// Minimum Hamming weight of the blocking assignment.
    pub minimum_blocking_weight: usize,
    /// Maximum number of source clauses retained while preserving QBF truth.
    pub maximum_retained_clauses: usize,
    /// One bit per source clause; true means that clause is blocked/dropped.
    pub blocker_assignment: Vec<bool>,
    /// Retained source clause indices in ascending source order.
    pub retained_clause_indices: Vec<usize>,
    /// Number of post-order DAG nodes evaluated by dynamic programming.
    pub dp_node_visits: u128,
}

/// Source-bound MaxQBF evidence. Producer counters are deliberately omitted:
/// the verifier derives its own DAG and direct-oracle measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxQbfEvidence {
    pub source_game_dag: QbfGameProofDag,
    pub transformed_game_dag: QbfGameProofDag,
    pub optimization: MaxQbfOptimizationReport,
}

/// Measurements independently derived by the MaxQBF verifier.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MaxQbfVerificationReport {
    pub source_dag: QbfGameDagVerificationReport,
    pub transformed_dag: QbfGameDagVerificationReport,
    pub dp_node_visits: u128,
    pub direct_subset_masks_checked: u128,
    pub better_subsets_rejected: u128,
    pub direct_game_node_visits: u128,
    pub direct_matrix_evaluations: u128,
}

/// Result of MaxQBF optimization and its independent verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxQbfResult {
    pub evidence: MaxQbfEvidence,
    pub verification: MaxQbfVerificationReport,
}

/// Fail-closed optimization, proof-binding, and direct-oracle errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaxQbfError {
    SourceClauseBudgetExceeded {
        required: usize,
        limit: usize,
    },
    BlockingVariableBudgetExceeded {
        required: usize,
        limit: usize,
    },
    SourceQbfMustBeFalse {
        actual: bool,
    },
    TransformedQbfMustBeTrue {
        actual: bool,
    },
    MatrixVariableCountOverflow,
    PrefixVariableCountOverflow,
    PrefixConstruction(QuantifierScopeError),
    GameDag(QbfGameDagError),
    DynamicProgrammingBudgetExceeded {
        required: u128,
        limit: u128,
    },
    DynamicProgrammingInfeasible,
    DynamicProgrammingCostOverflow,
    DirectSubsetMaskSpaceOverflow {
        clause_count: usize,
    },
    DirectSubsetOracleBudgetExceeded {
        required: u128,
        limit: u128,
    },
    DirectGameNodeBudgetExceeded {
        required: u128,
        limit: u128,
    },
    DirectMatrixEvaluationBudgetExceeded {
        required: u128,
        limit: u128,
    },
    DirectOracleAccountingOverflow,
    BetterClauseSubsetExists {
        claimed_maximum: usize,
        counterexample_clause_indices: Vec<usize>,
    },
    SelectedClauseSubsetIsFalse,
    SourceClauseSubsetMismatch,
    BlockerVectorLengthMismatch {
        expected: usize,
        actual: usize,
    },
    BlockerWeightMismatch {
        expected: usize,
        actual: usize,
    },
    RetainedClauseIndexOutOfRange {
        index: usize,
        source_clause_count: usize,
    },
    RetainedClauseIndicesNotCanonical,
    BlockerAssignmentMismatch,
    ObjectiveMismatch {
        expected: usize,
        actual: usize,
    },
    ResourceAccountingOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlockingEncoding {
    problem: CnfProblem,
    prefix: ValidatedQuantifierPrefix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct DirectOracleStats {
    game_node_visits: u128,
    matrix_evaluations: u128,
}

fn checked_power_of_two(exponent: usize) -> Result<u128, MaxQbfError> {
    let shift =
        u32::try_from(exponent).map_err(|_| MaxQbfError::DirectSubsetMaskSpaceOverflow {
            clause_count: exponent,
        })?;
    1u128
        .checked_shl(shift)
        .ok_or(MaxQbfError::DirectSubsetMaskSpaceOverflow {
            clause_count: exponent,
        })
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

fn fresh_blocker_name(index: usize, names: &mut BTreeSet<String>) -> String {
    let base = format!("__symthaea_maxqbf_blocker_{index}");
    let mut candidate = base.clone();
    let mut suffix = 0usize;
    while names.contains(&candidate) {
        suffix = suffix.saturating_add(1);
        candidate = format!("{base}_{suffix}");
    }
    names.insert(candidate.clone());
    candidate
}

fn build_blocking_encoding(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: MaxQbfBudget,
) -> Result<BlockingEncoding, MaxQbfError> {
    let clause_count = problem.clauses.len();
    if clause_count > budget.max_source_clauses {
        return Err(MaxQbfError::SourceClauseBudgetExceeded {
            required: clause_count,
            limit: budget.max_source_clauses,
        });
    }
    if clause_count > budget.max_blocking_variables {
        return Err(MaxQbfError::BlockingVariableBudgetExceeded {
            required: clause_count,
            limit: budget.max_blocking_variables,
        });
    }
    if problem.variables.len() != prefix.variable_count() {
        return Err(MaxQbfError::PrefixConstruction(
            QuantifierScopeError::VariableOccurrenceCountMismatch {
                expected: problem.variables.len(),
                actual: prefix.variable_count(),
            },
        ));
    }

    let augmented_variable_count = problem
        .variables
        .len()
        .checked_add(clause_count)
        .ok_or(MaxQbfError::MatrixVariableCountOverflow)?;
    if augmented_variable_count > budget.game_dag.max_variables {
        return Err(MaxQbfError::GameDag(
            QbfGameDagError::VariableBudgetExceeded {
                required: augmented_variable_count,
                limit: budget.game_dag.max_variables,
            },
        ));
    }

    let mut used_names: BTreeSet<String> = problem.variables.iter().cloned().collect();
    let blocker_names: Vec<String> = (0..clause_count)
        .map(|index| fresh_blocker_name(index, &mut used_names))
        .collect();

    let mut variables = blocker_names.clone();
    variables.extend(problem.variables.iter().cloned());

    let mut clauses = Vec::with_capacity(clause_count);
    for (clause_index, source_clause) in problem.clauses.iter().enumerate() {
        let mut transformed_clause = Vec::with_capacity(source_clause.len().saturating_add(1));
        for literal in source_clause {
            let shifted_variable = clause_count
                .checked_add(literal.variable)
                .ok_or(MaxQbfError::MatrixVariableCountOverflow)?;
            transformed_clause.push(Literal {
                variable: shifted_variable,
                positive: literal.positive,
            });
        }
        transformed_clause.push(Literal::positive(clause_index));
        clauses.push(transformed_clause);
    }

    let mut blocks = Vec::with_capacity(prefix.blocks().len().saturating_add(1));
    if clause_count > 0 {
        blocks.push(QuantifierBlock {
            quantifier: Quantifier::Exists,
            variables: (0..clause_count).collect(),
        });
    }

    for (block_index, source_block) in prefix.blocks().iter().enumerate() {
        let shifted_variables: Vec<usize> = source_block
            .variables
            .iter()
            .map(|variable| {
                clause_count
                    .checked_add(*variable)
                    .ok_or(MaxQbfError::PrefixVariableCountOverflow)
            })
            .collect::<Result<_, _>>()?;

        if block_index == 0 && source_block.quantifier == Quantifier::Exists && clause_count > 0 {
            blocks[0].variables.extend(shifted_variables);
        } else {
            blocks.push(QuantifierBlock {
                quantifier: source_block.quantifier,
                variables: shifted_variables,
            });
        }
    }

    let prefix_budget = QuantifierPrefixBudget {
        max_variables: budget.game_dag.max_variables,
        max_blocks: budget.game_dag.max_variables.saturating_add(1),
        max_variable_occurrences: augmented_variable_count,
    };
    let augmented_prefix =
        ValidatedQuantifierPrefix::new(augmented_variable_count, blocks, prefix_budget)
            .map_err(MaxQbfError::PrefixConstruction)?;

    Ok(BlockingEncoding {
        problem: CnfProblem { variables, clauses },
        prefix: augmented_prefix,
    })
}

fn min_finite(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn add_blocker_cost(value: Option<usize>) -> Result<Option<usize>, MaxQbfError> {
    value
        .map(|cost| {
            cost.checked_add(1)
                .ok_or(MaxQbfError::DynamicProgrammingCostOverflow)
        })
        .transpose()
}

fn dynamic_program_blocker_costs(
    dag: &QbfGameProofDag,
    blocker_count: usize,
    budget: MaxQbfBudget,
) -> Result<(Vec<Option<usize>>, u128), MaxQbfError> {
    let visits =
        u128::try_from(dag.nodes.len()).map_err(|_| MaxQbfError::ResourceAccountingOverflow)?;
    if visits > budget.max_dp_node_visits {
        return Err(MaxQbfError::DynamicProgrammingBudgetExceeded {
            required: visits,
            limit: budget.max_dp_node_visits,
        });
    }

    let mut costs = Vec::with_capacity(dag.nodes.len());
    for node in &dag.nodes {
        let cost = match &node.kind {
            QbfGameDagNodeKind::TrueTerminal => Some(0),
            QbfGameDagNodeKind::FalseTerminal => None,
            QbfGameDagNodeKind::ExistsDecision {
                variable,
                low_child,
                high_child,
            } => {
                let low = costs
                    .get(*low_child)
                    .copied()
                    .ok_or(MaxQbfError::DynamicProgrammingInfeasible)?;
                let high = costs
                    .get(*high_child)
                    .copied()
                    .ok_or(MaxQbfError::DynamicProgrammingInfeasible)?;
                if *variable < blocker_count {
                    min_finite(low, add_blocker_cost(high)?)
                } else {
                    min_finite(low, high)
                }
            }
            QbfGameDagNodeKind::ForAllDecision {
                low_child,
                high_child,
                ..
            } => {
                let low = costs
                    .get(*low_child)
                    .copied()
                    .ok_or(MaxQbfError::DynamicProgrammingInfeasible)?;
                let high = costs
                    .get(*high_child)
                    .copied()
                    .ok_or(MaxQbfError::DynamicProgrammingInfeasible)?;
                match (low, high) {
                    (Some(low), Some(high)) => Some(low.max(high)),
                    _ => None,
                }
            }
        };
        costs.push(cost);
    }
    Ok((costs, visits))
}

fn recover_blocker_assignment(
    dag: &QbfGameProofDag,
    costs: &[Option<usize>],
    blocker_count: usize,
) -> Result<Vec<bool>, MaxQbfError> {
    let mut blocker_assignment = vec![false; blocker_count];
    let mut current = dag.root;

    for expected_variable in 0..blocker_count {
        let node = dag
            .nodes
            .get(current)
            .ok_or(MaxQbfError::DynamicProgrammingInfeasible)?;
        match &node.kind {
            QbfGameDagNodeKind::TrueTerminal => break,
            QbfGameDagNodeKind::FalseTerminal => {
                return Err(MaxQbfError::DynamicProgrammingInfeasible);
            }
            QbfGameDagNodeKind::ExistsDecision {
                variable,
                low_child,
                high_child,
            } if *variable == expected_variable => {
                let low_cost = costs.get(*low_child).copied().flatten();
                let high_cost = add_blocker_cost(costs.get(*high_child).copied().flatten())?;
                let choose_high = match (low_cost, high_cost) {
                    (Some(low), Some(high)) => high < low,
                    (Some(_), None) => false,
                    (None, Some(_)) => true,
                    (None, None) => return Err(MaxQbfError::DynamicProgrammingInfeasible),
                };
                blocker_assignment[expected_variable] = choose_high;
                current = if choose_high {
                    *high_child
                } else {
                    *low_child
                };
            }
            _ => return Err(MaxQbfError::DynamicProgrammingInfeasible),
        }
    }
    Ok(blocker_assignment)
}

fn retained_clause_indices(blockers: &[bool]) -> Vec<usize> {
    blockers
        .iter()
        .enumerate()
        .filter_map(|(index, blocked)| (!blocked).then_some(index))
        .collect()
}

fn clause_subset_mask(indices: &[usize]) -> Result<u128, MaxQbfError> {
    let mut mask = 0u128;
    for &index in indices {
        let shift =
            u32::try_from(index).map_err(|_| MaxQbfError::DirectSubsetMaskSpaceOverflow {
                clause_count: index.saturating_add(1),
            })?;
        let bit = 1u128
            .checked_shl(shift)
            .ok_or(MaxQbfError::DirectSubsetMaskSpaceOverflow {
                clause_count: index.saturating_add(1),
            })?;
        mask |= bit;
    }
    Ok(mask)
}

fn direct_matrix_satisfied(problem: &CnfProblem, subset_mask: u128, assignment: &[bool]) -> bool {
    problem.clauses.iter().enumerate().all(|(index, clause)| {
        let selected = ((subset_mask >> index) & 1) != 0;
        !selected
            || clause
                .iter()
                .any(|literal| assignment[literal.variable] == literal.positive)
    })
}

fn direct_qbf_visit(
    position: usize,
    order: &[(Quantifier, usize)],
    problem: &CnfProblem,
    subset_mask: u128,
    assignment: &mut [bool],
    stats: &mut DirectOracleStats,
    budget: MaxQbfBudget,
) -> Result<bool, MaxQbfError> {
    stats.game_node_visits = stats
        .game_node_visits
        .checked_add(1)
        .ok_or(MaxQbfError::DirectOracleAccountingOverflow)?;
    if stats.game_node_visits > budget.max_direct_game_node_visits {
        return Err(MaxQbfError::DirectGameNodeBudgetExceeded {
            required: stats.game_node_visits,
            limit: budget.max_direct_game_node_visits,
        });
    }

    if position == order.len() {
        stats.matrix_evaluations = stats
            .matrix_evaluations
            .checked_add(1)
            .ok_or(MaxQbfError::DirectOracleAccountingOverflow)?;
        if stats.matrix_evaluations > budget.max_direct_matrix_evaluations {
            return Err(MaxQbfError::DirectMatrixEvaluationBudgetExceeded {
                required: stats.matrix_evaluations,
                limit: budget.max_direct_matrix_evaluations,
            });
        }
        return Ok(direct_matrix_satisfied(problem, subset_mask, assignment));
    }

    let (quantifier, variable) = order[position];
    assignment[variable] = false;
    let low = direct_qbf_visit(
        position + 1,
        order,
        problem,
        subset_mask,
        assignment,
        stats,
        budget,
    )?;

    match quantifier {
        Quantifier::Exists if low => Ok(true),
        Quantifier::ForAll if !low => Ok(false),
        Quantifier::Exists | Quantifier::ForAll => {
            assignment[variable] = true;
            direct_qbf_visit(
                position + 1,
                order,
                problem,
                subset_mask,
                assignment,
                stats,
                budget,
            )
        }
    }
}

fn direct_qbf_value(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    subset_mask: u128,
    stats: &mut DirectOracleStats,
    budget: MaxQbfBudget,
) -> Result<bool, MaxQbfError> {
    let order = ordered_prefix(prefix);
    let mut assignment = vec![false; problem.variables.len()];
    direct_qbf_visit(
        0,
        &order,
        problem,
        subset_mask,
        &mut assignment,
        stats,
        budget,
    )
}

fn count_better_subset_masks(
    clause_count: usize,
    claimed_maximum: usize,
) -> Result<u128, MaxQbfError> {
    let total = checked_power_of_two(clause_count)?;
    let mut count = 0u128;
    for mask in 0..total {
        if mask.count_ones() as usize > claimed_maximum {
            count = count
                .checked_add(1)
                .ok_or(MaxQbfError::ResourceAccountingOverflow)?;
        }
    }
    Ok(count)
}

fn preflight_direct_oracle(
    problem: &CnfProblem,
    claimed_maximum: usize,
    budget: MaxQbfBudget,
) -> Result<u128, MaxQbfError> {
    let subset_masks = count_better_subset_masks(problem.clauses.len(), claimed_maximum)?;
    if subset_masks > budget.max_direct_subset_masks {
        return Err(MaxQbfError::DirectSubsetOracleBudgetExceeded {
            required: subset_masks,
            limit: budget.max_direct_subset_masks,
        });
    }

    let variables = problem.variables.len();
    let matrix_rows = checked_power_of_two(variables)?;
    let full_tree_nodes = matrix_rows
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(MaxQbfError::DirectOracleAccountingOverflow)?;

    let direct_calls = subset_masks
        .checked_add(1)
        .ok_or(MaxQbfError::DirectOracleAccountingOverflow)?;
    let worst_nodes = direct_calls
        .checked_mul(full_tree_nodes)
        .ok_or(MaxQbfError::DirectOracleAccountingOverflow)?;
    if worst_nodes > budget.max_direct_game_node_visits {
        return Err(MaxQbfError::DirectGameNodeBudgetExceeded {
            required: worst_nodes,
            limit: budget.max_direct_game_node_visits,
        });
    }

    let worst_matrix = direct_calls
        .checked_mul(matrix_rows)
        .ok_or(MaxQbfError::DirectOracleAccountingOverflow)?;
    if worst_matrix > budget.max_direct_matrix_evaluations {
        return Err(MaxQbfError::DirectMatrixEvaluationBudgetExceeded {
            required: worst_matrix,
            limit: budget.max_direct_matrix_evaluations,
        });
    }
    Ok(subset_masks)
}

/// Construct the blocking-variable encoding, optimize the blocker weight over
/// the verified residual-CNF DAG, and independently qualify the result.
pub fn solve_maxqbf(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: MaxQbfBudget,
) -> Result<MaxQbfResult, MaxQbfError> {
    if problem.clauses.len() > budget.max_source_clauses {
        return Err(MaxQbfError::SourceClauseBudgetExceeded {
            required: problem.clauses.len(),
            limit: budget.max_source_clauses,
        });
    }

    let (source_game_dag, _) =
        build_qbf_game_proof_dag(problem, prefix, budget.game_dag).map_err(MaxQbfError::GameDag)?;
    let source_verification =
        verify_qbf_game_proof_dag(problem, prefix, &source_game_dag, budget.game_dag)
            .map_err(MaxQbfError::GameDag)?;
    if source_verification.root_value {
        return Err(MaxQbfError::SourceQbfMustBeFalse {
            actual: source_verification.root_value,
        });
    }

    let encoding = build_blocking_encoding(problem, prefix, budget)?;
    let (transformed_game_dag, _) =
        build_qbf_game_proof_dag(&encoding.problem, &encoding.prefix, budget.game_dag)
            .map_err(MaxQbfError::GameDag)?;
    let transformed_verification = verify_qbf_game_proof_dag(
        &encoding.problem,
        &encoding.prefix,
        &transformed_game_dag,
        budget.game_dag,
    )
    .map_err(MaxQbfError::GameDag)?;
    if !transformed_verification.root_value {
        return Err(MaxQbfError::TransformedQbfMustBeTrue {
            actual: transformed_verification.root_value,
        });
    }

    let (costs, dp_node_visits) =
        dynamic_program_blocker_costs(&transformed_game_dag, problem.clauses.len(), budget)?;
    let minimum_blocking_weight = costs
        .get(transformed_game_dag.root)
        .copied()
        .flatten()
        .ok_or(MaxQbfError::DynamicProgrammingInfeasible)?;
    let maximum_retained_clauses = problem
        .clauses
        .len()
        .checked_sub(minimum_blocking_weight)
        .ok_or(MaxQbfError::DynamicProgrammingCostOverflow)?;
    let blocker_assignment =
        recover_blocker_assignment(&transformed_game_dag, &costs, problem.clauses.len())?;
    let selected_indices = retained_clause_indices(&blocker_assignment);
    let optimization = MaxQbfOptimizationReport {
        minimum_blocking_weight,
        maximum_retained_clauses,
        blocker_assignment,
        retained_clause_indices: selected_indices,
        dp_node_visits,
    };

    let evidence = MaxQbfEvidence {
        source_game_dag,
        transformed_game_dag,
        optimization,
    };
    let verification = verify_maxqbf_evidence(problem, prefix, &evidence, budget)?;
    Ok(MaxQbfResult {
        evidence,
        verification,
    })
}

/// Independently verify both proof DAGs, reconstruct the blocking encoding,
/// recompute min/max costs, and reject every source-clause subset that is
/// larger than the claimed optimum using direct quantifier-game evaluation.
///
/// This verifier does not call the DAG builder or its dynamic-programming
/// implementation.
pub fn verify_maxqbf_evidence(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    evidence: &MaxQbfEvidence,
    budget: MaxQbfBudget,
) -> Result<MaxQbfVerificationReport, MaxQbfError> {
    let clause_count = problem.clauses.len();
    if clause_count > budget.max_source_clauses {
        return Err(MaxQbfError::SourceClauseBudgetExceeded {
            required: clause_count,
            limit: budget.max_source_clauses,
        });
    }
    if evidence.optimization.blocker_assignment.len() != clause_count {
        return Err(MaxQbfError::BlockerVectorLengthMismatch {
            expected: clause_count,
            actual: evidence.optimization.blocker_assignment.len(),
        });
    }
    if evidence
        .optimization
        .retained_clause_indices
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(MaxQbfError::RetainedClauseIndicesNotCanonical);
    }
    if let Some(&index) = evidence
        .optimization
        .retained_clause_indices
        .iter()
        .find(|&&index| index >= clause_count)
    {
        return Err(MaxQbfError::RetainedClauseIndexOutOfRange {
            index,
            source_clause_count: clause_count,
        });
    }

    let source_dag_report =
        verify_qbf_game_proof_dag(problem, prefix, &evidence.source_game_dag, budget.game_dag)
            .map_err(MaxQbfError::GameDag)?;
    if source_dag_report.root_value {
        return Err(MaxQbfError::SourceQbfMustBeFalse {
            actual: source_dag_report.root_value,
        });
    }

    let encoding = build_blocking_encoding(problem, prefix, budget)?;
    let transformed_dag_report = verify_qbf_game_proof_dag(
        &encoding.problem,
        &encoding.prefix,
        &evidence.transformed_game_dag,
        budget.game_dag,
    )
    .map_err(MaxQbfError::GameDag)?;
    if !transformed_dag_report.root_value {
        return Err(MaxQbfError::TransformedQbfMustBeTrue {
            actual: transformed_dag_report.root_value,
        });
    }

    let (costs, dp_node_visits) =
        dynamic_program_blocker_costs(&evidence.transformed_game_dag, clause_count, budget)?;
    let recomputed_minimum = costs
        .get(evidence.transformed_game_dag.root)
        .copied()
        .flatten()
        .ok_or(MaxQbfError::DynamicProgrammingInfeasible)?;
    if evidence.optimization.minimum_blocking_weight != recomputed_minimum {
        return Err(MaxQbfError::ObjectiveMismatch {
            expected: recomputed_minimum,
            actual: evidence.optimization.minimum_blocking_weight,
        });
    }

    let recomputed_retained = clause_count
        .checked_sub(recomputed_minimum)
        .ok_or(MaxQbfError::DynamicProgrammingCostOverflow)?;
    if evidence.optimization.maximum_retained_clauses != recomputed_retained {
        return Err(MaxQbfError::ObjectiveMismatch {
            expected: recomputed_retained,
            actual: evidence.optimization.maximum_retained_clauses,
        });
    }

    let blocker_weight = evidence
        .optimization
        .blocker_assignment
        .iter()
        .filter(|blocked| **blocked)
        .count();
    if blocker_weight != recomputed_minimum {
        return Err(MaxQbfError::BlockerWeightMismatch {
            expected: recomputed_minimum,
            actual: blocker_weight,
        });
    }

    let canonical_blockers =
        recover_blocker_assignment(&evidence.transformed_game_dag, &costs, clause_count)?;
    if canonical_blockers != evidence.optimization.blocker_assignment {
        return Err(MaxQbfError::BlockerAssignmentMismatch);
    }
    let canonical_retained = retained_clause_indices(&canonical_blockers);
    if canonical_retained != evidence.optimization.retained_clause_indices {
        return Err(MaxQbfError::SourceClauseSubsetMismatch);
    }

    let selected_mask = clause_subset_mask(&canonical_retained)?;
    let mut direct_stats = DirectOracleStats::default();
    if !direct_qbf_value(problem, prefix, selected_mask, &mut direct_stats, budget)? {
        return Err(MaxQbfError::SelectedClauseSubsetIsFalse);
    }

    let masks_to_check = preflight_direct_oracle(problem, recomputed_retained, budget)?;
    let mut better_subsets_rejected = 0u128;
    let total_masks = checked_power_of_two(clause_count)?;
    for mask in 0..total_masks {
        if mask.count_ones() as usize <= recomputed_retained {
            continue;
        }
        let value = direct_qbf_value(problem, prefix, mask, &mut direct_stats, budget)?;
        if value {
            if mask == checked_power_of_two(clause_count)?.saturating_sub(1) {
                return Err(MaxQbfError::SourceQbfMustBeFalse { actual: true });
            }
            let counterexample_clause_indices = (0..clause_count)
                .filter(|index| ((mask >> index) & 1) != 0)
                .collect();
            return Err(MaxQbfError::BetterClauseSubsetExists {
                claimed_maximum: recomputed_retained,
                counterexample_clause_indices,
            });
        }
        better_subsets_rejected = better_subsets_rejected
            .checked_add(1)
            .ok_or(MaxQbfError::ResourceAccountingOverflow)?;
    }

    Ok(MaxQbfVerificationReport {
        source_dag: source_dag_report,
        transformed_dag: transformed_dag_report,
        dp_node_visits,
        direct_subset_masks_checked: masks_to_check,
        better_subsets_rejected,
        direct_game_node_visits: direct_stats.game_node_visits,
        direct_matrix_evaluations: direct_stats.matrix_evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> MaxQbfBudget {
        MaxQbfBudget {
            max_source_clauses: 8,
            max_blocking_variables: 8,
            max_dp_node_visits: 1 << 12,
            max_direct_subset_masks: 1 << 8,
            max_direct_game_node_visits: 1 << 16,
            max_direct_matrix_evaluations: 1 << 12,
            game_dag: QbfGameDagBudget {
                max_variables: 16,
                max_total_variable_name_bytes: 4 * 1024,
                max_clauses: 64,
                max_literals: 256,
                max_proof_nodes: 1 << 12,
                max_builder_state_visits: 1 << 12,
                max_verifier_node_visits: 1 << 12,
                max_residual_literal_visits: 1 << 16,
                max_stored_residual_literals: 1 << 16,
                max_stored_residual_clauses: 1 << 14,
            },
        }
    }

    fn prefix(blocks: Vec<QuantifierBlock>, variable_count: usize) -> ValidatedQuantifierPrefix {
        ValidatedQuantifierPrefix::new(
            variable_count,
            blocks,
            QuantifierPrefixBudget {
                max_variables: 16,
                max_blocks: 16,
                max_variable_occurrences: 16,
            },
        )
        .unwrap()
    }

    fn block(quantifier: Quantifier, variables: &[usize]) -> QuantifierBlock {
        QuantifierBlock {
            quantifier,
            variables: variables.to_vec(),
        }
    }

    fn contradictory_unit_problem() -> CnfProblem {
        CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0)], vec![Literal::negative(0)]],
        }
    }

    #[test]
    fn exists_contradictory_units_retains_one_clause() {
        let problem = contradictory_unit_problem();
        let prefix = prefix(vec![block(Quantifier::Exists, &[0])], 1);
        let result = solve_maxqbf(&problem, &prefix, budget()).unwrap();
        assert_eq!(result.evidence.optimization.maximum_retained_clauses, 1);
        assert_eq!(result.evidence.optimization.minimum_blocking_weight, 1);
        assert_eq!(
            result.evidence.optimization.blocker_assignment,
            vec![false, true]
        );
        assert_eq!(
            result.evidence.optimization.retained_clause_indices,
            vec![0]
        );
        assert_eq!(result.verification.better_subsets_rejected, 1);
    }

    #[test]
    fn forall_contradictory_units_retains_no_clause() {
        let problem = contradictory_unit_problem();
        let prefix = prefix(vec![block(Quantifier::ForAll, &[0])], 1);
        let result = solve_maxqbf(&problem, &prefix, budget()).unwrap();
        assert_eq!(result.evidence.optimization.maximum_retained_clauses, 0);
        assert_eq!(result.evidence.optimization.minimum_blocking_weight, 2);
        assert_eq!(
            result.evidence.optimization.blocker_assignment,
            vec![true, true]
        );
        assert_eq!(result.evidence.optimization.retained_clause_indices, vec![]);
    }

    #[test]
    fn multi_alternation_preserves_the_largest_true_clause_subset() {
        let problem = CnfProblem {
            variables: vec!["u".into(), "e".into()],
            clauses: vec![
                vec![Literal::positive(0), Literal::positive(1)],
                vec![Literal::negative(0), Literal::negative(1)],
                vec![Literal::positive(0)],
            ],
        };
        let prefix = prefix(
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
            2,
        );
        let result = solve_maxqbf(&problem, &prefix, budget()).unwrap();
        assert_eq!(result.evidence.optimization.maximum_retained_clauses, 2);
        assert_eq!(
            result.evidence.optimization.retained_clause_indices,
            vec![0, 1]
        );
    }

    #[test]
    fn tautological_source_clause_is_retained_over_an_empty_clause() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![], vec![Literal::positive(0), Literal::negative(0)]],
        };
        let prefix = prefix(vec![block(Quantifier::ForAll, &[0])], 1);
        let result = solve_maxqbf(&problem, &prefix, budget()).unwrap();
        assert_eq!(result.evidence.optimization.maximum_retained_clauses, 1);
        assert_eq!(
            result.evidence.optimization.retained_clause_indices,
            vec![1]
        );
    }

    #[test]
    fn two_empty_source_clauses_retain_zero() {
        let problem = CnfProblem {
            variables: vec![],
            clauses: vec![vec![], vec![]],
        };
        let prefix = prefix(vec![], 0);
        let result = solve_maxqbf(&problem, &prefix, budget()).unwrap();
        assert_eq!(result.evidence.optimization.maximum_retained_clauses, 0);
        assert_eq!(result.evidence.optimization.blocker_assignment, vec![true, true]);
    }

    #[test]
    fn source_qbf_already_true_is_rejected() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![],
        };
        let prefix = prefix(vec![block(Quantifier::Exists, &[0])], 1);
        assert_eq!(
            solve_maxqbf(&problem, &prefix, budget()).unwrap_err(),
            MaxQbfError::SourceQbfMustBeFalse { actual: true }
        );
    }

    #[test]
    fn verifier_rejects_tampered_blocker_vector() {
        let problem = contradictory_unit_problem();
        let prefix = prefix(vec![block(Quantifier::Exists, &[0])], 1);
        let mut result = solve_maxqbf(&problem, &prefix, budget()).unwrap();
        result.evidence.optimization.blocker_assignment[0] = true;
        assert_eq!(
            verify_maxqbf_evidence(&problem, &prefix, &result.evidence, budget()).unwrap_err(),
            MaxQbfError::BlockerWeightMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn verifier_rejects_tampered_retained_clause_indices() {
        let problem = contradictory_unit_problem();
        let prefix = prefix(vec![block(Quantifier::Exists, &[0])], 1);
        let mut result = solve_maxqbf(&problem, &prefix, budget()).unwrap();
        result.evidence.optimization.retained_clause_indices = vec![1];
        assert_eq!(
            verify_maxqbf_evidence(&problem, &prefix, &result.evidence, budget()).unwrap_err(),
            MaxQbfError::SourceClauseSubsetMismatch
        );
    }

    #[test]
    fn verifier_finds_a_better_subset_when_claim_is_tampered() {
        let problem = contradictory_unit_problem();
        let prefix = prefix(vec![block(Quantifier::Exists, &[0])], 1);
        let mut result = solve_maxqbf(&problem, &prefix, budget()).unwrap();
        result.evidence.optimization.blocker_assignment = vec![true, true];
        result.evidence.optimization.retained_clause_indices.clear();
        result.evidence.optimization.minimum_blocking_weight = 2;
        result.evidence.optimization.maximum_retained_clauses = 0;
        assert!(matches!(
            verify_maxqbf_evidence(&problem, &prefix, &result.evidence, budget()),
            Err(MaxQbfError::BlockerAssignmentMismatch)
                | Err(MaxQbfError::ObjectiveMismatch { .. })
                | Err(MaxQbfError::BlockerWeightMismatch { .. })
                | Err(MaxQbfError::BetterClauseSubsetExists { .. })
        ));
    }

    #[test]
    fn dynamic_program_budget_exhaustion_is_fail_closed() {
        let problem = contradictory_unit_problem();
        let prefix = prefix(vec![block(Quantifier::Exists, &[0])], 1);
        let tight = MaxQbfBudget {
            max_dp_node_visits: 1,
            ..budget()
        };
        assert!(matches!(
            solve_maxqbf(&problem, &prefix, tight),
            Err(MaxQbfError::DynamicProgrammingBudgetExceeded { .. })
        ));
    }

    #[test]
    fn direct_subset_oracle_budget_exhaustion_is_fail_closed() {
        let problem = contradictory_unit_problem();
        let prefix = prefix(vec![block(Quantifier::Exists, &[0])], 1);
        let tight = MaxQbfBudget {
            max_direct_subset_masks: 0,
            ..budget()
        };
        assert!(matches!(
            solve_maxqbf(&problem, &prefix, tight),
            Err(MaxQbfError::DirectSubsetOracleBudgetExceeded { .. })
        ));
    }

    #[test]
    fn transformed_dag_tampering_is_rejected() {
        let problem = contradictory_unit_problem();
        let prefix = prefix(vec![block(Quantifier::Exists, &[0])], 1);
        let mut result = solve_maxqbf(&problem, &prefix, budget()).unwrap();
        result.evidence.transformed_game_dag.nodes[0].value =
            !result.evidence.transformed_game_dag.nodes[0].value;
        assert!(matches!(
            verify_maxqbf_evidence(&problem, &prefix, &result.evidence, budget()),
            Err(MaxQbfError::GameDag(_))
        ));
    }
}
