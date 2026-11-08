// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded prenex-QBF game proofs with independently checked intermediate values.
//!
//! This module emits a deterministic post-order proof tree. Decision nodes
//! apply the exact EXISTS/OR and FORALL/AND game recurrence. Early terminal
//! nodes carry either an all-clauses-satisfied reason or a falsified-clause
//! witness. The verifier reconstructs assignments from the original prefix
//! and validates every rule against the original CNF.
//!
//! This is a finite proof-trace reference, not a scalable QBF proof system.
//! The initial representation is a tree; it deliberately does not deduplicate
//! contextual subproofs into a DAG.

use super::decomposition::CnfProblem;
use super::quantified_scope::{Quantifier, ValidatedQuantifierPrefix};

/// Explicit resource limits for proof construction and independent checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantifiedProofBudget {
    pub max_variables: usize,
    pub max_total_variable_name_bytes: usize,
    pub max_clauses: usize,
    pub max_literals: usize,
    pub max_proof_nodes: usize,
    pub max_builder_node_visits: u128,
    pub max_verifier_node_visits: u128,
}

impl Default for QuantifiedProofBudget {
    fn default() -> Self {
        Self {
            max_variables: 20,
            max_total_variable_name_bytes: 4 * 1024,
            max_clauses: 1 << 16,
            max_literals: 1 << 20,
            max_proof_nodes: 1 << 20,
            max_builder_node_visits: 1 << 20,
            max_verifier_node_visits: 1 << 20,
        }
    }
}

/// One local derivation rule in the finite game proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QbfProofNodeKind {
    /// A complete assignment has reached the original matrix.
    MatrixLeaf,
    /// Every source clause already has an assigned satisfying literal.
    EarlyTrue,
    /// The cited source clause is fully assigned and every literal is false.
    EarlyFalseClause { clause_index: usize },
    /// Both child proofs are checked; value = low.value OR high.value.
    ExistsDecision {
        variable: usize,
        low_child: usize,
        high_child: usize,
    },
    /// Both child proofs are checked; value = low.value AND high.value.
    ForAllDecision {
        variable: usize,
        low_child: usize,
        high_child: usize,
    },
}

/// A post-order proof node. Its identifier is its index in the proof's node list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QbfProofNode {
    pub prefix_position: usize,
    pub value: bool,
    pub kind: QbfProofNodeKind,
}

/// Bounded finite proof tree for one source matrix and quantifier prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QbfGameProof {
    pub variable_count: usize,
    pub root: usize,
    pub nodes: Vec<QbfProofNode>,
}

/// Construction measurements. These counters are descriptive, not proof claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QbfProofBuildReport {
    pub proof_nodes: usize,
    pub builder_node_visits: u128,
    pub early_true_prunes: u128,
    pub early_false_prunes: u128,
    pub complete_matrix_leaves: u128,
    /// Full binary prefix expansion size before early matrix pruning.
    pub full_expansion_node_bound: u128,
    /// Full expansion node bound minus emitted proof nodes.
    pub proof_nodes_saved_vs_full_expansion: u128,
    pub root_value: bool,
}

/// Independently derived proof-verification evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QbfProofVerificationReport {
    pub proof_nodes_checked: usize,
    pub verifier_node_visits: u128,
    pub early_true_prunes_checked: u128,
    pub early_false_clause_witnesses_checked: u128,
    pub complete_matrix_leaves_checked: u128,
    pub decision_nodes_checked: u128,
    pub root_value: bool,
}

/// Fail-closed construction and verification errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantifiedProofError {
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
    BuilderVisitBudgetExceeded {
        required: u128,
        limit: u128,
    },
    ProofNodeBudgetExceeded {
        required: usize,
        limit: usize,
    },
    ProofNodeCountExceedsExpansionBound {
        proof_nodes: u128,
        full_expansion_nodes: u128,
    },
    MissingRoot,
    RootNotFinalNode {
        root: usize,
        final_node: usize,
    },
    RootOutOfRange {
        root: usize,
        node_count: usize,
    },
    ProofVariableCountMismatch {
        proof: usize,
        source: usize,
    },
    ChildOutOfRange {
        parent: usize,
        child: usize,
        node_count: usize,
    },
    ChildNotPostOrder {
        parent: usize,
        child: usize,
    },
    DuplicateChildReference {
        parent: usize,
        child: usize,
    },
    MultipleParents {
        child: usize,
        parents: usize,
    },
    UnreachableNode {
        node: usize,
    },
    VerifierVisitBudgetExceeded {
        required: u128,
        limit: u128,
    },
    PrefixPositionMismatch {
        node: usize,
        expected: usize,
        actual: usize,
    },
    DecisionVariableMismatch {
        node: usize,
        expected: usize,
        actual: usize,
    },
    QuantifierRuleMismatch {
        node: usize,
        expected: Quantifier,
    },
    IncorrectNodeValue {
        node: usize,
        expected: bool,
        actual: bool,
    },
    EarlyTrueNotEstablished {
        node: usize,
    },
    EarlyFalseClauseOutOfRange {
        node: usize,
        clause_index: usize,
        clause_count: usize,
    },
    EarlyFalseClauseNotEstablished {
        node: usize,
        clause_index: usize,
    },
    MatrixLeafBeforePrefixEnd {
        node: usize,
        position: usize,
        prefix_len: usize,
    },
    MatrixLeafValueMismatch {
        node: usize,
        expected: bool,
        actual: bool,
    },
    ResourceAccountingOverflow,
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

fn validate_source(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: QuantifiedProofBudget,
) -> Result<(), QuantifiedProofError> {
    if problem.variables.len() != prefix.variable_count() {
        return Err(QuantifiedProofError::MatrixPrefixVariableCountMismatch {
            matrix: problem.variables.len(),
            prefix: prefix.variable_count(),
        });
    }
    if problem.variables.len() > budget.max_variables {
        return Err(QuantifiedProofError::VariableBudgetExceeded {
            required: problem.variables.len(),
            limit: budget.max_variables,
        });
    }

    let name_bytes = problem.variables.iter().try_fold(0usize, |total, name| {
        total
            .checked_add(name.len())
            .ok_or(QuantifiedProofError::ResourceAccountingOverflow)
    })?;
    if name_bytes > budget.max_total_variable_name_bytes {
        return Err(QuantifiedProofError::VariableNameBudgetExceeded {
            required: name_bytes,
            limit: budget.max_total_variable_name_bytes,
        });
    }

    let mut names = std::collections::HashSet::with_capacity(problem.variables.len());
    for (index, name) in problem.variables.iter().enumerate() {
        if name.is_empty() {
            return Err(QuantifiedProofError::EmptyVariableName { index });
        }
        if !names.insert(name.as_str()) {
            return Err(QuantifiedProofError::DuplicateVariableName(name.clone()));
        }
    }

    if problem.clauses.len() > budget.max_clauses {
        return Err(QuantifiedProofError::ClauseBudgetExceeded {
            required: problem.clauses.len(),
            limit: budget.max_clauses,
        });
    }
    let literal_count = problem.clauses.iter().try_fold(0usize, |total, clause| {
        total
            .checked_add(clause.len())
            .ok_or(QuantifiedProofError::ResourceAccountingOverflow)
    })?;
    if literal_count > budget.max_literals {
        return Err(QuantifiedProofError::LiteralBudgetExceeded {
            required: literal_count,
            limit: budget.max_literals,
        });
    }
    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        for (literal_index, literal) in clause.iter().enumerate() {
            if literal.variable >= problem.variables.len() {
                return Err(QuantifiedProofError::LiteralVariableOutOfRange {
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

fn assignment_value(problem: &CnfProblem, assignment: &[Option<bool>]) -> bool {
    problem.clauses.iter().all(|clause| {
        clause.iter().any(|literal| {
            assignment[literal.variable].is_some_and(|value| value == literal.positive)
        })
    })
}

/// Determine whether the current partial assignment already fixes the CNF.
fn partial_matrix_status(
    problem: &CnfProblem,
    assignment: &[Option<bool>],
) -> Option<(bool, Option<usize>)> {
    let mut every_clause_satisfied = true;
    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        let mut clause_satisfied = false;
        let mut has_unassigned_literal = false;
        for literal in clause {
            match assignment[literal.variable] {
                Some(value) if value == literal.positive => {
                    clause_satisfied = true;
                    break;
                }
                Some(_) => {}
                None => has_unassigned_literal = true,
            }
        }

        if clause_satisfied {
            continue;
        }
        if !has_unassigned_literal {
            return Some((false, Some(clause_index)));
        }
        every_clause_satisfied = false;
    }

    every_clause_satisfied.then_some((true, None))
}

fn push_node(
    nodes: &mut Vec<QbfProofNode>,
    node: QbfProofNode,
    budget: QuantifiedProofBudget,
) -> Result<usize, QuantifiedProofError> {
    if nodes.len() >= budget.max_proof_nodes {
        return Err(QuantifiedProofError::ProofNodeBudgetExceeded {
            required: nodes.len().saturating_add(1),
            limit: budget.max_proof_nodes,
        });
    }
    let id = nodes.len();
    nodes.push(node);
    Ok(id)
}

struct BuilderState {
    nodes: Vec<QbfProofNode>,
    visits: u128,
    early_true: u128,
    early_false: u128,
    matrix_leaves: u128,
}

fn build_node(
    position: usize,
    order: &[(Quantifier, usize)],
    problem: &CnfProblem,
    assignment: &mut [Option<bool>],
    budget: QuantifiedProofBudget,
    state: &mut BuilderState,
) -> Result<usize, QuantifiedProofError> {
    state.visits = state
        .visits
        .checked_add(1)
        .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
    if state.visits > budget.max_builder_node_visits {
        return Err(QuantifiedProofError::BuilderVisitBudgetExceeded {
            required: state.visits,
            limit: budget.max_builder_node_visits,
        });
    }

    if position < order.len() {
        if let Some((value, clause_index)) = partial_matrix_status(problem, assignment) {
            if value {
                state.early_true = state
                    .early_true
                    .checked_add(1)
                    .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
                return push_node(
                    &mut state.nodes,
                    QbfProofNode {
                        prefix_position: position,
                        value: true,
                        kind: QbfProofNodeKind::EarlyTrue,
                    },
                    budget,
                );
            }
            let clause_index =
                clause_index.ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
            state.early_false = state
                .early_false
                .checked_add(1)
                .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
            return push_node(
                &mut state.nodes,
                QbfProofNode {
                    prefix_position: position,
                    value: false,
                    kind: QbfProofNodeKind::EarlyFalseClause { clause_index },
                },
                budget,
            );
        }
    }

    if position == order.len() {
        state.matrix_leaves = state
            .matrix_leaves
            .checked_add(1)
            .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
        let value = assignment_value(problem, assignment);
        return push_node(
            &mut state.nodes,
            QbfProofNode {
                prefix_position: position,
                value,
                kind: QbfProofNodeKind::MatrixLeaf,
            },
            budget,
        );
    }

    let (quantifier, variable) = order[position];
    assignment[variable] = Some(false);
    let low_child = build_node(position + 1, order, problem, assignment, budget, state)?;
    assignment[variable] = Some(true);
    let high_child = build_node(position + 1, order, problem, assignment, budget, state)?;
    assignment[variable] = None;

    let low_value = state.nodes[low_child].value;
    let high_value = state.nodes[high_child].value;
    let value = match quantifier {
        Quantifier::Exists => low_value || high_value,
        Quantifier::ForAll => low_value && high_value,
    };
    let kind = match quantifier {
        Quantifier::Exists => QbfProofNodeKind::ExistsDecision {
            variable,
            low_child,
            high_child,
        },
        Quantifier::ForAll => QbfProofNodeKind::ForAllDecision {
            variable,
            low_child,
            high_child,
        },
    };

    push_node(
        &mut state.nodes,
        QbfProofNode {
            prefix_position: position,
            value,
            kind,
        },
        budget,
    )
}

/// Build a post-order game proof with early matrix-pruning witnesses.
pub fn build_qbf_game_proof(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: QuantifiedProofBudget,
) -> Result<(QbfGameProof, QbfProofBuildReport), QuantifiedProofError> {
    validate_source(problem, prefix, budget)?;
    let order = ordered_prefix(prefix);
    let mut state = BuilderState {
        nodes: Vec::new(),
        visits: 0,
        early_true: 0,
        early_false: 0,
        matrix_leaves: 0,
    };
    let mut assignment = vec![None; problem.variables.len()];
    let root = build_node(0, &order, problem, &mut assignment, budget, &mut state)?;
    let proof = QbfGameProof {
        variable_count: problem.variables.len(),
        root,
        nodes: state.nodes,
    };
    let expansion_shift = u32::try_from(order.len().saturating_add(1))
        .map_err(|_| QuantifiedProofError::ResourceAccountingOverflow)?;
    let full_expansion_node_bound = 1u128
        .checked_shl(expansion_shift)
        .and_then(|value| value.checked_sub(1))
        .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
    let proof_node_count = u128::try_from(proof.nodes.len())
        .map_err(|_| QuantifiedProofError::ResourceAccountingOverflow)?;
    let proof_nodes_saved_vs_full_expansion = full_expansion_node_bound
        .checked_sub(proof_node_count)
        .ok_or(QuantifiedProofError::ProofNodeCountExceedsExpansionBound {
            proof_nodes: proof_node_count,
            full_expansion_nodes: full_expansion_node_bound,
        })?;
    let report = QbfProofBuildReport {
        proof_nodes: proof.nodes.len(),
        builder_node_visits: state.visits,
        early_true_prunes: state.early_true,
        early_false_prunes: state.early_false,
        complete_matrix_leaves: state.matrix_leaves,
        full_expansion_node_bound,
        proof_nodes_saved_vs_full_expansion,
        root_value: proof.nodes[proof.root].value,
    };
    Ok((proof, report))
}

fn verify_partial_true(problem: &CnfProblem, assignment: &[Option<bool>]) -> bool {
    problem.clauses.iter().all(|clause| {
        clause.iter().any(|literal| {
            assignment[literal.variable].is_some_and(|value| value == literal.positive)
        })
    })
}

fn verify_falsified_clause(
    problem: &CnfProblem,
    assignment: &[Option<bool>],
    clause_index: usize,
) -> bool {
    let Some(clause) = problem.clauses.get(clause_index) else {
        return false;
    };
    clause
        .iter()
        .all(|literal| assignment[literal.variable].is_some_and(|value| value != literal.positive))
}

struct VerifierState {
    visits: u128,
    early_true: u128,
    early_false: u128,
    matrix_leaves: u128,
    decisions: u128,
}

fn verify_node(
    node_id: usize,
    expected_position: usize,
    order: &[(Quantifier, usize)],
    problem: &CnfProblem,
    proof: &QbfGameProof,
    assignment: &mut [Option<bool>],
    budget: QuantifiedProofBudget,
    state: &mut VerifierState,
) -> Result<bool, QuantifiedProofError> {
    state.visits = state
        .visits
        .checked_add(1)
        .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
    if state.visits > budget.max_verifier_node_visits {
        return Err(QuantifiedProofError::VerifierVisitBudgetExceeded {
            required: state.visits,
            limit: budget.max_verifier_node_visits,
        });
    }
    let node = proof
        .nodes
        .get(node_id)
        .ok_or(QuantifiedProofError::RootOutOfRange {
            root: node_id,
            node_count: proof.nodes.len(),
        })?;
    if node.prefix_position != expected_position {
        return Err(QuantifiedProofError::PrefixPositionMismatch {
            node: node_id,
            expected: expected_position,
            actual: node.prefix_position,
        });
    }

    let expected_value = match &node.kind {
        QbfProofNodeKind::MatrixLeaf => {
            if expected_position != order.len() {
                return Err(QuantifiedProofError::MatrixLeafBeforePrefixEnd {
                    node: node_id,
                    position: expected_position,
                    prefix_len: order.len(),
                });
            }
            state.matrix_leaves = state
                .matrix_leaves
                .checked_add(1)
                .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
            let value = assignment_value(problem, assignment);
            if value != node.value {
                return Err(QuantifiedProofError::MatrixLeafValueMismatch {
                    node: node_id,
                    expected: value,
                    actual: node.value,
                });
            }
            value
        }
        QbfProofNodeKind::EarlyTrue => {
            if expected_position >= order.len() || !verify_partial_true(problem, assignment) {
                return Err(QuantifiedProofError::EarlyTrueNotEstablished { node: node_id });
            }
            state.early_true = state
                .early_true
                .checked_add(1)
                .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
            true
        }
        QbfProofNodeKind::EarlyFalseClause { clause_index } => {
            if expected_position >= order.len() {
                return Err(QuantifiedProofError::EarlyFalseClauseNotEstablished {
                    node: node_id,
                    clause_index: *clause_index,
                });
            }
            if *clause_index >= problem.clauses.len() {
                return Err(QuantifiedProofError::EarlyFalseClauseOutOfRange {
                    node: node_id,
                    clause_index: *clause_index,
                    clause_count: problem.clauses.len(),
                });
            }
            if !verify_falsified_clause(problem, assignment, *clause_index) {
                return Err(QuantifiedProofError::EarlyFalseClauseNotEstablished {
                    node: node_id,
                    clause_index: *clause_index,
                });
            }
            state.early_false = state
                .early_false
                .checked_add(1)
                .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
            false
        }
        QbfProofNodeKind::ExistsDecision {
            variable,
            low_child,
            high_child,
        } => {
            let Some(&(quantifier, expected_variable)) = order.get(expected_position) else {
                return Err(QuantifiedProofError::DecisionVariableMismatch {
                    node: node_id,
                    expected: usize::MAX,
                    actual: *variable,
                });
            };
            if quantifier != Quantifier::Exists {
                return Err(QuantifiedProofError::QuantifierRuleMismatch {
                    node: node_id,
                    expected: quantifier,
                });
            }
            if *variable != expected_variable {
                return Err(QuantifiedProofError::DecisionVariableMismatch {
                    node: node_id,
                    expected: expected_variable,
                    actual: *variable,
                });
            }
            state.decisions = state
                .decisions
                .checked_add(1)
                .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
            assignment[*variable] = Some(false);
            let low = verify_node(
                *low_child,
                expected_position + 1,
                order,
                problem,
                proof,
                assignment,
                budget,
                state,
            )?;
            assignment[*variable] = Some(true);
            let high = verify_node(
                *high_child,
                expected_position + 1,
                order,
                problem,
                proof,
                assignment,
                budget,
                state,
            )?;
            assignment[*variable] = None;
            low || high
        }
        QbfProofNodeKind::ForAllDecision {
            variable,
            low_child,
            high_child,
        } => {
            let Some(&(quantifier, expected_variable)) = order.get(expected_position) else {
                return Err(QuantifiedProofError::DecisionVariableMismatch {
                    node: node_id,
                    expected: usize::MAX,
                    actual: *variable,
                });
            };
            if quantifier != Quantifier::ForAll {
                return Err(QuantifiedProofError::QuantifierRuleMismatch {
                    node: node_id,
                    expected: quantifier,
                });
            }
            if *variable != expected_variable {
                return Err(QuantifiedProofError::DecisionVariableMismatch {
                    node: node_id,
                    expected: expected_variable,
                    actual: *variable,
                });
            }
            state.decisions = state
                .decisions
                .checked_add(1)
                .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
            assignment[*variable] = Some(false);
            let low = verify_node(
                *low_child,
                expected_position + 1,
                order,
                problem,
                proof,
                assignment,
                budget,
                state,
            )?;
            assignment[*variable] = Some(true);
            let high = verify_node(
                *high_child,
                expected_position + 1,
                order,
                problem,
                proof,
                assignment,
                budget,
                state,
            )?;
            assignment[*variable] = None;
            low && high
        }
    };

    if expected_value != node.value {
        return Err(QuantifiedProofError::IncorrectNodeValue {
            node: node_id,
            expected: expected_value,
            actual: node.value,
        });
    }
    Ok(expected_value)
}

fn validate_tree_shape(proof: &QbfGameProof) -> Result<(), QuantifiedProofError> {
    if proof.nodes.is_empty() {
        return Err(QuantifiedProofError::MissingRoot);
    }
    if proof.root >= proof.nodes.len() {
        return Err(QuantifiedProofError::RootOutOfRange {
            root: proof.root,
            node_count: proof.nodes.len(),
        });
    }
    let final_node = proof.nodes.len() - 1;
    if proof.root != final_node {
        return Err(QuantifiedProofError::RootNotFinalNode {
            root: proof.root,
            final_node,
        });
    }

    let mut parent_counts = vec![0usize; proof.nodes.len()];
    for (parent, node) in proof.nodes.iter().enumerate() {
        let children: [Option<usize>; 2] = match &node.kind {
            QbfProofNodeKind::ExistsDecision {
                low_child,
                high_child,
                ..
            }
            | QbfProofNodeKind::ForAllDecision {
                low_child,
                high_child,
                ..
            } => [Some(*low_child), Some(*high_child)],
            _ => [None, None],
        };
        if let (Some(low), Some(high)) = (children[0], children[1]) {
            if low == high {
                return Err(QuantifiedProofError::DuplicateChildReference { parent, child: low });
            }
        }
        for child in children.into_iter().flatten() {
            if child >= proof.nodes.len() {
                return Err(QuantifiedProofError::ChildOutOfRange {
                    parent,
                    child,
                    node_count: proof.nodes.len(),
                });
            }
            if child >= parent {
                return Err(QuantifiedProofError::ChildNotPostOrder { parent, child });
            }
            parent_counts[child] = parent_counts[child]
                .checked_add(1)
                .ok_or(QuantifiedProofError::ResourceAccountingOverflow)?;
            if parent_counts[child] > 1 {
                return Err(QuantifiedProofError::MultipleParents {
                    child,
                    parents: parent_counts[child],
                });
            }
        }
    }

    for (node, parent_count) in parent_counts.iter().enumerate() {
        let expected = usize::from(node != proof.root);
        if *parent_count != expected {
            return Err(QuantifiedProofError::UnreachableNode { node });
        }
    }
    Ok(())
}

/// Independently verify a game proof against the original CNF and prefix.
pub fn verify_qbf_game_proof(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    proof: &QbfGameProof,
    budget: QuantifiedProofBudget,
) -> Result<QbfProofVerificationReport, QuantifiedProofError> {
    validate_source(problem, prefix, budget)?;
    if proof.variable_count != problem.variables.len() {
        return Err(QuantifiedProofError::ProofVariableCountMismatch {
            proof: proof.variable_count,
            source: problem.variables.len(),
        });
    }
    if proof.nodes.len() > budget.max_proof_nodes {
        return Err(QuantifiedProofError::ProofNodeBudgetExceeded {
            required: proof.nodes.len(),
            limit: budget.max_proof_nodes,
        });
    }
    validate_tree_shape(proof)?;

    let order = ordered_prefix(prefix);
    let mut assignment = vec![None; problem.variables.len()];
    let mut state = VerifierState {
        visits: 0,
        early_true: 0,
        early_false: 0,
        matrix_leaves: 0,
        decisions: 0,
    };
    let root_value = verify_node(
        proof.root,
        0,
        &order,
        problem,
        proof,
        &mut assignment,
        budget,
        &mut state,
    )?;

    Ok(QbfProofVerificationReport {
        proof_nodes_checked: proof.nodes.len(),
        verifier_node_visits: state.visits,
        early_true_prunes_checked: state.early_true,
        early_false_clause_witnesses_checked: state.early_false,
        complete_matrix_leaves_checked: state.matrix_leaves,
        decision_nodes_checked: state.decisions,
        root_value,
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

    fn budget() -> QuantifiedProofBudget {
        QuantifiedProofBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 64,
            max_literals: 256,
            max_proof_nodes: 1 << 10,
            max_builder_node_visits: 1 << 10,
            max_verifier_node_visits: 1 << 10,
        }
    }

    fn block(quantifier: Quantifier, variables: &[usize]) -> QuantifierBlock {
        QuantifierBlock {
            quantifier,
            variables: variables.to_vec(),
        }
    }

    fn make_prefix(variables: usize, blocks: Vec<QuantifierBlock>) -> ValidatedQuantifierPrefix {
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

    fn assert_matches_strategy_oracle(
        problem: &CnfProblem,
        prefix: &ValidatedQuantifierPrefix,
    ) -> QbfProofVerificationReport {
        let (proof, built) = build_qbf_game_proof(problem, prefix, budget()).unwrap();
        let verified = verify_qbf_game_proof(problem, prefix, &proof, budget()).unwrap();
        let oracle =
            evaluate_qbf_with_strategy(problem, prefix, QuantifiedEvaluationBudget::default())
                .unwrap();
        assert_eq!(built.root_value, verified.root_value);
        assert_eq!(built.root_value, oracle.value);
        assert_eq!(verified.proof_nodes_checked, proof.nodes.len());
        assert_eq!(verified.verifier_node_visits, proof.nodes.len() as u128);
        verified
    }

    #[test]
    fn forall_exists_equality_has_true_game_proof() {
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        assert!(assert_matches_strategy_oracle(&equality_problem(), &prefix).root_value);
    }

    #[test]
    fn exists_forall_equality_has_false_game_proof() {
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[0]),
            ],
        );
        assert!(!assert_matches_strategy_oracle(&equality_problem(), &prefix).root_value);
    }

    #[test]
    fn forall_forall_equality_has_false_game_proof() {
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::ForAll, &[1]),
            ],
        );
        assert!(!assert_matches_strategy_oracle(&equality_problem(), &prefix).root_value);
    }

    #[test]
    fn exists_exists_equality_has_true_game_proof() {
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::Exists, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        assert!(assert_matches_strategy_oracle(&equality_problem(), &prefix).root_value);
    }

    #[test]
    fn empty_prefix_and_empty_matrix_is_true() {
        let problem = CnfProblem {
            variables: vec![],
            clauses: vec![],
        };
        let prefix = make_prefix(0, vec![]);
        let report = assert_matches_strategy_oracle(&problem, &prefix);
        assert!(report.root_value);
        assert_eq!(report.complete_matrix_leaves_checked, 1);
    }

    #[test]
    fn empty_clause_has_an_early_false_witness() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![]],
        };
        let prefix = make_prefix(1, vec![block(Quantifier::ForAll, &[0])]);
        let (proof, built) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        assert!(!built.root_value);
        assert_eq!(built.early_false_prunes, 1);
        assert_eq!(built.full_expansion_node_bound, 3);
        assert_eq!(built.proof_nodes_saved_vs_full_expansion, 2);
        assert_eq!(
            proof.nodes[proof.root].kind,
            QbfProofNodeKind::EarlyFalseClause { clause_index: 0 }
        );
        assert!(
            !verify_qbf_game_proof(&problem, &prefix, &proof, budget())
                .unwrap()
                .root_value
        );
    }

    #[test]
    fn all_clauses_satisfied_yields_early_true_child_proof() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::Exists, &[0]),
                block(Quantifier::ForAll, &[1]),
            ],
        );
        let (proof, built) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        assert!(built.root_value);
        assert_eq!(built.early_true_prunes, 1);
        assert_eq!(built.proof_nodes, 3);
        assert_eq!(built.full_expansion_node_bound, 7);
        assert_eq!(built.proof_nodes_saved_vs_full_expansion, 4);
        match &proof.nodes[proof.root].kind {
            QbfProofNodeKind::ExistsDecision { high_child, .. } => {
                assert_eq!(proof.nodes[*high_child].kind, QbfProofNodeKind::EarlyTrue)
            }
            other => panic!("expected existential decision, got {other:?}"),
        }
        let verified = verify_qbf_game_proof(&problem, &prefix, &proof, budget()).unwrap();
        assert!(verified.root_value);
        assert_eq!(verified.early_true_prunes_checked, 1);
    }

    #[test]
    fn early_false_witness_rejects_clause_with_unassigned_literal() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![vec![Literal::positive(0)], vec![Literal::positive(1)]],
        };
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::ForAll, &[1]),
            ],
        );
        let (proof, _) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        let mut tampered = proof.clone();
        let root = tampered.root;
        let low_child = match &tampered.nodes[root].kind {
            QbfProofNodeKind::ForAllDecision { low_child, .. } => *low_child,
            _ => unreachable!(),
        };
        tampered.nodes[low_child].kind = QbfProofNodeKind::EarlyFalseClause { clause_index: 1 };
        assert!(matches!(
            verify_qbf_game_proof(&problem, &prefix, &tampered, budget()),
            Err(QuantifiedProofError::EarlyFalseClauseNotEstablished { .. })
                | Err(QuantifiedProofError::PrefixPositionMismatch { .. })
        ));
    }

    #[test]
    fn verifier_rejects_mutated_decision_value() {
        let problem = equality_problem();
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (proof, _) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        let mut tampered = proof.clone();
        tampered.nodes[tampered.root].value = false;
        assert!(matches!(
            verify_qbf_game_proof(&problem, &prefix, &tampered, budget()),
            Err(QuantifiedProofError::IncorrectNodeValue { .. })
        ));
    }

    #[test]
    fn verifier_rejects_wrong_decision_variable() {
        let problem = equality_problem();
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (proof, _) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        let mut tampered = proof.clone();
        if let QbfProofNodeKind::ForAllDecision { variable, .. } =
            &mut tampered.nodes[tampered.root].kind
        {
            *variable = 1;
        }
        assert!(matches!(
            verify_qbf_game_proof(&problem, &prefix, &tampered, budget()),
            Err(QuantifiedProofError::DecisionVariableMismatch { .. })
        ));
    }

    #[test]
    fn verifier_rejects_incorrect_quantifier_rule() {
        let problem = equality_problem();
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (proof, _) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        let mut tampered = proof.clone();
        let root = &mut tampered.nodes[tampered.root];
        if let QbfProofNodeKind::ForAllDecision {
            variable,
            low_child,
            high_child,
        } = &root.kind
        {
            let (variable, low_child, high_child) = (*variable, *low_child, *high_child);
            root.kind = QbfProofNodeKind::ExistsDecision {
                variable,
                low_child,
                high_child,
            };
        }
        assert!(matches!(
            verify_qbf_game_proof(&problem, &prefix, &tampered, budget()),
            Err(QuantifiedProofError::QuantifierRuleMismatch { .. })
        ));
    }

    #[test]
    fn verifier_rejects_forward_child_reference() {
        let problem = equality_problem();
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (proof, _) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        let mut tampered = proof.clone();
        if let QbfProofNodeKind::ForAllDecision { low_child, .. } =
            &mut tampered.nodes[tampered.root].kind
        {
            *low_child = tampered.root;
        }
        assert!(matches!(
            verify_qbf_game_proof(&problem, &prefix, &tampered, budget()),
            Err(QuantifiedProofError::ChildNotPostOrder { .. })
        ));
    }

    #[test]
    fn verifier_rejects_missing_root_and_unreachable_padding() {
        let problem = equality_problem();
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (proof, _) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        let mut missing = proof.clone();
        missing.nodes.clear();
        assert_eq!(
            verify_qbf_game_proof(&problem, &prefix, &missing, budget()).unwrap_err(),
            QuantifiedProofError::MissingRoot
        );

        let mut padded = proof;
        padded.nodes.push(QbfProofNode {
            prefix_position: 2,
            value: true,
            kind: QbfProofNodeKind::MatrixLeaf,
        });
        assert!(matches!(
            verify_qbf_game_proof(&problem, &prefix, &padded, budget()),
            Err(QuantifiedProofError::RootNotFinalNode { .. })
        ));
    }

    #[test]
    fn verifier_rejects_early_true_when_one_clause_is_not_yet_satisfied() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::ForAll, &[1]),
            ],
        );
        let (proof, _) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        let mut tampered = proof.clone();
        let root = tampered.root;
        let low_child = match &tampered.nodes[root].kind {
            QbfProofNodeKind::ForAllDecision { low_child, .. } => *low_child,
            _ => unreachable!(),
        };
        tampered.nodes[low_child].kind = QbfProofNodeKind::EarlyTrue;
        tampered.nodes[low_child].value = true;
        assert!(matches!(
            verify_qbf_game_proof(&problem, &prefix, &tampered, budget()),
            Err(QuantifiedProofError::EarlyTrueNotEstablished { .. })
        ));
    }

    fn direct_game_oracle(problem: &CnfProblem, prefix: &ValidatedQuantifierPrefix) -> bool {
        let order = ordered_prefix(prefix);
        fn visit(
            position: usize,
            order: &[(Quantifier, usize)],
            problem: &CnfProblem,
            assignment: &mut [Option<bool>],
        ) -> bool {
            if position == order.len() {
                return problem.clauses.iter().all(|clause| {
                    clause.iter().any(|literal| {
                        assignment[literal.variable].is_some_and(|value| value == literal.positive)
                    })
                });
            }
            let (quantifier, variable) = order[position];
            assignment[variable] = Some(false);
            let low = visit(position + 1, order, problem, assignment);
            assignment[variable] = Some(true);
            let high = visit(position + 1, order, problem, assignment);
            assignment[variable] = None;
            match quantifier {
                Quantifier::Exists => low || high,
                Quantifier::ForAll => low && high,
            }
        }
        visit(0, &order, problem, &mut vec![None; problem.variables.len()])
    }

    #[test]
    fn exhaustive_two_variable_proofs_match_direct_game_semantics() {
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
                    let prefix = make_prefix(2, blocks);
                    let expected = direct_game_oracle(problem, &prefix);
                    let (proof, built) = build_qbf_game_proof(problem, &prefix, budget()).unwrap();
                    let checked =
                        verify_qbf_game_proof(problem, &prefix, &proof, budget()).unwrap();
                    assert_eq!(
                        built.root_value, expected,
                        "builder formula {formula_index}, order {order:?}, mask {quantifier_mask}"
                    );
                    assert_eq!(
                        checked.root_value, expected,
                        "checker formula {formula_index}, order {order:?}, mask {quantifier_mask}"
                    );
                }
            }
        }
    }

    #[test]
    fn proof_node_budget_fails_closed() {
        let problem = equality_problem();
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let tight = QuantifiedProofBudget {
            max_proof_nodes: 1,
            ..budget()
        };
        assert!(matches!(
            build_qbf_game_proof(&problem, &prefix, tight),
            Err(QuantifiedProofError::ProofNodeBudgetExceeded { .. })
        ));
    }

    #[test]
    fn verifier_visit_budget_fails_closed() {
        let problem = equality_problem();
        let prefix = make_prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (proof, _) = build_qbf_game_proof(&problem, &prefix, budget()).unwrap();
        let tight = QuantifiedProofBudget {
            max_verifier_node_visits: 1,
            ..budget()
        };
        assert!(matches!(
            verify_qbf_game_proof(&problem, &prefix, &proof, tight),
            Err(QuantifiedProofError::VerifierVisitBudgetExceeded { .. })
        ));
    }
}
