// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hash-consed residual-CNF game proof DAGs for bounded prenex QBF.
//!
//! Unlike the reference proof tree, this representation shares equal suffix
//! games. A state is keyed by its prefix position and its structurally
//! canonical residual CNF after assigning the quantified prefix already
//! traversed. The independent verifier validates each residual transition,
//! quantifier recurrence, proof edge and reachability without invoking the
//! builder.
//!
//! This is a bounded proof-representation reference, not a scalable QBF
//! solver, a formal proof-complexity result, or an external QBF certificate
//! format.

use super::decomposition::CnfProblem;
use super::quantified_scope::{Quantifier, ValidatedQuantifierPrefix};
use std::collections::BTreeMap;

/// One literal in a residual CNF clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResidualLiteral {
    pub variable: usize,
    pub positive: bool,
}

/// A deterministic canonical residual CNF.
///
/// Literals in each clause and clauses in the CNF are sorted. Duplicate
/// literals/clauses and tautological clauses are removed. Empty clauses are
/// retained because they represent FALSE.
pub type ResidualCnf = Vec<Vec<ResidualLiteral>>;

/// Explicit limits for residual-DAG construction and checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QbfGameDagBudget {
    pub max_variables: usize,
    pub max_total_variable_name_bytes: usize,
    pub max_clauses: usize,
    pub max_literals: usize,
    pub max_proof_nodes: usize,
    pub max_builder_state_visits: u128,
    pub max_verifier_node_visits: u128,
    pub max_residual_literal_visits: u128,
    pub max_stored_residual_literals: u128,
    pub max_stored_residual_clauses: u128,
}

impl Default for QbfGameDagBudget {
    fn default() -> Self {
        Self {
            max_variables: 20,
            max_total_variable_name_bytes: 4 * 1024,
            max_clauses: 1 << 16,
            max_literals: 1 << 20,
            max_proof_nodes: 1 << 20,
            max_builder_state_visits: 1 << 20,
            max_verifier_node_visits: 1 << 20,
            max_residual_literal_visits: 1 << 24,
            max_stored_residual_literals: 1 << 24,
            max_stored_residual_clauses: 1 << 22,
        }
    }
}

/// A local residual-game proof rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QbfGameDagNodeKind {
    /// No residual clauses remain, so every completion satisfies the matrix.
    TrueTerminal,
    /// At least one empty residual clause remains, so every completion fails.
    FalseTerminal,
    /// Value is low.value OR high.value.
    ExistsDecision {
        variable: usize,
        low_child: usize,
        high_child: usize,
    },
    /// Value is low.value AND high.value.
    ForAllDecision {
        variable: usize,
        low_child: usize,
        high_child: usize,
    },
}

/// A proof node stores its own residual state so every edge can be checked
/// locally against the source CNF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QbfGameDagNode {
    pub prefix_position: usize,
    pub residual: ResidualCnf,
    pub value: bool,
    pub kind: QbfGameDagNodeKind,
}

/// A post-order DAG bound to an exact ordered quantifier prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QbfGameProofDag {
    pub variable_count: usize,
    pub prefix_order: Vec<(Quantifier, usize)>,
    pub root: usize,
    pub nodes: Vec<QbfGameDagNode>,
}

/// Builder measurements. Stored residual counts include both output-node
/// residuals and transient memo-key copies while the builder is live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QbfGameDagBuildReport {
    pub proof_nodes: usize,
    pub builder_state_visits: u128,
    pub memo_hits: u128,
    pub true_terminal_nodes: u128,
    pub false_terminal_nodes: u128,
    pub decision_nodes: u128,
    pub residual_literal_visits: u128,
    pub peak_stored_residual_literals: u128,
    pub peak_stored_residual_clauses: u128,
    pub full_binary_tree_node_bound: u128,
    pub proof_nodes_saved_vs_full_tree: u128,
    pub root_value: bool,
}

/// Independently derived evidence from the DAG verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QbfGameDagVerificationReport {
    pub proof_nodes_checked: usize,
    pub verifier_node_visits: u128,
    pub residual_literal_visits: u128,
    pub shared_child_references: u128,
    pub shared_states: u128,
    pub root_value: bool,
}

/// Fail-closed construction and verification failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QbfGameDagError {
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
    VerifierVisitBudgetExceeded {
        required: u128,
        limit: u128,
    },
    ProofNodeBudgetExceeded {
        required: usize,
        limit: usize,
    },
    ResidualLiteralVisitBudgetExceeded {
        required: u128,
        limit: u128,
    },
    StoredResidualLiteralBudgetExceeded {
        required: u128,
        limit: u128,
    },
    StoredResidualClauseBudgetExceeded {
        required: u128,
        limit: u128,
    },
    ResourceAccountingOverflow,
    MissingRoot,
    RootOutOfRange {
        root: usize,
        node_count: usize,
    },
    RootNotFinalNode {
        root: usize,
        final_node: usize,
    },
    ProofVariableCountMismatch {
        proof: usize,
        source: usize,
    },
    PrefixBindingMismatch,
    NodePrefixPositionOutOfRange {
        node: usize,
        position: usize,
        prefix_len: usize,
    },
    NonCanonicalResidual {
        node: usize,
    },
    ResidualVariableOutOfRange {
        node: usize,
        variable: usize,
        variable_count: usize,
    },
    ResidualVariableBeforeSuffix {
        node: usize,
        variable: usize,
        prefix_position: usize,
        variable_position: usize,
    },
    IncorrectTerminalKind {
        node: usize,
    },
    DecisionAfterPrefixEnd {
        node: usize,
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
    ChildOutOfRange {
        parent: usize,
        child: usize,
        node_count: usize,
    },
    ChildNotPostOrder {
        parent: usize,
        child: usize,
    },
    ChildPrefixPositionMismatch {
        parent: usize,
        child: usize,
        expected: usize,
        actual: usize,
    },
    ChildResidualMismatch {
        parent: usize,
        child: usize,
        branch_value: bool,
    },
    IncorrectNodeValue {
        node: usize,
        expected: bool,
        actual: bool,
    },
    DuplicateStateNode {
        earlier: usize,
        duplicate: usize,
    },
    UnreachableNode {
        node: usize,
    },
    MatrixResidualMismatch,
    ProofRootValueMismatch {
        expected: bool,
        actual: bool,
    },
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

fn variable_positions(variable_count: usize, order: &[(Quantifier, usize)]) -> Vec<usize> {
    let mut positions = vec![usize::MAX; variable_count];
    for (position, &(_, variable)) in order.iter().enumerate() {
        positions[variable] = position;
    }
    positions
}

fn validate_source(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: QbfGameDagBudget,
) -> Result<(), QbfGameDagError> {
    if problem.variables.len() != prefix.variable_count() {
        return Err(QbfGameDagError::MatrixPrefixVariableCountMismatch {
            matrix: problem.variables.len(),
            prefix: prefix.variable_count(),
        });
    }
    if problem.variables.len() > budget.max_variables {
        return Err(QbfGameDagError::VariableBudgetExceeded {
            required: problem.variables.len(),
            limit: budget.max_variables,
        });
    }

    let name_bytes = problem.variables.iter().try_fold(0usize, |total, name| {
        total
            .checked_add(name.len())
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)
    })?;
    if name_bytes > budget.max_total_variable_name_bytes {
        return Err(QbfGameDagError::VariableNameBudgetExceeded {
            required: name_bytes,
            limit: budget.max_total_variable_name_bytes,
        });
    }

    let mut names = std::collections::BTreeSet::new();
    for (index, name) in problem.variables.iter().enumerate() {
        if name.is_empty() {
            return Err(QbfGameDagError::EmptyVariableName { index });
        }
        if !names.insert(name.as_str()) {
            return Err(QbfGameDagError::DuplicateVariableName(name.clone()));
        }
    }

    if problem.clauses.len() > budget.max_clauses {
        return Err(QbfGameDagError::ClauseBudgetExceeded {
            required: problem.clauses.len(),
            limit: budget.max_clauses,
        });
    }
    let literal_count = problem.clauses.iter().try_fold(0usize, |total, clause| {
        total
            .checked_add(clause.len())
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)
    })?;
    if literal_count > budget.max_literals {
        return Err(QbfGameDagError::LiteralBudgetExceeded {
            required: literal_count,
            limit: budget.max_literals,
        });
    }

    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        for (literal_index, literal) in clause.iter().enumerate() {
            if literal.variable >= problem.variables.len() {
                return Err(QbfGameDagError::LiteralVariableOutOfRange {
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

/// Canonicalize residual clauses. Tautologies are removed, empty clauses are
/// preserved, and structural ordering makes map keys collision-free.
fn canonicalize_residual(
    clauses: Vec<Vec<ResidualLiteral>>,
) -> Result<(ResidualCnf, u128), QbfGameDagError> {
    let mut literal_visits = 0u128;
    let mut canonical = Vec::with_capacity(clauses.len());
    for mut clause in clauses {
        literal_visits = literal_visits
            .checked_add(
                u128::try_from(clause.len())
                    .map_err(|_| QbfGameDagError::ResourceAccountingOverflow)?,
            )
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        clause.sort_unstable();
        clause.dedup();

        let tautology = clause.windows(2).any(|pair| {
            pair[0].variable == pair[1].variable && pair[0].positive != pair[1].positive
        });
        if !tautology {
            canonical.push(clause);
        }
    }
    canonical.sort_unstable();
    canonical.dedup();
    Ok((canonical, literal_visits))
}

fn source_residual(problem: &CnfProblem) -> Result<(ResidualCnf, u128), QbfGameDagError> {
    canonicalize_residual(
        problem
            .clauses
            .iter()
            .map(|clause| {
                clause
                    .iter()
                    .map(|literal| ResidualLiteral {
                        variable: literal.variable,
                        positive: literal.positive,
                    })
                    .collect()
            })
            .collect(),
    )
}

fn simplify_residual(
    residual: &ResidualCnf,
    variable: usize,
    value: bool,
) -> Result<(ResidualCnf, u128), QbfGameDagError> {
    let mut clauses = Vec::with_capacity(residual.len());
    let mut scanned_literals = 0u128;
    for clause in residual {
        let mut satisfied = false;
        let mut reduced = Vec::with_capacity(clause.len());
        for literal in clause {
            scanned_literals = scanned_literals
                .checked_add(1)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
            if literal.variable == variable {
                if literal.positive == value {
                    satisfied = true;
                    break;
                }
            } else {
                reduced.push(*literal);
            }
        }
        if !satisfied {
            clauses.push(reduced);
        }
    }
    let (canonical, canonicalization_visits) = canonicalize_residual(clauses)?;
    let total_visits = scanned_literals
        .checked_add(canonicalization_visits)
        .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
    Ok((canonical, total_visits))
}

fn residual_literal_count(residual: &ResidualCnf) -> Result<u128, QbfGameDagError> {
    residual.iter().try_fold(0u128, |total, clause| {
        total
            .checked_add(
                u128::try_from(clause.len())
                    .map_err(|_| QbfGameDagError::ResourceAccountingOverflow)?,
            )
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)
    })
}

fn residual_clause_count(residual: &ResidualCnf) -> Result<u128, QbfGameDagError> {
    u128::try_from(residual.len()).map_err(|_| QbfGameDagError::ResourceAccountingOverflow)
}

#[derive(Default)]
struct BuilderState {
    nodes: Vec<QbfGameDagNode>,
    memo: BTreeMap<(usize, ResidualCnf), usize>,
    visits: u128,
    memo_hits: u128,
    true_terminals: u128,
    false_terminals: u128,
    decisions: u128,
    literal_visits: u128,
    stored_literals: u128,
    stored_clauses: u128,
}

impl BuilderState {
    fn account_canonicalization(
        &mut self,
        work: u128,
        budget: QbfGameDagBudget,
    ) -> Result<(), QbfGameDagError> {
        self.literal_visits = self
            .literal_visits
            .checked_add(work)
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        if self.literal_visits > budget.max_residual_literal_visits {
            return Err(QbfGameDagError::ResidualLiteralVisitBudgetExceeded {
                required: self.literal_visits,
                limit: budget.max_residual_literal_visits,
            });
        }
        Ok(())
    }

    fn push_state(
        &mut self,
        position: usize,
        residual: ResidualCnf,
        kind: QbfGameDagNodeKind,
        value: bool,
        budget: QbfGameDagBudget,
    ) -> Result<usize, QbfGameDagError> {
        if self.nodes.len() >= budget.max_proof_nodes {
            return Err(QbfGameDagError::ProofNodeBudgetExceeded {
                required: self.nodes.len().saturating_add(1),
                limit: budget.max_proof_nodes,
            });
        }
        let literals = residual_literal_count(&residual)?;
        let clauses = residual_clause_count(&residual)?;
        // The memo key and output proof node each own a residual copy.
        let next_literals = self
            .stored_literals
            .checked_add(
                literals
                    .checked_mul(2)
                    .ok_or(QbfGameDagError::ResourceAccountingOverflow)?,
            )
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        let next_clauses = self
            .stored_clauses
            .checked_add(
                clauses
                    .checked_mul(2)
                    .ok_or(QbfGameDagError::ResourceAccountingOverflow)?,
            )
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        if next_literals > budget.max_stored_residual_literals {
            return Err(QbfGameDagError::StoredResidualLiteralBudgetExceeded {
                required: next_literals,
                limit: budget.max_stored_residual_literals,
            });
        }
        if next_clauses > budget.max_stored_residual_clauses {
            return Err(QbfGameDagError::StoredResidualClauseBudgetExceeded {
                required: next_clauses,
                limit: budget.max_stored_residual_clauses,
            });
        }

        let key = (position, residual.clone());
        let id = self.nodes.len();
        self.nodes.push(QbfGameDagNode {
            prefix_position: position,
            residual,
            value,
            kind,
        });
        self.memo.insert(key, id);
        self.stored_literals = next_literals;
        self.stored_clauses = next_clauses;
        Ok(id)
    }
}

fn build_node(
    position: usize,
    order: &[(Quantifier, usize)],
    residual: ResidualCnf,
    budget: QbfGameDagBudget,
    state: &mut BuilderState,
) -> Result<usize, QbfGameDagError> {
    state.visits = state
        .visits
        .checked_add(1)
        .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
    if state.visits > budget.max_builder_state_visits {
        return Err(QbfGameDagError::BuilderVisitBudgetExceeded {
            required: state.visits,
            limit: budget.max_builder_state_visits,
        });
    }

    let key = (position, residual.clone());
    if let Some(&node) = state.memo.get(&key) {
        state.memo_hits = state
            .memo_hits
            .checked_add(1)
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        return Ok(node);
    }

    if residual.is_empty() {
        state.true_terminals = state
            .true_terminals
            .checked_add(1)
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        return state.push_state(
            position,
            residual,
            QbfGameDagNodeKind::TrueTerminal,
            true,
            budget,
        );
    }
    if residual.iter().any(Vec::is_empty) {
        state.false_terminals = state
            .false_terminals
            .checked_add(1)
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        return state.push_state(
            position,
            residual,
            QbfGameDagNodeKind::FalseTerminal,
            false,
            budget,
        );
    }
    if position >= order.len() {
        return Err(QbfGameDagError::MatrixResidualMismatch);
    }

    let (quantifier, variable) = order[position];
    let (low_residual, low_work) = simplify_residual(&residual, variable, false)?;
    state.account_canonicalization(low_work, budget)?;
    let low_child = build_node(position + 1, order, low_residual, budget, state)?;

    let (high_residual, high_work) = simplify_residual(&residual, variable, true)?;
    state.account_canonicalization(high_work, budget)?;
    let high_child = build_node(position + 1, order, high_residual, budget, state)?;

    let low_value = state.nodes[low_child].value;
    let high_value = state.nodes[high_child].value;
    let value = match quantifier {
        Quantifier::Exists => low_value || high_value,
        Quantifier::ForAll => low_value && high_value,
    };
    let kind = match quantifier {
        Quantifier::Exists => QbfGameDagNodeKind::ExistsDecision {
            variable,
            low_child,
            high_child,
        },
        Quantifier::ForAll => QbfGameDagNodeKind::ForAllDecision {
            variable,
            low_child,
            high_child,
        },
    };
    state.decisions = state
        .decisions
        .checked_add(1)
        .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
    state.push_state(position, residual, kind, value, budget)
}

/// Build a residual-CNF hash-consed proof DAG.
pub fn build_qbf_game_proof_dag(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: QbfGameDagBudget,
) -> Result<(QbfGameProofDag, QbfGameDagBuildReport), QbfGameDagError> {
    validate_source(problem, prefix, budget)?;
    let order = ordered_prefix(prefix);
    let (root_residual, root_work) = source_residual(problem)?;
    let mut state = BuilderState::default();
    state.account_canonicalization(root_work, budget)?;
    let root = build_node(0, &order, root_residual, budget, &mut state)?;

    let expansion_shift = u32::try_from(order.len().saturating_add(1))
        .map_err(|_| QbfGameDagError::ResourceAccountingOverflow)?;
    let full_binary_tree_node_bound = 1u128
        .checked_shl(expansion_shift)
        .and_then(|value| value.checked_sub(1))
        .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
    let dag_nodes = u128::try_from(state.nodes.len())
        .map_err(|_| QbfGameDagError::ResourceAccountingOverflow)?;
    let proof_nodes_saved_vs_full_tree = full_binary_tree_node_bound
        .checked_sub(dag_nodes)
        .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;

    let proof = QbfGameProofDag {
        variable_count: problem.variables.len(),
        prefix_order: order,
        root,
        nodes: state.nodes,
    };
    let report = QbfGameDagBuildReport {
        proof_nodes: proof.nodes.len(),
        builder_state_visits: state.visits,
        memo_hits: state.memo_hits,
        true_terminal_nodes: state.true_terminals,
        false_terminal_nodes: state.false_terminals,
        decision_nodes: state.decisions,
        residual_literal_visits: state.literal_visits,
        peak_stored_residual_literals: state.stored_literals,
        peak_stored_residual_clauses: state.stored_clauses,
        full_binary_tree_node_bound,
        proof_nodes_saved_vs_full_tree,
        root_value: proof.nodes[proof.root].value,
    };
    Ok((proof, report))
}

/// Check canonical residual structure directly, without invoking the
/// builder's canonicalizer. Each clause must have strictly increasing
/// literals, no complementary pair, and the clause list must be strictly
/// ordered (which also excludes duplicate clauses).
fn validate_canonical_residual(
    node_id: usize,
    position: usize,
    residual: &ResidualCnf,
    variable_positions: &[usize],
    budget: QbfGameDagBudget,
    literal_visits: &mut u128,
) -> Result<(), QbfGameDagError> {
    for clause in residual {
        *literal_visits = literal_visits
            .checked_add(
                u128::try_from(clause.len())
                    .map_err(|_| QbfGameDagError::ResourceAccountingOverflow)?,
            )
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;

        if clause.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(QbfGameDagError::NonCanonicalResidual { node: node_id });
        }

        let mut polarity_by_variable = BTreeMap::new();
        for literal in clause {
            if let Some(previous) = polarity_by_variable.insert(literal.variable, literal.positive)
            {
                if previous != literal.positive {
                    return Err(QbfGameDagError::NonCanonicalResidual { node: node_id });
                }
            }
        }

        for literal in clause {
            let Some(&variable_position) = variable_positions.get(literal.variable) else {
                return Err(QbfGameDagError::ResidualVariableOutOfRange {
                    node: node_id,
                    variable: literal.variable,
                    variable_count: variable_positions.len(),
                });
            };
            if variable_position < position {
                return Err(QbfGameDagError::ResidualVariableBeforeSuffix {
                    node: node_id,
                    variable: literal.variable,
                    prefix_position: position,
                    variable_position,
                });
            }
        }
    }

    if residual.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(QbfGameDagError::NonCanonicalResidual { node: node_id });
    }
    if *literal_visits > budget.max_residual_literal_visits {
        return Err(QbfGameDagError::ResidualLiteralVisitBudgetExceeded {
            required: *literal_visits,
            limit: budget.max_residual_literal_visits,
        });
    }
    Ok(())
}

/// Re-derive a child residual using a deliberately separate verifier
/// algorithm: first detect a satisfied clause, then filter the decision
/// variable, then independently normalize the resulting clause set.
fn derive_child_residual_for_verification(
    residual: &ResidualCnf,
    variable: usize,
    value: bool,
) -> Result<(ResidualCnf, u128), QbfGameDagError> {
    let mut work = 0u128;
    let mut derived = Vec::with_capacity(residual.len());

    for clause in residual {
        let mut satisfied = false;
        for literal in clause {
            work = work
                .checked_add(1)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
            if literal.variable == variable && literal.positive == value {
                satisfied = true;
                break;
            }
        }
        if satisfied {
            continue;
        }

        let mut reduced = Vec::with_capacity(clause.len());
        for literal in clause {
            work = work
                .checked_add(1)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
            if literal.variable != variable {
                reduced.push(*literal);
            }
        }
        reduced.sort_unstable();
        reduced.dedup();

        let mut polarity_by_variable = BTreeMap::new();
        let mut tautological = false;
        for literal in &reduced {
            work = work
                .checked_add(1)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
            if let Some(previous) = polarity_by_variable.insert(literal.variable, literal.positive)
            {
                if previous != literal.positive {
                    tautological = true;
                    break;
                }
            }
        }
        if !tautological {
            derived.push(reduced);
        }
    }

    derived.sort_unstable();
    derived.dedup();
    Ok((derived, work))
}

/// Independently canonicalize the original source matrix for root binding.
/// This uses set-based polarity detection rather than the builder's
/// sort/window canonicalizer.
fn verifier_source_residual(problem: &CnfProblem) -> Result<(ResidualCnf, u128), QbfGameDagError> {
    let mut work = 0u128;
    let mut clauses = Vec::with_capacity(problem.clauses.len());
    for clause in &problem.clauses {
        let mut literals = Vec::with_capacity(clause.len());
        for literal in clause {
            work = work
                .checked_add(1)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
            literals.push(ResidualLiteral {
                variable: literal.variable,
                positive: literal.positive,
            });
        }
        literals.sort_unstable();
        literals.dedup();

        let mut polarity_by_variable = BTreeMap::new();
        let mut tautological = false;
        for literal in &literals {
            work = work
                .checked_add(1)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
            if let Some(previous) = polarity_by_variable.insert(literal.variable, literal.positive)
            {
                if previous != literal.positive {
                    tautological = true;
                    break;
                }
            }
        }
        if !tautological {
            clauses.push(literals);
        }
    }
    clauses.sort_unstable();
    clauses.dedup();
    Ok((clauses, work))
}

/// Independently verify a residual-CNF proof DAG against the original source.
pub fn verify_qbf_game_proof_dag(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    proof: &QbfGameProofDag,
    budget: QbfGameDagBudget,
) -> Result<QbfGameDagVerificationReport, QbfGameDagError> {
    validate_source(problem, prefix, budget)?;
    if proof.variable_count != problem.variables.len() {
        return Err(QbfGameDagError::ProofVariableCountMismatch {
            proof: proof.variable_count,
            source: problem.variables.len(),
        });
    }
    if proof.nodes.is_empty() {
        return Err(QbfGameDagError::MissingRoot);
    }
    if proof.root >= proof.nodes.len() {
        return Err(QbfGameDagError::RootOutOfRange {
            root: proof.root,
            node_count: proof.nodes.len(),
        });
    }
    let final_node = proof.nodes.len() - 1;
    if proof.root != final_node {
        return Err(QbfGameDagError::RootNotFinalNode {
            root: proof.root,
            final_node,
        });
    }
    if proof.nodes.len() > budget.max_proof_nodes {
        return Err(QbfGameDagError::ProofNodeBudgetExceeded {
            required: proof.nodes.len(),
            limit: budget.max_proof_nodes,
        });
    }

    let order = ordered_prefix(prefix);
    if proof.prefix_order != order {
        return Err(QbfGameDagError::PrefixBindingMismatch);
    }
    if proof.nodes[proof.root].prefix_position != 0 {
        return Err(QbfGameDagError::NodePrefixPositionOutOfRange {
            node: proof.root,
            position: proof.nodes[proof.root].prefix_position,
            prefix_len: order.len(),
        });
    }

    let (source, source_work) = verifier_source_residual(problem)?;
    if proof.nodes[proof.root].residual != source {
        return Err(QbfGameDagError::MatrixResidualMismatch);
    }
    let positions = variable_positions(problem.variables.len(), &order);

    let mut parent_counts = vec![0usize; proof.nodes.len()];
    let mut seen_states = BTreeMap::new();
    let mut residual_literal_visits = source_work;
    let mut stored_literals = 0u128;
    let mut stored_clauses = 0u128;
    let mut shared_child_references = 0u128;
    let mut verifier_node_visits = 0u128;

    for (node_id, node) in proof.nodes.iter().enumerate() {
        verifier_node_visits = verifier_node_visits
            .checked_add(1)
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        if verifier_node_visits > budget.max_verifier_node_visits {
            return Err(QbfGameDagError::VerifierVisitBudgetExceeded {
                required: verifier_node_visits,
                limit: budget.max_verifier_node_visits,
            });
        }
        if node.prefix_position > order.len() {
            return Err(QbfGameDagError::NodePrefixPositionOutOfRange {
                node: node_id,
                position: node.prefix_position,
                prefix_len: order.len(),
            });
        }

        validate_canonical_residual(
            node_id,
            node.prefix_position,
            &node.residual,
            &positions,
            budget,
            &mut residual_literal_visits,
        )?;

        let literals = residual_literal_count(&node.residual)?;
        let clauses = residual_clause_count(&node.residual)?;
        stored_literals = stored_literals
            .checked_add(literals)
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        stored_clauses = stored_clauses
            .checked_add(clauses)
            .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        if stored_literals > budget.max_stored_residual_literals {
            return Err(QbfGameDagError::StoredResidualLiteralBudgetExceeded {
                required: stored_literals,
                limit: budget.max_stored_residual_literals,
            });
        }
        if stored_clauses > budget.max_stored_residual_clauses {
            return Err(QbfGameDagError::StoredResidualClauseBudgetExceeded {
                required: stored_clauses,
                limit: budget.max_stored_residual_clauses,
            });
        }

        let state_key = (node.prefix_position, node.residual.clone());
        if let Some(&earlier) = seen_states.get(&state_key) {
            return Err(QbfGameDagError::DuplicateStateNode {
                earlier,
                duplicate: node_id,
            });
        }
        seen_states.insert(state_key, node_id);

        if node.residual.is_empty() {
            if node.kind != QbfGameDagNodeKind::TrueTerminal || !node.value {
                return Err(QbfGameDagError::IncorrectTerminalKind { node: node_id });
            }
            continue;
        }
        if node.residual.iter().any(Vec::is_empty) {
            if node.kind != QbfGameDagNodeKind::FalseTerminal || node.value {
                return Err(QbfGameDagError::IncorrectTerminalKind { node: node_id });
            }
            continue;
        }
        if node.prefix_position >= order.len() {
            return Err(QbfGameDagError::DecisionAfterPrefixEnd { node: node_id });
        }

        let (quantifier, variable) = order[node.prefix_position];
        let (actual_variable, low_child, high_child, actual_quantifier) = match &node.kind {
            QbfGameDagNodeKind::ExistsDecision {
                variable,
                low_child,
                high_child,
            } => (*variable, *low_child, *high_child, Quantifier::Exists),
            QbfGameDagNodeKind::ForAllDecision {
                variable,
                low_child,
                high_child,
            } => (*variable, *low_child, *high_child, Quantifier::ForAll),
            QbfGameDagNodeKind::TrueTerminal | QbfGameDagNodeKind::FalseTerminal => {
                return Err(QbfGameDagError::IncorrectTerminalKind { node: node_id });
            }
        };
        if actual_variable != variable {
            return Err(QbfGameDagError::DecisionVariableMismatch {
                node: node_id,
                expected: variable,
                actual: actual_variable,
            });
        }
        if actual_quantifier != quantifier {
            return Err(QbfGameDagError::QuantifierRuleMismatch {
                node: node_id,
                expected: quantifier,
            });
        }

        let child_ids = [low_child, high_child];
        for child in child_ids {
            if child >= proof.nodes.len() {
                return Err(QbfGameDagError::ChildOutOfRange {
                    parent: node_id,
                    child,
                    node_count: proof.nodes.len(),
                });
            }
            if child >= node_id {
                return Err(QbfGameDagError::ChildNotPostOrder {
                    parent: node_id,
                    child,
                });
            }
            parent_counts[child] = parent_counts[child]
                .checked_add(1)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        }
        if low_child == high_child {
            shared_child_references = shared_child_references
                .checked_add(1)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        }

        for (child, branch_value) in [(low_child, false), (high_child, true)] {
            let child_node = &proof.nodes[child];
            let expected_position = node.prefix_position + 1;
            if child_node.prefix_position != expected_position {
                return Err(QbfGameDagError::ChildPrefixPositionMismatch {
                    parent: node_id,
                    child,
                    expected: expected_position,
                    actual: child_node.prefix_position,
                });
            }
            let (expected_residual, work) =
                derive_child_residual_for_verification(&node.residual, variable, branch_value)?;
            residual_literal_visits = residual_literal_visits
                .checked_add(work)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
            if residual_literal_visits > budget.max_residual_literal_visits {
                return Err(QbfGameDagError::ResidualLiteralVisitBudgetExceeded {
                    required: residual_literal_visits,
                    limit: budget.max_residual_literal_visits,
                });
            }
            if child_node.residual != expected_residual {
                return Err(QbfGameDagError::ChildResidualMismatch {
                    parent: node_id,
                    child,
                    branch_value,
                });
            }
        }

        let low_value = proof.nodes[low_child].value;
        let high_value = proof.nodes[high_child].value;
        let expected_value = match quantifier {
            Quantifier::Exists => low_value || high_value,
            Quantifier::ForAll => low_value && high_value,
        };
        if node.value != expected_value {
            return Err(QbfGameDagError::IncorrectNodeValue {
                node: node_id,
                expected: expected_value,
                actual: node.value,
            });
        }
    }

    for (node, parents) in parent_counts.iter().enumerate() {
        if node == proof.root {
            if *parents != 0 {
                return Err(QbfGameDagError::UnreachableNode { node });
            }
        } else if *parents == 0 {
            return Err(QbfGameDagError::UnreachableNode { node });
        }
    }

    let mut shared_states = 0u128;
    for count in parent_counts {
        if count > 1 {
            shared_states = shared_states
                .checked_add(1)
                .ok_or(QbfGameDagError::ResourceAccountingOverflow)?;
        }
    }

    Ok(QbfGameDagVerificationReport {
        proof_nodes_checked: proof.nodes.len(),
        verifier_node_visits,
        residual_literal_visits,
        shared_child_references,
        shared_states,
        root_value: proof.nodes[proof.root].value,
    })
}

#[cfg(test)]
mod tests {
    use super::super::decomposition::Literal;
    use super::super::quantified_proof::{
        QuantifiedProofBudget, build_qbf_game_proof, verify_qbf_game_proof,
    };
    use super::super::quantified_scope::QuantifierBlock;
    use super::super::quantified_strategy::{
        QuantifiedEvaluationBudget, evaluate_qbf_with_strategy,
    };
    use super::*;

    fn budget() -> QbfGameDagBudget {
        QbfGameDagBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 64,
            max_literals: 256,
            max_proof_nodes: 1 << 12,
            max_builder_state_visits: 1 << 12,
            max_verifier_node_visits: 1 << 12,
            max_residual_literal_visits: 1 << 16,
            max_stored_residual_literals: 1 << 16,
            max_stored_residual_clauses: 1 << 14,
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
                vec![Literal::positive(0), Literal::positive(1)],
                vec![Literal::negative(0), Literal::positive(1)],
            ],
        }
    }

    fn assert_all_three_paths_agree(
        problem: &CnfProblem,
        prefix: &ValidatedQuantifierPrefix,
    ) -> (QbfGameProofDag, QbfGameDagBuildReport) {
        let (dag, built) = build_qbf_game_proof_dag(problem, prefix, budget()).unwrap();
        let checked = verify_qbf_game_proof_dag(problem, prefix, &dag, budget()).unwrap();
        let (tree, _) =
            build_qbf_game_proof(problem, prefix, QuantifiedProofBudget::default()).unwrap();
        let tree_verified =
            verify_qbf_game_proof(problem, prefix, &tree, QuantifiedProofBudget::default())
                .unwrap();
        let strategy =
            evaluate_qbf_with_strategy(problem, prefix, QuantifiedEvaluationBudget::default())
                .unwrap();
        assert_eq!(built.root_value, checked.root_value);
        assert_eq!(built.root_value, tree_verified.root_value);
        assert_eq!(built.root_value, strategy.value);
        (dag, built)
    }

    #[test]
    fn quantified_variable_irrelevant_to_residual_matrix_shares_child_state() {
        // x does not occur in the matrix; both x branches have the same suffix.
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![vec![Literal::positive(1)]],
        };
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (dag, report) = assert_all_three_paths_agree(&problem, &prefix);
        assert_eq!(report.memo_hits, 1);
        assert_eq!(dag.nodes.len(), 4);
        match &dag.nodes[dag.root].kind {
            QbfGameDagNodeKind::ForAllDecision {
                low_child,
                high_child,
                ..
            } => assert_eq!(low_child, high_child),
            other => panic!("expected root universal decision, got {other:?}"),
        }
        let verified = verify_qbf_game_proof_dag(&problem, &prefix, &dag, budget()).unwrap();
        assert_eq!(verified.shared_child_references, 1);
        assert_eq!(verified.shared_states, 1);
    }

    #[test]
    fn complementary_parent_literals_produce_identical_residual_suffixes() {
        // (x OR y) AND (!x OR y) is equivalent to y. Both x branches simplify
        // to the exact same residual formula and should share the same node.
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![
                vec![Literal::positive(0), Literal::positive(1)],
                vec![Literal::negative(0), Literal::positive(1)],
            ],
        };
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (dag, report) = assert_all_three_paths_agree(&problem, &prefix);
        assert!(report.memo_hits > 0);
        let root = &dag.nodes[dag.root];
        match &root.kind {
            QbfGameDagNodeKind::ForAllDecision {
                low_child,
                high_child,
                ..
            } => {
                assert_eq!(low_child, high_child);
                assert_eq!(
                    dag.nodes[*low_child].residual,
                    vec![vec![ResidualLiteral {
                        variable: 1,
                        positive: true
                    }]]
                );
            }
            ref other => panic!("expected root universal decision, got {other:?}"),
        }
    }

    #[test]
    fn canonicalization_removes_duplicates_and_tautologies_but_keeps_empty_clause() {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![
                vec![
                    Literal::positive(0),
                    Literal::negative(0),
                    Literal::positive(1),
                ],
                vec![Literal::positive(1), Literal::positive(1)],
                vec![],
            ],
        };
        let constrained_prefix = prefix(2, vec![block(Quantifier::Exists, &[0, 1])]);
        let (dag, _) = build_qbf_game_proof_dag(&problem, &constrained_prefix, budget()).unwrap();
        assert_eq!(dag.nodes[dag.root].residual.len(), 1);
        assert!(dag.nodes[dag.root].residual[0].is_empty());
        assert!(!dag.nodes[dag.root].value);
    }

    #[test]
    fn verifier_rejects_a_mutated_residual_transition() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (mut dag, _) = build_qbf_game_proof_dag(&problem, &prefix, budget()).unwrap();
        let root = dag.root;
        let child = match dag.nodes[root].kind {
            QbfGameDagNodeKind::ForAllDecision { low_child, .. } => low_child,
            _ => unreachable!(),
        };
        dag.nodes[child].residual.clear();
        assert!(matches!(
            verify_qbf_game_proof_dag(&problem, &prefix, &dag, budget()),
            Err(QbfGameDagError::NonCanonicalResidual { .. })
                | Err(QbfGameDagError::ChildResidualMismatch { .. })
        ));
    }

    #[test]
    fn verifier_rejects_wrong_quantifier_rule_and_node_value() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, budget()).unwrap();
        let mut wrong_rule = dag.clone();
        let root = wrong_rule.root;
        if let QbfGameDagNodeKind::ForAllDecision {
            variable,
            low_child,
            high_child,
        } = &wrong_rule.nodes[root].kind
        {
            let (variable, low_child, high_child) = (*variable, *low_child, *high_child);
            wrong_rule.nodes[root].kind = QbfGameDagNodeKind::ExistsDecision {
                variable,
                low_child,
                high_child,
            };
        }
        assert!(matches!(
            verify_qbf_game_proof_dag(&problem, &prefix, &wrong_rule, budget()),
            Err(QbfGameDagError::QuantifierRuleMismatch { .. })
        ));

        let mut wrong_value = dag;
        let root_value = wrong_value.nodes[wrong_value.root].value;
        wrong_value.nodes[wrong_value.root].value = !root_value;
        assert!(matches!(
            verify_qbf_game_proof_dag(&problem, &prefix, &wrong_value, budget()),
            Err(QbfGameDagError::IncorrectNodeValue { .. })
        ));
    }

    #[test]
    fn verifier_rejects_padded_or_unreachable_nodes() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, budget()).unwrap();
        let mut padded = dag;
        padded.nodes.push(QbfGameDagNode {
            prefix_position: 2,
            residual: vec![],
            value: true,
            kind: QbfGameDagNodeKind::TrueTerminal,
        });
        assert!(matches!(
            verify_qbf_game_proof_dag(&problem, &prefix, &padded, budget()),
            Err(QbfGameDagError::RootNotFinalNode { .. })
        ));
    }

    #[test]
    fn verifier_rejects_prefix_binding_mismatch() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let (mut dag, _) = build_qbf_game_proof_dag(&problem, &prefix, budget()).unwrap();
        dag.prefix_order.reverse();
        assert_eq!(
            verify_qbf_game_proof_dag(&problem, &prefix, &dag, budget()).unwrap_err(),
            QbfGameDagError::PrefixBindingMismatch
        );
    }

    #[test]
    fn generated_three_variable_dags_match_independent_game_oracle() {
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
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
                    let (dag, built) =
                        build_qbf_game_proof_dag(&problem, &prefix, budget()).unwrap();
                    let verified =
                        verify_qbf_game_proof_dag(&problem, &prefix, &dag, budget()).unwrap();
                    let expected = direct_game_oracle(&problem, &prefix);
                    assert_eq!(
                        built.root_value, expected,
                        "builder formula {formula_index}, order {order:?}, mask {quantifier_mask}"
                    );
                    assert_eq!(
                        verified.root_value, expected,
                        "verifier formula {formula_index}, order {order:?}, mask {quantifier_mask}"
                    );
                }
            }
        }
    }

    #[test]
    fn proof_node_and_builder_visit_budgets_fail_closed() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let tight_nodes = QbfGameDagBudget {
            max_proof_nodes: 1,
            ..budget()
        };
        assert!(matches!(
            build_qbf_game_proof_dag(&problem, &prefix, tight_nodes),
            Err(QbfGameDagError::ProofNodeBudgetExceeded { .. })
        ));
        let tight_visits = QbfGameDagBudget {
            max_builder_state_visits: 1,
            ..budget()
        };
        assert!(matches!(
            build_qbf_game_proof_dag(&problem, &prefix, tight_visits),
            Err(QbfGameDagError::BuilderVisitBudgetExceeded { .. })
        ));
    }

    #[test]
    fn residual_literal_storage_budget_fails_closed() {
        let problem = equality_problem();
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        let tight = QbfGameDagBudget {
            max_stored_residual_literals: 0,
            ..budget()
        };
        assert!(matches!(
            build_qbf_game_proof_dag(&problem, &prefix, tight),
            Err(QbfGameDagError::StoredResidualLiteralBudgetExceeded { .. })
        ));
    }

    fn direct_game_oracle(problem: &CnfProblem, prefix: &ValidatedQuantifierPrefix) -> bool {
        let order = ordered_prefix(prefix);
        fn visit(position: usize, order: &[(Quantifier, usize)], residual: &ResidualCnf) -> bool {
            if residual.is_empty() {
                return true;
            }
            if residual.iter().any(Vec::is_empty) {
                return false;
            }
            if position == order.len() {
                return false;
            }
            let (quantifier, variable) = order[position];
            let (low, _) = simplify_residual(residual, variable, false).unwrap();
            let (high, _) = simplify_residual(residual, variable, true).unwrap();
            let low = visit(position + 1, order, &low);
            let high = visit(position + 1, order, &high);
            match quantifier {
                Quantifier::Exists => low || high,
                Quantifier::ForAll => low && high,
            }
        }
        let (residual, _) = source_residual(problem).unwrap();
        visit(0, &order, &residual)
    }

    #[test]
    fn exhaustive_two_variable_cnf_dags_match_tree_and_strategy_oracles() {
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
                    let (dag, built) =
                        build_qbf_game_proof_dag(problem, &prefix, budget()).unwrap();
                    let verified =
                        verify_qbf_game_proof_dag(problem, &prefix, &dag, budget()).unwrap();
                    let direct = direct_game_oracle(problem, &prefix);
                    assert_eq!(built.root_value, direct, "formula {formula_index}");
                    assert_eq!(verified.root_value, direct, "formula {formula_index}");
                    assert_eq!(dag.prefix_order, ordered_prefix(&prefix));
                }
            }
        }
    }
}
