// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical reduced ordered BDD compilation for finite strategy functions.
//!
//! Truth tables remain the reference semantics. This module compiles them into
//! reduced ordered BDDs using the dependency order in the quantified prefix,
//! validates the DAG structure, and exhaustively checks semantic equivalence.
//! It is a bounded circuit-compilation experiment, not a scalable QBF solver.

use super::quantified_strategy::BooleanStrategyFunction;
use std::collections::BTreeMap;

/// A terminal or indexed decision node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BddRef {
    False,
    True,
    Node(usize),
}

/// A decision node whose index addresses the ordered dependency list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BddNode {
    pub dependency_index: usize,
    pub low: BddRef,
    pub high: BddRef,
}

/// A reduced ordered BDD bound to an explicit dependency-variable order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReducedOrderedBdd {
    pub dependencies: Vec<usize>,
    pub nodes: Vec<BddNode>,
    pub root: BddRef,
}

/// Limits for compiling and checking one strategy function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyBddBudget {
    pub max_dependency_variables: usize,
    pub max_table_entries: u128,
    pub max_bdd_nodes: usize,
}

impl Default for StrategyBddBudget {
    fn default() -> Self {
        Self {
            max_dependency_variables: 20,
            max_table_entries: 1 << 20,
            max_bdd_nodes: 1 << 20,
        }
    }
}

/// Structural, budget and semantic failures for strategy BDDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrategyBddError {
    DependencyBudgetExceeded { required: usize, limit: usize },
    TableEntryBudgetExceeded { required: u128, limit: u128 },
    NodeBudgetExceeded { required: usize, limit: usize },
    AssignmentSpaceOverflow { dependency_count: usize },
    FunctionTableEntryCountMismatch { expected: u128, actual: usize },
    FunctionTableKeyWidthMismatch { expected: usize, actual: usize },
    MissingTruthTableRow { key: Vec<bool> },
    DependencyScopeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    RootNodeOutOfRange { node: usize, node_count: usize },
    ChildNodeOutOfRange { parent: usize, child: usize, node_count: usize },
    NonTopologicalEdge { parent: usize, child: usize },
    DependencyOrderViolation {
        parent: usize,
        child: usize,
        parent_dependency: usize,
        child_dependency: usize,
    },
    UnreducedNode { node: usize },
    DuplicateNode { node: usize, earlier: usize },
    EvaluationAssignmentWidthMismatch { expected: usize, actual: usize },
    EvaluationStepBudgetExceeded { required: usize, limit: usize },
    SemanticMismatch { assignment: Vec<bool>, table_value: bool, bdd_value: bool },
    ResourceAccountingOverflow,
}

/// Compile/check measurements for one function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyBddReport {
    pub dependency_count: usize,
    pub truth_table_entries: u128,
    pub bdd_nodes: usize,
    pub assignments_checked: u128,
}

fn checked_power_of_two(exponent: usize) -> Result<u128, StrategyBddError> {
    let shift = u32::try_from(exponent)
        .map_err(|_| StrategyBddError::AssignmentSpaceOverflow {
            dependency_count: exponent,
        })?;
    1u128
        .checked_shl(shift)
        .ok_or(StrategyBddError::AssignmentSpaceOverflow {
            dependency_count: exponent,
        })
}

fn validate_truth_table(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<u128, StrategyBddError> {
    if function.dependencies.len() > budget.max_dependency_variables {
        return Err(StrategyBddError::DependencyBudgetExceeded {
            required: function.dependencies.len(),
            limit: budget.max_dependency_variables,
        });
    }
    let expected = checked_power_of_two(function.dependencies.len())?;
    if expected > budget.max_table_entries {
        return Err(StrategyBddError::TableEntryBudgetExceeded {
            required: expected,
            limit: budget.max_table_entries,
        });
    }
    if function.table.len() as u128 != expected {
        return Err(StrategyBddError::FunctionTableEntryCountMismatch {
            expected,
            actual: function.table.len(),
        });
    }
    for key in function.table.keys() {
        if key.len() != function.dependencies.len() {
            return Err(StrategyBddError::FunctionTableKeyWidthMismatch {
                expected: function.dependencies.len(),
                actual: key.len(),
            });
        }
    }
    Ok(expected)
}

fn build_level(
    level: usize,
    dependencies: &[usize],
    table: &BTreeMap<Vec<bool>, bool>,
    prefix: &mut Vec<bool>,
    nodes: &mut Vec<BddNode>,
    unique: &mut BTreeMap<(usize, BddRef, BddRef), usize>,
    budget: StrategyBddBudget,
) -> Result<BddRef, StrategyBddError> {
    if level == dependencies.len() {
        let Some(&value) = table.get(prefix) else {
            return Err(StrategyBddError::MissingTruthTableRow {
                key: prefix.clone(),
            });
        };
        return Ok(if value { BddRef::True } else { BddRef::False });
    }

    prefix.push(false);
    let low = build_level(
        level + 1, dependencies, table, prefix, nodes, unique, budget,
    )?;
    prefix.pop();

    prefix.push(true);
    let high = build_level(
        level + 1, dependencies, table, prefix, nodes, unique, budget,
    )?;
    prefix.pop();

    // Eliminate a test whose two branches are identical.
    if low == high {
        return Ok(low);
    }

    // Merge structurally identical nodes.
    let key = (level, low, high);
    if let Some(&existing) = unique.get(&key) {
        return Ok(BddRef::Node(existing));
    }
    if nodes.len() >= budget.max_bdd_nodes {
        return Err(StrategyBddError::NodeBudgetExceeded {
            required: nodes.len().saturating_add(1),
            limit: budget.max_bdd_nodes,
        });
    }

    let index = nodes.len();
    nodes.push(BddNode {
        dependency_index: level,
        low,
        high,
    });
    unique.insert(key, index);
    Ok(BddRef::Node(index))
}

/// Compile a complete strategy truth table into an ROBDD for its dependency
/// order. Node IDs are assigned deterministically in post-order.
pub fn compile_strategy_function_to_robdd(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<ReducedOrderedBdd, StrategyBddError> {
    validate_truth_table(function, budget)?;
    let mut nodes = Vec::new();
    let mut unique = BTreeMap::new();
    let mut prefix = Vec::with_capacity(function.dependencies.len());
    let root = build_level(
        0,
        &function.dependencies,
        &function.table,
        &mut prefix,
        &mut nodes,
        &mut unique,
        budget,
    )?;
    let bdd = ReducedOrderedBdd {
        dependencies: function.dependencies.clone(),
        nodes,
        root,
    };
    validate_robdd(&bdd)?;
    Ok(bdd)
}

/// Check topological node numbering, increasing variable order, reduction,
/// uniqueness, and root/edge validity without consulting the source function.
pub fn validate_robdd(bdd: &ReducedOrderedBdd) -> Result<(), StrategyBddError> {
    if let BddRef::Node(root) = bdd.root {
        if root >= bdd.nodes.len() {
            return Err(StrategyBddError::RootNodeOutOfRange {
                node: root,
                node_count: bdd.nodes.len(),
            });
        }
    }

    let mut unique = BTreeMap::new();
    for (parent, node) in bdd.nodes.iter().enumerate() {
        if node.dependency_index >= bdd.dependencies.len() {
            return Err(StrategyBddError::DependencyBudgetExceeded {
                required: node.dependency_index.saturating_add(1),
                limit: bdd.dependencies.len(),
            });
        }
        if node.low == node.high {
            return Err(StrategyBddError::UnreducedNode { node: parent });
        }

        for child_ref in [node.low, node.high] {
            if let BddRef::Node(child) = child_ref {
                if child >= bdd.nodes.len() {
                    return Err(StrategyBddError::ChildNodeOutOfRange {
                        parent,
                        child,
                        node_count: bdd.nodes.len(),
                    });
                }
                if child >= parent {
                    return Err(StrategyBddError::NonTopologicalEdge { parent, child });
                }
                let child_dependency = bdd.nodes[child].dependency_index;
                if child_dependency <= node.dependency_index {
                    return Err(StrategyBddError::DependencyOrderViolation {
                        parent,
                        child,
                        parent_dependency: node.dependency_index,
                        child_dependency,
                    });
                }
            }
        }

        let key = (node.dependency_index, node.low, node.high);
        if let Some(&earlier) = unique.get(&key) {
            return Err(StrategyBddError::DuplicateNode {
                node: parent,
                earlier,
            });
        }
        unique.insert(key, parent);
    }
    Ok(())
}

/// Evaluate an ROBDD for one assignment in dependency-list order.
pub fn evaluate_robdd(
    bdd: &ReducedOrderedBdd,
    assignment: &[bool],
) -> Result<bool, StrategyBddError> {
    validate_robdd(bdd)?;
    if assignment.len() != bdd.dependencies.len() {
        return Err(StrategyBddError::EvaluationAssignmentWidthMismatch {
            expected: bdd.dependencies.len(),
            actual: assignment.len(),
        });
    }

    let mut current = bdd.root;
    let mut steps = 0usize;
    loop {
        match current {
            BddRef::False => return Ok(false),
            BddRef::True => return Ok(true),
            BddRef::Node(index) => {
                steps = steps
                    .checked_add(1)
                    .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                if steps > bdd.nodes.len() {
                    return Err(StrategyBddError::EvaluationStepBudgetExceeded {
                        required: steps,
                        limit: bdd.nodes.len(),
                    });
                }
                let node = &bdd.nodes[index];
                current = if assignment[node.dependency_index] {
                    node.high
                } else {
                    node.low
                };
            }
        }
    }
}

/// Independently compare the BDD with every truth-table assignment.
pub fn verify_robdd_matches_strategy_function(
    bdd: &ReducedOrderedBdd,
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<StrategyBddReport, StrategyBddError> {
    validate_robdd(bdd)?;
    if bdd.dependencies != function.dependencies {
        return Err(StrategyBddError::DependencyScopeMismatch {
            expected: function.dependencies.clone(),
            actual: bdd.dependencies.clone(),
        });
    }
    let rows = validate_truth_table(function, budget)?;

    for mask in 0..rows {
        let assignment: Vec<bool> = (0..function.dependencies.len())
            .map(|index| ((mask >> index) & 1) == 1)
            .collect();
        let Some(&table_value) = function.table.get(&assignment) else {
            return Err(StrategyBddError::MissingTruthTableRow {
                key: assignment,
            });
        };
        let bdd_value = evaluate_robdd(bdd, &assignment)?;
        if table_value != bdd_value {
            return Err(StrategyBddError::SemanticMismatch {
                assignment,
                table_value,
                bdd_value,
            });
        }
    }

    Ok(StrategyBddReport {
        dependency_count: function.dependencies.len(),
        truth_table_entries: rows,
        bdd_nodes: bdd.nodes.len(),
        assignments_checked: rows,
    })
}

#[cfg(test)]
mod tests {
    use super::super::quantified_strategy::BooleanStrategyFunction;
    use super::*;

    fn budget() -> StrategyBddBudget {
        StrategyBddBudget {
            max_dependency_variables: 8,
            max_table_entries: 1 << 8,
            max_bdd_nodes: 1 << 8,
        }
    }

    fn function(dependencies: &[usize], values: &[bool]) -> BooleanStrategyFunction {
        let mut table = BTreeMap::new();
        for mask in 0..(1usize << dependencies.len()) {
            let key: Vec<bool> = (0..dependencies.len())
                .map(|index| ((mask >> index) & 1) != 0)
                .collect();
            table.insert(key, values[mask]);
        }
        BooleanStrategyFunction {
            variable: 99,
            dependencies: dependencies.to_vec(),
            table,
        }
    }

    #[test]
    fn constant_function_reduces_to_terminal() {
        let source = function(&[4, 2], &[false, false, false, false]);
        let bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        assert_eq!(bdd.root, BddRef::False);
        assert!(bdd.nodes.is_empty());
        let report = verify_robdd_matches_strategy_function(&bdd, &source, budget()).unwrap();
        assert_eq!(report.assignments_checked, 4);
        assert_eq!(report.bdd_nodes, 0);
    }

    #[test]
    fn identity_function_uses_one_node() {
        let source = function(&[7, 2], &[false, false, true, true]);
        let bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        assert_eq!(bdd.nodes.len(), 1);
        assert!(!evaluate_robdd(&bdd, &[false, false]).unwrap());
        assert!(evaluate_robdd(&bdd, &[false, true]).unwrap());
        verify_robdd_matches_strategy_function(&bdd, &source, budget()).unwrap();
    }

    #[test]
    fn xor_function_has_three_nodes_in_fixed_variable_order() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        assert_eq!(bdd.nodes.len(), 3);
        verify_robdd_matches_strategy_function(&bdd, &source, budget()).unwrap();
    }

    #[test]
    fn bdd_rejects_missing_truth_table_entries() {
        let mut source = function(&[0, 1], &[false, true, true, false]);
        source.table.remove(&vec![true, true]);
        assert_eq!(
            compile_strategy_function_to_robdd(&source, budget()).unwrap_err(),
            StrategyBddError::FunctionTableEntryCountMismatch {
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn bdd_node_budget_is_fail_closed() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let tight = StrategyBddBudget {
            max_bdd_nodes: 2,
            ..budget()
        };
        assert_eq!(
            compile_strategy_function_to_robdd(&source, tight).unwrap_err(),
            StrategyBddError::NodeBudgetExceeded {
                required: 3,
                limit: 2
            }
        );
    }

    #[test]
    fn verifier_rejects_dependency_scope_mismatch() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let mut bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        bdd.dependencies = vec![1, 0];
        assert!(matches!(
            verify_robdd_matches_strategy_function(&bdd, &source, budget()),
            Err(StrategyBddError::DependencyScopeMismatch { .. })
        ));
    }

    #[test]
    fn verifier_rejects_unreduced_nodes() {
        let source = function(&[0], &[false, true]);
        let mut bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        bdd.nodes[0].high = bdd.nodes[0].low;
        assert!(matches!(
            validate_robdd(&bdd),
            Err(StrategyBddError::UnreducedNode { node: 0 })
        ));
    }

    #[test]
    fn verifier_rejects_dependency_order_violation() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let mut bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        let root = match bdd.root {
            BddRef::Node(index) => index,
            _ => unreachable!(),
        };
        bdd.nodes[root].dependency_index = 1;
        assert!(matches!(
            validate_robdd(&bdd),
            Err(StrategyBddError::DependencyOrderViolation { .. })
        ));
    }
}
