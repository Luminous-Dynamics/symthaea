// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded AND-Inverter Graph compilation for Boolean strategy functions.
//!
//! The source truth table remains the reference semantics. This module
//! compiles it into a structurally hashed AIG using Shannon decomposition,
//! complemented edges and two-input AND nodes. A separate validator checks
//! graph structure and exhaustively compares the AIG with every truth-table
//! row. This is a second representation experiment beside ROBDDs, not a QBF
//! solver or proof system.

use super::quantified_strategy::BooleanStrategyFunction;
use std::collections::{BTreeMap, BTreeSet};

/// A constant or complemented reference to an AIG node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AigLiteral {
    Constant(bool),
    Node { index: usize, inverted: bool },
}

impl AigLiteral {
    pub const FALSE: Self = Self::Constant(false);
    pub const TRUE: Self = Self::Constant(true);

    pub const fn node(index: usize) -> Self {
        Self::Node {
            index,
            inverted: false,
        }
    }

    pub const fn not(self) -> Self {
        match self {
            Self::Constant(value) => Self::Constant(!value),
            Self::Node { index, inverted } => Self::Node {
                index,
                inverted: !inverted,
            },
        }
    }

    pub const fn is_complement_of(self, other: Self) -> bool {
        match (self, other) {
            (Self::Constant(left), Self::Constant(right)) => left != right,
            (
                Self::Node {
                    index: left_index,
                    inverted: left_inverted,
                },
                Self::Node {
                    index: right_index,
                    inverted: right_inverted,
                },
            ) => left_index == right_index && left_inverted != right_inverted,
            _ => false,
        }
    }
}

/// An input node or a two-input AND node. Inversion is carried by edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AigNode {
    Input { dependency_index: usize },
    And { left: AigLiteral, right: AigLiteral },
}

/// An AIG bound to an explicit semantic dependency list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyAig {
    pub dependencies: Vec<usize>,
    /// Shannon expansion order; this does not alter semantic dependencies.
    pub decomposition_order: Vec<usize>,
    pub nodes: Vec<AigNode>,
    pub output: AigLiteral,
}

/// Explicit limits for truth-table-to-AIG compilation and checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyAigBudget {
    pub max_dependency_variables: usize,
    pub max_truth_table_entries: u128,
    pub max_and_nodes: usize,
    pub max_assignments_checked: u128,
}

impl Default for StrategyAigBudget {
    fn default() -> Self {
        Self {
            max_dependency_variables: 20,
            max_truth_table_entries: 1 << 20,
            max_and_nodes: 1 << 20,
            max_assignments_checked: 1 << 20,
        }
    }
}

/// Structural, semantic and resource failures for strategy AIGs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrategyAigError {
    DependencyBudgetExceeded {
        required: usize,
        limit: usize,
    },
    TruthTableBudgetExceeded {
        required: u128,
        limit: u128,
    },
    AndNodeBudgetExceeded {
        required: usize,
        limit: usize,
    },
    AssignmentBudgetExceeded {
        required: u128,
        limit: u128,
    },
    AssignmentSpaceOverflow {
        dependency_count: usize,
    },
    TruthTableEntryCountMismatch {
        expected: u128,
        actual: usize,
    },
    TruthTableKeyWidthMismatch {
        expected: usize,
        actual: usize,
    },
    MissingTruthTableRow {
        key: Vec<bool>,
    },
    DependencyScopeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    DependencyOrderLengthMismatch {
        expected: usize,
        actual: usize,
    },
    DependencyOrderIndexOutOfRange {
        position: usize,
        dependency_index: usize,
        dependency_count: usize,
    },
    DuplicateDependencyOrderIndex {
        dependency_index: usize,
    },
    DuplicateSemanticDependency {
        dependency: usize,
    },
    InputNodeCountMismatch {
        expected: usize,
        actual: usize,
    },
    InputNodePositionMismatch {
        node_index: usize,
        expected_dependency: usize,
        actual_dependency: usize,
    },
    AndNodeBeforeAllInputs {
        node_index: usize,
        input_count: usize,
    },
    NonCanonicalAndNode {
        node_index: usize,
    },
    ConstantAndInputNotReduced {
        node_index: usize,
    },
    DuplicateAndNode {
        node_index: usize,
        earlier_node: usize,
    },
    ChildNodeOutOfRange {
        parent: usize,
        child: usize,
        node_count: usize,
    },
    NonTopologicalEdge {
        parent: usize,
        child: usize,
    },
    OutputNodeOutOfRange {
        node: usize,
        node_count: usize,
    },
    UnreachableAndNode {
        node_index: usize,
    },
    AssignmentWidthMismatch {
        expected: usize,
        actual: usize,
    },
    SemanticMismatch {
        assignment: Vec<bool>,
        truth_table_value: bool,
        aig_value: bool,
    },
    ResourceAccountingOverflow,
}

/// Metrics for one AIG compile and exhaustive check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyAigReport {
    pub dependency_count: usize,
    pub truth_table_entries: u128,
    pub input_nodes: usize,
    pub and_nodes: usize,
    pub complemented_edges: usize,
    pub assignments_checked: u128,
}

fn canonical_pair(left: AigLiteral, right: AigLiteral) -> (AigLiteral, AigLiteral) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn checked_power_of_two(exponent: usize) -> Result<u128, StrategyAigError> {
    let shift = u32::try_from(exponent).map_err(|_| StrategyAigError::AssignmentSpaceOverflow {
        dependency_count: exponent,
    })?;
    1u128
        .checked_shl(shift)
        .ok_or(StrategyAigError::AssignmentSpaceOverflow {
            dependency_count: exponent,
        })
}

fn validate_function(
    function: &BooleanStrategyFunction,
    budget: StrategyAigBudget,
) -> Result<u128, StrategyAigError> {
    let dependency_count = function.dependencies.len();
    if dependency_count > budget.max_dependency_variables {
        return Err(StrategyAigError::DependencyBudgetExceeded {
            required: dependency_count,
            limit: budget.max_dependency_variables,
        });
    }
    let entries = checked_power_of_two(dependency_count)?;
    if entries > budget.max_truth_table_entries {
        return Err(StrategyAigError::TruthTableBudgetExceeded {
            required: entries,
            limit: budget.max_truth_table_entries,
        });
    }
    if function.table.len() as u128 != entries {
        return Err(StrategyAigError::TruthTableEntryCountMismatch {
            expected: entries,
            actual: function.table.len(),
        });
    }
    for key in function.table.keys() {
        if key.len() != dependency_count {
            return Err(StrategyAigError::TruthTableKeyWidthMismatch {
                expected: dependency_count,
                actual: key.len(),
            });
        }
    }
    let mut seen = BTreeSet::new();
    for &dependency in &function.dependencies {
        if !seen.insert(dependency) {
            return Err(StrategyAigError::DuplicateSemanticDependency { dependency });
        }
    }
    Ok(entries)
}

fn validate_decomposition_order(
    dependency_count: usize,
    order: &[usize],
) -> Result<(), StrategyAigError> {
    if order.len() != dependency_count {
        return Err(StrategyAigError::DependencyOrderLengthMismatch {
            expected: dependency_count,
            actual: order.len(),
        });
    }
    let mut seen = vec![false; dependency_count];
    for (position, &dependency_index) in order.iter().enumerate() {
        if dependency_index >= dependency_count {
            return Err(StrategyAigError::DependencyOrderIndexOutOfRange {
                position,
                dependency_index,
                dependency_count,
            });
        }
        if seen[dependency_index] {
            return Err(StrategyAigError::DuplicateDependencyOrderIndex { dependency_index });
        }
        seen[dependency_index] = true;
    }
    Ok(())
}

struct AigBuilder {
    nodes: Vec<AigNode>,
    unique_and: BTreeMap<(AigLiteral, AigLiteral), usize>,
    input_count: usize,
    max_and_nodes: usize,
}

impl AigBuilder {
    fn new(dependencies: usize, max_and_nodes: usize) -> Self {
        let nodes = (0..dependencies)
            .map(|dependency_index| AigNode::Input { dependency_index })
            .collect();
        Self {
            nodes,
            unique_and: BTreeMap::new(),
            input_count: dependencies,
            max_and_nodes,
        }
    }

    fn input(&self, dependency_index: usize) -> AigLiteral {
        AigLiteral::node(dependency_index)
    }

    fn and(&mut self, left: AigLiteral, right: AigLiteral) -> Result<AigLiteral, StrategyAigError> {
        match (left, right) {
            (AigLiteral::Constant(false), _) | (_, AigLiteral::Constant(false)) => {
                return Ok(AigLiteral::FALSE);
            }
            (AigLiteral::Constant(true), other) | (other, AigLiteral::Constant(true)) => {
                return Ok(other);
            }
            _ => {}
        }
        if left == right {
            return Ok(left);
        }
        if left.is_complement_of(right) {
            return Ok(AigLiteral::FALSE);
        }

        let key = canonical_pair(left, right);
        if let Some(&index) = self.unique_and.get(&key) {
            return Ok(AigLiteral::node(index));
        }
        let and_nodes = self.nodes.len().saturating_sub(self.input_count);
        if and_nodes >= self.max_and_nodes {
            return Err(StrategyAigError::AndNodeBudgetExceeded {
                required: and_nodes.saturating_add(1),
                limit: self.max_and_nodes,
            });
        }

        let index = self.nodes.len();
        self.nodes.push(AigNode::And {
            left: key.0,
            right: key.1,
        });
        self.unique_and.insert(key, index);
        Ok(AigLiteral::node(index))
    }

    fn or(&mut self, left: AigLiteral, right: AigLiteral) -> Result<AigLiteral, StrategyAigError> {
        Ok(self.and(left.not(), right.not())?.not())
    }
}

fn build_shannon(
    level: usize,
    order: &[usize],
    function: &BooleanStrategyFunction,
    assignment: &mut [bool],
    builder: &mut AigBuilder,
) -> Result<AigLiteral, StrategyAigError> {
    if level == order.len() {
        let Some(&value) = function.table.get(assignment) else {
            return Err(StrategyAigError::MissingTruthTableRow {
                key: assignment.to_vec(),
            });
        };
        return Ok(AigLiteral::Constant(value));
    }

    let dependency_index = order[level];
    let input = builder.input(dependency_index);

    assignment[dependency_index] = false;
    let low = build_shannon(level + 1, order, function, assignment, builder)?;

    assignment[dependency_index] = true;
    let high = build_shannon(level + 1, order, function, assignment, builder)?;

    // Eliminate a Shannon test when both cofactors are identical.
    if low == high {
        return Ok(low);
    }

    let high_branch = builder.and(input, high)?;
    let low_branch = builder.and(input.not(), low)?;
    builder.or(high_branch, low_branch)
}

/// Compile a complete truth table into an AIG using natural dependency order.
pub fn compile_strategy_function_to_aig(
    function: &BooleanStrategyFunction,
    budget: StrategyAigBudget,
) -> Result<StrategyAig, StrategyAigError> {
    let order: Vec<usize> = (0..function.dependencies.len()).collect();
    compile_strategy_function_to_aig_with_order(function, &order, budget)
}

/// Compile with a caller-selected Shannon expansion order. The semantic
/// dependency list remains unchanged.
pub fn compile_strategy_function_to_aig_with_order(
    function: &BooleanStrategyFunction,
    decomposition_order: &[usize],
    budget: StrategyAigBudget,
) -> Result<StrategyAig, StrategyAigError> {
    validate_function(function, budget)?;
    validate_decomposition_order(function.dependencies.len(), decomposition_order)?;

    let mut builder = AigBuilder::new(function.dependencies.len(), budget.max_and_nodes);
    let mut assignment = vec![false; function.dependencies.len()];
    let output = build_shannon(
        0,
        decomposition_order,
        function,
        &mut assignment,
        &mut builder,
    )?;
    let aig = StrategyAig {
        dependencies: function.dependencies.clone(),
        decomposition_order: decomposition_order.to_vec(),
        nodes: builder.nodes,
        output,
    };
    validate_strategy_aig(&aig, budget)?;
    Ok(aig)
}

/// Validate the complete AIG structure without consulting the source table.
pub fn validate_strategy_aig(
    aig: &StrategyAig,
    budget: StrategyAigBudget,
) -> Result<(), StrategyAigError> {
    let dependency_count = aig.dependencies.len();
    if dependency_count > budget.max_dependency_variables {
        return Err(StrategyAigError::DependencyBudgetExceeded {
            required: dependency_count,
            limit: budget.max_dependency_variables,
        });
    }
    validate_decomposition_order(dependency_count, &aig.decomposition_order)?;

    let mut dependencies_seen = BTreeSet::new();
    for &dependency in &aig.dependencies {
        if !dependencies_seen.insert(dependency) {
            return Err(StrategyAigError::DuplicateSemanticDependency { dependency });
        }
    }

    let input_count = aig
        .nodes
        .iter()
        .take_while(|node| matches!(node, AigNode::Input { .. }))
        .count();
    if input_count != dependency_count {
        return Err(StrategyAigError::InputNodeCountMismatch {
            expected: dependency_count,
            actual: input_count,
        });
    }
    if let AigLiteral::Node { index, .. } = aig.output {
        if index >= aig.nodes.len() {
            return Err(StrategyAigError::OutputNodeOutOfRange {
                node: index,
                node_count: aig.nodes.len(),
            });
        }
    }

    let mut unique = BTreeMap::new();
    let mut and_count = 0usize;
    for (node_index, node) in aig.nodes.iter().enumerate() {
        match *node {
            AigNode::Input { dependency_index } => {
                if node_index >= dependency_count {
                    return Err(StrategyAigError::AndNodeBeforeAllInputs {
                        node_index,
                        input_count: dependency_count,
                    });
                }
                if dependency_index != node_index {
                    return Err(StrategyAigError::InputNodePositionMismatch {
                        node_index,
                        expected_dependency: node_index,
                        actual_dependency: dependency_index,
                    });
                }
            }
            AigNode::And { left, right } => {
                if node_index < dependency_count {
                    return Err(StrategyAigError::AndNodeBeforeAllInputs {
                        node_index,
                        input_count: dependency_count,
                    });
                }
                and_count = and_count
                    .checked_add(1)
                    .ok_or(StrategyAigError::ResourceAccountingOverflow)?;
                if and_count > budget.max_and_nodes {
                    return Err(StrategyAigError::AndNodeBudgetExceeded {
                        required: and_count,
                        limit: budget.max_and_nodes,
                    });
                }
                if left > right {
                    return Err(StrategyAigError::NonCanonicalAndNode { node_index });
                }
                if matches!(left, AigLiteral::Constant(_))
                    || matches!(right, AigLiteral::Constant(_))
                    || left == right
                    || left.is_complement_of(right)
                {
                    return Err(StrategyAigError::ConstantAndInputNotReduced { node_index });
                }
                for child in [left, right] {
                    if let AigLiteral::Node { index, .. } = child {
                        if index >= aig.nodes.len() {
                            return Err(StrategyAigError::ChildNodeOutOfRange {
                                parent: node_index,
                                child: index,
                                node_count: aig.nodes.len(),
                            });
                        }
                        if index >= node_index {
                            return Err(StrategyAigError::NonTopologicalEdge {
                                parent: node_index,
                                child: index,
                            });
                        }
                    }
                }
                let key = (left, right);
                if let Some(&earlier_node) = unique.get(&key) {
                    return Err(StrategyAigError::DuplicateAndNode {
                        node_index,
                        earlier_node,
                    });
                }
                unique.insert(key, node_index);
            }
        }
    }

    let mut reachable = vec![false; aig.nodes.len()];
    let mut pending = Vec::new();
    if let AigLiteral::Node { index, .. } = aig.output {
        pending.push(index);
    }
    while let Some(index) = pending.pop() {
        if reachable[index] {
            continue;
        }
        reachable[index] = true;
        if let AigNode::And { left, right } = aig.nodes[index] {
            for child in [left, right] {
                if let AigLiteral::Node {
                    index: child_index, ..
                } = child
                {
                    pending.push(child_index);
                }
            }
        }
    }
    for (node_index, node) in aig.nodes.iter().enumerate() {
        if matches!(node, AigNode::And { .. }) && !reachable[node_index] {
            return Err(StrategyAigError::UnreachableAndNode { node_index });
        }
    }
    Ok(())
}

fn evaluate_literal(values: &[bool], literal: AigLiteral) -> Result<bool, StrategyAigError> {
    match literal {
        AigLiteral::Constant(value) => Ok(value),
        AigLiteral::Node { index, inverted } => {
            let value = *values
                .get(index)
                .ok_or(StrategyAigError::OutputNodeOutOfRange {
                    node: index,
                    node_count: values.len(),
                })?;
            Ok(if inverted { !value } else { value })
        }
    }
}

fn evaluate_validated_aig(
    aig: &StrategyAig,
    assignment: &[bool],
) -> Result<bool, StrategyAigError> {
    let mut values = Vec::with_capacity(aig.nodes.len());
    for node in &aig.nodes {
        let value = match *node {
            AigNode::Input { dependency_index } => assignment[dependency_index],
            AigNode::And { left, right } => {
                evaluate_literal(&values, left)? && evaluate_literal(&values, right)?
            }
        };
        values.push(value);
    }
    evaluate_literal(&values, aig.output)
}

/// Evaluate the AIG under one ordered dependency assignment.
pub fn evaluate_strategy_aig(
    aig: &StrategyAig,
    assignment: &[bool],
    budget: StrategyAigBudget,
) -> Result<bool, StrategyAigError> {
    validate_strategy_aig(aig, budget)?;
    if assignment.len() != aig.dependencies.len() {
        return Err(StrategyAigError::AssignmentWidthMismatch {
            expected: aig.dependencies.len(),
            actual: assignment.len(),
        });
    }
    evaluate_validated_aig(aig, assignment)
}

/// Independently check AIG output against every source truth-table row.
pub fn verify_aig_matches_strategy_function(
    aig: &StrategyAig,
    function: &BooleanStrategyFunction,
    budget: StrategyAigBudget,
) -> Result<StrategyAigReport, StrategyAigError> {
    validate_strategy_aig(aig, budget)?;
    if aig.dependencies != function.dependencies {
        return Err(StrategyAigError::DependencyScopeMismatch {
            expected: function.dependencies.clone(),
            actual: aig.dependencies.clone(),
        });
    }
    let entries = validate_function(function, budget)?;
    if entries > budget.max_assignments_checked {
        return Err(StrategyAigError::AssignmentBudgetExceeded {
            required: entries,
            limit: budget.max_assignments_checked,
        });
    }

    for mask in 0..entries {
        let assignment: Vec<bool> = (0..function.dependencies.len())
            .map(|index| ((mask >> index) & 1) != 0)
            .collect();
        let Some(&truth_table_value) = function.table.get(&assignment) else {
            return Err(StrategyAigError::MissingTruthTableRow { key: assignment });
        };
        let aig_value = evaluate_validated_aig(aig, &assignment)?;
        if truth_table_value != aig_value {
            return Err(StrategyAigError::SemanticMismatch {
                assignment,
                truth_table_value,
                aig_value,
            });
        }
    }

    let and_nodes = aig
        .nodes
        .iter()
        .filter(|node| matches!(node, AigNode::And { .. }))
        .count();
    let complemented_edges = aig
        .nodes
        .iter()
        .map(|node| match node {
            AigNode::Input { .. } => 0,
            AigNode::And { left, right } => {
                usize::from(matches!(left, AigLiteral::Node { inverted: true, .. }))
                    + usize::from(matches!(right, AigLiteral::Node { inverted: true, .. }))
            }
        })
        .sum::<usize>()
        + usize::from(matches!(
            aig.output,
            AigLiteral::Node { inverted: true, .. }
        ));

    Ok(StrategyAigReport {
        dependency_count: function.dependencies.len(),
        truth_table_entries: entries,
        input_nodes: function.dependencies.len(),
        and_nodes,
        complemented_edges,
        assignments_checked: entries,
    })
}

#[cfg(test)]
mod tests {
    use super::super::quantified_strategy::BooleanStrategyFunction;
    use super::*;
    use std::collections::BTreeMap;

    fn budget() -> StrategyAigBudget {
        StrategyAigBudget {
            max_dependency_variables: 8,
            max_truth_table_entries: 1 << 8,
            max_and_nodes: 1 << 8,
            max_assignments_checked: 1 << 8,
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
    fn constant_functions_compile_to_constant_outputs() {
        for value in [false, true] {
            let source = function(&[0, 1, 2], &[value; 8]);
            let aig = compile_strategy_function_to_aig(&source, budget()).unwrap();
            assert_eq!(aig.output, AigLiteral::Constant(value));
            assert_eq!(aig.nodes.len(), 3);
            let report = verify_aig_matches_strategy_function(&aig, &source, budget()).unwrap();
            assert_eq!(report.and_nodes, 0);
            assert_eq!(report.assignments_checked, 8);
        }
    }

    #[test]
    fn identity_function_uses_a_direct_input_edge() {
        let source = function(&[7, 2], &[false, false, true, true]);
        let aig = compile_strategy_function_to_aig(&source, budget()).unwrap();
        assert_eq!(aig.output, AigLiteral::node(1));
        let report = verify_aig_matches_strategy_function(&aig, &source, budget()).unwrap();
        assert_eq!(report.and_nodes, 0);
        assert_eq!(report.input_nodes, 2);
    }

    #[test]
    fn every_three_input_boolean_function_matches_its_source_truth_table() {
        for function_mask in 0..256usize {
            let values: Vec<bool> = (0..8)
                .map(|assignment| ((function_mask >> assignment) & 1) != 0)
                .collect();
            let source = function(&[0, 1, 2], &values);
            let aig = compile_strategy_function_to_aig(&source, budget()).unwrap();
            let report = verify_aig_matches_strategy_function(&aig, &source, budget()).unwrap();
            assert_eq!(
                report.assignments_checked, 8,
                "function truth-table mask {function_mask}"
            );
        }
    }

    #[test]
    fn verifier_rejects_mismatched_semantic_scope() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let mut aig = compile_strategy_function_to_aig(&source, budget()).unwrap();
        aig.dependencies = vec![1, 0];
        assert!(matches!(
            verify_aig_matches_strategy_function(&aig, &source, budget()),
            Err(StrategyAigError::DependencyScopeMismatch { .. })
        ));
    }

    #[test]
    fn xor_function_uses_complemented_edges_and_checks_all_rows() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let aig = compile_strategy_function_to_aig(&source, budget()).unwrap();
        let report = verify_aig_matches_strategy_function(&aig, &source, budget()).unwrap();
        assert!(report.and_nodes > 0);
        assert!(report.complemented_edges > 0);
        assert_eq!(report.assignments_checked, 4);
    }

    #[test]
    fn selected_decomposition_order_preserves_semantics() {
        let source = function(
            &[10, 11, 12],
            &[false, true, false, true, false, false, true, true],
        );
        let aig =
            compile_strategy_function_to_aig_with_order(&source, &[2, 0, 1], budget()).unwrap();
        assert_eq!(aig.decomposition_order, vec![2, 0, 1]);
        verify_aig_matches_strategy_function(&aig, &source, budget()).unwrap();
    }

    #[test]
    fn node_budget_is_fail_closed() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let tight = StrategyAigBudget {
            max_and_nodes: 0,
            ..budget()
        };
        assert!(matches!(
            compile_strategy_function_to_aig(&source, tight),
            Err(StrategyAigError::AndNodeBudgetExceeded { .. })
        ));
    }

    #[test]
    fn assignment_budget_is_fail_closed() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let tight = StrategyAigBudget {
            max_assignments_checked: 3,
            ..budget()
        };
        let aig = compile_strategy_function_to_aig(&source, budget()).unwrap();
        assert!(matches!(
            verify_aig_matches_strategy_function(&aig, &source, tight),
            Err(StrategyAigError::AssignmentBudgetExceeded {
                required: 4,
                limit: 3
            })
        ));
    }

    #[test]
    fn validator_rejects_noncanonical_and_pair() {
        let source = function(&[0], &[false, true]);
        let mut aig = compile_strategy_function_to_aig(&source, budget()).unwrap();
        let index = aig.nodes.len();
        aig.nodes.push(AigNode::And {
            left: AigLiteral::node(0),
            right: AigLiteral::node(0),
        });
        aig.output = AigLiteral::node(index);
        assert!(matches!(
            validate_strategy_aig(&aig, budget()),
            Err(StrategyAigError::NonCanonicalAndNode { .. })
                | Err(StrategyAigError::ConstantAndInputNotReduced { .. })
        ));
    }

    #[test]
    fn validator_rejects_non_topological_children() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let mut aig = compile_strategy_function_to_aig(&source, budget()).unwrap();
        let and_index = aig
            .nodes
            .iter()
            .position(|node| matches!(node, AigNode::And { .. }))
            .unwrap();
        aig.nodes[and_index] = AigNode::And {
            left: AigLiteral::node(0),
            right: AigLiteral::node(and_index + 1),
        };
        assert!(matches!(
            validate_strategy_aig(&aig, budget()),
            Err(StrategyAigError::NonTopologicalEdge { .. })
        ));
    }
}
