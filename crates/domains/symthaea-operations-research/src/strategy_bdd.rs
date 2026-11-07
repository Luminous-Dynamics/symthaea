// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical reduced ordered BDD compilation for finite strategy functions.
//!
//! Truth tables remain the reference semantics. This module compiles them into
//! reduced ordered BDDs using the dependency order in the quantified prefix,
//! validates the DAG structure, and exhaustively checks semantic equivalence.
//! It is a bounded circuit-compilation experiment, not a scalable QBF solver.

use super::quantified_strategy::{BooleanStrategyFunction, StrategyOwner, WinningStrategy};
use std::collections::{BTreeMap, BTreeSet};

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
    /// Semantic dependency variables in prefix-derived order.
    pub dependencies: Vec<usize>,
    /// A permutation of dependency indices specifying the BDD decision order.
    pub variable_order: Vec<usize>,
    pub nodes: Vec<BddNode>,
    pub root: BddRef,
}

/// Limits for compiling and checking one strategy function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyBddBudget {
    pub max_dependency_variables: usize,
    pub max_table_entries: u128,
    pub max_bdd_nodes: usize,
    pub max_support_checks: u128,
    pub max_variable_order_evaluations: usize,
    pub max_sifting_passes: usize,
}

impl Default for StrategyBddBudget {
    fn default() -> Self {
        Self {
            max_dependency_variables: 20,
            max_table_entries: 1 << 20,
            max_bdd_nodes: 1 << 20,
            max_support_checks: 1 << 22,
            max_variable_order_evaluations: 1 << 12,
            max_sifting_passes: 2,
        }
    }
}

/// Structural, budget and semantic failures for strategy BDDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrategyBddError {
    DependencyBudgetExceeded {
        required: usize,
        limit: usize,
    },
    TableEntryBudgetExceeded {
        required: u128,
        limit: u128,
    },
    NodeBudgetExceeded {
        required: usize,
        limit: usize,
    },
    AssignmentSpaceOverflow {
        dependency_count: usize,
    },
    FunctionTableEntryCountMismatch {
        expected: u128,
        actual: usize,
    },
    FunctionTableKeyWidthMismatch {
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
    VariableOrderLengthMismatch {
        expected: usize,
        actual: usize,
    },
    VariableOrderEntryOutOfRange {
        position: usize,
        dependency_index: usize,
        dependency_count: usize,
    },
    DuplicateVariableOrderEntry {
        dependency_index: usize,
    },
    DuplicateDependencyVariable {
        variable: usize,
    },
    StrategyFunctionsNotCanonical {
        previous: usize,
        current: usize,
    },
    UnreachableNode {
        node: usize,
    },
    VariableOrderSearchBudgetExceeded {
        required: usize,
        limit: usize,
    },
    SupportCheckBudgetExceeded {
        required: u128,
        limit: u128,
    },
    StrategyVariableMismatch {
        original: usize,
        reduced: usize,
    },
    SupportReductionMismatch {
        assignment: Vec<bool>,
        original_value: bool,
        reduced_value: bool,
    },
    RootNodeOutOfRange {
        node: usize,
        node_count: usize,
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
    DependencyOrderViolation {
        parent: usize,
        child: usize,
        parent_dependency: usize,
        child_dependency: usize,
    },
    UnreducedNode {
        node: usize,
    },
    DuplicateNode {
        node: usize,
        earlier: usize,
    },
    EvaluationAssignmentWidthMismatch {
        expected: usize,
        actual: usize,
    },
    EvaluationStepBudgetExceeded {
        required: usize,
        limit: usize,
    },
    SemanticMismatch {
        assignment: Vec<bool>,
        table_value: bool,
        bdd_value: bool,
    },
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

/// Result of deterministic BDD variable-order sifting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyBddSiftingReport {
    pub initial_variable_order: Vec<usize>,
    pub optimized_variable_order: Vec<usize>,
    pub initial_node_count: usize,
    pub optimized_node_count: usize,
    pub order_evaluations: usize,
    pub bdd: ReducedOrderedBdd,
}

/// Evidence for exact semantic-support reduction of one strategy function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategySupportReductionReport {
    pub original_dependency_count: usize,
    pub reduced_dependency_count: usize,
    pub support_pair_checks: u128,
    pub original_assignments_checked: u128,
    pub reduced_table_entries: u128,
}

/// A support-minimized strategy function and its independent equivalence report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyFunctionSupportReduction {
    pub function: BooleanStrategyFunction,
    pub report: StrategySupportReductionReport,
}

/// Exact minimum BDD order among all permutations within the declared budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyBddExactOrderSearchReport {
    pub initial_variable_order: Vec<usize>,
    pub optimal_variable_order: Vec<usize>,
    pub initial_node_count: usize,
    pub minimum_node_count: usize,
    pub orders_evaluated: usize,
    pub bdd: ReducedOrderedBdd,
}

/// Strategy-function compiler selected by the bounded optimization pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyBddOptimizationMethod {
    ExactOrderSearch,
    Sifting,
}

/// One compiled strategy function, retaining the reduced truth-table
/// semantics and the independently checked ROBDD representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledStrategyFunction {
    pub variable: usize,
    pub original_dependencies: Vec<usize>,
    pub reduced_function: BooleanStrategyFunction,
    pub support_report: StrategySupportReductionReport,
    pub optimization_method: StrategyBddOptimizationMethod,
    pub order_evaluations: usize,
    pub bdd: ReducedOrderedBdd,
}

/// Collection of compiled functions belonging to one winning player.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledWinningStrategy {
    pub owner: StrategyOwner,
    pub functions: Vec<CompiledStrategyFunction>,
    pub original_dependency_count: usize,
    pub reduced_dependency_count: usize,
    pub original_truth_table_entries: u128,
    pub reduced_truth_table_entries: u128,
    pub total_bdd_nodes: usize,
    pub total_order_evaluations: usize,
}

fn checked_power_of_two(exponent: usize) -> Result<u128, StrategyBddError> {
    let shift = u32::try_from(exponent).map_err(|_| StrategyBddError::AssignmentSpaceOverflow {
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
    let mut seen_dependencies = BTreeSet::new();
    for &variable in &function.dependencies {
        if !seen_dependencies.insert(variable) {
            return Err(StrategyBddError::DuplicateDependencyVariable { variable });
        }
    }
    Ok(expected)
}

/// Independently verify that a reduced-support strategy function has the
/// same value as the original for every original dependency assignment.
pub fn verify_support_reduction_equivalence(
    original: &BooleanStrategyFunction,
    reduced: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<StrategySupportReductionReport, StrategyBddError> {
    if original.variable != reduced.variable {
        return Err(StrategyBddError::StrategyVariableMismatch {
            original: original.variable,
            reduced: reduced.variable,
        });
    }
    let original_entries = validate_truth_table(original, budget)?;
    let reduced_entries = validate_truth_table(reduced, budget)?;

    for (reduced_position, dependency) in reduced.dependencies.iter().enumerate() {
        let Some(original_position) = original
            .dependencies
            .iter()
            .position(|candidate| candidate == dependency)
        else {
            return Err(StrategyBddError::DependencyScopeMismatch {
                expected: original.dependencies.clone(),
                actual: reduced.dependencies.clone(),
            });
        };
        if reduced.dependencies[..reduced_position]
            .iter()
            .any(|earlier| {
                original
                    .dependencies
                    .iter()
                    .position(|candidate| candidate == earlier)
                    .is_some_and(|earlier_position| earlier_position >= original_position)
            })
        {
            return Err(StrategyBddError::DependencyScopeMismatch {
                expected: original.dependencies.clone(),
                actual: reduced.dependencies.clone(),
            });
        }
    }

    for mask in 0..original_entries {
        let original_key: Vec<bool> = (0..original.dependencies.len())
            .map(|index| ((mask >> index) & 1) != 0)
            .collect();
        let reduced_key: Vec<bool> = reduced
            .dependencies
            .iter()
            .map(|dependency| {
                original
                    .dependencies
                    .iter()
                    .position(|candidate| candidate == dependency)
                    .map(|position| original_key[position])
                    .unwrap_or(false)
            })
            .collect();
        let Some(&original_value) = original.table.get(&original_key) else {
            return Err(StrategyBddError::MissingTruthTableRow { key: original_key });
        };
        let Some(&reduced_value) = reduced.table.get(&reduced_key) else {
            return Err(StrategyBddError::MissingTruthTableRow { key: reduced_key });
        };
        if original_value != reduced_value {
            return Err(StrategyBddError::SupportReductionMismatch {
                assignment: original_key,
                original_value,
                reduced_value,
            });
        }
    }

    Ok(StrategySupportReductionReport {
        original_dependency_count: original.dependencies.len(),
        reduced_dependency_count: reduced.dependencies.len(),
        support_pair_checks: 0,
        original_assignments_checked: original_entries,
        reduced_table_entries: reduced_entries,
    })
}

/// Remove every declared dependency on which the Boolean function does not
/// actually depend. This is exact semantic support analysis, not syntactic
/// reachability. The result is exhaustively checked against the source table.
pub fn minimize_strategy_function_support(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<StrategyFunctionSupportReduction, StrategyBddError> {
    let entries = validate_truth_table(function, budget)?;
    let mut support_indices = Vec::new();
    let mut support_pair_checks = 0u128;

    for dependency_index in 0..function.dependencies.len() {
        let mut influential = false;
        for mask in 0..entries {
            if ((mask >> dependency_index) & 1) != 0 {
                continue;
            }
            support_pair_checks = support_pair_checks
                .checked_add(1)
                .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
            if support_pair_checks > budget.max_support_checks {
                return Err(StrategyBddError::SupportCheckBudgetExceeded {
                    required: support_pair_checks,
                    limit: budget.max_support_checks,
                });
            }
            let low_key: Vec<bool> = (0..function.dependencies.len())
                .map(|index| ((mask >> index) & 1) != 0)
                .collect();
            let mut high_key = low_key.clone();
            high_key[dependency_index] = true;
            let low = function.table.get(&low_key).ok_or_else(|| {
                StrategyBddError::MissingTruthTableRow {
                    key: low_key.clone(),
                }
            })?;
            let high = function.table.get(&high_key).ok_or_else(|| {
                StrategyBddError::MissingTruthTableRow {
                    key: high_key.clone(),
                }
            })?;
            if low != high {
                influential = true;
                break;
            }
        }
        if influential {
            support_indices.push(dependency_index);
        }
    }

    let reduced_entries = checked_power_of_two(support_indices.len())?;
    if reduced_entries > budget.max_table_entries {
        return Err(StrategyBddError::TableEntryBudgetExceeded {
            required: reduced_entries,
            limit: budget.max_table_entries,
        });
    }
    let dependencies: Vec<usize> = support_indices
        .iter()
        .map(|&index| function.dependencies[index])
        .collect();
    let mut table = BTreeMap::new();
    for mask in 0..reduced_entries {
        let reduced_key: Vec<bool> = (0..support_indices.len())
            .map(|index| ((mask >> index) & 1) != 0)
            .collect();
        let mut original_key = vec![false; function.dependencies.len()];
        for (reduced_index, &original_index) in support_indices.iter().enumerate() {
            original_key[original_index] = reduced_key[reduced_index];
        }
        let value = *function.table.get(&original_key).ok_or_else(|| {
            StrategyBddError::MissingTruthTableRow {
                key: original_key.clone(),
            }
        })?;
        table.insert(reduced_key, value);
    }

    let reduced_function = BooleanStrategyFunction {
        variable: function.variable,
        dependencies,
        table,
    };
    let mut report = verify_support_reduction_equivalence(function, &reduced_function, budget)?;
    report.support_pair_checks = support_pair_checks;

    Ok(StrategyFunctionSupportReduction {
        function: reduced_function,
        report,
    })
}

fn build_level(
    level: usize,
    variable_order: &[usize],
    table: &BTreeMap<Vec<bool>, bool>,
    prefix: &mut Vec<bool>,
    nodes: &mut Vec<BddNode>,
    unique: &mut BTreeMap<(usize, BddRef, BddRef), usize>,
    budget: StrategyBddBudget,
) -> Result<BddRef, StrategyBddError> {
    if level == variable_order.len() {
        let Some(&value) = table.get(prefix) else {
            return Err(StrategyBddError::MissingTruthTableRow {
                key: prefix.clone(),
            });
        };
        return Ok(if value { BddRef::True } else { BddRef::False });
    }

    let dependency_index = variable_order[level];
    prefix[dependency_index] = false;
    let low = build_level(
        level + 1,
        variable_order,
        table,
        prefix,
        nodes,
        unique,
        budget,
    )?;

    prefix[dependency_index] = true;
    let high = build_level(
        level + 1,
        variable_order,
        table,
        prefix,
        nodes,
        unique,
        budget,
    )?;

    // Eliminate a test whose two branches are identical.
    if low == high {
        return Ok(low);
    }

    // Merge structurally identical nodes.
    let key = (dependency_index, low, high);
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
        dependency_index,
        low,
        high,
    });
    unique.insert(key, index);
    Ok(BddRef::Node(index))
}

/// Validate that a decision order is a permutation of the semantic
/// dependency indices.
fn validate_variable_order(
    dependency_count: usize,
    variable_order: &[usize],
) -> Result<(), StrategyBddError> {
    if variable_order.len() != dependency_count {
        return Err(StrategyBddError::VariableOrderLengthMismatch {
            expected: dependency_count,
            actual: variable_order.len(),
        });
    }
    let mut seen = vec![false; dependency_count];
    for (position, &dependency_index) in variable_order.iter().enumerate() {
        if dependency_index >= dependency_count {
            return Err(StrategyBddError::VariableOrderEntryOutOfRange {
                position,
                dependency_index,
                dependency_count,
            });
        }
        if seen[dependency_index] {
            return Err(StrategyBddError::DuplicateVariableOrderEntry { dependency_index });
        }
        seen[dependency_index] = true;
    }
    Ok(())
}

/// Compile a strategy function using its semantic dependency order as the
/// default BDD decision order.
pub fn compile_strategy_function_to_robdd(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<ReducedOrderedBdd, StrategyBddError> {
    let variable_order: Vec<usize> = (0..function.dependencies.len()).collect();
    compile_strategy_function_to_robdd_with_order(function, &variable_order, budget)
}

/// Compile using a chosen permutation of dependency indices. Semantic scope
/// order is preserved; only the order in which the BDD tests those variables
/// changes.
pub fn compile_strategy_function_to_robdd_with_order(
    function: &BooleanStrategyFunction,
    variable_order: &[usize],
    budget: StrategyBddBudget,
) -> Result<ReducedOrderedBdd, StrategyBddError> {
    validate_truth_table(function, budget)?;
    validate_variable_order(function.dependencies.len(), variable_order)?;
    let mut nodes = Vec::new();
    let mut unique = BTreeMap::new();
    let mut prefix = vec![false; function.dependencies.len()];
    let root = build_level(
        0,
        variable_order,
        &function.table,
        &mut prefix,
        &mut nodes,
        &mut unique,
        budget,
    )?;
    let bdd = ReducedOrderedBdd {
        dependencies: function.dependencies.clone(),
        variable_order: variable_order.to_vec(),
        nodes,
        root,
    };
    validate_robdd_with_budget(&bdd, budget)?;
    Ok(bdd)
}

/// Check topological node numbering, increasing variable order, reduction,
/// uniqueness, and root/edge validity without consulting the source function.
/// Validate an ROBDD against explicit dependency and node budgets.
pub fn validate_robdd_with_budget(
    bdd: &ReducedOrderedBdd,
    budget: StrategyBddBudget,
) -> Result<(), StrategyBddError> {
    if bdd.dependencies.len() > budget.max_dependency_variables {
        return Err(StrategyBddError::DependencyBudgetExceeded {
            required: bdd.dependencies.len(),
            limit: budget.max_dependency_variables,
        });
    }
    if bdd.nodes.len() > budget.max_bdd_nodes {
        return Err(StrategyBddError::NodeBudgetExceeded {
            required: bdd.nodes.len(),
            limit: budget.max_bdd_nodes,
        });
    }
    validate_variable_order(bdd.dependencies.len(), &bdd.variable_order)?;

    let mut dependencies_seen = BTreeSet::new();
    for &variable in &bdd.dependencies {
        if !dependencies_seen.insert(variable) {
            return Err(StrategyBddError::DuplicateDependencyVariable { variable });
        }
    }

    if let BddRef::Node(root) = bdd.root {
        if root >= bdd.nodes.len() {
            return Err(StrategyBddError::RootNodeOutOfRange {
                node: root,
                node_count: bdd.nodes.len(),
            });
        }
    }

    let mut dependency_rank = vec![usize::MAX; bdd.dependencies.len()];
    for (rank, &dependency_index) in bdd.variable_order.iter().enumerate() {
        dependency_rank[dependency_index] = rank;
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
                if dependency_rank[child_dependency] <= dependency_rank[node.dependency_index] {
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

    // Every serialized node contributes to the claimed circuit size. Reject
    // detached padding nodes rather than reporting inflated/non-canonical DAGs.
    let mut reachable = vec![false; bdd.nodes.len()];
    let mut pending = Vec::new();
    if let BddRef::Node(root) = bdd.root {
        pending.push(root);
    }
    while let Some(node_index) = pending.pop() {
        if reachable[node_index] {
            continue;
        }
        reachable[node_index] = true;
        let node = &bdd.nodes[node_index];
        for child in [node.low, node.high] {
            if let BddRef::Node(child_index) = child {
                pending.push(child_index);
            }
        }
    }
    if let Some(node) = reachable.iter().position(|visited| !visited) {
        return Err(StrategyBddError::UnreachableNode { node });
    }
    Ok(())
}

/// Validate against the default resource envelope.
pub fn validate_robdd(bdd: &ReducedOrderedBdd) -> Result<(), StrategyBddError> {
    validate_robdd_with_budget(bdd, StrategyBddBudget::default())
}

/// Deterministically sift each dependency through every BDD order position.
///
/// The semantic dependency list and truth table are not modified. Sifting only
/// chooses the variable test order inside the ROBDD. For each variable, the
/// best insertion position is selected by node count, with lexicographic order
/// as a deterministic tie-break. Candidate orders that exceed the node budget
/// are discarded; exhausting the order-evaluation budget returns an error.
pub fn sift_strategy_function_robdd(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<StrategyBddSiftingReport, StrategyBddError> {
    validate_truth_table(function, budget)?;
    let dependency_count = function.dependencies.len();
    let mut order: Vec<usize> = (0..dependency_count).collect();
    let mut bdd = compile_strategy_function_to_robdd_with_order(function, &order, budget)?;
    let initial_variable_order = order.clone();
    let initial_node_count = bdd.nodes.len();
    let mut evaluations = 1usize;

    for _ in 0..budget.max_sifting_passes {
        let mut pass_improved = false;
        for dependency_index in 0..dependency_count {
            let mut remainder = order.clone();
            let Some(current_position) = remainder
                .iter()
                .position(|candidate| *candidate == dependency_index)
            else {
                return Err(StrategyBddError::VariableOrderEntryOutOfRange {
                    position: dependency_index,
                    dependency_index,
                    dependency_count,
                });
            };
            remainder.remove(current_position);

            let mut best_order = order.clone();
            let mut best_bdd = bdd.clone();
            for insertion_position in 0..dependency_count {
                let mut candidate = remainder.clone();
                candidate.insert(insertion_position, dependency_index);
                if candidate == order {
                    continue;
                }

                let required = evaluations
                    .checked_add(1)
                    .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                if required > budget.max_variable_order_evaluations {
                    return Err(StrategyBddError::VariableOrderSearchBudgetExceeded {
                        required,
                        limit: budget.max_variable_order_evaluations,
                    });
                }
                evaluations = required;

                match compile_strategy_function_to_robdd_with_order(function, &candidate, budget) {
                    Ok(candidate_bdd) => {
                        let candidate_nodes = candidate_bdd.nodes.len();
                        let best_nodes = best_bdd.nodes.len();
                        if candidate_nodes < best_nodes
                            || (candidate_nodes == best_nodes && candidate < best_order)
                        {
                            best_order = candidate;
                            best_bdd = candidate_bdd;
                        }
                    }
                    Err(StrategyBddError::NodeBudgetExceeded { .. }) => {}
                    Err(error) => return Err(error),
                }
            }

            if best_order != order {
                order = best_order;
                bdd = best_bdd;
                pass_improved = true;
            }
        }
        if !pass_improved {
            break;
        }
    }

    Ok(StrategyBddSiftingReport {
        initial_variable_order,
        optimized_variable_order: order,
        initial_node_count,
        optimized_node_count: bdd.nodes.len(),
        order_evaluations: evaluations,
        bdd,
    })
}

fn next_permutation(values: &mut [usize]) -> bool {
    if values.len() < 2 {
        return false;
    }
    let Some(pivot) = (0..values.len() - 1)
        .rev()
        .find(|&index| values[index] < values[index + 1])
    else {
        return false;
    };
    let successor = (pivot + 1..values.len())
        .rev()
        .find(|&index| values[index] > values[pivot])
        .expect("a pivot has a greater successor");
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}

/// Find the exact smallest ROBDD among every variable permutation. This is
/// intended as a small-scope oracle for measuring heuristic order quality.
/// The factorial work is preflighted against max_variable_order_evaluations.
pub fn find_exact_minimum_robdd_order(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<StrategyBddExactOrderSearchReport, StrategyBddError> {
    validate_truth_table(function, budget)?;
    let dependency_count = function.dependencies.len();
    let mut permutation_count = 1usize;
    for factor in 2..=dependency_count {
        permutation_count = permutation_count
            .checked_mul(factor)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
    }
    if permutation_count > budget.max_variable_order_evaluations {
        return Err(StrategyBddError::VariableOrderSearchBudgetExceeded {
            required: permutation_count,
            limit: budget.max_variable_order_evaluations,
        });
    }

    let mut order: Vec<usize> = (0..dependency_count).collect();
    let initial_variable_order = order.clone();
    let initial_bdd = compile_strategy_function_to_robdd_with_order(function, &order, budget)?;
    let initial_node_count = initial_bdd.nodes.len();
    let mut best_order = order.clone();
    let mut best_bdd = initial_bdd;
    // The identity order was compiled above and is the first lexicographic
    // permutation. Count it once, then enumerate every remaining permutation.
    let mut orders_evaluated = 1usize;

    while next_permutation(&mut order) {
        let candidate_bdd =
            compile_strategy_function_to_robdd_with_order(function, &order, budget)?;
        orders_evaluated = orders_evaluated
            .checked_add(1)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        if candidate_bdd.nodes.len() < best_bdd.nodes.len()
            || (candidate_bdd.nodes.len() == best_bdd.nodes.len() && order < best_order)
        {
            best_order = order.clone();
            best_bdd = candidate_bdd;
        }
    }

    Ok(StrategyBddExactOrderSearchReport {
        initial_variable_order,
        optimal_variable_order: best_order,
        initial_node_count,
        minimum_node_count: best_bdd.nodes.len(),
        orders_evaluated,
        bdd: best_bdd,
    })
}

fn exact_order_space_fits(
    dependency_count: usize,
    evaluation_budget: usize,
) -> bool {
    let mut permutations = 1usize;
    for factor in 2..=dependency_count {
        let Some(next) = permutations.checked_mul(factor) else {
            return false;
        };
        if next > evaluation_budget {
            return false;
        }
        permutations = next;
    }
    true
}

/// Compile all functions of a winning strategy into support-minimized ROBDDs.
///
/// The caller must first independently verify the source winning strategy
/// against its QBF prefix and CNF matrix. This function then independently
/// verifies each support reduction and each BDD against the corresponding
/// source truth table. It chooses exact permutation search whenever n! fits
/// the declared order budget, and deterministic sifting otherwise.
pub fn compile_winning_strategy_functions(
    strategy: &WinningStrategy,
    budget: StrategyBddBudget,
) -> Result<CompiledWinningStrategy, StrategyBddError> {
    let mut functions = Vec::with_capacity(strategy.functions.len());
    let mut previous_variable = None;
    let mut original_dependency_count = 0usize;
    let mut reduced_dependency_count = 0usize;
    let mut original_truth_table_entries = 0u128;
    let mut reduced_truth_table_entries = 0u128;
    let mut total_bdd_nodes = 0usize;
    let mut total_order_evaluations = 0usize;

    for source_function in &strategy.functions {
        if let Some(previous) = previous_variable {
            if source_function.variable <= previous {
                return Err(StrategyBddError::StrategyFunctionsNotCanonical {
                    previous,
                    current: source_function.variable,
                });
            }
        }
        previous_variable = Some(source_function.variable);

        let support = minimize_strategy_function_support(source_function, budget)?;
        let optimization_method;
        let order_evaluations;
        let bdd = if exact_order_space_fits(
            support.function.dependencies.len(),
            budget.max_variable_order_evaluations,
        ) {
            let report = find_exact_minimum_robdd_order(&support.function, budget)?;
            optimization_method = StrategyBddOptimizationMethod::ExactOrderSearch;
            order_evaluations = report.orders_evaluated;
            report.bdd
        } else {
            let report = sift_strategy_function_robdd(&support.function, budget)?;
            optimization_method = StrategyBddOptimizationMethod::Sifting;
            order_evaluations = report.order_evaluations;
            report.bdd
        };

        verify_robdd_matches_strategy_function(&bdd, &support.function, budget)?;

        original_dependency_count = original_dependency_count
            .checked_add(source_function.dependencies.len())
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        reduced_dependency_count = reduced_dependency_count
            .checked_add(support.function.dependencies.len())
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        original_truth_table_entries = original_truth_table_entries
            .checked_add(checked_power_of_two(source_function.dependencies.len())?)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        reduced_truth_table_entries = reduced_truth_table_entries
            .checked_add(support.report.reduced_table_entries)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        total_bdd_nodes = total_bdd_nodes
            .checked_add(bdd.nodes.len())
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        total_order_evaluations = total_order_evaluations
            .checked_add(order_evaluations)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;

        functions.push(CompiledStrategyFunction {
            variable: source_function.variable,
            original_dependencies: source_function.dependencies.clone(),
            reduced_function: support.function,
            support_report: support.report,
            optimization_method,
            order_evaluations,
            bdd,
        });
    }

    Ok(CompiledWinningStrategy {
        owner: strategy.owner,
        functions,
        original_dependency_count,
        reduced_dependency_count,
        original_truth_table_entries,
        reduced_truth_table_entries,
        total_bdd_nodes,
        total_order_evaluations,
    })
}

/// Evaluate an ROBDD using the default resource envelope.
pub fn evaluate_robdd(
    bdd: &ReducedOrderedBdd,
    assignment: &[bool],
) -> Result<bool, StrategyBddError> {
    evaluate_robdd_with_budget(bdd, assignment, StrategyBddBudget::default())
}

/// Evaluate an ROBDD under an explicit structural resource budget.
pub fn evaluate_robdd_with_budget(
    bdd: &ReducedOrderedBdd,
    assignment: &[bool],
    budget: StrategyBddBudget,
) -> Result<bool, StrategyBddError> {
    validate_robdd_with_budget(bdd, budget)?;
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
    validate_robdd_with_budget(bdd, budget)?;
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
            return Err(StrategyBddError::MissingTruthTableRow { key: assignment });
        };
        let bdd_value = evaluate_robdd_with_budget(bdd, &assignment, budget)?;
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
            max_support_checks: 1 << 12,
            max_variable_order_evaluations: 128,
            max_sifting_passes: 2,
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
    fn custom_bdd_order_preserves_the_same_semantics() {
        let source = function(
            &[10, 11, 12],
            &[false, true, false, true, false, false, true, true],
        );
        let order = [2, 0, 1];
        let bdd = compile_strategy_function_to_robdd_with_order(&source, &order, budget()).unwrap();
        assert_eq!(bdd.variable_order, order);
        verify_robdd_matches_strategy_function(&bdd, &source, budget()).unwrap();
    }

    #[test]
    fn variable_order_sifting_reduces_mux_strategy_size() {
        // Dependencies are [b, c, a], while f = a ? c : b.
        let source = function(
            &[11, 12, 10],
            &[false, true, false, true, false, false, true, true],
        );
        let initial = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        let report = sift_strategy_function_robdd(&source, budget()).unwrap();
        assert_eq!(initial.nodes.len(), 5);
        assert_eq!(report.initial_node_count, 5);
        assert_eq!(report.optimized_node_count, 3);
        assert!(report.optimized_node_count <= report.initial_node_count);
        verify_robdd_matches_strategy_function(&report.bdd, &source, budget()).unwrap();
    }

    #[test]
    fn variable_order_sifting_budget_is_fail_closed() {
        let source = function(
            &[11, 12, 10],
            &[false, true, false, true, false, false, true, true],
        );
        let tight = StrategyBddBudget {
            max_variable_order_evaluations: 1,
            ..budget()
        };
        assert!(matches!(
            sift_strategy_function_robdd(&source, tight),
            Err(StrategyBddError::VariableOrderSearchBudgetExceeded { .. })
        ));
    }

    #[test]
    fn validator_rejects_unreachable_padding_nodes() {
        let source = function(
            &[0, 1, 2],
            &[false, true, false, true, false, true, false, true],
        );
        let mut bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        bdd.nodes.push(BddNode {
            dependency_index: 2,
            low: BddRef::False,
            high: BddRef::True,
        });
        assert!(matches!(
            validate_robdd(&bdd),
            Err(StrategyBddError::UnreachableNode { .. })
        ));
    }

    #[test]
    fn validator_rejects_non_permutation_variable_order() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let mut bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        bdd.variable_order = vec![0, 0];
        assert!(matches!(
            validate_robdd(&bdd),
            Err(StrategyBddError::DuplicateVariableOrderEntry { .. })
        ));
    }

    #[test]
    fn validator_rejects_duplicate_semantic_dependencies() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let mut bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        bdd.dependencies = vec![0, 0];
        assert!(matches!(
            validate_robdd(&bdd),
            Err(StrategyBddError::DuplicateDependencyVariable { variable: 0 })
        ));
    }

    #[test]
    fn support_minimization_removes_irrelevant_dependency() {
        // Function is the first dependency only; the second is unused.
        let source = function(&[7, 2], &[false, true, false, true]);
        let reduced = minimize_strategy_function_support(&source, budget()).unwrap();
        assert_eq!(reduced.function.dependencies, vec![7]);
        assert_eq!(reduced.function.table.len(), 2);
        assert_eq!(reduced.report.original_dependency_count, 2);
        assert_eq!(reduced.report.reduced_dependency_count, 1);
        assert_eq!(reduced.report.original_assignments_checked, 4);
    }

    #[test]
    fn support_minimization_reduces_constant_function_to_zero_dependencies() {
        let source = function(&[4, 2, 9], &[true; 8]);
        let reduced = minimize_strategy_function_support(&source, budget()).unwrap();
        assert!(reduced.function.dependencies.is_empty());
        assert_eq!(reduced.function.table.get(&vec![]), Some(&true));
        assert_eq!(reduced.report.reduced_dependency_count, 0);
    }

    #[test]
    fn support_analysis_budget_is_fail_closed() {
        let source = function(
            &[0, 1, 2],
            &[false, true, false, true, false, true, false, true],
        );
        let tight = StrategyBddBudget {
            max_support_checks: 0,
            ..budget()
        };
        assert!(matches!(
            minimize_strategy_function_support(&source, tight),
            Err(StrategyBddError::SupportCheckBudgetExceeded { .. })
        ));
    }

    #[test]
    fn exact_order_search_finds_oracle_minimum_and_sifting_is_not_larger() {
        let source = function(
            &[11, 12, 10],
            &[false, true, false, true, false, false, true, true],
        );
        let exact = find_exact_minimum_robdd_order(&source, budget()).unwrap();
        let sifted = sift_strategy_function_robdd(&source, budget()).unwrap();
        assert_eq!(exact.orders_evaluated, 6);
        assert_eq!(exact.minimum_node_count, 3);
        assert!(sifted.optimized_node_count >= exact.minimum_node_count);
        assert!(sifted.optimized_node_count <= exact.initial_node_count);
        verify_robdd_matches_strategy_function(&exact.bdd, &source, budget()).unwrap();
    }

    #[test]
    fn exact_order_search_refuses_factorial_budget_overrun() {
        let source = function(&[0, 1, 2, 3], &[false; 16]);
        let tight = StrategyBddBudget {
            max_variable_order_evaluations: 23,
            ..budget()
        };
        assert!(matches!(
            find_exact_minimum_robdd_order(&source, tight),
            Err(StrategyBddError::VariableOrderSearchBudgetExceeded {
                required: 24,
                limit: 23
            })
        ));
    }

    #[test]
    fn every_three_variable_boolean_function_preserves_support_and_bdd_semantics() {
        let mut maximum_sift_gap = 0usize;
        for function_mask in 0..256usize {
            let values: Vec<bool> = (0..8)
                .map(|assignment| ((function_mask >> assignment) & 1) != 0)
                .collect();
            let source = function(&[0, 1, 2], &values);
            let support = minimize_strategy_function_support(&source, budget()).unwrap();
            verify_support_reduction_equivalence(&source, &support.function, budget()).unwrap();
            let exact = find_exact_minimum_robdd_order(&support.function, budget()).unwrap();
            let sifted = sift_strategy_function_robdd(&support.function, budget()).unwrap();
            assert!(exact.minimum_node_count <= sifted.optimized_node_count);
            assert!(sifted.optimized_node_count <= sifted.initial_node_count);
            verify_robdd_matches_strategy_function(&exact.bdd, &support.function, budget())
                .unwrap();
            maximum_sift_gap =
                maximum_sift_gap.max(sifted.optimized_node_count - exact.minimum_node_count);
        }
        assert_eq!(maximum_sift_gap, 0, "three-variable sifting should match exact order search");
    }

    #[test]
    fn verifier_enforces_serialized_node_budget_before_graph_walk() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        let tight = StrategyBddBudget {
            max_bdd_nodes: 2,
            ..budget()
        };
        assert_eq!(
            verify_robdd_matches_strategy_function(&bdd, &source, tight).unwrap_err(),
            StrategyBddError::NodeBudgetExceeded {
                required: 3,
                limit: 2
            }
        );
    }

    #[test]
    fn evaluator_enforces_explicit_node_budget() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let bdd = compile_strategy_function_to_robdd(&source, budget()).unwrap();
        let tight = StrategyBddBudget {
            max_bdd_nodes: 2,
            ..budget()
        };
        assert!(matches!(
            evaluate_robdd_with_budget(&bdd, &[false, false], tight),
            Err(StrategyBddError::NodeBudgetExceeded {
                required: 3,
                limit: 2
            })
        ));
    }

    #[test]
    fn high_level_strategy_compiler_reduces_support_and_builds_exact_bdd() {
        let source = BooleanStrategyFunction {
            variable: 3,
            dependencies: vec![0, 1],
            table: BTreeMap::from([
                (vec![false, false], false),
                (vec![false, true], true),
                (vec![true, false], false),
                (vec![true, true], true),
            ]),
        };
        let strategy = WinningStrategy {
            owner: StrategyOwner::Existential,
            functions: vec![source],
        };
        let compiled = compile_winning_strategy_functions(&strategy, budget()).unwrap();
        assert_eq!(compiled.owner, StrategyOwner::Existential);
        assert_eq!(compiled.functions.len(), 1);
        assert_eq!(compiled.original_dependency_count, 2);
        assert_eq!(compiled.reduced_dependency_count, 1);
        assert_eq!(compiled.original_truth_table_entries, 4);
        assert_eq!(compiled.reduced_truth_table_entries, 2);
        assert_eq!(compiled.total_bdd_nodes, 1);
        assert_eq!(
            compiled.functions[0].optimization_method,
            StrategyBddOptimizationMethod::ExactOrderSearch
        );
    }

    #[test]
    fn high_level_strategy_compiler_uses_sifting_when_exact_order_exceeds_budget() {
        let values: Vec<bool> = (0..64)
            .map(|mask| mask.count_ones() % 2 == 1)
            .collect();
        let source = function(&[0, 1, 2, 3, 4, 5], &values);
        let strategy = WinningStrategy {
            owner: StrategyOwner::Universal,
            functions: vec![source],
        };
        let constrained = StrategyBddBudget {
            max_variable_order_evaluations: 128,
            ..budget()
        };
        let compiled = compile_winning_strategy_functions(&strategy, constrained).unwrap();
        assert_eq!(
            compiled.functions[0].optimization_method,
            StrategyBddOptimizationMethod::Sifting
        );
        assert_eq!(compiled.functions[0].reduced_function.dependencies.len(), 6);
        assert!(compiled.total_bdd_nodes > 0);
        assert!(compiled.total_order_evaluations <= 128);
    }

    #[test]
    fn high_level_compiler_rejects_noncanonical_strategy_function_order() {
        let first = BooleanStrategyFunction {
            variable: 2,
            ..function(&[0], &[false, true])
        };
        let second = BooleanStrategyFunction {
            variable: 1,
            ..function(&[0], &[false, true])
        };
        let strategy = WinningStrategy {
            owner: StrategyOwner::Existential,
            functions: vec![first, second],
        };
        assert!(matches!(
            compile_winning_strategy_functions(&strategy, budget()),
            Err(StrategyBddError::StrategyFunctionsNotCanonical { .. })
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
