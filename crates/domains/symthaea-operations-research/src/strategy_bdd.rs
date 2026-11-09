// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical reduced ordered BDD compilation for finite strategy functions.
//!
//! Truth tables remain the reference semantics. This module compiles them into
//! reduced ordered BDDs using the dependency order in the quantified prefix,
//! validates the DAG structure, and exhaustively checks semantic equivalence.
//! It is a bounded circuit-compilation experiment, not a scalable QBF solver.

use super::decomposition::{CnfProblem, Literal};
use super::quantified_scope::{Quantifier, QuantifierBlock, ValidatedQuantifierPrefix};
use super::quantified_strategy::{
    BooleanStrategyFunction, QbfEvaluationResult, QuantifiedEvaluationBudget,
    QuantifiedStrategyError, StrategyOwner, StrategyVerificationReport, WinningStrategy,
    verify_winning_strategy,
};
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
    pub max_interaction_checks: u128,
    pub max_weighted_interaction_checks: u128,
    pub max_anf_transform_checks: u128,
    pub max_frontier_order_evaluations: usize,
    pub max_frontier_refinement_passes: usize,
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
            max_interaction_checks: 1 << 26,
            max_weighted_interaction_checks: 1 << 26,
            max_anf_transform_checks: 1 << 26,
            max_frontier_order_evaluations: 1 << 12,
            max_frontier_refinement_passes: 2,
            max_variable_order_evaluations: 1 << 13,
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
    InteractionGraphCheckBudgetExceeded {
        required: u128,
        limit: u128,
    },
    FrontierOrderSearchBudgetExceeded {
        required: usize,
        limit: usize,
    },
    InteractionGraphNodeOutOfRange {
        vertex: usize,
        neighbor: usize,
        vertex_count: usize,
    },
    InteractionGraphSelfEdge {
        vertex: usize,
    },
    InteractionGraphAsymmetricEdge {
        vertex: usize,
        neighbor: usize,
    },
    WeightedInteractionGraphCheckBudgetExceeded {
        required: u128,
        limit: u128,
    },
    AnfHypergraphInvalidEdge {
        dependency_count: usize,
        edge: Vec<usize>,
    },
    AnfTransformCheckBudgetExceeded {
        required: u128,
        limit: u128,
    },
    WeightedInteractionGraphRowWidthMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    WeightedInteractionGraphSelfWeight {
        vertex: usize,
    },
    WeightedInteractionGraphAsymmetricWeight {
        vertex: usize,
        neighbor: usize,
        forward: u128,
        reverse: u128,
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
    /// Whether every configured sifting pass completed. This is not a proof
    /// that the selected ordering is globally optimal.
    pub search_complete: bool,
    pub bdd: ReducedOrderedBdd,
}

/// Deterministic sources of a ROBDD variable-order starting point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StrategyBddOrderSeed {
    Natural,
    Reverse,
    InteractionMinFill,
    InteractionMinFillReverse,
    FrontierRefinedMinFill,
    FrontierRefinedMinFillReverse,
    ArithmeticWeightedMinFill,
    ArithmeticWeightedMinFillReverse,
    ArithmeticWeightedFrontierRefined,
    ArithmeticWeightedFrontierRefinedReverse,
    AnfHypergraphMinFill,
    AnfHypergraphMinFillReverse,
}

/// A bounded multi-start sifting result. The interaction graph is a heuristic
/// only; the semantic dependency list remains unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyBddMultiStartReport {
    pub seed_orders: Vec<(StrategyBddOrderSeed, Vec<usize>)>,
    pub selected_seed: StrategyBddOrderSeed,
    pub selected_initial_node_count: usize,
    pub optimized_node_count: usize,
    pub interaction_pair_checks: u128,
    pub arithmetic_interaction_context_checks: u128,
    pub arithmetic_interaction_total_weight: u128,
    pub anf_hyperedge_count: usize,
    pub anf_transform_checks: u128,
    pub anf_max_hyperedge_rank: usize,
    pub weighted_min_fill_max_frontier: usize,
    pub weighted_refined_max_frontier: usize,
    pub frontier_refinement_evaluations: usize,
    pub min_fill_max_frontier: usize,
    pub refined_max_frontier: usize,
    pub order_evaluations: usize,
    pub seeds_evaluated: usize,
    /// Whether every unique seed completed its configured sifting schedule.
    pub search_complete: bool,
    pub bdd: ReducedOrderedBdd,
}

/// Exact arithmetic mixed-derivative coupling weights for one Boolean strategy
/// function. Each pair weight is the sum of |f00 - f01 - f10 + f11| over
/// every assignment to the other inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyBddWeightedInteractionReport {
    pub weights: Vec<Vec<u128>>,
    pub contexts_checked: u128,
    pub total_interaction_weight: u128,
}

/// Exact algebraic-normal-form interactions recovered by the Boolean Möbius
/// transform. Each hyperedge is the support of one nonzero ANF monomial of
/// degree >= 2; singleton linear terms are represented separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyBddAnfHypergraphReport {
    pub hyperedges: Vec<Vec<usize>>,
    pub linear_variables: Vec<usize>,
    pub constant_term: bool,
    pub transform_checks: u128,
    pub max_hyperedge_rank: usize,
}

/// Evidence from bounded insertion refinement of an interaction-graph order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyBddFrontierRefinementReport {
    pub initial_order: Vec<usize>,
    pub refined_order: Vec<usize>,
    pub initial_frontier_profile: Vec<usize>,
    pub refined_frontier_profile: Vec<usize>,
    pub order_evaluations: usize,
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
    MultiStartSifting,
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
    pub order_seed: Option<StrategyBddOrderSeed>,
    pub interaction_pair_checks: u128,
    pub arithmetic_interaction_context_checks: u128,
    pub arithmetic_interaction_total_weight: u128,
    pub weighted_min_fill_max_frontier: usize,
    pub weighted_refined_max_frontier: usize,
    pub frontier_refinement_evaluations: usize,
    pub min_fill_max_frontier: usize,
    pub refined_max_frontier: usize,
    pub order_evaluations: usize,
    pub optimization_search_complete: bool,
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
    pub optimization_search_complete: bool,
}

/// Result of verifying the source QBF strategy and then compiling its
/// functions. The source verification report and representation metrics are
/// distinct evidence layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCompiledQbfStrategy {
    pub value: bool,
    pub source_strategy_verification: StrategyVerificationReport,
    pub representation: CompiledWinningStrategy,
}

/// Errors at the boundary between original-QBF verification and representation
/// compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifiedStrategyCompilationError {
    ResultOwnerMismatch { value: bool, owner: StrategyOwner },
    SourceStrategyInvalid(QuantifiedStrategyError),
    RepresentationInvalid(StrategyBddError),
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
    let initial_order: Vec<usize> = (0..function.dependencies.len()).collect();
    sift_strategy_function_robdd_from_order(function, &initial_order, budget)
}

/// Sift from an explicit permutation of the legal semantic support.
pub fn sift_strategy_function_robdd_from_order(
    function: &BooleanStrategyFunction,
    initial_order: &[usize],
    budget: StrategyBddBudget,
) -> Result<StrategyBddSiftingReport, StrategyBddError> {
    validate_truth_table(function, budget)?;
    let dependency_count = function.dependencies.len();
    validate_variable_order(dependency_count, initial_order)?;
    if budget.max_variable_order_evaluations == 0 {
        return Err(StrategyBddError::VariableOrderSearchBudgetExceeded {
            required: 1,
            limit: 0,
        });
    }

    // Establish and retain a valid baseline before spending the remaining
    // budget on alternative orderings.
    let mut order = initial_order.to_vec();
    let mut bdd = compile_strategy_function_to_robdd_with_order(function, &order, budget)?;
    let initial_variable_order = order.clone();
    let initial_node_count = bdd.nodes.len();
    let mut evaluations = 1usize;
    let mut search_complete = true;

    'passes: for _ in 0..budget.max_sifting_passes {
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
            let mut budget_exhausted = false;
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
                    budget_exhausted = true;
                    break;
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

            // Commit the best candidate seen so far for this variable before
            // returning an incomplete-search result.
            if best_order != order {
                order = best_order;
                bdd = best_bdd;
                pass_improved = true;
            }
            if budget_exhausted {
                search_complete = false;
                break 'passes;
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
        search_complete,
        bdd,
    })
}

fn strategy_table_values(
    function: &BooleanStrategyFunction,
) -> Result<Vec<bool>, StrategyBddError> {
    let entries = checked_power_of_two(function.dependencies.len())?;
    let mut values = vec![false; entries as usize];
    for mask in 0..entries {
        let key: Vec<bool> = (0..function.dependencies.len())
            .map(|index| ((mask >> index) & 1) != 0)
            .collect();
        let Some(&value) = function.table.get(&key) else {
            return Err(StrategyBddError::MissingTruthTableRow { key });
        };
        values[mask as usize] = value;
    }
    Ok(values)
}

/// Expand a context assignment to all dependency positions except i and j.
/// This bit-spreading helper is O(1), avoiding an O(n) key allocation per
/// interaction test.
fn expand_context_mask(mask: usize, first: usize, second: usize) -> usize {
    let low_mask = if first == 0 { 0 } else { (1usize << first) - 1 };
    let low = mask & low_mask;
    let after_first = mask >> first;
    let middle_width = second - first - 1;
    let middle_mask = if middle_width == 0 {
        0
    } else {
        (1usize << middle_width) - 1
    };
    let middle = after_first & middle_mask;
    let high = after_first >> middle_width;
    low | (middle << (first + 1)) | (high << (second + 1))
}

/// Build the pairwise Boolean-interaction graph of a truth table.
///
/// An edge (i,j) exists exactly when the mixed Boolean derivative
/// f00 XOR f01 XOR f10 XOR f11 is nonzero for at least one assignment to all
/// other inputs. The graph is an ordering heuristic only; it does not replace
/// semantic support or prove that higher-order interactions are absent.
pub fn strategy_interaction_graph(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<(Vec<BTreeSet<usize>>, u128), StrategyBddError> {
    let entries = validate_truth_table(function, budget)?;
    let dependency_count = function.dependencies.len();
    let values = strategy_table_values(function)?;
    let contexts_per_pair = checked_power_of_two(dependency_count.saturating_sub(2))?;
    let mut checks = 0u128;
    let mut graph = vec![BTreeSet::new(); dependency_count];

    for first in 0..dependency_count {
        for second in first + 1..dependency_count {
            let mut interacts = false;
            for context in 0..contexts_per_pair {
                checks = checks
                    .checked_add(1)
                    .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                if checks > budget.max_interaction_checks {
                    return Err(StrategyBddError::InteractionGraphCheckBudgetExceeded {
                        required: checks,
                        limit: budget.max_interaction_checks,
                    });
                }
                let base = expand_context_mask(context as usize, first, second);
                let f00 = values[base];
                let f01 = values[base | (1usize << second)];
                let f10 = values[base | (1usize << first)];
                let f11 = values[base | (1usize << first) | (1usize << second)];
                if f00 ^ f01 ^ f10 ^ f11 {
                    interacts = true;
                    break;
                }
            }
            if interacts {
                graph[first].insert(second);
                graph[second].insert(first);
            }
        }
    }

    // The table entry budget already ensures values.len() equals 2^n.
    let _ = entries;
    Ok((graph, checks))
}

/// Recover the algebraic normal form (ANF) of a Boolean truth table using
/// the in-place Boolean Möbius transform.
///
/// If f(x) = XOR_S a_S * product_{i in S} x_i, the transformed coefficient
/// at mask S is a_S. This representation retains higher-order interactions
/// which a pairwise graph can obscure.
pub fn strategy_anf_interaction_hypergraph(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<StrategyBddAnfHypergraphReport, StrategyBddError> {
    let entries = validate_truth_table(function, budget)?;
    let dependency_count = function.dependencies.len();
    let values = strategy_table_values(function)?;
    let required_checks = (dependency_count as u128)
        .checked_mul(entries / 2)
        .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
    if required_checks > budget.max_anf_transform_checks {
        return Err(StrategyBddError::AnfTransformCheckBudgetExceeded {
            required: required_checks,
            limit: budget.max_anf_transform_checks,
        });
    }

    let mut coefficients = values;
    let mut transform_checks = 0u128;
    for variable in 0..dependency_count {
        let bit = 1usize
            .checked_shl(variable as u32)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        for mask in 0..coefficients.len() {
            if mask & bit != 0 {
                coefficients[mask] ^= coefficients[mask ^ bit];
                transform_checks = transform_checks
                    .checked_add(1)
                    .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
            }
        }
    }

    let mut hyperedges = Vec::new();
    let mut linear_variables = Vec::new();
    let mut constant_term = false;
    let mut max_hyperedge_rank = 0usize;
    for (mask, coefficient) in coefficients.into_iter().enumerate() {
        if !coefficient {
            continue;
        }
        if mask == 0 {
            constant_term = true;
            continue;
        }
        let rank = mask.count_ones() as usize;
        // Hypergraph vertex IDs are positions in the semantic dependency
        // scope, matching the local indices accepted by BDD orderings.
        let support: Vec<usize> = (0..dependency_count)
            .filter(|variable| (mask & (1usize << variable)) != 0)
            .collect();
        if rank == 1 {
            linear_variables.extend(support);
        } else {
            max_hyperedge_rank = max_hyperedge_rank.max(rank);
            hyperedges.push(support);
        }
    }
    hyperedges.sort();
    linear_variables.sort_unstable();

    Ok(StrategyBddAnfHypergraphReport {
        hyperedges,
        linear_variables,
        constant_term,
        transform_checks,
        max_hyperedge_rank,
    })
}

/// Deterministic hypergraph min-fill order over ANF monomial supports.
///
/// Eliminating a vertex removes all incident hyperedges and introduces one
/// fill hyperedge over the union of their remaining vertices. Candidate
/// elimination is scored by introduced pair couplings, resulting hyperedge
/// rank, incident hyperedge count and then the semantic dependency index.
pub fn anf_hypergraph_min_fill_order(
    dependency_count: usize,
    hyperedges: &[Vec<usize>],
) -> Result<Vec<usize>, StrategyBddError> {
    let mut normalized = Vec::with_capacity(hyperedges.len());
    let mut seen_edges = BTreeSet::new();
    for edge in hyperedges {
        let mut sorted = edge.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() < 2 || sorted.iter().any(|&vertex| vertex >= dependency_count) {
            return Err(StrategyBddError::AnfHypergraphInvalidEdge {
                dependency_count,
                edge: edge.clone(),
            });
        }
        if seen_edges.insert(sorted.clone()) {
            normalized.push(sorted);
        }
    }

    let mut active = vec![true; dependency_count];
    let mut current_edges = normalized;
    let mut order = Vec::with_capacity(dependency_count);

    for _ in 0..dependency_count {
        let mut best: Option<(u128, usize, usize, usize)> = None;
        for variable in 0..dependency_count {
            if !active[variable] {
                continue;
            }
            let mut union = BTreeSet::new();
            let mut incident_count = 0usize;
            for edge in &current_edges {
                if edge.contains(&variable) {
                    incident_count = incident_count
                        .checked_add(1)
                        .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                    union.extend(edge.iter().copied().filter(|&vertex| vertex != variable));
                }
            }
            let neighbors: Vec<usize> = union.into_iter().collect();
            let possible_pairs = (neighbors.len() as u128)
                .checked_mul(neighbors.len().saturating_sub(1) as u128)
                .ok_or(StrategyBddError::ResourceAccountingOverflow)?
                / 2;
            let mut already_coupled = 0u128;
            for left in 0..neighbors.len() {
                for right in left + 1..neighbors.len() {
                    if current_edges.iter().any(|edge| {
                        edge.binary_search(&neighbors[left]).is_ok()
                            && edge.binary_search(&neighbors[right]).is_ok()
                    }) {
                        already_coupled = already_coupled
                            .checked_add(1)
                            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                    }
                }
            }
            let fill_pairs = possible_pairs.saturating_sub(already_coupled);
            let candidate = (fill_pairs, neighbors.len(), incident_count, variable);
            if best.is_none_or(|current| candidate < current) {
                best = Some(candidate);
            }
        }

        let Some((_, _, _, selected)) = best else {
            break;
        };
        let mut remaining_edges = Vec::new();
        let mut fill_union = BTreeSet::new();
        for edge in current_edges {
            if edge.contains(&selected) {
                fill_union.extend(edge.into_iter().filter(|&vertex| vertex != selected));
            } else {
                remaining_edges.push(edge);
            }
        }
        if fill_union.len() >= 2 {
            let fill_edge: Vec<usize> = fill_union.into_iter().collect();
            if !remaining_edges.contains(&fill_edge) {
                remaining_edges.push(fill_edge);
            }
        }
        remaining_edges.sort();
        remaining_edges.dedup();
        current_edges = remaining_edges;
        active[selected] = false;
        order.push(selected);
    }
    Ok(order)
}

/// Validate a symmetric, non-negative, zero-diagonal weighted interaction matrix.
fn validate_weighted_interaction_matrix(weights: &[Vec<u128>]) -> Result<(), StrategyBddError> {
    let variable_count = weights.len();
    for (vertex, row) in weights.iter().enumerate() {
        if row.len() != variable_count {
            return Err(StrategyBddError::WeightedInteractionGraphRowWidthMismatch {
                row: vertex,
                expected: variable_count,
                actual: row.len(),
            });
        }
        if row[vertex] != 0 {
            return Err(StrategyBddError::WeightedInteractionGraphSelfWeight { vertex });
        }
    }
    for vertex in 0..variable_count {
        for neighbor in vertex + 1..variable_count {
            if weights[vertex][neighbor] != weights[neighbor][vertex] {
                return Err(StrategyBddError::WeightedInteractionGraphAsymmetricWeight {
                    vertex,
                    neighbor,
                    forward: weights[vertex][neighbor],
                    reverse: weights[neighbor][vertex],
                });
            }
        }
    }
    Ok(())
}

/// Sum arithmetic mixed finite differences for every dependency pair.
///
/// This complements the GF(2) interaction graph. In particular, XOR functions
/// are affine over GF(2), so their GF(2) mixed derivatives vanish, while their
/// real-valued multilinear interactions can still provide a useful weighted
/// structural signal. The weights are ordering heuristics, never semantic
/// dependency evidence.
pub fn strategy_arithmetic_weighted_interaction_graph(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<StrategyBddWeightedInteractionReport, StrategyBddError> {
    validate_truth_table(function, budget)?;
    let dependency_count = function.dependencies.len();
    let values = strategy_table_values(function)?;
    let contexts_per_pair = checked_power_of_two(dependency_count.saturating_sub(2))?;
    let mut weights = vec![vec![0u128; dependency_count]; dependency_count];
    let mut contexts_checked = 0u128;
    let mut total_interaction_weight = 0u128;

    for first in 0..dependency_count {
        for second in first + 1..dependency_count {
            let mut pair_weight = 0u128;
            for context in 0..contexts_per_pair {
                contexts_checked = contexts_checked
                    .checked_add(1)
                    .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                if contexts_checked > budget.max_weighted_interaction_checks {
                    return Err(
                        StrategyBddError::WeightedInteractionGraphCheckBudgetExceeded {
                            required: contexts_checked,
                            limit: budget.max_weighted_interaction_checks,
                        },
                    );
                }

                let base = expand_context_mask(context as usize, first, second);
                let f00 = i16::from(values[base]);
                let f01 = i16::from(values[base | (1usize << second)]);
                let f10 = i16::from(values[base | (1usize << first)]);
                let f11 = i16::from(values[base | (1usize << first) | (1usize << second)]);
                let magnitude = (f00 - f01 - f10 + f11).abs() as u128;
                pair_weight = pair_weight
                    .checked_add(magnitude)
                    .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
            }
            weights[first][second] = pair_weight;
            weights[second][first] = pair_weight;
            total_interaction_weight = total_interaction_weight
                .checked_add(pair_weight)
                .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        }
    }

    validate_weighted_interaction_matrix(&weights)?;
    Ok(StrategyBddWeightedInteractionReport {
        weights,
        contexts_checked,
        total_interaction_weight,
    })
}

/// Project non-zero arithmetic pair weights into a simple interaction graph.
pub fn weighted_interaction_adjacency(
    weights: &[Vec<u128>],
) -> Result<Vec<BTreeSet<usize>>, StrategyBddError> {
    validate_weighted_interaction_matrix(weights)?;
    let mut graph = vec![BTreeSet::new(); weights.len()];
    for first in 0..weights.len() {
        for second in first + 1..weights.len() {
            if weights[first][second] > 0 {
                graph[first].insert(second);
                graph[second].insert(first);
            }
        }
    }
    Ok(graph)
}

/// Weighted min-fill using the arithmetic mixed-derivative matrix.
///
/// A candidate's primary cost is the product-weight of fill edges it would
/// introduce. Ties use fill-edge count, weighted degree, active degree and
/// variable index. The fill graph itself is updated using ordinary clique
/// completion, keeping the ordering deterministic.
pub fn weighted_interaction_min_fill_order(
    weights: &[Vec<u128>],
) -> Result<Vec<usize>, StrategyBddError> {
    validate_weighted_interaction_matrix(weights)?;
    let variable_count = weights.len();
    let mut adjacency = weighted_interaction_adjacency(weights)?;
    // Fill edges retain the induced coupling generated at their introduction.
    // This makes later weighted-degree/fill scores account for prior eliminations.
    let mut induced_weights = weights.to_vec();
    let mut active = vec![true; variable_count];
    let mut order = Vec::with_capacity(variable_count);

    for _ in 0..variable_count {
        let mut best: Option<(u128, usize, u128, usize, usize)> = None;
        for variable in 0..variable_count {
            if !active[variable] {
                continue;
            }
            let neighbors: Vec<usize> = adjacency[variable]
                .iter()
                .copied()
                .filter(|&neighbor| active[neighbor])
                .collect();
            let mut weighted_fill_cost = 0u128;
            let mut fill_edges = 0usize;
            let mut weighted_degree = 0u128;

            for &neighbor in &neighbors {
                weighted_degree = weighted_degree
                    .checked_add(induced_weights[variable][neighbor])
                    .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
            }
            for left in 0..neighbors.len() {
                for right in left + 1..neighbors.len() {
                    let a = neighbors[left];
                    let b = neighbors[right];
                    if !adjacency[a].contains(&b) {
                        fill_edges = fill_edges
                            .checked_add(1)
                            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                        let product = induced_weights[variable][a]
                            .checked_mul(induced_weights[variable][b])
                            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                        weighted_fill_cost = weighted_fill_cost
                            .checked_add(product)
                            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                    }
                }
            }

            let candidate = (
                weighted_fill_cost,
                fill_edges,
                weighted_degree,
                neighbors.len(),
                variable,
            );
            if best.is_none_or(|current| candidate < current) {
                best = Some(candidate);
            }
        }

        let Some((_, _, _, _, selected)) = best else {
            break;
        };
        let neighbors: Vec<usize> = adjacency[selected]
            .iter()
            .copied()
            .filter(|&neighbor| active[neighbor])
            .collect();
        for left in 0..neighbors.len() {
            for right in left + 1..neighbors.len() {
                let a = neighbors[left];
                let b = neighbors[right];
                if !adjacency[a].contains(&b) {
                    let induced_weight = induced_weights[selected][a]
                        .checked_mul(induced_weights[selected][b])
                        .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                    adjacency[a].insert(b);
                    adjacency[b].insert(a);
                    induced_weights[a][b] = induced_weight;
                    induced_weights[b][a] = induced_weight;
                }
            }
        }
        active[selected] = false;
        order.push(selected);
    }
    Ok(order)
}

/// Deterministic min-fill order for an interaction graph. Ties are broken by
/// active degree, then by the semantic dependency index.
pub fn interaction_graph_min_fill_order(
    graph: &[BTreeSet<usize>],
) -> Result<Vec<usize>, StrategyBddError> {
    validate_interaction_graph(graph)?;
    let variable_count = graph.len();
    let mut active = vec![true; variable_count];
    let mut adjacency = graph.to_vec();
    let mut order = Vec::with_capacity(variable_count);

    for _ in 0..variable_count {
        let mut best: Option<(usize, usize, usize)> = None;
        for variable in 0..variable_count {
            if !active[variable] {
                continue;
            }
            let neighbors: Vec<usize> = adjacency[variable]
                .iter()
                .copied()
                .filter(|&neighbor| active[neighbor])
                .collect();
            let mut fill_edges = 0usize;
            for left in 0..neighbors.len() {
                for right in left + 1..neighbors.len() {
                    if !adjacency[neighbors[left]].contains(&neighbors[right]) {
                        fill_edges = fill_edges.saturating_add(1);
                    }
                }
            }
            let candidate = (fill_edges, neighbors.len(), variable);
            if best.is_none_or(|current| candidate < current) {
                best = Some(candidate);
            }
        }

        let Some((_, _, selected)) = best else {
            break;
        };
        let neighbors: Vec<usize> = adjacency[selected]
            .iter()
            .copied()
            .filter(|&neighbor| active[neighbor])
            .collect();
        for left in 0..neighbors.len() {
            for right in left + 1..neighbors.len() {
                adjacency[neighbors[left]].insert(neighbors[right]);
                adjacency[neighbors[right]].insert(neighbors[left]);
            }
        }
        active[selected] = false;
        order.push(selected);
    }
    Ok(order)
}

/// Validate a simple, undirected interaction graph whose vertex IDs are
/// indices into the strategy function's dependency list.
fn validate_interaction_graph(graph: &[BTreeSet<usize>]) -> Result<(), StrategyBddError> {
    for (vertex, neighbors) in graph.iter().enumerate() {
        for &neighbor in neighbors {
            if neighbor >= graph.len() {
                return Err(StrategyBddError::InteractionGraphNodeOutOfRange {
                    vertex,
                    neighbor,
                    vertex_count: graph.len(),
                });
            }
            if neighbor == vertex {
                return Err(StrategyBddError::InteractionGraphSelfEdge { vertex });
            }
            if !graph[neighbor].contains(&vertex) {
                return Err(StrategyBddError::InteractionGraphAsymmetricEdge { vertex, neighbor });
            }
        }
    }
    Ok(())
}

/// Compute the live vertex-separation frontier after every non-final prefix
/// of an order. A processed vertex is live when it has an edge to an
/// unprocessed vertex.
pub fn interaction_graph_frontier_profile(
    graph: &[BTreeSet<usize>],
    order: &[usize],
) -> Result<Vec<usize>, StrategyBddError> {
    validate_interaction_graph(graph)?;
    validate_variable_order(graph.len(), order)?;
    let mut position = vec![usize::MAX; order.len()];
    for (rank, &vertex) in order.iter().enumerate() {
        position[vertex] = rank;
    }
    let mut profile = Vec::with_capacity(order.len().saturating_sub(1));
    for cut in 1..order.len() {
        let mut live = 0usize;
        for &vertex in &order[..cut] {
            if graph[vertex]
                .iter()
                .any(|&neighbor| position[neighbor] >= cut)
            {
                live = live
                    .checked_add(1)
                    .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
            }
        }
        profile.push(live);
    }
    Ok(profile)
}

fn frontier_score(
    graph: &[BTreeSet<usize>],
    order: &[usize],
) -> Result<(usize, usize, Vec<usize>), StrategyBddError> {
    let profile = interaction_graph_frontier_profile(graph, order)?;
    let maximum = profile.iter().copied().max().unwrap_or(0);
    let total = profile.iter().try_fold(0usize, |sum, width| {
        sum.checked_add(*width)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)
    })?;
    Ok((maximum, total, profile))
}

/// Refine a min-fill order by bounded variable insertion. Candidates are
/// accepted only when the lexicographic frontier score improves:
/// (maximum frontier, total frontier mass, profile). Therefore the maximum
/// frontier can never exceed that of the input seed.
pub fn refine_interaction_order_by_frontier(
    graph: &[BTreeSet<usize>],
    initial_order: &[usize],
    budget: StrategyBddBudget,
) -> Result<StrategyBddFrontierRefinementReport, StrategyBddError> {
    validate_interaction_graph(graph)?;
    validate_variable_order(graph.len(), initial_order)?;
    let initial_frontier_profile = interaction_graph_frontier_profile(graph, initial_order)?;
    let mut order = initial_order.to_vec();
    let mut best_score = frontier_score(graph, &order)?;
    let mut evaluations = 0usize;

    for _ in 0..budget.max_frontier_refinement_passes {
        let mut pass_improved = false;
        for vertex in 0..graph.len() {
            let Some(current_position) = order.iter().position(|candidate| *candidate == vertex)
            else {
                return Err(StrategyBddError::VariableOrderEntryOutOfRange {
                    position: vertex,
                    dependency_index: vertex,
                    dependency_count: graph.len(),
                });
            };
            let mut remainder = order.clone();
            remainder.remove(current_position);
            let mut best_order = order.clone();

            for insertion_position in 0..graph.len() {
                let mut candidate = remainder.clone();
                candidate.insert(insertion_position, vertex);
                if candidate == order {
                    continue;
                }
                let required = evaluations
                    .checked_add(1)
                    .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
                if required > budget.max_frontier_order_evaluations {
                    return Err(StrategyBddError::FrontierOrderSearchBudgetExceeded {
                        required,
                        limit: budget.max_frontier_order_evaluations,
                    });
                }
                evaluations = required;
                let candidate_score = frontier_score(graph, &candidate)?;
                if candidate_score < best_score {
                    best_score = candidate_score;
                    best_order = candidate;
                }
            }

            if best_order != order {
                order = best_order;
                pass_improved = true;
            }
        }
        if !pass_improved {
            break;
        }
    }

    let refined_frontier_profile = interaction_graph_frontier_profile(graph, &order)?;
    Ok(StrategyBddFrontierRefinementReport {
        initial_order: initial_order.to_vec(),
        refined_order: order,
        initial_frontier_profile,
        refined_frontier_profile,
        order_evaluations: evaluations,
    })
}

/// Run bounded sifting from natural, reverse, interaction min-fill and
/// frontier-refined starting orders.
/// starting orders. Each run receives only the remaining global order budget.
pub fn multi_start_sift_strategy_function_robdd(
    function: &BooleanStrategyFunction,
    budget: StrategyBddBudget,
) -> Result<StrategyBddMultiStartReport, StrategyBddError> {
    validate_truth_table(function, budget)?;
    if budget.max_variable_order_evaluations == 0 {
        return Err(StrategyBddError::VariableOrderSearchBudgetExceeded {
            required: 1,
            limit: 0,
        });
    }
    let dependency_count = function.dependencies.len();
    let (interaction_graph, interaction_pair_checks) =
        strategy_interaction_graph(function, budget)?;
    let min_fill = interaction_graph_min_fill_order(&interaction_graph)?;
    let frontier_report =
        refine_interaction_order_by_frontier(&interaction_graph, &min_fill, budget)?;

    let weighted_report = strategy_arithmetic_weighted_interaction_graph(function, budget)?;
    let weighted_graph = weighted_interaction_adjacency(&weighted_report.weights)?;
    let weighted_min_fill = weighted_interaction_min_fill_order(&weighted_report.weights)?;
    let weighted_frontier_report =
        refine_interaction_order_by_frontier(&weighted_graph, &weighted_min_fill, budget)?;

    let anf_report = strategy_anf_interaction_hypergraph(function, budget)?;
    let anf_min_fill = anf_hypergraph_min_fill_order(dependency_count, &anf_report.hyperedges)?;
    let weighted_min_fill_profile =
        interaction_graph_frontier_profile(&weighted_graph, &weighted_min_fill)?;
    let weighted_min_fill_max_frontier =
        weighted_min_fill_profile.iter().copied().max().unwrap_or(0);
    let weighted_refined_max_frontier = weighted_frontier_report
        .refined_frontier_profile
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let min_fill_profile = interaction_graph_frontier_profile(&interaction_graph, &min_fill)?;
    let min_fill_max_frontier = min_fill_profile.iter().copied().max().unwrap_or(0);
    let refined_max_frontier = frontier_report
        .refined_frontier_profile
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let frontier_refinement_evaluations = frontier_report.order_evaluations;
    let refined_order = frontier_report.refined_order;
    let mut candidates = vec![
        (
            StrategyBddOrderSeed::Natural,
            (0..dependency_count).collect::<Vec<_>>(),
        ),
        (
            StrategyBddOrderSeed::Reverse,
            (0..dependency_count).rev().collect::<Vec<_>>(),
        ),
        (StrategyBddOrderSeed::InteractionMinFill, min_fill.clone()),
        (
            StrategyBddOrderSeed::InteractionMinFillReverse,
            min_fill.into_iter().rev().collect::<Vec<_>>(),
        ),
        (
            StrategyBddOrderSeed::FrontierRefinedMinFill,
            refined_order.clone(),
        ),
        (
            StrategyBddOrderSeed::FrontierRefinedMinFillReverse,
            refined_order.into_iter().rev().collect::<Vec<_>>(),
        ),
        (
            StrategyBddOrderSeed::ArithmeticWeightedMinFill,
            weighted_min_fill.clone(),
        ),
        (
            StrategyBddOrderSeed::ArithmeticWeightedMinFillReverse,
            weighted_min_fill.into_iter().rev().collect::<Vec<_>>(),
        ),
        (
            StrategyBddOrderSeed::ArithmeticWeightedFrontierRefined,
            weighted_frontier_report.refined_order.clone(),
        ),
        (
            StrategyBddOrderSeed::ArithmeticWeightedFrontierRefinedReverse,
            weighted_frontier_report
                .refined_order
                .into_iter()
                .rev()
                .collect::<Vec<_>>(),
        ),
        (
            StrategyBddOrderSeed::AnfHypergraphMinFill,
            anf_min_fill.clone(),
        ),
        (
            StrategyBddOrderSeed::AnfHypergraphMinFillReverse,
            anf_min_fill.into_iter().rev().collect::<Vec<_>>(),
        ),
    ];

    let mut unique_orders: Vec<(StrategyBddOrderSeed, Vec<usize>)> = Vec::new();
    for (seed, order) in candidates.drain(..) {
        if !unique_orders.iter().any(|(_, previous)| *previous == order) {
            unique_orders.push((seed, order));
        }
    }

    let mut remaining_budget = budget.max_variable_order_evaluations;
    let mut order_evaluations = 0usize;
    let mut seeds_evaluated = 0usize;
    let mut search_complete = true;
    let mut best: Option<(StrategyBddOrderSeed, usize, StrategyBddSiftingReport)> = None;

    for (seed, order) in &unique_orders {
        if remaining_budget == 0 {
            search_complete = false;
            break;
        }
        let seed_budget = StrategyBddBudget {
            max_variable_order_evaluations: remaining_budget,
            ..budget
        };
        let report = sift_strategy_function_robdd_from_order(function, order, seed_budget)?;
        seeds_evaluated = seeds_evaluated
            .checked_add(1)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        order_evaluations = order_evaluations
            .checked_add(report.order_evaluations)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;
        remaining_budget = budget
            .max_variable_order_evaluations
            .checked_sub(order_evaluations)
            .ok_or(StrategyBddError::ResourceAccountingOverflow)?;

        let seed_search_complete = report.search_complete;
        let replace = best.as_ref().is_none_or(|(best_seed, _, best_report)| {
            report.optimized_node_count < best_report.optimized_node_count
                || (report.optimized_node_count == best_report.optimized_node_count
                    && (report.optimized_variable_order < best_report.optimized_variable_order
                        || (report.optimized_variable_order
                            == best_report.optimized_variable_order
                            && *seed < *best_seed)))
        });
        if replace {
            best = Some((*seed, report.initial_node_count, report));
        }
        if !seed_search_complete {
            search_complete = false;
            break;
        }
    }

    if seeds_evaluated < unique_orders.len() {
        search_complete = false;
    }
    let Some((selected_seed, selected_initial_node_count, selected_report)) = best else {
        return Err(StrategyBddError::ResourceAccountingOverflow);
    };

    Ok(StrategyBddMultiStartReport {
        seed_orders: unique_orders,
        selected_seed,
        selected_initial_node_count,
        optimized_node_count: selected_report.optimized_node_count,
        interaction_pair_checks,
        arithmetic_interaction_context_checks: weighted_report.contexts_checked,
        arithmetic_interaction_total_weight: weighted_report.total_interaction_weight,
        anf_hyperedge_count: anf_report.hyperedges.len(),
        anf_transform_checks: anf_report.transform_checks,
        anf_max_hyperedge_rank: anf_report.max_hyperedge_rank,
        weighted_min_fill_max_frontier,
        weighted_refined_max_frontier,
        frontier_refinement_evaluations,
        min_fill_max_frontier,
        refined_max_frontier,
        order_evaluations,
        seeds_evaluated,
        search_complete,
        bdd: selected_report.bdd,
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

fn exact_order_space_fits(dependency_count: usize, evaluation_budget: usize) -> bool {
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
        let (
            optimization_method,
            order_seed,
            interaction_pair_checks,
            arithmetic_interaction_context_checks,
            arithmetic_interaction_total_weight,
            weighted_min_fill_max_frontier,
            weighted_refined_max_frontier,
            frontier_refinement_evaluations,
            min_fill_max_frontier,
            refined_max_frontier,
            order_evaluations,
            optimization_search_complete,
            bdd,
        ) = if exact_order_space_fits(
            support.function.dependencies.len(),
            budget.max_variable_order_evaluations,
        ) {
            let report = find_exact_minimum_robdd_order(&support.function, budget)?;
            (
                StrategyBddOptimizationMethod::ExactOrderSearch,
                None,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                report.orders_evaluated,
                true,
                report.bdd,
            )
        } else {
            let report = multi_start_sift_strategy_function_robdd(&support.function, budget)?;
            (
                StrategyBddOptimizationMethod::MultiStartSifting,
                Some(report.selected_seed),
                report.interaction_pair_checks,
                report.arithmetic_interaction_context_checks,
                report.arithmetic_interaction_total_weight,
                report.weighted_min_fill_max_frontier,
                report.weighted_refined_max_frontier,
                report.frontier_refinement_evaluations,
                report.min_fill_max_frontier,
                report.refined_max_frontier,
                report.order_evaluations,
                report.search_complete,
                report.bdd,
            )
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
            order_seed,
            interaction_pair_checks,
            arithmetic_interaction_context_checks,
            arithmetic_interaction_total_weight,
            weighted_min_fill_max_frontier,
            weighted_refined_max_frontier,
            frontier_refinement_evaluations,
            min_fill_max_frontier,
            refined_max_frontier,
            order_evaluations,
            optimization_search_complete,
            bdd,
        });
    }

    let optimization_search_complete = functions
        .iter()
        .all(|function| function.optimization_search_complete);
    Ok(CompiledWinningStrategy {
        owner: strategy.owner,
        functions,
        original_dependency_count,
        reduced_dependency_count,
        original_truth_table_entries,
        reduced_truth_table_entries,
        total_bdd_nodes,
        total_order_evaluations,
        optimization_search_complete,
    })
}

/// Verify a candidate QBF result against the original matrix/prefix and only
/// then compile its winning functions into support-minimized ROBDDs.
///
/// Any producer-supplied verification report inside QbfEvaluationResult is
/// ignored and recomputed here. The compiled BDD is an additional representation
/// certificate; it never substitutes for source-matrix strategy verification.
pub fn verify_and_compile_qbf_strategy(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    candidate: &QbfEvaluationResult,
    qbf_budget: QuantifiedEvaluationBudget,
    bdd_budget: StrategyBddBudget,
) -> Result<VerifiedCompiledQbfStrategy, VerifiedStrategyCompilationError> {
    let expected_owner = if candidate.value {
        StrategyOwner::Existential
    } else {
        StrategyOwner::Universal
    };
    if candidate.strategy.owner != expected_owner {
        return Err(VerifiedStrategyCompilationError::ResultOwnerMismatch {
            value: candidate.value,
            owner: candidate.strategy.owner,
        });
    }

    let source_strategy_verification =
        verify_winning_strategy(problem, prefix, &candidate.strategy, qbf_budget)
            .map_err(VerifiedStrategyCompilationError::SourceStrategyInvalid)?;
    let representation = compile_winning_strategy_functions(&candidate.strategy, bdd_budget)
        .map_err(VerifiedStrategyCompilationError::RepresentationInvalid)?;

    Ok(VerifiedCompiledQbfStrategy {
        value: candidate.value,
        source_strategy_verification,
        representation,
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
    evaluate_validated_robdd(bdd, assignment)
}

/// Evaluate a diagram after its structure and assignment width have already
/// been validated. Kept separate so exhaustive equivalence checking validates
/// the untrusted graph once, rather than re-walking every node for every row.
fn evaluate_validated_robdd(
    bdd: &ReducedOrderedBdd,
    assignment: &[bool],
) -> Result<bool, StrategyBddError> {
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
        let bdd_value = evaluate_validated_robdd(bdd, &assignment)?;
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
            max_interaction_checks: 1 << 12,
            max_weighted_interaction_checks: 1 << 12,
            max_anf_transform_checks: 1 << 12,
            max_frontier_order_evaluations: 1 << 12,
            max_frontier_refinement_passes: 2,
            max_variable_order_evaluations: 1 << 13,
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
    fn variable_order_sifting_preserves_verified_incumbent_when_budget_is_exhausted() {
        let source = function(
            &[11, 12, 10],
            &[false, true, false, true, false, false, true, true],
        );
        let tight = StrategyBddBudget {
            max_variable_order_evaluations: 1,
            ..budget()
        };
        let baseline = compile_strategy_function_to_robdd(&source, tight).unwrap();
        let report = sift_strategy_function_robdd(&source, tight).unwrap();
        assert!(!report.search_complete);
        assert_eq!(report.order_evaluations, 1);
        assert_eq!(report.bdd, baseline);
        verify_robdd_matches_strategy_function(&report.bdd, &source, tight).unwrap();
    }

    #[test]
    fn zero_sifting_budget_fails_before_building_an_incumbent() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let tight = StrategyBddBudget {
            max_variable_order_evaluations: 0,
            ..budget()
        };
        assert!(matches!(
            sift_strategy_function_robdd(&source, tight),
            Err(StrategyBddError::VariableOrderSearchBudgetExceeded {
                required: 1,
                limit: 0
            })
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
        assert_eq!(
            maximum_sift_gap, 0,
            "three-variable sifting should match exact order search"
        );
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
        let values: Vec<bool> = (0usize..64)
            .map(|mask| mask.count_ones() % 2 == 1)
            .collect();
        let source = function(&[0, 1, 2, 3, 4, 5], &values);
        let strategy = WinningStrategy {
            owner: StrategyOwner::Universal,
            functions: vec![source],
        };
        let constrained = StrategyBddBudget {
            max_variable_order_evaluations: 512,
            ..budget()
        };
        let compiled = compile_winning_strategy_functions(&strategy, constrained).unwrap();
        assert_eq!(
            compiled.functions[0].optimization_method,
            StrategyBddOptimizationMethod::MultiStartSifting
        );
        assert_eq!(compiled.functions[0].reduced_function.dependencies.len(), 6);
        assert!(compiled.total_bdd_nodes > 0);
        assert!(compiled.total_order_evaluations <= 128);
    }

    fn equality_qbf_problem_and_prefix() -> (CnfProblem, ValidatedQuantifierPrefix) {
        let problem = CnfProblem {
            variables: vec!["u".into(), "x".into()],
            clauses: vec![
                vec![Literal::negative(0), Literal::positive(1)],
                vec![Literal::positive(0), Literal::negative(1)],
            ],
        };
        let prefix = ValidatedQuantifierPrefix::new(
            2,
            vec![
                QuantifierBlock {
                    quantifier: Quantifier::ForAll,
                    variables: vec![0],
                },
                QuantifierBlock {
                    quantifier: Quantifier::Exists,
                    variables: vec![1],
                },
            ],
            Default::default(),
        )
        .unwrap();
        (problem, prefix)
    }

    #[test]
    fn end_to_end_compiler_rechecks_source_strategy_and_compiles_verified_bdd() {
        let (problem, prefix) = equality_qbf_problem_and_prefix();
        let candidate = super::super::quantified_strategy::evaluate_qbf_with_strategy(
            &problem,
            &prefix,
            QuantifiedEvaluationBudget::default(),
        )
        .unwrap();
        let mut untrusted = candidate.clone();
        // A producer-supplied report is not authoritative and is recomputed.
        untrusted.verification.opponent_assignments_checked = 0;

        let verified = verify_and_compile_qbf_strategy(
            &problem,
            &prefix,
            &untrusted,
            QuantifiedEvaluationBudget::default(),
            budget(),
        )
        .unwrap();

        assert!(verified.value);
        assert_eq!(
            verified
                .source_strategy_verification
                .opponent_assignments_checked,
            2
        );
        assert_eq!(verified.representation.owner, StrategyOwner::Existential);
        assert_eq!(verified.representation.functions.len(), 1);
        assert_eq!(verified.representation.total_bdd_nodes, 1);
        assert_eq!(
            verified.representation.functions[0].optimization_method,
            StrategyBddOptimizationMethod::ExactOrderSearch
        );
    }

    #[test]
    fn end_to_end_compiler_rejects_tampered_source_strategy_before_bdd_compile() {
        let (problem, prefix) = equality_qbf_problem_and_prefix();
        let mut candidate = super::super::quantified_strategy::evaluate_qbf_with_strategy(
            &problem,
            &prefix,
            QuantifiedEvaluationBudget::default(),
        )
        .unwrap();
        let function = candidate
            .strategy
            .functions
            .iter_mut()
            .find(|function| function.variable == 1)
            .unwrap();
        function.table.insert(vec![false], true);
        function.table.insert(vec![true], false);

        assert!(matches!(
            verify_and_compile_qbf_strategy(
                &problem,
                &prefix,
                &candidate,
                QuantifiedEvaluationBudget::default(),
                budget()
            ),
            Err(VerifiedStrategyCompilationError::SourceStrategyInvalid(
                QuantifiedStrategyError::Counterexample { .. }
            ))
        ));
    }

    #[test]
    fn end_to_end_compiler_rejects_truth_value_owner_mismatch() {
        let (problem, prefix) = equality_qbf_problem_and_prefix();
        let mut candidate = super::super::quantified_strategy::evaluate_qbf_with_strategy(
            &problem,
            &prefix,
            QuantifiedEvaluationBudget::default(),
        )
        .unwrap();
        candidate.strategy.owner = StrategyOwner::Universal;
        assert!(matches!(
            verify_and_compile_qbf_strategy(
                &problem,
                &prefix,
                &candidate,
                QuantifiedEvaluationBudget::default(),
                budget()
            ),
            Err(VerifiedStrategyCompilationError::ResultOwnerMismatch {
                value: true,
                owner: StrategyOwner::Universal
            })
        ));
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
    fn anf_hypergraph_recovers_high_order_xor_monomial() {
        // f = x0 XOR x1 XOR x2 XOR (x0*x1*x2). The cubic term must
        // survive as a rank-three hyperedge, not be flattened away.
        let source = function(
            &[10, 11, 12],
            &[false, true, true, false, true, false, false, false],
        );
        let report = strategy_anf_interaction_hypergraph(&source, budget()).unwrap();
        assert!(report.hyperedges.contains(&vec![0, 1, 2]));
        assert_eq!(report.max_hyperedge_rank, 3);
        assert_eq!(report.linear_variables, vec![0, 1, 2]);
        assert_eq!(report.transform_checks, 12);
    }

    #[test]
    fn anf_transform_reconstructs_every_three_input_boolean_function() {
        for function_mask in 0..256usize {
            let values: Vec<bool> = (0..8)
                .map(|assignment| ((function_mask >> assignment) & 1) != 0)
                .collect();
            let source = function(&[0, 1, 2], &values);
            let report = strategy_anf_interaction_hypergraph(&source, budget()).unwrap();

            for (assignment, expected) in values.iter().copied().enumerate() {
                let mut reconstructed = report.constant_term;
                for &variable in &report.linear_variables {
                    reconstructed ^= ((assignment >> variable) & 1) != 0;
                }
                for monomial in &report.hyperedges {
                    if monomial
                        .iter()
                        .all(|&variable| ((assignment >> variable) & 1) != 0)
                    {
                        reconstructed = !reconstructed;
                    }
                }
                assert_eq!(
                    reconstructed, expected,
                    "function mask {function_mask}, assignment {assignment}"
                );
            }

            let order = anf_hypergraph_min_fill_order(3, &report.hyperedges).unwrap();
            assert_eq!(
                order.iter().copied().collect::<BTreeSet<_>>(),
                BTreeSet::from([0, 1, 2])
            );
        }
    }

    #[test]
    fn anf_hypergraph_min_fill_is_deterministic_and_covers_all_variables() {
        let edges = vec![vec![0, 1, 2], vec![2, 3], vec![3, 4, 5]];
        let first = anf_hypergraph_min_fill_order(6, &edges).unwrap();
        let second = anf_hypergraph_min_fill_order(6, &edges).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 6);
        assert_eq!(
            first.iter().copied().collect::<BTreeSet<_>>(),
            (0..6).collect()
        );
    }

    #[test]
    fn anf_transform_budget_is_fail_closed() {
        let source = function(
            &[0, 1, 2],
            &[false, true, true, false, true, false, false, true],
        );
        let tight = StrategyBddBudget {
            max_anf_transform_checks: 11,
            ..budget()
        };
        assert!(matches!(
            strategy_anf_interaction_hypergraph(&source, tight),
            Err(StrategyBddError::AnfTransformCheckBudgetExceeded {
                required: 12,
                limit: 11
            })
        ));
    }

    #[test]
    fn interaction_graph_detects_mixed_boolean_interaction() {
        let source = function(
            &[0, 1, 2],
            &[false, false, false, true, false, true, false, true],
        );
        let (graph, checks) = strategy_interaction_graph(&source, budget()).unwrap();
        assert!(graph[0].contains(&1));
        assert!(graph[0].contains(&2));
        assert!(graph[1].contains(&2));
        assert!(checks > 0);
    }

    #[test]
    fn interaction_graph_seed_order_is_deterministic() {
        let source = function(&[0, 1, 2, 3], &[false; 16]);
        let (graph, _) = strategy_interaction_graph(&source, budget()).unwrap();
        assert_eq!(
            interaction_graph_min_fill_order(&graph).unwrap(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn interaction_graph_budget_is_fail_closed() {
        let source = function(
            &[0, 1, 2],
            &[false, false, false, true, false, true, false, true],
        );
        let tight = StrategyBddBudget {
            max_interaction_checks: 0,
            ..budget()
        };
        assert!(matches!(
            strategy_interaction_graph(&source, tight),
            Err(StrategyBddError::InteractionGraphCheckBudgetExceeded { .. })
        ));
    }

    #[test]
    fn multi_start_includes_algebraic_hypergraph_seeds() {
        let source = function(
            &[0, 1, 2, 3],
            &[
                false, false, false, false, false, false, false, true, false, false, false, false,
                false, false, false, true,
            ],
        );
        let report = multi_start_sift_strategy_function_robdd(&source, budget()).unwrap();
        assert!(report.anf_transform_checks > 0);
        assert!(report.anf_hyperedge_count > 0);
        let anf = strategy_anf_interaction_hypergraph(&source, budget()).unwrap();
        let anf_order = anf_hypergraph_min_fill_order(4, &anf.hyperedges).unwrap();
        let anf_reverse: Vec<usize> = anf_order.iter().copied().rev().collect();
        assert!(
            report
                .seed_orders
                .iter()
                .any(|(_, order)| *order == anf_order || *order == anf_reverse),
            "the ANF-derived order or its reverse must be evaluated; identical orders are deduplicated"
        );
        verify_robdd_matches_strategy_function(&report.bdd, &source, budget()).unwrap();
    }

    #[test]
    fn multi_start_sifting_preserves_function_and_respects_global_budget() {
        let source = function(
            &[11, 12, 10],
            &[false, true, false, true, false, false, true, true],
        );
        let report = multi_start_sift_strategy_function_robdd(&source, budget()).unwrap();
        assert!(report.order_evaluations <= budget().max_variable_order_evaluations);
        assert!(!report.seed_orders.is_empty());
        assert!(report.arithmetic_interaction_context_checks > 0);
        assert!(report.arithmetic_interaction_total_weight > 0);
        assert!(report.optimized_node_count <= report.selected_initial_node_count);
        let compiled = compile_strategy_function_to_robdd_with_order(
            &source,
            &report.bdd.variable_order,
            budget(),
        )
        .unwrap();
        assert_eq!(compiled, report.bdd);
        verify_robdd_matches_strategy_function(&report.bdd, &source, budget()).unwrap();
    }

    #[test]
    fn arithmetic_weighted_graph_detects_xor_coupling_missed_by_gf2_graph() {
        // x XOR y XOR z is affine over GF(2): all second GF(2) mixed
        // derivatives vanish. Its arithmetic mixed derivatives are non-zero.
        let source = function(
            &[0, 1, 2],
            &[false, true, true, false, true, false, false, true],
        );
        let (gf2_graph, _) = strategy_interaction_graph(&source, budget()).unwrap();
        assert!(gf2_graph.iter().all(BTreeSet::is_empty));

        let report = strategy_arithmetic_weighted_interaction_graph(&source, budget()).unwrap();
        assert_eq!(report.contexts_checked, 6);
        assert_eq!(report.weights[0][1], 4);
        assert_eq!(report.weights[0][2], 4);
        assert_eq!(report.weights[1][2], 4);
        assert_eq!(report.total_interaction_weight, 12);
        let order = weighted_interaction_min_fill_order(&report.weights).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn arithmetic_weighted_graph_matches_independent_integer_derivative_oracle() {
        for function_mask in 0..256usize {
            let values: Vec<bool> = (0..8)
                .map(|assignment| ((function_mask >> assignment) & 1) != 0)
                .collect();
            let source = function(&[0, 1, 2], &values);
            let report = strategy_arithmetic_weighted_interaction_graph(&source, budget()).unwrap();

            for first in 0..3 {
                for second in first + 1..3 {
                    let other = (0..3)
                        .find(|&variable| variable != first && variable != second)
                        .unwrap();
                    let mut expected = 0u128;
                    for context in 0..2usize {
                        let base = if context == 0 { 0 } else { 1usize << other };
                        let f00 = i16::from(values[base]);
                        let f01 = i16::from(values[base | (1 << second)]);
                        let f10 = i16::from(values[base | (1 << first)]);
                        let f11 = i16::from(values[base | (1 << first) | (1 << second)]);
                        expected += (f00 - f01 - f10 + f11).abs() as u128;
                    }
                    assert_eq!(
                        report.weights[first][second], expected,
                        "function mask {function_mask}, pair ({first}, {second})"
                    );
                }
            }
        }
    }

    #[test]
    fn weighted_min_fill_reduces_an_observed_four_input_robdd() {
        let source = function(
            &[0, 1, 2, 3],
            &[
                true, false, false, true, true, false, false, true, true, false, true, false, true,
                false, false, true,
            ],
        );
        let natural_order = vec![0, 1, 2, 3];
        let natural =
            compile_strategy_function_to_robdd_with_order(&source, &natural_order, budget())
                .unwrap();

        let (gf2_graph, _) = strategy_interaction_graph(&source, budget()).unwrap();
        let unweighted_order = interaction_graph_min_fill_order(&gf2_graph).unwrap();
        let weighted = strategy_arithmetic_weighted_interaction_graph(&source, budget()).unwrap();
        let weighted_order = weighted_interaction_min_fill_order(&weighted.weights).unwrap();
        let weighted_bdd =
            compile_strategy_function_to_robdd_with_order(&source, &weighted_order, budget())
                .unwrap();
        let exact = find_exact_minimum_robdd_order(&source, budget()).unwrap();

        assert_eq!(unweighted_order, natural_order);
        assert_eq!(weighted_order, vec![2, 3, 0, 1]);
        assert_eq!(natural.nodes.len(), 7);
        assert_eq!(weighted_bdd.nodes.len(), 6);
        assert_eq!(exact.minimum_node_count, 5);
        verify_robdd_matches_strategy_function(&weighted_bdd, &source, budget()).unwrap();

        let report = multi_start_sift_strategy_function_robdd(&source, budget()).unwrap();
        assert!(report.seed_orders.contains(&(
            StrategyBddOrderSeed::ArithmeticWeightedMinFill,
            weighted_order,
        )));
        assert_eq!(report.optimized_node_count, exact.minimum_node_count);
        verify_robdd_matches_strategy_function(&report.bdd, &source, budget()).unwrap();
    }

    #[test]
    fn weighted_order_adds_a_distinct_seed_to_multistart_search() {
        let source = function(
            &[0, 1, 2, 3],
            &[
                true, true, false, true, true, true, true, true, true, false, false, true, false,
                false, true, false,
            ],
        );
        let weighted = strategy_arithmetic_weighted_interaction_graph(&source, budget()).unwrap();
        assert_eq!(weighted.contexts_checked, 24);
        assert_eq!(weighted.weights[0][1], 4);
        assert_eq!(weighted.weights[0][2], 4);
        assert_eq!(weighted.weights[0][3], 2);
        assert_eq!(weighted.weights[1][2], 4);
        assert_eq!(weighted.weights[1][3], 2);
        assert_eq!(weighted.weights[2][3], 2);
        assert_eq!(weighted.total_interaction_weight, 18);
        assert_eq!(
            weighted_interaction_min_fill_order(&weighted.weights).unwrap(),
            vec![3, 0, 1, 2]
        );

        let report = multi_start_sift_strategy_function_robdd(&source, budget()).unwrap();
        assert!(report.seed_orders.contains(&(
            StrategyBddOrderSeed::ArithmeticWeightedMinFill,
            vec![3, 0, 1, 2],
        )));
        assert!(report.order_evaluations <= budget().max_variable_order_evaluations);
        verify_robdd_matches_strategy_function(&report.bdd, &source, budget()).unwrap();
    }

    #[test]
    fn weighted_interaction_budget_is_fail_closed() {
        let source = function(
            &[0, 1, 2],
            &[false, true, true, false, true, false, false, true],
        );
        let tight = StrategyBddBudget {
            max_weighted_interaction_checks: 0,
            ..budget()
        };
        assert!(matches!(
            strategy_arithmetic_weighted_interaction_graph(&source, tight),
            Err(
                StrategyBddError::WeightedInteractionGraphCheckBudgetExceeded {
                    required: 1,
                    limit: 0
                }
            )
        ));
    }

    #[test]
    fn weighted_interaction_validation_rejects_asymmetry_and_nonzero_diagonal() {
        assert!(matches!(
            weighted_interaction_min_fill_order(&vec![vec![0, 1], vec![0, 0]]),
            Err(StrategyBddError::WeightedInteractionGraphAsymmetricWeight { .. })
        ));
        assert_eq!(
            weighted_interaction_min_fill_order(&vec![vec![1]]).unwrap_err(),
            StrategyBddError::WeightedInteractionGraphSelfWeight { vertex: 0 }
        );
        assert!(matches!(
            weighted_interaction_min_fill_order(&vec![vec![0, 1], vec![1]]),
            Err(StrategyBddError::WeightedInteractionGraphRowWidthMismatch { .. })
        ));
    }

    #[test]
    fn min_fill_rejects_out_of_range_graph_neighbor() {
        let graph = vec![BTreeSet::from([1])];
        assert_eq!(
            interaction_graph_min_fill_order(&graph).unwrap_err(),
            StrategyBddError::InteractionGraphNodeOutOfRange {
                vertex: 0,
                neighbor: 1,
                vertex_count: 1
            }
        );
    }

    #[test]
    fn min_fill_rejects_self_edges() {
        let graph = vec![BTreeSet::from([0])];
        assert_eq!(
            interaction_graph_min_fill_order(&graph).unwrap_err(),
            StrategyBddError::InteractionGraphSelfEdge { vertex: 0 }
        );
    }

    #[test]
    fn min_fill_rejects_asymmetric_edges() {
        let graph = vec![BTreeSet::from([1]), BTreeSet::new()];
        assert_eq!(
            interaction_graph_min_fill_order(&graph).unwrap_err(),
            StrategyBddError::InteractionGraphAsymmetricEdge {
                vertex: 0,
                neighbor: 1
            }
        );
    }

    #[test]
    fn multi_start_frontier_sifting_matches_exact_minimum_for_all_three_input_functions() {
        for function_mask in 0..256usize {
            let values: Vec<bool> = (0..8)
                .map(|assignment| ((function_mask >> assignment) & 1) != 0)
                .collect();
            let source = function(&[0, 1, 2], &values);
            let support = minimize_strategy_function_support(&source, budget()).unwrap();
            let exact = find_exact_minimum_robdd_order(&support.function, budget()).unwrap();
            let multi_start =
                multi_start_sift_strategy_function_robdd(&support.function, budget()).unwrap();

            assert_eq!(
                multi_start.optimized_node_count, exact.minimum_node_count,
                "function truth-table mask {function_mask}"
            );
            assert!(multi_start.order_evaluations <= budget().max_variable_order_evaluations);
            assert!(
                multi_start.refined_max_frontier <= multi_start.min_fill_max_frontier,
                "function truth-table mask {function_mask}"
            );
            verify_robdd_matches_strategy_function(&multi_start.bdd, &support.function, budget())
                .unwrap();
        }
    }

    #[test]
    fn frontier_refinement_reduces_vertex_separation_without_worsening_maximum() {
        let graph = vec![
            BTreeSet::from([3]),
            BTreeSet::from([2, 3]),
            BTreeSet::from([1, 3]),
            BTreeSet::from([0, 1, 2]),
            BTreeSet::new(),
        ];
        let min_fill = interaction_graph_min_fill_order(&graph).unwrap();
        assert_eq!(min_fill, vec![4, 0, 1, 2, 3]);
        let report = refine_interaction_order_by_frontier(&graph, &min_fill, budget()).unwrap();
        assert_eq!(report.initial_frontier_profile, vec![0, 1, 2, 3]);
        assert_eq!(report.refined_frontier_profile, vec![0, 1, 2, 1]);
        assert!(
            report.refined_frontier_profile.iter().max()
                <= report.initial_frontier_profile.iter().max()
        );
        assert!(report.order_evaluations > 0);
    }

    #[test]
    fn frontier_refinement_budget_is_fail_closed() {
        let graph = vec![
            BTreeSet::from([3]),
            BTreeSet::from([2, 3]),
            BTreeSet::from([1, 3]),
            BTreeSet::from([0, 1, 2]),
            BTreeSet::new(),
        ];
        let min_fill = interaction_graph_min_fill_order(&graph).unwrap();
        let tight = StrategyBddBudget {
            max_frontier_order_evaluations: 0,
            ..budget()
        };
        assert!(matches!(
            refine_interaction_order_by_frontier(&graph, &min_fill, tight),
            Err(StrategyBddError::FrontierOrderSearchBudgetExceeded { .. })
        ));
    }

    #[test]
    fn frontier_profile_rejects_malformed_interaction_graph() {
        let graph = vec![BTreeSet::from([1]), BTreeSet::new()];
        assert!(matches!(
            interaction_graph_frontier_profile(&graph, &[0, 1]),
            Err(StrategyBddError::InteractionGraphAsymmetricEdge { .. })
        ));
    }

    #[test]
    fn interaction_graph_matches_independent_mixed_derivative_oracle_for_all_three_variable_functions()
     {
        for function_mask in 0..256usize {
            let values: Vec<bool> = (0..8)
                .map(|assignment| ((function_mask >> assignment) & 1) != 0)
                .collect();
            let source = function(&[0, 1, 2], &values);
            let (graph, _) = strategy_interaction_graph(&source, budget()).unwrap();

            for first in 0..3 {
                for second in first + 1..3 {
                    let mut expected = false;
                    for context in 0..2 {
                        let mut base = [false; 3];
                        base[3 - first - second] = context != 0;
                        let f00 = base;
                        let mut f01 = base;
                        let mut f10 = base;
                        let mut f11 = base;
                        f01[second] = true;
                        f10[first] = true;
                        f11[first] = true;
                        f11[second] = true;
                        let read = |key: [bool; 3]| *source.table.get(key.as_slice()).unwrap();
                        if read(f00) ^ read(f01) ^ read(f10) ^ read(f11) {
                            expected = true;
                        }
                    }
                    assert_eq!(
                        graph[first].contains(&second),
                        expected,
                        "function mask {function_mask}, pair ({first}, {second})"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_global_order_budget_fails_before_interaction_graph_work() {
        let source = function(&[0, 1], &[false, true, true, false]);
        let tight = StrategyBddBudget {
            max_interaction_checks: 0,
            max_variable_order_evaluations: 0,
            ..budget()
        };
        assert_eq!(
            multi_start_sift_strategy_function_robdd(&source, tight).unwrap_err(),
            StrategyBddError::VariableOrderSearchBudgetExceeded {
                required: 1,
                limit: 0
            }
        );
    }

    #[test]
    fn multi_start_preserves_best_incumbent_when_global_budget_exhausts() {
        let source = function(
            &[11, 12, 10],
            &[false, true, false, true, false, false, true, true],
        );
        let tight = StrategyBddBudget {
            max_variable_order_evaluations: 1,
            ..budget()
        };
        let report = multi_start_sift_strategy_function_robdd(&source, tight).unwrap();
        assert!(!report.search_complete);
        assert_eq!(report.seeds_evaluated, 1);
        assert_eq!(report.order_evaluations, 1);
        verify_robdd_matches_strategy_function(&report.bdd, &source, tight).unwrap();
    }

    #[test]
    fn high_level_compiler_exposes_incomplete_order_search() {
        let values: Vec<bool> = (0usize..64)
            .map(|mask| mask.count_ones() % 2 == 1)
            .collect();
        let strategy = WinningStrategy {
            owner: StrategyOwner::Existential,
            functions: vec![function(&[0, 1, 2, 3, 4, 5], &values)],
        };
        let tight = StrategyBddBudget {
            max_variable_order_evaluations: 1,
            ..budget()
        };
        let compiled = compile_winning_strategy_functions(&strategy, tight).unwrap();
        assert!(!compiled.optimization_search_complete);
        assert!(!compiled.functions[0].optimization_search_complete);
        verify_robdd_matches_strategy_function(
            &compiled.functions[0].bdd,
            &compiled.functions[0].reduced_function,
            tight,
        )
        .unwrap();
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
