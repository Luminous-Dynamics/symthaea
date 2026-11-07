// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded finite semantic oracles for proof-carrying decomposition.
//!
//! This module is a finite reference checker, not a scalable SAT/#SAT solver
//! and not a proof of a generic decomposition theorem. It independently
//! enumerates a bounded CNF source problem and checks three aggregation laws:
//! Boolean cofactor branching, variable-disjoint component products, and
//! existential projected support.
//!
//! Any incomplete or over-budget enumeration is a typed error. It is never
//! converted into zero models, UNSAT, infeasibility, or search completion.

use std::collections::{BTreeSet, HashSet};

/// One literal in a CNF clause. The variable indexes the source universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Literal {
    pub variable: usize,
    pub positive: bool,
}

impl Literal {
    pub const fn positive(variable: usize) -> Self {
        Self {
            variable,
            positive: true,
        }
    }

    pub const fn negative(variable: usize) -> Self {
        Self {
            variable,
            positive: false,
        }
    }

    fn satisfied_by(self, assignment: &[bool]) -> bool {
        assignment[self.variable] == self.positive
    }
}

/// A finite CNF problem with an explicit variable universe.
///
/// Variables absent from all clauses remain in the universe and are free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CnfProblem {
    pub variables: Vec<String>,
    pub clauses: Vec<Vec<Literal>>,
}

/// Deterministic resource limits for exhaustive reference enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnumerationBudget {
    pub max_variables: usize,
    pub max_total_variable_name_bytes: usize,
    pub max_clauses: usize,
    pub max_literals: usize,
    pub max_total_assignment_evaluations: u128,
    pub max_projection_support: usize,
    pub max_projection_fragments: usize,
    pub max_projection_support_entries: usize,
}

impl Default for EnumerationBudget {
    fn default() -> Self {
        Self {
            max_variables: 20,
            max_total_variable_name_bytes: 64 * 1024,
            max_clauses: 1 << 16,
            max_literals: 1 << 20,
            max_total_assignment_evaluations: 1 << 20,
            max_projection_support: 1 << 20,
            max_projection_fragments: 1 << 16,
            max_projection_support_entries: 1 << 20,
        }
    }
}

/// Typed failure. A failed finite check never returns a partial count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionError {
    EmptyVariableName {
        index: usize,
    },
    DuplicateVariableName(String),
    LiteralVariableOutOfRange {
        clause: usize,
        literal: usize,
        variable: usize,
        variable_count: usize,
    },
    SplitVariableOutOfRange {
        variable: usize,
        variable_count: usize,
    },
    AssignmentSpaceOverflow {
        variable_count: usize,
    },
    VariableBudgetExceeded {
        required: usize,
        limit: usize,
    },
    VariableNameBudgetExceeded {
        required: usize,
        limit: usize,
    },
    AssignmentBudgetExceeded {
        required: u128,
        limit: u128,
    },
    ClauseBudgetExceeded {
        required: usize,
        limit: usize,
    },
    LiteralBudgetExceeded {
        required: usize,
        limit: usize,
    },
    ProjectionVariableOutOfRange {
        variable: usize,
        variable_count: usize,
    },
    DuplicateProjectionVariable(usize),
    EmptyProjectionFragments,
    ProjectionFragmentBudgetExceeded {
        required: usize,
        limit: usize,
    },
    ProjectionSupportEntryBudgetExceeded {
        required: usize,
        limit: usize,
    },
    ProjectionScopeMismatch {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    ProjectionAssignmentWidthMismatch {
        expected: usize,
        actual: usize,
    },
    ProjectionSupportBudgetExceeded {
        required_at_least: usize,
        limit: usize,
    },
    EmptyComponent {
        component: usize,
    },
    ComponentCountExceedsVariableCount {
        component_count: usize,
        variable_count: usize,
    },
    PartitionEntryCountMismatch {
        expected: usize,
        actual: usize,
    },
    ComponentVariableOutOfRange {
        component: usize,
        variable: usize,
        variable_count: usize,
    },
    VariableRepeatedInPartition {
        variable: usize,
    },
    VariableMissingFromPartition {
        variable: usize,
    },
    FreeVariableOccursInClause {
        variable: usize,
        clause: usize,
    },
    ClauseCrossesComponents {
        clause: usize,
    },
    EmptyClauseNotSupportedByComponentWitness {
        clause: usize,
    },
    CountOverflow,
    ResourceAccountingOverflow,
    SemanticMismatch {
        source_count: u128,
        aggregate_count: u128,
    },
}

fn validate_problem(
    problem: &CnfProblem,
    budget: EnumerationBudget,
) -> Result<(), DecompositionError> {
    if problem.variables.len() > budget.max_variables {
        return Err(DecompositionError::VariableBudgetExceeded {
            required: problem.variables.len(),
            limit: budget.max_variables,
        });
    }
    let variable_name_bytes = problem.variables.iter().try_fold(0usize, |total, name| {
        total
            .checked_add(name.len())
            .ok_or(DecompositionError::ResourceAccountingOverflow)
    })?;
    if variable_name_bytes > budget.max_total_variable_name_bytes {
        return Err(DecompositionError::VariableNameBudgetExceeded {
            required: variable_name_bytes,
            limit: budget.max_total_variable_name_bytes,
        });
    }
    if problem.clauses.len() > budget.max_clauses {
        return Err(DecompositionError::ClauseBudgetExceeded {
            required: problem.clauses.len(),
            limit: budget.max_clauses,
        });
    }
    let literal_count = problem.clauses.iter().try_fold(0usize, |total, clause| {
        total
            .checked_add(clause.len())
            .ok_or(DecompositionError::ResourceAccountingOverflow)
    })?;
    if literal_count > budget.max_literals {
        return Err(DecompositionError::LiteralBudgetExceeded {
            required: literal_count,
            limit: budget.max_literals,
        });
    }

    let mut names = HashSet::with_capacity(problem.variables.len());
    for (index, name) in problem.variables.iter().enumerate() {
        if name.is_empty() {
            return Err(DecompositionError::EmptyVariableName { index });
        }
        if !names.insert(name.as_str()) {
            return Err(DecompositionError::DuplicateVariableName(name.clone()));
        }
    }

    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        for (literal_index, literal) in clause.iter().enumerate() {
            if literal.variable >= problem.variables.len() {
                return Err(DecompositionError::LiteralVariableOutOfRange {
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

fn check_total_assignment_evaluations(
    variable_counts: &[usize],
    budget: EnumerationBudget,
) -> Result<(), DecompositionError> {
    let total = variable_counts.iter().try_fold(0u128, |sum, &count| {
        sum.checked_add(assignment_count(count, budget)?)
            .ok_or(DecompositionError::ResourceAccountingOverflow)
    })?;
    if total > budget.max_total_assignment_evaluations {
        return Err(DecompositionError::AssignmentBudgetExceeded {
            required: total,
            limit: budget.max_total_assignment_evaluations,
        });
    }
    Ok(())
}

fn assignment_count(
    variable_count: usize,
    budget: EnumerationBudget,
) -> Result<u128, DecompositionError> {
    if variable_count > budget.max_variables {
        return Err(DecompositionError::VariableBudgetExceeded {
            required: variable_count,
            limit: budget.max_variables,
        });
    }

    let shift = u32::try_from(variable_count)
        .map_err(|_| DecompositionError::AssignmentSpaceOverflow { variable_count })?;
    let required = 1u128
        .checked_shl(shift)
        .ok_or(DecompositionError::AssignmentSpaceOverflow { variable_count })?;

    if required > budget.max_total_assignment_evaluations {
        return Err(DecompositionError::AssignmentBudgetExceeded {
            required,
            limit: budget.max_total_assignment_evaluations,
        });
    }
    Ok(required)
}

fn satisfies(problem: &CnfProblem, assignment: &[bool]) -> bool {
    problem.clauses.iter().all(|clause| {
        clause
            .iter()
            .any(|literal| literal.satisfied_by(assignment))
    })
}

/// Count satisfying full assignments by bounded direct enumeration.
///
/// This is an exponential finite oracle, not a scalable solver. It preserves
/// the declared variable universe, including variables absent from clauses.
pub fn count_models(
    problem: &CnfProblem,
    budget: EnumerationBudget,
) -> Result<u128, DecompositionError> {
    validate_problem(problem, budget)?;
    let rows = assignment_count(problem.variables.len(), budget)?;
    let mut count = 0u128;

    for mask in 0..rows {
        let assignment: Vec<bool> = (0..problem.variables.len())
            .map(|index| ((mask >> index) & 1) == 1)
            .collect();
        if satisfies(problem, &assignment) {
            count = count
                .checked_add(1)
                .ok_or(DecompositionError::CountOverflow)?;
        }
    }
    Ok(count)
}

fn cofactor(
    problem: &CnfProblem,
    split_variable: usize,
    value: bool,
) -> Result<CnfProblem, DecompositionError> {
    if split_variable >= problem.variables.len() {
        return Err(DecompositionError::SplitVariableOutOfRange {
            variable: split_variable,
            variable_count: problem.variables.len(),
        });
    }

    let mut old_to_new = vec![usize::MAX; problem.variables.len()];
    let mut variables = Vec::with_capacity(problem.variables.len().saturating_sub(1));
    for (old_index, name) in problem.variables.iter().enumerate() {
        if old_index != split_variable {
            old_to_new[old_index] = variables.len();
            variables.push(name.clone());
        }
    }

    let mut clauses = Vec::with_capacity(problem.clauses.len());
    for clause in &problem.clauses {
        let mut satisfied_by_split = false;
        let mut residual = Vec::with_capacity(clause.len());
        for literal in clause {
            if literal.variable == split_variable {
                if literal.positive == value {
                    satisfied_by_split = true;
                    break;
                }
            } else {
                residual.push(Literal {
                    variable: old_to_new[literal.variable],
                    positive: literal.positive,
                });
            }
        }
        if !satisfied_by_split {
            clauses.push(residual);
        }
    }

    Ok(CnfProblem { variables, clauses })
}

/// Result of checking an exhaustive, disjoint Boolean cofactor partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchSumReport {
    pub split_variable: String,
    pub false_branch_count: u128,
    pub true_branch_count: u128,
    pub aggregate_count: u128,
    pub direct_source_count: u128,
}

/// Verify the finite identity #F = #F[x=false] + #F[x=true].
pub fn verify_branch_sum(
    problem: &CnfProblem,
    split_variable: usize,
    budget: EnumerationBudget,
) -> Result<BranchSumReport, DecompositionError> {
    validate_problem(problem, budget)?;
    if split_variable >= problem.variables.len() {
        return Err(DecompositionError::SplitVariableOutOfRange {
            variable: split_variable,
            variable_count: problem.variables.len(),
        });
    }

    let false_child = cofactor(problem, split_variable, false)?;
    let true_child = cofactor(problem, split_variable, true)?;
    check_total_assignment_evaluations(
        &[
            problem.variables.len(),
            false_child.variables.len(),
            true_child.variables.len(),
        ],
        budget,
    )?;
    let source_count = count_models(problem, budget)?;
    let false_count = count_models(&false_child, budget)?;
    let true_count = count_models(&true_child, budget)?;
    let aggregate = false_count
        .checked_add(true_count)
        .ok_or(DecompositionError::CountOverflow)?;

    if aggregate != source_count {
        return Err(DecompositionError::SemanticMismatch {
            source_count,
            aggregate_count: aggregate,
        });
    }

    Ok(BranchSumReport {
        split_variable: problem.variables[split_variable].clone(),
        false_branch_count: false_count,
        true_branch_count: true_count,
        aggregate_count: aggregate,
        direct_source_count: source_count,
    })
}

/// Result of checking a variable-disjoint component product.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndependentProductReport {
    pub component_counts: Vec<u128>,
    pub free_variable_count: usize,
    pub free_variable_multiplier: u128,
    pub aggregate_count: u128,
    pub direct_source_count: u128,
}

/// Verify exact #SAT multiplication across disjoint variable components.
///
/// Components partition constrained variables. Free variables partition the
/// remaining universe and may not occur in clauses. Every source variable
/// appears exactly once, and every non-empty clause lies in one component.
pub fn verify_independent_product(
    problem: &CnfProblem,
    components: &[Vec<usize>],
    free_variables: &[usize],
    budget: EnumerationBudget,
) -> Result<IndependentProductReport, DecompositionError> {
    validate_problem(problem, budget)?;
    let variable_count = problem.variables.len();
    if components.len() > variable_count {
        return Err(DecompositionError::ComponentCountExceedsVariableCount {
            component_count: components.len(),
            variable_count,
        });
    }
    if free_variables.len() > variable_count {
        return Err(DecompositionError::PartitionEntryCountMismatch {
            expected: variable_count,
            actual: free_variables.len(),
        });
    }

    let mut seen = vec![false; variable_count];
    let mut partition_entries = 0usize;
    let mut component_owner = vec![None; variable_count];
    let mut free = vec![false; variable_count];

    for (component_index, component) in components.iter().enumerate() {
        if component.is_empty() {
            return Err(DecompositionError::EmptyComponent {
                component: component_index,
            });
        }
        for &variable in component {
            if variable >= variable_count {
                return Err(DecompositionError::ComponentVariableOutOfRange {
                    component: component_index,
                    variable,
                    variable_count,
                });
            }
            if seen[variable] {
                return Err(DecompositionError::VariableRepeatedInPartition { variable });
            }
            partition_entries = partition_entries
                .checked_add(1)
                .ok_or(DecompositionError::ResourceAccountingOverflow)?;
            if partition_entries > variable_count {
                return Err(DecompositionError::PartitionEntryCountMismatch {
                    expected: variable_count,
                    actual: partition_entries,
                });
            }
            seen[variable] = true;
            component_owner[variable] = Some(component_index);
        }
    }

    for &variable in free_variables {
        if variable >= variable_count {
            return Err(DecompositionError::ComponentVariableOutOfRange {
                component: components.len(),
                variable,
                variable_count,
            });
        }
        if seen[variable] {
            return Err(DecompositionError::VariableRepeatedInPartition { variable });
        }
        partition_entries = partition_entries
            .checked_add(1)
            .ok_or(DecompositionError::ResourceAccountingOverflow)?;
        if partition_entries > variable_count {
            return Err(DecompositionError::PartitionEntryCountMismatch {
                expected: variable_count,
                actual: partition_entries,
            });
        }
        seen[variable] = true;
        free[variable] = true;
    }

    for (variable, present) in seen.iter().enumerate() {
        if !present {
            return Err(DecompositionError::VariableMissingFromPartition { variable });
        }
    }

    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        if clause.is_empty() {
            return Err(
                DecompositionError::EmptyClauseNotSupportedByComponentWitness {
                    clause: clause_index,
                },
            );
        }
        let mut owner = None;
        for literal in clause {
            if free[literal.variable] {
                return Err(DecompositionError::FreeVariableOccursInClause {
                    variable: literal.variable,
                    clause: clause_index,
                });
            }
            match (owner, component_owner[literal.variable]) {
                (None, Some(component)) => owner = Some(component),
                (Some(previous), Some(component)) if previous == component => {}
                _ => {
                    return Err(DecompositionError::ClauseCrossesComponents {
                        clause: clause_index,
                    });
                }
            }
        }
    }

    let mut enumeration_sizes = Vec::with_capacity(components.len() + 1);
    enumeration_sizes.push(problem.variables.len());
    enumeration_sizes.extend(components.iter().map(Vec::len));
    check_total_assignment_evaluations(&enumeration_sizes, budget)?;

    let source_count = count_models(problem, budget)?;
    let mut component_counts = Vec::with_capacity(components.len());

    for (component_index, component) in components.iter().enumerate() {
        let mut global_to_local = vec![usize::MAX; variable_count];
        let mut variables = Vec::with_capacity(component.len());
        for (local, &global) in component.iter().enumerate() {
            global_to_local[global] = local;
            variables.push(problem.variables[global].clone());
        }

        let mut clauses = Vec::new();
        for (clause_index, clause) in problem.clauses.iter().enumerate() {
            if clause
                .first()
                .is_some_and(|literal| component_owner[literal.variable] == Some(component_index))
            {
                let mut local_clause = Vec::with_capacity(clause.len());
                for literal in clause {
                    if component_owner[literal.variable] != Some(component_index) {
                        return Err(DecompositionError::ClauseCrossesComponents {
                            clause: clause_index,
                        });
                    }
                    local_clause.push(Literal {
                        variable: global_to_local[literal.variable],
                        positive: literal.positive,
                    });
                }
                clauses.push(local_clause);
            }
        }

        component_counts.push(count_models(&CnfProblem { variables, clauses }, budget)?);
    }

    let free_shift = u32::try_from(free_variables.len()).map_err(|_| {
        DecompositionError::AssignmentSpaceOverflow {
            variable_count: free_variables.len(),
        }
    })?;
    let free_multiplier =
        1u128
            .checked_shl(free_shift)
            .ok_or(DecompositionError::AssignmentSpaceOverflow {
                variable_count: free_variables.len(),
            })?;

    let product = component_counts.iter().try_fold(1u128, |acc, count| {
        acc.checked_mul(*count)
            .ok_or(DecompositionError::CountOverflow)
    })?;
    let aggregate = product
        .checked_mul(free_multiplier)
        .ok_or(DecompositionError::CountOverflow)?;

    if aggregate != source_count {
        return Err(DecompositionError::SemanticMismatch {
            source_count,
            aggregate_count: aggregate,
        });
    }

    Ok(IndependentProductReport {
        component_counts,
        free_variable_count: free_variables.len(),
        free_variable_multiplier: free_multiplier,
        aggregate_count: aggregate,
        direct_source_count: source_count,
    })
}

/// A deliberately small typed executable plan for the finite reference engine.
///
/// This is not the general strategy IR from #10899: it has no dynamic code,
/// transformations, external solver adapters, serialized representation, or
/// provenance identity. It does make the three currently qualified semantic
/// operations executable through one typed entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteReferenceStrategy {
    /// Exhaustive, disjoint Shannon/cofactor branching over one Boolean variable.
    BranchSum { split_variable: usize },
    /// Product of clause-disjoint components plus explicitly declared free variables.
    IndependentProduct {
        components: Vec<Vec<usize>>,
        free_variables: Vec<usize>,
    },
    /// Project full models to an ordered visible variable scope using support semantics.
    ProjectedSupport { visible_variables: Vec<usize> },
}

/// Typed result produced by the finite reference strategy interpreter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteReferenceResult {
    BranchSum(BranchSumReport),
    IndependentProduct(IndependentProductReport),
    ProjectedSupport(ProjectedSupportReport),
}

/// Execute one of the qualified finite reference strategies.
///
/// The selected operation and its aggregation semantics are fixed by the enum
/// variant; callers cannot supply an arbitrary aggregation operator. Each
/// operation compares its result with direct source enumeration under the
/// same explicit bounded reference implementation.
pub fn execute_finite_reference_strategy(
    problem: &CnfProblem,
    strategy: &FiniteReferenceStrategy,
    budget: EnumerationBudget,
) -> Result<FiniteReferenceResult, DecompositionError> {
    match strategy {
        FiniteReferenceStrategy::BranchSum { split_variable } => {
            verify_branch_sum(problem, *split_variable, budget)
                .map(FiniteReferenceResult::BranchSum)
        }
        FiniteReferenceStrategy::IndependentProduct {
            components,
            free_variables,
        } => verify_independent_product(problem, components, free_variables, budget)
            .map(FiniteReferenceResult::IndependentProduct),
        FiniteReferenceStrategy::ProjectedSupport { visible_variables } => {
            verify_projected_support(problem, visible_variables, budget)
                .map(FiniteReferenceResult::ProjectedSupport)
        }
    }
}

/// Result of direct projected model counting over a visible variable scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectedSupportReport {
    pub visible_variables: Vec<String>,
    pub full_model_count: u128,
    pub support: BTreeSet<Vec<bool>>,
    pub projected_count: u128,
}

/// Compute projected count as cardinality of existential visible support.
///
/// Full satisfying assignments are enumerated under the explicit budget.
/// Hidden witnesses yielding the same visible assignment collapse in the set.
pub fn verify_projected_support(
    problem: &CnfProblem,
    visible_variables: &[usize],
    budget: EnumerationBudget,
) -> Result<ProjectedSupportReport, DecompositionError> {
    validate_problem(problem, budget)?;
    let rows = assignment_count(problem.variables.len(), budget)?;
    let mut visible_seen = vec![false; problem.variables.len()];
    let mut visible_names = Vec::with_capacity(visible_variables.len());

    for &variable in visible_variables {
        if variable >= problem.variables.len() {
            return Err(DecompositionError::ProjectionVariableOutOfRange {
                variable,
                variable_count: problem.variables.len(),
            });
        }
        if visible_seen[variable] {
            return Err(DecompositionError::DuplicateProjectionVariable(variable));
        }
        visible_seen[variable] = true;
        visible_names.push(problem.variables[variable].clone());
    }

    let mut full_model_count = 0u128;
    let mut support = BTreeSet::new();
    for mask in 0..rows {
        let assignment: Vec<bool> = (0..problem.variables.len())
            .map(|index| ((mask >> index) & 1) == 1)
            .collect();
        if satisfies(problem, &assignment) {
            full_model_count = full_model_count
                .checked_add(1)
                .ok_or(DecompositionError::CountOverflow)?;
            let projected: Vec<bool> = visible_variables
                .iter()
                .map(|&variable| assignment[variable])
                .collect();
            if !support.contains(&projected) && support.len() >= budget.max_projection_support {
                return Err(DecompositionError::ProjectionSupportBudgetExceeded {
                    required_at_least: support.len().saturating_add(1),
                    limit: budget.max_projection_support,
                });
            }
            support.insert(projected);
        }
    }

    let projected_count =
        u128::try_from(support.len()).map_err(|_| DecompositionError::CountOverflow)?;
    Ok(ProjectedSupportReport {
        visible_variables: visible_names,
        full_model_count,
        support,
        projected_count,
    })
}

/// A child result containing support over an explicitly ordered visible scope.
///
/// The scope is part of the value: two assignments with the same Boolean
/// vector but different variable orderings are not interchangeable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionSupportFragment {
    pub visible_variables: Vec<String>,
    pub support: BTreeSet<Vec<bool>>,
}

/// Result of combining child projected supports under existential union.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectedSupportUnionReport {
    pub visible_variables: Vec<String>,
    pub support: BTreeSet<Vec<bool>>,
    pub projected_count: u128,
}

/// Combine projected supports by existential union, failing closed on scope
/// mismatch, malformed assignment widths, or support-budget exhaustion.
///
/// This function validates structural scope identity. The strategy/hyperedge
/// verifier must separately establish that each fragment is bound to the
/// correct child task and that the decomposition itself is sound.
pub fn existential_support_union(
    fragments: &[ProjectionSupportFragment],
    budget: EnumerationBudget,
) -> Result<ProjectedSupportUnionReport, DecompositionError> {
    if fragments.is_empty() {
        return Err(DecompositionError::EmptyProjectionFragments);
    }
    if fragments.len() > budget.max_projection_fragments {
        return Err(DecompositionError::ProjectionFragmentBudgetExceeded {
            required: fragments.len(),
            limit: budget.max_projection_fragments,
        });
    }
    let total_support_entries = fragments.iter().try_fold(0usize, |total, fragment| {
        total
            .checked_add(fragment.support.len())
            .ok_or(DecompositionError::ResourceAccountingOverflow)
    })?;
    if total_support_entries > budget.max_projection_support_entries {
        return Err(DecompositionError::ProjectionSupportEntryBudgetExceeded {
            required: total_support_entries,
            limit: budget.max_projection_support_entries,
        });
    }
    let first = fragments
        .first()
        .ok_or(DecompositionError::EmptyProjectionFragments)?;
    let visible_variables = &first.visible_variables;

    if visible_variables.len() > budget.max_variables {
        return Err(DecompositionError::VariableBudgetExceeded {
            required: visible_variables.len(),
            limit: budget.max_variables,
        });
    }
    let name_bytes = visible_variables.iter().try_fold(0usize, |total, name| {
        total
            .checked_add(name.len())
            .ok_or(DecompositionError::ResourceAccountingOverflow)
    })?;
    if name_bytes > budget.max_total_variable_name_bytes {
        return Err(DecompositionError::VariableNameBudgetExceeded {
            required: name_bytes,
            limit: budget.max_total_variable_name_bytes,
        });
    }

    let mut names = HashSet::with_capacity(visible_variables.len());
    for (index, name) in visible_variables.iter().enumerate() {
        if name.is_empty() {
            return Err(DecompositionError::EmptyVariableName { index });
        }
        if !names.insert(name.as_str()) {
            return Err(DecompositionError::DuplicateVariableName(name.clone()));
        }
    }

    let mut support = BTreeSet::new();
    for fragment in fragments {
        if fragment.visible_variables.as_slice() != visible_variables.as_slice() {
            return Err(DecompositionError::ProjectionScopeMismatch {
                expected: visible_variables.to_vec(),
                actual: fragment.visible_variables.to_vec(),
            });
        }
        for assignment in &fragment.support {
            if assignment.len() != visible_variables.len() {
                return Err(DecompositionError::ProjectionAssignmentWidthMismatch {
                    expected: visible_variables.len(),
                    actual: assignment.len(),
                });
            }
            if !support.contains(assignment) && support.len() >= budget.max_projection_support {
                return Err(DecompositionError::ProjectionSupportBudgetExceeded {
                    required_at_least: support.len().saturating_add(1),
                    limit: budget.max_projection_support,
                });
            }
            support.insert(assignment.clone());
        }
    }

    let projected_count =
        u128::try_from(support.len()).map_err(|_| DecompositionError::CountOverflow)?;
    Ok(ProjectedSupportUnionReport {
        visible_variables: visible_variables.to_vec(),
        support,
        projected_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> EnumerationBudget {
        EnumerationBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 64,
            max_literals: 256,
            max_total_assignment_evaluations: 1 << 8,
            max_projection_support: 1 << 8,
            max_projection_fragments: 64,
            max_projection_support_entries: 1 << 8,
        }
    }

    // Deliberately separate from the production evaluator: this test oracle
    // evaluates each literal directly from a bit mask and shares no helper
    // with count_models, cofactor, or the decomposition verifiers.
    fn independent_model_count(problem: &CnfProblem) -> u128 {
        let rows = 1usize << problem.variables.len();
        (0..rows)
            .filter(|mask| {
                problem.clauses.iter().all(|clause| {
                    clause.iter().any(|literal| {
                        let value = ((mask >> literal.variable) & 1) != 0;
                        value == literal.positive
                    })
                })
            })
            .count() as u128
    }

    fn independent_projected_support(
        problem: &CnfProblem,
        visible_variable: usize,
    ) -> (u128, BTreeSet<Vec<bool>>) {
        let rows = 1usize << problem.variables.len();
        let mut full_models = 0u128;
        let mut support = BTreeSet::new();
        for mask in 0..rows {
            let is_model = problem.clauses.iter().all(|clause| {
                clause.iter().any(|literal| {
                    let value = ((mask >> literal.variable) & 1) != 0;
                    value == literal.positive
                })
            });
            if is_model {
                full_models += 1;
                support.insert(vec![((mask >> visible_variable) & 1) != 0]);
            }
        }
        (full_models, support)
    }

    fn independent_projected_support_scope(
        problem: &CnfProblem,
        visible_variables: &[usize],
    ) -> (u128, BTreeSet<Vec<bool>>) {
        let rows = 1usize << problem.variables.len();
        let mut full_models = 0u128;
        let mut support = BTreeSet::new();
        for mask in 0..rows {
            let is_model = problem.clauses.iter().all(|clause| {
                clause.iter().any(|literal| {
                    let value = ((mask >> literal.variable) & 1) != 0;
                    value == literal.positive
                })
            });
            if is_model {
                full_models += 1;
                support.insert(
                    visible_variables
                        .iter()
                        .map(|&variable| ((mask >> variable) & 1) != 0)
                        .collect(),
                );
            }
        }
        (full_models, support)
    }

    fn all_two_variable_clauses() -> Vec<Vec<Literal>> {
        let literals = [
            Literal::positive(0),
            Literal::negative(0),
            Literal::positive(1),
            Literal::negative(1),
        ];
        (0..(1usize << literals.len()))
            .map(|mask| {
                literals
                    .iter()
                    .enumerate()
                    .filter_map(|(index, literal)| (((mask >> index) & 1) != 0).then_some(*literal))
                    .collect()
            })
            .collect()
    }

    fn branch_fixture() -> CnfProblem {
        // (x OR y) AND (!x OR z)
        CnfProblem {
            variables: vec!["x".into(), "y".into(), "z".into()],
            clauses: vec![
                vec![Literal::positive(0), Literal::positive(1)],
                vec![Literal::negative(0), Literal::positive(2)],
            ],
        }
    }

    fn independent_fixture() -> CnfProblem {
        // (a OR b) AND (!a OR b) AND (c OR d); e is free.
        CnfProblem {
            variables: vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
            clauses: vec![
                vec![Literal::positive(0), Literal::positive(1)],
                vec![Literal::negative(0), Literal::positive(1)],
                vec![Literal::positive(2), Literal::positive(3)],
            ],
        }
    }

    fn projection_fixture() -> CnfProblem {
        // (v AND h1) OR (v AND h2) is equivalent to v AND (h1 OR h2).
        CnfProblem {
            variables: vec!["v".into(), "h1".into(), "h2".into()],
            clauses: vec![
                vec![Literal::positive(0)],
                vec![Literal::positive(1), Literal::positive(2)],
            ],
        }
    }

    #[test]
    fn typed_reference_strategy_executes_all_three_finite_semantics() {
        let branch = execute_finite_reference_strategy(
            &branch_fixture(),
            &FiniteReferenceStrategy::BranchSum { split_variable: 0 },
            budget(),
        )
        .unwrap();
        assert!(matches!(
            branch,
            FiniteReferenceResult::BranchSum(BranchSumReport {
                aggregate_count: 4,
                direct_source_count: 4,
                ..
            })
        ));

        let independent = execute_finite_reference_strategy(
            &independent_fixture(),
            &FiniteReferenceStrategy::IndependentProduct {
                components: vec![vec![0, 1], vec![2, 3]],
                free_variables: vec![4],
            },
            budget(),
        )
        .unwrap();
        assert!(matches!(
            independent,
            FiniteReferenceResult::IndependentProduct(IndependentProductReport {
                aggregate_count: 12,
                direct_source_count: 12,
                free_variable_multiplier: 2,
                ..
            })
        ));

        let projected = execute_finite_reference_strategy(
            &projection_fixture(),
            &FiniteReferenceStrategy::ProjectedSupport {
                visible_variables: vec![0],
            },
            budget(),
        )
        .unwrap();
        assert!(matches!(
            projected,
            FiniteReferenceResult::ProjectedSupport(ProjectedSupportReport {
                full_model_count: 3,
                projected_count: 1,
                ..
            })
        ));
    }

    #[test]
    fn branch_sum_matches_direct_source_enumeration() {
        let report = verify_branch_sum(&branch_fixture(), 0, budget()).unwrap();
        assert_eq!(report.false_branch_count, 2);
        assert_eq!(report.true_branch_count, 2);
        assert_eq!(report.aggregate_count, 4);
        assert_eq!(report.direct_source_count, 4);
    }

    #[test]
    fn independent_product_preserves_free_variable_multiplicity() {
        let report = verify_independent_product(
            &independent_fixture(),
            &[vec![0, 1], vec![2, 3]],
            &[4],
            budget(),
        )
        .unwrap();
        assert_eq!(report.component_counts, vec![2, 3]);
        assert_eq!(report.free_variable_multiplier, 2);
        assert_eq!(report.aggregate_count, 12);
        assert_eq!(report.direct_source_count, 12);
    }

    #[test]
    fn projection_collapses_overlapping_hidden_witnesses() {
        let report = verify_projected_support(&projection_fixture(), &[0], budget()).unwrap();
        assert_eq!(report.full_model_count, 3);
        assert_eq!(report.projected_count, 1);
        assert_eq!(report.support, BTreeSet::from([vec![true]]));
    }

    #[test]
    fn existential_union_deduplicates_same_visible_assignment() {
        let left = ProjectionSupportFragment {
            visible_variables: vec!["v".into()],
            support: BTreeSet::from([vec![true]]),
        };
        let right = ProjectionSupportFragment {
            visible_variables: vec!["v".into()],
            support: BTreeSet::from([vec![true]]),
        };
        let report = existential_support_union(&[left, right], budget()).unwrap();
        assert_eq!(report.projected_count, 1);
        assert_eq!(report.support, BTreeSet::from([vec![true]]));
    }

    #[test]
    fn existential_union_rejects_visible_scope_mismatch() {
        let left = ProjectionSupportFragment {
            visible_variables: vec!["v".into()],
            support: BTreeSet::from([vec![true]]),
        };
        let right = ProjectionSupportFragment {
            visible_variables: vec!["h".into()],
            support: BTreeSet::from([vec![true]]),
        };
        assert_eq!(
            existential_support_union(&[left, right], budget()).unwrap_err(),
            DecompositionError::ProjectionScopeMismatch {
                expected: vec!["v".into()],
                actual: vec!["h".into()],
            }
        );
    }

    #[test]
    fn existential_union_rejects_assignment_width_mismatch() {
        let fragment = ProjectionSupportFragment {
            visible_variables: vec!["v".into()],
            support: BTreeSet::from([vec![true, false]]),
        };
        assert_eq!(
            existential_support_union(&[fragment], budget()).unwrap_err(),
            DecompositionError::ProjectionAssignmentWidthMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn existential_union_rejects_empty_fragment_list() {
        assert_eq!(
            existential_support_union(&[], budget()).unwrap_err(),
            DecompositionError::EmptyProjectionFragments
        );
    }

    #[test]
    fn existential_union_bounds_fragment_count() {
        let fragment = ProjectionSupportFragment {
            visible_variables: vec!["v".into()],
            support: BTreeSet::new(),
        };
        let tight = EnumerationBudget {
            max_projection_fragments: 1,
            ..budget()
        };
        assert_eq!(
            existential_support_union(&[fragment.clone(), fragment], tight).unwrap_err(),
            DecompositionError::ProjectionFragmentBudgetExceeded {
                required: 2,
                limit: 1,
            }
        );
    }

    #[test]
    fn existential_union_bounds_total_support_entries_even_when_overlapping() {
        let fragment = ProjectionSupportFragment {
            visible_variables: vec!["v".into()],
            support: BTreeSet::from([vec![true]]),
        };
        let tight = EnumerationBudget {
            max_projection_support_entries: 1,
            ..budget()
        };
        assert_eq!(
            existential_support_union(&[fragment.clone(), fragment], tight).unwrap_err(),
            DecompositionError::ProjectionSupportEntryBudgetExceeded {
                required: 2,
                limit: 1,
            }
        );
    }

    #[test]
    fn independent_product_rejects_partition_entry_count_before_allocation() {
        let problem = independent_fixture();
        let error = verify_independent_product(
            &problem,
            &[vec![0, 1, 2, 3]],
            &[4, 4, 4, 4, 4, 4],
            budget(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            DecompositionError::PartitionEntryCountMismatch {
                expected: 5,
                actual: 6,
            }
        );
    }

    #[test]
    fn independent_product_rejects_too_many_components_before_iteration() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![],
        };
        let error =
            verify_independent_product(&problem, &[vec![0], vec![]], &[], budget()).unwrap_err();
        assert_eq!(
            error,
            DecompositionError::ComponentCountExceedsVariableCount {
                component_count: 2,
                variable_count: 1,
            }
        );
    }

    #[test]
    fn independent_product_rejects_overlapping_component_scopes() {
        let error = verify_independent_product(
            &independent_fixture(),
            &[vec![0, 1], vec![1, 2, 3]],
            &[4],
            budget(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            DecompositionError::VariableRepeatedInPartition { variable: 1 }
        );
    }

    #[test]
    fn independent_product_rejects_lost_free_variable() {
        let error = verify_independent_product(
            &independent_fixture(),
            &[vec![0, 1], vec![2, 3]],
            &[],
            budget(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            DecompositionError::VariableMissingFromPartition { variable: 4 }
        );
    }

    #[test]
    fn independent_product_rejects_cross_component_clause() {
        let problem = CnfProblem {
            variables: vec!["a".into(), "b".into()],
            clauses: vec![vec![Literal::positive(0), Literal::positive(1)]],
        };
        let error =
            verify_independent_product(&problem, &[vec![0], vec![1]], &[], budget()).unwrap_err();
        assert_eq!(
            error,
            DecompositionError::ClauseCrossesComponents { clause: 0 }
        );
    }

    #[test]
    fn resource_limit_is_unknown_not_zero_models() {
        let tight = EnumerationBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 64,
            max_literals: 256,
            max_total_assignment_evaluations: 4,
            max_projection_support: 8,
            max_projection_fragments: 64,
            max_projection_support_entries: 1 << 8,
        };
        assert_eq!(
            count_models(&branch_fixture(), tight).unwrap_err(),
            DecompositionError::AssignmentBudgetExceeded {
                required: 8,
                limit: 4
            }
        );
    }

    #[test]
    fn branch_sum_enforces_composed_total_enumeration_budget() {
        let tight = EnumerationBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 64,
            max_literals: 256,
            max_total_assignment_evaluations: 12,
            max_projection_support: 8,
            max_projection_fragments: 64,
            max_projection_support_entries: 1 << 8,
        };
        assert_eq!(
            verify_branch_sum(&branch_fixture(), 0, tight).unwrap_err(),
            DecompositionError::AssignmentBudgetExceeded {
                required: 16,
                limit: 12
            }
        );
    }

    #[test]
    fn independent_product_enforces_composed_total_enumeration_budget() {
        let tight = EnumerationBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 64,
            max_literals: 256,
            max_total_assignment_evaluations: 35,
            max_projection_support: 8,
            max_projection_fragments: 64,
            max_projection_support_entries: 1 << 8,
        };
        assert_eq!(
            verify_independent_product(
                &independent_fixture(),
                &[vec![0, 1], vec![2, 3]],
                &[4],
                tight,
            )
            .unwrap_err(),
            DecompositionError::AssignmentBudgetExceeded {
                required: 40,
                limit: 35
            }
        );
    }

    #[test]
    fn variable_name_byte_budget_is_checked_before_cloning() {
        let tight = EnumerationBudget {
            max_total_variable_name_bytes: 2,
            ..budget()
        };
        assert_eq!(
            count_models(&branch_fixture(), tight).unwrap_err(),
            DecompositionError::VariableNameBudgetExceeded {
                required: 3,
                limit: 2
            }
        );
    }

    #[test]
    fn clause_budget_is_checked_before_enumeration() {
        let tight = EnumerationBudget {
            max_clauses: 1,
            ..budget()
        };
        assert_eq!(
            count_models(&branch_fixture(), tight).unwrap_err(),
            DecompositionError::ClauseBudgetExceeded {
                required: 2,
                limit: 1
            }
        );
    }

    #[test]
    fn literal_budget_is_checked_before_enumeration() {
        let tight = EnumerationBudget {
            max_literals: 3,
            ..budget()
        };
        assert_eq!(
            count_models(&branch_fixture(), tight).unwrap_err(),
            DecompositionError::LiteralBudgetExceeded {
                required: 4,
                limit: 3
            }
        );
    }

    #[test]
    fn projected_support_budget_stops_before_inserting_extra_support() {
        let tight = EnumerationBudget {
            max_projection_support: 0,
            ..budget()
        };
        assert_eq!(
            verify_projected_support(&projection_fixture(), &[0], tight).unwrap_err(),
            DecompositionError::ProjectionSupportBudgetExceeded {
                required_at_least: 1,
                limit: 0
            }
        );
    }

    #[test]
    fn branch_rejects_unknown_split_variable() {
        assert_eq!(
            verify_branch_sum(&branch_fixture(), 3, budget()).unwrap_err(),
            DecompositionError::SplitVariableOutOfRange {
                variable: 3,
                variable_count: 3
            }
        );
    }

    #[test]
    fn projection_rejects_duplicate_visible_variables() {
        assert_eq!(
            verify_projected_support(&projection_fixture(), &[0, 0], budget()).unwrap_err(),
            DecompositionError::DuplicateProjectionVariable(0)
        );
    }

    #[test]
    fn empty_cnf_is_true_over_its_declared_universe() {
        let problem = CnfProblem {
            variables: vec!["free".into()],
            clauses: vec![],
        };
        assert_eq!(count_models(&problem, budget()).unwrap(), 2);
    }

    #[test]
    fn empty_clause_is_false() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![]],
        };
        assert_eq!(count_models(&problem, budget()).unwrap(), 0);
    }

    #[test]
    fn exhaustive_two_variable_cnf_matches_independent_truth_table_oracle() {
        let clauses = all_two_variable_clauses();
        let mut formulas = vec![Vec::<Vec<Literal>>::new()];
        formulas.extend(clauses.iter().cloned().map(|clause| vec![clause]));
        for left in &clauses {
            for right in &clauses {
                formulas.push(vec![left.clone(), right.clone()]);
            }
        }

        assert_eq!(formulas.len(), 273);
        for (formula_index, formula) in formulas.into_iter().enumerate() {
            let problem = CnfProblem {
                variables: vec!["x".into(), "y".into()],
                clauses: formula,
            };
            let expected_models = independent_model_count(&problem);
            assert_eq!(
                count_models(&problem, budget()).unwrap(),
                expected_models,
                "full model count disagreed for formula {formula_index}"
            );

            for split_variable in 0..2 {
                let report = verify_branch_sum(&problem, split_variable, budget()).unwrap();
                let false_count = (0..4usize)
                    .filter(|mask| {
                        ((mask >> split_variable) & 1) == 0
                            && problem.clauses.iter().all(|clause| {
                                clause.iter().any(|literal| {
                                    let value = ((mask >> literal.variable) & 1) != 0;
                                    value == literal.positive
                                })
                            })
                    })
                    .count() as u128;
                let true_count = (0..4usize)
                    .filter(|mask| {
                        ((mask >> split_variable) & 1) != 0
                            && problem.clauses.iter().all(|clause| {
                                clause.iter().any(|literal| {
                                    let value = ((mask >> literal.variable) & 1) != 0;
                                    value == literal.positive
                                })
                            })
                    })
                    .count() as u128;
                assert_eq!(report.direct_source_count, expected_models);
                assert_eq!(report.false_branch_count, false_count);
                assert_eq!(report.true_branch_count, true_count);
                assert_eq!(report.aggregate_count, expected_models);
            }

            for visible_variable in 0..2 {
                let (expected_full, expected_support) =
                    independent_projected_support(&problem, visible_variable);
                let report =
                    verify_projected_support(&problem, &[visible_variable], budget()).unwrap();
                assert_eq!(report.full_model_count, expected_full);
                assert_eq!(report.support, expected_support);
                assert_eq!(report.projected_count, expected_support.len() as u128);
            }
        }
    }

    #[test]
    fn generated_three_variable_cnfs_match_independent_truth_table_oracle() {
        // Deterministic pseudo-random formula generation keeps the corpus
        // reproducible and dependency-free. It deliberately includes empty
        // clauses, repeated literals, tautologies and arbitrary clause sizes.
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

        for formula_index in 0..512 {
            let clause_count = (next() % 6) as usize;
            let mut clauses = Vec::with_capacity(clause_count);
            for _ in 0..clause_count {
                let literal_count = (next() % 6) as usize;
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
            let expected_models = independent_model_count(&problem);
            assert_eq!(
                count_models(&problem, budget()).unwrap(),
                expected_models,
                "full model count disagreed for generated formula {formula_index}"
            );

            for split_variable in 0..3 {
                let report = verify_branch_sum(&problem, split_variable, budget()).unwrap();
                let false_count = (0..8usize)
                    .filter(|mask| {
                        ((mask >> split_variable) & 1) == 0
                            && problem.clauses.iter().all(|clause| {
                                clause.iter().any(|literal| {
                                    let value = ((mask >> literal.variable) & 1) != 0;
                                    value == literal.positive
                                })
                            })
                    })
                    .count() as u128;
                let true_count = (0..8usize)
                    .filter(|mask| {
                        ((mask >> split_variable) & 1) != 0
                            && problem.clauses.iter().all(|clause| {
                                clause.iter().any(|literal| {
                                    let value = ((mask >> literal.variable) & 1) != 0;
                                    value == literal.positive
                                })
                            })
                    })
                    .count() as u128;
                assert_eq!(report.direct_source_count, expected_models);
                assert_eq!(report.false_branch_count, false_count);
                assert_eq!(report.true_branch_count, true_count);
                assert_eq!(report.aggregate_count, expected_models);
            }

            // Cover every visible scope: empty, each singleton, each pair,
            // and the complete universe.
            for visible_mask in 0..8usize {
                let visible: Vec<usize> = (0..3)
                    .filter(|variable| ((visible_mask >> variable) & 1) != 0)
                    .collect();
                let (expected_full, expected_support) =
                    independent_projected_support_scope(&problem, &visible);
                let report = verify_projected_support(&problem, &visible, budget()).unwrap();
                assert_eq!(report.full_model_count, expected_full);
                assert_eq!(report.support, expected_support);
                assert_eq!(
                    report.projected_count,
                    expected_support.len() as u128,
                    "projected count disagreed for formula {formula_index}, scope {visible_mask}"
                );
            }
        }
    }

    #[test]
    fn exhaustive_unary_component_products_match_independent_oracle() {
        let unit_clauses = [
            vec![Literal::positive(0)],
            vec![Literal::negative(0)],
            vec![Literal::positive(1)],
            vec![Literal::negative(1)],
        ];
        for mask in 0..(1usize << unit_clauses.len()) {
            let clauses = unit_clauses
                .iter()
                .enumerate()
                .filter_map(|(index, clause)| {
                    (((mask >> index) & 1) != 0).then_some(clause.clone())
                })
                .collect();
            let problem = CnfProblem {
                variables: vec!["x".into(), "y".into(), "free".into()],
                clauses,
            };
            let expected = independent_model_count(&problem);
            let report =
                verify_independent_product(&problem, &[vec![0], vec![1]], &[2], budget()).unwrap();
            assert_eq!(
                report.direct_source_count, expected,
                "unit clause mask {mask}"
            );
            assert_eq!(report.aggregate_count, expected, "unit clause mask {mask}");
        }
    }
}
