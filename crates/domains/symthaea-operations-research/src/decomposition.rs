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
        Self { variable, positive: true }
    }

    pub const fn negative(variable: usize) -> Self {
        Self { variable, positive: false }
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
        }
    }
}

/// Typed failure. A failed finite check never returns a partial count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionError {
    EmptyVariableName { index: usize },
    DuplicateVariableName(String),
    LiteralVariableOutOfRange {
        clause: usize,
        literal: usize,
        variable: usize,
        variable_count: usize,
    },
    SplitVariableOutOfRange { variable: usize, variable_count: usize },
    AssignmentSpaceOverflow { variable_count: usize },
    VariableBudgetExceeded { required: usize, limit: usize },
    VariableNameBudgetExceeded { required: usize, limit: usize },
    AssignmentBudgetExceeded { required: u128, limit: u128 },
    ClauseBudgetExceeded { required: usize, limit: usize },
    LiteralBudgetExceeded { required: usize, limit: usize },
    ProjectionVariableOutOfRange { variable: usize, variable_count: usize },
    DuplicateProjectionVariable(usize),
    ProjectionSupportBudgetExceeded { required_at_least: usize, limit: usize },
    EmptyComponent { component: usize },
    ComponentVariableOutOfRange {
        component: usize,
        variable: usize,
        variable_count: usize,
    },
    VariableRepeatedInPartition { variable: usize },
    VariableMissingFromPartition { variable: usize },
    FreeVariableOccursInClause { variable: usize, clause: usize },
    ClauseCrossesComponents { clause: usize },
    EmptyClauseNotSupportedByComponentWitness { clause: usize },
    CountOverflow,
    ResourceAccountingOverflow,
    SemanticMismatch { source_count: u128, aggregate_count: u128 },
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
    problem
        .clauses
        .iter()
        .all(|clause| clause.iter().any(|literal| literal.satisfied_by(assignment)))
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
            count = count.checked_add(1).ok_or(DecompositionError::CountOverflow)?;
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
    let mut seen = vec![false; variable_count];
    let mut component_owner = vec![None; variable_count];
    let mut free = vec![false; variable_count];

    for (component_index, component) in components.iter().enumerate() {
        if component.is_empty() {
            return Err(DecompositionError::EmptyComponent { component: component_index });
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
            return Err(DecompositionError::EmptyClauseNotSupportedByComponentWitness {
                clause: clause_index,
            });
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
                _ => return Err(DecompositionError::ClauseCrossesComponents {
                    clause: clause_index,
                }),
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
            if clause.first().is_some_and(|literal| {
                component_owner[literal.variable] == Some(component_index)
            }) {
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
    let free_multiplier = 1u128
        .checked_shl(free_shift)
        .ok_or(DecompositionError::AssignmentSpaceOverflow {
            variable_count: free_variables.len(),
        })?;

    let product = component_counts.iter().try_fold(1u128, |acc, count| {
        acc.checked_mul(*count).ok_or(DecompositionError::CountOverflow)
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

/// Combine projected supports by existential union, deduplicating witnesses.
pub fn existential_support_union(
    supports: &[BTreeSet<Vec<bool>>],
) -> BTreeSet<Vec<bool>> {
    supports.iter().flat_map(|support| support.iter().cloned()).collect()
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
        }
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
        let left = BTreeSet::from([vec![true]]);
        let right = BTreeSet::from([vec![true]]);
        assert_eq!(existential_support_union(&[left, right]).len(), 1);
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
}
