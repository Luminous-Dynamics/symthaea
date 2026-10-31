// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact binary sum-product elimination for sparse causal factor graphs.
//!
//! This module is deliberately independent of IIT scoring. It provides the
//! numerical primitive needed to replace global-state enumeration in the
//! IIT 4.0 Eq. 4 background-posterior contraction. The dense TPM implementation
//! remains the correctness oracle until differential tests cover that integration.
//!
//! Variable indices are global model-unit indices. Within a factor's table,
//! variables[0] is the least-significant assignment bit.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use thiserror::Error;

/// Failures while constructing or eliminating binary factors.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FactorGraphError {
    #[error("factor variable scope must be strictly ascending; found {left} before {right}")]
    ScopeNotStrictlyAscending { left: usize, right: usize },

    #[error("factor with {variables} variables cannot be represented by this platform")]
    ScopeTooWide { variables: usize },

    #[error("factor table length mismatch: expected {expected}, got {actual}")]
    TableLengthMismatch { expected: usize, actual: usize },

    #[error("factor table value at index {index} must be finite and nonnegative; got {value}")]
    InvalidFactorValue { index: usize, value: f64 },

    #[error("log-factor value at index {index} must not be NaN or positive infinity; got {value}")]
    InvalidLogFactorValue { index: usize, value: f64 },

    #[error("factor graph must contain at least one factor")]
    EmptyFactorGraph,

    #[error("elimination variable {variable} was requested more than once")]
    DuplicateEliminationVariable { variable: usize },

    #[error("cutset variable {variable} was requested more than once")]
    DuplicateCutsetVariable { variable: usize },
    #[error("cutset frontier requires at least one candidate separator")]
    EmptyCutsetCandidateSet,

    #[error("cutset variable {variable} is not in the elimination set")]
    CutsetVariableNotEliminated { variable: usize },

    #[error("cutset requires {assignments} branches, exceeding the configured limit {maximum}")]
    CutsetAssignmentBudgetExceeded { assignments: usize, maximum: usize },

    #[error("conditioned cutset branches returned different output scopes")]
    CutsetBranchScopeMismatch,

    #[error("factor scopes do not match the reusable cutset elimination plan")]
    FactorScopesDoNotMatchCutsetPlan,

    #[error("variable {variable} is not present in the factor scope")]
    VariableNotInFactor { variable: usize },

    #[error("factor intermediate has {entries} entries, exceeding the configured limit {limit}")]
    IntermediateTooLarge { entries: usize, limit: usize },

    #[error("factor-table index arithmetic overflow")]
    IndexOverflow,
}

/// A nonnegative function over a sorted set of binary variables.
///
/// The table has 2^variables.len() entries. Assignment bits follow the
/// variable order, with the first scope variable in the least-significant bit.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryFactor {
    variables: Vec<usize>,
    values: Vec<f64>,
}

impl BinaryFactor {
    /// Construct a factor. The scope must already be strictly ascending.
    pub fn new(
        variables: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, FactorGraphError> {
        for pair in variables.windows(2) {
            if pair[0] >= pair[1] {
                return Err(FactorGraphError::ScopeNotStrictlyAscending {
                    left: pair[0],
                    right: pair[1],
                });
            }
        }
        let expected = assignment_count(variables.len())?;
        if values.len() != expected {
            return Err(FactorGraphError::TableLengthMismatch {
                expected,
                actual: values.len(),
            });
        }
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(FactorGraphError::InvalidFactorValue { index, value });
            }
        }
        Ok(Self { variables, values })
    }

    /// Construct a scalar factor, which has an empty variable scope.
    pub fn scalar(value: f64) -> Result<Self, FactorGraphError> {
        Self::new(Vec::new(), vec![value])
    }

    pub fn variables(&self) -> &[usize] {
        &self.variables
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Read a table entry by local assignment index.
    pub fn value_at(&self, assignment_index: usize) -> Option<f64> {
        self.values.get(assignment_index).copied()
    }

    /// Multiply two factors, enforcing the intermediate-entry budget before
    /// allocating the joint table.
    pub fn multiply(
        &self,
        other: &Self,
        max_factor_entries: usize,
    ) -> Result<Self, FactorGraphError> {
        let variables = sorted_union(&self.variables, &other.variables);
        let entries = assignment_count(variables.len())?;
        check_budget(entries, max_factor_entries)?;
        let mut values = Vec::with_capacity(entries);
        for assignment in 0..entries {
            let left_index = project_assignment(assignment, &variables, &self.variables)?;
            let right_index = project_assignment(assignment, &variables, &other.variables)?;
            values.push(self.values[left_index] * other.values[right_index]);
        }
        Self::new(variables, values)
    }

    /// Sum out one variable. If the variable is not in this factor, this
    /// operation is an error rather than a silent no-op.
    pub fn sum_out(
        &self,
        variable: usize,
        max_factor_entries: usize,
    ) -> Result<Self, FactorGraphError> {
        let variable_slot = self
            .variables
            .binary_search(&variable)
            .map_err(|_| FactorGraphError::VariableNotInFactor { variable })?;
        let mut variables = self.variables.clone();
        variables.remove(variable_slot);
        let entries = assignment_count(variables.len())?;
        check_budget(entries, max_factor_entries)?;
        let mut values = Vec::with_capacity(entries);
        for assignment in 0..entries {
            let zero_index = insert_assignment_bit(assignment, variable_slot, false)?;
            let one_index = insert_assignment_bit(assignment, variable_slot, true)?;
            values.push(self.values[zero_index] + self.values[one_index]);
        }
        Self::new(variables, values)
    }
}

/// A binary factor stored in log space.
///
/// Negative infinity represents an exact zero weight. Products become sums,
/// and marginalization uses log-sum-exp, preventing underflow in long causal
/// likelihood products. Scope and assignment-bit ordering match BinaryFactor.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryLogFactor {
    variables: Vec<usize>,
    log_values: Vec<f64>,
}

impl BinaryLogFactor {
    /// Construct a log factor. NaN and positive infinity are rejected;
    /// negative infinity is the representation of zero probability.
    pub fn new(
        variables: Vec<usize>,
        log_values: Vec<f64>,
    ) -> Result<Self, FactorGraphError> {
        for pair in variables.windows(2) {
            if pair[0] >= pair[1] {
                return Err(FactorGraphError::ScopeNotStrictlyAscending {
                    left: pair[0],
                    right: pair[1],
                });
            }
        }
        let expected = assignment_count(variables.len())?;
        if log_values.len() != expected {
            return Err(FactorGraphError::TableLengthMismatch {
                expected,
                actual: log_values.len(),
            });
        }
        for (index, value) in log_values.iter().copied().enumerate() {
            if value.is_nan() || value == f64::INFINITY {
                return Err(FactorGraphError::InvalidLogFactorValue { index, value });
            }
        }
        Ok(Self { variables, log_values })
    }

    pub fn scalar(log_value: f64) -> Result<Self, FactorGraphError> {
        Self::new(Vec::new(), vec![log_value])
    }

    pub fn variables(&self) -> &[usize] {
        &self.variables
    }

    pub fn log_values(&self) -> &[f64] {
        &self.log_values
    }

    /// Multiply two log factors by adding their log tables.
    pub fn multiply(
        &self,
        other: &Self,
        max_factor_entries: usize,
    ) -> Result<Self, FactorGraphError> {
        let variables = sorted_union(&self.variables, &other.variables);
        let entries = assignment_count(variables.len())?;
        check_budget(entries, max_factor_entries)?;
        let mut log_values = Vec::with_capacity(entries);
        for assignment in 0..entries {
            let left_index = project_assignment(assignment, &variables, &self.variables)?;
            let right_index = project_assignment(assignment, &variables, &other.variables)?;
            let value = self.log_values[left_index] + other.log_values[right_index];
            if value.is_nan() || value == f64::INFINITY {
                return Err(FactorGraphError::InvalidLogFactorValue { index: assignment, value });
            }
            log_values.push(value);
        }
        Self::new(variables, log_values)
    }

    /// Condition one variable to a binary value and remove it from this factor's scope.
    ///
    /// The remaining table preserves the original sorted variable order and local
    /// least-significant-bit convention. Conditioning an absent variable is an
    /// identity operation, which is useful when a cutset variable is disconnected.
    pub fn condition(&self, variable: usize, value: bool) -> Result<Self, FactorGraphError> {
        self.condition_many(&[(variable, value)])
    }

    /// Condition an ascending set of variables in one table pass.
    ///
    /// Assignments whose variables are absent from this factor are ignored.
    /// Unlike repeated single-variable conditioning, this builds the reduced table
    /// once and avoids intermediate table allocations for a multi-variable cutset.
    /// Assignment order must be strictly ascending and duplicate-free.
    pub fn condition_many(
        &self,
        assignments: &[(usize, bool)],
    ) -> Result<Self, FactorGraphError> {
        for pair in assignments.windows(2) {
            if pair[0].0 >= pair[1].0 {
                return Err(FactorGraphError::ScopeNotStrictlyAscending {
                    left: pair[0].0,
                    right: pair[1].0,
                });
            }
        }

        let mut assignment_slot = 0usize;
        let mut fixed_values = vec![None; self.variables.len()];
        let mut variables = Vec::with_capacity(self.variables.len());
        for (local_slot, variable) in self.variables.iter().copied().enumerate() {
            while assignments
                .get(assignment_slot)
                .is_some_and(|(assigned, _)| *assigned < variable)
            {
                assignment_slot += 1;
            }
            if let Some((assigned, value)) = assignments.get(assignment_slot) {
                if *assigned == variable {
                    fixed_values[local_slot] = Some(*value);
                    assignment_slot += 1;
                    continue;
                }
            }
            variables.push(variable);
        }

        if variables.len() == self.variables.len() {
            return Ok(self.clone());
        }

        let entries = assignment_count(variables.len())?;
        let mut log_values = Vec::with_capacity(entries);
        for assignment in 0..entries {
            let mut original_index = 0usize;
            let mut output_slot = 0usize;
            for (local_slot, fixed_value) in fixed_values.iter().copied().enumerate() {
                let bit = match fixed_value {
                    Some(value) => usize::from(value),
                    None => {
                        let value = (assignment >> output_slot) & 1;
                        output_slot += 1;
                        value
                    }
                };
                original_index |= bit << local_slot;
            }
            log_values.push(self.log_values[original_index]);
        }
        Self::new(variables, log_values)
    }

    /// Sum out one variable with numerically stable log-sum-exp.
    pub fn sum_out(
        &self,
        variable: usize,
        max_factor_entries: usize,
    ) -> Result<Self, FactorGraphError> {
        let variable_slot = self
            .variables
            .binary_search(&variable)
            .map_err(|_| FactorGraphError::VariableNotInFactor { variable })?;
        let mut variables = self.variables.clone();
        variables.remove(variable_slot);
        let entries = assignment_count(variables.len())?;
        check_budget(entries, max_factor_entries)?;
        let mut log_values = Vec::with_capacity(entries);
        for assignment in 0..entries {
            let zero_index = insert_assignment_bit(assignment, variable_slot, false)?;
            let one_index = insert_assignment_bit(assignment, variable_slot, true)?;
            log_values.push(log_add_exp(
                self.log_values[zero_index],
                self.log_values[one_index],
            ));
        }
        Self::new(variables, log_values)
    }
}

/// Deterministic min-fill elimination plan computed before factor contraction.
///
/// `predicted_peak_factor_entries` is the maximum of the largest input factor
/// table and the largest planned bucket product table implied by induced width.
/// It is an exact allocation bound for the planned order, not a probabilistic
/// estimate; if it exceeds the caller's budget, planning fails before contraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorEliminationPlan {
    pub elimination_order: Vec<usize>,
    pub induced_width: usize,
    pub predicted_output_entries: usize,
    pub predicted_peak_factor_entries: usize,
    pub max_factor_entries: usize,
}
/// A preflight plan for exact cutset conditioning followed by min-fill.
///
/// The plan is independent of factor values and can be compared across candidate
/// separators before any conditioned branch is evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutsetFactorEliminationPlan {
    // The exact factor-scope sequence is the plan's structural identity. Numeric
    // factor values may change between executions, but scopes and order may not.
    scope_signature: Vec<Vec<usize>>,
    cutset_variables: Vec<usize>,
    cutset_assignments: usize,
    max_cutset_assignments: usize,
    max_factor_entries: usize,
    largest_input_factor_entries: usize,
    predicted_peak_factor_entries: usize,
    remaining_elimination_plan: FactorEliminationPlan,
}

impl CutsetFactorEliminationPlan {
    pub fn cutset_variables(&self) -> &[usize] {
        &self.cutset_variables
    }

    pub fn cutset_assignments(&self) -> usize {
        self.cutset_assignments
    }

    pub fn max_cutset_assignments(&self) -> usize {
        self.max_cutset_assignments
    }

    pub fn max_factor_entries(&self) -> usize {
        self.max_factor_entries
    }

    pub fn largest_input_factor_entries(&self) -> usize {
        self.largest_input_factor_entries
    }

    pub fn predicted_peak_factor_entries(&self) -> usize {
        self.predicted_peak_factor_entries
    }

    pub fn remaining_elimination_plan(&self) -> &FactorEliminationPlan {
        &self.remaining_elimination_plan
    }
}

/// One feasible candidate separator and its exact structural preflight.
#[derive(Debug, Clone, PartialEq)]
pub struct CutsetFrontierCandidate {
    /// Position of this separator in the input candidate list.
    pub candidate_index: usize,
    pub plan: CutsetFactorEliminationPlan,
}

/// A candidate separator rejected during preflight.
#[derive(Debug, Clone, PartialEq)]
pub struct CutsetFrontierRejection {
    /// Position of the rejected separator in the input candidate list.
    pub candidate_index: usize,
    pub error: FactorGraphError,
}

/// Feasible candidate plans, their non-dominated subset, and rejected candidates.
///
/// frontier_candidate_indices refer to positions in the original input list,
/// not positions in candidates. Exact metric ties retain the earliest input
/// candidate as the deterministic representative.
#[derive(Debug, Clone, PartialEq)]
pub struct CutsetPlanFrontierReport {
    /// Successfully preflighted candidates, in original input order.
    pub candidates: Vec<CutsetFrontierCandidate>,
    /// Original input indices of Pareto-optimal candidates, in input order.
    pub frontier_candidate_indices: Vec<usize>,
    /// Candidates that failed a work or factor-entry budget or validation.
    pub rejected_candidates: Vec<CutsetFrontierRejection>,
}

/// Plan exact binary-factor elimination without multiplying any factor tables.
///
/// The planner validates scopes and the elimination set, computes a deterministic
/// min-fill order, and calculates the largest input, bucket, or final output
/// table. Because an eliminated variable's incident-factor union contains exactly
/// its active primal-graph neighbours plus itself, the largest bucket has
/// `2^(width + 1)` entries. The final output factor spans every retained variable,
/// including disconnected retained components that the current API multiplies
/// into one factor. The plan is rejected before contraction if any table exceeds budget.
pub fn plan_min_fill_elimination(
    factor_scopes: &[Vec<usize>],
    eliminate: &[usize],
    max_factor_entries: usize,
) -> Result<FactorEliminationPlan, FactorGraphError> {
    if factor_scopes.is_empty() {
        return Err(FactorGraphError::EmptyFactorGraph);
    }
    let mut seen = BTreeSet::new();
    for variable in eliminate.iter().copied() {
        if !seen.insert(variable) {
            return Err(FactorGraphError::DuplicateEliminationVariable { variable });
        }
    }

    let mut largest_input_factor = 1usize;
    for scope in factor_scopes {
        for pair in scope.windows(2) {
            if pair[0] >= pair[1] {
                return Err(FactorGraphError::ScopeNotStrictlyAscending {
                    left: pair[0],
                    right: pair[1],
                });
            }
        }
        let entries = assignment_count(scope.len())?;
        check_budget(entries, max_factor_entries)?;
        largest_input_factor = largest_input_factor.max(entries);
    }

    let (elimination_order, induced_width) = min_fill_order_scopes(factor_scopes, eliminate)?;
    let largest_bucket = if induced_width == 0 {
        // Unary input factors already account for any represented zero-width
        // elimination. Completely absent variables produce scalar factors.
        1usize
    } else {
        let bucket_scope = induced_width
            .checked_add(1)
            .ok_or(FactorGraphError::IndexOverflow)?;
        assignment_count(bucket_scope)?
    };
    let retained_variables = factor_scopes
        .iter()
        .flatten()
        .copied()
        .filter(|variable| !seen.contains(variable))
        .collect::<BTreeSet<_>>();
    let predicted_output_entries = assignment_count(retained_variables.len())?;
    let predicted_peak_factor_entries = largest_input_factor
        .max(largest_bucket)
        .max(predicted_output_entries);
    check_budget(predicted_peak_factor_entries, max_factor_entries)?;

    Ok(FactorEliminationPlan {
        elimination_order,
        induced_width,
        predicted_output_entries,
        predicted_peak_factor_entries,
        max_factor_entries,
    })
}

/// Preflight exact cutset conditioning from factor scopes alone.
///
/// This validates both budgets, canonicalizes the cutset, computes the residual
/// min-fill plan, and accounts for the largest original input factor. It does not
/// inspect factor values or evaluate any cutset branch, so callers can rank several
/// separator candidates by branch count, residual width, and peak table size first.
pub fn plan_min_fill_cutset_elimination(
    factor_scopes: &[Vec<usize>],
    eliminate: &[usize],
    cutset: &[usize],
    max_factor_entries: usize,
    max_cutset_assignments: usize,
) -> Result<CutsetFactorEliminationPlan, FactorGraphError> {
    if factor_scopes.is_empty() {
        return Err(FactorGraphError::EmptyFactorGraph);
    }

    let elimination_set = eliminate.iter().copied().collect::<BTreeSet<_>>();
    if elimination_set.len() != eliminate.len() {
        let mut seen = BTreeSet::new();
        let duplicate = eliminate
            .iter()
            .copied()
            .find(|variable| !seen.insert(*variable))
            .ok_or(FactorGraphError::IndexOverflow)?;
        return Err(FactorGraphError::DuplicateEliminationVariable { variable: duplicate });
    }

    let mut largest_input_factor_entries = 1usize;
    for scope in factor_scopes {
        for pair in scope.windows(2) {
            if pair[0] >= pair[1] {
                return Err(FactorGraphError::ScopeNotStrictlyAscending {
                    left: pair[0],
                    right: pair[1],
                });
            }
        }
        let entries = assignment_count(scope.len())?;
        check_budget(entries, max_factor_entries)?;
        largest_input_factor_entries = largest_input_factor_entries.max(entries);
    }

    let mut canonical_cutset = cutset.to_vec();
    canonical_cutset.sort_unstable();
    for pair in canonical_cutset.windows(2) {
        if pair[0] == pair[1] {
            return Err(FactorGraphError::DuplicateCutsetVariable { variable: pair[0] });
        }
    }
    for variable in canonical_cutset.iter().copied() {
        if !elimination_set.contains(&variable) {
            return Err(FactorGraphError::CutsetVariableNotEliminated { variable });
        }
    }

    let cutset_assignments = assignment_count(canonical_cutset.len())?;
    if cutset_assignments > max_cutset_assignments {
        return Err(FactorGraphError::CutsetAssignmentBudgetExceeded {
            assignments: cutset_assignments,
            maximum: max_cutset_assignments,
        });
    }

    let cutset_set = canonical_cutset.iter().copied().collect::<BTreeSet<_>>();
    let remaining_eliminate = eliminate
        .iter()
        .copied()
        .filter(|variable| !cutset_set.contains(variable))
        .collect::<Vec<_>>();
    let reduced_scopes = factor_scopes
        .iter()
        .map(|scope| {
            scope
                .iter()
                .copied()
                .filter(|variable| !cutset_set.contains(variable))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let remaining_elimination_plan = plan_min_fill_elimination(
        &reduced_scopes,
        &remaining_eliminate,
        max_factor_entries,
    )?;
    let predicted_peak_factor_entries = largest_input_factor_entries
        .max(remaining_elimination_plan.predicted_peak_factor_entries);
    check_budget(predicted_peak_factor_entries, max_factor_entries)?;

    Ok(CutsetFactorEliminationPlan {
        scope_signature: factor_scopes.to_vec(),
        cutset_variables: canonical_cutset,
        cutset_assignments,
        max_cutset_assignments,
        max_factor_entries,
        largest_input_factor_entries,
        predicted_peak_factor_entries,
        remaining_elimination_plan,
    })
}
fn cutset_plan_dominates(
    left: &CutsetFrontierCandidate,
    right: &CutsetFrontierCandidate,
) -> bool {
    let left_width = left.plan.remaining_elimination_plan().induced_width;
    let right_width = right.plan.remaining_elimination_plan().induced_width;
    let left_branches = left.plan.cutset_assignments();
    let right_branches = right.plan.cutset_assignments();
    let left_peak = left.plan.predicted_peak_factor_entries();
    let right_peak = right.plan.predicted_peak_factor_entries();

    let no_worse = left_branches <= right_branches
        && left_width <= right_width
        && left_peak <= right_peak;
    let strictly_better = left_branches < right_branches
        || left_width < right_width
        || left_peak < right_peak;
    let exact_tie_precedes = left_branches == right_branches
        && left_width == right_width
        && left_peak == right_peak
        && left.candidate_index < right.candidate_index;
    no_worse && (strictly_better || exact_tie_precedes)
}

/// Preflight a supplied family of exact cutset candidates and return its Pareto frontier.
///
/// Each candidate is planned independently with the same factor-entry and
/// assignment budgets. Infeasible candidates are preserved in the report
/// rather than causing otherwise useful candidates to be discarded. Feasible
/// candidates are compared on three resources:
///
/// - exact number of conditioned branches (2^|cutset|);
/// - residual induced width after conditioning;
/// - predicted peak factor-table entries, including original inputs and the
///   residual plan's buckets/output.
///
/// A candidate is dominated only when another is no worse on all three axes
/// and strictly better on at least one. Exact metric ties keep the earliest
/// input candidate. This searches only the caller-supplied family; it does not
/// claim to find a globally optimal separator.
pub fn plan_min_fill_cutset_frontier(
    factor_scopes: &[Vec<usize>],
    eliminate: &[usize],
    candidate_cutsets: &[Vec<usize>],
    max_factor_entries: usize,
    max_cutset_assignments: usize,
) -> Result<CutsetPlanFrontierReport, FactorGraphError> {
    if candidate_cutsets.is_empty() {
        return Err(FactorGraphError::EmptyCutsetCandidateSet);
    }

    let mut candidates = Vec::new();
    let mut rejected_candidates = Vec::new();
    for (candidate_index, cutset) in candidate_cutsets.iter().enumerate() {
        match plan_min_fill_cutset_elimination(
            factor_scopes,
            eliminate,
            cutset,
            max_factor_entries,
            max_cutset_assignments,
        ) {
            Ok(plan) => candidates.push(CutsetFrontierCandidate {
                candidate_index,
                plan,
            }),
            Err(error) => rejected_candidates.push(CutsetFrontierRejection {
                candidate_index,
                error,
            }),
        }
    }

    let frontier_candidate_indices = candidates
        .iter()
        .filter(|candidate| {
            !candidates
                .iter()
                .any(|other| other.candidate_index != candidate.candidate_index
                    && cutset_plan_dominates(other, candidate))
        })
        .map(|candidate| candidate.candidate_index)
        .collect();

    Ok(CutsetPlanFrontierReport {
        candidates,
        frontier_candidate_indices,
        rejected_candidates,
    })
}

/// Telemetry for one completed log-domain min-fill contraction.
#[derive(Debug, Clone, PartialEq)]
pub struct LogFactorEliminationReport {
    pub factor: BinaryLogFactor,
    pub elimination_order: Vec<usize>,
    pub induced_width: usize,
    pub predicted_output_entries: usize,
    pub predicted_peak_factor_entries: usize,
    pub peak_factor_entries: usize,
    pub factor_multiplications: usize,
    pub variables_summed_out: usize,
}

/// Telemetry for exact log-domain cutset conditioning followed by min-fill.
///
/// The induced width describes the residual graph after conditioning the cutset.
/// Peak factor-entry counters include original input tables and residual
/// min-fill contractions. The result is the exact sum over all cutset assignments.
#[derive(Debug, Clone, PartialEq)]
pub struct CutsetLogFactorEliminationReport {
    pub factor: BinaryLogFactor,
    pub cutset_variables: Vec<usize>,
    pub cutset_assignments_evaluated: usize,
    pub max_cutset_assignments: usize,
    pub max_factor_entries: usize,
    /// Total log-table entries materialized while conditioning factor scopes.
    pub conditioned_factor_entries_written: usize,
    /// Pointwise log-add-exp operations used to combine conditioned branches.
    pub branch_output_entries_accumulated: usize,
    pub remaining_elimination_order: Vec<usize>,
    pub induced_width: usize,
    pub predicted_peak_factor_entries: usize,
    pub peak_factor_entries: usize,
    pub factor_multiplications: usize,
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    let minimum = left.min(right);
    maximum + (minimum - maximum).exp().ln_1p()
}

/// Telemetry for one completed deterministic min-fill contraction.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorEliminationReport {
    /// Product of remaining factors after all requested variables are summed out.
    pub factor: BinaryFactor,
    /// Deterministic min-fill order actually used.
    pub elimination_order: Vec<usize>,
    /// Maximum number of active neighbours at the point of elimination.
    pub induced_width: usize,
    /// Final output table size predicted by the preflight elimination plan.
    pub predicted_output_entries: usize,
    /// Peak table size predicted by the preflight elimination plan.
    pub predicted_peak_factor_entries: usize,
    /// Largest input or intermediate factor table observed.
    pub peak_factor_entries: usize,
    /// Number of binary factor multiplications performed.
    pub factor_multiplications: usize,
    /// Number of variables summed out.
    pub variables_summed_out: usize,
}

/// Eliminate the requested variables using deterministic min-fill ordering.
///
/// The routine is exact (ordinary floating-point sum/product), bounded by
/// max_factor_entries, and reports induced width and peak table size. Variables
/// absent from every input factor are independent fair binary variables; summing
/// one out contributes the correct scalar multiplier of two.
pub fn min_fill_sum_product(
    factors: &[BinaryFactor],
    eliminate: &[usize],
    max_factor_entries: usize,
) -> Result<FactorEliminationReport, FactorGraphError> {
    let scopes = factors
        .iter()
        .map(|factor| factor.variables.clone())
        .collect::<Vec<_>>();
    let plan = plan_min_fill_elimination(&scopes, eliminate, max_factor_entries)?;

    // Variable-to-factor incidence buckets avoid rescanning every factor for
    // every elimination step. A consumed factor is removed from each of its
    // remaining variable buckets before the reduced factor is inserted.
    let mut work = Vec::<Option<BinaryFactor>>::with_capacity(
        factors.len().saturating_add(eliminate.len()),
    );
    let mut buckets = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut peak_factor_entries = 1usize;
    for factor in factors {
        check_budget(factor.values.len(), max_factor_entries)?;
        peak_factor_entries = peak_factor_entries.max(factor.values.len());
        let factor_id = work.len();
        for variable in factor.variables.iter().copied() {
            buckets.entry(variable).or_default().insert(factor_id);
        }
        work.push(Some(factor.clone()));
    }

    let elimination_order = plan.elimination_order.clone();
    let induced_width = plan.induced_width;
    let mut factor_multiplications = 0usize;
    let mut variables_summed_out = 0usize;

    for variable in elimination_order.iter().copied() {
        let incident = buckets.remove(&variable).unwrap_or_default();
        if incident.is_empty() {
            // An unrepresented binary variable has a unit prior and sums to 2.
            work.push(Some(BinaryFactor::scalar(2.0)?));
            variables_summed_out += 1;
            continue;
        }

        let mut product: Option<BinaryFactor> = None;
        for factor_id in incident {
            let Some(factor) = work.get_mut(factor_id).and_then(Option::take) else {
                continue;
            };
            for other_variable in factor.variables.iter().copied() {
                if other_variable != variable {
                    if let Some(bucket) = buckets.get_mut(&other_variable) {
                        bucket.remove(&factor_id);
                    }
                }
            }

            product = Some(match product {
                None => factor,
                Some(previous) => {
                    factor_multiplications += 1;
                    previous.multiply(&factor, max_factor_entries)?
                }
            });
            if let Some(current) = &product {
                peak_factor_entries = peak_factor_entries.max(current.values.len());
            }
        }

        let product = product.ok_or(FactorGraphError::VariableNotInFactor { variable })?;
        let reduced = product.sum_out(variable, max_factor_entries)?;
        peak_factor_entries = peak_factor_entries.max(reduced.values.len());
        let factor_id = work.len();
        for remaining_variable in reduced.variables.iter().copied() {
            buckets.entry(remaining_variable).or_default().insert(factor_id);
        }
        work.push(Some(reduced));
        variables_summed_out += 1;
    }

    let mut remaining = work.into_iter().flatten();
    let mut result = remaining.next().ok_or(FactorGraphError::EmptyFactorGraph)?;
    for factor in remaining {
        result = result.multiply(&factor, max_factor_entries)?;
        factor_multiplications += 1;
        peak_factor_entries = peak_factor_entries.max(result.values.len());
    }

    Ok(FactorEliminationReport {
        factor: result,
        elimination_order,
        induced_width,
        predicted_output_entries: plan.predicted_output_entries,
        predicted_peak_factor_entries: plan.predicted_peak_factor_entries,
        peak_factor_entries,
        factor_multiplications,
        variables_summed_out,
    })
}

/// Exact log-domain sum-product elimination using deterministic min-fill.
///
/// This has the same scope/ordering/budget contract as `min_fill_sum_product`,
/// but factor multiplication is log addition and variable elimination is
/// log-sum-exp. Use it when a long product of individually valid likelihoods
/// could underflow even after local factor scaling.
pub fn min_fill_log_sum_product(
    factors: &[BinaryLogFactor],
    eliminate: &[usize],
    max_factor_entries: usize,
) -> Result<LogFactorEliminationReport, FactorGraphError> {
    let scopes = factors
        .iter()
        .map(|factor| factor.variables.clone())
        .collect::<Vec<_>>();
    let plan = plan_min_fill_elimination(&scopes, eliminate, max_factor_entries)?;
    min_fill_log_sum_product_with_plan(factors, &plan)
}

/// Reuse a checked plan when factor scopes are invariant across numeric queries.
fn min_fill_log_sum_product_with_plan(
    factors: &[BinaryLogFactor],
    plan: &FactorEliminationPlan,
) -> Result<LogFactorEliminationReport, FactorGraphError> {
    let mut work = Vec::<Option<BinaryLogFactor>>::with_capacity(
        factors.len().saturating_add(plan.elimination_order.len()),
    );
    let mut buckets = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut peak_factor_entries = 1usize;
    for factor in factors {
        check_budget(factor.log_values.len(), plan.max_factor_entries)?;
        peak_factor_entries = peak_factor_entries.max(factor.log_values.len());
        let factor_id = work.len();
        for variable in factor.variables.iter().copied() {
            buckets.entry(variable).or_default().insert(factor_id);
        }
        work.push(Some(factor.clone()));
    }

    let elimination_order = plan.elimination_order.clone();
    let induced_width = plan.induced_width;
    let mut factor_multiplications = 0usize;
    let mut variables_summed_out = 0usize;

    for variable in elimination_order.iter().copied() {
        let incident = buckets.remove(&variable).unwrap_or_default();
        if incident.is_empty() {
            work.push(Some(BinaryLogFactor::scalar(std::f64::consts::LN_2)?));
            variables_summed_out += 1;
            continue;
        }

        let mut product: Option<BinaryLogFactor> = None;
        for factor_id in incident {
            let Some(factor) = work.get_mut(factor_id).and_then(Option::take) else {
                continue;
            };
            for other_variable in factor.variables.iter().copied() {
                if other_variable != variable {
                    if let Some(bucket) = buckets.get_mut(&other_variable) {
                        bucket.remove(&factor_id);
                    }
                }
            }

            product = Some(match product {
                None => factor,
                Some(previous) => {
                    factor_multiplications += 1;
                    previous.multiply(&factor, plan.max_factor_entries)?
                }
            });
            if let Some(current) = &product {
                peak_factor_entries = peak_factor_entries.max(current.log_values.len());
            }
        }

        let product = product.ok_or(FactorGraphError::VariableNotInFactor { variable })?;
        let reduced = product.sum_out(variable, plan.max_factor_entries)?;
        peak_factor_entries = peak_factor_entries.max(reduced.log_values.len());
        let factor_id = work.len();
        for remaining_variable in reduced.variables.iter().copied() {
            buckets.entry(remaining_variable).or_default().insert(factor_id);
        }
        work.push(Some(reduced));
        variables_summed_out += 1;
    }

    let mut remaining = work.into_iter().flatten();
    let mut result = remaining.next().ok_or(FactorGraphError::EmptyFactorGraph)?;
    for factor in remaining {
        result = result.multiply(&factor, plan.max_factor_entries)?;
        factor_multiplications += 1;
        peak_factor_entries = peak_factor_entries.max(result.log_values.len());
    }

    Ok(LogFactorEliminationReport {
        factor: result,
        elimination_order,
        induced_width,
        predicted_output_entries: plan.predicted_output_entries,
        predicted_peak_factor_entries: plan.predicted_peak_factor_entries,
        peak_factor_entries,
        factor_multiplications,
        variables_summed_out,
    })
}

/// Exact bounded-memory sum-product by cutset conditioning followed by min-fill.
///
/// Every variable in the cutset must also appear in the elimination set. The
/// function enumerates all 2^k assignments, conditions each input factor,
/// eliminates the remaining variables with one preflighted min-fill plan, and
/// log-adds the branch outputs. This is exact, not approximate: it trades
/// 2^k branches for a potentially much smaller residual induced width.
///
/// The cutset can include variables absent from all factors; summing over their
/// two conditioned branches correctly contributes a factor of two. If the
/// residual plan or any input factor exceeds max_factor_entries, it fails
/// before evaluating a branch. max_cutset_assignments independently bounds
/// the number of branches.
pub fn min_fill_log_sum_product_with_cutset(
    factors: &[BinaryLogFactor],
    eliminate: &[usize],
    cutset: &[usize],
    max_factor_entries: usize,
    max_cutset_assignments: usize,
) -> Result<CutsetLogFactorEliminationReport, FactorGraphError> {
    let factor_scopes = factors
        .iter()
        .map(|factor| factor.variables.clone())
        .collect::<Vec<_>>();
    let plan = plan_min_fill_cutset_elimination(
        &factor_scopes,
        eliminate,
        cutset,
        max_factor_entries,
        max_cutset_assignments,
    )?;
    min_fill_log_sum_product_with_cutset_plan(factors, &plan)
}

/// Reuse a preflighted cutset plan across different numeric factors with identical
/// ordered scopes. The scope signature is checked before any branch is evaluated.
pub fn min_fill_log_sum_product_with_cutset_plan(
    factors: &[BinaryLogFactor],
    plan: &CutsetFactorEliminationPlan,
) -> Result<CutsetLogFactorEliminationReport, FactorGraphError> {
    if factors.len() != plan.scope_signature.len()
        || factors
            .iter()
            .zip(plan.scope_signature.iter())
            .any(|(factor, scope)| factor.variables != *scope)
    {
        return Err(FactorGraphError::FactorScopesDoNotMatchCutsetPlan);
    }
    let branch_count = plan.cutset_assignments;
    let canonical_cutset = &plan.cutset_variables;
    let residual_plan = &plan.remaining_elimination_plan;
    let predicted_peak_factor_entries = plan.predicted_peak_factor_entries;
    let original_peak = plan.largest_input_factor_entries;
    let mut accumulated: Option<BinaryLogFactor> = None;
    let mut peak_factor_entries = original_peak;
    let mut factor_multiplications = 0usize;
    let mut conditioned_factor_entries_written = 0usize;
    let mut branch_output_entries_accumulated = 0usize;

    for branch in 0..branch_count {
        let mut conditioned_factors = Vec::with_capacity(factors.len());
        let cutset_assignment = canonical_cutset
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, variable)| (variable, ((branch >> slot) & 1) != 0))
            .collect::<Vec<_>>();
        for factor in factors {
            let conditioned = factor.condition_many(&cutset_assignment)?;
            if conditioned.variables != factor.variables {
                conditioned_factor_entries_written = conditioned_factor_entries_written
                    .checked_add(conditioned.log_values.len())
                    .ok_or(FactorGraphError::IndexOverflow)?;
            }
            conditioned_factors.push(conditioned);
        }

        let report = min_fill_log_sum_product_with_plan(&conditioned_factors, residual_plan)?;
        if let Some(previous) = &accumulated {
            if previous.variables != report.factor.variables {
                return Err(FactorGraphError::CutsetBranchScopeMismatch);
            }
        }
        peak_factor_entries = peak_factor_entries.max(report.peak_factor_entries);
        factor_multiplications = factor_multiplications
            .checked_add(report.factor_multiplications)
            .ok_or(FactorGraphError::IndexOverflow)?;

        match &mut accumulated {
            None => accumulated = Some(report.factor),
            Some(previous) => {
                branch_output_entries_accumulated = branch_output_entries_accumulated
                    .checked_add(report.factor.log_values.len())
                    .ok_or(FactorGraphError::IndexOverflow)?;
                for (left, right) in previous
                    .log_values
                    .iter_mut()
                    .zip(report.factor.log_values.iter().copied())
                {
                    *left = log_add_exp(*left, right);
                }
            }
        }
    }

    let factor = accumulated.ok_or(FactorGraphError::CutsetAssignmentBudgetExceeded {
        assignments: branch_count,
        maximum: max_cutset_assignments,
    })?;

    Ok(CutsetLogFactorEliminationReport {
        factor,
        cutset_variables: canonical_cutset.to_vec(),
        cutset_assignments_evaluated: branch_count,
        max_cutset_assignments: plan.max_cutset_assignments,
        max_factor_entries: plan.remaining_elimination_plan.max_factor_entries,
        conditioned_factor_entries_written,
        branch_output_entries_accumulated,
        remaining_elimination_order: residual_plan.elimination_order.clone(),
        induced_width: residual_plan.induced_width,
        predicted_peak_factor_entries,
        peak_factor_entries,
        factor_multiplications,
    })
}
/// Compute the deterministic min-fill order and its induced width.
///
/// A lazy min-priority queue keeps the planner from rescanning every active
/// variable on every elimination step. Eliminating v only changes adjacency
/// among v's current neighbours, so only those neighbours need fresh scores.
/// Stale heap entries are discarded by per-variable generation counters.
#[cfg(test)]
fn min_fill_order(
    factors: &[BinaryFactor],
    eliminate: &[usize],
) -> Result<(Vec<usize>, usize), FactorGraphError> {
    let scopes = factors
        .iter()
        .map(|factor| factor.variables.clone())
        .collect::<Vec<_>>();
    min_fill_order_scopes(&scopes, eliminate)
}

fn min_fill_order_scopes(
    factor_scopes: &[Vec<usize>],
    eliminate: &[usize],
) -> Result<(Vec<usize>, usize), FactorGraphError> {
    let mut adjacency: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for variable in eliminate.iter().copied() {
        adjacency.entry(variable).or_default();
    }
    for scope in factor_scopes {
        for variable in scope.iter().copied() {
            adjacency.entry(variable).or_default();
        }
        for (left_index, a) in scope.iter().copied().enumerate() {
            for b in scope.iter().copied().skip(left_index + 1) {
                adjacency.entry(a).or_default().insert(b);
                adjacency.entry(b).or_default().insert(a);
            }
        }
    }

    let mut active: BTreeSet<usize> = eliminate.iter().copied().collect();
    let mut versions = BTreeMap::<usize, u64>::new();
    let mut queue = BinaryHeap::<Reverse<(usize, usize, usize, u64)>>::new();
    for variable in active.iter().copied() {
        versions.insert(variable, 0);
        let (fill_edges, degree) = fill_score(variable, &adjacency);
        queue.push(Reverse((fill_edges, degree, variable, 0)));
    }

    let mut order = Vec::with_capacity(active.len());
    let mut induced_width = 0usize;

    while !active.is_empty() {
        let variable = loop {
            let Reverse((_, degree, variable, version)) =
                queue.pop().ok_or(FactorGraphError::EmptyFactorGraph)?;
            if active.contains(&variable)
                && versions.get(&variable).copied() == Some(version)
            {
                induced_width = induced_width.max(degree);
                break variable;
            }
        };

        let neighbours = adjacency
            .get(&variable)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|neighbour| adjacency.contains_key(neighbour))
            .collect::<Vec<_>>();

        // Fill edges among N(variable) can also change the fill score of a
        // non-neighbour that is adjacent to two or more members of N(variable).
        // Track those affected candidates before mutating the graph.
        let mut affected = BTreeSet::new();
        let mut neighbour_hits = BTreeMap::<usize, usize>::new();
        for neighbour in &neighbours {
            if active.contains(neighbour) {
                affected.insert(*neighbour);
            }
            if let Some(edges) = adjacency.get(neighbour) {
                for candidate in edges.iter().copied() {
                    if candidate != variable && active.contains(&candidate) {
                        *neighbour_hits.entry(candidate).or_default() += 1;
                    }
                }
            }
        }
        for (candidate, hits) in neighbour_hits {
            if hits >= 2 {
                affected.insert(candidate);
            }
        }

        // Elimination makes the active neighbours a clique.
        for (left_index, a) in neighbours.iter().copied().enumerate() {
            for b in neighbours.iter().copied().skip(left_index + 1) {
                adjacency.entry(a).or_default().insert(b);
                adjacency.entry(b).or_default().insert(a);
            }
        }
        for neighbour in &neighbours {
            if let Some(edges) = adjacency.get_mut(neighbour) {
                edges.remove(&variable);
            }
        }
        adjacency.remove(&variable);
        active.remove(&variable);
        order.push(variable);

        // Refresh every remaining candidate whose degree or fill score changed.
        for candidate in affected {
            if !active.contains(&candidate) {
                continue;
            }
            let version = versions
                .get(&candidate)
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or(FactorGraphError::IndexOverflow)?;
            versions.insert(candidate, version);
            let (fill_edges, degree) = fill_score(candidate, &adjacency);
            queue.push(Reverse((fill_edges, degree, candidate, version)));
        }
    }

    Ok((order, induced_width))
}

fn fill_score(
    variable: usize,
    adjacency: &BTreeMap<usize, BTreeSet<usize>>,
) -> (usize, usize) {
    let neighbours = adjacency
        .get(&variable)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|neighbour| adjacency.contains_key(neighbour))
        .collect::<Vec<_>>();
    let mut fill_edges = 0usize;
    for (left_index, left) in neighbours.iter().copied().enumerate() {
        for right in neighbours.iter().copied().skip(left_index + 1) {
            if !adjacency[&left].contains(&right) {
                fill_edges += 1;
            }
        }
    }
    (fill_edges, neighbours.len())
}
fn sorted_union(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut output = Vec::with_capacity(left.len() + right.len());
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while left_index < left.len() || right_index < right.len() {
        let next = match (left.get(left_index), right.get(right_index)) {
            (Some(&a), Some(&b)) if a < b => {
                left_index += 1;
                a
            }
            (Some(&a), Some(&b)) if b < a => {
                right_index += 1;
                b
            }
            (Some(&a), Some(_)) => {
                left_index += 1;
                right_index += 1;
                a
            }
            (Some(&a), None) => {
                left_index += 1;
                a
            }
            (None, Some(&b)) => {
                right_index += 1;
                b
            }
            (None, None) => break,
        };
        output.push(next);
    }
    output
}

fn assignment_count(variable_count: usize) -> Result<usize, FactorGraphError> {
    1usize
        .checked_shl(variable_count as u32)
        .ok_or(FactorGraphError::ScopeTooWide {
            variables: variable_count,
        })
}

fn check_budget(entries: usize, limit: usize) -> Result<(), FactorGraphError> {
    if entries > limit {
        Err(FactorGraphError::IntermediateTooLarge { entries, limit })
    } else {
        Ok(())
    }
}

fn project_assignment(
    assignment: usize,
    joint_scope: &[usize],
    local_scope: &[usize],
) -> Result<usize, FactorGraphError> {
    let mut local = 0usize;
    for (local_slot, variable) in local_scope.iter().copied().enumerate() {
        let joint_slot = joint_scope
            .binary_search(&variable)
            .map_err(|_| FactorGraphError::IndexOverflow)?;
        if ((assignment >> joint_slot) & 1) != 0 {
            local |= 1usize << local_slot;
        }
    }
    Ok(local)
}

fn insert_assignment_bit(
    assignment: usize,
    bit_slot: usize,
    bit: bool,
) -> Result<usize, FactorGraphError> {
    let lower_mask = (1usize << bit_slot)
        .checked_sub(1)
        .ok_or(FactorGraphError::IndexOverflow)?;
    let low = assignment & lower_mask;
    let high = assignment & !lower_mask;
    let inserted = high
        .checked_shl(1)
        .ok_or(FactorGraphError::IndexOverflow)?;
    Ok(inserted | low | (usize::from(bit) << bit_slot))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() <= 1e-12, "{left} != {right}");
    }

    #[test]
    fn preflight_plan_matches_observed_peak_and_rejects_before_contraction() {
        let chain_scopes = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
        let plan = plan_min_fill_elimination(&chain_scopes, &[0, 1, 2], 8).unwrap();
        assert_eq!(plan.induced_width, 1);
        assert_eq!(plan.predicted_output_entries, 2);
        assert_eq!(plan.predicted_peak_factor_entries, 4);
        assert_eq!(plan.max_factor_entries, 8);

        let factors = vec![
            BinaryFactor::new(vec![0, 1], vec![0.9, 0.1, 0.2, 0.8]).unwrap(),
            BinaryFactor::new(vec![1, 2], vec![0.7, 0.3, 0.4, 0.6]).unwrap(),
            BinaryFactor::new(vec![2, 3], vec![0.6, 0.4, 0.3, 0.7]).unwrap(),
        ];
        let report = min_fill_sum_product(&factors, &[0, 1, 2], 8).unwrap();
        assert_eq!(report.predicted_output_entries, 2);
        assert_eq!(report.factor.values().len(), report.predicted_output_entries);
        assert_eq!(report.predicted_peak_factor_entries, 4);
        assert_eq!(report.peak_factor_entries, plan.predicted_peak_factor_entries);

        assert!(matches!(
            plan_min_fill_elimination(&chain_scopes, &[0, 1, 2], 3),
            Err(FactorGraphError::IntermediateTooLarge { entries: 4, limit: 3 })
        ));
        assert!(matches!(
            min_fill_sum_product(&factors, &[0, 1, 2], 3),
            Err(FactorGraphError::IntermediateTooLarge { entries: 4, limit: 3 })
        ));
    }

    #[test]
    fn preflight_plan_handles_unary_and_unrepresented_variables() {
        let unary = plan_min_fill_elimination(&[vec![0]], &[0], 2).unwrap();
        assert_eq!(unary.induced_width, 0);
        assert_eq!(unary.predicted_output_entries, 1);
        assert_eq!(unary.predicted_peak_factor_entries, 2);

        let scalar = plan_min_fill_elimination(&[vec![]], &[9], 1).unwrap();
        assert_eq!(scalar.induced_width, 0);
        assert_eq!(scalar.predicted_output_entries, 1);
        assert_eq!(scalar.predicted_peak_factor_entries, 1);
    }

    #[test]
    fn log_contraction_reports_preflight_peak() {
        let factors = vec![
            BinaryLogFactor::new(vec![0, 1], vec![0.0, -1.0, -2.0, -3.0]).unwrap(),
            BinaryLogFactor::new(vec![1, 2], vec![0.0, -1.0, -2.0, -3.0]).unwrap(),
        ];
        let report = min_fill_log_sum_product(&factors, &[0, 1], 4).unwrap();
        assert_eq!(report.induced_width, 1);
        assert_eq!(report.predicted_output_entries, 2);
        assert_eq!(report.factor.log_values().len(), report.predicted_output_entries);
        assert_eq!(report.predicted_peak_factor_entries, 4);
        assert_eq!(report.peak_factor_entries, 4);
    }

    #[test]
    fn preflight_plan_accounts_for_the_final_retained_scope_product() {
        let disconnected_retained = vec![vec![0], vec![1]];
        let plan = plan_min_fill_elimination(&disconnected_retained, &[9], 4).unwrap();
        assert_eq!(plan.induced_width, 0);
        assert_eq!(plan.predicted_output_entries, 4);
        assert_eq!(plan.predicted_peak_factor_entries, 4);
        assert!(matches!(
            plan_min_fill_elimination(&disconnected_retained, &[9], 3),
            Err(FactorGraphError::IntermediateTooLarge { entries: 4, limit: 3 })
        ));
    }
    #[test]
    fn preflight_plan_rejects_invalid_scopes_and_duplicate_elimination() {
        assert!(matches!(
            plan_min_fill_elimination(&[vec![1, 0]], &[0], 4),
            Err(FactorGraphError::ScopeNotStrictlyAscending { .. })
        ));
        assert!(matches!(
            plan_min_fill_elimination(&[vec![0]], &[0, 0], 4),
            Err(FactorGraphError::DuplicateEliminationVariable { variable: 0 })
        ));
    }

    #[test]
    fn min_fill_sum_product_matches_brute_force_on_binary_chain() {
        // f0(x0,x1) with x0 as the low bit; f1(x1,x2) with x1 as low bit.
        let f0 = BinaryFactor::new(vec![0, 1], vec![0.9, 0.1, 0.2, 0.8]).unwrap();
        let f1 = BinaryFactor::new(vec![1, 2], vec![0.7, 0.3, 0.4, 0.6]).unwrap();
        let report = min_fill_sum_product(&[f0.clone(), f1.clone()], &[0, 1], 8).unwrap();

        let mut expected = [0.0; 2]; // retain x2
        for (x2, expected_value) in expected.iter_mut().enumerate() {
            for x0 in 0..2 {
                for x1 in 0..2 {
                    let left_index = x0 | (x1 << 1);
                    let right_index = x1 | (x2 << 1);
                    *expected_value += f0.values[left_index] * f1.values[right_index];
                }
            }
        }
        close(report.factor.values[0], expected[0]);
        close(report.factor.values[1], expected[1]);
        assert_eq!(report.factor.variables, vec![2]);
        assert_eq!(report.induced_width, 1);
        assert_eq!(report.peak_factor_entries, 4);
        assert_eq!(report.variables_summed_out, 2);
    }

    #[test]
    fn min_fill_order_is_deterministic_and_avoids_chain_cliques() {
        let factors = vec![
            BinaryFactor::new(vec![0, 1], vec![1.0; 4]).unwrap(),
            BinaryFactor::new(vec![1, 2], vec![1.0; 4]).unwrap(),
            BinaryFactor::new(vec![2, 3], vec![1.0; 4]).unwrap(),
            BinaryFactor::new(vec![3, 4], vec![1.0; 4]).unwrap(),
        ];
        let first = min_fill_sum_product(&factors, &[0, 1, 2, 3], 16).unwrap();
        let second = min_fill_sum_product(&factors, &[0, 1, 2, 3], 16).unwrap();
        assert_eq!(first.elimination_order, second.elimination_order);
        assert_eq!(first.induced_width, 1);
        assert_eq!(first.factor.variables, vec![4]);
        assert_eq!(first.factor.values, vec![16.0, 16.0]);
    }

    fn reference_min_fill_order(
        factors: &[BinaryFactor],
        eliminate: &[usize],
    ) -> (Vec<usize>, usize) {
        let mut adjacency = BTreeMap::<usize, BTreeSet<usize>>::new();
        for variable in eliminate.iter().copied() {
            adjacency.entry(variable).or_default();
        }
        for factor in factors {
            for variable in factor.variables.iter().copied() {
                adjacency.entry(variable).or_default();
            }
            for (left_index, left) in factor.variables.iter().copied().enumerate() {
                for right in factor.variables.iter().copied().skip(left_index + 1) {
                    adjacency.entry(left).or_default().insert(right);
                    adjacency.entry(right).or_default().insert(left);
                }
            }
        }

        let mut active = eliminate.iter().copied().collect::<BTreeSet<_>>();
        let mut order = Vec::with_capacity(active.len());
        let mut width = 0usize;
        while !active.is_empty() {
            let (variable, degree) = active
                .iter()
                .copied()
                .map(|variable| {
                    let (fill, degree) = fill_score(variable, &adjacency);
                    ((fill, degree, variable), degree)
                })
                .min_by_key(|(score, _)| *score)
                .map(|((_, _, variable), degree)| (variable, degree))
                .unwrap();
            width = width.max(degree);
            let neighbours = adjacency.get(&variable).cloned().unwrap_or_default();
            for (left_index, left) in neighbours.iter().copied().enumerate() {
                for right in neighbours.iter().copied().skip(left_index + 1) {
                    adjacency.entry(left).or_default().insert(right);
                    adjacency.entry(right).or_default().insert(left);
                }
            }
            for neighbour in &neighbours {
                if let Some(edges) = adjacency.get_mut(neighbour) {
                    edges.remove(&variable);
                }
            }
            adjacency.remove(&variable);
            active.remove(&variable);
            order.push(variable);
        }
        (order, width)
    }

    #[test]
    fn lazy_min_fill_queue_matches_full_rescan_reference() {
        for variable_count in 4..=10 {
            for seed in 0..12 {
                let mut factors = Vec::new();
                for left in 0..variable_count {
                    for right in left + 1..variable_count {
                        let selector =
                            (left * 31 + right * 17 + seed * 13 + variable_count * 7) % 11;
                        if selector < 4 {
                            factors.push(
                                BinaryFactor::new(vec![left, right], vec![1.0; 4]).unwrap(),
                            );
                        }
                    }
                }
                if factors.is_empty() {
                    factors.push(BinaryFactor::scalar(1.0).unwrap());
                }
                let eliminate = (0..variable_count).collect::<Vec<_>>();
                let reference = reference_min_fill_order(&factors, &eliminate);
                let actual = min_fill_order(&factors, &eliminate).unwrap();
                assert_eq!(
                    actual, reference,
                    "lazy min-fill diverged for n={variable_count}, seed={seed}"
                );
            }
        }
    }

    #[test]
    fn min_fill_respects_memory_budget_before_large_allocation() {
        let factor = BinaryFactor::new(vec![0, 1, 2], vec![1.0; 8]).unwrap();
        assert!(matches!(
            min_fill_sum_product(&[factor], &[0], 4),
            Err(FactorGraphError::IntermediateTooLarge { entries: 8, limit: 4 })
        ));
    }

    #[test]
    fn log_sum_product_matches_probability_domain_on_moderate_chain() {
        let probabilities = [
            BinaryFactor::new(vec![0, 1], vec![0.9, 0.1, 0.2, 0.8]).unwrap(),
            BinaryFactor::new(vec![1, 2], vec![0.7, 0.3, 0.4, 0.6]).unwrap(),
        ];
        let log_factors = probabilities
            .iter()
            .map(|factor| {
                BinaryLogFactor::new(
                    factor.variables.clone(),
                    factor.values.iter().map(|value| value.ln()).collect(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let probability_report = min_fill_sum_product(&probabilities, &[0, 1], 8).unwrap();
        let log_report = min_fill_log_sum_product(&log_factors, &[0, 1], 8).unwrap();
        assert_eq!(probability_report.factor.variables, log_report.factor.variables);
        for (probability, log_probability) in probability_report
            .factor
            .values
            .iter()
            .copied()
            .zip(log_report.factor.log_values.iter().copied())
        {
            close(probability, log_probability.exp());
        }
        assert_eq!(probability_report.elimination_order, log_report.elimination_order);
        assert_eq!(probability_report.induced_width, log_report.induced_width);
    }

    #[test]
    fn log_sum_product_preserves_products_below_probability_underflow() {
        let factors = vec![
            BinaryLogFactor::scalar(-1_000.0).unwrap(),
            BinaryLogFactor::scalar(-1_000.0).unwrap(),
            BinaryLogFactor::scalar(-1_000.0).unwrap(),
        ];
        let report = min_fill_log_sum_product(&factors, &[], 1).unwrap();
        assert!(report.factor.variables.is_empty());
        assert_eq!(report.factor.log_values, vec![-3_000.0]);
        assert_eq!(report.factor.log_values[0].exp(), 0.0);
    }

    #[test]
    fn log_sum_out_uses_stable_logsumexp_for_tiny_weights() {
        let factor = BinaryLogFactor::new(vec![0], vec![-1_000.0, -1_001.0]).unwrap();
        let report = min_fill_log_sum_product(&[factor], &[0], 2).unwrap();
        let expected = -1_000.0 + (1.0 + (-1.0_f64).exp()).ln();
        assert_eq!(report.factor.variables, Vec::<usize>::new());
        assert!((report.factor.log_values[0] - expected).abs() <= 1e-12);
    }

    #[test]
    fn randomized_sparse_graphs_match_independent_full_joint_enumeration() {
        // Exercise arbitrary overlapping scopes, retained variables, disconnected
        // eliminated variables, and exact zero-support entries. The oracle enumerates
        // the full joint assignment directly and does not use the production
        // projection, multiplication, or elimination helpers.
        fn next_random(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *state
        }

        fn project(full_assignment: usize, scope: &[usize]) -> usize {
            let mut local = 0usize;
            for (slot, variable) in scope.iter().copied().enumerate() {
                if ((full_assignment >> variable) & 1) != 0 {
                    local |= 1usize << slot;
                }
            }
            local
        }

        for variable_count in 2..=7 {
            for seed in 0..16u64 {
                let eliminate = (0..variable_count)
                    .filter(|variable| (*variable as u64 + seed) % 3 != 0)
                    .collect::<Vec<_>>();
                let retained = (0..variable_count)
                    .filter(|variable| !eliminate.contains(variable))
                    .collect::<Vec<_>>();
                let mut random = seed ^ ((variable_count as u64) << 32);
                let mut factors = Vec::<BinaryFactor>::new();

                // Ensure every retained variable appears in the output scope.
                for variable in &retained {
                    let values = (0..2)
                        .map(|_| 0.15 + (next_random(&mut random) % 8) as f64 * 0.1)
                        .collect();
                    factors.push(BinaryFactor::new(vec![*variable], values).unwrap());
                }

                // Generate overlapping pair and triple factors. Some cells are
                // exact zeros, which become -infinity in the log-domain copy.
                for left in 0..variable_count {
                    for right in left + 1..variable_count {
                        if next_random(&mut random) % 5 < 2 {
                            let values = (0..4)
                                .map(|entry| {
                                    if (entry as u64 + next_random(&mut random)) % 13 == 0 {
                                        0.0
                                    } else {
                                        0.1 + (next_random(&mut random) % 9) as f64 * 0.1
                                    }
                                })
                                .collect();
                            factors.push(
                                BinaryFactor::new(vec![left, right], values).unwrap(),
                            );
                        }
                    }
                }
                if variable_count >= 3 {
                    for first in 0..variable_count {
                        for second in first + 1..variable_count {
                            for third in second + 1..variable_count {
                                if next_random(&mut random) % 19 == 0 {
                                    let values = (0..8)
                                        .map(|entry| {
                                            if (entry as u64 + next_random(&mut random)) % 17 == 0 {
                                                0.0
                                            } else {
                                                0.1 + (next_random(&mut random) % 9) as f64 * 0.1
                                            }
                                        })
                                        .collect();
                                    factors.push(
                                        BinaryFactor::new(
                                            vec![first, second, third],
                                            values,
                                        )
                                        .unwrap(),
                                    );
                                }
                            }
                        }
                    }
                }

                // Disconnected eliminated variables are intentionally left out
                // for some seeds; summing one must contribute an exact factor 2.
                let mut expected = vec![0.0; 1usize << retained.len()];
                for full_assignment in 0..(1usize << variable_count) {
                    let mut joint_weight = 1.0;
                    for factor in &factors {
                        let local_index = project(full_assignment, factor.variables());
                        joint_weight *= factor.value_at(local_index).unwrap();
                    }
                    let retained_index = project(full_assignment, &retained);
                    expected[retained_index] += joint_weight;
                }

                let probability_report = min_fill_sum_product(
                    &factors,
                    &eliminate,
                    1usize << variable_count,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "probability contraction failed: n={variable_count}, seed={seed}: {error}"
                    )
                });
                assert_eq!(
                    probability_report.factor.variables(),
                    retained,
                    "probability scope mismatch: n={variable_count}, seed={seed}"
                );
                assert_eq!(probability_report.factor.values().len(), expected.len());
                assert_eq!(probability_report.predicted_output_entries, expected.len());
                assert_eq!(
                    probability_report.predicted_peak_factor_entries,
                    probability_report.peak_factor_entries
                );
                for (index, (actual, expected)) in probability_report
                    .factor
                    .values()
                    .iter()
                    .copied()
                    .zip(expected.iter().copied())
                    .enumerate()
                {
                    close(actual, expected);
                    assert!(actual.is_finite() && actual >= 0.0,
                        "invalid probability output: n={variable_count}, seed={seed}, index={index}");
                }

                let log_factors = factors
                    .iter()
                    .map(|factor| {
                        BinaryLogFactor::new(
                            factor.variables().to_vec(),
                            factor.values().iter().map(|value| {
                                if *value == 0.0 { f64::NEG_INFINITY } else { value.ln() }
                            }).collect(),
                        ).unwrap()
                    })
                    .collect::<Vec<_>>();
                let log_report = min_fill_log_sum_product(
                    &log_factors,
                    &eliminate,
                    1usize << variable_count,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "log contraction failed: n={variable_count}, seed={seed}: {error}"
                    )
                });
                assert_eq!(
                    log_report.factor.variables(),
                    retained,
                    "log scope mismatch: n={variable_count}, seed={seed}"
                );
                assert_eq!(log_report.factor.log_values().len(), expected.len());
                assert_eq!(log_report.predicted_output_entries, expected.len());
                assert_eq!(
                    log_report.predicted_peak_factor_entries,
                    log_report.peak_factor_entries
                );
                for (index, (actual_log, expected)) in log_report
                    .factor
                    .log_values()
                    .iter()
                    .copied()
                    .zip(expected.iter().copied())
                    .enumerate()
                {
                    if expected == 0.0 {
                        assert_eq!(
                            actual_log,
                            f64::NEG_INFINITY,
                            "zero-support mismatch: n={variable_count}, seed={seed}, index={index}"
                        );
                    } else {
                        close(actual_log.exp(), expected);
                    }
                }
                assert_eq!(probability_report.elimination_order, log_report.elimination_order);
                assert_eq!(probability_report.induced_width, log_report.induced_width);
                assert_eq!(probability_report.peak_factor_entries, log_report.peak_factor_entries);
            }
        }
    }

    #[test]
    fn multi_variable_conditioning_matches_repeated_single_variable_conditioning() {
        let factor = BinaryLogFactor::new(
            vec![1, 3, 5, 7],
            (0..16)
                .map(|index| if index == 9 { f64::NEG_INFINITY } else { (index as f64 + 1.0).ln() })
                .collect(),
        ).unwrap();

        for mask in 0..16usize {
            let assignments = [1usize, 3, 5, 7]
                .into_iter()
                .enumerate()
                .map(|(slot, variable)| (variable, ((mask >> slot) & 1) != 0))
                .collect::<Vec<_>>();
            let repeated = assignments.iter().try_fold(
                factor.clone(),
                |current, (variable, value)| current.condition(*variable, *value),
            ).unwrap();
            let direct = factor.condition_many(&assignments).unwrap();
            assert_eq!(direct, repeated, "conditioning mask {mask:04b}");
        }

        assert!(matches!(
            factor.condition_many(&[(3, true), (1, false)]),
            Err(FactorGraphError::ScopeNotStrictlyAscending { left: 3, right: 1 })
        ));
        assert!(matches!(
            factor.condition_many(&[(1, true), (1, false)]),
            Err(FactorGraphError::ScopeNotStrictlyAscending { left: 1, right: 1 })
        ));
        assert_eq!(factor.condition_many(&[(99, true)]).unwrap(), factor);
    }

    #[test]
    fn log_factor_conditioning_removes_a_variable_and_preserves_scope_bit_order() {
        let factor = BinaryLogFactor::new(
            vec![1, 3],
            vec![0.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln(), 4.0_f64.ln()],
        ).unwrap();

        let variable_three_on = factor.condition(3, true).unwrap();
        assert_eq!(variable_three_on.variables(), &[1]);
        assert_eq!(variable_three_on.log_values(), &[3.0_f64.ln(), 4.0_f64.ln()]);

        let variable_one_off = factor.condition(1, false).unwrap();
        assert_eq!(variable_one_off.variables(), &[3]);
        assert_eq!(variable_one_off.log_values(), &[1.0_f64.ln(), 3.0_f64.ln()]);

        assert_eq!(factor.condition(99, true).unwrap(), factor);
    }

    #[test]
    fn cutset_conditioning_matches_full_joint_oracle_when_min_fill_exceeds_budget() {
        // A 4x4 grid has treewidth four. Removing the second row as a cutset
        // leaves a chain plus a 2x4 ladder (treewidth two), so exact contraction
        // fits in eight-entry factors although unconditioned min-fill does not.
        const WIDTH: usize = 4;
        const HEIGHT: usize = 4;
        const VARIABLES: usize = WIDTH * HEIGHT;
        const RETAINED: usize = VARIABLES - 1;
        const MAX_FACTOR_ENTRIES: usize = 8;
        const PAIR_VALUES: [f64; 4] = [0.83, 0.17, 0.29, 0.71];

        let mut factors = Vec::new();
        for row in 0..HEIGHT {
            for column in 0..WIDTH {
                let here = row * WIDTH + column;
                if column + 1 < WIDTH {
                    factors.push(BinaryLogFactor::new(
                        vec![here, here + 1],
                        PAIR_VALUES.iter().map(|value| value.ln()).collect(),
                    ).unwrap());
                }
                if row + 1 < HEIGHT {
                    factors.push(BinaryLogFactor::new(
                        vec![here, here + WIDTH],
                        PAIR_VALUES.iter().map(|value| value.ln()).collect(),
                    ).unwrap());
                }
            }
        }

        let eliminate = (0..VARIABLES)
            .filter(|variable| *variable != RETAINED)
            .collect::<Vec<_>>();
        let cutset = [4, 5, 6, 7]; // Entire second row.
        assert!(matches!(
            min_fill_log_sum_product(&factors, &eliminate, MAX_FACTOR_ENTRIES),
            Err(FactorGraphError::IntermediateTooLarge { limit: MAX_FACTOR_ENTRIES, .. })
        ));

        let report = min_fill_log_sum_product_with_cutset(
            &factors,
            &eliminate,
            &cutset,
            MAX_FACTOR_ENTRIES,
            1 << cutset.len(),
        ).unwrap();
        assert_eq!(report.factor.variables(), &[RETAINED]);
        assert_eq!(report.cutset_variables, cutset);
        assert_eq!(report.cutset_assignments_evaluated, 1 << cutset.len());
        assert!(report.conditioned_factor_entries_written > 0);
        assert_eq!(report.branch_output_entries_accumulated, 30);
        assert!(report.induced_width <= 2);
        assert_eq!(report.predicted_peak_factor_entries, report.peak_factor_entries);
        assert!(report.peak_factor_entries <= MAX_FACTOR_ENTRIES);

        // Independent full-joint enumeration is intentionally exponential and
        // used only as a small-system oracle for the conditioned contraction.
        let mut expected = vec![f64::NEG_INFINITY; 2];
        for assignment in 0..(1usize << VARIABLES) {
            let mut log_weight = 0.0;
            for factor in &factors {
                let scope = factor.variables();
                let local_index = usize::from((assignment >> scope[0]) & 1)
                    | (usize::from((assignment >> scope[1]) & 1) << 1);
                log_weight += factor.log_values()[local_index];
            }
            let retained_state = (assignment >> RETAINED) & 1;
            expected[retained_state] = log_add_exp(expected[retained_state], log_weight);
        }

        for (index, (actual, expected)) in report.factor.log_values()
            .iter()
            .copied()
            .zip(expected)
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1e-11,
                "retained state {index}: cutset={actual:.17}, brute_force={expected:.17}"
            );
        }
    }

    #[test]
    fn cutset_conditioning_matches_bruteforce_on_randomized_sparse_graphs() {
        for variable_count in 2..=7 {
            for seed in 0..12usize {
                let mut factors = Vec::new();
                for left in 0..variable_count - 1 {
                    let right = left + 1;
                    let values = (0..4).map(|entry| {
                        let code = (seed * 31 + left * 17 + entry * 7 + 1) % 23;
                        if code == 0 {
                            f64::NEG_INFINITY
                        } else {
                            (code as f64 / 24.0).ln()
                        }
                    }).collect();
                    factors.push(BinaryLogFactor::new(vec![left, right], values).unwrap());
                }
                for left in 0..variable_count.saturating_sub(2) {
                    if (left + seed) % 2 == 0 {
                        let right = left + 2;
                        let values = (0..4).map(|entry| {
                            let code = (seed * 13 + left * 11 + entry * 5 + 3) % 29;
                            if code == 0 {
                                f64::NEG_INFINITY
                            } else {
                                (code as f64 / 30.0).ln()
                            }
                        }).collect();
                        factors.push(BinaryLogFactor::new(vec![left, right], values).unwrap());
                    }
                }

                let retained = variable_count - 1;
                let eliminate = (0..retained).collect::<Vec<_>>();
                let mut cutset = eliminate.iter().copied()
                    .filter(|variable| (*variable + seed) % 3 == 0)
                    .collect::<Vec<_>>();
                if cutset.is_empty() {
                    cutset.push(seed % retained);
                }
                let max_entries = 1usize << variable_count;
                let baseline = min_fill_log_sum_product(
                    &factors,
                    &eliminate,
                    max_entries,
                ).unwrap_or_else(|error| {
                    panic!("baseline failed: n={variable_count}, seed={seed}: {error}")
                });
                let report = min_fill_log_sum_product_with_cutset(
                    &factors,
                    &eliminate,
                    &cutset,
                    max_entries,
                    1usize << cutset.len(),
                ).unwrap_or_else(|error| {
                    panic!("cutset failed: n={variable_count}, seed={seed}, cutset={cutset:?}: {error}")
                });

                let mut expected = vec![f64::NEG_INFINITY; 2];
                for assignment in 0..(1usize << variable_count) {
                    let mut log_weight = 0.0;
                    for factor in &factors {
                        let local_index = factor.variables().iter().copied().enumerate().fold(
                            0usize,
                            |index, (slot, variable)| {
                                index | (((assignment >> variable) & 1) << slot)
                            },
                        );
                        log_weight += factor.log_values()[local_index];
                    }
                    let retained_state = (assignment >> retained) & 1;
                    expected[retained_state] = log_add_exp(expected[retained_state], log_weight);
                }

                assert_eq!(baseline.factor.variables(), &[retained]);
                assert_eq!(report.factor.variables(), &[retained]);
                assert_eq!(report.cutset_assignments_evaluated, 1usize << cutset.len());
                assert_eq!(
                    report.predicted_peak_factor_entries,
                    report.peak_factor_entries,
                    "preflight peak differs: n={variable_count}, seed={seed}"
                );
                for state in 0..2 {
                    let baseline_value = baseline.factor.log_values()[state];
                    let cutset_value = report.factor.log_values()[state];
                    let expected_value = expected[state];
                    assert!(
                        (baseline_value == expected_value)
                            || (baseline_value.is_finite()
                                && expected_value.is_finite()
                                && (baseline_value - expected_value).abs() <= 1e-11),
                        "baseline/oracle mismatch: n={variable_count}, seed={seed}, state={state}, baseline={baseline_value}, expected={expected_value}"
                    );
                    assert!(
                        (cutset_value == expected_value)
                            || (cutset_value.is_finite()
                                && expected_value.is_finite()
                                && (cutset_value - expected_value).abs() <= 1e-11),
                        "cutset/oracle mismatch: n={variable_count}, seed={seed}, state={state}, cutset={cutset_value}, expected={expected_value}"
                    );
                }
            }
        }
    }

    #[test]
    fn cutset_plan_reuses_structure_across_numeric_factors_and_rejects_scope_drift() {
        let scopes = vec![vec![0, 1], vec![1, 2]];
        let eliminate = [0, 1];
        let cutset = [1];
        let plan = plan_min_fill_cutset_elimination(
            &scopes,
            &eliminate,
            &cutset,
            4,
            2,
        ).unwrap();

        let first = vec![
            BinaryLogFactor::new(vec![0, 1], vec![0.9_f64.ln(), 0.1_f64.ln(), 0.2_f64.ln(), 0.8_f64.ln()]).unwrap(),
            BinaryLogFactor::new(vec![1, 2], vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()]).unwrap(),
        ];
        let second = vec![
            BinaryLogFactor::new(vec![0, 1], vec![0.6_f64.ln(), 0.4_f64.ln(), 0.15_f64.ln(), 0.85_f64.ln()]).unwrap(),
            BinaryLogFactor::new(vec![1, 2], vec![0.55_f64.ln(), 0.45_f64.ln(), 0.25_f64.ln(), 0.75_f64.ln()]).unwrap(),
        ];

        for factors in [&first, &second] {
            let reused = min_fill_log_sum_product_with_cutset_plan(factors, &plan).unwrap();
            let rebuilt = min_fill_log_sum_product_with_cutset(
                factors,
                &eliminate,
                &cutset,
                4,
                2,
            ).unwrap();
            assert_eq!(reused.factor, rebuilt.factor);
            assert_eq!(reused.cutset_variables, rebuilt.cutset_variables);
            assert_eq!(reused.induced_width, rebuilt.induced_width);
            assert_eq!(reused.predicted_peak_factor_entries, rebuilt.predicted_peak_factor_entries);
        }

        let wrong_scopes = vec![
            BinaryLogFactor::new(vec![0, 2], vec![0.9_f64.ln(), 0.1_f64.ln(), 0.2_f64.ln(), 0.8_f64.ln()]).unwrap(),
            BinaryLogFactor::new(vec![1, 2], vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()]).unwrap(),
        ];
        assert!(matches!(
            min_fill_log_sum_product_with_cutset_plan(&wrong_scopes, &plan),
            Err(FactorGraphError::FactorScopesDoNotMatchCutsetPlan)
        ));
    }

    #[test]
    fn cutset_preflight_exposes_canonical_separator_and_residual_cost_before_execution() {
        const SIDE: usize = 4;
        let mut factors = Vec::new();
        for row in 0..SIDE {
            for column in 0..SIDE {
                let here = row * SIDE + column;
                if column + 1 < SIDE {
                    factors.push(vec![here, here + 1]);
                }
                if row + 1 < SIDE {
                    factors.push(vec![here, here + SIDE]);
                }
            }
        }
        let eliminate = (0..SIDE * SIDE - 1).collect::<Vec<_>>();
        let cutset = [7, 5, 6, 4]; // Deliberately not canonical input order.
        let plan = plan_min_fill_cutset_elimination(
            &factors,
            &eliminate,
            &cutset,
            8,
            16,
        ).unwrap();

        assert_eq!(plan.cutset_variables, vec![4, 5, 6, 7]);
        assert_eq!(plan.cutset_assignments, 16);
        assert_eq!(plan.max_cutset_assignments, 16);
        assert_eq!(plan.largest_input_factor_entries, 4);
        assert_eq!(plan.remaining_elimination_plan.induced_width, 2);
        assert_eq!(plan.remaining_elimination_plan.predicted_peak_factor_entries, 8);
        assert_eq!(plan.predicted_peak_factor_entries, 8);
        assert_eq!(plan.remaining_elimination_plan.max_factor_entries, 8);

        assert!(matches!(
            plan_min_fill_cutset_elimination(
                &factors,
                &eliminate,
                &cutset,
                8,
                15,
            ),
            Err(FactorGraphError::CutsetAssignmentBudgetExceeded {
                assignments: 16,
                maximum: 15
            })
        ));
        assert!(matches!(
            plan_min_fill_cutset_elimination(
                &factors,
                &eliminate,
                &cutset,
                7,
                16,
            ),
            Err(FactorGraphError::IntermediateTooLarge { limit: 7, .. })
        ));
    }

    #[test]
    fn cutset_conditioning_counts_absent_variables_and_rejects_invalid_budgets() {
        let factor = BinaryLogFactor::new(vec![0], vec![0.25_f64.ln(), 0.75_f64.ln()]).unwrap();
        let ordinary = min_fill_log_sum_product(&[factor.clone()], &[1], 2).unwrap();
        let conditioned = min_fill_log_sum_product_with_cutset(
            &[factor.clone()],
            &[1],
            &[1],
            2,
            2,
        ).unwrap();
        assert_eq!(conditioned.factor.variables(), ordinary.factor.variables());
        for (actual, expected) in conditioned.factor.log_values()
            .iter()
            .zip(ordinary.factor.log_values())
        {
            close(*actual, *expected);
        }

        assert!(matches!(
            min_fill_log_sum_product_with_cutset(&[factor.clone()], &[1], &[1, 1], 2, 4),
            Err(FactorGraphError::DuplicateCutsetVariable { variable: 1 })
        ));
        assert!(matches!(
            min_fill_log_sum_product_with_cutset(&[factor.clone()], &[1], &[0], 2, 2),
            Err(FactorGraphError::CutsetVariableNotEliminated { variable: 0 })
        ));
        assert!(matches!(
            min_fill_log_sum_product_with_cutset(&[factor], &[1, 2], &[1, 2], 2, 3),
            Err(FactorGraphError::CutsetAssignmentBudgetExceeded { assignments: 4, maximum: 3 })
        ));
    }

    #[test]
    fn cutset_frontier_preserves_tradeoffs_and_reports_rejected_candidates() {
        const SIDE: usize = 4;
        let mut scopes = Vec::new();
        for row in 0..SIDE {
            for column in 0..SIDE {
                let here = row * SIDE + column;
                if column + 1 < SIDE {
                    scopes.push(vec![here, here + 1]);
                }
                if row + 1 < SIDE {
                    scopes.push(vec![here, here + SIDE]);
                }
            }
        }
        let eliminate = (0..SIDE * SIDE - 1).collect::<Vec<_>>();
        let candidates = vec![
            vec![4, 5, 6, 7],            // 16 branches, width 2, peak 8
            (4..12).collect::<Vec<_>>(), // 256 branches, width 1, peak 4
            (0..15).collect::<Vec<_>>(), // Exceeds the 1,024-branch budget.
            vec![7, 6, 5, 4],            // Exact metric tie with candidate 0.
        ];

        let report = plan_min_fill_cutset_frontier(
            &scopes,
            &eliminate,
            &candidates,
            8,
            1_024,
        ).unwrap();

        assert_eq!(
            report.candidates.iter().map(|candidate| candidate.candidate_index).collect::<Vec<_>>(),
            vec![0, 1, 3]
        );
        assert_eq!(report.frontier_candidate_indices, vec![0, 1]);
        assert_eq!(report.rejected_candidates.len(), 1);
        assert_eq!(report.rejected_candidates[0].candidate_index, 2);
        assert!(matches!(
            &report.rejected_candidates[0].error,
            FactorGraphError::CutsetAssignmentBudgetExceeded {
                assignments: 32_768,
                maximum: 1_024
            }
        ));

        assert_eq!(report.candidates[0].plan.cutset_variables(), &[4, 5, 6, 7]);
        assert_eq!(report.candidates[0].plan.cutset_assignments(), 16);
        assert_eq!(
            report.candidates[0].plan.remaining_elimination_plan().induced_width,
            2
        );
        assert_eq!(report.candidates[0].plan.predicted_peak_factor_entries(), 8);

        assert_eq!(report.candidates[1].plan.cutset_variables(), &[4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(report.candidates[1].plan.cutset_assignments(), 256);
        assert_eq!(
            report.candidates[1].plan.remaining_elimination_plan().induced_width,
            1
        );
        assert_eq!(report.candidates[1].plan.predicted_peak_factor_entries(), 4);
    }

    #[test]
    fn cutset_frontier_rejects_an_empty_candidate_family() {
        assert!(matches!(
            plan_min_fill_cutset_frontier(&[vec![0, 1]], &[0], &[], 4, 4),
            Err(FactorGraphError::EmptyCutsetCandidateSet)
        ));
    }

    #[test]
    fn absent_elimination_variable_contributes_binary_sum_factor() {
        let factor = BinaryFactor::new(vec![0], vec![0.25, 0.75]).unwrap();
        let report = min_fill_sum_product(&[factor], &[1], 2).unwrap();
        assert_eq!(report.factor.variables, vec![0]);
        close(report.factor.values[0], 0.5);
        close(report.factor.values[1], 1.5);
    }
}
