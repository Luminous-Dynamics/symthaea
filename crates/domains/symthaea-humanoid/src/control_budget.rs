// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Solver admission and runtime-budget evidence.
//!
//! This layer enforces configured limits and fail-closed fallback. It does not
//! turn a general-purpose OS thread into a hard real-time system; deployments
//! still require an RT scheduler and independently supervised actuator expiry.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SolverBudget {
    pub maximum_variables: usize,
    pub maximum_constraints: usize,
    pub maximum_estimated_operations: u64,
    pub maximum_elapsed_micros: u64,
}

impl Default for SolverBudget {
    fn default() -> Self {
        Self {
            maximum_variables: 192,
            maximum_constraints: 192,
            maximum_estimated_operations: 24_000_000,
            maximum_elapsed_micros: 2_000,
        }
    }
}

impl SolverBudget {
    pub fn validate(&self) -> bool {
        self.maximum_variables > 0
            && self.maximum_constraints > 0
            && self.maximum_estimated_operations > 0
            && self.maximum_elapsed_micros > 0
    }

    pub fn admit_dense(&self, variables: usize, constraints: usize) -> BudgetAdmission {
        let dimension = variables.saturating_add(constraints) as u64;
        let estimated_operations = dimension
            .saturating_mul(dimension)
            .saturating_mul(dimension)
            .saturating_add((variables as u64).saturating_mul(constraints as u64));
        self.admit(variables, constraints, estimated_operations)
    }

    pub fn admit_sparse(
        &self,
        variables: usize,
        constraints: usize,
        maximum_iterations: usize,
    ) -> BudgetAdmission {
        let estimated_operations = (maximum_iterations as u64)
            .saturating_mul(constraints as u64)
            .saturating_mul(variables.max(1) as u64);
        self.admit(variables, constraints, estimated_operations)
    }

    fn admit(
        &self,
        variables: usize,
        constraints: usize,
        estimated_operations: u64,
    ) -> BudgetAdmission {
        let admitted = self.validate()
            && variables <= self.maximum_variables
            && constraints <= self.maximum_constraints
            && estimated_operations <= self.maximum_estimated_operations;
        BudgetAdmission {
            admitted,
            variables,
            constraints,
            estimated_operations,
            maximum_elapsed_micros: self.maximum_elapsed_micros,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BudgetAdmission {
    pub admitted: bool,
    pub variables: usize,
    pub constraints: usize,
    pub estimated_operations: u64,
    pub maximum_elapsed_micros: u64,
}

impl BudgetAdmission {
    pub fn start(self) -> BudgetTimer {
        BudgetTimer {
            admission: self,
            started_at: Instant::now(),
        }
    }
}

pub struct BudgetTimer {
    admission: BudgetAdmission,
    started_at: Instant,
}

impl BudgetTimer {
    pub fn finish(self) -> SolverBudgetEvidence {
        let elapsed = self.started_at.elapsed();
        let elapsed_micros = elapsed.as_micros().min(u64::MAX as u128) as u64;
        SolverBudgetEvidence {
            admitted: self.admission.admitted,
            elapsed_micros,
            deadline_missed: !self.admission.admitted
                || elapsed > Duration::from_micros(self.admission.maximum_elapsed_micros),
            estimated_operations: self.admission.estimated_operations,
            variables: self.admission.variables,
            constraints: self.admission.constraints,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SolverBudgetEvidence {
    pub admitted: bool,
    pub elapsed_micros: u64,
    pub deadline_missed: bool,
    pub estimated_operations: u64,
    pub variables: usize,
    pub constraints: usize,
}

impl SolverBudgetEvidence {
    pub const fn rejected() -> Self {
        Self {
            admitted: false,
            elapsed_micros: 0,
            deadline_missed: true,
            estimated_operations: 0,
            variables: 0,
            constraints: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_dense_problem_is_rejected_before_solve() {
        let budget = SolverBudget {
            maximum_variables: 10,
            ..SolverBudget::default()
        };
        assert!(!budget.admit_dense(11, 1).admitted);
    }

    #[test]
    fn operation_estimate_is_saturating() {
        let admission = SolverBudget::default().admit_dense(usize::MAX, usize::MAX);
        assert!(!admission.admitted);
        assert_eq!(admission.estimated_operations, u64::MAX);
    }
}
