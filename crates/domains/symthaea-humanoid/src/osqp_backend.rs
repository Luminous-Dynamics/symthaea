// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Production in-process OSQP adapter.
//!
//! Symthaea's bounded equality QP is mapped to OSQP's canonical form by
//! stacking the equality rows above an identity matrix for variable bounds:
//! `l <= [A_eq; I] x <= u`. The adapter preserves a workspace while the
//! canonical sparsity fingerprint is unchanged, warm-starts the primal state,
//! and independently recomputes every residual and objective before returning.

use std::time::Duration;

use osqp::{CscMatrix as OsqpCscMatrix, Problem, Settings, Status};
use serde::{Deserialize, Serialize};

use crate::csc_qp::CscEqualityQuadraticProgram;
use crate::inprocess_sparse_qp::{InProcessSparseQpEngine, InProcessSparseQpEngineResult};

pub const OSQP_ENGINE_ID: &str = "osqp-rust-1.0.1-symthaea-v1";

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OsqpEngineConfig {
    pub maximum_iterations: u32,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub time_limit_micros: u64,
    pub polishing: bool,
    pub adaptive_rho: bool,
}

impl Default for OsqpEngineConfig {
    fn default() -> Self {
        Self {
            maximum_iterations: 4_000,
            absolute_tolerance: 1.0e-7,
            relative_tolerance: 1.0e-6,
            time_limit_micros: 1_500,
            polishing: false,
            adaptive_rho: false,
        }
    }
}

impl OsqpEngineConfig {
    pub fn validate(&self) -> bool {
        self.maximum_iterations > 0
            && self.absolute_tolerance.is_finite()
            && self.absolute_tolerance > 0.0
            && self.relative_tolerance.is_finite()
            && self.relative_tolerance > 0.0
            && self.time_limit_micros > 0
    }

    fn settings(&self) -> Settings {
        Settings::default()
            .verbose(false)
            .max_iter(self.maximum_iterations)
            .eps_abs(self.absolute_tolerance)
            .eps_rel(self.relative_tolerance)
            .adaptive_rho(self.adaptive_rho)
            .polishing(self.polishing)
            .warm_starting(true)
            .time_limit(Some(Duration::from_micros(self.time_limit_micros)))
    }
}

struct OsqpWorkspace {
    structure_fingerprint: u64,
    variables: usize,
    equality_rows: usize,
    problem: Problem,
}

pub struct OsqpInProcessEngine {
    config: OsqpEngineConfig,
    workspace: Option<OsqpWorkspace>,
}

impl OsqpInProcessEngine {
    pub fn new(config: OsqpEngineConfig) -> Option<Self> {
        config.validate().then_some(Self {
            config,
            workspace: None,
        })
    }

    fn build_workspace(&self, problem: &CscEqualityQuadraticProgram) -> Option<OsqpWorkspace> {
        let variables = problem.diagonal_hessian.len();
        let equality_rows = problem.equality_target.len();
        let p = diagonal_osqp_matrix(&problem.diagonal_hessian);
        let a = combined_constraint_matrix(problem)?;
        let (lower, upper) = combined_bounds(problem);
        let solver = Problem::new(
            p,
            &problem.linear_term,
            a,
            &lower,
            &upper,
            &self.config.settings(),
        )
        .ok()?;
        Some(OsqpWorkspace {
            structure_fingerprint: problem.structure_fingerprint,
            variables,
            equality_rows,
            problem: solver,
        })
    }

    fn update_workspace(
        workspace: &mut OsqpWorkspace,
        problem: &CscEqualityQuadraticProgram,
    ) -> Option<()> {
        if workspace.structure_fingerprint != problem.structure_fingerprint
            || workspace.variables != problem.diagonal_hessian.len()
            || workspace.equality_rows != problem.equality_target.len()
        {
            return None;
        }
        let p = diagonal_osqp_matrix(&problem.diagonal_hessian);
        let a = combined_constraint_matrix(problem)?;
        let (lower, upper) = combined_bounds(problem);
        workspace.problem.update_P(p);
        workspace.problem.update_A(a);
        workspace.problem.update_lin_cost(&problem.linear_term);
        workspace.problem.update_bounds(&lower, &upper);
        Some(())
    }
}

impl Default for OsqpInProcessEngine {
    fn default() -> Self {
        Self::new(OsqpEngineConfig::default())
            .expect("default OSQP engine configuration must be valid")
    }
}

impl InProcessSparseQpEngine for OsqpInProcessEngine {
    fn engine_id(&self) -> &str {
        OSQP_ENGINE_ID
    }

    fn solve_sparse(
        &mut self,
        problem: &CscEqualityQuadraticProgram,
        warm_start: Option<&[f64]>,
    ) -> Option<InProcessSparseQpEngineResult> {
        if !problem.validate() || !self.config.validate() {
            return None;
        }
        let symbolic_pattern_reused = self
            .workspace
            .as_ref()
            .map(|workspace| {
                workspace.structure_fingerprint == problem.structure_fingerprint
                    && workspace.variables == problem.diagonal_hessian.len()
                    && workspace.equality_rows == problem.equality_target.len()
            })
            .unwrap_or(false);
        if symbolic_pattern_reused {
            Self::update_workspace(self.workspace.as_mut()?, problem)?;
        } else {
            self.workspace = Some(self.build_workspace(problem)?);
        }
        let workspace = self.workspace.as_mut()?;
        let warm_start_used = warm_start
            .map(|values| {
                values.len() == workspace.variables && values.iter().all(|value| value.is_finite())
            })
            .unwrap_or(false);
        if let Some(values) = warm_start.filter(|_| warm_start_used) {
            workspace.problem.warm_start_x(values);
        }

        let status = workspace.problem.solve();
        let iterations = status.iter() as usize;
        let (values, dual, primal_residual, dual_residual, converged) = match status {
            Status::Solved(solution) => (
                solution.x().to_vec(),
                solution.y().to_vec(),
                solution.pri_res(),
                solution.dua_res(),
                true,
            ),
            Status::SolvedInaccurate(solution) => (
                solution.x().to_vec(),
                solution.y().to_vec(),
                solution.pri_res(),
                solution.dua_res(),
                false,
            ),
            Status::MaxIterationsReached(solution) | Status::TimeLimitReached(solution) => (
                solution.x().to_vec(),
                solution.y().to_vec(),
                solution.pri_res(),
                solution.dua_res(),
                false,
            ),
            Status::PrimalInfeasible(_)
            | Status::PrimalInfeasibleInaccurate(_)
            | Status::DualInfeasible(_)
            | Status::DualInfeasibleInaccurate(_)
            | Status::NonConvex(_) => return None,
            _ => return None,
        };
        if values.len() != workspace.variables
            || dual.len() != workspace.equality_rows + workspace.variables
        {
            return None;
        }
        let equality_multipliers = dual[..workspace.equality_rows].to_vec();
        let maximum_equality_residual = maximum_equality_residual(problem, &values)?;
        let maximum_bound_violation = maximum_bound_violation(problem, &values)?;
        let objective = objective(problem, &values)?;
        let admitted_convergence = converged
            && maximum_equality_residual <= self.config.absolute_tolerance.max(1.0e-8)
            && maximum_bound_violation <= self.config.absolute_tolerance.max(1.0e-8);
        Some(InProcessSparseQpEngineResult {
            values,
            equality_multipliers,
            iterations: iterations.max(1),
            maximum_equality_residual,
            maximum_bound_violation,
            primal_residual: primal_residual.abs(),
            dual_residual: dual_residual.abs(),
            objective,
            converged: admitted_convergence,
            warm_start_used,
            symbolic_pattern_reused,
        })
    }
}

fn diagonal_osqp_matrix(diagonal: &[f64]) -> OsqpCscMatrix<'static> {
    let variables = diagonal.len();
    OsqpCscMatrix::from_row_iter(
        variables,
        variables,
        (0..variables).flat_map(|row| {
            (0..variables).map(move |column| if row == column { diagonal[row] } else { 0.0 })
        }),
    )
    .into_upper_tri()
}

fn combined_constraint_matrix(
    problem: &CscEqualityQuadraticProgram,
) -> Option<OsqpCscMatrix<'static>> {
    let variables = problem.diagonal_hessian.len();
    let equality_rows = problem.equality_target.len();
    let dense_equalities = dense_equality_rows(problem)?;
    Some(OsqpCscMatrix::from_row_iter(
        equality_rows + variables,
        variables,
        (0..equality_rows + variables).flat_map(|row| {
            let dense_equalities = &dense_equalities;
            (0..variables).map(move |column| {
                if row < equality_rows {
                    dense_equalities[row][column]
                } else if row - equality_rows == column {
                    1.0
                } else {
                    0.0
                }
            })
        }),
    ))
}

fn dense_equality_rows(problem: &CscEqualityQuadraticProgram) -> Option<Vec<Vec<f64>>> {
    let rows = problem.equality_target.len();
    let columns = problem.diagonal_hessian.len();
    let mut dense = vec![vec![0.0; columns]; rows];
    let matrix = &problem.equality_matrix;
    if !matrix.validate() || matrix.rows != rows || matrix.columns != columns {
        return None;
    }
    for column in 0..columns {
        for index in matrix.column_offsets[column]..matrix.column_offsets[column + 1] {
            dense[matrix.row_indices[index]][column] = matrix.values[index];
        }
    }
    Some(dense)
}

fn combined_bounds(problem: &CscEqualityQuadraticProgram) -> (Vec<f64>, Vec<f64>) {
    let mut lower = Vec::with_capacity(problem.equality_target.len() + problem.lower_bounds.len());
    lower.extend_from_slice(&problem.equality_target);
    lower.extend_from_slice(&problem.lower_bounds);
    let mut upper = Vec::with_capacity(problem.equality_target.len() + problem.upper_bounds.len());
    upper.extend_from_slice(&problem.equality_target);
    upper.extend_from_slice(&problem.upper_bounds);
    (lower, upper)
}

fn maximum_equality_residual(problem: &CscEqualityQuadraticProgram, values: &[f64]) -> Option<f64> {
    let applied = problem.equality_matrix.multiply(values)?;
    Some(
        applied
            .iter()
            .zip(problem.equality_target.iter())
            .map(|(value, target)| (value - target).abs())
            .fold(0.0, f64::max),
    )
}

fn maximum_bound_violation(problem: &CscEqualityQuadraticProgram, values: &[f64]) -> Option<f64> {
    (values.len() == problem.lower_bounds.len()).then(|| {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if *value < problem.lower_bounds[index] {
                    problem.lower_bounds[index] - *value
                } else if *value > problem.upper_bounds[index] {
                    *value - problem.upper_bounds[index]
                } else {
                    0.0
                }
            })
            .fold(0.0, f64::max)
    })
}

fn objective(problem: &CscEqualityQuadraticProgram, values: &[f64]) -> Option<f64> {
    (values.len() == problem.diagonal_hessian.len()).then(|| {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                0.5 * problem.diagonal_hessian[index] * value * value
                    + problem.linear_term[index] * value
            })
            .sum()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc_qp::{CSC_CANONICAL_ZERO_TOLERANCE, CscEqualityQuadraticProgram};
    use crate::equality_qp::DenseEqualityQuadraticProgram;

    fn problem() -> CscEqualityQuadraticProgram {
        CscEqualityQuadraticProgram::from_dense(
            &DenseEqualityQuadraticProgram {
                diagonal_hessian: vec![2.0, 2.0],
                linear_term: vec![-1.0, -1.0],
                lower_bounds: vec![0.0, 0.0],
                upper_bounds: vec![1.0, 1.0],
                equality_matrix: vec![vec![1.0, 1.0]],
                equality_target: vec![1.0],
            },
            CSC_CANONICAL_ZERO_TOLERANCE,
        )
        .unwrap()
    }

    #[test]
    fn osqp_solves_bounded_equality_problem() {
        let mut engine = OsqpInProcessEngine::default();
        let result = engine.solve_sparse(&problem(), None).unwrap();
        assert!(result.converged);
        assert!((result.values[0] - 0.5).abs() < 1.0e-5);
        assert!((result.values[1] - 0.5).abs() < 1.0e-5);
    }

    #[test]
    fn compatible_second_solve_reuses_symbolic_pattern() {
        let mut engine = OsqpInProcessEngine::default();
        let first = engine.solve_sparse(&problem(), None).unwrap();
        let second = engine
            .solve_sparse(&problem(), Some(&first.values))
            .unwrap();
        assert!(second.symbolic_pattern_reused);
        assert!(second.warm_start_used);
    }
}
