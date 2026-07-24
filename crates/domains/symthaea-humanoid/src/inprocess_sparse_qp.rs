// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! In-process sparse-QP engine boundary with a deterministic reference engine.
//!
//! Production wrappers for OSQP, ProxQP, qpOASES, or a platform-specific
//! solver can implement [`InProcessSparseQpEngine`]. The adapter owns engine
//! state behind a mutex so symbolic workspaces and warm starts survive between
//! control ticks while the public [`SparseQpBackend`] contract remains shared
//! with the deterministic and process backends.
//!
//! [`ReferenceProjectedAdmmEngine`] is intentionally modest: it is a bounded,
//! allocation-conscious qualification engine, not a claim of feature parity
//! with OSQP or ProxQP. Returned solutions still pass the independent residual
//! and objective validation in `sparse_qp_backend` before admission.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::csc_qp::{CSC_CANONICAL_ZERO_TOLERANCE, CscEqualityQuadraticProgram};
use crate::equality_qp::{DenseEqualityQuadraticProgram, EqualityQpSolution};
use crate::sparse_qp_backend::{
    SparseQpBackend, SparseQpBackendSolution, validate_external_solution,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InProcessSparseQpEngineResult {
    pub values: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub iterations: usize,
    pub maximum_equality_residual: f64,
    pub maximum_bound_violation: f64,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub objective: f64,
    pub converged: bool,
    pub warm_start_used: bool,
    pub symbolic_pattern_reused: bool,
}

impl InProcessSparseQpEngineResult {
    pub fn validate_for(&self, problem: &CscEqualityQuadraticProgram) -> bool {
        self.values.len() == problem.diagonal_hessian.len()
            && self.equality_multipliers.len() == problem.equality_target.len()
            && self.values.iter().all(|value| value.is_finite())
            && self
                .equality_multipliers
                .iter()
                .all(|value| value.is_finite())
            && self.iterations > 0
            && [
                self.maximum_equality_residual,
                self.maximum_bound_violation,
                self.primal_residual,
                self.dual_residual,
                self.objective,
            ]
            .iter()
            .all(|value| value.is_finite())
            && self.maximum_equality_residual >= 0.0
            && self.maximum_bound_violation >= 0.0
            && self.primal_residual >= 0.0
            && self.dual_residual >= 0.0
    }

    fn into_equality_solution(self, problem: &CscEqualityQuadraticProgram) -> EqualityQpSolution {
        let active_bound_count = self
            .values
            .iter()
            .enumerate()
            .filter(|(index, value)| {
                (**value - problem.lower_bounds[*index]).abs() <= 1.0e-8
                    || (**value - problem.upper_bounds[*index]).abs() <= 1.0e-8
            })
            .count();
        EqualityQpSolution {
            values: self.values,
            multipliers: self.equality_multipliers,
            active_bound_count,
            warm_start_active_bound_count: 0,
            warm_start_used: self.warm_start_used,
            active_set_iterations: self.iterations,
            maximum_equality_residual: self.maximum_equality_residual,
            maximum_bound_violation: self.maximum_bound_violation,
            objective: self.objective,
            converged: self.converged,
        }
    }
}

pub trait InProcessSparseQpEngine: Send {
    fn engine_id(&self) -> &str;

    fn solve_sparse(
        &mut self,
        problem: &CscEqualityQuadraticProgram,
        warm_start: Option<&[f64]>,
    ) -> Option<InProcessSparseQpEngineResult>;
}

pub struct InProcessSparseQpBackend<E: InProcessSparseQpEngine> {
    backend_id: String,
    engine: Mutex<E>,
    zero_tolerance: f64,
    validation_tolerance: f64,
}

impl<E: InProcessSparseQpEngine> InProcessSparseQpBackend<E> {
    pub fn new(engine: E) -> Option<Self> {
        let backend_id = engine.engine_id().trim().to_string();
        (!backend_id.is_empty()).then_some(Self {
            backend_id,
            engine: Mutex::new(engine),
            zero_tolerance: CSC_CANONICAL_ZERO_TOLERANCE,
            validation_tolerance: 1.0e-6,
        })
    }

    pub fn with_zero_tolerance(mut self, zero_tolerance: f64) -> Self {
        if zero_tolerance.is_finite()
            && (zero_tolerance - CSC_CANONICAL_ZERO_TOLERANCE).abs() <= f64::EPSILON
        {
            self.zero_tolerance = zero_tolerance;
        }
        self
    }

    pub fn with_validation_tolerance(mut self, validation_tolerance: f64) -> Self {
        if validation_tolerance.is_finite() && validation_tolerance >= 0.0 {
            self.validation_tolerance = validation_tolerance;
        }
        self
    }
}

impl<E: InProcessSparseQpEngine> SparseQpBackend for InProcessSparseQpBackend<E> {
    fn backend_id(&self) -> &str {
        &self.backend_id
    }

    fn solve(
        &self,
        problem: &DenseEqualityQuadraticProgram,
        structure_fingerprint: u64,
        warm_start: Option<&[f64]>,
    ) -> Option<SparseQpBackendSolution> {
        let sparse = CscEqualityQuadraticProgram::from_dense(problem, self.zero_tolerance)?;
        if structure_fingerprint != 0 && sparse.structure_fingerprint != structure_fingerprint {
            return None;
        }
        let mut engine = self.engine.lock().ok()?;
        if engine.engine_id() != self.backend_id {
            return None;
        }
        let result = engine.solve_sparse(&sparse, warm_start)?;
        if !result.validate_for(&sparse) {
            return None;
        }
        let symbolic_pattern_reused = result.symbolic_pattern_reused;
        let solution = result.into_equality_solution(&sparse);
        if !validate_external_solution(problem, &solution, self.validation_tolerance) {
            return None;
        }
        Some(SparseQpBackendSolution {
            solution,
            backend_id: self.backend_id.clone(),
            symbolic_pattern_reused,
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReferenceProjectedAdmmConfig {
    pub maximum_iterations: usize,
    pub rho: f64,
    pub equality_penalty: f64,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub pivot_tolerance: f64,
}

impl Default for ReferenceProjectedAdmmConfig {
    fn default() -> Self {
        Self {
            maximum_iterations: 256,
            rho: 4.0,
            equality_penalty: 1.0e8,
            absolute_tolerance: 1.0e-7,
            relative_tolerance: 1.0e-6,
            pivot_tolerance: 1.0e-14,
        }
    }
}

impl ReferenceProjectedAdmmConfig {
    pub fn validate(&self) -> bool {
        self.maximum_iterations > 0
            && self.rho.is_finite()
            && self.rho > 0.0
            && self.equality_penalty.is_finite()
            && self.equality_penalty > 0.0
            && self.absolute_tolerance.is_finite()
            && self.absolute_tolerance >= 0.0
            && self.relative_tolerance.is_finite()
            && self.relative_tolerance >= 0.0
            && self.pivot_tolerance.is_finite()
            && self.pivot_tolerance > 0.0
    }
}

#[derive(Debug, Clone)]
struct ReferenceWorkspace {
    structure_fingerprint: u64,
    values: Vec<f64>,
    projected: Vec<f64>,
    scaled_dual: Vec<f64>,
}

pub struct ReferenceProjectedAdmmEngine {
    config: ReferenceProjectedAdmmConfig,
    workspace: Option<ReferenceWorkspace>,
}

impl ReferenceProjectedAdmmEngine {
    pub fn new(config: ReferenceProjectedAdmmConfig) -> Option<Self> {
        config.validate().then_some(Self {
            config,
            workspace: None,
        })
    }
}

impl Default for ReferenceProjectedAdmmEngine {
    fn default() -> Self {
        Self::new(ReferenceProjectedAdmmConfig::default())
            .expect("default projected ADMM configuration must be valid")
    }
}

impl InProcessSparseQpEngine for ReferenceProjectedAdmmEngine {
    fn engine_id(&self) -> &str {
        "symthaea-reference-projected-admm-v1"
    }

    fn solve_sparse(
        &mut self,
        problem: &CscEqualityQuadraticProgram,
        warm_start: Option<&[f64]>,
    ) -> Option<InProcessSparseQpEngineResult> {
        if !problem.validate() || !self.config.validate() {
            return None;
        }
        let variables = problem.diagonal_hessian.len();
        let symbolic_pattern_reused = self
            .workspace
            .as_ref()
            .map(|workspace| {
                workspace.structure_fingerprint == problem.structure_fingerprint
                    && workspace.values.len() == variables
            })
            .unwrap_or(false);
        let warm_start_valid = warm_start
            .map(|values| values.len() == variables && values.iter().all(|value| value.is_finite()))
            .unwrap_or(false);
        let mut values = if let Some(values) = warm_start.filter(|_| warm_start_valid) {
            values.to_vec()
        } else if symbolic_pattern_reused {
            self.workspace.as_ref()?.values.clone()
        } else {
            vec![0.0; variables]
        };
        for index in 0..variables {
            values[index] =
                values[index].clamp(problem.lower_bounds[index], problem.upper_bounds[index]);
        }
        let mut projected = if symbolic_pattern_reused {
            self.workspace.as_ref()?.projected.clone()
        } else {
            values.clone()
        };
        let mut scaled_dual = if symbolic_pattern_reused {
            self.workspace.as_ref()?.scaled_dual.clone()
        } else {
            vec![0.0; variables]
        };
        if projected.len() != variables || scaled_dual.len() != variables {
            projected = values.clone();
            scaled_dual = vec![0.0; variables];
        }

        let mut normal = vec![vec![0.0; variables]; variables];
        for index in 0..variables {
            normal[index][index] = problem.diagonal_hessian[index] + self.config.rho;
        }
        for column_left in 0..variables {
            for column_right in column_left..variables {
                let product = csc_column_dot(&problem.equality_matrix, column_left, column_right)?;
                let value = self.config.equality_penalty * product;
                normal[column_left][column_right] += value;
                if column_left != column_right {
                    normal[column_right][column_left] += value;
                }
            }
        }
        let at_b = problem
            .equality_matrix
            .transpose_multiply(&problem.equality_target)?;
        let equality_rhs = at_b
            .iter()
            .map(|value| self.config.equality_penalty * value)
            .collect::<Vec<_>>();

        let mut iterations = 0usize;
        let mut primal_residual = f64::INFINITY;
        let mut dual_residual = f64::INFINITY;
        let mut maximum_equality_residual = f64::INFINITY;
        let mut converged = false;
        for iteration in 0..self.config.maximum_iterations {
            iterations = iteration + 1;
            let rhs = (0..variables)
                .map(|index| {
                    -problem.linear_term[index]
                        + self.config.rho * (projected[index] - scaled_dual[index])
                        + equality_rhs[index]
                })
                .collect::<Vec<_>>();
            values = solve_dense_system(&normal, &rhs, self.config.pivot_tolerance)?;
            let previous_projected = projected.clone();
            for index in 0..variables {
                projected[index] = (values[index] + scaled_dual[index])
                    .clamp(problem.lower_bounds[index], problem.upper_bounds[index]);
                scaled_dual[index] += values[index] - projected[index];
            }
            primal_residual = values
                .iter()
                .zip(projected.iter())
                .map(|(value, projected)| (value - projected).abs())
                .fold(0.0, f64::max);
            dual_residual = projected
                .iter()
                .zip(previous_projected.iter())
                .map(|(value, previous)| self.config.rho * (value - previous).abs())
                .fold(0.0, f64::max);
            let equality = problem.equality_matrix.multiply(&projected)?;
            maximum_equality_residual = equality
                .iter()
                .zip(problem.equality_target.iter())
                .map(|(value, target)| (value - target).abs())
                .fold(0.0, f64::max);
            let scale = projected
                .iter()
                .map(|value| value.abs())
                .fold(1.0, f64::max);
            let tolerance = self.config.absolute_tolerance + self.config.relative_tolerance * scale;
            if primal_residual <= tolerance
                && dual_residual <= tolerance
                && maximum_equality_residual <= tolerance
            {
                converged = true;
                break;
            }
        }

        let maximum_bound_violation = (0..variables)
            .map(|index| {
                (problem.lower_bounds[index] - projected[index])
                    .max(projected[index] - problem.upper_bounds[index])
                    .max(0.0)
            })
            .fold(0.0, f64::max);
        let equality = problem.equality_matrix.multiply(&projected)?;
        let equality_multipliers = equality
            .iter()
            .zip(problem.equality_target.iter())
            .map(|(value, target)| self.config.equality_penalty * (value - target))
            .collect::<Vec<_>>();
        let objective = (0..variables)
            .map(|index| {
                0.5 * problem.diagonal_hessian[index] * projected[index].powi(2)
                    + problem.linear_term[index] * projected[index]
            })
            .sum();
        let tolerance = self.config.absolute_tolerance.max(1.0e-12);
        converged &= maximum_equality_residual <= tolerance.max(self.config.relative_tolerance)
            && maximum_bound_violation <= tolerance;

        self.workspace = Some(ReferenceWorkspace {
            structure_fingerprint: problem.structure_fingerprint,
            values: projected.clone(),
            projected: projected.clone(),
            scaled_dual,
        });
        Some(InProcessSparseQpEngineResult {
            values: projected,
            equality_multipliers,
            iterations,
            maximum_equality_residual,
            maximum_bound_violation,
            primal_residual,
            dual_residual,
            objective,
            converged,
            warm_start_used: warm_start_valid || symbolic_pattern_reused,
            symbolic_pattern_reused,
        })
    }
}

fn csc_column_dot(matrix: &crate::csc_qp::CscMatrix, left: usize, right: usize) -> Option<f64> {
    if !matrix.validate() || left >= matrix.columns || right >= matrix.columns {
        return None;
    }
    let mut left_index = matrix.column_offsets[left];
    let left_end = matrix.column_offsets[left + 1];
    let mut right_index = matrix.column_offsets[right];
    let right_end = matrix.column_offsets[right + 1];
    let mut result = 0.0;
    while left_index < left_end && right_index < right_end {
        let left_row = matrix.row_indices[left_index];
        let right_row = matrix.row_indices[right_index];
        if left_row == right_row {
            result += matrix.values[left_index] * matrix.values[right_index];
            left_index += 1;
            right_index += 1;
        } else if left_row < right_row {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    Some(result)
}

fn solve_dense_system(matrix: &[Vec<f64>], rhs: &[f64], pivot_tolerance: f64) -> Option<Vec<f64>> {
    let dimension = matrix.len();
    if dimension == 0
        || rhs.len() != dimension
        || matrix.iter().any(|row| row.len() != dimension)
        || matrix.iter().flatten().any(|value| !value.is_finite())
        || rhs.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let mut augmented = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, value)| {
            let mut row = row.clone();
            row.push(*value);
            row
        })
        .collect::<Vec<_>>();
    for column in 0..dimension {
        let pivot = (column..dimension).max_by(|left, right| {
            augmented[*left][column]
                .abs()
                .partial_cmp(&augmented[*right][column].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if augmented[pivot][column].abs() <= pivot_tolerance {
            return None;
        }
        augmented.swap(column, pivot);
        let scale = augmented[column][column];
        for index in column..=dimension {
            augmented[column][index] /= scale;
        }
        for row in 0..dimension {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            if factor.abs() <= pivot_tolerance {
                continue;
            }
            for index in column..=dimension {
                augmented[row][index] -= factor * augmented[column][index];
            }
        }
    }
    Some(augmented.into_iter().map(|row| row[dimension]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem(target: f64) -> DenseEqualityQuadraticProgram {
        DenseEqualityQuadraticProgram {
            diagonal_hessian: vec![1.0, 1.0],
            linear_term: vec![0.0, 0.0],
            lower_bounds: vec![-1.0, -1.0],
            upper_bounds: vec![1.0, 1.0],
            equality_matrix: vec![vec![1.0, 1.0]],
            equality_target: vec![target],
        }
    }

    #[test]
    fn in_process_adapter_revalidates_reference_solution() {
        let backend = InProcessSparseQpBackend::new(ReferenceProjectedAdmmEngine::default())
            .unwrap()
            .with_validation_tolerance(1.0e-5);
        let sparse = CscEqualityQuadraticProgram::from_dense(&problem(0.0), 1.0e-14).unwrap();
        let result = backend
            .solve(&problem(0.0), sparse.structure_fingerprint, None)
            .unwrap();
        assert!(result.solution.converged);
        assert_eq!(result.backend_id, "symthaea-reference-projected-admm-v1");
    }

    #[test]
    fn second_solve_reuses_symbolic_pattern() {
        let backend = InProcessSparseQpBackend::new(ReferenceProjectedAdmmEngine::default())
            .unwrap()
            .with_validation_tolerance(1.0e-5);
        let sparse = CscEqualityQuadraticProgram::from_dense(&problem(0.0), 1.0e-14).unwrap();
        backend
            .solve(&problem(0.0), sparse.structure_fingerprint, None)
            .unwrap();
        let second = backend
            .solve(
                &problem(0.0),
                sparse.structure_fingerprint,
                Some(&[0.0, 0.0]),
            )
            .unwrap();
        assert!(second.symbolic_pattern_reused);
        assert!(second.solution.warm_start_used);
    }

    #[test]
    fn wrong_structure_fingerprint_fails_closed() {
        let backend =
            InProcessSparseQpBackend::new(ReferenceProjectedAdmmEngine::default()).unwrap();
        assert!(backend.solve(&problem(0.0), 999, None).is_none());
    }
}
