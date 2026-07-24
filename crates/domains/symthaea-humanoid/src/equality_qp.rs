// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic equality-constrained diagonal quadratic programming.
//!
//! The solver uses a bounded active-set loop. Variable bounds are promoted to
//! equality rows one at a time, and each KKT system is solved with pivoted
//! Gaussian elimination. It is intended for small, morphology-sized control
//! problems where deterministic fallback is more important than raw throughput.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseEqualityQuadraticProgram {
    pub diagonal_hessian: Vec<f64>,
    pub linear_term: Vec<f64>,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub equality_matrix: Vec<Vec<f64>>,
    pub equality_target: Vec<f64>,
}

impl DenseEqualityQuadraticProgram {
    pub fn validate(&self) -> bool {
        let variables = self.diagonal_hessian.len();
        variables > 0
            && self.linear_term.len() == variables
            && self.lower_bounds.len() == variables
            && self.upper_bounds.len() == variables
            && self.equality_matrix.len() == self.equality_target.len()
            && self
                .diagonal_hessian
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && self.linear_term.iter().all(|value| value.is_finite())
            && self
                .lower_bounds
                .iter()
                .zip(self.upper_bounds.iter())
                .all(|(lower, upper)| lower.is_finite() && upper.is_finite() && lower <= upper)
            && self.equality_target.iter().all(|value| value.is_finite())
            && self.equality_matrix.iter().all(|row| {
                row.len() == variables
                    && row.iter().all(|value| value.is_finite())
                    && row.iter().any(|value| value.abs() > 1.0e-14)
            })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EqualityQpSolverConfig {
    pub maximum_active_set_iterations: usize,
    pub feasibility_tolerance: f64,
    pub pivot_tolerance: f64,
    pub kkt_regularization: f64,
}

impl Default for EqualityQpSolverConfig {
    fn default() -> Self {
        Self {
            maximum_active_set_iterations: 96,
            feasibility_tolerance: 1.0e-7,
            pivot_tolerance: 1.0e-12,
            kkt_regularization: 1.0e-10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqualityQpSolution {
    pub values: Vec<f64>,
    pub multipliers: Vec<f64>,
    pub active_bound_count: usize,
    /// Bounds recovered from a structurally compatible previous solution.
    pub warm_start_active_bound_count: usize,
    pub warm_start_used: bool,
    pub active_set_iterations: usize,
    pub maximum_equality_residual: f64,
    pub maximum_bound_violation: f64,
    pub objective: f64,
    pub converged: bool,
}

pub struct DenseEqualityQpSolver {
    config: EqualityQpSolverConfig,
}

impl DenseEqualityQpSolver {
    pub fn new() -> Self {
        Self::with_config(EqualityQpSolverConfig::default())
    }

    pub const fn with_config(config: EqualityQpSolverConfig) -> Self {
        Self { config }
    }

    pub fn solve(&self, problem: &DenseEqualityQuadraticProgram) -> Option<EqualityQpSolution> {
        self.solve_with_warm_start(problem, None)
    }

    /// Solve while reusing active bounds from a previous solution with the
    /// same variable ordering. The warm start never bypasses feasibility
    /// checks: cached values only seed equality rows for bounds they already
    /// touch, and the normal bounded active-set loop still validates the result.
    pub fn solve_with_warm_start(
        &self,
        problem: &DenseEqualityQuadraticProgram,
        warm_start: Option<&[f64]>,
    ) -> Option<EqualityQpSolution> {
        if !problem.validate() {
            return None;
        }
        let variables = problem.diagonal_hessian.len();
        let tolerance = self.config.feasibility_tolerance.max(0.0);
        let mut matrix = problem.equality_matrix.clone();
        let mut target = problem.equality_target.clone();
        let mut fixed = vec![false; variables];
        let mut warm_start_active_bound_count = 0usize;
        let warm_start_used = warm_start
            .map(|values| values.len() == variables && values.iter().all(|value| value.is_finite()))
            .unwrap_or(false);
        if let Some(values) = warm_start.filter(|_| warm_start_used) {
            let activation_tolerance = (10.0 * tolerance).max(1.0e-10);
            for index in 0..variables {
                let lower = problem.lower_bounds[index];
                let upper = problem.upper_bounds[index];
                let value = values[index];
                let bound = if (value - lower).abs() <= activation_tolerance || value < lower {
                    Some(lower)
                } else if (value - upper).abs() <= activation_tolerance || value > upper {
                    Some(upper)
                } else {
                    None
                };
                if let Some(bound) = bound {
                    let mut row = vec![0.0; variables];
                    row[index] = 1.0;
                    matrix.push(row);
                    target.push(bound);
                    fixed[index] = true;
                    warm_start_active_bound_count += 1;
                }
            }
        }
        let mut last_values = vec![0.0; variables];
        let mut last_multipliers = Vec::new();
        let mut iterations = 0usize;

        for iteration in 0..self.config.maximum_active_set_iterations.max(1) {
            iterations = iteration + 1;
            let (values, multipliers) = self.solve_kkt(problem, &matrix, &target)?;
            last_values = values;
            last_multipliers = multipliers;

            let mut worst: Option<(usize, f64, f64)> = None;
            for index in 0..variables {
                if fixed[index] {
                    continue;
                }
                let value = last_values[index];
                let (violation, bound) = if value < problem.lower_bounds[index] - tolerance {
                    (
                        problem.lower_bounds[index] - value,
                        problem.lower_bounds[index],
                    )
                } else if value > problem.upper_bounds[index] + tolerance {
                    (
                        value - problem.upper_bounds[index],
                        problem.upper_bounds[index],
                    )
                } else {
                    continue;
                };
                if worst
                    .map(|(_, current, _)| violation > current)
                    .unwrap_or(true)
                {
                    worst = Some((index, violation, bound));
                }
            }
            let Some((index, _, bound)) = worst else {
                break;
            };
            let mut row = vec![0.0; variables];
            row[index] = 1.0;
            matrix.push(row);
            target.push(bound);
            fixed[index] = true;
        }

        let maximum_equality_residual = matrix
            .iter()
            .zip(target.iter())
            .map(|(row, expected)| (dot(row, &last_values) - expected).abs())
            .fold(0.0, f64::max);
        let maximum_bound_violation = (0..variables)
            .map(|index| {
                if last_values[index] < problem.lower_bounds[index] {
                    problem.lower_bounds[index] - last_values[index]
                } else if last_values[index] > problem.upper_bounds[index] {
                    last_values[index] - problem.upper_bounds[index]
                } else {
                    0.0
                }
            })
            .fold(0.0, f64::max);
        let objective = (0..variables)
            .map(|index| {
                0.5 * problem.diagonal_hessian[index] * last_values[index] * last_values[index]
                    + problem.linear_term[index] * last_values[index]
            })
            .sum();
        let converged =
            maximum_equality_residual <= tolerance && maximum_bound_violation <= tolerance;
        Some(EqualityQpSolution {
            values: last_values,
            multipliers: last_multipliers,
            active_bound_count: fixed.iter().filter(|value| **value).count(),
            warm_start_active_bound_count,
            warm_start_used,
            active_set_iterations: iterations,
            maximum_equality_residual,
            maximum_bound_violation,
            objective,
            converged,
        })
    }

    fn solve_kkt(
        &self,
        problem: &DenseEqualityQuadraticProgram,
        equality_matrix: &[Vec<f64>],
        equality_target: &[f64],
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        let variables = problem.diagonal_hessian.len();
        let constraints = equality_matrix.len();
        let dimension = variables + constraints;
        let mut augmented = vec![vec![0.0; dimension + 1]; dimension];
        for index in 0..variables {
            augmented[index][index] = problem.diagonal_hessian[index];
            augmented[index][dimension] = -problem.linear_term[index];
        }
        for (constraint, row) in equality_matrix.iter().enumerate() {
            let lambda_index = variables + constraint;
            for variable in 0..variables {
                augmented[variable][lambda_index] = row[variable];
                augmented[lambda_index][variable] = row[variable];
            }
            augmented[lambda_index][lambda_index] = -self.config.kkt_regularization.max(0.0);
            augmented[lambda_index][dimension] = equality_target[constraint];
        }
        let solution = gaussian_elimination(augmented, self.config.pivot_tolerance.max(1.0e-16))?;
        if solution.iter().any(|value| !value.is_finite()) {
            return None;
        }
        Some((
            solution[..variables].to_vec(),
            solution[variables..].to_vec(),
        ))
    }
}

impl Default for DenseEqualityQpSolver {
    fn default() -> Self {
        Self::new()
    }
}

fn gaussian_elimination(mut matrix: Vec<Vec<f64>>, pivot_tolerance: f64) -> Option<Vec<f64>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n + 1) {
        return None;
    }
    for column in 0..n {
        let pivot = (column..n).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .partial_cmp(&matrix[*right][column].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if matrix[pivot][column].abs() <= pivot_tolerance {
            return None;
        }
        matrix.swap(column, pivot);
        let scale = matrix[column][column];
        for entry in column..=n {
            matrix[column][entry] /= scale;
        }
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor.abs() <= pivot_tolerance {
                continue;
            }
            for entry in column..=n {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
        }
    }
    Some(matrix.into_iter().map(|row| row[n]).collect())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_is_satisfied() {
        let problem = DenseEqualityQuadraticProgram {
            diagonal_hessian: vec![1.0, 1.0],
            linear_term: vec![-1.0, -2.0],
            lower_bounds: vec![-10.0; 2],
            upper_bounds: vec![10.0; 2],
            equality_matrix: vec![vec![1.0, 1.0]],
            equality_target: vec![1.0],
        };
        let solution = DenseEqualityQpSolver::new().solve(&problem).unwrap();
        assert!(solution.converged);
        assert!((solution.values[0] + solution.values[1] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn active_set_enforces_bounds() {
        let problem = DenseEqualityQuadraticProgram {
            diagonal_hessian: vec![1.0, 1.0],
            linear_term: vec![-10.0, 0.0],
            lower_bounds: vec![0.0, 0.0],
            upper_bounds: vec![0.25, 1.0],
            equality_matrix: vec![vec![1.0, 1.0]],
            equality_target: vec![1.0],
        };
        let solution = DenseEqualityQpSolver::new().solve(&problem).unwrap();
        assert!(solution.converged);
        assert!((solution.values[0] - 0.25).abs() < 1.0e-6);
        assert!((solution.values[1] - 0.75).abs() < 1.0e-6);
        assert_eq!(solution.active_bound_count, 1);
    }

    #[test]
    fn malformed_problem_is_rejected() {
        let problem = DenseEqualityQuadraticProgram {
            diagonal_hessian: vec![0.0],
            linear_term: vec![0.0],
            lower_bounds: vec![-1.0],
            upper_bounds: vec![1.0],
            equality_matrix: Vec::new(),
            equality_target: Vec::new(),
        };
        assert!(DenseEqualityQpSolver::new().solve(&problem).is_none());
    }

    #[test]
    fn compatible_warm_start_reuses_active_bound() {
        let problem = DenseEqualityQuadraticProgram {
            diagonal_hessian: vec![1.0, 1.0],
            linear_term: vec![-4.0, 0.0],
            lower_bounds: vec![0.0, -10.0],
            upper_bounds: vec![1.0, 10.0],
            equality_matrix: vec![vec![1.0, 1.0]],
            equality_target: vec![1.0],
        };
        let solution = DenseEqualityQpSolver::new()
            .solve_with_warm_start(&problem, Some(&[1.0, 0.0]))
            .unwrap();
        assert!(solution.converged);
        assert!(solution.warm_start_used);
        assert_eq!(solution.warm_start_active_bound_count, 1);
    }
}
