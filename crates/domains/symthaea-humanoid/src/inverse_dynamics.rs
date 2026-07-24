// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic sparse projected-QP inverse-dynamics allocation.
//!
//! This solver is intentionally dependency-free and bounded. It solves a
//! strictly convex diagonal quadratic objective with sparse linear inequality
//! constraints by repeated metric projections. The resulting contract can be
//! replaced by OSQP or a model-derived inverse-dynamics backend without
//! changing hierarchical-control callers.

use serde::{Deserialize, Serialize};

use crate::contact::{BipedSupport, ContactFrame};
use crate::morphology::HumanoidMorphology;
use crate::types::{HumanoidCommand, HumanoidState};
use crate::whole_body::{
    ConstrainedWholeBodyController, WholeBodyControlReport, WholeBodyObjective,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLinearConstraint {
    pub indices: Vec<usize>,
    pub coefficients: Vec<f64>,
    pub lower: f64,
    pub upper: f64,
    pub label: String,
}

impl SparseLinearConstraint {
    pub fn validate(&self, variables: usize) -> bool {
        !self.indices.is_empty()
            && self.indices.len() == self.coefficients.len()
            && self.indices.iter().all(|index| *index < variables)
            && self.coefficients.iter().all(|value| value.is_finite())
            && self.lower.is_finite()
            && self.upper.is_finite()
            && self.lower <= self.upper
            && self.coefficients.iter().any(|value| value.abs() > 1.0e-12)
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.indices
            .iter()
            .zip(self.coefficients.iter())
            .map(|(index, coefficient)| x[*index] * coefficient)
            .sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseQuadraticProgram {
    pub diagonal_hessian: Vec<f64>,
    pub linear_term: Vec<f64>,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub constraints: Vec<SparseLinearConstraint>,
}

impl SparseQuadraticProgram {
    pub fn validate(&self) -> bool {
        let n = self.diagonal_hessian.len();
        n > 0
            && self.linear_term.len() == n
            && self.lower_bounds.len() == n
            && self.upper_bounds.len() == n
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
            && self
                .constraints
                .iter()
                .all(|constraint| constraint.validate(n))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SparseQpSolverConfig {
    pub maximum_iterations: usize,
    pub feasibility_tolerance: f64,
    pub relaxation: f64,
}

impl Default for SparseQpSolverConfig {
    fn default() -> Self {
        Self {
            maximum_iterations: 64,
            feasibility_tolerance: 1.0e-7,
            relaxation: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseQpSolution {
    pub values: Vec<f64>,
    pub iterations: usize,
    pub active_constraints: usize,
    pub maximum_violation: f64,
    pub objective: f64,
    pub converged: bool,
}

pub struct SparseProjectedQpSolver {
    config: SparseQpSolverConfig,
}

impl SparseProjectedQpSolver {
    pub fn new() -> Self {
        Self::with_config(SparseQpSolverConfig::default())
    }

    pub const fn with_config(config: SparseQpSolverConfig) -> Self {
        Self { config }
    }

    pub fn solve(&self, problem: &SparseQuadraticProgram) -> Option<SparseQpSolution> {
        if !problem.validate() {
            return None;
        }
        let n = problem.diagonal_hessian.len();
        let mut values = vec![0.0; n];
        for i in 0..n {
            values[i] = (-problem.linear_term[i] / problem.diagonal_hessian[i])
                .clamp(problem.lower_bounds[i], problem.upper_bounds[i]);
        }

        let mut iterations = 0usize;
        let mut maximum_violation = f64::INFINITY;
        let tolerance = self.config.feasibility_tolerance.max(0.0);
        let relaxation = self.config.relaxation.clamp(0.05, 1.5);
        for iteration in 0..self.config.maximum_iterations.max(1) {
            iterations = iteration + 1;
            maximum_violation = 0.0;
            for constraint in &problem.constraints {
                let current = constraint.value(&values);
                let target = if current < constraint.lower {
                    maximum_violation = maximum_violation.max(constraint.lower - current);
                    constraint.lower
                } else if current > constraint.upper {
                    maximum_violation = maximum_violation.max(current - constraint.upper);
                    constraint.upper
                } else {
                    continue;
                };
                let denominator = constraint
                    .indices
                    .iter()
                    .zip(constraint.coefficients.iter())
                    .map(|(index, coefficient)| {
                        coefficient * coefficient / problem.diagonal_hessian[*index]
                    })
                    .sum::<f64>();
                if denominator <= 1.0e-18 {
                    continue;
                }
                let multiplier = (target - current) / denominator * relaxation;
                for (index, coefficient) in constraint
                    .indices
                    .iter()
                    .zip(constraint.coefficients.iter())
                {
                    values[*index] = (values[*index]
                        + multiplier * coefficient / problem.diagonal_hessian[*index])
                        .clamp(problem.lower_bounds[*index], problem.upper_bounds[*index]);
                }
            }
            if maximum_violation <= tolerance {
                break;
            }
        }

        let mut active_constraints = 0usize;
        maximum_violation = 0.0;
        for constraint in &problem.constraints {
            let current = constraint.value(&values);
            let violation = if current < constraint.lower {
                constraint.lower - current
            } else if current > constraint.upper {
                current - constraint.upper
            } else {
                let lower_distance = (current - constraint.lower).abs();
                let upper_distance = (constraint.upper - current).abs();
                if lower_distance <= tolerance || upper_distance <= tolerance {
                    active_constraints += 1;
                }
                0.0
            };
            maximum_violation = maximum_violation.max(violation);
        }
        for i in 0..n {
            if (values[i] - problem.lower_bounds[i]).abs() <= tolerance
                || (problem.upper_bounds[i] - values[i]).abs() <= tolerance
            {
                active_constraints += 1;
            }
        }
        let objective = (0..n)
            .map(|i| {
                0.5 * problem.diagonal_hessian[i] * values[i] * values[i]
                    + problem.linear_term[i] * values[i]
            })
            .sum();
        Some(SparseQpSolution {
            values,
            iterations,
            active_constraints,
            maximum_violation,
            objective,
            converged: maximum_violation <= tolerance,
        })
    }
}

impl Default for SparseProjectedQpSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InverseDynamicsConfig {
    pub tracking_weight: f64,
    pub effort_weight: f64,
    pub smoothness_weight: f64,
    pub bilateral_moment_limit: f64,
    pub single_support_swing_limit: f64,
    pub fallback_to_projected_allocator: bool,
    pub solver: SparseQpSolverConfig,
}

impl Default for InverseDynamicsConfig {
    fn default() -> Self {
        Self {
            tracking_weight: 8.0,
            effort_weight: 0.35,
            smoothness_weight: 1.5,
            bilateral_moment_limit: 0.55,
            single_support_swing_limit: 0.75,
            fallback_to_projected_allocator: true,
            solver: SparseQpSolverConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InverseDynamicsReport {
    pub solver_iterations: usize,
    pub active_constraints: usize,
    pub maximum_constraint_violation: f64,
    pub objective: f64,
    pub converged: bool,
    pub used_fallback: bool,
    pub seed_report: WholeBodyControlReport,
}

pub struct SparseQpInverseDynamicsController {
    morphology: HumanoidMorphology,
    config: InverseDynamicsConfig,
    seed_allocator: ConstrainedWholeBodyController,
    solver: SparseProjectedQpSolver,
    names: Vec<String>,
}

impl SparseQpInverseDynamicsController {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, InverseDynamicsConfig::default())
    }

    pub fn with_config(morphology: HumanoidMorphology, config: InverseDynamicsConfig) -> Self {
        Self {
            morphology,
            seed_allocator: ConstrainedWholeBodyController::new(morphology),
            solver: SparseProjectedQpSolver::with_config(config.solver),
            names: morphology.joint_names(),
            config,
        }
    }

    pub fn allocate(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        nominal: &HumanoidCommand,
        objective: WholeBodyObjective,
    ) -> (HumanoidCommand, InverseDynamicsReport) {
        let (seed, seed_report) = self
            .seed_allocator
            .allocate(state, contacts, nominal, objective);
        let n = self.morphology.num_actuators();
        if seed.num_actuators() != n || nominal.num_actuators() != n {
            return self.failure(seed, seed_report, true);
        }
        let problem = self.build_problem(state, contacts, nominal, &seed);
        let Some(solution) = self.solver.solve(&problem) else {
            return self.failure(seed, seed_report, true);
        };
        let used_fallback = !solution.converged && self.config.fallback_to_projected_allocator;
        let command = if used_fallback {
            seed.clone()
        } else {
            HumanoidCommand {
                torques: solution.values.iter().map(|value| *value as f32).collect(),
            }
        };
        (
            command,
            InverseDynamicsReport {
                solver_iterations: solution.iterations,
                active_constraints: solution.active_constraints,
                maximum_constraint_violation: solution.maximum_violation,
                objective: solution.objective,
                converged: solution.converged,
                used_fallback,
                seed_report,
            },
        )
    }

    fn failure(
        &self,
        seed: HumanoidCommand,
        seed_report: WholeBodyControlReport,
        used_fallback: bool,
    ) -> (HumanoidCommand, InverseDynamicsReport) {
        (
            seed,
            InverseDynamicsReport {
                solver_iterations: 0,
                active_constraints: 0,
                maximum_constraint_violation: f64::INFINITY,
                objective: f64::INFINITY,
                converged: false,
                used_fallback,
                seed_report,
            },
        )
    }

    fn build_problem(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        nominal: &HumanoidCommand,
        seed: &HumanoidCommand,
    ) -> SparseQuadraticProgram {
        let n = self.morphology.num_actuators();
        let tracking = self.config.tracking_weight.max(1.0e-6);
        let effort = self.config.effort_weight.max(0.0);
        let smoothness = self.config.smoothness_weight.max(0.0);
        let diagonal_hessian = vec![tracking + effort + smoothness; n];
        let linear_term = (0..n)
            .map(|i| -tracking * seed.torques[i] as f64 - smoothness * nominal.torques[i] as f64)
            .collect::<Vec<_>>();
        let support_authority = match contacts.support() {
            BipedSupport::Double => 1.0,
            BipedSupport::Right | BipedSupport::Left => 0.88,
            BipedSupport::Flight => 0.50,
        };
        let lower_bounds = vec![-support_authority; n];
        let upper_bounds = vec![support_authority; n];
        let mut constraints = Vec::new();
        self.add_bilateral_constraints(&mut constraints);
        self.add_support_constraints(&mut constraints, contacts.support());
        self.add_joint_limit_barriers(&mut constraints, state);
        SparseQuadraticProgram {
            diagonal_hessian,
            linear_term,
            lower_bounds,
            upper_bounds,
            constraints,
        }
    }

    fn add_bilateral_constraints(&self, constraints: &mut Vec<SparseLinearConstraint>) {
        for (right_name, left_name) in [
            ("right_hip_x", "left_hip_x"),
            ("right_hip_y", "left_hip_y"),
            ("right_ankle_x", "left_ankle_x"),
            ("right_ankle_y", "left_ankle_y"),
        ] {
            let Some(right) = self.names.iter().position(|name| name == right_name) else {
                continue;
            };
            let Some(left) = self.names.iter().position(|name| name == left_name) else {
                continue;
            };
            constraints.push(SparseLinearConstraint {
                indices: vec![right, left],
                coefficients: vec![1.0, -1.0],
                lower: -self.config.bilateral_moment_limit,
                upper: self.config.bilateral_moment_limit,
                label: format!("bilateral:{right_name}:{left_name}"),
            });
        }
    }

    fn add_support_constraints(
        &self,
        constraints: &mut Vec<SparseLinearConstraint>,
        support: BipedSupport,
    ) {
        let swing_prefix = match support {
            BipedSupport::Right => Some("left_"),
            BipedSupport::Left => Some("right_"),
            _ => None,
        };
        let Some(prefix) = swing_prefix else {
            return;
        };
        let indices = self
            .names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| name.starts_with(prefix).then_some(index))
            .collect::<Vec<_>>();
        if indices.is_empty() {
            return;
        }
        constraints.push(SparseLinearConstraint {
            coefficients: vec![1.0; indices.len()],
            indices,
            lower: -(self.config.single_support_swing_limit),
            upper: self.config.single_support_swing_limit,
            label: format!("single-support-swing:{prefix}"),
        });
    }

    fn add_joint_limit_barriers(
        &self,
        constraints: &mut Vec<SparseLinearConstraint>,
        state: &HumanoidState,
    ) {
        let limits = self.morphology.joint_limits();
        for index in 0..limits.len().min(state.joint_angles.len()) {
            let [lower, upper] = limits[index];
            let range = (upper - lower).abs().max(1.0e-6);
            let lower_distance = (state.joint_angles[index] - lower) / range;
            let upper_distance = (upper - state.joint_angles[index]) / range;
            if lower_distance < 0.08 {
                constraints.push(SparseLinearConstraint {
                    indices: vec![index],
                    coefficients: vec![1.0],
                    lower: 0.0,
                    upper: 1.0,
                    label: format!("lower-limit-barrier:{}", self.names[index]),
                });
            }
            if upper_distance < 0.08 {
                constraints.push(SparseLinearConstraint {
                    indices: vec![index],
                    coefficients: vec![1.0],
                    lower: -1.0,
                    upper: 0.0,
                    label: format!("upper-limit-barrier:{}", self.names[index]),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_solver_satisfies_coupled_constraint() {
        let problem = SparseQuadraticProgram {
            diagonal_hessian: vec![1.0, 1.0],
            linear_term: vec![-1.0, -1.0],
            lower_bounds: vec![-2.0, -2.0],
            upper_bounds: vec![2.0, 2.0],
            constraints: vec![SparseLinearConstraint {
                indices: vec![0, 1],
                coefficients: vec![1.0, 1.0],
                lower: -10.0,
                upper: 1.0,
                label: "sum".to_string(),
            }],
        };
        let solution = SparseProjectedQpSolver::new().solve(&problem).unwrap();
        assert!(solution.converged);
        assert!(solution.values[0] + solution.values[1] <= 1.0 + 1.0e-6);
    }

    #[test]
    fn joint_limit_barrier_prevents_outward_command() {
        let controller = SparseQpInverseDynamicsController::new(HumanoidMorphology::Dmc21);
        let mut state = HumanoidState::standing();
        state.joint_angles[6] = -2.79;
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let mut nominal = HumanoidCommand::zero();
        nominal.torques[6] = -1.0;
        let (command, report) =
            controller.allocate(&state, &contacts, &nominal, WholeBodyObjective::default());
        assert!(command.torques[6] >= -1.0e-6);
        assert!(report.active_constraints > 0 || report.used_fallback);
    }

    #[test]
    fn malformed_problem_is_rejected() {
        let problem = SparseQuadraticProgram {
            diagonal_hessian: vec![0.0],
            linear_term: vec![0.0],
            lower_bounds: vec![-1.0],
            upper_bounds: vec![1.0],
            constraints: Vec::new(),
        };
        assert!(SparseProjectedQpSolver::new().solve(&problem).is_none());
    }
}
