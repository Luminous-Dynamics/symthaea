// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Equality-constrained floating-base inverse dynamics.
//!
//! The solver enforces all six unactuated base equations together with joint
//! equations and stance-site acceleration constraints. It is intentionally
//! bounded and deterministic: invalid models, oversized KKT systems, solver
//! failure, or a missed wall-time budget retain the previously safe command.

use std::cell::RefCell;

use serde::{Deserialize, Serialize};

use crate::contact::ContactFrame;
use crate::control_budget::{SolverBudget, SolverBudgetEvidence};
use crate::equality_qp::{DenseEqualityQuadraticProgram, EqualityQpSolverConfig};
use crate::floating_base::{FLOATING_BASE_DOF, FloatingBaseDynamicsSnapshot};
use crate::morphology::HumanoidMorphology;
use crate::sparse_qp_backend::{DeterministicActiveSetBackend, SparseQpBackend};
use crate::types::{HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FloatingBaseInverseDynamicsConfig {
    pub generalized_acceleration_weight: f64,
    pub actuator_effort_weight: f64,
    pub contact_force_weight: f64,
    pub maximum_base_acceleration: f64,
    pub maximum_joint_acceleration: f64,
    pub friction_coefficient: f64,
    pub maximum_normal_force_scale: f64,
    pub dynamics_tolerance: f64,
    pub contact_acceleration_tolerance: f64,
    pub solver: EqualityQpSolverConfig,
    pub budget: SolverBudget,
}

impl Default for FloatingBaseInverseDynamicsConfig {
    fn default() -> Self {
        Self {
            generalized_acceleration_weight: 1.0,
            actuator_effort_weight: 0.05,
            contact_force_weight: 1.0e-4,
            maximum_base_acceleration: 40.0,
            maximum_joint_acceleration: 80.0,
            friction_coefficient: 0.8,
            maximum_normal_force_scale: 2.0,
            dynamics_tolerance: 1.0e-4,
            contact_acceleration_tolerance: 1.0e-4,
            solver: EqualityQpSolverConfig::default(),
            budget: SolverBudget {
                maximum_variables: 256,
                maximum_constraints: 256,
                maximum_estimated_operations: 80_000_000,
                maximum_elapsed_micros: 2_500,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatingBaseInverseDynamicsReport {
    pub solver_derived_model: bool,
    pub converged: bool,
    pub used_fallback: bool,
    pub active_contacts: usize,
    pub active_set_iterations: usize,
    #[serde(default)]
    pub warm_start_used: bool,
    #[serde(default)]
    pub warm_start_active_bounds: usize,
    #[serde(default)]
    pub qp_structure_fingerprint: u64,
    #[serde(default)]
    pub solver_backend_id: Option<String>,
    #[serde(default)]
    pub symbolic_pattern_reused: bool,
    pub maximum_dynamics_residual: f64,
    pub maximum_contact_acceleration_residual: f64,
    pub maximum_friction_utilization: f64,
    pub objective: f64,
    pub budget: SolverBudgetEvidence,
    pub model_id: Option<String>,
}

impl FloatingBaseInverseDynamicsReport {
    pub const fn unavailable() -> Self {
        Self {
            solver_derived_model: false,
            converged: false,
            used_fallback: false,
            active_contacts: 0,
            active_set_iterations: 0,
            warm_start_used: false,
            warm_start_active_bounds: 0,
            qp_structure_fingerprint: 0,
            solver_backend_id: None,
            symbolic_pattern_reused: false,
            maximum_dynamics_residual: 0.0,
            maximum_contact_acceleration_residual: 0.0,
            maximum_friction_utilization: 0.0,
            objective: 0.0,
            budget: SolverBudgetEvidence {
                admitted: false,
                elapsed_micros: 0,
                deadline_missed: false,
                estimated_operations: 0,
                variables: 0,
                constraints: 0,
            },
            model_id: None,
        }
    }
}

#[derive(Debug, Clone)]
struct WarmStartEntry {
    structure_fingerprint: u64,
    values: Vec<f64>,
}

pub struct FloatingBaseInverseDynamicsController {
    morphology: HumanoidMorphology,
    config: FloatingBaseInverseDynamicsConfig,
    solver_backend: Box<dyn SparseQpBackend>,
    warm_start: RefCell<Option<WarmStartEntry>>,
}

impl FloatingBaseInverseDynamicsController {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, FloatingBaseInverseDynamicsConfig::default())
    }

    pub fn with_config(
        morphology: HumanoidMorphology,
        config: FloatingBaseInverseDynamicsConfig,
    ) -> Self {
        let backend = Box::new(DeterministicActiveSetBackend::new(config.solver));
        Self::with_backend(morphology, config, backend)
    }

    pub fn with_backend(
        morphology: HumanoidMorphology,
        config: FloatingBaseInverseDynamicsConfig,
        solver_backend: Box<dyn SparseQpBackend>,
    ) -> Self {
        Self {
            morphology,
            config,
            solver_backend,
            warm_start: RefCell::new(None),
        }
    }

    pub fn allocate(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        seed: &HumanoidCommand,
        snapshot: &FloatingBaseDynamicsSnapshot,
    ) -> (HumanoidCommand, FloatingBaseInverseDynamicsReport) {
        if state.validate_for(self.morphology).is_err()
            || seed.num_actuators() != self.morphology.num_actuators()
            || seed.torques.iter().any(|value| !value.is_finite())
            || snapshot.morphology != self.morphology
            || !snapshot.validate()
        {
            return self.failure(seed, snapshot, SolverBudgetEvidence::rejected());
        }
        let Some(problem) = self.build_problem(state, contacts, seed, snapshot) else {
            return self.failure(seed, snapshot, SolverBudgetEvidence::rejected());
        };
        let admission = self.config.budget.admit_dense(
            problem.diagonal_hessian.len(),
            problem.equality_matrix.len(),
        );
        if !admission.admitted {
            return self.failure(seed, snapshot, admission.start().finish());
        }
        let structure_fingerprint = crate::csc_qp::canonical_qp_structure_fingerprint(&problem);
        let warm_values = self
            .warm_start
            .borrow()
            .as_ref()
            .filter(|entry| entry.structure_fingerprint == structure_fingerprint)
            .map(|entry| entry.values.clone());
        let timer = admission.start();
        let Some(backend_solution) =
            self.solver_backend
                .solve(&problem, structure_fingerprint, warm_values.as_deref())
        else {
            self.warm_start.borrow_mut().take();
            return self.failure_with_structure(
                seed,
                snapshot,
                timer.finish(),
                structure_fingerprint,
            );
        };
        let backend_id = backend_solution.backend_id.clone();
        let symbolic_pattern_reused = backend_solution.symbolic_pattern_reused;
        let solution = backend_solution.solution;
        let budget = timer.finish();
        if budget.deadline_missed {
            self.warm_start.borrow_mut().take();
            return self.failure_with_structure(seed, snapshot, budget, structure_fingerprint);
        }

        let nv = snapshot.generalized_velocity_count;
        let n = self.morphology.num_actuators();
        let torque_offset = nv;
        let forces = active_contact_sites(contacts, snapshot);
        let force_offset = nv + n;
        let maximum_dynamics_residual = dynamics_residual(
            snapshot,
            &solution.values[..nv],
            &solution.values[torque_offset..torque_offset + n],
            &forces,
            &solution.values[force_offset..],
        );
        let solution_values = &solution.values;
        let maximum_contact_acceleration_residual = forces
            .iter()
            .enumerate()
            .flat_map(|(contact_index, (_, jacobian))| {
                (0..3).map(move |axis| {
                    let row = &jacobian.rows[3 + axis];
                    dot(row, &solution_values[..nv]).abs() + 0.0 * contact_index as f64
                })
            })
            .fold(0.0, f64::max);
        let maximum_friction_utilization = (0..forces.len())
            .map(|index| {
                let offset = force_offset + 3 * index;
                friction_utilization(
                    [
                        solution.values[offset],
                        solution.values[offset + 1],
                        solution.values[offset + 2],
                    ],
                    self.config.friction_coefficient,
                )
            })
            .fold(0.0, f64::max);
        let physically_feasible = solution.converged
            && maximum_dynamics_residual <= self.config.dynamics_tolerance.max(0.0)
            && maximum_contact_acceleration_residual
                <= self.config.contact_acceleration_tolerance.max(0.0)
            && maximum_friction_utilization <= 1.0 + 1.0e-6;
        if !physically_feasible {
            return (
                seed.clone(),
                FloatingBaseInverseDynamicsReport {
                    solver_derived_model: true,
                    converged: false,
                    used_fallback: true,
                    active_contacts: forces.len(),
                    active_set_iterations: solution.active_set_iterations,
                    warm_start_used: solution.warm_start_used,
                    warm_start_active_bounds: solution.warm_start_active_bound_count,
                    qp_structure_fingerprint: structure_fingerprint,
                    solver_backend_id: Some(backend_id.clone()),
                    symbolic_pattern_reused,
                    maximum_dynamics_residual,
                    maximum_contact_acceleration_residual,
                    maximum_friction_utilization,
                    objective: solution.objective,
                    budget,
                    model_id: Some(snapshot.model_id.clone()),
                },
            );
        }
        let command = HumanoidCommand {
            torques: solution.values[torque_offset..torque_offset + n]
                .iter()
                .map(|value| value.clamp(-1.0, 1.0) as f32)
                .collect(),
        };
        *self.warm_start.borrow_mut() = Some(WarmStartEntry {
            structure_fingerprint,
            values: solution.values.clone(),
        });
        (
            command,
            FloatingBaseInverseDynamicsReport {
                solver_derived_model: true,
                converged: true,
                used_fallback: false,
                active_contacts: forces.len(),
                active_set_iterations: solution.active_set_iterations,
                warm_start_used: solution.warm_start_used,
                warm_start_active_bounds: solution.warm_start_active_bound_count,
                qp_structure_fingerprint: structure_fingerprint,
                solver_backend_id: Some(backend_id),
                symbolic_pattern_reused,
                maximum_dynamics_residual,
                maximum_contact_acceleration_residual,
                maximum_friction_utilization,
                objective: solution.objective,
                budget,
                model_id: Some(snapshot.model_id.clone()),
            },
        )
    }

    fn build_problem(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        seed: &HumanoidCommand,
        snapshot: &FloatingBaseDynamicsSnapshot,
    ) -> Option<DenseEqualityQuadraticProgram> {
        let nv = snapshot.generalized_velocity_count;
        let n = self.morphology.num_actuators();
        let active_contacts = active_contact_sites(contacts, snapshot);
        if active_contacts.is_empty() {
            return None;
        }
        let variables = nv + n + 3 * active_contacts.len();
        let torque_offset = nv;
        let force_offset = nv + n;
        let acceleration_weight = self.config.generalized_acceleration_weight.max(1.0e-8);
        let effort_weight = self.config.actuator_effort_weight.max(1.0e-8);
        let force_weight = self.config.contact_force_weight.max(1.0e-10);
        let mut hessian = vec![acceleration_weight; nv];
        hessian.extend(vec![effort_weight; n]);
        hessian.extend(vec![force_weight; 3 * active_contacts.len()]);
        let mut linear = vec![0.0; variables];

        let base_velocity = [
            state.root_linear_velocity[0],
            state.root_linear_velocity[1],
            state.root_linear_velocity[2],
            state.root_angular_velocity[0],
            state.root_angular_velocity[1],
            state.root_angular_velocity[2],
        ];
        for axis in 0..FLOATING_BASE_DOF {
            linear[axis] = acceleration_weight * 3.0 * base_velocity[axis];
        }
        for joint in 0..n {
            let coordinate = snapshot.actuator_velocity_indices[joint];
            let diagonal = snapshot.mass_matrix[coordinate * nv + coordinate].max(1.0e-6);
            let desired_acceleration = (seed.torques[joint] as f64
                * snapshot.torque_limits_nm[joint]
                - snapshot.bias_force[coordinate])
                / diagonal;
            linear[coordinate] = -acceleration_weight
                * desired_acceleration.clamp(
                    -self.config.maximum_joint_acceleration,
                    self.config.maximum_joint_acceleration,
                );
            linear[torque_offset + joint] = -0.1 * effort_weight * seed.torques[joint] as f64;
        }

        let support_count = active_contacts.len() as f64;
        let nominal_normal =
            snapshot.total_mass_kg * snapshot.gravity_world_mps2[2].abs() / support_count.max(1.0);
        for contact in 0..active_contacts.len() {
            linear[force_offset + 3 * contact + 2] = -force_weight * nominal_normal;
        }

        let mut lower = vec![-self.config.maximum_base_acceleration.max(1.0); nv];
        let mut upper = vec![self.config.maximum_base_acceleration.max(1.0); nv];
        for coordinate in FLOATING_BASE_DOF..nv {
            lower[coordinate] = -self.config.maximum_joint_acceleration.max(1.0);
            upper[coordinate] = self.config.maximum_joint_acceleration.max(1.0);
        }
        lower.extend(vec![-1.0; n]);
        upper.extend(vec![1.0; n]);
        let maximum_normal = self.config.maximum_normal_force_scale.max(1.0)
            * snapshot.total_mass_kg
            * snapshot.gravity_world_mps2[2].abs();
        let maximum_tangent = self.config.friction_coefficient.max(0.0) * maximum_normal;
        for _ in 0..active_contacts.len() {
            lower.extend([-maximum_tangent, -maximum_tangent, 0.0]);
            upper.extend([maximum_tangent, maximum_tangent, maximum_normal]);
        }

        let mut equality_matrix = Vec::new();
        let mut equality_target = Vec::new();
        for row_index in 0..nv {
            let mut row = vec![0.0; variables];
            row[..nv].copy_from_slice(&snapshot.mass_matrix[row_index * nv..(row_index + 1) * nv]);
            if let Some(actuator) = snapshot
                .actuator_velocity_indices
                .iter()
                .position(|coordinate| *coordinate == row_index)
            {
                row[torque_offset + actuator] = -snapshot.torque_limits_nm[actuator];
            }
            for (contact_index, (_, jacobian)) in active_contacts.iter().enumerate() {
                for axis in 0..3 {
                    row[force_offset + 3 * contact_index + axis] =
                        -jacobian.rows[3 + axis][row_index];
                }
            }
            equality_matrix.push(row);
            equality_target.push(-snapshot.bias_force[row_index]);
        }
        for (_, jacobian) in &active_contacts {
            for axis in 0..3 {
                let mut row = vec![0.0; variables];
                row[..nv].copy_from_slice(&jacobian.rows[3 + axis]);
                equality_matrix.push(row);
                equality_target.push(0.0);
            }
        }

        Some(DenseEqualityQuadraticProgram {
            diagonal_hessian: hessian,
            linear_term: linear,
            lower_bounds: lower,
            upper_bounds: upper,
            equality_matrix,
            equality_target,
        })
    }

    fn failure(
        &self,
        seed: &HumanoidCommand,
        snapshot: &FloatingBaseDynamicsSnapshot,
        budget: SolverBudgetEvidence,
    ) -> (HumanoidCommand, FloatingBaseInverseDynamicsReport) {
        self.failure_with_structure(seed, snapshot, budget, 0)
    }

    fn failure_with_structure(
        &self,
        seed: &HumanoidCommand,
        snapshot: &FloatingBaseDynamicsSnapshot,
        budget: SolverBudgetEvidence,
        structure_fingerprint: u64,
    ) -> (HumanoidCommand, FloatingBaseInverseDynamicsReport) {
        (
            if seed.num_actuators() == self.morphology.num_actuators() {
                seed.clone()
            } else {
                HumanoidCommand::zero_for(self.morphology.num_actuators())
            },
            FloatingBaseInverseDynamicsReport {
                solver_derived_model: snapshot.validate(),
                converged: false,
                used_fallback: true,
                active_contacts: 0,
                active_set_iterations: 0,
                warm_start_used: false,
                warm_start_active_bounds: 0,
                qp_structure_fingerprint: structure_fingerprint,
                solver_backend_id: Some(self.solver_backend.backend_id().to_string()),
                symbolic_pattern_reused: false,
                maximum_dynamics_residual: f64::INFINITY,
                maximum_contact_acceleration_residual: f64::INFINITY,
                maximum_friction_utilization: f64::INFINITY,
                objective: f64::INFINITY,
                budget,
                model_id: snapshot.validate().then(|| snapshot.model_id.clone()),
            },
        )
    }
}

fn active_contact_sites<'a>(
    contacts: &ContactFrame,
    snapshot: &'a FloatingBaseDynamicsSnapshot,
) -> Vec<(
    &'static str,
    &'a crate::full_dynamics::SpatialContactJacobian,
)> {
    let mut active = Vec::new();
    for (enabled, names, label) in [
        (
            contacts.right.in_contact,
            ["r_foot_site", "right_foot"],
            "right",
        ),
        (
            contacts.left.in_contact,
            ["l_foot_site", "left_foot"],
            "left",
        ),
    ] {
        if !enabled {
            continue;
        }
        if let Some(jacobian) = snapshot
            .contacts
            .iter()
            .find(|jacobian| names.contains(&jacobian.site_id.as_str()))
        {
            active.push((label, jacobian));
        }
    }
    active
}

fn dynamics_residual(
    snapshot: &FloatingBaseDynamicsSnapshot,
    acceleration: &[f64],
    normalized_torque: &[f64],
    contacts: &[(&str, &crate::full_dynamics::SpatialContactJacobian)],
    contact_forces: &[f64],
) -> f64 {
    let nv = snapshot.generalized_velocity_count;
    let mut maximum = 0.0f64;
    for row in 0..nv {
        let mut value = dot(
            &snapshot.mass_matrix[row * nv..(row + 1) * nv],
            acceleration,
        ) + snapshot.bias_force[row];
        if let Some(actuator) = snapshot
            .actuator_velocity_indices
            .iter()
            .position(|coordinate| *coordinate == row)
        {
            value -= normalized_torque[actuator] * snapshot.torque_limits_nm[actuator];
        }
        for (contact_index, (_, jacobian)) in contacts.iter().enumerate() {
            for axis in 0..3 {
                value -= jacobian.rows[3 + axis][row] * contact_forces[3 * contact_index + axis];
            }
        }
        maximum = maximum.max(value.abs());
    }
    maximum
}

fn friction_utilization(force: [f64; 3], coefficient: f64) -> f64 {
    let tangent = force[0].hypot(force[1]);
    let capacity = coefficient.max(1.0e-9) * force[2].max(0.0);
    if capacity <= 1.0e-9 {
        if tangent <= 1.0e-9 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        tangent / capacity
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::ContactFrame;
    use crate::floating_base::FLOATING_BASE_DOF;
    use crate::full_dynamics::{DynamicsProvenance, SpatialContactJacobian};

    fn static_snapshot() -> FloatingBaseDynamicsSnapshot {
        let morphology = HumanoidMorphology::Dmc21;
        let nv = FLOATING_BASE_DOF + morphology.num_actuators();
        let mut mass = vec![0.0; nv * nv];
        for index in 0..nv {
            mass[index * nv + index] = 1.0;
        }
        let mut right = std::array::from_fn(|_| vec![0.0; nv]);
        let mut left = std::array::from_fn(|_| vec![0.0; nv]);
        right[5][2] = 0.5;
        left[5][2] = 0.5;
        FloatingBaseDynamicsSnapshot {
            morphology,
            sampled_at_s: 0.0,
            total_mass_kg: 1.0,
            gravity_world_mps2: [0.0, 0.0, -9.81],
            generalized_velocity_count: nv,
            mass_matrix: mass,
            bias_force: {
                let mut bias = vec![0.0; nv];
                bias[2] = 9.81;
                bias
            },
            actuator_velocity_indices: (FLOATING_BASE_DOF..nv).collect(),
            torque_limits_nm: vec![10.0; morphology.num_actuators()],
            centroidal_momentum_matrix: vec![0.0; 6 * nv],
            contacts: vec![
                SpatialContactJacobian {
                    site_id: "r_foot_site".to_string(),
                    rows: right,
                    confidence: 1.0,
                },
                SpatialContactJacobian {
                    site_id: "l_foot_site".to_string(),
                    rows: left,
                    confidence: 1.0,
                },
            ],
            provenance: DynamicsProvenance::mujoco_solver_with_morphology_limits(),
            model_id: "static-test".to_string(),
        }
    }

    #[test]
    fn invalid_model_falls_back_without_changing_seed() {
        let morphology = HumanoidMorphology::Dmc21;
        let state = HumanoidState::standing_for(morphology);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let seed = HumanoidCommand::zero_for(morphology.num_actuators());
        let mut snapshot = static_snapshot();
        snapshot.mass_matrix.clear();
        let (command, report) = FloatingBaseInverseDynamicsController::new(morphology)
            .allocate(&state, &contacts, &seed, &snapshot);
        assert_eq!(command.torques, seed.torques);
        assert!(report.used_fallback);
    }

    #[test]
    fn problem_contains_unactuated_base_equations() {
        let morphology = HumanoidMorphology::Dmc21;
        let state = HumanoidState::standing_for(morphology);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let seed = HumanoidCommand::zero_for(morphology.num_actuators());
        let snapshot = static_snapshot();
        let controller = FloatingBaseInverseDynamicsController::new(morphology);
        let problem = controller
            .build_problem(&state, &contacts, &seed, &snapshot)
            .unwrap();
        assert!(problem.equality_matrix.len() >= snapshot.generalized_velocity_count);
        for row in 0..FLOATING_BASE_DOF {
            assert!(problem.equality_matrix[row][row].abs() > 0.0);
        }
    }
}
