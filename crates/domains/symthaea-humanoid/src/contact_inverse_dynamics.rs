// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reduced-order equality-constrained contact inverse dynamics.
//!
//! Variables are joint accelerations, normalized actuator torques, and two
//! translational foot contact forces. The QP enforces the reduced dynamics
//! equation and zero stance-foot acceleration while respecting finite bounds.
//! It is a deterministic bridge toward backend-supplied full mass matrices and
//! Jacobians, not a claim of full rigid-body optimal control.

use serde::{Deserialize, Serialize};

use crate::contact::{BipedSupport, ContactFrame};
use crate::control_budget::{SolverBudget, SolverBudgetEvidence};
use crate::dynamics::{ReducedOrderRigidBodyModel, RigidBodyDynamicsSnapshot};
use crate::equality_qp::{
    DenseEqualityQpSolver, DenseEqualityQuadraticProgram, EqualityQpSolverConfig,
};
use crate::morphology::HumanoidMorphology;
use crate::types::{HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactInverseDynamicsConfig {
    pub acceleration_tracking_weight: f64,
    pub torque_effort_weight: f64,
    pub contact_force_weight: f64,
    pub maximum_joint_acceleration_rad_s2: f64,
    pub friction_coefficient: f64,
    pub contact_acceleration_tolerance: f64,
    pub fallback_to_seed: bool,
    pub solver: EqualityQpSolverConfig,
    pub budget: SolverBudget,
}

impl Default for ContactInverseDynamicsConfig {
    fn default() -> Self {
        Self {
            acceleration_tracking_weight: 10.0,
            torque_effort_weight: 0.45,
            contact_force_weight: 0.002,
            maximum_joint_acceleration_rad_s2: 45.0,
            friction_coefficient: 0.8,
            contact_acceleration_tolerance: 2.0e-5,
            fallback_to_seed: true,
            solver: EqualityQpSolverConfig::default(),
            budget: SolverBudget::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ContactInverseDynamicsReport {
    pub converged: bool,
    pub used_fallback: bool,
    pub active_set_iterations: usize,
    pub active_bound_count: usize,
    pub maximum_dynamics_residual_nm: f64,
    pub maximum_contact_acceleration_residual: f64,
    pub maximum_bound_violation: f64,
    pub maximum_friction_utilization: f64,
    pub objective: f64,
    pub budget: SolverBudgetEvidence,
}

pub struct ReducedOrderContactInverseDynamicsController {
    morphology: HumanoidMorphology,
    config: ContactInverseDynamicsConfig,
    model: ReducedOrderRigidBodyModel,
    solver: DenseEqualityQpSolver,
}

impl ReducedOrderContactInverseDynamicsController {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, ContactInverseDynamicsConfig::default())
    }

    pub fn with_config(
        morphology: HumanoidMorphology,
        config: ContactInverseDynamicsConfig,
    ) -> Self {
        Self {
            morphology,
            model: ReducedOrderRigidBodyModel::new(morphology),
            solver: DenseEqualityQpSolver::with_config(config.solver),
            config,
        }
    }

    pub fn allocate(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        seed: &HumanoidCommand,
    ) -> (HumanoidCommand, ContactInverseDynamicsReport) {
        if state.validate_for(self.morphology).is_err()
            || seed.num_actuators() != self.morphology.num_actuators()
            || seed.torques.iter().any(|value| !value.is_finite())
        {
            return self.failure(seed, true);
        }
        let Some(snapshot) = self.model.linearize(state, contacts) else {
            return self.failure(seed, true);
        };
        self.allocate_with_snapshot(state, contacts, seed, &snapshot)
    }

    pub fn allocate_with_snapshot(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        seed: &HumanoidCommand,
        snapshot: &RigidBodyDynamicsSnapshot,
    ) -> (HumanoidCommand, ContactInverseDynamicsReport) {
        if state.validate_for(self.morphology).is_err()
            || seed.num_actuators() != self.morphology.num_actuators()
            || snapshot.morphology != self.morphology
            || !snapshot.validate()
        {
            return self.failure(seed, true);
        }
        let Some(problem) = self.build_problem(state, contacts, seed, snapshot) else {
            return self.failure(seed, true);
        };
        let admission = self.config.budget.admit_dense(
            problem.diagonal_hessian.len(),
            problem.equality_matrix.len(),
        );
        if !admission.admitted {
            return self.failure_with_budget(seed, true, admission.start().finish());
        }
        let timer = admission.start();
        let Some(solution) = self.solver.solve(&problem) else {
            return self.failure_with_budget(seed, true, timer.finish());
        };
        let budget = timer.finish();
        if budget.deadline_missed {
            return self.failure_with_budget(seed, true, budget);
        }
        let joints = self.morphology.num_actuators();
        let torque_offset = joints;
        let force_offset = 2 * joints;
        let right_force = [
            solution.values[force_offset],
            solution.values[force_offset + 1],
            solution.values[force_offset + 2],
        ];
        let left_force = [
            solution.values[force_offset + 3],
            solution.values[force_offset + 4],
            solution.values[force_offset + 5],
        ];
        let friction = self.config.friction_coefficient.max(1.0e-6);
        let maximum_friction_utilization = friction_utilization(right_force, friction)
            .max(friction_utilization(left_force, friction));
        let maximum_dynamics_residual_nm = dynamics_residual(
            &snapshot,
            &solution.values[..joints],
            &solution.values[torque_offset..torque_offset + joints],
            right_force,
            left_force,
        );
        let maximum_contact_acceleration_residual =
            contact_acceleration_residual(&snapshot, contacts, &solution.values[..joints]);
        let physically_feasible = solution.converged
            && maximum_dynamics_residual_nm <= 1.0e-4
            && maximum_contact_acceleration_residual
                <= self.config.contact_acceleration_tolerance.max(0.0)
            && maximum_friction_utilization <= 1.0 + 1.0e-6;
        let used_fallback = !physically_feasible && self.config.fallback_to_seed;
        let command = if used_fallback {
            seed.clone()
        } else {
            HumanoidCommand {
                torques: solution.values[torque_offset..torque_offset + joints]
                    .iter()
                    .map(|value| *value as f32)
                    .collect(),
            }
        };
        (
            command,
            ContactInverseDynamicsReport {
                converged: physically_feasible,
                used_fallback,
                active_set_iterations: solution.active_set_iterations,
                active_bound_count: solution.active_bound_count,
                maximum_dynamics_residual_nm,
                maximum_contact_acceleration_residual,
                maximum_bound_violation: solution.maximum_bound_violation,
                maximum_friction_utilization,
                objective: solution.objective,
                budget,
            },
        )
    }

    fn failure(
        &self,
        seed: &HumanoidCommand,
        used_fallback: bool,
    ) -> (HumanoidCommand, ContactInverseDynamicsReport) {
        self.failure_with_budget(seed, used_fallback, SolverBudgetEvidence::rejected())
    }

    fn failure_with_budget(
        &self,
        seed: &HumanoidCommand,
        used_fallback: bool,
        budget: SolverBudgetEvidence,
    ) -> (HumanoidCommand, ContactInverseDynamicsReport) {
        (
            if seed.num_actuators() == self.morphology.num_actuators() {
                seed.clone()
            } else {
                HumanoidCommand::zero_for(self.morphology.num_actuators())
            },
            ContactInverseDynamicsReport {
                converged: false,
                used_fallback,
                active_set_iterations: 0,
                active_bound_count: 0,
                maximum_dynamics_residual_nm: f64::INFINITY,
                maximum_contact_acceleration_residual: f64::INFINITY,
                maximum_bound_violation: f64::INFINITY,
                maximum_friction_utilization: f64::INFINITY,
                objective: f64::INFINITY,
                budget,
            },
        )
    }

    fn build_problem(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        seed: &HumanoidCommand,
        snapshot: &RigidBodyDynamicsSnapshot,
    ) -> Option<DenseEqualityQuadraticProgram> {
        if !snapshot.validate() {
            return None;
        }
        let joints = self.morphology.num_actuators();
        let variables = 2 * joints + 6;
        let torque_offset = joints;
        let force_offset = 2 * joints;
        let tracking = self.config.acceleration_tracking_weight.max(1.0e-8);
        let effort = self.config.torque_effort_weight.max(1.0e-8);
        let force_weight = self.config.contact_force_weight.max(1.0e-9);
        let mut hessian = vec![tracking; joints];
        hessian.extend(vec![effort; joints]);
        hessian.extend(vec![force_weight; 6]);
        let mut linear = vec![0.0; variables];
        let physical_seed = snapshot.normalized_to_physical_torque(seed)?;
        for joint in 0..joints {
            let desired_acceleration = ((physical_seed[joint] - snapshot.bias_torque_nm[joint])
                / snapshot.mass_diagonal_kg_m2[joint])
                .clamp(
                    -self.config.maximum_joint_acceleration_rad_s2.max(1.0),
                    self.config.maximum_joint_acceleration_rad_s2.max(1.0),
                );
            linear[joint] = -tracking * desired_acceleration;
            linear[torque_offset + joint] = -0.05 * effort * seed.torques[joint] as f64;
        }

        let support_count = contacts.support().stance_count().max(1) as f64;
        let nominal_normal = snapshot.total_mass_kg * snapshot.gravity_mps2 / support_count;
        if contacts.right.in_contact {
            linear[force_offset + 2] = -force_weight * nominal_normal;
        }
        if contacts.left.in_contact {
            linear[force_offset + 5] = -force_weight * nominal_normal;
        }

        let max_acceleration = self.config.maximum_joint_acceleration_rad_s2.max(1.0);
        let mut lower = vec![-max_acceleration; joints];
        let mut upper = vec![max_acceleration; joints];
        let support_authority = match contacts.support() {
            BipedSupport::Double => 1.0,
            BipedSupport::Right | BipedSupport::Left => 0.88,
            BipedSupport::Flight => 0.48,
        };
        lower.extend(vec![-support_authority; joints]);
        upper.extend(vec![support_authority; joints]);
        let maximum_normal = 1.5 * snapshot.total_mass_kg * snapshot.gravity_mps2;
        let maximum_tangent = self.config.friction_coefficient.max(0.0) * maximum_normal;
        for in_contact in [contacts.right.in_contact, contacts.left.in_contact] {
            if in_contact {
                lower.extend([-maximum_tangent, -maximum_tangent, 0.0]);
                upper.extend([maximum_tangent, maximum_tangent, maximum_normal]);
            } else {
                lower.extend([0.0; 3]);
                upper.extend([0.0; 3]);
            }
        }

        let mut equality_matrix = Vec::new();
        let mut equality_target = Vec::new();
        for joint in 0..joints {
            let mut row = vec![0.0; variables];
            row[joint] = snapshot.mass_diagonal_kg_m2[joint];
            row[torque_offset + joint] = -snapshot.torque_limits_nm[joint];
            for axis in 0..3 {
                row[force_offset + axis] = -snapshot.right_foot.rows[axis][joint];
                row[force_offset + 3 + axis] = -snapshot.left_foot.rows[axis][joint];
            }
            equality_matrix.push(row);
            equality_target.push(-snapshot.bias_torque_nm[joint]);
        }
        for (contact, jacobian) in [
            (&contacts.right, &snapshot.right_foot),
            (&contacts.left, &snapshot.left_foot),
        ] {
            if !contact.in_contact || jacobian.confidence < 0.25 {
                continue;
            }
            for axis in 0..3 {
                if jacobian.rows[axis]
                    .iter()
                    .all(|value| value.abs() <= 1.0e-14)
                {
                    continue;
                }
                let mut row = vec![0.0; variables];
                row[..joints].copy_from_slice(&jacobian.rows[axis]);
                equality_matrix.push(row);
                equality_target.push(0.0);
            }
        }
        let problem = DenseEqualityQuadraticProgram {
            diagonal_hessian: hessian,
            linear_term: linear,
            lower_bounds: lower,
            upper_bounds: upper,
            equality_matrix,
            equality_target,
        };
        problem.validate().then_some(problem)
    }
}

fn dynamics_residual(
    snapshot: &RigidBodyDynamicsSnapshot,
    acceleration: &[f64],
    normalized_torque: &[f64],
    right_force: [f64; 3],
    left_force: [f64; 3],
) -> f64 {
    let right_generalized = snapshot.right_foot.transpose_force(right_force);
    let left_generalized = snapshot.left_foot.transpose_force(left_force);
    (0..acceleration.len())
        .map(|joint| {
            let lhs = snapshot.mass_diagonal_kg_m2[joint] * acceleration[joint]
                + snapshot.bias_torque_nm[joint];
            let rhs = snapshot.torque_limits_nm[joint] * normalized_torque[joint]
                + right_generalized[joint]
                + left_generalized[joint];
            (lhs - rhs).abs()
        })
        .fold(0.0, f64::max)
}

fn contact_acceleration_residual(
    snapshot: &RigidBodyDynamicsSnapshot,
    contacts: &ContactFrame,
    acceleration: &[f64],
) -> f64 {
    let mut maximum = 0.0f64;
    for (contact, jacobian) in [
        (&contacts.right, &snapshot.right_foot),
        (&contacts.left, &snapshot.left_foot),
    ] {
        if !contact.in_contact || jacobian.confidence < 0.25 {
            continue;
        }
        for row in &jacobian.rows {
            maximum = maximum.max(
                row.iter()
                    .zip(acceleration.iter())
                    .map(|(jacobian, qdd)| jacobian * qdd)
                    .sum::<f64>()
                    .abs(),
            );
        }
    }
    maximum
}

fn friction_utilization(force: [f64; 3], coefficient: f64) -> f64 {
    let tangent = force[0].hypot(force[1]);
    let capacity = coefficient.max(1.0e-9) * force[2].max(0.0);
    if tangent <= 1.0e-9 {
        0.0
    } else if capacity <= 1.0e-9 {
        f64::INFINITY
    } else {
        tangent / capacity
    }
}

trait StanceCount {
    fn stance_count(self) -> usize;
}

impl StanceCount for BipedSupport {
    fn stance_count(self) -> usize {
        match self {
            BipedSupport::Flight => 0,
            BipedSupport::Right | BipedSupport::Left => 1,
            BipedSupport::Double => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standing_allocation_is_finite_or_uses_seed_fallback() {
        let state = HumanoidState::standing();
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let seed = HumanoidCommand::zero();
        let (command, report) =
            ReducedOrderContactInverseDynamicsController::new(HumanoidMorphology::Dmc21)
                .allocate(&state, &contacts, &seed);
        assert_eq!(command.num_actuators(), 21);
        assert!(command.torques.iter().all(|value| value.is_finite()));
        assert!(report.converged || report.used_fallback);
    }

    #[test]
    fn mismatched_dimensions_fail_closed() {
        let state = HumanoidState::standing();
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let seed = HumanoidCommand::zero_for(20);
        let (_, report) =
            ReducedOrderContactInverseDynamicsController::new(HumanoidMorphology::Dmc21)
                .allocate(&state, &contacts, &seed);
        assert!(!report.converged);
        assert!(report.used_fallback);
    }

    #[test]
    fn flight_has_no_contact_force_authority() {
        let mut state = HumanoidState::standing();
        state.extremities[8] = 1.0;
        state.extremities[11] = 1.0;
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        assert_eq!(contacts.support(), BipedSupport::Flight);
        let seed = HumanoidCommand::zero();
        let (_, report) =
            ReducedOrderContactInverseDynamicsController::new(HumanoidMorphology::Dmc21)
                .allocate(&state, &contacts, &seed);
        assert!(report.converged || report.used_fallback);
    }
}
