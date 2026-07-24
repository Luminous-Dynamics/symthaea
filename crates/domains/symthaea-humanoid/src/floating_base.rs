// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Floating-base rigid-body dynamics contracts.
//!
//! A humanoid has six unactuated base velocities followed by morphology-owned
//! joint velocities. Keeping the base rows in the dynamics equation is
//! essential: an actuated-only model can balance joint torques while silently
//! violating Newton-Euler momentum balance at the pelvis.

use serde::{Deserialize, Serialize};

use crate::full_dynamics::{
    DynamicsComponentSource, DynamicsFidelity, DynamicsProvenance, FullRigidBodyDynamicsSnapshot,
    SpatialContactJacobian,
};
use crate::morphology::HumanoidMorphology;

pub const FLOATING_BASE_DOF: usize = 6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatingBaseDynamicsSnapshot {
    pub morphology: HumanoidMorphology,
    pub sampled_at_s: f64,
    pub total_mass_kg: f64,
    pub gravity_world_mps2: [f64; 3],
    /// Number of generalized velocity coordinates (`6 + actuators`).
    pub generalized_velocity_count: usize,
    /// Row-major symmetric positive-definite `nv x nv` mass matrix.
    pub mass_matrix: Vec<f64>,
    /// Full generalized bias force vector, including the floating base.
    pub bias_force: Vec<f64>,
    /// Generalized-velocity index driven by each actuator.
    pub actuator_velocity_indices: Vec<usize>,
    pub torque_limits_nm: Vec<f64>,
    /// Row-major 6 x `nv` centroidal momentum matrix.
    pub centroidal_momentum_matrix: Vec<f64>,
    /// Spatial site Jacobians with six rows and `nv` columns.
    pub contacts: Vec<SpatialContactJacobian>,
    pub provenance: DynamicsProvenance,
    pub model_id: String,
}

impl FloatingBaseDynamicsSnapshot {
    pub fn validate(&self) -> bool {
        let actuators = self.morphology.num_actuators();
        let nv = self.generalized_velocity_count;
        self.sampled_at_s.is_finite()
            && self.total_mass_kg.is_finite()
            && self.total_mass_kg > 0.0
            && self
                .gravity_world_mps2
                .iter()
                .all(|value| value.is_finite())
            && norm3(self.gravity_world_mps2) > 1.0
            && nv == FLOATING_BASE_DOF + actuators
            && self.mass_matrix.len() == nv * nv
            && self.bias_force.len() == nv
            && self.actuator_velocity_indices.len() == actuators
            && self
                .actuator_velocity_indices
                .iter()
                .copied()
                .eq(FLOATING_BASE_DOF..nv)
            && self.torque_limits_nm.len() == actuators
            && self
                .torque_limits_nm
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && self.mass_matrix.iter().all(|value| value.is_finite())
            && self.bias_force.iter().all(|value| value.is_finite())
            && self.centroidal_momentum_matrix.len() == 6 * nv
            && self
                .centroidal_momentum_matrix
                .iter()
                .all(|value| value.is_finite())
            && self.contacts.iter().all(|contact| contact.validate(nv))
            && unique_contact_ids(&self.contacts)
            && self.provenance.is_solver_derived_core()
            && !self.model_id.trim().is_empty()
            && is_symmetric_positive_definite(&self.mass_matrix, nv)
    }

    /// Project the actuated block into the legacy full-order contract. The
    /// resulting object stays solver-derived because all core components are
    /// exact column/row projections of the validated floating-base snapshot.
    pub fn actuated_projection(&self) -> Option<FullRigidBodyDynamicsSnapshot> {
        self.validate().then_some(())?;
        let n = self.morphology.num_actuators();
        let nv = self.generalized_velocity_count;
        let mut mass_matrix = vec![0.0; n * n];
        for row in 0..n {
            let source_row = self.actuator_velocity_indices[row];
            for column in 0..n {
                let source_column = self.actuator_velocity_indices[column];
                mass_matrix[row * n + column] = self.mass_matrix[source_row * nv + source_column];
            }
        }
        let bias_torque_nm = self
            .actuator_velocity_indices
            .iter()
            .map(|index| self.bias_force[*index])
            .collect();
        let mut centroidal = vec![0.0; 6 * n];
        for row in 0..6 {
            for (column, source_column) in
                self.actuator_velocity_indices.iter().copied().enumerate()
            {
                centroidal[row * n + column] =
                    self.centroidal_momentum_matrix[row * nv + source_column];
            }
        }
        let contacts = self
            .contacts
            .iter()
            .map(|contact| SpatialContactJacobian {
                site_id: contact.site_id.clone(),
                rows: std::array::from_fn(|row| {
                    self.actuator_velocity_indices
                        .iter()
                        .map(|column| contact.rows[row][*column])
                        .collect()
                }),
                confidence: contact.confidence,
            })
            .collect();
        let projected = FullRigidBodyDynamicsSnapshot {
            morphology: self.morphology,
            sampled_at_s: self.sampled_at_s,
            total_mass_kg: self.total_mass_kg,
            gravity_world_mps2: self.gravity_world_mps2,
            mass_matrix,
            bias_torque_nm,
            torque_limits_nm: self.torque_limits_nm.clone(),
            centroidal_momentum_matrix: centroidal,
            contacts,
            fidelity: DynamicsFidelity::SolverDerived,
            provenance: self.provenance,
            model_id: format!("{}.actuated-projection-v1", self.model_id),
        };
        projected.validate().then_some(projected)
    }

    pub fn has_solver_derived_actuator_limits(&self) -> bool {
        matches!(
            self.provenance.actuator_limits,
            DynamicsComponentSource::SimulatorSolver
        )
    }
}

/// Backend-neutral access to the latest validated floating-base snapshot.
pub trait FloatingBaseDynamicsProvider {
    fn floating_base_dynamics_snapshot(&self) -> Option<FloatingBaseDynamicsSnapshot>;
}

impl<T> FloatingBaseDynamicsProvider for T
where
    T: crate::simulator::HumanoidPhysicsSimulator + ?Sized,
{
    fn floating_base_dynamics_snapshot(&self) -> Option<FloatingBaseDynamicsSnapshot> {
        crate::simulator::HumanoidPhysicsSimulator::floating_base_dynamics_snapshot(self)
    }
}

impl FloatingBaseDynamicsProvider for crate::terrain::FlatTerrain {
    fn floating_base_dynamics_snapshot(&self) -> Option<FloatingBaseDynamicsSnapshot> {
        None
    }
}

impl FloatingBaseDynamicsProvider for crate::terrain::HeightFieldTerrain {
    fn floating_base_dynamics_snapshot(&self) -> Option<FloatingBaseDynamicsSnapshot> {
        None
    }
}

fn unique_contact_ids(contacts: &[SpatialContactJacobian]) -> bool {
    let mut ids = std::collections::BTreeSet::new();
    contacts
        .iter()
        .all(|contact| ids.insert(contact.site_id.as_str()))
}

fn norm3(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn is_symmetric_positive_definite(matrix: &[f64], dimension: usize) -> bool {
    if dimension == 0 || matrix.len() != dimension * dimension {
        return false;
    }
    let scale = (0..dimension)
        .map(|index| matrix[index * dimension + index].abs())
        .fold(1.0f64, f64::max);
    let mut lower = vec![0.0; matrix.len()];
    for row in 0..dimension {
        for column in 0..=row {
            let lhs = matrix[row * dimension + column];
            let rhs = matrix[column * dimension + row];
            if (lhs - rhs).abs() > 1.0e-7 * (1.0 + lhs.abs().max(rhs.abs())) {
                return false;
            }
            let mut value = lhs;
            for k in 0..column {
                value -= lower[row * dimension + k] * lower[column * dimension + k];
            }
            if row == column {
                if !value.is_finite() || value <= scale * 1.0e-12 {
                    return false;
                }
                lower[row * dimension + column] = value.sqrt();
            } else {
                let divisor = lower[column * dimension + column];
                if !divisor.is_finite() || divisor <= 0.0 {
                    return false;
                }
                lower[row * dimension + column] = value / divisor;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_snapshot() -> FloatingBaseDynamicsSnapshot {
        let morphology = HumanoidMorphology::Dmc21;
        let nv = FLOATING_BASE_DOF + morphology.num_actuators();
        let mut mass = vec![0.0; nv * nv];
        for index in 0..nv {
            mass[index * nv + index] = 1.0 + index as f64 * 0.01;
        }
        FloatingBaseDynamicsSnapshot {
            morphology,
            sampled_at_s: 1.0,
            total_mass_kg: 70.0,
            gravity_world_mps2: [0.0, 0.0, -9.81],
            generalized_velocity_count: nv,
            mass_matrix: mass,
            bias_force: vec![0.0; nv],
            actuator_velocity_indices: (FLOATING_BASE_DOF..nv).collect(),
            torque_limits_nm: vec![100.0; morphology.num_actuators()],
            centroidal_momentum_matrix: vec![0.0; 6 * nv],
            contacts: vec![SpatialContactJacobian {
                site_id: "right_foot".to_string(),
                rows: std::array::from_fn(|_| vec![0.0; nv]),
                confidence: 1.0,
            }],
            provenance: DynamicsProvenance::mujoco_solver_with_morphology_limits(),
            model_id: "synthetic-floating-base-v1".to_string(),
        }
    }

    #[test]
    fn valid_snapshot_projects_to_actuated_contract() {
        let snapshot = synthetic_snapshot();
        assert!(snapshot.validate());
        let projected = snapshot.actuated_projection().unwrap();
        assert!(projected.validate());
        assert_eq!(projected.fidelity, DynamicsFidelity::SolverDerived);
    }

    #[test]
    fn base_rows_cannot_be_omitted() {
        let mut snapshot = synthetic_snapshot();
        snapshot.generalized_velocity_count -= 1;
        assert!(!snapshot.validate());
    }

    #[test]
    fn duplicate_contact_sites_are_rejected() {
        let mut snapshot = synthetic_snapshot();
        snapshot.contacts.push(snapshot.contacts[0].clone());
        assert!(!snapshot.validate());
    }
}
