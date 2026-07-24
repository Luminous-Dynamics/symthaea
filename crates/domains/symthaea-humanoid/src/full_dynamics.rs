// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full-order rigid-body dynamics contracts.
//!
//! These contracts are intentionally backend-neutral. A MuJoCo backend may fill
//! them from `mj_fullM`, inverse dynamics, and site Jacobians; identified
//! hardware may provide a bounded approximation. The controller never treats a
//! reduced-order fallback as solver-derived truth.

use serde::{Deserialize, Serialize};

use crate::contact::ContactFrame;
use crate::dynamics::RigidBodyDynamicsSnapshot;
use crate::morphology::HumanoidMorphology;
use crate::types::HumanoidState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynamicsFidelity {
    ReducedOrder,
    Identified,
    SolverDerived,
}

/// Provenance for one physical component of a dynamics snapshot.
///
/// A backend may legitimately combine solver-derived mass and Jacobian data
/// with identified actuator limits. Recording provenance per component avoids
/// promoting the entire snapshot to solver truth when only part of it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynamicsComponentSource {
    MorphologyPrior,
    ReducedOrderModel,
    SystemIdentification,
    SimulatorSolver,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicsProvenance {
    pub mass_matrix: DynamicsComponentSource,
    pub bias_forces: DynamicsComponentSource,
    pub contact_jacobians: DynamicsComponentSource,
    pub centroidal_matrix: DynamicsComponentSource,
    pub actuator_limits: DynamicsComponentSource,
}

impl DynamicsProvenance {
    pub const fn reduced_order() -> Self {
        Self {
            mass_matrix: DynamicsComponentSource::ReducedOrderModel,
            bias_forces: DynamicsComponentSource::ReducedOrderModel,
            contact_jacobians: DynamicsComponentSource::ReducedOrderModel,
            centroidal_matrix: DynamicsComponentSource::ReducedOrderModel,
            actuator_limits: DynamicsComponentSource::MorphologyPrior,
        }
    }

    pub const fn mujoco_solver_with_morphology_limits() -> Self {
        Self {
            mass_matrix: DynamicsComponentSource::SimulatorSolver,
            bias_forces: DynamicsComponentSource::SimulatorSolver,
            contact_jacobians: DynamicsComponentSource::SimulatorSolver,
            centroidal_matrix: DynamicsComponentSource::SimulatorSolver,
            actuator_limits: DynamicsComponentSource::MorphologyPrior,
        }
    }

    pub const fn is_solver_derived_core(self) -> bool {
        matches!(self.mass_matrix, DynamicsComponentSource::SimulatorSolver)
            && matches!(self.bias_forces, DynamicsComponentSource::SimulatorSolver)
            && matches!(
                self.contact_jacobians,
                DynamicsComponentSource::SimulatorSolver
            )
            && matches!(
                self.centroidal_matrix,
                DynamicsComponentSource::SimulatorSolver
            )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialContactJacobian {
    pub site_id: String,
    /// Row-major 6 x N Jacobian: angular xyz followed by linear xyz.
    pub rows: [Vec<f64>; 6],
    pub confidence: f64,
}

impl SpatialContactJacobian {
    pub fn validate(&self, joints: usize) -> bool {
        !self.site_id.trim().is_empty()
            && self
                .rows
                .iter()
                .all(|row| row.len() == joints && row.iter().all(|value| value.is_finite()))
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullRigidBodyDynamicsSnapshot {
    pub morphology: HumanoidMorphology,
    pub sampled_at_s: f64,
    pub total_mass_kg: f64,
    pub gravity_world_mps2: [f64; 3],
    /// Row-major symmetric positive-definite N x N actuated mass matrix.
    pub mass_matrix: Vec<f64>,
    /// Coriolis, centrifugal, passive, and gravity generalized forces.
    pub bias_torque_nm: Vec<f64>,
    pub torque_limits_nm: Vec<f64>,
    /// Row-major 6 x N centroidal momentum matrix.
    pub centroidal_momentum_matrix: Vec<f64>,
    pub contacts: Vec<SpatialContactJacobian>,
    pub fidelity: DynamicsFidelity,
    pub provenance: DynamicsProvenance,
    pub model_id: String,
}

impl FullRigidBodyDynamicsSnapshot {
    pub fn validate(&self) -> bool {
        let joints = self.morphology.num_actuators();
        self.sampled_at_s.is_finite()
            && self.total_mass_kg.is_finite()
            && self.total_mass_kg > 0.0
            && self
                .gravity_world_mps2
                .iter()
                .all(|value| value.is_finite())
            && norm3(self.gravity_world_mps2) > 1.0
            && self.mass_matrix.len() == joints * joints
            && self.bias_torque_nm.len() == joints
            && self.torque_limits_nm.len() == joints
            && self.centroidal_momentum_matrix.len() == 6 * joints
            && self.mass_matrix.iter().all(|value| value.is_finite())
            && self.bias_torque_nm.iter().all(|value| value.is_finite())
            && self
                .torque_limits_nm
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && self
                .centroidal_momentum_matrix
                .iter()
                .all(|value| value.is_finite())
            && self.contacts.iter().all(|contact| contact.validate(joints))
            && fidelity_matches_provenance(self.fidelity, self.provenance)
            && !self.model_id.trim().is_empty()
            && is_symmetric_positive_definite(&self.mass_matrix, joints)
    }

    pub fn mass(&self, row: usize, column: usize) -> Option<f64> {
        let joints = self.morphology.num_actuators();
        (row < joints && column < joints).then_some(self.mass_matrix[row * joints + column])
    }

    pub fn centroidal_row(&self, row: usize) -> Option<&[f64]> {
        let joints = self.morphology.num_actuators();
        (row < 6).then(|| &self.centroidal_momentum_matrix[row * joints..(row + 1) * joints])
    }

    /// Explicitly convert the older diagonal snapshot into a reduced-order full
    /// matrix. The fidelity marker prevents this from being confused with a
    /// simulator-derived model.
    pub fn from_reduced(snapshot: &RigidBodyDynamicsSnapshot) -> Option<Self> {
        if !snapshot.validate() {
            return None;
        }
        let joints = snapshot.morphology.num_actuators();
        let mut mass_matrix = vec![0.0; joints * joints];
        for joint in 0..joints {
            mass_matrix[joint * joints + joint] = snapshot.mass_diagonal_kg_m2[joint];
        }
        let mut centroidal = vec![0.0; 6 * joints];
        // Conservative deterministic coupling. Backends should replace this
        // with a true centroidal momentum matrix whenever available.
        for joint in 0..joints {
            let scale = snapshot.mass_diagonal_kg_m2[joint].sqrt();
            centroidal[0 * joints + joint] = 0.04 * scale;
            centroidal[1 * joints + joint] = (if joint % 2 == 0 { 0.03 } else { -0.03 }) * scale;
            centroidal[2 * joints + joint] = 0.02 * scale;
            centroidal[3 * joints + joint] = 0.01 * scale;
            centroidal[4 * joints + joint] = (if joint % 2 == 0 { 0.015 } else { -0.015 }) * scale;
            centroidal[5 * joints + joint] = 0.005 * scale;
        }
        let contacts = [
            (&snapshot.right_foot, "right_foot"),
            (&snapshot.left_foot, "left_foot"),
        ]
        .into_iter()
        .map(|(source, site_id)| {
            let zeros = vec![0.0; joints];
            SpatialContactJacobian {
                site_id: site_id.to_string(),
                rows: [
                    zeros.clone(),
                    zeros.clone(),
                    zeros,
                    source.rows[0].clone(),
                    source.rows[1].clone(),
                    source.rows[2].clone(),
                ],
                confidence: source.confidence,
            }
        })
        .collect();
        let full = Self {
            morphology: snapshot.morphology,
            sampled_at_s: snapshot.sampled_at_s,
            total_mass_kg: snapshot.total_mass_kg,
            gravity_world_mps2: [0.0, 0.0, -snapshot.gravity_mps2],
            mass_matrix,
            bias_torque_nm: snapshot.bias_torque_nm.clone(),
            torque_limits_nm: snapshot.torque_limits_nm.clone(),
            centroidal_momentum_matrix: centroidal,
            contacts,
            fidelity: DynamicsFidelity::ReducedOrder,
            provenance: DynamicsProvenance::reduced_order(),
            model_id: format!("{}.full-adapter-v1", snapshot.model_id),
        };
        full.validate().then_some(full)
    }
}

pub trait FullRigidBodyDynamicsProvider {
    fn full_dynamics_snapshot(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
    ) -> Option<FullRigidBodyDynamicsSnapshot>;
}

impl<T> FullRigidBodyDynamicsProvider for T
where
    T: crate::simulator::HumanoidPhysicsSimulator + ?Sized,
{
    fn full_dynamics_snapshot(
        &self,
        _state: &HumanoidState,
        _contacts: &ContactFrame,
    ) -> Option<FullRigidBodyDynamicsSnapshot> {
        crate::simulator::HumanoidPhysicsSimulator::full_dynamics_snapshot(self)
    }
}

impl FullRigidBodyDynamicsProvider for crate::terrain::FlatTerrain {
    fn full_dynamics_snapshot(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
    ) -> Option<FullRigidBodyDynamicsSnapshot> {
        let morphology = morphology_from_state(state)?;
        let reduced = crate::dynamics::ReducedOrderRigidBodyModel::new(morphology)
            .linearize(state, contacts)?;
        FullRigidBodyDynamicsSnapshot::from_reduced(&reduced)
    }
}

impl FullRigidBodyDynamicsProvider for crate::terrain::HeightFieldTerrain {
    fn full_dynamics_snapshot(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
    ) -> Option<FullRigidBodyDynamicsSnapshot> {
        let morphology = morphology_from_state(state)?;
        let reduced = crate::dynamics::ReducedOrderRigidBodyModel::new(morphology)
            .linearize(state, contacts)?;
        FullRigidBodyDynamicsSnapshot::from_reduced(&reduced)
    }
}

fn fidelity_matches_provenance(fidelity: DynamicsFidelity, provenance: DynamicsProvenance) -> bool {
    match fidelity {
        DynamicsFidelity::ReducedOrder => !provenance.is_solver_derived_core(),
        DynamicsFidelity::Identified => {
            matches!(
                provenance.mass_matrix,
                DynamicsComponentSource::SystemIdentification
            ) || matches!(
                provenance.bias_forces,
                DynamicsComponentSource::SystemIdentification
            )
        }
        DynamicsFidelity::SolverDerived => provenance.is_solver_derived_core(),
    }
}

fn morphology_from_state(state: &HumanoidState) -> Option<HumanoidMorphology> {
    [
        HumanoidMorphology::Dmc21,
        HumanoidMorphology::WithNeckWrist,
        HumanoidMorphology::Dexterous53,
        HumanoidMorphology::FullSpine,
    ]
    .into_iter()
    .find(|morphology| state.validate_for(*morphology).is_ok())
}

fn is_symmetric_positive_definite(matrix: &[f64], dimension: usize) -> bool {
    if matrix.len() != dimension * dimension || dimension == 0 {
        return false;
    }
    for row in 0..dimension {
        for column in 0..dimension {
            let left = matrix[row * dimension + column];
            let right = matrix[column * dimension + row];
            if (left - right).abs() > 1.0e-7 * (1.0 + left.abs().max(right.abs())) {
                return false;
            }
        }
    }

    // Cholesky factorization is a direct, deterministic SPD check. A tiny
    // scale-relative floor rejects singular and numerically indefinite models.
    let diagonal_scale = (0..dimension)
        .map(|index| matrix[index * dimension + index].abs())
        .fold(0.0f64, f64::max)
        .max(1.0);
    let floor = diagonal_scale * 1.0e-12;
    let mut lower = vec![0.0f64; dimension * dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[row * dimension + column];
            for k in 0..column {
                value -= lower[row * dimension + k] * lower[column * dimension + k];
            }
            if row == column {
                if !value.is_finite() || value <= floor {
                    return false;
                }
                lower[row * dimension + column] = value.sqrt();
            } else {
                let divisor = lower[column * dimension + column];
                if divisor <= 0.0 || !divisor.is_finite() {
                    return false;
                }
                lower[row * dimension + column] = value / divisor;
            }
        }
    }
    true
}

fn norm3(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::ContactFrame;
    use crate::dynamics::ReducedOrderRigidBodyModel;
    use crate::types::HumanoidState;

    #[test]
    fn reduced_adapter_is_explicit_and_dimensionally_valid() {
        let morphology = HumanoidMorphology::Dmc21;
        let state = HumanoidState::default_for(morphology);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let reduced = ReducedOrderRigidBodyModel::new(morphology)
            .linearize(&state, &contacts)
            .unwrap();
        let full = FullRigidBodyDynamicsSnapshot::from_reduced(&reduced).unwrap();
        assert!(full.validate());
        assert_eq!(full.fidelity, DynamicsFidelity::ReducedOrder);
        assert_eq!(full.provenance, DynamicsProvenance::reduced_order());
        assert_eq!(full.mass_matrix.len(), morphology.num_actuators().pow(2));
    }

    #[test]
    fn asymmetric_mass_matrix_is_rejected() {
        let morphology = HumanoidMorphology::Dmc21;
        let state = HumanoidState::default_for(morphology);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let reduced = ReducedOrderRigidBodyModel::new(morphology)
            .linearize(&state, &contacts)
            .unwrap();
        let mut full = FullRigidBodyDynamicsSnapshot::from_reduced(&reduced).unwrap();
        full.mass_matrix[1] = 1.0;
        assert!(!full.validate());
    }

    #[test]
    fn symmetric_but_indefinite_mass_matrix_is_rejected() {
        let morphology = HumanoidMorphology::Dmc21;
        let state = HumanoidState::default_for(morphology);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let reduced = ReducedOrderRigidBodyModel::new(morphology)
            .linearize(&state, &contacts)
            .unwrap();
        let mut full = FullRigidBodyDynamicsSnapshot::from_reduced(&reduced).unwrap();
        let joints = morphology.num_actuators();
        full.mass_matrix[0] = 1.0;
        full.mass_matrix[1] = 2.0;
        full.mass_matrix[joints] = 2.0;
        full.mass_matrix[joints + 1] = 1.0;
        assert!(!full.validate());
    }

    #[test]
    fn solver_fidelity_requires_solver_provenance_for_every_core_component() {
        let morphology = HumanoidMorphology::Dmc21;
        let state = HumanoidState::default_for(morphology);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let reduced = ReducedOrderRigidBodyModel::new(morphology)
            .linearize(&state, &contacts)
            .unwrap();
        let mut full = FullRigidBodyDynamicsSnapshot::from_reduced(&reduced).unwrap();
        full.fidelity = DynamicsFidelity::SolverDerived;
        assert!(!full.validate());
        full.provenance = DynamicsProvenance::mujoco_solver_with_morphology_limits();
        assert!(full.validate());
    }
}
