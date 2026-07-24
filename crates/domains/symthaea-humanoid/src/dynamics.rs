// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reduced-order rigid-body dynamics contracts for morphology-aware control.
//!
//! The model in this module is intentionally explicit about its fidelity. It
//! provides a deterministic, state-dependent diagonal mass model, bias torques,
//! actuator limits, and sparse foot Jacobians. Simulator backends may replace
//! it with solver-derived matrices while preserving the same validated
//! snapshot contract.

use serde::{Deserialize, Serialize};

use crate::contact::ContactFrame;
use crate::footstep::FootSide;
use crate::morphology::HumanoidMorphology;
use crate::types::{HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FootContactJacobian {
    pub side: FootSide,
    /// Translational contact Jacobian rows `[x, y, z]`, each with one value per
    /// actuated joint.
    pub rows: [Vec<f64>; 3],
    pub confidence: f64,
}

impl FootContactJacobian {
    pub fn validate(&self, joints: usize) -> bool {
        self.rows
            .iter()
            .all(|row| row.len() == joints && row.iter().all(|value| value.is_finite()))
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
    }

    pub fn transpose_force(&self, force_world_n: [f64; 3]) -> Vec<f64> {
        let joints = self.rows[0].len();
        let mut generalized = vec![0.0; joints];
        for axis in 0..3 {
            for (joint, value) in self.rows[axis].iter().enumerate() {
                generalized[joint] += value * force_world_n[axis];
            }
        }
        generalized
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigidBodyDynamicsSnapshot {
    pub morphology: HumanoidMorphology,
    pub sampled_at_s: f64,
    pub total_mass_kg: f64,
    pub gravity_mps2: f64,
    /// Diagonal approximation of the actuated joint-space mass matrix.
    pub mass_diagonal_kg_m2: Vec<f64>,
    /// Coriolis, damping, and gravity terms in actuator coordinates.
    pub bias_torque_nm: Vec<f64>,
    pub torque_limits_nm: Vec<f64>,
    pub right_foot: FootContactJacobian,
    pub left_foot: FootContactJacobian,
    /// Stable provenance label for experiment manifests and solver reports.
    pub model_id: String,
}

impl RigidBodyDynamicsSnapshot {
    pub fn validate(&self) -> bool {
        let joints = self.morphology.num_actuators();
        self.sampled_at_s.is_finite()
            && self.total_mass_kg.is_finite()
            && self.total_mass_kg > 0.0
            && self.gravity_mps2.is_finite()
            && self.gravity_mps2 > 0.0
            && self.mass_diagonal_kg_m2.len() == joints
            && self.bias_torque_nm.len() == joints
            && self.torque_limits_nm.len() == joints
            && self
                .mass_diagonal_kg_m2
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && self.bias_torque_nm.iter().all(|value| value.is_finite())
            && self
                .torque_limits_nm
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && self.right_foot.validate(joints)
            && self.left_foot.validate(joints)
            && !self.model_id.trim().is_empty()
    }

    pub fn normalized_to_physical_torque(&self, command: &HumanoidCommand) -> Option<Vec<f64>> {
        if command.num_actuators() != self.torque_limits_nm.len()
            || command.torques.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        Some(
            command
                .torques
                .iter()
                .zip(self.torque_limits_nm.iter())
                .map(|(normalized, limit)| *normalized as f64 * limit)
                .collect(),
        )
    }
}

pub trait RigidBodyDynamicsProvider {
    fn dynamics_snapshot(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
    ) -> Option<RigidBodyDynamicsSnapshot>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReducedOrderDynamicsConfig {
    pub gravity_mps2: f64,
    pub nominal_mass_kg: f64,
    pub posture_inertia_gain: f64,
    pub velocity_bias_gain: f64,
    pub gravity_bias_gain: f64,
}

impl Default for ReducedOrderDynamicsConfig {
    fn default() -> Self {
        Self {
            gravity_mps2: 9.806_65,
            nominal_mass_kg: 70.0,
            posture_inertia_gain: 0.12,
            velocity_bias_gain: 1.0,
            gravity_bias_gain: 0.16,
        }
    }
}

pub struct ReducedOrderRigidBodyModel {
    morphology: HumanoidMorphology,
    config: ReducedOrderDynamicsConfig,
    names: Vec<String>,
    inertias: Vec<f64>,
    damping: Vec<f64>,
    torque_limits: Vec<f64>,
}

impl ReducedOrderRigidBodyModel {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, ReducedOrderDynamicsConfig::default())
    }

    pub fn with_config(morphology: HumanoidMorphology, config: ReducedOrderDynamicsConfig) -> Self {
        Self {
            morphology,
            config,
            names: morphology.joint_names(),
            inertias: morphology.joint_inertias(),
            damping: morphology.joint_damping(),
            torque_limits: morphology.joint_torque_scales(),
        }
    }

    pub fn linearize(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
    ) -> Option<RigidBodyDynamicsSnapshot> {
        state.validate_for(self.morphology).ok()?;
        let joints = self.morphology.num_actuators();
        if self.names.len() != joints
            || self.inertias.len() != joints
            || self.damping.len() != joints
            || self.torque_limits.len() != joints
        {
            return None;
        }

        let posture_scale = 1.0
            + self.config.posture_inertia_gain.max(0.0)
                * (1.0 - state.uprightness()).clamp(0.0, 1.0);
        let mass_diagonal_kg_m2 = self
            .inertias
            .iter()
            .map(|inertia| (inertia * posture_scale).max(1.0e-5))
            .collect::<Vec<_>>();
        let bias_torque_nm = (0..joints)
            .map(|index| {
                let velocity = state.joint_velocities[index];
                let angle = state.joint_angles[index];
                let damping =
                    self.damping[index] * velocity * self.config.velocity_bias_gain.max(0.0);
                let gravity = gravity_role_gain(&self.names[index])
                    * self.torque_limits[index]
                    * angle.sin()
                    * self.config.gravity_bias_gain.max(0.0);
                damping + gravity
            })
            .collect::<Vec<_>>();

        let right_confidence = if contacts.right.in_contact {
            contacts.right.confidence.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let left_confidence = if contacts.left.in_contact {
            contacts.left.confidence.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let snapshot = RigidBodyDynamicsSnapshot {
            morphology: self.morphology,
            sampled_at_s: state.timestamp,
            total_mass_kg: morphology_mass_kg(self.morphology, self.config.nominal_mass_kg),
            gravity_mps2: self.config.gravity_mps2.max(1.0e-6),
            mass_diagonal_kg_m2,
            bias_torque_nm,
            torque_limits_nm: self.torque_limits.clone(),
            right_foot: approximate_foot_jacobian(FootSide::Right, &self.names, right_confidence),
            left_foot: approximate_foot_jacobian(FootSide::Left, &self.names, left_confidence),
            model_id: format!("symthaea.reduced-rigid-body.{:?}.v1", self.morphology),
        };
        snapshot.validate().then_some(snapshot)
    }
}

impl RigidBodyDynamicsProvider for ReducedOrderRigidBodyModel {
    fn dynamics_snapshot(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
    ) -> Option<RigidBodyDynamicsSnapshot> {
        self.linearize(state, contacts)
    }
}

impl<T> RigidBodyDynamicsProvider for T
where
    T: crate::simulator::HumanoidPhysicsSimulator + ?Sized,
{
    fn dynamics_snapshot(
        &self,
        _state: &HumanoidState,
        _contacts: &ContactFrame,
    ) -> Option<RigidBodyDynamicsSnapshot> {
        crate::simulator::HumanoidPhysicsSimulator::dynamics_snapshot(self)
    }
}

impl RigidBodyDynamicsProvider for crate::terrain::FlatTerrain {
    fn dynamics_snapshot(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
    ) -> Option<RigidBodyDynamicsSnapshot> {
        let morphology = morphology_from_state(state)?;
        ReducedOrderRigidBodyModel::new(morphology).linearize(state, contacts)
    }
}

impl RigidBodyDynamicsProvider for crate::terrain::HeightFieldTerrain {
    fn dynamics_snapshot(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
    ) -> Option<RigidBodyDynamicsSnapshot> {
        let morphology = morphology_from_state(state)?;
        ReducedOrderRigidBodyModel::new(morphology).linearize(state, contacts)
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
    .find(|morphology| morphology.num_actuators() == state.num_actuators())
}

fn morphology_mass_kg(morphology: HumanoidMorphology, nominal_mass_kg: f64) -> f64 {
    let extension = match morphology {
        HumanoidMorphology::Dmc21 => 0.0,
        HumanoidMorphology::WithNeckWrist => 1.8,
        HumanoidMorphology::Dexterous53 => 3.2,
        HumanoidMorphology::FullSpine => 6.0,
    };
    nominal_mass_kg.max(1.0) + extension
}

fn gravity_role_gain(name: &str) -> f64 {
    if name.contains("hip") || name.contains("knee") {
        0.95
    } else if name.contains("ankle") || name.contains("abdomen") {
        0.65
    } else if name.contains("shoulder") || name.contains("elbow") {
        0.22
    } else if name.contains("spine") || name.starts_with("j_5") {
        0.38
    } else {
        0.10
    }
}

fn approximate_foot_jacobian(
    side: FootSide,
    names: &[String],
    confidence: f64,
) -> FootContactJacobian {
    let prefix = match side {
        FootSide::Right => "right_",
        FootSide::Left => "left_",
    };
    let lateral_sign = match side {
        FootSide::Right => -1.0,
        FootSide::Left => 1.0,
    };
    let mut rows = [
        vec![0.0; names.len()],
        vec![0.0; names.len()],
        vec![0.0; names.len()],
    ];
    for (index, name) in names.iter().enumerate() {
        if !name.starts_with(prefix) {
            continue;
        }
        if name.ends_with("hip_y") {
            rows[0][index] = 0.34;
            rows[2][index] = -0.10;
        } else if name.ends_with("knee") {
            rows[0][index] = 0.22;
            rows[2][index] = 0.18;
        } else if name.ends_with("ankle_y") {
            rows[0][index] = 0.11;
            rows[2][index] = -0.12;
        } else if name.ends_with("hip_x") {
            rows[1][index] = 0.28 * lateral_sign;
            rows[2][index] = 0.04;
        } else if name.ends_with("ankle_x") {
            rows[1][index] = 0.14 * lateral_sign;
            rows[2][index] = -0.04;
        } else if name.contains("toe") {
            rows[0][index] = 0.06;
            rows[2][index] = -0.03;
        }
    }
    FootContactJacobian {
        side,
        rows,
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_model_matches_morphology_dimensions() {
        let state = HumanoidState::standing_for(HumanoidMorphology::WithNeckWrist);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let snapshot = ReducedOrderRigidBodyModel::new(HumanoidMorphology::WithNeckWrist)
            .linearize(&state, &contacts)
            .unwrap();
        assert!(snapshot.validate());
        assert_eq!(snapshot.mass_diagonal_kg_m2.len(), 27);
        assert_eq!(snapshot.right_foot.rows[0].len(), 27);
    }

    #[test]
    fn normalized_torque_conversion_uses_physical_limits() {
        let state = HumanoidState::standing();
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let snapshot = ReducedOrderRigidBodyModel::new(HumanoidMorphology::Dmc21)
            .linearize(&state, &contacts)
            .unwrap();
        let mut command = HumanoidCommand::zero();
        command.torques[5] = 0.5;
        let physical = snapshot.normalized_to_physical_torque(&command).unwrap();
        assert!((physical[5] - 150.0).abs() < 1.0e-9);
    }

    #[test]
    fn non_matching_state_is_rejected() {
        let state = HumanoidState::standing();
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        assert!(
            ReducedOrderRigidBodyModel::new(HumanoidMorphology::WithNeckWrist)
                .linearize(&state, &contacts)
                .is_none()
        );
    }
}
