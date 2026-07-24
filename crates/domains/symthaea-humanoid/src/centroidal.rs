// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Centroidal momentum regulation above the contact-constrained allocator.
//!
//! The regulator produces a bounded normalized-torque correction. It does not
//! bypass the whole-body QP, safety projector, or backend actuation contract.

use serde::{Deserialize, Serialize};

use crate::contact::{BipedSupport, ContactFrame};
use crate::full_dynamics::FullRigidBodyDynamicsSnapshot;
use crate::morphology::HumanoidMorphology;
use crate::types::{HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CentroidalMomentumConfig {
    pub linear_damping: f64,
    pub angular_damping: f64,
    pub maximum_correction: f64,
    pub flight_authority: f64,
    pub minimum_contact_confidence: f64,
}

impl Default for CentroidalMomentumConfig {
    fn default() -> Self {
        Self {
            linear_damping: 0.24,
            angular_damping: 0.18,
            maximum_correction: 0.20,
            flight_authority: 0.25,
            minimum_contact_confidence: 0.30,
        }
    }
}

impl CentroidalMomentumConfig {
    pub fn validate(&self) -> bool {
        [
            self.linear_damping,
            self.angular_damping,
            self.maximum_correction,
            self.flight_authority,
            self.minimum_contact_confidence,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
            && self.maximum_correction <= 1.0
            && self.flight_authority <= 1.0
            && self.minimum_contact_confidence <= 1.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentroidalMomentumReport {
    pub valid_model: bool,
    pub support: BipedSupport,
    pub current_angular_momentum: [f64; 3],
    pub current_linear_momentum: [f64; 3],
    pub target_angular_momentum_rate: [f64; 3],
    pub target_linear_momentum_rate: [f64; 3],
    pub correction_norm: f64,
    pub authority: f64,
}

pub struct CentroidalMomentumController {
    morphology: HumanoidMorphology,
    config: CentroidalMomentumConfig,
}

impl CentroidalMomentumController {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, CentroidalMomentumConfig::default())
    }

    pub const fn with_config(
        morphology: HumanoidMorphology,
        config: CentroidalMomentumConfig,
    ) -> Self {
        Self { morphology, config }
    }

    pub fn correction(
        &self,
        state: &HumanoidState,
        contacts: &ContactFrame,
        dynamics: &FullRigidBodyDynamicsSnapshot,
    ) -> (HumanoidCommand, CentroidalMomentumReport) {
        let joints = self.morphology.num_actuators();
        let invalid = state.validate_for(self.morphology).is_err()
            || dynamics.morphology != self.morphology
            || !dynamics.validate()
            || !self.config.validate();
        if invalid {
            return (
                HumanoidCommand::zero_for(joints),
                CentroidalMomentumReport {
                    valid_model: false,
                    support: contacts.support(),
                    current_angular_momentum: [0.0; 3],
                    current_linear_momentum: [0.0; 3],
                    target_angular_momentum_rate: [0.0; 3],
                    target_linear_momentum_rate: [0.0; 3],
                    correction_norm: 0.0,
                    authority: 0.0,
                },
            );
        }

        let mut momentum = [0.0; 6];
        for row in 0..6 {
            let matrix_row = dynamics.centroidal_row(row).unwrap();
            momentum[row] = matrix_row
                .iter()
                .zip(state.joint_velocities.iter())
                .map(|(coefficient, velocity)| coefficient * velocity)
                .sum();
        }
        // Root motion is included explicitly because reduced actuator-only
        // models do not contain floating-base generalized velocities.
        for axis in 0..3 {
            momentum[axis] += dynamics.total_mass_kg * state.root_angular_velocity[axis];
            momentum[3 + axis] += dynamics.total_mass_kg * state.com_velocity[axis];
        }

        let angular_rate = [
            -self.config.angular_damping * momentum[0],
            -self.config.angular_damping * momentum[1],
            -self.config.angular_damping * momentum[2],
        ];
        let linear_rate = [
            -self.config.linear_damping * momentum[3],
            -self.config.linear_damping * momentum[4],
            -self.config.linear_damping * momentum[5],
        ];
        let contact_confidence = contacts.minimum_confidence();
        let support_authority = match contacts.support() {
            BipedSupport::Flight => self.config.flight_authority,
            BipedSupport::Right | BipedSupport::Left => 0.75,
            BipedSupport::Double => 1.0,
        };
        let confidence_gate = if contact_confidence < self.config.minimum_contact_confidence {
            0.35 * contact_confidence / self.config.minimum_contact_confidence.max(1.0e-9)
        } else {
            contact_confidence
        };
        let authority = (support_authority * confidence_gate).clamp(0.0, 1.0);

        let desired_rate = [
            angular_rate[0],
            angular_rate[1],
            angular_rate[2],
            linear_rate[0],
            linear_rate[1],
            linear_rate[2],
        ];
        let mut correction = vec![0.0f32; joints];
        for joint in 0..joints {
            let projected: f64 = (0..6)
                .map(|row| {
                    dynamics.centroidal_momentum_matrix[row * joints + joint] * desired_rate[row]
                })
                .sum();
            let scale = dynamics.torque_limits_nm[joint].max(1.0e-6);
            correction[joint] =
                (authority * projected / scale * self.config.maximum_correction.max(0.0)).clamp(
                    -self.config.maximum_correction,
                    self.config.maximum_correction,
                ) as f32;
        }
        let correction_norm = correction
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        (
            HumanoidCommand {
                torques: correction,
            },
            CentroidalMomentumReport {
                valid_model: true,
                support: contacts.support(),
                current_angular_momentum: [momentum[0], momentum[1], momentum[2]],
                current_linear_momentum: [momentum[3], momentum[4], momentum[5]],
                target_angular_momentum_rate: angular_rate,
                target_linear_momentum_rate: linear_rate,
                correction_norm,
                authority,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::ContactFrame;
    use crate::dynamics::ReducedOrderRigidBodyModel;
    use crate::full_dynamics::FullRigidBodyDynamicsSnapshot;

    #[test]
    fn correction_is_bounded() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut state = HumanoidState::default_for(morphology);
        state.com_velocity = [4.0, -3.0, 0.5];
        state.root_angular_velocity = [2.0, 1.0, -1.0];
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let reduced = ReducedOrderRigidBodyModel::new(morphology)
            .linearize(&state, &contacts)
            .unwrap();
        let full = FullRigidBodyDynamicsSnapshot::from_reduced(&reduced).unwrap();
        let controller = CentroidalMomentumController::new(morphology);
        let (command, report) = controller.correction(&state, &contacts, &full);
        assert!(report.valid_model);
        assert!(command.torques.iter().all(|value| value.abs() <= 0.200_001));
    }

    #[test]
    fn invalid_model_fails_to_zero_correction() {
        let morphology = HumanoidMorphology::Dmc21;
        let state = HumanoidState::default_for(morphology);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let reduced = ReducedOrderRigidBodyModel::new(morphology)
            .linearize(&state, &contacts)
            .unwrap();
        let mut full = FullRigidBodyDynamicsSnapshot::from_reduced(&reduced).unwrap();
        full.mass_matrix.clear();
        let (command, report) =
            CentroidalMomentumController::new(morphology).correction(&state, &contacts, &full);
        assert!(!report.valid_model);
        assert!(command.torques.iter().all(|value| *value == 0.0));
    }
}
