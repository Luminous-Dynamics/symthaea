// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Environment coupling for the 20-DOF Symtropy full-frame exoskeleton.
//!
//! SX-016 closes two previously separate simulation layers:
//! - validated Earth/Moon/Mars/custom gravity now drives the actual
//!   `PhysicsWorld` used by the full-frame body;
//! - pressure-garment resistance is computed from the full-frame's actual
//!   joint angles/velocities and injected back into those joints.
//!
//! This remains simulation evidence. It does not change the certified EVA
//! assist envelope, PLSS authority, or local actuator safety architecture.

#![cfg(feature = "symtropy")]

use serde::{Deserialize, Serialize};
use symtropy_physics::body::BodyHandle;

use crate::full_frame::{FullFrameSimulator, NUM_FULL_FRAME_JOINTS};
use crate::pressure_garment::PressureGarmentModel;
use crate::reduced_gravity::GravityEnvironment;
use crate::space_exosuit::ExosuitEvidenceLevel;

const SPINE_PLANES: [(usize, usize); 2] = [(0, 2), (0, 2)];
const ARM_PLANES: [(usize, usize); 6] = [
    (0, 2),
    (1, 2),
    (0, 1),
    (0, 2),
    (0, 2),
    (1, 2),
];
const LEG_PLANES: [(usize, usize); 3] = [(0, 2), (1, 2), (1, 2)];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FullFrameEnvironmentError {
    InvalidGravity,
    InvalidPressure,
    InvalidTimeStep,
    InvalidGarmentDimensions,
    InvalidJointState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullFrameEnvironmentConfig {
    pub gravity: GravityEnvironment,
    pub suit_pressure_pa: f64,
    pub ambient_pressure_pa: f64,
    pub pressure_garment: PressureGarmentModel,
    pub evidence: ExosuitEvidenceLevel,
}

impl FullFrameEnvironmentConfig {
    pub fn lunar_reference() -> Self {
        Self::airless_reference(GravityEnvironment::lunar_reference())
    }

    pub fn mars_reference() -> Self {
        Self::airless_reference(GravityEnvironment::mars_reference())
    }

    pub fn earth_reference() -> Self {
        Self {
            gravity: GravityEnvironment::earth_reference(),
            suit_pressure_pa: 30_000.0,
            ambient_pressure_pa: 101_325.0,
            pressure_garment: PressureGarmentModel::uniform_reference(NUM_FULL_FRAME_JOINTS),
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn airless_reference(gravity: GravityEnvironment) -> Self {
        Self {
            gravity,
            suit_pressure_pa: 30_000.0,
            ambient_pressure_pa: 0.0,
            pressure_garment: PressureGarmentModel::uniform_reference(NUM_FULL_FRAME_JOINTS),
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.gravity.is_valid()
            && self.suit_pressure_pa.is_finite()
            && self.suit_pressure_pa >= 0.0
            && self.ambient_pressure_pa.is_finite()
            && self.ambient_pressure_pa >= 0.0
            && self.pressure_garment.joints.len() == NUM_FULL_FRAME_JOINTS
            && self.pressure_garment.joints.iter().all(|joint| joint.is_valid())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullFrameEnvironmentStep {
    pub gravity_m_s2: f64,
    pub pressure_resistance_torque_nm: [f64; NUM_FULL_FRAME_JOINTS],
    pub pressure_resistance_power_w: f64,
    pub evidence: ExosuitEvidenceLevel,
}

#[derive(Debug, Clone)]
pub struct FullFrameEnvironmentCoupler {
    config: FullFrameEnvironmentConfig,
}

impl FullFrameEnvironmentCoupler {
    pub fn new(config: FullFrameEnvironmentConfig) -> Result<Self, FullFrameEnvironmentError> {
        if !config.gravity.is_valid() {
            return Err(FullFrameEnvironmentError::InvalidGravity);
        }
        if !config.suit_pressure_pa.is_finite()
            || config.suit_pressure_pa < 0.0
            || !config.ambient_pressure_pa.is_finite()
            || config.ambient_pressure_pa < 0.0
        {
            return Err(FullFrameEnvironmentError::InvalidPressure);
        }
        if config.pressure_garment.joints.len() != NUM_FULL_FRAME_JOINTS {
            return Err(FullFrameEnvironmentError::InvalidGarmentDimensions);
        }
        if !config.pressure_garment.joints.iter().all(|joint| joint.is_valid()) {
            return Err(FullFrameEnvironmentError::InvalidGarmentDimensions);
        }
        Ok(Self { config })
    }

    pub fn lunar_reference() -> Self {
        Self::new(FullFrameEnvironmentConfig::lunar_reference())
            .expect("lunar reference environment must be valid")
    }

    pub fn config(&self) -> &FullFrameEnvironmentConfig {
        &self.config
    }

    /// Apply the declared gravity directly to the existing Symtropy world.
    pub fn apply_gravity(&self, sim: &mut FullFrameSimulator) -> Result<(), FullFrameEnvironmentError> {
        if !self.config.gravity.is_valid() {
            return Err(FullFrameEnvironmentError::InvalidGravity);
        }
        sim.world.gravity = nalgebra::SVector::from([
            0.0,
            0.0,
            -self.config.gravity.gravity_m_s2,
        ]);
        Ok(())
    }

    /// Step the full-frame world with actual environment gravity and
    /// pose-dependent pressure-garment resistance.
    ///
    /// Resistance is injected using the same angular-velocity impulse pattern
    /// already used by the existing Symtropy exoskeleton backend. This is a
    /// simulation approximation, not a certified actuator/garment model.
    pub fn step(
        &self,
        sim: &mut FullFrameSimulator,
        dt_s: f64,
    ) -> Result<FullFrameEnvironmentStep, FullFrameEnvironmentError> {
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(FullFrameEnvironmentError::InvalidTimeStep);
        }
        if !self.config.is_valid() {
            return Err(FullFrameEnvironmentError::InvalidGarmentDimensions);
        }

        self.apply_gravity(sim)?;
        sim.human.update(dt_s);

        let joints = collect_joint_states(sim)?;
        if joints.len() != NUM_FULL_FRAME_JOINTS {
            return Err(FullFrameEnvironmentError::InvalidJointState);
        }

        let mut torques = [0.0; NUM_FULL_FRAME_JOINTS];
        let mut total_power_w = 0.0;

        for (index, entry) in joints.iter().enumerate() {
            let model = &self.config.pressure_garment.joints[index];
            let torque = model
                .resistive_torque_nm(
                    self.config.suit_pressure_pa,
                    self.config.ambient_pressure_pa,
                    entry.angle_rad,
                    entry.velocity_rad_s,
                )
                .ok_or(FullFrameEnvironmentError::InvalidJointState)?;
            let power = model
                .mechanical_power_cost_w(
                    self.config.suit_pressure_pa,
                    self.config.ambient_pressure_pa,
                    entry.angle_rad,
                    entry.velocity_rad_s,
                )
                .ok_or(FullFrameEnvironmentError::InvalidJointState)?;

            torques[index] = torque;
            total_power_w += power;

            if let Some(body) = sim.world.body_mut(entry.handle) {
                let current = body.angular_velocity.get(entry.plane.0, entry.plane.1);
                body.angular_velocity.set(
                    entry.plane.0,
                    entry.plane.1,
                    current + torque * dt_s,
                );
            } else {
                return Err(FullFrameEnvironmentError::InvalidJointState);
            }
        }

        sim.world.step_with_callback(dt_s, &mut sim.callback);

        Ok(FullFrameEnvironmentStep {
            gravity_m_s2: self.config.gravity.gravity_m_s2,
            pressure_resistance_torque_nm: torques,
            pressure_resistance_power_w: total_power_w,
            evidence: self.config.evidence,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct JointEntry {
    handle: BodyHandle,
    angle_rad: f64,
    velocity_rad_s: f64,
    plane: (usize, usize),
}

fn collect_joint_states(
    sim: &FullFrameSimulator,
) -> Result<Vec<JointEntry>, FullFrameEnvironmentError> {
    let mut out = Vec::with_capacity(NUM_FULL_FRAME_JOINTS);
    append_chain(&mut out, &sim.spine_chain, &SPINE_PLANES, &sim.world)?;
    append_chain(&mut out, &sim.left_arm, &ARM_PLANES, &sim.world)?;
    append_chain(&mut out, &sim.right_arm, &ARM_PLANES, &sim.world)?;
    append_chain(&mut out, &sim.left_leg, &LEG_PLANES, &sim.world)?;
    append_chain(&mut out, &sim.right_leg, &LEG_PLANES, &sim.world)?;
    Ok(out)
}

fn append_chain(
    out: &mut Vec<JointEntry>,
    chain: &symtropy_physics::articulation::ArticulatedChain,
    planes: &[(usize, usize)],
    world: &symtropy_physics::world::PhysicsWorld<3>,
) -> Result<(), FullFrameEnvironmentError> {
    let states = chain.read_joint_states(world);
    if states.len() != chain.links.len() || states.len() != planes.len() {
        return Err(FullFrameEnvironmentError::InvalidJointState);
    }

    for (((&handle, &(angle_rad, velocity_rad_s)), &plane), _) in chain
        .links
        .iter()
        .zip(states.iter())
        .zip(planes.iter())
        .zip(0..)
    {
        if !angle_rad.is_finite() || !velocity_rad_s.is_finite() {
            return Err(FullFrameEnvironmentError::InvalidJointState);
        }
        out.push(JointEntry {
            handle,
            angle_rad,
            velocity_rad_s,
            plane,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduced_gravity::{MARS_GRAVITY_M_S2, MOON_GRAVITY_M_S2};

    #[test]
    fn lunar_environment_drives_actual_full_frame_world_gravity() {
        let mut sim = FullFrameSimulator::new();
        let coupler = FullFrameEnvironmentCoupler::lunar_reference();
        coupler.apply_gravity(&mut sim).unwrap();
        assert!((sim.world.gravity[2] + MOON_GRAVITY_M_S2).abs() < 1e-12);
    }

    #[test]
    fn mars_environment_is_distinct_from_lunar_environment() {
        let mut sim = FullFrameSimulator::new();
        let coupler = FullFrameEnvironmentCoupler::new(FullFrameEnvironmentConfig::mars_reference())
            .unwrap();
        coupler.apply_gravity(&mut sim).unwrap();
        assert!((sim.world.gravity[2] + MARS_GRAVITY_M_S2).abs() < 1e-12);
        assert!(MARS_GRAVITY_M_S2 > MOON_GRAVITY_M_S2);
    }

    #[test]
    fn pressure_resistance_is_computed_for_all_twenty_joints() {
        let mut sim = FullFrameSimulator::new();
        let coupler = FullFrameEnvironmentCoupler::lunar_reference();
        let report = coupler.step(&mut sim, 0.001).unwrap();
        assert_eq!(report.pressure_resistance_torque_nm.len(), NUM_FULL_FRAME_JOINTS);
        assert!(report.pressure_resistance_power_w.is_finite());
        assert!(report.pressure_resistance_power_w >= 0.0);
        assert!(report
            .pressure_resistance_torque_nm
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn malformed_gravity_fails_closed() {
        let mut config = FullFrameEnvironmentConfig::lunar_reference();
        config.gravity = GravityEnvironment::custom(f64::NAN, ExosuitEvidenceLevel::Simulation);
        assert_eq!(
            FullFrameEnvironmentCoupler::new(config).unwrap_err(),
            FullFrameEnvironmentError::InvalidGravity
        );
    }

    #[test]
    fn garment_dimension_mismatch_is_rejected() {
        let mut config = FullFrameEnvironmentConfig::lunar_reference();
        config.pressure_garment = PressureGarmentModel::uniform_reference(6);
        assert_eq!(
            FullFrameEnvironmentCoupler::new(config).unwrap_err(),
            FullFrameEnvironmentError::InvalidGarmentDimensions
        );
    }
}
