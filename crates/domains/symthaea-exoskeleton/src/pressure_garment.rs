// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pressure-garment joint resistance model for EVA biomechanics.
//!
//! The coefficients are explicit research inputs. This module provides a
//! physically interpretable load for powered assistance to compensate; it is
//! not a pressure-garment certification model.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PressureJointModel {
    /// Effective torque coefficient per pressure differential, N m / Pa.
    pub pressure_torque_coeff_nm_pa: f64,
    /// Elastic joint stiffness around neutral position, N m / rad.
    pub stiffness_nm_rad: f64,
    /// Viscous resistance, N m s / rad.
    pub damping_nm_s_rad: f64,
    /// Coulomb-like seal/bearing breakaway torque, N m.
    pub breakaway_torque_nm: f64,
    pub neutral_angle_rad: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl PressureJointModel {
    pub fn simulation_reference() -> Self {
        Self {
            pressure_torque_coeff_nm_pa: 2.0e-5,
            stiffness_nm_rad: 4.0,
            damping_nm_s_rad: 0.8,
            breakaway_torque_nm: 0.7,
            neutral_angle_rad: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.pressure_torque_coeff_nm_pa.is_finite()
            && self.pressure_torque_coeff_nm_pa >= 0.0
            && self.stiffness_nm_rad.is_finite()
            && self.stiffness_nm_rad >= 0.0
            && self.damping_nm_s_rad.is_finite()
            && self.damping_nm_s_rad >= 0.0
            && self.breakaway_torque_nm.is_finite()
            && self.breakaway_torque_nm >= 0.0
            && self.neutral_angle_rad.is_finite()
    }

    /// Signed resistive torque opposing the requested joint motion.
    pub fn resistive_torque_nm(
        &self,
        suit_pressure_pa: f64,
        ambient_pressure_pa: f64,
        angle_rad: f64,
        velocity_rad_s: f64,
    ) -> Option<f64> {
        if !self.is_valid()
            || !suit_pressure_pa.is_finite()
            || suit_pressure_pa < 0.0
            || !ambient_pressure_pa.is_finite()
            || ambient_pressure_pa < 0.0
            || !angle_rad.is_finite()
            || !velocity_rad_s.is_finite()
        {
            return None;
        }

        let delta_p = (suit_pressure_pa - ambient_pressure_pa).max(0.0);
        let pressure = self.pressure_torque_coeff_nm_pa * delta_p;
        let elastic = self.stiffness_nm_rad * (angle_rad - self.neutral_angle_rad);
        let viscous = self.damping_nm_s_rad * velocity_rad_s;
        let direction = if velocity_rad_s.abs() > 1e-9 {
            velocity_rad_s.signum()
        } else if (angle_rad - self.neutral_angle_rad).abs() > 1e-9 {
            (angle_rad - self.neutral_angle_rad).signum()
        } else {
            0.0
        };
        let breakaway = self.breakaway_torque_nm * direction;

        // Resist motion/deformation: pressure + seal terms oppose motion,
        // while elastic torque restores toward neutral.
        Some(-(pressure * direction + breakaway + elastic + viscous))
    }

    pub fn mechanical_power_cost_w(
        &self,
        suit_pressure_pa: f64,
        ambient_pressure_pa: f64,
        angle_rad: f64,
        velocity_rad_s: f64,
    ) -> Option<f64> {
        let tau = self
            .resistive_torque_nm(suit_pressure_pa, ambient_pressure_pa, angle_rad, velocity_rad_s)?;
        Some((tau * velocity_rad_s).abs())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureGarmentModel {
    pub joints: Vec<PressureJointModel>,
}

impl PressureGarmentModel {
    pub fn uniform_reference(joint_count: usize) -> Self {
        Self {
            joints: vec![PressureJointModel::simulation_reference(); joint_count],
        }
    }

    pub fn total_resistance_power_w(
        &self,
        suit_pressure_pa: f64,
        ambient_pressure_pa: f64,
        joint_angles_rad: &[f64],
        joint_velocities_rad_s: &[f64],
    ) -> Option<f64> {
        if self.joints.len() != joint_angles_rad.len()
            || self.joints.len() != joint_velocities_rad_s.len()
        {
            return None;
        }

        self.joints
            .iter()
            .zip(joint_angles_rad)
            .zip(joint_velocities_rad_s)
            .try_fold(0.0, |acc, ((joint, angle), velocity)| {
                joint
                    .mechanical_power_cost_w(
                        suit_pressure_pa,
                        ambient_pressure_pa,
                        *angle,
                        *velocity,
                    )
                    .map(|power| acc + power)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vacuum_pressure_differential_increases_joint_resistance() {
        let joint = PressureJointModel::simulation_reference();
        let vacuum = joint
            .mechanical_power_cost_w(30_000.0, 0.0, 0.4, 1.0)
            .unwrap();
        let near_equalized = joint
            .mechanical_power_cost_w(30_000.0, 29_000.0, 0.4, 1.0)
            .unwrap();
        assert!(vacuum > near_equalized);
    }

    #[test]
    fn garment_power_sums_across_joints() {
        let garment = PressureGarmentModel::uniform_reference(2);
        let one = garment.joints[0]
            .mechanical_power_cost_w(30_000.0, 0.0, 0.2, 0.5)
            .unwrap();
        let total = garment
            .total_resistance_power_w(30_000.0, 0.0, &[0.2, 0.2], &[0.5, 0.5])
            .unwrap();
        assert!((total - 2.0 * one).abs() < 1e-9);
    }

    #[test]
    fn malformed_dimensions_are_rejected() {
        let garment = PressureGarmentModel::uniform_reference(2);
        assert!(garment
            .total_resistance_power_w(30_000.0, 0.0, &[0.0], &[0.0])
            .is_none());
    }
}
