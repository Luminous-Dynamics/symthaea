// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phi-proportional admittance controller.
//!
//! Prevents IK/DLS chatter by blending IK command torques with compliant
//! deflection torques as Phi drops. When Phi is high, the arm firmly follows
//! IK targets. When Phi drops, the arm yields to external forces.
//!
//! The blend uses Jacobian transpose mapping: τ_compliant = J^T × F_external
//! which maps Cartesian forces to joint-space torques.

use symthaea_manipulator::kinematics::ManipulatorKinematics;
use symthaea_manipulator::types::NUM_JOINTS;

/// Phi-proportional admittance controller.
pub struct AdmittanceController {
    /// Stiffness factor [0.1, 1.0]. Proportional to Phi.
    /// 1.0 = full IK authority, 0.1 = near-fully compliant.
    pub stiffness: f64,
}

impl Default for AdmittanceController {
    fn default() -> Self {
        Self { stiffness: 1.0 }
    }
}

impl AdmittanceController {
    /// Update stiffness from the current Phi value.
    pub fn update_from_phi(&mut self, phi: f64) {
        self.stiffness = phi.clamp(0.1, 1.0);
    }

    /// Blend IK torques with compliant deflection torques.
    ///
    /// - `ik_torques`: joint torques from IK-based PD controller [-1, 1]
    /// - `external_forces`: Cartesian forces on EE [Fx, Fy, Fz] in Newtons
    /// - `joint_angles`: current joint angles (for Jacobian computation)
    /// - `kinematics`: arm kinematics (for Jacobian)
    ///
    /// Returns blended joint torques [-1, 1].
    pub fn blend_torques(
        &self,
        ik_torques: &[f32; NUM_JOINTS],
        external_forces: &[f64; 3],
        joint_angles: &[f64],
        kinematics: &ManipulatorKinematics,
    ) -> [f32; NUM_JOINTS] {
        let compliance = 1.0 - self.stiffness;

        // If fully stiff (Green, Phi > 0.6), just return IK torques
        if compliance < 0.01 {
            return *ik_torques;
        }

        // If no external force, just scale IK torques by stiffness
        let force_mag =
            (external_forces[0].powi(2) + external_forces[1].powi(2) + external_forces[2].powi(2))
                .sqrt();
        if force_mag < 0.01 {
            let mut result = [0.0f32; NUM_JOINTS];
            for i in 0..NUM_JOINTS {
                result[i] = (self.stiffness as f32 * ik_torques[i]).clamp(-1.0, 1.0);
            }
            return result;
        }

        // Compute Jacobian transpose mapping: τ_compliant = J^T × F_ext
        let compliant_torques = jacobian_transpose_map(kinematics, joint_angles, external_forces);

        // Blend: stiffness × IK + compliance × J^T×F
        let mut result = [0.0f32; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            let ik_component = self.stiffness as f32 * ik_torques[i];
            let compliant_component = compliance as f32 * compliant_torques[i] as f32;
            result[i] = (ik_component + compliant_component).clamp(-1.0, 1.0);
        }
        result
    }
}

/// Map Cartesian forces to joint-space torques via Jacobian transpose.
///
/// τ = J^T × F where J is the 3×7 position Jacobian.
/// The result is normalized to [-1, 1] range using max torque scaling.
fn jacobian_transpose_map(
    kinematics: &ManipulatorKinematics,
    joint_angles: &[f64],
    force: &[f64; 3],
) -> [f64; NUM_JOINTS] {
    // Compute 6×7 Jacobian (only first 3 rows are position)
    let jac = kinematics.jacobian(joint_angles, 1e-6);

    let mut torques = [0.0f64; NUM_JOINTS];

    // τ_j = Σ_i J[i][j] × F[i] for i in 0..3 (position rows)
    for j in 0..NUM_JOINTS {
        for i in 0..3 {
            if i < jac.len() && j < jac[i].len() {
                torques[j] += jac[i][j] * force[i];
            }
        }
    }

    // Normalize to [-1, 1] range (divide by max torque per joint)
    // Panda: 87 Nm for J1-4, 12 Nm for J5-7
    let max_torques = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];
    for j in 0..NUM_JOINTS {
        torques[j] = (torques[j] / max_torques[j]).clamp(-1.0, 1.0);
    }

    torques
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_stiffness_passes_ik_through() {
        let ctrl = AdmittanceController { stiffness: 1.0 };
        let ik = [0.5f32, -0.3, 0.2, 0.0, 0.1, -0.1, 0.05];
        let kin = ManipulatorKinematics::default_7dof();
        let angles = [0.0; 7];
        let force = [10.0, 0.0, 0.0];

        let result = ctrl.blend_torques(&ik, &force, &angles, &kin);
        // Should be very close to original IK torques
        for i in 0..7 {
            assert!((result[i] - ik[i]).abs() < 0.01, "Joint {i}: {}", result[i]);
        }
    }

    #[test]
    fn low_stiffness_yields_to_force() {
        let ctrl = AdmittanceController { stiffness: 0.1 };
        let ik = [0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // IK wants to go forward
        let kin = ManipulatorKinematics::default_7dof();
        let angles = [0.0; 7];
        let force = [-30.0, 0.0, 0.0]; // Force pushing backward

        let result = ctrl.blend_torques(&ik, &force, &angles, &kin);
        // IK component is 0.1 × 0.5 = 0.05
        // Compliant component is 0.9 × J^T × [-30, 0, 0]
        // At least one joint should differ significantly from pure IK
        let ik_only = 0.1 * 0.5;
        let has_compliant_influence = result.iter().any(|&t| (t - ik_only).abs() > 0.02);
        assert!(
            has_compliant_influence,
            "Compliant force should influence output: {:?}",
            result
        );
    }

    #[test]
    fn update_from_phi() {
        let mut ctrl = AdmittanceController::default();
        assert_eq!(ctrl.stiffness, 1.0);

        ctrl.update_from_phi(0.5);
        assert!((ctrl.stiffness - 0.5).abs() < 0.01);

        ctrl.update_from_phi(0.05);
        assert!((ctrl.stiffness - 0.1).abs() < 0.01); // Clamped to 0.1
    }

    #[test]
    fn no_force_scales_ik_by_stiffness() {
        let ctrl = AdmittanceController { stiffness: 0.5 };
        let ik = [0.8f32, 0.6, 0.4, 0.2, 0.1, -0.1, -0.2];
        let kin = ManipulatorKinematics::default_7dof();
        let angles = [0.0; 7];
        let force = [0.0; 3]; // No external force

        let result = ctrl.blend_torques(&ik, &force, &angles, &kin);
        for i in 0..7 {
            let expected = 0.5 * ik[i];
            assert!(
                (result[i] - expected).abs() < 0.01,
                "Joint {i}: got {} expected {}",
                result[i],
                expected
            );
        }
    }

    #[test]
    fn jacobian_transpose_nonzero() {
        let kin = ManipulatorKinematics::default_7dof();
        let angles = [0.0; 7];
        let force = [10.0, 0.0, 0.0];

        let torques = jacobian_transpose_map(&kin, &angles, &force);
        let mag = torques.iter().map(|t| t.abs()).sum::<f64>();
        assert!(
            mag > 0.0,
            "J^T × F should produce nonzero torques: {:?}",
            torques
        );
    }
}
