// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Forward and inverse kinematics for 7-DOF manipulator arm.
//!
//! Uses Denavit-Hartenberg (DH) convention for forward kinematics and
//! Damped Least-Squares (DLS) for inverse kinematics.
//!
//! Science: Wampler (1986) — manipulator inverse kinematics with damping.

use serde::{Deserialize, Serialize};

/// Denavit-Hartenberg parameters for one joint.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DHParams {
    /// Link offset along z (meters).
    pub d: f64,
    /// Link length along x (meters).
    pub a: f64,
    /// Twist angle around x (radians).
    pub alpha: f64,
}

/// 4×4 homogeneous transformation matrix (row-major).
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub data: [[f64; 4]; 4],
}

impl Transform {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// DH transform for a single joint at angle theta.
    pub fn from_dh(dh: &DHParams, theta: f64) -> Self {
        let ct = theta.cos();
        let st = theta.sin();
        let ca = dh.alpha.cos();
        let sa = dh.alpha.sin();
        Self {
            data: [
                [ct, -st * ca, st * sa, dh.a * ct],
                [st, ct * ca, -ct * sa, dh.a * st],
                [0.0, sa, ca, dh.d],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Matrix multiplication: self × other.
    pub fn mul(&self, other: &Transform) -> Transform {
        let mut result = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Transform { data: result }
    }

    /// Extract position [x, y, z] from the transform.
    pub fn position(&self) -> [f64; 3] {
        [self.data[0][3], self.data[1][3], self.data[2][3]]
    }

    /// Extract rotation matrix (3×3 upper-left).
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        [
            [self.data[0][0], self.data[0][1], self.data[0][2]],
            [self.data[1][0], self.data[1][1], self.data[1][2]],
            [self.data[2][0], self.data[2][1], self.data[2][2]],
        ]
    }
}

/// 7-DOF manipulator kinematics.
///
/// Default DH parameters model a typical 7-DOF arm (e.g., Franka Emika Panda-like):
/// - Joints 1-3: shoulder (pitch, roll, pitch)
/// - Joint 4: elbow (pitch)
/// - Joints 5-7: wrist (pitch, roll, pitch)
pub struct ManipulatorKinematics {
    /// DH parameters for each of the 7 joints.
    pub dh_params: Vec<DHParams>,
    /// Joint limits [min, max] in radians.
    pub joint_limits: Vec<[f64; 2]>,
}

impl ManipulatorKinematics {
    /// Create with default 7-DOF arm parameters (Panda-like).
    pub fn default_7dof() -> Self {
        let dh_params = vec![
            DHParams {
                d: 0.333,
                a: 0.0,
                alpha: -std::f64::consts::FRAC_PI_2,
            }, // J1: shoulder yaw
            DHParams {
                d: 0.0,
                a: 0.0,
                alpha: std::f64::consts::FRAC_PI_2,
            }, // J2: shoulder pitch
            DHParams {
                d: 0.316,
                a: 0.0825,
                alpha: std::f64::consts::FRAC_PI_2,
            }, // J3: shoulder roll
            DHParams {
                d: 0.0,
                a: -0.0825,
                alpha: -std::f64::consts::FRAC_PI_2,
            }, // J4: elbow
            DHParams {
                d: 0.384,
                a: 0.0,
                alpha: std::f64::consts::FRAC_PI_2,
            }, // J5: wrist pitch
            DHParams {
                d: 0.0,
                a: 0.088,
                alpha: std::f64::consts::FRAC_PI_2,
            }, // J6: wrist roll
            DHParams {
                d: 0.107,
                a: 0.0,
                alpha: 0.0,
            }, // J7: wrist yaw
        ];
        let joint_limits = vec![
            [-2.89, 2.89],  // J1
            [-1.76, 1.76],  // J2
            [-2.89, 2.89],  // J3
            [-3.07, -0.07], // J4 (elbow, negative range)
            [-2.89, 2.89],  // J5
            [-0.02, 3.75],  // J6
            [-2.89, 2.89],  // J7
        ];
        Self {
            dh_params,
            joint_limits,
        }
    }

    /// Number of joints (DOF).
    pub fn num_joints(&self) -> usize {
        self.dh_params.len()
    }

    /// Forward kinematics: joint angles → end-effector transform.
    pub fn forward(&self, q: &[f64]) -> Transform {
        assert_eq!(q.len(), self.dh_params.len());
        let mut t = Transform::identity();
        for (i, dh) in self.dh_params.iter().enumerate() {
            t = t.mul(&Transform::from_dh(dh, q[i]));
        }
        t
    }

    /// End-effector position from joint angles.
    pub fn end_effector_position(&self, q: &[f64]) -> [f64; 3] {
        self.forward(q).position()
    }

    /// Numerical Jacobian (6×N): maps joint velocities to end-effector twist.
    ///
    /// Uses central finite differences with step size epsilon.
    pub fn jacobian(&self, q: &[f64], epsilon: f64) -> Vec<Vec<f64>> {
        let n = self.num_joints();
        let mut jac = vec![vec![0.0; n]; 6]; // 6 rows (3 position + 3 orientation)
        let base_pos = self.end_effector_position(q);
        let base_rot = self.forward(q).rotation();

        for j in 0..n {
            let mut q_plus = q.to_vec();
            let mut q_minus = q.to_vec();
            q_plus[j] += epsilon;
            q_minus[j] -= epsilon;

            let pos_plus = self.end_effector_position(&q_plus);
            let pos_minus = self.end_effector_position(&q_minus);

            // Position Jacobian (rows 0-2)
            for i in 0..3 {
                jac[i][j] = (pos_plus[i] - pos_minus[i]) / (2.0 * epsilon);
            }

            // Orientation Jacobian (rows 3-5) — approximate from rotation difference
            let rot_plus = self.forward(&q_plus).rotation();
            let rot_minus = self.forward(&q_minus).rotation();
            // Use skew-symmetric part of R_plus × R_minus^T for angular velocity
            jac[3][j] = (rot_plus[2][1] - rot_minus[2][1]) / (2.0 * epsilon);
            jac[4][j] = (rot_plus[0][2] - rot_minus[0][2]) / (2.0 * epsilon);
            jac[5][j] = (rot_plus[1][0] - rot_minus[1][0]) / (2.0 * epsilon);
        }

        jac
    }

    /// Damped Least-Squares IK (Wampler 1986).
    ///
    /// Solves: dq = J^T (J J^T + λ² I)^{-1} dx
    ///
    /// Returns joint angles that place the end-effector at `target_pos`,
    /// or None if convergence fails.
    pub fn ik_dls(
        &self,
        target_pos: &[f64; 3],
        q0: &[f64],
        lambda: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Option<Vec<f64>> {
        let n = self.num_joints();
        let mut q = q0.to_vec();

        for _iter in 0..max_iter {
            let current_pos = self.end_effector_position(&q);
            let dx = [
                target_pos[0] - current_pos[0],
                target_pos[1] - current_pos[1],
                target_pos[2] - current_pos[2],
            ];

            let error = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
            if error < tolerance {
                return Some(q);
            }

            // Compute position Jacobian (3×N)
            let full_jac = self.jacobian(&q, 1e-6);
            let jac: Vec<Vec<f64>> = full_jac[..3].to_vec(); // Position only

            // DLS: dq = J^T (J J^T + λ² I)^{-1} dx
            // For 3×N with N>3, this is underdetermined → use pseudoinverse
            // Simplified: dq_j = Σ_i J[i][j] * dx[i] / (Σ_k J[i][k]² + λ²)
            let mut dq = vec![0.0; n];
            for i in 0..3 {
                let row_norm_sq: f64 = jac[i].iter().map(|v| v * v).sum();
                let scale = 1.0 / (row_norm_sq + lambda * lambda);
                for j in 0..n {
                    dq[j] += jac[i][j] * dx[i] * scale;
                }
            }

            // Apply joint update with step size limiting
            let max_step = 0.1; // radians per iteration
            let step_norm: f64 = dq.iter().map(|v| v * v).sum::<f64>().sqrt();
            if step_norm > max_step {
                let scale = max_step / step_norm;
                for v in &mut dq {
                    *v *= scale;
                }
            }

            for j in 0..n {
                q[j] += dq[j];
                q[j] = q[j].clamp(self.joint_limits[j][0], self.joint_limits[j][1]);
            }
        }

        None // Failed to converge
    }

    /// Check if joint angles are within limits.
    pub fn within_limits(&self, q: &[f64]) -> bool {
        q.iter()
            .zip(&self.joint_limits)
            .all(|(&angle, limits)| angle >= limits[0] && angle <= limits[1])
    }
}

impl Default for ManipulatorKinematics {
    fn default() -> Self {
        Self::default_7dof()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_kinematics_identity() {
        let kin = ManipulatorKinematics::default_7dof();
        let q = vec![0.0; 7];
        let t = kin.forward(&q);
        let pos = t.position();
        // At zero configuration, end effector should be at a known position
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
        assert!(pos[2].is_finite());
        // Should be above the base (positive z)
        assert!(
            pos[2] > 0.0,
            "End effector should be above base: z={}",
            pos[2]
        );
    }

    #[test]
    fn test_forward_kinematics_changes_with_angles() {
        let kin = ManipulatorKinematics::default_7dof();
        let q_zero = vec![0.0; 7];
        let q_bent = vec![0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0];

        let pos_zero = kin.end_effector_position(&q_zero);
        let pos_bent = kin.end_effector_position(&q_bent);

        let dist = ((pos_zero[0] - pos_bent[0]).powi(2)
            + (pos_zero[1] - pos_bent[1]).powi(2)
            + (pos_zero[2] - pos_bent[2]).powi(2))
        .sqrt();

        assert!(
            dist > 0.01,
            "Different angles should produce different positions: dist={dist}"
        );
    }

    #[test]
    fn test_jacobian_shape() {
        let kin = ManipulatorKinematics::default_7dof();
        let q = vec![0.0; 7];
        let jac = kin.jacobian(&q, 1e-6);
        assert_eq!(jac.len(), 6, "Jacobian should have 6 rows");
        assert_eq!(jac[0].len(), 7, "Jacobian should have 7 columns");
    }

    #[test]
    fn test_jacobian_nonzero() {
        let kin = ManipulatorKinematics::default_7dof();
        let q = vec![0.1, 0.2, 0.1, -0.5, 0.1, 0.3, 0.1];
        let jac = kin.jacobian(&q, 1e-6);
        // At least some entries should be nonzero
        let max_val: f64 = jac
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(
            max_val > 0.01,
            "Jacobian should have nonzero entries: max={max_val}"
        );
    }

    #[test]
    fn test_ik_converges_to_reachable_target() {
        let kin = ManipulatorKinematics::default_7dof();
        // Get a known reachable point from forward kinematics
        let q_target = vec![0.1, 0.3, -0.2, -1.5, 0.1, 0.5, 0.0];
        let target_pos = kin.end_effector_position(&q_target);

        // Start from a different configuration
        let q0 = vec![0.0; 7];
        let result = kin.ik_dls(&target_pos, &q0, 0.1, 200, 0.01);

        assert!(
            result.is_some(),
            "IK should converge for a reachable target"
        );
        let q_solution = result.unwrap();
        let solution_pos = kin.end_effector_position(&q_solution);
        let error = ((solution_pos[0] - target_pos[0]).powi(2)
            + (solution_pos[1] - target_pos[1]).powi(2)
            + (solution_pos[2] - target_pos[2]).powi(2))
        .sqrt();
        assert!(
            error < 0.01,
            "IK solution should be within tolerance: error={error}"
        );
    }

    #[test]
    fn test_ik_respects_joint_limits() {
        let kin = ManipulatorKinematics::default_7dof();
        let q_target = vec![0.5, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0];
        let target_pos = kin.end_effector_position(&q_target);
        let q0 = vec![0.0; 7];
        let result = kin.ik_dls(&target_pos, &q0, 0.1, 200, 0.01);

        if let Some(q) = result {
            assert!(
                kin.within_limits(&q),
                "IK solution should respect joint limits"
            );
        }
    }

    #[test]
    fn test_dh_transform_identity() {
        let dh = DHParams {
            d: 0.0,
            a: 0.0,
            alpha: 0.0,
        };
        let t = Transform::from_dh(&dh, 0.0);
        let pos = t.position();
        assert!((pos[0]).abs() < 1e-10);
        assert!((pos[1]).abs() < 1e-10);
        assert!((pos[2]).abs() < 1e-10);
    }

    #[test]
    fn test_joint_limits() {
        let kin = ManipulatorKinematics::default_7dof();
        // Valid: within all joint ranges (J4 elbow is [-3.07, -0.07])
        let q_valid = vec![0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        assert!(kin.within_limits(&q_valid));

        let q_invalid = vec![10.0; 7]; // Way out of range
        assert!(!kin.within_limits(&q_invalid));
    }
}
