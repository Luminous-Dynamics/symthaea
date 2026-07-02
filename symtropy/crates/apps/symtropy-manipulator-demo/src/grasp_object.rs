// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Simulated grasp object: a rigid cube that attaches to the gripper on close.
//!
//! When the gripper closes near the object, the object "attaches" and adds
//! mass to the end-effector dynamics. This creates genuine prediction error
//! during transit — the arm's model of itself changes when holding the object.
//! The HDC encoder picks this up as a force channel perturbation.

/// State of the simulated grasp object.
#[derive(Debug, Clone)]
pub struct GraspObject {
    /// Object position [x, y, z] in meters (world frame).
    pub position: [f64; 3],
    /// Object mass in kg.
    pub mass: f64,
    /// Whether the object is currently grasped.
    pub grasped: bool,
    /// Size (half-extent of cube) in meters.
    pub half_size: f64,
    /// Initial position (for resetting after place).
    pub home_position: [f64; 3],
}

impl GraspObject {
    /// Create a new object at the pick position.
    pub fn new(position: [f64; 3], mass: f64) -> Self {
        Self {
            position,
            mass,
            grasped: false,
            half_size: 0.025, // 5cm cube
            home_position: position,
        }
    }

    /// Default: 0.5kg cube at the pick position.
    pub fn at_pick() -> Self {
        Self::new([0.4, -0.3, 0.08], 0.5)
    }

    /// Update object state based on gripper and EE position.
    ///
    /// - If gripper is closing (< 0.3) and EE is near object → grasp
    /// - If gripper is opening (> 0.7) and object is grasped → release
    /// - If grasped, object tracks EE position
    pub fn update(&mut self, ee_position: &[f64; 3], gripper_opening: f64) {
        if self.grasped {
            if gripper_opening > 0.7 {
                // Release: object stays where it is
                self.grasped = false;
            } else {
                // Track EE position (object moves with gripper)
                self.position = *ee_position;
                // Slight offset below EE (held in gripper)
                self.position[2] -= self.half_size;
            }
        } else if gripper_opening < 0.3 {
            // Attempt grasp: check proximity
            let dist = self.distance_to(ee_position);
            if dist < self.half_size * 3.0 {
                self.grasped = true;
            }
        }
    }

    /// Distance from object center to a point.
    pub fn distance_to(&self, point: &[f64; 3]) -> f64 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Gravitational force on EE when object is grasped [Fx, Fy, Fz].
    /// Returns [0, 0, -mass*g] when grasped, [0, 0, 0] otherwise.
    pub fn gravity_load(&self) -> [f64; 3] {
        if self.grasped {
            [0.0, 0.0, -self.mass * 9.81]
        } else {
            [0.0; 3]
        }
    }

    /// Reset object to home position (after a complete cycle).
    pub fn reset_to_home(&mut self) {
        self.position = self.home_position;
        self.grasped = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_at_pick_defaults() {
        let obj = GraspObject::at_pick();
        assert_eq!(obj.mass, 0.5);
        assert!(!obj.grasped);
        assert_eq!(obj.position, [0.4, -0.3, 0.08]);
    }

    #[test]
    fn grasp_on_proximity_and_close() {
        let mut obj = GraspObject::at_pick();
        let ee = [0.4, -0.3, 0.1]; // Near object
        obj.update(&ee, 0.2); // Gripper closing
        assert!(obj.grasped, "Should grasp when close and gripper closed");
    }

    #[test]
    fn no_grasp_when_far() {
        let mut obj = GraspObject::at_pick();
        let ee = [0.8, 0.5, 0.5]; // Far from object
        obj.update(&ee, 0.2);
        assert!(!obj.grasped, "Should not grasp when far");
    }

    #[test]
    fn release_on_open() {
        let mut obj = GraspObject::at_pick();
        // First grasp
        let ee = [0.4, -0.3, 0.1];
        obj.update(&ee, 0.1);
        assert!(obj.grasped);
        // Then release
        obj.update(&ee, 0.9);
        assert!(!obj.grasped, "Should release when gripper opens");
    }

    #[test]
    fn grasped_object_tracks_ee() {
        let mut obj = GraspObject::at_pick();
        let ee = [0.4, -0.3, 0.1];
        obj.update(&ee, 0.1); // Grasp
        assert!(obj.grasped);
        let new_ee = [0.5, 0.2, 0.4]; // Move EE
        obj.update(&new_ee, 0.1);
        assert!(
            (obj.position[0] - 0.5).abs() < 0.01,
            "Object should track EE x"
        );
        assert!(
            (obj.position[1] - 0.2).abs() < 0.01,
            "Object should track EE y"
        );
    }

    #[test]
    fn gravity_load_when_grasped() {
        let mut obj = GraspObject::at_pick();
        assert_eq!(obj.gravity_load(), [0.0; 3]); // Not grasped
        let ee = [0.4, -0.3, 0.1];
        obj.update(&ee, 0.1);
        let load = obj.gravity_load();
        assert!(load[2] < -4.0, "Should have downward gravity: {}", load[2]);
        assert!((load[2] - (-0.5 * 9.81)).abs() < 0.01);
    }

    #[test]
    fn reset_to_home() {
        let mut obj = GraspObject::at_pick();
        let ee = [0.4, -0.3, 0.1];
        obj.update(&ee, 0.1); // Grasp
        obj.update(&[0.5, 0.2, 0.4], 0.1); // Move
        obj.reset_to_home();
        assert!(!obj.grasped);
        assert_eq!(obj.position, obj.home_position);
    }
}
