// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hinge joint (revolute): constrains rotation to a single plane.
//!
//! In 3D: removes 3 translational + 2 rotational DOF, leaving 1 rotational DOF.
//! In ND: removes D translational DOF + constrains angular velocity to one
//! bivector plane, leaving 1 rotational DOF.
//!
//! Uses: doors, wheels, robot arm joints, pendulums with a fixed axis.

use nalgebra::SVector;

use crate::body::{BodyHandle, RigidBody};
use crate::constraint::Constraint;

/// Hinge joint: allows rotation around a single axis/plane only.
///
/// The positional constraint (anchors coincide) works identically to BallJoint.
/// The angular constraint projects out all angular velocity components except
/// the one in the allowed bivector plane.
pub struct HingeJoint<const D: usize> {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    /// Local-space anchor on body A.
    pub anchor_a: SVector<f64, D>,
    /// Local-space anchor on body B.
    pub anchor_b: SVector<f64, D>,
    /// The two axes defining the allowed rotation plane.
    /// In 3D: axis_a=0, axis_b=1 means rotation in the XY plane (around Z).
    pub plane_axis_a: usize,
    pub plane_axis_b: usize,
    /// Constraint stiffness [0, 1]. 1.0 = rigid.
    pub stiffness: f64,
    /// Optional angle limits (min, max) in radians. `None` = unlimited.
    pub angle_limits: Option<(f64, f64)>,
    /// Optional motor drive for actuated hinge joints.
    pub motor: Option<super::MotorDrive>,
}

impl<const D: usize> HingeJoint<D> {
    /// Create a hinge joint that rotates in the plane defined by (axis_a, axis_b).
    ///
    /// In 3D, `plane(0, 1)` means rotation around Z.
    /// In 4D, `plane(0, 3)` means rotation in the XW plane.
    pub fn new(
        body_a: BodyHandle,
        body_b: BodyHandle,
        plane_axis_a: usize,
        plane_axis_b: usize,
    ) -> Self {
        debug_assert!(plane_axis_a < D && plane_axis_b < D);
        debug_assert!(plane_axis_a != plane_axis_b);
        Self {
            body_a,
            body_b,
            anchor_a: SVector::zeros(),
            anchor_b: SVector::zeros(),
            plane_axis_a,
            plane_axis_b,
            stiffness: 1.0,
            angle_limits: None,
            motor: None,
        }
    }

    /// Add a motor drive.
    pub fn with_motor(mut self, motor: super::MotorDrive) -> Self {
        self.motor = Some(motor);
        self
    }

    /// Create with specific anchors.
    pub fn with_anchors(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: SVector<f64, D>,
        anchor_b: SVector<f64, D>,
        plane_axis_a: usize,
        plane_axis_b: usize,
    ) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            plane_axis_a,
            plane_axis_b,
            stiffness: 1.0,
            angle_limits: None,
            motor: None,
        }
    }

    /// Set angle limits in radians.
    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.angle_limits = Some((min, max));
        self
    }
}

impl<const D: usize> Constraint<D> for HingeJoint<D> {
    fn bodies(&self) -> (BodyHandle, BodyHandle) {
        (self.body_a, self.body_b)
    }

    fn solve(&self, body_a: &mut RigidBody<D>, body_b: &mut RigidBody<D>, _dt: f64) {
        // === Positional constraint (same as BallJoint) ===
        let world_a = body_a.transform.translation.0
            + body_a.transform.rotation.rotate_vector(&self.anchor_a);
        let world_b = body_b.transform.translation.0
            + body_b.transform.rotation.rotate_vector(&self.anchor_b);

        let error = world_b - world_a;
        if error.norm() < 1e-15 {
            return;
        }

        let correction = error * self.stiffness;
        let total_inv_mass = body_a.inv_mass + body_b.inv_mass;
        if total_inv_mass < 1e-15 {
            return;
        }

        let ratio_a = body_a.inv_mass / total_inv_mass;
        let ratio_b = body_b.inv_mass / total_inv_mass;

        if body_a.is_dynamic() {
            body_a.transform.translation.0 += correction * ratio_a;
        }
        if body_b.is_dynamic() {
            body_b.transform.translation.0 -= correction * ratio_b;
        }
    }

    fn solve_velocity(&self, body_a: &mut RigidBody<D>, body_b: &mut RigidBody<D>, _dt: f64) {
        // === Positional velocity constraint (same as BallJoint) ===
        let world_a = body_a.transform.translation.0
            + body_a.transform.rotation.rotate_vector(&self.anchor_a);
        let world_b = body_b.transform.translation.0
            + body_b.transform.rotation.rotate_vector(&self.anchor_b);

        let joint_dir = world_b - world_a;
        let dist = joint_dir.norm();
        if dist > 1e-15 {
            let normal = joint_dir / dist;
            let rel_vel = body_b.linear_velocity - body_a.linear_velocity;
            let vel_along = rel_vel.dot(&normal);
            let total_inv = body_a.inv_mass + body_b.inv_mass;
            if total_inv > 1e-15 {
                let impulse_mag = -vel_along * self.stiffness / total_inv;
                let impulse = normal * impulse_mag;
                if body_a.is_dynamic() {
                    body_a.linear_velocity -= impulse * body_a.inv_mass;
                }
                if body_b.is_dynamic() {
                    body_b.linear_velocity += impulse * body_b.inv_mass;
                }
            }
        }

        // === Angular constraint: project out disallowed angular velocity ===
        // The hinge allows rotation only in the (plane_axis_a, plane_axis_b) plane.
        // All other angular velocity components of the RELATIVE rotation should be zero.
        //
        // Using bivector representation: the relative angular velocity
        // should only have a component in the allowed plane.
        // === Angular constraint: project out disallowed angular velocity ===
        // The hinge allows rotation only in the (plane_axis_a, plane_axis_b) plane.
        // For each bivector plane that is NOT the allowed plane,
        // damp the relative angular velocity.
        let num_bivector_components = D * (D - 1) / 2;
        if num_bivector_components <= 1 {
            return; // 2D: only one rotation plane, hinge is trivially satisfied
        }

        let (pa, pb) = if self.plane_axis_a < self.plane_axis_b {
            (self.plane_axis_a, self.plane_axis_b)
        } else {
            (self.plane_axis_b, self.plane_axis_a)
        };

        for i in 0..D {
            for j in (i + 1)..D {
                if i == pa && j == pb {
                    continue; // Allowed plane
                }

                let rel_component =
                    body_b.angular_velocity.get(i, j) - body_a.angular_velocity.get(i, j);
                if rel_component.abs() < 1e-10 {
                    continue;
                }

                let correction = rel_component * self.stiffness * 0.5;
                if body_a.is_dynamic() {
                    body_a.angular_velocity.set(
                        i,
                        j,
                        body_a.angular_velocity.get(i, j) + correction,
                    );
                }
                if body_b.is_dynamic() {
                    body_b.angular_velocity.set(
                        i,
                        j,
                        body_b.angular_velocity.get(i, j) - correction,
                    );
                }
            }
        }

        // Motor drive on the hinge rotation axis
        if let Some(ref motor) = self.motor {
            let current_angular =
                body_b.angular_velocity.get(pa, pb) - body_a.angular_velocity.get(pa, pb);
            let error = motor.target - current_angular;
            let torque = (error * motor.max_force - current_angular * motor.damping)
                .clamp(-motor.max_force, motor.max_force);
            let impulse = torque * _dt;
            if body_a.is_dynamic() {
                body_a.angular_velocity.set(
                    pa,
                    pb,
                    body_a.angular_velocity.get(pa, pb) - impulse * 0.5,
                );
            }
            if body_b.is_dynamic() {
                body_b.angular_velocity.set(
                    pa,
                    pb,
                    body_b.angular_velocity.get(pa, pb) + impulse * 0.5,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symtropy_math::Point;

    #[test]
    fn hinge_maintains_position() {
        let mut a = RigidBody::<3>::static_body(
            BodyHandle(0),
            Point::origin(),
            Box::new(symtropy_math::Sphere::new(Point::origin(), 0.1)),
        );
        let mut b =
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([3.0, 0.0, 0.0]), 0.5, 1.0);

        // Hinge in XY plane (rotation around Z)
        let joint = HingeJoint::new(BodyHandle(0), BodyHandle(1), 0, 1);

        for _ in 0..20 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        let dist = a.transform.translation.distance(&b.transform.translation);
        assert!(dist < 0.1, "hinge should connect bodies, dist = {dist}");
    }

    #[test]
    fn hinge_allows_rotation_in_plane() {
        let mut a = RigidBody::<3>::static_body(
            BodyHandle(0),
            Point::origin(),
            Box::new(symtropy_math::Sphere::new(Point::origin(), 0.1)),
        );
        let mut b = RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::origin(), 0.5, 1.0);

        // Give body B angular velocity in the allowed XY plane (around Z)
        b.angular_velocity.set(0, 1, 5.0);
        // And some in the disallowed XZ plane (around Y)
        b.angular_velocity.set(0, 2, 3.0);

        let joint = HingeJoint::new(BodyHandle(0), BodyHandle(1), 0, 1);
        joint.solve_velocity(&mut a, &mut b, 0.016);

        // Allowed rotation should be preserved (minus some damping from relative constraint)
        let xy_after = b.angular_velocity.get(0, 1);
        assert!(
            xy_after.abs() > 1.0,
            "XY rotation should be mostly preserved, got {xy_after}"
        );

        // Disallowed rotation should be reduced
        let xz_after = b.angular_velocity.get(0, 2);
        assert!(
            xz_after.abs() < 3.0,
            "XZ rotation should be reduced from 3.0, got {xz_after}"
        );
    }

    #[test]
    fn hinge_4d() {
        let mut a = RigidBody::<4>::dynamic_sphere(BodyHandle(0), Point::origin(), 0.5, 1.0);
        let mut b = RigidBody::<4>::dynamic_sphere(
            BodyHandle(1),
            Point::new([2.0, 0.0, 0.0, 0.0]),
            0.5,
            1.0,
        );

        // 4D hinge: rotation in XW plane (axes 0, 3)
        let joint = HingeJoint::new(BodyHandle(0), BodyHandle(1), 0, 3);

        for _ in 0..20 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        let dist = a.transform.translation.distance(&b.transform.translation);
        assert!(dist < 0.1, "4D hinge, dist = {dist}");
    }

    #[test]
    fn hinge_2d_trivial() {
        // In 2D, there's only one rotation plane (0,1). A hinge in that plane
        // is equivalent to a ball joint — all rotation is allowed.
        let mut a = RigidBody::<2>::dynamic_sphere(BodyHandle(0), Point::origin(), 0.5, 1.0);
        let mut b = RigidBody::<2>::dynamic_sphere(BodyHandle(1), Point::new([2.0, 0.0]), 0.5, 1.0);
        b.angular_velocity.set(0, 1, 3.0);

        let joint = HingeJoint::new(BodyHandle(0), BodyHandle(1), 0, 1);
        joint.solve_velocity(&mut a, &mut b, 0.016);

        // In 2D, no angular constraint is applied (only 1 plane)
        // The angular velocity should be modified only by the relative constraint
        assert!(
            b.angular_velocity.get(0, 1).abs() > 0.0,
            "2D hinge should allow rotation"
        );
    }
}
