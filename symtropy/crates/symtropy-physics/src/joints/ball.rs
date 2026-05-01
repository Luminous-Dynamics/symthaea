// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ball joint (spherical joint): constrains a point on body A to coincide
//! with a point on body B, removing all translational DOF at the joint.
//!
//! In 3D: 3 translational DOF removed, 3 rotational DOF remain.
//! In ND: D translational DOF removed, D*(D-1)/2 rotational DOF remain.
//!
//! This is the fundamental articulation joint — pendulums, ragdoll shoulders,
//! robotic ball-and-socket connections.

use nalgebra::SVector;

use crate::body::{BodyHandle, RigidBody};
use crate::constraint::Constraint;

/// Ball joint: constrains local-space anchors on two bodies to coincide in world space.
pub struct BallJoint<const D: usize> {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    /// Local-space anchor on body A (relative to A's center of mass).
    pub anchor_a: SVector<f64, D>,
    /// Local-space anchor on body B (relative to B's center of mass).
    pub anchor_b: SVector<f64, D>,
    /// Constraint stiffness [0, 1]. 1.0 = rigid.
    pub stiffness: f64,
}

impl<const D: usize> BallJoint<D> {
    /// Create a ball joint with anchors at each body's origin.
    pub fn new(body_a: BodyHandle, body_b: BodyHandle) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a: SVector::zeros(),
            anchor_b: SVector::zeros(),
            stiffness: 1.0,
        }
    }

    /// Create with specific local-space anchors.
    pub fn with_anchors(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: SVector<f64, D>,
        anchor_b: SVector<f64, D>,
    ) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            stiffness: 1.0,
        }
    }

    /// Compute world-space anchor positions for given body states.
    fn world_anchors(
        &self,
        body_a: &RigidBody<D>,
        body_b: &RigidBody<D>,
    ) -> (SVector<f64, D>, SVector<f64, D>) {
        let wa = body_a.transform.translation.0
            + body_a.transform.rotation.rotate_vector(&self.anchor_a);
        let wb = body_b.transform.translation.0
            + body_b.transform.rotation.rotate_vector(&self.anchor_b);
        (wa, wb)
    }
}

impl<const D: usize> Constraint<D> for BallJoint<D> {
    fn bodies(&self) -> (BodyHandle, BodyHandle) {
        (self.body_a, self.body_b)
    }

    fn solve(&self, body_a: &mut RigidBody<D>, body_b: &mut RigidBody<D>, _dt: f64) {
        let (world_a, world_b) = self.world_anchors(body_a, body_b);
        let error = world_b - world_a;

        let error_mag = error.norm();
        if error_mag < 1e-15 {
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
        // Compute relative velocity at the joint anchor points
        // For ball joints, we only constrain translational velocity at the anchor;
        // rotational velocity is free.
        let (world_a, world_b) = self.world_anchors(body_a, body_b);
        let joint_dir = world_b - world_a;
        let dist = joint_dir.norm();
        if dist < 1e-15 {
            return;
        }
        let normal = joint_dir / dist;

        // Relative velocity projected along the joint axis
        let rel_vel = body_b.linear_velocity - body_a.linear_velocity;
        let vel_along = rel_vel.dot(&normal);

        let total_inv = body_a.inv_mass + body_b.inv_mass;
        if total_inv < 1e-15 {
            return;
        }

        // Apply impulse to eliminate relative velocity along the constraint direction
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

#[cfg(test)]
mod tests {
    use super::*;
    use symtropy_math::Point;

    #[test]
    fn ball_joint_connects_bodies() {
        let mut a =
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 0.5, 1.0);
        let mut b =
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([4.0, 0.0, 0.0]), 0.5, 1.0);

        let joint = BallJoint::new(BodyHandle(0), BodyHandle(1));

        for _ in 0..20 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        let dist = a.transform.translation.distance(&b.transform.translation);
        assert!(
            dist < 0.1,
            "ball joint should connect bodies, dist = {dist}"
        );
    }

    #[test]
    fn ball_joint_with_offset_anchors() {
        let mut a =
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 0.5, 1.0);
        let mut b =
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([5.0, 0.0, 0.0]), 0.5, 1.0);

        // Anchor at the tip of A (+1 along X) meets anchor at base of B (-1 along X)
        let joint = BallJoint::with_anchors(
            BodyHandle(0),
            BodyHandle(1),
            SVector::from([1.0, 0.0, 0.0]),
            SVector::from([-1.0, 0.0, 0.0]),
        );

        for _ in 0..30 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        let (wa, wb) = joint.world_anchors(&a, &b);
        let anchor_dist = (wb - wa).norm();
        assert!(
            anchor_dist < 0.1,
            "ball joint anchor points should coincide, dist = {anchor_dist}"
        );

        // Bodies should be ~2.0 apart (1.0 from each anchor offset)
        let body_dist = a.transform.translation.distance(&b.transform.translation);
        assert!(
            (body_dist - 2.0).abs() < 0.2,
            "bodies should be ~2.0 apart via anchors, dist = {body_dist}"
        );
    }

    #[test]
    fn ball_joint_maintains_anchor_after_displacement() {
        // Displace body B, then solve — anchors should converge back together
        let mut a = RigidBody::<3>::static_body(
            BodyHandle(0),
            Point::origin(),
            Box::new(symtropy_math::Sphere::new(Point::origin(), 0.1)),
        );
        let mut b =
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([4.0, 0.0, 0.0]), 0.5, 1.0);

        let joint = BallJoint::with_anchors(
            BodyHandle(0),
            BodyHandle(1),
            SVector::from([2.0, 0.0, 0.0]),
            SVector::from([-2.0, 0.0, 0.0]),
        );

        // Displace body B upward (simulating an external force)
        b.transform.translation.0[1] = 3.0;

        // Solve iterations — the joint should pull body B back
        for _ in 0..30 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        // Anchors should be close to coinciding
        let (wa, wb) = joint.world_anchors(&a, &b);
        let anchor_dist = (wb - wa).norm();
        assert!(
            anchor_dist < 0.5,
            "ball joint anchors should converge after displacement, dist = {anchor_dist}"
        );
    }

    #[test]
    fn ball_joint_4d() {
        let mut a = RigidBody::<4>::dynamic_sphere(BodyHandle(0), Point::origin(), 0.5, 1.0);
        let mut b = RigidBody::<4>::dynamic_sphere(
            BodyHandle(1),
            Point::new([3.0, 0.0, 0.0, 0.0]),
            0.5,
            1.0,
        );

        let joint = BallJoint::new(BodyHandle(0), BodyHandle(1));

        for _ in 0..20 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        let dist = a.transform.translation.distance(&b.transform.translation);
        assert!(dist < 0.1, "4D ball joint, dist = {dist}");
    }

    #[test]
    fn ball_joint_static_anchor_doesnt_move() {
        let mut anchor = RigidBody::<3>::static_body(
            BodyHandle(0),
            Point::new([0.0, 0.0, 0.0]),
            Box::new(symtropy_math::Sphere::new(Point::origin(), 0.1)),
        );
        let mut free =
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([5.0, 0.0, 0.0]), 0.5, 1.0);

        let joint = BallJoint::new(BodyHandle(0), BodyHandle(1));

        for _ in 0..20 {
            joint.solve(&mut anchor, &mut free, 0.016);
        }

        // Static body should not move
        assert!(
            anchor.transform.translation.coord(0).abs() < 1e-10,
            "static anchor moved!"
        );
        // Free body should move to the anchor
        let dist = anchor
            .transform
            .translation
            .distance(&free.transform.translation);
        assert!(dist < 0.1, "free body should reach anchor, dist = {dist}");
    }
}
