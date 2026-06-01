// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fixed joint: rigidly attaches two bodies (zero degrees of freedom).
//!
//! Both positional and rotational offsets are maintained at their initial values.
//! Useful for building compound bodies from multiple collision shapes.

use nalgebra::SVector;

use crate::body::{BodyHandle, RigidBody};
use crate::constraint::Constraint;

/// Fixed joint: locks two bodies at a fixed relative position.
///
/// The joint maintains the initial relative offset `anchor_b - anchor_a`
/// (in world space at the time of creation). All translational DOF are removed.
pub struct FixedJoint<const D: usize> {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    /// Local-space anchor on body A.
    pub anchor_a: SVector<f64, D>,
    /// Local-space anchor on body B.
    pub anchor_b: SVector<f64, D>,
    /// Constraint stiffness [0, 1]. 1.0 = rigid, <1.0 = allows some drift.
    pub stiffness: f64,
}

impl<const D: usize> FixedJoint<D> {
    /// Create a fixed joint connecting body A and body B at their current positions.
    /// The anchors default to the body origins.
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
}

impl<const D: usize> Constraint<D> for FixedJoint<D> {
    fn bodies(&self) -> (BodyHandle, BodyHandle) {
        (self.body_a, self.body_b)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn solve(&self, body_a: &mut RigidBody<D>, body_b: &mut RigidBody<D>, _dt: f64) {
        // Compute world-space anchor positions
        let world_a = body_a.transform.translation.0
            + body_a.transform.rotation.rotate_vector(&self.anchor_a);
        let world_b = body_b.transform.translation.0
            + body_b.transform.rotation.rotate_vector(&self.anchor_b);

        // Positional error: the two anchors should coincide
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

    fn solve_velocity(
        &self,
        body_a: &mut RigidBody<D>,
        body_b: &mut RigidBody<D>,
        _dt: f64,
        _callback: Option<&mut dyn crate::world::PhysicsCallback<D>>,
    ) {
        // Damp all relative velocity at the joint point
        let rel_vel = body_b.linear_velocity - body_a.linear_velocity;
        let speed = rel_vel.norm();
        if speed < 1e-15 {
            return;
        }

        let total_inv = body_a.inv_mass + body_b.inv_mass;
        if total_inv < 1e-15 {
            return;
        }

        // Apply impulse to eliminate relative velocity (scaled by stiffness)
        let impulse = rel_vel * (-self.stiffness / total_inv);
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
    fn fixed_joint_holds_position() {
        let mut a =
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 0.5, 1.0);
        let mut b =
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([3.0, 0.0, 0.0]), 0.5, 1.0);

        let joint = FixedJoint::new(BodyHandle(0), BodyHandle(1));

        // Pull them together over iterations
        for _ in 0..20 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        let dist = a.transform.translation.distance(&b.transform.translation);
        assert!(
            dist < 0.1,
            "fixed joint should bring bodies together, dist = {dist}"
        );
    }

    #[test]
    fn fixed_joint_with_anchors() {
        let mut a =
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 0.5, 1.0);
        let mut b =
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([5.0, 0.0, 0.0]), 0.5, 1.0);

        // Anchor at (+1,0,0) on A should meet anchor at (-1,0,0) on B
        // So bodies should end up 2 apart (not 0)
        let joint = FixedJoint::with_anchors(
            BodyHandle(0),
            BodyHandle(1),
            SVector::from([1.0, 0.0, 0.0]),
            SVector::from([-1.0, 0.0, 0.0]),
        );

        for _ in 0..30 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        // The world-space anchors should coincide, meaning bodies are 2.0 apart
        let world_a = a.transform.translation.0 + SVector::from([1.0, 0.0, 0.0]);
        let world_b = b.transform.translation.0 + SVector::from([-1.0, 0.0, 0.0]);
        let anchor_dist = (world_b - world_a).norm();
        assert!(
            anchor_dist < 0.1,
            "fixed joint anchors should coincide, dist = {anchor_dist}"
        );
    }

    #[test]
    fn fixed_joint_velocity_damping() {
        let mut a = RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::origin(), 0.5, 1.0);
        let mut b = RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::origin(), 0.5, 1.0);
        b.linear_velocity = SVector::from([10.0, 0.0, 0.0]);

        let joint = FixedJoint::new(BodyHandle(0), BodyHandle(1));
        joint.solve_velocity(&mut a, &mut b, 0.016);

        // Relative velocity should be reduced
        let rel = (b.linear_velocity - a.linear_velocity).norm();
        assert!(rel < 10.0, "velocity should be damped, rel = {rel}");
    }

    #[test]
    fn fixed_joint_4d() {
        let mut a = RigidBody::<4>::dynamic_sphere(BodyHandle(0), Point::origin(), 0.5, 1.0);
        let mut b = RigidBody::<4>::dynamic_sphere(
            BodyHandle(1),
            Point::new([2.0, 0.0, 0.0, 0.0]),
            0.5,
            1.0,
        );

        let joint = FixedJoint::new(BodyHandle(0), BodyHandle(1));

        for _ in 0..20 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        let dist = a.transform.translation.distance(&b.transform.translation);
        assert!(dist < 0.1, "4D fixed joint, dist = {dist}");
    }
}
