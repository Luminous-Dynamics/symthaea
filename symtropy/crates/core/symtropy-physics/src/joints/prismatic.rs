// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prismatic joint (slider): constrains motion to a single axis.
//!
//! Removes D-1 translational DOF and all rotational DOF, leaving 1 translational DOF.
//! Uses: suspension springs, elevator platforms, sliding doors, linear actuators.

#[cfg(test)]
use nalgebra::SVector;

use crate::body::{BodyHandle, RigidBody};
use crate::constraint::Constraint;

/// Prismatic joint: body B slides along one axis relative to body A.
pub struct PrismaticJoint<const D: usize> {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    /// The axis along which sliding is permitted (0=X, 1=Y, 2=Z, ...).
    pub axis: usize,
    /// Optional displacement limits (min, max) relative to the initial offset.
    pub limits: Option<(f64, f64)>,
    /// Constraint stiffness [0, 1]. 1.0 = rigid.
    pub stiffness: f64,
    /// Optional motor drive for actuated prismatic joints.
    pub motor: Option<super::MotorDrive>,
}

impl<const D: usize> PrismaticJoint<D> {
    /// Create a prismatic joint along the given axis.
    pub fn new(body_a: BodyHandle, body_b: BodyHandle, axis: usize) -> Self {
        debug_assert!(axis < D, "axis {axis} out of range for D={D}");
        Self {
            body_a,
            body_b,
            axis,
            limits: None,
            stiffness: 1.0,
            motor: None,
        }
    }

    /// Add displacement limits.
    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.limits = Some((min, max));
        self
    }

    /// Add a motor drive.
    pub fn with_motor(mut self, motor: super::MotorDrive) -> Self {
        self.motor = Some(motor);
        self
    }
}

impl<const D: usize> Constraint<D> for PrismaticJoint<D> {
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
        let delta = body_b.transform.translation.0 - body_a.transform.translation.0;

        // Correct all axes EXCEPT the slide axis
        let total_inv_mass = body_a.inv_mass + body_b.inv_mass;
        if total_inv_mass < 1e-15 {
            return;
        }

        let ratio_a = body_a.inv_mass / total_inv_mass;
        let ratio_b = body_b.inv_mass / total_inv_mass;

        for i in 0..D {
            if i == self.axis {
                // Apply limits on the slide axis
                if let Some((min, max)) = self.limits {
                    let displacement = delta[i];
                    if displacement < min {
                        let correction = (min - displacement) * self.stiffness;
                        if body_a.is_dynamic() {
                            body_a.transform.translation.0[i] -= correction * ratio_a;
                        }
                        if body_b.is_dynamic() {
                            body_b.transform.translation.0[i] += correction * ratio_b;
                        }
                    } else if displacement > max {
                        let correction = (displacement - max) * self.stiffness;
                        if body_a.is_dynamic() {
                            body_a.transform.translation.0[i] += correction * ratio_a;
                        }
                        if body_b.is_dynamic() {
                            body_b.transform.translation.0[i] -= correction * ratio_b;
                        }
                    }
                }
                continue;
            }

            // Constrain non-slide axes to zero displacement
            let error = delta[i];
            if error.abs() < 1e-15 {
                continue;
            }
            let correction = error * self.stiffness;
            if body_a.is_dynamic() {
                body_a.transform.translation.0[i] += correction * ratio_a;
            }
            if body_b.is_dynamic() {
                body_b.transform.translation.0[i] -= correction * ratio_b;
            }
        }
    }

    fn solve_velocity(
        &self,
        body_a: &mut RigidBody<D>,
        body_b: &mut RigidBody<D>,
        dt: f64,
        mut callback: Option<&mut dyn crate::world::PhysicsCallback<D>>,
    ) {
        let total_inv = body_a.inv_mass + body_b.inv_mass;
        if total_inv < 1e-15 {
            return;
        }

        // Damp velocity on non-slide axes
        for i in 0..D {
            if i == self.axis {
                continue;
            }
            let rel_vel = body_b.linear_velocity[i] - body_a.linear_velocity[i];
            if rel_vel.abs() < 1e-15 {
                continue;
            }
            let impulse = -rel_vel * self.stiffness / total_inv;
            if body_a.is_dynamic() {
                body_a.linear_velocity[i] -= impulse * body_a.inv_mass;
            }
            if body_b.is_dynamic() {
                body_b.linear_velocity[i] += impulse * body_b.inv_mass;
            }
        }

        // Motor drive on slide axis
        if let Some(ref motor) = self.motor {
            let displacement = body_b.transform.translation.0[self.axis]
                - body_a.transform.translation.0[self.axis];
            let rel_vel = body_b.linear_velocity[self.axis] - body_a.linear_velocity[self.axis];

            // PD controller: impulse = (kp * (target - pos) + kd * (target_vel - vel)) * dt
            let impulse = motor.calculate_impulse(displacement, rel_vel, dt);

            // METABOLIC HEARTBEAT: Calculate Work and Energy Recovery
            // Work = Impulse * Velocity
            let work_joules = impulse * rel_vel;
            if let Some(ref mut cb) = callback {
                cb.record_work(self.body_b, work_joules);
            }

            if body_a.is_dynamic() {
                body_a.linear_velocity[self.axis] -= impulse * body_a.inv_mass;
            }
            if body_b.is_dynamic() {
                body_b.linear_velocity[self.axis] += impulse * body_b.inv_mass;
            }
        }

        // Damp all angular velocity (prismatic = no rotation)
        for i in 0..D {
            for j in (i + 1)..D {
                let rel = body_b.angular_velocity.get(i, j) - body_a.angular_velocity.get(i, j);
                if rel.abs() < 1e-10 {
                    continue;
                }
                let correction = rel * self.stiffness * 0.5;
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symtropy_math::Point;

    #[test]
    fn slides_along_axis() {
        let mut a = RigidBody::<3>::static_body(
            BodyHandle(0),
            Point::origin(),
            Box::new(symtropy_math::Sphere::new(Point::origin(), 0.1)),
        );
        let mut b =
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([2.0, 3.0, 0.0]), 0.5, 1.0);

        // Prismatic along X — Y should be corrected to 0, X preserved
        let joint = PrismaticJoint::new(BodyHandle(0), BodyHandle(1), 0);

        for _ in 0..20 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        // Y should be near 0 (constrained)
        assert!(
            b.transform.translation.coord(1).abs() < 0.1,
            "Y should be constrained to 0, got {}",
            b.transform.translation.coord(1)
        );
        // X should be preserved (free axis)
        assert!(
            b.transform.translation.coord(0).abs() > 0.5,
            "X should be free, got {}",
            b.transform.translation.coord(0)
        );
    }

    #[test]
    fn limits_respected() {
        let mut a = RigidBody::<3>::static_body(
            BodyHandle(0),
            Point::origin(),
            Box::new(symtropy_math::Sphere::new(Point::origin(), 0.1)),
        );
        let mut b =
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([10.0, 0.0, 0.0]), 0.5, 1.0);

        let joint = PrismaticJoint::new(BodyHandle(0), BodyHandle(1), 0).with_limits(-2.0, 5.0);

        for _ in 0..30 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        let x = b.transform.translation.coord(0);
        assert!(x <= 5.5, "X should be clamped to max 5.0, got {x}");
    }

    #[test]
    fn perpendicular_velocity_damped() {
        let mut a = RigidBody::<3>::static_body(
            BodyHandle(0),
            Point::origin(),
            Box::new(symtropy_math::Sphere::new(Point::origin(), 0.1)),
        );
        let mut b = RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::origin(), 0.5, 1.0);
        b.linear_velocity = SVector::from([0.0, 10.0, 5.0]); // perpendicular velocity

        let joint = PrismaticJoint::new(BodyHandle(0), BodyHandle(1), 0);
        joint.solve_velocity(&mut a, &mut b, 0.016, None);

        assert!(
            b.linear_velocity[1].abs() < 10.0,
            "Y velocity should be damped, got {}",
            b.linear_velocity[1]
        );
    }

    #[test]
    fn prismatic_4d() {
        let mut a = RigidBody::<4>::dynamic_sphere(BodyHandle(0), Point::origin(), 0.5, 1.0);
        let mut b = RigidBody::<4>::dynamic_sphere(
            BodyHandle(1),
            Point::new([0.0, 0.0, 0.0, 5.0]),
            0.5,
            1.0,
        );

        // Slide along W axis
        let joint = PrismaticJoint::new(BodyHandle(0), BodyHandle(1), 3);

        for _ in 0..20 {
            joint.solve(&mut a, &mut b, 0.016);
        }

        // W should be free, others constrained near 0
        assert!(
            b.transform.translation.coord(0).abs() < 0.1,
            "X should be constrained"
        );
    }
}
