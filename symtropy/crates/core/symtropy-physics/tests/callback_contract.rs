// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

use nalgebra::SVector;
use symtropy_math::Point;
use symtropy_physics::body::BodyHandle;
use symtropy_physics::contact::CollisionEvent;
use symtropy_physics::world::{PhysicsCallback, PhysicsWorld};

struct MockCallback {
    force_gain: f64,
    impulse_gain: f64,
    friction_mult: f64,
    collision_called: bool,
    dissipation_recorded: f64,
}

impl MockCallback {
    fn new() -> Self {
        Self {
            force_gain: 1.0,
            impulse_gain: 1.0,
            friction_mult: 1.0,
            collision_called: false,
            dissipation_recorded: 0.0,
        }
    }
}

impl PhysicsCallback<2> for MockCallback {
    fn modulate_force(&self, _: BodyHandle, force: &SVector<f64, 2>) -> SVector<f64, 2> {
        force * self.force_gain
    }

    fn modulate_impulse(&self, impulse: f64, _: &SVector<f64, 2>) -> f64 {
        impulse * self.impulse_gain
    }

    fn friction_multiplier(&self, _: &SVector<f64, 2>, _: BodyHandle) -> f64 {
        self.friction_mult
    }

    fn on_collision(&mut self, _: &CollisionEvent<2>) {
        self.collision_called = true;
    }

    fn record_dissipation(&mut self, energy: f64) {
        self.dissipation_recorded += energy;
    }

    fn apply_trauma(&mut self, _: &CollisionEvent<2>) {}

    fn record_work(&mut self, _: BodyHandle, _: f64) {}
}

#[test]
fn test_modulate_force_blocks_or_preserves() {
    let gravity = SVector::zeros();
    let mut world = PhysicsWorld::<2>::new(gravity);
    let h = world.add_sphere(Point::origin(), 1.0, 1.0);

    // Disable damping for clean unit testing
    world.body_mut(h).unwrap().linear_damping = 0.0;

    // 1. Zero gain blocks force
    {
        let mut cb = MockCallback::new();
        cb.force_gain = 0.0;

        world.body_mut(h).unwrap().force_accumulator = SVector::from([10.0, 0.0]);
        world.step_with_callback(0.1, &mut cb);

        let vel = world.body(h).unwrap().linear_velocity;
        assert_eq!(vel.norm(), 0.0, "Zero gain should block force");
    }

    // 2. Unit gain preserves force
    {
        // Reset body
        world.body_mut(h).unwrap().linear_velocity = SVector::zeros();
        world.body_mut(h).unwrap().transform.translation = Point::origin();

        let mut cb = MockCallback::new();
        cb.force_gain = 1.0;

        world.body_mut(h).unwrap().force_accumulator = SVector::from([10.0, 0.0]);
        world.step_with_callback(0.1, &mut cb);

        let vel = world.body(h).unwrap().linear_velocity;
        // f = m*a => 10 = 1*a => a = 10. v = a*dt = 10*0.1 = 1.0
        assert!(
            (vel.norm() - 1.0).abs() < 1e-10,
            "Unit gain should preserve force, got {}",
            vel.norm()
        );
    }
}

#[test]
fn test_modulate_impulse_blocks_or_preserves() {
    let gravity = SVector::zeros();
    let mut world = PhysicsWorld::<2>::new(gravity);

    // Two spheres already overlapping slightly to ensure collision resolution is needed
    let h1 = world.add_sphere(Point::new([-0.9, 0.0]), 1.0, 1.0);
    let h2 = world.add_sphere(Point::new([0.9, 0.0]), 1.0, 1.0);

    // Disable damping
    world.body_mut(h1).unwrap().linear_damping = 0.0;
    world.body_mut(h2).unwrap().linear_damping = 0.0;

    // Set velocities towards each other
    world.body_mut(h1).unwrap().linear_velocity = SVector::from([1.0, 0.0]);
    world.body_mut(h2).unwrap().linear_velocity = SVector::from([-1.0, 0.0]);

    // 1. Zero gain blocks impulse (bodies keep their velocities)
    {
        let mut cb = MockCallback::new();
        cb.impulse_gain = 0.0;

        world.step_with_callback(0.1, &mut cb);

        let v1 = world.body(h1).unwrap().linear_velocity;
        let v2 = world.body(h2).unwrap().linear_velocity;

        assert_eq!(v1[0], 1.0, "Zero gain should block collision impulse");
        assert_eq!(v2[0], -1.0, "Zero gain should block collision impulse");
    }

    // 2. Unit gain preserves impulse (bounce)
    {
        // Reset bodies
        world.body_mut(h1).unwrap().transform.translation = Point::new([-0.9, 0.0]);
        world.body_mut(h2).unwrap().transform.translation = Point::new([0.9, 0.0]);
        world.body_mut(h1).unwrap().linear_velocity = SVector::from([1.0, 0.0]);
        world.body_mut(h2).unwrap().linear_velocity = SVector::from([-1.0, 0.0]);

        let mut cb = MockCallback::new();
        cb.impulse_gain = 1.0;

        world.step_with_callback(0.1, &mut cb);

        let v1 = world.body(h1).unwrap().linear_velocity;
        let v2 = world.body(h2).unwrap().linear_velocity;

        // With unit gain, impulse should change velocities significantly
        assert!(
            v1[0] < 0.5,
            "Unit gain should resolve collision, got v1={}",
            v1[0]
        );
        assert!(
            v2[0] > -0.5,
            "Unit gain should resolve collision, got v2={}",
            v2[0]
        );
    }
}
