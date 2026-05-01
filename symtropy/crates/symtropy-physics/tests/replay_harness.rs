// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use nalgebra::SVector;
use symtropy_math::{Point, Sphere};
use symtropy_physics::body::{BodyHandle, RigidBody};
use symtropy_physics::replay::{apply_commands, ReplayTape, WorldCommand, WorldSnapshot};
use symtropy_physics::PhysicsWorld;

fn build_world() -> (PhysicsWorld<3>, [BodyHandle; 3]) {
    // Use power-of-two fractions where possible to reduce noise and keep tape "clean".
    let gravity = SVector::from([0.0, -10.0, 0.0]);
    let mut world = PhysicsWorld::<3>::new(gravity);
    world.solver_iterations = 6;
    world.sleep_threshold = 0.25;
    world.sleep_ticks = 32;

    // Ground: sphere of radius 64 centered at y=-64 → top at y=0.
    let ground = RigidBody::<3>::static_body(
        BodyHandle(0),
        Point::new([0.0, -64.0, 0.0]),
        Box::new(Sphere::<3>::new(Point::origin(), 64.0)),
    );
    world.add_body(ground);

    let a = world.add_sphere(Point::new([-1.5, 6.0, 0.0]), 0.5, 1.0);
    let b = world.add_sphere(Point::new([1.5, 6.0, 0.0]), 0.5, 1.0);
    let c = world.add_sphere(Point::new([0.0, 9.0, 0.0]), 0.5, 2.0);

    world.body_mut(a).unwrap().linear_velocity = SVector::from([2.0, 0.0, 0.0]);
    world.body_mut(b).unwrap().linear_velocity = SVector::from([-2.0, 0.0, 0.0]);

    (world, [a, b, c])
}

fn build_tape(handles: [BodyHandle; 3]) -> ReplayTape<3> {
    let [a, b, c] = handles;
    let dt = 1.0 / 64.0;

    let mut tape = ReplayTape::<3>::default();
    for tick in 0..192u32 {
        let fx = ((tick as i32 % 32) - 16) as f64 / 128.0; // exact multiples of 1/128
        let fz = ((tick as i32 % 16) - 8) as f64 / 256.0; // exact multiples of 1/256

        let mut commands = vec![
            WorldCommand::ApplyForce {
                body: a,
                force: SVector::from([fx, 0.0, fz]),
            },
            WorldCommand::ApplyForce {
                body: b,
                force: SVector::from([-fx, 0.0, -fz]),
            },
        ];

        // Occasionally "nudge" the third body so collision timing depends on commands.
        if tick % 24 == 0 {
            commands.push(WorldCommand::ApplyImpulse {
                body: c,
                impulse: SVector::from([0.25, 0.0, 0.0]),
            });
        }

        tape.push_frame(dt, commands);
    }
    tape
}

#[test]
fn record_and_replay_are_bitwise_identical_each_tick() {
    let (mut record_world, handles) = build_world();
    let tape = build_tape(handles);

    let mut recorded: Vec<WorldSnapshot<3>> = Vec::with_capacity(tape.frames.len());
    for frame in &tape.frames {
        apply_commands(&mut record_world, &frame.commands).unwrap();
        record_world.step(frame.dt);
        recorded.push(WorldSnapshot::capture(&record_world));
    }

    let (mut replay_world, _handles2) = build_world();
    for (tick, frame) in tape.frames.iter().enumerate() {
        apply_commands(&mut replay_world, &frame.commands).unwrap();
        replay_world.step(frame.dt);
        let snapshot = WorldSnapshot::capture(&replay_world);
        assert_eq!(snapshot, recorded[tick], "diverged at tick {tick}");
    }
}
