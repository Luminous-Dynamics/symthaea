// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Bouncing spheres: minimal physics demo.
//!
//! 20 spheres fall under gravity, bounce off the floor and each other.
//! Demonstrates: PhysicsWorld, GJK collision, EPA contact, Coulomb friction.
//!
//! Run: cargo run -p symtropy-physics --example bouncing_spheres

use nalgebra::SVector;
use symtropy_math::{Point, Sphere};
use symtropy_physics::{BodyHandle, PhysicsWorld, RigidBody};

fn main() {
    let mut world = PhysicsWorld::<3>::new(SVector::from([0.0, -9.81, 0.0]));

    // 20 spheres in a grid, some moving toward each other
    for i in 0..20 {
        let x = (i % 5) as f64 * 4.0 - 8.0;
        let y = (i / 5) as f64 * 4.0 + 2.0;
        let h = world.add_sphere(Point::new([x, y, 0.0]), 0.5, 1.0);
        world.body_mut(h).unwrap().restitution = 0.7;
        world.body_mut(h).unwrap().linear_damping = 0.0;
    }

    // Simulate 5 seconds at 64Hz
    println!("Simulating 20 spheres for 5 seconds...");
    println!(
        "{:<8} {:<12} {:<12} {:<8}",
        "Time", "KE (J)", "Sleeping", "Contacts"
    );
    println!("{}", "-".repeat(44));

    for step in 0..(5 * 64) {
        world.step(1.0 / 64.0);

        if step % 64 == 0 {
            let time = step as f64 / 64.0;
            let ke = world.total_kinetic_energy();
            let sleeping = world.sleeping_count();
            let contacts = world.contacts.len();
            println!(
                "{:<8.1} {:<12.3} {:<12} {:<8}",
                time, ke, sleeping, contacts
            );
        }
    }

    println!("\nFinal state:");
    for body in &world.bodies {
        if body.body_type == symtropy_physics::BodyType::Dynamic {
            let p = body.position();
            println!(
                "  Body {}: y={:.2} sleeping={}",
                body.handle.0,
                p.coord(1),
                body.sleeping
            );
        }
    }
}
