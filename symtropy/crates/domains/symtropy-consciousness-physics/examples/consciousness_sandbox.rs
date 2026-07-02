// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration Sandbox: demonstrates the full Φ-coupled physics loop.
//!
//! 10 agents with consciousness in a 3D world:
//! - Agents move, consume energy, collide
//! - Collisions spike prediction error → motor precision drops
//! - Harmony resonance between nearby agents regenerates energy
//! - Agents with no energy collapse (Φ → 0, motor halt)
//! - Recovery requires another agent's harmony field
//!
//! Run: cargo run -p symtropy-consciousness-physics --example consciousness_sandbox

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::{BodyHandle, PhysicsWorld, RigidBody};

fn main() {
    println!("=== Symtropy Consciousness Sandbox ===\n");

    let mut world = PhysicsWorld::<3>::new(SVector::from([0.0, 0.0, 0.0])); // no gravity (top-down)
    let mut consciousness = ConsciousnessField::<3>::new();

    // Tune constants for fast demonstration
    consciousness.constants.consciousness_maintenance_per_tick = 0.5; // faster drain
    consciousness.constants.harmony_resonance_regen_rate = 0.8;
    consciousness.constants.ambient_regen_rate = 0.05;
    consciousness.constants.initial_energy = 200.0;
    consciousness.constants.max_energy = 200.0;

    // Spawn 10 agents in a circle
    let mut handles = Vec::new();
    for i in 0..10 {
        let angle = (i as f64) * std::f64::consts::TAU / 10.0;
        let x = angle.cos() * 20.0;
        let y = angle.sin() * 20.0;
        let h = world.add_sphere(Point::new([x, y, 0.0]), 1.0, 1.0);
        world.body_mut(h).unwrap().linear_damping = 0.1;

        // Give agents initial velocity toward center (they'll collide)
        world.body_mut(h).unwrap().linear_velocity = SVector::from([-x * 0.3, -y * 0.3, 0.0]);

        consciousness.register(h, 200.0, 15.0);

        // Set harmony activations — alternate between two profiles
        if let Some(entity) = consciousness.entities.get_mut(&h) {
            if i % 2 == 0 {
                entity.harmony_activations = [0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.5];
            // Stillness-heavy
            } else {
                entity.harmony_activations = [0.3, 0.1, 0.8, 0.1, 0.1, 0.1, 0.7, 0.3, 0.5];
                // Progression-heavy
            }
        }
        handles.push(h);
    }

    println!(
        "{:<6} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Tick", "Coll.Φ", "Alive", "Collapsed", "Pred.Err", "Energy"
    );
    println!("{}", "-".repeat(56));

    // Simulate 500 ticks
    for tick in 0..500 {
        // Update consciousness for all agents
        for &h in &handles {
            let phi = consciousness.phi(h);
            let inputs = ConsciousnessInputs {
                phi: if consciousness
                    .entities
                    .get(&h)
                    .map(|e| e.energy.is_collapsed())
                    .unwrap_or(false)
                {
                    0.0
                } else {
                    0.6 + phi * 0.2
                },
                broadcast: 0.7,
                working_memory: 0.6,
                attention: 0.6,
                recurrence: 0.5,
                embodiment: 0.7,
                knowledge: 0.5,
                synchrony: 0.6,
            };
            consciousness.update_entity(h, &inputs, Point::origin());
        }

        // Consciousness maintenance cost
        for &h in &handles {
            if let Some(entity) = consciousness.entities.get_mut(&h) {
                let phi = entity.phi();
                let cost =
                    consciousness.constants.consciousness_maintenance_per_tick * (1.0 + phi * 0.5);
                entity.energy.consume(cost);

                // Ambient regen
                entity
                    .energy
                    .regenerate(consciousness.constants.ambient_regen_rate);
            }
        }

        // Harmony resonance between nearby agents
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let ha = handles[i];
                let hb = handles[j];

                let (harm_a, harm_b) = {
                    let ea = consciousness.entities.get(&ha);
                    let eb = consciousness.entities.get(&hb);
                    match (ea, eb) {
                        (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                        _ => continue,
                    }
                };

                let resonance =
                    symtropy_consciousness_physics::harmony_field::HarmonyField::<3>::resonance(
                        &harm_a, &harm_b,
                    );
                if resonance > 0.5 {
                    let regen = consciousness.constants.harmony_resonance_regen_rate
                        * (resonance - 0.5)
                        * 2.0;
                    if let Some(entity) = consciousness.entities.get_mut(&ha) {
                        entity.energy.regenerate(regen);
                    }
                    if let Some(entity) = consciousness.entities.get_mut(&hb) {
                        entity.energy.regenerate(regen);
                    }
                }
            }
        }

        // Tick prediction error decay
        consciousness.tick_prediction_errors();

        // Physics step with consciousness callback
        world.step_with_callback(1.0 / 64.0, &mut consciousness);

        // Report every 50 ticks
        if tick % 50 == 0 {
            let alive = handles
                .iter()
                .filter(|h| {
                    consciousness
                        .entities
                        .get(h)
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false)
                })
                .count();
            let collapsed = handles.len() - alive;
            let avg_error: f64 = handles
                .iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.prediction_error))
                .sum::<f64>()
                / handles.len() as f64;
            let avg_energy: f64 = handles
                .iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
                .sum::<f64>()
                / handles.len() as f64;

            println!(
                "{:<6} {:<10.3} {:<10} {:<10} {:<10.3} {:<10.1}",
                tick, consciousness.collective_phi, alive, collapsed, avg_error, avg_energy,
            );
        }
    }

    println!("\n=== Final State ===");
    for (i, &h) in handles.iter().enumerate() {
        if let Some(entity) = consciousness.entities.get(&h) {
            let status = if entity.energy.is_collapsed() {
                "COLLAPSED"
            } else {
                "alive"
            };
            println!(
                "Agent {}: {} | energy={:.1} | phi={:.3} | pred_err={:.3} | precision={:.3}",
                i,
                status,
                entity.energy.available,
                entity.phi(),
                entity.prediction_error,
                entity.motor_precision,
            );
        }
    }
}
