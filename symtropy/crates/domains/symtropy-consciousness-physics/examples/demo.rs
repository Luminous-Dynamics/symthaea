// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Symtropy Demo: Cooperation Emerges from Thermodynamic Necessity
//!
//! Run: `cargo run --example demo --release`
//!
//! This demonstrates the engine's core thesis in 10 seconds:
//! agents under Phi-coupled thermodynamic enforcement MUST cooperate
//! to survive, while agents without enforcement don't.
//!
//! Watch the numbers: enforced agents cluster, share energy via harmony
//! resonance, and survive. Free agents spread out, waste energy, and
//! the system drifts.

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::coupling::ConsciousnessField;
use symtropy_consciousness_physics::fep_gradient::free_energy_gradient_phi;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

fn main() {
    println!();
    println!("  Symtropy Engine Demo");
    println!("  Integrated information (Phi) as a first-class physics parameter");
    println!("  ---------------------------------------------------------------");
    println!();

    let results_enforced = run_simulation(true, 42);
    let results_free = run_simulation(false, 42);

    println!("  {:>20} {:>12} {:>12}", "", "ENFORCED", "FREE");
    println!("  {:>20} {:>12} {:>12}", "---", "---", "---");
    println!(
        "  {:>20} {:>12} {:>12}",
        "Agents alive",
        format!("{}/{}", results_enforced.alive, results_enforced.total),
        format!("{}/{}", results_free.alive, results_free.total)
    );
    println!(
        "  {:>20} {:>12.1} {:>12.1}",
        "Avg clustering", results_enforced.avg_clustering, results_free.avg_clustering
    );
    println!(
        "  {:>20} {:>12} {:>12}",
        "Cooperation ticks", results_enforced.coop_ticks, results_free.coop_ticks
    );
    println!(
        "  {:>20} {:>12.1} {:>12.1}",
        "Avg energy", results_enforced.avg_energy, results_free.avg_energy
    );
    println!(
        "  {:>20} {:>12.4} {:>12.4}",
        "Final collective Phi", results_enforced.final_phi, results_free.final_phi
    );

    if results_enforced.avg_clustering < results_free.avg_clustering {
        let pct = (1.0 - results_enforced.avg_clustering / results_free.avg_clustering) * 100.0;
        println!();
        println!("  Result: Enforced agents cluster {pct:.1}% more tightly.");
        println!("  Thesis confirmed: cooperation emerges from thermodynamic necessity.");
    }
    println!();
}

struct SimResult {
    total: usize,
    alive: usize,
    avg_clustering: f64,
    coop_ticks: usize,
    avg_energy: f64,
    final_phi: f64,
}

fn run_simulation(enforce: bool, seed: u64) -> SimResult {
    let n_agents = 12;
    let n_ticks = 3000;
    let mut rng_state = seed;

    // Create physics world (2D for simplicity)
    let mut world = PhysicsWorld::<2>::new(SVector::zeros());
    let mut field = ConsciousnessField::<2>::new();

    if enforce {
        // Research-grade constants: scarcity-driven dynamics
        // Solo agent ~1000 ticks. Cooperation extends survival to 5000+.
        field.constants =
            symtropy_consciousness_physics::thermodynamics::ThermodynamicConstants::research();
    } else {
        // Free mode: no energy costs, no enforcement
        let mut c =
            symtropy_consciousness_physics::thermodynamics::ThermodynamicConstants::research();
        c.consciousness_maintenance_per_tick = 0.0;
        c.movement_cost_per_unit = 0.0;
        c.collision_energy_drain = 0.0;
        c.initial_energy = 200.0;
        c.max_energy = 200.0;
        field.constants = c;
    }

    // Spawn agents in a circle
    let mut handles = Vec::new();
    for i in 0..n_agents {
        let angle = (i as f64 / n_agents as f64) * std::f64::consts::TAU;
        let x = angle.cos() * 30.0;
        let y = angle.sin() * 30.0;
        let h = world.add_sphere(Point::new([x, y]), 0.5, 1.0);
        field.register(h, field.constants.initial_energy, 15.0);

        // Random harmony activations
        if let Some(entity) = field.entities.get_mut(&h) {
            for j in 0..8 {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                entity.harmony_activations[j] = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            }
        }
        handles.push(h);
    }

    // Place energy wells
    let wells: Vec<SVector<f64, 2>> = vec![
        SVector::from([0.0, 0.0]),
        SVector::from([15.0, 15.0]),
        SVector::from([-15.0, -15.0]),
    ];

    let mut coop_ticks = 0;

    for tick in 0..n_ticks {
        // Compute consciousness and FEP gradient for each agent
        for &h in &handles {
            if !field.has_energy(h) {
                continue;
            }

            let phi = field.phi(h);
            let body = world.body(h).unwrap();
            let pos = body.position();
            let energy_frac = field
                .entities
                .get(&h)
                .map(|e| e.energy.fraction_remaining())
                .unwrap_or(0.0);
            let harmony = field
                .entities
                .get(&h)
                .map(|e| e.harmony_activations)
                .unwrap_or([0.0; 9]);

            // Nearby agents for FEP gradient
            let nearby: Vec<_> = handles
                .iter()
                .filter(|&&other| other != h && field.has_energy(other))
                .filter_map(|&other| {
                    let ob = world.body(other)?;
                    let op = SVector::from([ob.position()[0], ob.position()[1]]);
                    let oh = field.entities.get(&other)?.harmony_activations;
                    Some((op, oh))
                })
                .collect();

            let energy_wells: Vec<_> = wells.iter().map(|w| (*w, 200.0f64)).collect();

            let grad = free_energy_gradient_phi(
                &pos,
                energy_frac,
                Some(phi),
                &harmony,
                &nearby,
                &energy_wells,
                None,
                0.0,
            );

            // Apply movement
            if let Some(body) = world.body_mut(h) {
                body.linear_velocity = grad * 3.0;
            }

            // Consciousness inputs
            let inputs = ConsciousnessInputs {
                phi: 0.5 + energy_frac * 0.3,
                broadcast: 0.6,
                working_memory: 0.5,
                attention: 0.5,
                recurrence: 0.4,
                embodiment: 0.6,
                knowledge: 0.4,
                synchrony: 0.5,
            };
            let body = world.body(h).unwrap();
            let pos = Point(body.position());
            field.update_entity(h, &inputs, pos);
        }

        // Energy enforcement
        if enforce {
            for &h in &handles {
                field.consume_energy(
                    h,
                    field.constants.consciousness_maintenance_per_tick * (1.0 + 0.5 * field.phi(h)),
                );
            }

            // Harmony resonance regeneration
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    let (ha, hb) = (handles[i], handles[j]);
                    if !field.has_energy(ha) || !field.has_energy(hb) {
                        continue;
                    }
                    let (ba, bb) = (world.body(ha), world.body(hb));
                    if let (Some(ba), Some(bb)) = (ba, bb) {
                        let dist = ba.position().metric_distance(&bb.position());
                        if dist < field.constants.harmony_range {
                            let harm_a = field
                                .entities
                                .get(&ha)
                                .map(|e| e.harmony_activations)
                                .unwrap_or([0.0; 9]);
                            let harm_b = field
                                .entities
                                .get(&hb)
                                .map(|e| e.harmony_activations)
                                .unwrap_or([0.0; 9]);
                            let res = HarmonyField::<2>::resonance(&harm_a, &harm_b);
                            if res > 0.5 {
                                let regen = field.constants.harmony_resonance_regen_rate
                                    * (res - 0.5)
                                    * 2.0;
                                if let Some(ea) = field.entities.get_mut(&ha) {
                                    ea.energy.regenerate(regen);
                                }
                                if let Some(eb) = field.entities.get_mut(&hb) {
                                    eb.energy.regenerate(regen);
                                }
                                coop_ticks += 1;
                            }
                        }
                    }
                }
            }
        }

        world.step(0.016);

        // Progress output every 500 ticks
        if tick > 0 && tick % 500 == 0 {
            let alive = handles.iter().filter(|&&h| field.has_energy(h)).count();
            let tag = if enforce { "ENFORCED" } else { "FREE    " };
            print!(
                "  [{tag}] tick {tick:>5}: {alive:>2} alive, collective Phi = {:.4}\r",
                field.collective_phi
            );
        }
    }

    // Compute final metrics
    let alive_handles: Vec<_> = handles
        .iter()
        .filter(|&&h| field.has_energy(h))
        .cloned()
        .collect();
    let alive = alive_handles.len();

    let avg_clustering = if alive >= 2 {
        let mut total_dist = 0.0;
        let mut pairs = 0;
        for i in 0..alive_handles.len() {
            for j in (i + 1)..alive_handles.len() {
                if let (Some(a), Some(b)) =
                    (world.body(alive_handles[i]), world.body(alive_handles[j]))
                {
                    total_dist += a.position().metric_distance(&b.position());
                    pairs += 1;
                }
            }
        }
        if pairs > 0 {
            total_dist / pairs as f64
        } else {
            0.0
        }
    } else {
        0.0
    };

    let avg_energy = alive_handles
        .iter()
        .filter_map(|&h| field.entities.get(&h))
        .map(|e| e.energy.available)
        .sum::<f64>()
        / alive.max(1) as f64;

    println!();

    SimResult {
        total: n_agents,
        alive,
        avg_clustering,
        coop_ticks,
        avg_energy,
        final_phi: field.collective_phi,
    }
}
