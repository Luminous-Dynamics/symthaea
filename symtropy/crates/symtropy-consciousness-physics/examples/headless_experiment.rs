// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Headless experiment: full integration-metric-coupled physics simulation without rendering.
//!
//! Tests the core hypothesis: does strict energy conservation produce
//! emergent cooperation? Outputs CSV data for analysis.
//!
//! Run: cargo run -p symtropy-consciousness-physics --example headless_experiment
//!
//! Output: CSV to stdout with columns:
//!   tick, collective_phi, alive, collapsed, avg_energy, avg_pred_error,
//!   cooperation_events, joules_per_phi, conservation_error

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const NUM_AGENTS: usize = 20;
const TICKS: usize = 2000;
const DT: f64 = 1.0 / 64.0;

fn main() {
    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let enforce_thermodynamics = !args.contains(&"--no-thermo".to_string());
    let seed: u64 = args
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    eprintln!(
        "=== Headless Experiment: {} agents, {} ticks, thermo={}, seed={} ===",
        NUM_AGENTS, TICKS, enforce_thermodynamics, seed
    );

    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    // Tune for observable dynamics
    consciousness.constants = ThermodynamicConstants {
        initial_energy: 500.0,
        max_energy: 500.0,
        consciousness_maintenance_per_tick: if enforce_thermodynamics { 0.15 } else { 0.0 },
        movement_cost_per_unit: if enforce_thermodynamics { 0.01 } else { 0.0 },
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: if enforce_thermodynamics { 0.05 } else { 0.0 },
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.25,
        ambient_regen_rate: 0.03,
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: 30.0,
    };

    // Spawn agents in a cluster
    let mut rng = simple_rng(seed);
    let mut handles = Vec::new();

    for i in 0..NUM_AGENTS {
        let angle = (i as f64) * std::f64::consts::TAU / NUM_AGENTS as f64;
        let radius = 15.0 + rng_f64(&mut rng) * 10.0;
        let x = angle.cos() * radius;
        let y = angle.sin() * radius;

        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(body) = world.body_mut(h) {
            body.linear_damping = 0.3;
            // Random initial velocity toward center
            body.linear_velocity = SVector::from([-x * 0.2, -y * 0.2]);
        }

        consciousness.register(h, 500.0, 20.0);

        // Assign harmony profiles — 3 groups
        if let Some(entity) = consciousness.entities.get_mut(&h) {
            match i % 3 {
                0 => {
                    entity.harmony_activations = [0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5];
                    // Stillness group
                }
                1 => {
                    entity.harmony_activations = [0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.2, 0.5];
                    // Progression group
                }
                _ => {
                    entity.harmony_activations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
                    // Balanced (resonates with all)
                }
            }
        }

        handles.push(h);
    }

    // CSV header
    println!("tick,collective_phi,alive,collapsed,avg_energy,avg_pred_error,cooperation_events,joules_per_phi,conservation_error");

    let mut cooperation_events = 0u64;

    for tick in 0..TICKS {
        // Update consciousness
        for &h in &handles {
            let collapsed = consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.is_collapsed())
                .unwrap_or(true);

            let inputs = ConsciousnessInputs {
                phi: if collapsed { 0.0 } else { 0.5 },
                broadcast: 0.6,
                working_memory: 0.5,
                attention: 0.5,
                recurrence: 0.4,
                embodiment: 0.6,
                knowledge: 0.4,
                synchrony: 0.5,
            };
            consciousness.update_entity(h, &inputs, Point::origin());
        }

        // Consciousness maintenance + ambient regen
        let regen_mult = consciousness.resource_regeneration_multiplier();
        for &h in &handles {
            if let Some(entity) = consciousness.entities.get_mut(&h) {
                entity.energy.tick_reset();
                let phi = entity.phi();
                let maintenance =
                    consciousness.constants.consciousness_maintenance_per_tick * (1.0 + phi * 0.5);
                entity.energy.consume(maintenance);
                entity
                    .energy
                    .regenerate(consciousness.constants.ambient_regen_rate * regen_mult);
            }
        }

        // Harmony resonance
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let (harm_a, harm_b) = {
                    let ea = consciousness.entities.get(&ha);
                    let eb = consciousness.entities.get(&hb);
                    match (ea, eb) {
                        (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                        _ => continue,
                    }
                };

                let resonance = HarmonyField::<2>::resonance(&harm_a, &harm_b);
                if resonance > 0.5 {
                    let regen = consciousness.constants.harmony_resonance_regen_rate
                        * (resonance - 0.5)
                        * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.energy.regenerate(regen);
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.regenerate(regen);
                    }
                    cooperation_events += 1;

                    // Collapse recovery
                    if resonance > consciousness.constants.collapse_recovery_harmony_threshold {
                        if consciousness
                            .entities
                            .get(&ha)
                            .map(|e| e.energy.is_collapsed())
                            .unwrap_or(false)
                        {
                            if let Some(e) = consciousness.entities.get_mut(&ha) {
                                e.energy.regenerate(50.0);
                            }
                        }
                        if consciousness
                            .entities
                            .get(&hb)
                            .map(|e| e.energy.is_collapsed())
                            .unwrap_or(false)
                        {
                            if let Some(e) = consciousness.entities.get_mut(&hb) {
                                e.energy.regenerate(50.0);
                            }
                        }
                    }
                }
            }
        }

        // Prediction error decay
        consciousness.tick_prediction_errors();

        // Physics step with consciousness callback
        world.step_with_callback(DT, &mut consciousness);

        // Thermodynamic balance
        let balance = consciousness.tick_thermodynamics();

        // Output CSV row every 10 ticks
        if tick % 10 == 0 {
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
            let collapsed = NUM_AGENTS - alive;
            let avg_energy: f64 = handles
                .iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
                .sum::<f64>()
                / NUM_AGENTS as f64;
            let avg_pred_error: f64 = handles
                .iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.prediction_error))
                .sum::<f64>()
                / NUM_AGENTS as f64;
            let j_per_phi = balance
                .joules_per_phi
                .map(|j| format!("{:.4}", j))
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "{},{:.4},{},{},{:.2},{:.4},{},{},{}",
                tick,
                consciousness.collective_phi,
                alive,
                collapsed,
                avg_energy,
                avg_pred_error,
                cooperation_events,
                j_per_phi,
                format!("{:.6}", balance.conservation_error),
            );
        }
    }

    // Summary
    let final_alive = handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count();

    eprintln!("\n=== Results ===");
    eprintln!("Final alive: {}/{}", final_alive, NUM_AGENTS);
    eprintln!("Cooperation events: {}", cooperation_events);
    eprintln!(
        "Lifetime conservation error: {:.6}",
        consciousness.ledger.lifetime_error_rate()
    );

    if enforce_thermodynamics {
        eprintln!(
            "\nHypothesis test: With thermodynamic enforcement, {} of {} agents survived.",
            final_alive, NUM_AGENTS
        );
        if final_alive > 0 {
            eprintln!("RESULT: Cooperation sustained life under energy scarcity.");
        } else {
            eprintln!("RESULT: All collapsed. Constants may need tuning.");
        }
    }
}

// Simple deterministic RNG (no external deps)
fn simple_rng(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}
fn rng_f64(state: &mut u64) -> f64 {
    *state = simple_rng(*state);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}
