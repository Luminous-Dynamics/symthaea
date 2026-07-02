// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Self-tuning experiment: the engine iteratively improves its own parameters.
//!
//! Runs multiple simulations with different ThermodynamicConstants,
//! measures survival rates and cooperation, and adjusts toward the
//! optimal balance where:
//! - Solo survival is possible but finite (~4 minutes)
//! - Cooperation extends survival indefinitely
//! - The game is neither trivial nor impossible
//!
//! This is not AGI. This is automated playtesting.
//!
//! Run: cargo run -p symtropy-consciousness-physics --example self_tuning

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 10;
const TICKS: usize = 1000; // ~15 seconds of gameplay
const RUNS_PER_CONFIG: usize = 3;

/// Results from a single simulation run.
#[derive(Debug, Clone)]
struct RunResult {
    alive_at_end: usize,
    cooperation_events: u64,
    avg_energy_at_end: f64,
    ticks_to_first_collapse: Option<usize>,
    collective_phi_final: f64,
}

/// Score a configuration: higher is better.
/// We want a sweet spot where cooperation matters.
fn score_config(results: &[RunResult]) -> f64 {
    let avg_alive =
        results.iter().map(|r| r.alive_at_end as f64).sum::<f64>() / results.len() as f64;
    let avg_coop = results
        .iter()
        .map(|r| r.cooperation_events as f64)
        .sum::<f64>()
        / results.len() as f64;
    let avg_first_collapse = results
        .iter()
        .filter_map(|r| r.ticks_to_first_collapse.map(|t| t as f64))
        .sum::<f64>()
        / results
            .iter()
            .filter(|r| r.ticks_to_first_collapse.is_some())
            .count()
            .max(1) as f64;

    // Ideal: some agents alive (not all, not none), cooperation happening, first collapse not too early/late
    let survival_score = if avg_alive < 1.0 {
        0.0 // All dead = bad config
    } else if avg_alive > (AGENTS as f64 - 1.0) {
        0.5 // All alive = too easy (cooperation not necessary)
    } else {
        1.0 // Some alive, some dead = interesting dynamics
    };

    let coop_score = (avg_coop / 1000.0).min(1.0); // More cooperation = better (up to saturation)

    let timing_score = if avg_first_collapse < 100.0 {
        0.3 // Too fast — players die before learning
    } else if avg_first_collapse > 800.0 {
        0.5 // Too slow — no pressure
    } else {
        1.0 // ~200-500 ticks = good tension
    };

    survival_score * 0.4 + coop_score * 0.3 + timing_score * 0.3
}

/// Run one simulation with given constants.
fn run_simulation(constants: &ThermodynamicConstants, seed: u64) -> RunResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = constants.clone();

    let mut handles = Vec::new();
    let mut rng_state = seed;

    for i in 0..AGENTS {
        let angle = (i as f64) * std::f64::consts::TAU / AGENTS as f64;
        let radius = 40.0 + next_rng(&mut rng_state) * 30.0; // Spread out — not all within harmony range
        let x = angle.cos() * radius;
        let y = angle.sin() * radius;

        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(body) = world.body_mut(h) {
            body.linear_damping = 0.3;
            body.linear_velocity = SVector::from([-x * 0.2, -y * 0.2]);
        }

        consciousness.register(h, constants.initial_energy, 20.0);

        if let Some(entity) = consciousness.entities.get_mut(&h) {
            match i % 3 {
                0 => entity.harmony_activations = [0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5],
                1 => entity.harmony_activations = [0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.2, 0.5],
                _ => entity.harmony_activations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            }
        }
        handles.push(h);
    }

    let mut cooperation_events = 0u64;
    let mut first_collapse: Option<usize> = None;

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

        // Maintenance + ambient regen
        let regen_mult = consciousness.resource_regeneration_multiplier();
        for &h in &handles {
            if let Some(entity) = consciousness.entities.get_mut(&h) {
                entity.energy.tick_reset();
                let phi = entity.phi();
                entity
                    .energy
                    .consume(constants.consciousness_maintenance_per_tick * (1.0 + phi * 0.5));
                entity
                    .energy
                    .regenerate(constants.ambient_regen_rate * regen_mult);
            }
        }

        // Epistemic offloading (thermodynamically honest — reduces costs, doesn't generate energy)
        // Only agents WITHIN RANGE benefit (not all pairs)
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);

                // Check spatial proximity via physics body positions
                let dist = {
                    let ba = world.body(ha);
                    let bb = world.body(hb);
                    match (ba, bb) {
                        (Some(a), Some(b)) => a.position().metric_distance(&b.position()),
                        _ => continue,
                    }
                };
                if dist > constants.harmony_range {
                    continue; // Too far apart — no epistemic offloading
                }

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
                    let offload_factor = (resonance - 0.5) * 2.0;
                    // Epistemic offloading: faster prediction error decay + maintenance refund
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.prediction_error *= 1.0 - offload_factor * 0.1;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        e.energy.regenerate(
                            constants.consciousness_maintenance_per_tick * offload_factor * 0.5,
                        );
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.prediction_error *= 1.0 - offload_factor * 0.1;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        e.energy.regenerate(
                            constants.consciousness_maintenance_per_tick * offload_factor * 0.5,
                        );
                    }
                    cooperation_events += 1;
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(1.0 / 64.0, &mut consciousness);
        consciousness.tick_thermodynamics();

        // Check for first collapse
        if first_collapse.is_none() {
            let any_collapsed = handles.iter().any(|h| {
                consciousness
                    .entities
                    .get(h)
                    .map(|e| e.energy.is_collapsed())
                    .unwrap_or(false)
            });
            if any_collapsed {
                first_collapse = Some(tick);
            }
        }
    }

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
    let avg_energy = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / AGENTS as f64;

    RunResult {
        alive_at_end: alive,
        cooperation_events,
        avg_energy_at_end: avg_energy,
        ticks_to_first_collapse: first_collapse,
        collective_phi_final: consciousness.collective_phi,
    }
}

fn main() {
    println!("=== Symtropy Self-Tuning Experiment ===\n");
    println!(
        "Testing {} configurations, {} runs each, {} agents, {} ticks\n",
        7, RUNS_PER_CONFIG, AGENTS, TICKS
    );

    // Test different maintenance costs (the most impactful parameter)
    let maintenance_values = [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35];

    println!(
        "{:<12} {:<8} {:<8} {:<10} {:<12} {:<8}",
        "Maintenance", "Alive", "Coop", "1st Death", "AvgEnergy", "Score"
    );
    println!("{}", "-".repeat(62));

    let mut best_score = 0.0f64;
    let mut best_maintenance = 0.0f64;

    for &maint in &maintenance_values {
        let mut constants = ThermodynamicConstants::default();
        constants.consciousness_maintenance_per_tick = maint;
        constants.initial_energy = 500.0;
        constants.max_energy = 500.0;

        let mut results = Vec::new();
        for run in 0..RUNS_PER_CONFIG {
            results.push(run_simulation(&constants, 42 + run as u64 * 1000));
        }

        let avg_alive =
            results.iter().map(|r| r.alive_at_end as f64).sum::<f64>() / results.len() as f64;
        let avg_coop = results
            .iter()
            .map(|r| r.cooperation_events as f64)
            .sum::<f64>()
            / results.len() as f64;
        let avg_first = results
            .iter()
            .filter_map(|r| r.ticks_to_first_collapse.map(|t| t as f64))
            .sum::<f64>()
            / results
                .iter()
                .filter(|r| r.ticks_to_first_collapse.is_some())
                .count()
                .max(1) as f64;
        let avg_energy =
            results.iter().map(|r| r.avg_energy_at_end).sum::<f64>() / results.len() as f64;
        let score = score_config(&results);

        println!(
            "{:<12.3} {:<8.1} {:<8.0} {:<10.0} {:<12.1} {:<8.3}",
            maint, avg_alive, avg_coop, avg_first, avg_energy, score
        );

        if score > best_score {
            best_score = score;
            best_maintenance = maint;
        }
    }

    println!("\n=== OPTIMAL CONFIGURATION ===");
    println!("Best maintenance cost: {:.3} J/tick", best_maintenance);
    println!("Score: {:.3}", best_score);
    println!(
        "\nRecommendation: set consciousness_maintenance_per_tick = {:.3}",
        best_maintenance
    );
}

fn next_rng(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}
