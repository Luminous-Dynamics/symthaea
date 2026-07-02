// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dunbar Number Experiment with thermodynamic enforcement.
//!
//! Sweeps population sizes under real energy scarcity to find:
//! - Finding 21: Optimal group size for survival
//! - Finding 22: Maximum cluster size (harmony field range limit)
//! - Finding 23: Per-capita energy scaling
//!
//! Run: cargo run --example dunbar_number -p symtropy-consciousness-physics --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::ThermodynamicConstants;
use symtropy_consciousness_physics::coupling::ConsciousnessField;
use symtropy_math::Point;
use symtropy_physics::{BodyHandle, PhysicsWorld};

const TICKS: u64 = 5000;
const DT: f64 = 1.0 / 60.0;
const SEEDS: u64 = 10;

struct TrialResult {
    clustering: f64,
    alive: usize,
    per_capita_energy: f64,
    survival_ticks: u64,
}

fn run_trial(num_agents: usize, seed: u64) -> TrialResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut field = ConsciousnessField::<2>::new();
    field.phi_gravity_strength = 0.3;
    field.constants = ThermodynamicConstants::research(); // Enable energy scarcity!

    // Place agents on ring with seeded offset
    let r = (num_agents as f64).sqrt() * 5.0;
    let mut handles = Vec::new();
    for i in 0..num_agents {
        let angle = std::f64::consts::TAU * (i as f64 + seed as f64 * 0.37) / num_agents as f64;
        let h = world.add_sphere(Point::new([r * angle.cos(), r * angle.sin()]), 1.0, 1.0);
        field.register(h, 200.0, 3.0);
        handles.push(h);
    }

    // Place 2 energy wells (fixed positions, scale with arena)
    let well_positions: Vec<Point<2>> =
        vec![Point::new([r * 0.5, 0.0]), Point::new([-r * 0.5, 0.0])];

    let inputs = ConsciousnessInputs {
        phi: 0.5,
        broadcast: 0.5,
        working_memory: 0.5,
        attention: 0.5,
        recurrence: 0.5,
        embodiment: 0.5,
        knowledge: 0.5,
        synchrony: 0.5,
    };

    let mut last_all_alive_tick: u64 = 0;

    for tick in 0..TICKS {
        // Update consciousness
        for &h in &handles {
            let pos = world
                .body(h)
                .map(|b| {
                    let p = b.position();
                    Point::new([p[0], p[1]])
                })
                .unwrap_or_else(Point::origin);
            field.update_entity(h, &inputs, pos);
        }

        // Simulate energy well regeneration for agents near wells
        for &h in &handles {
            if let Some(entity) = field.entities.get_mut(&h) {
                let agent_pos = world
                    .body(h)
                    .map(|b| {
                        let p = b.position();
                        [p[0], p[1]]
                    })
                    .unwrap_or([0.0, 0.0]);

                for well in &well_positions {
                    let dx = agent_pos[0] - well[0];
                    let dy = agent_pos[1] - well[1];
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < 15.0 {
                        // well radius
                        entity.energy.available = (entity.energy.available
                            + field.constants.energy_well_regen_rate)
                            .min(entity.energy.max_energy);
                    }
                }

                // Consciousness maintenance cost
                let phi = entity.phi();
                let cost = field.constants.consciousness_maintenance_per_tick * (1.0 + phi * 0.5);
                entity.energy.available = (entity.energy.available - cost).max(0.0);
            }
        }

        field.tick_prediction_errors();

        // Emotional contagion
        let positions: Vec<(BodyHandle, Point<2>)> = handles
            .iter()
            .map(|&h| {
                let pos = world
                    .body(h)
                    .map(|b| {
                        let p = b.position();
                        Point::new([p[0], p[1]])
                    })
                    .unwrap_or_else(Point::origin);
                (h, pos)
            })
            .collect();
        field.spread_emotional_contagion(&positions, DT);

        world.step_with_callback(DT, &mut field);

        let alive_now = handles
            .iter()
            .filter(|&&h| {
                field
                    .entities
                    .get(&h)
                    .map(|e| e.energy.available > 0.0)
                    .unwrap_or(false)
            })
            .count();
        if alive_now == num_agents {
            last_all_alive_tick = tick;
        }
    }

    // Final metrics
    let alive = handles
        .iter()
        .filter(|&&h| {
            field
                .entities
                .get(&h)
                .map(|e| e.energy.available > 0.0)
                .unwrap_or(false)
        })
        .count();
    let total_energy: f64 = handles
        .iter()
        .filter_map(|h| field.entities.get(h).map(|e| e.energy.available))
        .sum();

    // Clustering
    let n = handles.len();
    let mut total_dist = 0.0;
    let mut pairs = 0u64;
    let pos_vec: Vec<[f64; 2]> = handles
        .iter()
        .map(|&h| {
            world
                .body(h)
                .map(|b| {
                    let p = b.position();
                    [p[0], p[1]]
                })
                .unwrap_or([0.0, 0.0])
        })
        .collect();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pos_vec[i][0] - pos_vec[j][0];
            let dy = pos_vec[i][1] - pos_vec[j][1];
            total_dist += (dx * dx + dy * dy).sqrt();
            pairs += 1;
        }
    }
    let clustering = if pairs > 0 {
        total_dist / pairs as f64
    } else {
        0.0
    };

    TrialResult {
        clustering,
        alive,
        per_capita_energy: total_energy / num_agents as f64,
        survival_ticks: last_all_alive_tick,
    }
}

fn main() {
    eprintln!("Dunbar Number Experiment (thermodynamic enforcement)\n");
    eprintln!(
        "  {} ticks, {} seeds per group size, {} J/tick maintenance\n",
        TICKS,
        SEEDS,
        ThermodynamicConstants::research().consciousness_maintenance_per_tick
    );

    let group_sizes = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64];

    println!("N,clustering,clustering_std,alive_pct,per_capita_energy,survival_ticks");

    for &n in &group_sizes {
        let mut results: Vec<TrialResult> = Vec::new();
        for seed in 0..SEEDS {
            results.push(run_trial(n, seed));
        }

        let c_mean = results.iter().map(|r| r.clustering).sum::<f64>() / SEEDS as f64;
        let c_std = (results
            .iter()
            .map(|r| (r.clustering - c_mean).powi(2))
            .sum::<f64>()
            / (SEEDS.max(2) - 1) as f64)
            .sqrt();
        let alive_pct = results
            .iter()
            .map(|r| r.alive as f64 / n as f64 * 100.0)
            .sum::<f64>()
            / SEEDS as f64;
        let e_mean = results.iter().map(|r| r.per_capita_energy).sum::<f64>() / SEEDS as f64;
        let surv_mean = results.iter().map(|r| r.survival_ticks as f64).sum::<f64>() / SEEDS as f64;

        println!("{n},{c_mean:.2},{c_std:.2},{alive_pct:.1},{e_mean:.1},{surv_mean:.0}");
        eprintln!(
            "  N={n:>3}: cluster={c_mean:.1}, alive={alive_pct:.0}%, E/cap={e_mean:.1}, surv={surv_mean:.0}"
        );
    }
}
