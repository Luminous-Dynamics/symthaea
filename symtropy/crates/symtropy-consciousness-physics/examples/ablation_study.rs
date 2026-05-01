// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ABLATION STUDY: Which ingredients produce cooperation?
//!
//! Tests 6 conditions to disentangle causal factors:
//! 1. FREE:        No costs, random drift (control)
//! 2. ENERGY_ONLY: Energy costs but no offloading, no gradient
//! 3. E+OFFLOAD:   Energy costs + epistemic offloading, random movement
//! 4. E+GRADIENT:  Energy costs + FEP gradient, no offloading
//! 5. E+OFF+RAND:  Energy + offloading + random movement (no gradient)
//! 6. FULL:        Energy + offloading + gradient (complete model)
//!
//! This answers: is clustering due to the gradient? The offloading? Both?
//!
//! Also measures multiple cooperation metrics beyond clustering:
//! - Nearest-neighbor distance (clustering)
//! - Aggregate survival time (do groups live longer?)
//! - Collapse rate (do clustered agents collapse less?)
//! - Shared well access (do groups coordinate around resources?)

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 12;
const TICKS: usize = 5000;
const SEEDS: &[u64] = &[42, 137, 2718, 31415, 27182];

#[derive(Debug, Clone)]
struct Condition {
    name: &'static str,
    energy_costs: bool,
    offloading: bool,
    fep_gradient: bool,
}

#[derive(Debug, Clone)]
struct Result {
    avg_clustering: f64,
    avg_survival_ticks: f64, // average ticks alive per agent
    collapse_rate: f64,      // fraction of agents that collapsed
    avg_energy_at_end: f64,
    cooperation_ticks: u64,
    well_sharing_events: u64, // ticks where 2+ agents share a well
}

fn run(cond: &Condition, seed: u64) -> Result {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    consciousness.constants = ThermodynamicConstants {
        initial_energy: 400.0,
        max_energy: 400.0,
        consciousness_maintenance_per_tick: if cond.energy_costs { 0.12 } else { 0.0 },
        movement_cost_per_unit: if cond.energy_costs { 0.008 } else { 0.0 },
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: if cond.energy_costs { 0.05 } else { 0.0 },
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.25,
        ambient_regen_rate: if cond.energy_costs { 0.02 } else { 0.0 },
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: 40.0,
    };

    let mut rng = seed;
    let mut handles = Vec::new();

    // Place 2 energy wells at fixed positions
    let well_positions = vec![SVector::from([30.0, 30.0]), SVector::from([-30.0, -30.0])];

    for i in 0..AGENTS {
        let x = (next_rng(&mut rng) - 0.5) * 100.0;
        let y = (next_rng(&mut rng) - 0.5) * 100.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(body) = world.body_mut(h) {
            body.linear_damping = 0.2;
            body.linear_velocity = SVector::from([
                (next_rng(&mut rng) - 0.5) * 10.0,
                (next_rng(&mut rng) - 0.5) * 10.0,
            ]);
        }
        consciousness.register(h, 400.0, 20.0);
        if let Some(entity) = consciousness.entities.get_mut(&h) {
            match i % 4 {
                0 => entity.harmony_activations = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5],
                1 => entity.harmony_activations = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.5],
                2 => entity.harmony_activations = [0.1, 0.1, 0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.5],
                _ => entity.harmony_activations = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5],
            }
        }
        handles.push(h);
    }

    let mut coop_ticks = 0u64;
    let mut well_sharing = 0u64;
    let mut alive_ticks: Vec<u64> = vec![0; AGENTS]; // ticks each agent was alive

    for tick in 0..TICKS {
        // Consciousness update
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

        // Track alive ticks
        for (idx, &h) in handles.iter().enumerate() {
            if !consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.is_collapsed())
                .unwrap_or(true)
            {
                alive_ticks[idx] += 1;
            }
        }

        // FEP gradient movement (if enabled)
        if cond.fep_gradient {
            let agent_data: Vec<_> = handles
                .iter()
                .filter_map(|&h| {
                    let body = world.body(h)?;
                    let entity = consciousness.entities.get(&h)?;
                    Some((body.position().0, entity.harmony_activations))
                })
                .collect();

            let well_data: Vec<_> = well_positions.iter().map(|wp| (*wp, 1.0f64)).collect();

            for (idx, &h) in handles.iter().enumerate() {
                let collapsed = consciousness
                    .entities
                    .get(&h)
                    .map(|e| e.energy.is_collapsed())
                    .unwrap_or(true);
                if collapsed {
                    continue;
                }

                let pos = match world.body(h) {
                    Some(b) => b.position().0,
                    None => continue,
                };
                let energy_frac = consciousness
                    .entities
                    .get(&h)
                    .map(|e| e.energy.fraction_remaining())
                    .unwrap_or(0.0);
                let harmony = consciousness
                    .entities
                    .get(&h)
                    .map(|e| e.harmony_activations)
                    .unwrap_or([0.5; 9]);

                let nearby: Vec<_> = agent_data
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != idx)
                    .map(|(_, d)| d.clone())
                    .collect();

                let dir = fep_gradient::free_energy_gradient(
                    &pos,
                    energy_frac,
                    &harmony,
                    &nearby,
                    &well_data,
                    None,
                    0.0,
                );
                if let Some(body) = world.body_mut(h) {
                    body.linear_velocity = dir * 30.0;
                }
            }
        }

        // Maintenance + ambient
        if cond.energy_costs {
            let regen_mult = consciousness.resource_regeneration_multiplier();
            for &h in &handles {
                if let Some(entity) = consciousness.entities.get_mut(&h) {
                    entity.energy.tick_reset();
                    let phi = entity.phi();
                    entity.energy.consume(
                        consciousness.constants.consciousness_maintenance_per_tick
                            * (1.0 + phi * 0.5),
                    );
                    entity
                        .energy
                        .regenerate(consciousness.constants.ambient_regen_rate * regen_mult);
                }
            }

            // Energy well regeneration
            for &h in &handles {
                let pos = match world.body(h) {
                    Some(b) => b.position().0,
                    None => continue,
                };
                for wp in &well_positions {
                    if (pos - wp).norm() < 40.0 {
                        if let Some(entity) = consciousness.entities.get_mut(&h) {
                            entity
                                .energy
                                .regenerate(consciousness.constants.energy_well_regen_rate);
                        }
                    }
                }
            }

            // Check well sharing
            for wp in &well_positions {
                let agents_at_well = handles
                    .iter()
                    .filter(|&&h| {
                        world
                            .body(h)
                            .map(|b| (b.position().0 - wp).norm() < 40.0)
                            .unwrap_or(false)
                    })
                    .count();
                if agents_at_well >= 2 {
                    well_sharing += 1;
                }
            }
        }

        // Epistemic offloading (if enabled)
        if cond.offloading && cond.energy_costs {
            let mut any = false;
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    let (ha, hb) = (handles[i], handles[j]);
                    let dist = match (world.body(ha), world.body(hb)) {
                        (Some(a), Some(b)) => a.position().distance(b.position()),
                        _ => continue,
                    };
                    if dist > consciousness.constants.harmony_range {
                        continue;
                    }
                    let (harm_a, harm_b) = match (
                        consciousness.entities.get(&ha),
                        consciousness.entities.get(&hb),
                    ) {
                        (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                        _ => continue,
                    };
                    let res = HarmonyField::<2>::resonance(&harm_a, &harm_b);
                    if res > 0.5 {
                        let off = (res - 0.5) * 2.0;
                        if let Some(e) = consciousness.entities.get_mut(&ha) {
                            e.prediction_error *= 1.0 - off * 0.1;
                            e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                            e.energy.regenerate(
                                consciousness.constants.consciousness_maintenance_per_tick
                                    * off
                                    * 0.5,
                            );
                        }
                        if let Some(e) = consciousness.entities.get_mut(&hb) {
                            e.prediction_error *= 1.0 - off * 0.1;
                            e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                            e.energy.regenerate(
                                consciousness.constants.consciousness_maintenance_per_tick
                                    * off
                                    * 0.5,
                            );
                        }
                        any = true;
                    }
                }
            }
            if any {
                coop_ticks += 1;
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(1.0 / 64.0, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    // Final measurements
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

    let mut total_nearest = 0.0;
    for i in 0..handles.len() {
        let mut min_d = f64::MAX;
        for j in 0..handles.len() {
            if i == j {
                continue;
            }
            if let (Some(a), Some(b)) = (world.body(handles[i]), world.body(handles[j])) {
                let d = a.position().distance(b.position());
                if d < min_d {
                    min_d = d;
                }
            }
        }
        total_nearest += min_d;
    }

    let avg_alive_ticks = alive_ticks.iter().sum::<u64>() as f64 / AGENTS as f64;

    Result {
        avg_clustering: total_nearest / AGENTS as f64,
        avg_survival_ticks: avg_alive_ticks,
        collapse_rate: (AGENTS - alive) as f64 / AGENTS as f64,
        avg_energy_at_end: avg_energy,
        cooperation_ticks: coop_ticks,
        well_sharing_events: well_sharing,
    }
}

fn main() {
    let conditions = vec![
        Condition {
            name: "FREE        ",
            energy_costs: false,
            offloading: false,
            fep_gradient: false,
        },
        Condition {
            name: "ENERGY_ONLY ",
            energy_costs: true,
            offloading: false,
            fep_gradient: false,
        },
        Condition {
            name: "E+OFFLOAD   ",
            energy_costs: true,
            offloading: true,
            fep_gradient: false,
        },
        Condition {
            name: "E+GRADIENT  ",
            energy_costs: true,
            offloading: false,
            fep_gradient: true,
        },
        Condition {
            name: "E+OFF+RAND  ",
            energy_costs: true,
            offloading: true,
            fep_gradient: false,
        },
        Condition {
            name: "FULL        ",
            energy_costs: true,
            offloading: true,
            fep_gradient: true,
        },
    ];

    println!(
        "╔═══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                        ABLATION STUDY: What Causes Cooperation?                      ║"
    );
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!(
        "\n{} agents, {} ticks, {} seeds, 2 energy wells\n",
        AGENTS,
        TICKS,
        SEEDS.len()
    );

    println!(
        "{:<14} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Condition", "Cluster", "Survive", "Collapse%", "Energy", "CoopTick", "WellShare"
    );
    println!("{}", "─".repeat(74));

    for cond in &conditions {
        let mut results: Vec<Result> = Vec::new();
        for &seed in SEEDS {
            results.push(run(cond, seed));
        }

        let n = results.len() as f64;
        let avg_cluster = results.iter().map(|r| r.avg_clustering).sum::<f64>() / n;
        let avg_survive = results.iter().map(|r| r.avg_survival_ticks).sum::<f64>() / n;
        let avg_collapse = results.iter().map(|r| r.collapse_rate).sum::<f64>() / n;
        let avg_energy = results.iter().map(|r| r.avg_energy_at_end).sum::<f64>() / n;
        let avg_coop = results
            .iter()
            .map(|r| r.cooperation_ticks as f64)
            .sum::<f64>()
            / n;
        let avg_well = results
            .iter()
            .map(|r| r.well_sharing_events as f64)
            .sum::<f64>()
            / n;

        println!(
            "{:<14} {:<10.1} {:<10.0} {:<10.1}% {:<10.1} {:<10.0} {:<10.0}",
            cond.name,
            avg_cluster,
            avg_survive,
            avg_collapse * 100.0,
            avg_energy,
            avg_coop,
            avg_well
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("KEY: Cluster = avg nearest-neighbor distance (lower = tighter)");
    println!("     Survive = avg ticks alive per agent (max {})", TICKS);
    println!("     Collapse% = fraction that ran out of energy");
    println!("     CoopTick = ticks with active epistemic offloading");
    println!("     WellShare = ticks with 2+ agents at same well");
    println!("═══════════════════════════════════════════════════════════════════");
}

fn next_rng(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}
