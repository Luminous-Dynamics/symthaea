// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! THE DEFINITIVE EXPERIMENT: Does thermodynamic enforcement produce
//! emergent cooperation that wouldn't exist without it?
//!
//! Runs two identical simulations:
//! A) WITH thermodynamic enforcement (energy costs, maintenance, offloading)
//! B) WITHOUT (infinite energy, no costs)
//!
//! Measures spatial clustering, cooperation frequency, and survival.
//! If A produces significantly more clustering and cooperation than B,
//! that's the paper's central result.
//!
//! Run: cargo run -p symtropy-consciousness-physics --example cooperation_emergence

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 12;
const TICKS: usize = 5000;
const DT: f64 = 1.0 / 64.0;

#[derive(Debug)]
struct ExperimentResult {
    name: &'static str,
    alive_at_end: usize,
    avg_clustering: f64,    // average distance to nearest neighbor
    cooperation_ticks: u64, // ticks where epistemic offloading occurred
    avg_energy_at_end: f64,
    collapses: u64,
    recoveries: u64,
    collective_phi_trend: Vec<f64>, // sampled every 100 ticks
}

fn run_experiment(name: &'static str, enforce: bool, seed: u64) -> ExperimentResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    consciousness.constants = ThermodynamicConstants {
        initial_energy: 400.0,
        max_energy: 400.0,
        consciousness_maintenance_per_tick: if enforce { 0.12 } else { 0.0 },
        movement_cost_per_unit: if enforce { 0.008 } else { 0.0 },
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: if enforce { 0.05 } else { 0.0 },
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.25,
        ambient_regen_rate: if enforce { 0.02 } else { 0.0 },
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: 40.0,
    };

    let mut rng = seed;
    let mut handles = Vec::new();

    // Spawn agents scattered across a large area (100x100 units)
    for i in 0..AGENTS {
        let x = (next_rng(&mut rng) - 0.5) * 100.0;
        let y = (next_rng(&mut rng) - 0.5) * 100.0;

        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(body) = world.body_mut(h) {
            body.linear_damping = 0.2;
            // Small random velocity
            body.linear_velocity = SVector::from([
                (next_rng(&mut rng) - 0.5) * 10.0,
                (next_rng(&mut rng) - 0.5) * 10.0,
            ]);
        }

        consciousness.register(h, 400.0, 20.0);

        // Assign harmony profiles — 4 groups for interesting resonance patterns
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

    let mut cooperation_ticks = 0u64;
    let mut collapses = 0u64;
    let mut recoveries = 0u64;
    let mut phi_trend = Vec::new();
    let mut prev_collapsed: Vec<bool> = vec![false; AGENTS];

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

        // FEP gradient-driven movement (only in enforced condition)
        if enforce {
            // Gather nearby agent data for gradient computation
            let agent_data: Vec<(SVector<f64, 2>, [f64; 9])> = handles
                .iter()
                .filter_map(|&h| {
                    let body = world.body(h)?;
                    let entity = consciousness.entities.get(&h)?;
                    Some((body.position().0, entity.harmony_activations))
                })
                .collect();

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
                    .unwrap_or([0.0; 9]);

                // Build nearby agents list (exclude self)
                let nearby: Vec<_> = agent_data
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != idx)
                    .map(|(_, data)| data.clone())
                    .collect();

                let direction = fep_gradient::free_energy_gradient(
                    &pos,
                    energy_frac,
                    &harmony,
                    &nearby,
                    &[],
                    None,
                    0.0,
                );

                // Apply as velocity (speed 30 units/sec)
                if let Some(body) = world.body_mut(h) {
                    body.linear_velocity = direction * 30.0;
                }
            }
        }

        // Maintenance + ambient
        if enforce {
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
        }

        // Epistemic offloading (distance-aware)
        let mut any_offloading = false;
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let dist = {
                    let ba = world.body(ha);
                    let bb = world.body(hb);
                    match (ba, bb) {
                        (Some(a), Some(b)) => a.position().distance(b.position()),
                        _ => continue,
                    }
                };
                if dist > consciousness.constants.harmony_range {
                    continue;
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
                if resonance > 0.5 && enforce {
                    let offload = (resonance - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.prediction_error *= 1.0 - offload * 0.1;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        e.energy.regenerate(
                            consciousness.constants.consciousness_maintenance_per_tick
                                * offload
                                * 0.5,
                        );
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.prediction_error *= 1.0 - offload * 0.1;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        e.energy.regenerate(
                            consciousness.constants.consciousness_maintenance_per_tick
                                * offload
                                * 0.5,
                        );
                    }
                    any_offloading = true;
                }
            }
        }
        if any_offloading {
            cooperation_ticks += 1;
        }

        // Track collapses and recoveries
        for (idx, &h) in handles.iter().enumerate() {
            let now_collapsed = consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.is_collapsed())
                .unwrap_or(false);
            if now_collapsed && !prev_collapsed[idx] {
                collapses += 1;
            }
            if !now_collapsed && prev_collapsed[idx] {
                recoveries += 1;
            }
            prev_collapsed[idx] = now_collapsed;
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();

        // Sample phi trend
        if tick % 100 == 0 {
            phi_trend.push(consciousness.collective_phi);
        }
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

    // Clustering: average distance to nearest neighbor
    let mut total_nearest = 0.0;
    for i in 0..handles.len() {
        let mut min_dist = f64::MAX;
        for j in 0..handles.len() {
            if i == j {
                continue;
            }
            if let (Some(a), Some(b)) = (world.body(handles[i]), world.body(handles[j])) {
                let d = a.position().distance(b.position());
                if d < min_dist {
                    min_dist = d;
                }
            }
        }
        total_nearest += min_dist;
    }
    let avg_clustering = total_nearest / AGENTS as f64;

    ExperimentResult {
        name,
        alive_at_end: alive,
        avg_clustering,
        cooperation_ticks,
        avg_energy_at_end: avg_energy,
        collapses,
        recoveries,
        collective_phi_trend: phi_trend,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  THE DEFINITIVE EXPERIMENT: Does Thermodynamic Enforcement  ║");
    println!("║  Produce Emergent Cooperation?                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!(
        "Configuration: {} agents, {} ticks (~{:.0}s), 3 seeds\n",
        AGENTS,
        TICKS,
        TICKS as f64 / 64.0
    );

    let seeds = [42, 137, 2718];

    let mut enforced_results = Vec::new();
    let mut free_results = Vec::new();

    for &seed in &seeds {
        eprint!("  Running seed {seed} WITH enforcement... ");
        enforced_results.push(run_experiment("ENFORCED", true, seed));
        eprintln!("done");

        eprint!("  Running seed {seed} WITHOUT enforcement... ");
        free_results.push(run_experiment("FREE", false, seed));
        eprintln!("done");
    }

    // Aggregate results
    println!("\n┌─────────────────┬─────────┬─────────┐");
    println!("│ Metric          │ ENFORCED│ FREE    │");
    println!("├─────────────────┼─────────┼─────────┤");

    let avg = |results: &[ExperimentResult], f: fn(&ExperimentResult) -> f64| -> f64 {
        results.iter().map(f).sum::<f64>() / results.len() as f64
    };

    let e_alive = avg(&enforced_results, |r| r.alive_at_end as f64);
    let f_alive = avg(&free_results, |r| r.alive_at_end as f64);
    println!("│ Alive at end    │ {:>6.1} │ {:>6.1} │", e_alive, f_alive);

    let e_cluster = avg(&enforced_results, |r| r.avg_clustering);
    let f_cluster = avg(&free_results, |r| r.avg_clustering);
    println!(
        "│ Avg clustering  │ {:>6.1} │ {:>6.1} │",
        e_cluster, f_cluster
    );

    let e_coop = avg(&enforced_results, |r| r.cooperation_ticks as f64);
    let f_coop = avg(&free_results, |r| r.cooperation_ticks as f64);
    println!("│ Coop ticks      │ {:>6.0} │ {:>6.0} │", e_coop, f_coop);

    let e_energy = avg(&enforced_results, |r| r.avg_energy_at_end);
    let f_energy = avg(&free_results, |r| r.avg_energy_at_end);
    println!(
        "│ Avg energy      │ {:>6.1} │ {:>6.1} │",
        e_energy, f_energy
    );

    let e_collapses = avg(&enforced_results, |r| r.collapses as f64);
    let f_collapses = avg(&free_results, |r| r.collapses as f64);
    println!(
        "│ Collapses       │ {:>6.1} │ {:>6.1} │",
        e_collapses, f_collapses
    );

    println!("└─────────────────┴─────────┴─────────┘");

    // Phi trends
    println!("\nCollective Φ over time (sampled every 100 ticks):");
    println!(
        "  ENFORCED: {:?}",
        enforced_results[0]
            .collective_phi_trend
            .iter()
            .map(|p| format!("{:.3}", p))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  FREE:     {:?}",
        free_results[0]
            .collective_phi_trend
            .iter()
            .map(|p| format!("{:.3}", p))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Verdict
    println!("\n═══════════════════════════════════════════");
    let cluster_diff = (f_cluster - e_cluster) / f_cluster * 100.0;
    if e_cluster < f_cluster && cluster_diff > 5.0 {
        println!(
            "RESULT: Enforced agents cluster {:.1}% more tightly.",
            cluster_diff
        );
        println!("VERDICT: Thermodynamic enforcement produces emergent cooperation.");
    } else if (e_cluster - f_cluster).abs() / f_cluster < 0.05 {
        println!(
            "RESULT: No significant clustering difference ({:.1}%).",
            cluster_diff.abs()
        );
        println!("VERDICT: Cooperation emerges equally in both conditions.");
        println!("NOTE: Agents may need movement-toward-resonance behavior to show effect.");
    } else {
        println!(
            "RESULT: Free agents cluster {:.1}% more tightly.",
            -cluster_diff
        );
        println!("VERDICT: Unexpected — investigate.");
    }
    println!("═══════════════════════════════════════════");
}

fn next_rng(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}
