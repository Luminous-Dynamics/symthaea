// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SCALING + PHASE DIAGRAM: Does cooperation scale? Where does it emerge?
//!
//! Part A: Scaling — N = {12, 24, 48, 96}, 20 seeds each
//! Part B: Phase diagram — energy_cost × agent_density, 10×10 grid, 10 seeds

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const TICKS: usize = 2000;

struct RunResult {
    clustering: f64,
    alive_frac: f64,
    avg_energy: f64,
}

fn run_sim(n_agents: usize, maint_cost: f64, area_size: f64, seed: u64) -> RunResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants {
        initial_energy: 400.0,
        max_energy: 400.0,
        consciousness_maintenance_per_tick: maint_cost,
        movement_cost_per_unit: 0.008,
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: 0.05,
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.25,
        ambient_regen_rate: 0.02,
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: 40.0,
    };

    let mut rng = seed;
    let mut handles = Vec::new();
    for i in 0..n_agents {
        let x = (nr(&mut rng) - 0.5) * area_size;
        let y = (nr(&mut rng) - 0.5) * area_size;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.2;
            b.linear_velocity =
                SVector::from([(nr(&mut rng) - 0.5) * 10.0, (nr(&mut rng) - 0.5) * 10.0]);
        }
        consciousness.register(h, 400.0, 20.0);
        if let Some(e) = consciousness.entities.get_mut(&h) {
            match i % 4 {
                0 => e.harmony_activations = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5],
                1 => e.harmony_activations = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.5],
                2 => e.harmony_activations = [0.1, 0.1, 0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.5],
                _ => e.harmony_activations = [0.4; 9],
            }
        }
        handles.push(h);
    }

    for _tick in 0..TICKS {
        for &h in &handles {
            let ef = consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.fraction_remaining())
                .unwrap_or(0.0);
            let inputs = ConsciousnessInputs {
                phi: if ef > 0.0 { 0.5 } else { 0.0 },
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

        // FEP gradient
        let ad: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position().0,
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        for (idx, &h) in handles.iter().enumerate() {
            if consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.is_collapsed())
                .unwrap_or(true)
            {
                continue;
            }
            let pos = match world.body(h) {
                Some(b) => b.position().0,
                None => continue,
            };
            let ef = consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.fraction_remaining())
                .unwrap_or(0.0);
            let harm = consciousness
                .entities
                .get(&h)
                .map(|e| e.harmony_activations)
                .unwrap_or([0.5; 9]);
            let nearby: Vec<_> = ad
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, d)| d.clone())
                .collect();
            let dir = fep_gradient::free_energy_gradient(&pos, ef, &harm, &nearby, &[], None, 0.0);
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = dir * 30.0;
            }
        }

        let rm = consciousness.resource_regeneration_multiplier();
        for &h in &handles {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
                let phi = e.phi();
                e.energy.consume(
                    consciousness.constants.consciousness_maintenance_per_tick * (1.0 + phi * 0.5),
                );
                e.energy
                    .regenerate(consciousness.constants.ambient_regen_rate * rm);
            }
        }
        // Offloading
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
                let (hah, hbh) = match (
                    consciousness.entities.get(&ha),
                    consciousness.entities.get(&hb),
                ) {
                    (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                    _ => continue,
                };
                let res = HarmonyField::<2>::resonance(&hah, &hbh);
                if res > 0.5 {
                    let off = (res - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.prediction_error *= 1.0 - off * 0.1;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        e.energy.regenerate(
                            consciousness.constants.consciousness_maintenance_per_tick * off * 0.5,
                        );
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.prediction_error *= 1.0 - off * 0.1;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        e.energy.regenerate(
                            consciousness.constants.consciousness_maintenance_per_tick * off * 0.5,
                        );
                    }
                }
            }
        }
        consciousness.tick_prediction_errors();
        world.step_with_callback(1.0 / 64.0, &mut consciousness);
        consciousness.tick_thermodynamics();
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
    let ae = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / n_agents as f64;
    let mut tn = 0.0;
    for i in 0..handles.len() {
        let mut md = f64::MAX;
        for j in 0..handles.len() {
            if i == j {
                continue;
            }
            if let (Some(a), Some(b)) = (world.body(handles[i]), world.body(handles[j])) {
                let d = a.position().distance(b.position());
                if d < md {
                    md = d;
                }
            }
        }
        tn += md;
    }
    RunResult {
        clustering: tn / n_agents as f64,
        alive_frac: alive as f64 / n_agents as f64,
        avg_energy: ae,
    }
}

fn main() {
    // ═══════════════════════════════════════
    // PART A: SCALING EXPERIMENT
    // ═══════════════════════════════════════
    println!("╔════════════════════════════════════════════════════╗");
    println!("║  PART A: SCALING — Does cooperation scale with N? ║");
    println!("╚════════════════════════════════════════════════════╝\n");

    let populations = [12, 24, 48, 96];
    let seeds_per = 20;

    println!(
        "{:<8} {:<12} {:<8} {:<12} {:<12}",
        "N", "Clustering", "±CI", "Alive%", "Energy"
    );
    println!("{}", "─".repeat(52));

    for &n in &populations {
        let area = (n as f64).sqrt() * 30.0; // scale area with population
        let mut clusterings = Vec::new();
        let mut alives = Vec::new();
        let mut energies = Vec::new();

        for s in 0..seeds_per {
            let r = run_sim(n, 0.12, area, 42 + s as u64 * 997);
            clusterings.push(r.clustering);
            alives.push(r.alive_frac);
            energies.push(r.avg_energy);
        }

        let mc = clusterings.iter().sum::<f64>() / seeds_per as f64;
        let sc = (clusterings.iter().map(|x| (x - mc).powi(2)).sum::<f64>()
            / (seeds_per as f64 - 1.0))
            .sqrt();
        let ci = 1.96 * sc / (seeds_per as f64).sqrt();
        let ma = alives.iter().sum::<f64>() / seeds_per as f64;
        let me = energies.iter().sum::<f64>() / seeds_per as f64;

        println!(
            "{:<8} {:<12.2} ±{:<6.2} {:<12.1}% {:<12.1}",
            n,
            mc,
            ci,
            ma * 100.0,
            me
        );
    }

    // ═══════════════════════════════════════
    // PART B: PHASE DIAGRAM
    // ═══════════════════════════════════════
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  PART B: PHASE DIAGRAM — energy_cost × density        ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    let costs = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.5];
    let densities = [6, 12, 24, 48]; // agents in 100x100 area
    let phase_seeds = 10;

    println!(
        "{:<10} {}",
        "Cost\\Dens",
        densities
            .iter()
            .map(|d| format!("{:<10}", d))
            .collect::<String>()
    );
    println!("{}", "─".repeat(50));

    for &cost in &costs {
        print!("{:<10.2} ", cost);
        for &n in &densities {
            let mut alive_sum = 0.0;
            for s in 0..phase_seeds {
                let r = run_sim(n, cost, 100.0, 42 + s as u64 * 31 + cost as u64 * 1000);
                alive_sum += r.alive_frac;
            }
            let avg_alive = alive_sum / phase_seeds as f64;
            // Phase: C=cooperate(survive), X=extinct, P=partial
            let phase = if avg_alive > 0.8 {
                "C"
            } else if avg_alive > 0.2 {
                "P"
            } else {
                "X"
            };
            print!("{:<4}{:<6.0}%  ", phase, avg_alive * 100.0);
        }
        println!();
    }

    println!("\nC = Cooperate (>80% survive), P = Partial (20-80%), X = Extinct (<20%)");
}

fn nr(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
