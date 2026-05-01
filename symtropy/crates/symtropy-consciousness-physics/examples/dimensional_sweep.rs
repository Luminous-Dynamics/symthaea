// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DIMENSIONAL SWEEP: How does physics change from 2D to 9D?
//!
//! Runs identical experiments at D = 2, 3, 4 (max supported by Bivector<D>).
//! For each dimension:
//! - Agents live on the D-1 "brane" (last coordinate = 0)
//! - Drifters slide along the Dth axis
//! - Measures prediction error, survival, clustering, dark energy
//!
//! Tests whether higher dimensions produce more leakage, more confusion,
//! or qualitatively different cooperation patterns.
//!
//! NOTE: Bivector<D> supports D ≤ 4 (6 components max).
//! For D > 4 we'd need to expand MAX_BIVECTOR_COMPONENTS.

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 10;
const DRIFTERS: usize = 3;
const TICKS: usize = 2000;
const SEEDS: usize = 10;

#[derive(Default)]
struct DimResult {
    pred_error: f64,
    max_pe: f64,
    alive: f64,
    energy: f64,
    dark_energy: f64,
    clustering: f64,
}

/// Run experiment at a specific dimension using trait dispatch.
/// Since const generics can't be runtime values, we dispatch manually.
fn run_at_dimension(dim: usize, with_drift: bool, seed: u64) -> DimResult {
    match dim {
        2 => run_dim::<2>(with_drift, seed),
        3 => run_dim::<3>(with_drift, seed),
        4 => run_dim::<4>(with_drift, seed),
        5 => run_dim::<5>(with_drift, seed),
        6 => run_dim::<6>(with_drift, seed),
        7 => run_dim::<7>(with_drift, seed),
        8 => run_dim::<8>(with_drift, seed),
        9 => run_dim::<9>(with_drift, seed),
        _ => {
            eprintln!("  D={} not supported (max D=9)", dim);
            DimResult::default()
        }
    }
}

fn run_dim<const D: usize>(with_drift: bool, seed: u64) -> DimResult {
    let gravity = SVector::<f64, D>::zeros(); // no gravity
    let mut world = PhysicsWorld::<D>::new(gravity);
    let mut consciousness = ConsciousnessField::<D>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let mut rng = seed;
    let mut agent_handles = Vec::new();
    let mut drifter_handles = Vec::new();

    // Spawn agents on the brane (last coordinate = 0)
    for i in 0..AGENTS {
        let mut coords = [0.0f64; D];
        // Spread in first 2 dimensions (even in higher D, agents live on a 2D surface)
        coords[0] = (nr(&mut rng) - 0.5) * 80.0;
        if D > 1 {
            coords[1] = (nr(&mut rng) - 0.5) * 80.0;
        }
        // All higher coords = 0 (on the brane)

        let h = world.add_sphere(Point::new(coords), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.05; // LTC: tau=20s, gentle air resistance
            let mut vel = SVector::<f64, D>::zeros();
            vel[0] = (nr(&mut rng) - 0.5) * 8.0;
            if D > 1 {
                vel[1] = (nr(&mut rng) - 0.5) * 8.0;
            }
            b.linear_velocity = vel;
        }

        consciousness.register(
            h,
            consciousness.constants.initial_energy,
            consciousness.constants.harmony_range,
        );
        if let Some(e) = consciousness.entities.get_mut(&h) {
            match i % 4 {
                0 => e.harmony_activations = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5],
                1 => e.harmony_activations = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.5],
                2 => e.harmony_activations = [0.1, 0.1, 0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.5],
                _ => e.harmony_activations = [0.4; 9],
            }
        }
        agent_handles.push(h);
    }

    // Spawn drifters that slide along the LAST axis (the "extra" dimension)
    for _ in 0..DRIFTERS {
        let mut coords = [0.0f64; D];
        coords[0] = (nr(&mut rng) - 0.5) * 40.0;
        if D > 1 {
            coords[1] = (nr(&mut rng) - 0.5) * 40.0;
        }

        let h = world.add_sphere(Point::new(coords), 2.0, 5.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.02; // very low damping — let them drift far
            if with_drift && D >= 2 {
                // Velocity along the LAST axis (the hidden dimension)
                let mut vel = SVector::<f64, D>::zeros();
                vel[D - 1] = (nr(&mut rng) + 0.2) * 3.0; // drift in last dimension
                b.linear_velocity = vel;
            }
        }
        drifter_handles.push(h);
    }

    let brane_threshold = 5.0; // visible if |last_coord| < 5
    let mut max_pe = 0.0f64;
    let mut prev_visible_ke = 0.0f64;

    for tick in 0..TICKS {
        // Consciousness
        for &h in &agent_handles {
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

        // FEP gradient (agents move in first 2 dims only — brane-locked)
        let ad: Vec<_> = agent_handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position().0,
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        for (idx, &h) in agent_handles.iter().enumerate() {
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
                let mut vel = dir * 25.0;
                // Lock agent to brane: zero out velocity in dimensions > 1
                for d in 2..D {
                    vel[d] = 0.0;
                }
                b.linear_velocity = vel;
            }
        }

        // Maintenance
        let rm = consciousness.resource_regeneration_multiplier();
        for &h in &agent_handles {
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

        // Detect leakage: visible KE changes
        let mut visible_ke = 0.0f64;
        let mut total_ke = 0.0f64;
        for &h in &drifter_handles {
            if let Some(body) = world.body(h) {
                let ke = body.kinetic_energy();
                total_ke += ke;
                if body.position().coord(D - 1).abs() < brane_threshold {
                    visible_ke += ke;
                }
            }
        }

        if tick > 10 && with_drift {
            let delta = (visible_ke - prev_visible_ke).abs();
            if delta > 0.5 {
                let err = (delta * 0.02).min(0.5);
                for &h in &agent_handles {
                    if let Some(e) = consciousness.entities.get_mut(&h) {
                        e.prediction_error += err;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        if e.prediction_error > max_pe {
                            max_pe = e.prediction_error;
                        }
                    }
                }
            }
        }
        prev_visible_ke = visible_ke;

        // Offloading
        for i in 0..agent_handles.len() {
            for j in (i + 1)..agent_handles.len() {
                let (ha, hb) = (agent_handles[i], agent_handles[j]);
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
                let res = HarmonyField::<D>::resonance(&hah, &hbh);
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

    // Final measurements
    let alive = agent_handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count();
    let ae = agent_handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / AGENTS as f64;
    let avg_pe = agent_handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.prediction_error))
        .sum::<f64>()
        / AGENTS as f64;

    let mut tn = 0.0;
    for i in 0..agent_handles.len() {
        let mut md = f64::MAX;
        for j in 0..agent_handles.len() {
            if i == j {
                continue;
            }
            if let (Some(a), Some(b)) = (world.body(agent_handles[i]), world.body(agent_handles[j]))
            {
                let d = a.position().distance(b.position());
                if d < md {
                    md = d;
                }
            }
        }
        tn += md;
    }

    let mut total_4d_ke = 0.0;
    let mut visible_3d_ke = 0.0;
    for &h in &drifter_handles {
        if let Some(body) = world.body(h) {
            total_4d_ke += body.kinetic_energy();
            if body.position().coord(D - 1).abs() < brane_threshold {
                visible_3d_ke += body.kinetic_energy();
            }
        }
    }

    DimResult {
        pred_error: avg_pe,
        max_pe,
        alive: alive as f64,
        energy: ae,
        dark_energy: total_4d_ke - visible_3d_ke,
        clustering: tn / AGENTS as f64,
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  DIMENSIONAL SWEEP: How does physics change from 2D to 9D?  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    println!(
        "{} agents + {} drifters, {} ticks, {} seeds per condition\n",
        AGENTS, DRIFTERS, TICKS, SEEDS
    );

    // Full dimensional sweep: 2D through 9D
    let dimensions = [2, 3, 4, 5, 6, 7, 8, 9];

    println!(
        "{:<6} {:<8} {:<10} {:<10} {:<8} {:<10} {:<10} {:<10}",
        "D", "Cond", "PredErr", "MaxPE", "Alive", "Energy", "Cluster", "DarkE"
    );
    println!("{}", "─".repeat(72));

    for &d in &dimensions {
        for (name, drift) in &[("STATIC", false), ("DRIFT", true)] {
            let mut pe = Vec::new();
            let mut mp = Vec::new();
            let mut al = Vec::new();
            let mut en = Vec::new();
            let mut cl = Vec::new();
            let mut de = Vec::new();

            for s in 0..SEEDS {
                let r = run_at_dimension(d, *drift, 42 + s as u64 * 997);
                pe.push(r.pred_error);
                mp.push(r.max_pe);
                al.push(r.alive);
                en.push(r.energy);
                cl.push(r.clustering);
                de.push(r.dark_energy);
            }

            let n = SEEDS as f64;
            println!(
                "{:<6} {:<8} {:<10.4} {:<10.3} {:<8.1} {:<10.1} {:<10.2} {:<10.3}",
                format!("{}D", d),
                name,
                pe.iter().sum::<f64>() / n,
                mp.iter().sum::<f64>() / n,
                al.iter().sum::<f64>() / n,
                en.iter().sum::<f64>() / n,
                cl.iter().sum::<f64>() / n,
                de.iter().sum::<f64>() / n,
            );
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("Agents live on first 2 axes. Drifters slide along LAST axis.");
    println!("MaxPE > 0 means agents detected the leakage.");
    println!("DarkE = energy in bulk invisible to brane agents.");
    println!("═══════════════════════════════════════════════════════════");
}

fn nr(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
