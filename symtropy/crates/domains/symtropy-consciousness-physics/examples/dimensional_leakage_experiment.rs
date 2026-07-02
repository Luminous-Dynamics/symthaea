// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DIMENSIONAL LEAKAGE EXPERIMENT: What happens when energy bleeds
//! through the 4th dimension and 3D agents can't explain it?
//!
//! Compares:
//! A) NORMAL:  Standard 3D physics, no leakage
//! B) LEAKAGE: Energy sinks/sources from 4D — agents see conservation violations
//!
//! Measures:
//! - Prediction error (should spike near leakage points)
//! - Clustering patterns (do agents cluster differently near anomalies?)
//! - Survival (does leakage help or hurt?)
//! - "Dark energy" — apparent conservation violation magnitude

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::dimensional_leakage::DimensionalLeakage;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 16;
const TICKS: usize = 3000;
const SEEDS: usize = 15;

struct Result {
    clustering: f64,
    alive: usize,
    avg_energy: f64,
    avg_pred_error: f64,
    agents_near_source: f64, // fraction of alive agents near an energy source
    agents_near_sink: f64,   // fraction near a sink
    dark_energy: f64,        // apparent conservation violation
}

fn run(with_leakage: bool, seed: u64) -> Result {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants {
        initial_energy: 400.0,
        max_energy: 400.0,
        consciousness_maintenance_per_tick: 0.12,
        movement_cost_per_unit: 0.008,
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: 0.05,
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.25,
        ambient_regen_rate: 0.02,
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: 40.0,
    };

    // Dimensional leakage setup
    let mut leakage = DimensionalLeakage::new();
    if with_leakage {
        // A wormhole: energy drains at (-30, 0) and appears at (30, 0)
        // 3D agents see energy vanish on the left and appear on the right
        leakage.create_wormhole(
            SVector::from([-30.0, 0.0, 0.0]),
            SVector::from([30.0, 0.0, 0.0]),
            0.3,
            40.0,
        );
        // A pure sink: energy drains at (0, 40) into the bulk — never returns to 3D
        leakage.add_point(
            symtropy_consciousness_physics::dimensional_leakage::LeakagePoint::sink(
                SVector::from([0.0, 40.0, 0.0]),
                2.0,
                0.2,
                30.0,
            ),
        );
    }

    let mut rng = seed;
    let mut handles = Vec::new();
    for i in 0..AGENTS {
        let x = (nr(&mut rng) - 0.5) * 100.0;
        let y = (nr(&mut rng) - 0.5) * 100.0;
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

        // FEP gradient movement
        let ad: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position(),
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
                Some(b) => b.position(),
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

        // Maintenance + offloading
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

        // === DIMENSIONAL LEAKAGE ===
        // Energy drains/appears at leakage points. Agents don't know why.
        if with_leakage {
            for &h in &handles {
                if let Some(body) = world.body(h) {
                    let pos_2d = body.position();
                    let pos_3d = SVector::from([pos_2d[0], pos_2d[1], 0.0]); // project to 3D
                    let energy_effect = leakage.total_effect_at(&pos_3d);

                    if let Some(entity) = consciousness.entities.get_mut(&h) {
                        if energy_effect > 0.0 {
                            // Energy appearing from nowhere (source)
                            entity.energy.regenerate(energy_effect);
                            // This SHOULD spike prediction error — energy from nothing!
                            entity.prediction_error += energy_effect * 0.5;
                            entity.motor_precision = 1.0 / (1.0 + entity.prediction_error);
                        } else if energy_effect < 0.0 {
                            // Energy vanishing (sink)
                            entity.energy.consume(energy_effect.abs());
                            // Also spikes prediction error — energy disappeared!
                            entity.prediction_error += energy_effect.abs() * 0.5;
                            entity.motor_precision = 1.0 / (1.0 + entity.prediction_error);
                        }
                    }
                }
            }
            leakage.tick();
        }

        // Offloading
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let dist = match (world.body(ha), world.body(hb)) {
                    (Some(a), Some(b)) => a.position().metric_distance(&b.position()),
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

    // Measurements
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
        / AGENTS as f64;
    let avg_pe = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.prediction_error))
        .sum::<f64>()
        / AGENTS as f64;

    let mut tn = 0.0;
    for i in 0..handles.len() {
        let mut md = f64::MAX;
        for j in 0..handles.len() {
            if i == j {
                continue;
            }
            if let (Some(a), Some(b)) = (world.body(handles[i]), world.body(handles[j])) {
                let d = a.position().metric_distance(&b.position());
                if d < md {
                    md = d;
                }
            }
        }
        tn += md;
    }

    // Count agents near source/sink
    let near_source = handles
        .iter()
        .filter(|&&h| {
            world
                .body(h)
                .map(|b| {
                    let p = b.position();
                    leakage.points.iter().any(|lp| {
                        lp.flow_rate > 0.0 && lp.distance(&SVector::from([p[0], p[1], 0.0])) < 30.0
                    })
                })
                .unwrap_or(false)
        })
        .count() as f64
        / alive.max(1) as f64;

    let near_sink = handles
        .iter()
        .filter(|&&h| {
            world
                .body(h)
                .map(|b| {
                    let p = b.position();
                    leakage.points.iter().any(|lp| {
                        lp.flow_rate < 0.0 && lp.distance(&SVector::from([p[0], p[1], 0.0])) < 30.0
                    })
                })
                .unwrap_or(false)
        })
        .count() as f64
        / alive.max(1) as f64;

    Result {
        clustering: tn / AGENTS as f64,
        alive,
        avg_energy: ae,
        avg_pred_error: avg_pe,
        agents_near_source: near_source,
        agents_near_sink: near_sink,
        dark_energy: leakage.apparent_violation,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  DIMENSIONAL LEAKAGE: Energy bleeds through the 4th dimension  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    println!(
        "{} agents, {} ticks, {} seeds per condition\n",
        AGENTS, TICKS, SEEDS
    );

    println!(
        "{:<12} {:<10} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Condition", "Cluster", "Alive", "Energy", "PredErr", "NearSrc", "NearSink", "DarkE"
    );
    println!("{}", "─".repeat(80));

    for (name, leakage) in &[("NORMAL", false), ("LEAKAGE", true)] {
        let mut cs = Vec::new();
        let mut al = Vec::new();
        let mut en = Vec::new();
        let mut pe = Vec::new();
        let mut ns = Vec::new();
        let mut nk = Vec::new();
        let mut de = Vec::new();

        for s in 0..SEEDS {
            let r = run(*leakage, 42 + s as u64 * 997);
            cs.push(r.clustering);
            al.push(r.alive as f64);
            en.push(r.avg_energy);
            pe.push(r.avg_pred_error);
            ns.push(r.agents_near_source);
            nk.push(r.agents_near_sink);
            de.push(r.dark_energy);
        }

        let n = SEEDS as f64;
        println!(
            "{:<12} {:<10.2} {:<8.1} {:<10.1} {:<10.4} {:<10.1}% {:<10.1}% {:<10.1}",
            name,
            cs.iter().sum::<f64>() / n,
            al.iter().sum::<f64>() / n,
            en.iter().sum::<f64>() / n,
            pe.iter().sum::<f64>() / n,
            ns.iter().sum::<f64>() / n * 100.0,
            nk.iter().sum::<f64>() / n * 100.0,
            de.iter().sum::<f64>() / n,
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("NORMAL:  Standard 3D physics, no dimensional effects");
    println!("LEAKAGE: Energy sinks/sources from 4D bulk");
    println!("PredErr: Average prediction error (higher = more confused)");
    println!("NearSrc: % of alive agents near energy source (4D → 3D)");
    println!("NearSink: % near energy sink (3D → 4D)");
    println!("DarkE:   Cumulative energy that left 3D (the 'dark energy')");
    println!("═══════════════════════════════════════════════════════════════");
}

fn nr(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
