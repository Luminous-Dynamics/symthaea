// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scaling Benchmark — wall-clock performance from N=10 to N=1000.
//!
//! Compares brute-force O(n²) harmony loop vs spatial-hash O(n×k) version.
//! Reports per-tick wall-clock time at each population size.
//!
//! Run: cargo run --example scaling_benchmark --release

use nalgebra::SVector;
use std::time::Instant;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::spatial_hash::SpatialHash;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const TICKS: usize = 500;
const DT: f64 = 1.0 / 64.0;
const POPULATIONS: [usize; 7] = [10, 50, 100, 200, 500, 750, 1000];

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

fn run_benchmark(n_agents: usize, use_spatial_hash: bool) -> (f64, f64, usize) {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![
        SVector::from([50.0, 0.0]),
        SVector::from([-50.0, 0.0]),
        SVector::from([0.0, 50.0]),
        SVector::from([0.0, -50.0]),
    ];
    let mut well_remaining = vec![10000.0f64; 4]; // large wells for benchmark stability

    let mut rng = 42u64;
    let mut handles = Vec::new();

    for i in 0..n_agents {
        let x = (rng_f64(&mut rng) - 0.5) * 200.0;
        let y = (rng_f64(&mut rng) - 0.5) * 200.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.05;
        }
        consciousness.register(
            h,
            consciousness.constants.initial_energy,
            consciousness.constants.harmony_range,
        );
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.harmony_activations = HARMONY_PROFILES[i % 4];
        }
        handles.push(h);
    }

    let mut spatial = SpatialHash::<2>::new(consciousness.constants.harmony_range);
    let mut coop = 0u64;

    let start = Instant::now();

    for _tick in 0..TICKS {
        // Consciousness update
        for &h in &handles {
            let e = consciousness.entities.get(&h);
            let ef = e.map(|e| e.energy.fraction_remaining()).unwrap_or(0.0);
            let pe = e.map(|e| e.prediction_error).unwrap_or(0.0);
            let ht = e.map(|e| e.total_harmony_energy()).unwrap_or(0.0);
            let collapsed = e.map(|e| e.energy.is_collapsed()).unwrap_or(true);
            let inputs = if collapsed {
                ConsciousnessInputs {
                    phi: 0.0,
                    broadcast: 0.0,
                    working_memory: 0.0,
                    attention: 0.0,
                    recurrence: 0.0,
                    embodiment: 0.0,
                    knowledge: 0.0,
                    synchrony: 0.0,
                }
            } else {
                ConsciousnessInputs {
                    phi: ef,
                    broadcast: 0.5,
                    working_memory: (1.0 - pe).max(0.0),
                    attention: 0.5,
                    recurrence: 1.0,
                    embodiment: 0.7,
                    knowledge: (ht / 8.0).min(1.0),
                    synchrony: consciousness.collective_phi.max(0.5),
                }
            };
            consciousness.update_entity(h, &inputs, Point::origin());
        }

        // Build spatial hash
        if use_spatial_hash {
            spatial.clear();
            for (idx, &h) in handles.iter().enumerate() {
                if let Some(b) = world.body(h) {
                    spatial.insert(idx, &b.position());
                }
            }
        }

        // FEP gradient
        let adata: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position(),
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let wdata: Vec<_> = wells
            .iter()
            .zip(well_remaining.iter())
            .filter(|&(_, &r)| r > 0.0)
            .map(|(&p, &r)| (p, (r / 10000.0).min(1.0)))
            .collect();

        for (idx, &h) in handles.iter().enumerate() {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position();

            let near: Vec<_> = if use_spatial_hash {
                spatial
                    .query_neighbors(&pos)
                    .into_iter()
                    .filter(|&j| j != idx)
                    .filter_map(|j| {
                        let (p, h) = &adata[j];
                        let d = (p - pos).norm();
                        if d > 2.0 && d < consciousness.constants.harmony_range {
                            Some((p.clone(), h.clone()))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                adata
                    .iter()
                    .filter(|(p, _)| {
                        let d = (p - pos).norm();
                        d > 2.0 && d < consciousness.constants.harmony_range
                    })
                    .cloned()
                    .collect()
            };

            let dir = fep_gradient::free_energy_gradient(
                &pos,
                e.energy.fraction_remaining(),
                &e.harmony_activations,
                &near,
                &wdata,
                None,
                0.0,
            );
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = dir * 20.0;
            }
        }

        // Maintenance
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar);
                if let Some(b) = world.body(h) {
                    let pos = b.position();
                    for (wi, &w) in wells.iter().enumerate() {
                        if (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
                            let d = wr.min(well_remaining[wi]);
                            e.energy.regenerate(d);
                            well_remaining[wi] -= d;
                            break;
                        }
                    }
                }
            }
        }

        // Cooperation (with or without spatial hash)
        if use_spatial_hash {
            for (idx, &h) in handles.iter().enumerate() {
                let Some(e_a) = consciousness.entities.get(&h) else {
                    continue;
                };
                let harm_a = e_a.harmony_activations;
                let pos_a = match world.body(h) {
                    Some(b) => b.position(),
                    None => continue,
                };

                for &j in &spatial.query_neighbors(&pos_a) {
                    if j <= idx {
                        continue;
                    } // avoid double-counting
                    let hb = handles[j];
                    let in_range = match world.body(hb) {
                        Some(b) => {
                            (b.position() - pos_a).norm() < consciousness.constants.harmony_range
                        }
                        None => false,
                    };
                    if !in_range {
                        continue;
                    }
                    let harm_b = match consciousness.entities.get(&hb) {
                        Some(e) => e.harmony_activations,
                        None => continue,
                    };
                    let res = HarmonyField::<2>::resonance(&harm_a, &harm_b);
                    if res > 0.5 {
                        let rg = consciousness.constants.harmony_resonance_regen_rate
                            * (res - 0.5)
                            * 2.0;
                        if let Some(e) = consciousness.entities.get_mut(&h) {
                            e.energy.regenerate(rg);
                        }
                        if let Some(e) = consciousness.entities.get_mut(&hb) {
                            e.energy.regenerate(rg);
                        }
                        coop += 1;
                    }
                }
            }
        } else {
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    let (ha, hb) = (handles[i], handles[j]);
                    let in_range = match (world.body(ha), world.body(hb)) {
                        (Some(a), Some(b)) => {
                            a.position().metric_distance(&b.position())
                                < consciousness.constants.harmony_range
                        }
                        _ => false,
                    };
                    if !in_range {
                        continue;
                    }
                    let (a, b) = match (
                        consciousness.entities.get(&ha),
                        consciousness.entities.get(&hb),
                    ) {
                        (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                        _ => continue,
                    };
                    let res = HarmonyField::<2>::resonance(&a, &b);
                    if res > 0.5 {
                        let rg = consciousness.constants.harmony_resonance_regen_rate
                            * (res - 0.5)
                            * 2.0;
                        if let Some(e) = consciousness.entities.get_mut(&ha) {
                            e.energy.regenerate(rg);
                        }
                        if let Some(e) = consciousness.entities.get_mut(&hb) {
                            e.energy.regenerate(rg);
                        }
                        coop += 1;
                    }
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let per_tick_ms = total_ms / TICKS as f64;
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

    (total_ms, per_tick_ms, alive)
}

fn main() {
    eprintln!("=== Scaling Benchmark ===");
    eprintln!("{TICKS} ticks per run, release mode");
    eprintln!();

    println!("n,method,total_ms,per_tick_ms,alive");

    eprintln!("── Brute Force O(n²) ──");
    for &n in &POPULATIONS {
        if n > 500 {
            eprintln!("  N={n} skipped (too slow for brute force)");
            continue;
        }
        eprintln!("  N={n}...");
        let (total, per_tick, alive) = run_benchmark(n, false);
        println!("{n},brute,{total:.1},{per_tick:.3},{alive}");
        eprintln!("    {per_tick:.2} ms/tick ({total:.0} ms total), {alive} alive");
    }

    eprintln!("\n── Spatial Hash O(n×k) ──");
    for &n in &POPULATIONS {
        eprintln!("  N={n}...");
        let (total, per_tick, alive) = run_benchmark(n, true);
        println!("{n},spatial,{total:.1},{per_tick:.3},{alive}");
        eprintln!("    {per_tick:.2} ms/tick ({total:.0} ms total), {alive} alive");
    }

    eprintln!("\n── Speedup at N=500 ──");
    let (brute_500, _, _) = run_benchmark(500, false);
    let (spatial_500, _, _) = run_benchmark(500, true);
    eprintln!("  Brute: {brute_500:.0} ms, Spatial: {spatial_500:.0} ms");
    eprintln!("  Speedup: {:.1}×", brute_500 / spatial_500);

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
