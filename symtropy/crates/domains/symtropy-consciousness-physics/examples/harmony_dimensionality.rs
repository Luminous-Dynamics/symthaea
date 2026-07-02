// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Harmony Dimensionality — does the number of harmony channels matter?
//!
//! We use 8 harmony channels but never tested whether this matters.
//! This experiment sweeps the effective dimensionality by zeroing out
//! channels beyond a threshold: 1, 2, 4, 8 active channels.
//!
//! With fewer channels, agents have fewer ways to be "compatible" —
//! resonance becomes more binary (match/mismatch). With more channels,
//! resonance is more graded (partial compatibility).
//!
//! Run: cargo run --example harmony_dimensionality --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 8_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;

const DIMENSIONALITIES: [usize; 4] = [1, 2, 4, 8];

// Full 8-channel profiles
const FULL_PROFILES: [[f64; 9]; 6] = [
    [0.8, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.7, 0.5],
    [0.3, 0.7, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
    [0.6, 0.2, 0.4, 0.3, 0.5, 0.4, 0.2, 0.5, 0.5],
    [0.3, 0.5, 0.2, 0.5, 0.3, 0.5, 0.5, 0.3, 0.5],
];

struct DimResult {
    dim: usize,
    alive: f64,
    clustering: f64,
    cooperation: f64,
    mean_resonance: f64,
    resonance_variance: f64,
}

fn truncate_harmony(full: &[f64; 9], active_dims: usize) -> [f64; 9] {
    let mut h = [0.0f64; 9];
    for i in 0..active_dims.min(9) {
        h[i] = full[i];
    }
    h
}

fn run_experiment(active_dims: usize, seed: u64) -> DimResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![2500.0f64; 2];

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 100.0;
        let y = (rng_f64(&mut rng) - 0.5) * 100.0;
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
            e.harmony_activations = truncate_harmony(&FULL_PROFILES[i % 6], active_dims);
        }
        handles.push(h);
    }

    let mut coop = 0u64;

    for _tick in 0..TICKS {
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
            .map(|(&p, &r)| (p, (r / 2500.0).min(1.0)))
            .collect();
        for &h in &handles {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position();
            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();
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

        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
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
                    let rg =
                        consciousness.constants.harmony_resonance_regen_rate * (res - 0.5) * 2.0;
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

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
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
        .count() as f64;
    let pos: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|&h| world.body(h).map(|b| b.position()))
        .collect();
    let clustering = if pos.len() >= 2 {
        pos.iter()
            .enumerate()
            .map(|(i, p)| {
                pos.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, q)| (p - q).norm())
                    .fold(f64::MAX, f64::min)
            })
            .sum::<f64>()
            / pos.len() as f64
    } else {
        f64::MAX
    };

    // Compute all pairwise resonances
    let harms: Vec<[f64; 9]> = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.harmony_activations))
        .collect();
    let mut resonances = Vec::new();
    for i in 0..harms.len() {
        for j in (i + 1)..harms.len() {
            resonances.push(HarmonyField::<2>::resonance(&harms[i], &harms[j]));
        }
    }
    let mean_res = if resonances.is_empty() {
        0.0
    } else {
        resonances.iter().sum::<f64>() / resonances.len() as f64
    };
    let var_res = if resonances.len() < 2 {
        0.0
    } else {
        resonances
            .iter()
            .map(|r| (r - mean_res).powi(2))
            .sum::<f64>()
            / (resonances.len() - 1) as f64
    };

    DimResult {
        dim: active_dims,
        alive,
        clustering: if clustering.is_finite() {
            clustering
        } else {
            0.0
        },
        cooperation: coop as f64,
        mean_resonance: mean_res,
        resonance_variance: var_res,
    }
}

fn main() {
    eprintln!("=== Harmony Dimensionality Experiment ===");
    eprintln!("Does the number of harmony channels matter?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    println!("dims,seed,alive,clustering,cooperation,mean_resonance,resonance_variance");

    let mut all: Vec<(usize, Vec<DimResult>)> = Vec::new();

    for &d in &DIMENSIONALITIES {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  dim={d} seed={seed}...");
            let r = run_experiment(d, seed);
            println!(
                "{d},{seed},{:.1},{:.2},{:.0},{:.4},{:.4}",
                r.alive, r.clustering, r.cooperation, r.mean_resonance, r.resonance_variance
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → dim={d}: alive={:.1}, mean_res={:.3}, var_res={:.4}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.mean_resonance).sum::<f64>() / n,
            results.iter().map(|r| r.resonance_variance).sum::<f64>() / n
        );
        all.push((d, results));
    }

    eprintln!("\n── Dimensionality Effects ──");
    eprintln!("  Dims  Alive  Cluster  Coop       MeanRes  VarRes");
    for (d, results) in &all {
        let n = results.len() as f64;
        eprintln!(
            "  {:3}   {:5.1}  {:6.2}   {:8.0}   {:.4}   {:.4}",
            d,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n,
            results.iter().map(|r| r.mean_resonance).sum::<f64>() / n,
            results.iter().map(|r| r.resonance_variance).sum::<f64>() / n
        );
    }

    // Key: does dimensionality affect survival?
    let d1 = &all[0].1; // 1 dim
    let d8 = &all[3].1; // 8 dims
    let a1: Vec<f64> = d1.iter().map(|r| r.alive).collect();
    let a8: Vec<f64> = d8.iter().map(|r| r.alive).collect();
    let d = cohens_d(&a1, &a8);
    eprintln!("\n── 1 dim vs 8 dims ──");
    eprintln!(
        "  Cohen's d = {d:.3} ({})",
        if d.abs() > 0.8 {
            "LARGE"
        } else if d.abs() > 0.5 {
            "medium"
        } else {
            "small"
        }
    );

    // Resonance variance tells us about selectivity
    let v1 = d1.iter().map(|r| r.resonance_variance).sum::<f64>() / d1.len() as f64;
    let v8 = d8.iter().map(|r| r.resonance_variance).sum::<f64>() / d8.len() as f64;
    eprintln!("  1-dim resonance variance: {v1:.4} (binary compatibility)");
    eprintln!("  8-dim resonance variance: {v8:.4} (graded compatibility)");
    if v8 > v1 * 1.5 {
        eprintln!("  More dimensions = more nuanced social differentiation");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
