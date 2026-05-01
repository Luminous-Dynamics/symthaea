// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D Cooperation — does cooperation emerge in three-dimensional space?
//!
//! Validates the engine's N-dimensional claim by running the core cooperation
//! experiment in 3D. Compares survival, clustering, and cooperation rates
//! between 2D and 3D to determine whether the extra dimension changes the
//! emergent dynamics.
//!
//! Hypothesis: 3D should produce LESS cooperation (agents spread across more
//! volume, harder to find each other) but the FEP gradient should still
//! produce clustering.
//!
//! Run: cargo run --example cooperation_3d --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 8_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 8;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

struct CoopResult {
    dim: usize,
    alive: f64,
    energy: f64,
    clustering: f64,
    cooperation: f64,
}

fn run_2d(seed: u64) -> CoopResult {
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
            e.harmony_activations = HARMONY_PROFILES[i % 4];
        }
        handles.push(h);
    }

    let mut cooperation = 0u64;
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
                    world.body(h)?.position().0,
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let wdata: Vec<_> = wells
            .iter()
            .zip(well_remaining.iter())
            .filter(|(_, &r)| r > 0.0)
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
            let pos = b.position().0;
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
        for (idx, &h) in handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                let pos = world.body(h).map(|b| b.position().0);
                if let Some(pos) = pos {
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
                        a.position().distance(b.position()) < consciousness.constants.harmony_range
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
                    cooperation += 1;
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
    let energy = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / AGENTS as f64;
    let pos: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|h| world.body(*h).map(|b| b.position().0))
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

    CoopResult {
        dim: 2,
        alive,
        energy,
        clustering,
        cooperation: cooperation as f64,
    }
}

fn run_3d(seed: u64) -> CoopResult {
    let mut world = PhysicsWorld::<3>::new(SVector::from([0.0, 0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<3>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![
        SVector::from([30.0, 0.0, 0.0]),
        SVector::from([-30.0, 0.0, 0.0]),
    ];
    let mut well_remaining = vec![2500.0f64; 2];

    let mut rng = seed;
    let mut handles = Vec::new();
    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 100.0;
        let y = (rng_f64(&mut rng) - 0.5) * 100.0;
        let z = (rng_f64(&mut rng) - 0.5) * 100.0;
        let h = world.add_sphere(Point::new([x, y, z]), 1.0, 1.0);
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

    let mut cooperation = 0u64;
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
                    world.body(h)?.position().0,
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let wdata: Vec<_> = wells
            .iter()
            .zip(well_remaining.iter())
            .filter(|(_, &r)| r > 0.0)
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
            let pos = b.position().0;
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
                let pos = world.body(h).map(|b| b.position().0);
                if let Some(pos) = pos {
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
                        a.position().distance(b.position()) < consciousness.constants.harmony_range
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
                let res = HarmonyField::<3>::resonance(&a, &b);
                if res > 0.5 {
                    let rg =
                        consciousness.constants.harmony_resonance_regen_rate * (res - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.energy.regenerate(rg);
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.regenerate(rg);
                    }
                    cooperation += 1;
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
    let energy = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / AGENTS as f64;
    let pos: Vec<SVector<f64, 3>> = handles
        .iter()
        .filter_map(|h| world.body(*h).map(|b| b.position().0))
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

    CoopResult {
        dim: 3,
        alive,
        energy,
        clustering,
        cooperation: cooperation as f64,
    }
}

fn main() {
    eprintln!("=== 3D Cooperation Experiment ===");
    eprintln!("Does cooperation emerge in 3D as well as 2D?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    println!("dim,seed,alive,energy,clustering,cooperation");

    let mut results_2d = Vec::new();
    let mut results_3d = Vec::new();

    for s in 0..SEEDS {
        let seed = 42 + s as u64 * 997;
        eprintln!("  2D seed={seed}...");
        let r = run_2d(seed);
        println!(
            "2,{seed},{:.1},{:.1},{:.2},{:.0}",
            r.alive, r.energy, r.clustering, r.cooperation
        );
        results_2d.push(r);
    }
    for s in 0..SEEDS {
        let seed = 42 + s as u64 * 997;
        eprintln!("  3D seed={seed}...");
        let r = run_3d(seed);
        println!(
            "3,{seed},{:.1},{:.1},{:.2},{:.0}",
            r.alive, r.energy, r.clustering, r.cooperation
        );
        results_3d.push(r);
    }

    let n = SEEDS as f64;
    eprintln!("\n── 2D ──");
    eprintln!(
        "  Alive:      {:.1}/{AGENTS}",
        results_2d.iter().map(|r| r.alive).sum::<f64>() / n
    );
    eprintln!(
        "  Energy:     {:.1} J",
        results_2d.iter().map(|r| r.energy).sum::<f64>() / n
    );
    eprintln!(
        "  Clustering: {:.2}",
        results_2d.iter().map(|r| r.clustering).sum::<f64>() / n
    );
    eprintln!(
        "  Cooperation: {:.0}",
        results_2d.iter().map(|r| r.cooperation).sum::<f64>() / n
    );

    eprintln!("\n── 3D ──");
    eprintln!(
        "  Alive:      {:.1}/{AGENTS}",
        results_3d.iter().map(|r| r.alive).sum::<f64>() / n
    );
    eprintln!(
        "  Energy:     {:.1} J",
        results_3d.iter().map(|r| r.energy).sum::<f64>() / n
    );
    eprintln!(
        "  Clustering: {:.2}",
        results_3d.iter().map(|r| r.clustering).sum::<f64>() / n
    );
    eprintln!(
        "  Cooperation: {:.0}",
        results_3d.iter().map(|r| r.cooperation).sum::<f64>() / n
    );

    // Statistical comparison
    let a2: Vec<f64> = results_2d.iter().map(|r| r.alive).collect();
    let a3: Vec<f64> = results_3d.iter().map(|r| r.alive).collect();
    let (u, z, p) = mann_whitney_u(&a2, &a3);
    let d = cohens_d(&a2, &a3);
    eprintln!("\n── 2D vs 3D Survival ──");
    eprintln!("  Mann-Whitney U={u:.1}, z={z:.3}, p={p:.4}");
    eprintln!(
        "  Cohen's d={d:.3} ({})",
        if d.abs() > 0.8 {
            "large"
        } else if d.abs() > 0.5 {
            "medium"
        } else {
            "small"
        }
    );

    let c2: Vec<f64> = results_2d.iter().map(|r| r.cooperation).collect();
    let c3: Vec<f64> = results_3d.iter().map(|r| r.cooperation).collect();
    let d2 = cohens_d(&c2, &c3);
    eprintln!("\n── 2D vs 3D Cooperation ──");
    eprintln!(
        "  Cohen's d={d2:.3} ({})",
        if d2.abs() > 0.8 {
            "large"
        } else if d2.abs() > 0.5 {
            "medium"
        } else {
            "small"
        }
    );

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
