// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tragedy of the Commons — do agents overexploit shared resources?
//!
//! Compares 3 conditions:
//! 1. ABUNDANT: Large wells (10,000J each) — no scarcity
//! 2. SCARCE: Small wells (1,000J each) — moderate scarcity
//! 3. DEPLETING: Wells start at 2,000J but deplete 2x faster when crowded
//!
//! The tragedy of the commons (Hardin 1968) predicts that shared resources
//! are overexploited when individual incentives conflict with collective good.
//! In our engine, FEP gradient drives agents toward wells (individual benefit),
//! but crowding depletes wells faster (collective cost).
//!
//! Key metrics: well depletion rate, survival, spatial distribution,
//! whether agents voluntarily disperse or crowd-and-crash.
//!
//! Run: cargo run --example tragedy_commons --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 24;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 8;

#[derive(Clone, Copy)]
enum Condition {
    Abundant,
    Scarce,
    Depleting,
}

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

struct CommonsResult {
    condition: &'static str,
    alive: f64,
    energy: f64,
    wells_depleted_tick: f64, // tick when first well depleted (or TICKS if never)
    well_remaining_total: f64,
    clustering: f64,
    cooperation: f64,
    crowding_events: f64, // ticks where >8 agents at same well
}

fn run_experiment(condition: Condition, seed: u64) -> CommonsResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![
        SVector::from([40.0, 0.0]),
        SVector::from([-40.0, 0.0]),
        SVector::from([0.0, 40.0]),
    ];
    let well_capacity: f64 = match condition {
        Condition::Abundant => 10_000.0,
        Condition::Scarce => 1_000.0,
        Condition::Depleting => 2_000.0,
    };
    let mut well_remaining = vec![well_capacity; 3];
    let cond_name = match condition {
        Condition::Abundant => "ABUNDANT",
        Condition::Scarce => "SCARCE",
        Condition::Depleting => "DEPLETING",
    };

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 120.0;
        let y = (rng_f64(&mut rng) - 0.5) * 120.0;
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

    let mut cooperation_events = 0u64;
    let mut first_depletion_tick = TICKS;
    let mut crowding_events = 0u64;

    for tick in 0..TICKS {
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
            .map(|(&p, &r)| (p, (r / well_capacity).min(1.0)))
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

        // Count agents at each well
        let well_counts: Vec<usize> = wells
            .iter()
            .enumerate()
            .map(|(wi, &w)| {
                handles
                    .iter()
                    .filter(|&&h| {
                        world
                            .body(h)
                            .map(|b| (b.position().0 - w).norm() < 35.0)
                            .unwrap_or(false)
                    })
                    .count()
            })
            .collect();

        // Check crowding (>8 agents at one well)
        if well_counts.iter().any(|&c| c > 8) {
            crowding_events += 1;
        }

        // Maintenance + regen with crowding penalty for DEPLETING
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
                // Find nearest well
                if let Some(b) = world.body(h) {
                    let pos = b.position().0;
                    for (wi, &w) in wells.iter().enumerate() {
                        if (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
                            let base_draw = wr.min(well_remaining[wi]);
                            // DEPLETING: crowding accelerates depletion
                            let draw = match condition {
                                Condition::Depleting => {
                                    let crowd_factor =
                                        1.0 + (well_counts[wi] as f64 - 4.0).max(0.0) * 0.3;
                                    base_draw * crowd_factor
                                }
                                _ => base_draw,
                            };
                            let actual = draw.min(well_remaining[wi]);
                            e.energy.regenerate(actual.min(base_draw)); // agent only gets base rate
                            well_remaining[wi] -= actual; // but well depletes at crowd rate
                            break;
                        }
                    }
                }
            }
        }

        // Check for first well depletion
        if first_depletion_tick == TICKS && well_remaining.iter().any(|&r| r <= 0.0) {
            first_depletion_tick = tick;
        }

        // Cooperation
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
                    cooperation_events += 1;
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
    let clustering = avg_nn(&world, &handles);
    let well_total: f64 = well_remaining.iter().sum();

    CommonsResult {
        condition: cond_name,
        alive,
        energy,
        wells_depleted_tick: first_depletion_tick as f64,
        well_remaining_total: well_total,
        clustering,
        cooperation: cooperation_events as f64,
        crowding_events: crowding_events as f64,
    }
}

fn main() {
    eprintln!("=== Tragedy of the Commons Experiment ===");
    eprintln!("Do agents overexploit shared resources?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds, 3 conditions");

    println!(
        "condition,seed,alive,energy,depletion_tick,well_remaining,clustering,cooperation,crowding"
    );

    let conditions = [
        (Condition::Abundant, "ABUNDANT"),
        (Condition::Scarce, "SCARCE"),
        (Condition::Depleting, "DEPLETING"),
    ];
    let mut all_results: Vec<(&str, Vec<CommonsResult>)> = Vec::new();

    for &(cond, name) in &conditions {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(cond, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.0},{:.0},{:.2},{:.0},{:.0}",
                r.condition,
                r.alive,
                r.energy,
                r.wells_depleted_tick,
                r.well_remaining_total,
                r.clustering,
                r.cooperation,
                r.crowding_events
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}/{AGENTS}, depletion_tick={:.0}, wells_left={:.0}J",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.wells_depleted_tick).sum::<f64>() / n,
            results.iter().map(|r| r.well_remaining_total).sum::<f64>() / n
        );
        all_results.push((name, results));
    }

    // Summary
    eprintln!("\n── Resource Economics ──");
    eprintln!("  Condition    Alive   Energy   1st Depletion  Wells Left  Crowding");
    for (name, results) in &all_results {
        let n = results.len() as f64;
        eprintln!(
            "  {:10} {:5.1}    {:5.1}J   tick {:6.0}    {:7.0}J    {:5.0}",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.energy).sum::<f64>() / n,
            results.iter().map(|r| r.wells_depleted_tick).sum::<f64>() / n,
            results.iter().map(|r| r.well_remaining_total).sum::<f64>() / n,
            results.iter().map(|r| r.crowding_events).sum::<f64>() / n
        );
    }

    // Statistical tests: Scarce vs Depleting (same starting resources, different crowding penalty)
    if all_results.len() >= 3 {
        let scarce = &all_results[1].1;
        let depleting = &all_results[2].1;
        let s_alive: Vec<f64> = scarce.iter().map(|r| r.alive).collect();
        let d_alive: Vec<f64> = depleting.iter().map(|r| r.alive).collect();
        let (u, z, p) = mann_whitney_u(&s_alive, &d_alive);
        let d = cohens_d(&s_alive, &d_alive);
        eprintln!("\n── SCARCE vs DEPLETING (survival) ──");
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
        eprintln!(
            "  {}",
            if p < 0.05 {
                "SIGNIFICANT — crowding penalty affects survival"
            } else {
                "not significant"
            }
        );

        // Cooperation comparison
        let s_coop: Vec<f64> = scarce.iter().map(|r| r.cooperation).collect();
        let d_coop: Vec<f64> = depleting.iter().map(|r| r.cooperation).collect();
        let d2 = cohens_d(&s_coop, &d_coop);
        eprintln!("\n── Cooperation: SCARCE vs DEPLETING ──");
        eprintln!(
            "  Cohen's d={d2:.3} ({})",
            if d2.abs() > 0.8 {
                "large"
            } else {
                "medium/small"
            }
        );
    }

    eprintln!("\n=== Complete ===");
}

fn avg_nn(world: &PhysicsWorld<2>, handles: &[symtropy_physics::BodyHandle]) -> f64 {
    let pos: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|h| world.body(*h).map(|b| b.position().0))
        .collect();
    if pos.len() < 2 {
        return f64::MAX;
    }
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
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
