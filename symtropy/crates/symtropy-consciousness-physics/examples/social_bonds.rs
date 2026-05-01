// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Social Bonds — do physics constraints create cooperative structures?
//!
//! When two agents achieve high harmony resonance (>0.7), a DistanceConstraint
//! physically connects them. Bonds persist once formed (no remove API),
//! modeling lasting social bonds.
//!
//! 6 distinct harmony archetypes ensure selective bonding — not everyone
//! resonates with everyone. High survival pressure (small wells, harsh
//! maintenance) creates genuine life-or-death stakes.
//!
//! Run: cargo run --example social_bonds --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::constraint::DistanceConstraint;
use symtropy_physics::{BodyHandle, PhysicsWorld};

const AGENTS: usize = 20;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 10;
const BOND_FORM_THRESHOLD: f64 = 0.7; // resonance to form bond
const BOND_REST_LENGTH: f64 = 8.0; // preferred distance when bonded
const BOND_STIFFNESS: f64 = 0.5;

/// 6 distinct harmony archetypes with genuine diversity.
/// Some pairs resonate highly (>0.7), others don't (<0.5).
const HARMONY_PROFILES: [[f64; 9]; 6] = [
    [0.9, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.8, 0.5], // Stillness+Stewardship
    [0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.5], // Play+Kinship
    [0.1, 0.2, 0.8, 0.2, 0.8, 0.1, 0.2, 0.1, 0.5], // Craft+Curiosity
    [0.1, 0.1, 0.2, 0.9, 0.2, 0.8, 0.1, 0.1, 0.5], // Justice+Celebration
    [0.7, 0.3, 0.3, 0.1, 0.3, 0.1, 0.3, 0.7, 0.5], // variant of Stillness (compatible with 0)
    [0.3, 0.7, 0.1, 0.2, 0.1, 0.3, 0.7, 0.3, 0.5], // variant of Play (compatible with 1)
];

struct BondResult {
    alive: f64,
    energy: f64,
    clustering: f64,
    bonds_formed: f64,
    bonds_active_end: f64,
    sanctuary_ticks: f64,
    mean_phi: f64,
}

fn run_experiment(use_bonds: bool, seed: u64) -> BondResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.35; // harsh pressure
    consciousness.constants = constants;

    // Small wells = real scarcity
    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![2000.0f64; 2];

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
            e.harmony_activations = HARMONY_PROFILES[i % 6];
        }
        handles.push(h);
    }

    let mut active_bonds: Vec<(usize, usize, bool)> = Vec::new();
    let mut total_bonds_formed = 0u64;
    let mut sanctuary_ticks = 0u64;
    let mut phi_sum = 0.0f64;
    let mut phi_count = 0u64;

    for tick in 0..TICKS {
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
                    broadcast: 0.6,
                    working_memory: (1.0 - pe).max(0.0),
                    attention: 0.6,
                    recurrence: 1.0,
                    embodiment: 0.7,
                    knowledge: (ht / 8.0).min(1.0),
                    synchrony: consciousness.collective_phi.max(0.5),
                }
            };
            consciousness.update_entity(h, &inputs, Point::origin());
        }

        // Track phi
        if tick % 100 == 0 {
            for &h in &handles {
                let p = consciousness.phi(h);
                phi_sum += p;
                phi_count += 1;
            }
        }

        // FEP gradient
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
            .map(|(&p, &r)| (p, (r / 2000.0).min(1.0)))
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

        // Bond management (only in BONDS condition)
        if use_bonds && tick % 50 == 0 {
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    if active_bonds
                        .iter()
                        .any(|&(a, b, active)| active && ((a == i && b == j) || (a == j && b == i)))
                    {
                        continue;
                    }
                    let (ha, hb) = (handles[i], handles[j]);
                    let in_range = match (world.body(ha), world.body(hb)) {
                        (Some(a), Some(b)) => {
                            a.position().distance(b.position())
                                < consciousness.constants.harmony_range
                        }
                        _ => false,
                    };
                    if !in_range {
                        continue;
                    }
                    // Both must be alive
                    let a_alive = consciousness
                        .entities
                        .get(&ha)
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false);
                    let b_alive = consciousness
                        .entities
                        .get(&hb)
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false);
                    if !a_alive || !b_alive {
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

                    if res > BOND_FORM_THRESHOLD {
                        world.add_constraint(Box::new(DistanceConstraint {
                            body_a: ha,
                            body_b: hb,
                            rest_length: BOND_REST_LENGTH,
                            stiffness: BOND_STIFFNESS,
                        }));
                        active_bonds.push((i, j, true));
                        total_bonds_formed += 1;
                    }
                }
            }
        }

        // Maintenance + regen
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let nw: Vec<Option<usize>> = handles
            .iter()
            .map(|&h| {
                world.body(h).and_then(|b| {
                    let pos = b.position().0;
                    wells
                        .iter()
                        .enumerate()
                        .find(|(i, &w)| (pos - w).norm() < 35.0 && well_remaining[*i] > 0.0)
                        .map(|(i, _)| i)
                })
            })
            .collect();
        for (idx, &h) in handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                if let Some(wi) = nw[idx] {
                    let d = wr.min(well_remaining[wi]);
                    e.energy.regenerate(d);
                    well_remaining[wi] -= d;
                }
            }
        }

        // Harmony resonance regen
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
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
                }
            }
        }

        // Check sanctuary activation
        let any_sanctuary = consciousness.sanctuaries.values().any(|s| s.active);
        if any_sanctuary {
            sanctuary_ticks += 1;
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
    let bonds_active = active_bonds.iter().filter(|b| b.2).count() as f64;
    let mean_phi = if phi_count > 0 {
        phi_sum / phi_count as f64
    } else {
        0.0
    };

    BondResult {
        alive,
        energy,
        clustering,
        bonds_formed: total_bonds_formed as f64,
        bonds_active_end: bonds_active,
        sanctuary_ticks: sanctuary_ticks as f64,
        mean_phi,
    }
}

fn main() {
    eprintln!("=== Social Bonds Experiment ===");
    eprintln!("Do physics constraints create cooperative structures?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");
    eprintln!("6 harmony archetypes, selective bonding (resonance > {BOND_FORM_THRESHOLD})");

    println!(
        "condition,seed,alive,energy,clustering,bonds_formed,bonds_active,sanctuary_ticks,mean_phi"
    );

    let mut nobond_results = Vec::new();
    let mut bond_results = Vec::new();

    for s in 0..SEEDS {
        let seed = 42 + s as u64 * 997;
        eprintln!("  NO_BONDS seed={seed}...");
        let r = run_experiment(false, seed);
        println!(
            "NO_BONDS,{seed},{:.1},{:.1},{:.2},{:.0},{:.0},{:.0},{:.4}",
            r.alive,
            r.energy,
            r.clustering,
            r.bonds_formed,
            r.bonds_active_end,
            r.sanctuary_ticks,
            r.mean_phi
        );
        nobond_results.push(r);
    }
    for s in 0..SEEDS {
        let seed = 42 + s as u64 * 997;
        eprintln!("  BONDS seed={seed}...");
        let r = run_experiment(true, seed);
        println!(
            "BONDS,{seed},{:.1},{:.1},{:.2},{:.0},{:.0},{:.0},{:.4}",
            r.alive,
            r.energy,
            r.clustering,
            r.bonds_formed,
            r.bonds_active_end,
            r.sanctuary_ticks,
            r.mean_phi
        );
        bond_results.push(r);
    }

    let n = SEEDS as f64;
    eprintln!("\n── NO_BONDS ──");
    eprintln!(
        "  Alive:      {:.1}/{AGENTS}",
        nobond_results.iter().map(|r| r.alive).sum::<f64>() / n
    );
    eprintln!(
        "  Energy:     {:.1} J",
        nobond_results.iter().map(|r| r.energy).sum::<f64>() / n
    );
    eprintln!(
        "  Clustering: {:.2}",
        nobond_results.iter().map(|r| r.clustering).sum::<f64>() / n
    );
    eprintln!(
        "  Mean Φ:     {:.4}",
        nobond_results.iter().map(|r| r.mean_phi).sum::<f64>() / n
    );
    eprintln!(
        "  Sanctuary:  {:.0} ticks",
        nobond_results
            .iter()
            .map(|r| r.sanctuary_ticks)
            .sum::<f64>()
            / n
    );

    eprintln!("\n── BONDS ──");
    eprintln!(
        "  Alive:      {:.1}/{AGENTS}",
        bond_results.iter().map(|r| r.alive).sum::<f64>() / n
    );
    eprintln!(
        "  Energy:     {:.1} J",
        bond_results.iter().map(|r| r.energy).sum::<f64>() / n
    );
    eprintln!(
        "  Clustering: {:.2}",
        bond_results.iter().map(|r| r.clustering).sum::<f64>() / n
    );
    eprintln!(
        "  Bonds formed: {:.1}",
        bond_results.iter().map(|r| r.bonds_formed).sum::<f64>() / n
    );
    eprintln!(
        "  Bonds active: {:.1}",
        bond_results.iter().map(|r| r.bonds_active_end).sum::<f64>() / n
    );
    eprintln!(
        "  Mean Φ:     {:.4}",
        bond_results.iter().map(|r| r.mean_phi).sum::<f64>() / n
    );
    eprintln!(
        "  Sanctuary:  {:.0} ticks",
        bond_results.iter().map(|r| r.sanctuary_ticks).sum::<f64>() / n
    );

    // Statistical comparison — Survival
    let nb_alive: Vec<f64> = nobond_results.iter().map(|r| r.alive).collect();
    let b_alive: Vec<f64> = bond_results.iter().map(|r| r.alive).collect();
    let (u, z, p) = mann_whitney_u(&nb_alive, &b_alive);
    let d = cohens_d(&nb_alive, &b_alive);
    eprintln!("\n── Survival: NO_BONDS vs BONDS ──");
    eprintln!("  Mann-Whitney U={:.1}, z={:.3}, p={:.4}", u, z, p);
    eprintln!("  Cohen's d={:.3} ({})", d, effect_label(d));
    eprintln!(
        "  {}",
        if p < 0.05 {
            "SIGNIFICANT"
        } else {
            "not significant"
        }
    );

    // Statistical comparison — Clustering
    let nb_cluster: Vec<f64> = nobond_results.iter().map(|r| r.clustering).collect();
    let b_cluster: Vec<f64> = bond_results.iter().map(|r| r.clustering).collect();
    let (u2, z2, p2) = mann_whitney_u(&nb_cluster, &b_cluster);
    let d2 = cohens_d(&nb_cluster, &b_cluster);
    eprintln!("\n── Clustering: NO_BONDS vs BONDS ──");
    eprintln!("  Mann-Whitney U={:.1}, z={:.3}, p={:.4}", u2, z2, p2);
    eprintln!("  Cohen's d={:.3} ({})", d2, effect_label(d2));

    // Statistical comparison — Energy
    let nb_energy: Vec<f64> = nobond_results.iter().map(|r| r.energy).collect();
    let b_energy: Vec<f64> = bond_results.iter().map(|r| r.energy).collect();
    let (u3, z3, p3) = mann_whitney_u(&nb_energy, &b_energy);
    let d3 = cohens_d(&nb_energy, &b_energy);
    eprintln!("\n── Energy: NO_BONDS vs BONDS ──");
    eprintln!("  Mann-Whitney U={:.1}, z={:.3}, p={:.4}", u3, z3, p3);
    eprintln!("  Cohen's d={:.3} ({})", d3, effect_label(d3));

    // Bond selectivity analysis
    if !bond_results.is_empty() {
        let max_possible = (AGENTS * (AGENTS - 1)) as f64 / 2.0;
        let mean_bonds = bond_results.iter().map(|r| r.bonds_formed).sum::<f64>() / n;
        eprintln!("\n── Bond Selectivity ──");
        eprintln!("  Possible pairs: {:.0}", max_possible);
        eprintln!(
            "  Bonds formed:   {:.1} ({:.1}%)",
            mean_bonds,
            mean_bonds / max_possible * 100.0
        );
    }

    eprintln!("\n=== Complete ===");
}

fn effect_label(d: f64) -> &'static str {
    if d.abs() > 0.8 {
        "large"
    } else if d.abs() > 0.5 {
        "medium"
    } else {
        "small"
    }
}

fn avg_nn(world: &PhysicsWorld<2>, handles: &[BodyHandle]) -> f64 {
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
