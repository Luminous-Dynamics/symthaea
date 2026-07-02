// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complementary Hunt — does genuine coordination make FEP indispensable?
//!
//! Prey requires COMPLEMENTARY agents (driver + ambusher with low mutual
//! resonance) to capture. This tests whether the FEP social gradient
//! finally outperforms WELL_ONLY when the task requires coordination,
//! not just co-location.
//!
//! Run: cargo run --example complementary_hunt --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::prey::{Prey, check_complementary_hunt};
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20; // 10 drivers + 10 ambushers
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;
const PREY_COUNT: usize = 5;
const PREY_ENERGY: f64 = 30.0;
const PREY_RESPAWN: usize = 500;

#[derive(Clone, Copy)]
enum Ctrl {
    Fep,
    WellOnly,
    Greedy,
    PartnerOnly,
}

fn run_experiment(ctrl: Ctrl, require_complementary: bool, seed: u64) -> (f64, f64, f64) {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.15;
    constants.ambient_regen_rate = 0.002;
    consciousness.constants = constants;

    // NO traditional wells — prey is the only substantial energy source
    let mut prey_list: Vec<Prey<2>> = Vec::new();
    let mut rng = seed;
    for _ in 0..PREY_COUNT {
        let x = (rng_f64(&mut rng) - 0.5) * 100.0;
        let y = (rng_f64(&mut rng) - 0.5) * 100.0;
        prey_list.push(Prey::new(
            SVector::from([x, y]),
            PREY_ENERGY,
            30.0,
            PREY_RESPAWN,
        ));
    }

    let mut handles = Vec::new();
    // First half: drivers (harmony[0] dominant)
    for i in 0..AGENTS / 2 {
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
            e.harmony_activations = [0.9, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.5];
        }
        handles.push(h);
    }
    // Second half: ambushers (harmony[2] dominant)
    for i in 0..AGENTS / 2 {
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
            e.harmony_activations = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5];
        }
        handles.push(h);
    }

    let mut hunts = 0u64;

    for _tick in 0..TICKS {
        // Tick prey respawn
        for p in &mut prey_list {
            p.tick();
        }

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

        // Prey positions as "wells" for navigation
        let adata: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position(),
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let wdata: Vec<_> = prey_list
            .iter()
            .filter(|p| p.is_active())
            .map(|p| (p.position, 0.8f64))
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

            let vel: SVector<f64, 2> = match ctrl {
                Ctrl::Fep => {
                    let near: Vec<_> = adata
                        .iter()
                        .filter(|(p, _)| {
                            let d = (p - pos).norm();
                            d > 2.0 && d < consciousness.constants.harmony_range
                        })
                        .cloned()
                        .collect();
                    fep_gradient::free_energy_gradient(
                        &pos,
                        e.energy.fraction_remaining(),
                        &e.harmony_activations,
                        &near,
                        &wdata,
                        None,
                        0.0,
                    ) * 20.0
                }
                Ctrl::WellOnly => {
                    let mut best_dir = SVector::zeros();
                    let mut best_dist = f64::MAX;
                    for (wp, _) in &wdata {
                        let delta = wp - pos;
                        let d = delta.norm();
                        if d < best_dist && d > 1.0 {
                            best_dist = d;
                            best_dir = delta / d;
                        }
                    }
                    best_dir * 20.0
                }
                Ctrl::Greedy => {
                    let mut best_dir = SVector::zeros();
                    let mut best_gain = f64::NEG_INFINITY;
                    for ai in 0..8 {
                        let angle = ai as f64 * std::f64::consts::TAU / 8.0;
                        let td = SVector::from([angle.cos(), angle.sin()]);
                        let tp = pos + td * 5.0;
                        let mut gain = 0.0;
                        for (wp, _) in &wdata {
                            if (tp - wp).norm() < 30.0 {
                                gain += 1.0;
                            }
                        }
                        if gain > best_gain {
                            best_gain = gain;
                            best_dir = td;
                        }
                    }
                    best_dir * 20.0
                }
                Ctrl::PartnerOnly => {
                    let mut best_dir = SVector::zeros();
                    let mut best_score = 0.0f64;
                    for (ap, _ah) in &adata {
                        let delta = ap - pos;
                        let d = delta.norm();
                        if d > 2.0 && d < consciousness.constants.harmony_range {
                            if d < 1.0 / best_score.max(0.01) {
                                best_score = 1.0 / d;
                                best_dir = delta / d;
                            }
                        }
                    }
                    best_dir * 20.0
                }
            };
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = vel;
            }
        }

        // Maintenance
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
            }
        }

        // Hunt check — prey yields energy only under complementary condition
        for prey in &mut prey_list {
            if !prey.is_active() {
                continue;
            }

            let agents_at_prey: Vec<(usize, [f64; 9])> = handles
                .iter()
                .enumerate()
                .filter(|&(_, &h)| {
                    world
                        .body(h)
                        .map(|b| (b.position() - prey.position).norm() < prey.radius)
                        .unwrap_or(false)
                        && consciousness
                            .entities
                            .get(&h)
                            .map(|e| !e.energy.is_collapsed())
                            .unwrap_or(false)
                })
                .map(|(idx, &h)| {
                    (
                        idx,
                        consciousness.entities.get(&h).unwrap().harmony_activations,
                    )
                })
                .collect();

            let hunt_success = if require_complementary {
                check_complementary_hunt(&agents_at_prey, 0, 2, 0.6, 0.3).is_some()
            } else {
                !agents_at_prey.is_empty() // any agent can capture
            };

            if hunt_success {
                // Distribute energy to all agents at prey
                for &(idx, _) in &agents_at_prey {
                    let h = handles[idx];
                    if let Some(e) = consciousness.entities.get_mut(&h) {
                        e.energy
                            .regenerate(prey.energy_value / agents_at_prey.len() as f64);
                    }
                }
                prey.capture();
                hunts += 1;
            }
        }

        // Passive resonance (always active)
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
    (alive, energy, hunts as f64)
}

fn main() {
    eprintln!("=== Complementary Hunt ===");
    eprintln!("Does genuine coordination make FEP indispensable?");
    eprintln!(
        "{AGENTS} agents (10 drivers + 10 ambushers), {PREY_COUNT} prey, {TICKS} ticks, {SEEDS} seeds"
    );

    println!("controller,complementary,seed,alive,energy,hunts");

    let ctrls = [
        (Ctrl::Fep, "FEP"),
        (Ctrl::WellOnly, "WELL"),
        (Ctrl::Greedy, "GREEDY"),
        (Ctrl::PartnerOnly, "PARTNER"),
    ];

    // Run with complementary requirement
    eprintln!("\n── Complementary Hunt (coordination required) ──");
    let mut comp_results: Vec<(&str, Vec<f64>)> = Vec::new();
    for &(ctrl, name) in &ctrls {
        let mut alive_vec = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            let (alive, energy, hunts) = run_experiment(ctrl, true, seed);
            println!("{name},true,{seed},{alive:.1},{energy:.1},{hunts:.0}");
            alive_vec.push(alive);
        }
        let mean = alive_vec.iter().sum::<f64>() / SEEDS as f64;
        eprintln!("  {name}: alive={mean:.1}");
        comp_results.push((name, alive_vec));
    }

    // Run without complementary requirement (baseline)
    eprintln!("\n── Individual Hunt (no coordination) ──");
    let mut solo_results: Vec<(&str, Vec<f64>)> = Vec::new();
    for &(ctrl, name) in &ctrls {
        let mut alive_vec = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            let (alive, energy, hunts) = run_experiment(ctrl, false, seed);
            println!("{name},false,{seed},{alive:.1},{energy:.1},{hunts:.0}");
            alive_vec.push(alive);
        }
        let mean = alive_vec.iter().sum::<f64>() / SEEDS as f64;
        eprintln!("  {name}: alive={mean:.1}");
        solo_results.push((name, alive_vec));
    }

    // Summary
    eprintln!("\n── Summary ──");
    eprintln!("  Controller  Individual  Complementary  Δ       FEP helps?");
    for i in 0..ctrls.len() {
        let (_, name) = ctrls[i];
        let solo = solo_results[i].1.iter().sum::<f64>() / SEEDS as f64;
        let comp = comp_results[i].1.iter().sum::<f64>() / SEEDS as f64;
        let delta = comp - solo;
        eprintln!(
            "  {:8}     {:5.1}       {:5.1}          {:+5.1}",
            name, solo, comp, delta
        );
    }

    // KEY TEST: Does FEP outperform WELL under complementary condition?
    let fep_comp = &comp_results[0].1;
    let well_comp = &comp_results[1].1;
    let (_, _, p) = mann_whitney_u(well_comp, fep_comp);
    let d = cohens_d(well_comp, fep_comp);
    eprintln!("\n── KEY TEST: FEP vs WELL under complementary hunt ──");
    eprintln!(
        "  FEP={:.1}, WELL={:.1}, d={d:.3}, p={p:.4}",
        fep_comp.iter().sum::<f64>() / fep_comp.len() as f64,
        well_comp.iter().sum::<f64>() / well_comp.len() as f64
    );
    if d < -0.5 {
        eprintln!("  FEP WINS: Social gradient IS indispensable under complementary tasks.");
    } else if d > 0.5 {
        eprintln!("  WELL STILL WINS: Even complementary tasks don't require social coordination.");
    } else {
        eprintln!("  NO DIFFERENCE: Controllers perform similarly.");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
