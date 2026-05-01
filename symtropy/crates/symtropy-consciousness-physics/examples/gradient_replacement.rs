// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gradient Replacement Study — is cooperation controller-invariant?
//!
//! The most important experiment for the book's thesis. Tests whether
//! cooperative clustering emerges under DIFFERENT movement controllers,
//! not just the handcrafted FEP gradient.
//!
//! If cooperation appears across multiple controllers → universal result.
//! If only under our FEP gradient → the gradient IS the mechanism, not thermodynamics.
//!
//! 6 controllers × 20 seeds × 8,000 ticks:
//! 1. FEP_GRADIENT: current handcrafted gradient (baseline)
//! 2. WELL_ONLY: seek nearest well, ignore other agents entirely
//! 3. PARTNER_ONLY: seek nearest resonant partner, ignore wells
//! 4. GREEDY_ENERGY: move toward whatever maximizes energy gain next tick
//! 5. RANDOM_WALK: uniform random direction each tick
//! 6. STATIONARY: don't move at all
//!
//! All controllers share identical thermodynamics (maintenance, resonance
//! regen, well regen, energy budget). Only the movement rule changes.
//!
//! Run: cargo run --example gradient_replacement --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 8_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

#[derive(Clone, Copy)]
enum Controller {
    FepGradient,
    WellOnly,
    PartnerOnly,
    GreedyEnergy,
    RandomWalk,
    Stationary,
}

struct ReplaceResult {
    controller: &'static str,
    alive: f64,
    clustering: f64,
    cooperation: f64,
    energy: f64,
}

fn run_experiment(ctrl: Controller, seed: u64) -> ReplaceResult {
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

    let name = match ctrl {
        Controller::FepGradient => "FEP_GRADIENT",
        Controller::WellOnly => "WELL_ONLY",
        Controller::PartnerOnly => "PARTNER_ONLY",
        Controller::GreedyEnergy => "GREEDY_ENERGY",
        Controller::RandomWalk => "RANDOM_WALK",
        Controller::Stationary => "STATIONARY",
    };

    let mut coop = 0u64;

    for _tick in 0..TICKS {
        // Consciousness update (same for all controllers)
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

        // Movement — THE VARIABLE UNDER TEST
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

        for (idx, &h) in handles.iter().enumerate() {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position().0;

            let vel: SVector<f64, 2> = match ctrl {
                Controller::FepGradient => {
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

                Controller::WellOnly => {
                    // Move toward nearest available well
                    let mut best_dir = SVector::zeros();
                    let mut best_dist = f64::MAX;
                    for (wp, wr) in &wdata {
                        if *wr < 0.01 {
                            continue;
                        }
                        let delta = wp - pos;
                        let d = delta.norm();
                        if d < best_dist && d > 1.0 {
                            best_dist = d;
                            best_dir = delta / d;
                        }
                    }
                    best_dir * 20.0
                }

                Controller::PartnerOnly => {
                    // Move toward nearest resonant partner, ignore wells
                    let mut best_dir = SVector::zeros();
                    let mut best_score = 0.0f64;
                    for (ap, ah) in &adata {
                        let delta = ap - pos;
                        let d = delta.norm();
                        if d < 2.0 || d > consciousness.constants.harmony_range {
                            continue;
                        }
                        let res = HarmonyField::<2>::resonance(&e.harmony_activations, ah);
                        if res > best_score {
                            best_score = res;
                            best_dir = delta / d;
                        }
                    }
                    best_dir * 20.0
                }

                Controller::GreedyEnergy => {
                    // Sample 8 directions, pick the one that would maximize energy gain
                    let mut best_dir = SVector::zeros();
                    let mut best_gain = f64::NEG_INFINITY;
                    for angle_idx in 0..8 {
                        let angle = angle_idx as f64 * std::f64::consts::TAU / 8.0;
                        let test_dir = SVector::from([angle.cos(), angle.sin()]);
                        let test_pos = pos + test_dir * 5.0; // look 5 units ahead

                        // Estimate energy gain at test position
                        let mut gain = 0.0;
                        // Well proximity bonus
                        for (wp, wr) in &wdata {
                            if *wr < 0.01 {
                                continue;
                            }
                            let d = (test_pos - wp).norm();
                            if d < 35.0 {
                                gain += consciousness.constants.energy_well_regen_rate * wr;
                            }
                        }
                        // Resonance proximity bonus
                        for (ap, ah) in &adata {
                            let d = (test_pos - ap).norm();
                            if d < 2.0 || d > consciousness.constants.harmony_range {
                                continue;
                            }
                            let res = HarmonyField::<2>::resonance(&e.harmony_activations, ah);
                            if res > 0.5 {
                                gain += consciousness.constants.harmony_resonance_regen_rate
                                    * (res - 0.5)
                                    * 2.0;
                            }
                        }
                        if gain > best_gain {
                            best_gain = gain;
                            best_dir = test_dir;
                        }
                    }
                    best_dir * 20.0
                }

                Controller::RandomWalk => {
                    let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                    SVector::from([angle.cos() * 20.0, angle.sin() * 20.0])
                }

                Controller::Stationary => SVector::zeros(),
            };

            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = vel;
            }
        }

        // Maintenance + regen (IDENTICAL for all controllers)
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
                    let pos = b.position().0;
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

        // Cooperation (IDENTICAL for all controllers — passive, not chosen)
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

    ReplaceResult {
        controller: name,
        alive,
        energy,
        clustering: if clustering.is_finite() {
            clustering
        } else {
            0.0
        },
        cooperation: coop as f64,
    }
}

fn main() {
    eprintln!("=== Gradient Replacement Study ===");
    eprintln!("Is cooperation controller-invariant?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds, 6 controllers");

    println!("controller,seed,alive,energy,clustering,cooperation");

    let controllers = [
        (Controller::FepGradient, "FEP_GRADIENT"),
        (Controller::WellOnly, "WELL_ONLY"),
        (Controller::PartnerOnly, "PARTNER_ONLY"),
        (Controller::GreedyEnergy, "GREEDY_ENERGY"),
        (Controller::RandomWalk, "RANDOM_WALK"),
        (Controller::Stationary, "STATIONARY"),
    ];
    let mut all: Vec<(&str, Vec<ReplaceResult>)> = Vec::new();

    for &(ctrl, name) in &controllers {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(ctrl, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.2},{:.0}",
                r.controller, r.alive, r.energy, r.clustering, r.cooperation
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}, cluster={:.2}, coop={:.0}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n
        );
        all.push((name, results));
    }

    // Summary
    eprintln!("\n── Controller Comparison ──");
    eprintln!("  Controller       Alive  Cluster  Cooperation  Energy");
    for (name, results) in &all {
        let n = results.len() as f64;
        eprintln!(
            "  {:15} {:5.1}  {:6.2}   {:9.0}    {:5.1}J",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n,
            results.iter().map(|r| r.energy).sum::<f64>() / n
        );
    }

    // Statistical tests: each controller vs FEP baseline (Holm-Bonferroni)
    let fep = &all[0].1;
    let fep_alive: Vec<f64> = fep.iter().map(|r| r.alive).collect();
    let fep_coop: Vec<f64> = fep.iter().map(|r| r.cooperation).collect();

    let mut p_alive = Vec::new();
    let mut d_alive = Vec::new();
    let mut p_coop = Vec::new();
    let mut d_coop = Vec::new();

    for (name, results) in &all[1..] {
        let a: Vec<f64> = results.iter().map(|r| r.alive).collect();
        let c: Vec<f64> = results.iter().map(|r| r.cooperation).collect();
        let (_, _, pa) = mann_whitney_u(&fep_alive, &a);
        let (_, _, pc) = mann_whitney_u(&fep_coop, &c);
        p_alive.push((*name, pa));
        d_alive.push(cohens_d(&fep_alive, &a));
        p_coop.push((*name, pc));
        d_coop.push(cohens_d(&fep_coop, &c));
    }

    let corrected_alive = holm_bonferroni(&p_alive, 0.05);
    let corrected_coop = holm_bonferroni(&p_coop, 0.05);

    eprintln!("\n── vs FEP_GRADIENT: Survival (Holm-Bonferroni, k=5) ──");
    for (i, &(label, adj_p, sig)) in corrected_alive.iter().enumerate() {
        let d = d_alive[i];
        let size = if d.abs() > 0.8 {
            "LARGE"
        } else if d.abs() > 0.5 {
            "medium"
        } else {
            "small"
        };
        eprintln!(
            "  vs {:15}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
            label,
            if sig { "← SIG" } else { "" }
        );
    }

    eprintln!("\n── vs FEP_GRADIENT: Cooperation (Holm-Bonferroni, k=5) ──");
    for (i, &(label, adj_p, sig)) in corrected_coop.iter().enumerate() {
        let d = d_coop[i];
        let size = if d.abs() > 0.8 {
            "LARGE"
        } else if d.abs() > 0.5 {
            "medium"
        } else {
            "small"
        };
        eprintln!(
            "  vs {:15}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
            label,
            if sig { "← SIG" } else { "" }
        );
    }

    // THE KEY QUESTION
    let controllers_with_coop: Vec<&str> = all
        .iter()
        .filter(|(_, results)| {
            let mean_coop =
                results.iter().map(|r| r.cooperation).sum::<f64>() / results.len() as f64;
            mean_coop > 50_000.0 // substantial cooperation
        })
        .map(|(name, _)| *name)
        .collect();

    eprintln!("\n── VERDICT ──");
    eprintln!(
        "  Controllers producing substantial cooperation: {:?}",
        controllers_with_coop
    );
    if controllers_with_coop.len() >= 4 {
        eprintln!(
            "  CONTROLLER-INVARIANT: Cooperation emerges across {} of 6 controllers.",
            controllers_with_coop.len()
        );
        eprintln!(
            "  The thesis is UNIVERSAL — cooperation depends on thermodynamics, not the gradient."
        );
    } else if controllers_with_coop.len() >= 2 {
        eprintln!(
            "  PARTIALLY INVARIANT: Cooperation emerges under {} controllers.",
            controllers_with_coop.len()
        );
        eprintln!("  The gradient matters but is not uniquely privileged.");
    } else {
        eprintln!("  CONTROLLER-DEPENDENT: Only the FEP gradient produces cooperation.");
        eprintln!("  The thesis is NARROW — the gradient IS the mechanism, not thermodynamics.");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
