// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Resonance Only — what happens when cooperation IS the only energy source?
//!
//! All prior experiments include energy wells. F46-52 showed that wells
//! dominate all social dynamics. This experiment removes wells entirely.
//! The ONLY way to regenerate energy is harmony resonance with a
//! compatible partner.
//!
//! Under this regime:
//! - WELL_ONLY agents have nothing to seek → should fail
//! - GREEDY agents see no well signal → should fail
//! - FEP agents seek resonant partners → may survive
//! - STATIONARY agents wait for random encounters → baseline
//!
//! If FEP outperforms under resonance-only: social seeking IS indispensable
//! when resources are social (not spatial). This is the clean test.
//!
//! Run: cargo run --example resonance_only --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;

const HARMONY_PROFILES: [[f64; 9]; 6] = [
    [0.9, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.8, 0.5],
    [0.8, 0.3, 0.2, 0.1, 0.3, 0.2, 0.2, 0.7, 0.5],
    [0.1, 0.2, 0.8, 0.2, 0.8, 0.1, 0.2, 0.1, 0.5],
    [0.2, 0.1, 0.7, 0.3, 0.7, 0.2, 0.1, 0.2, 0.5],
    [0.1, 0.8, 0.1, 0.2, 0.1, 0.2, 0.8, 0.1, 0.5],
    [0.2, 0.7, 0.2, 0.1, 0.2, 0.1, 0.7, 0.2, 0.5],
];

#[derive(Clone, Copy)]
enum Ctrl {
    Fep,
    WellOnly,
    Greedy,
    Random,
    Stationary,
    PartnerOnly,
}

struct ResOnlyResult {
    controller: &'static str,
    alive: f64,
    energy: f64,
    cooperation: f64,
    clustering: f64,
}

fn run_experiment(ctrl: Ctrl, seed: u64) -> ResOnlyResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    // Boost resonance regen to make it viable as sole energy source
    constants.harmony_resonance_regen_rate = 0.12; // 2× default
    constants.ambient_regen_rate = 0.0; // NO ambient — resonance only
    constants.consciousness_maintenance_per_tick = 0.15; // slightly lower maintenance
    consciousness.constants = constants;

    // NO WELLS — empty well list
    let wells: Vec<SVector<f64, 2>> = vec![];
    let well_remaining: Vec<f64> = vec![];

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 80.0;
        let y = (rng_f64(&mut rng) - 0.5) * 80.0;
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

    let ctrl_name = match ctrl {
        Ctrl::Fep => "FEP",
        Ctrl::WellOnly => "WELL_ONLY",
        Ctrl::Greedy => "GREEDY",
        Ctrl::Random => "RANDOM",
        Ctrl::Stationary => "STATIONARY",
        Ctrl::PartnerOnly => "PARTNER_ONLY",
    };

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
        // Empty well data — no wells exist
        let wdata: Vec<(SVector<f64, 2>, f64)> = vec![];

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
                    // No wells to seek — wander randomly
                    let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                    SVector::from([angle.cos() * 15.0, angle.sin() * 15.0])
                }
                Ctrl::Greedy => {
                    // No wells in lookahead — check for resonance benefit instead
                    let mut best_dir = SVector::zeros();
                    let mut best_gain = f64::NEG_INFINITY;
                    for ai in 0..8 {
                        let angle = ai as f64 * std::f64::consts::TAU / 8.0;
                        let td = SVector::from([angle.cos(), angle.sin()]);
                        let tp = pos + td * 5.0;
                        let mut gain = 0.0;
                        for (ap, ah) in &adata {
                            let d = (tp - ap).norm();
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
                            best_dir = td;
                        }
                    }
                    if best_gain <= 0.0 {
                        let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                        SVector::from([angle.cos() * 10.0, angle.sin() * 10.0])
                    } else {
                        best_dir * 20.0
                    }
                }
                Ctrl::Random => {
                    let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                    SVector::from([angle.cos() * 15.0, angle.sin() * 15.0])
                }
                Ctrl::Stationary => SVector::zeros(),
                Ctrl::PartnerOnly => {
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
            };
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = vel;
            }
        }

        // Maintenance (no well regen, no ambient)
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
        }

        // Resonance — the ONLY energy source
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
    let energy = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / AGENTS as f64;
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
        0.0
    };

    ResOnlyResult {
        controller: ctrl_name,
        alive,
        energy,
        cooperation: coop as f64,
        clustering: if clustering.is_finite() {
            clustering
        } else {
            0.0
        },
    }
}

fn main() {
    eprintln!("=== Resonance Only ===");
    eprintln!("NO WELLS. Resonance is the ONLY energy source.");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");
    eprintln!("Regen: 0.12 J/tick per resonant pair, maintenance: 0.15 J/tick, ambient: 0");

    println!("controller,seed,alive,energy,cooperation,clustering");

    let ctrls = [
        (Ctrl::Fep, "FEP"),
        (Ctrl::PartnerOnly, "PARTNER_ONLY"),
        (Ctrl::Greedy, "GREEDY"),
        (Ctrl::WellOnly, "WELL_ONLY"),
        (Ctrl::Random, "RANDOM"),
        (Ctrl::Stationary, "STATIONARY"),
    ];
    let mut all: Vec<(&str, Vec<ResOnlyResult>)> = Vec::new();

    for &(ctrl, name) in &ctrls {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(ctrl, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.0},{:.2}",
                r.controller, r.alive, r.energy, r.cooperation, r.clustering
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}, coop={:.0}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n
        );
        all.push((name, results));
    }

    eprintln!("\n── Resonance-Only Results ──");
    eprintln!("  Controller      Alive  Energy  Coop       Cluster");
    for (name, results) in &all {
        let n = results.len() as f64;
        eprintln!(
            "  {:14} {:5.1}  {:5.1}J  {:8.0}   {:6.2}",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.energy).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n
        );
    }

    // Statistical tests: FEP/PARTNER_ONLY vs RANDOM/STATIONARY
    let fep_alive: Vec<f64> = all[0].1.iter().map(|r| r.alive).collect();
    let partner_alive: Vec<f64> = all[1].1.iter().map(|r| r.alive).collect();
    let random_alive: Vec<f64> = all[4].1.iter().map(|r| r.alive).collect();
    let stationary_alive: Vec<f64> = all[5].1.iter().map(|r| r.alive).collect();

    let tests = vec![
        ("FEP vs RANDOM", {
            let (_, _, p) = mann_whitney_u(&fep_alive, &random_alive);
            p
        }),
        ("FEP vs STATIONARY", {
            let (_, _, p) = mann_whitney_u(&fep_alive, &stationary_alive);
            p
        }),
        ("PARTNER vs RANDOM", {
            let (_, _, p) = mann_whitney_u(&partner_alive, &random_alive);
            p
        }),
        ("PARTNER vs STATIONARY", {
            let (_, _, p) = mann_whitney_u(&partner_alive, &stationary_alive);
            p
        }),
    ];
    let effects = [
        cohens_d(&random_alive, &fep_alive),
        cohens_d(&stationary_alive, &fep_alive),
        cohens_d(&random_alive, &partner_alive),
        cohens_d(&stationary_alive, &partner_alive),
    ];
    let corrected = holm_bonferroni(&tests, 0.05);

    eprintln!("\n── Social Seeking vs Passive (Holm-Bonferroni, k=4) ──");
    for (i, &(label, adj_p, sig)) in corrected.iter().enumerate() {
        let d = effects[i];
        let size = if d.abs() > 0.8 {
            "LARGE"
        } else if d.abs() > 0.5 {
            "medium"
        } else {
            "small"
        };
        eprintln!(
            "  {label:22}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
            if sig { "← SIG" } else { "" }
        );
    }

    // THE VERDICT
    let fep_mean = fep_alive.iter().sum::<f64>() / fep_alive.len() as f64;
    let partner_mean = partner_alive.iter().sum::<f64>() / partner_alive.len() as f64;
    let random_mean = random_alive.iter().sum::<f64>() / random_alive.len() as f64;
    let stat_mean = stationary_alive.iter().sum::<f64>() / stationary_alive.len() as f64;

    let social_better = (fep_mean > random_mean + 2.0) || (partner_mean > random_mean + 2.0);

    eprintln!("\n── VERDICT ──");
    if social_better {
        eprintln!("  SOCIAL SEEKING IS INDISPENSABLE under resonance-only conditions.");
        eprintln!("  When cooperation IS the resource, social drive IS survival.");
        eprintln!(
            "  The FEP gradient transitions from liability to necessity when wells are removed."
        );
    } else {
        eprintln!("  Social seeking provides NO advantage even under resonance-only.");
        eprintln!("  Random encounters produce sufficient resonance for survival.");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
