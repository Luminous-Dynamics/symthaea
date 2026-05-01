// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Large Arena — does scale break by-product mutualism?
//!
//! F49 showed hidden wells don't matter because the 100×100 arena is
//! small relative to 35-unit well radius. This experiment uses a
//! 1000×1000 arena where random exploration should FAIL to find wells.
//!
//! Well radius (35) = 3.5% of arena width. Random walkers cover ~15
//! units/tick × 10,000 ticks = ~150,000 units of exploration. Arena
//! area = 1,000,000 sq units. Well coverage = π×35² ≈ 3,848 sq units
//! per well × 3 wells = 11,544 sq units = 1.2% of arena.
//!
//! Prediction: random exploration has ~low probability of finding all
//! wells. FEP social gradient may help: agents who found wells attract
//! others via social component (producer-scrounger dynamics).
//!
//! Run: cargo run --example large_arena --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::spatial_hash::SpatialHash;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;
const ARENA: f64 = 1000.0;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

#[derive(Clone, Copy)]
enum Ctrl {
    Fep,
    WellOnly,
    Greedy,
    Random,
}

fn run_experiment(ctrl: Ctrl, seed: u64) -> (f64, f64, f64) {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    // Wells scattered FAR from center in 1000×1000 arena
    let mut rng = seed;
    let wells: Vec<SVector<f64, 2>> = (0..3)
        .map(|_| {
            SVector::from([
                (rng_f64(&mut rng) - 0.5) * ARENA * 0.8,
                (rng_f64(&mut rng) - 0.5) * ARENA * 0.8,
            ])
        })
        .collect();
    let mut well_remaining = vec![5000.0f64; 3];

    let mut handles = Vec::new();
    // All agents start clustered near center
    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 50.0;
        let y = (rng_f64(&mut rng) - 0.5) * 50.0;
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

        // Wells only visible within 200 units (FEP gradient limit)
        let adata: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position().0,
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
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

            // Only wells within 200 units are visible
            let wdata: Vec<_> = wells
                .iter()
                .zip(well_remaining.iter())
                .filter(|(w, &r)| r > 0.0 && (pos - *w).norm() < 200.0)
                .map(|(&p, &r)| (p, (r / 5000.0).min(1.0)))
                .collect();

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
                    if let Some((wp, _)) = wdata.iter().min_by(|(a, _), (b, _)| {
                        (a - pos).norm().partial_cmp(&(b - pos).norm()).unwrap()
                    }) {
                        let delta = wp - pos;
                        let d = delta.norm();
                        if d > 1.0 {
                            delta / d * 20.0
                        } else {
                            SVector::zeros()
                        }
                    } else {
                        // No visible well — explore
                        let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                        SVector::from([angle.cos() * 20.0, angle.sin() * 20.0])
                    }
                }
                Ctrl::Greedy => {
                    let mut best_dir = SVector::zeros();
                    let mut best_gain = f64::NEG_INFINITY;
                    for ai in 0..8 {
                        let angle = ai as f64 * std::f64::consts::TAU / 8.0;
                        let td = SVector::from([angle.cos(), angle.sin()]);
                        let tp = pos + td * 10.0;
                        let mut gain = 0.0;
                        for (wp, wr) in &wdata {
                            if (tp - wp).norm() < 35.0 {
                                gain += consciousness.constants.energy_well_regen_rate * wr;
                            }
                        }
                        for (ap, ah) in &adata {
                            let d = (tp - ap).norm();
                            if d > 2.0 && d < consciousness.constants.harmony_range {
                                let res = HarmonyField::<2>::resonance(&e.harmony_activations, ah);
                                if res > 0.5 {
                                    gain += consciousness.constants.harmony_resonance_regen_rate
                                        * (res - 0.5);
                                }
                            }
                        }
                        if gain > best_gain {
                            best_gain = gain;
                            best_dir = td;
                        }
                    }
                    if best_gain <= 0.0 {
                        let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                        SVector::from([angle.cos() * 20.0, angle.sin() * 20.0])
                    } else {
                        best_dir * 20.0
                    }
                }
                Ctrl::Random => {
                    let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
                    SVector::from([angle.cos() * 20.0, angle.sin() * 20.0])
                }
            };
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = vel;
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
    (alive, energy, coop as f64)
}

fn main() {
    eprintln!("=== Large Arena (1000×1000) ===");
    eprintln!("Does scale break by-product mutualism?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds, arena {ARENA}×{ARENA}");

    println!("controller,seed,alive,energy,cooperation");

    let ctrls = [
        (Ctrl::Fep, "FEP"),
        (Ctrl::WellOnly, "WELL"),
        (Ctrl::Greedy, "GREEDY"),
        (Ctrl::Random, "RANDOM"),
    ];
    let mut all: Vec<(&str, Vec<f64>)> = Vec::new();

    for &(ctrl, name) in &ctrls {
        let mut alive_vec = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let (alive, energy, coop) = run_experiment(ctrl, seed);
            println!("{name},{seed},{alive:.1},{energy:.1},{coop:.0}");
            alive_vec.push(alive);
        }
        let mean = alive_vec.iter().sum::<f64>() / SEEDS as f64;
        eprintln!("  → {name}: alive={mean:.1}");
        all.push((name, alive_vec));
    }

    eprintln!("\n── Large Arena Results ──");
    for (name, alive) in &all {
        let mean = alive.iter().sum::<f64>() / alive.len() as f64;
        eprintln!("  {:6}: alive={mean:.1}/{AGENTS}", name);
    }

    // KEY: Does FEP help in large arena?
    let fep = &all[0].1;
    let well = &all[1].1;
    let random = &all[3].1;
    let d_fw = cohens_d(well, fep);
    let d_fr = cohens_d(random, fep);
    let (_, _, p_fw) = mann_whitney_u(well, fep);

    eprintln!("\n── FEP vs WELL in large arena ──");
    eprintln!("  d={d_fw:.3}, p={p_fw:.4}");
    if d_fw < -0.5 {
        eprintln!("  FEP WINS AT SCALE: Social gradient helps find wells in large arenas.");
    } else if d_fw > 0.5 {
        eprintln!("  WELL STILL WINS: Even at 1000×1000, well-seeking dominates.");
    } else {
        eprintln!("  NO DIFFERENCE at this scale.");
    }

    eprintln!("\n── FEP vs RANDOM ──");
    eprintln!("  d={d_fr:.3}");
    if d_fr < -0.5 {
        eprintln!("  FEP outperforms RANDOM: social gradient provides information value.");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
