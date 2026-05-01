// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Temporal Volatility — does resource instability make cooperation necessary?
//!
//! Findings 46-48 showed cooperation is by-product mutualism when wells are
//! static: agents survive equally with and without resonance. But in ecology,
//! social foraging advantages emerge under temporal resource volatility
//! (Giraldeau & Caraco 2000).
//!
//! When wells cycle on/off, agents at a deactivated well must survive on
//! resonance alone until they find an active well. If resonance is disabled,
//! they die. This makes cooperation a SURVIVAL mechanism, not a byproduct.
//!
//! 4 volatility conditions × 3 controllers × 2 regen modes × 20 seeds
//!
//! Run: cargo run --example temporal_volatility --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 20;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

#[derive(Clone, Copy)]
enum Volatility {
    Stable,
    Seasonal,
    Random,
    Migrating,
}
#[derive(Clone, Copy)]
enum Ctrl {
    Fep,
    WellOnly,
    Greedy,
}

fn run_experiment(vol: Volatility, ctrl: Ctrl, use_regen: bool, seed: u64) -> (f64, f64) {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let mut wells = vec![
        SVector::from([40.0, 10.0]),
        SVector::from([-30.0, -20.0]),
        SVector::from([10.0, -40.0]),
    ];
    let mut well_remaining = vec![1500.0f64; 3]; // moderate capacity
    let mut well_active = vec![true; 3];
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

    for tick in 0..TICKS {
        // Volatility dynamics
        match vol {
            Volatility::Stable => {} // wells always active
            Volatility::Seasonal => {
                // Cycle: 1500 ticks on, 1500 ticks off, staggered
                for i in 0..3 {
                    let phase = (tick + i * 500) % 3000;
                    well_active[i] = phase < 1500;
                }
            }
            Volatility::Random => {
                // Each well: 50% chance of flipping every 500 ticks
                if tick % 500 == 0 {
                    for i in 0..3 {
                        well_active[i] = rng_f64(&mut rng) > 0.5;
                    }
                }
            }
            Volatility::Migrating => {
                // Wells relocate every 1500 ticks
                if tick > 0 && tick % 1500 == 0 {
                    for i in 0..3 {
                        wells[i] = SVector::from([
                            (rng_f64(&mut rng) - 0.5) * 100.0,
                            (rng_f64(&mut rng) - 0.5) * 100.0,
                        ]);
                        well_remaining[i] = 1500.0; // refill on move
                        well_active[i] = true;
                    }
                }
            }
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

        // Only show active wells to agents
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
            .zip(well_active.iter())
            .filter(|((_, &r), &active)| r > 0.0 && active)
            .map(|((&p, &r), _)| (p, (r / 1500.0).min(1.0)))
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
                Ctrl::Greedy => {
                    let mut best_dir = SVector::zeros();
                    let mut best_gain = f64::NEG_INFINITY;
                    for ai in 0..8 {
                        let angle = ai as f64 * std::f64::consts::TAU / 8.0;
                        let td = SVector::from([angle.cos(), angle.sin()]);
                        let tp = pos + td * 5.0;
                        let mut gain = 0.0;
                        for (wp, wr) in &wdata {
                            if *wr < 0.01 {
                                continue;
                            }
                            if (tp - wp).norm() < 35.0 {
                                gain += consciousness.constants.energy_well_regen_rate * wr;
                            }
                        }
                        if gain > best_gain {
                            best_gain = gain;
                            best_dir = td;
                        }
                    }
                    best_dir * 20.0
                }
            };
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = vel;
            }
        }

        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr_rate = consciousness.constants.energy_well_regen_rate;
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
                        if well_active[wi] && (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
                            let d = wr_rate.min(well_remaining[wi]);
                            e.energy.regenerate(d);
                            well_remaining[wi] -= d;
                            break;
                        }
                    }
                }
            }
        }

        if use_regen {
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
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
    (alive, energy)
}

fn main() {
    eprintln!("=== Temporal Volatility ===");
    eprintln!("Does resource instability make cooperation necessary?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    println!("volatility,controller,regen,seed,alive,energy");

    let vols = [
        (Volatility::Stable, "STABLE"),
        (Volatility::Seasonal, "SEASONAL"),
        (Volatility::Random, "RANDOM"),
        (Volatility::Migrating, "MIGRATING"),
    ];
    let ctrls = [
        (Ctrl::WellOnly, "WELL"),
        (Ctrl::Fep, "FEP"),
        (Ctrl::Greedy, "GREEDY"),
    ];

    // Focus on WELL_ONLY ± REGEN across volatility levels (the key test)
    eprintln!("\n── WELL_ONLY: Does volatility make resonance necessary? ──");
    eprintln!("  Volatility    REGEN   NO_REGEN  Δ       d       Necessary?");

    for &(vol, vname) in &vols {
        let mut regen_alive = Vec::new();
        let mut noregen_alive = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            let (a_r, _) = run_experiment(vol, Ctrl::WellOnly, true, seed);
            let (a_nr, _) = run_experiment(vol, Ctrl::WellOnly, false, seed);
            println!("{vname},WELL,REGEN,{seed},{a_r:.1},0");
            println!("{vname},WELL,NO_REGEN,{seed},{a_nr:.1},0");
            regen_alive.push(a_r);
            noregen_alive.push(a_nr);
        }
        let r_mean = regen_alive.iter().sum::<f64>() / SEEDS as f64;
        let nr_mean = noregen_alive.iter().sum::<f64>() / SEEDS as f64;
        let delta = r_mean - nr_mean;
        let d = cohens_d(&regen_alive, &noregen_alive);
        let (_, _, p) = mann_whitney_u(&regen_alive, &noregen_alive);
        let necessary = delta > 2.0 && d.abs() > 0.5;
        let marker = if necessary { " ← NECESSARY" } else { "" };
        eprintln!(
            "  {:10}    {:5.1}   {:5.1}     {:+5.1}   {:+.2}   {marker}",
            vname, r_mean, nr_mean, delta, d
        );
    }

    // Also test FEP under volatility
    eprintln!("\n── FEP: Does volatility help the social gradient? ──");
    eprintln!("  Volatility    FEP+R   WELL+R    Δ       d");
    for &(vol, vname) in &vols {
        let mut fep_alive = Vec::new();
        let mut well_alive = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            let (a_f, _) = run_experiment(vol, Ctrl::Fep, true, seed);
            let (a_w, _) = run_experiment(vol, Ctrl::WellOnly, true, seed);
            println!("{vname},FEP,REGEN,{seed},{a_f:.1},0");
            fep_alive.push(a_f);
            well_alive.push(a_w);
        }
        let f_mean = fep_alive.iter().sum::<f64>() / SEEDS as f64;
        let w_mean = well_alive.iter().sum::<f64>() / SEEDS as f64;
        let d = cohens_d(&well_alive, &fep_alive);
        eprintln!(
            "  {:10}    {:5.1}   {:5.1}     {:+5.1}   {:+.2}{}",
            vname,
            f_mean,
            w_mean,
            f_mean - w_mean,
            d,
            if f_mean > w_mean + 2.0 {
                "  ← FEP WINS"
            } else {
                ""
            }
        );
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
