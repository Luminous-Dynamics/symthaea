// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scarcity Threshold — where does cooperation become NECESSARY?
//!
//! Finding 47 showed cooperation is epiphenomenal at 2500J wells.
//! This experiment finds the well capacity where REGEN > NO_REGEN —
//! the boundary where passive cooperation transitions from noise
//! to survival mechanism.
//!
//! Crosses 3 controllers × 5 well capacities × 2 regen modes × 20 seeds.
//!
//! Run: cargo run --example scarcity_threshold --release

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
const SEEDS: usize = 20;

const WELL_CAPACITIES: [f64; 5] = [2500.0, 1500.0, 1000.0, 500.0, 250.0];

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
}

fn run_experiment(ctrl: Ctrl, well_cap: f64, use_regen: bool, seed: u64) -> (f64, f64) {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![well_cap; 2];
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
            .map(|(&p, &r)| (p, (r / well_cap).min(1.0)))
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
                        if (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
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
    eprintln!("=== Scarcity Threshold ===");
    eprintln!("Where does cooperation become NECESSARY?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");
    eprintln!("3 controllers × 5 capacities × 2 regen modes = 30 conditions");

    println!("controller,well_cap,regen,seed,alive,energy");

    let ctrls = [
        (Ctrl::WellOnly, "WELL"),
        (Ctrl::Fep, "FEP"),
        (Ctrl::Greedy, "GREEDY"),
    ];

    // Store results for phase diagram
    let mut phase_data: Vec<(&str, f64, bool, Vec<f64>)> = Vec::new(); // (ctrl, cap, regen, alive_vec)

    for &(ctrl, cname) in &ctrls {
        for &cap in &WELL_CAPACITIES {
            for &regen in &[true, false] {
                let rname = if regen { "REGEN" } else { "NO_REGEN" };
                let mut alive_vec = Vec::new();
                for s in 0..SEEDS {
                    let seed = 42 + s as u64 * 997;
                    let (alive, energy) = run_experiment(ctrl, cap, regen, seed);
                    println!("{cname},{cap:.0},{rname},{seed},{alive:.1},{energy:.1}");
                    alive_vec.push(alive);
                }
                let n = alive_vec.len() as f64;
                let mean = alive_vec.iter().sum::<f64>() / n;
                eprintln!("  {cname}+{rname} cap={cap:.0}: alive={mean:.1}");
                phase_data.push((cname, cap, regen, alive_vec));
            }
        }
    }

    // Phase diagram: for each controller, find where REGEN > NO_REGEN
    eprintln!("\n── Phase Diagram: Cooperation Necessity ──");
    eprintln!("  Controller  WellCap  REGEN    NO_REGEN  Δ        d       Coop Necessary?");

    for ctrl_name in &["WELL", "FEP", "GREEDY"] {
        for &cap in &WELL_CAPACITIES {
            let regen_data = phase_data
                .iter()
                .find(|(c, cp, r, _)| c == ctrl_name && (*cp - cap).abs() < 1.0 && *r);
            let noregen_data = phase_data
                .iter()
                .find(|(c, cp, r, _)| c == ctrl_name && (*cp - cap).abs() < 1.0 && !*r);

            if let (Some((_, _, _, r_alive)), Some((_, _, _, nr_alive))) =
                (regen_data, noregen_data)
            {
                let r_mean = r_alive.iter().sum::<f64>() / r_alive.len() as f64;
                let nr_mean = nr_alive.iter().sum::<f64>() / nr_alive.len() as f64;
                let delta = r_mean - nr_mean;
                let d = cohens_d(r_alive, nr_alive);
                let necessary = delta > 2.0 && d.abs() > 0.5;
                let marker = if necessary {
                    " ← COOPERATION NECESSARY"
                } else {
                    ""
                };
                eprintln!(
                    "  {:8}     {:5.0}    {:5.1}    {:5.1}     {:+5.1}   {:+.2}   {marker}",
                    ctrl_name, cap, r_mean, nr_mean, delta, d
                );
            }
        }
        eprintln!();
    }

    eprintln!("=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
