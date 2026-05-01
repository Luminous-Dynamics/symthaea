// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cooperation Decomposition — is passive cooperation load-bearing?
//!
//! Finding 46 showed that WELL_ONLY agents (no social seeking) cooperate
//! MORE and survive BETTER than FEP_GRADIENT agents. But do they NEED
//! the cooperation? This experiment decomposes the answer.
//!
//! 6 conditions crossing controller × resonance:
//! 1. FEP + REGEN: original baseline
//! 2. FEP + NO_REGEN: social seeking without cooperation benefit
//! 3. WELL_ONLY + REGEN: well seeking with passive cooperation
//! 4. WELL_ONLY + NO_REGEN: well seeking without any cooperation
//! 5. GREEDY + REGEN: greedy with passive cooperation
//! 6. GREEDY + NO_REGEN: greedy without cooperation
//!
//! If WELL_ONLY+NO_REGEN survives as well as WELL_ONLY+REGEN:
//!   → cooperation is epiphenomenal (Thesis A)
//! If WELL_ONLY+NO_REGEN dies while WELL_ONLY+REGEN survives:
//!   → passive cooperation is load-bearing (Thesis B)
//!
//! Run: cargo run --example cooperation_decomposition --release

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
    Fep,
    WellOnly,
    Greedy,
}
#[derive(Clone, Copy)]
enum RegenMode {
    WithRegen,
    NoRegen,
}

struct DecompResult {
    condition: String,
    alive: f64,
    energy: f64,
    cooperation: f64,
    well_energy_gained: f64,
    resonance_energy_gained: f64,
}

fn run_experiment(ctrl: Controller, regen: RegenMode, seed: u64) -> DecompResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![2500.0f64; 2];
    let mut rng = seed;
    let mut handles = Vec::new();
    let use_regen = matches!(regen, RegenMode::WithRegen);

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

    let ctrl_name = match ctrl {
        Controller::Fep => "FEP",
        Controller::WellOnly => "WELL",
        Controller::Greedy => "GREEDY",
    };
    let regen_name = match regen {
        RegenMode::WithRegen => "REGEN",
        RegenMode::NoRegen => "NO_REGEN",
    };
    let condition = format!("{}+{}", ctrl_name, regen_name);

    let mut coop = 0u64;
    let mut total_well_energy = 0.0f64;
    let mut total_res_energy = 0.0f64;

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
                Controller::Fep => {
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
                Controller::Greedy => {
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
                        if use_regen {
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

        // Maintenance + well regen (always on)
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
                            total_well_energy += d;
                            break;
                        }
                    }
                }
            }
        }

        // Resonance regeneration (only if use_regen)
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
                    coop += 1;
                    if use_regen {
                        let rg = consciousness.constants.harmony_resonance_regen_rate
                            * (res - 0.5)
                            * 2.0;
                        if let Some(e) = consciousness.entities.get_mut(&ha) {
                            e.energy.regenerate(rg);
                        }
                        if let Some(e) = consciousness.entities.get_mut(&hb) {
                            e.energy.regenerate(rg);
                        }
                        total_res_energy += rg * 2.0;
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

    DecompResult {
        condition,
        alive,
        energy,
        cooperation: coop as f64,
        well_energy_gained: total_well_energy,
        resonance_energy_gained: total_res_energy,
    }
}

fn main() {
    eprintln!("=== Cooperation Decomposition ===");
    eprintln!("Is passive cooperation load-bearing, or epiphenomenal?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds, 6 conditions");

    println!("condition,seed,alive,energy,cooperation,well_energy,resonance_energy");

    let conditions: Vec<(Controller, RegenMode, &str)> = vec![
        (Controller::Fep, RegenMode::WithRegen, "FEP+REGEN"),
        (Controller::Fep, RegenMode::NoRegen, "FEP+NO_REGEN"),
        (Controller::WellOnly, RegenMode::WithRegen, "WELL+REGEN"),
        (Controller::WellOnly, RegenMode::NoRegen, "WELL+NO_REGEN"),
        (Controller::Greedy, RegenMode::WithRegen, "GREEDY+REGEN"),
        (Controller::Greedy, RegenMode::NoRegen, "GREEDY+NO_REGEN"),
    ];
    let mut all: Vec<(&str, Vec<DecompResult>)> = Vec::new();

    for &(ctrl, regen, name) in &conditions {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(ctrl, regen, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.0},{:.0},{:.0}",
                r.condition,
                r.alive,
                r.energy,
                r.cooperation,
                r.well_energy_gained,
                r.resonance_energy_gained
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}, well_E={:.0}J, res_E={:.0}J",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.well_energy_gained).sum::<f64>() / n,
            results
                .iter()
                .map(|r| r.resonance_energy_gained)
                .sum::<f64>()
                / n
        );
        all.push((name, results));
    }

    eprintln!("\n── Cooperation Decomposition ──");
    eprintln!("  Condition          Alive  Energy  Coop     Well_E    Res_E");
    for (name, results) in &all {
        let n = results.len() as f64;
        eprintln!(
            "  {:18} {:5.1}  {:5.1}J  {:8.0}  {:7.0}J  {:7.0}J",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.energy).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n,
            results.iter().map(|r| r.well_energy_gained).sum::<f64>() / n,
            results
                .iter()
                .map(|r| r.resonance_energy_gained)
                .sum::<f64>()
                / n
        );
    }

    // THE KEY TEST: WELL+REGEN vs WELL+NO_REGEN
    let wr = &all[2].1; // WELL+REGEN
    let wnr = &all[3].1; // WELL+NO_REGEN
    let wr_alive: Vec<f64> = wr.iter().map(|r| r.alive).collect();
    let wnr_alive: Vec<f64> = wnr.iter().map(|r| r.alive).collect();
    let (_, _, p) = mann_whitney_u(&wr_alive, &wnr_alive);
    let d = cohens_d(&wr_alive, &wnr_alive);

    eprintln!("\n── THE KEY TEST: WELL+REGEN vs WELL+NO_REGEN ──");
    eprintln!("  Mann-Whitney p={p:.4}, Cohen's d={d:.3}");

    let wr_mean = wr_alive.iter().sum::<f64>() / wr_alive.len() as f64;
    let wnr_mean = wnr_alive.iter().sum::<f64>() / wnr_alive.len() as f64;

    if (wr_mean - wnr_mean).abs() < 2.0 && p > 0.05 {
        eprintln!("\n  THESIS A: Cooperation is EPIPHENOMENAL.");
        eprintln!("  Wells alone sustain agents. Passive resonance is noise.");
        eprintln!(
            "  WELL+REGEN={wr_mean:.1}, WELL+NO_REGEN={wnr_mean:.1} (no significant difference)"
        );
    } else if wr_mean > wnr_mean + 2.0 {
        eprintln!("\n  THESIS B: Passive cooperation is LOAD-BEARING.");
        eprintln!("  Agents co-located at wells benefit measurably from resonance.");
        eprintln!("  WELL+REGEN={wr_mean:.1} > WELL+NO_REGEN={wnr_mean:.1}");
    } else {
        eprintln!(
            "\n  INCONCLUSIVE: WELL+REGEN={wr_mean:.1}, WELL+NO_REGEN={wnr_mean:.1}, p={p:.4}"
        );
    }

    // Energy source decomposition
    eprintln!("\n── Energy Source Decomposition ──");
    for (name, results) in &all {
        let n = results.len() as f64;
        let well = results.iter().map(|r| r.well_energy_gained).sum::<f64>() / n;
        let res = results
            .iter()
            .map(|r| r.resonance_energy_gained)
            .sum::<f64>()
            / n;
        let total = well + res;
        let pct_well = if total > 0.0 {
            well / total * 100.0
        } else {
            0.0
        };
        let pct_res = if total > 0.0 {
            res / total * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {:18}: {pct_well:5.1}% wells, {pct_res:5.1}% resonance",
            name
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
