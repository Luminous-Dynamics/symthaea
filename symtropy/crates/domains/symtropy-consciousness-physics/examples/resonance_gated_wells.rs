// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Resonance-Gated Wells — does COMPATIBILITY make the FEP gradient indispensable?
//!
//! F51 showed that threshold wells (count-gated) don't require social seeking
//! because all agents naturally co-locate at the same wells. The problem:
//! co-location is free when everyone wants the same thing.
//!
//! This experiment changes the lock mechanism: wells only dispense energy to
//! an agent if that agent has at least one RESONANT partner (resonance > 0.5)
//! within the well radius. Being at the well isn't enough — you must be there
//! WITH a compatible partner.
//!
//! This tests whether HARMONY COMPATIBILITY (not just spatial proximity) is
//! a survival mechanism. WELL_ONLY agents arrive at wells but don't seek
//! compatible partners — they may co-locate with incompatible strangers.
//! FEP agents seek resonant partners, which should produce compatible pairs
//! at wells.
//!
//! 3 controllers × 3 resonance thresholds × 20 seeds
//!
//! Run: cargo run --example resonance_gated_wells --release

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

// Resonance thresholds for well access
const RES_THRESHOLDS: [f64; 3] = [0.0, 0.5, 0.7];

// 6 diverse harmony profiles — some pairs resonate, some don't
const HARMONY_PROFILES: [[f64; 9]; 6] = [
    [0.9, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.8, 0.5], // A: Stillness
    [0.8, 0.3, 0.2, 0.1, 0.3, 0.2, 0.2, 0.7, 0.5], // A': compatible with A
    [0.1, 0.2, 0.8, 0.2, 0.8, 0.1, 0.2, 0.1, 0.5], // B: Craft+Curiosity
    [0.2, 0.1, 0.7, 0.3, 0.7, 0.2, 0.1, 0.2, 0.5], // B': compatible with B
    [0.1, 0.8, 0.1, 0.2, 0.1, 0.2, 0.8, 0.1, 0.5], // C: Play+Kinship
    [0.2, 0.7, 0.2, 0.1, 0.2, 0.1, 0.7, 0.2, 0.5], // C': compatible with C
];

#[derive(Clone, Copy)]
enum Ctrl {
    Fep,
    WellOnly,
    Greedy,
}

struct ResGateResult {
    controller: &'static str,
    res_threshold: f64,
    alive: f64,
    energy: f64,
    cooperation: f64,
    gated_access_ticks: f64, // ticks where an agent was at well but gated out
}

fn run_experiment(ctrl: Ctrl, res_threshold: f64, seed: u64) -> ResGateResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![
        SVector::from([35.0, 10.0]),
        SVector::from([-30.0, -15.0]),
        SVector::from([5.0, -35.0]),
    ];
    let mut well_remaining = vec![3000.0f64; 3];
    let mut rng = seed;
    let mut handles = Vec::new();

    let ctrl_name = match ctrl {
        Ctrl::Fep => "FEP",
        Ctrl::WellOnly => "WELL",
        Ctrl::Greedy => "GREEDY",
    };

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
            // Use 6 diverse profiles — creates compatible PAIRS, not universal compatibility
            e.harmony_activations = HARMONY_PROFILES[i % 6];
        }
        handles.push(h);
    }

    let mut coop = 0u64;
    let mut gated_out = 0u64;

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
        let wdata: Vec<_> = wells
            .iter()
            .zip(well_remaining.iter())
            .filter(|&(_, &r)| r > 0.0)
            .map(|(&p, &r)| (p, (r / 3000.0).min(1.0)))
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

        // RESONANCE-GATED well access
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr_rate = consciousness.constants.energy_well_regen_rate;

        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e_ref) = consciousness.entities.get(&h) {
                let my_harmony = e_ref.harmony_activations;
                let collapsed = e_ref.energy.is_collapsed();
                drop(e_ref);

                if let Some(e) = consciousness.entities.get_mut(&h) {
                    e.energy.regenerate(ar * rm);
                }

                if !collapsed {
                    if let Some(b) = world.body(h) {
                        let pos = b.position();
                        for (wi, &w) in wells.iter().enumerate() {
                            if (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
                                // RESONANCE GATE: check if ANY other agent at this well
                                // is resonant with this agent
                                let has_resonant_partner = if res_threshold <= 0.0 {
                                    true // no gate
                                } else {
                                    handles.iter().any(|&other| {
                                        if other == h {
                                            return false;
                                        }
                                        let other_at_well = world
                                            .body(other)
                                            .map(|ob| {
                                                (SVector::from(b.position()) - w).norm() < 35.0
                                            })
                                            .unwrap_or(false);
                                        if !other_at_well {
                                            return false;
                                        }
                                        let other_harmony = consciousness
                                            .entities
                                            .get(&other)
                                            .map(|oe| oe.harmony_activations);
                                        match other_harmony {
                                            Some(oh) => {
                                                HarmonyField::<2>::resonance(&my_harmony, &oh)
                                                    >= res_threshold
                                            }
                                            None => false,
                                        }
                                    })
                                };

                                if has_resonant_partner {
                                    if let Some(e) = consciousness.entities.get_mut(&h) {
                                        let d = wr_rate.min(well_remaining[wi]);
                                        e.energy.regenerate(d);
                                        well_remaining[wi] -= d;
                                    }
                                } else {
                                    // At well but GATED OUT — no resonant partner present
                                    gated_out += 1;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Resonance cooperation (passive)
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

    ResGateResult {
        controller: ctrl_name,
        res_threshold,
        alive,
        energy,
        cooperation: coop as f64,
        gated_access_ticks: gated_out as f64,
    }
}

fn main() {
    eprintln!("=== Resonance-Gated Wells ===");
    eprintln!("Wells require a COMPATIBLE partner, not just any neighbor");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");
    eprintln!("6 diverse harmony profiles (3 compatible pairs)");

    println!("controller,res_threshold,seed,alive,energy,cooperation,gated_ticks");

    let ctrls = [
        (Ctrl::WellOnly, "WELL"),
        (Ctrl::Fep, "FEP"),
        (Ctrl::Greedy, "GREEDY"),
    ];
    let mut all: Vec<(&str, f64, Vec<ResGateResult>)> = Vec::new();

    for &(ctrl, cname) in &ctrls {
        for &thresh in &RES_THRESHOLDS {
            let mut results = Vec::new();
            for s in 0..SEEDS {
                let seed = 42 + s as u64 * 997;
                eprintln!("  {cname} res≥{thresh:.1} seed={seed}...");
                let r = run_experiment(ctrl, thresh, seed);
                println!(
                    "{},{:.1},{seed},{:.1},{:.1},{:.0},{:.0}",
                    r.controller,
                    r.res_threshold,
                    r.alive,
                    r.energy,
                    r.cooperation,
                    r.gated_access_ticks
                );
                results.push(r);
            }
            let n = results.len() as f64;
            let mean_alive = results.iter().map(|r| r.alive).sum::<f64>() / n;
            let mean_gated = results.iter().map(|r| r.gated_access_ticks).sum::<f64>() / n;
            eprintln!(
                "  → {cname} res≥{thresh:.1}: alive={mean_alive:.1}, gated_out={mean_gated:.0}"
            );
            all.push((cname, thresh, results));
        }
    }

    // Summary table
    eprintln!("\n── Resonance-Gated Results ──");
    eprintln!("  Res Gate  WELL     FEP      GREEDY   FEP wins?");
    for &thresh in &RES_THRESHOLDS {
        let w = all
            .iter()
            .find(|(c, t, _)| *c == "WELL" && (*t - thresh).abs() < 0.01)
            .map(|(_, _, r)| r.iter().map(|r| r.alive).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let f = all
            .iter()
            .find(|(c, t, _)| *c == "FEP" && (*t - thresh).abs() < 0.01)
            .map(|(_, _, r)| r.iter().map(|r| r.alive).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let g = all
            .iter()
            .find(|(c, t, _)| *c == "GREEDY" && (*t - thresh).abs() < 0.01)
            .map(|(_, _, r)| r.iter().map(|r| r.alive).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let fep_wins = f > w + 1.0 && f > g + 1.0;
        eprintln!(
            "  ≥{:.1}       {:5.1}    {:5.1}    {:5.1}    {}",
            thresh,
            w,
            f,
            g,
            if fep_wins { "← FEP WINS" } else { "" }
        );
    }

    // Gated-out analysis
    eprintln!("\n── Gated-Out Events (at well but no resonant partner) ──");
    for &thresh in &RES_THRESHOLDS {
        if thresh < 0.01 {
            continue;
        } // no gate at 0
        let w = all
            .iter()
            .find(|(c, t, _)| *c == "WELL" && (*t - thresh).abs() < 0.01)
            .map(|(_, _, r)| r.iter().map(|r| r.gated_access_ticks).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let f = all
            .iter()
            .find(|(c, t, _)| *c == "FEP" && (*t - thresh).abs() < 0.01)
            .map(|(_, _, r)| r.iter().map(|r| r.gated_access_ticks).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let g = all
            .iter()
            .find(|(c, t, _)| *c == "GREEDY" && (*t - thresh).abs() < 0.01)
            .map(|(_, _, r)| r.iter().map(|r| r.gated_access_ticks).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        eprintln!(
            "  ≥{:.1}: WELL={w:.0}, FEP={f:.0}, GREEDY={g:.0} gated-out ticks",
            thresh
        );
    }

    // KEY TEST: At highest gate, does FEP outperform?
    let fep_07 = all
        .iter()
        .find(|(c, t, _)| *c == "FEP" && (*t - 0.7).abs() < 0.01);
    let well_07 = all
        .iter()
        .find(|(c, t, _)| *c == "WELL" && (*t - 0.7).abs() < 0.01);
    if let (Some((_, _, fep_r)), Some((_, _, well_r))) = (fep_07, well_07) {
        let f_alive: Vec<f64> = fep_r.iter().map(|r| r.alive).collect();
        let w_alive: Vec<f64> = well_r.iter().map(|r| r.alive).collect();
        let (_, _, p) = mann_whitney_u(&w_alive, &f_alive);
        let d = cohens_d(&w_alive, &f_alive);
        eprintln!("\n── KEY TEST: FEP vs WELL at resonance gate ≥0.7 ──");
        eprintln!(
            "  FEP={:.1}, WELL={:.1}",
            f_alive.iter().sum::<f64>() / f_alive.len() as f64,
            w_alive.iter().sum::<f64>() / w_alive.len() as f64
        );
        eprintln!("  Mann-Whitney p={p:.4}, Cohen's d={d:.3}");
        if d < -0.5 {
            eprintln!("  FEP WINS: Harmony compatibility IS a survival mechanism.");
            eprintln!("  Social seeking produces compatible pairs at wells.");
            eprintln!("  The FEP gradient is INDISPENSABLE under resonance-gated resources.");
        } else if d > 0.5 {
            eprintln!("  WELL STILL WINS: Agents naturally co-locate with compatible partners.");
        } else {
            eprintln!("  NO DIFFERENCE: Resonance gating doesn't differentiate controllers.");
        }
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
