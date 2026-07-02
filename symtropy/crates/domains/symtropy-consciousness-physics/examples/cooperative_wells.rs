// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cooperative Wells — obligate mutualism via threshold public goods.
//!
//! Findings 46-50 showed cooperation is epiphenomenal when wells provide
//! individual access. This experiment converts wells into an n-player
//! Stag Hunt: wells ONLY dispense energy when ≥N agents are co-located.
//!
//! This enforces "obligate mutualism" — solitary agents literally cannot
//! access resources. The FEP gradient's social drive, previously a
//! metabolic liability, should become indispensable because it naturally
//! produces the synchronized co-location that threshold wells require.
//!
//! 3 controllers × 4 thresholds × 20 seeds = 240 runs
//!
//! Run: cargo run --example cooperative_wells --release

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

const THRESHOLDS: [usize; 4] = [1, 2, 3, 4];

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

struct CoopWellResult {
    controller: &'static str,
    threshold: usize,
    alive: f64,
    energy: f64,
    cooperation: f64,
    well_unlocks: f64, // ticks where threshold was met at any well
}

fn run_experiment(ctrl: Ctrl, threshold: usize, seed: u64) -> CoopWellResult {
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
            e.harmony_activations = HARMONY_PROFILES[i % 4];
        }
        handles.push(h);
    }

    let mut coop = 0u64;
    let mut well_unlock_ticks = 0u64;

    for _tick in 0..TICKS {
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

        // Movement
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

        // Count agents at each well for threshold check
        let well_counts: Vec<usize> = wells
            .iter()
            .map(|w| {
                handles
                    .iter()
                    .filter(|&&h| {
                        world
                            .body(h)
                            .map(|b| (b.position() - w).norm() < 35.0)
                            .unwrap_or(false)
                            && consciousness
                                .entities
                                .get(&h)
                                .map(|e| !e.energy.is_collapsed())
                                .unwrap_or(false)
                    })
                    .count()
            })
            .collect();

        // Track unlock events
        if well_counts.iter().any(|&c| c >= threshold) {
            well_unlock_ticks += 1;
        }

        // Maintenance + THRESHOLD-GATED well regen
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

                // COOPERATIVE WELL ACCESS: only dispense if threshold met
                if let Some(b) = world.body(h) {
                    let pos = b.position();
                    for (wi, &w) in wells.iter().enumerate() {
                        if (pos - w).norm() < 35.0 && well_remaining[wi] > 0.0 {
                            // THE KEY MECHANISM: well only works if enough agents present
                            if well_counts[wi] >= threshold {
                                let d = wr_rate.min(well_remaining[wi]);
                                e.energy.regenerate(d);
                                well_remaining[wi] -= d;
                            }
                            // If below threshold: agent is AT the well but gets NOTHING
                            break;
                        }
                    }
                }
            }
        }

        // Resonance cooperation (always active — passive benefit)
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

    CoopWellResult {
        controller: ctrl_name,
        threshold,
        alive,
        energy,
        cooperation: coop as f64,
        well_unlocks: well_unlock_ticks as f64,
    }
}

fn main() {
    eprintln!("=== Cooperative Wells ===");
    eprintln!("Obligate mutualism: wells require ≥N agents to operate");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");
    eprintln!("3 controllers × 4 thresholds = 12 conditions");

    println!("controller,threshold,seed,alive,energy,cooperation,well_unlocks");

    let ctrls = [
        (Ctrl::WellOnly, "WELL"),
        (Ctrl::Fep, "FEP"),
        (Ctrl::Greedy, "GREEDY"),
    ];
    let mut all: Vec<(&str, usize, Vec<CoopWellResult>)> = Vec::new();

    for &(ctrl, cname) in &ctrls {
        for &thresh in &THRESHOLDS {
            let mut results = Vec::new();
            for s in 0..SEEDS {
                let seed = 42 + s as u64 * 997;
                eprintln!("  {cname} thresh={thresh} seed={seed}...");
                let r = run_experiment(ctrl, thresh, seed);
                println!(
                    "{},{},{seed},{:.1},{:.1},{:.0},{:.0}",
                    r.controller, r.threshold, r.alive, r.energy, r.cooperation, r.well_unlocks
                );
                results.push(r);
            }
            let n = results.len() as f64;
            eprintln!(
                "  → {cname} thresh={thresh}: alive={:.1}, unlocks={:.0}",
                results.iter().map(|r| r.alive).sum::<f64>() / n,
                results.iter().map(|r| r.well_unlocks).sum::<f64>() / n
            );
            all.push((cname, thresh, results));
        }
    }

    // Summary table
    eprintln!("\n── Cooperative Well Results ──");
    eprintln!("  Threshold  WELL     FEP      GREEDY   FEP wins?");
    for &thresh in &THRESHOLDS {
        let w = all
            .iter()
            .find(|(c, t, _)| *c == "WELL" && *t == thresh)
            .map(|(_, _, r)| r.iter().map(|r| r.alive).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let f = all
            .iter()
            .find(|(c, t, _)| *c == "FEP" && *t == thresh)
            .map(|(_, _, r)| r.iter().map(|r| r.alive).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let g = all
            .iter()
            .find(|(c, t, _)| *c == "GREEDY" && *t == thresh)
            .map(|(_, _, r)| r.iter().map(|r| r.alive).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let fep_wins = f > w + 1.0 && f > g + 1.0;
        eprintln!(
            "  {:5}       {:5.1}    {:5.1}    {:5.1}    {}",
            thresh,
            w,
            f,
            g,
            if fep_wins { "← YES" } else { "" }
        );
    }

    // Statistical test at threshold=3 (expected crossover point)
    let fep_3 = all.iter().find(|(c, t, _)| *c == "FEP" && *t == 3);
    let well_3 = all.iter().find(|(c, t, _)| *c == "WELL" && *t == 3);
    if let (Some((_, _, fep_r)), Some((_, _, well_r))) = (fep_3, well_3) {
        let f_alive: Vec<f64> = fep_r.iter().map(|r| r.alive).collect();
        let w_alive: Vec<f64> = well_r.iter().map(|r| r.alive).collect();
        let (_, _, p) = mann_whitney_u(&w_alive, &f_alive);
        let d = cohens_d(&w_alive, &f_alive);
        eprintln!("\n── KEY TEST: FEP vs WELL at threshold=3 ──");
        eprintln!(
            "  FEP={:.1}, WELL={:.1}",
            f_alive.iter().sum::<f64>() / f_alive.len() as f64,
            w_alive.iter().sum::<f64>() / w_alive.len() as f64
        );
        eprintln!("  Mann-Whitney p={p:.4}, Cohen's d={d:.3}");
        if d < -0.5 {
            eprintln!("  FEP WINS: Social drive is INDISPENSABLE under obligate mutualism.");
            eprintln!("  The Stag Hunt environment makes the FEP gradient a survival mechanism.");
        } else if d > 0.5 {
            eprintln!("  WELL STILL WINS: Even threshold wells don't require social coordination.");
        } else {
            eprintln!("  NO DIFFERENCE: Both controllers handle threshold wells similarly.");
        }
    }

    // Well unlock analysis
    eprintln!("\n── Well Unlock Rate (ticks where threshold met) ──");
    for &thresh in &THRESHOLDS {
        let w_unlocks = all
            .iter()
            .find(|(c, t, _)| *c == "WELL" && *t == thresh)
            .map(|(_, _, r)| r.iter().map(|r| r.well_unlocks).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let f_unlocks = all
            .iter()
            .find(|(c, t, _)| *c == "FEP" && *t == thresh)
            .map(|(_, _, r)| r.iter().map(|r| r.well_unlocks).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        let g_unlocks = all
            .iter()
            .find(|(c, t, _)| *c == "GREEDY" && *t == thresh)
            .map(|(_, _, r)| r.iter().map(|r| r.well_unlocks).sum::<f64>() / r.len() as f64)
            .unwrap_or(0.0);
        eprintln!(
            "  thresh={thresh}: WELL={w_unlocks:.0}, FEP={f_unlocks:.0}, GREEDY={g_unlocks:.0} unlock-ticks (of {TICKS})"
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
