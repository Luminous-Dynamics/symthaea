// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Falsification Attempt — can we find parameters where cooperation FAILS?
//!
//! Searches for the hardest possible parameter regime: high maintenance,
//! low range, low regen, large arena, distant wells. If cooperation still
//! emerges, that's a universality result. If it dies, we've found the
//! boundary of the engine's claims.
//!
//! 5 conditions from easy to brutal:
//! 1. EASY: low maintenance, close wells, wide range
//! 2. NORMAL: research() defaults
//! 3. HARSH: high maintenance, moderate range
//! 4. BRUTAL: extreme maintenance, tiny range, no ambient regen
//! 5. IMPOSSIBLE: maintenance > max possible regen rate
//!
//! Run: cargo run --example falsification --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni};
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

struct FalsifyResult {
    condition: &'static str,
    alive: f64,
    cooperation: f64,
    clustering: f64,
}

fn run_condition(
    maintenance: f64,
    range: f64,
    regen: f64,
    ambient: f64,
    well_dist: f64,
    well_cap: f64,
    arena: f64,
    seed: u64,
) -> FalsifyResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = maintenance;
    constants.harmony_range = range;
    constants.harmony_resonance_regen_rate = regen;
    constants.ambient_regen_rate = ambient;
    consciousness.constants = constants;

    let wells = vec![
        SVector::from([well_dist, 0.0]),
        SVector::from([-well_dist, 0.0]),
    ];
    let mut well_remaining = vec![well_cap; 2];

    let mut rng = seed;
    let mut handles = Vec::new();
    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * arena;
        let y = (rng_f64(&mut rng) - 0.5) * arena;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.05;
        }
        consciousness.register(h, consciousness.constants.initial_energy, range);
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
                    d > 2.0 && d < range
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

        let rm = consciousness.resource_regeneration_multiplier();
        let wr = consciousness.constants.energy_well_regen_rate;
        for &h in handles.iter() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, maintenance * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ambient * rm);
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
                    (Some(a), Some(b)) => a.position().distance(b.position()) < range,
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
                    let rg = regen * (res - 0.5) * 2.0;
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

    FalsifyResult {
        condition: "",
        alive,
        cooperation: coop as f64,
        clustering,
    }
}

fn main() {
    eprintln!("=== Falsification Attempt ===");
    eprintln!("Can we find parameters where cooperation completely fails?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    //                        maint  range regen  ambient well_d well_c arena  name
    let conditions: Vec<(f64, f64, f64, f64, f64, f64, f64, &str)> = vec![
        (0.10, 60.0, 0.09, 0.01, 20.0, 5000.0, 80.0, "EASY"),
        (0.20, 40.0, 0.06, 0.005, 30.0, 2500.0, 100.0, "NORMAL"),
        (0.40, 25.0, 0.03, 0.002, 50.0, 1500.0, 150.0, "HARSH"),
        (0.70, 15.0, 0.015, 0.0, 80.0, 800.0, 200.0, "BRUTAL"),
        (1.50, 10.0, 0.005, 0.0, 100.0, 500.0, 300.0, "IMPOSSIBLE"),
    ];

    println!("condition,seed,alive,cooperation,clustering");

    let mut all_results: Vec<(&str, Vec<FalsifyResult>)> = Vec::new();

    for &(maint, range, regen, ambient, well_d, well_c, arena, name) in &conditions {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let mut r = run_condition(maint, range, regen, ambient, well_d, well_c, arena, seed);
            r.condition = name;
            println!(
                "{name},{seed},{:.1},{:.0},{:.2}",
                r.alive, r.cooperation, r.clustering
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}/{AGENTS}, coop={:.0}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n
        );
        all_results.push((name, results));
    }

    eprintln!("\n── Falsification Gradient ──");
    eprintln!("  Condition    Alive   Coop       Clustering  Cooperating?");
    for (name, results) in &all_results {
        let n = results.len() as f64;
        let alive = results.iter().map(|r| r.alive).sum::<f64>() / n;
        let coop = results.iter().map(|r| r.cooperation).sum::<f64>() / n;
        let clust = results.iter().map(|r| r.clustering).sum::<f64>() / n;
        let verdict = if alive < 1.0 {
            "DEAD"
        } else if coop < 100.0 {
            "NO COOP"
        } else if alive < AGENTS as f64 * 0.3 {
            "struggling"
        } else {
            "cooperating"
        };
        eprintln!(
            "  {:10} {:5.1}    {:9.0}   {:6.2}      {verdict}",
            name, alive, coop, clust
        );
    }

    // Did we kill cooperation?
    let impossible = &all_results.last().unwrap().1;
    let impossible_alive: f64 =
        impossible.iter().map(|r| r.alive).sum::<f64>() / impossible.len() as f64;
    let impossible_coop: f64 =
        impossible.iter().map(|r| r.cooperation).sum::<f64>() / impossible.len() as f64;

    eprintln!("\n── Verdict ──");
    if impossible_alive < 1.0 && impossible_coop < 100.0 {
        eprintln!("  FALSIFIED: Cooperation completely fails under IMPOSSIBLE conditions.");
        eprintln!(
            "  The engine's cooperation is NOT universal — it requires minimum range and regen."
        );
    } else if impossible_alive < AGENTS as f64 * 0.2 {
        eprintln!(
            "  NEARLY FALSIFIED: Most agents die under extreme conditions ({:.0}% survive).",
            impossible_alive / AGENTS as f64 * 100.0
        );
        eprintln!("  Cooperation is degraded but not eliminated.");
    } else {
        eprintln!("  NOT FALSIFIED: Cooperation persists even under IMPOSSIBLE conditions ({:.0}% survive).", impossible_alive / AGENTS as f64 * 100.0);
        eprintln!("  This is either a universality result or a design flaw.");
    }

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
