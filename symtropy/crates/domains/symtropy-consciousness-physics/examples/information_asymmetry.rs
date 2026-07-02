// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Information Asymmetry — does knowing more help or harm the collective?
//!
//! Compares 3 conditions:
//! 1. EQUAL_SIGHT: All agents see wells within 200 units
//! 2. SCOUTS: 4 agents see 400 units, rest see 200
//! 3. BLIND: 4 agents can't see wells at all (0 range), rest normal
//!
//! Tests whether information advantages (scouts) benefit the group through
//! knowledge spillover, or whether information disadvantages (blind agents)
//! drag everyone down. Also tracks whether scouts/blind agents are
//! disproportionately alive or dead at the end.
//!
//! Run: cargo run --example information_asymmetry --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const SCOUTS: usize = 4;
const TICKS: usize = 10_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 8;
const NORMAL_WELL_RANGE: f64 = 200.0;
const SCOUT_WELL_RANGE: f64 = 400.0;

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

#[derive(Clone, Copy)]
enum InfoCondition {
    Equal,
    Scouts,
    Blind,
}

struct InfoResult {
    condition: &'static str,
    alive: f64,
    special_alive: f64, // scout/blind survival rate
    normal_alive: f64,  // normal agents survival rate
    energy: f64,
    clustering: f64,
}

fn run_experiment(condition: InfoCondition, seed: u64) -> InfoResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.25;
    consciousness.constants = constants;

    // Wells placed far apart to reward information
    let wells = vec![
        SVector::from([80.0, 0.0]),
        SVector::from([-80.0, 0.0]),
        SVector::from([0.0, 80.0]),
    ];
    let mut well_remaining = vec![2000.0f64; 3];

    let mut rng = seed;
    let mut handles = Vec::new();

    // Agent sight ranges
    let mut well_sight: Vec<f64> = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 60.0; // start clustered near center
        let y = (rng_f64(&mut rng) - 0.5) * 60.0;
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

        let sight = match condition {
            InfoCondition::Equal => NORMAL_WELL_RANGE,
            InfoCondition::Scouts => {
                if i < SCOUTS {
                    SCOUT_WELL_RANGE
                } else {
                    NORMAL_WELL_RANGE
                }
            }
            InfoCondition::Blind => {
                if i < SCOUTS {
                    0.0
                } else {
                    NORMAL_WELL_RANGE
                }
            }
        };
        well_sight.push(sight);
    }

    let cond_name = match condition {
        InfoCondition::Equal => "EQUAL_SIGHT",
        InfoCondition::Scouts => "SCOUTS",
        InfoCondition::Blind => "BLIND",
    };

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

        // Each agent sees wells based on THEIR sight range
        for (idx, &h) in handles.iter().enumerate() {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position();
            let sight = well_sight[idx];

            // Filter wells by this agent's sight range
            let wdata: Vec<_> = wells
                .iter()
                .zip(well_remaining.iter())
                .filter(|&(ref w, &r)| r > 0.0 && (pos - *w).norm() < sight)
                .map(|(&p, &r)| (p, (r / 2000.0).min(1.0)))
                .collect();

            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
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
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let nw: Vec<Option<usize>> = handles
            .iter()
            .map(|&h| {
                world.body(h).and_then(|b| {
                    let pos = b.position();
                    wells
                        .iter()
                        .enumerate()
                        .find(|&(ref i, &w)| (pos - w).norm() < 35.0 && well_remaining[*i] > 0.0)
                        .map(|(i, _)| i)
                })
            })
            .collect();
        for (idx, &h) in handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            consciousness.consume_energy(h, mr * (1.0 + consciousness.phi(h) * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                if let Some(wi) = nw[idx] {
                    let d = wr.min(well_remaining[wi]);
                    e.energy.regenerate(d);
                    well_remaining[wi] -= d;
                }
            }
        }

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

    // Special agents survival (first SCOUTS agents)
    let special_alive = (0..SCOUTS)
        .filter(|&i| {
            consciousness
                .entities
                .get(&handles[i])
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count() as f64
        / SCOUTS as f64;
    let normal_alive = (SCOUTS..AGENTS)
        .filter(|&i| {
            consciousness
                .entities
                .get(&handles[i])
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count() as f64
        / (AGENTS - SCOUTS) as f64;

    let clustering = avg_nn(&world, &handles);

    InfoResult {
        condition: cond_name,
        alive,
        special_alive,
        normal_alive,
        energy,
        clustering,
    }
}

fn main() {
    eprintln!("=== Information Asymmetry Experiment ===");
    eprintln!("Does knowing more help or harm the collective?");
    eprintln!("{AGENTS} agents, {SCOUTS} special, {TICKS} ticks, {SEEDS} seeds");

    println!("condition,seed,alive,special_survival,normal_survival,energy,clustering");

    let conditions = [
        (InfoCondition::Equal, "EQUAL_SIGHT"),
        (InfoCondition::Scouts, "SCOUTS"),
        (InfoCondition::Blind, "BLIND"),
    ];
    let mut all_results: Vec<(&str, Vec<InfoResult>)> = Vec::new();

    for &(cond, name) in &conditions {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(cond, seed);
            println!(
                "{},{seed},{:.1},{:.3},{:.3},{:.1},{:.2}",
                r.condition, r.alive, r.special_alive, r.normal_alive, r.energy, r.clustering
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}/{AGENTS}, special={:.0}%, normal={:.0}%",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.special_alive).sum::<f64>() / n * 100.0,
            results.iter().map(|r| r.normal_alive).sum::<f64>() / n * 100.0
        );
        all_results.push((name, results));
    }

    eprintln!("\n── Information Effects ──");
    eprintln!("  Condition      Total Alive  Special%  Normal%   Energy");
    for (name, results) in &all_results {
        let n = results.len() as f64;
        eprintln!(
            "  {:13} {:5.1}       {:5.0}%    {:5.0}%    {:5.1}J",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.special_alive).sum::<f64>() / n * 100.0,
            results.iter().map(|r| r.normal_alive).sum::<f64>() / n * 100.0,
            results.iter().map(|r| r.energy).sum::<f64>() / n
        );
    }

    // Key test: do scouts benefit the group?
    if all_results.len() >= 2 {
        let equal_alive: Vec<f64> = all_results[0].1.iter().map(|r| r.alive).collect();
        let scout_alive: Vec<f64> = all_results[1].1.iter().map(|r| r.alive).collect();
        let blind_alive: Vec<f64> = all_results[2].1.iter().map(|r| r.alive).collect();
        let (_, _, p1) = mann_whitney_u(&equal_alive, &scout_alive);
        let d1 = cohens_d(&equal_alive, &scout_alive);
        let (_, _, p2) = mann_whitney_u(&equal_alive, &blind_alive);
        let d2 = cohens_d(&equal_alive, &blind_alive);
        eprintln!("\n── EQUAL vs SCOUTS ──");
        eprintln!(
            "  p={p1:.4}, d={d1:.3} ({})",
            if d1.abs() > 0.8 {
                "large"
            } else if d1.abs() > 0.5 {
                "medium"
            } else {
                "small"
            }
        );
        eprintln!("\n── EQUAL vs BLIND ──");
        eprintln!(
            "  p={p2:.4}, d={d2:.3} ({})",
            if d2.abs() > 0.8 {
                "large"
            } else if d2.abs() > 0.5 {
                "medium"
            } else {
                "small"
            }
        );
    }

    eprintln!("\n=== Complete ===");
}

fn avg_nn(world: &PhysicsWorld<2>, handles: &[symtropy_physics::BodyHandle]) -> f64 {
    let pos: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|&h| world.body(h).map(|b| b.position()))
        .collect();
    if pos.len() < 2 {
        return f64::MAX;
    }
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
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
