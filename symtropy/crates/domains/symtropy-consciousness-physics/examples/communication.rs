// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Communication — does signaling change cooperation?
//!
//! Every prior experiment uses deaf-mute agents. This tests whether
//! broadcasting energy state and known well locations to nearby agents
//! changes survival, clustering, or cooperation patterns.
//!
//! 3 conditions:
//! 1. SILENT: current baseline (no communication)
//! 2. ENERGY_BROADCAST: agents share their energy fraction with neighbors
//!    (low-energy agents attract help via modified FEP gradient)
//! 3. WELL_BROADCAST: agents share discovered well locations with neighbors
//!    (agents navigate to wells they've never seen via social information)
//!
//! Run: cargo run --example communication --release

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

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.7, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.6, 0.5],
    [0.3, 0.6, 0.3, 0.2, 0.2, 0.4, 0.6, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.4, 0.6, 0.2, 0.3, 0.2, 0.5],
    [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
];

#[derive(Clone, Copy)]
enum CommMode {
    Silent,
    EnergyBroadcast,
    WellBroadcast,
}

struct CommResult {
    condition: &'static str,
    alive: f64,
    energy: f64,
    clustering: f64,
    cooperation: f64,
    wells_known: f64,
}

fn run_experiment(mode: CommMode, seed: u64) -> CommResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.25;
    consciousness.constants = constants;

    // Wells placed far from center — communication should help find them
    let wells = vec![
        SVector::from([70.0, 30.0]),
        SVector::from([-60.0, -40.0]),
        SVector::from([20.0, -70.0]),
    ];
    let mut well_remaining = vec![2000.0f64; 3];

    let mut rng = seed;
    let mut handles = Vec::new();

    // Start clustered near center
    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 30.0;
        let y = (rng_f64(&mut rng) - 0.5) * 30.0;
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

    let use_energy_broadcast = matches!(mode, CommMode::EnergyBroadcast);
    let use_well_broadcast = matches!(mode, CommMode::WellBroadcast);
    let cond_name = match mode {
        CommMode::Silent => "SILENT",
        CommMode::EnergyBroadcast => "ENERGY_BC",
        CommMode::WellBroadcast => "WELL_BC",
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

        // Well discovery + memory
        for &h in &handles {
            if let Some(b) = world.body(h) {
                let pos = b.position();
                for &w in &wells {
                    if (pos - w).norm() < 35.0 {
                        if let Some(e) = consciousness.entities.get_mut(&h) {
                            let w2 = SVector::from([w[0], w[1]]);
                            e.memory.discover_well(&w2);
                        }
                    }
                }
            }
        }

        // WELL_BROADCAST: share known wells with nearby agents
        if use_well_broadcast {
            // Collect all known wells per agent
            let agent_wells: Vec<Vec<SVector<f64, 2>>> = handles
                .iter()
                .map(|h| {
                    consciousness
                        .entities
                        .get(h)
                        .map(|e| e.memory.known_wells.clone())
                        .unwrap_or_default()
                })
                .collect();
            // Share with neighbors
            for i in 0..handles.len() {
                let pos_i = match world.body(handles[i]) {
                    Some(b) => b.position(),
                    None => continue,
                };
                for j in 0..handles.len() {
                    if i == j {
                        continue;
                    }
                    let pos_j = match world.body(handles[j]) {
                        Some(b) => b.position(),
                        None => continue,
                    };
                    if (pos_i - pos_j).norm() > consciousness.constants.harmony_range {
                        continue;
                    }
                    // Agent j shares its wells with agent i
                    for well in &agent_wells[j] {
                        if let Some(e) = consciousness.entities.get_mut(&handles[i]) {
                            e.memory.discover_well(well);
                        }
                    }
                }
            }
        }

        // FEP gradient — build well data including memory
        let adata: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position(),
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();

        // ENERGY_BROADCAST: collect energy states of all agents
        let energy_states: Vec<(SVector<f64, 2>, f64)> = if use_energy_broadcast {
            handles
                .iter()
                .filter_map(|&h| {
                    let pos = world.body(h)?.position();
                    let ef = consciousness.entities.get(&h)?.energy.fraction_remaining();
                    Some((pos, ef))
                })
                .collect()
        } else {
            vec![]
        };

        for (idx, &h) in handles.iter().enumerate() {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position();

            // Build well data (visible + remembered)
            let mut wdata: Vec<_> = wells
                .iter()
                .zip(well_remaining.iter())
                .filter(|&(ref w, &r)| r > 0.0 && (pos - *w).norm() < 200.0)
                .map(|(&p, &r)| (p, (r / 2000.0).min(1.0)))
                .collect();

            // Add remembered wells (from memory or broadcast)
            for known in &e.memory.known_wells {
                let kf = SVector::from([known[0], known[1]]);
                if !wdata.iter().any(|(p, _)| (p - kf).norm() < 10.0) {
                    wdata.push((kf, 0.5));
                }
            }

            let mut near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();

            // ENERGY_BROADCAST: boost attraction to low-energy neighbors (help the weak)
            let mut dir = fep_gradient::free_energy_gradient(
                &pos,
                e.energy.fraction_remaining(),
                &e.harmony_activations,
                &near,
                &wdata,
                None,
                0.0,
            );

            if use_energy_broadcast {
                // Add attraction toward low-energy agents (altruistic gradient)
                let mut help_dir = SVector::<f64, 2>::zeros();
                for (other_pos, other_ef) in &energy_states {
                    let delta = other_pos - pos;
                    let dist = delta.norm();
                    if dist < 2.0 || dist > consciousness.constants.harmony_range {
                        continue;
                    }
                    if *other_ef < 0.3 {
                        // help agents below 30% energy
                        let urgency = (0.3 - other_ef) * 2.0;
                        help_dir += delta / dist * urgency;
                    }
                }
                let help_norm = help_dir.norm();
                if help_norm > 1e-6 {
                    dir = (dir + help_dir / help_norm * 0.3).normalize(); // 30% altruistic pull
                }
            }

            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = dir * 20.0;
            }
        }

        // Maintenance + regen
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
                    let pos = b.position();
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
        f64::MAX
    };
    let wells_known = handles
        .iter()
        .filter_map(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| e.memory.known_wells.len() as f64)
        })
        .sum::<f64>()
        / AGENTS as f64;

    CommResult {
        condition: cond_name,
        alive,
        energy,
        clustering,
        cooperation: coop as f64,
        wells_known,
    }
}

fn main() {
    eprintln!("=== Communication Experiment ===");
    eprintln!("Does signaling change cooperation?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");
    eprintln!("Wells placed far from center — communication should help");

    println!("condition,seed,alive,energy,clustering,cooperation,wells_known");

    let modes = [
        (CommMode::Silent, "SILENT"),
        (CommMode::EnergyBroadcast, "ENERGY_BC"),
        (CommMode::WellBroadcast, "WELL_BC"),
    ];
    let mut all: Vec<(&str, Vec<CommResult>)> = Vec::new();

    for &(mode, name) in &modes {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(mode, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.2},{:.0},{:.1}",
                r.condition, r.alive, r.energy, r.clustering, r.cooperation, r.wells_known
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}, wells={:.1}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.wells_known).sum::<f64>() / n
        );
        all.push((name, results));
    }

    eprintln!("\n── Communication Effects ──");
    eprintln!("  Condition    Alive  Energy  Cluster  Coop       Wells");
    for (name, results) in &all {
        let n = results.len() as f64;
        eprintln!(
            "  {:10} {:5.1}  {:5.1}J  {:6.2}   {:8.0}   {:4.1}",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.energy).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n,
            results.iter().map(|r| r.wells_known).sum::<f64>() / n
        );
    }

    // Statistical tests
    let s_alive: Vec<f64> = all[0].1.iter().map(|r| r.alive).collect();
    let e_alive: Vec<f64> = all[1].1.iter().map(|r| r.alive).collect();
    let w_alive: Vec<f64> = all[2].1.iter().map(|r| r.alive).collect();

    let tests = vec![
        ("SILENT vs ENERGY_BC", {
            let (_, _, p) = mann_whitney_u(&s_alive, &e_alive);
            p
        }),
        ("SILENT vs WELL_BC", {
            let (_, _, p) = mann_whitney_u(&s_alive, &w_alive);
            p
        }),
    ];
    let effects = [cohens_d(&s_alive, &e_alive), cohens_d(&s_alive, &w_alive)];
    let corrected = holm_bonferroni(&tests, 0.05);

    eprintln!("\n── Survival (Holm-Bonferroni, k=2) ──");
    for (i, &(label, adj_p, sig)) in corrected.iter().enumerate() {
        let d = effects[i];
        let size = if d.abs() > 0.8 {
            "large"
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

    eprintln!("\n=== Complete ===");
}

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
