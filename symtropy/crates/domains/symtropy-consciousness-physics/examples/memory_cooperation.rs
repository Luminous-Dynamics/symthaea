// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Memory Cooperation — does temporal state change cooperation patterns?
//!
//! 3 conditions:
//! 1. STATELESS: current baseline (agents forget every tick)
//! 2. MEMORY: agents remember well positions + partner history (no learning)
//! 3. MEMORY+LEARN: memory + reward-modulated plasticity with windowed rewards
//!
//! The memory condition lets agents navigate to KNOWN wells instead of
//! relying on FEP gradient to discover them. The learning condition also
//! adapts FEP weights based on 100-tick energy trends.
//!
//! Run: cargo run --example memory_cooperation --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, holm_bonferroni, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient::{self, LearnedFepWeights};
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
enum Mode {
    Stateless,
    Memory,
    MemoryLearn,
}

struct MemResult {
    condition: &'static str,
    alive: f64,
    energy: f64,
    clustering: f64,
    cooperation: f64,
    wells_discovered: f64,
    partners_known: f64,
    weight_drift: f64,
}

fn run_experiment(mode: Mode, seed: u64) -> MemResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.25;
    consciousness.constants = constants;

    // Wells placed far from center to reward discovery
    let wells = vec![
        SVector::from([60.0, 20.0]),
        SVector::from([-50.0, -30.0]),
        SVector::from([10.0, -60.0]),
    ];
    let mut well_remaining = vec![2000.0f64; 3];

    let mut rng = seed;
    let mut handles = Vec::new();
    let mut agent_weights: Vec<LearnedFepWeights> = Vec::new();

    // All agents start near center (must discover distant wells)
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
        agent_weights.push(LearnedFepWeights::default());
    }

    let cond_name = match mode {
        Mode::Stateless => "STATELESS",
        Mode::Memory => "MEMORY",
        Mode::MemoryLearn => "MEM+LEARN",
    };
    let use_memory = matches!(mode, Mode::Memory | Mode::MemoryLearn);
    let use_learning = matches!(mode, Mode::MemoryLearn);

    let mut coop = 0u64;

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

        // Memory: record energy + discover wells
        if use_memory {
            for (idx, &h) in handles.iter().enumerate() {
                let ef = consciousness
                    .entities
                    .get(&h)
                    .map(|e| e.energy.fraction_remaining())
                    .unwrap_or(0.0);
                if let Some(e) = consciousness.entities.get_mut(&h) {
                    e.memory.record_energy(ef);
                }
                // Discover wells when near them
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
        }

        // FEP gradient — memory agents include known wells even if out of sight
        let adata: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position(),
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();

        for (idx, &h) in handles.iter().enumerate() {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position();

            // Standard well data (visible wells)
            let mut wdata: Vec<_> = wells
                .iter()
                .zip(well_remaining.iter())
                .filter(|&(ref w, &r)| r > 0.0 && (pos - *w).norm() < 200.0)
                .map(|(&p, &r)| (p, (r / 2000.0).min(1.0)))
                .collect();

            // Memory: add remembered wells even if not currently visible
            if use_memory {
                for known in &e.memory.known_wells {
                    let known_full = SVector::from([known[0], known[1]]);
                    // Add if not already in visible set
                    let already = wdata.iter().any(|(p, _)| (p - known_full).norm() < 10.0);
                    if !already {
                        wdata.push((known_full, 0.5)); // assume 50% remaining if unseen
                    }
                }
            }

            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| {
                    let d = (p - pos).norm();
                    d > 2.0 && d < consciousness.constants.harmony_range
                })
                .cloned()
                .collect();

            if use_learning {
                let (dir, contributions) = fep_gradient::free_energy_gradient_learned(
                    &pos,
                    e.energy.fraction_remaining(),
                    None,
                    &e.harmony_activations,
                    &near,
                    &wdata,
                    None,
                    0.0,
                    &agent_weights[idx],
                );
                // Use windowed reward from memory
                let reward = e.memory.windowed_reward();
                agent_weights[idx].update_windowed(reward, contributions);
                if let Some(b) = world.body_mut(h) {
                    b.linear_velocity = dir * 20.0;
                }
            } else {
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
        }

        // Maintenance
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
                            if use_memory {
                                e.memory.ticks_since_regen = 0;
                            }
                            break;
                        }
                    }
                }
                if use_memory {
                    e.memory.ticks_since_regen += 1;
                }
            }
        }

        // Cooperation + partner memory
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
                        if use_memory {
                            e.memory.record_partner(hb, res);
                        }
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.regenerate(rg);
                        if use_memory {
                            e.memory.record_partner(ha, res);
                        }
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

    let wells_disc = handles
        .iter()
        .filter_map(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| e.memory.known_wells.len() as f64)
        })
        .sum::<f64>()
        / AGENTS as f64;
    let partners = handles
        .iter()
        .filter_map(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| e.memory.partner_history.len() as f64)
        })
        .sum::<f64>()
        / AGENTS as f64;
    let drift = agent_weights
        .iter()
        .map(|w| w.drift_from_default())
        .sum::<f64>()
        / AGENTS as f64;

    MemResult {
        condition: cond_name,
        alive,
        energy,
        clustering,
        cooperation: coop as f64,
        wells_discovered: wells_disc,
        partners_known: partners,
        weight_drift: drift,
    }
}

fn main() {
    eprintln!("=== Memory Cooperation Experiment ===");
    eprintln!("Does temporal state change cooperation patterns?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");

    println!("condition,seed,alive,energy,clustering,cooperation,wells_disc,partners,weight_drift");

    let modes = [
        (Mode::Stateless, "STATELESS"),
        (Mode::Memory, "MEMORY"),
        (Mode::MemoryLearn, "MEM+LEARN"),
    ];
    let mut all_results: Vec<(&str, Vec<MemResult>)> = Vec::new();

    for &(mode, name) in &modes {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(mode, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.2},{:.0},{:.1},{:.1},{:.4}",
                r.condition,
                r.alive,
                r.energy,
                r.clustering,
                r.cooperation,
                r.wells_discovered,
                r.partners_known,
                r.weight_drift
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}, wells={:.1}, partners={:.1}, drift={:.4}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.wells_discovered).sum::<f64>() / n,
            results.iter().map(|r| r.partners_known).sum::<f64>() / n,
            results.iter().map(|r| r.weight_drift).sum::<f64>() / n
        );
        all_results.push((name, results));
    }

    // Summary
    eprintln!("\n── Memory Effects ──");
    eprintln!("  Condition    Alive  Energy  Cluster  Coop     Wells  Partners  Drift");
    for (name, results) in &all_results {
        let n = results.len() as f64;
        eprintln!(
            "  {:10} {:5.1}  {:5.1}J  {:6.2}   {:8.0}  {:4.1}   {:5.1}     {:.4}",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.energy).sum::<f64>() / n,
            results.iter().map(|r| r.clustering).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation).sum::<f64>() / n,
            results.iter().map(|r| r.wells_discovered).sum::<f64>() / n,
            results.iter().map(|r| r.partners_known).sum::<f64>() / n,
            results.iter().map(|r| r.weight_drift).sum::<f64>() / n
        );
    }

    // Statistical tests (Holm-Bonferroni)
    let stateless = &all_results[0].1;
    let memory = &all_results[1].1;
    let memlearn = &all_results[2].1;

    let s_alive: Vec<f64> = stateless.iter().map(|r| r.alive).collect();
    let m_alive: Vec<f64> = memory.iter().map(|r| r.alive).collect();
    let ml_alive: Vec<f64> = memlearn.iter().map(|r| r.alive).collect();

    let (_, _, p1) = mann_whitney_u(&s_alive, &m_alive);
    let d1 = cohens_d(&s_alive, &m_alive);
    let (_, _, p2) = mann_whitney_u(&s_alive, &ml_alive);
    let d2 = cohens_d(&s_alive, &ml_alive);
    let (_, _, p3) = mann_whitney_u(&m_alive, &ml_alive);
    let d3 = cohens_d(&m_alive, &ml_alive);

    let tests = vec![
        ("STATELESS vs MEMORY", p1),
        ("STATELESS vs MEM+LEARN", p2),
        ("MEMORY vs MEM+LEARN", p3),
    ];
    let corrected = holm_bonferroni(&tests, 0.05);
    let effects = [d1, d2, d3];

    eprintln!("\n── Survival Comparisons (Holm-Bonferroni, k=3) ──");
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
            "  {label:25}: p_adj={adj_p:.4}, d={d:.3} ({size}) {}",
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
