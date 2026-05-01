// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase Transition — is there a critical temperature where cooperation collapses?
//!
//! Sweeps maintenance pressure (analogous to environmental temperature) from
//! gentle to extreme. At each level, measures survival, clustering, cooperation
//! events, mean temperature, and entropy production. Looks for a sharp transition
//! point — the Ising model analogy for social physics.
//!
//! The "order parameter" is cooperation rate: high cooperation = ordered phase
//! (like ferromagnetic), zero cooperation = disordered (paramagnetic).
//!
//! Run: cargo run --example phase_transition --release

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
const SEEDS: usize = 8;

/// Maintenance pressure levels to sweep (J/tick).
/// Range from gentle (0.05) to lethal (1.0).
const PRESSURES: [f64; 12] = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00,
];

/// 4 harmony archetypes with two compatible pairs.
const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.8, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.7, 0.5], // A
    [0.7, 0.4, 0.3, 0.1, 0.3, 0.2, 0.3, 0.6, 0.5], // A' (resonates with A)
    [0.1, 0.2, 0.7, 0.3, 0.8, 0.2, 0.1, 0.2, 0.5], // B
    [0.2, 0.1, 0.6, 0.4, 0.7, 0.3, 0.2, 0.1, 0.5], // B' (resonates with B)
];

struct PhaseResult {
    pressure: f64,
    alive: f64,
    energy: f64,
    clustering: f64,
    cooperation_events: f64,
    mean_temperature: f64,
    total_entropy: f64,
    helmholtz: f64,
}

fn run_experiment(maintenance: f64, seed: u64) -> PhaseResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = maintenance;
    consciousness.constants = constants;

    let wells = vec![SVector::from([25.0, 0.0]), SVector::from([-25.0, 0.0])];
    let mut well_remaining = vec![2500.0f64; 2];

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

    let mut cooperation_events = 0u64;
    let mut temp_sum = 0.0f64;
    let mut temp_count = 0u64;

    for tick in 0..TICKS {
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

        // FEP gradient
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

        // Maintenance + regen
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let nw: Vec<Option<usize>> = handles
            .iter()
            .map(|&h| {
                world.body(h).and_then(|b| {
                    let pos = b.position().0;
                    wells
                        .iter()
                        .enumerate()
                        .find(|(i, &w)| (pos - w).norm() < 35.0 && well_remaining[*i] > 0.0)
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

        // Cooperation (harmony resonance regen)
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
                    cooperation_events += 1;
                }
            }
        }

        // Sample temperature every 200 ticks
        if tick % 200 == 0 {
            for &h in &handles {
                if let Some(e) = consciousness.entities.get(&h) {
                    temp_sum += e.energy.temperature;
                    temp_count += 1;
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
    let clustering = avg_nn(&world, &handles);
    let mean_temp = if temp_count > 0 {
        temp_sum / temp_count as f64
    } else {
        310.0
    };
    let total_entropy: f64 = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.entropy))
        .sum();
    let helmholtz: f64 = handles
        .iter()
        .filter_map(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| e.energy.available_work())
        })
        .sum::<f64>()
        / AGENTS as f64;

    PhaseResult {
        pressure: maintenance,
        alive,
        energy,
        clustering,
        cooperation_events: cooperation_events as f64,
        mean_temperature: mean_temp,
        total_entropy: total_entropy,
        helmholtz,
    }
}

fn main() {
    eprintln!("=== Phase Transition Experiment ===");
    eprintln!("Is there a critical pressure where cooperation collapses?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds per pressure level");
    eprintln!(
        "Sweeping {} maintenance pressures from {:.2} to {:.2} J/tick",
        PRESSURES.len(),
        PRESSURES[0],
        PRESSURES[PRESSURES.len() - 1]
    );

    println!("pressure,seed,alive,energy,clustering,cooperation,mean_temp,entropy,helmholtz");

    let mut all_results: Vec<(f64, Vec<PhaseResult>)> = Vec::new();

    for &pressure in &PRESSURES {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  pressure={pressure:.2} seed={seed}...");
            let r = run_experiment(pressure, seed);
            println!(
                "{:.2},{seed},{:.1},{:.1},{:.2},{:.0},{:.1},{:.2},{:.1}",
                r.pressure,
                r.alive,
                r.energy,
                r.clustering,
                r.cooperation_events,
                r.mean_temperature,
                r.total_entropy,
                r.helmholtz
            );
            results.push(r);
        }

        let n = results.len() as f64;
        let alive = results.iter().map(|r| r.alive).sum::<f64>() / n;
        let coop = results.iter().map(|r| r.cooperation_events).sum::<f64>() / n;
        let temp = results.iter().map(|r| r.mean_temperature).sum::<f64>() / n;
        let ent = results.iter().map(|r| r.total_entropy).sum::<f64>() / n;
        let helm = results.iter().map(|r| r.helmholtz).sum::<f64>() / n;
        eprintln!("  → alive={alive:.1}/{AGENTS}, coop={coop:.0}, T={temp:.1}K, S={ent:.1}J/K, F={helm:.1}J");

        all_results.push((pressure, results));
    }

    // Find critical threshold: first pressure where survival < 50%
    eprintln!("\n── Phase Diagram ──");
    eprintln!("  Pressure  Alive  Coop     Temp    Entropy  Helmholtz");
    let mut critical = None;
    for (pressure, results) in &all_results {
        let n = results.len() as f64;
        let alive = results.iter().map(|r| r.alive).sum::<f64>() / n;
        let coop = results.iter().map(|r| r.cooperation_events).sum::<f64>() / n;
        let temp = results.iter().map(|r| r.mean_temperature).sum::<f64>() / n;
        let ent = results.iter().map(|r| r.total_entropy).sum::<f64>() / n;
        let helm = results.iter().map(|r| r.helmholtz).sum::<f64>() / n;
        let survival_rate = alive / AGENTS as f64;
        let marker = if survival_rate < 0.5 && critical.is_none() {
            critical = Some(*pressure);
            " ← CRITICAL"
        } else {
            ""
        };
        eprintln!("  {pressure:.2}    {alive:5.1}  {coop:7.0}  {temp:7.1}K  {ent:8.1}   {helm:7.1}J{marker}");
    }

    if let Some(tc) = critical {
        eprintln!("\n  Critical pressure: {tc:.2} J/tick");
        eprintln!("  Below this: cooperative phase (ordered)");
        eprintln!("  Above this: collapse phase (disordered)");
    } else {
        eprintln!("\n  No critical threshold found in range — all survived or all collapsed");
    }

    // Statistical test: lowest vs highest pressure
    if all_results.len() >= 2 {
        let low = &all_results[0].1;
        let high = &all_results.last().unwrap().1;
        let l_alive: Vec<f64> = low.iter().map(|r| r.alive).collect();
        let h_alive: Vec<f64> = high.iter().map(|r| r.alive).collect();
        let (u, z, p) = mann_whitney_u(&l_alive, &h_alive);
        let d = cohens_d(&l_alive, &h_alive);
        eprintln!("\n── Low (0.05) vs High (1.00) Pressure ──");
        eprintln!("  Mann-Whitney U={u:.1}, z={z:.3}, p={p:.4}");
        eprintln!(
            "  Cohen's d={d:.3} ({})",
            if d.abs() > 0.8 {
                "large"
            } else if d.abs() > 0.5 {
                "medium"
            } else {
                "small"
            }
        );

        // Cooperation events
        let l_coop: Vec<f64> = low.iter().map(|r| r.cooperation_events).collect();
        let h_coop: Vec<f64> = high.iter().map(|r| r.cooperation_events).collect();
        let (u2, z2, p2) = mann_whitney_u(&l_coop, &h_coop);
        let d2 = cohens_d(&l_coop, &h_coop);
        eprintln!("\n── Cooperation: Low vs High Pressure ──");
        eprintln!("  Mann-Whitney U={u2:.1}, z={z2:.3}, p={p2:.4}");
        eprintln!(
            "  Cohen's d={d2:.3} ({})",
            if d2.abs() > 0.8 {
                "large"
            } else if d2.abs() > 0.5 {
                "medium"
            } else {
                "small"
            }
        );

        // Entropy
        let l_ent: Vec<f64> = low.iter().map(|r| r.total_entropy).collect();
        let h_ent: Vec<f64> = high.iter().map(|r| r.total_entropy).collect();
        let d3 = cohens_d(&l_ent, &h_ent);
        eprintln!("\n── Entropy Production: Low vs High ──");
        eprintln!(
            "  Cohen's d={d3:.3} ({})",
            if d3.abs() > 0.8 {
                "large"
            } else if d3.abs() > 0.5 {
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
        .filter_map(|h| world.body(*h).map(|b| b.position().0))
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
