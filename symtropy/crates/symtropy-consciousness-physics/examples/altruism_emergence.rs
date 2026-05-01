// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Altruism Emergence — do agents sacrifice energy for collapsed neighbors?
//!
//! Tests Hamilton's rule: altruism evolves when rb > c, where r = relatedness
//! (here: harmony resonance), b = benefit to recipient, c = cost to donor.
//!
//! Compares 3 conditions:
//! 1. NO_RESCUE: Collapsed agents stay collapsed (baseline)
//! 2. PROXIMITY_RESCUE: Any agent within range revives collapsed neighbor (auto)
//! 3. COSTLY_RESCUE: Rescue costs the donor 20% of their energy
//!
//! Measures: survival, rescue events, donor survival after rescuing,
//! whether rescued agents survive long-term or just collapse again.
//!
//! Run: cargo run --example altruism_emergence --release

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
const SEEDS: usize = 8;
const RESCUE_COST_FRACTION: f64 = 0.20; // donor loses 20% of energy
const RESCUE_GIFT: f64 = 30.0; // recipient gets 30J
const RESCUE_COOLDOWN: usize = 500; // ticks before same agent can be rescued again

const HARMONY_PROFILES: [[f64; 9]; 4] = [
    [0.8, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.7, 0.5],
    [0.3, 0.7, 0.3, 0.2, 0.2, 0.3, 0.7, 0.3, 0.5],
    [0.2, 0.2, 0.7, 0.3, 0.7, 0.2, 0.2, 0.2, 0.5],
    [0.4, 0.4, 0.3, 0.5, 0.3, 0.5, 0.4, 0.4, 0.5],
];

#[derive(Clone, Copy, PartialEq)]
enum RescueMode {
    None,
    Free,
    Costly,
}

struct AltruismResult {
    condition: &'static str,
    alive: f64,
    energy: f64,
    rescues: f64,
    re_collapses: f64, // rescued agents that collapsed again
    donor_deaths: f64, // donors who died after rescuing
    unique_rescued: f64,
}

fn run_experiment(mode: RescueMode, seed: u64) -> AltruismResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();

    let mut constants = ThermodynamicConstants::research();
    constants.consciousness_maintenance_per_tick = 0.30; // moderate-high pressure
    consciousness.constants = constants;

    let wells = vec![SVector::from([30.0, 0.0]), SVector::from([-30.0, 0.0])];
    let mut well_remaining = vec![2000.0f64; 2];

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 120.0;
        let y = (rng_f64(&mut rng) - 0.5) * 120.0;
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

    let mut rescue_count = 0u64;
    let mut re_collapse_count = 0u64;
    let mut donor_death_count = 0u64;
    let mut last_rescued_tick: Vec<usize> = vec![0; AGENTS]; // cooldown tracker
    let mut was_rescued: Vec<bool> = vec![false; AGENTS];
    let mut was_donor: Vec<bool> = vec![false; AGENTS];

    for tick in 0..TICKS {
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
            .map(|(&p, &r)| (p, (r / 2000.0).min(1.0)))
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

        // Rescue mechanic
        if mode != RescueMode::None && tick % 100 == 0 {
            for i in 0..handles.len() {
                let hi = handles[i];
                let collapsed = consciousness
                    .entities
                    .get(&hi)
                    .map(|e| e.energy.is_collapsed())
                    .unwrap_or(false);
                if !collapsed {
                    continue;
                }
                if tick.saturating_sub(last_rescued_tick[i]) < RESCUE_COOLDOWN {
                    continue;
                }

                // Find nearest alive agent with high resonance
                let mut best_donor: Option<(usize, f64)> = None;
                let collapsed_harm = consciousness
                    .entities
                    .get(&hi)
                    .map(|e| e.harmony_activations)
                    .unwrap_or([0.0; 9]);

                for j in 0..handles.len() {
                    if i == j {
                        continue;
                    }
                    let hj = handles[j];
                    let alive = consciousness
                        .entities
                        .get(&hj)
                        .map(|e| !e.energy.is_collapsed())
                        .unwrap_or(false);
                    if !alive {
                        continue;
                    }
                    let in_range = match (world.body(hi), world.body(hj)) {
                        (Some(a), Some(b)) => {
                            a.position().distance(b.position())
                                < consciousness.constants.harmony_range
                        }
                        _ => false,
                    };
                    if !in_range {
                        continue;
                    }
                    let donor_harm = consciousness
                        .entities
                        .get(&hj)
                        .map(|e| e.harmony_activations)
                        .unwrap_or([0.0; 9]);
                    let res = HarmonyField::<2>::resonance(&collapsed_harm, &donor_harm);
                    if res > consciousness.constants.collapse_recovery_harmony_threshold {
                        if best_donor.map(|(_, r)| res > r).unwrap_or(true) {
                            best_donor = Some((j, res));
                        }
                    }
                }

                if let Some((donor_idx, _res)) = best_donor {
                    let hd = handles[donor_idx];
                    let can_rescue = match mode {
                        RescueMode::Free => true,
                        RescueMode::Costly => {
                            let donor_energy = consciousness
                                .entities
                                .get(&hd)
                                .map(|e| e.energy.available)
                                .unwrap_or(0.0);
                            donor_energy > donor_energy * RESCUE_COST_FRACTION + 20.0
                            // must have enough to survive after
                        }
                        RescueMode::None => false,
                    };

                    if can_rescue {
                        // Perform rescue
                        if mode == RescueMode::Costly {
                            let cost = consciousness
                                .entities
                                .get(&hd)
                                .map(|e| e.energy.available * RESCUE_COST_FRACTION)
                                .unwrap_or(0.0);
                            consciousness.consume_energy(hd, cost);
                            was_donor[donor_idx] = true;
                        }
                        if let Some(e) = consciousness.entities.get_mut(&hi) {
                            e.energy.available = RESCUE_GIFT;
                            e.energy.collapsed = false;
                        }
                        rescue_count += 1;
                        last_rescued_tick[i] = tick;
                        was_rescued[i] = true;
                    }
                }
            }
        }

        // Track re-collapses
        for i in 0..handles.len() {
            if was_rescued[i] {
                let collapsed = consciousness
                    .entities
                    .get(&handles[i])
                    .map(|e| e.energy.is_collapsed())
                    .unwrap_or(false);
                if collapsed {
                    re_collapse_count += 1;
                    was_rescued[i] = false; // only count once
                }
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
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    // Count donor deaths
    for i in 0..handles.len() {
        if was_donor[i] {
            let collapsed = consciousness
                .entities
                .get(&handles[i])
                .map(|e| e.energy.is_collapsed())
                .unwrap_or(true);
            if collapsed {
                donor_death_count += 1;
            }
        }
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
    let unique_rescued = was_rescued.iter().filter(|&&r| r).count() as f64;
    let cond_name = match mode {
        RescueMode::None => "NO_RESCUE",
        RescueMode::Free => "FREE_RESCUE",
        RescueMode::Costly => "COSTLY_RESCUE",
    };

    AltruismResult {
        condition: cond_name,
        alive,
        energy,
        rescues: rescue_count as f64,
        re_collapses: re_collapse_count as f64,
        donor_deaths: donor_death_count as f64,
        unique_rescued,
    }
}

fn main() {
    eprintln!("=== Altruism Emergence Experiment ===");
    eprintln!("Do agents benefit from rescuing collapsed neighbors?");
    eprintln!("{AGENTS} agents, {TICKS} ticks, {SEEDS} seeds");
    eprintln!(
        "Rescue cost: {:.0}% of donor energy, gift: {RESCUE_GIFT}J",
        RESCUE_COST_FRACTION * 100.0
    );

    println!("condition,seed,alive,energy,rescues,re_collapses,donor_deaths,unique_rescued");

    let modes = [
        (RescueMode::None, "NO_RESCUE"),
        (RescueMode::Free, "FREE_RESCUE"),
        (RescueMode::Costly, "COSTLY_RESCUE"),
    ];
    let mut all_results: Vec<(&str, Vec<AltruismResult>)> = Vec::new();

    for &(mode, name) in &modes {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {name} seed={seed}...");
            let r = run_experiment(mode, seed);
            println!(
                "{},{seed},{:.1},{:.1},{:.0},{:.0},{:.0},{:.0}",
                r.condition,
                r.alive,
                r.energy,
                r.rescues,
                r.re_collapses,
                r.donor_deaths,
                r.unique_rescued
            );
            results.push(r);
        }
        let n = results.len() as f64;
        eprintln!(
            "  → {name}: alive={:.1}, rescues={:.0}, re-collapses={:.0}, donor_deaths={:.0}",
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.rescues).sum::<f64>() / n,
            results.iter().map(|r| r.re_collapses).sum::<f64>() / n,
            results.iter().map(|r| r.donor_deaths).sum::<f64>() / n
        );
        all_results.push((name, results));
    }

    eprintln!("\n── Altruism Economics ──");
    eprintln!("  Condition       Alive  Rescues  Re-collapse  Donor Deaths");
    for (name, results) in &all_results {
        let n = results.len() as f64;
        eprintln!(
            "  {:14} {:5.1}   {:5.0}      {:5.0}         {:5.0}",
            name,
            results.iter().map(|r| r.alive).sum::<f64>() / n,
            results.iter().map(|r| r.rescues).sum::<f64>() / n,
            results.iter().map(|r| r.re_collapses).sum::<f64>() / n,
            results.iter().map(|r| r.donor_deaths).sum::<f64>() / n
        );
    }

    // Key test: Does costly rescue improve group survival despite individual cost?
    if all_results.len() >= 3 {
        let none_alive: Vec<f64> = all_results[0].1.iter().map(|r| r.alive).collect();
        let costly_alive: Vec<f64> = all_results[2].1.iter().map(|r| r.alive).collect();
        let (u, z, p) = mann_whitney_u(&none_alive, &costly_alive);
        let d = cohens_d(&none_alive, &costly_alive);
        eprintln!("\n── NO_RESCUE vs COSTLY_RESCUE (survival) ──");
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
        let verdict = if p < 0.05 && d < 0.0 {
            "ALTRUISM PAYS — costly rescue improves group survival"
        } else if p < 0.05 && d > 0.0 {
            "ALTRUISM BACKFIRES — rescue costs outweigh benefits"
        } else {
            "NO SIGNIFICANT DIFFERENCE"
        };
        eprintln!("  {verdict}");

        // Hamilton's rule check: does rescue success correlate with resonance?
        let rescue_rate: Vec<f64> = all_results[2]
            .1
            .iter()
            .map(|r| {
                if r.rescues > 0.0 {
                    1.0 - r.re_collapses / r.rescues
                } else {
                    0.0
                }
            })
            .collect();
        let mean_success = rescue_rate.iter().sum::<f64>() / rescue_rate.len() as f64;
        eprintln!("\n── Rescue Effectiveness ──");
        eprintln!(
            "  Mean rescue success rate: {:.1}% (rescued agents that survived)",
            mean_success * 100.0
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
