// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial Threshold — what fraction of hostile agents breaks cooperation?
//!
//! Anti-agents emit dissonant harmony (inverted activations), increase friction
//! for nearby cooperators, and drain energy from resonant interactions.
//! Sweeps anti-agent fraction from 0% to 50% to find the critical threshold
//! where cooperative society collapses.
//!
//! Run: cargo run --example adversarial_threshold --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const TOTAL_AGENTS: usize = 20;
const TICKS: usize = 5_000;
const DT: f64 = 1.0 / 64.0;
const SEEDS: usize = 10;

struct AdversarialResult {
    fraction: f64,
    cooperator_alive: f64,
    adversary_alive: f64,
    clustering: f64,
    polarization: f64,
    coop_energy: f64,
    adv_energy: f64,
    cooperation_events: f64,
}

fn run_experiment(adversary_fraction: f64, seed: u64) -> AdversarialResult {
    let num_adversaries = (TOTAL_AGENTS as f64 * adversary_fraction).round() as usize;
    let num_cooperators = TOTAL_AGENTS - num_adversaries;

    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([25.0, 0.0]), SVector::from([-25.0, 0.0])];
    let mut well_remaining = vec![3000.0f64; 2];

    let mut rng = seed;
    let mut coop_handles = Vec::new();
    let mut adv_handles = Vec::new();

    // Spawn cooperators with diverse but compatible harmonies
    for i in 0..num_cooperators {
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
            match i % 4 {
                0 => e.harmony_activations = [0.8, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.7, 0.3],
                1 => e.harmony_activations = [0.3, 0.7, 0.3, 0.2, 0.3, 0.2, 0.6, 0.3, 0.5],
                2 => e.harmony_activations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                _ => e.harmony_activations = [0.4, 0.4, 0.6, 0.3, 0.4, 0.5, 0.4, 0.4, 0.4],
            }
        }
        coop_handles.push(h);
    }

    // Spawn adversaries with INVERTED harmonies (anti-resonant)
    for _ in 0..num_adversaries {
        let x = (rng_f64(&mut rng) - 0.5) * 120.0;
        let y = (rng_f64(&mut rng) - 0.5) * 120.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.5); // slightly heavier
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.03;
        } // faster
        consciousness.register(
            h,
            consciousness.constants.initial_energy,
            consciousness.constants.harmony_range,
        );
        if let Some(e) = consciousness.entities.get_mut(&h) {
            // Inverted: high where cooperators are low, low where they're high
            e.harmony_activations = [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9];
        }
        adv_handles.push(h);
    }

    let all_handles: Vec<_> = coop_handles
        .iter()
        .chain(adv_handles.iter())
        .cloned()
        .collect();
    let mut cooperation_events = 0u64;

    for _tick in 0..TICKS {
        // Consciousness update
        for &h in &all_handles {
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
        let adata: Vec<_> = all_handles
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
            .map(|(&p, &r)| (p, (r / 3000.0).min(1.0)))
            .collect();
        for &h in &all_handles {
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
                    (p - pos).norm() > 2.0
                        && (p - pos).norm() < consciousness.constants.harmony_range
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
                b.linear_velocity = dir * 25.0;
            }
        }

        // Maintenance + regen with well depletion
        let rm = consciousness.resource_regeneration_multiplier();
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let nw: Vec<Option<usize>> = all_handles
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
        for (idx, &h) in all_handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            let phi = consciousness.phi(h);
            consciousness.consume_energy(h, mr * (1.0 + phi * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                if let Some(wi) = nw[idx] {
                    let draw = wr.min(well_remaining[wi]);
                    e.energy.regenerate(draw);
                    well_remaining[wi] -= draw;
                }
            }
        }

        // Cooperation + adversarial disruption
        for i in 0..all_handles.len() {
            for j in (i + 1)..all_handles.len() {
                let (ha, hb) = (all_handles[i], all_handles[j]);
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
                } else if res < 0.2 {
                    // Adversarial disruption: low resonance DRAINS energy (conflict cost)
                    let drain = (0.2 - res) * 0.05;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.energy.consume(drain);
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.consume(drain);
                    }
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    let coop_alive = coop_handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count() as f64;
    let adv_alive = adv_handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count() as f64;
    let coop_energy = coop_handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / num_cooperators.max(1) as f64;
    let adv_energy = if num_adversaries > 0 {
        adv_handles
            .iter()
            .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
            .sum::<f64>()
            / num_adversaries as f64
    } else {
        0.0
    };

    let pos: Vec<SVector<f64, 2>> = all_handles
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

    // Polarization: variance of harmony across all agents
    let n = all_handles.len() as f64;
    let mut mean_h = [0.0f64; 9];
    for &h in &all_handles {
        if let Some(e) = consciousness.entities.get(&h) {
            for i in 0..8 {
                mean_h[i] += e.harmony_activations[i];
            }
        }
    }
    for v in &mut mean_h {
        *v /= n;
    }
    let polarization: f64 = all_handles
        .iter()
        .filter_map(|h| {
            consciousness.entities.get(h).map(|e| {
                (0..8)
                    .map(|i| (e.harmony_activations[i] - mean_h[i]).powi(2))
                    .sum::<f64>()
            })
        })
        .sum::<f64>()
        / n;

    AdversarialResult {
        fraction: adversary_fraction,
        cooperator_alive: coop_alive,
        adversary_alive: adv_alive,
        clustering,
        polarization,
        coop_energy,
        adv_energy,
        cooperation_events: cooperation_events as f64,
    }
}

fn main() {
    eprintln!("=== Adversarial Threshold Experiment ===");
    eprintln!("What fraction of hostile agents breaks cooperation?");

    println!("fraction,seed,coop_alive,adv_alive,clustering,polarization,coop_energy,adv_energy,cooperation");

    let fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50];
    let mut all_results: Vec<(f64, Vec<AdversarialResult>)> = Vec::new();

    for &frac in &fractions {
        let mut results = Vec::new();
        for s in 0..SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {:.0}% adversaries, seed={seed}...", frac * 100.0);
            let r = run_experiment(frac, seed);
            println!(
                "{:.2},{seed},{:.1},{:.1},{:.2},{:.4},{:.1},{:.1},{:.0}",
                r.fraction,
                r.cooperator_alive,
                r.adversary_alive,
                r.clustering,
                r.polarization,
                r.coop_energy,
                r.adv_energy,
                r.cooperation_events
            );
            results.push(r);
        }

        let n = results.len() as f64;
        let nc = (TOTAL_AGENTS as f64 * (1.0 - frac)).round();
        let na = (TOTAL_AGENTS as f64 * frac).round();
        eprintln!("  {:.0}% adversaries: coop_alive={:.1}/{:.0}, adv_alive={:.1}/{:.0}, cluster={:.2}, coop_events={:.0}",
            frac * 100.0,
            results.iter().map(|r| r.cooperator_alive).sum::<f64>() / n, nc,
            results.iter().map(|r| r.adversary_alive).sum::<f64>() / n, na,
            results.iter().map(|r| r.clustering).sum::<f64>() / n,
            results.iter().map(|r| r.cooperation_events).sum::<f64>() / n);

        all_results.push((frac, results));
    }

    // Find critical threshold: first fraction where cooperator survival < 50%
    eprintln!("\n── Critical Threshold Analysis ──");
    for (frac, results) in &all_results {
        let nc = (TOTAL_AGENTS as f64 * (1.0 - frac)).round();
        let mean_alive =
            results.iter().map(|r| r.cooperator_alive).sum::<f64>() / results.len() as f64;
        let survival_rate = if nc > 0.0 { mean_alive / nc } else { 1.0 };
        eprintln!(
            "  {:.0}%: cooperator survival = {:.1}%{}",
            frac * 100.0,
            survival_rate * 100.0,
            if survival_rate < 0.5 {
                " ← CRITICAL THRESHOLD"
            } else {
                ""
            }
        );
    }

    // Statistical comparison: 0% vs highest fraction
    if all_results.len() >= 2 {
        let baseline = &all_results[0].1;
        let worst = &all_results.last().unwrap().1;
        let b_alive: Vec<f64> = baseline.iter().map(|r| r.cooperator_alive).collect();
        let w_alive: Vec<f64> = worst.iter().map(|r| r.cooperator_alive).collect();
        let (u, z, p) = mann_whitney_u(&b_alive, &w_alive);
        let d = cohens_d(&b_alive, &w_alive);
        eprintln!("\n  0% vs 50% adversaries (cooperator survival):");
        eprintln!("    Mann-Whitney U={:.1}, z={:.3}, p={:.4}", u, z, p);
        eprintln!(
            "    Cohen's d={:.3} ({})",
            d,
            if d.abs() > 0.8 {
                "large"
            } else {
                "medium/small"
            }
        );
        eprintln!(
            "    {}",
            if p < 0.05 {
                "SIGNIFICANT"
            } else {
                "not significant"
            }
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
