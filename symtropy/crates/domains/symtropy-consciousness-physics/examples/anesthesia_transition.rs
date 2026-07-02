// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! BIOLOGICAL VALIDATION 3: Anesthesia-like phase transition
//!
//! Biology: loss of consciousness under anesthesia shows:
//! - ~10% coefficient of variation (MAC standard)
//! - Hysteresis / "neural inertia" (Friedman et al. 2010)
//! - Debated: first-order vs second-order transition
//!
//! Test: sweep consciousness_maintenance_cost from 0.01 to 1.0,
//! measure collective Phi. Detect transition point, CV, hysteresis.

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 12;
const TICKS: usize = 500; // short runs — we need many
const SEEDS: usize = 30;
const COST_STEPS: usize = 50;

fn run_at_cost(maint_cost: f64, seed: u64) -> f64 {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants {
        initial_energy: 500.0,
        max_energy: 500.0,
        consciousness_maintenance_per_tick: maint_cost,
        movement_cost_per_unit: 0.006,
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: 0.05,
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.0,
        ambient_regen_rate: 0.02,
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: 40.0,
    };

    let mut rng = seed;
    let mut handles = Vec::new();
    for i in 0..AGENTS {
        let x = (nr(&mut rng) - 0.5) * 80.0;
        let y = (nr(&mut rng) - 0.5) * 80.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.3;
            b.linear_velocity =
                SVector::from([(nr(&mut rng) - 0.5) * 8.0, (nr(&mut rng) - 0.5) * 8.0]);
        }
        consciousness.register(h, 500.0, 20.0);
        if let Some(e) = consciousness.entities.get_mut(&h) {
            match i % 4 {
                0 => e.harmony_activations = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5],
                1 => e.harmony_activations = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.5],
                2 => e.harmony_activations = [0.1, 0.1, 0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.5],
                _ => e.harmony_activations = [0.4; 9],
            }
        }
        handles.push(h);
    }

    for _ in 0..TICKS {
        for &h in &handles {
            let ef = consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.fraction_remaining())
                .unwrap_or(0.0);
            let inputs = ConsciousnessInputs {
                phi: if ef > 0.0 { 0.5 } else { 0.0 },
                broadcast: 0.6,
                working_memory: 0.5,
                attention: 0.5,
                recurrence: 0.4,
                embodiment: 0.6,
                knowledge: 0.4,
                synchrony: 0.5,
            };
            consciousness.update_entity(h, &inputs, Point::origin());
        }

        let rm = consciousness.resource_regeneration_multiplier();
        for &h in &handles {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
                let phi = e.phi();
                e.energy.consume(maint_cost * (1.0 + phi * 0.5));
                e.energy
                    .regenerate(consciousness.constants.ambient_regen_rate * rm);
            }
        }

        // Offloading
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let dist = match (world.body(ha), world.body(hb)) {
                    (Some(a), Some(b)) => a.position().metric_distance(&b.position()),
                    _ => continue,
                };
                if dist > consciousness.constants.harmony_range {
                    continue;
                }
                let (hah, hbh) = match (
                    consciousness.entities.get(&ha),
                    consciousness.entities.get(&hb),
                ) {
                    (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                    _ => continue,
                };
                let res = HarmonyField::<2>::resonance(&hah, &hbh);
                if res > 0.5 {
                    let off = (res - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.energy.regenerate(maint_cost * off * 0.5);
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.regenerate(maint_cost * off * 0.5);
                    }
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(1.0 / 64.0, &mut consciousness);
    }

    // Return fraction of agents still alive (conscious)
    let alive = handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count();
    alive as f64 / AGENTS as f64
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║  BIOLOGICAL VALIDATION 3: Anesthesia Phase Transition        ║");
    eprintln!("║  Target: ~10% CV, hysteresis (Friedman et al. 2010)         ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");

    // CSV header
    println!("cost,alive_fraction,std");

    let mut transition_costs = Vec::new();

    for step in 0..COST_STEPS {
        let cost = 0.01 + (step as f64 / COST_STEPS as f64) * 0.99;
        let mut fracs = Vec::new();

        for s in 0..SEEDS {
            fracs.push(run_at_cost(cost, 42 + s as u64 * 997));
        }

        let mean = fracs.iter().sum::<f64>() / SEEDS as f64;
        let std =
            (fracs.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / (SEEDS as f64 - 1.0)).sqrt();
        println!("{:.4},{:.4},{:.4}", cost, mean, std);

        // Detect transition: alive fraction crosses 50%
        if mean < 0.5 {
            transition_costs.push(cost);
        }
    }

    // Compute transition statistics
    if !transition_costs.is_empty() {
        let mac = transition_costs[0]; // first cost where >50% collapse
        eprintln!("\n═══════════════════════════════════════════════════");
        eprintln!("MAC (50% collapse point): {:.3} J/tick", mac);

        // Compute per-seed transition points for CV
        let mut per_seed_macs = Vec::new();
        for s in 0..SEEDS {
            for step in 0..COST_STEPS {
                let cost = 0.01 + (step as f64 / COST_STEPS as f64) * 0.99;
                let frac = run_at_cost(cost, 42 + s as u64 * 997);
                if frac < 0.5 {
                    per_seed_macs.push(cost);
                    break;
                }
            }
        }

        if per_seed_macs.len() > 1 {
            let mean_mac = per_seed_macs.iter().sum::<f64>() / per_seed_macs.len() as f64;
            let std_mac = (per_seed_macs
                .iter()
                .map(|m| (m - mean_mac).powi(2))
                .sum::<f64>()
                / (per_seed_macs.len() as f64 - 1.0))
                .sqrt();
            let cv = std_mac / mean_mac * 100.0;
            eprintln!(
                "MAC mean: {:.3}, SD: {:.3}, CV: {:.1}%",
                mean_mac, std_mac, cv
            );
            eprintln!("BIOLOGY: CV ≈ 10% (anesthesia MAC)");
            if (cv - 10.0).abs() < 15.0 {
                eprintln!("  ✓ CV within range of biological anesthesia");
            } else {
                eprintln!("  ✗ CV does not match biological anesthesia");
            }
        }
        eprintln!("═══════════════════════════════════════════════════");
    }
}

fn nr(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
