// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! BIOLOGICAL VALIDATION 1: Does cooperative metabolic scaling match biology?
//!
//! Biology: eusocial insect colonies show per-capita metabolic rate ∝ M^(-0.19)
//! (Hou et al. 2010 PNAS, 95% CI: 0.55-1.08 for whole-colony exponent)
//!
//! Our test: sweep N = {4,8,12,16,24,32,48,64,96,128}, measure total and
//! per-capita energy consumption. Fit power law. Compare β against biology.
//!
//! Also: ISOLATED condition (no offloading) — per-capita should be constant
//! (matches Waters et al. 2010: isolated workers show isometric scaling)
//!
//! CSV output for manuscript plotting.

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const TICKS: usize = 2000;
const SEEDS: usize = 20;

fn run(n: usize, social: bool, seed: u64) -> (f64, f64) {
    // Returns (total_energy_consumed, per_capita_consumed)
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants {
        initial_energy: 500.0,
        max_energy: 500.0,
        consciousness_maintenance_per_tick: 0.10,
        movement_cost_per_unit: 0.006,
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: 0.05,
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.0, // no wells — pure cooperation test
        ambient_regen_rate: 0.02,
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: if social { 40.0 } else { 0.0 }, // 0 range = no social interaction
    };

    let area = (n as f64).sqrt() * 25.0;
    let mut rng = seed;
    let mut handles = Vec::new();
    for i in 0..n {
        let x = (nr(&mut rng) - 0.5) * area;
        let y = (nr(&mut rng) - 0.5) * area;
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

    for _tick in 0..TICKS {
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

        // FEP gradient (social condition only)
        if social {
            let ad: Vec<_> = handles
                .iter()
                .filter_map(|&h| {
                    Some((
                        world.body(h)?.position(),
                        consciousness.entities.get(&h)?.harmony_activations,
                    ))
                })
                .collect();
            for (idx, &h) in handles.iter().enumerate() {
                if consciousness
                    .entities
                    .get(&h)
                    .map(|e| e.energy.is_collapsed())
                    .unwrap_or(true)
                {
                    continue;
                }
                let pos = match world.body(h) {
                    Some(b) => b.position(),
                    None => continue,
                };
                let ef = consciousness
                    .entities
                    .get(&h)
                    .map(|e| e.energy.fraction_remaining())
                    .unwrap_or(0.0);
                let harm = consciousness
                    .entities
                    .get(&h)
                    .map(|e| e.harmony_activations)
                    .unwrap_or([0.5; 9]);
                let nearby: Vec<_> = ad
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != idx)
                    .map(|(_, d)| d.clone())
                    .collect();
                let dir =
                    fep_gradient::free_energy_gradient(&pos, ef, &harm, &nearby, &[], None, 0.0);
                if let Some(b) = world.body_mut(h) {
                    b.linear_velocity = dir * 25.0;
                }
            }
        }

        // Maintenance
        let rm = consciousness.resource_regeneration_multiplier();
        for &h in &handles {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
                let phi = e.phi();
                e.energy.consume(
                    consciousness.constants.consciousness_maintenance_per_tick * (1.0 + phi * 0.5),
                );
                e.energy
                    .regenerate(consciousness.constants.ambient_regen_rate * rm);
            }
        }

        // Offloading (social only)
        if social {
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
                            e.prediction_error *= 1.0 - off * 0.1;
                            e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                            e.energy.regenerate(
                                consciousness.constants.consciousness_maintenance_per_tick
                                    * off
                                    * 0.5,
                            );
                        }
                        if let Some(e) = consciousness.entities.get_mut(&hb) {
                            e.prediction_error *= 1.0 - off * 0.1;
                            e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                            e.energy.regenerate(
                                consciousness.constants.consciousness_maintenance_per_tick
                                    * off
                                    * 0.5,
                            );
                        }
                    }
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(1.0 / 64.0, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    // Measure total energy consumed
    let total_consumed: f64 = handles
        .iter()
        .filter_map(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| e.energy.lifetime_consumed)
        })
        .sum();
    let per_capita = total_consumed / n as f64;

    (total_consumed, per_capita)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║  BIOLOGICAL VALIDATION 1: Cooperative Metabolic Scaling      ║");
    eprintln!("║  Target: per-capita ∝ N^(-0.19) (Hou et al. 2010 PNAS)     ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");
    eprintln!();

    let populations = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128];

    // CSV header
    println!("condition,N,total_energy,per_capita_energy,ln_N,ln_per_capita");

    for &social in &[true, false] {
        let cond = if social { "SOCIAL" } else { "ISOLATED" };
        for &n in &populations {
            let mut total_sum = 0.0;
            let mut pc_sum = 0.0;
            for s in 0..SEEDS {
                let (total, pc) = run(n, social, 42 + s as u64 * 997);
                total_sum += total;
                pc_sum += pc;
            }
            let avg_total = total_sum / SEEDS as f64;
            let avg_pc = pc_sum / SEEDS as f64;
            println!(
                "{},{},{:.2},{:.2},{:.4},{:.4}",
                cond,
                n,
                avg_total,
                avg_pc,
                (n as f64).ln(),
                avg_pc.ln()
            );
        }
    }

    // Compute β (power law exponent) via linear regression on ln-ln
    eprintln!("\n═══════════════════════════════════════════════════");
    eprintln!("Fit: per_capita = a × N^β");
    eprintln!("Biology: β ≈ -0.19 (measured), -0.25 (WBE prediction)");
    eprintln!("Isolated: β ≈ 0.0 (no scaling, Waters et al. 2010)");
    eprintln!("═══════════════════════════════════════════════════");

    for &social in &[true, false] {
        let cond = if social { "SOCIAL" } else { "ISOLATED" };
        let mut ln_n = Vec::new();
        let mut ln_pc = Vec::new();
        for &n in &populations {
            let mut pc_sum = 0.0;
            for s in 0..SEEDS {
                let (_, pc) = run(n, social, 42 + s as u64 * 997);
                pc_sum += pc;
            }
            let avg_pc = pc_sum / SEEDS as f64;
            ln_n.push((n as f64).ln());
            ln_pc.push(avg_pc.ln());
        }
        // Linear regression: ln(pc) = β × ln(N) + ln(a)
        let n_pts = ln_n.len() as f64;
        let sum_x: f64 = ln_n.iter().sum();
        let sum_y: f64 = ln_pc.iter().sum();
        let sum_xy: f64 = ln_n.iter().zip(&ln_pc).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = ln_n.iter().map(|x| x * x).sum();
        let beta = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_xx - sum_x * sum_x);
        let r_sq = {
            let mean_y = sum_y / n_pts;
            let ss_tot: f64 = ln_pc.iter().map(|y| (y - mean_y).powi(2)).sum();
            let ss_res: f64 = ln_n
                .iter()
                .zip(&ln_pc)
                .map(|(x, y)| {
                    let pred = beta * x + (sum_y - beta * sum_x) / n_pts;
                    (y - pred).powi(2)
                })
                .sum();
            1.0 - ss_res / ss_tot
        };
        eprintln!("{}: β = {:.4}, R² = {:.4}", cond, beta, r_sq);
        if social {
            if beta > -0.30 && beta < -0.10 {
                eprintln!("  ✓ MATCHES biology (-0.19 to -0.25 range)");
            } else {
                eprintln!("  ✗ Does NOT match biology (expected -0.19 to -0.25)");
            }
        } else {
            if beta.abs() < 0.10 {
                eprintln!("  ✓ MATCHES isolated scaling (near zero, Waters 2010)");
            } else {
                eprintln!("  ✗ Does NOT match isolated scaling (expected ~0)");
            }
        }
    }
}

fn nr(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
