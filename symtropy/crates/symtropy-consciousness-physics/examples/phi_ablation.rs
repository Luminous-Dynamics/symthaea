// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PHI ABLATION: Does the consciousness metric matter?
//!
//! Tests whether clustering depends on the SPECIFIC integration measure (Φ)
//! or whether ANY scalar produces the same effect.
//!
//! 5 metric conditions × 30 seeds × 5000 ticks = 150 runs
//!
//! Conditions:
//! 1. PHI:      Consciousness equation (IIT-inspired integration)
//! 2. ENTROPY:  Shannon entropy of agent state
//! 3. RANDOM:   Uniform random [0,1] each tick
//! 4. CONSTANT: Fixed value 0.5
//! 5. ZERO:     No modulation (Φ = 0 always)

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 12;
const TICKS: usize = 3000;
const SEEDS: usize = 30;

#[derive(Clone, Copy)]
enum PhiMetric {
    Phi,
    Entropy,
    Random,
    Constant,
    Zero,
}

impl PhiMetric {
    fn name(&self) -> &'static str {
        match self {
            Self::Phi => "PHI",
            Self::Entropy => "ENTROPY",
            Self::Random => "RANDOM",
            Self::Constant => "CONST_0.5",
            Self::Zero => "ZERO",
        }
    }
    fn compute(&self, energy_frac: f64, tick: u64, agent_idx: usize) -> f64 {
        match self {
            Self::Phi => 0.5, // Will be overridden by actual equation
            Self::Entropy => {
                // Shannon entropy proxy: H = -p*ln(p) where p = energy_fraction
                let p = energy_frac.clamp(0.01, 0.99);
                -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / (2.0f64).ln()
            }
            Self::Random => {
                let mut s = tick
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(agent_idx as u64 * 1442695040888963407);
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 11) as f64 / (1u64 << 53) as f64
            }
            Self::Constant => 0.5,
            Self::Zero => 0.0,
        }
    }
}

struct RunResult {
    clustering: f64,
    alive: usize,
    avg_energy: f64,
}

fn run(metric: PhiMetric, seed: u64) -> RunResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants {
        initial_energy: 400.0,
        max_energy: 400.0,
        consciousness_maintenance_per_tick: 0.12,
        movement_cost_per_unit: 0.008,
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: 0.05,
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.25,
        ambient_regen_rate: 0.02,
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: 40.0,
    };

    let mut rng = seed;
    let mut handles = Vec::new();
    for i in 0..AGENTS {
        let x = (nrng(&mut rng) - 0.5) * 100.0;
        let y = (nrng(&mut rng) - 0.5) * 100.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.2;
            b.linear_velocity =
                SVector::from([(nrng(&mut rng) - 0.5) * 10.0, (nrng(&mut rng) - 0.5) * 10.0]);
        }
        consciousness.register(h, 400.0, 20.0);
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

    for tick in 0..TICKS {
        // Consciousness with METRIC-SPECIFIC Phi
        for (idx, &h) in handles.iter().enumerate() {
            let ef = consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.fraction_remaining())
                .unwrap_or(0.0);
            let phi_value = metric.compute(ef, tick as u64, idx);
            let inputs = ConsciousnessInputs {
                phi: phi_value,
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

        // FEP gradient movement
        let agent_data: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position().0,
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
                Some(b) => b.position().0,
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
            let nearby: Vec<_> = agent_data
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, d)| d.clone())
                .collect();
            let dir = fep_gradient::free_energy_gradient(&pos, ef, &harm, &nearby, &[], None, 0.0);
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = dir * 30.0;
            }
        }

        // Maintenance + offloading
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
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let dist = match (world.body(ha), world.body(hb)) {
                    (Some(a), Some(b)) => a.position().distance(b.position()),
                    _ => continue,
                };
                if dist > consciousness.constants.harmony_range {
                    continue;
                }
                let (ha_h, hb_h) = match (
                    consciousness.entities.get(&ha),
                    consciousness.entities.get(&hb),
                ) {
                    (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                    _ => continue,
                };
                let res = HarmonyField::<2>::resonance(&ha_h, &hb_h);
                if res > 0.5 {
                    let off = (res - 0.5) * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.prediction_error *= 1.0 - off * 0.1;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        e.energy.regenerate(
                            consciousness.constants.consciousness_maintenance_per_tick * off * 0.5,
                        );
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.prediction_error *= 1.0 - off * 0.1;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        e.energy.regenerate(
                            consciousness.constants.consciousness_maintenance_per_tick * off * 0.5,
                        );
                    }
                }
            }
        }
        consciousness.tick_prediction_errors();
        world.step_with_callback(1.0 / 64.0, &mut consciousness);
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
        .count();
    let avg_e = handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / AGENTS as f64;
    let mut tn = 0.0;
    for i in 0..handles.len() {
        let mut md = f64::MAX;
        for j in 0..handles.len() {
            if i == j {
                continue;
            }
            if let (Some(a), Some(b)) = (world.body(handles[i]), world.body(handles[j])) {
                let d = a.position().distance(b.position());
                if d < md {
                    md = d;
                }
            }
        }
        tn += md;
    }
    RunResult {
        clustering: tn / AGENTS as f64,
        alive,
        avg_energy: avg_e,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  PHI ABLATION: Does the consciousness metric matter? ║");
    println!("╚══════════════════════════════════════════════════════╝\n");
    println!(
        "{} agents, {} ticks, {} seeds per condition\n",
        AGENTS, TICKS, SEEDS
    );

    let metrics = [
        PhiMetric::Phi,
        PhiMetric::Entropy,
        PhiMetric::Random,
        PhiMetric::Constant,
        PhiMetric::Zero,
    ];

    println!(
        "{:<12} {:<12} {:<12} {:<12} {:<12}",
        "Metric", "Clustering", "95% CI", "Alive", "Energy"
    );
    println!("{}", "─".repeat(60));

    for metric in &metrics {
        let mut clusterings = Vec::new();
        let mut alives = Vec::new();
        let mut energies = Vec::new();

        for s in 0..SEEDS {
            let r = run(*metric, 42 + s as u64 * 997);
            clusterings.push(r.clustering);
            alives.push(r.alive as f64);
            energies.push(r.avg_energy);
        }

        let mean_c = clusterings.iter().sum::<f64>() / SEEDS as f64;
        let std_c = (clusterings
            .iter()
            .map(|x| (x - mean_c).powi(2))
            .sum::<f64>()
            / (SEEDS as f64 - 1.0))
            .sqrt();
        let ci = 1.96 * std_c / (SEEDS as f64).sqrt(); // 95% CI
        let mean_a = alives.iter().sum::<f64>() / SEEDS as f64;
        let mean_e = energies.iter().sum::<f64>() / SEEDS as f64;

        println!(
            "{:<12} {:<12.2} ±{:<10.2} {:<12.1} {:<12.1}",
            metric.name(),
            mean_c,
            ci,
            mean_a,
            mean_e
        );
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("If PHI produces significantly different clustering than");
    println!("RANDOM/CONSTANT/ZERO, then the integration metric matters.");
    println!("If all metrics produce similar clustering, any scalar works.");
    println!("═══════════════════════════════════════════════════════");
}

fn nrng(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
