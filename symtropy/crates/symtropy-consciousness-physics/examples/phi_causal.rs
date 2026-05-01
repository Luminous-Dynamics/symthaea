// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PHI CAUSAL TEST: Does Φ matter when properly wired into the gradient?
//!
//! Previous experiments showed Φ is epiphenomenal because the gradient
//! function didn't take Φ as input. Now we use free_energy_gradient_phi()
//! which couples Φ into:
//! 1. Cooperation urgency (Φ amplifies social drive)
//! 2. Resonance gating (Φ expands who you're attracted to)
//! 3. Danger sensitivity (Φ lowers threat detection threshold)
//!
//! If Φ now produces DIFFERENT clustering than random/zero, then
//! integration-specific effects exist when properly wired.
//! If still identical, the universality result holds even under coupling.

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 24;
const TICKS: usize = 3000;
const SEEDS: usize = 20;

#[derive(Clone, Copy)]
enum Metric {
    PhiCoupled,
    EntropyCoupled,
    RandomCoupled,
    ConstantCoupled,
    ZeroCoupled,
    Decoupled,
}
impl Metric {
    fn name(&self) -> &'static str {
        match self {
            Self::PhiCoupled => "Φ-COUPLED",
            Self::EntropyCoupled => "H-COUPLED",
            Self::RandomCoupled => "RAND-COUPLED",
            Self::ConstantCoupled => "CONST-COUPLED",
            Self::ZeroCoupled => "ZERO-COUPLED",
            Self::Decoupled => "DECOUPLED",
        }
    }
    fn phi_for_gradient(&self, ef: f64, tick: u64, idx: usize) -> Option<f64> {
        match self {
            Self::PhiCoupled => Some(0.5),
            Self::EntropyCoupled => {
                let p = ef.clamp(0.01, 0.99);
                Some(-(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / (2.0f64).ln())
            }
            Self::RandomCoupled => {
                let mut s = tick
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(idx as u64 * 1442695040888963407);
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                Some((s >> 11) as f64 / (1u64 << 53) as f64)
            }
            Self::ConstantCoupled => Some(0.5),
            Self::ZeroCoupled => Some(0.0),
            Self::Decoupled => None, // original gradient, no Φ
        }
    }
}

struct Result {
    clustering: f64,
    alive: usize,
    avg_energy: f64,
    avg_survival: f64,
}

fn run(metric: Metric, seed: u64) -> Result {
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
        let x = (nr(&mut rng) - 0.5) * 120.0;
        let y = (nr(&mut rng) - 0.5) * 120.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.2;
            b.linear_velocity =
                SVector::from([(nr(&mut rng) - 0.5) * 10.0, (nr(&mut rng) - 0.5) * 10.0]);
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
    let mut alive_sum = vec![0u64; AGENTS];

    for tick in 0..TICKS {
        for (idx, &h) in handles.iter().enumerate() {
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
            if !consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.is_collapsed())
                .unwrap_or(true)
            {
                alive_sum[idx] += 1;
            }
        }

        // FEP gradient WITH PHI COUPLING
        let ad: Vec<_> = handles
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
            let nearby: Vec<_> = ad
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, d)| d.clone())
                .collect();

            let phi_val = metric.phi_for_gradient(ef, tick as u64, idx);
            let dir = fep_gradient::free_energy_gradient_phi(
                &pos,
                ef,
                phi_val,
                &harm,
                &nearby,
                &[],
                None,
                0.0,
            );
            if let Some(b) = world.body_mut(h) {
                b.linear_velocity = dir * 30.0;
            }
        }

        // Maintenance + offloading (same for all conditions)
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
    let ae = handles
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
    let avg_surv = alive_sum.iter().sum::<u64>() as f64 / AGENTS as f64;
    Result {
        clustering: tn / AGENTS as f64,
        alive,
        avg_energy: ae,
        avg_survival: avg_surv,
    }
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  PHI CAUSAL: Does Φ matter when wired INTO the gradient?  ║");
    println!("╚═════════════════════════════════════════════════════════════╝\n");
    println!("{} agents, {} ticks, {} seeds\n", AGENTS, TICKS, SEEDS);

    let metrics = [
        Metric::PhiCoupled,
        Metric::EntropyCoupled,
        Metric::RandomCoupled,
        Metric::ConstantCoupled,
        Metric::ZeroCoupled,
        Metric::Decoupled,
    ];

    println!(
        "{:<16} {:<10} {:<8} {:<10} {:<10} {:<10}",
        "Condition", "Cluster", "±CI", "Alive", "Energy", "Survival"
    );
    println!("{}", "─".repeat(64));

    for m in &metrics {
        let mut cs = Vec::new();
        let mut als = Vec::new();
        let mut es = Vec::new();
        let mut ss = Vec::new();
        for s in 0..SEEDS {
            let r = run(*m, 42 + s as u64 * 997);
            cs.push(r.clustering);
            als.push(r.alive as f64);
            es.push(r.avg_energy);
            ss.push(r.avg_survival);
        }
        let n = SEEDS as f64;
        let mc = cs.iter().sum::<f64>() / n;
        let sc = (cs.iter().map(|x| (x - mc).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let ci = 1.96 * sc / n.sqrt();
        println!(
            "{:<16} {:<10.2} ±{:<6.2} {:<10.1} {:<10.1} {:<10.0}",
            m.name(),
            mc,
            ci,
            als.iter().sum::<f64>() / n,
            es.iter().sum::<f64>() / n,
            ss.iter().sum::<f64>() / n
        );
    }

    println!("\n═════════════════════════════════════════════════════════");
    println!("DECOUPLED = original gradient (Φ not in function)");
    println!("Φ-COUPLED = Φ modulates urgency, resonance gating, danger sensitivity");
    println!("If Φ-COUPLED ≠ ZERO-COUPLED, then integration matters when wired.");
    println!("If Φ-COUPLED = ZERO-COUPLED, universality holds even under coupling.");
    println!("═════════════════════════════════════════════════════════");
}

fn nr(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
