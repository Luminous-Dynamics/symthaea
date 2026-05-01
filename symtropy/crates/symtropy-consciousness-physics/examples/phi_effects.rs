// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PHI EFFECTS: What does the consciousness metric ACTUALLY affect?
//!
//! The phi_ablation showed clustering is Φ-invariant. This experiment
//! tests 5 dependent variables where Φ SHOULD matter:
//!
//! 1. SURVIVAL TIME: Higher Φ = higher maintenance cost = faster death?
//! 2. MOTOR QUALITY: Φ determines safety tier = movement smoothness?
//! 3. COOPERATION STABILITY: Do clusters persist differently?
//! 4. PERTURBATION RESILIENCE: Mid-run danger injection
//! 5. INFORMATION FLOW: Velocity correlation between nearby agents
//!
//! 5 metrics × 20 seeds × 3000 ticks

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, SafetyTier, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 24;
const TICKS: usize = 3000;
const SEEDS: usize = 20;
const DANGER_START_TICK: usize = 1500; // Inject danger halfway

#[derive(Clone, Copy)]
enum Metric {
    Phi,
    Entropy,
    Random,
    Constant,
    Zero,
}
impl Metric {
    fn name(&self) -> &'static str {
        match self {
            Self::Phi => "PHI",
            Self::Entropy => "ENTROPY",
            Self::Random => "RANDOM",
            Self::Constant => "CONST",
            Self::Zero => "ZERO",
        }
    }
    fn value(&self, ef: f64, tick: u64, idx: usize) -> f64 {
        match self {
            Self::Phi => 0.5,
            Self::Entropy => {
                let p = ef.clamp(0.01, 0.99);
                -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / (2.0f64).ln()
            }
            Self::Random => {
                let mut s = tick
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(idx as u64 * 1442695040888963407);
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 11) as f64 / (1u64 << 53) as f64
            }
            Self::Constant => 0.5,
            Self::Zero => 0.0,
        }
    }
}

#[derive(Default)]
struct Results {
    avg_survival_ticks: f64,    // 1. SURVIVAL
    avg_velocity_variance: f64, // 2. MOTOR QUALITY (lower = smoother)
    cluster_persistence: f64,   // 3. STABILITY (fraction of time in stable cluster)
    post_danger_survival: f64,  // 4. RESILIENCE (alive after danger injection)
    velocity_correlation: f64,  // 5. INFO FLOW (avg correlation of nearby agent velocities)
}

fn run(metric: Metric, seed: u64) -> Results {
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
    let area = 120.0;
    for i in 0..AGENTS {
        let x = (nr(&mut rng) - 0.5) * area;
        let y = (nr(&mut rng) - 0.5) * area;
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

    let mut alive_ticks = vec![0u64; AGENTS];
    let mut velocity_variances = vec![Vec::new(); AGENTS];
    let mut prev_velocities: Vec<Option<SVector<f64, 2>>> = vec![None; AGENTS];
    let mut cluster_stable_ticks = 0u64;
    let mut total_vel_correlation = 0.0;
    let mut correlation_samples = 0u64;
    let mut prev_nn = vec![usize::MAX; AGENTS]; // previous nearest neighbor

    for tick in 0..TICKS {
        let danger = if tick >= DANGER_START_TICK { 0.8 } else { 0.0 };
        let danger_pos = SVector::from([0.0, 0.0]); // center

        // Consciousness with metric
        for (idx, &h) in handles.iter().enumerate() {
            let ef = consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.fraction_remaining())
                .unwrap_or(0.0);
            let phi_val = metric.value(ef, tick as u64, idx);
            let inputs = ConsciousnessInputs {
                phi: phi_val,
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

        // FEP gradient + danger avoidance
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
            let ds = if danger > 0.1 {
                Some(&danger_pos)
            } else {
                None
            };
            let dir = fep_gradient::free_energy_gradient(&pos, ef, &harm, &nearby, &[], ds, danger);
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

        // === MEASUREMENTS ===

        // 1. Survival ticks
        for (idx, &h) in handles.iter().enumerate() {
            if !consciousness
                .entities
                .get(&h)
                .map(|e| e.energy.is_collapsed())
                .unwrap_or(true)
            {
                alive_ticks[idx] += 1;
            }
        }

        // 2. Velocity variance (motor smoothness)
        for (idx, &h) in handles.iter().enumerate() {
            if let Some(b) = world.body(h) {
                let v = b.linear_velocity;
                if let Some(pv) = &prev_velocities[idx] {
                    let dv = (v - pv).norm();
                    velocity_variances[idx].push(dv);
                }
                prev_velocities[idx] = Some(v);
            }
        }

        // 3. Cluster stability (is my nearest neighbor the same as last tick?)
        if tick % 10 == 0 {
            let mut stable = 0;
            for i in 0..handles.len() {
                let mut nn = 0;
                let mut mnd = f64::MAX;
                for j in 0..handles.len() {
                    if i == j {
                        continue;
                    }
                    if let (Some(a), Some(b)) = (world.body(handles[i]), world.body(handles[j])) {
                        let d = a.position().distance(b.position());
                        if d < mnd {
                            mnd = d;
                            nn = j;
                        }
                    }
                }
                if nn == prev_nn[i] {
                    stable += 1;
                }
                prev_nn[i] = nn;
            }
            if tick > 10 {
                cluster_stable_ticks += stable as u64;
            }
        }

        // 5. Velocity correlation (info flow between nearby agents)
        if tick % 20 == 0 {
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    if let (Some(a), Some(b)) = (world.body(handles[i]), world.body(handles[j])) {
                        let dist = a.position().distance(b.position());
                        if dist < 40.0 {
                            let va = a.linear_velocity;
                            let vb = b.linear_velocity;
                            let na = va.norm();
                            let nb = vb.norm();
                            if na > 0.1 && nb > 0.1 {
                                let cos = va.dot(&vb) / (na * nb);
                                total_vel_correlation += cos;
                                correlation_samples += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Aggregate
    let avg_surv = alive_ticks.iter().sum::<u64>() as f64 / AGENTS as f64;
    let avg_vv: f64 = velocity_variances
        .iter()
        .filter(|v| !v.is_empty())
        .map(|v| v.iter().sum::<f64>() / v.len() as f64)
        .sum::<f64>()
        / AGENTS as f64;
    let total_stability_samples = ((TICKS / 10) - 1) as f64 * AGENTS as f64;
    let cluster_pers = if total_stability_samples > 0.0 {
        cluster_stable_ticks as f64 / total_stability_samples
    } else {
        0.0
    };

    // 4. Post-danger survival
    let post_danger_alive = handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count();

    let avg_corr = if correlation_samples > 0 {
        total_vel_correlation / correlation_samples as f64
    } else {
        0.0
    };

    Results {
        avg_survival_ticks: avg_surv,
        avg_velocity_variance: avg_vv,
        cluster_persistence: cluster_pers,
        post_danger_survival: post_danger_alive as f64 / AGENTS as f64,
        velocity_correlation: avg_corr,
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  PHI EFFECTS: What does the consciousness metric AFFECT?  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    println!(
        "{} agents, {} ticks, {} seeds, danger injected at tick {}\n",
        AGENTS, TICKS, SEEDS, DANGER_START_TICK
    );

    let metrics = [
        Metric::Phi,
        Metric::Entropy,
        Metric::Random,
        Metric::Constant,
        Metric::Zero,
    ];

    println!(
        "{:<10} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "Metric", "Survival", "MotorVar", "ClustStab", "PostDanger", "VelCorr"
    );
    println!("{}", "─".repeat(70));

    for m in &metrics {
        let mut rs = Vec::new();
        for s in 0..SEEDS {
            rs.push(run(*m, 42 + s as u64 * 997));
        }
        let n = SEEDS as f64;
        let surv = rs.iter().map(|r| r.avg_survival_ticks).sum::<f64>() / n;
        let vv = rs.iter().map(|r| r.avg_velocity_variance).sum::<f64>() / n;
        let cs = rs.iter().map(|r| r.cluster_persistence).sum::<f64>() / n;
        let pd = rs.iter().map(|r| r.post_danger_survival).sum::<f64>() / n;
        let vc = rs.iter().map(|r| r.velocity_correlation).sum::<f64>() / n;

        println!(
            "{:<10} {:<12.0} {:<12.4} {:<12.3} {:<12.1}% {:<12.4}",
            m.name(),
            surv,
            vv,
            cs,
            pd * 100.0,
            vc
        );
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("SURVIVAL:    Avg ticks alive per agent (max {})", TICKS);
    println!("MOTOR VAR:   Velocity change per tick (lower = smoother)");
    println!("CLUST STAB:  Fraction of ticks with same nearest neighbor");
    println!(
        "POST DANGER: % alive after danger injection at tick {}",
        DANGER_START_TICK
    );
    println!("VEL CORR:    Velocity alignment of nearby agents [-1,1]");
    println!("═══════════════════════════════════════════════════════");
}

fn nr(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
