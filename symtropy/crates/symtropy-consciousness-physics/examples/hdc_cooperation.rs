// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HDC vs Scalar cooperation experiment.
//!
//! Requires: `cargo run --example hdc_cooperation --features consciousness-hdc`
//!
//! Compares two Φ-computation substrates under identical thermodynamic pressure:
//! - SCALAR: MasterConsciousnessEquation → fixed Phi (current baseline)
//! - HDC: 16,384D thought vectors via HdcLtcUnifiedNetwork → emergent Phi
//!
//! The HDC condition uses phi_from_thought() which varies dynamically as the
//! LTC network evolves — producing genuine consciousness dynamics rather than
//! near-static equation output.
//!
//! Run: cargo run --example hdc_cooperation --release --features consciousness-hdc

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::ConvergenceDetector;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::hdc_context::{self, HdcConsciousnessContext};
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 12;
const MAX_TICKS: usize = 5_000;
const DT: f64 = 1.0 / 64.0;
const NUM_SEEDS: usize = 10;

#[derive(Clone, Copy)]
enum Condition {
    Scalar,
    Hdc,
}

impl Condition {
    fn name(&self) -> &'static str {
        match self {
            Condition::Scalar => "SCALAR",
            Condition::Hdc => "HDC",
        }
    }
}

struct RunResult {
    converged: bool,
    convergence_tick: usize,
    final_jphi: f64,
    final_phi: f64,
    alive: usize,
    clustering: f64,
    thought_diversity: f64, // mean pairwise thought similarity (HDC only)
}

fn run_experiment(condition: Condition, seed: u64) -> RunResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let well_positions = vec![SVector::from([25.0, 25.0]), SVector::from([-25.0, -25.0])];

    // HDC contexts (one per agent for HDC condition)
    let mut hdc_contexts: Vec<Option<HdcConsciousnessContext>> = Vec::new();

    let mut rng = seed;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 80.0;
        let y = (rng_f64(&mut rng) - 0.5) * 80.0;
        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(body) = world.body_mut(h) {
            body.linear_damping = 0.05;
        }
        consciousness.register(
            h,
            consciousness.constants.initial_energy,
            consciousness.constants.harmony_range,
        );

        if let Some(entity) = consciousness.entities.get_mut(&h) {
            match i % 4 {
                0 => entity.harmony_activations = [0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5],
                1 => entity.harmony_activations = [0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.2, 0.5],
                2 => entity.harmony_activations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                _ => entity.harmony_activations = [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.3, 0.5],
            }
        }

        // Create HDC context for HDC condition
        let hdc = match condition {
            Condition::Hdc => Some(HdcConsciousnessContext::new(seed + i as u64 * 31)),
            Condition::Scalar => None,
        };
        hdc_contexts.push(hdc);
        handles.push(h);
    }

    let mut jphi_detector = ConvergenceDetector::new(200, 1e-3);
    let mut convergence_tick = MAX_TICKS;
    let mut converged = false;

    for tick in 0..MAX_TICKS {
        // 1. Update consciousness
        for (idx, &h) in handles.iter().enumerate() {
            let entity = consciousness.entities.get(&h);
            let collapsed = entity.map(|e| e.energy.is_collapsed()).unwrap_or(true);
            let energy_frac = entity.map(|e| e.energy.fraction_remaining()).unwrap_or(0.0);
            let pred_error = entity.map(|e| e.prediction_error).unwrap_or(0.0);
            let motor_prec = entity.map(|e| e.motor_precision).unwrap_or(1.0);
            let harmony_total = entity.map(|e| e.total_harmony_energy()).unwrap_or(0.0);
            let harmony_acts = entity.map(|e| e.harmony_activations).unwrap_or([0.0; 9]);

            let nearby = if let Some(body) = world.body(h) {
                let pos = body.position().0;
                handles
                    .iter()
                    .filter(|&&oh| {
                        oh != h
                            && world
                                .body(oh)
                                .map(|ob| {
                                    (ob.position().0 - pos).norm()
                                        < consciousness.constants.harmony_range
                                })
                                .unwrap_or(false)
                    })
                    .count()
            } else {
                0
            };

            if !collapsed {
                match condition {
                    Condition::Scalar => {
                        let inputs = ConsciousnessInputs {
                            phi: energy_frac,
                            broadcast: (nearby as f64 / 5.0).min(1.0),
                            working_memory: (1.0 - pred_error).max(0.0),
                            attention: 0.5,
                            recurrence: motor_prec,
                            embodiment: (1.0 - energy_frac * 0.3).max(0.0),
                            knowledge: (harmony_total / 8.0).min(1.0),
                            synchrony: consciousness.collective_phi.max(0.5),
                        };
                        let old_phi = consciousness.phi(h);
                        consciousness.update_entity(h, &inputs, Point::origin());
                        let new_phi = consciousness.phi(h);
                        consciousness.ledger.record_phi_change(new_phi - old_phi);
                    }
                    Condition::Hdc => {
                        // Step HDC context with dynamic inputs
                        let inputs_arr = hdc_context::inputs_from_state(
                            energy_frac,
                            nearby,
                            pred_error,
                            0.0,
                            motor_prec,
                            harmony_total,
                            consciousness.collective_phi,
                        );
                        if let Some(ref mut hdc) = hdc_contexts[idx] {
                            hdc.step(&inputs_arr, &harmony_acts, DT as f32);
                            let hdc_phi = hdc.phi_from_thought(&harmony_acts);

                            // Use HDC phi to update safety tier
                            if let Some(entity) = consciousness.entities.get_mut(&h) {
                                let old_phi = entity.phi();
                                // Override the equation result with HDC phi
                                entity.result =
                                    Some(symthaea_consciousness_equation::ConsciousnessResult {
                                        consciousness_level: hdc_phi,
                                        bottleneck_factor: energy_frac,
                                        weighted_sum: hdc_phi,
                                        embodiment_factor: 1.0,
                                        narrative_coherence: 1.0,
                                        social_embedding: 1.0,
                                        temporal_stability: 1.0,
                                        bottleneck_name: "hdc".to_string(),
                                        factors: ConsciousnessInputs {
                                            phi: energy_frac,
                                            broadcast: 0.5,
                                            working_memory: 0.5,
                                            attention: 0.5,
                                            recurrence: 0.5,
                                            embodiment: 0.5,
                                            knowledge: 0.5,
                                            synchrony: 0.5,
                                        },
                                    });
                                entity.safety_tier =
                                    symtropy_consciousness_physics::SafetyTier::from_phi(hdc_phi);
                                consciousness.ledger.record_phi_change(hdc_phi - old_phi);
                            }
                        }
                    }
                }
            } else {
                // Collapsed: zero phi
                let inputs = ConsciousnessInputs {
                    phi: 0.0,
                    broadcast: 0.0,
                    working_memory: 0.0,
                    attention: 0.0,
                    recurrence: 0.0,
                    embodiment: 0.0,
                    knowledge: 0.0,
                    synchrony: 0.0,
                };
                consciousness.update_entity(h, &inputs, Point::origin());
            }
        }

        // 2. FEP gradient movement
        let agent_data: Vec<(SVector<f64, 2>, [f64; 9])> = handles
            .iter()
            .filter_map(|&h| {
                let body = world.body(h)?;
                let entity = consciousness.entities.get(&h)?;
                Some((body.position().0, entity.harmony_activations))
            })
            .collect();
        let well_data: Vec<(SVector<f64, 2>, f64)> =
            well_positions.iter().map(|&p| (p, 1.0)).collect();

        for &h in &handles {
            let Some(body) = world.body(h) else { continue };
            let Some(entity) = consciousness.entities.get(&h) else {
                continue;
            };
            if entity.energy.is_collapsed() {
                continue;
            }
            let pos = body.position().0;
            let ef = entity.energy.fraction_remaining();
            let harmony = entity.harmony_activations;
            let nearby: Vec<_> = agent_data
                .iter()
                .filter(|(p, _)| (p - pos).norm() > 2.0)
                .cloned()
                .collect();
            let dir = fep_gradient::free_energy_gradient(
                &pos, ef, &harmony, &nearby, &well_data, None, 0.0,
            );
            if let Some(body) = world.body_mut(h) {
                body.linear_velocity = dir * 25.0;
            }
        }

        // 3. Maintenance + regen
        let maintenance_rate = consciousness.constants.consciousness_maintenance_per_tick;
        let ambient_rate = consciousness.constants.ambient_regen_rate;
        let well_rate = consciousness.constants.energy_well_regen_rate;
        let regen_mult = consciousness.resource_regeneration_multiplier();

        let near_wells: Vec<bool> = handles
            .iter()
            .map(|&h| {
                world
                    .body(h)
                    .map(|body| {
                        let pos = body.position().0;
                        well_positions.iter().any(|&wp| (pos - wp).norm() < 35.0)
                    })
                    .unwrap_or(false)
            })
            .collect();

        for (idx, &h) in handles.iter().enumerate() {
            if let Some(entity) = consciousness.entities.get_mut(&h) {
                entity.energy.tick_reset();
            }
            let phi = consciousness.phi(h);
            consciousness.consume_energy(h, maintenance_rate * (1.0 + phi * 0.5));
            if let Some(entity) = consciousness.entities.get_mut(&h) {
                entity.energy.regenerate(ambient_rate * regen_mult);
                if near_wells[idx] {
                    entity.energy.regenerate(well_rate);
                }
            }
        }

        // 4. Harmony offloading
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
                let (harm_a, harm_b) = {
                    let ea = consciousness.entities.get(&ha);
                    let eb = consciousness.entities.get(&hb);
                    match (ea, eb) {
                        (Some(a), Some(b)) => (a.harmony_activations, b.harmony_activations),
                        _ => continue,
                    }
                };
                let resonance = HarmonyField::<2>::resonance(&harm_a, &harm_b);
                if resonance > 0.5 {
                    let regen = consciousness.constants.harmony_resonance_regen_rate
                        * (resonance - 0.5)
                        * 2.0;
                    if let Some(e) = consciousness.entities.get_mut(&ha) {
                        e.energy.regenerate(regen);
                    }
                    if let Some(e) = consciousness.entities.get_mut(&hb) {
                        e.energy.regenerate(regen);
                    }
                }
            }
        }

        // 5. Physics + thermo
        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        let balance = consciousness.tick_thermodynamics();

        // 6. Track convergence
        if let Some(jphi) = balance.joules_per_phi {
            if jphi.is_finite() && jphi_detector.push(jphi) && !converged {
                converged = true;
                convergence_tick = tick;
            }
        }

        // CSV every 500 ticks
        if tick % 500 == 0 {
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
            let clustering = avg_nn(&world, &handles);
            let jphi_str = balance
                .joules_per_phi
                .filter(|j| j.is_finite())
                .map(|j| format!("{j:.2}"))
                .unwrap_or("N/A".into());
            println!(
                "{},{seed},{tick},{:.4},{jphi_str},{alive},{clustering:.2}",
                condition.name(),
                consciousness.collective_phi
            );
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
        .count();

    // Thought diversity (HDC only)
    let thought_diversity = match condition {
        Condition::Hdc => {
            let active: Vec<&HdcConsciousnessContext> =
                hdc_contexts.iter().filter_map(|c| c.as_ref()).collect();
            if active.len() >= 2 {
                let mut total_sim = 0.0;
                let mut count = 0;
                for i in 0..active.len() {
                    for j in (i + 1)..active.len() {
                        total_sim += hdc_context::thought_resonance(
                            active[i].thought_hv(),
                            active[j].thought_hv(),
                        );
                        count += 1;
                    }
                }
                if count > 0 {
                    total_sim / count as f64
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        Condition::Scalar => 0.0,
    };

    RunResult {
        converged,
        convergence_tick,
        final_jphi: jphi_detector.rolling_mean(),
        final_phi: consciousness.collective_phi,
        alive,
        clustering: avg_nn(&world, &handles),
        thought_diversity,
    }
}

fn main() {
    eprintln!("=== HDC vs Scalar Cooperation Experiment ===");
    eprintln!("Agents: {AGENTS}, Ticks: {MAX_TICKS}, Seeds: {NUM_SEEDS}");

    println!("condition,seed,tick,collective_phi,joules_per_phi,alive,clustering");

    for condition in &[Condition::Scalar, Condition::Hdc] {
        let mut results = Vec::new();
        for s in 0..NUM_SEEDS {
            let seed = 42 + s as u64 * 997;
            eprintln!("  {} seed={seed}...", condition.name());
            results.push(run_experiment(*condition, seed));
        }

        let n = results.len() as f64;
        let converged_count = results.iter().filter(|r| r.converged).count();
        let mean_phi = results.iter().map(|r| r.final_phi).sum::<f64>() / n;
        let mean_alive = results.iter().map(|r| r.alive as f64).sum::<f64>() / n;
        let mean_cluster = results.iter().map(|r| r.clustering).sum::<f64>() / n;
        let mean_diversity = results.iter().map(|r| r.thought_diversity).sum::<f64>() / n;

        let phi_vals: Vec<f64> = results.iter().map(|r| r.final_phi).collect();
        let phi_ci = 1.96 * std_dev(&phi_vals) / n.sqrt();

        eprintln!(
            "\n── {} ({}/{} converged) ──",
            condition.name(),
            converged_count,
            NUM_SEEDS
        );
        eprintln!("  Final Φ:           {:.4} ± {:.4}", mean_phi, phi_ci);
        eprintln!("  Alive:             {:.1}/{AGENTS}", mean_alive);
        eprintln!("  Clustering:        {:.2}", mean_cluster);
        if matches!(condition, Condition::Hdc) {
            eprintln!(
                "  Thought diversity: {:.4} (mean pairwise similarity)",
                mean_diversity
            );
        }
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

fn std_dev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn rng_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}
