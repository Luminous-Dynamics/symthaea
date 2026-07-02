// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! EPISTEMIC CRISIS: FEP agents in D dimensions encountering D+1 physics.
//!
//! The definitive experiment for the dimensional leakage paper.
//!
//! Setup: Agents live on a 2D brane (first 2 coordinates).
//! Energy objects ("resources") exist in full D-dimensional space.
//! Resources drift along ALL hidden axes (not just the last one).
//! Agents can only sense the 2D projection.
//!
//! When resources drift away from the brane (hidden coords > threshold),
//! they vanish from the agents' perspective. Energy disappears.
//! The agents' 2D world model cannot explain this.
//!
//! Measures:
//! 1. Prediction error vs number of hidden dimensions (D-2)
//! 2. Survival impact of leakage
//! 3. "Dark energy" — total energy in the bulk invisible to brane agents
//! 4. Whether agents cluster differently under epistemic crisis
//! 5. Resource disappearance rate vs dimensionality

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 10;
const RESOURCES: usize = 6;
const TICKS: usize = 2000;
const SEEDS: usize = 15;
const BRANE_THRESHOLD: f64 = 3.0; // visible if all hidden coords < this

struct DimResult {
    pred_error_mean: f64,
    pred_error_max: f64,
    alive: f64,
    energy: f64,
    resources_visible: f64, // fraction still on brane at end
    dark_energy_frac: f64,  // fraction of total KE invisible to brane
    clustering: f64,
    disappearance_tick: f64, // avg tick when first resource vanishes
}

fn run_at_dim(dim: usize, seed: u64) -> DimResult {
    match dim {
        2 => run::<2>(seed),
        3 => run::<3>(seed),
        4 => run::<4>(seed),
        _ => DimResult {
            pred_error_mean: 0.0,
            pred_error_max: 0.0,
            alive: 0.0,
            energy: 0.0,
            resources_visible: 0.0,
            dark_energy_frac: 0.0,
            clustering: 0.0,
            disappearance_tick: 0.0,
        },
    }
}

fn run<const D: usize>(seed: u64) -> DimResult {
    let mut world = PhysicsWorld::<D>::new(SVector::zeros());
    let mut consciousness = ConsciousnessField::<D>::new();
    consciousness.constants = ThermodynamicConstants {
        initial_energy: 500.0,
        max_energy: 500.0,
        consciousness_maintenance_per_tick: 0.10,
        movement_cost_per_unit: 0.006,
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: 0.05,
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.0,
        ambient_regen_rate: 0.03,
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: 40.0,
    };

    let hidden_dims = if D > 2 { D - 2 } else { 0 };
    let mut rng = seed;
    let mut agent_handles = Vec::new();
    let mut resource_handles = Vec::new();

    // Agents on the 2D brane
    for i in 0..AGENTS {
        let mut coords = [0.0f64; D];
        coords[0] = (nr(&mut rng) - 0.5) * 80.0;
        if D > 1 {
            coords[1] = (nr(&mut rng) - 0.5) * 80.0;
        }

        let h = world.add_sphere(Point::new(coords), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.3;
            let mut vel = SVector::<f64, D>::zeros();
            vel[0] = (nr(&mut rng) - 0.5) * 8.0;
            if D > 1 {
                vel[1] = (nr(&mut rng) - 0.5) * 8.0;
            }
            b.linear_velocity = vel;
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
        agent_handles.push(h);
    }

    // Resources: start on brane, drift into hidden dimensions
    for _ in 0..RESOURCES {
        let mut coords = [0.0f64; D];
        coords[0] = (nr(&mut rng) - 0.5) * 60.0;
        if D > 1 {
            coords[1] = (nr(&mut rng) - 0.5) * 60.0;
        }

        let h = world.add_sphere(Point::new(coords), 1.5, 3.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.0; // NO DAMPING — resources drift freely
            b.restitution = 0.9;
            // Give velocity in ALL hidden dimensions
            let mut vel = SVector::<f64, D>::zeros();
            for d in 2..D {
                vel[d] = (nr(&mut rng) - 0.3) * 1.5; // mostly positive = drifting away
            }
            b.linear_velocity = vel;
        }
        resource_handles.push(h);
    }

    let mut max_pe = 0.0f64;
    let mut pe_sum = 0.0f64;
    let mut pe_samples = 0u64;
    let mut first_disappearance = TICKS;
    let mut prev_visible_count = RESOURCES;

    for tick in 0..TICKS {
        // Consciousness
        for &h in &agent_handles {
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

        // FEP gradient (brane-locked)
        let ad: Vec<_> = agent_handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position(),
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        for (idx, &h) in agent_handles.iter().enumerate() {
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
            let dir = fep_gradient::free_energy_gradient(&pos, ef, &harm, &nearby, &[], None, 0.0);
            if let Some(b) = world.body_mut(h) {
                let mut vel = dir * 25.0;
                for d in 2..D {
                    vel[d] = 0.0;
                } // locked to brane
                b.linear_velocity = vel;
            }
        }

        // Maintenance
        let rm = consciousness.resource_regeneration_multiplier();
        for &h in &agent_handles {
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

        // Offloading
        for i in 0..agent_handles.len() {
            for j in (i + 1)..agent_handles.len() {
                let (ha, hb) = (agent_handles[i], agent_handles[j]);
                let dist = match (world.body(ha), world.body(hb)) {
                    (Some(a), Some(b)) => {
                        // Distance in BRANE dimensions only (first 2)
                        let pa = a.position();
                        let pb = b.position();
                        let dx = pa[0] - pb[0];
                        let dy = if D > 1 { pa[1] - pb[1] } else { 0.0 };
                        (dx * dx + dy * dy).sqrt()
                    }
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
                let res = HarmonyField::<D>::resonance(&hah, &hbh);
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

        // Count visible resources (all hidden coords within threshold)
        let visible = resource_handles
            .iter()
            .filter(|&&h| {
                if let Some(body) = world.body(h) {
                    let pos = body.position();
                    for d in 2..D {
                        if pos[d].abs() > BRANE_THRESHOLD {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            })
            .count();

        // Detect disappearance → prediction error spike
        if visible < prev_visible_count {
            let vanished = prev_visible_count - visible;
            let err = vanished as f64 * 0.5; // strong error per vanished resource
            for &h in &agent_handles {
                if let Some(e) = consciousness.entities.get_mut(&h) {
                    e.prediction_error += err;
                    e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                    if e.prediction_error > max_pe {
                        max_pe = e.prediction_error;
                    }
                }
            }
            if first_disappearance == TICKS {
                first_disappearance = tick;
            }
        }
        prev_visible_count = visible;

        // Track prediction error
        for &h in &agent_handles {
            if let Some(e) = consciousness.entities.get(&h) {
                pe_sum += e.prediction_error;
                pe_samples += 1;
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(1.0 / 64.0, &mut consciousness);
        consciousness.tick_thermodynamics();
    }

    // Final measurements
    let alive = agent_handles
        .iter()
        .filter(|h| {
            consciousness
                .entities
                .get(h)
                .map(|e| !e.energy.is_collapsed())
                .unwrap_or(false)
        })
        .count();
    let ae = agent_handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
        .sum::<f64>()
        / AGENTS as f64;

    let final_visible = resource_handles
        .iter()
        .filter(|&&h| {
            if let Some(body) = world.body(h) {
                let pos = body.position();
                for d in 2..D {
                    if pos[d].abs() > BRANE_THRESHOLD {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        })
        .count();

    let mut total_ke = 0.0f64;
    let mut visible_ke = 0.0f64;
    for &h in &resource_handles {
        if let Some(body) = world.body(h) {
            let ke = body.kinetic_energy();
            total_ke += ke;
            let pos = body.position();
            let on_brane = (2..D).all(|d| pos[d].abs() < BRANE_THRESHOLD);
            if on_brane {
                visible_ke += ke;
            }
        }
    }

    let mut tn = 0.0;
    for i in 0..agent_handles.len() {
        let mut md = f64::MAX;
        for j in 0..agent_handles.len() {
            if i == j {
                continue;
            }
            if let (Some(a), Some(b)) = (world.body(agent_handles[i]), world.body(agent_handles[j]))
            {
                let d = a.position().metric_distance(&b.position());
                if d < md {
                    md = d;
                }
            }
        }
        tn += md;
    }

    DimResult {
        pred_error_mean: if pe_samples > 0 {
            pe_sum / pe_samples as f64
        } else {
            0.0
        },
        pred_error_max: max_pe,
        alive: alive as f64,
        energy: ae,
        resources_visible: final_visible as f64 / RESOURCES as f64,
        dark_energy_frac: if total_ke > 0.0 {
            1.0 - visible_ke / total_ke
        } else {
            0.0
        },
        clustering: tn / AGENTS as f64,
        disappearance_tick: first_disappearance as f64,
    }
}

fn main() {
    println!(
        "dim,hidden,pred_error_mean,pred_error_max,alive,energy,visible_frac,dark_energy_frac,clustering,first_vanish"
    );

    for d in [2, 3, 4] {
        let hidden = if d > 2 { d - 2 } else { 0 };
        let mut pe_m = Vec::new();
        let mut pe_x = Vec::new();
        let mut al = Vec::new();
        let mut en = Vec::new();
        let mut vi = Vec::new();
        let mut de = Vec::new();
        let mut cl = Vec::new();
        let mut fv = Vec::new();

        for s in 0..SEEDS {
            let r = run_at_dim(d, 42 + s as u64 * 997);
            pe_m.push(r.pred_error_mean);
            pe_x.push(r.pred_error_max);
            al.push(r.alive);
            en.push(r.energy);
            vi.push(r.resources_visible);
            de.push(r.dark_energy_frac);
            cl.push(r.clustering);
            fv.push(r.disappearance_tick);
        }

        let n = SEEDS as f64;
        println!(
            "{},{},{:.4},{:.3},{:.1},{:.1},{:.1},{:.3},{:.2},{:.0}",
            d,
            hidden,
            pe_m.iter().sum::<f64>() / n,
            pe_x.iter().sum::<f64>() / n,
            al.iter().sum::<f64>() / n,
            en.iter().sum::<f64>() / n,
            vi.iter().sum::<f64>() / n * 100.0,
            de.iter().sum::<f64>() / n,
            cl.iter().sum::<f64>() / n,
            fv.iter().sum::<f64>() / n,
        );
    }

    eprintln!("\n═══════════════════════════════════════════════════════════");
    eprintln!("EPISTEMIC CRISIS: FEP agents on a 2D brane in D-dimensional space");
    eprintln!("Resources drift into hidden dimensions. Agents can't see them leave.");
    eprintln!("hidden=0: no leakage (2D control). hidden=1,2: resources escape.");
    eprintln!("═══════════════════════════════════════════════════════════");
}

fn nr(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
