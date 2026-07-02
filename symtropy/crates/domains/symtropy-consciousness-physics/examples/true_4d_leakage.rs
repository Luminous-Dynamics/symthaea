// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TRUE 4D LEAKAGE: Physics runs in 4D. Agents sense only 3D.
//!
//! The physics world is PhysicsWorld<4>. All bodies have (x,y,z,w) coordinates.
//! Agents can only observe (x,y,z) — the w-coordinate is invisible.
//!
//! When an object's w-coordinate drifts away from w=0, it "disappears"
//! from the agents' 3D perspective. Its energy leaves their observable
//! universe. The agents experience this as a conservation violation.
//!
//! This is the computational equivalent of brane cosmology:
//! our 3D universe is a slice of a 4D bulk, and energy can leak out.

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 12;
const DRIFTERS: usize = 4; // objects that drift along W axis
const TICKS: usize = 3000;
const SEEDS: usize = 10;

/// Project a 4D position to 3D by dropping W.
fn project_3d(pos: &SVector<f64, 4>) -> SVector<f64, 3> {
    SVector::from([pos[0], pos[1], pos[2]])
}

/// Distance in 3D (ignoring W axis) — what a 3D agent perceives.
fn distance_3d(a: &SVector<f64, 4>, b: &SVector<f64, 4>) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Whether an object is "visible" to 3D agents (w close to 0).
fn visible_in_3d(pos: &SVector<f64, 4>, w_threshold: f64) -> bool {
    pos[3].abs() < w_threshold
}

struct Result {
    avg_pred_error: f64,
    alive: usize,
    avg_energy: f64,
    objects_visible: f64,      // fraction of drifters visible at end
    total_3d_energy: f64,      // total energy in 3D-visible objects
    total_4d_energy: f64,      // total energy including W-displaced objects
    apparent_violation: f64,   // difference = "dark energy"
    max_pred_error_spike: f64, // peak prediction error during the run
}

fn run(with_drift: bool, seed: u64) -> Result {
    // REAL 4D PHYSICS WORLD
    let mut world = PhysicsWorld::<4>::new(SVector::from([0.0, 0.0, 0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<4>::new();
    consciousness.constants = ThermodynamicConstants {
        initial_energy: 500.0,
        max_energy: 500.0,
        consciousness_maintenance_per_tick: 0.10,
        movement_cost_per_unit: 0.006,
        sprint_cost_multiplier: 2.5,
        collision_energy_drain: 0.05,
        harmony_resonance_regen_rate: 0.12,
        energy_well_regen_rate: 0.25,
        ambient_regen_rate: 0.03,
        collapse_recovery_harmony_threshold: 0.5,
        harmony_range: 40.0,
    };

    let mut rng = seed;
    let mut agent_handles = Vec::new();
    let mut drifter_handles = Vec::new();

    // Spawn agents at w=0 (on the 3D brane)
    for i in 0..AGENTS {
        let x = (nr(&mut rng) - 0.5) * 80.0;
        let y = (nr(&mut rng) - 0.5) * 80.0;
        let h = world.add_sphere(Point::new([x, y, 0.0, 0.0]), 1.0, 1.0);
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.3;
            b.linear_velocity = SVector::from([
                (nr(&mut rng) - 0.5) * 8.0,
                (nr(&mut rng) - 0.5) * 8.0,
                0.0,
                0.0,
            ]);
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

    // Spawn "energy objects" (drifters) that will slide along W axis
    for i in 0..DRIFTERS {
        let x = (nr(&mut rng) - 0.5) * 60.0;
        let y = (nr(&mut rng) - 0.5) * 60.0;
        let h = world.add_sphere(Point::new([x, y, 0.0, 0.0]), 2.0, 5.0); // heavier
        if let Some(b) = world.body_mut(h) {
            b.linear_damping = 0.1;
            if with_drift {
                // Give them a velocity along W axis — they'll drift out of 3D!
                let w_speed = (nr(&mut rng) - 0.3) * 2.0; // mostly positive = drifting away
                b.linear_velocity = SVector::from([0.0, 0.0, 0.0, w_speed]);
            }
        }
        // Drifters carry energy but don't have consciousness
        drifter_handles.push(h);
    }

    let w_visibility = 5.0; // objects with |w| < 5 are "visible" to 3D agents
    let mut max_pe_spike = 0.0f64;

    // Track how much energy was VISIBLE last tick (for violation detection)
    let mut prev_visible_energy = 0.0f64;

    for tick in 0..TICKS {
        // Consciousness update for agents
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

        // FEP gradient — agents sense in 3D only
        // Build "nearby agents" list using 3D projected positions
        let agent_data_3d: Vec<(SVector<f64, 4>, [f64; 9])> = agent_handles
            .iter()
            .filter_map(|&h| {
                let body = world.body(h)?;
                let entity = consciousness.entities.get(&h)?;
                Some((
                    SVector::from(world.body(h).expect("body").position()),
                    entity.harmony_activations,
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
                Some(b) => world.body(h).expect("body").position(),
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

            // 3D-BLIND: agent can only see other agents' xyz, not w
            let nearby: Vec<_> = agent_data_3d
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, (p, h))| (*p, *h)) // passes 4D but gradient only uses first dims
                .collect();

            let dir = fep_gradient::free_energy_gradient(&pos, ef, &harm, &nearby, &[], None, 0.0);
            if let Some(b) = world.body_mut(h) {
                // Agent moves in XY only — it doesn't know W exists
                let mut vel = dir * 25.0;
                vel[2] = 0.0; // no Z movement (2.5D game)
                vel[3] = 0.0; // CANNOT move in W — trapped on the brane
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

        // === THE KEY MECHANIC: detect 3D-visible energy changes ===
        // Calculate total "visible" energy (objects with |w| < threshold)
        let mut current_visible_energy = 0.0f64;
        let mut current_total_energy = 0.0f64;
        for &h in &drifter_handles {
            if let Some(body) = world.body(h) {
                let ke = body.kinetic_energy();
                current_total_energy += ke;
                if visible_in_3d(
                    &SVector::from(world.body(h).expect("body").position()),
                    w_visibility,
                ) {
                    current_visible_energy += ke;
                }
            }
        }
        // Add agent energies (always visible — they're on the brane)
        for &h in &agent_handles {
            if let Some(e) = consciousness.entities.get(&h) {
                current_visible_energy += e.energy.available;
                current_total_energy += e.energy.available;
            }
        }

        // Detect conservation violation: visible energy changed without cause
        if tick > 10 && with_drift {
            let delta = (current_visible_energy - prev_visible_energy).abs();
            if delta > 1.0 {
                // Energy appeared or disappeared from 3D perspective!
                // Inject prediction error to all nearby agents
                let error_per_agent = (delta * 0.01).min(1.0);
                for &h in &agent_handles {
                    if let Some(e) = consciousness.entities.get_mut(&h) {
                        e.prediction_error += error_per_agent;
                        e.motor_precision = 1.0 / (1.0 + e.prediction_error);
                        if e.prediction_error > max_pe_spike {
                            max_pe_spike = e.prediction_error;
                        }
                    }
                }
            }
        }
        prev_visible_energy = current_visible_energy;

        // Offloading (same as before)
        for i in 0..agent_handles.len() {
            for j in (i + 1)..agent_handles.len() {
                let (ha, hb) = (agent_handles[i], agent_handles[j]);
                let dist = match (world.body(ha), world.body(hb)) {
                    (Some(a), Some(b)) => distance_3d(&a.position(), &b.position()),
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
                let res = HarmonyField::<4>::resonance(&hah, &hbh);
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
    let avg_pe = agent_handles
        .iter()
        .filter_map(|h| consciousness.entities.get(h).map(|e| e.prediction_error))
        .sum::<f64>()
        / AGENTS as f64;

    let visible_drifters = drifter_handles
        .iter()
        .filter(|&&h| {
            world
                .body(h)
                .map(|b| visible_in_3d(&world.body(h).expect("body").position(), w_visibility))
                .unwrap_or(false)
        })
        .count();

    let mut total_3d = 0.0f64;
    let mut total_4d = 0.0f64;
    for &h in &drifter_handles {
        if let Some(body) = world.body(h) {
            total_4d += body.kinetic_energy();
            if visible_in_3d(
                &SVector::from(world.body(h).expect("body").position()),
                w_visibility,
            ) {
                total_3d += body.kinetic_energy();
            }
        }
    }

    Result {
        avg_pred_error: avg_pe,
        alive,
        avg_energy: ae,
        objects_visible: visible_drifters as f64 / DRIFTERS as f64,
        total_3d_energy: total_3d,
        total_4d_energy: total_4d,
        apparent_violation: total_4d - total_3d,
        max_pred_error_spike: max_pe_spike,
    }
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════╗");
    println!("║  TRUE 4D: PhysicsWorld<4> with 3D-blind agents                ║");
    println!("║  Objects drift along W axis. Agents can't see W.              ║");
    println!("║  Energy that moves in W vanishes from agents' reality.         ║");
    println!("╚═════════════════════════════════════════════════════════════════╝\n");
    println!(
        "{} agents + {} drifters in PhysicsWorld<4>, {} ticks, {} seeds\n",
        AGENTS, DRIFTERS, TICKS, SEEDS
    );

    println!(
        "{:<10} {:<8} {:<8} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Cond", "PredErr", "MaxPE", "Alive", "Energy", "Visible", "3D_E", "4D_E", "DarkE"
    );
    println!("{}", "─".repeat(84));

    for (name, drift) in &[("STATIC", false), ("DRIFTING", true)] {
        let mut pe = Vec::new();
        let mut mp = Vec::new();
        let mut al = Vec::new();
        let mut en = Vec::new();
        let mut vi = Vec::new();
        let mut e3 = Vec::new();
        let mut e4 = Vec::new();
        let mut de = Vec::new();

        for s in 0..SEEDS {
            let r = run(*drift, 42 + s as u64 * 997);
            pe.push(r.avg_pred_error);
            mp.push(r.max_pred_error_spike);
            al.push(r.alive as f64);
            en.push(r.avg_energy);
            vi.push(r.objects_visible);
            e3.push(r.total_3d_energy);
            e4.push(r.total_4d_energy);
            de.push(r.apparent_violation);
        }

        let n = SEEDS as f64;
        println!(
            "{:<10} {:<8.4} {:<8.3} {:<8.1} {:<10.1} {:<10.1}% {:<10.2} {:<10.2} {:<10.2}",
            name,
            pe.iter().sum::<f64>() / n,
            mp.iter().sum::<f64>() / n,
            al.iter().sum::<f64>() / n,
            en.iter().sum::<f64>() / n,
            vi.iter().sum::<f64>() / n * 100.0,
            e3.iter().sum::<f64>() / n,
            e4.iter().sum::<f64>() / n,
            de.iter().sum::<f64>() / n,
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("STATIC:   Drifters stay at w=0 (normal 3D physics)");
    println!("DRIFTING: Drifters slide along W axis (leave 3D brane)");
    println!("Visible:  Fraction of drifters with |w| < {} at end", 5.0);
    println!("DarkE:    Energy in 4D bulk invisible to 3D agents");
    println!("MaxPE:    Peak prediction error spike during run");
    println!("═══════════════════════════════════════════════════════════════");
}

fn nr(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
