// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Engine scorecard: single run validation of all systems.
//!
//! Run: cargo run --example scorecard --release

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::convergence::ConvergenceDetector;
use symtropy_consciousness_physics::fep_gradient;
use symtropy_consciousness_physics::harmony_field::HarmonyField;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 5_000;
const DT: f64 = 1.0 / 64.0;

fn main() {
    eprintln!("=== Symtropy Engine Scorecard ===");

    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    consciousness.constants = ThermodynamicConstants::research();

    let wells = vec![SVector::from([25.0, 25.0]), SVector::from([-25.0, -25.0])];
    let mut rng = 42u64;
    let mut handles = Vec::new();

    for i in 0..AGENTS {
        let x = (rng_f64(&mut rng) - 0.5) * 80.0;
        let y = (rng_f64(&mut rng) - 0.5) * 80.0;
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
                0 => e.harmony_activations = [0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.5],
                1 => e.harmony_activations = [0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.2, 0.5],
                2 => e.harmony_activations = [0.5; 9],
                _ => e.harmony_activations = [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.3, 0.5],
            }
        }
        handles.push(h);
    }

    let mut jphi_detector = ConvergenceDetector::new(200, 1e-3);
    let mut coop_events = 0u64;
    let mut peak_phi = 0.0f64;
    let mut peak_tick = 0;

    println!("tick,phi,alive,avg_energy,clustering,coop_events");

    for tick in 0..TICKS {
        // Consciousness update with dynamic inputs
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
            let old = consciousness.phi(h);
            consciousness.update_entity(h, &inputs, Point::origin());
            consciousness
                .ledger
                .record_phi_change(consciousness.phi(h) - old);
        }

        // FEP gradient
        let adata: Vec<_> = handles
            .iter()
            .filter_map(|&h| {
                Some((
                    world.body(h)?.position(),
                    consciousness.entities.get(&h)?.harmony_activations,
                ))
            })
            .collect();
        let wdata: Vec<_> = wells.iter().map(|&p| (p, 1.0)).collect();
        for &h in &handles {
            let Some(b) = world.body(h) else { continue };
            let Some(e) = consciousness.entities.get(&h) else {
                continue;
            };
            if e.energy.is_collapsed() {
                continue;
            }
            let pos = b.position();
            let near: Vec<_> = adata
                .iter()
                .filter(|(p, _)| (p - pos).norm() > 2.0)
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

        // Maintenance + regen
        let mr = consciousness.constants.consciousness_maintenance_per_tick;
        let ar = consciousness.constants.ambient_regen_rate;
        let wr = consciousness.constants.energy_well_regen_rate;
        let rm = consciousness.resource_regeneration_multiplier();
        let nw: Vec<bool> = handles
            .iter()
            .map(|&h| {
                world
                    .body(h)
                    .map(|b| wells.iter().any(|&w| (b.position() - w).norm() < 35.0))
                    .unwrap_or(false)
            })
            .collect();
        for (i, &h) in handles.iter().enumerate() {
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.tick_reset();
            }
            let phi = consciousness.phi(h);
            consciousness.consume_energy(h, mr * (1.0 + phi * 0.5));
            if let Some(e) = consciousness.entities.get_mut(&h) {
                e.energy.regenerate(ar * rm);
                if nw[i] {
                    e.energy.regenerate(wr);
                }
            }
        }

        // Harmony offloading
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let (ha, hb) = (handles[i], handles[j]);
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
                    coop_events += 1;
                }
            }
        }

        consciousness.tick_prediction_errors();
        world.step_with_callback(DT, &mut consciousness);
        let bal = consciousness.tick_thermodynamics();
        if let Some(j) = bal.joules_per_phi {
            if j.is_finite() {
                jphi_detector.push(j);
            }
        }
        if consciousness.collective_phi > peak_phi {
            peak_phi = consciousness.collective_phi;
            peak_tick = tick;
        }

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
            let ae: f64 = handles
                .iter()
                .filter_map(|h| consciousness.entities.get(h).map(|e| e.energy.available))
                .sum::<f64>()
                / AGENTS as f64;
            let cl = avg_nn(&world, &handles);
            println!(
                "{tick},{:.4},{alive},{ae:.1},{cl:.2},{coop_events}",
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
    eprintln!("\n╔══════════════════════════════════════════╗");
    eprintln!("║         ENGINE SCORECARD                 ║");
    eprintln!("╠══════════════════════════════════════════╣");
    eprintln!(
        "║ Alive:      {:>3}/{:<3}                      ║",
        alive, AGENTS
    );
    eprintln!(
        "║ Peak Φ:     {:.4} (tick {:<5})            ║",
        peak_phi, peak_tick
    );
    eprintln!(
        "║ Clustering: {:.2}                          ║",
        avg_nn(&world, &handles)
    );
    eprintln!("║ Coop events:{:<10}                   ║", coop_events);
    eprintln!(
        "║ J/Φ conv:   {}                    ║",
        if jphi_detector.is_converged() {
            "YES"
        } else {
            "NO "
        }
    );
    eprintln!("╚══════════════════════════════════════════╝");
    eprintln!(
        "  Survival: {}",
        if alive > AGENTS / 2 { "PASS" } else { "FAIL" }
    );
    eprintln!(
        "  Cooperation: {}",
        if coop_events > 0 { "PASS" } else { "FAIL" }
    );
}

fn avg_nn(w: &PhysicsWorld<2>, h: &[symtropy_physics::BodyHandle]) -> f64 {
    let p: Vec<_> = h
        .iter()
        .filter_map(|h| w.body(*h).map(|b| b.position()))
        .collect();
    if p.len() < 2 {
        return f64::MAX;
    }
    p.iter()
        .enumerate()
        .map(|(i, a)| {
            p.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, b)| (a - b).norm())
                .fold(f64::MAX, f64::min)
        })
        .sum::<f64>()
        / p.len() as f64
}
fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
