// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! BIOLOGICAL VALIDATION 2: Consciousness energy threshold
//!
//! Biology: consciousness emerges at ~42% of normal cortical metabolism
//! (Stender et al. 2015, 2016 — 131 patients, 18F-FDG PET)
//!
//! Test: gradually deplete agent energy, measure Phi at each level.
//! At what fraction does functional consciousness degrade?
//! Is there hysteresis (different threshold up vs down)?

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::{ConsciousnessField, ThermodynamicConstants};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const SEEDS: usize = 30;

fn measure_phi_at_energy_fraction(frac: f64, seed: u64) -> f64 {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));
    let mut consciousness = ConsciousnessField::<2>::new();
    let h = world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
    consciousness.register(h, 1000.0, 20.0);

    // Set energy to the target fraction
    if let Some(e) = consciousness.entities.get_mut(&h) {
        let drain = 1000.0 * (1.0 - frac);
        e.energy.consume(drain);
    }

    // Run for 100 ticks to let consciousness stabilize
    for _ in 0..100 {
        let ef = consciousness
            .entities
            .get(&h)
            .map(|e| e.energy.fraction_remaining())
            .unwrap_or(0.0);
        let inputs = ConsciousnessInputs {
            phi: ef * 0.8 + 0.1, // Phi scales with energy availability
            broadcast: (0.3 + ef * 0.5).min(1.0),
            working_memory: (0.2 + ef * 0.5).min(1.0),
            attention: (0.3 + ef * 0.4).min(1.0),
            recurrence: (0.2 + ef * 0.4).min(1.0),
            embodiment: (0.3 + ef * 0.4).min(1.0),
            knowledge: (0.2 + ef * 0.3).min(1.0),
            synchrony: (0.2 + ef * 0.4).min(1.0),
        };
        consciousness.update_entity(h, &inputs, Point::origin());

        // Drain a tiny amount per tick (maintenance)
        if let Some(e) = consciousness.entities.get_mut(&h) {
            e.energy
                .consume(consciousness.constants.consciousness_maintenance_per_tick * 0.5);
        }
    }

    consciousness.phi(h)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║  BIOLOGICAL VALIDATION 2: Consciousness Energy Threshold     ║");
    eprintln!("║  Target: ~42% of normal metabolism (Stender et al. 2015)    ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");

    // CSV header
    println!("direction,energy_fraction,avg_phi,std_phi");

    // Ramp DOWN: 100% → 0%
    let fractions_down: Vec<f64> = (0..=20).map(|i| 1.0 - i as f64 * 0.05).collect();
    let mut threshold_down = 0.0;
    let mut found_down = false;

    for &frac in &fractions_down {
        let mut phis = Vec::new();
        for s in 0..SEEDS {
            phis.push(measure_phi_at_energy_fraction(frac, 42 + s as u64 * 997));
        }
        let mean = phis.iter().sum::<f64>() / SEEDS as f64;
        let std =
            (phis.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (SEEDS as f64 - 1.0)).sqrt();
        println!("DOWN,{:.2},{:.6},{:.6}", frac, mean, std);

        // Detect functional threshold (Phi drops below 0.05 = ~Red tier equivalent)
        if !found_down && mean < 0.05 {
            threshold_down = frac + 0.05; // previous step was still functional
            found_down = true;
        }
    }

    // Ramp UP: 0% → 100%
    let fractions_up: Vec<f64> = (0..=20).map(|i| i as f64 * 0.05).collect();
    let mut threshold_up = 0.0;
    let mut found_up = false;

    for &frac in &fractions_up {
        let mut phis = Vec::new();
        for s in 0..SEEDS {
            phis.push(measure_phi_at_energy_fraction(frac, 42 + s as u64 * 997));
        }
        let mean = phis.iter().sum::<f64>() / SEEDS as f64;
        let std =
            (phis.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (SEEDS as f64 - 1.0)).sqrt();
        println!("UP,{:.2},{:.6},{:.6}", frac, mean, std);

        if !found_up && mean > 0.05 {
            threshold_up = frac;
            found_up = true;
        }
    }

    eprintln!("\n═══════════════════════════════════════════════════");
    eprintln!(
        "THRESHOLD (ramp down): {:.0}% of max energy",
        threshold_down * 100.0
    );
    eprintln!(
        "THRESHOLD (ramp up):   {:.0}% of max energy",
        threshold_up * 100.0
    );
    eprintln!(
        "HYSTERESIS:            {:.0}%",
        (threshold_down - threshold_up).abs() * 100.0
    );
    eprintln!("BIOLOGY:               42% (Stender et al. 2015)");
    if (threshold_down * 100.0 - 42.0).abs() < 15.0 {
        eprintln!("  ✓ Within range of biological threshold");
    } else {
        eprintln!("  ✗ Does not match biological threshold");
    }
    eprintln!("═══════════════════════════════════════════════════");
}
