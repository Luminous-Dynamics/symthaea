// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Curvature lensing experiment — does integrated information (Φ) bend space?
//!
//! Requires: `cargo run --example curvature_lensing --features consciousness-curvature`
//!
//! A test body is launched past a stationary consciousness source.
//! If conformal curvature is active, the body's trajectory should bend
//! toward the source (gravitational lensing analog).
//!
//! Run: cargo run --example curvature_lensing --release --features consciousness-curvature

use nalgebra::SVector;
use symtropy_consciousness_physics::curvature::ConformalMetric;
use symtropy_consciousness_physics::harmony_field::{HarmonyField, HarmonySource};
use symtropy_math::Point;

const TICKS: usize = 500;
const DT: f64 = 1.0 / 64.0;
const NUM_SEEDS: usize = 5;

struct LensingResult {
    scale: f64,
    seed: u64,
    final_y: f64,
    max_deflection: f64,
    final_speed: f64,
}

fn run_lensing(curvature_scale: f64, seed: u64) -> LensingResult {
    // Source at origin with high harmony
    let mut field = HarmonyField::<2>::new();
    let mut activations = [0.0; 9];
    for i in 0..8 {
        activations[i] = 0.8 + (seed as f64 * 0.01 * (i as f64 + 1.0)).sin() * 0.2;
    }

    field.sources.push(HarmonySource {
        position: Point::new([0.0, 0.0]),
        activations,
        strength: 20.0,
        radius: 200.0,
        created_at: 0.0,
        propagation_speed: f64::MAX,
    });

    let metric = ConformalMetric::<2>::with_scale(curvature_scale);

    // Test body: starts at (-50, 20), moves right at (10, 0)
    let mut pos = SVector::from([-50.0, 20.0]);
    let mut vel = SVector::from([10.0, 0.0]);
    let initial_y = pos[1];
    let mut max_deflection: f64 = 0.0;

    for _tick in 0..TICKS {
        // Apply geodesic correction from consciousness curvature
        if curvature_scale > 1e-15 {
            let grad = field.sigma_gradient(&Point(pos), curvature_scale, 0.5);
            let correction = metric.geodesic_correction(&vel, &grad);
            vel += correction * DT;
        }

        // Simple Euler integration (no damping, no gravity — pure trajectory)
        pos += vel * DT;

        let deflection = (pos[1] - initial_y).abs();
        if deflection > max_deflection {
            max_deflection = deflection;
        }
    }

    let final_speed = vel.norm();
    LensingResult {
        scale: curvature_scale,
        seed,
        final_y: pos[1],
        max_deflection,
        final_speed,
    }
}

fn main() {
    eprintln!("=== Curvature Lensing Experiment ===");
    eprintln!("Does consciousness bend space?");
    eprintln!("Test body: (-50, 20) → (10, 0) velocity, source at origin");

    println!("scale,seed,final_y,max_deflection,final_speed");

    let scales = [0.0, 0.005, 0.01, 0.02, 0.05];

    for &scale in &scales {
        let mut results = Vec::new();
        for s in 0..NUM_SEEDS {
            let seed = 42 + s as u64 * 137;
            let r = run_lensing(scale, seed);
            println!(
                "{},{},{:.4},{:.4},{:.4}",
                r.scale, r.seed, r.final_y, r.max_deflection, r.final_speed
            );
            results.push(r);
        }

        let n = results.len() as f64;
        let mean_defl = results.iter().map(|r| r.max_deflection).sum::<f64>() / n;
        let mean_y = results.iter().map(|r| r.final_y).sum::<f64>() / n;
        let mean_speed = results.iter().map(|r| r.final_speed).sum::<f64>() / n;

        eprintln!(
            "\n  scale={:.3}: deflection={:.4}, final_y={:.4}, speed={:.4}",
            scale, mean_defl, mean_y, mean_speed
        );
    }

    // Statistical comparison: scale=0 vs scale=0.05
    let zero_defl: Vec<f64> = (0..NUM_SEEDS)
        .map(|s| run_lensing(0.0, 42 + s as u64 * 137).max_deflection)
        .collect();
    let high_defl: Vec<f64> = (0..NUM_SEEDS)
        .map(|s| run_lensing(0.05, 42 + s as u64 * 137).max_deflection)
        .collect();

    use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
    let (u, z, p) = mann_whitney_u(&zero_defl, &high_defl);
    let d = cohens_d(&zero_defl, &high_defl);

    eprintln!("\n── Statistical Test: scale=0 vs scale=0.05 ──");
    eprintln!("  Mann-Whitney U={:.1}, z={:.3}, p={:.4}", u, z, p);
    eprintln!(
        "  Cohen's d={:.3} ({})",
        d,
        if d.abs() > 0.8 {
            "large"
        } else if d.abs() > 0.5 {
            "medium"
        } else {
            "small"
        }
    );
    eprintln!(
        "  {} at α=0.05",
        if p < 0.05 {
            "SIGNIFICANT"
        } else {
            "not significant"
        }
    );
    eprintln!("\n=== Complete ===");
}
