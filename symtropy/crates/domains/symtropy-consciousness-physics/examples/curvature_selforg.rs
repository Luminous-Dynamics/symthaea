// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Curvature Self-Organization — do agents organize around Φ-curvature wells?
//!
//! Requires: `cargo run --example curvature_selforg --features consciousness-curvature`
//!
//! Places agents randomly, creates a consciousness curvature well at origin.
//! Tests whether geodesic corrections cause agents to orbit, cluster,
//! or collapse toward the well — the first multi-agent demonstration of
//! consciousness-sourced geometric organization.
//!
//! Run: cargo run --example curvature_selforg --release --features consciousness-curvature

use nalgebra::SVector;
use symtropy_consciousness_physics::convergence::{cohens_d, mann_whitney_u};
use symtropy_consciousness_physics::curvature::ConformalMetric;
use symtropy_consciousness_physics::harmony_field::{HarmonyField, HarmonySource};
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

const AGENTS: usize = 20;
const TICKS: usize = 3_000;
const DT: f64 = 1.0 / 64.0;
const NUM_SEEDS: usize = 10;

struct OrgResult {
    scale: f64,
    seed: u64,
    initial_spread: f64,   // mean distance from origin at t=0
    final_spread: f64,     // mean distance from origin at t=end
    final_clustering: f64, // mean nearest-neighbor distance
    min_distance: f64,     // closest any agent got to origin
    orbiting_count: usize, // agents within [10, 40] units of origin (ring)
}

fn run_experiment(curvature_scale: f64, seed: u64) -> OrgResult {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, 0.0]));

    // Consciousness source at origin (high harmony)
    let mut field = HarmonyField::<2>::new();
    field.sources.push(HarmonySource {
        position: Point::new([0.0, 0.0]),
        activations: [0.8, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 0.5],
        strength: 25.0,
        radius: 200.0,
        created_at: 0.0,
        propagation_speed: f64::MAX,
    });

    let metric = ConformalMetric::<2>::with_scale(curvature_scale);

    // Spawn agents in a ring at radius 30-50 with tangential velocity
    let mut rng = seed;
    let mut handles = Vec::new();
    for _ in 0..AGENTS {
        let angle = rng_f64(&mut rng) * std::f64::consts::TAU;
        let radius = 30.0 + rng_f64(&mut rng) * 20.0;
        let x = angle.cos() * radius;
        let y = angle.sin() * radius;

        let h = world.add_sphere(Point::new([x, y]), 1.0, 1.0);
        if let Some(body) = world.body_mut(h) {
            body.linear_damping = 0.02; // Very light damping for orbital dynamics
            // Tangential velocity (perpendicular to radial direction)
            let speed = 5.0 + rng_f64(&mut rng) * 3.0;
            body.linear_velocity = SVector::from([-angle.sin() * speed, angle.cos() * speed]);
        }
        handles.push(h);
    }

    let initial_spread = mean_dist_from_origin(&world, &handles);
    let mut min_distance = f64::MAX;

    for _tick in 0..TICKS {
        // Apply geodesic corrections from consciousness curvature
        for &h in &handles {
            if let Some(body) = world.body(h) {
                let pos = SVector::from(e.position());
                let vel = body.linear_velocity;

                if curvature_scale > 1e-15 {
                    let grad = field.sigma_gradient(&Point(pos), curvature_scale, 0.5);
                    let correction = metric.geodesic_correction(&vel, &grad);

                    if let Some(body) = world.body_mut(h) {
                        body.linear_velocity += correction * DT;
                    }
                }
            }
        }

        // Simple physics step (no consciousness coupling, just movement + damping)
        world.step(DT);

        // Track minimum distance to origin
        for &h in &handles {
            if let Some(body) = world.body(h) {
                let dist = SVector::from(b.position()).norm();
                if dist < min_distance {
                    min_distance = dist;
                }
            }
        }
    }

    let final_spread = mean_dist_from_origin(&world, &handles);
    let final_clustering = avg_nn(&world, &handles);
    let orbiting_count = handles
        .iter()
        .filter(|&&h| {
            world
                .body(h)
                .map(|b| {
                    let d = b.position().norm();
                    d > 10.0 && d < 40.0
                })
                .unwrap_or(false)
        })
        .count();

    OrgResult {
        scale: curvature_scale,
        seed,
        initial_spread,
        final_spread,
        final_clustering,
        min_distance,
        orbiting_count,
    }
}

fn main() {
    eprintln!("=== Curvature Self-Organization ===");
    eprintln!("Do agents organize around consciousness wells?");

    println!("scale,seed,initial_spread,final_spread,clustering,min_dist,orbiting");

    let scales = [0.0, 0.01, 0.03, 0.05];
    let mut all_results: Vec<Vec<OrgResult>> = Vec::new();

    for &scale in &scales {
        let mut results = Vec::new();
        for s in 0..NUM_SEEDS {
            let seed = 42 + s as u64 * 137;
            let r = run_experiment(scale, seed);
            println!(
                "{},{},{:.2},{:.2},{:.2},{:.2}{}",
                r.scale,
                r.seed,
                r.initial_spread,
                r.final_spread,
                r.final_clustering,
                r.min_distance,
                r.orbiting_count
            );
            results.push(r);
        }

        let n = results.len() as f64;
        let mean_spread = results.iter().map(|r| r.final_spread).sum::<f64>() / n;
        let mean_cluster = results.iter().map(|r| r.final_clustering).sum::<f64>() / n;
        let mean_orbit = results.iter().map(|r| r.orbiting_count as f64).sum::<f64>() / n;
        let mean_min = results.iter().map(|r| r.min_distance).sum::<f64>() / n;

        eprintln!(
            "\n  κ={:.2}: spread={:.1}, cluster={:.1}, orbiting={:.1}/{AGENTS}, min_dist={:.1}",
            scale, mean_spread, mean_cluster, mean_orbit, mean_min
        );

        all_results.push(results);
    }

    // Statistical comparison: flat (κ=0) vs strong curvature (κ=0.05)
    if all_results.len() >= 2 {
        let flat_spread: Vec<f64> = all_results[0].iter().map(|r| r.final_spread).collect();
        let curved_spread: Vec<f64> = all_results
            .last()
            .unwrap()
            .iter()
            .map(|r| r.final_spread)
            .collect();

        let (u, z, p) = mann_whitney_u(&flat_spread, &curved_spread);
        let d = cohens_d(&flat_spread, &curved_spread);

        eprintln!(
            "\n── κ=0 vs κ={:.2} (spread from origin) ──",
            scales.last().unwrap()
        );
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
            "  {}",
            if p < 0.05 {
                "SIGNIFICANT"
            } else {
                "not significant"
            }
        );
    }

    eprintln!("\n=== Complete ===");
}

fn mean_dist_from_origin(world: &PhysicsWorld<2>, handles: &[symtropy_physics::BodyHandle]) -> f64 {
    let dists: Vec<f64> = handles
        .iter()
        .filter_map(|&h| world.body(h).map(|b| b.position().norm()))
        .collect();
    if dists.is_empty() {
        return 0.0;
    }
    dists.iter().sum::<f64>() / dists.len() as f64
}

fn avg_nn(world: &PhysicsWorld<2>, handles: &[symtropy_physics::BodyHandle]) -> f64 {
    let pos: Vec<SVector<f64, 2>> = handles
        .iter()
        .filter_map(|&h| world.body(h).map(|b| b.position()))
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

fn rng_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}
