// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmark: Conjunction Screening Pipeline
//!
//! Measures throughput of the full screening pipeline:
//! 1. TLE parsing
//! 2. SGP4 propagation
//! 3. Alfano 2D collision probability
//! 4. Full screening (N objects over T hours)
//!
//! Run: cargo bench -p orbital-mechanics

use std::time::Instant;

use chrono::Duration;
use nalgebra::Vector3;
use orbital_mechanics::conjunction::ConjunctionAnalyzer;
use orbital_mechanics::covariance::CovarianceMatrix;
use orbital_mechanics::propagator::Propagator;
use orbital_mechanics::state::{OrbitalState, StateVector};
use orbital_mechanics::tle::TwoLineElement;

/// Real ISS TLE (valid checksum, known to parse and propagate)
const ISS_TLE: &str = "ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577";

fn bench_tle_parsing(iterations: u32) -> (f64, u32) {
    let start = Instant::now();
    let mut count = 0u32;

    for _ in 0..iterations {
        let _ = TwoLineElement::parse(ISS_TLE);
        count += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, count)
}

fn bench_sgp4_propagation(iterations: u32) -> (f64, u32) {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();
    let epoch = tle.epoch;

    let start = Instant::now();
    let mut count = 0u32;

    for i in 0..iterations {
        let time = epoch + Duration::minutes(i as i64);
        let _ = prop.propagate_to(time);
        count += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, count)
}

/// Generate synthetic orbital states at different altitudes/inclinations
/// for Alfano Pc benchmarking (ISS TLE only gives us one orbit).
fn make_synthetic_states(n: usize, epoch: chrono::DateTime<chrono::Utc>) -> Vec<OrbitalState> {
    let earth_radius: f64 = 6371.0;
    let altitudes: [f64; 8] = [420.0, 450.0, 500.0, 550.0, 600.0, 700.0, 800.0, 1000.0];
    let inclinations_rad: [f64; 8] = [
        0.9012, 0.9300, 0.8500, 1.4400, 1.7100, 0.5000, 1.2000, 0.7500,
    ];

    (0..n)
        .map(|i| {
            let alt: f64 = altitudes[i % altitudes.len()];
            let inc: f64 = inclinations_rad[i % inclinations_rad.len()];
            let r: f64 = earth_radius + alt;
            let v: f64 = (398600.4418_f64 / r).sqrt(); // circular orbit velocity

            // Spread objects around the orbit by phase
            let phase: f64 = (i as f64) * std::f64::consts::PI * 2.0 / (n as f64);

            let pos = Vector3::new(
                r * phase.cos() * inc.cos(),
                r * phase.sin(),
                r * phase.cos() * inc.sin(),
            );
            let vel = Vector3::new(
                -v * phase.sin() * inc.cos(),
                v * phase.cos(),
                -v * phase.sin() * inc.sin(),
            );

            let sv = StateVector::from_vectors(pos, vel);
            let cov = CovarianceMatrix::from_tle_age(24.0); // 1 day old

            OrbitalState::new(
                25544 + i as u32,
                epoch,
                sv,
                orbital_mechanics::state::DataSource::SpaceTrack,
            )
            .with_covariance(cov)
        })
        .collect()
}

fn bench_alfano_pc(iterations: u32, n_objects: usize) -> (f64, u32) {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let states = make_synthetic_states(n_objects, tle.epoch);
    let analyzer = ConjunctionAnalyzer::new().with_hbr(20.0);

    let start = Instant::now();
    let mut count = 0u32;

    for _ in 0..iterations {
        for j in 0..states.len() {
            for k in (j + 1)..states.len() {
                let _ = analyzer.assess(&states[j], &states[k]);
                count += 1;
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, count)
}

fn bench_full_screening(n_objects: usize, hours: u64, step_minutes: u64) -> (f64, u32, u32) {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();
    let prop = Propagator::from_tle(&tle).unwrap();
    let epoch = tle.epoch;

    let analyzer = ConjunctionAnalyzer::new()
        .with_hbr(20.0)
        .with_screening_threshold(50.0);

    let steps = (hours * 60) / step_minutes;
    let start = Instant::now();
    let mut total_assessments = 0u32;
    let mut conjunctions_found = 0u32;

    for step in 0..steps {
        let time = epoch + Duration::minutes((step * step_minutes) as i64);

        // Propagate the ISS and generate synthetic positions for others
        let iss_state = match prop.propagate_to(time) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let mut states = make_synthetic_states(n_objects - 1, time);
        states.insert(0, iss_state);

        for j in 0..states.len() {
            for k in (j + 1)..states.len() {
                let assessment = analyzer.assess(&states[j], &states[k]);
                total_assessments += 1;
                if assessment.collision_probability.pc > 1e-10 {
                    conjunctions_found += 1;
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, total_assessments, conjunctions_found)
}

fn main() {
    println!("=================================================================");
    println!("  Mycelix Space — Conjunction Screening Pipeline Benchmark");
    println!("=================================================================\n");

    // 1. TLE Parsing
    let (elapsed, count) = bench_tle_parsing(100_000);
    println!("1. TLE Parsing");
    println!(
        "   {:>8} parses in {:.3}s = {:>10.0} TLEs/sec\n",
        count,
        elapsed,
        count as f64 / elapsed
    );

    // 2. SGP4 Propagation
    let (elapsed, count) = bench_sgp4_propagation(100_000);
    println!("2. SGP4 Propagation");
    println!(
        "   {:>8} propagations in {:.3}s = {:>10.0} props/sec\n",
        count,
        elapsed,
        count as f64 / elapsed
    );

    // 3. Alfano 2D Pc Computation
    let n_objects = 8;
    let n_pairs = n_objects * (n_objects - 1) / 2;
    let (elapsed, count) = bench_alfano_pc(10_000, n_objects);
    println!(
        "3. Alfano 2D Collision Probability ({} objects, {} pairs)",
        n_objects, n_pairs
    );
    println!(
        "   {:>8} assessments in {:.3}s = {:>10.0} Pc/sec\n",
        count,
        elapsed,
        count as f64 / elapsed
    );

    // 4. Full Screening Pipeline
    println!("4. Full Screening Pipeline ({} objects)\n", n_objects);

    for (hours, step) in [(24, 5), (72, 5), (168, 10)] {
        let (elapsed, assessments, conjunctions) = bench_full_screening(n_objects, hours, step);
        let steps = (hours * 60) / step;
        println!("   {}h window, {}min steps ({} steps):", hours, step, steps);
        println!(
            "     {:.3}s total | {} assessments | {} conjunctions (Pc > 1e-10)",
            elapsed, assessments, conjunctions
        );
        println!(
            "     {:>10.0} assessments/sec | {:.2}ms per screening cycle\n",
            assessments as f64 / elapsed,
            (elapsed / steps as f64) * 1000.0
        );
    }

    // 5. Scalability: more objects
    println!("5. Scalability (72h window, 5min steps)\n");
    for n in [10, 25, 50, 100] {
        let n_p = n * (n - 1) / 2;
        let (elapsed, assessments, _) = bench_full_screening(n, 72, 5);
        println!(
            "   {:>4} objects ({:>5} pairs): {:.3}s | {:>10.0} assessments/sec",
            n,
            n_p,
            elapsed,
            assessments as f64 / elapsed
        );
    }

    println!("\n=================================================================");
    println!("  Pure Rust | SGP4 + Alfano 2D analytical Pc | No external deps");
    println!("=================================================================");
}
