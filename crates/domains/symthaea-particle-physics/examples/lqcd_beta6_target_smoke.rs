// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Manual β=6.0 target-volume throughput smoke subject.
//!
//! This executable is deliberately not a scientific ensemble run. It exists to
//! measure target-volume transition/operator costs before burn-in, stride,
//! plateau windows, retained sample count, or seeds are frozen for production.
//! None of its gauge fields or measurements are authorized for final physics.

use std::time::Instant;

use symthaea_particle_physics::{
    HeatbathOverrelaxationSchedule, LatticeChaCha8Stream, LatticeStreamCoordinates,
    LatticeStreamDomain, SpatialApeConfig, SubgroupForceBackend, WilsonGaugeField,
    average_cubic_bresenham_mixed_wilson_loop, heatbath_overrelaxation_cycle,
    spatial_ape_smear, su3_diagonal,
};

const DIMS: [usize; 4] = [16, 16, 16, 32];
const BETA: f64 = 6.0;

fn schedule() -> HeatbathOverrelaxationSchedule {
    HeatbathOverrelaxationSchedule {
        force_backend: SubgroupForceBackend::Staple,
        overrelaxation_sweeps: 3,
        max_heatbath_attempts: 256,
    }
}

fn stream() -> LatticeChaCha8Stream {
    LatticeChaCha8Stream::new(
        [0x42; 32],
        LatticeStreamCoordinates {
            domain: LatticeStreamDomain::Qualification,
            ensemble_slot: 0x021b,
            replica: 0,
            rank: 0,
        },
    )
    .expect("qualification stream must be valid")
}

/// Deterministic non-equilibrium SU(3) field used only to benchmark operator cost.
fn synthetic_operator_fixture() -> WilsonGaugeField {
    let mut field = WilsonGaugeField::identity(DIMS).expect("target dimensions must be valid");
    for x in 0..DIMS[0] {
        for y in 0..DIMS[1] {
            for z in 0..DIMS[2] {
                for t in 0..DIMS[3] {
                    for mu in 0..4 {
                        let phase = 0.002
                            * (1 + x + 3 * y + 5 * z + 7 * t + 11 * mu) as f64;
                        let secondary = -0.0015
                            * (1 + 2 * x + y + 4 * z + 3 * t + 5 * mu) as f64;
                        field
                            .set_link([x, y, z, t], mu, su3_diagonal(phase, secondary))
                            .expect("diagonal fixture link must be SU(3)");
                    }
                }
            }
        }
    }
    field
}

fn ape_config() -> SpatialApeConfig {
    SpatialApeConfig {
        alpha: 0.7,
        iterations: 19,
        projection_tolerance: 2.0e-15,
        projection_max_iterations: 40,
    }
}

fn vector_program() -> Vec<[i32; 3]> {
    let mut vectors = Vec::with_capacity(24);
    for n in 1..=7 {
        vectors.push([n, 0, 0]);
    }
    for n in 1..=7 {
        vectors.push([n, n, 0]);
    }
    for n in 1..=7 {
        vectors.push([n, n, n]);
    }
    for n in 1..=3 {
        vectors.push([2 * n, n, 0]);
    }
    vectors
}

fn transition_smoke() {
    let mut field = WilsonGaugeField::identity(DIMS).expect("target dimensions must be valid");
    let mut rng = stream();
    let started = Instant::now();
    let stats = heatbath_overrelaxation_cycle(&mut field, BETA, schedule(), &mut rng)
        .expect("target-volume HB+3OR cycle failed");
    let elapsed = started.elapsed().as_secs_f64();
    let plaquette = field
        .average_plaquette()
        .expect("post-cycle plaquette measurement failed");
    println!(
        "phase=transition scope=development_only seconds={elapsed:.9} sampler={} stream_id={} heatbath_updates={} scalar_attempts={} or_updates={} fallbacks={} max_or_drift={:.17e} post_plaquette={plaquette:.17e}",
        schedule().identity(),
        rng.stream_id(),
        stats.heatbath.subgroup_updates,
        stats.heatbath.scalar_rejection_attempts,
        stats.overrelaxation_subgroup_updates,
        stats.heatbath.reference_fallback_updates + stats.overrelaxation_reference_fallback_updates,
        stats.max_abs_overrelaxation_trace_drift,
    );
}

fn smear_fixture() -> (WilsonGaugeField, WilsonGaugeField, f64) {
    let original = synthetic_operator_fixture();
    let started = Instant::now();
    let operator = spatial_ape_smear(&original, &ape_config())
        .expect("target-volume APE19 construction failed");
    let elapsed = started.elapsed().as_secs_f64();
    (original, operator, elapsed)
}

fn ape_smoke() {
    let (original, operator, elapsed) = smear_fixture();
    let before = original
        .average_plaquette()
        .expect("fixture plaquette measurement failed");
    let after = operator
        .average_plaquette()
        .expect("operator plaquette measurement failed");
    println!(
        "phase=ape scope=development_only seconds={elapsed:.9} alpha=0.7 iterations=19 epsilon_n_proxy={:.17e} input_plaquette={before:.17e} operator_plaquette={after:.17e}",
        19.0 / 4.7,
    );
}

fn wilson_sample_smoke() {
    let (original, operator, ape_seconds) = smear_fixture();
    let probes = [
        ([1, 0, 0], 1usize),
        ([2, 1, 0], 4usize),
        ([6, 6, 6], 4usize),
        ([7, 7, 7], 8usize),
    ];
    let started = Instant::now();
    let mut checksum = 0.0;
    for (vector, temporal_extent) in probes {
        let measured = average_cubic_bresenham_mixed_wilson_loop(
            &original,
            &operator,
            vector,
            temporal_extent,
            24,
            21,
        )
        .expect("sample bounded Wilson measurement failed");
        checksum += measured.value;
        println!(
            "phase=wilson_sample_item scope=development_only r={:?} t={} value={:.17e} orbit={} steps_per_orientation={}",
            vector,
            temporal_extent,
            measured.value,
            measured.orbit_size,
            measured.steps_per_orientation,
        );
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "phase=wilson_sample scope=development_only ape_seconds={ape_seconds:.9} measurement_seconds={elapsed:.9} observable_count={} checksum={checksum:.17e}",
        probes.len(),
    );
}

fn wilson_full_smoke() {
    let (original, operator, ape_seconds) = smear_fixture();
    let vectors = vector_program();
    let started = Instant::now();
    let mut count = 0usize;
    let mut checksum = 0.0;
    for vector in vectors {
        for temporal_extent in 1..=8 {
            let measured = average_cubic_bresenham_mixed_wilson_loop(
                &original,
                &operator,
                vector,
                temporal_extent,
                24,
                21,
            )
            .expect("full bounded Wilson measurement failed");
            checksum += measured.value;
            count += 1;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "phase=wilson_full scope=development_only ape_seconds={ape_seconds:.9} measurement_seconds={elapsed:.9} observable_count={count} checksum={checksum:.17e}"
    );
}

fn print_usage() {
    eprintln!("usage: cargo run -p symthaea-particle-physics --example lqcd_beta6_target_smoke -- <transition|ape|wilson-sample|wilson-full|all>");
    eprintln!("all runs transition + ape + wilson-sample only; wilson-full is always explicit");
}

fn main() {
    let Some(mode) = std::env::args().nth(1) else {
        print_usage();
        return;
    };
    match mode.as_str() {
        "transition" => transition_smoke(),
        "ape" => ape_smoke(),
        "wilson-sample" => wilson_sample_smoke(),
        "wilson-full" => wilson_full_smoke(),
        "all" => {
            transition_smoke();
            ape_smoke();
            wilson_sample_smoke();
        }
        _ => {
            print_usage();
            std::process::exit(2);
        }
    }
}
