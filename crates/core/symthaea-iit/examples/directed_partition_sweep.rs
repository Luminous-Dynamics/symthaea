// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible microbenchmark for the directed IIT 4.x partition enumerator.
//!
//! Run with:
//! cargo run --release --locked -p symthaea-iit --example directed_partition_sweep
//!
//! This benchmark measures the current Rust enumerator only. It does not compare
//! against PyPhi, and elapsed times are descriptive rather than a qualification gate.

use std::time::{Duration, Instant};

use symthaea_iit::enumerate_iit4_system_partitions_with_stats;

#[derive(Clone, Copy)]
struct Fixture {
    units: usize,
    expected_assignments: usize,
    expected_unique_cuts: usize,
}

const FIXTURES: [Fixture; 4] = [
    Fixture { units: 2, expected_assignments: 9, expected_unique_cuts: 3 },
    Fixture { units: 3, expected_assignments: 54, expected_unique_cuts: 22 },
    Fixture { units: 4, expected_assignments: 306, expected_unique_cuts: 150 },
    Fixture { units: 5, expected_assignments: 1_863, expected_unique_cuts: 1_061 },
];

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn parse_repeats() -> Result<usize, String> {
    let mut args = std::env::args().skip(1);
    let mut repeats = 7usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--repeats" => {
                let value = args.next().ok_or("--repeats requires a positive integer")?;
                repeats = value.parse::<usize>()
                    .map_err(|_| format!("invalid --repeats value: {value}"))?;
                if repeats == 0 || repeats > 1_000 {
                    return Err("--repeats must be in 1..=1000".to_string());
                }
            }
            "--help" | "-h" => {
                println!("directed_partition_sweep [--repeats N]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(repeats)
}

fn run() -> Result<(), String> {
    let repeats = parse_repeats()?;
    println!(
        "n_units,repeats,assignments_evaluated,unique_non_total_cuts,duplicates_suppressed,median_ns,min_ns,validated"
    );

    for fixture in FIXTURES {
        let candidate: Vec<usize> = (0..fixture.units).collect();
        let max_combinations = fixture.expected_assignments;

        // Warm-up is not included in the timing samples.
        let warmup = enumerate_iit4_system_partitions_with_stats(
            &candidate,
            fixture.units,
            false,
            max_combinations,
        ).map_err(|err| format!("warm-up failed for n={}: {err}", fixture.units))?;
        if warmup.block_direction_assignments_evaluated != fixture.expected_assignments
            || warmup.unique_non_total_cuts != fixture.expected_unique_cuts
        {
            return Err(format!(
                "count mismatch for n={}: assignments={} (expected {}), unique={} (expected {})",
                fixture.units,
                warmup.block_direction_assignments_evaluated,
                fixture.expected_assignments,
                warmup.unique_non_total_cuts,
                fixture.expected_unique_cuts,
            ));
        }

        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            let report = enumerate_iit4_system_partitions_with_stats(
                &candidate,
                fixture.units,
                false,
                max_combinations,
            ).map_err(|err| format!("enumeration failed for n={}: {err}", fixture.units))?;
            samples.push(start.elapsed());
            if report.block_direction_assignments_evaluated != fixture.expected_assignments
                || report.unique_non_total_cuts != fixture.expected_unique_cuts
            {
                return Err(format!("partition counts changed during timed trial for n={}", fixture.units));
            }
        }

        let median_ns = median_duration(&mut samples).as_nanos();
        let min_ns = samples.iter().copied().min().unwrap_or_default().as_nanos();
        let duplicates = warmup.duplicate_cut_matrices_suppressed;
        let validated = warmup.block_direction_assignments_evaluated == fixture.expected_assignments
            && warmup.unique_non_total_cuts == fixture.expected_unique_cuts
            && duplicates == fixture.expected_assignments - fixture.expected_unique_cuts;

        println!(
            "{},{},{},{},{},{},{},{}",
            fixture.units,
            repeats,
            warmup.block_direction_assignments_evaluated,
            warmup.unique_non_total_cuts,
            duplicates,
            median_ns,
            min_ns,
            validated,
        );
        if !validated {
            return Err(format!("statistics invariant failed for n={}", fixture.units));
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("benchmark failed: {error}");
        std::process::exit(1);
    }
}
