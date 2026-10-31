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

use symthaea_iit::{
    enumerate_iit4_system_partitions_with_stats, estimate_iit4_system_partition_work,
};

#[derive(Clone, Copy)]
struct Fixture {
    units: usize,
    raw_direction_assignments: usize,
    canonical_assignments: usize,
    unique_cuts: usize,
}

const FIXTURES: [Fixture; 6] = [
    Fixture { units: 2, raw_direction_assignments: 9, canonical_assignments: 3, unique_cuts: 3 },
    Fixture { units: 3, raw_direction_assignments: 54, canonical_assignments: 22, unique_cuts: 22 },
    Fixture { units: 4, raw_direction_assignments: 306, canonical_assignments: 150, unique_cuts: 150 },
    Fixture { units: 5, raw_direction_assignments: 1_863, canonical_assignments: 1_061, unique_cuts: 1_061 },
    Fixture { units: 6, raw_direction_assignments: 12_348, canonical_assignments: 7_896, unique_cuts: 7_896 },
    Fixture { units: 7, raw_direction_assignments: 88_560, canonical_assignments: 61_888, unique_cuts: 61_888 },
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
        "n_units,repeats,preflight_median_ns,preflight_min_ns,enumeration_median_ns,enumeration_min_ns,raw_direction_assignments_equivalent,canonical_assignments_evaluated,redundant_assignments_pruned,unique_non_total_cuts,validated"
    );

    for fixture in FIXTURES {
        let candidate: Vec<usize> = (0..fixture.units).collect();
        let max_combinations = fixture.canonical_assignments;

        // The estimate is exact and does not construct partition objects.
        let estimate_warmup = estimate_iit4_system_partition_work(
            &candidate,
            fixture.units,
            false,
        ).map_err(|err| format!("preflight estimate failed for n={}: {err}", fixture.units))?;
        if estimate_warmup.raw_direction_labelings_equivalent != fixture.raw_direction_assignments
            || estimate_warmup.canonical_direction_assignments != fixture.canonical_assignments
            || estimate_warmup.redundant_direction_labelings_pruned
                != fixture.raw_direction_assignments - fixture.canonical_assignments
        {
            return Err(format!("preflight estimate mismatch for n={}", fixture.units));
        }

        // Warm-up is not included in the timing samples.
        let warmup = enumerate_iit4_system_partitions_with_stats(
            &candidate,
            fixture.units,
            false,
            max_combinations,
        ).map_err(|err| format!("warm-up failed for n={}: {err}", fixture.units))?;
        if warmup.canonical_direction_assignments_evaluated != fixture.canonical_assignments
            || warmup.raw_direction_assignments_equivalent != fixture.raw_direction_assignments
            || warmup.redundant_direction_assignments_pruned
                != fixture.raw_direction_assignments - fixture.canonical_assignments
            || warmup.unique_non_total_cuts != fixture.unique_cuts
        {
            return Err(format!(
                "count mismatch for n={}: raw={} (expected {}), canonical={} (expected {}), unique={} (expected {})",
                fixture.units,
                warmup.raw_direction_assignments_equivalent,
                fixture.raw_direction_assignments,
                warmup.canonical_direction_assignments_evaluated,
                fixture.canonical_assignments,
                warmup.unique_non_total_cuts,
                fixture.unique_cuts,
            ));
        }

        let mut preflight_samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            let estimate = estimate_iit4_system_partition_work(
                &candidate,
                fixture.units,
                false,
            ).map_err(|err| format!("preflight estimate failed for n={}: {err}", fixture.units))?;
            preflight_samples.push(start.elapsed());
            if estimate.raw_direction_labelings_equivalent != fixture.raw_direction_assignments
                || estimate.canonical_direction_assignments != fixture.canonical_assignments
                || estimate.redundant_direction_labelings_pruned
                    != fixture.raw_direction_assignments - fixture.canonical_assignments
            {
                return Err(format!("preflight estimate changed during timed trial for n={}", fixture.units));
            }
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
            if report.canonical_direction_assignments_evaluated != fixture.canonical_assignments
                || report.raw_direction_assignments_equivalent != fixture.raw_direction_assignments
                || report.redundant_direction_assignments_pruned
                    != fixture.raw_direction_assignments - fixture.canonical_assignments
                || report.unique_non_total_cuts != fixture.unique_cuts
            {
                return Err(format!("partition counts changed during timed trial for n={}", fixture.units));
            }
        }

        let preflight_median_ns = median_duration(&mut preflight_samples).as_nanos();
        let preflight_min_ns = preflight_samples.iter().copied().min().unwrap_or_default().as_nanos();
        let median_ns = median_duration(&mut samples).as_nanos();
        let min_ns = samples.iter().copied().min().unwrap_or_default().as_nanos();
        let validated = estimate_warmup.raw_direction_labelings_equivalent == fixture.raw_direction_assignments
            && estimate_warmup.canonical_direction_assignments == fixture.canonical_assignments
            && warmup.raw_direction_assignments_equivalent == fixture.raw_direction_assignments
            && warmup.canonical_direction_assignments_evaluated == fixture.canonical_assignments
            && warmup.redundant_direction_assignments_pruned
                == fixture.raw_direction_assignments - fixture.canonical_assignments
            && warmup.unique_non_total_cuts == fixture.unique_cuts;

        println!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            fixture.units,
            repeats,
            preflight_median_ns,
            preflight_min_ns,
            median_ns,
            min_ns,
            warmup.raw_direction_assignments_equivalent,
            warmup.canonical_direction_assignments_evaluated,
            warmup.redundant_direction_assignments_pruned,
            warmup.unique_non_total_cuts,
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
