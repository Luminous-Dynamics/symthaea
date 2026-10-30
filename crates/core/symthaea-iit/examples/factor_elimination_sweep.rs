// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Topology sweep for the exact binary min-fill factor-elimination kernel.
//!
//! Run with:
//! cargo run --release --locked -p symthaea-iit --example factor_elimination_sweep
//!
//! This measures the standalone sum-product primitive. It is not yet an IIT
//! end-to-end benchmark and must not be compared with PyPhi's SIA timing.

use std::time::{Duration, Instant};

use symthaea_iit::{min_fill_sum_product, BinaryFactor};

#[derive(Clone, Copy)]
struct Fixture {
    topology: &'static str,
    variables: usize,
    width: usize,
    height: usize,
}

const FIXTURES: [Fixture; 18] = [
    Fixture { topology: "chain", variables: 32, width: 0, height: 0 },
    Fixture { topology: "chain", variables: 64, width: 0, height: 0 },
    Fixture { topology: "chain", variables: 128, width: 0, height: 0 },
    Fixture { topology: "chain", variables: 256, width: 0, height: 0 },
    Fixture { topology: "chain", variables: 512, width: 0, height: 0 },
    Fixture { topology: "chain", variables: 1_024, width: 0, height: 0 },
    Fixture { topology: "chain", variables: 2_048, width: 0, height: 0 },
    Fixture { topology: "ring", variables: 32, width: 0, height: 0 },
    Fixture { topology: "ring", variables: 64, width: 0, height: 0 },
    Fixture { topology: "ring", variables: 128, width: 0, height: 0 },
    Fixture { topology: "ring", variables: 256, width: 0, height: 0 },
    Fixture { topology: "ring", variables: 512, width: 0, height: 0 },
    Fixture { topology: "grid", variables: 16, width: 4, height: 4 },
    Fixture { topology: "grid", variables: 36, width: 6, height: 6 },
    Fixture { topology: "grid", variables: 64, width: 8, height: 8 },
    Fixture { topology: "grid", variables: 100, width: 10, height: 10 },
    Fixture { topology: "grid", variables: 144, width: 12, height: 12 },
    Fixture { topology: "grid", variables: 196, width: 14, height: 14 },
];

const PAIR_FACTOR: [f64; 4] = [0.83, 0.17, 0.29, 0.71];
const MAX_FACTOR_ENTRIES: usize = 1 << 16;
const REPEATS: usize = 9;

fn add_edge(factors: &mut Vec<BinaryFactor>, left: usize, right: usize) -> Result<(), String> {
    let scope = if left < right {
        vec![left, right]
    } else {
        vec![right, left]
    };
    let factor = BinaryFactor::new(scope, PAIR_FACTOR.to_vec())
        .map_err(|error| error.to_string())?;
    factors.push(factor);
    Ok(())
}

fn build_factors(fixture: Fixture) -> Result<Vec<BinaryFactor>, String> {
    let mut factors = Vec::new();
    match fixture.topology {
        "chain" => {
            for variable in 0..fixture.variables.saturating_sub(1) {
                add_edge(&mut factors, variable, variable + 1)?;
            }
        }
        "ring" => {
            if fixture.variables < 3 {
                return Err("ring fixtures require at least three variables".to_string());
            }
            for variable in 0..fixture.variables {
                add_edge(
                    &mut factors,
                    variable,
                    (variable + 1) % fixture.variables,
                )?;
            }
        }
        "grid" => {
            if fixture.width == 0
                || fixture.height == 0
                || fixture.width * fixture.height != fixture.variables
            {
                return Err("grid dimensions do not match the fixture variable count".to_string());
            }
            for row in 0..fixture.height {
                for column in 0..fixture.width {
                    let here = row * fixture.width + column;
                    if column + 1 < fixture.width {
                        add_edge(&mut factors, here, here + 1)?;
                    }
                    if row + 1 < fixture.height {
                        add_edge(&mut factors, here, here + fixture.width)?;
                    }
                }
            }
        }
        other => return Err(format!("unknown topology: {other}")),
    }
    Ok(factors)
}

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn run() -> Result<(), String> {
    println!(
        "topology,variables,factors,eliminated_variables,induced_width,peak_factor_entries,multiplications,median_ns,min_ns,validated"
    );
    for fixture in FIXTURES {
        let factors = build_factors(fixture)?;
        let eliminate = (0..fixture.variables - 1).collect::<Vec<_>>();
        let warmup = min_fill_sum_product(&factors, &eliminate, MAX_FACTOR_ENTRIES)
            .map_err(|error| format!("warm-up failed for {}-{}: {error}", fixture.topology, fixture.variables))?;
        if warmup.factor.variables() != [fixture.variables - 1]
            || warmup.factor.values().len() != 2
            || warmup.factor.values().iter().any(|value| !value.is_finite())
            || warmup.variables_summed_out != fixture.variables - 1
        {
            return Err(format!(
                "result-shape or finiteness invariant failed for {}-{}",
                fixture.topology, fixture.variables
            ));
        }

        let mut samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let start = Instant::now();
            let report = min_fill_sum_product(&factors, &eliminate, MAX_FACTOR_ENTRIES)
                .map_err(|error| format!("elimination failed: {error}"))?;
            samples.push(start.elapsed());
            if report.factor.variables() != [fixture.variables - 1]
                || report.factor.values().len() != 2
                || report.induced_width != warmup.induced_width
                || report.peak_factor_entries != warmup.peak_factor_entries
            {
                return Err(format!(
                    "deterministic-work invariant failed for {}-{}",
                    fixture.topology, fixture.variables
                ));
            }
        }
        let median_ns = median_duration(&mut samples).as_nanos();
        let min_ns = samples.iter().copied().min().unwrap_or_default().as_nanos();
        let validated = warmup.factor.values().iter().all(|value| value.is_finite());
        println!(
            "{},{},{},{},{},{},{},{},{},{}",
            fixture.topology,
            fixture.variables,
            factors.len(),
            eliminate.len(),
            warmup.induced_width,
            warmup.peak_factor_entries,
            warmup.factor_multiplications,
            median_ns,
            min_ns,
            validated,
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("factor-elimination benchmark failed: {error}");
        std::process::exit(1);
    }
}
