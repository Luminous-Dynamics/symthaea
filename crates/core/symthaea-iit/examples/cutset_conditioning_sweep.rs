// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact cutset-conditioning versus ordinary min-fill on square binary grids.
//!
//! Run with:
//! cargo run --release --locked -p symthaea-iit --example cutset_conditioning_sweep
//!
//! The comparison uses identical log factors and checks the complete retained
//! factor before reporting timing. The baseline is allowed a much larger factor
//! table; the cutset path is constrained to eight entries per table and therefore
//! reports a time-for-peak-table-width tradeoff, not a speedup.

use std::time::{Duration, Instant};

use symthaea_iit::{
    min_fill_log_sum_product, min_fill_log_sum_product_with_cutset, BinaryLogFactor,
};

#[derive(Clone, Copy)]
struct Fixture {
    side: usize,
}

const FIXTURES: [Fixture; 4] = [
    Fixture { side: 4 },
    Fixture { side: 5 },
    Fixture { side: 6 },
    Fixture { side: 8 },
];

const PAIR_POTENTIAL: [f64; 4] = [0.83, 0.17, 0.29, 0.71];
const CUTSET_FACTOR_ENTRY_BUDGET: usize = 8;
const BASELINE_FACTOR_ENTRY_BUDGET: usize = 1 << 16;

fn grid_factors(side: usize) -> Result<Vec<BinaryLogFactor>, String> {
    let mut factors = Vec::new();
    for row in 0..side {
        for column in 0..side {
            let here = row * side + column;
            if column + 1 < side {
                factors.push(BinaryLogFactor::new(
                    vec![here, here + 1],
                    PAIR_POTENTIAL.iter().map(|value| value.ln()).collect(),
                ).map_err(|error| error.to_string())?);
            }
            if row + 1 < side {
                factors.push(BinaryLogFactor::new(
                    vec![here, here + side],
                    PAIR_POTENTIAL.iter().map(|value| value.ln()).collect(),
                ).map_err(|error| error.to_string())?);
            }
        }
    }
    Ok(factors)
}

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn parse_repeats() -> Result<usize, String> {
    let mut args = std::env::args().skip(1);
    let mut repeats = 5usize;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--repeats" => {
                let value = args.next().ok_or("--repeats requires a positive integer")?;
                repeats = value.parse().map_err(|_| format!("invalid repeats: {value}"))?;
                if !(1..=25).contains(&repeats) {
                    return Err("--repeats must be in 1..=25".to_string());
                }
            }
            "--help" | "-h" => {
                println!("cutset_conditioning_sweep [--repeats N]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(repeats)
}

fn equivalent(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            if a == b { true } else { (a - b).abs() <= 1e-10 }
        })
}

fn run() -> Result<(), String> {
    let repeats = parse_repeats()?;
    println!(
        "side,variables,factors,cutset_variables,cutset_assignments,baseline_induced_width,baseline_peak_entries,cutset_induced_width,cutset_peak_entries,baseline_median_ns,cutset_median_ns,cutset_over_baseline_time_ratio,exact_output_parity,within_cutset_budget"
    );

    for fixture in FIXTURES {
        let side = fixture.side;
        let variables = side * side;
        let retained = variables - 1;
        let factors = grid_factors(side)?;
        let eliminate = (0..variables)
            .filter(|variable| *variable != retained)
            .collect::<Vec<_>>();
        let cutset = (side..(2 * side)).collect::<Vec<_>>();
        let branch_budget = 1usize << cutset.len();

        let baseline = min_fill_log_sum_product(
            &factors,
            &eliminate,
            BASELINE_FACTOR_ENTRY_BUDGET,
        ).map_err(|error| format!("baseline failed for {side}x{side}: {error}"))?;
        let cutset_report = min_fill_log_sum_product_with_cutset(
            &factors,
            &eliminate,
            &cutset,
            CUTSET_FACTOR_ENTRY_BUDGET,
            branch_budget,
        ).map_err(|error| format!("cutset path failed for {side}x{side}: {error}"))?;

        if baseline.factor.variables() != [retained]
            || cutset_report.factor.variables() != [retained]
            || !equivalent(
                baseline.factor.log_values(),
                cutset_report.factor.log_values(),
            )
        {
            return Err(format!("complete retained-factor parity failed for {side}x{side}"));
        }
        if cutset_report.peak_factor_entries > CUTSET_FACTOR_ENTRY_BUDGET
            || cutset_report.predicted_peak_factor_entries != cutset_report.peak_factor_entries
            || cutset_report.cutset_assignments_evaluated != branch_budget
        {
            return Err(format!("cutset resource invariant failed for {side}x{side}"));
        }

        let mut baseline_samples = Vec::with_capacity(repeats);
        let mut cutset_samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            let baseline_trial = min_fill_log_sum_product(
                &factors,
                &eliminate,
                BASELINE_FACTOR_ENTRY_BUDGET,
            ).map_err(|error| error.to_string())?;
            baseline_samples.push(start.elapsed());

            let start = Instant::now();
            let cutset_trial = min_fill_log_sum_product_with_cutset(
                &factors,
                &eliminate,
                &cutset,
                CUTSET_FACTOR_ENTRY_BUDGET,
                branch_budget,
            ).map_err(|error| error.to_string())?;
            cutset_samples.push(start.elapsed());

            if !equivalent(
                baseline_trial.factor.log_values(),
                cutset_trial.factor.log_values(),
            ) {
                return Err(format!("timed-trial parity changed for {side}x{side}"));
            }
        }

        let baseline_median = median_duration(&mut baseline_samples).as_nanos();
        let cutset_median = median_duration(&mut cutset_samples).as_nanos();
        let ratio = if baseline_median == 0 {
            0.0
        } else {
            cutset_median as f64 / baseline_median as f64
        };
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{:.4},{},{}",
            side,
            variables,
            factors.len(),
            cutset.len(),
            branch_budget,
            baseline.induced_width,
            baseline.peak_factor_entries,
            cutset_report.induced_width,
            cutset_report.peak_factor_entries,
            baseline_median,
            cutset_median,
            ratio,
            true,
            cutset_report.peak_factor_entries <= CUTSET_FACTOR_ENTRY_BUDGET,
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cutset conditioning sweep failed: {error}");
        std::process::exit(1);
    }
}
